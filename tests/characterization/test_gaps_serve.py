# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: the real HTTP ``serve()`` on loopback.

``streaming_server.serve`` builds ``MCPStarletteApp`` and runs it with
``uvicorn.Config(host, port, timeout_graceful_shutdown=10)``. The coverage checkpoint found it
reached only by a unit test that patches uvicorn out, and the planned migration replaces it.
Every other HTTP characterization test drives the Starlette app in-process (``http_app``, no
socket). Here ``serve()`` itself runs, in this process, listening on ``127.0.0.1`` only, on a
port the OS picks (``port=0``), for the life of the test; a real HTTP client talks to it over
loopback. Pinned under ``golden/gaps_serve/``:

- ``streamable_http/session``: ``GET /health``, the ``initialize`` handshake, ``tools/list``
  and one ``tools/call`` over ``POST /mcp``, as the socket carries them (uvicorn's ``date`` and
  ``server`` headers, chunked bodies), and the requests that reached the fake OpenSearch.
- ``sse/session``: the same over ``GET /sse`` and its messages path, and what the client of
  the open ``GET /sse`` sees when the server shuts down under it.
- ``<transport>/server``: what ``serve()`` gave uvicorn (host, port, graceful-shutdown timeout,
  read from the running uvicorn server), where it listened, the uvicorn and lifespan log lines
  from start to stop, the signal re-raised after ``serve()`` returned, and that the port is
  closed afterwards.
- ``graceful_shutdown``: the graceful-shutdown timeout as a client sees it: a request still
  being received when the server stops is cut after 10 s (the test checks the wait), and the
  log lines of the stopping server and of that request (two lists: two tasks log at once).

A run stops as ``kill <pid>`` stops the server: ``SIGTERM`` (``signal.raise_signal``), which
uvicorn's handler turns into a graceful shutdown. ``serve()`` must return within
``STOP_TIMEOUT`` seconds; the run's ``finally`` stops it in any case (SIGTERM, then uvicorn's
``force_exit``, then cancelling the task) and closes any listener left, so no test leaves a
socket open. While ``serve()`` runs, uvicorn's ``Server.handle_exit`` is the process's SIGTERM
handler; its ``__self__`` is the running uvicorn server (sse-starlette finds it the same
way), which is how the test reads the config, the listening sockets and the requests in
flight (``server_state.tasks``): nothing in uvicorn or the server is patched.

What a run changes outside the server and puts back afterwards: the SIGTERM handler (a
recorder while ``serve()`` runs, which also catches the SIGTERM uvicorn raises again once
``serve()`` returns), the log record factory (records are read where they are made: uvicorn
configures its own loggers, which do not propagate to pytest's capture), the ``uvicorn``
loggers' configuration (``uvicorn.Config`` calls ``logging.config.dictConfig``) and
sse-starlette's process-wide ``AppStatus.should_exit``, which uvicorn's SIGTERM sets and
nothing resets (left set, it would end every later test's event streams at once).

The module is skipped where a loopback TCP port cannot be bound, listened on and connected to.
"""

import asyncio
import codecs
import contextlib
import httpx
import json
import logging
import pytest
import re
import signal
import socket
import threading
import time
import uvicorn
from dataclasses import dataclass, field
from mcp_server_opensearch import streaming_server
from sse_starlette import sse as sse_starlette
from typing import Any


AREA = 'gaps_serve'

# The only interface any test here binds to.
LOOPBACK = '127.0.0.1'
# How every test here calls serve(): loopback only, on a port the OS picks.
SERVE_ARGS = {'host': LOOPBACK, 'port': 0}
# `serve()`'s own default port: an OS-picked port must differ from it.
SERVE_DEFAULT_PORT = 9900
# What `serve()` passes uvicorn as `timeout_graceful_shutdown`.
GRACEFUL_SHUTDOWN_SECONDS = 10

# Seconds to wait for serve() to listen, to return after SIGTERM (the graceful-shutdown wait
# included), for one event or reply from the server, and for a connect to a closed port to be
# refused (Windows retries a refused loopback connect for about two seconds).
STARTUP_TIMEOUT = 30
STOP_TIMEOUT = 30
FORCED_STOP_TIMEOUT = 5
READ_TIMEOUT = 15
CONNECT_TIMEOUT = 10
# How long sse-starlette's shutdown watcher may take to notice the shutdown (it polls every
# 0.5 s).
SSE_WATCHER_TIMEOUT = 5
POLL_SECONDS = 0.02

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}

PROTOCOL_VERSION = '2025-11-25'
CLIENT_INFO = {'name': 'characterization-harness', 'version': '0.0.0'}
CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
]
LIST_INDEX_CALL = {'name': 'ListIndexTool', 'arguments': {'include_detail': False}}

# Log records a `server` golden keeps, from uvicorn's first line on: uvicorn's own
# (`uvicorn.error`: start, stop, errors; `uvicorn.access`: one line per request) and the app's
# lifespan lines (streaming_server's `MCPStarletteApp.lifespan` and the session manager it
# runs). Tool, OpenSearch and per-request SDK lines are pinned elsewhere.
UVICORN_LOGGERS = ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi')
LIFESPAN_LOGGERS = ('mcp.server.streamable_http_manager',)
LIFESPAN_MODULE = 'streaming_server'

# Volatile parts of those log lines: loopback addresses with a port (the server's, picked by
# the OS, and each client connection's) and the process id uvicorn prints.
LOOPBACK_ADDRESS = re.compile(r'\b127\.0\.0\.1:\d+\b')
PROCESS_ID = re.compile(r'(process \[)\d+(\])')


# ---------------------------------------------------------------------------
# skip where a loopback port cannot be used
# ---------------------------------------------------------------------------


def _loopback_unavailable() -> str | None:
    """Why a TCP port on 127.0.0.1 cannot be bound, listened on and connected to; else None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((LOOPBACK, 0))
            listener.listen(1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(CONNECT_TIMEOUT)
                client.connect(listener.getsockname())
    except OSError as error:
        return f'{type(error).__name__}: {error}'
    return None


_LOOPBACK_UNAVAILABLE = _loopback_unavailable()
if _LOOPBACK_UNAVAILABLE is not None:
    pytest.skip(
        'serve() needs a listening TCP socket on 127.0.0.1, which this environment does not '
        f'allow ({_LOOPBACK_UNAVAILABLE})',
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# a run of the real serve()
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    """A log record made while ``serve()`` ran: when (monotonic) and in which asyncio task."""

    at: float
    record: logging.LogRecord
    task: asyncio.Task | None


def log_line(record: logging.LogRecord) -> dict:
    """A log record as a golden keeps it: logger, level, message and any exception."""
    message = PROCESS_ID.sub(r'\1<PID>\2', record.getMessage())
    line = {
        'logger': record.name,
        'level': record.levelname,
        'message': LOOPBACK_ADDRESS.sub(f'{LOOPBACK}:<PORT>', message),
    }
    if record.exc_info and record.exc_info[1] is not None:
        error = record.exc_info[1]
        line['exception'] = type(error).__name__
        # A cancelled task's message depends on where the cancellation was caught: keep
        # only the type for CancelledError.
        if not isinstance(error, asyncio.CancelledError):
            line['exception'] += f': {error}'
    return line


def _running_uvicorn_server() -> uvicorn.Server | None:
    """The uvicorn server whose ``handle_exit`` is the SIGTERM handler now, if any."""
    server = getattr(signal.getsignal(signal.SIGTERM), '__self__', None)
    return server if isinstance(server, uvicorn.Server) else None


def _sse_watcher_running() -> bool:
    """Whether sse-starlette's shutdown watcher (one per thread) is still running.

    The state is private to sse-starlette (3.x); without it, take the watcher to be running,
    so the caller waits its whole timeout.
    """
    shutdown_state = getattr(sse_starlette, '_get_shutdown_state', None)
    return True if shutdown_state is None else bool(shutdown_state().watcher_started)


@dataclass
class ServeRun:
    """``streaming_server.serve(**SERVE_ARGS)`` running in a task of this test."""

    task: asyncio.Task
    log: list[LogEntry]
    raised_again: list[str]
    uvicorn_server: uvicorn.Server | None = None
    port: int = 0
    listening: list[dict] = field(default_factory=list)
    # The tasks uvicorn runs the app in for requests, as `wait_requests_in_flight` saw them.
    request_tasks: set = field(default_factory=set)
    stopped: bool = False

    @property
    def base_url(self) -> str:
        return f'http://{LOOPBACK}:{self.port}'

    def client(self) -> httpx.AsyncClient:
        """A real HTTP client for the server (no proxy or other settings from the environment)."""
        return httpx.AsyncClient(base_url=self.base_url, trust_env=False, timeout=READ_TIMEOUT)

    async def wait_started(self) -> None:
        """Wait until uvicorn listens; record the port it got and its listening sockets."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STARTUP_TIMEOUT
        while True:
            if self.task.done():
                pytest.fail(
                    f'serve() ended before listening: {self.task!r}\n{self._log_text()}',
                    pytrace=False,
                )
            server = _running_uvicorn_server()
            if server is not None and server.started:
                break
            if loop.time() > deadline:
                pytest.fail(f'serve() did not listen in {STARTUP_TIMEOUT}s', pytrace=False)
            await asyncio.sleep(POLL_SECONDS)
        self.uvicorn_server = server
        sockets = [sock for listener in server.servers for sock in listener.sockets]
        self.port = sockets[0].getsockname()[1]
        for sock in sockets:
            host, port = sock.getsockname()[:2]
            self.listening.append(
                {
                    'family': socket.AddressFamily(sock.family).name,
                    'host': host,
                    'port': '<PORT>' if port == self.port else port,
                }
            )

    async def wait_requests_in_flight(self, count: int) -> None:
        """Wait until uvicorn runs the app for ``count`` requests (its server state's tasks)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + READ_TIMEOUT
        while len(self.uvicorn_server.server_state.tasks) < count:
            if loop.time() > deadline:
                pytest.fail(f'uvicorn did not start {count} request(s)', pytrace=False)
            await asyncio.sleep(POLL_SECONDS)
        self.request_tasks = set(self.uvicorn_server.server_state.tasks)

    async def stop(self) -> None:
        """Stop ``serve()`` as ``kill <pid>`` does (SIGTERM); wait until it returns."""
        self.stopped = True
        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(asyncio.shield(self.task), STOP_TIMEOUT)

    def lines(self, *, in_request_tasks: bool | None = None) -> list[dict]:
        """The log lines a golden keeps (see ``UVICORN_LOGGERS``), from uvicorn's first on.

        ``in_request_tasks`` True keeps only the lines logged in ``request_tasks``, False
        only the others: two tasks running at once log in an order a golden cannot rely on.
        """
        start = next(
            (i for i, entry in enumerate(self.log) if entry.record.name in UVICORN_LOGGERS),
            len(self.log),
        )
        return [
            log_line(entry.record)
            for entry in self.log[start:]
            if (
                entry.record.name in UVICORN_LOGGERS + LIFESPAN_LOGGERS
                or (entry.record.name == 'root' and entry.record.module == LIFESPAN_MODULE)
            )
            and (
                in_request_tasks is None or (entry.task in self.request_tasks) == in_request_tasks
            )
        ]

    def logged_at(self, logger: str, message_start: str) -> float:
        """When the first ``logger`` record whose message starts with ``message_start`` was made."""
        for entry in self.log:
            if entry.record.name == logger and entry.record.getMessage().startswith(message_start):
                return entry.at
        pytest.fail(f'no {logger} record starting {message_start!r}', pytrace=False)

    def summary(self, port_closed: bool) -> dict:
        """What the ``server`` golden keeps about this run."""
        config = self.uvicorn_server.config
        return {
            'serve_called_with': SERVE_ARGS,
            'uvicorn_config': {
                'host': config.host,
                'port': config.port,
                'timeout_graceful_shutdown': config.timeout_graceful_shutdown,
            },
            'listening': self.listening,
            'log': self.lines(),
            'serve_returned': repr(self.task.result()),
            # characterization: current behaviour: uvicorn raises the SIGTERM it handled again
            # once serve() has returned, so under main()'s asyncio.run(serve(...)) the process
            # then ends by the signal's default action.
            'signals_raised_again_after_serve_returned': self.raised_again,
            'port_closed_after_serve_returned': port_closed,
        }

    def _log_text(self) -> str:
        return '\n'.join(f'{e.record.name}: {e.record.getMessage()}' for e in self.log)


async def _wait_done(task: asyncio.Task, timeout: float) -> bool:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    return bool(done)


async def _shut_down(run: ServeRun) -> None:
    """Make sure ``serve()`` has stopped and nothing listens any more, whatever happened."""
    task = run.task
    server = run.uvicorn_server or _running_uvicorn_server()
    if not task.done():
        if server is not None and not run.stopped:
            run.stopped = True
            signal.raise_signal(signal.SIGTERM)
        if not await _wait_done(task, STOP_TIMEOUT if server is not None else 0) and server:
            server.force_exit = True
            await _wait_done(task, FORCED_STOP_TIMEOUT)
        if not task.done():
            task.cancel()
            await _wait_done(task, FORCED_STOP_TIMEOUT)
    if server is not None:
        for listener in getattr(server, 'servers', []):
            listener.close()
    # sse-starlette's watcher ends once it sees the shutdown; until then it may set
    # AppStatus.should_exit again (from the uvicorn server it found), so wait for it to end
    # before that flag is put back.
    if run.stopped:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SSE_WATCHER_TIMEOUT
        while _sse_watcher_running() and loop.time() < deadline:
            await asyncio.sleep(POLL_SECONDS)


@contextlib.contextmanager
def _uvicorn_logging_kept():
    """Put the ``uvicorn`` loggers back as they were (``uvicorn.Config`` reconfigures them)."""
    loggers = [logging.getLogger(name) for name in UVICORN_LOGGERS]
    saved = [
        (lg, lg.level, lg.propagate, lg.disabled, list(lg.handlers), list(lg.filters))
        for lg in loggers
    ]
    try:
        yield
    finally:
        for lg, level, propagate, disabled, handlers, filters in saved:
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled
            lg.handlers[:] = handlers
            lg.filters[:] = filters


@pytest.fixture
def serving(monkeypatch, caplog):
    """Factory running the real ``serve()`` on loopback for the block it opens.

    Use as ``async with serving(env=None) as run:``. ``serve(host='127.0.0.1', port=0)``
    starts in a task; the block is entered once uvicorn listens (``run.port``,
    ``run.client()``). ``await run.stop()`` stops it inside the test; on leaving the block it
    is stopped in any case (see ``_shut_down``). ``env`` is set for the test.
    """
    if threading.current_thread() is not threading.main_thread():
        pytest.skip('serve() handles SIGTERM only in the main thread')

    @contextlib.asynccontextmanager
    async def start(env: dict | None = None):
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        # Put back at teardown: uvicorn's SIGTERM sets it and nothing resets it.
        monkeypatch.setattr(sse_starlette.AppStatus, 'should_exit', False)
        # The app's lifespan lines are logged at INFO on the root logger.
        caplog.set_level(logging.INFO)
        log: list[LogEntry] = []
        raised_again: list[str] = []
        factory = logging.getLogRecordFactory()

        def recording_factory(*args, **kwargs) -> logging.LogRecord:
            record = factory(*args, **kwargs)
            try:
                task = asyncio.current_task()
            except RuntimeError:  # a record made outside the event loop
                task = None
            log.append(LogEntry(time.monotonic(), record, task))
            return record

        def on_sigterm(signum, frame) -> None:
            raised_again.append(signal.Signals(signum).name)

        with _uvicorn_logging_kept():
            previous_handler = signal.signal(signal.SIGTERM, on_sigterm)
            logging.setLogRecordFactory(recording_factory)
            task = asyncio.create_task(streaming_server.serve(**SERVE_ARGS))
            run = ServeRun(task, log, raised_again)
            try:
                await run.wait_started()
                yield run
            finally:
                try:
                    await _shut_down(run)
                finally:
                    logging.setLogRecordFactory(factory)
                    signal.signal(signal.SIGTERM, previous_handler)

    return start


async def port_is_closed(port: int) -> bool:
    """Whether a connect to ``127.0.0.1:port`` is refused (True) or accepted (False)."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(LOOPBACK, port), CONNECT_TIMEOUT
        )
    except asyncio.TimeoutError:
        pytest.fail(f'connecting to port {port} was neither accepted nor refused', pytrace=False)
    except OSError:
        return True
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return False


# ---------------------------------------------------------------------------
# a real HTTP client
# ---------------------------------------------------------------------------


def reply_headers(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    """Response headers with names lower-cased (repeated ones joined), as ``http_app`` keeps them."""
    headers: dict[str, str] = {}
    for name, value in raw_headers:
        name, value = name.decode('latin-1').lower(), value.decode('latin-1')
        headers[name] = f'{headers[name]}, {value}' if name in headers else value
    return headers


def reply_summary(harness, response: httpx.Response) -> dict:
    """A complete response as ``HttpReply.summary()`` gives it (status, headers, body)."""
    reply = harness.HttpReply(
        response.status_code, reply_headers(response.headers.raw), response.text
    )
    return reply.summary()


def rpc(request_id: int, method: str, params: dict | None = None) -> dict:
    message: dict[str, Any] = {'jsonrpc': '2.0', 'id': request_id, 'method': method}
    if params is not None:
        message['params'] = params
    return message


def notification(method: str) -> dict:
    return {'jsonrpc': '2.0', 'method': method}


def initialize_request(request_id: int) -> dict:
    params = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {}, 'clientInfo': CLIENT_INFO}
    return rpc(request_id, 'initialize', params)


def error_text(error: BaseException) -> str:
    return f'{type(error).__name__}: {error}'


class EventStream:
    """An open ``text/event-stream`` response over the socket, read one event at a time."""

    def __init__(self, response: httpx.Response, parse_sse):
        self.response = response
        self._parse_sse = parse_sse
        self._chunks = response.aiter_bytes()
        self._decoder = codecs.getincrementaldecoder('utf-8')()
        self._text = ''
        self._events_read = 0

    async def next_event(self) -> dict | None:
        """The next event; ``None`` once the response has ended. Fails after ``READ_TIMEOUT``."""
        while True:
            events = self._parse_sse(self._text)
            if len(events) > self._events_read:
                self._events_read += 1
                return events[self._events_read - 1]
            try:
                chunk = await asyncio.wait_for(self._chunks.__anext__(), READ_TIMEOUT)
            except StopAsyncIteration:
                return None
            self._text += self._decoder.decode(chunk)

    async def rest(self) -> dict:
        """Read to the end of the response: the events still to come and how it ended."""
        events = []
        try:
            while (event := await self.next_event()) is not None:
                events.append(event)
        except httpx.HTTPError as error:
            return {'events': events, 'ended_with': error_text(error)}
        return {'events': events, 'ended_with': 'the end of the response'}


def with_json_data(event: dict | None) -> dict | None:
    """An SSE event with its ``data`` parsed when it is JSON."""
    if event is None or 'data' not in event:
        return event
    try:
        return {**event, 'data': json.loads(event['data'])}
    except ValueError:
        return event


def opensearch_requests(fake_opensearch) -> list[str]:
    return [f'{r["method"]} {r["path"]}' for r in fake_opensearch.requests]


# ---------------------------------------------------------------------------
# streamable HTTP
# ---------------------------------------------------------------------------


async def test_streamable_http_over_loopback(serving, harness, fake_opensearch, golden, normalize):
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)

    async def post(client: httpx.AsyncClient, message: dict) -> dict:
        response = await client.post(
            '/mcp', headers=harness.MCP_POST_HEADERS, content=json.dumps(message)
        )
        return reply_summary(harness, response)

    async with serving(env=NO_AUTH) as run:
        async with run.client() as client:
            session = {
                'GET /health': reply_summary(harness, await client.get('/health')),
                'initialize': await post(client, initialize_request(1)),
                'notifications/initialized': await post(
                    client, notification('notifications/initialized')
                ),
                'tools/list': await post(client, rpc(2, 'tools/list')),
                'tools/call': await post(client, rpc(3, 'tools/call', LIST_INDEX_CALL)),
            }
        await run.stop()
    port_closed = await port_is_closed(run.port)

    assert run.port not in (0, SERVE_DEFAULT_PORT)
    assert port_closed, f'127.0.0.1:{run.port} still accepts connections after serve() returned'
    session['opensearch_requests'] = opensearch_requests(fake_opensearch)
    golden(f'{AREA}/streamable_http/session', normalize(session))
    golden(f'{AREA}/streamable_http/server', normalize(run.summary(port_closed)))


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


async def sse_exchange(harness, client, stream: EventStream, path: str, message: dict) -> dict:
    """POST ``message`` to the messages path; for a request, read its answer off the stream."""
    response = await client.post(
        path, headers={'content-type': 'application/json'}, content=json.dumps(message)
    )
    exchange = {'post': reply_summary(harness, response)}
    if 'id' in message:
        exchange['event'] = with_json_data(await stream.next_event())
    return exchange


async def test_sse_over_loopback(serving, harness, fake_opensearch, golden, normalize):
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)

    async with serving(env=NO_AUTH) as run:
        async with run.client() as client:
            async with client.stream('GET', '/sse') as response:
                stream = EventStream(response, harness.parse_sse)
                endpoint = await stream.next_event()
                assert endpoint is not None and endpoint.get('event') == 'endpoint', endpoint
                path = endpoint['data']
                session = {
                    'GET /sse': {
                        'status': response.status_code,
                        'headers': reply_headers(response.headers.raw),
                        'endpoint_event': endpoint,
                    },
                    'initialize': await sse_exchange(
                        harness, client, stream, path, initialize_request(0)
                    ),
                    'notifications/initialized': await sse_exchange(
                        harness, client, stream, path, notification('notifications/initialized')
                    ),
                    'tools/list': await sse_exchange(
                        harness, client, stream, path, rpc(1, 'tools/list')
                    ),
                    'tools/call': await sse_exchange(
                        harness, client, stream, path, rpc(2, 'tools/call', LIST_INDEX_CALL)
                    ),
                }
                # Stop the server with the GET /sse stream still open.
                await run.stop()
                # characterization: current behaviour: on shutdown sse-starlette ends the
                # stream, then handle_sse returns a second response (`Response()`), which
                # uvicorn refuses ("Exception in ASGI application", RuntimeError) and closes
                # the connection: the client's stream ends without its last chunk.
                session['GET /sse after serve() stopped'] = await stream.rest()
    port_closed = await port_is_closed(run.port)

    assert run.port not in (0, SERVE_DEFAULT_PORT)
    assert port_closed, f'127.0.0.1:{run.port} still accepts connections after serve() returned'
    session['opensearch_requests'] = opensearch_requests(fake_opensearch)
    golden(f'{AREA}/sse/session', normalize(session))
    golden(f'{AREA}/sse/server', normalize(run.summary(port_closed)))


# ---------------------------------------------------------------------------
# the graceful-shutdown timeout
# ---------------------------------------------------------------------------

# A request whose body never arrives in full: uvicorn has started the app for it, which waits
# for the rest of the body, so the connection stays busy through the shutdown. (An event
# stream would not: sse-starlette ends those as soon as the shutdown starts.)
STUCK_BODY = json.dumps(rpc(1, 'tools/list')).encode('utf-8')
STUCK_BODY_SENT = 10


def stuck_request(port: int) -> bytes:
    head = (
        'POST /mcp HTTP/1.1\r\n'
        f'host: {LOOPBACK}:{port}\r\n'
        'accept: application/json, text/event-stream\r\n'
        'content-type: application/json\r\n'
        f'content-length: {len(STUCK_BODY)}\r\n'
        '\r\n'
    )
    return head.encode('ascii') + STUCK_BODY[:STUCK_BODY_SENT]


async def test_graceful_shutdown_timeout(serving, golden, normalize):
    async with serving() as run:
        reader, writer = await asyncio.open_connection(LOOPBACK, run.port)
        try:
            writer.write(stuck_request(run.port))
            await writer.drain()
            await run.wait_requests_in_flight(1)
            await run.stop()
            reply = await asyncio.wait_for(reader.read(), READ_TIMEOUT)
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
    port_closed = await port_is_closed(run.port)

    # uvicorn waits `timeout_graceful_shutdown` seconds for the connection, then cancels it.
    waited = run.logged_at('uvicorn.error', 'Cancel ') - run.logged_at(
        'uvicorn.error', 'Waiting for connections to close.'
    )
    assert GRACEFUL_SHUTDOWN_SECONDS - 0.5 <= waited < GRACEFUL_SHUTDOWN_SECONDS + 10, waited
    assert port_closed, f'127.0.0.1:{run.port} still accepts connections after serve() returned'
    summary = run.summary(port_closed)
    assert len(run.request_tasks) == 1, run.request_tasks
    golden(
        f'{AREA}/graceful_shutdown',
        normalize(
            {
                'request': {
                    'sent': f'POST /mcp, content-length {len(STUCK_BODY)}, '
                    f'{STUCK_BODY_SENT} bytes of the body',
                    # The reply the client reads once the wait is over, line by line.
                    'reply': reply.decode('utf-8', 'replace').split('\r\n'),
                },
                'serve_called_with': summary['serve_called_with'],
                'uvicorn_config': summary['uvicorn_config'],
                # The stopping server's lines and the stuck request's are logged by two tasks
                # at once (uvicorn cancels the request, then starts the lifespan shutdown):
                # each list is in order, the two are not ordered with each other.
                'log': run.lines(in_request_tasks=False),
                'request_log': run.lines(in_request_tasks=True),
                'signals_raised_again_after_serve_returned': summary[
                    'signals_raised_again_after_serve_returned'
                ],
                'port_closed_after_serve_returned': port_closed,
            }
        ),
    )
