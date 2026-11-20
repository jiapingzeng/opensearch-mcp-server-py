# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""request_context_var and client_name_var through the real app, on every transport.

``OpenSearchMCPServer.call_tool`` puts the transport's request in ``request_context_var``
for the call and resets it after; ``ClientNameMiddleware`` puts ``x-mcp-client-name`` in
``client_name_var``. Each test serves recorder tools (each call records both variables)
through a real entry point, with only the enabled tools replaced:

- streamable HTTP and SSE: ``MCPStarletteApp(...).create_app()`` from
  ``streaming_server.create_mcp_server()``, driven in-process (httpx ``ASGITransport``, and
  the app's ASGI interface for the open ``GET /sse`` stream): no socket;
- stdio: ``stdio_server.serve()``, with the process's stdin and stdout replaced by memory
  streams;
- in-memory: ``mcp.client.Client`` on the server.

Header auth (``OPENSEARCH_HEADER_AUTH=true``) reads the request in ``request_context_var``
(``opensearch.client._get_auth_from_headers``): a tool that opens the OpenSearch client
shows the request's auth headers reaching the client, with ``AsyncOpenSearch`` replaced by
a fake that records what it is built with.
"""

import anyio
import asyncio
import base64
import contextlib
import httpx
import json
import logging
import pytest
from dataclasses import dataclass
from mcp.client import Client
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.message import SessionMessage
from mcp.types import TextContent
from mcp_server_opensearch import global_state, stdio_server, streaming_server
from mcp_server_opensearch.client_context import client_name_var, request_context_var
from mcp_types import JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse
from opensearch import client as opensearch_client
from opensearch.client import get_opensearch_client
from starlette.requests import Request
from tools.tool_params import baseToolArgs
from typing import Any, Callable
from unittest.mock import AsyncMock


TIMEOUT_SECONDS = 10
BASE_URL = 'http://127.0.0.1:9900'
MCP_POST_HEADERS = {
    'accept': 'application/json, text/event-stream',
    'content-type': 'application/json',
}
PROTOCOL_VERSION = '2025-11-25'
INITIALIZE_PARAMS = {
    'protocolVersion': PROTOCOL_VERSION,
    'capabilities': {},
    'clientInfo': {'name': 'unit-test', 'version': '0.0.0'},
}

ENV_URL = 'http://env-cluster.invalid:9200'
HEADER_URL = 'http://header-cluster.invalid:9200'


# --- recorder tools ---


@dataclass
class Seen:
    """What one tool call saw: the request in request_context_var and the client name."""

    request: Any
    client_name: str


def _tool(function: Callable) -> dict:
    return {
        'description': 'A recorder tool',
        'input_schema': {'type': 'object', 'properties': {}},
        'args_model': baseToolArgs,
        'function': function,
    }


def _recorder_tools(seen: list[Seen]) -> dict:
    """``Record`` returns a result, ``Raise`` raises RuntimeError, ``RaiseMCPError`` MCPError."""

    def recorder(error: Callable[[], Exception] | None) -> Callable:
        async def function(args: baseToolArgs) -> list:
            seen.append(Seen(request_context_var.get(), client_name_var.get()))
            if error is not None:
                raise error()
            return [TextContent(type='text', text='recorded')]

        return function

    return {
        'Record': _tool(recorder(None)),
        'Raise': _tool(recorder(lambda: RuntimeError('boom'))),
        'RaiseMCPError': _tool(recorder(lambda: MCPError(-32001, 'mcp boom'))),
    }


def _observe_call_tool(server: Any, after: list) -> None:
    """Record request_context_var just before and just after each OpenSearchMCPServer.call_tool.

    The SDK's tools/call handler calls ``self.call_tool``: this wraps the server's own
    override, in the task that runs it, so ``after`` shows what the override left behind.
    """
    call_tool = server.call_tool

    async def observed(name, arguments, context=None):
        before = request_context_var.get()
        try:
            return await call_tool(name, arguments, context)
        finally:
            after.append((before, request_context_var.get()))

    server.call_tool = observed


def _assert_reset_after_every_call(after: list, calls: int) -> None:
    """Each call_tool left request_context_var as it found it (None)."""
    assert after == [(None, None)] * calls


# --- streamable HTTP and SSE: the app create_app builds ---


@pytest.fixture(autouse=True)
def single_mode(monkeypatch):
    """Single mode, no profile, as a server started with no options; restored after."""
    monkeypatch.setattr(global_state, '_current_mode', 'single')
    monkeypatch.setattr(global_state, '_current_profile', None)


@contextlib.asynccontextmanager
async def _http_app(monkeypatch, enabled_tools: dict, after: list):
    """The server create_mcp_server builds (these tools), its app, and an httpx client on it."""
    monkeypatch.setattr(
        streaming_server, 'prepare_enabled_tools', AsyncMock(return_value=enabled_tools)
    )
    server = await streaming_server.create_mcp_server()
    _observe_call_tool(server, after)
    app = streaming_server.MCPStarletteApp(server, stateless=True).create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            yield app, client


def _sse_data(text: str) -> list[dict]:
    """The JSON data of each event of a text/event-stream body."""
    events = text.replace('\r\n', '\n').split('\n\n')
    return [
        json.loads('\n'.join(line[5:].lstrip() for line in lines if line.startswith('data:')))
        for lines in (event.split('\n') for event in events)
        if any(line.startswith('data:') for line in lines)
    ]


async def _post_mcp(client: httpx.AsyncClient, name: str, headers: dict, request_id=1) -> dict:
    """One tools/call POSTed to '/mcp' (stateless, no initialize); the JSON-RPC answer."""
    message = {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'tools/call',
        'params': {'name': name, 'arguments': {}},
    }
    response = await asyncio.wait_for(
        client.post('/mcp', headers={**MCP_POST_HEADERS, **headers}, content=json.dumps(message)),
        TIMEOUT_SECONDS,
    )
    assert response.status_code == 200, response.text
    answers = [m for m in _sse_data(response.text) if m.get('id') == request_id]
    assert len(answers) == 1, response.text
    return answers[0]


async def test_streamable_http_tool_sees_the_starlette_request_with_the_callers_headers(
    monkeypatch,
):
    """Streamable HTTP: request_context_var is the POST /mcp Request, with its headers."""
    seen: list[Seen] = []
    after: list = []
    async with _http_app(monkeypatch, _recorder_tools(seen), after) as (_, client):
        first = await _post_mcp(client, 'Record', {'x-caller': 'first', 'x-mcp-client-name': 'a'})
        second = await _post_mcp(client, 'Record', {'x-caller': 'second'})

    assert first['result'] == {'content': [{'type': 'text', 'text': 'recorded'}], 'isError': False}
    assert second['result'] == first['result']
    assert [type(s.request) for s in seen] == [Request, Request]
    for s in seen:
        assert s.request.method == 'POST'
        assert s.request.url.path == '/mcp'
        assert s.request.headers['content-type'] == 'application/json'
    # Each call sees its own request's headers, and only those.
    assert seen[0].request.headers['x-caller'] == 'first'
    assert seen[1].request.headers['x-caller'] == 'second'
    assert seen[0].request.headers['x-mcp-client-name'] == 'a'
    assert 'x-mcp-client-name' not in seen[1].request.headers
    # x-mcp-client-name reaches client_name_var through ClientNameMiddleware.
    assert [s.client_name for s in seen] == ['a', 'unknown']
    _assert_reset_after_every_call(after, 2)


async def test_streamable_http_request_context_reset_after_a_call_that_raises_or_mcp_error(
    monkeypatch,
):
    """Streamable HTTP: reset after every call, one that raises and one raising MCPError too."""
    seen: list[Seen] = []
    after: list = []
    async with _http_app(monkeypatch, _recorder_tools(seen), after) as (_, client):
        raised = await _post_mcp(client, 'Raise', {'x-caller': 'raise'})
        mcp_error = await _post_mcp(client, 'RaiseMCPError', {'x-caller': 'mcp-error'})
        recorded = await _post_mcp(client, 'Record', {'x-caller': 'record'})

    # Today's answers: an exception is an isError result, an MCPError a JSON-RPC error.
    assert raised['result'] == {'content': [{'type': 'text', 'text': 'boom'}], 'isError': True}
    assert mcp_error['error'] == {'code': -32001, 'message': 'mcp boom'}
    assert recorded['result']['isError'] is False
    assert [s.request.headers['x-caller'] for s in seen] == ['raise', 'mcp-error', 'record']
    _assert_reset_after_every_call(after, 3)


# --- SSE: GET /sse holds the stream, POST /messages/ carries each message ---


class _SseStream:
    """GET /sse sent to the app's ASGI interface; its events read as they arrive.

    ``httpx.ASGITransport`` returns a response only once the app has sent all of it, which
    the open event stream never does: this calls the app itself, with the scope httpx
    would build for the request.
    """

    def __init__(self, app: Any, headers: dict):
        self._app = app
        self._headers = headers
        self._chunks: asyncio.Queue = asyncio.Queue()
        self._buffer = ''
        self._disconnected = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> '_SseStream':
        request = httpx.Request('GET', f'{BASE_URL}/sse', headers=self._headers)
        scope = {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': 'GET',
            'headers': [(name.lower(), value) for name, value in request.headers.raw],
            'scheme': 'http',
            'path': '/sse',
            'raw_path': b'/sse',
            'query_string': b'',
            'server': ('127.0.0.1', 9900),
            'client': ('127.0.0.1', 123),
            'root_path': '',
        }
        request_sent = False

        async def receive() -> dict:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            await self._disconnected.wait()
            return {'type': 'http.disconnect'}

        async def send(message: dict) -> None:
            if message['type'] == 'http.response.body':
                self._chunks.put_nowait(message.get('body', b''))

        self._task = asyncio.create_task(self._app(scope, receive, send))
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._disconnected.set()
        await asyncio.wait_for(self._task, TIMEOUT_SECONDS)

    async def next_event(self) -> dict:
        """The next event with data (comments such as pings skipped): its fields."""
        with anyio.fail_after(TIMEOUT_SECONDS):
            while True:
                if '\n\n' in self._buffer:
                    block, self._buffer = self._buffer.split('\n\n', 1)
                    fields = dict(line.split(':', 1) for line in block.split('\n') if ':' in line)
                    fields = {k: v.lstrip() for k, v in fields.items() if k}
                    if 'data' in fields:
                        return fields
                    continue
                chunk = await self._chunks.get()
                self._buffer = (self._buffer + chunk.decode()).replace('\r\n', '\n')


class _SseSession:
    """An initialized SSE session: messages POSTed to its endpoint, answers read off GET /sse."""

    def __init__(self, stream: _SseStream, client: httpx.AsyncClient, endpoint: str):
        self.stream = stream
        self.client = client
        self.endpoint = endpoint

    async def post(self, message: dict, headers: dict) -> None:
        response = await self.client.post(
            self.endpoint,
            headers={'content-type': 'application/json', **headers},
            content=json.dumps(message),
        )
        assert response.status_code == 202, response.text

    async def request(self, request_id: int, method: str, params: dict, headers: dict) -> dict:
        await self.post(
            {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}, headers
        )
        while True:
            answer = json.loads((await self.stream.next_event())['data'])
            if answer.get('id') == request_id:
                return answer

    async def call_tool(self, request_id: int, name: str, headers: dict) -> dict:
        return await self.request(
            request_id, 'tools/call', {'name': name, 'arguments': {}}, headers
        )


@contextlib.asynccontextmanager
async def _sse_session(app: Any, client: httpx.AsyncClient, get_headers: dict):
    async with _SseStream(app, get_headers) as stream:
        endpoint = await stream.next_event()
        assert endpoint['event'] == 'endpoint'
        session = _SseSession(stream, client, endpoint['data'])
        await session.request(0, 'initialize', INITIALIZE_PARAMS, {})
        await session.post({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, {})
        yield session


async def test_sse_tool_sees_the_post_messages_request(monkeypatch):
    """SSE: request_context_var is the POST /messages/ Request that carried the tools/call.

    The client name is that POST's too, not the GET /sse connection's: the SDK's SSE
    transport hands each message to the server with the context of the request that
    posted it (as before MCPServer, on the same transport).
    """
    seen: list[Seen] = []
    after: list = []
    async with _http_app(monkeypatch, _recorder_tools(seen), after) as (app, client):
        get_headers = {'x-mcp-client-name': 'sse-agent', 'x-caller': 'get-sse'}
        async with _sse_session(app, client, get_headers) as session:
            first = await session.call_tool(1, 'Record', {'x-caller': 'first'})
            second = await session.call_tool(
                2, 'Record', {'x-caller': 'second', 'x-mcp-client-name': 'post-agent'}
            )
            session_id = session.endpoint.split('session_id=', 1)[1]

    assert first['result'] == {'content': [{'type': 'text', 'text': 'recorded'}], 'isError': False}
    assert second['result'] == first['result']
    assert [type(s.request) for s in seen] == [Request, Request]
    for s in seen:
        assert s.request.method == 'POST'
        assert s.request.url.path == '/messages/'
        assert s.request.query_params['session_id'] == session_id
    assert [s.request.headers['x-caller'] for s in seen] == ['first', 'second']
    # x-mcp-client-name reaches client_name_var through ClientNameMiddleware on the POST.
    assert [s.client_name for s in seen] == ['unknown', 'post-agent']
    _assert_reset_after_every_call(after, 2)


async def test_sse_request_context_reset_after_a_call_that_raises_or_mcp_error(monkeypatch):
    """SSE: reset after every call on the one session, one that raises and an MCPError too."""
    seen: list[Seen] = []
    after: list = []
    async with _http_app(monkeypatch, _recorder_tools(seen), after) as (app, client):
        async with _sse_session(app, client, {}) as session:
            raised = await session.call_tool(1, 'Raise', {'x-caller': 'raise'})
            mcp_error = await session.call_tool(2, 'RaiseMCPError', {'x-caller': 'mcp-error'})
            recorded = await session.call_tool(3, 'Record', {'x-caller': 'record'})

    assert raised['result'] == {'content': [{'type': 'text', 'text': 'boom'}], 'isError': True}
    assert mcp_error['error'] == {'code': -32001, 'message': 'mcp boom'}
    assert recorded['result']['isError'] is False
    assert [s.request.headers['x-caller'] for s in seen] == ['raise', 'mcp-error', 'record']
    assert [s.client_name for s in seen] == ['unknown'] * 3
    _assert_reset_after_every_call(after, 3)


# --- stdio and in-memory: no request ---


async def test_stdio_tool_sees_no_request_and_is_reset_after_every_call(monkeypatch):
    """Over stdio (stdio_server.serve): request_context_var is None in every call, reset after."""
    seen: list[Seen] = []
    after: list = []
    enabled_tools = _recorder_tools(seen)
    monkeypatch.setattr(
        stdio_server, 'prepare_enabled_tools', AsyncMock(return_value=enabled_tools)
    )
    build_server = stdio_server.build_server

    def observed_build_server(tools):
        server = build_server(tools)
        _observe_call_tool(server, after)
        return server

    monkeypatch.setattr(stdio_server, 'build_server', observed_build_server)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams

        @contextlib.asynccontextmanager
        async def memory_stdio():
            yield server_streams

        monkeypatch.setattr(stdio_server, 'stdio_server', memory_stdio)
        serving = asyncio.create_task(stdio_server.serve())

        async def send(message) -> None:
            await client_write.send(SessionMessage(message))

        async def answer(request_id: int, method: str, params: dict) -> dict:
            await send(JSONRPCRequest(jsonrpc='2.0', id=request_id, method=method, params=params))
            while True:
                item = await asyncio.wait_for(client_read.receive(), TIMEOUT_SECONDS)
                if isinstance(item, Exception):
                    raise item
                message = item.message
                if isinstance(message, (JSONRPCResponse, JSONRPCError)) and (
                    message.id == request_id
                ):
                    return message.model_dump(mode='json', by_alias=True, exclude_none=True)

        try:
            await answer(0, 'initialize', INITIALIZE_PARAMS)
            await send(JSONRPCNotification(jsonrpc='2.0', method='notifications/initialized'))
            recorded = await answer(1, 'tools/call', {'name': 'Record', 'arguments': {}})
            raised = await answer(2, 'tools/call', {'name': 'Raise', 'arguments': {}})
            mcp_error = await answer(3, 'tools/call', {'name': 'RaiseMCPError', 'arguments': {}})
        finally:
            await client_write.aclose()
            await asyncio.wait_for(serving, TIMEOUT_SECONDS)

    assert recorded['result'] == {
        'content': [{'type': 'text', 'text': 'recorded'}],
        'isError': False,
    }
    assert raised['result'] == {'content': [{'type': 'text', 'text': 'boom'}], 'isError': True}
    assert mcp_error['error'] == {'code': -32001, 'message': 'mcp boom'}
    assert seen == [Seen(None, 'unknown')] * 3
    _assert_reset_after_every_call(after, 3)


async def test_in_memory_tool_sees_no_request_and_is_reset_after_every_call():
    """In-memory (mcp.client.Client): request_context_var is None in every call, reset after."""
    from mcp_server_opensearch.server_factory import build_server

    seen: list[Seen] = []
    after: list = []
    server = build_server(_recorder_tools(seen))
    _observe_call_tool(server, after)

    async with Client(server) as client:
        recorded = await asyncio.wait_for(client.call_tool('Record', {}), TIMEOUT_SECONDS)
        raised = await asyncio.wait_for(client.call_tool('Raise', {}), TIMEOUT_SECONDS)
        with pytest.raises(MCPError) as mcp_error:
            await asyncio.wait_for(client.call_tool('RaiseMCPError', {}), TIMEOUT_SECONDS)

    assert recorded.is_error is False
    assert [c.text for c in recorded.content] == ['recorded']
    assert raised.is_error is True
    assert [c.text for c in raised.content] == ['boom']
    assert mcp_error.value.error.code == -32001
    assert seen == [Seen(None, 'unknown')] * 3
    _assert_reset_after_every_call(after, 3)


# --- header auth: the request's auth headers reach the OpenSearch client ---


class _FakeAsyncOpenSearch:
    """Stands in for opensearchpy.AsyncOpenSearch in opensearch.client: records its kwargs."""

    built: list[dict] = []

    def __init__(self, **kwargs):
        self.built.append(kwargs)

    async def close(self) -> None:
        pass


def _connect_tools(seen: list[Seen]) -> dict:
    """``Connect`` opens the OpenSearch client as every tool does (get_opensearch_client)."""

    async def connect(args: baseToolArgs) -> list:
        seen.append(Seen(request_context_var.get(), client_name_var.get()))
        async with get_opensearch_client(args):
            pass
        return [TextContent(type='text', text='connected')]

    return {'Connect': _tool(connect)}


def _basic(credentials: str) -> str:
    return 'Basic ' + base64.b64encode(credentials.encode()).decode('ascii')


@pytest.fixture
def opensearch_env(monkeypatch):
    """The server's own connection settings: ENV_URL and basic credentials only."""
    for name in (
        'OPENSEARCH_HEADER_AUTH',
        'OPENSEARCH_NO_AUTH',
        'OPENSEARCH_DYNAMIC_CONNECTION',
        'OPENSEARCH_SSRF_GUARD',
        'OPENSEARCH_ALLOW_AMBIENT_AWS_FALLBACK',
        'OPENSEARCH_TIMEOUT',
        'OPENSEARCH_SSL_VERIFY',
        'OPENSEARCH_MAX_RESPONSE_SIZE',
        'OPENSEARCH_CA_CERT_PATH',
        'OPENSEARCH_CLIENT_CERT_PATH',
        'OPENSEARCH_CLIENT_KEY_PATH',
        'AWS_IAM_ARN',
        'AWS_PROFILE',
        'AWS_OPENSEARCH_SERVERLESS',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENSEARCH_URL', ENV_URL)
    monkeypatch.setenv('OPENSEARCH_USERNAME', 'env-user')
    monkeypatch.setenv('OPENSEARCH_PASSWORD', 'env-password')
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    built: list[dict] = []
    monkeypatch.setattr(_FakeAsyncOpenSearch, 'built', built)
    monkeypatch.setattr(opensearch_client, 'AsyncOpenSearch', _FakeAsyncOpenSearch)
    return built


def _signer(kwargs: dict) -> dict:
    signer = kwargs['http_auth'].signer
    credentials = signer.credentials
    return {
        'access_key': credentials.access_key,
        'secret_key': credentials.secret_key,
        'token': credentials.token,
        'region': signer.region,
        'service': signer.service,
    }


@pytest.mark.parametrize(
    'headers, expected',
    [
        pytest.param(
            {'opensearch-url': HEADER_URL, 'authorization': _basic('hdr-user:hdr-password')},
            {'hosts': [HEADER_URL], 'http_auth': ('hdr-user', 'hdr-password')},
            id='basic_and_url',
        ),
        pytest.param(
            {'authorization': 'Bearer hdr-token'},
            {'hosts': [ENV_URL], 'headers': {'Authorization': 'Bearer hdr-token'}},
            id='bearer',
        ),
        pytest.param(
            {
                'aws-access-key-id': 'AKIDHEADER',
                'aws-secret-access-key': 'hdr-secret',
                'aws-session-token': 'hdr-session',
                'aws-region': 'eu-west-1',
                'aws-service-name': 'aoss',
            },
            {
                'hosts': [ENV_URL],
                'signer': {
                    'access_key': 'AKIDHEADER',
                    'secret_key': 'hdr-secret',
                    'token': 'hdr-session',
                    'region': 'eu-west-1',
                    'service': 'aoss',
                },
            },
            id='aws_keys',
        ),
    ],
)
async def test_header_auth_headers_reach_the_opensearch_client(
    monkeypatch, opensearch_env, caplog, headers, expected
):
    """OPENSEARCH_HEADER_AUTH=true over streamable HTTP: the request's auth headers build the client.

    x-mcp-client-name on the same request reaches client_name_var (ClientNameMiddleware),
    so the tool_execution event carries it.
    """
    monkeypatch.setenv('OPENSEARCH_HEADER_AUTH', 'true')
    seen: list[Seen] = []
    after: list = []
    caplog.set_level(logging.INFO, logger='mcp_server_opensearch.tool_executor')
    async with _http_app(monkeypatch, _connect_tools(seen), after) as (_, client):
        answer = await _post_mcp(client, 'Connect', {**headers, 'x-mcp-client-name': 'hdr-agent'})

    assert answer['result'] == {
        'content': [{'type': 'text', 'text': 'connected'}],
        'isError': False,
    }
    assert len(opensearch_env) == 1
    built = opensearch_env[0]
    assert built['hosts'] == expected['hosts']
    if 'http_auth' in expected:
        assert built['http_auth'] == expected['http_auth']
    if 'headers' in expected:
        assert built['headers'] == expected['headers']
    if 'signer' in expected:
        assert _signer(built) == expected['signer']
    assert [s.client_name for s in seen] == ['hdr-agent']
    events = [r for r in caplog.records if getattr(r, 'event_type', None) == 'tool_execution']
    assert [(e.tool_name, e.status, e.client_name) for e in events] == [
        ('Connect', 'success', 'hdr-agent')
    ]
    _assert_reset_after_every_call(after, 1)


async def test_header_auth_off_ignores_the_request_headers(monkeypatch, opensearch_env):
    """Without OPENSEARCH_HEADER_AUTH the auth headers are ignored: the server's own settings."""
    seen: list[Seen] = []
    after: list = []
    headers = {'opensearch-url': HEADER_URL, 'authorization': _basic('hdr-user:hdr-password')}
    async with _http_app(monkeypatch, _connect_tools(seen), after) as (_, client):
        answer = await _post_mcp(client, 'Connect', headers)

    assert answer['result']['isError'] is False
    assert len(opensearch_env) == 1
    assert opensearch_env[0]['hosts'] == [ENV_URL]
    assert opensearch_env[0]['http_auth'] == ('env-user', 'env-password')
    # The request was there all the same: only the setting decides.
    assert isinstance(seen[0].request, Request)


async def test_header_auth_over_stdio_has_no_headers_to_read(monkeypatch, opensearch_env):
    """OPENSEARCH_HEADER_AUTH=true in memory (as on stdio): no request, the server's own settings."""
    from mcp_server_opensearch.server_factory import build_server

    monkeypatch.setenv('OPENSEARCH_HEADER_AUTH', 'true')
    seen: list[Seen] = []
    server = build_server(_connect_tools(seen))

    async with Client(server) as client:
        result = await asyncio.wait_for(client.call_tool('Connect', {}), TIMEOUT_SECONDS)

    assert result.is_error is False
    assert seen == [Seen(None, 'unknown')]
    assert opensearch_env[0]['hosts'] == [ENV_URL]
    assert opensearch_env[0]['http_auth'] == ('env-user', 'env-password')
