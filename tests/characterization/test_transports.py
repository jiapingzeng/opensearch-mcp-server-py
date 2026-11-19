# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: the wire behaviour of each transport (stdio, SSE, streamable HTTP).

A client sees the server only through its transports. These tests pin, under
``golden/transports/``, what each transport sends back, so that a new server object behind
the same transports can be shown to look the same to every client:

- ``stdio/...``: the real entry point (``main()``) in a subprocess (``stdio_launcher``),
  spoken to with raw JSON-RPC lines: the handshake, ``tools/list``, ``tools/call``, what
  stdout carries, the exit when stdin closes, start-up with the OpenAPI fetch failing.
- ``http/...``: the Starlette app ``streaming_server.serve`` runs (``http_app``): the route
  table, ``/health``, the streamable-HTTP endpoint (``/mcp``, stateless, as ``serve`` builds
  it), the SSE endpoint and its messages path, unknown paths, the lifespan log lines and the
  DNS-rebinding surface (``Host``/``Origin``).
- ``request_context/...``: what reaches the request context: header auth (the fake
  OpenSearch's request log, auth kind only) and the client name
  (``x-mcp-client-name``, ``ClientNameMiddleware``) in the ``tool_execution`` log record,
  over streamable HTTP, SSE and stdio.
"""

import base64
import contextlib
import json
import logging
import pytest
from dataclasses import dataclass
from typing import Any


AREA = 'transports'

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
# Turns on header auth (client.py's `_get_auth_from_headers`).
HEADER_AUTH = {'OPENSEARCH_HEADER_AUTH': 'true'}

EXECUTOR_LOGGER = 'mcp_server_opensearch.tool_executor'
CLIENT_NAME_HEADER = 'x-mcp-client-name'

# The handshake version and client the harness's own helpers offer (conftest.py).
PROTOCOL_VERSION = '2025-11-25'
CLIENT_INFO = {'name': 'characterization-harness', 'version': '0.0.0'}

CAT_INDICES = [
    {
        'health': 'green',
        'status': 'open',
        'index': 'logs-2020.01.01',
        'uuid': 'fake-uuid-logs-0000001',
        'pri': '1',
        'rep': '1',
        'docs.count': '1200',
        'docs.deleted': '0',
        'store.size': '1.2mb',
        'pri.store.size': '620.4kb',
    },
]

LIST_INDEX_CALL = {'name': 'ListIndexTool', 'arguments': {'include_detail': False}}
UNKNOWN_TOOL_CALL = {'name': 'NoSuchTool', 'arguments': {}}

# The fields of the `tool_execution` record (tool_executor.py) a golden keeps. The JSON log
# line has more (`timestamp`, and on some Python versions `taskName`); these are the event.
EVENT_FIELDS = (
    'event_type',
    'tool_name',
    'tool_key',
    'status',
    'error_type',
    'duration_ms',
    'client_name',
)


def basic(username: str, password: str) -> str:
    token = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('ascii')
    return f'Basic {token}'


def basic_username(authorization: str) -> str:
    """The username of a ``Basic`` ``authorization`` header value."""
    scheme, _, token = authorization.partition(' ')
    assert scheme == 'Basic', authorization
    return base64.b64decode(token).decode('utf-8').partition(':')[0]


def rpc(request_id: int, method: str, params: dict | None = None) -> dict:
    message: dict[str, Any] = {'jsonrpc': '2.0', 'id': request_id, 'method': method}
    if params is not None:
        message['params'] = params
    return message


def notification(method: str) -> dict:
    return {'jsonrpc': '2.0', 'method': method}


def initialize_request(request_id: int = 1) -> dict:
    params = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {}, 'clientInfo': CLIENT_INFO}
    return rpc(request_id, 'initialize', params)


def tool_execution_records(caplog) -> list[dict]:
    """The ``tool_execution`` records the test process logged (see test_call_tool_core.py)."""
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
        }
        for record in caplog.records
        if getattr(record, 'event_type', None) == 'tool_execution'
    ]


def json_log_tool_records(stderr: str) -> list[dict]:
    """The ``tool_execution`` events of a ``--log-format json`` server's stderr."""
    records = []
    for line in stderr.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get('event_type') == 'tool_execution':
            records.append(
                {
                    'logger': entry.get('logger'),
                    'level': entry.get('level'),
                    'message': entry.get('message'),
                    **{name: entry[name] for name in EVENT_FIELDS if name in entry},
                }
            )
    return records


def stdout_shape(lines: list[str]) -> list[Any]:
    """Each stdout line as what it carries: a JSON-RPC message's kind and id, or the text."""
    shape: list[Any] = []
    for line in lines:
        try:
            message = json.loads(line)
        except ValueError:
            shape.append({'text': line})
            continue
        if isinstance(message, dict) and message.get('jsonrpc') == '2.0':
            kind = next((k for k in ('result', 'error', 'method') if k in message), None)
            shape.append({'jsonrpc': kind, 'id': message.get('id')})
        else:
            shape.append({'json': message})
    return shape


# ---------------------------------------------------------------------------
# an SSE session over an open stream (`HttpApp.open_stream`, conftest.py)
# ---------------------------------------------------------------------------


@dataclass
class SseSession:
    """An open ``GET /sse`` and the messages path its endpoint event named."""

    http: Any
    stream: Any  # the `OpenStream` of the GET /sse (conftest.py)
    endpoint: dict
    post_headers: dict

    @property
    def messages_path(self) -> str:
        return self.endpoint['data']

    async def post(self, message: Any, headers: dict | None = None) -> Any:
        """POST ``message`` to the messages path; return the HTTP reply."""
        return await self.http.request(
            'POST',
            self.messages_path,
            headers={'content-type': 'application/json', **self.post_headers, **(headers or {})},
            content=json.dumps(message),
        )

    async def exchange(self, message: dict, headers: dict | None = None) -> dict:
        """POST a request; return its HTTP reply and the event that answers it on the stream."""
        reply = await self.post(message, headers)
        events = []
        while True:
            event = await self.stream.next_event()
            if event is None:
                break
            data = json.loads(event['data']) if event.get('event') == 'message' else None
            events.append({**event, 'data': data if data is not None else event.get('data')})
            if isinstance(data, dict) and data.get('id') == message.get('id'):
                break
        return {'post': reply.summary(), 'events': events}

    async def handshake(self, headers: dict | None = None) -> dict:
        initialized = await self.exchange(initialize_request(0), headers)
        notified = await self.post(notification('notifications/initialized'), headers)
        return {'initialize': initialized, 'notifications/initialized': notified.summary()}


@contextlib.asynccontextmanager
async def sse_session(http, get_headers=None, post_headers=None):
    async with http.open_stream('GET', '/sse', get_headers) as stream:
        endpoint = await stream.next_event()
        assert endpoint is not None and endpoint.get('event') == 'endpoint', stream.body
        yield SseSession(http, stream, endpoint, post_headers or {})


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------


async def stdio_exchange(server, messages: list[dict]) -> list[dict]:
    """Write each message as one raw line; after a request, read until its response."""
    received = []
    for message in messages:
        await server.send_line(json.dumps(message))
        if 'id' not in message:
            continue
        while True:
            reply = await server.receive()
            received.append(reply)
            if (
                isinstance(reply, dict)
                and reply.get('id') == message['id']
                and ('result' in reply or 'error' in reply)
            ):
                break
    return received


def stdio_session_messages() -> list[dict]:
    return [
        initialize_request(1),
        notification('notifications/initialized'),
        rpc(2, 'tools/list'),
        rpc(3, 'tools/call', LIST_INDEX_CALL),
        rpc(4, 'tools/call', UNKNOWN_TOOL_CALL),
    ]


# `--transport stdio` and no flag at all (stdio is the default) must behave the same.
STDIO_ARGVS = [
    pytest.param((), id='default'),
    pytest.param(('--transport', 'stdio'), id='transport-stdio'),
]


@pytest.mark.parametrize('argv', STDIO_ARGVS)
async def test_stdio_raw_jsonrpc_session(stdio_launcher, fake_opensearch, golden, normalize, argv):
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    async with stdio_launcher(*argv, env=NO_AUTH) as server:
        responses = await stdio_exchange(server, stdio_session_messages())
        lines_before_close = list(server.stdout_lines)

    golden(f'{AREA}/stdio/session', normalize({'responses': responses}))
    # stdout carries nothing but JSON-RPC: one line per response, in order.
    golden(f'{AREA}/stdio/session_stdout', {'stdout': stdout_shape(lines_before_close)})
    assert all(json.loads(line)['jsonrpc'] == '2.0' for line in server.stdout_lines)
    assert server.returncode == 0


@pytest.mark.parametrize('argv', STDIO_ARGVS)
async def test_stdio_exit_when_stdin_closes(stdio_launcher, golden, normalize, argv):
    # After a handshake, and before any input at all: closing stdin ends the server.
    async with stdio_launcher(*argv) as after_handshake:
        await stdio_exchange(after_handshake, stdio_session_messages()[:2])
        lines_before_close = len(after_handshake.stdout_lines)
    async with stdio_launcher(*argv) as without_input:
        pass

    result = {
        name: {
            'returncode': server.returncode,
            'killed': server.killed,
            'stdout_after_close': stdout_shape(server.stdout_lines[before:]),
            'traceback_on_stderr': 'Traceback (most recent call last)' in server.stderr,
        }
        for name, server, before in (
            ('after_handshake', after_handshake, lines_before_close),
            ('without_input', without_input, 0),
        )
    }
    golden(f'{AREA}/stdio/exit_on_stdin_close', normalize(result))


# tool_generator.py logs the failure (it does not print it), so stdout carries only JSON-RPC.
# Both buffering modes are set explicitly (a PYTHONUNBUFFERED inherited from the environment,
# as some CI runners set, would otherwise pick one): a regression that writes to stdout again
# shows up in the golden whichever way stdout is buffered.
OPENAPI_FAILING_BUFFERING = [
    # A client starting the server on pipes: stdout is block-buffered and the stdio transport
    # flushes its own writes; a stray write would be flushed when the process exits.
    pytest.param(None, id='buffered'),
    # PYTHONUNBUFFERED=1 (as `python -u`): a stray write would land at once, before any JSON-RPC.
    pytest.param('1', id='unbuffered'),
]


@pytest.mark.parametrize('unbuffered', OPENAPI_FAILING_BUFFERING)
async def test_stdio_startup_with_openapi_fetch_failing(
    stdio_launcher, openapi_unavailable, golden, normalize, unbuffered
):
    async with stdio_launcher(env={'PYTHONUNBUFFERED': unbuffered}) as server:
        responses = await stdio_exchange(
            server,
            [initialize_request(1), notification('notifications/initialized')]
            + [rpc(2, 'tools/list')],
        )

    tools = [tool['name'] for tool in responses[-1]['result']['tools']]
    # tool_generator.py logs the failure rather than printing it, so stdout (the stdio
    # transport's channel) holds only the JSON-RPC responses, and the server still starts
    # with the static tools.
    mode = 'unbuffered' if unbuffered else 'buffered'
    golden(
        f'{AREA}/stdio/openapi_fetch_failing_{mode}',
        normalize(
            {
                'stdout': stdout_shape(server.stdout_lines),
                'initialize': responses[0],
                'tool_names': tools,
                'returncode': server.returncode,
            }
        ),
    )


async def test_stdio_tools_call_client_name_is_unknown(
    stdio_launcher, fake_opensearch, golden, normalize
):
    # No HTTP request over stdio, so no x-mcp-client-name: the record says "unknown".
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    async with stdio_launcher('--log-format', 'json', env=NO_AUTH) as server:
        result = await server.call_tool(LIST_INDEX_CALL['name'], LIST_INDEX_CALL['arguments'])

    assert result['isError'] is False
    golden(
        f'{AREA}/request_context/stdio_client_name',
        normalize({'records': json_log_tool_records(server.stderr)}),
    )


# ---------------------------------------------------------------------------
# streamable HTTP and SSE: the app `streaming_server.serve` runs
# ---------------------------------------------------------------------------


def sorted_allow(summary: dict) -> dict:
    """``summary`` with its ``allow`` header's methods sorted.

    Starlette's 405 for a ``Route`` joins the route's method set, so its order follows the
    process's hash seed (``HEAD, GET`` on one run, ``GET, HEAD`` on the next).
    """
    headers = dict(summary['headers'])
    if 'allow' in headers:
        headers['allow'] = ', '.join(sorted(m.strip() for m in headers['allow'].split(',')))
    return {**summary, 'headers': headers}


def route_table(app) -> list[dict]:
    table = []
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        table.append(
            {
                'path': route.path,
                'methods': sorted(methods) if methods is not None else None,
            }
        )
    return table


async def test_http_route_table(http_app, golden):
    async with http_app() as http:
        table = route_table(http.app)

    # Methods None: the route takes every method (the endpoint is a raw ASGI app or a Mount).
    # `/mcp` appears twice: a Route for `/mcp` itself, then a Mount for `/mcp/...`.
    golden(f'{AREA}/http/route_table', {'routes': table})


async def test_http_health(http_app, golden, normalize):
    async with http_app() as http:
        get = await http.get('/health')
        post = await http.request('POST', '/health')

    summaries = {'GET': get.summary(), 'POST': sorted_allow(post.summary())}
    golden(f'{AREA}/http/health', normalize(summaries))


# The last line `create_mcp_server` logs; the app's lifespan starts after it.
SERVER_BUILT_PREFIX = 'Enabled tools: '


async def test_http_lifespan_log_lines(http_app, caplog, golden, normalize):
    caplog.set_level(logging.INFO)
    async with http_app():
        started = len(caplog.records)
    records = [
        {'logger': r.name, 'level': r.levelname, 'message': r.getMessage()} for r in caplog.records
    ]
    # http_app builds the server (create_mcp_server logs its own lines) before the lifespan
    # runs: keep only what follows the server's last line.
    built = max(
        i for i, r in enumerate(records[:started]) if r['message'].startswith(SERVER_BUILT_PREFIX)
    )

    golden(
        f'{AREA}/http/lifespan_log',
        normalize({'startup': records[built + 1 : started], 'shutdown': records[started:]}),
    )


def streamable_exchange(reply) -> dict:
    """An HTTP reply to a POST on /mcp, with what a client reads from its headers."""
    return {
        **reply.summary(),
        'content_type': reply.content_type,
        'has_mcp_session_id': 'mcp-session-id' in reply.headers,
    }


async def test_http_streamable_session(http_app, fake_opensearch, golden, normalize):
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    exchanges = {}
    async with http_app(env=NO_AUTH) as http:
        exchanges['initialize'] = await http.post_mcp(initialize_request(1))
        exchanges['notifications/initialized'] = await http.post_mcp(
            notification('notifications/initialized')
        )
        exchanges['tools/list'] = await http.post_mcp(rpc(2, 'tools/list'))
        exchanges['tools/call'] = await http.post_mcp(rpc(3, 'tools/call', LIST_INDEX_CALL))
        exchanges['tools/call unknown tool'] = await http.post_mcp(
            rpc(4, 'tools/call', UNKNOWN_TOOL_CALL)
        )

    for name, reply in exchanges.items():
        golden(
            f'{AREA}/http/streamable/{name.replace("/", "_").replace(" ", "_")}',
            normalize(streamable_exchange(reply)),
        )
    # Stateless (the only mode `serve` uses): no session id on any reply.
    assert not any('mcp-session-id' in reply.headers for reply in exchanges.values())


async def test_http_streamable_paths_without_redirect(http_app, golden, normalize):
    # /mcp is a Route and /mcp/... a Mount: both answer directly, neither redirects.
    async with http_app() as http:
        replies = {
            path: await http.post_mcp(rpc(1, 'tools/list'), path=path)
            for path in ('/mcp', '/mcp/', '/mcp/sub/path')
        }

    summaries = {
        path: {
            'status': reply.status,
            'location': reply.headers.get('location'),
            'content_type': reply.content_type,
            'tool_count': len(reply.message()['result']['tools']),
        }
        for path, reply in replies.items()
    }
    golden(f'{AREA}/http/streamable/paths', normalize(summaries))
    assert replies['/mcp'].message() == replies['/mcp/'].message()


async def test_http_streamable_get_delete_and_other_methods(http_app, golden, normalize):
    async with http_app() as http:
        # Stateless GET opens a server-to-client event stream that sends nothing and stays
        # open until the client goes: record how it starts and everything it sends until it
        # falls quiet (`drain`), then disconnect.
        async with http.open_stream('GET', '/mcp') as get_default_accept:
            await get_default_accept.drain()
            assert not get_default_accept.ended
        async with http.open_stream(
            'GET', '/mcp', {'accept': 'text/event-stream'}
        ) as get_sse_accept:
            await get_sse_accept.drain()
            assert not get_sse_accept.ended
        get_json_accept = await http.get('/mcp', headers={'accept': 'application/json'})
        delete = await http.request('DELETE', '/mcp')
        put = await http.request('PUT', '/mcp', content=b'{}')

    result = {
        'GET accept */*': {
            **get_default_accept.summary(),
            'body': get_default_accept.body.decode(),
        },
        'GET accept text/event-stream': {
            **get_sse_accept.summary(),
            'body': get_sse_accept.body.decode(),
        },
        'GET accept application/json': get_json_accept.summary(),
        'DELETE': delete.summary(),
        'PUT': put.summary(),
    }
    golden(f'{AREA}/http/streamable/get_delete_put', normalize(result))


async def test_open_stream_drain_reads_what_a_stream_sends(http_app):
    # The empty GET /mcp bodies above mean nothing was sent: drain does collect what an open
    # stream sends at once, here GET /sse's endpoint event, without reading it event by event.
    # The quiet window starts once the endpoint event is in (`min_events`, within drain's
    # 10 s overall timeout): a slow runner may take longer than the window to send it.
    async with http_app() as http:
        async with http.open_stream('GET', '/sse') as stream:
            body = await stream.drain(min_events=1)
            assert not stream.ended

    assert [event['event'] for event in stream.events()] == ['endpoint'], body


async def test_http_streamable_without_accept(http_app, golden, normalize):
    async with http_app() as http:
        initialize = await http.post_mcp(initialize_request(1), {'accept': None})
        tools_list = await http.post_mcp(rpc(2, 'tools/list'), {'accept': None})

    golden(
        f'{AREA}/http/streamable/without_accept',
        normalize({'initialize': initialize.summary(), 'tools/list': tools_list.summary()}),
    )


async def test_http_sse_endpoint_and_messages(http_app, golden, normalize):
    async with http_app() as http:
        async with sse_session(http) as session:
            opened = {**session.stream.summary(), 'endpoint_event': session.endpoint}
            handshake = await session.handshake()
            tools_list = await session.exchange(rpc(1, 'tools/list'))
            messages_path = session.messages_path
        # The stream is gone: its messages path no longer finds the session.
        after_close = await http.request(
            'POST',
            messages_path,
            headers={'content-type': 'application/json'},
            content=json.dumps(rpc(2, 'tools/list')),
        )

    tools_list['events'] = [
        {**event, 'data': {**event['data'], 'result': '<tools/list result>'}}
        for event in tools_list['events']
    ]
    golden(
        f'{AREA}/http/sse/session',
        normalize(
            {
                'GET /sse': opened,
                **handshake,
                'tools/list': tools_list,
                'POST after the stream closed': after_close.summary(),
            }
        ),
    )


async def test_http_sse_tools_list_matches_streamable(http_app):
    async with http_app() as http:
        async with sse_session(http) as session:
            await session.handshake()
            over_sse = await session.exchange(rpc(1, 'tools/list'))
        over_streamable = await http.mcp('tools/list')

    assert over_sse['events'][-1]['data']['result'] == over_streamable['result']


async def test_http_messages_path_other_requests(http_app, golden, normalize):
    # The messages Mount is a raw ASGI app: any method reaches handle_post_message.
    async with http_app() as http:
        result = {
            'GET /messages/': (await http.get('/messages/')).summary(),
            'GET /messages/ with content-type json': (
                await http.get('/messages/', headers={'content-type': 'application/json'})
            ).summary(),
            'GET /messages/?session_id=<unknown>': (
                await http.get(
                    '/messages/?session_id=00000000000000000000000000000000',
                    headers={'content-type': 'application/json'},
                )
            ).summary(),
            'POST /messages/ without session_id': (
                await http.request(
                    'POST',
                    '/messages/',
                    headers={'content-type': 'application/json'},
                    content=b'{}',
                )
            ).summary(),
            'POST /messages/?session_id=not-hex': (
                await http.request(
                    'POST',
                    '/messages/?session_id=not-hex',
                    headers={'content-type': 'application/json'},
                    content=b'{}',
                )
            ).summary(),
            'POST /messages': (
                await http.request(
                    'POST',
                    '/messages',
                    headers={'content-type': 'application/json'},
                    content=b'{}',
                )
            ).summary(),
        }

    golden(f'{AREA}/http/sse/messages_path', normalize(result))


async def test_http_unknown_path(http_app, golden, normalize):
    async with http_app() as http:
        result = {
            'GET /nope': (await http.get('/nope')).summary(),
            'POST /nope': (await http.request('POST', '/nope', content=b'{}')).summary(),
            'GET /': (await http.get('/')).summary(),
            'POST /sse': sorted_allow(
                (await http.request('POST', '/sse', content=b'{}')).summary()
            ),
        }

    golden(f'{AREA}/http/unknown_path', normalize(result))


# characterization: current behaviour. No DNS-rebinding protection: `MCPStarletteApp`
# builds `SseServerTransport` and `StreamableHTTPSessionManager` without security settings,
# so a foreign Host or Origin is served like any other (the SDK's own `MCPServer` apps
# turn the protection on and would answer 421 / 403).
REBINDING_HEADERS = [
    pytest.param({'host': 'example.com'}, id='host'),
    pytest.param({'origin': 'http://evil.example'}, id='origin'),
]


@pytest.mark.parametrize('headers', REBINDING_HEADERS)
async def test_http_dns_rebinding_surface(http_app, golden, normalize, headers):
    async with http_app() as http:
        initialize = await http.post_mcp(initialize_request(1), headers)
        sse = await http.sse_first_event('/sse', headers)

    name = next(iter(headers))
    golden(
        f'{AREA}/http/dns_rebinding_{name}',
        normalize(
            {
                'headers_sent': headers,
                'POST /mcp initialize': streamable_exchange(initialize),
                'GET /sse': sse,
            }
        ),
    )


# ---------------------------------------------------------------------------
# request context: header auth and client name
# ---------------------------------------------------------------------------


@dataclass
class AuthCase:
    name: str
    headers: dict


# The headers `_get_auth_from_headers` (opensearch/client.py) reads, with OPENSEARCH_HEADER_AUTH.
AUTH_CASES = [
    AuthCase('basic', {'authorization': basic('header-user', 'header-pass')}),
    AuthCase(
        'basic_with_url',
        {
            'authorization': basic('header-user', 'header-pass'),
            'opensearch-url': 'http://header-cluster.invalid:9200',
        },
    ),
    AuthCase('bearer', {'authorization': 'Bearer header-token'}),
    AuthCase(
        'aws',
        {
            'aws-access-key-id': 'AKIDHEADEREXAMPLE',
            'aws-secret-access-key': 'header-secret-key',
            'aws-session-token': 'header-session-token',
            'aws-region': 'eu-west-1',
            'aws-service-name': 'es',
        },
    ),
    AuthCase(
        'aws_serverless',
        {
            'aws-access-key-id': 'AKIDHEADEREXAMPLE',
            'aws-secret-access-key': 'header-secret-key',
            'aws-region': 'eu-west-1',
            'aws-service-name': 'aoss',
        },
    ),
]


def redacted_headers(headers: dict) -> dict:
    """The header names a case sends; values that carry no secret are kept."""
    kept = {'opensearch-url', 'aws-region', 'aws-service-name', CLIENT_NAME_HEADER}
    return {name: value if name in kept else '<redacted>' for name, value in headers.items()}


@pytest.mark.parametrize('case', AUTH_CASES, ids=lambda case: case.name)
async def test_http_tools_call_with_header_auth(
    http_app, fake_opensearch, caplog, golden, normalize, case
):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    headers = {**case.headers, CLIENT_NAME_HEADER: 'char-client'}
    async with http_app(env=HEADER_AUTH) as http:
        at_startup = len(fake_opensearch.requests)
        response = await http.mcp('tools/call', LIST_INDEX_CALL, headers=headers)

    golden(
        f'{AREA}/request_context/header_auth_{case.name}',
        normalize(
            {
                'headers_sent': redacted_headers(headers),
                'response': response,
                'opensearch_requests_at_startup': fake_opensearch.requests[:at_startup],
                'opensearch_requests_for_the_call': fake_opensearch.requests[at_startup:],
                'tool_log': tool_execution_records(caplog),
            }
        ),
    )


CLIENT_NAMES = [
    pytest.param(None, id='absent'),
    pytest.param('char-client.v1_2', id='valid'),
    pytest.param('', id='empty'),
    pytest.param('bad name!', id='invalid'),
    # characterization: current behaviour. The name is cut to 64 characters before it is
    # checked: an over-long valid name is kept truncated (not "unknown"), and an invalid
    # character past the 64th is dropped instead of rejecting the name.
    pytest.param('a' * 65, id='over_long'),
    pytest.param('b' * 64 + ' !', id='over_long_invalid_tail'),
    pytest.param('c' * 63 + ' !', id='invalid_within_64'),
]


@pytest.mark.parametrize('client_name', CLIENT_NAMES)
async def test_http_client_name(
    request, http_app, fake_opensearch, caplog, golden, normalize, client_name
):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    headers = {} if client_name is None else {CLIENT_NAME_HEADER: client_name}
    async with http_app(env=NO_AUTH) as http:
        response = await http.mcp('tools/call', LIST_INDEX_CALL, headers=headers)

    assert response['result']['isError'] is False
    case = request.node.callspec.id
    golden(
        f'{AREA}/request_context/client_name_{case}',
        normalize({'header_sent': headers, 'tool_log': tool_execution_records(caplog)}),
    )


@dataclass
class SseClientNameCase:
    name: str
    get_headers: dict
    post_headers: dict


SSE_CLIENT_NAME_CASES = [
    SseClientNameCase(
        'on_get_and_post',
        {CLIENT_NAME_HEADER: 'sse-client'},
        {CLIENT_NAME_HEADER: 'sse-client'},
    ),
    SseClientNameCase('on_get_only', {CLIENT_NAME_HEADER: 'sse-client'}, {}),
    SseClientNameCase('on_post_only', {}, {CLIENT_NAME_HEADER: 'sse-client'}),
    SseClientNameCase(
        'different',
        {CLIENT_NAME_HEADER: 'get-client'},
        {CLIENT_NAME_HEADER: 'post-client'},
    ),
]


@pytest.mark.parametrize('case', SSE_CLIENT_NAME_CASES, ids=lambda case: case.name)
async def test_sse_client_name(http_app, fake_opensearch, caplog, golden, normalize, case):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    async with http_app(env=NO_AUTH) as http:
        async with sse_session(http, case.get_headers, case.post_headers) as session:
            await session.handshake()
            call = await session.exchange(rpc(1, 'tools/call', LIST_INDEX_CALL))

    assert call['events'][-1]['data']['result']['isError'] is False
    golden(
        f'{AREA}/request_context/sse_client_name_{case.name}',
        normalize(
            {
                'GET /sse headers': case.get_headers,
                'POST headers': case.post_headers,
                'tool_log': tool_execution_records(caplog),
            }
        ),
    )


async def test_sse_tools_call_with_header_auth(http_app, fake_opensearch, golden, normalize):
    # Header auth over SSE: which request's headers (the GET that opened the stream or the
    # POST carrying the call) reach the OpenSearch client. The GET and the POST carry
    # different Basic users; the fake's request log shows the one it received.
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    get_headers = {'authorization': basic('get-user', 'get-pass')}
    post_headers = {'authorization': basic('post-user', 'post-pass')}
    async with http_app(env=HEADER_AUTH) as http:
        at_startup = len(fake_opensearch.requests)
        async with sse_session(http, get_headers, post_headers) as session:
            await session.handshake()
            call = await session.exchange(rpc(1, 'tools/call', LIST_INDEX_CALL))

    golden(
        f'{AREA}/request_context/sse_header_auth',
        normalize(
            {
                # What was sent, read back from the headers themselves.
                'GET /sse auth user': basic_username(get_headers['authorization']),
                'POST auth user': basic_username(post_headers['authorization']),
                'tools/call events': call['events'],
                'opensearch_requests_for_the_call': fake_opensearch.requests[at_startup:],
            }
        ),
    )
