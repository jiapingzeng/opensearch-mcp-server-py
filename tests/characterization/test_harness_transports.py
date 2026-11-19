# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the transport fixtures: http_app, stdio_launcher, memory_enabled_session."""

import contextlib
import httpx
import json
import pytest
import re
from mcp.types import ListToolsResult
from starlette.applications import Starlette


MEMORY_TOOLS = {'SaveMemoryTool', 'SearchMemoryTool', 'DeleteMemoryTool'}


# --- http_app ----------------------------------------------------------------


async def test_http_app_health(http_app, golden, normalize):
    async with http_app() as http:
        reply = await http.get('/health')

    golden('harness_transports/http_health', normalize(reply.summary()))


async def test_http_app_tools_list_over_mcp(http_app, golden, normalize):
    async with http_app() as http:
        reply = await http.post_mcp({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})

    golden('harness_transports/http_mcp_tools_list', normalize(reply.summary()))
    assert reply.content_type == 'text/event-stream'
    assert reply.message()['id'] == 1


async def test_http_app_mcp_helper_returns_the_reply_message(
    harness, http_app, mcp_session, normalize
):
    async with http_app() as http:
        response = await http.mcp('tools/list')

    # The same server as in-process, reached over HTTP.
    over_http = harness.dump(ListToolsResult.model_validate(response['result']))
    async with mcp_session() as session:
        assert normalize(over_http) == normalize(await session.list_tools())


async def test_http_app_sse_first_event(http_app, golden, normalize):
    async with http_app() as http:
        opened = await http.sse_first_event('/sse')

    golden('harness_transports/http_sse_first_event', normalize(opened))
    assert opened['event']['event'] == 'endpoint'


async def test_http_app_failure_in_the_block_reaches_the_test_as_itself(http_app):
    # Not wrapped in an ExceptionGroup: golden diffs and pytest.fail messages stay readable.
    with pytest.raises(pytest.fail.Exception, match='probe failure'):
        async with http_app():
            pytest.fail('probe failure', pytrace=False)


async def test_asgi_lifespan_startup_failure_names_its_cause(harness):
    @contextlib.asynccontextmanager
    async def failing_lifespan(app):
        raise RuntimeError('probe startup failure')
        yield

    with pytest.raises(AssertionError, match='lifespan startup failed') as excinfo:
        async with harness.asgi_lifespan(Starlette(lifespan=failing_lifespan)):
            pass
    assert repr(excinfo.value.__cause__) == "RuntimeError('probe startup failure')"


async def _echo_app(scope, receive, send):
    """Answer any request with what reached the app: method, path, headers, JSON body."""
    body = b''
    while True:
        message = await receive()
        body += message.get('body', b'')
        if not message.get('more_body', False):
            break
    received = json.loads(body) if body else None
    result = {
        'method': scope['method'],
        'path': scope['path'],
        'headers': {name.decode(): value.decode() for name, value in scope['headers']},
        'body': received,
    }
    request_id = received.get('id') if isinstance(received, dict) else None
    reply = json.dumps({'jsonrpc': '2.0', 'id': request_id, 'result': result}).encode()
    await send(
        {
            'type': 'http.response.start',
            'status': 200,
            'headers': [(b'content-type', b'application/json')],
        }
    )
    await send({'type': 'http.response.body', 'body': reply})


@contextlib.asynccontextmanager
async def _echo_http(harness):
    """``HttpApp``'s helpers in front of ``_echo_app``, to see exactly what they send."""
    transport = httpx.ASGITransport(app=_echo_app)
    async with httpx.AsyncClient(transport=transport, base_url=harness.HTTP_BASE_URL) as client:
        yield harness.HttpApp(None, _echo_app, client)


async def test_http_app_post_mcp_header_none_sends_no_such_header(harness):
    # None drops a header even where httpx would add it by default (Accept: */*, User-Agent).
    message = {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'}
    headers = {'Accept': None, 'user-agent': None, 'X-Extra': 'probe'}
    async with _echo_http(harness) as http:
        received = (await http.post_mcp(message, headers)).message()['result']

    assert 'accept' not in received['headers']
    assert 'user-agent' not in received['headers']
    assert received['headers']['x-extra'] == 'probe'
    assert received['headers']['content-type'] == 'application/json'
    assert received['headers']['host'] == '127.0.0.1:9900'
    assert received['body'] == message


async def test_http_app_get_header_none_sends_no_such_header(harness):
    async with _echo_http(harness) as http:
        reply = await http.get('/health', headers={'accept': None, 'x-extra': 'probe'})

    received = reply.message()['result']
    assert 'accept' not in received['headers']
    assert received['headers']['x-extra'] == 'probe'
    assert (received['method'], received['path'], received['body']) == ('GET', '/health', None)


async def test_http_app_mcp_sends_headers_protocol_version_and_modern_envelope(harness):
    async with _echo_http(harness) as http:
        plain = await http.mcp('tools/list', headers={'x-mcp-client-name': 'probe'})
        handshake = await http.mcp('tools/list', protocol_version='2025-06-18')
        modern = await http.mcp(
            'tools/list', {'cursor': 'c'}, protocol_version='2026-07-28', request_id=7
        )
        named = await http.mcp(
            'tools/call', {'name': 'ListIndexTool'}, protocol_version='2026-07-28'
        )
        overridden = await http.mcp(
            'tools/call',
            {'name': 'ListIndexTool'},
            protocol_version='2026-07-28',
            headers={'mcp-name': None, 'Mcp-Method': 'tools/list'},
        )

    plain = plain['result']
    assert plain['headers']['x-mcp-client-name'] == 'probe'
    assert plain['headers']['accept'] == 'application/json, text/event-stream'
    assert 'mcp-protocol-version' not in plain['headers']
    assert plain['body'] == {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}
    # A handshake version: the version header only.
    handshake = handshake['result']
    assert handshake['headers']['mcp-protocol-version'] == '2025-06-18'
    assert 'mcp-method' not in handshake['headers']
    assert handshake['body'] == {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}
    # A modern version: the routing headers and the per-request _meta envelope.
    modern = modern['result']
    assert modern['headers']['mcp-protocol-version'] == '2026-07-28'
    assert modern['headers']['mcp-method'] == 'tools/list'
    assert 'mcp-name' not in modern['headers']
    assert modern['body'] == {
        'jsonrpc': '2.0',
        'id': 7,
        'method': 'tools/list',
        'params': {
            'cursor': 'c',
            '_meta': {
                harness.PROTOCOL_VERSION_META_KEY: '2026-07-28',
                harness.CLIENT_INFO_META_KEY: harness.HARNESS_CLIENT_INFO,
                harness.CLIENT_CAPABILITIES_META_KEY: {},
            },
        },
    }
    assert named['result']['headers']['mcp-method'] == 'tools/call'
    assert named['result']['headers']['mcp-name'] == 'ListIndexTool'
    # The test's own headers win over the derived ones; None sends none.
    assert overridden['result']['headers']['mcp-method'] == 'tools/list'
    assert 'mcp-name' not in overridden['result']['headers']


async def test_http_app_mcp_without_accept(http_app, golden, normalize):
    # What the real app answers a POST with no Accept header at all.
    async with http_app() as http:
        reply = await http.post_mcp(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}, headers={'accept': None}
        )

    golden('harness_transports/http_mcp_without_accept', normalize(reply.summary()))


async def test_http_app_mcp_at_protocol_versions(http_app, golden, normalize):
    async with http_app() as http:
        default = await http.mcp('tools/list')
        handshake = await http.mcp('tools/list', protocol_version='2025-06-18')
        modern = await http.mcp('tools/list', protocol_version='2026-07-28')

    assert handshake == default
    golden('harness_transports/http_mcp_tools_list_2026_07_28', normalize(modern))


def test_parse_sse_reads_line_endings_multiline_data_and_comments(harness):
    text = (
        ': a comment\r\n'
        'event: message\r\n'
        'data: {"a":\r\n'
        'data:1}\r\n'
        '\r\n'
        'id: 7\n'
        'data: second\n'
        '\n'
        'event: unfinished\r'
    )

    assert harness.parse_sse(text) == [
        {'event': 'message', 'data': '{"a":\n1}'},
        {'id': '7', 'data': 'second'},
    ]


# --- stdio_launcher ------------------------------------------------------------


async def test_stdio_launcher_initialize_and_tools_list(
    harness, stdio_launcher, mcp_session, golden, normalize
):
    async with stdio_launcher() as server:
        initialize = await server.initialize()
        tools = await server.request('tools/list')

    golden('harness_transports/stdio_initialize', normalize(initialize))
    golden('harness_transports/stdio_tools_list', normalize(tools))
    assert server.returncode == 0
    assert not server.killed
    # The launched entry point serves what the in-process server does (`required` lists are
    # built from a set: their order follows the process's hash seed, so compare normalized).
    launched = harness.dump(ListToolsResult.model_validate(tools['result']))
    async with mcp_session() as session:
        assert normalize(launched) == normalize(await session.list_tools())


async def test_stdio_launcher_failure_in_the_block_reaches_the_test_as_itself(stdio_launcher):
    with pytest.raises(pytest.fail.Exception, match='probe failure'):
        async with stdio_launcher() as server:
            await server.initialize()
            pytest.fail('probe failure', pytrace=False)
    # The server was still shut down: stdin closed, clean exit.
    assert server.returncode == 0


async def test_stdio_raw_initialize_then_helpers_complete_the_same_handshake(stdio_launcher):
    # raw('initialize') sends the request alone; the next request completes that handshake
    # (notifications/initialized) instead of initializing a second time.
    async with stdio_launcher() as server:
        initialize = await server.raw('initialize', protocol_version='2025-06-18')
        tools = await server.list_tools()
        listed = await server.raw('tools/list')

    assert initialize['result']['protocolVersion'] == '2025-06-18'
    assert [message['method'] for message in server.sent] == [
        'initialize',
        'notifications/initialized',
        'tools/list',
        'tools/list',
    ]
    assert tools['tools']
    assert 'result' in listed


async def test_stdio_helpers_use_the_version_the_session_was_initialized_at(
    stdio_launcher, fake_opensearch
):
    fake_opensearch.reply('GET', '/_cat/indices', [{'index': 'logs', 'health': 'green'}])
    async with stdio_launcher(env={'OPENSEARCH_NO_AUTH': 'true'}) as server:
        await server.initialize('2025-06-18')
        tools = await server.list_tools()
        result = await server.call_tool('ListIndexTool', {'include_detail': False})
        listed = await server.raw('tools/list', protocol_version='2025-06-18')
        with pytest.raises(AssertionError, match='initialized at 2025-06-18'):
            await server.raw('tools/list', protocol_version='2025-11-25')

    assert [message['method'] for message in server.sent] == [
        'initialize',
        'notifications/initialized',
        'tools/list',
        'tools/call',
        'tools/list',
    ]
    assert server.sent[0]['params']['protocolVersion'] == '2025-06-18'
    assert tools['tools']
    assert result['isError'] is False
    assert 'result' in listed


async def test_stdio_launcher_replays_fake_routes_and_returns_requests(
    stdio_launcher, fake_opensearch
):
    fake_opensearch.version = '2.19.0'
    fake_opensearch.reply('GET', '/_cat/indices', [{'index': 'logs', 'health': 'green'}])
    env = {'OPENSEARCH_USERNAME': 'admin', 'OPENSEARCH_PASSWORD': 's3cret'}
    async with stdio_launcher(env=env) as server:
        result = await server.call_tool('ListIndexTool', {'include_detail': False})
        launched = server.opensearch_requests()

    assert result['isError'] is False
    assert 'logs' in result['content'][0]['text']
    requests = [(r['method'], r['path'], r['params']) for r in launched]
    assert requests == [
        ('GET', '/', {}),
        ('GET', '/', {}),
        ('GET', '/_cat/indices', {'format': 'json'}),
    ]
    assert launched[0]['auth'] == {'kind': 'basic', 'username': 'admin'}
    # Merged into this process's fake when the server exited.
    assert fake_opensearch.requests == launched
    assert fake_opensearch.unmatched == []


async def test_stdio_launcher_unmatched_request_reaches_the_fake(stdio_launcher, fake_opensearch):
    async with stdio_launcher(env={'OPENSEARCH_NO_AUTH': 'true'}) as server:
        result = await server.call_tool('ListIndexTool', {})

    assert result['isError'] is True
    assert {(r['method'], r['path']) for r in fake_opensearch.unmatched} == {
        ('GET', '/_cat/indices')
    }
    with pytest.raises(pytest.fail.Exception, match='no route'):
        fake_opensearch.verify()
    # Consumed here; left in place, the fixture's teardown would fail this test.
    fake_opensearch.unmatched.clear()


def test_fake_table_round_trip(harness, fake_opensearch, pinned_openapi):
    fake_opensearch.version = None
    fake_opensearch.reply('GET', '/_cat/indices', [{'index': 'logs'}], headers={'x-a': '1'})
    fake_opensearch.reply('POST', re.compile(r'/[^/]+/_search', re.IGNORECASE), b'\x00raw')
    fake_opensearch.fail_connection('GET', '/_cluster/health')
    pinned_openapi.error = OSError('fake: offline')

    fake, pinned = harness.fake_from_table(harness.fake_table(fake_opensearch, pinned_openapi))

    assert fake.version is None
    assert fake._routes == fake_opensearch._routes
    assert str(pinned.error) == 'fake: offline'


def test_fake_table_rejects_callable_bodies(harness, fake_opensearch, pinned_openapi):
    fake_opensearch.reply('GET', '/', lambda request: {})

    with pytest.raises(TypeError, match='callable body'):
        harness.fake_table(fake_opensearch, pinned_openapi)


# --- memory_enabled_session ----------------------------------------------------


async def test_memory_enabled_session_lists_memory_tools(
    memory_enabled_session, golden, normalize
):
    async with memory_enabled_session() as session:
        tools = await session.list_tools()

    golden('harness_transports/memory_enabled_tools_list', normalize(tools))
    assert MEMORY_TOOLS <= {tool['name'] for tool in tools['tools']}
