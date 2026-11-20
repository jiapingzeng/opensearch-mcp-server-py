# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the harness hooks the coverage gaps need (see conftest.py and README.md).

- ``fake_opensearch.fail_streaming``: the streaming attempt of
  ``BufferedAsyncHttpConnection.perform_request`` fails with an unexpected error that is not
  a connection failure (``RuntimeError``), which still reaches the fallback, and the
  fallback retry (``_fallback_perform_request``) is answered;
- ``stdio_launcher(..., entry='module')``: the server started as
  ``python -m mcp_server_opensearch``;
- ``plant_tool``: a tool for one test that raises or returns what the test gives it.
"""

import json
import logging
import os
import pytest
import yarl
from mcp.shared.exceptions import NoBackChannelError
from tools.tools import TOOL_REGISTRY


AREA = 'harness_gap_hooks'
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
CONNECTION_LOGGER = 'opensearch.connection'
EXECUTOR_LOGGER = 'mcp_server_opensearch.tool_executor'
# Fields of the structured log events (connection.py, tool_executor.py) the goldens keep.
EVENT_FIELDS = (
    'event_type',
    'http_method',
    'endpoint',
    'status',
    'status_code',
    'response_size',
    'error',
    'tool_name',
    'tool_key',
    'error_type',
    'duration_ms',
)
# Every tool this module plants starts with this, so their absence can be checked.
PLANTED_PREFIX = 'CharPlanted'
JSON_TYPE = {'content-type': 'application/json; charset=UTF-8'}
INDICES = [{'index': 'logs'}]
INDICES_TEXT = json.dumps(INDICES, separators=(',', ':'))


def log_records(caplog, *loggers: str) -> list[dict]:
    """The records of ``loggers`` caplog holds: message, level and structured event fields."""
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
        }
        for record in caplog.records
        if record.name in loggers
    ]


def attempts(requests: list[dict], path: str) -> list[str | None]:
    """The ``attempt`` marks of the requests to ``path``, in order (``None``: no mark)."""
    return [r.get('attempt') for r in requests if r['path'] == path]


# --- fake_opensearch.fail_streaming ------------------------------------------


async def test_fail_streaming_fails_the_streaming_attempt_and_answers_others(
    harness, fake_opensearch
):
    fake_opensearch.fail_streaming('GET', '/_cat/indices', b'\x00raw', status=201)
    session = harness._FakeAiohttpSession(fake_opensearch)
    url = yarl.URL(f'{fake_opensearch.url}/_cat/indices')

    # What BufferedAsyncHttpConnection.perform_request sends: allow_redirects is set.
    with pytest.raises(RuntimeError, match='during streaming'):
        async with session.request('GET', url, allow_redirects=True):
            pass
    # What opensearch-py's AsyncHttpConnection.perform_request (the fallback) sends.
    async with session.request('GET', url, timeout=None, fingerprint=None) as response:
        status, text = response.status, await response.text()

    assert (status, text) == (201, b'\x00raw')
    assert attempts(fake_opensearch.requests, '/_cat/indices') == ['streaming', 'fallback']


@pytest.mark.parametrize(
    'body',
    ['text', None, 7, INDICES],
    ids=['str', 'none', 'int', 'list'],
)
async def test_fail_streaming_fallback_text_is_the_body_itself(harness, fake_opensearch, body):
    fake_opensearch.fail_streaming('GET', '/', body)
    session = harness._FakeAiohttpSession(fake_opensearch)

    async with session.request('GET', yarl.URL(f'{fake_opensearch.url}/')) as response:
        assert await response.text() is body


FALLBACK_BODIES = [
    # The three branches of _fallback_perform_request's size check: str, bytes, other.
    pytest.param(INDICES_TEXT, JSON_TYPE, id='str'),
    pytest.param(INDICES_TEXT.encode('utf-8'), JSON_TYPE, id='bytes'),
    pytest.param(INDICES, None, id='other'),
]


@pytest.mark.parametrize('body, headers', FALLBACK_BODIES)
async def test_fail_streaming_reaches_the_fallback(
    mcp_session, fake_opensearch, golden, normalize, caplog, request, body, headers
):
    caplog.set_level(logging.INFO, logger=CONNECTION_LOGGER)
    fake_opensearch.fail_streaming('GET', '/_cat/indices', body, headers=headers)
    arguments = {'include_detail': False}
    async with mcp_session(env=NO_AUTH) as session:
        result = await session.call_tool('ListIndexTool', arguments)

    # One streaming attempt, one fallback: two requests for one perform_request.
    assert attempts(fake_opensearch.requests, '/_cat/indices') == ['streaming', 'fallback']
    records = log_records(caplog, CONNECTION_LOGGER)
    assert any('falling back to parent implementation' in r['message'] for r in records)
    golden(
        f'{AREA}/fail_streaming_{request.node.callspec.id}',
        normalize(
            {
                'call': {'name': 'ListIndexTool', 'arguments': arguments},
                'result': result,
                'requests': fake_opensearch.requests,
                'connection_log': records,
            }
        ),
    )


async def test_fail_streaming_fallback_status_and_headers(
    mcp_session, fake_opensearch, golden, normalize, caplog
):
    caplog.set_level(logging.INFO, logger=CONNECTION_LOGGER)
    error = {
        'error': {'type': 'index_not_found_exception', 'reason': 'no such index [missing]'},
        'status': 404,
    }
    fake_opensearch.fail_streaming(
        'GET', '/missing/_mapping', json.dumps(error), status=404, headers=JSON_TYPE
    )
    arguments = {'index': 'missing'}
    async with mcp_session(env=NO_AUTH) as session:
        result = await session.call_tool('IndexMappingTool', arguments)

    assert result['isError'] is True
    assert attempts(fake_opensearch.requests, '/missing/_mapping') == ['streaming', 'fallback']
    golden(
        f'{AREA}/fail_streaming_status_404',
        normalize(
            {
                'call': {'name': 'IndexMappingTool', 'arguments': arguments},
                'result': result,
                'requests': fake_opensearch.requests,
                'connection_log': log_records(caplog, CONNECTION_LOGGER),
            }
        ),
    )


def test_fail_streaming_routes_cross_the_fake_table(harness, fake_opensearch, pinned_openapi):
    fake_opensearch.fail_streaming('GET', '/a', 'text', status=404, headers=JSON_TYPE)
    fake_opensearch.fail_streaming('GET', '/b', b'\x00raw', message='fake: reset')
    fake_opensearch.fail_streaming('GET', '/c', INDICES)
    fake_opensearch.fail_streaming('GET', '/d', None)

    fake, _ = harness.fake_from_table(harness.fake_table(fake_opensearch, pinned_openapi))

    assert fake._routes == fake_opensearch._routes


async def test_fail_streaming_in_a_launched_server(
    stdio_launcher, mcp_session, fake_opensearch, normalize
):
    fake_opensearch.fail_streaming('GET', '/_cat/indices', INDICES_TEXT, headers=JSON_TYPE)
    arguments = {'include_detail': False}
    async with stdio_launcher(env=NO_AUTH) as server:
        launched = await server.call_tool('ListIndexTool', arguments)
    launched_requests = list(fake_opensearch.requests)
    fake_opensearch.requests.clear()
    async with mcp_session(env=NO_AUTH) as session:
        in_process = await session.call_tool('ListIndexTool', arguments)

    assert attempts(launched_requests, '/_cat/indices') == ['streaming', 'fallback']
    assert launched_requests == fake_opensearch.requests
    assert normalize(launched) == normalize(in_process)


# --- stdio_launcher(entry='module') -------------------------------------------


@pytest.mark.parametrize('entry', ['main', 'module'])
async def test_stdio_launcher_entry(stdio_launcher, golden, normalize, entry):
    args = ('--mode', 'single', '--log-format', 'text')
    async with stdio_launcher(*args, env=NO_AUTH, entry=entry) as server:
        initialize = await server.initialize()
        tools = await server.list_tools()
        started = server.entry()
    returncode = server.returncode

    assert returncode == 0, server.stderr
    assert started['entry'] == entry
    assert started['args'] == list(args)
    golden(
        f'{AREA}/stdio_entry_{entry}',
        normalize(
            {
                'entry': started,
                'initialize': initialize,
                'tools': [tool['name'] for tool in tools['tools']],
                'returncode': returncode,
            }
        ),
    )


async def test_stdio_launcher_module_entry_serves_what_main_serves(stdio_launcher, normalize):
    served = {}
    for entry in ('main', 'module'):
        async with stdio_launcher(env=NO_AUTH, entry=entry) as server:
            served[entry] = (
                await server.initialize(),
                await server.list_tools(),
                await server.raw('tools/call', {'name': 'NoSuchTool', 'arguments': {}}),
            )

    assert normalize(served['module']) == normalize(served['main'])


async def test_stdio_launcher_rejects_an_unknown_entry(stdio_launcher):
    with pytest.raises(ValueError, match='entry must be one of'):
        async with stdio_launcher(entry='console'):
            pass


# --- plant_tool ----------------------------------------------------------------


def planted_names() -> list[str]:
    return [name for name in TOOL_REGISTRY if name.startswith(PLANTED_PREFIX)]


def test_no_planted_tool_at_test_start():
    # Whichever order the tests run in: every planted tool went with the test that planted it.
    assert planted_names() == []
    assert PLANTED_PREFIX not in os.environ.get('OPENSEARCH_ENABLED_TOOLS', '')


RAISES = [
    pytest.param(
        lambda harness: harness.SDK_MCP_ERROR(
            -32602, 'characterization: planted MCP error', {'reason': 'planted'}
        ),
        id='mcp_error',
    ),
    pytest.param(lambda harness: ValueError('characterization: planted ValueError'), id='value'),
    pytest.param(lambda harness: RuntimeError('characterization: planted'), id='runtime'),
]


@pytest.mark.parametrize('make_error', RAISES)
async def test_planted_tool_raises(
    harness, plant_tool, mcp_session, golden, normalize, caplog, request, make_error
):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    name = f'{PLANTED_PREFIX}Raises'
    planted = plant_tool(name, raises=make_error(harness))
    params = {'name': name, 'arguments': {}}
    async with mcp_session(env=NO_AUTH) as session:
        listed = [t for t in (await session.list_tools())['tools'] if t['name'] == name]
        response = await session.raw('tools/call', params)

    assert len(listed) == 1
    # The executor parsed the arguments with the tool's args model, then called it.
    assert planted.calls == [{'opensearch_cluster_name': ''}]
    golden(
        f'{AREA}/planted_raises_{request.node.callspec.id}',
        normalize(
            {
                'tool': listed[0],
                'request': params,
                'response': response,
                'executor_log': log_records(caplog, EXECUTOR_LOGGER),
            }
        ),
    )


RETURNS = [
    pytest.param([], id='empty_list'),
    pytest.param(None, id='none'),
    pytest.param([{'type': 'text', 'text': 'planted'}], id='text'),
    pytest.param([{'type': 'text', 'text': 'planted error', 'is_error': True}], id='soft_error'),
]


@pytest.mark.parametrize('content', RETURNS)
async def test_planted_tool_returns(
    plant_tool, mcp_session, golden, normalize, caplog, request, content
):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    name = f'{PLANTED_PREFIX}Returns'
    plant_tool(name, returns=content)
    async with mcp_session(env=NO_AUTH) as session:
        result = await session.call_tool(name, {})

    golden(
        f'{AREA}/planted_returns_{request.node.callspec.id}',
        normalize({'result': result, 'executor_log': log_records(caplog, EXECUTOR_LOGGER)}),
    )


async def test_planted_tool_goes_through_the_filter(plant_tool, mcp_session):
    # Not enabled: single mode's filter keeps only core_tools, as for any such real tool.
    name = f'{PLANTED_PREFIX}NotEnabled'
    plant_tool(name, returns=[], enable=False)
    async with mcp_session(env=NO_AUTH) as session:
        single = [t['name'] for t in (await session.list_tools())['tools']]
        call = await session.call_tool(name, {})
    async with mcp_session(mode='multi') as session:
        multi = [t['name'] for t in (await session.list_tools())['tools']]

    assert name not in single
    assert call['isError'] is True
    assert f'Unknown or disabled tool: {name}' in call['content'][0]['text']
    # Multi mode does not filter by category.
    assert name in multi


async def test_planted_tool_in_a_launched_server(
    harness, plant_tool, stdio_launcher, mcp_session, golden, normalize
):
    error = harness.SDK_MCP_ERROR(-32001, 'characterization: planted in a launched server')
    plant_tool(f'{PLANTED_PREFIX}Raises', raises=error)
    plant_tool(f'{PLANTED_PREFIX}ReturnsNone', returns=None)
    calls = [{'name': tool.name, 'arguments': {}} for tool in plant_tool.planted]
    async with stdio_launcher(env=NO_AUTH) as server:
        launched = [await server.raw('tools/call', params) for params in calls]
    async with mcp_session(env=NO_AUTH) as session:
        in_process = [await session.raw('tools/call', params) for params in calls]

    # The request ids differ: the stdio session counts them up, raw() always sends id 1.
    def without_id(responses: list[dict]) -> list[dict]:
        return [{k: v for k, v in response.items() if k != 'id'} for response in responses]

    assert normalize(without_id(launched)) == normalize(without_id(in_process))
    golden(f'{AREA}/planted_stdio', normalize({'calls': calls, 'responses': launched}))


async def test_planted_tool_over_http_app(harness, plant_tool, http_app, mcp_session):
    name = f'{PLANTED_PREFIX}Raises'
    plant_tool(name, raises=harness.SDK_MCP_ERROR(-32001, 'characterization: over HTTP'))
    params = {'name': name, 'arguments': {}}
    async with http_app(env=NO_AUTH) as http:
        over_http = await http.mcp('tools/call', params)
    async with mcp_session(env=NO_AUTH) as session:
        in_process = await session.raw('tools/call', params)

    assert over_http == in_process
    assert over_http['error']['message'] == 'characterization: over HTTP'


async def test_planted_tool_is_undone_with_its_monkeypatch(harness, mcp_session):
    name = f'{PLANTED_PREFIX}Scoped'
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness.ToolPlanter(monkeypatch)(name, returns=[])
        assert name in TOOL_REGISTRY
        assert os.environ['OPENSEARCH_ENABLED_TOOLS'].split(',')[-1] == name

    assert name not in TOOL_REGISTRY
    assert 'OPENSEARCH_ENABLED_TOOLS' not in os.environ
    async with mcp_session(mode='multi') as session:
        assert name not in [t['name'] for t in (await session.list_tools())['tools']]


def test_plant_tool_keeps_enabled_tools_already_set(plant_tool, monkeypatch):
    monkeypatch.setenv('OPENSEARCH_ENABLED_TOOLS', 'GetClusterStateTool')
    plant_tool(f'{PLANTED_PREFIX}Kept', returns=[])

    assert os.environ['OPENSEARCH_ENABLED_TOOLS'] == f'GetClusterStateTool,{PLANTED_PREFIX}Kept'


def test_plant_tool_refuses_a_real_tool_name(plant_tool):
    with pytest.raises(ValueError, match='already in TOOL_REGISTRY'):
        plant_tool('ListIndexTool', returns=[])


def test_planted_exception_must_be_rebuildable_to_cross(harness, plant_tool):
    plant_tool(f'{PLANTED_PREFIX}Ok', raises=harness.SDK_MCP_ERROR(-1, 'crosses', {'a': [1]}))
    assert plant_tool.specs() == [
        {
            'name': f'{PLANTED_PREFIX}Ok',
            'raises': {
                'class': 'mcp.shared.exceptions:MCPError',
                'args': [-1, 'crosses', {'a': [1]}],
            },
        }
    ]
    rebuilt = harness.exception_from_spec(plant_tool.specs()[0]['raises'])
    assert (type(rebuilt), rebuilt.code, rebuilt.message) == (harness.SDK_MCP_ERROR, -1, 'crosses')

    # Its args (code, message, data) do not match its constructor (method).
    plant_tool(f'{PLANTED_PREFIX}NoCross', raises=NoBackChannelError('sampling/createMessage'))
    with pytest.raises(TypeError, match='cannot be rebuilt in a launched server'):
        plant_tool.specs()


def test_planted_content_must_be_json_to_cross(plant_tool):
    plant_tool(f'{PLANTED_PREFIX}Object', returns=[object()])
    with pytest.raises(TypeError, match='give it as JSON data'):
        plant_tool.specs()
