# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenSearchMCPServer: tools, call_tool, capabilities and request context as before MCPServer."""

import anyio
import pytest
from mcp.client import Client
from mcp.client._memory import InMemoryTransport
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.mcpserver import Context, MCPServer, RequestStateBoundary
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp.types import TextContent
from mcp_server_opensearch import mcp_server as mcp_server_module
from mcp_server_opensearch import tool_executor
from mcp_server_opensearch.client_context import request_context_var
from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)
from pydantic import BaseModel
from tools.tool_params import baseToolArgs
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


HANDSHAKE_VERSION = '2025-11-25'
MODERN_VERSION = '2026-07-28'
CLIENT_INFO = {'name': 'unit-test', 'version': '0.0.0'}

# What a client sees today (the characterization goldens under initialize/): only tools.
LEGACY_CAPABILITIES = {'experimental': {}, 'tools': {'listChanged': False}}
MODERN_CAPABILITIES = {'tools': {'listChanged': False}}

# The six handlers MCPServer.__init__ registers that the server must not serve.
UNSERVED_REQUESTS = (
    ('prompts/list', {}),
    ('prompts/get', {'name': 'p'}),
    ('resources/list', {}),
    ('resources/templates/list', {}),
    ('resources/read', {'uri': 'file:///x'}),
    ('subscriptions/listen', {'notifications': {}}),
)


class RequiredArgs(baseToolArgs):
    index: str


def _tool(
    display_name: str | None = None,
    function: AsyncMock | None = None,
    description: str = 'A test tool',
    input_schema: dict | None = None,
    args_model: type[BaseModel] | Mock | None = None,
) -> dict:
    info = {
        'description': description,
        'input_schema': input_schema
        if input_schema is not None
        else {'type': 'object', 'properties': {}},
        'args_model': args_model or Mock(),
        'function': function
        or AsyncMock(return_value=[TextContent(type='text', text='tool result')]),
    }
    if display_name is not None:
        info['display_name'] = display_name
    return info


def _dump(model: BaseModel) -> dict:
    return model.model_dump(mode='json', by_alias=True, exclude_none=True)


def _texts(result) -> list[str]:
    return [c.text for c in result.content]


async def _raw(server: OpenSearchMCPServer, method: str, params: dict, version: str) -> dict:
    """One raw JSON-RPC request on a fresh in-memory connection at ``version``."""
    with anyio.fail_after(10):
        async with InMemoryTransport(server) as (read_stream, write_stream):

            async def send(message) -> None:
                await write_stream.send(SessionMessage(message))

            async def response_to(request_id: int) -> dict:
                while True:
                    item = await read_stream.receive()
                    if isinstance(item, Exception):
                        raise item
                    message = item.message
                    if isinstance(message, (JSONRPCResponse, JSONRPCError)) and (
                        message.id == request_id
                    ):
                        return message.model_dump(mode='json', by_alias=True, exclude_none=True)

            if version == MODERN_VERSION:
                params = {
                    **params,
                    '_meta': {
                        PROTOCOL_VERSION_META_KEY: version,
                        CLIENT_INFO_META_KEY: CLIENT_INFO,
                        CLIENT_CAPABILITIES_META_KEY: {},
                    },
                }
            else:
                init = {'protocolVersion': version, 'capabilities': {}, 'clientInfo': CLIENT_INFO}
                await send(JSONRPCRequest(jsonrpc='2.0', id=0, method='initialize', params=init))
                await response_to(0)
                await send(JSONRPCNotification(jsonrpc='2.0', method='notifications/initialized'))
            await send(JSONRPCRequest(jsonrpc='2.0', id=1, method=method, params=params))
            return await response_to(1)


# --- construction ---


def test_is_an_mcpserver_named_with_instructions_and_no_version(monkeypatch):
    """OpenSearchMCPServer(enabled_tools): an MCPServer, today's name, instructions, version ""."""
    monkeypatch.setattr(mcp_server_module, 'get_server_instructions', lambda: 'the instructions')
    enabled = {'Known': _tool()}

    server = OpenSearchMCPServer(enabled)

    assert isinstance(server, MCPServer)
    assert server.name == 'opensearch-mcp-server'
    assert server.instructions == 'the instructions'
    assert server.version == ''
    assert server.enabled_tools is enabled


# --- tools: names, descriptions, schemas, order, no outputSchema ---


async def test_tool_list_names_descriptions_schemas_and_order():
    """Tool list: display_name or key, the description and input_schema as given, dict order."""
    schema_a = {
        'type': 'object',
        'properties': {'index': {'type': 'string'}},
        'required': ['index'],
    }
    schema_b = {'type': 'object', 'properties': {}}
    enabled = {
        'Zeta': _tool(description='zeta tool', input_schema=schema_a),
        'Alpha': _tool(display_name='Shown', description='alpha tool', input_schema=schema_b),
        'Mid': _tool(description='mid tool'),
    }
    server = OpenSearchMCPServer(enabled)

    tools = await server.list_tools()

    assert [t.name for t in tools] == ['Zeta', 'Shown', 'Mid']
    assert [t.description for t in tools] == ['zeta tool', 'alpha tool', 'mid tool']
    assert tools[0].input_schema == schema_a
    assert tools[1].input_schema == schema_b


async def test_tool_list_has_no_output_schema_or_extra_fields():
    """Tool list, no outputSchema: each tool is exactly name, description and inputSchema."""
    schema = {'type': 'object', 'properties': {'q': {'type': 'string'}}}
    server = OpenSearchMCPServer({'T': _tool(input_schema=schema)})

    async with Client(server, mode='legacy', cache=None) as client:
        listed = await client.list_tools()

    assert [_dump(t) for t in listed.tools] == [
        {'name': 'T', 'description': 'A test tool', 'inputSchema': schema}
    ]
    assert all(t.output_schema is None for t in server._tool_manager.list_tools())


async def test_tool_list_invalid_field_fails_at_tools_list_not_at_startup():
    """Tool list: a numeric description (CLI YAML) starts the server; tools/list answers -32602."""
    server = OpenSearchMCPServer({'Numeric': _tool(description=123)})

    answer = await _raw(server, 'tools/list', {}, HANDSHAKE_VERSION)

    assert answer['error']['code'] == -32602
    assert answer['error']['message'] == 'Invalid request parameters'


async def test_tool_list_empty_when_no_tools_enabled():
    """Tool list: no enabled tools, an empty list."""
    assert await OpenSearchMCPServer({}).list_tools() == []


# --- call_tool ---


async def test_call_tool_success_resolves_display_name():
    """call_tool success: the tool is found by display name and its content returned."""
    function = AsyncMock(return_value=[TextContent(type='text', text='renamed result')])
    server = OpenSearchMCPServer({'Renamed': _tool('ShownName', function)})

    result = await server.call_tool('ShownName', {})

    function.assert_awaited_once()
    assert result.is_error is False
    assert _texts(result) == ['renamed result']
    assert result.structured_content is None


async def test_call_tool_soft_error_is_error_result():
    """call_tool soft error: a log_tool_error-shaped result becomes isError, text unchanged."""
    function = AsyncMock(
        return_value=[{'type': 'text', 'text': 'Error listing: boom', 'is_error': True}]
    )
    server = OpenSearchMCPServer({'Soft': _tool(function=function)})

    result = await server.call_tool('Soft', {})

    assert result.is_error is True
    assert _texts(result) == ['Error listing: boom']


async def test_call_tool_unknown_tool_is_error_result():
    """call_tool unknown tool: an isError text result, not a protocol error."""
    server = OpenSearchMCPServer({'Known': _tool()})

    result = await server.call_tool('Missing', {})

    assert result.is_error is True
    assert _texts(result) == ['Unknown or disabled tool: Missing']


async def test_call_tool_hidden_key_is_unknown_when_display_name_set():
    """call_tool unknown tool: a renamed tool is not reachable under its registry key."""
    server = OpenSearchMCPServer({'Renamed': _tool('ShownName')})

    result = await server.call_tool('Renamed', {})

    assert result.is_error is True
    assert _texts(result) == ['Unknown or disabled tool: Renamed']


async def test_call_tool_validation_error_is_error_result(monkeypatch):
    """call_tool validation error: the validate_args_for_mode text as an isError result."""
    monkeypatch.setattr('tools.tool_params.get_mode', lambda: 'single')
    function = AsyncMock()
    server = OpenSearchMCPServer({'Needs': _tool(function=function, args_model=RequiredArgs)})

    result = await server.call_tool('Needs', {'other': 1})

    function.assert_not_awaited()
    assert result.is_error is True
    assert _texts(result) == ["Missing required field: 'index'\n\nProvided: {'other': 1}"]


async def test_call_tool_exception_text_is_error_result():
    """call_tool exception text: a tool raising ends as an isError result with str(e)."""
    server = OpenSearchMCPServer(
        {'Failing': _tool(function=AsyncMock(side_effect=RuntimeError('tool failed')))}
    )

    result = await server.call_tool('Failing', {})

    assert result.is_error is True
    assert len(result.content) == 1
    assert result.content[0].type == 'text'
    assert result.content[0].text == 'tool failed'


async def test_call_tool_reraises_mcp_error(monkeypatch):
    """call_tool MCPError passthrough: re-raised, so it reaches the SDK as a protocol error."""
    error = MCPError(-32602, 'bad params')
    monkeypatch.setattr(tool_executor, 'execute_tool', AsyncMock(side_effect=error))
    server = OpenSearchMCPServer({'Known': _tool()})

    with pytest.raises(MCPError) as raised:
        await server.call_tool('Known', {})
    assert raised.value is error


async def test_call_tool_mcp_error_is_a_jsonrpc_error_on_the_wire(monkeypatch):
    """call_tool MCPError passthrough: the client gets the JSON-RPC error, not a result."""
    monkeypatch.setattr(
        tool_executor, 'execute_tool', AsyncMock(side_effect=MCPError(-32602, 'bad params'))
    )
    server = OpenSearchMCPServer({'Known': _tool()})

    answer = await _raw(
        server, 'tools/call', {'name': 'Known', 'arguments': {}}, HANDSHAKE_VERSION
    )

    assert answer['error']['code'] == -32602
    assert answer['error']['message'] == 'bad params'
    assert 'result' not in answer


async def test_call_tool_goes_through_execute_tool_with_empty_arguments(monkeypatch):
    """call_tool: execute_tool is the only path, None arguments reach it as {}."""
    fake = AsyncMock(return_value='executed')
    monkeypatch.setattr(tool_executor, 'execute_tool', fake)
    enabled = {'Known': _tool()}
    server = OpenSearchMCPServer(enabled)

    assert await server.call_tool('Known', None) == 'executed'
    fake.assert_awaited_once_with('Known', {}, enabled)


async def test_call_tool_over_the_wire_is_today_result():
    """call_tool success over tools/call: content and isError only, no structuredContent."""
    server = OpenSearchMCPServer({'Known': _tool()})

    answer = await _raw(
        server, 'tools/call', {'name': 'Known', 'arguments': {}}, HANDSHAKE_VERSION
    )

    assert answer['result'] == {
        'content': [{'type': 'text', 'text': 'tool result'}],
        'isError': False,
    }


@pytest.mark.parametrize('version', [HANDSHAKE_VERSION, MODERN_VERSION])
async def test_call_tool_with_request_state_still_runs_the_tool(version):
    """call_tool: a requestState field is ignored as before, not refused by RequestStateBoundary."""
    server = OpenSearchMCPServer({'Known': _tool()})

    assert not any(isinstance(m, RequestStateBoundary) for m in server.middleware)
    answer = await _raw(
        server,
        'tools/call',
        {'name': 'Known', 'arguments': {}, 'requestState': 'not-sealed'},
        version,
    )

    assert 'error' not in answer
    assert answer['result']['isError'] is False
    assert answer['result']['content'] == [{'type': 'text', 'text': 'tool result'}]


# --- request_context_var: set during the call, reset after ---


@pytest.mark.parametrize(
    'side_effect',
    [None, RuntimeError('boom'), MCPError(-32602, 'bad params')],
    ids=['success', 'exception', 'mcp_error'],
)
async def test_request_context_var_set_and_reset(monkeypatch, side_effect):
    """request_context_var set and reset: the context's request during the call, then restored."""
    seen = []

    async def fake_execute_tool(name, arguments, enabled_tools):
        seen.append(request_context_var.get())
        if side_effect is not None:
            raise side_effect
        return 'executed'

    monkeypatch.setattr(tool_executor, 'execute_tool', fake_execute_tool)
    server = OpenSearchMCPServer({'Known': _tool()})
    request = object()
    context = Context(request_context=SimpleNamespace(request=request), mcp_server=server)
    outer = object()
    token = request_context_var.set(outer)
    try:
        if isinstance(side_effect, MCPError):
            with pytest.raises(MCPError):
                await server.call_tool('Known', {}, context)
        else:
            await server.call_tool('Known', {}, context)

        assert seen == [request]
        assert request_context_var.get() is outer
    finally:
        request_context_var.reset(token)


@pytest.mark.parametrize(
    'context',
    [None, Context(), Context(request_context=SimpleNamespace(request=None))],
    ids=['no_context', 'context_without_request_context', 'stdio_request_none'],
)
async def test_request_context_var_none_without_a_request(monkeypatch, context):
    """request_context_var set and reset: None when there is no request (as on stdio today)."""
    seen = []

    async def fake_execute_tool(name, arguments, enabled_tools):
        seen.append(request_context_var.get())
        return 'executed'

    monkeypatch.setattr(tool_executor, 'execute_tool', fake_execute_tool)
    server = OpenSearchMCPServer({'Known': _tool()})
    outer = object()
    token = request_context_var.set(outer)
    try:
        await server.call_tool('Known', {}, context)

        assert seen == [None]
        assert request_context_var.get() is outer
    finally:
        request_context_var.reset(token)


async def test_request_context_var_is_the_in_memory_request_over_tools_call(monkeypatch):
    """request_context_var set and reset: over tools/call it is the transport's ctx.request."""
    seen = []

    async def fake_execute_tool(name, arguments, enabled_tools):
        seen.append(request_context_var.get())
        return tool_executor._build_call_tool_result([], is_error=False)

    monkeypatch.setattr(tool_executor, 'execute_tool', fake_execute_tool)
    server = OpenSearchMCPServer({'Known': _tool()})

    await _raw(server, 'tools/call', {'name': 'Known', 'arguments': {}}, HANDSHAKE_VERSION)

    # The in-memory transport, like stdio, carries no HTTP request.
    assert seen == [None]
    assert request_context_var.get() is None


# --- capabilities: tools only, listChanged false, at both protocol eras ---


def test_capabilities_from_the_lowlevel_server_at_both_versions():
    """Capabilities: tools only, listChanged false, at 2025-11-25 and 2026-07-28."""
    lowlevel = OpenSearchMCPServer({'Known': _tool()})._lowlevel_server

    for version in (None, HANDSHAKE_VERSION, MODERN_VERSION):
        capabilities = lowlevel.get_capabilities(
            NotificationOptions(), {}, protocol_version=version
        )
        assert _dump(capabilities) == LEGACY_CAPABILITIES, version


@pytest.mark.parametrize(
    'mode, version, expected',
    [
        ('legacy', HANDSHAKE_VERSION, LEGACY_CAPABILITIES),
        ('auto', MODERN_VERSION, MODERN_CAPABILITIES),
    ],
)
async def test_capabilities_a_client_sees(mode, version, expected):
    """Capabilities: an SDK client sees exactly today's capabilities at each protocol era."""
    server = OpenSearchMCPServer({'Known': _tool()})

    async with Client(server, mode=mode, cache=None) as client:
        assert client.protocol_version == version
        assert _dump(client.server_capabilities) == expected
        assert _dump(client.server_info) == {'name': 'opensearch-mcp-server', 'version': ''}


async def test_initialize_capabilities_on_the_wire():
    """Capabilities: the 2025-11-25 initialize answer carries tools only."""
    server = OpenSearchMCPServer({'Known': _tool()})

    with anyio.fail_after(10):
        async with InMemoryTransport(server) as (read_stream, write_stream):
            init = {
                'protocolVersion': HANDSHAKE_VERSION,
                'capabilities': {},
                'clientInfo': CLIENT_INFO,
            }
            request = JSONRPCRequest(jsonrpc='2.0', id=0, method='initialize', params=init)
            await write_stream.send(SessionMessage(request))
            message = (await read_stream.receive()).message

    result = message.model_dump(mode='json', by_alias=True, exclude_none=True)['result']
    assert result['protocolVersion'] == HANDSHAKE_VERSION
    assert result['capabilities'] == LEGACY_CAPABILITIES


async def test_discover_capabilities_on_the_wire():
    """Capabilities: the 2026-07-28 server/discover answer carries tools only, listChanged false."""
    server = OpenSearchMCPServer({'Known': _tool()})

    answer = await _raw(server, 'server/discover', {}, MODERN_VERSION)

    assert answer['result']['capabilities'] == MODERN_CAPABILITIES


def test_unserved_handlers_are_removed():
    """Capabilities: the six handlers MCPServer adds are gone; tools/list and tools/call stay."""
    lowlevel = OpenSearchMCPServer({})._lowlevel_server

    for method, _ in UNSERVED_REQUESTS:
        assert lowlevel.get_request_handler(method) is None, method
    assert lowlevel.get_request_handler('tools/list') is not None
    assert lowlevel.get_request_handler('tools/call') is not None


@pytest.mark.parametrize('version', [HANDSHAKE_VERSION, MODERN_VERSION])
@pytest.mark.parametrize(
    'method, params', UNSERVED_REQUESTS, ids=[m for m, _ in UNSERVED_REQUESTS]
)
async def test_unserved_methods_answer_method_not_found(method, params, version):
    """Capabilities: prompts/*, resources/* and subscriptions/listen answer -32601 as today."""
    server = OpenSearchMCPServer({'Known': _tool()})

    answer = await _raw(server, method, params, version)

    assert answer['error']['code'] == -32601
    assert answer['error']['message'] == 'Method not found'
