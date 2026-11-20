# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolRequestParams, PaginatedRequestParams, TextContent
from mcp_server_opensearch import server_factory
from mcp_server_opensearch.client_context import request_context_var
from tools.tools import TOOL_REGISTRY
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


ENABLED_TOOLS = {'ToolA': {'description': 'a'}, 'ToolB': {'description': 'b'}}
CUSTOMIZED_REGISTRY = {'customized': {}}

PIPELINE_STEPS = (
    'set_mode',
    'set_profile',
    'set_config_file_path',
    'load_clusters_from_yaml',
    'generate_tools_from_openapi',
    'apply_custom_tool_config',
    'get_tools',
)


@pytest.fixture
def pipeline(monkeypatch):
    """Replace every pipeline step in server_factory with a child of one recording mock."""
    manager = Mock()
    manager.attach_mock(Mock(), 'set_mode')
    manager.attach_mock(Mock(), 'set_profile')
    manager.attach_mock(Mock(), 'set_config_file_path')
    manager.attach_mock(AsyncMock(), 'load_clusters_from_yaml')
    manager.attach_mock(AsyncMock(), 'generate_tools_from_openapi')
    manager.attach_mock(Mock(return_value=CUSTOMIZED_REGISTRY), 'apply_custom_tool_config')
    manager.attach_mock(AsyncMock(return_value=ENABLED_TOOLS), 'get_tools')
    for name in PIPELINE_STEPS:
        monkeypatch.setattr(server_factory, name, getattr(manager, name))
    return manager


def _steps(manager: Mock) -> list[str]:
    return [name for name, _, _ in manager.mock_calls]


# --- prepare_enabled_tools: the startup pipeline ---


async def test_pipeline_runs_every_step_in_order(pipeline):
    """Call order: mode, profile, config path, clusters, generator, custom config, filter."""
    result = await server_factory.prepare_enabled_tools(
        mode='multi',
        profile='my-profile',
        config_file_path='some/path.yml',
        cli_tool_overrides={'key': 'val'},
    )

    assert result is ENABLED_TOOLS
    assert _steps(pipeline) == list(PIPELINE_STEPS)
    pipeline.set_mode.assert_called_once_with('multi')
    pipeline.set_profile.assert_called_once_with('my-profile')
    pipeline.set_config_file_path.assert_called_once_with('some/path.yml')
    pipeline.load_clusters_from_yaml.assert_awaited_once_with('some/path.yml')
    pipeline.generate_tools_from_openapi.assert_awaited_once_with()
    pipeline.apply_custom_tool_config.assert_called_once_with(
        TOOL_REGISTRY, 'some/path.yml', {'key': 'val'}
    )
    assert pipeline.apply_custom_tool_config.call_args.args[0] is TOOL_REGISTRY
    pipeline.get_tools.assert_awaited_once_with(
        tool_registry=CUSTOMIZED_REGISTRY, config_file_path='some/path.yml'
    )


async def test_pipeline_defaults_set_only_the_mode(pipeline):
    """Globals set only when truthy: an empty profile and config path are not set."""
    await server_factory.prepare_enabled_tools()

    pipeline.set_mode.assert_called_once_with('single')
    pipeline.set_profile.assert_not_called()
    pipeline.set_config_file_path.assert_not_called()
    pipeline.apply_custom_tool_config.assert_called_once_with(TOOL_REGISTRY, '', {})
    pipeline.get_tools.assert_awaited_once_with(
        tool_registry=CUSTOMIZED_REGISTRY, config_file_path=''
    )


@pytest.mark.parametrize(
    'profile, config_file_path, profile_set, config_set',
    [
        ('p', '', True, False),
        ('', 'c.yml', False, True),
        ('p', 'c.yml', True, True),
    ],
)
async def test_pipeline_sets_profile_and_config_path_only_when_truthy(
    pipeline, profile, config_file_path, profile_set, config_set
):
    """Globals set only when truthy, each independently of the other."""
    await server_factory.prepare_enabled_tools(profile=profile, config_file_path=config_file_path)

    assert pipeline.set_profile.called is profile_set
    assert pipeline.set_config_file_path.called is config_set
    if profile_set:
        pipeline.set_profile.assert_called_once_with(profile)
    if config_set:
        pipeline.set_config_file_path.assert_called_once_with(config_file_path)


async def test_pipeline_single_mode_does_not_load_clusters(pipeline):
    """Multi mode loads clusters: single mode does not."""
    await server_factory.prepare_enabled_tools(mode='single', config_file_path='c.yml')

    pipeline.load_clusters_from_yaml.assert_not_awaited()
    assert 'load_clusters_from_yaml' not in _steps(pipeline)


async def test_pipeline_multi_mode_loads_clusters_even_without_config(pipeline):
    """Multi mode loads clusters from the config path, even when it is empty."""
    await server_factory.prepare_enabled_tools(mode='multi')

    pipeline.load_clusters_from_yaml.assert_awaited_once_with('')


async def test_pipeline_none_overrides_become_empty_dict(pipeline):
    """cli_tool_overrides=None reaches apply_custom_tool_config as {}."""
    await server_factory.prepare_enabled_tools(cli_tool_overrides=None)

    assert pipeline.apply_custom_tool_config.call_args.args[2] == {}


async def test_pipeline_logs_enabled_tools(pipeline, caplog):
    """The pipeline ends with the 'Enabled tools: [...]' info log."""
    caplog.set_level(logging.INFO)

    await server_factory.prepare_enabled_tools()

    records = [r for r in caplog.records if r.getMessage().startswith('Enabled tools:')]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert records[0].getMessage() == "Enabled tools: ['ToolA', 'ToolB']"


# --- build_server: the server and its handlers ---


def _tool(display_name: str | None = None, function: AsyncMock | None = None) -> dict:
    info = {
        'description': 'A test tool',
        'input_schema': {'type': 'object', 'properties': {}},
        'args_model': Mock(),
        'function': function
        or AsyncMock(return_value=[TextContent(type='text', text='tool result')]),
    }
    if display_name is not None:
        info['display_name'] = display_name
    return info


def _handler(server, method: str):
    entry = server.get_request_handler(method)
    assert entry is not None, f'no handler for {method}'
    return entry.handler


async def _call(server, name: str, arguments: dict | None = None, request=None):
    ctx = SimpleNamespace(request=request)
    params = CallToolRequestParams(name=name, arguments=arguments)
    return await _handler(server, 'tools/call')(ctx, params)


def test_build_server_name_and_instructions(monkeypatch):
    """The server is named opensearch-mcp-server and carries get_server_instructions()."""
    monkeypatch.setattr(server_factory, 'get_server_instructions', lambda: 'the instructions')

    server = server_factory.build_server({})

    assert server.name == 'opensearch-mcp-server'
    assert server.instructions == 'the instructions'


async def test_list_tools_uses_display_name_with_key_fallback():
    """Display name: listed under display_name when set, else under the registry key."""
    server = server_factory.build_server(
        {'KeyOnly': _tool(), 'Renamed': _tool(display_name='ShownName')}
    )

    result = await _handler(server, 'tools/list')(
        SimpleNamespace(request=None), PaginatedRequestParams()
    )

    assert [tool.name for tool in result.tools] == ['KeyOnly', 'ShownName']
    assert all(tool.description == 'A test tool' for tool in result.tools)
    assert all(tool.input_schema == {'type': 'object', 'properties': {}} for tool in result.tools)


async def test_call_tool_resolves_display_name():
    """Display name: tools/call finds the tool by its display name."""
    function = AsyncMock(return_value=[TextContent(type='text', text='renamed result')])
    server = server_factory.build_server({'Renamed': _tool('ShownName', function)})

    result = await _call(server, 'ShownName', {})

    function.assert_awaited_once()
    assert result.is_error is False
    assert [c.text for c in result.content] == ['renamed result']


async def test_call_tool_unknown_tool_is_error_result():
    """Unknown tool: an isError text result, not a protocol error."""
    server = server_factory.build_server({'Known': _tool()})

    result = await _call(server, 'Missing', {})

    assert result.is_error is True
    assert [c.text for c in result.content] == ['Unknown or disabled tool: Missing']


async def test_call_tool_passes_empty_arguments_when_none(monkeypatch):
    """Arguments left out reach execute_tool as {}."""
    from mcp_server_opensearch import tool_executor

    fake = AsyncMock(return_value='executed')
    monkeypatch.setattr(tool_executor, 'execute_tool', fake)
    enabled = {'Known': _tool()}
    server = server_factory.build_server(enabled)

    assert await _call(server, 'Known', None) == 'executed'
    fake.assert_awaited_once_with('Known', {}, enabled)


async def test_call_tool_reraises_mcp_error(monkeypatch):
    """MCPError re-raised: it reaches the SDK as a protocol error."""
    from mcp_server_opensearch import tool_executor

    error = MCPError(-32602, 'bad params')
    monkeypatch.setattr(tool_executor, 'execute_tool', AsyncMock(side_effect=error))
    server = server_factory.build_server({'Known': _tool()})

    with pytest.raises(MCPError) as raised:
        await _call(server, 'Known', {})
    assert raised.value is error


async def test_call_tool_exception_becomes_is_error_result(monkeypatch):
    """Exception to isError: any other exception is returned as an isError text result."""
    from mcp_server_opensearch import tool_executor

    monkeypatch.setattr(tool_executor, 'execute_tool', AsyncMock(side_effect=RuntimeError('boom')))
    server = server_factory.build_server({'Known': _tool()})

    result = await _call(server, 'Known', {})

    assert result.is_error is True
    assert len(result.content) == 1
    assert result.content[0].type == 'text'
    assert result.content[0].text == 'boom'


async def test_call_tool_tool_exception_becomes_is_error_result():
    """Exception to isError: a tool function raising ends as an isError result."""
    server = server_factory.build_server(
        {'Failing': _tool(function=AsyncMock(side_effect=RuntimeError('tool failed')))}
    )

    result = await _call(server, 'Failing', {})

    assert result.is_error is True
    assert [c.text for c in result.content] == ['tool failed']


@pytest.mark.parametrize(
    'side_effect',
    [None, RuntimeError('boom'), MCPError(-32602, 'bad params')],
    ids=['success', 'exception', 'mcp_error'],
)
async def test_call_tool_sets_and_resets_request_context_var(monkeypatch, side_effect):
    """request_context_var reset after the call: set to ctx.request during it, then restored."""
    from mcp_server_opensearch import tool_executor

    seen = []

    async def fake_execute_tool(name, arguments, enabled_tools):
        seen.append(request_context_var.get())
        if side_effect is not None:
            raise side_effect
        return 'executed'

    monkeypatch.setattr(tool_executor, 'execute_tool', fake_execute_tool)
    server = server_factory.build_server({'Known': _tool()})
    outer = object()
    request = object()
    token = request_context_var.set(outer)
    try:
        if isinstance(side_effect, MCPError):
            with pytest.raises(MCPError):
                await _call(server, 'Known', {}, request=request)
        else:
            await _call(server, 'Known', {}, request=request)

        assert seen == [request]
        assert request_context_var.get() is outer
    finally:
        request_context_var.reset(token)
