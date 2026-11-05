# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from mcp.client.client import Client
from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp.shared.exceptions import MCPError
from mcp_server_opensearch.mcp_handlers import build_handlers, configure_apps
from mcp_server_opensearch.tool_executor import _build_call_tool_result
from tools.ui_resources import RESULTS_VIEW_URI, STRUCTURED_CONTENT_MAX_BYTES
from unittest.mock import AsyncMock, Mock, patch


STRUCTURED = {'kind': 'search_hits', 'version': 1, 'cluster': '', 'columns': [], 'rows': []}


def _tool(function, ui=True, category='core_tools'):
    info = {
        'display_name': None,
        'description': 'd',
        'input_schema': {'type': 'object'},
        'args_model': Mock(),
        'function': function,
        'category': category,
    }
    if ui:
        info['ui_resource_uri'] = RESULTS_VIEW_URI
    return info


def _registry():
    search = AsyncMock(return_value=[{'type': 'text', 'text': 'hits', 'structured': STRUCTURED}])
    plain = AsyncMock(return_value=[{'type': 'text', 'text': 'indices'}])
    tools = {'SearchIndexTool': _tool(search), 'ListIndexTool': _tool(plain, ui=False)}
    for name, info in tools.items():
        info['display_name'] = name
    return tools


def _server(tools):
    server = Server('opensearch-mcp-server', **build_handlers(tools))
    configure_apps(server)
    return server


@pytest.fixture
def apps_on(monkeypatch):
    monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')


@pytest.fixture
def apps_off(monkeypatch):
    monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)


@pytest.fixture(autouse=True)
def single_mode():
    with (
        patch('mcp_server_opensearch.global_state.get_mode', return_value='single'),
        patch('tools.tool_params.get_mode', return_value='single'),
        patch('opensearch.client.is_serverless_connection', return_value=False),
        patch('tools.tool_params.validate_args_for_mode', side_effect=lambda a, m: a),
    ):
        yield


class TestFlagOff:
    def test_capabilities_unchanged(self, apps_off):
        caps = _server(_registry()).get_capabilities()
        assert caps.extensions is None
        assert caps.resources is None

    @pytest.mark.asyncio
    async def test_tools_list_unchanged(self, apps_off):
        async with Client(_server(_registry())) as client:
            tools = (await client.list_tools()).tools
        assert [t.meta for t in tools] == [{'category': 'core_tools'}, {'category': 'core_tools'}]

    @pytest.mark.asyncio
    async def test_resources_not_served(self, apps_off):
        async with Client(_server(_registry())) as client:
            with pytest.raises(MCPError):
                await client.list_resources()

    @pytest.mark.asyncio
    async def test_call_tool_has_no_structured_content(self, apps_off):
        tools = _registry()
        tools['SearchIndexTool']['function'].return_value = [{'type': 'text', 'text': 'hits'}]
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.structured_content is None
        assert result.content[0].text == 'hits'


class TestFlagOn:
    def test_extension_advertised(self, apps_on):
        caps = _server(_registry()).get_capabilities()
        assert caps.extensions == {EXTENSION_ID: {}}
        assert caps.resources is not None

    @pytest.mark.asyncio
    async def test_ui_meta_only_on_bound_tools(self, apps_on):
        async with Client(_server(_registry())) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools['SearchIndexTool'].meta == {
            'category': 'core_tools',
            'ui': {'resourceUri': RESULTS_VIEW_URI},
        }
        assert tools['ListIndexTool'].meta == {'category': 'core_tools'}

    @pytest.mark.asyncio
    async def test_ui_meta_without_category(self, apps_on):
        tools = _registry()
        del tools['SearchIndexTool']['category']
        async with Client(_server(tools)) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
        assert listed['SearchIndexTool'].meta == {'ui': {'resourceUri': RESULTS_VIEW_URI}}

    @pytest.mark.asyncio
    async def test_list_and_read_resource(self, apps_on):
        async with Client(_server(_registry())) as client:
            resources = (await client.list_resources()).resources
            read = await client.read_resource(RESULTS_VIEW_URI)
        assert len(resources) == 1
        assert str(resources[0].uri) == RESULTS_VIEW_URI
        assert resources[0].mime_type == APP_MIME_TYPE
        assert resources[0].meta == {'ui': {'prefersBorder': True}}
        content = read.contents[0]
        assert content.mime_type == APP_MIME_TYPE
        assert content.meta == {'ui': {'prefersBorder': True}}
        assert content.text.startswith('<!DOCTYPE html>')

    @pytest.mark.asyncio
    async def test_filtered_out_tool_hides_resource(self, apps_on):
        tools = _registry()
        del tools['SearchIndexTool']
        async with Client(_server(tools)) as client:
            assert (await client.list_resources()).resources == []
            with pytest.raises(MCPError, match='Resource not found'):
                await client.read_resource(RESULTS_VIEW_URI)

    @pytest.mark.asyncio
    async def test_unknown_uri_errors(self, apps_on):
        async with Client(_server(_registry())) as client:
            with pytest.raises(MCPError, match='Resource not found'):
                await client.read_resource('ui://opensearch/unknown')

    @pytest.mark.asyncio
    async def test_call_tool_returns_structured_content(self, apps_on):
        async with Client(_server(_registry())) as client:
            result = await client.call_tool('SearchIndexTool', {})
            plain = await client.call_tool('ListIndexTool', {})
        assert result.structured_content == STRUCTURED
        assert result.content[0].text == 'hits'
        assert plain.structured_content is None


class TestBuildCallToolResult:
    def test_structured_passed_through(self):
        result = _build_call_tool_result(
            [{'type': 'text', 'text': 't', 'structured': STRUCTURED}], False
        )
        assert result.structured_content == STRUCTURED
        assert result.content[0].text == 't'

    def test_structured_over_cap_omitted(self):
        big = {**STRUCTURED, 'rows': [['x' * STRUCTURED_CONTENT_MAX_BYTES]]}
        result = _build_call_tool_result([{'type': 'text', 'text': 't', 'structured': big}], False)
        assert result.structured_content is None
        assert result.content[0].text == 't'

    def test_error_result_has_no_structured(self):
        result = _build_call_tool_result(
            [{'type': 'text', 'text': 'e', 'structured': STRUCTURED}], True
        )
        assert result.structured_content is None

    def test_wire_format_without_structured_unchanged(self):
        result = _build_call_tool_result([{'type': 'text', 'text': 't'}], False)
        assert 'structuredContent' not in result.model_dump(by_alias=True, exclude_none=True)


class TestTransports:
    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.streaming_server.get_tools')
    @patch('mcp_server_opensearch.streaming_server.generate_tools_from_openapi')
    async def test_streaming_server_apps(self, mock_generate, mock_get_tools, apps_on):
        from mcp_server_opensearch.streaming_server import create_mcp_server

        mock_get_tools.return_value = _registry()
        server = await create_mcp_server()
        assert server.get_capabilities().extensions == {EXTENSION_ID: {}}
        async with Client(server) as client:
            assert len((await client.list_resources()).resources) == 1

    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.streaming_server.get_tools')
    @patch('mcp_server_opensearch.streaming_server.generate_tools_from_openapi')
    async def test_streaming_server_flag_off(self, mock_generate, mock_get_tools, apps_off):
        from mcp_server_opensearch.streaming_server import create_mcp_server

        mock_get_tools.return_value = _registry()
        server = await create_mcp_server()
        assert server.get_capabilities().extensions is None
        assert server.get_capabilities().resources is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize('enabled', [True, False])
    async def test_stdio_server_apps(self, monkeypatch, enabled):
        from contextlib import asynccontextmanager

        if enabled:
            monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        else:
            monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        built = {}

        def fake_server(*args, **kwargs):
            built['server'] = Server(*args, **kwargs)
            built['server'].run = AsyncMock()
            return built['server']

        @asynccontextmanager
        async def fake_stdio():
            yield AsyncMock(), AsyncMock()

        with (
            patch('mcp_server_opensearch.stdio_server.Server', side_effect=fake_server),
            patch('mcp_server_opensearch.stdio_server.stdio_server', fake_stdio),
            patch(
                'mcp_server_opensearch.stdio_server.get_tools', AsyncMock(return_value=_registry())
            ),
            patch('mcp_server_opensearch.stdio_server.generate_tools_from_openapi', AsyncMock()),
        ):
            from mcp_server_opensearch.stdio_server import serve

            await serve()
        caps = built['server'].get_capabilities()
        assert (caps.extensions == {EXTENSION_ID: {}}) is enabled
        assert (caps.resources is not None) is enabled
