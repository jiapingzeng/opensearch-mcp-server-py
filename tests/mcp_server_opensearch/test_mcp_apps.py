# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from mcp.client.client import Client
from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp.shared.exceptions import MCPError
from mcp_server_opensearch.mcp_handlers import build_handlers, configure_apps
from mcp_server_opensearch.tool_executor import _build_call_tool_result
from tools.tool_annotations import read_only_annotations, write_annotations
from tools.ui_resources import RESULTS_VIEW_URI, STRUCTURED_CONTENT_MAX_BYTES
from unittest.mock import AsyncMock, Mock, patch


STRUCTURED = {'kind': 'search_hits', 'version': 1, 'cluster': '', 'columns': [], 'rows': []}


def _tool(function, ui=True, category='core_tools', read_only=True):
    info = {
        'display_name': None,
        'description': 'd',
        'input_schema': {'type': 'object', 'properties': {'index': {'type': 'string'}}},
        'args_model': Mock(),
        'function': function,
        'category': category,
        'annotations': read_only_annotations('t')
        if read_only
        else write_annotations('t', destructive=True, idempotent=False),
    }
    if ui:
        info['ui_resource_uri'] = RESULTS_VIEW_URI
        info['ui_kind'] = 'search_hits'
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
        assert all('narrative' not in t.input_schema['properties'] for t in tools)

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
    async def test_ui_meta_and_visibility(self, apps_on):
        tools = _registry()
        tools['DeleteIndexTool'] = _tool(AsyncMock(), ui=False, read_only=False)
        tools['DeleteIndexTool']['display_name'] = 'DeleteIndexTool'
        tools['GenericOpenSearchApiTool'] = _tool(AsyncMock())
        tools['GenericOpenSearchApiTool']['display_name'] = 'GenericOpenSearchApiTool'
        tools['WriteViewTool'] = _tool(AsyncMock(), read_only=False)
        tools['WriteViewTool']['display_name'] = 'WriteViewTool'
        async with Client(_server(tools)) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
        assert listed['SearchIndexTool'].meta == {
            'category': 'core_tools',
            'ui': {'resourceUri': RESULTS_VIEW_URI},
        }
        model_only = {'visibility': ['model']}
        assert listed['ListIndexTool'].meta == {'category': 'core_tools', 'ui': model_only}
        assert listed['DeleteIndexTool'].meta['ui'] == model_only
        assert listed['GenericOpenSearchApiTool'].meta['ui'] == {
            'resourceUri': RESULTS_VIEW_URI,
            **model_only,
        }
        assert listed['WriteViewTool'].meta['ui']['visibility'] == ['model']

    @pytest.mark.asyncio
    async def test_presentation_args_only_on_bound_tools(self, apps_on):
        async with Client(_server(_registry())) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
        search = listed['SearchIndexTool'].input_schema['properties']
        assert search['narrative']['type'] == 'string'
        assert search['suggestions']['maxItems'] == 3
        assert 'index' in search
        assert 'narrative' not in listed['ListIndexTool'].input_schema['properties']

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
        assert result.structured_content == {**STRUCTURED, 'tool': 'SearchIndexTool'}
        assert result.content[0].text == 'hits'
        assert plain.structured_content is None

    @pytest.mark.asyncio
    async def test_presentation_args_moved_to_payload(self, apps_on):
        tools = _registry()
        args = {'index': 'logs', 'narrative': 'Mostly 200s.', 'suggestions': ['Show 500s']}
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', args)
        assert tools['SearchIndexTool']['function'].call_args.args[0] == {'index': 'logs'}
        assert result.structured_content['narrative'] == 'Mostly 200s.'
        assert result.structured_content['suggestions'] == ['Show 500s']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'extra',
        [
            {'narrative': 5},
            {'narrative': 'x' * 2001},
            {'suggestions': 'one'},
            {'suggestions': ['a', 'b', 'c', 'd']},
            {'suggestions': ['']},
            {'suggestions': ['x' * 121]},
        ],
    )
    async def test_invalid_presentation_args_rejected(self, apps_on, extra):
        tools = _registry()
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {'index': 'logs', **extra})
        assert result.is_error
        assert tools['SearchIndexTool']['function'].call_count == 0
        assert result.structured_content['omitted'] == 'error'

    @pytest.mark.asyncio
    async def test_error_marker(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 'boom', 'is_error': True}
        ]
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.is_error
        assert result.structured_content == {
            'kind': 'search_hits',
            'version': 1,
            'omitted': 'error',
            'tool': 'SearchIndexTool',
        }

    @pytest.mark.asyncio
    async def test_unsupported_format_marker(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 'a|b', 'structured': None}
        ]
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.structured_content['omitted'] == 'unsupported_format'

    @pytest.mark.asyncio
    async def test_too_large_marker(self, apps_on):
        tools = _registry()
        big = {**STRUCTURED, 'columns': ['a'], 'rows': [['x' * STRUCTURED_CONTENT_MAX_BYTES]]}
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 't', 'structured': big}
        ]
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {'narrative': 'n'})
        sc = result.structured_content
        assert sc['omitted'] == 'too_large' and sc['bytes'] > STRUCTURED_CONTENT_MAX_BYTES
        assert sc['narrative'] == 'n' and sc['tool'] == 'SearchIndexTool'
        assert 'rows' not in sc
        assert result.content[0].text == 't'

    def test_malformed_dashboards_url_fails_at_startup(self, apps_on, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'dashboards.local')
        with pytest.raises(ValueError, match='OPENSEARCH_DASHBOARDS_URL'):
            build_handlers(_registry())


class TestBuildCallToolResult:
    def test_structured_passed_through(self):
        result = _build_call_tool_result(
            [{'type': 'text', 'text': 't', 'structured': STRUCTURED}], False
        )
        assert result.structured_content == STRUCTURED
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
