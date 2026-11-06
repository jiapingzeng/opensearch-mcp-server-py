# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import pytest
from mcp.client.client import Client
from mcp.client.extension import advertise
from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp.shared.exceptions import MCPError
from mcp.types import (
    METHOD_NOT_FOUND,
    ClientCapabilities,
    DiscoverResult,
    ResourcesCapability,
    RootsCapability,
    SamplingCapability,
    ServerCapabilities,
    ToolsCapability,
)
from mcp_server_opensearch.mcp_handlers import (
    _gate_handshake,
    build_handlers,
    client_apps_support,
    configure_apps,
)
from tools.tool_annotations import read_only_annotations, write_annotations
from tools.tool_params import SearchIndexArgs
from tools.tool_params import validate_args_for_mode as real_validate_args_for_mode
from tools.ui_resources import (
    MAPPING_VIEW_URI,
    RESULTS_VIEW_URI,
    STRUCTURED_CONTENT_MAX_BYTES,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


SEARCH_RESPONSE = {
    'took': 1,
    'hits': {
        'total': {'value': 1, 'relation': 'eq'},
        'hits': [{'_index': 'logs', '_id': '1', '_score': 1.0, '_source': {'a': 1}}],
    },
}
UI_CLIENT = [advertise(EXTENSION_ID, {'mimeTypes': [APP_MIME_TYPE]})]
UNKNOWN = SimpleNamespace(session=SimpleNamespace(client_capabilities=None), request=None)


def _tool(function, uri=RESULTS_VIEW_URI, kind='search_hits', category='core_tools', **extra):
    info = {
        'display_name': None,
        'description': 'd',
        'input_schema': {'type': 'object', 'properties': {'index': {'type': 'string'}}},
        'args_model': Mock(),
        'function': function,
        'category': category,
        'annotations': read_only_annotations('t'),
        **extra,
    }
    if uri:
        info.update(ui_resource_uri=uri, ui_kind=kind)
    return info


def _registry():
    search = AsyncMock(
        return_value=[{'type': 'text', 'text': 'hits', 'view_data': SEARCH_RESPONSE}]
    )
    plain = AsyncMock(return_value=[{'type': 'text', 'text': 'indices'}])
    tools = {
        'SearchIndexTool': _tool(search, ui_app_callable=True),
        'ListIndexTool': _tool(plain, uri=None),
    }
    for name, info in tools.items():
        info['display_name'] = name
    return tools


def _server(tools):
    server = Server('opensearch-mcp-server', **build_handlers(tools))
    configure_apps(server)
    return server


def _wire(result):
    dumped = result.model_dump(by_alias=True, exclude_none=True, mode='json')
    return {k: v for k, v in dumped.items() if k in ('content', 'isError', 'structuredContent')}


@pytest.fixture
def apps_on(monkeypatch):
    monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')


@pytest.fixture
def apps_off(monkeypatch):
    monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)


@pytest.fixture(autouse=True)
def single_mode(monkeypatch):
    from mcp_server_opensearch import global_state

    # Set the mode itself: patching get_mode leaks a mock into modules first imported here.
    monkeypatch.setattr(global_state, '_current_mode', 'single')
    with (
        patch('tools.tool_params.get_mode', return_value='single'),
        patch('opensearch.client.is_serverless_connection', return_value=False),
        patch(
            'tools.tool_params.validate_args_for_mode',
            side_effect=lambda a, m: SimpleNamespace(**a),
        ),
    ):
        yield


class TestClientSupport:
    def _ctx(self, caps):
        return SimpleNamespace(session=SimpleNamespace(client_capabilities=caps))

    def test_unknown_when_capabilities_missing(self):
        assert client_apps_support(self._ctx(None)) is None

    def test_false_when_known_without_ui(self):
        assert client_apps_support(self._ctx(ClientCapabilities())) is False
        no_mime = ClientCapabilities(extensions={EXTENSION_ID: {}})
        assert client_apps_support(self._ctx(no_mime)) is False

    def test_true_when_ui_advertised(self):
        caps = ClientCapabilities(extensions={EXTENSION_ID: {'mimeTypes': [APP_MIME_TYPE]}})
        assert client_apps_support(self._ctx(caps)) is True


class TestHandshakeGate:
    RESULT = {
        'protocolVersion': '2025-11-25',
        'capabilities': {
            'tools': {'listChanged': False},
            'resources': {'subscribe': False, 'listChanged': False},
            'extensions': {EXTENSION_ID: {}, 'other.example/ext': {}},
        },
    }

    async def _run(self, method, params, caps=None):
        ctx = SimpleNamespace(
            method=method, params=params, session=SimpleNamespace(client_capabilities=caps)
        )
        return await _gate_handshake(ctx, AsyncMock(return_value=self.RESULT))

    @pytest.mark.asyncio
    async def test_legacy_initialize_without_ui_gets_flag_off_capabilities(self):
        result = await self._run('initialize', {'capabilities': {}})
        assert result['capabilities'] == {
            'tools': {'listChanged': False},
            'extensions': {'other.example/ext': {}},
        }

    @pytest.mark.asyncio
    async def test_legacy_initialize_with_ui_unchanged(self):
        ui = {'extensions': {EXTENSION_ID: {'mimeTypes': [APP_MIME_TYPE]}}}
        assert await self._run('initialize', {'capabilities': ui}) == self.RESULT

    @pytest.mark.asyncio
    async def test_discover_with_unknown_capabilities_unchanged(self):
        assert await self._run('server/discover', {}) == self.RESULT

    @pytest.mark.asyncio
    async def test_discover_with_known_non_ui_capabilities_gets_flag_off(self):
        caps = ClientCapabilities(
            roots=RootsCapability(list_changed=True),
            sampling=SamplingCapability(),
            extensions={'other.example/ext': {}},
        )
        result = await self._run('server/discover', {}, caps)
        assert result['capabilities'] == {
            'tools': {'listChanged': False},
            'extensions': {'other.example/ext': {}},
        }

    @pytest.mark.asyncio
    async def test_base_model_result_is_normalized_then_gated(self):
        model = DiscoverResult(
            supported_versions=['2026-07-28'],
            capabilities=ServerCapabilities(
                tools=ToolsCapability(list_changed=False),
                resources=ResourcesCapability(subscribe=False, list_changed=False),
                extensions={EXTENSION_ID: {}},
            ),
        )
        ctx = SimpleNamespace(
            method='server/discover',
            params={},
            session=SimpleNamespace(client_capabilities=ClientCapabilities()),
        )
        dumped = model.model_dump(by_alias=True, mode='json', exclude_none=True)
        result = await _gate_handshake(ctx, AsyncMock(return_value=model))
        assert result == {**dumped, 'capabilities': {'tools': {'listChanged': False}}}
        assert result['supportedVersions'] == ['2026-07-28']
        ui = ClientCapabilities(extensions={EXTENSION_ID: {'mimeTypes': [APP_MIME_TYPE]}})
        ctx.session.client_capabilities = ui
        assert await _gate_handshake(ctx, AsyncMock(return_value=model)) == dumped

    @pytest.mark.asyncio
    async def test_unexpected_result_type_logs_warning_and_passes_through(self, caplog):
        caplog.set_level(logging.WARNING, logger='mcp_server_opensearch.mcp_handlers')
        ctx = SimpleNamespace(method='initialize', params={'capabilities': {}}, session=None)
        result = await _gate_handshake(ctx, AsyncMock(return_value=['unexpected']))
        assert result == ['unexpected']
        assert [r.getMessage() for r in caplog.records] == [
            'MCP Apps gating skipped for initialize: unexpected result type list'
        ]

    @pytest.mark.asyncio
    async def test_other_methods_untouched(self):
        assert await self._run('tools/list', {}, ClientCapabilities()) == self.RESULT


class TestFlagOff:
    def test_capabilities_unchanged(self, apps_off):
        caps = _server(_registry()).get_capabilities()
        assert caps.extensions is None
        assert caps.resources is None

    @pytest.mark.asyncio
    async def test_tools_list_unchanged(self, apps_off):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            tools = (await client.list_tools()).tools
        assert [t.meta for t in tools] == [{'category': 'core_tools'}, {'category': 'core_tools'}]
        assert all('intent' not in t.input_schema['properties'] for t in tools)

    @pytest.mark.asyncio
    async def test_resources_not_served(self, apps_off):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            with pytest.raises(MCPError):
                await client.list_resources()

    @pytest.mark.asyncio
    async def test_call_tool_has_no_structured_content(self, apps_off):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.structured_content is None
        assert _wire(result) == {
            'content': [{'type': 'text', 'text': 'hits'}],
            'isError': False,
        }


class TestSharedHandlerParity:
    """The shared handler, flag off and flag on with a non-Apps client, matches upstream."""

    @pytest.fixture(params=['off', 'gated'])
    def mode(self, request, monkeypatch):
        if request.param == 'off':
            monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        else:
            monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        return request.param

    @pytest.mark.asyncio
    async def test_tool_exception_becomes_error_text(self, mode):
        tools = _registry()
        tools['SearchIndexTool']['function'].side_effect = RuntimeError('boom')
        async with Client(_server(tools)) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert _wire(result) == {'content': [{'type': 'text', 'text': 'boom'}], 'isError': True}

    @pytest.mark.asyncio
    async def test_mcp_error_propagates(self, mode):
        tools = _registry()
        tools['SearchIndexTool']['function'].side_effect = MCPError(code=-32001, message='nope')
        async with Client(_server(tools)) as client:
            with pytest.raises(MCPError, match='nope'):
                await client.call_tool('SearchIndexTool', {})

    @pytest.mark.asyncio
    async def test_custom_display_name(self, mode):
        tools = _registry()
        tools['SearchIndexTool']['display_name'] = 'FindDocs'
        async with Client(_server(tools)) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            result = await client.call_tool('FindDocs', {'index': 'logs'})
        assert names == ['FindDocs', 'ListIndexTool']
        assert _wire(result) == {'content': [{'type': 'text', 'text': 'hits'}], 'isError': False}

    @pytest.mark.asyncio
    async def test_unknown_tool(self, mode):
        async with Client(_server(_registry())) as client:
            result = await client.call_tool('NoSuchTool', {})
        assert _wire(result) == {
            'content': [{'type': 'text', 'text': 'Unknown or disabled tool: NoSuchTool'}],
            'isError': True,
        }

    @pytest.mark.asyncio
    async def test_tools_list_and_handshake_match_flag_off(self, mode, monkeypatch):
        async with Client(_server(_registry())) as client:
            tools = [t.model_dump() for t in (await client.list_tools()).tools]
            caps = client.server_capabilities.model_dump(exclude_none=True)
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        async with Client(_server(_registry())) as client:
            assert tools == [t.model_dump() for t in (await client.list_tools()).tools]
            assert caps == client.server_capabilities.model_dump(exclude_none=True)

    @pytest.mark.asyncio
    async def test_resources_method_not_found(self, mode):
        async with Client(_server(_registry())) as client:
            with pytest.raises(MCPError) as err:
                await client.list_resources()
            assert err.value.error.code == METHOD_NOT_FOUND
            with pytest.raises(MCPError) as err:
                await client.read_resource(RESULTS_VIEW_URI)
            assert err.value.error.code == METHOD_NOT_FOUND


class TestFlagOnUiClient:
    def test_extension_advertised(self, apps_on):
        caps = _server(_registry()).get_capabilities()
        assert caps.extensions == {EXTENSION_ID: {}}
        assert caps.resources is not None

    @pytest.mark.asyncio
    async def test_handshake_advertises_extension(self, apps_on):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            caps = client.server_capabilities
        assert caps.extensions == {EXTENSION_ID: {}}
        assert caps.resources is not None

    @pytest.mark.asyncio
    async def test_ui_meta_and_visibility(self, apps_on):
        tools = _registry()
        extra = {
            'DeleteIndexTool': _tool(
                AsyncMock(),
                uri=None,
                annotations=write_annotations('t', destructive=True, idempotent=False),
            ),
            'GenericOpenSearchApiTool': _tool(AsyncMock(), ui_app_callable=True),
            'WriteViewTool': _tool(
                AsyncMock(),
                ui_app_callable=True,
                annotations=write_annotations('t', destructive=False, idempotent=False),
            ),
            'StaticViewTool': _tool(AsyncMock(), uri=MAPPING_VIEW_URI, kind='index_mapping'),
        }
        for name, info in extra.items():
            info['display_name'] = name
        tools.update(extra)
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
        assert listed['SearchIndexTool'].meta == {
            'category': 'core_tools',
            'ui': {'resourceUri': RESULTS_VIEW_URI, 'visibility': ['model', 'app']},
        }
        model_only = ['model']
        assert listed['ListIndexTool'].meta['ui'] == {'visibility': model_only}
        assert listed['DeleteIndexTool'].meta['ui'] == {'visibility': model_only}
        assert listed['GenericOpenSearchApiTool'].meta['ui']['visibility'] == model_only
        assert listed['WriteViewTool'].meta['ui']['visibility'] == model_only
        assert listed['StaticViewTool'].meta['ui'] == {
            'resourceUri': MAPPING_VIEW_URI,
            'visibility': model_only,
        }

    @pytest.mark.asyncio
    async def test_presentation_args_only_on_query_views(self, apps_on):
        tools = _registry()
        tools['IndexMappingTool'] = _tool(AsyncMock(), uri=MAPPING_VIEW_URI, kind='index_mapping')
        tools['IndexMappingTool']['display_name'] = 'IndexMappingTool'
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
        search = listed['SearchIndexTool'].input_schema['properties']
        assert search['intent']['type'] == 'string'
        assert 'narrative' not in search
        assert search['suggestions']['maxItems'] == 3
        assert 'intent' not in listed['ListIndexTool'].input_schema['properties']
        assert 'intent' not in listed['IndexMappingTool'].input_schema['properties']

    @pytest.mark.asyncio
    async def test_presentation_args_on_renamed_view_tool(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['display_name'] = 'FindDocs'
        args = {'index': 'logs', 'intent': 'Find the 500s.', 'suggestions': ['Group by host']}
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
            result = await client.call_tool('FindDocs', args)
        assert 'intent' in listed['FindDocs'].input_schema['properties']
        assert vars(tools['SearchIndexTool']['function'].call_args.args[0]) == {'index': 'logs'}
        sc = result.structured_content
        assert sc['tool'] == 'FindDocs'
        assert sc['intent'] == 'Find the 500s.' and sc['suggestions'] == ['Group by host']
        assert result.content[0].text == 'hits'

    @pytest.mark.asyncio
    async def test_real_args_model_after_split(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['args_model'] = SearchIndexArgs
        args = {
            'index': 'logs',
            'query_dsl': {'query': {'match_all': {}}, 'from': 1},
            'size': 5,
            'intent': 'Find the 500s.',
            'suggestions': ['Group by host'],
        }
        with patch('tools.tool_params.validate_args_for_mode', real_validate_args_for_mode):
            async with Client(_server(tools), extensions=UI_CLIENT) as client:
                result = await client.call_tool('SearchIndexTool', args)
        parsed = tools['SearchIndexTool']['function'].call_args.args[0]
        assert isinstance(parsed, SearchIndexArgs)
        assert parsed.model_dump(exclude_none=True) == {
            'opensearch_cluster_name': '',
            'index': 'logs',
            'query_dsl': {'query': {'match_all': {}}, 'from': 1},
            'format': 'json',
            'size': 5,
        }
        assert not hasattr(parsed, 'intent') and not hasattr(parsed, 'suggestions')
        sc = result.structured_content
        assert sc['intent'] == 'Find the 500s.' and sc['suggestions'] == ['Group by host']
        assert sc['meta']['index'] == 'logs' and sc['meta']['from'] == 1

    @pytest.mark.asyncio
    async def test_list_and_read_resource(self, apps_on):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            resources = (await client.list_resources()).resources
            read = await client.read_resource(RESULTS_VIEW_URI)
        assert [str(r.uri) for r in resources] == [RESULTS_VIEW_URI]
        meta = {'ui': {'prefersBorder': True, 'permissions': {'clipboardWrite': {}}}}
        assert resources[0].mime_type == APP_MIME_TYPE
        assert resources[0].meta == meta
        content = read.contents[0]
        assert content.mime_type == APP_MIME_TYPE and content.meta == meta
        assert content.text.startswith('<!DOCTYPE html>')

    @pytest.mark.asyncio
    async def test_filtered_out_tool_hides_resource(self, apps_on):
        tools = _registry()
        del tools['SearchIndexTool']
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            assert (await client.list_resources()).resources == []
            with pytest.raises(MCPError, match='Resource not found'):
                await client.read_resource(RESULTS_VIEW_URI)

    @pytest.mark.asyncio
    async def test_unknown_uri_errors(self, apps_on):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            with pytest.raises(MCPError, match='Resource not found'):
                await client.read_resource('ui://opensearch/unknown')

    @pytest.mark.asyncio
    async def test_call_tool_builds_payload_in_handler(self, apps_on):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {'index': 'logs'})
            plain = await client.call_tool('ListIndexTool', {})
        sc = result.structured_content
        assert sc['kind'] == 'search_hits' and sc['tool'] == 'SearchIndexTool'
        assert sc['refreshable'] is True
        assert sc['rows'] == [['logs', '1', 1.0, 1]]
        assert result.content[0].text == 'hits'
        assert 'view_data' not in str(_wire(result)['content'])
        assert plain.structured_content is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'extra',
        [
            {'intent': 5},
            {'intent': 'x' * 2001},
            {'intent': 'hidden\u202etext'},
            {'intent': 'bell\x07'},
            {'suggestions': 'one'},
            {'suggestions': ['a', 'b', 'c', 'd']},
            {'suggestions': ['']},
            {'suggestions': ['x' * 121]},
            {'suggestions': ['line\nbreak']},
            {'suggestions': ['zero\u200bwidth']},
        ],
    )
    async def test_invalid_presentation_args_rejected(self, apps_on, extra, caplog):
        tools = _registry()
        caplog.set_level(logging.INFO, logger='mcp_server_opensearch.tool_executor')
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {'index': 'logs', **extra})
        assert result.is_error
        assert tools['SearchIndexTool']['function'].call_count == 0
        assert result.structured_content == {
            'kind': 'search_hits',
            'version': 1,
            'omitted': 'error',
            'tool': 'SearchIndexTool',
            'refreshable': True,
        }
        events = [r for r in caplog.records if getattr(r, 'event_type', '') == 'tool_execution']
        assert [(e.status, e.error_type) for e in events] == [('error', 'ValidationError')]

    @pytest.mark.asyncio
    async def test_intent_allows_newlines_and_tabs(self, apps_on):
        async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {'intent': 'a\n\tb'})
        assert result.structured_content['intent'] == 'a\n\tb'

    @pytest.mark.asyncio
    async def test_soft_error_marker(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 'boom', 'is_error': True}
        ]
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.is_error
        assert result.structured_content['omitted'] == 'error'

    @pytest.mark.asyncio
    async def test_unsupported_format_marker(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['function'].return_value = [{'type': 'text', 'text': 'a|b'}]
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {})
        assert result.structured_content['omitted'] == 'unsupported_format'

    @pytest.mark.asyncio
    async def test_builder_failure_keeps_query_result(self, apps_on):
        with patch.dict(
            'tools.ui_resources.VIEW_BUILDERS', {'search_hits': Mock(side_effect=KeyError('x'))}
        ):
            async with Client(_server(_registry()), extensions=UI_CLIENT) as client:
                result = await client.call_tool('SearchIndexTool', {'intent': 'why'})
        assert not result.is_error
        assert result.content[0].text == 'hits'
        assert result.structured_content['omitted'] == 'unsupported_format'
        assert result.structured_content['intent'] == 'why'

    @pytest.mark.asyncio
    async def test_oversized_rows_are_halved_until_they_fit(self, apps_on):
        tools = _registry()
        hits = [{'_source': {'a': 'x' * 300_000}} for _ in range(8)]
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 't', 'view_data': {'hits': {'hits': hits}}}
        ]
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {})
        sc = result.structured_content
        assert len(sc['rows']) == 2
        assert sc['meta']['truncated'] is True and sc['meta']['row_count'] == 8

    @pytest.mark.asyncio
    async def test_aggregation_dominated_result_is_halved(self, apps_on):
        tools = _registry()
        aggs = {'big': {'buckets': [{'key': 'k' * STRUCTURED_CONTENT_MAX_BYTES, 'doc_count': 1}]}}
        big = {'hits': {'hits': [{'_source': {'a': 1}}]}, 'aggregations': aggs}
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 't', 'view_data': big}
        ]
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {'intent': 'n'})
        sc = result.structured_content
        assert 'omitted' not in sc and sc['rows'] == [[None, None, None, 1]]
        table = sc['meta']['aggregations'][0]
        assert table['rows'] == [] and table['truncated'] is True and table['row_count'] == 1
        assert 'truncated' not in sc['meta']

    @pytest.mark.asyncio
    async def test_too_large_marker(self, apps_on):
        tools = _registry()
        big = {'hits': {'hits': [{'_source': {'k' * STRUCTURED_CONTENT_MAX_BYTES: 1}}]}}
        tools['SearchIndexTool']['function'].return_value = [
            {'type': 'text', 'text': 't', 'view_data': big}
        ]
        async with Client(_server(tools), extensions=UI_CLIENT) as client:
            result = await client.call_tool('SearchIndexTool', {'intent': 'n'})
        sc = result.structured_content
        assert sc['omitted'] == 'too_large' and sc['bytes'] > STRUCTURED_CONTENT_MAX_BYTES
        assert sc['intent'] == 'n' and sc['tool'] == 'SearchIndexTool'
        assert 'rows' not in sc
        assert result.content[0].text == 't'

    def test_malformed_dashboards_url_fails_at_startup(self, apps_on, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'dashboards.local')
        with pytest.raises(ValueError, match='OPENSEARCH_DASHBOARDS_URL'):
            build_handlers(_registry())


class TestFlagOnUnknownCapabilities:
    """Stateless streamable HTTP: capabilities unknown, so Apps behave as on."""

    @pytest.mark.asyncio
    async def test_tools_list_has_ui(self, apps_on):
        handlers = build_handlers(_registry())
        listed = {t.name: t for t in (await handlers['on_list_tools'](UNKNOWN, None)).tools}
        assert listed['SearchIndexTool'].meta['ui']['resourceUri'] == RESULTS_VIEW_URI
        assert 'intent' in listed['SearchIndexTool'].input_schema['properties']

    @pytest.mark.asyncio
    async def test_call_has_structured_content(self, apps_on):
        handlers = build_handlers(_registry())
        params = SimpleNamespace(name='SearchIndexTool', arguments={'intent': 'why'})
        result = await handlers['on_call_tool'](UNKNOWN, params)
        assert result.structured_content['kind'] == 'search_hits'
        assert result.structured_content['intent'] == 'why'

    @pytest.mark.asyncio
    async def test_error_marker_from_handler_exception(self, apps_on):
        tools = _registry()
        tools['SearchIndexTool']['function'].side_effect = RuntimeError('boom')
        handlers = build_handlers(tools)
        params = SimpleNamespace(name='SearchIndexTool', arguments={})
        result = await handlers['on_call_tool'](UNKNOWN, params)
        assert result.is_error and result.content[0].text == 'boom'
        assert result.structured_content['omitted'] == 'error'

    @pytest.mark.asyncio
    async def test_resources_listed(self, apps_on):
        handlers = build_handlers(_registry())
        listed = await handlers['on_list_resources'](UNKNOWN, None)
        assert [str(r.uri) for r in listed.resources] == [RESULTS_VIEW_URI]


class TestTransports:
    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.streaming_server.get_tools')
    @patch('mcp_server_opensearch.streaming_server.generate_tools_from_openapi')
    async def test_streaming_server_apps(self, mock_generate, mock_get_tools, apps_on):
        from mcp_server_opensearch.streaming_server import create_mcp_server

        mock_get_tools.return_value = _registry()
        server = await create_mcp_server()
        assert server.get_capabilities().extensions == {EXTENSION_ID: {}}
        async with Client(server, extensions=UI_CLIENT) as client:
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
