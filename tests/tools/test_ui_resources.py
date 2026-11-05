# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import re
from tools.ui_resources import (
    RESULTS_VIEW_URI,
    STRUCTURED_CONTENT_MAX_BYTES,
    STRUCTURED_CONTENT_MAX_COLUMNS,
    STRUCTURED_CONTENT_MAX_ROWS,
    UI_RESOURCES,
    apps_enabled,
    bound_resource_uris,
    cap_structured_content,
    dashboards_base_url,
    discover_ppl_url,
    finalize_view_payload,
    load_ui_html,
    ppl_rows_payload,
    search_hits_payload,
    split_presentation_args,
    tool_ui_meta,
    view_input_schema,
)
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import unquote


SEARCH_RESPONSE = {
    'took': 3,
    'timed_out': False,
    '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0},
    'hits': {
        'total': {'value': 2, 'relation': 'eq'},
        'max_score': 1.0,
        'hits': [
            {'_index': 'logs', '_id': '1', '_score': 1.0, '_source': {'a': 1, 'b': 'x'}},
            {'_index': 'logs', '_id': '2', '_score': 0.5, '_source': {'b': 'y', 'c': {'d': 2}}},
        ],
    },
}

PPL_RESPONSE = {
    'schema': [{'name': 'count()', 'type': 'bigint'}, {'name': 'host', 'type': 'string'}],
    'datarows': [[2, 'h1'], [1, 'h2']],
    'total': 2,
    'size': 2,
    'status': 200,
}


class TestFlag:
    @pytest.mark.parametrize(
        'value,expected', [('true', True), ('TRUE', True), ('false', False), ('', False)]
    )
    def test_apps_enabled(self, monkeypatch, value, expected):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', value)
        assert apps_enabled() is expected

    def test_apps_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        assert apps_enabled() is False


class TestPayloads:
    def test_search_hits_shape(self):
        payload = search_hits_payload(SEARCH_RESPONSE, 'c1', 'logs')
        assert payload['kind'] == 'search_hits'
        assert payload['version'] == 1
        assert payload['cluster'] == 'c1'
        assert payload['columns'] == ['_index', '_id', '_score', 'a', 'b', 'c']
        assert payload['rows'] == [
            ['logs', '1', 1.0, 1, 'x', None],
            ['logs', '2', 0.5, None, 'y', {'d': 2}],
        ]
        meta = payload['meta']
        assert meta['total'] == 2 and meta['total_relation'] == 'eq'
        assert meta['took'] == 3 and meta['timed_out'] is False
        assert meta['_shards']['successful'] == 1
        assert 'aggregations' not in meta

    def test_search_hits_aggregations_as_tables(self):
        response = {
            **SEARCH_RESPONSE,
            'aggregations': {
                'by_service': {
                    'buckets': [
                        {'key': 'auth', 'doc_count': 2, 'avg_latency': {'value': 11.5}},
                        {'key': 'search', 'doc_count': 1, 'avg_latency': {'value': 410.0}},
                    ]
                },
                'by_filter': {'buckets': {'errors': {'doc_count': 3}}},
                'max_latency': {'value': 950.0},
                'latency_stats': {'count': 3, 'min': 1, 'max': 9, 'avg': 5.0, 'sum': 15},
            },
        }
        tables = {
            t['name']: t for t in search_hits_payload(response, '', 'logs')['meta']['aggregations']
        }
        assert tables['by_service']['columns'] == ['key', 'doc_count', 'avg_latency']
        assert tables['by_service']['rows'] == [['auth', 2, 11.5], ['search', 1, 410.0]]
        assert tables['by_filter']['rows'] == [['errors', 3]]
        assert tables['max_latency'] == {
            'name': 'max_latency',
            'columns': ['value'],
            'rows': [[950.0]],
        }
        assert tables['latency_stats']['columns'] == ['count', 'min', 'max', 'avg', 'sum']

    def test_search_hits_total_as_int(self):
        response = {'hits': {'total': 5, 'hits': []}}
        payload = search_hits_payload(response, '', 'i')
        assert payload['meta']['total'] == 5 and payload['rows'] == []

    @pytest.mark.parametrize('response', ['text', None, {'error': 'x'}, {'hits': []}])
    def test_search_hits_non_search_response(self, response):
        assert search_hits_payload(response, '', 'i') is None

    def test_ppl_rows_shape(self):
        payload = ppl_rows_payload(PPL_RESPONSE, 'c1', 'source=x | stats count() by host')
        assert payload['kind'] == 'ppl_rows'
        assert payload['columns'] == ['count()', 'host']
        assert payload['rows'] == [[2, 'h1'], [1, 'h2']]
        assert payload['meta']['column_types'] == ['bigint', 'string']
        assert payload['meta']['total'] == 2

    @pytest.mark.parametrize('response', ['a|b\n1|2', None, {'schema': []}])
    def test_ppl_rows_non_jdbc_response(self, response):
        assert ppl_rows_payload(response, '', 'q') is None

    def test_cap_keeps_small_payload(self):
        payload = search_hits_payload(SEARCH_RESPONSE, '', 'logs')
        assert cap_structured_content(payload) == payload

    def test_cap_replaces_payload_over_limit_with_marker(self):
        big = {
            'kind': 'search_hits',
            'version': 1,
            'columns': ['a'],
            'rows': [['x' * STRUCTURED_CONTENT_MAX_BYTES]],
            'tool': 'SearchIndexTool',
        }
        marker = cap_structured_content(big)
        assert marker['omitted'] == 'too_large'
        assert marker['kind'] == 'search_hits' and marker['version'] == 1
        assert marker['bytes'] > STRUCTURED_CONTENT_MAX_BYTES
        assert marker['tool'] == 'SearchIndexTool'
        assert 'rows' not in marker

    def test_cap_truncates_rows_and_columns(self):
        columns = [f'c{i}' for i in range(120)]
        rows = [list(range(120)) for _ in range(600)]
        payload = {
            'kind': 'ppl_rows',
            'version': 1,
            'columns': columns,
            'rows': rows,
            'meta': {'column_types': ['int'] * 120},
        }
        capped = cap_structured_content(payload)
        assert len(capped['rows']) == STRUCTURED_CONTENT_MAX_ROWS
        assert len(capped['columns']) == STRUCTURED_CONTENT_MAX_COLUMNS
        assert all(len(r) == STRUCTURED_CONTENT_MAX_COLUMNS for r in capped['rows'])
        assert capped['meta'] == {
            'column_types': ['int'] * 100,
            'truncated': True,
            'row_count': 600,
            'column_count': 120,
        }
        assert len(payload['rows']) == 600  # input not mutated

    def test_cap_truncates_aggregation_tables(self):
        payload = search_hits_payload(SEARCH_RESPONSE, '', 'logs')
        payload['meta']['aggregations'] = [
            {'name': 'big', 'columns': ['key', 'doc_count'], 'rows': [['k', 1]] * 501}
        ]
        capped = cap_structured_content(payload)
        table = capped['meta']['aggregations'][0]
        assert len(table['rows']) == 500 and table['truncated'] is True
        assert 'truncated' not in capped['meta']

    def test_cap_under_limits_not_marked(self):
        payload = ppl_rows_payload(PPL_RESPONSE, '', 'q')
        assert 'truncated' not in cap_structured_content(payload)['meta']


class TestViewPayload:
    def test_finalize_adds_tool_and_presentation(self):
        payload = finalize_view_payload(
            ppl_rows_payload(PPL_RESPONSE, '', 'q'), 'ppl_rows', 'PPL', {'narrative': 'n'}, False
        )
        assert payload['tool'] == 'PPL' and payload['narrative'] == 'n'
        assert payload['rows'] == PPL_RESPONSE['datarows']

    @pytest.mark.parametrize(
        'structured,is_error,reason',
        [
            (None, False, 'unsupported_format'),
            (None, True, 'error'),
            ({'rows': []}, True, 'error'),
        ],
    )
    def test_finalize_markers(self, structured, is_error, reason):
        payload = finalize_view_payload(structured, 'ppl_rows', 'PPLQueryTool', {}, is_error)
        assert payload == {
            'kind': 'ppl_rows',
            'version': 1,
            'omitted': reason,
            'tool': 'PPLQueryTool',
        }

    def test_split_presentation_args(self):
        args, presentation = split_presentation_args(
            {'index': 'i', 'narrative': 'n', 'suggestions': ['a', 'b']}
        )
        assert args == {'index': 'i'}
        assert presentation == {'narrative': 'n', 'suggestions': ['a', 'b']}
        assert split_presentation_args({'index': 'i', 'suggestions': []}) == ({'index': 'i'}, {})

    def test_view_input_schema_does_not_mutate(self):
        schema = {'type': 'object', 'properties': {'q': {'type': 'string'}}}
        extended = view_input_schema(schema)
        assert set(extended['properties']) == {'q', 'narrative', 'suggestions'}
        assert set(schema['properties']) == {'q'}


class TestDashboardsLink:
    @pytest.fixture(autouse=True)
    def single(self):
        with patch('mcp_server_opensearch.global_state.get_mode', return_value='single'):
            yield

    def test_no_link_without_env(self, monkeypatch):
        monkeypatch.delenv('OPENSEARCH_DASHBOARDS_URL', raising=False)
        assert discover_ppl_url('source=logs') is None
        assert 'dashboards_url' not in ppl_rows_payload(PPL_RESPONSE, '', 'source=logs')['meta']

    def test_ppl_link(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'https://osd.example.com/')
        url = discover_ppl_url("source=logs | where msg = 'it!s'")
        assert url.startswith('https://osd.example.com/app/data-explorer/discover#?_q=')
        state = unquote(url.split('_q=', 1)[1])
        assert state == (
            "(filters:!(),query:(dataset:(id:'::logs',title:'logs',type:INDEXES),"
            "language:PPL,query:'source=logs | where msg = !'it!!s!''))"
        )

    def test_payloads_carry_link(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'http://127.0.0.1:5601')
        search = search_hits_payload(SEARCH_RESPONSE, '', 'logs')
        assert "query:'source = logs'" in unquote(search['meta']['dashboards_url'])
        ppl = ppl_rows_payload(PPL_RESPONSE, '', 'source=logs | head 1')
        assert 'dashboards_url' in ppl['meta']

    def test_no_link_without_source(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'http://127.0.0.1:5601')
        assert discover_ppl_url('describe logs') is None

    def test_no_link_in_multi_mode(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'http://127.0.0.1:5601')
        with patch('mcp_server_opensearch.global_state.get_mode', return_value='multi'):
            assert discover_ppl_url('source=logs') is None

    @pytest.mark.parametrize('value', ['osd.example.com', 'javascript:alert(1)', 'file:///x'])
    def test_malformed_url_rejected(self, monkeypatch, value):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', value)
        with pytest.raises(ValueError, match='OPENSEARCH_DASHBOARDS_URL'):
            dashboards_base_url()


class TestToolVisibility:
    def test_registry_visibility(self):
        from tools.tools import TOOL_REGISTRY

        default = sorted(
            k for k, v in TOOL_REGISTRY.items() if 'visibility' not in tool_ui_meta(k, v)
        )
        assert default == ['PPLQueryTool', 'SearchIndexTool']
        assert tool_ui_meta(
            'GenericOpenSearchApiTool', TOOL_REGISTRY['GenericOpenSearchApiTool']
        ) == {'visibility': ['model']}


class TestRegistry:
    def test_results_view_loads_from_package(self):
        html = load_ui_html(RESULTS_VIEW_URI)
        assert html.startswith('<!DOCTYPE html>')
        assert 'ui/initialize' in html

    def test_results_view_is_self_contained_and_safe(self):
        html = load_ui_html(RESULTS_VIEW_URI)
        assert '<script src' not in html and '<link' not in html
        assert '@import' not in html and 'url(' not in html
        assert 'fetch(' not in html and 'XMLHttpRequest' not in html
        urls = re.findall(r'https?://[^\s\'"]*', html)
        assert urls == ['http://www.w3.org/2000/svg']  # SVG namespace, not a load

    @pytest.mark.parametrize('name', sorted({v['html_file'] for v in UI_RESOURCES.values()}))
    def test_view_has_no_html_injection_sinks(self, name):
        from importlib import resources

        html = resources.files('tools.ui').joinpath(name).read_text(encoding='utf-8')
        sinks = re.findall(
            r'innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\s*\(|new\s+Function\s*\(',
            html,
        )
        assert sinks == [], f'{name} uses {sinks}; build DOM nodes and set textContent instead'

    def test_every_ui_file_is_registered(self):
        from importlib import resources

        files = {p.name for p in resources.files('tools.ui').iterdir() if p.name.endswith('.html')}
        assert files == {v['html_file'] for v in UI_RESOURCES.values()}

    def test_every_registered_resource_loads(self):
        for uri in UI_RESOURCES:
            assert uri.startswith('ui://')
            assert load_ui_html(uri)

    def test_bound_resource_uris_only_for_enabled_tools(self):
        tools = {
            'A': {'ui_resource_uri': RESULTS_VIEW_URI},
            'B': {'ui_resource_uri': RESULTS_VIEW_URI},
            'C': {},
            'D': {'ui_resource_uri': 'ui://unknown/view'},
        }
        assert bound_resource_uris(tools) == [RESULTS_VIEW_URI]
        assert bound_resource_uris({'C': {}}) == []

    def test_registry_binds_search_and_ppl(self):
        from tools.tools import TOOL_REGISTRY

        bound = sorted(k for k, v in TOOL_REGISTRY.items() if v.get('ui_resource_uri'))
        assert bound == ['PPLQueryTool', 'SearchIndexTool']


class TestToolStructuredResults:
    def setup_method(self):
        self.mock_client = Mock()
        self.mock_client.search = AsyncMock(return_value=SEARCH_RESPONSE)
        self.mock_client.transport.perform_request = AsyncMock(return_value=PPL_RESPONSE)
        self.mock_client.info = AsyncMock(return_value={'version': {'number': '2.19.0'}})
        self.patcher = patch('opensearch.client.initialize_client', return_value=self.mock_client)
        self.patcher.start()
        import sys

        for module in ['tools.tools', 'opensearch.helper']:
            sys.modules.pop(module, None)
        from tools.tools import PplQueryArgs, SearchIndexArgs, ppl_query_tool, search_index_tool

        self.search = search_index_tool
        self.ppl = ppl_query_tool
        self.SearchIndexArgs = SearchIndexArgs
        self.PplQueryArgs = PplQueryArgs

    def teardown_method(self):
        self.patcher.stop()

    def _search_args(self, **kw):
        return self.SearchIndexArgs(
            index='logs', query_dsl={'match_all': {}}, opensearch_cluster_name='c1', **kw
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('fmt', ['json', 'csv'])
    async def test_search_text_unchanged_and_structured_when_enabled(self, monkeypatch, fmt):
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        off = await self.search(self._search_args(format=fmt))
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        on = await self.search(self._search_args(format=fmt))
        assert 'structured' not in off[0]
        assert on[0]['text'] == off[0]['text']
        assert on[0]['structured']['kind'] == 'search_hits'
        assert on[0]['structured']['cluster'] == 'c1'

    @pytest.mark.asyncio
    async def test_ppl_text_unchanged_and_structured_when_enabled(self, monkeypatch):
        args = self.PplQueryArgs(opensearch_cluster_name='c1', query='source=x')
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        off = await self.ppl(args)
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        on = await self.ppl(args)
        assert 'structured' not in off[0]
        assert on[0]['text'] == off[0]['text']
        assert on[0]['structured']['kind'] == 'ppl_rows'
        assert on[0]['structured']['meta']['query'] == 'source=x'

    @pytest.mark.asyncio
    async def test_ppl_text_format_has_no_structured(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        self.mock_client.transport.perform_request.return_value = 'count()|host\n2|h1'
        args = self.PplQueryArgs(opensearch_cluster_name='', query='source=x', format='raw')
        result = await self.ppl(args)
        assert result[0]['structured'] is None

    @pytest.mark.asyncio
    async def test_search_error_has_no_structured(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        self.mock_client.search.side_effect = Exception('boom')
        result = await self.search(self._search_args())
        assert result[0].get('is_error') is True
        assert 'structured' not in result[0]

    @pytest.mark.asyncio
    async def test_payload_json_serializable(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        result = await self.search(self._search_args())
        json.dumps(result[0]['structured'])
