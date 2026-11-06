# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import re
from importlib import resources
from tools.ui_resources import (
    EXPLAIN_VIEW_URI,
    HEALTH_VIEW_URI,
    MAPPING_VIEW_URI,
    RESULTS_VIEW_URI,
    STRUCTURED_CONTENT_MAX_BYTES,
    STRUCTURED_CONTENT_MAX_COLUMNS,
    STRUCTURED_CONTENT_MAX_DEPTH,
    STRUCTURED_CONTENT_MAX_NODES,
    STRUCTURED_CONTENT_MAX_ROWS,
    TABLE_VIEW_URI,
    UI_RESOURCES,
    VIEW_BUILDERS,
    apps_enabled,
    bound_resource_uris,
    cap_structured_content,
    cat_table_payload,
    cluster_health_payload,
    dashboards_base_url,
    discover_ppl_url,
    explain_payload,
    finalize_view_payload,
    index_mapping_payload,
    list_indices_payload,
    load_ui_html,
    ppl_rows_payload,
    resource_meta,
    search_hits_payload,
    split_presentation_args,
    tool_ui_meta,
    view_input_schema,
    view_structured_content,
)
from types import SimpleNamespace
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

CAT_INDICES = [
    {
        'health': 'yellow',
        'status': 'open',
        'index': 'logs',
        'pri': '1',
        'docs.count': '6',
        'store.size': '12.5kb',
        'pri.store.size': '12.5kb',
    },
    {
        'health': 'green',
        'status': 'open',
        'index': 'ts',
        'pri': '1',
        'docs.count': '33',
        'store.size': '1.1mb',
        'pri.store.size': '1.1mb',
    },
]

HEALTH = {
    'cluster_name': 'docker-cluster',
    'status': 'yellow',
    'timed_out': False,
    'number_of_nodes': 1,
    'number_of_data_nodes': 1,
    'active_primary_shards': 5,
    'active_shards': 5,
    'relocating_shards': 0,
    'initializing_shards': 0,
    'unassigned_shards': 4,
    'delayed_unassigned_shards': 0,
    'number_of_pending_tasks': 0,
    'number_of_in_flight_fetch': 0,
    'task_max_waiting_in_queue_millis': 0,
    'active_shards_percent_as_number': 55.55,
    'discovered_cluster_manager': True,
}

MAPPING = {
    'logs': {
        'mappings': {
            'dynamic': 'strict',
            'properties': {
                'message': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}},
                },
                'user': {'properties': {'id': {'type': 'keyword'}}},
                'ts': {'type': 'date', 'format': 'strict_date_optional_time'},
            },
        }
    }
}

EXPLAIN = {
    '_index': 'logs',
    '_id': '1',
    'matched': True,
    'explanation': {
        'value': 1.5,
        'description': 'sum of:',
        'details': [
            {'value': 1.0, 'description': 'weight(message:db in 0)', 'details': []},
            {'value': 0.5, 'description': 'weight(message:timeout in 0)', 'details': []},
        ],
    },
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


class TestSearchAndPplPayloads:
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
        assert meta['from'] == 0
        assert 'aggregations' not in meta

    @pytest.mark.parametrize(
        'query_dsl,expected',
        [
            ({'from': 20, 'query': {}}, 20),
            ('{"from": 10}', 10),
            ('not json', 0),
            ({'from': -1}, 0),
            ({'from': True}, 0),
            (None, 0),
        ],
    )
    def test_search_hits_page_offset(self, query_dsl, expected):
        payload = search_hits_payload(SEARCH_RESPONSE, '', 'logs', query_dsl)
        assert payload['meta']['from'] == expected

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
            },
        }
        tables = {
            t['name']: t for t in search_hits_payload(response, '', 'logs')['meta']['aggregations']
        }
        assert tables['by_service']['columns'] == ['key', 'doc_count', 'avg_latency']
        assert tables['by_service']['rows'] == [['auth', 2, 11.5], ['search', 1, 410.0]]
        assert tables['by_filter']['rows'] == [['errors', 3]]
        assert tables['max_latency']['rows'] == [[950.0]]

    @pytest.mark.parametrize('response', ['text', None, {'error': 'x'}, {'hits': []}])
    def test_search_hits_non_search_response(self, response):
        assert search_hits_payload(response, '', 'i') is None

    def test_ppl_rows_shape(self):
        payload = ppl_rows_payload(PPL_RESPONSE, 'c1', 'source=x | stats count() by host')
        assert payload['kind'] == 'ppl_rows'
        assert payload['columns'] == ['count()', 'host']
        assert payload['rows'] == [[2, 'h1'], [1, 'h2']]
        assert payload['meta']['column_types'] == ['bigint', 'string']

    @pytest.mark.parametrize('response', ['a|b\n1|2', None, {'schema': []}])
    def test_ppl_rows_non_jdbc_response(self, response):
        assert ppl_rows_payload(response, '', 'q') is None


class TestCatTablePayloads:
    def test_types_numbers_and_byte_sizes(self):
        payload = cat_table_payload(CAT_INDICES, 'c1', 'Indices', '_cat/indices')
        assert payload['kind'] == 'cat_table' and payload['cluster'] == 'c1'
        assert payload['meta']['title'] == 'Indices'
        types = dict(zip(payload['columns'], payload['meta']['column_types']))
        assert types == {
            'health': 'string',
            'status': 'string',
            'index': 'string',
            'pri': 'number',
            'docs.count': 'number',
            'store.size': 'bytes',
            'pri.store.size': 'bytes',
        }
        row = dict(zip(payload['columns'], payload['rows'][1]))
        assert row['docs.count'] == 33
        assert row['store.size'] == round(1.1 * 1024 * 1024)
        assert dict(zip(payload['columns'], payload['rows'][0]))['store.size'] == 12800

    def test_mixed_and_missing_values(self):
        rows = [
            {'state': 'STARTED', 'docs': '10', 'store': '1kb', 'node': 'n1', 'load': '1.5'},
            {'state': 'UNASSIGNED', 'docs': None, 'store': None, 'node': None, 'load': '0'},
            {'state': 'STARTED', 'docs': 'n/a', 'store': '2gb', 'extra': 'x', 'load': '2'},
        ]
        payload = cat_table_payload(rows, '', 'Shards', '_cat/shards')
        types = dict(zip(payload['columns'], payload['meta']['column_types']))
        assert types['docs'] == 'string'  # one non-numeric value keeps the raw strings
        assert types['store'] == 'bytes' and types['load'] == 'number'
        assert payload['columns'][-1] == 'extra'
        assert payload['rows'][1][:3] == ['UNASSIGNED', None, None]
        assert payload['rows'][0][4] == 1.5 and payload['rows'][1][4] == 0

    @pytest.mark.parametrize('entries', [None, {'a': 1}, ['x'], 'text'])
    def test_non_table_input(self, entries):
        assert cat_table_payload(entries, '', 't', 's') is None

    def test_list_indices_mirrors_text(self):
        names = list_indices_payload(CAT_INDICES, '', SimpleNamespace(include_detail=False))
        assert names['columns'] == ['index'] and names['rows'] == [['logs'], ['ts']]
        full = list_indices_payload(
            CAT_INDICES, '', SimpleNamespace(include_detail=True, index='')
        )
        assert 'store.size' in full['columns']
        one = SimpleNamespace(include_detail=True, index='logs')
        assert list_indices_payload({'logs': {}}, '', one) is None


class TestTreePayloads:
    def test_cluster_health(self):
        payload = cluster_health_payload(HEALTH, 'c1')
        assert payload['kind'] == 'cluster_health' and payload['cluster'] == 'c1'
        assert payload['health']['unassigned_shards'] == 4
        assert 'discovered_cluster_manager' not in payload['health']
        assert 'rows' not in payload

    def test_cluster_health_indices_level(self):
        response = {**HEALTH, 'indices': {'logs': {'status': 'yellow', 'unassigned_shards': 1}}}
        payload = cluster_health_payload(response, '')
        assert payload['columns'] == ['index', 'status', 'unassigned_shards']
        assert payload['rows'] == [['logs', 'yellow', 1]]

    @pytest.mark.parametrize('response', [None, 'x', {'cluster_name': 'c'}])
    def test_cluster_health_rejects_other_shapes(self, response):
        assert cluster_health_payload(response, '') is None

    def test_index_mapping_tree(self):
        payload = index_mapping_payload(MAPPING, '')
        assert payload['kind'] == 'index_mapping'
        index = payload['indices'][0]
        assert index['name'] == 'logs' and index['dynamic'] == 'strict'
        fields = {f['name']: f for f in index['fields']}
        assert fields['message']['type'] == 'text'
        assert fields['message']['details'] == {'analyzer': 'standard'}
        assert fields['message']['fields'] == [
            {
                'name': 'keyword',
                'path': 'message.keyword',
                'type': 'keyword',
                'details': {'ignore_above': 256},
            }
        ]
        assert fields['user']['type'] == 'object'
        assert fields['user']['children'][0]['path'] == 'user.id'
        assert payload['meta'] == {'field_count': 5}

    def test_index_mapping_node_cap(self):
        props = {f'f{i}': {'type': 'keyword'} for i in range(STRUCTURED_CONTENT_MAX_NODES + 10)}
        payload = index_mapping_payload({'big': {'mappings': {'properties': props}}}, '')
        assert len(payload['indices'][0]['fields']) == STRUCTURED_CONTENT_MAX_NODES
        assert payload['meta'] == {'field_count': STRUCTURED_CONTENT_MAX_NODES, 'truncated': True}

    @pytest.mark.parametrize('response', [None, {}, {'logs': {'settings': {}}}])
    def test_index_mapping_rejects_other_shapes(self, response):
        assert index_mapping_payload(response, '') is None

    def test_explain_tree(self):
        payload = explain_payload(EXPLAIN, 'c1')
        assert payload['kind'] == 'explain' and payload['matched'] is True
        assert payload['index'] == 'logs' and payload['id'] == '1'
        assert payload['explanation']['value'] == 1.5
        assert [d['value'] for d in payload['explanation']['details']] == [1.0, 0.5]
        assert 'details' not in payload['explanation']['details'][0]
        assert payload['meta'] == {'node_count': 3}

    def test_explain_node_and_depth_caps(self):
        deep = {'value': 0, 'description': 'leaf'}
        for _ in range(2000):
            deep = {'value': 1, 'description': 'n', 'details': [deep]}
        payload = explain_payload({'matched': True, 'explanation': deep}, '')
        assert payload['meta'] == {'node_count': STRUCTURED_CONTENT_MAX_DEPTH, 'truncated': True}
        wide = {
            'value': 1,
            'description': 'w',
            'details': [{'value': 0, 'description': 'x'}] * 6000,
        }
        payload = explain_payload({'matched': True, 'explanation': wide}, '')
        assert payload['meta'] == {'node_count': STRUCTURED_CONTENT_MAX_NODES, 'truncated': True}

    def test_generated_tool_text_is_parsed(self):
        args = SimpleNamespace(opensearch_cluster_name='')
        assert VIEW_BUILDERS['cluster_health'](json.dumps(HEALTH), args)['health']
        assert VIEW_BUILDERS['explain'](json.dumps(EXPLAIN), args)['matched'] is True
        assert VIEW_BUILDERS['explain']('not json', args) is None


class TestCaps:
    def test_cap_keeps_small_payload(self):
        payload = search_hits_payload(SEARCH_RESPONSE, '', 'logs')
        assert cap_structured_content(payload) == payload

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

    def test_byte_cap_halves_rows_until_it_fits(self):
        rows = [['x' * 200_000] for _ in range(10)]
        payload = {'kind': 'search_hits', 'version': 1, 'columns': ['a'], 'rows': rows}
        capped = cap_structured_content(payload)
        assert len(capped['rows']) == 5
        assert capped['meta'] == {'truncated': True, 'row_count': 10, 'column_count': 1}
        assert len(json.dumps(capped)) <= STRUCTURED_CONTENT_MAX_BYTES

    def test_byte_cap_can_reach_zero_rows(self):
        payload = {
            'kind': 'search_hits',
            'version': 1,
            'columns': ['a'],
            'rows': [['x' * (STRUCTURED_CONTENT_MAX_BYTES + 1)]],
        }
        capped = cap_structured_content(payload)
        assert capped['rows'] == []
        assert capped['meta']['truncated'] is True and capped['meta']['row_count'] == 1

    def test_too_large_only_when_zero_rows_still_exceed(self):
        payload = {
            'kind': 'search_hits',
            'version': 1,
            'columns': ['a'],
            'rows': [['small']],
            'meta': {'index': 'i' * (STRUCTURED_CONTENT_MAX_BYTES + 1)},
            'tool': 'SearchIndexTool',
            'refreshable': True,
            'intent': 'why',
        }
        marker = cap_structured_content(payload)
        assert marker['omitted'] == 'too_large'
        assert marker['bytes'] > STRUCTURED_CONTENT_MAX_BYTES
        assert {k: marker[k] for k in ('tool', 'refreshable', 'intent')} == {
            'tool': 'SearchIndexTool',
            'refreshable': True,
            'intent': 'why',
        }
        assert 'rows' not in marker

    def test_too_large_tree_payload(self):
        payload = {
            'kind': 'explain',
            'version': 1,
            'explanation': {'description': 'x' * (STRUCTURED_CONTENT_MAX_BYTES + 1)},
        }
        assert cap_structured_content(payload)['omitted'] == 'too_large'

    def test_cap_truncates_aggregation_tables(self):
        payload = search_hits_payload(SEARCH_RESPONSE, '', 'logs')
        payload['meta']['aggregations'] = [
            {'name': 'big', 'columns': ['key', 'doc_count'], 'rows': [['k', 1]] * 501}
        ]
        capped = cap_structured_content(payload)
        table = capped['meta']['aggregations'][0]
        assert len(table['rows']) == 500 and table['truncated'] is True
        assert 'truncated' not in capped['meta']


class TestViewPayload:
    def test_finalize_adds_tool_refreshable_and_presentation(self):
        payload = finalize_view_payload(
            ppl_rows_payload(PPL_RESPONSE, '', 'q'),
            'ppl_rows',
            'PPL',
            {'intent': 'n'},
            False,
            True,
        )
        assert payload['tool'] == 'PPL' and payload['intent'] == 'n'
        assert payload['refreshable'] is True
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

    def _info(self, kind='search_hits', **extra):
        from tools.tool_annotations import read_only_annotations

        return {
            'ui_resource_uri': RESULTS_VIEW_URI,
            'ui_kind': kind,
            'annotations': read_only_annotations('t'),
            **extra,
        }

    def test_view_structured_content_builds_from_view_data(self):
        items = [{'type': 'text', 'text': 't', 'view_data': SEARCH_RESPONSE}]
        args = SimpleNamespace(opensearch_cluster_name='c1', index='logs', query_dsl={})
        sc = view_structured_content(
            'SearchIndexTool',
            self._info(ui_app_callable=True),
            'Search',
            items,
            args,
            {},
            False,
        )
        assert sc['kind'] == 'search_hits' and sc['cluster'] == 'c1'
        assert sc['tool'] == 'Search' and sc['refreshable'] is True

    def test_view_structured_content_without_view_data(self):
        sc = view_structured_content(
            'X', self._info(), 'X', [{'type': 'text', 'text': 't'}], None, {}, False
        )
        assert sc['omitted'] == 'unsupported_format' and 'refreshable' not in sc

    def test_cat_markers_use_payload_kind(self):
        info = self._info(kind='cat_indices', ui_resource_uri=TABLE_VIEW_URI)
        args = SimpleNamespace(include_detail=True, index='logs')
        items = [{'type': 'text', 'text': 't', 'view_data': {'logs': {}}}]
        sc = view_structured_content('ListIndexTool', info, 'L', items, args, {}, False)
        assert sc['kind'] == 'cat_table' and sc['omitted'] == 'unsupported_format'
        assert view_structured_content('L', info, 'L', [], args, {}, True)['kind'] == 'cat_table'

    def test_view_structured_content_builder_failure(self):
        items = [{'type': 'text', 'text': 't', 'view_data': {'hits': 'oops'}}]
        broken = Mock(side_effect=TypeError('bad'))
        with patch.dict('tools.ui_resources.VIEW_BUILDERS', {'search_hits': broken}):
            sc = view_structured_content(
                'X', self._info(), 'X', items, None, {'intent': 'i'}, False
            )
        assert sc['omitted'] == 'unsupported_format' and sc['intent'] == 'i'

    def test_view_structured_content_finalize_failure(self):
        with patch('tools.ui_resources.finalize_view_payload', side_effect=RuntimeError('boom')):
            sc = view_structured_content('X', self._info(), 'X', [], None, {}, False)
        assert sc == {
            'kind': 'search_hits',
            'version': 1,
            'omitted': 'unsupported_format',
            'tool': 'X',
        }


class TestPresentationArgs:
    def test_split(self):
        args, presentation = split_presentation_args(
            {'index': 'i', 'intent': 'n', 'suggestions': ['a', 'b']}
        )
        assert args == {'index': 'i'}
        assert presentation == {'intent': 'n', 'suggestions': ['a', 'b']}
        assert split_presentation_args({'index': 'i', 'suggestions': []}) == ({'index': 'i'}, {})
        assert split_presentation_args({'intent': '  '}) == ({}, {})

    def test_narrative_is_no_longer_special(self):
        assert split_presentation_args({'narrative': 'n'}) == ({'narrative': 'n'}, {})

    @pytest.mark.parametrize(
        'args',
        [
            {'intent': 'rtl\u202eoverride'},
            {'intent': 'nul\x00'},
            {'intent': 'esc\x1b[31m'},
            {'suggestions': ['soft\u00adhyphen']},
            {'suggestions': ['tab\there']},
            {'suggestions': ['bom\ufeff']},
        ],
    )
    def test_control_and_format_chars_rejected(self, args):
        with pytest.raises(ValueError):
            split_presentation_args(args)

    def test_view_input_schema(self):
        schema = {'type': 'object', 'properties': {'q': {'type': 'string'}}}
        extended = view_input_schema(schema)
        assert set(extended['properties']) == {'q', 'intent', 'suggestions'}
        assert 'Why this query' in extended['properties']['intent']['description']
        assert 'next steps' in extended['properties']['suggestions']['description']
        assert set(schema['properties']) == {'q'}


class TestDashboardsLink:
    @pytest.fixture(autouse=True)
    def single(self):
        with patch('mcp_server_opensearch.global_state.get_mode', return_value='single'):
            yield

    def test_no_link_without_env(self, monkeypatch):
        monkeypatch.delenv('OPENSEARCH_DASHBOARDS_URL', raising=False)
        assert discover_ppl_url('source=logs') is None

    def test_ppl_link(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'https://osd.example.com/base/')
        url = discover_ppl_url("source=logs | where msg = 'it!s'")
        assert url.startswith('https://osd.example.com/base/app/data-explorer/discover#?_q=')
        state = unquote(url.split('_q=', 1)[1])
        assert state == (
            "(filters:!(),query:(dataset:(id:'::logs',title:'logs',type:INDEXES),"
            "language:PPL,query:'source=logs | where msg = !'it!!s!''))"
        )

    def test_backticks_stripped_from_source(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'http://127.0.0.1:5601')
        state = unquote(discover_ppl_url('source=`logs-*` | head 5').split('_q=', 1)[1])
        assert "dataset:(id:'::logs-*',title:'logs-*',type:INDEXES)" in state
        assert "query:'source=`logs-*` | head 5'" in state
        assert discover_ppl_url('source=`` | head 5') is None

    def test_no_link_without_source_or_in_multi_mode(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'http://127.0.0.1:5601')
        assert discover_ppl_url('describe logs') is None
        with patch('mcp_server_opensearch.global_state.get_mode', return_value='multi'):
            assert discover_ppl_url('source=logs') is None

    @pytest.mark.parametrize(
        'value',
        [
            'osd.example.com',
            'javascript:alert(1)',
            'file:///x',
            'https://user:pass@osd.example.com',
            'https://user@osd.example.com',
            'https://osd.example.com/?tenant=x',
            'https://osd.example.com/#/home',
            'https://osd.example.com/;p',
            'https://',
        ],
    )
    def test_malformed_url_rejected(self, monkeypatch, value):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', value)
        with pytest.raises(ValueError, match='OPENSEARCH_DASHBOARDS_URL'):
            dashboards_base_url()

    def test_valid_url_with_port_and_path(self, monkeypatch):
        monkeypatch.setenv('OPENSEARCH_DASHBOARDS_URL', 'https://osd.example.com:5601/osd/')
        assert dashboards_base_url() == 'https://osd.example.com:5601/osd'


class TestToolVisibility:
    def test_registry_visibility(self):
        from tools.tool_generator import OPERATION_VIEWS
        from tools.tools import TOOL_REGISTRY

        registry = dict(TOOL_REGISTRY)
        from tools.tool_annotations import read_only_annotations

        registry['ClusterHealthTool'] = {
            **OPERATION_VIEWS['cluster.health'],
            'annotations': read_only_annotations('h'),
        }
        registry['ExplainTool'] = {
            **OPERATION_VIEWS['explain'],
            'annotations': read_only_annotations('e'),
        }
        app = sorted(
            k for k, v in registry.items() if tool_ui_meta(k, v)['visibility'] == ['model', 'app']
        )
        assert app == [
            'CatNodesTool',
            'ClusterHealthTool',
            'GetAllocationTool',
            'GetShardsTool',
            'ListIndexTool',
            'PPLQueryTool',
            'SearchIndexTool',
        ]
        assert tool_ui_meta('IndexMappingTool', registry['IndexMappingTool']) == {
            'resourceUri': MAPPING_VIEW_URI,
            'visibility': ['model'],
        }
        assert tool_ui_meta('ExplainTool', registry['ExplainTool'])['visibility'] == ['model']
        assert tool_ui_meta('GenericOpenSearchApiTool', registry['GenericOpenSearchApiTool']) == {
            'visibility': ['model']
        }

    def test_generic_api_tool_never_app_callable(self):
        from tools.tool_annotations import read_only_annotations

        info = {
            'ui_resource_uri': RESULTS_VIEW_URI,
            'ui_kind': 'search_hits',
            'ui_app_callable': True,
            'annotations': read_only_annotations('g'),
        }
        assert tool_ui_meta('GenericOpenSearchApiTool', info)['visibility'] == ['model']
        assert tool_ui_meta('Other', info)['visibility'] == ['model', 'app']

    def test_registry_bindings(self):
        from tools.tools import TOOL_REGISTRY

        bound = {
            k: (v['ui_resource_uri'], v['ui_kind'])
            for k, v in TOOL_REGISTRY.items()
            if v.get('ui_resource_uri')
        }
        assert bound == {
            'ListIndexTool': (TABLE_VIEW_URI, 'cat_indices'),
            'IndexMappingTool': (MAPPING_VIEW_URI, 'index_mapping'),
            'SearchIndexTool': (RESULTS_VIEW_URI, 'search_hits'),
            'GetShardsTool': (TABLE_VIEW_URI, 'cat_shards'),
            'CatNodesTool': (TABLE_VIEW_URI, 'cat_nodes'),
            'GetAllocationTool': (TABLE_VIEW_URI, 'cat_allocation'),
            'PPLQueryTool': (RESULTS_VIEW_URI, 'ppl_rows'),
        }

    def test_generated_tools_bind_views(self):
        from tools.tool_generator import generate_tool_from_group

        def group(op):
            return [{'path': '/_x', 'method': 'get', 'details': {'x-operation-group': op}}]

        health = generate_tool_from_group('ClusterHealth', group('cluster.health'))
        explain = generate_tool_from_group('Explain', group('explain'))
        count = generate_tool_from_group('Count', group('count'))
        assert (health['ui_resource_uri'], health['ui_kind']) == (
            HEALTH_VIEW_URI,
            'cluster_health',
        )
        assert health['ui_app_callable'] is True
        assert (explain['ui_resource_uri'], explain['ui_kind']) == (EXPLAIN_VIEW_URI, 'explain')
        assert 'ui_app_callable' not in explain
        assert 'ui_resource_uri' not in count


class TestRegistry:
    SINKS = (
        r'innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\s*\(|new\s+Function\s*\('
        r'|setTimeout\s*\(\s*[\'"]|setInterval\s*\(\s*[\'"]'
    )

    @pytest.mark.parametrize('uri', sorted(UI_RESOURCES))
    def test_view_assembles_and_is_self_contained(self, uri):
        html = load_ui_html(uri)
        assert html.startswith('<!DOCTYPE html>')
        assert '{{' not in html
        assert 'ui/initialize' in html and 'startView(' in html
        assert '<script src' not in html and '<link' not in html
        assert '@import' not in html and 'url(' not in html
        assert 'fetch(' not in html and 'XMLHttpRequest' not in html
        assert 'WebSocket' not in html and 'EventSource' not in html
        urls = re.findall(r'https?://[^\s\'"]*', html)
        assert urls == ['http://www.w3.org/2000/svg']  # SVG namespace, not a load
        assert html.count('startView(') == 2  # definition and one call

    @pytest.mark.parametrize(
        'name',
        sorted(
            p.name
            for p in resources.files('tools.ui').iterdir()
            if p.name.endswith(('.html', '.js', '.css'))
        ),
    )
    def test_view_files_have_no_html_injection_sinks(self, name):
        text = resources.files('tools.ui').joinpath(name).read_text(encoding='utf-8')
        sinks = re.findall(self.SINKS, text)
        assert sinks == [], f'{name} uses {sinks}; build DOM nodes and set textContent instead'

    def test_every_ui_file_is_used(self):
        files = {
            p.name
            for p in resources.files('tools.ui').iterdir()
            if p.name.endswith(('.html', '.js', '.css'))
        }
        used = {'shell.html', 'common.css'}
        for spec in UI_RESOURCES.values():
            used.update(spec['scripts'])
        assert files == used

    def test_resource_meta(self):
        clip = {'ui': {'prefersBorder': True, 'permissions': {'clipboardWrite': {}}}}
        assert resource_meta(RESULTS_VIEW_URI) == clip
        assert resource_meta(TABLE_VIEW_URI) == clip
        for uri in (HEALTH_VIEW_URI, MAPPING_VIEW_URI, EXPLAIN_VIEW_URI):
            assert resource_meta(uri) == {'ui': {'prefersBorder': True}}

    def test_bound_resource_uris_only_for_enabled_tools(self):
        tools = {
            'A': {'ui_resource_uri': RESULTS_VIEW_URI},
            'B': {'ui_resource_uri': TABLE_VIEW_URI},
            'C': {},
            'D': {'ui_resource_uri': 'ui://unknown/view'},
        }
        assert bound_resource_uris(tools) == [RESULTS_VIEW_URI, TABLE_VIEW_URI]
        assert bound_resource_uris({'C': {}}) == []


class TestToolsExposeViewData:
    """Tools attach the raw response for the handler; the text is unchanged with the flag on or off."""

    def setup_method(self):
        self.mock_client = Mock()
        self.mock_client.search = AsyncMock(return_value=SEARCH_RESPONSE)
        self.mock_client.transport.perform_request = AsyncMock(return_value=PPL_RESPONSE)
        self.mock_client.info = AsyncMock(return_value={'version': {'number': '2.19.0'}})
        self.mock_client.cat.indices = AsyncMock(return_value=CAT_INDICES)
        self.mock_client.indices.get_mapping = AsyncMock(return_value=MAPPING)
        self.patcher = patch('opensearch.client.initialize_client', return_value=self.mock_client)
        self.patcher.start()
        import sys

        for module in ['tools.tools', 'opensearch.helper']:
            sys.modules.pop(module, None)
        import tools.tools as t

        self.t = t

    def teardown_method(self):
        self.patcher.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('flag', ['true', 'false'])
    @pytest.mark.parametrize('fmt', ['json', 'csv'])
    async def test_search(self, monkeypatch, flag, fmt):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', flag)
        args = self.t.SearchIndexArgs(
            index='logs', query_dsl={'match_all': {}}, opensearch_cluster_name='', format=fmt
        )
        result = await self.t.search_index_tool(args)
        assert result[0]['view_data'] == SEARCH_RESPONSE
        assert result[0]['text'].startswith(f'Search results from logs ({fmt.upper()} format)')
        assert set(result[0]) == {'type', 'text', 'view_data'}

    @pytest.mark.asyncio
    async def test_ppl(self):
        args = self.t.PplQueryArgs(opensearch_cluster_name='', query='source=x')
        result = await self.t.ppl_query_tool(args)
        assert result[0]['view_data'] == PPL_RESPONSE
        assert result[0]['text'].startswith('PPL query results:')

    @pytest.mark.asyncio
    async def test_list_indices(self):
        args = self.t.ListIndicesArgs(opensearch_cluster_name='', include_detail=False)
        result = await self.t.list_indices_tool(args)
        assert result[0]['view_data'] == CAT_INDICES
        assert json.loads(result[0]['text'].split('\n', 1)[1]) == ['logs', 'ts']

    @pytest.mark.asyncio
    async def test_mapping(self):
        args = self.t.GetIndexMappingArgs(opensearch_cluster_name='', index='logs')
        result = await self.t.get_index_mapping_tool(args)
        assert result[0]['view_data'] == MAPPING

    @pytest.mark.asyncio
    async def test_search_error_has_no_view_data(self):
        self.mock_client.search.side_effect = Exception('boom')
        args = self.t.SearchIndexArgs(
            index='logs', query_dsl={'match_all': {}}, opensearch_cluster_name=''
        )
        result = await self.t.search_index_tool(args)
        assert result[0].get('is_error') is True
        assert 'view_data' not in result[0]

    def test_tools_module_does_not_read_the_flag(self):
        source = resources.files('tools').joinpath('tools.py').read_text(encoding='utf-8')
        assert 'apps_enabled' not in source
