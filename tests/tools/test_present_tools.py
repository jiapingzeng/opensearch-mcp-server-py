# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import pytest
from mcp.client.client import Client
from mcp.client.extension import advertise
from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp_server_opensearch.mcp_handlers import build_handlers, configure_apps
from tools import present_tools
from tools.present_tools import (
    ClusterOverviewArgs,
    IndexExplorerArgs,
    cluster_overview_tool,
    format_bytes,
    index_explorer_tool,
    resolve_present_tools,
)
from tools.tool_annotations import read_only_annotations
from tools.tool_generator import OPERATION_VIEWS
from tools.tool_params import baseToolArgs
from tools.ui_resources import (
    CLUSTER_OVERVIEW_VIEW_URI,
    INDEX_EXPLORER_VIEW_URI,
    STRUCTURED_CONTENT_MAX_BYTES,
    STRUCTURED_CONTENT_MAX_ROWS,
)
from unittest.mock import AsyncMock, Mock, patch


UI_CLIENT = [advertise(EXTENSION_ID, {'mimeTypes': [APP_MIME_TYPE]})]
HEALTH = {
    'cluster_name': 'demo',
    'status': 'red',
    'timed_out': False,
    'number_of_nodes': 2,
    'number_of_data_nodes': 2,
    'active_primary_shards': 2,
    'active_shards': 2,
    'unassigned_shards': 2,
    'number_of_pending_tasks': 0,
    'active_shards_percent_as_number': 50.0,
    'not_a_health_field': 'dropped',
}
NODES = [
    {'name': 'n1', 'ip': '10.0.0.1', 'node.role': 'dimr', 'heap.percent': '40', 'cpu': '3'},
    {'name': 'n2', 'ip': '10.0.0.2', 'node.role': 'dimr', 'heap.percent': '72', 'cpu': '9'},
]


def _index(name, health, docs, store, rep='0'):
    return {
        'health': health,
        'status': 'open',
        'index': name,
        'uuid': f'u-{name}',
        'pri': '1',
        'rep': rep,
        'docs.count': docs,
        'docs.deleted': '0' if docs is not None else None,
        'store.size': store,
        'pri.store.size': store,
    }


INDICES = [
    _index('orders', 'green', '20', '5000'),
    _index('logs', 'yellow', '6', '7000', rep='1'),
    _index('audit', 'red', None, None),
    _index('.hidden', 'green', '1', '100'),
]
SHARDS = [
    {'index': 'orders', 'shard': '0', 'prirep': 'p', 'state': 'STARTED', 'docs': '20',
     'store': '5000', 'node': 'n1', 'unassigned.reason': None, 'unassigned.for': None},
    {'index': 'logs', 'shard': '0', 'prirep': 'p', 'state': 'STARTED', 'docs': '6',
     'store': '7000', 'node': 'n2', 'unassigned.reason': None, 'unassigned.for': None},
    {'index': 'logs', 'shard': '0', 'prirep': 'r', 'state': 'UNASSIGNED', 'docs': None,
     'store': None, 'node': None, 'unassigned.reason': 'INDEX_CREATED', 'unassigned.for': '1m'},
    {'index': 'audit', 'shard': '0', 'prirep': 'p', 'state': 'UNASSIGNED', 'docs': None,
     'store': None, 'node': None, 'unassigned.reason': 'INDEX_CREATED', 'unassigned.for': '1m'},
]  # fmt: skip
ALLOCATION = [
    {'shards': '1', 'disk.indices': '5000', 'disk.used': '900', 'disk.avail': '100',
     'disk.total': '1000', 'disk.percent': '90', 'host': 'h1', 'ip': '10.0.0.1', 'node': 'n1'},
    {'shards': '2', 'disk.indices': None, 'disk.used': None, 'disk.avail': None,
     'disk.total': None, 'disk.percent': None, 'host': None, 'ip': None, 'node': 'UNASSIGNED'},
]  # fmt: skip
GET_LOGS = {
    'logs': {
        'aliases': {'logs-read': {}},
        'mappings': {'properties': {'a': {'type': 'keyword'}}},
        'settings': {'index': {'number_of_shards': '1', 'number_of_replicas': '1'}},
    }
}


def _explain(body):
    return {
        'index': body['index'],
        'shard': body['shard'],
        'primary': body['primary'],
        'current_state': 'unassigned',
        'unassigned_info': {'reason': 'INDEX_CREATED'},
        'can_allocate': 'no',
        'allocate_explanation': 'cannot allocate because allocation is not permitted',
        'node_allocation_decisions': [
            {
                'node_name': 'n1',
                'node_decision': 'no',
                'deciders': [
                    {'decider': 'filter', 'decision': 'NO', 'explanation': 'node does not match'},
                    {'decider': 'disk', 'decision': 'YES', 'explanation': 'enough disk'},
                ],
            }
        ],
    }


def _client():
    c = Mock()
    c.close = AsyncMock()
    c.info = AsyncMock(return_value={'version': {'number': '2.19.0'}})
    c.cluster.health = AsyncMock(return_value=HEALTH)
    c.cluster.allocation_explain = AsyncMock(side_effect=lambda body: _explain(body))
    c.cat.nodes = AsyncMock(return_value=NODES)
    c.cat.indices = AsyncMock(return_value=INDICES)
    c.cat.shards = AsyncMock(return_value=SHARDS)
    c.cat.allocation = AsyncMock(return_value=ALLOCATION)
    c.indices.get = AsyncMock(return_value=GET_LOGS)
    return c


@pytest.fixture(autouse=True)
def single_mode(monkeypatch):
    from mcp_server_opensearch import global_state

    monkeypatch.setattr(global_state, '_current_mode', 'single')
    present_tools._PLANS.clear()
    with patch('opensearch.client.is_serverless_connection', return_value=False):
        yield
    present_tools._PLANS.clear()


@pytest.fixture
def client():
    c = _client()
    with patch('opensearch.client.initialize_client', return_value=c) as init:
        c.init = init
        yield c


def _overview(**kw):
    return ClusterOverviewArgs(opensearch_cluster_name='', **kw)


def _explorer(index=''):
    return IndexExplorerArgs(opensearch_cluster_name='', index=index)


def _col(table, name):
    i = table['columns'].index(name)
    return [r[i] for r in table['rows']]


class TestClusterOverview:
    @pytest.mark.asyncio
    async def test_payload_shape(self, client):
        [item] = await cluster_overview_tool(_overview())
        p = item['view_data']
        assert p['kind'] == 'cluster_overview' and p['version'] == 1 and p['cluster'] == ''
        assert p['requested'] == ['health', 'nodes', 'indices', 'shards', 'allocation']
        assert {k: v['status'] for k, v in p['sections'].items()} == dict.fromkeys(
            p['requested'], 'ok'
        )
        assert 'not_a_health_field' not in p['sections']['health']['data']
        idx = p['sections']['indices']
        assert _col(idx, 'index') == ['logs', 'orders', '.hidden', 'audit']  # largest first
        types = dict(zip(idx['columns'], idx['meta']['column_types']))
        assert types['store.size'] == 'bytes' and types['docs.count'] == 'number'
        assert types['index'] == 'string'
        assert idx['summary'] == {
            'count': 4,
            'by_health': {'green': 2, 'red': 1, 'yellow': 1},
            'docs': 27,
            'store_bytes': 12100,
            'largest': [['logs', 7000], ['orders', 5000], ['.hidden', 100], ['audit', 0]],
        }
        sh = p['sections']['shards']
        assert _col(sh, 'index')[:2] == ['audit', 'logs'] and _col(sh, 'state')[:2] == [
            'UNASSIGNED',
            'UNASSIGNED',
        ]
        assert sh['summary'] == {
            'total': 4,
            'by_state': {'STARTED': 2, 'UNASSIGNED': 2},
            'unassigned': 2,
            'unassigned_by_reason': {'INDEX_CREATED': 2},
            'per_node': {
                'n1': {'primaries': 1, 'replicas': 0},
                'n2': {'primaries': 1, 'replicas': 0},
            },
        }
        assert [(e['index'], e['primary']) for e in sh['explanations']] == [
            ('audit', True),
            ('logs', False),
        ]
        assert sh['explanations'][0]['nodes'] == [
            {'node': 'n1', 'decision': 'no', 'reasons': ['node does not match']}
        ]
        bodies = [c.kwargs['body'] for c in client.cluster.allocation_explain.call_args_list]
        assert bodies == [
            {'index': 'audit', 'shard': 0, 'primary': True},
            {'index': 'logs', 'shard': 0, 'primary': False},
        ]
        al = p['sections']['allocation']
        assert dict(zip(al['columns'], al['meta']['column_types']))['disk.used'] == 'bytes'
        assert p['tools'] == {'explorer': 'IndexExplorerTool'}

    @pytest.mark.asyncio
    async def test_reads_only_with_byte_units_and_columns(self, client):
        await cluster_overview_tool(_overview())
        assert client.cat.indices.call_args.kwargs['bytes'] == 'b'
        assert client.cat.shards.call_args.kwargs['h'].endswith('unassigned.reason,unassigned.for')
        assert client.cat.allocation.call_args.kwargs == {'format': 'json', 'bytes': 'b'}
        called = {name for name, *_ in client.mock_calls if not name.startswith('close')}
        assert called == {
            'info',
            'cluster.health',
            'cluster.allocation_explain',
            'cat.nodes',
            'cat.indices',
            'cat.shards',
            'cat.allocation',
        }

    @pytest.mark.asyncio
    async def test_text_summary_is_short(self, client):
        [item] = await cluster_overview_tool(_overview())
        lines = item['text'].split('\n')
        assert lines == [
            'demo is RED: 2 nodes (2 data), 2 active shards, 2 unassigned, 0 pending tasks.',
            'Indices: 4 (2 green, 1 red, 1 yellow), 27 documents, 11.8 KiB. Largest: logs '
            '(6.84 KiB), orders (4.88 KiB), .hidden (100 B).',
            'Shards: 4 (2 STARTED, 2 UNASSIGNED). Unassigned by reason: 2 INDEX_CREATED. '
            'audit[0] primary: cannot allocate because allocation is not permitted.',
            'Nodes: 2; highest heap n2 72%.',
            'Disk: highest use n1 at 90%.',
            'The user sees each section in the cluster overview view.',
        ]
        assert len(item['text']) < 600
        assert 'u-orders' not in item['text']  # no row data beyond names

    @pytest.mark.asyncio
    async def test_one_failing_section_does_not_affect_the_others(self, client):
        client.cat.nodes.side_effect = RuntimeError('nodes down')
        [item] = await cluster_overview_tool(_overview())
        s = item['view_data']['sections']
        assert s['nodes'] == {'status': 'error', 'error': 'Error getting nodes: nodes down'}
        assert all(s[k]['status'] == 'ok' for k in ('health', 'indices', 'shards', 'allocation'))
        assert 'The nodes section failed: Error getting nodes: nodes down' in item['text']
        assert not item.get('is_error')

    @pytest.mark.asyncio
    async def test_failed_client_creation_is_per_section(self, client):
        calls = {'n': 0}

        def flaky(args):
            calls['n'] += 1
            if calls['n'] == 3:
                raise ConnectionError('refused')
            return client

        client.init.side_effect = flaky
        [item] = await cluster_overview_tool(_overview())
        errors = [k for k, v in item['view_data']['sections'].items() if v['status'] == 'error']
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_every_section_failing_is_an_error(self, client):
        for m in (client.cluster.health, client.cat.nodes, client.cat.indices, client.cat.shards):
            m.side_effect = RuntimeError('down')
        client.cat.allocation.side_effect = RuntimeError('down')
        [item] = await cluster_overview_tool(_overview())
        assert item['is_error'] is True and 'view_data' not in item
        assert item['text'].count('section failed') == 5

    @pytest.mark.asyncio
    async def test_explain_failure_stays_in_the_shards_section(self, client):
        def explain(body):
            if body['index'] == 'audit':
                raise RuntimeError('forbidden')
            return _explain(body)

        client.cluster.allocation_explain.side_effect = explain
        [item] = await cluster_overview_tool(_overview())
        sh = item['view_data']['sections']['shards']
        assert sh['status'] == 'ok' and [e['index'] for e in sh['explanations']] == ['logs']
        assert sh['explain_error'] == 'Error getting allocation explanations: forbidden'

    @pytest.mark.asyncio
    async def test_explains_at_most_three_primaries_first(self, client):
        many = [{**SHARDS[2], 'index': f'r{i}', 'prirep': 'r'} for i in range(5)] + [
            {**SHARDS[3], 'index': 'zz', 'prirep': 'p'}
        ]
        client.cat.shards.return_value = many
        [item] = await cluster_overview_tool(_overview())
        assert [e['index'] for e in item['view_data']['sections']['shards']['explanations']] == [
            'zz',
            'r0',
            'r1',
        ]

    @pytest.mark.asyncio
    async def test_no_explain_without_unassigned_shards(self, client):
        client.cat.shards.return_value = SHARDS[:2]
        [item] = await cluster_overview_tool(_overview())
        assert 'explanations' not in item['view_data']['sections']['shards']
        client.cluster.allocation_explain.assert_not_called()

    @pytest.mark.asyncio
    async def test_sections_argument_fetches_only_those(self, client):
        [item] = await cluster_overview_tool(_overview(sections=['nodes']))
        p = item['view_data']
        assert p['requested'] == ['nodes'] and list(p['sections']) == ['nodes']
        client.cat.indices.assert_not_called()
        client.cluster.health.assert_not_called()

    def test_sections_argument_is_validated(self):
        with pytest.raises(ValueError):
            ClusterOverviewArgs(opensearch_cluster_name='', sections=['secrets'])

    @pytest.mark.asyncio
    async def test_optional_sections_off_when_their_tool_is_filtered(self, client):
        resolve_present_tools(_enabled(drop=['CatNodesTool', 'GetAllocationTool']))
        [item] = await cluster_overview_tool(_overview())
        s = item['view_data']['sections']
        assert s['nodes'] == {'status': 'disabled', 'tool': 'CatNodesTool'}
        assert s['allocation'] == {'status': 'disabled', 'tool': 'GetAllocationTool'}
        client.cat.nodes.assert_not_called()
        client.cat.allocation.assert_not_called()
        assert (
            'The nodes section is off because CatNodesTool is not enabled on this server.'
            in item['text']
        )

    @pytest.mark.asyncio
    async def test_row_cap_keeps_the_largest_and_full_summary(self, client):
        client.cat.indices.return_value = [
            _index(f'i{n:04d}', 'green', '1', str(n)) for n in range(600)
        ]
        [item] = await cluster_overview_tool(_overview(sections=['indices']))
        idx = item['view_data']['sections']['indices']
        assert len(idx['rows']) == STRUCTURED_CONTENT_MAX_ROWS
        assert idx['meta']['truncated'] is True and idx['meta']['row_count'] == 600
        assert _col(idx, 'index')[0] == 'i0599' and idx['summary']['count'] == 600

    @pytest.mark.asyncio
    async def test_byte_cap_halves_the_largest_table(self, client):
        client.cat.shards.return_value = [
            {**SHARDS[0], 'index': f'x{n}', 'node': 'n' * 5000} for n in range(400)
        ]
        [item] = await cluster_overview_tool(_overview())
        p = item['view_data']
        sh = p['sections']['shards']
        assert len(json.dumps(p).encode()) <= STRUCTURED_CONTENT_MAX_BYTES
        assert sh['meta']['truncated'] is True and sh['meta']['row_count'] == 400
        assert 0 < len(sh['rows']) < 400 and sh['summary']['total'] == 400
        assert 'truncated' not in p['sections']['indices']['meta']

    @pytest.mark.asyncio
    async def test_cluster_name_is_passed_to_every_client(self, client):
        await cluster_overview_tool(ClusterOverviewArgs(opensearch_cluster_name='prod'))
        assert {c.args[0].opensearch_cluster_name for c in client.init.call_args_list} == {'prod'}

    def test_format_bytes_matches_the_views(self):
        assert [format_bytes(n) for n in (0, 512, 7680, 144384, 1319413953331)] == [
            '0 B',
            '512 B',
            '7.50 KiB',
            '141 KiB',
            '1.20 TiB',
        ]


class TestIndexExplorer:
    @pytest.mark.asyncio
    async def test_all_indices_select_the_first_visible(self, client):
        [item] = await index_explorer_tool(_explorer())
        p = item['view_data']
        assert p['kind'] == 'index_explorer' and p['pattern'] == ''
        assert _col(p['list'], 'index') == ['audit', 'logs', 'orders', '.hidden']
        assert p['selected'] == 'audit' and p['sample_size'] == 10
        assert p['tools'] == {
            'detail': 'ListIndexTool',
            'mapping': 'IndexMappingTool',
            'shards': 'GetShardsTool',
            'search': 'SearchIndexTool',
        }
        assert client.cat.indices.call_args.kwargs['index'] is None
        client.indices.get.assert_called_once_with(index='audit')

    @pytest.mark.asyncio
    async def test_concrete_index_is_selected_with_settings_and_aliases(self, client):
        [item] = await index_explorer_tool(_explorer('logs'))
        p = item['view_data']
        assert p['selected'] == 'logs'
        assert p['detail'] == {
            'status': 'ok',
            'index': {
                'name': 'logs',
                'aliases': ['logs-read'],
                'settings': {'number_of_replicas': '1', 'number_of_shards': '1'},
            },
        }
        assert item['text'].split('\n') == [
            "Index explorer for 'logs': 4 indices (2 green, 1 red, 1 yellow), 27 documents, "
            '11.8 KiB.',
            'Selected logs: 1 primary shard, 1 replica, aliases logs-read.',
            'The user can open the overview, mapping, shards and sample documents of any listed '
            'index in the view.',
        ]

    @pytest.mark.asyncio
    async def test_pattern_is_listed_not_selected(self, client):
        [item] = await index_explorer_tool(_explorer('logs*'))
        assert client.cat.indices.call_args.kwargs['index'] == 'logs*'
        assert item['view_data']['selected'] == 'audit'

    @pytest.mark.asyncio
    async def test_list_failure_still_shows_a_concrete_index(self, client):
        client.cat.indices.side_effect = RuntimeError('no monitor permission')
        [item] = await index_explorer_tool(_explorer('logs'))
        p = item['view_data']
        assert p['list'] == {
            'status': 'error',
            'error': 'Error getting the index list: no monitor permission',
        }
        assert p['selected'] == 'logs' and p['detail']['status'] == 'ok'

    @pytest.mark.asyncio
    async def test_nothing_to_show_is_an_error(self, client):
        client.cat.indices.side_effect = RuntimeError('down')
        [item] = await index_explorer_tool(_explorer('logs-*'))
        assert item['is_error'] is True and 'view_data' not in item

    @pytest.mark.asyncio
    async def test_detail_failure_is_isolated(self, client):
        client.indices.get.side_effect = RuntimeError('forbidden')
        [item] = await index_explorer_tool(_explorer('logs'))
        p = item['view_data']
        assert p['list']['status'] == 'ok'
        assert p['detail'] == {'status': 'error', 'error': 'Error getting index logs: forbidden'}

    @pytest.mark.asyncio
    async def test_empty_match_has_no_selection(self, client):
        client.cat.indices.return_value = []
        [item] = await index_explorer_tool(_explorer('none-*'))
        p = item['view_data']
        assert p['selected'] is None and 'detail' not in p
        client.indices.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_rows_are_capped(self, client):
        client.cat.indices.return_value = [
            _index(f'i{n:04d}', 'green', '1', '1') for n in range(700)
        ]
        [item] = await index_explorer_tool(_explorer())
        lst = item['view_data']['list']
        assert len(lst['rows']) == STRUCTURED_CONTENT_MAX_ROWS
        assert lst['meta']['row_count'] == 700 and lst['summary']['count'] == 700


def _health_tool():
    async def health(args):
        return [{'type': 'text', 'text': json.dumps(HEALTH), 'view_data': HEALTH}]

    return {
        'display_name': 'ClusterHealthTool',
        'description': 'health',
        'input_schema': baseToolArgs.model_json_schema(),
        'function': health,
        'args_model': baseToolArgs,
        'http_methods': 'GET',
        'annotations': read_only_annotations('Health'),
        **OPERATION_VIEWS['cluster.health'],
    }


DEPENDENCIES = [
    'ListIndexTool',
    'IndexMappingTool',
    'SearchIndexTool',
    'GetShardsTool',
    'CatNodesTool',
    'GetAllocationTool',
    'GenericOpenSearchApiTool',
    'ClusterHealthTool',
    'ClusterOverviewTool',
    'IndexExplorerTool',
]


def _enabled(drop=(), renames=None, multi=False):
    """Enabled tools as get_tools returns them: single mode keys by display name."""
    from tools.tools import TOOL_REGISTRY

    out = {}
    for key in DEPENDENCIES:
        if key in drop:
            continue
        info = dict(_health_tool() if key == 'ClusterHealthTool' else TOOL_REGISTRY[key])
        info['display_name'] = (renames or {}).get(key, key)
        info['category'] = 'core_tools'
        out[key if multi else info['display_name']] = info
    return out


def _server(tools):
    server = Server('opensearch-mcp-server', **build_handlers(tools))
    configure_apps(server, tools)
    return server


async def _list(tools, ui=True):
    async with Client(_server(tools), extensions=UI_CLIENT if ui else None) as c:
        listed = {t.name: t for t in (await c.list_tools()).tools}
        try:
            uris = [str(r.uri) for r in (await c.list_resources()).resources]
        except Exception:
            uris = None
    return listed, uris


class TestListing:
    @pytest.mark.asyncio
    async def test_flag_off_the_tools_do_not_exist(self, monkeypatch, client):
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        listed, uris = await _list(_enabled())
        assert 'ClusterOverviewTool' not in listed and 'IndexExplorerTool' not in listed
        assert uris is None and all(t.meta == {'category': 'core_tools'} for t in listed.values())
        async with Client(_server(_enabled()), extensions=UI_CLIENT) as c:
            r = await c.call_tool('ClusterOverviewTool', {})
        assert r.is_error and r.content[0].text == 'Unknown or disabled tool: ClusterOverviewTool'
        assert r.structured_content is None

    def test_enabled_tools_log_lists_only_served_tools(self, monkeypatch, caplog):
        caplog.set_level('INFO', logger='mcp_server_opensearch.mcp_handlers')
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        build_handlers(_enabled())
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        build_handlers(_enabled())
        logged = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith('Enabled tools:')
        ]
        assert len(logged) == 2
        assert 'ClusterOverviewTool' not in logged[0] and 'IndexExplorerTool' not in logged[0]
        assert "'ListIndexTool'" in logged[0]
        assert 'ClusterOverviewTool' in logged[1] and 'IndexExplorerTool' in logged[1]

    @pytest.mark.asyncio
    async def test_ui_client_lists_read_only_present_tools(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        listed, uris = await _list(_enabled())
        for name, uri in (
            ('ClusterOverviewTool', CLUSTER_OVERVIEW_VIEW_URI),
            ('IndexExplorerTool', INDEX_EXPLORER_VIEW_URI),
        ):
            t = listed[name]
            assert t.meta == {
                'category': 'core_tools',
                'ui': {'resourceUri': uri, 'visibility': ['model', 'app']},
            }
            assert t.annotations.read_only_hint is True
            props = t.input_schema['properties']
            assert props['intent']['maxLength'] == 2000 and 'suggestions' not in props
            assert uri in uris
        assert listed['IndexMappingTool'].meta['ui']['visibility'] == ['model', 'app']
        assert listed['GenericOpenSearchApiTool'].meta['ui'] == {'visibility': ['model']}

    @pytest.mark.asyncio
    async def test_non_ui_client_matches_flag_off_and_gets_a_clear_error(
        self, monkeypatch, client
    ):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        on, _ = await _list(_enabled(), ui=False)
        monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED')
        off, _ = await _list(_enabled(), ui=False)
        assert [t.model_dump() for t in on.values()] == [t.model_dump() for t in off.values()]
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        async with Client(_server(_enabled())) as c:
            r = await c.call_tool('IndexExplorerTool', {'index': 'logs'})
        assert r.is_error and r.structured_content is None
        assert r.content[0].text == (
            'IndexExplorerTool shows an interactive view, which this client cannot display (it '
            'does not support the MCP Apps extension io.modelcontextprotocol/ui). Use '
            'ListIndexTool, IndexMappingTool, GetShardsTool, SearchIndexTool for text results '
            'instead.'
        )
        client.cat.indices.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_filtered_required_dependency_removes_the_tool(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        listed, uris = await _list(_enabled(drop=['GetShardsTool']))
        assert 'ClusterOverviewTool' not in listed and 'IndexExplorerTool' not in listed
        assert CLUSTER_OVERVIEW_VIEW_URI not in uris and INDEX_EXPLORER_VIEW_URI not in uris
        async with Client(_server(_enabled(drop=['GetShardsTool'])), extensions=UI_CLIENT) as c:
            r = await c.call_tool('IndexExplorerTool', {})
        assert r.content[0].text == 'Unknown or disabled tool: IndexExplorerTool'

    @pytest.mark.asyncio
    async def test_explorer_needs_mapping_and_the_overview_loses_its_link(
        self, monkeypatch, client
    ):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled(drop=['IndexMappingTool'])
        listed, _ = await _list(tools)
        assert 'IndexExplorerTool' not in listed and 'ClusterOverviewTool' in listed
        async with Client(_server(tools), extensions=UI_CLIENT) as c:
            r = await c.call_tool('ClusterOverviewTool', {'sections': ['health']})
        assert r.structured_content['tools'] == {}

    @pytest.mark.asyncio
    async def test_view_called_tools_are_app_visible_only_with_the_explorer(
        self, monkeypatch, client
    ):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        listed, _ = await _list(_enabled(drop=['IndexExplorerTool']))
        assert listed['IndexMappingTool'].meta['ui']['visibility'] == ['model']

    @pytest.mark.asyncio
    async def test_optional_dependencies_do_not_gate_listing(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled(drop=['CatNodesTool', 'GetAllocationTool'])
        listed, _ = await _list(tools)
        assert 'ClusterOverviewTool' in listed
        async with Client(_server(tools), extensions=UI_CLIENT) as c:
            r = await c.call_tool('ClusterOverviewTool', {})
        assert r.structured_content['sections']['nodes']['status'] == 'disabled'

    @pytest.mark.asyncio
    async def test_display_names_are_resolved_by_registry_key(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        renames = {
            'ListIndexTool': 'Index_Manager',
            'IndexMappingTool': 'Map_It',
            'IndexExplorerTool': 'Browse_Indices',
            'ClusterOverviewTool': 'Cluster_Dash',
        }
        tools = _enabled(renames=renames)
        listed, _ = await _list(tools)
        assert {'Browse_Indices', 'Cluster_Dash', 'Map_It'} <= set(listed)
        assert listed['Map_It'].meta['ui']['visibility'] == ['model', 'app']
        async with Client(_server(tools), extensions=UI_CLIENT) as c:
            ex = await c.call_tool('Browse_Indices', {'index': 'logs'})
            ov = await c.call_tool('Cluster_Dash', {'sections': ['indices']})
        assert ex.structured_content['tool'] == 'Browse_Indices'
        assert ex.structured_content['tools'] == {
            'detail': 'Index_Manager',
            'mapping': 'Map_It',
            'shards': 'GetShardsTool',
            'search': 'SearchIndexTool',
        }
        assert ov.structured_content['tools'] == {'explorer': 'Browse_Indices'}

    @pytest.mark.asyncio
    async def test_multi_mode_keys_and_cluster_name(self, monkeypatch, client):
        from mcp_server_opensearch import global_state

        monkeypatch.setattr(global_state, '_current_mode', 'multi')
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled(multi=True, renames={'ListIndexTool': 'Index_Manager'})
        listed, _ = await _list(tools)
        assert 'opensearch_cluster_name' in listed['IndexExplorerTool'].input_schema['required']
        async with Client(_server(tools), extensions=UI_CLIENT) as c:
            r = await c.call_tool('IndexExplorerTool', {'opensearch_cluster_name': 'prod'})
            missing = await c.call_tool('IndexExplorerTool', {})
        assert r.structured_content['cluster'] == 'prod'
        assert r.structured_content['tools']['detail'] == 'Index_Manager'
        names = {c.args[0].opensearch_cluster_name for c in client.init.call_args_list}
        assert names == {'prod'}
        assert missing.is_error and 'opensearch_cluster_name' in missing.content[0].text

    @pytest.mark.asyncio
    async def test_intent_is_shown_and_validated(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        async with Client(_server(_enabled()), extensions=UI_CLIENT) as c:
            ok = await c.call_tool(
                'ClusterOverviewTool',
                {
                    'intent': 'Why is the cluster red?',
                    'suggestions': ['x'],
                    'sections': ['health'],
                },
            )
            bad = await c.call_tool('ClusterOverviewTool', {'intent': 'hidden \u202e text'})
        sc = ok.structured_content
        assert sc['intent'] == 'Why is the cluster red?' and 'suggestions' not in sc
        assert sc['tool'] == 'ClusterOverviewTool' and sc['refreshable'] is True
        assert bad.is_error and bad.content[0].text.startswith("'intent' must be a string")

    @pytest.mark.asyncio
    async def test_structured_content_stays_within_the_byte_cap(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        client.cat.indices.return_value = [
            {**_index(f'i{n}', 'green', '1', '1'), 'uuid': 'u' * 4000} for n in range(500)
        ]
        async with Client(_server(_enabled()), extensions=UI_CLIENT) as c:
            r = await c.call_tool('IndexExplorerTool', {})
        sc = r.structured_content
        size = len(json.dumps(sc, ensure_ascii=False, separators=(',', ':')).encode())
        assert size <= STRUCTURED_CONTENT_MAX_BYTES and 'omitted' not in sc
        assert sc['list']['meta']['truncated'] is True and sc['list']['meta']['row_count'] == 500


FULL_NOTE = (
    'This client shows interactive views for some tools. Prefer them when they fit the request:\n'
    '- When the user asks how the cluster is doing (health, status, unassigned shards, disk), '
    'call ClusterOverviewTool.\n'
    '- When the user wants to explore or browse an index or index pattern, call '
    'IndexExplorerTool.\n'
    '- To answer a question about fields or before writing a query, call IndexMappingTool.\n'
    '- Results of ListIndexTool, IndexMappingTool, SearchIndexTool, GetShardsTool, CatNodesTool, '
    'GetAllocationTool and ClusterHealthTool also render as interactive views.\n'
    'When a tool takes an intent argument, set it to one sentence on why you are calling it; the '
    'view shows it to the user.\n'
    'The user can see the view, so summarize what matters instead of repeating its rows or fields.'
)


async def _guidance(tools, ui=True):
    async with Client(_server(tools), extensions=UI_CLIENT if ui else None) as c:
        return c.instructions, {t.name: t.description for t in (await c.list_tools()).tools}


class TestModelGuidance:
    @pytest.mark.asyncio
    async def test_ui_client_instructions_name_the_listed_tools(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        instructions, _ = await _guidance(_enabled())
        assert instructions == FULL_NOTE

    @pytest.mark.asyncio
    async def test_instructions_use_display_names_and_skip_unlisted_tools(
        self, monkeypatch, client
    ):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled(drop=['GetShardsTool'], renames={'IndexMappingTool': 'Map_It'})
        instructions, listed = await _guidance(tools)
        assert 'ClusterOverviewTool' not in listed and 'IndexExplorerTool' not in listed
        assert 'ClusterOverviewTool' not in instructions
        assert 'IndexExplorerTool' not in instructions
        assert (
            '- Results of ListIndexTool, Map_It, SearchIndexTool, CatNodesTool, ' in instructions
        )
        assert '- To answer a question about fields or before writing a query, call Map_It.' in (
            instructions
        )

    @pytest.mark.asyncio
    async def test_present_tool_descriptions_say_when_to_use_them(self, monkeypatch, client):
        from tools.tools import TOOL_REGISTRY

        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        _, listed = await _guidance(_enabled())
        overview, explorer = listed['ClusterOverviewTool'], listed['IndexExplorerTool']
        assert overview == (
            f'{TOOL_REGISTRY["ClusterOverviewTool"]["description"]} The details are in the '
            'view, so do not repeat them.'
        )
        for phrase in ('"how is my cluster"', '"is the cluster healthy"', 'unassigned shards'):
            assert phrase in overview
        assert explorer == (
            f'{TOOL_REGISTRY["IndexExplorerTool"]["description"]} For fields or mappings, use '
            'IndexMappingTool. The details are in the view, so do not repeat them.'
        )
        assert 'explore or browse' in explorer and 'inspect' not in explorer
        assert 'look like' not in explorer
        assert 'The user also sees' not in overview + explorer

    @pytest.mark.asyncio
    async def test_explorer_points_to_the_mapping_tool_display_name(self, monkeypatch, client):
        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        _, listed = await _guidance(_enabled(renames={'IndexMappingTool': 'Map_It'}))
        assert 'For fields or mappings, use Map_It.' in listed['IndexExplorerTool']

    @pytest.mark.asyncio
    async def test_customized_descriptions_are_sent_as_written(self, monkeypatch, client):
        from tools.config import _apply_validated_configs

        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled()
        _apply_validated_configs(
            tools,
            {
                'IndexMappingTool': {'description': 'Custom mapping.'},
                'IndexExplorerTool': {'description': 'Custom explorer.'},
            },
        )
        instructions, listed = await _guidance(tools)
        assert listed['IndexMappingTool'] == 'Custom mapping.'
        assert listed['IndexExplorerTool'] == 'Custom explorer.'
        assert listed['ListIndexTool'].endswith('as an interactive, sortable table.')
        assert instructions == FULL_NOTE

    @pytest.mark.asyncio
    async def test_bound_tools_get_one_view_sentence(self, monkeypatch, client):
        from tools.tools import TOOL_REGISTRY

        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        _, listed = await _guidance(_enabled())
        base = TOOL_REGISTRY['IndexMappingTool']['description']
        assert listed['IndexMappingTool'] == (
            f'{base}. The user also sees the mapping as an interactive, searchable field tree.'
        )
        assert listed['ClusterHealthTool'] == (
            'health. The user also sees the result as an interactive cluster health card.'
        )
        assert listed['ListIndexTool'].endswith(
            'returns full metadata. The user also sees the result as an interactive, sortable '
            'table.'
        )
        generic = TOOL_REGISTRY['GenericOpenSearchApiTool']['description']
        assert listed['GenericOpenSearchApiTool'] == generic

    @pytest.mark.asyncio
    @pytest.mark.parametrize('enabled', [False, True])
    async def test_without_apps_descriptions_match_the_registry(
        self, monkeypatch, client, enabled
    ):
        if enabled:
            monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        else:
            monkeypatch.delenv('OPENSEARCH_MCP_APPS_ENABLED', raising=False)
        tools = _enabled()
        instructions, listed = await _guidance(tools, ui=not enabled)
        expected = {
            info['display_name']: info['description']
            for info in tools.values()
            if not info.get('ui_present')
        }
        assert instructions is None
        assert listed == expected


@contextlib.asynccontextmanager
async def _stateless_http(server, monkeypatch):
    import uvicorn
    from mcp_server_opensearch.streaming_server import MCPStarletteApp
    from sse_starlette.sse import AppStatus

    app = MCPStarletteApp(server, stateless=True).create_app()
    http = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
    # sse_starlette's shutdown watcher outlives a server and would end this server's streams.
    monkeypatch.setattr(AppStatus, 'should_exit', False)
    monkeypatch.setattr(AppStatus, 'enable_automatic_graceful_drain', False)
    task = asyncio.create_task(http.serve())
    try:
        while not http.started:
            await asyncio.sleep(0.01)
        yield f'http://127.0.0.1:{http.servers[0].sockets[0].getsockname()[1]}/mcp'
    finally:
        http.should_exit = True
        await task


class TestStatelessHttpGuidance:
    """Stateless HTTP before 2026-07-28: tools/list cannot tell a UI client from a plain one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('ui', [False, True])
    async def test_tools_list_has_no_view_notes(self, monkeypatch, client, ui):
        from tools.tools import TOOL_REGISTRY

        monkeypatch.setenv('OPENSEARCH_MCP_APPS_ENABLED', 'true')
        tools = _enabled()
        async with _stateless_http(_server(tools), monkeypatch) as url:
            async with Client(url, extensions=UI_CLIENT if ui else None, mode='legacy') as c:
                assert c.protocol_version == '2025-11-25'
                instructions = c.instructions
                listed = {t.name: t.description for t in (await c.list_tools()).tools}
        assert instructions == (FULL_NOTE if ui else None)
        assert 'IndexExplorerTool' in listed and 'ClusterOverviewTool' in listed
        for description in listed.values():
            assert 'The user also sees' not in description
            assert 'do not repeat' not in description
        expected = {info['display_name']: info['description'] for info in tools.values()}
        expected['IndexExplorerTool'] = (
            f'{TOOL_REGISTRY["IndexExplorerTool"]["description"]} For fields or mappings, use '
            'IndexMappingTool.'
        )
        assert listed == expected
