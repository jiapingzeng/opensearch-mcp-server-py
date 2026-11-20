# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: ``tools/call`` for the core, generic and generated tools, and error shapes.

Each case drives the server through ``mcp_session`` and ``tools/call`` and goldens, under
``golden/call_tool_core/``, the call (name and arguments), the whole call result (content,
``isError``, structured content if any) and the ``fake_opensearch`` request log.

- ``tools/<Tool>/...``: every core ``TOOL_REGISTRY`` tool outside search relevance, skills,
  memory and agentic memory: one success with realistic canned replies and one OpenSearch
  error reply (4xx with an OpenSearch error body) each, plus a few variants.
- ``generic_api/...``: ``GenericOpenSearchApiTool``, every method and its validation paths.
- ``generated/...``: the four tools ``tool_generator`` builds from the pinned OpenAPI spec.
- ``errors/...``: error shapes shared by every tool.
- ``log/...``: the structured ``tool_execution`` log record ``tool_executor.py`` emits.

Tools are enabled the way a user enables them: ``OPENSEARCH_ENABLED_TOOLS`` or a config
file, never by editing the registry. Canned replies use fixed dates far from today (see
README.md), so the normalizer leaves them alone.
"""

import importlib
import logging
import pytest
import yaml
from dataclasses import dataclass, field
from mcp.shared.exceptions import MCPError
from typing import Any


AREA = 'call_tool_core'

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
ALLOW_WRITE_OFF = {'OPENSEARCH_SETTINGS_ALLOW_WRITE': 'false'}


def enable(*tools: str) -> dict:
    """Environment that enables ``tools`` on top of the default ``core_tools`` category."""
    return {'OPENSEARCH_ENABLED_TOOLS': ','.join(tools)}


def write_yaml(path, data: dict) -> str:
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


# ---------------------------------------------------------------------------
# canned OpenSearch replies
# ---------------------------------------------------------------------------


def os_error(status: int, error_type: str, reason: str, **fields: Any) -> dict:
    """An OpenSearch error body as the REST layer writes it."""
    cause = {'type': error_type, 'reason': reason, **fields}
    return {'error': {'root_cause': [cause], **cause}, 'status': status}


def index_not_found(index: str) -> dict:
    return os_error(
        404,
        'index_not_found_exception',
        f'no such index [{index}]',
        index=index,
        **{'resource.type': 'index_or_alias', 'resource.id': index, 'index_uuid': '_na_'},
    )


def forbidden(action: str) -> dict:
    return os_error(
        403,
        'security_exception',
        f'no permissions for [{action}] and User [name=reader, backend_roles=[], '
        'requestedTenant=null]',
    )


CAT_INDICES = [
    {
        'health': 'green',
        'status': 'open',
        'index': 'logs-2020.01.01',
        'uuid': 'fake-uuid-logs-0000001',
        'pri': '1',
        'rep': '1',
        'docs.count': '1200',
        'docs.deleted': '0',
        'store.size': '1.2mb',
        'pri.store.size': '620.4kb',
    },
    {
        'health': 'yellow',
        'status': 'open',
        'index': 'metrics',
        'uuid': 'fake-uuid-metrics-00001',
        'pri': '1',
        'rep': '1',
        'docs.count': '42',
        'docs.deleted': '3',
        'store.size': '88.1kb',
        'pri.store.size': '88.1kb',
    },
]

LOGS_MAPPINGS = {
    'properties': {
        '@timestamp': {'type': 'date'},
        'level': {'type': 'keyword'},
        'message': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
    }
}

INDEX_INFO = {
    'logs': {
        'aliases': {'logs-read': {}},
        'mappings': LOGS_MAPPINGS,
        'settings': {
            'index': {
                'creation_date': '1577836800000',
                'number_of_shards': '1',
                'number_of_replicas': '1',
                'uuid': 'fake-uuid-logs-0000001',
                'version': {'created': '136407927'},
                'provided_name': 'logs',
            }
        },
    }
}

SEARCH_HITS = {
    'took': 3,
    'timed_out': False,
    '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0},
    'hits': {
        'total': {'value': 2, 'relation': 'eq'},
        'max_score': 1.2,
        'hits': [
            {
                '_index': 'logs',
                '_id': '1',
                '_score': 1.2,
                '_source': {
                    '@timestamp': '2020-01-01T00:00:00Z',
                    'level': 'error',
                    'message': 'disk full',
                    'host': {'name': 'web-1'},
                },
            },
            {
                '_index': 'logs',
                '_id': '2',
                '_score': 0.9,
                '_source': {
                    '@timestamp': '2020-01-01T00:05:00Z',
                    'level': 'error',
                    'message': 'timeout, retrying',
                    'host': {'name': 'web-2'},
                },
            },
        ],
    },
}

SEARCH_PARSING_ERROR = os_error(
    400,
    'parsing_exception',
    'unknown query [mtch]',
    line=1,
    col=19,
)

CAT_SHARDS = [
    {
        'index': 'logs',
        'shard': '0',
        'prirep': 'p',
        'state': 'STARTED',
        'docs': '1200',
        'store': '620.4kb',
        'ip': '10.0.0.1',
        'node': 'node-1',
    },
    {
        'index': 'logs',
        'shard': '0',
        'prirep': 'r',
        'state': 'UNASSIGNED',
        'docs': None,
        'store': None,
        'ip': None,
        'node': None,
    },
]

CLUSTER_STATE = {
    'cluster_name': 'fake-cluster',
    'cluster_uuid': 'fake-cluster-uuid',
    'metadata': {
        'cluster_uuid': 'fake-cluster-uuid',
        'cluster_uuid_committed': True,
        'templates': {},
        'indices': {
            'logs': {
                'version': 4,
                'state': 'open',
                'settings': {'index': {'number_of_shards': '1', 'number_of_replicas': '1'}},
                'aliases': ['logs-read'],
            }
        },
    },
}

CAT_SEGMENTS = [
    {
        'index': 'logs',
        'shard': '0',
        'prirep': 'p',
        'ip': '10.0.0.1',
        'segment': '_0',
        'generation': '0',
        'docs.count': '1200',
        'docs.deleted': '0',
        'size': '610.2kb',
        'size.memory': '0',
        'committed': 'true',
        'searchable': 'true',
        'version': '10.2.1',
        'compound': 'true',
    }
]

CAT_NODES = [
    {'name': 'node-1', 'ip': '10.0.0.1', 'heap.percent': '35', 'cpu': '4'},
    {'name': 'node-2', 'ip': '10.0.0.2', 'heap.percent': '61', 'cpu': '12'},
]

INDEX_STATS = {
    '_shards': {'total': 2, 'successful': 1, 'failed': 0},
    '_all': {
        'primaries': {'docs': {'count': 1200, 'deleted': 0}, 'store': {'size_in_bytes': 635289}},
        'total': {'docs': {'count': 1200, 'deleted': 0}, 'store': {'size_in_bytes': 635289}},
    },
    'indices': {
        'logs': {
            'uuid': 'fake-uuid-logs-0000001',
            'primaries': {
                'docs': {'count': 1200, 'deleted': 0},
                'store': {'size_in_bytes': 635289},
            },
            'total': {'docs': {'count': 1200, 'deleted': 0}, 'store': {'size_in_bytes': 635289}},
        }
    },
}

TOP_QUERIES = {
    'top_queries': [
        {
            'timestamp': 1577836800000,
            'id': 'fake-query-record-1',
            'search_type': 'query_then_fetch',
            'indices': ['logs'],
            'source': {'query': {'match': {'level': {'query': 'error'}}}},
            'total_shards': 1,
            'node_id': 'fake-node-id-1',
            'measurements': {
                'latency': {'number': 12, 'count': 1, 'aggregationType': 'NONE'},
                'cpu': {'number': 1450000, 'count': 1, 'aggregationType': 'NONE'},
                'memory': {'number': 20480, 'count': 1, 'aggregationType': 'NONE'},
            },
        }
    ]
}

NO_HANDLER_INSIGHTS = {
    'error': 'no handler found for uri [/_insights/top_queries] and method [GET]',
    'status': 400,
}

HOT_THREADS = (
    '::: {node-1}{fake-node-id-1}{fake-ephemeral-id}{10.0.0.1}{10.0.0.1:9300}{dimr}\n'
    '   Hot threads at 2020-01-01T00:00:00.000Z, interval=500ms, busiestThreads=3, '
    'ignoreIdleThreads=true:\n'
    '\n'
    "    1.2% (6ms out of 500ms) cpu usage by thread 'opensearch[node-1][search][T#1]'\n"
)

CAT_ALLOCATION = [
    {
        'shards': '5',
        'disk.indices': '1.2mb',
        'disk.used': '20.1gb',
        'disk.avail': '79.9gb',
        'disk.total': '100gb',
        'disk.percent': '20',
        'host': '10.0.0.1',
        'ip': '10.0.0.1',
        'node': 'node-1',
    },
    {
        'shards': '1',
        'disk.indices': None,
        'disk.used': None,
        'disk.avail': None,
        'disk.total': None,
        'disk.percent': None,
        'host': None,
        'ip': None,
        'node': 'UNASSIGNED',
    },
]


def cat_task(task_id: str, action: str, running_time: str, running_time_ns: str) -> dict:
    return {
        'action': action,
        'task_id': task_id,
        'parent_task_id': '-',
        'type': 'transport',
        'start_time': '1577836800000',
        'timestamp': '00:00:00',
        'running_time': running_time,
        'running_time_ns': running_time_ns,
        'ip': '10.0.0.1',
        'node': 'node-1',
    }


CAT_TASKS = [
    cat_task('fake-node-id-1:101', 'indices:data/read/search', '5.2s', '5200000000'),
    cat_task('fake-node-id-1:102', 'indices:data/write/bulk', '1.1s', '1100000000'),
    cat_task('fake-node-id-1:103', 'cluster:monitor/tasks/lists', '1ms', '1000000'),
]

NODES_JVM = {
    '_nodes': {'total': 1, 'successful': 1, 'failed': 0},
    'cluster_name': 'fake-cluster',
    'nodes': {
        'fake-node-id-1': {
            'name': 'node-1',
            'transport_address': '10.0.0.1:9300',
            'host': '10.0.0.1',
            'ip': '10.0.0.1',
            'version': '3.1.0',
            'roles': ['cluster_manager', 'data', 'ingest'],
            'jvm': {
                'pid': 1234,
                'version': '21.0.7',
                'vm_name': 'OpenJDK 64-Bit Server VM',
                'start_time_in_millis': 1577836800000,
                'mem': {'heap_init_in_bytes': 1073741824, 'heap_max_in_bytes': 1073741824},
            },
        }
    },
}

UNRECOGNIZED_METRIC = os_error(
    400,
    'illegal_argument_exception',
    'request [/_nodes/_local/bogus] contains unrecognized metric: [bogus]',
)

PPL_JDBC = {
    'schema': [{'name': 'level', 'type': 'string'}, {'name': 'message', 'type': 'string'}],
    'datarows': [['error', 'disk full'], ['error', 'timeout, retrying']],
    'total': 2,
    'size': 2,
}

PPL_SYNTAX_ERROR = {
    'error': {
        'reason': 'Invalid Query',
        'details': "[mtch] is not a valid term at this part of the query: 'source=logs | mtch' "
        "<-- HERE. extraneous input 'mtch' expecting <EOF>",
        'type': 'SyntaxCheckException',
    },
    'status': 400,
}

CLUSTER_HEALTH = {
    'cluster_name': 'fake-cluster',
    'status': 'yellow',
    'timed_out': False,
    'number_of_nodes': 1,
    'number_of_data_nodes': 1,
    'discovered_master': True,
    'discovered_cluster_manager': True,
    'active_primary_shards': 5,
    'active_shards': 5,
    'relocating_shards': 0,
    'initializing_shards': 0,
    'unassigned_shards': 1,
    'delayed_unassigned_shards': 0,
    'number_of_pending_tasks': 0,
    'number_of_in_flight_fetch': 0,
    'task_max_waiting_in_queue_millis': 0,
    'active_shards_percent_as_number': 83.33333333333334,
}

MSEARCH_RESPONSES = {
    'took': 4,
    'responses': [
        {
            'took': 3,
            'timed_out': False,
            '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0},
            'hits': {'total': {'value': 1200, 'relation': 'eq'}, 'max_score': None, 'hits': []},
            'status': 200,
        }
    ],
}

EXPLANATION = {
    '_index': 'logs',
    '_id': '1',
    'matched': True,
    'explanation': {
        'value': 1.2,
        'description': 'weight(level:error in 0) [PerFieldSimilarity], result of:',
        'details': [],
    },
}

COUNT = {'count': 42, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}}


# ---------------------------------------------------------------------------
# harness for this module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    body: Any = None
    status: int = 200
    # The connection breaks before any response (fake_opensearch.fail_connection).
    connection_fails: bool = False


def ok(method: str, path: str, body: Any) -> Route:
    return Route(method, path, body)


def err(method: str, path: str, body: dict) -> Route:
    return Route(method, path, body, status=body['status'])


@dataclass(frozen=True)
class Case:
    """One ``tools/call``: golden name (below ``call_tool_core/``), tool, arguments, routes."""

    name: str
    tool: str
    arguments: dict
    routes: tuple = ()
    env: dict = field(default_factory=dict)
    is_error: bool = False


def add_routes(fake_opensearch, routes) -> None:
    for route in routes:
        if route.connection_fails:
            fake_opensearch.fail_connection(route.method, route.path)
        else:
            fake_opensearch.reply(route.method, route.path, route.body, status=route.status)


@pytest.fixture
def pin(fake_opensearch, golden, normalize):
    """``pin(name, tool, arguments, result, **extra)``: golden the call and the request log."""

    def pin(name: str, tool: str, arguments: dict, result: dict, **extra: Any) -> None:
        data = {
            'call': {'name': tool, 'arguments': arguments},
            'result': result,
            'requests': fake_opensearch.requests,
            **extra,
        }
        golden(f'{AREA}/{name}', normalize(data))

    return pin


@pytest.fixture
def run_case(mcp_session, fake_opensearch, pin):
    """Add the case's routes, build the server with its environment, call, golden."""

    async def run_case(case: Case, **session_kwargs: Any) -> dict:
        add_routes(fake_opensearch, case.routes)
        async with mcp_session(env={**NO_AUTH, **case.env}, **session_kwargs) as session:
            result = await session.call_tool(case.tool, case.arguments)
        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError') is case.is_error, result
        pin(case.name, case.tool, case.arguments, result)
        return result

    return run_case


@pytest.fixture
def request_headers(fake_opensearch, monkeypatch):
    """Headers of each request the fake receives, in the order of ``fake_opensearch.requests``.

    The fake's request log keeps no headers besides auth; this module needs them only for
    ``GenericOpenSearchApiTool``'s ``headers`` argument.
    """
    seen: list[dict] = []
    exchange = fake_opensearch._exchange

    def recording_exchange(method, url, data, auth, headers):
        seen.append(dict(headers or {}))
        return exchange(method, url, data, auth, headers)

    monkeypatch.setattr(fake_opensearch, '_exchange', recording_exchange)
    return seen


def tools_module():
    """``tools.tools`` as the server uses it (the harness pins it in ``sys.modules``)."""
    return importlib.import_module('tools.tools')


# ---------------------------------------------------------------------------
# core tools: success and OpenSearch error reply per tool
# ---------------------------------------------------------------------------

CORE_TOOL_CASES = [
    # ListIndexTool
    Case(
        'tools/ListIndexTool/success',
        'ListIndexTool',
        {},
        (ok('GET', '/_cat/indices', CAT_INDICES),),
    ),
    Case(
        'tools/ListIndexTool/success_names_only',
        'ListIndexTool',
        {'include_detail': False},
        (ok('GET', '/_cat/indices', CAT_INDICES),),
    ),
    Case(
        'tools/ListIndexTool/success_single_index',
        'ListIndexTool',
        {'index': 'logs'},
        (ok('GET', '/logs', INDEX_INFO),),
    ),
    Case(
        'tools/ListIndexTool/error_404',
        'ListIndexTool',
        {'index': 'missing', 'include_detail': False},
        (err('GET', '/_cat/indices/missing', index_not_found('missing')),),
        is_error=True,
    ),
    # A 403 on _cat/indices falls back to _resolve/index (names only).
    Case(
        'tools/ListIndexTool/forbidden_falls_back_to_resolve',
        'ListIndexTool',
        {},
        (
            err('GET', '/_cat/indices', forbidden('indices:monitor/settings/get')),
            ok(
                'GET',
                '/_resolve/index/*',
                {
                    'indices': [
                        {'name': 'logs-2020.01.01', 'attributes': ['open']},
                        {'name': 'metrics', 'aliases': ['metrics-read'], 'attributes': ['open']},
                    ],
                    'aliases': [{'name': 'metrics-read', 'indices': ['metrics']}],
                    'data_streams': [],
                },
            ),
        ),
    ),
    # IndexMappingTool
    Case(
        'tools/IndexMappingTool/success',
        'IndexMappingTool',
        {'index': 'logs'},
        (ok('GET', '/logs/_mapping', {'logs': {'mappings': LOGS_MAPPINGS}}),),
    ),
    Case(
        'tools/IndexMappingTool/error_404',
        'IndexMappingTool',
        {'index': 'missing'},
        (err('GET', '/missing/_mapping', index_not_found('missing')),),
        is_error=True,
    ),
    # SearchIndexTool
    Case(
        'tools/SearchIndexTool/success',
        'SearchIndexTool',
        {'index': 'logs', 'query_dsl': {'query': {'match': {'level': 'error'}}}, 'size': 2},
        (ok('POST', '/logs/_search', SEARCH_HITS),),
    ),
    Case(
        'tools/SearchIndexTool/success_csv',
        'SearchIndexTool',
        {
            'index': 'logs',
            'query_dsl': '{"query": {"match": {"level": "error"}}}',
            'format': 'csv',
        },
        (ok('POST', '/logs/_search', SEARCH_HITS),),
    ),
    Case(
        'tools/SearchIndexTool/error_400',
        'SearchIndexTool',
        {'index': 'logs', 'query_dsl': {'query': {'mtch': {'level': 'error'}}}},
        (err('POST', '/logs/_search', SEARCH_PARSING_ERROR),),
        is_error=True,
    ),
    # GetShardsTool
    Case(
        'tools/GetShardsTool/success',
        'GetShardsTool',
        {'index': 'logs'},
        (ok('GET', '/_cat/shards/logs', CAT_SHARDS),),
    ),
    Case(
        'tools/GetShardsTool/error_404',
        'GetShardsTool',
        {'index': 'missing'},
        (err('GET', '/_cat/shards/missing', index_not_found('missing')),),
        is_error=True,
    ),
    # GetClusterStateTool
    Case(
        'tools/GetClusterStateTool/success',
        'GetClusterStateTool',
        {'metric': 'metadata', 'index': 'logs'},
        (ok('GET', '/_cluster/state/metadata/logs', CLUSTER_STATE),),
        env=enable('GetClusterStateTool'),
    ),
    Case(
        'tools/GetClusterStateTool/error_403',
        'GetClusterStateTool',
        {},
        (err('GET', '/_cluster/state', forbidden('cluster:monitor/state')),),
        env=enable('GetClusterStateTool'),
        is_error=True,
    ),
    # GetSegmentsTool
    Case(
        'tools/GetSegmentsTool/success',
        'GetSegmentsTool',
        {'index': 'logs'},
        (ok('GET', '/_cat/segments/logs', CAT_SEGMENTS),),
        env=enable('GetSegmentsTool'),
    ),
    Case(
        'tools/GetSegmentsTool/error_404',
        'GetSegmentsTool',
        {'index': 'missing'},
        (err('GET', '/_cat/segments/missing', index_not_found('missing')),),
        env=enable('GetSegmentsTool'),
        is_error=True,
    ),
    # CatNodesTool
    Case(
        'tools/CatNodesTool/success',
        'CatNodesTool',
        {'metrics': 'name,ip,heap.percent,cpu'},
        (ok('GET', '/_cat/nodes', CAT_NODES),),
        env=enable('CatNodesTool'),
    ),
    Case(
        'tools/CatNodesTool/error_403',
        'CatNodesTool',
        {},
        (err('GET', '/_cat/nodes', forbidden('cluster:monitor/nodes/info')),),
        env=enable('CatNodesTool'),
        is_error=True,
    ),
    # GetIndexInfoTool
    Case(
        'tools/GetIndexInfoTool/success',
        'GetIndexInfoTool',
        {'index': 'logs'},
        (ok('GET', '/logs', INDEX_INFO),),
        env=enable('GetIndexInfoTool'),
    ),
    Case(
        'tools/GetIndexInfoTool/error_404',
        'GetIndexInfoTool',
        {'index': 'missing'},
        (err('GET', '/missing', index_not_found('missing')),),
        env=enable('GetIndexInfoTool'),
        is_error=True,
    ),
    # GetIndexStatsTool
    Case(
        'tools/GetIndexStatsTool/success',
        'GetIndexStatsTool',
        {'index': 'logs', 'metric': 'docs,store'},
        (ok('GET', '/logs/_stats/docs,store', INDEX_STATS),),
        env=enable('GetIndexStatsTool'),
    ),
    Case(
        'tools/GetIndexStatsTool/error_404',
        'GetIndexStatsTool',
        {'index': 'missing'},
        (err('GET', '/missing/_stats', index_not_found('missing')),),
        env=enable('GetIndexStatsTool'),
        is_error=True,
    ),
    # GetQueryInsightsTool
    Case(
        'tools/GetQueryInsightsTool/success',
        'GetQueryInsightsTool',
        {},
        (ok('GET', '/_insights/top_queries', TOP_QUERIES),),
        env=enable('GetQueryInsightsTool'),
    ),
    # What a cluster without the query insights plugin answers.
    Case(
        'tools/GetQueryInsightsTool/error_400',
        'GetQueryInsightsTool',
        {},
        (err('GET', '/_insights/top_queries', NO_HANDLER_INSIGHTS),),
        env=enable('GetQueryInsightsTool'),
        is_error=True,
    ),
    # GetNodesHotThreadsTool (text/plain reply)
    Case(
        'tools/GetNodesHotThreadsTool/success',
        'GetNodesHotThreadsTool',
        {},
        (ok('GET', '/_nodes/hot_threads', HOT_THREADS),),
        env=enable('GetNodesHotThreadsTool'),
    ),
    Case(
        'tools/GetNodesHotThreadsTool/error_403',
        'GetNodesHotThreadsTool',
        {},
        (err('GET', '/_nodes/hot_threads', forbidden('cluster:monitor/nodes/hot_threads')),),
        env=enable('GetNodesHotThreadsTool'),
        is_error=True,
    ),
    # GetAllocationTool
    Case(
        'tools/GetAllocationTool/success',
        'GetAllocationTool',
        {},
        (ok('GET', '/_cat/allocation', CAT_ALLOCATION),),
        env=enable('GetAllocationTool'),
    ),
    Case(
        'tools/GetAllocationTool/error_403',
        'GetAllocationTool',
        {},
        (err('GET', '/_cat/allocation', forbidden('cluster:monitor/state')),),
        env=enable('GetAllocationTool'),
        is_error=True,
    ),
    # GetLongRunningTasksTool: three tasks, limit 2.
    Case(
        'tools/GetLongRunningTasksTool/success',
        'GetLongRunningTasksTool',
        {'limit': 2},
        (ok('GET', '/_cat/tasks', CAT_TASKS),),
        env=enable('GetLongRunningTasksTool'),
    ),
    Case(
        'tools/GetLongRunningTasksTool/error_403',
        'GetLongRunningTasksTool',
        {},
        (err('GET', '/_cat/tasks', forbidden('cluster:monitor/tasks/lists')),),
        env=enable('GetLongRunningTasksTool'),
        is_error=True,
    ),
    # GetNodesTool
    Case(
        'tools/GetNodesTool/success',
        'GetNodesTool',
        {'node_id': '_local', 'metric': 'jvm'},
        (ok('GET', '/_nodes/_local/jvm', NODES_JVM),),
        env=enable('GetNodesTool'),
    ),
    Case(
        'tools/GetNodesTool/error_400',
        'GetNodesTool',
        {'node_id': '_local', 'metric': 'bogus'},
        (err('GET', '/_nodes/_local/bogus', UNRECOGNIZED_METRIC),),
        env=enable('GetNodesTool'),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', CORE_TOOL_CASES, ids=lambda case: case.name)
async def test_core_tool(run_case, case):
    await run_case(case)


# PPLQueryTool is enabled through a config file (the `observability` category).
PPL_CASES = [
    Case(
        'tools/PPLQueryTool/success',
        'PPLQueryTool',
        {'query': 'source=logs | where level="error" | fields level, message | head 2'},
        (ok('POST', '/_plugins/_ppl', PPL_JDBC),),
    ),
    Case(
        'tools/PPLQueryTool/success_csv',
        'PPLQueryTool',
        {'query': 'source=logs | fields level, message | head 2', 'format': 'csv'},
        (
            ok(
                'POST',
                '/_plugins/_ppl',
                'level,message\nerror,disk full\nerror,"timeout, retrying"\n',
            ),
        ),
    ),
    Case(
        'tools/PPLQueryTool/error_400',
        'PPLQueryTool',
        {'query': 'source=logs | mtch'},
        (err('POST', '/_plugins/_ppl', PPL_SYNTAX_ERROR),),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', PPL_CASES, ids=lambda case: case.name)
async def test_ppl_query_tool(run_case, case, tmp_path):
    config = write_yaml(
        tmp_path / 'config.yml', {'tool_filters': {'enabled_categories': ['observability']}}
    )
    await run_case(case, config_file=config)


async def test_search_index_tool_max_size_limit_from_config_file(run_case, tmp_path):
    # tools.SearchIndexTool.max_size_limit caps the `size` sent to OpenSearch (50 -> 5).
    config = write_yaml(
        tmp_path / 'config.yml', {'tools': {'SearchIndexTool': {'max_size_limit': 5}}}
    )
    case = Case(
        'tools/SearchIndexTool/max_size_limit_from_config_file',
        'SearchIndexTool',
        {'index': 'logs', 'query_dsl': {'query': {'match_all': {}}}, 'size': 50},
        (ok('POST', '/logs/_search', SEARCH_HITS),),
    )
    await run_case(case, config_file=config)


def clusters_config(tmp_path, fake_opensearch) -> str:
    """Multi-mode config file with two clusters, both served by the fake."""
    cluster = {'opensearch_url': fake_opensearch.url, 'opensearch_no_auth': True}
    return write_yaml(
        tmp_path / 'clusters.yml', {'clusters': {'local': dict(cluster), 'backup': dict(cluster)}}
    )


async def test_list_clusters_tool_multi_mode(run_case, tmp_path, fake_opensearch):
    # ListClustersTool is multi-only and never calls OpenSearch, so it has no OpenSearch
    # error reply; its other shape is the single-mode call below.
    config = clusters_config(tmp_path, fake_opensearch)
    await run_case(
        Case('tools/ListClustersTool/success_multi_mode', 'ListClustersTool', {}),
        mode='multi',
        config_file=config,
    )


async def test_list_clusters_tool_single_mode_is_unknown(run_case):
    await run_case(
        Case(
            'tools/ListClustersTool/single_mode_unknown',
            'ListClustersTool',
            {},
            env=enable('ListClustersTool'),
            is_error=True,
        )
    )


# ---------------------------------------------------------------------------
# GenericOpenSearchApiTool
# ---------------------------------------------------------------------------

DOC_WRITE = {
    '_index': 'logs',
    '_id': 'fake-doc-id-1',
    '_version': 1,
    'result': 'created',
    '_shards': {'total': 2, 'successful': 1, 'failed': 0},
    '_seq_no': 0,
    '_primary_term': 1,
}

GENERIC_API_CASES = [
    Case(
        'generic_api/get',
        'GenericOpenSearchApiTool',
        {'path': '/_cluster/health'},
        (ok('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
    Case(
        'generic_api/post_dict_body',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_doc', 'method': 'POST', 'body': {'level': 'info', 'message': 'hello'}},
        (Route('POST', '/logs/_doc', DOC_WRITE, status=201),),
    ),
    Case(
        'generic_api/put_string_body',
        'GenericOpenSearchApiTool',
        {
            'path': '/logs/_doc/1',
            'method': 'PUT',
            'body': '{"level": "warn", "message": "string body"}',
        },
        (
            ok(
                'PUT',
                '/logs/_doc/1',
                {**DOC_WRITE, '_id': '1', '_version': 2, 'result': 'updated'},
            ),
        ),
    ),
    Case(
        'generic_api/delete',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_doc/1', 'method': 'DELETE'},
        (
            ok(
                'DELETE',
                '/logs/_doc/1',
                {**DOC_WRITE, '_id': '1', '_version': 3, 'result': 'deleted'},
            ),
        ),
    ),
    Case(
        'generic_api/head',
        'GenericOpenSearchApiTool',
        {'path': '/logs', 'method': 'HEAD'},
        (ok('HEAD', '/logs', None),),
    ),
    # The security plugin's REST API takes PATCH with a JSON Patch (list) body.
    Case(
        'generic_api/patch_list_body',
        'GenericOpenSearchApiTool',
        {
            'path': '/_plugins/_security/api/internalusers/reader',
            'method': 'PATCH',
            'body': [{'op': 'replace', 'path': '/backend_roles', 'value': ['readers']}],
        },
        (
            ok(
                'PATCH',
                '/_plugins/_security/api/internalusers/reader',
                {'status': 'OK', 'message': "'reader' updated."},
            ),
        ),
    ),
    Case(
        'generic_api/lowercase_method',
        'GenericOpenSearchApiTool',
        {'path': '/_cluster/health', 'method': 'get'},
        (ok('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
    # A GET sends its body as JSON, as the `body` description ("Request body for
    # GET/POST/PUT") says. The case id is historical: the body is no longer dropped.
    Case(
        'generic_api/get_body_dropped',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_search', 'body': {'query': {'match': {'level': 'error'}}}},
        (ok('GET', '/logs/_search', SEARCH_HITS),),
    ),
    Case(
        'generic_api/invalid_method',
        'GenericOpenSearchApiTool',
        {'path': '/_search', 'method': 'FETCH'},
        is_error=True,
    ),
    Case(
        'generic_api/path_without_slash',
        'GenericOpenSearchApiTool',
        {'path': '_cat/indices'},
        is_error=True,
    ),
    Case(
        'generic_api/query_params',
        'GenericOpenSearchApiTool',
        {
            'path': '/_cat/indices',
            'query_params': {'format': 'json', 'v': True, 'h': 'index,health'},
        },
        (ok('GET', '/_cat/indices', [{'index': 'logs', 'health': 'green'}]),),
    ),
    Case(
        'generic_api/post_allow_write_off',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_doc', 'method': 'POST', 'body': {'level': 'info'}},
        env=ALLOW_WRITE_OFF,
        is_error=True,
    ),
    # A GET still works with allow-write off.
    Case(
        'generic_api/get_allow_write_off',
        'GenericOpenSearchApiTool',
        {'path': '/_cluster/health'},
        (ok('GET', '/_cluster/health', CLUSTER_HEALTH),),
        env=ALLOW_WRITE_OFF,
    ),
    Case(
        'generic_api/error_404',
        'GenericOpenSearchApiTool',
        {'path': '/missing/_search', 'method': 'POST', 'body': {'query': {'match_all': {}}}},
        (err('POST', '/missing/_search', index_not_found('missing')),),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', GENERIC_API_CASES, ids=lambda case: case.name)
async def test_generic_api_tool(run_case, case):
    await run_case(case)


# Only these request headers are pinned: the rest (user-agent with the package version,
# connection) is the client library's, not this tool's.
PINNED_HEADERS = ('accept', 'content-type', 'x-opaque-id')


async def test_generic_api_tool_headers(mcp_session, fake_opensearch, pin, request_headers):
    arguments = {
        'path': '/_cat/indices',
        'query_params': {'format': 'json'},
        'headers': {'X-Opaque-Id': 'characterization', 'Accept': 'application/json'},
    }
    fake_opensearch.reply('GET', '/_cat/indices', [{'index': 'logs', 'health': 'green'}])
    async with mcp_session(env=NO_AUTH) as session:
        result = await session.call_tool('GenericOpenSearchApiTool', arguments)

    assert result['isError'] is False, result
    headers = [
        {name.lower(): value for name, value in sent.items() if name.lower() in PINNED_HEADERS}
        for sent in request_headers
    ]
    pin('generic_api/headers', 'GenericOpenSearchApiTool', arguments, result, headers=headers)


# ---------------------------------------------------------------------------
# generated tools (pinned_openapi)
# ---------------------------------------------------------------------------

MSEARCH_NDJSON = '{"index":"logs"}\n{"query":{"match_all":{}}}\n'
MSEARCH_ITEMS = [{'index': 'logs'}, {'query': {'match_all': {}}}]

# A generated tool sends only the query parameters the caller sets (`q=level:error` on
# `/logs/_count`); the ones left out are not sent.
# Each call opens a client, then checks the version (GET /) before any body is processed.
GENERATED_CASES = [
    # MsearchTool: body as NDJSON, as a JSON array and as a string holding a JSON array.
    Case(
        'generated/MsearchTool/ndjson',
        'MsearchTool',
        {'body': MSEARCH_NDJSON},
        (ok('GET', '/_msearch', MSEARCH_RESPONSES),),
    ),
    Case(
        'generated/MsearchTool/json_array',
        'MsearchTool',
        {'body': MSEARCH_ITEMS},
        (ok('GET', '/_msearch', MSEARCH_RESPONSES),),
    ),
    Case(
        'generated/MsearchTool/string_holding_array',
        'MsearchTool',
        {'body': '[{"index":"logs"},{"query":{"match_all":{}}}]'},
        (ok('GET', '/_msearch', MSEARCH_RESPONSES),),
    ),
    # Endpoint selection with a path param; NDJSON without its final newline.
    Case(
        'generated/MsearchTool/with_index',
        'MsearchTool',
        {'index': 'logs', 'body': '{}\n{"query":{"match_all":{}}}'},
        (ok('GET', '/logs/_msearch', MSEARCH_RESPONSES),),
    ),
    Case(
        'generated/MsearchTool/error_400',
        'MsearchTool',
        {'body': '{"index":"logs"}\n'},
        (
            err(
                'GET',
                '/_msearch',
                os_error(
                    400,
                    'action_request_validation_exception',
                    'Validation Failed: 1: no requests added;',
                ),
            ),
        ),
        is_error=True,
    ),
    # ExplainTool
    Case(
        'generated/ExplainTool/success',
        'ExplainTool',
        {'index': 'logs', 'id': '1', 'body': {'query': {'match': {'level': 'error'}}}},
        (ok('GET', '/logs/_explain/1', EXPLANATION),),
    ),
    Case(
        'generated/ExplainTool/invalid_json_string_body',
        'ExplainTool',
        {'index': 'logs', 'id': '1', 'body': '{"query": {"match": '},
        is_error=True,
    ),
    Case(
        'generated/ExplainTool/error_404',
        'ExplainTool',
        {'index': 'missing', 'id': '1', 'body': {'query': {'match_all': {}}}},
        (err('GET', '/missing/_explain/1', index_not_found('missing')),),
        is_error=True,
    ),
    # CountTool: endpoint selection without and with the {index} path param.
    Case(
        'generated/CountTool/without_index',
        'CountTool',
        {},
        (ok('GET', '/_count', COUNT),),
    ),
    Case(
        'generated/CountTool/with_index_and_query_param',
        'CountTool',
        {'index': 'logs', 'q': 'level:error'},
        (ok('GET', '/logs/_count', COUNT),),
    ),
    Case(
        'generated/CountTool/error_404',
        'CountTool',
        {'index': 'missing'},
        (err('GET', '/missing/_count', index_not_found('missing')),),
        is_error=True,
    ),
    # ClusterHealthTool: endpoint selection without and with the {index} path param.
    # The generated tools' `timeout` argument goes to OpenSearch as a query parameter, and
    # the client request timeout stays the configured one. The `with_timeout` case pins it:
    # its recorded `/_cluster/health` request has `timeout=5s` in its query parameters.
    Case(
        'generated/ClusterHealthTool/without_index',
        'ClusterHealthTool',
        {},
        (ok('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
    Case(
        'generated/ClusterHealthTool/with_index',
        'ClusterHealthTool',
        {'index': 'logs', 'wait_for_status': 'yellow', 'level': 'indices'},
        (ok('GET', '/_cluster/health/logs', CLUSTER_HEALTH),),
    ),
    Case(
        'generated/ClusterHealthTool/with_timeout',
        'ClusterHealthTool',
        {'timeout': '5s'},
        (ok('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
    Case(
        'generated/ClusterHealthTool/error_403',
        'ClusterHealthTool',
        {},
        (err('GET', '/_cluster/health', forbidden('cluster:monitor/health')),),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', GENERATED_CASES, ids=lambda case: case.name)
async def test_generated_tool(run_case, pinned_openapi, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# generic error shapes
# ---------------------------------------------------------------------------

MANY_INDICES = [
    {'index': f'logs-2020.01.{day:02d}', 'health': 'green', 'docs.count': '1200'}
    for day in range(1, 31)
]

ERROR_CASES = [
    Case('errors/unknown_tool', 'NoSuchTool', {}, is_error=True),
    Case('errors/missing_required_argument', 'IndexMappingTool', {}, is_error=True),
    Case(
        'errors/wrong_argument_type',
        'ListIndexTool',
        {'include_detail': 'sometimes'},
        is_error=True,
    ),
    # Unknown arguments are ignored: the call succeeds.
    Case(
        'errors/extra_unknown_argument',
        'ListIndexTool',
        {'include_detail': False, 'no_such_argument': 'ignored'},
        (ok('GET', '/_cat/indices', CAT_INDICES),),
    ),
    # A write-only tool (PUT) is filtered out of the registry when writes are off, so the
    # call meets the unknown-tool path. No core tool outside search relevance is write-only.
    Case(
        'errors/write_tool_allow_write_off',
        'CreateQuerySetTool',
        {'name': 'set-1', 'description': 'queries', 'queries': '["error"]'},
        env={**ALLOW_WRITE_OFF, **enable('CreateQuerySetTool')},
        is_error=True,
    ),
    # The connection breaks before any response: opensearch-py's transport tries four
    # times and each try is sent once, so OpenSearch sees four `/_cat/indices` requests.
    Case(
        'errors/connection_failure',
        'ListIndexTool',
        {'include_detail': False},
        (Route('GET', '/_cat/indices', connection_fails=True),),
        is_error=True,
    ),
    Case(
        'errors/server_error_500',
        'ListIndexTool',
        {'include_detail': False},
        (
            err(
                'GET',
                '/_cat/indices',
                os_error(500, 'null_pointer_exception', 'Cannot invoke "Object.toString()"'),
            ),
        ),
        is_error=True,
    ),
    # OPENSEARCH_MAX_RESPONSE_SIZE: GET / (about 500 bytes) fits, the _cat/indices reply
    # (about 2 KB) does not.
    Case(
        'errors/response_over_size_limit',
        'ListIndexTool',
        {'include_detail': False},
        (ok('GET', '/_cat/indices', MANY_INDICES),),
        env={'OPENSEARCH_MAX_RESPONSE_SIZE': '1024'},
        is_error=True,
    ),
    # OPENSEARCH_URL is set, so dynamic mode is off and a per-call override is refused.
    Case(
        'errors/dynamic_override_rejected',
        'ListIndexTool',
        {'include_detail': False, 'opensearch_url': 'http://other-cluster.invalid:9200'},
        is_error=True,
    ),
    # characterization: current behaviour. A missing-field error echoes every argument the
    # caller sent, opensearch_password included (known issue; pinned as is).
    Case(
        'errors/missing_field_echoes_password',
        'IndexMappingTool',
        {'opensearch_username': 'admin', 'opensearch_password': 'hunter2'},
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', ERROR_CASES, ids=lambda case: case.name)
async def test_error_shape(run_case, case):
    await run_case(case)


async def test_error_version_outside_min_version(mcp_session, fake_opensearch, pin):
    # GetQueryInsightsTool needs 2.12.0+: listed at startup (3.1.0), refused once the
    # cluster answers 2.11.0 at call time.
    fake_opensearch.reply('GET', '/_insights/top_queries', TOP_QUERIES)
    async with mcp_session(env={**NO_AUTH, **enable('GetQueryInsightsTool')}) as session:
        fake_opensearch.version = '2.11.0'
        result = await session.call_tool('GetQueryInsightsTool', {})

    assert result['isError'] is True, result
    pin('errors/version_outside_min_version', 'GetQueryInsightsTool', {}, result)


async def test_error_multi_mode_unknown_cluster(run_case, tmp_path, fake_opensearch):
    config = clusters_config(tmp_path, fake_opensearch)
    case = Case(
        'errors/multi_mode_unknown_cluster',
        'ListIndexTool',
        {'opensearch_cluster_name': 'no-such-cluster', 'include_detail': False},
        is_error=True,
    )
    await run_case(case, mode='multi', config_file=config)


async def test_error_helper_raises_caught_by_tool(run_case, monkeypatch):
    # A helper's exception is caught by the tool function and returned via log_tool_error.
    async def failing_helper(args):
        raise RuntimeError('characterization: helper failed')

    monkeypatch.setattr(tools_module(), 'get_index_mapping', failing_helper)
    await run_case(
        Case('errors/helper_raises', 'IndexMappingTool', {'index': 'logs'}, is_error=True)
    )


def let_tool_function_raise(monkeypatch, exception: Exception) -> None:
    """Make IndexMappingTool's function raise ``exception`` out of the tool, uncaught.

    Every core tool function catches its helpers' exceptions and returns them through
    ``log_tool_error``; with that re-raising instead, the helper's exception leaves the tool
    function as it would from a tool without its own error handling.
    """

    async def failing_helper(args):
        raise exception

    def reraise(tool_name, error, *args, **kwargs):
        raise error

    module = tools_module()
    monkeypatch.setattr(module, 'get_index_mapping', failing_helper)
    monkeypatch.setattr(module, 'log_tool_error', reraise)


async def test_error_tool_function_raises_uncaught(run_case, monkeypatch):
    # The server turns an uncaught exception into an isError result holding str(exception).
    let_tool_function_raise(monkeypatch, RuntimeError('characterization: uncaught in tool'))
    await run_case(
        Case(
            'errors/tool_function_raises_uncaught',
            'IndexMappingTool',
            {'index': 'logs'},
            is_error=True,
        )
    )


async def test_error_mcp_error_passes_through(
    mcp_session, fake_opensearch, golden, normalize, monkeypatch
):
    # An MCPError leaving the tool is re-raised and answered as a JSON-RPC error.
    let_tool_function_raise(
        monkeypatch,
        MCPError(
            code=-32602, message='characterization: MCPError in tool', data={'index': 'logs'}
        ),
    )
    params = {'name': 'IndexMappingTool', 'arguments': {'index': 'logs'}}
    async with mcp_session(env=NO_AUTH) as session:
        response = await session.raw('tools/call', params)

    assert 'error' in response, response
    golden(
        f'{AREA}/errors/mcp_error_passthrough',
        normalize({'request': params, 'response': response, 'requests': fake_opensearch.requests}),
    )


# ---------------------------------------------------------------------------
# structured log record (tool_executor.py)
# ---------------------------------------------------------------------------

EXECUTOR_LOGGER = 'mcp_server_opensearch.tool_executor'
EVENT_FIELDS = (
    'event_type',
    'tool_name',
    'tool_key',
    'status',
    'error_type',
    'duration_ms',
    'client_name',
)


def tool_execution_records(caplog) -> list[dict]:
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
        }
        for record in caplog.records
        if getattr(record, 'event_type', None) == 'tool_execution'
    ]


LOG_CASES = [
    Case(
        'log/success',
        'ListIndexTool',
        {'include_detail': False},
        (ok('GET', '/_cat/indices', CAT_INDICES),),
    ),
    # An OpenSearch error the tool returns through log_tool_error (is_error: True).
    # characterization: current behaviour. Such a soft error is logged with status "error"
    # but no `error_type`, although tool_executor.py's docstring promises "Error rates by
    # error type and tool"; only exceptions raised out of the tool get one.
    Case(
        'log/error',
        'IndexMappingTool',
        {'index': 'missing'},
        (err('GET', '/missing/_mapping', index_not_found('missing')),),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', LOG_CASES, ids=lambda case: case.name)
async def test_tool_execution_log_record(
    mcp_session, fake_opensearch, golden, normalize, caplog, case
):
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    add_routes(fake_opensearch, case.routes)
    async with mcp_session(env=NO_AUTH) as session:
        result = await session.call_tool(case.tool, case.arguments)

    assert result['isError'] is case.is_error, result
    golden(f'{AREA}/{case.name}', normalize({'records': tool_execution_records(caplog)}))
