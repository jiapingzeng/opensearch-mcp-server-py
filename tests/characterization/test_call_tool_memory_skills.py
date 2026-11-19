# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: ``tools/call`` for the skills, memory and agentic memory tools.

Each case calls one tool through the MCP protocol and goldens, under
``golden/call_tool_memory_skills/``, the call (name and arguments), the whole call result
(content, ``isError``, structured content if any) and the requests the call sent to the fake
OpenSearch. Per tool there is one success case with canned replies modelled on
``tests/tools/test_skills_tools.py``, ``test_memory_tools.py``,
``test_agentic_memory_tools.py`` and ``agentic_memory_data.py``, and one case where
OpenSearch answers with a 4xx and an OpenSearch error body.

The tools are enabled the way a user enables them:

- skills (``SKILLS_TOOLS_REGISTRY``): ``OPENSEARCH_ENABLED_CATEGORIES=skills``, in process
  (``mcp_session``);
- memory (``MEMORY_TOOLS_REGISTRY``): ``MEMORY_TOOLS_ENABLED=true``, which is read when
  ``tools.memory_tools`` is imported, so the server runs as a subprocess
  (``memory_enabled_session``); this covers the index auto-create on first use and what a
  memory tool call returns in multi mode;
- agentic memory (``AGENTIC_MEMORY_TOOLS_REGISTRY``):
  ``OPENSEARCH_ENABLED_CATEGORIES=agentic_memory``, or a config file's ``tool_filters``
  for the memory container id default from that file.

``OPENSEARCH_NO_AUTH`` lets the server reach the fake. The fake reports OpenSearch 3.3.0,
the ``min_version`` of the agentic memory tools (the skills tools need 1.0.0 or 2.19.0, the
memory tools 1.0.0). Canned data uses fixed dates in 2020, far from today (see README.md).
"""

import json
import pytest
import yaml
from dataclasses import dataclass
from typing import Any, Callable


AREA = 'call_tool_memory_skills'
OPENSEARCH_VERSION = '3.3.0'

# Without auth configured the server takes the AWS path and fails it before the fake.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
SKILLS_ENV = {**NO_AUTH, 'OPENSEARCH_ENABLED_CATEGORIES': 'skills'}
AGENTIC_ENV = {**NO_AUTH, 'OPENSEARCH_ENABLED_CATEGORIES': 'agentic_memory'}

SKILLS_TOOLS = ('DataDistributionTool', 'LogPatternAnalysisTool', 'MetricChangeAnalysisTool')
MEMORY_TOOLS = ('SaveMemoryTool', 'SearchMemoryTool', 'DeleteMemoryTool')
AGENTIC_MEMORY_TOOLS = (
    'CreateAgenticMemorySessionTool',
    'AddAgenticMemoriesTool',
    'GetAgenticMemoryTool',
    'UpdateAgenticMemoryTool',
    'DeleteAgenticMemoryByIDTool',
    'DeleteAgenticMemoryByQueryTool',
    'SearchAgenticMemoryTool',
)

# The analysis windows the skills cases use, as the caller writes them and as the server
# sends them to OpenSearch (format_time_string).
SELECTION = ('2020-01-01 10:00:00', '2020-01-01 10:30:00')
BASELINE = ('2020-01-01 09:00:00', '2020-01-01 09:30:00')
SELECTION_START_SENT = '2020-01-01T10:00:00.000Z'
BASELINE_START_SENT = '2020-01-01T09:00:00.000Z'
WINDOWS = {
    'selectionTimeRangeStart': SELECTION[0],
    'selectionTimeRangeEnd': SELECTION[1],
    'timeField': '@timestamp',
}

MEMORY_INDEX = 'agent-memory'
CONTAINER_ID = 'HudqiJkB1SltqOcZusVU'
CONTAINERS = f'/_plugins/_ml/memory_containers/{CONTAINER_ID}/memories'


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
        index_uuid='_na_',
        **{'resource.id': index, 'resource.type': 'index_or_alias'},
    )


def security_exception(action: str, user: str) -> dict:
    return os_error(
        403,
        'security_exception',
        f'no permissions for [{action}] and User [name={user}, backend_roles=[], '
        'requestedTenant=null]',
    )


def ml_status_exception(status: int, reason: str) -> dict:
    """The error body ml-commons writes for a memory container request it rejects."""
    return os_error(status, 'status_exception', reason)


SHARDS = {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}


def search_reply(index: str, hits: list[dict], total: int | None = None) -> dict:
    return {
        'took': 4,
        'timed_out': False,
        '_shards': SHARDS,
        'hits': {
            'total': {'value': len(hits) if total is None else total, 'relation': 'eq'},
            'max_score': max((hit.get('_score') or 0.0 for hit in hits), default=None),
            'hits': [{'_index': index, **hit} for hit in hits],
        },
    }


def by_window(selection: Any, baseline: Any) -> Callable[[dict], Any]:
    """A reply body that depends on the analysis window named in the request.

    The skills tools send one request per window to the same path; the selection window
    is the one whose start time appears in the request body.
    """

    def reply(request: dict) -> Any:
        sent = json.dumps(request['body'])
        if SELECTION_START_SENT in sent:
            return selection
        assert BASELINE_START_SENT in sent, f'request names neither window: {sent}'
        return baseline

    return reply


# --- DataDistributionTool --------------------------------------------------------------


def log_docs(start: str, rows: list[tuple[str, str, int]]) -> list[dict]:
    """Search hits for ``(status, service, bytes)`` rows, one minute apart from ``start``."""
    return [
        {
            '_id': f'{start[11:13]}-{i}',
            '_score': 1.0,
            '_source': {
                '@timestamp': f'{start[:10]}T{start[11:13]}:{i:02d}:00Z',
                'status': status,
                'service': service,
                'bytes': size,
                # Unique per document: above the cardinality cap, so never analyzed.
                'message': f'request {start[11:13]}-{i} served by {service}',
            },
        }
        for i, (status, service, size) in enumerate(rows)
    ]


# Divergences differ per field (bytes 0.7, status 0.5, service 0.1) and value shares differ
# within each field, so the result order never depends on set iteration order.
SELECTION_LOGS = log_docs(
    SELECTION[0],
    [('error', 'auth', 4096)] * 3
    + [('error', 'cart', 4096)] * 2
    + [('error', 'pay', 4096), ('ok', 'auth', 4096)]
    + [('ok', 'auth', 512), ('ok', 'cart', 512), ('ok', 'pay', 512)],
)
BASELINE_LOGS = log_docs(
    BASELINE[0],
    [('error', 'auth', 512)]
    + [('ok', 'auth', 512)] * 3
    + [('ok', 'cart', 512)] * 4
    + [('ok', 'pay', 512)] * 2,
)

LOGS_MAPPING = {
    'logs-app': {
        'mappings': {
            'properties': {
                '@timestamp': {'type': 'date'},
                'status': {'type': 'keyword'},
                'service': {'type': 'keyword'},
                'bytes': {'type': 'long'},
                'message': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}},
                },
            }
        }
    }
}


# --- MetricChangeAnalysisTool ----------------------------------------------------------

METRICS_MAPPING = {
    'metrics-app': {
        'mappings': {
            'properties': {
                '@timestamp': {'type': 'date'},
                'service': {'type': 'keyword'},
                'latency': {'type': 'double'},
                'cpu_pct': {'type': 'float'},
                'errors': {'type': 'long'},
            }
        }
    }
}


def percentiles_reply(hits: int, values: dict[str, tuple[float, float]]) -> dict:
    return {
        'took': 12,
        'timed_out': False,
        '_shards': SHARDS,
        'hits': {'total': {'value': hits, 'relation': 'eq'}, 'max_score': None, 'hits': []},
        'aggregations': {
            name: {'values': {'50.0': p50, '90.0': p90}} for name, (p50, p90) in values.items()
        },
    }


# Change scores differ per field (latency > cpu_pct > errors, whose baseline is all zero).
METRICS_SELECTION = percentiles_reply(
    1800, {'latency': (250.0, 900.0), 'cpu_pct': (45.0, 70.0), 'errors': (0.0, 2.0)}
)
METRICS_BASELINE = percentiles_reply(
    1750, {'latency': (120.0, 300.0), 'cpu_pct': (40.0, 60.0), 'errors': (0.0, 0.0)}
)


def count_reply(count: int) -> dict:
    return {'count': count, '_shards': SHARDS}


# --- LogPatternAnalysisTool ------------------------------------------------------------


def ppl_patterns(rows: list[tuple[int, str]]) -> dict:
    """A PPL (jdbc format) reply for the pattern aggregation the tool runs."""
    return {
        'schema': [
            {'name': 'pattern_count', 'type': 'bigint'},
            {'name': 'patterns_field', 'type': 'string'},
        ],
        'datarows': [list(row) for row in rows],
        'total': len(rows),
        'size': len(rows),
        'status': 200,
    }


PATTERNS_BASELINE = ppl_patterns(
    [(1000, 'GET /api/<*> 200'), (900, 'User <*> logged in'), (10, 'Connection timeout to <*>')]
)
# The timeout pattern surges (lift above 3); the NullPointerException one is new, and its
# repeated wildcards collapse into one.
PATTERNS_SELECTION = ppl_patterns(
    [
        (900, 'GET /api/<*> 200'),
        (800, 'User <*> logged in'),
        (300, 'Connection timeout to <*>'),
        (50, 'NullPointerException at <*> <*>'),
    ]
)

PPL_ERROR = {
    'error': {
        'reason': 'Invalid Query',
        'details': "can't resolve Symbol(namespace=FIELD_NAME, name=message) in type env",
        'type': 'SemanticCheckException',
    },
    'status': 400,
}


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    body: Any = None
    status: int = 200


@dataclass(frozen=True)
class Case:
    """One tool call: its arguments and the routes the fake answers it with."""

    tool: str
    name: str
    arguments: dict
    routes: tuple[Route, ...]
    env: tuple[tuple[str, str], ...] = ()

    @property
    def id(self) -> str:
        return f'{self.tool}/{self.name}'


def install(fake_opensearch, routes: tuple[Route, ...]) -> None:
    for route in routes:
        fake_opensearch.reply(route.method, route.path, route.body, status=route.status)


LOG_PATTERN_ARGS = {
    'index': 'logs-app',
    'logFieldName': 'message',
    **WINDOWS,
    'baseTimeRangeStart': BASELINE[0],
    'baseTimeRangeEnd': BASELINE[1],
}
DISTRIBUTION_ARGS = {
    'index': 'logs-app',
    **WINDOWS,
    'baselineTimeRangeStart': BASELINE[0],
    'baselineTimeRangeEnd': BASELINE[1],
    'size': 100,
}
METRIC_ARGS = {
    'index': 'metrics-app',
    **WINDOWS,
    'baselineTimeRangeStart': BASELINE[0],
    'baselineTimeRangeEnd': BASELINE[1],
    'topN': 5,
}

SKILLS_CASES = (
    Case(
        'DataDistributionTool',
        'success',
        DISTRIBUTION_ARGS,
        (
            Route(
                'POST',
                '/logs-app/_search',
                by_window(
                    search_reply('logs-app', SELECTION_LOGS),
                    search_reply('logs-app', BASELINE_LOGS),
                ),
            ),
            Route('GET', '/logs-app/_mapping', LOGS_MAPPING),
        ),
    ),
    Case(
        'DataDistributionTool',
        'opensearch_error',
        DISTRIBUTION_ARGS,
        (Route('POST', '/logs-app/_search', index_not_found('logs-app'), 404),),
    ),
    Case(
        'MetricChangeAnalysisTool',
        'success',
        METRIC_ARGS,
        (
            Route('GET', '/metrics-app/_mapping', METRICS_MAPPING),
            Route('POST', '/metrics-app/_count', by_window(count_reply(1800), count_reply(1750))),
            Route('POST', '/metrics-app/_search', by_window(METRICS_SELECTION, METRICS_BASELINE)),
        ),
    ),
    # A 400 on the window count, caused by the filter: the tool rewrites it as a hint.
    Case(
        'MetricChangeAnalysisTool',
        'opensearch_error',
        {**METRIC_ARGS, 'filter': '{"wildcard": {"_id": "req-*"}}'},
        (
            Route('GET', '/metrics-app/_mapping', METRICS_MAPPING),
            Route(
                'POST',
                '/metrics-app/_count',
                os_error(
                    400,
                    'query_shard_exception',
                    'Can only use wildcard queries on keyword and text fields - not on [_id] '
                    'which is of type [_id]',
                    index='metrics-app',
                    index_uuid='Yk3c6mGfQnqF1x2kS9VbWA',
                ),
                400,
            ),
        ),
    ),
    # Diff mode (a baseline, no trace field). Insight mode (no baseline) is not pinned: its
    # PPL query lists ERROR_KEYWORDS, a set, so the request text changes with the hash seed.
    Case(
        'LogPatternAnalysisTool',
        'success',
        LOG_PATTERN_ARGS,
        (Route('POST', '/_plugins/_ppl', by_window(PATTERNS_SELECTION, PATTERNS_BASELINE)),),
    ),
    Case(
        'LogPatternAnalysisTool',
        'opensearch_error',
        LOG_PATTERN_ARGS,
        (Route('POST', '/_plugins/_ppl', PPL_ERROR, 400),),
    ),
)


# --- memory tools (subprocess) ---------------------------------------------------------

MEMORY_DOC_PATH = f'/{MEMORY_INDEX}/_doc'
MEMORY_INDEX_EXISTS = Route('HEAD', f'/{MEMORY_INDEX}')
MEMORY_INDEX_MISSING = Route('HEAD', f'/{MEMORY_INDEX}', status=404)

SAVE_ARGS = {
    'memory': 'User prefers vegetarian food and is allergic to nuts',
    'user_id': 'user-123',
    'session_id': 'session-42',
    # Tags are split on commas and stripped; empty ones are dropped.
    'tags': 'preference, dietary,',
}
SEARCH_ARGS = {'query': 'dietary preferences', 'user_id': 'user-123', 'size': 5}
DELETE_ARGS = {'memory_id': 'mem-7Kx2'}

MEMORY_HITS = [
    {
        '_id': 'mem-7Kx2',
        '_score': 2.4,
        '_source': {
            'memory_text': 'User prefers vegetarian food and is allergic to nuts',
            'user_id': 'user-123',
            'session_id': 'session-42',
            'tags': ['preference', 'dietary'],
            'created_at': '2020-01-01T00:00:00+00:00',
            'updated_at': '2020-01-01T00:00:00+00:00',
        },
    },
    {
        '_id': 'mem-9Pq1',
        '_score': 0.8,
        '_source': {
            'memory_text': 'User likes spicy Thai curries',
            'user_id': 'user-123',
            'agent_id': 'food-assistant',
            'created_at': '2020-01-02T00:00:00+00:00',
        },
    },
]

MEMORY_CASES = (
    Case(
        'SaveMemoryTool',
        'success',
        SAVE_ARGS,
        (
            MEMORY_INDEX_EXISTS,
            Route(
                'POST',
                MEMORY_DOC_PATH,
                {
                    '_index': MEMORY_INDEX,
                    '_id': 'mem-7Kx2',
                    '_version': 1,
                    'result': 'created',
                    '_shards': {'total': 2, 'successful': 1, 'failed': 0},
                    '_seq_no': 0,
                    '_primary_term': 1,
                },
                201,
            ),
        ),
    ),
    Case(
        'SaveMemoryTool',
        'opensearch_error',
        SAVE_ARGS,
        (
            MEMORY_INDEX_EXISTS,
            Route(
                'POST',
                MEMORY_DOC_PATH,
                security_exception('indices:data/write/index', 'memory-reader'),
                403,
            ),
        ),
    ),
    # First use: the index does not exist, so the tool creates it through the AWS
    # OpenSearch Service API (boto3), not through OpenSearch. The fake OPENSEARCH_URL names
    # no AWS domain, so no domain name can be derived and the call fails before boto3.
    Case(
        'SaveMemoryTool',
        'index_auto_create_no_domain',
        SAVE_ARGS,
        (MEMORY_INDEX_MISSING,),
    ),
    # characterization: current behaviour. First use with the domain named: the tool calls
    # boto3's `opensearch` create_index with camelCase parameters (domainName, indexName,
    # indexSchema) where the API takes DomainName, IndexName and IndexSchema, so botocore's
    # parameter validation rejects the call before it is signed or sent: on a managed domain
    # the memory index is never auto-created.
    Case(
        'SaveMemoryTool',
        'index_auto_create_domain',
        SAVE_ARGS,
        (MEMORY_INDEX_MISSING,),
        env=(('AWS_OPENSEARCH_DOMAIN_NAME', 'agent-domain'), ('AWS_REGION', 'eu-west-1')),
    ),
    # First use on OpenSearch Serverless: before create_index the tool sets up a data access
    # policy for the caller, starting with STS get_caller_identity, which fails without AWS
    # credentials (the harness keeps boto3 off local AWS config and instance metadata)
    # before any request leaves the process.
    Case(
        'SaveMemoryTool',
        'index_auto_create_serverless_no_credentials',
        SAVE_ARGS,
        (MEMORY_INDEX_MISSING,),
        env=(
            ('AWS_OPENSEARCH_SERVERLESS', 'true'),
            ('AWS_OPENSEARCH_COLLECTION_ID', 'k2x9memcoll'),
            ('AWS_REGION', 'eu-west-1'),
        ),
    ),
    Case(
        'SearchMemoryTool',
        'success',
        SEARCH_ARGS,
        (
            MEMORY_INDEX_EXISTS,
            Route('POST', f'/{MEMORY_INDEX}/_search', search_reply(MEMORY_INDEX, MEMORY_HITS)),
        ),
    ),
    # "*" lists memories newest first, without the recency script; the user id comes from
    # MEMORY_USER_ID and a custom index from MEMORY_INDEX_NAME.
    Case(
        'SearchMemoryTool',
        'list_all_env_defaults',
        {'query': '*'},
        (
            Route('HEAD', '/team-memory'),
            Route('POST', '/team-memory/_search', search_reply('team-memory', MEMORY_HITS[:1])),
        ),
        env=(('MEMORY_USER_ID', 'user-123'), ('MEMORY_INDEX_NAME', 'team-memory')),
    ),
    Case(
        'SearchMemoryTool',
        'opensearch_error',
        SEARCH_ARGS,
        (
            MEMORY_INDEX_EXISTS,
            Route(
                'POST',
                f'/{MEMORY_INDEX}/_search',
                os_error(
                    400,
                    'search_phase_execution_exception',
                    'all shards failed',
                    phase='query',
                    grouped=True,
                    failed_shards=[
                        {
                            'shard': 0,
                            'index': MEMORY_INDEX,
                            'reason': {
                                'type': 'illegal_argument_exception',
                                'reason': 'No field found for [created_at] in mapping',
                            },
                        }
                    ],
                ),
                400,
            ),
        ),
    ),
    # Search before any memory was saved: the index check answers 404, no search runs.
    Case('SearchMemoryTool', 'index_missing', SEARCH_ARGS, (MEMORY_INDEX_MISSING,)),
    Case(
        'DeleteMemoryTool',
        'success',
        DELETE_ARGS,
        (
            Route(
                'DELETE',
                f'{MEMORY_DOC_PATH}/mem-7Kx2',
                {
                    '_index': MEMORY_INDEX,
                    '_id': 'mem-7Kx2',
                    '_version': 2,
                    'result': 'deleted',
                    '_shards': {'total': 2, 'successful': 1, 'failed': 0},
                    '_seq_no': 1,
                    '_primary_term': 1,
                },
            ),
        ),
    ),
    Case(
        'DeleteMemoryTool',
        'opensearch_error',
        DELETE_ARGS,
        (Route('DELETE', f'{MEMORY_DOC_PATH}/mem-7Kx2', index_not_found(MEMORY_INDEX), 404),),
    ),
)


# --- agentic memory tools --------------------------------------------------------------

CREATE_SESSION_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'session_id': 'custom_session_123',
    'summary': 'Session for user onboarding',
    'namespace': {'user_id': 'alice', 'agent_id': 'onboarding_bot'},
    'metadata': {'priority': 'high', 'category': 'onboarding'},
}
ADD_MEMORIES_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'messages': [
        {
            'role': 'user',
            'content': [{'text': "I'm Bob, I really like swimming.", 'type': 'text'}],
        },
        {
            'role': 'assistant',
            'content': [{'text': 'Cool, nice. Hope you enjoy your life.', 'type': 'text'}],
        },
    ],
    'namespace': {'user_id': 'bob'},
    'metadata': {
        'status': 'checkpoint',
        'branch': {'branch_name': 'high', 'root_event_id': '228nadfs879mtgk'},
    },
    'tags': {'topic': 'personal info'},
    'infer': True,
    'payload_type': 'conversational',
}
GET_MEMORY_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'type': 'working',
    'id': 'XyEuiJkBeh2gPPwzjYWM',
}
UPDATE_MEMORY_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'type': 'long-term',
    'id': 'DcxjTpkBvwXRq366C1Zz',
    'memory': "User's name is Bob Smith",
    'tags': {'topic': 'personal info', 'updated': 'true'},
}
DELETE_BY_ID_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'type': 'working',
    'id': 'XyEuiJkBeh2gPPwzjYWM',
}
DELETE_BY_QUERY_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'type': 'working',
    'query': {'match': {'owner_id': 'admin'}},
}
SEARCH_AGENTIC_ARGS = {
    'memory_container_id': CONTAINER_ID,
    'type': 'sessions',
    'query': {'match_all': {}},
    'sort': [{'created_time': {'order': 'desc'}}],
}

CONTAINER_NOT_FOUND = ml_status_exception(404, f'Memory container {CONTAINER_ID} not found')
MEMORY_NOT_FOUND = ml_status_exception(404, 'Memory XyEuiJkBeh2gPPwzjYWM not found')


def agentic(tool: str, arguments: dict, method: str, path: str, reply: Any, status: int, error):
    """The success and opensearch_error cases of one agentic memory tool."""
    return (
        Case(tool, 'success', arguments, (Route(method, path, reply),)),
        Case(tool, 'opensearch_error', arguments, (Route(method, path, error, status),)),
    )


AGENTIC_CASES = (
    *agentic(
        'CreateAgenticMemorySessionTool',
        CREATE_SESSION_ARGS,
        'POST',
        f'{CONTAINERS}/sessions',
        {'session_id': 'custom_session_123', 'status': 'created'},
        404,
        CONTAINER_NOT_FOUND,
    ),
    *agentic(
        'AddAgenticMemoriesTool',
        ADD_MEMORIES_ARGS,
        'POST',
        CONTAINERS,
        {'session_id': 'XSEuiJkBeh2gPPwzjYVh', 'working_memory_id': 'XyEuiJkBeh2gPPwzjYWM'},
        400,
        os_error(
            400,
            'illegal_argument_exception',
            'Namespace [user_id] does not match any strategy namespace of the memory container',
        ),
    ),
    *agentic(
        'GetAgenticMemoryTool',
        GET_MEMORY_ARGS,
        'GET',
        f'{CONTAINERS}/working/XyEuiJkBeh2gPPwzjYWM',
        {
            'memory_container_id': CONTAINER_ID,
            'payload_type': 'conversational',
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': "I'm Bob, I really like swimming.", 'type': 'text'}],
                }
            ],
            'namespace': {'user_id': 'bob'},
            'namespace_size': 1,
            'tags': {'topic': 'personal info'},
            'infer': True,
            'created_time': 1577836800000,
            'last_updated_time': 1577836800000,
        },
        404,
        MEMORY_NOT_FOUND,
    ),
    *agentic(
        'UpdateAgenticMemoryTool',
        UPDATE_MEMORY_ARGS,
        'PUT',
        f'{CONTAINERS}/long-term/DcxjTpkBvwXRq366C1Zz',
        {
            '_index': '.plugins-ml-am-default-memory-long-term',
            '_id': 'DcxjTpkBvwXRq366C1Zz',
            '_version': 2,
            'result': 'updated',
            '_shards': {'total': 1, 'successful': 1, 'failed': 0},
            '_seq_no': 5,
            '_primary_term': 1,
        },
        404,
        ml_status_exception(404, 'Memory DcxjTpkBvwXRq366C1Zz not found'),
    ),
    *agentic(
        'DeleteAgenticMemoryByIDTool',
        DELETE_BY_ID_ARGS,
        'DELETE',
        f'{CONTAINERS}/working/XyEuiJkBeh2gPPwzjYWM',
        {
            '_index': '.plugins-ml-am-default-memory-working',
            '_id': 'XyEuiJkBeh2gPPwzjYWM',
            '_version': 3,
            'result': 'deleted',
            '_shards': {'total': 1, 'successful': 1, 'failed': 0},
            '_seq_no': 6,
            '_primary_term': 1,
        },
        404,
        MEMORY_NOT_FOUND,
    ),
    *agentic(
        'DeleteAgenticMemoryByQueryTool',
        DELETE_BY_QUERY_ARGS,
        'POST',
        f'{CONTAINERS}/working/_delete_by_query',
        {
            'took': 12,
            'timed_out': False,
            'total': 2,
            'deleted': 2,
            'batches': 1,
            'version_conflicts': 0,
            'noops': 0,
            'retries': {'bulk': 0, 'search': 0},
            'throttled_millis': 0,
            'requests_per_second': -1.0,
            'throttled_until_millis': 0,
            'failures': [],
        },
        400,
        os_error(
            400,
            'parsing_exception',
            '[match] query malformed, no start_object after query name',
            line=1,
            col=19,
        ),
    ),
    *agentic(
        'SearchAgenticMemoryTool',
        SEARCH_AGENTIC_ARGS,
        'GET',
        f'{CONTAINERS}/sessions/_search',
        search_reply(
            '.plugins-ml-am-default-memory-sessions',
            [
                {
                    '_id': 'CcxjTpkBvwXRq366A1aE',
                    '_score': None,
                    '_source': {
                        'memory_container_id': CONTAINER_ID,
                        'summary': 'This is a test session',
                        'namespace': {'user_id': 'bob'},
                        'created_time': 1577836800000,
                        'last_updated_time': 1577836800000,
                    },
                    'sort': [1577836800000],
                }
            ],
        ),
        403,
        security_exception('cluster:admin/opensearch/ml/memory_containers/search', 'bob'),
    ),
)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def opensearch_version(fake_opensearch):
    fake_opensearch.version = OPENSEARCH_VERSION


def write_yaml(path, data: dict) -> str:
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def recorded(case: Case, result: dict, requests: list[dict]) -> dict:
    return {'tool': case.tool, 'arguments': case.arguments, 'result': result, 'requests': requests}


async def call_in_process(session, fake_opensearch, case: Case) -> dict:
    """The call result and the requests the call alone sent (not the startup ``GET /``)."""
    start = len(fake_opensearch.requests)
    result = await session.call_tool(case.tool, case.arguments)
    return recorded(case, result, fake_opensearch.requests[start:])


async def call_launched(session, case: Case) -> dict:
    """As ``call_in_process``, for a server launched by ``memory_enabled_session``."""
    await session.initialize()
    start = len(session.opensearch_requests())
    result = await session.call_tool(case.tool, case.arguments)
    return recorded(case, result, session.opensearch_requests()[start:])


async def listed_tools(session) -> dict[str, dict]:
    return {tool['name']: tool for tool in (await session.list_tools())['tools']}


# ---------------------------------------------------------------------------
# skills tools
# ---------------------------------------------------------------------------


async def test_skills_tools_are_listed_when_the_category_is_enabled(mcp_session):
    # Precondition of the skills cases: the env enables every SKILLS_TOOLS_REGISTRY tool.
    async with mcp_session(env=SKILLS_ENV) as session:
        assert set(SKILLS_TOOLS) <= set(await listed_tools(session))


# characterization: current behaviour. BufferedAsyncHttpConnection.perform_request
# (src/opensearch/connection.py) passes no content type to opensearch-py's error mapping, so
# in every opensearch_error case of this module (skills, memory and agentic memory) the
# error text reads like `NotFoundError(404, '<raw JSON body>')`, not with the OpenSearch
# error type and reason.
@pytest.mark.parametrize('case', SKILLS_CASES, ids=lambda case: case.id)
async def test_skills_tool_call(mcp_session, fake_opensearch, golden, normalize, case):
    install(fake_opensearch, case.routes)
    async with mcp_session(env=SKILLS_ENV) as session:
        result = await call_in_process(session, fake_opensearch, case)
    golden(f'{AREA}/{case.id}', normalize(result))


# ---------------------------------------------------------------------------
# memory tools
# ---------------------------------------------------------------------------


async def test_memory_tools_are_listed_when_enabled(memory_enabled_session):
    # Precondition of the memory cases: MEMORY_TOOLS_ENABLED=true registers all three.
    async with memory_enabled_session(env=NO_AUTH) as session:
        assert set(MEMORY_TOOLS) <= set(await listed_tools(session))


@pytest.mark.parametrize('case', MEMORY_CASES, ids=lambda case: case.id)
async def test_memory_tool_call(memory_enabled_session, fake_opensearch, golden, normalize, case):
    install(fake_opensearch, case.routes)
    async with memory_enabled_session(env={**NO_AUTH, **dict(case.env)}) as session:
        result = await call_launched(session, case)
    golden(f'{AREA}/{case.id}', normalize(result))


async def test_memory_tool_call_in_multi_mode(
    memory_enabled_session, fake_opensearch, golden, normalize, tmp_path
):
    # Multi mode leaves memory tools out of the enabled tools (tool_filter.get_tools), so a
    # call is answered as an unknown tool, without reaching OpenSearch.
    config = write_yaml(
        tmp_path / 'config.yml',
        {
            'version': '1.0',
            'clusters': {
                'fake-cluster': {'opensearch_url': fake_opensearch.url, 'opensearch_no_auth': True}
            },
        },
    )
    case = Case(
        'SaveMemoryTool',
        'multi_mode',
        {'opensearch_cluster_name': 'fake-cluster', **SAVE_ARGS},
        (),
    )
    async with memory_enabled_session(mode='multi', config_file=config) as session:
        result = await call_launched(session, case)
        listed = set(await listed_tools(session))
    golden(f'{AREA}/{case.id}', normalize(result))
    assert not listed & set(MEMORY_TOOLS)


# ---------------------------------------------------------------------------
# agentic memory tools
# ---------------------------------------------------------------------------


async def test_agentic_memory_tools_are_listed_when_the_category_is_enabled(mcp_session):
    # Precondition of the agentic memory cases: the env enables all seven tools.
    async with mcp_session(env=AGENTIC_ENV) as session:
        assert set(AGENTIC_MEMORY_TOOLS) <= set(await listed_tools(session))


# Every agentic memory tool calls check_tool_compatibility, which sends its own `GET /`
# before the tool's request.
@pytest.mark.parametrize('case', AGENTIC_CASES, ids=lambda case: case.id)
async def test_agentic_memory_tool_call(mcp_session, fake_opensearch, golden, normalize, case):
    install(fake_opensearch, case.routes)
    async with mcp_session(env=AGENTIC_ENV) as session:
        result = await call_in_process(session, fake_opensearch, case)
    golden(f'{AREA}/{case.id}', normalize(result))


GET_WITHOUT_CONTAINER = {'type': 'working', 'id': 'XyEuiJkBeh2gPPwzjYWM'}


# characterization: current behaviour. A memory_container_id in the config file's
# `agentic_memory` section becomes the schema default (tools/list shows it, no longer
# required), but the args model injects a default from OPENSEARCH_MEMORY_CONTAINER_ID only
# (get_memory_container_id_from_config() is called without the config path), so a call that
# relies on the advertised default fails validation without reaching OpenSearch.
async def test_agentic_memory_container_id_default_from_config_file(
    mcp_session, fake_opensearch, golden, normalize, tmp_path
):
    config = write_yaml(
        tmp_path / 'config.yml',
        {
            'version': '1.0',
            'agentic_memory': {'memory_container_id': CONTAINER_ID},
            'tool_filters': {'enabled_categories': ['agentic_memory']},
        },
    )
    # Would answer the call if the default reached the tool; the golden shows no request.
    install(
        fake_opensearch,
        (Route('GET', f'{CONTAINERS}/working/XyEuiJkBeh2gPPwzjYWM', {'payload_type': 'data'}),),
    )
    case = Case('GetAgenticMemoryTool', 'container_id_from_config_file', GET_WITHOUT_CONTAINER, ())
    async with mcp_session(config_file=config, env=NO_AUTH) as session:
        schema = (await listed_tools(session))[case.tool]['inputSchema']
        result = await call_in_process(session, fake_opensearch, case)
    result['listed_schema'] = {
        'memory_container_id': schema['properties']['memory_container_id'],
        'required': schema.get('required', []),
    }
    golden(f'{AREA}/{case.id}', normalize(result))


async def test_agentic_memory_container_id_default_from_env(
    mcp_session, fake_opensearch, golden, normalize
):
    # OPENSEARCH_MEMORY_CONTAINER_ID sets the schema default and the args model injects it.
    install(
        fake_opensearch,
        (
            Route(
                'GET',
                f'{CONTAINERS}/working/XyEuiJkBeh2gPPwzjYWM',
                {'memory_container_id': CONTAINER_ID, 'payload_type': 'data'},
            ),
        ),
    )
    case = Case('GetAgenticMemoryTool', 'container_id_from_env', GET_WITHOUT_CONTAINER, ())
    env = {**AGENTIC_ENV, 'OPENSEARCH_MEMORY_CONTAINER_ID': CONTAINER_ID}
    async with mcp_session(env=env) as session:
        schema = (await listed_tools(session))[case.tool]['inputSchema']
        result = await call_in_process(session, fake_opensearch, case)
    result['listed_schema'] = {
        'memory_container_id': schema['properties']['memory_container_id'],
        'required': schema.get('required', []),
    }
    golden(f'{AREA}/{case.id}', normalize(result))
