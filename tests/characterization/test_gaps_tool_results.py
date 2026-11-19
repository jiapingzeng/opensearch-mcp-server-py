# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: tool result shapes and endpoint edge cases.

Cases the coverage checkpoint found unpinned, around the code the planned
``tool_generator.py`` / ``generic_api_tool.py`` fixes change. Each case drives the server
through ``mcp_session`` and ``tools/call`` and goldens, under ``golden/gaps_tool_results/``,
the call (name and arguments), the whole call result and the ``fake_opensearch`` request log.

- ``error_body/<Tool>``: a ``tools/tools.py`` tool when OpenSearch answers HTTP 200 with a
  body ``{"error": ...}`` (the tools' ``'error' in result`` branch).
- ``empty_list/<Tool>``: a ``_cat`` tool when OpenSearch answers ``[]``.
- ``generic_api/plain_text``: ``GenericOpenSearchApiTool`` passing a ``text/plain`` reply
  through unformatted.
- ``non_utf8/...``: ``BufferedAsyncHttpConnection`` given a response body that is not UTF-8
  (``opensearch/connection.py``: the ``UnicodeDecodeError`` fallback).
- ``generated/...``: the tools ``tool_generator`` builds: a whitespace-only ``body``, the
  endpoint-selection fallback, and a path parameter the tool does not fill in.

Tools are enabled the way a user enables them (``OPENSEARCH_ENABLED_TOOLS``). Canned replies
use fixed dates far from today (see README.md), so the normalizer leaves them alone.
"""

import pytest
import yaml
from dataclasses import dataclass, field
from typing import Any


AREA = 'gaps_tool_results'

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}


def enable(*tools: str) -> dict:
    """Environment that enables ``tools`` on top of the default ``core_tools`` category."""
    return {'OPENSEARCH_ENABLED_TOOLS': ','.join(tools)}


# ---------------------------------------------------------------------------
# harness for this module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    body: Any = None
    status: int = 200
    headers: dict | None = None


@dataclass(frozen=True)
class Case:
    """One ``tools/call``: golden name (below ``gaps_tool_results/``), tool, arguments, routes."""

    name: str
    tool: str
    arguments: dict
    routes: tuple = ()
    env: dict = field(default_factory=dict)
    is_error: bool = False


@pytest.fixture
def run_case(mcp_session, fake_opensearch, golden, normalize):
    """Add the case's routes, build the server with its environment, call, golden."""

    async def run_case(case: Case) -> dict:
        for route in case.routes:
            fake_opensearch.reply(
                route.method, route.path, route.body, status=route.status, headers=route.headers
            )
        async with mcp_session(env={**NO_AUTH, **case.env}) as session:
            result = await session.call_tool(case.tool, case.arguments)
        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError') is case.is_error, result
        data = {
            'call': {'name': case.tool, 'arguments': case.arguments},
            'result': result,
            'requests': fake_opensearch.requests,
        }
        golden(f'{AREA}/{case.name}', normalize(data))
        return result

    return run_case


# ---------------------------------------------------------------------------
# 1. tools/tools.py: HTTP 200 with an `error` body, and `[]`
# ---------------------------------------------------------------------------

# What a proxy or plugin in front of OpenSearch may answer with a 200 status.
ERROR_BODY = {'error': 'fake: error reported with HTTP 200'}
# The same, shaped as an OpenSearch error body: the tools turn the `error` value into
# `Exception(value)`, so a dict shows up as its Python repr.
OPENSEARCH_SHAPED_ERROR_BODY = {
    'error': {
        'root_cause': [{'type': 'fake_error', 'reason': 'fake: error reported with HTTP 200'}],
        'type': 'fake_error',
        'reason': 'fake: error reported with HTTP 200',
    },
    'status': 200,
}

ERROR_BODY_CASES = [
    Case(
        'error_body/GetShardsTool',
        'GetShardsTool',
        {'index': 'logs'},
        (Route('GET', '/_cat/shards/logs', ERROR_BODY),),
        is_error=True,
    ),
    Case(
        'error_body/GetShardsTool_opensearch_shaped',
        'GetShardsTool',
        {'index': 'logs'},
        (Route('GET', '/_cat/shards/logs', OPENSEARCH_SHAPED_ERROR_BODY),),
        is_error=True,
    ),
    Case(
        'error_body/GetSegmentsTool',
        'GetSegmentsTool',
        {'index': 'logs'},
        (Route('GET', '/_cat/segments/logs', ERROR_BODY),),
        env=enable('GetSegmentsTool'),
        is_error=True,
    ),
    Case(
        'error_body/CatNodesTool',
        'CatNodesTool',
        {},
        (Route('GET', '/_cat/nodes', ERROR_BODY),),
        env=enable('CatNodesTool'),
        is_error=True,
    ),
    Case(
        'error_body/GetAllocationTool',
        'GetAllocationTool',
        {},
        (Route('GET', '/_cat/allocation', ERROR_BODY),),
        env=enable('GetAllocationTool'),
        is_error=True,
    ),
    Case(
        'error_body/GetNodesTool',
        'GetNodesTool',
        {},
        (Route('GET', '/_nodes', ERROR_BODY),),
        env=enable('GetNodesTool'),
        is_error=True,
    ),
    # The `limit` slice applies to a list only, so the dict reaches the error branch whole.
    Case(
        'error_body/GetLongRunningTasksTool',
        'GetLongRunningTasksTool',
        {'limit': 2},
        (Route('GET', '/_cat/tasks', ERROR_BODY),),
        env=enable('GetLongRunningTasksTool'),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', ERROR_BODY_CASES, ids=lambda case: case.name)
async def test_tool_error_body_with_http_200(run_case, case):
    """Gap 1: GetShards, GetSegments, CatNodes, GetAllocation, GetNodes, GetLongRunningTasks
    answering HTTP 200 with a body ``{"error": ...}``."""
    await run_case(case)


EMPTY_LIST_CASES = [
    Case(
        'empty_list/CatNodesTool',
        'CatNodesTool',
        {},
        (Route('GET', '/_cat/nodes', []),),
        env=enable('CatNodesTool'),
    ),
    Case(
        'empty_list/GetAllocationTool',
        'GetAllocationTool',
        {},
        (Route('GET', '/_cat/allocation', []),),
        env=enable('GetAllocationTool'),
    ),
    Case(
        'empty_list/GetLongRunningTasksTool',
        'GetLongRunningTasksTool',
        {},
        (Route('GET', '/_cat/tasks', []),),
        env=enable('GetLongRunningTasksTool'),
    ),
]


@pytest.mark.parametrize('case', EMPTY_LIST_CASES, ids=lambda case: case.name)
async def test_tool_empty_list(run_case, case):
    """Gap 1: CatNodes, GetAllocation and GetLongRunningTasks answering ``[]``."""
    await run_case(case)


# ---------------------------------------------------------------------------
# 2. generic_api_tool.py: a plain-text response
# ---------------------------------------------------------------------------

CAT_HEALTH_TEXT = (
    'epoch      timestamp cluster      status node.total node.data shards pri relo init\n'
    '1577836800 00:00:00  fake-cluster green           1         1      2   2    0    0\n'
)


async def test_generic_api_plain_text_response(run_case):
    """Gap 2: a ``text/plain`` reply reaches the tool as a str and is passed through as is."""
    await run_case(
        Case(
            'generic_api/plain_text',
            'GenericOpenSearchApiTool',
            {'path': '/_cat/health', 'query_params': {'v': 'true'}},
            (Route('GET', '/_cat/health', CAT_HEALTH_TEXT),),
        )
    )


# ---------------------------------------------------------------------------
# 3. opensearch/connection.py: a response body that is not UTF-8
# ---------------------------------------------------------------------------

# Latin-1 'café' and a stray 0xff: not valid UTF-8.
NOT_UTF8_TEXT = b'caf\xe9 \xff\n'
NOT_UTF8_JSON = b'{"name": "caf\xe9"}'

# characterization: current behaviour. A body that does not decode as UTF-8 becomes the
# Python repr of the bytes (`str(b'...')`, so "b'caf\\xe9 \\xff\\n'"), which opensearch-py
# then deserializes by content type: text/plain passes the repr through as the response,
# application/json fails to parse it, and an error status carries it as the error text.
NON_UTF8_CASES = [
    Case(
        'non_utf8/text_plain',
        'GenericOpenSearchApiTool',
        {'path': '/_cat/health'},
        (
            Route(
                'GET',
                '/_cat/health',
                NOT_UTF8_TEXT,
                headers={'content-type': 'text/plain; charset=ISO-8859-1'},
            ),
        ),
    ),
    Case(
        'non_utf8/application_json',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_doc/1'},
        (
            Route(
                'GET',
                '/logs/_doc/1',
                NOT_UTF8_JSON,
                headers={'content-type': 'application/json'},
            ),
        ),
        is_error=True,
    ),
    Case(
        'non_utf8/error_status',
        'GenericOpenSearchApiTool',
        {'path': '/logs/_doc/1'},
        (
            Route(
                'GET',
                '/logs/_doc/1',
                NOT_UTF8_JSON,
                status=400,
                headers={'content-type': 'application/json'},
            ),
        ),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', NON_UTF8_CASES, ids=lambda case: case.name)
async def test_response_body_not_utf8(run_case, case):
    """Gap 3: the connection's ``UnicodeDecodeError`` fallback, as a tool result."""
    await run_case(case)


# ---------------------------------------------------------------------------
# 4. tool_generator.py: whitespace-only body, endpoint fallback, unfilled path parameter
# ---------------------------------------------------------------------------

EXPLANATION = {
    '_index': 'logs',
    '_id': '1',
    'matched': False,
    'explanation': {'value': 0.0, 'description': 'no matching term', 'details': []},
}
COUNT = {'count': 42, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}}
CLUSTER_HEALTH = {
    'cluster_name': 'fake-cluster',
    'status': 'green',
    'timed_out': False,
    'number_of_nodes': 1,
    'number_of_data_nodes': 1,
    'active_primary_shards': 2,
    'active_shards': 2,
    'relocating_shards': 0,
    'initializing_shards': 0,
    'unassigned_shards': 0,
}

# characterization: current behaviour. A `body` that is only whitespace is dropped (no
# request body is sent), even for ExplainTool, whose `body` the input schema requires.
WHITESPACE_BODY_CASES = [
    Case(
        'generated/whitespace_body/CountTool',
        'CountTool',
        {'body': '   '},
        (Route('GET', '/_count', COUNT),),
    ),
    Case(
        'generated/whitespace_body/ExplainTool',
        'ExplainTool',
        {'index': 'logs', 'id': '1', 'body': ' \n\t '},
        (Route('GET', '/logs/_explain/1', EXPLANATION),),
    ),
]


@pytest.mark.parametrize('case', WHITESPACE_BODY_CASES, ids=lambda case: case.name)
async def test_generated_tool_whitespace_only_body(run_case, case):
    """Gap 4: ``process_body`` with a whitespace-only string ``body``."""
    await run_case(case)


# characterization: current behaviour. When no endpoint has all its path parameters filled
# (ExplainTool has only `/{index}/_explain/{id}`, GET first), the fallback takes the first
# endpoint and still substitutes every path parameter: one left out becomes the text "None"
# in the path, an empty one an empty segment, and the leading slashes that leaves collapse.
# The input schema lists `index` and `id` as required, but nothing enforces it: the args
# model gives every generated parameter the default None.
ENDPOINT_FALLBACK_CASES = [
    Case(
        'generated/endpoint_fallback/ExplainTool_missing_id',
        'ExplainTool',
        {'index': 'logs', 'body': {'query': {'match_all': {}}}},
        (Route('GET', '/logs/_explain/None', EXPLANATION),),
    ),
    Case(
        'generated/endpoint_fallback/ExplainTool_empty_id',
        'ExplainTool',
        {'index': 'logs', 'id': '', 'body': {'query': {'match_all': {}}}},
        (Route('GET', '/logs/_explain/', EXPLANATION),),
    ),
    Case(
        'generated/endpoint_fallback/ExplainTool_empty_index',
        'ExplainTool',
        {'index': '', 'id': '1', 'body': {'query': {'match_all': {}}}},
        (Route('GET', '/_explain/1', EXPLANATION),),
    ),
]


@pytest.mark.parametrize('case', ENDPOINT_FALLBACK_CASES, ids=lambda case: case.name)
async def test_generated_tool_endpoint_selection_fallback(run_case, case):
    """Gap 4: ``select_endpoint``'s fallback when no endpoint has its path parameters."""
    await run_case(case)


@pytest.fixture
def body_path_param_spec(pinned_openapi, tmp_path):
    """Serve the pinned ``_core.yaml`` and a ``cluster.yaml`` whose ``cluster.health`` group
    has a path parameter named ``body``.

    ``tool_func`` takes ``body`` (and the connection arguments) out of the call's parameters
    before it fills the path in, so a path parameter with that name is the one way a path
    parameter can be missing from them: every other one is there, as ``None`` when left out.
    """
    spec_dir = tmp_path / 'openapi'
    spec_dir.mkdir()
    core = (pinned_openapi.directory / '_core.yaml').read_text(encoding='utf-8')
    with open(spec_dir / '_core.yaml', 'w', encoding='utf-8', newline='\n') as f:
        f.write(core)
    health = {
        'x-operation-group': 'cluster.health',
        'x-version-added': '1.0',
        'description': 'Returns basic information about the health of the cluster.',
        'responses': {'200': {'description': 'OK'}},
    }
    cluster = {
        'openapi': '3.1.0',
        'info': {'title': 'OpenSearch Cluster API (body path parameter)', 'version': '1.0.0'},
        'paths': {
            '/_cluster/health': {'get': {**health, 'operationId': 'cluster.health.0'}},
            '/_cluster/health/{body}': {
                'get': {
                    **health,
                    'operationId': 'cluster.health.1',
                    'parameters': [
                        {
                            'name': 'body',
                            'in': 'path',
                            'required': True,
                            'description': 'A path parameter that shares its name with body.',
                            'schema': {'type': 'string'},
                        }
                    ],
                }
            },
        },
    }
    with open(spec_dir / 'cluster.yaml', 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(cluster, f, sort_keys=False)
    pinned_openapi.directory = spec_dir
    return pinned_openapi


# characterization: current behaviour. A path parameter named `body` is never put in the
# path: the tool takes its value as the request body, selects the endpoint without it
# (`/_cluster/health`) and skips it when filling the path in.
UNFILLED_PATH_PARAM_CASES = [
    Case(
        'generated/unfilled_path_param/body_left_out',
        'ClusterHealthTool',
        {},
        (Route('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
    Case(
        'generated/unfilled_path_param/body_given',
        'ClusterHealthTool',
        {'body': '"logs"'},
        (Route('GET', '/_cluster/health', CLUSTER_HEALTH),),
    ),
]


@pytest.mark.parametrize('case', UNFILLED_PATH_PARAM_CASES, ids=lambda case: case.name)
async def test_generated_tool_unfilled_path_parameter(run_case, body_path_param_spec, case):
    """Gap 4: a path parameter missing from the call's parameters when the path is filled."""
    await run_case(case)
