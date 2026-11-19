# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: ``tools/call`` for the Search Relevance Workbench (SRW) tools.

Each case calls one tool through the MCP protocol (``mcp_session``) and goldens the whole
call result (content, ``isError``, structured content when there is any) together with the
requests the call sent to the fake OpenSearch. Per tool there is one success case with a
canned reply modelled on ``tests/tools/test_query_set_tools.py``,
``test_experiment_tools.py``, ``test_judgment_tools.py`` and ``test_srw_search_tools.py``,
and one case where OpenSearch answers with a 4xx and an OpenSearch error body.

The tools are enabled the way a user enables them: ``OPENSEARCH_ENABLED_CATEGORIES`` names
the built-in ``search_relevance`` category. ``OPENSEARCH_NO_AUTH`` lets the server reach the
fake, and the fake reports OpenSearch 3.5.0, the newest ``min_version`` among these tools
(the ``Search*Tool`` search tools need 3.5.0, the others 3.1.0).
"""

import pytest
from dataclasses import dataclass
from typing import Any


SRW_OPENSEARCH_VERSION = '3.5.0'
SRW_ENV = {
    'OPENSEARCH_NO_AUTH': 'true',
    'OPENSEARCH_ENABLED_CATEGORIES': 'search_relevance',
}
SRW = '/_plugins/_search_relevance'

# Fixed dates far from today, so the normalizer keeps them as they are (see README.md).
CREATED = '2020-01-01T00:00:00Z'

SRW_TOOLS = (
    'GetQuerySetTool',
    'CreateQuerySetTool',
    'SampleQuerySetTool',
    'DeleteQuerySetTool',
    'GetExperimentTool',
    'CreateExperimentTool',
    'DeleteExperimentTool',
    'SearchQuerySetsTool',
    'SearchSearchConfigurationsTool',
    'SearchJudgmentsTool',
    'SearchExperimentsTool',
    'CreateSearchConfigurationTool',
    'GetSearchConfigurationTool',
    'DeleteSearchConfigurationTool',
    'GetJudgmentListTool',
    'CreateJudgmentListTool',
    'CreateUBIJudgmentListTool',
    'DeleteJudgmentListTool',
    'CreateLLMJudgmentListTool',
)


def os_error(status: int, error_type: str, reason: str, **details: Any) -> dict:
    """An OpenSearch error body, as the REST layer writes it."""
    cause = {'type': error_type, 'reason': reason, **details}
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


def security_exception(action: str) -> dict:
    return os_error(
        403,
        'security_exception',
        f'no permissions for [{action}] and User [name=srw-reader, backend_roles=[], '
        'requestedTenant=null]',
    )


def deleted(index: str, doc_id: str) -> dict:
    return {
        '_index': index,
        '_id': doc_id,
        '_version': 2,
        'result': 'deleted',
        '_shards': {'total': 1, 'successful': 1, 'failed': 0},
        '_seq_no': 7,
        '_primary_term': 1,
    }


def search_reply(index: str, hits: list[tuple[str, dict]]) -> dict:
    return {
        'took': 3,
        'timed_out': False,
        '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0},
        'hits': {
            'total': {'value': len(hits), 'relation': 'eq'},
            'max_score': 1.0,
            'hits': [
                {'_index': index, '_id': doc_id, '_score': 1.0, '_source': source}
                for doc_id, source in hits
            ],
        },
    }


@dataclass(frozen=True)
class Case:
    """One SRW tool: its arguments, the request it sends and the two canned replies."""

    tool: str
    arguments: dict
    method: str
    path: str
    reply: Any
    error_status: int
    error: dict


CASES = (
    Case(
        tool='GetQuerySetTool',
        arguments={'query_set_id': 'qs-laptops'},
        method='GET',
        path=f'{SRW}/query_sets/qs-laptops',
        reply={
            '_id': 'qs-laptops',
            '_source': {
                'id': 'qs-laptops',
                'name': 'my-query-set',
                'description': 'Test queries',
                'sampling': 'manual',
                'timestamp': CREATED,
                'querySetQueries': [{'queryText': 'laptop'}, {'queryText': 'gaming laptop'}],
            },
        },
        error_status=404,
        error=os_error(404, 'resource_not_found_exception', 'QuerySet not found: qs-laptops'),
    ),
    Case(
        tool='CreateQuerySetTool',
        # A plain string, a {"queryText"} object and a number: the three branches of
        # create_query_set; no description, so the server writes its default one.
        arguments={
            'name': 'product-queries',
            'queries': '["laptop", {"queryText": "wireless headphones"}, 42]',
        },
        method='PUT',
        path=f'{SRW}/query_sets',
        reply={'_id': 'qs-new', 'result': 'created'},
        error_status=400,
        error=os_error(
            400,
            'illegal_argument_exception',
            'Query set with name [product-queries] already exists',
        ),
    ),
    Case(
        tool='SampleQuerySetTool',
        arguments={'name': 'top-queries', 'query_set_size': 20, 'sampling': 'random'},
        method='POST',
        path=f'{SRW}/query_sets',
        reply={'_id': 'qs-sampled', 'result': 'created'},
        error_status=404,
        error=index_not_found('ubi_queries'),
    ),
    Case(
        tool='DeleteQuerySetTool',
        arguments={'query_set_id': 'qs-laptops'},
        method='DELETE',
        path=f'{SRW}/query_sets/qs-laptops',
        reply=deleted('search-relevance-queryset', 'qs-laptops'),
        error_status=404,
        error=os_error(404, 'resource_not_found_exception', 'QuerySet not found: qs-laptops'),
    ),
    Case(
        tool='GetExperimentTool',
        arguments={'experiment_id': 'exp-abc123'},
        method='GET',
        path=f'{SRW}/experiments/exp-abc123',
        reply={
            '_id': 'exp-abc123',
            '_source': {
                'id': 'exp-abc123',
                'type': 'PAIRWISE_COMPARISON',
                'status': 'COMPLETED',
                'querySetId': 'qs-laptops',
                'searchConfigurationList': ['sc-baseline', 'sc-candidate'],
                'size': 10,
                'timestamp': CREATED,
                'results': [
                    {
                        'queryText': 'laptop',
                        'metrics': [{'metric': 'jaccard', 'value': 0.43}],
                    }
                ],
            },
        },
        error_status=404,
        error=os_error(404, 'resource_not_found_exception', 'Experiment not found: exp-abc123'),
    ),
    Case(
        tool='CreateExperimentTool',
        arguments={
            'query_set_id': 'qs-laptops',
            'search_configuration_ids': '["sc-baseline"]',
            'experiment_type': 'POINTWISE_EVALUATION',
            'judgment_list_ids': '["jl-manual"]',
        },
        method='PUT',
        path=f'{SRW}/experiments',
        reply={'_id': 'exp-new', 'result': 'created'},
        error_status=400,
        error=os_error(400, 'illegal_argument_exception', 'Failed to find query set: qs-laptops'),
    ),
    Case(
        tool='DeleteExperimentTool',
        arguments={'experiment_id': 'exp-abc123'},
        method='DELETE',
        path=f'{SRW}/experiments/exp-abc123',
        reply=deleted('search-relevance-experiment', 'exp-abc123'),
        error_status=403,
        error=security_exception('cluster:admin/opensearch/search_relevance/experiment/delete'),
    ),
    Case(
        tool='SearchQuerySetsTool',
        # No query_body: the server sends its default match_all query.
        arguments={},
        method='POST',
        path=f'{SRW}/query_sets/_search',
        reply=search_reply(
            'search-relevance-queryset',
            [
                (
                    'qs-laptops',
                    {
                        'id': 'qs-laptops',
                        'name': 'my-query-set',
                        'sampling': 'manual',
                        'timestamp': CREATED,
                        'querySetQueries': [{'queryText': 'laptop'}],
                    },
                )
            ],
        ),
        error_status=404,
        error=index_not_found('search-relevance-queryset'),
    ),
    Case(
        tool='SearchSearchConfigurationsTool',
        # query_body as a JSON string: the server validates and parses it.
        arguments={'query_body': '{"query": {"match": {"name": "baseline"}}, "size": 20}'},
        method='POST',
        path=f'{SRW}/search_configurations/_search',
        reply=search_reply(
            'search-relevance-search-config',
            [
                (
                    'sc-baseline',
                    {
                        'id': 'sc-baseline',
                        'name': 'baseline',
                        'index': 'products',
                        'query': '{"query":{"match":{"title":"%SearchText%"}}}',
                        'timestamp': CREATED,
                    },
                )
            ],
        ),
        error_status=400,
        error=os_error(
            400,
            'parsing_exception',
            "[match] query doesn't support multiple fields, found [name] and [index]",
            line=1,
            col=48,
        ),
    ),
    Case(
        tool='SearchJudgmentsTool',
        # query_body as a JSON object.
        arguments={'query_body': {'query': {'term': {'type': 'IMPORT_JUDGMENT'}}}},
        method='POST',
        path=f'{SRW}/judgments/_search',
        reply=search_reply(
            'search-relevance-judgment',
            [
                (
                    'jl-manual',
                    {
                        'id': 'jl-manual',
                        'name': 'manual-judgments',
                        'type': 'IMPORT_JUDGMENT',
                        'status': 'COMPLETED',
                        'timestamp': CREATED,
                    },
                )
            ],
        ),
        error_status=400,
        error=os_error(
            400,
            'x_content_parse_exception',
            '[1:20] [term] unknown field [typo]',
        ),
    ),
    Case(
        tool='SearchExperimentsTool',
        arguments={'query_body': {'query': {'match': {'status': 'COMPLETED'}}, 'size': 5}},
        method='POST',
        path=f'{SRW}/experiments/_search',
        reply=search_reply(
            'search-relevance-experiment',
            [
                (
                    'exp-abc123',
                    {
                        'id': 'exp-abc123',
                        'type': 'PAIRWISE_COMPARISON',
                        'status': 'COMPLETED',
                        'querySetId': 'qs-laptops',
                        'timestamp': CREATED,
                    },
                )
            ],
        ),
        error_status=403,
        error=security_exception('indices:data/read/search'),
    ),
    Case(
        tool='CreateSearchConfigurationTool',
        arguments={
            'name': 'baseline',
            'index': 'products',
            'query': '{"query":{"multi_match":{"query":"%SearchText%",'
            '"fields":["title","description"]}}}',
        },
        method='PUT',
        path=f'{SRW}/search_configurations',
        reply={'_id': 'sc-baseline', 'result': 'created'},
        error_status=400,
        error=os_error(
            400,
            'illegal_argument_exception',
            'Search configuration with name [baseline] already exists',
        ),
    ),
    Case(
        tool='GetSearchConfigurationTool',
        arguments={'search_configuration_id': 'sc-baseline'},
        method='GET',
        path=f'{SRW}/search_configurations/sc-baseline',
        reply={
            '_id': 'sc-baseline',
            '_source': {
                'id': 'sc-baseline',
                'name': 'baseline',
                'index': 'products',
                'query': '{"query":{"match":{"title":"%SearchText%"}}}',
                'timestamp': CREATED,
            },
        },
        error_status=404,
        error=os_error(
            404, 'resource_not_found_exception', 'SearchConfiguration not found: sc-baseline'
        ),
    ),
    Case(
        tool='DeleteSearchConfigurationTool',
        arguments={'search_configuration_id': 'sc-baseline'},
        method='DELETE',
        path=f'{SRW}/search_configurations/sc-baseline',
        reply=deleted('search-relevance-search-config', 'sc-baseline'),
        error_status=404,
        error=os_error(
            404, 'resource_not_found_exception', 'SearchConfiguration not found: sc-baseline'
        ),
    ),
    # characterization: current behaviour. The success text reads "Judgment list: jl-manual:"
    # (two colons), unlike the other Get tools ("Query set qs-laptops:").
    Case(
        tool='GetJudgmentListTool',
        arguments={'judgment_id': 'jl-manual'},
        method='GET',
        path=f'{SRW}/judgments/jl-manual',
        reply={
            '_id': 'jl-manual',
            '_source': {
                'id': 'jl-manual',
                'name': 'manual-judgments',
                'type': 'IMPORT_JUDGMENT',
                'status': 'COMPLETED',
                'timestamp': CREATED,
                'judgmentRatings': [
                    {
                        'query': 'laptop',
                        'ratings': [
                            {'docId': 'doc1', 'rating': '3'},
                            {'docId': 'doc2', 'rating': '1'},
                        ],
                    }
                ],
            },
        },
        error_status=404,
        error=os_error(404, 'resource_not_found_exception', 'Judgment not found: jl-manual'),
    ),
    Case(
        tool='CreateJudgmentListTool',
        arguments={
            'name': 'manual-judgments',
            'description': 'Manual relevance judgments for product search',
            'judgment_ratings': '[{"query": "laptop", "ratings": '
            '[{"docId": "doc1", "rating": 3}, {"docId": "doc2", "rating": 1}]}]',
        },
        method='PUT',
        path=f'{SRW}/judgments',
        reply={'_id': 'jl-manual', 'result': 'created'},
        error_status=400,
        error=os_error(
            400,
            'illegal_argument_exception',
            'Invalid judgmentRatings: rating must be a number for docId [doc1]',
        ),
    ),
    Case(
        tool='CreateUBIJudgmentListTool',
        arguments={
            'name': 'ubi-judgments',
            'click_model': 'coec',
            'max_rank': 20,
            'start_date': '2020-01-01',
            'end_date': '2020-01-31',
        },
        method='PUT',
        path=f'{SRW}/judgments',
        reply={'_id': 'jl-ubi', 'result': 'created', 'status': 'PROCESSING'},
        error_status=404,
        error=index_not_found('ubi_events'),
    ),
    Case(
        tool='DeleteJudgmentListTool',
        arguments={'judgment_id': 'jl-manual'},
        method='DELETE',
        path=f'{SRW}/judgments/jl-manual',
        reply=deleted('search-relevance-judgment', 'jl-manual'),
        error_status=403,
        error=security_exception('cluster:admin/opensearch/search_relevance/judgment/delete'),
    ),
    Case(
        tool='CreateLLMJudgmentListTool',
        # The fake answers for the plugin: no LLM is called.
        arguments={
            'name': 'llm-judgments',
            'query_set_id': 'qs-laptops',
            'search_configuration_id': 'sc-baseline',
            'model_id': 'ml-model-1',
            'size': 5,
            'context_fields': '["title", "description"]',
        },
        method='PUT',
        path=f'{SRW}/judgments',
        reply={'_id': 'jl-llm', 'result': 'created', 'status': 'PROCESSING'},
        error_status=400,
        error=os_error(400, 'illegal_argument_exception', 'Failed to find model: ml-model-1'),
    ),
)

assert tuple(case.tool for case in CASES) == SRW_TOOLS


@pytest.fixture
def srw_session(mcp_session, fake_opensearch):
    """Factory for a server with the SRW tools enabled by env, used as an async context."""
    fake_opensearch.version = SRW_OPENSEARCH_VERSION
    return lambda: mcp_session(env=SRW_ENV)


async def call_and_record(session, fake_opensearch, case: Case) -> dict:
    """The call result and the requests the call alone sent (not the startup ``GET /``)."""
    start = len(fake_opensearch.requests)
    result = await session.call_tool(case.tool, case.arguments)
    return {
        'tool': case.tool,
        'arguments': case.arguments,
        'result': result,
        'requests': fake_opensearch.requests[start:],
    }


async def test_srw_tools_are_listed_when_the_category_is_enabled(srw_session):
    # Precondition of every case below: the env enables all 19 SRW tools.
    async with srw_session() as session:
        listed = {tool['name'] for tool in (await session.list_tools())['tools']}
    assert set(SRW_TOOLS) <= listed


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.tool)
async def test_call_tool_success(srw_session, fake_opensearch, golden, normalize, case):
    fake_opensearch.reply(case.method, case.path, case.reply)
    async with srw_session() as session:
        recorded = await call_and_record(session, fake_opensearch, case)
    golden(f'call_tool_srw/{case.tool}/success', normalize(recorded))


# characterization: current behaviour. BufferedAsyncHttpConnection.perform_request
# (src/opensearch/connection.py) calls `self._raise_error(status, raw_data)` without the
# response content type, so opensearch-py reads the body as text/plain: the exception's
# `error` is the raw JSON body instead of the OpenSearch error type, and its `info` is None.
# The error text below therefore reads `RequestError(400, '{"error":...}')`, not
# `RequestError(400, 'illegal_argument_exception', ...)`.
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.tool)
async def test_call_tool_opensearch_error(srw_session, fake_opensearch, golden, normalize, case):
    fake_opensearch.error(case.method, case.path, case.error_status, case.error)
    async with srw_session() as session:
        recorded = await call_and_record(session, fake_opensearch, case)
    golden(f'call_tool_srw/{case.tool}/opensearch_error', normalize(recorded))
