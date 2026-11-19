# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: ``LogPatternAnalysisTool`` in insight mode (no baseline).

Insight mode (``_log_insight`` in ``src/tools/analysis/log_pattern_analysis.py``) runs one
PPL query whose ``match(<field>, '<keywords>')`` argument is ``' '.join(ERROR_KEYWORDS)``.
``ERROR_KEYWORDS`` (``src/tools/analysis/constants.py``) is a set, so the order of the words,
and with it the request text, changes with ``PYTHONHASHSEED``.
``test_call_tool_memory_skills.py`` pins diff mode only for that reason.

This module pins insight mode's request and result through a test-local normalizer rule,
``sort_error_keywords``: it finds the ``match(...)`` argument in each recorded PPL query,
asserts that its word set is exactly ``ERROR_KEYWORDS`` (no word missing, added or
repeated), and writes the words sorted. Then the harness ``normalize`` runs as usual. The
fake's PPL route is a fixed reply on the path: it does not read the query, so the reply
does not depend on the keyword order either.

Each case is pinned three ways, against the same golden under ``golden/gaps_skills/``: in
process (the pytest process's own hash seed), and on a server launched over stdio with
``PYTHONHASHSEED`` set to each of ``HASH_SEEDS``. The launched test also checks that the
raw keyword order differs between those seeds, so the seeds did take effect and the rule
is what makes the goldens compare the same.

The skills tools are enabled as a user enables them (``OPENSEARCH_ENABLED_CATEGORIES=skills``)
and ``OPENSEARCH_NO_AUTH`` lets the server reach the fake, which reports OpenSearch 3.3.0.
Canned data uses fixed dates in 2020, far from today (see README.md).
"""

import copy
import pytest
import re
from dataclasses import dataclass
from tools.analysis.constants import ERROR_KEYWORDS
from typing import Any


AREA = 'gaps_skills'
OPENSEARCH_VERSION = '3.3.0'
TOOL = 'LogPatternAnalysisTool'
PPL_PATH = '/_plugins/_ppl'

# Without auth configured the server takes the AWS path and fails it before the fake.
SKILLS_ENV = {'OPENSEARCH_NO_AUTH': 'true', 'OPENSEARCH_ENABLED_CATEGORIES': 'skills'}

# PYTHONHASHSEED values the launched server runs under (at least two, per the task).
HASH_SEEDS = ('0', '1')


# ---------------------------------------------------------------------------
# normalizer rule (this module only)
# ---------------------------------------------------------------------------

# Insight mode's keyword filter: `match(<field>, '<keywords>')`, the words joined by one space.
KEYWORDS_MATCH = re.compile(r"match\((?P<field>[^,()]+), '(?P<keywords>[^']*)'\)")


def keyword_words(query: str) -> list[str]:
    """The words of the one ``match(<field>, '<keywords>')`` argument in ``query``, in order."""
    matches = KEYWORDS_MATCH.findall(query)
    assert len(matches) == 1, (
        f"expected exactly one match(<field>, '<keywords>') in the PPL query: {query!r}"
    )
    return matches[0][1].split(' ')


def sort_error_keywords(query: str) -> str:
    """``query`` with its ``match(<field>, '<keywords>')`` words, ``ERROR_KEYWORDS``, sorted.

    Fails if the words are not exactly ``ERROR_KEYWORDS``: a missing, extra or repeated word
    is a behaviour change, not hash-seed noise.
    """
    words = keyword_words(query)
    assert len(words) == len(set(words)), f'repeated keyword in match(): {words}'
    assert set(words) == ERROR_KEYWORDS, (
        'match() keywords differ from ERROR_KEYWORDS: '
        f'missing {sorted(ERROR_KEYWORDS - set(words))}, '
        f'extra {sorted(set(words) - ERROR_KEYWORDS)}'
    )
    match = KEYWORDS_MATCH.search(query)
    start, end = match.span('keywords')
    return query[:start] + ' '.join(sorted(words)) + query[end:]


def sort_request_keywords(requests: list[dict]) -> list[dict]:
    """Copy of ``requests`` with ``sort_error_keywords`` applied to every PPL query."""
    requests = copy.deepcopy(requests)
    for request in requests:
        if request['path'] == PPL_PATH:
            request['body']['query'] = sort_error_keywords(request['body']['query'])
    return requests


# ---------------------------------------------------------------------------
# canned OpenSearch replies
# ---------------------------------------------------------------------------


def ppl_insights(rows: list[list[Any]]) -> dict:
    """A PPL (jdbc format) reply for insight mode's pattern query."""
    return {
        'schema': [
            {'name': 'patterns_field', 'type': 'string'},
            {'name': 'pattern_count', 'type': 'bigint'},
            {'name': 'sample_logs', 'type': 'array'},
        ],
        'datarows': rows,
        'total': len(rows),
        'size': len(rows),
        'status': 200,
    }


PPL_INSIGHTS = ppl_insights(
    [
        [
            'Connection timeout to <*>',
            300,
            [
                '2020-01-01 10:05:00 ERROR Connection timeout to db-1.internal',
                '2020-01-01 10:06:00 ERROR Connection timeout to db-2.internal',
            ],
        ],
        # characterization: current behaviour. Insight mode returns patterns as PPL sends
        # them: the run of wildcards is not collapsed into one `<*>` as diff mode does.
        [
            'NullPointerException at <*> <*>',
            50,
            ['2020-01-01 10:07:00 FATAL NullPointerException at OrderService.java:42 checkout'],
        ],
        [
            'Request <*> failed: invalid token',
            12,
            ['2020-01-01 10:08:00 WARN Request req-1 failed: invalid token'],
        ],
        # characterization: current behaviour. A row without exactly three columns is
        # dropped without a word.
        ['Disk <*> degraded', 3],
    ]
)
PPL_NO_INSIGHTS = ppl_insights([])

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

INSIGHT_ARGS = {
    'index': 'logs-app',
    'logFieldName': 'message',
    'timeField': '@timestamp',
    'selectionTimeRangeStart': '2020-01-01 10:00:00',
    'selectionTimeRangeEnd': '2020-01-01 10:30:00',
}


@dataclass(frozen=True)
class Case:
    """One insight-mode call: its arguments and the fake's fixed reply to the PPL query."""

    name: str
    arguments: dict
    reply: Any
    status: int = 200


CASES = (
    Case('insight', INSIGHT_ARGS, PPL_INSIGHTS),
    # The filter goes into the query as its own `where` stage, before the keyword match.
    Case(
        'insight_with_filter',
        {**INSIGHT_ARGS, 'filter': "serviceName='ts-auth-service'"},
        PPL_NO_INSIGHTS,
    ),
    # characterization: current behaviour. Insight mode runs whenever the baseline is not
    # complete: a baseline start without an end and a trace field are ignored, no warning.
    Case(
        'insight_partial_baseline_and_trace_field',
        {
            **INSIGHT_ARGS,
            'baseTimeRangeStart': '2020-01-01 09:00:00',
            'traceFieldName': 'traceId',
        },
        PPL_INSIGHTS,
    ),
    Case('insight_opensearch_error', INSIGHT_ARGS, PPL_ERROR, 400),
)
CASE_IDS = [case.name for case in CASES]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def opensearch_version(fake_opensearch):
    fake_opensearch.version = OPENSEARCH_VERSION


def install(fake_opensearch, case: Case) -> None:
    # Data, not a callable: the same route crosses into a launched server, and it never
    # reads the query, so the reply does not depend on the keyword order.
    fake_opensearch.reply('POST', PPL_PATH, case.reply, status=case.status)


def recorded(case: Case, result: dict, requests: list[dict]) -> dict:
    assert [r['path'] for r in requests] == [PPL_PATH], 'insight mode sends one PPL query'
    return {
        'tool': TOOL,
        'arguments': case.arguments,
        'result': result,
        'requests': sort_request_keywords(requests),
    }


# ---------------------------------------------------------------------------
# the normalizer rule itself
# ---------------------------------------------------------------------------


def test_sort_error_keywords_sorts_the_match_argument():
    shuffled = sorted(ERROR_KEYWORDS, key=lambda word: word[::-1])
    query = f"source=x | where match(message, '{' '.join(shuffled)}') | head 5"
    assert sort_error_keywords(query) == (
        f"source=x | where match(message, '{' '.join(sorted(ERROR_KEYWORDS))}') | head 5"
    )


@pytest.mark.parametrize(
    'words',
    [
        sorted(ERROR_KEYWORDS)[1:],
        [*sorted(ERROR_KEYWORDS), 'warning'],
        [*sorted(ERROR_KEYWORDS), 'error'],
    ],
    ids=['missing', 'extra', 'repeated'],
)
def test_sort_error_keywords_rejects_another_word_list(words):
    with pytest.raises(AssertionError):
        sort_error_keywords(f"source=x | where match(message, '{' '.join(words)}')")


@pytest.mark.parametrize(
    'query',
    ['source=x | head 5', "where match(a, 'error') and match(b, 'error')"],
    ids=['none', 'two'],
)
def test_sort_error_keywords_needs_exactly_one_match(query):
    with pytest.raises(AssertionError):
        sort_error_keywords(query)


# ---------------------------------------------------------------------------
# insight mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
async def test_log_insight_in_process(mcp_session, fake_opensearch, golden, normalize, case):
    install(fake_opensearch, case)
    async with mcp_session(env=SKILLS_ENV) as session:
        start = len(fake_opensearch.requests)
        result = await session.call_tool(TOOL, case.arguments)
        requests = fake_opensearch.requests[start:]
    golden(f'{AREA}/{TOOL}/{case.name}', normalize(recorded(case, result, requests)))


# characterization: current behaviour. The keyword order in insight mode's PPL query follows
# set iteration order, so the request text differs from one PYTHONHASHSEED to another.
@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
async def test_log_insight_under_hash_seeds(
    stdio_launcher, fake_opensearch, golden, normalize, case
):
    install(fake_opensearch, case)
    orders = {}
    for seed in HASH_SEEDS:
        env = {**SKILLS_ENV, 'PYTHONHASHSEED': seed}
        async with stdio_launcher('--mode', 'single', env=env) as server:
            await server.initialize()
            start = len(server.opensearch_requests())
            result = await server.call_tool(TOOL, case.arguments)
            requests = server.opensearch_requests()[start:]
        orders[seed] = tuple(keyword_words(requests[0]['body']['query']))
        golden(f'{AREA}/{TOOL}/{case.name}', normalize(recorded(case, result, requests)))
    assert len(set(orders.values())) == len(HASH_SEEDS), (
        f'the keyword order did not change between PYTHONHASHSEED values {HASH_SEEDS}: '
        'the seeds did not take effect, or the order no longer follows the hash seed'
    )
