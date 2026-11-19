# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: header auth edge cases, over streamable HTTP.

Header auth here is ``OPENSEARCH_HEADER_AUTH`` (single mode) or a cluster's
``opensearch_header_auth`` (multi mode). Each case makes one ``tools/call`` of ``ListIndexTool`` (``include_detail: false``) through
``http_app`` (``ctx.request`` exists only over HTTP) with an ``Authorization`` header the
header parser (``opensearch.client._get_auth_from_headers``) does not turn into credentials,
and goldens, under ``golden/gaps_header_auth/``:

- ``setup``: mode, environment, request headers and cluster config;
- ``result``: the whole call result;
- ``requests``: the ``fake_opensearch`` request log (URL, the auth it saw, or ``none``) with,
  per request, the header names sent (never a value);
- ``events``: the structured log events (``opensearch_request``, ``datasource_connection``,
  ``tool_error``, ``tool_execution``), normalized;
- ``client_log``: every record the ``opensearch.client`` logger emits (level and message),
  debug records included: the header parser reports a header it cannot read only there.

In single mode the environment also holds basic credentials (``env-user``), so a header the
parser drops shows up as the server's own credentials reaching the fake. The server checks
the version at startup with no request headers: the first ``GET /`` of a single-mode case.
"""

import base64
import logging
import pytest
import yaml
from dataclasses import dataclass, field
from opensearchpy.connection.http_async import AsyncHttpConnection


AREA = 'gaps_header_auth'
TOOL = 'ListIndexTool'
ARGS = {'include_detail': False}

FAKE_URL = 'http://fake-opensearch.invalid:9200'
HEADER_URL = 'http://header-cluster.invalid:9200'

CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
]

HEADER_AUTH = {'OPENSEARCH_HEADER_AUTH': 'true'}
BASIC_ENV = {'OPENSEARCH_USERNAME': 'env-user', 'OPENSEARCH_PASSWORD': 'env-password'}
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}


def basic_header(raw: bytes) -> str:
    """``Basic`` and ``raw`` base64-encoded: any bytes, not only ``user:password``."""
    return 'Basic ' + base64.b64encode(raw).decode('ascii')


@dataclass(frozen=True)
class Case:
    """One ``tools/call`` over HTTP: golden name (below ``gaps_header_auth/``) and setup.

    ``clusters`` set means multi mode with that config file.
    """

    name: str
    headers: dict
    env: dict = field(default_factory=lambda: {**HEADER_AUTH, **BASIC_ENV})
    clusters: dict | None = None
    arguments: dict = field(default_factory=lambda: dict(ARGS))
    is_error: bool = False


# ---------------------------------------------------------------------------
# recorders
# ---------------------------------------------------------------------------


@pytest.fixture
def sent_header_names(monkeypatch, fake_opensearch) -> list[list[str]]:
    """Per request the fake receives (same order as its log): the header names sent.

    Wraps the aiohttp session the harness's fake stands in for, at opensearch-py's boundary.
    """
    sent: list[list[str]] = []
    # The harness's fake session factory, installed by ``fake_opensearch``.
    create_fake_session = AsyncHttpConnection._create_aiohttp_session

    async def recording_create_session(connection) -> None:
        await create_fake_session(connection)
        request = connection.session.request

        def recording_request(method, url, data=None, auth=None, headers=None, **options):
            sent.append(sorted(str(name).lower() for name in (headers or {})))
            return request(method, url, data=data, auth=auth, headers=headers, **options)

        connection.session.request = recording_request

    monkeypatch.setattr(AsyncHttpConnection, '_create_aiohttp_session', recording_create_session)
    return sent


CLIENT_LOGGER = 'opensearch.client'
# Loggers whose records a case reads, and the level each is set to for the test.
LOG_LEVELS = {
    'opensearch': logging.INFO,
    CLIENT_LOGGER: logging.DEBUG,
    'mcp_server_opensearch.tool_executor': logging.INFO,
    'tools.tool_logging': logging.INFO,
}
EVENT_TYPES = ('opensearch_request', 'datasource_connection', 'tool_error', 'tool_execution')
EVENT_FIELDS = (
    'event_type',
    'http_method',
    'endpoint',
    'status',
    'status_code',
    'response_size',
    'duration_ms',
    'error',
    'auth_method',
    'datasource_type',
    'opensearch_url',
    'tool_name',
    'tool_key',
    'client_name',
    'error_type',
    'exception_type',
    'root_cause',
)


def log_events(caplog) -> list[dict]:
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
        }
        for record in caplog.records
        if getattr(record, 'event_type', None) in EVENT_TYPES
    ]


def client_log(caplog) -> list[dict]:
    """Every ``opensearch.client`` record, debug included, as level and message."""
    return [
        {'level': record.levelname, 'message': record.getMessage()}
        for record in caplog.records
        if record.name == CLIENT_LOGGER
    ]


@pytest.fixture
def run_case(http_app, fake_opensearch, sent_header_names, caplog, golden, normalize, tmp_path):
    """Build the server for the case, make the one call over HTTP, golden what happened."""

    async def run_case(case: Case) -> dict:
        assert fake_opensearch.url == FAKE_URL
        for logger_name, level in LOG_LEVELS.items():
            caplog.set_level(level, logger=logger_name)
        # The handler takes the lowest level; each logger's own level does the filtering.
        caplog.handler.setLevel(logging.DEBUG)
        fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
        config_file = ''
        mode = 'single'
        if case.clusters is not None:
            mode = 'multi'
            config_file = tmp_path / 'clusters.yml'
            with open(config_file, 'w', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump({'clusters': case.clusters}, f, sort_keys=False)

        async with http_app(mode=mode, config_file=config_file, env=case.env) as http:
            message = await http.mcp(
                'tools/call', {'name': TOOL, 'arguments': case.arguments}, headers=case.headers
            )
        assert 'result' in message, message
        result = message['result']

        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError') is case.is_error, result
        requests = fake_opensearch.requests
        assert len(sent_header_names) == len(requests), (sent_header_names, requests)
        data = {
            'setup': {
                'mode': mode,
                'transport': 'http',
                'env': case.env,
                'headers': case.headers,
                'clusters': case.clusters,
            },
            'call': {'name': TOOL, 'arguments': case.arguments},
            'result': result,
            'requests': [
                {**logged, 'header_names': names}
                for logged, names in zip(requests, sent_header_names)
            ],
            'events': log_events(caplog),
            'client_log': client_log(caplog),
        }
        golden(f'{AREA}/{case.name}', normalize(data))
        return data

    return run_case


# ---------------------------------------------------------------------------
# single mode (OPENSEARCH_HEADER_AUTH=true): Authorization headers the parser drops
# ---------------------------------------------------------------------------

SINGLE_CASES = [
    # Gap client.py 1196->1214 (``if token:`` false) is not reachable through a header: the
    # parser strips the whole header value first, so 'Bearer ' is read as 'Bearer', which is
    # neither 'bearer ' nor 'basic ' (1198->1214). characterization: current behaviour. An
    # empty Bearer token is dropped without a log record; the server's own credentials are used.
    Case('single/bearer_empty_token', headers={'authorization': 'Bearer '}),
    # The same for 'Basic ' with nothing after it.
    Case('single/basic_empty_credentials', headers={'authorization': 'Basic '}),
    # Gap 1198->1214: a scheme that is neither Bearer nor Basic is dropped without a log
    # record; the server's own credentials are used.
    Case('single/unknown_scheme', headers={'authorization': 'Token char-opaque-token'}),
    # Gap 1207->1214: Basic credentials that decode to text without ':' are dropped without a
    # log record; the server's own credentials are used.
    Case('single/basic_without_colon', headers={'authorization': basic_header(b'header-user')}),
    # Gap 1211-1212: invalid base64 raises inside the parser; it is swallowed with a debug log
    # record and the server's own credentials are used.
    Case('single/basic_invalid_base64', headers={'authorization': 'Basic abc'}),
    # Gap 1211-1212: base64 of bytes that are not UTF-8, the same.
    Case(
        'single/basic_not_utf8',
        headers={'authorization': basic_header(b'\xff\xfe:header-password')},
    ),
    # characterization: current behaviour. The exception leaves what the parser read before
    # the Authorization header in place: the opensearch-url header is still used.
    Case(
        'single/basic_invalid_base64_keeps_header_url',
        env={**HEADER_AUTH, **NO_AUTH},
        headers={'opensearch-url': HEADER_URL, 'authorization': 'Basic abc'},
    ),
]


@pytest.mark.parametrize('case', SINGLE_CASES, ids=lambda case: case.name)
async def test_single_mode_header_auth_dropped_authorization(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# multi mode (per-cluster opensearch_header_auth): empty Basic password
# ---------------------------------------------------------------------------

CLUSTERS = {
    'header-cluster': {
        'opensearch_url': FAKE_URL,
        'opensearch_username': 'cluster-user',
        'opensearch_password': 'cluster-password',
        'opensearch_header_auth': True,
    },
    'no-auth-header-cluster': {
        'opensearch_url': FAKE_URL,
        'opensearch_no_auth': True,
        'opensearch_header_auth': True,
    },
}


def on(cluster: str) -> dict:
    return {**ARGS, 'opensearch_cluster_name': cluster}


EMPTY_PASSWORD = {'authorization': basic_header(b'header-user:')}

MULTI_CASES = [
    # Gap client.py 671-672: header Basic credentials with an empty password raise
    # AuthenticationError (the cluster's own credentials are not used instead).
    Case(
        'multi/basic_empty_password',
        env={},
        clusters=CLUSTERS,
        arguments=on('header-cluster'),
        headers=EMPTY_PASSWORD,
        is_error=True,
    ),
    # The check is skipped for a cluster with opensearch_no_auth: the call goes out unauthenticated.
    Case(
        'multi/basic_empty_password_no_auth_cluster',
        env={},
        clusters=CLUSTERS,
        arguments=on('no-auth-header-cluster'),
        headers=EMPTY_PASSWORD,
    ),
]


@pytest.mark.parametrize('case', MULTI_CASES, ids=lambda case: case.name)
async def test_multi_mode_header_auth_empty_password(run_case, case):
    await run_case(case)
