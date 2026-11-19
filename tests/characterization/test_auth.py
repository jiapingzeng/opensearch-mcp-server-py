# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: auth modes and connection resolution.

Each case makes one ``tools/call`` of ``ListIndexTool`` (``include_detail: false``) and
goldens, under ``golden/auth/``, how the call reached OpenSearch:

- ``setup``: mode, transport, environment, HTTP headers, cluster config and ``--profile``;
- ``result``: the whole call result;
- ``requests``: the ``fake_opensearch`` request log (URL, auth kind) with, per request, the
  header names sent, the ``user-agent`` value (versions as ``<VERSION>``), for SigV4
  the signed header names and which fake identity signed (never a header value), the TLS
  options the opensearch-py connection was built with, the timeout and whether redirects
  are followed;
- ``events``: the structured ``opensearch_request`` and ``datasource_connection`` log
  events (normalized); ``client_warnings``: the warnings the client setup logs;
- ``aws_api_calls``: every AWS API call (only STS ``AssumeRole`` is answered, at the boto
  boundary; any other call fails the test); ``dns_lookups`` where the SSRF guard resolves.

The calls go through ``mcp_session``, or through ``http_app`` where request headers matter
(header auth): ``ctx.request`` exists only over HTTP. Each call opens a client for the
version check (``GET /``) and another for ``_cat/indices``; in single mode the server also
checks the version at startup, with no request headers.

No real AWS or network call is made: the harness blocks the network, botocore's API calls
are intercepted here, AWS keys are fixed fakes and botocore's signing clock is frozen.
"""

import base64
import botocore.auth
import botocore.client
import logging
import pytest
import re
import socket
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from opensearchpy.connection.http_async import AsyncHttpConnection
from pathlib import Path
from typing import Any


AREA = 'auth'
TOOL = 'ListIndexTool'
ARGS = {'include_detail': False}

FAKE_URL = 'http://fake-opensearch.invalid:9200'
FAKE_HTTPS_URL = 'https://fake-opensearch.invalid:9200'

CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
    {'health': 'yellow', 'status': 'open', 'index': 'metrics', 'docs.count': '42'},
]

# botocore signs with this moment, so the SigV4 header set is the same in every run.
FROZEN_SIGNING_TIME = datetime(2020, 1, 1, 0, 0, 0)

# Fake AWS identities: each has its own access key id, so a golden names which one signed.
ENV_KEY = 'AKIACHARENV000000001'
PROFILE_A_KEY = 'AKIACHARPROFILEA0001'
PROFILE_B_KEY = 'AKIACHARPROFILEB0001'
HEADER_KEY = 'AKIACHARHEADER000001'
ASSUMED_KEY = 'ASIACHARASSUMED00001'
IDENTITIES = {
    ENV_KEY: 'env',
    PROFILE_A_KEY: 'profile:char-a',
    PROFILE_B_KEY: 'profile:char-b',
    HEADER_KEY: 'headers',
    ASSUMED_KEY: 'sts:assumed-role',
}
ROLE_ARN = 'arn:aws:iam::123456789012:role/char-role'

AWS_CONFIG = f"""\
[profile char-a]
region = eu-west-1
aws_access_key_id = {PROFILE_A_KEY}
aws_secret_access_key = char-profile-a-secret

[profile char-b]
region = eu-west-2
aws_access_key_id = {PROFILE_B_KEY}
aws_secret_access_key = char-profile-b-secret
"""

NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
BASIC_ENV = {'OPENSEARCH_USERNAME': 'env-user', 'OPENSEARCH_PASSWORD': 'env-password'}
AWS_ENV_KEYS = {'AWS_ACCESS_KEY_ID': ENV_KEY, 'AWS_SECRET_ACCESS_KEY': 'char-env-secret'}
AWS_ENV = {**AWS_ENV_KEYS, 'AWS_REGION': 'us-west-2'}
HEADER_AUTH = {'OPENSEARCH_HEADER_AUTH': 'true'}
SSRF_GUARD = {'OPENSEARCH_SSRF_GUARD': 'true'}
# No OPENSEARCH_URL: dynamic (per-call) connection mode turns on by itself.
NO_URL = {'OPENSEARCH_URL': None}

# Hosts the fake resolver answers for the SSRF guard. Nothing connects to these addresses:
# every request goes to the fake OpenSearch. 1.1.1.1 stands for any public address.
DNS_ANSWERS = {
    'private-cluster.invalid': '10.0.0.8',
    'public-cluster.invalid': '1.1.1.1',
}


def basic_header(username: str, password: str) -> str:
    return 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode('ascii')


@dataclass(frozen=True)
class TmpPath:
    """A path below the test's ``tmp_path``, resolved when the case runs."""

    name: str


AWS_CONFIG_FILE = TmpPath('aws_config')
CERT_FILE = TmpPath('client-cert.pem')
KEY_FILE = TmpPath('client-key.pem')
MISSING_FILE = TmpPath('missing.pem')
AWS_PROFILES = {'AWS_CONFIG_FILE': AWS_CONFIG_FILE}


@dataclass(frozen=True)
class Case:
    """One ``tools/call`` of ``ListIndexTool``: golden name (below ``auth/``) and setup.

    ``headers`` set means the call goes over HTTP (``http_app``); else through
    ``mcp_session``. ``clusters`` set means multi mode with that config file.
    """

    name: str
    env: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=lambda: dict(ARGS))
    headers: dict | None = None
    clusters: dict | None = None
    profile: str = ''
    is_error: bool = False


# ---------------------------------------------------------------------------
# recorders
# ---------------------------------------------------------------------------


def _option(value: Any) -> Any:
    """A connection keyword as JSON: opensearch-py's default sentinels become '<default>'."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return '<default>' if type(value) is object else repr(value)


def _tls_options(connection: AsyncHttpConnection, kwargs: dict) -> dict:
    """The TLS options a connection was built with, and the SSL context they produced."""
    context = connection._ssl_context
    return {
        'use_ssl': connection.use_ssl,
        'verify_certs': _option(kwargs.get('verify_certs', object())),
        'ssl_show_warn': _option(kwargs.get('ssl_show_warn', object())),
        'ca_certs': _option(kwargs.get('ca_certs')),
        'client_cert': _option(kwargs.get('client_cert')),
        'client_key': _option(kwargs.get('client_key')),
        'ssl_assert_fingerprint': _option(kwargs.get('ssl_assert_fingerprint')),
        'ssl_context': None
        if context is None
        else {'verify_mode': context.verify_mode.name, 'check_hostname': context.check_hostname},
    }


SIGNED_HEADERS = re.compile(r'SignedHeaders=([^,\s]+)')
CREDENTIAL = re.compile(r'Credential=([^/,\s]+)/')
# The server's user agent, and opensearch-py's default one (sent when the server's is lost):
# versions become <VERSION>, so a release or another Python does not read as a change.
USER_AGENTS = (
    (re.compile(r'^opensearch-mcp-server-py/\S+$'), 'opensearch-mcp-server-py/<VERSION>'),
    (
        re.compile(r'^opensearch-py/\S+ \(Python \S+\)$'),
        'opensearch-py/<VERSION> (Python <VERSION>)',
    ),
)


def _user_agent(value: str | None) -> str | None:
    for pattern, replacement in USER_AGENTS:
        if value is not None and pattern.match(value):
            return replacement
    return value


def _describe(connection: AsyncHttpConnection, headers: dict | None, options: dict) -> dict:
    """What a request carried besides what the fake logs: header names, never secret values."""
    headers = {str(name).lower(): str(value) for name, value in (headers or {}).items()}
    timeout = options.get('timeout')
    record = {
        'header_names': sorted(headers),
        'user_agent': _user_agent(headers.get('user-agent')),
        'tls': getattr(connection, '_char_tls', None),
        'timeout': getattr(timeout, 'total', timeout),
        'allow_redirects': options.get('allow_redirects'),
    }
    authorization = headers.get('authorization', '')
    if authorization.lower().startswith('aws4-hmac-sha256'):
        signed = SIGNED_HEADERS.search(authorization)
        credential = CREDENTIAL.search(authorization)
        key = credential.group(1) if credential else None
        record['sigv4'] = {
            'signed_headers': signed.group(1).split(';') if signed else None,
            'identity': IDENTITIES.get(key, '<unknown access key>'),
        }
    return record


@pytest.fixture
def wire(monkeypatch, fake_opensearch) -> list[dict]:
    """Per request the fake receives (same order as its log): headers, TLS, transport options.

    Wraps opensearch-py's connection at its boundary: ``__init__`` (TLS keywords and the SSL
    context) and the aiohttp session the harness's fake stands in for (headers and options).
    """
    sent: list[dict] = []
    original_init = AsyncHttpConnection.__init__
    # The harness's fake session factory, installed by ``fake_opensearch``.
    create_fake_session = AsyncHttpConnection._create_aiohttp_session

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._char_tls = _tls_options(self, kwargs)

    async def recording_create_session(connection) -> None:
        await create_fake_session(connection)
        request = connection.session.request

        def recording_request(method, url, data=None, auth=None, headers=None, **options):
            sent.append(_describe(connection, headers, options))
            return request(method, url, data=data, auth=auth, headers=headers, **options)

        connection.session.request = recording_request

    monkeypatch.setattr(AsyncHttpConnection, '__init__', recording_init)
    monkeypatch.setattr(AsyncHttpConnection, '_create_aiohttp_session', recording_create_session)
    return sent


@pytest.fixture(autouse=True)
def aws_api(monkeypatch) -> list[dict]:
    """Every botocore API call, intercepted: STS AssumeRole is answered, anything else fails."""
    calls: list[dict] = []
    unexpected: list[str] = []

    def fake_api_call(client, operation_name, api_params):
        service = client.meta.service_model.service_name
        calls.append(
            {
                'service': service,
                'region': client.meta.region_name,
                'operation': operation_name,
                'params': dict(api_params),
            }
        )
        if (service, operation_name) == ('sts', 'AssumeRole'):
            return {
                'Credentials': {
                    'AccessKeyId': ASSUMED_KEY,
                    'SecretAccessKey': 'char-assumed-secret',
                    'SessionToken': 'char-assumed-session-token',
                    'Expiration': datetime(2020, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                'AssumedRoleUser': {
                    'AssumedRoleId': 'AROACHARASSUMED00001:OpenSearchClientSession',
                    'Arn': 'arn:aws:sts::123456789012:assumed-role/char-role/OpenSearchClientSession',
                },
            }
        unexpected.append(f'{service}.{operation_name}')
        raise AssertionError(f'unexpected AWS API call in a characterization test: {service}')

    monkeypatch.setattr(botocore.client.BaseClient, '_make_api_call', fake_api_call)
    yield calls
    if unexpected:
        pytest.fail(f'AWS API calls attempted: {unexpected}', pytrace=False)


@pytest.fixture(autouse=True)
def frozen_signing_clock(monkeypatch) -> None:
    """Freeze the time botocore's SigV4 signer reads."""
    monkeypatch.setattr(
        botocore.auth, 'get_current_datetime', lambda remove_tzinfo=True: FROZEN_SIGNING_TIME
    )


@pytest.fixture
def fake_dns(monkeypatch, no_network) -> list[str]:
    """Resolve the ``DNS_ANSWERS`` hosts (for the SSRF guard); any other lookup stays blocked."""
    lookups: list[str] = []
    guarded_getaddrinfo = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        lookups.append(f'{host}:{port}')
        if host in DNS_ANSWERS:
            address = (DNS_ANSWERS[host], port)
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', address)]
        return guarded_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    return lookups


EVENT_TYPES = ('opensearch_request', 'datasource_connection')
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


CLIENT_LOGGER = 'opensearch.client'


def client_warnings(caplog) -> list[str]:
    """Warnings the client setup logs (fallbacks it takes, inputs it ignores)."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == CLIENT_LOGGER
        and record.levelno == logging.WARNING
        and not hasattr(record, 'event_type')
    ]


def write_client_cert(cert_path: Path, key_path: Path) -> None:
    """A self-signed EC certificate and its key, as PEM (usable as a CA bundle too)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'characterization')])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime(2020, 1, 1))
        .not_valid_after(datetime(2040, 1, 1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _resolve(value: Any, tmp_path: Path) -> Any:
    if isinstance(value, TmpPath):
        return str(tmp_path / value.name)
    if isinstance(value, dict):
        return {k: _resolve(v, tmp_path) for k, v in value.items()}
    return value


@pytest.fixture
def run_case(
    mcp_session, http_app, fake_opensearch, wire, aws_api, caplog, golden, normalize, tmp_path
):
    """Build the server for the case, make the one call, golden what reached OpenSearch."""

    async def run_case(case: Case, **extra: Any) -> dict:
        assert fake_opensearch.url == FAKE_URL
        caplog.set_level(logging.INFO, logger='opensearch')
        fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
        with open(tmp_path / AWS_CONFIG_FILE.name, 'w', encoding='utf-8', newline='\n') as f:
            f.write(AWS_CONFIG)
        write_client_cert(tmp_path / CERT_FILE.name, tmp_path / KEY_FILE.name)
        env = _resolve(case.env, tmp_path)
        config_file = ''
        mode = 'single'
        if case.clusters is not None:
            mode = 'multi'
            clusters = _resolve(case.clusters, tmp_path)
            config_file = tmp_path / 'clusters.yml'
            with open(config_file, 'w', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump({'clusters': clusters}, f, sort_keys=False)

        if case.headers is None:
            transport = 'in-memory'
            async with mcp_session(
                mode=mode, config_file=config_file, env=env, profile=case.profile
            ) as session:
                result = await session.call_tool(TOOL, case.arguments)
        else:
            transport = 'http'
            async with http_app(
                mode=mode, config_file=config_file, env=env, profile=case.profile
            ) as http:
                message = await http.mcp(
                    'tools/call', {'name': TOOL, 'arguments': case.arguments}, headers=case.headers
                )
            assert 'result' in message, message
            result = message['result']

        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError') is case.is_error, result
        assert len(wire) == len(fake_opensearch.requests), (wire, fake_opensearch.requests)
        data = {
            'setup': {
                'mode': mode,
                'transport': transport,
                'env': env,
                'headers': case.headers,
                'clusters': case.clusters and _resolve(case.clusters, tmp_path),
                'profile': case.profile or None,
            },
            'call': {'name': TOOL, 'arguments': case.arguments},
            'result': result,
            'requests': [
                {**logged, **sent} for logged, sent in zip(fake_opensearch.requests, wire)
            ],
            'events': log_events(caplog),
            'client_warnings': client_warnings(caplog),
            'aws_api_calls': aws_api,
            **extra,
        }
        golden(f'{AREA}/{case.name}', normalize(data))
        return data

    return run_case


# ---------------------------------------------------------------------------
# single mode: auth from the environment
# ---------------------------------------------------------------------------

SINGLE_CASES = [
    Case('single/no_auth', env=NO_AUTH),
    Case('single/basic_env', env=BASIC_ENV),
    # Nothing configured but OPENSEARCH_URL: the AWS credentials path, which has no region.
    Case('single/no_credentials', is_error=True),
    Case('single/sigv4_env_static_keys', env=AWS_ENV),
    Case(
        'single/sigv4_env_session_token',
        env={**AWS_ENV, 'AWS_SESSION_TOKEN': 'char-env-session-token'},
    ),
    # Basic auth wins over AWS credentials in the environment.
    Case('single/basic_env_over_aws_env', env={**AWS_ENV, **BASIC_ENV}),
    # No auth wins over everything.
    Case('single/no_auth_over_basic_env', env={**NO_AUTH, **BASIC_ENV, **AWS_ENV}),
    # AWS_PROFILE: credentials and region from the profile in the AWS config file.
    Case('single/sigv4_aws_profile_env', env={**AWS_PROFILES, 'AWS_PROFILE': 'char-a'}),
    # --profile: the same, from the command line.
    Case('single/sigv4_profile_cli', env=AWS_PROFILES, profile='char-b'),
    Case(
        'single/sigv4_profile_cli_over_aws_profile_env',
        env={**AWS_PROFILES, 'AWS_PROFILE': 'char-a'},
        profile='char-b',
    ),
    # characterization: current behaviour. --profile wins over AWS_PROFILE for the credentials
    # and the region, but a profile that does not exist only logs a warning and the client
    # falls back to the default session.
    Case(
        'single/sigv4_profile_cli_unknown_falls_back',
        env={**AWS_PROFILES, **AWS_ENV},
        profile='no-such-profile',
    ),
    Case(
        'single/iam_role_assumption',
        env={**AWS_ENV, 'AWS_IAM_ARN': ROLE_ARN},
    ),
    # IAM role assumption wins over basic auth.
    Case(
        'single/iam_role_over_basic_env',
        env={**AWS_ENV, **BASIC_ENV, 'AWS_IAM_ARN': ROLE_ARN},
    ),
    Case('single/serverless_aoss', env={**AWS_ENV, 'AWS_OPENSEARCH_SERVERLESS': 'true'}),
    Case(
        'single/serverless_iam_role_assumption',
        env={**AWS_ENV, 'AWS_IAM_ARN': ROLE_ARN, 'AWS_OPENSEARCH_SERVERLESS': 'true'},
    ),
]

REGION_CASES = [
    Case('region/aws_region_env', env={**AWS_ENV_KEYS, 'AWS_REGION': 'ap-south-1'}),
    # boto3's default session reads AWS_DEFAULT_REGION.
    Case('region/aws_default_region_env', env={**AWS_ENV_KEYS, 'AWS_DEFAULT_REGION': 'sa-east-1'}),
    Case(
        'region/aws_region_over_aws_default_region',
        env={**AWS_ENV_KEYS, 'AWS_REGION': 'ap-south-1', 'AWS_DEFAULT_REGION': 'sa-east-1'},
    ),
    Case(
        'region/aws_region_over_profile_region',
        env={**AWS_PROFILES, 'AWS_PROFILE': 'char-a', 'AWS_REGION': 'us-east-2'},
    ),
    # characterization: current behaviour. The region is not read from an Amazon OpenSearch
    # Service (or Serverless) URL: with no region configured, AWS auth fails.
    Case(
        'region/not_read_from_service_url',
        env={
            **AWS_ENV_KEYS,
            'OPENSEARCH_URL': 'https://search-char-abc123.eu-west-3.es.amazonaws.com',
        },
        is_error=True,
    ),
    Case(
        'region/not_read_from_serverless_url',
        env={
            **AWS_ENV_KEYS,
            'OPENSEARCH_URL': 'https://charcollection123.eu-west-3.aoss.amazonaws.com',
            'AWS_OPENSEARCH_SERVERLESS': 'true',
        },
        is_error=True,
    ),
    Case(
        'region/iam_role_without_region',
        env={**AWS_ENV_KEYS, 'AWS_IAM_ARN': ROLE_ARN},
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', SINGLE_CASES + REGION_CASES, ids=lambda case: case.name)
async def test_single_mode_auth(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# header-based auth (OPENSEARCH_HEADER_AUTH=true), over HTTP
# ---------------------------------------------------------------------------

HEADER_CREDENTIALS = {
    'aws-access-key-id': HEADER_KEY,
    'aws-secret-access-key': 'char-header-secret',
    'aws-session-token': 'char-header-session-token',
}

HEADER_CASES = [
    Case(
        'header/opensearch_url',
        env={**HEADER_AUTH, **NO_AUTH},
        headers={'opensearch-url': 'http://header-cluster.invalid:9200'},
    ),
    # The aws-region header overrides AWS_REGION for the server's own (env) credentials.
    Case(
        'header/aws_region',
        env={**HEADER_AUTH, **AWS_ENV},
        headers={'aws-region': 'eu-central-1'},
    ),
    Case(
        'header/aws_credentials',
        env=HEADER_AUTH,
        headers={**HEADER_CREDENTIALS, 'aws-region': 'eu-central-1'},
    ),
    # Header credentials without a region in headers or environment: not used.
    Case(
        'header/aws_credentials_without_region',
        env=HEADER_AUTH,
        headers=HEADER_CREDENTIALS,
        is_error=True,
    ),
    Case(
        'header/aws_service_name_aoss',
        env=HEADER_AUTH,
        headers={**HEADER_CREDENTIALS, 'aws-region': 'eu-central-1', 'aws-service-name': 'aoss'},
    ),
    # characterization: current behaviour. A Bearer token replaces the client's headers, so
    # the server's user-agent is lost and opensearch-py's default one is sent instead.
    Case(
        'header/bearer',
        env=HEADER_AUTH,
        headers={'authorization': 'Bearer char-bearer-token'},
    ),
    # Bearer wins over the server's basic credentials and over AWS credentials in headers.
    Case(
        'header/bearer_over_basic_env_and_aws_headers',
        env={**HEADER_AUTH, **BASIC_ENV},
        headers={
            **HEADER_CREDENTIALS,
            'aws-region': 'eu-central-1',
            'authorization': 'Bearer char-bearer-token',
        },
    ),
    # Basic credentials from the header replace the server's.
    Case(
        'header/basic',
        env={**HEADER_AUTH, **BASIC_ENV},
        headers={'authorization': basic_header('header-user', 'header-password')},
    ),
    Case(
        'header/basic_empty_password',
        env={**HEADER_AUTH, **BASIC_ENV},
        headers={'authorization': basic_header('header-user', '')},
        is_error=True,
    ),
    # A header URL with the server's credentials in the environment: they are not used.
    Case(
        'header/url_with_env_credentials_not_used',
        env={**HEADER_AUTH, **BASIC_ENV, **AWS_ENV},
        headers={'opensearch-url': 'http://header-cluster.invalid:9200'},
        is_error=True,
    ),
    Case(
        'header/url_with_header_basic',
        env={**HEADER_AUTH, **BASIC_ENV},
        headers={
            'opensearch-url': 'http://header-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
        },
    ),
    # Header auth off: the same headers are ignored and the environment's basic auth is used.
    Case(
        'header/headers_ignored_when_header_auth_off',
        env=BASIC_ENV,
        headers={
            'opensearch-url': 'http://header-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
            **HEADER_CREDENTIALS,
            'aws-region': 'eu-central-1',
        },
    ),
]

HEADER_SSRF_CASES = [
    Case(
        'header/ssrf_guard_private_url',
        env={**HEADER_AUTH, **SSRF_GUARD},
        headers={
            'opensearch-url': 'https://private-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
        },
        is_error=True,
    ),
    Case(
        'header/ssrf_guard_plain_http_url',
        env={**HEADER_AUTH, **SSRF_GUARD},
        headers={
            'opensearch-url': 'http://public-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
        },
        is_error=True,
    ),
    # Accepted: the request goes out with redirects turned off.
    Case(
        'header/ssrf_guard_public_url',
        env={**HEADER_AUTH, **SSRF_GUARD},
        headers={
            'opensearch-url': 'https://public-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
        },
    ),
]


@pytest.mark.parametrize('case', HEADER_CASES, ids=lambda case: case.name)
async def test_header_auth(run_case, case):
    await run_case(case)


@pytest.mark.parametrize('case', HEADER_SSRF_CASES, ids=lambda case: case.name)
async def test_header_auth_ssrf_guard(run_case, fake_dns, case):
    await run_case(case, dns_lookups=fake_dns)


# ---------------------------------------------------------------------------
# per-call dynamic connection overrides (tool arguments)
# ---------------------------------------------------------------------------

DYNAMIC_URL = 'http://dynamic-cluster.invalid:9200'

DYNAMIC_CASES = [
    Case(
        'dynamic/accepted_no_preconfigured_url',
        env=NO_URL,
        arguments={
            **ARGS,
            'opensearch_url': DYNAMIC_URL,
            'opensearch_username': 'call-user',
            'opensearch_password': 'call-password',
        },
    ),
    Case(
        'dynamic/accepted_dynamic_connection_true',
        env={**BASIC_ENV, 'OPENSEARCH_DYNAMIC_CONNECTION': 'true'},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL, 'opensearch_no_auth': True},
    ),
    # Arguments without a URL override the environment's settings for the server's own URL.
    Case(
        'dynamic/accepted_credentials_only',
        env={**BASIC_ENV, 'OPENSEARCH_DYNAMIC_CONNECTION': 'true'},
        arguments={
            **ARGS,
            'opensearch_username': 'call-user',
            'opensearch_password': 'call-password',
        },
    ),
    Case(
        'dynamic/rejected_dynamic_mode_off',
        env=BASIC_ENV,
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL, 'opensearch_no_auth': True},
        is_error=True,
    ),
    Case(
        'dynamic/rejected_dynamic_connection_false',
        env={**NO_URL, 'OPENSEARCH_DYNAMIC_CONNECTION': 'false'},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL, 'opensearch_no_auth': True},
        is_error=True,
    ),
    # A caller URL never borrows the server's basic credentials or (by default) AWS identity.
    Case(
        'dynamic/ambient_aws_fallback_off',
        env={**NO_URL, **AWS_ENV, **BASIC_ENV},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL},
        is_error=True,
    ),
    Case(
        'dynamic/ambient_aws_fallback_on',
        env={**NO_URL, **AWS_ENV, **BASIC_ENV, 'OPENSEARCH_ALLOW_AMBIENT_AWS_FALLBACK': 'true'},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL},
    ),
    # characterization: current behaviour. A per-call aws_profile supplies the credentials
    # but not its region: the region comes from AWS_REGION, --profile or AWS_PROFILE only,
    # so without one the call fails although the profile names a region.
    Case(
        'dynamic/aws_profile_argument_region_not_read',
        env={**NO_URL, **AWS_PROFILES},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL, 'aws_profile': 'char-a'},
        is_error=True,
    ),
    Case(
        'dynamic/aws_profile_argument',
        env={**NO_URL, **AWS_PROFILES},
        arguments={
            **ARGS,
            'opensearch_url': DYNAMIC_URL,
            'aws_profile': 'char-a',
            'aws_region': 'eu-north-1',
        },
    ),
    Case(
        'dynamic/iam_role_argument_without_base_credentials',
        env={**NO_URL, **AWS_ENV},
        arguments={**ARGS, 'opensearch_url': DYNAMIC_URL, 'aws_iam_arn': ROLE_ARN},
        is_error=True,
    ),
    Case(
        'dynamic/iam_role_argument_with_profile',
        env={**NO_URL, **AWS_PROFILES},
        arguments={
            **ARGS,
            'opensearch_url': DYNAMIC_URL,
            'aws_iam_arn': ROLE_ARN,
            'aws_profile': 'char-b',
            'aws_region': 'eu-north-1',
            'aws_opensearch_serverless': True,
        },
    ),
    # A caller may not turn certificate verification off.
    Case(
        'dynamic/ssl_verify_false_ignored',
        env=NO_URL,
        arguments={
            **ARGS,
            'opensearch_url': 'https://dynamic-cluster.invalid:9200',
            'opensearch_no_auth': True,
            'opensearch_ssl_verify': False,
            'opensearch_timeout': 7,
        },
    ),
]

DYNAMIC_SSRF_CASES = [
    Case(
        'dynamic/ssrf_guard_private_url',
        env={**NO_URL, **SSRF_GUARD},
        arguments={
            **ARGS,
            'opensearch_url': 'https://private-cluster.invalid:9200',
            'opensearch_username': 'call-user',
            'opensearch_password': 'call-password',
        },
        is_error=True,
    ),
    # The operator's own URL is trusted: the guard does not check it.
    Case(
        'dynamic/ssrf_guard_skips_operator_url',
        env={**NO_AUTH, **SSRF_GUARD},
    ),
]


@pytest.mark.parametrize('case', DYNAMIC_CASES, ids=lambda case: case.name)
async def test_dynamic_connection_overrides(run_case, case):
    await run_case(case)


@pytest.mark.parametrize('case', DYNAMIC_SSRF_CASES, ids=lambda case: case.name)
async def test_dynamic_connection_ssrf_guard(run_case, fake_dns, case):
    await run_case(case, dns_lookups=fake_dns)


# ---------------------------------------------------------------------------
# multi mode: clusters from a config file
# ---------------------------------------------------------------------------

CLUSTERS = {
    'basic-cluster': {
        'opensearch_url': FAKE_URL,
        'opensearch_username': 'cluster-user',
        'opensearch_password': 'cluster-password',
    },
    'iam-cluster': {
        'opensearch_url': 'http://iam-cluster.invalid:9200',
        'iam_arn': ROLE_ARN,
        'aws_region': 'us-east-1',
    },
    'profile-cluster': {
        'opensearch_url': 'http://profile-cluster.invalid:9200',
        'profile': 'char-a',
    },
    'aws-cluster': {'opensearch_url': 'http://aws-cluster.invalid:9200'},
    'header-cluster': {
        'opensearch_url': 'http://header-cluster.invalid:9200',
        'opensearch_username': 'cluster-user',
        'opensearch_password': 'cluster-password',
        'opensearch_header_auth': True,
    },
}


def on(cluster: str, **arguments: Any) -> dict:
    return {**ARGS, 'opensearch_cluster_name': cluster, **arguments}


MULTI_CASES = [
    Case('multi/basic_cluster', clusters=CLUSTERS, arguments=on('basic-cluster')),
    Case(
        'multi/iam_cluster',
        clusters=CLUSTERS,
        env=AWS_ENV_KEYS,
        arguments=on('iam-cluster'),
    ),
    Case('multi/missing_cluster_name', clusters=CLUSTERS, arguments=dict(ARGS), is_error=True),
    Case(
        'multi/unknown_cluster_name',
        clusters=CLUSTERS,
        arguments=on('no-such-cluster'),
        is_error=True,
    ),
    # characterization: current behaviour. Connection override arguments are accepted and
    # silently ignored in multi mode (no error, the cluster's own settings are used).
    Case(
        'multi/override_arguments_ignored',
        clusters=CLUSTERS,
        arguments=on(
            'basic-cluster',
            opensearch_url=DYNAMIC_URL,
            opensearch_username='call-user',
            opensearch_password='call-password',
            opensearch_no_auth=True,
            aws_region='ap-south-1',
        ),
    ),
    # Profile: the cluster's own, else --profile, else AWS_PROFILE.
    Case(
        'multi/profile_from_cluster',
        clusters=CLUSTERS,
        env=AWS_PROFILES,
        arguments=on('profile-cluster'),
    ),
    Case(
        'multi/profile_cluster_over_cli',
        clusters=CLUSTERS,
        env={**AWS_PROFILES, 'AWS_PROFILE': 'char-b'},
        profile='char-b',
        arguments=on('profile-cluster'),
    ),
    Case(
        'multi/profile_from_cli',
        clusters=CLUSTERS,
        env={**AWS_PROFILES, 'AWS_PROFILE': 'char-a'},
        profile='char-b',
        arguments=on('aws-cluster'),
    ),
    Case(
        'multi/profile_from_aws_profile_env',
        clusters=CLUSTERS,
        env={**AWS_PROFILES, 'AWS_PROFILE': 'char-a'},
        arguments=on('aws-cluster'),
    ),
    # Environment auth settings are not read in multi mode; AWS credentials still are.
    Case(
        'multi/env_auth_settings_not_used',
        clusters=CLUSTERS,
        env={**NO_AUTH, **BASIC_ENV, **AWS_ENV},
        arguments=on('aws-cluster'),
    ),
    # Per-cluster opensearch_header_auth: header credentials, header URL ignored.
    Case(
        'multi/header_auth_cluster',
        clusters=CLUSTERS,
        arguments=on('header-cluster'),
        headers={
            'opensearch-url': 'http://other-cluster.invalid:9200',
            'authorization': basic_header('header-user', 'header-password'),
        },
    ),
    Case(
        'multi/header_auth_cluster_aws_headers',
        clusters=CLUSTERS,
        arguments=on('header-cluster'),
        headers={**HEADER_CREDENTIALS, 'aws-region': 'eu-central-1', 'aws-service-name': 'aoss'},
    ),
    # A cluster without opensearch_header_auth ignores headers, OPENSEARCH_HEADER_AUTH too.
    Case(
        'multi/headers_ignored_without_cluster_header_auth',
        clusters=CLUSTERS,
        env=HEADER_AUTH,
        arguments=on('basic-cluster'),
        headers={'authorization': basic_header('header-user', 'header-password')},
    ),
]


@pytest.mark.parametrize('case', MULTI_CASES, ids=lambda case: case.name)
async def test_multi_mode_auth(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------

HTTPS = {'OPENSEARCH_URL': FAKE_HTTPS_URL, **BASIC_ENV}

TLS_CASES = [
    Case('tls/https_default', env=HTTPS),
    Case('tls/ssl_verify_false', env={**HTTPS, 'OPENSEARCH_SSL_VERIFY': 'false'}),
    # TLS options mean nothing to an http URL.
    Case(
        'tls/ssl_verify_false_plain_http',
        env={**BASIC_ENV, 'OPENSEARCH_SSL_VERIFY': 'false'},
    ),
    Case('tls/ca_cert_path', env={**HTTPS, 'OPENSEARCH_CA_CERT_PATH': CERT_FILE}),
    Case(
        'tls/client_cert_paths',
        env={
            **HTTPS,
            'OPENSEARCH_CLIENT_CERT_PATH': CERT_FILE,
            'OPENSEARCH_CLIENT_KEY_PATH': KEY_FILE,
        },
    ),
    Case(
        'tls/client_cert_paths_ssl_verify_false',
        env={
            **HTTPS,
            'OPENSEARCH_SSL_VERIFY': 'false',
            'OPENSEARCH_CA_CERT_PATH': CERT_FILE,
            'OPENSEARCH_CLIENT_CERT_PATH': CERT_FILE,
            'OPENSEARCH_CLIENT_KEY_PATH': KEY_FILE,
        },
    ),
    Case(
        'tls/missing_ca_cert_file',
        env={**HTTPS, 'OPENSEARCH_CA_CERT_PATH': MISSING_FILE},
        is_error=True,
    ),
    Case(
        'tls/missing_client_key_file',
        env={
            **HTTPS,
            'OPENSEARCH_CLIENT_CERT_PATH': CERT_FILE,
            'OPENSEARCH_CLIENT_KEY_PATH': MISSING_FILE,
        },
        is_error=True,
    ),
    Case(
        'tls/client_cert_without_key',
        env={**HTTPS, 'OPENSEARCH_CLIENT_CERT_PATH': CERT_FILE},
        is_error=True,
    ),
    Case(
        'tls/multi_cluster_tls_settings',
        clusters={
            'tls-cluster': {
                'opensearch_url': FAKE_HTTPS_URL,
                'opensearch_no_auth': True,
                'ssl_verify': False,
                'timeout': 9,
                'opensearch_ca_cert_path': CERT_FILE,
                'opensearch_client_cert_path': CERT_FILE,
                'opensearch_client_key_path': KEY_FILE,
            }
        },
        arguments=on('tls-cluster'),
    ),
]


@pytest.mark.parametrize('case', TLS_CASES, ids=lambda case: case.name)
async def test_tls(run_case, case):
    await run_case(case)
