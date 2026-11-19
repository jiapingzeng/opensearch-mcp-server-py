# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: connection settings edge cases.

Cases the coverage checkpoint (``coverage-checkpoint-v252``) found no characterization test
for, all in ``src/opensearch/client.py``: URLs without a host or scheme, a password-less
userinfo, a non-http(s) scheme, invalid ``OPENSEARCH_MAX_RESPONSE_SIZE`` and timeout values,
and TLS file edge cases. Each ``Case`` names the source lines it covers (as of the
checkpoint); those line numbers stay out of the goldens, so a refactor that only moves code
leaves them unchanged.

Each case makes one ``tools/call`` of ``ListIndexTool`` (``include_detail: false``) through
``mcp_session`` and goldens, under ``golden/gaps_connection_config/``:

- ``setup``: mode, environment, cluster config and the call's arguments;
- ``result``: the whole call result;
- ``requests``: the ``fake_opensearch`` request log;
- ``connections``: per opensearch-py connection the client built (in order), the settings it
  got: host, URL prefix, SSL, timeout, response size limit, redirects and TLS files;
- ``logs``: every warning and error the server logs while it starts and serves the call
  (logger, level, message and the fields of a structured event).

In single mode the server checks the version at startup (``GET /``); each call then opens a
client for the version check and another for ``_cat/indices``, so a setting is read, and a
warning about it logged, up to three times.
"""

import logging
import opensearch.client
import os
import pytest
import sys
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


AREA = 'gaps_connection_config'
TOOL = 'ListIndexTool'
ARGS = {'include_detail': False}

FAKE_URL = 'http://fake-opensearch.invalid:9200'
FAKE_HTTPS_URL = 'https://fake-opensearch.invalid:9200'

CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
]

NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
# No OPENSEARCH_URL: dynamic (per-call) connection mode turns on by itself.
NO_URL = {'OPENSEARCH_URL': None}
SSRF_GUARD = {'OPENSEARCH_SSRF_GUARD': 'true'}


@dataclass(frozen=True)
class TmpPath:
    """A path below the test's ``tmp_path``, resolved when the case runs."""

    name: str


CERT_FILE = TmpPath('client-cert.pem')
KEY_FILE = TmpPath('client-key.pem')
UNREADABLE_FILE = TmpPath('unreadable-ca.pem')


@dataclass(frozen=True)
class Case:
    """One ``tools/call`` of ``ListIndexTool``: golden name (below the area) and setup.

    ``gap`` names the lines of ``src/opensearch/client.py`` the case covers (for the
    reader; not goldened). ``clusters`` set means multi mode with that config file.
    """

    name: str
    gap: str
    env: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=lambda: dict(ARGS))
    clusters: dict | None = None
    is_error: bool = False


# ---------------------------------------------------------------------------
# recorders
# ---------------------------------------------------------------------------


def _setting(value: Any) -> Any:
    """A connection setting as JSON: opensearch-py's default sentinels become '<default>'."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return '<default>' if type(value) is object else repr(value)


@pytest.fixture
def connections(monkeypatch) -> list[dict]:
    """The settings of every connection the server's client builds, in order."""
    built: list[dict] = []
    # The class the server's client module (as pinned by the harness) passes opensearch-py.
    connection_class = sys.modules['opensearch.client'].BufferedAsyncHttpConnection
    original_init = connection_class.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(
            {
                'host': self.host,
                'url_prefix': self.url_prefix,
                'use_ssl': self.use_ssl,
                'timeout': self.timeout,
                'max_response_size': self.max_response_size,
                'follow_redirects': self.follow_redirects,
                'verify_certs': _setting(kwargs.get('verify_certs', object())),
                'ca_certs': _setting(kwargs.get('ca_certs')),
                'client_cert': _setting(kwargs.get('client_cert')),
                'client_key': _setting(kwargs.get('client_key')),
                'http_auth': None if kwargs.get('http_auth') is None else '<set>',
            }
        )

    monkeypatch.setattr(connection_class, '__init__', recording_init)
    return built


# The attributes every LogRecord has; anything else on a record came from `extra=`.
_STANDARD_RECORD_FIELDS = frozenset(logging.LogRecord('', 0, '', 0, '', (), None).__dict__) | {
    'message',
    'asctime',
}


# The server's source tree (src/): only records logged from code under it are kept, so a
# library's or the event loop's own warnings (which can differ between OSes) stay out.
SERVER_SOURCE_ROOT = Path(opensearch.client.__file__).resolve().parents[1]


def _logged_by_server(record: logging.LogRecord) -> bool:
    try:
        return Path(record.pathname).resolve().is_relative_to(SERVER_SOURCE_ROOT)
    except (OSError, ValueError):
        return False


def server_logs(caplog) -> list[dict]:
    """Every warning and error the server logs, with the fields passed as ``extra``.

    Durations in a structured event (``duration_ms``) are normalized by the harness.
    """
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{
                name: _setting(value)
                for name, value in record.__dict__.items()
                if name not in _STANDARD_RECORD_FIELDS
            },
        }
        for record in caplog.records
        if record.levelno >= logging.WARNING and _logged_by_server(record)
    ]


def write_client_cert(cert_path: Path, key_path: Path) -> None:
    """A self-signed EC certificate and its key, as PEM."""
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
def run_case(mcp_session, fake_opensearch, connections, caplog, golden, normalize, tmp_path):
    """Build the server for the case, make the one call, golden what it did and logged."""

    async def run_case(case: Case) -> dict:
        assert fake_opensearch.url == FAKE_URL
        caplog.set_level(logging.WARNING)
        fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
        write_client_cert(tmp_path / CERT_FILE.name, tmp_path / KEY_FILE.name)
        env = _resolve(case.env, tmp_path)
        config_file = ''
        mode = 'single'
        clusters = None
        if case.clusters is not None:
            mode = 'multi'
            clusters = _resolve(case.clusters, tmp_path)
            config_file = tmp_path / 'clusters.yml'
            with open(config_file, 'w', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump({'clusters': clusters}, f, sort_keys=False)

        async with mcp_session(mode=mode, config_file=config_file, env=env) as session:
            result = await session.call_tool(TOOL, case.arguments)

        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError', False) is case.is_error, result
        data = {
            'setup': {'mode': mode, 'env': env, 'clusters': clusters},
            'call': {'name': TOOL, 'arguments': case.arguments},
            'result': result,
            'requests': fake_opensearch.requests,
            'connections': connections,
            'logs': server_logs(caplog),
        }
        golden(f'{AREA}/{case.name}', normalize(data))
        return data

    return run_case


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

URL_CASES = [
    # A URL with no host (here no "//", so the userinfo-looking text is a path) is
    # rejected, and the error and logs show it redacted whole: the secret in it is not
    # repeated.
    Case(
        'url/no_host_redacted_in_error',
        gap='189-190, 784-785',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'https:char-user:char-secret@fake-opensearch.invalid'},
        is_error=True,
    ),
    # A caller URL with no host, under the SSRF guard: rejected before any DNS lookup, the
    # URL redacted in the error.
    Case(
        'url/ssrf_guard_caller_url_without_host',
        gap='239-240',
        env={**NO_URL, **SSRF_GUARD},
        arguments={
            **ARGS,
            'opensearch_url': 'https://char-user:char-secret@/',
            'opensearch_no_auth': True,
        },
        is_error=True,
    ),
    # characterization: current behaviour. An operator URL with a username but no password
    # and no port gets its default port added with the userinfo kept as "user@"; opensearch-py
    # then fails to build the client (it unquotes the missing password), so no request is
    # made and the call fails with a TypeError's text.
    Case(
        'url/username_without_password',
        gap='285->287',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'http://char-user@fake-opensearch.invalid'},
        is_error=True,
    ),
    # characterization: current behaviour. A non-http(s) scheme without a port gets no port
    # added and is not rejected: opensearch-py connects over plain http to its own default
    # port, 9200.
    Case(
        'url/non_http_scheme_without_port',
        gap='296-297',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'ftp://fake-opensearch.invalid'},
    ),
    # characterization: current behaviour. With "//" but no host after the userinfo, the URL
    # passes validation (it has a netloc) and gets no default port; opensearch-py then fails
    # on the missing host while it sets up the connection, before the connection's __init__
    # finishes, so no connection is recorded and no request is made. That raw TypeError (not
    # a ConfigurationError) is the call's error, and no WARNING or higher log shows the URL.
    Case(
        'url/userinfo_without_host',
        gap='189-190, 296-297',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'https://char-user:char-secret@'},
        is_error=True,
    ),
    Case(
        'url/no_scheme',
        gap='784-785',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'fake-opensearch.invalid'},
        is_error=True,
    ),
    # characterization: current behaviour. "host:port" parses as scheme "host" with path
    # "port", so it is rejected as having no host, like a URL with no scheme.
    Case(
        'url/host_and_port_without_scheme',
        gap='784-785',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'fake-opensearch.invalid:9200'},
        is_error=True,
    ),
    Case(
        'url/scheme_without_host',
        gap='784-785',
        env={**NO_AUTH, 'OPENSEARCH_URL': 'http://'},
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', URL_CASES, ids=lambda case: case.name)
async def test_url_edge_cases(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# OPENSEARCH_MAX_RESPONSE_SIZE
# ---------------------------------------------------------------------------

NO_AUTH_CLUSTER = {'opensearch_url': FAKE_URL, 'opensearch_no_auth': True}


def on(cluster: str, **arguments: Any) -> dict:
    return {**ARGS, 'opensearch_cluster_name': cluster, **arguments}


MAX_RESPONSE_SIZE_CASES = [
    # Single mode: an invalid value logs a warning and the default limit is used.
    Case(
        'max_response_size/single_zero',
        gap='381-385',
        env={**NO_AUTH, 'OPENSEARCH_MAX_RESPONSE_SIZE': '0'},
    ),
    Case(
        'max_response_size/single_negative',
        gap='381-385',
        env={**NO_AUTH, 'OPENSEARCH_MAX_RESPONSE_SIZE': '-10'},
    ),
    Case(
        'max_response_size/single_not_integer',
        gap='386-387',
        env={**NO_AUTH, 'OPENSEARCH_MAX_RESPONSE_SIZE': '10MB'},
    ),
    # Multi mode, a cluster without max_response_size: the environment variable is read.
    Case(
        'max_response_size/multi_env_valid',
        gap='619-622',
        env={'OPENSEARCH_MAX_RESPONSE_SIZE': '4096'},
        clusters={'plain-cluster': NO_AUTH_CLUSTER},
        arguments=on('plain-cluster'),
    ),
    Case(
        'max_response_size/multi_env_zero',
        gap='619-626',
        env={'OPENSEARCH_MAX_RESPONSE_SIZE': '0'},
        clusters={'plain-cluster': NO_AUTH_CLUSTER},
        arguments=on('plain-cluster'),
    ),
    Case(
        'max_response_size/multi_env_not_integer',
        gap='619-621, 627-628',
        env={'OPENSEARCH_MAX_RESPONSE_SIZE': '10MB'},
        clusters={'plain-cluster': NO_AUTH_CLUSTER},
        arguments=on('plain-cluster'),
    ),
    # The cluster's own max_response_size wins; the environment variable is not read, so
    # its invalid value logs nothing.
    Case(
        'max_response_size/multi_cluster_value_over_env',
        gap='617->632',
        env={'OPENSEARCH_MAX_RESPONSE_SIZE': '10MB'},
        clusters={'sized-cluster': {**NO_AUTH_CLUSTER, 'max_response_size': 2048}},
        arguments=on('sized-cluster'),
    ),
]


@pytest.mark.parametrize('case', MAX_RESPONSE_SIZE_CASES, ids=lambda case: case.name)
async def test_max_response_size(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------

TIMEOUT_CASES = [
    Case('timeout/single_zero', gap='809-811', env={**NO_AUTH, 'OPENSEARCH_TIMEOUT': '0'}),
    Case('timeout/single_negative', gap='809-811', env={**NO_AUTH, 'OPENSEARCH_TIMEOUT': '-5'}),
    # A per-call opensearch_timeout of 0, with per-call overrides allowed.
    Case(
        'timeout/argument_zero',
        gap='809-811',
        env={**NO_AUTH, 'OPENSEARCH_DYNAMIC_CONNECTION': 'true'},
        arguments={**ARGS, 'opensearch_timeout': 0},
    ),
    Case(
        'timeout/multi_cluster_negative',
        gap='809-811',
        clusters={'timeout-cluster': {**NO_AUTH_CLUSTER, 'timeout': -1}},
        arguments=on('timeout-cluster'),
    ),
]


@pytest.mark.parametrize('case', TIMEOUT_CASES, ids=lambda case: case.name)
async def test_timeout(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# TLS files
# ---------------------------------------------------------------------------

HTTPS_NO_AUTH = {**NO_AUTH, 'OPENSEARCH_URL': FAKE_HTTPS_URL}


def _running_as_root() -> bool:
    return hasattr(os, 'geteuid') and os.geteuid() == 0


@pytest.mark.skipif(sys.platform == 'win32', reason='file modes do not deny reads on Windows')
@pytest.mark.skipif(_running_as_root(), reason='root reads a file whatever its mode')
async def test_tls_file_exists_but_unreadable(run_case, tmp_path):
    """Gap 1011-1012: a CA bundle that exists but cannot be read fails the client setup."""
    unreadable = tmp_path / UNREADABLE_FILE.name
    unreadable.write_text('not read\n', encoding='utf-8', newline='\n')
    unreadable.chmod(0)
    try:
        if os.access(unreadable, os.R_OK):
            # e.g. a container whose non-root user has CAP_DAC_OVERRIDE
            pytest.skip('this process can read a file with no read permission')
        await run_case(
            Case(
                'tls/ca_cert_file_unreadable',
                gap='1011-1012',
                env={**HTTPS_NO_AUTH, 'OPENSEARCH_CA_CERT_PATH': UNREADABLE_FILE},
                is_error=True,
            )
        )
    finally:
        unreadable.chmod(0o600)


async def test_mtls_without_ssl_verify_or_ca(run_case):
    """Gap 1045-1046: mTLS with verification off and no CA bundle logs a warning."""
    await run_case(
        Case(
            'tls/mtls_ssl_verify_off_without_ca',
            gap='1045-1046',
            env={
                **HTTPS_NO_AUTH,
                'OPENSEARCH_SSL_VERIFY': 'false',
                'OPENSEARCH_CLIENT_CERT_PATH': CERT_FILE,
                'OPENSEARCH_CLIENT_KEY_PATH': KEY_FILE,
            },
        )
    )
