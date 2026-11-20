# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: AWS profile and region resolution.

Pins the branches of ``get_aws_region_single_mode`` and ``get_aws_region_multi_mode``
(``src/opensearch/client.py``) the coverage checkpoint found unreached: a profile with no
region, the default boto3 session failing, and in multi mode a cluster profile with no
region or one that does not exist.

Each case makes one ``tools/call`` of ``ListIndexTool`` (``include_detail: false``) through
``mcp_session`` and goldens, under ``golden/gaps_aws_region/``:

- ``setup``: mode, environment, ``--profile``, cluster config and the AWS config and
  credentials files (temporary files under ``tmp_path``, passed in ``env``);
- ``result``: the whole call result;
- ``requests``: the ``fake_opensearch`` request log (the SigV4 region and service it
  parsed from the request) and, per request, which fake identity signed it;
- ``region_log``: what the two region functions logged (which branch ran);
- ``client_warnings``: the warnings the client setup logs;
- ``aws_api_calls``: every botocore API call. None is expected and any one fails the test.

Nothing reaches AWS: the harness blocks the network and keeps instance metadata off, the
botocore API boundary is intercepted here, and every key is a fixed fake.
"""

import botocore.client
import logging
import pytest
import re
import yaml
from dataclasses import dataclass, field
from opensearchpy.connection.http_async import AsyncHttpConnection
from pathlib import Path
from typing import Any


AREA = 'gaps_aws_region'
TOOL = 'ListIndexTool'
ARGS = {'include_detail': False}

FAKE_URL = 'http://fake-opensearch.invalid:9200'

CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
]

# Fake AWS identities: each has its own access key id, so a golden names which one signed.
DEFAULT_KEY = 'AKIACHARDEFAULT00001'
NO_REGION_KEY = 'AKIACHARNOREGION0001'
WEST_KEY = 'AKIACHARWEST00000001'
ENV_KEY = 'AKIACHARENV000000001'
IDENTITIES = {
    DEFAULT_KEY: 'profile:default',
    NO_REGION_KEY: 'profile:char-noregion',
    WEST_KEY: 'profile:char-west',
    ENV_KEY: 'env',
}

# Keys live in the shared credentials file, regions in the config file, as the AWS CLI
# writes them. ``char-noregion`` has keys but no region anywhere.
CREDENTIALS = f"""\
[default]
aws_access_key_id = {DEFAULT_KEY}
aws_secret_access_key = char-default-secret

[char-noregion]
aws_access_key_id = {NO_REGION_KEY}
aws_secret_access_key = char-noregion-secret

[char-west]
aws_access_key_id = {WEST_KEY}
aws_secret_access_key = char-west-secret
"""

# No [default] section: the default session has no region of its own.
CONFIG = """\
[profile char-noregion]
output = json

[profile char-west]
region = eu-west-1
"""

# The same, plus a [default] section with a region.
CONFIG_DEFAULT_REGION = """\
[default]
region = ap-northeast-1

[profile char-noregion]
output = json

[profile char-west]
region = eu-west-1
"""

AWS_FILES = {
    'aws_config': CONFIG,
    'aws_config_default_region': CONFIG_DEFAULT_REGION,
    'aws_credentials': CREDENTIALS,
}

MISSING_PROFILE = 'char-no-such-profile'
AWS_ENV_KEYS = {'AWS_ACCESS_KEY_ID': ENV_KEY, 'AWS_SECRET_ACCESS_KEY': 'char-env-secret'}

REGION_FUNCTIONS = ('get_aws_region_single_mode', 'get_aws_region_multi_mode')
CLIENT_LOGGER = 'opensearch.client'


@dataclass(frozen=True)
class TmpPath:
    """A path below the test's ``tmp_path``, resolved when the case runs."""

    name: str


def aws_files(config: str = 'aws_config') -> dict:
    """``env`` pointing boto3 at this test's own AWS config and credentials files."""
    return {
        'AWS_CONFIG_FILE': TmpPath(config),
        'AWS_SHARED_CREDENTIALS_FILE': TmpPath('aws_credentials'),
    }


AWS = aws_files()
AWS_DEFAULT_REGION_PROFILE = aws_files('aws_config_default_region')


@dataclass(frozen=True)
class Case:
    """One ``tools/call`` of ``ListIndexTool``: golden name (below ``gaps_aws_region/``).

    ``clusters`` set means multi mode with that cluster config file.
    """

    name: str
    env: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=lambda: dict(ARGS))
    clusters: dict | None = None
    profile: str = ''
    is_error: bool = False


def _resolve(value: Any, tmp_path: Path) -> Any:
    if isinstance(value, TmpPath):
        return str(tmp_path / value.name)
    if isinstance(value, dict):
        return {k: _resolve(v, tmp_path) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# recorders
# ---------------------------------------------------------------------------

CREDENTIAL = re.compile(r'Credential=([^/,\s]+)/')


def _identity(headers: dict | None) -> str | None:
    """Which fake identity signed a request (from the access key id), never a secret."""
    authorization = next(
        (str(v) for k, v in (headers or {}).items() if str(k).lower() == 'authorization'), ''
    )
    if not authorization.lower().startswith('aws4-hmac-sha256'):
        return None
    credential = CREDENTIAL.search(authorization)
    key = credential.group(1) if credential else None
    return IDENTITIES.get(key, '<unknown access key>')


@pytest.fixture
def signers(monkeypatch, fake_opensearch) -> list[str | None]:
    """Per request the fake receives (same order as its log): the identity that signed it."""
    signed: list[str | None] = []
    # The harness's fake session factory, installed by ``fake_opensearch``.
    create_fake_session = AsyncHttpConnection._create_aiohttp_session

    async def recording_create_session(connection) -> None:
        await create_fake_session(connection)
        request = connection.session.request

        def recording_request(method, url, data=None, auth=None, headers=None, **options):
            signed.append(_identity(headers))
            return request(method, url, data=data, auth=auth, headers=headers, **options)

        connection.session.request = recording_request

    monkeypatch.setattr(AsyncHttpConnection, '_create_aiohttp_session', recording_create_session)
    return signed


@pytest.fixture(autouse=True)
def aws_api(monkeypatch) -> list[dict]:
    """Every botocore API call, intercepted: none is expected, any one fails the test."""
    calls: list[dict] = []

    def fake_api_call(client, operation_name, api_params):
        calls.append(
            {
                'service': client.meta.service_model.service_name,
                'region': client.meta.region_name,
                'operation': operation_name,
            }
        )
        raise AssertionError('unexpected AWS API call in a characterization test')

    monkeypatch.setattr(botocore.client.BaseClient, '_make_api_call', fake_api_call)
    yield calls
    if calls:
        pytest.fail(f'AWS API calls attempted: {calls}', pytrace=False)


def region_log(caplog) -> list[dict]:
    """What the region functions logged, in order: the branches they took."""
    return [
        {'function': record.funcName, 'level': record.levelname, 'message': record.getMessage()}
        for record in caplog.records
        if record.name == CLIENT_LOGGER and record.funcName in REGION_FUNCTIONS
    ]


def client_warnings(caplog) -> list[str]:
    """Warnings the client setup logs (fallbacks it takes, inputs it ignores)."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == CLIENT_LOGGER
        and record.levelno == logging.WARNING
        and not hasattr(record, 'event_type')
    ]


@pytest.fixture
def run_case(mcp_session, fake_opensearch, signers, aws_api, caplog, golden, normalize, tmp_path):
    """Build the server for the case, make the one call, golden what reached OpenSearch."""

    async def run_case(case: Case) -> dict:
        assert fake_opensearch.url == FAKE_URL
        caplog.set_level(logging.DEBUG, logger=CLIENT_LOGGER)
        fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
        for name, text in AWS_FILES.items():
            with open(tmp_path / name, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
        env = _resolve(case.env, tmp_path)
        # Never the developer's own AWS files: every case names its temporary ones.
        assert {'AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE'} <= set(env), env
        config_file = ''
        mode = 'single'
        if case.clusters is not None:
            mode = 'multi'
            config_file = tmp_path / 'clusters.yml'
            with open(config_file, 'w', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump({'clusters': case.clusters}, f, sort_keys=False)

        async with mcp_session(
            mode=mode, config_file=config_file, env=env, profile=case.profile
        ) as session:
            result = await session.call_tool(TOOL, case.arguments)

        # Guards the golden against recording a harness failure (e.g. an unmatched route).
        assert result.get('isError') is case.is_error, result
        assert len(signers) == len(fake_opensearch.requests), (signers, fake_opensearch.requests)
        files = {
            Path(path).name: AWS_FILES[Path(path).name].splitlines()
            for name, path in env.items()
            if name in ('AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE')
        }
        data = {
            'setup': {
                'mode': mode,
                'env': env,
                'aws_files': files,
                'clusters': case.clusters,
                'profile': case.profile or None,
            },
            'call': {'name': TOOL, 'arguments': case.arguments},
            'result': result,
            'requests': [
                {**logged, 'signed_by': identity}
                for logged, identity in zip(fake_opensearch.requests, signers)
            ],
            'region_log': region_log(caplog),
            'client_warnings': client_warnings(caplog),
            'aws_api_calls': aws_api,
        }
        golden(f'{AREA}/{case.name}', normalize(data))
        return data

    return run_case


# ---------------------------------------------------------------------------
# single mode
# ---------------------------------------------------------------------------

SINGLE_CASES = [
    # client.py 1078->1085: AWS_PROFILE names a profile with no region, so the region
    # lookup falls through to the default session, which reads the same AWS_PROFILE and
    # finds no region either. AWS auth then fails for want of a region; nothing is sent.
    Case(
        'single/aws_profile_without_region',
        env={**AWS, 'AWS_PROFILE': 'char-noregion'},
        is_error=True,
    ),
    # characterization: current behaviour. --profile names a profile with no region: the
    # region comes from the default session's [default] profile while the request is signed
    # with the --profile credentials (1078->1085, then 1086-1090).
    Case(
        'single/cli_profile_without_region_default_profile_region',
        env=AWS_DEFAULT_REGION_PROFILE,
        profile='char-noregion',
    ),
    # characterization: current behaviour. --profile wins over AWS_PROFILE for the
    # credentials, but when the --profile profile has no region the default session reads
    # AWS_PROFILE, so the region comes from the other profile (char-west, eu-west-1).
    Case(
        'single/cli_profile_without_region_aws_profile_env_region',
        env={**AWS, 'AWS_PROFILE': 'char-west'},
        profile='char-noregion',
    ),
    # client.py 1091-1092: AWS_PROFILE names a profile that does not exist. boto3 raises
    # ProfileNotFound for the profile session (1081-1082) and again for the default session,
    # which reads AWS_PROFILE (1091-1092): no region.
    # characterization: current behaviour. The client setup then falls back to a bare
    # session after the profile fails, which raises the same ProfileNotFound outside any
    # handler, so the call fails as an "unexpected error" ConfigurationError.
    Case(
        'single/default_session_fails_missing_aws_profile',
        env={**AWS, 'AWS_PROFILE': MISSING_PROFILE},
        is_error=True,
    ),
    # The same with AWS_REGION set: the region comes first, so neither session is built for
    # it (no region warnings), and the client setup still fails on the missing profile.
    Case(
        'single/missing_aws_profile_with_aws_region',
        env={**AWS, 'AWS_PROFILE': MISSING_PROFILE, 'AWS_REGION': 'us-east-2'},
        is_error=True,
    ),
    # 1078->1085 then 1091-1092: --profile has no region and the default session fails on
    # the missing AWS_PROFILE. The --profile session still builds the client, which has no
    # region: AWS auth fails.
    Case(
        'single/cli_profile_without_region_default_session_fails',
        env={**AWS, 'AWS_PROFILE': MISSING_PROFILE},
        profile='char-noregion',
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', SINGLE_CASES, ids=lambda case: case.name)
async def test_single_mode_region(run_case, case):
    await run_case(case)


# ---------------------------------------------------------------------------
# multi mode
# ---------------------------------------------------------------------------

CLUSTERS = {
    'noregion-cluster': {
        'opensearch_url': 'http://noregion-cluster.invalid:9200',
        'profile': 'char-noregion',
    },
    'noregion-serverless-cluster': {
        'opensearch_url': 'http://noregion-serverless-cluster.invalid:9200',
        'profile': 'char-noregion',
        'is_serverless': True,
    },
    'missing-profile-cluster': {
        'opensearch_url': 'http://missing-profile-cluster.invalid:9200',
        'profile': MISSING_PROFILE,
    },
}


def on(cluster: str) -> dict:
    return {**ARGS, 'opensearch_cluster_name': cluster}


MULTI_CASES = [
    # client.py 1131->1142: the cluster's profile has no region, so the single-mode lookup
    # runs and AWS_REGION gives it; the cluster profile's credentials sign.
    Case(
        'multi/cluster_profile_without_region_aws_region_env',
        clusters=CLUSTERS,
        env={**AWS, 'AWS_REGION': 'us-east-2'},
        arguments=on('noregion-cluster'),
    ),
    # The same for a serverless cluster: signed for aoss.
    Case(
        'multi/serverless_cluster_profile_without_region_aws_region_env',
        clusters=CLUSTERS,
        env={**AWS, 'AWS_REGION': 'us-east-2'},
        arguments=on('noregion-serverless-cluster'),
    ),
    # 1131->1142 with no region anywhere: AWS auth fails; nothing is sent.
    Case(
        'multi/cluster_profile_without_region_no_region',
        clusters=CLUSTERS,
        env=AWS,
        arguments=on('noregion-cluster'),
        is_error=True,
    ),
    # characterization: current behaviour. 1131->1142, then the default session's
    # [default] profile gives the region; the cluster profile's credentials sign.
    Case(
        'multi/cluster_profile_without_region_default_profile_region',
        clusters=CLUSTERS,
        env=AWS_DEFAULT_REGION_PROFILE,
        arguments=on('noregion-cluster'),
    ),
    # client.py 1136-1137: the cluster's profile does not exist; boto3 raises
    # ProfileNotFound, logged as a warning, and the single-mode lookup gives the [default]
    # region. The client setup then fails the call with the profile named (its warning is
    # still logged): nothing is sent, the [default] credentials are not used (13g, fixed).
    Case(
        'multi/cluster_profile_missing_default_profile',
        clusters=CLUSTERS,
        env=AWS_DEFAULT_REGION_PROFILE,
        arguments=on('missing-profile-cluster'),
        is_error=True,
    ),
    # 1136-1137 with AWS_PROFILE set: the region comes from AWS_PROFILE (char-west), but its
    # credentials are not used for the cluster's own (missing) profile: the call fails.
    Case(
        'multi/cluster_profile_missing_aws_profile_env',
        clusters=CLUSTERS,
        env={**AWS, 'AWS_PROFILE': 'char-west'},
        arguments=on('missing-profile-cluster'),
        is_error=True,
    ),
    # 1136-1137 with AWS_REGION and AWS keys in the environment: the environment's keys are
    # not used for the cluster's own (missing) profile: the call fails.
    Case(
        'multi/cluster_profile_missing_env_keys',
        clusters=CLUSTERS,
        env={**AWS, **AWS_ENV_KEYS, 'AWS_REGION': 'us-east-2'},
        arguments=on('missing-profile-cluster'),
        is_error=True,
    ),
]


@pytest.mark.parametrize('case', MULTI_CASES, ids=lambda case: case.name)
async def test_multi_mode_region(run_case, case):
    await run_case(case)
