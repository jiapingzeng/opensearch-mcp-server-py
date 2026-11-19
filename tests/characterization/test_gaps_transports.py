# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: stdio entry points and executor edge cases.

Pins, under ``golden/gaps_transports/``, the paths the coverage checkpoint found unreached
around the stdio server (``stdio_server.serve``) and the tool executor
(``tool_executor.execute_tool``), both of which the planned migration replaces:

- ``stdio_profile/...``: ``--profile`` given to the stdio transport
  (``stdio_server.py``: ``if profile: set_profile(profile)``). The real entry point runs in a
  subprocess (``stdio_launcher``) with ``--debug --log-format json`` and this test's own AWS
  config and credentials files; each case pins the ``tools/call`` responses, what reached
  the fake OpenSearch (the SigV4 region and service it parsed) and the log lines that name
  the profile and the region the server took.
- ``python_m/...``: the server started as ``python -m mcp_server_opensearch``
  (``__main__.py``, through ``stdio_launcher(entry='module')``): its ``initialize`` and
  ``tools/list`` responses, which must equal those of the console-script entry (``main()``).
- ``empty_result/...``: a planted tool returning empty content (``[]``) or ``None``
  (``tool_executor.py``: the soft-error check skipped), in-process and over stdio: the
  ``tools/call`` result and the ``tool_execution`` log record.
- ``mcp_error/...``: a planted tool raising the SDK's MCP error over stdio
  (``stdio_server.py``: ``except MCPError: raise``): the JSON-RPC error the client gets, the
  ``tool_execution`` record and that the session goes on.

Planted tools are left without ``raises`` when they return: ``plant_tool(name,
raises=None)`` raises ``TypeError`` (``None`` is no exception).
"""

import json
import logging
import pytest
import yaml
from dataclasses import dataclass, field


AREA = 'gaps_transports'

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
EXECUTOR_LOGGER = 'mcp_server_opensearch.tool_executor'
# Every tool this module plants starts with this.
PLANTED_PREFIX = 'CharGapPlanted'

# The fields of the `tool_execution` record (tool_executor.py) a golden keeps. The JSON log
# line has more (`timestamp`, and on some Python versions `taskName`); these are the event.
EVENT_FIELDS = (
    'event_type',
    'tool_name',
    'tool_key',
    'status',
    'error_type',
    'duration_ms',
    'client_name',
)

CAT_INDICES = [
    {'health': 'green', 'status': 'open', 'index': 'logs-2020.01.01', 'docs.count': '1200'},
]
TOOL = 'ListIndexTool'
ARGS = {'include_detail': False}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def json_log_lines(stderr: str) -> list[dict]:
    """The JSON log lines of a ``--log-format json`` server's stderr (other lines skipped)."""
    lines = []
    for line in stderr.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            lines.append(entry)
    return lines


def event_record(entry: dict) -> dict:
    """A ``tool_execution`` event as a golden keeps it: logger, level, message and fields."""
    return {
        'logger': entry.get('logger'),
        'level': entry.get('level'),
        'message': entry.get('message'),
        **{name: entry[name] for name in EVENT_FIELDS if name in entry},
    }


def stderr_tool_records(stderr: str) -> list[dict]:
    """The ``tool_execution`` events a launched server wrote to stderr."""
    return [
        event_record(entry)
        for entry in json_log_lines(stderr)
        if entry.get('event_type') == 'tool_execution'
    ]


def caplog_tool_records(caplog) -> list[dict]:
    """The ``tool_execution`` events this process logged, as ``stderr_tool_records`` gives them."""
    return [
        event_record(
            {
                'logger': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
            }
        )
        for record in caplog.records
        if getattr(record, 'event_type', None) == 'tool_execution'
    ]


def without_id(response: dict) -> dict:
    """A JSON-RPC response without its ``id``: stdio counts ids up, ``mcp_session.raw`` sends 1."""
    return {key: value for key, value in response.items() if key != 'id'}


def has_traceback(stderr: str) -> bool:
    return 'Traceback (most recent call last)' in stderr


# ---------------------------------------------------------------------------
# --profile over stdio
# ---------------------------------------------------------------------------

# Fake AWS identities. The fake OpenSearch records the SigV4 region and service of a request,
# never its key, so each profile has its own region: the region a request is signed for names
# the profile. `char-keyless` has a region but no keys anywhere. No [default] section: the
# default session has no region and no keys of its own.
AWS_CREDENTIALS = """\
[char-east]
aws_access_key_id = AKIACHAREAST00000001
aws_secret_access_key = char-east-secret

[char-west]
aws_access_key_id = AKIACHARWEST00000001
aws_secret_access_key = char-west-secret
"""

AWS_CONFIG = """\
[profile char-east]
region = us-east-2

[profile char-west]
region = eu-west-1

[profile char-keyless]
region = us-west-2
"""

AWS_FILES = {'aws_config': AWS_CONFIG, 'aws_credentials': AWS_CREDENTIALS}

# The log lines that name the profile and region the server took: global_state's setter,
# client.py's region lookup and its client setup. Matched by logger and message start (never
# by a word anywhere: a temp path holds the test's name, which says "profile").
PROFILE_LOGGERS = ('mcp_server_opensearch.global_state', 'opensearch.client')
PROFILE_MESSAGES = (
    'Set global profile to:',
    'Using AWS_REGION',
    'Using region from',
    'Using cluster-specific AWS region',
    'Using default boto3 session region',
    'No AWS region found',
    'Failed to get region',
    'Failed to create boto3 session',
    '[AWS CREDS]',
    'Initializing multi mode OpenSearch client',
)
# Every launch of these tests logs at DEBUG, as JSON lines, so the lines above reach stderr.
LOG_ARGS = ('--debug', '--log-format', 'json')

CLUSTERS = {
    # No profile of its own: the client setup takes --profile.
    'no-profile-cluster': {'opensearch_url': 'http://no-profile-cluster.invalid:9200'},
    # Its own profile wins over --profile.
    'east-profile-cluster': {
        'opensearch_url': 'http://east-profile-cluster.invalid:9200',
        'profile': 'char-east',
    },
}


@dataclass(frozen=True)
class ProfileCase:
    """One launched stdio server: golden name (below ``stdio_profile/``) and what it gets.

    ``clusters`` set means multi mode with that cluster config file; ``calls`` are the
    ``ListIndexTool`` arguments, one ``tools/call`` each.
    """

    name: str
    profile: str = ''
    env: dict = field(default_factory=dict)
    clusters: dict | None = None
    calls: tuple = (ARGS,)
    errors: tuple = (False,)


def on(cluster: str) -> dict:
    return {**ARGS, 'opensearch_cluster_name': cluster}


PROFILE_CASES = [
    # stdio_server.py 31->32: --profile is set, and wins over AWS_PROFILE: the request is
    # signed for char-west's region (eu-west-1), not char-east's.
    # characterization: current behaviour. The region lookup logs the --profile profile as
    # "Using region from AWS_PROFILE 'char-west'", though AWS_PROFILE names char-east.
    ProfileCase('single_cli_profile', profile='char-west', env={'AWS_PROFILE': 'char-east'}),
    # The same server without --profile (31->35): AWS_PROFILE gives the profile.
    ProfileCase('single_no_cli_profile', env={'AWS_PROFILE': 'char-east'}),
    # --profile names a profile with a region but no keys: the credentials come from it too,
    # not from AWS_PROFILE's (which has keys), so AWS auth fails and nothing is sent.
    ProfileCase(
        'single_cli_profile_without_keys',
        profile='char-keyless',
        env={'AWS_PROFILE': 'char-east'},
        errors=(True,),
    ),
    # Multi mode: a cluster without a profile takes --profile (over AWS_PROFILE); one with
    # its own profile keeps it.
    ProfileCase(
        'multi_cli_profile',
        profile='char-west',
        env={'AWS_PROFILE': 'char-east'},
        clusters=CLUSTERS,
        calls=(on('no-profile-cluster'), on('east-profile-cluster')),
        errors=(False, False),
    ),
]


def profile_log(stderr: str) -> list[dict]:
    """What the server logged about the profile and region it took, in order."""
    return [
        {'logger': entry['logger'], 'level': entry['level'], 'message': entry['message']}
        for entry in json_log_lines(stderr)
        if entry.get('logger') in PROFILE_LOGGERS
        and 'event_type' not in entry
        and str(entry.get('message', '')).startswith(PROFILE_MESSAGES)
    ]


def request_summary(requests: list[dict]) -> list[dict]:
    return [
        {key: r[key] for key in ('method', 'host', 'path', 'params', 'auth')} for r in requests
    ]


@pytest.fixture
def profile_setup(tmp_path, fake_opensearch):
    """Write this test's AWS files (and cluster config); return the launch args and env."""

    def setup(case: ProfileCase) -> tuple[list[str], dict, dict]:
        fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
        for name, text in AWS_FILES.items():
            with open(tmp_path / name, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
        env = {
            'AWS_CONFIG_FILE': str(tmp_path / 'aws_config'),
            'AWS_SHARED_CREDENTIALS_FILE': str(tmp_path / 'aws_credentials'),
            **case.env,
        }
        args = ['--mode', 'single']
        if case.clusters is not None:
            config_file = tmp_path / 'clusters.yml'
            with open(config_file, 'w', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump({'clusters': case.clusters}, f, sort_keys=False)
            args = ['--mode', 'multi', '--config', str(config_file)]
        if case.profile:
            args += ['--profile', case.profile]
        args += LOG_ARGS
        setup_data = {
            'args': args,
            'env': env,
            'aws_files': {name: text.splitlines() for name, text in AWS_FILES.items()},
            'clusters': case.clusters,
        }
        return args, env, setup_data

    return setup


@pytest.mark.parametrize('case', PROFILE_CASES, ids=lambda case: case.name)
async def test_stdio_profile(
    stdio_launcher, fake_opensearch, profile_setup, golden, normalize, case
):
    args, env, setup = profile_setup(case)
    calls = [{'name': TOOL, 'arguments': arguments} for arguments in case.calls]
    async with stdio_launcher(*args, env=env) as server:
        responses = [await server.raw('tools/call', params) for params in calls]

    assert server.returncode == 0, server.stderr
    # Guards the golden against recording a harness failure (e.g. an unmatched route).
    assert [r['result']['isError'] for r in responses] == list(case.errors), responses
    golden(
        f'{AREA}/stdio_profile/{case.name}',
        normalize(
            {
                'setup': setup,
                'calls': calls,
                'responses': responses,
                'requests': request_summary(fake_opensearch.requests),
                'profile_log': profile_log(server.stderr),
                'returncode': server.returncode,
            }
        ),
    )


async def test_stdio_profile_matches_in_process(
    stdio_launcher, mcp_session, fake_opensearch, profile_setup, normalize
):
    # The stdio server and create_mcp_server (what mcp_session and the HTTP transports build)
    # take --profile the same way: same response, same requests signed for the same region.
    case = PROFILE_CASES[0]
    args, env, _ = profile_setup(case)
    params = {'name': TOOL, 'arguments': ARGS}
    async with stdio_launcher(*args, env=env) as server:
        launched = await server.raw('tools/call', params)
    launched_requests = request_summary(fake_opensearch.requests)
    fake_opensearch.requests.clear()
    async with mcp_session(env=env, profile=case.profile) as session:
        in_process = await session.raw('tools/call', params)

    assert launched_requests == request_summary(fake_opensearch.requests)
    assert {r['auth']['region'] for r in launched_requests} == {'eu-west-1'}
    assert normalize(without_id(launched)) == normalize(without_id(in_process))


# ---------------------------------------------------------------------------
# python -m mcp_server_opensearch
# ---------------------------------------------------------------------------

ENTRY_ARGS = [
    pytest.param((), id='default'),
    pytest.param(('--mode', 'multi'), id='multi'),
]


@pytest.mark.parametrize('args', ENTRY_ARGS)
async def test_python_m_serves_what_main_serves(stdio_launcher, golden, normalize, request, args):
    served = {}
    for entry in ('module', 'main'):
        async with stdio_launcher(*args, env=NO_AUTH, entry=entry) as server:
            initialize = await server.initialize()
            tools = await server.raw('tools/list')
            started = server.entry()
        served[entry] = {
            'entry': started,
            'initialize': initialize,
            'tools_list': tools,
            'returncode': server.returncode,
            'notifications': server.notifications,
        }

    module, main = served['module'], served['main']
    assert module['entry']['entry'] == 'module'
    assert module['entry']['main_module'] == 'mcp_server_opensearch.__main__'
    assert main['entry']['main_module'] is None
    assert module['entry']['args'] == main['entry']['args'] == list(args)
    # Everything but how the process entered the server is the same.
    assert normalize({**module, 'entry': None}) == normalize({**main, 'entry': None})
    golden(
        f'{AREA}/python_m/{request.node.callspec.id}',
        normalize({**module, 'main_entry': main['entry']}),
    )


# ---------------------------------------------------------------------------
# planted tool returning empty content or None
# ---------------------------------------------------------------------------

EMPTY_RESULTS = [
    pytest.param([], id='empty_list'),
    pytest.param(None, id='none'),
]


@pytest.mark.parametrize('content', EMPTY_RESULTS)
async def test_planted_tool_empty_result(
    plant_tool, mcp_session, stdio_launcher, caplog, golden, normalize, request, content
):
    # tool_executor.py 93->98: no content, so no soft-error check: a success, empty content.
    caplog.set_level(logging.INFO, logger=EXECUTOR_LOGGER)
    name = f'{PLANTED_PREFIX}Returns'
    planted = plant_tool(name, returns=content)
    params = {'name': name, 'arguments': {}}
    async with mcp_session(env=NO_AUTH) as session:
        client_result = await session.call_tool(name, {})
        in_process = await session.raw('tools/call', params)
    in_process_log = caplog_tool_records(caplog)
    async with stdio_launcher('--log-format', 'json', env=NO_AUTH) as server:
        launched = await server.raw('tools/call', params)
    launched_log = stderr_tool_records(server.stderr)

    assert planted.calls == [{'opensearch_cluster_name': ''}] * 2
    assert server.returncode == 0, server.stderr
    assert normalize(without_id(launched)) == normalize(without_id(in_process))
    # One record per call; the stdio server's is the in-process one.
    assert len(in_process_log) == 2 and len(launched_log) == 1
    assert normalize(launched_log) == normalize(in_process_log[1:])
    golden(
        f'{AREA}/empty_result/{request.node.callspec.id}',
        normalize(
            {
                'request': params,
                'in_process': {
                    'client_result': client_result,
                    'response': in_process,
                    'executor_log': in_process_log,
                },
                'stdio': {'response': launched, 'executor_log': launched_log},
            }
        ),
    )


# ---------------------------------------------------------------------------
# planted tool raising the SDK's MCP error over stdio
# ---------------------------------------------------------------------------

MCP_ERRORS = [
    pytest.param(
        (-32001, 'characterization: planted MCP error', {'reason': 'planted', 'n': [1, None]}),
        id='custom_code_with_data',
    ),
    pytest.param((-32602, 'characterization: invalid params'), id='invalid_params_no_data'),
    pytest.param(
        (-32603, 'characterization: non-ASCII — é 日本'),
        id='internal_error_non_ascii',
    ),
]


@pytest.mark.parametrize('error_args', MCP_ERRORS)
async def test_planted_mcp_error_over_stdio(
    harness, plant_tool, stdio_launcher, mcp_session, golden, normalize, request, error_args
):
    name = f'{PLANTED_PREFIX}McpError'
    plant_tool(name, raises=harness.SDK_MCP_ERROR(*error_args))
    params = {'name': name, 'arguments': {}}
    async with stdio_launcher('--log-format', 'json', env=NO_AUTH) as server:
        initialize = await server.initialize()
        response = await server.raw('tools/call', params)
        # The session goes on after the error.
        ping = await server.raw('ping')
    async with mcp_session(env=NO_AUTH) as session:
        in_process = await session.raw('tools/call', params)

    assert 'result' in initialize
    assert server.returncode == 0, server.stderr
    assert normalize(without_id(response)) == normalize(without_id(in_process))
    golden(
        f'{AREA}/mcp_error/{request.node.callspec.id}',
        normalize(
            {
                'request': params,
                'response': response,
                'next_request': {'method': 'ping', 'response': ping},
                'executor_log': stderr_tool_records(server.stderr),
                'warnings': [
                    {'logger': e['logger'], 'level': e['level'], 'message': e['message']}
                    for e in json_log_lines(server.stderr)
                    if e.get('level') in ('WARNING', 'ERROR', 'CRITICAL') and 'event_type' not in e
                ],
                'notifications': server.notifications,
                'traceback_on_stderr': has_traceback(server.stderr),
                'returncode': server.returncode,
            }
        ),
    )
