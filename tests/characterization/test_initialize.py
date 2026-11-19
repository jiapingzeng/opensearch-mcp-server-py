# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: the protocol surface a client sees at connect time.

Pinned through the MCP protocol (``mcp_session`` and its ``raw()``), for each server mode:

- ``initialize/<case>/raw``: the raw JSON-RPC answer to ``initialize`` and to
  ``server/discover`` at every protocol version the installed SDK supports (server name and
  version, negotiated protocol version, capabilities, instructions; at 2026-07-28 the
  ``_meta``/``resultType``/``cacheScope``/``ttlMs`` fields the SDK adds);
- ``initialize/<case>/client``: what the SDK ``Client`` reports after connecting, in
  ``legacy`` mode (initialize handshake) and ``auto`` mode (``server/discover``);
- ``initialize/<case>/instructions``: the server instructions text, split into lines so a
  diff shows the line that changed, or ``None``.

And, once: what a client gets for the methods the server registers no handler for
(``initialize/client_methods``), how the handshake negotiates versions it does not know
(``initialize/negotiation``), and the fields the SDK adds to a result
(``initialize/result_envelope``).
"""

import pytest
import yaml
from mcp.client import Client
from mcp_types.version import MODERN_PROTOCOL_VERSIONS, SUPPORTED_PROTOCOL_VERSIONS


# Every version the installed SDK supports. A new SDK version adding or dropping one changes
# the goldens below: that is a change in what a client can connect with.
PROTOCOL_VERSIONS = tuple(SUPPORTED_PROTOCOL_VERSIONS)

LATEST_MODERN_VERSION = '2026-07-28'
LATEST_HANDSHAKE_VERSION = '2025-11-25'

CLIENT_INFO = {'name': 'characterization-client', 'version': '0.0.0'}


def test_sdk_supports_the_versions_this_module_pins():
    # The goldens cover every supported version; these two at least must be among them.
    assert LATEST_MODERN_VERSION in PROTOCOL_VERSIONS
    assert LATEST_HANDSHAKE_VERSION in PROTOCOL_VERSIONS
    assert LATEST_MODERN_VERSION in MODERN_PROTOCOL_VERSIONS


def _dump(model) -> dict:
    return model.model_dump(mode='json', by_alias=True, exclude_none=True)


# --- config files (written to tmp_path) ---------------------------------------------------

# Two clusters modelled on example_config.yml (basic auth, and no auth). Nothing connects to
# them at connect time; the URLs point at hosts that never resolve all the same.
TWO_CLUSTERS = {
    'local-cluster': {
        'opensearch_url': 'http://local-cluster.invalid:9200',
        'opensearch_username': 'admin',
        'opensearch_password': 'your_password_here',
        'max_response_size': 5242880,
    },
    'no-auth-cluster': {
        'opensearch_url': 'http://no-auth-cluster.invalid:9200',
        'opensearch_no_auth': True,
    },
}

SKILLS_FILTER = {'tool_filters': {'enabled_categories': ['skills']}}


def _config(*, clusters: bool, skills: bool) -> dict:
    config = {'version': '1.0', 'description': 'Characterization config'}
    if clusters:
        config['clusters'] = TWO_CLUSTERS
    if skills:
        config.update(SKILLS_FILTER)
    return config


@pytest.fixture
def write_config(tmp_path):
    """Write a YAML config file under ``tmp_path`` and return its path."""

    def write(config: dict):
        path = tmp_path / 'config.yml'
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return path

    return write


# --- connect-time cases --------------------------------------------------------------------

# (case, mode, env, config or None). The harness sets OPENSEARCH_URL to the fake by default;
# ``None`` as an env value unsets a variable.
CASES = [
    # OPENSEARCH_URL is set, so dynamic connection mode auto-detects as off: no instructions.
    ('single_default', 'single', {}, None),
    # No OPENSEARCH_URL: dynamic connection mode auto-detects as on.
    ('single_dynamic_auto', 'single', {'OPENSEARCH_URL': None}, None),
    # OPENSEARCH_DYNAMIC_CONNECTION=true turns it on although a URL is configured.
    ('single_dynamic_enabled', 'single', {'OPENSEARCH_DYNAMIC_CONNECTION': 'true'}, None),
    # OPENSEARCH_DYNAMIC_CONNECTION=false turns it off although no URL is configured.
    (
        'single_dynamic_disabled',
        'single',
        {'OPENSEARCH_URL': None, 'OPENSEARCH_DYNAMIC_CONNECTION': 'false'},
        None,
    ),
    # Skills category enabled from the environment: the skills section only.
    ('single_skills_env', 'single', {'OPENSEARCH_ENABLED_CATEGORIES': 'skills'}, None),
    # Dynamic mode and skills: both sections, dynamic first.
    (
        'single_dynamic_skills_env',
        'single',
        {'OPENSEARCH_DYNAMIC_CONNECTION': 'true', 'OPENSEARCH_ENABLED_CATEGORIES': 'skills'},
        None,
    ),
    # Disabled wins over enabled.
    (
        'single_skills_env_enabled_and_disabled',
        'single',
        {'OPENSEARCH_ENABLED_CATEGORIES': 'skills', 'OPENSEARCH_DISABLED_CATEGORIES': 'skills'},
        None,
    ),
    # Skills category enabled from the config file's tool_filters.
    ('single_skills_config', 'single', {}, _config(clusters=False, skills=True)),
    # A config file replaces the environment as the source of categories, even when it has no
    # tool_filters: the env's skills category is ignored.
    (
        'single_config_ignores_skills_env',
        'single',
        {'OPENSEARCH_ENABLED_CATEGORIES': 'skills'},
        _config(clusters=False, skills=False),
    ),
    # Multi mode, two clusters: no dynamic section in multi mode, so instructions are None.
    ('multi_two_clusters', 'multi', {}, _config(clusters=True, skills=False)),
    # Dynamic connection mode is single-mode only, even when forced on and no URL is set.
    (
        'multi_dynamic_env',
        'multi',
        {'OPENSEARCH_URL': None, 'OPENSEARCH_DYNAMIC_CONNECTION': 'true'},
        _config(clusters=True, skills=False),
    ),
    # characterization: current behaviour. example_config.yml documents tool_filters as
    # "ONLY supported in Single Mode", yet the instructions read the skills category from
    # them in multi mode too and add the skills section.
    ('multi_skills_config', 'multi', {}, _config(clusters=True, skills=True)),
    # Multi mode always has a config file, so the env's skills category is ignored.
    (
        'multi_skills_env',
        'multi',
        {'OPENSEARCH_ENABLED_CATEGORIES': 'skills'},
        _config(clusters=True, skills=False),
    ),
]


# initialize, and server/discover, its 2026-07-28 replacement. Default params: at a handshake
# version raw('initialize') is the handshake itself, offering that version; at 2026-07-28 it
# is a bare request in the per-request envelope.
CONNECT_METHODS = (('initialize', None), ('server/discover', None))


async def _raw_by_version(session, requests) -> dict:
    """``{version: {method: raw answer}}``, each request on a fresh connection at that version."""
    answers = {}
    for version in PROTOCOL_VERSIONS:
        answers[version] = {}
        for method, params in requests:
            answers[version][method] = await session.raw(method, params, protocol_version=version)
    return answers


def _client_view(client: Client) -> dict:
    """What an SDK client knows about the server once connected."""
    return {
        'protocol_version': client.protocol_version,
        'server_info': _dump(client.server_info) if client.server_info is not None else None,
        'server_capabilities': _dump(client.server_capabilities),
        'instructions': client.instructions,
    }


@pytest.mark.parametrize(('case', 'mode', 'env', 'config'), CASES, ids=[case[0] for case in CASES])
async def test_connect(mcp_session, write_config, golden, normalize, case, mode, env, config):
    config_file = write_config(config) if config is not None else ''
    async with mcp_session(mode=mode, config_file=config_file, env=env) as session:
        raw = await _raw_by_version(session, CONNECT_METHODS)
        # characterization: current behaviour. The server is built without a version, so
        # serverInfo.version is "" at every protocol version.
        golden(f'initialize/{case}/raw', normalize(raw))

        # mcp_session's client is a legacy (initialize handshake) client; a second client on
        # the same server shows what an 'auto' client (server/discover) sees.
        legacy = _client_view(session.client)
        async with Client(session.server, mode='auto', cache=None) as auto_client:
            auto = _client_view(auto_client)
        golden(f'initialize/{case}/client', normalize({'legacy': legacy, 'auto': auto}))

        instructions = legacy['instructions']
        # Every path a client can learn the instructions from agrees.
        assert auto['instructions'] == instructions
        for answers in raw.values():
            for answer in answers.values():
                if 'result' in answer:
                    assert answer['result'].get('instructions') == instructions
        golden(
            f'initialize/{case}/instructions',
            normalize(
                {'instructions': None}
                if instructions is None
                else {'instructions_lines': instructions.split('\n')}
            ),
        )


# --- methods without a handler, negotiation, result envelope -------------------------------

# Requests a client may send at connect time. The server registers only tools/list and
# tools/call; ping is the SDK's default handler.
CLIENT_METHODS = (
    ('ping', None),
    ('prompts/list', None),
    ('resources/list', None),
    ('resources/templates/list', None),
    (
        'completion/complete',
        {
            'ref': {'type': 'ref/prompt', 'name': 'any-prompt'},
            'argument': {'name': 'any-argument', 'value': 'a'},
        },
    ),
    ('logging/setLevel', {'level': 'debug'}),
)


async def test_client_methods(mcp_session, golden, normalize):
    async with mcp_session() as session:
        answers = await _raw_by_version(session, CLIENT_METHODS)
    # characterization: current behaviour. ping answers {} at the handshake versions and
    # -32601 "Method not found" at 2026-07-28 (the SDK removed it there); every other method
    # answers -32601 at every version.
    golden('initialize/client_methods', normalize(answers))


def _initialize_params(protocol_version: str) -> dict:
    return {'protocolVersion': protocol_version, 'capabilities': {}, 'clientInfo': CLIENT_INFO}


async def test_negotiation(mcp_session, golden, normalize):
    async with mcp_session() as session:
        answers = {
            # Handshake offers of versions the handshake does not speak: the server answers
            # with its latest handshake version.
            'handshake_offers_2026_07_28': await session.raw(
                'initialize', _initialize_params(LATEST_MODERN_VERSION)
            ),
            'handshake_offers_unknown_future_version': await session.raw(
                'initialize', _initialize_params('2099-01-01')
            ),
            'handshake_offers_unknown_old_version': await session.raw(
                'initialize', _initialize_params('2024-01-01')
            ),
            # characterization: current behaviour. Full initialize params inside the
            # 2026-07-28 per-request envelope run a handshake and negotiate 2025-11-25,
            # while a bare initialize at 2026-07-28 (initialize/<case>/raw) is rejected
            # with -32602.
            'modern_envelope_with_initialize_params': await session.raw(
                'initialize',
                _initialize_params(LATEST_MODERN_VERSION),
                protocol_version=LATEST_MODERN_VERSION,
            ),
        }
    golden('initialize/negotiation', normalize(answers))


async def test_result_envelope(mcp_session, golden, normalize):
    """The fields the SDK puts around a result, at each version (tools/list as the example)."""
    async with mcp_session() as session:
        envelopes = {}
        for version in PROTOCOL_VERSIONS:
            answer = await session.raw('tools/list', protocol_version=version)
            # The tool list itself is pinned by the tools/list characterizations; keep only
            # what surrounds it.
            tools = answer['result'].pop('tools')
            assert isinstance(tools, list) and tools
            envelopes[version] = answer
    # At 2026-07-28 the SDK adds resultType, cacheScope, ttlMs and _meta (with serverInfo,
    # whose version is ""); the handshake versions add nothing.
    golden('initialize/result_envelope', normalize(envelopes))
