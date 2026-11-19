# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: the ``tools/list`` result for every configuration and filter mode.

Each case builds the server through ``mcp_session`` (or ``memory_enabled_session`` for the
memory tools, which are decided at import time), lists its tools through the MCP protocol
and pins the whole result: names, descriptions, input JSON schemas and order, with the
tools generated from ``pinned_openapi`` included. One golden per case,
``list_tools/<area>/<case>``, holding:

- ``inputs``: mode, environment on top of the harness's, config file contents (or that it
  is missing), CLI overrides, the version the fake OpenSearch reports, the OpenAPI fetch;
- ``tool_names``: the listed names in order, for a readable diff;
- ``result``: the full ``tools/list`` result (``None`` when the server fails to start);
- ``error``: the startup error, type and message (``None`` when it starts);
- ``list_error``: the raw JSON-RPC answer when the server starts but ``tools/list`` fails;
- ``opensearch_requests``: method, path and auth kind of what the server sent the fake
  OpenSearch (the version probe ``GET /`` in single mode).

Unless a case says otherwise, the server runs with ``OPENSEARCH_NO_AUTH=true`` so the version
probe reaches the fake OpenSearch, which reports ``3.1.0``.

Where the code reads keys (``tool_filter.py``, ``config.py``, ``server_instructions.py``):

- env filters: ``OPENSEARCH_ENABLED_TOOLS``, ``OPENSEARCH_DISABLED_TOOLS``,
  ``OPENSEARCH_TOOL_CATEGORIES``, ``OPENSEARCH_ENABLED_CATEGORIES``,
  ``OPENSEARCH_DISABLED_CATEGORIES``, ``OPENSEARCH_ENABLED_TOOLS_REGEX``,
  ``OPENSEARCH_DISABLED_TOOLS_REGEX``, ``OPENSEARCH_SETTINGS_ALLOW_WRITE``,
  ``OPENSEARCH_SETTINGS_ALLOW_WRITE_CATEGORIES``; all ignored once a config file is given;
- config file: ``tool_category``, ``tool_filters.{enabled_tools, disabled_tools,
  enabled_categories, disabled_categories, enabled_tools_regex, disabled_tools_regex}``,
  ``tool_filters.settings.{allow_write, allow_write_categories}``,
  ``tools.<Key>.{display_name, description, args, max_size_limit}``,
  ``agentic_memory.memory_container_id``;
- CLI: ``tool.<Key>.{display_name, description, args, max_size_limit}...`` (``cli_overrides``),
  used only when the config file has no ``tools`` section;
- schema: ``OPENSEARCH_DYNAMIC_CONNECTION``, ``OPENSEARCH_URL``, ``OPENSEARCH_HEADER_AUTH``,
  ``OPENSEARCH_MEMORY_CONTAINER_ID``; mode ``multi``; the cluster version (``min_version``).
"""

import pytest
import yaml
from dataclasses import dataclass, field
from mcp.shared.exceptions import MCPError
from typing import Any


NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}

# Highest min_version any tool declares (search relevance's Search*Tool): every tool listed.
LATEST_VERSION = '3.5.0'

# Matches every tool name, to list everything a version or mode allows.
ALL_TOOLS = {'OPENSEARCH_ENABLED_TOOLS_REGEX': '.*'}

# Two clusters modelled on example_config.yml. Nothing connects to them while listing; the
# hosts never resolve all the same.
TWO_CLUSTERS = {
    'local-cluster': {
        'opensearch_url': 'http://local-cluster.invalid:9200',
        'opensearch_username': 'admin',
        'opensearch_password': 'your_password_here',
    },
    'no-auth-cluster': {
        'opensearch_url': 'http://no-auth-cluster.invalid:9200',
        'opensearch_no_auth': True,
    },
}

SERVERLESS_CLUSTER = {
    'serverless-cluster': {
        'opensearch_url': 'https://collection-id.us-east-1.aoss.invalid',
        'aws_region': 'us-east-1',
        'is_serverless': True,
    },
}

# Static AWS credentials for the serverless case (SigV4 signs locally; no AWS call).
AWS_STATIC_CREDENTIALS = {
    'AWS_ACCESS_KEY_ID': 'AKIDCHARACTERIZATION',
    'AWS_SECRET_ACCESS_KEY': 'characterization-secret',
    'AWS_REGION': 'us-east-1',
}

# A tools section for config-file customization (and the same through the CLI below).
CUSTOM_TOOLS = {
    'ListIndexTool': {
        'display_name': 'Index_Manager',
        'description': 'List and manage OpenSearch indices',
        'args': {'index': "Custom description for the 'index' argument in ListIndexTool."},
    },
    'GetShardsTool': {'description': 'Retrieve detailed information about OpenSearch shards'},
    'SearchIndexTool': {'display_name': 'Super-Searcher'},
    # A tool generated from the OpenAPI spec.
    'MsearchTool': {
        'description': 'Run several searches in one request',
        'args': {'index': 'Default index for the searches.'},
    },
}

CLI_CUSTOM_TOOLS = {
    'tool.ListIndexTool.display_name': 'Index_Manager',
    'tool.ListIndexTool.description': 'List and manage OpenSearch indices',
    'tool.ListIndexTool.args.index.description': 'Custom description for the index argument.',
    'tool.GetShardsTool.description': 'Retrieve detailed information about OpenSearch shards',
    'tool.SearchIndexTool.display_name': 'Super-Searcher',
    'tool.MsearchTool.args.index.description': 'Default index for the searches.',
}


@dataclass
class Case:
    """One server configuration whose ``tools/list`` result is pinned."""

    mode: str = 'single'
    env: dict = field(default_factory=lambda: dict(NO_AUTH))
    # Config file contents (dict dumped as YAML), or None for no --config.
    config: Any = None
    # --config names a file that does not exist.
    missing_config: bool = False
    cli_overrides: dict | None = None
    # Version the fake OpenSearch reports for GET /; None makes GET / fail.
    version: str | None = '3.1.0'
    # The GitHub OpenAPI spec fetch fails (no generated tools).
    openapi_unavailable: bool = False
    # Run through memory_enabled_session (MEMORY_TOOLS_ENABLED=true at import, subprocess).
    memory: bool = False
    # List twice in the same session; pin the second and whether it equals the first.
    twice: bool = False


def _env(**extra) -> dict:
    return {**NO_AUTH, **extra}


CASES = {
    # --- single mode, defaults: only core_tools enabled ----------------------------------
    'single/defaults': Case(),
    'single/defaults_version_1_3_0': Case(version='1.3.0'),
    'single/defaults_version_2_19_0': Case(version='2.19.0'),
    # GET / fails: no version, so no version filtering.
    'single/defaults_version_unknown': Case(version=None),
    # The GitHub spec fetch fails: the generated tools (ClusterHealthTool, CountTool,
    # ExplainTool, MsearchTool) are missing.
    'single/defaults_openapi_unavailable': Case(openapi_unavailable=True),
    # The harness's own environment (OPENSEARCH_URL only): the AWS credentials path fails
    # (no region), the probe never reaches OpenSearch and the version is unknown.
    'single/defaults_aws_credentials_unavailable': Case(env={}),
    # --- version filtering, every tool enabled --------------------------------------------
    'versions/all_tools_version_1_3_0': Case(env=_env(**ALL_TOOLS), version='1.3.0'),
    'versions/all_tools_version_2_19_0': Case(env=_env(**ALL_TOOLS), version='2.19.0'),
    'versions/all_tools_version_3_1_0': Case(env=_env(**ALL_TOOLS)),
    'versions/all_tools_version_3_5_0': Case(env=_env(**ALL_TOOLS), version=LATEST_VERSION),
    'versions/all_tools_version_unknown': Case(env=_env(**ALL_TOOLS), version=None),
    # --- multi mode --------------------------------------------------------------------------
    # Connection fields stripped, ListClustersTool present, no tool filtering, no version.
    'multi/two_clusters': Case(mode='multi', config={'clusters': TWO_CLUSTERS}),
    # No --config: nothing loaded, the same tools.
    'multi/no_config_file': Case(mode='multi'),
    # characterization: current behaviour: multi mode reads no env filter, and
    # OPENSEARCH_SETTINGS_ALLOW_WRITE=false does not remove write tools: every tool is listed.
    'multi/env_filters_allow_write_false': Case(
        mode='multi',
        config={'clusters': TWO_CLUSTERS},
        env=_env(
            OPENSEARCH_DISABLED_TOOLS='ListIndexTool',
            OPENSEARCH_ENABLED_CATEGORIES='search_relevance',
            OPENSEARCH_DISABLED_TOOLS_REGEX='^Get.*',
            OPENSEARCH_SETTINGS_ALLOW_WRITE='false',
        ),
    ),
    # Custom display names and descriptions apply in multi mode too.
    'multi/custom_tools': Case(
        mode='multi', config={'clusters': TWO_CLUSTERS, 'tools': CUSTOM_TOOLS}
    ),
    # Dynamic connection mode is single-mode only: fields stay stripped although forced on.
    'multi/dynamic_connection_true': Case(
        mode='multi',
        config={'clusters': TWO_CLUSTERS},
        env=_env(OPENSEARCH_URL=None, OPENSEARCH_DYNAMIC_CONNECTION='true'),
    ),
    # --- dynamic connection mode ---------------------------------------------------------
    # No OPENSEARCH_URL: dynamic mode auto-detects as on; override fields kept and
    # opensearch_url required. No URL, so the version probe fails: no version filtering.
    'dynamic/auto_no_url': Case(env=_env(OPENSEARCH_URL=None)),
    # Forced on with a URL configured: fields kept, opensearch_url not required.
    'dynamic/forced_true_with_url': Case(env=_env(OPENSEARCH_DYNAMIC_CONNECTION='true')),
    # Forced on with "1" and no URL: fields kept, opensearch_url required.
    'dynamic/forced_1_no_url': Case(
        env=_env(OPENSEARCH_URL=None, OPENSEARCH_DYNAMIC_CONNECTION='1')
    ),
    # Forced off with no URL: fields stripped.
    'dynamic/forced_false_no_url': Case(
        env=_env(OPENSEARCH_URL=None, OPENSEARCH_DYNAMIC_CONNECTION='false')
    ),
    # An unrecognised value auto-detects: a URL is set, so off.
    'dynamic/unrecognised_value_with_url': Case(env=_env(OPENSEARCH_DYNAMIC_CONNECTION='yes')),
    # Header auth and no URL: dynamic on, fields kept, opensearch_url not required (the URL
    # may come from headers).
    'dynamic/header_auth_no_url': Case(
        env=_env(OPENSEARCH_URL=None, OPENSEARCH_HEADER_AUTH='true')
    ),
    # Header auth with a URL: dynamic off, fields stripped.
    'dynamic/header_auth_with_url': Case(env=_env(OPENSEARCH_HEADER_AUTH='true')),
    # Header auth, dynamic forced on, URL set: fields kept, opensearch_url not required.
    'dynamic/header_auth_forced_true': Case(
        env=_env(OPENSEARCH_HEADER_AUTH='true', OPENSEARCH_DYNAMIC_CONNECTION='true')
    ),
    # --- quirks -------------------------------------------------------------------------------
    # characterization: current behaviour: a config file with tool_filters but no settings
    # applies the write filter (allow_write is None, read as false): write-only tools of an
    # enabled category disappear.
    'quirks/config_tool_filters_without_settings': Case(
        config={'tool_filters': {'enabled_categories': ['search_relevance']}}
    ),
    # The same with settings.allow_write: true: write tools kept.
    'quirks/config_tool_filters_allow_write_true': Case(
        config={
            'tool_filters': {
                'enabled_categories': ['search_relevance'],
                'settings': {'allow_write': True},
            }
        }
    ),
    # characterization: current behaviour: --config naming a missing file ignores the env
    # filters (and still applies the write filter); single mode starts anyway.
    'quirks/config_file_missing': Case(
        missing_config=True,
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='search_relevance',
            OPENSEARCH_DISABLED_TOOLS='ListIndexTool',
        ),
    ),
    # A second tools/list in the same session lists the same (the schemas get_tools edits in
    # place were edited once, at startup).
    'quirks/second_list_default': Case(twice=True),
    'quirks/second_list_dynamic': Case(env=_env(OPENSEARCH_URL=None), twice=True),
    'quirks/second_list_multi': Case(mode='multi', config={'clusters': TWO_CLUSTERS}, twice=True),
    # characterization: current behaviour: USER_GUIDE.md documents the YAML key
    # `enable_tools_regex`, but tool_filter.py reads `enabled_tools_regex`: the documented key
    # is ignored.
    'quirks/config_enable_tools_regex_user_guide_key': Case(
        config={'tool_filters': {'enable_tools_regex': ['^GetSegments.*']}}
    ),
    # Invalid JSON in OPENSEARCH_TOOL_CATEGORIES: a warning, the category is not defined.
    'quirks/env_tool_categories_invalid_json': Case(
        env=_env(
            OPENSEARCH_TOOL_CATEGORIES='{"my_tools": [GetSegmentsTool]',
            OPENSEARCH_ENABLED_CATEGORIES='my_tools',
        )
    ),
    # Enabled and disabled at once: disabled wins.
    'quirks/env_enabled_and_disabled_same_tool': Case(
        env=_env(
            OPENSEARCH_ENABLED_TOOLS='GetSegmentsTool,CatNodesTool',
            OPENSEARCH_DISABLED_TOOLS='GetSegmentsTool',
        )
    ),
    # --- env filters, one each ------------------------------------------------------------
    # Names are case-insensitive and trimmed; unknown names are ignored.
    'env_filters/enabled_tools': Case(
        env=_env(OPENSEARCH_ENABLED_TOOLS='GetSegmentsTool, catnodestool,NoSuchTool')
    ),
    'env_filters/disabled_tools': Case(
        env=_env(OPENSEARCH_DISABLED_TOOLS='ListIndexTool, msearchtool,NoSuchTool')
    ),
    # Unknown categories are ignored with a warning.
    'env_filters/enabled_categories': Case(
        env=_env(OPENSEARCH_ENABLED_CATEGORIES='observability, no_such_category')
    ),
    # Disabling core_tools leaves only the other enabled categories.
    'env_filters/disabled_categories': Case(
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='observability,skills',
            OPENSEARCH_DISABLED_CATEGORIES='core_tools',
        )
    ),
    # A user-defined category, enabled.
    'env_filters/tool_categories': Case(
        env=_env(
            OPENSEARCH_TOOL_CATEGORIES='{"my_tools": ["GetSegmentsTool", "catnodestool"]}',
            OPENSEARCH_ENABLED_CATEGORIES='my_tools',
        )
    ),
    # re.match, case-insensitive, against display names.
    'env_filters/enabled_tools_regex': Case(
        env=_env(OPENSEARCH_ENABLED_TOOLS_REGEX='^get.*,^Cat')
    ),
    'env_filters/disabled_tools_regex': Case(
        env=_env(OPENSEARCH_DISABLED_TOOLS_REGEX='.*Index.*,^count')
    ),
    # Write-only tools of the enabled search_relevance category disappear.
    'env_filters/allow_write_false': Case(
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='search_relevance',
            OPENSEARCH_SETTINGS_ALLOW_WRITE='false',
        )
    ),
    # search_relevance's write tools exempt; agentic_memory, not exempt, keeps only its GET
    # tools (GetAgenticMemoryTool, SearchAgenticMemoryTool); skills tools bypass the filter.
    'env_filters/allow_write_categories': Case(
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='search_relevance,agentic_memory,skills',
            OPENSEARCH_SETTINGS_ALLOW_WRITE='false',
            OPENSEARCH_SETTINGS_ALLOW_WRITE_CATEGORIES='search_relevance',
        ),
        version=LATEST_VERSION,
    ),
    # --- the same filters from a config file ----------------------------------------------
    # characterization: current behaviour: every config-file case without
    # tool_filters.settings applies the write filter (see quirks/).
    'config_filters/enabled_tools': Case(
        config={'tool_filters': {'enabled_tools': ['GetSegmentsTool', 'catnodestool', 'NoSuch']}}
    ),
    'config_filters/disabled_tools': Case(
        config={'tool_filters': {'disabled_tools': ['ListIndexTool', 'msearchtool', 'NoSuch']}}
    ),
    'config_filters/enabled_categories': Case(
        config={'tool_filters': {'enabled_categories': ['observability', 'no_such_category']}}
    ),
    'config_filters/disabled_categories': Case(
        config={
            'tool_filters': {
                'enabled_categories': ['observability', 'skills'],
                'disabled_categories': ['core_tools'],
            }
        }
    ),
    'config_filters/tool_category': Case(
        config={
            'tool_category': {'my_tools': ['GetSegmentsTool', 'catnodestool']},
            'tool_filters': {'enabled_categories': ['my_tools']},
        }
    ),
    'config_filters/enabled_tools_regex': Case(
        config={'tool_filters': {'enabled_tools_regex': ['^get.*', '^Cat']}}
    ),
    'config_filters/disabled_tools_regex': Case(
        config={'tool_filters': {'disabled_tools_regex': ['.*Index.*', '^count']}}
    ),
    'config_filters/allow_write_false': Case(
        config={
            'tool_filters': {
                'enabled_categories': ['search_relevance'],
                'settings': {'allow_write': False},
            }
        }
    ),
    # As env_filters/allow_write_categories: agentic_memory keeps only its GET tools.
    'config_filters/allow_write_categories': Case(
        config={
            'tool_filters': {
                'enabled_categories': ['search_relevance', 'agentic_memory', 'skills'],
                'settings': {'allow_write': False, 'allow_write_categories': ['search_relevance']},
            }
        },
        version=LATEST_VERSION,
    ),
    # Config file and env vars together: the config file wins, the env filters are ignored.
    'config_filters/config_and_env': Case(
        config={
            'tool_filters': {
                'enabled_categories': ['observability'],
                'settings': {'allow_write': True},
            }
        },
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='search_relevance',
            OPENSEARCH_DISABLED_TOOLS='PPLQueryTool',
            OPENSEARCH_SETTINGS_ALLOW_WRITE='false',
        ),
    ),
    # --- optional tool groups --------------------------------------------------------------
    'optional_groups/skills': Case(env=_env(OPENSEARCH_ENABLED_CATEGORIES='skills')),
    # Agentic memory tools need OpenSearch 3.3.0: none at 3.1.0, all at 3.5.0.
    'optional_groups/agentic_memory_version_3_1_0': Case(
        env=_env(OPENSEARCH_ENABLED_CATEGORIES='agentic_memory')
    ),
    'optional_groups/agentic_memory': Case(
        env=_env(OPENSEARCH_ENABLED_CATEGORIES='agentic_memory'), version=LATEST_VERSION
    ),
    # memory_container_id gets a default and is no longer required.
    'optional_groups/agentic_memory_container_id_env': Case(
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='agentic_memory',
            OPENSEARCH_MEMORY_CONTAINER_ID='env-container',
        ),
        version=LATEST_VERSION,
    ),
    'optional_groups/agentic_memory_container_id_config': Case(
        config={
            'agentic_memory': {'memory_container_id': 'file-container'},
            'tool_filters': {
                'enabled_categories': ['agentic_memory'],
                'settings': {'allow_write': True},
            },
        },
        version=LATEST_VERSION,
    ),
    # The config file's container id wins over the env var.
    'optional_groups/agentic_memory_container_id_config_and_env': Case(
        config={
            'agentic_memory': {'memory_container_id': 'file-container'},
            'tool_filters': {
                'enabled_categories': ['agentic_memory'],
                'settings': {'allow_write': True},
            },
        },
        env=_env(OPENSEARCH_MEMORY_CONTAINER_ID='env-container'),
        version=LATEST_VERSION,
    ),
    'optional_groups/search_relevance': Case(
        env=_env(OPENSEARCH_ENABLED_CATEGORIES='search_relevance'), version=LATEST_VERSION
    ),
    'optional_groups/observability': Case(env=_env(OPENSEARCH_ENABLED_CATEGORIES='observability')),
    'optional_groups/all': Case(
        env=_env(
            OPENSEARCH_ENABLED_CATEGORIES='skills,agentic_memory,search_relevance,observability'
        ),
        version=LATEST_VERSION,
    ),
    # Memory tools (MEMORY_TOOLS_ENABLED=true at import): the memory category auto-enabled.
    'optional_groups/memory': Case(memory=True),
    # Memory tools bypass the write filter.
    'optional_groups/memory_allow_write_false': Case(
        memory=True, env=_env(OPENSEARCH_SETTINGS_ALLOW_WRITE='false')
    ),
    'optional_groups/memory_disabled_category': Case(
        memory=True, env=_env(OPENSEARCH_DISABLED_CATEGORIES='memory')
    ),
    # Multi mode excludes memory tools.
    'optional_groups/memory_multi': Case(
        memory=True, mode='multi', config={'clusters': TWO_CLUSTERS}
    ),
    # --- serverless -------------------------------------------------------------------------
    # tool_filter.py has no serverless detection: AWS_OPENSEARCH_SERVERLESS only changes the
    # signing service (aoss); the version probe and filtering are as for any cluster.
    'serverless/single_env': Case(
        env={**AWS_STATIC_CREDENTIALS, 'AWS_OPENSEARCH_SERVERLESS': 'true'}
    ),
    # Serverless with GET / unavailable: no version, no version filtering.
    'serverless/single_env_version_unknown': Case(
        env={**AWS_STATIC_CREDENTIALS, 'AWS_OPENSEARCH_SERVERLESS': 'true', **ALL_TOOLS},
        version=None,
    ),
    'serverless/multi_cluster': Case(mode='multi', config={'clusters': SERVERLESS_CLUSTER}),
    # --- custom tool display names, descriptions and args --------------------------------
    'custom_tools/config': Case(config={'tools': CUSTOM_TOOLS}),
    'custom_tools/cli': Case(cli_overrides=CLI_CUSTOM_TOOLS),
    # max_size_limit does not show in the listing.
    'custom_tools/config_max_size_limit': Case(
        config={'tools': {'SearchIndexTool': {'max_size_limit': 20}}}
    ),
    'custom_tools/cli_max_size_limit': Case(
        cli_overrides={'tool.SearchIndexTool.max_size_limit': '20'}
    ),
    'custom_tools/config_invalid_display_name': Case(
        config={'tools': {'ListIndexTool': {'display_name': 'Index Manager'}}}
    ),
    'custom_tools/cli_invalid_display_name': Case(
        cli_overrides={'tool.ListIndexTool.display_name': 'Index Manager'}
    ),
    'custom_tools/config_duplicate_display_name': Case(
        config={'tools': {'ListIndexTool': {'display_name': 'SearchIndexTool'}}}
    ),
    'custom_tools/cli_duplicate_display_name': Case(
        cli_overrides={
            'tool.ListIndexTool.display_name': 'Same_Name',
            'tool.GetShardsTool.display_name': 'Same_Name',
        }
    ),
    'custom_tools/config_unknown_arg': Case(
        config={'tools': {'ListIndexTool': {'args': {'no_such_arg': 'Description.'}}}}
    ),
    'custom_tools/cli_unknown_arg': Case(
        cli_overrides={'tool.ListIndexTool.args.no_such_arg.description': 'Description.'}
    ),
    # characterization: current behaviour: an unknown key under tools.<Key> in the YAML fails
    # startup, while the same key on the CLI (cli_unknown_key) is dropped silently. The error
    # message names only display_name, description and args as supported, yet max_size_limit
    # is accepted too (config_max_size_limit).
    'custom_tools/config_unknown_key': Case(
        config={'tools': {'ListIndexTool': {'http_methods': 'POST'}}}
    ),
    # characterization: current behaviour: on the CLI an unknown key is dropped silently.
    'custom_tools/cli_unknown_key': Case(
        cli_overrides={'tool.ListIndexTool.http_methods': 'POST'}
    ),
    'custom_tools/config_unknown_tool': Case(
        config={'tools': {'NoSuchTool': {'description': 'Nothing.'}}}
    ),
    'custom_tools/cli_unknown_tool': Case(
        cli_overrides={'tool.NoSuchTool.description': 'Nothing.'}
    ),
    # characterization: current behaviour: CLI values go through yaml.safe_load, so a numeric
    # description becomes an int; the server starts, and tools/list fails (Invalid request
    # parameters) building the Tool.
    'custom_tools/cli_numeric_description': Case(
        cli_overrides={'tool.GetShardsTool.description': '123'}
    ),
    # characterization: current behaviour: an arg override without `.description` sets the
    # arg to a string, which validation then calls `.get` on: startup fails, AttributeError.
    'custom_tools/cli_arg_without_description_key': Case(
        cli_overrides={'tool.ListIndexTool.args.index': 'Index description.'}
    ),
    # CLI overrides ignored when the YAML has a tools section.
    'custom_tools/cli_ignored_with_config_tools': Case(
        config={'tools': {'GetShardsTool': {'description': 'From the config file.'}}},
        cli_overrides={
            'tool.GetShardsTool.description': 'From the CLI.',
            'tool.ListIndexTool.display_name': 'Index_Manager',
        },
    ),
    # ...and used when the config file has no tools section.
    'custom_tools/cli_used_with_config_without_tools': Case(
        config={'tool_filters': {'settings': {'allow_write': True}}},
        cli_overrides={'tool.GetShardsTool.description': 'From the CLI.'},
    ),
    # Custom display names feed the filters: core_tools follows the new name, and an env
    # filter naming the display name removes the tool.
    'custom_tools/cli_display_name_filtered': Case(
        cli_overrides={
            'tool.ListIndexTool.display_name': 'Index_Manager',
            'tool.GetShardsTool.display_name': 'Shard_Info',
        },
        env=_env(OPENSEARCH_DISABLED_TOOLS='Shard_Info'),
    ),
    # characterization: current behaviour: an env filter naming the registry key of a renamed
    # tool is ignored (filters match display names only): GetShardsTool stays, as Shard_Info.
    'custom_tools/cli_display_name_filtered_by_key': Case(
        cli_overrides={
            'tool.ListIndexTool.display_name': 'Index_Manager',
            'tool.GetShardsTool.display_name': 'Shard_Info',
        },
        env=_env(OPENSEARCH_DISABLED_TOOLS='GetShardsTool'),
    ),
}


def _error(exc: BaseException) -> dict:
    return {'type': type(exc).__name__, 'message': str(exc)}


def _requests(records: list[dict]) -> list[dict]:
    return [{'method': r['method'], 'path': r['path'], 'auth': r['auth']} for r in records]


def _write_config(tmp_path, config: Any) -> str:
    path = tmp_path / 'config.yml'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return str(path)


async def _list(session, twice: bool) -> tuple[dict, bool | None]:
    first = await session.list_tools()
    if not twice:
        return first, None
    second = await session.list_tools()
    return second, second == first


@pytest.mark.parametrize('name', CASES)
async def test_list_tools(
    name,
    tmp_path,
    request,
    fake_opensearch,
    mcp_session,
    golden,
    normalize,
):
    case = CASES[name]
    fake_opensearch.version = case.version
    if case.openapi_unavailable:
        request.getfixturevalue('openapi_unavailable')
    config_file = ''
    if case.missing_config:
        config_file = str(tmp_path / 'missing.yml')
    elif case.config is not None:
        config_file = _write_config(tmp_path, case.config)

    result = error = list_error = same_as_first = None
    requests: list[dict] = []
    if case.memory:
        memory_enabled_session = request.getfixturevalue('memory_enabled_session')
        async with memory_enabled_session(
            mode=case.mode, config_file=config_file, env=case.env
        ) as session:
            result, same_as_first = await _list(session, case.twice)
            requests = session.opensearch_requests()
    else:
        try:
            async with mcp_session(
                mode=case.mode,
                config_file=config_file,
                cli_overrides=case.cli_overrides,
                env=case.env,
            ) as session:
                try:
                    result, same_as_first = await _list(session, case.twice)
                except MCPError:
                    # The server started but tools/list answers a JSON-RPC error: pin the
                    # raw answer.
                    list_error = await session.raw('tools/list')
        except Exception as exc:
            error = _error(exc)
        requests = fake_opensearch.requests

    data = {
        'inputs': {
            'mode': case.mode,
            'env': case.env,
            'config': case.config,
            'config_file_missing': case.missing_config,
            'cli_overrides': case.cli_overrides,
            'opensearch_version': case.version,
            'openapi_unavailable': case.openapi_unavailable,
            'memory_tools_enabled': case.memory,
        },
        'tool_names': None if result is None else [tool['name'] for tool in result['tools']],
        'result': result,
        'error': error,
        'list_error': list_error,
        'opensearch_requests': _requests(requests),
    }
    if case.twice:
        data['second_list_same_as_first'] = same_as_first
    golden(f'list_tools/{name}', normalize(data))
