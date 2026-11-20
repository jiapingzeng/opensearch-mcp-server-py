# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: two enabled tools listed under the same name.

``tools/config.py`` ``_validate_config`` refuses a display name that is another tool's
registry key or a display name already given, but it lets two through:

- a display name equal to the key of a tool that is itself configured (a configured key
  leaves the set of names it checks against) but not renamed;
- the empty display name, which every check skips (they only look at a truthy name).

Every registry tool's display name is its key, so either input leaves two tools with the
same display name, through the config file's ``tools`` section or the ``--tool.*`` CLI
overrides alike. Multi mode enables every tool by registry key, so both are listed under
that name, in registry order, each with its own description and schema; single mode
(``tool_filter.get_tools``) keys the enabled tools by display name, so one is listed.

Each case pins the ``tools/list`` names, the entries listed under the shared name, the
server's logs and any "already exists" warning from any logger.
"""

import logging
import pytest
from mcp_server_opensearch import parse_unknown_args_to_dict


SERVER_LOGGER_PREFIXES = ('mcp_server_opensearch', 'tools')

# Single mode reaches the fake OpenSearch for its version with no-auth.
SINGLE_MODE_ENV = {'OPENSEARCH_NO_AUTH': 'true'}

# ListIndexTool is listed as IndexMappingTool, the key of a tool that is configured (its
# description) but not renamed.
RENAME_ONTO_CONFIGURED_KEY = """\
tools:
  ListIndexTool:
    display_name: IndexMappingTool
    description: ListIndexTool, listed as IndexMappingTool
  IndexMappingTool:
    description: IndexMappingTool, configured but not renamed
"""

TWO_EMPTY_DISPLAY_NAMES = """\
tools:
  ListIndexTool:
    display_name: ''
  IndexMappingTool:
    display_name: ''
"""

CASES = {
    'multi_file_rename_onto_configured_key': {
        'mode': 'multi',
        'config': RENAME_ONTO_CONFIGURED_KEY,
        'shared': 'IndexMappingTool',
    },
    'multi_file_two_empty_display_names': {
        'mode': 'multi',
        'config': TWO_EMPTY_DISPLAY_NAMES,
        'shared': '',
    },
    'multi_cli_rename_onto_configured_key': {
        'mode': 'multi',
        'argv': [
            '--tool.ListIndexTool.display_name=IndexMappingTool',
            '--tool.IndexMappingTool.description=IndexMappingTool, configured but not renamed',
        ],
        'shared': 'IndexMappingTool',
    },
    'multi_cli_two_empty_display_names': {
        'mode': 'multi',
        'argv': [
            '--tool.ListIndexTool.display_name=',
            '--tool.IndexMappingTool.display_name=',
        ],
        'shared': '',
    },
    # characterization: current behaviour: single mode keys the enabled tools by display
    # name, so the same input lists the name once.
    'single_file_rename_onto_configured_key': {
        'mode': 'single',
        'config': RENAME_ONTO_CONFIGURED_KEY,
        'shared': 'IndexMappingTool',
    },
    'single_cli_two_empty_display_names': {
        'mode': 'single',
        'argv': [
            '--tool.ListIndexTool.display_name=',
            '--tool.IndexMappingTool.display_name=',
        ],
        'shared': '',
    },
}


def _records(caplog) -> list:
    return [
        {'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        for record in caplog.records
        if record.name != 'asyncio'
    ]


def _server_records(caplog) -> list:
    """Records of the server's loggers: root, ``mcp_server_opensearch``, ``tools``."""
    return [
        record
        for record in _records(caplog)
        if record['logger'] == 'root' or record['logger'].startswith(SERVER_LOGGER_PREFIXES)
    ]


@pytest.mark.parametrize('case', CASES)
async def test_duplicate_listed_names(case, tmp_path, mcp_session, caplog, golden, normalize):
    spec = CASES[case]
    mode = spec['mode']
    session_args = {'mode': mode, 'env': SINGLE_MODE_ENV if mode == 'single' else None}
    data = {'mode': mode}
    if 'config' in spec:
        path = tmp_path / 'config.yml'
        path.write_text(spec['config'], encoding='utf-8')
        session_args['config_file'] = path
        data['config'] = spec['config']
    else:
        # As main() does: unknown --tool.* flags become a dict, which serve() hands to
        # apply_custom_tool_config.
        overrides = parse_unknown_args_to_dict(spec['argv'])
        session_args['cli_overrides'] = overrides
        data.update(argv=spec['argv'], overrides=overrides)
    caplog.set_level(logging.INFO)

    async with mcp_session(**session_args) as session:
        tools = (await session.list_tools())['tools']

    data['tool_names'] = [tool['name'] for tool in tools]
    data['shared_name'] = spec['shared']
    data['listed_under_shared_name'] = [tool for tool in tools if tool['name'] == spec['shared']]
    data['logs'] = _server_records(caplog)
    data['already_exists_warnings'] = [
        record for record in _records(caplog) if 'already exists' in record['message']
    ]
    golden(f'gaps_duplicate_names/{case}', normalize(data))
