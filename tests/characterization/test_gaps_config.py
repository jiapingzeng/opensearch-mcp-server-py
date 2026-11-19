# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: config file and CLI override edge cases.

Pins the edges the coverage checkpoint found unpinned in ``tools/config.py`` (a YAML
``args`` that is no mapping, an argument description that is no string, ``args: {}``, CLI
override values that are empty or not valid YAML, args on a tool whose schema has no
properties, a non-string CLI argument description), ``tools/tool_filter.py``'s error path
(an invalid regex in the filters) and ``clusters_information.load_clusters_from_yaml``'s
read errors (a file that is not UTF-8, a directory). Every file is a temporary one.

A config or CLI case is pinned twice: ``apply_custom_tool_config`` called directly (what it
changes in the registry it returns, in the global ``TOOL_REGISTRY`` and on the args
models), and the server built from the same input (what ``tools/list`` shows, or how
startup fails). The module state is restored between the two.
"""

import copy
import json
import logging
import pytest
import sys
from mcp_server_opensearch import parse_unknown_args_to_dict
from mcp_server_opensearch.clusters_information import cluster_registry, load_clusters_from_yaml
from tools.config import apply_custom_tool_config, parse_cli_to_nested_config
from tools.tools import TOOL_REGISTRY


# Registry entry values that are Python objects, not data.
OBJECT_ENTRY_FIELDS = ('function', 'args_model')

SERVER_LOGGER_PREFIXES = ('mcp_server_opensearch', 'tools')

# Single mode reaches the fake OpenSearch for its version with no-auth.
SINGLE_MODE_ENV = {'OPENSEARCH_NO_AUTH': 'true'}


def _env(mode: str) -> dict | None:
    return SINGLE_MODE_ENV if mode == 'single' else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _error(exc: BaseException) -> dict:
    return {'type': type(exc).__name__, 'message': str(exc)}


def _records(caplog) -> list:
    return [
        {'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        for record in caplog.records
        if record.name != 'asyncio'
    ]


def _server_records(caplog) -> list:
    """Records of the server's loggers: root, ``mcp_server_opensearch``, ``tools``.

    Left out: the OpenSearch client's connection setup (``opensearch.*``) and third-party
    libraries.
    """
    return [
        record
        for record in _records(caplog)
        if record['logger'] == 'root' or record['logger'].startswith(SERVER_LOGGER_PREFIXES)
    ]


def _diff(before, after, prefix: str = '') -> dict:
    """Dotted paths whose value differs between ``before`` and ``after``, with the new value."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = {}
        for key in sorted(set(before) | set(after), key=str):
            path = f'{prefix}.{key}' if prefix else str(key)
            if key not in after:
                changes[path] = '<REMOVED>'
            elif key not in before:
                changes[path] = after[key]
            else:
                changes.update(_diff(before[key], after[key], path))
        return changes
    return {} if before == after else {prefix: after}


def _registry_view(registry: dict) -> dict:
    return {
        name: {
            key: copy.deepcopy(value)
            for key, value in entry.items()
            if key not in OBJECT_ENTRY_FIELDS
        }
        for name, entry in registry.items()
    }


def _model_descriptions(registry: dict) -> dict:
    return {
        name: {field: info.description for field, info in entry['args_model'].model_fields.items()}
        for name, entry in registry.items()
        if entry.get('args_model') is not None
    }


def _apply_directly(isolated_state, config_file_path: str, cli_tool_overrides: dict) -> dict:
    """``apply_custom_tool_config`` on the global registry: what it changed, or its error.

    The module state is put back afterwards, so the server built next starts from the same
    registry.
    """
    state = isolated_state.snapshot()
    before = _registry_view(TOOL_REGISTRY)
    descriptions = _model_descriptions(TOOL_REGISTRY)
    try:
        resolved = apply_custom_tool_config(TOOL_REGISTRY, config_file_path, cli_tool_overrides)
    except Exception as exc:
        result = {'error': _error(exc)}
    else:
        result = {
            'resolved_changes': _diff(before, _registry_view(resolved)),
            # apply_custom_tool_config also writes the resolved entries back into the
            # global TOOL_REGISTRY (pinned in test_cli_config.py).
            'global_registry_changes': _diff(before, _registry_view(TOOL_REGISTRY)),
            # An argument description is also written on the args model class.
            'args_model_description_changes': _diff(
                descriptions, _model_descriptions(TOOL_REGISTRY)
            ),
        }
    state.restore()
    return result


async def _serve(mcp_session, show: tuple, **session_args) -> dict:
    """Build the server; its ``tools/list`` names and the entries named in ``show``."""
    result = {'error': None, 'tool_names': None, 'tools': None}
    try:
        async with mcp_session(**session_args) as session:
            tools = (await session.list_tools())['tools']
    except Exception as exc:
        result['error'] = _error(exc)
        return result
    result['tool_names'] = [tool['name'] for tool in tools]
    result['tools'] = [tool for tool in tools if tool['name'] in show]
    return result


# ---------------------------------------------------------------------------
# config.py: the tools section of the YAML config file
# ---------------------------------------------------------------------------

FILE_TOOLS_CASES = {
    # config.py 33-37: `args` that is not a mapping is dropped with a warning; the
    # tool's other fields still apply.
    'args_string': {
        'config': 'tools:\n  ListIndexTool:\n    description: kept\n    args: just a string\n',
        'show': ('ListIndexTool',),
    },
    'args_list': {
        'config': 'tools:\n  ListIndexTool:\n    description: kept\n    args:\n      - index\n',
        'show': ('ListIndexTool',),
    },
    # characterization: current behaviour: an `args:` key with no value (YAML null) is
    # warned about as "not a mapping" too.
    'args_null': {
        'config': 'tools:\n  ListIndexTool:\n    description: kept\n    args:\n',
        'show': ('ListIndexTool',),
    },
    # config.py 40->43, 43: an argument description that is not a string fails startup.
    'arg_description_int': {
        'config': 'tools:\n  ListIndexTool:\n    args:\n      index: 123\n',
        'show': ('ListIndexTool',),
    },
    # characterization: current behaviour: the YAML file takes `arg: <description>`; the
    # CLI's `args.<arg>.description` shape (a mapping) is refused as "not a string".
    'arg_description_mapping': {
        'config': 'tools:\n  ListIndexTool:\n    args:\n      index:\n        description: nested\n',
        'show': ('ListIndexTool',),
    },
    'arg_description_null': {
        'config': 'tools:\n  ListIndexTool:\n    args:\n      index:\n',
        'show': ('ListIndexTool',),
    },
    # config.py 56->58: `args: {}` adds no args key; the description still applies.
    'args_empty_mapping': {
        'config': 'tools:\n  ListIndexTool:\n    description: kept\n    args: {}\n',
        'show': ('ListIndexTool',),
    },
    # config.py 172->175: ListClustersTool is the one registry tool whose input schema has
    # no properties, so any argument named for it does not exist.
    # It is a multi-mode tool, so these cases run in multi mode, where it is listed.
    'no_properties_tool_arg': {
        'config': 'tools:\n  ListClustersTool:\n    args:\n      foo: x\n',
        'show': ('ListClustersTool',),
        'mode': 'multi',
    },
    'no_properties_tool_args_empty': {
        'config': 'tools:\n  ListClustersTool:\n    description: kept\n    args: {}\n',
        'show': ('ListClustersTool',),
        'mode': 'multi',
    },
}


@pytest.mark.parametrize('case', FILE_TOOLS_CASES)
async def test_file_tools_section(
    case, tmp_path, isolated_state, mcp_session, caplog, golden, normalize
):
    spec = FILE_TOOLS_CASES[case]
    mode = spec.get('mode', 'single')
    path = tmp_path / 'config.yml'
    path.write_text(spec['config'], encoding='utf-8')
    caplog.set_level(logging.INFO)
    applied = _apply_directly(isolated_state, str(path), {})
    applied['logs'] = _server_records(caplog)
    caplog.clear()
    server = await _serve(mcp_session, spec['show'], mode=mode, config_file=path, env=_env(mode))
    server['logs'] = _server_records(caplog)
    golden(
        f'gaps_config/file_tools/{case}',
        normalize({'config': spec['config'], 'mode': mode, 'applied': applied, 'server': server}),
    )


# ---------------------------------------------------------------------------
# config.py: --tool.* overrides on the command line
# ---------------------------------------------------------------------------

CLI_CASES = {
    # config.py 80->86: an empty or blank value is not given to the YAML parser: it stays
    # the string it is.
    'empty_description': {
        'argv': ['--tool.ListIndexTool.description='],
        'show': ('ListIndexTool',),
    },
    'blank_description': {
        'argv': ['--tool.ListIndexTool.description=   '],
        'show': ('ListIndexTool',),
    },
    # characterization: current behaviour: an empty display name skips the display-name
    # checks (they only look at a truthy name) and is applied: the tool is listed under
    # the name "" (core_tools matches it by that display name, so it stays enabled).
    'empty_display_name': {
        'argv': ['--tool.ListIndexTool.display_name='],
        'show': ('ListIndexTool', ''),
    },
    # config.py 83-84: a value that is not valid YAML is kept as the raw string.
    'invalid_yaml_unclosed_flow': {
        'argv': ['--tool.ListIndexTool.description=[unclosed'],
        'show': ('ListIndexTool',),
    },
    'invalid_yaml_reserved_indicator': {
        'argv': ['--tool.ListIndexTool.description=@mention'],
        'show': ('ListIndexTool',),
    },
    'invalid_yaml_mapping_in_mapping': {
        'argv': ['--tool.ListIndexTool.description=a: b: c'],
        'show': ('ListIndexTool',),
    },
    # config.py 186-187: "123" is coerced to the int 123, which the args check refuses.
    'arg_description_int': {
        'argv': ['--tool.ListIndexTool.args.index.description=123'],
        'show': ('ListIndexTool',),
    },
    # characterization: current behaviour: "null" is coerced to None, which the args check
    # lets through (it only refuses a non-None non-string), and the argument's schema
    # description becomes null.
    'arg_description_null': {
        'argv': ['--tool.ListIndexTool.args.index.description=null'],
        'show': ('ListIndexTool',),
    },
    # characterization: current behaviour: a value given for the argument itself, not its
    # description, fails startup with an AttributeError (a str has no .get).
    'arg_value_not_mapping': {
        'argv': ['--tool.ListIndexTool.args.index=plain'],
        'show': ('ListIndexTool',),
    },
    # config.py 172->175: ListClustersTool's schema has no properties.
    'no_properties_tool_arg': {
        'argv': ['--tool.ListClustersTool.args.foo.description=x'],
        'show': ('ListClustersTool',),
        'mode': 'multi',
    },
}


@pytest.mark.parametrize('case', CLI_CASES)
async def test_cli_tool_overrides(case, isolated_state, mcp_session, caplog, golden, normalize):
    spec = CLI_CASES[case]
    mode = spec.get('mode', 'single')
    caplog.set_level(logging.INFO)
    # As main() does: unknown --tool.* flags become a dict, which serve() hands to
    # apply_custom_tool_config.
    overrides = parse_unknown_args_to_dict(spec['argv'])
    nested = parse_cli_to_nested_config(overrides)
    applied = _apply_directly(isolated_state, '', overrides)
    applied['logs'] = _server_records(caplog)
    caplog.clear()
    server = await _serve(
        mcp_session, spec['show'], mode=mode, cli_overrides=overrides, env=_env(mode)
    )
    server['logs'] = _server_records(caplog)
    golden(
        f'gaps_config/cli_overrides/{case}',
        normalize(
            {
                'argv': spec['argv'],
                'mode': mode,
                'overrides': overrides,
                'nested_config': nested,
                'applied': applied,
                'server': server,
            }
        ),
    )


# ---------------------------------------------------------------------------
# tool_filter.py 411-412: an invalid regex in the filters
# ---------------------------------------------------------------------------

# Every filter is given, so the golden shows which ones ran before the regex error: the
# write filter (allow_write: false) runs first, then the regex patterns are compiled and
# the error abandons the rest. characterization: current behaviour: the enabled and
# disabled lists, and the default core_tools category, are then never applied, so every
# tool that passed the write filter is listed.
INVALID_REGEX_CONFIG = """\
tool_filters:
  enabled_tools:
    - ListIndexTool
  disabled_tools:
    - SearchIndexTool
  disabled_tools_regex:
    - '['
  settings:
    allow_write: false
"""

INVALID_REGEX_ENV = {
    'OPENSEARCH_ENABLED_TOOLS': 'ListIndexTool',
    'OPENSEARCH_DISABLED_TOOLS': 'SearchIndexTool',
    'OPENSEARCH_ENABLED_TOOLS_REGEX': '[',
    'OPENSEARCH_SETTINGS_ALLOW_WRITE': 'false',
}


async def test_invalid_regex_config_file(tmp_path, mcp_session, caplog, golden, normalize):
    path = tmp_path / 'config.yml'
    path.write_text(INVALID_REGEX_CONFIG, encoding='utf-8')
    caplog.set_level(logging.INFO)
    async with mcp_session(mode='single', config_file=path, env=SINGLE_MODE_ENV) as session:
        listed = await session.list_tools()
    golden(
        'gaps_config/tool_filter/invalid_regex_config_file',
        normalize(
            {'config': INVALID_REGEX_CONFIG, 'tools_list': listed, 'logs': _server_records(caplog)}
        ),
    )


async def test_invalid_regex_env(mcp_session, caplog, golden, normalize):
    caplog.set_level(logging.INFO)
    env = {**SINGLE_MODE_ENV, **INVALID_REGEX_ENV}
    async with mcp_session(mode='single', env=env) as session:
        listed = await session.list_tools()
    golden(
        'gaps_config/tool_filter/invalid_regex_env',
        normalize({'env': env, 'tools_list': listed, 'logs': _server_records(caplog)}),
    )


# ---------------------------------------------------------------------------
# clusters_information.py: read errors of the clusters file
# ---------------------------------------------------------------------------

# Latin-1 "é" (0xe9) followed by a newline: not UTF-8.
NOT_UTF8_CONFIG = (
    b'clusters:\n'
    b'  local:\n'
    b'    opensearch_url: http://localhost:9200\n'
    b'    opensearch_username: caf\xe9\n'
)


def _unicode_error(exc: BaseException) -> dict:
    data = _error(exc)
    if isinstance(exc, UnicodeDecodeError):
        data.update(
            encoding=exc.encoding,
            start=exc.start,
            end=exc.end,
            reason=exc.reason,
            object=exc.object.decode('latin-1'),
        )
    return data


async def test_clusters_file_not_utf8(tmp_path, mcp_session, caplog, golden, normalize):
    # clusters_information.py 88-89: the decode error is raised again with the path in its
    # reason. Multi mode reads the clusters file first, so nothing else reads it before.
    path = tmp_path / 'clusters.yml'
    path.write_bytes(NOT_UTF8_CONFIG)
    caplog.set_level(logging.INFO)
    result = {'config': NOT_UTF8_CONFIG.decode('latin-1')}
    try:
        await load_clusters_from_yaml(str(path))
    except Exception as exc:
        result['load_clusters_error'] = _unicode_error(exc)
    else:
        result['load_clusters_error'] = None
    result['load_clusters_registry'] = sorted(cluster_registry)
    result['load_clusters_logs'] = _server_records(caplog)
    caplog.clear()
    cluster_registry.clear()
    server = await _serve(mcp_session, (), mode='multi', config_file=path)
    server['clusters'] = sorted(cluster_registry)
    server['logs'] = _server_records(caplog)
    result['server'] = server
    golden('gaps_config/clusters/not_utf8', normalize(result))


def _open_error(path) -> OSError:
    """What ``open()`` raises on this OS for ``path`` (a directory)."""
    try:
        with open(path, encoding='utf-8'):
            pass
    except OSError as exc:
        return exc
    pytest.fail(f'open({path!r}) did not raise')


async def test_clusters_config_is_directory(tmp_path, mcp_session, caplog, golden, normalize):
    config_dir = tmp_path / 'config_dir'
    config_dir.mkdir()
    caplog.set_level(logging.INFO)
    try:
        await load_clusters_from_yaml(str(config_dir))
    except Exception as exc:
        direct = _error(exc)
    else:
        direct = None
    direct_logs = _server_records(caplog)
    caplog.clear()
    # As `--mode multi --config <directory>`: serve() loads the clusters first.
    server = await _serve(mcp_session, (), mode='multi', config_file=config_dir)
    server['clusters'] = sorted(cluster_registry)
    server['logs'] = _server_records(caplog)
    result = {'load_clusters_error': direct, 'load_clusters_logs': direct_logs, 'server': server}
    if sys.platform == 'win32':
        # characterization: current behaviour: on Windows open() on a directory raises
        # PermissionError ([Errno 13]), so the loader takes its PermissionError branch
        # (clusters_information.py 86-87), not the OSError one (96-97) it takes on Linux
        # and macOS (IsADirectoryError, [Errno 21]); the golden holds the POSIX outcome.
        open_error = _open_error(str(config_dir))
        assert isinstance(open_error, PermissionError), repr(open_error)
        expected_error = {
            'type': 'PermissionError',
            'message': f'Permission denied reading YAML file {config_dir}: {open_error}',
        }
        assert result == {
            'load_clusters_error': expected_error,
            'load_clusters_logs': [],
            'server': {
                'error': expected_error,
                'tool_names': None,
                'tools': None,
                'clusters': [],
                'logs': [],
            },
        }, json.dumps(result, indent=2)
        return
    # clusters_information.py 96-97: the OSError is raised again with the path in front.
    golden('gaps_config/clusters/config_is_directory', normalize(result))
