# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization: how ``main()`` turns argv, env and the config file into server settings.

Covers the command line of ``mcp_server_opensearch.main()`` (flags, unknown ``--tool.*``
overrides, ``--help``, bad values), the ``memory install`` and ``install-hooks``
subcommands (run with a temporary HOME and cwd), the YAML config file (clusters, tool
customization, memory container id, tool filters, errors) and ``configure_logging``.

``stdio_server.serve`` and ``streaming_server.serve`` are replaced by recorders only at the
call boundary of ``main()``: what is pinned is the call ``main()`` makes, not a running server.
"""

import asyncio
import contextlib
import copy
import importlib
import io
import json
import logging
import pytest
import re
import sys
import yaml
from datetime import datetime
from mcp_server_opensearch import logging_config, main, stdio_server, streaming_server
from mcp_server_opensearch.clusters_information import cluster_registry, load_clusters_from_yaml
from mcp_server_opensearch.logging_config import configure_logging, start_memory_monitor
from pathlib import Path
from tools.config import (
    apply_custom_tool_config,
    get_memory_container_id_from_config,
    parse_cli_to_nested_config,
)
from tools.tool_filter import get_allow_write_setting
from tools.tool_generator import generate_tools_from_openapi
from tools.tools import TOOL_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / 'example_config.yml'

# argv[0] as the console script names it; argparse prints it in usage and errors.
PROG = 'opensearch-mcp-server-py'

# Cluster fields holding a secret; the goldens record only whether they are set.
SECRET_CLUSTER_FIELDS = ('opensearch_password',)

# Registry entry values that are Python objects, not data.
OBJECT_ENTRY_FIELDS = ('function', 'args_model')

# The installer modules are imported inside main() (after HOME is set); see install_env.
INSTALL_MODULES = ('mcp_server_opensearch.install_hooks', 'mcp_server_opensearch.installer')

# Relative paths install-hooks prints and logs, as str(Path) renders them ("\" on Windows).
RELATIVE_INSTALL_PATHS = (
    Path('.kiro', 'hooks', 'memory-search-on-prompt.json'),
    Path('.kiro', 'hooks', 'memory-save-on-stop.json'),
    Path('.claude', 'settings.json'),
    Path('.cursor', 'hooks.json'),
)

SERVER_LOGGER_PREFIXES = ('mcp_server_opensearch', 'tools')

# Pydantic validation messages end with a docs link naming the installed pydantic version.
PYDANTIC_DOCS_VERSION = re.compile(r'(https://errors\.pydantic\.dev/)[^/\s]+/')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _error(exc: BaseException) -> dict:
    return {'type': type(exc).__name__, 'message': str(exc)}


def _lines(text: str) -> list:
    """Output as a list of lines; a line that is a JSON object (a json log record) is parsed."""
    lines = []
    for line in text.replace('\r\n', '\n').split('\n'):
        try:
            parsed = json.loads(line)
        except ValueError:
            parsed = None
        lines.append(parsed if isinstance(parsed, dict) else line)
    return lines


def _without_asyncio_logs(lines: list) -> list:
    # asyncio logs the event loop it picks at DEBUG ("Using selector: EpollSelector",
    # "Using proactor: IocpProactor"): a property of the OS, not of the server.
    return [
        line
        for line in lines
        if not (isinstance(line, dict) and line.get('logger') == 'asyncio')
        and not (isinstance(line, str) and ' - asyncio - ' in line)
    ]


def _records(caplog) -> list:
    return [
        {
            'level': record.levelname,
            'logger': record.name,
            'message': PYDANTIC_DOCS_VERSION.sub(r'\1<VERSION>/', record.getMessage()),
        }
        for record in caplog.records
        if record.name != 'asyncio'
    ]


def _server_records(caplog) -> list:
    """Records of the loggers the config code uses: root, ``mcp_server_opensearch``, ``tools``.

    Left out: the OpenSearch client's connection setup (``opensearch.*``) and third-party
    libraries (botocore's debug output depends on its version).
    """
    return [
        record
        for record in _records(caplog)
        if record['logger'] == 'root' or record['logger'].startswith(SERVER_LOGGER_PREFIXES)
    ]


def _posix_relative_paths(data):
    if isinstance(data, str):
        for path in RELATIVE_INSTALL_PATHS:
            data = data.replace(str(path), path.as_posix())
        return data
    if isinstance(data, dict):
        return {key: _posix_relative_paths(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_posix_relative_paths(value) for value in data]
    return data


def _redacted(cluster) -> dict:
    data = cluster.model_dump(mode='json')
    for field in SECRET_CLUSTER_FIELDS:
        if data.get(field) is not None:
            data[field] = '<REDACTED>'
    return data


def _clusters() -> dict:
    return {name: _redacted(info) for name, info in cluster_registry.items()}


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


def _resolve_tool_config(config_file_path: str, cli_tool_overrides: dict) -> dict:
    """Run ``apply_custom_tool_config`` on the global registry; describe what it changed."""
    before = _registry_view(TOOL_REGISTRY)
    descriptions = _model_descriptions(TOOL_REGISTRY)
    try:
        resolved = apply_custom_tool_config(TOOL_REGISTRY, config_file_path, cli_tool_overrides)
    except Exception as exc:
        return {'error': _error(exc)}
    return {
        'resolved_changes': _diff(before, _registry_view(resolved)),
        # characterization: current behaviour: apply_custom_tool_config also writes the
        # resolved entries back into the global TOOL_REGISTRY (default_tool_registry.update).
        'global_registry_changes': _diff(before, _registry_view(TOOL_REGISTRY)),
        # characterization: current behaviour: an argument description is also written on
        # the args model class, which every copy of the registry shares.
        'args_model_description_changes': _diff(descriptions, _model_descriptions(TOOL_REGISTRY)),
    }


def _files(root: Path) -> dict:
    """Every file below ``root``: JSON files parsed (and whether they are indent-2 dumps)."""
    files = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.json':
            try:
                data = json.loads(text)
            except ValueError:
                files[path.relative_to(root).as_posix()] = {'text': text}
                continue
            files[path.relative_to(root).as_posix()] = {
                'json': data,
                'written_as_json_dumps_indent_2': text == json.dumps(data, indent=2) + '\n',
            }
        else:
            files[path.relative_to(root).as_posix()] = {'text': text}
    return files


class _AnswerStdin(io.StringIO):
    """Standard input for ``input()``: each line read is echoed to stdout, as on a terminal."""

    def readline(self, size: int = -1) -> str:
        line = super().readline(size)
        sys.stdout.write(line)
        return line


class _SentinelHandler(logging.NullHandler):
    """A root handler present before main(), to see whether main() keeps it."""


# ---------------------------------------------------------------------------
# fixtures (this module only)
# ---------------------------------------------------------------------------


@pytest.fixture
def serve_calls(monkeypatch) -> list:
    """Replace both ``serve`` functions with recorders; the list of calls ``main()`` made."""
    calls = []

    def recorder(name):
        async def serve(*args, **kwargs):
            calls.append({'server': name, 'args': list(args), 'kwargs': kwargs})

        return serve

    monkeypatch.setattr(stdio_server, 'serve', recorder('stdio_server.serve'))
    monkeypatch.setattr(streaming_server, 'serve', recorder('streaming_server.serve'))
    return calls


@pytest.fixture
def run_main(monkeypatch, capsys):
    """``run_main(argv, stdin=None, root_level=WARNING)``: run ``main()``; return what it did.

    ``root_level`` is set on the root logger first, so the recorded level is main()'s doing;
    the installer tests pass caplog's level instead.
    """
    # argparse wraps help and usage at the terminal width; COLUMNS fixes it.
    monkeypatch.setenv('COLUMNS', '80')
    root = logging.getLogger()

    def run(argv: list, stdin=None, root_level: int = logging.WARNING) -> dict:
        monkeypatch.setattr(sys, 'argv', [PROG, *argv])
        if stdin is None:
            # Empty, so a server that did start would see EOF and stop.
            stdin = io.TextIOWrapper(io.BytesIO(b''), encoding='utf-8')
        monkeypatch.setattr(sys, 'stdin', stdin)
        root.setLevel(root_level)
        sentinel = _SentinelHandler()
        root.addHandler(sentinel)
        before = set(root.handlers)
        capsys.readouterr()
        result = {'argv': argv, 'exit_code': None, 'error': None}
        try:
            main()
        except SystemExit as exc:
            result['exit_code'] = exc.code
        except Exception as exc:
            result['error'] = _error(exc)
        out, err = capsys.readouterr()
        result['stdout'] = _lines(out)
        result['stderr'] = _without_asyncio_logs(_lines(err))
        result['logging'] = {
            'root_level': logging.getLevelName(root.level),
            'kept_existing_handler': sentinel in root.handlers,
            'added_handlers': [
                {
                    'type': type(handler).__name__,
                    'level': logging.getLevelName(handler.level),
                    'stream_is_stderr': getattr(handler, 'stream', None) is sys.stderr,
                    'formatter': type(handler.formatter).__name__,
                    'format': getattr(handler.formatter, '_fmt', None),
                }
                for handler in root.handlers
                if handler not in before and not type(handler).__module__.startswith('_pytest')
            ],
        }
        root.removeHandler(sentinel)
        return result

    return run


def _fresh_import(monkeypatch, name: str):
    """Import module ``name`` afresh; the test's undo puts back what was there before."""
    package, _, child = name.rpartition('.')
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    monkeypatch.setattr(sys.modules[package], child, None, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def install_env(monkeypatch, tmp_path):
    """A temporary HOME (``tmp_path/home``) and cwd (``tmp_path/cwd``) for the installers."""
    home = tmp_path / 'home'
    cwd = tmp_path / 'cwd'
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv('HOME', str(home))
    # Path.home() reads USERPROFILE on Windows.
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.chdir(cwd)
    assert Path.home() == home
    # characterization: current behaviour: installer.IDE_CONFIGS resolves Path.home() when
    # the module is imported, not when the installer runs. Import it now, with HOME in place,
    # so main() uses this copy; refuse to run if its paths are not under the temporary HOME.
    for name in INSTALL_MODULES:
        module = _fresh_import(monkeypatch, name)
    paths = [
        path
        for ide in module.IDE_CONFIGS.values()
        for key, path in ide.items()
        if key.endswith('_path')
    ]
    if not all(home in path.parents for path in paths):
        pytest.fail(f'installer paths are not under the temporary HOME {home}: {paths}')
    return {'home': home, 'cwd': cwd}


def _write_files(root: Path, files: dict) -> None:
    for relative, content in files.items():
        path = root.joinpath(*relative.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(content, str):
            content = json.dumps(content, indent=2) + '\n'
        path.write_text(content, encoding='utf-8')


def _install_files(env: dict) -> dict:
    return {'home': _files(env['home']), 'cwd': _files(env['cwd'])}


# ---------------------------------------------------------------------------
# main(): flags
# ---------------------------------------------------------------------------

FLAG_CASES = {
    'no_args': [],
    'transport_stdio': ['--transport', 'stdio'],
    'transport_stream': ['--transport', 'stream'],
    # characterization: current behaviour: --host and --port are accepted with the stdio
    # transport and silently dropped (only the streaming server receives them).
    'host': ['--host', '0.0.0.0'],
    'port': ['--port', '8080'],
    'transport_stream_host_port': ['--transport', 'stream', '--host', '0.0.0.0', '--port', '8080'],
    'mode_single': ['--mode', 'single'],
    'mode_multi': ['--mode', 'multi'],
    'profile': ['--profile', 'my-profile'],
    # The path is passed on as given: not resolved, not checked for existence.
    'config': ['--config', 'my-config.yml'],
    'debug': ['--debug'],
    'log_format_text': ['--log-format', 'text'],
    'log_format_json': ['--log-format', 'json'],
    # characterization: current behaviour: a word that is not a subcommand ("memory" without
    # "install", or "install" as the installer's docstring spells it) is ignored and the
    # stdio server starts.
    'memory_without_install': ['memory'],
    'install_without_memory': ['install'],
}


@pytest.mark.parametrize('case', FLAG_CASES)
def test_main_flag(case, run_main, serve_calls, golden, normalize):
    result = run_main(FLAG_CASES[case])
    result['serve_calls'] = serve_calls
    golden(f'cli_config/main/{case}', normalize(result))


BAD_VALUE_CASES = {
    'port_not_int': ['--port', 'abc'],
    'transport_not_a_choice': ['--transport', 'bogus'],
}


@pytest.mark.parametrize('case', BAD_VALUE_CASES)
def test_main_bad_flag_value(case, run_main, serve_calls, golden, normalize):
    result = run_main(BAD_VALUE_CASES[case])
    result['serve_calls'] = serve_calls
    golden(f'cli_config/main/bad_value_{case}', normalize(result))


def test_main_help(run_main, serve_calls, golden, normalize):
    result = run_main(['--help'])
    result['serve_calls'] = serve_calls
    golden('cli_config/main/help', normalize(result))


# ---------------------------------------------------------------------------
# main(): unknown --tool.* overrides (parse_unknown_args_to_dict)
# ---------------------------------------------------------------------------

OVERRIDE_CASES = {
    'equals_form': [
        '--tool.ListIndexTool.display_name=Index_Lister',
        '--tool.ListIndexTool.args.index.description=Which index to list',
        '--tool.SearchIndexTool.max_size_limit=100',
        # Only the first "=" separates the key: the value keeps the rest.
        '--tool.GetShardsTool.description=a=b',
    ],
    'space_form': [
        '--tool.ListIndexTool.description',
        'Lists the indices',
        '--tool.GetShardsTool.display_name',
        'Shard_Lister',
    ],
    'duplicate_last_wins': [
        '--tool.ListIndexTool.description=first',
        '--tool.ListIndexTool.description=second',
    ],
    # parse_unknown_args_to_dict keeps every key; parse_cli_to_nested_config then drops keys
    # that are not tool.<name>.<field>, and fields other than display_name, description,
    # args and max_size_limit (http_methods among them).
    'dropped_keys': [
        '--foo=bar',
        '--tool.ListIndexTool=whole-tool',
        '--tool.ListIndexTool.http_methods=POST',
        '--tool.ListIndexTool.description=kept',
    ],
    # characterization: current behaviour: override values go through yaml.safe_load, so
    # "yes" becomes True, "null" None and "1e3" stays a string (YAML 1.1 floats need a dot).
    'yaml_coerced_values': [
        '--tool.GetIndexInfoTool.description=yes',
        '--tool.ListIndexTool.description=null',
        '--tool.SearchIndexTool.max_size_limit=1e3',
        '--tool.GetShardsTool.max_size_limit=2048',
    ],
    # A flag with no value: the overrides parser exits with status 2.
    'valueless_flag': ['--tool.ListIndexTool.description'],
}


@pytest.mark.parametrize('case', OVERRIDE_CASES)
def test_main_tool_overrides(case, run_main, serve_calls, golden, normalize):
    result = run_main(OVERRIDE_CASES[case])
    result['serve_calls'] = serve_calls
    if serve_calls:
        overrides = serve_calls[0]['kwargs']['cli_tool_overrides']
        # What the server does next with them (serve -> apply_custom_tool_config).
        result['nested_config'] = parse_cli_to_nested_config(overrides)
        result['applied'] = _resolve_tool_config('', overrides)
    golden(f'cli_config/overrides/{case}', normalize(result))


def test_main_tool_override_unknown_tool_fails_startup(run_main, golden, normalize):
    # The real stdio serve runs here: startup fails in apply_custom_tool_config before the
    # server reads stdin (which is empty, so a server that did start would stop at EOF).
    result = run_main(['--tool.NoSuchTool.description=x'])
    golden('cli_config/overrides/unknown_tool', normalize(result))


# ---------------------------------------------------------------------------
# memory install (interactive)
# ---------------------------------------------------------------------------

AOSS_URL = 'https://abc123example.us-west-2.aoss.amazonaws.com'
ES_URL = 'https://search-example.eu-west-1.es.amazonaws.com'
GIT_URL = 'https://github.com/example/opensearch-mcp-server-py.git@feature'

MEMORY_INSTALL_CASES = {
    # All three IDEs detected; defaults accepted. Not an .aoss. URL: region defaults to us-east-1.
    'plain': {
        'args': [],
        'ides': ['.kiro', '.claude', '.cursor'],
        'answers': [ES_URL, '', '', '', 'y', 'yes'],
    },
    'from_git_space': {
        'args': ['--from-git', GIT_URL],
        'ides': ['.claude'],
        'answers': [ES_URL, 'eu-west-1', 'dev-profile', 'y'],
    },
    'from_git_equals': {
        'args': [f'--from-git={GIT_URL}'],
        'ides': ['.cursor'],
        'answers': [ES_URL, '', '', 'y'],
    },
    'from_local_path': {
        'args': ['--from-local', 'checkout/opensearch-mcp-server-py'],
        'ides': ['.kiro'],
        'answers': [ES_URL, '', '', 'y'],
    },
    'from_local_equals': {
        'args': ['--from-local=checkout'],
        'ides': ['.kiro'],
        'answers': [ES_URL, '', '', 'y'],
    },
    # No value: the current directory.
    'from_local_no_path': {
        'args': ['--from-local'],
        'ides': ['.kiro'],
        'answers': [ES_URL, '', '', 'y'],
    },
    # characterization: current behaviour: "--from-local" followed by another flag takes
    # that flag as its path; the flag is then also parsed, and --from-local wins.
    'from_local_then_from_git': {
        'args': ['--from-local', '--from-git', GIT_URL],
        'ides': ['.kiro'],
        'answers': [ES_URL, '', '', 'y'],
    },
    'empty_url_stops': {
        'args': [],
        'ides': ['.kiro', '.claude', '.cursor'],
        'answers': [''],
    },
    # Region parsed from the .aoss. URL; a declined IDE is left alone.
    'aoss_url_region_parsed': {
        'args': [],
        'ides': ['.kiro', '.claude'],
        'answers': [AOSS_URL, '', '', 'n', 'N'],
    },
    'no_ide_detected': {
        'args': [],
        'ides': [],
        'answers': [AOSS_URL, '', ''],
    },
    # Every target already has its key: nothing is written.
    'existing_keys_skipped': {
        'args': [],
        'ides': ['.kiro', '.claude', '.cursor'],
        'given': {
            '.kiro/settings/mcp.json': {'mcpServers': {'opensearch-memory': {'command': 'old'}}},
            '.kiro/steering/opensearch-memory.md': 'old steering\n',
            '.claude.json': {'mcpServers': {'opensearch-memory': {'command': 'old'}}},
            '.claude/CLAUDE.md': '# Mine\n\n## OpenSearch Agent Memory\n\nold\n',
            '.cursor/mcp.json': {'mcpServers': {'opensearch-memory': {'command': 'old'}}},
            '.cursor/hooks.json': {
                'version': 1,
                'hooks': {'stop': [{'command': 'echo SaveMemoryTool'}]},
            },
        },
        'answers': [AOSS_URL, '', '', 'y', 'y', 'y'],
    },
    # Existing files without our keys: merged, other keys kept, CLAUDE.md appended to.
    'existing_files_merged': {
        'args': [],
        'ides': ['.kiro', '.claude', '.cursor'],
        'given': {
            '.kiro/settings/mcp.json': {'other': True},
            '.claude.json': {'numStartups': 3, 'mcpServers': {'other-server': {'command': 'x'}}},
            '.claude/CLAUDE.md': '# Mine\n',
            '.cursor/mcp.json': {'mcpServers': {'other-server': {'command': 'x'}}},
            '.cursor/hooks.json': {'hooks': {'stop': [{'command': 'echo other'}]}},
        },
        'answers': [AOSS_URL, '', '', 'y', 'y', 'y'],
    },
    'claude_json_unparseable_skipped': {
        'args': [],
        'ides': ['.claude'],
        'given': {'.claude.json': '{not json'},
        'answers': [AOSS_URL, '', '', 'y'],
    },
    # characterization: current behaviour: Kiro and Cursor do not guard against an
    # unparseable mcp.json (Claude Code does): the installer stops with the JSON error.
    'cursor_mcp_json_unparseable_fails': {
        'args': [],
        'ides': ['.cursor'],
        'given': {'.cursor/mcp.json': '{not json'},
        'answers': [AOSS_URL, '', '', 'y'],
    },
}


@pytest.mark.parametrize('case', MEMORY_INSTALL_CASES)
def test_memory_install(case, install_env, run_main, caplog, golden, normalize):
    spec = MEMORY_INSTALL_CASES[case]
    home = install_env['home']
    for ide in spec['ides']:
        (home / ide).mkdir(parents=True, exist_ok=True)
    _write_files(home, spec.get('given', {}))
    caplog.set_level(logging.DEBUG)
    stdin = _AnswerStdin(''.join(f'{answer}\n' for answer in spec['answers']))
    result = run_main(['memory', 'install', *spec['args']], stdin=stdin, root_level=logging.DEBUG)
    # The prompts, with the answers echoed: what a terminal shows.
    result['transcript'] = result.pop('stdout')
    result['stdin_left_unread'] = stdin.read()
    del result['logging']
    result['answers'] = spec['answers']
    result['given_home'] = spec.get('given', {})
    result['files'] = _install_files(install_env)
    result['logs'] = _records(caplog)
    golden(f'cli_config/memory_install/{case}', _posix_relative_paths(normalize(result)))


# ---------------------------------------------------------------------------
# install-hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('scope', ['workspace', 'user'])
@pytest.mark.parametrize('client', ['kiro', 'claude-code', 'cursor'])
def test_install_hooks(client, scope, install_env, run_main, caplog, golden, normalize):
    # characterization: current behaviour: kiro with --scope user writes to the workspace
    # (cwd) after a warning; the other clients write under HOME for the user scope.
    caplog.set_level(logging.DEBUG)
    argv = ['install-hooks', '--client', client, '--scope', scope]
    runs = []
    # The second run is the idempotency check: it must find the hooks in place.
    for _ in range(2):
        caplog.clear()
        result = run_main(argv, root_level=logging.DEBUG)
        del result['logging']
        result['logs'] = _records(caplog)
        result['files'] = _install_files(install_env)
        runs.append(result)
    golden(
        f'cli_config/install_hooks/{client}_{scope}',
        _posix_relative_paths(
            normalize({'argv': argv, 'first_run': runs[0], 'second_run': runs[1]})
        ),
    )


INSTALL_HOOKS_EXISTING_CASES = {
    # characterization: current behaviour: the old flat-list "hooks" format is "migrated" by
    # dropping the old entries and writing the event map; other settings keys are kept.
    'claude_code_old_list_format': {
        'client': 'claude-code',
        'given': {
            '.claude/settings.json': {
                'model': 'some-model',
                'hooks': [{'event': 'Stop', 'command': 'echo old list hook'}],
            }
        },
    },
    'claude_code_existing_map_merged': {
        'client': 'claude-code',
        'given': {
            '.claude/settings.json': {
                'hooks': {'Stop': [{'hooks': [{'type': 'command', 'command': 'echo mine'}]}]}
            }
        },
    },
    # A hook whose base64 payload names a memory tool counts as installed.
    'claude_code_legacy_base64_detected': {
        'client': 'claude-code',
        'given': {
            '.claude/settings.json': {
                'hooks': {
                    'Stop': [
                        {
                            'hooks': [
                                {
                                    'type': 'command',
                                    'command': (
                                        'python3 -c "import base64; '
                                        "base64.b64decode('U2F2ZU1lbW9yeVRvb2w=')\""
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        },
    },
    'cursor_existing_hooks_merged': {
        'client': 'cursor',
        'given': {
            '.cursor/hooks.json': {'version': 0, 'hooks': {'stop': [{'command': 'echo mine'}]}}
        },
    },
    'kiro_one_hook_exists': {
        'client': 'kiro',
        'given': {'.kiro/hooks/memory-save-on-stop.json': {'custom': True}},
    },
}


@pytest.mark.parametrize('case', INSTALL_HOOKS_EXISTING_CASES)
def test_install_hooks_existing_files(case, install_env, run_main, caplog, golden, normalize):
    spec = INSTALL_HOOKS_EXISTING_CASES[case]
    _write_files(install_env['cwd'], spec['given'])
    caplog.set_level(logging.DEBUG)
    argv = ['install-hooks', '--client', spec['client']]
    result = run_main(argv, root_level=logging.DEBUG)
    del result['logging']
    result['given_cwd'] = spec['given']
    result['logs'] = _records(caplog)
    result['files'] = _install_files(install_env)
    golden(f'cli_config/install_hooks/{case}', _posix_relative_paths(normalize(result)))


INSTALL_HOOKS_BAD_ARGS = {
    'missing_client': ['install-hooks'],
    'unknown_client': ['install-hooks', '--client', 'vim'],
    'unknown_scope': ['install-hooks', '--client', 'kiro', '--scope', 'global'],
}


@pytest.mark.parametrize('case', INSTALL_HOOKS_BAD_ARGS)
def test_install_hooks_bad_args(case, install_env, run_main, golden, normalize):
    result = run_main(INSTALL_HOOKS_BAD_ARGS[case])
    del result['logging']
    result['files'] = _install_files(install_env)
    golden(f'cli_config/install_hooks/bad_args_{case}', normalize(result))


# ---------------------------------------------------------------------------
# config file: example_config.yml
# ---------------------------------------------------------------------------


async def test_example_config_clusters(caplog, golden, normalize):
    caplog.set_level(logging.DEBUG)
    await load_clusters_from_yaml(str(EXAMPLE_CONFIG))
    golden(
        'cli_config/config_file/example_clusters',
        normalize({'clusters': _clusters(), 'logs': _records(caplog)}),
    )


async def test_example_config_tool_config(caplog, golden, normalize):
    caplog.set_level(logging.DEBUG)
    # As at startup: the OpenAPI-generated tools first, then the custom tool config.
    await generate_tools_from_openapi()
    caplog.clear()
    # characterization: current behaviour: with a config file that has a tools section,
    # CLI overrides are ignored entirely.
    result = _resolve_tool_config(
        str(EXAMPLE_CONFIG), {'tool.GetSegmentsTool.description': 'from the CLI'}
    )
    result['logs'] = _records(caplog)
    golden('cli_config/config_file/example_tool_config', normalize(result))


async def test_example_config_multi_mode_server(mcp_session, caplog, golden, normalize):
    caplog.set_level(logging.INFO)
    async with mcp_session(mode='multi', config_file=EXAMPLE_CONFIG) as session:
        tools = (await session.list_tools())['tools']
    customized = {'My_Custom_Index_Lister', 'Super_Searcher', 'GetShardsTool'}
    golden(
        'cli_config/config_file/example_multi_mode_server',
        normalize(
            {
                'tool_names': [tool['name'] for tool in tools],
                'customized_tools': [tool for tool in tools if tool['name'] in customized],
                'clusters': _clusters(),
                'logs': _records(caplog),
            }
        ),
    )


async def test_example_config_memory_container_id(monkeypatch, caplog, golden, normalize):
    caplog.set_level(logging.DEBUG)
    result = {}
    for env_value in (None, 'env-container'):
        if env_value is None:
            monkeypatch.delenv('OPENSEARCH_MEMORY_CONTAINER_ID', raising=False)
        else:
            monkeypatch.setenv('OPENSEARCH_MEMORY_CONTAINER_ID', env_value)
        caplog.clear()
        result[f'env={env_value}'] = {
            'container_id': get_memory_container_id_from_config(str(EXAMPLE_CONFIG)),
            'logs': _records(caplog),
        }
    golden('cli_config/config_file/example_memory_container_id', normalize(result))


# ---------------------------------------------------------------------------
# config file: errors
# ---------------------------------------------------------------------------

CONFIG_ERROR_CASES = {
    'missing_file': None,
    'malformed_yaml': 'clusters:\n  local: [unclosed\n',
    'unknown_tool_field': 'tools:\n  ListIndexTool:\n    bogus: 1\n',
    'unknown_tool_name': 'tools:\n  NoSuchTool:\n    description: x\n',
    # characterization: current behaviour: unknown keys elsewhere are ignored without a word.
    'unknown_keys_ignored': (
        'bogus_section:\n'
        '  a: 1\n'
        'clusters:\n'
        '  local:\n'
        '    opensearch_url: http://localhost:9200\n'
        '    bogus_cluster_key: 1\n'
        'tool_filters:\n'
        '  bogus_filter: [ListIndexTool]\n'
        '  settings:\n'
        '    bogus_setting: true\n'
    ),
    'cluster_without_url': 'clusters:\n  broken:\n    opensearch_username: admin\n',
    # characterization: current behaviour: an empty file fails multi-mode startup with an
    # AttributeError (None has no .get).
    'empty_file': '',
}


@pytest.mark.parametrize('mode', ['multi', 'single'])
@pytest.mark.parametrize('case', CONFIG_ERROR_CASES)
async def test_config_file_errors(case, mode, tmp_path, mcp_session, caplog, golden, normalize):
    content = CONFIG_ERROR_CASES[case]
    path = tmp_path / 'config.yml'
    if content is not None:
        path.write_text(content, encoding='utf-8')
    caplog.set_level(logging.DEBUG)
    result = {'config': content, 'mode': mode, 'error': None, 'tool_names': None}
    try:
        # Single mode reaches the fake OpenSearch for its version with no-auth.
        async with mcp_session(
            mode=mode, config_file=path, env={'OPENSEARCH_NO_AUTH': 'true'}
        ) as session:
            result['tool_names'] = [tool['name'] for tool in (await session.list_tools())['tools']]
    except Exception as exc:
        result['error'] = _error(exc)
    result['clusters'] = _clusters()
    result['logs'] = _server_records(caplog)
    golden(f'cli_config/config_errors/{case}_{mode}', normalize(result))


# ---------------------------------------------------------------------------
# config keys example_config.yml lacks
# ---------------------------------------------------------------------------

CLUSTER_OPTIONS_CONFIG = """\
clusters:
  explicit:
    opensearch_url: https://a.example.com
    timeout: 30
    ssl_verify: false
  as_strings:
    opensearch_url: https://b.example.com
    timeout: "45"
    ssl_verify: "no"
  verify_on:
    opensearch_url: https://c.example.com
    ssl_verify: true
  defaults:
    opensearch_url: https://d.example.com
  fractional_timeout:
    opensearch_url: https://e.example.com
    timeout: 1.5
  bad_ssl_verify:
    opensearch_url: https://f.example.com
    ssl_verify: sometimes
"""


async def test_cluster_timeout_and_ssl_verify(tmp_path, caplog, golden, normalize):
    path = tmp_path / 'clusters.yml'
    path.write_text(CLUSTER_OPTIONS_CONFIG, encoding='utf-8')
    caplog.set_level(logging.DEBUG)
    await load_clusters_from_yaml(str(path))
    golden(
        'cli_config/config_keys/cluster_timeout_ssl_verify',
        normalize(
            {
                'config': yaml.safe_load(CLUSTER_OPTIONS_CONFIG),
                'clusters': _clusters(),
                'logs': _records(caplog),
            }
        ),
    )


TOOL_FILTER_CASES = {
    'tool_category_enabled': {
        'tool_category': {'my_tools': ['GetSegmentsTool', 'CatNodesTool']},
        'tool_filters': {'enabled_categories': ['my_tools']},
    },
    'disabled_tools_regex_categories': {
        'tool_category': {'my_tools': ['MsearchTool']},
        'tool_filters': {
            'disabled_tools': ['ExplainTool'],
            'disabled_tools_regex': ['^Get.*'],
            'disabled_categories': ['my_tools'],
        },
    },
    'enabled_tools_and_regex': {
        'tool_filters': {
            'enabled_tools': ['GetClusterStateTool'],
            'enabled_tools_regex': ['^Search.*Tool$'],
        },
    },
    'allow_write_false': {
        'tool_filters': {
            'enabled_categories': ['search_relevance'],
            'settings': {'allow_write': False},
        },
    },
    'allow_write_categories': {
        'tool_filters': {
            'enabled_categories': ['search_relevance'],
            'settings': {'allow_write': False, 'allow_write_categories': ['search_relevance']},
        },
    },
    'unknown_names': {
        'tool_filters': {
            'enabled_tools': ['NoSuchTool'],
            'enabled_categories': ['no_such_category'],
            'disabled_tools': ['AlsoMissingTool'],
        },
    },
    # Filters match the display name a tools section gives.
    'display_name_filtered': {
        'tools': {'ListIndexTool': {'display_name': 'Index_Lister'}},
        'tool_filters': {'disabled_tools': ['Index_Lister']},
    },
}

# Environment filters, ignored when a config file is given. characterization: current
# behaviour: the "Both config file and environment variables are set" warning is logged with
# any config file, env or not, because the env defaults include allow_write=True.
TOOL_FILTER_ENV = {'OPENSEARCH_DISABLED_TOOLS': 'ListIndexTool', 'OPENSEARCH_NO_AUTH': 'true'}


@pytest.mark.parametrize('case', TOOL_FILTER_CASES)
async def test_tool_filter_config_keys(case, tmp_path, mcp_session, caplog, golden, normalize):
    config = TOOL_FILTER_CASES[case]
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    caplog.set_level(logging.INFO)
    async with mcp_session(mode='single', config_file=path, env=TOOL_FILTER_ENV) as session:
        tools = (await session.list_tools())['tools']
    golden(
        f'cli_config/config_keys/tool_filter_{case}',
        normalize(
            {
                'config': config,
                'env': TOOL_FILTER_ENV,
                'tool_names': [tool['name'] for tool in tools],
                'allow_write_setting': get_allow_write_setting(),
                'logs': _server_records(caplog),
            }
        ),
    )


async def test_tool_filters_ignored_in_multi_mode(tmp_path, mcp_session, golden, normalize):
    config = {
        'clusters': {'local': {'opensearch_url': 'http://localhost:9200'}},
        'tool_category': {'my_tools': ['GetSegmentsTool']},
        'tool_filters': {
            'disabled_tools': ['ListIndexTool'],
            'enabled_categories': ['my_tools'],
            'settings': {'allow_write': False},
        },
    }
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    async with mcp_session(mode='multi', config_file=path) as session:
        tools = (await session.list_tools())['tools']
    golden(
        'cli_config/config_keys/tool_filters_multi_mode',
        normalize(
            {
                'config': config,
                'tool_names': [tool['name'] for tool in tools],
                # Resolved from tool_filters.settings in multi mode too.
                'allow_write_setting': get_allow_write_setting(),
            }
        ),
    )


AGENTIC_MEMORY_TOOLS = (
    'CreateAgenticMemorySessionTool',
    'AddAgenticMemoriesTool',
    'GetAgenticMemoryTool',
    'UpdateAgenticMemoryTool',
    'DeleteAgenticMemoryByIDTool',
    'DeleteAgenticMemoryByQueryTool',
    'SearchAgenticMemoryTool',
)


async def test_agentic_memory_container_id(tmp_path, monkeypatch, caplog, golden, normalize):
    configs = {
        'with_id': 'agentic_memory:\n  memory_container_id: file-container\n',
        'empty_id': 'agentic_memory:\n  memory_container_id: ""\n',
        'not_a_mapping': 'agentic_memory: file-container\n',
        'malformed_yaml': 'agentic_memory: [unclosed\n',
    }
    paths = {'no_config': ''}
    for name, content in configs.items():
        path = tmp_path / f'{name}.yml'
        path.write_text(content, encoding='utf-8')
        paths[name] = str(path)
    paths['missing_file'] = str(tmp_path / 'missing.yml')
    caplog.set_level(logging.DEBUG)
    lookups = {}
    for env_value in (None, 'env-container'):
        if env_value is None:
            monkeypatch.delenv('OPENSEARCH_MEMORY_CONTAINER_ID', raising=False)
        else:
            monkeypatch.setenv('OPENSEARCH_MEMORY_CONTAINER_ID', env_value)
        for name, path in paths.items():
            caplog.clear()
            lookups[f'{name}/env={env_value}'] = {
                'container_id': get_memory_container_id_from_config(path),
                'logs': _records(caplog),
            }
    monkeypatch.delenv('OPENSEARCH_MEMORY_CONTAINER_ID', raising=False)
    resolved = apply_custom_tool_config(TOOL_REGISTRY, paths['with_id'], {})
    schemas = {
        name: {
            'memory_container_id': resolved[name]['input_schema']['properties'].get(
                'memory_container_id'
            ),
            'required': resolved[name]['input_schema'].get('required'),
        }
        for name in AGENTIC_MEMORY_TOOLS
    }
    golden(
        'cli_config/config_keys/agentic_memory_container_id',
        normalize({'configs': configs, 'lookups': lookups, 'resolved_schemas': schemas}),
    )


async def test_agentic_memory_container_id_server(tmp_path, mcp_session, golden, normalize):
    path = tmp_path / 'config.yml'
    path.write_text('agentic_memory:\n  memory_container_id: file-container\n', encoding='utf-8')
    async with mcp_session(
        mode='single', config_file=path, env={'OPENSEARCH_NO_AUTH': 'true'}
    ) as session:
        tools = (await session.list_tools())['tools']
    # characterization: current behaviour: example_config.yml says a configured container id
    # enables the agentic memory tools; the tool filter does not enable their category.
    golden(
        'cli_config/config_keys/agentic_memory_container_id_server',
        normalize(
            {
                'tool_names': [tool['name'] for tool in tools],
                'agentic_memory_tools_listed': [
                    tool for tool in tools if tool['name'] in AGENTIC_MEMORY_TOOLS
                ],
            }
        ),
    )


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class _FixedStr:
    def __str__(self) -> str:
        return 'fixed-str'


@pytest.mark.parametrize('log_format', ['text', 'json'])
def test_configure_logging_record(log_format, capsys, golden, normalize):
    configure_logging(level=logging.INFO, log_format=log_format)
    logger = logging.getLogger('characterization.cli_config')
    logger.debug('below the level: not written')
    capsys.readouterr()
    logger.info(
        'one record: %s',
        'formatted',
        extra={
            'event_type': 'probe',
            'count': 3,
            'when': datetime(2020, 1, 1),
            'obj': _FixedStr(),
            '_private': 'dropped',
        },
    )
    out, err = capsys.readouterr()
    golden(
        f'cli_config/logging/record_{log_format}',
        normalize({'stdout': out, 'stderr': _lines(err)}),
    )


async def _first_snapshot(monkeypatch, capsys) -> dict:
    """Run the memory monitor until its first snapshot; its sleeps and what it logged."""
    sleeps = []

    async def recording_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            await asyncio.Event().wait()  # park until cancelled

    monkeypatch.setattr(logging_config, '_async_sleep', recording_sleep)
    capsys.readouterr()
    task = start_memory_monitor()
    for _ in range(1000):
        if len(sleeps) > 1:
            break
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    out, err = capsys.readouterr()
    return {'sleep_seconds': sleeps, 'stdout': out, 'stderr': _lines(err)}


@pytest.mark.parametrize('log_format', ['json', 'text'])
async def test_memory_snapshot_event(log_format, monkeypatch, capsys, golden, normalize):
    monkeypatch.setenv('OPENSEARCH_MEMORY_MONITOR_INTERVAL', '1')
    configure_logging(level=logging.INFO, log_format=log_format)
    result = await _first_snapshot(monkeypatch, capsys)
    golden(f'cli_config/logging/memory_snapshot_{log_format}', normalize(result))


async def test_memory_monitor_interval_from_env(monkeypatch, capsys, golden, normalize):
    configure_logging(level=logging.WARNING)
    result = {}
    for value in (None, '1', '0', '-5', '30', 'abc'):
        if value is None:
            monkeypatch.delenv('OPENSEARCH_MEMORY_MONITOR_INTERVAL', raising=False)
        else:
            monkeypatch.setenv('OPENSEARCH_MEMORY_MONITOR_INTERVAL', value)
        try:
            snapshot = await _first_snapshot(monkeypatch, capsys)
        except Exception as exc:
            result[f'env={value}'] = {'error': _error(exc)}
        else:
            result[f'env={value}'] = {'sleep_seconds': snapshot['sleep_seconds']}
    golden('cli_config/logging/memory_monitor_interval', normalize(result))
