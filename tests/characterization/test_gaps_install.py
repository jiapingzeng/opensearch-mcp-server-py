# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: installer and hook edge cases the other modules leave unpinned.

Run through ``main()`` (``install-hooks`` and ``memory install``) with a temporary HOME and
cwd, as ``test_cli_config.py`` does; never the real HOME:

- ``install_hooks._claude_hooks_already_installed``: the legacy base64 idempotency check
  when the command names no ``b64decode('...')`` call, when the payload names no memory
  tool, and when decoding fails (bad padding, not UTF-8).
- ``install_hooks._cursor_hooks_already_installed``: the same check for Cursor, and a
  legacy payload that names a memory tool.
- ``install_hooks._install_cursor_hooks``: an existing hooks file without a ``hooks`` key.
- ``installer._configure_cursor_mcp``: an existing ``~/.cursor/mcp.json`` without
  ``mcpServers``.

Each golden holds the files the run left under HOME and cwd, what it printed and logged.
``install-hooks`` runs twice: the second run shows whether the first run's result counts
as installed.
"""

import importlib
import io
import json
import logging
import pytest
import sys
from mcp_server_opensearch import main
from pathlib import Path


# argv[0] as the console script names it.
PROG = 'opensearch-mcp-server-py'

# The installer modules, imported afresh once HOME is set; see install_env.
INSTALL_MODULES = ('mcp_server_opensearch.install_hooks', 'mcp_server_opensearch.installer')

# Relative paths install-hooks prints, as str(Path) renders them ("\" on Windows).
RELATIVE_INSTALL_PATHS = (
    Path('.claude', 'settings.json'),
    Path('.cursor', 'hooks.json'),
)

AOSS_URL = 'https://abc123example.us-west-2.aoss.amazonaws.com'

# base64 payloads for the legacy-format check.
MEMORY_TOOL_PAYLOAD = 'U2F2ZU1lbW9yeVRvb2w='  # "SaveMemoryTool"
OTHER_PAYLOAD = 'aGVsbG8='  # "hello"
BAD_PADDING_PAYLOAD = 'abc'  # binascii.Error: Incorrect padding
NOT_UTF8_PAYLOAD = '/w=='  # b'\xff': UnicodeDecodeError


def _decode_command(payload: str, quote: str = "'") -> str:
    """A legacy-style hook command decoding ``payload`` (quoted with ``quote``)."""
    return f'python3 -c "import base64; base64.b64decode({quote}{payload}{quote})"'


# 'base64' in the command, but no b64decode('...') call for the check's pattern to find.
NO_DECODE_CALL_COMMAND = 'python3 -c "import base64; print(base64.b64encode(b\'x\'))"'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lines(text: str) -> list:
    return text.replace('\r\n', '\n').split('\n')


def _records(caplog) -> list:
    return [
        {'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        for record in caplog.records
        if record.name != 'asyncio'
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


def _files(root: Path) -> dict:
    """Every file below ``root``: JSON files parsed (and whether they are indent-2 dumps)."""
    files = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8')
        try:
            data = json.loads(text) if path.suffix == '.json' else None
        except ValueError:
            data = None
        if data is None:
            files[relative] = {'text': text}
        else:
            files[relative] = {
                'json': data,
                'written_as_json_dumps_indent_2': text == json.dumps(data, indent=2) + '\n',
            }
    return files


def _write_files(root: Path, files: dict) -> None:
    for relative, content in files.items():
        path = root.joinpath(*relative.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(content, str):
            content = json.dumps(content, indent=2) + '\n'
        path.write_text(content, encoding='utf-8')


class _AnswerStdin(io.StringIO):
    """Standard input for ``input()``: each line read is echoed to stdout, as on a terminal."""

    def readline(self, size: int = -1) -> str:
        line = super().readline(size)
        sys.stdout.write(line)
        return line


def _fresh_import(monkeypatch, name: str):
    """Import module ``name`` afresh; the test's undo puts back what was there before."""
    package, _, child = name.rpartition('.')
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    monkeypatch.setattr(sys.modules[package], child, None, raising=False)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# fixtures (this module only)
# ---------------------------------------------------------------------------


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
    # installer.IDE_CONFIGS resolves Path.home() at import: import both modules afresh with
    # HOME in place, so main() uses these copies, and refuse to run if a path escapes HOME.
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


@pytest.fixture
def run_main(monkeypatch, capsys, caplog, install_env):
    """``run_main(argv, answers=None)``: run ``main()`` once; return what it did."""
    caplog.set_level(logging.DEBUG)

    def run(argv: list, answers: list | None = None) -> dict:
        monkeypatch.setattr(sys, 'argv', [PROG, *argv])
        stdin = _AnswerStdin(''.join(f'{answer}\n' for answer in answers or []))
        monkeypatch.setattr(sys, 'stdin', stdin)
        capsys.readouterr()
        caplog.clear()
        result = {'exit_code': None, 'error': None}
        try:
            main()
        except SystemExit as exc:
            result['exit_code'] = exc.code
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        out, err = capsys.readouterr()
        result['stdout'] = _lines(out)
        result['stderr'] = _lines(err)
        result['stdin_left_unread'] = stdin.read()
        result['logs'] = _records(caplog)
        result['files'] = {
            'home': _files(install_env['home']),
            'cwd': _files(install_env['cwd']),
        }
        return result

    return run


# ---------------------------------------------------------------------------
# install-hooks: legacy base64 idempotency check, Cursor file without "hooks"
# ---------------------------------------------------------------------------


def _claude_settings(*commands: str) -> dict:
    return {
        '.claude/settings.json': {
            'model': 'some-model',
            'hooks': {
                'Stop': [{'hooks': [{'type': 'command', 'command': c} for c in commands]}],
            },
        }
    }


def _cursor_hooks(*commands: str) -> dict:
    return {
        '.cursor/hooks.json': {
            'version': 1,
            'hooks': {'stop': [{'command': c} for c in commands]},
        }
    }


INSTALL_HOOKS_CASES = {
    # install_hooks.py 260->248: 'base64' in the command, no b64decode('...') to decode.
    'claude_code_base64_no_decode_call': {
        'client': 'claude-code',
        'given': _claude_settings(NO_DECODE_CALL_COMMAND),
    },
    # install_hooks.py 260->248. characterization: current behaviour: the pattern only
    # matches a single-quoted b64decode('...'), so a double-quoted payload naming a memory
    # tool is not seen and the memory hooks are added next to it.
    'claude_code_base64_double_quoted_memory_payload': {
        'client': 'claude-code',
        'given': _claude_settings(_decode_command(MEMORY_TOOL_PAYLOAD, quote='"')),
    },
    # install_hooks.py 262->248: the payload decodes, and names no memory tool.
    'claude_code_base64_payload_without_memory_tool': {
        'client': 'claude-code',
        'given': _claude_settings(_decode_command(OTHER_PAYLOAD)),
    },
    # install_hooks.py 264-265: decoding fails (binascii.Error); logged at DEBUG, skipped.
    'claude_code_base64_bad_padding': {
        'client': 'claude-code',
        'given': _claude_settings(_decode_command(BAD_PADDING_PAYLOAD)),
    },
    # install_hooks.py 264-265: the payload is not UTF-8 (UnicodeDecodeError).
    'claude_code_base64_not_utf8': {
        'client': 'claude-code',
        'given': _claude_settings(_decode_command(NOT_UTF8_PAYLOAD)),
    },
    # install_hooks.py 264-265 then 248: a decode error does not end the check; the next
    # hook's legacy memory payload still counts as installed.
    'claude_code_decode_error_then_legacy_memory_hook': {
        'client': 'claude-code',
        'given': _claude_settings(
            _decode_command(BAD_PADDING_PAYLOAD), _decode_command(MEMORY_TOOL_PAYLOAD)
        ),
    },
    # install_hooks.py 329-338: a legacy payload naming a memory tool counts as installed.
    'cursor_legacy_base64_detected': {
        'client': 'cursor',
        'given': _cursor_hooks(_decode_command(MEMORY_TOOL_PAYLOAD)),
    },
    # install_hooks.py 335->325: 'base64' in the command, no b64decode('...') to decode.
    'cursor_base64_no_decode_call': {
        'client': 'cursor',
        'given': _cursor_hooks(NO_DECODE_CALL_COMMAND),
    },
    # install_hooks.py 337->325: the payload decodes, and names no memory tool.
    'cursor_base64_payload_without_memory_tool': {
        'client': 'cursor',
        'given': _cursor_hooks(_decode_command(OTHER_PAYLOAD)),
    },
    # install_hooks.py 339-340: decoding fails (binascii.Error); logged at DEBUG, skipped.
    'cursor_base64_bad_padding': {
        'client': 'cursor',
        'given': _cursor_hooks(_decode_command(BAD_PADDING_PAYLOAD)),
    },
    # install_hooks.py 339-340: the payload is not UTF-8 (UnicodeDecodeError).
    'cursor_base64_not_utf8': {
        'client': 'cursor',
        'given': _cursor_hooks(_decode_command(NOT_UTF8_PAYLOAD)),
    },
    # install_hooks.py 361-362: no "hooks" key; it is added, "version" set, other keys kept.
    'cursor_hooks_file_without_hooks_key_workspace': {
        'client': 'cursor',
        'given': {'.cursor/hooks.json': {'other': 'kept'}},
    },
    # install_hooks.py 361-362, the user scope: the file under HOME.
    'cursor_hooks_file_without_hooks_key_user': {
        'client': 'cursor',
        'scope': 'user',
        'given_home': {'.cursor/hooks.json': {'other': 'kept'}},
    },
}


@pytest.mark.parametrize('case', INSTALL_HOOKS_CASES)
def test_install_hooks_gaps(case, install_env, run_main, golden, normalize):
    spec = INSTALL_HOOKS_CASES[case]
    given_cwd = spec.get('given', {})
    given_home = spec.get('given_home', {})
    _write_files(install_env['cwd'], given_cwd)
    _write_files(install_env['home'], given_home)
    argv = ['install-hooks', '--client', spec['client']]
    if 'scope' in spec:
        argv += ['--scope', spec['scope']]
    # The second run is the idempotency check on what the first run left.
    first_run = run_main(argv)
    second_run = run_main(argv)
    result = {
        'argv': argv,
        'given_cwd': given_cwd,
        'given_home': given_home,
        'first_run': first_run,
        'second_run': second_run,
    }
    golden(f'gaps_install/install_hooks_{case}', _posix_relative_paths(normalize(result)))


# ---------------------------------------------------------------------------
# memory install: Cursor mcp.json without "mcpServers"
# ---------------------------------------------------------------------------


MEMORY_INSTALL_CASES = {
    # installer.py 181-182: "mcpServers" is added and the other keys are kept.
    'cursor_mcp_json_without_mcp_servers': {
        'given_home': {'.cursor/mcp.json': {'other': 'kept'}},
        'answers': [AOSS_URL, '', '', 'y'],
    },
}


@pytest.mark.parametrize('case', MEMORY_INSTALL_CASES)
def test_memory_install_gaps(case, install_env, run_main, golden, normalize):
    spec = MEMORY_INSTALL_CASES[case]
    # Only Cursor is detected: its config directory exists under HOME.
    (install_env['home'] / '.cursor').mkdir()
    _write_files(install_env['home'], spec['given_home'])
    argv = ['memory', 'install']
    result = run_main(argv, answers=spec['answers'])
    # The prompts, with the answers echoed: what a terminal shows.
    result['transcript'] = result.pop('stdout')
    result.update(argv=argv, answers=spec['answers'], given_home=spec['given_home'])
    golden(f'gaps_install/memory_install_{case}', _posix_relative_paths(normalize(result)))
