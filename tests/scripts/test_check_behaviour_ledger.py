# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/check_behaviour_ledger.py: no real git; main() runs with git stubbed."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'check_behaviour_ledger.py'
_spec = importlib.util.spec_from_file_location('check_behaviour_ledger', SCRIPT)
ledger_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ledger_check)

check = ledger_check.check
summary = ledger_check.summary
GOLDEN = 'tests/characterization/golden/'
LEDGER = 'docs/mcpserver-migration-behaviour-changes.md'
CHANGELOG = 'CHANGELOG.md'
NAME = 'call_tool_core/errors/missing_field_echoes_password.json'
LEDGER_TEXT = f'### Fix 22\n\n**Tests and goldens**:\n- `{NAME}`\n'


def test_no_golden_change_passes():
    changes = [('M', 'src/tools/tools.py'), ('A', 'tests/tools/test_new.py')]
    assert check(changes, '', require_entry=False) == []


def test_only_added_goldens_pass_without_ledger_or_changelog():
    # New characterization, not a behaviour change.
    changes = [('A', GOLDEN + 'area/new_case.json'), ('A', 'tests/characterization/test_x.py')]
    assert check(changes, '', require_entry=False) == []


def test_modified_golden_with_ledger_changelog_and_name_passes():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER), ('M', CHANGELOG)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == []


def test_modified_golden_without_ledger_fails():
    changes = [('M', GOLDEN + NAME), ('M', CHANGELOG)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'missing: {LEDGER} is not changed'
    ]


def test_modified_golden_without_changelog_fails():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'missing: {CHANGELOG} is not changed'
    ]


def test_modified_golden_not_named_in_ledger_fails():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER), ('M', CHANGELOG)]
    assert check(changes, '### Fix 22\n', require_entry=False) == [
        f'golden not named in {LEDGER}: {NAME}'
    ]


def test_modified_golden_without_anything_reports_every_problem():
    changes = [('M', GOLDEN + NAME)]
    assert check(changes, '', require_entry=False) == [
        f'missing: {LEDGER} is not changed',
        f'missing: {CHANGELOG} is not changed',
        f'golden not named in {LEDGER}: {NAME}',
    ]


def test_deleted_golden_must_be_named():
    changes = [('D', GOLDEN + 'area/old_case.json'), ('M', LEDGER), ('M', CHANGELOG)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'golden not named in {LEDGER}: area/old_case.json'
    ]
    assert check(changes, '- `area/old_case.json`\n', require_entry=False) == []


def test_deleted_golden_alone_is_a_behaviour_change():
    changes = [('D', GOLDEN + 'area/old_case.json')]
    problems = check(changes, '', require_entry=False)
    assert f'missing: {LEDGER} is not changed' in problems
    assert f'missing: {CHANGELOG} is not changed' in problems


def test_renamed_golden_old_and_new_paths_must_be_named():
    # A rename arrives as its old path deleted and its new path added: both must be named.
    changes = [
        ('D', GOLDEN + 'area/old_name.json'),
        ('A', GOLDEN + 'area/new_name.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    assert check(changes, '- `area/old_name.json`\n', require_entry=False) == [
        f'golden not named in {LEDGER}: area/new_name.json'
    ]
    assert check(changes, '- `area/new_name.json`\n', require_entry=False) == [
        f'golden not named in {LEDGER}: area/old_name.json'
    ]
    text = '- `area/old_name.json`\n- `area/new_name.json`\n'
    assert check(changes, text, require_entry=False) == []


def test_renamed_golden_next_to_a_modified_one_names_both_rename_paths():
    # The change also modifies a golden: the modified golden and both rename paths are named.
    changes = [
        ('M', GOLDEN + NAME),
        ('D', GOLDEN + 'area/old_name.json'),
        ('A', GOLDEN + 'area/new_name.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'golden not named in {LEDGER}: area/new_name.json',
        f'golden not named in {LEDGER}: area/old_name.json',
    ]
    assert check(changes, LEDGER_TEXT + '- `area/old_name.json`\n', require_entry=False) == [
        f'golden not named in {LEDGER}: area/new_name.json'
    ]
    text = LEDGER_TEXT + f'- `area/old_name.json`\n- `{GOLDEN}area/new_name.json`\n'
    assert check(changes, text, require_entry=False) == []


def test_golden_added_next_to_a_modified_one_must_be_named():
    changes = [
        ('M', GOLDEN + NAME),
        ('A', GOLDEN + 'area/new_case.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'golden not named in {LEDGER}: area/new_case.json'
    ]
    text = LEDGER_TEXT + '- `area/new_case.json`\n'
    assert check(changes, text, require_entry=False) == []


def test_golden_added_next_to_a_modified_one_may_be_named_by_its_full_repo_path():
    changes = [
        ('M', GOLDEN + NAME),
        ('A', GOLDEN + 'area/new_case.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    text = LEDGER_TEXT + f'- `{GOLDEN}area/new_case.json`\n'
    assert check(changes, text, require_entry=False) == []


def test_golden_added_next_to_a_deleted_one_must_be_named():
    changes = [
        ('D', GOLDEN + 'area/old_case.json'),
        ('A', GOLDEN + 'other/new_case.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    assert check(changes, '- `area/old_case.json`\n', require_entry=False) == [
        f'golden not named in {LEDGER}: other/new_case.json'
    ]


def test_several_only_added_goldens_pass_without_ledger_or_changelog():
    # However many goldens are added, a change with none modified or deleted passes.
    changes = [
        ('A', GOLDEN + 'area/new_case.json'),
        ('A', GOLDEN + 'other/second_case.json'),
        ('A', 'tests/characterization/test_x.py'),
    ]
    assert check(changes, '', require_entry=False) == []


def test_only_added_goldens_with_require_entry_need_the_ledger_but_no_names():
    changes = [('A', GOLDEN + 'area/new_case.json')]
    assert check(changes, '', require_entry=True) == [f'missing: {LEDGER} is not changed']
    assert check(changes + [('M', LEDGER)], '', require_entry=True) == []


def test_name_inside_a_longer_path_does_not_count():
    changes = [('M', GOLDEN + 'generated/CountTool.json'), ('M', LEDGER), ('M', CHANGELOG)]
    text = '- `gaps_tool_results/generated/CountTool.json`\n- `generated/CountTool.json.bak`\n'
    assert check(changes, text, require_entry=False) == [
        f'golden not named in {LEDGER}: generated/CountTool.json'
    ]
    assert check(changes, 'See generated/CountTool.json.', require_entry=False) == []


def test_deleted_ledger_does_not_count_as_changed():
    changes = [('M', GOLDEN + NAME), ('D', LEDGER), ('M', CHANGELOG)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == [
        f'missing: {LEDGER} is not changed'
    ]


def test_require_entry_without_ledger_fails_even_with_no_golden_change():
    changes = [('M', 'src/mcp_server_opensearch/server.py')]
    assert check(changes, '', require_entry=True) == [f'missing: {LEDGER} is not changed']


def test_require_entry_with_ledger_passes_without_changelog():
    # Without a golden change, --require-entry asks for the ledger only.
    changes = [('M', 'src/mcp_server_opensearch/server.py'), ('M', LEDGER)]
    assert check(changes, '', require_entry=True) == []


def test_require_entry_still_checks_a_golden_change():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER)]
    assert check(changes, '', require_entry=True) == [
        f'missing: {CHANGELOG} is not changed',
        f'golden not named in {LEDGER}: {NAME}',
    ]


def test_windows_separators_are_normalized():
    win_golden = (GOLDEN + NAME).replace('/', '\\')
    win_ledger = LEDGER.replace('/', '\\')
    changes = [('M', win_golden), ('M', win_ledger), ('M', CHANGELOG)]
    assert check(changes, LEDGER_TEXT, require_entry=False) == []
    assert check([('M', win_golden), ('M', win_ledger), ('M', CHANGELOG)], '', False) == [
        f'golden not named in {LEDGER}: {NAME}'
    ]


def test_windows_separators_only_added_golden_passes():
    changes = [('A', (GOLDEN + 'area/new_case.json').replace('/', '\\'))]
    assert check(changes, '', require_entry=False) == []


# Step 2: a golden written as its full repo path counts as named, like its relative path.


def test_golden_named_by_its_full_repo_path_passes():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER), ('M', CHANGELOG)]
    assert check(changes, f'- `{GOLDEN}{NAME}`\n', require_entry=False) == []


def test_golden_named_by_its_relative_path_passes():
    changes = [('M', GOLDEN + NAME), ('M', LEDGER), ('M', CHANGELOG)]
    assert check(changes, f'- `{NAME}`\n', require_entry=False) == []


def test_full_repo_path_inside_a_longer_path_does_not_count():
    changes = [('M', GOLDEN + 'a/b.json'), ('M', LEDGER), ('M', CHANGELOG)]
    expected = [f'golden not named in {LEDGER}: a/b.json']
    assert check(changes, '- `x/a/b.json`\n', require_entry=False) == expected
    assert check(changes, f'- `x/{GOLDEN}a/b.json`\n', require_entry=False) == expected
    assert check(changes, f'- `{GOLDEN}a/b.json.bak`\n', require_entry=False) == expected
    assert check(changes, f'- `{GOLDEN}x/a/b.json`\n', require_entry=False) == expected
    assert check(changes, f'See {GOLDEN}a/b.json.', require_entry=False) == []


# Step 3: a missing git executable exits 2 with one line, not a traceback.


def test_missing_git_exits_2_with_one_line(monkeypatch, capsys):
    def no_git(*args, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory', 'git')

    monkeypatch.setattr(ledger_check.subprocess, 'run', no_git)
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py'])
    assert ledger_check.main() == 2
    out, err = capsys.readouterr()
    assert out == ''
    assert len(err.strip().splitlines()) == 1
    assert err.startswith('git could not run: ')
    assert 'Traceback' not in err


def test_git_os_error_exits_2_with_one_line(monkeypatch, capsys):
    def no_permission(*args, **kwargs):
        raise PermissionError(13, 'Permission denied', 'git')

    monkeypatch.setattr(ledger_check.subprocess, 'run', no_permission)
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py'])
    assert ledger_check.main() == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_os_error_outside_git_is_not_reported_as_git(monkeypatch, capsys):
    # Only the git subprocess calls are covered: an OSError from Path.cwd() propagates.
    def no_cwd():
        raise FileNotFoundError(2, 'No such file or directory')

    def git_must_not_run(*args, **kwargs):
        raise AssertionError('git ran')

    monkeypatch.setattr(ledger_check.Path, 'cwd', staticmethod(no_cwd))
    monkeypatch.setattr(ledger_check.subprocess, 'run', git_must_not_run)
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py'])
    with pytest.raises(FileNotFoundError):
        ledger_check.main()
    assert 'git could not run' not in capsys.readouterr().err


def test_os_error_reading_the_ledger_is_not_reported_as_git(monkeypatch, capsys, tmp_path):
    # The ledger path is a directory, so reading it raises an OSError after git has run.
    (tmp_path / LEDGER).mkdir(parents=True)
    monkeypatch.setattr(ledger_check, '_git', lambda root, *args: f'{tmp_path}\n')
    monkeypatch.setattr(ledger_check, 'collect_changes', lambda root, base: [])
    ledger = tmp_path / LEDGER
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda self: self == ledger or is_file(self))
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py'])
    with pytest.raises(OSError):
        ledger_check.main()
    assert 'git could not run' not in capsys.readouterr().err


def test_git_error_still_exits_2_with_one_line(monkeypatch, capsys):
    def git_fails(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd, stderr='fatal: not a git repository\n')

    monkeypatch.setattr(ledger_check.subprocess, 'run', git_fails)
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py'])
    assert ledger_check.main() == 2
    err = capsys.readouterr().err
    assert err == 'git failed: git rev-parse --show-toplevel: fatal: not a git repository\n'


# Step 4: the success line counts distinct golden paths and says plainly when there are none.


def test_summary_with_no_golden_changed_says_so():
    changes = [('M', 'src/tools/tools.py'), ('M', LEDGER)]
    assert summary(changes, 'agent-work') == 'no characterization goldens changed since agent-work'
    assert summary([], 'main') == 'no characterization goldens changed since main'


def test_summary_with_only_added_goldens_does_not_claim_the_ledger_covers_them():
    # Only-added goldens are not checked against the ledger, so the line must not say so.
    changes = [('A', GOLDEN + 'area/new_case.json'), ('A', GOLDEN + 'other/second.json')]
    assert summary(changes, 'b') == (
        '2 golden(s) added since b and none modified or deleted; '
        'new characterization is not a behaviour change'
    )


def test_summary_counts_added_goldens_next_to_a_modified_one():
    # With the rule restored, the added golden is named too, so the ledger covers both.
    changes = [('M', GOLDEN + NAME), ('A', GOLDEN + 'area/new_case.json'), ('M', LEDGER)]
    assert summary(changes, 'b') == '2 golden(s) changed since b; the behaviour ledger covers them'


def test_summary_counts_a_rename_as_two_distinct_paths():
    changes = [('D', GOLDEN + 'area/old_name.json'), ('A', GOLDEN + 'area/new_name.json')]
    assert summary(changes, 'agent-work') == (
        '2 golden(s) changed since agent-work; the behaviour ledger covers them'
    )


def test_summary_counts_each_golden_path_once():
    # The same path in two records (deleted from the index, still on disk as untracked).
    changes = [
        ('D', GOLDEN + NAME),
        ('A', GOLDEN + NAME),
        ('M', (GOLDEN + 'area/case.json').replace('/', '\\')),
        ('M', GOLDEN + 'area/case.json'),
        ('M', CHANGELOG),
    ]
    assert summary(changes, 'b') == '2 golden(s) changed since b; the behaviour ledger covers them'


def test_main_prints_the_summary_line(monkeypatch, capsys, tmp_path):
    changes = [('M', GOLDEN + NAME), ('M', LEDGER), ('M', CHANGELOG)]
    (tmp_path / 'docs').mkdir()
    (tmp_path / LEDGER).write_text(LEDGER_TEXT, encoding='utf-8')
    monkeypatch.setattr(ledger_check, '_git', lambda root, *args: f'{tmp_path}\n')
    monkeypatch.setattr(ledger_check, 'collect_changes', lambda root, base: changes)
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py', '--base', 'b'])
    assert ledger_check.main() == 0
    assert capsys.readouterr().out == (
        '1 golden(s) changed since b; the behaviour ledger covers them\n'
    )


def test_main_with_no_golden_changed_says_so(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ledger_check, '_git', lambda root, *args: f'{tmp_path}\n')
    monkeypatch.setattr(ledger_check, 'collect_changes', lambda root, base: [('M', 'README.md')])
    monkeypatch.setattr(sys, 'argv', ['check_behaviour_ledger.py', '--base', 'b'])
    assert ledger_check.main() == 0
    assert capsys.readouterr().out == 'no characterization goldens changed since b\n'
