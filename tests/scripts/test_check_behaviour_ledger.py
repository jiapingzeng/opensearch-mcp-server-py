# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/check_behaviour_ledger.py: the pure check(), no git, no subprocess."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'check_behaviour_ledger.py'
_spec = importlib.util.spec_from_file_location('check_behaviour_ledger', SCRIPT)
ledger_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ledger_check)

check = ledger_check.check
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
    # A rename arrives as its old path deleted and its new path added.
    changes = [
        ('D', GOLDEN + 'area/old_name.json'),
        ('A', GOLDEN + 'area/new_name.json'),
        ('M', LEDGER),
        ('M', CHANGELOG),
    ]
    assert check(changes, '- `area/old_name.json`\n', require_entry=False) == [
        f'golden not named in {LEDGER}: area/new_name.json'
    ]
    text = '- `area/old_name.json`\n- `area/new_name.json`\n'
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
