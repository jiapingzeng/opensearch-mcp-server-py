#!/usr/bin/env python3
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check that every behaviour change is recorded in the behaviour-change ledger.

A behaviour change is a modified or deleted characterization golden. When there is one,
the ledger and CHANGELOG.md must be changed too, and the ledger must name every modified or
deleted golden (its path relative to the golden directory, or its full repo path). An added
golden is new characterization: it needs no name, and a change that only adds goldens passes.
A rename is its old path deleted and its new path added, so the old path must be named.

Usage: python scripts/check_behaviour_ledger.py [--base REF] [--require-entry]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


GOLDEN_DIR = 'tests/characterization/golden/'
LEDGER = 'docs/mcpserver-migration-behaviour-changes.md'
CHANGELOG = 'CHANGELOG.md'


def _named(name: str, text: str) -> bool:
    """Return True when text names the golden name (its path relative to GOLDEN_DIR).

    The ledger may write the relative path or the full repo path (GOLDEN_DIR + name). Either
    must appear as a whole path, not inside a longer one.
    """
    for path in (name, GOLDEN_DIR + name):
        pattern = r'(?<![\w./\\-])' + re.escape(path) + r'(?!\.?[\w/\\-])'
        if re.search(pattern, text) is not None:
            return True
    return False


def check(changes: list[tuple[str, str]], ledger_text: str, require_entry: bool) -> list[str]:
    """Return the problems found; an empty list means the change is recorded.

    Args:
        changes: (status, path) pairs, status 'A', 'M' or 'D', path relative to the repo.
        ledger_text: the ledger's text as it is in the working tree.
        require_entry: the ledger must be changed even when no golden changed.
    """
    changes = [(status, path.replace('\\', '/')) for status, path in changes]
    changed = {path for status, path in changes if status != 'D'}
    goldens = [(status, path) for status, path in changes if path.startswith(GOLDEN_DIR)]
    behaviour_change = any(status in ('M', 'D') for status, _ in goldens)
    problems = []
    if (behaviour_change or require_entry) and LEDGER not in changed:
        problems.append(f'missing: {LEDGER} is not changed')
    if behaviour_change:
        if CHANGELOG not in changed:
            problems.append(f'missing: {CHANGELOG} is not changed')
        changed_goldens = {path for status, path in goldens if status in ('M', 'D')}
        for name in sorted(path[len(GOLDEN_DIR) :] for path in changed_goldens):
            if not _named(name, ledger_text):
                problems.append(f'golden not named in {LEDGER}: {name}')
    return problems


def summary(changes: list[tuple[str, str]], base: str) -> str:
    """Return the success line: how many distinct golden paths changed since base."""
    paths = {path.replace('\\', '/') for _, path in changes}
    goldens = len({path for path in paths if path.startswith(GOLDEN_DIR)})
    if not goldens:
        return f'no characterization goldens changed since {base}'
    return f'{goldens} golden(s) changed since {base}; the behaviour ledger covers them'


def _git(root: Path, *args: str) -> str:
    """Run git with args in root and return its stdout."""
    result = subprocess.run(
        ['git', *args], cwd=root, capture_output=True, text=True, encoding='utf-8', check=True
    )
    return result.stdout


def collect_changes(root: Path, base: str) -> list[tuple[str, str]]:
    """Return (status, path) for every path changed since the merge base of base and HEAD."""
    merge_base = _git(root, 'merge-base', base, 'HEAD').strip()
    fields = _git(root, 'diff', '--name-status', '-M', '-z', merge_base).split('\0')
    changes = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i][0]
        if status in ('R', 'C'):
            old, new = fields[i + 1], fields[i + 2]
            if status == 'R':
                changes.append(('D', old))
            changes.append(('A', new))
            i += 3
            continue
        changes.append((status if status in ('A', 'D') else 'M', fields[i + 1]))
        i += 2
    untracked = _git(root, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')
    changes.extend(('A', path) for path in untracked if path)
    return changes


def main() -> int:
    """Run the check on the current repository and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='agent-work', help='compare with this ref')
    parser.add_argument(
        '--require-entry', action='store_true', help='the ledger must change in any case'
    )
    args = parser.parse_args()
    try:
        root = Path(_git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
        changes = collect_changes(root, args.base)
    except subprocess.CalledProcessError as e:
        print(f'git failed: {" ".join(e.cmd)}: {e.stderr.strip()}', file=sys.stderr)
        return 2
    except OSError as e:
        print(f'git could not run: {e}', file=sys.stderr)
        return 2
    ledger = root / LEDGER
    ledger_text = ledger.read_text(encoding='utf-8') if ledger.is_file() else ''
    problems = check(changes, ledger_text, args.require_entry)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(summary(changes, args.base))
    return 0


if __name__ == '__main__':
    sys.exit(main())
