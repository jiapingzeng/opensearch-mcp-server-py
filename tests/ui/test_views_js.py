# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runs the node --test suite for the MCP Apps views (tests/ui/views.test.js).

Skipped when node is not on PATH or jsdom is not installed. CI runs it by installing the
test-only dependency before pytest: `npm ci --prefix tests/ui`.
"""

import pytest
import re
import shutil
import subprocess
from pathlib import Path
from tools.ui_resources import RESULTS_VIEW_URI, TABLE_VIEW_URI, UI_RESOURCES


HERE = Path(__file__).parent
SUITE = HERE / 'views.test.js'
NODE = shutil.which('node')


def test_suite_assembles_views_like_the_server():
    text = SUITE.read_text(encoding='utf-8')
    for name, uri in (('results', RESULTS_VIEW_URI), ('table', TABLE_VIEW_URI)):
        match = re.search(rf"{name}: \{{ title: '([^']+)', scripts: \[([^\]]+)\]", text)
        assert match, f'views.test.js does not define the {name} view'
        assert match.group(1) == UI_RESOURCES[uri]['title']
        assert re.findall(r"'([^']+)'", match.group(2)) == UI_RESOURCES[uri]['scripts']


@pytest.mark.skipif(NODE is None, reason='node is not on PATH')
def test_view_scripts_under_node():
    if not (HERE / 'node_modules' / 'jsdom').is_dir():
        pytest.skip('jsdom is not installed; run `npm ci --prefix tests/ui`')
    proc = subprocess.run(
        [NODE, '--test', SUITE.name],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
