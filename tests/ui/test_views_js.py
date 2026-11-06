# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runs the node --test suite for the MCP Apps views (tests/ui/views.test.js).

Needs node on PATH and jsdom installed (`npm ci --prefix tests/ui`). Locally it is skipped
when either is missing. CI installs both on Linux and macOS, so there a missing node or
jsdom fails the test instead; Windows CI does not install them and skips it.
"""

import os
import pytest
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tools.ui_resources import RESULTS_VIEW_URI, TABLE_VIEW_URI, UI_RESOURCES


HERE = Path(__file__).parent
SUITE = HERE / 'views.test.js'
NODE = shutil.which('node')
REQUIRED = bool(os.environ.get('CI')) and sys.platform != 'win32'


def _missing(reason):
    if REQUIRED:
        pytest.fail(f'{reason}; CI must run the view suite')
    pytest.skip(reason)


def test_suite_assembles_views_like_the_server():
    text = SUITE.read_text(encoding='utf-8')
    for name, uri in (('results', RESULTS_VIEW_URI), ('table', TABLE_VIEW_URI)):
        match = re.search(rf"{name}: \{{ title: '([^']+)', scripts: \[([^\]]+)\]", text)
        assert match, f'views.test.js does not define the {name} view'
        assert match.group(1) == UI_RESOURCES[uri]['title']
        assert re.findall(r"'([^']+)'", match.group(2)) == UI_RESOURCES[uri]['scripts']


def test_view_scripts_under_node():
    if NODE is None:
        _missing('node is not on PATH')
    if not (HERE / 'node_modules' / 'jsdom').is_dir():
        _missing('jsdom is not installed; run `npm ci --prefix tests/ui`')
    proc = subprocess.run(
        [NODE, '--test', SUITE.name],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
