# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the characterization harness itself (see conftest.py and README.md)."""

import asyncio
import contextvars
import copy
import importlib
import json
import logging
import os
import pytest
import re
import socket
import sys
import tools as tools_package
import yaml
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from mcp_server_opensearch import clusters_information, global_state
from mcp_server_opensearch.client_context import client_name_var, request_context_var
from mcp_server_opensearch.clusters_information import ClusterInfo, add_cluster
from mcp_server_opensearch.logging_config import configure_logging
from pathlib import Path
from tools import config as tool_config
from tools import tool_filter
from tools.tool_params import ListIndicesArgs
from tools.tools import TOOL_REGISTRY


FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'

# --- the server through the MCP protocol -----------------------------------


async def test_list_tools_default(mcp_session, fake_opensearch, golden, normalize):
    async with mcp_session() as session:
        golden('harness/list_tools_default', normalize(await session.list_tools()))
    # Defaults set only OPENSEARCH_URL: with no auth configured the server takes the AWS
    # credentials path, fails it (no region), never sends GET / and so sees no version, which
    # turns version filtering off. Reaching the fake needs auth, e.g. OPENSEARCH_NO_AUTH=true.
    assert fake_opensearch.requests == []


async def test_raw_ping(mcp_session, golden):
    async with mcp_session() as session:
        golden('harness/raw_ping', await session.raw('ping'))
        # ping no longer exists at 2026-07-28: pins the per-protocol-version path of raw().
        golden(
            'harness/raw_ping_2026_07_28', await session.raw('ping', protocol_version='2026-07-28')
        )


# --- golden ------------------------------------------------------------------


def test_golden_existing_file_is_never_overwritten(golden, tmp_path, monkeypatch):
    golden.directory = tmp_path
    monkeypatch.setenv('CHAR_GOLDEN_WRITE', '1')
    golden('probe/existing', {'b': 1, 'a': 'é'})
    path = tmp_path / 'probe' / 'existing.json'
    original = path.read_bytes()
    # Canonical form: sorted keys, 2-space indent, non-ASCII kept, trailing newline; UTF-8
    # with LF line endings on every OS.
    assert original == '{\n  "a": "é",\n  "b": 1\n}\n'.encode('utf-8')

    with pytest.raises(pytest.fail.Exception) as excinfo:
        golden('probe/existing', {'b': 2, 'a': 'é'})

    assert '-  "b": 1' in str(excinfo.value)
    assert '+  "b": 2' in str(excinfo.value)
    assert path.read_bytes() == original


def test_golden_crlf_checkout_fails_with_line_ending_hint(golden, tmp_path):
    # What a checkout with core.autocrlf=true would write without the directory's .gitattributes.
    golden.directory = tmp_path
    path = tmp_path / 'probe' / 'crlf.json'
    path.parent.mkdir()
    path.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')

    with pytest.raises(pytest.fail.Exception, match='CRLF line endings'):
        golden('probe/crlf', {'a': 1})


def test_golden_files_are_lf_only_utf8(golden):
    # Goldens must compare the same on Linux, macOS and Windows.
    files = sorted(golden.directory.rglob('*.json'))
    assert files
    for path in files:
        content = path.read_bytes()
        assert b'\r' not in content, path
        assert content.decode('utf-8').endswith('}\n'), path


@pytest.mark.parametrize('name', ['/abs', 'a/../b', 'a/b.json', 'a\\b', 'C:/a'])
def test_golden_rejects_names_that_differ_between_os(golden, name):
    with pytest.raises(ValueError, match='invalid golden name'):
        golden.path(name)


def test_golden_missing_file_fails_without_write_env(golden, tmp_path, monkeypatch):
    golden.directory = tmp_path
    monkeypatch.delenv('CHAR_GOLDEN_WRITE', raising=False)

    with pytest.raises(pytest.fail.Exception, match='CHAR_GOLDEN_WRITE=1'):
        golden('probe/missing', {'value': 1})

    assert not (tmp_path / 'probe' / 'missing.json').exists()


# --- normalize ---------------------------------------------------------------


def test_normalize_replaces_only_volatile_values(normalize, tmp_path):
    now = datetime.now(timezone.utc)
    event = {
        'event_type': 'tool_execution',
        'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'message': 'Tool executed: ListIndexTool (12.34ms)',
        'duration_ms': 12.34,
    }
    data = {
        'event': event,
        'log': f'{now:%Y-%m-%d %H:%M:%S},123 - x - INFO - Memory snapshot: 98.5 MB (pid=4242)',
        'headers': {'Mcp-Session-Id': 'anything', 'date': f'{now:%a, %d %b %Y %H:%M:%S} GMT'},
        'ids': ['0b5a0f2e-3c1d-4e5f-9a8b-7c6d5e4f3a2b', '0b5a0f2e3c1d4e5f9a8b7c6d5e4f3a2b'],
        'url': 'http://127.0.0.1:54321/mcp',
        'config': str(tmp_path / 'config.yml'),
        'now_ms': int(now.timestamp() * 1000),
        'schema': {'required': ['index', 'body', 'id']},
        # Static values that merely look volatile stay as they are.
        'static': {
            'example_date': '2024-01-01T00:00:00Z',
            'example_url': 'https://localhost:9200',
            'pid': 7,
            'took_ms': '(5 ms)',
            'size': 1024,
        },
    }

    assert normalize(data) == {
        'event': {
            'event_type': 'tool_execution',
            'timestamp': '<TIMESTAMP>',
            'message': 'Tool executed: ListIndexTool (<DURATION>ms)',
            'duration_ms': '<DURATION>',
        },
        'log': '<TIMESTAMP> - x - INFO - Memory snapshot: <MEMORY> MB (pid=<PID>)',
        'headers': {'Mcp-Session-Id': '<SESSION_ID>', 'date': '<TIMESTAMP>'},
        'ids': ['<UUID>', '<SESSION_ID>'],
        'url': 'http://127.0.0.1:<PORT>/mcp',
        'config': '<TMP>/config.yml',
        'now_ms': '<TIMESTAMP>',
        'schema': {'required': ['body', 'id', 'index']},
        'static': data['static'],
    }
    assert data['schema']['required'] == ['index', 'body', 'id']


def test_normalize_windows_temp_paths(normalize):
    # What tmp_path and tempfile give on a Windows runner: drive letter, backslashes, the
    # system temp directory under its 8.3 short name and pytest's base temp resolved.
    basetemp = r'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0'
    normalizer = type(normalize)(basetemp, temp_dirs=[r'C:\Users\RUNNER~1\AppData\Local\Temp'])
    test_dir = basetemp + r'\test_windows_temp_paths0'
    data = {
        'config': test_dir + r'\config.yml',
        'nested': test_dir + r'\sub\dir\config.yml',
        'message': f'cannot read {test_dir}\\config.yml: not found',
        'json_escaped': json.dumps({'path': test_dir + r'\config.yml'}),
        'forward_slashes': test_dir.replace('\\', '/') + '/config.yml',
        'lower_case_drive': 'c' + test_dir[1:] + r'\config.yml',
        'test_dir': test_dir,
        'mkstemp': r'C:\Users\RUNNER~1\AppData\Local\Temp\tmpab12_xyz\out.json',
        # Not under a temp directory: left alone, backslashes included.
        'other': r'C:\Program Files\OpenSearch\config.yml',
    }

    assert normalizer(data) == {
        'config': '<TMP>/config.yml',
        'nested': '<TMP>/sub/dir/config.yml',
        'message': 'cannot read <TMP>/config.yml: not found',
        'json_escaped': '{"path": "<TMP>/config.yml"}',
        'forward_slashes': '<TMP>/config.yml',
        'lower_case_drive': '<TMP>/config.yml',
        'test_dir': '<TMP>',
        'mkstemp': '<TMP>/out.json',
        'other': r'C:\Program Files\OpenSearch\config.yml',
    }


def test_normalize_posix_temp_paths_keep_backslashes(normalize):
    # POSIX paths only split on "/": a backslash after one is text (e.g. an escape), kept.
    normalizer = type(normalize)('/tmp/pytest-of-ci/pytest-3', temp_dirs=['/var/folders/x/T'])
    assert normalizer(
        {
            'config': '/tmp/pytest-of-ci/pytest-3/test_a0/config.yml\\n',
            'mkstemp': '/var/folders/x/T/tmpab12_xyz/out.json',
        }
    ) == {'config': '<TMP>/config.yml\\n', 'mkstemp': '<TMP>/out.json'}


# Epoch ints only under timestamp keys (spec step 1). 1795000000 is 2026-11-18 11:06:40 UTC:
# the tests pin the normalizer's clock there, so they test the same thing on any day.
EPOCH = 1795000000
EPOCH_MOMENT = datetime.fromtimestamp(EPOCH, timezone.utc).replace(tzinfo=None)
FAR_EPOCH = 1577836800  # 2020-01-01T00:00:00Z

# Keys that name a time: as listed, containing "time"/"date", ending in _at/_ms/_ts/_epoch,
# in any case.
TIMESTAMP_KEY_EXAMPLES = [
    'now',
    'time',
    'date',
    'timestamp',
    'epoch',
    'now_ms',
    'created_at',
    'start_time',
    'last_updated_time',
    'event_ts',
    'start_epoch',
    'update_date',
    'startTime',
    'lastModifiedDate',
    'Timestamp',
    'NOW_MS',
    'Created_At',
]
NON_TIMESTAMP_KEY_EXAMPLES = ['size', 'count', 'bytes', 'doc_count', 'store_size', 'took', 'id']


def _normalizer_at(normalize, moment: datetime):
    """A normalizer (no temp rules) whose clock reads ``moment`` (naive UTC)."""

    class PinnedClock(type(normalize)):
        @staticmethod
        def _clock() -> datetime:
            return moment

    return PinnedClock(temp_dirs=[])


def test_normalize_epoch_int_kept_under_size_replaced_under_now_ms(normalize):
    # Step 5: 1795000000 reads as now in epoch seconds; as a byte count it is kept.
    normalizer = _normalizer_at(normalize, EPOCH_MOMENT)
    assert normalizer({'size': EPOCH, 'now_ms': EPOCH * 1000}) == {
        'size': EPOCH,
        'now_ms': '<TIMESTAMP>',
    }

    # The same on the real clock: today's epoch in seconds and milliseconds.
    now = datetime.now(timezone.utc)
    seconds, millis = int(now.timestamp()), int(now.timestamp() * 1000)
    assert normalize({'size': seconds, 'bytes': millis, 'now_ms': millis}) == {
        'size': seconds,
        'bytes': millis,
        'now_ms': '<TIMESTAMP>',
    }


@pytest.mark.parametrize('key', TIMESTAMP_KEY_EXAMPLES)
def test_normalize_epoch_int_under_timestamp_key_replaced(normalize, key):
    # Step 1 / step 5: each key that names a time, as epoch seconds and as milliseconds.
    normalizer = _normalizer_at(normalize, EPOCH_MOMENT)
    assert type(normalize).is_timestamp_key(key)
    assert normalizer({key: EPOCH}) == {key: '<TIMESTAMP>'}
    assert normalizer({'nested': {key: EPOCH * 1000}}) == {'nested': {key: '<TIMESTAMP>'}}
    # Far from now, an int under a timestamp key is kept like any other value.
    assert normalizer({key: FAR_EPOCH, 'x': {key: FAR_EPOCH * 1000}}) == {
        key: FAR_EPOCH,
        'x': {key: FAR_EPOCH * 1000},
    }


@pytest.mark.parametrize('key', NON_TIMESTAMP_KEY_EXAMPLES)
def test_normalize_epoch_int_under_other_key_kept(normalize, key):
    # Step 1 / step 5: sizes, counts and ids keep their value, however close to now it reads.
    normalizer = _normalizer_at(normalize, EPOCH_MOMENT)
    assert not type(normalize).is_timestamp_key(key)
    data = {key: EPOCH, 'nested': {key: EPOCH * 1000}}
    assert normalizer(data) == data


def test_normalize_epoch_int_without_key_kept_strings_unchanged(normalize):
    # Step 1: list items and bare ints have no key, so they stay; bools are never timestamps.
    # The ISO 8601 and HTTP date rules for strings apply under any key, as before.
    normalizer = _normalizer_at(normalize, EPOCH_MOMENT)
    data = {
        'now_ms': [EPOCH * 1000, EPOCH],
        'values': [EPOCH, {'size': EPOCH}],
        'timestamp': True,
    }
    assert normalizer(data) == data
    assert normalizer([EPOCH]) == [EPOCH]
    assert normalizer(EPOCH) == EPOCH
    assert normalizer(
        {
            'size': f'{EPOCH_MOMENT:%Y-%m-%dT%H:%M:%S}Z',
            'message': f'at {format_datetime(EPOCH_MOMENT.replace(tzinfo=timezone.utc), usegmt=True)}',
            'old': '2020-01-01T00:00:00Z',
        }
    ) == {'size': '<TIMESTAMP>', 'message': 'at <TIMESTAMP>', 'old': '2020-01-01T00:00:00Z'}


def test_normalize_posix_temp_base_ends_at_path_boundary(normalize):
    # Step 2: base ".../pytest-3" must not match the start of ".../pytest-30" ("<TMP>0/...").
    normalizer = type(normalize)('/tmp/pytest-of-u/pytest-3', temp_dirs=['/tmp'])
    data = {
        'longer': '/tmp/pytest-of-u/pytest-30/test_a0/config.yml',
        'longer_alone': '/tmp/pytest-of-u/pytest-30',
        'longer_tempfile': '/tmp/tmpab12_xyz9/out.json',
        'base': '/tmp/pytest-of-u/pytest-3/test_a0/config.yml',
        'base_alone': '/tmp/pytest-of-u/pytest-3',
        'base_spaced': 'in /tmp/pytest-of-u/pytest-3 now',
        'base_quoted': "open('/tmp/pytest-of-u/pytest-3')",
        'base_json': json.dumps({'path': '/tmp/pytest-of-u/pytest-3'}),
        'tempfile': '/tmp/tmpab12_xyz/out.json',
    }

    assert normalizer(data) == {
        'longer': data['longer'],
        'longer_alone': data['longer_alone'],
        'longer_tempfile': data['longer_tempfile'],
        'base': '<TMP>/config.yml',
        'base_alone': '<TMP>',
        'base_spaced': 'in <TMP> now',
        'base_quoted': "open('<TMP>')",
        'base_json': '{"path": "<TMP>"}',
        'tempfile': '<TMP>/out.json',
    }


def test_normalize_windows_temp_base_ends_at_path_boundary(normalize):
    # Step 2 on Windows, where "\" or "/" separates: "pytest-3" is not the start of "pytest-30".
    base = r'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-3'
    temp = r'C:\Users\RUNNER~1\AppData\Local\Temp'
    normalizer = type(normalize)(base, temp_dirs=[temp])
    longer = base + '0'
    data = {
        'longer': longer + r'\test_a0\config.yml',
        'longer_slashes': longer.replace('\\', '/') + '/test_a0/config.yml',
        'longer_alone': longer,
        'longer_tempfile': temp + r'\tmpab12_xyz9\out.json',
        'base': base + r'\test_a0\config.yml',
        'base_slashes': base.replace('\\', '/') + '/test_a0/config.yml',
        'base_alone': base,
        'base_spaced': f'in {base} now',
        'base_json': json.dumps({'path': base}),
        'tempfile': temp + r'\tmpab12_xyz\out.json',
    }

    assert normalizer(data) == {
        'longer': data['longer'],
        'longer_slashes': data['longer_slashes'],
        'longer_alone': data['longer_alone'],
        'longer_tempfile': data['longer_tempfile'],
        'base': '<TMP>/config.yml',
        'base_slashes': '<TMP>/config.yml',
        'base_alone': '<TMP>',
        'base_spaced': 'in <TMP> now',
        'base_json': '{"path": "<TMP>"}',
        'tempfile': '<TMP>/out.json',
    }


# --- no_network --------------------------------------------------------------


def test_no_network_lets_loopback_socketpair_through(no_network, monkeypatch):
    # Windows' asyncio event loop makes its self-pipe with this fallback, which connects to
    # 127.0.0.1; pytest-asyncio creates the loop after the guard is in place.
    first, second = socket._fallback_socketpair()
    with first, second:
        first.sendall(b'ping')
        assert second.recv(4) == b'ping'

    # The Windows path end to end: socket.socketpair *is* the fallback there.
    monkeypatch.setattr(socket, 'socketpair', socket._fallback_socketpair)
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result='ran')) == 'ran'
    finally:
        loop.close()
    assert no_network == []


def test_no_network_blocks_lookups_and_connects_off_this_machine(no_network):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match='network access is disabled'):
            sock.connect(('192.0.2.1', 9200))  # TEST-NET-1: documentation only, never routed.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match='network access is disabled'):
            sock.connect_ex(('192.0.2.1', 9200))
    with pytest.raises(OSError, match='network access is disabled'):
        socket.getaddrinfo('localhost', 9200)

    assert len(no_network) == 3
    # Consumed here; left in place, the guard's teardown would fail this test.
    no_network.clear()


# --- fake_opensearch / pinned_openapi ---------------------------------------


async def test_unmatched_fake_request_fails(mcp_session, fake_opensearch):
    async with mcp_session(env={'OPENSEARCH_NO_AUTH': 'true'}) as session:
        result = await session.call_tool('ListIndexTool', {})

    assert result['isError'] is True
    assert 'unmatched fake OpenSearch request: GET /_cat/indices' in result['content'][0]['text']
    assert {(r['method'], r['path']) for r in fake_opensearch.unmatched} == {
        ('GET', '/_cat/indices')
    }
    with pytest.raises(pytest.fail.Exception, match='no route'):
        fake_opensearch.verify()
    # Consumed here; left in place, the fixture's teardown would fail this test.
    fake_opensearch.unmatched.clear()


async def test_fake_answers_version_probe_and_routes(mcp_session, fake_opensearch):
    fake_opensearch.version = '2.19.0'
    fake_opensearch.reply('GET', '/_cat/indices', [{'index': 'logs', 'health': 'green'}])
    env = {'OPENSEARCH_USERNAME': 'admin', 'OPENSEARCH_PASSWORD': 's3cret'}
    async with mcp_session(env=env) as session:
        result = await session.call_tool('ListIndexTool', {'include_detail': False})

    assert result['isError'] is False
    # Startup probes the version for tool filtering; the tool probes it again before its call.
    assert [(r['method'], r['path'], r['params']) for r in fake_opensearch.requests] == [
        ('GET', '/', {}),
        ('GET', '/', {}),
        ('GET', '/_cat/indices', {'format': 'json'}),
    ]
    assert fake_opensearch.requests[0]['auth'] == {'kind': 'basic', 'username': 'admin'}
    assert 's3cret' not in repr(fake_opensearch.requests)


async def test_version_probe_failure_and_openapi_unavailable(
    mcp_session, fake_opensearch, openapi_unavailable
):
    fake_opensearch.version = None
    async with mcp_session(env={'OPENSEARCH_NO_AUTH': 'true'}) as session:
        names = [tool['name'] for tool in (await session.list_tools())['tools']]

    assert [(r['method'], r['path']) for r in fake_opensearch.requests] == [('GET', '/')]
    assert openapi_unavailable.calls == ['cluster.yaml']
    assert 'MsearchTool' not in names
    assert 'ListIndexTool' in names


# --- isolated_state ----------------------------------------------------------


async def test_isolated_state_restores_registry_and_schemas(isolated_state):
    from mcp_server_opensearch.streaming_server import create_mcp_server
    from tools.tool_filter import get_tools

    entries = dict(TOOL_REGISTRY)
    schemas = copy.deepcopy({name: info['input_schema'] for name, info in entries.items()})
    index_description = ListIndicesArgs.model_fields['index'].description
    # The same snapshot the fixture takes before, and restores after, every test.
    snapshot = isolated_state.snapshot()

    # Multi mode strips connection fields from the registry's own schemas, in place.
    global_state.set_mode('multi')
    await get_tools(tool_registry=TOOL_REGISTRY)
    # A server build replaces entries, adds generated tools, edits the args model's field
    # description on the class and strips schemas in place again (single mode).
    add_cluster('leftover', ClusterInfo(opensearch_url='http://other.invalid:9200'))
    await create_mcp_server(
        mode='single',
        profile='leaky-profile',
        cli_tool_overrides={'tool.ListIndexTool.args.index.description': 'patched'},
    )
    # Set directly: the server only sets it to an id when a container id is configured,
    # and this build configures none.
    tool_config._resolved_memory_container_id = 'leaky-container'

    assert 'opensearch_url' not in entries['ListIndexTool']['input_schema']['properties']
    assert TOOL_REGISTRY['ListIndexTool'] is not entries['ListIndexTool']
    assert 'MsearchTool' in TOOL_REGISTRY
    assert ListIndicesArgs.model_fields['index'].description == 'patched'
    assert global_state._current_profile == 'leaky-profile'
    assert tool_filter._resolved_allow_write_setting is True
    assert clusters_information.cluster_registry

    snapshot.restore()

    assert list(TOOL_REGISTRY) == list(entries)
    assert all(TOOL_REGISTRY[name] is entry for name, entry in entries.items())
    assert {name: info['input_schema'] for name, info in TOOL_REGISTRY.items()} == schemas
    assert ListIndicesArgs.model_fields['index'].description == index_description
    assert global_state._current_mode is None
    assert global_state._current_profile is None
    assert tool_filter._resolved_allow_write_setting is None
    assert tool_config._resolved_memory_container_id is None
    assert clusters_information.cluster_registry == {}


def _own_handlers(logger: logging.Logger) -> list:
    # pytest installs fresh capture handlers for every test phase; the harness leaves those alone.
    return [h for h in logger.handlers if not type(h).__module__.startswith('_pytest')]


def test_isolated_state_resets_before_and_restores_after(isolated_state, fake_opensearch):
    root = logging.getLogger()
    leaked_handler = logging.NullHandler()
    with pytest.MonkeyPatch.context() as outer:
        # State an earlier, non-characterization test could leave behind.
        outer.setenv('OPENSEARCH_URL', 'http://leaked.invalid:9200')
        outer.setenv('MEMORY_TOOLS_ENABLED', 'true')
        outer.setenv('AWS_REGION', 'leaked-region-1')
        outer.setattr(tool_config, '_resolved_memory_container_id', 'leaked-container')
        global_state.set_mode('multi')
        add_cluster('leaked', ClusterInfo(opensearch_url='http://leaked.invalid:9200'))
        leaked_token = client_name_var.set('leaked-client')
        root.addHandler(leaked_handler)
        handlers, level = _own_handlers(root), root.level
        entries = dict(TOOL_REGISTRY)
        index_description = ListIndicesArgs.model_fields['index'].description
        try:
            # The fixture's own setup and teardown, around a stand-in test body.
            with (
                pytest.MonkeyPatch.context() as inner,
                isolated_state.isolation(inner, fake_opensearch.url),
            ):
                # Before the test body: import-time state and a clean environment.
                assert global_state._current_mode is None
                assert tool_config._resolved_memory_container_id is None
                assert clusters_information.cluster_registry == {}
                assert client_name_var.get() == 'unknown'
                assert os.environ['OPENSEARCH_URL'] == fake_opensearch.url
                assert 'MEMORY_TOOLS_ENABLED' not in os.environ
                assert 'AWS_REGION' not in os.environ

                # The test body changes everything the server can change.
                inner.setenv('OPENSEARCH_NO_AUTH', 'true')
                configure_logging(level=logging.DEBUG)
                client_name_var.set('inner-client')
                global_state.set_mode('single')
                global_state.set_profile('inner-profile')
                tool_filter.set_allow_write_setting(False)
                tool_config._resolved_memory_container_id = 'inner-container'
                add_cluster('inner', ClusterInfo(opensearch_url='http://inner.invalid:9200'))
                TOOL_REGISTRY['InnerTool'] = {'display_name': 'InnerTool'}
                TOOL_REGISTRY['ListIndexTool']['input_schema']['properties'].pop('index')
                ListIndicesArgs.model_fields['index'].description = 'inner'

            # After: back to what the earlier test left, not to import-time state.
            assert os.environ['OPENSEARCH_URL'] == 'http://leaked.invalid:9200'
            assert os.environ['MEMORY_TOOLS_ENABLED'] == 'true'
            assert os.environ['AWS_REGION'] == 'leaked-region-1'
            assert 'OPENSEARCH_NO_AUTH' not in os.environ
            assert global_state._current_mode == 'multi'
            assert global_state._current_profile is None
            assert tool_filter._resolved_allow_write_setting is None
            assert tool_config._resolved_memory_container_id == 'leaked-container'
            assert list(clusters_information.cluster_registry) == ['leaked']
            assert client_name_var.get() == 'leaked-client'
            assert _own_handlers(root) == handlers
            assert root.level == level
            assert list(TOOL_REGISTRY) == list(entries)
            assert all(TOOL_REGISTRY[name] is entry for name, entry in entries.items())
            assert 'index' in TOOL_REGISTRY['ListIndexTool']['input_schema']['properties']
            assert ListIndicesArgs.model_fields['index'].description == index_description
        finally:
            client_name_var.reset(leaked_token)
            root.removeHandler(leaked_handler)


def test_reset_fails_for_a_token_from_another_context():
    # The failure the isolation fallback handles (step 3), as the next two tests simulate it.
    token = contextvars.copy_context().run(client_name_var.set, 'elsewhere')
    with pytest.raises(ValueError):
        client_name_var.reset(token)


def test_isolation_fallback_restores_value_from_before_the_test(isolated_state, fake_opensearch):
    # Step 3: when var.reset(token) fails, the variable gets back the value it had before the
    # test (token.old_value), not its default. Setup runs in a copy of this Context, so the
    # tokens it makes cannot be reset here, as when an async fixture's teardown runs elsewhere.
    before_test = client_name_var.set('before-test')
    request_token = request_context_var.set('request-before-test')
    try:
        with pytest.MonkeyPatch.context() as inner:
            fixture = isolated_state.isolation(inner, fake_opensearch.url)
            contextvars.copy_context().run(fixture.__enter__)
            client_name_var.set('set-by-test')
            request_context_var.set('request-set-by-test')
            fixture.__exit__(None, None, None)

        assert client_name_var.get() == 'before-test'
        assert request_context_var.get() == 'request-before-test'
    finally:
        request_context_var.reset(request_token)
        client_name_var.reset(before_test)


def test_isolation_fallback_uses_default_when_unset_before(isolated_state, fake_opensearch):
    # Step 3: with no value before the test (token.old_value is Token.MISSING), the default.
    def setup_and_teardown_in_two_contexts():
        with pytest.MonkeyPatch.context() as inner:
            fixture = isolated_state.isolation(inner, fake_opensearch.url)
            contextvars.Context().run(fixture.__enter__)
            client_name_var.set('set-by-test')
            request_context_var.set('request-set-by-test')
            fixture.__exit__(None, None, None)
        return client_name_var.get(), request_context_var.get()

    assert contextvars.Context().run(setup_and_teardown_in_two_contexts) == ('unknown', None)


async def test_isolated_state_pins_server_modules_other_tests_reimport(
    isolated_state, fake_opensearch, mcp_session, golden, normalize
):
    with pytest.MonkeyPatch.context() as outer:
        # What tests/tools does: drop modules from sys.modules and import them again (here
        # with memory tools on), leaving a second TOOL_REGISTRY behind for later importers.
        for name in ('tools.memory_tools', 'tools.tools'):
            child = name.rpartition('.')[2]
            outer.delitem(sys.modules, name)
            outer.setattr(tools_package, child, getattr(tools_package, child))
        outer.setenv('MEMORY_TOOLS_ENABLED', 'true')
        importlib.import_module('tools.memory_tools')
        reimported = importlib.import_module('tools.tools')
        assert reimported.TOOL_REGISTRY is not TOOL_REGISTRY
        assert 'SaveMemoryTool' in reimported.TOOL_REGISTRY

        with (
            pytest.MonkeyPatch.context() as inner,
            isolated_state.isolation(inner, fake_opensearch.url),
        ):
            assert sys.modules['tools.tools'] is not reimported
            assert sys.modules['tools.tools'].TOOL_REGISTRY is TOOL_REGISTRY
            async with mcp_session() as session:
                golden('harness/list_tools_default', normalize(await session.list_tools()))

        assert sys.modules['tools.tools'] is reimported


# --- canned data -------------------------------------------------------------

# Dates in canned data must be far from today: the normalizer turns anything within two days
# of now into <TIMESTAMP>, so a near date would record a placeholder that later reads as a
# behaviour change. The guard (spec step 4) flags anything within this distance of now.
NEAR_NOW = timedelta(days=400)
ISO_DATE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})(?!\d)')
# `key: 123`, `"key": 123` or `key=123` in a file that is not parsed as JSON or YAML.
KEYED_INT = re.compile(r'["\']?([A-Za-z_][\w.-]*)["\']?\s*[:=]\s*(-?\d+)(?![\w.])')
STRUCTURED_LOADERS = {
    '.json': lambda text: [json.loads(text)],
    '.yaml': lambda text: list(yaml.safe_load_all(text)),
    '.yml': lambda text: list(yaml.safe_load_all(text)),
}


def _keyed_ints(path: Path, text: str) -> list[tuple]:
    """(key, int) pairs: from the parsed data of a JSON or YAML file, else from the text."""
    loader = STRUCTURED_LOADERS.get(path.suffix.lower())
    try:
        stack = loader(text) if loader else None
    except (ValueError, yaml.YAMLError):
        stack = None
    if stack is None:
        return [(match.group(1), int(match.group(2))) for match in KEYED_INT.finditer(text)]
    pairs = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, int) and not isinstance(item, bool):
                    pairs.append((key, item))
                stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
    return pairs


def near_now_values(path: Path, normalizer: type, now: datetime) -> list[str]:
    """Dates and timestamp-key epoch ints in ``path`` within ``NEAR_NOW`` of ``now`` (naive UTC)."""
    text = path.read_text(encoding='utf-8')
    found = []

    def near(moment: datetime) -> bool:
        return abs(moment - now) <= NEAR_NOW

    for match in ISO_DATE.finditer(text):
        try:
            moment = datetime(*(int(part) for part in match.groups()))
        except ValueError:
            continue
        if near(moment):
            found.append(f'ISO 8601 date {match.group(0)!r}')
    for match in normalizer.HTTP_DATE.finditer(text):
        moment = normalizer.http_date_moment(match)
        if moment is not None and near(moment):
            found.append(f'HTTP date {match.group(0)!r}')
    for key, number in _keyed_ints(path, text):
        if normalizer.is_timestamp_key(key) and any(
            near(moment) for moment in normalizer.epoch_moments(number)
        ):
            found.append(f'epoch int {key}: {number}')
    return found


def test_fixtures_hold_no_dates_near_now(normalize):
    # Step 4: every file under fixtures/ (JSON, YAML, text) keeps its dates far from today.
    files = sorted(path for path in FIXTURE_DIR.rglob('*') if path.is_file())
    assert files
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    problems = [
        f'{path.relative_to(FIXTURE_DIR).as_posix()}: {value}'
        for path in files
        for value in near_now_values(path, type(normalize), now)
    ]
    assert not problems, (
        'canned data must use fixed dates far from today, e.g. 2020-01-01T00:00:00Z '
        '(see README.md):\n' + '\n'.join(problems)
    )


def test_fixture_guard_names_near_now_dates_and_epochs(normalize, tmp_path):
    # Step 4: the guard itself finds what it must, in JSON, YAML and text, and nothing else.
    now = datetime.now(timezone.utc)
    soon = now + timedelta(days=300)  # inside the guard's 400 days, outside the normalizer's 2
    later = now + timedelta(days=500)
    seconds, millis = int(soon.timestamp()), int(soon.timestamp() * 1000)
    iso, http = f'{soon:%Y-%m-%dT%H:%M:%S}Z', format_datetime(soon, usegmt=True)
    date = f'{soon:%Y-%m-%d}'
    files = {
        'near.json': json.dumps(
            {
                'created_at': seconds,
                'nested': [{'last_updated_time': seconds, 'now_ms': millis}],
                'size': seconds,
                'values': [millis],
                'when': iso,
                'old': '2020-01-01T00:00:00Z',
                'later': f'{later:%Y-%m-%dT%H:%M:%S}Z',
                'old_ms': FAR_EPOCH * 1000,
            }
        ),
        'near.yaml': f"start_time: {seconds}\ncount: {seconds}\nheader: '{http}'\n",
        'near.txt': f'timestamp={seconds} doc_count={seconds}\nsince 2020-01-01\n',
        'far.json': json.dumps({'created_at': FAR_EPOCH, 'date': '2020-01-01T00:00:00Z'}),
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding='utf-8', newline='\n')

    def found(name):
        naive_now = now.replace(tzinfo=None)
        return sorted(near_now_values(tmp_path / name, type(normalize), naive_now))

    assert found('near.json') == sorted(
        [
            f'ISO 8601 date {date!r}',
            f'epoch int created_at: {seconds}',
            f'epoch int last_updated_time: {seconds}',
            f'epoch int now_ms: {millis}',
        ]
    )
    assert found('near.yaml') == sorted(
        [f'HTTP date {http!r}', f'epoch int start_time: {seconds}']
    )
    assert found('near.txt') == [f'epoch int timestamp: {seconds}']
    assert found('far.json') == []
