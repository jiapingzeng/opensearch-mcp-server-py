# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the characterization harness itself (see conftest.py and README.md)."""

import copy
import importlib
import logging
import os
import pytest
import sys
import tools as tools_package
from datetime import datetime, timezone
from mcp_server_opensearch import clusters_information, global_state
from mcp_server_opensearch.client_context import client_name_var
from mcp_server_opensearch.clusters_information import ClusterInfo, add_cluster
from mcp_server_opensearch.logging_config import configure_logging
from tools import tool_filter
from tools.tool_params import ListIndicesArgs
from tools.tools import TOOL_REGISTRY


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
    original = path.read_text(encoding='utf-8')
    # Canonical form: sorted keys, 2-space indent, non-ASCII kept, trailing newline.
    assert original == '{\n  "a": "é",\n  "b": 1\n}\n'

    with pytest.raises(pytest.fail.Exception) as excinfo:
        golden('probe/existing', {'b': 2, 'a': 'é'})

    assert '-  "b": 1' in str(excinfo.value)
    assert '+  "b": 2' in str(excinfo.value)
    assert path.read_text(encoding='utf-8') == original


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
