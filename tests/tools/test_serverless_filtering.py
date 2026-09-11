# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from semver import Version
from tools.tool_filter import (
    SERVERLESS_INCOMPATIBLE_TOOLS,
    filter_serverless_incompatible,
    get_tools,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _tool(display_name, min_version='1.0.0'):
    return {
        'display_name': display_name,
        'description': display_name,
        'input_schema': {'type': 'object', 'properties': {}},
        'function': MagicMock(),
        'args_model': MagicMock(),
        'min_version': min_version,
        'http_methods': 'GET',
    }


def _registry():
    return {
        'ListIndexTool': _tool('ListIndexTool'),
        'ClusterHealthTool': _tool('ClusterHealthTool'),
        'GetShardsTool': _tool('GetShardsTool'),
    }


class TestFilterHelper:
    def test_filter_removes_incompatible_in_place(self):
        registry = _registry()
        filter_serverless_incompatible(registry)
        assert 'ListIndexTool' in registry
        assert 'ClusterHealthTool' not in registry
        assert 'GetShardsTool' not in registry

    def test_expected_tools_are_flagged(self):
        for name in ('ClusterHealthTool', 'GetShardsTool', 'CatNodesTool', 'GetIndexStatsTool'):
            assert name in SERVERLESS_INCOMPATIBLE_TOOLS
        for name in ('ListIndexTool', 'SearchIndexTool', 'CountTool', 'PPLQueryTool'):
            assert name not in SERVERLESS_INCOMPATIBLE_TOOLS


class TestSingleModeListFiltering:
    def setup_method(self):
        from mcp_server_opensearch.global_state import set_mode

        set_mode('single')

    @pytest.mark.asyncio
    async def test_serverless_excludes_incompatible_tools(self):
        with (
            patch('tools.tool_filter.get_opensearch_version', return_value=None),
            patch('tools.tool_filter.is_tool_compatible', return_value=True),
            patch.dict('os.environ', {'AWS_OPENSEARCH_SERVERLESS': 'true'}),
        ):
            result = await get_tools(_registry())
        assert 'ListIndexTool' in result
        assert 'ClusterHealthTool' not in result
        assert 'GetShardsTool' not in result

    @pytest.mark.asyncio
    async def test_non_serverless_keeps_incompatible_tools(self):
        with (
            patch('tools.tool_filter.get_opensearch_version', return_value=Version.parse('2.5.0')),
            patch('tools.tool_filter.is_tool_compatible', return_value=True),
            patch.dict('os.environ', {'AWS_OPENSEARCH_SERVERLESS': 'false'}, clear=False),
        ):
            result = await get_tools(_registry())
        assert 'ClusterHealthTool' in result
        assert 'GetShardsTool' in result

    @pytest.mark.asyncio
    async def test_serverless_detected_from_aoss_url(self):
        url = 'https://abc123.us-east-1.aoss.amazonaws.com'
        with (
            patch('tools.tool_filter.get_opensearch_version', return_value=None),
            patch('tools.tool_filter.is_tool_compatible', return_value=True),
            patch.dict('os.environ', {'OPENSEARCH_URL': url}, clear=False),
        ):
            result = await get_tools(_registry())
        assert 'ClusterHealthTool' not in result


class TestMultiModeListFiltering:
    @pytest.mark.asyncio
    async def test_all_serverless_clusters_excludes_incompatible(self):
        from mcp_server_opensearch.global_state import set_mode

        set_mode('multi')
        registry = _registry()
        clusters = {
            'a': SimpleNamespace(is_serverless=True),
            'b': SimpleNamespace(is_serverless=True),
        }
        with patch.dict(
            'mcp_server_opensearch.clusters_information.cluster_registry', clusters, clear=True
        ):
            result = await get_tools(registry)
        assert 'ListIndexTool' in result
        assert 'ClusterHealthTool' not in result

    @pytest.mark.asyncio
    async def test_mixed_clusters_keeps_incompatible(self):
        from mcp_server_opensearch.global_state import set_mode

        set_mode('multi')
        registry = _registry()
        clusters = {
            'a': SimpleNamespace(is_serverless=True),
            'b': SimpleNamespace(is_serverless=False),
        }
        with patch.dict(
            'mcp_server_opensearch.clusters_information.cluster_registry', clusters, clear=True
        ):
            result = await get_tools(registry)
        assert 'ClusterHealthTool' in result


class TestCallTimeGuard:
    @pytest.mark.asyncio
    async def test_incompatible_tool_rejected_on_serverless(self):
        from mcp_server_opensearch.tool_executor import execute_tool

        fn = AsyncMock()
        enabled = {
            'ClusterHealthTool': {
                'display_name': 'ClusterHealthTool',
                'function': fn,
                'args_model': MagicMock(),
            }
        }
        with (
            patch('tools.tool_params.validate_args_for_mode', return_value=object()),
            patch('opensearch.client.is_serverless_connection', return_value=True),
        ):
            result = await execute_tool('ClusterHealthTool', {}, enabled)
        assert result.is_error is True
        assert 'Serverless' in result.content[0].text
        fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_incompatible_tool_allowed_on_non_serverless(self):
        from mcp_server_opensearch.tool_executor import execute_tool

        fn = AsyncMock(return_value=[])
        enabled = {
            'ClusterHealthTool': {
                'display_name': 'ClusterHealthTool',
                'function': fn,
                'args_model': MagicMock(),
            }
        }
        with (
            patch('tools.tool_params.validate_args_for_mode', return_value=object()),
            patch('opensearch.client.is_serverless_connection', return_value=False),
        ):
            await execute_tool('ClusterHealthTool', {}, enabled)
        fn.assert_awaited_once()


class TestIsServerlessConnection:
    def _args(self, **kw):
        base = {
            'aws_opensearch_serverless': None,
            'opensearch_cluster_name': '',
            'opensearch_url': None,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_explicit_arg_flag_wins(self):
        from opensearch.client import is_serverless_connection

        assert is_serverless_connection(self._args(aws_opensearch_serverless=True)) is True
        assert is_serverless_connection(self._args(aws_opensearch_serverless=False)) is False

    def test_resolved_datasource_wins(self):
        # In multi mode the per-request datasource is resolved the same way the
        # client builds it, so an aligned aws-service-name comma list is honored.
        from mcp_server_opensearch.clusters_information import ClusterInfo
        from opensearch.client import is_serverless_connection

        with (
            patch('opensearch.client.get_mode', return_value='multi'),
            patch(
                'mcp_server_opensearch.server_instructions.is_header_auth_enabled',
                return_value=True,
            ),
        ):
            with patch(
                'opensearch.client.resolve_header_cluster',
                return_value=ClusterInfo(opensearch_url='https://x', is_serverless=True),
            ):
                assert is_serverless_connection(self._args(opensearch_cluster_name='ds1')) is True
            with patch(
                'opensearch.client.resolve_header_cluster',
                return_value=ClusterInfo(opensearch_url='https://x', is_serverless=False),
            ):
                assert is_serverless_connection(self._args(opensearch_cluster_name='ds1')) is False

    def test_env_flag_and_url_heuristic(self):
        from opensearch.client import is_serverless_connection

        with patch('opensearch.client.get_mode', return_value='single'):
            with patch.dict('os.environ', {'AWS_OPENSEARCH_SERVERLESS': 'true'}, clear=False):
                assert is_serverless_connection(self._args()) is True
            assert (
                is_serverless_connection(
                    self._args(opensearch_url='https://x.us-east-1.aoss.amazonaws.com')
                )
                is True
            )

    def test_defaults_to_false(self):
        from opensearch.client import is_serverless_connection

        with (
            patch('opensearch.client.get_mode', return_value='single'),
            patch.dict(
                'os.environ', {'AWS_OPENSEARCH_SERVERLESS': '', 'OPENSEARCH_URL': ''}, clear=False
            ),
        ):
            assert is_serverless_connection(self._args()) is False
