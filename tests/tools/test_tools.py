# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from unittest.mock import Mock, patch


class TestTools:
    def setup_method(self):
        """Setup that runs before each test method."""
        # Create a properly configured mock client
        self.mock_client = Mock()

        # Configure mock client methods to return proper data structures
        # These will be overridden in individual tests as needed
        self.mock_client.cat.indices.return_value = []
        self.mock_client.indices.get_mapping.return_value = {}
        self.mock_client.indices.get.return_value = {}
        self.mock_client.search.return_value = {}
        self.mock_client.cat.shards.return_value = []
        self.mock_client.info.return_value = {'version': {'number': '2.11.1'}}

        # Patch initialize_client to always return our mock client
        self.init_client_patcher = patch(
            'opensearch.client.initialize_client', return_value=self.mock_client
        )
        self.init_client_patcher.start()

        # Clear any existing imports to ensure fresh imports
        import sys

        modules_to_clear = [
            'tools.tools',
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        # Import after patching to ensure fresh imports
        from tools.tools import (
            TOOL_REGISTRY,
            GetIndexMappingArgs,
            GetShardsArgs,
            ListIndicesArgs,
            SearchIndexArgs,
            get_index_mapping_tool,
            get_shards_tool,
            list_indices_tool,
            search_index_tool,
        )

        self.ListIndicesArgs = ListIndicesArgs
        self.GetIndexMappingArgs = GetIndexMappingArgs
        self.SearchIndexArgs = SearchIndexArgs
        self.GetShardsArgs = GetShardsArgs
        self.TOOL_REGISTRY = TOOL_REGISTRY
        self._list_indices_tool = list_indices_tool
        self._get_index_mapping_tool = get_index_mapping_tool
        self._search_index_tool = search_index_tool
        self._get_shards_tool = get_shards_tool

    def teardown_method(self):
        """Cleanup after each test method."""
        self.init_client_patcher.stop()

    @pytest.mark.asyncio
    async def test_list_indices_tool(self):
        """Test list_indices_tool returns full JSON info for all indices."""
        # Setup: mock full index info as returned by OpenSearch cat.indices
        self.mock_client.cat.indices.return_value = [
            {
                'health': 'green',
                'status': 'open',
                'index': 'index1',
                'uuid': 'uuid1',
                'pri': '1',
                'rep': '1',
                'docs.count': '100',
                'docs.deleted': '5',
                'store.size': '1mb',
                'pri.store.size': '0.5mb',
            },
            {
                'health': 'yellow',
                'status': 'open',
                'index': 'index2',
                'uuid': 'uuid2',
                'pri': '2',
                'rep': '2',
                'docs.count': '200',
                'docs.deleted': '10',
                'store.size': '2mb',
                'pri.store.size': '1mb',
            },
        ]
        # Execute
        result = await self._list_indices_tool(self.ListIndicesArgs())
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        # Should include the full JSON output
        assert '"index": "index1"' in result[0]['text']
        assert '"docs.count": "100"' in result[0]['text']
        assert '"index": "index2"' in result[0]['text']
        assert '"docs.count": "200"' in result[0]['text']
        self.mock_client.cat.indices.assert_called_once_with(format='json')

    @pytest.mark.asyncio
    async def test_list_indices_tool_with_index(self):
        """Test list_indices_tool returns detailed info for a single index."""
        # Setup: mock detailed index info as returned by OpenSearch indices.get
        mock_index_info = {
            'index1': {
                'aliases': {},
                'mappings': {'properties': {'field1': {'type': 'text'}}},
                'settings': {'index': {'number_of_shards': '1', 'number_of_replicas': '1'}},
            }
        }
        self.mock_client.indices.get.return_value = mock_index_info
        # Execute
        args = self.ListIndicesArgs(index='index1')
        result = await self._list_indices_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Index information for index1' in result[0]['text']
        assert '"index1"' in result[0]['text']
        assert '"number_of_shards": "1"' in result[0]['text']
        assert (
            '"field1": {"type": "text"}' in result[0]['text']
            or '"type": "text"' in result[0]['text']
        )
        self.mock_client.indices.get.assert_called_once_with(index='index1')

    @pytest.mark.asyncio
    async def test_list_indices_tool_error(self):
        """Test list_indices_tool exception handling."""
        # Setup
        self.mock_client.cat.indices.side_effect = Exception('Test error')
        # Execute
        result = await self._list_indices_tool(self.ListIndicesArgs())
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Error listing indices: Test error' in result[0]['text']
        self.mock_client.cat.indices.assert_called_once_with(format='json')

    @pytest.mark.asyncio
    async def test_get_index_mapping_tool(self):
        """Test get_index_mapping_tool successful."""
        # Setup
        mock_mapping = {'mappings': {'properties': {'field1': {'type': 'text'}}}}
        self.mock_client.indices.get_mapping.return_value = mock_mapping
        # Execute
        args = self.GetIndexMappingArgs(index='test-index')
        result = await self._get_index_mapping_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Mapping for test-index' in result[0]['text']
        assert json.loads(result[0]['text'].split('\n', 1)[1]) == mock_mapping
        self.mock_client.indices.get_mapping.assert_called_once_with(index='test-index')

    @pytest.mark.asyncio
    async def test_get_index_mapping_tool_error(self):
        """Test get_index_mapping_tool exception handling."""
        # Setup
        self.mock_client.indices.get_mapping.side_effect = Exception('Test error')
        # Execute
        args = self.GetIndexMappingArgs(index='test-index')
        result = await self._get_index_mapping_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Error getting mapping: Test error' in result[0]['text']
        self.mock_client.indices.get_mapping.assert_called_once_with(index='test-index')

    @pytest.mark.asyncio
    async def test_search_index_tool(self):
        """Test search_index_tool successful."""
        # Setup
        mock_results = {'hits': {'total': {'value': 1}, 'hits': [{'_source': {'field': 'value'}}]}}
        self.mock_client.search.return_value = mock_results
        # Execute
        args = self.SearchIndexArgs(index='test-index', query={'match_all': {}})
        result = await self._search_index_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Search results from test-index' in result[0]['text']
        assert json.loads(result[0]['text'].split('\n', 1)[1]) == mock_results
        self.mock_client.search.assert_called_once_with(index='test-index', body={'match_all': {}})

    @pytest.mark.asyncio
    async def test_search_index_tool_error(self):
        """Test search_index_tool exception handling."""
        # Setup
        self.mock_client.search.side_effect = Exception('Test error')
        # Execute
        args = self.SearchIndexArgs(index='test-index', query={'match_all': {}})
        result = await self._search_index_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Error searching index: Test error' in result[0]['text']
        self.mock_client.search.assert_called_once_with(index='test-index', body={'match_all': {}})

    @pytest.mark.asyncio
    async def test_get_shards_tool(self):
        """Test get_shards_tool successful."""
        # Setup
        mock_shards = [
            {
                'index': 'test-index',
                'shard': '0',
                'prirep': 'p',
                'state': 'STARTED',
                'docs': '1000',
                'store': '1mb',
                'ip': '127.0.0.1',
                'node': 'node1',
            }
        ]
        self.mock_client.cat.shards.return_value = mock_shards
        # Execute
        args = self.GetShardsArgs(index='test-index')
        result = await self._get_shards_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'index | shard | prirep | state | docs | store | ip | node' in result[0]['text']
        assert 'test-index | 0 | p | STARTED | 1000 | 1mb | 127.0.0.1 | node1' in result[0]['text']
        self.mock_client.cat.shards.assert_called_once_with(index='test-index', format='json')

    @pytest.mark.asyncio
    async def test_get_shards_tool_error(self):
        """Test get_shards_tool exception handling."""
        # Setup
        self.mock_client.cat.shards.side_effect = Exception('Test error')
        # Execute
        args = self.GetShardsArgs(index='test-index')
        result = await self._get_shards_tool(args)
        # Assert
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert 'Error getting shards information: Test error' in result[0]['text']
        self.mock_client.cat.shards.assert_called_once_with(index='test-index', format='json')

    def test_tool_registry(self):
        """Test TOOL_REGISTRY structure."""
        expected_tools = ['ListIndexTool', 'IndexMappingTool', 'SearchIndexTool', 'GetShardsTool']

        for tool in expected_tools:
            assert tool in self.TOOL_REGISTRY
            assert 'description' in self.TOOL_REGISTRY[tool]
            assert 'input_schema' in self.TOOL_REGISTRY[tool]
            assert 'function' in self.TOOL_REGISTRY[tool]
            assert 'args_model' in self.TOOL_REGISTRY[tool]

    def test_input_models(self):
        """Test input models validation."""
        with pytest.raises(ValueError):
            self.GetIndexMappingArgs()  # Should fail without index

        with pytest.raises(ValueError):
            self.SearchIndexArgs(index='test')  # Should fail without query

        # Test valid inputs
        assert self.GetIndexMappingArgs(index='test').index == 'test'
        assert self.SearchIndexArgs(index='test', query={'match': {}}).index == 'test'
        assert self.GetShardsArgs(index='test').index == 'test'
        assert isinstance(self.ListIndicesArgs(), self.ListIndicesArgs)
