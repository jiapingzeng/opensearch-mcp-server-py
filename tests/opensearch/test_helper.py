# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tools.tool_params import (
    GetIndexMappingArgs,
    GetShardsArgs,
    ListIndicesArgs,
    SearchIndexArgs,
    baseToolArgs,
)
from unittest.mock import patch


class TestOpenSearchHelper:
    def setup_method(self):
        """Setup that runs before each test method."""
        from opensearch.helper import (
            get_index_mapping,
            get_shards,
            list_indices,
            search_index,
        )

        # Store functions
        self.list_indices = list_indices
        self.get_index_mapping = get_index_mapping
        self.search_index = search_index
        self.get_shards = get_shards

    @patch('opensearch.client.initialize_client')
    def test_list_indices(self, mock_initialize_client):
        """Test list_indices function."""
        # Setup mock response
        mock_response = [
            {'index': 'index1', 'health': 'green', 'status': 'open'},
            {'index': 'index2', 'health': 'yellow', 'status': 'open'},
        ]
        mock_client = mock_initialize_client.return_value
        mock_client.cat.indices.return_value = mock_response

        # Execute
        result = self.list_indices(ListIndicesArgs())

        # Assert
        assert result == mock_response
        mock_initialize_client.assert_called_once_with(ListIndicesArgs())
        mock_client.cat.indices.assert_called_once_with(format='json')

    @patch('opensearch.client.initialize_client')
    def test_get_index_mapping(self, mock_initialize_client):
        """Test get_index_mapping function."""
        # Setup mock response
        mock_response = {
            'test-index': {
                'mappings': {
                    'properties': {
                        'field1': {'type': 'text'},
                        'field2': {'type': 'keyword'},
                    }
                }
            }
        }
        mock_client = mock_initialize_client.return_value
        mock_client.indices.get_mapping.return_value = mock_response

        # Execute
        result = self.get_index_mapping(GetIndexMappingArgs(index='test-index'))

        # Assert
        assert result == mock_response
        mock_initialize_client.assert_called_once_with(GetIndexMappingArgs(index='test-index'))
        mock_client.indices.get_mapping.assert_called_once_with(index='test-index')

    @patch('opensearch.client.initialize_client')
    def test_search_index(self, mock_initialize_client):
        """Test search_index function."""
        # Setup mock response
        mock_response = {
            'hits': {
                'total': {'value': 1},
                'hits': [{'_index': 'test-index', '_id': '1', '_source': {'field': 'value'}}],
            }
        }
        mock_client = mock_initialize_client.return_value
        mock_client.search.return_value = mock_response

        # Setup test query
        test_query = {'query': {'match_all': {}}}

        # Execute
        result = self.search_index(
            SearchIndexArgs(
                index='test-index',
                query=test_query,
            )
        )

        # Assert
        assert result == mock_response
        mock_initialize_client.assert_called_once_with(
            SearchIndexArgs(
                index='test-index',
                query=test_query,
            )
        )
        mock_client.search.assert_called_once_with(index='test-index', body=test_query)

    @patch('opensearch.client.initialize_client')
    def test_get_shards(self, mock_initialize_client):
        """Test get_shards function."""
        # Setup mock response
        mock_response = [
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
        mock_client = mock_initialize_client.return_value
        mock_client.cat.shards.return_value = mock_response

        # Execute
        result = self.get_shards(GetShardsArgs(index='test-index'))

        # Assert
        assert result == mock_response
        mock_initialize_client.assert_called_once_with(GetShardsArgs(index='test-index'))
        mock_client.cat.shards.assert_called_once_with(index='test-index', format='json')

    @patch('opensearch.client.initialize_client')
    def test_list_indices_error(self, mock_initialize_client):
        """Test list_indices error handling."""
        # Setup mock to raise exception
        mock_client = mock_initialize_client.return_value
        mock_client.cat.indices.side_effect = Exception('Connection error')

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            self.list_indices(ListIndicesArgs())
        assert str(exc_info.value) == 'Connection error'

    @patch('opensearch.client.initialize_client')
    def test_get_index_mapping_error(self, mock_initialize_client):
        """Test get_index_mapping error handling."""
        # Setup mock to raise exception
        mock_client = mock_initialize_client.return_value
        mock_client.indices.get_mapping.side_effect = Exception('Index not found')

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            self.get_index_mapping(GetIndexMappingArgs(index='non-existent-index'))
        assert str(exc_info.value) == 'Index not found'

    @patch('opensearch.client.initialize_client')
    def test_search_index_error(self, mock_initialize_client):
        """Test search_index error handling."""
        # Setup mock to raise exception
        mock_client = mock_initialize_client.return_value
        mock_client.search.side_effect = Exception('Invalid query')

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            self.search_index(
                SearchIndexArgs(
                    index='test-index',
                    query={'invalid': 'query'},
                )
            )
        assert str(exc_info.value) == 'Invalid query'

    @patch('opensearch.client.initialize_client')
    def test_get_shards_error(self, mock_initialize_client):
        """Test get_shards error handling."""
        # Setup mock to raise exception
        mock_client = mock_initialize_client.return_value
        mock_client.cat.shards.side_effect = Exception('Shard not found')

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            self.get_shards(GetShardsArgs(index='non-existent-index'))
        assert str(exc_info.value) == 'Shard not found'

    @patch('opensearch.client.initialize_client')
    def test_get_opensearch_version(self, mock_initialize_client):
        from opensearch.helper import get_opensearch_version

        # Setup mock response
        mock_response = {'version': {'number': '2.11.1'}}
        mock_client = mock_initialize_client.return_value
        mock_client.info.return_value = mock_response
        # Execute
        args = baseToolArgs()
        result = get_opensearch_version(args)
        # Assert
        assert str(result) == '2.11.1'
        mock_initialize_client.assert_called_once_with(args)
        mock_client.info.assert_called_once_with()

    @patch('opensearch.client.initialize_client')
    def test_get_opensearch_version_error(self, mock_initialize_client):
        from opensearch.helper import get_opensearch_version
        from tools.tool_params import baseToolArgs

        # Setup mock to raise exception
        mock_client = mock_initialize_client.return_value
        mock_client.info.side_effect = Exception('Failed to get version')
        args = baseToolArgs()
        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            get_opensearch_version(args)
        assert str(exc_info.value) == 'Failed to get version'
