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
from unittest.mock import patch, AsyncMock, MagicMock


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

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_list_indices(self, mock_get_client):
        """Test list_indices function."""
        # Setup mock response
        mock_response = [
            {'index': 'index1', 'health': 'green', 'status': 'open'},
            {'index': 'index2', 'health': 'yellow', 'status': 'open'},
        ]
        mock_client = AsyncMock()
        mock_client.cat.indices = AsyncMock(return_value=mock_response)

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute
        result = await self.list_indices(ListIndicesArgs(opensearch_cluster_name=''))

        # Assert
        assert result == mock_response
        mock_get_client.assert_called_once_with(ListIndicesArgs(opensearch_cluster_name=''))
        mock_client.cat.indices.assert_called_once_with()

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_index_mapping(self, mock_get_client):
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
        mock_client = AsyncMock()
        mock_client.indices.get_mapping = AsyncMock(return_value=mock_response)

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute
        result = await self.get_index_mapping(
            GetIndexMappingArgs(index='test-index', opensearch_cluster_name='')
        )

        # Assert
        assert result == mock_response
        mock_get_client.assert_called_once_with(
            GetIndexMappingArgs(index='test-index', opensearch_cluster_name='')
        )
        mock_client.indices.get_mapping.assert_called_once_with(index='test-index')

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_search_index(self, mock_get_client):
        """Test search_index function."""
        # Setup mock response
        mock_response = {
            'hits': {
                'total': {'value': 1},
                'hits': [{'_index': 'test-index', '_id': '1', '_source': {'field': 'value'}}],
            }
        }
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=mock_response)

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Setup test query
        test_query = {'query': {'match_all': {}}}

        # Execute
        result = await self.search_index(
            SearchIndexArgs(index='test-index', query=test_query, opensearch_cluster_name='')
        )

        # Assert
        assert result == mock_response
        mock_get_client.assert_called_once_with(
            SearchIndexArgs(index='test-index', query=test_query, opensearch_cluster_name='')
        )
        mock_client.search.assert_called_once_with(index='test-index', body=test_query)

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_shards(self, mock_get_client):
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
        mock_client = AsyncMock()
        mock_client.cat.shards = AsyncMock(return_value=mock_response)

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute
        result = await self.get_shards(
            GetShardsArgs(index='test-index', opensearch_cluster_name='')
        )

        # Assert
        assert result == mock_response
        mock_get_client.assert_called_once_with(
            GetShardsArgs(index='test-index', opensearch_cluster_name='')
        )
        mock_client.cat.shards.assert_called_once_with(index='test-index', format='json')

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_list_indices_error(self, mock_get_client):
        """Test list_indices error handling."""
        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_client.cat.indices = AsyncMock(side_effect=Exception('Connection error'))

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            await self.list_indices(ListIndicesArgs(opensearch_cluster_name=''))
        assert str(exc_info.value) == 'Connection error'

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_index_mapping_error(self, mock_get_client):
        """Test get_index_mapping error handling."""
        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_client.indices.get_mapping = AsyncMock(side_effect=Exception('Index not found'))

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            await self.get_index_mapping(
                GetIndexMappingArgs(index='non-existent-index', opensearch_cluster_name='')
            )
        assert str(exc_info.value) == 'Index not found'

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_search_index_error(self, mock_get_client):
        """Test search_index error handling."""
        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=Exception('Invalid query'))

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            await self.search_index(
                SearchIndexArgs(
                    index='test-index', query={'invalid': 'query'}, opensearch_cluster_name=''
                )
            )
        assert str(exc_info.value) == 'Invalid query'

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_shards_error(self, mock_get_client):
        """Test get_shards error handling."""
        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_client.cat.shards = AsyncMock(side_effect=Exception('Shard not found'))

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            await self.get_shards(
                GetShardsArgs(index='non-existent-index', opensearch_cluster_name='')
            )
        assert str(exc_info.value) == 'Shard not found'

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_opensearch_version(self, mock_get_client):
        from opensearch.helper import get_opensearch_version

        # Setup mock response
        mock_response = {'version': {'number': '2.11.1'}}
        mock_client = AsyncMock()
        mock_client.info = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Execute
        args = baseToolArgs(opensearch_cluster_name='')
        result = await get_opensearch_version(args)
        # Assert
        assert str(result) == '2.11.1'
        mock_get_client.assert_called_once_with(args)
        mock_client.info.assert_called_once_with()

    @pytest.mark.asyncio
    @patch('opensearch.client.get_opensearch_client')
    async def test_get_opensearch_version_error(self, mock_get_client):
        from opensearch.helper import get_opensearch_version
        from tools.tool_params import baseToolArgs

        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_client.info = AsyncMock(side_effect=Exception('Failed to get version'))
        mock_client.close = AsyncMock()

        # Setup async context manager
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        args = baseToolArgs(opensearch_cluster_name='')
        # Execute and assert
        result = await get_opensearch_version(args)
        assert result is None
