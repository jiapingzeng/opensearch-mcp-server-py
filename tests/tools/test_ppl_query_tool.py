# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, Mock, patch


class TestPPLQueryTool:
    def setup_method(self):
        self.mock_client = Mock()
        self.mock_client.transport.perform_request = AsyncMock(return_value={})
        self.mock_client.info = AsyncMock(return_value={'version': {'number': '2.19.0'}})

        self.init_client_patcher = patch(
            'opensearch.client.initialize_client', return_value=self.mock_client
        )
        self.init_client_patcher.start()

        import sys

        for module in ['tools.tools', 'opensearch.helper']:
            if module in sys.modules:
                del sys.modules[module]

        from tools.tools import PplQueryArgs, ppl_query_tool

        self._ppl_query_tool = ppl_query_tool
        self.PplQueryArgs = PplQueryArgs

    def teardown_method(self):
        self.init_client_patcher.stop()

    @pytest.mark.asyncio
    async def test_basic_ppl_query(self):
        """Test basic PPL query execution."""
        self.mock_client.transport.perform_request.return_value = {
            'schema': [{'name': 'host', 'type': 'string'}],
            'datarows': [['host1'], ['host2']],
            'total': 2,
            'size': 2,
            'status': 200,
        }

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index | stats count() by host',
        )
        result = await self._ppl_query_tool(args)

        assert len(result) == 1
        assert 'PPL query results' in result[0]['text']
        assert 'host1' in result[0]['text']

        self.mock_client.transport.perform_request.assert_called_once_with(
            method='POST',
            url='/_plugins/_ppl',
            body='{"query": "source=my_index | stats count() by host | head 10"}',
            params={},
        )

    @pytest.mark.asyncio
    async def test_ppl_query_with_csv_format(self):
        """Test PPL query with non-default format passes format param."""
        self.mock_client.transport.perform_request.return_value = 'host\nhost1\nhost2'

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index | fields host',
            format='csv',
        )
        await self._ppl_query_tool(args)

        self.mock_client.transport.perform_request.assert_called_once_with(
            method='POST',
            url='/_plugins/_ppl',
            body='{"query": "source=my_index | fields host | head 10"}',
            params={'format': 'csv'},
        )

    @pytest.mark.asyncio
    async def test_ppl_query_error_handling(self):
        """Test PPL query error is properly reported."""
        self.mock_client.transport.perform_request.side_effect = Exception(
            'SyntaxError: Invalid query'
        )

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='invalid query',
        )
        result = await self._ppl_query_tool(args)

        assert 'Error' in result[0]['text']
        assert 'PPL query' in result[0]['text']

    @pytest.mark.asyncio
    async def test_ppl_query_jdbc_format_no_param(self):
        """Test that jdbc (default) format does not pass format param."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index',
            format='jdbc',
        )
        await self._ppl_query_tool(args)

        call_kwargs = self.mock_client.transport.perform_request.call_args[1]
        assert call_kwargs['params'] == {}

    @pytest.mark.asyncio
    async def test_ppl_query_default_size_appends_head_10(self):
        """Test that the default size appends '| head 10' to the query."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index',
        )
        await self._ppl_query_tool(args)

        body = self.mock_client.transport.perform_request.call_args[1]['body']
        assert body == '{"query": "source=my_index | head 10"}'

    @pytest.mark.asyncio
    async def test_ppl_query_custom_size(self):
        """Test that a custom size is used in the appended head stage."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index',
            size=25,
        )
        await self._ppl_query_tool(args)

        body = self.mock_client.transport.perform_request.call_args[1]['body']
        assert body == '{"query": "source=my_index | head 25"}'

    @pytest.mark.asyncio
    async def test_ppl_query_size_zero_skips_head(self):
        """Test that size=0 does not append a '| head' stage (aggregation escape hatch)."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index | stats count() by status',
            size=0,
        )
        await self._ppl_query_tool(args)

        body = self.mock_client.transport.perform_request.call_args[1]['body']
        assert body == '{"query": "source=my_index | stats count() by status"}'

    @pytest.mark.asyncio
    async def test_ppl_query_size_capped_at_100(self):
        """Test that size is capped at the default max of 100."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        args = self.PplQueryArgs(
            opensearch_cluster_name='test',
            query='source=my_index',
            size=1000,
        )
        await self._ppl_query_tool(args)

        body = self.mock_client.transport.perform_request.call_args[1]['body']
        assert body == '{"query": "source=my_index | head 100"}'

    @pytest.mark.asyncio
    async def test_ppl_query_size_capped_by_configured_max(self):
        """Test that size is capped by a configured max_size_limit on the registry."""
        self.mock_client.transport.perform_request.return_value = {'status': 200}

        from tools.tools import TOOL_REGISTRY

        original = TOOL_REGISTRY['PPLQueryTool'].get('max_size_limit')
        TOOL_REGISTRY['PPLQueryTool']['max_size_limit'] = 5
        try:
            args = self.PplQueryArgs(
                opensearch_cluster_name='test',
                query='source=my_index',
                size=50,
            )
            await self._ppl_query_tool(args)

            body = self.mock_client.transport.perform_request.call_args[1]['body']
            assert body == '{"query": "source=my_index | head 5"}'
        finally:
            if original is None:
                TOOL_REGISTRY['PPLQueryTool'].pop('max_size_limit', None)
            else:
                TOOL_REGISTRY['PPLQueryTool']['max_size_limit'] = original

    def test_ppl_query_negative_size_rejected(self):
        """Test that a negative size is rejected at validation (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            self.PplQueryArgs(
                opensearch_cluster_name='test',
                query='source=my_index',
                size=-1,
            )
