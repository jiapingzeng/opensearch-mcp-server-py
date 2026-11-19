# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from mcp.types import TextContent
from pydantic import BaseModel
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch


class TestToolGenerator:
    def setup_method(self):
        """Setup that runs before each test method."""
        # Mock OpenSearch client
        self.mock_client = Mock()
        self.mock_client.transport.perform_request = AsyncMock()

        # Mock spec
        self.mock_spec = {
            'paths': {
                '/_cluster/health/{index}': {
                    'get': {
                        'description': 'Returns cluster health',
                        'x-operation-group': 'cluster.health',
                    }
                },
                '/_count': {
                    'get': {
                        'description': 'Returns count of documents',
                        'x-operation-group': 'count',
                        'requestBody': {'content': {'application/json': {'schema': {}}}},
                    }
                },
            }
        }

        # Import after mocking
        from tools.tool_generator import (
            SPEC_FILES,
            extract_parameters,
            fetch_github_spec,
            generate_tool_from_group,
            generate_tools_from_openapi,
            group_endpoints_by_operation,
            process_body,
            select_endpoint,
        )

        self.fetch_github_spec = fetch_github_spec
        self.group_endpoints_by_operation = group_endpoints_by_operation
        self.extract_parameters = extract_parameters
        self.process_body = process_body
        self.select_endpoint = select_endpoint
        self.generate_tool_from_group = generate_tool_from_group
        self.generate_tools_from_openapi = generate_tools_from_openapi
        self.SPEC_FILES = SPEC_FILES

    @pytest.mark.asyncio
    async def test_fetch_github_spec(self):
        """Test fetching OpenSearch API specification from GitHub."""
        # Create a mock response
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=yaml.dump(self.mock_spec))
        mock_response.raise_for_status = Mock()

        # Create a mock for the get response context manager
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)

        # Create a mock session
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_get_context)

        # Create a mock for the ClientSession context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_context.__aexit__ = AsyncMock(return_value=None)

        # Patch aiohttp.ClientSession
        with patch('aiohttp.ClientSession', return_value=mock_client_context):
            # Call the function
            result = await self.fetch_github_spec('test.yaml')

            # Verify result
            assert result == self.mock_spec
            mock_session.get.assert_called_once()
            mock_response.raise_for_status.assert_called_once()
            mock_response.text.assert_called_once()

    def test_group_endpoints_by_operation(self):
        """Test grouping endpoints by operation."""
        # Call the function
        result = self.group_endpoints_by_operation(self.mock_spec['paths'])

        # Verify result
        assert 'cluster.health' in result
        assert 'count' in result
        assert len(result['cluster.health']) == 1
        assert result['cluster.health'][0]['path'] == '/_cluster/health/{index}'
        assert result['cluster.health'][0]['method'] == 'get'

    def test_extract_parameters(self):
        """Test extracting parameters from endpoints."""
        # Create endpoints from mock_spec
        endpoints = []
        for path, methods in self.mock_spec['paths'].items():
            for method, details in methods.items():
                endpoints.append({'path': path, 'method': method, 'details': details})

        # Call the function
        all_parameters, path_parameters, required_parameters = self.extract_parameters(endpoints)

        # Verify results
        assert 'opensearch_cluster_name' in all_parameters
        assert 'index' in all_parameters
        assert 'body' in all_parameters

        # Check path parameters
        assert 'index' in path_parameters
        assert len(path_parameters) == 1

        # Check parameter details
        assert all_parameters['index']['title'] == 'Index'
        assert all_parameters['index']['type'] == 'string'
        assert all_parameters['body']['title'] == 'Body'

    def test_process_body(self):
        """Test processing request body."""
        # Test with dictionary containing float values
        body = {'settings': {'number_of_shards': 1.0, 'number_of_replicas': 2.0}}
        result = self.process_body(body, 'IndicesCreate')
        assert result['settings']['number_of_shards'] == 1
        assert result['settings']['number_of_replicas'] == 2

        # Test with msearch NDJSON
        body = '{"index":"test"}\n{"query":{"match_all":{}}}'
        result = self.process_body(body, 'MsearchTool')
        assert result.endswith('\n')

        # Test with JSON string
        body = '{"query":{"match_all":{}}}'
        result = self.process_body(body, 'Count')
        assert isinstance(result, dict)
        assert 'query' in result

        # Test with JSON array for msearch
        body = '[{"index":"test"},{"query":{"match_all":{}}}]'
        result = self.process_body(body, 'MsearchTool')
        assert result.endswith('\n')
        assert '{"index": "test"}\n{"query": {"match_all": {}}}' in result

    def test_select_endpoint(self):
        """Test selecting the most appropriate endpoint based on parameters."""
        # Setup test endpoints
        endpoints = [
            {
                'path': '/_cluster/health',
                'method': 'get',
                'details': {'description': 'Returns cluster health for index'},
            },
            {
                'path': '/_cluster/health/{index}',
                'method': 'get',
                'details': {'description': 'Returns cluster health'},
            },
            {
                'path': '/_cluster/health/{index}/{shard}',
                'method': 'get',
                'details': {'description': 'Returns cluster health for specific shard'},
            },
        ]

        # Test case 1: With index parameter - should select the endpoint with {index}
        params = {'index': 'test-index'}
        selected = self.select_endpoint(endpoints, params)
        assert selected['path'] == '/_cluster/health/{index}'

        # Test case 2: Without index parameter - should select the simpler endpoint
        params = {}
        selected = self.select_endpoint(endpoints, params)
        assert selected['path'] == '/_cluster/health'

        # Test case 3: With both index and shard parameters - should select the most specific endpoint
        params = {'index': 'test-index', 'shard': '0'}
        selected = self.select_endpoint(endpoints, params)
        assert selected['path'] == '/_cluster/health/{index}/{shard}'

        # Test case 4: With index and extra parameters - should still select the index endpoint
        params = {'index': 'test-index', 'other_param': 'value'}
        selected = self.select_endpoint(endpoints, params)
        assert selected['path'] == '/_cluster/health/{index}'

        # Test case 5: With shard but no index - should select the base endpoint
        # since the shard endpoint requires index parameter
        params = {'shard': '0'}
        selected = self.select_endpoint(endpoints, params)
        assert selected['path'] == '/_cluster/health'

    @pytest.mark.asyncio
    async def test_generate_tool_from_group(self):
        """Test generating a tool from a group of endpoints."""
        # Setup
        endpoints = [
            {
                'path': '/_count',
                'method': 'get',
                'details': {
                    'description': 'Returns count of documents',
                    'x-operation-group': 'count',
                    'requestBody': {'content': {'application/json': {'schema': {}}}},
                },
            }
        ]

        # Mock initialize_client to return our mock client
        with patch('opensearch.client.initialize_client', return_value=self.mock_client):
            # Generate the tool
            tool = self.generate_tool_from_group('Count', endpoints)

            # Verify tool structure
            assert 'description' in tool
            assert 'input_schema' in tool
            assert 'function' in tool
            assert 'args_model' in tool

            # Test the tool function with proper Pydantic model
            class MockParams(BaseModel):
                body: dict = {'query': {'match_all': {}}}
                opensearch_cluster_name: str = 'test-cluster'

            # Mock the tool function to avoid actual execution
            mock_tool_function = AsyncMock(
                return_value=[TextContent(type='text', text='test result')]
            )
            tool['function'] = mock_tool_function

            result = await tool['function'](MockParams())

            # Verify result
            assert len(result) == 1
            assert result[0].type == 'text'
            mock_tool_function.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_tools_from_openapi(self):
        """Test generating tools from OpenAPI specifications."""
        # Setup
        mock_spec = {
            'paths': {
                '/_cluster/health': {
                    'get': {
                        'description': 'Returns cluster health',
                        'x-operation-group': 'cluster.health',
                    }
                },
                '/_count': {
                    'get': {
                        'description': 'Returns count of documents',
                        'x-operation-group': 'count',
                        'requestBody': {'content': {'application/json': {'schema': {}}}},
                    }
                },
            }
        }

        mock_registry = {}
        mock_tool = {
            'description': 'Test description',
            'input_schema': {},
            'function': AsyncMock(),
            'args_model': Mock(),
        }

        # Patch the names generate_tools_from_openapi looks up in its own module,
        # so the test never fetches the real spec from GitHub (runs offline).
        with (
            patch(
                'tools.tool_generator.fetch_github_spec', AsyncMock(return_value=mock_spec)
            ) as mock_fetch,
            patch(
                'tools.tool_generator.generate_tool_from_group', return_value=mock_tool
            ) as mock_generate,
            patch('tools.tool_generator.TOOL_REGISTRY', mock_registry),
        ):
            result = await self.generate_tools_from_openapi()

        # Verify the spec was fetched once per spec file, from the mock
        assert [c.args for c in mock_fetch.await_args_list] == [(f,) for f in self.SPEC_FILES]

        # Verify tools were generated for each supported operation group
        generated_names = {c.args[0] for c in mock_generate.call_args_list}
        assert generated_names == {'ClusterHealth', 'Count'}

        # Verify results were written to the patched registry and returned
        assert result is mock_registry
        assert 'ClusterHealthTool' in result
        assert 'CountTool' in result
        assert 'description' in result['ClusterHealthTool']
        assert 'input_schema' in result['ClusterHealthTool']
        assert 'function' in result['ClusterHealthTool']
        assert 'args_model' in result['ClusterHealthTool']

    @pytest.mark.asyncio
    async def test_generate_tools_from_openapi_fetch_failure_logs_not_stdout(self, capsys, caplog):
        """A failed spec fetch is logged at ERROR, nothing reaches stdout, registry unchanged.

        On stdio, stdout is the JSON-RPC stream, so the failure must go to the log.
        """
        existing_tool = {'description': 'Existing tool'}
        mock_registry = {'ExistingTool': existing_tool}
        before = dict(mock_registry)

        with (
            patch(
                'tools.tool_generator.fetch_github_spec',
                AsyncMock(side_effect=RuntimeError('cannot connect to host')),
            ),
            patch('tools.tool_generator.TOOL_REGISTRY', mock_registry),
            caplog.at_level('ERROR', logger='tools.tool_generator'),
        ):
            result = await self.generate_tools_from_openapi()

        # Nothing is written to stdout
        assert capsys.readouterr().out == ''

        # The error is logged at ERROR with the same text the print used
        records = [r for r in caplog.records if r.name == 'tools.tool_generator']
        assert [(r.levelname, r.getMessage()) for r in records] == [
            ('ERROR', 'Error generating tools: cannot connect to host')
        ]

        # The registry is returned unchanged
        assert result is mock_registry
        assert result == before
        assert result['ExistingTool'] is existing_tool


class TestGeneratedToolQueryParams:
    """Query parameters a generated tool sends to OpenSearch."""

    # One endpoint with a required path parameter and typed optional query parameters.
    ENDPOINTS = [
        {
            'path': '/{index}/_count',
            'method': 'get',
            'details': {
                'description': 'Returns count of documents',
                'x-operation-group': 'count',
                'parameters': [
                    {'name': 'allow_no_indices', 'in': 'query', 'schema': {'type': 'boolean'}},
                    {'name': 'min_score', 'in': 'query', 'schema': {'type': 'number'}},
                    {'name': 'terminate_after', 'in': 'query', 'schema': {'type': 'integer'}},
                    {'name': 'q', 'in': 'query', 'schema': {'type': 'string'}},
                    {'name': 'routing', 'in': 'query', 'schema': {'type': 'string'}},
                ],
                'requestBody': {'content': {'application/json': {'schema': {}}}},
            },
        }
    ]

    def setup_method(self):
        from tools.tool_generator import generate_tool_from_group

        self.tool = generate_tool_from_group('Count', self.ENDPOINTS)
        self.mock_client = Mock()
        self.mock_client.transport.perform_request = AsyncMock(return_value={'count': 0})

    async def call(self, params: BaseModel) -> dict:
        """Run the generated tool function; return the kwargs of its one perform_request."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def fake_get_opensearch_client(args):
            yield self.mock_client

        with (
            patch('opensearch.client.get_opensearch_client', fake_get_opensearch_client),
            patch('tools.tool_generator.check_tool_compatibility', AsyncMock()),
        ):
            result = await self.tool['function'](params)

        assert result == [TextContent(type='text', text='{"count": 0}')]
        self.mock_client.transport.perform_request.assert_awaited_once()
        return self.mock_client.transport.perform_request.await_args.kwargs

    @pytest.mark.asyncio
    async def test_required_arguments_only_sends_no_none_query_param(self):
        """Step 1 reproduction: with only the required arguments, no query parameter is None."""
        params = self.tool['args_model'](opensearch_cluster_name='', index='logs')

        sent = await self.call(params)

        assert None not in sent['params'].values()
        assert sent == {'method': 'GET', 'url': '/logs/_count', 'params': {}, 'body': None}

    @pytest.mark.asyncio
    async def test_false_zero_and_empty_values_are_still_sent(self):
        """A parameter set to False or 0 (or '', [], {}) is still sent as it is today."""

        # The generated model types every query parameter as str; a subclass lets the
        # caller pass the falsy non-None values that must not be dropped with None.
        class TypedCountArgs(self.tool['args_model']):
            allow_no_indices: Optional[bool] = None
            min_score: Optional[float] = None
            terminate_after: Optional[int] = None
            routing: Any = None

        params = TypedCountArgs(
            opensearch_cluster_name='',
            index='logs',
            allow_no_indices=False,
            min_score=0.0,
            terminate_after=0,
            q='',
            routing=[],
        )

        sent = await self.call(params)

        assert sent['params'] == {
            'allow_no_indices': False,
            'min_score': 0.0,
            'terminate_after': 0,
            'q': '',
            'routing': [],
        }

        # An empty dict is kept too.
        self.mock_client.transport.perform_request.reset_mock()
        sent = await self.call(
            TypedCountArgs(opensearch_cluster_name='', index='logs', routing={})
        )
        assert sent['params'] == {'routing': {}}

    @pytest.mark.asyncio
    async def test_parameter_set_by_caller_is_sent_with_its_value(self):
        """A parameter the caller set is still sent with its value; the omitted ones are not."""
        params = self.tool['args_model'](
            opensearch_cluster_name='', index='logs', q='level:error', routing='shard-1'
        )

        sent = await self.call(params)

        assert sent == {
            'method': 'GET',
            'url': '/logs/_count',
            'params': {'q': 'level:error', 'routing': 'shard-1'},
            'body': None,
        }


class TestGeneratedToolTimeoutParam:
    """A generated tool's `timeout` argument is OpenSearch's query parameter.

    The call goes through a real opensearch-py client and transport, configured with a
    client timeout as the server configures it, down to a connection that records what it
    receives.
    """

    ENDPOINTS = [
        {
            'path': '/_cluster/health',
            'method': 'get',
            'details': {
                'description': 'Returns cluster health',
                'x-operation-group': 'cluster.health',
                'parameters': [
                    {'name': 'timeout', 'in': 'query', 'schema': {'type': 'string'}},
                    {'name': 'level', 'in': 'query', 'schema': {'type': 'string'}},
                ],
            },
        }
    ]
    CONFIGURED_TIMEOUT = 30

    def setup_method(self):
        from tools.tool_generator import generate_tool_from_group

        self.tool = generate_tool_from_group('ClusterHealth', self.ENDPOINTS)

    async def call(self, **arguments) -> tuple[dict, dict]:
        """Run the tool; return the transport call's kwargs and what the connection got."""
        from contextlib import asynccontextmanager
        from opensearchpy import AsyncOpenSearch
        from opensearchpy._async.http_aiohttp import AsyncConnection

        received = []

        class RecordingConnection(AsyncConnection):
            async def perform_request(
                self, method, url, params=None, body=None, timeout=None, ignore=(), headers=None
            ):
                received.append(
                    {
                        'method': method,
                        'url': url,
                        'params': dict(params or {}),
                        'body': body,
                        'timeout': timeout,
                        'configured_timeout': self.timeout,
                    }
                )
                return 200, {'content-type': 'application/json'}, '{"status": "green"}'

            async def close(self):
                pass

        client = AsyncOpenSearch(
            hosts=['http://localhost:9200'],
            connection_class=RecordingConnection,
            timeout=self.CONFIGURED_TIMEOUT,
        )
        spy = AsyncMock(wraps=client.transport.perform_request)

        @asynccontextmanager
        async def fake_get_opensearch_client(args):
            yield client

        try:
            with (
                patch.object(client.transport, 'perform_request', spy),
                patch('opensearch.client.get_opensearch_client', fake_get_opensearch_client),
                patch('tools.tool_generator.check_tool_compatibility', AsyncMock()),
            ):
                result = await self.tool['function'](
                    self.tool['args_model'](opensearch_cluster_name='', **arguments)
                )
        finally:
            await client.close()

        assert result == [TextContent(type='text', text='{"status": "green"}')]
        spy.assert_awaited_once()
        assert len(received) == 1
        return spy.await_args.kwargs, received[0]

    @pytest.mark.asyncio
    async def test_caller_timeout_is_sent_as_query_parameter(self):
        """Step 1 reproduction: a caller's `timeout: '5s'` reaches OpenSearch in the query."""
        _, received = await self.call(timeout='5s')

        assert received['url'] == '/_cluster/health'
        assert received['params'] == {'timeout': '5s'}

    @pytest.mark.asyncio
    async def test_caller_timeout_leaves_the_client_timeout_configured(self):
        """The request timeout the connection gets is the client's configured one, not '5s'."""
        _, received = await self.call(timeout='5s', level='indices')

        assert received['timeout'] == self.CONFIGURED_TIMEOUT
        assert received['configured_timeout'] == self.CONFIGURED_TIMEOUT
        assert received['params'] == {'timeout': '5s', 'level': 'indices'}

    @pytest.mark.asyncio
    async def test_call_without_timeout_is_unchanged(self):
        """Without `timeout`, the transport call and the request are as before the fix."""
        transport_call, received = await self.call(level='indices')

        assert transport_call == {
            'method': 'GET',
            'url': '/_cluster/health',
            'params': {'level': 'indices'},
            'body': None,
        }
        assert received == {
            'method': 'GET',
            'url': '/_cluster/health',
            'params': {'level': 'indices'},
            'body': None,
            'timeout': None,
            'configured_timeout': self.CONFIGURED_TIMEOUT,
        }
