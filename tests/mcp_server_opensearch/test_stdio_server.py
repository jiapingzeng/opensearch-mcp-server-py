# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pytest
from contextlib import asynccontextmanager
from mcp.types import TextContent
from unittest.mock import AsyncMock, Mock, patch


# Set environment variables
os.environ['OPENSEARCH_URL'] = 'https://test-domain.us-west-2.es.amazonaws.com'
os.environ['AWS_REGION'] = 'us-west-2'
os.environ['AWS_ACCESS_KEY_ID'] = 'test-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test-secret-key'

# Mock tool registry for testing
MOCK_TOOL_REGISTRY = {
    'test_tool': {
        'description': 'Test tool',
        'input_schema': {'type': 'object', 'properties': {}},
        'args_model': Mock(),
        'function': AsyncMock(return_value=[TextContent(type='text', text='test result')]),
    }
}


@pytest.fixture(autouse=True)
def patch_opensearch_version():
    """Mock OpenSearch client and version check."""
    from unittest.mock import AsyncMock

    mock_client = Mock()
    mock_client.info = AsyncMock(return_value={'version': {'number': '3.0.0'}})

    async def mock_get_version(*args, **kwargs):
        from semver import Version

        return Version.parse('2.9.0')

    with (
        patch('opensearch.helper.get_opensearch_version', side_effect=mock_get_version),
        patch('opensearch.client.initialize_client', return_value=mock_client),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_generate_tools():
    """Mock the generate_tools_from_openapi function."""

    async def mock_gen_tools():
        return None

    with patch(
        'mcp_server_opensearch.server_factory.generate_tools_from_openapi',
        side_effect=mock_gen_tools,
    ):
        yield


@pytest.fixture
def mock_server():
    """Replace OpenSearchMCPServer in the factory; the transport runs its low-level server."""
    # Create mock class and instance
    mock = Mock()
    mock_instance = Mock()
    lowlevel = mock_instance._lowlevel_server

    # Set up the low-level server's create_initialization_options and run
    lowlevel.create_initialization_options.return_value = {
        'protocolVersion': '1.0',
        'serverInfo': {'name': 'test-server', 'version': '1.0'},
    }
    lowlevel.run = AsyncMock()

    # Make the mock class return our mock instance
    mock.return_value = mock_instance

    with patch('mcp_server_opensearch.server_factory.OpenSearchMCPServer', mock):
        yield mock  # Return the mock class


def _real_server(mock_server):
    """A real OpenSearchMCPServer built from the enabled tools serve() passed to the class."""
    from mcp_server_opensearch.mcp_server import OpenSearchMCPServer

    (enabled_tools,) = mock_server.call_args.args
    return OpenSearchMCPServer(enabled_tools)


@pytest.fixture
def mock_stdio():
    """Create mock stdio reader and writer."""
    reader = AsyncMock()
    writer = AsyncMock()

    @asynccontextmanager
    async def mock_context():
        yield reader, writer

    with patch('mcp_server_opensearch.stdio_server.stdio_server', mock_context):
        yield reader, writer


@pytest.fixture
def mock_tool_registry():
    """Replace the tool registry with test data."""

    async def mock_get_tools(*args, **kwargs):
        return MOCK_TOOL_REGISTRY

    with patch(
        'mcp_server_opensearch.server_factory.get_tools',
        side_effect=mock_get_tools,
    ):
        yield MOCK_TOOL_REGISTRY


@pytest.mark.asyncio
async def test_serve_initialization(
    mock_server, mock_stdio, mock_tool_registry, mock_generate_tools
):
    """Test server initialization."""
    reader, writer = mock_stdio

    # Start the server
    from mcp_server_opensearch.stdio_server import serve

    asyncio.create_task(serve())

    # Wait for the server to start
    await asyncio.sleep(0.1)

    # Verify server was initialized correctly, from the enabled tools
    mock_server.assert_called_once_with(MOCK_TOOL_REGISTRY)
    lowlevel = mock_server.return_value._lowlevel_server
    lowlevel.create_initialization_options.assert_called_once()
    lowlevel.run.assert_awaited_once_with(
        reader,
        writer,
        lowlevel.create_initialization_options.return_value,
        raise_exceptions=True,
    )


@pytest.mark.asyncio
async def test_list_tools(mock_server, mock_stdio, mock_tool_registry, mock_generate_tools):
    """Test list_tools functionality."""
    reader, writer = mock_stdio

    # Start the server
    from mcp_server_opensearch.stdio_server import serve

    asyncio.create_task(serve())
    await asyncio.sleep(0.1)

    # The server serve() built lists the enabled tools
    tools = await _real_server(mock_server).list_tools()
    assert len(tools) == 1
    assert tools[0].name == 'test_tool'
    assert tools[0].description == 'Test tool'
    assert tools[0].input_schema == {'type': 'object', 'properties': {}}


@pytest.mark.asyncio
async def test_call_tool(mock_server, mock_stdio, mock_tool_registry, mock_generate_tools):
    """Test call_tool functionality."""
    reader, writer = mock_stdio

    # Start the server
    from mcp_server_opensearch.stdio_server import serve

    asyncio.create_task(serve())
    await asyncio.sleep(0.1)

    # Test the call_tool of the server serve() built
    result = await _real_server(mock_server).call_tool('test_tool', {})
    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].text == 'test result'


@pytest.mark.asyncio
async def test_server_error_handling(mock_server, mock_stdio, mock_tool_registry):
    """Test server error handling."""
    reader, writer = mock_stdio

    # Simulate an error in the server
    mock_server.return_value._lowlevel_server.run.side_effect = Exception('Test error')

    # Start the server
    from mcp_server_opensearch.stdio_server import serve

    with pytest.raises(Exception, match='Test error'):
        await serve()


@pytest.mark.asyncio
async def test_tool_execution_error(
    mock_server, mock_stdio, mock_tool_registry, mock_generate_tools
):
    """Test tool execution error handling."""
    reader, writer = mock_stdio

    # Modify mock tool to raise an error
    mock_tool_registry['test_tool']['function'].side_effect = Exception('Tool error')

    # Start the server
    from mcp_server_opensearch.stdio_server import serve

    asyncio.create_task(serve())
    await asyncio.sleep(0.1)

    # Test tool execution error: the error text comes back as an isError result
    try:
        result = await _real_server(mock_server).call_tool('test_tool', {})
    finally:
        mock_tool_registry['test_tool']['function'].side_effect = None
    assert result.is_error is True
    assert [c.text for c in result.content] == ['Tool error']
