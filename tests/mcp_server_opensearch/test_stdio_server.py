# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pytest
from contextlib import asynccontextmanager
from mcp.server.lowlevel.server import Server
from mcp.types import TextContent
from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
from unittest.mock import ANY, AsyncMock, Mock, patch


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
def events():
    """What happens, in order: the stdio context, the memory monitor and the low-level run."""
    return []


@pytest.fixture
def lowlevel_run(events):
    """Stub the low-level Server.run that run_streams calls; the mock records its arguments."""

    async def run(*args, **kwargs):
        events.append('run')

    with patch.object(Server, 'run', autospec=True, side_effect=run) as mock:
        yield mock


@pytest.fixture
def initialization_options():
    """Stub the low-level Server's create_initialization_options: one object, recorded per call."""
    options = object()
    with patch.object(
        Server, 'create_initialization_options', autospec=True, return_value=options
    ) as mock:
        mock.options = options
        yield mock


@pytest.fixture
def served(lowlevel_run):
    """Spy on OpenSearchMCPServer.run_streams: the real one runs, and each call records self."""
    with patch.object(
        OpenSearchMCPServer,
        'run_streams',
        autospec=True,
        side_effect=OpenSearchMCPServer.run_streams,
    ) as mock:
        yield mock


def _served_server(served) -> OpenSearchMCPServer:
    """The server serve() actually ran: the self of its one run_streams call."""
    served.assert_awaited_once()
    server = served.call_args.args[0]
    assert isinstance(server, OpenSearchMCPServer)
    return server


@pytest.fixture
def monitor(events):
    """Replace start_memory_monitor with a task that waits until it is cancelled."""
    tasks = []

    def finished(task):
        events.append('monitor cancelled' if task.cancelled() else 'monitor ended')

    def start():
        events.append('monitor started')
        task = asyncio.create_task(asyncio.Event().wait())
        # Registered before serve() awaits the task, so it runs before serve() resumes.
        task.add_done_callback(finished)
        tasks.append(task)
        return task

    with patch('mcp_server_opensearch.logging_config.start_memory_monitor', side_effect=start):
        yield tasks


@pytest.fixture
def mock_stdio(events):
    """Create mock stdio reader and writer."""
    reader = AsyncMock()
    writer = AsyncMock()

    @asynccontextmanager
    async def mock_context():
        events.append('stdio entered')
        try:
            yield reader, writer
        finally:
            events.append('stdio exited')

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


async def _serve() -> None:
    """Run serve() to its end: the stubbed low-level run returns at once."""
    from mcp_server_opensearch.stdio_server import serve

    await asyncio.wait_for(serve(), timeout=10)


@pytest.mark.asyncio
async def test_serve_initialization(
    served,
    lowlevel_run,
    initialization_options,
    monitor,
    events,
    mock_stdio,
    mock_tool_registry,
    mock_generate_tools,
):
    """Test server initialization: from the enabled tools, monitored, raise_exceptions=True."""
    reader, writer = mock_stdio

    await _serve()

    # The server serve() runs is built from the enabled tools and runs on the stdio streams
    server = _served_server(served)
    assert server.enabled_tools is MOCK_TOOL_REGISTRY
    served.assert_awaited_once_with(server, reader, writer)
    # run_streams runs the server's low-level server, with that server's own initialization
    # options (the very object it returned) and raise_exceptions=True
    lowlevel = server._lowlevel_server
    initialization_options.assert_called_once_with(lowlevel)
    lowlevel_run.assert_awaited_once()
    assert lowlevel_run.call_args.args == (lowlevel, reader, writer, ANY)
    assert lowlevel_run.call_args.args[3] is initialization_options.options
    assert lowlevel_run.call_args.kwargs == {'raise_exceptions': True}
    # The memory monitor starts inside the stdio context and is cancelled after the run
    assert len(monitor) == 1
    assert monitor[0].cancelled()
    assert events == [
        'stdio entered',
        'monitor started',
        'run',
        'monitor cancelled',
        'stdio exited',
    ]


@pytest.mark.asyncio
async def test_list_tools(served, monitor, mock_stdio, mock_tool_registry, mock_generate_tools):
    """Test list_tools functionality."""
    await _serve()

    # The server serve() ran lists the enabled tools
    tools = await _served_server(served).list_tools()
    assert len(tools) == 1
    assert tools[0].name == 'test_tool'
    assert tools[0].description == 'Test tool'
    assert tools[0].input_schema == {'type': 'object', 'properties': {}}


@pytest.mark.asyncio
async def test_call_tool(served, monitor, mock_stdio, mock_tool_registry, mock_generate_tools):
    """Test call_tool functionality."""
    await _serve()

    # Test the call_tool of the server serve() ran
    result = await _served_server(served).call_tool('test_tool', {})
    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].text == 'test result'


@pytest.mark.asyncio
async def test_server_error_handling(
    served, lowlevel_run, monitor, events, mock_stdio, mock_tool_registry
):
    """Test server error handling: the run's exception ends serve(); the monitor is cancelled."""
    # Simulate an error in the server
    lowlevel_run.side_effect = Exception('Test error')

    with pytest.raises(Exception, match='Test error'):
        await _serve()

    assert len(monitor) == 1
    assert monitor[0].cancelled()
    assert events == ['stdio entered', 'monitor started', 'monitor cancelled', 'stdio exited']


@pytest.mark.asyncio
async def test_tool_execution_error(
    served, monitor, mock_stdio, mock_tool_registry, mock_generate_tools
):
    """Test tool execution error handling."""
    # Modify mock tool to raise an error
    mock_tool_registry['test_tool']['function'].side_effect = Exception('Tool error')

    await _serve()

    # Test tool execution error: the error text comes back as an isError result
    try:
        result = await _served_server(served).call_tool('test_tool', {})
    finally:
        mock_tool_registry['test_tool']['function'].side_effect = None
    assert result.is_error is True
    assert [c.text for c in result.content] == ['Tool error']
