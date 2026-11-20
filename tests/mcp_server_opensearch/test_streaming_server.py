# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import anyio
import contextlib
import httpx
import json
import pytest
import pytest_asyncio
from mcp.types import TextContent
from starlette.responses import Response
from unittest.mock import ANY, AsyncMock, Mock, patch


@pytest.fixture(autouse=True)
def patch_opensearch_version():
    """Mock OpenSearch client and version check."""
    from semver import Version
    from unittest.mock import AsyncMock

    mock_client = Mock()
    mock_client.info = AsyncMock(return_value={'version': {'number': '3.0.0'}})

    async def mock_get_version(*args, **kwargs):
        return Version.parse('2.9.0')

    with (
        patch('opensearch.helper.get_opensearch_version', side_effect=mock_get_version),
        patch('opensearch.client.initialize_client', return_value=mock_client),
        patch('mcp_server_opensearch.global_state.get_mode', return_value='single'),
    ):
        yield


class TestMCPServer:
    @pytest.fixture
    def mock_tool_registry(self):
        """Provides a mock tool registry for testing."""
        return {
            'test-tool': {
                'description': 'Test tool',
                'input_schema': {'type': 'object'},
                'args_model': Mock(),
                'function': AsyncMock(return_value=[TextContent(type='text', text='test result')]),
            }
        }

    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.server_factory.apply_custom_tool_config')
    @patch('mcp_server_opensearch.server_factory.get_tools')
    @patch('mcp_server_opensearch.server_factory.generate_tools_from_openapi')
    @patch('mcp_server_opensearch.server_factory.load_clusters_from_yaml')
    async def test_create_mcp_server(
        self,
        mock_load_clusters,
        mock_generate_tools,
        mock_get_tools,
        mock_apply_config,
        mock_tool_registry,
    ):
        """Test MCP server creation."""
        # Setup mocks
        mock_get_tools.return_value = mock_tool_registry
        mock_apply_config.return_value = mock_tool_registry  # Assume no changes for this test
        mock_generate_tools.return_value = None
        mock_load_clusters.return_value = None

        # Create server
        from mcp_server_opensearch.streaming_server import create_mcp_server

        server = await create_mcp_server(
            mode='single', config_file_path='some/path', cli_tool_overrides={'key': 'val'}
        )

        from mcp_server_opensearch.mcp_server import OpenSearchMCPServer

        assert isinstance(server, OpenSearchMCPServer)
        assert server.enabled_tools is mock_tool_registry
        assert server.name == 'opensearch-mcp-server'
        mock_generate_tools.assert_called_once()
        mock_apply_config.assert_called_once_with(ANY, 'some/path', {'key': 'val'})
        mock_get_tools.assert_called_once_with(
            tool_registry=mock_tool_registry,
            config_file_path='some/path',
        )

    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.server_factory.get_tools')
    @patch('mcp_server_opensearch.server_factory.generate_tools_from_openapi')
    @patch('mcp_server_opensearch.server_factory.load_clusters_from_yaml')
    async def test_list_tools(
        self, mock_load_clusters, mock_generate_tools, mock_get_tools, mock_tool_registry
    ):
        """Test listing available tools."""
        # Setup mocks
        mock_get_tools.return_value = mock_tool_registry
        mock_generate_tools.return_value = None
        mock_load_clusters.return_value = None

        # Create server
        from mcp_server_opensearch.streaming_server import create_mcp_server

        server = await create_mcp_server()

        # Get the tools the server lists
        tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0].name == 'test-tool'
        assert tools[0].description == 'Test tool'
        assert tools[0].input_schema == {'type': 'object'}

    @pytest.mark.asyncio
    @patch('mcp_server_opensearch.server_factory.get_tools')
    @patch('mcp_server_opensearch.server_factory.generate_tools_from_openapi')
    @patch('mcp_server_opensearch.server_factory.load_clusters_from_yaml')
    async def test_call_tool(
        self, mock_load_clusters, mock_generate_tools, mock_get_tools, mock_tool_registry
    ):
        """Test calling the tool."""
        # Setup mocks
        mock_get_tools.return_value = mock_tool_registry
        mock_generate_tools.return_value = None
        mock_load_clusters.return_value = None
        mock_tool_registry['test-tool']['function'].return_value = [
            TextContent(type='text', text='result')
        ]

        # Create server
        from mcp_server_opensearch.streaming_server import create_mcp_server

        server = await create_mcp_server()

        # Test the server's call_tool
        result = await server.call_tool('test-tool', {'param': 'value'})

        assert result.is_error is False
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == 'result'
        mock_tool_registry['test-tool']['function'].assert_awaited_once()


class TestMCPStarletteApp:
    @pytest_asyncio.fixture
    async def app_handler(self):
        """Provides an MCPStarletteApp instance for testing."""
        from mcp_server_opensearch.streaming_server import MCPStarletteApp, create_mcp_server

        # Mock dependencies
        with (
            patch(
                'mcp_server_opensearch.server_factory.get_tools',
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                'mcp_server_opensearch.server_factory.generate_tools_from_openapi',
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                'mcp_server_opensearch.server_factory.load_clusters_from_yaml',
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            server = await create_mcp_server()
            return MCPStarletteApp(server)

    def test_transports_use_the_mcp_servers_session_manager(self, app_handler):
        """Streamable HTTP runs on OpenSearchMCPServer's own session manager, stateless, SSE replies."""
        session_manager = app_handler.session_manager
        assert session_manager is app_handler.mcp_server.session_manager
        assert session_manager.app is app_handler.mcp_server._lowlevel_server
        assert session_manager.stateless is True
        assert session_manager.json_response is False
        assert session_manager.security_settings.enable_dns_rebinding_protection is False

    def test_stateless_false_reaches_the_session_manager(self):
        """MCPStarletteApp(stateless=False) builds a stateful session manager."""
        from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
        from mcp_server_opensearch.streaming_server import MCPStarletteApp

        app_handler = MCPStarletteApp(OpenSearchMCPServer({}), stateless=False)
        assert app_handler.session_manager.stateless is False

    def test_sse_routes_are_picked_by_path(self):
        """The '/sse' Route and '/messages/' Mount are picked by path, whatever else sse_app has."""
        from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
        from mcp_server_opensearch.streaming_server import MCPStarletteApp
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        server = OpenSearchMCPServer({})
        real_sse_app = server.sse_app

        def sse_app_with_more_routes(**kwargs):
            routes = real_sse_app(**kwargs).routes
            extra_before = Route('/extra', endpoint=lambda request: Response())
            extra_after = Mount('/other', app=Starlette())
            return Starlette(routes=[extra_before, *reversed(routes), extra_after])

        server.sse_app = sse_app_with_more_routes
        app_handler = MCPStarletteApp(server)

        assert isinstance(app_handler.sse_route, Route)
        assert app_handler.sse_route.path == '/sse'
        assert isinstance(app_handler.messages_mount, Mount)
        assert app_handler.messages_mount.path == '/messages'
        # The app's route table and order are as before: the extra routes are not in it.
        paths = [(type(r).__name__, r.path) for r in app_handler.create_app().routes]
        assert paths == [
            ('Route', '/sse'),
            ('Route', '/health'),
            ('Mount', '/messages'),
            ('Route', '/mcp'),
            ('Mount', '/mcp'),
        ]

    def test_create_app(self, app_handler):
        """Test Starlette application creation and configuration."""
        from starlette.routing import Mount, Route

        app = app_handler.create_app()
        assert len(app.routes) == 5

        # Check routes, in order: '/sse', '/health', '/messages/', then '/mcp' twice.
        assert app.routes[0].path == '/sse' and isinstance(app.routes[0], Route)
        assert app.routes[0].methods == {'GET', 'HEAD'}
        assert app.routes[0] is app_handler.sse_route
        assert app.routes[1].path == '/health' and isinstance(app.routes[1], Route)
        assert app.routes[1].methods == {'GET', 'HEAD'}
        assert app.routes[2].path == '/messages' and isinstance(app.routes[2], Mount)
        assert app.routes[2] is app_handler.messages_mount
        # Bare '/mcp' is a Route (served directly, no 307); sub-paths go through the Mount.
        assert app.routes[3].path == '/mcp' and isinstance(app.routes[3], Route)
        assert app.routes[4].path == '/mcp' and isinstance(app.routes[4], Mount)
        assert [m.cls.__name__ for m in app.user_middleware] == ['ClientNameMiddleware']

    @pytest.mark.asyncio
    async def test_mcp_endpoint_no_redirect(self, app_handler):
        """Bare '/mcp' is served directly without a 307 redirect to '/mcp/' (issue #271)."""
        from starlette.responses import PlainTextResponse
        from starlette.testclient import TestClient

        async def fake_handle_request(scope, receive, send):
            await PlainTextResponse('ok')(scope, receive, send)

        app_handler.session_manager.handle_request = fake_handle_request
        app = app_handler.create_app()
        app.router.lifespan_context = None  # skip the real session manager startup
        client = TestClient(app)

        for method in ('GET', 'POST', 'DELETE'):
            response = client.request(method, '/mcp', follow_redirects=False)
            assert response.status_code == 200
            assert response.headers.get('location') is None

    @pytest.mark.asyncio
    async def test_handle_sse(self):
        """Test SSE connection handling: GET '/sse' runs the server over the SSE streams."""
        from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
        from mcp_server_opensearch.streaming_server import MCPStarletteApp

        mock_request = Mock()

        # Mock SSE connection context
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = (mock_read_stream, mock_write_stream)
        mock_context.__aexit__.return_value = None

        # The SSE transport MCPServer.sse_app builds, with connect_sse returning our context
        mock_sse = Mock()
        mock_sse.connect_sse = Mock(return_value=mock_context)

        server = OpenSearchMCPServer({})
        with patch(
            'mcp.server.mcpserver.server.SseServerTransport', return_value=mock_sse
        ) as mock_sse_class:
            app_handler = MCPStarletteApp(server)

        # Messages are posted to '/messages/', with no DNS-rebinding protection
        mock_sse_class.assert_called_once()
        assert mock_sse_class.call_args.args == ('/messages/',)
        security_settings = mock_sse_class.call_args.kwargs['security_settings']
        assert security_settings.enable_dns_rebinding_protection is False
        assert app_handler.messages_mount.app is mock_sse.handle_post_message

        # Mock the low-level server run method to return immediately
        lowlevel_server = server._lowlevel_server
        lowlevel_server.run = AsyncMock(return_value=None)
        lowlevel_server.create_initialization_options = Mock(return_value={})

        response = await app_handler.sse_route.endpoint(mock_request)

        # Verify SSE connection was established
        mock_sse.connect_sse.assert_called_once_with(
            mock_request.scope, mock_request.receive, mock_request._send
        )

        # Verify context manager was used
        mock_context.__aenter__.assert_called_once()
        mock_context.__aexit__.assert_called_once()

        # Verify server.run was called with correct arguments
        lowlevel_server.run.assert_called_once_with(mock_read_stream, mock_write_stream, {})

        # An empty response, so Starlette has one to send after the stream ends
        assert isinstance(response, Response)

    @pytest.mark.asyncio
    async def test_health(self, app_handler):
        """GET '/health' answers 200 'OK'."""
        async with _http_client(app_handler.create_app()) as client:
            response = await client.get('/health')

        assert response.status_code == 200
        assert response.text == 'OK'

    @pytest.mark.parametrize('path', ['/mcp', '/mcp/'])
    @pytest.mark.asyncio
    async def test_mcp_and_mcp_slash_answer_an_event_stream_without_session(
        self, app_handler, path
    ):
        """'/mcp' and '/mcp/' both answer an initialize with 200 text/event-stream, stateless."""
        async with _http_client(app_handler.create_app()) as client:
            response = await client.post(
                path, headers=MCP_POST_HEADERS, content=json.dumps(INITIALIZE)
            )

        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        assert 'mcp-session-id' not in response.headers
        assert '"serverInfo"' in response.text

    @pytest.mark.asyncio
    async def test_foreign_host_is_not_refused(self, app_handler):
        """A 'Host: example.com' request is served: no DNS-rebinding protection (no 421)."""
        headers = {**MCP_POST_HEADERS, 'host': 'example.com', 'origin': 'http://evil.example'}
        async with _http_client(app_handler.create_app()) as client:
            mcp_response = await client.post(
                '/mcp', headers=headers, content=json.dumps(INITIALIZE)
            )
            # Past the host check, the SSE message endpoint asks for its session_id.
            messages_response = await client.post('/messages/', headers=headers, content=b'{}')

        assert mcp_response.status_code == 200
        assert mcp_response.headers['content-type'].startswith('text/event-stream')
        assert messages_response.status_code == 400
        assert messages_response.text == 'session_id is required'


MCP_POST_HEADERS = {
    'accept': 'application/json, text/event-stream',
    'content-type': 'application/json',
}
INITIALIZE = {
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'protocolVersion': '2025-11-25',
        'capabilities': {},
        'clientInfo': {'name': 'unit-test', 'version': '0.0.0'},
    },
}


@contextlib.asynccontextmanager
async def _http_client(app):
    """An httpx client on the app, in-process, with the app's lifespan running."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url='http://127.0.0.1:9900'
        ) as client:
            with anyio.fail_after(10):
                yield client


@pytest.mark.asyncio
async def test_serve():
    """Test server startup and configuration."""
    from mcp_server_opensearch.streaming_server import serve

    # Mock uvicorn server
    mock_server = AsyncMock()
    mock_config = Mock()

    with (
        patch('uvicorn.Server', return_value=mock_server) as mock_server_class,
        patch('uvicorn.Config', return_value=mock_config) as mock_config_class,
        patch('mcp_server_opensearch.server_factory.get_tools', return_value={}),
        patch(
            'mcp_server_opensearch.server_factory.generate_tools_from_openapi', return_value=None
        ),
        patch('mcp_server_opensearch.server_factory.load_clusters_from_yaml', return_value=None),
    ):
        await serve(host='localhost', port=8000)

        # Verify config
        mock_config_class.assert_called_once()
        config_args = mock_config_class.call_args[1]
        assert config_args['host'] == 'localhost'
        assert config_args['port'] == 8000

        # Verify server started
        mock_server_class.assert_called_once_with(mock_config)
        mock_server.serve.assert_called_once()
