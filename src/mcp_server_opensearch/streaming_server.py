# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging
import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from mcp_server_opensearch.client_context import ClientNameMiddleware
from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
from mcp_server_opensearch.server_factory import build_server, prepare_enabled_tools
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

# Not used here: the characterization harness checks this module is bound to the registry.
from tools.tools import TOOL_REGISTRY  # noqa: F401
from typing import AsyncIterator


async def create_mcp_server(
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict | None = None,
) -> OpenSearchMCPServer:
    """Create and configure the MCP server instance."""
    return build_server(
        await prepare_enabled_tools(mode, profile, config_file_path, cli_tool_overrides)
    )


class _ASGIApp:
    """ASGI app object wrapping a handler, so Starlette's Route treats it as a raw ASGI endpoint."""

    def __init__(self, handler):
        self._handler = handler

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._handler(scope, receive, send)


# No DNS-rebinding protection, as before: a foreign Host or Origin is served like any other.
# MCPServer's apps turn the protection on by themselves when the host is 127.0.0.1.
_NO_DNS_REBINDING_PROTECTION = TransportSecuritySettings(enable_dns_rebinding_protection=False)


class MCPStarletteApp:
    """Starlette application wrapper for the MCP server."""

    def __init__(self, mcp_server: OpenSearchMCPServer, stateless: bool = True):
        """Initialize the MCP Starlette application."""
        self.mcp_server = mcp_server
        # MCPServer's own apps are not mounted (they would answer '/mcp/' with 307 and have no
        # '/health'): only their session manager and SSE routes go into create_app's table.
        mcp_server.streamable_http_app(
            stateless_http=stateless,
            json_response=False,
            transport_security=_NO_DNS_REBINDING_PROTECTION,
        )
        self.session_manager = mcp_server.session_manager
        # GET '/sse' (runs the server over the event stream) and the '/messages/' Mount.
        self.sse_route, self.messages_mount = mcp_server.sse_app(
            message_path='/messages/',
            transport_security=_NO_DNS_REBINDING_PROTECTION,
        ).routes

    async def handle_health(self, request: Request) -> Response:
        """Handle health check requests."""
        return Response('OK', status_code=200)

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager lifecycle.

        Ensures proper startup and shutdown of the session manager.
        """
        from mcp_server_opensearch.logging_config import start_memory_monitor

        async with self.session_manager.run():
            logging.info('Application started with StreamableHTTP session manager!')
            monitor_task = start_memory_monitor()
            try:
                yield
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except (asyncio.CancelledError, Exception):
                    pass
                logging.info('Application shutting down...')

    async def handle_streamable_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle streamable HTTP requests."""
        await self.session_manager.handle_request(scope, receive, send)

    def create_app(self) -> Starlette:
        """Create the Starlette application with routes."""
        # Serve bare '/mcp' via Route (a Mount alone 307-redirects '/mcp' to '/mcp/'); Mount handles sub-paths.
        streamable_http_app = _ASGIApp(self.handle_streamable_http)
        app = Starlette(
            routes=[
                self.sse_route,
                Route('/health', endpoint=self.handle_health, methods=['GET']),
                self.messages_mount,
                Route('/mcp', endpoint=streamable_http_app),
                Mount('/mcp', app=streamable_http_app),
            ],
            lifespan=self.lifespan,
        )
        app.add_middleware(ClientNameMiddleware)
        return app


async def serve(
    host: str = '127.0.0.1',
    port: int = 9900,
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict | None = None,
    stateless: bool = True,
) -> None:
    """Start the MCP server in streaming HTTP mode."""
    mcp_server = await create_mcp_server(mode, profile, config_file_path, cli_tool_overrides)
    app_handler = MCPStarletteApp(mcp_server, stateless=stateless)
    app = app_handler.create_app()

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        timeout_graceful_shutdown=10,
    )
    server = uvicorn.Server(config)
    await server.serve()
