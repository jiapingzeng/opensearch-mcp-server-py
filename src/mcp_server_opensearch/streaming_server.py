# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import uvicorn
import contextlib
from typing import AsyncIterator
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from mcp_server_opensearch.clusters_information import load_clusters_from_yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from tools.tool_filter import get_tools
from tools.tool_generator import generate_tools_from_openapi
from starlette.types import Scope, Receive, Send
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from tools.tools import TOOL_REGISTRY
from tools.config import apply_custom_tool_config


async def create_mcp_server(
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict = None,
) -> Server:
    # Set the global profile if provided
    if profile:
        from opensearch.client import set_profile

        set_profile(profile)

    # Load clusters from YAML file
    if mode == 'multi':
        load_clusters_from_yaml(config_file_path)

    server = Server('opensearch-mcp-server')
    # Call tool generator
    await generate_tools_from_openapi()
    # Apply custom tool config (custom name and description)
    customized_registry = apply_custom_tool_config(
        TOOL_REGISTRY, config_file_path, cli_tool_overrides or {}
    )
    # Get enabled tools (tool filter)
    enabled_tools = get_tools(
        tool_registry=customized_registry, mode=mode, config_file_path=config_file_path
    )
    logging.info(f'Enabled tools: {list(enabled_tools.keys())}')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []
        for tool_name, tool_info in enabled_tools.items():
            tools.append(
                Tool(
                    name=tool_info.get('display_name', tool_name),
                    description=tool_info['description'],
                    inputSchema=tool_info['input_schema'],
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # Find the tool by its display name, which is what the client sees
        found_tool_key = None
        for key, tool_info in enabled_tools.items():
            if tool_info.get('display_name', key) == name:
                found_tool_key = key
                break

        if not found_tool_key:
            raise ValueError(f'Unknown or disabled tool: {name}')

        tool = enabled_tools.get(found_tool_key)
        parsed = tool['args_model'](**arguments)
        return await tool['function'](parsed)

    return server


class MCPStarletteApp:
    def __init__(self, mcp_server: Server, stateless: bool = False):
        self.mcp_server = mcp_server
        self.sse = SseServerTransport('/messages/')
        self.session_manager = StreamableHTTPSessionManager(
            app=self.mcp_server,
            event_store=None,
            json_response=False,
            stateless=stateless,
        )

    async def handle_sse(self, request: Request) -> None:
        async with self.sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await self.mcp_server.run(
                read_stream,
                write_stream,
                self.mcp_server.create_initialization_options(),
            )

        # Done to prevent 'NoneType' errors. For more details: https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/sse.py#L33-L37
        return Response()

    async def handle_health(self, request: Request) -> Response:
        return Response('OK', status_code=200)

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        """
        Context manager for session manager lifecycle.
        Ensures proper startup and shutdown of the session manager.
        """
        async with self.session_manager.run():
            logging.info('Application started with StreamableHTTP session manager!')
            try:
                yield
            finally:
                logging.info('Application shutting down...')

    async def handle_streamable_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle streamable HTTP requests."""
        await self.session_manager.handle_request(scope, receive, send)

    def create_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route('/sse', endpoint=self.handle_sse, methods=['GET']),
                Route('/health', endpoint=self.handle_health, methods=['GET']),
                Mount('/messages/', app=self.sse.handle_post_message),
                Mount('/mcp', app=self.handle_streamable_http),
            ],
            lifespan=self.lifespan,
        )


async def serve(
    host: str = '0.0.0.0',
    port: int = 9900,
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict = None,
    stateless: bool = False,
) -> None:
    mcp_server = await create_mcp_server(mode, profile, config_file_path, cli_tool_overrides)
    app_handler = MCPStarletteApp(mcp_server, stateless=stateless)
    app = app_handler.create_app()

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
    )
    server = uvicorn.Server(config)
    await server.serve()
