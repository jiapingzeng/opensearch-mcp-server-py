# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from mcp.server.stdio import stdio_server
from mcp_server_opensearch.server_factory import build_server, prepare_enabled_tools

# Not used here: the characterization harness checks this module is bound to the registry.
from tools.tools import TOOL_REGISTRY  # noqa: F401


# --- Server setup ---
async def serve(
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict | None = None,
) -> None:
    """Start the MCP server in stdio mode."""
    enabled_tools = await prepare_enabled_tools(
        mode, profile, config_file_path, cli_tool_overrides
    )
    # temporary: removed by 20-remove-lowlevel-server-path (stdio runs the low-level server)
    server = build_server(enabled_tools)._lowlevel_server

    # Start stdio-based MCP server
    from mcp_server_opensearch.logging_config import start_memory_monitor

    options = server.create_initialization_options()
    async with stdio_server() as (reader, writer):
        monitor_task = start_memory_monitor()
        try:
            await server.run(reader, writer, options, raise_exceptions=True)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except (asyncio.CancelledError, Exception):
                pass
