# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""The startup pipeline and the MCP server construction shared by stdio and streaming HTTP."""

import logging
from mcp_server_opensearch.clusters_information import load_clusters_from_yaml
from mcp_server_opensearch.global_state import set_config_file_path, set_mode, set_profile
from mcp_server_opensearch.mcp_server import OpenSearchMCPServer
from tools.config import apply_custom_tool_config
from tools.tool_filter import get_tools
from tools.tool_generator import generate_tools_from_openapi
from tools.tools import TOOL_REGISTRY


async def prepare_enabled_tools(
    mode: str = 'single',
    profile: str = '',
    config_file_path: str = '',
    cli_tool_overrides: dict | None = None,
) -> dict:
    """Set the global state, build the tool registry and return the enabled tools."""
    # Set the global mode
    set_mode(mode)

    # Set the global profile if provided
    if profile:
        set_profile(profile)

    # Set the global config file path
    if config_file_path:
        set_config_file_path(config_file_path)

    # Load clusters from YAML file
    if mode == 'multi':
        await load_clusters_from_yaml(config_file_path)

    # Call tool generator
    await generate_tools_from_openapi()
    # Apply custom tool config (custom name and description)
    customized_registry = apply_custom_tool_config(
        TOOL_REGISTRY, config_file_path, cli_tool_overrides or {}
    )
    # Get enabled tools (tool filter)
    enabled_tools = await get_tools(
        tool_registry=customized_registry, config_file_path=config_file_path
    )
    logging.info(f'Enabled tools: {list(enabled_tools.keys())}')
    return enabled_tools


def build_server(enabled_tools: dict) -> OpenSearchMCPServer:
    """Create the MCP server that lists and calls the enabled tools."""
    return OpenSearchMCPServer(enabled_tools)
