# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSearch MCP server: a thin subclass of the MCP SDK's MCPServer.

This module is the only place that reaches MCPServer's low-level server, for two permanent
uses, each commented where it is made: taking out the handlers this server does not serve
(the SDK has no public unregister API), and run_streams' run with raise_exceptions=True
(exact stdio parity).
"""

from mcp.server.mcpserver import Context, MCPServer, RequestStateBoundary
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent, Tool
from mcp_server_opensearch.server_instructions import get_server_instructions
from typing import Any


SERVER_NAME = 'opensearch-mcp-server'

# Handlers MCPServer.__init__ registers that this server does not serve. Each one would add
# a `prompts` or `resources` capability and, at protocol 2026-07-28, turn `listChanged` on
# (subscriptions/listen). Without them these methods answer -32601 "Method not found".
_UNSERVED_METHODS = (
    'prompts/list',
    'prompts/get',
    'resources/list',
    'resources/templates/list',
    'resources/read',
    'subscriptions/listen',
)


class OpenSearchMCPServer(MCPServer):
    """MCPServer that lists and calls the enabled OpenSearch tools."""

    def __init__(self, enabled_tools: dict):
        """Build the server from the enabled tools, keeping them for call_tool."""
        # No tools go to the SDK's ToolManager: it keys them by name, so of two enabled tools
        # with the same listed name it would keep one and log "Tool already exists", where
        # the low-level Server listed both. list_tools and call_tool are overridden instead.
        super().__init__(
            SERVER_NAME,
            # Server instructions guide the LLM on dynamic connection params (single mode only)
            instructions=get_server_instructions(),
        )
        # Permanent: the SDK has no public way to unregister a handler, so the ones this
        # server does not serve are taken out of the low-level server's table directly.
        for method in _UNSERVED_METHODS:
            self._lowlevel_server._request_handlers.pop(method, None)
        # MCPServer also adds a RequestStateBoundary, which answers a tools/call carrying any
        # `requestState` with -32602 "Invalid or expired requestState"; the low-level Server
        # ran the tool. This server never returns `input_required`, so there is no state to
        # seal: drop it to keep the call path as it was.
        self.middleware[:] = [
            m for m in self.middleware if not isinstance(m, RequestStateBoundary)
        ]
        self.enabled_tools = enabled_tools

    async def run_streams(self, read_stream: Any, write_stream: Any) -> None:
        """Serve one connection over the given streams (stdio) until the read side closes."""
        # Permanent, on purpose: the one low-level run call, kept for exact stdio parity. The
        # public run_stdio_async() drops raise_exceptions (it runs without raising), so a
        # non-MCP exception inside an SDK handler would become a JSON-RPC error instead of
        # ending the process as it does today. Replacing this call is a separate, visible
        # behaviour change.
        await self._lowlevel_server.run(
            read_stream,
            write_stream,
            self._lowlevel_server.create_initialization_options(),
            raise_exceptions=True,
        )

    async def list_tools(self) -> list[Tool]:
        """One Tool per enabled tool, in the dict's order, duplicate listed names included.

        Overridden, as the low-level Server's tools/list handler was written: the name is the
        display_name, else the registry key; the description and input_schema are as given,
        with no outputSchema. Nothing is validated at startup: a field of the wrong type, such
        as a description the CLI parsed as a number, still starts the server, and tools/list
        then fails when this Tool is validated (-32602).
        """
        return [
            Tool(
                name=info.get('display_name', key),
                description=info['description'],
                input_schema=info['input_schema'],
            )
            for key, info in self.enabled_tools.items()
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
    ) -> CallToolResult:
        """Call an enabled tool through execute_tool, with the request in request_context_var."""
        from mcp_server_opensearch import tool_executor
        from mcp_server_opensearch.client_context import request_context_var

        token = request_context_var.set(_request_of(context))
        try:
            return await tool_executor.execute_tool(name, arguments or {}, self.enabled_tools)
        except MCPError:
            raise
        except Exception as e:
            return tool_executor._build_call_tool_result(
                [
                    TextContent(
                        type='text',
                        text=str(e),
                    )
                ],
                is_error=True,
            )
        finally:
            request_context_var.reset(token)


def _request_of(context: Context | None) -> Any:
    """The transport request of the call's context, or None (stdio, or no request context)."""
    if context is None:
        return None
    try:
        request_context = context.request_context
    except ValueError:  # a Context made outside a request
        return None
    return request_context.request
