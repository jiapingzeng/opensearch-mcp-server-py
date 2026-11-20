# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSearch MCP server: a thin subclass of the MCP SDK's MCPServer."""

from mcp.server.mcpserver import Context, MCPServer, RequestStateBoundary
from mcp.server.mcpserver.tools import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
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


async def _never_called(**kwargs: Any) -> Any:
    """Tool.fn placeholder: OpenSearchMCPServer.call_tool runs every call itself."""
    raise NotImplementedError('OpenSearchMCPServer.call_tool handles tool calls')


def _build_tools(enabled_tools: dict) -> list[Tool]:
    """One SDK Tool per enabled tool, in the dict's order, with its schema as it is.

    Built without validation (model_construct), as the low-level server took the registry:
    a field of the wrong type, such as a description the CLI parsed as a number, still
    starts the server, and tools/list then fails when the listed Tool is validated (-32602).
    """
    return [
        Tool.model_construct(
            fn=_never_called,
            name=info.get('display_name', key),
            description=info['description'],
            parameters=info['input_schema'],
            # An empty FuncMetadata: no output schema, so no outputSchema in tools/list.
            fn_metadata=FuncMetadata(arg_model=ArgModelBase),
            is_async=True,
        )
        for key, info in enabled_tools.items()
    ]


class OpenSearchMCPServer(MCPServer):
    """MCPServer that lists and calls the enabled OpenSearch tools."""

    def __init__(self, enabled_tools: dict):
        """Build the server from the enabled tools, keeping them for call_tool."""
        # Server instructions guide the LLM on dynamic connection params (single mode only)
        super().__init__(
            SERVER_NAME,
            instructions=get_server_instructions(),
            tools=_build_tools(enabled_tools),
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
