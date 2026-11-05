# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP request handlers shared by the stdio and streaming transports."""

from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)
from tools.ui_resources import (
    UI_RESOURCE_META,
    UI_RESOURCES,
    apps_enabled,
    bound_resource_uris,
    load_ui_html,
)


def build_handlers(enabled_tools: dict) -> dict:
    """Return the on_* keyword arguments for Server(...) for the enabled tools."""
    apps = apps_enabled()

    async def _list_tools(ctx, params) -> ListToolsResult:
        tools = []
        for tool_name, tool_info in enabled_tools.items():
            meta = {'category': tool_info['category']} if tool_info.get('category') else {}
            if apps and tool_info.get('ui_resource_uri') in UI_RESOURCES:
                meta['ui'] = {'resourceUri': tool_info['ui_resource_uri']}
            tools.append(
                Tool(
                    name=tool_info.get('display_name', tool_name),
                    description=tool_info['description'],
                    input_schema=tool_info['input_schema'],
                    annotations=tool_info.get('annotations'),
                    meta=meta or None,  # type: ignore[call-arg]
                )
            )
        return ListToolsResult(tools=tools)

    async def _call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
        from mcp_server_opensearch.client_context import request_context_var
        from mcp_server_opensearch.tool_executor import _build_call_tool_result, execute_tool

        token = request_context_var.set(ctx.request)
        try:
            return await execute_tool(params.name, params.arguments or {}, enabled_tools)
        except MCPError:
            raise
        except Exception as e:
            return _build_call_tool_result(
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

    handlers = {'on_list_tools': _list_tools, 'on_call_tool': _call_tool}
    if not apps:
        return handlers

    uris = bound_resource_uris(enabled_tools)

    async def _list_resources(ctx, params) -> ListResourcesResult:
        return ListResourcesResult(
            resources=[
                Resource(
                    uri=uri,
                    name=UI_RESOURCES[uri]['name'],
                    description=UI_RESOURCES[uri]['description'],
                    mime_type=APP_MIME_TYPE,
                    meta=UI_RESOURCE_META,  # type: ignore[call-arg]
                )
                for uri in uris
            ]
        )

    async def _read_resource(ctx, params: ReadResourceRequestParams) -> ReadResourceResult:
        uri = str(params.uri)
        if uri not in uris:
            raise MCPError(code=INVALID_PARAMS, message=f'Resource not found: {uri}')
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mime_type=APP_MIME_TYPE,
                    text=load_ui_html(uri),
                    meta=UI_RESOURCE_META,  # type: ignore[call-arg]
                )
            ]
        )

    handlers['on_list_resources'] = _list_resources
    handlers['on_read_resource'] = _read_resource
    return handlers


def configure_apps(server: Server) -> None:
    """Advertise the MCP Apps extension when OPENSEARCH_MCP_APPS_ENABLED is on."""
    if apps_enabled():
        server.extensions[EXTENSION_ID] = {}
