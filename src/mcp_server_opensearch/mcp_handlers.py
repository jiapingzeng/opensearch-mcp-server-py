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
    dashboards_base_url,
    finalize_view_payload,
    is_view_bound,
    load_ui_html,
    split_presentation_args,
    tool_ui_meta,
    view_input_schema,
)


def _view_bound_tool(enabled_tools: dict, display_name: str) -> dict | None:
    for key, tool_info in enabled_tools.items():
        if tool_info.get('display_name', key) == display_name:
            return tool_info if is_view_bound(tool_info) else None
    return None


def build_handlers(enabled_tools: dict) -> dict:
    """Return the on_* keyword arguments for Server(...) for the enabled tools."""
    apps = apps_enabled()
    if apps:
        dashboards_base_url()  # fail at startup on a malformed URL

    async def _list_tools(ctx, params) -> ListToolsResult:
        tools = []
        for tool_name, tool_info in enabled_tools.items():
            meta = {'category': tool_info['category']} if tool_info.get('category') else {}
            input_schema = tool_info['input_schema']
            if apps:
                meta['ui'] = tool_ui_meta(tool_name, tool_info)
                if is_view_bound(tool_info):
                    input_schema = view_input_schema(input_schema)
            tools.append(
                Tool(
                    name=tool_info.get('display_name', tool_name),
                    description=tool_info['description'],
                    input_schema=input_schema,
                    annotations=tool_info.get('annotations'),
                    meta=meta or None,  # type: ignore[call-arg]
                )
            )
        return ListToolsResult(tools=tools)

    async def _call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
        from mcp_server_opensearch.client_context import request_context_var
        from mcp_server_opensearch.tool_executor import _build_call_tool_result, execute_tool

        arguments = params.arguments or {}
        view_tool = _view_bound_tool(enabled_tools, params.name) if apps else None
        presentation: dict = {}
        token = request_context_var.set(ctx.request)
        try:
            if view_tool:
                arguments, presentation = split_presentation_args(arguments)
            result = await execute_tool(params.name, arguments, enabled_tools)
        except MCPError:
            raise
        except Exception as e:
            result = _build_call_tool_result(
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
        if view_tool:
            structured = finalize_view_payload(
                result.structured_content,
                view_tool['ui_kind'],
                params.name,
                presentation,
                bool(result.is_error),
            )
            result = result.model_copy(update={'structured_content': structured})
        return result

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
