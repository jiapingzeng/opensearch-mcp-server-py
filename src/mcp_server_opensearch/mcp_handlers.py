# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP request handlers shared by the stdio and streaming transports."""

from mcp.server import Server
from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID, client_supports_apps
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
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
    UI_RESOURCES,
    apps_enabled,
    bound_resource_uris,
    dashboards_base_url,
    finalize_view_payload,
    is_app_callable,
    is_view_bound,
    load_ui_html,
    payload_kind,
    resource_meta,
    takes_presentation_args,
    tool_ui_meta,
    view_input_schema,
)
from typing import Any


def client_apps_support(ctx: Any) -> bool | None:
    """Return True/False when the client's capabilities are known, None when they are not.

    Stateless streamable HTTP on protocols before 2026-07-28 never sees the client's
    capabilities, so those requests report None.
    """
    session = getattr(ctx, 'session', None)
    if getattr(session, 'client_capabilities', None) is None:
        return None
    return client_supports_apps(ctx)


def _advertises_apps(capabilities: Any) -> bool:
    extensions = capabilities.get('extensions') if isinstance(capabilities, dict) else None
    settings = extensions.get(EXTENSION_ID) if isinstance(extensions, dict) else None
    mime_types = settings.get('mimeTypes') if isinstance(settings, dict) else None
    return isinstance(mime_types, list) and APP_MIME_TYPE in mime_types


async def _gate_handshake(ctx, call_next):
    # A client known not to support Apps gets the flag-off capabilities in its handshake.
    result = await call_next(ctx)
    if ctx.method not in ('initialize', 'server/discover') or not isinstance(result, dict):
        return result
    if ctx.method == 'initialize':
        capabilities = (ctx.params or {}).get('capabilities')
        if not isinstance(capabilities, dict) or _advertises_apps(capabilities):
            return result
    elif client_apps_support(ctx) is not False:
        return result
    server_caps = dict(result.get('capabilities') or {})
    server_caps.pop('resources', None)
    extensions = {
        k: v for k, v in (server_caps.get('extensions') or {}).items() if k != EXTENSION_ID
    }
    if extensions:
        server_caps['extensions'] = extensions
    else:
        server_caps.pop('extensions', None)
    return {**result, 'capabilities': server_caps}


def _view_bound_tool(enabled_tools: dict, display_name: str) -> tuple[str, dict] | None:
    for key, tool_info in enabled_tools.items():
        if tool_info.get('display_name', key) == display_name:
            return (key, tool_info) if is_view_bound(tool_info) else None
    return None


def build_handlers(enabled_tools: dict) -> dict:
    """Return the on_* keyword arguments for Server(...) for the enabled tools."""
    apps = apps_enabled()
    if apps:
        dashboards_base_url()  # fail at startup on a malformed URL

    def active(ctx) -> bool:
        # Hybrid gating: on unless the client is known not to support Apps.
        return apps and client_apps_support(ctx) is not False

    async def _list_tools(ctx, params) -> ListToolsResult:
        apps_active = active(ctx)
        tools = []
        for tool_name, tool_info in enabled_tools.items():
            meta = {'category': tool_info['category']} if tool_info.get('category') else {}
            input_schema = tool_info['input_schema']
            if apps_active:
                meta['ui'] = tool_ui_meta(tool_name, tool_info)
                if takes_presentation_args(tool_info):
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

        view_tool = _view_bound_tool(enabled_tools, params.name) if active(ctx) else None
        token = request_context_var.set(ctx.request)
        try:
            return await execute_tool(
                params.name, params.arguments or {}, enabled_tools, view=view_tool is not None
            )
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
            if view_tool:
                key, info = view_tool
                marker = finalize_view_payload(
                    None,
                    payload_kind(info['ui_kind']),
                    params.name,
                    {},
                    True,
                    is_app_callable(key, info),
                )
                result = result.model_copy(update={'structured_content': marker})
            return result
        finally:
            request_context_var.reset(token)

    handlers = {'on_list_tools': _list_tools, 'on_call_tool': _call_tool}
    if not apps:
        return handlers

    uris = bound_resource_uris(enabled_tools)

    def require_active(ctx, method: str) -> None:
        if not active(ctx):
            raise MCPError(code=METHOD_NOT_FOUND, message='Method not found', data=method)

    async def _list_resources(ctx, params) -> ListResourcesResult:
        require_active(ctx, 'resources/list')
        return ListResourcesResult(
            resources=[
                Resource(
                    uri=uri,
                    name=UI_RESOURCES[uri]['name'],
                    description=UI_RESOURCES[uri]['description'],
                    mime_type=APP_MIME_TYPE,
                    meta=resource_meta(uri),  # type: ignore[call-arg]
                )
                for uri in uris
            ]
        )

    async def _read_resource(ctx, params: ReadResourceRequestParams) -> ReadResourceResult:
        require_active(ctx, 'resources/read')
        uri = str(params.uri)
        if uri not in uris:
            raise MCPError(code=INVALID_PARAMS, message=f'Resource not found: {uri}')
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mime_type=APP_MIME_TYPE,
                    text=load_ui_html(uri),
                    meta=resource_meta(uri),  # type: ignore[call-arg]
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
        server.middleware.append(_gate_handshake)
