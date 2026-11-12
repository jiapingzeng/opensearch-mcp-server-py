# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP request handlers shared by the stdio and streaming transports."""

import functools
import logging
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
from mcp_server_opensearch.server_instructions import get_apps_instructions
from pydantic import BaseModel
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
    takes_suggestions,
    tool_ui_meta,
    view_description,
    view_input_schema,
)
from typing import Any


logger = logging.getLogger(__name__)


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


async def _gate_handshake(ctx, call_next, apps_note: str | None = None):
    # A client known not to support Apps gets the flag-off handshake; others get apps_note too.
    result = await call_next(ctx)
    if ctx.method not in ('initialize', 'server/discover'):
        return result
    if isinstance(result, BaseModel):
        result = result.model_dump(by_alias=True, mode='json', exclude_none=True)
    elif not isinstance(result, dict):
        logger.warning(
            f'MCP Apps gating skipped for {ctx.method}: unexpected result type '
            f'{type(result).__name__}'
        )
        return result
    if ctx.method == 'initialize':
        capabilities = (ctx.params or {}).get('capabilities')
        applies = not isinstance(capabilities, dict) or _advertises_apps(capabilities)
    else:
        applies = client_apps_support(ctx) is not False
    if applies:
        if not apps_note:
            return result
        base = result.get('instructions')
        return {**result, 'instructions': f'{base}\n\n{apps_note}' if base else apps_note}
    server_caps = dict(result.get('capabilities') or {})
    # UI resources are the only resources this server serves, so the capability goes too.
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


def _present_tools(enabled_tools: dict, apps: bool) -> tuple[dict, set, dict]:
    """Drop present tools that can never be listed; return the tools, view-called keys and plans.

    With the flag off every present tool is dropped, so the server behaves as if they did not
    exist. With it on, a present tool stays only when all its required dependencies survived
    filtering; the tools its view calls are returned so they can be marked app-visible.
    """
    if not any(info.get('ui_present') for info in enabled_tools.values()):
        return enabled_tools, set(), {}
    if not apps:
        return {k: v for k, v in enabled_tools.items() if not v.get('ui_present')}, set(), {}
    from tools.present_tools import resolve_present_tools

    plans = resolve_present_tools(enabled_tools)
    kept = {k: v for k, v in enabled_tools.items() if k not in plans or plans[k]['listed']}
    called = {dep for plan in plans.values() if plan['listed'] for dep in plan['calls']}
    return kept, called, plans


def _not_available(name: str, info: dict, plan: dict) -> CallToolResult:
    from mcp_server_opensearch.tool_executor import _build_call_tool_result

    others = ', '.join(plan['names'].get(d, d) for d in info['ui_present']['requires'])
    return _build_call_tool_result(
        [
            TextContent(
                type='text',
                text=f'{name} shows an interactive view, which this client cannot display '
                f'(it does not support the MCP Apps extension {EXTENSION_ID}). Use {others} '
                'for text results instead.',
            )
        ],
        is_error=True,
    )


def build_handlers(enabled_tools: dict) -> dict:
    """Return the on_* keyword arguments for Server(...) for the enabled tools."""
    apps = apps_enabled()
    if apps:
        dashboards_base_url()  # fail at startup on a malformed URL
    enabled_tools, view_called, plans = _present_tools(enabled_tools, apps)
    for key, plan in plans.items():
        if not plan['listed']:
            logger.info(f'{key} is not listed: it needs {", ".join(plan["missing"])}')
    logger.info(f'Enabled tools: {list(enabled_tools.keys())}')

    mapping_names = {key: plan['names'].get('IndexMappingTool') for key, plan in plans.items()}

    def active(ctx) -> bool:
        # Hybrid gating: on unless the client is known not to support Apps.
        return apps and client_apps_support(ctx) is not False

    async def _list_tools(ctx, params) -> ListToolsResult:
        apps_active = active(ctx)
        # View notes only for a client known to render views; unknown gets the registry text.
        known = apps_active and client_apps_support(ctx) is True
        tools = []
        for tool_name, tool_info in enabled_tools.items():
            if tool_info.get('ui_present') and not apps_active:
                continue
            meta = {'category': tool_info['category']} if tool_info.get('category') else {}
            input_schema = tool_info['input_schema']
            if apps_active:
                meta['ui'] = tool_ui_meta(tool_name, tool_info, tool_name in view_called)
                if takes_presentation_args(tool_info):
                    input_schema = view_input_schema(input_schema, takes_suggestions(tool_info))
            tools.append(
                Tool(
                    name=tool_info.get('display_name', tool_name),
                    description=view_description(tool_info, known, mapping_names.get(tool_name))
                    if apps_active
                    else tool_info['description'],
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
        if not active(ctx):
            hidden = _view_bound_tool(enabled_tools, params.name)
            if hidden and hidden[1].get('ui_present'):
                return _not_available(params.name, hidden[1], plans[hidden[0]])
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


def apps_instructions(enabled_tools: dict) -> str | None:
    """Return the instructions paragraph for clients that render views, from the listed tools."""
    listed, _, _ = _present_tools(enabled_tools, True)

    def name(kind: str, present: bool = True) -> str | None:
        for key, info in listed.items():
            if bool(info.get('ui_present')) == present and info.get('ui_kind') == kind:
                return info.get('display_name', key)
        return None

    view_tools = [
        info.get('display_name', key)
        for key, info in listed.items()
        if is_view_bound(info) and not info.get('ui_present')
    ]
    return get_apps_instructions(
        name('cluster_overview'),
        name('index_explorer'),
        view_tools,
        any(takes_presentation_args(info) for info in listed.values()),
        name('index_mapping', present=False),
    )


def configure_apps(server: Server, enabled_tools: dict) -> None:
    """Advertise the MCP Apps extension and its instructions when OPENSEARCH_MCP_APPS_ENABLED is on."""
    if apps_enabled():
        server.extensions[EXTENSION_ID] = {}
        server.middleware.append(
            functools.partial(_gate_handshake, apps_note=apps_instructions(enabled_tools))
        )
