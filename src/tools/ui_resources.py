# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP Apps (io.modelcontextprotocol/ui) views and structured result payloads."""

import copy
import json
import logging
import os
import re
from functools import lru_cache
from importlib import resources
from typing import Any
from urllib.parse import quote, urlparse


logger = logging.getLogger(__name__)

APPS_ENABLED_ENV = 'OPENSEARCH_MCP_APPS_ENABLED'
DASHBOARDS_URL_ENV = 'OPENSEARCH_DASHBOARDS_URL'
RESULTS_VIEW_URI = 'ui://opensearch/results'
STRUCTURED_CONTENT_VERSION = 1
STRUCTURED_CONTENT_MAX_BYTES = 1024 * 1024
STRUCTURED_CONTENT_MAX_ROWS = 500
STRUCTURED_CONTENT_MAX_COLUMNS = 100

OMITTED_TOO_LARGE = 'too_large'
OMITTED_UNSUPPORTED_FORMAT = 'unsupported_format'
OMITTED_ERROR = 'error'

# Tools a view may call back to keep the default visibility; everything else is model-only.
NEVER_APP_CALLABLE_TOOLS = frozenset({'GenericOpenSearchApiTool'})

NARRATIVE_MAX_CHARS = 2000
SUGGESTIONS_MAX_ITEMS = 3
SUGGESTION_MAX_CHARS = 120
PRESENTATION_PROPERTIES: dict[str, Any] = {
    'narrative': {
        'type': 'string',
        'maxLength': NARRATIVE_MAX_CHARS,
        'description': 'Optional. A short interpretation of the result, shown next to the '
        'result in hosts that render MCP Apps views. Does not change the query.',
    },
    'suggestions': {
        'type': 'array',
        'items': {'type': 'string', 'minLength': 1, 'maxLength': SUGGESTION_MAX_CHARS},
        'maxItems': SUGGESTIONS_MAX_ITEMS,
        'description': 'Optional. Up to 3 short follow-up questions shown as buttons in the '
        'results view; clicking one sends it as a user message.',
    },
}

# Static per build: hosts may prefetch and cache these, so no per-request data.
UI_RESOURCES: dict[str, dict[str, str]] = {
    RESULTS_VIEW_URI: {
        'name': 'opensearch-results',
        'description': 'Table and JSON view for OpenSearch search hits and PPL rows.',
        'html_file': 'results.html',
    },
}

UI_RESOURCE_META: dict[str, Any] = {'ui': {'prefersBorder': True}}


def apps_enabled() -> bool:
    """Return True when OPENSEARCH_MCP_APPS_ENABLED is set to true."""
    return os.getenv(APPS_ENABLED_ENV, '').strip().lower() == 'true'


def dashboards_base_url() -> str | None:
    """Return OPENSEARCH_DASHBOARDS_URL without a trailing slash, or None when unset."""
    raw = os.getenv(DASHBOARDS_URL_ENV, '').strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError(
            f'{DASHBOARDS_URL_ENV} must be an http(s) URL such as https://dashboards.example.com; '
            f'got {raw!r}.'
        )
    return raw.rstrip('/')


@lru_cache(maxsize=None)
def load_ui_html(uri: str) -> str:
    """Load the packaged HTML for a registered ui:// resource."""
    html_file = UI_RESOURCES[uri]['html_file']
    return resources.files('tools.ui').joinpath(html_file).read_text(encoding='utf-8')


def bound_resource_uris(enabled_tools: dict) -> list[str]:
    """Return registered UI resource URIs bound to at least one enabled tool."""
    uris = []
    for tool_info in enabled_tools.values():
        uri = tool_info.get('ui_resource_uri')
        if uri in UI_RESOURCES and uri not in uris:
            uris.append(uri)
    return uris


def is_view_bound(tool_info: dict) -> bool:
    """Return True when the tool renders through a registered UI resource."""
    return tool_info.get('ui_resource_uri') in UI_RESOURCES


def tool_ui_meta(tool_key: str, tool_info: dict) -> dict:
    """Return the _meta.ui entry for a tool in tools/list when Apps are enabled."""
    ui: dict[str, Any] = {}
    if is_view_bound(tool_info):
        ui['resourceUri'] = tool_info['ui_resource_uri']
    read_only = getattr(tool_info.get('annotations'), 'read_only_hint', None) is True
    if not (is_view_bound(tool_info) and read_only and tool_key not in NEVER_APP_CALLABLE_TOOLS):
        ui['visibility'] = ['model']
    return ui


def view_input_schema(input_schema: dict) -> dict:
    """Return the input schema with the optional view presentation arguments added."""
    schema = copy.deepcopy(input_schema)
    schema.setdefault('properties', {}).update(copy.deepcopy(PRESENTATION_PROPERTIES))
    return schema


def split_presentation_args(arguments: dict) -> tuple[dict, dict]:
    """Remove and validate narrative/suggestions; return (tool arguments, presentation)."""
    args = dict(arguments)
    presentation: dict[str, Any] = {}
    narrative = args.pop('narrative', None)
    suggestions = args.pop('suggestions', None)
    if narrative is not None:
        if not isinstance(narrative, str) or len(narrative) > NARRATIVE_MAX_CHARS:
            raise ValueError(
                f"'narrative' must be a string of at most {NARRATIVE_MAX_CHARS} characters."
            )
        if narrative.strip():
            presentation['narrative'] = narrative
    if suggestions is not None:
        valid = (
            isinstance(suggestions, list)
            and len(suggestions) <= SUGGESTIONS_MAX_ITEMS
            and all(
                isinstance(s, str) and 0 < len(s.strip()) and len(s) <= SUGGESTION_MAX_CHARS
                for s in suggestions
            )
        )
        if not valid:
            raise ValueError(
                f"'suggestions' must be a list of at most {SUGGESTIONS_MAX_ITEMS} non-empty "
                f'strings of at most {SUGGESTION_MAX_CHARS} characters each.'
            )
        if suggestions:
            presentation['suggestions'] = suggestions
    return args, presentation


def _rison_string(value: str) -> str:
    return "'" + value.replace('!', '!!').replace("'", "!'") + "'"


def discover_ppl_url(ppl_query: str) -> str | None:
    """Build an OpenSearch Dashboards Discover link that opens the PPL query on its source index."""
    base = dashboards_base_url()
    if not base:
        return None
    from mcp_server_opensearch.global_state import get_mode

    # One Dashboards URL cannot address every cluster in multi mode.
    if get_mode() != 'single':
        return None
    match = re.search(r'\bsource\s*=\s*([^\s|]+)', ppl_query or '', re.IGNORECASE)
    if not match:
        return None
    index = match.group(1)
    dataset = f'(id:{_rison_string("::" + index)},title:{_rison_string(index)},type:INDEXES)'
    state = (
        f'(filters:!(),query:(dataset:{dataset},language:PPL,query:{_rison_string(ppl_query)}))'
    )
    encoded = quote(state, safe="!'(),:*-._~")
    return f'{base}/app/data-explorer/discover#?_q={encoded}'


def _structured(kind: str, cluster: str, columns: list, rows: list, meta: dict) -> dict:
    return {
        'kind': kind,
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': cluster or '',
        'columns': columns,
        'rows': rows,
        'meta': meta,
    }


def omitted_marker(kind: str, reason: str, size: int | None = None) -> dict:
    """Return the small payload that tells the view why the full result is not included."""
    marker: dict[str, Any] = {
        'kind': kind,
        'version': STRUCTURED_CONTENT_VERSION,
        'omitted': reason,
    }
    if size is not None:
        marker['bytes'] = size
    return marker


def _agg_tables(aggregations: dict) -> list[dict]:
    # One table per top-level aggregation: bucket aggs become key/doc_count rows,
    # metric aggs become a single row of their scalar values.
    tables = []
    for name, agg in (aggregations or {}).items():
        if not isinstance(agg, dict):
            continue
        buckets = agg.get('buckets')
        if isinstance(buckets, dict):
            buckets = [{'key': k, **v} for k, v in buckets.items() if isinstance(v, dict)]
        if isinstance(buckets, list):
            columns = ['key', 'doc_count']
            rows = []
            for bucket in buckets:
                row = {'key': bucket.get('key_as_string', bucket.get('key'))}
                row['doc_count'] = bucket.get('doc_count')
                for sub_name, sub in bucket.items():
                    if isinstance(sub, dict) and 'value' in sub:
                        row[sub_name] = sub.get('value_as_string', sub['value'])
                        if sub_name not in columns:
                            columns.append(sub_name)
                rows.append([row.get(c) for c in columns])
            tables.append({'name': name, 'columns': columns, 'rows': rows})
        else:
            scalars = {k: v for k, v in agg.items() if not isinstance(v, (dict, list))}
            if scalars:
                tables.append(
                    {'name': name, 'columns': list(scalars), 'rows': [list(scalars.values())]}
                )
    return tables


def search_hits_payload(response: Any, cluster: str, index: str) -> dict | None:
    """Normalize an OpenSearch search response into the search_hits view shape."""
    if not isinstance(response, dict) or not isinstance(response.get('hits'), dict):
        return None
    hits = response['hits'].get('hits') or []
    columns = ['_index', '_id', '_score']
    for hit in hits:
        for key in (hit.get('_source') or {}).keys():
            if key not in columns:
                columns.append(key)
    rows = []
    for hit in hits:
        source = hit.get('_source') or {}
        rows.append(
            [hit.get(c) if c in ('_index', '_id', '_score') else source.get(c) for c in columns]
        )
    total = response['hits'].get('total')
    meta = {
        'index': index,
        'total': total.get('value') if isinstance(total, dict) else total,
        'total_relation': total.get('relation') if isinstance(total, dict) else None,
        'took': response.get('took'),
        'timed_out': response.get('timed_out'),
        '_shards': response.get('_shards'),
    }
    if response.get('aggregations'):
        meta['aggregations'] = _agg_tables(response['aggregations'])
    # Discover cannot take query DSL, so the link opens the index without the query.
    link = discover_ppl_url(f'source = {index}')
    if link:
        meta['dashboards_url'] = link
    return _structured('search_hits', cluster, columns, rows, meta)


def ppl_rows_payload(response: Any, cluster: str, query: str) -> dict | None:
    """Normalize a PPL jdbc response into the ppl_rows view shape."""
    if not isinstance(response, dict) or 'datarows' not in response:
        return None
    schema = response.get('schema') or []
    columns = [c.get('name') for c in schema]
    meta = {
        'query': query,
        'total': response.get('total'),
        'size': response.get('size'),
        'column_types': [c.get('type') for c in schema],
    }
    link = discover_ppl_url(query)
    if link:
        meta['dashboards_url'] = link
    return _structured('ppl_rows', cluster, columns, response.get('datarows') or [], meta)


def _cap_table(columns: list, rows: list) -> tuple[list, list, bool]:
    truncated = False
    if len(rows) > STRUCTURED_CONTENT_MAX_ROWS:
        rows, truncated = rows[:STRUCTURED_CONTENT_MAX_ROWS], True
    if len(columns) > STRUCTURED_CONTENT_MAX_COLUMNS:
        columns = columns[:STRUCTURED_CONTENT_MAX_COLUMNS]
        rows = [row[:STRUCTURED_CONTENT_MAX_COLUMNS] for row in rows]
        truncated = True
    return columns, rows, truncated


def cap_structured_content(payload: dict) -> dict:
    """Apply the row, column and byte caps; return the payload or a too_large marker."""
    if 'omitted' not in payload and isinstance(payload.get('rows'), list):
        payload = dict(payload)
        meta = dict(payload.get('meta') or {})
        columns, rows, truncated = _cap_table(payload['columns'], payload['rows'])
        if truncated:
            meta.update(
                truncated=True,
                row_count=len(payload['rows']),
                column_count=len(payload['columns']),
            )
            if isinstance(meta.get('column_types'), list):
                meta['column_types'] = meta['column_types'][:STRUCTURED_CONTENT_MAX_COLUMNS]
        tables = []
        for table in meta.get('aggregations') or []:
            t_cols, t_rows, t_trunc = _cap_table(table['columns'], table['rows'])
            capped = {**table, 'columns': t_cols, 'rows': t_rows}
            if t_trunc:
                capped['truncated'] = True
            tables.append(capped)
        if tables:
            meta['aggregations'] = tables
        payload.update(columns=columns, rows=rows)
        if meta or 'meta' in payload:
            payload['meta'] = meta
    size = len(json.dumps(payload, default=str, separators=(',', ':')).encode('utf-8'))
    if size > STRUCTURED_CONTENT_MAX_BYTES:
        logger.info(f'Omitting structuredContent: {size} bytes exceeds cap')
        extra = {k: payload[k] for k in ('tool', 'narrative', 'suggestions') if k in payload}
        return {**omitted_marker(payload.get('kind', ''), OMITTED_TOO_LARGE, size), **extra}
    return payload


def finalize_view_payload(
    structured: dict | None,
    kind: str,
    tool_name: str,
    presentation: dict,
    is_error: bool,
) -> dict:
    """Return the structuredContent for a view-bound tool call: payload or omitted marker."""
    if is_error:
        payload = omitted_marker(kind, OMITTED_ERROR)
    elif structured is None:
        payload = omitted_marker(kind, OMITTED_UNSUPPORTED_FORMAT)
    else:
        payload = dict(structured)
    payload['tool'] = tool_name
    payload.update(presentation)
    return cap_structured_content(payload)
