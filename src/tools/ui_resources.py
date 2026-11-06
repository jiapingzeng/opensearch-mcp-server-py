# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP Apps (io.modelcontextprotocol/ui) views and structured result payloads."""

import copy
import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from importlib import resources
from typing import Any
from urllib.parse import quote, urlparse


logger = logging.getLogger(__name__)

APPS_ENABLED_ENV = 'OPENSEARCH_MCP_APPS_ENABLED'
DASHBOARDS_URL_ENV = 'OPENSEARCH_DASHBOARDS_URL'
RESULTS_VIEW_URI = 'ui://opensearch/results'
TABLE_VIEW_URI = 'ui://opensearch/table'
HEALTH_VIEW_URI = 'ui://opensearch/cluster-health'
MAPPING_VIEW_URI = 'ui://opensearch/mapping'
EXPLAIN_VIEW_URI = 'ui://opensearch/explain'
STRUCTURED_CONTENT_VERSION = 1
STRUCTURED_CONTENT_MAX_BYTES = 1024 * 1024
STRUCTURED_CONTENT_MAX_ROWS = 500
STRUCTURED_CONTENT_MAX_COLUMNS = 100
STRUCTURED_CONTENT_MAX_NODES = 5000
STRUCTURED_CONTENT_MAX_DEPTH = 32

OMITTED_TOO_LARGE = 'too_large'
OMITTED_UNSUPPORTED_FORMAT = 'unsupported_format'
OMITTED_ERROR = 'error'

# Never app-callable, even if a view were bound to them: they can reach any API path.
ALWAYS_MODEL_ONLY_TOOLS = frozenset({'GenericOpenSearchApiTool'})

# Kinds whose tools take the intent/suggestions presentation arguments.
PRESENTATION_KINDS = frozenset({'search_hits', 'ppl_rows'})
INTENT_MAX_CHARS = 2000
SUGGESTIONS_MAX_ITEMS = 3
SUGGESTION_MAX_CHARS = 120
PRESENTATION_PROPERTIES: dict[str, Any] = {
    'intent': {
        'type': 'string',
        'maxLength': INTENT_MAX_CHARS,
        'description': 'Optional. Why this query is being run, in one or two sentences. Shown '
        'above the result in hosts that render MCP Apps views. Does not change the query.',
    },
    'suggestions': {
        'type': 'array',
        'items': {'type': 'string', 'minLength': 1, 'maxLength': SUGGESTION_MAX_CHARS},
        'maxItems': SUGGESTIONS_MAX_ITEMS,
        'description': 'Optional. Up to 3 short next steps you plan to take after this '
        'query, shown as buttons in the results view; clicking one sends it as a user message.',
    },
}

_SHARED_SCRIPTS = ['common.js']
_TABLE_SCRIPTS = ['table.js']
# Static per build: hosts may prefetch and cache these, so no per-request data.
UI_RESOURCES: dict[str, dict[str, Any]] = {
    RESULTS_VIEW_URI: {
        'name': 'opensearch-results',
        'title': 'OpenSearch results',
        'description': 'Table, charts and JSON view for OpenSearch search hits and PPL rows.',
        'scripts': _SHARED_SCRIPTS + _TABLE_SCRIPTS + ['charts.js', 'results.js'],
        'clipboard': True,
    },
    TABLE_VIEW_URI: {
        'name': 'opensearch-table',
        'title': 'OpenSearch table',
        'description': 'Sortable table for OpenSearch _cat results (indices, shards, nodes, '
        'allocation).',
        'scripts': _SHARED_SCRIPTS + _TABLE_SCRIPTS + ['cat_table.js'],
        'clipboard': True,
    },
    HEALTH_VIEW_URI: {
        'name': 'opensearch-cluster-health',
        'title': 'OpenSearch cluster health',
        'description': 'Cluster health card with status, node and shard counts.',
        'scripts': _SHARED_SCRIPTS + ['health.js'],
    },
    MAPPING_VIEW_URI: {
        'name': 'opensearch-mapping',
        'title': 'OpenSearch index mapping',
        'description': 'Collapsible, searchable tree of index mapping fields.',
        'scripts': _SHARED_SCRIPTS + ['tree.js', 'mapping.js'],
    },
    EXPLAIN_VIEW_URI: {
        'name': 'opensearch-explain',
        'title': 'OpenSearch score explanation',
        'description': 'Collapsible score breakdown for an explain response.',
        'scripts': _SHARED_SCRIPTS + ['tree.js', 'explain.js'],
    },
}


def resource_meta(uri: str) -> dict:
    """Return the _meta for a UI resource list entry and its read content item."""
    ui: dict[str, Any] = {'prefersBorder': True}
    if UI_RESOURCES[uri].get('clipboard'):
        ui['permissions'] = {'clipboardWrite': {}}
    return {'ui': ui}


def apps_enabled() -> bool:
    """Return True when OPENSEARCH_MCP_APPS_ENABLED is set to true."""
    return os.getenv(APPS_ENABLED_ENV, '').strip().lower() == 'true'


def dashboards_base_url() -> str | None:
    """Return OPENSEARCH_DASHBOARDS_URL without a trailing slash, or None when unset."""
    raw = os.getenv(DASHBOARDS_URL_ENV, '').strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if (
        parsed.scheme not in ('http', 'https')
        or not parsed.hostname
        or '@' in parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.params
    ):
        raise ValueError(
            f'{DASHBOARDS_URL_ENV} must be an http(s) URL with no credentials, query or '
            f'fragment, such as https://dashboards.example.com; got {raw!r}.'
        )
    return raw.rstrip('/')


@lru_cache(maxsize=None)
def load_ui_html(uri: str) -> str:
    """Assemble the packaged shell, shared CSS and the view's scripts into one HTML document."""
    spec = UI_RESOURCES[uri]
    ui = resources.files('tools.ui')
    shell = ui.joinpath('shell.html').read_text(encoding='utf-8')
    style = ui.joinpath('common.css').read_text(encoding='utf-8')
    script = '\n'.join(ui.joinpath(name).read_text(encoding='utf-8') for name in spec['scripts'])
    return (
        shell.replace('{{title}}', spec['title'])
        .replace('/*{{style}}*/', style)
        .replace('/*{{script}}*/', script)
    )


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
    return tool_info.get('ui_resource_uri') in UI_RESOURCES and tool_info.get('ui_kind') in (
        VIEW_BUILDERS
    )


def is_app_callable(tool_key: str, tool_info: dict) -> bool:
    """Return True when a view calls this tool back (refresh, paging, re-run)."""
    read_only = getattr(tool_info.get('annotations'), 'read_only_hint', None) is True
    return (
        is_view_bound(tool_info)
        and read_only
        and tool_info.get('ui_app_callable') is True
        and tool_key not in ALWAYS_MODEL_ONLY_TOOLS
    )


def tool_ui_meta(tool_key: str, tool_info: dict) -> dict:
    """Return the _meta.ui entry for a tool in tools/list when Apps are active."""
    ui: dict[str, Any] = {}
    if is_view_bound(tool_info):
        ui['resourceUri'] = tool_info['ui_resource_uri']
    ui['visibility'] = ['model', 'app'] if is_app_callable(tool_key, tool_info) else ['model']
    return ui


def takes_presentation_args(tool_info: dict) -> bool:
    """Return True when the tool's view shows intent and suggestions."""
    return is_view_bound(tool_info) and tool_info.get('ui_kind') in PRESENTATION_KINDS


def view_input_schema(input_schema: dict) -> dict:
    """Return the input schema with the optional view presentation arguments added."""
    schema = copy.deepcopy(input_schema)
    schema.setdefault('properties', {}).update(copy.deepcopy(PRESENTATION_PROPERTIES))
    return schema


def _has_control_chars(text: str, allowed: str = '') -> bool:
    return any(unicodedata.category(ch) in ('Cc', 'Cf') and ch not in allowed for ch in text)


def split_presentation_args(arguments: dict) -> tuple[dict, dict]:
    """Remove and validate intent/suggestions; return (tool arguments, presentation)."""
    args = dict(arguments)
    presentation: dict[str, Any] = {}
    intent = args.pop('intent', None)
    suggestions = args.pop('suggestions', None)
    if intent is not None:
        if (
            not isinstance(intent, str)
            or len(intent) > INTENT_MAX_CHARS
            or _has_control_chars(intent, allowed='\n\t')
        ):
            raise ValueError(
                f"'intent' must be a string of at most {INTENT_MAX_CHARS} characters with no "
                'control or formatting characters other than newlines and tabs.'
            )
        if intent.strip():
            presentation['intent'] = intent
    if suggestions is not None:
        valid = (
            isinstance(suggestions, list)
            and len(suggestions) <= SUGGESTIONS_MAX_ITEMS
            and all(
                isinstance(s, str)
                and 0 < len(s.strip())
                and len(s) <= SUGGESTION_MAX_CHARS
                and not _has_control_chars(s)
                for s in suggestions
            )
        )
        if not valid:
            raise ValueError(
                f"'suggestions' must be a list of at most {SUGGESTIONS_MAX_ITEMS} non-empty "
                f'strings of at most {SUGGESTION_MAX_CHARS} characters each, with no control '
                'or formatting characters.'
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
    index = match.group(1).replace('`', '')
    if not index:
        return None
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


def _query_from(query_dsl: Any) -> int:
    if isinstance(query_dsl, str):
        try:
            query_dsl = json.loads(query_dsl)
        except ValueError:
            return 0
    value = query_dsl.get('from') if isinstance(query_dsl, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def search_hits_payload(
    response: Any, cluster: str, index: str, query_dsl: Any = None
) -> dict | None:
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
        'from': _query_from(query_dsl),
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


# _cat sizes use 1024-based units; see the OpenSearch ByteSizeUnit.
_BYTE_UNITS = {'b': 1, 'kb': 1 << 10, 'mb': 1 << 20, 'gb': 1 << 30, 'tb': 1 << 40, 'pb': 1 << 50}
_BYTE_RE = re.compile(r'^(\d+(?:\.\d+)?)(b|kb|mb|gb|tb|pb)$')
# No leading zeros, so values such as "007" stay strings.
_NUMBER_RE = re.compile(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?$')
# Names, addresses and ordinals that look numeric but are labels, not quantities.
_IDENTIFIER_COLUMNS = frozenset(
    {
        'index',
        'node',
        'id',
        'ip',
        'host',
        'name',
        'shard',
        'prirep',
        'uuid',
        'alias',
        'pid',
        'port',
    }
)
_IDENTIFIER_SUFFIXES = ('.id', '_id', 'uuid', '.name', '.ip')


def _is_identifier(column: str) -> bool:
    name = str(column).lower()
    return name in _IDENTIFIER_COLUMNS or name.endswith(_IDENTIFIER_SUFFIXES)


def _column_type(column: str, values: list) -> str:
    present = [v for v in values if v is not None and v != '']
    if _is_identifier(column) or not present or not all(isinstance(v, str) for v in present):
        return 'string'
    if all(_NUMBER_RE.match(v) for v in present):
        return 'number'
    if all(_BYTE_RE.match(v.lower()) for v in present):
        return 'bytes'
    return 'string'


def _convert(value: Any, column_type: str) -> Any:
    if value is None or value == '':
        return None
    if column_type == 'number':
        number = float(value)
        return int(number) if number.is_integer() and '.' not in value else number
    if column_type == 'bytes':
        amount, unit = _BYTE_RE.match(value.lower()).groups()
        return round(float(amount) * _BYTE_UNITS[unit])
    return value


def cat_table_payload(entries: Any, cluster: str, title: str, source: str) -> dict | None:
    """Normalize _cat JSON rows into the cat_table view shape, typing numbers and byte sizes."""
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return None
    columns: list[str] = []
    for entry in entries:
        for key in entry:
            if key not in columns:
                columns.append(key)
    types = [_column_type(c, [e.get(c) for e in entries]) for c in columns]
    rows = [[_convert(e.get(c), t) for c, t in zip(columns, types)] for e in entries]
    meta = {'title': title, 'source': source, 'column_types': types}
    return _structured('cat_table', cluster, columns, rows, meta)


def list_indices_payload(response: Any, cluster: str, args: Any) -> dict | None:
    """Build the cat_table payload that matches what ListIndexTool printed."""
    if getattr(args, 'include_detail', False):
        if getattr(args, 'index', None):
            return None  # index details are a nested document, not a table
        return cat_table_payload(response, cluster, 'Indices', '_cat/indices')
    if not isinstance(response, list):
        return None
    names = [{'index': e['index']} for e in response if isinstance(e, dict) and 'index' in e]
    return cat_table_payload(names, cluster, 'Indices', '_cat/indices')


HEALTH_FIELDS = (
    'cluster_name',
    'status',
    'timed_out',
    'number_of_nodes',
    'number_of_data_nodes',
    'active_primary_shards',
    'active_shards',
    'relocating_shards',
    'initializing_shards',
    'unassigned_shards',
    'delayed_unassigned_shards',
    'number_of_pending_tasks',
    'number_of_in_flight_fetch',
    'task_max_waiting_in_queue_millis',
    'active_shards_percent_as_number',
)


def cluster_health_payload(response: Any, cluster: str) -> dict | None:
    """Normalize a _cluster/health response into the cluster_health view shape."""
    if not isinstance(response, dict) or 'status' not in response:
        return None
    health = {k: response[k] for k in HEALTH_FIELDS if k in response}
    indices = response.get('indices')
    payload = {
        'kind': 'cluster_health',
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': cluster or '',
        'health': health,
    }
    if isinstance(indices, dict):
        rows = [
            [name, info.get('status'), info.get('unassigned_shards')]
            for name, info in indices.items()
            if isinstance(info, dict)
        ]
        payload['columns'] = ['index', 'status', 'unassigned_shards']
        payload['rows'] = rows
    return payload


class _NodeBudget:
    def __init__(self, limit: int):
        self.left = limit
        self.used = 0
        self.truncated = False

    def take(self, depth: int = 0) -> bool:
        if self.left <= 0 or depth >= STRUCTURED_CONTENT_MAX_DEPTH:
            self.truncated = True
            return False
        self.left -= 1
        self.used += 1
        return True


_MAPPING_DETAIL_KEYS = (
    'analyzer',
    'search_analyzer',
    'normalizer',
    'format',
    'ignore_above',
    'index',
    'doc_values',
    'store',
    'enabled',
    'dynamic',
    'dimension',
    'method',
    'copy_to',
    'null_value',
    'scaling_factor',
)


def _mapping_nodes(
    properties: dict, prefix: str, budget: _NodeBudget, depth: int = 0
) -> list[dict]:
    nodes = []
    for name, spec in properties.items():
        if not isinstance(spec, dict) or not budget.take(depth):
            continue
        path = f'{prefix}{name}'
        node: dict[str, Any] = {'name': name, 'path': path, 'type': spec.get('type', 'object')}
        details = {k: spec[k] for k in _MAPPING_DETAIL_KEYS if k in spec}
        if details:
            node['details'] = details
        if isinstance(spec.get('fields'), dict):
            multi = []
            for sub, sub_spec in spec['fields'].items():
                if isinstance(sub_spec, dict) and budget.take(depth + 1):
                    entry: dict[str, Any] = {
                        'name': sub,
                        'path': f'{path}.{sub}',
                        'type': sub_spec.get('type', 'object'),
                    }
                    sub_details = {k: sub_spec[k] for k in _MAPPING_DETAIL_KEYS if k in sub_spec}
                    if sub_details:
                        entry['details'] = sub_details
                    multi.append(entry)
            node['fields'] = multi
        if isinstance(spec.get('properties'), dict):
            node['children'] = _mapping_nodes(spec['properties'], f'{path}.', budget, depth + 1)
        nodes.append(node)
    return nodes


def index_mapping_payload(response: Any, cluster: str) -> dict | None:
    """Normalize a GET _mapping response into the index_mapping view shape."""
    if not isinstance(response, dict) or not response:
        return None
    budget = _NodeBudget(STRUCTURED_CONTENT_MAX_NODES)
    indices = []
    for index, body in response.items():
        mappings = body.get('mappings') if isinstance(body, dict) else None
        if not isinstance(mappings, dict):
            return None
        indices.append(
            {
                'name': index,
                'dynamic': mappings.get('dynamic'),
                'fields': _mapping_nodes(mappings.get('properties') or {}, '', budget),
            }
        )
    meta: dict[str, Any] = {'field_count': budget.used}
    if budget.truncated:
        meta['truncated'] = True
    return {
        'kind': 'index_mapping',
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': cluster or '',
        'indices': indices,
        'meta': meta,
    }


def _explain_node(node: dict, budget: _NodeBudget, depth: int = 0) -> dict | None:
    if not isinstance(node, dict) or not budget.take(depth):
        return None
    out: dict[str, Any] = {
        'value': node.get('value'),
        'description': str(node.get('description', '')),
    }
    children = [_explain_node(d, budget, depth + 1) for d in node.get('details') or []]
    children = [c for c in children if c is not None]
    if children:
        out['details'] = children
    return out


def explain_payload(response: Any, cluster: str) -> dict | None:
    """Normalize an _explain response into the explain view shape."""
    if not isinstance(response, dict) or 'matched' not in response:
        return None
    budget = _NodeBudget(STRUCTURED_CONTENT_MAX_NODES)
    explanation = _explain_node(response.get('explanation') or {}, budget)
    meta: dict[str, Any] = {'node_count': budget.used}
    if budget.truncated:
        meta['truncated'] = True
    return {
        'kind': 'explain',
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': cluster or '',
        'index': response.get('_index'),
        'id': response.get('_id'),
        'matched': response.get('matched'),
        'explanation': explanation,
        'meta': meta,
    }


def _parsed_json(raw: Any) -> Any:
    # Generated tools return the response text verbatim; parse it back for the view.
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return raw


def _cluster(args: Any) -> str:
    return getattr(args, 'opensearch_cluster_name', '') or ''


# kind -> builder(raw tool response, parsed tool args) returning a payload or None.
VIEW_BUILDERS: dict[str, Any] = {
    'search_hits': lambda raw, args: search_hits_payload(
        raw, _cluster(args), getattr(args, 'index', ''), getattr(args, 'query_dsl', None)
    ),
    'ppl_rows': lambda raw, args: ppl_rows_payload(
        raw, _cluster(args), getattr(args, 'query', '')
    ),
    'cat_indices': lambda raw, args: list_indices_payload(raw, _cluster(args), args),
    'cat_shards': lambda raw, args: cat_table_payload(
        raw, _cluster(args), 'Shards', '_cat/shards'
    ),
    'cat_nodes': lambda raw, args: cat_table_payload(raw, _cluster(args), 'Nodes', '_cat/nodes'),
    'cat_allocation': lambda raw, args: cat_table_payload(
        raw, _cluster(args), 'Allocation', '_cat/allocation'
    ),
    'cluster_health': lambda raw, args: cluster_health_payload(_parsed_json(raw), _cluster(args)),
    'index_mapping': lambda raw, args: index_mapping_payload(raw, _cluster(args)),
    'explain': lambda raw, args: explain_payload(_parsed_json(raw), _cluster(args)),
}


def payload_kind(builder: str) -> str:
    """Return the structuredContent kind a builder produces, also used for its markers."""
    return 'cat_table' if builder.startswith('cat_') else builder


def _serialized_size(payload: dict) -> int:
    return len(json.dumps(payload, default=str, separators=(',', ':')).encode('utf-8'))


def _cap_table(columns: list, rows: list) -> tuple[list, list, bool]:
    truncated = False
    if len(rows) > STRUCTURED_CONTENT_MAX_ROWS:
        rows, truncated = rows[:STRUCTURED_CONTENT_MAX_ROWS], True
    if len(columns) > STRUCTURED_CONTENT_MAX_COLUMNS:
        columns = columns[:STRUCTURED_CONTENT_MAX_COLUMNS]
        rows = [row[:STRUCTURED_CONTENT_MAX_COLUMNS] for row in rows]
        truncated = True
    return columns, rows, truncated


def _too_large(payload: dict, size: int) -> dict:
    logger.info(f'Omitting structuredContent: {size} bytes exceeds cap')
    extra = {
        k: payload[k] for k in ('tool', 'refreshable', 'intent', 'suggestions') if k in payload
    }
    return {**omitted_marker(payload.get('kind', ''), OMITTED_TOO_LARGE, size), **extra}


def cap_structured_content(payload: dict) -> dict:
    """Apply the row, column and byte caps; halve the largest table to fit, else too_large."""
    has_rows = 'omitted' not in payload and isinstance(payload.get('rows'), list)
    if has_rows:
        payload = dict(payload)
        meta = dict(payload.get('meta') or {})
        original_rows, original_columns = len(payload['rows']), len(payload.get('columns') or [])
        columns, rows, truncated = _cap_table(payload.get('columns') or [], payload['rows'])
        if isinstance(meta.get('column_types'), list):
            meta['column_types'] = meta['column_types'][:STRUCTURED_CONTENT_MAX_COLUMNS]
        tables = []
        for table in meta.get('aggregations') or []:
            t_cols, t_rows, t_trunc = _cap_table(table['columns'], table['rows'])
            capped = {**table, 'columns': t_cols, 'rows': t_rows}
            if t_trunc:
                capped['truncated'] = True
            if len(t_rows) < len(table['rows']):
                capped['row_count'] = len(table['rows'])
            if len(t_cols) < len(table['columns']):
                capped['column_count'] = len(table['columns'])
            tables.append(capped)
        if tables:
            meta['aggregations'] = tables
        payload.update(columns=columns, rows=rows)
        size = _serialized_size({**payload, 'meta': meta})
        # Halve whichever table is largest, hits or an aggregation, until the payload fits.
        while size > STRUCTURED_CONTENT_MAX_BYTES:
            candidates = [t for t in [payload, *tables] if t['rows']]
            if not candidates:
                break
            target = max(candidates, key=lambda t: _serialized_size(t['rows']))
            if target is payload:
                truncated = True
            else:
                target.setdefault('row_count', len(target['rows']))
                target['truncated'] = True
            target['rows'] = target['rows'][: len(target['rows']) // 2]
            size = _serialized_size({**payload, 'meta': meta})
        if truncated:
            meta.update(truncated=True, row_count=original_rows, column_count=original_columns)
        if meta or 'meta' in payload:
            payload['meta'] = meta
    size = _serialized_size(payload)
    if size > STRUCTURED_CONTENT_MAX_BYTES:
        return _too_large(payload, size)
    return payload


def finalize_view_payload(
    structured: dict | None,
    kind: str,
    tool_name: str,
    presentation: dict,
    is_error: bool,
    refreshable: bool = False,
) -> dict:
    """Return the structuredContent for a view-bound tool call: payload or omitted marker."""
    if is_error:
        payload = omitted_marker(kind, OMITTED_ERROR)
    elif structured is None:
        payload = omitted_marker(kind, OMITTED_UNSUPPORTED_FORMAT)
    else:
        payload = dict(structured)
    payload['tool'] = tool_name
    if refreshable:
        payload['refreshable'] = True
    payload.update(presentation)
    return cap_structured_content(payload)


def view_structured_content(
    tool_key: str,
    tool_info: dict,
    tool_name: str,
    items: list,
    args: Any,
    presentation: dict,
    is_error: bool,
) -> dict:
    """Build the structuredContent for a view-bound tool from its raw response.

    A builder failure never fails the tool call: the view gets an unsupported_format marker.
    """
    builder = tool_info['ui_kind']
    kind = payload_kind(builder)
    refreshable = is_app_callable(tool_key, tool_info)
    structured = None
    if not is_error:
        try:
            first = items[0] if items else None
            raw = first.get('view_data') if isinstance(first, dict) else None
            structured = VIEW_BUILDERS[builder](raw, args) if raw is not None else None
        except Exception as e:
            logger.warning(f'Could not build the {kind} view payload: {type(e).__name__}: {e}')
            structured = None
    try:
        return finalize_view_payload(
            structured, kind, tool_name, presentation, is_error, refreshable
        )
    except Exception as e:
        logger.warning(f'Could not finalize the {kind} view payload: {type(e).__name__}: {e}')
        return {
            **omitted_marker(kind, OMITTED_UNSUPPORTED_FORMAT),
            'tool': tool_name,
            **presentation,
        }
