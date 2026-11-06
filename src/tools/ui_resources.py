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
from urllib.parse import quote, urlsplit, urlunsplit


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


# A DNS name or IPv4 address (underscores allowed for container names), or a bracketed IPv6.
_HOST_RE = re.compile(
    r'^(?:[A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?(?:\.|$))+$|^\[[0-9A-Fa-f:.]+\]$'
)
# RFC 3986 path characters, without the quote that would end an attribute or rison string.
_PATH_RE = re.compile(r'^[A-Za-z0-9\-._~!$&()*+,=:@/%]*$')


def _url_error(raw: str) -> ValueError:
    return ValueError(
        f'{DASHBOARDS_URL_ENV} must be an http(s) URL with a valid host and port and no '
        f'credentials, query, fragment or spaces, such as https://dashboards.example.com; '
        f'got {raw!r}.'
    )


def dashboards_base_url() -> str | None:
    """Return OPENSEARCH_DASHBOARDS_URL without a trailing slash, or None when unset."""
    raw = os.getenv(DASHBOARDS_URL_ENV, '').strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    host = parsed.netloc.rsplit(':', 1)[0] if not parsed.netloc.endswith(']') else parsed.netloc
    try:
        port = parsed.port
    except ValueError:
        raise _url_error(raw) from None
    if (
        parsed.scheme not in ('http', 'https')
        or not parsed.hostname
        or '@' in parsed.netloc
        or '?' in raw
        or '#' in raw
        or any(ch.isspace() or ord(ch) < 0x20 for ch in raw)
        or not _HOST_RE.match(host)
        or (':' in parsed.netloc.rsplit(']', 1)[-1] and port is None)
        or port == 0
        or not _PATH_RE.match(parsed.path)
    ):
        raise _url_error(raw)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', ''))


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


# Subdivision flags: black flag, 3-6 tag letters or digits, cancel tag.
_EMOJI_TAG_SEQUENCE = re.compile(
    '\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]{3,6}\U000e007f'
)
_ZWJ, _ZWNJ = '\u200d', '\u200c'
_VS15, _VS16 = '\ufe0e', '\ufe0f'
_KEYCAP_BASES, _KEYCAP = '0123456789#*', '\u20e3'
# Scripts with a virama whose Unicode name is two words; the rest are one word.
_TWO_WORD_VIRAMA_SCRIPTS = frozenset(
    {
        'DIVES AKURU',
        'GUNJALA GONDI',
        'MASARAM GONDI',
        'MEETEI MAYEK',
        'SYLOTI NAGRI',
        'TAI THAM',
        'ZANABAZAR SQUARE',
    }
)
# Invisible characters outside Cc/Cf: variation selectors, combining grapheme joiner, fillers.
_INVISIBLE = re.compile('[\ufe00-\ufe0d\U000e0100-\U000e01ef\u034f\u115f\u1160\u3164\uffa0]')


def _is_pictographic(ch: str) -> bool:
    return unicodedata.category(ch) == 'So'


def _is_emoji_joiner(text: str, i: int) -> bool:
    # A ZWJ joins two pictographs; the left one may carry VS16 or a skin-tone modifier.
    before = i - 1
    if before > 0 and (text[before] == _VS16 or '\U0001f3fb' <= text[before] <= '\U0001f3ff'):
        before -= 1
    return (
        before >= 0
        and i + 1 < len(text)
        and _is_pictographic(text[before])
        and _is_pictographic(text[i + 1])
    )


def _virama_script(ch: str) -> str:
    # The script a virama belongs to, read from the start of its Unicode name.
    words = unicodedata.name(ch).split()
    two = ' '.join(words[:2])
    return two if two in _TWO_WORD_VIRAMA_SCRIPTS else words[0]


def _is_virama_joiner(text: str, i: int) -> bool:
    # A ZWJ or ZWNJ right after a virama (combining class 9), before a letter of its script.
    if i == 0 or i + 1 >= len(text) or unicodedata.combining(text[i - 1]) != 9:
        return False
    letter = text[i + 1]
    return unicodedata.category(letter).startswith('L') and unicodedata.name(
        letter, ''
    ).startswith(_virama_script(text[i - 1]) + ' ')


def _is_keycap_selector(text: str, i: int) -> bool:
    return i > 0 and text[i - 1] in _KEYCAP_BASES and text[i + 1 : i + 2] == _KEYCAP


def _has_control_chars(text: str, allowed: str = '') -> bool:
    # Rejects control, format (bidi, zero-width, tags) and invisible characters. Allowed:
    # a ZWJ between pictographs, a ZWJ or ZWNJ in an Indic conjunct, VS15/VS16 right after
    # a pictograph, VS16 in a keycap, and subdivision flags.
    text = _EMOJI_TAG_SEQUENCE.sub('\U0001f3f4', text)
    if _INVISIBLE.search(text):
        return True
    for i, ch in enumerate(text):
        if ch in allowed:
            continue
        if ch == _ZWJ:
            if not (_is_emoji_joiner(text, i) or _is_virama_joiner(text, i)):
                return True
        elif ch == _ZWNJ:
            if not _is_virama_joiner(text, i):
                return True
        elif ch in (_VS15, _VS16):
            if not (
                (i > 0 and _is_pictographic(text[i - 1]))
                or (ch == _VS16 and _is_keycap_selector(text, i))
            ):
                return True
        elif unicodedata.category(ch) in ('Cc', 'Cf'):
            return True
    return False


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
                'control characters, bidi controls or invisible characters; newlines, tabs and '
                'emoji are fine.'
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
                'characters, bidi controls or invisible characters.'
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
    parts = urlsplit(base)
    path = parts.path + '/app/data-explorer/discover'
    return urlunsplit((parts.scheme, parts.netloc, path, '', f'?_q={encoded}'))


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


# Largest integer a JSON number keeps exactly in a JavaScript view (Number.MAX_SAFE_INTEGER).
MAX_SAFE_INTEGER = 2**53 - 1


def _exact_integers(value: Any, path: str, paths: list[str]) -> Any:
    """Return value with unsafe integers as decimal strings, adding each one's dotted path."""
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_INTEGER:
            if path not in paths:
                paths.append(path)
            return str(value)
        return value
    if isinstance(value, dict):
        return {k: _exact_integers(v, f'{path}.{k}', paths) for k, v in value.items()}
    if isinstance(value, list):
        return [_exact_integers(v, path, paths) for v in value]
    return value


def _exact_rows(columns: list, rows: list) -> tuple[list, list[str]]:
    """Encode unsafe integers in rows; return the rows and the column paths that hold them."""
    paths: list[str] = []
    out = []
    for row in rows:
        if isinstance(row, list):
            tail = row[len(columns) :]
            row = [_exact_integers(v, str(c), paths) for c, v in zip(columns, row)] + tail
        out.append(row)
    return out, paths


def _with_exact_integers(table: dict) -> dict:
    rows, paths = _exact_rows(table['columns'], table['rows'])
    table['rows'] = rows
    if paths:
        table['integer_strings'] = paths
    return table


# Bucket fields that are not columns of their own (key is placed first).
_BUCKET_KEYS = ('key', 'key_as_string')
_AGG_NESTING = ('aggs', 'aggregations')


def _sub_column(out: dict, prefix: str, name: str, suffixes: list[str] | None) -> str:
    """Column base for sub-aggregation `name`, with an agg: prefix when needed.

    The prefix keeps names that look like response metadata (meta, *_as_string) and names
    that would overwrite an existing column. `suffixes` are the metric's column suffixes;
    None for a single-bucket aggregation, whose columns start with the base and a dot.
    """
    base = prefix + name
    if suffixes is None:
        taken = any(c.startswith(base + '.') for c in out)
    else:
        taken = any(base + s in out for s in suffixes)
    if name == 'meta' or name.endswith('_as_string') or taken:
        return f'{prefix}agg:{name}'
    return base


def _agg_types(query_dsl: Any) -> dict[tuple, str]:
    """Map each aggregation path in the request body to its type, such as ('a', 'b') -> 'terms'."""
    if isinstance(query_dsl, str):
        try:
            query_dsl = json.loads(query_dsl)
        except ValueError:
            return {}
    types: dict[tuple, str] = {}

    def walk(aggs: Any, prefix: tuple) -> None:
        if not isinstance(aggs, dict):
            return
        for name, spec in aggs.items():
            if not isinstance(spec, dict):
                continue
            kinds = [k for k in spec if k not in _AGG_NESTING and k != 'meta']
            types[prefix + (name,)] = kinds[0] if kinds else 'unknown'
            for key in _AGG_NESTING:
                walk(spec.get(key), prefix + (name,))

    if isinstance(query_dsl, dict):
        for key in _AGG_NESTING:
            walk(query_dsl.get(key), ())
    return types


def _hit_rows(hits: list) -> tuple[list, list]:
    columns = ['_index', '_id', '_score']
    for hit in hits:
        for key in (hit.get('_source') or {}) if isinstance(hit, dict) else ():
            if key not in columns:
                columns.append(key)
    rows = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get('_source') or {}
        rows.append(
            [hit.get(c) if c in ('_index', '_id', '_score') else source.get(c) for c in columns]
        )
    return columns, rows


def _metric_cells(agg: dict, depth: int = 0) -> tuple[dict, bool]:
    """Flatten a metric result to {suffix: scalar}; suffix '' is the single value.

    Returns the cells and whether anything (a list field) was left out.
    """
    if 'value' in agg and not isinstance(agg['value'], (dict, list)):
        single = {'': agg.get('value_as_string', agg['value'])}
        if isinstance(agg.get('keys'), list):  # max_bucket, min_bucket
            single['.keys'] = agg['keys']
        return single, False
    cells: dict[str, Any] = {}
    partial = False
    for k, v in agg.items():
        if k == 'meta' or k.endswith('_as_string'):
            continue
        if k == 'values' and isinstance(v, dict):  # percentiles, percentile_ranks
            cells.update({f'[{p}]': x for p, x in v.items() if not p.endswith('_as_string')})
        elif k == 'values' and isinstance(v, list):  # keyed: false
            for item in v:
                if isinstance(item, dict) and 'key' in item:
                    cells[f'[{item["key"]}]'] = item.get('value')
                else:
                    partial = True
        elif isinstance(v, dict) and depth < 3:  # geo_bounds, extended_stats bounds
            sub, sub_partial = _metric_cells(v, depth + 1)
            cells.update({f'.{k}{s}': x for s, x in sub.items()})
            partial = partial or sub_partial
        elif isinstance(v, (dict, list)):
            partial = True
        else:
            cells[f'.{k}'] = v
    return cells, partial


class _AggTables:
    """Turns an aggregations response into view tables and notes for what it cannot show."""

    def __init__(self, types: dict[tuple, str]):
        self.types = types
        self.tables: list[dict] = []
        self.notes: list[str] = []
        self.order: dict[tuple, int] = {}

    def note(self, path: tuple, partly: bool = False) -> None:
        label = ' > '.join(path)
        kind = self.types.get(path, 'unknown')
        shown = 'only partly' if partly else 'not'
        text = f'aggregation {label} ({kind}) is {shown} shown; see the text result'
        if text not in self.notes:
            self.notes.append(text)

    def table(self, path: tuple, columns: list, rows: list[dict]) -> None:
        # Child tables from many parent buckets share one table per aggregation path.
        if path not in self.order:
            self.order[path] = len(self.tables)
            self.tables.append({'name': ' > '.join(path), 'columns': [], 'rows': []})
        t = self.tables[self.order[path]]
        for c in columns:
            if c not in t['columns']:
                t['columns'].append(c)
        t['rows'].extend(rows)

    def cells(self, path: tuple, obj: dict, prefix: str, out: dict, children: list) -> None:
        """Columns of one bucket (or single-bucket aggregation) and its sub-aggregations."""
        for k, v in obj.items():
            sub_path = path + (k,)
            if k in _BUCKET_KEYS:
                continue
            if k == 'meta' and sub_path not in self.types:  # the aggregation's own metadata
                continue
            if k.endswith('_as_string') and not isinstance(v, dict):  # formatted copy
                continue
            if not isinstance(v, (dict, list)):
                out[prefix + k] = v
            elif isinstance(v, list):
                self.note(path, partly=True)
            elif 'buckets' in v:
                children.append((sub_path, v))
            elif isinstance(v.get('hits'), dict) and isinstance(v['hits'].get('hits'), list):
                out[_sub_column(out, prefix, k, [''])] = [
                    h.get('_source') for h in v['hits']['hits'] if isinstance(h, dict)
                ]
            elif 'doc_count' in v:
                self.cells(sub_path, v, f'{_sub_column(out, prefix, k, None)}.', out, children)
            else:
                metric, partial = _metric_cells(v)
                if not metric:
                    self.note(sub_path)
                elif partial:
                    self.note(sub_path, partly=True)
                base = _sub_column(out, prefix, k, list(metric))
                out.update({f'{base}{s}': x for s, x in metric.items()})

    def buckets(self, path: tuple, buckets: list, parent: dict) -> None:
        rows: list[dict] = []
        columns = list(parent)
        children: list = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            row = dict(parent)
            key = bucket.get('key_as_string', bucket.get('key'))
            if isinstance(key, dict):  # composite: one column per source
                for part, value in key.items():
                    row[part if part not in row else f'key.{part}'] = value
            else:
                row['key'] = key
            own: list = []
            self.cells(path, bucket, '', row, own)
            for sub_path, sub in own:
                sub_parent = dict(parent)
                if isinstance(key, dict):
                    sub_parent.update({f'{path[-1]}.{p}': v for p, v in key.items()})
                else:
                    name = path[-1] if path[-1] not in sub_parent else ' > '.join(path)
                    sub_parent[name] = key
                children.append((sub_path, sub, sub_parent))
            for c in row:
                if c not in columns:
                    columns.append(c)
            rows.append(row)
        self.table(path, columns, rows)
        for sub_path, sub, sub_parent in children:
            self.agg(sub_path, sub, sub_parent)

    def agg(self, path: tuple, agg: Any, parent: dict) -> None:
        if not isinstance(agg, dict):
            self.note(path)
            return
        buckets = agg.get('buckets')
        if isinstance(buckets, dict):  # keyed filters and ranges
            buckets = [{'key': k, **v} for k, v in buckets.items() if isinstance(v, dict)]
        if isinstance(buckets, list):
            self.buckets(path, buckets, parent)
            return
        hits = agg.get('hits')
        if isinstance(hits, dict) and isinstance(hits.get('hits'), list):  # top_hits
            columns, rows = _hit_rows(hits['hits'])
            self.table(
                path, list(parent) + columns, [{**parent, **dict(zip(columns, r))} for r in rows]
            )
            return
        if 'doc_count' in agg:  # single bucket: filter, nested, global, missing, ...
            row = dict(parent)
            children: list = []
            self.cells(path, agg, '', row, children)
            self.table(path, list(row), [row])
            for sub_path, sub in children:
                self.agg(sub_path, sub, dict(parent))
            return
        metric, partial = _metric_cells(agg)
        if not metric:
            self.note(path)
            return
        if partial:
            self.note(path, partly=True)
        if not parent and all(s.startswith('[') for s in metric):  # percentiles: a row each
            rows = [{'key': s[1:-1], 'value': v} for s, v in metric.items()]
            self.table(path, ['key', 'value'], rows)
            return
        row = dict(parent)
        row.update({(s.lstrip('.') or 'value'): v for s, v in metric.items()})
        self.table(path, list(row), [row])

    def result(self) -> tuple[list[dict], list[str]]:
        tables = []
        for t in self.tables:
            rows = [[r.get(c) for c in t['columns']] for r in t['rows']]
            tables.append(_with_exact_integers({**t, 'rows': rows}))
        return tables, self.notes


def _agg_tables(aggregations: dict, query_dsl: Any = None) -> tuple[list[dict], list[str]]:
    """Return one table per aggregation path and notes for aggregations left out.

    Sub-bucket tables carry their parent bucket keys as leading columns.
    """
    builder = _AggTables(_agg_types(query_dsl))
    for name, agg in (aggregations or {}).items():
        builder.agg((name,), agg, {})
    return builder.result()


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
    columns, rows = _hit_rows(hits)
    rows, exact = _exact_rows(columns, rows)
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
    if exact:
        meta['integer_strings'] = exact
    if response.get('aggregations'):
        tables, notes = _agg_tables(response['aggregations'], query_dsl)
        if tables:
            meta['aggregations'] = tables
        if notes:
            meta['aggregation_notes'] = notes
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
    rows, exact = _exact_rows(columns, response.get('datarows') or [])
    meta = {
        'query': query,
        'total': response.get('total'),
        'size': response.get('size'),
        'column_types': [c.get('type') for c in schema],
    }
    if exact:
        meta['integer_strings'] = exact
    link = discover_ppl_url(query)
    if link:
        meta['dashboards_url'] = link
    return _structured('ppl_rows', cluster, columns, rows, meta)


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
        return float(value) if '.' in value else int(value)
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
    rows, exact = _exact_rows(columns, rows)
    meta = {'title': title, 'source': source, 'column_types': types}
    if exact:
        meta['integer_strings'] = exact
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


def _serialized_size(payload: Any) -> int:
    # UTF-8 bytes of compact JSON with raw non-ASCII: what stdio, SSE and pre-2026 HTTP send.
    return len(
        json.dumps(payload, default=str, ensure_ascii=False, separators=(',', ':')).encode(
            'utf-8', 'surrogatepass'
        )
    )


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
            # Name only the dimensions that were cut, as aggregation tables do.
            meta['truncated'] = True
            if len(payload['rows']) < original_rows:
                meta['row_count'] = original_rows
            if len(payload['columns']) < original_columns:
                meta['column_count'] = original_columns
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
