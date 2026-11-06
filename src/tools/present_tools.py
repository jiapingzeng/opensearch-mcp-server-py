# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Task-oriented MCP Apps tools: one call renders a view that spans several read-only APIs.

They exist only for hosts that render MCP Apps views. The model gets a short text summary; the
view gets every section in structuredContent and drills down through the existing tools.
"""

import asyncio
import logging
from .tool_params import baseToolArgs
from .ui_resources import (
    HEALTH_FIELDS,
    STRUCTURED_CONTENT_VERSION,
    cap_section_tables,
    cat_table_payload,
    index_detail_payload,
)
from pydantic import Field
from typing import Any, Literal, Optional


logger = logging.getLogger(__name__)

OVERVIEW_SECTIONS = ('health', 'nodes', 'indices', 'shards', 'allocation')
OverviewSection = Literal['health', 'nodes', 'indices', 'shards', 'allocation']
# Unassigned shards explained with _cluster/allocation/explain, one cheap call each.
EXPLAIN_MAX_SHARDS = 3
EXPLAIN_MAX_NODES = 10
LARGEST_INDICES = 10
SAMPLE_SIZE = 10

_INDEX_COLUMNS = (
    'health,status,index,uuid,pri,rep,docs.count,docs.deleted,store.size,pri.store.size'
)
_NODE_COLUMNS = (
    'name,ip,node.role,cluster_manager,heap.percent,ram.percent,cpu,load_1m,load_5m,load_15m,'
    'disk.used_percent'
)
_SHARD_COLUMNS = 'index,shard,prirep,state,docs,store,node,unassigned.reason,unassigned.for'
# Requested with bytes=b, so these arrive as plain integers and are typed as byte sizes.
_BYTE_COLUMNS = frozenset(
    {
        'store.size',
        'pri.store.size',
        'store',
        'disk.indices',
        'disk.used',
        'disk.avail',
        'disk.total',
    }
)


class ClusterOverviewArgs(baseToolArgs):
    """Arguments for ClusterOverviewTool."""

    sections: Optional[list[OverviewSection]] = Field(
        default=None,
        description='Optional. Fetch only these sections (health, nodes, indices, shards, '
        'allocation); omit for all of them.',
    )


class IndexExplorerArgs(baseToolArgs):
    """Arguments for IndexExplorerTool."""

    index: str = Field(
        default='',
        description='Optional index name or pattern (such as logs-*) to list; omit for all '
        'indices. A concrete index name is selected first.',
    )


# Present tool registry key -> dependency display names and sections, set when the handlers
# are built so a renamed or filtered dependency is reflected in the payload.
_PLANS: dict[str, dict] = {}


def resolve_present_tools(enabled_tools: dict) -> dict[str, dict]:
    """Return, per enabled present tool (by its enabled key), whether it can be listed.

    Dependencies are registry keys, matched to enabled entries by their function, so they are
    found under custom display names and in both the single-mode (display name) and multi-mode
    (registry key) registries.
    """
    from .tools import TOOL_REGISTRY

    by_function = {
        id(info['function']): key for key, info in TOOL_REGISTRY.items() if info.get('function')
    }
    enabled_by_registry: dict[str, str] = {}
    for key, info in enabled_tools.items():
        enabled_by_registry.setdefault(by_function.get(id(info.get('function')), key), key)
    plans: dict[str, dict] = {}
    _PLANS.clear()
    for key, info in enabled_tools.items():
        spec = info.get('ui_present')
        if not spec:
            continue
        deps = [
            *spec['requires'],
            *spec.get('optional', {}).values(),
            *spec.get('links', {}).values(),
        ]
        names = {
            dep: enabled_tools[enabled_by_registry[dep]].get('display_name') or dep
            for dep in deps
            if dep in enabled_by_registry
        }
        plan = {
            'listed': all(dep in names for dep in spec['requires']),
            'missing': [dep for dep in spec['requires'] if dep not in names],
            'names': names,
            'calls': [enabled_by_registry[dep] for dep in spec.get('calls', []) if dep in names],
        }
        plans[key] = plan
        _PLANS[by_function.get(id(info.get('function')), key)] = plan
    # A link to another present tool counts only when that tool is listed too.
    for plan in _PLANS.values():
        for dep in list(plan['names']):
            if dep in _PLANS and not _PLANS[dep]['listed']:
                del plan['names'][dep]
    return plans


def _names(tool: str, spec: dict) -> dict[str, str]:
    plan = _PLANS.get(tool)
    if plan is None:  # called outside a server, as in unit tests: assume every dependency
        deps = [
            *spec['requires'],
            *spec.get('optional', {}).values(),
            *spec.get('links', {}).values(),
        ]
        return {dep: dep for dep in deps}
    return plan['names']


def format_bytes(n: Any) -> str:
    """Format a byte count in 1024-based units with 3 significant digits, like the views."""
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return str(n)
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    v, i = float(abs(n)), 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    sign = '-' if n < 0 else ''
    if i == 0:
        return f'{sign}{round(v)} B'
    return f'{sign}{v:.{2 if v < 10 else 1 if v < 100 else 0}f} {units[i]}'


def _error_text(what: str, e: Exception) -> str:
    return f'Error getting {what}: {e}'


def _table(entries: Any, cluster: str, source: str) -> dict:
    payload = cat_table_payload(entries, cluster, '', source)
    if payload is None:
        raise ValueError(f'{source} returned an unexpected response shape')
    meta = payload['meta']
    meta['column_types'] = [
        'bytes' if c in _BYTE_COLUMNS and t == 'number' else t
        for c, t in zip(payload['columns'], meta['column_types'])
    ]
    meta.pop('title', None)
    return {'columns': payload['columns'], 'rows': payload['rows'], 'meta': meta}


def _col(table: dict, name: str) -> int:
    return table['columns'].index(name) if name in table['columns'] else -1


def _num(row: list, i: int) -> float:
    v = row[i] if i >= 0 else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


async def _health(client, args) -> dict:
    response = await client.cluster.health()
    if not isinstance(response, dict) or 'status' not in response:
        raise ValueError('_cluster/health returned an unexpected response shape')
    return {'data': {k: response[k] for k in HEALTH_FIELDS if k in response}}


async def _nodes(client, args) -> dict:
    entries = await client.cat.nodes(format='json', h=_NODE_COLUMNS)
    return _table(entries, args.opensearch_cluster_name, '_cat/nodes')


async def _indices(client, args) -> dict:
    entries = await client.cat.indices(format='json', bytes='b', h=_INDEX_COLUMNS)
    table = _table(entries, args.opensearch_cluster_name, '_cat/indices')
    health, store, docs, name = (
        _col(table, 'health'),
        _col(table, 'store.size'),
        _col(table, 'docs.count'),
        _col(table, 'index'),
    )
    # Largest first, so a row cut keeps the biggest indices.
    table['rows'].sort(key=lambda r: -_num(r, store))
    by_health: dict[str, int] = {}
    for row in table['rows']:
        key = str(row[health]) if health >= 0 and row[health] is not None else 'unknown'
        by_health[key] = by_health.get(key, 0) + 1
    table['summary'] = {
        'count': len(table['rows']),
        'by_health': dict(sorted(by_health.items())),
        'docs': sum(_num(r, docs) for r in table['rows']),
        'store_bytes': sum(_num(r, store) for r in table['rows']),
        'largest': [
            [r[name], _num(r, store)] for r in table['rows'][:LARGEST_INDICES] if name >= 0
        ],
    }
    return table


def _explanation(response: dict) -> dict:
    nodes = []
    for decision in (response.get('node_allocation_decisions') or [])[:EXPLAIN_MAX_NODES]:
        reasons = [
            d.get('explanation')
            for d in decision.get('deciders') or []
            if str(d.get('decision', '')).upper() == 'NO' and d.get('explanation')
        ]
        nodes.append(
            {
                'node': decision.get('node_name'),
                'decision': decision.get('node_decision'),
                'reasons': reasons[:3],
            }
        )
    return {
        'index': response.get('index'),
        'shard': response.get('shard'),
        'primary': response.get('primary'),
        'unassigned_reason': (response.get('unassigned_info') or {}).get('reason'),
        'can_allocate': response.get('can_allocate'),
        'explanation': response.get('allocate_explanation'),
        'nodes': nodes,
    }


async def _shards(client, args) -> dict:
    entries = await client.cat.shards(format='json', bytes='b', h=_SHARD_COLUMNS)
    table = _table(entries, args.opensearch_cluster_name, '_cat/shards')
    state, prirep, node, reason, index, shard = (
        _col(table, c) for c in ('state', 'prirep', 'node', 'unassigned.reason', 'index', 'shard')
    )
    by_state: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    per_node: dict[str, dict] = {}
    unassigned = []
    for row in table['rows']:
        s = str(row[state]) if state >= 0 else 'UNKNOWN'
        by_state[s] = by_state.get(s, 0) + 1
        if s == 'UNASSIGNED':
            unassigned.append(row)
            r = str(row[reason]) if reason >= 0 and row[reason] else 'UNKNOWN'
            by_reason[r] = by_reason.get(r, 0) + 1
        elif node >= 0 and row[node]:
            counts = per_node.setdefault(str(row[node]), {'primaries': 0, 'replicas': 0})
            counts['primaries' if row[prirep] == 'p' else 'replicas'] += 1
    # Unassigned first (primaries before replicas), so a row cut keeps them.
    table['rows'].sort(key=lambda r: (r[state] != 'UNASSIGNED', r[prirep] != 'p', str(r[index])))
    table['summary'] = {
        'total': len(table['rows']),
        'by_state': dict(sorted(by_state.items())),
        'unassigned': len(unassigned),
        'unassigned_by_reason': dict(sorted(by_reason.items())),
        'per_node': dict(sorted(per_node.items())),
    }
    picked = sorted(unassigned, key=lambda r: (r[prirep] != 'p', str(r[index])))[
        :EXPLAIN_MAX_SHARDS
    ]

    async def explain(row):
        body = {'index': row[index], 'shard': int(row[shard]), 'primary': row[prirep] == 'p'}
        return _explanation(await client.cluster.allocation_explain(body=body))

    if picked:
        results = await asyncio.gather(*(explain(r) for r in picked), return_exceptions=True)
        table['explanations'] = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            table['explain_error'] = _error_text('allocation explanations', errors[0])
    return table


async def _allocation(client, args) -> dict:
    entries = await client.cat.allocation(format='json', bytes='b')
    return _table(entries, args.opensearch_cluster_name, '_cat/allocation')


_FETCHERS = {
    'health': (_health, 'cluster health'),
    'nodes': (_nodes, 'nodes'),
    'indices': (_indices, 'indices'),
    'shards': (_shards, 'shards'),
    'allocation': (_allocation, 'disk allocation'),
}


async def _section(name: str, args) -> dict:
    """Fetch one section with its own client, so one failure never affects the others."""
    from opensearch.client import get_opensearch_client

    fetch, label = _FETCHERS[name]
    try:
        async with get_opensearch_client(args) as client:
            return {'status': 'ok', **(await fetch(client, args))}
    except Exception as e:
        logger.warning(f'Cluster overview section {name} failed: {type(e).__name__}: {e}')
        return {'status': 'error', 'error': _error_text(label, e)}


def _plural(n: Any, word: str, many: str = '') -> str:
    return f'{n:,} ' + (word if n == 1 else many or word + 's')


def _counts(counts: dict) -> str:
    return ', '.join(f'{v:,} {k}' for k, v in counts.items())


def _setting_count(value: Any, word: str) -> str:
    # Index settings arrive as strings, such as "1".
    try:
        return _plural(int(value), word)
    except (TypeError, ValueError):
        return f'{value if value is not None else "?"} {word}s'


def overview_text(payload: dict) -> str:
    """A few lines for the model: the cluster's state, not its data."""
    s = payload['sections']
    lines = []
    h = s.get('health', {})
    if h.get('status') == 'ok':
        d = h['data']
        name = d.get('cluster_name') or payload.get('cluster') or 'The cluster'
        lines.append(
            f'{name} is {str(d.get("status", "unknown")).upper()}: '
            f'{_plural(d.get("number_of_nodes", 0), "node")} '
            f'({d.get("number_of_data_nodes", 0):,} data), '
            f'{_plural(d.get("active_shards", 0), "active shard")}, '
            f'{d.get("unassigned_shards", 0):,} unassigned, '
            f'{_plural(d.get("number_of_pending_tasks", 0), "pending task")}.'
        )
    i = s.get('indices', {})
    if i.get('status') == 'ok':
        sm = i['summary']
        largest = ', '.join(f'{n} ({format_bytes(b)})' for n, b in sm['largest'][:3])
        lines.append(
            f'Indices: {sm["count"]:,} ({_counts(sm["by_health"]) or "none"}), '
            f'{_plural(sm["docs"], "document")}, {format_bytes(sm["store_bytes"])}.'
            + (f' Largest: {largest}.' if largest else '')
        )
    sh = s.get('shards', {})
    if sh.get('status') == 'ok':
        sm = sh['summary']
        line = f'Shards: {sm["total"]:,} ({_counts(sm["by_state"]) or "none"}).'
        if sm['unassigned']:
            line += f' Unassigned by reason: {_counts(sm["unassigned_by_reason"])}.'
            for e in sh.get('explanations', [])[:1]:
                kind = 'primary' if e.get('primary') else 'replica'
                line += f' {e.get("index")}[{e.get("shard")}] {kind}: {e.get("explanation")}.'
        lines.append(line)
    n = s.get('nodes', {})
    if n.get('status') == 'ok' and n['rows']:
        heap, name = _col(n, 'heap.percent'), _col(n, 'name')
        top = max(n['rows'], key=lambda r: _num(r, heap))
        lines.append(
            f'Nodes: {len(n["rows"]):,}; highest heap {top[name] if name >= 0 else "?"} '
            f'{_num(top, heap):g}%.'
        )
    a = s.get('allocation', {})
    if a.get('status') == 'ok':
        pct, node = _col(a, 'disk.percent'), _col(a, 'node')
        disks = [r for r in a['rows'] if pct >= 0 and isinstance(r[pct], (int, float))]
        if disks:
            top = max(disks, key=lambda r: r[pct])
            lines.append(f'Disk: highest use {top[node]} at {top[pct]:g}%.')
    for name in payload['requested']:
        sec = s.get(name, {})
        if sec.get('status') == 'error':
            lines.append(f'The {name} section failed: {sec["error"][:300]}')
        elif sec.get('status') == 'disabled':
            lines.append(
                f'The {name} section is off because {sec["tool"]} is not enabled on this server.'
            )
    lines.append('The user sees each section in the cluster overview view.')
    return '\n'.join(lines)


async def cluster_overview_tool(args: ClusterOverviewArgs) -> list[dict]:
    """Fetch every requested overview section concurrently and return text plus the payload."""
    from .tools import TOOL_REGISTRY, check_tool_compatibility

    try:
        await check_tool_compatibility('ClusterOverviewTool', args)
    except Exception as e:
        return [{'type': 'text', 'text': str(e), 'is_error': True}]
    spec = TOOL_REGISTRY['ClusterOverviewTool']['ui_present']
    names = _names('ClusterOverviewTool', spec)
    requested = [s for s in OVERVIEW_SECTIONS if not args.sections or s in args.sections]
    disabled = {
        sec: tool
        for sec, tool in spec['optional'].items()
        if tool not in names and sec in requested
    }
    fetched = [s for s in requested if s not in disabled]
    results = await asyncio.gather(*(_section(s, args) for s in fetched))
    sections: dict[str, Any] = dict(zip(fetched, results))
    for sec, tool in disabled.items():
        sections[sec] = {'status': 'disabled', 'tool': tool}
    payload: dict[str, Any] = {
        'kind': 'cluster_overview',
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': args.opensearch_cluster_name or '',
        'requested': requested,
        'sections': {s: sections[s] for s in requested},
        'tools': {role: names[dep] for role, dep in spec.get('links', {}).items() if dep in names},
    }
    tables = [sec for sec in payload['sections'].values() if 'rows' in sec]
    cap_section_tables(payload, tables)
    text = overview_text(payload)
    failed = [s for s in fetched if sections[s]['status'] == 'error']
    if fetched and len(failed) == len(fetched):
        return [{'type': 'text', 'text': text, 'is_error': True}]
    return [{'type': 'text', 'text': text, 'view_data': payload}]


def _is_pattern(index: str) -> bool:
    return any(c in index for c in '*?,') or index.startswith('-')


def explorer_text(payload: dict) -> str:
    """A few lines for the model about what the explorer shows."""
    pattern = payload['pattern']
    scope = f"for '{pattern}'" if pattern else 'for all indices'
    lst = payload['list']
    lines = []
    if lst.get('status') == 'ok':
        sm = lst['summary']
        lines.append(
            f'Index explorer {scope}: {_plural(sm["count"], "index", "indices")}'
            f' ({_counts(sm["by_health"]) or "none"}), {_plural(sm["docs"], "document")}, '
            f'{format_bytes(sm["store_bytes"])}.'
        )
    else:
        lines.append(f'Index explorer {scope}: the index list failed: {lst["error"][:300]}')
    detail = payload.get('detail')
    if payload.get('selected') and detail:
        sel = payload['selected']
        if detail.get('status') == 'ok':
            st = detail['index'].get('settings', {})
            aliases = detail['index'].get('aliases') or []
            lines.append(
                f'Selected {sel}: {_setting_count(st.get("number_of_shards"), "primary shard")}, '
                f'{_setting_count(st.get("number_of_replicas"), "replica")}'
                + (f', aliases {", ".join(aliases[:5])}' if aliases else ', no aliases')
                + '.'
            )
        else:
            lines.append(f'Selected {sel}: its details failed: {detail["error"][:300]}')
    lines.append(
        'The user can open the overview, mapping, shards and sample documents of any listed '
        'index in the view.'
    )
    return '\n'.join(lines)


async def index_explorer_tool(args: IndexExplorerArgs) -> list[dict]:
    """List indices for the explorer and fetch the selected index's settings and aliases."""
    from .tools import TOOL_REGISTRY, check_tool_compatibility
    from opensearch.client import get_opensearch_client

    try:
        await check_tool_compatibility('IndexExplorerTool', args)
    except Exception as e:
        return [{'type': 'text', 'text': str(e), 'is_error': True}]
    spec = TOOL_REGISTRY['IndexExplorerTool']['ui_present']
    names = _names('IndexExplorerTool', spec)
    pattern = (args.index or '').strip()
    listing: dict[str, Any]
    try:
        async with get_opensearch_client(args) as client:
            entries = await client.cat.indices(
                index=pattern or None, format='json', bytes='b', h=_INDEX_COLUMNS
            )
        listing = {'status': 'ok', **_table(entries, args.opensearch_cluster_name, '_cat/indices')}
        name, health, store, docs = (
            _col(listing, c) for c in ('index', 'health', 'store.size', 'docs.count')
        )
        listing['rows'].sort(key=lambda r: (str(r[name]).startswith('.'), str(r[name])))
        by_health: dict[str, int] = {}
        for row in listing['rows']:
            key = str(row[health]) if health >= 0 and row[health] is not None else 'unknown'
            by_health[key] = by_health.get(key, 0) + 1
        listing['summary'] = {
            'count': len(listing['rows']),
            'by_health': dict(sorted(by_health.items())),
            'docs': sum(_num(r, docs) for r in listing['rows']),
            'store_bytes': sum(_num(r, store) for r in listing['rows']),
        }
        listed = [r[name] for r in listing['rows']]
    except Exception as e:
        logger.warning(f'Index explorer list failed: {type(e).__name__}: {e}')
        listing = {'status': 'error', 'error': _error_text('the index list', e)}
        listed = []
    if pattern and not _is_pattern(pattern) and (pattern in listed or not listed):
        selected = pattern
    else:
        selected = next(
            (n for n in listed if not str(n).startswith('.')), listed[0] if listed else None
        )
    payload: dict[str, Any] = {
        'kind': 'index_explorer',
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': args.opensearch_cluster_name or '',
        'pattern': pattern,
        'list': listing,
        'selected': selected,
        'sample_size': SAMPLE_SIZE,
        'tools': {role: names[dep] for role, dep in spec['roles'].items() if dep in names},
    }
    if selected:
        try:
            async with get_opensearch_client(args) as client:
                detail = index_detail_payload(
                    await client.indices.get(index=selected), args.opensearch_cluster_name
                )
            if not detail or not detail['indices']:
                raise ValueError('GET <index> returned an unexpected response shape')
            payload['detail'] = {'status': 'ok', 'index': detail['indices'][0]}
        except Exception as e:
            logger.warning(f'Index explorer detail failed: {type(e).__name__}: {e}')
            payload['detail'] = {'status': 'error', 'error': _error_text(f'index {selected}', e)}
    if listing['status'] == 'ok':
        cap_section_tables(payload, [listing])
    text = explorer_text(payload)
    if listing['status'] == 'error' and payload.get('detail', {}).get('status') != 'ok':
        return [{'type': 'text', 'text': text, 'is_error': True}]
    return [{'type': 'text', 'text': text, 'view_data': payload}]
