# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCP Apps (io.modelcontextprotocol/ui) views and structured result payloads."""

import json
import logging
import os
from functools import lru_cache
from importlib import resources
from typing import Any


logger = logging.getLogger(__name__)

APPS_ENABLED_ENV = 'OPENSEARCH_MCP_APPS_ENABLED'
RESULTS_VIEW_URI = 'ui://opensearch/results'
STRUCTURED_CONTENT_VERSION = 1
STRUCTURED_CONTENT_MAX_BYTES = 1024 * 1024

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


def _structured(kind: str, cluster: str, columns: list, rows: list, meta: dict) -> dict:
    return {
        'kind': kind,
        'version': STRUCTURED_CONTENT_VERSION,
        'cluster': cluster or '',
        'columns': columns,
        'rows': rows,
        'meta': meta,
    }


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
    return _structured('ppl_rows', cluster, columns, response.get('datarows') or [], meta)


def cap_structured_content(payload: dict | None) -> dict | None:
    """Drop the payload when its serialized size exceeds the structuredContent cap."""
    if payload is None:
        return None
    size = len(json.dumps(payload, default=str, separators=(',', ':')).encode('utf-8'))
    if size > STRUCTURED_CONTENT_MAX_BYTES:
        logger.info(f'Omitting structuredContent: {size} bytes exceeds cap')
        return None
    return payload
