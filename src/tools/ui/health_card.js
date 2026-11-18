// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

// Cluster health status badge and metric tiles, shared by the health and overview views.
var HEALTH_METRICS = [
  ['number_of_nodes', 'Nodes'],
  ['number_of_data_nodes', 'Data nodes'],
  ['active_primary_shards', 'Active primary shards'],
  ['active_shards', 'Active shards'],
  ['relocating_shards', 'Relocating shards', 'warn'],
  ['initializing_shards', 'Initializing shards', 'warn'],
  ['unassigned_shards', 'Unassigned shards', 'alert'],
  ['delayed_unassigned_shards', 'Delayed unassigned shards', 'alert'],
  ['number_of_pending_tasks', 'Pending tasks', 'warn'],
  ['number_of_in_flight_fetch', 'In-flight fetches'],
  ['task_max_waiting_in_queue_millis', 'Longest task wait (ms)', 'warn']
];
// Shown as large tiles; the rest are a row of small ones.
var PRIMARY_METRICS = ['number_of_nodes', 'number_of_data_nodes', 'active_primary_shards', 'active_shards', 'unassigned_shards'];
var STATUS_ORDER = dict({ red: 0, yellow: 1, green: 2 });

function healthStatus(h) { return typeof h.status === 'string' ? h.status.toLowerCase() : 'unknown'; }
function statusBadge(status) {
  var badge = el('span', status, 'status-badge ' + (status in STATUS_ORDER ? status : 'unknown'));
  badge.setAttribute('role', 'status');
  return badge;
}
// Status tile: the badge, the active-shard percentage and a bar for it.
function statusTile(h) {
  var tile = el('div', null, 'kpi status-tile');
  tile.appendChild(el('div', 'Status', 'kpi-label'));
  tile.appendChild(statusBadge(healthStatus(h)));
  if (typeof h.active_shards_percent_as_number === 'number') {
    var pct = h.active_shards_percent_as_number;
    tile.appendChild(el('div', formatPercent(pct) + ' of shards active', 'kpi-sub active-pct'));
    tile.appendChild(usageBar(pct, formatPercent(pct) + ' of shards active', pct >= 100 ? 'ok' : pct >= 50 ? 'warn' : 'high'));
  }
  return tile;
}
function metricTile(h, spec) {
  var v = h[spec[0]];
  var flagged = typeof v === 'number' && v > 0 && spec[2];
  var box = el('div', null, 'metric ' + spec[0] + (flagged === 'alert' ? ' alert' : flagged === 'warn' ? ' warn' : ''));
  box.appendChild(el('div', formatNumber(v), 'value' + (flagged === 'warn' ? ' warning' : '')));
  box.appendChild(el('div', spec[1], 'muted'));
  return box;
}
// The status tile and one tile per metric present in `h`, primary metrics first. `extra` tiles
// (overview: indices, disk) go after the primary ones.
function healthCard(h, metrics, extra) {
  var frag = document.createDocumentFragment();
  var main = el('div', null, 'card');
  main.appendChild(statusTile(h));
  var rest = el('div', null, 'card secondary');
  (metrics || HEALTH_METRICS).forEach(function (spec) {
    if (!(spec[0] in h)) return;
    (PRIMARY_METRICS.indexOf(spec[0]) >= 0 ? main : rest).appendChild(metricTile(h, spec));
  });
  (extra || []).forEach(function (t) { if (t) main.appendChild(t); });
  frag.appendChild(main);
  if (rest.childNodes.length) frag.appendChild(rest);
  return frag;
}
