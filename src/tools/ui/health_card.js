// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

// Cluster health status and metric tiles, shared by the health and overview views. Labels stay
// short so a row of tiles keeps one height; the title attribute has the full name.
var HEALTH_METRICS = [
  ['number_of_nodes', 'Nodes'],
  ['number_of_data_nodes', 'Data nodes'],
  ['active_primary_shards', 'Active primaries', null, 'Active primary shards'],
  ['active_shards', 'Active shards'],
  ['relocating_shards', 'Relocating', 'warn', 'Relocating shards'],
  ['initializing_shards', 'Initializing', 'warn', 'Initializing shards'],
  ['unassigned_shards', 'Unassigned shards', 'alert'],
  ['delayed_unassigned_shards', 'Delayed', 'alert', 'Delayed unassigned shards'],
  ['number_of_pending_tasks', 'Pending tasks', 'warn'],
  ['number_of_in_flight_fetch', 'In-flight fetches'],
  ['task_max_waiting_in_queue_millis', 'Task wait (ms)', 'warn', 'Longest task wait in the queue (ms)']
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
// Status tile: the health dot and word, the active-shard percentage and a bar for it.
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
  box.appendChild(el('div', formatNumber(v), 'value'));
  var label = el('div', spec[1], 'muted');
  if (spec[3]) label.title = spec[3];
  box.appendChild(label);
  return box;
}
// Secondary counters are shown only when non-zero; primary ones always.
function shownSecondary(h, key) { return key in h && h[key] !== 0; }
// The status tile and one tile per metric present in `h`, primary metrics first. `extra` tiles
// (overview: indices, disk) go after the primary ones.
function healthCard(h, metrics, extra) {
  var frag = document.createDocumentFragment();
  var main = el('div', null, 'card');
  main.appendChild(statusTile(h));
  var rest = el('div', null, 'card secondary');
  (metrics || HEALTH_METRICS).forEach(function (spec) {
    if (!(spec[0] in h)) return;
    if (PRIMARY_METRICS.indexOf(spec[0]) >= 0) main.appendChild(metricTile(h, spec));
    else if (shownSecondary(h, spec[0])) rest.appendChild(metricTile(h, spec));
  });
  (extra || []).forEach(function (t) { if (t) main.appendChild(t); });
  frag.appendChild(main);
  if (rest.childNodes.length) frag.appendChild(rest);
  return frag;
}
