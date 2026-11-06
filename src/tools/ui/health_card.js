// Cluster health status badge and metric cards, shared by the health and overview views.
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
var STATUS_ORDER = dict({ red: 0, yellow: 1, green: 2 });

function healthStatus(h) { return typeof h.status === 'string' ? h.status.toLowerCase() : 'unknown'; }
function statusBadge(status) {
  var badge = el('span', status, 'status-badge ' + (status in STATUS_ORDER ? status : 'unknown'));
  badge.setAttribute('role', 'status');
  return badge;
}
// Badge with the active-shard percentage, then one card per metric present in `h`.
function healthCard(h, metrics) {
  var frag = document.createDocumentFragment();
  var top = el('div', null, 'toolbar');
  top.appendChild(statusBadge(healthStatus(h)));
  if (typeof h.active_shards_percent_as_number === 'number') {
    var pct = h.active_shards_percent_as_number;
    top.appendChild(el('span', formatPercent(pct) + ' of shards active', 'stat active-pct'));
    var meter = el('span', null, 'meter');
    meter.style.width = Math.round(Math.max(0, Math.min(100, pct)) * 1.6) + 'px';
    top.appendChild(meter);
  }
  frag.appendChild(top);
  var card = el('div', null, 'card');
  (metrics || HEALTH_METRICS).forEach(function (spec) {
    if (!(spec[0] in h)) return;
    var v = h[spec[0]];
    var flagged = typeof v === 'number' && v > 0 && spec[2];
    var box = el('div', null, 'metric ' + spec[0] + (flagged === 'alert' ? ' alert' : ''));
    box.appendChild(el('div', formatNumber(v), 'value' + (flagged === 'warn' ? ' warning' : '')));
    box.appendChild(el('div', spec[1], 'muted'));
    card.appendChild(box);
  });
  frag.appendChild(card);
  return frag;
}
