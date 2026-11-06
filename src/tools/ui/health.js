// Cluster health card for ClusterHealthTool: status badge, node and shard counts, with
// unassigned shards highlighted; refresh re-calls the tool with the same arguments.
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
var STATUS_ORDER = { red: 0, yellow: 1, green: 2 };

function renderHealth(data) {
  var h = data.health || {};
  var status = typeof h.status === 'string' ? h.status.toLowerCase() : 'unknown';
  app.appendChild(header(data, 'Cluster health' + (h.cluster_name ? ': ' + h.cluster_name : ''), h.timed_out ? ['timed out'] : []));
  var top = el('div', null, 'toolbar');
  var badge = el('span', status, 'status-badge ' + (status in STATUS_ORDER ? status : 'unknown'));
  badge.setAttribute('role', 'status');
  top.appendChild(badge);
  if (typeof h.active_shards_percent_as_number === 'number') {
    var pct = h.active_shards_percent_as_number;
    top.appendChild(el('span', formatPercent(pct) + ' of shards active', 'stat active-pct'));
    var meter = el('span', null, 'meter');
    meter.style.width = Math.round(Math.max(0, Math.min(100, pct)) * 1.6) + 'px';
    top.appendChild(meter);
  }
  app.appendChild(top);
  var card = el('div', null, 'card');
  HEALTH_METRICS.forEach(function (spec) {
    if (!(spec[0] in h)) return;
    var v = h[spec[0]];
    var flagged = typeof v === 'number' && v > 0 && spec[2];
    var box = el('div', null, 'metric ' + spec[0] + (flagged === 'alert' ? ' alert' : ''));
    box.appendChild(el('div', formatNumber(v), 'value' + (flagged === 'warn' ? ' warning' : '')));
    box.appendChild(el('div', spec[1], 'muted'));
    card.appendChild(box);
  });
  app.appendChild(card);
  if (typeof h.unassigned_shards === 'number' && h.unassigned_shards > 0) {
    app.appendChild(el('div', plural(h.unassigned_shards, 'unassigned shard') + '. On a single-node cluster, replicas stay unassigned; otherwise check allocation (GetShardsTool, GetAllocationTool).', 'notice warning unassigned'));
  }
  if (Array.isArray(data.rows) && data.rows.length) {
    var rows = data.rows.slice().sort(function (a, b) {
      return ((STATUS_ORDER[a[1]] !== undefined ? STATUS_ORDER[a[1]] : 3) - (STATUS_ORDER[b[1]] !== undefined ? STATUS_ORDER[b[1]] : 3)) || compare(a[0], b[0]);
    });
    app.appendChild(el('h3', 'Indices'));
    var scroll = el('div', null, 'scroll');
    var table = el('table');
    var hr = el('tr');
    data.columns.forEach(function (c) { hr.appendChild(el('th', c)); });
    var thead = el('thead'); thead.appendChild(hr); table.appendChild(thead);
    var tbody = el('tbody');
    rows.forEach(function (r) {
      var tr = el('tr');
      r.forEach(function (v, i) {
        tr.appendChild(el('td', cellText(v), data.columns[i] === 'status' && typeof v === 'string' ? 'status-' + v : ''));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); scroll.appendChild(table); app.appendChild(scroll);
  }
}

startView({
  name: 'opensearch-cluster-health',
  unsupportedText: 'This response has no cluster health card; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'cluster_health' && data.health && typeof data.health === 'object'; },
  render: renderHealth
});
