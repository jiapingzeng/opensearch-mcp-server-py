// Cluster health card for ClusterHealthTool: status badge, node and shard counts, with
// unassigned shards highlighted; refresh re-calls the tool with the same arguments.
function renderHealth(data) {
  var h = data.health || {};
  app.appendChild(header(data, 'Cluster health' + (h.cluster_name ? ': ' + h.cluster_name : ''), h.timed_out ? ['timed out'] : []));
  app.appendChild(healthCard(h));
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
