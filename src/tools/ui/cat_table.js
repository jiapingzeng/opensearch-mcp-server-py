// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

// Table view for _cat-style tools (ListIndexTool, GetShardsTool, CatNodesTool, GetAllocationTool):
// numbers and byte sizes sort numerically and show units; health and state are coloured.
var catState = newTableState();

function catSummary(data, types) {
  var parts = [];
  ['health', 'state'].forEach(function (name) {
    var i = data.columns.indexOf(name);
    if (i < 0) return;
    var counts = dict();
    data.rows.forEach(function (r) { var k = r[i] === null || r[i] === undefined ? '(none)' : String(r[i]); counts[k] = (counts[k] || 0) + 1; });
    parts.push(name + ': ' + Object.keys(counts).sort().map(function (k) { return k + ' ' + counts[k]; }).join(', '));
  });
  var total = 0, sized = false;
  data.columns.forEach(function (c, i) {
    if (types[i] === 'bytes' && (c === 'store.size' || c === 'store')) {
      sized = true;
      data.rows.forEach(function (r) { if (typeof r[i] === 'number') total += r[i]; });
    }
  });
  if (sized) parts.push('total store: ' + formatBytes(total));
  return parts;
}

function renderCatTable(data) {
  if (data.kind === 'index_detail') {
    // Settings and aliases of named indices: shown by the index explorer, not as a table.
    app.appendChild(header(data, 'Indices', []));
    app.appendChild(el('div', currentView.unsupportedText, 'notice'));
    return;
  }
  var m = data.meta || {};
  var types = columnTypes(data.columns, data.rows, m.column_types, m.integer_strings);
  var rowsStat = rowCountStat(data.rows.length);
  var stats = [rowsStat.node];
  if (m.source) stats.push(m.source);
  app.appendChild(header(data, m.title || 'Table', stats.concat(catSummary(data, types))));
  var trunc = truncationNotice(data);
  if (trunc) app.appendChild(trunc);
  var unassigned = data.columns.indexOf('state') >= 0 ? data.rows.filter(function (r) { return r[data.columns.indexOf('state')] === 'UNASSIGNED'; }).length : 0;
  if (unassigned) app.appendChild(el('div', plural(unassigned, 'unassigned shard') + '.', 'notice warning unassigned'));
  if (data.rows.length === 0) { if (!trunc) app.appendChild(el('div', 'No rows.', 'notice')); return; }
  var section = buildDataSection(data, data.columns, data.rows, types, catState, { chipSkipNumeric: true, rowsStat: rowsStat });
  if (section.chips) app.appendChild(section.chips);
  app.appendChild(section.toolbar);
  app.appendChild(section.stats);
  app.appendChild(section.table.node);
}

startView({
  name: 'opensearch-table',
  unsupportedText: 'This result is not a table (for example, details for one index); the text result is unchanged.',
  accepts: function (data) { return (data.kind === 'cat_table' && Array.isArray(data.columns) && Array.isArray(data.rows)) || data.kind === 'index_detail'; },
  render: renderCatTable
});
