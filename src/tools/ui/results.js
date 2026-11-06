// Results view for SearchIndexTool (search_hits) and PPLQueryTool (ppl_rows): table, chips,
// bar or time-series chart, paging and refresh (search), edit and re-run (PPL), export.
var resultsState = { table: newTableState(), chart: true, pageSize: 0, editorOpen: false };

function withFrom(queryDsl, from) {
  var parsed = queryDsl;
  if (typeof queryDsl === 'string') {
    try { parsed = JSON.parse(queryDsl); } catch (e) { return null; }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  var copy = deepCopy(parsed);
  copy.from = from;
  return typeof queryDsl === 'string' ? JSON.stringify(copy) : copy;
}

function pagingControls(data) {
  if (data.kind !== 'search_hits' || !canCallBack(data) || !toolInput || toolInput.query_dsl === undefined) return null;
  var m = data.meta || {};
  var from = typeof m.from === 'number' ? m.from : 0;
  var n = data.rows.length;
  if (from === 0 && n) resultsState.pageSize = n;
  var size = resultsState.pageSize || n || 10;
  var exact = m.total_relation !== 'gte' && typeof m.total === 'number';
  var bar = el('span', null, 'paging');
  var label = n === 1 ? 'hit ' + formatNumber(from + 1) : n ? 'hits ' + formatNumber(from + 1) + '\u2013' + formatNumber(from + n) : 'no hits from ' + formatNumber(from + 1);
  if (typeof m.total === 'number') label += ' of ' + formatNumber(m.total) + (m.total_relation === 'gte' ? '+' : '');
  var prev = button('Previous page', function () {
    var q = withFrom(toolInput.query_dsl, Math.max(0, from - size));
    if (q !== null) callBack(data, { query_dsl: q }, 'Loading page\u2026');
  }, 'callback page-prev');
  prev.disabled = from === 0;
  var next = button('Next page', function () {
    var q = withFrom(toolInput.query_dsl, from + n);
    if (q !== null) callBack(data, { query_dsl: q }, 'Loading page\u2026');
  }, 'callback page-next');
  next.disabled = n === 0 || (exact && from + n >= m.total);
  bar.appendChild(prev); bar.appendChild(document.createTextNode(' '));
  bar.appendChild(el('span', label, 'stat page-label')); bar.appendChild(document.createTextNode(' '));
  bar.appendChild(next);
  return bar;
}

function pplEditor(data) {
  if (data.kind !== 'ppl_rows' || !canCallBack(data) || typeof toolInput.query !== 'string') return null;
  var box = el('details', null, 'ppl-editor');
  box.open = resultsState.editorOpen;
  box.addEventListener('toggle', function () { resultsState.editorOpen = box.open; });
  box.appendChild(el('summary', 'Edit and re-run query'));
  var area = el('textarea', null, 'callback');
  area.rows = 3; area.value = toolInput.query; area.spellcheck = false;
  box.appendChild(area);
  var bar = el('div', null, 'toolbar');
  // The assistant's intent and suggestions described its own query, not the edited one.
  bar.appendChild(button('Run', function () {
    if (area.value.trim()) callBack(data, { query: area.value, intent: undefined, suggestions: undefined }, 'Running\u2026');
  }, 'callback ppl-run'));
  bar.appendChild(el('span', 'Runs any PPL you enter through ' + data.tool + ' with the same connection and other arguments; the assistant sees only its own calls.', 'muted'));
  box.appendChild(bar);
  return box;
}

function deepLink(data) {
  var m = data.meta || {};
  if (!hostCapabilities.openLinks) return null;
  if (typeof m.dashboards_url !== 'string' || !/^https?:\/\//i.test(m.dashboards_url)) return null;
  var link = button('View in OpenSearch Dashboards', null, 'deep-link');
  link.title = data.kind === 'search_hits' ? 'Opens the index in Discover; the query DSL is not carried over.' : 'Opens this PPL query in Discover.';
  link.addEventListener('click', function () {
    request('ui/open-link', { url: m.dashboards_url }).then(function (r) {
      if (r && r.isError) link.textContent = 'Host blocked the link';
    }, function () { link.textContent = 'Host could not open the link'; });
  });
  return link;
}

function renderResults(data) {
  var m = data.meta || {};
  var title = data.kind === 'ppl_rows' ? 'PPL' : 'Search ' + (m.index || '');
  var stats = [];
  if (m.total !== undefined && m.total !== null) {
    stats.push('total: ' + formatNumber(m.total) + (data.kind === 'search_hits' && m.total_relation === 'gte' ? '+' : ''));
  }
  var rowsStat = rowCountStat(data.rows.length);
  stats.push(rowsStat.node);
  if (m.took !== undefined && m.took !== null) stats.push('took: ' + formatNumber(m.took) + ' ms');
  if (m._shards) stats.push('shards: ' + m._shards.successful + '/' + m._shards.total);
  if (m.timed_out) stats.push('timed out');
  app.appendChild(header(data, title, stats, [deepLink(data)]));
  var paging = pagingControls(data);
  if (paging) { var pbar = el('div', null, 'toolbar paging-bar'); pbar.appendChild(paging); app.appendChild(pbar); }
  var editor = pplEditor(data);
  if (editor) app.appendChild(editor);
  var trunc = truncationNotice(data);
  if (trunc) app.appendChild(trunc);
  if (data.rows.length === 0) { if (!trunc) app.appendChild(el('div', 'No rows.', 'notice')); appendAggregations(m); return; }
  // Display shape: one level of nested objects becomes dotted columns. Copy and download
  // follow the displayed columns; the text result keeps the original documents.
  var flat = flattenColumns(data.columns, data.rows, m.column_types, VIEW_MAX_COLUMNS);
  var view = flat ? Object.assign({}, data, { columns: flat.columns, rows: flat.rows }) : data;
  var types = columnTypes(view.columns, view.rows, flat ? flat.types : m.column_types);
  var ts = timeSeriesSpec(view, types);
  var bars = ts ? null : barSpec(view, types);
  var chart = ts ? buildTimeSeries(view, ts, types) : bars ? buildBarChart(view, bars) : null;
  var section = buildDataSection(view, view.columns, view.rows, types, resultsState.table, {
    chipExclude: ['_id', '_score'],
    chipSkipNumeric: data.kind === 'ppl_rows',
    rowsStat: rowsStat,
    onRows: function (rows) { if (chart) chart.setRows(rows); }
  });
  if (section.chips) app.appendChild(section.chips);
  app.appendChild(section.toolbar);
  app.appendChild(section.stats);
  if (chart) {
    var bar = el('div', null, 'toolbar');
    var showChart = button('Chart'), showTable = button('Table');
    var mode = function (useChart) {
      resultsState.chart = useChart;
      showChart.setAttribute('aria-pressed', String(useChart));
      showTable.setAttribute('aria-pressed', String(!useChart));
      chart.node.hidden = !useChart; section.table.node.hidden = useChart;
    };
    showChart.addEventListener('click', function () { mode(true); });
    showTable.addEventListener('click', function () { mode(false); });
    bar.appendChild(showChart); bar.appendChild(showTable);
    app.appendChild(bar);
    app.appendChild(chart.node);
    chart.setRows(section.filtered());
    mode(resultsState.chart);
  }
  app.appendChild(section.table.node);
  appendAggregations(m);
}

function appendAggregations(m) {
  (m.aggregations || []).forEach(function (t) {
    var cut = truncationParts(t.rows.length, t.row_count, t.columns.length, t.column_count);
    var note = !t.truncated ? '' : cut.length ? ' (showing ' + cut.join(' and ') + ')' : ' (truncated)';
    app.appendChild(el('h3', 'Aggregation: ' + t.name + note));
    app.appendChild(buildTable(t.columns, t.rows, columnTypes(t.columns, t.rows), newTableState(), {}).node);
  });
}

startView({
  name: 'opensearch-results',
  unsupportedText: 'This output format has no table view. Use the jdbc (PPL) or json (search) format to see a table; the text result is unchanged.',
  accepts: function (data) { return (data.kind === 'search_hits' || data.kind === 'ppl_rows') && Array.isArray(data.columns) && Array.isArray(data.rows); },
  render: renderResults
});
