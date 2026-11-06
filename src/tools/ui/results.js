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
  var label = n ? 'hits ' + (from + 1) + '\u2013' + (from + n) : 'no hits from ' + (from + 1);
  if (typeof m.total === 'number') label += ' of ' + m.total + (m.total_relation === 'gte' ? '+' : '');
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
  bar.appendChild(button('Run', function () { if (area.value.trim()) callBack(data, { query: area.value }, 'Running\u2026'); }, 'callback ppl-run'));
  bar.appendChild(el('span', 'Runs ' + data.tool + ' from this view with the other arguments unchanged; the assistant sees only its own calls.', 'muted'));
  box.appendChild(bar);
  return box;
}

function deepLink(data) {
  var m = data.meta || {};
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
  if (data.kind === 'search_hits') {
    if (m.total !== undefined && m.total !== null) stats.push('total: ' + m.total + (m.total_relation === 'gte' ? '+' : ''));
  } else if (m.total !== undefined && m.total !== null) stats.push('total: ' + m.total);
  stats.push('rows: ' + data.rows.length);
  if (m.took !== undefined && m.took !== null) stats.push('took: ' + m.took + ' ms');
  if (m._shards) stats.push('shards: ' + m._shards.successful + '/' + m._shards.total);
  if (m.timed_out) stats.push('timed out');
  app.appendChild(header(data, title, stats, [deepLink(data)]));
  var paging = pagingControls(data);
  if (paging) { var pbar = el('div', null, 'toolbar'); pbar.appendChild(paging); app.appendChild(pbar); }
  var editor = pplEditor(data);
  if (editor) app.appendChild(editor);
  var trunc = truncationNotice(data);
  if (trunc) app.appendChild(trunc);
  if (data.rows.length === 0) { if (!trunc) app.appendChild(el('div', 'No rows.', 'notice')); return; }
  var types = columnTypes(data.columns, data.rows, m.column_types);
  var ts = timeSeriesSpec(data, types);
  var bars = ts ? null : barSpec(data, types);
  var chart = ts ? buildTimeSeries(data, ts) : bars ? buildBarChart(data, bars) : null;
  var section = buildDataSection(data, data.columns, data.rows, types, resultsState.table, {
    chipExclude: ['_id', '_score'],
    chipSkipNumeric: data.kind === 'ppl_rows',
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
  (m.aggregations || []).forEach(function (t) {
    app.appendChild(el('h3', 'Aggregation: ' + t.name + (t.truncated ? ' (truncated)' : '')));
    app.appendChild(buildTable(t.columns, t.rows, columnTypes(t.columns, t.rows), newTableState(), {}).node);
  });
}

startView({
  name: 'opensearch-results',
  unsupportedText: 'This output format has no table view. Use the jdbc (PPL) or json (search) format to see a table; the text result is unchanged.',
  accepts: function (data) { return (data.kind === 'search_hits' || data.kind === 'ppl_rows') && Array.isArray(data.columns) && Array.isArray(data.rows); },
  render: renderResults
});
