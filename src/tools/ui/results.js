// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

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
  var label = n === 1 ? 'hit ' + formatNumber(from + 1) : n ? 'hits ' + formatNumber(from + 1) + '\u2013' + formatNumber(from + n) : from ? 'No hits after hit ' + formatNumber(from) : 'No hits';
  if (typeof m.total === 'number' && (n || m.total)) label += ' of ' + formatNumber(m.total) + (m.total_relation === 'gte' ? '+' : '');
  var prev = iconButton('Previous page', 'prev', function () {
    var q = withFrom(toolInput.query_dsl, Math.max(0, from - size));
    if (q !== null) callBack(data, { query_dsl: q }, 'Loading page\u2026');
  }, 'callback page-prev');
  prev.disabled = from === 0;
  var next = iconButton('Next page', 'next', function () {
    var q = withFrom(toolInput.query_dsl, from + n);
    if (q !== null) callBack(data, { query_dsl: q }, 'Loading page\u2026');
  }, 'callback page-next');
  next.disabled = n === 0 || (exact && from + n >= m.total);
  bar.appendChild(el('span', label, 'stat page-label'));
  bar.appendChild(prev);
  bar.appendChild(next);
  return bar;
}

// The PPL query as code, with commands and pipes set off.
var PPL_KEYWORDS = /^(source|search|where|fields|stats|sort|head|tail|eval|dedup|rename|top|rare|parse|grok|patterns|lookup|join|fillnull|trendline|eventstats|timechart|bin|by|as|span|and|or|not|in|like)$/i;
function pplQueryBlock(query) {
  var pre = el('pre', null, 'query');
  pre.setAttribute('aria-label', 'PPL query');
  String(query).split(/(\||\b[A-Za-z_]+\b)/).forEach(function (part) {
    if (!part) return;
    if (part === '|') pre.appendChild(el('span', part, 'pipe'));
    else if (PPL_KEYWORDS.test(part)) pre.appendChild(el('span', part, 'kw'));
    else pre.appendChild(document.createTextNode(part));
  });
  return pre;
}
// One line for the search DSL: the main clause, then size, sort and aggregations.
function dslSummary(dsl) {
  if (typeof dsl === 'string') { try { dsl = JSON.parse(dsl); } catch (e) { return clip(dsl, 240); } }
  if (!dsl || typeof dsl !== 'object' || Array.isArray(dsl)) return '';
  var q = dsl.query && typeof dsl.query === 'object' ? dsl.query : (dsl.size === undefined && !dsl.aggs && !dsl.sort ? dsl : null);
  var parts = [];
  var keys = q ? Object.keys(q) : [];
  if (keys.length === 1 && q[keys[0]] && typeof q[keys[0]] === 'object') {
    var k = keys[0], v = q[keys[0]];
    if (k === 'match_all') parts.push('match_all');
    else if (typeof v.query === 'string') parts.push(k + ' "' + v.query + '"' + (Array.isArray(v.fields) ? ' in ' + v.fields.join(', ') : ''));
    else if (k === 'bool') parts.push('bool: ' + ['must', 'filter', 'should', 'must_not'].filter(function (c) { return v[c]; }).map(function (c) { return c + ' ' + (Array.isArray(v[c]) ? v[c].length : 1); }).join(', '));
    else {
      var f = Object.keys(v)[0];
      var val = f && v[f] && typeof v[f] === 'object' ? (v[f].query !== undefined ? v[f].query : v[f].value) : f && v[f];
      parts.push(k + (f ? ' ' + f + (val !== undefined && typeof val !== 'object' ? ' = ' + JSON.stringify(val) : '') : ''));
    }
  } else if (q) parts.push(JSON.stringify(q));
  if (typeof dsl.size === 'number') parts.push('size ' + dsl.size);
  if (dsl.sort) parts.push('sort ' + JSON.stringify(dsl.sort));
  if (dsl.aggs || dsl.aggregations) parts.push('aggregations: ' + Object.keys(dsl.aggs || dsl.aggregations).join(', '));
  return clip(parts.join(' \u00b7 '), 240);
}
function searchQueryBlock() {
  if (!toolInput || toolInput.query_dsl === undefined) return null;
  var text = dslSummary(toolInput.query_dsl);
  if (!text) return null;
  var pre = el('pre', text, 'query');
  pre.setAttribute('aria-label', 'Query');
  pre.title = typeof toolInput.query_dsl === 'string' ? toolInput.query_dsl : JSON.stringify(toolInput.query_dsl, null, 2);
  return pre;
}
function pplSource(query) {
  var m = /\bsource\s*=\s*([^\s|]+)/i.exec(query || '');
  return m ? m[1].replace(/`/g, '') : null;
}
// A PPL title that says what the rows are, such as "count() by service"; the query block below
// already names the source. Charts name their axes; otherwise the last stats clause is used.
function pplTitle(view, ts, bars, query) {
  var col = function (i) { return cellText(view.columns[i], 60); };
  if (ts) return ts.values.map(col).join(', ') + (ts.category >= 0 ? ' by ' + col(ts.category) : '') + ' over time';
  if (bars) return col(bars.value) + ' by ' + col(bars.label);
  var stats = String(query || '').split('|').map(function (c) { return c.trim(); }).filter(function (c) { return /^stats\s/i.test(c); }).pop();
  if (stats) return clip(stats.replace(/^stats\s+/i, ''), 80);
  var source = pplSource(query);
  return source ? 'Rows from ' + source : 'PPL query';
}

function pplEditor(data) {
  if (data.kind !== 'ppl_rows' || !canCallBack(data) || typeof toolInput.query !== 'string') return null;
  var box = el('details', null, 'ppl-editor');
  box.open = resultsState.editorOpen;
  box.addEventListener('toggle', function () { resultsState.editorOpen = box.open; });
  box.appendChild(el('summary', 'Edit and re-run query'));
  var area = el('textarea', null, 'callback');
  area.setAttribute('aria-label', 'PPL query');
  area.rows = 3; area.value = toolInput.query; area.spellcheck = false;
  box.appendChild(area);
  var bar = el('div', null, 'toolbar');
  // The assistant's intent and suggestions described its own query, not the edited one.
  var run = function () {
    if (area.value.trim()) callBack(data, { query: area.value, intent: undefined, suggestions: undefined }, 'Running\u2026');
  };
  area.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); run(); } });
  bar.appendChild(withIcon(button('Run', run, 'primary callback ppl-run'), 'play'));
  bar.appendChild(el('span', 'Runs any PPL you enter through ' + data.tool + ' with the same connection and other arguments; the assistant sees only its own calls. Ctrl+Enter runs it.', 'muted'));
  box.appendChild(bar);
  return box;
}

function deepLink(data) {
  var m = data.meta || {};
  if (!hostCapabilities.openLinks) return null;
  if (typeof m.dashboards_url !== 'string' || !/^https?:\/\//i.test(m.dashboards_url)) return null;
  var link = button('View in OpenSearch Dashboards', null, 'deep-link');
  link.appendChild(icon('external'));
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
  var ppl = data.kind === 'ppl_rows';
  var stats = [];
  if (m.total !== undefined && m.total !== null) {
    stats.push('total: ' + formatNumber(m.total) + (data.kind === 'search_hits' && m.total_relation === 'gte' ? '+' : ''));
  }
  var paging = pagingControls(data);
  var rowsStat = rowCountStat(data.rows.length, !!paging);
  stats.push(rowsStat.node);
  if (m.took !== undefined && m.took !== null) stats.push('took: ' + formatNumber(m.took) + ' ms');
  if (m._shards) stats.push('shards: ' + m._shards.successful + '/' + m._shards.total);
  if (m.timed_out) stats.push('timed out');
  // Display shape: one level of nested objects becomes dotted columns. Copy and download
  // follow the displayed columns; the text result keeps the original documents.
  var flat = data.rows.length ? flattenColumns(data.columns, data.rows, m.column_types, VIEW_MAX_COLUMNS) : null;
  var view = flat ? Object.assign({}, data, { columns: flat.columns, rows: flat.rows }) : data;
  var types = columnTypes(view.columns, view.rows, flat ? flat.types : m.column_types, m.integer_strings);
  var ts = data.rows.length ? timeSeriesSpec(view, types) : null;
  var skipped = null;
  if (ts && ts.tooMany) {
    skipped = el('div', 'Not charted: ' + cellText(view.columns[ts.category]) + ' has ' + formatNumber(ts.tooMany) + ' distinct values, more than the ' + SERIES_SPLIT_MAX + ' a chart can tell apart. Filter the query, or use top or head to keep fewer.', 'notice chart-skipped');
    ts = null;
  }
  var bars = ts || skipped || !data.rows.length ? null : barSpec(view, types);
  var title = ppl ? pplTitle(view, ts, bars, m.query) : 'Search ' + (m.index || '');
  app.appendChild(header(data, title, stats, [deepLink(data)], { icon: ts ? 'line' : bars ? 'bars' : ppl ? 'table' : 'search' }));
  var queryBlock = ppl ? (typeof m.query === 'string' && m.query ? pplQueryBlock(m.query) : null) : searchQueryBlock();
  if (queryBlock) app.appendChild(queryBlock);
  var editor = pplEditor(data);
  if (editor) app.appendChild(editor);
  var trunc = truncationNotice(data);
  if (trunc) app.appendChild(trunc);
  if (skipped) app.appendChild(skipped);
  var pbar = null;
  if (paging) { pbar = el('div', null, 'toolbar paging-bar'); pbar.appendChild(paging); }
  if (data.rows.length === 0) {
    if (!trunc) app.appendChild(el('div', 'No rows.', 'notice empty'));
    if (pbar) app.appendChild(pbar);
    appendAggregations(m);
    return;
  }
  var chart = ts ? buildTimeSeries(view, ts, types) : bars ? buildBarChart(view, bars) : null;
  var section = buildDataSection(view, view.columns, view.rows, types, resultsState.table, {
    chipExclude: ['_id', '_score'],
    chipSkipNumeric: ppl,
    rowsStat: rowsStat,
    onRows: function (rows) { if (chart) chart.setRows(rows); }
  });
  if (section.chips) app.appendChild(section.chips);
  if (chart) {
    var seg = el('div', null, 'seg');
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', 'Show as');
    var showChart = withIcon(button('Chart'), ts ? 'line' : 'bars'), showTable = withIcon(button('Table'), 'table');
    var mode = function (useChart) {
      resultsState.chart = useChart;
      showChart.setAttribute('aria-pressed', String(useChart));
      showTable.setAttribute('aria-pressed', String(!useChart));
      chart.node.hidden = !useChart; section.table.node.hidden = useChart;
    };
    showChart.addEventListener('click', function () { mode(true); });
    showTable.addEventListener('click', function () { mode(false); });
    seg.appendChild(showChart); seg.appendChild(showTable);
    section.toolbar.insertBefore(seg, section.toolbar.firstChild);
  }
  app.appendChild(section.toolbar);
  app.appendChild(section.stats);
  if (chart) {
    app.appendChild(chart.node);
    chart.setRows(section.filtered());
    mode(resultsState.chart);
  }
  app.appendChild(section.table.node);
  if (pbar) app.appendChild(pbar);
  appendAggregations(m);
}

function appendAggregations(m) {
  (m.aggregations || []).forEach(function (t) {
    var cut = truncationParts(t.rows.length, t.row_count, t.columns.length, t.column_count);
    var note = !t.truncated ? '' : cut.length ? ' (showing ' + cut.join(' and ') + ')' : ' (truncated)';
    app.appendChild(el('h3', 'Aggregation: ' + t.name + note));
    app.appendChild(buildTable(t.columns, t.rows, columnTypes(t.columns, t.rows, null, t.integer_strings), newTableState(), {}).node);
  });
  (m.aggregation_notes || []).forEach(function (text) {
    if (typeof text === 'string') app.appendChild(el('div', text.charAt(0).toUpperCase() + text.slice(1) + '.', 'notice agg-note'));
  });
}

startView({
  name: 'opensearch-results',
  icon: 'search',
  unsupportedText: 'This output format has no table view. Use the jdbc (PPL) or json (search) format to see a table; the text result is unchanged.',
  accepts: function (data) { return (data.kind === 'search_hits' || data.kind === 'ppl_rows') && Array.isArray(data.columns) && Array.isArray(data.rows); },
  render: renderResults
});
