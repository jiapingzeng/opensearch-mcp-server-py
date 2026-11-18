// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

// Cluster overview for ClusterOverviewTool: health, nodes, indices, shards (unassigned reasons
// and allocation explanations) and disk allocation. Each section renders, fails and refreshes on
// its own; clicking a node or an unassigned shard filters the sections, clicking an index asks
// the assistant to open the index explorer for it.
var SECTION_ORDER = ['health', 'nodes', 'indices', 'shards', 'allocation'];
var SECTION_TITLES = { health: 'Health', nodes: 'Nodes', indices: 'Indices', shards: 'Shards', allocation: 'Disk allocation' };
var SECTION_TOOLS = { health: 'ClusterHealthTool', nodes: 'CatNodesTool', indices: 'ListIndexTool', shards: 'GetShardsTool', allocation: 'GetAllocationTool' };
var OVERVIEW_METRICS = HEALTH_METRICS.filter(function (m) { return ['number_of_in_flight_fetch', 'task_max_waiting_in_queue_millis', 'delayed_unassigned_shards'].indexOf(m[0]) < 0; });
var LARGEST_SHOWN = 10;
// OpenSearch default disk watermarks (low, high, flood stage); the cluster may override them.
var DISK_WATERMARKS = [85, 90, 95];
// Bars mark only the high watermark: three ticks in the last 15% crowd together.
var HIGH_WATERMARK_MARK = [{ at: DISK_WATERMARKS[1], label: 'High disk watermark, ' + DISK_WATERMARKS[1] + '% (OpenSearch default)' }];
// Node columns shown by default; the column picker has the rest.
var NODE_COLUMNS_SHOWN = ['name', 'node.role', 'heap.percent', 'cpu', 'disk.used_percent'];
var overviewState = { filter: { node: null, index: null }, tables: dict(), allIndices: false, allShards: false, busy: dict(), errors: dict(), refreshed: dict() };
var overview = null;
var sectionNodes = dict();

function sectionTable(name, hidden) {
  if (!overviewState.tables[name]) {
    overviewState.tables[name] = newTableState();
    (hidden || []).forEach(function (c) { overviewState.tables[name].hidden[c] = true; });
  }
  return overviewState.tables[name];
}
// The section with `first` columns moved to the front, so names stay visible in narrow panels.
function leading(sec, first) {
  var order = first.map(function (c) { return sec.columns.indexOf(c); }).filter(function (i) { return i >= 0; });
  sec.columns.forEach(function (_, i) { if (order.indexOf(i) < 0) order.push(i); });
  var types = sectionTypes(sec);
  return {
    columns: order.map(function (i) { return sec.columns[i]; }),
    rows: sec.rows.map(function (r) { return order.map(function (i) { return r[i]; }); }),
    types: order.map(function (i) { return types[i]; })
  };
}
function colIndex(sec, name) { return sec.columns.indexOf(name); }
function sectionTypes(sec) { var m = sec.meta || {}; return columnTypes(sec.columns, sec.rows, m.column_types, m.integer_strings); }
function rowsWhere(sec, column, value) {
  var i = colIndex(sec, column);
  return i < 0 ? sec.rows : sec.rows.filter(function (r) { return r[i] === value; });
}
function cutNote(sec, shown) {
  var m = sec.meta || {};
  if (!m.truncated) return null;
  var parts = truncationParts(shown, m.row_count, sec.columns.length, m.column_count);
  return el('div', (parts.length ? 'Showing ' + parts.join(' and ') : 'Shortened to fit the view') + '; the text tools (' + SECTION_TOOLS[sec.name] + ') return everything.', 'notice truncated');
}

// Health ignores the filters, so it is not redrawn: a redraw would drop focus inside it.
function drawFiltered() {
  overview.requested.forEach(function (n) { if (n !== 'health') drawSection(n); });
}
function setFilter(kind, value) {
  overviewState.filter[kind] = value;
  drawFilterBar();
  drawFiltered();
}
function drawFilterBar() {
  var bar = app.querySelector('.filter-bar');
  if (!bar) return;
  bar.textContent = '';
  var f = overviewState.filter;
  var active = ['node', 'index'].filter(function (k) { return f[k] !== null; });
  bar.hidden = !active.length;
  if (!active.length) return;
  bar.appendChild(el('span', 'Filtered by', 'name'));
  active.forEach(function (k) {
    var chip = button(k + ': ' + cellText(f[k]) + ' \u2715', function () { setFilter(k, null); }, 'chip filter-chip');
    chip.setAttribute('aria-label', 'Remove filter ' + k + ' ' + cellText(f[k]));
    bar.appendChild(chip);
  });
  bar.appendChild(button('Clear filters', function () { overviewState.filter = { node: null, index: null }; drawFilterBar(); drawFiltered(); }, 'clear-filters'));
}

// Names that would read differently in the message than on screen: hidden, control, format
// (bidi, zero-width) and separator characters.
var HIDDEN_CHAR_RE = new RegExp(HIDDEN_CHARS_RE.source, 'u');
var UNSENDABLE_RE = /[\p{Cc}\p{Cf}\p{Z}]/u;
function sendableName(v) { var s = rawText(v); return !HIDDEN_CHAR_RE.test(s) && !UNSENDABLE_RE.test(s); }

// Index names open the index explorer through the assistant: views cannot open other views.
function openExplorer(index) {
  var status = app.querySelector('.handoff-status');
  var tool = overview.tools && overview.tools.explorer;
  var name = cellText(index);
  var ask = 'Ask the assistant: "Open the index explorer for ' + name + '"' + (overview.cluster ? ' on cluster "' + cellText(overview.cluster) + '"' : '') + '.';
  if (!tool) {
    status.textContent = 'The index explorer is not available on this server. Ask the assistant about "' + name + '" instead.';
    return;
  }
  if (!canSendMessage()) {
    status.textContent = 'This host cannot send requests from views. ' + ask;
    return;
  }
  if (!sendableName(index) || (overview.cluster && !sendableName(overview.cluster))) {
    status.textContent = 'The index or cluster name has hidden or control characters, so the view does not send it. ' + ask;
    return;
  }
  var where = overview.cluster ? ' on cluster "' + overview.cluster + '"' : '';
  status.textContent = 'Asking the assistant to open the index explorer for ' + name + '\u2026';
  sendUserMessage('Open ' + tool + ' for index "' + index + '"' + where + '.').then(function (ok) {
    status.textContent = ok ? 'Asked the assistant to open the index explorer for ' + name + '.' : 'The host did not send the request. Ask the assistant: "Open the index explorer for ' + name + '".';
  });
}

function refreshSection(name) {
  if (overviewState.busy[name] || callPending || !canCallBack(overview)) return;
  overviewState.busy[name] = true;
  delete overviewState.errors[name];
  drawSection(name);
  var current = sameRender();
  callTool(overview.tool, callArguments({ sections: [name] })).then(function (r) {
    if (!current()) return;
    overviewState.busy[name] = false;
    var sc = r.result && r.result.structuredContent;
    if (r.error) overviewState.errors[name] = r.error;
    else if (!sc || sc.kind !== 'cluster_overview' || sc.version !== PAYLOAD_VERSION || !sc.sections || !sc.sections[name]) overviewState.errors[name] = 'The tool returned no data for this section.';
    else { overview.sections[name] = sc.sections[name]; overviewState.refreshed[name] = new Date(); }
    drawSection(name);
    // The health tiles summarise indices and disk use; only the tiles are redrawn.
    if (!r.error && (name === 'indices' || name === 'allocation')) redrawHealthTiles();
  });
}

function sectionShell(name) {
  var sec = el('section', null, 'panel section section-' + name);
  sec.setAttribute('aria-labelledby', 'sec-' + name);
  var head = el('div', null, 'section-head');
  var h = el('h2', SECTION_TITLES[name]);
  h.id = 'sec-' + name;
  head.appendChild(h);
  var status = el('span', null, 'muted section-status');
  status.setAttribute('role', 'status');
  if (overviewState.busy[name]) status.textContent = 'Refreshing\u2026';
  else if (overviewState.refreshed[name]) status.textContent = 'refreshed ' + overviewState.refreshed[name].toISOString().slice(11, 19) + ' UTC';
  head.appendChild(status);
  head.appendChild(el('span', null, 'spacer'));
  if (canCallBack(overview)) {
    var b = iconButton('Refresh ' + SECTION_TITLES[name].toLowerCase(), 'refresh', function () { refreshSection(name); }, 'callback section-refresh');
    b.disabled = !!overviewState.busy[name] || callPending;
    head.appendChild(b);
  }
  sec.appendChild(head);
  return sec;
}

function drawSection(name) {
  var data = overview.sections[name] || { status: 'error', error: 'This section is missing from the result.' };
  var node = sectionShell(name);
  if (overviewState.errors[name]) {
    var banner = el('div', null, 'notice error section-error');
    banner.setAttribute('role', 'alert');
    banner.appendChild(errorBlock(overviewState.errors[name]));
    node.appendChild(banner);
    if (data.status === 'ok') node.appendChild(el('div', 'Showing the previous result.', 'muted stale-label'));
  }
  var body = el('div', null, 'section-body');
  if (data.status === 'error') {
    var notice = el('div', null, 'notice error');
    notice.appendChild(errorBlock(data.error || 'This section failed.'));
    body.appendChild(notice);
  } else if (data.status === 'disabled') {
    body.appendChild(el('div', 'Off: this section needs ' + cellText(data.tool) + ', which is not enabled on this server.', 'notice'));
  } else if (data.status === 'ok') {
    data.name = name;
    try { SECTION_RENDERERS[name](data, body); } catch (e) {
      body.textContent = '';
      body.appendChild(el('div', 'This section could not be shown (' + e.message + '); the other sections are unaffected.', 'notice error'));
    }
  } else {
    body.appendChild(el('div', 'This section has an unknown state; the text result is unchanged.', 'notice'));
  }
  if (overviewState.errors[name] && data.status === 'ok') body.classList.add('stale');
  node.appendChild(body);
  if (sectionNodes[name] && sectionNodes[name].parentNode) sectionNodes[name].parentNode.replaceChild(node, sectionNodes[name]);
  sectionNodes[name] = node;
  return node;
}
// Replaces the health section's tiles in place, leaving its heading and buttons (and focus) alone.
function redrawHealthTiles() {
  var sec = overview.sections.health;
  var node = sectionNodes.health;
  if (!sec || sec.status !== 'ok' || !node || overviewState.busy.health) return;
  var old = node.querySelector('.health-tiles');
  if (!old) return;
  var fresh = healthTiles(sec);
  old.parentNode.replaceChild(fresh, old);
}

// Disk totals across data nodes from the allocation section, or null when it is not loaded.
function diskTotals() {
  var sec = overview.sections.allocation;
  if (!sec || sec.status !== 'ok' || !Array.isArray(sec.columns)) return null;
  var used = colIndex(sec, 'disk.used'), total = colIndex(sec, 'disk.total');
  if (used < 0 || total < 0) return null;
  var u = 0, t = 0;
  sec.rows.forEach(function (r) { if (typeof r[used] === 'number' && typeof r[total] === 'number') { u += r[used]; t += r[total]; } });
  return t > 0 ? { used: u, total: t, pct: 100 * u / t } : null;
}
function diskTone(pct) { return pct >= DISK_WATERMARKS[1] ? 'high' : pct >= DISK_WATERMARKS[0] ? 'warn' : ''; }

// KPI row: status, nodes, indices, shards, unassigned and disk used, then the non-zero smaller counters.
function healthSection(sec, body) { body.appendChild(healthTiles(sec)); }
function healthTiles(sec) {
  var h = sec.data || {};
  var wrap = el('div', null, 'health-tiles section-body');
  var main = el('div', null, 'card kpis');
  main.appendChild(statusTile(h));
  var tile = function (key, label, sub) {
    if (!(key in h)) return null;
    var spec = HEALTH_METRICS.filter(function (m) { return m[0] === key; })[0];
    var t = metricTile(h, [key, label, spec && spec[2]]);
    if (sub) t.appendChild(el('div', sub, 'kpi-sub'));
    return t;
  };
  var add = function (n) { if (n) main.appendChild(n); };
  add(tile('number_of_nodes', 'Nodes', typeof h.number_of_data_nodes === 'number' ? plural(h.number_of_data_nodes, 'data node') : null));
  var idx = overview.sections.indices;
  if (idx && idx.status === 'ok' && idx.summary) {
    var by = idx.summary.by_health || {};
    var parts = ['green', 'yellow', 'red'].filter(function (k) { return by[k]; }).map(function (k) { return formatNumber(by[k]) + ' ' + k; });
    add(kpiTile('Indices', formatNumber(idx.summary.count || 0), parts.join(' \u00b7 ') || null, null, 'kpi-indices'));
  }
  add(tile('active_shards', 'Active shards', typeof h.active_primary_shards === 'number' ? plural(h.active_primary_shards, 'primary', 'primaries') : null));
  add(tile('unassigned_shards', 'Unassigned', h.unassigned_shards > 0 && typeof h.delayed_unassigned_shards === 'number' && h.delayed_unassigned_shards ? formatNumber(h.delayed_unassigned_shards) + ' delayed' : null));
  var disk = diskTotals();
  if (disk) {
    var value = el('span', formatPercent(disk.pct));
    var t = kpiTile('Disk used', value, formatBytes(disk.used) + ' of ' + formatBytes(disk.total), diskTone(disk.pct) === 'high' ? 'alert' : diskTone(disk.pct) ? 'warn' : null, 'kpi-disk');
    t.appendChild(usageBar(disk.pct, formatPercent(disk.pct) + ' of disk used across data nodes', diskTone(disk.pct), HIGH_WATERMARK_MARK));
    add(t);
  }
  wrap.appendChild(main);
  var rest = el('div', null, 'card secondary');
  OVERVIEW_METRICS.forEach(function (spec) {
    if (shownSecondary(h, spec[0]) && PRIMARY_METRICS.indexOf(spec[0]) < 0) rest.appendChild(metricTile(h, spec));
  });
  if (rest.childNodes.length) wrap.appendChild(rest);
  return wrap;
}

function nodesSection(sec, body) {
  var rows = overviewState.filter.node !== null ? rowsWhere(sec, 'name', overviewState.filter.node) : sec.rows;
  body.appendChild(el('div', plural(sec.rows.length, 'node') + (rows.length !== sec.rows.length ? '; ' + formatNumber(rows.length) + ' shown' : '') + '. Select a node name to filter shards and disk use by it.', 'muted'));
  var note = cutNote(sec, sec.rows.length);
  if (note) body.appendChild(note);
  var nameCol = colIndex(sec, 'name');
  body.appendChild(buildTable(sec.columns, rows, sectionTypes(sec), sectionTable('nodes', sec.columns.filter(function (c) { return NODE_COLUMNS_SHOWN.indexOf(c) < 0; })), {
    picker: true,
    rowAction: { column: 'name', label: function (r) { return 'Filter by node ' + cellText(r[nameCol]); }, run: function (r) { setFilter('node', r[nameCol]); } }
  }).node);
}

function healthCounts(byHealth) {
  var line = el('div', null, 'toolbar health-counts');
  ['green', 'yellow', 'red'].concat(Object.keys(byHealth).filter(function (k) { return ['green', 'yellow', 'red'].indexOf(k) < 0; })).forEach(function (k) {
    if (!(k in byHealth)) return;
    line.appendChild(el('span', formatNumber(byHealth[k]) + ' ' + k, 'count status-text-' + k));
  });
  return line;
}

function indicesSection(sec, body) {
  var sm = sec.summary || {};
  var head = healthCounts(sm.by_health || {});
  head.insertBefore(el('span', plural(sm.count || 0, 'index', 'indices'), 'stat'), head.firstChild);
  head.appendChild(el('span', plural(sm.docs || 0, 'document'), 'stat'));
  head.appendChild(el('span', formatBytes(sm.store_bytes || 0) + ' total', 'stat'));
  body.appendChild(head);
  var note = cutNote(sec, sec.rows.length);
  if (note) body.appendChild(note);
  // Name and size first, so the size stays in view in a half-width card and at 360px.
  var t = leading(sec, ['index', 'store.size', 'health', 'docs.count', 'pri', 'rep']);
  // pri and rep are in the column picker; four columns fit a half-width card without scrolling.
  var keep = ['index', 'health', 'docs.count', 'store.size'];
  var hidden = dict();
  t.columns.forEach(function (c) { if (keep.indexOf(c) < 0) hidden[c] = true; });
  var idx = t.columns.indexOf('index');
  var action = { column: 'index', label: function (r) { return 'Open the index explorer for ' + cellText(r[idx]); }, run: function (r) { openExplorer(r[idx]); } };
  var f = overviewState.filter.index;
  if (f !== null) {
    body.appendChild(el('div', 'Index ' + cellText(f), 'label'));
    var st = sectionTable('indices-filtered'); st.hidden = hidden;
    body.appendChild(buildTable(t.columns, t.rows.filter(function (r) { return r[idx] === f; }), t.types, st, { rowAction: action }).node);
    return;
  }
  body.appendChild(el('div', 'Largest indices', 'label'));
  var lt = sectionTable('indices-largest'); lt.hidden = hidden;
  var largest = buildTable(t.columns, t.rows.slice(0, LARGEST_SHOWN), t.types, lt, { rowAction: action, barColumns: ['store.size'] }).node;
  largest.classList.add('largest-indices');
  body.appendChild(largest);
  if (t.rows.length > LARGEST_SHOWN) {
    var all = el('details', null, 'all-indices');
    all.open = overviewState.allIndices;
    all.appendChild(el('summary', 'All ' + plural(t.rows.length, 'index', 'indices')));
    var built = false;
    var build = function () {
      if (built || !all.open) return;
      built = true;
      var at = sectionTable('indices-all');
      if (!at.init) { at.hidden = dict(hidden); at.init = true; }
      all.appendChild(buildTable(t.columns, t.rows, t.types, at, { picker: true, rowAction: action }).node);
    };
    all.addEventListener('toggle', function () { overviewState.allIndices = all.open; build(); });
    body.appendChild(all);
    build();
  }
}

function explainRequest(index, shard, primary) {
  return 'GET _cluster/allocation/explain\n' + JSON.stringify({ index: index, shard: Number(shard), primary: primary === 'p' });
}

function shardsSection(sec, body) {
  var sm = sec.summary || {};
  var f = overviewState.filter;
  var line = el('div', null, 'toolbar shard-states');
  line.appendChild(el('span', plural(sm.total || 0, 'shard'), 'stat'));
  Object.keys(sm.by_state || {}).forEach(function (s) { line.appendChild(el('span', formatNumber(sm.by_state[s]) + ' ' + s.toLowerCase(), 'count status-text-' + s.toLowerCase())); });
  body.appendChild(line);
  var note = cutNote(sec, sec.rows.length);
  if (note) body.appendChild(note);
  if (f.node !== null && sm.per_node) {
    var counts = sm.per_node[f.node] || { primaries: 0, replicas: 0 };
    body.appendChild(el('div', 'On ' + cellText(f.node) + ': ' + plural(counts.primaries, 'primary', 'primaries') + ', ' + plural(counts.replicas, 'replica') + '.', 'muted node-counts'));
  }
  var stateCol = colIndex(sec, 'state'), indexCol = colIndex(sec, 'index'), nodeCol = colIndex(sec, 'node');
  var unassigned = sec.rows.filter(function (r) { return r[stateCol] === 'UNASSIGNED' && (f.index === null || r[indexCol] === f.index); });
  var types = sectionTypes(sec);
  if (sm.unassigned) {
    var title = el('div', null, 'label');
    title.textContent = 'Unassigned shards (' + formatNumber(sm.unassigned) + (f.index !== null ? '; ' + formatNumber(unassigned.length) + ' in ' + cellText(f.index) : '') + ')';
    body.appendChild(title);
    var reasons = Object.keys(sm.unassigned_by_reason || {}).map(function (k) { return k + ' ' + formatNumber(sm.unassigned_by_reason[k]); });
    if (reasons.length) body.appendChild(el('div', 'Reasons: ' + reasons.join(', ') + '. Select an index to filter by it.', 'muted'));
    var ut = sectionTable('unassigned');
    // unassigned.for would push the table past a half-width card.
    ut.hidden = dict({ state: true, docs: true, store: true, node: true, 'unassigned.for': true });
    body.appendChild(buildTable(sec.columns, unassigned, types, ut, {
      rowAction: { column: 'index', label: function (r) { return 'Filter by index ' + cellText(r[indexCol]) + ', shard ' + cellText(r[colIndex(sec, 'shard')]); }, run: function (r) { setFilter('index', r[indexCol]); } }
    }).node);
    body.appendChild(explanations(sec, unassigned));
  } else {
    body.appendChild(el('div', 'No unassigned shards.', 'muted success'));
  }
  if (f.node !== null || f.index !== null) {
    var matching = sec.rows.filter(function (r) { return (f.node === null || r[nodeCol] === f.node) && (f.index === null || r[indexCol] === f.index); });
    body.appendChild(el('div', 'Shards matching the filter (' + formatNumber(matching.length) + ')', 'label'));
    body.appendChild(buildTable(sec.columns, matching, types, sectionTable('shards-filtered')).node);
  }
}

function explanations(sec, unassigned) {
  var wrap = el('div', null, 'explanations');
  var shown = (sec.explanations || []).filter(function (e) { return overviewState.filter.index === null || e.index === overviewState.filter.index; });
  shown.forEach(function (e) {
    var box = el('div', null, 'panel explanation');
    box.appendChild(el('div', cellText(e.index) + '[' + cellText(e.shard) + '] ' + (e.primary ? 'primary' : 'replica') + ': can allocate ' + cellText(e.can_allocate), 'name'));
    if (e.explanation) box.appendChild(el('div', e.explanation));
    var list = el('ul', null, 'reasons');
    (e.nodes || []).forEach(function (n) {
      (n.reasons || []).forEach(function (reason) { list.appendChild(el('li', cellText(n.node) + ' (' + cellText(n.decision) + '): ' + reason)); });
    });
    if (list.childNodes.length) box.appendChild(list);
    wrap.appendChild(box);
  });
  if (sec.explain_error) {
    var err = el('div', null, 'notice error');
    err.appendChild(errorBlock(sec.explain_error));
    wrap.appendChild(err);
  }
  // Shards without an explanation: offer the read-only request instead of calling it.
  var explained = dict();
  (sec.explanations || []).forEach(function (e) { explained[e.index + '\u0000' + e.shard + '\u0000' + (e.primary ? 'p' : 'r')] = true; });
  var c = function (n) { return colIndex(sec, n); };
  var next = unassigned.filter(function (r) { return !explained[r[c('index')] + '\u0000' + Number(r[c('shard')]) + '\u0000' + r[c('prirep')]]; })[0];
  if (next) {
    var bar = el('div', null, 'toolbar');
    var status = el('span', null, 'muted copy-status');
    var text = explainRequest(next[c('index')], next[c('shard')], next[c('prirep')]);
    bar.appendChild(el('span', 'Explain ' + cellText(next[c('index')]) + '[' + cellText(next[c('shard')]) + '] with _cluster/allocation/explain:', 'muted'));
    bar.appendChild(button('Copy request', function () { copyText(text, 'request', status); }, 'copy-explain'));
    bar.appendChild(status);
    wrap.appendChild(bar);
  }
  return wrap;
}

function allocationSection(sec, body) {
  var f = overviewState.filter.node;
  var pct = colIndex(sec, 'disk.percent'), node = colIndex(sec, 'node');
  var high = sec.rows.filter(function (r) { return typeof r[pct] === 'number' && r[pct] >= DISK_WATERMARKS[0]; });
  if (high.length) body.appendChild(el('div', plural(high.length, 'node') + ' at or above 85% disk use (the default low watermark): ' + high.map(function (r) { return cellText(r[node]); }).join(', ') + '.', 'notice warning'));
  var t = leading(sec, ['node', 'disk.percent', 'shards']);
  var rows = f !== null ? t.rows.filter(function (r) { return r[0] === f; }) : t.rows;
  body.appendChild(diskBars(sec, rows.map(function (r) { return sec.rows[t.rows.indexOf(r)]; })));
  var details = el('details', null, 'disk-table');
  details.open = !!overviewState.diskTable;
  details.addEventListener('toggle', function () { overviewState.diskTable = details.open; });
  details.appendChild(el('summary', 'Disk allocation table'));
  details.appendChild(buildTable(t.columns, rows, t.types, sectionTable('allocation', ['host', 'ip']), {
    picker: true,
    rowAction: { column: 'node', when: function (r) { return r[0] !== 'UNASSIGNED'; }, label: function (r) { return 'Filter by node ' + cellText(r[0]); }, run: function (r) { setFilter('node', r[0]); } }
  }).node);
  body.appendChild(details);
}

// One usage bar per data node with ticks at the default watermarks; unassigned shards below.
function diskBars(sec, rows) {
  var c = function (n) { return colIndex(sec, n); };
  var wrap = el('div', null, 'disk-list');
  var unassigned = 0;
  rows.forEach(function (r) {
    var name = r[c('node')];
    if (name === 'UNASSIGNED') { if (typeof r[c('shards')] === 'number') unassigned += r[c('shards')]; return; }
    var pct = r[c('disk.percent')];
    var row = el('div', null, 'disk-row');
    var who = el('div', null, 'who');
    var link = button(cellText(name), function () { setFilter('node', name); }, 'link disk-node');
    link.setAttribute('aria-label', 'Filter by node ' + cellText(name));
    who.appendChild(link);
    row.appendChild(who);
    row.appendChild(usageBar(pct, cellText(name) + ': ' + (typeof pct === 'number' ? pct + '% disk used' : 'disk use unknown'), typeof pct === 'number' ? diskTone(pct) : '', HIGH_WATERMARK_MARK));
    var figs = el('div', null, 'figs');
    figs.appendChild(el('strong', typeof pct === 'number' ? pct + '%' : '\u2014'));
    var used = r[c('disk.used')], total = r[c('disk.total')], shards = r[c('shards')];
    var extra = [];
    if (typeof used === 'number' && typeof total === 'number') extra.push(formatBytes(used) + ' of ' + formatBytes(total));
    if (typeof shards === 'number') extra.push(plural(shards, 'shard'));
    if (extra.length) figs.appendChild(document.createTextNode(' \u00b7 ' + extra.join(' \u00b7 ')));
    row.appendChild(figs);
    wrap.appendChild(row);
  });
  if (!wrap.childNodes.length) wrap.appendChild(el('div', 'No data nodes report disk use.', 'notice empty'));
  var legend = el('div', null, 'legend-line');
  var item = el('span');
  item.appendChild(el('span', null, 'tick'));
  item.appendChild(document.createTextNode('High watermark ' + DISK_WATERMARKS[1] + '% (OpenSearch default; low ' + DISK_WATERMARKS[0] + '%, flood stage ' + DISK_WATERMARKS[2] + '%)'));
  legend.appendChild(item);
  if (unassigned) legend.appendChild(el('span', plural(unassigned, 'shard') + ' unassigned', 'danger'));
  wrap.appendChild(legend);
  return wrap;
}

var SECTION_RENDERERS = { health: healthSection, nodes: nodesSection, indices: indicesSection, shards: shardsSection, allocation: allocationSection };

function renderOverview(data) {
  overview = data;
  sectionNodes = dict();
  overviewState.errors = dict();
  overviewState.busy = dict();
  if (refreshedAt) overviewState.refreshed = dict();
  var h = data.sections.health && data.sections.health.status === 'ok' ? data.sections.health.data || {} : {};
  var name = h.cluster_name || '';
  // The header's own Refresh would say less; this one names what it refreshes.
  // Status is the first health tile, so the header has no status badge.
  app.appendChild(header(Object.assign({}, data, { refreshable: false }), 'Cluster overview' + (name ? ': ' + name : ''), [], [
    canCallBack(data) ? withIcon(button('Refresh all', function () { callBack(data, {}, 'Refreshing all sections\u2026'); }, 'callback refresh refresh-all'), 'refresh') : null
  ]));
  if (!canCallBack(data)) app.appendChild(el('div', 'This host does not let views call tools, so sections cannot be refreshed here. Ask the assistant to run ' + (data.tool || 'ClusterOverviewTool') + ' again.', 'notice degraded'));
  var status = el('div', null, 'muted handoff-status');
  status.setAttribute('role', 'status');
  app.appendChild(status);
  var bar = el('div', null, 'chips filter-bar');
  app.appendChild(bar);
  drawFilterBar();
  var grid = el('div', null, 'sections');
  overview.requested = (data.requested || []).filter(function (n) { return SECTION_ORDER.indexOf(n) >= 0; });
  overview.requested.forEach(function (n) { grid.appendChild(drawSection(n)); });
  app.appendChild(grid);
}

startView({
  name: 'opensearch-cluster-overview',
  icon: 'cluster',
  unsupportedText: 'This response has no cluster overview; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'cluster_overview' && !!data.sections && typeof data.sections === 'object' && Array.isArray(data.requested); },
  render: renderOverview
});
