// Cluster overview for ClusterOverviewTool: health, nodes, indices, shards (unassigned reasons
// and allocation explanations) and disk allocation. Each section renders, fails and refreshes on
// its own; clicking a node or an unassigned shard filters the sections, clicking an index asks
// the assistant to open the index explorer for it.
var SECTION_ORDER = ['health', 'nodes', 'indices', 'shards', 'allocation'];
var SECTION_TITLES = { health: 'Health', nodes: 'Nodes', indices: 'Indices', shards: 'Shards', allocation: 'Disk allocation' };
var SECTION_TOOLS = { health: 'ClusterHealthTool', nodes: 'CatNodesTool', indices: 'ListIndexTool', shards: 'GetShardsTool', allocation: 'GetAllocationTool' };
var OVERVIEW_METRICS = HEALTH_METRICS.filter(function (m) { return ['number_of_in_flight_fetch', 'task_max_waiting_in_queue_millis', 'delayed_unassigned_shards'].indexOf(m[0]) < 0; });
var LARGEST_SHOWN = 10;
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

function setFilter(kind, value) {
  overviewState.filter[kind] = value;
  drawFilterBar();
  overview.requested.forEach(drawSection);
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
  bar.appendChild(button('Clear filters', function () { overviewState.filter = { node: null, index: null }; drawFilterBar(); overview.requested.forEach(drawSection); }, 'clear-filters'));
}

// Index names open the index explorer through the assistant: views cannot open other views.
function openExplorer(index) {
  var status = app.querySelector('.handoff-status');
  var tool = overview.tools && overview.tools.explorer;
  var where = overview.cluster ? ' on cluster "' + overview.cluster + '"' : '';
  if (!tool) {
    status.textContent = 'The index explorer is not available on this server. Ask the assistant about "' + index + '" instead.';
    return;
  }
  if (!canSendMessage()) {
    status.textContent = 'This host cannot send requests from views. Ask the assistant: "Open the index explorer for ' + index + '"' + where + '.';
    return;
  }
  status.textContent = 'Asking the assistant to open the index explorer for ' + index + '\u2026';
  sendUserMessage('Open ' + tool + ' for index "' + index + '"' + where + '.').then(function (ok) {
    status.textContent = ok ? 'Asked the assistant to open the index explorer for ' + index + '.' : 'The host did not send the request. Ask the assistant: "Open the index explorer for ' + index + '".';
  });
}

function refreshSection(name) {
  if (overviewState.busy[name] || callPending || !canCallBack(overview)) return;
  overviewState.busy[name] = true;
  delete overviewState.errors[name];
  drawSection(name);
  callTool(overview.tool, callArguments({ sections: [name] })).then(function (r) {
    overviewState.busy[name] = false;
    var sc = r.result && r.result.structuredContent;
    if (r.error) overviewState.errors[name] = r.error;
    else if (!sc || sc.kind !== 'cluster_overview' || sc.version !== PAYLOAD_VERSION || !sc.sections || !sc.sections[name]) overviewState.errors[name] = 'The tool returned no data for this section.';
    else { overview.sections[name] = sc.sections[name]; overviewState.refreshed[name] = new Date(); }
    drawSection(name);
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
    var b = button('Refresh', function () { refreshSection(name); }, 'callback section-refresh');
    b.setAttribute('aria-label', 'Refresh ' + SECTION_TITLES[name].toLowerCase());
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

function healthSection(sec, body) {
  body.appendChild(healthCard(sec.data || {}, OVERVIEW_METRICS));
}

function nodesSection(sec, body) {
  var rows = overviewState.filter.node !== null ? rowsWhere(sec, 'name', overviewState.filter.node) : sec.rows;
  body.appendChild(el('div', plural(sec.rows.length, 'node') + (rows.length !== sec.rows.length ? '; ' + formatNumber(rows.length) + ' shown' : '') + '. Select a node name to filter shards and disk use by it.', 'muted'));
  var note = cutNote(sec, sec.rows.length);
  if (note) body.appendChild(note);
  var nameCol = colIndex(sec, 'name');
  body.appendChild(buildTable(sec.columns, rows, sectionTypes(sec), sectionTable('nodes', ['ip', 'load_5m', 'load_15m']), {
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
  var types = sectionTypes(sec);
  var keep = ['index', 'health', 'status', 'docs.count', 'store.size', 'pri', 'rep'];
  var hidden = dict();
  sec.columns.forEach(function (c) { if (keep.indexOf(c) < 0) hidden[c] = true; });
  var action = { column: 'index', label: function (r) { return 'Open the index explorer for ' + cellText(r[colIndex(sec, 'index')]); }, run: function (r) { openExplorer(r[colIndex(sec, 'index')]); } };
  var f = overviewState.filter.index;
  if (f !== null) {
    body.appendChild(el('div', 'Index ' + cellText(f), 'label'));
    var st = sectionTable('indices-filtered'); st.hidden = hidden;
    body.appendChild(buildTable(sec.columns, rowsWhere(sec, 'index', f), types, st, { rowAction: action }).node);
    return;
  }
  body.appendChild(el('div', 'Largest indices', 'label'));
  var lt = sectionTable('indices-largest'); lt.hidden = hidden;
  body.appendChild(buildTable(sec.columns, sec.rows.slice(0, LARGEST_SHOWN), types, lt, { rowAction: action }).node);
  if (sec.rows.length > LARGEST_SHOWN) {
    var all = el('details', null, 'all-indices');
    all.open = overviewState.allIndices;
    all.appendChild(el('summary', 'All ' + plural(sec.rows.length, 'index', 'indices')));
    var built = false;
    var build = function () {
      if (built || !all.open) return;
      built = true;
      var at = sectionTable('indices-all');
      if (!at.init) { at.hidden = dict(hidden); at.init = true; }
      all.appendChild(buildTable(sec.columns, sec.rows, types, at, { picker: true, rowAction: action }).node);
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
    ut.hidden = dict({ state: true, docs: true, store: true, node: true });
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
  var high = sec.rows.filter(function (r) { return typeof r[pct] === 'number' && r[pct] >= 85; });
  if (high.length) body.appendChild(el('div', plural(high.length, 'node') + ' at or above 85% disk use (the default low watermark): ' + high.map(function (r) { return cellText(r[node]); }).join(', ') + '.', 'notice warning'));
  var t = leading(sec, ['node', 'disk.percent', 'shards']);
  var rows = f !== null ? t.rows.filter(function (r) { return r[0] === f; }) : t.rows;
  body.appendChild(buildTable(t.columns, rows, t.types, sectionTable('allocation', ['host', 'ip']), {
    picker: true,
    rowAction: { column: 'node', when: function (r) { return r[0] !== 'UNASSIGNED'; }, label: function (r) { return 'Filter by node ' + cellText(r[0]); }, run: function (r) { setFilter('node', r[0]); } }
  }).node);
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
  app.appendChild(header(Object.assign({}, data, { refreshable: false }), 'Cluster overview' + (name ? ': ' + name : ''), [h.status ? statusBadge(healthStatus(h)) : null], [
    canCallBack(data) ? button('Refresh all', function () { callBack(data, {}, 'Refreshing all sections\u2026'); }, 'callback refresh refresh-all') : null
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
  unsupportedText: 'This response has no cluster overview; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'cluster_overview' && !!data.sections && typeof data.sections === 'object' && Array.isArray(data.requested); },
  render: renderOverview
});
