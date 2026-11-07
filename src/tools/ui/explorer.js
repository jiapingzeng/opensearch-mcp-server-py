// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

// Index explorer for IndexExplorerTool: a searchable index list and, for the selected index,
// Overview, Mapping, Shards and Sample documents tabs. Tabs load through the existing read-only
// tools (named in data.tools) on first open and are cached per index and tab.
var TABS = [['overview', 'Overview'], ['mapping', 'Mapping'], ['shards', 'Shards'], ['samples', 'Sample documents']];
var OWN_ARGS = ['index', 'intent'];
var KEY_SETTINGS = ['number_of_shards', 'number_of_replicas', 'auto_expand_replicas', 'refresh_interval', 'codec',
  'creation_date', 'version.created', 'uuid', 'hidden', 'knn', 'max_result_window'];
var explorerState = { query: '', selected: null, tab: 'overview', cache: dict(), tables: dict(), mappings: dict() };
var explorer = null;

function listColumn(name) { return explorer.list && explorer.list.columns ? explorer.list.columns.indexOf(name) : -1; }
function listRow(index) {
  var i = listColumn('index');
  if (i < 0) return null;
  return explorer.list.rows.filter(function (r) { return r[i] === index; })[0] || null;
}
function listed() {
  var i = listColumn('index');
  return explorer.list.status === 'ok' && i >= 0 ? explorer.list.rows.map(function (r) { return r[i]; }) : [];
}
function cacheKey(index, tab) { return index + '\u0000' + tab; }
function canDrill() { return canCallTools(); }
// The tool and arguments each tab calls; every call repeats the cluster and connection arguments.
function tabCall(tab, index) {
  var spec = {
    overview: ['detail', { index: index, include_detail: true }],
    mapping: ['mapping', { index: index }],
    shards: ['shards', { index: index }],
    samples: ['search', { index: index, query_dsl: { query: { match_all: {} } }, size: explorer.sample_size || 10 }]
  }[tab];
  var tool = explorer.tools && explorer.tools[spec[0]];
  return tool ? { name: tool, arguments: Object.assign(passThroughArgs(OWN_ARGS), spec[1]) } : null;
}
var TAB_KINDS = { overview: 'index_detail', mapping: 'index_mapping', shards: 'cat_table', samples: 'search_hits' };
var ASK = {
  overview: 'Show the settings and aliases of ', mapping: 'Show the mapping of ', shards: 'Show the shards of ', samples: 'Show sample documents from '
};

function load(index, tab) {
  var key = cacheKey(index, tab);
  if (explorerState.cache[key]) return;
  var call = tabCall(tab, index);
  if (!call) { explorerState.cache[key] = { status: 'error', error: 'The tool for this tab is not enabled on this server.' }; return; }
  explorerState.cache[key] = { status: 'loading' };
  var current = sameRender();
  callTool(call.name, call.arguments).then(function (r) {
    if (!current()) return;
    var sc = r.result && r.result.structuredContent;
    var entry;
    if (r.error) entry = { status: 'error', error: r.error };
    else if (!sc || sc.version !== PAYLOAD_VERSION) entry = { status: 'error', error: 'The tool returned no view data (unsupported or newer payload version); its text result is unchanged.' };
    else if (sc.omitted) entry = { status: 'error', error: sc.omitted === 'too_large' ? 'The result is too large for the 1 MiB view limit; ask the assistant instead.' : 'The tool returned no view data for this index.' };
    else if (sc.kind !== TAB_KINDS[tab]) entry = { status: 'error', error: 'The tool returned an unexpected result for this tab.' };
    else entry = { status: 'ok', data: sc };
    explorerState.cache[key] = entry;
    if (explorerState.selected === index && explorerState.tab === tab) drawPanel();
  });
}

function retryButton(index, tab) {
  return button('Retry', function () { delete explorerState.cache[cacheKey(index, tab)]; drawPanel(); }, 'retry');
}

function healthDot(health) {
  var dot = el('span', null, 'dot status-bg-' + (['green', 'yellow', 'red'].indexOf(health) >= 0 ? health : 'unknown'));
  dot.setAttribute('aria-hidden', 'true');
  return dot;
}

function drawList() {
  var holder = app.querySelector('.index-list');
  var count = app.querySelector('.list-count');
  if (!holder) return;
  holder.textContent = '';
  var lst = explorer.list;
  if (lst.status !== 'ok') {
    var err = el('li', null, 'notice error');
    err.appendChild(errorBlock(lst.error || 'The index list failed.'));
    holder.appendChild(err);
    count.textContent = '';
    return;
  }
  var i = listColumn('index'), h = listColumn('health'), d = listColumn('docs.count'), s = listColumn('store.size');
  var q = explorerState.query.trim().toLowerCase();
  var rows = lst.rows.filter(function (r) { return !q || String(r[i]).toLowerCase().indexOf(q) >= 0; });
  var m = lst.meta || {};
  var total = m.row_count || lst.rows.length;
  count.textContent = (rows.length === lst.rows.length ? plural(lst.rows.length, 'index', 'indices') : formatNumber(rows.length) + ' of ' + plural(lst.rows.length, 'index', 'indices')) +
    (total > lst.rows.length ? ' (first ' + formatNumber(lst.rows.length) + ' of ' + formatNumber(total) + ')' : '');
  rows.forEach(function (r) {
    var li = el('li');
    var b = button(null, function () { select(r[i]); }, 'index-item');
    b.dataset.index = String(r[i]);
    var current = r[i] === explorerState.selected;
    b.setAttribute('aria-current', current ? 'true' : 'false');
    b.appendChild(healthDot(r[h]));
    b.appendChild(el('span', cellText(r[i], 60), 'name'));
    // The dot's meaning for screen readers; the button's name is its visible text plus this.
    b.appendChild(el('span', ' health ' + (r[h] || 'unknown'), 'sr-only'));
    var meta = [];
    if (d >= 0 && typeof r[d] === 'number') meta.push(plural(r[d], 'doc'));
    if (s >= 0 && typeof r[s] === 'number') meta.push(formatBytes(r[s]));
    b.appendChild(el('span', meta.join(' \u00b7 '), 'muted meta'));
    li.appendChild(b);
    holder.appendChild(li);
  });
  if (!rows.length) holder.appendChild(el('li', q ? 'No indices match.' : 'No indices.', 'muted'));
}

// Arrow keys move between indices; Home and End jump to the ends.
function listKeys(ev) {
  var items = Array.prototype.slice.call(app.querySelectorAll('.index-list button.index-item'));
  var at = items.indexOf(document.activeElement);
  if (at < 0) return;
  var next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[ev.key];
  if (next === undefined) return;
  ev.preventDefault();
  if (items[Math.max(0, Math.min(items.length - 1, next))]) items[Math.max(0, Math.min(items.length - 1, next))].focus();
}

function select(index) {
  if (explorerState.selected === index) return;
  explorerState.selected = index;
  app.querySelectorAll('.index-list button.index-item').forEach(function (b) { b.setAttribute('aria-current', b.dataset.index === String(index) ? 'true' : 'false'); });
  drawDetail();
}

function setTab(tab, focus) {
  explorerState.tab = tab;
  app.querySelectorAll('[role="tab"]').forEach(function (t) {
    var on = t.dataset.tab === tab;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    if (on && focus) t.focus();
  });
  drawPanel();
}

function drawDetail() {
  var pane = app.querySelector('.detail-pane');
  if (!pane) return;
  pane.textContent = '';
  var index = explorerState.selected;
  if (index === null || index === undefined) { pane.appendChild(el('div', 'Select an index.', 'notice')); return; }
  var head = el('div', null, 'toolbar detail-head');
  var row = listRow(index);
  var h = listColumn('health');
  if (row && h >= 0 && row[h]) head.appendChild(statusLabel(row[h]));
  var title = el('h2', cellText(index));
  title.id = 'detail-title';
  head.appendChild(title);
  pane.appendChild(head);
  var tabs = el('div', null, 'tabs');
  tabs.setAttribute('role', 'tablist');
  tabs.setAttribute('aria-labelledby', 'detail-title');
  TABS.forEach(function (t) {
    var b = button(t[1], function () { setTab(t[0]); }, 'tab');
    b.setAttribute('role', 'tab');
    b.id = 'tab-' + t[0];
    b.dataset.tab = t[0];
    b.setAttribute('aria-controls', 'tab-panel');
    b.setAttribute('aria-selected', String(t[0] === explorerState.tab));
    b.tabIndex = t[0] === explorerState.tab ? 0 : -1;
    tabs.appendChild(b);
  });
  tabs.addEventListener('keydown', function (ev) {
    var at = TABS.map(function (t) { return t[0]; }).indexOf(explorerState.tab);
    var next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: TABS.length - 1 }[ev.key];
    if (next === undefined) return;
    ev.preventDefault();
    setTab(TABS[(next + TABS.length) % TABS.length][0], true);
  });
  pane.appendChild(tabs);
  var panel = el('div', null, 'tab-panel');
  panel.id = 'tab-panel';
  panel.setAttribute('role', 'tabpanel');
  panel.tabIndex = 0;
  pane.appendChild(panel);
  drawPanel();
}

function statusLabel(health) {
  var s = el('span', health, 'status-badge ' + (['green', 'yellow', 'red'].indexOf(health) >= 0 ? health : 'unknown'));
  s.setAttribute('aria-label', 'health ' + health);
  return s;
}

function drawPanel() {
  var panel = app.querySelector('.tab-panel');
  if (!panel) return;
  var index = explorerState.selected, tab = explorerState.tab;
  panel.setAttribute('aria-labelledby', 'tab-' + tab);
  panel.textContent = '';
  var key = cacheKey(index, tab);
  if (!explorerState.cache[key] && canDrill()) load(index, tab);
  var entry = explorerState.cache[key];
  if (!entry) {
    if (tab === 'overview') renderOverviewTab(panel, index, null);
    var ask = el('div', 'This host does not let views call tools, so this cannot load here. Ask the assistant: "' + ASK[tab] + cellText(index) + '".', 'notice degraded-tab');
    panel.appendChild(ask);
    return;
  }
  if (entry.status === 'loading') {
    var busy = el('div', 'Loading ' + TABS.filter(function (t) { return t[0] === tab; })[0][1].toLowerCase() + ' for ' + cellText(index) + '\u2026', 'notice loading');
    busy.setAttribute('role', 'status');
    panel.appendChild(busy);
    return;
  }
  if (entry.status === 'error') {
    if (tab === 'overview') renderOverviewTab(panel, index, null);
    var err = el('div', null, 'notice error tab-error');
    err.setAttribute('role', 'alert');
    err.appendChild(errorBlock(entry.error));
    if (canDrill()) { err.appendChild(document.createTextNode(' ')); err.appendChild(retryButton(index, tab)); }
    panel.appendChild(err);
    return;
  }
  ({ overview: renderOverviewTab, mapping: renderMappingTab, shards: renderShardsTab, samples: renderSamplesTab })[tab](panel, index, entry.data);
}

function dl(pairs) {
  var card = el('div', null, 'card');
  pairs.forEach(function (p) {
    if (p[1] === null || p[1] === undefined || p[1] === '') return;
    var box = el('div', null, 'metric');
    box.appendChild(el('div', p[1], 'value'));
    box.appendChild(el('div', p[0], 'muted'));
    card.appendChild(box);
  });
  return card;
}
function settingValue(name, v) {
  if (name === 'creation_date' && /^\d{10,}$/.test(String(v))) return new Date(Number(v)).toISOString().replace('.000Z', 'Z');
  return cellText(v);
}

// Overview from the list row (health, docs, size) plus settings and aliases when loaded.
function renderOverviewTab(panel, index, detail) {
  var row = listRow(index);
  var col = function (n) { var i = listColumn(n); return row && i >= 0 ? row[i] : null; };
  var bytes = function (v) { return typeof v === 'number' ? formatBytes(v) : null; };
  var num = function (v) { return typeof v === 'number' ? formatNumber(v) : null; };
  panel.appendChild(dl([
    ['health', col('health')], ['status', col('status')], ['documents', num(col('docs.count'))], ['deleted documents', num(col('docs.deleted'))],
    ['store size', bytes(col('store.size'))], ['primary store size', bytes(col('pri.store.size'))], ['primary shards', num(col('pri'))], ['replicas', num(col('rep'))]
  ]));
  var info = detail && detail.indices && detail.indices.filter(function (x) { return x.name === index; })[0];
  if (!info && !detail && index === explorer.selected && explorer.detail && explorer.detail.status === 'ok') info = explorer.detail.index;
  if (!info) return;
  panel.appendChild(el('div', 'Aliases', 'label'));
  panel.appendChild(el('div', info.aliases && info.aliases.length ? info.aliases.map(function (a) { return cellText(a); }).join(', ') : 'None.', 'aliases'));
  if (info.data_stream) panel.appendChild(el('div', 'Backing index of data stream ' + cellText(info.data_stream) + '.', 'muted'));
  var settings = info.settings || {};
  panel.appendChild(el('div', 'Settings', 'label'));
  var t = el('table', null, 'settings key-settings');
  KEY_SETTINGS.forEach(function (k) {
    if (!(k in settings)) return;
    var tr = el('tr'); tr.appendChild(el('th', k)); tr.appendChild(el('td', settingValue(k, settings[k]))); t.appendChild(tr);
  });
  panel.appendChild(t);
  var names = Object.keys(settings);
  var all = el('details', null, 'all-settings');
  all.appendChild(el('summary', 'All settings (' + formatNumber(names.length) + ')'));
  var at = el('table', null, 'settings');
  names.forEach(function (k) { var tr = el('tr'); tr.appendChild(el('th', k)); tr.appendChild(el('td', cellText(settings[k]))); at.appendChild(tr); });
  all.appendChild(at);
  panel.appendChild(all);
}

function renderMappingTab(panel, index, data) {
  if (!explorerState.mappings[index]) explorerState.mappings[index] = newMappingState();
  panel.appendChild(mappingPanel(data, explorerState.mappings[index]));
}

function tableState(name) {
  if (!explorerState.tables[name]) explorerState.tables[name] = newTableState();
  return explorerState.tables[name];
}
function dataTab(panel, data, columns, rows, types, key, opts) {
  var trunc = truncationNotice(Object.assign({}, data, { columns: columns, rows: rows }));
  if (trunc) panel.appendChild(trunc);
  if (!rows.length) { panel.appendChild(el('div', 'No rows.', 'notice')); return; }
  var section = buildDataSection(data, columns, rows, types, tableState(key), opts);
  if (section.chips) panel.appendChild(section.chips);
  panel.appendChild(section.toolbar);
  panel.appendChild(section.stats);
  panel.appendChild(section.table.node);
}
function renderShardsTab(panel, index, data) {
  var m = data.meta || {};
  var types = columnTypes(data.columns, data.rows, m.column_types, m.integer_strings);
  var st = data.columns.indexOf('state');
  var unassigned = st < 0 ? 0 : data.rows.filter(function (r) { return r[st] === 'UNASSIGNED'; }).length;
  panel.appendChild(el('div', plural(data.rows.length, 'shard copy', 'shard copies') + (unassigned ? ', ' + formatNumber(unassigned) + ' unassigned' : '') + '.', unassigned ? 'notice warning unassigned' : 'muted'));
  dataTab(panel, data, data.columns, data.rows, types, 'shards:' + index, { chipSkipNumeric: true });
}
function renderSamplesTab(panel, index, data) {
  var m = data.meta || {};
  var total = typeof m.total === 'number' ? ' of ' + formatNumber(m.total) + (m.total_relation === 'gte' ? '+' : '') : '';
  panel.appendChild(el('div', 'First ' + plural(data.rows.length, 'document') + total + ' (match_all, unsorted).', 'muted'));
  var flat = flattenColumns(data.columns, data.rows, m.column_types, VIEW_MAX_COLUMNS);
  var columns = flat ? flat.columns : data.columns, rows = flat ? flat.rows : data.rows;
  var types = columnTypes(columns, rows, flat ? flat.types : m.column_types, m.integer_strings);
  dataTab(panel, data, columns, rows, types, 'samples:' + index, { chipExclude: ['_id', '_score'] });
}

function renderExplorer(data) {
  explorer = data;
  explorerState.cache = dict();
  if (data.selected && data.detail) {
    explorerState.cache[cacheKey(data.selected, 'overview')] = data.detail.status === 'ok'
      ? { status: 'ok', data: { kind: 'index_detail', indices: [data.detail.index] } }
      : { status: 'error', error: data.detail.error || 'The index details failed.' };
  }
  var names = listed();
  // Keep the user's selection across a refresh while the index still exists.
  if (explorerState.selected === null || names.indexOf(explorerState.selected) < 0) explorerState.selected = data.selected === undefined ? null : data.selected;
  var stats = [];
  if (data.list && data.list.status === 'ok' && data.list.summary) {
    var sm = data.list.summary;
    stats.push(plural(sm.docs || 0, 'document'), formatBytes(sm.store_bytes || 0));
  }
  app.appendChild(header(data, 'Index explorer' + (data.pattern ? ': ' + data.pattern : ''), stats));
  if (!canDrill()) {
    var hint = names.filter(function (n) { return n !== explorerState.selected; })[0];
    app.appendChild(el('div', 'This host does not let views call tools, so other indices and tabs cannot load here. Ask the assistant instead, for example "Show the mapping of ' +
      (explorerState.selected ? cellText(explorerState.selected) : 'an index') + '"' + (hint ? ' or "Open the index explorer for ' + cellText(hint) + '"' : '') + '.', 'notice degraded'));
  }
  var layout = el('div', null, 'explorer');
  var left = el('div', null, 'list-pane');
  var search = document.createElement('input');
  search.type = 'search'; search.className = 'index-search'; search.placeholder = 'Filter indices';
  search.value = explorerState.query;
  search.setAttribute('aria-label', 'Filter indices by name');
  search.addEventListener('input', function () { explorerState.query = search.value; drawList(); });
  left.appendChild(search);
  var count = el('div', null, 'muted list-count');
  count.setAttribute('aria-live', 'polite');
  left.appendChild(count);
  var list = el('ul', null, 'index-list');
  list.setAttribute('aria-label', 'Indices');
  list.addEventListener('keydown', listKeys);
  left.appendChild(list);
  layout.appendChild(left);
  var right = el('div', null, 'detail-pane');
  layout.appendChild(right);
  app.appendChild(layout);
  drawList();
  drawDetail();
}

startView({
  name: 'opensearch-index-explorer',
  unsupportedText: 'This response has no index explorer; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'index_explorer' && !!data.list && typeof data.list === 'object'; },
  render: renderExplorer
});
