// Typed, sortable table with column picker, JSON rows, row selection, filter chips, field
// statistics and CSV/JSON export. State is keyed by column name so it survives a refresh.
var CHIP_MAX_VALUES = 8;
var CHIP_MAX_COLUMNS = 3;
var STATS_TOP_VALUES = 5;
var NUMERIC_TYPES = ['byte', 'short', 'integer', 'int', 'long', 'bigint', 'float', 'double', 'half_float',
  'scaled_float', 'decimal', 'unsigned_long', 'number', 'bytes'];
var STATUS_COLUMNS = ['health', 'status', 'state'];

function isNumericType(t) { return NUMERIC_TYPES.indexOf(String(t || '').toLowerCase()) >= 0; }

// Server types win; otherwise a column whose values are all numbers is numeric.
function columnTypes(columns, rows, declared) {
  return columns.map(function (_, i) {
    var t = declared && declared[i];
    if (t) return String(t).toLowerCase();
    var seen = false;
    for (var r = 0; r < rows.length; r++) {
      var v = rows[r][i];
      if (v === null || v === undefined) continue;
      if (typeof v !== 'number') return 'string';
      seen = true;
    }
    return seen ? 'number' : 'string';
  });
}

function formatCell(v, type) {
  if (type === 'bytes') return formatBytes(v);
  if (typeof v === 'number' && isNumericType(type)) return formatNumber(v);
  return cellText(v);
}

// Flattens object-valued columns one level into dotted columns (user -> user.id, user.geo);
// arrays and deeper objects stay as JSON. Null when nothing flattens, a column mixes objects
// with other values, names collide, or the result would exceed `maxColumns`.
function flattenColumns(columns, rows, types, maxColumns) {
  var plans = columns.map(function (_, i) {
    var keys = [], seen = {}, objects = 0;
    for (var r = 0; r < rows.length; r++) {
      var v = rows[r][i];
      if (v === null || v === undefined) continue;
      if (typeof v !== 'object' || Array.isArray(v)) return null;
      objects++;
      Object.keys(v).forEach(function (k) { if (!seen[k]) { seen[k] = true; keys.push(k); } });
    }
    return objects && keys.length ? keys : null;
  });
  if (!plans.some(Boolean)) return null;
  var names = [], sources = [];
  columns.forEach(function (c, i) {
    if (plans[i]) plans[i].forEach(function (k) { names.push(c + '.' + k); sources.push({ index: i, key: k }); });
    else { names.push(c); sources.push({ index: i }); }
  });
  if (names.length > maxColumns || new Set(names).size !== names.length) return null;
  return {
    columns: names,
    rows: rows.map(function (r) {
      return sources.map(function (s) {
        var v = r[s.index];
        if (s.key === undefined) return v;
        return v && Object.prototype.hasOwnProperty.call(v, s.key) ? v[s.key] : null;
      });
    }),
    types: sources.map(function (s) { return s.key === undefined && types ? types[s.index] : undefined; })
  };
}

function newTableState() { return { sort: null, hidden: {}, filters: {}, stats: '' }; }

function buildTable(columns, rows, types, state, opts) {
  opts = opts || {};
  var selected = new Set();
  var wrap = el('div');
  if (opts.picker && columns.length > 4) {
    var picker = el('details', null, 'columns');
    picker.appendChild(el('summary', 'Columns'));
    columns.forEach(function (c) {
      var label = el('label');
      var box = document.createElement('input');
      box.type = 'checkbox'; box.checked = !state.hidden[c];
      box.addEventListener('change', function () { if (box.checked) delete state.hidden[c]; else state.hidden[c] = true; draw(); });
      label.appendChild(box); label.appendChild(document.createTextNode(' ' + c));
      picker.appendChild(label);
    });
    wrap.appendChild(picker);
  }
  var scroll = el('div', null, 'scroll');
  var table = el('table');
  scroll.appendChild(table);
  wrap.appendChild(scroll);

  function visible() { return columns.map(function (_, i) { return i; }).filter(function (i) { return !state.hidden[columns[i]]; }); }
  function order() {
    var idx = rows.map(function (_, i) { return i; });
    var col = state.sort ? columns.indexOf(state.sort.name) : -1;
    if (col >= 0) idx.sort(function (x, y) { return state.sort.dir * compare(rows[x][col], rows[y][col]) || x - y; });
    return idx;
  }
  function draw() {
    table.textContent = '';
    var vis = visible();
    var thead = el('thead'); var hr = el('tr');
    if (opts.selectable) {
      var th0 = el('th', null, 'select');
      var all = document.createElement('input');
      all.type = 'checkbox'; all.title = 'Select all shown rows';
      all.checked = rows.length > 0 && rows.every(function (r) { return selected.has(r); });
      all.addEventListener('change', function () { rows.forEach(function (r) { if (all.checked) selected.add(r); else selected.delete(r); }); draw(); });
      th0.appendChild(all); hr.appendChild(th0);
    }
    vis.forEach(function (i) {
      var th = el('th', columns[i], isNumericType(types[i]) ? 'num' : '');
      if (state.sort && state.sort.name === columns[i]) th.setAttribute('aria-sort', state.sort.dir > 0 ? 'ascending' : 'descending');
      th.addEventListener('click', function () {
        var dir = state.sort && state.sort.name === columns[i] ? -state.sort.dir : 1;
        state.sort = { name: columns[i], dir: dir }; draw();
      });
      hr.appendChild(th);
    });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = el('tbody');
    order().forEach(function (r) {
      var row = rows[r];
      var tr = el('tr', null, 'row');
      if (opts.selectable) {
        var td0 = el('td', null, 'select');
        var box = document.createElement('input');
        box.type = 'checkbox'; box.checked = selected.has(row);
        box.addEventListener('click', function (ev) { ev.stopPropagation(); });
        box.addEventListener('change', function () { if (box.checked) selected.add(row); else selected.delete(row); });
        td0.appendChild(box); tr.appendChild(td0);
      }
      vis.forEach(function (i) {
        var v = row[i];
        var cls = [];
        if (v === null || v === undefined) cls.push('null');
        if (isNumericType(types[i])) cls.push('num');
        if (STATUS_COLUMNS.indexOf(columns[i]) >= 0 && typeof v === 'string' && /^[a-z_]+$/i.test(v)) cls.push('status-' + v.toLowerCase());
        var td = el('td', null, cls.join(' '));
        td.appendChild(el('span', formatCell(v, types[i]), 'cell'));
        if (v !== null && v !== undefined) td.title = types[i] === 'bytes' && typeof v === 'number' ? plural(v, 'byte') : rawText(v);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
      if (opts.detail) {
        tr.addEventListener('click', function () {
          var next = tr.nextSibling;
          if (next && next.className === 'detail') { next.remove(); return; }
          var obj = {};
          columns.forEach(function (c, i) { if (row[i] !== undefined) obj[c] = row[i]; });
          var dtr = el('tr', null, 'detail'); var td = el('td');
          td.colSpan = vis.length + (opts.selectable ? 1 : 0);
          td.appendChild(el('pre', JSON.stringify(obj, null, 2)));
          dtr.appendChild(td);
          tr.parentNode.insertBefore(dtr, tr.nextSibling);
        });
      }
    });
    table.appendChild(tbody);
  }
  draw();
  return {
    node: wrap,
    setRows: function (next) { rows = next; selected.forEach(function (r) { if (next.indexOf(r) < 0) selected.delete(r); }); draw(); },
    shownColumns: function () { var v = visible(); return { indexes: v, names: v.map(function (i) { return columns[i]; }) }; },
    shownRows: function () { return order().map(function (i) { return rows[i]; }); },
    selectedRows: function () { return order().map(function (i) { return rows[i]; }).filter(function (r) { return selected.has(r); }); }
  };
}

function chipColumns(columns, rows, exclude) {
  var out = [];
  if (rows.length < 3) return out;
  columns.forEach(function (c, i) {
    if (out.length >= CHIP_MAX_COLUMNS || (exclude || []).indexOf(c) >= 0) return;
    var counts = {}, order = [], labels = {};
    for (var r = 0; r < rows.length; r++) {
      var v = rows[r][i];
      if (v !== null && typeof v === 'object') return;
      var k = valueKey(v);
      if (!(k in counts)) {
        counts[k] = 0; order.push(k); labels[k] = v === null || v === undefined ? '(null)' : cellText(v);
        if (order.length > CHIP_MAX_VALUES) return;
      }
      counts[k]++;
    }
    if (order.length >= 2 && order.length < rows.length) out.push({ index: i, name: c, keys: order, labels: labels, counts: counts });
  });
  return out;
}

function buildChips(groups, filters, onChange) {
  var wrap = el('div');
  var clear = button('Clear filters', function () { Object.keys(filters).forEach(function (k) { delete filters[k]; }); sync(); });
  function sync() {
    wrap.querySelectorAll('button.chip').forEach(function (b) {
      b.setAttribute('aria-pressed', filters[b.dataset.col] === b.dataset.key ? 'true' : 'false');
    });
    clear.hidden = Object.keys(filters).length === 0;
    onChange();
  }
  groups.forEach(function (g) {
    var line = el('div', null, 'chips');
    line.appendChild(el('span', g.name, 'name'));
    g.keys.forEach(function (k) {
      var chip = button(g.labels[k] + ' (' + g.counts[k] + ')', function () {
        if (filters[g.name] === k) delete filters[g.name]; else filters[g.name] = k;
        sync();
      }, 'chip');
      chip.dataset.col = g.name; chip.dataset.key = k;
      line.appendChild(chip);
    });
    wrap.appendChild(line);
  });
  wrap.appendChild(clear);
  sync();
  return wrap;
}

function fieldStats(rows, index, type) {
  var values = rows.map(function (r) { return r[index]; });
  var present = values.filter(function (v) { return v !== null && v !== undefined; });
  var counts = {}, labels = {}, order = [];
  present.forEach(function (v) {
    var k = valueKey(v);
    if (!(k in counts)) { counts[k] = 0; labels[k] = cellText(v); order.push(k); }
    counts[k]++;
  });
  order.sort(function (a, b) { return counts[b] - counts[a]; });
  var stats = { count: values.length, missing: values.length - present.length, distinct: order.length,
    top: order.slice(0, STATS_TOP_VALUES).map(function (k) { return { label: labels[k], count: counts[k] }; }) };
  var nums = present.filter(function (v) { return typeof v === 'number' && isFinite(v); });
  if (nums.length && (isNumericType(type) || nums.length === present.length)) {
    var sum = nums.reduce(function (a, b) { return a + b; }, 0);
    stats.min = Math.min.apply(null, nums); stats.max = Math.max.apply(null, nums); stats.avg = sum / nums.length;
  }
  return stats;
}

// getRows(column) returns { rows, filters, ignored }: the rows to describe, labels of the
// filters applied, and the column's own filter, which is left out so its stats stay useful.
function statsPanel(columns, types, state, getRows) {
  var wrap = el('span');
  var select = el('select', null, 'stats-column');
  select.appendChild(el('option', 'Column stats\u2026'));
  select.options[0].value = '';
  columns.forEach(function (c) { var o = el('option', c); o.value = c; select.appendChild(o); });
  var panel = el('div', null, 'panel stats');
  panel.hidden = true;
  function draw() {
    var i = columns.indexOf(state.stats);
    select.value = i >= 0 ? state.stats : '';
    panel.textContent = '';
    panel.hidden = i < 0;
    if (i < 0) return;
    var info = getRows(columns[i]);
    var s = fieldStats(info.rows, i, types[i]);
    var fmt = function (v) { return types[i] === 'bytes' ? formatBytes(v) : formatNumber(v); };
    var scope = columns[i] + ' over ' + plural(s.count, 'row');
    if (info.filters.length) scope += ' matching ' + info.filters.join(' and ');
    if (info.ignored) scope += '; its own filter (' + info.ignored + ') is not applied';
    panel.appendChild(el('div', scope + ' (computed in the view, not by OpenSearch)', 'label stats-scope'));
    var t = el('table');
    var add = function (k, v) { var tr = el('tr'); tr.appendChild(el('th', k)); tr.appendChild(el('td', v, 'num')); t.appendChild(tr); };
    add('missing', formatNumber(s.missing)); add('distinct', formatNumber(s.distinct));
    if (s.min !== undefined) { add('min', fmt(s.min)); add('max', fmt(s.max)); add('avg', fmt(s.avg)); }
    panel.appendChild(t);
    if (s.top.length) {
      panel.appendChild(el('div', 'Top values', 'label'));
      var top = el('table', null, 'top-values');
      s.top.forEach(function (x) {
        var tr = el('tr');
        tr.appendChild(el('td', x.label));
        tr.appendChild(el('td', formatNumber(x.count), 'num'));
        var bar = el('td'); var m = el('span', null, 'meter');
        m.style.width = Math.max(2, Math.round(120 * x.count / Math.max(1, s.count))) + 'px';
        bar.appendChild(m); tr.appendChild(bar);
        top.appendChild(tr);
      });
      panel.appendChild(top);
    }
  }
  select.addEventListener('change', function () { state.stats = select.value; draw(); });
  wrap.appendChild(select);
  return { control: wrap, panel: panel, refresh: draw };
}

// CSV cells that start with a formula character are prefixed with ' so spreadsheets
// show index data as text instead of evaluating it.
function csvCell(v) {
  if (v === null || v === undefined) return '';
  var s = rawText(v);
  if (typeof v === 'string' && /^[=+\-@\t\r]/.test(s)) s = "'" + s;
  return /[",\r\n]/.test(s) || s !== s.trim() ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function toCsv(names, indexes, rows) {
  var lines = [names.map(csvCell).join(',')];
  rows.forEach(function (r) { lines.push(indexes.map(function (i) { return csvCell(r[i]); }).join(',')); });
  return lines.join('\r\n') + '\r\n';
}
function toJsonRows(names, indexes, rows) {
  return JSON.stringify(rows.map(function (r) {
    var o = {}; indexes.forEach(function (i, n) { o[names[n]] = r[i]; }); return o;
  }), null, 2);
}

function csvFileName(data) {
  return 'opensearch-' + String(data.tool || 'results').replace(/[^A-Za-z0-9_-]/g, '_') + '.csv';
}

function exportControls(table, data) {
  var wrap = el('span');
  var status = el('span', null, 'muted export-status');
  function rowsOrSelected() { var s = table.selectedRows(); return s.length ? s : table.shownRows(); }
  wrap.appendChild(button('Copy CSV', function () {
    var cols = table.shownColumns(); var rows = rowsOrSelected();
    copyText(toCsv(cols.names, cols.indexes, rows), 'CSV of ' + plural(rows.length, 'row'), status);
  }, 'copy-csv'));
  wrap.appendChild(document.createTextNode(' '));
  wrap.appendChild(button('Copy JSON', function () {
    var cols = table.shownColumns(); var rows = rowsOrSelected();
    copyText(toJsonRows(cols.names, cols.indexes, rows), 'JSON of ' + plural(rows.length, 'row'), status);
  }, 'copy-json'));
  wrap.appendChild(document.createTextNode(' '));
  if (hostCapabilities.downloadFile) {
    // Sandboxed views cannot download directly; the host saves the file, usually after a prompt.
    wrap.appendChild(button('Download CSV', function () {
      var cols = table.shownColumns(); var rows = rowsOrSelected();
      var resource = { uri: 'file:///' + csvFileName(data), mimeType: 'text/csv', text: toCsv(cols.names, cols.indexes, rows) };
      request('ui/download-file', { contents: [{ type: 'resource', resource: resource }] }).then(function (r) {
        status.textContent = r && r.isError ? 'Host did not download the file.' : 'Sent CSV of ' + plural(rows.length, 'row') + ' to the host to download.';
      }, function () { status.textContent = 'Host could not download the file.'; });
    }, 'download-csv'));
    wrap.appendChild(document.createTextNode(' '));
  }
  wrap.appendChild(status);
  return wrap;
}

// Header stat for the row count: "rows: 6", or "rows: 3 of 6 shown" while filters apply.
function rowCountStat(total) {
  var node = el('span', null, 'stat rows-stat');
  var update = function (shown) {
    node.textContent = 'rows: ' + (shown === total ? formatNumber(total) : formatNumber(shown) + ' of ' + formatNumber(total) + ' shown');
  };
  update(total);
  return { node: node, update: update };
}

// Chips, toolbar (export, stats, model context), and the table, wired to one filter state.
function buildDataSection(data, columns, rows, types, state, opts) {
  opts = opts || {};
  var section = { node: el('div') };
  // Drop filters for columns or values a refresh removed.
  var groups = chipColumns(columns, rows, (opts.chipExclude || []).concat(opts.chipSkipNumeric
    ? columns.filter(function (_, i) { return isNumericType(types[i]); }) : []));
  Object.keys(state.filters).forEach(function (name) {
    var g = groups.filter(function (x) { return x.name === name; })[0];
    if (!g || g.keys.indexOf(state.filters[name]) < 0) delete state.filters[name];
  });
  // Rows passing every filter except the one on `skip`.
  function filtered(skip) {
    var keys = Object.keys(state.filters).filter(function (k) { return k !== skip; });
    if (!keys.length) return rows;
    var idx = keys.map(function (k) { return columns.indexOf(k); });
    return rows.filter(function (r) { return keys.every(function (k, n) { return valueKey(r[idx[n]]) === state.filters[k]; }); });
  }
  function filterLabel(name) {
    var g = groups.filter(function (x) { return x.name === name; })[0];
    return name + ' = ' + (g ? g.labels[state.filters[name]] : '?');
  }
  var table = buildTable(columns, filtered(), types, state, { picker: true, detail: true, selectable: true });
  var stats = statsPanel(columns, types, state, function (name) {
    return {
      rows: filtered(name),
      filters: Object.keys(state.filters).filter(function (k) { return k !== name; }).map(filterLabel),
      ignored: name in state.filters ? filterLabel(name) : null
    };
  });
  section.table = table;
  section.filtered = function () { return filtered(); };
  function apply() {
    var current = filtered();
    table.setRows(current);
    stats.refresh();
    if (opts.rowsStat) opts.rowsStat.update(current.length);
    if (opts.onRows) opts.onRows(current);
  }
  section.chips = groups.length ? buildChips(groups, state.filters, function () { if (section.ready) apply(); }) : null;
  var bar = el('div', null, 'toolbar data-tools');
  bar.appendChild(exportControls(table, data));
  bar.appendChild(stats.control);
  var ctx = contextControls(data, function () { return table.shownColumns(); }, function () { return table.selectedRows(); });
  if (ctx) bar.appendChild(ctx);
  section.toolbar = bar;
  section.stats = stats.panel;
  section.apply = apply;
  section.ready = true;
  stats.refresh();
  if (opts.rowsStat) opts.rowsStat.update(filtered().length);
  return section;
}
