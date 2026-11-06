// Inline-SVG charts: bars for one category and one number (PPL `stats ... by field`), and a
// line/area time series for a time column plus numbers, optionally split by one category.
// Both size to their container and redraw on resize; the rest of the view is untouched.
var CHART_MAX_BARS = 30;
var SERIES_MAX = 8;
var FILL_MAX_BUCKETS = 2000;
var CHART_DEFAULT_WIDTH = 720;
var CHAR_PX = 7.3;  // 12px monospace advance, for label layout
var TIME_TYPES = ['timestamp', 'date', 'datetime', 'time', 'date_nanos'];
var FIXED_UNITS = dict({ ms: 1, s: 1000, m: 60000, h: 3600000, d: 86400000, w: 604800000 });
var UNIT_ALIASES = dict({
  millisecond: 'ms', milliseconds: 'ms', second: 's', seconds: 's', sec: 's', minute: 'm', minutes: 'm',
  min: 'm', hour: 'h', hours: 'h', day: 'd', days: 'd', week: 'w', weeks: 'w'
});
var TIME_STEPS = [1e3, 5e3, 15e3, 3e4, 6e4, 3e5, 9e5, 18e5, 36e5, 72e5, 108e5, 216e5, 432e5,
  864e5, 1728e5, 6048e5, 12096e5, 2592e6, 7776e6, 31536e6];

function parseTime(v) {
  if (typeof v !== 'string') return NaN;
  var s = v.trim().replace(' ', 'T');
  if (!/[zZ]|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';  // PPL returns UTC without a zone
  return Date.parse(s);
}
function timeLabel(ms) { return new Date(ms).toISOString().slice(0, 16).replace('T', ' '); }

// Bucket width in ms from the query's span(field, N unit), or null for calendar units
// (M, q, y), numeric spans and queries without a span.
function spanInterval(query) {
  var m = /\bspan\s*\(\s*[^,()]+,\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)?\s*\)/i.exec(query || '');
  if (!m || !m[2]) return null;
  var unit = m[2];
  var key = unit.length <= 2 && unit !== 'M' ? unit.toLowerCase() : UNIT_ALIASES[unit.toLowerCase()];
  return key && FIXED_UNITS[key] ? Number(m[1]) * FIXED_UNITS[key] : null;
}

// { ms, regular }: the span() bucket width and whether every timestamp sits on that grid.
// Null without a fixed-unit span (raw rows are not buckets) or with fewer than two timestamps.
function inferInterval(ts, query) {
  var ms = spanInterval(query);
  if (ts.length < 2 || !ms || !isFinite(ms) || ms <= 0) return null;
  var regular = ts.every(function (t) { var k = (t - ts[0]) / ms; return Math.abs(k - Math.round(k)) < 1e-6; });
  return { ms: ms, regular: regular };
}

// Column names whose empty buckets mean 0: count(), dc(), distinct_count(), and their `as` aliases.
function countLikeColumns(columns, query) {
  var aliases = dict();
  var re = /\b(?:count|dc|distinct_count)\s*\([^)]*\)\s+as\s+`?([\w.@]+)`?/gi, m;
  while ((m = re.exec(query || ''))) aliases[m[1]] = true;
  return columns.map(function (c) { return /^(count|dc|distinct_count)\s*\(/i.test(String(c)) || c === 'count' || !!aliases[c]; });
}

// Fills missing buckets with 0 when the series is count-like and the grid is regular; returns
// the shared timestamps. Otherwise the timestamps stay as given and lines break at gaps.
function fillBuckets(series, ts, interval, countLike) {
  if (!interval || !interval.regular || !series.some(function (s) { return countLike(s); })) return ts;
  var n = Math.round((ts[ts.length - 1] - ts[0]) / interval.ms) + 1;
  if (n > FILL_MAX_BUCKETS) return ts;
  var grid = [];
  for (var i = 0; i < n; i++) grid.push(ts[0] + i * interval.ms);
  series.forEach(function (s) {
    if (!countLike(s)) return;
    grid.forEach(function (t) { if (!(t in s.points)) { s.points[t] = 0; s.filled = (s.filled || 0) + 1; } });
  });
  return grid;
}

// Runs of consecutive present points, split where the spacing exceeds 1.5x the interval.
function segments(points, ts, interval) {
  var out = [], current = [];
  var limit = interval ? interval.ms * 1.5 : Infinity;
  var last = null;
  ts.forEach(function (t) {
    if (!(t in points)) return;
    if (last !== null && t - last > limit) { out.push(current); current = []; }
    current.push(t); last = t;
  });
  if (current.length) out.push(current);
  return out;
}

// About `count` evenly spaced ticks with 1/2/2.5/5 x 10^k steps covering [min, max]; min
// defaults to 0 and only a negative min extends the axis below 0.
function niceTicks(max, count, integer, min) {
  if (!(max > 0) || !isFinite(max)) max = min < 0 ? 0 : 1;
  min = min < 0 && isFinite(min) ? min : 0;
  var raw = (max - min) / (count || 4);
  var mag = Math.pow(10, Math.floor(Math.log10(raw)));
  var norm = raw / mag;
  var mult = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 && !integer ? 2.5 : norm <= 5 ? 5 : 10;
  var step = mult * mag;
  if (integer && step < 1) step = 1;
  // Subnormal ranges underflow the step to 0 or overflow it; fall back to the range itself.
  if (!(step > 0) || !isFinite(step) || (max - min) / step > 1000) return min < 0 ? [min, 0, max] : [0, max];
  var first = Math.floor(min / step + 1e-9);
  var ticks = [];
  for (var i = first; i * step < max + step * 1e-9 || ticks.length < 2; i++) ticks.push(Number((i * step).toPrecision(12)) || 0);
  if (ticks[ticks.length - 1] < max) {
    var end = Number(((first + ticks.length) * step).toPrecision(12));
    ticks.push(isFinite(end) ? end : max);
  }
  return ticks;
}

// UTC-aligned time ticks between t0 and t1, at most maxTicks of them.
function timeTicks(t0, t1, maxTicks) {
  for (var i = 0; i < TIME_STEPS.length; i++) {
    var step = TIME_STEPS[i];
    var first = Math.ceil(t0 / step) * step;
    var n = first > t1 ? 0 : Math.floor((t1 - first) / step) + 1;
    if (n <= maxTicks && n >= 1) {
      var out = [];
      for (var t = first; t <= t1; t += step) out.push(t);
      return { ticks: out, step: step };
    }
  }
  return { ticks: [t0, t1], step: t1 - t0 };
}
function sameUtcDay(a, b) { return new Date(a).toISOString().slice(0, 10) === new Date(b).toISOString().slice(0, 10); }
function tickLabel(t, step, oneDay) {
  var iso = new Date(t).toISOString();
  if (step >= 864e5) return iso.slice(0, 10);
  if (oneDay) return step < 6e4 ? iso.slice(11, 19) : iso.slice(11, 16);
  return iso.slice(5, 16).replace('T', ' ');
}

function chartWidth(node) {
  var w = node.clientWidth;
  return w > 0 ? Math.floor(w) : CHART_DEFAULT_WIDTH;
}
// Redraws when the container width changes; hidden containers (width 0) are skipped. Stops
// observing once the node leaves the document (a re-render replaced it).
function onWidthChange(node, redraw) {
  if (typeof ResizeObserver !== 'function') return;
  var last = 0;
  var observer = new ResizeObserver(function () {
    if (!node.isConnected) { observer.disconnect(); return; }
    var w = Math.floor(node.clientWidth);
    if (w > 0 && w !== last) { last = w; redraw(); }
  });
  observer.observe(node);
}

function barSpec(data, types) {
  if (data.kind !== 'ppl_rows' || data.columns.length !== 2 || data.rows.length === 0) return null;
  var numeric = [0, 1].map(function (i) {
    return isNumericType(types[i]) && data.rows.every(function (r) { return r[i] === null || typeof r[i] === 'number'; });
  });
  if (numeric[0] === numeric[1]) return null;
  return numeric[0] ? { value: 0, label: 1 } : { value: 1, label: 0 };
}

function timeSeriesSpec(data, types) {
  if (data.kind !== 'ppl_rows' || data.rows.length < 2) return null;
  var time = -1, values = [], category = -1, others = 0;
  data.columns.forEach(function (_, i) {
    var t = types[i];
    if (time < 0 && TIME_TYPES.indexOf(t) >= 0 && data.rows.every(function (r) { return r[i] === null || !isNaN(parseTime(r[i])); })) time = i;
    else if (isNumericType(t) && data.rows.every(function (r) { return r[i] === null || typeof r[i] === 'number'; })) values.push(i);
    else { others++; if (category < 0) category = i; }
  });
  if (time < 0 || !values.length || others > 1) return null;
  if (category >= 0) {
    var distinct = {};
    data.rows.forEach(function (r) { distinct[valueKey(r[category])] = 1; });
    if (values.length > 1 || Object.keys(distinct).length > SERIES_MAX) return null;
  }
  return { time: time, values: values.slice(0, SERIES_MAX), category: category };
}

function buildBarChart(data, spec) {
  var wrap = el('div', null, 'chart-wrap bar-chart');
  var holder = el('div');
  var note = el('div', null, 'muted');
  wrap.appendChild(holder); wrap.appendChild(note);
  var current = [];
  function render() {
    holder.textContent = '';
    var shown = current.slice(0, CHART_MAX_BARS);
    note.textContent = current.length > shown.length ? 'Showing the first ' + shown.length + ' of ' + current.length + ' bars; switch to the table for all rows.' : '';
    var max = 0;
    shown.forEach(function (r) { if (typeof r[spec.value] === 'number' && r[spec.value] > max) max = r[spec.value]; });
    var width = chartWidth(wrap);
    var rowH = 22, labelW = width < 480 ? 120 : 180, valueW = 90, barW = Math.max(40, width - labelW - valueW);
    var chart = svg('svg', { 'class': 'chart', width: '100%', viewBox: '0 0 ' + width + ' ' + Math.max(rowH, shown.length * rowH), role: 'img' });
    chart.appendChild(svg('title', {}, data.columns[spec.value] + ' by ' + data.columns[spec.label]));
    var labelChars = Math.floor((labelW - 10) / CHAR_PX);
    shown.forEach(function (r, i) {
      var v = typeof r[spec.value] === 'number' ? r[spec.value] : 0;
      var y = i * rowH;
      var label = r[spec.label] === null || r[spec.label] === undefined ? '(null)' : cellText(r[spec.label]);
      chart.appendChild(svg('text', { x: labelW - 6, y: y + 15, 'text-anchor': 'end' }, clip(label, labelChars)));
      var w = max > 0 && v > 0 ? Math.max(1, (v / max) * barW) : 0;
      chart.appendChild(svg('rect', { 'class': 'bar', x: labelW, y: y + 3, width: w, height: rowH - 6, rx: 2 }));
      chart.appendChild(svg('text', { x: labelW + w + 6, y: y + 15 }, formatNumber(r[spec.value])));
    });
    holder.appendChild(chart);
  }
  onWidthChange(wrap, render);
  return { node: wrap, setRows: function (rows) { current = rows; render(); } };
}

function buildTimeSeries(data, spec) {
  var H = 280, R = 16, T = 12, B = 54;
  var query = (data.meta && data.meta.query) || '';
  var countLike = countLikeColumns(data.columns, query);
  var wrap = el('div', null, 'chart-wrap time-series');
  var legend = el('div', null, 'legend');
  var holder = el('div');
  var note = el('div', null, 'muted chart-note');
  var tip = el('div', null, 'tooltip');
  tip.hidden = true;
  wrap.appendChild(legend); wrap.appendChild(holder); wrap.appendChild(note); wrap.appendChild(tip);
  var valueName = spec.values.map(function (i) { return data.columns[i]; }).join(', ');

  function seriesOf(rows) {
    var series = [], byKey = dict();
    function get(key, name, column) {
      if (!(key in byKey)) { byKey[key] = series.length; series.push({ name: name, column: column, points: {} }); }
      return series[byKey[key]];
    }
    rows.forEach(function (r) {
      var t = parseTime(r[spec.time]);
      if (isNaN(t)) return;
      if (spec.category >= 0) {
        var v = r[spec.values[0]];
        var c = r[spec.category];
        if (typeof v === 'number') get(valueKey(c), c === null || c === undefined ? '(null)' : cellText(c), spec.values[0]).points[t] = v;
      } else {
        spec.values.forEach(function (i) { if (typeof r[i] === 'number') get('col:' + i, data.columns[i], i).points[t] = r[i]; });
      }
    });
    return series.slice(0, SERIES_MAX);
  }

  var model = null;
  function prepare(rows) {
    var series = seriesOf(rows);
    var times = dict();
    series.forEach(function (s) { Object.keys(s.points).forEach(function (t) { times[t] = 1; }); });
    var ts = Object.keys(times).map(Number).sort(function (a, b) { return a - b; });
    var interval = inferInterval(ts, query);
    var isCount = function (s) { return countLike[s.column]; };
    ts = ts.length >= 2 ? fillBuckets(series, ts, interval, isCount) : ts;
    var filled = series.reduce(function (n, s) { return n + (s.filled || 0); }, 0);
    var gaps = !filled && series.some(function (s) { return segments(s.points, ts, interval).length > 1; });
    var integer = series.every(function (s) { return Object.keys(s.points).every(function (t) { return Number.isInteger(s.points[t]); }); });
    model = { series: series, ts: ts, interval: interval, filled: filled, gaps: gaps, integer: integer };
  }

  function render() {
    holder.textContent = ''; legend.textContent = ''; tip.hidden = true; note.textContent = '';
    var series = model.series, ts = model.ts, interval = model.interval;
    if (ts.length < 2) { holder.appendChild(el('div', 'Not enough time buckets to chart; see the table.', 'muted')); return; }
    if (model.filled) note.textContent = plural(model.filled, 'empty bucket') + ' shown as 0 (PPL omits empty buckets).';
    else if (model.gaps) note.textContent = 'Gaps mark missing buckets.';
    var max = 0, min = 0;
    series.forEach(function (s) { Object.keys(s.points).forEach(function (t) { var v = s.points[t]; if (v > max) max = v; if (v < min) min = v; }); });
    var yTicks = niceTicks(max, 4, model.integer, min);
    var yMin = yTicks[0], yMax = yTicks[yTicks.length - 1];
    var yLabels = yTicks.map(formatNumber);
    var L = 30 + Math.ceil(Math.max.apply(null, yLabels.map(function (s) { return s.length; })) * CHAR_PX);
    var W = chartWidth(wrap);
    var plotW = Math.max(60, W - L - R);
    var t0 = ts[0], t1 = ts[ts.length - 1];
    var oneDay = sameUtcDay(t0, t1);
    var labelChars = oneDay ? 5 : 11;
    var xt = timeTicks(t0, t1, Math.max(2, Math.floor(plotW / (labelChars * CHAR_PX + 18))));
    if (oneDay && xt.step < 6e4) labelChars = 8;
    var x = function (t) { return L + (t - t0) / (t1 - t0) * plotW; };
    var y = function (v) { return T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B); };
    var chart = svg('svg', { 'class': 'chart', width: '100%', viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
    chart.appendChild(svg('title', {}, series.map(function (s) { return s.name; }).join(', ') + ' over ' + data.columns[spec.time]));
    yTicks.forEach(function (v, i) {
      chart.appendChild(svg('line', { 'class': v === 0 ? 'axis' : 'grid', x1: L, y1: y(v), x2: L + plotW, y2: y(v) }));
      chart.appendChild(svg('text', { 'class': 'y-tick', x: L - 6, y: y(v) + 4, 'text-anchor': 'end' }, yLabels[i]));
    });
    chart.appendChild(svg('line', { 'class': 'axis', x1: L, y1: T, x2: L, y2: H - B }));
    var midY = T + (H - T - B) / 2;
    chart.appendChild(svg('text', { 'class': 'axis-label y-label', x: 12, y: midY, 'text-anchor': 'middle', transform: 'rotate(-90 12 ' + midY + ')' }, clip(valueName, 40)));
    var half = labelChars * CHAR_PX / 2;
    xt.ticks.forEach(function (t) {
      var px = x(t);
      var anchor = px - half < L - 8 ? 'start' : px + half > W ? 'end' : 'middle';
      chart.appendChild(svg('line', { 'class': 'axis', x1: px, y1: H - B, x2: px, y2: H - B + 4 }));
      chart.appendChild(svg('text', { 'class': 'x-tick', x: px, y: H - B + 18, 'text-anchor': anchor }, tickLabel(t, xt.step, oneDay)));
    });
    var caption = data.columns[spec.time] + ' (UTC' + (oneDay ? ', ' + new Date(t0).toISOString().slice(0, 10) : '') + ')';
    chart.appendChild(svg('text', { 'class': 'axis-label x-label', x: L + plotW / 2, y: H - 8, 'text-anchor': 'middle' }, caption));
    var base = y(Math.min(Math.max(0, yMin), yMax)).toFixed(1);
    series.forEach(function (s, n) {
      var segs = segments(s.points, ts, interval);
      var line = '', area = '';
      segs.forEach(function (seg) {
        var d = seg.map(function (t, i) { return (i ? 'L' : 'M') + x(t).toFixed(1) + ' ' + y(s.points[t]).toFixed(1); }).join(' ');
        line += (line ? ' ' : '') + d;
        if (seg.length > 1) area += (area ? ' ' : '') + d + ' L' + x(seg[seg.length - 1]).toFixed(1) + ' ' + base + ' L' + x(seg[0]).toFixed(1) + ' ' + base + ' Z';
        if (seg.length === 1) chart.appendChild(svg('circle', { 'class': 'point series-' + n, cx: x(seg[0]).toFixed(1), cy: y(s.points[seg[0]]).toFixed(1), r: 3.5 }));
      });
      if (series.length === 1 && area) chart.appendChild(svg('path', { 'class': 'area series-' + n, d: area }));
      chart.appendChild(svg('path', { 'class': 'line series-' + n, d: line }));
      var item = el('span');
      var sw = svg('svg', { width: 10, height: 10, 'class': 'swatch series-' + n });
      sw.appendChild(svg('rect', { width: 10, height: 10 }));
      item.appendChild(sw); item.appendChild(document.createTextNode(s.name));
      legend.appendChild(item);
    });
    var guide = svg('line', { 'class': 'axis guide', x1: 0, y1: T, x2: 0, y2: H - B, visibility: 'hidden' });
    chart.appendChild(guide);
    var dots = series.map(function (s, n) { var c = svg('circle', { r: 4, 'class': 'series-' + n, visibility: 'hidden' }); chart.appendChild(c); return c; });
    var overlay = svg('rect', { 'class': 'hover', x: L, y: T, width: plotW, height: H - T - B, fill: 'transparent' });
    chart.appendChild(overlay);
    function hide() { tip.hidden = true; guide.setAttribute('visibility', 'hidden'); dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); }); }
    overlay.addEventListener('mousemove', function (ev) {
      var box = chart.getBoundingClientRect();
      var px = (ev.clientX - box.left) * W / box.width;
      var target = t0 + (px - L) / plotW * (t1 - t0);
      var best = ts[0];
      ts.forEach(function (t) { if (Math.abs(t - target) < Math.abs(best - target)) best = t; });
      guide.setAttribute('x1', x(best)); guide.setAttribute('x2', x(best)); guide.setAttribute('visibility', 'visible');
      var lines = [timeLabel(best) + ' UTC'];
      series.forEach(function (s, n) {
        var has = best in s.points;
        dots[n].setAttribute('visibility', has ? 'visible' : 'hidden');
        if (has) { dots[n].setAttribute('cx', x(best)); dots[n].setAttribute('cy', y(s.points[best])); }
        lines.push(s.name + ': ' + (has ? formatNumber(s.points[best]) : '\u2014'));
      });
      tip.textContent = lines.join('\n');
      tip.hidden = false;
      // Flip to the other side of the cursor near the right and top edges.
      var wrapBox = wrap.getBoundingClientRect();
      var cx = ev.clientX - wrapBox.left, cy = ev.clientY - wrapBox.top;
      var left = cx + 12;
      if (left + tip.offsetWidth > wrapBox.width) left = cx - tip.offsetWidth - 12;
      var top = cy - tip.offsetHeight - 8;
      if (top < 0) top = cy + 16;
      tip.style.left = Math.max(0, Math.min(left, wrapBox.width - tip.offsetWidth)) + 'px';
      tip.style.top = Math.max(0, Math.min(top, wrapBox.height - tip.offsetHeight)) + 'px';
    });
    overlay.addEventListener('mouseleave', hide);
    holder.appendChild(chart);
  }
  onWidthChange(wrap, function () { if (model) render(); });
  return { node: wrap, setRows: function (rows) { prepare(rows); render(); } };
}
