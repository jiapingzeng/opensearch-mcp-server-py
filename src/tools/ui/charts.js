// Inline-SVG charts: bars for one category and one number (PPL `stats ... by field`), and a
// line/area time series for a time column plus numbers, optionally split by one category.
var CHART_MAX_BARS = 30;
var SERIES_MAX = 8;
var TIME_TYPES = ['timestamp', 'date', 'datetime', 'time', 'date_nanos'];

function parseTime(v) {
  if (typeof v !== 'string') return NaN;
  var s = v.trim().replace(' ', 'T');
  if (!/[zZ]|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';  // PPL returns UTC without a zone
  return Date.parse(s);
}
function timeLabel(ms) { return new Date(ms).toISOString().slice(0, 16).replace('T', ' '); }

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
  var wrap = el('div', null, 'bar-chart');
  var holder = el('div');
  var note = el('div', null, 'muted');
  wrap.appendChild(holder); wrap.appendChild(note);
  function draw(rows) {
    holder.textContent = '';
    var shown = rows.slice(0, CHART_MAX_BARS);
    note.textContent = rows.length > shown.length ? 'Showing the first ' + shown.length + ' of ' + rows.length + ' bars; switch to the table for all rows.' : '';
    var max = 0;
    shown.forEach(function (r) { if (typeof r[spec.value] === 'number' && r[spec.value] > max) max = r[spec.value]; });
    var rowH = 22, labelW = 180, valueW = 90, width = 640, barW = width - labelW - valueW;
    var chart = svg('svg', { 'class': 'chart', viewBox: '0 0 ' + width + ' ' + Math.max(rowH, shown.length * rowH), role: 'img' });
    chart.appendChild(svg('title', {}, data.columns[spec.value] + ' by ' + data.columns[spec.label]));
    shown.forEach(function (r, i) {
      var v = typeof r[spec.value] === 'number' ? r[spec.value] : 0;
      var y = i * rowH;
      var label = r[spec.label] === null || r[spec.label] === undefined ? '(null)' : cellText(r[spec.label]);
      chart.appendChild(svg('text', { x: labelW - 6, y: y + 15, 'text-anchor': 'end' }, label.length > 24 ? label.slice(0, 23) + '\u2026' : label));
      var w = max > 0 && v > 0 ? Math.max(1, (v / max) * barW) : 0;
      chart.appendChild(svg('rect', { 'class': 'bar', x: labelW, y: y + 3, width: w, height: rowH - 6, rx: 2 }));
      chart.appendChild(svg('text', { x: labelW + w + 6, y: y + 15 }, cellText(r[spec.value])));
    });
    holder.appendChild(chart);
  }
  return { node: wrap, setRows: draw };
}

function buildTimeSeries(data, spec) {
  var W = 720, H = 260, L = 64, R = 16, T = 12, B = 40;
  var wrap = el('div', null, 'chart-wrap time-series');
  var legend = el('div', null, 'legend');
  var holder = el('div');
  var tip = el('div', null, 'tooltip');
  tip.hidden = true;
  wrap.appendChild(legend); wrap.appendChild(holder); wrap.appendChild(tip);

  function seriesOf(rows) {
    var series = [], byKey = {};
    function get(key, name) {
      if (!(key in byKey)) { byKey[key] = series.length; series.push({ name: name, points: {} }); }
      return series[byKey[key]];
    }
    rows.forEach(function (r) {
      var t = parseTime(r[spec.time]);
      if (isNaN(t)) return;
      if (spec.category >= 0) {
        var v = r[spec.values[0]];
        var c = r[spec.category];
        if (typeof v === 'number') get(valueKey(c), c === null || c === undefined ? '(null)' : cellText(c)).points[t] = v;
      } else {
        spec.values.forEach(function (i) { if (typeof r[i] === 'number') get('col:' + i, data.columns[i]).points[t] = r[i]; });
      }
    });
    return series.slice(0, SERIES_MAX);
  }

  function draw(rows) {
    holder.textContent = ''; legend.textContent = ''; tip.hidden = true;
    var series = seriesOf(rows);
    var times = {};
    series.forEach(function (s) { Object.keys(s.points).forEach(function (t) { times[t] = 1; }); });
    var ts = Object.keys(times).map(Number).sort(function (a, b) { return a - b; });
    if (ts.length < 2) { holder.appendChild(el('div', 'Not enough time buckets to chart; see the table.', 'muted')); return; }
    var max = 0;
    series.forEach(function (s) { Object.keys(s.points).forEach(function (t) { if (s.points[t] > max) max = s.points[t]; }); });
    if (max <= 0) max = 1;
    var t0 = ts[0], t1 = ts[ts.length - 1];
    var x = function (t) { return L + (t - t0) / (t1 - t0) * (W - L - R); };
    var y = function (v) { return T + (1 - v / max) * (H - T - B); };
    var chart = svg('svg', { 'class': 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
    chart.appendChild(svg('title', {}, series.map(function (s) { return s.name; }).join(', ') + ' over ' + data.columns[spec.time]));
    chart.appendChild(svg('line', { 'class': 'axis', x1: L, y1: H - B, x2: W - R, y2: H - B }));
    chart.appendChild(svg('line', { 'class': 'axis', x1: L, y1: T, x2: L, y2: H - B }));
    chart.appendChild(svg('text', { x: L - 6, y: T + 4, 'text-anchor': 'end' }, formatNumber(max)));
    chart.appendChild(svg('text', { x: L - 6, y: H - B + 4, 'text-anchor': 'end' }, '0'));
    [t0, t0 + (t1 - t0) / 2, t1].forEach(function (t, i) {
      chart.appendChild(svg('text', { x: x(t), y: H - B + 18, 'text-anchor': ['start', 'middle', 'end'][i] }, timeLabel(t)));
    });
    series.forEach(function (s, n) {
      var pts = ts.filter(function (t) { return t in s.points; });
      var d = pts.map(function (t, i) { return (i ? 'L' : 'M') + x(t).toFixed(1) + ' ' + y(s.points[t]).toFixed(1); }).join(' ');
      if (series.length === 1 && pts.length) {
        chart.appendChild(svg('path', { 'class': 'area series-' + n, d: d + ' L' + x(pts[pts.length - 1]).toFixed(1) + ' ' + (H - B) + ' L' + x(pts[0]).toFixed(1) + ' ' + (H - B) + ' Z' }));
      }
      chart.appendChild(svg('path', { 'class': 'line series-' + n, d: d }));
      var item = el('span');
      var sw = svg('svg', { width: 10, height: 10, 'class': 'swatch series-' + n });
      sw.appendChild(svg('rect', { width: 10, height: 10 }));
      item.appendChild(sw); item.appendChild(document.createTextNode(s.name));
      legend.appendChild(item);
    });
    var guide = svg('line', { 'class': 'axis guide', x1: 0, y1: T, x2: 0, y2: H - B, visibility: 'hidden' });
    chart.appendChild(guide);
    var dots = series.map(function (s, n) { var c = svg('circle', { r: 4, 'class': 'series-' + n, visibility: 'hidden' }); chart.appendChild(c); return c; });
    var overlay = svg('rect', { 'class': 'hover', x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
    chart.appendChild(overlay);
    function hide() { tip.hidden = true; guide.setAttribute('visibility', 'hidden'); dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); }); }
    overlay.addEventListener('mousemove', function (ev) {
      var box = chart.getBoundingClientRect();
      var px = (ev.clientX - box.left) * W / box.width;
      var target = t0 + (px - L) / (W - L - R) * (t1 - t0);
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
      var wrapBox = wrap.getBoundingClientRect();
      var left = ev.clientX - wrapBox.left + 12;
      if (left + tip.offsetWidth > wrapBox.width) left = ev.clientX - wrapBox.left - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = Math.max(0, ev.clientY - wrapBox.top - tip.offsetHeight - 8) + 'px';
    });
    overlay.addEventListener('mouseleave', hide);
    holder.appendChild(chart);
  }
  return { node: wrap, setRows: draw };
}
