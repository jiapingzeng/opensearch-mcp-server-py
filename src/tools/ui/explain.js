// Score explanation tree for ExplainTool: total score, top contributing clauses, and the
// collapsible breakdown OpenSearch returned.
var TOP_CONTRIBUTORS = 5;
var explainState = { expanded: dict() };

function explainChildren(node) { return node.details || []; }
// Ids are positional paths, stable for the same response shape across re-renders.
function indexExplanation(node, path) {
  node._id = path;
  explainChildren(node).forEach(function (c, i) { indexExplanation(c, path + '.' + i); });
}
// Term-level clauses (weight(...)) carry the per-field contributions; fall back to leaves.
function contributors(root) {
  var weights = [], leaves = [];
  (function walk(n) {
    if (/^weight\(/.test(n.description)) { weights.push(n); return; }
    var kids = explainChildren(n);
    if (!kids.length) leaves.push(n);
    kids.forEach(walk);
  })(root);
  var list = (weights.length ? weights : leaves).filter(function (n) { return typeof n.value === 'number'; });
  list.sort(function (a, b) { return b.value - a.value; });
  return list.slice(0, TOP_CONTRIBUTORS);
}

function renderExplain(data) {
  var e = data.explanation;
  var m = data.meta || {};
  var stats = ['matched: ' + (data.matched ? 'yes' : 'no')];
  if (e && typeof e.value === 'number') stats.push('score: ' + formatNumber(e.value));
  app.appendChild(header(data, 'Explain ' + (data.index || '') + (data.id !== undefined && data.id !== null ? ' / ' + data.id : ''), stats));
  if (m.truncated) app.appendChild(el('div', 'Showing the first ' + plural(m.node_count, 'explanation node') + '; the text result has the full tree.', 'notice truncated'));
  if (!data.matched) app.appendChild(el('div', 'The document does not match the query' + (e && e.description ? ': ' + e.description : '.'), 'notice not-matched'));
  if (!e) return;
  indexExplanation(e, 'r');
  var top = contributors(e);
  if (data.matched && top.length) {
    app.appendChild(el('div', 'Top contributors', 'label'));
    var t = el('table', null, 'contributors');
    var total = typeof e.value === 'number' && e.value > 0 ? e.value : null;
    top.forEach(function (n) {
      var tr = el('tr');
      tr.appendChild(el('td', formatNumber(n.value), 'num'));
      var bar = el('td');
      if (total) {
        var meter = el('span', null, 'meter');
        meter.style.width = Math.max(2, Math.round(160 * Math.min(1, n.value / total))) + 'px';
        bar.appendChild(meter);
        bar.appendChild(document.createTextNode(' ' + formatPercent(100 * n.value / total)));
      }
      tr.appendChild(bar);
      tr.appendChild(el('td', cellText(n.description), 'description'));
      t.appendChild(tr);
    });
    app.appendChild(t);
  }
  var treeOpts = {
    id: function (n) { return n._id; }, children: explainChildren, expanded: explainState.expanded, defaultDepth: 2,
    render: function (n, line) {
      line.appendChild(el('span', typeof n.value === 'number' ? formatNumber(n.value) : cellText(n.value), 'score'));
      var d = el('span', n.description, 'description');
      line.appendChild(d);
    }
  };
  var bar = el('div', null, 'toolbar');
  bar.appendChild(el('span', 'Breakdown', 'label'));
  bar.appendChild(button('Expand all', function () { setAllExpanded([e], treeOpts, true); draw(); }, 'expand-all'));
  bar.appendChild(button('Collapse all', function () { setAllExpanded([e], treeOpts, false); draw(); }, 'collapse-all'));
  app.appendChild(bar);
  var holder = el('div', null, 'explain-tree');
  app.appendChild(holder);
  function draw() { holder.textContent = ''; holder.appendChild(buildTree([e], treeOpts)); }
  draw();
}

startView({
  name: 'opensearch-explain',
  unsupportedText: 'This response has no score explanation; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'explain'; },
  render: renderExplain
});
