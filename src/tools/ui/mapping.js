// Index mapping tree for IndexMappingTool: collapsible objects, field types, multi-fields,
// mapping parameters, and a search box that filters by field path or type.
var mappingState = { query: '', expanded: dict() };

function mappingChildren(node) {
  var multi = (node.fields || []).map(function (f) { return Object.assign({ multi: true }, f); });
  return multi.concat(node.children || []);
}
function matches(node, q) {
  return node.path.toLowerCase().indexOf(q) >= 0 || String(node.type || '').toLowerCase().indexOf(q) >= 0;
}
// Nodes that match, or have a matching descendant; ancestors of matches are kept.
function filterMapping(nodes, q) {
  var out = [];
  nodes.forEach(function (n) {
    var kids = filterMapping(mappingChildren(n), q);
    if (matches(n, q) || kids.length) {
      var copy = Object.assign({}, n, { fields: [], children: kids, filtered: true });
      out.push(copy);
    }
  });
  return out;
}
function typeCounts(nodes, counts) {
  nodes.forEach(function (n) { counts[n.type] = (counts[n.type] || 0) + 1; typeCounts(mappingChildren(n), counts); });
  return counts;
}

function renderMapping(data) {
  var m = data.meta || {};
  var names = data.indices.map(function (i) { return i.name; });
  app.appendChild(header(data, 'Mapping: ' + names.join(', '), ['fields: ' + formatNumber(m.field_count || 0)]));
  if (m.truncated) app.appendChild(el('div', 'Showing the first ' + plural(m.field_count, 'field') + '; the text result has the full mapping.', 'notice truncated'));
  var bar = el('div', null, 'toolbar');
  var search = document.createElement('input');
  search.type = 'search'; search.placeholder = 'Filter by field path or type'; search.value = mappingState.query;
  search.className = 'mapping-search';
  search.setAttribute('aria-label', 'Filter fields by path or type');
  bar.appendChild(search);
  var treeOpts = { id: function (n) { return n.path; }, children: mappingChildren, expanded: mappingState.expanded, defaultDepth: 1, render: renderField };
  bar.appendChild(button('Expand all', function () { data.indices.forEach(function (i) { setAllExpanded(i.fields, treeOpts, true); }); drawTrees(); }, 'expand-all'));
  bar.appendChild(button('Collapse all', function () { data.indices.forEach(function (i) { setAllExpanded(i.fields, treeOpts, false); }); drawTrees(); }, 'collapse-all'));
  var count = el('span', null, 'muted match-count');
  bar.appendChild(count);
  app.appendChild(bar);
  var holder = el('div');
  app.appendChild(holder);

  function renderField(n, line) {
    var q = mappingState.query;
    line.appendChild(highlighted(n.name, n.filtered && q && n.name.toLowerCase().indexOf(q.toLowerCase()) >= 0 ? q : '', 'name'));
    line.appendChild(highlighted(n.type || 'object', q && String(n.type || '').toLowerCase().indexOf(q.toLowerCase()) >= 0 ? q : '', 'type'));
    if (n.multi) line.appendChild(el('span', 'multi-field', 'muted'));
    var d = n.details || {};
    var parts = Object.keys(d).map(function (k) { return k + '=' + rawText(d[k]); });
    if (parts.length) line.appendChild(el('span', parts.join(' '), 'muted details'));
    if (n.multi || mappingChildren(n).length === 0) line.title = n.path;
  }
  function drawTrees() {
    holder.textContent = '';
    var q = mappingState.query.trim().toLowerCase();
    var shown = 0;
    data.indices.forEach(function (idx) {
      if (data.indices.length > 1) holder.appendChild(el('h3', idx.name));
      var nodes = q ? filterMapping(idx.fields, q) : idx.fields;
      // While filtering, show every kept node expanded without touching the saved expansion.
      var opts = q ? Object.assign({}, treeOpts, { expanded: dict(), defaultDepth: 1e9 }) : treeOpts;
      (function countAll(list) { list.forEach(function (n) { if (!q || matches(n, q)) shown++; countAll(mappingChildren(n)); }); })(nodes);
      if (!nodes.length) holder.appendChild(el('div', q ? 'No fields match.' : 'No fields.', 'muted'));
      else holder.appendChild(buildTree(nodes, opts));
    });
    count.textContent = q ? plural(shown, 'matching field') : '';
    var types = dict();
    data.indices.forEach(function (idx) { typeCounts(idx.fields, types); });
    summary.textContent = 'types: ' + Object.keys(types).sort().map(function (t) { return t + ' ' + types[t]; }).join(', ');
  }
  var summary = el('div', null, 'muted type-summary');
  app.insertBefore(summary, holder);
  search.addEventListener('input', function () { mappingState.query = search.value; drawTrees(); });
  drawTrees();
}

startView({
  name: 'opensearch-mapping',
  unsupportedText: 'This response has no mapping tree; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'index_mapping' && Array.isArray(data.indices); },
  render: renderMapping
});
