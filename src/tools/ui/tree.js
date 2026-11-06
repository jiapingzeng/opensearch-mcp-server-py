// Collapsible tree. Expansion is keyed by node id in `expanded`, so it survives re-renders.
function buildTree(nodes, opts) {
  var rootList = el('ul', null, 'tree root');
  function add(list, node, depth) {
    var kids = opts.children(node) || [];
    var id = opts.id(node);
    var li = el('li');
    var line = el('div', null, 'node');
    var open = id in opts.expanded ? opts.expanded[id] : depth < (opts.defaultDepth || 0);
    var toggle = button(kids.length ? (open ? '\u25BE' : '\u25B8') : '', null, 'toggle' + (kids.length ? '' : ' leaf'));
    toggle.setAttribute('aria-label', kids.length ? (open ? 'Collapse' : 'Expand') : '');
    toggle.setAttribute('aria-expanded', kids.length ? String(open) : 'false');
    line.appendChild(toggle);
    opts.render(node, line);
    li.appendChild(line);
    if (kids.length) {
      var sub = el('ul', null, 'tree');
      sub.hidden = !open;
      kids.forEach(function (k) { add(sub, k, depth + 1); });
      li.appendChild(sub);
      toggle.addEventListener('click', function () {
        var nowOpen = sub.hidden;
        sub.hidden = !nowOpen;
        opts.expanded[id] = nowOpen;
        toggle.textContent = nowOpen ? '\u25BE' : '\u25B8';
        toggle.setAttribute('aria-expanded', String(nowOpen));
        toggle.setAttribute('aria-label', nowOpen ? 'Collapse' : 'Expand');
      });
    }
    list.appendChild(li);
  }
  nodes.forEach(function (n) { add(rootList, n, 0); });
  return rootList;
}

// Every id in the tree mapped to `open`, for expand all / collapse all.
function setAllExpanded(nodes, opts, open) {
  (function walk(list) {
    list.forEach(function (n) {
      var kids = opts.children(n) || [];
      if (kids.length) { opts.expanded[opts.id(n)] = open; walk(kids); }
    });
  })(nodes);
}

// Text with every case-insensitive match of `query` wrapped in <mark>, built with textContent.
function highlighted(text, query, cls) {
  var span = el('span', null, cls);
  var s = String(text);
  if (!query) { span.textContent = s; return span; }
  var lower = s.toLowerCase(), q = query.toLowerCase(), at = 0, i;
  while ((i = lower.indexOf(q, at)) >= 0) {
    if (i > at) span.appendChild(document.createTextNode(s.slice(at, i)));
    span.appendChild(el('mark', s.slice(i, i + q.length)));
    at = i + q.length;
  }
  if (at < s.length) span.appendChild(document.createTextNode(s.slice(at)));
  return span;
}
