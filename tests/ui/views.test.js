// Tests for the MCP Apps views in src/tools/ui. Run with `npm ci && npm test` in this directory,
// or through tests/ui/test_views_js.py under pytest.
'use strict';
const { test, describe, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');

const UI = path.resolve(__dirname, '../../src/tools/ui');
// Mirrors UI_RESOURCES[...]['scripts'] in src/tools/ui_resources.py.
const VIEWS = {
  results: { title: 'OpenSearch results', scripts: ['common.js', 'table.js', 'charts.js', 'results.js'] },
  table: { title: 'OpenSearch table', scripts: ['common.js', 'table.js', 'cat_table.js'] },
};
const read = (name) => fs.readFileSync(path.join(UI, name), 'utf8');
const scripts = (view) => VIEWS[view].scripts.map(read).join('\n');

// Same assembly as load_ui_html.
function assemble(view) {
  return read('shell.html')
    .replace('{{title}}', VIEWS[view].title)
    .replace('/*{{style}}*/', read('common.css'))
    .replace('/*{{script}}*/', scripts(view));
}

const tick = () => new Promise((resolve) => setImmediate(resolve));
async function flush() { for (let i = 0; i < 5; i++) await tick(); }
// Values from the view's realm have other prototypes; compare them as plain JSON.
const plain = (value) => JSON.parse(JSON.stringify(value));

// Loads a view with a fake host as window.parent. `sent` records every message the view posts;
// `deliver` hands the view a message as if the host had posted it.
const opened = [];
after(() => opened.forEach((w) => w.close()));  // clears the views' pending call timeouts
function load(view, { assembled = true } = {}) {
  const sent = [];
  const host = { postMessage: (msg) => sent.push(JSON.parse(JSON.stringify(msg))) };
  let listener = null;
  const html = assembled ? assemble(view) : '<!DOCTYPE html><html><body><div id="app"></div></body></html>';
  const dom = new JSDOM(html, {
    runScripts: assembled ? 'dangerously' : 'outside-only',
    pretendToBeVisual: true,
    beforeParse(window) {
      Object.defineProperty(window, 'parent', { value: host, configurable: true });
      window.ResizeObserver = class { observe() {} disconnect() {} };
      const add = window.addEventListener.bind(window);
      window.addEventListener = (type, fn, opts) => { if (type === 'message') listener = fn; return add(type, fn, opts); };
    },
  });
  const { window } = dom;
  opened.push(window);
  if (!assembled) window.eval(scripts(view));
  const deliver = (data) => listener({ source: host, data: { jsonrpc: '2.0', ...data } });
  const request = (method) => sent.filter((m) => m.method === method && m.id !== undefined).pop();
  const reply = async (method, result) => { deliver({ id: request(method).id, result }); await flush(); };
  return { window, doc: window.document, sent, deliver, request, reply };
}

async function openView(view, hostCapabilities, input, result) {
  const v = load(view);
  await v.reply('ui/initialize', { protocolVersion: '2026-01-26', hostCapabilities, hostContext: { theme: 'dark' } });
  if (input) v.deliver({ method: 'ui/notifications/tool-input', params: { arguments: input } });
  if (result) v.deliver({ method: 'ui/notifications/tool-result', params: result });
  await flush();
  return v;
}

const SEARCH_INPUT = {
  opensearch_cluster_name: 'c1', index: 'logs', size: 2,
  query_dsl: { query: { match_all: {} } }, intent: 'Why.', suggestions: ['Next step'],
};
function searchResult(extra = {}) {
  return {
    content: [{ type: 'text', text: 'hits' }],
    structuredContent: {
      kind: 'search_hits', version: 1, cluster: 'c1', tool: 'SearchIndexTool', refreshable: true,
      columns: ['_index', '_id', 'service'], rows: [['logs', 'd1', 'checkout'], ['logs', 'd2', 'search']],
      meta: { index: 'logs', total: 6, total_relation: 'eq', from: 0, dashboards_url: 'http://127.0.0.1:5601/app/x' },
      intent: 'Why.', suggestions: ['Next step'], ...extra,
    },
  };
}

describe('assembled views', () => {
  for (const view of Object.keys(VIEWS)) {
    test(`${view} has exactly one </script>`, () => {
      assert.equal(assemble(view).split('</script>').length - 1, 1);
    });
  }
});

describe('pure functions', () => {
  const { window: w } = load('results', { assembled: false });

  test('csvCell escapes formula prefixes in strings only', () => {
    assert.equal(w.csvCell('=SUM(A1)'), "'=SUM(A1)");
    assert.equal(w.csvCell('+1'), "'+1");
    assert.equal(w.csvCell('-1'), "'-1");
    assert.equal(w.csvCell('@cmd'), "'@cmd");
    assert.equal(w.csvCell('\tx'), "'\tx");
    assert.equal(w.csvCell('\rx'), "\"'\rx\"");
    assert.equal(w.csvCell(-1), '-1');
    assert.equal(w.csvCell('a,b'), '"a,b"');
    assert.equal(w.csvCell('say "hi"'), '"say ""hi"""');
    assert.equal(w.csvCell(' padded'), '" padded"');
    assert.equal(w.csvCell(null), '');
    assert.equal(w.csvCell({ a: 1 }), '"{""a"":1}"');
    assert.equal(w.toCsv(['a', '=b'], [0, 1], [['=1', 2]]), "a,'=b\r\n'=1,2\r\n");
  });

  test('valueKey keeps types and full values apart', () => {
    assert.notEqual(w.valueKey(1), w.valueKey('1'));
    assert.notEqual(w.valueKey(null), w.valueKey('null'));
    assert.equal(w.valueKey(undefined), w.valueKey(null));
    const long = 'x'.repeat(200);
    assert.notEqual(w.valueKey(long + 'a'), w.valueKey(long + 'b'));
    assert.notEqual(w.valueKey({ a: 1 }), w.valueKey('{"a":1}'));
  });

  test('chipColumns picks low-cardinality scalar columns', () => {
    const rows = [[1, 'a', { o: 1 }, 'u1'], ['1', 'a', { o: 2 }, 'u2'], [1, 'b', { o: 3 }, 'u3'], [null, 'b', { o: 4 }, 'u4']];
    const groups = plain(w.chipColumns(['n', 's', 'obj', 'uniq'], rows));
    assert.deepEqual(groups.map((g) => g.name), ['n', 's']);
    const n = groups[0];
    assert.deepEqual(n.keys, ['number:1', 'string:1', 'null:']);
    assert.equal(n.counts['number:1'], 2);
    assert.equal(n.labels['null:'], '(null)');
    assert.deepEqual(plain(w.chipColumns(['n', 's'], rows, ['n'])).map((g) => g.name), ['s']);
    assert.deepEqual(plain(w.chipColumns(['a'], [['x'], ['y']])), []);
    const many = Array.from({ length: 20 }, (_, i) => [String(i % 9)]);
    assert.deepEqual(plain(w.chipColumns(['nine'], many)), []);
    const four = Array.from({ length: 6 }, () => ['a', 'b', 'c', 'd']);
    four[0] = ['z', 'y', 'x', 'w'];
    assert.equal(w.chipColumns(['p', 'q', 'r', 's'], four).length, 3);
  });

  const ppl = (columns, rows) => ({ kind: 'ppl_rows', columns, rows });
  test('timeSeriesSpec detects a time column plus numbers', () => {
    const rows = [[3, '2026-11-01 00:00:00'], [5, '2026-11-01 01:00:00']];
    assert.deepEqual(plain(w.timeSeriesSpec(ppl(['count()', 'span'], rows), ['bigint', 'timestamp'])), { time: 1, values: [0], category: -1 });
    const split = [[3, '2026-11-01 00:00:00', 'a'], [5, '2026-11-01 01:00:00', 'b']];
    assert.equal(w.timeSeriesSpec(ppl(['c', 't', 'svc'], split), ['bigint', 'timestamp', 'string']).category, 2);
    const nine = Array.from({ length: 9 }, (_, i) => [i, '2026-11-01 0' + i + ':00:00', 's' + i]);
    assert.equal(w.timeSeriesSpec(ppl(['c', 't', 'svc'], nine), ['bigint', 'timestamp', 'string']), null);
    assert.equal(w.timeSeriesSpec(ppl(['c', 't'], [[1, 'not a time'], [2, 'x']]), ['bigint', 'timestamp']), null);
    assert.equal(w.timeSeriesSpec({ ...ppl(['count()', 'span'], rows), kind: 'search_hits' }, ['bigint', 'timestamp']), null);
    assert.equal(w.timeSeriesSpec(ppl(['count()', 'span'], rows.slice(0, 1)), ['bigint', 'timestamp']), null);
  });

  test('barSpec needs exactly one numeric and one label column', () => {
    const rows = [[3, 'checkout'], [2, 'search']];
    assert.deepEqual(plain(w.barSpec(ppl(['count()', 'service'], rows), ['bigint', 'string'])), { value: 0, label: 1 });
    assert.deepEqual(plain(w.barSpec(ppl(['service', 'count()'], rows.map((r) => [r[1], r[0]])), ['string', 'bigint'])), { value: 1, label: 0 });
    assert.equal(w.barSpec(ppl(['a', 'b'], [[1, 2]]), ['bigint', 'bigint']), null);
    assert.equal(w.barSpec(ppl(['a', 'b', 'c'], [[1, 'x', 2]]), ['bigint', 'string', 'bigint']), null);
    assert.equal(w.barSpec(ppl(['a', 'b'], [['3', 'x']]), ['bigint', 'string']), null);
    assert.equal(w.barSpec(ppl(['count()', 'service'], []), ['bigint', 'string']), null);
    assert.deepEqual(plain(w.columnTypes(['a', 'b', 'c'], [[1, 'x', null]], ['LONG'])), ['long', 'string', 'string']);
  });

  test('withFrom sets from on a copy, keeping the input shape', () => {
    const dsl = { query: { match_all: {} }, size: 2 };
    assert.deepEqual(plain(w.withFrom(dsl, 4)), { query: { match_all: {} }, size: 2, from: 4 });
    assert.equal(dsl.from, undefined);
    assert.equal(w.withFrom('{"size":2}', 6), '{"size":2,"from":6}');
    assert.equal(w.withFrom('not json', 2), null);
    assert.equal(w.withFrom([1], 2), null);
    assert.equal(w.withFrom(null, 2), null);
  });

  test('truncateUtf8 cuts on character boundaries within the byte budget', () => {
    const accented = w.truncateUtf8('\u00e9'.repeat(1000), 1024);
    assert.ok(Buffer.byteLength(accented) <= 1024 && accented.endsWith('\u2026'));
    assert.equal(w.utf8Length(accented), Buffer.byteLength(accented));
    const emoji = w.truncateUtf8('\u{1F600}'.repeat(400), 1024);
    assert.ok(Buffer.byteLength(emoji) <= 1024 && !/[\ud800-\udbff]\u2026$/.test(emoji));
    assert.equal(w.truncateUtf8('short', 1024), 'short');
  });
});

describe('ui/initialize exchange', () => {
  test('sends ui/initialize first, then initialized and size after the reply', async () => {
    const v = load('results');
    await flush();
    assert.equal(v.sent.length, 1);
    const init = v.sent[0];
    assert.equal(init.method, 'ui/initialize');
    assert.deepEqual(init.params, {
      appInfo: { name: 'opensearch-results', version: '3' },
      appCapabilities: { availableDisplayModes: ['inline', 'fullscreen'] },
      protocolVersion: '2026-01-26',
    });
    await v.reply('ui/initialize', { protocolVersion: '2026-01-26', hostCapabilities: {}, hostContext: { theme: 'dark', displayMode: 'inline' } });
    assert.deepEqual(v.sent.slice(1).map((m) => m.method), ['ui/notifications/initialized', 'ui/notifications/size-changed']);
    assert.equal(v.doc.documentElement.dataset.theme, 'dark');
  });

  test('a failed handshake is shown in the view', async () => {
    const v = load('table');
    v.deliver({ id: v.request('ui/initialize').id, error: { code: -32603, message: 'nope' } });
    await flush();
    assert.match(v.doc.getElementById('app').textContent, /Host handshake failed: nope/);
    assert.equal(v.sent.length, 1);
  });

  test('answers ping and rejects unknown host requests', async () => {
    const v = await openView('table', {});
    v.deliver({ id: 'p1', method: 'ping' });
    v.deliver({ id: 'u1', method: 'ui/unknown' });
    assert.deepEqual(v.sent.slice(-2), [
      { jsonrpc: '2.0', id: 'p1', result: {} },
      { jsonrpc: '2.0', id: 'u1', error: { code: -32601, message: 'Method not found' } },
    ]);
  });
});

describe('view-initiated tool calls', () => {
  test('paging copies every argument and changes only query_dsl.from', async () => {
    const v = await openView('results', { serverTools: {}, message: {} }, SEARCH_INPUT, searchResult());
    v.doc.querySelector('button.page-next').click();
    const call = v.request('tools/call');
    assert.deepEqual(call.params, {
      name: 'SearchIndexTool',
      arguments: { ...SEARCH_INPUT, query_dsl: { query: { match_all: {} }, from: 2 } },
    });
  });

  test('PPL re-run sends the new query and drops the assistant intent and suggestions', async () => {
    const input = { opensearch_cluster_name: '', query: 'source=logs | head 5', size: 0, intent: 'Why.', suggestions: ['Next'] };
    const result = {
      content: [{ type: 'text', text: 'rows' }],
      structuredContent: {
        kind: 'ppl_rows', version: 1, cluster: '', tool: 'PPLQueryTool', refreshable: true,
        columns: ['service'], rows: [['a']], meta: { query: input.query, column_types: ['string'] }, intent: 'Why.',
      },
    };
    const v = await openView('results', { serverTools: {} }, input, result);
    v.doc.querySelector('details.ppl-editor textarea').value = 'source=logs | where status >= 500';
    v.doc.querySelector('button.ppl-run').click();
    assert.deepEqual(v.request('tools/call').params, {
      name: 'PPLQueryTool',
      arguments: { opensearch_cluster_name: '', query: 'source=logs | where status >= 500', size: 0 },
    });
    await v.reply('tools/call', { ...result, structuredContent: { ...result.structuredContent, intent: undefined } });
    v.doc.querySelector('button.refresh').click();
    assert.deepEqual(v.request('tools/call').params.arguments, { opensearch_cluster_name: '', query: 'source=logs | where status >= 500', size: 0 });
  });

  test('no callbacks without serverTools', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult());
    assert.equal(v.doc.querySelector('button.page-next'), null);
    assert.equal(v.doc.querySelector('button.refresh'), null);
  });
});

describe('host capability gating', () => {
  test('pills need message, the deep link needs openLinks', async () => {
    const none = await openView('results', {}, SEARCH_INPUT, searchResult());
    assert.equal(none.doc.querySelectorAll('button.pill').length, 0);
    assert.equal(none.doc.querySelector('button.deep-link'), null);
    assert.equal(none.doc.querySelector('.intent').textContent, 'Why.');
    const all = await openView('results', { message: { text: {} }, openLinks: {} }, SEARCH_INPUT, searchResult());
    assert.deepEqual([...all.doc.querySelectorAll('button.pill')].map((b) => b.textContent), ['Next step']);
    all.doc.querySelector('button.pill').click();
    const msg = all.request('ui/message').params;
    assert.equal(msg.role, 'user');
    assert.ok(msg.content[0].text.startsWith('[Follow-up from an OpenSearch view.') && msg.content[0].text.endsWith('Next step'));
    all.doc.querySelector('button.deep-link').click();
    assert.deepEqual(all.request('ui/open-link').params, { url: 'http://127.0.0.1:5601/app/x' });
  });

  test('model context needs updateModelContext with text', async () => {
    for (const [caps, shown] of [[{}, false], [{ updateModelContext: { image: {} } }, false], [{ updateModelContext: { text: {} } }, true]]) {
      const v = await openView('results', caps, SEARCH_INPUT, searchResult());
      assert.equal(!!v.doc.querySelector('button.add-context'), shown, JSON.stringify(caps));
    }
  });

  test('Download CSV needs downloadFile and sends an embedded text/csv resource', async () => {
    const none = await openView('table', {}, {}, catResult());
    assert.equal(none.doc.querySelector('button.download-csv'), null);
    const v = await openView('table', { downloadFile: {} }, {}, catResult());
    v.doc.querySelector('button.download-csv').click();
    assert.deepEqual(v.request('ui/download-file').params, {
      contents: [{ type: 'resource', resource: { uri: 'file:///opensearch-CatNodesTool.csv', mimeType: 'text/csv', text: "name,heap\r\nn1,10\r\n'=cmd,20\r\n" } }],
    });
    await v.reply('ui/download-file', {});
    assert.equal(v.doc.querySelector('.export-status').textContent, 'Sent CSV of 2 rows to the host to download.');
    v.doc.querySelector('button.download-csv').click();
    await v.reply('ui/download-file', { isError: true });
    assert.equal(v.doc.querySelector('.export-status').textContent, 'Host did not download the file.');
  });
});

function catResult() {
  return {
    content: [{ type: 'text', text: 'nodes' }],
    structuredContent: {
      kind: 'cat_table', version: 1, cluster: '', tool: 'CatNodesTool',
      columns: ['name', 'heap'], rows: [['n1', 10], ['=cmd', 20]],
      meta: { title: 'Nodes', source: '_cat/nodes', column_types: ['string', 'number'] },
    },
  };
}

describe('model context size caps', () => {
  test('cells cut to 1 KB, rows dropped to fit 32 KB, byte count shown', async () => {
    const rows = Array.from({ length: 30 }, (_, i) => ['y'.repeat(2000), 'd' + i, 'x'.repeat(2000)]);
    const v = await openView('results', { updateModelContext: { text: {} } }, SEARCH_INPUT, searchResult({ rows }));
    v.doc.querySelector('th.select input').click();
    v.doc.querySelector('button.add-context').click();
    const text = v.request('ui/update-model-context').params.content[0].text;
    const bytes = Buffer.byteLength(text);
    assert.ok(bytes <= 32768, String(bytes));
    const [prefix, json] = text.split('\n');
    assert.ok(prefix.startsWith('[Added by the user from an OpenSearch view.'));
    const payload = JSON.parse(json);
    const sentRows = payload.rows.length;
    assert.ok(sentRows > 0 && sentRows < 20, String(sentRows));
    for (const r of payload.rows) {
      for (const cell of [r[0], r[2]]) assert.ok(Buffer.byteLength(cell) <= 1024 && cell.endsWith('\u2026'));
    }
    assert.deepEqual(payload.rows.map((r) => r[1]), rows.slice(0, sentRows).map((r) => r[1]));
    assert.equal(payload.note, `first ${sentRows} of 30 selected rows; ${30 - sentRows} dropped to stay within 20 rows and 32 KB; ${2 * sentRows} cells over 1 KB truncated`);
    await v.reply('ui/update-model-context', {});
    assert.equal(v.doc.querySelector('.context-status').textContent,
      `Added ${sentRows} of 30 selected rows (${bytes.toLocaleString('en-US')} bytes) to the model context.`);
  });

  test('small selections go through unchanged without a note', async () => {
    const v = await openView('results', { updateModelContext: { text: {} } }, SEARCH_INPUT, searchResult());
    v.doc.querySelector('tbody td.select input').click();
    v.doc.querySelector('button.add-context').click();
    const payload = JSON.parse(v.request('ui/update-model-context').params.content[0].text.split('\n')[1]);
    assert.deepEqual(payload, { tool: 'SearchIndexTool', cluster: 'c1', columns: ['_index', '_id', 'service'], rows: [['logs', 'd1', 'checkout']] });
  });
});
