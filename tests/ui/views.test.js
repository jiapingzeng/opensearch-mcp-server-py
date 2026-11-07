// Copyright OpenSearch Contributors
// SPDX-License-Identifier: Apache-2.0

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
  health: { title: 'OpenSearch cluster health', scripts: ['common.js', 'health_card.js', 'health.js'] },
  mapping: { title: 'OpenSearch index mapping', scripts: ['common.js', 'tree.js', 'mapping.js', 'mapping_view.js'] },
  explain: { title: 'OpenSearch score explanation', scripts: ['common.js', 'tree.js', 'explain.js'] },
  overview: { title: 'OpenSearch cluster overview', scripts: ['common.js', 'table.js', 'health_card.js', 'overview.js'] },
  explorer: { title: 'OpenSearch index explorer', scripts: ['common.js', 'table.js', 'tree.js', 'mapping.js', 'explorer.js'] },
};
const read = (name) => fs.readFileSync(path.join(UI, name), 'utf8');
const scripts = (view) => VIEWS[view].scripts.map(read).join('\n');

// Same assembly as load_ui_html; function replacements so `$` patterns in the files stay literal.
function assemble(view) {
  return read('shell.html')
    .replace('{{title}}', () => VIEWS[view].title)
    .replace('/*{{style}}*/', () => read('common.css'))
    .replace('/*{{script}}*/', () => scripts(view));
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
      // Records observers so tests can fire resizes and check disconnects.
      window.observers = [];
      window.ResizeObserver = class {
        constructor(cb) { this.cb = cb; this.nodes = []; this.disconnected = false; window.observers.push(this); }
        observe(node) { this.nodes.push(node); }
        disconnect() { this.disconnected = true; }
      };
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

  test('pills need message with text', async () => {
    for (const [caps, count] of [[{ message: {} }, 0], [{ message: { image: {} } }, 0], [{ message: { text: {} } }, 1]]) {
      const v = await openView('results', caps, SEARCH_INPUT, searchResult());
      assert.equal(v.doc.querySelectorAll('button.pill').length, count, JSON.stringify(caps));
    }
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

describe('aggregation headers', () => {
  test('name the dimension that was cut', async () => {
    const aggregations = [
      { name: 'whole', columns: ['k'], rows: [['a']] },
      { name: 'rows', columns: ['k'], rows: [['a']], truncated: true, row_count: 9 },
      { name: 'cols', columns: ['k'], rows: [['a'], ['b']], truncated: true, column_count: 4 },
      { name: 'both', columns: ['k'], rows: [['a']], truncated: true, row_count: 9, column_count: 4 },
      { name: 'bare', columns: ['k'], rows: [['a']], truncated: true },
    ];
    const base = searchResult();
    const result = { ...base, structuredContent: { ...base.structuredContent, meta: { ...base.structuredContent.meta, aggregations } } };
    const v = await openView('results', {}, SEARCH_INPUT, result);
    assert.deepEqual([...v.doc.querySelectorAll('h3')].map((h) => h.textContent), [
      'Aggregation: whole',
      'Aggregation: rows (showing 1 of 9 rows)',
      'Aggregation: cols (showing 1 of 4 columns)',
      'Aggregation: both (showing 1 of 9 rows and 1 of 4 columns)',
      'Aggregation: bare (truncated)',
    ]);
  });
  test('prefixed sub-aggregation columns and bucket keys are shown', async () => {
    const aggregations = [
      { name: 'm', columns: ['value', 'keys'], rows: [[3, ['a', 'b']]] },
      { name: 'by_host', columns: ['key', 'doc_count', 'agg:meta', 'agg:price_as_string'], rows: [['h1', 5, 1, '9.0']] },
    ];
    const base = searchResult();
    const result = { ...base, structuredContent: { ...base.structuredContent, meta: { ...base.structuredContent.meta, aggregations } } };
    const v = await openView('results', {}, SEARCH_INPUT, result);
    const tables = [...v.doc.querySelectorAll('h3 + *')];
    const cells = (t) => [...t.querySelectorAll('tbody tr.row td:not(.select)')].map((td) => td.textContent);
    const heads = (t) => [...t.querySelectorAll('thead th')].map((th) => th.textContent).join(' ');
    assert.deepEqual(cells(tables[0]), ['3', '["a","b"]']);
    assert.match(heads(tables[1]), /agg:meta.*agg:price_as_string/);
    assert.deepEqual(cells(tables[1]), ['h1', '5', '1', '9.0']);
  });
});

// Tool error texts captured from a local OpenSearch 3.8 through PPLQueryTool and SearchIndexTool.
const ERRORS = JSON.parse(fs.readFileSync(path.join(__dirname, 'opensearch_errors.json'), 'utf8'));

describe('error messages', () => {
  const { window: w } = load('results', { assembled: false });

  test('errorSummary extracts status and reason from OpenSearch error JSON', () => {
    assert.equal(w.errorSummary(ERRORS.ppl_404), 'OpenSearch returned 404: no such index [no-such-index]');
    assert.equal(w.errorSummary(ERRORS.ppl_400), "OpenSearch returned 400: Invalid Query: [wher] is not a valid term at this part of the query: 'source=logs-demo | wher' <-- HERE. Expecting one of 59 possible tokens. Some examples: EOF, 'WHERE', 'FIELDS', 'TABLE', 'RENAME', ...");
    assert.equal(w.errorSummary(ERRORS.search_404), 'OpenSearch returned 404: no such index [missing-index]');
    assert.match(w.errorSummary(ERRORS.search_400), /^OpenSearch returned 400: failed to parse date field \[garbage\]/);
  });

  test('errorSummary handles other shapes and returns null without an OpenSearch error', () => {
    assert.equal(w.errorSummary("Error: NotFoundError(404, 'index_not_found_exception', 'no such index [x]', x, index_or_alias)"), 'OpenSearch returned 404: no such index [x]');
    assert.equal(w.errorSummary("Error: RequestError(400, 'search_phase_execution_exception', {'error': {'root_cause': [{'reason': 'bad \\'thing\\''}]}})"), "OpenSearch returned 400: bad 'thing'");
    assert.equal(w.errorSummary('Error: {"error": "plain message", "status": 503}'), 'OpenSearch returned 503: plain message');
    assert.equal(w.errorSummary('Error: connection refused'), null);
    assert.equal(w.errorSummary('The host could not call PPLQueryTool (no response after 60 s).'), null);
    assert.ok(w.errorSummary('x RequestError(400, \'' + 'a'.repeat(5000) + '\')').length < 400);
  });
});

function pplResult(query, columns, rows, types, extra = {}) {
  return {
    content: [{ type: 'text', text: 'rows' }],
    structuredContent: {
      kind: 'ppl_rows', version: 1, cluster: '', tool: 'PPLQueryTool', refreshable: true,
      columns, rows, meta: { query, column_types: types }, ...extra,
    },
  };
}

describe('failed in-view calls', () => {
  test('banner shows the extracted message with the full text collapsed; the old result is marked stale', async () => {
    const input = { opensearch_cluster_name: '', query: 'source=logs | stats count() by service', size: 0 };
    const ok = pplResult(input.query, ['count()', 'service'], [[3, 'checkout'], [2, 'search']], ['bigint', 'string']);
    const v = await openView('results', { serverTools: {} }, input, ok);
    v.doc.querySelector('details.ppl-editor textarea').value = 'source=no-such-index | head 1';
    v.doc.querySelector('button.ppl-run').click();
    await v.reply('tools/call', { isError: true, content: [{ type: 'text', text: ERRORS.ppl_404 }] });
    const banner = v.doc.querySelector('.banner');
    assert.equal(banner.querySelector('.error-summary').textContent, 'OpenSearch returned 404: no such index [no-such-index]');
    assert.equal(banner.querySelector('details.error-details').open, false);
    assert.equal(banner.querySelector('details.error-details pre').textContent, ERRORS.ppl_404);
    const app = v.doc.getElementById('app');
    assert.ok(app.classList.contains('stale'));
    assert.equal(v.doc.querySelector('.stale-label').textContent, 'Showing previous result for: source=logs | stats count() by service');
    assert.equal(v.doc.querySelectorAll('svg.chart rect.bar').length, 2);
    v.doc.querySelector('button.ppl-run').click();
    await v.reply('tools/call', ok);
    assert.ok(!app.classList.contains('stale'));
    assert.equal(v.doc.querySelector('.stale-label'), null);
  });

  test('a tool error result shows the extracted message', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, {
      isError: true, content: [{ type: 'text', text: ERRORS.search_404 }],
      structuredContent: { kind: 'search_hits', version: 1, omitted: 'error', tool: 'SearchIndexTool' },
    });
    assert.equal(v.doc.querySelector('.notice.error .error-summary').textContent, 'OpenSearch returned 404: no such index [missing-index]');
  });
});

describe('truncation notice', () => {
  const cols = (n) => Array.from({ length: n }, (_, i) => 'c' + i);
  async function notice(rows, columns, meta) {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({
      columns, rows, meta: { index: 'big', total: 10, from: 0, truncated: true, ...meta },
    }));
    return v.doc.querySelector('.notice.truncated').textContent;
  }
  const five = Array.from({ length: 5 }, (_, i) => cols(5).map((c) => c + i));

  test('names only the dimensions that were cut', async () => {
    assert.equal(await notice(five, cols(5), { row_count: 10 }), 'Showing 5 of 10 rows; the text result has everything.');
    assert.equal(await notice(five, cols(5), { row_count: 10, column_count: 5 }), 'Showing 5 of 10 rows; the text result has everything.');
    assert.equal(await notice(five, cols(5), { column_count: 120 }), 'Showing 5 of 120 columns; the text result has everything.');
    assert.equal(await notice(five, cols(5), { row_count: 1500, column_count: 120 }), 'Showing 5 of 1,500 rows and 5 of 120 columns; the text result has everything.');
    assert.equal(await notice(five, cols(5), {}), 'This result was shortened to fit the view; the text result has everything.');
  });

  test('zero rows: singular and plural', async () => {
    assert.equal(await notice([], ['a'], { row_count: 1 }), 'The only row is too large for the 1 MiB view limit; see the text result, or narrow the query.');
    assert.equal(await notice([], ['a'], { row_count: 3 }), 'None of the 3 rows fit in the 1 MiB view limit; see the text result, or narrow the query.');
  });
});

describe('time buckets', () => {
  const { window: w } = load('results', { assembled: false });
  const H = 3600000;

  test('spanInterval reads fixed units from span() and ignores calendar units', () => {
    assert.equal(w.spanInterval('source=x | stats count() by span(@timestamp, 1h)'), H);
    assert.equal(w.spanInterval('stats count() by span(t, 5m), service'), 5 * 60000);
    assert.equal(w.spanInterval('stats count() by span(t,15 minute)'), 15 * 60000);
    assert.equal(w.spanInterval('stats count() by span(t, 1d)'), 24 * H);
    assert.equal(w.spanInterval('stats count() by span(t, 1M)'), null);
    assert.equal(w.spanInterval('stats count() by span(t, 1y)'), null);
    assert.equal(w.spanInterval('stats count() by span(bytes, 10)'), null);
    assert.equal(w.spanInterval('source=x | head 5'), null);
  });

  test('inferInterval needs a fixed-unit span and checks the grid', () => {
    const q = 'stats count() by span(t, 1h)';
    assert.deepEqual(plain(w.inferInterval([0, H, 3 * H], q)), { ms: H, regular: true });
    assert.deepEqual(plain(w.inferInterval([0, 2 * H, 3 * H], q)), { ms: H, regular: true });
    assert.deepEqual(plain(w.inferInterval([0, H, 2.5 * H], q)), { ms: H, regular: false });
    assert.equal(w.inferInterval([0], q), null);
    // Raw rows (no span) are not buckets, whatever their spacing.
    assert.equal(w.inferInterval([0, H, 3 * H], ''), null);
    assert.equal(w.inferInterval([0, H, 3 * H], 'source=logs | fields @timestamp, latency'), null);
    assert.equal(w.inferInterval([0, H, 3 * H], 'stats count() by span(t, 1M)'), null);
  });

  test('countLikeColumns covers count(), dc() and their aliases', () => {
    assert.deepEqual(plain(w.countLikeColumns(['count()', 'dc(user)', 'avg(b)', 'n', 'span'], 'stats count() as n by span(t, 1h)')), [true, true, false, true, false]);
  });

  test('fillBuckets fills a shared grid with 0 for count-like series only', () => {
    const a = { name: 'a', column: 0, points: { 0: 1, [H]: 2, [3 * H]: 5 } };
    const b = { name: 'b', column: 0, points: { 0: 4, [H]: 1 } };
    const grid = w.fillBuckets([a, b], [0, H, 3 * H], { ms: H, regular: true }, () => true);
    assert.deepEqual(plain(grid), [0, H, 2 * H, 3 * H]);
    assert.deepEqual(plain(a.points), { 0: 1, [H]: 2, [2 * H]: 0, [3 * H]: 5 });
    assert.deepEqual(plain(b.points), { 0: 4, [H]: 1, [2 * H]: 0, [3 * H]: 0 });
    const c = { points: { 0: 1, [3 * H]: 2 } };
    assert.deepEqual(plain(w.fillBuckets([c], [0, 3 * H], { ms: H, regular: true }, () => false)), [0, 3 * H]);
    assert.deepEqual(plain(c.points), { 0: 1, [3 * H]: 2 });
    assert.deepEqual(plain(w.fillBuckets([c], [0, 3 * H], { ms: H, regular: false }, () => true)), [0, 3 * H]);
  });

  test('segments break lines at gaps over 1.5x the interval', () => {
    const points = { 0: 1, [H]: 1, [3 * H]: 1, [4 * H]: 1, [6 * H]: 1 };
    const ts = [0, H, 3 * H, 4 * H, 6 * H];
    assert.deepEqual(plain(w.segments(points, ts, { ms: H })), [[0, H], [3 * H, 4 * H], [6 * H]]);
    assert.deepEqual(plain(w.segments(points, ts, null)), [ts]);
  });

  test('niceTicks and timeTicks', () => {
    assert.deepEqual(plain(w.niceTicks(5, 4, true)), [0, 2, 4, 6]);
    assert.deepEqual(plain(w.niceTicks(1500, 4)), [0, 500, 1000, 1500]);
    assert.deepEqual(plain(w.niceTicks(0.37, 4)), [0, 0.1, 0.2, 0.3, 0.4]);
    assert.deepEqual(plain(w.niceTicks(0, 4, true)), [0, 1]);
    const t0 = Date.parse('2026-11-01T00:00:00Z');
    assert.equal(w.timeTicks(t0, t0 + 11 * H, 22).ticks.length, 12);
    const few = w.timeTicks(t0, t0 + 11 * H, 4);
    assert.ok(few.ticks.length <= 4 && few.step === 3 * H, JSON.stringify(plain(few)));
  });

  const span = (h) => `2026-11-01 ${String(h).padStart(2, '0')}:00:00`;
  test('count series split by category share the x domain, with empty buckets drawn at 0', async () => {
    const q = 'source=ts | stats count() by span(@timestamp, 1h), service';
    const rows = [[1, span(0), 'a'], [2, span(1), 'a'], [3, span(3), 'a'], [2, span(0), 'b'], [1, span(1), 'b']];
    const v = await openView('results', {}, {}, pplResult(q, ['count()', 'span(@timestamp,1h)', 'service'], rows, ['bigint', 'timestamp', 'string']));
    const paths = [...v.doc.querySelectorAll('.time-series path.line')].map((p) => p.getAttribute('d'));
    assert.equal(paths.length, 2);
    const ends = paths.map((d) => d.trim().split(/[ML]/).filter(Boolean).pop().split(' ')[0]);
    assert.equal(ends[0], ends[1]);
    assert.ok(paths.every((d) => (d.match(/M/g) || []).length === 1));
    assert.equal(v.doc.querySelector('.chart-note').textContent, '3 empty buckets shown as 0 (PPL omits empty buckets).');
    assert.deepEqual([...v.doc.querySelectorAll('.y-tick')].map((t) => t.textContent), ['0', '1', '2', '3']);
    assert.equal(v.doc.querySelector('.y-label').textContent, 'count()');
  });

  test('non-count series break at missing buckets instead of bridging them', async () => {
    const q = 'source=ts | stats avg(bytes) by span(@timestamp, 1h)';
    const rows = [[10.5, span(0)], [11, span(1)], [12, span(4)], [9, span(5)]];
    const v = await openView('results', {}, {}, pplResult(q, ['avg(bytes)', 'span'], rows, ['double', 'timestamp']));
    const d = v.doc.querySelector('.time-series path.line').getAttribute('d');
    assert.equal((d.match(/M/g) || []).length, 2);
    assert.equal(v.doc.querySelector('.chart-note').textContent, 'Gaps mark missing buckets.');
  });
});

describe('formatting and plurals', () => {
  const { window: w } = load('results', { assembled: false });

  test('byte sizes use 3 significant digits', () => {
    assert.equal(w.formatBytes(512), '512 B');
    assert.equal(w.formatBytes(7680), '7.50 KiB');
    assert.equal(w.formatBytes(141 * 1024 + 300), '141 KiB');
    assert.equal(w.formatBytes(1.2 * 2 ** 40), '1.20 TiB');
    assert.equal(w.formatBytes(99.97 * 1024), '100 KiB');
    assert.equal(w.formatBytes(1048575), '1.00 MiB');
    assert.equal(w.formatBytes(-2048), '-2.00 KiB');
  });

  test('percentages, counts and plurals', () => {
    assert.equal(w.formatPercent(64.28571428571429), '64.3%');
    assert.equal(w.formatPercent(100), '100%');
    assert.equal(w.formatNumber(1234567), '1,234,567');
    assert.equal(w.plural(1, 'row'), '1 row');
    assert.equal(w.plural(0, 'row'), '0 rows');
    assert.equal(w.plural(1500, 'matching field'), '1,500 matching fields');
  });

  test('export status names one row in the singular', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult());
    Object.defineProperty(v.window.navigator, 'clipboard', { value: { writeText: () => Promise.resolve() }, configurable: true });
    v.doc.querySelector('tbody td.select input').click();
    v.doc.querySelector('button.copy-csv').click();
    await flush();
    assert.equal(v.doc.querySelector('.export-status').textContent, 'Copied CSV of 1 row.');
    v.doc.querySelector('tbody td.select input').click();
    v.doc.querySelector('button.copy-json').click();
    await flush();
    assert.equal(v.doc.querySelector('.export-status').textContent, 'Copied JSON of 2 rows.');
  });

  test('numeric cells get thousands separators and at most 3 decimals; long cells keep the full value in the title', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({
      columns: ['_index', 'n', 'blob'], rows: [['logs', 1234567, 'x'.repeat(300)], ['logs', 52.650000000000006, 'y']],
    }));
    const cells = [...v.doc.querySelectorAll('tbody tr.row')][0].querySelectorAll('td:not(.select)');
    assert.equal(cells[1].textContent, '1,234,567');
    const second = [...v.doc.querySelectorAll('tbody tr.row')][1].querySelectorAll('td:not(.select)');
    assert.equal(second[1].textContent, '52.65');
    assert.equal(second[1].title, '52.650000000000006');
    assert.equal(cells[2].title, 'x'.repeat(300));
    assert.ok(cells[2].querySelector('span.cell'));
    cells[2].click();
    assert.match(v.doc.querySelector('tr.detail pre').textContent, new RegExp('x{300}'));
  });
});

describe('filters and stats', () => {
  const rows = [
    ['logs', 'd1', 'checkout', 200], ['logs', 'd2', 'checkout', 500], ['logs', 'd3', 'search', 200],
    ['logs', 'd4', 'search', 200], ['logs', 'd5', 'payments', 503], ['logs', 'd6', 'checkout', 200],
  ];
  const result = () => searchResult({ columns: ['_index', '_id', 'service', 'status'], rows });

  test('header counts filtered rows; stats leave out their own column filter and name the others', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, result());
    assert.equal(v.doc.querySelector('.rows-stat').textContent, 'rows: 6');
    [...v.doc.querySelectorAll('button.chip')].find((b) => b.textContent === 'checkout (3)').click();
    assert.equal(v.doc.querySelector('.rows-stat').textContent, 'rows: 3 of 6 shown');
    const select = v.doc.querySelector('select.stats-column');
    select.value = 'service'; select.dispatchEvent(new v.window.Event('change'));
    assert.equal(v.doc.querySelector('.stats-scope').textContent, 'service over 6 rows; its own filter (service = checkout) is not applied (computed in the view, not by OpenSearch)');
    assert.deepEqual([...v.doc.querySelectorAll('table.top-values tr')].map((t) => t.textContent), ['checkout3', 'search2', 'payments1']);
    select.value = 'status'; select.dispatchEvent(new v.window.Event('change'));
    assert.equal(v.doc.querySelector('.stats-scope').textContent, 'status over 3 rows matching service = checkout (computed in the view, not by OpenSearch)');
    [...v.doc.querySelectorAll('button.chip')].find((b) => b.textContent === 'checkout (3)').click();
    assert.equal(v.doc.querySelector('.rows-stat').textContent, 'rows: 6');
  });
});

describe('nested objects', () => {
  const { window: w } = load('results', { assembled: false });

  test('flattenColumns flattens one level and keeps arrays and deeper objects as values', () => {
    const rows = [['d1', { id: 'u1', geo: { c: 'US' } }, [1, 2]], ['d2', { id: 'u2' }, null], ['d3', null, []]];
    const flat = plain(w.flattenColumns(['_id', 'user', 'tags'], rows, null, 100));
    assert.deepEqual(flat.columns, ['_id', 'user.id', 'user.geo', 'tags']);
    assert.deepEqual(flat.rows, [['d1', 'u1', { c: 'US' }, [1, 2]], ['d2', 'u2', null, null], ['d3', null, null, []]]);
    assert.equal(w.flattenColumns(['a', 'b'], [[{ x: 1 }, 'plain'], ['s', 2]], null, 100), null);
    assert.equal(w.flattenColumns(['user', 'user.id'], [[{ id: 1 }, 2]], null, 100), null);
    assert.equal(w.flattenColumns(['user'], [[{ a: 1, b: 2, c: 3 }]], null, 2), null);
    assert.equal(w.flattenColumns(['a'], [[1], [2]], null, 100), null);
    assert.deepEqual(plain(w.flattenColumns(['s', 'o'], [['x', { k: 1 }]], ['keyword', 'struct'], 100).types), ['keyword', null]);
  });

  test('the view shows dotted columns and Copy JSON follows them', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({
      columns: ['_index', '_id', 'user'], rows: [['logs', 'd1', { id: 'u-1', geo: { country: 'US' } }], ['logs', 'd2', { id: 'u-2' }]],
    }));
    assert.deepEqual([...v.doc.querySelectorAll('thead th:not(.select)')].map((t) => t.textContent), ['_index', '_id', 'user.id', 'user.geo']);
    let copied = null;
    Object.defineProperty(v.window.navigator, 'clipboard', { value: { writeText: (t) => { copied = t; return Promise.resolve(); } }, configurable: true });
    v.doc.querySelector('button.copy-json').click();
    await flush();
    assert.deepEqual(JSON.parse(copied)[0], { _index: 'logs', _id: 'd1', 'user.id': 'u-1', 'user.geo': { country: 'US' } });
  });
});

async function copied(v, selector) {
  let text = null;
  Object.defineProperty(v.window.navigator, 'clipboard', { value: { writeText: (t) => { text = t; return Promise.resolve(); } }, configurable: true });
  v.doc.querySelector(selector).click();
  await flush();
  return text;
}
const headerNames = (doc) => [...doc.querySelectorAll('thead th:not(.select)')].map((t) => t.textContent);
const columnCells = (doc, n) => [...doc.querySelectorAll('tbody tr.row')].map((tr) => tr.querySelectorAll('td:not(.select)')[n].textContent);

describe('keys that shadow Object.prototype', () => {
  const { window: w } = load('results', { assembled: false });
  const hostile = '{"constructor":"c1","toString":"t1","valueOf":"v","hasOwnProperty":5,"__proto__":{"x":1},"id":"u1"}';

  test('flattenColumns keeps every key', () => {
    const flat = plain(w.flattenColumns(['_id', 'user'], [['d1', JSON.parse(hostile)]], null, 100));
    assert.deepEqual(flat.columns, ['_id', 'user.constructor', 'user.toString', 'user.valueOf', 'user.hasOwnProperty', 'user.__proto__', 'user.id']);
    assert.deepEqual(flat.rows, [['d1', 'c1', 't1', 'v', 5, { x: 1 }, 'u1']]);
  });

  test('countLikeColumns, units and dict lookups ignore inherited names', () => {
    assert.deepEqual(plain(w.countLikeColumns(['constructor', 'toString', 'x'], '')), [false, false, false]);
    assert.equal(w.spanInterval('stats count() by span(t, 1 constructor)'), null);
    const d = w.dict();
    assert.equal('constructor' in d, false);
    assert.equal(d.toString, undefined);
  });

  test('columns named constructor and __proto__ show, filter, sort and export', async () => {
    const rows = [['logs', 'a', 'x'], ['logs', 'a', 'y'], ['logs', 'b', 'z']];
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', 'constructor', '__proto__'], rows }));
    assert.deepEqual(headerNames(v.doc), ['_index', 'constructor', '__proto__']);
    [...v.doc.querySelectorAll('button.chip')].find((b) => b.textContent === 'a (2)').click();
    assert.equal(v.doc.querySelector('.rows-stat').textContent, 'rows: 2 of 3 shown');
    const select = v.doc.querySelector('select.stats-column');
    select.value = 'constructor'; select.dispatchEvent(new v.window.Event('change'));
    assert.match(v.doc.querySelector('.stats-scope').textContent, /its own filter \(constructor = a\) is not applied/);
    const json = JSON.parse(await copied(v, 'button.copy-json'));
    assert.deepEqual(Object.keys(json[0]), ['_index', 'constructor', '__proto__']);
    assert.equal(json[0].__proto__, 'x');  // own property, parsed back by JSON.parse
    v.doc.querySelector('tbody tr.row td:not(.select)').click();
    assert.match(v.doc.querySelector('tr.detail pre').textContent, /"__proto__": "x"/);
  });

  test('health status and mapping paths named like prototype members', async () => {
    const health = await openView('health', {}, {}, {
      content: [{ type: 'text', text: 'h' }],
      structuredContent: { kind: 'cluster_health', version: 1, cluster: '', tool: 'ClusterHealthTool', health: { status: 'constructor' },
        columns: ['index', 'status', 'unassigned_shards'], rows: [['b', 'green', 0], ['a', 'toString', 1], ['c', 'red', 2]] },
    });
    assert.equal(health.doc.querySelector('.status-badge').className, 'status-badge unknown');
    assert.deepEqual([...health.doc.querySelectorAll('tbody tr')].map((tr) => tr.firstChild.textContent), ['c', 'b', 'a']);
    const field = (name, path, extra = {}) => ({ name, path, type: 'keyword', ...extra });
    const mapping = await openView('mapping', {}, {}, {
      content: [{ type: 'text', text: 'm' }],
      structuredContent: { kind: 'index_mapping', version: 1, cluster: '', tool: 'IndexMappingTool', meta: { field_count: 3 },
        indices: [{ name: 'i', fields: [{ name: 'a', path: 'a', type: 'object', children: [{ name: 'toString', path: 'toString', type: 'object', children: [field('x', 'toString.x')] }] }] }] },
    });
    const sub = mapping.doc.querySelectorAll('ul.tree ul.tree')[1];
    assert.equal(sub.hidden, true, 'depth-1 node collapsed by default, not opened by an inherited key');
    assert.equal(mapping.doc.querySelector('.type-summary').textContent, 'types: keyword 1, object 2');
  });
});

describe('small numbers', () => {
  const { window: w } = load('results', { assembled: false });

  test('formatNumber never rounds non-zero to 0 and never prints -0', () => {
    assert.equal(w.formatNumber(0.0004), '0.0004');
    assert.equal(w.formatNumber(-0.0001), '-0.0001');
    assert.equal(w.formatNumber(0.00012345), '0.000123');
    assert.equal(w.formatNumber(1e-7), '1e-7');
    assert.equal(w.formatNumber(-1.234e-9), '-1.23e-9');
    assert.equal(w.formatNumber(5e-324), '5e-324');
    assert.equal(w.formatNumber(-0), '0');
    assert.equal(w.formatNumber(0.5), '0.5');
    assert.equal(w.formatNumber(123.45678), '123.457');
    assert.equal(w.formatNumber(-0.99999), '-1');
    assert.equal(w.formatCell(0.0004, 'double'), '0.0004');
    assert.equal(w.formatPercent(0.01), '<0.1%');
    assert.equal(w.formatPercent(-0), '0%');
    assert.equal(w.formatPercent(12.34), '12.3%');
  });

  test('y-axis ticks for small values are distinct and non-zero above the base', () => {
    const labels = plain(w.niceTicks(0.0004, 4, false)).map(w.formatNumber);
    assert.deepEqual(labels, ['0', '0.0001', '0.0002', '0.0003', '0.0004']);
  });

  test('table cells show small decimals', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', 'n'], rows: [['logs', 0.0004], ['logs', -0.0001], ['logs', 1e-7]] }));
    assert.deepEqual(columnCells(v.doc, 1), ['0.0004', '-0.0001', '1e-7']);
  });
});

describe('time series without span()', () => {
  const at = (s) => `2026-11-01 00:00:${String(s).padStart(2, '0')}`;
  test('raw rows draw one continuous line with no fill and no note', async () => {
    const q = 'source=logs | fields @timestamp, latency';
    const rows = [0, 1, 3, 4, 9, 10, 17, 30, 31, 45].map((s, i) => [at(s), i + 0.5]);
    const v = await openView('results', {}, {}, pplResult(q, ['@timestamp', 'latency'], rows, ['timestamp', 'double']));
    const d = v.doc.querySelector('.time-series path.line').getAttribute('d');
    assert.equal((d.match(/M/g) || []).length, 1);
    assert.equal(v.doc.querySelector('.chart-note').textContent, '');
  });

  test('a count column without span() is not filled', async () => {
    const q = 'source=logs | stats count() by @timestamp';
    const rows = [[1, at(0)], [2, at(1)], [3, at(5)]];
    const v = await openView('results', {}, {}, pplResult(q, ['count()', '@timestamp'], rows, ['bigint', 'timestamp']));
    assert.equal(v.doc.querySelector('.chart-note').textContent, '');
    const d = v.doc.querySelector('.time-series path.line').getAttribute('d');
    assert.equal((d.match(/[ML]/g) || []).length, 3);
  });

  test('fillBuckets stops at FILL_MAX_BUCKETS', () => {
    const { window: w } = load('results', { assembled: false });
    const s = { points: { 0: 1, [w.FILL_MAX_BUCKETS * 1000]: 2 } };
    const ts = [0, w.FILL_MAX_BUCKETS * 1000];
    assert.deepEqual(plain(w.fillBuckets([s], ts, { ms: 1000, regular: true }, () => true)), ts);
    assert.equal(s.filled, undefined);
    const ok = { points: { 0: 1, [(w.FILL_MAX_BUCKETS - 1) * 1000]: 2 } };
    assert.equal(w.fillBuckets([ok], [0, (w.FILL_MAX_BUCKETS - 1) * 1000], { ms: 1000, regular: true }, () => true).length, w.FILL_MAX_BUCKETS);
  });
});

describe('negative values', () => {
  test('the y axis extends below 0 and no point drops into the x labels', async () => {
    const q = 'source=ts | stats avg(delta) by span(@timestamp, 1h)';
    const rows = [[-5, '2026-11-01 00:00:00'], [3, '2026-11-01 01:00:00'], [-12.5, '2026-11-01 02:00:00']];
    const v = await openView('results', {}, {}, pplResult(q, ['avg(delta)', 'span'], rows, ['double', 'timestamp']));
    const ticks = [...v.doc.querySelectorAll('.y-tick')].map((t) => Number(t.textContent));
    assert.ok(ticks[0] <= -12.5 && ticks[ticks.length - 1] >= 3, JSON.stringify(ticks));
    const ys = v.doc.querySelector('path.line').getAttribute('d').match(/-?\d+(\.\d+)? (-?\d+(\.\d+)?)/g).map((p) => Number(p.split(' ')[1]));
    assert.ok(ys.every((y) => y >= 12 && y <= 280 - 54), JSON.stringify(ys));
  });

  test('niceTicks covers a negative minimum', () => {
    const { window: w } = load('results', { assembled: false });
    assert.deepEqual(plain(w.niceTicks(3, 4, false, -12.5)), [-15, -10, -5, 0, 5]);
    assert.deepEqual(plain(w.niceTicks(0, 4, true, -3)), [-3, -2, -1, 0]);
  });
});

describe('degenerate tick ranges', () => {
  const { window: w } = load('results', { assembled: false });
  test('niceTicks terminates for subnormal and extreme maxima', () => {
    for (const max of [5e-324, 1e-320, 1e-310, Number.MIN_VALUE * 7, 1e308, Number.MAX_VALUE, Infinity, NaN]) {
      const t = plain(w.niceTicks(max, 4, false));
      assert.ok(t.length >= 2 && t.length < 20 && t.every(Number.isFinite), `${max}: ${JSON.stringify(t)}`);
    }
    assert.deepEqual(plain(w.niceTicks(5e-324, 4, false)), [0, 5e-324]);
  });
});

describe('hostile error text', () => {
  const { window: w } = load('results', { assembled: false });
  const timed = (fn) => { const t = Date.now(); const r = fn(); return { r, ms: Date.now() - t }; };

  test('parsing is bounded for brace floods', () => {
    for (const text of [
      'x RequestError(400, \'' + '{'.repeat(200000) + '\')',
      '{'.repeat(200000),
      '{a'.repeat(100000) + '}'.repeat(100000),
      '{"'.repeat(100000),
      "'reason': '" + '\\'.repeat(100000),
    ]) {
      const { ms } = timed(() => w.errorSummary(text));
      assert.ok(ms < 500, `${text.slice(0, 20)}: ${ms} ms`);
    }
  });

  test('JSON is found after a few stray braces but not after many', () => {
    const body = '{"error": {"reason": "boom"}, "status": 400}';
    assert.equal(w.errorSummary('x {y} {z} ' + body), 'OpenSearch returned 400: boom');
    assert.equal(w.firstJsonObject('{ '.repeat(25) + body), null);
    assert.equal(w.parseOpenSearchError('x'.repeat(70000) + body), null);
  });
});

describe('resize observers', () => {
  test('a replaced chart stops observing', async () => {
    const input = { opensearch_cluster_name: '', query: 'source=logs | stats count() by service', size: 0 };
    const ok = pplResult(input.query, ['count()', 'service'], [[3, 'checkout'], [2, 'search']], ['bigint', 'string']);
    const v = await openView('results', { serverTools: {} }, input, ok);
    const first = v.window.observers.filter((o) => o.nodes.some((n) => n.classList && n.classList.contains('chart-wrap')));
    assert.equal(first.length, 1);
    v.doc.querySelector('button.refresh').click();
    await v.reply('tools/call', ok);
    first[0].cb([]);
    assert.equal(first[0].disconnected, true);
    const live = v.window.observers.filter((o) => !o.disconnected && o.nodes.some((n) => n.classList && n.classList.contains('chart-wrap')));
    live.forEach((o) => o.cb([]));
    assert.equal(live.length, 1);
    assert.equal(live[0].disconnected, false);
  });
});

describe('exports', () => {
  test('table cells are cut to 40 characters; title and CSV keep the whole value', async () => {
    const long = 'x'.repeat(39) + 'yz';
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', 's'], rows: [['logs', long]] }));
    const td = v.doc.querySelector('tbody tr.row').querySelectorAll('td:not(.select)')[1];
    assert.equal(td.textContent, 'x'.repeat(39) + '\u2026');
    assert.equal(td.title, long);
    assert.equal(await copied(v, 'button.copy-csv'), `_index,s\r\nlogs,${long}\r\n`);
  });
  test('CSV carries raw values, not display formatting', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', 'n', 'f'], rows: [['logs', 1234567, 0.0004], ['logs', -0, 52.650000000000006]] }));
    assert.equal(columnCells(v.doc, 1)[0], '1,234,567');
    assert.equal(await copied(v, 'button.copy-csv'), '_index,n,f\r\nlogs,1234567,0.0004\r\nlogs,0,52.650000000000006\r\n');
  });

  test('dotted columns from nested objects escape formulas in headers and values', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({
      columns: ['_index', '@user'], rows: [['logs', { id: '=cmd|calc', name: '\uff1dSUM(A1)' }], ['logs', { id: 'ok', name: '-1' }]],
    }));
    assert.deepEqual(headerNames(v.doc), ['_index', '@user.id', '@user.name']);
    assert.equal(await copied(v, 'button.copy-csv'), "_index,'@user.id,'@user.name\r\nlogs,'=cmd|calc,'\uff1dSUM(A1)\r\nlogs,ok,'-1\r\n");
  });

  test('full-width formula characters are neutralized', () => {
    const { window: w } = load('results', { assembled: false });
    for (const c of ['\uff1d', '\uff0b', '\uff0d', '\uff20']) assert.equal(w.csvCell(c + '1+1'), "'" + c + '1+1');
    assert.equal(w.csvCell('a\uff1d'), 'a\uff1d');
  });
});

describe('integers beyond 2^53', () => {
  const big = ['9007199254740993', '9223372036854775807', '-9223372036854775808'];
  const rows = [['logs', big[0], 'a'], ['logs', 5, 'b'], ['logs', big[1], 'c'], ['logs', big[2], 'd'], ['logs', null, 'e']];
  const result = () => searchResult({ columns: ['_index', 'big', 'k'], rows, meta: { index: 'logs', total: 5, from: 0, integer_strings: ['big'] } });

  test('shown exactly with separators, right-aligned, raw value in the title', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, result());
    assert.deepEqual(columnCells(v.doc, 1), ['9,007,199,254,740,993', '5', '9,223,372,036,854,775,807', '-9,223,372,036,854,775,808', '']);
    const td = v.doc.querySelectorAll('tbody tr.row')[0].querySelectorAll('td:not(.select)')[1];
    assert.ok(td.classList.contains('num'));
    assert.equal(td.title, big[0]);
  });

  test('sort numerically with BigInt, nulls last', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, result());
    v.doc.querySelectorAll('thead th:not(.select)')[1].querySelector('button').click();
    assert.deepEqual(columnCells(v.doc, 2), ['d', 'b', 'a', 'c', 'e']);
    v.doc.querySelectorAll('thead th:not(.select)')[1].querySelector('button').click();
    assert.deepEqual(columnCells(v.doc, 2), ['c', 'a', 'b', 'd', 'e']);
  });

  test('CSV, JSON export, row detail and stats keep every digit', async () => {
    const v = await openView('results', {}, SEARCH_INPUT, result());
    assert.equal(await copied(v, 'button.copy-csv'), `_index,big,k\r\nlogs,${big[0]},a\r\nlogs,5,b\r\nlogs,${big[1]},c\r\nlogs,${big[2]},d\r\nlogs,,e\r\n`);
    const json = await copied(v, 'button.copy-json');
    assert.match(json, new RegExp(`"big": ${big[0]},`));
    assert.match(json, new RegExp(`"big": ${big[2]},`));
    assert.match(json, /"big": null,/);
    assert.deepEqual(JSON.parse(json).map((r) => r.k), ['a', 'b', 'c', 'd', 'e']);
    v.doc.querySelector('tbody tr.row td:not(.select)').click();
    assert.match(v.doc.querySelector('tr.detail pre').textContent, new RegExp(`"big": ${big[0]},`));
    const select = v.doc.querySelector('select.stats-column');
    select.value = 'big'; select.dispatchEvent(new v.window.Event('change'));
    const stat = (k) => [...v.doc.querySelectorAll('.panel.stats tr')].find((tr) => tr.firstChild.textContent === k).lastChild.textContent;
    assert.equal(stat('min'), '-9,223,372,036,854,775,808');
    assert.equal(stat('max'), '9,223,372,036,854,775,807');
  });

  test('PPL bigint and _cat number columns, and flattened nested fields', async () => {
    const ppl = await openView('results', {}, {}, pplResult('source=x', ['id', 'n'], [[big[0], 1], ['12', 2]], ['bigint', 'int'], {}));
    assert.deepEqual(columnCells(ppl.doc, 0), ['9,007,199,254,740,993', '12']);
    const nested = await openView('results', {}, SEARCH_INPUT, searchResult({
      columns: ['_index', 'user'], rows: [['logs', { id: big[1] }], ['logs', { id: 7 }]],
      meta: { index: 'logs', total: 2, from: 0, integer_strings: ['user.id'] },
    }));
    assert.deepEqual(columnCells(nested.doc, 1), ['9,223,372,036,854,775,807', '7']);
    const cat = await openView('table', {}, {}, {
      content: [{ type: 'text', text: 'n' }],
      structuredContent: { kind: 'cat_table', version: 1, cluster: '', tool: 'CatNodesTool', columns: ['name', 'docs'], rows: [['n1', big[0]], ['n2', 3]],
        meta: { title: 'Nodes', source: '_cat/nodes', column_types: ['string', 'number'], integer_strings: ['docs'] } },
    });
    assert.deepEqual(columnCells(cat.doc, 1), ['9,007,199,254,740,993', '3']);
  });

  test('a digit string in a non-numeric column stays text', () => {
    const { window: w } = load('results', { assembled: false });
    assert.equal(w.formatCell('9007199254740993', 'string'), '9007199254740993');
    assert.equal(w.csvCell('-5', 'keyword'), "'-5");
    assert.equal(w.csvCell('-5', 'long'), '-5');
  });
});

describe('accessibility', () => {
  const rows = [['logs', 'd1', 'b'], ['logs', 'd2', 'a'], ['logs', 'd3', 'c']];
  const view = () => openView('results', { updateModelContext: { text: {} }, serverTools: {} }, SEARCH_INPUT, searchResult({ columns: ['_index', '_id', 'service'], rows }));

  test('sortable headers are buttons with aria-sort', async () => {
    const v = await view();
    const ths = [...v.doc.querySelectorAll('thead th:not(.select)')];
    assert.ok(ths.every((th) => th.querySelector('button.sort') && th.getAttribute('aria-sort') === 'none'));
    const btn = ths[2].querySelector('button.sort');
    assert.equal(btn.tabIndex, 0);
    btn.click();
    assert.equal(v.doc.querySelectorAll('thead th:not(.select)')[2].getAttribute('aria-sort'), 'ascending');
    assert.deepEqual(columnCells(v.doc, 2), ['a', 'b', 'c']);
    v.doc.querySelectorAll('thead th:not(.select)')[2].querySelector('button.sort').click();
    assert.equal(v.doc.querySelectorAll('thead th:not(.select)')[2].getAttribute('aria-sort'), 'descending');
    assert.equal(v.doc.querySelectorAll('thead th:not(.select)')[0].getAttribute('aria-sort'), 'none');
  });

  test('row details open from a labelled toggle button with aria-expanded', async () => {
    const v = await view();
    const toggle = v.doc.querySelector('tbody tr.row button.row-toggle');
    assert.equal(toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(toggle.getAttribute('aria-label'), 'Details for row d1');
    toggle.click();
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.match(v.doc.querySelector('tr.detail pre').textContent, /"_id": "d1"/);
    v.doc.querySelectorAll('thead th:not(.select)')[2].querySelector('button').click();
    assert.equal(v.doc.querySelectorAll('tr.detail').length, 1, 'detail follows its row across a sort');
    assert.equal(v.doc.querySelector('tr.detail').previousSibling.querySelector('button.row-toggle'), toggle);
    toggle.click();
    assert.equal(toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(v.doc.querySelector('tr.detail'), null);
  });

  test('every form control has an accessible name', async () => {
    for (const v of [await view(), await openView('results', { serverTools: {} }, { query: 'source=x' }, pplResult('source=x', ['a'], [['x']], ['string']))]) {
      const unnamed = [...v.doc.querySelectorAll('input, select, textarea, button')].filter((c) =>
        !(c.getAttribute('aria-label') || '').trim() && !c.closest('label') && !c.textContent.trim());
      assert.deepEqual(unnamed.map((c) => c.outerHTML), []);
    }
    const m = await openView('mapping', {}, {}, mappingResult());
    assert.equal(m.doc.querySelector('input.mapping-search').getAttribute('aria-label'), 'Filter fields by path or type');
  });

  test('mapping leaves are not tab stops', async () => {
    const m = await openView('mapping', {}, {}, mappingResult());
    const toggles = [...m.doc.querySelectorAll('.node .toggle')];
    assert.equal(toggles.filter((t) => t.tagName === 'BUTTON').length, 1);
    const leaves = toggles.filter((t) => t.classList.contains('leaf'));
    assert.equal(leaves.length, 2);
    assert.ok(leaves.every((t) => t.tagName === 'SPAN' && t.getAttribute('aria-hidden') === 'true'));
  });

  test('sorting moves the existing row elements instead of rebuilding them', async () => {
    const v = await view();
    const before = new Set(v.doc.querySelectorAll('tbody tr.row'));
    v.doc.querySelectorAll('thead th:not(.select)')[2].querySelector('button').click();
    const after = [...v.doc.querySelectorAll('tbody tr.row')];
    assert.ok(after.every((tr) => before.has(tr)));
    v.doc.querySelector('th.select input').click();
    assert.ok([...v.doc.querySelectorAll('tbody td.select input')].every((b) => b.checked));
  });
});

function mappingResult() {
  return {
    content: [{ type: 'text', text: 'm' }],
    structuredContent: { kind: 'index_mapping', version: 1, cluster: '', tool: 'IndexMappingTool', meta: { field_count: 3 },
      indices: [{ name: 'i', fields: [{ name: 'user', path: 'user', type: 'object', children: [{ name: 'id', path: 'user.id', type: 'keyword' }] }, { name: 'ts', path: 'ts', type: 'date', details: { format: 'strict_date_optional_time||epoch_millis' } }] }] },
  };
}

describe('paging labels', () => {
  const page = async (meta, rows) => {
    const v = await openView('results', { serverTools: {} }, SEARCH_INPUT, searchResult({ rows, meta: { index: 'logs', total_relation: 'eq', ...meta } }));
    return v.doc.querySelector('.page-label').textContent;
  };
  test('empty index and past-the-end pages', async () => {
    assert.equal(await page({ total: 0, from: 0 }, []), 'No hits');
    assert.equal(await page({ total: 20, from: 20 }, []), 'No hits after hit 20 of 20');
    assert.equal(await page({ total: 6, from: 0 }, [['logs', 'd1', 'x']]), 'hit 1 of 6');
  });
});

describe('bidi and invisible characters', () => {
  const { window: w } = load('results', { assembled: false });
  test('cells show controls as visible markers; raw values stay raw', async () => {
    assert.equal(w.cellText('\u202eevil.exe\u202c'), '\u27e6U+202E\u27e7evil.exe\u27e6U+202C\u27e7');
    assert.equal(w.cellText('a\u200bb\u2066c\ufeff'), 'a\u27e6U+200B\u27e7b\u27e6U+2066\u27e7c\u27e6U+FEFF\u27e7');
    assert.equal(w.cellText('\u{1F468}\u200d\u{1F469}'), '\u{1F468}\u200d\u{1F469}');
    assert.equal(w.cellText('\u0645\u0631\u062d\u0628\u0627'), '\u0645\u0631\u062d\u0628\u0627');
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', 'f'], rows: [['logs', '\u202eevil.exe\u202c'], ['logs', 'x']] }));
    const td = v.doc.querySelectorAll('tbody tr.row')[0].querySelectorAll('td:not(.select)')[1];
    assert.equal(td.title, '\u202eevil.exe\u202c');
    assert.match(read('common.css'), /td > \.cell \{[^}]*unicode-bidi: plaintext/);
    assert.equal(await copied(v, 'button.copy-csv'), '_index,f\r\nlogs,\u202eevil.exe\u202c\r\nlogs,x\r\n');
  });
  const mark = (cp) => '\u27e6U+' + cp.toString(16).toUpperCase().padStart(4, '0') + '\u27e7';
  test('tags, selector supplement, fillers and stray joiners are marked', () => {
    const tags = [...'hi'].map((c) => String.fromCodePoint(0xE0000 + c.charCodeAt(0))).join('');
    assert.equal(w.cellText('ok' + tags + '\u{E007F}'), 'ok' + mark(0xE0068) + mark(0xE0069) + mark(0xE007F));
    assert.equal(w.cellText('x\u{E0100}\u{E01EF}'), 'x' + mark(0xE0100) + mark(0xE01EF));
    assert.equal(w.cellText('a\u200cb\u034fc\u3164d\uffa0e'), 'a' + mark(0x200C) + 'b' + mark(0x34F) + 'c' + mark(0x3164) + 'd' + mark(0xFFA0) + 'e');
    assert.equal(w.cellText('de\u200dlete'), 'de' + mark(0x200D) + 'lete');
    assert.equal(w.cellText('\u{1F468}\u200d'), '\u{1F468}' + mark(0x200D));
    assert.equal(w.cellText('\u200d\u{1F468}'), mark(0x200D) + '\u{1F468}');
    assert.equal(w.cellText('a\u200db'), 'a' + mark(0x200D) + 'b');
    assert.equal(w.cellText('\u0915\u094d\u200dx'), '\u0915\u094d' + mark(0x200D) + 'x');
    assert.equal(w.cellText('\u0915\u094d\u200cx'), '\u0915\u094d' + mark(0x200C) + 'x');
    assert.equal(w.cellText('\u0915\u094d\u200d\u0995'), '\u0915\u094d' + mark(0x200D) + '\u0995');
    assert.equal(w.cellText('\u0915\u094d\u200d'), '\u0915\u094d' + mark(0x200D));
    assert.equal(w.cellText('\u0915\u200d\u094d\u0937'), '\u0915' + mark(0x200D) + '\u094d\u0937');
  });
  test('every default-ignorable character except VS15 and VS16 is marked', () => {
    const ignorable = /^\p{Default_Ignorable_Code_Point}$/u;
    const missed = [];
    for (let cp = 0; cp <= 0x10FFFF; cp++) {
      if (cp >= 0xD800 && cp <= 0xDFFF) continue;
      const c = String.fromCodePoint(cp);
      if (ignorable.test(c) && cp !== 0xFE0E && cp !== 0xFE0F && w.cellText('a' + c + 'b') !== 'a' + mark(cp) + 'b') missed.push(cp.toString(16));
    }
    assert.deepEqual(missed, []);
    assert.equal(w.cellText('a\u17b4\u180b\ufe00\ufff0\u{e0fff}b'), 'a' + mark(0x17B4) + mark(0x180B) + mark(0xFE00) + mark(0xFFF0) + mark(0xE0FFF) + 'b');
  });
  test('keycaps and joiners in Indic conjuncts stay', () => {
    for (const s of ['1\ufe0f\u20e3 #\ufe0f\u20e3 *\ufe0f\u20e3', '\u0915\u094d\u200d\u0937', '\u0915\u094d\u200c\u0937', '\u0d28\u0d4d\u200d\u0d31', '\u{11013}\u{11046}\u200d\u{11030}']) {
      assert.equal(w.cellText(s), s);
    }
  });
  test('joiners inside emoji sequences stay', () => {
    for (const s of ['\u{1F469}\u{1F3FB}\u200d\u2764\ufe0f\u200d\u{1F468}\u{1F3FC}', '\u{1F3F3}\ufe0f\u200d\u{1F308}', '\u{1F9D1}\u{1F3FE}\u200d\u2695\ufe0f']) {
      assert.equal(w.cellText(s), s);
    }
  });
});

describe('cutting long cells', () => {
  const { window: w } = load('results', { assembled: false });
  const cut = (s) => w.formatCell(s, 'keyword');
  test('an emoji at the cut stays whole', () => {
    const s = cut('a'.repeat(38) + '\u{1F600}\u{1F600}');
    assert.equal(s, 'a'.repeat(38) + '\u2026');
    assert.doesNotMatch(s, /[\ud800-\udbff](?![\udc00-\udfff])/);
  });
  test('a marker at the cut is dropped whole', () => {
    assert.equal(cut('a'.repeat(35) + '\u202ebbbbbbbb'), 'a'.repeat(35) + '\u2026');
    assert.equal(cut('a'.repeat(30) + '\u202ebbbbbbbb'), 'a'.repeat(30) + '\u27e6U+202E\u27e7' + 'b\u2026');
  });
  test('a combining mark stays with its letter, and family emoji stay whole', () => {
    assert.equal(cut('a'.repeat(38) + 'e\u0301xyz'), 'a'.repeat(38) + '\u2026');
    assert.equal(cut('a'.repeat(33) + '\u{1F468}\u200d\u{1F469}\u200d\u{1F467}xyz'), 'a'.repeat(33) + '\u2026');
  });
  test('one grapheme longer than the cut falls back to code points', () => {
    const s = cut('a' + '\u0301'.repeat(60));
    assert.equal(s.length, 40);
    assert.ok(s.endsWith('\u2026'));
  });
  test('without Intl.Segmenter surrogate pairs and markers are still kept whole', () => {
    const v = load('results', { assembled: false });
    v.window.eval('graphemes = null;');
    assert.equal(v.window.formatCell('a'.repeat(38) + '\u{1F600}\u{1F600}', 'keyword'), 'a'.repeat(38) + '\u2026');
    assert.equal(v.window.formatCell('a'.repeat(35) + '\u202ebbbbbbbb', 'keyword'), 'a'.repeat(35) + '\u2026');
    assert.equal(v.window.clip('a'.repeat(37) + '\u{1F600}b', 39), 'a'.repeat(37) + '\u2026');
  });
});

describe('empty strings', () => {
  test('chips and top values label an empty string', async () => {
    const rows = [['logs', 'd1', ''], ['logs', 'd2', '9'], ['logs', 'd3', '9']];
    const v = await openView('results', {}, SEARCH_INPUT, searchResult({ columns: ['_index', '_id', 'zip'], rows }));
    assert.deepEqual([...v.doc.querySelectorAll('button.chip')].map((b) => b.textContent), ['(empty) (1)', '9 (2)']);
    const select = v.doc.querySelector('select.stats-column');
    select.value = 'zip'; select.dispatchEvent(new v.window.Event('change'));
    assert.deepEqual([...v.doc.querySelectorAll('table.top-values tr')].map((t) => t.firstChild.textContent), ['9', '(empty)']);
  });
});

describe('aggregation notes', () => {
  test('server notes for aggregations without a table are shown', async () => {
    const base = searchResult();
    const meta = { ...base.structuredContent.meta, aggregation_notes: ['aggregation m (matrix_stats) is not shown; see the text result'] };
    const v = await openView('results', {}, SEARCH_INPUT, { ...base, structuredContent: { ...base.structuredContent, meta } });
    assert.equal(v.doc.querySelector('.agg-note').textContent, 'Aggregation m (matrix_stats) is not shown; see the text result.');
  });
});

// ---- Task-oriented views: cluster overview and index explorer
const OV_CAPS = { serverTools: {}, message: { text: {} } };
function table(columns, rows, types, extra = {}) {
  return { status: 'ok', columns, rows, meta: { source: 'x', column_types: types }, ...extra };
}
function overviewResult(over = {}) {
  const sections = {
    health: { status: 'ok', data: { cluster_name: 'demo', status: 'yellow', number_of_nodes: 2, unassigned_shards: 1, active_shards_percent_as_number: 75 } },
    nodes: table(['name', 'ip', 'heap.percent'], [['n1', '10.0.0.1', 40], ['n2', '10.0.0.2', 72]], ['string', 'string', 'number']),
    indices: table(['health', 'index', 'docs.count', 'store.size'], [['yellow', 'logs', 6, 7000], ['green', 'orders', 20, 5000]], ['string', 'string', 'number', 'bytes'],
      { summary: { count: 2, by_health: { green: 1, yellow: 1 }, docs: 26, store_bytes: 12000, largest: [['logs', 7000], ['orders', 5000]] } }),
    shards: table(['index', 'shard', 'prirep', 'state', 'docs', 'store', 'node', 'unassigned.reason', 'unassigned.for'],
      [['logs', '0', 'r', 'UNASSIGNED', null, null, null, 'INDEX_CREATED', '1m'], ['logs', '0', 'p', 'STARTED', 6, 7000, 'n2', null, null], ['orders', '0', 'p', 'STARTED', 20, 5000, 'n1', null, null]],
      ['string', 'string', 'string', 'string', 'number', 'bytes', 'string', 'string', 'string'],
      { summary: { total: 3, by_state: { STARTED: 2, UNASSIGNED: 1 }, unassigned: 1, unassigned_by_reason: { INDEX_CREATED: 1 }, per_node: { n1: { primaries: 1, replicas: 0 }, n2: { primaries: 1, replicas: 0 } } },
        explanations: [{ index: 'logs', shard: 0, primary: false, can_allocate: 'no', explanation: 'cannot allocate', nodes: [{ node: 'n2', decision: 'no', reasons: ['a copy of this shard is already allocated to this node'] }] }] }),
    allocation: table(['shards', 'disk.percent', 'node'], [[1, 90, 'n1'], [1, 20, 'n2'], [1, null, 'UNASSIGNED']], ['number', 'number', 'string']),
    ...(over.sections || {}),
  };
  return {
    content: [{ type: 'text', text: 'summary' }],
    structuredContent: {
      kind: 'cluster_overview', version: 1, cluster: over.cluster === undefined ? 'c1' : over.cluster, tool: 'ClusterOverviewTool', refreshable: true,
      requested: over.requested || ['health', 'nodes', 'indices', 'shards', 'allocation'], sections,
      tools: over.tools || { explorer: 'IndexExplorerTool' }, intent: 'Why yellow?', ...(over.extra || {}),
    },
  };
}
const OV_INPUT = { opensearch_cluster_name: 'c1', intent: 'Why yellow?' };
const sectionText = (v, name) => v.doc.querySelector('.section-' + name).textContent;
const lastCall = (v) => v.request('tools/call');

describe('cluster overview', () => {
  test('renders every section with summary counts, explanations and a disk warning', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    assert.deepEqual([...v.doc.querySelectorAll('section.section h2')].map((h) => h.textContent), ['Health', 'Nodes', 'Indices', 'Shards', 'Disk allocation']);
    assert.match(v.doc.querySelector('header').textContent, /Cluster overview: demo/);
    assert.match(v.doc.querySelector('header').textContent, /cluster: c1/);
    assert.equal(v.doc.querySelector('.section-health .status-badge').textContent, 'yellow');
    assert.match(sectionText(v, 'indices'), /2 indices1 green1 yellow26 documents11\.7 KiB total/);
    assert.match(sectionText(v, 'shards'), /Unassigned shards \(1\)/);
    assert.match(sectionText(v, 'shards'), /Reasons: INDEX_CREATED 1/);
    assert.match(v.doc.querySelector('.explanation').textContent, /logs\[0\] replica: can allocate no.*n2 \(no\): a copy/);
    assert.match(sectionText(v, 'allocation'), /1 node at or above 85% disk use .*: n1\./);
    assert.equal(v.doc.querySelector('.intent').textContent, 'Why yellow?');
  });

  test('a failed section shows its error in place while the others render', async () => {
    const err = 'Error getting nodes: TransportError(500, \'{"error":{"reason":"node stats failed"},"status":500}\')';
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { nodes: { status: 'error', error: err }, allocation: { status: 'disabled', tool: 'GetAllocationTool' } } }));
    assert.equal(v.doc.querySelector('.section-nodes .error-summary').textContent, 'OpenSearch returned 500: node stats failed');
    assert.equal(v.doc.querySelectorAll('.section-nodes table').length, 0);
    assert.match(sectionText(v, 'allocation'), /needs GetAllocationTool, which is not enabled/);
    assert.ok(v.doc.querySelector('.section-indices table'));
    assert.ok(v.doc.querySelector('.section-shards table'));
  });

  test('a section that throws while rendering is contained', async () => {
    const bad = { status: 'ok', columns: null, rows: [], meta: {} };
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { nodes: bad } }));
    assert.match(sectionText(v, 'nodes'), /could not be shown .*the other sections are unaffected/);
    assert.ok(v.doc.querySelector('.section-indices table'));
  });

  test('refreshing one section sends only that section and replaces only it', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    const indicesNode = v.doc.querySelector('.section-indices');
    v.doc.querySelector('.section-nodes button.section-refresh').click();
    await flush();
    const call = lastCall(v);
    assert.deepEqual(call.params, { name: 'ClusterOverviewTool', arguments: { opensearch_cluster_name: 'c1', intent: 'Why yellow?', sections: ['nodes'] } });
    assert.equal(v.doc.querySelector('.section-nodes .section-status').textContent, 'Refreshing\u2026');
    const fresh = overviewResult({ requested: ['nodes'], sections: { nodes: table(['name', 'heap.percent'], [['n9', 5]], ['string', 'number']) } });
    v.deliver({ id: call.id, result: fresh });
    await flush();
    assert.match(sectionText(v, 'nodes'), /n9/);
    assert.match(v.doc.querySelector('.section-nodes .section-status').textContent, /^refreshed \d\d:\d\d:\d\d UTC$/);
    assert.equal(v.doc.querySelector('.section-indices'), indicesNode);
  });

  test('a failed section refresh keeps the old data, dimmed, with the error', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    v.doc.querySelector('.section-shards button.section-refresh').click();
    await flush();
    v.deliver({ id: lastCall(v).id, result: { isError: true, content: [{ type: 'text', text: 'boom' }] } });
    await flush();
    assert.equal(v.doc.querySelector('.section-shards .section-error .error-summary').textContent, 'boom');
    assert.ok(v.doc.querySelector('.section-shards .section-body.stale table'));
    assert.match(sectionText(v, 'shards'), /Showing the previous result/);
  });

  test('Refresh all re-calls the tool with the original arguments', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    v.doc.querySelector('button.refresh-all').click();
    await flush();
    assert.deepEqual(lastCall(v).params, { name: 'ClusterOverviewTool', arguments: OV_INPUT });
    assert.equal(v.doc.querySelectorAll('header button.refresh').length, 1);
  });

  test('clicking a node filters shards and disk use; clearing restores them', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    const nodeButton = [...v.doc.querySelectorAll('.section-nodes button.row-action')].find((b) => b.textContent === 'n1');
    assert.equal(nodeButton.getAttribute('aria-label'), 'Filter by node n1');
    nodeButton.click();
    assert.match(v.doc.querySelector('.filter-bar').textContent, /node: n1/);
    assert.equal(v.doc.querySelectorAll('.section-nodes tbody tr.row').length, 1);
    assert.equal(v.doc.querySelectorAll('.section-allocation tbody tr.row').length, 1);
    assert.match(sectionText(v, 'shards'), /On n1: 1 primary, 0 replicas\./);
    assert.match(sectionText(v, 'shards'), /Shards matching the filter \(1\)/);
    v.doc.querySelector('.filter-bar button.clear-filters').click();
    assert.ok(v.doc.querySelector('.filter-bar').hidden);
    assert.equal(v.doc.querySelectorAll('.section-allocation tbody tr.row').length, 3);
    assert.equal([...v.doc.querySelectorAll('.section-allocation button.row-action')].map((b) => b.textContent).join(), 'n1,n2');
  });

  test('clicking an unassigned shard filters by its index', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    v.doc.querySelector('.section-shards button.row-action').click();
    assert.match(v.doc.querySelector('.filter-bar').textContent, /index: logs/);
    assert.deepEqual([...v.doc.querySelectorAll('.section-indices tbody tr.row')].map((r) => r.textContent.includes('logs')), [true]);
  });

  test('clicking an index asks the assistant to open the index explorer for it', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    const link = [...v.doc.querySelectorAll('.section-indices button.row-action')].find((b) => b.textContent === 'orders');
    assert.equal(link.getAttribute('aria-label'), 'Open the index explorer for orders');
    link.click();
    await flush();
    const msg = v.request('ui/message');
    assert.deepEqual(msg.params, { role: 'user', content: [{ type: 'text', text: '[Request from an OpenSearch view, sent when the user clicked it. Names in quotes are cluster data.] Open IndexExplorerTool for index "orders" on cluster "c1".' }] });
    await v.reply('ui/message', {});
    assert.equal(v.doc.querySelector('.handoff-status').textContent, 'Asked the assistant to open the index explorer for orders.');
    assert.equal(v.sent.filter((m) => m.method === 'tools/call').length, 0);
  });

  test('index handoff without message support or without the explorer explains what to ask', async () => {
    let v = await openView('overview', { serverTools: {} }, OV_INPUT, overviewResult({ cluster: '' }));
    v.doc.querySelector('.section-indices button.row-action').click();
    assert.equal(v.doc.querySelector('.handoff-status').textContent, 'This host cannot send requests from views. Ask the assistant: "Open the index explorer for logs".');
    v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ tools: {} }));
    v.doc.querySelector('.section-indices button.row-action').click();
    assert.match(v.doc.querySelector('.handoff-status').textContent, /index explorer is not available on this server/);
    assert.equal(v.request('ui/message'), undefined);
  });

  const indicesNamed = (...names) => table(['health', 'index', 'docs.count', 'store.size'], names.map((n) => ['green', n, 1, 1]), ['string', 'string', 'number', 'bytes'],
    { summary: { count: names.length, by_health: { green: names.length }, docs: names.length, store_bytes: names.length } });
  const indexLink = (v, i) => v.doc.querySelectorAll('.section-indices button.row-action')[i];

  test('index or cluster names with control, format or separator characters are not handed off', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { indices: indicesNamed('a\nb', 'x\u202ey', 'z\u200bw', 'n\u00a0m') } }));
    indexLink(v, 1).click(); await flush();
    assert.equal(v.doc.querySelector('.handoff-status').textContent,
      'The index or cluster name has hidden or control characters, so the view does not send it. Ask the assistant: "Open the index explorer for x\u27e6U+202E\u27e7y" on cluster "c1".');
    indexLink(v, 2).click(); await flush();
    assert.match(v.doc.querySelector('.handoff-status').textContent, /for z\u27e6U\+200B\u27e7w" on cluster "c1"\.$/);
    for (const i of [0, 3]) {
      indexLink(v, i).click(); await flush();
      assert.match(v.doc.querySelector('.handoff-status').textContent, /^The index or cluster name has hidden or control characters/);
    }
    assert.equal(v.request('ui/message'), undefined);
    const c = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ cluster: 'c\u202e1' }));
    indexLink(c, 0).click(); await flush();
    assert.match(c.doc.querySelector('.handoff-status').textContent, /for logs" on cluster "c\u27e6U\+202E\u27e71"\.$/);
    assert.equal(c.request('ui/message'), undefined);
  });

  test('ordinary non-ASCII index names still hand off', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { indices: indicesNamed('caf\u00e9-\u043b\u043e\u0433\u0438.2026') } }));
    indexLink(v, 0).click(); await flush();
    assert.match(v.request('ui/message').params.content[0].text, /Open IndexExplorerTool for index "caf\u00e9-\u043b\u043e\u0433\u0438\.2026" on cluster "c1"\.$/);
  });

  test('a section refresh that resolves after Refresh all does not overwrite the newer data', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    v.doc.querySelector('.section-nodes button.section-refresh').click(); await flush();
    const slow = lastCall(v);
    v.doc.querySelector('button.refresh-all').click(); await flush();
    const all = lastCall(v);
    assert.notEqual(all.id, slow.id);
    v.deliver({ id: all.id, result: overviewResult({ sections: { nodes: table(['name'], [['n7']], ['string']) } }) });
    await flush();
    v.deliver({ id: slow.id, result: overviewResult({ requested: ['nodes'], sections: { nodes: table(['name'], [['n9']], ['string']) } }) });
    await flush();
    assert.match(sectionText(v, 'nodes'), /n7/);
    assert.doesNotMatch(sectionText(v, 'nodes'), /n9/);
    assert.ok(!v.doc.querySelector('.section-nodes button.section-refresh').disabled);
  });

  test('without serverTools there are no refresh buttons and a degraded notice', async () => {
    const v = await openView('overview', {}, OV_INPUT, overviewResult());
    assert.equal(v.doc.querySelectorAll('button.section-refresh, button.refresh-all, button.refresh').length, 0);
    assert.match(v.doc.querySelector('.notice.degraded').textContent, /Ask the assistant to run ClusterOverviewTool again/);
    assert.ok(v.doc.querySelector('.section-nodes table'));
  });

  test('a host-context change keeps filters and the rendered DOM', async () => {
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult());
    [...v.doc.querySelectorAll('.section-nodes button.row-action')][1].click();
    const node = v.doc.querySelector('.section-shards');
    v.deliver({ method: 'ui/notifications/host-context-changed', params: { theme: 'light' } });
    await flush();
    assert.equal(v.doc.documentElement.dataset.theme, 'light');
    assert.equal(v.doc.querySelector('.section-shards'), node);
    assert.match(v.doc.querySelector('.filter-bar').textContent, /node: n2/);
  });

  test('truncated sections say so; unknown versions and omitted markers are handled', async () => {
    const idx = overviewResult().structuredContent.sections.indices;
    let v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { indices: { ...idx, meta: { ...idx.meta, truncated: true, row_count: 900 } } } }));
    assert.match(sectionText(v, 'indices'), /Showing 2 of 900 rows; the text tools \(ListIndexTool\) return everything\./);
    v = await openView('overview', OV_CAPS, OV_INPUT, { content: [], structuredContent: { ...overviewResult().structuredContent, version: 2 } });
    assert.match(v.doc.getElementById('app').textContent, /view format version 2/);
    v = await openView('overview', OV_CAPS, OV_INPUT, { content: [{ type: 'text', text: 'x' }], structuredContent: { kind: 'cluster_overview', version: 1, omitted: 'too_large', bytes: 2097152, tool: 'ClusterOverviewTool' } });
    assert.match(v.doc.getElementById('app').textContent, /too large to show here \(2\.0 MiB\)/);
  });

  test('hostile names render as text with hidden characters marked', async () => {
    const nodes = table(['name'], [['<img src=x onerror=alert(1)>\u202e']], ['string']);
    const v = await openView('overview', OV_CAPS, OV_INPUT, overviewResult({ sections: { nodes } }));
    assert.equal(v.doc.querySelectorAll('img').length, 0);
    assert.equal(v.doc.querySelector('.section-nodes button.row-action').textContent, '<img src=x onerror=alert(1)>\u27e6U+202E\u27e7');
  });
});

function explorerResult(over = {}) {
  const list = table(['health', 'status', 'index', 'docs.count', 'store.size'],
    over.rows || [['yellow', 'open', 'logs', 6, 7000], ['green', 'open', 'orders', 20, 5000], ['red', 'open', 'audit', null, null]],
    ['string', 'string', 'string', 'number', 'bytes'], { summary: { count: 3, by_health: { green: 1, red: 1, yellow: 1 }, docs: 26, store_bytes: 12000 } });
  return {
    content: [{ type: 'text', text: 'summary' }],
    structuredContent: {
      kind: 'index_explorer', version: 1, cluster: 'c1', tool: 'IndexExplorerTool', refreshable: true, pattern: over.pattern || '',
      list: over.list || list, selected: over.selected === undefined ? 'logs' : over.selected, sample_size: 10,
      tools: over.tools || { detail: 'ListIndexTool', mapping: 'IndexMappingTool', shards: 'GetShardsTool', search: 'SearchIndexTool' },
      detail: over.detail || { status: 'ok', index: { name: 'logs', aliases: ['logs-read'], settings: { number_of_shards: '1', number_of_replicas: '1', creation_date: '1793925075273' } } },
    },
  };
}
const EX_INPUT = { opensearch_cluster_name: 'c1', index: '', intent: 'Browse.' };
const MAPPING_SC = { kind: 'index_mapping', version: 1, cluster: 'c1', tool: 'IndexMappingTool', indices: [{ name: 'orders', fields: [
  { name: 'customer', path: 'customer', type: 'object', children: [{ name: 'address', path: 'customer.address', type: 'nested', children: [{ name: 'city', path: 'customer.address.city', type: 'keyword' }] }] },
  { name: 'total', path: 'total', type: 'double' }] }], meta: { field_count: 4 } };
const SHARDS_SC = { kind: 'cat_table', version: 1, cluster: 'c1', tool: 'GetShardsTool', columns: ['index', 'shard', 'prirep', 'state', 'node'],
  rows: [['orders', '0', 'p', 'STARTED', 'n1'], ['orders', '0', 'r', 'UNASSIGNED', null]], meta: { title: 'Shards', source: '_cat/shards', column_types: ['string', 'string', 'string', 'string', 'string'] } };
const SEARCH_SC = { kind: 'search_hits', version: 1, cluster: 'c1', tool: 'SearchIndexTool', columns: ['_index', '_id', '_score', 'customer'],
  rows: [['orders', 'o1', 1, { name: 'Ann', tier: 'gold' }], ['orders', 'o2', 1, { name: 'Bo', tier: 'gold' }]], meta: { index: 'orders', total: 20, total_relation: 'eq', from: 0 } };
const ok = (sc) => ({ content: [{ type: 'text', text: 'text' }], structuredContent: sc });
const calls = (v) => v.sent.filter((m) => m.method === 'tools/call');
async function answer(v, sc) { v.deliver({ id: calls(v).pop().id, result: ok(sc) }); await flush(); }
const tab = (v, name) => v.doc.querySelector(`[role="tab"][data-tab="${name}"]`);

describe('index explorer', () => {
  test('lists indices with health, docs and size, selects the payload selection and shows its overview', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    const items = [...v.doc.querySelectorAll('.index-list button.index-item')];
    assert.deepEqual(items.map((b) => b.querySelector('.name').textContent), ['logs', 'orders', 'audit']);
    assert.equal(items[0].textContent, 'logs health yellow6 docs \u00b7 6.84 KiB');
    assert.equal(items[0].getAttribute('aria-label'), null);
    assert.equal(items[0].getAttribute('aria-current'), 'true');
    assert.equal(v.doc.querySelector('.list-count').textContent, '3 indices');
    const panel = v.doc.querySelector('.tab-panel').textContent;
    assert.match(panel, /yellowhealth/);
    assert.match(panel, /Aliases\s*logs-read/);
    assert.match(panel, /creation_date2026-/);
    assert.equal(calls(v).length, 0); // the overview of the selected index came with the payload
  });

  test('search filters the list by name', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    const box = v.doc.querySelector('input.index-search');
    box.value = 'ORD'; box.dispatchEvent(new v.window.Event('input'));
    assert.deepEqual([...v.doc.querySelectorAll('.index-item .name')].map((n) => n.textContent), ['orders']);
    assert.equal(v.doc.querySelector('.list-count').textContent, '1 of 3 indices');
    box.value = 'zzz'; box.dispatchEvent(new v.window.Event('input'));
    assert.match(v.doc.querySelector('.index-list').textContent, /No indices match/);
  });

  test('selecting an index loads its overview through ListIndexTool with the cluster name', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    v.doc.querySelectorAll('.index-item')[1].click();
    await flush();
    assert.deepEqual(calls(v)[0].params, { name: 'ListIndexTool', arguments: { opensearch_cluster_name: 'c1', index: 'orders', include_detail: true } });
    assert.match(v.doc.querySelector('.tab-panel .loading').textContent, /Loading overview for orders/);
    await answer(v, { kind: 'index_detail', version: 1, indices: [{ name: 'orders', aliases: [], settings: { number_of_shards: '3' } }] });
    assert.match(v.doc.querySelector('.tab-panel').textContent, /AliasesNone\./);
    assert.match(v.doc.querySelector('.tab-panel').textContent, /number_of_shards3/);
  });

  test('each tab calls its read-only tool once per index and is cached', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult({ selected: 'orders', detail: { status: 'ok', index: { name: 'orders', aliases: [], settings: {} } } }));
    tab(v, 'mapping').click(); await flush();
    assert.deepEqual(calls(v).pop().params, { name: 'IndexMappingTool', arguments: { opensearch_cluster_name: 'c1', index: 'orders' } });
    await answer(v, MAPPING_SC);
    assert.deepEqual([...v.doc.querySelectorAll('.tab-panel ul.tree.root > li > .node .name')].map((n) => n.textContent), ['customer', 'total']);
    tab(v, 'shards').click(); await flush();
    assert.deepEqual(calls(v).pop().params, { name: 'GetShardsTool', arguments: { opensearch_cluster_name: 'c1', index: 'orders' } });
    await answer(v, SHARDS_SC);
    assert.match(v.doc.querySelector('.tab-panel').textContent, /2 shard copies, 1 unassigned\./);
    assert.equal(v.doc.querySelectorAll('.tab-panel td.status-unassigned').length, 1);
    tab(v, 'samples').click(); await flush();
    assert.deepEqual(calls(v).pop().params, { name: 'SearchIndexTool', arguments: { opensearch_cluster_name: 'c1', index: 'orders', query_dsl: { query: { match_all: {} } }, size: 10 } });
    await answer(v, SEARCH_SC);
    assert.deepEqual([...v.doc.querySelectorAll('.tab-panel thead th button.sort')].map((b) => b.textContent), ['_index', '_id', '_score', 'customer.name', 'customer.tier']);
    assert.match(v.doc.querySelector('.tab-panel').textContent, /First 2 documents of 20/);
    const before = calls(v).length;
    tab(v, 'mapping').click(); tab(v, 'shards').click(); tab(v, 'overview').click(); tab(v, 'samples').click();
    await flush();
    assert.equal(calls(v).length, before);
    assert.ok(calls(v).every((c) => ['ListIndexTool', 'IndexMappingTool', 'GetShardsTool', 'SearchIndexTool'].includes(c.params.name)));
  });

  test('tab state (mapping filter, sort) survives switching tabs and indices', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult({ selected: 'orders', detail: { status: 'ok', index: { name: 'orders', aliases: [], settings: {} } } }));
    tab(v, 'mapping').click(); await flush(); await answer(v, MAPPING_SC);
    const box = v.doc.querySelector('.tab-panel input.mapping-search');
    box.value = 'city'; box.dispatchEvent(new v.window.Event('input'));
    v.doc.querySelectorAll('.index-item')[0].click(); await flush();
    v.doc.querySelectorAll('.index-item')[1].click(); await flush();
    assert.equal(v.doc.querySelector('.tab-panel input.mapping-search').value, 'city');
    assert.deepEqual([...v.doc.querySelectorAll('.tab-panel .node .name')].map((n) => n.textContent), ['customer', 'address', 'city']);
  });

  test('the selected index and tab persist across a host-context change without re-rendering', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    v.doc.querySelectorAll('.index-item')[1].click(); await flush();
    tab(v, 'shards').click(); await flush(); await answer(v, SHARDS_SC);
    const panel = v.doc.querySelector('.tab-panel');
    v.deliver({ method: 'ui/notifications/host-context-changed', params: { theme: 'light', displayMode: 'fullscreen' } });
    await flush();
    assert.equal(v.doc.querySelector('.tab-panel'), panel);
    assert.equal(v.doc.querySelector('[aria-current="true"] .name').textContent, 'orders');
    assert.equal(v.doc.querySelector('[role="tab"][aria-selected="true"]').dataset.tab, 'shards');
  });

  test('a failed tab load shows the error with Retry, and other tabs still work', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    tab(v, 'mapping').click(); await flush();
    v.deliver({ id: calls(v).pop().id, result: { isError: true, content: [{ type: 'text', text: 'Error getting mapping: NotFoundError(404, \'index_not_found_exception\', \'no such index [logs]\')' }] } });
    await flush();
    assert.equal(v.doc.querySelector('.tab-error .error-summary').textContent, 'OpenSearch returned 404: no such index [logs]');
    v.doc.querySelector('.tab-error button.retry').click(); await flush();
    assert.equal(calls(v).length, 2);
    await answer(v, { ...MAPPING_SC, indices: [{ ...MAPPING_SC.indices[0], name: 'logs' }] });
    assert.ok(v.doc.querySelector('.tab-panel ul.tree'));
  });

  test('unsupported or wrong-kind tab results become in-tab errors', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    tab(v, 'shards').click(); await flush();
    await answer(v, { kind: 'cat_table', version: 1, omitted: 'too_large', bytes: 2e6 });
    assert.match(v.doc.querySelector('.tab-error').textContent, /too large for the 1 MiB view limit/);
    tab(v, 'samples').click(); await flush();
    await answer(v, { kind: 'search_hits', version: 7 });
    assert.match(v.doc.querySelector('.tab-error').textContent, /newer payload version/);
  });

  test('a response for an index the user left is cached but not drawn over the current one', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    tab(v, 'mapping').click(); await flush();
    const slow = calls(v).pop();
    v.doc.querySelectorAll('.index-item')[1].click(); await flush();
    v.deliver({ id: slow.id, result: ok({ ...MAPPING_SC, indices: [{ ...MAPPING_SC.indices[0], name: 'logs' }] }) });
    await flush();
    assert.match(v.doc.querySelector('.tab-panel').textContent, /Loading mapping for orders/);
    v.doc.querySelectorAll('.index-item')[0].click(); await flush();
    assert.ok(v.doc.querySelector('.tab-panel ul.tree'));
  });

  test('keyboard: arrows move through the index list and the tabs', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    const items = [...v.doc.querySelectorAll('.index-item')];
    items[0].focus();
    v.doc.querySelector('.index-list').dispatchEvent(new v.window.KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    assert.equal(v.doc.activeElement, items[1]);
    v.doc.querySelector('.index-list').dispatchEvent(new v.window.KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    assert.equal(v.doc.activeElement, items[2]);
    tab(v, 'overview').focus();
    v.doc.querySelector('[role="tablist"]').dispatchEvent(new v.window.KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    assert.equal(v.doc.activeElement.dataset.tab, 'samples');
    assert.equal(tab(v, 'samples').tabIndex, 0);
    assert.equal(tab(v, 'overview').tabIndex, -1);
    assert.equal(v.doc.querySelector('[role="tabpanel"]').getAttribute('aria-labelledby'), 'tab-samples');
  });

  test('degraded host: no calls, the payload overview still shows, tabs explain what to ask', async () => {
    const v = await openView('explorer', {}, EX_INPUT, explorerResult());
    assert.match(v.doc.querySelector('.notice.degraded').textContent, /does not let views call tools.*"Show the mapping of logs" or "Open the index explorer for orders"/);
    assert.match(v.doc.querySelector('.tab-panel').textContent, /Aliases/);
    tab(v, 'mapping').click(); await flush();
    assert.match(v.doc.querySelector('.degraded-tab').textContent, /Ask the assistant: "Show the mapping of logs"/);
    v.doc.querySelectorAll('.index-item')[1].click(); await flush();
    assert.match(v.doc.querySelector('.degraded-tab').textContent, /"Show the mapping of orders"/);
    tab(v, 'overview').click(); await flush();
    assert.match(v.doc.querySelector('.tab-panel').textContent, /20documents/);
    assert.equal(calls(v).length, 0);
    assert.equal(v.doc.querySelectorAll('button.retry, button.refresh').length, 0);
  });

  test('refresh keeps the selection while the index still exists', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    v.doc.querySelectorAll('.index-item')[2].click(); await flush();
    v.doc.querySelector('header button.refresh').click(); await flush();
    assert.deepEqual(calls(v).pop().params, { name: 'IndexExplorerTool', arguments: EX_INPUT });
    await answer(v, explorerResult().structuredContent);
    assert.equal(v.doc.querySelector('[aria-current="true"] .name').textContent, 'audit');
  });

  test('a tab response from before a refresh is dropped, not cached into the refreshed view', async () => {
    const v = await openView('explorer', OV_CAPS, EX_INPUT, explorerResult());
    tab(v, 'mapping').click(); await flush();
    const slow = calls(v).pop();
    v.doc.querySelector('header button.refresh').click(); await flush();
    await answer(v, explorerResult().structuredContent);
    const fresh = calls(v).pop();
    assert.equal(fresh.params.name, 'IndexMappingTool');
    assert.notEqual(fresh.id, slow.id);
    v.deliver({ id: slow.id, result: ok({ ...MAPPING_SC, indices: [{ ...MAPPING_SC.indices[0], name: 'logs' }] }) });
    await flush();
    assert.match(v.doc.querySelector('.tab-panel').textContent, /Loading mapping for logs/);
    v.deliver({ id: fresh.id, result: ok({ ...MAPPING_SC, indices: [{ name: 'logs', fields: [{ name: 'now', path: 'now', type: 'date' }] }] }) });
    await flush();
    assert.deepEqual([...v.doc.querySelectorAll('.tab-panel .node .name')].map((n) => n.textContent), ['now']);
  });

  test('degraded-host hints mark hidden characters in index names', async () => {
    const rows = [['green', 'open', 'a\u202eb', 1, 1], ['green', 'open', 'c\u200bd', 1, 1]];
    const v = await openView('explorer', {}, EX_INPUT, explorerResult({ rows, selected: 'a\u202eb', detail: { status: 'ok', index: { name: 'a\u202eb', aliases: [], settings: {} } } }));
    assert.match(v.doc.querySelector('.notice.degraded').textContent, /"Show the mapping of a\u27e6U\+202E\u27e7b" or "Open the index explorer for c\u27e6U\+200B\u27e7d"/);
    tab(v, 'mapping').click(); await flush();
    assert.match(v.doc.querySelector('.degraded-tab').textContent, /Ask the assistant: "Show the mapping of a\u27e6U\+202E\u27e7b"\.$/);
  });

  test('a failed list shows the error and the selected index still renders', async () => {
    const v = await openView('explorer', OV_CAPS, { ...EX_INPUT, index: 'logs' }, explorerResult({ list: { status: 'error', error: 'Error getting the index list: AuthorizationException(403, \'security_exception\', \'no permissions for [indices:monitor/settings/get]\')' } }));
    assert.equal(v.doc.querySelector('.index-list .error-summary').textContent, 'OpenSearch returned 403: no permissions for [indices:monitor/settings/get]');
    assert.equal(v.doc.querySelector('.detail-head h2').textContent, 'logs');
    assert.match(v.doc.querySelector('.tab-panel').textContent, /logs-read/);
  });

  test('list truncation is stated in the count', async () => {
    const sc = explorerResult().structuredContent;
    const list = { ...sc.list, meta: { ...sc.list.meta, truncated: true, row_count: 900 } };
    const v = await openView('explorer', OV_CAPS, EX_INPUT, ok({ ...sc, list }));
    assert.equal(v.doc.querySelector('.list-count').textContent, '3 indices (first 3 of 900)');
  });
});
