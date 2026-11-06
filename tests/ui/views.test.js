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

  test('inferInterval falls back to the smallest spacing and checks the grid', () => {
    assert.deepEqual(plain(w.inferInterval([0, H, 3 * H], '')), { ms: H, regular: true });
    assert.deepEqual(plain(w.inferInterval([0, 2 * H, 3 * H], 'stats count() by span(t, 1h)')), { ms: H, regular: true });
    assert.deepEqual(plain(w.inferInterval([0, H, 2.5 * H], '')), { ms: H, regular: false });
    assert.equal(w.inferInterval([0], ''), null);
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
