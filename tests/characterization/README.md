# Characterization tests

Characterization (golden) tests pin what the server does today, as a client sees it, so
that a refactor (such as moving the server to the SDK's `MCPServer`) can prove it changed
nothing. They do not say the behaviour is right: they record it. A test drives the server
through its public entry point (`create_mcp_server`) and the MCP protocol, then compares
the result with a committed JSON file in `golden/`.

Tests never reach into internals a migration would replace. The server is built with
`create_mcp_server(...)` and driven with the installed SDK's in-process `Client` and
in-memory transport, which accept both the low-level `Server` and `MCPServer`; through the
Starlette app `streaming_server.serve` runs (`http_app`); or as the real entry point
`main()` in a subprocess over stdio (`stdio_launcher`).

## Run

```sh
uv run --offline python -m pytest tests/characterization -q
```

Nothing in this directory touches the network: OpenSearch is faked, the OpenAPI spec comes
from `fixtures/openapi/`, and any DNS lookup or connection off this machine fails the test,
in a server launched as a subprocess too. `http_app` opens no socket.
The tests run the same on Linux, macOS and Windows (the project CI runs all three).

## The rule

**A golden file is never edited to make a change pass. A golden diff is a behaviour change.**

If a golden comparison fails, the code changed observable behaviour. Either fix the code,
or, when the change is intended, delete the golden file in the same commit (so the diff
shows the old behaviour going away) and recreate it as below. The harness never overwrites
an existing golden, not even with `CHAR_GOLDEN_WRITE=1`.

## Create a new golden

1. Write the test with `golden('<area>/<case>', normalize(data))`.
2. Run it once with `CHAR_GOLDEN_WRITE=1`: the missing file
   `golden/<area>/<case>.json` is created. Without the variable a missing golden fails.
   ```sh
   CHAR_GOLDEN_WRITE=1 uv run --offline python -m pytest tests/characterization -q -k <test>
   ```
   In PowerShell: `$env:CHAR_GOLDEN_WRITE = '1'` before the same command, and
   `Remove-Item Env:CHAR_GOLDEN_WRITE` after.
3. Read the new file: it is the behaviour you are pinning. Run again without the variable,
   then commit the test and the golden together.

Goldens are canonical JSON: sorted keys, two-space indent, non-ASCII kept, trailing newline,
UTF-8 with LF line endings. They are compared byte for byte, so they must read the same on
every OS: `.gitattributes` here checks `golden/` and `fixtures/` out with LF even where
`core.autocrlf` is on (Git for Windows' default), and a golden name uses `/` only.
On a mismatch the failure shows a unified diff (golden vs actual).

## Fixtures (`conftest.py`)

- `golden(name, data)`: compare `data` with `golden/<name>.json`.
- `normalize(data)`: copy of `data` with run-to-run values replaced by placeholders:
  recent timestamps (`<TIMESTAMP>`, see "Timestamps and canned data" below), log
  durations (`<DURATION>`), UUIDs (`<UUID>`), session ids (`<SESSION_ID>`), ephemeral
  loopback ports (`<PORT>`), pytest/tempfile paths (`<TMP>`, on Windows too: `\` or `/`
  separators, any drive-letter case, the path below `<TMP>` written with `/`),
  memory-monitor figures (`<MEMORY>`, `<PID>`), and the order of JSON-schema `required`
  lists (built from a set, so sorted). Each rule is narrow and documented in `Normalizer`;
  nothing else is altered.
  A temp path must end at a path boundary (a separator, a quote, whitespace or the end of
  the text): base `.../pytest-3` never matches inside `.../pytest-30`, and a tempfile name
  is `tmp` plus 8 characters, so `mkstemp(suffix='.yml')` names are not replaced; put such
  files under `tmp_path` or a `mkdtemp()` directory.
- `fake_opensearch`: fake cluster behind the aiohttp session opensearch-py's connections
  use, so the server's connection class, auth and error mapping run unchanged.
  `reply(method, path, body)`, `error(method, path, status, body=None)`,
  `fail_connection(method, path)`, `fail_streaming(method, path, body, status=200,
  headers=None)` (below); `requests` lists every request (method, host, path,
  params, decoded body, auth kind without secrets). `GET /` answers `version`
  (default `3.1.0`); `version = None` makes it fail, so the server sees no version and
  skips version filtering. A request with no route fails the test.
  Note: the default environment sets only `OPENSEARCH_URL`, so the server takes the AWS
  credentials path, fails it (no region) and never reaches the fake: it sees no version.
  To reach the fake, configure auth, e.g. `env={'OPENSEARCH_NO_AUTH': 'true'}` or
  `OPENSEARCH_USERNAME`/`OPENSEARCH_PASSWORD`.
  `fail_streaming(method, path, body, *, status=200, headers=None, message=...)` reaches the
  connection's fallback: the streaming attempt `BufferedAsyncHttpConnection.perform_request`
  makes fails at the connection level, before any response (`aiohttp.ClientConnectionError`,
  as `fail_connection`), and the retry `_fallback_perform_request` makes through
  opensearch-py's `AsyncHttpConnection.perform_request` is answered with `status`, `headers`
  and `body`. That retry reads the body with `response.text()`, which returns `body` as it
  is: a `str`, `bytes` or any other object (`None`, a list, an int), so each reaches the
  server's fallback code unchanged; the content-type defaults as for `reply` (`headers`
  override it, e.g. JSON for a bytes body the client should parse). The fake tells the two
  attempts apart by the request: the streaming one passes `allow_redirects`, opensearch-py's
  does not. Each attempt is one entry in `requests`, marked `'attempt': 'streaming'` or
  `'fallback'`, so a test shows how many requests were made (two per call, or one when
  the server does not fall back).
- `pinned_openapi`: `tools.tool_generator.fetch_github_spec` serves `fixtures/openapi/*.yaml`.
  `openapi_unavailable` makes the fetch raise instead.
- `isolated_state` (autouse): each test starts from the module state at import time and a
  clean environment (`OPENSEARCH_*`, `AWS_*`, `MEMORY_*` and proxy variables cleared,
  `OPENSEARCH_URL` pointing at the fake, boto3 kept off local AWS config and instance
  metadata); everything is restored afterwards. Some unit tests delete server modules
  from `sys.modules` and import them again, which leaves a second `TOOL_REGISTRY` behind;
  the conftest imports every server module before any test runs and pins those module
  objects in `sys.modules` during each test, then checks the server's bindings and fails
  loudly if they still differ. `isolated_state.snapshot()` takes the same snapshot the
  fixture restores; `isolated_state.isolation(monkeypatch, url)` is the fixture's own
  setup and teardown as a context manager.
- `no_network` (autouse): fails the test on any DNS lookup or connection off this machine.
  Unix sockets and loopback IP literals (127.0.0.0/8, `::1`) are let through, because
  asyncio's event loop on Windows makes its self-pipe with a loopback `socketpair()`.
- `mcp_session(mode='single', config_file='', cli_overrides=None, env=None)`:
  `async with mcp_session() as session:` builds the server and connects a client.
  `session.list_tools()` / `session.call_tool(name, args)` return results dumped with
  `model_dump(mode='json', by_alias=True, exclude_none=True)`;
  `session.raw(method, params=None, protocol_version=None)` returns the raw JSON-RPC
  response (`result` or `error`) from a fresh connection at that protocol version
  (default `2025-11-25` via the initialize handshake; `2026-07-28` uses the per-request
  envelope); `session.client` is the SDK `Client`.
- `http_app(mode='single', config_file='', env=None)`: `async with http_app() as http:`
  builds the server with `create_mcp_server` and the Starlette app exactly as
  `streaming_server.serve` does (`MCPStarletteApp(server, stateless=True).create_app()`),
  runs its lifespan, and drives it with `httpx.AsyncClient(transport=httpx.ASGITransport(app))`
  (no socket; requests go to `http://127.0.0.1:9900`). `http.get(path)` and
  `http.post_mcp(message, headers=None)` return an `HttpReply` (`status`, `headers`, `text`,
  `events` parsed from `text/event-stream`, `messages`, `message()`, and `summary()` for
  goldens). `headers` go on top of httpx's defaults (and, for `post_mcp`, the
  streamable-HTTP `accept`/`content-type`); a header given as `None` is not sent at all,
  even one httpx adds by default (`{'accept': None}`: no Accept header).
  `http.mcp(method, params=None, headers=None, protocol_version=None)` returns the one
  JSON-RPC message the POST gets back; `protocol_version` sets `mcp-protocol-version`, and
  a modern version (2026-07-28) is stamped as the SDK client does it: `_meta` envelope,
  `mcp-method` and, for name-bearing methods, `mcp-name` (`headers` override them).
  `async with http.open_stream(method, path, headers=None) as stream:` drives the app's ASGI
  interface itself (httpx returns a response only once it is complete, which an event
  stream never is) and yields an `OpenStream` once the response starts: `status`,
  `headers`, `body` so far, `ended`, `events()`, `summary()`, `await stream.next_event()`
  (10 s timeout; `None` at the end of the response) and `await stream.drain()` (reads until
  the stream is 0.5 s silent; `min_events=n` starts that window only once `n` events are
  in). On exit the client disconnects and the app must return (10 s); an exception leaving
  the block is raised as itself, a cleanup failure (the app not returning, or raising) is
  chained as its `__context__`. `http.sse_first_event('/sse')` opens the SSE stream with
  `open_stream`, reads its first event (10 s timeout) and disconnects.
  `ctx.request` is `None` over stdio and in-memory, so anything
  header-based (header auth, `x-mcp-client-name`) is reachable only through `http_app`.
- `plant_tool(name, *, raises=..., returns=None, enable=True)`: a tool for this test only.
  It goes in `TOOL_REGISTRY` with an entry shaped as a real tool's (`baseToolArgs` as its
  args model, `http_methods` `GET`), so it takes the same custom-config, filter and
  executor path (`tool_executor.execute_tool`) as one, in every server the test builds:
  `mcp_session`, `http_app`, and `stdio_launcher`, which copies the tools planted so far
  into the launched server. On every call it raises `raises` (an exception instance, or a
  class instantiated without arguments) or else returns `returns` as the tool's raw content
  (`[]`, `None`, a list of content dicts). The SDK's MCP error class is
  `harness.SDK_MCP_ERROR`: `MCPError(code, message, data=None)`, the name the locked `mcp`
  (2.0.0) exports from `mcp.shared.exceptions` (1.x called it `McpError`); raised from a
  tool, the server answers `tools/call` with that JSON-RPC error. `enable` adds `name` to
  `OPENSEARCH_ENABLED_TOOLS` (keeping its value), as a user enables a tool outside
  `core_tools` in single mode (multi mode does not filter by category); a test that sets
  that variable itself, or passes a config file (whose `tool_filters` win over the
  environment), names the tool there. The registry entry and the variable are undone when
  the test ends, so the tool exists in no other test; a name already in the registry is
  refused. The returned `PlantedTool` lists in `calls` the arguments (the fields set) of
  each call made in this process. Into a launched server an exception crosses as its class
  and `args` (it must be rebuilt by `cls(*args)` from JSON data, as `MCPError` and the
  built-in exceptions are) and content as JSON data; otherwise the launch fails with
  `TypeError`.
- `stdio_launcher(*args, env=None, cwd=None, entry='main')`:
  `async with stdio_launcher('--mode', 'multi') as server:` runs `stdio_launcher.py` with `sys.executable` (no shell): it installs the same
  fake OpenSearch, pinned OpenAPI and network guard in the subprocess, then calls
  `mcp_server_opensearch.main()` with `args`. The subprocess gets this test's (isolated)
  environment plus `env`, UTF-8 stdio and a fresh directory under `tmp_path` as its cwd.
  `fake_opensearch`'s routes (data bodies only, no callables) and version and the
  `pinned_openapi` setting are copied in at launch, as a JSON reply table whose path is in
  `CHAR_FAKE_OPENSEARCH_TABLE`. `server.initialize()`, `request(method, params)`, `notify`,
  `send`/`receive` speak raw JSON-RPC (`\r\n` stripped; `sent` lists what was written);
  `list_tools()`, `call_tool()` and `raw()` return what `mcp_session`'s do. One process is
  one connection: the handshake runs once, at the version the test first initializes with
  (`initialize(version)`, or `raw('initialize', ...)`, whose `notifications/initialized`
  follows before the next request), else at `2025-11-25`; the helpers then use that
  version. `stderr`, `stdout_lines`, `returncode` and `opensearch_requests()` show what the
  process did. On exit stdin is closed and the server waited for; its fake's requests are
  added to `fake_opensearch.requests` (unmatched ones fail the test) and its blocked
  network attempts fail the test.
  `entry='module'` starts the server as `python -m mcp_server_opensearch` does, through the
  package's `__main__.py` (run as `__main__` with `runpy.run_module`, so `sys.argv[0]` is
  its path), with the same arguments and environment; the default `entry='main'` calls
  `main()` named as the console script. `server.entry()` shows which ran, recorded as
  `main()` starts: `{'entry', 'main_module', 'prog', 'args'}` (`main_module` is
  `mcp_server_opensearch.__main__` under `python -m`, `None` otherwise; `prog` the name of
  `sys.argv[0]`). How to start and what to plant reach the launcher in a JSON file whose path
  is in `CHAR_LAUNCH_SPEC`, set for every launch.
- `memory_enabled_session(mode='single', config_file='', env=None)`: `stdio_launcher` with
  `MEMORY_TOOLS_ENABLED=true`, so the memory tools are registered; same helpers as
  `mcp_session`. Memory tools are registered when `tools.memory_tools` is imported and
  several modules hold `TOOL_REGISTRY` references, so reloading in-process is unsafe.
- `harness`: the conftest module itself, for tests of its helpers (`harness.parse_sse`,
  `harness.fake_table`, `harness.SDK_MCP_ERROR`, `harness.exception_spec`): `import conftest`
  is not reliable, since every package-less `conftest.py` under `tests/` imports under that
  name.

Goldens assume the test process starts without `MEMORY_TOOLS_ENABLED=true`: the memory
tool registry is decided when `tools.memory_tools` is imported. Use
`memory_enabled_session` for a server with memory tools.

## Timestamps and canned data

The normalizer turns a timestamp into `<TIMESTAMP>` only when it lies within two days of
the moment of normalizing:

- strings: ISO 8601 / RFC 3339 timestamps (and logging's `2026-11-19 10:00:00,123`) and
  HTTP dates (`Wed, 19 Nov 2026 10:00:00 GMT`), anywhere in any string;
- ints: epoch seconds or milliseconds, only as the value of a dict key that names a time.
  Compared case-insensitively, the key is `now`, `time`, `date`, `timestamp` or `epoch`,
  contains `time` or `date`, or ends with `_at`, `_ms`, `_ts` or `_epoch`: `now_ms`,
  `created_at`, `start_time`, `last_updated_time` and `timestamp` match; `size`, `count`,
  `bytes` and `doc_count` do not. Any other int, including a list item (it has no key),
  is kept as it is, however close to now it reads.

So canned data (files under `fixtures/` and the fake replies written inline in test files)
must use fixed dates far from today, for example `2020-01-01T00:00:00Z`, and no int under
a timestamp key that could read as a current epoch (use e.g. `1577836800`, that same
date). A date near today would be recorded as `<TIMESTAMP>` on one day and as itself a few
days later, which reads as a behaviour change. `test_fixtures_hold_no_dates_near_now`
fails, naming the file and the value, on any ISO 8601 or HTTP date in `fixtures/` within
400 days of now, and on any int under a timestamp key within 400 days of now as epoch
seconds or milliseconds.
