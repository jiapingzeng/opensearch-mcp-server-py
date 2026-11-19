# Characterization tests

Characterization (golden) tests pin what the server does today, as a client sees it, so
that a refactor (such as moving the server to the SDK's `MCPServer`) can prove it changed
nothing. They do not say the behaviour is right: they record it. A test drives the server
through its public entry point (`create_mcp_server`) and the MCP protocol, then compares
the result with a committed JSON file in `golden/`.

Tests never reach into internals a migration would replace. The server is built with
`create_mcp_server(...)` and driven with the installed SDK's in-process `Client` and
in-memory transport, which accept both the low-level `Server` and `MCPServer`.

## Run

```sh
uv run --offline python -m pytest tests/characterization -q
```

Nothing in this directory touches the network: OpenSearch is faked, the OpenAPI spec comes
from `fixtures/openapi/`, and any DNS lookup or connection off this machine fails the test.
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
  `fail_connection(method, path)`; `requests` lists every request (method, host, path,
  params, decoded body, auth kind without secrets). `GET /` answers `version`
  (default `3.1.0`); `version = None` makes it fail, so the server sees no version and
  skips version filtering. A request with no route fails the test.
  Note: the default environment sets only `OPENSEARCH_URL`, so the server takes the AWS
  credentials path, fails it (no region) and never reaches the fake: it sees no version.
  To reach the fake, configure auth, e.g. `env={'OPENSEARCH_NO_AUTH': 'true'}` or
  `OPENSEARCH_USERNAME`/`OPENSEARCH_PASSWORD`.
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

Goldens assume the test process starts without `MEMORY_TOOLS_ENABLED=true`: the memory
tool registry is decided when `tools.memory_tools` is imported.

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
