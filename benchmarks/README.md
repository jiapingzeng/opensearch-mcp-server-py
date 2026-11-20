# Benchmarks

## `tools/call` timing in one session (`tools_call_case.py`)

Measures how long `tools/call` takes in one MCP session, in-process and without a cluster,
on several source trees, to tell whether a change made calls slower. It was written for the
step 16 question (is `OpenSearchMCPServer` on the SDK's `MCPServer` slower than the
low-level `Server`?); the findings are in
[`docs/mcpserver-migration-behaviour-changes.md`](../docs/mcpserver-migration-behaviour-changes.md#performance-toolscall-timing-after-step-16).

```sh
uv run --offline python scripts/bench_tools_call.py --quick   # 2 rounds: checks that it runs
uv run --offline python scripts/bench_tools_call.py           # full run, 30 rounds
uv run --offline python scripts/bench_tools_call.py --profile memory-concurrent-n50-0ms
```

The full run takes about an hour on a loaded host (each round runs every workload on every
tree) and writes `benchmarks/results/tools_call.json`; `--quick` writes to the temp
directory. Options: `--rev REV` adds a revision (repeatable), `--only-revs` runs only the
`--rev` ones, `--rounds N` (at least 30 for a full run), `--transport memory|http`,
`--warmup N` (untimed calls per session, default 2), `--batches N` (timed batches per
session, default 1), `--out PATH`. Standard library only, no new dependency.

### How it runs

- Trees: `508e806` (the parity baseline), `c71570c` (the last point within ±3%, still the
  low-level `Server`) and `18b7002` (steps 16 and 16b). Each is extracted with
  `git archive <rev> src tests/characterization pyproject.toml` into a temporary
  directory, never checked out; the driver copies `tools_call_case.py` into the tree's
  `tests/characterization/` and makes the tree read-only.
- A run is one pytest process in that tree (`-c` and `--rootdir` the tree's,
  `-p no:cacheprovider`), with the tree's `src` first on `PYTHONPATH`. The case uses the
  tree's own characterization harness: the server is built by
  `streaming_server.create_mcp_server`, OpenSearch is the harness's `fake_opensearch`, and
  nothing touches the network. Each run reports where every loaded module of
  `mcp_server_opensearch`, `tools` and `opensearch`, the harness and the case came from;
  the driver fails if one is outside the tree (the venv's editable install points at this
  worktree's `src`).
- Rounds: each round runs every tree once, and the trees' order rotates each round, so
  drift in host load hits them alike. The run records `os.getloadavg()` at the start, at
  each round and at the end.

### Workloads

Each workload is `<transport>-<mode>-n<N>-<latency>ms`, run in its own MCP session: 2
untimed warm-up calls, then N `ListIndexTool` calls (each makes 2 fake OpenSearch requests:
`GET /` for the version and `GET /_cat/indices`).

- Transport `memory`: the SDK's in-memory `Client` (as the harness's `mcp_session`).
  `http`: the SDK's streamable-HTTP client (`ClientSession` over `streamable_http_client`,
  as the integration test's `mcp_client`) through `httpx2.ASGITransport` into the
  Starlette app the harness's `http_app` runs (`MCPStarletteApp(server, stateless=True)`,
  as `serve` builds it). Each transport builds its server once per run; each workload opens
  a new session on it.
- Mode `sequential`: one `await` after another. `concurrent`: N tasks wait on one event and
  are released together, then gathered, as
  `integration_tests/concurrency/test_concurrent_single_client.py` does.
- N: 5 (the integration test) and 50.
- Latency: 0 ms (pure per-call overhead) or 50 ms, an `asyncio.sleep` before each fake
  OpenSearch reply, so concurrent calls can overlap.

### Results

- Per-call time of a round: the batch's wall-clock time / N (release to the last reply for
  `concurrent`). Per workload and tree: the median and p10/p90 over the rounds, the ratio
  of medians to `508e806` and to `c71570c`, and a paired sign test over the rounds (how many
  rounds the tree was slower, two-sided exact p-value, ties dropped).
- `overlap_on_baseline_50ms`: on `508e806` at 50 ms, the concurrent batch's time over the
  sequential one's; under 0.5 means the transport overlaps calls in-process. Concurrency
  conclusions are drawn only for transports that do.
- `slower_than_reference`: workloads where `18b7002` is more than 5% slower than `c71570c`
  with p < 0.05.
- `--profile WORKLOAD` runs that workload under `cProfile` on each tree and prints the
  functions with the most own time and the largest per-call differences against the first
  tree (`--top`, `--profile-out`). cProfile sees the event loop's thread only: time the
  server spends in `asyncio.to_thread` shows as the loop waiting in `epoll`.
