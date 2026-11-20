# Behaviour changes: the MCPServer migration and the fixes around it

This file records every behaviour change made by the work on upstream issue #34, "Switch
from low-level MCP SDK to FastMCP" (implemented in this repository on the SDK's
`MCPServer`), and by the bug fixes made before and after it, so that each can become an
upstream pull request or issue later.

The characterization goldens under `tests/characterization/golden/` pin today's behaviour
as a client sees it, so a golden diff is a behaviour change. The rule: a change that
modifies or deletes any golden changes both this file and `CHANGELOG.md`, and this file
names every golden the change touches, whether added, modified or deleted. A renamed golden
is its old path deleted and its new path added, so both paths are named. A golden is named
by its path relative to `tests/characterization/golden/` or by its full repo path, written
as a whole path, not inside a longer one. A change that only adds new goldens is new
characterization, not a behaviour change, and passes with no entry here. This command
checks it: `uv run --offline python scripts/check_behaviour_ledger.py`.

## Fixed before the migration

### Fix 22 and 22b: secrets in the missing-field validation error are masked

**What changed**: Before, a call that left out a required field got the error
`Missing required field(s): ...` followed by `Provided: {...}`, which echoed the caller's
raw arguments, `opensearch_password`, tokens and auth headers included. Now the value of
every secret-looking key (passwords, secrets, credentials, tokens, API, access and private
keys, authorization and cookie headers) shows as `'***'`, recursively in nested dicts and lists;
with 22b, hyphenated header names such as `X-Api-Key` are masked too. The rest of the
message, the non-secret values included, is unchanged.

**Why**: Bug fix: the error text leaked credentials to whatever reads tool results. Found
by the call_tool characterization (06) and the MCPServer research (14).

**Who is affected**: Any client whose tool errors reach logs or a model. No config or
client action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_tool_params_redaction.py`. Golden
changed:
- `call_tool_core/errors/missing_field_echoes_password.json`

**Upstream**: Not opened yet: the user decides.

### Fix 23: the OpenAPI start-up error goes to the log, not stdout

**What changed**: Before, when the OpenAPI spec fetch failed at start-up, the server
printed `Error generating tools: <error>` to stdout; on the stdio transport stdout is the
JSON-RPC channel, so the client read a line that is not JSON-RPC (before the first response
when stdout is unbuffered, after the last one when it is buffered). Now the same text is
logged at ERROR level (stderr or the configured log), stdout carries only JSON-RPC, and
start-up continues with the static tools as before.

**Why**: Bug fix: a stray stdout write corrupts the stdio JSON-RPC stream. Found by the
MCPServer research (14), pinned by the transport characterization (09). A follow-up
(`46c5b94`) only updated the characterization test's comments to match.

**Who is affected**: stdio clients when the OpenAPI spec fetch fails. No config or client
action is needed.

**Tests and goldens**: Unit test in `tests/tools/test_tool_generator.py`. Goldens changed:
- `transports/stdio/openapi_fetch_failing_buffered.json`
- `transports/stdio/openapi_fetch_failing_unbuffered.json`

**Upstream**: Not opened yet: the user decides.

### Fix 24: generated tools no longer send omitted query parameters as `None`

**What changed**: Before, a tool generated from the OpenAPI spec (such as `CountTool`,
`ClusterHealthTool`, `ExplainTool`, `MsearchTool`) sent every optional query parameter the
caller left out as the literal string `None`, for example `/_count?allow_no_indices=None`.
Now omitted parameters are not sent at all; a parameter the caller sets is sent with its
value, and `False`, `0` and empty values are still sent.

**Why**: Bug fix: a real cluster may reject `None` as a parameter value. Found by the
call_tool characterization (06), finding (1).

**Who is affected**: Anyone calling the generated tools; calls that failed on a `None`
parameter now reach the cluster as intended. No config or client action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_tool_generator.py`. Goldens changed:
- `call_tool_core/generated/ClusterHealthTool/error_403.json`
- `call_tool_core/generated/ClusterHealthTool/with_index.json`
- `call_tool_core/generated/ClusterHealthTool/without_index.json`
- `call_tool_core/generated/CountTool/error_404.json`
- `call_tool_core/generated/CountTool/with_index_and_query_param.json`
- `call_tool_core/generated/CountTool/without_index.json`
- `call_tool_core/generated/ExplainTool/error_404.json`
- `call_tool_core/generated/ExplainTool/success.json`
- `call_tool_core/generated/MsearchTool/error_400.json`
- `call_tool_core/generated/MsearchTool/json_array.json`
- `call_tool_core/generated/MsearchTool/ndjson.json`
- `call_tool_core/generated/MsearchTool/string_holding_array.json`
- `call_tool_core/generated/MsearchTool/with_index.json`
- `gaps_tool_results/generated/endpoint_fallback/ExplainTool_empty_id.json`
- `gaps_tool_results/generated/endpoint_fallback/ExplainTool_empty_index.json`
- `gaps_tool_results/generated/endpoint_fallback/ExplainTool_missing_id.json`
- `gaps_tool_results/generated/whitespace_body/CountTool.json`
- `gaps_tool_results/generated/whitespace_body/ExplainTool.json`

**Upstream**: Not opened yet: the user decides.

### Fix 25: the generic API tool sends the body on GET

**What changed**: The generic API tool documents its `body` for GET, POST and PUT, but
before it dropped the body on GET, so a `GET /_search` with a query ran as match-all. Now
GET sends the body (a dict or list JSON-encoded, a string as it is); DELETE and HEAD still
send none.

**Why**: Bug fix: the tool did not do what its description says, and searches silently
returned unfiltered hits. Found by the call_tool characterization (06), finding (2).

**Who is affected**: Clients sending GET requests with a body through the generic API tool;
they now get the filtered result they asked for. No config or client action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_generic_tool.py`. Golden changed
(its case id is historical: it now pins the body being sent):
- `call_tool_core/generic_api/get_body_dropped.json`

**Upstream**: Not opened yet: the user decides.

### Fix 26: a connection failure is sent once per transport try, not twice

**What changed**: Before, when a request failed at the connection level (refused, reset,
timed out, or a certificate fingerprint mismatch), `BufferedAsyncHttpConnection` sent it
again through opensearch-py's own connection, so each of the transport's four tries reached
the host twice: eight requests instead of four. Now such a failure raises the error
opensearch-py's connection raises for it (`ConnectionError`, `ConnectionTimeout` or
`SSLError`) and the transport retries it as usual. The error the tool returns and the
`opensearch_request` error log event are unchanged; the "Streaming request failed (...),
falling back to parent implementation" warning is no longer logged for these failures. Any
other unexpected error, and any failure before the request is built, still falls back as
before.

**Why**: Bug fix: every connection failure doubled the requests sent to an unreachable or
overloaded cluster. Found by the call_tool characterization (06).

**Who is affected**: Deployments whose cluster is unreachable, resets connections or times
out; the cluster now gets half the requests. No config or client action is needed.

**Tests and goldens**: Unit tests in `tests/opensearch/test_response_size_limiting.py`.
Golden changed (eight `/_cat/indices` requests become four):
- `call_tool_core/errors/connection_failure.json`

The harness hook `fake_opensearch.fail_streaming` reached the fallback with a connection
error, which no longer falls back, so it now fails the streaming attempt with a
`RuntimeError`. Its goldens change only in the fallback warning's exception
(`ClientConnectionError: fake: connection reset during streaming` becomes
`RuntimeError: fake: unexpected failure during streaming`); this is a harness change, not a
server behaviour change:
- `gaps_fallback/limit_bytes_utf16_over.json`
- `gaps_fallback/limit_bytes_utf8_at_limit.json`
- `gaps_fallback/limit_other_bytearray_over.json`
- `gaps_fallback/limit_str_at_limit.json`
- `gaps_fallback/limit_str_over.json`
- `gaps_fallback/measured_bytes_utf16.json`
- `gaps_fallback/measured_other_bytearray.json`
- `gaps_fallback/measured_other_tuple.json`
- `gaps_fallback/measured_str.json`
- `harness_gap_hooks/fail_streaming_bytes.json`
- `harness_gap_hooks/fail_streaming_other.json`
- `harness_gap_hooks/fail_streaming_status_404.json`
- `harness_gap_hooks/fail_streaming_str.json`

**Upstream**: Not opened yet: the user decides.

### Fix 27: generated tools send the `timeout` argument to OpenSearch

**What changed**: Before, a tool generated from the OpenAPI spec (such as
`ClusterHealthTool`) called with `timeout`, OpenSearch's own wait timeout such as `5s`,
never sent it: opensearch-py's transport took `timeout` out of the query parameters and
used it as the HTTP client timeout, so `GET /_cluster/health` went out without
`?timeout=5s`, and against a real cluster the string `5s` as a client timeout failed the
call. Now `timeout` is sent as a query parameter (`/_cluster/health?timeout=5s`) and the
HTTP client timeout stays the one the client is configured with (`OPENSEARCH_TIMEOUT` or
its default, for example); the golden below shows `timeout=5s` now reaching OpenSearch. A
call without `timeout` sends the same request as before.

**Why**: Bug fix: the documented `timeout` parameter did not reach OpenSearch and broke the
call. Found by the call_tool characterization review (06), finding (a).

**Who is affected**: Clients passing `timeout` to a generated tool; their calls now reach
the cluster with the wait timeout they asked for. No config or client action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_tool_generator.py`. Golden changed
(the `/_cluster/health` request's `params` go from `{}` to `{"timeout": "5s"}`):
- `call_tool_core/generated/ClusterHealthTool/with_timeout.json`

**Upstream**: Not opened yet: the user decides.

### Fix 28: the memory index is created on a managed domain

**What changed**: Before, on an Amazon OpenSearch Service managed domain, the memory tools'
auto-create of the memory index (`_ensure_memory_index`, on a `SaveMemoryTool` call while
the index does not exist) called the boto3 `opensearch` client's `create_index` with
`domainName`, `indexName` and `indexSchema` (the schema as a JSON string), names the API
does not have: botocore rejected the call before signing or sending it, so the index was
never created and the save failed with `Error saving memory: Parameter validation failed:
...`. Now the call sends `DomainName`, `IndexName` and `IndexSchema` (the schema document),
reaches the service, and the memory is saved; an answer that the index already exists
still counts as success, and any other error still fails the save. The serverless
(`opensearchserverless`) call is unchanged.

**Why**: Bug fix: the memory index was never auto-created on a managed domain. Found by the
memory and skills characterization (08), finding (13).

**Who is affected**: Users of the memory tools (`MEMORY_TOOLS_ENABLED=true`) on a managed
domain whose memory index does not exist yet; the first save now creates it. The call now
reaches AWS, so it needs AWS credentials allowed to create the index on the domain (without
credentials the save fails with `Unable to locate credentials`). No other config or client
action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_memory_tools.py` (a real boto3
client with botocore's `Stubber`). Golden changed (the error text goes from `Parameter
validation failed: ...` to `Unable to locate credentials`: the harness has no AWS
credentials, so the now valid call fails at signing, still before anything is sent):
- `call_tool_memory_skills/SaveMemoryTool/index_auto_create_domain.json`

Golden deleted (it pinned the rejection, the same for every reply):
- `gaps_memory_index/managed_domain_create_index_rejected.json`

Goldens added (one per stub reply: the stub receives one `CreateIndex` call with
`DomainName`, `IndexName` and `IndexSchema`, and the memory is indexed):
- `gaps_memory_index/managed_domain_create_index_created.json`
- `gaps_memory_index/managed_domain_create_index_already_exists.json`
- `gaps_memory_index/managed_domain_create_index_already_exists_message.json`

The module docstring of `tests/characterization/test_gaps_memory_index.py` and the
comment above the `index_auto_create_domain` case in
`tests/characterization/test_call_tool_memory_skills.py` still describe the rejection; they
are left as they are for a follow-up.

**Upstream**: Not opened yet: the user decides.

### Agentic memory tools: a call uses the configured container id

**What changed**: Before, a `memory_container_id` set in the config file's `agentic_memory`
section showed in `tools/list` as the schema default of the agentic memory tools (and the
field was no longer required), but a call that left the field out, relying on that default,
failed with `Missing required field: 'memory_container_id'` without reaching OpenSearch:
at call time the args model read only `OPENSEARCH_MEMORY_CONTAINER_ID`. Now the call uses
the id the server resolved at start-up and advertised (config file, then
`OPENSEARCH_MEMORY_CONTAINER_ID`) and reaches OpenSearch with it; with both set, such a
call now uses the config file's id, the one `tools/list` shows, where it used the env
variable's before. An explicit `memory_container_id` from the caller still wins, the env
variable alone still works, with neither the error is unchanged, and `tools/list` is
unchanged.

**Why**: Bug fix: the advertised default did not work. Found by the memory and skills
characterization (08), finding (14).

**Who is affected**: Users who set `agentic_memory.memory_container_id` in the config file
and call the agentic memory tools without `memory_container_id`; those calls now succeed
(or, with a different `OPENSEARCH_MEMORY_CONTAINER_ID` also set, use the config file's id).
No config or client action is needed.

**Tests and goldens**: Unit tests in `tests/tools/test_config.py`. Golden changed (the call
now sends `GET /` and the `GET` of the working memory in container `HudqiJkB1SltqOcZusVU`,
the configured id, to the fake and returns the memory instead of the missing-field error):
- `call_tool_memory_skills/GetAgenticMemoryTool/container_id_from_config_file.json`

The comment above `test_agentic_memory_container_id_default_from_config_file` in
`tests/characterization/test_call_tool_memory_skills.py` still describes the failure; it
is left as it is for a follow-up.

**Upstream**: Not opened yet: the user decides.

## Migration steps (issue #34)

Each migration step adds one line here.

| Step | Spec | Visible change | Characterization |
|------|------|----------------|------------------|
| shared server factory (still the low-level Server) | `15-mcpserver-server-factory` | None | 760 passed, 0 skipped, every golden unchanged |
| OpenSearchMCPServer on MCPServer (tools, call_tool, capabilities, request context) | `16-mcpserver-tool-registration` | None | 760 passed, 0 skipped, every golden unchanged |
| tools/list keeps duplicate listed names (fix to step 16) | `16b-mcpserver-duplicate-tool-names` | None (restores the c71570c listing) | 766 passed, 0 skipped, every existing golden unchanged |
| stdio transport on MCPServer | `17-mcpserver-stdio-transport` | None | 766 passed, 0 skipped, every golden unchanged |
| SSE and streamable HTTP on MCPServer | `18-mcpserver-streaming-transports` | None | 766 passed, 0 skipped, every golden unchanged |
| request context and auth headers on MCPServer, verification and cleanup | `19-mcpserver-request-context-auth` | None | 766 passed, 0 skipped, every golden unchanged |
| Remove the remaining low-level Server path | `20-remove-lowlevel-server-path` | None | 766 passed, 0 skipped, every golden unchanged |

Together the steps made no visible change: from step 15 to step 20 no golden was modified
or deleted, and the only goldens added are 16b's six, which pin the duplicate-name listing
as c71570c had it.

## Fixed after the migration

### A missing AWS profile fails the call instead of signing with other credentials

**What changed**: Before, when the boto3 session for a named AWS profile could not be
created (boto3's `ProfileNotFound`), the client setup logged `Failed to create boto3
session with profile '<name>': ...` as a warning and went on with a bare session, so the
request was signed with whatever the default credential chain found (`[default]`,
`AWS_PROFILE` or the env keys) instead of the profile that was asked for. Now it logs the
same warning and fails: the tool call returns `is_error` with `Failed to initialize single
mode client: The config profile (<name>) could not be found` (in multi mode `Failed to
initialize multi mode client for cluster "<url>": ...`), and no client is built, so
nothing is sent. This covers an unknown `--profile`, a cluster `profile` in the cluster
config that does not exist, and a per-call `aws_profile` with no per-call URL. A per-call
`aws_profile` with a per-call URL already failed this way (`requested profile`) and is
unchanged, and so is a missing `AWS_PROFILE` alone, which already failed with the same
text. With no profile asked for, the default credential chain is used as before.

**Why**: Security fix, fail closed: a request is never signed with credentials other than
the configured ones, which may be broader than the profile chosen. Found by the auth
characterization (11), finding 6 (unknown `--profile`), and the AWS region gaps (13g, a
cluster whose profile does not exist); kept through the migration by the user's choice
(2026-11-19) and fixed right after it.

**Who is affected**: Deployments that name a profile which does not exist (a typo in
`--profile`, a cluster's `profile`, or a per-call `aws_profile`) and relied, knowingly or
not, on the default credentials signing instead; their calls now fail until the profile
name is fixed or removed. The session is created before the auth method is chosen, so
this holds even when the call would authenticate another way (basic auth, no auth, header
credentials). No action is needed where every named profile exists.

**Tests and goldens**: Unit tests in `tests/opensearch/test_client.py`
(`TestMissingProfileFailsClosed`: both cases fail with the profile named and no client
built, and a valid profile and the no-profile default are unchanged). Goldens changed (the
call now fails and sends nothing, where it was signed by `profile:char-west` and by `env`):
- `gaps_aws_region/multi/cluster_profile_missing_aws_profile_env.json`
- `gaps_aws_region/multi/cluster_profile_missing_env_keys.json`

Goldens deleted (their names said the request fell back, signed by `env` and by
`profile:default`):
- `auth/single/sigv4_profile_cli_unknown_falls_back.json`
- `gaps_aws_region/multi/cluster_profile_missing_default_identity.json`

Goldens added (the same cases, now failing with nothing sent; the warnings are unchanged):
- `auth/single/sigv4_profile_cli_unknown_fails.json`
- `gaps_aws_region/multi/cluster_profile_missing_default_profile.json`

The comment above the `single/default_session_fails_missing_aws_profile` case in
`tests/characterization/test_gaps_aws_region.py` still says the client setup falls back to
a bare session (it now re-raises the profile's own error, with the same text, so its
golden is unchanged); it is left as it is for a follow-up. The memory tools' own session
helper (`_get_boto3_session` in `src/tools/memory_tools.py`) still falls back to a bare
session for a missing profile; it is outside this fix.

**Upstream**: Not opened yet: the user decides.

## Performance: tools/call timing after step 16

The host parity run at 18b7002 (2026-11-20) found the single-client concurrency integration
test (5 concurrent `ListIndexTool` calls in one streamable-HTTP session) about 19% slower
than on 508e806 (8 alternating pairs, sums 7.95 vs 6.70 s). Only steps 16 and 16b changed
`src` between c71570c (within ±3%) and 18b7002, so `tools/call` was timed in-process on the
three trees (`benchmarks/README.md`, task `benchmark-tools-call-rwpo`).

**Method.** `scripts/bench_tools_call.py` extracts each revision with `git archive` and runs
`benchmarks/tools_call_case.py` in that tree's own characterization harness: the server is
built by `streaming_server.create_mcp_server`, OpenSearch is `fake_opensearch`, and the
driver checks that every server module came from that tree. 16 workloads, each in its own
MCP session after 2 warm-up calls: N = 5 or 50 `ListIndexTool` calls, sequential or
concurrent (released together, as the integration test does), with a fake OpenSearch
latency of 0 or 50 ms per request (each call makes 2 requests). Two transports: the SDK's
in-memory `Client` (`memory`) and the SDK's streamable-HTTP `ClientSession` through
`httpx2.ASGITransport` into the `MCPStarletteApp(server, stateless=True)` app the harness's
`http_app` runs (`http`, the integration test's transport). 30 rounds, each running every
tree once in a new process, the trees' order rotated each round. A round's value is the
batch's wall-clock time / N; the sign test pairs 18b7002 with c71570c round by round
(two-sided, exact, ties dropped).

**Host.** 32 CPUs, Python 3.10.20, mcp 2.0.0; load average 95.7 at the start and 42.7 at
the end (per-round 1-minute load 15 to 129), a heavily shared host, which is why the
p10–p90 bands are wide. On 508e806 at 50 ms both transports overlap concurrent calls
(concurrent / sequential time: memory 0.34 and 0.28, http 0.37 and 0.28 for N = 5 and 50),
so both support concurrency conclusions. The full results are in
`benchmarks/results/tools_call.json`.

Per-call time in ms, median [p10–p90] over 30 rounds; "paired" is the median of the
per-round 18b7002/c71570c ratios; "slower" is the number of rounds 18b7002 was slower.

| Workload | 508e806 | c71570c | 18b7002 | 18b7002/c71570c (medians) | paired | slower | p |
|---|---|---|---|---|---|---|---|
| `memory-sequential-n5-0ms` | 23.6 [18.6–30.9] | 26.1 [18.7–32.1] | 22.6 [18.8–29.1] | 0.867 | 0.951 | 14/30 | 0.86 |
| `memory-concurrent-n5-0ms` | 25.7 [19.9–36.1] | 26.6 [20.1–39.1] | 22.2 [20.2–44.4] | 0.833 | 0.996 | 13/30 | 0.58 |
| `memory-sequential-n50-0ms` | 29.0 [21.9–43.1] | 31.0 [22.1–42.4] | 27.8 [21.9–39.4] | 0.896 | 0.983 | 13/30 | 0.58 |
| `memory-concurrent-n50-0ms` | 37.8 [25.6–65.4] | 38.2 [29.1–54.7] | 33.6 [27.7–50.6] | 0.879 | 0.950 | 10/30 | 0.099 |
| `memory-sequential-n5-50ms` | 138.1 [124.2–161.4] | 133.7 [122.7–188.7] | 129.3 [122.3–164.8] | 0.967 | 0.995 | 13/30 | 0.58 |
| `memory-concurrent-n5-50ms` | 47.3 [38.4–103.4] | 48.8 [39.6–83.1] | 44.0 [38.8–104.6] | 0.901 | 0.947 | 12/30 | 0.36 |
| `memory-sequential-n50-50ms` | 134.0 [127.0–153.7] | 135.7 [125.0–168.3] | 135.1 [125.7–159.8] | 0.996 | 0.976 | 10/30 | 0.099 |
| `memory-concurrent-n50-50ms` | 37.6 [27.4–47.9] | 41.4 [30.0–58.0] | 40.3 [27.4–52.1] | 0.974 | 0.960 | 14/30 | 0.86 |
| `http-sequential-n5-0ms` | 30.6 [23.2–49.7] | 31.0 [23.8–61.9] | 32.7 [22.6–50.2] | 1.057 | 0.948 | 14/30 | 0.86 |
| `http-concurrent-n5-0ms` | 27.8 [22.5–56.5] | 28.5 [24.2–68.0] | 38.2 [23.0–61.0] | 1.341 | 1.056 | 17/30 | 0.58 |
| `http-sequential-n50-0ms` | 32.8 [26.0–50.2] | 33.4 [27.8–60.5] | 40.2 [26.2–55.0] | 1.201 | 1.050 | 18/30 | 0.36 |
| `http-concurrent-n50-0ms` | 37.9 [30.9–58.2] | 39.5 [33.0–65.7] | 42.5 [30.6–58.2] | 1.076 | 0.974 | 14/30 | 0.86 |
| `http-sequential-n5-50ms` | 130.6 [125.2–190.5] | 138.5 [128.1–164.8] | 139.6 [125.1–158.6] | 1.008 | 0.988 | 13/30 | 0.58 |
| `http-concurrent-n5-50ms` | 48.0 [41.4–77.6] | 48.9 [41.8–82.9] | 46.9 [41.0–87.2] | 0.958 | 0.954 | 12/30 | 0.36 |
| `http-sequential-n50-50ms` | 136.2 [128.0–158.1] | 145.0 [131.2–173.3] | 139.6 [127.7–175.1] | 0.963 | 0.976 | 13/30 | 0.58 |
| `http-concurrent-n50-50ms` | 38.3 [31.1–68.3] | 43.0 [33.2–58.8] | 41.7 [31.4–65.2] | 0.971 | 0.935 | 12/30 | 0.36 |

**Verdict: no detectable slowdown in the timings; "no slowdown" rests on the code path.**
Each piece of evidence supports a different claim.

- The timings found no detectable slowdown. No workload has 18b7002 more than 5% slower
  than c71570c with p < 0.05 and more rounds slower than faster, and no sign test reaches
  p < 0.05: the smallest p is 0.0987, for `memory-concurrent-n50-0ms` and
  `memory-sequential-n50-50ms`, where 18b7002 was faster in 20 of 30 rounds; every other
  p is 0.36 or more. 18b7002 was slower in 10 to 18 of 30 rounds, and the paired ratios
  run from 0.935 to 1.056. The `http` 0 ms workloads have the highest ratios of medians
  (1.341 and 1.201), inside wide bands, and the pairing (17/30 and 18/30 slower, p 0.58
  and 0.36) does not separate them from chance. The closest in-process analogue of the
  integration test is `http-concurrent-n5-50ms` (5 concurrent calls in one
  streamable-HTTP session): 0.958 by medians, 0.954 paired, 12/30 slower, p 0.36. But
  the run was at a host load around 95 on 32 CPUs, and its ratios of medians spread from
  0.83 to 1.34: these timings cannot resolve a 19% effect, so they neither show nor rule
  out the integration test's gap.
- "No slowdown" rests on the structural analysis below: step 16 adds one SDK `Context`
  per call, microseconds against 22 to 40 ms per call.
- The director's later A/B at c6360aa, at a lower host load, found +3%, within noise.

**What step 16 adds per call** (measured, too small to see above). A cProfile of
`memory-sequential-n50-0ms` on c71570c and 18b7002 differs in one place. It ran
`--only-revs --rev c71570c --rev 18b7002 --profile memory-sequential-n50-0ms --batches 2`:
2 timed batches of 50 calls after the warm-up, so 100 profiled calls per tree. On 18b7002
the SDK's `MCPServer._handle_call_tool` (`mcp/server/mcpserver/server.py`, the `MCPServer`
handler layer) wraps `OpenSearchMCPServer.call_tool` and builds one
`mcp.server.mcpserver.Context` per call (a pydantic `BaseModel.__init__` with private
attributes). Under the profiler that `__init__` (`mcp/server/mcpserver/context.py:71`)
took 41 µs per call cumulative in the step 16c run and 47 µs in a rerun of that command
(2026-11-20). Unprofiled, `--context-cost` (it times building that `Context` the way
`_handle_call_tool` does, 7 repeats of 50,000) gives 7.8 µs minimum and 9.4 µs median on
mcp 2.0.0 at load 25, against 22 to 40 ms per call: under 0.05%. Everything else is the
same code on both trees. The low-level `Server`'s runner, its OpenTelemetry middleware,
the params validation and the result serialization all run unchanged.
`RequestStateBoundary` is removed, and nothing locks or serializes concurrent calls in a
session. Per-call time on every tree is dominated by the server building an OpenSearch
client in `asyncio.to_thread` for each request (the version check and the call), not by
the SDK.

The integration test's 19% (about 150 ms on a 1 s concurrent batch) cannot come from
about 10 µs per call. Its run talks to a remote cluster over the network on a shared
host, so the difference is most likely cluster and network variance across the 8 pairs.

**Options.**

- Accept (recommended): nothing changes. A user sees no difference in `tools/call` latency
  or concurrency from step 16; the extra `Context` per call is about 10 µs.
- Fix: there is nothing to fix at this size. The only lever would be overriding
  `_handle_call_tool` in `src/mcp_server_opensearch/mcp_server.py` to skip the `Context`
  (passing the request straight to `request_context_var`). That is about 10 lines, but it
  overrides a private SDK method that would have to follow every SDK release, to save
  about 10 µs per call. If the integration test's gap persists, the next step is to rerun
  it with more alternating pairs against the same cluster, not to change the server.

### Streamable HTTP after step 18 (457a4bc)

Step 18 (457a4bc) moved SSE and streamable HTTP onto `OpenSearchMCPServer`'s own session
manager. At the director's request (2026-11-20), the `http` workloads were run on 457a4bc,
interleaved with 508e806 and 18b7002, with the fixed driver (10 rounds, the default 1
batch per session, 2026-11-20 05:26 to 05:37 UTC):

```sh
uv run --offline python scripts/bench_tools_call.py --only-revs \
  --rev 508e806 --rev 18b7002 --rev 457a4bc --candidate 457a4bc --reference 18b7002 \
  --transport http --rounds 10 --out benchmarks/results/tools_call_http_457a4bc.json
```

**Host.** 32 CPUs, Python 3.10.20, mcp 2.0.0; load average 33.1 at the start and 59.6 at
the end (per-round 1-minute load 30 to 107). On 508e806 at 50 ms the transport overlaps
concurrent calls (concurrent / sequential time 0.50 and 0.28 for N = 5 and 50; the N = 5
share is just under the 0.5 bar, so that row supports concurrency conclusions only
weakly). The results are in `benchmarks/results/tools_call_http_457a4bc.json`.

Per-call time in ms, median [p10–p90] over 10 rounds; "paired" is the median of the
per-round ratios; "slower" is the number of rounds 457a4bc was slower.

| Workload | 508e806 | 18b7002 | 457a4bc | 457a4bc/18b7002 (medians) | paired | slower | p | 457a4bc/508e806 (medians) | paired | slower | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `http-sequential-n5-0ms` | 26.4 [21.6–33.4] | 25.3 [20.9–30.3] | 28.2 [21.5–51.2] | 1.112 | 1.087 | 7/10 | 0.34 | 1.068 | 0.996 | 4/10 | 0.75 |
| `http-concurrent-n5-0ms` | 32.2 [23.3–35.8] | 26.3 [22.6–32.2] | 28.7 [23.2–53.0] | 1.090 | 1.144 | 6/10 | 0.75 | 0.892 | 0.981 | 4/10 | 0.75 |
| `http-sequential-n50-0ms` | 33.0 [28.2–41.1] | 29.1 [26.5–35.8] | 32.7 [25.7–55.1] | 1.123 | 1.108 | 5/10 | 1 | 0.990 | 0.950 | 4/10 | 0.75 |
| `http-concurrent-n50-0ms` | 48.2 [33.7–62.7] | 37.3 [33.7–46.7] | 40.4 [32.3–60.7] | 1.084 | 0.973 | 4/10 | 0.75 | 0.838 | 0.935 | 3/10 | 0.34 |
| `http-sequential-n5-50ms` | 153.6 [125.9–195.2] | 130.7 [126.6–155.1] | 134.3 [125.7–170.1] | 1.027 | 1.005 | 5/10 | 1 | 0.874 | 0.966 | 4/10 | 0.75 |
| `http-concurrent-n5-50ms` | 76.5 [40.4–217.1] | 46.4 [40.3–75.7] | 43.4 [40.5–65.7] | 0.937 | 1.010 | 5/10 | 1 | 0.568 | 0.772 | 3/10 | 0.34 |
| `http-sequential-n50-50ms` | 145.3 [130.9–159.8] | 137.4 [131.9–149.3] | 141.0 [135.1–170.0] | 1.027 | 1.070 | 7/10 | 0.34 | 0.971 | 1.036 | 6/10 | 0.75 |
| `http-concurrent-n50-50ms` | 41.4 [34.0–55.2] | 41.8 [31.0–65.4] | 50.9 [37.3–58.7] | 1.217 | 1.155 | 7/10 | 0.34 | 1.230 | 1.108 | 8/10 | 0.11 |

**Verdict: no detectable slowdown in the timings; they cannot rule out one of 10 to 20%.**
Each piece of evidence supports a different claim.

- The timings found no detectable slowdown. No workload has 457a4bc more than 5% slower
  than 18b7002 with p < 0.05 and more rounds slower than faster (`slower_than_reference`
  is empty), so no profile was run. Against 18b7002, 457a4bc was slower in 4 to 7 of 10
  rounds (p 0.34 to 1); against 508e806, in 3 to 8 of 10 (p 0.11 to 0.75).
- They are too coarse to show "no slowdown". Five of the eight ratios of medians against
  18b7002 are above 1.05 (1.084 to 1.217, the highest `http-concurrent-n50-50ms`: 1.155
  paired, 7/10 slower), inside wide bands. With 10 rounds the sign test reaches p < 0.05
  only when one tree wins at least 9 of the 10, so at this load (ratios of medians 0.57 to
  1.23 across both comparisons) a 10 to 20% effect stays unresolved either way. The
  run's noise shows in its control pair: 18b7002 was faster than 508e806 in 9 of 10
  rounds on `http-sequential-n5-0ms` and `http-sequential-n50-0ms` (p 0.021), which the
  30-round run did not show (p 0.58 for both); with 24 sign tests in the run, about 0.5
  results with a p that low are expected by chance, so two are not surprising. The
  closest analogue of the integration test, `http-concurrent-n5-50ms`, is 0.937 by
  medians and 1.010 paired against 18b7002.
- What "no slowdown" would rest on is the code path, not these timings: step 18 still
  builds one SDK `StreamableHTTPSessionManager` on the same low-level server, now through
  `MCPServer.streamable_http_app` instead of directly, with the same stateless, JSON,
  event-store and body-size settings. Its security settings are now explicit, with
  DNS-rebinding protection off as before (it was `None`, which the SDK treats as off), so
  its per-request check still returns at once. The per-call path is the same class with
  equivalent arguments; this is read from the code, not measured.

## Known, unchanged

These quirks are pinned by the characterization tests and kept as they are through the
migration by the user's choice (2026-11-19). Each is a candidate for an upstream issue.

### Security-relevant

- **Fix right after the migration.** With header auth, an `Authorization` header that
  cannot be used (an empty Bearer token, an unknown scheme, Basic without `:`, or a failed
  base64 decode) is dropped without a log record and the server falls back to its own
  credentials; after a failed base64 decode, header values read before `Authorization` stay
  in effect (13a).
- **Fix right after the migration.** Serverless memory index policy race: if the
  data-access policy is deleted between read and update, a new policy is created for the
  caller alone and the principals read before are dropped, silently narrowing access (13l).
- No DNS-rebinding protection on SSE and streamable HTTP: a request with
  `Host: example.com` or `Origin: http://evil.example` is served on `POST /mcp` and
  `GET /sse`. The migration keeps it off explicitly for parity; adding protection would be
  an opt-in setting, a later feature (09, finding 11).
- Serverless memory index: an existing data-access policy whose rule set lacks `Principal`
  is caught as a generic error, only warned about and left unchanged, and the index is
  created anyway (13l).
- Per-call connection override arguments are accepted and silently ignored in
  multi-cluster mode, where they are not in the tool schemas (11, finding 10).
- An `ftp://` OpenSearch URL is accepted and connects over plain http to port 9200 (13f).

### Other

- A config file whose `tool_filters` has no `settings` (or only a `tools` section, or a
  missing file) hides write tools as if `allow_write` were false, while the documented
  default is true; this is the safe side, and fixing it would expose write tools on
  existing deployments without an opt-in (05, finding 1).
- `--config` pointing at a missing file silently ignores the env filters, and single mode
  starts anyway (05, finding 2).
- CLI values are parsed as YAML, so `--tool.X.description=123` becomes a number and
  `tools/list` then fails with -32602 (05, finding 4).
- `--tool.X.args.index=...` without `.description`, and `--tool.T.args.A=plain`, crash
  startup with `AttributeError` (05, finding 5; 13j).
- The region is never read from an `*.es.amazonaws.com` or `*.aoss.amazonaws.com` URL, so
  AWS auth without a configured region fails (11, finding 7).
- A bearer `Authorization` header replaces the client's headers, so the server's
  `user-agent` is lost (11, finding 8).
- A per-call `aws_profile` supplies the profile's credentials but not its region; a profile
  without a region is signed with another profile's region (11, finding 9; 13g).
- A missing `AWS_PROFILE` ends in an unhandled `ConfigurationError` even with `AWS_REGION`
  set (13g).
- An over-long `x-mcp-client-name` is cut to 64 characters before it is checked, so an
  invalid character past position 64 is dropped and the name accepted (09, finding 12).
- OpenSearch error texts show the raw JSON body (`NotFoundError(404, '<raw JSON>')`)
  instead of the error type and reason (08, finding 15).
- The `required` list of each tool generated from the OpenAPI spec is built from a set, so
  its order in `tools/list` changes with `PYTHONHASHSEED`; JSON Schema gives the order no
  meaning (05, finding 16).
- A soft tool error (returned through `log_tool_error`) gets a `tool_execution` log record
  with status `error` but no `error_type`; the `tool_error` record carries
  `exception_type` (06 review, finding b; the user chose to keep it).
- Connection URLs: a username without a password fails with a raw `TypeError` text wrapped
  as `AuthenticationError`; userinfo with no host passes the URL check, then fails with a
  raw `TypeError`; `host:port` without a scheme is rejected as having no host (13f).
- Config: an empty `--tool.T.display_name=` lists the tool under `""`;
  `--tool.T.args.A.description=null` sets a `null` description while the same null from
  the config file is refused at startup; an empty `args:` is warned about as not a mapping;
  the CLI-shaped `args: {A: {description: x}}` in the file is rejected as not a string; an
  invalid regex abandons filtering after the write filter, so every tool that survived it
  is listed (13j).
- Tool results: a non-UTF-8 body becomes the Python repr of the bytes; a dict error value
  shows as its Python repr; a whitespace-only `body` is dropped even where the schema
  requires it (ExplainTool); ExplainTool's required `index` and `id` are not enforced, so a
  left-out `id` goes into the path as `None` (`/logs/_explain/None`); a path parameter
  named `body` becomes the request body; `select_endpoint`'s fallback can only return the
  first endpoint (13h).
- On the connection fallback path, a `bytearray` body is measured by the length of its
  repr, so a body that would pass as `bytes` can be refused by the size limit (13i).
- With `--profile`, the region log line says "Using region from AWS_PROFILE '<name>'"
  naming the `--profile` profile, even when `AWS_PROFILE` names another (13d; cosmetic).
- The legacy install-hook check only matches a single-quoted `b64decode('...')`, so a
  double-quoted legacy hook gets the memory hooks added next to it (13k).
