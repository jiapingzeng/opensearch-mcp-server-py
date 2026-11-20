# Behaviour changes: the MCPServer migration and the fixes before it

This file records every behaviour change made by the work on upstream issue #34, "Switch
from low-level MCP SDK to FastMCP" (implemented in this repository on the SDK's
`MCPServer`), and by the bug fixes made before it, so that each can become an upstream pull
request or issue later.

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

## Known, unchanged

These quirks are pinned by the characterization tests and kept as they are through the
migration by the user's choice (2026-11-19). Each is a candidate for an upstream issue.

### Security-relevant

- **Fix right after the migration.** A missing AWS profile signs with other credentials:
  in multi-cluster mode, a cluster whose `aws_profile` does not exist only warns and is
  signed with other credentials (`[default]`, `AWS_PROFILE` or env keys) (13g); an unknown
  `--profile` likewise only logs a warning and falls back to the default credential chain
  (11, finding 6).
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
