![OpenSearch logo](https://github.com/opensearch-project/opensearch-py/raw/main/OpenSearch.svg)

# OpenSearch MCP Server Python Developer Guide

## Table of Contents
- [Overview](#overview)
- [Development Setup](#development-setup)
- [Local Testing and Development Workflow](#local-testing-and-development-workflow)
- [Running the Server](#running-the-server)
- [Foolproof Build Testing Workflow](#foolproof-build-testing-workflow)
- [Development Configuration](#development-configuration)
- [Managing Dependencies](#managing-dependencies)
- [Server Architecture](#server-architecture)
- [Adding Custom Tools](#adding-custom-tools)
- [Testing](#testing)

## Overview

This guide is for developers who want to contribute to the OpenSearch MCP Server Python project. It covers local development setup, project structure, and how to add new tools to the server.

## Development Setup

### 1. Clone the Repository

```bash
git clone git@github.com:opensearch-project/opensearch-mcp-server-py.git
cd opensearch-mcp-server-py
```

### 2. Set Up Development Environment

Follow the [Local Testing and Development Workflow](#local-testing-and-development-workflow) section below for complete setup instructions.

## Local Testing and Development Workflow

### Foolproof Local Testing Workflow

**All commands must be run from the project root directory** (where `pyproject.toml` is located).

#### First Time Setup

```bash
# From project root
cd /path/to/opensearch-mcp-server-py

# 1. Update lock file (ensures uv.lock matches pyproject.toml)
uv lock

# 2. Create venv and install everything (dependencies + package in editable mode)
uv sync

# 3. Activate the venv
source .venv/bin/activate

# 4. Run the server
python -m mcp_server_opensearch --transport stream --port 9900 --debug
```

#### Daily Testing (After Setup)

```bash
# From project root
source .venv/bin/activate
python -m mcp_server_opensearch --transport stream --port 9900 --debug
```

#### After Changing Dependencies in pyproject.toml

```bash
# From project root
# 1. Update lock file
uv lock

# 2. Sync environment (updates dependencies and package)
uv sync

# 3. Run the server
source .venv/bin/activate
python -m mcp_server_opensearch --transport stream --port 9900 --debug
```

### Common Questions

**Q: Do I need to run `uv lock` before testing?**  
A: Only if you changed `pyproject.toml`. For code changes in `src/`, just activate venv and run.

**Q: What does `uv sync` do?**  
A: Creates `.venv`, installs all dependencies from `uv.lock`, and installs your package in editable mode. Everything in one command.

**Q: Where do I run commands from?**  
A: Project root (where `pyproject.toml` is located).

**Q: Do I need to rebuild after code changes?**  
A: No. `uv sync` installs the package in editable mode, so changes in `src/` are picked up automatically.

**Q: Can I use a different venv name?**  
A: Yes, but `.venv` is the default. Use `uv sync --python <path>` or set `UV_PROJECT_ENVIRONMENT` environment variable.

**Q: How do I clean/reset my venv and start fresh?**  
A: Delete `.venv` and run `uv sync` again:
```bash
# From project root
rm -rf .venv
uv sync
source .venv/bin/activate
python -m mcp_server_opensearch --transport stream --port 9900 --debug
```
This gives you a completely fresh environment with all dependencies reinstalled.

## Running the Server

After setting up your environment (see above), you can run the server from the project root:

```bash
# Activate venv (if not already activated)
source .venv/bin/activate

# Run streaming server (recommended for testing)
python -m mcp_server_opensearch --transport stream --port 9900 --debug

# Run stdio server (default)
python -m mcp_server_opensearch

# Run with custom AWS profile
python -m mcp_server_opensearch --profile my-profile

# Run in multi mode with config file
python -m mcp_server_opensearch --mode multi --config config/clusters.yml --transport stream
```

## Foolproof Build Testing Workflow

Test the package exactly as end users will install it (before releasing to PyPI).

**All commands must be run from the project root directory.**

#### Complete Build and Test Workflow

```bash
# From project root
cd /path/to/opensearch-mcp-server-py

# Step 1: Build the package
python3 -m venv .venv-clean
source .venv-clean/bin/activate
pip install --upgrade build twine
rm -rf dist/ build/
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
python -m build

# Step 2: Test the built package in a fresh environment
python3 -m venv .venv-test-install
source .venv-test-install/bin/activate
pip install dist/opensearch_mcp_server_py-*.whl
opensearch-mcp-server-py --transport stream --port 9900 --debug
```

#### Common Questions

**Q: Why do I need separate venvs for build and test?**  
A: To ensure a clean build without contamination from your dev environment, and a fresh test that matches end-user installation.

**Q: Can I reuse the build/test venvs?**  
A: Yes, but delete and recreate them for truly clean builds before releases.

**Q: What files are created?**  
A: `dist/opensearch_mcp_server_py-X.X.X-py3-none-any.whl` and `.tar.gz` - these are what get published to PyPI.

## Development Configuration

### MCP Configuration for Development

Create an MCP configuration file for your AI agent to connect to your development server:

#### For Q Developer CLI (`~/.aws/amazonq/mcp.json`):
```json
{
    "mcpServers": {
        "opensearch-mcp-server": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/your/clone/opensearch-mcp-server-py",
                "run",
                "--",
                "python",
                "-m",
                "mcp_server_opensearch"
            ],
            "env": {
                "OPENSEARCH_URL": "<your_opensearch_domain_url>",
                "OPENSEARCH_USERNAME": "<your_opensearch_domain_username>",
                "OPENSEARCH_PASSWORD": "<your_opensearch_domain_password>"
            }
        }
    }
}
```

#### For Claude Desktop (`claude_desktop_config.json`):
```json
{
    "mcpServers": {
        "opensearch-mcp-server": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/your/clone/opensearch-mcp-server-py",
                "run",
                "--",
                "python",
                "-m",
                "mcp_server_opensearch"
            ],
            "env": {
                "OPENSEARCH_URL": "<your_opensearch_domain_url>",
                "OPENSEARCH_USERNAME": "<your_username>",
                "OPENSEARCH_PASSWORD": "<your_password>"
            }
        }
    }
}
```

## Managing Dependencies

### Adding New Dependencies
```bash
# Add a new package (automatically updates pyproject.toml, uv.lock, and installs)
uv add <package-name>

# Add with specific version
uv add <package-name>==1.2.3

# Add as development dependency
uv add --dev <package-name>
```

### Updating Dependencies
```bash
# After manually editing pyproject.toml, update lock file and sync
uv lock
uv sync

# Update all dependencies to latest versions
uv lock --upgrade
uv sync
```

## Server Architecture

The server runs on the MCP Python SDK's `MCPServer` (`mcp.server.mcpserver`, what issue #34
calls FastMCP), through one subclass, `OpenSearchMCPServer` in
`src/mcp_server_opensearch/mcp_server.py`. Its tools come from `TOOL_REGISTRY`, not from SDK
decorators.

### Startup and the server factory

`main()` (`src/mcp_server_opensearch/__init__.py`) parses the CLI, configures logging and
starts one transport: `stdio_server.serve()` (`--transport stdio`, the default) or
`streaming_server.serve()` (`--transport stream`). Both build the server with the server
factory, `src/mcp_server_opensearch/server_factory.py`:

1. `prepare_enabled_tools(mode, profile, config_file_path, cli_tool_overrides)`:
   - sets the global mode, AWS profile and config file path (`global_state.py`), and in
     multi mode loads the clusters from the config file;
   - `generate_tools_from_openapi()` (`src/tools/tool_generator.py`) fetches the OpenSearch
     API specification and adds the generated tools (`MsearchTool`, `ExplainTool`,
     `CountTool`, `ClusterHealthTool`) to `TOOL_REGISTRY`; if the fetch fails it logs the
     error and the server starts with the static tools;
   - `apply_custom_tool_config()` (`src/tools/config.py`) returns a copy of the registry
     with the custom names, descriptions and argument descriptions from the config file and
     the `--tool.*` CLI overrides, and the agentic memory container id as the default;
   - `get_tools()` (`src/tools/tool_filter.py`) returns the enabled tools and adapts their
     schemas to the mode: in single mode the category and tool filters, write protection
     and the OpenSearch version range (the server asks the cluster for its version); in
     multi mode every tool except the memory tools.
2. `build_server(enabled_tools)` returns `OpenSearchMCPServer(enabled_tools)`.

The static tools (`TOOL_REGISTRY` in `src/tools/tools.py`, which also merges the skills,
agentic memory and memory tool registries) and the generated tools take the same path:
nothing registers a tool with the SDK.

### Requests

- `initialize`: the server advertises the `tools` capability and no `prompts` or
  `resources` capability. Its instructions come from `server_instructions.py` (the per-call
  connection parameters in single mode when dynamic connections are on, the skills tools
  when they are enabled), else there are none.
- `tools/list`: `OpenSearchMCPServer.list_tools()` returns one `mcp.types.Tool` per enabled
  tool, in order. The name is the `display_name`, else the registry key; the
  `description` and `input_schema` are the entry's own, with no `outputSchema`.
- `tools/call`: `OpenSearchMCPServer.call_tool()` sets `request_context_var`
  (`client_context.py`) to the transport's HTTP request (`None` over stdio), which header
  auth in `src/opensearch/client.py` reads, and calls `tool_executor.execute_tool()`. That
  resolves the first enabled tool with the name, validates the arguments with the tool's
  `args_model`, runs the tool function and logs a `tool_execution` event. A result whose
  first item has `is_error: True` (from `log_tool_error()`) is returned as a
  `CallToolResult` with `is_error` set; any other exception (an unknown tool, invalid
  arguments, an exception the tool raised) becomes an `is_error` result holding its text;
  an `MCPError` is answered as a JSON-RPC error.

### Transports

- **stdio** (`stdio_server.py`): `serve()` opens the SDK's `stdio_server()` streams and calls
  `server.run_streams(reader, writer)`, which serves the one connection until stdin closes.
- **SSE and streamable HTTP** (`streaming_server.py`): `MCPStarletteApp` builds one Starlette
  app, served by uvicorn on `--host` and `--port` (default `127.0.0.1:9900`):
  - `/mcp` (bare, as a `Route`, so it is not redirected to `/mcp/`) and the paths under it:
    stateless streamable HTTP, through the session manager `MCPServer.streamable_http_app()`
    creates;
  - `GET /sse` and `POST /messages/`: SSE, the two routes taken from `MCPServer.sse_app()`;
  - `GET /health`: answers `OK`.

  `ClientNameMiddleware` (`client_context.py`) reads the optional `X-MCP-Client-Name` header
  into `client_name_var`, which the `tool_execution` log events carry.

Both transports start the memory monitor (`logging_config.py`) for the life of the server.

### Deliberate departures from a plain `MCPServer`

Each of these keeps a behaviour the server had on the low-level `Server`, and each is pinned
by the characterization tests and commented in the code. Changing one is a behaviour change
(see [Characterization tests](#characterization-tests)).

- **No `RequestStateBoundary`.** `MCPServer` adds a `RequestStateBoundary` middleware, which
  answers a `tools/call` carrying any `requestState` with -32602 "Invalid or expired
  requestState", where the low-level `Server` ran the tool. This server never returns
  `input_required`, so there is no state to seal: `OpenSearchMCPServer.__init__` drops it.
- **Tools are not registered with the SDK, and nothing is validated at startup.** No tool
  goes into `MCPServer`'s `ToolManager` (no `@mcp.tool()`, no `add_tool()`):
  `list_tools()` and `call_tool()` are overridden and read the enabled tools. The schemas
  are the registry's as they are, not derived from a function signature; the arguments are
  validated by the tool's `args_model` in `execute_tool()`. `list_tools()` builds each
  `Tool` when `tools/list` is answered, so a field of the wrong type, such as a description
  the CLI parsed as a number (`--tool.X.description=123`), still starts the server, and
  `tools/list` then fails with -32602.
- **`tools/list` shows one entry per enabled tool, even when names repeat.** The SDK's
  `ToolManager` keys tools by name: of two enabled tools with the same listed name (a
  display name equal to another tool's key, or two empty display names) it would keep one
  and log "Tool already exists". This server lists both, in the enabled-tools order, as the
  low-level `Server` did; `tools/call` runs the first (`test_gaps_duplicate_names.py`).
- **The one low-level run call, for stdio parity.** `run_streams()` calls the low-level
  server's `run(..., raise_exceptions=True)` instead of the public `run_stdio_async()`,
  which runs without `raise_exceptions`. A non-MCP exception inside an SDK handler then ends
  the process, as before, instead of becoming a JSON-RPC error.
- **Unserved methods are taken out.** `MCPServer.__init__` registers prompt, resource and
  subscription handlers, which would add the `prompts` and `resources` capabilities. The
  SDK has no public way to unregister a handler, so `__init__` removes `prompts/*`,
  `resources/*` and `subscriptions/listen` from the low-level server's handler table; they
  answer -32601 "Method not found", as before.
- **The HTTP app is built from `MCPServer`'s parts.** `MCPServer`'s own Starlette apps are
  not mounted (they would answer `/mcp/` with a 307 redirect and have no `/health`); only
  their session manager and SSE routes go into `MCPStarletteApp.create_app()`.
  DNS-rebinding protection stays off on both transports, as before: it is turned off
  explicitly, since `MCPServer`'s apps turn it on by themselves for host `127.0.0.1`.

`mcp_server.py` is the only module that reaches `MCPServer`'s low-level server, for the run
call and the handler table above.

## Adding Custom Tools

Every tool has 4 pieces. Follow the existing patterns exactly — read the referenced files to see how current tools are structured.

A new tool needs no server change: `OpenSearchMCPServer` lists and calls the tools `prepare_enabled_tools()` returns (see [Server Architecture](#server-architecture)), so a `TOOL_REGISTRY` entry in an enabled category is all it takes. Do not register tools with `@mcp.tool()` or `MCPServer.add_tool()`: such a tool bypasses the tool filter and `tool_executor`, and `tools/list` does not show it.

### 1. Pydantic params — `src/tools/tool_params.py`

Add your args model here, extending `baseToolArgs`. See existing models like `SearchIndexArgs` or `GetIndexMappingArgs` for the pattern.

```python
class YourToolArgs(baseToolArgs):
    param1: str = Field(description="Description of param1")
    optional_param: Optional[str] = Field(default=None, description="Optional param")
```

### 2. Async helper — `src/opensearch/helper.py`

Add your OpenSearch call here. Helpers should be simple — make the call and let exceptions propagate. **Do not add try/except in helpers**; error handling belongs in the tool function.

```python
async def your_helper(args: YourToolArgs):
    async with get_opensearch_client(args) as client:
        return await client.some_api(param=args.param1)
```

### 3. Async tool function — `src/tools/tools.py`

Add your tool function here. For error handling, **always use `log_tool_error()`** from `src/tools/tool_logging.py`:

```python
async def your_tool_function(args: YourToolArgs) -> list[dict]:
    try:
        await check_tool_compatibility('YourToolName', args)
        result = await your_helper(args)
        formatted = json.dumps(result, separators=(',', ':'))
        return [{'type': 'text', 'text': f'Result:\n{formatted}'}]
    except Exception as e:
        return log_tool_error('YourToolName', e, 'description of what failed')
```

This is critical — `log_tool_error()` returns responses with `is_error: True`, which `tool_executor.py` uses for metrics and monitoring. Without it, errors are silently logged as successes.

### 4. `TOOL_REGISTRY` entry — `src/tools/tools.py`

Add your tool to the static `TOOL_REGISTRY` dict. Look at any existing entry for the format.

```python
'YourToolName': {
    'display_name': 'YourToolName',
    'description': 'What the tool does',
    'input_schema': YourToolArgs.model_json_schema(),
    'function': your_tool_function,
    'args_model': YourToolArgs,
    'http_methods': 'GET',          # Controls write protection filter
    'min_version': '2.0.0',         # Optional: minimum OpenSearch version
},
```

Key fields:
- `display_name` — name exposed to MCP clients (must match the registry key)
- `http_methods` — `'GET'`, `'POST'`, `'PUT'`, `'DELETE'`, or `'GET, POST'`. When write protection is enabled, non-GET tools are filtered out
- `min_version` / `max_version` — optional semver strings for OpenSearch version gating
- `multi_only` — optional bool, if `True` the tool only appears in multi-cluster mode

### 5. Tool category — `src/tools/tool_filter.py`

Add your tool to the appropriate category list (`core_tools`, `search_relevance`, or a new category). Tools not in any enabled category are **filtered out** and invisible to clients. See the `search_relevance` category definition for how to add a new opt-in category.

Categories are enabled via YAML config or environment variable. Only `core_tools` is enabled by default.

**Via YAML config**:
```yaml
tool_filters:
  enabled_categories:
    - search_relevance
    - my_custom_category
```

**Via environment variable**:
```bash
export OPENSEARCH_ENABLED_CATEGORIES="search_relevance,my_custom_category"
```

In multi-cluster mode, all tools are returned without category filtering.

### 6. Tests

Add unit tests in `tests/tools/` following existing test patterns.

A new tool shows in `tools/list`, so it changes the `list_tools` characterization goldens that list it (multi mode lists every tool except the memory tools). That is a behaviour change: recreate those goldens and record it as described in [Characterization tests](#characterization-tests).

### Reference

The Search Relevance Workbench added 18 tools with zero infrastructure changes — all params in `tool_params.py`, helpers in `helper.py`, tool functions in `tools.py`, entries in `TOOL_REGISTRY`, and an opt-in `search_relevance` category in `tool_filter.py`. Use it as a blueprint for adding tool groups.

> **Note**: All new source files must include the Apache 2.0 license header:
> ```python
> # Copyright OpenSearch Contributors
> # SPDX-License-Identifier: Apache-2.0
> ```


## Testing

### Running Unit Tests

```bash
# Run all unit tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=mcp_server_opensearch

# Run specific test file
uv run pytest tests/test_tools.py

# Run tests with verbose output
uv run pytest -v
```

### Characterization tests

The characterization (golden) tests in `tests/characterization/` pin what the server does today, as a client sees it. They build the server with `create_mcp_server()` and drive it over the MCP protocol: in process with the SDK's `Client`, through the Starlette app `streaming_server.serve()` runs, or as `main()` in a subprocess over stdio. OpenSearch is faked, the OpenAPI specification comes from `tests/characterization/fixtures/`, and nothing touches the network. They pin:

- `tools/list` for every mode, configuration and filter (names, descriptions, input schemas, order);
- `tools/call` results and error shapes, and the requests each call sends to OpenSearch, for the core, generated, generic API, memory, agentic memory, skills and Search Relevance Workbench tools;
- `initialize`: capabilities, instructions and protocol version negotiation;
- the stdio, SSE and streamable HTTP transports, `/health`, header auth and `X-MCP-Client-Name`;
- the CLI flags, config files and environment variables, and the known gaps (`test_gaps_*.py`).

They run with the unit tests, or alone:

```bash
uv run --offline python -m pytest tests/characterization -q
```

To add a golden:

1. Write the test with `golden('<area>/<case>', normalize(data))`.
2. Run it once with `CHAR_GOLDEN_WRITE=1` to create `tests/characterization/golden/<area>/<case>.json`:
   ```bash
   CHAR_GOLDEN_WRITE=1 uv run --offline python -m pytest tests/characterization -q -k <test>
   ```
3. Read the new file: it is the behaviour you are pinning. Run the test again without the variable, then commit the test and the golden together.

The harness never overwrites an existing golden, not even with `CHAR_GOLDEN_WRITE=1`.

**A golden diff is a behaviour change.** Never edit a golden to make a change pass. When a golden comparison fails, either fix the code or, if the change is intended, delete the golden in the same commit and recreate it as above, then record the change in `docs/mcpserver-migration-behaviour-changes.md` (naming every golden it changed) and in `CHANGELOG.md` under `[Unreleased]`. This command checks it:

```bash
uv run --offline python scripts/check_behaviour_ledger.py
```

A change that only adds goldens is new characterization, not a behaviour change. The fixtures, the normalizer and the harness are described in [`tests/characterization/README.md`](tests/characterization/README.md).

### Running Integration Tests Locally

Integration tests run against a real OpenSearch cluster. Set the following environment variables, then run:

```bash
# Required — OpenSearch cluster endpoint and basic auth credentials
export IT_OPENSEARCH_URL="https://your-opensearch-endpoint"
export IT_BASIC_AUTH_USERNAME="admin"
export IT_BASIC_AUTH_PASSWORD="your-password"

# Required for AWS auth tests — temporary AWS credentials
export IT_AWS_ACCESS_KEY_ID="your-access-key"
export IT_AWS_SECRET_ACCESS_KEY="your-secret-key"
export IT_AWS_SESSION_TOKEN="your-session-token"
export IT_AWS_REGION="us-west-2"

# Required for IAM role assumption tests
export IT_IAM_ROLE_ARN="arn:aws:iam::123456789012:role/your-role"

# Optional — custom test index name (defaults to mcp-integration-test)
export IT_TEST_INDEX="my-test-index"

# Install the package and IT dependencies, then run
pip install dist/*.whl
pip install pytest-asyncio pytest-timeout "httpx[http2]"
python -m pytest integration_tests/ -v --tb=short --timeout=300
```

Tests that require missing environment variables will be automatically skipped.

### Agent runner merge gate

The Agent Runner merges each task only after the checks in `.agent-runner/config.json` (`ci.full`) pass. The gate runs these three commands offline, from the host's uv cache:

```bash
uv run --offline ruff format --check .
uv run --offline ruff check .
uv run --offline python -m pytest tests
```

These match the lint job and unit tests in `.github/workflows/ci.yml`. The tests in `integration_tests/` need a live OpenSearch cluster and credentials, so they are not part of the gate. They run only in GitHub CI, or locally as described in [Running Integration Tests Locally](#running-integration-tests-locally).

### Code Quality

```bash
# Format code
uv run ruff format .

# Check code quality
uv run ruff check .

# Run type checking
uv run mypy src/
```

#### Pre-commit Hooks

Install [pre-commit](https://pre-commit.com/) to automatically run ruff format and lint on staged files before each commit:

```bash
uv tool install pre-commit
pre-commit install
```

> **Note**: Make sure to run tests and code quality checks before submitting your changes.
