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

## Adding Custom Tools

To add a new tool to the MCP server, follow these steps:

### 1. Create the Tool Function

Add your tool function in `src/mcp_server_opensearch/tools/tools.py`:

```python
async def your_tool_function(args: YourToolArgs) -> list[dict]:
    """
    Description of what your tool does.
    
    Args:
        args: Tool arguments
        
    Returns:
        List of response objects
    """
    try:
        # Your tool implementation here
        result = your_implementation()
        return [{
            "type": "text",
            "text": result
        }]
    except Exception as e:
        return [{
            "type": "text",
            "text": f"Error: {str(e)}"
        }]
```

### 2. Define the Arguments Model

Create a Pydantic model for your tool's arguments that extends `baseToolArgs`:

```python
from pydantic import Field
from .tool_params import baseToolArgs

class YourToolArgs(baseToolArgs):
    """Arguments for the YourTool tool."""
    
    param1: str = Field(description="Description of param1")
    param2: int = Field(description="Description of param2", ge=0)
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "param1": "example_value",
                    "param2": 10
                }
            ]
        }
```

### 3. Add Helper Functions

Create helper functions in `src/mcp_server_opensearch/opensearch/helper.py`:

```python
def your_helper_function(args: YourToolArgs) -> dict:
    """
    Helper function that performs a single REST call to OpenSearch.

    Returns:
        OpenSearch response data
        
    Raises:
        OpenSearchException: If the OpenSearch request fails
    """
    # Your OpenSearch REST call implementation here
    return result
```

### 4. Register Your Tool

Add your tool to the `TOOL_REGISTRY` dictionary in `src/mcp_server_opensearch/tools/tools.py`:

```python
TOOL_REGISTRY = {
    # ... existing tools ...
    "YourToolName": {
        "description": "Description of what your tool does",
        "input_schema": YourToolArgs.model_json_schema(),
        "function": your_tool_function,
        "args_model": YourToolArgs,
    }
}
```

### 5. Import Helper Functions

Import and use the helper functions in your tool:

```python
from mcp_server_opensearch.opensearch.helper import your_helper_function
```


The tool will be automatically available through the MCP server after registration.

> Note: Each helper function should perform a single REST call to OpenSearch. This design promotes:
> - Clear separation of concerns
> - Easy testing and maintenance
> - Extensible architecture
> - Reusable OpenSearch operations


## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=mcp_server_opensearch

# Run specific test file
uv run pytest tests/test_tools.py

# Run tests with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check code quality
uv run ruff check .

# Run type checking
uv run mypy src/
```

> **Note**: Make sure to run tests and code quality checks before submitting your changes.
