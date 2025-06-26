![OpenSearch logo](https://github.com/opensearch-project/opensearch-py/raw/main/OpenSearch.svg)

# OpenSearch MCP Server Python Developer Guide

## Table of Contents
- [Overview](#overview)
- [Development Setup](#development-setup)
- [Running the Server](#running-the-server)
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

```bash
# Create & activate a virtual environment
uv venv 
source .venv/bin/activate

# Install dependencies
uv sync
```

### 3. Verify Installation

```bash
# Navigate to src directory (required for running the server)
cd src

# Test that the server can start
uv run python -m mcp_server_opensearch --help
```

## Running the Server

**Important**: These commands must be run from the `src` directory.

### Development Mode
```bash
cd src

# Run stdio server (default)
uv run python -m mcp_server_opensearch 

# Run streaming server (SSE/HTTP streaming)
uv run python -m mcp_server_opensearch --transport stream

# Run with debug logging
uv run python -m mcp_server_opensearch --log-level debug

# Run with custom AWS profile
uv run python -m mcp_server_opensearch --profile my-profile
```

### Multi Mode Development
```bash
cd src

# Run stdio server in multi mode with custom config file
uv run python -m mcp_server_opensearch --mode multi --config ../config/dev-clusters.yml

# Run streaming server (SSE/HTTP streaming) in multi mode with custom config file
uv run python -m mcp_server_opensearch --mode multi --config ../config/dev-clusters.yml --transport stream
```

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
# Add a new package
uv add <package-name>

# Add with specific version
uv add <package-name>==1.2.3

# Add as development dependency
uv add --dev <package-name>
```

> **Note**: This automatically updates `pyproject.toml`, `uv.lock`, and installs in the virtual environment.

### Updating Dependencies
```bash
# Update after manual pyproject.toml changes
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
