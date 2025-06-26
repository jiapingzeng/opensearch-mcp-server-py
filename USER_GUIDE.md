![OpenSearch logo](https://github.com/opensearch-project/opensearch-py/raw/main/OpenSearch.svg)

# OpenSearch MCP Server Python User Guide

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Server Modes](#server-modes)
- [Authentication](#authentication)
- [Running the Server](#running-the-server)
- [Tool Filter](#tool-filter)
- [LangChain Integration](#langchain-integration)

## Overview

The OpenSearch MCP (Model Context Protocol) Server Python provides a bridge between AI agents and OpenSearch clusters. It supports both single-cluster and multi-cluster configurations with various authentication methods including IAM roles, basic authentication, and AWS credentials.
The server can be started in both STDIO and streaming modes.

### Streaming Server

The OpenSearch MCP server supports a streaming transport that includes both SSE and HTTP streaming. The previous SSE server is now part of the streaming server and will be deprecated in the future. Use the streaming transport for all new integrations.

## Installation

### Option 1: Using uvx (Recommended - No Installation Required)

Install `uv` via `pip` or [standalone installer](https://github.com/astral-sh/uv?tab=readme-ov-file#installation):
```bash
pip install uv
```

The OpenSearch MCP server can be used directly via `uvx` without installation.

### Option 2: Local Installation

Install from [PyPI](https://pypi.org/project/opensearch-mcp-server-py/):
```bash
pip install opensearch-mcp-server-py
```

## Quick Start

### Prerequisites
1. Install `uv` (see [Installation](#installation))
2. Configure your AI agent of choice

### AI Agent Configuration

#### For Q Developer CLI
Configure `~/.aws/amazonq/mcp.json`. See [here](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-configuration.html) for additional configuration options.

#### For Claude Desktop
Configure `claude_desktop_config.json` from Settings > Developer. See [here](https://modelcontextprotocol.io/quickstart/user#2-add-the-filesystem-mcp-server) for more details.

### Basic Setup

#### Single Mode (Recommended for Beginners)
```json
{
  "mcpServers": {
    "opensearch-mcp-server": {
      "command": "uvx",
      "args": ["opensearch-mcp-server-py"],
      "env": {
        "OPENSEARCH_URL": "<your_opensearch_domain_url>",
        "OPENSEARCH_USERNAME": "<your_username>",
        "OPENSEARCH_PASSWORD": "<your_password>"
      }
    }
  }
}
```

**Supported Environment Variables for Single Mode:**
- `OPENSEARCH_URL` (required): Your OpenSearch cluster endpoint
- `OPENSEARCH_USERNAME` & `OPENSEARCH_PASSWORD`: For basic authentication
- `AWS_IAM_ARN`: For IAM role authentication
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`: For AWS credentials
- `AWS_REGION`: The AWS region of the cluster
- `AWS_OPENSEARCH_SERVERLESS`: Set to "true" for OpenSearch Serverless
- `AWS_PROFILE`: AWS profile name (optional)
- `OPENSEARCH_SSL_VERIFY`: Control SSL certificate verification (default: "true", set to "false" to disable)

See [Authentication](#authentication) section for detailed authentication setup.

#### Multi Mode (For Multiple Clusters)
```json
{
  "mcpServers": {
    "opensearch-mcp-server": {
      "command": "uvx",
      "args": [
        "opensearch-mcp-server-py",
        "--mode", "multi",
        "--config", "/path/to/your/clusters.yml"
      ],
      "env": {}
    }
  }
}
```

**Example YAML Configuration File (`clusters.yml`):**
```yaml
version: "1.0"
description: "OpenSearch cluster configurations"

clusters:
  # Cluster name: "local-dev" - used as opensearch_cluster_name parameter in tool calls
  local-dev:
    opensearch_url: "http://localhost:9200"
    opensearch_username: "admin"
    opensearch_password: "admin123"

  # Cluster name: "production" - used as opensearch_cluster_name parameter in tool calls
  production:
    opensearch_url: "https://prod-opensearch.us-east-1.es.amazonaws.com"
    iam_arn: "arn:aws:iam::123456789012:role/OpenSearchProductionRole"
    aws_region: "us-east-1"
    profile: "production"

  # Cluster name: "staging" - used as opensearch_cluster_name parameter in tool calls
  staging:
    opensearch_url: "https://staging-opensearch.us-west-2.es.amazonaws.com"
    profile: "staging"
```

**Key Points about Multi Mode:**
- **Cluster Names**: The keys under `clusters` (e.g., `local-dev`, `production`, `staging`) are used as the `opensearch_cluster_name` parameter when calling tools
- **Authentication**: Each cluster can use different authentication methods (basic auth, IAM roles, AWS profiles)
- **Tool Usage**: When using tools, you must specify which cluster to use: `{"opensearch_cluster_name": "production", "index": "users"}`

That's it! You are now ready to use your AI agent with OpenSearch tools.

**Next Steps:**
- For detailed authentication setup, see [Authentication](#authentication)
- For running the server manually, see [Running the Server](#running-the-server)

## Server Modes

The OpenSearch MCP server supports two modes of operation:

### Single Mode (Default)
- Connects to a single OpenSearch cluster
- Uses environment variables for configuration
- Automatically filters tools based on OpenSearch version compatibility
- Simple tool schemas

### Multi Mode
- Supports multiple OpenSearch clusters
- Uses a YAML configuration file to define clusters
- All tools are available regardless of version
- Full tool schemas with all parameters exposed
- **Important**: LLMs must provide an `opensearch_cluster_name` parameter to specify which cluster to use
- If a tool is not compatible with the OpenSearch version, an error is raised during tool execution
- **Fallback**: If no config file is provided, multi mode falls back to single mode behavior

### Cluster Name Parameter in Multi Mode

In multi mode, all tools have an additional parameter:
- `opensearch_cluster_name`: The name of the cluster as defined in your YAML configuration file

The LLM needs to have context about the available cluster names to make informed decisions about which cluster to use for each operation.

#### Example Tool Calls
```json
{
  "opensearch_cluster_name": "local-dev",
  "index": "my_index"
}
```

```json
{
  "opensearch_cluster_name": "production",
  "index": "users",
  "query": {
    "match": {
      "status": "active"
    }
  }
}
```

The LLM should choose the appropriate cluster based on the operation context (e.g., use `local-dev` for testing, `production` for production data).

## Authentication

### Authentication Methods

The server supports multiple authentication methods with the following priority order:
1. **IAM Role Authentication**
2. **Basic Authentication**
3. **AWS Credentials Authentication**

### Single Mode Authentication

#### Basic Authentication
```bash
export OPENSEARCH_URL="<your_opensearch_domain_url>"
export OPENSEARCH_USERNAME="<your_opensearch_domain_username>"
export OPENSEARCH_PASSWORD="<your_opensearch_domain_password>"
```

#### IAM Role Authentication
```bash
export OPENSEARCH_URL="<your_opensearch_domain_url>"
export AWS_IAM_ARN="arn:aws:iam::123456789012:role/YourOpenSearchRole"
export AWS_REGION="<your_aws_region>"
```

#### AWS Credentials Authentication
```bash
export OPENSEARCH_URL="<your_opensearch_domain_url>"
export AWS_REGION="<your_aws_region>"
export AWS_ACCESS_KEY_ID="<your_aws_access_key>"
export AWS_SECRET_ACCESS_KEY="<your_aws_secret_access_key>"
export AWS_SESSION_TOKEN="<your_aws_session_token>"
```

#### OpenSearch Serverless
```bash
export OPENSEARCH_URL="<your_opensearch_serverless_endpoint>"
export AWS_OPENSEARCH_SERVERLESS=true
export AWS_REGION="<your_aws_region>"
export AWS_ACCESS_KEY_ID="<your_aws_access_key>"
export AWS_SECRET_ACCESS_KEY="<your_aws_secret_access_key>"
export AWS_SESSION_TOKEN="<your_aws_session_token>"
```

### Multi Mode Authentication

Multi mode uses a YAML configuration file to define authentication for each cluster:

```yaml
version: "1.0"
description: "OpenSearch cluster configurations"

clusters:
  # Basic Authentication
  local-cluster:
    opensearch_url: "http://localhost:9200"
    opensearch_username: "admin"
    opensearch_password: "your_password_here"

  # AWS Credentials Authentication
  remote-cluster:
    opensearch_url: "https://your-opensearch-domain.us-east-2.es.amazonaws.com"
    profile: "your-aws-profile"
  
  # IAM Role Authentication
  remote-cluster-with-iam:
    opensearch_url: "https://your-opensearch-domain.us-east-2.es.amazonaws.com"
    iam_arn: "arn:aws:iam::123456789012:role/YourOpenSearchRole"
    aws_region: "us-east-2"
    profile: "your-aws-profile"
```

#### Authentication Methods in Multi Mode:

1. **IAM Role Authentication:**
   - Requires: `opensearch_url`, `iam_arn`, `aws_region`, `profile` (optional)
   - **Process**: The server assumes the specified IAM role using AWS STS and then connects to the cluster using those temporary credentials

2. **Basic Authentication:**
   - Requires: `opensearch_url`, `opensearch_username`, `opensearch_password`

3. **AWS Credentials Authentication:**
   - Requires: `opensearch_url`, `profile` (optional)
   - Uses AWS credentials from the specified profile or default credentials

### AWS Profile Support

You can specify an AWS profile to use for authentication in both modes:

#### Single Mode
```bash
# Using environment variable
export AWS_PROFILE="my-aws-profile"
python -m mcp_server_opensearch

# Using command line argument
python -m mcp_server_opensearch --profile my-aws-profile
```

#### Multi Mode
```bash
# Profile specified in config file (recommended)
python -m mcp_server_opensearch --mode multi --config clusters.yml

# Profile as fallback if not in config file
export AWS_PROFILE="my-aws-profile"
python -m mcp_server_opensearch --mode multi --config clusters.yml
```

## Running the Server

### Single Mode
```bash
# Stdio Server
python -m mcp_server_opensearch

# Streaming Server (SSE/HTTP streaming)
python -m mcp_server_opensearch --transport stream

# With AWS Profile
python -m mcp_server_opensearch --profile my-aws-profile
```

### Multi Mode
```bash
# Stdio Server with config file
python -m mcp_server_opensearch --mode multi --config clusters.yml

# Streaming Server with config file
python -m mcp_server_opensearch --mode multi --config clusters.yml --transport stream

# With AWS Profile (fallback if not in config)
python -m mcp_server_opensearch --mode multi --config clusters.yml --profile my-aws-profile

# Fallback to single mode behavior (no config file)
python -m mcp_server_opensearch --mode multi
```

## Tool Filter

OpenSearch MCP server supports tool filtering to disable specific tools by name, category, or operation type. You can configure filtering using either a YAML configuration file or environment variables.

### Configuration Methods

1. YAML Configuration File

Create a YAML file with your tool filtering configuration:
```yaml
# Define custom tool categories
tool_category:
  <category_name>:
    - <tool_name>

# Configure tool filters
tool_filters:
  disabled_tools:
    - <tool_name_to_disable>
  disabled_categories:
    - <category_name_to_disable>
  disabled_tools_regex:
    - <regex_pattern_to_disable>  # (e.g., search.*)
  settings:
    allow_write: true  # Enable/disable write-only operations
```

To use your configuration file, start the server with the `--config` flag:

```bash
# Run stdio server with tool filter config file
python -m mcp_server_opensearch --config path/to/config.yml

# Run Streaming server with tool filter config file
python -m mcp_server_opensearch --transport stream --config path/to/config.yml
```

2. Environment Variables

Set environment variables for tool filtering:
```bash
# Tool Categories
export OPENSEARCH_TOOL_CATEGORIES='{"<name_of_category>":["<tool_name>","<tool_name>"]}'

# Disable Specific Tools
export OPENSEARCH_DISABLED_TOOLS="<tool_name>"

# Disable Categories
export OPENSEARCH_DISABLED_CATEGORIES="<category_name>"

# Regex Pattern
export OPENSEARCH_DISABLED_TOOLS_REGEX="<regex_pattern>"

# Operation Settings
export OPENSEARCH_SETTINGS_ALLOW_WRITE=true
```

### Important Notes
- Tool names are case-insensitive
- All configuration fields are optional
- When both config file and environment variables are provided, the config file will be prioritized

## LangChain Integration

The OpenSearch MCP server can be easily integrated with LangChain using the SSE server transport.

### Prerequisites
1. Install required packages:
```bash
pip install langchain langchain-mcp-adapters langchain-openai
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="<your-openai-key>"
```

3. Ensure OpenSearch MCP server is running in streaming mode:
```bash
# Single Mode
python -m mcp_server_opensearch --transport stream

# Multi Mode
python -m mcp_server_opensearch --mode multi --config clusters.yml --transport stream
```

### Example Integration Script
```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent

# Initialize LLM (can use any LangChain-compatible LLM)
model = ChatOpenAI(model="gpt-4o")

async def main():
    # Connect to MCP server and create agent
    async with MultiServerMCPClient({
        "opensearch-mcp-server": {
            "transport": "sse",
            "url": "http://localhost:9900/sse",  # SSE server endpoint
            "headers": {
                "Authorization": "Bearer secret-token",
            }
        }
    }) as client:
        tools = client.get_tools()
        agent = initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,  # Enables detailed output of the agent's thought process
        )

        # Example query
        await agent.ainvoke({"input": "List all indices"})

if __name__ == "__main__":
    asyncio.run(main())
```

### Important Notes
- The script is compatible with any LLM that integrates with LangChain and supports tool calling
- Make sure the OpenSearch MCP server is running before executing the script
- Configure authentication and environment variables as needed
- In multi mode, you can specify which cluster to use in your queries by providing the `opensearch_cluster_name` parameter
- The LLM needs to know the available cluster names from your configuration file to make informed decisions

## Important Notes

- In single mode, the `OPENSEARCH_URL` must be set via environment variables
- In multi mode, the `OPENSEARCH_URL` must be provided in the config file for each cluster
- **Multi Mode Requirement**: LLMs must provide an `opensearch_cluster_name` parameter to specify which cluster to use
- The LLM needs context about available cluster names from your configuration file
