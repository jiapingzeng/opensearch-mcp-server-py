# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
from typing import Dict, List


def parse_unknown_args_to_dict(unknown_args: List[str]) -> Dict[str, str]:
    """Parses a list of unknown arguments into a dictionary."""
    parser = argparse.ArgumentParser()

    # Extract argument keys and track duplicates with warnings
    seen = set()
    duplicates = set()
    arg_keys = set()

    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg.split('=')[0]
            arg_keys.add(key)
            if key in seen:
                duplicates.add(key)
                logging.warning(f"Duplicate argument '{key}' found. Using the latest value.")
            else:
                seen.add(key)

    for key in arg_keys:
        parser.add_argument(key)

    try:
        parsed_args, _ = parser.parse_known_args(unknown_args)
        return {k: v for k, v in vars(parsed_args).items() if v is not None}
    except Exception as e:
        logging.error(f'Error parsing unknown arguments: {e}')
        return {}


def main() -> None:
    """
    Main entry point for the OpenSearch MCP Server.
    Handles command line arguments and starts the appropriate server based on transport type.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='OpenSearch MCP Server')
    parser.add_argument(
        '--transport',
        choices=['stdio', 'stream'],
        default='stdio',
        help='Transport type (stdio or stream)',
    )
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (streaming only)')
    parser.add_argument(
        '--port', type=int, default=9900, help='Port to listen on (streaming only)'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multi'],
        default='single',
        help='Server mode: single (default) uses environment variables for OpenSearch connection, multi requires explicit connection parameters',
    )
    parser.add_argument(
        '--profile', default='', help='AWS profile to use for OpenSearch connection'
    )
    parser.add_argument(
        '--config',
        dest='config_file_path',
        default='',
        help='Path to a YAML configuration file',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )

    args, unknown = parser.parse_known_args()

    # Configure logging with appropriate level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info('Starting MCP server...')
    cli_tool_overrides = parse_unknown_args_to_dict(unknown)

    # Import servers lazily to avoid circular imports at module load time
    from .stdio_server import serve as serve_stdio
    from .streaming_server import serve as serve_streaming

    # Start the appropriate server based on transport type
    if args.transport == 'stdio':
        asyncio.run(
            serve_stdio(
                mode=args.mode,
                profile=args.profile,
                config_file_path=args.config_file_path,
                cli_tool_overrides=cli_tool_overrides,
            )
        )
    else:
        asyncio.run(
            serve_streaming(
                host=args.host,
                port=args.port,
                mode=args.mode,
                profile=args.profile,
                config_file_path=args.config_file_path,
                cli_tool_overrides=cli_tool_overrides,
            )
        )


if __name__ == '__main__':
    main()
