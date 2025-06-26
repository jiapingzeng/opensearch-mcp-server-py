# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from . import main


# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info('Starting MCP server...')
main()
