# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
from semver import Version
from tools.tool_params import *


# List all the helper functions, these functions perform a single rest call to opensearch
# these functions will be used in tools folder to eventually write more complex tools
def list_indices(args: ListIndicesArgs) -> json:
    from .client import initialize_client

    client = initialize_client(args)
    response = client.cat.indices(format='json')
    return response


def get_index_mapping(args: GetIndexMappingArgs) -> json:
    from .client import initialize_client

    client = initialize_client(args)
    response = client.indices.get_mapping(index=args.index)
    return response


def search_index(args: SearchIndexArgs) -> json:
    from .client import initialize_client

    client = initialize_client(args)
    response = client.search(index=args.index, body=args.query)
    return response


def get_shards(args: GetShardsArgs) -> json:
    from .client import initialize_client

    client = initialize_client(args)
    response = client.cat.shards(index=args.index, format='json')
    return response


def get_opensearch_version(args: baseToolArgs) -> Version:
    """Get the version of OpenSearch cluster.

    Returns:
        Version: The version of OpenSearch cluster (SemVer style)
    """
    from .client import initialize_client

    client = initialize_client(args)
    response = client.info()
    return Version.parse(response['version']['number'])
