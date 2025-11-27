# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import csv
import io
from semver import Version
from tools.tool_params import *

# Configure logging
logger = logging.getLogger(__name__)


# List all the helper functions, these functions perform a single rest call to opensearch
# these functions will be used in tools folder to eventually write more complex tools
async def list_indices(args: ListIndicesArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.cat.indices(format='json')
        return response


async def get_index(args: ListIndicesArgs) -> json:
    """Get detailed information about a specific index.

    Args:
        args: ListIndicesArgs containing the index name

    Returns:
        json: Detailed index information including settings and mappings
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get(index=args.index)
        return response


async def get_index_mapping(args: GetIndexMappingArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get_mapping(index=args.index)
        return response


async def search_index(args: SearchIndexArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.search(index=args.index, body=args.query)
        return response


async def get_shards(args: GetShardsArgs) -> json:
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.cat.shards(index=args.index, format='json')
        return response


async def get_segments(args: GetSegmentsArgs) -> json:
    """Get information about Lucene segments in indices.

    Args:
        args: GetSegmentsArgs containing optional index filter

    Returns:
        json: Segment information for the specified indices or all indices
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # If index is provided, filter by that index
        index_param = args.index if args.index else None

        response = await client.cat.segments(index=index_param, format='json')
        return response


async def get_cluster_state(args: GetClusterStateArgs) -> json:
    """Get the current state of the cluster.

    Args:
        args: GetClusterStateArgs containing optional metric and index filters

    Returns:
        json: Cluster state information based on the requested metrics and indices
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build parameters dictionary with non-None values
        params = {}
        if args.metric:
            params['metric'] = args.metric
        if args.index:
            params['index'] = args.index

        response = await client.cluster.state(**params)
        return response


async def get_nodes(args: CatNodesArgs) -> json:
    """Get information about nodes in the cluster.

    Args:
        args: GetNodesArgs containing optional metrics filter

    Returns:
        json: Node information for the cluster
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # If metrics is provided, use it as a parameter
        metrics_param = args.metrics if args.metrics else None

        response = await client.cat.nodes(format='json', h=metrics_param)
        return response


async def get_index_info(args: GetIndexInfoArgs) -> json:
    """Get detailed information about an index including mappings, settings, and aliases.

    Args:
        args: GetIndexInfoArgs containing the index name

    Returns:
        json: Detailed index information
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        response = await client.indices.get(index=args.index)
        return response


async def get_index_stats(args: GetIndexStatsArgs) -> json:
    """Get statistics about an index.

    Args:
        args: GetIndexStatsArgs containing the index name and optional metric filter

    Returns:
        json: Index statistics
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build parameters dictionary with non-None values
        params = {}
        if args.metric:
            params['metric'] = args.metric

        response = await client.indices.stats(index=args.index, **params)
        return response


async def get_query_insights(args: GetQueryInsightsArgs) -> json:
    """Get insights about top queries in the cluster.

    Args:
        args: GetQueryInsightsArgs containing connection parameters

    Returns:
        json: Query insights from the /_insights/top_queries endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # since the Python client might not have a dedicated method for this endpoint
        response = await client.transport.perform_request(
            method='GET', url='/_insights/top_queries'
        )

        return response


async def get_nodes_hot_threads(args: GetNodesHotThreadsArgs) -> str:
    """Get information about hot threads in the cluster nodes.

    Args:
        args: GetNodesHotThreadsArgs containing connection parameters

    Returns:
        str: Hot threads information from the /_nodes/hot_threads endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # The hot_threads API returns text, not JSON
        response = await client.transport.perform_request(method='GET', url='/_nodes/hot_threads')

        return response


async def get_allocation(args: GetAllocationArgs) -> json:
    """Get information about shard allocation across nodes in the cluster.

    Args:
        args: GetAllocationArgs containing connection parameters

    Returns:
        json: Allocation information from the /_cat/allocation endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the cat.allocation method with JSON format
        response = await client.cat.allocation(format='json')

        return response


async def get_long_running_tasks(args: GetLongRunningTasksArgs) -> json:
    """Get information about long-running tasks in the cluster, sorted by running time.

    Args:
        args: GetLongRunningTasksArgs containing limit parameter

    Returns:
        json: Task information from the /_cat/tasks endpoint, sorted by running time
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Use the transport.perform_request method to make a direct REST API call
        # since we need to sort by running_time which might not be directly supported by the client
        response = await client.transport.perform_request(
            method='GET',
            url='/_cat/tasks',
            params={
                's': 'running_time:desc',  # Sort by running time in descending order
                'format': 'json',
            },
        )

        # Limit the number of tasks returned if specified
        if args.limit and isinstance(response, list):
            return response[: args.limit]

        return response


async def get_nodes_info(args: GetNodesArgs) -> json:
    """Get detailed information about nodes in the cluster.

    Args:
        args: GetNodesArgs containing optional node_id, metric filters, and other parameters

    Returns:
        json: Detailed node information from the /_nodes endpoint
    """
    from .client import get_opensearch_client

    async with get_opensearch_client(args) as client:
        # Build the URL path based on provided parameters
        url_parts = ['/_nodes']

        # Add node_id if provided
        if args.node_id:
            url_parts.append(args.node_id)

        # Add metric if provided
        if args.metric:
            url_parts.append(args.metric)

        url = '/'.join(url_parts)

        # Use the transport.perform_request method to make a direct REST API call
        response = await client.transport.perform_request(method='GET', url=url)

        return response


def convert_search_results_to_csv(search_results: dict) -> str:
    """Convert OpenSearch search results to CSV format.
    
    Args:
        search_results: The JSON response from search_index function
        
    Returns:
        str: CSV formatted string of the search results
    """
    if not search_results:
        return "No search results to convert"
    
    has_hits = 'hits' in search_results and search_results['hits']['hits']
    has_aggregations = 'aggregations' in search_results
    
    # Handle aggregations-only queries
    if has_aggregations and not has_hits:
        return json.dumps(search_results['aggregations'], indent=2)
    
    # Handle hits-only queries
    if has_hits and not has_aggregations:
        return _convert_hits_to_csv(search_results['hits']['hits'])
    
    # Handle queries with both hits and aggregations
    if has_hits and has_aggregations:
        hits_csv = _convert_hits_to_csv(search_results['hits']['hits'])
        aggregations_json = json.dumps(search_results['aggregations'], indent=2)
        return f"SEARCH HITS:\n{hits_csv}\n\nAGGREGATIONS:\n{aggregations_json}"
    
    return "No search results to convert"


def _convert_hits_to_csv(hits: list) -> str:
    """Convert search hits to CSV format.
    
    Args:
        hits: List of search hits
        
    Returns:
        str: CSV formatted string
    """
    if not hits:
        return "No documents found in search results"
    
    # Extract all unique field names from all documents (flattened)
    all_fields = set()
    for hit in hits:
        if '_source' in hit:
            _flatten_fields(hit['_source'], all_fields)
        # Also include metadata fields
        all_fields.update(['_index', '_id', '_score'])
    
    # Convert to sorted list for consistent column order
    fieldnames = sorted(list(all_fields))
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    # Write each document as a row
    for hit in hits:
        row = {}
        # Add metadata fields
        row['_index'] = hit.get('_index', '')
        row['_id'] = hit.get('_id', '')
        row['_score'] = hit.get('_score', '')
        
        # Add source fields (flattened)
        if '_source' in hit:
            _flatten_object(hit['_source'], row)
        
        writer.writerow(row)
    
    return output.getvalue()


def _flatten_fields(obj: dict, fields: set, prefix: str = '') -> None:
    """Extract all field names from nested objects.
    
    Args:
        obj: Object to extract field names from
        fields: Set to add field names to
        prefix: Current field prefix
    """
    for key, value in obj.items():
        field_name = f'{prefix}{key}' if prefix else key
        if isinstance(value, dict):
            _flatten_fields(value, fields, f'{field_name}.')
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # For arrays of objects, flatten the first object to get field structure
            _flatten_fields(value[0], fields, f'{field_name}.')
            fields.add(field_name)  # Also keep the array field itself
        else:
            fields.add(field_name)


def _flatten_object(obj: dict, row: dict, prefix: str = '') -> None:
    """Flatten nested objects into separate columns.
    
    Args:
        obj: Object to flatten
        row: Row dictionary to add flattened fields to
        prefix: Current field prefix
    """
    for key, value in obj.items():
        field_name = f'{prefix}{key}' if prefix else key
        if isinstance(value, dict):
            _flatten_object(value, row, f'{field_name}.')
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # For arrays of objects, flatten first object and keep array as JSON
                _flatten_object(value[0], row, f'{field_name}.')
                row[field_name] = json.dumps(value)
            else:
                # For simple arrays, convert to JSON
                row[field_name] = json.dumps(value)
        else:
            row[field_name] = str(value) if value is not None else ''


async def get_opensearch_version(args: baseToolArgs) -> Version:
    """Get the version of OpenSearch cluster.

    Returns:
        Version: The version of OpenSearch cluster (SemVer style)
    """
    from .client import get_opensearch_client

    try:
        async with get_opensearch_client(args) as client:
            response = await client.info()
            return Version.parse(response['version']['number'])
    except Exception as e:
        logger.error(f'Error getting OpenSearch version: {e}')
        return None
