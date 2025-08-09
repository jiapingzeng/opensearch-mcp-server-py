# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import boto3
import logging
import os
from mcp_server_opensearch.clusters_information import ClusterInfo, get_cluster
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from tools.tool_params import baseToolArgs
from typing import Any, Dict
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OPENSEARCH_SERVICE = 'es'
OPENSEARCH_SERVERLESS_SERVICE = 'aoss'

# global profile variable from command line
arg_profile = None


def set_profile(profile: str) -> None:
    global arg_profile
    arg_profile = profile


def get_aws_region(cluster_info: ClusterInfo | None) -> str:
    """Get the AWS region based on priority order.

    Priority with cluster_info (multi mode):
    1. cluster_info.aws_region
    2. Region from cluster_info.profile
    3. AWS_REGION environment variable
    4. Command-line profile (arg_profile)
    5. AWS_PROFILE environment variable
    6. Default boto3 session region

    Priority without cluster_info (single mode):
    1. AWS_REGION environment variable
    2. Command-line profile (arg_profile)
    3. AWS_PROFILE environment variable
    4. Default boto3 session region

    Args:
        cluster_info (ClusterInfo): Optional cluster information

    Returns:
        str: AWS region
    """
    if cluster_info:
        if cluster_info.aws_region:
            return cluster_info.aws_region
        if cluster_info.profile:
            session = boto3.Session(profile_name=cluster_info.profile)
            return session.region_name
        if os.getenv('AWS_REGION', ''):
            return os.getenv('AWS_REGION', '')
        if arg_profile:
            session = boto3.Session(profile_name=arg_profile)
            return session.region_name
        if os.getenv('AWS_PROFILE', ''):
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', ''))
            return session.region_name
        else:
            session = boto3.Session()
            return session.region_name

    else:
        if os.getenv('AWS_REGION', ''):
            return os.getenv('AWS_REGION', '')
        if arg_profile:
            session = boto3.Session(profile_name=arg_profile)
            return session.region_name
        if os.getenv('AWS_PROFILE', ''):
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', ''))
            return session.region_name
        else:
            session = boto3.Session()
            return session.region_name


def is_serverless(cluster_info: ClusterInfo | None) -> bool:
    """Check if the OpenSearch instance is serverless.

    Args:
        args_or_cluster_info: Either ClusterInfo, or None

    Returns:
        bool: True if serverless, False otherwise
    """
    # Check cluster_info first
    if cluster_info:
        return cluster_info.is_serverless

    # If cluster_info is not provided, check the environment variable
    return os.getenv('AWS_OPENSEARCH_SERVERLESS', '').lower() == 'true'


def initialize_client_with_cluster(cluster_info: ClusterInfo | None) -> OpenSearch:
    """Initialize an OpenSearch client with authentication.

    Authentication methods (in order):
    1. No authentication (only if OPENSEARCH_NO_AUTH=true environment variable is set)
    2. IAM role authentication (if iam_arn is provided)
    3. Basic authentication (username/password)
    4. AWS credentials from boto3 session

    Service name depends on serverless mode:
    - 'aoss' for OpenSearch Serverless
    - 'es' for standard OpenSearch

    Args:
        cluster_info: Optional cluster information

    Returns:
        OpenSearch: Client instance

    Raises:
        ValueError: If opensearch_url is missing
        RuntimeError: If authentication fails
    """
    opensearch_url = (
        cluster_info.opensearch_url if cluster_info else os.getenv('OPENSEARCH_URL', '')
    )
    if not opensearch_url:
        raise ValueError(
            'OpenSearch URL must be provided using config file or OPENSEARCH_URL environment variable'
        )
    opensearch_username = (
        cluster_info.opensearch_username if cluster_info else os.getenv('OPENSEARCH_USERNAME', '')
    )
    opensearch_password = (
        cluster_info.opensearch_password if cluster_info else os.getenv('OPENSEARCH_PASSWORD', '')
    )
    iam_arn = cluster_info.iam_arn if cluster_info else os.getenv('AWS_IAM_ARN', '')
    profile = cluster_info.profile if cluster_info else arg_profile
    if not profile:
        profile = os.getenv('AWS_PROFILE', '')

    is_serverless_mode = is_serverless(cluster_info)
    service_name = OPENSEARCH_SERVERLESS_SERVICE if is_serverless_mode else OPENSEARCH_SERVICE

    if is_serverless_mode:
        logger.info('Using OpenSearch Serverless with service name: aoss')

    opensearch_timeout = (
        cluster_info.timeout if cluster_info else os.getenv('OPENSEARCH_TIMEOUT', None)
    )

    # Parse the OpenSearch domain URL
    parsed_url = urlparse(opensearch_url)

    # Common client configuration
    client_kwargs: Dict[str, Any] = {
        'hosts': [opensearch_url],
        'use_ssl': (parsed_url.scheme == 'https'),
        'verify_certs': os.getenv('OPENSEARCH_SSL_VERIFY', 'true').lower() != 'false',
        'connection_class': RequestsHttpConnection,
    }

    if opensearch_timeout:
        client_kwargs['timeout'] = int(opensearch_timeout)

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    aws_region = get_aws_region(cluster_info)
    print('aws_region is', aws_region)

    # 1. Try no authentication if explicitly enabled
    if os.getenv('OPENSEARCH_NO_AUTH', '').lower() == 'true':
        logger.info(
            '[NO AUTH] Attempting connection without authentication (OPENSEARCH_NO_AUTH=true)'
        )
        try:
            return OpenSearch(**client_kwargs)
        except Exception as e:
            logger.error(f'[NO AUTH] Failed to connect without authentication: {str(e)}')

    # 2. Try IAM auth
    if iam_arn:
        logger.info(f'[IAM AUTH] Using IAM role authentication: {iam_arn}')
        try:
            if not aws_region:
                raise RuntimeError(
                    'AWS region not found, please specify region using `aws configure`'
                )

            sts_client = session.client('sts', region_name=aws_region)
            assumed_role = sts_client.assume_role(
                RoleArn=iam_arn, RoleSessionName='OpenSearchClientSession'
            )
            credentials = assumed_role['Credentials']

            aws_auth = AWS4Auth(
                credentials['AccessKeyId'],
                credentials['SecretAccessKey'],
                aws_region,
                service_name,
                session_token=credentials['SessionToken'],
            )
            client_kwargs['http_auth'] = aws_auth
            return OpenSearch(**client_kwargs)
        except Exception as e:
            logger.error(f'[IAM AUTH] Failed to assume IAM role {iam_arn}: {str(e)}')

    # 3. Try basic auth
    if opensearch_username and opensearch_password:
        logger.info(f'[BASIC AUTH] Using basic authentication: {opensearch_username}')
        client_kwargs['http_auth'] = (opensearch_username, opensearch_password)
        return OpenSearch(**client_kwargs)

    # 4. Try to get credentials from boto3 session
    try:
        logger.info(f'[AWS CREDS] Using AWS credentials authentication')
        credentials = session.get_credentials()
        if not aws_region:
            raise RuntimeError('AWS region not found, please specify region using `aws configure`')
        if credentials:
            aws_auth = AWS4Auth(
                refreshable_credentials=credentials,
                service=service_name,
                region=aws_region,
            )
            client_kwargs['http_auth'] = aws_auth
            return OpenSearch(**client_kwargs)
    except (boto3.exceptions.Boto3Error, Exception) as e:
        logger.error(f'[AWS CREDS] Failed to get AWS credentials: {str(e)}')

    raise RuntimeError('No valid AWS or basic authentication provided for OpenSearch')


def initialize_client(args: baseToolArgs) -> OpenSearch:
    """Initialize and return an OpenSearch client based on provided arguments.

    Supports two modes:
    - Multi-cluster: When args.opensearch_cluster_name is provided
    - Single-cluster: When no cluster name is provided (uses environment variables)

    Args:
        args (baseToolArgs): Arguments containing optional opensearch_cluster_name

    Returns:
        OpenSearch: An initialized OpenSearch client instance
    """
    cluster_info = None
    if args and args.opensearch_cluster_name:
        cluster_info = get_cluster(args.opensearch_cluster_name)
    return initialize_client_with_cluster(cluster_info)
