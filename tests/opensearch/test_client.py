# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

import boto3
import os
import pytest
from opensearch.client import initialize_client
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from tools.tool_params import baseToolArgs
from unittest.mock import Mock, patch


class TestOpenSearchClient:
    def setup_method(self):
        """Setup that runs before each test method."""
        # Clear any existing environment variables
        self.original_env = {}
        for key in ['OPENSEARCH_USERNAME', 'OPENSEARCH_PASSWORD', 'AWS_REGION', 'OPENSEARCH_URL']:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]

    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            os.environ[key] = value

    def test_initialize_client_empty_url(self):
        """Test that initialize_client raises ValueError when opensearch_url is empty."""
        with pytest.raises(ValueError) as exc_info:
            initialize_client(baseToolArgs())

        assert (
            str(exc_info.value)
            == 'OpenSearch URL must be provided using config file or OPENSEARCH_URL environment variable'
        )

    @patch('opensearch.client.OpenSearch')
    def test_initialize_client_basic_auth(self, mock_opensearch):
        """Test client initialization with basic authentication."""
        # Set environment variables
        os.environ['OPENSEARCH_USERNAME'] = 'test-user'
        os.environ['OPENSEARCH_PASSWORD'] = 'test-password'
        os.environ['OPENSEARCH_URL'] = 'https://test-opensearch-domain.com'

        # Mock OpenSearch client
        mock_client = Mock()
        mock_opensearch.return_value = mock_client

        # Execute
        client = initialize_client(baseToolArgs())

        # Assert
        assert client == mock_client
        mock_opensearch.assert_called_once_with(
            hosts=['https://test-opensearch-domain.com'],
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            http_auth=('test-user', 'test-password'),
        )

    @patch('opensearch.client.OpenSearch')
    @patch('opensearch.client.boto3.Session')
    def test_initialize_client_aws_auth(self, mock_session, mock_opensearch):
        """Test client initialization with AWS IAM authentication."""
        # Set environment variables
        os.environ['AWS_REGION'] = 'us-west-2'
        os.environ['OPENSEARCH_URL'] = 'https://test-opensearch-domain.com'

        # Mock AWS credentials
        mock_credentials = Mock()
        mock_credentials.access_key = 'test-access-key'
        mock_credentials.secret_key = 'test-secret-key'
        mock_credentials.token = 'test-token'

        mock_session_instance = Mock()
        mock_session_instance.get_credentials.return_value = mock_credentials
        mock_session.return_value = mock_session_instance

        # Mock OpenSearch client
        mock_client = Mock()
        mock_opensearch.return_value = mock_client

        # Execute
        client = initialize_client(baseToolArgs())

        # Assert
        assert client == mock_client
        mock_opensearch.assert_called_once()
        call_kwargs = mock_opensearch.call_args[1]
        assert call_kwargs['hosts'] == ['https://test-opensearch-domain.com']
        assert call_kwargs['use_ssl'] is True
        assert call_kwargs['verify_certs'] is True
        assert call_kwargs['connection_class'] == RequestsHttpConnection
        assert isinstance(call_kwargs['http_auth'], AWS4Auth)

    @patch('opensearch.client.OpenSearch')
    @patch('opensearch.client.boto3.Session')
    def test_initialize_client_aws_auth_error(self, mock_session, mock_opensearch):
        """Test client initialization when AWS authentication fails."""
        # Set environment variables
        os.environ['AWS_REGION'] = 'us-west-2'
        os.environ['OPENSEARCH_URL'] = 'https://test-opensearch-domain.com'

        # Mock AWS session to raise an error
        mock_session_instance = Mock()
        mock_session_instance.get_credentials.side_effect = boto3.exceptions.Boto3Error(
            'AWS credentials error'
        )
        mock_session.return_value = mock_session_instance

        # Execute and assert
        with pytest.raises(RuntimeError) as exc_info:
            initialize_client(baseToolArgs())
        assert (
            str(exc_info.value) == 'No valid AWS or basic authentication provided for OpenSearch'
        )

    @patch('opensearch.client.OpenSearch')
    @patch('opensearch.client.boto3.Session')
    def test_initialize_client_no_auth(self, mock_session, mock_opensearch):
        """Test client initialization when no authentication is available."""
        # Set environment variable
        os.environ['OPENSEARCH_URL'] = 'https://test-opensearch-domain.com'

        # Mock AWS session to return no credentials
        mock_session_instance = Mock()
        mock_session_instance.get_credentials.return_value = None
        mock_session.return_value = mock_session_instance

        # Execute and assert
        with pytest.raises(RuntimeError) as exc_info:
            initialize_client(baseToolArgs())
        assert (
            str(exc_info.value) == 'No valid AWS or basic authentication provided for OpenSearch'
        )
