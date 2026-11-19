# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for response size limiting functionality."""

import aiohttp
import asyncio
import contextlib
import logging
import os
import pytest
import tempfile
from mcp_server_opensearch.clusters_information import ClusterInfo
from opensearch.client import (
    ConfigurationError,
    _create_opensearch_client,
)
from opensearch.connection import (
    DEFAULT_MAX_RESPONSE_SIZE,
    BufferedAsyncHttpConnection,
    ResponseSizeExceededError,
)
from opensearchpy import AsyncHttpConnection, AsyncTransport
from opensearchpy.exceptions import ConnectionError, ConnectionTimeout, TransportError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestBufferedAsyncHttpConnection:
    """Test the BufferedAsyncHttpConnection class."""

    def test_init_default_max_response_size(self):
        """Test initialization with default max_response_size (None - no limit)."""
        connection = BufferedAsyncHttpConnection(host='localhost', port=9200, use_ssl=False)

        assert connection.max_response_size == DEFAULT_MAX_RESPONSE_SIZE
        assert connection.max_response_size is None  # No limit by default
        assert connection.host == 'http://localhost:9200'

    def test_init_custom_max_response_size(self):
        """Test initialization with custom max_response_size."""
        custom_size = 5 * 1024 * 1024  # 5MB
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=custom_size
        )

        assert connection.max_response_size == custom_size

    @pytest.mark.asyncio
    async def test_perform_request_fallback_to_parent(self):
        """Test that perform_request falls back to parent implementation on an unexpected error.

        A connection failure no longer falls back (see TestConnectionFailuresSendEachTryOnce),
        so the streaming request fails here with an error that is not one.
        """
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=1024
        )
        connection.session = MagicMock()
        connection.session.request = MagicMock(side_effect=RuntimeError('unexpected'))

        # Mock parent class perform_request to return a successful response
        with patch.object(connection.__class__.__bases__[0], 'perform_request') as mock_parent:
            mock_parent.return_value = (
                200,
                {'content-type': 'application/json'},
                '{"status": "ok"}',
            )

            # The streaming implementation will fail and fall back to parent
            status, headers, data = await connection.perform_request(method='GET', url='/test')

            assert status == 200
            assert data == '{"status": "ok"}'  # Should be string, not bytes
            mock_parent.assert_called_once()

    def test_response_decoding_logic(self):
        """Test the response decoding logic."""
        # Test UTF-8 decoding
        json_bytes = b'{"test": "value"}'
        try:
            decoded = json_bytes.decode('utf-8')
            assert decoded == '{"test": "value"}'
            assert isinstance(decoded, str)
        except UnicodeDecodeError:
            # Should not happen for valid UTF-8
            assert False, 'Valid UTF-8 should decode successfully'

        # Test binary data handling
        binary_data = b'\x89PNG\r\n\x1a\n'  # PNG header
        try:
            decoded = binary_data.decode('utf-8')
            # Should not reach here for binary data
            assert False, 'Binary data should not decode as UTF-8'
        except UnicodeDecodeError:
            # This is expected for binary data
            assert isinstance(binary_data, bytes)

    def test_response_size_exceeded_error_creation(self):
        """Test creating ResponseSizeExceededError with proper message."""
        max_size = 50
        actual_size = 100

        error = ResponseSizeExceededError(
            f'Response size exceeded limit of {max_size} bytes. '
            f'Stopped reading at {actual_size} bytes to prevent memory exhaustion. '
            f'Consider increasing max_response_size or refining your query to return less data.'
        )

        error_msg = str(error)
        assert 'Response size exceeded limit of 50 bytes' in error_msg
        assert 'Stopped reading at 100 bytes' in error_msg
        assert 'prevent memory exhaustion' in error_msg

    def test_connection_attributes(self):
        """Test that connection has proper attributes set."""
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=2048
        )

        assert connection.max_response_size == 2048
        assert connection.host == 'http://localhost:9200'
        assert hasattr(connection, 'perform_request')

    def test_ssl_connection_attributes(self):
        """Test SSL connection configuration."""
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9443, use_ssl=True, verify_certs=True, max_response_size=1024
        )

        assert connection.max_response_size == 1024
        assert connection.host == 'https://localhost:9443'
        # Test that SSL attributes are accessible
        assert getattr(connection, 'use_ssl', True) is True
        assert getattr(connection, 'verify_certs', True) is True

    def test_no_ssl_verification_attributes(self):
        """Test connection with SSL verification disabled."""
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9443, use_ssl=True, verify_certs=False, max_response_size=1024
        )

        assert connection.max_response_size == 1024
        assert getattr(connection, 'use_ssl', True) is True
        # Test that the connection was created successfully with SSL disabled verification
        assert connection.host == 'https://localhost:9443'

    def test_url_construction_with_params(self):
        """Test URL construction with parameters."""
        from urllib.parse import urlencode

        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=1024
        )

        # Test URL construction logic (similar to what perform_request does)
        base_url = connection.host + '/test'
        params = {'q': 'search term', 'size': 10}
        full_url = f'{base_url}?{urlencode(params)}'

        assert 'http://localhost:9200/test' in full_url
        assert 'q=search+term' in full_url or 'q=search%20term' in full_url
        assert 'size=10' in full_url

    def test_inheritance_structure(self):
        """Test that BufferedAsyncHttpConnection properly inherits from AsyncHttpConnection."""
        from opensearchpy import AsyncHttpConnection

        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=1024
        )

        assert isinstance(connection, AsyncHttpConnection)
        assert hasattr(connection, 'max_response_size')
        assert connection.max_response_size == 1024


class TestCreateOpenSearchClient:
    """Test the _create_opensearch_client function with max_response_size."""

    @patch('opensearch.client.AsyncOpenSearch')
    def test_create_client_with_max_response_size(self, mock_opensearch):
        """Test client creation with max_response_size parameter."""
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client

        _create_opensearch_client(
            opensearch_url='https://localhost:9200',
            opensearch_no_auth=True,  # Use no auth to avoid authentication complexity
            max_response_size=5242880,  # 5MB
        )

        # Verify AsyncOpenSearch was called with our custom connection class and max_response_size
        mock_opensearch.assert_called_once()
        call_kwargs = mock_opensearch.call_args[1]

        assert call_kwargs['connection_class'] == BufferedAsyncHttpConnection
        assert call_kwargs['max_response_size'] == 5242880
        assert call_kwargs['hosts'] == ['https://localhost:9200']

    @patch('opensearch.client.AsyncOpenSearch')
    def test_create_client_default_max_response_size(self, mock_opensearch):
        """Test client creation with default max_response_size."""
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client

        _create_opensearch_client(
            opensearch_url='https://localhost:9200',
            opensearch_no_auth=True,  # Use no auth to avoid authentication complexity
        )

        # Verify default max_response_size is used
        mock_opensearch.assert_called_once()
        call_kwargs = mock_opensearch.call_args[1]

        assert call_kwargs['max_response_size'] == DEFAULT_MAX_RESPONSE_SIZE

    @patch('opensearch.client.AsyncOpenSearch')
    def test_create_client_with_all_parameters(self, mock_opensearch):
        """Test client creation with all parameters including max_response_size."""
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client

        _create_opensearch_client(
            opensearch_url='https://test.com:9200',
            opensearch_username='user',
            opensearch_password='pass',
            opensearch_timeout=60,
            ssl_verify=False,
            max_response_size=1048576,  # 1MB
        )

        mock_opensearch.assert_called_once()
        call_kwargs = mock_opensearch.call_args[1]

        assert call_kwargs['connection_class'] == BufferedAsyncHttpConnection
        assert call_kwargs['max_response_size'] == 1048576
        assert call_kwargs['timeout'] == 60
        assert call_kwargs['verify_certs'] is False
        assert call_kwargs['http_auth'] == ('user', 'pass')

    @patch('opensearch.client.AsyncOpenSearch')
    def test_create_client_with_mtls_parameters(self, mock_opensearch):
        """Test client creation with CA, client cert, and client key."""
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client

        with (
            tempfile.NamedTemporaryFile() as ca_file,
            tempfile.NamedTemporaryFile() as cert_file,
            tempfile.NamedTemporaryFile() as key_file,
        ):
            _create_opensearch_client(
                opensearch_url='https://test.com:9200',
                opensearch_no_auth=True,
                opensearch_ca_cert_path=ca_file.name,
                opensearch_client_cert_path=cert_file.name,
                opensearch_client_key_path=key_file.name,
            )

        call_kwargs = mock_opensearch.call_args[1]
        assert call_kwargs['ca_certs'] == ca_file.name
        assert call_kwargs['client_cert'] == cert_file.name
        assert call_kwargs['client_key'] == key_file.name

    def test_create_client_rejects_partial_mtls_config(self):
        """Test client creation rejects missing client key when client cert is set."""
        with tempfile.NamedTemporaryFile() as cert_file:
            with pytest.raises(ConfigurationError) as exc_info:
                _create_opensearch_client(
                    opensearch_url='https://test.com:9200',
                    opensearch_no_auth=True,
                    opensearch_client_cert_path=cert_file.name,
                )

        assert 'requires both client certificate and client key paths' in str(exc_info.value)

    def test_create_client_rejects_missing_ca_file(self):
        """Test client creation rejects non-existent CA file paths."""
        missing_path = os.path.join(tempfile.gettempdir(), 'missing-ca.pem')
        if os.path.exists(missing_path):
            os.unlink(missing_path)

        with pytest.raises(ConfigurationError) as exc_info:
            _create_opensearch_client(
                opensearch_url='https://test.com:9200',
                opensearch_no_auth=True,
                opensearch_ca_cert_path=missing_path,
            )

        assert 'CA certificate file does not exist' in str(exc_info.value)


class TestClusterInfoMaxResponseSize:
    """Test ClusterInfo with max_response_size field."""

    def test_cluster_info_with_max_response_size(self):
        """Test ClusterInfo creation with max_response_size."""
        cluster_info = ClusterInfo(
            opensearch_url='https://test.com:9200',
            max_response_size=2097152,  # 2MB
        )

        assert cluster_info.opensearch_url == 'https://test.com:9200'
        assert cluster_info.max_response_size == 2097152

    def test_cluster_info_without_max_response_size(self):
        """Test ClusterInfo creation without max_response_size (should be None)."""
        cluster_info = ClusterInfo(opensearch_url='https://test.com:9200')

        assert cluster_info.opensearch_url == 'https://test.com:9200'
        assert cluster_info.max_response_size is None

    def test_cluster_info_all_fields_including_max_response_size(self):
        """Test ClusterInfo with all fields including max_response_size."""
        cluster_info = ClusterInfo(
            opensearch_url='https://test.com:9200',
            opensearch_username='admin',
            opensearch_password='password',
            timeout=30,
            ssl_verify=True,
            max_response_size=5242880,  # 5MB
        )

        assert cluster_info.opensearch_url == 'https://test.com:9200'
        assert cluster_info.opensearch_username == 'admin'
        assert cluster_info.opensearch_password == 'password'
        assert cluster_info.timeout == 30
        assert cluster_info.ssl_verify is True
        assert cluster_info.max_response_size == 5242880


class TestResponseSizeExceededError:
    """Test the ResponseSizeExceededError exception."""

    def test_exception_creation(self):
        """Test creating ResponseSizeExceededError."""
        error = ResponseSizeExceededError('Test error message')

        assert str(error) == 'Test error message'
        assert isinstance(error, Exception)

    def test_exception_inheritance(self):
        """Test that ResponseSizeExceededError inherits from OpenSearchClientError."""
        from opensearch.client import OpenSearchClientError

        error = ResponseSizeExceededError('Test error')

        assert isinstance(error, OpenSearchClientError)
        assert isinstance(error, Exception)

    def test_exception_with_detailed_message(self):
        """Test exception with detailed message format."""
        max_size = 1024
        actual_size = 2048

        message = (
            f'Response size exceeded limit of {max_size} bytes. '
            f'Stopped reading at {actual_size} bytes to prevent memory exhaustion. '
            f'Consider increasing max_response_size or refining your query to return less data.'
        )

        error = ResponseSizeExceededError(message)

        assert 'exceeded limit of 1024 bytes' in str(error)
        assert 'Stopped reading at 2048 bytes' in str(error)
        assert 'prevent memory exhaustion' in str(error)
        assert 'Consider increasing max_response_size' in str(error)


class TestRedirectsDisabled:
    """Redirects are refused only for a caller URL under the SSRF guard.

    An operator's own URL keeps following them, since a proxy in front of a
    cluster may legitimately redirect.
    """

    @pytest.mark.asyncio
    async def test_caller_url_under_guard_does_not_follow_redirects(self):
        import contextlib
        from unittest.mock import AsyncMock

        captured = {}

        response = MagicMock()

        async def chunk_iter(_size):
            for _ in range(1):
                yield b'ok'

        response.content.iter_chunked = chunk_iter
        response.headers.getall = MagicMock(return_value=())
        response.status = 200

        @contextlib.asynccontextmanager
        async def fake_request(*args, **kwargs):
            captured.update(kwargs)
            yield response

        conn = BufferedAsyncHttpConnection(
            host='localhost',
            port=9200,
            use_ssl=False,
            max_response_size=1024,
            follow_redirects=False,
        )
        conn.session = MagicMock()
        conn.session.request = fake_request
        conn._create_aiohttp_session = AsyncMock()
        conn.loop = MagicMock()
        conn.loop.time = MagicMock(return_value=0.0)
        conn._http_auth = None

        await conn.perform_request(method='GET', url='/test')

        assert captured.get('allow_redirects') is False

    @pytest.mark.asyncio
    async def test_operator_url_still_follows_redirects(self):
        """A proxy in front of an operator's cluster may redirect, so keep following."""
        import contextlib
        from unittest.mock import AsyncMock

        captured = {}

        response = MagicMock()

        async def chunk_iter(_size):
            yield b'ok'

        response.content.iter_chunked = chunk_iter
        response.headers.getall = MagicMock(return_value=())
        response.status = 200

        @contextlib.asynccontextmanager
        async def fake_request(*_args, **kwargs):
            captured.update(kwargs)
            yield response

        conn = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=1024
        )
        conn.session = MagicMock()
        conn.session.request = fake_request
        conn._create_aiohttp_session = AsyncMock()
        conn.loop = MagicMock()
        conn.loop.time = MagicMock(return_value=0.0)
        conn._http_auth = None

        await conn.perform_request(method='GET', url='/test')

        assert captured.get('allow_redirects') is True

    @pytest.mark.asyncio
    async def test_no_fallback_when_redirects_must_be_refused(self):
        """The parent always follows redirects, so falling back would undo the refusal."""
        import contextlib
        from unittest.mock import AsyncMock

        @contextlib.asynccontextmanager
        async def broken_request(*_args, **_kwargs):
            raise OSError('connection reset mid-body')
            yield  # pragma: no cover

        conn = BufferedAsyncHttpConnection(
            host='localhost',
            port=9200,
            use_ssl=False,
            max_response_size=1024,
            follow_redirects=False,
        )
        conn.session = MagicMock()
        conn.session.request = broken_request
        conn._create_aiohttp_session = AsyncMock()
        conn.loop = MagicMock()
        conn.loop.time = MagicMock(return_value=0.0)
        conn._http_auth = None

        with patch.object(conn.__class__.__bases__[0], 'perform_request') as mock_parent:
            with pytest.raises(OSError):
                await conn.perform_request(method='GET', url='/test')
            mock_parent.assert_not_called()

    @pytest.mark.asyncio
    async def test_redirect_is_not_retried_through_parent(self):
        """Retrying a redirect through the parent would follow it, undoing the refusal."""
        import contextlib
        from opensearchpy.exceptions import TransportError
        from unittest.mock import AsyncMock

        response = MagicMock()

        async def chunk_iter(_size):
            yield b'moved'

        response.content.iter_chunked = chunk_iter
        response.headers.getall = MagicMock(return_value=())
        response.status = 302

        @contextlib.asynccontextmanager
        async def fake_request(*_args, **_kwargs):
            yield response

        conn = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=1024
        )
        conn.session = MagicMock()
        conn.session.request = fake_request
        conn._create_aiohttp_session = AsyncMock()
        conn.loop = MagicMock()
        conn.loop.time = MagicMock(return_value=0.0)
        conn._http_auth = None

        with patch.object(conn.__class__.__bases__[0], 'perform_request') as mock_parent:
            with pytest.raises(TransportError):
                await conn.perform_request(method='GET', url='/test')
            mock_parent.assert_not_called()


class TestIntegrationScenarios:
    """Integration tests for various response size limiting scenarios."""

    def test_size_limit_calculation(self):
        """Test size limit calculation logic."""
        # Test that we can calculate when a response would exceed limits
        max_size = 100
        chunk1_size = 50
        chunk2_size = 60  # This would exceed the limit

        total_size = chunk1_size
        # First chunk is within limit
        assert total_size <= max_size

        # Second chunk would exceed limit
        assert total_size + chunk2_size > max_size

        # This simulates the logic in perform_request
        would_exceed = (total_size + chunk2_size) > max_size
        assert would_exceed is True

    def test_chunk_processing_logic(self):
        """Test the chunk processing logic used in streaming."""
        chunks = [b'x' * 30, b'y' * 40, b'z' * 50]  # 30, 40, 50 bytes
        max_size = 100

        processed_chunks = []
        total_size = 0

        for chunk in chunks:
            if total_size + len(chunk) > max_size:
                # This simulates stopping when limit would be exceeded
                break
            processed_chunks.append(chunk)
            total_size += len(chunk)

        # Should process first two chunks (70 bytes total)
        assert len(processed_chunks) == 2
        assert total_size == 70
        assert total_size <= max_size

    def test_large_limit_handling(self):
        """Test behavior with very large limits."""
        large_limit = 100 * 1024 * 1024  # 100MB
        connection = BufferedAsyncHttpConnection(
            host='localhost', port=9200, use_ssl=False, max_response_size=large_limit
        )

        assert connection.max_response_size == large_limit

        # Simulate a response much smaller than the limit
        simulated_response_size = 20 * 1024  # 20KB
        assert simulated_response_size < large_limit


class _FailingSession:
    """Stands in for ``aiohttp.ClientSession``: records each request and fails it.

    ``fail(kind)`` returns the exception a request of that kind raises, or ``None`` to answer
    it. The streaming request passes ``allow_redirects``; the parent's does not, so the kind
    is ``'streaming'`` or ``'parent'``. ``stage`` is where the failure happens: on connect
    (``__aenter__``) or while the body is read.
    """

    def __init__(self, fail, stage='connect'):
        self.fail = fail
        self.stage = stage
        self.requests = []

    def request(self, method, url, **kwargs):
        kind = 'streaming' if 'allow_redirects' in kwargs else 'parent'
        self.requests.append(kind)
        error = self.fail(kind)

        @contextlib.asynccontextmanager
        async def exchange():
            if error is not None and self.stage == 'connect':
                raise error
            yield self._response(error)

        return exchange()

    @staticmethod
    def _response(error):
        async def chunks(_size):
            if error is not None:
                raise error
            yield b'[]'

        response = MagicMock()
        response.status = 200
        response.headers.getall = MagicMock(return_value=())
        response.content.iter_chunked = chunks
        response.text = AsyncMock(side_effect=error, return_value='[]')
        return response

    async def close(self):
        pass


def _transport(connection_class, session, **kwargs):
    """An ``AsyncTransport`` (four tries by default) whose connections all use ``session``."""

    class _Connection(connection_class):
        async def _create_aiohttp_session(self):
            self.session = session

    return AsyncTransport(
        [{'host': 'localhost', 'port': 9200}], connection_class=_Connection, **kwargs
    )


async def _raised(transport):
    with pytest.raises(TransportError) as info:
        await transport.perform_request('GET', '/_cat/indices')
    await transport.close()
    return info.value


def _error_events(caplog):
    """The structured ``opensearch_request`` error events, as the fallback logged them."""
    return [
        (r.levelname, r.http_method, r.endpoint, r.error, hasattr(r, 'status_code'))
        for r in caplog.records
        if getattr(r, 'event_type', None) == 'opensearch_request' and r.status == 'error'
    ]


def _refused():
    key = SimpleNamespace(host='localhost', port=9200, ssl=True)
    return aiohttp.ClientConnectorError(key, ConnectionRefusedError(111, 'Connect call failed'))


CONNECTION_FAILURES = {
    'connection_refused': _refused,
    'connection_reset': lambda: aiohttp.ClientConnectionError('connection reset'),
    'fingerprint_mismatch': lambda: aiohttp.ServerFingerprintMismatch(
        b'expected', b'got', 'localhost', 9200
    ),
}

TIMEOUTS = {
    'asyncio_timeout': asyncio.TimeoutError,
    'aiohttp_server_timeout': lambda: aiohttp.ServerTimeoutError('Timeout on reading data'),
}


class TestConnectionFailuresSendEachTryOnce:
    """A connection failure on the streaming path is sent once per transport try, not twice.

    It raises the ``TransportError`` the parent ``AsyncHttpConnection`` raises for it (the
    reference: today the fallback calls the parent, which raises that error), so the
    transport retries it without the fallback re-sending each try.
    """

    @pytest.mark.parametrize('make_error', CONNECTION_FAILURES.values(), ids=CONNECTION_FAILURES)
    async def test_connection_refused_makes_one_request_per_try(self, make_error, caplog):
        caplog.set_level(logging.WARNING)
        session = _FailingSession(lambda _kind: make_error())
        raised = await _raised(_transport(BufferedAsyncHttpConnection, session))

        reference_session = _FailingSession(lambda _kind: make_error())
        expected = await _raised(_transport(AsyncHttpConnection, reference_session))

        # Four tries, one streaming request each; the parent's request is never sent.
        assert session.requests == ['streaming'] * 4
        assert len(reference_session.requests) == 4
        assert type(raised) is type(expected)
        assert str(raised) == str(expected)
        assert isinstance(raised, ConnectionError)
        assert 'falling back to parent implementation' not in caplog.text
        # The error event the fallback logged for each try is still logged, once per try.
        assert _error_events(caplog) == [('ERROR', 'GET', '/_cat/indices', str(raised), False)] * 4
        # log_request_fail is called once per try, as the parent does.
        failures = [r for r in caplog.records if r.name == 'opensearch']
        assert len(failures) == 8  # four from this transport, four from the reference
        assert all(
            r.getMessage().startswith('GET http://localhost:9200/_cat/indices [status:N/A')
            for r in failures
        )

    def test_connection_refused_message_is_the_parents(self):
        error = _refused()
        assert str(ConnectionError('N/A', str(error), error)) == (
            'ConnectionError(Cannot connect to host localhost:9200 ssl:default '
            '[Connect call failed]) caused by: ClientConnectorError(Cannot connect to host '
            'localhost:9200 ssl:default [Connect call failed])'
        )

    @pytest.mark.parametrize('stage', ['connect', 'body'])
    @pytest.mark.parametrize('retry_on_timeout, tries', [(False, 1), (True, 4)])
    @pytest.mark.parametrize('make_error', TIMEOUTS.values(), ids=TIMEOUTS)
    async def test_timeout_makes_one_request_per_try(
        self, make_error, retry_on_timeout, tries, stage, caplog
    ):
        caplog.set_level(logging.WARNING, logger='opensearch.connection')
        session = _FailingSession(lambda _kind: make_error(), stage)
        raised = await _raised(
            _transport(BufferedAsyncHttpConnection, session, retry_on_timeout=retry_on_timeout)
        )

        reference_session = _FailingSession(lambda _kind: make_error(), stage)
        expected = await _raised(
            _transport(AsyncHttpConnection, reference_session, retry_on_timeout=retry_on_timeout)
        )

        assert session.requests == ['streaming'] * tries
        assert len(reference_session.requests) == tries
        assert type(raised) is type(expected) is ConnectionTimeout
        assert str(raised) == str(expected)
        assert 'falling back to parent implementation' not in caplog.text
        expected_event = ('ERROR', 'GET', '/_cat/indices', str(raised), False)
        assert _error_events(caplog) == [expected_event] * tries

    async def test_unexpected_exception_still_falls_back(self, caplog):
        caplog.set_level(logging.WARNING, logger='opensearch.connection')
        session = _FailingSession(
            lambda kind: RuntimeError('not a connection failure') if kind == 'streaming' else None
        )
        transport = _transport(BufferedAsyncHttpConnection, session)

        assert await transport.perform_request('GET', '/_cat/indices') == []
        await transport.close()

        # One try: the streaming request fails, the parent's answers.
        assert session.requests == ['streaming', 'parent']
        assert (
            'Streaming request failed (RuntimeError: not a connection failure), '
            'falling back to parent implementation'
        ) in caplog.text

    async def test_connection_failure_without_redirects_is_still_reraised(self):
        """The redirect refusal is unchanged: a failure re-raises as it is, without retries."""
        session = _FailingSession(lambda _kind: _refused())

        class _Connection(BufferedAsyncHttpConnection):
            async def _create_aiohttp_session(self):
                self.session = session

        transport = AsyncTransport(
            [{'host': 'localhost', 'port': 9200}],
            connection_class=_Connection,
            follow_redirects=False,
        )
        with pytest.raises(aiohttp.ClientConnectorError):
            await transport.perform_request('GET', '/_cat/indices')
        await transport.close()

        assert session.requests == ['streaming']
