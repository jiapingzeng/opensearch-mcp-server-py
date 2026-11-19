# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the characterization (golden) tests.

The tests in this directory pin the server's observable behaviour through its
public entry points and the MCP protocol. This module provides:

- ``golden``: compare canonical JSON with a committed file in ``golden/``.
- ``normalize``: replace volatile values (timestamps, ids, ports, ...) with placeholders.
- ``fake_opensearch``: a fake OpenSearch behind the HTTP session the server's
  ``AsyncOpenSearch`` connections use, with a route table and a request log.
- ``pinned_openapi`` / ``openapi_unavailable``: committed OpenAPI fixtures instead of GitHub.
- ``isolated_state`` (autouse): resets and restores every piece of module state
  the server mutates, and the environment it reads.
- ``no_network`` (autouse): any DNS lookup or connection off this machine fails the test.
- ``mcp_session``: build the server with ``create_mcp_server`` and talk to it in-process.
"""

import aiohttp
import anyio
import asyncio
import base64
import contextlib
import difflib
import errno
import gzip
import ipaddress
import json
import logging
import os
import pytest
import re
import socket
import sys
import tempfile
import yaml
import yarl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from mcp.client import Client
from mcp.client._memory import InMemoryTransport
from mcp.shared.message import SessionMessage
from mcp_server_opensearch import (
    clusters_information,
    global_state,
    logging_config,
    server_instructions,
    stdio_server,
    streaming_server,
    tool_executor,
)
from mcp_server_opensearch.client_context import client_name_var, request_context_var
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from multidict import CIMultiDict, CIMultiDictProxy
from opensearchpy._async.http_aiohttp import AIOHttpConnection
from opensearchpy.connection.http_async import AsyncHttpConnection
from pathlib import Path, PurePosixPath
from pydantic import BaseModel
from tools import config as tool_config
from tools import tool_filter, tool_generator
from tools.agentic_memory.actions import AGENTIC_MEMORY_TOOLS_REGISTRY
from tools.memory_tools import MEMORY_TOOLS_REGISTRY
from tools.skills_tools import SKILLS_TOOLS_REGISTRY
from tools.tool_params import baseToolArgs
from tools.tools import TOOL_REGISTRY
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / 'golden'
OPENAPI_FIXTURE_DIR = HERE / 'fixtures' / 'openapi'

# `.invalid` never resolves, so a request that slipped past the fake could not reach a host.
FAKE_OPENSEARCH_URL = 'http://fake-opensearch.invalid:9200'
DEFAULT_OPENSEARCH_VERSION = '3.1.0'

# Protocol version `raw()` uses when none is given: the newest initialize-handshake version.
DEFAULT_PROTOCOL_VERSION = '2025-11-25'
HARNESS_CLIENT_INFO = {'name': 'characterization-harness', 'version': '0.0.0'}
RAW_TIMEOUT_SECONDS = 30

GOLDEN_WRITE_ENV = 'CHAR_GOLDEN_WRITE'


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


def canonical_json(data: Any) -> str:
    """Serialize ``data`` the way every golden file is written."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + '\n'


class Golden:
    """Compare data with ``<directory>/<name>.json``; create it only when missing and allowed."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, name: str) -> Path:
        parts = PurePosixPath(name).parts
        # "/" is the only separator, so a name maps to the same file on every OS.
        if (
            not parts
            or name.startswith('/')
            or '..' in parts
            or name.endswith('.json')
            or any(char in name for char in '\\:')
        ):
            raise ValueError(f'invalid golden name {name!r}: use a relative name like "area/case"')
        return self.directory.joinpath(*parts).with_name(parts[-1] + '.json')

    def __call__(self, name: str, data: Any) -> None:
        path = self.path(name)
        actual = canonical_json(data)
        if not path.exists():
            if os.environ.get(GOLDEN_WRITE_ENV) != '1':
                pytest.fail(
                    f'golden file {path} does not exist. If this is a new characterization, '
                    f'create it with {GOLDEN_WRITE_ENV}=1, review it and commit it.',
                    pytrace=False,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            # Mode 'x' refuses to replace a file, so a golden is never overwritten.
            with open(path, 'x', encoding='utf-8', newline='\n') as f:
                f.write(actual)
            return
        # Read untranslated (newline='\n'): the comparison is byte for byte on every OS.
        with open(path, encoding='utf-8', newline='\n') as f:
            expected = f.read()
        if expected == actual:
            return
        if expected.replace('\r\n', '\n') == actual:
            pytest.fail(
                f'golden {path} has CRLF line endings; goldens are LF-only. The checkout '
                'converted them: tests/characterization/.gitattributes sets eol=lf, so check '
                'that it is in place and re-checkout the file.',
                pytrace=False,
            )
        diff = ''.join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f'{name}.json (golden)',
                tofile=f'{name}.json (actual)',
            )
        )
        pytest.fail(
            f'behaviour differs from golden {path}.\n'
            'A golden diff is a behaviour change: fix the code, or, if the change is intended, '
            f'delete the golden in a visible diff and recreate it ({GOLDEN_WRITE_ENV}=1 never '
            f'overwrites an existing file).\n{diff}',
            pytrace=False,
        )


@pytest.fixture
def golden() -> Golden:
    """``golden(name, data)``: assert ``data`` matches ``golden/<name>.json``."""
    return Golden(GOLDEN_DIR)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class Normalizer:
    """Replace volatile values with fixed placeholders; leave everything else untouched.

    Each rule targets a value that differs between runs, and is narrow on purpose (a
    timestamp only counts when it is close to now, a port only when it is ephemeral), so
    static text such as example dates or ports in tool descriptions is never rewritten.
    """

    # Timestamps count as volatile only within this distance of the moment of normalizing.
    RECENT = timedelta(days=2)

    # Fields of the server's structured log events (dicts with an `event_type`; see
    # tool_executor.py, opensearch/connection.py, logging_config.py) that vary per run.
    EVENT_FIELD_RULES = {
        # Durations: tool_execution / opensearch_request timing.
        'duration_ms': '<DURATION>',
        # Memory figures: memory_snapshot's RSS and the pid it reports.
        'memory_rss_mb': '<MEMORY>',
        'pid': '<PID>',
    }

    # Session ids: the streamable-HTTP session header (any case), wherever headers are kept.
    SESSION_HEADER = 'mcp-session-id'

    # Timestamps: ISO 8601 / RFC 3339 and logging's asctime ("2026-11-19 10:00:00,123").
    ISO_TIMESTAMP = re.compile(
        r'\b(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?'
    )
    # Timestamps: HTTP dates ("Wed, 19 Nov 2026 10:00:00 GMT").
    HTTP_DATE = re.compile(
        r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) '
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT'
    )
    MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

    # Applied in order to every string value after the timestamp rules (keys are left alone).
    STRING_RULES = [
        # UUIDs in canonical 8-4-4-4-12 form.
        (
            re.compile(
                r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
            ),
            '<UUID>',
        ),
        # Session ids: `uuid4().hex`, the form the SDK mints MCP session ids in.
        (re.compile(r'\b[0-9a-f]{32}\b'), '<SESSION_ID>'),
        # Durations as the server's log messages print them: "Tool executed: X (12.34ms)".
        (re.compile(r'\(\d+(?:\.\d+)?ms\)'), '(<DURATION>ms)'),
        # Memory figures as the memory monitor prints them: "Memory snapshot: 9.5 MB (pid=42)".
        (
            re.compile(r'(Memory snapshot: )-?\d+(?:\.\d+)? MB \(pid=\d+\)'),
            r'\1<MEMORY> MB (pid=<PID>)',
        ),
    ]

    # Ports: a loopback/any-address host with a port in the ephemeral range (Linux and
    # macOS; Windows' 49152-65535 lies inside it), which is what a test gets when it binds
    # port 0.
    LOOPBACK_PORT = re.compile(r'(?<![\w.])(127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]):(\d{5})\b')
    EPHEMERAL_PORTS = range(32768, 65536)

    # A Windows path starts with a drive letter ("C:") or a UNC prefix ("\\").
    WINDOWS_PATH = re.compile(r'[A-Za-z]:|\\\\')

    def __init__(
        self,
        pytest_basetemp: os.PathLike | str | None = None,
        temp_dirs: list[str] | None = None,
    ):
        # Temp paths: pytest's per-test directory (the base temp plus one component) and
        # tempfile.mkdtemp/mkstemp names ("tmp" + 8 characters) in the system temp directory
        # (as named, and resolved: macOS's /var -> /private/var, Windows' 8.3 short names)
        # become <TMP>. The path below them (the file a test created) is kept.
        # Windows paths match "\" or "/" as the separator (runs of them too, as in
        # JSON-escaped text), case-insensitively as Windows compares paths, and the path kept
        # below <TMP> is written with "/", so a golden reads the same on every OS.
        if temp_dirs is None:
            system_tmp = tempfile.gettempdir()
            temp_dirs = [system_tmp, os.path.realpath(system_tmp)]
        rules = []
        if pytest_basetemp is not None:
            rules.append(self._temp_rule(str(pytest_basetemp), r'(?:{sep}{name})?'))
        for tmp in temp_dirs:
            rules.append(self._temp_rule(str(tmp), r'{sep}tmp[A-Za-z0-9_]{{8}}'))
        # Longest base first, so /private/var/... is replaced before /var/... can match in it.
        rules.sort(key=lambda rule: len(rule[0]), reverse=True)
        self._temp_rules = [(pattern, replacement) for _, pattern, replacement in rules]
        self._now = datetime.now(timezone.utc).replace(tzinfo=None)

    @classmethod
    def _temp_rule(cls, base: str, child: str) -> tuple[str, re.Pattern, Any]:
        """``(base, pattern, replacement)`` turning ``base`` + ``child`` into ``<TMP>``.

        ``child`` is a regex template: ``{sep}`` is a path separator, ``{name}`` one path
        component.
        """
        base = base.rstrip('\\/') or base
        if not cls.WINDOWS_PATH.match(base):
            child = child.format(sep='/', name=r'[^/\s\'"]+')
            return base, re.compile(re.escape(base) + child), '<TMP>'
        sep, name = r'[\\/]+', r'[^\\/\s\'"]+'
        prefix = sep.join(re.escape(part) for part in re.split(r'[\\/]+', base))
        # The path below <TMP>, captured so its separators can be rewritten as "/".
        below = f'((?:{sep}{name})*)'
        pattern = re.compile(prefix + child.format(sep=sep, name=name) + below, re.IGNORECASE)
        return base, pattern, lambda match: '<TMP>' + re.sub(sep, '/', match.group(1))

    def __call__(self, data: Any) -> Any:
        self._now = datetime.now(timezone.utc).replace(tzinfo=None)
        return self._walk(data)

    def _walk(self, value: Any, key: Any = None) -> Any:
        if isinstance(value, dict):
            return {k: self._entry(value, k, v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            items = [self._walk(v) for v in value]
            # JSON-schema `required` lists: tool_generator builds them from a Python set, so
            # their order changes with PYTHONHASHSEED. Order carries no meaning there; sort.
            if key == 'required' and all(isinstance(v, str) for v in items):
                items.sort()
            return type(value)(items)
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return self._epoch(value)
        return value

    def _entry(self, parent: dict, key: Any, value: Any) -> Any:
        if 'event_type' in parent and key in self.EVENT_FIELD_RULES:
            return self.EVENT_FIELD_RULES[key]
        if isinstance(key, str) and key.lower() == self.SESSION_HEADER:
            return '<SESSION_ID>'
        return self._walk(value, key)

    def _string(self, text: str) -> str:
        for pattern, replacement in self._temp_rules:
            text = pattern.sub(replacement, text)
        text = self.ISO_TIMESTAMP.sub(self._iso_timestamp, text)
        text = self.HTTP_DATE.sub(self._http_date, text)
        for pattern, replacement in self.STRING_RULES:
            text = pattern.sub(replacement, text)
        return self.LOOPBACK_PORT.sub(self._port, text)

    def _recent(self, moment: datetime) -> bool:
        # Naive comparison; the two-day window absorbs any local-time/UTC offset.
        return abs(moment - self._now) <= self.RECENT

    def _iso_timestamp(self, match: re.Match) -> str:
        try:
            moment = datetime(*(int(part) for part in match.groups()))
        except ValueError:
            return match.group(0)
        return '<TIMESTAMP>' if self._recent(moment) else match.group(0)

    def _http_date(self, match: re.Match) -> str:
        day, month, year, hour, minute, second = match.groups()
        month_number = self.MONTHS.index(month) + 1
        try:
            moment = datetime(
                int(year), month_number, int(day), int(hour), int(minute), int(second)
            )
        except ValueError:
            return match.group(0)
        return '<TIMESTAMP>' if self._recent(moment) else match.group(0)

    def _epoch(self, number: int) -> Any:
        # Timestamps: epoch seconds or milliseconds (e.g. memory search's `now_ms`).
        for per_second in (1, 1000):
            try:
                moment = datetime.fromtimestamp(number / per_second, timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue
            if self._recent(moment.replace(tzinfo=None)):
                return '<TIMESTAMP>'
        return number

    def _port(self, match: re.Match) -> str:
        if int(match.group(2)) in self.EPHEMERAL_PORTS:
            return f'{match.group(1)}:<PORT>'
        return match.group(0)


@pytest.fixture
def normalize(tmp_path_factory) -> Normalizer:
    """``normalize(data)``: a copy of ``data`` with volatile values replaced by placeholders."""
    return Normalizer(tmp_path_factory.getbasetemp())


# ---------------------------------------------------------------------------
# fake_opensearch
# ---------------------------------------------------------------------------


class UnmatchedFakeRequestError(AssertionError):
    """A request reached the fake OpenSearch with no route for it."""


@dataclass
class _Route:
    method: str
    path: str | re.Pattern
    status: int = 200
    body: Any = None
    headers: dict | None = None
    connection_error: str | None = None

    def matches(self, method: str, path: str) -> bool:
        if method != self.method:
            return False
        if isinstance(self.path, re.Pattern):
            return self.path.fullmatch(path) is not None
        return path == self.path


class _FakeContent:
    def __init__(self, body: bytes):
        self._body = body

    async def iter_chunked(self, size: int):
        for start in range(0, len(self._body), size):
            yield self._body[start : start + size]


class _FakeResponse:
    """The parts of ``aiohttp.ClientResponse`` the OpenSearch connections read."""

    def __init__(self, status: int, body: bytes, headers: dict):
        self.status = status
        self.headers = CIMultiDictProxy(CIMultiDict(headers))
        self.content = _FakeContent(body)
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def text(self, encoding: Any = None, errors: str = 'strict') -> str:
        return self._body.decode('utf-8', 'surrogatepass')

    def release(self) -> None:
        pass


class _FakeExchange:
    """What ``session.request(...)`` returns: an async context manager yielding the response."""

    def __init__(self, fake: 'FakeOpenSearch', method, url, data, auth, headers):
        self._fake = fake
        self._args = (method, url, data, auth, headers)

    async def __aenter__(self) -> _FakeResponse:
        return self._fake._exchange(*self._args)

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeAiohttpSession:
    """Stands in for the ``aiohttp.ClientSession`` an OpenSearch connection creates."""

    def __init__(self, fake: 'FakeOpenSearch'):
        self._fake = fake
        self.closed = False

    def request(self, method, url, data=None, auth=None, headers=None, **_transport_options):
        return _FakeExchange(self._fake, method, url, data, auth, headers)

    async def close(self) -> None:
        self.closed = True


def _decode_body(data: Any, headers: dict) -> tuple[str, Any]:
    if data is None or data == b'' or data == '':
        return 'none', None
    if isinstance(data, (bytes, bytearray)):
        if str(headers.get('content-encoding', '')).lower() == 'gzip':
            data = gzip.decompress(data)
        text = bytes(data).decode('utf-8', 'surrogatepass')
    else:
        text = str(data)
    try:
        return 'json', json.loads(text)
    except ValueError:
        pass
    try:
        return 'ndjson', [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError:
        return 'text', text


def _auth_kind(auth: Any, headers: dict) -> dict:
    """Describe how a request authenticated, without any secret (no password, token, key)."""
    if isinstance(auth, aiohttp.BasicAuth):
        return {'kind': 'basic', 'username': auth.login}
    value = next((v for k, v in headers.items() if k.lower() == 'authorization'), None)
    if not value:
        return {'kind': 'none'}
    scheme, _, rest = str(value).partition(' ')
    scheme = scheme.lower()
    if scheme == 'basic':
        try:
            username = base64.b64decode(rest.strip()).decode('utf-8').split(':', 1)[0]
        except ValueError:
            username = None
        return {'kind': 'basic', 'username': username}
    if scheme == 'bearer':
        return {'kind': 'bearer'}
    if scheme == 'aws4-hmac-sha256':
        # Credential=<access key>/<date>/<region>/<service>/aws4_request: keep region and service.
        match = re.search(r'Credential=[^/,\s]+/\d{8}/([^/,\s]+)/([^/,\s]+)/aws4_request', rest)
        return {
            'kind': 'sigv4',
            'region': match.group(1) if match else None,
            'service': match.group(2) if match else None,
        }
    return {'kind': 'other', 'scheme': scheme}


def _encode_reply(body: Any, headers: dict | None) -> tuple[bytes, dict]:
    if isinstance(body, (bytes, bytearray)):
        payload, content_type = bytes(body), 'application/octet-stream'
    elif isinstance(body, str):
        payload, content_type = body.encode('utf-8'), 'text/plain; charset=UTF-8'
    elif body is None:
        payload, content_type = b'', None
    else:
        payload = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        content_type = 'application/json; charset=UTF-8'
    reply_headers = {'content-type': content_type} if content_type else {}
    reply_headers.update(headers or {})
    return payload, reply_headers


def _opensearch_error(status: int, error_type: str, reason: str) -> dict:
    cause = {'type': error_type, 'reason': reason}
    return {'error': {'root_cause': [cause], **cause}, 'status': status}


class FakeOpenSearch:
    """A fake OpenSearch cluster answering every ``AsyncOpenSearch`` connection in the test.

    Routes map (method, path) to a canned reply. ``path`` is matched exactly against the
    decoded URL path, or with ``fullmatch`` when it is a compiled regex; the newest matching
    route wins. A request with no route is recorded in ``unmatched``, fails inside the server
    and fails the test at teardown. ``GET /`` answers ``version`` unless a route overrides it;
    ``version = None`` makes it fail, so the server sees no version.
    """

    def __init__(self, url: str = FAKE_OPENSEARCH_URL):
        self.url = url
        self.version: str | None = DEFAULT_OPENSEARCH_VERSION
        self.requests: list[dict] = []
        self.unmatched: list[dict] = []
        self._routes: list[_Route] = []

    def reply(
        self,
        method: str,
        path: str | re.Pattern,
        body: Any,
        *,
        status: int = 200,
        headers: dict | None = None,
    ) -> None:
        """Answer ``method path`` with ``body``.

        ``body`` may be JSON data (sent as application/json), a str (text/plain), bytes, or
        a callable receiving the recorded request and returning one of those.
        """
        self._routes.append(_Route(method.upper(), path, status, body, headers))

    def error(
        self,
        method: str,
        path: str | re.Pattern,
        status: int,
        body: Any = None,
        *,
        headers: dict | None = None,
    ) -> None:
        """Answer ``method path`` with HTTP ``status``; ``body`` defaults to an OpenSearch error."""
        if body is None:
            body = _opensearch_error(status, 'fake_error', f'fake error for {method} {path}')
        self.reply(method, path, body, status=status, headers=headers)

    def fail_connection(
        self, method: str, path: str | re.Pattern, message: str = 'fake: connection refused'
    ) -> None:
        """Make ``method path`` fail as if the connection broke before any response."""
        self._routes.append(_Route(method.upper(), path, connection_error=message))

    def verify(self) -> None:
        """Fail the test if any request had no route."""
        if self.unmatched:
            listing = '\n'.join(
                f'  {r["method"]} {r["path"]} {r["params"]}' for r in self.unmatched
            )
            pytest.fail(
                f'fake OpenSearch received requests with no route:\n{listing}\n'
                'Add a route with fake_opensearch.reply(...) or .error(...).',
                pytrace=False,
            )

    def _record(self, method: str, url: yarl.URL, data: Any, auth: Any, headers: dict) -> dict:
        body_format, body = _decode_body(data, headers)
        return {
            'method': method,
            'host': f'{url.scheme}://{url.host}:{url.port}',
            'path': url.path,
            'params': dict(url.query),
            'body_format': body_format,
            'body': body,
            'auth': _auth_kind(auth, headers),
        }

    def _exchange(self, method, url, data, auth, headers) -> _FakeResponse:
        url = url if isinstance(url, yarl.URL) else yarl.URL(str(url))
        method = str(method).upper()
        record = self._record(method, url, data, auth, dict(headers or {}))
        self.requests.append(record)
        route = next((r for r in reversed(self._routes) if r.matches(method, url.path)), None)
        if route is None and (method, url.path) == ('GET', '/'):
            route = self._version_route()
        if route is None:
            self.unmatched.append(record)
            raise UnmatchedFakeRequestError(
                f'unmatched fake OpenSearch request: {method} {url.path} {record["params"]}'
            )
        if route.connection_error is not None:
            raise aiohttp.ClientConnectionError(route.connection_error)
        body = route.body(record) if callable(route.body) else route.body
        payload, reply_headers = _encode_reply(body, route.headers)
        return _FakeResponse(route.status, payload, reply_headers)

    def _version_route(self) -> _Route:
        if self.version is None:
            # 500, not 503: opensearch-py retries 502/503/504, which would add requests.
            return _Route(
                'GET',
                '/',
                status=500,
                body=_opensearch_error(500, 'fake_version_unavailable', 'GET / disabled'),
            )
        return _Route(
            'GET',
            '/',
            body={
                'name': 'fake-node-1',
                'cluster_name': 'fake-cluster',
                'cluster_uuid': 'fake-cluster-uuid',
                'version': {
                    'distribution': 'opensearch',
                    'number': self.version,
                    'build_type': 'tar',
                    'build_hash': 'fake-build-hash',
                    'build_date': '2025-01-01T00:00:00.000000Z',
                    'build_snapshot': False,
                    'lucene_version': '10.2.1',
                    'minimum_wire_compatibility_version': '2.19.0',
                    'minimum_index_compatibility_version': '2.0.0',
                },
                'tagline': 'The OpenSearch Project: https://opensearch.org/',
            },
        )


@pytest.fixture
def fake_opensearch(monkeypatch):
    """The fake OpenSearch every ``AsyncOpenSearch`` connection talks to during the test.

    It replaces the aiohttp session opensearch-py's connections create, so the server's own
    connection class (``BufferedAsyncHttpConnection``), client setup, auth and error mapping
    all run unchanged; only the HTTP exchange is canned.
    """
    fake = FakeOpenSearch()

    async def create_fake_session(connection) -> None:
        if connection.loop is None:
            connection.loop = asyncio.get_running_loop()
        connection.session = _FakeAiohttpSession(fake)

    monkeypatch.setattr(AsyncHttpConnection, '_create_aiohttp_session', create_fake_session)
    monkeypatch.setattr(AIOHttpConnection, '_create_aiohttp_session', create_fake_session)
    yield fake
    fake.verify()


# ---------------------------------------------------------------------------
# pinned_openapi
# ---------------------------------------------------------------------------


class PinnedOpenApi:
    """Serves ``fixtures/openapi/<file>`` in place of the GitHub API specification."""

    def __init__(self, directory: Path = OPENAPI_FIXTURE_DIR):
        self.directory = directory
        self.calls: list[str] = []
        self.error: Exception | None = None

    async def fetch(self, file_name: str) -> dict:
        self.calls.append(file_name)
        if self.error is not None:
            raise self.error
        with open(self.directory / file_name, encoding='utf-8') as f:
            return yaml.safe_load(f)


@pytest.fixture
def pinned_openapi(monkeypatch) -> PinnedOpenApi:
    """Patch ``tool_generator.fetch_github_spec`` to load the committed OpenAPI fixtures."""
    pinned = PinnedOpenApi()
    monkeypatch.setattr(tool_generator, 'fetch_github_spec', pinned.fetch)
    return pinned


@pytest.fixture
def openapi_unavailable(pinned_openapi) -> PinnedOpenApi:
    """Variant of ``pinned_openapi`` whose fetch raises, as when GitHub is unreachable."""
    pinned_openapi.error = aiohttp.ClientConnectionError(
        'fake: cannot connect to host raw.githubusercontent.com:443'
    )
    return pinned_openapi


# ---------------------------------------------------------------------------
# no_network
# ---------------------------------------------------------------------------


def _stays_on_this_machine(sock: socket.socket, address: Any) -> bool:
    """Whether connecting ``sock`` to ``address`` cannot leave the machine.

    True for Unix sockets and for loopback IP literals (127.0.0.0/8, ::1, ::ffff:127.0.0.0/104).
    A host name is never accepted here: it would be resolved by the C library, out of reach of
    the getaddrinfo guard.
    """
    if sock.family == getattr(socket, 'AF_UNIX', None):
        return True
    if sock.family not in (socket.AF_INET, socket.AF_INET6) or not isinstance(address, tuple):
        return False
    try:
        ip = ipaddress.ip_address(address[0])
    except (IndexError, TypeError, ValueError):
        return False
    return (getattr(ip, 'ipv4_mapped', None) or ip).is_loopback


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test if anything resolves a host name or connects off this machine.

    Connections to Unix sockets and loopback IP literals are let through: on Windows,
    asyncio's event loop builds its self-pipe with ``socket.socketpair()``, whose fallback
    connects to 127.0.0.1, and pytest-asyncio creates the loop after this guard is in place.
    Nothing reaches a real OpenSearch that way: the fake answers at a ``.invalid`` host,
    without sockets, and DNS lookups stay blocked, so no host name resolves to loopback.
    """
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def blocked(what: str) -> OSError:
        attempts.append(what)
        return OSError(errno.ENETUNREACH, f'network access is disabled in this test: {what}')

    def guarded_getaddrinfo(host, *args, **kwargs):
        raise blocked(f'getaddrinfo({host!r})')

    def guarded_connect(sock, address):
        if _stays_on_this_machine(sock, address):
            return real_connect(sock, address)
        raise blocked(f'connect({address!r})')

    def guarded_connect_ex(sock, address):
        if _stays_on_this_machine(sock, address):
            return real_connect_ex(sock, address)
        raise blocked(f'connect_ex({address!r})')

    monkeypatch.setattr(socket, 'getaddrinfo', guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, 'connect', guarded_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', guarded_connect_ex)
    yield attempts
    if attempts:
        pytest.fail(f'network access attempted: {attempts}', pytrace=False)


# ---------------------------------------------------------------------------
# isolated_state
# ---------------------------------------------------------------------------

# Module-level containers the server mutates. The three sub-registries share their entry
# objects with TOOL_REGISTRY. The server keeps no OpenSearch client cache (a client is built
# and closed per call in opensearch.client.get_opensearch_client), so there is none to track.
_TRACKED_CONTAINERS = (
    TOOL_REGISTRY,
    SKILLS_TOOLS_REGISTRY,
    AGENTIC_MEMORY_TOOLS_REGISTRY,
    MEMORY_TOOLS_REGISTRY,
    # Never cleared by the server; any entry turns dynamic mode and its instructions off.
    clusters_information.cluster_registry,
)

# Module-level scalars. global_state's setters only run for truthy values, so they leak.
_TRACKED_SCALARS = (
    (global_state, '_current_mode'),
    (global_state, '_current_profile'),
    (global_state, '_current_config_file_path'),
    (tool_filter, '_resolved_allow_write_setting'),
)

_CONTEXT_VARS = ((client_name_var, 'unknown'), (request_context_var, None))

# Environment the server (or boto3/aiohttp underneath it) reads.
_CLEARED_ENV_PREFIXES = ('OPENSEARCH_', 'AWS_', 'MEMORY_')
_CLEARED_ENV_NAMES = frozenset(
    {'BOTO_CONFIG', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'}
    | {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}
)


def _owned_by_pytest(handler: logging.Handler) -> bool:
    return type(handler).__module__.startswith('_pytest')


class StateSnapshot:
    """Module state at one moment, restorable in place.

    Every dict, list and set reachable from the tracked containers is recorded by identity
    with a shallow copy of its contents, so ``restore()`` puts back the original objects with
    their original contents: entries the server replaced (``apply_custom_tool_config``) come
    back, and ``input_schema`` ``properties``/``required`` that ``get_tools`` edited in place
    are refilled. Also restored: the args models' field descriptions (``config.py`` edits
    them on the class), the tracked scalars and the root logger's non-pytest handlers.
    """

    def __init__(self):
        self._containers: list[tuple[Any, Any]] = []
        models = {baseToolArgs}
        seen: set[int] = set()
        stack = list(_TRACKED_CONTAINERS)
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, dict):
                self._containers.append((obj, dict(obj)))
                model = obj.get('args_model')
                if isinstance(model, type) and issubclass(model, BaseModel):
                    models.add(model)
                stack.extend(obj.values())
            elif isinstance(obj, list):
                self._containers.append((obj, list(obj)))
                stack.extend(obj)
            elif isinstance(obj, set):
                self._containers.append((obj, set(obj)))
        self._descriptions = [
            (info, info.description) for model in models for info in model.model_fields.values()
        ]
        self._scalars = [
            (module, name, getattr(module, name)) for module, name in _TRACKED_SCALARS
        ]
        root = logging.getLogger()
        self._root_level = root.level
        self._root_handlers = [h for h in root.handlers if not _owned_by_pytest(h)]

    def restore(self, *, include_logging: bool = True) -> None:
        for container, contents in self._containers:
            if isinstance(container, list):
                container[:] = contents
            else:
                container.clear()
                container.update(contents)
        for info, description in self._descriptions:
            info.description = description
        for module, name, value in self._scalars:
            setattr(module, name, value)
        if include_logging:
            # configure_logging() replaces root handlers; leave pytest's own capture alone.
            root = logging.getLogger()
            for handler in list(root.handlers):
                if not _owned_by_pytest(handler) and handler not in self._root_handlers:
                    root.removeHandler(handler)
            for handler in self._root_handlers:
                if handler not in root.handlers:
                    root.addHandler(handler)
            root.setLevel(self._root_level)


# Taken when this conftest is imported, before any test has run.
_PRISTINE_STATE = StateSnapshot()

# Packages of the server under test. Some unit tests delete modules of these packages from
# sys.modules and import them again (tests/tools does so for tools.tools, tools.memory_tools,
# tools.skills_tools and opensearch.helper), so a module first imported after that binds to
# a second TOOL_REGISTRY. This conftest imports every server module up front (see the
# imports above), so they all share the objects the harness tracks, and pins exactly those
# module objects in sys.modules for the duration of each characterization test.
_SERVER_PACKAGES = ('tools', 'opensearch', 'mcp_server_opensearch')


def _is_server_module(name: str) -> bool:
    return name.partition('.')[0] in _SERVER_PACKAGES


_PINNED_MODULES = {name: module for name, module in sys.modules.items() if _is_server_module(name)}

# The server modules imported above, so that they are loaded (and pinned) before any test runs.
_SERVER_MODULES = (
    logging_config,
    server_instructions,
    stdio_server,
    streaming_server,
    tool_config,
    tool_executor,
    tool_filter,
    tool_generator,
)
assert all(module in _PINNED_MODULES.values() for module in _SERVER_MODULES)

# (module, attribute, tracked object): what the server code must resolve to, or the harness
# would snapshot and restore objects the server does not use.
_REQUIRED_BINDINGS = (
    ('tools.tools', 'TOOL_REGISTRY', TOOL_REGISTRY),
    ('tools.memory_tools', 'MEMORY_TOOLS_REGISTRY', MEMORY_TOOLS_REGISTRY),
    ('tools.skills_tools', 'SKILLS_TOOLS_REGISTRY', SKILLS_TOOLS_REGISTRY),
    (
        'tools.agentic_memory.actions',
        'AGENTIC_MEMORY_TOOLS_REGISTRY',
        AGENTIC_MEMORY_TOOLS_REGISTRY,
    ),
    ('tools.config', 'default_tool_registry', TOOL_REGISTRY),
    ('tools.tool_generator', 'TOOL_REGISTRY', TOOL_REGISTRY),
    ('tools.tool_filter', 'SKILLS_TOOLS_REGISTRY', SKILLS_TOOLS_REGISTRY),
    ('mcp_server_opensearch.streaming_server', 'TOOL_REGISTRY', TOOL_REGISTRY),
    ('mcp_server_opensearch.stdio_server', 'TOOL_REGISTRY', TOOL_REGISTRY),
    (
        'mcp_server_opensearch.clusters_information',
        'cluster_registry',
        clusters_information.cluster_registry,
    ),
)


def _pin_server_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the conftest-time server modules back in sys.modules (and on their packages)."""
    for name, module in _PINNED_MODULES.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent_name, _, child = name.rpartition('.')
        if parent_name in _PINNED_MODULES:
            monkeypatch.setattr(_PINNED_MODULES[parent_name], child, module, raising=False)
    # A server module first imported later may be bound to re-imported copies: drop it, so a
    # test that needs it imports it afresh against the pinned modules.
    for name in [n for n in sys.modules if _is_server_module(n) and n not in _PINNED_MODULES]:
        monkeypatch.delitem(sys.modules, name)
        parent_name, _, child = name.rpartition('.')
        parent = _PINNED_MODULES.get(parent_name)
        if parent is not None and child in vars(parent):
            monkeypatch.delattr(parent, child)


def _check_bindings() -> None:
    wrong = [
        f'{module_name}.{attribute}'
        for module_name, attribute, expected in _REQUIRED_BINDINGS
        if getattr(sys.modules.get(module_name), attribute, None) is not expected
    ]
    if wrong:
        pytest.fail(
            'the server is bound to different module state than the harness tracks: '
            f'{", ".join(wrong)}. Something replaced these objects; isolation cannot hold.',
            pytrace=False,
        )


class IsolatedState:
    """Handle the ``isolated_state`` fixture yields."""

    def snapshot(self) -> StateSnapshot:
        """Take a snapshot the same way the fixture does before each test."""
        return StateSnapshot()

    @staticmethod
    def isolation(monkeypatch: pytest.MonkeyPatch, opensearch_url: str):
        """The fixture's own setup and teardown as a context manager (see ``isolation``)."""
        return isolation(monkeypatch, opensearch_url)


@contextlib.contextmanager
def isolation(monkeypatch: pytest.MonkeyPatch, opensearch_url: str):
    """Body of ``isolated_state``; ``monkeypatch``'s undo restores modules and environment.

    On entry: pin the server modules, check the server's bindings, snapshot the current
    state, reset it to the state at import, reset the context variables, clear the
    environment the server reads and point ``OPENSEARCH_URL`` at ``opensearch_url``.
    On exit: restore the snapshot and the context variables.
    """
    _pin_server_modules(monkeypatch)
    _check_bindings()
    before = StateSnapshot()
    _PRISTINE_STATE.restore(include_logging=False)
    tokens = [(var, default, var.set(default)) for var, default in _CONTEXT_VARS]

    for name in list(os.environ):
        if name.startswith(_CLEARED_ENV_PREFIXES) or name in _CLEARED_ENV_NAMES:
            monkeypatch.delenv(name)
    monkeypatch.setenv('OPENSEARCH_URL', opensearch_url)
    # Keep boto3 off the developer's AWS config and credentials and off instance metadata.
    monkeypatch.setenv('AWS_CONFIG_FILE', os.devnull)
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', os.devnull)
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')
    monkeypatch.setenv('BOTO_CONFIG', os.devnull)
    try:
        yield IsolatedState()
    finally:
        before.restore()
        for var, default, token in reversed(tokens):
            try:
                var.reset(token)
            except (RuntimeError, ValueError):
                var.set(default)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, fake_opensearch, pinned_openapi):
    """Run each test from import-time module state and a clean environment; undo it after.

    See ``isolation`` for what happens before and after the test.
    """
    with isolation(monkeypatch, fake_opensearch.url) as state:
        yield state


# ---------------------------------------------------------------------------
# mcp_session
# ---------------------------------------------------------------------------


def dump(model: BaseModel) -> dict:
    """Dump an SDK result model the way the characterization tests compare them."""
    return model.model_dump(mode='json', by_alias=True, exclude_none=True)


def _wire(message: BaseModel) -> dict:
    # The exact JSON the SDK's stdio and HTTP transports write for this message.
    return json.loads(message.model_dump_json(by_alias=True, exclude_unset=True))


async def raw_jsonrpc(
    server: Any,
    method: str,
    params: dict | None = None,
    protocol_version: str | None = None,
) -> dict:
    """Send one JSON-RPC request on a fresh in-memory connection; return the raw response.

    ``server`` is whatever ``create_mcp_server`` returned (a low-level ``Server`` or an
    ``MCPServer``); the SDK's in-memory transport accepts both.

    Handshake protocol versions (the default, or any version not in the SDK's
    ``MODERN_PROTOCOL_VERSIONS``) first run ``initialize`` with that version and send
    ``notifications/initialized``; if ``initialize`` itself fails, its error is returned.
    For ``method='initialize'`` the request itself is the handshake, with ``params`` (or
    default params offering ``protocol_version``). Modern versions send the request alone,
    with the per-request ``_meta`` envelope merged into ``params``.
    """
    version = protocol_version or DEFAULT_PROTOCOL_VERSION
    with anyio.fail_after(RAW_TIMEOUT_SECONDS):
        async with InMemoryTransport(server) as (read_stream, write_stream):

            async def send(message: BaseModel) -> None:
                await write_stream.send(SessionMessage(message))

            async def response_to(request_id: int) -> dict:
                while True:
                    try:
                        item = await read_stream.receive()
                    except anyio.EndOfStream:
                        raise AssertionError(f'connection closed before response {request_id}')
                    if isinstance(item, Exception):
                        raise item
                    message = item.message
                    if isinstance(message, (JSONRPCResponse, JSONRPCError)) and (
                        message.id == request_id
                    ):
                        return _wire(message)

            if version in MODERN_PROTOCOL_VERSIONS:
                params = dict(params or {})
                params['_meta'] = {
                    PROTOCOL_VERSION_META_KEY: version,
                    CLIENT_INFO_META_KEY: HARNESS_CLIENT_INFO,
                    CLIENT_CAPABILITIES_META_KEY: {},
                    **(params.get('_meta') or {}),
                }
            elif method == 'initialize':
                if params is None:
                    params = _initialize_params(version)
            else:
                init_params = _initialize_params(version)
                await send(
                    JSONRPCRequest(jsonrpc='2.0', id=0, method='initialize', params=init_params)
                )
                initialized = await response_to(0)
                if 'error' in initialized:
                    return initialized
                await send(JSONRPCNotification(jsonrpc='2.0', method='notifications/initialized'))
            await send(JSONRPCRequest(jsonrpc='2.0', id=1, method=method, params=params))
            return await response_to(1)


def _initialize_params(version: str) -> dict:
    return {'protocolVersion': version, 'capabilities': {}, 'clientInfo': HARNESS_CLIENT_INFO}


class McpSession:
    """A server built by ``create_mcp_server`` and an in-process SDK ``Client`` connected to it."""

    def __init__(self, server: Any, client: Client):
        self.server = server
        self.client = client

    async def list_tools(self) -> dict:
        return dump(await self.client.list_tools())

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return dump(await self.client.call_tool(name, arguments or {}))

    async def raw(
        self, method: str, params: dict | None = None, protocol_version: str | None = None
    ) -> dict:
        """Raw JSON-RPC response (``result`` or ``error``); see ``raw_jsonrpc``."""
        return await raw_jsonrpc(self.server, method, params, protocol_version)


@pytest.fixture
def mcp_session(monkeypatch, fake_opensearch, pinned_openapi) -> Callable:
    """Factory that builds the server and connects an in-process client to it.

    Use as ``async with mcp_session(...) as session``.

    Arguments: ``mode`` ('single' or 'multi'), ``config_file`` (path or ''),
    ``cli_overrides`` (dict as parsed from unknown CLI args), ``env`` (variables to set before
    the server is built; ``None`` as a value unsets one), ``profile`` and ``client_mode``
    (the SDK ``Client`` mode; 'legacy' runs the initialize handshake over the in-memory
    transport, as stdio clients do).
    """

    @contextlib.asynccontextmanager
    async def open_session(
        mode: str = 'single',
        config_file: str | os.PathLike = '',
        cli_overrides: dict | None = None,
        env: dict | None = None,
        *,
        profile: str = '',
        client_mode: str = 'legacy',
    ):
        for name, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        server = await streaming_server.create_mcp_server(
            mode=mode,
            profile=profile,
            config_file_path=str(config_file) if config_file else '',
            cli_tool_overrides=cli_overrides,
        )
        async with Client(server, mode=client_mode, cache=None) as client:
            yield McpSession(server, client)

    return open_session
