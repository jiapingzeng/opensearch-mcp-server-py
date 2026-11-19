# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: the connection's fallback path.

``BufferedAsyncHttpConnection.perform_request`` (``src/opensearch/connection.py``) streams the
response itself; when that attempt fails it logs a warning and retries through
``_fallback_perform_request``, which calls opensearch-py's ``AsyncHttpConnection`` and then
measures the body it got back by type before checking ``max_response_size``:

- a ``str`` is measured as its UTF-8 encoding (``len(text.encode('utf-8'))``);
- ``bytes`` by their length;
- any other type as the UTF-8 encoding of its ``str()`` (so a ``bytearray`` is measured by
  its repr, ``bytearray(b'...')``, not by its length);

and a body over the limit raises ``ResponseSizeExceededError``, which the fallback re-raises
as it is (no ``opensearch_request`` error event, unlike any other exception there).

Each case drives one ``tools/call`` of ``ListIndexTool`` (``GET /_cat/indices``) through
``mcp_session``, with ``fake_opensearch.fail_streaming`` failing the streaming attempt and
answering the retry, and goldens, under ``golden/gaps_fallback/``: the body type, the limit
(``OPENSEARCH_MAX_RESPONSE_SIZE``), the call and its result, the requests the fake saw (and
how many) and the ``opensearch.connection`` log lines (the fallback warning, the size error,
the measured ``response_size``).

The three branches of the size check are told apart by the same index list given as a
``str``, as UTF-8 or UTF-16 ``bytes`` and as a ``bytearray`` or a ``tuple``: each measures a
different size (logged as ``response_size``, or quoted in the size error), and one limit
(the UTF-8 length of the text) lets the ``str`` and UTF-8 ``bytes`` through and stops the
UTF-16 ``bytes`` and the ``bytearray``. The index names hold non-ASCII characters, so the
UTF-8 length differs from the character count. Every body is larger than the ``GET /``
version reply (438 bytes), so the limits never stop the version check.

Other body types are pinned in-process only: in a launched server the body crosses the JSON
fake table, where a tuple becomes a list and a ``bytearray`` becomes ``bytes``.
"""

import codecs
import json
import logging
import pytest
from dataclasses import dataclass
from typing import Any


AREA = 'gaps_fallback'

# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
CONNECTION_LOGGER = 'opensearch.connection'
# Fields of connection.py's structured log events the goldens keep.
EVENT_FIELDS = (
    'event_type',
    'http_method',
    'endpoint',
    'status',
    'status_code',
    'response_size',
    'error',
    'duration_ms',
)
JSON_TYPE = {'content-type': 'application/json; charset=UTF-8'}
TOOL = 'ListIndexTool'
ARGUMENTS = {'include_detail': False}
PATH = '/_cat/indices'
FALLBACK_WARNING = 'falling back to parent implementation'
SIZE_ERROR = 'Response size exceeded limit'

# The same index list in each form the fallback may get it in.
INDICES = [{'index': f'journal-été-{i:02d}'} for i in range(24)]
TEXT = json.dumps(INDICES, separators=(',', ':'), ensure_ascii=False)
UTF8 = TEXT.encode('utf-8')
# Explicit byte order: 'utf-16' would use the platform's. json.loads reads it by its BOM.
UTF16 = codecs.BOM_UTF16_LE + TEXT.encode('utf-16-le')
# The sizes each branch of the size check measures (asserted below, so the goldens'
# numbers are known to come from that branch).
TEXT_CHARS = 649
TEXT_UTF8_SIZE = 697  # str branch: len(TEXT.encode('utf-8')); bytes branch for UTF8
UTF16_SIZE = 1300  # bytes branch: len(UTF16)
BYTEARRAY_SIZE = 999  # other branch: len(str(bytearray(UTF8)).encode('utf-8'))
TUPLE_SIZE = 744  # other branch: len(str(tuple(INDICES)).encode('utf-8'))
VERSION_REPLY_SIZE = 438  # GET /, streamed (no fallback): every limit here is above it


@dataclass(frozen=True)
class Case:
    """One ``tools/call`` whose ``_cat/indices`` request falls back and gets ``body``.

    ``size`` is what the size check measures for ``body``; ``limit`` the
    ``OPENSEARCH_MAX_RESPONSE_SIZE`` the server runs with (``None``: not set, no limit).
    """

    name: str
    body: Any
    size: int
    limit: int | None = None
    headers: dict | None = None
    is_error: bool = False

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.size > self.limit


# --- the body measured by type (no limit): 304->306, 306-310 -----------------------------

MEASURED = [
    # str branch (304-305): the UTF-8 length (697), not the character count (649).
    Case('measured_str', TEXT, TEXT_UTF8_SIZE, headers=JSON_TYPE),
    # bytes branch (304->306, 306-307): the raw length of UTF-16 bytes (1300). The client
    # parses them (json.loads reads the BOM), so the result is the str case's.
    Case('measured_bytes_utf16', UTF16, UTF16_SIZE, headers=JSON_TYPE),
    # other branch (308-310): a bytearray is measured by its repr (999), not its length
    # (697); json.loads still parses it, so the call succeeds.
    # characterization: current behaviour
    Case('measured_other_bytearray', bytearray(UTF8), BYTEARRAY_SIZE, headers=JSON_TYPE),
    # other branch (308-310): a tuple is measured by its repr (744); the client cannot
    # parse it (json.loads: TypeError), so the call fails after the success event.
    Case('measured_other_tuple', tuple(INDICES), TUPLE_SIZE, is_error=True),
]

# --- one limit, the UTF-8 length of the text: 312-316 and the re-raise, 334-335 ----------

LIMITED = [
    # str branch, at the limit: 697 > 697 is false, the call succeeds.
    Case('limit_str_at_limit', TEXT, TEXT_UTF8_SIZE, TEXT_UTF8_SIZE, JSON_TYPE),
    # str branch, one byte under: exceeded, the error quotes the UTF-8 length (697); a
    # character count (649) would have passed.
    Case('limit_str_over', TEXT, TEXT_UTF8_SIZE, TEXT_UTF8_SIZE - 1, JSON_TYPE, True),
    # bytes branch, UTF-8 bytes of the same text at the same limit: 697, the call succeeds.
    Case('limit_bytes_utf8_at_limit', UTF8, TEXT_UTF8_SIZE, TEXT_UTF8_SIZE, JSON_TYPE),
    # bytes branch, UTF-16 bytes of the same text: 1300 > 697, exceeded.
    Case('limit_bytes_utf16_over', UTF16, UTF16_SIZE, TEXT_UTF8_SIZE, JSON_TYPE, True),
    # other branch, a bytearray of the UTF-8 bytes that passed above: its repr (999) is
    # measured, so the same 697 bytes are refused.
    # characterization: current behaviour
    Case(
        'limit_other_bytearray_over',
        bytearray(UTF8),
        BYTEARRAY_SIZE,
        TEXT_UTF8_SIZE,
        JSON_TYPE,
        True,
    ),
]


def log_records(caplog) -> list[dict]:
    """The ``opensearch.connection`` records: logger, level, message and event fields."""
    return [
        {
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            **{name: getattr(record, name) for name in EVENT_FIELDS if hasattr(record, name)},
        }
        for record in caplog.records
        if record.name == CONNECTION_LOGGER
    ]


def test_payload_sizes():
    # The constants the cases rely on, so a golden's numbers name the branch that made them.
    assert len(TEXT) == TEXT_CHARS
    assert len(UTF8) == TEXT_UTF8_SIZE
    assert len(UTF16) == UTF16_SIZE
    assert len(str(bytearray(UTF8)).encode('utf-8')) == BYTEARRAY_SIZE
    assert len(str(tuple(INDICES)).encode('utf-8')) == TUPLE_SIZE
    sizes = {TEXT_CHARS, TEXT_UTF8_SIZE, UTF16_SIZE, BYTEARRAY_SIZE, TUPLE_SIZE}
    assert len(sizes) == 5
    assert min(sizes) > VERSION_REPLY_SIZE


@pytest.mark.parametrize('case', MEASURED + LIMITED, ids=lambda case: case.name)
async def test_fallback(mcp_session, fake_opensearch, golden, normalize, caplog, case):
    caplog.set_level(logging.INFO, logger=CONNECTION_LOGGER)
    fake_opensearch.fail_streaming('GET', PATH, case.body, headers=case.headers)
    env = dict(NO_AUTH)
    if case.limit is not None:
        env['OPENSEARCH_MAX_RESPONSE_SIZE'] = str(case.limit)
    async with mcp_session(env=env) as session:
        result = await session.call_tool(TOOL, ARGUMENTS)

    assert result.get('isError') is case.is_error, result
    requests = fake_opensearch.requests
    attempts = [r.get('attempt') for r in requests if r['path'] == PATH]
    # One streaming attempt, one fallback; a size error is not retried.
    assert attempts == ['streaming', 'fallback']
    records = log_records(caplog)
    warnings = [r for r in records if r['level'] != 'INFO']
    assert [r['level'] for r in warnings if FALLBACK_WARNING in r['message']] == ['WARNING']
    events = [r for r in records if r.get('endpoint') == PATH]
    size_errors = [r for r in records if r['message'].startswith(SIZE_ERROR)]
    if case.exceeded:
        # 312-316: the error log and the error quote the size this branch measured.
        assert [r['message'] for r in size_errors] == [
            f'{SIZE_ERROR}: {case.size} > {case.limit} bytes'
        ]
        assert f'Received {case.size} bytes.' in result['content'][0]['text']
        # 334-335: re-raised as it is, so no opensearch_request error event (the other
        # exceptions in the fallback log one before re-raising).
        assert events == []
    else:
        assert size_errors == []
        # The success event after the size check logs the size this branch measured.
        assert [(r['status'], r['response_size']) for r in events] == [('success', case.size)]
    golden(
        f'{AREA}/{case.name}',
        normalize(
            {
                'body_type': type(case.body).__name__,
                'max_response_size': case.limit,
                'call': {'name': TOOL, 'arguments': ARGUMENTS},
                'result': result,
                'request_count': len(requests),
                'requests': requests,
                'connection_log': records,
                'warnings': [r['message'] for r in warnings],
            }
        ),
    )
