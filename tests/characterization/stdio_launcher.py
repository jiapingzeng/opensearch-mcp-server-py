# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the server's real entry point with the characterization fakes in place.

Usage::

    python stdio_launcher.py [server arguments...]

The ``stdio_launcher`` fixture (``conftest.py``) starts this script as a subprocess. It
installs, in this process, the patches the characterization fixtures install in the test
process, then calls ``mcp_server_opensearch.main()`` with the given arguments, named as the
``opensearch-mcp-server-py`` console script:

- fake OpenSearch: the aiohttp session of every OpenSearch connection is the harness's
  fake, answering from the reply table in the JSON file named by
  ``CHAR_FAKE_OPENSEARCH_TABLE`` (written with ``conftest.fake_table``);
- pinned OpenAPI: ``tool_generator.fetch_github_spec`` serves ``fixtures/openapi/``, or
  raises when the table carries an ``openapi_error``;
- network guard: DNS lookups and connections off this machine fail, as under ``no_network``.

Every request the fake receives and every network attempt the guard blocks is appended as
one JSON line to the file named by ``CHAR_FAKE_OPENSEARCH_LOG`` as it happens, so the test
process can read them even if this process is killed.
"""

import asyncio
import errno
import json
import os
import socket
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


class _Log:
    """The launcher log: one JSON object per line, appended and closed at once."""

    def __init__(self, path: str):
        self.path = path

    def write(self, entry: dict) -> None:
        # ASCII-escaped, so a body decoded with surrogates still writes as UTF-8.
        with open(self.path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(entry) + '\n')


def _install_fakes(harness, table: dict, log: _Log) -> None:
    from opensearchpy._async.http_aiohttp import AIOHttpConnection
    from opensearchpy.connection.http_async import AsyncHttpConnection
    from tools import tool_generator

    fake, pinned = harness.fake_from_table(table)
    exchange = fake._exchange

    def logged_exchange(*args):
        seen, unmatched = len(fake.requests), len(fake.unmatched)
        try:
            return exchange(*args)
        finally:
            if len(fake.requests) > seen:
                matched = len(fake.unmatched) == unmatched
                log.write({'kind': 'request', 'record': fake.requests[-1], 'matched': matched})

    fake._exchange = logged_exchange

    # As the fake_opensearch fixture does.
    async def create_fake_session(connection) -> None:
        if connection.loop is None:
            connection.loop = asyncio.get_running_loop()
        connection.session = harness._FakeAiohttpSession(fake)

    AsyncHttpConnection._create_aiohttp_session = create_fake_session
    AIOHttpConnection._create_aiohttp_session = create_fake_session
    # As the pinned_openapi fixture does.
    tool_generator.fetch_github_spec = pinned.fetch


def _install_network_guard(harness, log: _Log) -> None:
    """As the no_network fixture does; attempts are logged for the test process to fail on."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def blocked(what: str) -> OSError:
        log.write({'kind': 'network', 'attempt': what})
        return OSError(errno.ENETUNREACH, f'network access is disabled in this test: {what}')

    def guarded_getaddrinfo(host, *args, **kwargs):
        raise blocked(f'getaddrinfo({host!r})')

    def guarded_connect(sock, address):
        if harness._stays_on_this_machine(sock, address):
            return real_connect(sock, address)
        raise blocked(f'connect({address!r})')

    def guarded_connect_ex(sock, address):
        if harness._stays_on_this_machine(sock, address):
            return real_connect_ex(sock, address)
        raise blocked(f'connect_ex({address!r})')

    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def main() -> None:
    # The harness (conftest.py) sits next to this script; it imports the server's modules.
    sys.path.insert(0, str(HERE))
    import conftest as harness

    with open(os.environ[harness.FAKE_TABLE_ENV], encoding='utf-8') as f:
        table = json.load(f)
    log = _Log(os.environ[harness.FAKE_LOG_ENV])
    _install_fakes(harness, table, log)
    _install_network_guard(harness, log)

    from mcp_server_opensearch import main as server_main

    sys.argv = [harness.SERVER_PROG, *sys.argv[1:]]
    server_main()


if __name__ == '__main__':
    main()
