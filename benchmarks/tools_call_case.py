# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark case: ``tools/call`` timing in one session, run inside a tree under test.

``scripts/bench_tools_call.py`` copies this module into a tree's ``tests/characterization/``
and runs it there with pytest, so it uses that tree's own harness (``conftest.py``): the
server is built by ``streaming_server.create_mcp_server``, OpenSearch is ``fake_opensearch``
and nothing touches the network. The driver passes the run as JSON in the environment
variable ``BENCH_TOOLS_CALL_CONFIG``:

- ``out_dir``: where each transport's results go (``<transport>.json``);
- ``workloads``: a list of ``{'transport', 'mode', 'n', 'latency_ms'}``;
- ``warmup``: untimed calls in each session before the timed ones;
- ``batches``: timed batches of ``n`` calls per workload, in the same session;
- ``profile``: ``None``, or a directory: then each workload's timed batches run under
  ``cProfile`` and its stats are dumped there (``<workload id>.pstats``).

Transports: ``memory`` is the SDK's in-memory ``Client`` (as the harness's ``mcp_session``
connects it); ``http`` is the SDK's streamable-HTTP client (``ClientSession``, as the
integration test's ``mcp_client``) over ``httpx2.ASGITransport`` into the app the harness's
``http_app`` runs (``MCPStarletteApp(server, stateless=True)``, lifespan running). Each
transport builds its server once per run, as a long-running server is; each workload then
opens its own client session on it, warms it up and times its calls there.

Modes: ``sequential`` awaits one call after another; ``concurrent`` starts ``n`` tasks that
wait on one event and releases them together, as
``integration_tests/concurrency/test_concurrent_single_client.py`` does. The fake's latency is
an ``asyncio.sleep`` before each fake OpenSearch reply, so concurrent calls can overlap.
"""

import asyncio
import cProfile
import httpx2
import json
import logging
import os
import pytest
import sys
import time
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pathlib import Path


CONFIG_ENV = 'BENCH_TOOLS_CALL_CONFIG'
CONFIG = json.loads(os.environ.get(CONFIG_ENV, '{}'))
TOOL = 'ListIndexTool'
# Reaches the fake: without auth configured the server takes the AWS path and fails it.
NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
CAT_INDICES = [
    {
        'health': 'green',
        'status': 'open',
        'index': 'logs-2020.01.01',
        'uuid': 'fake-uuid-logs',
        'pri': '1',
        'rep': '1',
        'docs.count': '42',
        'docs.deleted': '0',
        'store.size': '10kb',
        'pri.store.size': '5kb',
    },
    {
        'health': 'yellow',
        'status': 'open',
        'index': 'metrics-2020.01.01',
        'uuid': 'fake-uuid-metrics',
        'pri': '1',
        'rep': '1',
        'docs.count': '7',
        'docs.deleted': '0',
        'store.size': '3kb',
        'pri.store.size': '3kb',
    },
]
SERVER_PACKAGES = ('mcp_server_opensearch', 'tools', 'opensearch')


def workload_id(workload: dict) -> str:
    """``memory-concurrent-n5-50ms``: transport, mode, number of calls, fake latency."""
    return (
        f'{workload["transport"]}-{workload["mode"]}-n{workload["n"]}-{workload["latency_ms"]}ms'
    )


def module_location(module) -> str | None:
    """``__file__``; for a namespace package (no ``__file__``), its one ``__path__`` entry."""
    location = getattr(module, '__file__', None)
    if location is None:
        paths = list(getattr(module, '__path__', []))
        location = paths[0] if len(paths) == 1 else None
    return location


def server_module_files() -> dict:
    """Where every loaded module of the server's packages came from."""
    return {
        name: module_location(module)
        for name, module in sorted(sys.modules.items())
        if module is not None and name.split('.')[0] in SERVER_PACKAGES
    }


def logging_state() -> dict:
    """The root logger's level and handlers, as the server left them."""
    root = logging.getLogger()
    return {
        'level': logging.getLevelName(root.level),
        'handlers': [f'{type(h).__module__}.{type(h).__qualname__}' for h in root.handlers],
    }


class FakeLatency:
    """An ``asyncio.sleep`` of ``seconds`` before every fake OpenSearch reply."""

    def __init__(self, monkeypatch, fake_opensearch):
        """Patch the harness's fake exchange for the test (``monkeypatch`` undoes it)."""
        self.seconds = 0.0
        self.replies = 0
        harness = sys.modules[type(fake_opensearch).__module__]
        exchange = harness._FakeExchange
        original = exchange.__aenter__
        latency = self

        async def delayed_aenter(self):
            latency.replies += 1
            if latency.seconds:
                await asyncio.sleep(latency.seconds)
            return await original(self)

        monkeypatch.setattr(exchange, '__aenter__', delayed_aenter)


def check_result(result) -> None:
    """Fail the run unless the call succeeded and listed the canned indices."""
    text = ' '.join(getattr(item, 'text', '') for item in result.content)
    if result.is_error or 'logs-2020.01.01' not in text:
        raise AssertionError(f'{TOOL} call failed: is_error={result.is_error} {text[:500]}')


async def timed_batch(call, mode: str, n: int) -> dict:
    """Run ``n`` calls; return the batch's wall-clock time and each call's latency."""
    latencies: list[float] = []
    if mode == 'sequential':
        start = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            check_result(await call())
            latencies.append(time.perf_counter() - t0)
        wall = time.perf_counter() - start
    elif mode == 'concurrent':
        release = asyncio.Event()

        async def worker():
            await release.wait()
            t0 = time.perf_counter()
            result = await call()
            return time.perf_counter() - t0, result

        tasks = [asyncio.create_task(worker()) for _ in range(n)]
        # Let every task reach the event, as the integration test does.
        await asyncio.sleep(0.01)
        start = time.perf_counter()
        release.set()
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
        for latency, result in results:
            check_result(result)
            latencies.append(latency)
    else:
        raise ValueError(f'unknown mode {mode!r}')
    return {'wall_s': wall, 'per_call_s': wall / n, 'latencies_s': latencies}


def summarize_latencies(latencies: list[float]) -> dict:
    """The fastest, median and slowest call of a batch."""
    ordered = sorted(latencies)
    return {
        'min_s': ordered[0],
        'median_s': ordered[len(ordered) // 2],
        'max_s': ordered[-1],
    }


async def run_workload(call, workload: dict, latency: FakeLatency) -> dict:
    """Warm the session up, then time ``batches`` batches of ``n`` calls in it."""
    latency.seconds = 0.0
    for _ in range(CONFIG.get('warmup', 2)):
        check_result(await call())
    latency.seconds = workload['latency_ms'] / 1000
    replies_before = latency.replies
    profile_dir = CONFIG.get('profile')
    profiler = cProfile.Profile() if profile_dir else None
    batches = []
    if profiler is not None:
        profiler.enable()
    try:
        for _ in range(CONFIG.get('batches', 1)):
            batches.append(await timed_batch(call, workload['mode'], workload['n']))
    finally:
        if profiler is not None:
            profiler.disable()
        latency.seconds = 0.0
    if profiler is not None:
        profiler.dump_stats(str(Path(profile_dir) / f'{workload_id(workload)}.pstats'))
    calls = workload['n'] * len(batches)
    return {
        'workload': workload_id(workload),
        **workload,
        'batches': [
            {
                'wall_s': b['wall_s'],
                'per_call_s': b['per_call_s'],
                **summarize_latencies(b['latencies_s']),
            }
            for b in batches
        ],
        'fake_replies_per_call': (latency.replies - replies_before) / calls,
    }


async def run_memory(workloads: list[dict], mcp_session, latency: FakeLatency, harness) -> list:
    """One server (``mcp_session``); each workload in its own in-memory ``Client`` session."""
    results = []
    async with mcp_session(env=NO_AUTH) as built:
        for workload in workloads:
            async with harness.Client(built.server, mode='legacy', cache=None) as client:

                async def call():
                    return await client.call_tool(TOOL, {})

                results.append(await run_workload(call, workload, latency))
    return results


async def run_http(workloads: list[dict], http_app, latency: FakeLatency, harness) -> list:
    """One app (``http_app``); each workload in its own streamable-HTTP ``ClientSession``."""
    results = []
    base_url = harness.HTTP_BASE_URL
    async with http_app(env=NO_AUTH) as http:
        transport = httpx2.ASGITransport(app=http.app)
        async with httpx2.AsyncClient(transport=transport, base_url=base_url) as client:
            for workload in workloads:
                async with (
                    streamable_http_client(f'{base_url}/mcp', http_client=client) as streams,
                    ClientSession(streams[0], streams[1]) as session,
                ):
                    await session.initialize()

                    async def call():
                        return await session.call_tool(TOOL, arguments={})

                    results.append(await run_workload(call, workload, latency))
    return results


@pytest.mark.parametrize('transport', ['memory', 'http'])
async def test_tools_call_timing(transport, request, monkeypatch, fake_opensearch):
    """Time the configured workloads of ``transport``; write ``<out_dir>/<transport>.json``."""
    workloads = [w for w in CONFIG.get('workloads', []) if w['transport'] == transport]
    if not workloads:
        pytest.skip(f'no {transport} workload in {CONFIG_ENV}')
    fake_opensearch.reply('GET', '/_cat/indices', CAT_INDICES)
    latency = FakeLatency(monkeypatch, fake_opensearch)
    harness = sys.modules[type(fake_opensearch).__module__]
    if transport == 'memory':
        mcp_session = request.getfixturevalue('mcp_session')
        results = await run_memory(workloads, mcp_session, latency, harness)
    else:
        http_app = request.getfixturevalue('http_app')
        results = await run_http(workloads, http_app, latency, harness)
    report = {
        'transport': transport,
        'modules': server_module_files(),
        'harness': harness.__file__,
        'case': __file__,
        'logging': logging_state(),
        'results': results,
    }
    out = Path(CONFIG['out_dir']) / f'{transport}.json'
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
