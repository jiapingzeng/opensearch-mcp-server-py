# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ``tools/call`` timing in one session across source trees (see benchmarks/README.md).

Each revision's ``src``, ``tests/characterization`` and ``pyproject.toml`` are extracted with
``git archive`` into a temporary directory (nothing is checked out), the case module
``benchmarks/tools_call_case.py`` is copied into the tree's ``tests/characterization/``, and
the tree is made read-only. A run is one pytest process in that tree, with the tree's
``src`` first on ``PYTHONPATH``: every server module it imports must come from that tree, or
the driver fails. Rounds interleave the trees, rotating their order each round.

    uv run --offline python scripts/bench_tools_call.py            # full run, >= 30 rounds
    uv run --offline python scripts/bench_tools_call.py --quick    # 2 rounds, checks only
    uv run --offline python scripts/bench_tools_call.py --profile memory-concurrent-n50-0ms
    uv run --offline python scripts/bench_tools_call.py --context-cost

The results file is rewritten after every round, so a failure in a late round keeps the
rounds already finished (``meta.rounds_done``, ``meta.complete``).

Standard library only: ``time.perf_counter`` in the case, ``statistics`` and ``math.comb``
for the summary and the sign test, ``cProfile``/``pstats`` for ``--profile``, ``timeit`` for
``--context-cost`` (which times the SDK already installed, no new dependency).
"""

import argparse
import datetime
import importlib.metadata
import io
import json
import math
import os
import platform
import pstats
import shutil
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import timeit
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CASE_MODULE = REPO / 'benchmarks' / 'tools_call_case.py'
CASE_NAME = 'tools_call_case.py'
RESULTS_DIR = REPO / 'benchmarks' / 'results'
DEFAULT_OUT = RESULTS_DIR / 'tools_call.json'
CONFIG_ENV = 'BENCH_TOOLS_CALL_CONFIG'
ARCHIVE_PATHS = ('src', 'tests/characterization', 'pyproject.toml')
SERVER_PACKAGES = ('mcp_server_opensearch', 'tools', 'opensearch')
# 508e806: the parity baseline; c71570c: the last point within +-3% (the low-level Server);
# 18b7002: steps 16 and 16b (OpenSearchMCPServer on MCPServer).
BASELINE, REFERENCE, CANDIDATE = '508e806', 'c71570c', '18b7002'
DEFAULT_REVS = (BASELINE, REFERENCE, CANDIDATE)
TRANSPORTS = ('memory', 'http')
MODES = ('sequential', 'concurrent')
CALL_COUNTS = (5, 50)
LATENCIES_MS = (0, 50)
FULL_ROUNDS = 30
QUICK_ROUNDS = 2
# A workload is "slower" when its ratio of medians is above this bar, the sign test has
# p < SIGNIFICANCE and more rounds were slower than faster.
SLOWER_RATIO = 1.05
SIGNIFICANCE = 0.05
# The trees' pytest-timeout (``timeout = 120``) is turned off; the driver stops one pytest
# process after this base plus this much per call it makes (warm-up and timed, every
# workload), so larger ``--batches`` (and ``--profile``) get proportionally longer.
RUN_TIMEOUT_BASE_S = 120
RUN_TIMEOUT_PER_CALL_S = 2
CONTEXT_COST_REPEATS = 7
# On the baseline at 50 ms, a transport overlaps calls when the concurrent batch takes less
# than this share of the sequential one (n calls of 50 ms overlapped: about 1/n).
OVERLAP_SHARE = 0.5


def workload_id(workload: dict) -> str:
    """``memory-concurrent-n5-50ms``: transport, mode, number of calls, fake latency."""
    return (
        f'{workload["transport"]}-{workload["mode"]}-n{workload["n"]}-{workload["latency_ms"]}ms'
    )


def all_workloads() -> list[dict]:
    """Every transport x latency x n x mode."""
    return [
        {'transport': t, 'mode': m, 'n': n, 'latency_ms': ms}
        for t in TRANSPORTS
        for ms in LATENCIES_MS
        for n in CALL_COUNTS
        for m in MODES
    ]


def git(*args: str) -> bytes:
    """Run a read-only git command in this repository; return its stdout."""
    return subprocess.run(
        ['git', '-C', str(REPO), *args], check=True, stdout=subprocess.PIPE
    ).stdout


class Tree:
    """One revision's files, extracted read-only under ``root``."""

    def __init__(self, rev: str, root: Path):
        """Resolve ``rev`` to its commit; the files go under ``root`` on ``extract``."""
        self.rev = rev
        try:
            self.sha = git('rev-parse', '--verify', f'{rev}^{{commit}}').decode().strip()
        except subprocess.CalledProcessError:
            raise SystemExit(f'unknown revision {rev!r}') from None
        self.root = root
        self.src = root / 'src'

    def extract(self) -> None:
        """``git archive`` the revision into ``root``, add the case, make it read-only."""
        archive = git('archive', '--format=tar', self.sha, *ARCHIVE_PATHS)
        self.root.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(self.root, filter='data')
            else:  # pragma: no cover - Pythons without extraction filters
                tar.extractall(self.root)
        shutil.copyfile(CASE_MODULE, self.root / 'tests' / 'characterization' / CASE_NAME)
        set_writable(self.root, False)

    def owns(self, path: str | None) -> bool:
        """Whether ``path`` is inside this tree's ``src``."""
        if not path:
            return False
        return Path(path).resolve().is_relative_to(self.src.resolve())


def set_writable(root: Path, writable: bool) -> None:
    """Make every file and directory under ``root`` (and itself) writable or read-only."""
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in [root, *root.rglob('*')]:
        mode = path.stat().st_mode
        path.chmod((mode | stat.S_IWUSR) if writable else (mode & ~write_bits))


def run_timeout_s(workloads: list[dict], options: dict) -> int:
    """How long one pytest process may run: the base plus a budget per call it makes."""
    calls = sum(options['warmup'] + w['n'] * options['batches'] for w in workloads)
    return RUN_TIMEOUT_BASE_S + RUN_TIMEOUT_PER_CALL_S * calls


def run_tree(tree: Tree, workloads: list[dict], scratch: Path, options: dict) -> dict:
    """One pytest process in ``tree``; return its results, checked to come from the tree."""
    out_dir = Path(tempfile.mkdtemp(prefix=f'{tree.rev}-', dir=scratch))
    config = {'out_dir': str(out_dir), 'workloads': workloads, **options}
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(tree.src), *filter(None, [os.environ.get('PYTHONPATH')])]
    )
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env[CONFIG_ENV] = json.dumps(config)
    command = [
        sys.executable,
        '-m',
        'pytest',
        '-c',
        str(tree.root / 'pyproject.toml'),
        '--rootdir',
        str(tree.root),
        '-p',
        'no:cacheprovider',
        # The tree's ``timeout = 120`` would kill long --batches or --profile runs; the
        # driver's own timeout below scales with the calls instead.
        '-p',
        'no:timeout',
        '-q',
        '-x',
        '--no-header',
        f'tests/characterization/{CASE_NAME}',
    ]
    timeout = run_timeout_s(workloads, options)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=tree.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        output = expired.output or ''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        raise SystemExit(
            f'{tree.rev}: pytest still running after {timeout} s, stopped\n'
            f'{" ".join(command)}\n{output}'
        ) from None
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(
            f'{tree.rev}: pytest exited {proc.returncode}\n{" ".join(command)}\n{proc.stdout}'
        )
    run = {'elapsed_s': elapsed, 'workloads': {}, 'modules': {}, 'logging': {}}
    transports = sorted({w['transport'] for w in workloads})
    for transport in transports:
        path = out_dir / f'{transport}.json'
        if not path.is_file():
            raise SystemExit(f'{tree.rev}: no results for {transport}\n{proc.stdout}')
        report = json.loads(path.read_text(encoding='utf-8'))
        check_imports(tree, report)
        run['modules'][transport] = {
            name: report['modules'][name] for name in SERVER_PACKAGES if name in report['modules']
        }
        run['modules'][transport]['count'] = len(report['modules'])
        run['logging'][transport] = report['logging']
        for result in report['results']:
            run['workloads'][result['workload']] = result
    missing = {workload_id(w) for w in workloads} - set(run['workloads'])
    if missing:
        raise SystemExit(f'{tree.rev}: workloads with no result: {sorted(missing)}')
    shutil.rmtree(out_dir)
    return run


def check_imports(tree: Tree, report: dict) -> None:
    """Fail unless every server module, the harness and the case came from ``tree``."""
    outside = {name: f for name, f in report['modules'].items() if not tree.owns(f)}
    for package in SERVER_PACKAGES[:2]:
        if package not in report['modules']:
            outside[package] = 'not imported'
    for name in ('harness', 'case'):
        if not Path(report[name]).resolve().is_relative_to(tree.root.resolve()):
            outside[name] = report[name]
    if outside:
        listing = '\n'.join(f'  {name}: {f}' for name, f in sorted(outside.items()))
        raise SystemExit(f'{tree.rev}: imported from outside {tree.root}:\n{listing}')


def round_value(result: dict) -> float:
    """A round's per-call time: the median over its batches of wall-clock time / n."""
    return statistics.median(b['per_call_s'] for b in result['batches'])


def percentile_bounds(values: list[float]) -> tuple[float, float]:
    """p10 and p90 of ``values`` (inclusive method)."""
    if len(values) < 2:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=10, method='inclusive')
    return cuts[0], cuts[-1]


def sign_test(slower: int, faster: int) -> float:
    """Two-sided exact sign test p-value for ``slower`` vs ``faster`` rounds (ties dropped)."""
    n = slower + faster
    if n == 0:
        return 1.0
    k = max(slower, faster)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def comparison_refs(revs: list[str], reference: str) -> list[str]:
    """The trees every other tree is compared with: the baseline and the reference, if run."""
    return [ref for ref in dict.fromkeys((BASELINE, reference)) if ref in revs]


def summarize(rounds: list[dict], revs: list[str], workloads: list[dict], reference: str) -> dict:
    """Per workload: each tree's median, p10, p90, ratios and paired sign tests."""
    refs = comparison_refs(revs, reference)
    summary = {}
    for workload in workloads:
        wid = workload_id(workload)
        values = {rev: [r['runs'][rev]['workloads'][wid] for r in rounds] for rev in revs}
        per_rev = {}
        for rev in revs:
            series = [round_value(result) for result in values[rev]]
            p10, p90 = percentile_bounds(series)
            per_rev[rev] = {
                'median_s': statistics.median(series),
                'p10_s': p10,
                'p90_s': p90,
                'rounds': len(series),
                'fake_replies_per_call': values[rev][0]['fake_replies_per_call'],
            }
        comparisons = {}
        for rev in revs:
            for ref in refs:
                if ref == rev:
                    continue
                ratio = per_rev[rev]['median_s'] / per_rev[ref]['median_s']
                per_rev[rev][f'ratio_vs_{ref}'] = ratio
                pairs = [
                    (round_value(a), round_value(b)) for a, b in zip(values[rev], values[ref])
                ]
                slower = sum(1 for a, b in pairs if a > b)
                faster = sum(1 for a, b in pairs if a < b)
                comparisons[f'{rev}_vs_{ref}'] = {
                    'ratio_of_medians': ratio,
                    'median_paired_ratio': statistics.median(a / b for a, b in pairs),
                    'rounds_slower': slower,
                    'rounds_faster': faster,
                    'rounds_tied': len(pairs) - slower - faster,
                    'sign_test_p': sign_test(slower, faster),
                }
        summary[wid] = {'workload': workload, 'trees': per_rev, 'comparisons': comparisons}
    return summary


def overlap_check(summary: dict, rev: str) -> dict:
    """Per transport and n at 50 ms on ``rev``: does the concurrent batch overlap calls?"""
    checks = {}
    for transport in TRANSPORTS:
        for n in CALL_COUNTS:
            seq = summary.get(
                workload_id({'transport': transport, 'mode': MODES[0], 'n': n, 'latency_ms': 50})
            )
            con = summary.get(
                workload_id({'transport': transport, 'mode': MODES[1], 'n': n, 'latency_ms': 50})
            )
            if seq is None or con is None or rev not in seq['trees']:
                continue
            share = con['trees'][rev]['median_s'] / seq['trees'][rev]['median_s']
            checks[f'{transport}-n{n}'] = {
                'concurrent_over_sequential': share,
                'overlaps': share < OVERLAP_SHARE,
            }
    return checks


def is_slower(comparison: dict) -> bool:
    """Above the bar by medians, p < 0.05, and the sign test points the same way."""
    return (
        comparison['ratio_of_medians'] > SLOWER_RATIO
        and comparison['sign_test_p'] < SIGNIFICANCE
        and comparison['rounds_slower'] > comparison['rounds_faster']
    )


def verdicts(summary: dict, candidate: str, reference: str) -> list[dict]:
    """Workloads where ``candidate`` is slower than ``reference`` (see ``is_slower``)."""
    found = []
    for wid, entry in summary.items():
        comparison = entry['comparisons'].get(f'{candidate}_vs_{reference}')
        if comparison is not None and is_slower(comparison):
            found.append({'workload': wid, **comparison})
    return found


def format_table(summary: dict, revs: list[str], reference: str) -> str:
    """A Markdown table: per workload, median per-call ms [p10-p90] per tree, and ratios."""
    header = ['workload', *[f'{rev} median [p10-p90] ms' for rev in revs]]
    ratio_columns = [
        (rev, ref)
        for rev in revs
        for ref in comparison_refs(revs, reference)
        if ref != rev and not (rev == BASELINE and ref == reference)
    ]
    header += [f'{rev}/{ref} (slower rounds, p)' for rev, ref in ratio_columns]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for wid, entry in summary.items():
        cells = [wid]
        for rev in revs:
            t = entry['trees'][rev]
            cells.append(
                f'{t["median_s"] * 1000:.3f} [{t["p10_s"] * 1000:.3f}-{t["p90_s"] * 1000:.3f}]'
            )
        for rev, ref in ratio_columns:
            c = entry['comparisons'][f'{rev}_vs_{ref}']
            rounds = c['rounds_slower'] + c['rounds_faster'] + c['rounds_tied']
            cells.append(
                f'{c["ratio_of_medians"]:.3f} ({c["rounds_slower"]}/{rounds}, '
                f'p={c["sign_test_p"]:.2g})'
            )
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def overlap_rev(revs: list[str]) -> str:
    """The tree the overlap check runs on: 508e806 when it is in the run, else the first."""
    return BASELINE if BASELINE in revs else revs[0]


def build_results(meta: dict, rounds: list[dict], revs: list[str], workloads: list[dict]) -> dict:
    """The results of the rounds finished so far: summary, overlap check, verdicts, table."""
    reference = meta['reference']
    summary = summarize(rounds, revs, workloads, reference)
    on = overlap_rev(revs)
    meta = {
        **meta,
        'rounds_done': len(rounds),
        'complete': len(rounds) == meta['rounds'],
        'loadavg_end': os.getloadavg(),
        'finished': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    return {
        'meta': meta,
        'summary': summary,
        'overlap_50ms': {'rev': on, 'checks': overlap_check(summary, on)},
        'slower_than_reference': verdicts(summary, meta['candidate'], reference),
        'table': format_table(summary, revs, reference),
        'rounds': rounds,
    }


def write_results(out: Path, results: dict) -> None:
    """Replace ``out`` with ``results`` in one step (a temporary file, then a rename)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + '.partial')
    partial.write_text(json.dumps(results, indent=1) + '\n', encoding='utf-8')
    os.replace(partial, out)


def benchmark(args, trees: list[Tree], workloads: list[dict], scratch: Path, out: Path) -> dict:
    """Run the rounds, rotating the trees' order; write the results to ``out`` after each."""
    rounds_count = QUICK_ROUNDS if args.quick else args.rounds
    options = {'warmup': args.warmup, 'batches': args.batches, 'profile': None}
    meta = {
        'command': sys.argv,
        'quick': args.quick,
        'rounds': rounds_count,
        'warmup': args.warmup,
        'batches': args.batches,
        'trees': [{'rev': t.rev, 'sha': t.sha} for t in trees],
        'candidate': args.candidate,
        'reference': args.reference,
        'workloads': [workload_id(w) for w in workloads],
        'run_timeout_s': run_timeout_s(workloads, options),
        'python': sys.version,
        'mcp': importlib.metadata.version('mcp'),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'started': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'loadavg_start': os.getloadavg(),
    }
    revs = [t.rev for t in trees]
    rounds = []
    results = {}
    for index in range(rounds_count):
        shift = index % len(trees)
        order = trees[shift:] + trees[:shift]
        record = {'round': index, 'order': [t.rev for t in order], 'runs': {}}
        record['loadavg'] = os.getloadavg()
        for tree in order:
            record['runs'][tree.rev] = run_tree(tree, workloads, scratch, options)
        rounds.append(record)
        results = build_results(meta, rounds, revs, workloads)
        write_results(out, results)
        elapsed = ', '.join(
            f'{rev} {run["elapsed_s"]:.1f}s' for rev, run in record['runs'].items()
        )
        print(f'round {index + 1}/{rounds_count}: {elapsed}; written to {out}', flush=True)
    return results


def profile(args, trees: list[Tree], workloads: list[dict], scratch: Path) -> None:
    """Profile one workload on every tree with cProfile; print where the time goes."""
    wanted = [w for w in workloads if workload_id(w) == args.profile]
    if not wanted:
        raise SystemExit(
            f'unknown workload {args.profile!r}: {[workload_id(w) for w in workloads]}'
        )
    out_dir = Path(args.profile_out or tempfile.mkdtemp(prefix='bench_tools_call_profile_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = wanted[0]['n'] * args.batches
    stats = {}
    for tree in trees:
        tree_dir = out_dir / tree.rev
        tree_dir.mkdir(exist_ok=True)
        options = {'warmup': args.warmup, 'batches': args.batches, 'profile': str(tree_dir)}
        run = run_tree(tree, wanted, scratch, options)
        per_call = round_value(run['workloads'][args.profile])
        path = tree_dir / f'{args.profile}.pstats'
        stats[tree.rev] = load_stats(path, tree)
        print(f'{tree.rev}: {per_call * 1000:.3f} ms per call (profiled), stats in {path}')
    for rev, table in stats.items():
        print(f'\n== {rev}: top {args.top} by own time, us per call (own, cumulative, calls/call)')
        top = sorted(table.items(), key=lambda kv: kv[1][0], reverse=True)[: args.top]
        for name, (own, cumulative, ncalls) in top:
            print(
                f'{own / calls * 1e6:9.1f} {cumulative / calls * 1e6:9.1f} {ncalls / calls:7.2f}  {name}'
            )
    revs = list(stats)
    for rev in revs[1:]:
        base = stats[revs[0]]
        other = stats[rev]
        names = set(base) | set(other)
        print(f'\n== {rev} - {revs[0]}: cumulative us per call, largest differences')
        diffs = sorted(
            names,
            key=lambda name: abs(other.get(name, (0, 0, 0))[1] - base.get(name, (0, 0, 0))[1]),
            reverse=True,
        )
        for name in diffs[: args.top]:
            b = base.get(name, (0, 0, 0))
            o = other.get(name, (0, 0, 0))
            print(
                f'{(o[1] - b[1]) / calls * 1e6:+9.1f} cum {(o[0] - b[0]) / calls * 1e6:+9.1f} own '
                f'{(o[2] - b[2]) / calls:+7.2f} calls  {name}'
            )


def load_stats(path: Path, tree: Tree) -> dict:
    """``{function: (own s, cumulative s, calls)}``, paths made comparable across trees."""
    raw = pstats.Stats(str(path)).stats
    table: dict = {}
    for (filename, line, function), (_cc, ncalls, own, cumulative, _callers) in raw.items():
        name = f'{short_path(filename, tree)}:{line}({function})'
        prev = table.get(name, (0.0, 0.0, 0))
        table[name] = (prev[0] + own, prev[1] + cumulative, prev[2] + ncalls)
    return table


def short_path(filename: str, tree: Tree) -> str:
    """``filename`` with the tree or venv prefix replaced by ``<tree>`` or ``<venv>``."""
    for prefix, label in ((str(tree.root), '<tree>'), (sys.prefix, '<venv>')):
        if filename.startswith(prefix):
            return label + filename[len(prefix) :]
    return filename


def context_cost() -> dict:
    """Time building one SDK ``Context`` as ``MCPServer._handle_call_tool`` does per call.

    Unprofiled, with ``timeit`` (``autorange``, then ``CONTEXT_COST_REPEATS`` repeats), in
    this interpreter: the trees run on the same venv, so on the same SDK.
    """
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import CallToolRequestParams

    server = MCPServer('bench-context-cost')
    params = CallToolRequestParams(name='ListIndexTool', arguments={})
    # Context stores the request context as a private attribute without validating it.
    request_context = object()
    subscriptions = server._subscriptions

    def build():
        return Context(
            request_context=request_context,
            mcp_server=server,
            input_params=params,
            subscriptions=subscriptions,
        )

    timer = timeit.Timer(build)
    number, _ = timer.autorange()
    samples = [t / number for t in timer.repeat(repeat=CONTEXT_COST_REPEATS, number=number)]
    return {
        'mcp': importlib.metadata.version('mcp'),
        'python': sys.version.split()[0],
        'loops': number,
        'repeats': CONTEXT_COST_REPEATS,
        'min_s': min(samples),
        'median_s': statistics.median(samples),
        'loadavg': os.getloadavg(),
    }


def parse_args() -> argparse.Namespace:
    """The command line; sets ``args.revs``, ``args.candidate`` and ``args.reference``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--quick',
        action='store_true',
        help=f'{QUICK_ROUNDS} rounds; checks only that every workload ran on every tree',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=FULL_ROUNDS,
        help=f'default {FULL_ROUNDS}; fewer only with --out (never into {DEFAULT_OUT.name})',
    )
    parser.add_argument('--rev', action='append', default=[], help='add a revision (repeatable)')
    parser.add_argument(
        '--only-revs', action='store_true', help='run only the --rev revisions, not the defaults'
    )
    parser.add_argument(
        '--candidate', help=f'the tree the verdict is about (default {CANDIDATE}); must be run'
    )
    parser.add_argument(
        '--reference', help=f'the tree it is compared with (default {REFERENCE}); must be run'
    )
    parser.add_argument('--transport', choices=TRANSPORTS, help='run only this transport')
    parser.add_argument('--warmup', type=int, default=2, help='untimed calls per session')
    parser.add_argument('--batches', type=int, default=1, help='timed batches per session')
    parser.add_argument('--out', type=Path, help=f'results JSON (full run: {DEFAULT_OUT})')
    parser.add_argument('--profile', metavar='WORKLOAD', help='cProfile one workload instead')
    parser.add_argument('--profile-out', type=Path, help='directory for the .pstats files')
    parser.add_argument('--top', type=int, default=25, help='rows per profile listing')
    parser.add_argument(
        '--context-cost',
        action='store_true',
        help='time one SDK Context as MCPServer builds it per tools/call, unprofiled; no trees',
    )
    args = parser.parse_args()
    if args.only_revs and not args.rev:
        parser.error('--only-revs needs at least one --rev')
    if args.rounds < 1 or args.batches < 1 or args.warmup < 0:
        parser.error('--rounds and --batches must be at least 1, --warmup at least 0')
    full = not args.quick and args.profile is None and not args.context_cost
    if (
        full
        and args.rounds < FULL_ROUNDS
        and (args.out is None or args.out.resolve() == DEFAULT_OUT)
    ):
        parser.error(
            f'a run of fewer than {FULL_ROUNDS} rounds needs --out (not {DEFAULT_OUT.name})'
        )
    revs = list(args.rev) if args.only_revs else [*DEFAULT_REVS, *args.rev]
    args.revs = list(dict.fromkeys(revs))
    for name, default in (('candidate', CANDIDATE), ('reference', REFERENCE)):
        value = getattr(args, name)
        if value is not None and value not in args.revs:
            parser.error(f'--{name} {value} is not one of the revisions run: {args.revs}')
        setattr(args, name, value or default)
    return args


def main() -> int:
    """Extract the trees, run the benchmark (or profile), write and print the results."""
    args = parse_args()
    if args.context_cost:
        cost = context_cost()
        print(
            f'SDK Context per tools/call (mcp {cost["mcp"]}, Python {cost["python"]}, '
            f'{cost["repeats"]} x {cost["loops"]} loops, unprofiled): '
            f'min {cost["min_s"] * 1e6:.1f} us, median {cost["median_s"] * 1e6:.1f} us; '
            f'loadavg {cost["loadavg"]}'
        )
        return 0
    workloads = all_workloads()
    if args.transport:
        workloads = [w for w in workloads if w['transport'] == args.transport]
    out = args.out or (None if args.quick else DEFAULT_OUT)
    if out is None:
        out = Path(tempfile.gettempdir()) / 'bench_tools_call_quick.json'
    scratch = Path(tempfile.mkdtemp(prefix='bench_tools_call_'))
    trees: list[Tree] = []
    try:
        # Inside the try: a revision that does not resolve still removes the scratch dir.
        for rev in args.revs:
            trees.append(Tree(rev, scratch / 'trees' / rev))
        for tree in trees:
            tree.extract()
        if args.profile:
            profile(args, trees, workloads, scratch)
            return 0
        results = benchmark(args, trees, workloads, scratch, out)
    finally:
        for tree in trees:
            if tree.root.exists():
                set_writable(tree.root, True)
        shutil.rmtree(scratch, ignore_errors=True)
    print(results['table'])
    overlap = results['overlap_50ms']
    print(f'overlap at 50 ms on {overlap["rev"]}:', json.dumps(overlap['checks']))
    meta = results['meta']
    if args.quick:
        print(
            f'quick: {len(workloads)} workloads x {len(trees)} trees x {QUICK_ROUNDS} rounds ran, '
            'each importing its own tree; no conclusions from 2 rounds'
        )
    elif meta['candidate'] in args.revs and meta['reference'] in args.revs:
        print(
            f'{meta["candidate"]} slower than {meta["reference"]} '
            f'(>{SLOWER_RATIO - 1:.0%}, p<{SIGNIFICANCE}, more rounds slower):',
            results['slower_than_reference'],
        )
    else:
        print(f'no verdict: {meta["candidate"]} or {meta["reference"]} was not run')
    print(f'loadavg start {meta["loadavg_start"]} end {meta["loadavg_end"]}; results in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
