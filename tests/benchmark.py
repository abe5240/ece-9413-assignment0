"""
Benchmark runner for NTT latency measurement.

Usage:
    uv run python -m tests.benchmark
    uv run python -m tests.benchmark --no-latency
    uv run python -m tests.benchmark --no-correctness
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

import provided
import student


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MIN_LOGN = 10
DEFAULT_MAX_LOGN = 18
DEFAULT_BATCH = 1
DEFAULT_RUNS = 20
DEFAULT_WARMUP = 5
DEFAULT_BIT_LENGTH = 31
DEFAULT_SEED = 42


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchStats:
    """Timing statistics for a single benchmark configuration."""
    compile_s: float
    median_s: float
    p90_s: float
    min_s: float
    max_s: float


# -----------------------------------------------------------------------------
# Benchmark Helpers
# -----------------------------------------------------------------------------

def summarize(compile_s, times):
    """
    Compute summary statistics from timing samples.

    Args:
        compile_s: JIT compilation time in seconds
        times: List of execution times in seconds

    Returns:
        BenchStats: Summary statistics
    """
    arr = np.asarray(times, dtype=np.float64)
    return BenchStats(
        compile_s=compile_s,
        median_s=float(np.median(arr)),
        p90_s=float(np.quantile(arr, 0.90)),
        min_s=float(np.min(arr)),
        max_s=float(np.max(arr)),
    )


def bench_single(N, batch, q, psi, runs, warmup, rng):
    """
    Benchmark NTT at a single (N, batch) configuration.

    Args:
        N: Transform size
        batch: Batch size
        q: Prime modulus
        psi: Primitive 2N-th root of unity
        runs: Number of timed iterations
        warmup: Number of warmup iterations (includes compile)
        rng: NumPy random generator

    Returns:
        BenchStats: Timing statistics
    """
    x_np = rng.integers(0, q, size=(batch, N), dtype=np.int64)
    x = jnp.asarray(x_np, dtype=jnp.uint32)

    fn = jax.jit(lambda z: student.ntt(z, q=q, psi=psi))

    # First call triggers JIT compilation
    t0 = time.perf_counter()
    y = fn(x)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t0

    # Additional warmup
    for _ in range(max(0, warmup - 1)):
        y = fn(x)
        jax.block_until_ready(y)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        times.append(time.perf_counter() - t0)

    return summarize(compile_s, times)


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------

def run_correctness():
    """Run pytest test suite."""
    subprocess.run(["uv", "run", "pytest"], check=True)


def run_latency(args):
    """
    Run latency benchmarks and print results table.

    Args:
        args: Parsed command-line arguments
    """
    console = Console()

    if args.min_logn > args.max_logn:
        raise ValueError(
            f"min_logn ({args.min_logn}) must be <= max_logn ({args.max_logn})"
        )

    N_max = 1 << args.max_logn
    if args.q is None:
        q = provided.generate_ntt_modulus(
            N_max,
            bit_length=args.bit_length,
        )
    else:
        q = args.q
    psi_max = provided.negacyclic_psi(N_max, q)
    rng = np.random.default_rng(args.seed)

    table = Table(title="Negacyclic NTT Latency")
    table.add_column("log₂(N)", justify="right")
    table.add_column("N", justify="right")
    table.add_column("compile (ms)", justify="right")
    table.add_column("median (ms)", justify="right")
    table.add_column("p90 (ms)", justify="right")
    table.add_column("Mcoeff/s", justify="right")

    for logn in range(args.min_logn, args.max_logn + 1):
        N = 1 << logn
        psi = provided.negacyclic_psi_from_max(psi_max, N_max, N, q)

        stats = bench_single(
            N=N,
            batch=args.batch,
            q=q,
            psi=psi,
            runs=args.runs,
            warmup=args.warmup,
            rng=rng,
        )

        throughput = (args.batch * N) / stats.median_s / 1e6

        table.add_row(
            str(logn),
            str(N),
            f"{stats.compile_s * 1e3:.2f}",
            f"{stats.median_s * 1e3:.3f}",
            f"{stats.p90_s * 1e3:.3f}",
            f"{throughput:.2f}",
        )

    console.print(table)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    """Build argument parser for benchmark CLI."""
    p = argparse.ArgumentParser(
        description="Run correctness tests and/or NTT latency benchmarks."
    )

    p.add_argument(
        "--correctness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pytest suite (default: yes)",
    )
    p.add_argument(
        "--latency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run latency benchmarks (default: yes)",
    )
    p.add_argument(
        "--q",
        type=int,
        default=None,
        help="Prime modulus (auto-generated if omitted)",
    )
    p.add_argument(
        "--bit-length",
        type=int,
        default=DEFAULT_BIT_LENGTH,
        help=(
            "Bit-length for auto-generated q "
            f"(default: {DEFAULT_BIT_LENGTH})"
        ),
    )
    p.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )
    p.add_argument(
        "--min-logn",
        type=int,
        default=DEFAULT_MIN_LOGN,
        help=f"Smallest log2(N) (default: {DEFAULT_MIN_LOGN})",
    )
    p.add_argument(
        "--max-logn",
        type=int,
        default=DEFAULT_MAX_LOGN,
        help=f"Largest log2(N) (default: {DEFAULT_MAX_LOGN})",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Timed iterations per size (default: {DEFAULT_RUNS})",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=(
            "Warmup iterations including compile "
            f"(default: {DEFAULT_WARMUP})"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )

    return p


def main():
    """Entry point for benchmark CLI."""
    args = build_parser().parse_args()

    if args.correctness:
        run_correctness()
    if args.latency:
        run_latency(args)


if __name__ == "__main__":
    main()
