"""
Public tests for the negacyclic NTT implementation.

Covers:
- Reference agreement across sizes and batches
- JIT and vmap consistency
- Linearity in the ring

Usage:
    pytest test_ntt.py          # Full mode (default)
    pytest test_ntt.py --fast   # Fast mode (subset of sizes)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import provided
import student
from tests.reference import negacyclic_ntt_oracle


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

FAST_LOG_SIZES = (4, 8, 10)
FULL_LOG_SIZES = (1, 2, 3, 4, 5, 8, 10, 13, 15, 17, 20)

FAST_BATCHES = (1, 4)
FULL_BATCHES = (1, 4, 16, 256)

SEED = 42


def pytest_generate_tests(metafunc):
    """Inject log_sizes and batches based on --fast flag."""
    fast = metafunc.config.getoption("--fast")

    if "logn" in metafunc.fixturenames:
        sizes = FAST_LOG_SIZES if fast else FULL_LOG_SIZES
        metafunc.parametrize("logn", sizes)

    if "batch" in metafunc.fixturenames:
        batches = FAST_BATCHES if fast else FULL_BATCHES
        metafunc.parametrize("batch", batches)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ntt_params(request):
    """Generate shared (q, psi_max, N_max) for all tests."""
    fast = request.config.getoption("--fast")
    max_logn = max(FAST_LOG_SIZES if fast else FULL_LOG_SIZES)

    N_max = 1 << max_logn
    q = provided.generate_ntt_modulus(N_max, bit_length=31)
    psi_max = provided.negacyclic_psi(N_max, q)
    return q, psi_max, N_max


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def params_for_size(logn, ntt_params):
    """
    Extract NTT parameters for a specific log2 size.

    Args:
        logn: Log2 of transform size
        ntt_params: Tuple of (q, psi_max, N_max)

    Returns:
        tuple: (N, q, psi) for the requested size
    """
    q, psi_max, N_max = ntt_params
    N = 1 << logn
    psi = provided.negacyclic_psi_from_max(psi_max, N_max, N, q)
    return N, q, psi


def random_input(rng, q, shape):
    """
    Generate random uint32 array with values in [0, q).

    Args:
        rng: NumPy random generator
        q: Modulus (exclusive upper bound)
        shape: Output shape

    Returns:
        jnp.ndarray: Random values as uint32
    """
    x = rng.integers(0, q, size=shape, dtype=np.int64)
    return jnp.asarray(x, dtype=jnp.uint32)


def to_int64(x):
    """Convert JAX array to int64 NumPy array."""
    return np.asarray(x, dtype=np.int64)


def reference_ntt(x_np, q, psi):
    """
    Compute batched reference NTT using SymPy oracle.

    Args:
        x_np: Input array, shape (batch, N)
        q: Modulus
        psi: Primitive 2N-th root of unity

    Returns:
        np.ndarray: Reference outputs, shape (batch, N)
    """
    refs = [negacyclic_ntt_oracle(row.tolist(), q=q, psi=psi) for row in x_np]
    return np.asarray(refs, dtype=np.int64)


# -----------------------------------------------------------------------------
# Correctness Tests
# -----------------------------------------------------------------------------

def test_matches_reference(logn, batch, ntt_params):
    """NTT output matches SymPy reference across sizes and batch shapes."""
    N, q, psi = params_for_size(logn, ntt_params)

    rng = np.random.default_rng(SEED)
    x = random_input(rng, q, shape=(batch, N))
    y = student.ntt(x, q=q, psi=psi)

    assert y.shape == x.shape

    y_np = to_int64(y) % q
    ref = reference_ntt(to_int64(x), q, psi)
    np.testing.assert_array_equal(y_np, ref)


# -----------------------------------------------------------------------------
# JAX Compatibility Tests
# -----------------------------------------------------------------------------

def test_jit_matches_eager(logn, ntt_params):
    """JIT-compiled NTT matches eager execution."""
    N, q, psi = params_for_size(logn, ntt_params)

    rng = np.random.default_rng(SEED)
    x = random_input(rng, q, shape=(1, N))

    y_eager = student.ntt(x, q=q, psi=psi)
    y_jit = jax.jit(lambda z: student.ntt(z, q=q, psi=psi))(x)
    jax.block_until_ready(y_jit)

    np.testing.assert_array_equal(to_int64(y_eager), to_int64(y_jit))


def test_vmap_matches_direct(logn, batch, ntt_params):
    """vmap over batch dimension matches direct batched call."""
    N, q, psi = params_for_size(logn, ntt_params)

    rng = np.random.default_rng(SEED)
    x = random_input(rng, q, shape=(batch, N))

    y_direct = student.ntt(x, q=q, psi=psi)
    y_vmapped = jax.vmap(lambda row: student.ntt(row, q=q, psi=psi))(x)
    jax.block_until_ready(y_vmapped)

    assert y_vmapped.shape == y_direct.shape
    np.testing.assert_array_equal(to_int64(y_vmapped), to_int64(y_direct))


# -----------------------------------------------------------------------------
# Algebraic Property Tests
# -----------------------------------------------------------------------------

def test_linearity(logn, ntt_params):
    """NTT is linear: NTT(a + b) ≡ NTT(a) + NTT(b) (mod q)."""
    N, q, psi = params_for_size(logn, ntt_params)

    rng = np.random.default_rng(SEED)
    a = random_input(rng, q, shape=(N,))
    b = random_input(rng, q, shape=(N,))

    left = student.ntt((a + b) % q, q=q, psi=psi)
    right = (student.ntt(a, q=q, psi=psi) + student.ntt(b, q=q, psi=psi)) % q

    np.testing.assert_array_equal(to_int64(left), to_int64(right))
