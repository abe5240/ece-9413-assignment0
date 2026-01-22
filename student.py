"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """(a + b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    s = a + b
    return jnp.where(s >= q, s - q, s)


def mod_sub(a, b, q):
    """(a - b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    d = a + (q - b)
    return jnp.where(d >= q, d - q, d)


def mod_mul(a, b, q):
    """(a · b) mod q, elementwise.

    Uses uint64 intermediate to avoid overflow.
    """
    q = jnp.asarray(q, dtype=jnp.uint64)
    prod = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    return (prod % q).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Precomputation
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def precompute_tables(N, q, psi):
    """
    Precompute power tables for negacyclic NTT.

    Args:
        N: Transform size (must be power of two)
        q: Prime modulus
        psi: Primitive 2N-th root of unity

    Returns:
        tuple: (psi_powers, twiddles) as uint32 arrays

    Where:
        psi_powers[n] = ψ^n mod q (for negacyclic twist)
        twiddles[span:2*span] = stage twiddles for Stockham NTT
    """
    if N <= 0 or N & (N - 1) != 0:
        raise ValueError(f"N must be a positive power of two, got {N}")

    q, psi = int(q), int(psi)
    omega = pow(psi, 2, q)

    # ψ^n for negacyclic twist (use Python int to avoid uint32 overflow)
    psi_powers = np.empty(N, dtype=np.uint32)
    cur = 1
    for i in range(N):
        psi_powers[i] = cur
        cur = (cur * psi) % q

    # Stockham stage twiddles for cyclic NTT with ω = ψ²
    twiddles = np.ones(N, dtype=np.uint32)
    stages = N.bit_length() - 1

    for s in range(stages):
        span = 1 << s
        stride = N // (2 * span)
        step = pow(omega, stride, q)
        cur = 1
        for j in range(span):
            twiddles[span + j] = cur
            cur = (cur * step) % q

    return psi_powers, twiddles


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

def stockham_ntt(x, q, twiddles):
    """
    Cyclic radix-2 NTT using Stockham autosort.

    Args:
        x: Input array, shape (batch, N)
        q: Prime modulus
        twiddles: Precomputed twiddle table, shape (N,)

    Returns:
        jnp.ndarray: NTT output, shape (batch, N)
    """
    N = x.shape[-1]
    stages = N.bit_length() - 1

    for s in range(stages):
        span = 1 << s
        stride = N // (2 * span)

        x = x.reshape((x.shape[0], span, 2, stride))
        x = x.transpose((0, 2, 1, 3))

        a, b = x[:, 0], x[:, 1]
        tw = twiddles[span:2 * span][None, :, None]
        b = mod_mul(b, tw, q)

        top = mod_add(a, b, q)
        bot = mod_sub(a, b, q)

        x = jnp.stack((top, bot), axis=1).reshape((x.shape[0], N))

    return x


def ntt(x, *, q, psi):
    """
    Compute the forward negacyclic NTT.

    Args:
        x: Input coefficients, shape (N,) or (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi: Primitive 2N-th root of unity (ψ^N ≡ -1 mod q)

    Returns:
        jnp.ndarray: NTT output, same shape as input

    Example:
        >>> x = jnp.array([1, 2, 3, 4], dtype=jnp.uint32)
        >>> y = ntt(x, q=17, psi=2)
        >>> y.shape
        (4,)
    """
    x = jnp.asarray(x, dtype=jnp.uint32)

    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    elif x.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    N = x.shape[-1]
    psi_powers, twiddles = precompute_tables(N, int(q), int(psi))
    psi_powers = jnp.asarray(psi_powers)
    twiddles = jnp.asarray(twiddles)

    # Negacyclic twist: x'[n] = x[n] · ψ^n
    x = mod_mul(x, psi_powers[None, :], q)

    # Cyclic NTT with ω = ψ²
    y = stockham_ntt(x, q, twiddles)

    return y[0] if squeeze else y
