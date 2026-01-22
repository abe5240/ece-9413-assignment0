# Assignment 0 — Negacyclic NTT in JAX

Implement a **negacyclic** Number Theoretic Transform (NTT) in JAX.

Your implementation goes in **`student.py`** — that's the only file you edit.

---

## What you are implementing

The forward **negacyclic** NTT for polynomials modulo `x^N + 1`:

```
y[k] = sum_{n=0}^{N-1} x[n] * psi^{(2k+1)n}   (mod q)
```

Where:
* `N` is the transform size
* `q` is a prime modulus where `(q - 1)` is divisible by `2N`
* `psi` is a primitive `2N`-th root of unity modulo `q`, so `psi^N ≡ -1 (mod q)`

Your function must handle inputs shaped `(N,)` and `(B, N)` (batch dimension `B`).

---

## Setup

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then from this directory:

```bash
uv sync
```

This installs CPU JAX. For GPU, see https://docs.jax.dev/en/latest/installation.html

---

## Running tests

```bash
uv run pytest
```

---

## Running benchmarks

```bash
uv run python -m tests.benchmark
```

Options:
```bash
uv run python -m tests.benchmark --no-correctness   # skip tests, just benchmark
uv run python -m tests.benchmark --no-latency       # just run tests
uv run python -m tests.benchmark --min-logn 10 --max-logn 18
```

---

## Submission

```bash
bash make_submission.sh
```

This runs tests and produces `code.zip`. Upload to Brightspace.

---

## Helpers (optional)

Generate your own `(q, psi)` pairs for experimentation:

```python
import provided

N = 1 << 14
q = provided.generate_ntt_modulus(N, bit_length=31)
psi = provided.negacyclic_psi(N, q)
```
