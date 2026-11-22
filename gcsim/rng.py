"""Deterministic random streams for the GC simulator.

The simulator uses a single seeded RNG stream per service so that every
experiment is exactly reproducible: same service definition + same traffic
curve + same GC parameters => byte-identical GC log and metrics.  Several
independent streams are derived from the master seed so that, e.g., changing
the promotion-age parameter does not perturb the object-lifetime draws (which
would make A/B comparisons noisy).

This module intentionally depends only on the standard library so the core
simulator can be used in any Python >= 3.8 environment.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence


def derive_seed(master: int, *parts: object) -> int:
    """Derive a stable 64-bit seed from a master seed and arbitrary labels."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(master).encode("utf-8"))
    for p in parts:
        h.update(b"\x1f")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def rng(master: int, *parts: object) -> random.Random:
    return random.Random(derive_seed(master, *parts))


def poisson(r: random.Random, lam: float) -> int:
    """Draw a Poisson sample.

    Uses the exact Knuth algorithm for small means and a normal approximation
    for large means; the mean is always the largest request rate in practice,
    so the approximation error is far below what matters for GC behaviour.
    """
    if lam <= 0.0:
        return 0
    if lam < 60.0:
        # Knuth's algorithm, O(lam).
        limit = 2.718281828459045 ** (-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= r.random()
            if p <= limit:
                return k - 1
    # Normal approximation with continuity correction.
    x = r.gauss(lam, lam ** 0.5)
    return max(0, int(round(x)))


def weighted_choice(r: random.Random, weights: Sequence[float]) -> int:
    total = 0.0
    for w in weights:
        if w < 0:
            raise ValueError("weights must be non-negative")
        total += w
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    x = r.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x < acc:
            return i
    return len(weights) - 1


def loguniform(r: random.Random, lo: float, hi: float) -> float:
    """Log-uniform draw on [lo, hi]; safe for lo > 0 only."""
    import math

    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError("loguniform requires 0 < lo <= hi")
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


def pareto(r: random.Random, xm: float, alpha: float) -> float:
    """Pareto draw with minimum xm and tail index alpha (alpha > 1 for finite mean)."""
    u = r.random()
    return xm / (u ** (1.0 / alpha))


def geometric_capped(r: random.Random, p: float, cap: int) -> int:
    """Number of failures before first success, capped (survival-age model)."""
    n = 0
    while n < cap and r.random() >= p:
        n += 1
    return n
