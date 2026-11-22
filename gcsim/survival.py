"""Scalar survival functions.

The simulator evaluates survival for a handful of segments/groups per
collection and for a bounded set of old-generation blocks per occupancy
update.  Those arrays are small (tens of entries), so scalar ``math`` calls
are faster than the numpy dispatch overhead.  This is the only survival
implementation used by the runtime; it is cheap enough that vectorising it
would add more overhead than it saves.
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def surv(kind: str, params, age: float) -> float:
    """P(lifetime > age) for one object/byte of the given class."""
    if age <= 0.0:
        return 1.0
    if kind == "lognormal_mix":
        med, sig, lf, lmed, lsig = params
        body = 0.5 * math.erfc((math.log(age) - math.log(med)) / (sig * _SQRT2))
        if lf <= 0.0:
            return body
        tail = 0.5 * math.erfc((math.log(age) - math.log(lmed)) / (lsig * _SQRT2))
        return (1.0 - lf) * body + lf * tail
    if kind == "pareto":
        xm, alpha, tmax = params
        if age <= xm:
            return 1.0
        if age >= tmax:
            return 0.0
        return (xm / age) ** alpha
    if kind == "deterministic":
        life, jit = params
        if jit <= 0.0:
            return 1.0 if age < life else 0.0
        lo = life * (1.0 - jit)
        hi = life * (1.0 + jit)
        if age <= lo:
            return 1.0
        if age >= hi:
            return 0.0
        return (hi - age) / (hi - lo)
    raise ValueError(f"unknown survival kind {kind!r}")


def surv_batch(kind: str, params, ages) -> list:
    return [surv(kind, params, a) for a in ages]


def quantile_age(kind: str, params, p: float) -> float:
    """Age at which survival drops to p -- used to bound old-gen horizons."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return 0.0
    if kind == "lognormal_mix":
        med, sig, lf, lmed, lsig = params
        # Solve the mixture numerically by bisection; mixture survival is
        # monotone.
        lo, hi = 1e-6, 1e12
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if surv(kind, params, mid) > p:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)
    if kind == "pareto":
        xm, alpha, tmax = params
        t = xm * (1.0 / p) ** (1.0 / alpha)
        return min(t, tmax)
    if kind == "deterministic":
        life, jit = params
        return life * (1.0 + jit) if p < 1 else life * (1.0 - jit)
    return math.inf
