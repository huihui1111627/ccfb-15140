"""Cohort-level (expectation-exact) workload model.

Why cohorts
-----------
A trading front end allocates hundreds of thousands of objects per second; a
full day is on the order of a billion objects.  Simulating each object would
make the tool uselessly slow and would drown the GC dynamics in sampling noise.
The collector behaviour we care about is governed by *aggregate* quantities:
bytes allocated per interval, the survival function of each traffic class, and
how many bytes of each age are resident at the moment of a scavenge.

So the workload is represented as a set of **cohorts**: all objects of one
traffic class allocated during a small time slice ``dt``.  A cohort carries an
expected byte volume and the class's survival function ``S(a)`` (probability a
byte is still reachable ``a`` seconds after allocation).  Every collector
question is then answered in expectation:

    bytes of cohort still live at time t   = V * S(t - birth)
    bytes surviving an interval [a, b)      = V * (S(a) - S(b))

This is exact for the lifetime model in the same sense as a renewal process
mean-field: errors are O(1/sqrt(n)) *relative* on per-collection quantities
only through the finite number of objects in a cohort, and vanishingly small at
production alloc volumes.  The discrete object-level simulator
(:mod:`gcsim.oracle`) implements the same mechanism object by object and is
used to validate the cohort engine.

The survival functions are supplied by :mod:`gcsim.lifetime`, so the *same*
distribution objects define both engine and oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ServiceProfile, TrafficProfile
from .lifetime import TrafficClass, weighted_sizes
from .rng import rng as make_rng
from .rng import poisson


@dataclass
class Cohort:
    """An expected byte volume of one traffic class born in [t0, t0+dt)."""

    birth: float
    dt: float
    mean_size: float
    bytes: float
    class_index: int
    # survival function parameters (class-level)
    kind: str                         # "lognormal_mix" | "pareto"
    p1: tuple                         # model-specific parameters
    p2: Optional[tuple] = None


@dataclass
class CohortTrace:
    """All cohorts of a day plus bucket-level metadata.

    ``births`` is an array of cohort start times (seconds).  Zeros in the
    volume column are dropped except for class coverage.  ``bucket_seconds``
    is retained for allocation-rate reporting.
    """

    cohorts: List[Cohort]
    bucket_seconds: float
    total_bytes: float
    total_objects: float
    class_names: List[str]
    class_mean_sizes: List[float]
    class_shares: List[float]
    traffic: TrafficProfile
    # per bucket (t, bytes, objects) all classes merged, for rate plotting
    bucket_bytes: np.ndarray
    bucket_objects: np.ndarray
    bucket_t0: np.ndarray

    def n_cohorts(self) -> int:
        return len(self.cohorts)


def survival_params(tc: TrafficClass):
    """Encode a lifetime model as (kind, params) for fast array evaluation."""
    m = tc.lifetime
    from .lifetime import (
        RequestLifetime, SessionLifetime, CacheLifetime,
        DeterministicLifetime, BimodalLifetime,
    )
    if isinstance(m, RequestLifetime):
        return (
            "lognormal_mix",
            (m.median_ms / 1000.0, m.sigma, m.long_tail_frac,
             m.long_median_ms / 1000.0, m.long_sigma),
            None,
        )
    if isinstance(m, SessionLifetime):
        return (
            "lognormal_mix",
            (m.median_s, m.sigma, m.long_frac, m.long_median_s, m.long_sigma),
            None,
        )
    if isinstance(m, CacheLifetime):
        return (
            "pareto",
            (m.min_lifetime_s, m.alpha, m.max_lifetime_s),
            None,
        )
    if isinstance(m, DeterministicLifetime):
        return ("deterministic", (m.lifetime_s, m.jitter), None)
    if isinstance(m, BimodalLifetime):
        raise TypeError(
            "BimodalLifetime is supported by the object-level oracle; "
            "use a plain lifetime model for the cohort engine"
        )
    raise TypeError(f"unsupported lifetime model {type(m).__name__}")


def build_trace(
    service: ServiceProfile,
    traffic: TrafficProfile,
    seed: int = 12345,
    cohort_seconds: float = 1.0,
    noise: bool = True,
) -> CohortTrace:
    """Build the cohort representation of one day.

    The traffic curve is sampled every ``cohort_seconds``; each slice's object
    count is either the exact expectation (``noise=False``) or a Poisson draw
    (``noise=True``) so that day-to-day variation can be modelled.  Bursts are
    part of the traffic curve and are therefore represented here.

    ``cohort_seconds`` trades fidelity for speed: 1 s slices resolve pauses and
    scavenge pacing far more finely than any GC interval in these heaps, while
    keeping the cohort count at a few hundred thousand at most.
    """
    total_share = sum(c.share for c in service.classes)
    if total_share <= 0:
        raise ValueError("service has no traffic classes")
    shares, _mean_bytes = weighted_sizes(service.classes)
    class_names = [c.name for c in service.classes]
    mean_sizes = [float(c.mean_size) for c in service.classes]
    enc = [survival_params(c) for c in service.classes]

    n_slices = int(math.ceil(traffic.day_seconds / cohort_seconds))
    tr = make_rng(seed, "cohorts", service.name)
    bucket_bytes = np.zeros(n_slices, dtype=np.float64)
    bucket_objs = np.zeros(n_slices, dtype=np.float64)
    bucket_t0 = np.arange(n_slices, dtype=np.float64) * cohort_seconds
    # Per-slice traffic multiplier (with optional noise) -- vectorised over the
    # curve control points rather than calling traffic.multiplier per slice.
    hours = (bucket_t0 / 3600.0) % 24.0
    mult = np.interp(hours, [p[0] for p in sorted(traffic.curve)],
                     [p[1] for p in sorted(traffic.curve)])
    if traffic.hourly_amplitude:
        mult = mult * (1.0 + traffic.hourly_amplitude *
                       np.sin(2 * np.pi * hours / 24.0))
    for a, b_end, m in traffic.bursts:
        mult = np.where((hours >= a) & (hours <= b_end), mult * m, mult)
    mult = np.maximum(0.0, mult)
    if noise and traffic.noise_sigma > 0:
        noise_vec = np.array([tr.gauss(0.0, traffic.noise_sigma)
                              for _ in range(n_slices)])
        mult = mult * np.maximum(0.0, 1.0 + noise_vec)
    dts = np.minimum(cohort_seconds, traffic.day_seconds - bucket_t0)
    exp_objects = traffic.base_alloc_rate * mult * dts
    if noise:
        # Poisson per slice; vectorised normal approximation for large lambda.
        lam = exp_objects
        counts = np.empty(n_slices, dtype=np.float64)
        small = lam < 60
        if small.any():
            counts[small] = [float(poisson(tr, float(v))) for v in lam[small]]
        big = ~small
        if big.any():
            counts[big] = np.maximum(
                0.0, np.rint(np.array([
                    tr.gauss(float(lam_i), float(lam_i) ** 0.5)
                    for lam_i in lam[big]])))
    else:
        counts = exp_objects
    counts = np.maximum(0.0, counts)

    # Build one cohort per (slice, class) with nonzero bytes.
    births = np.repeat(bucket_t0, len(service.classes))
    rep_counts = np.repeat(counts, len(service.classes))
    rep_dts = np.repeat(dts, len(service.classes))
    class_idx = np.tile(np.arange(len(service.classes)), n_slices)
    shares_arr = np.array([shares[ci] for ci in range(len(service.classes))])
    mean_arr = np.array(mean_sizes)
    objs = rep_counts * shares_arr[class_idx]
    byts = objs * mean_arr[class_idx]
    nz = byts > 0
    idx = np.nonzero(nz)[0]
    cohorts = []
    for j in idx.tolist():
        ci = int(class_idx[j])
        kind, p1, p2 = enc[ci]
        cohorts.append(Cohort(
            birth=float(births[j]), dt=float(rep_dts[j]),
            mean_size=float(mean_arr[ci]), bytes=float(byts[j]),
            class_index=ci, kind=kind, p1=p1, p2=p2,
        ))
    bucket_bytes = np.bincount(
        (idx // len(service.classes)).astype(np.int64), weights=byts[idx],
        minlength=n_slices)
    bucket_objs = np.bincount(
        (idx // len(service.classes)).astype(np.int64), weights=objs[idx],
        minlength=n_slices)
    total_bytes = float(bucket_bytes.sum())
    total_objects = float(bucket_objs.sum())
    return CohortTrace(
        cohorts=cohorts,
        bucket_seconds=cohort_seconds,
        total_bytes=total_bytes,
        total_objects=total_objects,
        class_names=class_names,
        class_mean_sizes=mean_sizes,
        class_shares=shares,
        traffic=traffic,
        bucket_bytes=bucket_bytes,
        bucket_objects=bucket_objs,
        bucket_t0=bucket_t0,
    )
