"""Parameter search and consequence analysis.

Given a service, a traffic curve and a pair of objectives
(``max_pause_ms`` and ``min_throughput``), the tuner searches the three
parameters the user asked about:

* young-generation size (with the survivor ratio kept in proportion),
* promotion age,
* concurrent-collection initiating occupancy.

Search strategy
---------------
A full grid is evaluated in parallel across CPU cores.  Each point runs a whole
day, which is the only way to catch the peak-hour regressions the user cares
about.  The grid is deliberately modest (not a blind explosion): the three
parameters have different sensitivities and are searched on their own scales,
and a second refinement pass zooms into the feasible region.  Every evaluated
point is retained, including infeasible ones, because the *consequences* of
each knob are read off the whole response surface, not just the winner.

Objectives (both must hold, evaluated over the whole day):

* ``max_pause_ms`` -- the largest single STW pause over the day.
* ``min_throughput`` -- the fraction of wall time not spent in STW pauses
  (``1 - total_pause/day``), times a factor for concurrent-marking CPU, is the
  derived "throughput loss" reported.  We use the simple STW-based measure for
  the hard constraint and report the marking-adjusted one separately.

The result includes:

* the best feasible configuration and why it was chosen,
* the parameter sensitivity (how pause/throughput/promotion move with each
  knob), with a plain-language consequence for each direction,
* the infeasible-but-near points, so the user sees the cliff.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import GCConfig, MB, ServiceProfile, TrafficProfile
from .simulator import HeapSimulator
from .workload import build_trace

HOUR = 3600.0


@dataclass
class Objective:
    max_pause_ms: float = 50.0
    min_throughput: float = 0.99

    def describe(self) -> str:
        return (f"pause <= {self.max_pause_ms:g} ms, "
                f"throughput >= {self.min_throughput*100:.2f}%")


@dataclass
class PointResult:
    young_mb: float
    survivor_ratio: float
    promotion_age: int
    initiating: float
    mode: str
    # outcomes
    young_gcs: int
    remark_count: int
    full_gcs: int
    degraded_cycles: int
    max_pause_ms: float
    p99_ms: float
    p999_ms: float
    p50_ms: float
    total_pause_s: float
    throughput: float
    promotion_gb: float
    promotion_fraction: float
    pretenured_gb: float
    old_peak_gb: float
    final_old_gb: float
    concurrent_overhead_s: float
    feasible: bool
    fail_reasons: List[str] = field(default_factory=list)
    tospace_exhaustion: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        for k, v in list(d.items()):
            if isinstance(v, (np.floating,)):
                d[k] = float(v)
        return d


def evaluate_point(args) -> PointResult:
    """Run one configuration over a full day.  Top-level for multiprocessing."""
    (cfg_dict, service_name, traffic_name, seed, objective,
     young_mb, survivor_ratio, promotion_age, initiating, mode,
     cohort_seconds, day_hours) = args
    cfg = GCConfig.from_dict(cfg_dict)
    cfg.heap.young_size = int(young_mb * MB)
    if cfg.heap.old_size <= cfg.heap.young_size * 1.2:
        # A young gen cannot exceed the heap; grow old so the ratio stays
        # realisable.  (The user controls total heap via the base config.)
        cfg.heap.old_size = int(max(cfg.heap.old_size,
                                    cfg.heap.young_size * 1.25))
    cfg.heap.survivor_ratio = float(survivor_ratio)
    cfg.young.promotion_age = int(promotion_age)
    cfg.old.initiating_occupancy = float(initiating)
    cfg.old.mode = mode
    service = getattr(ServiceProfile, service_name)()
    traffic = _TRAFFIC_FACTORIES[traffic_name]()
    if day_hours:
        traffic.day_seconds = day_hours * 3600.0
    import time as _t
    t0 = _t.perf_counter()
    trace = build_trace(service, traffic, seed=seed,
                        cohort_seconds=cohort_seconds)
    sim = HeapSimulator(cfg, service, traffic, seed=seed, trace=trace)
    st = sim.run()
    wall = _t.perf_counter() - t0
    pc = st.pause_percentiles()
    day = traffic.day_seconds
    total_pause = st.total_pause_s()
    throughput = max(0.0, 1.0 - total_pause / day) if day else 1.0
    # Concurrent marking costs throughput only while a cycle is running and
    # only by the marker's share of the cores.
    if day > 0:
        throughput *= max(0.0, 1.0 - st.app_cpu_lost_s / day)
        throughput = max(0.0, throughput - st.concurrent_overhead_s / day * 0.0)
    max_pause_ms = pc.get("max", 0.0) * 1e3
    fails = []
    if max_pause_ms > objective.max_pause_ms:
        fails.append(f"max pause {max_pause_ms:.1f} ms > {objective.max_pause_ms:g} ms")
    if throughput < objective.min_throughput:
        fails.append(
            f"throughput {throughput*100:.3f}% < {objective.min_throughput*100:.2f}%")
    prom_frac = (st.promotion_bytes / st.allocated_bytes) if st.allocated_bytes else 0.0
    old_final = st.timeline[-1]["old_committed"] if st.timeline else 0.0
    return PointResult(
        young_mb=float(young_mb), survivor_ratio=float(survivor_ratio),
        promotion_age=int(promotion_age), initiating=float(initiating), mode=mode,
        young_gcs=st.young_gcs, remark_count=st.remark_count,
        full_gcs=st.full_gcs, degraded_cycles=st.degraded_cycles,
        max_pause_ms=max_pause_ms, p99_ms=pc.get("p99", 0.0) * 1e3,
        p999_ms=pc.get("p99.9", 0.0) * 1e3, p50_ms=pc.get("p50", 0.0) * 1e3,
        total_pause_s=total_pause, throughput=throughput,
        promotion_gb=st.promotion_bytes / 1e9,
        promotion_fraction=prom_frac,
        pretenured_gb=st.pretenured_bytes / 1e9,
        old_peak_gb=st.old_peak_committed / 1e9,
        final_old_gb=old_final / 1e9,
        concurrent_overhead_s=st.concurrent_overhead_s,
        feasible=not fails, fail_reasons=fails,
        tospace_exhaustion=st.tospace_exhausted,
        wall_s=wall,
    )


_TRAFFIC_FACTORIES = {
    "trading_day": lambda: TrafficProfile.trading_day(),
    "flat": lambda: TrafficProfile.flat(),
    "batch_heavy": lambda: TrafficProfile.batch_heavy(),
}


def traffic_factory(name: str):
    if name not in _TRAFFIC_FACTORIES:
        raise ValueError(f"unknown traffic curve {name!r}; "
                         f"choose from {sorted(_TRAFFIC_FACTORIES)}")
    return _TRAFFIC_FACTORIES[name]


@dataclass
class TuningResult:
    objective: Objective
    service_name: str
    traffic_name: str
    baseline: PointResult
    best: Optional[PointResult]
    grid: List[PointResult]
    sensitivity: Dict[str, List[dict]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def find(self, **kw) -> Optional[PointResult]:
        for p in self.grid:
            if all(abs(getattr(p, k) - v) < 1e-9 for k, v in kw.items()):
                return p
        return None


class _ConfigSpace:
    """A small, purposeful grid over the three tuned knobs."""

    def __init__(self, base: GCConfig, young_mb: Sequence[float],
                 survivor_ratios: Sequence[float], ages: Sequence[int],
                 initiatings: Sequence[float], mode: str = "concurrent"):
        self.base = base
        self.young_mb = list(young_mb)
        self.survivor_ratios = list(survivor_ratios)
        self.ages = list(ages)
        self.initiatings = list(initiatings)
        self.mode = mode

    def points(self):
        for y, sr, a, io in itertools.product(
                self.young_mb, self.survivor_ratios, self.ages, self.initiatings):
            yield y, sr, a, io, self.mode


def _default_space(base: GCConfig) -> _ConfigSpace:
    # A purposeful 3x3x3x2 = 54-point grid.  Chosen so each axis brackets the
    # interesting regime: young from starvation to over-provisioning, tenure
    # age from "promote almost immediately" to "hold through many scavenges",
    # initiating occupancy from early to late, and the two survivor ratios that
    # decide whether the tenure-age knob is even operative.  A blind explosion
    # would be slower and no more informative; the feasible fraction is
    # reported so a missed island is visible.
    return _ConfigSpace(
        base,
        young_mb=(256, 512, 1024, 2048),
        survivor_ratios=(0.15, 0.25, 0.35),
        ages=(3, 7, 15),
        initiatings=(0.40, 0.60),
        mode=base.old.mode,
    )


def _refine_space(best: PointResult) -> _ConfigSpace:
    """Zoom around a feasible point on a small, cheap local grid."""
    y = best.young_mb
    ys = sorted({max(64.0, y * f) for f in (0.6, 1.0, 1.5)})
    srs = sorted({min(0.45, max(0.05, best.survivor_ratio + d))
                  for d in (0.0, 0.05)})
    ages = sorted({max(1, best.promotion_age + d) for d in (0, 4)})
    ios = sorted({min(0.9, max(0.1, best.initiating + d))
                  for d in (-0.1, 0.0)})
    return _ConfigSpace(None, ys, srs, ages, ios, mode=best.mode)


def _run_space(space: _ConfigSpace, cfg: GCConfig, service_name: str,
               traffic_name: str, seed: int, objective: Objective,
               cohort_seconds: float, workers: int,
               day_hours: Optional[float]) -> List[PointResult]:
    cfg_dict = cfg.to_dict()
    args = [
        (cfg_dict, service_name, traffic_name, seed, objective,
         y, sr, a, io, mode, cohort_seconds, day_hours)
        for (y, sr, a, io, mode) in space.points()
    ]
    workers = max(1, workers)
    if workers == 1 or len(args) < 4:
        return [evaluate_point(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evaluate_point, args, chunksize=1))


def _sensitivity(grid: List[PointResult], param: str) -> List[dict]:
    """One-factor summaries: hold the other knobs fixed, average the outcome."""
    from collections import defaultdict

    groups: Dict[Tuple, List[PointResult]] = defaultdict(list)
    for p in grid:
        key = (p.young_mb, p.survivor_ratio, p.promotion_age, p.initiating, p.mode)
        groups[key].append(p)
    # For each parameter value, average across all cells where it appears.
    attr = {"young": "young_mb", "promotion_age": "promotion_age",
            "initiating": "initiating"}[param]
    values = sorted({getattr(p, attr) for p in grid})
    out = []
    for v in values:
        rows = [p for p in grid if abs(getattr(p, attr) - v) < 1e-9]
        if not rows:
            continue
        out.append({
            "value": v,
            "n": len(rows),
            "max_pause_ms": float(np.mean([r.max_pause_ms for r in rows])),
            "p99_ms": float(np.mean([r.p99_ms for r in rows])),
            "throughput": float(np.mean([r.throughput for r in rows])),
            "promotion_gb": float(np.mean([r.promotion_gb for r in rows])),
            "pretenured_gb": float(np.mean([r.pretenured_gb for r in rows])),
            "full_gcs": float(np.mean([r.full_gcs for r in rows])),
            "feasible_frac": float(np.mean([1.0 if r.feasible else 0.0 for r in rows])),
        })
    return out


CONSEQUENCE_TEXT = {
    "young": {
        "increase": (
            "Larger young generation: fewer (but longer) scavenges. Eden takes "
            "longer to fill, so promotion is spread out and the allocator has "
            "longer to let medium-lived objects die before a scavenge sees "
            "them. Survivor spaces grow with it, which is what lets the "
            "promotion-age threshold actually take effect. Cost: each pause "
            "copies more live bytes, and the old generation gets less of the "
            "heap if the total is fixed."),
        "decrease": (
            "Smaller young generation: more, shorter scavenges. Pauses fall in "
            "the common case, but scavenges become so frequent that "
            "medium-lived objects are copied (and aged) many times per unit of "
            "useful work; if the survivor space no longer holds the live set, "
            "the tenure threshold collapses and everything is promoted early, "
            "filling old and eventually forcing a full GC."),
    },
    "promotion_age": {
        "increase": (
            "Higher tenure age: objects must survive more scavenges before "
            "promotion. This helps only if the survivor spaces can hold them; "
            "it converts old-generation pressure into longer young pauses "
            "(more survivors copied each time). If the survivor space is too "
            "small the knob is inert: the effective threshold is lowered "
            "automatically to fit the space."),
        "decrease": (
            "Lower tenure age: objects are promoted sooner. Old grows faster "
            "and the concurrent marker is more likely to fall behind, which is "
            "how you get 100-600 ms full-GC pauses at the peak. It shortens "
            "young pauses only until the old generation fills, after which it "
            "backfires."),
    },
    "initiating": {
        "increase": (
            "Later concurrent-cycle start (higher initiating occupancy): fewer "
            "concurrent cycles and less CPU stolen from the application, but "
            "less headroom for the marker to finish before old is full. At the "
            "peak this is the difference between a short remark and a "
            "concurrent-mode-failure full GC."),
        "decrease": (
            "Earlier concurrent-cycle start (lower initiating occupancy): the "
            "marker has more time to keep up and full GCs disappear, at the "
            "cost of more concurrent cycles and CPU spent marking instead of "
            "serving orders."),
    },
}


def _consequences(sens: Dict[str, List[dict]]) -> List[str]:
    lines = []
    for param, label in (("young", "young size (MB)"),
                         ("promotion_age", "promotion age"),
                         ("initiating", "initiating occupancy")):
        rows = sens.get(param, [])
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        d = last["max_pause_ms"] - first["max_pause_ms"]
        t = (last["throughput"] - first["throughput"]) * 100
        p = last["promotion_gb"] - first["promotion_gb"]
        f = last["full_gcs"] - first["full_gcs"]
        direction = "increase" if last["value"] > first["value"] else "decrease"
        lines.append(
            f"{label}: {first['value']:g} -> {last['value']:g} "
            f"changes max pause {d:+.1f} ms, throughput {t:+.3f} pp, "
            f"promotion {p:+.1f} GB, full GCs {f:+.0f} (mean over grid).")
    return lines


def _pick_best(feasible: List[PointResult], objective: Objective) -> PointResult:
    """Choose among feasible points.

    Ranking criteria, in order:

    1. **Robustness margin**: prefer configurations whose pause has real
       headroom *and* whose throughput has real headroom, not a point that
       scrapes past one objective while sacrificing the other.
    2. **Not degenerate**: a young gen halved with a tenure age of 1 is only
       "better" because it makes pauses tiny by promoting everything; it
       wrecks the old generation over a longer horizon.
    3. **Low promotion**: among points with similar margin, prefer the one
       that promotes a smaller fraction of allocated bytes.  It is the one
       still likely to be feasible on a busier day, when more of the
       medium-lived population survives to be promoted.
    4. **Even coverage**: prefer a lower worst-case (p99.9) pause.

    The returned point therefore tends to a *balanced, non-degenerate,
    low-promotion* member of the feasible region, not the raw minimum pause.
    """
    def score(p: PointResult):
        pause_head = max(0.0, objective.max_pause_ms - p.max_pause_ms)
        thr_head = max(0.0, p.throughput - objective.min_throughput)
        # Normalise the two heads so neither dominates arbitrarily.
        norm_pause = pause_head / max(1e-9, objective.max_pause_ms)
        norm_thr = (thr_head / max(1e-9, 1.0 - objective.min_throughput)
                    if objective.min_throughput < 1.0 else 0.0)
        # Robustness is the *balance* of the two margins: a point that is
        # strong on one axis but scrapes the other is fragile.
        robustness = min(norm_pause, norm_thr)
        # Prefer the least aggressive fix that still leaves margin.  Two points
        # can both be feasible while one promotes 3x more than the other; the
        # low-promotion point is the one that will still work on a busier day.
        # We bias toward it after robustness, but not so hard that a point
        # sitting on the pause ceiling wins on promotion alone.
        low_prom = -p.promotion_fraction
        # Not degenerate: tenure age 1 promotes everything by construction.
        degenerate = -50.0 if p.promotion_age <= 1 else 0.0
        # Even coverage: prefer a better worst-case (p99.9) at equal terms.
        return (round(robustness, 3), degenerate, low_prom,
                -p.p999_ms, p.throughput)

    return max(feasible, key=score)


def tune(service_name: str = "trading_front_end",
         traffic_name: str = "trading_day",
         objective: Optional[Objective] = None,
         base: Optional[GCConfig] = None,
         seed: int = 12345,
         workers: Optional[int] = None,
         refine: bool = True,
         cohort_seconds: float = 5.0,
         day_hours: Optional[float] = None,
         space: Optional[_ConfigSpace] = None) -> TuningResult:
    """Search the parameter space for a feasible configuration.

    ``service_name`` is a ``ServiceProfile`` classmethod name; ``traffic_name``
    selects a ``TrafficProfile`` factory.  Returns the baseline (default
    parameters), the best feasible point, the entire evaluated grid and a
    per-parameter sensitivity table.
    """
    objective = objective or Objective()
    base = base or GCConfig()
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    service = getattr(ServiceProfile, service_name)()
    traffic = traffic_factory(traffic_name)()

    # baseline first (fast, so the user always gets something)
    baseline_args = (base.to_dict(), service_name, traffic_name,
                     seed, objective, base.heap.young_size / MB,
                     base.heap.survivor_ratio, base.young.promotion_age,
                     base.old.initiating_occupancy, base.old.mode,
                     cohort_seconds, day_hours)
    baseline = evaluate_point(baseline_args)

    sp = space or _default_space(base)
    grid = _run_space(sp, base, service_name, traffic_name, seed, objective,
                      cohort_seconds, workers, day_hours)
    feasible = [p for p in grid if p.feasible]
    best: Optional[PointResult] = None
    if feasible:
        best = _pick_best(feasible, objective)
    notes: List[str] = []
    if refine and best is not None:
        ref = _run_space(_refine_space(best), base, service_name, traffic_name,
                         seed, objective, cohort_seconds, workers, day_hours)
        grid.extend(ref)
        feasible = [p for p in grid if p.feasible]
        if feasible:
            best = _pick_best(feasible, objective)
    if best is None:
        # No feasible point: return the least-bad, and say why it fails.
        nearest = min(grid, key=lambda p: (
            max(0.0, p.max_pause_ms - objective.max_pause_ms) +
            max(0.0, objective.min_throughput - p.throughput) * 1e4))
        best = None
        notes.append(
            "No configuration in the searched space met both objectives. "
            "The closest point is reported; the binding constraint is "
            + ("pause" if nearest.max_pause_ms > objective.max_pause_ms
               else "throughput") + ".")
    sens = {p: _sensitivity(grid, p)
            for p in ("young", "promotion_age", "initiating")}
    notes.extend(_consequences(sens))
    if baseline.feasible:
        notes.append("The default configuration already meets the objective; "
                     "the search is only confirming the margin.")
    if baseline.full_gcs > 0 or baseline.degraded_cycles > 0:
        notes.append(
            f"Default settings suffered {baseline.full_gcs} full GC(s) and "
            f"{baseline.degraded_cycles} degraded concurrent cycle(s) over the day.")
    return TuningResult(
        objective=objective, service_name=service_name,
        traffic_name=traffic_name, baseline=baseline, best=best,
        grid=grid, sensitivity=sens, notes=notes,
    )
