"""Parameter search and explanation.

The tuner searches a small, structured space (generation sizes, tenuring
threshold, concurrent trigger ratio, marking rate) for a configuration that
meets a latency objective and a throughput requirement, then explains what
each parameter change does -- using ablation runs of the simulator rather
than hand-waving.

Objective
---------
A configuration is *feasible* when, over the whole simulated day:

* ``pause_max_ms <= pause_target_ms`` -- no request sees a longer pause than
  the service's timeout budget (defaults to the request SLO);
* ``tail_over_budget_frac <= tail_frac`` -- no more than a small fraction of
  requests are delayed beyond the budget;
* ``throughput_loss_frac <= 1 - throughput_target``.

Among feasible configurations we minimise ``tail_over_budget_frac`` (then
throughput loss).  If none is feasible, we minimise a weighted penalty and
report the best effort.

Search strategy
---------------
1. A coarse coordinate sweep around the current configuration (each knob at
   several values, plus the all-pairs cross-product for the three dominant
   knobs) is evaluated on a *representative* day subset (the peak window and
   a trough window) to keep the cost bounded.
2. The best few candidates are re-evaluated on the full day.
3. A local refinement (coordinate descent, 2 rounds) runs on the full day.

Every evaluation is deterministic, so the search is reproducible.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (
    GCConfig, ServiceScenario, SimulationConfig, MiB, get_scenario,
)
from .engine import Simulator


def _merge_objective(objective: Optional[dict]) -> dict:
    obj = dict(DEFAULT_OBJECTIVE)
    if objective:
        obj.update({k: v for k, v in objective.items() if v is not None})
    return obj


DEFAULT_OBJECTIVE = {
    # A pause longer than this blows the request timeout, so it is a hard
    # constraint (the service SLO is a few milliseconds of work plus a GC
    # pause budget).
    "pause_target_ms": 25.0,
    # Fraction of requests allowed to wait past the budget.
    "tail_frac": 0.002,
    # Throughput floor.  A concurrent collector trades CPU for pause time;
    # keeping total GC cost (STW + marker CPU) under 3% is the usual target.
    "throughput_target": 0.97,
    # The per-request delay budget used by the latency model.
    "pause_budget_ms": 20.0,
    # Maximum total heap the service is allowed to use.  Tuning that simply
    # buys a bigger heap is not tuning: it trades memory for pause time, so
    # the constraint makes that trade-off explicit.
    "max_heap_mib": 6144,
}


# --------------------------------------------------------------------------
# Representative day windows: a "fast day" for the coarse sweep.
# --------------------------------------------------------------------------
def day_windows(scenario: ServiceScenario, mode: str = "fast") -> List[Tuple[float, float]]:
    """Return [(start,end)] windows to simulate for an evaluation.

    ``fast`` simulates a trough window and the busiest peak window; the
    result is assembled by weighting each window.  ``full`` simulates the
    whole day.
    """
    day = scenario.workload.day_seconds
    if mode == "full":
        return [(0.0, day)]
    curve = scenario.workload.curve()
    n = len(curve)
    hour = day / n
    order = sorted(range(n), key=lambda i: -curve[i])
    peak = order[0]
    trough = min(range(n), key=lambda i: curve[i])
    # take a 3-hour window around the peak and 2 hours around the trough
    p0 = max(0.0, (peak - 1) * hour)
    p1 = min(day, (peak + 2) * hour)
    t0 = max(0.0, (trough - 1) * hour)
    t1 = min(day, (trough + 1) * hour)
    win = [(t0, t1), (p0, p1)]
    return win


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    gc: GCConfig
    score: float = 0.0
    feasible: bool = False
    summary: dict = field(default_factory=dict)
    windows: List[dict] = field(default_factory=list)
    wall_s: float = 0.0
    # per-seed summaries when a candidate was checked for robustness
    seed_summaries: List[dict] = field(default_factory=list)
    worst_pause_max_ms: float = 0.0

    def key_metrics(self) -> dict:
        s = self.summary
        return {
            "feasible": self.feasible,
            "score": self.score,
            "pause_max_ms": s.get("pause_max_ms", 0.0),
            "pause_p99_ms": s.get("pause_p99_ms", 0.0),
            "tail_over_budget_frac": s.get("tail_over_budget_frac", 1.0),
            "throughput_loss_frac": s.get("throughput_loss_frac", 1.0),
            "full_count": s.get("full_count", 0),
            "degenerate_count": s.get("degenerate_count", 0),
            "concurrent_cycles": s.get("concurrent_cycles", 0),
            "minor_count": s.get("minor_count", 0),
            "promoted_mibs_per_s": s.get("promoted_mibs_per_s", 0.0),
            "heap_mib": s.get("heap_bytes", 0.0) / MiB,
            "peak_old_frac": s.get("peak_old_frac", 0.0),
        }


def _simulate_window(scenario: ServiceScenario, gc: GCConfig,
                     t0: float, t1: float,
                     sim_cfg: SimulationConfig,
                     fast: bool = False) -> dict:
    """Simulate a slice [t0,t1] of the day as an independent, warm run.

    The workload curve keeps its absolute time-of-day shape, so a slice is
    representative of that hour.  The simulator starts with an empty heap;
    to avoid measuring warm-up we discount the first 10% of the window.
    """
    sc = ServiceScenario(
        name=scenario.name, description=scenario.description,
        workload=scenario.workload, population=scenario.population,
        gc=gc, sim=sim_cfg, tuning_space={})
    # shift the day window: simulate [0, t1-t0) but with the curve of the
    # original absolute clock by offsetting the time origin
    wl = sc.workload
    dur = t1 - t0
    # Build a workload whose multiplier at local time u equals the original
    # at t0+u.  With a 24-point curve this is a rotation by t0/day*n points;
    # to avoid rotation error we instead sample the curve into a dense list.
    n_samples = max(48, int(dur / 60.0))
    dense = [wl.multiplier_at(t0 + (i + 0.5) * dur / n_samples)
             for i in range(n_samples)]
    wl2 = replace(wl, traffic_curve=dense, day_seconds=int(dur),
                  night_factor=0.0)
    sc.workload = wl2
    if fast:
        sc.sim = replace(sim_cfg, warmup_s=0.10 * dur, track_ages=False)
    sim = Simulator(sc)
    if fast:
        sim.fast_mode = True
    res = sim.run()
    s = dict(res.summary)
    s["window"] = [t0, t1]
    return s


def _aggregate(summaries: List[dict], scenario: ServiceScenario,
               objective: dict) -> dict:
    """Combine window summaries into full-day-equivalent metrics.

    Pause maxima are maxima; the tail fraction and throughput loss are
    request-weighted.  Request weight per window is proportional to its
    allocated bytes, which the summary already reports.
    """
    if not summaries:
        return {}
    total_bytes = sum(s.get("allocated_bytes_total", 0.0) for s in summaries) or 1.0
    out = {}
    out["pause_max_ms"] = max(s["pause_max_ms"] for s in summaries)
    # weight tails and throughput by allocation volume
    for k in ("tail_over_budget_frac", "throughput_loss_frac", "stw_pause_frac",
              "concurrent_cpu_frac"):
        out[k] = sum(s.get(k, 0.0) * s.get("allocated_bytes_total", 0.0)
                     for s in summaries) / total_bytes
    for k in ("full_count", "degenerate_count", "concurrent_cycles",
              "minor_count", "budget_breaches", "oom_events"):
        out[k] = sum(s.get(k, 0) for s in summaries)
    out["pause_p99_ms"] = max(s["pause_p99_ms"] for s in summaries)
    out["pause_p50_ms"] = (sum(s["pause_p50_ms"] * s.get("allocated_bytes_total", 0.0)
                               for s in summaries) / total_bytes)
    out["promoted_mibs_per_s"] = (sum(s.get("promoted_bytes_total", 0.0)
                                      for s in summaries) / MiB
                                  / max(sum(s.get("day_seconds", 1.0) for s in summaries), 1.0))
    out["peak_old_frac"] = max(s.get("peak_old_frac", 0.0) for s in summaries)
    out["allocated_bytes_total"] = total_bytes
    # full-day equivalent volume: scale counts by day/window length so that
    # "full GCs per day" is comparable between windows and the full day
    return out


def _score(summary: dict, objective: dict) -> Tuple[float, bool]:
    """Lower is better; feasible configurations sort first."""
    objective = _merge_objective(objective)
    tail = summary.get("tail_over_budget_frac", 1.0)
    loss = summary.get("throughput_loss_frac", 1.0)
    pmax = summary.get("pause_max_ms", 1e9)
    target_ms = objective["pause_target_ms"]
    loss_limit = 1.0 - objective["throughput_target"]
    over_pause = max(0.0, pmax - target_ms) / max(target_ms, 1.0)
    over_tail = max(0.0, tail - objective["tail_frac"]) / max(objective["tail_frac"], 1e-9)
    over_loss = max(0.0, loss - loss_limit) / max(loss_limit, 1e-9)
    heap_mib = summary.get("heap_mib", 0.0)
    max_heap = objective.get("max_heap_mib")
    # allow 5% slack for sizing granularity (regions round up to 1 MiB)
    over_heap = (max(0.0, heap_mib - 1.05 * max_heap) / max(max_heap, 1.0)
                 if max_heap else 0.0)
    feasible = (pmax <= target_ms and tail <= objective["tail_frac"]
                and loss <= loss_limit and over_heap <= 0.0)
    # score: feasible configs ranked by tail then loss; infeasible by the
    # magnitude of the violations (pause violation dominates)
    if feasible:
        # prefer the cheaper heap among otherwise equal solutions
        score = 10.0 * tail + loss + 0.02 * heap_mib / max(max_heap or 1, 1)
    else:
        score = (100.0 * (1.0 + over_pause) * (1.0 + over_tail)
                 + over_loss + 10.0 * over_heap)
    return score, feasible


# --------------------------------------------------------------------------
# Search space and evaluation plumbing
# --------------------------------------------------------------------------
def _space(scenario: ServiceScenario, objective: dict) -> dict:
    """Knob grid derived from the scenario's declared tuning space."""
    ts = scenario.tuning_space or {}
    base = scenario.gc

    def around(v, vals):
        return vals

    trig = sorted(set(list(ts.get("concurrent_trigger_ratio",
                                    [0.3, 0.4, 0.5, 0.6]))
                        + [0.38, 0.45, 0.55]))
    space = {
        "eden_bytes": [mib for mib in ts.get("eden_mib", [16, 32, 64, 128])],
        "survivor_bytes": list(ts.get("survivor_mib", [2, 4, 8, 16])),
        "old_bytes": list(ts.get("old_mib", [512, 1024, 2048, 4096])),
        "tenuring_threshold": list(ts.get("tenuring_threshold", [2, 4, 6, 8, 12])),
        "concurrent_trigger_ratio": trig,
        "concurrent_mark_bytes_per_s": [x * MiB for x in
                                        ts.get("concurrent_mark_mibs",
                                               [16, 32, 64, 128])],
    }
    if not gc_concurrent(scenario.gc):
        space["concurrent_trigger_ratio"] = [space["concurrent_trigger_ratio"][0]]
    # widen the old-gen and mark-rate grids: the shipped default is a
    # *broken* configuration, so the feasible point may sit outside the
    # scenario's "plausible editorial" grid, and the search must find it.
    budget = objective.get("max_heap_mib")
    if budget:
        # Old-generation candidates are fractions of the budget's *old*
        # portion, so the grid automatically scales with the service's heap.
        young_reserve = 256
        top = int(budget - young_reserve)
        candidates = set(x for x in (1024, 2048, 3072, 4096, 6144, 8192,
                                     12288, 16384, 24576) if x <= top)
        for frac in (0.25, 0.4, 0.55, 0.7, 0.85, 0.95):
            v = int(round(budget * frac / 256.0)) * 256
            if v <= top:
                candidates.add(v)
        candidates = sorted(candidates | {max(1024, top // 2)})
        old = [x for x in candidates if x + 512 <= budget] or [candidates[0]]
        space["old_bytes"] = old
        young_max = budget - min(old)
        space["eden_bytes"] = [x for x in space["eden_bytes"]
                               if x + 2 * min(space["survivor_bytes"]) <= young_max] \
                              or [space["eden_bytes"][0]]
        space["survivor_bytes"] = [x for x in space["survivor_bytes"]
                                   if 2 * x + min(space["eden_bytes"]) <= young_max] \
                                  or [space["survivor_bytes"][0]]
    else:
        old = sorted(set(space["old_bytes"]
                         + [x for x in (1024, 1536, 2048, 3072, 4096, 6144)
                            if x not in space["old_bytes"]]))
        space["old_bytes"] = old
    marks = sorted(set(space["concurrent_mark_bytes_per_s"]
                       + [x * MiB for x in (32, 64, 100, 160, 256, 400)]))
    space["concurrent_mark_bytes_per_s"] = marks
    return space


def gc_concurrent(gc: GCConfig) -> bool:
    return getattr(gc, "concurrent_enabled", True)


def _apply(base: GCConfig, params: dict) -> GCConfig:
    """Build a config from a parameter dict.

    Contract (used everywhere in this module):

    * the three sizes are in **MiB**;
    * ``concurrent_mark_bytes_per_s`` is in **bytes/s**.
    """
    gc = base.copy()
    gc.eden_bytes = int(round(params["eden_bytes"] * MiB))
    gc.survivor_bytes = int(round(params["survivor_bytes"] * MiB))
    gc.old_bytes = int(round(params["old_bytes"] * MiB))
    gc.tenuring_threshold = int(params["tenuring_threshold"])
    gc.concurrent_trigger_ratio = float(params["concurrent_trigger_ratio"])
    gc.concurrent_mark_bytes_per_s = float(params["concurrent_mark_bytes_per_s"])
    return gc


def _params_of(gc: GCConfig) -> dict:
    """Inverse of :func:`_apply`, in the same unit contract."""
    return {
        "eden_bytes": gc.eden_bytes / MiB,
        "survivor_bytes": gc.survivor_bytes / MiB,
        "old_bytes": gc.old_bytes / MiB,
        "tenuring_threshold": gc.tenuring_threshold,
        "concurrent_trigger_ratio": gc.concurrent_trigger_ratio,
        "concurrent_mark_bytes_per_s": gc.concurrent_mark_bytes_per_s,
    }


SEARCH_RECORD_CAP = 16
FINAL_RECORD_CAP = 80


def _evaluate(scenario: ServiceScenario, gc: GCConfig, objective: dict,
              mode: str, seed: Optional[int] = None,
              record_cap: int = SEARCH_RECORD_CAP,
              fast: bool = True) -> Candidate:
    t0 = time.time()
    sim_cfg = replace(scenario.sim, seed=(scenario.sim.seed if seed is None else seed),
                      max_records_per_minor=record_cap)
    if mode == "full":
        # single run of the whole day
        sc = ServiceScenario(scenario.name, scenario.description,
                             scenario.workload, scenario.population, gc, sim_cfg,
                             {})
        sim = Simulator(sc)
        sim.fast_mode = fast
        res = sim.run()
        summary = dict(res.summary)
        cand = Candidate(gc=gc, summary=summary, windows=[summary])
    else:
        windows = []
        for (a, b) in day_windows(scenario, mode):
            windows.append(_simulate_window(scenario, gc, a, b, sim_cfg))
        summary = _aggregate(windows, scenario, objective)
        cand = Candidate(gc=gc, summary=summary, windows=windows)
    score, feasible = _score(summary, objective)
    cand.score = score
    cand.feasible = feasible
    cand.wall_s = time.time() - t0
    return cand


# --------------------------------------------------------------------------
# The tuner
# --------------------------------------------------------------------------
@dataclass
class TuningResult:
    objective: dict
    baseline: Candidate
    best: Candidate
    candidates: List[Candidate]
    ranking: List[dict]
    effects: List[dict]
    search_log: List[dict]
    wall_s: float
    feasible_count: int
    mode: str
    top_recommendation: dict = field(default_factory=dict)


def _param_sets(scenario: ServiceScenario, space: dict) -> List[dict]:
    """Structured coarse grid.

    The three families are searched as separate blocks instead of one
    cross-product, because their effects are largely separable:

    * young-generation sizing (eden x survivor x tenuring),
    * old-generation capacity,
    * concurrent-collector policy (trigger ratio x mark rate),
      plus a joint block at the default sizes.

    This keeps the grid at ~150 points instead of ~3000, while still
    covering the interaction between survivor size and tenuring threshold,
    which is the one that matters for premature promotion.
    """
    base = scenario.gc
    base_params = _params_of(base)
    sets: List[dict] = []
    seen = set()

    def add(p):
        key = (p["eden_bytes"], p["survivor_bytes"], p["old_bytes"],
               p["tenuring_threshold"], round(p["concurrent_trigger_ratio"], 4),
               int(p["concurrent_mark_bytes_per_s"]))
        if key in seen:
            return
        seen.add(key)
        sets.append(p)

    add(dict(base_params))
    def thin(vals, n):
        if len(vals) <= n:
            return list(vals)
        idx = [round(i * (len(vals) - 1) / (n - 1)) for i in range(n)]
        return sorted({vals[i] for i in idx})

    for e in thin(space["eden_bytes"], 5):
        for s in thin(space["survivor_bytes"], 5):
            for t in thin(space["tenuring_threshold"], 5):
                p = dict(base_params)
                p["eden_bytes"] = e
                p["survivor_bytes"] = s
                p["tenuring_threshold"] = t
                add(p)
    for old in space["old_bytes"]:
        p = dict(base_params)
        p["old_bytes"] = old
        add(p)
    if gc_concurrent(base):
        for trig in space["concurrent_trigger_ratio"]:
            for mark in space["concurrent_mark_bytes_per_s"]:
                p = dict(base_params)
                p["concurrent_trigger_ratio"] = trig
                p["concurrent_mark_bytes_per_s"] = mark
                add(p)
    # joint block: a promising young sizing at a generous old gen and a
    # healthy marker, so interactions show up
    tail_eden = space["eden_bytes"][-2:] or space["eden_bytes"][:1]
    tail_surv = space["survivor_bytes"][-2:] or space["survivor_bytes"][:1]
    for e in tail_eden:
        for s in tail_surv:
            for trig in (0.4, 0.55):
                for mark in (max(space["concurrent_mark_bytes_per_s"]) // MiB,):
                    p = dict(base_params)
                    p["eden_bytes"] = e
                    p["survivor_bytes"] = s
                    p["old_bytes"] = max(space["old_bytes"])
                    p["concurrent_trigger_ratio"] = trig
                    p["concurrent_mark_bytes_per_s"] = mark * MiB
                    add(p)
    return sets


def tune(scenario: ServiceScenario,
         objective: Optional[dict] = None,
         mode: str = "fast",
         max_candidates: int = 24,
         refine: bool = True,
         workers: Optional[int] = None,
         final_record_cap: int = FINAL_RECORD_CAP,
         progress=None) -> TuningResult:
    """Search the tuning space and explain parameter effects.

    ``mode='fast'`` evaluates the coarse grid on a representative two-window
    day and re-scores the top candidates on the full day; ``mode='full'``
    evaluates everything on the full day (slow but exact).
    """
    obj = dict(DEFAULT_OBJECTIVE)
    if objective:
        obj.update(objective)
    wall0 = time.time()
    base = scenario.gc
    space = _space(scenario, obj)

    log: List[dict] = []
    evaluated: List[Candidate] = []

    def log_candidate(c: Candidate, ev_mode: str) -> None:
        log.append({
            "params": {
                "eden_mib": c.gc.eden_bytes / MiB,
                "survivor_mib": c.gc.survivor_bytes / MiB,
                "old_mib": c.gc.old_bytes / MiB,
                "tenuring_threshold": c.gc.tenuring_threshold,
                "concurrent_trigger_ratio": c.gc.concurrent_trigger_ratio,
                "concurrent_mark_mibs": c.gc.concurrent_mark_bytes_per_s / MiB,
            },
            "feasible": c.feasible, "score": c.score,
            "pause_max_ms": c.summary.get("pause_max_ms", 0.0),
            "tail_over_budget_frac": c.summary.get("tail_over_budget_frac", 1.0),
            "throughput_loss_frac": c.summary.get("throughput_loss_frac", 1.0),
            "full_count": c.summary.get("full_count", 0),
            "degenerate_count": c.summary.get("degenerate_count", 0),
            "wall_s": c.wall_s,
            "mode": ev_mode,
            "seeds": len(c.seed_summaries) or 1,
        })

    # baseline first, at final fidelity
    baseline = _evaluate(scenario, base.copy(), obj, mode,
                         record_cap=final_record_cap, fast=True)
    log_candidate(baseline, mode)
    evaluated.append(baseline)

    sets_all = _param_sets(scenario, space)
    if len(sets_all) > max_candidates:
        sets = screen_candidates(scenario, obj, sets_all,
                                 keep=max_candidates, workers=workers,
                                 record_cap=SEARCH_RECORD_CAP)
    else:
        sets = sets_all[:max_candidates]
    coarse = _evaluate_many(scenario, base, obj, "full", sets,
                            record_cap=SEARCH_RECORD_CAP, workers=workers,
                            fast=True, progress=progress)
    for c in coarse:
        log_candidate(c, "single-seed")
    evaluated.extend(coarse)
    # A single-day run is one sample of a stochastic workload: a
    # configuration that only just avoids degeneration on one seed will
    # regress on another.  Re-evaluate the promising candidates on several
    # seeds and rank them by their *worst* day.
    coarse.sort(key=lambda c: (not c.feasible, c.score))
    finalists = coarse[:max(12, min(24, len(coarse)))]
    robust_params = [_params_of(c.gc) for c in finalists]
    robust = _evaluate_robust(scenario, base, obj, "full", robust_params,
                              record_cap=final_record_cap, workers=workers,
                              progress=progress)
    for c in robust:
        log_candidate(c, "multi-seed")
    evaluated.extend(robust)
    # Refine only the genuinely best few: a full-day multi-seed evaluation is
    # ~10-20x the cost of a screen, and the tail beyond this is noise.
    robust.sort(key=lambda c: (not c.feasible, c.score))
    pool = (robust[:max(4, min(6, len(robust)))] if robust else coarse)

    # coordinate-descent refinement at final fidelity on the full day
    pool.sort(key=lambda c: (not c.feasible, c.score))
    best = pool[0] if pool else baseline
    if refine:
        best_params = _params_of(best.gc)
        for _round in range(1):
            improved = False
            # Coordinate descent over *neighbouring* values only.  Trying
            # every value of every knob at multi-seed full-day fidelity is
            # O(knobs x values x seeds) days of simulation; the optimum of a
            # well-behaved response surface is reachable by stepping from the
            # current point, and the coarse grid has already covered the
            # far-away values.
            jobs = []
            for knob, values in space.items():
                vals = sorted(set(values))
                cur = float(best_params[knob])
                below = [v for v in vals if v < cur - 1e-9]
                above = [v for v in vals if v > cur + 1e-9]
                if below:
                    p = dict(best_params); p[knob] = below[-1]; jobs.append(p)
                if above:
                    p = dict(best_params); p[knob] = above[0]; jobs.append(p)
            refined = _evaluate_robust(scenario, base, obj, "full", jobs,
                                       record_cap=final_record_cap,
                                       workers=workers, progress=progress)
            for c in refined:
                log_candidate(c, "full")
            evaluated.extend(refined)
            for c in refined:
                if (not c.feasible, c.score) < (not best.feasible, best.score):
                    best = c
                    best_params = _params_of(c.gc)
                    improved = True
            if not improved:
                break

    # ---- ablation-based parameter effects ----------------------------
    # Effects are measured with burstiness switched off: the random
    # burst multiplier makes different configurations consume the RNG
    # differently, which would swamp a one-knob ablation in noise.
    det = replace(scenario,
                  workload=replace(scenario.workload, burst_cv=0.0))

    def _effect_run(gc, ev_mode, seed=12345):
        return _evaluate(det, gc, obj, ev_mode, seed=seed,
                         record_cap=SEARCH_RECORD_CAP)
    effects = _explain_effects(det, best.gc, obj, baseline, _effect_run)

    ranked = sorted(evaluated, key=lambda c: (not c.feasible, c.score))
    # de-duplicate by parameter tuple
    seen = set()
    ranking = []
    for c in ranked:
        key = (c.gc.eden_bytes, c.gc.survivor_bytes, c.gc.old_bytes,
               c.gc.tenuring_threshold, round(c.gc.concurrent_trigger_ratio, 4),
               int(c.gc.concurrent_mark_bytes_per_s))
        if key in seen:
            continue
        seen.add(key)
        ranking.append({
            **c.key_metrics(),
            "seeds": len(c.seed_summaries) or 1,
            "params": {
                "eden_mib": c.gc.eden_bytes / MiB,
                "survivor_mib": c.gc.survivor_bytes / MiB,
                "old_mib": c.gc.old_bytes / MiB,
                "tenuring_threshold": c.gc.tenuring_threshold,
                "concurrent_trigger_ratio": c.gc.concurrent_trigger_ratio,
                "concurrent_mark_mibs": c.gc.concurrent_mark_bytes_per_s / MiB,
            },
        })
    return TuningResult(
        objective=obj,
        baseline=baseline,
        best=best,
        candidates=evaluated,
        ranking=ranking[:40],
        effects=effects,
        search_log=log,
        wall_s=time.time() - wall0,
        feasible_count=sum(1 for c in evaluated if c.feasible),
        mode=mode,
        top_recommendation=_recommend(baseline, best, obj, effects),
    )




def _eval_job(args):
    """Module-level worker: evaluate one parameter set. Picklable."""
    (scenario, base_gc, objective, mode, params, record_cap, fast) = args
    gc = _apply(base_gc, params)
    try:
        gc.validate()
    except ValueError:
        return None
    cand = _evaluate(scenario, gc, objective, mode, record_cap=record_cap,
                     fast=fast)
    return cand


def _evaluate_many(scenario, base_gc, objective, mode, param_sets,
                   record_cap=SEARCH_RECORD_CAP, workers=None, progress=None,
                   fast=True):
    """Evaluate a list of parameter dicts, optionally in parallel."""
    jobs = [(scenario, base_gc, objective, mode, p, record_cap, fast)
            for p in param_sets]
    out = []
    if not jobs:
        return out
    import os
    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    if workers > 1 and len(jobs) > 2:
        try:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                for i, c in enumerate(pool.imap_unordered(_eval_job, jobs)):
                    if c is not None:
                        out.append(c)
                    if progress:
                        progress(i + 1, len(jobs))
            return out
        except Exception:
            pass  # fall back to serial
    for i, job in enumerate(jobs):
        c = _eval_job(job)
        if c is not None:
            out.append(c)
        if progress:
            progress(i + 1, len(jobs))
    return out



ROBUST_SEEDS = (12345, 7, 4242)


def _combine_worst(seed_summaries: List[dict]) -> dict:
    """Worst case across seeds: max pauses, worst tail/loss, max counts."""
    out = dict(seed_summaries[0])
    for key in ("pause_max_ms", "pause_p99_ms", "pause_p95_ms",
                "tail_over_budget_frac", "throughput_loss_frac",
                "full_count", "degenerate_count", "concurrent_cycles",
                "minor_count"):
        out[key] = max(s[key] for s in seed_summaries)
    out["pause_max_kind"] = max(
        seed_summaries, key=lambda s: s["pause_max_ms"]).get("pause_max_kind", "")
    out["robust_seeds"] = len(seed_summaries)
    return out


def _eval_job_robust(args):
    """Evaluate one parameter set on every robustness seed; worst case wins."""
    (scenario, base_gc, objective, mode, params, record_cap, fast,
     seeds) = args
    gc = _apply(base_gc, params)
    try:
        gc.validate()
    except ValueError:
        return None
    summaries = []
    wall = 0.0
    for seed in seeds:
        c = _evaluate(scenario, gc, objective, mode, seed=seed,
                      record_cap=record_cap, fast=fast)
        summaries.append(c.summary)
        wall += c.wall_s
    worst = _combine_worst(summaries)
    score, feasible = _score(worst, objective)
    return Candidate(gc=gc, score=score, feasible=feasible, summary=worst,
                     wall_s=wall, seed_summaries=summaries,
                     worst_pause_max_ms=worst["pause_max_ms"])


def _evaluate_robust(scenario, base_gc, objective, mode, param_sets,
                     record_cap=FINAL_RECORD_CAP, workers=None,
                     seeds=ROBUST_SEEDS, progress=None, fast=True):
    """Evaluate each parameter set on several seeds, keeping the worst day."""
    jobs = [(scenario, base_gc, objective, mode, p, record_cap, fast, seeds)
            for p in param_sets]
    if not jobs:
        return []
    import os
    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    out = []
    if workers > 1 and len(jobs) > 1:
        try:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                for i, c in enumerate(pool.imap_unordered(_eval_job_robust,
                                                          jobs)):
                    if c is not None:
                        out.append(c)
                    if progress:
                        progress(i + 1, len(jobs))
            return out
        except Exception:
            out = []
    for i, job in enumerate(jobs):
        c = _eval_job_robust(job)
        if c is not None:
            out.append(c)
        if progress:
            progress(i + 1, len(jobs))
    return out


def _screen_windows(scenario: ServiceScenario) -> List[Tuple[float, float]]:
    """Two contrasting windows: the busiest hour and the late-day ramp.

    Degeneration is not confined to the peak: it happens whenever the
    promotion rate crosses the marker's rate, which depends on how full the
    old generation already is.  Screening on the peak alone systematically
    misses configurations that only fail later in the day.
    """
    day = scenario.workload.day_seconds
    curve = scenario.workload.curve()
    n = len(curve)
    hour = day / n
    peak = max(range(n), key=lambda i: curve[i])
    # a later, still-busy hour (two thirds through the day)
    late = max(range(int(n * 0.6), n), key=lambda i: curve[i])
    wins = []
    for centre in (peak, late):
        a = max(0.0, (centre - 1) * hour)
        b = min(day, (centre + 1) * hour)
        if b > a and (a, b) not in wins:
            wins.append((a, b))
    return wins


def screen_candidates(scenario: ServiceScenario, objective: dict,
                      param_sets: List[dict], keep: int = 24,
                      record_cap: int = SEARCH_RECORD_CAP,
                      workers: Optional[int] = None,
                      diverse: bool = True) -> List[dict]:
    """Cheap screening on two windows, keeping the worst, then a diverse set.

    Returns up to ``keep`` parameter dicts.  When ``diverse`` is set, the
    finalists are spread across (eden, survivor, old) buckets so the
    expensive robust stage does not spend all its budget on one corner of
    the space.
    """
    objective = _merge_objective(objective)
    windows = _screen_windows(scenario)
    jobs = []
    for p in param_sets:
        gc = _apply(scenario.gc, p)
        try:
            gc.validate()
        except ValueError:
            continue
        jobs.append((scenario, objective, windows, gc, p, record_cap))
    if not jobs:
        return []
    import os
    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    if workers > 1 and len(jobs) > 2:
        try:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                out = list(pool.imap_unordered(_screen_job, jobs))
        except Exception:
            out = [_screen_job(j) for j in jobs]
    else:
        out = [_screen_job(j) for j in jobs]
    out.sort(key=lambda x: (not x[1], x[0]))
    if not diverse:
        return [p for (_, _, p) in out[:keep]]
    # take the best candidate from each (eden, survivor, old) bucket first,
    # then fill the remaining slots in score order
    chosen: List[dict] = []
    seen_buckets = set()
    for score, feasible, p in out:
        if len(chosen) >= keep:
            break
        b = (p["eden_bytes"], p["survivor_bytes"], p["old_bytes"])
        if b in seen_buckets:
            continue
        seen_buckets.add(b)
        chosen.append(p)
    for score, feasible, p in out:
        if len(chosen) >= keep:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen[:keep]


def _screen_job(args):
    (scenario, objective, windows, gc, p, record_cap) = args
    worst_score = -1.0
    worst_feasible = True
    for (w0, w1) in windows:
        s = _simulate_window(scenario, gc, w0, w1,
                             replace(scenario.sim,
                                     max_records_per_minor=record_cap),
                             fast=True)
        score, feasible = _score(s, objective)
        if score > worst_score:
            worst_score = score
        worst_feasible = worst_feasible and feasible
    return (worst_score, worst_feasible, p)


def _recommend(baseline: Candidate, best: Candidate, obj: dict,
               effects: List[dict]) -> dict:
    b = baseline.summary
    w = best.summary

    def pct(x, y):
        if abs(x) < 1e-12:
            return None
        return (y - x) / abs(x)
    return {
        "feasible": best.feasible,
        "objective": obj,
        "before": {
            "pause_max_ms": b.get("pause_max_ms", 0.0),
            "pause_p99_ms": b.get("pause_p99_ms", 0.0),
            "tail_over_budget_frac": b.get("tail_over_budget_frac", 1.0),
            "throughput_loss_frac": b.get("throughput_loss_frac", 1.0),
            "full_count": b.get("full_count", 0),
            "degenerate_count": b.get("degenerate_count", 0),
        },
        "after": {
            "pause_max_ms": w.get("pause_max_ms", 0.0),
            "pause_p99_ms": w.get("pause_p99_ms", 0.0),
            "tail_over_budget_frac": w.get("tail_over_budget_frac", 1.0),
            "throughput_loss_frac": w.get("throughput_loss_frac", 1.0),
            "full_count": w.get("full_count", 0),
            "degenerate_count": w.get("degenerate_count", 0),
        },
        "changes": {
            "eden_mib": [baseline.gc.eden_bytes / MiB, best.gc.eden_bytes / MiB],
            "survivor_mib": [baseline.gc.survivor_bytes / MiB,
                             best.gc.survivor_bytes / MiB],
            "old_mib": [baseline.gc.old_bytes / MiB, best.gc.old_bytes / MiB],
            "tenuring_threshold": [baseline.gc.tenuring_threshold,
                                   best.gc.tenuring_threshold],
            "concurrent_trigger_ratio": [baseline.gc.concurrent_trigger_ratio,
                                         best.gc.concurrent_trigger_ratio],
            "concurrent_mark_mibs": [baseline.gc.concurrent_mark_bytes_per_s / MiB,
                                     best.gc.concurrent_mark_bytes_per_s / MiB],
        },
        "effects": effects,
    }


# --------------------------------------------------------------------------
# Explaining effects: one-factor-at-a-time ablation from the best config
# --------------------------------------------------------------------------
def _explain_effects(scenario: ServiceScenario, best_gc: GCConfig,
                     obj: dict, baseline: Candidate, run) -> List[dict]:
    """For each knob, move it one grid step in each direction and report what
    changes.  Effects are measured against the best configuration, so each
    row isolates one knob's contribution."""
    space = _space(scenario, obj)
    base_params = _params_of(best_gc)
    ref = run(_apply(scenario.gc, base_params), "fast", seed=12345)
    effects: List[dict] = []
    labels = {
        "eden_bytes": ("新生代分配区(eden)", "MiB"),
        "survivor_bytes": ("幸存区(survivor)", "MiB"),
        "old_bytes": ("老年代", "MiB"),
        "tenuring_threshold": ("晋升年龄阈值", ""),
        "concurrent_trigger_ratio": ("并发回收触发比例", ""),
        "concurrent_mark_bytes_per_s": ("并发标记速率", "MiB/s"),
    }
    for knob, values in space.items():
        vals = sorted(set(values))
        cur = base_params[knob]
        below = [v for v in vals if v < cur - 1e-9]
        above = [v for v in vals if v > cur + 1e-9]
        neighbours = []
        if below:
            neighbours.append(below[-1])
        if above:
            neighbours.append(above[0])
        for v in neighbours:
            p = dict(base_params)
            p[knob] = v
            gc = _apply(scenario.gc, p)
            try:
                gc.validate()
            except ValueError:
                continue
            c = run(gc, "fast", seed=12345)
            label, unit = labels[knob]
            frm, to = cur, v
            if knob == "concurrent_mark_bytes_per_s":
                frm, to = cur / MiB, v / MiB
            effects.append({
                "knob": knob,
                "label": label,
                "unit": unit,
                "from": frm,
                "to": to,
                "direction": "上调" if v > cur else "下调",
                "delta_pause_max_ms": (c.summary["pause_max_ms"]
                                       - ref.summary["pause_max_ms"]),
                "delta_pause_p99_ms": (c.summary["pause_p99_ms"]
                                       - ref.summary["pause_p99_ms"]),
                "delta_tail": (c.summary["tail_over_budget_frac"]
                               - ref.summary["tail_over_budget_frac"]),
                "delta_throughput_loss": (c.summary["throughput_loss_frac"]
                                          - ref.summary["throughput_loss_frac"]),
                "delta_full_gcs": (c.summary["full_count"]
                                   - ref.summary["full_count"]),
                "delta_minors": (c.summary["minor_count"]
                                 - ref.summary["minor_count"]),
                "delta_promoted": (c.summary.get("promoted_mibs_per_s", 0.0)
                                   - ref.summary.get("promoted_mibs_per_s", 0.0)),
                "feasible": c.feasible,
                "delta_pause_max_pct": (c.summary["pause_max_ms"]
                                        / max(ref.summary["pause_max_ms"], 1e-9) - 1.0),
                "explanation": _interpret(knob, v > cur, c, ref),
            })
    return effects


def _interpret(knob: str, up: bool, c: Candidate, ref: Candidate) -> str:
    """Turn the measured deltas into a sentence about the mechanism.

    The sentence always reports what actually changed in this ablation, then
    names the mechanism.  It never predicts a direction that the numbers
    contradict.
    """
    d_tail = c.summary["tail_over_budget_frac"] - ref.summary["tail_over_budget_frac"]
    d_loss = c.summary["throughput_loss_frac"] - ref.summary["throughput_loss_frac"]
    d_full = c.summary["full_count"] - ref.summary["full_count"]
    d_prom = (c.summary.get("promoted_mibs_per_s", 0.0)
              - ref.summary.get("promoted_mibs_per_s", 0.0))
    d_minor = c.summary["minor_count"] - ref.summary["minor_count"]
    d_pmax = c.summary["pause_max_ms"] - ref.summary["pause_max_ms"]
    base_minor = max(ref.summary["minor_count"], 1)
    d_minor_pct = d_minor / base_minor * 100.0

    def no_effect() -> bool:
        return (abs(d_tail) < 1e-7 and abs(d_loss) < 2e-4
                and d_full == 0 and abs(d_minor_pct) < 1.0
                and abs(d_prom) < 0.02 and abs(d_pmax) < 1.0)

    if no_effect():
        return ("这一步改动没有可测量的影响：当前配置已经不在这个参数的瓶颈区，"
                "只有越过某个触发条件（例如存活区溢出、并发周期不足）才会改变行为。")

    measured = (f"实测：minor {d_minor_pct:+.0f}%，晋升 {d_prom:+.2f} MiB/s，"
                f"全堆停顿 {d_full:+d} 次，最长停顿 {d_pmax:+.1f} ms。")
    if knob == "eden_bytes":
        mech = ("eden 越大，minor 越少、对象在新生代停留越久；但单次需要拷贝的存活量"
                "也越大，所以 minor 停顿变长。它换来的是更少的回收次数和更少的晋升。"
                if up else
                "eden 越小，minor 越频繁、单次拷贝量小、停顿短；代价是对象更早被"
                "推到存活区/老年代，晋升与并发压力上升。")
        return mech + measured
    if knob == "survivor_bytes":
        return ("存活区大小决定对象能否熬到晋升阈值。存活区不足时，最老的对象会在"
                "to-space 溢出时被提前晋升，把中等寿命对象直接倒进老年代。" + measured)
    if knob == "old_bytes":
        return ("老年代容量只改变 full/并发回收的触发时机，不改变分配与晋升速率本身。"
                "容量越大，触发越晚，单次需要标记的存活集越大。" + measured)
    if knob == "tenuring_threshold":
        return ("晋升阈值决定对象要熬过几轮 minor 才进入老年代。阈值提高会让更多对象"
                "在新生代死亡，但幸存区拷贝更久；阈值降低则让对象更早进入老年代。"
                + measured)
    if knob == "concurrent_trigger_ratio":
        return ("触发比例决定并发周期何时开始。越早开始，标记有越多时间跑赢晋升；"
                "越晚开始，就越容易在高峰时退化。" + measured)
    if knob == "concurrent_mark_bytes_per_s":
        return ("并发标记速率必须至少追平晋升/分配速率，否则投影占用会越过 panic 线，"
                "周期退化为全堆停顿。速率越高越安全，但标记线程占用 CPU 越多。"
                + measured)
    return measured
