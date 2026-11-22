"""Cohort-level generational GC simulation (the main engine).

Semantics of an object
----------------------
An object belongs to a traffic class with survival function ``S(a)``, the
probability it is still reachable ``a`` seconds after allocation.  It is *not*
killed by a collection.  A collection merely decides what to do with the
objects that happen to be alive:

* **scavenge** -- live eden objects are copied to a survivor space and their
  scavenge age becomes 1; live survivor objects are copied with age+1.  An
  object whose age reaches ``promotion_age`` is promoted to old.  If the
  survivor space cannot hold the live objects, the oldest are pre-tenured
  (promotion threshold effectively lowered).  On to-space exhaustion
  everything is promoted.
* **full GC** -- the whole heap is collected in one pause; live young objects
  are promoted and old garbage is reclaimed.

Representation
--------------
A lone object cannot be tracked individually at production volume (a trading
day is ~10^9 allocations), so the engine works on **cohorts**: the expected
byte volume of one traffic class allocated in a 0.2 s slice.  Cohort live
bytes at time ``t`` are ``V * S(t - birth)`` exactly.  Age bookkeeping is per
cohort, so the promotion-age mechanism is implemented on real scavenge counts.
The discrete object-level engine in :mod:`gcsim.oracle` implements the same
mechanism object by object; the test suite compares the two.

Old-generation occupancy
------------------------
Old occupancy has two distinct quantities, as in a real collector:

* ``committed`` -- bytes promoted into old and not yet reclaimed.  Garbage
  occupies the heap until a collection sweeps it, so this is what triggers GC.
* ``live`` -- expected reachable bytes, derived from promoted cohorts'
  survival functions.  This is what a marking cycle discovers.

A concurrent cycle starts at ``initiating_occupancy * old_size``.  Marking
scans live old bytes at ``mark_scan_bytes_per_s``; when it completes there is a
short remark pause and the garbage found is reclaimed.  If old committed
occupancy reaches the emergency threshold first, the cycle is a *concurrent
mode failure* and degrades to a full stop-the-world collection.  This is the
mechanism that turns a slightly-too-small old generation or a slightly-too-high
promotion rate into occasional 100-600 ms pauses.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GCConfig, ServiceProfile, TrafficProfile
from .gclog import GCLogWriter
from .survival import surv, quantile_age
from .workload import CohortTrace, build_trace

HOUR = 3600.0
_EPS = 1e-12


@dataclass
class Segment:
    """A piece of an allocation cohort as it moves through the young gen."""

    __slots__ = ("birth", "bytes", "pid", "mean", "first_scav")

    def __init__(self, birth: float, nbytes: float, pid: int, mean: float,
                 first_scav: int = -1):
        self.birth = birth
        self.bytes = nbytes
        self.pid = pid
        self.mean = mean
        self.first_scav = first_scav


@dataclass
class RunStats:
    pauses: List[Tuple[float, float, str]] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    timeline: List[dict] = field(default_factory=list)
    young_gcs: int = 0
    mixed_gcs: int = 0
    remark_count: int = 0
    full_gcs: int = 0
    concurrent_cycles: int = 0
    degraded_cycles: int = 0
    promotion_bytes: float = 0.0
    promotion_objects: float = 0.0
    pretenured_bytes: float = 0.0
    pretenured_objects: float = 0.0
    tospace_exhausted: int = 0
    allocated_bytes: float = 0.0
    allocated_objects: float = 0.0
    copied_bytes: float = 0.0
    copied_objects: float = 0.0
    promoted_age_bytes: Dict[int, float] = field(default_factory=dict)
    promoted_age_objects: Dict[int, float] = field(default_factory=dict)
    promoted_class_bytes: Dict[str, float] = field(default_factory=dict)
    old_capacity: int = 0
    young_capacity: int = 0
    old_peak_committed: float = 0.0
    old_peak_live: float = 0.0
    oom_events: int = 0
    surv_occ_series: List[Tuple[float, float]] = field(default_factory=list)
    surv_occupancy_peak: float = 0.0
    promotion_rate_samples: List[Tuple[float, float]] = field(default_factory=list)
    alloc_rate_samples: List[Tuple[float, float]] = field(default_factory=list)
    hour_summary: List[dict] = field(default_factory=list)
    collection_wall_time: float = 0.0
    pause_reasons: Dict[str, int] = field(default_factory=dict)
    concurrent_overhead_s: float = 0.0   # wall time a cycle was concurrently
                                          # marking (not a throughput loss by
                                          # itself)
    app_cpu_lost_s: float = 0.0           # app throughput lost to marking
                                          # threads competing for cores
    # age histogram of *live young* bytes sampled at scavenges (age -> bytes)
    young_age_bytes: Dict[int, float] = field(default_factory=dict)
    young_age_samples: int = 0
    # pause over the run in (t, pause) form for plotting
    pause_series: List[Tuple[float, float, str]] = field(default_factory=list)

    def pause_percentiles(self) -> Dict[str, float]:
        if not self.pauses:
            return {}
        xs = np.fromiter((p for _, p, _ in self.pauses), dtype=np.float64)
        out = {}
        for q in (50, 90, 95, 99, 99.9):
            out[f"p{q:g}"] = float(np.percentile(xs, q))
        out["max"] = float(xs.max())
        return out

    def total_pause_s(self) -> float:
        return float(sum(p for _, p, _ in self.pauses))

    def stw_pause_count(self) -> int:
        return len(self.pauses)

    def survivor_bytes_peak(self) -> float:
        return self.surv_occupancy_peak


class HeapSimulator:
    """Run one day of one service under one GC configuration."""

    def __init__(self, cfg: GCConfig, service: ServiceProfile,
                 traffic: TrafficProfile, seed: int = 12345,
                 trace: Optional[CohortTrace] = None,
                 log_writer: Optional[GCLogWriter] = None,
                 record_timeline_s: float = 30.0,
                 cohort_seconds: float = 0.2,
                 timeline_max_points: int = 4000):
        errs = cfg.validate()
        if errs:
            raise ValueError("invalid GC config: " + "; ".join(errs))
        self.cfg = cfg
        self.service = service
        self.traffic = traffic
        self.seed = seed
        self.trace = trace if trace is not None else build_trace(
            service, traffic, seed, cohort_seconds)
        self.writer = log_writer
        self.timeline_every = record_timeline_s
        self.timeline_max_points = timeline_max_points
        self.stats = RunStats(old_capacity=cfg.heap.old_size,
                              young_capacity=cfg.heap.young_size)
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        tr = self.trace
        n = len(tr.cohorts)
        self.n_cohorts = n
        self.birth = np.fromiter((c.birth for c in tr.cohorts), np.float64, n)
        self.bytes = np.fromiter((c.bytes for c in tr.cohorts), np.float64, n)
        self.dt = np.fromiter((c.dt for c in tr.cohorts), np.float64, n)
        self.mean = np.fromiter((c.mean_size for c in tr.cohorts), np.float64, n)
        self.pid = np.fromiter((c.class_index for c in tr.cohorts), np.int16, n)
        self.cum = np.cumsum(self.bytes)
        # survival parameters per class index (unique set)
        self.params: List[tuple] = []
        seen: Dict[tuple, int] = {}
        for c in tr.cohorts:
            key = (c.kind, c.p1)
            if key not in seen:
                seen[key] = len(self.params)
                self.params.append(key)
        self.ppid = np.fromiter(
            (seen[(c.kind, c.p1)] for c in tr.cohorts), np.int16, n
        )
        # per class mean size for object-count estimates
        self.class_mean = np.asarray(tr.class_mean_sizes, dtype=np.float64)
        self.class_names = tr.class_names

        # ---- heap state ----
        self.eden: List[Segment] = []
        self.eden_bytes = 0.0
        self.eden_cum = 0.0        # cumulative allocated bytes at last scavenge
        self.cursor = 0            # next cohort to allocate
        self.pending: Optional[Segment] = None   # remainder of a split cohort
        self.survivors: List[Segment] = []
        self.survivor_bytes = 0.0
        self.old_committed = 0.0
        # old buckets: (time_bucket, class) -> [bytes-weighted birth, bytes]
        self._old_bucket: Dict[Tuple[int, int], list] = {}
        self._old_dirty = True
        self._old_prune_birth = None
        self.old_bucket_s = 60.0
        self._live_cache_t = -1e18
        self._live_cache_v = 0.0
        self.live_cache_window = 10.0
        self.scav_count = 0
        self.eden_limit = float(self.cfg.heap.eden())
        self.t = 0.0
        self.degraded_left = 0
        self.bad_scavenges = 0
        # adaptive tenure threshold starts at the configured promotion age
        self.tenure_threshold = self.cfg.young.promotion_age
        # concurrent cycle
        self.marking = False
        self._last_t = 0.0
        self._last_prune_t = -1e18
        self.mark_start = 0.0
        self.mark_last_t = 0.0
        self.mark_scan_done = 0.0
        self.mark_base_live = 0.0
        self.mark_cycle_promoted = 0.0
        self._last_timeline = -1e18
        self._last_concurrent_t = 0.0
        self._marking_cpu = 0.0

    # ------------------------------------------------------------------
    # survival helpers
    # ------------------------------------------------------------------
    def _s(self, pid: int, age: float) -> float:
        kind, p = self.params[pid]
        return surv(kind, p, age)

    def _live(self, seg: Segment, t: float) -> float:
        return seg.bytes * self._s(seg.pid, t - seg.birth)

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------
    def _allocate_until_eden_full(self) -> None:
        """Allocate cohorts into eden until the eden budget is exhausted.

        Consumes from ``cursor``; a cohort that straddles the boundary is split
        so the eden exactly fills, with the remainder carried in ``pending``.
        """
        limit = self.eden_limit
        if self.pending is not None:
            seg = self.pending
            room = limit - self.eden_bytes
            if seg.bytes <= room:
                self.eden.append(seg)
                self.eden_bytes += seg.bytes
                self.pending = None
            else:
                f = room / seg.bytes if seg.bytes else 0.0
                part = Segment(seg.birth, room, seg.pid, seg.mean)
                rest = Segment(self.t, seg.bytes - room, seg.pid, seg.mean)
                self.eden.append(part)
                self.eden_bytes += room
                self.pending = rest
                return
        while self.cursor < self.n_cohorts:
            room = limit - self.eden_bytes
            if room <= 0:
                return
            b = float(self.bytes[self.cursor])
            birth = float(self.birth[self.cursor])
            pid = int(self.ppid[self.cursor])
            mean = float(self.mean[self.cursor])
            if b <= room:
                self.eden.append(Segment(birth, b, pid, mean))
                self.eden_bytes += b
                self.eden_cum += b
                self.cursor += 1
            else:
                f = room / b if b else 0.0
                self.eden.append(Segment(birth, room, pid, mean))
                self.eden_bytes += room
                # remainder starts after this slice; its birth is the slice end
                rem_birth = birth + self.dt[self.cursor] * f
                self.pending = Segment(rem_birth, b - room, pid, mean)
                self.eden_cum += room
                self.cursor += 1
                return

    def _next_alloc_time(self) -> float:
        """Time of the next byte of allocation (for advancing the clock)."""
        if self.pending is not None:
            return self.pending.birth
        if self.cursor < self.n_cohorts:
            return float(self.birth[self.cursor])
        return math.inf

    # ------------------------------------------------------------------
    # scavenge
    # ------------------------------------------------------------------
    def scavenge(self, t: float) -> None:
        cfg = self.cfg
        self.scav_count += 1
        sc = self.scav_count
        eden_b = self.eden_bytes
        surv_b = self.survivor_bytes
        heap_b = self.eden_bytes + self.survivor_bytes + self.old_committed
        old_b = self.old_committed

        # 1. Live bytes now.
        eden_live = [self._live(s, t) for s in self.eden]
        surv_live = [self._live(s, t) for s in self.survivors]
        eden_live_total = sum(eden_live)
        surv_live_total = sum(surv_live)

        age_limit = cfg.young.promotion_age
        promoted: List[Tuple[Segment, float, int]] = []   # (seg, live_bytes, age)
        candidates: List[Tuple[Segment, float, int, int]] = []  # seg, live, age_after, class

        # 2. Existing survivors: age after this copy is sc - first_scav + 1.
        for s, live in zip(self.survivors, surv_live):
            if live <= 0.0:
                continue
            age_after = sc - s.first_scav + 1
            if age_after >= age_limit:
                promoted.append((s, live, age_after))
            else:
                candidates.append((s, live, age_after, s.pid))
        # 3. Eden survivors: age after copy is 1.
        for s, live in zip(self.eden, eden_live):
            if live <= 0.0:
                continue
            candidates.append((s, live, 1, s.pid))

        # 4. Survivor-space fit and the tenuring threshold.
        #
        # The configured ``promotion_age`` is the *maximum* tenure age.  The
        # collector may tenure earlier: the survivor spaces have a fixed size,
        # and HotSpot lowers the effective tenuring threshold when they fill
        # (DesiredSurvivorSizePolicy).  The rule modelled here is the one the
        # mechanism actually implements:
        #
        #     effective_age = smallest t such that the bytes of all survivors
        #                     with age_after < t fit in the survivor space,
        #                     capped at promotion_age.
        #
        # If even t = 1 does not fit (the survivors of a single scavenge exceed
        # the space), that is to-space exhaustion and everything is promoted.
        # This makes the survivor ratio a real knob: a survivor space too small
        # for the medium-lived cohort forces the threshold down to 1 and
        # promotes everything early, regardless of how high promotion_age is.
        space = float(cfg.heap.survivor_space())
        candidates.sort(key=lambda x: (x[2], -x[0].birth))
        by_age: List[float] = [0.0] * (cfg.young.promotion_age + 1)
        for seg, live, age_after, pid in candidates:
            a = min(age_after, cfg.young.promotion_age)
            by_age[a] += live
        # Choose the highest threshold whose kept bytes (objects with
        # age_after < t) fit in the survivor space, capped at promotion_age.
        eff_age = cfg.young.promotion_age
        if cfg.young.adaptive_tenuring:
            run = 0.0
            fitted = cfg.young.promotion_age
            for thr in range(1, cfg.young.promotion_age + 1):
                # keep objects whose age_after < thr, i.e. ages 1..thr-1
                run += by_age[thr - 1]
                if run > space:
                    fitted = max(1, thr - 1)
                    break
            eff_age = fitted
            if sum(by_age) <= space:
                eff_age = cfg.young.promotion_age
        keep_flags: List[bool] = []
        kept_total = 0.0
        pretenured_bytes = 0.0
        pretenured_objs = 0.0
        exhausted = False
        for seg, live, age_after, pid in candidates:
            if age_after >= eff_age:
                # promoted because it reached the (effective) tenure age
                promoted.append((seg, live, age_after))
                if age_after < cfg.young.promotion_age:
                    pretenured_bytes += live
                    pretenured_objs += live / max(_EPS, seg.mean)
                keep_flags.append(False)
            else:
                keep_flags.append(True)
                kept_total += live
        # A single-scavenge survivor set larger than the whole space cannot be
        # copied at all: everything is promoted.
        if kept_total > space:
            exhausted = True
            for i, (seg, live, age_after, pid) in enumerate(candidates):
                if keep_flags[i]:
                    promoted.append((seg, live, age_after))
                    keep_flags[i] = False
            pretenured_bytes = 0.0
            pretenured_objs = 0.0
            kept_total = 0.0

        # 5. Install new survivor state: kept segments get their age anchored
        #    to this scavenge if this is their first copy.
        new_survivors: List[Segment] = []
        copied_objs = 0.0
        for flag, (seg, live, age_after, pid) in zip(keep_flags, candidates):
            if flag:
                if seg.first_scav <= 0:
                    seg.first_scav = sc
                new_survivors.append(seg)
                copied_objs += live / max(_EPS, seg.mean)
        self.survivors = new_survivors
        self.survivor_bytes = kept_total
        copied_bytes = kept_total

        # 6. Promote to old.
        promoted_live = sum(p[1] for p in promoted)
        promoted_objs = sum(p[1] / max(_EPS, p[0].mean) for p in promoted)
        if promoted_live > 0:
            self._promote(t, promoted, promoted_objs)

        # 7. Pause.
        pause = self._scavenge_pause(copied_bytes, copied_objs, promoted_live)
        self.stats.young_gcs += 1
        self.stats.copied_bytes += copied_bytes
        self.stats.copied_objects += copied_objs
        if exhausted:
            self.stats.tospace_exhausted += 1
        self.stats.pretenured_bytes += pretenured_bytes
        self.stats.pretenured_objects += pretenured_objs
        self._stats_young_age(t, new_survivors)
        self.eden = []
        self.eden_bytes = 0.0
        self._record_event(
            t, "young", pause,
            eden_before=eden_b, eden_after=0.0,
            surv_before=surv_b, surv_after=kept_total,
            heap_before=heap_b, heap_after=self.heap_bytes(),
            old_after=self.old_committed, promotion=promoted_live,
            pretenured=pretenured_bytes, survivors=copied_objs,
            copied=copied_bytes,
            detail={"promoted_bytes": promoted_live,
                    "eden_live": eden_live_total,
                    "surv_live": surv_live_total,
                    "exhausted": exhausted,
                    "eff_age": eff_age,
                    "promoted_ages": [a for _, _, a in promoted]},
        )
        self.stats.surv_occ_series.append((t, kept_total))
        if kept_total > self.stats.surv_occupancy_peak:
            self.stats.surv_occupancy_peak = kept_total
        if self.writer:
            self.writer.young(
                t, pause, eden_b, self.eden_bytes, float(self.eden_limit),
                surv_b, kept_total, heap_b, self.heap_bytes(),
                float(self.cfg.heap.old_size + self.cfg.heap.young_size),
                cause="G1 Evacuation Pause",
                pretenured=int(pretenured_objs),
                old_b=old_b, old_a=self.old_committed,
                promoted_cum=self.stats.promotion_bytes,
            )

        # 8. Adaptive degradation feedback (survival ratio too high).
        live_in = eden_b + surv_b
        ratio = (copied_bytes + promoted_live) / live_in if live_in > 0 else 0.0
        self._update_degradation(ratio)
        # 9. Old-gen pacing.
        self._after_young_gc(t)
        self._maybe_timeline(t)

    def _scavenge_pause(self, copied_bytes: float, copied_objs: float,
                        promoted_bytes: float) -> float:
        h = self.cfg.heap
        ns = (
            copied_bytes * h.copy_ns_per_byte * h.copy_graph_factor
            + copied_objs * h.copy_ns_per_object
            + promoted_bytes * h.copy_ns_per_byte * h.rs_fixup_frac
        )
        return 0.5e-3 + ns * 1e-9

    def _reclaim_old(self, target_live_frac: float = 1.0) -> float:
        """Reduce committed old bytes to ``live + (1-frac)*garbage``.

        Reclaiming must operate on the *buckets* (the source of truth for old
        occupancy), not on a scalar, or the two drift apart and old never
        fills (or fills without becoming live).  Each bucket's committed bytes
        are scaled by the same factor; its birth distribution is unchanged, so
        subsequent live estimates stay consistent.
        """
        live = self.old_live(self._last_t)
        committed = self.old_committed
        if committed <= 0:
            return 0.0
        garbage = max(0.0, committed - live)
        reclaim = garbage * (1.0 - target_live_frac)
        if reclaim <= 0:
            return 0.0
        new_committed = committed - reclaim
        scale = new_committed / committed if committed > 0 else 1.0
        for v in self._old_bucket.values():
            v[1] *= scale
        self.old_committed = new_committed
        self._old_dirty = True
        self._live_cache_t = -1e18
        return reclaim

    def _promote(self, t: float, promoted: List[Tuple[Segment, float, int]],
                 promoted_objs: float) -> None:
        """Move live segments to old and account for them.

        Old occupancy is tracked in two layers:

        * ``old_committed`` -- promoted bytes not yet reclaimed.  This is the
          pressure that drives collection.
        * **old buckets** -- promotions aggregated per (class, promotion-time
          bucket).  A bucket knows promotion time, total bytes and the
          bytes-weighted mean birth time.  Since survival is evaluated from
          each object's birth, the expected live bytes of a bucket at time t is
          ``bytes * mean_S(t - birth)`` -- exact for buckets where all members
          have the same class, and a very tight approximation otherwise (birth
          times within one bucket differ by at most the bucket width).

        Bucketing keeps old accounting O(#buckets * #classes) instead of
        O(#scavenges * #classes), which matters at one-day scale.
        """
        by_pid: Dict[int, List[Tuple[float, float]]] = {}
        total = 0.0
        for seg, live, age in promoted:
            if live <= 0.0:
                continue
            total += live
            by_pid.setdefault(seg.pid, []).append((seg.birth, live))
            self.stats.promoted_age_bytes[age] = (
                self.stats.promoted_age_bytes.get(age, 0.0) + live)
            self.stats.promoted_age_objects[age] = (
                self.stats.promoted_age_objects.get(age, 0.0)
                + live / max(_EPS, seg.mean))
            cname = self.class_names[seg.pid] if seg.pid < len(self.class_names) else str(seg.pid)
            self.stats.promoted_class_bytes[cname] = (
                self.stats.promoted_class_bytes.get(cname, 0.0) + live)
        if total <= 0:
            return
        bucket = int(t // self.old_bucket_s)
        for pid, items in by_pid.items():
            sb = sum(v for _, v in items)
            wb = sum(b * v for b, v in items) / sb
            key = (bucket, pid)
            cur = self._old_bucket.get(key)
            if cur is None:
                # [weighted mean birth, committed bytes at promotion,
                #  promotion time]
                self._old_bucket[key] = [wb, sb, t]
            else:
                new_total = cur[1] + sb
                cur[0] = (cur[0] * cur[1] + wb * sb) / new_total
                cur[1] = new_total
        self._old_dirty = True
        self._live_cache_t = -1e18
        self.old_committed += total
        self.stats.promotion_bytes += total
        self.stats.promotion_objects += promoted_objs
        if self.marking:
            self.mark_cycle_promoted += total

    # ------------------------------------------------------------------
    # old generation
    # ------------------------------------------------------------------
    def _old_arrays(self) -> None:
        """Materialise old buckets into parallel numpy arrays for fast sums.

        Rebuilt only when the bucket set or its contents changed.  Rebuilding
        is O(#buckets), and #buckets grows by a handful per scavenge, so after
        warm-up this is a small fixed cost per collection rather than a scan of
        the whole day's promotions.
        """
        if not self._old_dirty:
            return
        keys = list(self._old_bucket.keys())
        n = len(keys)
        self._o_bucket = np.array([k[0] for k in keys], dtype=np.int64)
        self._o_pid = np.array([k[1] for k in keys], dtype=np.int16)
        self._o_wbirth = np.array([self._old_bucket[k][0] for k in keys], dtype=np.float64)
        self._o_committed = np.array([self._old_bucket[k][1] for k in keys], dtype=np.float64)
        self._o_tp = np.array([self._old_bucket[k][2] for k in keys], dtype=np.float64)
        self._old_dirty = False

    def old_live(self, t: float) -> float:
        """Expected live bytes in old at time t (marking-visible live set).

        A bucket is the set of objects promoted at time ``tp``; ``committed``
        is what they occupied on arrival, i.e. ``V * S(tp - birth)``.  From
        then on they die according to the *conditional* survival of having
        reached ``tp``:

            live(t) = committed * S(t - birth) / S(tp - birth)

        Applying unconditional ``S(t - birth)`` instead would make a cohort
        appear dead the moment it is promoted whenever its age at promotion is
        already a large fraction of its lifetime -- an easy and fatal mistake.
        """
        if not self._old_bucket:
            return 0.0
        if abs(t - self._live_cache_t) < self.live_cache_window:
            return self._live_cache_v
        self._old_arrays()
        a = t - self._o_wbirth
        ap = self._o_tp - self._o_wbirth
        total = 0.0
        for k in range(len(self.params)):
            m = self._o_pid == k
            if not m.any():
                continue
            kind, p = self.params[k]
            if kind == "lognormal_mix":
                med, sig, lf, lmed, lsig = p
                s_now = _lnmix_vec(a[m], med, sig, lf, lmed, lsig)
                s_p = _lnmix_vec(ap[m], med, sig, lf, lmed, lsig)
            elif kind == "pareto":
                xm, alpha, tmax = p
                s_now = _pareto_vec(a[m], xm, alpha, tmax)
                s_p = _pareto_vec(ap[m], xm, alpha, tmax)
            else:
                life, jit = p
                s_now = _det_vec(a[m], life, jit)
                s_p = _det_vec(ap[m], life, jit)
            ok = s_p > 1e-6
            if ok.any():
                total += float(
                    (self._o_committed[m][ok] * s_now[ok] / s_p[ok]).sum())
        self._live_cache_t = t
        self._live_cache_v = total
        return total

    def _prune_old_rows(self, t: float) -> None:
        """Drop buckets whose members have certainly died.

        A bucket is pruned once all of its members have aged past the point
        where survival is effectively zero (the 99.99th-percentile lifetime of
        the bucket's class).  Pruning runs at most every few minutes of
        simulated time -- running it on every collection would dominate the
        runtime for no accuracy benefit, since culled buckets contribute at
        most 1e-4 of their bytes.
        """
        if not self._old_bucket:
            return
        if t - self._last_prune_t < 30.0:
            return
        self._last_prune_t = t
        cutoff = self._old_prune_birth
        if cutoff is None:
            cutoff = {}
            for k in range(len(self.params)):
                kind, p = self.params[k]
                cutoff[k] = quantile_age(kind, p, 1e-4)
            self._old_prune_birth = cutoff
        # Vectorised: gather bucket births and classes, find dead keys.
        keys = list(self._old_bucket.keys())
        ages = np.array([t - self._old_bucket[k][0] for k in keys])
        pids = np.array([k[1] for k in keys])
        cut = np.array([cutoff[int(pid)] for pid in pids])
        dead_idx = np.nonzero(ages > cut)[0]
        if dead_idx.size:
            for j in dead_idx.tolist():
                del self._old_bucket[keys[j]]
            self._old_dirty = True
            self._live_cache_t = -1e18

    def _after_young_gc(self, t: float) -> None:
        cfg = self.cfg
        old_size = cfg.heap.old_size
        occ = self.old_committed / old_size
        if cfg.old.mode == "concurrent":
            if self.marking:
                self._advance_marking(t)
                if not self.marking:
                    return
                if self.old_committed >= cfg.old.emergency_occupancy * old_size:
                    self._concurrent_failure(t)
                    return
            else:
                if occ >= cfg.old.initiating_occupancy:
                    self._start_marking(t)
                    if self.old_committed >= cfg.old.emergency_occupancy * old_size:
                        self._concurrent_failure(t)
                        return
        else:
            if occ >= cfg.old.initiating_occupancy:
                self._full_gc(t, "Allocation Failure")

    def _start_marking(self, t: float) -> None:
        cfg = self.cfg
        self._prune_old_rows(t)
        live = self.old_live(t)
        self.marking = True
        self.mark_start = t
        self.mark_scan_done = 0.0
        self.mark_base_live = max(live, 1.0)
        self.mark_cycle_promoted = 0.0
        self.mark_last_t = t
        self.stats.concurrent_cycles += 1
        if self.writer:
            self.writer.concurrent(t, "mark-start")

    def _advance_marking(self, t: float) -> None:
        """Advance concurrent marking to time ``t``.

        Marking scans at a fixed byte rate.  Work is the live set at cycle
        start plus everything promoted since (SATB requires re-marking new
        old objects).  If promotion outruns the scan rate, the cycle never
        completes and the old generation keeps filling; the emergency check in
        :meth:`_after_young_gc` then degrades to a full GC.  That is the
        modelled failure mode.
        """
        cfg = self.cfg
        if not self.marking:
            return
        # The clock must advance monotonically for marking accounting (the main
        # loop may call this with an earlier t after a scavenge), and no single
        # call may be credited more than a bounded interval: a jump across an
        # idle gap must not be booked as marking time.
        if t <= self.mark_last_t:
            return
        dt = min(t - self.mark_last_t, cfg.old.max_cycle_s)
        self.mark_last_t = t
        thr = cfg.old.mark_scan_bytes_per_s * (1.0 - cfg.old.mark_overhead)
        self.mark_scan_done += dt * thr
        self._marking_cpu += dt
        work = self.mark_base_live + self.mark_cycle_promoted
        # A concurrent cycle must terminate.  If the promoted-during-cycle
        # bytes keep the work ahead of the scan for longer than a bounded
        # horizon, the marker has lost the race: force the remark (which will
        # then see the backlog and degrade).  Without this a cycle can starve
        # forever while promotion trickles in, and the marking threads would
        # appear to run all day, which is not what a real collector does.
        stale = (t - self.mark_start) > cfg.old.max_cycle_s
        if self.mark_scan_done >= work or stale:
            self._remark(t)

    def _remark(self, t: float) -> None:
        cfg = self.cfg
        self.marking = False
        self._prune_old_rows(t)
        live = self.old_live(t)
        heap_b = self.heap_bytes()
        work = self.mark_base_live + self.mark_cycle_promoted
        # The cycle is only useful if it scanned at least the base live set;
        # if the live set outgrew what was scanned, the remark has to finish
        # the remaining work in a stop-the-world fashion.
        backlog = max(0.0, work - self.mark_scan_done)
        kept_up = backlog <= 0.02 * work
        # Reclaim garbage discovered by marking.  Mixed collections reclaim
        # most, not all, of it; the remainder waits for the next cycle.
        self._last_t = t
        reclaimed = self._reclaim_old(cfg.old.mixed_residual_garbage)
        # Remark pause = base + finishing backlog + re-scan of roots.
        pause = (cfg.old.remark_base_ms * 1e-3
                 + live * cfg.old.remark_ns_per_live_byte * 1e-9
                 + backlog / max(1.0, cfg.old.mark_scan_bytes_per_s) * 0.5)
        if not kept_up:
            pause += 0.010
            self.stats.degraded_cycles += 1
        self.stats.mixed_gcs += 1
        self.stats.remark_count += 1
        # Marking wall time can never exceed the cycle's elapsed wall time;
        # clamp as a safety net against pathological clock motion so a single
        # bad interval cannot swamp the day's throughput accounting.
        cycle_wall = max(0.0, t - self.mark_start)
        marked = min(self._marking_cpu, cycle_wall) if cycle_wall > 0 else self._marking_cpu
        self.stats.concurrent_overhead_s += marked
        # The app loses the marker threads' share of CPU *while marking runs*.
        share = min(0.9, cfg.old.conc_gc_threads / max(1, cfg.old.app_cores))
        self.stats.app_cpu_lost_s += marked * share
        self._marking_cpu = 0.0
        self._record_event(
            t, "remark", pause,
            eden_before=self.eden_bytes, eden_after=self.eden_bytes,
            surv_before=self.survivor_bytes, surv_after=self.survivor_bytes,
            heap_before=heap_b, heap_after=self.heap_bytes(),
            old_after=self.old_committed, promotion=0.0, pretenured=0.0,
            survivors=0.0, copied=0.0,
            detail={"live": live, "reclaimed": reclaimed,
                    "kept_up": kept_up, "backlog": backlog,
                    "cycle_promoted": self.mark_cycle_promoted},
        )
        if self.writer:
            self.writer.remark(t, pause, heap_b, self.heap_bytes(),
                               self.heap_cap, mark_kept_up=kept_up)
        if not kept_up and self.old_committed >= cfg.old.initiating_occupancy * cfg.heap.old_size:
            self._full_gc(t + 1e-6, "Concurrent Mode Failure")
        elif self.old_committed >= cfg.old.emergency_occupancy * cfg.heap.old_size:
            self._full_gc(t + 1e-6, "To-space Exhausted")

    def _concurrent_failure(self, t: float) -> None:
        self.marking = False
        self.stats.degraded_cycles += 1
        self._full_gc(t, "Concurrent Mode Failure")

    def _full_gc(self, t: float, cause: str) -> None:
        cfg = self.cfg
        self.marking = False
        heap_b = self.heap_bytes()
        live_old = self.old_live(t)
        # Live young: everything alive in eden and survivor space moves to old
        # during the full collection (the young generation is cleared).
        young_live = 0.0
        young_objs = 0.0
        for s in self.eden:
            v = self._live(s, t)
            young_live += v
            young_objs += v / max(_EPS, s.mean)
        for s in self.survivors:
            v = self._live(s, t)
            young_live += v
            young_objs += v / max(_EPS, s.mean)
        live = live_old + young_live
        pause = (cfg.old.full_gc_base_ms * 1e-3
                 + live * cfg.old.full_gc_ns_per_live_mb * 1e-9 / (1024 ** 2))
        if cfg.old.full_gc_compacts and live_old:
            pause += live_old * cfg.heap.compact_ns_per_live_mb * 1e-9 / (1024 ** 2)
        # Account the young evacuation as promotion into old.
        if young_live > 0:
            segs = []
            for s in self.eden + self.survivors:
                v = self._live(s, t)
                if v <= 0:
                    continue
                age = (self.scav_count - s.first_scav + 1) if s.first_scav > 0 else 1
                segs.append((s, v, age))
            self._promote(t, segs, young_objs)
        self.eden = []
        self.eden_bytes = 0.0
        self.survivors = []
        self.survivor_bytes = 0.0
        # Full GC reclaims all garbage: committed collapses to exactly live.
        live_after = self.old_live(t)
        if self.old_committed > 0:
            scale = (live_after / self.old_committed) if live_after > 0 else 0.0
            for v in self._old_bucket.values():
                v[1] *= scale
            self.old_committed = live_after
            self._old_dirty = True
            self._live_cache_t = -1e18
        self._prune_old_rows(t)
        if live > cfg.heap.old_size:
            self.stats.oom_events += 1
        self.stats.full_gcs += 1
        self._record_event(
            t, "full", pause,
            eden_before=heap_b, eden_after=0.0, surv_before=0.0, surv_after=0.0,
            heap_before=heap_b, heap_after=self.heap_bytes(),
            old_after=self.old_committed, promotion=young_live,
            pretenured=0.0, survivors=0.0, copied=0.0,
            detail={"cause": cause, "live": live, "live_old": live_old,
                    "young_live": young_live},
        )
        if self.writer:
            self.writer.full(t, pause, heap_b, self.heap_bytes(), self.heap_cap,
                             cause=cause)
        self._maybe_timeline(t, force=True)

    # ------------------------------------------------------------------
    # feedback / stats
    # ------------------------------------------------------------------
    def _update_degradation(self, ratio: float) -> None:
        cfg = self.cfg
        if ratio > cfg.young.degraded_trigger_survival:
            self.bad_scavenges += 1
            if (self.bad_scavenges >= cfg.young.degraded_trigger_n
                    and self.degraded_left == 0):
                self.degraded_left = cfg.young.degraded_scavenges
                self.eden_limit = max(
                    cfg.heap.eden() // 4,
                    cfg.heap.eden() * cfg.young.degraded_fraction)
        else:
            self.bad_scavenges = 0
        if self.degraded_left > 0:
            self.degraded_left -= 1
            if self.degraded_left == 0:
                self.eden_limit = float(cfg.heap.eden())

    def _stats_young_age(self, t: float, kept: List[Segment]) -> None:
        """Record the age distribution of the live young population."""
        self.stats.young_age_samples += 1
        for s in kept:
            age = (self.scav_count - s.first_scav + 1) if s.first_scav > 0 else 1
            v = self._live(s, t)
            self.stats.young_age_bytes[age] = (
                self.stats.young_age_bytes.get(age, 0.0) + v)

    def _record_event(self, t, kind, pause, **kw) -> None:
        self.stats.pauses.append((t, pause, kind))
        self.stats.pause_series.append((t, pause, kind))
        self.stats.pause_reasons[kind] = self.stats.pause_reasons.get(kind, 0) + 1
        ev = {"t": t, "pause": pause, "kind": kind}
        ev.update(kw)
        ev.setdefault("old_committed", self.old_committed)
        ev.setdefault("old_live", kw.get("detail", {}).get("live"))
        ev.setdefault("young", self.young_bytes())
        self.stats.events.append(ev)
        self.stats.old_peak_committed = max(self.stats.old_peak_committed, self.old_committed)

    def _maybe_timeline(self, t: float, force: bool = False) -> None:
        if not force and t - self._last_timeline < self.timeline_every:
            return
        self._last_timeline = t
        live = self.old_live(t) if (self._old_bucket and self.cfg.heap.old_size) else 0.0
        self.stats.old_peak_live = max(self.stats.old_peak_live, live)
        st = self.stats
        st.timeline.append({
            "t": t,
            "eden": self.eden_bytes,
            "survivor": self.survivor_bytes,
            "young": self.eden_bytes + self.survivor_bytes,
            "old_committed": self.old_committed,
            "old_live": live,
            "heap": self.heap_bytes(),
            "old_frac": self.old_committed / self.cfg.heap.old_size,
            "young_frac": (self.eden_bytes + self.survivor_bytes) / self.cfg.heap.young_size,
            "promoted_cum": st.promotion_bytes,
            "alloc_cum": float(self.cum[self.cursor - 1]) if self.cursor else 0.0,
            "pauses_cum": len(st.pauses),
            "scav_count": self.scav_count,
        })
        # thin the timeline if it grows beyond the cap
        if len(st.timeline) > self.timeline_max_points:
            st.timeline = st.timeline[::2]

    # ------------------------------------------------------------------
    def young_bytes(self) -> float:
        return self.eden_bytes + self.survivor_bytes

    def heap_bytes(self) -> float:
        return self.eden_bytes + self.survivor_bytes + self.old_committed

    @property
    def heap_cap(self) -> int:
        return self.cfg.heap.old_size + self.cfg.heap.young_size

    def _mark_finish_estimate(self, t: float) -> Optional[float]:
        """Earliest time marking could complete if no further scavenge occurs.

        Between scavenges the marking work is constant, so this is exact.  If
        the scan rate is zero or the backlog is unbounded, returns None.
        """
        if not self.marking:
            return None
        cfg = self.cfg
        thr = cfg.old.mark_scan_bytes_per_s * (1.0 - cfg.old.mark_overhead)
        if thr <= 0:
            return None
        work = self.mark_base_live + self.mark_cycle_promoted
        remaining = work - self.mark_scan_done
        if remaining <= 0:
            return t
        return t + remaining / thr

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, max_seconds: Optional[float] = None) -> RunStats:
        t0 = _time.perf_counter()
        end = self.traffic.day_seconds
        if max_seconds is not None:
            end = min(end, max_seconds)
        t = 0.0
        stats = self.stats
        while t < end:
            # 1. Fill eden (allocation).
            self._allocate_until_eden_full()
            # 2. Advance the clock to the time of the last allocated cohort.
            if self.cursor > 0:
                t = min(end, max(t, float(self.birth[self.cursor - 1])))
            # 3. If a concurrent cycle is running, let it make progress in the
            #    wall-clock time that has passed.
            if self.marking:
                self._advance_marking(t)
            # 4. Scavenge when eden is full.
            if self.eden_bytes + 1e-9 >= self.eden_limit:
                self.scavenge(self._eden_fill_time())
                continue
            # 5. Nothing left to allocate: drain any in-flight cycle, then stop.
            if self.cursor >= self.n_cohorts and self.pending is None:
                while self.marking and t < end:
                    fin = self._mark_finish_estimate(t)
                    if fin is None:
                        break
                    t = min(end, max(t, fin))
                    self._advance_marking(t)
                break
            # 6. Move the clock to the next allocation, or to a marking
            #    completion if it comes first, or to end of day.
            nxt = min(self._next_alloc_time(), end)
            if self.marking:
                fin = self._mark_finish_estimate(t)
                if fin is not None and fin < nxt and fin <= end:
                    t = max(t, fin)
                    self._advance_marking(t)
                    continue
            if nxt <= t:
                # make progress even when allocation time does not advance
                t += 1e-9
            t = min(nxt, end)
        # end of day: finish an in-flight cycle at the end timestamp
        if self.marking:
            self._advance_marking(max(t, end))
        self._maybe_timeline(max(t, end), force=True)
        stats.allocated_bytes = float(self.cum[self.cursor - 1]) if self.cursor else 0.0
        stats.allocated_objects = float(
            (self.bytes[:self.cursor] / np.maximum(self.mean[:self.cursor], 1.0)).sum()
        ) if self.cursor else 0.0
        stats.collection_wall_time = _time.perf_counter() - t0
        # rate samples
        self._finalise_rates()
        return stats

    def _eden_fill_time(self) -> float:
        """Time at which eden became full.

        Cohorts are born uniformly across their slice ``[birth, birth+dt)``.
        Eden is filled by the last bytes placed in it, so the fill time is the
        birth of those bytes.  When a cohort straddled the boundary it was
        split; ``pending.birth`` is the split time.  Otherwise the last cohort
        completed exactly at its slice end.
        """
        if self.pending is not None and self.cursor > 0:
            return float(self.pending.birth)
        if self.cursor > 0:
            i = self.cursor - 1
            return float(self.birth[i]) + float(self.dt[i])
        return self.t

    def _finalise_rates(self) -> None:
        """Per-10-minute promotion and allocation rates for reports."""
        bucket = 600.0
        acc: Dict[int, float] = {}
        for e in self.stats.events:
            p = e.get("promotion") or 0.0
            if p:
                b = int(e["t"] // bucket)
                acc[b] = acc.get(b, 0.0) + p
        self.stats.promotion_rate_samples = [
            (b * bucket, v / bucket) for b, v in sorted(acc.items())
        ]
        tr = self.trace
        self.stats.alloc_rate_samples = [
            (float(t0), float(b) / tr.bucket_seconds)
            for t0, b in zip(tr.bucket_t0, tr.bucket_bytes)
        ]
        # hourly summaries
        hours: Dict[int, dict] = {}
        for t, p, kind in self.stats.pauses:
            h = int(t // HOUR)
            d = hours.setdefault(h, {"h": h, "pause_s": 0.0, "pauses": 0,
                                     "max_ms": 0.0, "young": 0, "remark": 0,
                                     "full": 0})
            d["pause_s"] += p
            d["pauses"] += 1
            d["max_ms"] = max(d["max_ms"], p * 1e3)
            if kind in d:
                d[kind] += 1
        self.stats.hour_summary = [hours[k] for k in sorted(hours)]


# ----------------------------------------------------------------------
# vectorised survival kernels for old-row accounting
# ----------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)


def _lnmix_vec(age: np.ndarray, med, sig, lf, lmed, lsig) -> np.ndarray:
    out = np.ones_like(age)
    pos = age > 0
    if pos.any():
        ap = age[pos]
        body = 0.5 * _erfc_vec((np.log(ap) - math.log(med)) / (sig * _SQRT2))
        if lf > 0:
            tail = 0.5 * _erfc_vec((np.log(ap) - math.log(lmed)) / (lsig * _SQRT2))
            out[pos] = (1.0 - lf) * body + lf * tail
        else:
            out[pos] = body
    return out


def _pareto_vec(age: np.ndarray, xm, alpha, tmax) -> np.ndarray:
    out = np.ones_like(age)
    over = age > xm
    gone = age >= tmax
    if over.any():
        out[over] = (xm / age[over]) ** alpha
    out[gone] = 0.0
    return out


def _det_vec(age: np.ndarray, life, jit) -> np.ndarray:
    if jit <= 0:
        return (age < life).astype(np.float64)
    lo = life * (1.0 - jit)
    hi = life * (1.0 + jit)
    out = np.ones_like(age)
    out[age >= hi] = 0.0
    between = (age > lo) & (age < hi)
    out[between] = (hi - age[between]) / (hi - lo)
    return out


try:
    from scipy.special import erfc as _scipy_erfc

    def _erfc_vec(x: np.ndarray) -> np.ndarray:
        return _scipy_erfc(x)
except Exception:  # pragma: no cover - scipy is optional
    def _erfc_vec(x: np.ndarray) -> np.ndarray:
        """Vectorised erfc via the Numerical Recipes rational approximation.

        Accurate to ~1.2e-7; used when scipy is not installed.  Avoids the
        per-element Python calls of np.vectorize, which dominate the runtime at
        one-day scale.
        """
        z = np.abs(np.asarray(x, dtype=np.float64))
        tt = 1.0 / (1.0 + 0.5 * z)
        poly = (1.00002368 + tt * (0.37409196 + tt * (0.09678418 +
                tt * (-0.18628806 + tt * (0.27886807 + tt * (-1.13520398 +
                tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277))))))))
        ans = tt * np.exp(-z * z - 1.26551223 + tt * poly)
        return np.where(np.asarray(x) >= 0, ans, 2.0 - ans)
