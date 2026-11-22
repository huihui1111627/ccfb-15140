"""Event-driven generational collector simulator.

Why sampling
------------
A literal object-by-object simulation of one day is impossible: at ~50 MB/s
with ~700-byte objects a service allocates billions of objects per day.  The
engine integrates allocation exactly from the traffic model and materialises
individual records only for objects that *survive* a young collection -- the
only objects that participate in copying, ageing, promotion and old-gen
collection.  Each surviving record carries a weight ``w``: how many real
objects it stands for.  Within a class, records materialised for one
collection share the same weight, so ``sum(w * size)`` is the true survivor
byte count in expectation.  Survivor-space overflow, tenuring thresholds,
promotion, floating garbage and concurrent-cycle debt are therefore all
modelled on real state, not on a mean-field approximation.

Collector mechanics modelled
----------------------------
* eden + two survivor spaces; allocation only in eden;
* a young GC copies live eden objects into the survivor space, ageing the
  objects already there by one;
* objects whose age exceeds ``tenuring_threshold`` are promoted;
* if the live set does not fit into the survivor space, the *oldest* objects
  are promoted early (premature promotion) -- the classic "to-space
  overflow" behaviour;
* a young pause is proportional to the number of live bytes/objects copied;
* old collections are either stop-the-world full mark-compact, or a
  concurrent cycle (initial-mark pause, concurrent mark, remark pause).  A
  cycle that cannot keep up with promotion/allocation is detected with the
  keep-up inequality (projected old occupancy at completion, including
  write-barrier debt, reaches the panic line) and then *degenerates* into a
  full stop-the-world collection.

Units: times in seconds, sizes in bytes.  All reported numbers refer to the
real service; sampling only affects how many records are kept in memory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    GCConfig, WorkloadConfig, ObjectPopulation, SimulationConfig,
    ServiceScenario, MiB, POP_NAMES,
)

EV_YOUNG = 0
EV_CONC_START = 1
EV_CONC_END = 2
EV_FULL = 3

EVENT_KIND_NAMES = {0: "young", 1: "concurrent-start", 2: "concurrent-end",
                    3: "full"}

FULL_REASON_NAMES = {
    "degen-start": "并发周期启动时老年代已越过 panic 线",
    "degen-keepup": "并发标记跟不上晋升/分配，投影占用将越过 panic 线",
    "degen-panic": "并发标记过程中老年代越过 panic 线",
    "promotion-failure": "晋升失败：老年代无可用空间",
    "heap-exhausted": "整堆耗尽",
    "system": "显式/系统触发",
    "old-trigger": "全停顿收集器到达触发阈值",
}


# --------------------------------------------------------------------------
# Weighted object records
# --------------------------------------------------------------------------
class ObjectStore:
    """A set of weighted object records in structure-of-arrays form."""

    __slots__ = ("death", "size", "age", "cls", "w")

    def __init__(self) -> None:
        self.death = np.empty(0, dtype=np.float64)
        self.size = np.empty(0, dtype=np.float64)
        self.age = np.empty(0, dtype=np.int16)
        self.cls = np.empty(0, dtype=np.int8)
        self.w = np.empty(0, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.death.shape[0])

    def append_store(self, other: "ObjectStore") -> None:
        if len(other) == 0:
            return
        if len(self) == 0:
            self.death = other.death.copy()
            self.size = other.size.copy()
            self.age = other.age.copy()
            self.cls = other.cls.copy()
            self.w = other.w.copy()
            return
        self.death = np.concatenate((self.death, other.death))
        self.size = np.concatenate((self.size, other.size))
        self.age = np.concatenate((self.age, other.age))
        self.cls = np.concatenate((self.cls, other.cls))
        self.w = np.concatenate((self.w, other.w))

    def take(self, mask: np.ndarray) -> "ObjectStore":
        out = ObjectStore()
        out.death = self.death[mask]
        out.size = self.size[mask]
        out.age = self.age[mask]
        out.cls = self.cls[mask]
        out.w = self.w[mask]
        return out

    def subset(self, idx: np.ndarray) -> "ObjectStore":
        out = ObjectStore()
        out.death = self.death[idx]
        out.size = self.size[idx]
        out.age = self.age[idx]
        out.cls = self.cls[idx]
        out.w = self.w[idx]
        return out

    def total_bytes(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(np.sum(self.w * self.size))

    def alive_bytes(self, t: float) -> float:
        if len(self) == 0:
            return 0.0
        m = self.death > t
        return float(np.sum(self.w[m] * self.size[m]))

    def alive_count(self, t: float) -> float:
        if len(self) == 0:
            return 0.0
        return float(np.sum(self.w[self.death > t]))


@dataclass
class Event:
    t: float
    kind: int
    pause_s: float = 0.0
    eden_before: float = 0.0
    eden_after: float = 0.0
    surv_before: float = 0.0
    surv_after: float = 0.0
    old_before: float = 0.0
    old_after: float = 0.0
    copied_bytes: float = 0.0
    promoted_bytes: float = 0.0
    live_bytes: float = 0.0
    tenuring: int = 0
    reason: str = ""
    mark_rate: float = 0.0
    growth_rate: float = 0.0
    keepup_ratio: float = 0.0


@dataclass
class SimulationResult:
    scenario: dict
    gc: dict
    workload: dict
    population: dict
    sim: dict
    events: List[dict] = field(default_factory=list)
    minutes: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    pause_hist: dict = field(default_factory=dict)
    delay_hist: dict = field(default_factory=dict)
    age_hist: dict = field(default_factory=dict)
    class_stats: dict = field(default_factory=dict)
    tuning: Optional[dict] = None
    calibration: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "gc": self.gc,
            "workload": self.workload, "population": self.population,
            "sim": self.sim, "events": self.events,
            "minutes": self.minutes, "summary": self.summary,
            "pause_hist": self.pause_hist, "delay_hist": self.delay_hist,
            "age_hist": self.age_hist, "class_stats": self.class_stats,
            "tuning": self.tuning, "calibration": self.calibration,
        }


class Simulator:
    def __init__(self, scenario: ServiceScenario) -> None:
        scenario.validate()
        self.sc = scenario
        self.gc = scenario.gc
        self.wl = scenario.workload
        self.pop = scenario.population
        self.sim = scenario.sim
        self.rng = np.random.default_rng(self.sim.seed)
        self.wall_start = time.time()

        # ---- population constants -------------------------------------
        self.class_frac = np.asarray(self.pop.class_mix(), dtype=np.float64)
        self.class_size = np.asarray(
            [self.pop.request_size_bytes, self.pop.session_size_bytes,
             self.pop.cache_size_bytes], dtype=np.float64)
        self.class_scale = np.asarray(
            [self.pop.request_scale_s, self.pop.session_scale_s,
             self.pop.cache_scale_s], dtype=np.float64)
        self.class_shape = np.asarray(
            [self.pop.request_shape, self.pop.session_shape,
             self.pop.cache_shape], dtype=np.float64)
        bytes_per_class = self.class_frac * self.class_size
        self.bytes_share = bytes_per_class / bytes_per_class.sum()
        self.mean_size = float(bytes_per_class.sum())
        cv = np.asarray([0.70, 0.55, 0.45], dtype=np.float64)
        self.class_size_mu = np.log(self.class_size) - 0.5 * np.log1p(cv ** 2)
        self.class_size_sigma = np.sqrt(np.log1p(cv ** 2))

        # ---- heap state -----------------------------------------------
        self.now = 0.0
        self.eden_used = 0.0
        self.surv_used = 0.0
        self.old_used = 0.0
        self.old_floating = 0.0
        self.allocation_scale = 1.0
        self.young = ObjectStore()
        self.old = ObjectStore()
        self.tenuring = int(self.gc.tenuring_threshold)
        self.burst_x = 0.0

        # concurrent cycle
        self.cycle_active = False
        self.cycle_t0 = 0.0
        self.cycle_work = 0.0
        self.cycle_debt = 0.0
        self.cycle_old0 = 0.0
        self.cycle_promoted = 0.0
        self.last_minor_promoted = 0.0
        self.last_minor_dt = 1.0
        self.promo_rate_ewma = 0.0
        self.promo_rate_tau = 30.0

        # accumulators
        self.gc_id = 0
        self.events: List[Event] = []
        self.record_events = True
        self.fast_mode = False
        self.minutes: Dict[int, dict] = {}
        self.pauses_all: List[float] = []
        self.minor_pauses: List[float] = []
        self.full_pauses: List[float] = []
        self.conc_pauses: List[float] = []
        self.total_paused_s = 0.0
        self.concurrent_cpu_s = 0.0
        self.minor_count = 0
        self.full_count = 0
        self.concurrent_cycles = 0
        self.degen_count = 0
        self.full_reason_counts: Dict[str, int] = {}
        self.oom_events = 0
        self.heap_exhausted_events = 0
        self.premature_promotions = 0.0     # weighted object count
        self.premature_promoted_bytes = 0.0
        self.promotions = 0.0               # weighted object count
        self.promoted_bytes = 0.0
        self.promoted_bytes_by_class = np.zeros(3)
        self.promoted_objects_by_class = np.zeros(3)
        self.promoted_age_bytes = np.zeros(17)
        self.reclaimed_old = 0.0
        self.sweep_count = 0
        self.age_bytes_sum = np.zeros(17)
        self.age_samples = 0
        self.eden_live_sum = 0.0
        self.max_pause_s = 0.0
        self.max_pause_t = 0.0
        self.max_pause_kind = -1
        self.delay_hist = np.zeros(160, dtype=np.float64)   # 0..800ms, 5ms bins
        self.delay_bin_s = 0.005
        self.total_requests = 0.0
        self.over_budget_requests = 0.0
        self.alloc_start = 0.0
        self._total_alloc_bytes = 0.0
        self.sampled_records = 0
        self.max_record_weight = 1.0
        self._record_cap = int(self.sim.max_records_per_minor)
        self._total_promotion_work = 0.0
        self._post_warmup_alloc_bytes = 0.0
        self._minor_times: List[float] = []
        self._loops = 0
        self._eden_fill_bytes = 0.0
        self._last_prune_t = -1e9
        self._peak_old = 0.0
        self._peak_live = 0.0

    # ------------------------------------------------------------------
    # Minute buckets: allocation, pauses, occupancy integrals
    # ------------------------------------------------------------------
    def _bucket(self, t: float) -> dict:
        m = int(t // 60.0)
        b = self.minutes.get(m)
        if b is None:
            b = {
                "m": m, "alloc_bytes": 0.0, "promoted_bytes": 0.0,
                "minor_count": 0, "full_count": 0, "conc_cycles": 0,
                "degen": 0, "pause_ms": 0.0, "max_pause_ms": 0.0,
                "eden_area": 0.0, "surv_area": 0.0, "old_area": 0.0,
                "eden_max": 0.0, "old_max": 0.0, "eden": 0.0,
                "survivor": 0.0, "old": 0.0, "heap": 0.0, "live": 0.0,
                "alloc_rate_mibs": 0.0, "req_rate": 0.0,
                "promoted_mibs": 0.0, "full_pause_ms": 0.0,
                "tenuring": 0,
            }
            self.minutes[m] = b
        return b

    def _add_area(self, t0: float, t1: float, eden0: float, eden1: float,
                  surv: float, old: float) -> None:
        """Integrate occupancy over [t0,t1] into minute buckets."""
        if t1 <= t0:
            return
        if self.fast_mode:
            # minute-resolution occupancy is not needed during a search, but
            # the peak occupancy is: track it on the scalar state
            peak = max(old, old + surv)
            if peak > self._peak_live:
                self._peak_live = peak
            if old > self._peak_old:
                self._peak_old = old
            return
        span = t1 - t0
        t = t0
        while t < t1 - 1e-12:
            m_end = (math.floor(t / 60.0) + 1.0) * 60.0
            nxt = min(t1, m_end)
            f0 = (t - t0) / span
            f1 = (nxt - t0) / span
            e0 = eden0 + (eden1 - eden0) * f0
            e1 = eden0 + (eden1 - eden0) * f1
            b = self._bucket(t)
            b["eden_area"] += 0.5 * (e0 + e1) * (nxt - t)
            b["surv_area"] += surv * (nxt - t)
            b["old_area"] += old * (nxt - t)
            b["eden_max"] = max(b["eden_max"], max(e0, e1))
            b["old_max"] = max(b["old_max"], old)
            b["survivor"] = surv
            b["old"] = old
            b["eden"] = e1
            b["heap"] = e1 + surv + old
            b["live"] = surv + old
            b["tenuring"] = self.tenuring
            t = nxt

    def _add_alloc(self, t0: float, t1: float, total_bytes: float) -> None:
        if t1 <= t0 or total_bytes <= 0:
            return
        self._total_alloc_bytes += total_bytes
        if t0 >= self.sim.warmup_s:
            self._post_warmup_alloc_bytes += total_bytes
        elif t1 > self.sim.warmup_s:
            self._post_warmup_alloc_bytes += total_bytes * (
                (t1 - self.sim.warmup_s) / (t1 - t0))
        t = t0
        while t < t1 - 1e-12:
            m_end = (math.floor(t / 60.0) + 1.0) * 60.0
            nxt = min(t1, m_end)
            self._bucket(t)["alloc_bytes"] += total_bytes * (nxt - t) / (t1 - t0)
            t = nxt

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------
    def _burst_mult(self) -> float:
        cv = max(self.wl.burst_cv, 0.0)
        if cv <= 0:
            return 1.0
        sigma = math.sqrt(math.log1p(cv * cv))
        return float(math.exp(self.burst_x - 0.5 * sigma * sigma))

    def _step_burst(self, dt: float) -> None:
        cv = max(self.wl.burst_cv, 0.0)
        if cv <= 0 or dt <= 0:
            return
        sigma = math.sqrt(math.log1p(cv * cv))
        tau = max(self.wl.burst_timescale_s, 1e-6)
        theta = 1.0 - math.exp(-dt / tau)
        self.burst_x = (self.burst_x * (1.0 - theta)
                        + sigma * math.sqrt(2.0 * theta)
                        * float(self.rng.standard_normal()))

    def alloc_rate(self, t: float) -> float:
        return self.wl.alloc_rate_bytes_per_s(t) * self._burst_mult()

    def old_occupancy(self) -> float:
        """Real old-generation occupancy, including floating garbage."""
        return self.old_used

    def _request_rate(self, t: float) -> float:
        mibs = self.wl.alloc_rate_bytes_per_s(t) * self._burst_mult() / MiB
        return self.sim.requests_per_mib_rate * mibs

    # ------------------------------------------------------------------
    # Sampling of eden survivors
    # ------------------------------------------------------------------
    def _survival(self, age: np.ndarray, cls: int) -> np.ndarray:
        age = np.maximum(age, 0.0)
        return np.exp(-np.power(age / self.class_scale[cls],
                                self.class_shape[cls]))

    def _sample_birth(self, n: int, t0: float, t1: float, cls: int,
                      g: float) -> np.ndarray:
        span = t1 - t0
        if n <= 0:
            return np.empty(0)
        if span <= 1e-9:
            return np.full(n, t0)
        k = 8
        edges = np.linspace(t0, t1, k + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        w = self._survival(g - centers, cls) * (edges[1:] - edges[:-1])
        cdf = np.cumsum(w)
        total = cdf[-1]
        if not np.isfinite(total) or total <= 0:
            return self.rng.uniform(t0, t1, n)
        cdf = cdf / total
        u = self.rng.random(n)
        idx = np.minimum(np.searchsorted(cdf, u, side="right"), k - 1)
        return edges[idx] + self.rng.random(n) * (edges[idx + 1] - edges[idx])

    def _sample_death(self, birth: np.ndarray, cls: int, g: float) -> np.ndarray:
        age = np.maximum(g - birth, 0.0)
        scale = self.class_scale[cls]
        shape = self.class_shape[cls]
        u = self.rng.random(birth.shape[0])
        base = np.power(age / scale, shape) - np.log(np.maximum(u, 1e-300))
        total_age = scale * np.power(np.maximum(base, 0.0), 1.0 / shape)
        return birth + total_age

    def _sample_size(self, cls: int, n: int) -> np.ndarray:
        z = self.rng.standard_normal(n)
        return np.maximum(np.exp(self.class_size_mu[cls]
                                 + self.class_size_sigma[cls] * z), 32.0)

    def _eden_survivors(self, t0: float, g: float) -> ObjectStore:
        """Materialise survivors of the eden fill [t0,g] as weighted records.

        All three object classes are sampled in a handful of vectorised calls
        (this method runs once per young collection, so per-call overhead
        matters more than the number of records).
        """
        span = max(g - t0, 1e-9)
        total_bytes = self._eden_fill_bytes
        # --- expected live count per class, from the survival integral ----
        counts = []
        live = []
        for c in range(3):
            n_alloc = total_bytes * self.bytes_share[c] / self.class_size[c]
            surv_frac = self._mean_survival(t0, g, c)
            n_live = n_alloc * surv_frac
            live.append(n_live)
            n = 0
            if n_live > 1e-9:
                n = int(min(max(1, math.ceil(n_live)), self._record_cap))
            counts.append(n)
        total_n = sum(counts)
        if total_n == 0:
            return ObjectStore()
        cls = np.repeat(np.arange(3, dtype=np.int8), counts)
        n = total_n
        # --- birth times: inverse CDF over 8 equal segments --------------
        edges = np.empty(9)
        edges[0] = t0
        edges[8] = g
        step = span / 8.0
        for i in range(1, 8):
            edges[i] = t0 + i * step
        widths = np.empty(8)
        widths[0] = edges[1] - edges[0]
        for i in range(1, 8):
            widths[i] = edges[i + 1] - edges[i]
        centers = edges[:-1] + 0.5 * widths
        # survival for every (class, segment) pair in one call
        cc = np.repeat(np.arange(3, dtype=np.int8), 8)
        seg = np.tile(np.arange(8), 3)
        surv = self._survival(g - np.repeat(centers, 3),
                              cc) * np.repeat(widths, 3)
        surv = surv.reshape(3, 8)
        cdf = np.cumsum(surv, axis=1)
        seg_idx = np.empty(n, dtype=np.intp)
        for c in range(3):
            m = cls == c
            if not m.any():
                continue
            k = counts[c]
            tot = cdf[c, -1]
            if not np.isfinite(tot) or tot <= 0:
                seg_idx[m] = 0
                continue
            u = self.rng.random(k) * tot
            seg_idx[m] = np.searchsorted(cdf[c, :], u, side="right").clip(0, 7)
        u2 = self.rng.random(n)
        birth = edges[seg_idx] + u2 * widths[seg_idx]
        # --- death times, conditional on being alive at g -----------------
        age = np.maximum(g - birth, 0.0)
        scale = self.class_scale[cls]
        shape = self.class_shape[cls]
        u3 = self.rng.random(n)
        base = np.power(age / scale, shape) - np.log(np.maximum(u3, 1e-300))
        death = birth + scale * np.power(np.maximum(base, 0.0), 1.0 / shape)
        # --- sizes ---------------------------------------------------------
        z = self.rng.standard_normal(n)
        size = np.maximum(np.exp(self.class_size_mu[cls]
                                 + self.class_size_sigma[cls] * z), 32.0)
        weights = np.array([live[c] / counts[c] if counts[c] else 0.0
                            for c in range(3)])
        w = weights[cls]
        local_max = float(weights.max()) if n else 1.0
        if local_max > self.max_record_weight:
            self.max_record_weight = local_max
        self.sampled_records += n
        store = ObjectStore()
        store.death = death
        store.size = size
        store.age = np.ones(n, dtype=np.int16)
        store.cls = cls
        store.w = w
        return store

    def _mean_survival(self, t0: float, g: float, c: int) -> float:
        """Time-averaged survival probability of eden objects in [t0,g]."""
        span = max(g - t0, 1e-9)
        step = span / 8.0
        s = 0.0
        for i in range(8):
            mid = t0 + (i + 0.5) * step
            age = g - mid
            x = max(age, 0.0) / self.class_scale[c]
            s += math.exp(-(x ** self.class_shape[c]))
        return s / 8.0

    # ------------------------------------------------------------------
    # Pauses and latency model
    # ------------------------------------------------------------------
    def _record_pause(self, t: float, pause: float, kind: int) -> None:
        if pause <= 0:
            return
        if t < self.sim.warmup_s:
            return
        self.total_paused_s += pause
        self.pauses_all.append(pause)
        if kind == EV_FULL:
            self.full_pauses.append(pause)
        elif kind == EV_YOUNG:
            self.minor_pauses.append(pause)
        else:
            self.conc_pauses.append(pause)
        if pause > self.max_pause_s:
            self.max_pause_s = pause
            self.max_pause_t = t
            self.max_pause_kind = kind
        b = self._bucket(t)
        b["pause_ms"] += pause * 1000.0
        b["max_pause_ms"] = max(b["max_pause_ms"], pause * 1000.0)
        if kind != EV_YOUNG:
            b["full_pause_ms"] += pause * 1000.0
        # Requests arrive uniformly.  One caught in a pause of length L is
        # delayed by U(0,L); its count is request_rate * L.
        r = self._request_rate(t)
        n_bins = min(self.delay_hist.shape[0],
                     max(1, int(math.ceil(pause / self.delay_bin_s))))
        for i in range(n_bins):
            lo = i * self.delay_bin_s
            hi = min(pause, lo + self.delay_bin_s)
            if hi > lo:
                self.delay_hist[i] += r * (hi - lo)
        max_delay = self.delay_hist.shape[0] * self.delay_bin_s
        if pause > max_delay:
            self.delay_hist[-1] += r * (pause - max_delay)
        budget_s = self.sim.pause_budget_ms / 1000.0
        if pause > budget_s:
            self.over_budget_requests += r * (pause - budget_s)

    # ------------------------------------------------------------------
    # Young collection
    # ------------------------------------------------------------------
    def _young_gc(self, g: float, alloc_t0: float) -> float:
        gc = self.gc
        self.gc_id += 1
        counted = g >= self.sim.warmup_s
        eden_before = self.eden_used
        surv_before = self.surv_used
        old_before = self.old_used

        # 1. objects already in the survivor space age by one
        if len(self.young) > 0:
            kept = self.young.take(self.young.death > g)
            kept.age = (kept.age + 1).astype(np.int16)
        else:
            kept = ObjectStore()

        # 2. eden survivors (sampled, weighted)
        eden_live = self._eden_survivors(alloc_t0, g)
        eden_live_bytes = eden_live.total_bytes()
        if g >= self.sim.warmup_s:
            self.eden_live_sum += eden_live_bytes
        cand = ObjectStore()
        cand.append_store(kept)
        cand.append_store(eden_live)
        live_bytes = cand.total_bytes()

        # 3. tenuring: objects older than the threshold are promoted; if the
        #    survivor space still overflows, the oldest are promoted *early*
        #    (age < threshold) -- that is premature promotion.
        promoted = ObjectStore()
        premature_bytes = 0.0
        premature_count = 0.0
        if len(cand) > 0:
            over = cand.age > self.tenuring
            if over.any():
                promoted.append_store(cand.take(over))
                cand = cand.take(~over)
        cap = float(gc.survivor_bytes)
        if len(cand) > 0 and cand.total_bytes() > cap:
            order = np.argsort(-cand.age, kind="stable")
            cum = np.cumsum(cand.w[order] * cand.size[order])
            need = cand.total_bytes() - cap
            n_prom = min(int(np.searchsorted(cum, need, side="left")) + 1,
                         len(cand))
            prom_idx = order[:n_prom]
            mask = np.ones(len(cand), dtype=bool)
            mask[prom_idx] = False
            early = cand.subset(prom_idx)
            premature_bytes = early.total_bytes()
            premature_count = float(np.sum(early.w))
            promoted.append_store(early)
            cand = cand.subset(np.where(mask)[0])

        prom_bytes = promoted.total_bytes()
        if len(promoted):
            wcount = float(np.sum(promoted.w))
            prem = premature_count
            self.promotions += wcount
            self.premature_promotions += prem
            self.premature_promoted_bytes += premature_bytes
            self.promoted_bytes += prom_bytes if counted else 0.0
            b = self._bucket(g)
            b["promoted_bytes"] += prom_bytes
            if counted:
                for c in range(3):
                    m = promoted.cls == c
                    if m.any():
                        self.promoted_bytes_by_class[c] += float(
                            np.sum(promoted.w[m] * promoted.size[m]))
                        self.promoted_objects_by_class[c] += float(
                            np.sum(promoted.w[m]))
                self.promoted_age_bytes += _age_hist(promoted, 17)

        # 4. commit young/survivor state
        self.young = cand
        self.surv_used = cand.total_bytes()
        self.eden_used = 0.0
        if self.sim.track_ages and len(cand):
            self.age_bytes_sum += _age_hist(cand, 17)
            self.age_samples += 1

        # 5. pause proportional to live objects copied
        n_copied = float(np.sum(cand.w) + np.sum(promoted.w))
        pause = (gc.minor_fixed_s
                 + gc.minor_per_live_byte_s * live_bytes
                 + gc.minor_per_live_object_s * n_copied)
        self._record_pause(g, pause, EV_YOUNG)
        self.minor_count += 1 if counted else 0

        # 6. old generation receives promoted objects
        if len(promoted):
            self.old.append_store(promoted)
            self.old_used += prom_bytes
        self._minor_times.append(g)
        dt = g - self._minor_times[-2] if len(self._minor_times) > 1 else 1.0
        self.last_minor_promoted = prom_bytes
        self.last_minor_dt = max(dt, 1e-6)
        inst = prom_bytes / self.last_minor_dt
        alpha = 1.0 - math.exp(-self.last_minor_dt / self.promo_rate_tau)
        self.promo_rate_ewma += alpha * (inst - self.promo_rate_ewma)
        self._total_promotion_work += prom_bytes

        if self.record_events:
            self.events.append(Event(t=g, kind=EV_YOUNG, pause_s=pause,
                   eden_before=eden_before, eden_after=0.0,
                   surv_before=surv_before, surv_after=self.surv_used,
                   old_before=old_before, old_after=self.old_used,
                   copied_bytes=live_bytes, promoted_bytes=prom_bytes,
                       live_bytes=eden_live_bytes, tenuring=self.tenuring))
        b = self._bucket(g)
        b["minor_count"] += 1

        # 7. adaptive tenuring: aim survivor occupancy at the target.  The
        #    threshold never drops below the configured value; overflow is
        #    handled by early promotion instead (that is the pathology we
        #    want to expose, not hide).
        if gc.adaptive_tenuring and self.surv_used > 0:
            target = gc.target_survivor_ratio * gc.survivor_bytes
            if self.surv_used < 0.85 * target:
                self.tenuring = min(gc.max_tenuring_threshold, self.tenuring + 1)
            elif self.surv_used > 0.98 * gc.survivor_bytes:
                self.tenuring = max(gc.tenuring_threshold, self.tenuring - 1)
        return pause

    # ------------------------------------------------------------------
    # Old generation
    # ------------------------------------------------------------------
    def _sweep_old(self, now: float) -> None:
        self.reclaimed_old += self.old_floating
        self.old_floating = 0.0
        if len(self.old) == 0:
            self.old_used = 0.0
            return
        alive = self.old.death > now
        self.reclaimed_old += float(
            np.sum(self.old.w[~alive] * self.old.size[~alive]))
        self.old = self.old.take(alive)
        self.old_used = self.old.total_bytes()
        self.sweep_count += 1

    def _drop_dead_old_records(self, now: float, force: bool = False) -> None:
        """Drop records that are already dead from the sampled table.

        Floating garbage still counts towards ``old_used`` -- the collector
        cannot reclaim it until the next old-generation cycle -- so this only
        bounds *memory*: the freed bytes are kept in ``self.old_floating``
        and charged to occupancy until the next sweep.
        """
        if len(self.old) < 4 * self._record_cap:
            return
        if not force and now - self._last_prune_t < 30.0:
            return
        alive = self.old.death > now
        if alive.all():
            self._last_prune_t = now
            return
        dead = len(self.old) - int(alive.sum())
        if not force and dead < 0.5 * len(self.old):
            self._last_prune_t = now
            return
        dead_bytes = float(
            np.sum(self.old.w[~alive] * self.old.size[~alive]))
        self.old = self.old.take(alive)
        self.old_floating += dead_bytes
        self._last_prune_t = now

    def _cycle_projection(self, now: float) -> Tuple[float, float, float]:
        """Projected old occupancy at cycle completion and the keep-up ratio.

        The marker must scan ``work`` bytes at ``mark_rate``.  While it does
        so, the application keeps promoting at the recent promotion rate and
        every promotion adds write-barrier debt (``barrier_slope``).  The
        cycle succeeds only if the projected occupancy stays below the panic
        line; otherwise it degenerates into a full GC.
        """
        mark = max(self.effective_mark_rate(), 1.0)
        growth = self.promo_rate_ewma
        work = self.cycle_work + self.cycle_debt
        remaining = max(work - mark * (now - self.cycle_t0), 0.0)
        if remaining <= 0:
            return self.old_used, 0.0, 0.0
        # effective completion time: marking plus the barrier debt added by
        # the promotions occurring meanwhile
        net = mark - growth * self.gc.concurrent_barrier_slope
        if net <= 1e-9:
            return float("inf"), growth / mark, float("inf")
        finish = remaining / net
        projected = self.old_used + growth * finish
        return projected, growth / mark, finish

    def effective_mark_rate(self) -> float:
        """Mark rate actually achievable: the configured rate, capped by the
        number of marker threads the service will give up (4 cores)."""
        core = self.gc.concurrent_core_scan_bytes_per_s
        return min(self.gc.concurrent_mark_bytes_per_s, 4.0 * core)

    def _concurrent_service(self, now: float) -> Optional[float]:
        """Advance a running cycle at ``now``.

        Returns the pause it consumed (completion remark, or a full GC), or
        ``None`` if the cycle is still running.
        """
        if not self.cycle_active:
            return None
        panic = self.gc.concurrent_panic_ratio * self.gc.old_bytes
        projected, keepup, finish = self._cycle_projection(now)
        if self.old_used >= panic:
            self.cycle_active = False
            return self._full_gc(now, "degen-panic")
        if projected == float("inf") or projected >= panic:
            self.cycle_active = False
            return self._full_gc(now, "degen-keepup", keepup=keepup,
                                 growth=self.promo_rate_ewma)
        mark = self.effective_mark_rate()
        work = self.cycle_work + self.cycle_debt
        if mark * (now - self.cycle_t0) >= work:
            return self._finish_concurrent(now, keepup)
        return None

    def _finish_concurrent(self, now: float, keepup: float) -> float:
        elapsed = max(now - self.cycle_t0, 0.0)
        # Remark/remap drains the dirty-card / SATB residue left by mutation
        # during the cycle; concurrent refinement keeps it proportional to
        # the young generation rather than to the whole old heap.
        pause = (self.gc.concurrent_pause_remark_s
                 + self.gc.concurrent_remark_per_mib_s
                 * (self.gc.young_bytes() / MiB))
        self._record_pause(now, pause, EV_CONC_END)
        counted = now >= self.sim.warmup_s
        # The marker occupies CPU for the whole cycle, not just at the
        # remark: cost = elapsed * (mark_rate / core_scan_rate).
        threads = (self.effective_mark_rate()
                   / max(self.gc.concurrent_core_scan_bytes_per_s, 1.0))
        threads = min(threads, 4.0)
        self.concurrent_cpu_s += elapsed * threads / max(self.gc.service_cores, 1)
        self.concurrent_cycles += 1 if counted else 0
        old_before = self.old_used
        self._sweep_old(now)
        ev = Event(t=now, kind=EV_CONC_END, pause_s=pause,
                   old_before=old_before, old_after=self.old_used,
                   reason="concurrent-cycle", keepup_ratio=keepup)
        if self.record_events:
            self.events.append(ev)
        if counted:
            self._bucket(now)["conc_cycles"] += 1
        self.cycle_active = False
        return pause

    def _start_concurrent(self, now: float) -> float:
        if self.old_used >= self.gc.concurrent_panic_ratio * self.gc.old_bytes:
            return self._full_gc(now, "degen-start")
        self.gc_id += 1
        self.cycle_active = True
        self.cycle_t0 = now
        self.cycle_old0 = self.old_used
        self.cycle_debt = 0.0
        self.cycle_promoted = 0.0
        self.cycle_work = self.old.alive_bytes(now)
        pause = self.gc.concurrent_pause_initial_s
        self._record_pause(now, pause, EV_CONC_START)
        ev = Event(t=now, kind=EV_CONC_START, pause_s=pause,
                   old_before=self.old_used, old_after=self.old_used,
                   reason="initial-mark",
                   mark_rate=self.gc.concurrent_mark_bytes_per_s)
        if self.record_events:
            self.events.append(ev)
        return pause

    def _full_gc(self, now: float, reason: str, keepup: float = 0.0,
                 growth: float = 0.0) -> float:
        gc = self.gc
        self.gc_id += 1
        counted = now >= self.sim.warmup_s
        eden_before = self.eden_used
        surv_before = self.surv_used
        old_before = self.old_used
        live_young = ObjectStore()
        if len(self.young) > 0:
            live_young = self.young.take(self.young.death > now)
        live_bytes = self.old.alive_bytes(now) + live_young.total_bytes()
        live_objects = self.old.alive_count(now) + float(np.sum(live_young.w))
        pause = (gc.full_fixed_s
                 + gc.full_per_live_byte_s * live_bytes
                 + gc.full_per_live_object_s * live_objects)
        self._record_pause(now, pause, EV_FULL)
        counted = now >= self.sim.warmup_s
        self.old = self.old.take(self.old.death > now)
        self.old.append_store(live_young)
        self.old_used = self.old.total_bytes()
        self.old_floating = 0.0
        self.young = ObjectStore()
        self.surv_used = 0.0
        self.eden_used = 0.0
        self.cycle_active = False
        if counted:
            self.full_count += 1
            self.full_reason_counts[reason] = self.full_reason_counts.get(reason, 0) + 1
            if reason.startswith("degen"):
                self.degen_count += 1
        if self.old_used > gc.old_bytes:
            self.oom_events += 1
        ev = Event(t=now, kind=EV_FULL, pause_s=pause if counted else 0.0,
                   eden_before=eden_before, eden_after=0.0,
                   surv_before=surv_before, surv_after=0.0,
                   old_before=old_before, old_after=self.old_used,
                   live_bytes=live_bytes, reason=reason,
                   keepup_ratio=keepup, growth_rate=growth,
                   mark_rate=self.gc.concurrent_mark_bytes_per_s)
        if self.record_events:
            self.events.append(ev)
        b = self._bucket(now)
        if counted:
            b["full_count"] += 1
            if reason.startswith("degen"):
                b["degen"] += 1
        return pause

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        if self.fast_mode:
            self.record_events = False
            self.sim.track_ages = False
        gc = self.gc
        day = float(self.wl.day_seconds)
        self.now = 0.0
        self.eden_used = 0.0
        self.alloc_start = 0.0

        while self.now < day:
            if self._loops > 200_000_000:
                raise RuntimeError("simulation made no progress")
            self._loops += 1
            rate = max(self.alloc_rate(self.now), 1.0)
            room = max(gc.eden_bytes - self.eden_used, 0.0)
            t_eden = self.now + room / rate if room > 1e-9 else self.now

            # when can a running concurrent cycle finish if nothing changes?
            t_conc = float("inf")
            if self.cycle_active:
                mark = max(self.effective_mark_rate(), 1.0)
                work = self.cycle_work + self.cycle_debt
                t_conc = self.cycle_t0 + work / mark

            t_next = min(t_eden, t_conc, day)
            if t_next <= self.now + 1e-12:
                t_next = min(day, self.now + 1e-6)
            dt = t_next - self.now
            self._step_burst(dt)

            if t_next >= day:
                add = min(rate * dt, max(gc.eden_bytes - self.eden_used, 0.0))
                self._add_alloc(self.now, day, add)
                self._add_area(self.now, day, self.eden_used,
                               self.eden_used + add,
                               self.surv_used, self.old_used)
                self.eden_used = self.eden_used + add
                self.now = day
                break

            add = min(rate * dt, max(gc.eden_bytes - self.eden_used, 0.0))
            self._add_alloc(self.now, t_next, add)
            self._add_area(self.now, t_next, self.eden_used,
                           self.eden_used + add, self.surv_used, self.old_used)
            self.eden_used += add
            self.now = t_next

            if t_conc <= t_eden:
                # concurrent cycle completes before eden fills
                pause = self._concurrent_service(self.now)
                if pause:
                    self.now += pause
                    self.alloc_start = self.now
                continue

            # ---- eden is full: young collection ----------------------
            self.eden_used = gc.eden_bytes
            self._eden_fill_bytes = gc.eden_bytes
            pause = self._young_gc(self.now, self.alloc_start)
            self.now += pause
            self._drop_dead_old_records(self.now)
            self.eden_used = 0.0
            # concurrent cycle sees promotions & advances
            if self.cycle_active:
                self.cycle_debt += self.last_minor_promoted * gc.concurrent_barrier_slope
                self.cycle_promoted += self.last_minor_promoted
                extra = self._concurrent_service(self.now)
                if extra:
                    self.now += extra
            # ---- old generation policy -------------------------------
            if self.old_used > gc.heap_bytes():
                self.heap_exhausted_events += 1
                self.now += self._full_gc(self.now, "heap-exhausted")
            elif self.old_used > gc.old_bytes:
                self.now += self._full_gc(self.now, "promotion-failure")
            elif not self.cycle_active:
                if self.old_used >= gc.concurrent_trigger_ratio * gc.old_bytes:
                    if gc.concurrent_enabled:
                        self.now += self._start_concurrent(self.now)
                    else:
                        # stop-the-world collector: no concurrent marking, so
                        # the trigger line directly schedules a full GC
                        self.now += self._full_gc(self.now, "old-trigger")
            # the next eden fill starts now, after every pause of this
            # collection (minor, remark, full).  Using an earlier start would
            # stretch the fill interval and bias the survivor sampling.
            self.alloc_start = self.now
            if self.now >= day:
                break

        self._finalize()
        return self._build_result()

    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        if not self.minutes:
            return
        last = max(self.minutes)
        prev = None
        base = {"m": 0, "alloc_bytes": 0.0, "promoted_bytes": 0.0,
                "minor_count": 0, "full_count": 0, "conc_cycles": 0,
                "degen": 0, "pause_ms": 0.0, "max_pause_ms": 0.0,
                "eden_area": 0.0, "surv_area": 0.0, "old_area": 0.0,
                "eden_max": 0.0, "old_max": 0.0, "eden": 0.0, "survivor": 0.0,
                "old": 0.0, "heap": 0.0, "live": 0.0, "alloc_rate_mibs": 0.0,
                "req_rate": 0.0, "promoted_mibs": 0.0, "full_pause_ms": 0.0,
                "tenuring": 0}
        for m in range(last + 1):
            b = self.minutes.get(m)
            if b is None:
                b = dict(base)
                b["m"] = m
                if prev is not None:
                    for k in ("eden", "survivor", "old", "heap", "live",
                              "tenuring"):
                        b[k] = prev[k]
                self.minutes[m] = b
            prev = b
        self.total_requests = (self._post_warmup_alloc_bytes / MiB) * self.sim.requests_per_mib_rate
        for b in self.minutes.values():
            secs = 60.0
            b["eden"] = b["eden_area"] / secs
            b["survivor"] = b["surv_area"] / secs
            b["old"] = b["old_area"] / secs
            b["live"] = b["old"] + b["survivor"]
            b["heap"] = b["old"] + b["survivor"] + b["eden"]
            b["alloc_rate_mibs"] = b["alloc_bytes"] / MiB / secs
            b["req_rate"] = b["alloc_rate_mibs"] * self.sim.requests_per_mib_rate
            b["promoted_mibs"] = b["promoted_bytes"] / MiB / secs
            b["eden_area"] = b["eden_area"] / secs
            b["surv_area"] = b["surv_area"] / secs
            b["old_area"] = b["old_area"] / secs

    def _build_fast_result(self) -> SimulationResult:
        return SimulationResult(
            scenario={"name": self.sc.name,
                      "description": self.sc.description},
            gc=self.gc.to_dict(), workload=self.wl.to_dict(),
            population=self.pop.to_dict(), sim=self.sim.to_dict(),
            events=[], minutes=[self.minutes[m] for m in sorted(self.minutes)],
            summary=self.fast_summary())

    def fast_summary(self) -> dict:
        """Summary for search runs: skips the per-event and histogram work."""
        day = max(float(self.wl.day_seconds) - float(self.sim.warmup_s), 1e-9)
        pauses = self.pauses_all
        loss = (self.total_paused_s + self.concurrent_cpu_s) / day
        tail = (self.over_budget_requests / self.total_requests
                if self.total_requests > 0 else 0.0)
        return {
            "pause_max_ms": self.max_pause_s * 1000.0,
            "pause_p99_ms": self._pct(pauses, 99),
            "pause_p95_ms": self._pct(pauses, 95),
            "pause_p50_ms": self._pct(pauses, 50),
            "mean_added_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "tail_over_budget_frac": float(tail),
            "throughput_loss_frac": float(loss),
            "stw_pause_frac": float(self.total_paused_s / day),
            "concurrent_cpu_frac": float(self.concurrent_cpu_s / day),
            "full_count": self.full_count,
            "degenerate_count": self.degen_count,
            "concurrent_cycles": self.concurrent_cycles,
            "minor_count": self.minor_count,
            "promoted_bytes_total": self.promoted_bytes,
            "promoted_mibs_per_s": (
                self.promoted_bytes / MiB
                / max(float(self.wl.day_seconds) - float(self.sim.warmup_s), 1e-9)),
            "peak_old_frac": self._peak_old / max(self.gc.old_bytes, 1),
            "allocated_bytes_total": self._post_warmup_alloc_bytes,
            "heap_bytes": self.gc.heap_bytes(),
            "day_seconds": float(self.wl.day_seconds),
            "oom_events": self.oom_events,
            "budget_breaches": len(pauses),
        }

    def _pct(self, arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        return float(np.percentile(np.asarray(arr, dtype=np.float64) * 1000.0, q))

    def _build_result(self) -> SimulationResult:
        if self.fast_mode:
            return self._build_fast_result()
        day = float(self.wl.day_seconds) - float(self.sim.warmup_s)
        pauses = self.pauses_all
        total_req = max(self.total_requests, 1.0)
        budget_s = self.sim.pause_budget_ms / 1000.0
        base_latency_ms = 1.5
        cum = np.cumsum(self.delay_hist)
        delayed = float(cum[-1]) if len(cum) else 0.0

        def delay_pct(frac: float) -> float:
            if delayed <= 0:
                return 0.0
            idx = int(np.searchsorted(cum, frac * total_req))
            idx = min(idx, len(cum) - 1)
            return (idx + 0.5) * self.delay_bin_s

        mean_delay = 0.0
        if delayed > 0:
            vals = (np.arange(len(self.delay_hist)) + 0.5) * self.delay_bin_s
            mean_delay = float(np.sum(self.delay_hist * vals) / delayed)
        stw = self.total_paused_s / day
        day = max(day, 1e-9)
        throughput_loss = (self.total_paused_s + self.concurrent_cpu_s) / day
        breaches = int(sum(1 for x in pauses if x > budget_s))
        minor_live_mean = self.eden_live_sum / self.minor_count

        peak_old = max((b["old_max"] for b in self.minutes.values()), default=0.0)
        peak_heap = max((b["heap"] for b in self.minutes.values()), default=0.0)
        summary = {
            "day_seconds": float(self.wl.day_seconds),
            "measured_seconds": day,
            "minor_count": self.minor_count,
            "minor_interval_mean_s": (day / self.minor_count
                                      if self.minor_count else 0.0),
            "concurrent_cycles": self.concurrent_cycles,
            "full_count": self.full_count,
            "degenerate_count": self.degen_count,
            "full_reason_counts": dict(self.full_reason_counts),
            "full_reason_labels": {
                FULL_REASON_NAMES.get(k, k): v
                for k, v in self.full_reason_counts.items()},
            "oom_events": self.oom_events,
            "heap_exhausted_events": self.heap_exhausted_events,
            "premature_promotions": self.premature_promotions,
            "premature_promotion_frac": (self.premature_promotions
                                         / self.promotions
                                         if self.promotions else 0.0),
            "premature_promoted_bytes": self.premature_promoted_bytes,
            "premature_promoted_bytes_frac": (
                self.premature_promoted_bytes / self.promoted_bytes
                if self.promoted_bytes else 0.0),
            "promotions": self.promotions,
            "pause_count": len(pauses),
            "pause_mean_ms": (float(np.mean(pauses)) * 1000.0 if pauses else 0.0),
            "pause_p50_ms": self._pct(pauses, 50),
            "pause_p95_ms": self._pct(pauses, 95),
            "pause_p99_ms": self._pct(pauses, 99),
            "pause_p999_ms": self._pct(pauses, 99.9),
            "pause_max_ms": self.max_pause_s * 1000.0,
            "pause_max_t": self.max_pause_t,
            "pause_max_kind": EVENT_KIND_NAMES.get(self.max_pause_kind, ""),
            "minor_pause_p50_ms": self._pct(self.minor_pauses, 50),
            "minor_pause_p99_ms": self._pct(self.minor_pauses, 99),
            "minor_pause_max_ms": self._pct(self.minor_pauses, 100),
            "full_pause_p50_ms": self._pct(self.full_pauses, 50),
            "full_pause_max_ms": self._pct(self.full_pauses, 100),
            "budget_ms": self.sim.pause_budget_ms,
            "budget_breaches": breaches,
            "total_requests": total_req,
            "mean_added_latency_ms": mean_delay * 1000.0,
            "tail_over_budget_frac": float(self.over_budget_requests / total_req),
            "p99_latency_ms": base_latency_ms + delay_pct(0.99) * 1000.0,
            "p999_latency_ms": base_latency_ms + delay_pct(0.999) * 1000.0,
            "max_latency_ms": base_latency_ms + self.max_pause_s * 1000.0,
            "throughput_loss_frac": float(throughput_loss),
            "stw_pause_frac": float(stw),
            "concurrent_cpu_frac": float(self.concurrent_cpu_s / day),
            "throughput_target": self.sim.throughput_target,
            "throughput_ok": bool(throughput_loss <= 1.0 - self.sim.throughput_target),
            "target_loss_frac": float(1.0 - self.sim.throughput_target),
            "promoted_bytes_total": self.promoted_bytes,
            "promoted_mibs_per_s": (
                self.promoted_bytes / MiB
                / max(float(self.wl.day_seconds) - float(self.sim.warmup_s), 1e-9)),
            "promoted_mibs_per_s": self.promoted_bytes / MiB / day,
            "promoted_bytes_by_class": {
                POP_NAMES[c]: float(self.promoted_bytes_by_class[c])
                for c in range(3)},
            "promoted_share_by_class": {
                POP_NAMES[c]: float(self.promoted_bytes_by_class[c]
                                    / max(self.promoted_bytes, 1.0))
                for c in range(3)},
            "allocated_bytes_total": self._total_alloc_bytes,
            "reclaimed_old_gib": self.reclaimed_old / (1024 ** 3),
            "sweeps": self.sweep_count,
            "final_old_bytes": self.old_used,
            "final_old_frac": self.old_used / self.gc.old_bytes,
            "peak_old_bytes": peak_old,
            "peak_old_frac": peak_old / self.gc.old_bytes,
            "peak_heap_bytes": peak_heap,
            "minor_live_bytes_mean": minor_live_mean,
            "sampled_records": self.sampled_records,
            "max_record_weight": self.max_record_weight,
            "mean_object_size": self.mean_size,
            "heap_bytes": self.gc.heap_bytes(),
            "young_bytes": self.gc.young_bytes(),
            "wall_time_s": time.time() - self.wall_start,
        }

        edges = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500,
                 1000, 1e9]
        hist = {f"{_fmt(edges[i])}-{_fmt(edges[i+1])}": 0
                for i in range(len(edges) - 1)}
        for x in pauses:
            ms = x * 1000.0
            for i in range(len(edges) - 1):
                if edges[i] <= ms < edges[i + 1]:
                    hist[f"{_fmt(edges[i])}-{_fmt(edges[i+1])}"] += 1
                    break

        age_hist: dict = {}
        if self.age_samples:
            avg = self.age_bytes_sum / self.age_samples
            age_hist = {
                "age": list(range(len(avg))),
                "bytes": [float(x) for x in avg],
                "promoted_bytes": [float(x) for x in self.promoted_age_bytes],
            }
        class_stats = {
            "mix_by_count": {POP_NAMES[i]: float(self.class_frac[i])
                             for i in range(3)},
            "mix_by_bytes": {POP_NAMES[i]: float(self.bytes_share[i])
                             for i in range(3)},
            "mean_size": {POP_NAMES[i]: float(self.class_size[i])
                          for i in range(3)},
            "median_lifetime_s": {
                POP_NAMES[i]: float(self.pop.median_lifetime(i))
                for i in range(3)},
            "promoted_bytes": {
                POP_NAMES[i]: float(self.promoted_bytes_by_class[i])
                for i in range(3)},
            "promoted_objects": {
                POP_NAMES[i]: float(self.promoted_objects_by_class[i])
                for i in range(3)},
        }
        events = [{
            "t": e.t, "kind": e.kind, "kind_name": EVENT_KIND_NAMES[e.kind],
            "pause_ms": e.pause_s * 1000.0,
            "eden_before": e.eden_before, "eden_after": e.eden_after,
            "surv_before": e.surv_before, "surv_after": e.surv_after,
            "old_before": e.old_before, "old_after": e.old_after,
            "copied_bytes": e.copied_bytes,
            "promoted_bytes": e.promoted_bytes,
            "live_bytes": e.live_bytes, "tenuring": e.tenuring,
            "reason": e.reason, "mark_rate": e.mark_rate,
            "growth_rate": e.growth_rate,
            "keepup_ratio": e.keepup_ratio,
        } for e in self.events]

        minutes = [self.minutes[m] for m in sorted(self.minutes)]
        return SimulationResult(
            scenario={"name": self.sc.name,
                      "description": self.sc.description},
            gc=self.gc.to_dict(),
            workload=self.wl.to_dict(),
            population=self.pop.to_dict(),
            sim=self.sim.to_dict(),
            events=events, minutes=minutes, summary=summary,
            pause_hist=hist,
            delay_hist={"bin_s": self.delay_bin_s,
                        "counts": [float(x) for x in self.delay_hist]},
            age_hist=age_hist, class_stats=class_stats,
        )


def _fmt(x: float) -> str:
    return "inf" if x >= 1e8 else f"{x:g}"


def _age_hist(store: ObjectStore, nbins: int) -> np.ndarray:
    h = np.zeros(nbins, dtype=np.float64)
    if len(store) == 0:
        return h
    idx = np.clip(store.age.astype(np.int64), 0, nbins - 1)
    np.add.at(h, idx, store.w * store.size)
    return h
