"""Discrete object-level simulator: the validation oracle.

The cohort engine (:mod:`gcsim.simulator`) answers collector questions in
expectation.  This module answers them object by object, with the same
mechanism, at small scale.  It exists so that the assumptions behind the
cohort engine can be *checked against a faithful per-object implementation*
rather than asserted.

Scope: it allocates real objects with real sampled lifetimes, runs real
scavenges with real survivor spaces and per-object ages, and promotes/copies
exactly as a collector would.  It is deliberately simple and slow (pure Python),
suitable for thousands of objects and dozens of collections, not for a day.

Usage
-----
    cfg = GCConfig(); ...
    oracle = ObjectOracle(cfg, classes, seed)
    result = oracle.run_day(duration_s=..., rate_obj_per_s=..., size_range=...)
    # result.pause micros, promotion bytes, survivors per scavenge ...
    cohort = HeapSimulator(cfg, ...) with the same *expected* rate
    # compare promotion volume and pause totals

The class of lifetime models is shared, so a discrepancy indicates an error in
the cohort algebra, not a difference of definition.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import GCConfig
from .lifetime import LifetimeModel
from .rng import derive_seed


@dataclass
class _Obj:
    __slots__ = ("birth", "death", "size", "age", "cls")

    def __init__(self, birth, death, size, cls):
        self.birth = birth
        self.death = death
        self.size = size
        self.age = 0
        self.cls = cls


@dataclass
class OracleResult:
    pauses: List[float] = field(default_factory=list)
    pause_times: List[float] = field(default_factory=list)
    scavenges: int = 0
    promotions_bytes: float = 0.0
    promotions_objs: int = 0
    pretenured_bytes: float = 0.0
    tospace_exhaustion: int = 0
    promoted_ages: Dict[int, int] = field(default_factory=dict)
    copied_bytes: float = 0.0
    old_bytes_series: List[Tuple[float, float]] = field(default_factory=list)
    counted_objects: int = 0


class ObjectOracle:
    """Object-by-object generational collector with the same rules.

    ``classes`` is a list of ``(share, mean_size, LifetimeModel)``.  Objects
    are allocated at an exponential/Poisson rate; each carries a sampled death
    time.  Eden holds a list of live objects; a scavenge copies survivors into
    a fresh survivor list with age+1, promotes at the tenure age, and pre-tenures
    on space pressure exactly as the cohort engine's tenure rule.
    """

    def __init__(self, cfg: GCConfig, classes: Sequence[Tuple[float, float, LifetimeModel]],
                 seed: int = 1):
        self.cfg = cfg
        self.classes = list(classes)
        total = sum(c[0] for c in self.classes)
        self.shares = [c[0] / total for c in self.classes]
        self.mean_sizes = [c[1] for c in self.classes]
        self.models = [c[2] for c in self.classes]
        self.rng = random.Random(derive_seed(seed, "oracle"))
        self.class_rngs = [random.Random(derive_seed(seed, "oracle", i))
                           for i in range(len(self.classes))]
        self.eden: List[_Obj] = []
        self.eden_bytes = 0.0
        self.survivors: List[_Obj] = []
        self.survivor_bytes = 0.0
        self.old_objs: List[_Obj] = []
        self.old_bytes = 0.0
        self.old_committed = 0.0
        self.clock = 0.0
        self.result = OracleResult()
        self._tenure_age = cfg.young.promotion_age

    def _sample_class(self) -> int:
        x = self.rng.random()
        acc = 0.0
        for i, s in enumerate(self.shares):
            acc += s
            if x < acc:
                return i
        return len(self.shares) - 1

    def _new_object(self, t: float) -> _Obj:
        ci = self._sample_class()
        r = self.class_rngs[ci]
        size = max(8, int(round(self.mean_sizes[ci] *
                                math.exp(r.gauss(-0.5 * 0.35 ** 2, 0.35)))))
        life = self.models[ci].sample_lifetime(r, t)
        return _Obj(t, t + max(life, 1e-4), size, ci)

    def _retire(self, t: float) -> None:
        if self.eden:
            self.eden = [o for o in self.eden if o.death > t]
            self.eden_bytes = sum(o.size for o in self.eden)
        if self.survivors:
            self.survivors = [o for o in self.survivors if o.death > t]
            self.survivor_bytes = sum(o.size for o in self.survivors)
        if self.old_objs:
            n0 = len(self.old_objs)
            self.old_objs = [o for o in self.old_objs if o.death > t]
            if len(self.old_objs) != n0:
                self.old_bytes = sum(o.size for o in self.old_objs)
                self.old_committed = max(self.old_bytes, self.old_committed -
                                         (self.old_committed - self.old_bytes))

    def _scavenge_pause(self, copied_bytes, copied_objs, promoted_bytes) -> float:
        h = self.cfg.heap
        ns = (copied_bytes * h.copy_ns_per_byte * h.copy_graph_factor
              + copied_objs * h.copy_ns_per_object
              + promoted_bytes * h.copy_ns_per_byte * h.rs_fixup_frac)
        return 0.5e-3 + ns * 1e-9

    def scavenge(self, t: float) -> None:
        self._retire(t)
        eden_b = self.eden_bytes
        surv_b = self.survivor_bytes
        # candidates: eden objs (age 1) + survivor objs (age+1)
        cand = []
        for o in self.eden:
            cand.append((o, 1))
        for o in self.survivors:
            cand.append((o, o.age + 1))
        space = float(self.cfg.heap.survivor_space())
        # tenure age: highest that fits, capped at promotion_age
        cand.sort(key=lambda x: (x[1], -x[0].birth))
        by_age: Dict[int, int] = {}
        for o, a in cand:
            a2 = min(a, self.cfg.young.promotion_age)
            by_age[a2] = by_age.get(a2, 0) + o.size
        eff_age = self.cfg.young.promotion_age
        if self.cfg.young.adaptive_tenuring:
            run = 0
            for t_age in range(1, self.cfg.young.promotion_age + 1):
                run += by_age.get(t_age - 1, 0)
                if run > space:
                    eff_age = max(1, t_age - 1)
                    break
            if sum(o.size for o, _ in cand) <= space:
                eff_age = self.cfg.young.promotion_age
        promoted = []
        kept = []
        kept_bytes = 0
        for o, a in cand:
            if a >= eff_age:
                promoted.append((o, a))
            else:
                kept.append(o)
                kept_bytes += o.size
        exhausted = kept_bytes > space
        if exhausted:
            for o in kept:
                promoted.append((o, o.age + 1))
            kept = []
            kept_bytes = 0
            self.result.tospace_exhaustion += 1
        # install
        for o in kept:
            if o.age == 0:
                o.age = 1
            else:
                o.age += 1
        self.survivors = kept
        self.survivor_bytes = kept_bytes
        copied_objs = len(kept)
        self.eden = []
        self.eden_bytes = 0.0
        # promote
        prom_bytes = 0
        for o, a in promoted:
            self.old_objs.append(o)
            prom_bytes += o.size
            self.result.promoted_ages[a] = self.result.promoted_ages.get(a, 0) + 1
        self.old_bytes += prom_bytes
        self.old_committed += prom_bytes
        self.result.promotions_bytes += prom_bytes
        self.result.promotions_objs += len(promoted)
        pause = self._scavenge_pause(kept_bytes, copied_objs, prom_bytes)
        self.result.pauses.append(pause)
        self.result.pause_times.append(t)
        self.result.scavenges += 1
        self.result.copied_bytes += kept_bytes

    def run_day(self, duration_s: float, rate_obj_per_s: float,
                cohort_pause_stop: Optional[float] = None) -> OracleResult:
        """Allocate at a constant Poisson rate for ``duration_s`` seconds."""
        eden_cap = self.cfg.heap.eden()
        t = 0.0
        # pre-generate the next arrival time
        lam = max(rate_obj_per_s, 1e-9)
        nxt = t + self.rng.expovariate(lam)
        while t < duration_s:
            while nxt <= duration_s:
                o = self._new_object(nxt)
                self.eden.append(o)
                self.eden_bytes += o.size
                self.result.counted_objects += 1
                if self.eden_bytes >= eden_cap:
                    self.clock = nxt
                    self.scavenge(nxt)
                    if len(self.result.old_bytes_series) < 100000:
                        self.result.old_bytes_series.append((nxt, self.old_committed))
                nxt += self.rng.expovariate(lam)
            # no more arrivals within the window
            t = duration_s
        return self.result
