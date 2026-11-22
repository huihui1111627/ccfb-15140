"""Configuration objects and the built-in service scenarios.

Everything the simulator needs is described here so that a scenario can be
exported to JSON, edited, and replayed deterministically.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
MiB = 1024 * 1024
KiB = 1024

# Population classes: request objects, session objects, cache objects.
POP_TRANSIENT = 0
POP_SESSION = 1
POP_CACHE = 2
POP_NAMES = ("request", "session", "cache")
POP_LABELS = {
    "request": "请求瞬时对象",
    "session": "会话对象",
    "cache": "缓存对象",
}


def mib(x: float) -> int:
    return int(round(x * MiB))


def kib(x: float) -> int:
    return int(round(x * KiB))


# --------------------------------------------------------------------------
# Collector configuration
# --------------------------------------------------------------------------
@dataclass
class GCConfig:
    """Sizing / policy knobs of the collector -- the things you tune.

    Sizes are in bytes.  ``eden_bytes`` etc. are byte capacities; the model
    also tracks the fraction of a region that is occupied.
    """

    eden_bytes: int = mib(32)
    survivor_bytes: int = mib(4)          # per survivor space (from/to)
    old_bytes: int = mib(2048)            # old generation capacity

    # Young-collection copies per object (both copies of a minor) plus the
    # fixed cost of reaching a safepoint and scanning roots.  The wall-clock
    # pause of a minor is ``minor_fixed_s + per_live_s * live_objects``.
    # Live object sizes dominate because they are the ones being copied.
    minor_fixed_s: float = 0.6e-3
    minor_per_live_object_s: float = 3.0e-9
    minor_per_live_byte_s: float = 0.35e-9

    # Old generation.  ``full_gc_*`` describe stop-the-world mark-sweep.
    full_fixed_s: float = 10e-3
    full_per_live_object_s: float = 2.0e-9
    full_per_live_byte_s: float = 0.45e-9

    # Concurrent collector: an initial pause, a concurrent mark phase that
    # proceeds at ``concurrent_mark_bytes_per_s`` of heap scanning, and a
    # short remark/remap pause.  The cycle must finish before allocation
    # fills the heap; otherwise it degenerates into a full stop-the-world GC.
    concurrent_enabled: bool = True
    concurrent_pause_initial_s: float = 1.5e-3
    concurrent_pause_remark_s: float = 2.0e-3
    # Remark/remap rescans the dirty-card / SATB residue that concurrent
    # refinement did not drain.  The outstanding residue is bounded by the
    # young generation (the dominant mutation source between cleanups), not
    # by the whole old generation, so the pause is proportional to the young
    # generation size.
    concurrent_remark_per_mib_s: float = 0.04e-3
    concurrent_mark_bytes_per_s: float = 120e6
    concurrent_remark_ratio: float = 0.02     # remark scan fraction of heap
    # CPU lost to the concurrent marker.  A marker thread scans heap at a
    # finite rate; a single core sustains about ``concurrent_core_scan_bytes_per_s``
    # of marking, so marking at mark_rate costs mark_rate/core_rate cores of
    # application throughput.  This is why "just mark faster" is not free.
    concurrent_core_scan_bytes_per_s: float = 400e6
    concurrent_cpu_cost: float = 0.20
    # Number of cores the service runs on.  A marker thread occupies one
    # core, so marking at rate R costs (R / core_scan_rate) / service_cores
    # of the total application throughput.
    service_cores: int = 8
    # Trigger a concurrent cycle when old occupancy exceeds this fraction of
    # the old generation.  Also the "old gen is too full" degeneration gate.
    concurrent_trigger_ratio: float = 0.40
    # If a concurrent cycle is still running when old occupancy reaches this
    # fraction, it degenerates to a full GC.
    concurrent_panic_ratio: float = 0.90
    # Young collections during a concurrent cycle push the cycle's
    # "allocation pressure"; each minor's promotion adds to the concurrent
    # marking debt by this factor (write-barrier / dirty-card overhead).
    concurrent_barrier_slope: float = 1.0

    # Promotion policy
    tenuring_threshold: int = 6          # max age of a young object
    target_survivor_ratio: float = 0.5   # adaptive survivor sizing target
    max_tenuring_threshold: int = 15
    adaptive_tenuring: bool = True
    promotion_failure_policy: str = "promote"  # "promote" | "expand"

    # Human-readable name used in reports
    name: str = "default"

    # ------------------------------------------------------------------
    def young_bytes(self) -> int:
        return self.eden_bytes + 2 * self.survivor_bytes

    def heap_bytes(self) -> int:
        return self.young_bytes() + self.old_bytes

    def validate(self) -> None:
        for f in ("eden_bytes", "survivor_bytes", "old_bytes"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be positive")
        if self.tenuring_threshold < 1:
            raise ValueError("tenuring_threshold must be >= 1")
        if self.tenuring_threshold > self.max_tenuring_threshold:
            raise ValueError("tenuring_threshold > max_tenuring_threshold")
        if not 0.0 < self.concurrent_trigger_ratio < 1.0:
            raise ValueError("concurrent_trigger_ratio must be in (0,1)")
        if self.concurrent_enabled and self.concurrent_trigger_ratio >= self.concurrent_panic_ratio:
            raise ValueError("trigger ratio must be below panic ratio")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GCConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy(self) -> "GCConfig":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------
# Object population / lifespan model
# --------------------------------------------------------------------------
@dataclass
class ObjectPopulation:
    """Composition of the application's object population.

    Each class of objects is characterised by an *age-based* survival curve:
    the probability that a freshly allocated object is still alive when it
    reaches age ``t`` (seconds since allocation).  This mirrors the service
    behaviour described in the problem statement:

    * ``request``  -- per-request objects; almost all die within a few ms
                      (typically within the same young generation);
    * ``session``  -- session objects with an intermediate lifetime
                      (seconds to minutes); these are the ones that get
                      prematurely promoted and slowly fill the old gen;
    * ``cache``    -- long-lived objects, often pinned in caches.

    Lifetimes are Weibull distributed::

        S(t) = exp(-(t/scale)^shape)

    with ``shape < 1`` for heavy-tailed request objects (many very short
    lived, some stragglers), ``shape > 1`` for cache/lease objects whose
    lifetime is concentrated around ``scale``.
    """

    request_fraction: float = 0.90
    session_fraction: float = 0.07
    cache_fraction: float = 0.03

    request_scale_s: float = 0.004
    request_shape: float = 0.55
    session_scale_s: float = 40.0
    session_shape: float = 1.6
    cache_scale_s: float = 3600.0
    cache_shape: float = 4.0

    # Mean object sizes in bytes (per class, per allocation)
    request_size_bytes: float = 640.0
    session_size_bytes: float = 3000.0
    cache_size_bytes: float = 1200.0

    def validate(self) -> None:
        s = self.request_fraction + self.session_fraction + self.cache_fraction
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {s}")
        for f in ("request_fraction", "session_fraction", "cache_fraction"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")
        for f in ("request_scale_s", "session_scale_s", "cache_scale_s"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be > 0")
        for f in ("request_shape", "session_shape", "cache_shape"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be > 0")

    def mean_size(self) -> float:
        return (self.request_fraction * self.request_size_bytes
                + self.session_fraction * self.session_size_bytes
                + self.cache_fraction * self.cache_size_bytes)

    def survival(self, t: float, cls: int) -> float:
        """S(t) for class ``cls`` (0=request, 1=session, 2=cache)."""
        if cls == 0:
            scale, shape = self.request_scale_s, self.request_shape
        elif cls == 1:
            scale, shape = self.session_scale_s, self.session_shape
        else:
            scale, shape = self.cache_scale_s, self.cache_shape
        if t <= 0:
            return 1.0
        return math.exp(-((t / scale) ** shape))

    def class_mix(self) -> Tuple[float, float, float]:
        return (self.request_fraction, self.session_fraction, self.cache_fraction)

    def median_lifetime(self, cls: int) -> float:
        """Age at which S(t)=0.5; useful as a compact descriptor."""
        if cls == 0:
            scale, shape = self.request_scale_s, self.request_shape
        elif cls == 1:
            scale, shape = self.session_scale_s, self.session_shape
        else:
            scale, shape = self.cache_scale_s, self.cache_shape
        return scale * (math.log(2.0) ** (1.0 / shape))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectPopulation":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy(self) -> "ObjectPopulation":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------
# Traffic / workload model
# --------------------------------------------------------------------------
@dataclass
class WorkloadConfig:
    """Traffic model for one simulated day.

    ``day_seconds`` seconds of simulated time, driven by a piecewise-linear
    traffic curve with ``night_factor`` at the trough and 1.0 at the peak.
    ``alloc_rate_peak_mibs`` is the peak allocation rate reached when the
    traffic curve is at 1.0.  ``background_alloc_mibs`` is allocation that
    happens even at zero traffic (timers, monitoring, log flushing).

    The curve is defined by hourly median multipliers in ``traffic_curve``.
    If it is ``None`` a default dual-peak trading-day curve is used.
    """

    day_seconds: int = 86_400
    alloc_rate_peak_mibs: float = 48.0      # peak steady-state allocation MB/s
    alloc_rate_min_mibs: float = 1.5        # background allocations MB/s
    night_factor: float = 0.12              # trough multiplier of the curve
    traffic_curve: Optional[List[float]] = None  # 24 or N hourly multipliers
    # Burstiness: an Ornstein-Uhlenbeck multiplier on the traffic curve.
    burst_cv: float = 0.35                  # coefficient of variation
    burst_timescale_s: float = 90.0         # mean reversion time
    # Large-object allocations: fraction of bytes allocated as objects of
    # ``large_object_bytes``; big objects are allocated straight in the old
    # generation, as in a real collector's "humongous" path.
    large_object_fraction: float = 0.0
    large_object_bytes: int = 512 * KiB

    def validate(self) -> None:
        if self.day_seconds <= 0:
            raise ValueError("day_seconds must be positive")
        if self.alloc_rate_peak_mibs <= 0:
            raise ValueError("alloc_rate_peak_mibs must be > 0")
        if self.alloc_rate_min_mibs < 0:
            raise ValueError("alloc_rate_min_mibs must be >= 0")
        if self.traffic_curve is not None and len(self.traffic_curve) < 2:
            raise ValueError("traffic_curve needs at least 2 points")

    def curve(self) -> List[float]:
        if self.traffic_curve is not None:
            return list(self.traffic_curve)
        return DEFAULT_TRADING_DAY

    def curve_array(self) -> "np.ndarray":
        """Cached numpy view of the daily curve, for the hot allocation path."""
        arr = getattr(self, "_curve_cache", None)
        if arr is None or len(arr) != len(self.curve()):
            import numpy as _np
            arr = _np.asarray(self.curve(), dtype=float)
            self._curve_cache = arr
        return arr

    def multiplier_at(self, t_s: float) -> float:
        """Piecewise-linear daily traffic multiplier in [night_factor, 1]."""
        vals = (self.night_factor + (1.0 - self.night_factor)
                * self.curve_array())
        n = vals.shape[0]
        x = (t_s % self.day_seconds) / self.day_seconds * n
        i = int(x) % n
        f = x - math.floor(x)
        j = (i + 1) % n
        return float(vals[i] * (1.0 - f) + vals[j] * f)

    def alloc_rate_bytes_per_s(self, t_s: float) -> float:
        m = self.multiplier_at(t_s)
        rate = self.alloc_rate_min_mibs + (self.alloc_rate_peak_mibs
                                           - self.alloc_rate_min_mibs) * m
        return rate * MiB

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkloadConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy(self) -> "WorkloadConfig":
        return copy.deepcopy(self)


DEFAULT_TRADING_DAY = [
    # 24 hourly multipliers, 0..1.  Trough overnight, ramp at open, midday
    # lull, afternoon peak, close burst.
    0.05, 0.03, 0.02, 0.02, 0.03, 0.06, 0.14, 0.38,
    0.72, 0.95, 1.00, 0.88, 0.62, 0.70, 0.84, 0.93,
    1.00, 0.86, 0.55, 0.36, 0.24, 0.15, 0.10, 0.07,
]


# --------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """How the simulation is sampled and reported."""

    seed: int = 12345
    # A pause is a tail-latency failure if a request is caught in it and the
    # pause exceeds this many milliseconds.  (Incoming requests are assumed
    # to arrive uniformly, so the chance a request meets a pause of length L
    # is L / wall-clock time.)
    pause_budget_ms: float = 20.0
    # Share of the wall clock that may be spent in GC pauses; also the
    # throughput floor, since mutator work stops during a pause.
    throughput_target: float = 0.97
    # Sample mapping: average requests per second per MB/s of allocation
    # rate.  A pause of L seconds delays ~ (requests_per_mib * alloc_mibs)
    # requests; this turns pauses into an estimate of tail latency.
    requests_per_mib_rate: float = 900.0
    # Keep the sampled object table / age histogram in the result.
    track_ages: bool = True
    # Statistics collected before this time are excluded (the heap is still
    # filling); the simulation itself still runs, so state is warm.
    warmup_s: float = 0.0
    log_format: str = "hotspot"     # "hotspot" | "jvm" | "compact"
    # Weighted sampling: at most this many records are materialised per
    # class per young collection.  Each record stands for ``weight`` real
    # objects, so byte counts and promotion rates stay unbiased.
    max_records_per_minor: int = 160

    def validate(self) -> None:
        if self.max_records_per_minor < 8:
            raise ValueError("max_records_per_minor must be >= 8")
        if self.pause_budget_ms <= 0:
            raise ValueError("pause_budget_ms must be > 0")
        if not 0.0 < self.throughput_target < 1.0:
            raise ValueError("throughput_target must be in (0,1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
@dataclass
class ServiceScenario:
    name: str
    description: str
    workload: WorkloadConfig
    population: ObjectPopulation
    gc: GCConfig
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    # Optional tuning space definition used by the auto-tuner
    tuning_space: dict = field(default_factory=dict)
    # Maximum heap this service is allowed to use; the tuner treats it as a
    # hard constraint so it cannot "solve" a pause problem by buying memory.
    heap_budget_mib: int = 6144

    def validate(self) -> None:
        self.workload.validate()
        self.population.validate()
        self.gc.validate()
        self.sim.validate()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "workload": self.workload.to_dict(),
            "population": self.population.to_dict(),
            "gc": self.gc.to_dict(),
            "sim": self.sim.to_dict(),
            "tuning_space": copy.deepcopy(self.tuning_space),
            "heap_budget_mib": self.heap_budget_mib,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ServiceScenario":
        return cls(
            name=d.get("name", "scenario"),
            description=d.get("description", ""),
            workload=WorkloadConfig.from_dict(d.get("workload", {})),
            population=ObjectPopulation.from_dict(d.get("population", {})),
            gc=GCConfig.from_dict(d.get("gc", {})),
            sim=SimulationConfig.from_dict(d.get("sim", {})),
            tuning_space=copy.deepcopy(d.get("tuning_space", {})),
            heap_budget_mib=int(d.get("heap_budget_mib", 6144)),
        )


# -- built-in scenarios -----------------------------------------------------

def _scenario_trading_gateway() -> ServiceScenario:
    """Low-latency order gateway: tiny request objects, a session cache.

    The shipped default exhibits the classic pathology.  Sessions live
    ~20-25 s, much longer than a young-collection interval at peak, so they
    are promoted early; the old generation fills up during the afternoon
    peak, and the concurrent collector -- whose marking rate is comparable
    to the promotion rate -- fails to keep up, degenerating into repeated
    full stop-the-world collections.
    """
    pop = ObjectPopulation(
        request_fraction=0.88, session_fraction=0.09, cache_fraction=0.03,
        request_scale_s=0.004, request_shape=0.55,
        session_scale_s=25.0, session_shape=1.6,
        cache_scale_s=300.0, cache_shape=4.0,
        request_size_bytes=560.0, session_size_bytes=1600.0,
        cache_size_bytes=1400.0,
    )
    wl = WorkloadConfig(
        alloc_rate_peak_mibs=34.0, alloc_rate_min_mibs=1.6,
        night_factor=0.10, traffic_curve=list(DEFAULT_TRADING_DAY),
        burst_cv=0.40, burst_timescale_s=75.0,
    )
    gc = GCConfig(
        eden_bytes=mib(48), survivor_bytes=mib(4), old_bytes=mib(3072),
        tenuring_threshold=6, concurrent_trigger_ratio=0.50,
        concurrent_mark_bytes_per_s=16e6, concurrent_barrier_slope=1.0,
        name="gateway-default",
    )
    tuning_space = {
        "eden_mib": [16, 24, 32, 48, 64, 96, 128],
        "survivor_mib": [2, 4, 8, 12, 16, 24],
        "old_mib": [2048, 3072, 4096, 6144],
        "tenuring_threshold": [2, 3, 4, 6, 8, 10, 12],
        "concurrent_trigger_ratio": [0.30, 0.38, 0.45, 0.52, 0.60, 0.68],
        "concurrent_mark_mibs": [8, 16, 28, 45, 70],
    }
    return ServiceScenario(
        name="trading-gateway",
        description=(
            "Order-matching front end.  Heavy short-lived request objects, a "
            "9% session population that gets prematurely promoted, a small "
            "long-lived cache.  Dual-peak trading day with a closing burst."
        ),
        workload=wl, population=pop, gc=gc, tuning_space=tuning_space,
    )


def _scenario_market_data() -> ServiceScenario:
    """Market-data fan-out: high allocation, many medium-lived sessions.

    Sessions dominate the allocation (about half of all bytes at peak), so
    the concurrent collector is the deciding factor.  Burstiness is high and
    the evening tail is long.
    """
    pop = ObjectPopulation(
        request_fraction=0.82, session_fraction=0.15, cache_fraction=0.03,
        request_scale_s=0.0025, request_shape=0.50,
        session_scale_s=32.0, session_shape=1.35,
        cache_scale_s=90.0, cache_shape=5.0,
        request_size_bytes=420.0, session_size_bytes=2400.0,
        cache_size_bytes=1800.0,
    )
    wl = WorkloadConfig(
        alloc_rate_peak_mibs=56.0, alloc_rate_min_mibs=3.0,
        night_factor=0.18, burst_cv=0.55, burst_timescale_s=45.0,
        traffic_curve=[
            0.08, 0.05, 0.04, 0.04, 0.05, 0.09, 0.22, 0.55,
            0.88, 1.00, 0.97, 0.80, 0.58, 0.75, 0.92, 0.99,
            1.00, 0.82, 0.48, 0.30, 0.20, 0.14, 0.11, 0.09,
        ],
    )
    gc = GCConfig(
        eden_bytes=mib(96), survivor_bytes=mib(6), old_bytes=mib(6144),
        tenuring_threshold=8, concurrent_trigger_ratio=0.45,
        concurrent_mark_bytes_per_s=20e6, concurrent_barrier_slope=1.0,
        name="market-data-default",
    )
    tuning_space = {
        "eden_mib": [32, 48, 64, 96, 128, 192],
        "survivor_mib": [4, 6, 8, 12, 16, 24, 32],
        "old_mib": [2048, 3072, 4096, 6144],
        "tenuring_threshold": [3, 5, 6, 8, 10, 12, 15],
        "concurrent_trigger_ratio": [0.28, 0.35, 0.42, 0.50, 0.60, 0.68],
        "concurrent_mark_mibs": [30, 48, 80, 128, 192],
    }
    return ServiceScenario(
        name="market-data", heap_budget_mib=8192,
        description=(
            "Market-data fan-out with many medium-lived subscriber sessions; "
            "higher burstiness and a long evening tail."
        ),
        workload=wl, population=pop, gc=gc, tuning_space=tuning_space,
    )


def _scenario_order_lifecycle() -> ServiceScenario:
    """Order lifecycle: session objects live for minutes (fills, ack windows).

    A deliberately hostile population for a young-only sizing fix: the
    median session lifetime (minutes) is far longer than any reasonable
    tenuring threshold, so promotion is inevitable and the old collector
    must be sized and scheduled correctly.
    """
    pop = ObjectPopulation(
        request_fraction=0.76, session_fraction=0.21, cache_fraction=0.03,
        request_scale_s=0.006, request_shape=0.60,
        session_scale_s=120.0, session_shape=1.1,
        cache_scale_s=600.0, cache_shape=6.0,
        request_size_bytes=800.0, session_size_bytes=2200.0,
        cache_size_bytes=1800.0,
    )
    wl = WorkloadConfig(
        alloc_rate_peak_mibs=22.0, alloc_rate_min_mibs=1.0,
        night_factor=0.08, burst_cv=0.28, burst_timescale_s=120.0,
        traffic_curve=[
            0.04, 0.03, 0.03, 0.03, 0.04, 0.07, 0.16, 0.42,
            0.78, 0.93, 1.00, 0.92, 0.70, 0.76, 0.88, 0.96,
            0.98, 0.90, 0.62, 0.40, 0.26, 0.17, 0.11, 0.07,
        ],
    )
    gc = GCConfig(
        eden_bytes=mib(32), survivor_bytes=mib(4), old_bytes=mib(4096),
        tenuring_threshold=5, concurrent_trigger_ratio=0.45,
        concurrent_mark_bytes_per_s=18e6, concurrent_barrier_slope=1.0,
        name="order-lifecycle-default",
    )
    tuning_space = {
        "eden_mib": [16, 24, 32, 48, 64, 96],
        "survivor_mib": [2, 4, 8, 12, 16],
        "old_mib": [1536, 2048, 3072, 4096],
        "tenuring_threshold": [2, 3, 4, 5, 7, 9, 12],
        "concurrent_trigger_ratio": [0.25, 0.32, 0.40, 0.48, 0.55, 0.62],
        "concurrent_mark_mibs": [18, 30, 45, 70, 100],
    }
    return ServiceScenario(
        name="order-lifecycle",
        description=(
            "Order-lifecycle service: session objects live for minutes "
            "(fills, ack windows).  Any tenuring threshold that fixes the "
            "young promotion would have to be huge; the old collector must "
            "carry the load."
        ),
        workload=wl, population=pop, gc=gc, tuning_space=tuning_space,
    )


SCENARIOS: Dict[str, callable] = {
    "trading-gateway": _scenario_trading_gateway,
    "market-data": _scenario_market_data,
    "order-lifecycle": _scenario_order_lifecycle,
}


def get_scenario(name: str) -> ServiceScenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name]()


def run_scenario(name: str, gc: Optional[GCConfig] = None,
                 sim: Optional[SimulationConfig] = None,
                 **overrides) -> "SimulationResult":
    from .engine import Simulator
    sc = get_scenario(name)
    if gc is not None:
        sc.gc = gc
    if sim is not None:
        sc.sim = sim
    for k, v in overrides.items():
        if hasattr(sc.gc, k):
            setattr(sc.gc, k, v)
        elif hasattr(sc.workload, k):
            setattr(sc.workload, k, v)
        elif hasattr(sc.population, k):
            setattr(sc.population, k, v)
        elif hasattr(sc.sim, k):
            setattr(sc.sim, k, v)
        else:
            raise AttributeError(k)
    sc.validate()
    return Simulator(sc).run()


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
