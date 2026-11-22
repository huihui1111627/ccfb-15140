"""Configuration objects for the generational GC simulator.

Everything the simulator needs to know is captured in dataclasses so that a
whole experiment (heap layout, GC tunables, traffic curve, lifetime mix) can
be serialised, diffed, and swept by the tuner.  All sizes are bytes, all times
are seconds, all rates are per second.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

from .lifetime import (
    CacheLifetime,
    RequestLifetime,
    SessionLifetime,
    TrafficClass,
)

KB = 1024
MB = 1024 * 1024
GB = 1024 * MB


@dataclass
class HeapConfig:
    """Heap layout and object-graph assumptions.

    The heap is split into a young generation and an old generation.  Young is
    an eden plus two survivor spaces (semi-space pair).  Old is a regioned
    mark-sweep space, as in the G1-style collector this models.
    """

    young_size: int = 2048 * MB         # eden + 2 * survivor_space
    survivor_ratio: float = 0.25        # each survivor = ratio * young_size
    # Fraction of surviving young bytes that does NOT survive the copy
    # (metaspace/klass references, weak refs, JNI locals).  Kept separate from
    # object lifetime so the copy cost model is honest.
    copy_nonsurvivor_frac: float = 0.0
    old_size: int = 6 * GB
    # "Marking" cost of a young survivor during the copy: bytes of object
    # graph visited / bytes copied.  >1 because copying a small object still
    # touches its class metadata and remembered-set entries.
    copy_graph_factor: float = 1.25
    # Cost to copy one byte (ns).  ~0.15 ns/B is ~6.7 GB/s of effective copy
    # bandwidth after graph traversal, consistent with G1 scavenges where
    # 20 MB of survivors costs a few ms and 130 MB costs ~30 ms.
    copy_ns_per_byte: float = 0.15
    # Fixed per-object cost of copying (ns): header update, forwarding ptr.
    copy_ns_per_object: float = 18.0
    # Card/remembered-set fixup for promoted objects, as a fraction of copy.
    rs_fixup_frac: float = 0.35
    # Old-gen compaction cost per live MB when a full GC compacts.
    compact_ns_per_live_mb: float = 150_000.0
    object_header: int = 16

    def survivor_space(self) -> int:
        return max(KB, int(self.young_size * self.survivor_ratio))

    def eden(self) -> int:
        return self.young_size - 2 * self.survivor_space()

    def young_from(self) -> int:
        return self.survivor_space()

    def young_to(self) -> int:
        return self.survivor_space()

    def validate(self) -> List[str]:
        errs = []
        if self.young_size < 4 * KB:
            errs.append("young_size must be >= 4 KiB")
        if not 0.01 <= self.survivor_ratio <= 0.45:
            errs.append("survivor_ratio must be in [0.01, 0.45]")
        if self.eden() <= 0:
            errs.append("eden must be positive; check survivor_ratio")
        if self.old_size <= self.young_size:
            errs.append("old_size should exceed young_size")
        return errs


@dataclass
class OldCollectorConfig:
    """Old-generation collector tunables.

    Two modes:

    * ``stop_the_world``: every old collection is a full-heap STW mark-sweep
      with optional compaction.  Pause ~= live set * scavenge cost.
    * ``concurrent``: start a concurrent marking cycle when old occupancy
      exceeds ``initiating_occupancy`` of old_size, or when a promotion cannot
      be satisfied.  The application keeps allocating while marking runs.
      At the remark pause (short, STW) the model checks whether the marking
      kept up with old-gen allocation during the cycle; if not, the concurrent
      mode *degrades* to a full STW collection (this is the real failure mode
      that produces mysterious 100-600 ms pauses).
    """

    mode: str = "concurrent"             # "concurrent" | "stop_the_world"
    initiating_occupancy: float = 0.45   # cycle starts at 45% old occupancy
    # Concurrent marking throughput, in "heap bytes scanned per second".
    # The marking thread scans the live old graph.  When the promotion rate
    # into old during a cycle exceeds this, marking can never catch up and the
    # cycle degrades to a full STW collection.
    mark_scan_bytes_per_s: float = 250 * MB
    # Fraction of marking work lost to the barrier/concurrency overhead.
    mark_overhead: float = 0.15
    # Application cores and concurrent marking threads.  Marking runs in
    # parallel with the application, but the marking threads occupy CPU and
    # steal cycles from the app.  The throughput cost of a cycle is modelled
    # as ``mark_threads / app_cores`` of the application's capacity while the
    # cycle runs (capped).  With 8 app cores and 2 marker threads, that is a
    # 25% capacity loss *during marking only*, not across the whole day.
    app_cores: int = 8
    conc_gc_threads: int = 2
    # A concurrent cycle that has not finished within this many seconds has
    # lost the race with allocation; the marker is forced to a remark (which
    # then degrades if it is behind).  Bounds cycle lifetime so marking cannot
    # be modelled as running indefinitely.
    max_cycle_s: float = 30.0
    # Remark (STW) pause: base + per-byte-of-live cost.  A remark of a 2 GB
    # live set is ~10-25 ms.
    remark_base_ms: float = 6.0
    remark_ns_per_live_byte: float = 0.004
    # Full GC (or degradation fallback) STW cost model.
    full_gc_base_ms: float = 12.0
    full_gc_ns_per_live_mb: float = 300_000.0   # mark+sweep cost per live MB
    # Whether a full GC compacts (costs copy time but returns free space).
    full_gc_compacts: bool = True
    # Fraction of the garbage a concurrent cycle discovers that is *not*
    # reclaimed at the remark (mixed collections reclaim only regions whose
    # liveness was below the threshold).  The rest waits for the next cycle.
    mixed_residual_garbage: float = 0.15
    # Marking is concurrent and allocates a little; used for pacing.
    remark_region_fraction: float = 0.02
    # Hard cap: if old occupancy > this after remark, force a full GC
    # regardless of the concurrent-mode state (evacuation failure).
    emergency_occupancy: float = 0.97

    def validate(self) -> List[str]:
        errs = []
        if self.mode not in ("concurrent", "stop_the_world"):
            errs.append("old mode must be 'concurrent' or 'stop_the_world'")
        if not 0.05 <= self.initiating_occupancy <= 0.95:
            errs.append("initiating_occupancy must be in [0.05, 0.95]")
        if self.mark_scan_bytes_per_s <= 0:
            errs.append("mark_scan_bytes_per_s must be positive")
        return errs


@dataclass
class YoungCollectorConfig:
    """Young-generation (scavenge) tunables."""

    # Promotion age: an object that survives this many scavenges is promoted.
    promotion_age: int = 7
    # Below this survivor occupancy the collector may pre-tenure the oldest
    # survivors instead of copying them (adaptive tenuring).
    adaptive_tenuring: bool = True
    tenuring_threshold_occupancy: float = 0.50
    max_pretenure_age: int = 3
    # Degraded mode: if young survival ratio has been above this for
    # ``degraded_trigger_n`` consecutive scavenges, eden is halved for
    # ``degraded_scavenges`` scavenges.  Models the observed behaviour of
    # reducing young-gen to avoid promotion storms -- and its side effect
    # (more frequent, cheaper minor GCs).
    degraded_trigger_survival: float = 0.85
    degraded_trigger_n: int = 3
    degraded_fraction: float = 0.5
    degraded_scavenges: int = 8

    def validate(self) -> List[str]:
        errs = []
        if self.promotion_age < 1:
            errs.append("promotion_age must be >= 1")
        if not 0.1 <= self.tenuring_threshold_occupancy <= 1.0:
            errs.append("tenuring_threshold_occupancy must be in [0.1, 1.0]")
        return errs


@dataclass
class GCConfig:
    heap: HeapConfig = field(default_factory=HeapConfig)
    young: YoungCollectorConfig = field(default_factory=YoungCollectorConfig)
    old: OldCollectorConfig = field(default_factory=OldCollectorConfig)

    def validate(self) -> List[str]:
        errs = self.heap.validate() + self.young.validate() + self.old.validate()
        # Promotion age must be reachable: if the survivor spaces can never
        # hold (age * survivor-size-worth) of live data, all objects are
        # force-promoted early and the tuning parameter is meaningless.
        return errs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GCConfig":
        return cls(
            heap=HeapConfig(**d.get("heap", {})),
            young=YoungCollectorConfig(**d.get("young", {})),
            old=OldCollectorConfig(**d.get("old", {})),
        )


@dataclass
class TrafficProfile:
    """One day of traffic, sampled at fixed resolution.

    ``base_alloc_rate`` is allocations/second at 1.0 traffic.  The ``curve``
    is a list of ``(hour, multiplier)`` control points, linearly interpolated,
    passed through optional smoothing (e.g. one-hour moving average).  A
    periodic hourly seasonality term plus Gaussian noise is added.

    The simulator computes per-tick allocation counts from
    ``base_alloc_rate * multiplier(t) * (1 + noise)``.
    """

    day_seconds: float = 24 * 3600.0
    tick_s: float = 0.25
    base_alloc_rate: float = 260_000.0   # objects/s at multiplier 1.0
    curve: List[Tuple[float, float]] = field(default_factory=list)
    hourly_amplitude: float = 0.0
    noise_sigma: float = 0.04
    # Burst events: (start_hour, end_hour, multiplier).  e.g. a batch job.
    bursts: List[Tuple[float, float, float]] = field(default_factory=list)

    @classmethod
    def trading_day(cls) -> "TrafficProfile":
        """A trading-front-end day: pre-open ramp, open burst, lunch dip,
        close spike, evening batch."""
        return cls(
            tick_s=0.25,
            base_alloc_rate=260_000.0,
            curve=[
                (0.0, 0.10), (5.0, 0.08), (8.0, 0.35), (9.0, 0.75),
                (9.28, 1.30), (9.5, 1.00), (11.5, 0.95), (12.0, 0.55),
                (13.0, 0.70), (14.5, 1.05), (14.85, 1.45), (15.0, 0.85),
                (16.0, 0.40), (18.0, 0.25), (20.0, 0.30), (21.0, 0.55),
                (22.0, 0.20), (24.0, 0.10),
            ],
            hourly_amplitude=0.06,
            noise_sigma=0.035,
            bursts=[(21.05, 21.35, 1.9)],  # end-of-day settlement batch
        )

    @classmethod
    def flat(cls, rate: Optional[float] = None) -> "TrafficProfile":
        return cls(
            base_alloc_rate=rate if rate is not None else 220_000.0,
            curve=[(0.0, 1.0), (24.0, 1.0)],
            hourly_amplitude=0.0,
            noise_sigma=0.0,
        )

    @classmethod
    def batch_heavy(cls) -> "TrafficProfile":
        return cls(
            tick_s=0.25,
            base_alloc_rate=150_000.0,
            curve=[
                (0.0, 0.20), (6.0, 0.15), (8.0, 0.30), (10.0, 0.40),
                (12.0, 0.35), (14.0, 0.37), (16.0, 0.32), (18.0, 0.50),
                (20.0, 1.25), (23.0, 0.60), (24.0, 0.20),
            ],
            hourly_amplitude=0.03,
            noise_sigma=0.05,
            bursts=[(19.5, 22.5, 1.35), (3.0, 3.4, 2.4)],
        )

    def multiplier(self, t: float) -> float:
        import math

        if not self.curve:
            return 1.0
        hours = (t / 3600.0) % 24.0
        pts = sorted(self.curve)
        v = self._interp(pts, hours)
        if self.hourly_amplitude:
            v *= 1.0 + self.hourly_amplitude * math.sin(2 * math.pi * hours / 24.0)
        for a, b, m in self.bursts:
            if a <= hours <= b:
                v *= m
        return max(0.0, v)

    @staticmethod
    def _interp(pts: Sequence[Tuple[float, float]], x: float) -> float:
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y1
                f = (x - x0) / (x1 - x0)
                return y0 + f * (y1 - y0)
        return pts[-1][1]


@dataclass
class ServiceProfile:
    """The object mix of one service."""

    name: str = "mixed"
    classes: List[TrafficClass] = field(default_factory=list)

    @classmethod
    def trading_front_end(cls) -> "ServiceProfile":
        """Order-routing front end.

        Byte shares are what matter for GC: requests dominate by count but are
        small (~80% of bytes), sessions are a small object count but ~10% of
        bytes and are the *medium-lived* population that survives several
        scavenges and is prone to premature promotion, and cache/metadata are
        a long-lived minority of bytes that forms the stable old-gen floor.
        """
        return cls(
            name="trading-front-end",
            classes=[
                TrafficClass("request", RequestLifetime(median_ms=6.0), share=0.966,
                             mean_size=104),
                TrafficClass("session", SessionLifetime(median_s=45.0, sigma=1.0,
                                                    long_frac=0.10, long_median_s=900.0),
                             share=0.022, mean_size=560),
                TrafficClass("cache", CacheLifetime(alpha=1.25), share=0.004,
                             mean_size=1800),
                TrafficClass("metadata", CacheLifetime(alpha=1.05, max_lifetime_s=24 * 3600),
                             share=0.008, mean_size=300),
            ],
        )

    @classmethod
    def session_heavy(cls) -> "ServiceProfile":
        """A service whose working set is dominated by session state: the
        hard case for promotion tuning."""
        return cls(
            name="session-heavy",
            classes=[
                TrafficClass("request", RequestLifetime(median_ms=10.0), share=0.90,
                             mean_size=88),
                TrafficClass("session", SessionLifetime(median_s=90.0, sigma=1.1,
                                                    long_frac=0.20, long_median_s=1800.0),
                             share=0.088, mean_size=900),
                TrafficClass("cache", CacheLifetime(alpha=1.1), share=0.004,
                             mean_size=1400),
                TrafficClass("metadata", CacheLifetime(alpha=1.02, max_lifetime_s=24 * 3600),
                             share=0.008, mean_size=300),
            ],
        )

    @classmethod
    def cache_heavy(cls) -> "ServiceProfile":
        """A quote/reference-data service: a large share of allocated *bytes*
        are cache entries that live for minutes to hours, so promotion is high
        by nature and the old generation fills steadily."""
        return cls(
            name="cache-heavy",
            classes=[
                TrafficClass("request", RequestLifetime(median_ms=4.0), share=0.88,
                             mean_size=80),
                TrafficClass("cache", CacheLifetime(alpha=1.08), share=0.09,
                             mean_size=1800),
                TrafficClass("session", SessionLifetime(median_s=40.0, long_frac=0.1),
                             share=0.02, mean_size=480),
                TrafficClass("metadata", CacheLifetime(alpha=1.01, max_lifetime_s=24 * 3600),
                             share=0.01, mean_size=256),
            ],
        )

    @classmethod
    def short_lived(cls) -> "ServiceProfile":
        """A service where almost everything dies young -- tuning is easy and
        the default parameters should already be excellent."""
        return cls(
            name="short-lived-batch",
            classes=[
                TrafficClass("request", RequestLifetime(median_ms=2.0, long_tail_frac=0.002),
                             share=0.985, mean_size=64),
                TrafficClass("session", SessionLifetime(median_s=8.0, long_frac=0.05),
                             share=0.012, mean_size=384),
                TrafficClass("cache", CacheLifetime(), share=0.003, mean_size=1024),
            ],
        )

    def mean_alloc_bytes(self) -> float:
        total = sum(c.share for c in self.classes)
        return sum(c.share / total * c.mean_size for c in self.classes)

    def describe(self) -> str:
        return self.name + " [" + "; ".join(c.describe() for c in self.classes) + "]"


def dump_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
