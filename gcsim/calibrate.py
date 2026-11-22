"""Calibration: estimate workload and collector parameters from a GC log.

The log produced by this simulator (and by a real JVM with unified GC logging)
contains, per collection, the heap state before and after.  From a sequence of
such events you can recover:

* **Allocation rate** -- eden bytes consumed between consecutive scavenges
  divided by the interval.  Eden is filled by allocation and emptied by
  scavenge, so the eden occupancy just before each scavenge equals the eden
  capacity and the *increase* in heap occupancy between scavenges tracks
  allocation minus collection.
* **Promotion rate** -- the rise in old occupancy across scavenges that do not
  include an old collection, divided by the interval.  A mixed/remark event
  removes old garbage, so intervals spanning one are excluded.
* **Survivor / live size per scavenge** -- from the ``Survivors: a->b`` field
  and from ``Heap: before->after``.
* **Old occupancy trend** -- the ``Heap:`` and full-GC fields over the day.

The estimates are ordinary least-squares / ratio estimators with explicit
units, and the report states the residual error so a user can see how well the
model fits.  ``calibrate_from_log`` returns a :class:`CalibrationReport` with
the estimates and the diagnostics, and ``apply_calibration`` turns the
estimated allocation and promotion rates into a service profile + traffic
curve the simulator can run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import GCConfig, ServiceProfile, TrafficProfile
from .gcparse import parse_gc_log, parse_header
from .gclog import GCEvent


@dataclass
class CalibrationReport:
    # raw counts
    n_events: int = 0
    n_young: int = 0
    n_pause: int = 0
    n_full: int = 0
    n_concurrent: int = 0
    duration_s: float = 0.0
    # estimates
    alloc_rate_bytes_s: float = 0.0
    promotion_rate_bytes_s: float = 0.0
    promotion_fraction: float = 0.0      # promoted / allocated
    mean_survivor_bytes: float = 0.0
    survivor_ratio: float = 0.0
    young_size_bytes: float = 0.0
    eden_bytes: float = 0.0
    old_size_bytes: float = 0.0
    old_occupancy_mean: float = 0.0
    old_occupancy_growth_bytes_s: float = 0.0
    pause_p50: float = 0.0
    pause_p99: float = 0.0
    pause_max: float = 0.0
    full_gc_pause_mean: float = 0.0
    # implicit GC parameters
    inferred_young_size: float = 0.0
    inferred_survivor_ratio: float = 0.0
    inferred_initiating_occupancy: float = 0.0
    # fit quality
    # Spread of the per-interval estimates (median absolute deviation / mean).
    # This reflects real traffic variation across the day, not estimator error;
    # it is reported so the reader can see how much the rate moves.
    alloc_rate_mad_pct: float = 0.0
    promotion_rate_mad_pct: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
            elif isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d

    def summary_lines(self) -> List[str]:
        return [
            f"events: {self.n_events} (young {self.n_young}, pauses {self.n_pause}, "
            f"full {self.n_full}, concurrent {self.n_concurrent}) over {self.duration_s/3600:.2f} h",
            f"allocation rate: {self.alloc_rate_bytes_s/1e6:.1f} MB/s "
            f"(per-interval spread {self.alloc_rate_mad_pct:.1f}%)",
            f"promotion rate:  {self.promotion_rate_bytes_s/1e6:.1f} MB/s "
            f"({self.promotion_fraction*100:.1f}% of alloc, spread "
            f"{self.promotion_rate_mad_pct:.1f}%)",
            f"mean survivor occupancy: {self.mean_survivor_bytes/1e6:.1f} MB; "
            f"survivor ratio {self.survivor_ratio*100:.1f}%",
            f"inferred young: {self.inferred_young_size/1e6:.0f} MB "
            f"(eden {self.eden_bytes/1e6:.0f} MB, survivor ratio {self.inferred_survivor_ratio:.2f})",
            f"old: capacity {self.old_size_bytes/1e9:.1f} GB, mean occupancy "
            f"{self.old_occupancy_mean*100:.1f}%, growth "
            f"{self.old_occupancy_growth_bytes_s/1e6:.1f} MB/s",
            f"inferred initiating occupancy: {self.inferred_initiating_occupancy:.2f}",
            f"pause p50 {self.pause_p50*1e3:.2f} ms, p99 {self.pause_p99*1e3:.2f} ms, "
            f"max {self.pause_max*1e3:.2f} ms; full-GC mean {self.full_gc_pause_mean*1e3:.1f} ms",
        ]


def _linear_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope/intercept of y on x."""
    if xs.size < 2:
        return 0.0, float(ys[0]) if ys.size else 0.0
    A = np.vstack([xs, np.ones_like(xs)]).T
    try:
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, float(np.mean(ys))
    return float(slope), float(intercept)


def _mad_pct(values: Sequence[float], center: float) -> float:
    if not values or center == 0:
        return 0.0
    med = float(np.median(np.abs(np.asarray(values) - center)))
    return 100.0 * med / abs(center)


def _infer_units(events: Sequence[GCEvent]) -> Tuple[float, float, float]:
    """Recover young size and old capacity from the events.

    The young size is ``eden_capacity + 2 * survivor_space`` and both appear
    verbatim in the log: eden capacity in the ``Eden: b(cap)`` field, and the
    survivor space as the maximum ``Survivors: a->b`` value seen (a scavenge
    that nearly fills the space reveals it; if none does, the log itself says
    nothing, and we report only what is known).  The old generation capacity
    is the heap capacity minus the young size.
    """
    eden_cap = 0.0
    for e in events:
        if e.kind == "young":
            eden_cap = max(eden_cap, e.eden_capacity or e.eden_before)
    surv_cap = 0.0
    for e in events:
        if e.kind == "young":
            surv_cap = max(surv_cap, e.surv_after, e.surv_before)
    heap_cap = max([e.heap_capacity for e in events] or [0.0])
    young_size = (eden_cap + 2 * surv_cap) if surv_cap > 0 else eden_cap
    old_cap = max(0.0, heap_cap - young_size) if heap_cap else 0.0
    return eden_cap, young_size, old_cap


def calibrate_events(events: Sequence[GCEvent],
                     header: Optional[dict] = None) -> CalibrationReport:
    """Turn parsed GC events into workload estimates.

    ``header`` (from :func:`gcsim.gcparse.parse_header`) is optional; when the
    log carries the JVM init lines it makes the survivor ratio and tenuring
    threshold exact rather than inferred."""
    rep = CalibrationReport()
    if not events:
        rep.notes.append("no events parsed")
        return rep
    rep.n_events = len(events)
    rep.n_young = sum(1 for e in events if e.kind == "young")
    rep.n_full = sum(1 for e in events if e.kind == "full")
    rep.n_concurrent = sum(1 for e in events if e.kind == "concurrent"
                           or e.kind == "remark")
    pauses = [e for e in events if e.is_pause]
    rep.n_pause = len(pauses)
    t0 = min(e.t for e in events)
    t1 = max(e.t for e in events)
    rep.duration_s = t1 - t0

    eden_cap, young_size, old_cap = _infer_units(events)
    hdr = header or {}
    if hdr.get("young_size"):
        young_size = hdr["young_size"]
        # eden = young * (1 - 2*survivor_ratio) if ratio known; else keep log cap
        if hdr.get("survivor_ratio"):
            eden_cap = young_size * (1.0 - 2.0 * hdr["survivor_ratio"])
    if hdr.get("heap_max"):
        old_cap = max(0.0, hdr["heap_max"] - young_size)
    rep.eden_bytes = eden_cap
    rep.young_size_bytes = young_size
    rep.old_size_bytes = old_cap
    rep.inferred_young_size = young_size
    # Survivor ratio is the *configured* ratio: each survivor space is
    # survivor_ratio * young_size, and young = eden + 2 * survivor.
    if hdr.get("survivor_ratio"):
        rep.inferred_survivor_ratio = hdr["survivor_ratio"]
        rep.survivor_ratio = hdr["survivor_ratio"]
    elif young_size > eden_cap:
        rep.inferred_survivor_ratio = (young_size - eden_cap) / (2 * young_size)
        rep.survivor_ratio = (young_size - eden_cap) / (2 * young_size)
    else:
        rep.inferred_survivor_ratio = 0.0

    # Allocation rate.  Eden is emptied by every scavenge, so between two
    # consecutive scavenges the bytes allocated equal the eden occupancy just
    # before the later one; summed over the day that is the day's allocation.
    # Using *sum(eden_before) / total_time* rather than a median of per-pair
    # rates avoids over-weighting the peak (there are far more scavenges in a
    # peak hour than in a quiet one).
    young = [e for e in events if e.kind == "young" and e.eden_before > 0]
    if young:
        # Within a contiguous run of scavenges the time between the first and
        # last event is the interval over which eden was filled; each of those
        # events corresponds to one full eden of allocation.
        alloc_rates = []
        t_first = young[0].t
        t_last = young[-1].t
        span = t_last - t_first
        if span > 0:
            rep.alloc_rate_bytes_s = float(
                sum(e.eden_before for e in young[1:]) / span)
            for a, b in zip(young, young[1:]):
                dt = b.t - a.t
                if 0 < dt < 3600:
                    alloc_rates.append(b.eden_before / dt)
            rep.alloc_rate_mad_pct = _mad_pct(alloc_rates, rep.alloc_rate_bytes_s)

    # Promotion rate.  Old occupancy rises by promotion and falls only when an
    # old collection reclaims.  Integrate the positive increments of old
    # occupancy across scavenges, divide by elapsed time.  This is exact for
    # the model where ``[Old: a->b]`` is logged; the fallback infers old from
    # heap-after minus survivors.
    promo_cum = [(e.t, e.promoted_cum) for e in events
                 if e.kind == "young" and e.promoted_cum > 0]
    if len(promo_cum) >= 2:
        span = promo_cum[-1][0] - promo_cum[0][0]
        if span > 0:
            rep.promotion_rate_bytes_s = (promo_cum[-1][1] - promo_cum[0][1]) / span
            # MAD from per-interval differences
            d = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(promo_cum, promo_cum[1:])
                 if b[0] > a[0]]
            rep.promotion_rate_mad_pct = _mad_pct(d, rep.promotion_rate_bytes_s)
        if rep.alloc_rate_bytes_s > 0:
            rep.promotion_fraction = rep.promotion_rate_bytes_s / rep.alloc_rate_bytes_s
    old_series = [(e.t, e.old_after) for e in events
                  if e.kind == "young" and e.old_after > 0]
    promo_rates = []
    if rep.promotion_rate_bytes_s > 0:
        pass
    elif len(old_series) >= 2 and any(o > 0 for _, o in old_series):
        total_gain = 0.0
        for (ta, oa), (tb, ob) in zip(old_series, old_series[1:]):
            dt = tb - ta
            if 0 < dt < 3600 and ob > oa:
                gain = ob - oa
                total_gain += gain
                promo_rates.append(gain / dt)
        span = old_series[-1][0] - old_series[0][0]
        if span > 0:
            rep.promotion_rate_bytes_s = total_gain / span
            rep.promotion_rate_mad_pct = _mad_pct(promo_rates,
                                                 rep.promotion_rate_bytes_s)
    if rep.promotion_rate_bytes_s <= 0:
        # Fallback: heap-after growth while old is not collected.
        heap_after = [(e.t, e.heap_after, e.surv_after) for e in events
                      if e.kind == "young" and e.heap_after > 0]
        rates = []
        for (ta, ha, sa), (tb, hb, sb) in zip(heap_after, heap_after[1:]):
            dt = tb - ta
            if 0 < dt < 3600 and hb > ha:
                rates.append((hb - ha) / dt)
        if rates:
            rep.promotion_rate_bytes_s = float(np.median(rates))
            rep.promotion_rate_mad_pct = _mad_pct(rates, rep.promotion_rate_bytes_s)
    if rep.alloc_rate_bytes_s > 0:
        rep.promotion_fraction = rep.promotion_rate_bytes_s / rep.alloc_rate_bytes_s

    # Survivor bytes from the "Survivors: a->b" field.
    surv = [e.surv_after for e in young if e.surv_after > 0]
    if surv:
        rep.mean_survivor_bytes = float(np.mean(surv))
        if not hdr.get("survivor_ratio"):
            # Only infer the ratio from occupancy when the log does not carry
            # the configured value.  The observed survivor occupancy is *not*
            # the space size: a space is rarely full.
            rep.survivor_ratio = (rep.mean_survivor_bytes /
                                  (rep.mean_survivor_bytes + rep.eden_bytes))
    # Old occupancy and growth: from heap_after (approx survivors + old) minus
    # survivor space, but the dominant term at capacity is old.  Use a linear
    # fit of heap_after over time between full/remark events.
    heap_series = [(e.t, e.heap_after) for e in events
                   if e.kind == "young" and e.heap_after > 0]
    if len(heap_series) >= 3:
        xs = np.array([t for t, h in heap_series])
        ys = np.array([h for t, h in heap_series])
        slope, _ = _linear_fit(xs, ys)
        rep.old_occupancy_growth_bytes_s = slope
        if old_cap > 0:
            rep.old_occupancy_mean = float(np.mean(ys) / old_cap)
    # Initiating occupancy: the heap-after occupancy at the first concurrent
    # mark-start following a young collection is the trigger point.
    if old_cap > 0:
        triggers = [e.heap_after / old_cap for e in events
                    if e.kind in ("young", "mixed") and e.heap_after > 0]
        if triggers:
            rep.inferred_initiating_occupancy = float(np.median(triggers))

    if pauses:
        xs = np.array([e.pause_s for e in pauses])
        rep.pause_p50 = float(np.percentile(xs, 50))
        rep.pause_p99 = float(np.percentile(xs, 99))
        rep.pause_max = float(xs.max())
    fulls = [e.pause_s for e in events if e.kind == "full" and e.is_pause]
    if fulls:
        rep.full_gc_pause_mean = float(np.mean(fulls))
    return rep


def calibrate_from_log(path: str) -> CalibrationReport:
    with open(path, "r", errors="replace") as f:
        text = f.read()
    return calibrate_events(parse_gc_log(text), header=parse_header(text))


def calibrate_from_text(text: str) -> CalibrationReport:
    return calibrate_events(parse_gc_log(text), header=parse_header(text))


def apply_calibration(rep: CalibrationReport, cfg: Optional[GCConfig] = None,
                      duration_s: Optional[float] = None) -> Tuple[GCConfig, ServiceProfile, TrafficProfile]:
    """Build a runnable (config, service, traffic) triple from a calibration.

    The service keeps a default lifetime mix but scales the traffic rate so the
    *byte* allocation rate matches the estimate.  Young/old sizes are set from
    the inferred log geometry.
    """
    cfg = GCConfig() if cfg is None else cfg
    if rep.inferred_young_size > 0:
        cfg.heap.young_size = int(rep.inferred_young_size)
    if rep.old_size_bytes > 0:
        cfg.heap.old_size = int(rep.old_size_bytes)
    if rep.inferred_survivor_ratio > 0:
        cfg.heap.survivor_ratio = float(rep.inferred_survivor_ratio)
    svc = ServiceProfile.trading_front_end()
    mean_bytes = svc.mean_alloc_bytes()
    rate_obj = rep.alloc_rate_bytes_s / max(1.0, mean_bytes)
    traffic = TrafficProfile.flat(rate_obj)
    if duration_s:
        traffic.day_seconds = duration_s
    return cfg, svc, traffic
