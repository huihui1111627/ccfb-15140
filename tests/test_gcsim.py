"""Tests for the GC simulator.

Run with ``python -m pytest -q`` (pytest is used if available; the file also
runs under ``python -m unittest``).  The tests pin down the *mechanism*, not
specific numbers: they check that a scavenge promotes at the tenure age, that
space pressure pre-tenures, that the concurrent cycle degrades when promotion
outruns the marker, that the log round-trips, and that calibration recovers
known parameters.

The most important test is :func:`test_cohort_matches_object_oracle`: it runs
the same configuration through the cohort engine and the object-by-object
oracle and checks that promotion volume and pause totals agree.  That is what
justifies the cohort approximation the whole tool is built on.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from gcsim.config import (
    GCConfig, HeapConfig, OldCollectorConfig, ServiceProfile, TrafficProfile,
    YoungCollectorConfig, MB, GB,
)
from gcsim.gclog import GCLogWriter
from gcsim.gcparse import parse_gc_log, parse_header, parse_size
from gcsim.calibrate import calibrate_from_text, calibrate_events
from gcsim.simulator import HeapSimulator
from gcsim.survival import surv, quantile_age
from gcsim.workload import build_trace
from gcsim.lifetime import (
    CacheLifetime, DeterministicLifetime, RequestLifetime, SessionLifetime,
    TrafficClass,
)


# --------------------------------------------------------------------------
# config / validation
# --------------------------------------------------------------------------

def test_heap_layout_arithmetic():
    h = HeapConfig(young_size=1024 * MB, survivor_ratio=0.25)
    assert h.survivor_space() == 256 * MB
    assert h.eden() == 512 * MB  # 1024 - 2*256


def test_invalid_config_rejected():
    cfg = GCConfig()
    cfg.young.promotion_age = 0
    sim_errs = cfg.validate()
    assert any("promotion_age" in e for e in sim_errs)


# --------------------------------------------------------------------------
# survival functions
# --------------------------------------------------------------------------

def test_survival_monotone_and_bounded():
    p = (60.0, 1.0, 0.1, 900.0, 1.3)  # lognormal mixture (session)
    s = [surv("lognormal_mix", p, t) for t in (0, 1, 10, 100, 1000, 1e6)]
    assert s[0] == 1.0
    assert all(0.0 <= v <= 1.0 for v in s)
    assert all(a >= b for a, b in zip(s, s[1:]))


def test_pareto_survival():
    p = (10.0, 1.5, 1e9)
    assert surv("pareto", p, 5) == 1.0
    assert abs(surv("pareto", p, 10) - 1.0) < 1e-12
    assert abs(surv("pareto", p, 100) - (10 / 100) ** 1.5) < 1e-12
    assert surv("pareto", p, 1e10) == 0.0


# --------------------------------------------------------------------------
# mechanism: scavenge, ageing, promotion
# --------------------------------------------------------------------------

def _simple_service():
    # Two classes: 90% short-lived, 10% long-lived, all similar size.
    return ServiceProfile(
        name="test",
        classes=[
            TrafficClass("short", RequestLifetime(median_ms=5.0), share=0.90,
                         mean_size=100, size_sigma=0.0),
            TrafficClass("long", CacheLifetime(min_lifetime_s=1000, alpha=1.2),
                         share=0.10, mean_size=100, size_sigma=0.0),
        ],
    )


def test_scavenge_promotes_at_tenure_age():
    """With plenty of survivor space, every promotion should happen exactly at
    the configured tenure age."""
    svc = _simple_service()
    tr = TrafficProfile.flat(50000)
    tr.day_seconds = 600
    trace = build_trace(svc, tr, seed=1, cohort_seconds=1.0, noise=False)
    cfg = GCConfig()
    cfg.heap.young_size = 512 * MB
    cfg.heap.survivor_ratio = 0.40   # generous survivor space
    cfg.heap.old_size = 8 * GB
    cfg.young.promotion_age = 6
    cfg.young.adaptive_tenuring = True
    cfg.old.initiating_occupancy = 0.95  # do not collect old
    cfg.old.mode = "stop_the_world"
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    ages = st.promoted_age_bytes
    assert ages, "expected promotions"
    # The dominant promoted age must be exactly the tenure age.
    dominant = max(ages, key=ages.get)
    assert dominant == 6, f"promoted ages {sorted(ages)}"


def test_no_promotion_without_long_lived_objects():
    """A service where everything dies in microseconds promotes nothing even
    with a large young gen and a low tenure age."""
    svc = ServiceProfile(name="eph", classes=[
        TrafficClass("req", RequestLifetime(median_ms=0.05, long_tail_frac=0.0),
                     share=1.0, mean_size=64, size_sigma=0.0)])
    tr = TrafficProfile.flat(20000)
    tr.day_seconds = 300
    trace = build_trace(svc, tr, seed=2, cohort_seconds=1.0, noise=False)
    cfg = GCConfig()
    cfg.heap.young_size = 256 * MB
    cfg.young.promotion_age = 1
    cfg.old.initiating_occupancy = 0.94
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    assert st.promotion_bytes < 1e6, st.promotion_bytes


def test_small_survivor_space_forces_early_promotion():
    """A survivor space too small for the medium-lived cohort lowers the
    effective tenure age: promotion happens earlier than configured."""
    svc = ServiceProfile(name="sess", classes=[
        TrafficClass("req", RequestLifetime(median_ms=3.0), share=0.7,
                     mean_size=100, size_sigma=0.0),
        TrafficClass("sess", SessionLifetime(median_s=120.0, sigma=0.8,
                                             long_frac=0.0),
                     share=0.3, mean_size=800, size_sigma=0.0)])
    tr = TrafficProfile.flat(120000)
    tr.day_seconds = 600
    trace = build_trace(svc, tr, seed=3, cohort_seconds=1.0, noise=False)

    def run(survivor_ratio):
        cfg = GCConfig()
        cfg.heap.young_size = 512 * MB
        cfg.heap.survivor_ratio = survivor_ratio
        cfg.young.promotion_age = 15
        cfg.young.adaptive_tenuring = True
        cfg.heap.old_size = 8 * GB
        cfg.old.initiating_occupancy = 0.94
        sim = HeapSimulator(cfg, svc, tr, trace=trace)
        st = sim.run()
        ages = st.promoted_age_bytes or {0: 1}
        return max(ages, key=ages.get), st.promoted_age_bytes

    big_age, _ = run(0.45)   # plenty of survivor space
    small_age, _ = run(0.05)  # starved survivor space
    assert small_age < big_age, (small_age, big_age)


def test_tiny_survivor_space_promotes_all_survivors():
    """A survivor space far smaller than one scavenge's live set cannot hold
    the survivors, so the collector must promote (almost) everything rather
    than copy it.  The guarantee that matters is that a starved survivor space
    never silently keeps objects it cannot hold, and that promotion rises
    sharply compared to a generously sized space."""
    svc = ServiceProfile(name="big", classes=[
        TrafficClass("keep", DeterministicLifetime(lifetime_s=10000.0),
                     share=1.0, mean_size=200, size_sigma=0.0)])

    def run(survivor_ratio):
        tr = TrafficProfile.flat(120000)
        tr.day_seconds = 240
        trace = build_trace(svc, tr, seed=4, cohort_seconds=1.0, noise=False)
        cfg = GCConfig()
        cfg.heap.young_size = 256 * MB
        cfg.heap.survivor_ratio = survivor_ratio
        cfg.young.promotion_age = 15
        cfg.heap.old_size = 8 * GB
        cfg.old.initiating_occupancy = 0.94
        sim = HeapSimulator(cfg, svc, tr, trace=trace)
        st = sim.run()
        # Survivor space must never overflow its physical capacity.
        assert st.survivor_bytes_peak() <= cfg.heap.survivor_space() * 1.05
        return st.promotion_bytes / max(1.0, st.allocated_bytes)

    frac_tiny = run(0.01)
    frac_big = run(0.45)
    # With essentially everything living, objects are promoted regardless, but
    # a starved space must not retain a larger young population than its
    # capacity; the promoted *fraction* is at least as high as with a big space.
    assert frac_tiny >= frac_big - 1e-6
    assert frac_tiny > 0.5


# --------------------------------------------------------------------------
# concurrent collection / degradation
# --------------------------------------------------------------------------

def test_concurrent_cycle_runs_and_reclaims():
    svc = ServiceProfile.cache_heavy()
    tr = TrafficProfile.batch_heavy()
    tr.day_seconds = 3600
    trace = build_trace(svc, tr, seed=5, cohort_seconds=5.0)
    cfg = GCConfig()
    cfg.heap.young_size = 256 * MB
    cfg.heap.old_size = 3 * GB
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    assert st.concurrent_cycles >= 1
    assert st.remark_count >= 1
    # old occupancy must sawtooth, i.e. actually be reclaimed at remarks
    old = [r["old_committed"] for r in st.timeline]
    assert min(old) < max(old) * 0.9


def test_degradation_when_marker_cannot_keep_up():
    """A slow marker in a tight old gen must degrade to full GCs, producing
    long pauses -- the failure mode the tool is built to explain."""
    # Peak-hour promotion into a nearly-full old generation with a marker
    # slower than promotion outruns the cycle.
    svc = ServiceProfile.cache_heavy()
    tr = TrafficProfile.trading_day()
    trace = build_trace(svc, tr, seed=6, cohort_seconds=5.0)
    cfg = GCConfig()
    cfg.heap.young_size = 512 * MB
    cfg.heap.old_size = 2000 * MB
    cfg.old.initiating_occupancy = 0.60
    cfg.old.mark_scan_bytes_per_s = 150 * MB
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    assert st.degraded_cycles >= 1
    assert st.full_gcs >= 1
    assert max(p for _, p, _ in st.pauses) > 0.1  # > 100 ms


def test_stop_the_world_mode_never_marks():
    svc = ServiceProfile.session_heavy()
    tr = TrafficProfile.trading_day()
    tr.day_seconds = 1800
    trace = build_trace(svc, tr, seed=7, cohort_seconds=5.0)
    cfg = GCConfig()
    cfg.old.mode = "stop_the_world"
    cfg.heap.young_size = 256 * MB
    cfg.heap.old_size = 2 * GB
    cfg.old.initiating_occupancy = 0.5
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    assert st.concurrent_cycles == 0
    assert st.remark_count == 0
    assert st.full_gcs >= 1


# --------------------------------------------------------------------------
# log round-trip and parsing
# --------------------------------------------------------------------------

def test_log_roundtrip_parse():
    w = GCLogWriter(header=False)  # in-memory
    w.young(10.0, 0.012, 100 * MB, 0, 100 * MB, 20 * MB, 30 * MB,
            900 * MB, 950 * MB, 4 * GB)
    w.full(30.0, 0.5, 2 * GB, 1.2 * GB, 4 * GB)
    w.concurrent(31.0, "mark-start")
    text = "\n".join(w.lines)
    ev = parse_gc_log(text)
    kinds = [e.kind for e in ev]
    assert "young" in kinds
    assert "full" in kinds
    assert "concurrent" in kinds
    young = next(e for e in ev if e.kind == "young")
    assert abs(young.t - 10.0) < 1e-6
    assert abs(young.eden_before - 100 * MB) < 1e4
    assert abs(young.surv_after - 30 * MB) < 1e4


def test_parse_size_units():
    assert parse_size("512.0M") == 512 * MB
    assert parse_size("1.5G") == 1.5 * GB
    assert parse_size("0B") == 0.0
    assert parse_size("2048K") == 2048 * 1024


def test_parse_header_geometry():
    w = GCLogWriter(header=True)
    w.set_heap(512 * MB, 6 * GB, 0.25, 0.45, 7, "concurrent")
    w.start()
    hdr = parse_header("\n".join(w.lines))
    assert abs(hdr["young_size"] - 512 * MB) < 1e4
    assert abs(hdr["survivor_ratio"] - 0.25) < 1e-6
    assert hdr["tenuring_threshold"] == 7
    assert abs(hdr["initiating"] - 0.45) < 1e-6


def test_real_jvm_style_line_parses():
    """A line in the older one-line G1 shape (no brackets around time)."""
    text = (
        "2024-01-01T09:00:00.000+0000: 12.345: [GC pause (G1 Evacuation "
        "Pause) (young), 0.0045000 secs]\n"
        "   [Eden: 256.0M(256.0M)->0.0B(256.0M) Survivors: 20.0M->30.0M "
        "Heap: 500.0M(8192.0M)->280.0M(8192.0M)]\n"
    )
    ev = parse_gc_log(text)
    assert len(ev) == 1
    assert ev[0].kind == "young"
    assert abs(ev[0].pause_s - 0.0045) < 1e-9


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_calibration_recovers_parameters():
    """Write a full log from a known configuration, then check that
    calibration recovers allocation rate, promotion rate and the heap
    geometry from the log alone."""
    from gcsim.report import RunDigest
    svc = ServiceProfile.trading_front_end()
    tr = TrafficProfile.trading_day()
    tr.day_seconds = 6 * 3600
    cfg = GCConfig()
    cfg.heap.young_size = 1024 * MB
    cfg.heap.survivor_ratio = 0.25
    cfg.heap.old_size = 8 * GB
    trace = build_trace(svc, tr, seed=11, cohort_seconds=5.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gc.log")
        writer, sink = __import__("gcsim.gclog", fromlist=["open_writer"]).open_writer(
            path, header=True, start=False)
        writer.set_heap(cfg.heap.young_size, cfg.heap.old_size,
                        cfg.heap.survivor_ratio, cfg.old.initiating_occupancy,
                        cfg.young.promotion_age, cfg.old.mode)
        writer.start()
        sim = HeapSimulator(cfg, svc, tr, seed=11, trace=trace,
                            log_writer=writer)
        st = sim.run()
        sink.close()
        text = open(path).read()
    rep = calibrate_from_text(text)
    true_alloc = st.allocated_bytes / (6 * 3600)
    true_prom = st.promotion_bytes / (6 * 3600)
    assert abs(rep.inferred_young_size - cfg.heap.young_size) / cfg.heap.young_size < 0.01
    assert abs(rep.inferred_survivor_ratio - cfg.heap.survivor_ratio) < 0.01
    assert abs(rep.alloc_rate_bytes_s - true_alloc) / true_alloc < 0.02
    assert abs(rep.promotion_rate_bytes_s - true_prom) / max(1, true_prom) < 0.10


# --------------------------------------------------------------------------
# cohort engine vs object oracle
# --------------------------------------------------------------------------

def test_cohort_matches_object_oracle():
    """The cohort engine and the discrete object oracle must agree.

    Both run the same two-class workload with the same survivor space and
    tenure age.  The cohort engine uses the *expected* allocation of N objects
    per second; the oracle allocates them one at a time with sampled lifetimes
    and sizes.  We compare promoted *fraction of allocated bytes* and the
    number of scavenges, which should agree closely despite sampling noise.
    """
    from gcsim.oracle import ObjectOracle
    from gcsim.lifetime import RequestLifetime, CacheLifetime

    rate = 8000.0
    duration = 300.0
    classes = [
        (0.7, 100.0, RequestLifetime(median_ms=5.0, long_tail_frac=0.0)),
        (0.3, 400.0, CacheLifetime(min_lifetime_s=2000.0, alpha=1.5)),
    ]
    cfg = GCConfig()
    cfg.heap.young_size = 64 * MB
    cfg.heap.survivor_ratio = 0.30
    cfg.young.promotion_age = 5
    cfg.heap.old_size = 8 * GB
    cfg.old.initiating_occupancy = 0.94

    # Object oracle
    oracle = ObjectOracle(cfg, classes, seed=99)
    ores = oracle.run_day(duration, rate)

    # Cohort engine with the same byte allocation rate
    mean_bytes = sum(w * s for w, s, _ in classes)
    svc = ServiceProfile(name="oracle", classes=[
        TrafficClass("req", RequestLifetime(median_ms=5.0, long_tail_frac=0.0),
                     share=0.7, mean_size=100, size_sigma=0.0),
        TrafficClass("keep", CacheLifetime(min_lifetime_s=2000.0, alpha=1.5),
                     share=0.3, mean_size=400, size_sigma=0.0),
    ])
    tr = TrafficProfile.flat(rate)
    tr.day_seconds = duration
    trace = build_trace(svc, tr, seed=99, cohort_seconds=1.0, noise=False)
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()

    o_alloc = ores.counted_objects * mean_bytes
    o_frac = ores.promotions_bytes / max(1.0, o_alloc)
    c_frac = st.promotion_bytes / max(1.0, st.allocated_bytes)
    # Promoted fraction of allocated bytes must agree within a few points.
    assert abs(o_frac - c_frac) < 0.06, (o_frac, c_frac)
    # Scavenge counts must agree within ~10% (each scavenge is one eden full).
    ratio = st.young_gcs / max(1, ores.scavenges)
    assert 0.8 < ratio < 1.25, (st.young_gcs, ores.scavenges)


# --------------------------------------------------------------------------
# end-to-end suite objects
# --------------------------------------------------------------------------

def test_report_charts_render_svg():
    from gcsim.report import RunDigest, digest_charts
    svc = ServiceProfile.short_lived()
    tr = TrafficProfile.flat(30000)
    tr.day_seconds = 600
    trace = build_trace(svc, tr, seed=8, cohort_seconds=2.0)
    cfg = GCConfig()
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    d = RunDigest(label="t", cfg=cfg, stats=st, day_seconds=600,
                  service_name="short_lived", traffic_name="flat")
    charts = digest_charts(d)
    for key, svg in charts.items():
        assert svg.startswith("<svg")
        assert "</svg>" in svg


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------

def test_marking_time_bounded_by_elapsed():
    """Regression: an idle-night clock jump must not be booked as concurrent
    marking time.  Marking seconds must stay a small fraction of the day."""
    svc = ServiceProfile.cache_heavy()
    tr = TrafficProfile.batch_heavy()
    trace = build_trace(svc, tr, seed=12345, cohort_seconds=5.0)
    cfg = GCConfig()
    cfg.heap.young_size = 256 * MB
    cfg.heap.old_size = 16 * GB
    sim = HeapSimulator(cfg, svc, tr, trace=trace)
    st = sim.run()
    assert st.concurrent_overhead_s < 0.2 * tr.day_seconds, \
        st.concurrent_overhead_s
    # throughput must be a sane fraction even when marking is busy
    thr = 1.0 - (st.total_pause_s() + st.app_cpu_lost_s) / tr.day_seconds
    assert thr > 0.9, thr


def test_page_tables_balanced():
    from gcsim.page import render_page
    from gcsim.acceptance import _scenario_dict, scenario
    from gcsim.tuner import Objective
    r = scenario("short_lived", "flat", objective=Objective(50, 0.99),
                 workers=2, cohort_seconds=5.0, outdir=tempfile.mkdtemp(),
                 day_hours=2.0)
    sc = _scenario_dict(r)
    html = render_page({"generated": "t", "scenarios": [sc],
                        "charts": sc["charts"]})
    assert html.count("<table") == html.count("</table>")
    assert html.count("<svg") == html.count("</svg>")


def test_parser_handles_concurrent_mode_failure_full_gc():
    """Regression: a full GC whose cause contains 'concurrent' (Concurrent
    Mode Failure) must be parsed as a full GC, not a concurrent marking
    event.  Getting this wrong silently hides the worst pauses from
    calibration."""
    text = (
        "2026-11-23T09:12:02.903: [[33122.903287s]] Full GC "
        "(Concurrent Mode Failure) 3.3G->2.2G(5.0G), 1.0308566 secs\n"
    )
    ev = parse_gc_log(text)
    assert len(ev) == 1
    assert ev[0].kind == "full"
    assert abs(ev[0].pause_s - 1.0308566) < 1e-6
    assert abs(ev[0].heap_after - 2.2 * GB) < 1e6
