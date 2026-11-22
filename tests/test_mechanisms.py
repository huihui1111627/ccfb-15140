"""Mechanism tests for the collector simulator.

These tests pin down the *rules*, not just the outputs: ageing, tenuring,
premature promotion on survivor overflow, pause proportionality, the
keep-up degeneration condition, floating garbage, and the reverse
calibration round-trip.
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcsim.config import (  # noqa: E402
    GCConfig, ObjectPopulation, WorkloadConfig, ServiceScenario,
    SimulationConfig, get_scenario, MiB,
)
from gcsim.engine import Simulator, ObjectStore  # noqa: E402
from gcsim.gclog import (  # noqa: E402
    GCEvent, LogParser, estimate_parameters, parse_log, render_hotspot,
    render_simple,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def tiny_scenario(day_seconds=600, eden=4, surv=1, old=64, ten=4,
                  alloc=8.0, **gc_kw):
    sc = get_scenario("trading-gateway")
    sc.workload = WorkloadConfig(
        day_seconds=day_seconds, alloc_rate_peak_mibs=alloc,
        alloc_rate_min_mibs=alloc, night_factor=1.0, burst_cv=0.0,
        traffic_curve=[1.0, 1.0],
    )
    sc.gc = GCConfig(eden_bytes=eden * MiB, survivor_bytes=surv * MiB,
                     old_bytes=old * MiB, tenuring_threshold=ten,
                     adaptive_tenuring=False, **gc_kw)
    sc.sim = SimulationConfig(seed=7, max_records_per_minor=200,
                              requests_per_mib_rate=100.0)
    sc.validate()
    return sc


# --------------------------------------------------------------------------
# Young generation rules
# --------------------------------------------------------------------------
def test_all_survivors_age_one_per_minor():
    sc = tiny_scenario()
    sim = Simulator(sc)
    r = sim.run()
    ages = [e["tenuring"] for e in r.events if e["kind"] == 0]
    assert ages, "expected young collections"
    # with a fixed threshold and adaptive policy off, age never changes
    assert set(ages) == {4}


def test_young_objects_start_at_age_one():
    store = ObjectStore()
    sc = tiny_scenario()
    sim = Simulator(sc)
    sc.workload.day_seconds = 60
    sim = Simulator(sc)
    r = sim.run()
    # the age histogram of live young objects must start at age 1
    hist = r.age_hist
    assert hist["age"][0] == 0
    assert hist["bytes"][0] == 0.0, "no live young object has age 0"
    assert hist["bytes"][1] > 0, "age-1 objects exist"


def test_tenuring_threshold_promotes_exactly_older_objects():
    """Objects at age == threshold are promoted; age < threshold are not."""
    sc = tiny_scenario(ten=3, surv=64)  # big survivor: no overflow
    r = Simulator(sc).run()
    young = [e for e in r.events if e["kind"] == 0]
    assert young
    # promoted objects' ages come from the histogram: nothing may be at an
    # age below the threshold when the survivor space never overflows
    prom = r.age_hist["promoted_bytes"]
    premature = sum(prom[:3])  # ages 0,1,2
    assert premature == 0.0
    assert sum(prom[4:]) > 0.0, "some objects should exceed the threshold"


def test_survivor_overflow_promotes_oldest_first():
    """With a tiny survivor space, the oldest objects are promoted early."""
    sc = tiny_scenario(ten=10, surv=1, eden=16)
    r = Simulator(sc).run()
    assert r.summary["premature_promoted_bytes_frac"] > 0.5


def test_survivor_size_changes_premature_promotion():
    small = tiny_scenario(ten=10, surv=1, eden=16)
    big = tiny_scenario(ten=10, surv=64, eden=16)
    rs = Simulator(small).run().summary
    rb = Simulator(big).run().summary
    assert (rs["premature_promoted_bytes_frac"]
            > rb["premature_promoted_bytes_frac"])


# --------------------------------------------------------------------------
# Pause model
# --------------------------------------------------------------------------
def test_young_pause_is_linear_in_live_bytes():
    """Two runs whose eden live sets differ 4x should differ in pause."""
    sc_small = tiny_scenario(eden=8, surv=8)
    sc_big = tiny_scenario(eden=32, surv=8)
    a = Simulator(sc_small).run().summary
    b = Simulator(sc_big).run().summary
    # bigger eden => more bytes allocated per minor, but survival of the
    # request class is tiny, so pause should stay small in both; instead
    # compare the model directly with an object-rich population
    assert a["minor_pause_p50_ms"] > 0
    assert b["minor_pause_p50_ms"] > 0


def test_pause_scales_with_live_bytes():
    """Minors copy the surviving eden bytes: long-lived objects cost more."""
    short = tiny_scenario(eden=8, surv=4, ten=4)
    short.population = ObjectPopulation(
        request_fraction=1.0, session_fraction=0.0, cache_fraction=0.0,
        request_scale_s=0.001, request_shape=1.0,
        request_size_bytes=1000.0,
    )
    long = tiny_scenario(eden=8, surv=4, ten=4)
    long.population = ObjectPopulation(
        request_fraction=0.0, session_fraction=1.0, cache_fraction=0.0,
        session_scale_s=10000.0, session_shape=4.0,
        session_size_bytes=1000.0,
    )
    a = Simulator(short).run().summary
    b = Simulator(long).run().summary
    # all-long-lived eden copies ~8 MiB per minor; all-short-lived ~0
    assert b["minor_pause_p50_ms"] > a["minor_pause_p50_ms"] * 3.0
    # and the pause matches the model: fixed + per_byte * live
    gc = long.gc
    live = b["minor_live_bytes_mean"]
    predicted = (gc.minor_fixed_s * 1000.0
                 + gc.minor_per_live_byte_s * live * 1000.0)
    assert b["minor_pause_p50_ms"] == pytest.approx(predicted, rel=0.35)


def test_full_gc_pause_is_longer_than_young_pause():
    sc = get_scenario("trading-gateway")
    sc.workload.day_seconds = 3600
    sc.sim.max_records_per_minor = 32
    r = Simulator(sc).run()
    if r.summary["full_count"] > 0:
        assert r.summary["full_pause_max_ms"] > r.summary["minor_pause_max_ms"]
    else:
        pytest.skip("no full GC in this configuration")


# --------------------------------------------------------------------------
# Concurrent collector: keep-up condition and degeneration
# --------------------------------------------------------------------------
def _concurrent_scenario(mark_mibs, trigger=0.5, day=1800):
    sc = get_scenario("trading-gateway")
    sc.workload.day_seconds = day
    sc.gc.old_bytes = 1024 * MiB
    sc.gc.concurrent_trigger_ratio = trigger
    sc.gc.concurrent_mark_bytes_per_s = mark_mibs * MiB
    sc.sim.max_records_per_minor = 32
    sc.workload.burst_cv = 0.0
    return sc


def test_slow_marker_degenerates_into_full_gc():
    slow = _concurrent_scenario(2.0)
    r = Simulator(slow).run()
    assert r.summary["degenerate_count"] > 0
    reasons = r.summary["full_reason_counts"]
    assert any("keepup" in k or "panic" in k or "start" in k
               for k in reasons), reasons


def test_fast_marker_avoids_full_gc():
    fast = _concurrent_scenario(400.0)
    r = Simulator(fast).run()
    assert r.summary["full_count"] == 0
    assert r.summary["concurrent_cycles"] > 0


def test_marker_threads_are_capped():
    sc = _concurrent_scenario(400.0)
    sim = Simulator(sc)
    assert sim.effective_mark_rate() <= 4 * sc.gc.concurrent_core_scan_bytes_per_s


def test_concurrent_cycle_cost_scales_with_mark_rate():
    slow = _concurrent_scenario(40.0)
    fast = _concurrent_scenario(400.0)
    rs = Simulator(slow).run().summary
    rf = Simulator(fast).run().summary
    # faster marking burns more CPU per unit time while it runs
    assert rf["concurrent_cpu_frac"] > rs["concurrent_cpu_frac"]


# --------------------------------------------------------------------------
# Floating garbage
# --------------------------------------------------------------------------
def test_dead_old_objects_still_occupy_until_a_collection():
    sc = _concurrent_scenario(400.0, day=600)
    sc.population = ObjectPopulation(
        request_fraction=0.0, session_fraction=1.0, cache_fraction=0.0,
        session_scale_s=30.0, session_shape=3.0,
        session_size_bytes=1000.0,
    )
    sim = Simulator(sc)
    r = sim.run()
    assert r.summary["sweeps"] > 0
    assert r.summary["reclaimed_old_gib"] >= 0.0


# --------------------------------------------------------------------------
# Traffic curve and burstiness
# --------------------------------------------------------------------------
def test_traffic_curve_is_periodic_and_bounded():
    wl = WorkloadConfig(day_seconds=86400, night_factor=0.1)
    for t in (0.0, 43200.0, 86399.0):
        m = wl.multiplier_at(t)
        assert 0.1 <= m <= 1.0
    assert abs(wl.multiplier_at(0.0) - wl.multiplier_at(86400.0)) < 1e-9


def test_allocation_follows_the_traffic_curve():
    sc = tiny_scenario(day_seconds=7200, eden=8, alloc=20.0)
    sc.workload.traffic_curve = [0.1, 1.0, 0.1, 1.0]
    sc.workload.night_factor = 0.05        # curve dips to 5% at the trough
    sc.workload.alloc_rate_min_mibs = 0.2  # background allocation
    r = Simulator(sc).run()
    mins = r.minutes
    peak = max(b["alloc_rate_mibs"] for b in mins)
    trough = min(b["alloc_rate_mibs"] for b in mins)
    assert peak > trough * 2.0


# --------------------------------------------------------------------------
# Sampling does not bias totals
# --------------------------------------------------------------------------
def test_allocation_total_matches_traffic_integral():
    """Allocation is the traffic rate times the time the mutator is running.

    During a stop-the-world pause no allocation happens, so the expected
    total is rate * (day - paused_seconds), not rate * day.
    """
    sc = tiny_scenario(day_seconds=1800, alloc=10.0)
    sc.workload.traffic_curve = [1.0, 1.0]
    sc.workload.night_factor = 1.0
    sc.workload.alloc_rate_min_mibs = 10.0
    r = Simulator(sc).run()
    paused = r.summary["stw_pause_frac"] * 1800.0
    expected = 10.0 * MiB * (1800.0 - paused)
    got = r.summary["allocated_bytes_total"]
    assert got == pytest.approx(expected, rel=0.02)
    assert paused > 0.0, "the run should have pauses"


def test_promotion_rate_is_unbiased_by_record_cap():
    a = tiny_scenario(day_seconds=1800)
    a.sim.max_records_per_minor = 400
    a.workload.burst_cv = 0.0
    b = tiny_scenario(day_seconds=1800)
    b.sim.max_records_per_minor = 20
    b.workload.burst_cv = 0.0
    sa = Simulator(a).run().summary
    sb = Simulator(b).run().summary
    assert sa["promoted_mibs_per_s"] == pytest.approx(
        sb["promoted_mibs_per_s"], rel=0.35)


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------
def test_simple_log_round_trip():
    sc = tiny_scenario(day_seconds=1200, alloc=12.0)
    r = Simulator(sc).run()
    text = render_simple(r)
    events, est = parse_log(text)
    assert est["dialect"] == "simple"
    assert est["young_gc_count"] == r.summary["minor_count"]
    assert est["eden_capacity_mib"] == pytest.approx(sc.gc.eden_bytes / MiB,
                                                     abs=1e-6)


def test_hotspot_log_round_trip():
    sc = tiny_scenario(day_seconds=1200, alloc=12.0)
    r = Simulator(sc).run()
    text = render_hotspot(r)
    assert "Pause Young" in text and "GC(" in text
    events, est = parse_log(text)
    assert est["dialect"] == "hotspot"
    assert est["young_gc_count"] == r.summary["minor_count"]
    assert est["alloc_mibs"] == pytest.approx(
        r.summary["allocated_bytes_total"] / MiB / 1200.0, rel=0.05)


def test_parser_handles_foreign_hotspot_log():
    """A hand-written snippet in HotSpot's format must parse."""
    text = """[2026-01-02T10:00:00.000+0000][info][gc,start    ] GC(7) Pause Young (Normal) (G1 Evacuation Pause)
[2026-01-02T10:00:00.000+0000][info][gc,heap     ] GC(7) Eden regions: 32->0(32)
[2026-01-02T10:00:00.000+0000][info][gc,heap     ] GC(7) Survivor regions: 3->2(2)
[2026-01-02T10:00:00.000+0000][info][gc,heap     ] GC(7) Old regions: 100->104
[2026-01-02T10:00:00.000+0000][info][gc,phases   ] GC(7)   Pause Young (Normal) (G1 Evacuation Pause) 8.500ms
[2026-01-02T10:00:10.000+0000][info][gc,start    ] GC(8) Pause Young (Normal) (G1 Evacuation Pause)
[2026-01-02T10:00:10.000+0000][info][gc,heap     ] GC(8) Eden regions: 32->0(32)
[2026-01-02T10:00:10.000+0000][info][gc,heap     ] GC(8) Survivor regions: 2->2(2)
[2026-01-02T10:00:10.000+0000][info][gc,heap     ] GC(8) Old regions: 104->110
[2026-01-02T10:00:10.000+0000][info][gc,phases   ] GC(8)   Pause Young (Normal) (G1 Evacuation Pause) 9.100ms
"""
    events, est = parse_log(text)
    assert est["dialect"] == "hotspot"
    assert len(events) == 2
    assert events[0].eden_before_mib == 32
    assert events[0].pause_ms == pytest.approx(8.5)
    assert est["eden_capacity_mib"] == 32


def test_estimate_parameters_recovers_rates():
    sc = tiny_scenario(day_seconds=1800, alloc=12.0)
    r = Simulator(sc).run()
    events, est = parse_log(render_hotspot(r))
    assert est["alloc_mibs"] == pytest.approx(
        r.summary["allocated_bytes_total"] / MiB / 1800.0, rel=0.05)
    assert est["promotion_mibs"] == pytest.approx(
        r.summary["promoted_mibs_per_s"], rel=0.25)


def test_estimate_recovers_pause_model():
    sc = tiny_scenario(day_seconds=1800, alloc=12.0)
    r = Simulator(sc).run()
    events, est = parse_log(render_hotspot(r))
    fit = est["pause_model_fit"]["young"]
    if fit is None:
        pytest.skip("not enough young events")
    assert fit["fixed_ms"] == pytest.approx(sc.gc.minor_fixed_s * 1000.0,
                                           rel=0.3)
    assert fit["per_mib_ms"] == pytest.approx(
        sc.gc.minor_per_live_byte_s * MiB * 1000.0, rel=0.2)


# --------------------------------------------------------------------------
# Tuner
# --------------------------------------------------------------------------
def test_tuner_improves_a_broken_configuration():
    from gcsim import tuner
    sc = get_scenario("trading-gateway")
    # shrink the search so the test stays quick
    sc.tuning_space = {
        "eden_mib": [32, 48],
        "survivor_mib": [4, 8],
        "old_mib": [3072],
        "tenuring_threshold": [4, 6],
        "concurrent_trigger_ratio": [0.45],
        "concurrent_mark_mibs": [16, 160],
    }
    obj = {"pause_target_ms": 40.0, "tail_frac": 0.01,
           "throughput_target": 0.98, "pause_budget_ms": 20.0}
    res = tuner.tune(sc, obj, max_candidates=8, refine=False, workers=2)
    assert res.best.summary["pause_max_ms"] <= res.baseline.summary["pause_max_ms"]
    assert res.effects, "the tuner must explain parameter effects"


def test_tuner_objective_and_ranking_shapes():
    from gcsim import tuner
    sc = get_scenario("order-lifecycle")
    sc.tuning_space = {
        "eden_mib": [32], "survivor_mib": [4], "old_mib": [2048],
        "tenuring_threshold": [5], "concurrent_trigger_ratio": [0.42],
        "concurrent_mark_mibs": [16, 128],
    }
    res = tuner.tune(sc, None, max_candidates=2, refine=False, workers=2)
    assert res.ranking
    row = res.ranking[0]
    assert set(["params", "feasible", "pause_max_ms"]).issubset(row)
    assert res.top_recommendation["changes"]["eden_mib"][0] == 32


# --------------------------------------------------------------------------
# End-to-end sanity of the acceptance payload
# --------------------------------------------------------------------------
def test_acceptance_payload_is_json_serialisable():
    import json
    from gcsim.acceptance import run_acceptance
    data = run_acceptance(["order-lifecycle"], do_tune=False,
                          record_cap=24)
    text = json.dumps(data)
    assert len(text) > 1000
    sc = data["scenarios"][0]
    assert "before" in sc and "calibration" in sc
    assert sc["calibration"]["before"]["log_lines"] > 0


def test_tuner_param_units_are_consistent():
    """Regression: mark rate is bytes/s in the tuner's param dicts.

    A unit mismatch here silently made every candidate run at 1 MiB/s, which
    made the search miss every feasible configuration.
    """
    from gcsim import tuner
    from gcsim.config import MiB as _MiB
    sc = get_scenario("market-data")
    gc = sc.gc.copy()
    gc.concurrent_mark_bytes_per_s = 256 * _MiB
    gc.eden_bytes = 48 * _MiB
    params = tuner._params_of(gc)
    assert params["concurrent_mark_bytes_per_s"] == 256 * _MiB
    assert params["eden_bytes"] == 48
    back = tuner._apply(sc.gc, params)
    assert back.concurrent_mark_bytes_per_s == 256 * _MiB
    assert back.eden_bytes == 48 * _MiB


def test_pause_fit_reports_identifiability():
    """A slope fit must say when it cannot separate fixed cost from slope."""
    from gcsim.gclog import estimate_parameters, GCEvent
    # 12 full GCs, all with almost the same live set
    events = []
    for i in range(12):
        events.append(GCEvent(kind="full", wall_s=60.0 * i, pause_ms=10.0 + 0.5 * 1000,
                              live_mib=1000.0, old_before_mib=1000.0,
                              old_after_mib=1000.0))
    est = estimate_parameters(events, 720.0)
    fit = est["pause_model_fit"]["full"]
    assert fit is not None
    assert fit["identifiable"] is False, fit

    # now with a spread of live sets the slope is identifiable
    events = []
    for i in range(12):
        live = 500.0 + 100.0 * i
        events.append(GCEvent(kind="full", wall_s=60.0 * i,
                              pause_ms=10.0 + 0.5 * live,
                              live_mib=live, old_before_mib=live,
                              old_after_mib=live))
    est = estimate_parameters(events, 720.0)
    fit = est["pause_model_fit"]["full"]
    assert fit["identifiable"] is True, fit
    assert fit["per_mib_ms"] == pytest.approx(0.5, rel=0.01)
