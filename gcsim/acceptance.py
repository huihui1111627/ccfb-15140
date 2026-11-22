"""Acceptance harness: full-day runs, tuning, before/after, calibration.

For every scenario this produces:

* before/after pause distributions, maximum pause, tail latency;
* throughput loss, full stop-the-world GC counts and their reasons;
* minute-resolution occupancy series (young, survivor, old, live);
* age histograms of live young objects and of promoted objects;
* promotion composition by object class;
* a *reverse-calibration* pass: render the simulator's GC log in a
  HotSpot-like dialect, parse it with the independent parser, and compare
  the estimated parameters with the settings that produced the log.

The returned document is what the HTML dashboard renders.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .config import (
    GCConfig, ServiceScenario, SimulationConfig, MiB, get_scenario,
)
from .engine import Simulator
from .gclog import parse_log, render_hotspot, render_simple
from . import tuner as tuning


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def run_full_day(scenario: ServiceScenario, gc: Optional[GCConfig] = None,
                 record_cap: int = 120) -> "SimulationResult":
    gc = gc or scenario.gc
    sc = ServiceScenario(
        name=scenario.name, description=scenario.description,
        workload=scenario.workload, population=scenario.population,
        gc=gc, sim=replace(scenario.sim, max_records_per_minor=record_cap),
        tuning_space=scenario.tuning_space)
    return Simulator(sc).run()


# --------------------------------------------------------------------------
# Reverse calibration
# --------------------------------------------------------------------------
def _nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# (display name, truth key, estimate key, unit, relative tolerance)
CALIBRATION_ROWS = [
    ("分配速率", "alloc_mibs", "alloc_mibs", "MiB/s", 0.05),
    ("晋升速率", "promotion_mibs", "promotion_mibs", "MiB/s", 0.15),
    ("eden 容量", "eden_mib", "eden_capacity_mib", "MiB", 0.02),
    ("幸存区容量(单区)", "survivor_mib", "survivor_capacity_mib", "MiB", 0.05),
    ("晋升年龄阈值", "tenuring", "tenuring_threshold", "次", 0.0),
    ("young GC 次数", "minor_count", "young_gc_count", "次", 0.02),
    ("full GC 次数", "full_count", "full_gc_count", "次", 0.05),
    ("退化全堆停顿次数", "degenerate_count", "degenerate_full_gc_count",
     "次", 0.05),
    ("并发周期数", "concurrent_cycles", "concurrent_cycles", "次", 0.15),
    ("并发标记速率", "mark_mibs", "concurrent_mark_mibs", "MiB/s", 0.60),
    ("young 停顿固定开销", "minor_fixed_ms",
     "pause_model_fit.young.fixed_ms", "ms", 0.20),
    ("young 停顿/MiB 拷贝", "minor_per_mib_ms",
     "pause_model_fit.young.per_mib_ms", "ms/MiB", 0.10),
    ("full 停顿固定开销", "full_fixed_ms",
     "pause_model_fit.full.fixed_ms", "ms", 0.30),
    ("full 停顿/MiB 存活", "full_per_mib_ms",
     "pause_model_fit.full.per_mib_ms", "ms/MiB", 0.10),
]


def calibration_report(scenario: ServiceScenario, gc: GCConfig,
                       record_cap: int = 120,
                       log_format: str = "hotspot") -> dict:
    """Simulate, emit a GC log, parse it back, compare with the truth."""
    res = run_full_day(scenario, gc, record_cap=record_cap)
    text = render_simple(res) if log_format == "simple" else render_hotspot(res)
    events, est = parse_log(text)
    second = max(res.summary.get("day_seconds", 1.0), 1.0)
    truth = {
        "alloc_mibs": res.summary["allocated_bytes_total"] / MiB / second,
        "promotion_mibs": res.summary["promoted_mibs_per_s"],
        "eden_mib": gc.eden_bytes / MiB,
        "survivor_mib": gc.survivor_bytes / MiB,
        "tenuring": gc.tenuring_threshold,
        "minor_count": res.summary["minor_count"],
        "full_count": res.summary["full_count"],
        "degenerate_count": res.summary["degenerate_count"],
        "concurrent_cycles": res.summary["concurrent_cycles"],
        "mark_mibs": min(gc.concurrent_mark_bytes_per_s,
                         4.0 * gc.concurrent_core_scan_bytes_per_s) / MiB,
        "minor_fixed_ms": gc.minor_fixed_s * 1000.0,
        "minor_per_mib_ms": gc.minor_per_live_byte_s * MiB * 1000.0,
        "full_fixed_ms": gc.full_fixed_s * 1000.0,
        "full_per_mib_ms": gc.full_per_live_byte_s * MiB * 1000.0,
    }
    rows = []
    for name, tkey, ekey, unit, tol in CALIBRATION_ROWS:
        t = truth[tkey]
        e = _nested(est, ekey)
        # some estimates are simply absent when the phenomenon never occurs
        if e is None:
            rows.append({"parameter": name, "truth": t, "estimated": None,
                         "unit": unit, "abs_error": None, "rel_error": None,
                         "tolerance": tol, "ok": None,
                         "note": "本次运行未出现该现象，无法估计"})
            continue
        rel = (e - t) / t if abs(t) > 1e-12 else (0.0 if abs(e) < 1e-12 else None)
        note = ""
        fit = None
        if ekey.startswith("pause_model_fit."):
            fit = _nested(est, ekey.rsplit(".", 1)[0])
        if isinstance(fit, dict) and not fit.get("identifiable", True):
            note = ("本次运行中该类收集的存活集几乎恒定"
                    f"（{fit.get('x_min', 0):.0f}–{fit.get('x_max', 0):.0f} {unit.split('/')[0]}），"
                    "斜率不可辨识")
            rows.append({
                "parameter": name, "truth": t, "estimated": e, "unit": unit,
                "abs_error": e - t, "rel_error": rel, "tolerance": tol,
                "ok": None, "note": note,
            })
            continue
        rows.append({
            "parameter": name, "truth": t, "estimated": e, "unit": unit,
            "abs_error": e - t, "rel_error": rel, "tolerance": tol,
            "ok": (abs(rel) <= tol) if rel is not None else None,
            "note": note,
        })
    applicable = [r for r in rows if r["ok"] is not None]
    return {
        "scenario": scenario.name,
        "log_format": log_format,
        "log_lines": text.count("\n"),
        "log_bytes": len(text.encode("utf-8")),
        "log_excerpt": "\n".join(text.splitlines()[:16]),
        "events_parsed": len(events),
        "dialect": est.get("dialect"),
        "rows": rows,
        "passed": sum(1 for r in applicable if r["ok"]),
        "total": len(applicable),
        "estimates": {
            "alloc_mibs": est["alloc_mibs"],
            "promotion_mibs": est["promotion_mibs"],
            "live_set_mib": est["live_set_mib"],
            "young_capacity_mib": est["young_capacity_mib"],
            "tenuring_threshold": est["tenuring_threshold"],
            "concurrent_mark_mibs": est["concurrent_mark_mibs"],
            "pause_ms": est["pause_ms"],
            "pause_model_fit": est.get("pause_model_fit"),
            "throughput_loss_frac": est["throughput_loss_frac"],
            "occupancy": est["occupancy"],
            "full_reasons": est.get("full_reasons", {}),
        },
        "truth": truth,
    }


# --------------------------------------------------------------------------
# Payload helpers
# --------------------------------------------------------------------------
EVENT_KEYS = ("t", "kind", "kind_name", "pause_ms", "old_before", "old_after",
              "promoted_bytes", "live_bytes", "tenuring", "reason")


def _round_event(e: dict) -> dict:
    out = {}
    for k in EVENT_KEYS:
        if k not in e:
            continue
        v = e[k]
        if isinstance(v, float):
            if k == "t":
                out[k] = round(v, 2)
            elif k in ("pause_ms",):
                out[k] = round(v, 3)
            elif k in ("reason", "kind_name"):
                out[k] = v
            else:
                out[k] = int(v)
        else:
            out[k] = v
    return out


def _thin_events(events: List[dict], max_events: int = 6000) -> List[dict]:
    """Downsample the event stream for the web payload.

    Young collections are thinned uniformly; every full / concurrent event
    is kept, because those are the ones the dashboard annotates.
    """
    if len(events) > max_events:
        keep_kinds = {1, 2, 3}
        urgent = [e for e in events if e["kind"] in keep_kinds]
        rest = [e for e in events if e["kind"] not in keep_kinds]
        room = max(max_events - len(urgent), 1)
        stride = max(1, len(rest) // room)
        events = sorted(urgent + rest[::stride], key=lambda e: e["t"])
    return [_round_event(e) for e in events]


MINUTE_KEYS = ("m", "alloc_bytes", "promoted_bytes", "minor_count",
               "full_count", "conc_cycles", "pause_ms", "max_pause_ms",
               "eden", "survivor", "old", "heap", "live", "alloc_rate_mibs",
               "promoted_mibs", "req_rate", "full_pause_ms", "degen",
               "tenuring")


def _thin_minutes(minutes: List[dict], every: int = 1) -> List[dict]:
    out = []
    for b in minutes[::every]:
        row = {}
        for k in MINUTE_KEYS:
            v = b.get(k, 0)
            if isinstance(v, float):
                # keep precision where the UI needs it, integers elsewhere
                row[k] = round(v, 2) if k in ("alloc_rate_mibs",
                                              "promoted_mibs") else int(v)
            else:
                row[k] = v
        out.append(row)
    return out


def _scenario_payload(r: "SimulationResult") -> dict:
    return {
        "gc": r.gc,
        "summary": r.summary,
        "minutes": _thin_minutes(r.minutes, every=(
            1 if len(r.minutes) <= 1500 else 2)),
        "events": _thin_events(r.events),
        "pause_hist": r.pause_hist,
        "age_hist": {
            "age": r.age_hist.get("age", []),
            "bytes": [round(x, 1) for x in r.age_hist.get("bytes", [])],
            "promoted_bytes": [round(x, 1)
                               for x in r.age_hist.get("promoted_bytes", [])],
        } if r.age_hist else {},
        "class_stats": r.class_stats,
    }


COMPARE_KEYS = [
    ("pause_max_ms", "最大停顿", "ms"),
    ("pause_p99_ms", "P99 停顿", "ms"),
    ("pause_p95_ms", "P95 停顿", "ms"),
    ("pause_p50_ms", "P50 停顿", "ms"),
    ("tail_over_budget_frac", "超预算请求占比", ""),
    ("throughput_loss_frac", "吞吐损失", ""),
    ("full_count", "全堆停顿次数", "次"),
    ("degenerate_count", "并发退化为全堆停顿", "次"),
    ("concurrent_cycles", "并发回收周期", "次"),
    ("minor_count", "新生代回收次数", "次"),
    ("budget_breaches", "超过停顿预算的次数", "次"),
    ("promoted_mibs_per_s", "晋升速率", "MiB/s"),
    ("peak_old_frac", "老年代峰值占用", ""),
    ("mean_added_latency_ms", "平均附加延迟", "ms"),
    ("p99_latency_ms", "P99 请求延迟", "ms"),
]


def _compare(before: dict, after: dict) -> List[dict]:
    out = []
    for key, label, unit in COMPARE_KEYS:
        b = before.get(key)
        a = after.get(key)
        delta = (a - b) if (b is not None and a is not None) else None
        rel = (delta / b) if (delta is not None and abs(b) > 1e-12) else None
        out.append({"key": key, "label": label, "unit": unit,
                    "before": b, "after": a, "delta": delta, "rel": rel})
    return out


# --------------------------------------------------------------------------
# Top-level acceptance run
# --------------------------------------------------------------------------
def run_acceptance(scenario_names: Optional[List[str]] = None,
                   objective: Optional[dict] = None,
                   do_tune: bool = True,
                   workers: Optional[int] = None,
                   record_cap: int = 120,
                   max_candidates: int = 24,
                   progress=None) -> dict:
    names = scenario_names or ["trading-gateway", "market-data",
                               "order-lifecycle"]
    obj = dict(tuning.DEFAULT_OBJECTIVE)
    if objective:
        obj.update(objective)
    # The objective is a hard constraint on the *day*: the pause budget and
    # the throughput floor are the same for every scenario.
    out_scenarios = []
    t_all = time.time()
    for idx, name in enumerate(names):
        scen = get_scenario(name)
        obj = dict(obj)
        obj["max_heap_mib"] = scen.heap_budget_mib
        scen.sim = replace(scen.sim, pause_budget_ms=obj["pause_budget_ms"])
        if progress:
            progress(idx, len(names), f"{name}: 基线全天仿真")
        before_res = run_full_day(scen, scen.gc, record_cap=record_cap)
        tuning_res = None
        best_gc = scen.gc
        if do_tune:
            if progress:
                progress(idx, len(names), f"{name}: 参数搜索")
            def tune_progress(i, n):
                if progress:
                    progress(idx, len(names),
                             f"{name}: 参数搜索 {i}/{n}")
            tuning_res = tuning.tune(scen, obj, max_candidates=max_candidates,
                                     refine=True, workers=workers,
                                     final_record_cap=record_cap,
                                     progress=tune_progress)
            best_gc = tuning_res.best.gc
        if progress:
            progress(idx, len(names), f"{name}: 调优后全天仿真")
        after_res = run_full_day(scen, best_gc, record_cap=record_cap)
        if progress:
            progress(idx, len(names), f"{name}: 反向校准")
        calib_before = calibration_report(scen, scen.gc, record_cap=record_cap)
        calib_after = calibration_report(scen, best_gc, record_cap=record_cap)
        out_scenarios.append({
            "name": name,
            "description": scen.description,
            "objective": obj,
            "population": scen.population.to_dict(),
            "workload": scen.workload.to_dict(),
            "before": _scenario_payload(before_res),
            "after": _scenario_payload(after_res),
            "comparison": _compare(before_res.summary, after_res.summary),
            "tuning": None if tuning_res is None else {
                "objective": tuning_res.objective,
                "baseline": tuning_res.baseline.key_metrics(),
                "best": tuning_res.best.key_metrics(),
                "recommendation": tuning_res.top_recommendation,
                "ranking": tuning_res.ranking,
                "effects": tuning_res.effects,
                "wall_s": tuning_res.wall_s,
                "feasible_count": tuning_res.feasible_count,
                "candidates": len(tuning_res.candidates),
            },
            "calibration": {"before": calib_before, "after": calib_after},
        })
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_s": time.time() - t_all,
        "objective": obj,
        "scenarios": out_scenarios,
    }


def summary_markdown(acceptance: dict) -> str:
    """Render a compact markdown summary of an acceptance document."""
    lines = ["# 验收摘要", ""]
    obj = acceptance.get("objective", {})
    lines.append(
        f"目标：最长停顿 ≤ {obj.get('pause_target_ms')} ms，吞吐损失 ≤ "
        f"{(1 - obj.get('throughput_target', 0)) * 100:.1f}%，"
        f"超预算请求 ≤ {obj.get('tail_frac', 0) * 100:.2f}%。")
    lines.append("")
    lines.append("| 服务 | 堆预算 | 最长停顿 (前→后) | P99 停顿 | 吞吐损失 | 全堆停顿 | 退化 | 反向校准 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sc in acceptance.get("scenarios", []):
        b, a = sc["before"]["summary"], sc["after"]["summary"]
        budget = sc.get("objective", {}).get("max_heap_mib", "—")
        cal = sc["calibration"]["after"]
        lines.append(
            f"| {sc['name']} | {budget} MiB | "
            f"{b['pause_max_ms']:.0f} → {a['pause_max_ms']:.1f} ms | "
            f"{b['pause_p99_ms']:.1f} → {a['pause_p99_ms']:.1f} ms | "
            f"{b['throughput_loss_frac'] * 100:.2f}% → "
            f"{a['throughput_loss_frac'] * 100:.2f}% | "
            f"{b['full_count']} → {a['full_count']} | "
            f"{b['degenerate_count']} → {a['degenerate_count']} | "
            f"{cal['passed']}/{cal['total']} |")
    lines.append("")
    lines.append("## 参数变化")
    lines.append("")
    names = {"eden_mib": "eden", "survivor_mib": "survivor",
             "old_mib": "老年代", "tenuring_threshold": "阈值",
             "concurrent_trigger_ratio": "触发比", "concurrent_mark_mibs": "标记速率"}
    for sc in acceptance.get("scenarios", []):
        tun = sc.get("tuning") or {}
        rec = tun.get("recommendation") or {}
        ch = rec.get("changes", {})
        changed = [f"{names.get(k, k)} {v[0]:g}→{v[1]:g}"
                   for k, v in ch.items() if abs(v[1] - v[0]) > 1e-9]
        lines.append(f"* **{sc['name']}**：" + ("，".join(changed) if changed else "无"))
    lines.append("")
    lines.append("## 反向校准（调优后）")
    lines.append("")
    lines.append("| 服务 | 分配速率 | 晋升速率 | eden | 阈值 | young 停顿模型 R² |")
    lines.append("|---|---|---|---|---|---|")
    for sc in acceptance.get("scenarios", []):
        cal = sc["calibration"]["after"]
        e = cal["estimates"]
        fit = ((e.get("pause_model_fit") or {}).get("young") or {})
        r2 = fit.get("r2")
        r2_text = f"{r2:.3f}" if isinstance(r2, (int, float)) else "—"
        lines.append(
            f"| {sc['name']} | {e['alloc_mibs']:.1f} MiB/s | "
            f"{e['promotion_mibs']:.2f} MiB/s | "
            f"{e['young_capacity_mib']:.0f} MiB | "
            f"{e['tenuring_threshold']} | {r2_text} |")
    lines.append("")
    lines.append("通过数不计入“不可辨识”项：当某类收集的存活集几乎恒定、或调优后不再"
                 "发生 full GC 时，对应的斜率/模型本来就无法从日志估计。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the full acceptance suite")
    ap.add_argument("--out", default="reports/acceptance.json")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--record-cap", type=int, default=120)
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--scenario", action="append")
    ap.add_argument("--smoke", action="store_true",
                    help="quick run: shortened day, tiny search (for CI)")
    args = ap.parse_args()
    if args.smoke:
        # A smoke run is not a substitute for acceptance: it simulates a
        # shortened day and a 6-point search, to check the plumbing only.
        _orig = get_scenario

        def shortened(name):
            sc = _orig(name)
            sc.workload.day_seconds = 7200
            return sc

        globals()["get_scenario"] = shortened
        args.record_cap = min(args.record_cap, 32)
        args.max_candidates = min(args.max_candidates, 6)

    def show(i, n, msg):
        print(f"[{i + 1}/{n}] {msg}", flush=True)

    data = run_acceptance(args.scenario, do_tune=not args.no_tune,
                          workers=args.workers, record_cap=args.record_cap,
                          max_candidates=args.max_candidates, progress=show)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"wrote {args.out} ({len(data['scenarios'])} scenarios, "
          f"{data['wall_s']:.1f}s)")
