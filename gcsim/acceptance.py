"""Acceptance experiment suite.

Runs the scenarios the acceptance criteria call for and writes the artefacts:

1. For several services (different lifetime mixes, hence different promotion
   behaviour) over several traffic curves, simulate a whole day with the
   default parameters and a tuned set.
2. Produce pause distribution, maximum pause, throughput loss, full-GC count,
   and the before/after comparison.
3. Write a GC log from the simulation, then run the reverse calibration and
   compare the estimate with the parameters actually used.
4. Emit JSON + SVG charts consumed by the web page, plus a Markdown summary.

The suite is deliberately explicit about what it does rather than clever, so
the output can be inspected and trusted.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import gclog
from .calibrate import CalibrationReport, calibrate_from_text
from .config import GCConfig, MB, GB, ServiceProfile, TrafficProfile
from .report import (
    RunDigest, digest_charts, markdown_table, svg_bars, write_json,
)
from .simulator import HeapSimulator, RunStats
from .tuner import Objective, PointResult, TuningResult, tune
from .workload import build_trace

HOUR = 3600.0

# Each scenario pairs a service with a traffic curve and a heap budget.  The
# budget is part of the scenario because it is part of reality: a cache-heavy
# service holding a 1.5 GB live set cannot meet a 1% throughput target on a
# 6 GB heap no matter how the three knobs are set, and the tool should say so
# rather than pretend a tuning exists.  Objectives are chosen per service to be
# demanding but achievable at its budget, so the suite demonstrates both a
# successful tune and an honest infeasible result.
#
#   (service, traffic, old_gen_gb, max_pause_ms, min_throughput)
SERVICES = [
    ("trading_front_end", "trading_day", 6, 50.0, 0.99),
    ("session_heavy", "trading_day", 16, 50.0, 0.99),
    # A promotion-heavy service: the marking threads alone cost ~1% of
    # throughput, so 98.5% is the demanding-but-achievable target here.
    ("cache_heavy", "batch_heavy", 16, 50.0, 0.985),
    ("short_lived", "flat", 6, 50.0, 0.99),
    # Deliberately under-provisioned: the live set nearly fills old, and no
    # young/age/initiating combination can meet the target.  Shows the tool
    # reporting infeasibility and naming the binding constraint.
    ("cache_heavy", "trading_day", 3, 50.0, 0.995),
]
TRAFFIC_NAMES = {"trading_day": "trading_day", "batch_heavy": "batch_heavy",
                 "flat": "flat"}


@dataclass
class ScenarioResult:
    service: str
    traffic: str
    objective: Objective
    baseline_digest: RunDigest
    tuned_digest: Optional[RunDigest]
    tuning: Optional[TuningResult]
    calibration: Optional[dict]
    charts: Dict[str, str]
    gc_log_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _run(service_name: str, traffic_name: str, cfg: GCConfig,
         seed: int, cohort_seconds: float,
         log_path: Optional[str] = None,
         day_hours: Optional[float] = None,
         old_gb: Optional[float] = None) -> Tuple[RunDigest, RunStats]:
    cfg = GCConfig.from_dict(cfg.to_dict())
    if old_gb is not None:
        cfg.heap.old_size = int(old_gb * GB)
    service = getattr(ServiceProfile, service_name)()
    traffic = getattr(TrafficProfile, traffic_name)()
    if day_hours:
        traffic.day_seconds = day_hours * 3600.0
    trace = build_trace(service, traffic, seed=seed, cohort_seconds=cohort_seconds)
    writer = None
    sink = None
    if log_path:
        writer, sink = gclog.open_writer(log_path, header=True, start=False)
        writer.set_heap(cfg.heap.young_size, cfg.heap.old_size,
                        cfg.heap.survivor_ratio, cfg.old.initiating_occupancy,
                        cfg.young.promotion_age, cfg.old.mode)
        writer.start()
        writer._emit(f"# model: {service.name} / {traffic_name}")
    sim = HeapSimulator(cfg, service, traffic, seed=seed, trace=trace,
                        log_writer=writer)
    st = sim.run()
    if sink:
        sink.close()
    label = service.name
    d = RunDigest(label=label, cfg=cfg, stats=st, day_seconds=traffic.day_seconds,
                  service_name=service_name, traffic_name=traffic_name)
    return d, st


def _cfg_from_point(p: PointResult) -> GCConfig:
    cfg = GCConfig()
    cfg.heap.young_size = int(p.young_mb * MB)
    cfg.heap.survivor_ratio = p.survivor_ratio
    cfg.young.promotion_age = p.promotion_age
    cfg.old.initiating_occupancy = p.initiating
    cfg.old.mode = p.mode
    return cfg


def scenario(service_name: str, traffic_name: str,
             objective: Optional[Objective] = None,
             seed: int = 12345, cohort_seconds: float = 5.0,
             outdir: str = "reports", workers: Optional[int] = None,
             do_calibration: bool = True,
             day_hours: Optional[float] = None,
             old_gb: Optional[float] = None) -> ScenarioResult:
    objective = objective or Objective()
    t0 = time.perf_counter()
    base_cfg = GCConfig()
    if old_gb is not None:
        base_cfg.heap.old_size = int(old_gb * GB)
    # 1. Baseline day.
    base_digest, base_stats = _run(service_name, traffic_name, base_cfg,
                                   seed, cohort_seconds, day_hours=day_hours)
    # 2. Tune.
    tuning = tune(service_name, traffic_name, objective=objective, seed=seed,
                  workers=workers, cohort_seconds=cohort_seconds,
                  day_hours=day_hours, base=base_cfg)
    notes = list(tuning.notes)
    tuned_digest = None
    if tuning.best is not None:
        cfg = _cfg_from_point(tuning.best)
        cfg.heap.old_size = base_cfg.heap.old_size
        tuned_digest, _ = _run(service_name, traffic_name, cfg, seed,
                               cohort_seconds, day_hours=day_hours)
        tuned_digest.label = f"{base_digest.label} (tuned)"
    else:
        notes.append(
            "No combination of young size, tenure age and initiating "
            "occupancy met both objectives at this heap size. The binding "
            "constraint is the heap budget, not the three knobs -- see the "
            "sensitivity table for how far each knob moves the outcome.")
    # 3. Charts.
    charts = digest_charts(base_digest)
    if tuned_digest:
        charts_t = digest_charts(tuned_digest)
        charts["pause_hist_tuned"] = charts_t["pause_hist"]
        charts["occupancy_tuned"] = charts_t["occupancy"]
        charts["age_hist_tuned"] = charts_t["age_hist"]
        charts["compare_pauses"] = _compare_pause_chart(base_digest, tuned_digest)
    # 4. Reverse calibration from a simulated log.
    cal = None
    log_path = None
    if do_calibration:
        log_path = os.path.join(outdir, "logs",
                                f"{service_name}__{traffic_name}.gc.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Log the *baseline* run so calibration can be checked against the
        # parameters actually used.
        _, log_stats = _run(service_name, traffic_name, base_cfg, seed,
                            cohort_seconds, log_path=log_path,
                            day_hours=day_hours)
        with open(log_path) as f:
            text = f.read()
        rep = calibrate_from_text(text)
        truth = {
            "eden_bytes": base_cfg.heap.eden(),
            "young_size_bytes": base_cfg.heap.young_size,
            "old_size_bytes": base_cfg.heap.old_size,
            "survivor_ratio": base_cfg.heap.survivor_ratio,
            "alloc_rate_bytes_s_true": log_stats.allocated_bytes / _day(log_stats_day(traffic_name), day_hours),
            "promotion_rate_bytes_s_true": log_stats.promotion_bytes / _day(log_stats_day(traffic_name), day_hours),
            "promotion_fraction_true": (log_stats.promotion_bytes /
                                         log_stats.allocated_bytes
                                         if log_stats.allocated_bytes else 0.0),
        }
        cal = {
            "log_path": log_path,
            "log_lines": len(text.splitlines()),
            "estimate": rep.to_dict(),
            "truth": truth,
            "error": _calibration_error(rep, truth),
        }
    res = ScenarioResult(
        service=service_name, traffic=traffic_name, objective=objective,
        baseline_digest=base_digest, tuned_digest=tuned_digest,
        tuning=tuning, calibration=cal, charts=charts,
        gc_log_path=log_path, notes=notes,
    )
    res._wall = time.perf_counter() - t0
    return res


def log_stats_day(traffic_name: str) -> float:
    return getattr(TrafficProfile, traffic_name)().day_seconds


def _day(default: float, day_hours: Optional[float]) -> float:
    return day_hours * 3600.0 if day_hours else default


def _calibration_error(rep: CalibrationReport, truth: dict) -> dict:
    def pct(est, true):
        if not true:
            return None
        return 100.0 * (est - true) / true
    return {
        "young_size_pct": pct(rep.inferred_young_size, truth["young_size_bytes"]),
        "eden_pct": pct(rep.eden_bytes, truth["eden_bytes"]),
        "old_size_pct": pct(rep.old_size_bytes, truth["old_size_bytes"]),
        "survivor_ratio_pct": pct(rep.inferred_survivor_ratio,
                                  truth["survivor_ratio"]),
        "alloc_rate_pct": pct(rep.alloc_rate_bytes_s,
                              truth["alloc_rate_bytes_s_true"]),
        "promotion_rate_pct": pct(rep.promotion_rate_bytes_s,
                                  truth["promotion_rate_bytes_s_true"]),
        "promotion_fraction_pct": pct(rep.promotion_fraction,
                                      truth["promotion_fraction_true"]),
    }


def _compare_pause_chart(a: RunDigest, b: RunDigest) -> str:
    cats = ["p50", "p90", "p99", "max"]
    av = a.stats.pause_percentiles()
    bv = b.stats.pause_percentiles()
    ma = [av.get(k, 0) * 1e3 for k in ("p50", "p90", "p99", "max")]
    mb = [bv.get(k, 0) * 1e3 for k in ("p50", "p90", "p99", "max")]
    return svg_bars(cats,
                    [{"name": a.label, "values": ma},
                     {"name": b.label, "values": mb}],
                    "Pause percentiles: baseline vs tuned", "ms",
                    height=240)


def _compare_metrics_chart(a: RunDigest, b: RunDigest) -> str:
    ma, mb = a.metrics(), b.metrics()
    cats = ["max pause (ms)", "p99 pause (ms)", "total STW (s)",
            "full GCs", "degraded cycles", "promotion (GB)"]
    va = [ma["pause_max_ms"], ma["pause_p99_ms"], ma["total_stw_s"],
          ma["full_gcs"], ma["degraded_cycles"], ma["promotion_gb"]]
    vb = [mb["pause_max_ms"], mb["pause_p99_ms"], mb["total_stw_s"],
          mb["full_gcs"], mb["degraded_cycles"], mb["promotion_gb"]]
    return svg_bars(cats,
                    [{"name": a.label, "values": va},
                     {"name": b.label, "values": vb}],
                    "Before/after tuning", "", height=260)


# ----------------------------------------------------------------------------
# suite driver
# ----------------------------------------------------------------------------

def run_suite(outdir: str = "reports", seed: int = 12345,
              cohort_seconds: float = 5.0,
              objectives: Optional[Dict[str, Objective]] = None,
              services: Optional[Sequence[tuple]] = None,
              workers: Optional[int] = None,
              do_calibration: bool = True,
              day_hours: Optional[float] = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    scenarios_spec = list(services or SERVICES)
    objectives = objectives or {}
    results: List[ScenarioResult] = []
    seen: Dict[str, int] = {}
    for spec in scenarios_spec:
        svc, tr, old_gb, mp, thr = spec
        obj = objectives.get(svc) or Objective(max_pause_ms=mp,
                                               min_throughput=thr)
        r = scenario(svc, tr, objective=obj, seed=seed,
                     cohort_seconds=cohort_seconds,
                     outdir=outdir, workers=workers,
                     do_calibration=do_calibration,
                     day_hours=day_hours, old_gb=old_gb)
        # disambiguate duplicate services run at different budgets
        n = seen.get(svc, 0)
        seen[svc] = n + 1
        if n:
            r.service = f"{svc}#{n + 1}"
        results.append(r)
    bundle = _bundle(results, outdir)
    _write_html(bundle, os.path.join(outdir, "index.html"))
    write_json(os.path.join(outdir, "results.json"), bundle)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(_markdown_summary(bundle))
    return bundle


def _scenario_dict(r: ScenarioResult) -> dict:
    b = r.baseline_digest
    t = r.tuned_digest
    out = {
        "service": r.service, "traffic": r.traffic,
        "objective": {"max_pause_ms": r.objective.max_pause_ms,
                      "min_throughput": r.objective.min_throughput},
        "baseline": b.metrics(),
        "tuned": t.metrics() if t else None,
        "notes": r.notes,
        "charts": r.charts,
        "gc_log_path": r.gc_log_path,
        "calibration": r.calibration,
        "tuning": _tuning_dict(r.tuning) if r.tuning else None,
        "wall_s": getattr(r, "_wall", 0.0),
    }
    return out


def _tuning_dict(t: TuningResult) -> dict:
    best = t.best
    return {
        "objective": {"max_pause_ms": t.objective.max_pause_ms,
                      "min_throughput": t.objective.min_throughput},
        "baseline": t.baseline.to_dict(),
        "best": best.to_dict() if best else None,
        "grid_size": len(t.grid),
        "feasible": sum(1 for p in t.grid if p.feasible),
        "sensitivity": t.sensitivity,
        "notes": t.notes,
        "grid": [p.to_dict() for p in t.grid],
    }


def _bundle(results: List[ScenarioResult], outdir: str) -> dict:
    scen = [_scenario_dict(r) for r in results]
    charts = {}
    for s in scen:
        charts.update(s["charts"])
    bundle = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "outdir": os.path.abspath(outdir),
        "scenarios": scen,
        "charts": charts,
    }
    return bundle


def _markdown_summary(bundle: dict) -> str:
    lines = ["# Generational GC tuning report", "",
             f"Generated {bundle['generated']}", "",
             "Every number below comes from the simulator in `gcsim/`. Pauses, "
             "promotion and heap figures are model output; the "
             "reverse-calibration table at the end of each section shows how "
             "well the log estimator recovers the known parameters.", ""]
    for s in bundle["scenarios"]:
        lines.append(f"## {s['service']} / {s['traffic']}")
        lines.append("")
        lines.append(f"Objective: pause <= {s['objective']['max_pause_ms']:g} ms, "
                     f"throughput >= {s['objective']['min_throughput']*100:.2f}%")
        lines.append("")
        b = s["baseline"]
        rows = [["metric", "baseline", "tuned"]]
        if s["tuned"]:
            t = s["tuned"]
            for key, fmt in _SUMMARY_KEYS:
                rows.append([key, _fmt_metric(b, key, fmt), _fmt_metric(t, key, fmt)])
        else:
            for key, fmt in _SUMMARY_KEYS:
                rows.append([key, _fmt_metric(b, key, fmt), "-"])
        lines.append(markdown_table(rows))
        lines.append("")
        tuning = s.get("tuning")
        if tuning and tuning.get("best"):
            bp = tuning["best"]
            lines.append(
                f"Chosen parameters: young {bp['young_mb']:.0f} MB, survivor "
                f"ratio {bp['survivor_ratio']:.2f}, tenure age "
                f"{bp['promotion_age']}, initiating occupancy "
                f"{bp['initiating']:.2f}; giving max pause "
                f"{bp['max_pause_ms']:.1f} ms, throughput "
                f"{bp['throughput']*100:.3f}%, full GCs {bp['full_gcs']}. "
                f"{tuning['feasible']} of {tuning['grid_size']} evaluated "
                "configurations met both objectives.")
            lines.append("")
            if tuning.get("sensitivity"):
                lines.append("Per-parameter response (mean over the grid):")
                lines.append("")
                srows = [["knob", "value", "mean max pause", "mean p99",
                          "mean throughput", "mean promotion", "mean full GCs"]]
                for param, label in (("young", "young MB"),
                                     ("promotion_age", "tenure age"),
                                     ("initiating", "initiating occ.")):
                    for r in tuning["sensitivity"].get(param, []):
                        srows.append([
                            label, f"{r['value']:.2f}",
                            f"{r['max_pause_ms']:.1f} ms",
                            f"{r['p99_ms']:.1f} ms",
                            f"{r['throughput']*100:.3f}%",
                            f"{r['promotion_gb']:.1f} GB",
                            f"{r['full_gcs']:.1f}",
                        ])
                lines.append(markdown_table(srows))
                lines.append("")
        if s["notes"]:
            lines.append("Tuning notes:")
            for n in s["notes"]:
                lines.append(f"- {n}")
            lines.append("")
        cal = s.get("calibration")
        if cal:
            lines.append("Reverse calibration from the simulated GC log:")
            lines.append("")
            est = cal["estimate"]
            tr = cal["truth"]
            err = cal["error"]
            rows = [["quantity", "estimate", "true", "error %"]]
            rows.append(["young size", f"{est['inferred_young_size']/1e9:.2f} GB",
                         f"{tr['young_size_bytes']/1e9:.2f} GB",
                         _pct(err["young_size_pct"])])
            rows.append(["eden", f"{est['eden_bytes']/1e9:.2f} GB",
                         f"{tr['eden_bytes']/1e9:.2f} GB", _pct(err["eden_pct"])])
            rows.append(["old size", f"{est['old_size_bytes']/1e9:.2f} GB",
                         f"{tr['old_size_bytes']/1e9:.2f} GB", _pct(err["old_size_pct"])])
            rows.append(["survivor ratio", f"{est['inferred_survivor_ratio']:.3f}",
                         f"{tr['survivor_ratio']:.3f}", _pct(err["survivor_ratio_pct"])])
            rows.append(["alloc rate", f"{est['alloc_rate_bytes_s']/1e6:.1f} MB/s",
                         f"{tr['alloc_rate_bytes_s_true']/1e6:.1f} MB/s",
                         _pct(err["alloc_rate_pct"])])
            rows.append(["promotion rate", f"{est['promotion_rate_bytes_s']/1e6:.1f} MB/s",
                         f"{tr['promotion_rate_bytes_s_true']/1e6:.1f} MB/s",
                         _pct(err["promotion_rate_pct"])])
            rows.append(["promotion fraction", f"{est['promotion_fraction']*100:.2f}%",
                         f"{tr['promotion_fraction_true']*100:.2f}%",
                         _pct(err["promotion_fraction_pct"])])
            lines.append(markdown_table(rows))
            lines.append("")
    return "\n".join(lines)


_SUMMARY_KEYS = [
    ("young_gcs", "d"), ("remark_pauses", "d"), ("full_gcs", "d"),
    ("degraded_cycles", "d"), ("pause_p50_ms", ".2f"), ("pause_p99_ms", ".1f"),
    ("pause_max_ms", ".1f"), ("throughput_stw", ".5%"),
    ("throughput_adj", ".5%"), ("promotion_gb", ".1f"),
    ("promotion_fraction", ".2%"), ("pretenured_gb", ".1f"),
    ("old_peak_committed_gb", ".2f"),
]


def _fmt_metric(m: dict, key: str, fmt: str) -> str:
    v = m.get(key, 0.0)
    if fmt == "d":
        return str(int(v))
    return format(v, fmt)


def _pct(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.2f}"


def _write_html(bundle: dict, path: str) -> None:
    """A self-contained HTML page (no external assets) for the report."""
    from .page import render_page
    html = render_page(bundle)
    with open(path, "w") as f:
        f.write(html)
