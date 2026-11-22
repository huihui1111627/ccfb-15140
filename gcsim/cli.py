"""Command-line entry points.

    python3 -m gcsim.cli sim    --scenario trading-gateway --out run.json
    python3 -m gcsim.cli log    --scenario trading-gateway --format hotspot
    python3 -m gcsim.cli calib  --log gc.log
    python3 -m gcsim.cli tune   --scenario trading-gateway --workers 8
    python3 -m gcsim.cli serve  --report reports/acceptance.json

Everything is offline; ``serve`` just opens the generated dashboard file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

from .config import GCConfig, ServiceScenario, get_scenario, load_json, save_json
from .engine import Simulator
from .gclog import parse_log, render_hotspot, render_simple
from . import tuner


def _load_scenario(name: str, config_path: str = None) -> ServiceScenario:
    sc = get_scenario(name)
    if config_path:
        cfg = load_json(config_path)
        gc_part = cfg.get("gc", cfg)
        sc.gc = GCConfig.from_dict(gc_part)
        if "workload" in cfg:
            from .config import WorkloadConfig
            sc.workload = WorkloadConfig.from_dict(cfg["workload"])
        if "population" in cfg:
            from .config import ObjectPopulation
            sc.population = ObjectPopulation.from_dict(cfg["population"])
        sc.validate()
    return sc


def cmd_sim(args) -> int:
    sc = _load_scenario(args.scenario, args.config)
    if args.day_seconds:
        sc.workload.day_seconds = args.day_seconds
    sim = Simulator(sc)
    res = sim.run()
    payload = res.to_dict()
    payload["scenario"]["name"] = sc.name
    save_json(payload, args.out)
    s = res.summary
    print(f"{sc.name}: 停顿 P50/P99/最长 = "
          f"{s['pause_p50_ms']:.1f}/{s['pause_p99_ms']:.1f}/"
          f"{s['pause_max_ms']:.1f} ms, 吞吐损失 "
          f"{s['throughput_loss_frac'] * 100:.2f}%, "
          f"全堆停顿 {s['full_count']} 次")
    return 0


def cmd_log(args) -> int:
    sc = _load_scenario(args.scenario, args.config)
    if args.day_seconds:
        sc.workload.day_seconds = args.day_seconds
    res = Simulator(sc).run()
    text = render_simple(res) if args.format == "simple" else render_hotspot(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_calib(args) -> int:
    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()
    events, est = parse_log(text)
    out = {k: v for k, v in est.items() if k != "occupancy"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_tune(args) -> int:
    sc = _load_scenario(args.scenario, args.config)
    obj = {}
    if args.pause_target_ms:
        obj["pause_target_ms"] = args.pause_target_ms
    if args.throughput_target:
        obj["throughput_target"] = args.throughput_target
    res = tuner.tune(sc, obj, max_candidates=args.max_candidates,
                     refine=not args.no_refine, workers=args.workers)
    payload = {
        "objective": res.objective,
        "baseline": res.baseline.key_metrics(),
        "best": res.best.key_metrics(),
        "recommendation": res.top_recommendation,
        "ranking": res.ranking,
        "effects": res.effects,
        "wall_s": res.wall_s,
    }
    save_json(payload, args.out)
    rec = res.top_recommendation
    print(f"可行 {res.feasible_count}/{len(res.candidates)}；"
          f"最长停顿 {rec['before']['pause_max_ms']:.1f} → "
          f"{rec['after']['pause_max_ms']:.1f} ms；"
          f"全堆停顿 {rec['before']['full_count']} → "
          f"{rec['after']['full_count']}")
    for k, v in rec["changes"].items():
        if abs(v[1] - v[0]) > 1e-9:
            print(f"  {k}: {v[0]} → {v[1]}")
    return 0


def cmd_fit_population(args) -> int:
    import csv
    from .fitting import fit_population
    lifetimes, sizes, classes = [], [], []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lifetimes.append(float(row["lifetime_s"]))
            sizes.append(float(row.get("size_bytes") or 512.0))
            if "class" in row and row["class"]:
                classes.append(row["class"])
    pop = fit_population(lifetimes, sizes,
                         classes=classes if classes else None)
    payload = pop.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        save_json(payload, args.out)
    return 0


def cmd_fit_from_log(args) -> int:
    from .fitting import population_from_log, survival_curve_from_events
    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()
    events, _ = parse_log(text)
    known = get_scenario(args.scenario).population if args.scenario else None
    curve = survival_curve_from_events(events)
    pop = population_from_log(events, known_mix=known)
    payload = pop.to_dict()
    payload["_estimate"] = {
        "source": "gc-log",
        "one_step_survival": curve["one_step_survival"],
        "one_step_lower": curve["one_step_lower"],
        "one_step_upper": curve["one_step_upper"],
        "overflow_dominated": curve["overflow_dominated"],
        "note": ("日志只能给出存活率的区间：幸存区溢出会把仍存活的对象提前晋升、"
                 "移出幸存区，下界因此偏低。若需要精确寿命分布，请用对象寿命轨迹"
                 "配合 fit-population。"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        save_json(payload, args.out)
    return 0


def cmd_report(args) -> int:
    from .acceptance import run_acceptance, summary_markdown
    data = run_acceptance(args.scenario, do_tune=not args.no_tune,
                          workers=args.workers,
                          record_cap=args.record_cap,
                          max_candidates=args.max_candidates)
    save_json(data, args.out)
    md = summary_markdown(data)
    md_path = args.out.rsplit(".", 1)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote {args.out} and {md_path}")
    return 0


def cmd_serve(args) -> int:
    from .web.build import load_and_build
    if args.report.endswith(".html"):
        path = args.report
    else:
        path = os.path.join(os.path.dirname(args.report) or ".",
                            "dashboard.html")
        load_and_build(args.report, path)
        print(f"built {path}")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(path))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gcsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sim", help="simulate one scenario")
    p.add_argument("--scenario", default="trading-gateway")
    p.add_argument("--config", default=None)
    p.add_argument("--day-seconds", type=int, default=None)
    p.add_argument("--out", default="run.json")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("log", help="render a GC log")
    p.add_argument("--scenario", default="trading-gateway")
    p.add_argument("--config", default=None)
    p.add_argument("--format", choices=["hotspot", "simple"], default="hotspot")
    p.add_argument("--day-seconds", type=int, default=None,
                   help="simulate a shorter day (e.g. 3600 for a sample log)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("calib", help="estimate parameters from a GC log")
    p.add_argument("--log", required=True)
    p.set_defaults(func=cmd_calib)

    p = sub.add_parser("tune", help="search collector parameters")
    p.add_argument("--scenario", default="trading-gateway")
    p.add_argument("--config", default=None)
    p.add_argument("--pause-target-ms", type=float, default=None)
    p.add_argument("--throughput-target", type=float, default=None)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="tuning.json")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("fit-population",
                       help="recover the object population from a lifetime trace")
    p.add_argument("--input", required=True,
                   help="CSV with columns lifetime_s,size_bytes[,class]")
    p.add_argument("--out", default=None,
                   help="write the fitted ObjectPopulation as JSON")
    p.set_defaults(func=cmd_fit_population)

    p = sub.add_parser("fit-from-log",
                       help="recover the object population from a GC log")
    p.add_argument("--log", required=True)
    p.add_argument("--scenario", default=None,
                   help="keep this scenario's class mix and only refit lifetimes")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_fit_from_log)

    p = sub.add_parser("report", help="run acceptance and write JSON + summary")
    p.add_argument("--out", default="reports/acceptance.json")
    p.add_argument("--scenario", action="append")
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--record-cap", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=24)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="build/open the dashboard")
    p.add_argument("--report", default="reports/acceptance.json")
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
