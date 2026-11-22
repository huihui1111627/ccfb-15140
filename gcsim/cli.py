"""Command-line interface for the GC simulator, tuner and calibration.

Examples
--------
Run the full acceptance suite (four services x full day, tuning, reverse
calibration, HTML report)::

    python -m gcsim suite --outdir reports

Run one scenario at 6 simulated hours for a quick look::

    python -m gcsim suite --outdir reports --hours 6 --service session_heavy

Simulate one configuration and print its metrics as JSON::

    python -m gcsim simulate --service trading_front_end --traffic trading_day \\
        --young-mb 512 --survivor-ratio 0.25 --tenure-age 7 --initiating 0.45

Reverse-calibrate a real (or simulated) GC log::

    python -m gcsim calibrate path/to/gc.log

Write a GC log for a configuration (for feeding the calibrator or a real
log parser)::

    python -m gcsim log --service cache_heavy --traffic batch_heavy -o out.gc.log

Render the static HTML page from an existing results bundle::

    python -m gcsim render reports/results.json -o reports/index.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=12345,
                   help="random seed (default: 12345)")
    p.add_argument("--hours", type=float, default=None,
                   help="simulate only this many hours instead of a full day")
    p.add_argument("--cohort-seconds", type=float, default=None,
                   help="workload cohort resolution in seconds "
                        "(default: 1.0 for suite, 0.5 for single runs)")


def cmd_suite(args) -> int:
    from .acceptance import SERVICES, run_suite

    services = SERVICES
    if args.service:
        match = [s for s in SERVICES if s[0] == args.service]
        if not match:
            print(f"unknown service {args.service!r}; choose from "
                  f"{sorted({s[0] for s in SERVICES})}", file=sys.stderr)
            return 2
        services = match
    objectives = {}
    if args.max_pause_ms or args.min_throughput:
        # Override the per-scenario objective for every selected scenario.
        services = [
            (svc, tr, old, (args.max_pause_ms or mp),
             (args.min_throughput or thr))
            for (svc, tr, old, mp, thr) in services
        ]
    cohort = args.cohort_seconds or 5.0
    t0 = time.perf_counter()
    bundle = run_suite(
        outdir=args.outdir, seed=args.seed, cohort_seconds=cohort,
        objectives=objectives or None, services=services,
        workers=args.workers, do_calibration=not args.no_calibration,
        day_hours=args.hours,
    )
    dt = time.perf_counter() - t0
    print(f"wrote {args.outdir}/index.html, results.json, summary.md "
          f"({dt:.1f}s)")
    for s in bundle["scenarios"]:
        b = s["baseline"]
        t = s["tuned"]
        print(f"  {s['service']:<20}/{s['traffic']:<12} "
              f"baseline max {b['pause_max_ms']:7.1f} ms  "
              + (f"tuned max {t['pause_max_ms']:7.1f} ms  "
                 f"(full GCs {b['full_gcs']} -> {t['full_gcs']})"
                 if t else "no feasible tuning point"))
    return 0


def _make_traffic(name: str, hours: Optional[float]):
    from .config import TrafficProfile
    tr = getattr(TrafficProfile, name)()
    if hours:
        tr.day_seconds = hours * 3600.0
    return tr


def cmd_simulate(args) -> int:
    from .config import GCConfig, ServiceProfile
    from .simulator import HeapSimulator
    from .workload import build_trace
    from .report import RunDigest, comparison_table, digest_charts

    if not hasattr(ServiceProfile, args.service):
        print(f"unknown service {args.service!r}", file=sys.stderr)
        return 2
    service = getattr(ServiceProfile, args.service)()
    traffic = _make_traffic(args.traffic, args.hours)
    cfg = GCConfig()
    if args.young_mb:
        cfg.heap.young_size = int(args.young_mb * 1024 * 1024)
    if args.survivor_ratio:
        cfg.heap.survivor_ratio = args.survivor_ratio
    if args.tenure_age:
        cfg.young.promotion_age = args.tenure_age
    if args.initiating:
        cfg.old.initiating_occupancy = args.initiating
    if args.old_gb:
        cfg.heap.old_size = int(args.old_gb * 1024 ** 3)
    if args.old_mode:
        cfg.old.mode = args.old_mode
    cohort = args.cohort_seconds or 5.0
    trace = build_trace(service, traffic, seed=args.seed,
                        cohort_seconds=cohort)
    sim = HeapSimulator(cfg, service, traffic, seed=args.seed, trace=trace)
    st = sim.run()
    d = RunDigest(label="run", cfg=cfg, stats=st,
                  day_seconds=traffic.day_seconds,
                  service_name=args.service, traffic_name=args.traffic)
    out = d.metrics()
    print(json.dumps(out, indent=2))
    if args.charts:
        charts = digest_charts(d)
        os.makedirs(args.charts, exist_ok=True)
        for k, v in charts.items():
            with open(os.path.join(args.charts, k + ".svg"), "w") as f:
                f.write(v)
        print(f"wrote charts to {args.charts}", file=sys.stderr)
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate_from_log
    rep = calibrate_from_log(args.path)
    print("\n".join(rep.summary_lines()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def cmd_log(args) -> int:
    from . import gclog
    from .config import GCConfig, ServiceProfile
    from .simulator import HeapSimulator
    from .workload import build_trace

    service = getattr(ServiceProfile, args.service)()
    traffic = _make_traffic(args.traffic, args.hours)
    cfg = GCConfig()
    writer, sink = gclog.open_writer(args.output, header=True, start=False)
    writer.set_heap(cfg.heap.young_size, cfg.heap.old_size,
                    cfg.heap.survivor_ratio, cfg.old.initiating_occupancy,
                    cfg.young.promotion_age, cfg.old.mode)
    writer.start()
    cohort = args.cohort_seconds or 5.0
    trace = build_trace(service, traffic, seed=args.seed,
                        cohort_seconds=cohort)
    sim = HeapSimulator(cfg, service, traffic, seed=args.seed, trace=trace,
                        log_writer=writer)
    sim.run()
    sink.close()
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def cmd_render(args) -> int:
    from .page import render_page
    with open(args.input) as f:
        bundle = json.load(f)
    html = render_page(bundle)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    from .webui import serve
    serve(host=args.host, port=args.port, results=args.results)
    return 0


def cmd_compare(args) -> int:
    """Compare two configurations on one service/traffic over a day."""
    from .config import GCConfig, ServiceProfile
    from .report import RunDigest, markdown_table, comparison_table
    from .simulator import HeapSimulator
    from .workload import build_trace

    def parse(spec):
        cfg = GCConfig()
        for part in spec.split(","):
            if not part:
                continue
            k, v = part.split("=")
            k = k.strip(); v = float(v)
            if k == "young_mb":
                cfg.heap.young_size = int(v * 1024 * 1024)
            elif k == "survivor_ratio":
                cfg.heap.survivor_ratio = v
            elif k == "tenure_age":
                cfg.young.promotion_age = int(v)
            elif k == "initiating":
                cfg.old.initiating_occupancy = v
            elif k == "old_gb":
                cfg.heap.old_size = int(v * 1024 ** 3)
            elif k == "mode":
                cfg.old.mode = "stop_the_world" if int(v) else "concurrent"
            else:
                raise SystemExit(f"unknown config key {k!r}")
        return cfg

    service = getattr(ServiceProfile, args.service)()
    traffic = _make_traffic(args.traffic, args.hours)
    cohort = args.cohort_seconds or 5.0
    trace = build_trace(service, traffic, seed=args.seed,
                        cohort_seconds=cohort)
    rows = []
    for i, spec in enumerate(args.configs):
        cfg = parse(spec)
        sim = HeapSimulator(cfg, service, traffic, seed=args.seed, trace=trace)
        st = sim.run()
        rows.append(RunDigest(label=f"cfg{i}: {spec or 'default'}", cfg=cfg,
                              stats=st, day_seconds=traffic.day_seconds,
                              service_name=args.service,
                              traffic_name=args.traffic))
    print(markdown_table(comparison_table(rows)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    p = argparse.ArgumentParser(
        prog="gcsim",
        description="Generational GC simulator, parameter tuner and GC-log "
                    "calibrator.")
    p.add_argument("--version", action="version", version=f"gcsim {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("suite", help="run the acceptance suite and write a report")
    _add_common(sp)
    sp.add_argument("--outdir", default="reports")
    sp.add_argument("--service", default=None,
                    help="run only this service")
    sp.add_argument("--max-pause-ms", type=float, default=None)
    sp.add_argument("--min-throughput", type=float, default=None,
                    help="fraction, e.g. 0.99")
    sp.add_argument("--workers", type=int, default=None)
    sp.add_argument("--no-calibration", action="store_true")
    sp.set_defaults(func=cmd_suite)

    sp = sub.add_parser("simulate", help="run one configuration, print metrics")
    _add_common(sp)
    sp.add_argument("--service", default="trading_front_end")
    sp.add_argument("--traffic", default="trading_day")
    sp.add_argument("--young-mb", type=float)
    sp.add_argument("--survivor-ratio", type=float)
    sp.add_argument("--tenure-age", type=int)
    sp.add_argument("--initiating", type=float)
    sp.add_argument("--old-gb", type=float)
    sp.add_argument("--old-mode", choices=["concurrent", "stop_the_world"])
    sp.add_argument("--charts", default=None, help="write SVG charts to this dir")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("calibrate", help="reverse-calibrate a GC log")
    sp.add_argument("path")
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("log", help="write a simulated GC log")
    _add_common(sp)
    sp.add_argument("--service", default="trading_front_end")
    sp.add_argument("--traffic", default="trading_day")
    sp.add_argument("-o", "--output", default="gc.log")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("compare", help="compare configurations on one run")
    _add_common(sp)
    sp.add_argument("--service", default="trading_front_end")
    sp.add_argument("--traffic", default="trading_day")
    sp.add_argument("--configs", nargs="+", required=True,
                    help="comma-separated key=value specs, e.g. "
                         "'young_mb=512,tenure_age=7'")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("render", help="render results.json to an HTML page")
    sp.add_argument("input")
    sp.add_argument("-o", "--output", default="index.html")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("serve", help="start the interactive web UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--results", default="reports/results.json")
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
