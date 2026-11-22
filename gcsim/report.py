"""Report generation: JSON for the UI, Markdown/HTML for humans, SVG charts.

Everything the acceptance criteria asks for is produced here:

* pause distribution and maximum pause,
* throughput loss (STW and marking-adjusted),
* number of full-heap pauses and degraded concurrent cycles,
* before/after tuning comparison,
* reverse calibration of a simulated GC log: estimated vs true parameters,
* per-generation occupancy over time, pause timeline, promoted-age
  distribution.

The SVG charts are dependency-free (no matplotlib) so the tool runs anywhere.
Each chart is a plain string of SVG; the web page embeds them.  The data is
also exported as JSON so a different front end can render it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import GCConfig
from .simulator import RunStats

# ----------------------------------------------------------------------------
# small SVG helpers
# ----------------------------------------------------------------------------

PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
           "#b279a2", "#9d755d", "#b9b9b9", "#eeca3b", "#ff9da6"]


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n * 1.5:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def svg_line_chart(series: Sequence[dict], title: str, ylabel: str,
                   xlabel: str = "hours", width: int = 720, height: int = 260,
                   ymax: Optional[float] = None, xmax: Optional[float] = None,
                   xmin: float = 0.0) -> str:
    """series: list of {name, points:[(x,y)], color?, fill?, width?, opacity?}."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [p[0] for s in series for p in s["points"]] or [0.0]
    ys = [p[1] for s in series for p in s["points"]] or [0.0]
    xlo = xmin
    xhi = xmax if xmax is not None else max(xs) if xs else 1.0
    yhi = ymax if ymax is not None else (max(ys) if ys else 1.0)
    if yhi <= 0:
        yhi = 1.0
    yhi = yhi * 1.08

    def sx(x):
        return pad_l + (x - xlo) / (xhi - xlo or 1) * w

    def sy(y):
        return pad_t + h - (y / (yhi or 1)) * h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'class="chart" role="img" aria-label="{_esc(title)}">']
    out.append(f'<text x="{pad_l}" y="18" class="ct">{_esc(title)}</text>')
    # grid + y ticks
    for yt in _nice_ticks(0, yhi, 5):
        yy = sy(yt)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+w}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" class="tick ta-r">{_fmt_tick(yt)}</text>')
    # x ticks
    for xt in _nice_ticks(xlo, xhi, 6):
        xx = sx(xt)
        out.append(f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t+h}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{pad_t+h+16}" class="tick ta-c">{_fmt_tick(xt)}</text>')
    out.append(f'<text x="{pad_l+w/2:.0f}" y="{height-4}" class="axis ta-c">{_esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{pad_t+h/2:.0f}" class="axis" transform="rotate(-90 14 {pad_t+h/2:.0f})" text-anchor="middle">{_esc(ylabel)}</text>')
    for i, s in enumerate(series):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        pts = s["points"]
        if not pts:
            continue
        if s.get("fill"):
            d = f"M {sx(pts[0][0]):.1f} {sy(0):.1f} "
            for x, y in pts:
                d += f"L {sx(x):.1f} {sy(y):.1f} "
            d += f"L {sx(pts[-1][0]):.1f} {sy(0):.1f} Z"
            out.append(f'<path d="{d}" fill="{color}" opacity="{s.get("opacity",0.18)}"/>')
        step = max(1, len(pts) // 900)
        d = "M " + " L ".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in pts[::step])
        out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                   f'stroke-width="{s.get("width",1.6)}" opacity="{s.get("opacity_line",0.95)}"/>')
    # legend
    lx = pad_l + 8
    ly = pad_t + 4
    for i, s in enumerate(series):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        out.append(f'<rect x="{lx}" y="{ly-8}" width="10" height="10" fill="{color}"/>')
        out.append(f'<text x="{lx+14}" y="{ly+1}" class="legend">{_esc(s["name"])}</text>')
        lx += 18 + 7 * len(s["name"]) + 14
        if lx > width - 140:
            lx = pad_l + 8
            ly += 16
    out.append("</svg>")
    return "".join(out)


def _fmt_tick(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f}G"
    if a >= 1e6:
        return f"{v/1e6:.0f}M"
    if a >= 1e3:
        return f"{v/1e3:.0f}k"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def svg_histogram(values: Sequence[float], title: str, xlabel: str,
                  ylabel: str = "count", bins: int = 40,
                  width: int = 720, height: int = 240,
                  log_y: bool = False, color: str = PALETTE[0]) -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 40
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    vals = list(values) or [0.0]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    counts = [0] * bins
    for v in vals:
        b = int((v - lo) / (hi - lo) * bins)
        b = min(bins - 1, max(0, b))
        counts[b] += 1
    ytop = max(counts) if not log_y else max(1, max(counts))
    def sy(c):
        if log_y:
            c = math.log10(max(1, c) + 1) if False else c
            return pad_t + h - (math.log10(max(1, c)) / math.log10(max(2, ytop))) * h
        return pad_t + h - (c / max(1, ytop)) * h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{_esc(title)}">']
    out.append(f'<text x="{pad_l}" y="18" class="ct">{_esc(title)}</text>')
    bw = w / bins
    for i, c in enumerate(counts):
        x = pad_l + i * bw
        y = sy(c)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1,bw-1):.1f}" '
                   f'height="{pad_t+h-y:.1f}" fill="{color}" opacity="0.85"/>')
    for yt in _nice_ticks(0, ytop, 4):
        if log_y:
            continue
        yy = sy(yt)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+w}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" class="tick ta-r">{_fmt_tick(yt)}</text>')
    for i in range(6):
        xv = lo + (hi - lo) * i / 5
        xx = pad_l + w * i / 5
        out.append(f'<text x="{xx:.1f}" y="{pad_t+h+16}" class="tick ta-c">{_fmt_tick(xv)}</text>')
    out.append(f'<text x="{pad_l+w/2:.0f}" y="{height-6}" class="axis ta-c">{_esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{pad_t+h/2:.0f}" class="axis" transform="rotate(-90 14 {pad_t+h/2:.0f})" text-anchor="middle">{_esc(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_bars(categories: Sequence[str], groups: Sequence[dict],
             title: str, ylabel: str = "",
             width: int = 720, height: int = 240) -> str:
    """Grouped bar chart. groups: [{name, values:[...], color?}]."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 50
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    ymax = max([v for g in groups for v in g["values"]] + [1.0]) * 1.12
    ncat = max(1, len(categories))
    ng = max(1, len(groups))
    group_w = w / ncat
    bar_w = group_w * 0.8 / ng
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{_esc(title)}">']
    out.append(f'<text x="{pad_l}" y="18" class="ct">{_esc(title)}</text>')
    for yt in _nice_ticks(0, ymax, 4):
        yy = pad_t + h - yt / ymax * h
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+w}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" class="tick ta-r">{_fmt_tick(yt)}</text>')
    for ci, cat in enumerate(categories):
        gx = pad_l + ci * group_w + group_w * 0.1
        for gi, g in enumerate(groups):
            v = g["values"][ci] if ci < len(g["values"]) else 0
            color = g.get("color") or PALETTE[gi % len(PALETTE)]
            bh = (v / ymax) * h
            out.append(f'<rect x="{gx+gi*bar_w:.1f}" y="{pad_t+h-bh:.1f}" '
                       f'width="{bar_w*0.92:.1f}" height="{bh:.1f}" fill="{color}"/>')
        out.append(f'<text x="{pad_l+ci*group_w+group_w/2:.1f}" y="{pad_t+h+16}" '
                   f'class="tick ta-c">{_esc(cat)}</text>')
    lx = pad_l
    for gi, g in enumerate(groups):
        color = g.get("color") or PALETTE[gi % len(PALETTE)]
        out.append(f'<rect x="{lx}" y="{pad_t+h+30}" width="10" height="10" fill="{color}"/>')
        out.append(f'<text x="{lx+14}" y="{pad_t+h+39}" class="legend">{_esc(g["name"])}</text>')
        lx += 28 + 7 * len(g["name"])
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------
# run summaries
# ----------------------------------------------------------------------------

@dataclass
class RunDigest:
    """Everything a report or a page needs for one run."""

    label: str
    cfg: GCConfig
    stats: RunStats
    day_seconds: float
    service_name: str
    traffic_name: str
    alloc_bytes: float = 0.0

    def metrics(self) -> dict:
        st = self.stats
        pc = st.pause_percentiles()
        day = self.day_seconds or 1.0
        stw = st.total_pause_s()
        thr_stw = max(0.0, 1.0 - stw / day)
        thr_adj = max(0.0, 1.0 - (stw + st.app_cpu_lost_s) / day)
        prom_frac = st.promotion_bytes / st.allocated_bytes if st.allocated_bytes else 0.0
        return {
            "label": self.label,
            "service": self.service_name,
            "traffic": self.traffic_name,
            "day_hours": day / 3600.0,
            "young_gcs": st.young_gcs,
            "mixed_gcs": st.mixed_gcs,
            "remark_pauses": st.remark_count,
            "full_gcs": st.full_gcs,
            "concurrent_cycles": st.concurrent_cycles,
            "degraded_cycles": st.degraded_cycles,
            "tospace_exhaustion": st.tospace_exhausted,
            "pause_p50_ms": pc.get("p50", 0.0) * 1e3,
            "pause_p90_ms": pc.get("p90", 0.0) * 1e3,
            "pause_p99_ms": pc.get("p99", 0.0) * 1e3,
            "pause_p999_ms": pc.get("p99.9", 0.0) * 1e3,
            "pause_max_ms": pc.get("max", 0.0) * 1e3,
            "total_stw_s": stw,
            "throughput_stw": thr_stw,
            "throughput_adj": thr_adj,
            "throughput_loss_pct": (1 - thr_adj) * 100,
            "promotion_gb": st.promotion_bytes / 1e9,
            "promotion_fraction": prom_frac,
            "pretenured_gb": st.pretenured_bytes / 1e9,
            "allocated_gb": st.allocated_bytes / 1e9,
            "old_peak_committed_gb": st.old_peak_committed / 1e9,
            "old_peak_live_gb": st.old_peak_live / 1e9,
            "concurrent_overhead_s": st.concurrent_overhead_s,
            "app_cpu_lost_s": st.app_cpu_lost_s,
            "wall_s": st.collection_wall_time,
            "oom_events": st.oom_events,
        }


def digest(label: str, cfg: GCConfig, service, traffic, stats: RunStats) -> RunDigest:
    return RunDigest(label=label, cfg=cfg, stats=stats,
                     day_seconds=traffic.day_seconds,
                     service_name=service.name,
                     traffic_name=getattr(traffic, "_name", "custom"))


def digest_charts(d: RunDigest, width: int = 720) -> Dict[str, str]:
    st = d.stats
    hours = 3600.0
    charts = {}
    # occupancy over time
    tl = st.timeline
    charts["occupancy"] = svg_line_chart(
        [
            {"name": "old (committed)", "points": [(r["t"] / hours, r["old_committed"]) for r in tl]},
            {"name": "old (live)", "points": [(r["t"] / hours, r["old_live"]) for r in tl]},
            {"name": "young", "points": [(r["t"] / hours, r["young"]) for r in tl]},
            {"name": "eden", "points": [(r["t"] / hours, r["eden"]) for r in tl]},
        ],
        f"Heap occupancy -- {d.label}", "bytes", "hours", width=width,
        ymax=max(1.0, d.cfg.heap.old_size / 1.05),
    )
    # pause timeline
    ps = [(t / hours, p * 1e3) for t, p, kind in st.pause_series]
    charts["pauses"] = svg_line_chart(
        [{"name": "STW pause", "points": ps, "width": 1.0}],
        f"Stop-the-world pauses -- {d.label}", "ms", "hours", width=width,
    )
    # pause histogram
    charts["pause_hist"] = svg_histogram(
        [p * 1e3 for _, p, _ in st.pauses], f"Pause distribution -- {d.label}",
        "pause (ms)", "collections", bins=36, width=width, height=220)
    # promotion age histogram (bytes)
    ab = st.promoted_age_bytes
    if ab:
        ages = sorted(ab)
        charts["age_hist"] = svg_bars(
            [str(a) for a in ages],
            [{"name": "promoted bytes", "values": [ab[a] for a in ages]}],
            f"Promoted bytes by tenure age -- {d.label}", "bytes", width=width,
            height=220)
    else:
        charts["age_hist"] = svg_bars(["none"], [{"name": "no promotion", "values": [0]}],
                                      f"Promoted bytes by tenure age -- {d.label}",
                                      "bytes", width=width, height=180)
    # survivor-space occupancy over scavenges
    charts["survivor"] = svg_line_chart(
        [{"name": "survivor bytes", "points": [(t / hours, b) for t, b in st.surv_occ_series]}],
        f"Survivor-space occupancy -- {d.label}", "bytes", "hours", width=width,
        ymax=d.cfg.heap.survivor_space() * 1.05)
    # per-hour pause budget
    if st.hour_summary:
        cats = [f"{h['h']:02d}" for h in st.hour_summary]
        charts["hourly"] = svg_bars(
            cats,
            [{"name": "STW seconds in hour",
              "values": [h["pause_s"] for h in st.hour_summary]}],
            f"STW time per hour -- {d.label}", "seconds", width=width, height=220)
    return charts


def comparison_table(rows: Sequence[RunDigest]) -> List[List[str]]:
    header = ["metric"] + [r.label for r in rows]
    keys = [
        ("p50 pause (ms)", lambda m: f"{m['pause_p50_ms']:.2f}"),
        ("p99 pause (ms)", lambda m: f"{m['pause_p99_ms']:.1f}"),
        ("max pause (ms)", lambda m: f"{m['pause_max_ms']:.1f}"),
        ("young GCs", lambda m: f"{m['young_gcs']}"),
        ("remark pauses", lambda m: f"{m['remark_pauses']}"),
        ("full GCs", lambda m: f"{m['full_gcs']}"),
        ("degraded cycles", lambda m: f"{m['degraded_cycles']}"),
        ("total STW (s)", lambda m: f"{m['total_stw_s']:.1f}"),
        ("throughput (STW)", lambda m: f"{m['throughput_stw']*100:.3f}%"),
        ("throughput (adj)", lambda m: f"{m['throughput_adj']*100:.3f}%"),
        ("promotion (GB)", lambda m: f"{m['promotion_gb']:.1f}"),
        ("promotion % of alloc", lambda m: f"{m['promotion_fraction']*100:.2f}%"),
        ("pre-tenured (GB)", lambda m: f"{m['pretenured_gb']:.1f}"),
        ("old peak committed (GB)", lambda m: f"{m['old_peak_committed_gb']:.2f}"),
        ("to-space exhaustion", lambda m: f"{m['tospace_exhaustion']}"),
    ]
    table = [header]
    ms = [r.metrics() for r in rows]
    for name, fn in keys:
        table.append([name] + [fn(m) for m in ms])
    return table


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (set,)):
        return sorted(o)
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def markdown_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(str(c) for c in rows[0]) + " |",
           "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
