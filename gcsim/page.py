"""Render the results bundle as a single self-contained HTML page.

No external JavaScript or CSS: charts are inline SVG, interactions are a
handful of lines of vanilla JS (scenario tabs and chart toggles).  This makes
the report portable -- open the file, no server required -- while the same
bundle also backs the optional live web UI in :mod:`gcsim.webui`.
"""

from __future__ import annotations

import html
from typing import Dict

CSS = """
:root{--bg:#0e1117;--panel:#161b22;--line:#2d333b;--fg:#e6edf3;--mut:#8b949e;
--acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,
BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif}
header{padding:20px 28px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 10px;border-left:3px solid var(--acc);padding-left:10px}
h3{font-size:14px;color:var(--mut);margin:20px 0 8px;text-transform:uppercase;
letter-spacing:.06em}
.sub{color:var(--mut);font-size:13px}
main{padding:18px 28px;max-width:1500px;margin:0 auto}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.tab{padding:8px 14px;border:1px solid var(--line);border-radius:8px;
background:var(--panel);cursor:pointer;font-size:13px}
.tab.active{border-color:var(--acc);color:var(--acc)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.kpi{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{color:var(--mut);font-size:12px}
.kpi .d{font-size:12px;margin-top:2px}
.up{color:var(--bad)}.down{color:var(--good)}.neutral{color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.chart{width:100%;height:auto;background:var(--panel);border-radius:8px;margin:6px 0}
.ct{fill:var(--fg);font-size:13px;font-weight:600}
.tick{fill:var(--mut);font-size:10px}
.axis{fill:var(--mut);font-size:11px}
.legend{fill:var(--fg);font-size:11px}
.ta-r{text-anchor:end}.ta-c{text-anchor:middle}
.grid line,.chart line.grid{stroke:var(--line);stroke-width:.5}
.note{color:var(--mut);font-size:13px;margin:6px 0}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
details{background:var(--bg);border:1px solid var(--line);border-radius:8px;
padding:10px 14px;margin:8px 0}
summary{cursor:pointer;color:var(--acc)}
code{background:#1f242c;padding:1px 5px;border-radius:4px;font-size:12px}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--acc);
vertical-align:middle;margin-left:6px}
footer{color:var(--mut);padding:24px 28px;border-top:1px solid var(--line);font-size:12px}
"""


def _kpi(label: str, value: str, delta: str = "", cls: str = "") -> str:
    d = f'<div class="d {cls}">{delta}</div>' if delta else ""
    return (f'<div class="kpi"><div class="v">{html.escape(value)}</div>'
            f'<div class="l">{html.escape(label)}</div>{d}</div>')


def _delta(new: float, old: float, unit: str = "", lower_better: bool = True,
           fmt: str = "{:+.2f}") -> tuple:
    """Return (text, css-class) describing the change old -> new."""
    if old == 0:
        return (f"{fmt.format(new)}{unit}", "neutral")
    ch = (new - old) / abs(old) * 100
    arrow = "▼" if new < old else "▲" if new > old else "＝"
    good = (new < old) if lower_better else (new > old)
    cls = "neutral" if abs(ch) < 0.05 else ("good" if good else "bad")
    return (f"{arrow} {abs(ch):.1f}% vs baseline", cls)


def _metric_table(b: dict, t: dict | None) -> str:
    keys = [
        ("young GCs", "young_gcs", "d", False),
        ("remark pauses", "remark_pauses", "d", False),
        ("full GCs", "full_gcs", "d", True),
        ("degraded concurrent cycles", "degraded_cycles", "d", True),
        ("to-space exhaustion", "tospace_exhaustion", "d", True),
        ("p50 pause", "pause_p50_ms", ".2f", True),
        ("p90 pause", "pause_p90_ms", ".1f", True),
        ("p99 pause", "pause_p99_ms", ".1f", True),
        ("p99.9 pause", "pause_p999_ms", ".1f", True),
        ("max pause", "pause_max_ms", ".1f", True),
        ("total STW", "total_stw_s", ".1f", True),
        ("throughput (STW)", "throughput_stw", ".5%", False),
        ("throughput (marking-adjusted)", "throughput_adj", ".5%", False),
        ("promotion", "promotion_gb", ".1f", True),
        ("promotion fraction", "promotion_fraction", ".2%", True),
        ("pre-tenured", "pretenured_gb", ".1f", False),
        ("old peak (committed)", "old_peak_committed_gb", ".2f", False),
        ("allocated", "allocated_gb", ".1f", False),
    ]
    rows = ['<tr><th>metric</th><th class="n">baseline</th>'
            '<th class="n">tuned</th><th class="n">change</th></tr>']
    for name, key, fmt, lower_better in keys:
        bv = b.get(key, 0)
        bs = _fmt(bv, fmt)
        if t:
            tv = t.get(key, 0)
            ts = _fmt(tv, fmt)
            txt, cls = _delta(tv, bv, lower_better=lower_better)
            rows.append(f'<tr><td>{html.escape(name)}</td><td class="n">{bs}</td>'
                        f'<td class="n">{ts}</td><td class="n {cls}">{txt}</td></tr>')
        else:
            rows.append(f'<tr><td>{html.escape(name)}</td><td class="n">{bs}</td>'
                        f'<td class="n">-</td><td class="n">-</td></tr>')
    return "<table>" + "".join(rows) + "</table>"


def _fmt(v, fmt: str) -> str:
    if fmt == "d":
        return f"{int(v):,}"
    if fmt == ".5%":
        return f"{v*100:.3f}%"
    if fmt == ".2%":
        return f"{v*100:.2f}%"
    return format(v, fmt)


def _sensitivity_table(sens: Dict[str, list], names: Dict[str, str]) -> str:
    out = []
    for param, label in (("young", "young size"), ("promotion_age", "promotion age"),
                         ("initiating", "initiating occupancy")):
        rows = sens.get(param, [])
        if not rows:
            continue
        unit = " MB" if param == "young" else ""
        out.append(f"<h3>{html.escape(label)}</h3><table>"
                   '<tr><th class="n">value</th><th class="n">mean max pause</th>'
                   '<th class="n">mean p99</th><th class="n">mean throughput</th>'
                   '<th class="n">mean promotion</th><th class="n">mean full GCs</th>'
                   '<th class="n">feasible</th></tr>')
        for r in rows:
            v = r["value"]
            vs = f"{v:.0f}{unit}" if param == "young" else (
                f"{v:.2f}" if param == "initiating" else f"{v:g}")
            out.append(
                f'<tr><td class="n">{vs}</td>'
                f'<td class="n">{r["max_pause_ms"]:.1f} ms</td>'
                f'<td class="n">{r["p99_ms"]:.1f} ms</td>'
                f'<td class="n">{r["throughput"]*100:.3f}%</td>'
                f'<td class="n">{r["promotion_gb"]:.1f} GB</td>'
                f'<td class="n">{r["full_gcs"]:.1f}</td>'
                f'<td class="n">{r["feasible_frac"]*100:.0f}%</td></tr>')
        out.append("</table>")
    return "".join(out)


def _grid_table(tuning: dict) -> str:
    grid = tuning.get("grid", [])
    if not grid:
        return ""
    # show the 12 best-feasible or closest rows
    def score(p):
        return (0 if p["feasible"] else 1,
                max(0.0, p["max_pause_ms"] - tuning["objective"]["max_pause_ms"]),
                -p["throughput"])
    rows = sorted(grid, key=score)[:16]
    out = ['<tr><th class="n">young MB</th><th class="n">surv ratio</th>'
           '<th class="n">tenure age</th><th class="n">initiating</th>'
           '<th class="n">max pause</th><th class="n">p99</th>'
           '<th class="n">throughput</th><th class="n">full GCs</th>'
           '<th class="n">promotion GB</th><th>feasible</th></tr>']
    for p in rows:
        mark = ('<span class="good">yes</span>' if p["feasible"]
                else f'<span class="bad">no</span> <span class="note">'
                     f'{html.escape("; ".join(p["fail_reasons"]))}</span>')
        out.append(
            f'<tr><td class="n">{p["young_mb"]:.0f}</td>'
            f'<td class="n">{p["survivor_ratio"]:.2f}</td>'
            f'<td class="n">{p["promotion_age"]}</td>'
            f'<td class="n">{p["initiating"]:.2f}</td>'
            f'<td class="n">{p["max_pause_ms"]:.1f} ms</td>'
            f'<td class="n">{p["p99_ms"]:.1f} ms</td>'
            f'<td class="n">{p["throughput"]*100:.3f}%</td>'
            f'<td class="n">{p["full_gcs"]}</td>'
            f'<td class="n">{p["promotion_gb"]:.1f}</td>'
            f'<td>{mark}</td></tr>')
    return "<table>" + "".join(out) + "</table>"


def _calibration_block(cal: dict) -> str:
    est = cal["estimate"]
    tr = cal["truth"]
    err = cal["error"]

    def cell(v):
        if v is None:
            return '<td class="n">n/a</td>'
        cls = "good" if abs(v) <= 10 else ("warn" if abs(v) <= 30 else "bad")
        return f'<td class="n {cls}">{v:+.1f}%</td>'

    rows = ['<tr><th>quantity</th><th class="n">estimated</th>'
            '<th class="n">true</th><th class="n">error</th></tr>']
    def gb(x): return f"{x/1e9:.3f} GB"
    def mbs(x): return f"{x/1e6:.1f} MB/s"
    rows.append(f'<tr><td>young size</td><td class="n">{gb(est["inferred_young_size"])}</td>'
                f'<td class="n">{gb(tr["young_size_bytes"])}</td>{cell(err["young_size_pct"])}</tr>')
    rows.append(f'<tr><td>eden</td><td class="n">{gb(est["eden_bytes"])}</td>'
                f'<td class="n">{gb(tr["eden_bytes"])}</td>{cell(err["eden_pct"])}</tr>')
    rows.append(f'<tr><td>old size</td><td class="n">{gb(est["old_size_bytes"])}</td>'
                f'<td class="n">{gb(tr["old_size_bytes"])}</td>{cell(err["old_size_pct"])}</tr>')
    rows.append(f'<tr><td>survivor ratio</td><td class="n">{est["inferred_survivor_ratio"]:.3f}</td>'
                f'<td class="n">{tr["survivor_ratio"]:.3f}</td>{cell(err["survivor_ratio_pct"])}</tr>')
    rows.append(f'<tr><td>allocation rate</td><td class="n">{mbs(est["alloc_rate_bytes_s"])}</td>'
                f'<td class="n">{mbs(tr["alloc_rate_bytes_s_true"])}</td>{cell(err["alloc_rate_pct"])}</tr>')
    rows.append(f'<tr><td>promotion rate</td><td class="n">{mbs(est["promotion_rate_bytes_s"])}</td>'
                f'<td class="n">{mbs(tr["promotion_rate_bytes_s_true"])}</td>{cell(err["promotion_rate_pct"])}</tr>')
    rows.append(f'<tr><td>promotion fraction</td><td class="n">{est["promotion_fraction"]*100:.2f}%</td>'
                f'<td class="n">{tr["promotion_fraction_true"]*100:.2f}%</td>{cell(err["promotion_fraction_pct"])}</tr>')
    return ("<table>" + "".join(rows) + "</table>"
            f'<p class="note">{cal["log_lines"]:,} log lines parsed from '
            f'<code>{html.escape(cal["log_path"])}</code>. '
            'The “true” column is what the simulator was configured with; '
            'the estimate uses nothing but the log text.</p>')


def render_page(bundle: dict) -> str:
    scenarios = bundle["scenarios"]
    charts = bundle.get("charts", {})
    names = [f"{s['service']} / {s['traffic']}" for s in scenarios]

    # KPI overview across scenarios
    kpi_cards = []
    for s in scenarios:
        b = s["baseline"]
        t = s["tuned"]
        concl = ("meets objective" if t else "no feasible point")
        cls = "good" if t else "bad"
        kpi_cards.append(
            f'<div class="kpi"><div class="l">{html.escape(s["service"])} · '
            f'{html.escape(s["traffic"])}</div>'
            f'<div class="v">{b["pause_max_ms"]:.0f} → '
            f'{(t["pause_max_ms"] if t else float("nan")):.0f} ms</div>'
            f'<div class="d">max pause baseline → tuned</div>'
            f'<div class="d {cls}">{concl}</div></div>')
    overview = "".join(kpi_cards)

    # tabs
    tab_btns = []
    tab_panels = []
    for i, s in enumerate(scenarios):
        sid = f"sc{i}"
        active = " active" if i == 0 else ""
        tab_btns.append(f'<div class="tab{active}" data-t="{sid}" '
                        f'onclick="showTab(\'{sid}\')">'
                        f'{html.escape(s["service"])} / {html.escape(s["traffic"])}</div>')
        b = s["baseline"]
        t = s["tuned"]
        obj = s["objective"]
        head = (f'<p class="sub">Objective: max pause ≤ '
                f'{obj["max_pause_ms"]:g} ms, throughput ≥ '
                f'{obj["min_throughput"]*100:.2f}%</p>')

        # headline KPIs
        kp = []
        if t:
            txt, cls = _delta(t["pause_max_ms"], b["pause_max_ms"], lower_better=True)
            kp.append(_kpi("max pause (ms)", f'{t["pause_max_ms"]:.1f}',
                           f'from {b["pause_max_ms"]:.1f} · {txt}', cls))
            txt, cls = _delta(t["pause_p99_ms"], b["pause_p99_ms"], lower_better=True)
            kp.append(_kpi("p99 pause (ms)", f'{t["pause_p99_ms"]:.2f}',
                           f'from {b["pause_p99_ms"]:.2f} · {txt}', cls))
            txt, cls = _delta(t["throughput_adj"], b["throughput_adj"],
                              lower_better=False)
            kp.append(_kpi("throughput", f'{t["throughput_adj"]*100:.3f}%',
                           f'from {b["throughput_adj"]*100:.3f}% · {txt}', cls))
            kp.append(_kpi("full GCs", f'{t["full_gcs"]}',
                           f'from {b["full_gcs"]} · degraded {t["degraded_cycles"]}'))
            kp.append(_kpi("promotion", f'{t["promotion_gb"]:.0f} GB',
                           f'from {b["promotion_gb"]:.0f} GB'))
        else:
            kp.append(_kpi("max pause (ms)", f'{b["pause_max_ms"]:.1f}'))
            kp.append(_kpi("p99 pause (ms)", f'{b["pause_p99_ms"]:.2f}'))
            kp.append(_kpi("throughput", f'{b["throughput_adj"]*100:.3f}%'))
            kp.append(_kpi("full GCs", f'{b["full_gcs"]}'))
            kp.append(_kpi("promotion", f'{b["promotion_gb"]:.0f} GB'))
        kpi_html = '<div class="grid">' + "".join(kp) + '</div>'

        sc_charts = s.get("charts", {}) or {}
        charts_html = []
        def add_chart(key, title):
            if key in sc_charts:
                charts_html.append(f'<h3>{html.escape(title)}</h3>' + sc_charts[key])

        add_chart("occupancy", "Heap occupancy over the day (baseline)")
        if "occupancy_tuned" in sc_charts:
            add_chart("occupancy_tuned", "Heap occupancy over the day (tuned)")
        add_chart("pauses", "Every stop-the-world pause (baseline)")
        col2 = []
        def add_col2(key, title):
            if key in sc_charts:
                col2.append(f'<h3>{html.escape(title)}</h3>' + sc_charts[key])
        add_col2("pause_hist", "Pause distribution")
        if "pause_hist_tuned" in sc_charts:
            add_col2("pause_hist_tuned", "Pause distribution (tuned)")
        add_col2("age_hist", "Promoted bytes by tenure age")
        add_col2("survivor", "Survivor-space occupancy per scavenge")
        add_col2("hourly", "STW seconds per hour of day")
        if "compare_pauses" in sc_charts:
            add_col2("compare_pauses", "Pause percentiles: before/after")

        tune_html = ""
        if s.get("tuning"):
            tuning = s["tuning"]
            best = tuning.get("best")
            if best:
                tune_html = (
                    "<div class='panel'><h3>Chosen parameters</h3>"
                    f"<p>young size <b>{best['young_mb']:.0f} MB</b>, "
                    f"survivor ratio <b>{best['survivor_ratio']:.2f}</b>, "
                    f"tenure age <b>{best['promotion_age']}</b>, "
                    f"initiating occupancy <b>{best['initiating']:.2f}</b>, "
                    f"mode <b>{best['mode']}</b> — max pause "
                    f"<b>{best['max_pause_ms']:.1f} ms</b>, throughput "
                    f"<b>{best['throughput']*100:.3f}%</b>, full GCs "
                    f"<b>{best['full_gcs']}</b>, promotion "
                    f"<b>{best['promotion_gb']:.0f} GB</b>.</p>"
                    f"<p class='note'>{tuning['feasible']} of "
                    f"{tuning['grid_size']} evaluated configurations met both "
                    "objectives.</p>"
                    + _sensitivity_table(tuning["sensitivity"], {})
                    + "<details><summary>Evaluated configurations (best 16)"
                    "</summary>" + _grid_table(tuning) + "</details></div>")
            else:
                tune_html = ("<div class='panel'><h3>Chosen parameters</h3>"
                             "<p class='bad'>No configuration met both "
                             "objectives within the searched space.</p>"
                             + "<p class='note'>"
                             + html.escape(" ".join(tuning.get("notes", [])))
                             + "</p></div>")
            notes_html = ""
            if tuning.get("notes"):
                notes_html = ("<div class='panel'><h3>What each knob does</h3>"
                              "<ul>" + "".join(f"<li>{html.escape(n)}</li>"
                                               for n in tuning["notes"])
                              + "</ul></div>")
        else:
            notes_html = ""

        cal_html = ""
        if s.get("calibration"):
            cal_html = ("<div class='panel'><h3>Reverse calibration from the "
                        "simulated GC log</h3>" + _calibration_block(s["calibration"])
                        + "</div>")

        panel = (f'<div id="{sid}" class="scenario" style="display:'
                 f'{"block" if i == 0 else "none"}">'
                 + head + kpi_html
                 + '<div class="panel"><h3>Baseline vs tuned</h3>'
                 + _metric_table(b, t) + '</div>'
                 + tune_html + notes_html
                 + '<div class="panel"><h3>Charts</h3>' + "".join(charts_html)
                 + '<div class="grid2">' + "".join(col2) + '</div></div>'
                 + cal_html
                 + '</div>')
        tab_panels.append(panel)

    js = """
function showTab(id){document.querySelectorAll('.scenario').forEach(e=>e.style.display='none');
document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
document.getElementById(id).style.display='block';
document.querySelector('.tab[data-t="'+id+'"]').classList.add('active');}
"""
    css_extra = ".grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;align-items:start}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generational GC tuning report</title>
<style>{CSS}{css_extra}</style></head>
<body>
<header><h1>Generational GC tuning report</h1>
<div class="sub">Simulated trading-front-end services over a full day ·
generated {html.escape(bundle.get('generated',''))}</div></header>
<main>
<div class="panel"><h3>Overview</h3><div class="grid">{overview}</div></div>
<div class="tabs">{''.join(tab_btns)}</div>
{''.join(tab_panels)}
</main>
<footer>All numbers come from the simulator in <code>gcsim/</code>. Pause,
promotion and heap figures are model outputs calibrated to G1-like behaviour;
the reverse-calibration table shows how well the log estimator recovers the
known parameters. Raw data: <code>results.json</code>, <code>summary.md</code>.</footer>
<script>{js}</script></body></html>"""
