"""GC log writing and parsing.

Two log dialects are supported:

``hotspot``
    Close to the ``-Xlog:gc*`` unified logging output of HotSpot:

    ```
    [2026-11-22T09:05:12.418+0800][info][gc,start    ] GC(1841) Pause Young (Normal) (G1 Evacuation Pause)
    [2026-11-22T09:05:12.418+0800][info][gc,heap     ] GC(1841) Eden regions: 48->0(48)
    [2026-11-22T09:05:12.424+0800][info][gc,heap     ] GC(1841) Survivor regions: 3->4(4)
    [2026-11-22T09:05:12.424+0800][info][gc,heap     ] GC(1841) Old regions: 612->655
    [2026-11-22T09:05:12.424+0800][info][gc,phases   ] GC(1841)   Evacuate Collection Set: 4.87ms
    [2026-11-22T09:05:12.424+0800][info][gc,phases   ] GC(1841)   Other: 1.83ms
    [2026-11-22T09:05:12.424+0800][info][gc,phases   ] GC(1841)   Pause Young (Normal) (G1 Evacuation Pause) 6.70ms
    [2026-11-22T09:05:12.424+0800][info][gc          ] GC(1841) Heap after GC invocations=1841 (full 12):
    [2026-11-22T09:05:12.424+0800][info][gc          ] GC(1841) garbage-first heap   total 4194304K, used 2686976K
    ...
    ```

``simple``
    A compact, line-oriented format used by the calibration tool when it is
    fed logs that were not produced by a recognised dialect.  It is also what
    ``--format simple`` emits.  Every line is ``key=value`` tokens after an
    event tag, e.g.::

        YOUNG t=12.340 pause=6.7 eden=48.0->0.0 surv=3.0->4.0 old=612.0->655.0 promoted=7.2
        FULL  t=48.900 pause=487.2 old=2900.0->2100.0 live=2100.0
        CONC_START t=90.1 pause=1.5 old=1500.0
        CONC_END   t=105.9 pause=3.1 old=1900.0->1620.0

The parser turns either dialect into a list of :class:`GCEvent` records and
then estimates the collector's operating parameters with
:func:`estimate_parameters`.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

MiB = 1024 * 1024
KiB = 1024


# --------------------------------------------------------------------------
# Event model shared by writer and parser
# --------------------------------------------------------------------------
@dataclass
class GCEvent:
    kind: str                 # young | full | conc-start | conc-end | remark
    wall_s: float             # seconds since the start of the log window
    pause_ms: float
    eden_before_mib: float = 0.0
    eden_after_mib: float = 0.0
    surv_before_mib: float = 0.0
    surv_after_mib: float = 0.0
    old_before_mib: float = 0.0
    old_after_mib: float = 0.0
    live_mib: float = 0.0
    promoted_mib: float = 0.0
    tenuring: int = 0
    reason: str = ""
    heap_total_mib: float = 0.0

    @property
    def is_stw(self) -> bool:
        return self.kind in ("young", "full", "conc-start", "remark")


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
def _iso(epoch: float, base_epoch: float) -> Tuple[str, float]:
    dt = _dt.datetime.fromtimestamp(base_epoch + epoch)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}", epoch


def _fmt_ms(x: float) -> str:
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


class HotSpotWriter:
    """Render simulator events in a HotSpot-like unified-GC-log dialect."""

    def __init__(self, base_epoch: Optional[float] = None, region_mib: float = 1.0,
                 heap_total_mib: float = 0.0) -> None:
        if base_epoch is None:
            # A fixed, reproducible wall-clock start (09:00 local time).
            base_epoch = _dt.datetime(2026, 11, 21, 9, 0, 0).timestamp()
        self.base_epoch = base_epoch
        self.region_mib = region_mib
        self.heap_total_mib = heap_total_mib
        self.gc_seq = 0
        self.full_seq = 0

    def _stamp(self, t: float) -> str:
        dt = _dt.datetime.fromtimestamp(self.base_epoch + t)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}+0800"

    def _regions(self, mib: float) -> int:
        return int(round(mib / self.region_mib))

    def young(self, t: float, pause_ms: float, eden0: float, eden1: float,
              surv0: float, surv1: float, old0: float, old1: float,
              tenuring: int, promoted_mib: float) -> List[str]:
        self.gc_seq += 1
        g = self.gc_seq
        ts = self._stamp(t)
        evac = max(pause_ms - 0.9, 0.1)
        lines = [
            f"[{ts}][info][gc,start    ] GC({g}) Pause Young (Normal) (G1 Evacuation Pause)",
            f"[{ts}][info][gc,heap     ] GC({g}) Eden regions: {self._regions(eden0)}->0({self._regions(eden0)})",
            f"[{ts}][info][gc,heap     ] GC({g}) Survivor regions: {self._regions(surv0)}->{self._regions(surv1)}({self._regions(surv1)})",
            f"[{ts}][info][gc,heap     ] GC({g}) Old regions: {self._regions(old0)}->{self._regions(old1)}",
            f"[{ts}][info][gc,phases   ] GC({g})   Evacuate Collection Set: {_fmt_ms(evac)}ms",
            f"[{ts}][info][gc,phases   ] GC({g})   Other: {_fmt_ms(max(pause_ms - evac, 0.01))}ms",
            f"[{ts}][info][gc,phases   ] GC({g})   Pause Young (Normal) (G1 Evacuation Pause) {_fmt_ms(pause_ms)}ms",
            f"[{ts}][info][gc,age      ] GC({g}) Desired survivor size {int(surv1 * MiB)} bytes, new threshold {tenuring} (max 15)",
            f"[{ts}][info][gc,cpu      ] GC({g}) User=0.02s Sys=0.00s Real={pause_ms / 1000:.2f}s",
        ]
        if promoted_mib >= 0.01:
            lines.append(
                f"[{ts}][info][gc,metaspace] GC({g}) Promotion: {promoted_mib:.3f} MiB")
        lines.append(
            f"[{ts}][info][gc          ] GC({g}) Heap after GC invocations={g} "
            f"(full {self.full_seq}): garbage-first heap   total "
            f"{int(self.heap_total_mib * 1024)}K, used {int((eden1 + surv1 + old1) * 1024)}K")
        return lines

    def full(self, t: float, pause_ms: float, old0: float, old1: float,
             live_mib: float, reason: str) -> List[str]:
        self.gc_seq += 1
        g = self.gc_seq
        self.full_seq += 1
        ts = self._stamp(t)
        lines = [
            f"[{ts}][info][gc,start    ] GC({g}) Pause Full ({reason})",
            f"[{ts}][info][gc,heap     ] GC({g}) Old regions: {self._regions(old0)}->{self._regions(old1)}",
            f"[{ts}][info][gc,phases   ] GC({g})   Mark Live Objects: {_fmt_ms(pause_ms * 0.62)}ms",
            f"[{ts}][info][gc,phases   ] GC({g})   Compact: {_fmt_ms(pause_ms * 0.30)}ms",
            f"[{ts}][info][gc,phases   ] GC({g})   Pause Full ({reason}) {_fmt_ms(pause_ms)}ms",
            f"[{ts}][info][gc,heap     ] GC({g}) Heap after full GC: live {live_mib:.1f} MiB",
            f"[{ts}][info][gc          ] GC({g}) Heap after GC invocations={g} "
            f"(full {self.full_seq}): garbage-first heap   total "
            f"{int(self.heap_total_mib * MiB)}K, used {int(old1 * MiB)}K",
        ]
        return lines

    def conc_start(self, t: float, pause_ms: float, old0: float) -> List[str]:
        self.gc_seq += 1
        g = self.gc_seq
        ts = self._stamp(t)
        return [
            f"[{ts}][info][gc,start    ] GC({g}) Pause Young (Concurrent Start) (G1 Humongous Allocation)",
            f"[{ts}][info][gc,heap     ] GC({g}) Old regions: {self._regions(old0)}->{self._regions(old0)}",
            f"[{ts}][info][gc,phases   ] GC({g})   Pause Young (Concurrent Start) {_fmt_ms(pause_ms)}ms",
            f"[{ts}][info][gc,marking  ] GC({g}) Concurrent Mark (initial-mark) started",
        ]

    def conc_end(self, t: float, pause_ms: float, old0: float,
                 old1: float) -> List[str]:
        self.gc_seq += 1
        g = self.gc_seq
        ts = self._stamp(t)
        return [
            f"[{ts}][info][gc,start    ] GC({g}) Pause Remark",
            f"[{ts}][info][gc,phases   ] GC({g})   Pause Remark {_fmt_ms(pause_ms)}ms",
            f"[{ts}][info][gc,marking  ] GC({g}) Concurrent Mark (remark) completed",
            f"[{ts}][info][gc,heap     ] GC({g}) Old regions: {self._regions(old0)}->{self._regions(old1)}",
            f"[{ts}][info][gc,cleanup  ] GC({g}) Concurrent Cleanup: {max(pause_ms * 2, 1.0):.2f}ms",
        ]

    def render(self, result) -> str:
        out: List[str] = []
        self.heap_total_mib = result.summary.get("heap_bytes", 0) / MiB
        start_wall = _dt.datetime(2026, 11, 21, 9, 0, 0).timestamp()
        out.append(
            f"[{self._stamp(0.0)}][info][gc          ] Using G1")
        out.append(
            f"[{self._stamp(0.0)}][info][gc,init     ] Version: openjdk 21.0.2+13 "
            f"(simulated), heap total {int(self.heap_total_mib)} MiB")
        for e in result.events:
            kind = e["kind"]
            t = e["t"]
            pause = e["pause_ms"]
            g0 = e["old_before"] / MiB
            g1 = e["old_after"] / MiB
            if kind == 0:
                out.extend(self.young(
                    t, pause, e["eden_before"] / MiB, e["eden_after"] / MiB,
                    e["surv_before"] / MiB, e["surv_after"] / MiB, g0, g1,
                    e.get("tenuring", 0), e.get("promoted_bytes", 0.0) / MiB))
            elif kind == 3:
                reason = e.get("reason", "system") or "system"
                label = {
                    "degen-keepup": "Allocation Stall",
                    "degen-panic": "Allocation Stall",
                    "degen-start": "Allocation Stall",
                    "promotion-failure": "to-space exhausted",
                    "heap-exhausted": "Heap Exhausted",
                    "old-trigger": "G1 Evacuation Pause",
                    "system": "System.gc()",
                }.get(reason, reason)
                out.extend(self.full(
                    t, pause, g0, g1, e.get("live_bytes", 0.0) / MiB, label))
            elif kind == 1:
                out.extend(self.conc_start(t, pause, g0))
            elif kind == 2:
                out.extend(self.conc_end(t, pause, g0, g1))
        end_t = result.summary.get("day_seconds", 0.0)
        out.append(
            f"[{self._stamp(end_t)}][info][gc,stats   ] Total pauses="
            f"{result.summary.get('pause_count', 0)} full="
            f"{result.summary.get('full_count', 0)} young="
            f"{result.summary.get('minor_count', 0)}")
        return "\n".join(out) + "\n"


def render_hotspot(result) -> str:
    return HotSpotWriter().render(result)


def render_simple(result) -> str:
    """Compact line-oriented dialect (easy to diff, easy to parse)."""
    out: List[str] = []
    out.append("# gcsim gc log v1")
    for e in result.events:
        kind = e["kind"]
        t = e["t"]
        pause = e["pause_ms"]
        if kind == 0:
            out.append(
                f"YOUNG t={t:.3f} pause={pause:.3f} "
                f"eden={e['eden_before'] / MiB:.1f}->{e['eden_after'] / MiB:.1f} "
                f"surv={e['surv_before'] / MiB:.1f}->{e['surv_after'] / MiB:.1f} "
                f"old={e['old_before'] / MiB:.1f}->{e['old_after'] / MiB:.1f} "
                f"promoted={e['promoted_bytes'] / MiB:.3f} "
                f"tenuring={e.get('tenuring', 0)}")
        elif kind == 3:
            out.append(
                f"FULL t={t:.3f} pause={pause:.3f} "
                f"old={e['old_before'] / MiB:.1f}->{e['old_after'] / MiB:.1f} "
                f"live={e.get('live_bytes', 0.0) / MiB:.1f} "
                f"reason={e.get('reason', '') or 'system'}")
        elif kind == 1:
            out.append(f"CONC_START t={t:.3f} pause={pause:.3f} "
                       f"old={e['old_before'] / MiB:.1f}")
        elif kind == 2:
            out.append(
                f"CONC_END t={t:.3f} pause={pause:.3f} "
                f"old={e['old_before'] / MiB:.1f}->{e['old_after'] / MiB:.1f}")
    out.append(f"# end t={result.summary.get('day_seconds', 0):.0f}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
_TS_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})(?P<tz>[+-]\d{4})?\]")
_GC_TAG_RE = re.compile(r"\[(?:info|debug|warning)\]\[gc[^\]]*\]")

_REGIONS_RE = re.compile(r"(Eden|Survivor|Old) regions: (\d+)->(\d+)(?:\((\d+)\))?")
_PAUSE_RE = re.compile(r"Pause (Young|Full|Remark|Initial Mark|Concurrent Start)"
                       r"(?: \(([^)]*)\))?(?:[^(]*)?\s+([\d.]+)ms\s*$")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _to_epoch(ts: str, tz: Optional[str]) -> float:
    base = _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
    if tz:
        sign = 1 if tz[0] == "+" else -1
        offset = sign * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
        return base.timestamp() - offset
    return base.timestamp()


class LogParser:
    """Parse a GC log into :class:`GCEvent` records and aggregate statistics."""

    def __init__(self) -> None:
        self.events: List[GCEvent] = []
        self.dialect = "unknown"
        self.region_mib = 1.0
        self.warnings: List[str] = []
        self.heap_total_mib = 0.0
        self.day_seconds = 0.0

    # -- detection ------------------------------------------------------
    def parse(self, text: str) -> List[GCEvent]:
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        stripped = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        if not stripped:
            return []
        if stripped[0].startswith(("YOUNG ", "FULL ", "CONC_START ", "CONC_END ")):
            self.dialect = "simple"
            return self._parse_simple(stripped)
        if any("[gc" in ln for ln in stripped[:50]):
            self.dialect = "hotspot"
            return self._parse_hotspot(lines)
        # last resort: treat as simple
        self.dialect = "simple"
        self.warnings.append("unrecognised header; attempting simple dialect")
        return self._parse_simple(stripped)

    # -- simple dialect --------------------------------------------------
    def _parse_simple(self, lines: List[str]) -> List[GCEvent]:
        out: List[GCEvent] = []
        for ln in lines:
            if ln.startswith("#"):
                if ln.startswith("# end"):
                    m = re.search(r"t=([\d.]+)", ln)
                    if m:
                        self.day_seconds = float(m.group(1))
                continue
            parts = ln.split()
            tag = parts[0]
            kv = dict(_KV_RE.findall(ln))
            try:
                t = float(kv.get("t", "0"))
                pause = float(kv.get("pause", "0"))
            except ValueError:
                continue
            ev = GCEvent(kind="", wall_s=t, pause_ms=pause)

            def rng(key):
                v = kv.get(key, "")
                if "->" in v:
                    a, b = v.split("->", 1)
                    return float(a), float(b)
                try:
                    return float(v), float(v)
                except ValueError:
                    return 0.0, 0.0

            if tag == "YOUNG":
                ev.kind = "young"
                ev.eden_before_mib, ev.eden_after_mib = rng("eden")
                ev.surv_before_mib, ev.surv_after_mib = rng("surv")
                ev.old_before_mib, ev.old_after_mib = rng("old")
                ev.promoted_mib = float(kv.get("promoted", 0))
                ev.tenuring = int(float(kv.get("tenuring", 0) or 0))
            elif tag == "FULL":
                ev.kind = "full"
                ev.old_before_mib, ev.old_after_mib = rng("old")
                ev.live_mib = float(kv.get("live", 0))
                ev.reason = kv.get("reason", "system")
            elif tag == "CONC_START":
                ev.kind = "conc-start"
                ev.old_before_mib, ev.old_after_mib = rng("old")
            elif tag == "CONC_END":
                ev.kind = "conc-end"
                ev.old_before_mib, ev.old_after_mib = rng("old")
            else:
                continue
            out.append(ev)
        self.events = out
        if out:
            self.day_seconds = self.day_seconds or (out[-1].wall_s - out[0].wall_s)
        return out

    # -- hotspot dialect --------------------------------------------------
    def _parse_hotspot(self, lines: List[str]) -> List[GCEvent]:
        blocks: Dict[int, GCEvent] = {}
        order: List[int] = []
        t0: Optional[float] = None
        region_mib = 1.0

        def block(n: int, wall: float) -> GCEvent:
            ev = blocks.get(n)
            if ev is None:
                ev = GCEvent(kind="", wall_s=(wall - t0) if (wall is not None and t0 is not None) else 0.0,
                             pause_ms=0.0)
                blocks[n] = ev
                order.append(n)
            return ev

        for ln in lines:
            m_ts = _TS_RE.match(ln)
            body = ln
            wall = None
            if m_ts:
                wall = _to_epoch(m_ts.group("ts"), m_ts.group("tz"))
                if t0 is None:
                    t0 = wall
                body = ln[m_ts.end():]
            if "heap total" in body:
                mh = re.search(r"heap total (\d+)", body)
                if mh:
                    self.heap_total_mib = float(mh.group(1))
            mg = re.search(r"GC\((\d+)\)", body)
            if not mg:
                continue
            gnum = int(mg.group(1))
            # classify the line
            kind = None
            if "Pause Young (Concurrent Start)" in body or "Pause Initial Mark" in body:
                kind = "conc-start"
            elif "Pause Remark" in body:
                kind = "remark"
            elif "Pause Young" in body:
                kind = "young"
            elif "Pause Full" in body:
                kind = "full"
            ev = block(gnum, wall)
            if kind and not ev.kind:
                ev.kind = kind
            if kind == "full":
                mr = re.search(r"Pause Full \(([^)]*)\)", body)
                if mr:
                    ev.reason = mr.group(1)
            # pause summary line: "... 12.34ms" at the end of the line.  A
            # block can contain the same "Pause Full (...)" text twice (start
            # line and summary line); only the summary carries a duration.
            if re.search(r"([\d.]+)ms\s*$", body):
                mp = re.search(r"([\d.]+)ms\s*$", body)
                if mp:
                    ev.pause_ms = float(mp.group(1))
            mr = re.search(r"Old regions: (\d+)->(\d+)", body)
            if mr:
                ev.old_before_mib = int(mr.group(1)) * region_mib
                ev.old_after_mib = int(mr.group(2)) * region_mib
            mr = re.search(r"Eden regions: (\d+)->(\d+)", body)
            if mr:
                ev.eden_before_mib = int(mr.group(1)) * region_mib
                ev.eden_after_mib = int(mr.group(2)) * region_mib
            mr = re.search(r"Survivor regions: (\d+)->(\d+)", body)
            if mr:
                ev.surv_before_mib = int(mr.group(1)) * region_mib
                ev.surv_after_mib = int(mr.group(2)) * region_mib
            mr = re.search(r"Promotion: ([\d.]+) MiB", body)
            if mr:
                ev.promoted_mib = float(mr.group(1))
            mr = re.search(r"new threshold (\d+)", body)
            if mr:
                ev.tenuring = int(mr.group(1))
            mr = re.search(r"Heap after full GC: live ([\d.]+) MiB", body)
            if mr:
                ev.live_mib = float(mr.group(1))
            mr = re.search(r"used (\d+)K", body)
            if mr:
                used = int(mr.group(1)) / 1024.0 / 1024.0   # K -> MiB
                if ev.kind == "full":
                    if not ev.live_mib:
                        ev.live_mib = used
                    if not ev.old_after_mib:
                        ev.old_after_mib = used
                elif ev.kind in ("young", "conc-start") and not ev.old_after_mib:
                    ev.old_after_mib = used
        out: List[GCEvent] = []
        for n in order:
            ev = blocks[n]
            if not ev.kind:
                continue
            if ev.kind == "remark":
                # remark completes a concurrent cycle
                ev.kind = "conc-end"
            out.append(ev)
        out.sort(key=lambda e: e.wall_s)
        self.events = out
        if out:
            self.day_seconds = out[-1].wall_s - out[0].wall_s
        return out


# --------------------------------------------------------------------------
# Parameter estimation ("inverse calibration")
# --------------------------------------------------------------------------
def estimate_parameters(events: List[GCEvent],
                        day_seconds: float,
                        region_mib: float = 1.0,
                        heap_total_mib: float = 0.0,
                        minor_cost_model: Optional[Tuple[float, float, float]] = None,
                        ) -> dict:
    """Estimate operating parameters from a parsed GC log.

    Returns allocation rate, promotion rate, live-set size, occupancy series,
    implied young-generation size, implied tenuring behaviour, full-GC and
    degeneration counts, and -- when a cost model is supplied -- the implied
    concurrent marking rate and pause coefficients.
    """
    ev = [e for e in events if e.is_stw or e.kind in ("conc-start", "conc-end")]
    ev.sort(key=lambda e: e.wall_s)
    youngs = [e for e in events if e.kind == "young"]
    fulls = [e for e in events if e.kind == "full"]
    conc_starts = [e for e in events if e.kind == "conc-start"]
    conc_ends = [e for e in events if e.kind == "conc-end"]
    if len(events) < 2:
        raise ValueError("need at least two events to estimate parameters")

    t_start = events[0].wall_s
    t_end = events[-1].wall_s
    window = max(t_end - t_start, 1e-6)
    if day_seconds and day_seconds > window:
        window = day_seconds

    # ---- allocation / promotion rates ---------------------------------
    # Every young event reports the eden occupied just before the collection;
    # summing eden_before over the window gives allocated bytes (plus the
    # current eden, which is negligible at these scales).
    total_eden = sum(e.eden_before_mib for e in youngs)
    alloc_mibs = total_eden / window
    # promotion: prefer the explicit counter, else use the old-gen delta
    promoted = sum(e.promoted_mib for e in youngs)
    if promoted <= 0:
        promoted = sum(max(0.0, e.old_after_mib - e.old_before_mib)
                       for e in youngs)
    promoted_mibs = promoted / window
    # The live set is the old generation *after* a sweep.  A full GC reports
    # it directly; a completed concurrent cycle's old_after is the same
    # quantity (the sweep just happened).  Raw old occupancy between cycles
    # is useless for this purpose because it includes floating garbage.
    live_sizes = [e.old_after_mib for e in fulls if e.old_after_mib > 0]
    if not live_sizes:
        live_sizes = [e.old_after_mib for e in conc_ends if e.old_after_mib > 0]
    live_mib = (sorted(live_sizes)[len(live_sizes) // 2]
                if live_sizes else 0.0)

    # ---- young generation sizing --------------------------------------
    # A young event's eden_before is the eden capacity (full).  The survivor
    # capacity is the max survivor_after observed (plus the to-space copy).
    eden_caps = [e.eden_before_mib for e in youngs if e.eden_before_mib > 0]
    eden_cap = _mode(eden_caps) if eden_caps else 0.0
    surv_caps = [max(e.surv_after_mib, e.surv_before_mib) for e in youngs]
    surv_cap = _mode([x for x in surv_caps if x > 0]) if surv_caps else 0.0
    young_mib = eden_cap + 2 * surv_cap if eden_cap else 0.0

    # ---- tenuring -------------------------------------------------------
    thresholds = [e.tenuring for e in youngs if e.tenuring > 0]
    # the adaptive policy ramps the threshold over the first collections;
    # the steady-state value is the mode of the second half
    tail = thresholds[len(thresholds) // 2:] or thresholds
    tenuring = _mode(tail) if tail else 0
    # per-collection survivor survival rate: fraction of eden survivors that
    # survive the next young collection (from old-gen growth and survivor
    # occupancy transitions)
    surv_survival = []
    for a, b in zip(youngs, youngs[1:]):
        if a.surv_after_mib > 0:
            surv_survival.append(min(1.0, b.surv_before_mib / a.surv_after_mib))
    surv_rate = float(sum(surv_survival) / len(surv_survival)) if surv_survival else 0.0
    # implied mean young-object lifetime: interval * -1/ln(surv_rate) only
    # meaningful for a geometric survival curve
    minor_interval = window / len(youngs) if youngs else 0.0
    implied_lifetime = (minor_interval / -math.log(max(surv_rate, 1e-9))
                        if 0 < surv_rate < 1 else 0.0)

    # ---- stalls / degeneration -----------------------------------------
    # A full GC is a degeneration if the log shows an "Allocation Stall" or a
    # full GC that follows a concurrent start closely (marker lost the race).
    degen = 0
    open_cycle = False
    for e in events:
        if e.kind == "conc-start":
            open_cycle = True
        elif e.kind == "conc-end":
            open_cycle = False
        elif e.kind == "full":
            if open_cycle or "Allocation Stall" in e.reason or "to-space" in e.reason:
                degen += 1
            open_cycle = False

    # ---- concurrent marking rate ---------------------------------------
    # A cycle runs from its concurrent start to the next remark (completed)
    # or to the next full GC (degenerated).  Marking must cover the old
    # generation's live set at the start plus the promotions that arrive
    # while it runs, so a completed cycle of duration dt implies
    #     mark_rate ~= (live_set + promotion_rate * dt) / dt.
    # Degenerated cycles only tell us the marker lost the race; they are
    # counted separately and do not enter the rate estimate.
    cycle_pairs = []
    mark_ests = []
    open_ev = None
    for e in events:
        if e.kind == "conc-start":
            open_ev = e
        elif e.kind == "conc-end" and open_ev is not None:
            dt = max(e.wall_s - open_ev.wall_s, 1e-6)
            cycle_pairs.append((open_ev, e, dt, "completed"))
            # work scanned = live at start + promotions that joined the live
            # set during the cycle; the post-sweep occupancy is the live set
            # at remark, so this is a tight lower-ish bound on the rate.
            work = max(e.old_after_mib, 1.0) + promoted_mibs * dt
            mark_ests.append(work / dt)
            open_ev = None
        elif e.kind == "full" and open_ev is not None:
            dt = max(e.wall_s - open_ev.wall_s, 1e-6)
            cycle_pairs.append((open_ev, e, dt, "degenerated"))
            open_ev = None
    mark_mibs = (float(sorted(mark_ests)[len(mark_ests) // 2])
                 if mark_ests else 0.0)

    # ---- pause model fit ------------------------------------------------
    pause_fit = _fit_pause_model(youngs, fulls)

    # ---- occupancy series (sampled) ------------------------------------
    occ = _occupancy_series(events, max(1, int(window // 300) + 1))

    # ---- throughput loss ------------------------------------------------
    stw_s = sum(e.pause_ms for e in ev) / 1000.0
    # concurrent cycles cost CPU; estimate from their duration and a default
    conc_duration = sum(c[2] for c in cycle_pairs)
    conc_cost = conc_duration * 0.20

    return {
        "dialect": None,
        "window_s": window,
        "alloc_mibs": alloc_mibs,
        "promotion_mibs": promoted_mibs,
        "live_set_mib": live_mib,
        "young_capacity_mib": young_mib,
        "eden_capacity_mib": eden_cap,
        "survivor_capacity_mib": surv_cap,
        "tenuring_threshold": tenuring,
        "survivor_survival_rate": surv_rate,
        "implied_object_lifetime_s": implied_lifetime,
        "minor_interval_s": minor_interval,
        "young_gc_count": len(youngs),
        "full_gc_count": len(fulls),
        "degenerate_full_gc_count": degen,
        "concurrent_cycles": len([c for c in cycle_pairs if c[3] == "completed"]),
        "concurrent_degenerated": len([c for c in cycle_pairs if c[3] == "degenerated"]),
        "concurrent_mark_mibs": mark_mibs,
        "stw_seconds": stw_s,
        "throughput_loss_frac": (stw_s + conc_cost) / window if window else 0.0,
        "pause_ms": {
            "mean": _mean([e.pause_ms for e in ev]),
            "p50": _pct([e.pause_ms for e in ev], 50),
            "p99": _pct([e.pause_ms for e in ev], 99),
            "max": max((e.pause_ms for e in ev), default=0.0),
        },
        "pause_model_fit": pause_fit,
        "occupancy": occ,
        "full_reasons": _count(e.reason for e in fulls),
    }


def _fit_pause_model(youngs: List[GCEvent],
                     fulls: List[GCEvent]) -> dict:
    """Least-squares fit of pause = fixed + per_byte * live_bytes.

    For a young collection the log exposes the copied bytes as
    ``survivor_after + promoted`` (the bytes evacuated into the survivor
    space plus those promoted straight to old).  For a full collection the
    log exposes the live set directly.
    """
    def fit(events, live_attr):
        pts = []
        for e in events:
            if live_attr == "copied":
                x = e.surv_after_mib + e.promoted_mib
            else:
                x = getattr(e, live_attr)
            if x > 0:
                pts.append((x, e.pause_ms))
        if len(pts) < 3:
            return None
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if abs(den) < 1e-12:
            # all live sets equal: slope is not identifiable, but the fixed
            # cost is still the observed pause.  Report it explicitly rather
            # than returning nothing.
            return {"fixed_ms": sy / n, "per_mib_ms": 0.0, "r2": 0.0,
                    "n": n, "x_min": min(p[0] for p in pts),
                    "x_max": max(p[0] for p in pts), "x_cv": 0.0,
                    "identifiable": False}
        slope = (n * sxy - sx * sy) / den
        intercept = (sy - slope * sx) / n
        # r^2
        my = sy / n
        ss_tot = sum((p[1] - my) ** 2 for p in pts)
        ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        xs = [p[0] for p in pts]
        x_mean = sx / n
        x_var = sxx / n - x_mean * x_mean
        cv = math.sqrt(max(x_var, 0.0)) / max(abs(x_mean), 1e-12)
        return {"fixed_ms": intercept, "per_mib_ms": slope,
                "r2": r2, "n": n,
                "x_min": min(xs), "x_max": max(xs),
                "x_cv": cv,
                # a slope cannot be identified when every collection has the
                # same live size (e.g. every full GC is triggered by the same
                # panic threshold)
                "identifiable": cv >= 0.03 and (max(xs) - min(xs)) > 4.0}
    return {
        "young": fit(youngs, "copied"),
        "full": fit(fulls, "live_mib"),
    }


def _occupancy_series(events: List[GCEvent], buckets: int) -> Dict[str, list]:
    """Old-generation occupancy sampled from post-GC values."""
    if not events:
        return {"t": [], "old": [], "young": []}
    t0 = events[0].wall_s
    t1 = events[-1].wall_s
    span = max(t1 - t0, 1e-6)
    buckets = max(buckets, 2)
    width = span / buckets
    series = {"t": [], "old": [], "young": []}
    idx = 0
    cur_old = events[0].old_after_mib
    cur_young = 0.0
    for b in range(buckets):
        target = t0 + (b + 1) * width
        while idx < len(events) and events[idx].wall_s <= target:
            e = events[idx]
            cur_old = e.old_after_mib
            cur_young = e.surv_after_mib + e.eden_after_mib
            idx += 1
        series["t"].append(target - t0)
        series["old"].append(cur_old)
        series["young"].append(cur_young)
    return series


def _mode(xs: Iterable[float]) -> float:
    xs = [round(x, 3) for x in xs]
    if not xs:
        return 0.0
    counts: Dict[float, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def _count(xs: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        x = x or "unknown"
        out[x] = out.get(x, 0) + 1
    return out


def parse_log(text: str) -> Tuple[List[GCEvent], dict]:
    p = LogParser()
    events = p.parse(text)
    est = estimate_parameters(events, p.day_seconds or 0.0,
                              heap_total_mib=p.heap_total_mib)
    est["dialect"] = p.dialect
    est["warnings"] = p.warnings
    est["region_mib"] = p.region_mib
    return events, est
