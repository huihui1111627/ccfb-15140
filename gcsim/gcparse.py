"""Parser for GC logs in HotSpot ``-Xlog:gc`` / G1 style.

Accepts the format written by :mod:`gcsim.gclog` and real logs with the usual
variations:

* ISO timestamps (``2026-11-23T09:30:00.123+0800``) or bare elapsed times
  (``[12.345s]``, ``2026-11-23T09:30:00.123+0800: 12.345:``);
* one-line summaries and the two-line G1 shape;
* units ``B/K/M/G`` with or without ``B`` suffix, and ``0.0041234 secs``;
* young pauses, mixed, remark, concurrent phases and full GCs;
* ``Full GC (...) 3G->1.2G(4G), 0.4871234 secs`` as emitted by ParallelGC
  or SerialGC.

The parser's job is only to produce a stream of :class:`GCEvent`.  All
interpretation (allocation rate, promotion rate, live-size estimation) lives in
:mod:`gcsim.calibrate`.
"""

from __future__ import annotations

import re
from typing import List

from .gclog import GCEvent

_UNITS = {
    "": 1.0,
    "B": 1.0,
    "K": 1024.0,
    "M": 1024.0 ** 2,
    "G": 1024.0 ** 3,
    "T": 1024.0 ** 4,
}


def parse_size(s: str) -> float:
    s = s.strip().rstrip(",")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([BKMGT]?)B?", s.upper().replace(" ", ""))
    if not m:
        raise ValueError(f"cannot parse size {s!r}")
    return float(m.group(1)) * _UNITS.get(m.group(2), 1.0)


_SIZE = r"([0-9]*\.?[0-9]+[BKMG]?B?)"

_TS = r"(?:\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-]\d{4})?)?\s*(?:\[?\s*)?(\d+(?:\.\d+)?)s?"

# Two-line G1 young pause:
# 2026-...: [   0.123s] GC pause (G1 Evacuation Pause) (young), 0.0041234 secs
#   2026-...: [   0.123s] [Eden: 24.0M(24.0M)->0.0B(24.0M) Survivors: 4.0M->5.0M Heap: ...]
_YOUNG_HEAD = re.compile(
    r"GC pause \(([^)]*)\)\s*(\(young\)|\(mixed\))?[^,]*,?\s*([0-9.]+)\s*secs",
    re.IGNORECASE,
)
_OLD_FIELD = re.compile(r"\[Old:\s*" + _SIZE + r"->" + _SIZE + r"\]",
                        re.IGNORECASE)
_PROMOTED_FIELD = re.compile(r"\[Promoted:\s*" + _SIZE + r"\]", re.IGNORECASE)

_YOUNG_DETAIL = re.compile(
    r"Eden:\s*"
    + _SIZE + r"\(" + _SIZE + r"\)->" + _SIZE + r"\(" + _SIZE + r"\)"
    r"\s+Survivors:\s*" + _SIZE + r"->" + _SIZE +
    r"\s+Heap:\s*" + _SIZE + r"\(" + _SIZE + r"\)->" + _SIZE + r"\(" + _SIZE + r"\)",
    re.IGNORECASE,
)

# Full GC:  Full GC (Allocation Failure) 3G->1.2G(4G), 0.4871234 secs
# Sometimes: [Full GC (Ergonomics) [PSYoungGen: ...] [ParOldGen: ...] 3G->1.2G(4G), 0.48 secs]
_FULL_LINE = re.compile(
    r"Full GC \(([^)]*)\)[^0-9]*" + _SIZE + r"->" + _SIZE + r"\(" + _SIZE +
    r"\),\s*([0-9.]+)\s*secs",
    re.IGNORECASE,
)

_CONC = re.compile(r"GC concurrent-([a-z-]+?)(?:-start|-end)?\b", re.IGNORECASE)

_REMARK = re.compile(
    r"GC pause \(([^)]*)\)[^,]*(?:\[(?:concurrent-)?([a-z-]+)\])?[^,]*,?\s*"
    r"([0-9.]+)\s*secs",
    re.IGNORECASE,
)

_TIME_BRACKET = re.compile(r"\[\s*(\d+(?:\.\d+)?)s\s*\]")
_TIME = re.compile(r"(?:^|\s)\[?\s*(\d+(?:\.\d+)?)s\s*\]?")
_TIME2 = re.compile(r":\s*(\d+(?:\.\d+)?):\s")


def _event_time(line: str, prev_t: float) -> float:
    """Extract elapsed seconds from a log line, with fallbacks."""
    m = _TIME_BRACKET.search(line)
    if m:
        return float(m.group(1))
    m = _TIME.search(line)
    if m:
        return float(m.group(1))
    m = _TIME2.search(line)
    if m:
        return float(m.group(1))
    # ISO wall clock: convert to elapsed by differencing; the caller passes
    # prev_t and we use it only as an ordering hint.
    return prev_t


_HEADER_NEWSIZE = re.compile(r"NewSize:\s*" + _SIZE + r"", re.IGNORECASE)
_HEADER_RATIO = re.compile(r"SurvivorRatio:\s*(\d+)", re.IGNORECASE)
_HEADER_HEAPMAX = re.compile(r"Heap Max Capacity:\s*" + _SIZE, re.IGNORECASE)
_HEADER_IHOP = re.compile(r"InitiatingHeapOccupancyPercent:\s*([0-9.]+)",
                          re.IGNORECASE)
_HEADER_TENURE = re.compile(r"MaxTenuringThreshold:\s*(\d+)", re.IGNORECASE)


def parse_header(text: str) -> dict:
    """Extract the JVM geometry lines from an ``-Xlog:gc`` header, if present.

    Returns a dict with any of: ``young_size``, ``survivor_ratio`` (per-space
    fraction of young), ``heap_max``, ``initiating``, ``tenuring_threshold``.
    Values are bytes/floats.  Missing keys simply are not set.  This is what
    lets calibration report the *configured* survivor ratio rather than
    guessing it from occupancy (a survivor space is often only partly used).
    """
    out = {}
    m = _HEADER_RATIO.search(text)
    if m:
        sr = int(m.group(1))
        # HotSpot SurvivorRatio = eden / each survivor space; young = eden +
        # 2*space = space*(sr+2), so space/young = 1/(sr+2).
        out["survivor_ratio"] = 1.0 / (sr + 2)
    m = _HEADER_NEWSIZE.search(text)
    if m:
        out["young_size"] = parse_size(m.group(1))
    m = _HEADER_HEAPMAX.search(text)
    if m:
        out["heap_max"] = parse_size(m.group(1))
    m = _HEADER_IHOP.search(text)
    if m:
        out["initiating"] = float(m.group(1)) / 100.0
    m = _HEADER_TENURE.search(text)
    if m:
        out["tenuring_threshold"] = int(m.group(1))
    return out


def parse_gc_log(text: str) -> List[GCEvent]:
    events: List[GCEvent] = []
    lines = text.splitlines()
    i = 0
    prev_t = 0.0
    n = len(lines)
    while i < n:
        line = lines[i]
        low = line.lower()
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        # Full GC must be tested before the concurrent-phase branch: a
        # concurrent-mode-failure full GC line contains the word "concurrent"
        # in its cause and would otherwise be misclassified as a marking
        # event.
        if "full gc" in low:
            m = _FULL_LINE.search(line)
            t = _event_time(line, prev_t)
            if m:
                events.append(GCEvent(
                    t=t, kind="full", cause=m.group(1).strip(),
                    pause_s=float(m.group(5)),
                    heap_before=parse_size(m.group(2)),
                    heap_after=parse_size(m.group(3)),
                    heap_capacity=parse_size(m.group(4)),
                    raw=line,
                ))
            else:
                pm = re.search(r"([0-9.]+)\s*secs", line)
                events.append(GCEvent(
                    t=t, kind="full", cause="Full GC",
                    pause_s=float(pm.group(1)) if pm else 0.0, raw=line,
                ))
            prev_t = max(prev_t, t)
            i += 1
            continue
        if "concurrent" in low and "gc" in low and "pause" not in low:
            m = _CONC.search(line)
            phase = (m.group(1) if m else "mark").lower()
            t = _event_time(line, prev_t)
            events.append(GCEvent(t=t, kind="concurrent", cause="concurrent",
                                  phase=phase, raw=line))
            prev_t = max(prev_t, t)
            i += 1
            continue
        if "gc pause" in low:
            m = _YOUNG_HEAD.search(line)
            if not m:
                i += 1
                continue
            cause = m.group(1).strip()
            genpart = (m.group(2) or "").lower()
            pause = float(m.group(3))
            t = _event_time(line, prev_t)
            # Peek at the next line(s) for the Eden/Survivors/Heap detail.
            eden_b = eden_a = eden_cap = 0.0
            surv_b = surv_a = 0.0
            heap_b = heap_a = heap_cap = 0.0
            if i + 1 < n:
                dm = _YOUNG_DETAIL.search(lines[i + 1])
                if dm:
                    g = [parse_size(x) for x in dm.groups()]
                    eden_b, eden_cap, eden_a, _eden_cap2 = g[0:4]
                    surv_b, surv_a = g[4], g[5]
                    heap_b, heap_cap, heap_a, _hc2 = g[6:10]
            old_b = old_a = 0.0
            promoted_cum = 0.0
            if i + 1 < n:
                om = _OLD_FIELD.search(lines[i + 1])
                if om:
                    old_b = parse_size(om.group(1))
                    old_a = parse_size(om.group(2))
                pm = _PROMOTED_FIELD.search(lines[i + 1])
                if pm:
                    promoted_cum = parse_size(pm.group(1))
            is_remark = "remark" in low or "cleanup" in low or "mixed" in low
            if "remark" in low or "cleanup" in low:
                kind = "remark"
            elif "mixed" in genpart or "mixed" in low:
                kind = "mixed"
            else:
                kind = "young"
            events.append(GCEvent(
                t=t, kind=kind, cause=cause, pause_s=pause,
                heap_before=heap_b, heap_after=heap_a, heap_capacity=heap_cap,
                eden_before=eden_b, eden_after=eden_a, eden_capacity=eden_cap,
                surv_before=surv_b, surv_after=surv_a,
                young_after=eden_a + surv_a, old_before=old_b, old_after=old_a,
                promoted_cum=promoted_cum,
                raw=line + ("\n" + lines[i + 1] if i + 1 < n else ""),
            ))
            prev_t = max(prev_t, t)
            i += 2
            continue
        i += 1
    events.sort(key=lambda e: e.t)
    return events


def parse_gc_log_file(path: str) -> List[GCEvent]:
    with open(path, "r", errors="replace") as f:
        return parse_gc_log(f.read())
