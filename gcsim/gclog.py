"""Collector event log in a familiar GC/trace format.

The text format mirrors the one emitted by OpenJDK's ``-Xlog:gc`` unified
logging and by ``G1GC`` log parsers, so it can be eyeballed by a human and
parsed by the same regular expressions used on production logs:

    2026-11-23T09:30:00.123+0800: [GC pause (G1 Evacuation Pause) (young),
    0.0041234 secs] [Eden: 24.0M(24.0M)->0.0B(24.0M) Survivors: 4.0M->5.0M
    Heap: 900.0M(4096.0M)->450.1M(4096.0M)], [Metaspace: ...]

    2026-11-23T09:31:14.004+0800: [GC concurrent-root-region-scan-start]
    2026-11-23T09:31:14.912+0800: [GC concurrent-mark-start]
    2026-11-23T09:31:17.113+0800: [GC pause (G1 Humongous Allocation) (young)...]

    2026-11-23T21:31:41.000+0800: [Full GC (Allocation Failure) 3G->1.2G(4G),
    0.4871234 secs]

The writer emits a header block and a footer summary, like a real rotating GC
log with JVM flags.  The parser in ``gcsim.parse_gc_log`` deliberately accepts
small variations (with/without date, commas, different units, the older
ParallelGC line shape) so that real logs can be fed to it.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Tuple

MB = 1024.0 * 1024.0


def fmt_bytes(b: float) -> str:
    """Format like HotSpot: 512.0M, 1.2G, 0.0B."""
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.1f}G"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f}M"
    if b >= 1024:
        return f"{b / 1024:.1f}K"
    return f"{b:.0f}B"


def _ts(base: _dt.datetime, t: float) -> str:
    d = base + _dt.timedelta(seconds=t)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}"


class GCEvent:
    """One parsed GC event.

    ``cause`` is the collection type.  The parser normalises the many
    collector-family spellings into a small enum:

        young      -- scavenge / young GC / G1 Evacuation Pause (young)
        mixed      -- G1 mixed collection (young + some old regions)
        full       -- full STW collection of the whole heap
        concurrent -- concurrent marking progress events (no pause)
        remark     -- STW remark / cleanup pauses

    ``kind`` distinguishes pauses from bookkeeping events.  ``pause_s`` is the
    stop-the-world time in seconds (0 for concurrent events).
    """

    __slots__ = (
        "t", "kind", "cause", "pause_s", "heap_before", "heap_after",
        "heap_capacity", "eden_before", "eden_after", "surv_before",
        "surv_after", "raw", "phase", "old_before", "old_after",
        "eden_capacity", "young_after", "promoted_cum",
    )

    def __init__(self, t, kind, cause, pause_s=0.0, heap_before=0,
                 heap_after=0, heap_capacity=0, eden_before=0, eden_after=0,
                 surv_before=0, surv_after=0, raw="", phase="", old_before=0,
                 old_after=0, eden_capacity=0, young_after=0, promoted_cum=0.0):
        self.t = t
        self.kind = kind
        self.cause = cause
        self.pause_s = pause_s
        self.heap_before = heap_before
        self.heap_after = heap_after
        self.heap_capacity = heap_capacity
        self.eden_before = eden_before
        self.eden_after = eden_after
        self.surv_before = surv_before
        self.surv_after = surv_after
        self.old_before = old_before
        self.old_after = old_after
        self.raw = raw
        self.phase = phase
        self.eden_capacity = eden_capacity
        self.young_after = young_after
        self.promoted_cum = promoted_cum

    @property
    def is_pause(self) -> bool:
        return self.pause_s > 0.0

    def __repr__(self) -> str:
        return (
            f"GCEvent(t={self.t:.3f}, {self.kind}/{self.cause}, "
            f"pause={self.pause_s * 1e3:.2f}ms, heap={fmt_bytes(self.heap_after)})"
        )


def _survivor_ratio_to_flag(ratio: float) -> str:
    """Convert survivor ratio (of young size, per space) to HotSpot's
    SurvivorRatio flag: eden / each survivor space = (1-2r)/r."""
    r = max(0.01, min(0.45, ratio))
    return str(int(round((1 - 2 * r) / r)))


class GCLogWriter:
    """Writes events to a file or a list of strings."""

    def __init__(self, stream=None, base_date: Optional[_dt.datetime] = None,
                 header: bool = True, jvm_args: Optional[List[str]] = None):
        self.stream = stream
        self.base = base_date or _dt.datetime(2026, 11, 23, 0, 0, 0)
        self.lines: List[str] = []
        self.jvm_args = jvm_args or [
            "-XX:+UseG1GC",
            "-Xms4g", "-Xmx4g",
            "-XX:MaxGCPauseMillis=200",
        ]
        self._started = False
        self._header = header
        self._heap = None
        self.now: float = 0.0

    def _emit(self, s: str) -> None:
        self.lines.append(s)
        if self.stream is not None:
            self.stream.write(s + "\n")

    def set_heap(self, young_size: int, old_size: int, survivor_ratio: float,
                 initiating: float, promotion_age: int, mode: str) -> None:
        """Record the heap geometry the log's header advertises."""
        self._heap = dict(young_size=young_size, old_size=old_size,
                          survivor_ratio=survivor_ratio,
                          initiating=initiating, promotion_age=promotion_age,
                          mode=mode)

    def start(self) -> None:
        if self._header:
            h = getattr(self, "_heap", None)
            self._emit("[0.000s][info][gc] Using G1")
            self._emit(
                "[0.000s][info][gc,init] Version model-gcsim (1.0) "
                + " ".join(self.jvm_args))
            self._emit("[0.000s][info][gc,init] Heap Region Size: 1M")
            if h:
                total = h["young_size"] + h["old_size"]
                self._emit(f"[0.000s][info][gc,init] NewSize: "
                           f"{h['young_size']/1024/1024:.0f}M "
                           f"SurvivorRatio: {_survivor_ratio_to_flag(h['survivor_ratio'])}")
                self._emit(f"[0.000s][info][gc,init] Heap Min Capacity: "
                           f"{total/1024/1024:.0f}M")
                self._emit(f"[0.000s][info][gc,init] Heap Initial Capacity: "
                           f"{total/1024/1024:.0f}M")
                self._emit(f"[0.000s][info][gc,init] Heap Max Capacity: "
                           f"{total/1024/1024:.0f}M")
                self._emit(f"[0.000s][info][gc,init] InitiatingHeapOccupancyPercent: "
                           f"{h['initiating']*100:.0f}")
                self._emit(f"[0.000s][info][gc,init] MaxTenuringThreshold: "
                           f"{h['promotion_age']}")
                self._emit(f"[0.000s][info][gc,init] Concurrent collection: "
                           f"{h['mode']}")
            self._emit("[0.000s][info][gc,init] ConcGCThreads: 4")
            self._emit("[0.000s][info][gc,init] ParallelGCThreads: 8")
        self._started = True

    def _head(self, t: float) -> str:
        return f"[{t:9.6f}s]"

    def young(
        self, t: float, pause: float, eden_b: float, eden_a: float,
        eden_cap: float, surv_b: float, surv_a: float, heap_b: float,
        heap_a: float, heap_cap: float, cause: str = "G1 Evacuation Pause",
        pretenured: int = 0, old_b: Optional[float] = None,
        old_a: Optional[float] = None, promoted_cum: Optional[float] = None,
    ) -> str:
        line = (
            f"{_ts(self.base, t)}: {self._head(t)} "
            f"GC pause ({cause}) (young), {pause:.7f} secs\n"
            f"   {_ts(self.base, t)}: {self._head(t)} "
            f"[Eden: {fmt_bytes(eden_b)}({fmt_bytes(eden_cap)})->{fmt_bytes(eden_a)}"
            f"({fmt_bytes(eden_cap)}) Survivors: {fmt_bytes(surv_b)}->{fmt_bytes(surv_a)}"
            f" Heap: {fmt_bytes(heap_b)}({fmt_bytes(heap_cap)})->"
            f"{fmt_bytes(heap_a)}({fmt_bytes(heap_cap)})], "
            f"[Metaspace: 102400K->102400K(1056768K)]"
        )
        if old_a is not None:
            ob = fmt_bytes(old_b) if old_b is not None else "?"
            line += f" [Old: {ob}->{fmt_bytes(old_a)}]"
        if promoted_cum is not None:
            line += f" [Promoted: {fmt_bytes(promoted_cum)}]"
        if pretenured:
            line += f", [Pretenured: {pretenured}]"
        self._emit(line)
        return line

    def concurrent(self, t: float, phase: str) -> str:
        line = f"{_ts(self.base, t)}: [{self._head(t)}] GC concurrent-{phase}"
        self._emit(line)
        return line

    def remark(
        self, t: float, pause: float, heap_b: float, heap_a: float,
        heap_cap: float, cause: str = "G1 Evacuation Pause",
        phase: str = "remark", mark_kept_up: bool = True,
    ) -> str:
        tag = "concurrent-mark-remark" if phase == "remark" else phase
        tail = "" if mark_kept_up else " (Marking Overloaded)"
        line = (
            f"{_ts(self.base, t)}: [{self._head(t)}] GC pause ({cause}) "
            f"(young) [{tag}]{tail}, {pause:.7f} secs\n"
            f"   {_ts(self.base, t)}: {self._head(t)} "
            f"[Heap: {fmt_bytes(heap_b)}({fmt_bytes(heap_cap)})->"
            f"{fmt_bytes(heap_a)}({fmt_bytes(heap_cap)})]"
        )
        self._emit(line)
        return line

    def full(
        self, t: float, pause: float, heap_b: float, heap_a: float,
        heap_cap: float, cause: str = "Allocation Failure",
    ) -> str:
        line = (
            f"{_ts(self.base, t)}: [{self._head(t)}] "
            f"Full GC ({cause}) {fmt_bytes(heap_b)}->{fmt_bytes(heap_a)}"
            f"({fmt_bytes(heap_cap)}), {pause:.7f} secs"
        )
        self._emit(line)
        return line

    def footer(self, summary: str) -> None:
        self._emit(f"# {summary}")


class _FileSink:
    def __init__(self, path: str):
        self.f = open(path, "w")

    def write(self, s: str) -> None:
        self.f.write(s)

    def close(self) -> None:
        self.f.close()


def open_writer(path: Optional[str], start: bool = True,
                **kw) -> Tuple[GCLogWriter, Optional[_FileSink]]:
    """Open a writer.  ``start=False`` defers the header so the caller can call
    :meth:`GCLogWriter.set_heap` first (the header advertises the heap
    geometry)."""
    sink = _FileSink(path) if path else None
    w = GCLogWriter(stream=sink, **kw)
    if start:
        w.start()
    return w, sink
