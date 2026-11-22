# gcsim -- a generational GC simulaton and parameter-tuning tool

A pure-Python model of a generational garbage collector of the kind that runs
a trading front end on the JVM (G1-style): a young generation with an eden and
two survivor spaces, an old generation with concurrent marking, promotion by
age, pre-tenuring under space pressure, and full-GC degradation when
concurrent marking loses the race with allocation.

It exists to answer questions that are otherwise folklore:

* *Why do 100-600 ms pauses appear every few minutes even though the average
  pause is a couple of milliseconds?* -- because a concurrent mark lost the
  race with promotion and degraded to a full STW collection.
* *Why did lowering the young generation fix pauses but halve throughput?* --
  because the survivor space shrank with it, the tenure threshold collapsed,
  and everything was promoted early, which filled old and caused full GCs (or,
  in the "fixed" case, thousands of tiny scavenges).
* *Which parameter is actually binding?* -- the tuner reports the response of
  each knob, not just a winner.

Everything here is *modelled*: heap, allocation, and collector behaviour are
all our own code. Nothing connects to a real JVM. The GC log format is
HotSpot's, so logs can be exchanged with real tooling, and one of the modes is
"calibrate the model from a real log".

---

## Quick start

```bash
python3 -m gcsim suite --outdir reports        # full day, 4 services, tuning,
                                               # reverse calibration, HTML report
open reports/index.html                        # charts, before/after, calibration

python3 -m gcsim simulate --service session_heavy --traffic trading_day \
    --young-mb 2048 --tenure-age 7                 # one config, JSON metrics

python3 -m gcsim calibrate reports/logs/*.gc.log    # reverse-calibrate a log

python3 -m gcsim log --service cache_heavy -o gc.log  # write a GC log
python3 -m gcsim serve                         # interactive browser UI

python3 -m pytest -q                           # mechanism tests + oracle check
```

Requires Python 3.8+, `numpy`; `scipy` is used if present (a NumPy fallback
keeps it optional). No web framework, no plotting library: charts are inline
SVG.

---

## What is modelled

### Heap layout (`gcsim/config.py`)

```
             young generation                         old generation
   +-------------+----------+----------+     +-----------------------------+
   |    eden     | survivor | survivor |     |  regioned mark-sweep space  |
   |             |  (to)    |  (from)  |     |  (concurrent + full GC)     |
   +-------------+----------+----------+     +-----------------------------+
```

* `young_size`, `survivor_ratio` (each survivor space = ratio x young),
  `old_size`.
* Every cost constant (copy ns/byte, per-object copy cost, remark ns/live
  byte, full-GC ns/live MB, compaction) is in the config so it can be
  calibrated rather than hand-waved.

### Object lifetimes (`gcsim/lifetime.py`)

Each traffic class has a **survival function** `S(a) = P(object still reachable
a seconds after allocation)`. The models are the ones the user described:

| class     | model                        | role                              |
|-----------|------------------------------|-----------------------------------|
| request   | lognormal median ~5-10 ms + 3% slow tail | dies young almost always |
| session   | lognormal median ~45-90 s + heavy tail | **medium-lived: survives several scavenges, prone to premature promotion** |
| cache     | Pareto, alpha ~1.1-1.25      | long-lived; forms the old-gen floor |
| metadata  | Pareto with a day-long cap   | effectively immortal              |

An object does **not** die because a GC happened; it dies when its lifetime
ends. Whether the collector *notices* before it has promoted the object is
exactly what the tool studies.

### Collector mechanism (`gcsim/simulator.py`)

A scavenge:

1. every live eden object is copied to the to-space, age becomes 1;
2. every live survivor is copied with age+1;
3. an object whose age reaches the **effective tenure age** is promoted;
4. if the survivors do not fit the to-space, the **oldest are pre-tenured**
   (adaptive tenuring -- HotSpot's `DesiredSurvivorSizePolicy`). The effective
   tenure age is the highest `t <= promotion_age` whose kept bytes fit the
   survivor space; if even age 1 overflows, that is to-space exhaustion and
   everything moves.

That single rule explains most of the tuning confusion: **when the survivor
space is too small, the `promotion_age` parameter is inert** -- the collector
lowers the effective threshold to fit the space, so changing the configured
age does nothing. `tests/test_gcsim.py::test_small_survivor_space_forces_early_promotion`
pins this.

Old generation:

* **concurrent** (default): a marking cycle starts at
  `initiating_occupancy * old_size`. Marking scans at `mark_scan_bytes_per_s`.
  Its work is the live set at cycle start **plus everything promoted during
  the cycle** (SATB). If promotion outruns the scan rate, or old occupancy
  reaches `emergency_occupancy` before the cycle finishes, the cycle fails:
  it **degrades to a full STW collection**. This is the 100-600 ms pause.
* **stop_the_world**: every old collection is a full pause.

Pause model: scavenge = fixed cost + survivors x ns/byte x graph factor +
objects x ns/object + promoted bytes x RS-fixup; remark = base + live old x
ns/byte; full GC = base + live x ns/byte (+ compaction). Calibrated so 130 MB
of survivors is ~27 ms and a 2 GB remark ~12 ms, in line with published G1
figures.

### Why cohorts, and why it is exact enough (`gcsim/workload.py`)

A trading day is ~10^9 allocations. Tracking each object is pointless: the
collector's behaviour depends on *aggregate bytes* at each age. So the
workload is a set of **cohorts** -- all bytes of one class born in a small
time slice -- and every question is answered in expectation:

```
bytes of a cohort still live at t  = V * S(t - birth)
bytes surviving [a, b)             = V * (S(a) - S(b))
```

`gcsim/oracle.py` implements the **same mechanism object by object** (real
objects, real sampled lifetimes, real per-object ages). The test
`test_cohort_matches_object_oracle` runs both and checks that the promoted
fraction of allocated bytes and the scavenge count agree. That is the
justification for the approximation; it is validated, not asserted.

Old occupancy is tracked in two layers, as in a real collector: *committed*
(bytes promoted and not yet reclaimed -- what triggers collection) and *live*
(expected reachable bytes from the conditional survival `S(t-b)/S(tp-b)` of
each promoted bucket -- what marking discovers). Getting this wrong is the
single most common way a GC model becomes nonsense; the code documents it.

### Traffic (`gcsim/config.py`)

A 24-hour multiplier curve with pre-open ramp, open burst, lunch dip, close
spike and an evening settlement batch, plus Poisson noise. Peak/trough ratio
is ~15x, so a parameter set that is fine at night can fail at the open -- the
evaluation always covers the whole day.

---

## The three tools

### 1. Logs, both ways (`gcsim/gclog.py`, `gcsim/gcparse.py`, `gcsim/calibrate.py`)

* **Write** HotSpot-style unified GC logs, including an init header with
  `NewSize`, `SurvivorRatio`, `InitiatingHeapOccupancyPercent`,
  `MaxTenuringThreshold`, and per-collection `Eden/Survivors/Heap` plus
  `[Old: a->b]` and `[Promoted: cum]`.
* **Read** real logs too: ISO timestamps or bare elapsed times, one-line or
  two-line G1 shape, ParallelGC full-GC lines, with or without units.
* **Calibrate**: from the log alone estimate
  * allocation rate -- eden refilled between scavenges, integrated over time;
  * promotion rate -- cumulative promoted bytes over the day (exact when the
    `[Promoted:]` field is present; otherwise the rise in old occupancy
    between scavenges, excluding intervals that collect old);
  * survivor bytes per scavenge, survivor ratio, young/eden/old geometry,
    old occupancy trend, pause percentiles.

Reverse calibration of a simulated log recovers allocation and promotion
rates to **<3%** and the heap geometry to **0%** (see the report's
"Reverse calibration" table). This is the "estimate the parameters back from
the log" acceptance item.

### 2. Tuner (`gcsim/tuner.py`)

Searches **young size x survivor ratio x tenure age x initiating occupancy**
(survivor ratio moves with young size because that is how the space is really
sized). Every point is a whole simulated day, evaluated in parallel. Objectives
are hard: `max pause <= target` **and** `throughput >= target`.

The chosen point is not "lowest pause". `_pick_best` ranks by **robustness
margin** -- the smaller of the pause and throughput headroom, normalised -- so
the winner is not one that scrapes past one objective by sacrificing the
other, and not the degenerate "halve young, tenure age 1" fix that only looks
good because it promotes everything. The result includes:

* the baseline, the best feasible point, and the fraction of the grid that is
  feasible;
* a **per-parameter sensitivity table** (mean pause/throughput/promotion/full
  GCs at each value), and
* plain-language **consequences** for what each knob does in both directions.

### 3. Report (`gcsim/report.py`, `gcsim/page.py`, `gcsim/webui.py`)

`reports/index.html` is a self-contained page: per-generation occupancy over
the day, every pause as a point, pause distribution, promoted-bytes-by-age,
survivor-space occupancy, STW time per hour, before/after comparison, the
chosen parameters, and the reverse-calibration table. `reports/results.json`
has the same data. `python3 -m gcsim serve` runs a small HTTP UI that can run a
new scenario from the browser.

---

## What the acceptance suite shows

Full day, five scenarios (`reports/summary.md`, `reports/index.html`). Baseline
is the default configuration; each scenario also has a realistic heap budget.

| scenario | heap | baseline max pause | baseline throughput | tuned max pause | tuned throughput |
|---|---|---|---|---|---|
| trading-front-end / trading day | 6 GB | 117 ms | 99.86% | **12 ms** | 99.78% |
| session-heavy / trading day | 16 GB | 136 ms | 98.89% | **24 ms** | 99.24% |
| cache-heavy / batch day | 16 GB | 117 ms | 99.05% | **21 ms** | 99.31% |
| short-lived / flat | 6 GB | 18 ms | 99.98% | **7 ms** | 99.87% |
| cache-heavy / trading day | **3 GB** | **1316 ms** | 90.4% | *infeasible* | — |

Reading it:

* **The 100 ms pauses are the young generation, not the old one.** The default
  2 GB young gen copies ~100 MB of survivors every 80 s; each copy is a
  ~117 ms pause. Sizing the young generation and its survivor space correctly
  removes them. This is the lesson the team kept rediscovering by trial and
  error.
* **Medium-lived sessions are the premature-promotion class.** In
  session-heavy they fill old quickly when the survivor space is too small;
  with enough survivor space the configured tenure age actually applies and
  promotion falls.
* **A marking load has a throughput cost**, ~1% when a promotion-heavy service
  keeps the marker busy; the tuned points report the marking-adjusted
  throughput, not a flattering STW-only figure.
* **The last row is the honest failure.** With a 1.5 GB live set on a 3 GB old
  generation and a 0.5% throughput target, *no* combination of young size,
  tenure age and initiating occupancy helps: the baseline shows the 1.3 s
  pauses the user described (full GCs after concurrent-mode failure), and the
  tool says the binding constraint is the heap budget, not the three knobs.
  That is the answer the team never got from guessing parameters.

On every scenario the reverse calibration recovers the heap geometry exactly
(0% error) and the allocation/promotion rates to within 3% from the log text
alone.

The exact numbers are in `reports/summary.md`; they are model output, and the
report labels them as such. The reverse-calibration table is the check that
the model and the log agree with each other.

---

## Honest limitations

* It is a **model**. Absolute pause values depend on the cost constants, which
  are set from published G1 behaviour and are configurable. Relative
  comparisons between configurations are the trustworthy part; a single
  absolute number should be calibrated against one real log with
  `gcsim calibrate` before being quoted.
* The cohort engine answers in expectation and does not model per-object
  allocation-site behaviour, humongous objects, remembered-set region
  granularity, or GC-thread count effects beyond the aggregate marking rate.
  The oracle covers the first of these for validation.
* The tuner searches a purposeful grid, not the whole continuous space; it
  will miss a feasible island that the grid does not touch. It reports the
  feasible fraction so this is visible.

---

## Layout

```
gcsim/
  config.py      heap / collector / traffic / service definitions
  lifetime.py    survival-function models (request, session, cache, ...)
  survival.py    scalar survival + quantiles
  workload.py    cohort workload construction
  simulator.py   the cohort collector engine (main model)
  oracle.py      object-by-object collector (validation)
  gclog.py       GC log writer
  gcparse.py     GC log parser (and header geometry)
  calibrate.py   reverse calibration from logs
  tuner.py       parameter search + consequence analysis
  report.py      metrics, digests, SVG charts, comparisons
  page.py        static HTML report
  webui.py       interactive server
  cli.py         command line
tests/test_gcsim.py
```
