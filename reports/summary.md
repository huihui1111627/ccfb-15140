# Generational GC tuning report

Generated 2026-11-23T07:09:48

Every number below comes from the simulator in `gcsim/`. Pauses, promotion and heap figures are model output; the reverse-calibration table at the end of each section shows how well the log estimator recovers the known parameters.

## trading_front_end / trading_day

Objective: pause <= 50 ms, throughput >= 99.00%

| metric | baseline | tuned |
|---|---|---|
| young_gcs | 1072 | 6130 |
| remark_pauses | 7 | 71 |
| full_gcs | 0 | 0 |
| degraded_cycles | 0 | 0 |
| pause_p50_ms | 74.67 | 8.29 |
| pause_p99_ms | 111.6 | 10.8 |
| pause_max_ms | 116.7 | 12.4 |
| throughput_stw | 99.91428% | 99.94323% |
| throughput_adj | 99.85716% | 99.77706% |
| promotion_gb | 17.0 | 172.8 |
| promotion_fraction | 1.48% | 15.00% |
| pretenured_gb | 0.0 | 172.8 |
| old_peak_committed_gb | 2.93 | 3.33 |

Chosen parameters: young 256 MB, survivor ratio 0.15, tenure age 7, initiating occupancy 0.50; giving max pause 12.4 ms, throughput 99.777%, full GCs 0. 58 of 96 evaluated configurations met both objectives.

Per-parameter response (mean over the grid):

| knob | value | mean max pause | mean p99 | mean throughput | mean promotion | mean full GCs |
|---|---|---|---|---|---|---|
| young MB | 153.60 | 14.0 ms | 8.8 ms | 99.400% | 325.8 GB | 0.0 |
| young MB | 256.00 | 24.5 ms | 20.6 ms | 99.363% | 258.0 GB | 0.0 |
| young MB | 384.00 | 18.6 ms | 18.0 ms | 99.774% | 147.4 GB | 0.0 |
| young MB | 512.00 | 50.1 ms | 27.7 ms | 99.714% | 124.3 GB | 0.0 |
| young MB | 1024.00 | 54.6 ms | 50.0 ms | 99.740% | 70.0 GB | 0.0 |
| young MB | 2048.00 | 90.3 ms | 86.9 ms | 99.823% | 30.3 GB | 0.0 |
| tenure age | 3.00 | 43.9 ms | 36.0 ms | 99.631% | 143.7 GB | 0.0 |
| tenure age | 7.00 | 44.9 ms | 37.9 ms | 99.622% | 156.5 GB | 0.0 |
| tenure age | 11.00 | 15.9 ms | 13.0 ms | 99.656% | 214.3 GB | 0.0 |
| tenure age | 15.00 | 64.4 ms | 55.4 ms | 99.600% | 120.1 GB | 0.0 |
| initiating occ. | 0.40 | 55.9 ms | 47.2 ms | 99.557% | 130.4 GB | 0.0 |
| initiating occ. | 0.50 | 15.9 ms | 13.0 ms | 99.622% | 214.3 GB | 0.0 |
| initiating occ. | 0.60 | 45.9 ms | 38.7 ms | 99.673% | 151.4 GB | 0.0 |

Tuning notes:
- young size (MB): 153.6 -> 2048 changes max pause +76.2 ms, throughput +0.424 pp, promotion -295.5 GB, full GCs +0 (mean over grid).
- promotion age: 3 -> 15 changes max pause +20.5 ms, throughput -0.031 pp, promotion -23.6 GB, full GCs +0 (mean over grid).
- initiating occupancy: 0.4 -> 0.6 changes max pause -10.0 ms, throughput +0.116 pp, promotion +21.0 GB, full GCs +0 (mean over grid).

Reverse calibration from the simulated GC log:

| quantity | estimate | true | error % |
|---|---|---|---|
| young size | 2.15 GB | 2.15 GB | +0.00 |
| eden | 1.07 GB | 1.07 GB | +0.00 |
| old size | 6.44 GB | 6.44 GB | +0.00 |
| survivor ratio | 0.250 | 0.250 | +0.00 |
| alloc rate | 13.4 MB/s | 13.3 MB/s | +0.55 |
| promotion rate | 0.2 MB/s | 0.2 MB/s | +3.00 |
| promotion fraction | 1.51% | 1.48% | +2.44 |

## session_heavy / trading_day

Objective: pause <= 50 ms, throughput >= 99.00%

| metric | baseline | tuned |
|---|---|---|
| young_gcs | 1458 | 9720 |
| remark_pauses | 125 | 129 |
| full_gcs | 0 | 0 |
| degraded_cycles | 0 | 0 |
| pause_p50_ms | 107.87 | 4.88 |
| pause_p99_ms | 135.4 | 15.9 |
| pause_max_ms | 136.2 | 24.5 |
| throughput_stw | 99.83213% | 99.93879% |
| throughput_adj | 98.88757% | 99.23608% |
| promotion_gb | 563.0 | 817.6 |
| promotion_fraction | 35.96% | 52.22% |
| pretenured_gb | 563.0 | 817.6 |
| old_peak_committed_gb | 9.27 | 11.14 |

Chosen parameters: young 256 MB, survivor ratio 0.20, tenure age 3, initiating occupancy 0.60; giving max pause 24.5 ms, throughput 99.237%, full GCs 0. 34 of 96 evaluated configurations met both objectives.

Per-parameter response (mean over the grid):

| knob | value | mean max pause | mean p99 | mean throughput | mean promotion | mean full GCs |
|---|---|---|---|---|---|---|
| young MB | 153.60 | 5218.2 ms | 10.4 ms | 99.127% | 919.3 GB | 0.0 |
| young MB | 256.00 | 846.2 ms | 22.2 ms | 98.996% | 832.1 GB | 0.0 |
| young MB | 384.00 | 23.8 ms | 20.1 ms | 99.129% | 801.9 GB | 0.0 |
| young MB | 512.00 | 39.9 ms | 33.5 ms | 98.942% | 765.7 GB | 0.0 |
| young MB | 1024.00 | 65.1 ms | 58.7 ms | 98.887% | 697.7 GB | 0.0 |
| young MB | 2048.00 | 116.7 ms | 112.4 ms | 98.908% | 585.5 GB | 0.0 |
| tenure age | 3.00 | 829.6 ms | 43.3 ms | 98.990% | 764.8 GB | 0.0 |
| tenure age | 7.00 | 821.2 ms | 43.6 ms | 98.990% | 762.5 GB | 0.0 |
| tenure age | 15.00 | 354.2 ms | 57.4 ms | 98.914% | 720.9 GB | 0.0 |
| initiating occ. | 0.40 | 100.2 ms | 57.1 ms | 98.767% | 722.1 GB | 0.0 |
| initiating occ. | 0.50 | 1191.3 ms | 16.7 ms | 99.092% | 845.6 GB | 0.0 |
| initiating occ. | 0.60 | 1042.2 ms | 46.9 ms | 99.094% | 753.0 GB | 0.0 |

Tuning notes:
- young size (MB): 153.6 -> 2048 changes max pause -5101.5 ms, throughput -0.220 pp, promotion -333.8 GB, full GCs +0 (mean over grid).
- promotion age: 3 -> 15 changes max pause -475.4 ms, throughput -0.076 pp, promotion -43.9 GB, full GCs +0 (mean over grid).
- initiating occupancy: 0.4 -> 0.6 changes max pause +941.9 ms, throughput +0.327 pp, promotion +30.9 GB, full GCs +0 (mean over grid).

Reverse calibration from the simulated GC log:

| quantity | estimate | true | error % |
|---|---|---|---|
| young size | 2.15 GB | 2.15 GB | +0.00 |
| eden | 1.07 GB | 1.07 GB | +0.00 |
| old size | 17.18 GB | 17.18 GB | +0.00 |
| survivor ratio | 0.250 | 0.250 | +0.00 |
| alloc rate | 18.2 MB/s | 18.1 MB/s | +0.24 |
| promotion rate | 6.6 MB/s | 6.5 MB/s | +1.18 |
| promotion fraction | 36.29% | 35.96% | +0.93 |

## cache_heavy / batch_heavy

Objective: pause <= 50 ms, throughput >= 98.50%

| metric | baseline | tuned |
|---|---|---|
| young_gcs | 1371 | 7836 |
| remark_pauses | 123 | 140 |
| full_gcs | 0 | 0 |
| degraded_cycles | 0 | 0 |
| pause_p50_ms | 40.21 | 7.45 |
| pause_p99_ms | 116.1 | 11.0 |
| pause_max_ms | 117.1 | 21.0 |
| throughput_stw | 99.89869% | 99.92976% |
| throughput_adj | 99.04845% | 99.30966% |
| promotion_gb | 677.8 | 1045.2 |
| promotion_fraction | 46.03% | 70.98% |
| pretenured_gb | 677.1 | 1045.2 |
| old_peak_committed_gb | 8.52 | 11.41 |

Chosen parameters: young 256 MB, survivor ratio 0.15, tenure age 3, initiating occupancy 0.60; giving max pause 21.0 ms, throughput 99.310%, full GCs 0. 33 of 96 evaluated configurations met both objectives.

Per-parameter response (mean over the grid):

| knob | value | mean max pause | mean p99 | mean throughput | mean promotion | mean full GCs |
|---|---|---|---|---|---|---|
| young MB | 153.60 | 5306.3 ms | 11.1 ms | 99.208% | 1054.6 GB | 0.0 |
| young MB | 256.00 | 2642.7 ms | 18.2 ms | 99.098% | 1058.3 GB | 0.0 |
| young MB | 384.00 | 20.4 ms | 15.1 ms | 99.151% | 1036.9 GB | 0.0 |
| young MB | 512.00 | 1250.0 ms | 25.1 ms | 98.964% | 1025.2 GB | 0.0 |
| young MB | 1024.00 | 447.8 ms | 55.6 ms | 98.901% | 901.4 GB | 0.0 |
| young MB | 2048.00 | 112.0 ms | 109.8 ms | 99.076% | 661.5 GB | 0.0 |
| tenure age | 3.00 | 1753.9 ms | 39.4 ms | 99.063% | 959.1 GB | 0.0 |
| tenure age | 7.00 | 1576.2 ms | 39.5 ms | 99.062% | 956.4 GB | 0.0 |
| tenure age | 15.00 | 1000.7 ms | 52.9 ms | 98.993% | 911.4 GB | 0.0 |
| initiating occ. | 0.40 | 222.7 ms | 53.0 ms | 98.837% | 912.9 GB | 0.0 |
| initiating occ. | 0.50 | 1956.8 ms | 13.5 ms | 99.146% | 1045.8 GB | 0.0 |
| initiating occ. | 0.60 | 2341.7 ms | 42.6 ms | 99.176% | 946.2 GB | 0.0 |

Tuning notes:
- young size (MB): 153.6 -> 2048 changes max pause -5194.3 ms, throughput -0.132 pp, promotion -393.1 GB, full GCs +0 (mean over grid).
- promotion age: 3 -> 15 changes max pause -753.2 ms, throughput -0.070 pp, promotion -47.6 GB, full GCs +0 (mean over grid).
- initiating occupancy: 0.4 -> 0.6 changes max pause +2119.0 ms, throughput +0.339 pp, promotion +33.2 GB, full GCs +0 (mean over grid).

Reverse calibration from the simulated GC log:

| quantity | estimate | true | error % |
|---|---|---|---|
| young size | 2.15 GB | 2.15 GB | +0.00 |
| eden | 1.07 GB | 1.07 GB | +0.00 |
| old size | 17.18 GB | 17.18 GB | +0.00 |
| survivor ratio | 0.250 | 0.250 | +0.00 |
| alloc rate | 17.1 MB/s | 17.0 MB/s | +0.14 |
| promotion rate | 7.9 MB/s | 7.8 MB/s | +0.92 |
| promotion fraction | 46.39% | 46.03% | +0.78 |

## short_lived / flat

Objective: pause <= 50 ms, throughput >= 99.00%

| metric | baseline | tuned |
|---|---|---|
| young_gcs | 1251 | 7153 |
| remark_pauses | 0 | 27 |
| full_gcs | 0 | 0 |
| degraded_cycles | 0 | 0 |
| pause_p50_ms | 16.52 | 6.19 |
| pause_p99_ms | 18.1 | 7.0 |
| pause_max_ms | 18.4 | 7.2 |
| throughput_stw | 99.97608% | 99.94844% |
| throughput_adj | 99.97608% | 99.87198% |
| promotion_gb | 2.6 | 60.0 |
| promotion_fraction | 0.20% | 4.46% |
| pretenured_gb | 0.0 | 0.0 |
| old_peak_committed_gb | 2.63 | 2.59 |

Chosen parameters: young 256 MB, survivor ratio 0.15, tenure age 3, initiating occupancy 0.40; giving max pause 7.2 ms, throughput 99.872%, full GCs 0. 96 of 96 evaluated configurations met both objectives.

Per-parameter response (mean over the grid):

| knob | value | mean max pause | mean p99 | mean throughput | mean promotion | mean full GCs |
|---|---|---|---|---|---|---|
| young MB | 153.60 | 8.7 ms | 5.8 ms | 99.876% | 70.7 GB | 0.0 |
| young MB | 256.00 | 14.0 ms | 9.0 ms | 99.869% | 44.9 GB | 0.0 |
| young MB | 384.00 | 11.4 ms | 11.0 ms | 99.882% | 28.1 GB | 0.0 |
| young MB | 512.00 | 13.6 ms | 12.7 ms | 99.901% | 22.5 GB | 0.0 |
| young MB | 1024.00 | 16.3 ms | 15.3 ms | 99.934% | 10.4 GB | 0.0 |
| young MB | 2048.00 | 17.8 ms | 17.5 ms | 99.964% | 4.6 GB | 0.0 |
| tenure age | 3.00 | 11.1 ms | 9.1 ms | 99.900% | 45.3 GB | 0.0 |
| tenure age | 7.00 | 14.7 ms | 12.7 ms | 99.908% | 22.5 GB | 0.0 |
| tenure age | 15.00 | 18.9 ms | 16.6 ms | 99.915% | 8.0 GB | 0.0 |
| initiating occ. | 0.30 | 9.9 ms | 8.5 ms | 99.866% | 48.4 GB | 0.0 |
| initiating occ. | 0.40 | 14.4 ms | 12.3 ms | 99.905% | 27.4 GB | 0.0 |
| initiating occ. | 0.60 | 15.9 ms | 13.6 ms | 99.923% | 20.4 GB | 0.0 |

Tuning notes:
- young size (MB): 153.6 -> 2048 changes max pause +9.2 ms, throughput +0.088 pp, promotion -66.1 GB, full GCs +0 (mean over grid).
- promotion age: 3 -> 15 changes max pause +7.8 ms, throughput +0.015 pp, promotion -37.2 GB, full GCs +0 (mean over grid).
- initiating occupancy: 0.3 -> 0.6 changes max pause +6.0 ms, throughput +0.057 pp, promotion -27.9 GB, full GCs +0 (mean over grid).
- The default configuration already meets the objective; the search is only confirming the margin.

Reverse calibration from the simulated GC log:

| quantity | estimate | true | error % |
|---|---|---|---|
| young size | 2.15 GB | 2.15 GB | +0.00 |
| eden | 1.07 GB | 1.07 GB | +0.00 |
| old size | 6.44 GB | 6.44 GB | +0.00 |
| survivor ratio | 0.250 | 0.250 | +0.00 |
| alloc rate | 15.6 MB/s | 15.6 MB/s | +0.00 |
| promotion rate | 0.0 MB/s | 0.0 MB/s | +2.53 |
| promotion fraction | 0.20% | 0.20% | +2.53 |

## cache_heavy#2 / trading_day

Objective: pause <= 50 ms, throughput >= 99.50%

| metric | baseline | tuned |
|---|---|---|
| young_gcs | 2142 | - |
| remark_pauses | 1629 | - |
| full_gcs | 60 | - |
| degraded_cycles | 60 | - |
| pause_p50_ms | 34.69 | - |
| pause_p99_ms | 1009.9 | - |
| pause_max_ms | 1315.8 | - |
| throughput_stw | 99.79546% | - |
| throughput_adj | 90.37236% | - |
| promotion_gb | 1419.8 | - |
| promotion_fraction | 61.72% | - |
| pretenured_gb | 1418.4 | - |
| old_peak_committed_gb | 3.86 | - |

Tuning notes:
- No configuration in the searched space met both objectives. The closest point is reported; the binding constraint is pause.
- young size (MB): 256 -> 2048 changes max pause +830.6 ms, throughput +1.032 pp, promotion -359.2 GB, full GCs +43 (mean over grid).
- promotion age: 3 -> 15 changes max pause -51.3 ms, throughput -0.047 pp, promotion +2.7 GB, full GCs -1 (mean over grid).
- initiating occupancy: 0.4 -> 0.6 changes max pause -0.5 ms, throughput +2.419 pp, promotion -0.0 GB, full GCs +1 (mean over grid).
- Default settings suffered 60 full GC(s) and 60 degraded concurrent cycle(s) over the day.
- No combination of young size, tenure age and initiating occupancy met both objectives at this heap size. The binding constraint is the heap budget, not the three knobs -- see the sensitivity table for how far each knob moves the outcome.

Reverse calibration from the simulated GC log:

| quantity | estimate | true | error % |
|---|---|---|---|
| young size | 2.15 GB | 2.15 GB | +0.00 |
| eden | 1.07 GB | 1.07 GB | +0.00 |
| old size | 3.22 GB | 3.22 GB | +0.00 |
| survivor ratio | 0.250 | 0.250 | +0.00 |
| alloc rate | 26.7 MB/s | 26.6 MB/s | +0.20 |
| promotion rate | 16.7 MB/s | 16.4 MB/s | +1.46 |
| promotion fraction | 62.50% | 61.72% | +1.26 |
