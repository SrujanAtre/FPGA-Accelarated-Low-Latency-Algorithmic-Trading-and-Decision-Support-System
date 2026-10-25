# Measured CPU scalability results

Test date: 21 October 2026. Hardware: Intel Core i3-1115G4, 2 physical cores, 4 logical processors. Python 3.12.10; LightGBM 4.7.0.

Run from low_latency: `python cpu_pipeline_ml.py --scalability`

## Offered-load table for the research paper

| Offered /s | Achieved /s | Max. external wait |
|---:|---:|---:|
| 3,134.238 | 3,096.590 | 17,431.319 us |
| 6,268.476 | 4,550.660 | 53,369.891 us |
| 12,536.952 | 4,674.018 | 98,831.281 us |

Measured scheduled-load replay on the CPU, not RTL simulation. Each rate was tested five times using all 643 held-out inputs, batch one and one inference thread. Achieved rate is the median across five runs; maximum wait is the worst observed across those runs. Offered rates are 0.5, 1 and 2 times the median direct single-thread replay throughput.

The harness assigns arrival deadlines and waits when early. External wait is the actual service start minus scheduled arrival, so it includes Windows scheduling lateness and accumulated backlog. It is not a pure queue-residence measurement or a hardware FIFO test. Nonzero waiting even at the lowest rate is possible.

## Thread and worker sweep

| Configuration | Median throughput /s | Mean service us | p95 service us | Speedup vs same-mode level 1 |
|---|---:|---:|---:|---:|
| inference_threads = 1 | 6,268.476 | 134.832 | 235.300 | 1.000x |
| inference_threads = 2 | 347.217 | 3,089.220 | 9,101.180 | 0.055x |
| inference_threads = 4 | 594.008 | 1,612.413 | 5,492.760 | 0.095x |
| process_workers = 1 | 2,328.566 | 140.869 | 245.760 | 1.000x |
| process_workers = 2 | 2,404.357 | 167.742 | 256.300 | 1.033x |
| process_workers = 4 | 2,258.305 | 224.108 | 305.260 | 0.970x |

Each sweep configuration has five runs of 1,929 calls (three replays of the 643 inputs), totaling 9,645 calls. All 67,515 timed predictions across the sweep and offered-load tests matched the reference. Sweep raw scores also matched within 1e-9. This is implementation consistency, not increased market-label accuracy.

Service latency includes saved scaling, batch-one prediction API and argmax. It excludes process submission/queueing/IPC. Throughput includes result collection and correctness bookkeeping; process-worker mode additionally includes IPC. Worker startup, model/scaler loading and warm-up are excluded. Process workers each use one inference thread and an independent model.

Findings: the direct one-thread configuration was fastest among these tested arrangements. More inference threads reduced throughput. Two process workers yielded only about 1.033x relative to one process worker; four workers did not improve it. These figures do not demonstrate useful parallel scaling for this small per-sample workload. No power was measured.

Replace the CPU scalability entry "Parallel CPU sweep not tested" with: "Measured 1/2/4-thread and worker sweeps; scheduled-load replay completed; direct single-thread path performed best among tested configurations."

The FPGA table remains RTL simulation. The CPU table is measured Python/Windows scheduled replay; their clocks, interfaces and timing boundaries differ. Do not present their ratio as matched system speedup.

## Raw evidence

- cpu_scalability_results.json: metadata, hashes, run-level results and limitations
- cpu_scalability_runs.csv and cpu_scalability_summary.csv: sweep measurements
- cpu_scalability_offered_load.csv and cpu_scalability_offered_summary.csv: offered-load measurements

Original model, scaler, historical/live CSVs and default live behavior were preserved. The research paper itself has not been edited.