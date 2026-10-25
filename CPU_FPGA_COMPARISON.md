# CPU and FPGA comparison from saved evidence

These are descriptive results from separate runs. Percentages are arithmetic differences between recorded values, not a matched controlled CPU-versus-FPGA experiment. Historical rows are not ten-minute live-run results.

| Metric | CPU | FPGA | Recorded advantage |
|---|---|---|---|
| Model-stage mean latency | 119.970 us, prediction API + argmax | 7.397 us, core only | FPGA stage time 93.83% lower; boundaries differ |
| Host prediction-path mean latency | 131.157 us, scaling + prediction | 15636.521 us, preprocessing + UART + core | CPU path time 99.16% lower in saved runs |
| Observed historical replay throughput | 4,936.321/s | 61.831/s | CPU 7,883.58% higher in saved runs |
| Agreement with saved-model decisions | 100% in CPU consistency checks | 100% on 643 physical test inputs | Tie; 0 percentage points |
| Historical market-label accuracy of deployed model | 48.83% on original test set | 48.83% on same test set | Tie; 0 percentage points |
| Power | 19.114 W, RAPL package telemetry | 0.606 W, Vivado on-chip UART-workload estimate | No valid winner percentage: methods, scopes and workloads differ |
| Energy per inference / decision | 1.194 mJ, repeated historical workload | 7.314 mJ, estimated UART scenario | No valid winner percentage |
| Energy efficiency | 837.870 decisions/J | 136.727 inferences/J, estimated UART scenario | No valid winner percentage |
| FPGA resource utilization | LUT/FF/BRAM metrics not applicable | LUT 0.76%; FF 0.50%; BRAM 2.40%; DSP 0% | Device allocation percentages are not cross-platform efficiency scores |
| Scalability | Measured thread/worker and scheduled-load replay; direct single-thread path best in tested arrangements | Core offered-load RTL simulation, saturation about 134,897/s | No valid cross-platform winner percentage: different test methods and offered loads |

## Percentage definitions

Latency reduction = (comparison latency - lower latency) / comparison latency x 100%. Throughput increase = (higher throughput - comparison throughput) / comparison throughput x 100%. Accuracy differences are percentage points.

CPU latency/throughput rows use the retained 30-sample historical run. FPGA rows use the retained 643-sample physical run. Full-history feature construction and initial loading are excluded. CPU core/API and FPGA core timing boundaries differ; host FPGA path includes UART. New CPU power-workload throughput must not be substituted for the older CPU throughput without changing the table scope.

## Live data availability

FPGA live_fpga.csv: 17 records on October 15, 13:13:38 to 13:22:00 IST; mean core latency 7.604 us, mean UART round trip 18.108 ms. The records have no run identifier or saved launch/end duration. They cannot be certified as one complete ten-minute run.

CPU live_cpu.csv currently has 73 appended records. October 15 has 52 records across several periods. October 22 has 21 records from 10:31:24 to 10:41:00 IST, mean scaling-plus-prediction latency 416.490 us. CPU and FPGA streams are not paired or simultaneous. No verified paired ten-minute live efficiency or throughput comparison is available.

## New CPU energy evidence

The dedicated historical power replay completed 79,914 checked predictions over 4.990 meter seconds, consuming 95.378 J of CPU-package energy. All decisions matched the saved reference. This is RAPL hardware-reported telemetry including other package activity, not whole-PC or process-only energy. It is not a live-run measurement.

## Reproduce power output

From low_latency, use `python cpu_pipeline_ml.py` for default live mode. Its final summary automatically reads package energy over the live event loop, including feed waits and feature/logging work. Results save to live_cpu_power.json.

Use `python cpu_pipeline_ml.py --historical` for the normal 30-row historical output followed by a separate five-second energy replay. The energy replay prints no extra sample rows and saves historical_cpu_power.json. Disable that extra phase with --power-seconds 0.

Use `python cpu_pipeline_ml.py --power-benchmark --seconds 10` to measure energy on repeated held-out inference without overwriting historical/live latency CSVs. Results save to cpu_power_benchmark.json.

If Windows denies access or the meter is missing/stalled/reset, the pipeline reports the exact unavailable reason. No driver installation or system settings changes are performed. The row table and existing predictions remain unchanged.