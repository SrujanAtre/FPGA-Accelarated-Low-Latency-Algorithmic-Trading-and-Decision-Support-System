from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parent
def read(name):return (ROOT/name).read_text(errors='replace')
a=json.loads(read('reports/model_audit.json'))
s=json.loads(read('reports/performance_summary.json'))
c=json.loads(read('reports/local_cpu_reference.json'))
h=json.loads(read('reports/historical_preprocessing.json'))
powerfile='reports/power_uart_activity.rpt' if (ROOT/'reports/power_uart_activity.rpt').exists() else 'reports/power.rpt'
power=read(powerfile)
def watts(label):return float(re.search(r'\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)',power).group(1))
total,dynamic,static=[watts(n) for n in ['Total On-Chip Power (W)','Dynamic (W)','Device Static (W)']]
uart=s['uart_simulation'];uart_rate=1e9/(sum(r['total_ns'] for r in uart)/len(uart))
program=read('reports/program.log') if (ROOT/'reports/program.log').exists() else ''
physical=(ROOT/'reports/hardware_results.csv').exists()
workrows='\n'.join(f"| {w['offered_per_second']:,.0f} | {w['achieved_per_second']:,.2f} | {w['min_latency_cycles']}–{w['max_latency_cycles']} | {w['max_external_queue_delay_us']:.3f} |" for w in s['workload_simulation'])
bit=ROOT/'vivado/fpga_ml.runs/impl_1/fpga_ml_top.bit'
files='\n'.join('- '+str(p.relative_to(ROOT)) for p in sorted(ROOT.iterdir()) if p.is_file())
report=f'''# FPGA ML deployment and performance evaluation

Completed on 7 October 2026. Target: AMD ZCU104, xczu7ev-ffvc1156-2-e. Vivado 2024.2 was actually used for synthesis, placement, routing, timing analysis, power estimation, and generation of the ML bitstream. The original baseline project remains separate.

## Completion and evidence

| Stage | Result | Evidence type |
|---|---|---|
| Saved CNN/LSTM ONNX audit | Graphs checked; model inputs and operators recorded | Tool inspection |
| Saved LightGBM conversion | 72 trees, 4,084 learned nodes, maximum depth 6 | Actual model export |
| Historical feature reproduction | 4,329 raw bars, 4,277 feature rows; labels identical | Measured host execution |
| Fixed-point/model representation | 643/643 predictions agree | Software validation |
| Accelerator RTL validation | 643/643 predictions and all three integer scores agree | Vivado XSIM simulation |
| UART integration | BUY, HOLD and SELL response framing passed | Vivado XSIM simulation |
| Workload validation | 1,929 predictions passed across three rates | Vivado XSIM simulation |
| Synthesis and implementation | Completed successfully | Actual Vivado run |
| Bitstream | {bit.stat().st_size:,} bytes | Actual generated artifact |
| Physical programming | {'Programming command succeeded' if 'ML_BITSTREAM_PROGRAMMED' in program else 'Not successfully completed; see program.log'} | Actual Hardware Manager attempt |
| Board inference measurements | {'Available in hardware_results.csv' if physical else 'Unavailable: board PL TAP/power access blocked'} | Explicitly unavailable unless measured |

Bitstream SHA-256: `{hashlib.sha256(bit.read_bytes()).hexdigest()}`.
Model SHA-256: `{a['model_sha256']}`.

## Model selection and actual computation

The existing trained LightGBM was selected after inspecting all three models. Its input is one bar of 27 features, whereas CNN/LSTM consume 30 × 27 windows. The LSTM has three recurrent layers, hidden dimension 64, LayerNormalization and two dense layers. Its ONNX graph contains three LSTM operators and LayerNormalization. The CNN has causal convolutions 27→64→128→64 with kernel 3, ReLU, temporal average pooling, and 64→32→3 dense layers. Both exported graphs passed ONNX checking; their audited shape/operator tables are in model_audit.json. Dynamic-shape declarations alone do not prove arbitrary-batch deployability.

The tree ensemble fits roughly 32 KB of node ROM, needs signed comparisons and addition, and can be implemented directly with synchronous BRAM in Vivado. This avoids introducing a PS software stack or DPU for this model. This is a custom trained-model RTL deployment, not an ONNX neural-network conversion and not a rule-based wrapper. Every branch threshold, feature index, and leaf score originates from lgbm.pkl. All 72 trees execute. The final argmax is 0=BUY, 1=HOLD, 2=SELL. Softmax is unnecessary because it preserves argmax. No confidence gate or EMA/ATR/crossover decision is substituted.

## Exact implemented architecture

Historical OHLCV → existing host Python feature engineering → saved RobustScaler → float32 → lossless ordered feature keys → PL UART RX → 27 feature registers → BRAM trained-tree ROM and sequential traversal → three 64-bit score accumulators → argmax → PL UART TX.

Clock: F23/E23 differential 125 MHz input through IBUFDS and BUFG. Reset: initialized six-bit startup counter. UART RX: A20; UART TX: C19, LVCMOS18, 115200 8N1. Existing validated UART RX/TX were copied. The old project was not overwritten.

ML memory: {s['rom_bytes']:,} bytes of logical 64-bit node words plus 72 roots. A node stores leaf flag, learned feature index, child pointers, and either learned threshold or leaf score. Vivado optimizes unused ROM bits. Feature storage is 27×32 bits; score accumulation is 3×64 bits; response buffer is 240 bits. Memory usage and hardware resources remain fixed with workload. There is no DDR, PS, AXI, DMA, interrupt, or artificial Block Design; this small model needs none. The actual RTL hierarchy is fpga_ml_top with core, RX, TX and clock buffers, visible in Vivado RTL/synthesized schematics.

Numerics: finite float32 feature values are encoded losslessly into ordered signed 32-bit keys. Learned thresholds are converted to the largest float32 no greater than their original double value. This preserves <= semantics. NaN/infinity are rejected. Zero is canonicalized. Leaf scores have 24 fractional bits in signed 32-bit words; scores sum in signed 64-bit accumulators. Ties choose the lowest index.

Request: A5 followed by 108 bytes of little-endian feature keys. Response: 5A, class, 32-bit little-endian inference cycles, three 64-bit little-endian scores. The host uses one outstanding request at a time. The protocol has no checksum, truncated-packet timeout or hardware input queue. Extra bytes sent while compute/transmit is busy are discarded; overload requires host pacing. The new protocol/class mapping differs from the baseline.

## Preprocessing and dataset

The file called nifty_1m.csv actually contains historical five-minute bars (median interval {h['median_bar_interval_seconds']:.0f} seconds). No live feed was introduced. Existing feature functions regenerated the cached values to CSV rounding precision; maximum absolute difference is {max(h['max_absolute_difference_by_feature'].values()):.3g}, and labels are identical. The saved scaler is reused, never refitted.

Feature generation is host preprocessing, not PL logic. The model inference itself runs in PL. A measured batch feature-generation run took {h['feature_generation_batch_ns']/1e6:.3f} ms for 4,329 bars. The amortized {h['amortized_ns_per_raw_bar']/1000:.3f} µs/bar is a calculated batch average, not a measured incremental streaming latency. Hardware end-to-end testing additionally times per-bar scaling/encoding. Historical CSV loading and complete batch feature engineering are separate setup stages.

Feature order: {', '.join(a['features'])}.

## Functional correctness

The time-based held-out split has {a['samples']} bars from {a['data_first']} to {a['data_last']}. Original software predictions and RTL predictions are recorded for every sample. Matching: 643; mismatches: 0; agreement: 100%. RTL class scores also match the exported integer reference exactly.

Classification accuracy versus labels: {a['accuracy']*100:.6f}%, identical to the original model. Prediction agreement is not classification accuracy. The model's predictive quality is modest; FPGA conversion does not improve it. An initial Q16.16 feature experiment mismatched one sample; lossless float32 ordering fixed that threshold precision failure. The final feature representation adds zero error relative to float32 inputs. Maximum raw-score error against the original model: {a['max_raw_score_error']:.12g}. Leaf rounding uses 24 fractional bits. These are held-out validation results, not a universal proof of future prediction agreement.

Confusion matrix, rows=true / columns=predicted, order BUY/HOLD/SELL:

| True class | BUY | HOLD | SELL |
|---|---:|---:|---:|
| BUY | {a['confusion_matrix'][0][0]} | {a['confusion_matrix'][0][1]} | {a['confusion_matrix'][0][2]} |
| HOLD | {a['confusion_matrix'][1][0]} | {a['confusion_matrix'][1][1]} | {a['confusion_matrix'][1][2]} |
| SELL | {a['confusion_matrix'][2][0]} | {a['confusion_matrix'][2][1]} | {a['confusion_matrix'][2][2]} |

## Latency and throughput

All values in this section are RTL simulations or calculations from those simulations until physical results exist. One 125 MHz cycle = 8 ns.

| Metric | Cycles | ns | µs |
|---|---:|---:|---:|
| ML inference minimum | 783 | 6,264 | 6.264 |
| ML inference mean | {s['inference']['mean_cycles']:.6f} | {s['inference']['mean_ns']:.6f} | {s['inference']['mean_ns']/1000:.6f} |
| ML inference maximum in held-out set | 993 | 7,944 | 7.944 |
| Architectural bound (72 trees × at most 7 visited nodes × 2 cycles + argmax) | 1,009 | 8,072 | 8.072 |

The engine traverses one node in two cycles (synchronous ROM read and evaluate). Trees execute sequentially, with no overlap between inferences. Core latency varies with learned path depth; it is bounded, not constant. It must not be described as a fully pipelined one-inference-per-cycle accelerator.

The UART top adds two cycles from final payload receipt to the observed done event. Serial simulation observed 917, 951 and 937 processing cycles (7.336, 7.608 and 7.496 µs) for BUY/HOLD/SELL examples. Input transfer: approximately 9.456880 ms; output transfer and framing: 2.604712 ms. Total serial transaction: 12.068932–12.069200 ms, corresponding to roughly 1,508,617–1,508,650 clock periods. These transaction boundaries include serial bits and top-level handshaking, but exclude host scheduling, USB buffering and host preprocessing.

Calculated serial capacity from these simulations: {uart_rate:.3f} inferences/s. This is not a measured physical sustained rate. The ML core's simulated saturated replay rate is 134,896.77 inferences/s. UART limits the demonstrator far earlier than ML compute does. Serial packets include raw scores for validation, which increases transfer time.

## Workload scalability

Each offered rate uses 643 inputs; every prediction passes. Pending overload requests are held by the simulation testbench outside the accelerator, not by an invented hardware FIFO.

| Offered inferences/s | Achieved finite-window inferences/s | Core latency cycles | Maximum external waiting µs |
|---:|---:|---:|---:|
{workrows}

Finite-window rates can slightly exceed the offered nominal rate because the measurement excludes a trailing interarrival gap. Below capacity no waiting accumulates. At 250,000 requests/s the core saturates and external waiting grows to 2.191088 ms. Computation latency still spans 783–993 cycles. Memory and FPGA resource usage are static. No physical workload power/stability measurement has been obtained. UART overload must be handled by stop-and-wait, not by these core-only testbench arrival rates.

## Timing, utilization and Fmax

Actual routed Vivado timing: WNS +2.969 ns, TNS 0, setup violations 0; WHS +0.005 ns, hold violations 0; pulse width slack +3.458 ns. All specified timing constraints pass at 125 MHz. No internal endpoints are unconstrained. The reused XDC has one asynchronous UART input without an input delay and one UART output without an output delay; external UART timing is verified by serial simulation, and these ports are not a synchronous external bus. Do not claim external synchronous I/O closure from this report.

Implemented frequency: 125 MHz. Calculated timing-based estimate 1/(8−2.969 ns) ≈ 198.77 MHz is not a tested Fmax or a validated alternative operating frequency; no higher-frequency build or physical frequency sweep was performed.

| Resource | Actual routed used | Device available | Utilization |
|---|---:|---:|---:|
| LUT | 1,754 | 230,400 | 0.76% |
| FF | 2,324 | 460,800 | 0.50% |
| BRAM 36Kb tile equivalents | 7.5 (7 RAMB36 + 1 RAMB18) | 312 | 2.40% |
| DSP | 0 | 1,728 | 0% |
| URAM | 0 | 96 | 0% |
| Bonded I/O | 4 | 360 | 1.11% |
| BUFGCE | 1 | 208 | 0.48% |

BRAM has the highest percentage utilization due to trained model storage. Feature selection muxes, signed comparison, score addition and UART control consume LUTs/FFs. The tree ensemble requires no multipliers, so DSP usage is zero. See hierarchical utilization for accelerator/interface attribution.

## Power and efficiency

Hierarchical resources: core uses 823 LUTs, 1,106 FFs, all seven RAMB36 and one RAMB18, and no DSP/URAM. UART RX uses 807 LUTs/56 FFs; UART TX uses 30 LUTs/26 FFs; top-level packet/feature/response handling uses 96 LUTs/1,136 FFs.

The SAIF power analysis matched 1,119/4,892 nets (23%); optimized unmatched nets use propagation. Vivado reports Medium confidence. Estimated accelerator core dynamic power is 0.009 W by hierarchy.

Source: {powerfile}. Total on-chip power {total:.3f} W, dynamic {dynamic:.3f} W, static {static:.3f} W. This is a routed Vivado estimate, not measured board power. UART activity SAIF represents three request/response transactions at 125 MHz; unmatched optimized nets use tool activity propagation. The estimate excludes host PC, USB adapter, supply losses and board peripherals. It does not measure a saturated core workload.

For this UART activity scenario only, estimated energy per transaction = estimated total power × simulated transaction duration ≈ {total/uart_rate*1e6:.3f} µJ/inference. Corresponding calculated efficiency ≈ {uart_rate/total:.3f} inferences/s/W. These mixed estimated/simulated metrics are not hardware energy measurements. Dividing UART-workload power by core-only saturated throughput would mix incompatible workloads and is deliberately omitted. Accelerator-related dynamic power is listed by hierarchy in the power report; static device power cannot be uniquely assigned to the accelerator.

AMD documents Vivado's vectorless and SAIF estimation behavior: https://docs.amd.com/r/en-US/ug907-vivado-power-analysis-optimization/Estimating-Power-Vivado-Design-Flow-Stage . Functional SAIF does not capture routing glitches; accurate rail measurements remain a physical evaluation step.

## CPU comparison

The existing low_latency/latency_log.csv and fpga/fpga_latency.csv measure EMA/ATR/crossover baseline behavior, including an old FPGA cycle counter that captures before model inference. They are not ML deployment results and are excluded from ML speedup claims. Teammates' optimized CPU ML latency/throughput/power are unavailable in the local project/history; no values are invented.

A separately labeled local PC measurement ran the actual saved LightGBM with batch 1 and one inference thread, 20 warm-ups and 1,929 timed calls. Host: {c['processor']}; platform {c['platform']}. Mean {c['latency_ns']['mean']/1000:.3f} µs, median {c['latency_ns']['50%']/1000:.3f} µs, p95 {c['latency_ns']['95%']/1000:.3f} µs. Prediction API and argmax overhead are included; preprocessing is excluded. This is not the teammates' optimized CPU implementation.

| Comparison metric | Local measured CPU reference | FPGA |
|---|---|---|
| Mean model inference latency | {c['latency_ns']['mean']/1000:.3f} µs | 7.397 µs simulated |
| Sustained model rate | {c['sustained_inferences_per_second_excluding_loop_overhead']:.2f}/s, excluding loop overhead | 134,896.77/s simulated saturated core replay |
| UART end-to-end transfer/compute | Not applicable | ≈12.069 ms simulated, host overhead excluded |
| Same-model class agreement | Reference | 100% on 643 held-out inputs in RTL |
| Classification accuracy | 48.833593% | 48.833593% in RTL |
| Hardware resources | PC CPU; no equivalent LUT metric | Actual routed utilization above |
| Power/energy | Unavailable | Estimated workload-specific values above |
| Teammates' optimized pipeline | Unavailable | Awaiting comparable actual CPU data |

This supports a compute-only comparison but not a measured physical speedup claim. The current UART demonstrator has much larger end-to-end latency than local model inference.

## Files and jury demonstration

All created files are under fpga_ml. Existing trained models, scaler, historical data, Python master pipeline and original Vivado baseline project were preserved. New RTL, ROMs, constraints, testbenches, scripts and reports implement the ML design; modifications during this task were limited to these new files while debugging.

{files}

Key demonstration artifacts: vivado/fpga_ml.xpr, vivado/fpga_ml.runs/impl_1/fpga_ml_top.bit, reports/ml_routed.dcp, reports/timing.rpt, reports/utilization.rpt, reports/power_uart_activity.rpt, rtl/rtl_results.csv, rtl/uart_sim_results.csv, rtl/workload_sim_results.csv. The Vivado project shows the actual target, RTL hierarchy, synthesized/routed implementation and successful runs. README.md contains the implemented architecture and exact protocol. Reports and bitstream are evidence of real hardware implementation, while simulations are labeled as simulations.

## Remaining physical work and objective status

{'Physical results exist; see hardware_results.csv for exact predictions and counter/timing measurements.' if physical else 'Vivado detects the ZCU104 JTAG adapter and xczu7 device, but repeated probes report PL Power Status OFF / cannot connect PL TAP. The user powered on the board; the state persisted. A physical POR_B reset was requested. Successful programming and COM7 inference therefore remain blocked by board access, not by missing accelerator/build files.'}

Once PL access is available, hardware.tcl programs this exact bitstream and host_test.py sends the 643 held-out inputs and validates exact class scores while recording actual cycle counts, per-bar scaling/encoding, UART round-trip and host end-to-end times. These scripts were created; their physical results must not be claimed unless they actually complete. Actual rail power requires additional board measurement access.

The trained-model PL implementation, Vivado build, bitstream, reproducible functional simulation, timing/resource analysis and workload evaluation satisfy the implementation portion of the capstone FPGA objective. Full physical deployment/performance demonstration remains incomplete until programming and board inference succeed. There are no fabricated FPGA predictions, CPU measurements, or power measurements.
'''
if physical:
 hw=s['physical'];st=hw['host_stages_ns']
 pipeline=s.get('physical_pipeline',{})
 physical_workrows='\n'.join(f"| {w['workload']} | {w['samples']} | {w['sustained_per_second']:.3f} | {w['roundtrip_ns']['mean']/1e6:.3f} | {w['min_cycles']}–{w['max_cycles']} |" for w in s['physical_workloads'])
 physical_section=f'''## Actual physical ZCU104 measurements

Vivado successfully programmed the generated ML bitstream. An initial PL power-status warning persisted during probes, but programming succeeded and subsequent refresh reported a programmed device. COM7 then returned correct predictions and exact learned class scores. No external reset was required to complete programming.

Primary physical validation: 643/643 predictions match the saved LightGBM, with zero mismatches; all three score accumulators match the integer reference for every input. The on-board cycle counter matches RTL for every sample. Therefore 783–993 cycles (6.264–7.944 µs), mean 924.636 cycles (7.397089 µs), are measured hardware cycle results with nanoseconds calculated from the 125 MHz clock. Label accuracy is 48.833593% on the physical board, identical to software.

The optimized host path uses saved scaler.center_ and scaler.scale_ directly with the same float64 subtraction/division then float32 cast. Encoded features are asserted identical to the original scaler.transform inputs for all 643 bars. The original slower measurement remains in hardware_results_initial.csv.

Measured final host stages, from an existing unscaled 27-feature historical bar through receipt of the complete FPGA response:

| Stage | Mean | Median | p95 | Interpretation |
|---|---:|---:|---:|---|
| Host scaling/encoding/packing and validation | {st['preprocessing_ns']['mean']/1000:.3f} µs | {st['preprocessing_ns']['50%']/1000:.3f} µs | {st['preprocessing_ns']['95%']/1000:.3f} µs | Measured host preprocessing |
| Serial write + flush API | {st['input_write_ns']['mean']/1000:.3f} µs | {st['input_write_ns']['50%']/1000:.3f} µs | {st['input_write_ns']['95%']/1000:.3f} µs | Measured host blocking; not isolated physical wire time |
| UART/USB complete round-trip | {st['uart_roundtrip_ns']['mean']/1e6:.6f} ms | {st['uart_roundtrip_ns']['50%']/1e6:.6f} ms | {st['uart_roundtrip_ns']['95%']/1e6:.6f} ms | Measured communication + FPGA compute |
| Total per-bar end-to-end | {st['end_to_end_ns']['mean']/1e6:.6f} ms | {st['end_to_end_ns']['50%']/1e6:.6f} ms | {st['end_to_end_ns']['95%']/1e6:.6f} ms | Measured preprocessing + communication + inference |

Final measured complete-loop wall-clock sustained rate: {pipeline.get('actual_wall_clock_sustained_per_second',0):.3f} inferences/s, including host preprocessing, communication, all 643 inferences, loop overhead and progress logging. Reciprocal summed per-bar latency: {hw['sustained_per_second_excluding_between_iteration_overhead']:.3f} inferences/s. Per-bar total ranges from {st['end_to_end_ns']['min']/1e6:.6f} to {st['end_to_end_ns']['max']/1e6:.6f} ms. Do not describe UART/USB/Windows timing as deterministic merely because FPGA compute is bounded.

Separating physical input-wire and output-wire latency requires additional hardware timestamps or an external analyzer; only the FPGA inference counter and host round-trip are independently measured. Serial simulation supplies the approximately 9.457 ms input and 2.605 ms output wire breakdown. FPGA processing from the final received payload byte is core cycles + 2, calculated from measured core counts and validated top-level RTL timing (6.280–7.960 µs), not an additional independent board counter.

Physical workload replay, with exact prediction and score agreement throughout:

| Workload | Samples | Actual wall-clock rate /s | Mean round-trip ms | Core cycles |
|---|---:|---:|---:|---:|
{physical_workrows}

These workloads were run before the final optimized 643-bar replay and under variable PC load. The observed 43.432/s continuous workload and the final {pipeline.get('actual_wall_clock_sustained_per_second',0):.3f}/s optimized complete-loop rate are separate measured runs. The earlier optimized run achieved 61.962/s by reciprocal summed per-bar latency and remains in hardware_results_optimized_initial.csv. Different PC load and boundaries explain differences; these runs are not identical benchmarks. Three full 643-sample runs, 336 workload samples and the three jury-demo transactions give 2,268 successful physical inference transactions in recorded runs, with 643 unique held-out bars.

Memory usage is fixed, no FIFO backlog is stored on the FPGA, and stop-and-wait kept all tested physical runs stable. Clock counts for each input agree with simulation irrespective of replay pacing. Host communication jitter, rather than FPGA computation, limits end-to-end predictability. Physical power per workload remains unavailable; power below is estimated.

'''
 report=report.replace('## Timing, utilization and Fmax',physical_section+'## Timing, utilization and Fmax')
 report=report.replace('All values in this section are RTL simulations or calculations from those simulations until physical results exist.','Core cycle counts in this section were verified on all 643 physical FPGA inferences and match RTL simulation. Core times are calculated from those measured counts. UART serial-wire and saturated core rates remain simulation-derived.')
 report=report.replace('100% on 643 held-out inputs in RTL','100% on 643 held-out physical FPGA inputs')
 report=report.replace('| 7.397 µs simulated |','| 7.397 µs measured FPGA cycles converted at 125 MHz |')
 report=report.replace('This supports a compute-only comparison but not a measured physical speedup claim.','The mean CPU prediction API time divided by measured FPGA core time is a compute-boundary ratio, not an end-to-end system speedup; the CPU and FPGA boundaries include different host/API overhead.')
 report=report.replace('Full physical deployment/performance demonstration remains incomplete until programming and board inference succeed.','Physical programming, learned-model inference, correctness and workload replay have now succeeded. Remaining evaluation gaps are direct rail-power measurement, a tested higher-frequency Fmax sweep, and teammates’ comparable optimized CPU measurements.')
 report=report.replace('Once PL access is available, hardware.tcl programs this exact bitstream and host_test.py sends the 643 held-out inputs and validates exact class scores while recording actual cycle counts, per-bar scaling/encoding, UART round-trip and host end-to-end times. These scripts were created; their physical results must not be claimed unless they actually complete. Actual rail power requires additional board measurement access.','hardware.tcl successfully programmed this exact bitstream. host_test.py completed three 643-bar physical runs, host_workloads.py completed another 336 physical transactions, and demo.py produced actual BUY/HOLD/SELL outputs. CSV files retain predictions, exact scores, hardware cycles and host timings. Actual rail power requires additional board measurement access.')
(ROOT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
print('Wrote',ROOT/'FINAL_REPORT.md')
