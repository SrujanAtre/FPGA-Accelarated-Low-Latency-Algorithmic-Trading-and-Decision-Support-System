# Provisional live decisions every thirty seconds

The existing CPU and FPGA entry points now load recent genuine Yahoo five-minute OHLCV history to initialize the existing 27 indicators and saved scaler. A Yahoo WebSocket supplies fresh NIFTY prices. The program updates the current unfinished five-minute candle and emits a provisional decision for each new wall-clock thirty-second slot when fresh prices are available. The first decision can appear as soon as history loads and a usable live update arrives. There is no wait to collect fifty new candles.

From the main project:

```powershell
python .\low_latency\cpu_pipeline_ml.py
python .\fpga_ml\live_fpga.py
```

From each respective folder, use `python cpu_pipeline_ml.py` or `python live_fpga.py`. Both default to a ten-minute window measured from launch, including history loading and connection setup. A launch at 13:00 ends at approximately 13:10, and predictions completed after the deadline are not printed or appended. A network operation can delay process exit, but does not extend the prediction window. Use `--max-runtime-seconds 0` for unlimited time or `--max-runtime-seconds 300` for five minutes. `--max-samples N` optionally stops earlier after N predictions. Summary statistics are printed when the run ends. Only the FPGA needs the connected, powered and ML-programmed board on COM7 (`--port COMx` selects another port). No CPU fallback fabricates FPGA responses.

## Interpretation

These are **provisional live decisions on an unfinished five-minute candle**, not completed thirty-second-candle decisions. The original model/scaler, five-minute indicator periods and FPGA bitstream are retained. Within a five-minute bucket each update replaces that unfinished row; it does not append a new feature-history row. Signals can change before candle close. Predictive quality on unfinished candles has not been validated, so historical completed-candle accuracy must not be claimed for this mode.

The default timing is aligned to :00 and :30 each minute. The first fresh decision can occur mid-slot immediately after startup. The next decision is allowed in the next slot, so the first two outputs can be less than thirty seconds apart. Feed delay, disconnects, history refresh or stale data can delay or skip a slot. Updates older than fifteen seconds are rejected; no fresh data means no new prediction. Yahoo prices are observed feed updates, not guaranteed every exchange trade. NIFTY index volume is zero, consistent with the historical Yahoo index data.

Initial history is fetched from Yahoo, not from old training CSVs. Current partial OHLC is seeded from the available Yahoo snapshot and updated using observed stream prices. At a new five-minute bucket and after reconnect, recent Yahoo history is refreshed to obtain officially supplied completed bars. If the preceding same-session completed bar has not arrived, inference waits/retries rather than promoting an incomplete observed bar to completed history. Completed history is kept separately from the provisional bar. Recent history/download time and full feature-rebuild time are excluded from model latency and logged separately.

## Output files

All remain in the existing CPU/FPGA folders:

- `provisional_5m_history.csv`: real downloaded completed bars for indicator initialization.
- `live_cpu.csv` / `live_fpga.csv`: predictions with source `yahoo_history_plus_live_stream` and status `provisional_unfinished_5m_candle`.
- `provisional_cpu_state.json` / `provisional_fpga_state.json`: last processed thirty-second slot keyed to model/scaler hashes, suppressing normal duplicate slots across restarts. As with the other modes, a crash between CSV append and state commit can repeat a row.

`time` is the actual decision-input evaluation time; `update_slot_ist` identifies the half-minute slot. `bar_open_ist` / `bar_close_ist` identify the underlying five-minute candle. `last_tick_ist` and `tick_age_seconds` show freshness. The initial candle opening timestamp and initial Open value are printed once above the continuous table. CPU console tables display time, close, signal and latency_ns. FPGA console tables display time, close, signal, fpga_cycles and latency_ns. The table header does not repeat at five-minute candle boundaries; rows continue for the full ten-minute launch window. The actual per-candle Open remains in the saved CSV. Raw scores remain in the CSV for audit and are omitted from console tables. Startup and summary also label both backends as provisional.

CPU latency includes scaling, batch-one single-thread LightGBM prediction API and argmax; separate inference/scaling timing is saved. FPGA core ns is hardware cycle count times 8 ns at 125 MHz, while UART round-trip is separate. No five-minute or thirty-second input waiting period is part of the core computation time.

Independently connected pipelines may observe different price updates or revised Yahoo history. Compare input OHLC and model versions before claiming the same decisions. Historical physical 643/643 agreement remains evidence for the earlier identical completed-bar input set, not automatic proof about independently sampled live inputs.

## Other supported modes

```powershell
# Original completed five-minute candles
python .\low_latency\cpu_pipeline_ml.py --five-minute
python .\fpga_ml\live_fpga.py --five-minute

# Previous experimental completed 30-second bars, requiring real fifty-bar warm-up
python .\low_latency\cpu_pipeline_ml.py --completed-30s
python .\fpga_ml\live_fpga.py --completed-30s

# Historical CPU benchmark, one run and thirty samples
python .\low_latency\cpu_pipeline_ml.py --historical
```

The previous streaming caches and results are preserved. Provisional results are stored separately from both those experiments and the verified historical benchmarks.

## Throughput in the terminal

The final summary prints actual live throughput as predictions divided by run duration from launch (including startup and feed waiting, excluding final socket cleanup). It also prints the reciprocal mean model inference time as a compute-rate estimate, and for FPGA the reciprocal mean UART round-trip as a transaction-rate estimate. These estimates are explicitly not measured saturated throughput. CPU model timing includes prediction API and argmax overhead. FPGA model timing uses its cycle counter; UART timing includes the request/response. With zero predictions, live throughput is zero and the model-rate estimate is unavailable.
