**Current default:** provisional live decisions every 30 seconds, initialized from actual five-minute Yahoo history. See `fpga_ml/PROVISIONAL_LIVE_README.md` from the main project. The older modes described below remain available through their explicit options.

**Updated default:** the entry point now uses experimental automatic 30-second streaming. See `fpga_ml/STREAMING_30S_README.md` from the main project. The original instructions below apply with `--five-minute`; historical and explicit replay modes remain available.

# Current market data to the ZCU104 FPGA

Run `python live_fpga.py` from the `fpga_ml` folder inside the main project. Stop with Ctrl+C. The script polls Yahoo every 30 seconds and processes each newly completed five-minute NIFTY (`^NSEI`) bar once. It prints FPGA signal and measured core latency, without CPU signals. The existing `fpga_ml` folder is used; no new project folder, model training, scaler fitting, or bitstream change is required.

`python live_fpga.py --once` fetches current bars, processes the latest eligible unprocessed bar, and exits. If already processed, stale, or outside the session, it prints a waiting message. `--max-polls 2` is a bounded polling check. The board must be powered, programmed with the ML bitstream, and connected to COM7; only one serial script may use COM7 at a time.

The saved RobustScaler and deployed trained LightGBM are reused. Initial Yahoo history uses a month of five-minute bars to warm indicators. Later polling fetches five days and merges with the retained history so that the history start does not shift every poll. Cached values are updated for incoming bars. Feature engineering is causal and preserves the existing training definitions; `build_features(..., include_labels=False)` retains the latest rows without generating future-dependent training labels. The default training call still includes labels and produces the same cached historical features.

NSE weekday regular-session bars beginning at 09:15–15:25 IST are eligible. A bar is complete only after its timestamp plus five minutes plus a 10-second grace period. Missing session days yield no new predictions; there is no fabricated fallback feed. Bars more than 15 minutes past their close are skipped by default. Yahoo timestamps and the logged age must be used to assess freshness: this is a polling market-data demo, not a claim of undelayed exchange ticks or high-frequency data arrival. Recomputing features over retained history is measured separately as batch preprocessing, not incremental FPGA latency.

Files stored in the existing `fpga_ml` folder:

- `live_fpga.py`: current Yahoo data -> 27 features -> saved scaler -> UART -> FPGA learned-model inference.
- `live_market_cache.csv`: a separate current-market cache, leaving historical training CSVs unchanged.
- `live_fpga_state.json`: last processed bar, keyed by ticker/model/scaler hashes; suppresses duplicate prediction across restarts.
- `live_fpga_results.csv`: actual FPGA responses, market timestamps, source, input age, download time, batch feature time, scaling time and UART round-trip. Scores are recorded for audit but not printed as CPU comparisons.
- `test_live_fpga.py`: checks unchanged training features/labels, causal live features, latest-row retention, and completed-bar grace logic.

CSV logging occurs before the last-bar state is committed. A crash between those writes can cause a repeated row on restart; this prototype does not place orders and does not claim transactional exactly-once execution. If the state file is removed, the next run can process the latest bar again.

For explicit offline checking: `python live_fpga.py --replay-file data/raw/nifty_1m.csv --replay-samples 3`. Those records are marked `offline_replay` and do not update the live last-bar state. They are not presented as current market data.

Network/data errors produce a message and retry with a bounded wait; malformed FPGA responses stop the run. If the board was powered off, reprogram the existing ML bitstream first. The script does not silently switch to CPU inference. Model inference latency is measured by the FPGA cycle counter at 125 MHz; HTTP/USB/host time and five-minute bar formation are distinct timings.

Yahoo interface documentation: [yfinance history](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html). This uses five-minute OHLCV polling, not a WebSocket tick reconstruction.

Console rows use the historical benchmark format: timestamp, Close, Latency in cycles/ns/us, and Signal. On Ctrl+C or a bounded run ending, a summary prints sample count, average/minimum/maximum latency, signal counts, and a sample results table. Feed freshness and UART timings remain in the CSV.
