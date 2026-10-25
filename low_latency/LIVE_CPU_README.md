**Current default:** provisional live decisions every 30 seconds, initialized from actual five-minute Yahoo history. See `fpga_ml/PROVISIONAL_LIVE_README.md` from the main project. The older modes described below remain available through their explicit options.

**Updated default:** the entry point now uses experimental automatic 30-second streaming. See `fpga_ml/STREAMING_30S_README.md` from the main project. The original instructions below apply with `--five-minute`; historical and explicit replay modes remain available.

# CPU ML pipeline

From this folder run `python cpu_pipeline_ml.py --historical` for one run of 30 held-out samples from the FPGA test set, saved to `latency_log_ml.csv`.

Run `python cpu_pipeline_ml.py` for continuous Yahoo NIFTY five-minute completed candles. Stop with Ctrl+C for final latency statistics. `--once` checks once and exits. Output uses CPU table columns, with no FPGA comparison in the console. Results append to `live_cpu_results.csv`; replay rows are explicitly marked `offline_replay`.

The CPU uses the same saved LightGBM, RobustScaler, float32 inputs, feature order and BUY/HOLD/SELL mapping as the FPGA. It polls every 30 seconds, skips candles older than 15 minutes and suppresses duplicate timestamps. Its cache and state are `live_cpu_market_cache.csv` and `live_cpu_state.json` in this folder. The initial cache can be seeded from the FPGA live cache to preserve indicator history. Subsequent feeds may revise bars or have different history; compare identical input snapshots when checking agreement. FPGA fixed-point leaf scores can differ slightly from CPU floating-point scores.

Latency includes scaling, the single-thread batch-one LightGBM prediction API and argmax. CSV loading, model loading, network download and batch feature generation are measured separately or excluded. Use `inference_latency_ns` for model timing comparisons. No hardware connection is required for CPU inference.

Offline check: `python cpu_pipeline_ml.py --live --replay-file fpga_ml/live_market_cache.csv --replay-samples 3` (paths fall back to the main project). Offline replay does not update live state. Historical and replay checks do not establish future prediction accuracy. The live feed does not generate new completed candles outside its regular session.
