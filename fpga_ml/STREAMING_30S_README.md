**Current default:** provisional live decisions every 30 seconds, initialized from actual five-minute Yahoo history. See `fpga_ml/PROVISIONAL_LIVE_README.md` from the main project. The older modes described below remain available through their explicit options.

# Automatic thirty-second live candles

Both existing entry points now default to Yahoo WebSocket prices and automatic observed-price thirty-second candles:

```powershell
python .\low_latency\cpu_pipeline_ml.py
python .\fpga_ml\live_fpga.py
```

No manual prices, splitting historical candles, repeated old predictions, or fabricated candles are used. A socket receives updates continuously; there is no candle-download polling interval in this mode. Each UTC-aligned thirty-second interval closes with a two-second lateness allowance. Initial partial intervals and empty intervals are skipped. Out-of-order ticks within an open candle are ordered by their event timestamps. Duplicate timestamp/price pairs and stale prices are ignored. A disconnect discards the unfinished candle and retries after five seconds.

## Important status

**This is an experimental input-interval change. The deployed LightGBM and RobustScaler were trained on five-minute bars. Their prediction accuracy on thirty-second features has not been established.** The existing model/scaler/FPGA bitstream have not been overwritten or retrained. Both console startup and each prediction CSV record identify this status. Collection and CPU/FPGA calculation can be demonstrated, but do not claim a validated thirty-second trading model.

Indicators need **50 contiguous real thirty-second bars (about 25 minutes)** before the first decision. Start both programs early. They print incoming candle closes and warm-up progress. Caches are retained in their existing respective folders. A feed gap or new session resets indicator history rather than inventing missing bars. Once warm, each eligible new candle can produce one decision. By default the run stops after 30 predictions; `--max-samples 0` runs until Ctrl+C. Fresh data availability is controlled by Yahoo; no fixed decision cadence is guaranteed.

For NIFTY the volume field is zero because the instrument is an index, consistent with the original Yahoo NIFTY inputs. Prices and OHLC reflect only the stream updates observed by the application, not every trade on the exchange. This is not a reconstructed official exchange candle feed. Independently received stream subsets or missed messages can make the two programs' candles differ; compare their OHLC caches before claiming identical inputs.

## Files and timing

Each existing folder gets `stream_30s_candles.csv` containing actual observed OHLC, zero index volume and update count. Prediction files are `low_latency/live_cpu_30s_results.csv` and `fpga_ml/live_fpga_30s_results.csv`. These are created only after an actual inference response. Last processed timestamps are stored separately in `stream_30s_cpu_state.json` / `stream_30s_fpga_state.json`. State follows the model/scaler hashes. CSV append precedes state commit, so a crash between them can repeat a row.

CPU output keeps time, close, scores, signal and latency. CPU timing includes scaling plus single-thread LightGBM prediction API and argmax; separate inference/scaling fields are saved. FPGA output keeps time, close, cycles, ns/us and signal. It uses physical COM7 inference only, with no CPU fallback. FPGA UART round-trip and preprocessing times are separately logged. The two-second close allowance and feed latency are not part of FPGA core inference latency.

## Other modes

```powershell
# Bounded feed/aggregation check; does not bypass warm-up
python .\low_latency\cpu_pipeline_ml.py --max-runtime-seconds 90
python .\fpga_ml\live_fpga.py --max-runtime-seconds 90

# Original verified five-minute live input path
python .\low_latency\cpu_pipeline_ml.py --five-minute
python .\fpga_ml\live_fpga.py --five-minute

# Existing historical CPU benchmark: one run, 30 samples
python .\low_latency\cpu_pipeline_ml.py --historical
```

Both live programs can run simultaneously; only FPGA opens COM7, after features are ready. The board must be programmed with the ML bitstream. No new folder is required.

## Verified during implementation

Six tests passed, covering unchanged historical feature definitions, causal features, completed bars, first partial-candle rejection, duplicates/out-of-order ticks, stale/empty data and the fifty-bar warm-up. Live Yahoo subscriptions delivered NIFTY updates. Both entry points automatically saved the same October 15 12:01:30 IST candle (open/high 22967.40, low 22963.05, close 22964.75, 28 observed updates). This proves real collection and aggregation, not completion of a full warm-up or new physical thirty-second inference validation.

A validated thirty-second model requires collecting enough genuine bars, training/holding out data at that interval, exporting the selected model/scaler to the FPGA and repeating CPU/FPGA correctness and timing checks.

Yahoo streaming API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.WebSocket.html
