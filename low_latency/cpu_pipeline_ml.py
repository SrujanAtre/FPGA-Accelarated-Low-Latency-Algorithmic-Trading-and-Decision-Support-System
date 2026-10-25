"""Single-sample CPU LightGBM benchmark using the FPGA model and held-out data.
Run: python cpu_pipeline_ml.py for provisional live decisions every 30 seconds;
--five-minute for original live candles; --historical for the stored benchmark.
"""
from pathlib import Path
import argparse
import sys
import time
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from config.config import FEATURES, SPLIT
from fpga_ml.preprocessing import scale_features
from fpga_ml.power_reporting import (add_power_arguments, validate_power_arguments, power_summary,
                                    start_cpu_energy, save_cpu_energy)


def main():
    if '--power-benchmark' in sys.argv:
        sys.argv.remove('--power-benchmark')
        return cpu_power_benchmark_main()
    if '--scalability' in sys.argv:
        sys.argv.remove('--scalability')
        return scalability_main()
    if '--historical' not in sys.argv:
        if '--live' in sys.argv:
            sys.argv.remove('--live')
        if '--five-minute' in sys.argv or '--replay-file' in sys.argv:
            if '--five-minute' in sys.argv:sys.argv.remove('--five-minute')
            return live_main()
        from fpga_ml.streaming_30s import run_stream
        return run_stream('cpu')
    sys.argv.remove('--historical')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--samples', type=int, default=30, help='Held-out samples per run; 0 uses all samples')
    parser.add_argument('--power-seconds',type=float,default=5,
                        help='Separate repeated energy workload duration; default 5 seconds; 0 disables')
    add_power_arguments(parser)
    args = parser.parse_args()
    validate_power_arguments(parser,args)
    if args.runs < 1 or args.samples < 0:
        parser.error('--runs must be positive and --samples must be nonnegative')
    if not np.isfinite(args.power_seconds) or (args.power_seconds!=0 and args.power_seconds<1):
        parser.error('--power-seconds must be 0 (disabled) or finite and at least 1')
    print('\n[CPU ML] Loading historical market data...')
    frame = pd.read_csv(ROOT / 'data/processed/features.csv', index_col=0)
    start = int(len(frame)*SPLIT['train_frac']) + int(len(frame)*SPLIT['val_frac'])
    frame = frame.iloc[start:]
    if args.samples:
        frame = frame.head(args.samples)
    if frame.empty:
        raise ValueError('No held-out test samples available')
    stamps = pd.to_datetime(frame.index, utc=True).tz_convert('Asia/Kolkata')
    values = frame[FEATURES['feature_cols']].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Model inputs must be finite')
    market = pd.read_csv(ROOT / 'data/raw/nifty_1m.csv', index_col=0)
    market.index = pd.to_datetime(market.index, utc=True)
    closes = market.loc[stamps.tz_convert('UTC'), 'close'].to_numpy()
    model = joblib.load(ROOT / 'models/saved/lgbm.pkl')
    scaler = joblib.load(ROOT / 'models/saved/scaler.pkl')
    names = ['BUY', 'HOLD', 'SELL']
    print(f'[CPU ML] Data loaded: {len(values)} held-out rows')
    print('\n[HELD-OUT TEST SAMPLES]')
    print('Model: trained LightGBM | Batch size: 1 | Threads: 1')
    print(f'From: {stamps[0]}')
    print(f'To  : {stamps[-1]}')
    print(f'Filtered Rows: {len(values)}')
    # Warm the prediction API before measuring, without changing the trained model.
    warm = scale_features(scaler, values[0]).reshape(1, -1)
    for _ in range(20):
        model.booster_.predict(warm, raw_score=True, num_threads=1)
    print(f'\n[CPU ML] Running benchmark ({args.runs} runs)...')
    all_runs = []
    for run in range(1, args.runs+1):
        results = []
        loop_begin = time.perf_counter_ns()
        for i, value in enumerate(values):
            begin = time.perf_counter_ns()
            scaled = scale_features(scaler, value).reshape(1, -1)
            inference_begin = time.perf_counter_ns()
            scores = model.booster_.predict(scaled, raw_score=True, num_threads=1)[0]
            prediction = int(np.argmax(scores))
            end = time.perf_counter_ns()
            results.append({'run':run, 'time':str(stamps[i]), 'close':float(closes[i]),
                'score_buy':float(scores[0]), 'score_hold':float(scores[1]),
                'score_sell':float(scores[2]), 'signal':names[prediction],
                'latency_ns':end-begin, 'inference_latency_ns':end-inference_begin,
                'scaling_latency_ns':inference_begin-begin})
        loop_ns = time.perf_counter_ns()-loop_begin
        table = pd.DataFrame(results)
        table['run_loop_elapsed_ns'] = loop_ns
        all_runs.append(table)
        print(f'\n========== RUN {run} ==========')
        print(table[['time','close','score_buy','score_hold','score_sell','signal','latency_ns']].to_string(index=False))
    combined = pd.concat(all_runs, ignore_index=True)
    output = HERE / 'latency_log_ml.csv'
    combined.to_csv(output, index=False)
    print('\n[CPU ML] Benchmark completed successfully!')
    print('\n==============================================')
    print('       FINAL CPU ML LATENCY STATISTICS')
    print('==============================================')
    print(f'Number of Runs        : {args.runs}')
    print(f'Samples per Run       : {len(values)}')
    print(f'Total Latency Samples : {len(combined)}')
    print('----------------------------------------------')
    print(f'Average Latency (ns)  : {combined.latency_ns.mean():.2f}')
    print(f'Minimum Latency (ns)  : {combined.latency_ns.min():.2f}')
    print(f'Maximum Latency (ns)  : {combined.latency_ns.max():.2f}')
    print(f'Mean ML inference (ns): {combined.inference_latency_ns.mean():.2f}')
    print(f'Mean scaling (ns)     : {combined.scaling_latency_ns.mean():.2f}')
    print('==============================================')
    print('\nSignal Counts:')
    print(combined.signal.value_counts().to_string())
    print('\nLatency includes saved-scaler transformation, batch-one LightGBM prediction API and argmax.')
    print('CSV loading, feature generation, model loading, warm-up and table printing are excluded.')
    print('Use inference_latency_ns when comparing with FPGA model inference timing.')
    print()
    loop_seconds = sum(float(run.run_loop_elapsed_ns.iloc[0]) for run in all_runs) / 1e9
    energy_measurement=None
    if args.average_power_w is None and args.power_seconds:
        print(f'\n[CPU ML] Separate {args.power_seconds:g}-second historical energy replay (no extra sample rows).',flush=True)
        energy_measurement=cpu_energy_workload(model.booster_,scaler,values,args.power_seconds)
        save_cpu_energy(HERE/'historical_cpu_power.json',energy_measurement)
    for line in power_summary('cpu',len(combined),loop_seconds,args,measurement=energy_measurement):print(line)
    if energy_measurement is not None:print(f'Power results saved to: {HERE / "historical_cpu_power.json"}')
    print(f'\n[CPU ML] Results saved to: {output}')
    return 0



# Live market mode shares the historical model, scaler and CPU output columns.
from pathlib import Path
import argparse,contextlib,csv,hashlib,io,json,struct,time,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib,yfinance as yf
from config.config import DATA,FEATURES
from features.feature_engineering import build_features
from fpga_ml.preprocessing import scale_features

COLS=['open','high','low','close','volume']
NAMES=['BUY','HOLD','SELL']
CACHE=HERE/'live_cpu_market_cache.csv'
STATE=HERE/'live_cpu_state.json'
LOG=HERE/'live_cpu_results.csv'

def clean_bars(frame):
    if frame is None or frame.empty:raise ValueError('Yahoo returned no OHLCV bars')
    frame=frame.copy();frame.columns=[str(c).lower().replace(' ','_') for c in frame.columns]
    if not set(COLS).issubset(frame.columns):raise ValueError('Missing OHLCV columns')
    frame=frame[COLS].apply(pd.to_numeric,errors='coerce')
    frame.index=pd.to_datetime(frame.index,utc=True)
    frame=frame[~frame.index.duplicated(keep='last')].sort_index().dropna()
    local=frame.index.tz_convert('Asia/Kolkata')
    minutes=local.hour*60+local.minute
    frame=frame[(local.dayofweek<5)&(minutes>=555)&(minutes<930)&(local.minute%5==0)]
    finite=np.isfinite(frame.to_numpy()).all(axis=1)
    return frame[finite&(frame.close>0)&(frame.volume>=0)&(frame.high>=frame.low)]

def completed_bars(frame,now,grace_seconds=10):
    return frame[frame.index+pd.Timedelta(minutes=5,seconds=grace_seconds)<=now]

def features_for_inference(frame):
    with contextlib.redirect_stdout(io.StringIO()):
        features=build_features(frame,include_labels=False)
    if 'label' in features:raise RuntimeError('Live features must not require future labels')
    return features

def print_latency_summary(results):
    print('\n==============================================')
    print('       FINAL CPU ML LIVE LATENCY STATISTICS')
    print('==============================================')
    print(f'Total Latency Samples : {len(results)}')
    if not results:
        print('No new fresh completed bars were processed in this run.')
        return
    table=pd.DataFrame(results)
    print(f'Average Latency (ns)  : {table.latency_ns.mean():.2f}')
    print(f'Minimum Latency (ns)  : {table.latency_ns.min():.2f}')
    print(f'Maximum Latency (ns)  : {table.latency_ns.max():.2f}')
    print(f'Mean ML inference (ns): {table.inference_latency_ns.mean():.2f}')
    print('\nSignal Counts:')
    print(table.signal.value_counts().to_string())
    print('\nSample CPU Results:')
    print(table[['time','close','score_buy','score_hold','score_sell','signal','latency_ns']].head().to_string(index=False))

def live_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--max-age-minutes',type=float,default=15)
    parser.add_argument('--max-polls',type=int,default=0,help='Stop after N polls; 0 runs until Ctrl+C')
    parser.add_argument('--replay-file',type=Path)
    parser.add_argument('--replay-samples',type=int,default=3)
    add_power_arguments(parser)
    args=parser.parse_args()
    validate_power_arguments(parser,args)
    if args.poll_seconds<10 or args.max_age_minutes<=0 or args.replay_samples<1 or args.max_polls<0:
        parser.error('Poll interval must be >=10 seconds; age and replay sample count must be positive')
    ticker=DATA['ticker']
    model_hash=hashlib.sha256((ROOT/'models/saved/lgbm.pkl').read_bytes()).hexdigest()
    scaler=joblib.load(ROOT/'models/saved/scaler.pkl')
    model=joblib.load(ROOT/'models/saved/lgbm.pkl')
    warm=np.zeros((1,27),dtype=np.float32)
    for _ in range(20):model.booster_.predict(warm,raw_score=True,num_threads=1)
    scaler_hash=hashlib.sha256((ROOT/'models/saved/scaler.pkl').read_bytes()).hexdigest()
    identity={'ticker':ticker,'model_sha256':model_hash,'scaler_sha256':scaler_hash}
    last=None
    if STATE.exists() and not args.replay_file:
        state=json.loads(STATE.read_text())
        if all(state.get(k)==v for k,v in identity.items()):last=pd.Timestamp(state['last_bar_utc'])
    history=None
    seed=CACHE if CACHE.exists() else ROOT/'fpga_ml/live_market_cache.csv'
    if seed.exists() and not args.replay_file:
        history=clean_bars(pd.read_csv(seed,index_col=0))
    source=yf.Ticker(ticker)
    print(f'\n[CPU LIVE] {ticker} | Completed 5-minute bars | Trained LightGBM | Batch size: 1 | Threads: 1')
    print('[CPU LIVE] Mode: '+('OFFLINE REPLAY' if args.replay_file else 'YAHOO CURRENT MARKET DATA'))
    print('[CPU LIVE] Press Ctrl+C to stop. Waiting/stale bars do not generate signals.\n')
    count=0;first_fetch=history is None;polls=0;results=[];last_status=None
    energy_meter=start_cpu_energy(args)
    energy_loop_start=time.monotonic()
    try:
        while True:
            polls+=1
            download_begin=time.perf_counter_ns()
            try:
                if args.replay_file:
                    fetched=clean_bars(pd.read_csv(args.replay_file if args.replay_file.is_absolute() or args.replay_file.exists() else ROOT/args.replay_file,index_col=0))
                    now=fetched.index[-1]+pd.Timedelta(minutes=5,seconds=10)
                else:
                    fetched=clean_bars(source.history(period='1mo' if first_fetch else '5d',interval='5m',auto_adjust=False,actions=False,timeout=20))
                    now=pd.Timestamp.now(tz='UTC')
                download_ns=time.perf_counter_ns()-download_begin
                fetched=completed_bars(fetched,now)
                history=clean_bars(pd.concat([history,fetched])) if history is not None else fetched
                history=completed_bars(history,now)
                if history.empty:raise ValueError('No completed bars available')
                if not args.replay_file:history.to_csv(CACHE)
                compute_begin=time.perf_counter_ns()
                features=features_for_inference(history)
                feature_ns=time.perf_counter_ns()-compute_begin
            except Exception as exc:
                print(f'[CPU LIVE] Data unavailable: {exc}',flush=True)
                if args.once or args.replay_file or (args.max_polls and polls>=args.max_polls):return 1
                time.sleep(min(60,args.poll_seconds));continue
            first_fetch=False
            if args.replay_file:
                candidates=features.tail(args.replay_samples)
            elif last is None:
                candidates=features.tail(1)
            else:
                candidates=features[features.index>last]
            processed=False
            for stamp,row in candidates.iterrows():
                close_time=stamp+pd.Timedelta(minutes=5)
                age=(now-close_time).total_seconds()
                if not args.replay_file and age>args.max_age_minutes*60:continue
                value=row[FEATURES['feature_cols']].to_numpy(dtype=np.float64)
                begin=time.perf_counter_ns()
                scaled=scale_features(scaler,value).reshape(1,-1)
                inference_begin=time.perf_counter_ns()
                scores=model.booster_.predict(scaled,raw_score=True,num_threads=1)[0]
                prediction=int(np.argmax(scores))
                end=time.perf_counter_ns()
                preprocessing_ns=inference_begin-begin
                inference_ns=end-inference_begin
                latency_ns=end-begin
                count+=1;processed=True
                record={'source':'offline_replay' if args.replay_file else 'yahoo','ticker':ticker,'time':str(stamp.tz_convert('Asia/Kolkata')),'bar_close_ist':str(close_time.tz_convert('Asia/Kolkata')),'observed_at_utc':str(now),'close':float(history.loc[stamp,'close']),'score_buy':float(scores[0]),'score_hold':float(scores[1]),'score_sell':float(scores[2]),'signal':NAMES[prediction],'latency_ns':latency_ns,'inference_latency_ns':inference_ns,'scaling_latency_ns':preprocessing_ns,'input_bar_age_seconds':age,'download_ns':download_ns,'feature_rebuild_batch_ns':feature_ns}
                results.append(record)
                exists=LOG.exists() and LOG.stat().st_size>0
                with LOG.open('a',newline='',encoding='utf-8') as log:
                    writer=csv.DictWriter(log,fieldnames=list(record))
                    if not exists:writer.writeheader()
                    writer.writerow(record)
                if not args.replay_file:
                    state={**identity,'last_bar_utc':str(stamp)}
                    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)
                    last=stamp
                print(pd.DataFrame([record])[['time','close','score_buy','score_hold','score_sell','signal','latency_ns']].to_string(index=False,header=count==1),flush=True)
            if not processed:
                latest=history.index[-1].tz_convert('Asia/Kolkata')
                status=f'[CPU LIVE] Waiting for a new fresh completed 5-minute bar. Latest available: {latest}'
                if status!=last_status:
                    print(status,flush=True)
                    last_status=status
            else:
                last_status=None
            if args.once or args.replay_file or (args.max_polls and polls>=args.max_polls):break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print('\n[CPU LIVE] Stopped.')
    finally:
        measurement=energy_meter.stop(len(results),'Completed-bar live/replay loop, including data acquisition, feature building, logging and waits') if energy_meter else None
        save_cpu_energy(HERE/'live_cpu_power.json',measurement)
        if measurement is not None or args.average_power_w is not None:
            print()
            for line in power_summary('cpu',len(results),time.monotonic()-energy_loop_start,args,measurement=measurement):print(line)
    print_latency_summary(results)
    print(f'[CPU LIVE] Results saved to: {LOG}')
    return 0



def cpu_energy_workload(booster, scaler, values, seconds):
    """Dedicated batch-one replay; keep energy and normal latency runs separate."""
    expected=[int(np.argmax(booster.predict(scale_features(scaler,v).reshape(1,-1),
                  raw_score=True,num_threads=1)[0])) for v in values]
    meter=start_cpu_energy()
    count=0;mismatches=0
    try:
        if meter.reason is None:
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                index=count%len(values)
                scaled=scale_features(scaler,values[index]).reshape(1,-1)
                prediction=int(np.argmax(booster.predict(scaled,raw_score=True,num_threads=1)[0]))
                mismatches+=int(prediction!=expected[index]);count+=1
    except KeyboardInterrupt:
        print('\n[CPU POWER] Measurement stopped.',flush=True)
    finally:
        measurement=meter.stop(count,'Dedicated historical replay: saved scaling + batch-one prediction + argmax and correctness bookkeeping; loading/warm-up/printing excluded')
    measurement.update(reference_rows=len(values),mismatching_decisions=mismatches,inference_threads=1)
    if mismatches:raise RuntimeError('Prediction mismatch during CPU power benchmark')
    return measurement


def cpu_power_benchmark_main():
    parser=argparse.ArgumentParser(description='Measure RAPL CPU-package energy during repeated historical ML inference; existing latency/live CSVs are preserved')
    parser.add_argument('--seconds',type=float,default=10)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<1:parser.error('--seconds must be finite and at least 1')
    frame=pd.read_csv(ROOT/'data/processed/features.csv',index_col=0)
    start=int(len(frame)*SPLIT['train_frac'])+int(len(frame)*SPLIT['val_frac'])
    values=frame.iloc[start:][FEATURES['feature_cols']].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():raise ValueError('No finite held-out inputs')
    model_path=ROOT/'models/saved/lgbm.pkl';scaler_path=ROOT/'models/saved/scaler.pkl'
    model=joblib.load(model_path).booster_;scaler=joblib.load(scaler_path)
    for _ in range(20):model.predict(scale_features(scaler,values[0]).reshape(1,-1),raw_score=True,num_threads=1)
    print(f'[CPU POWER] {args.seconds:g}-second measurement | {len(values)} held-out inputs replayed | batch 1 | threads 1',flush=True)
    measurement=cpu_energy_workload(model,scaler,values,args.seconds)
    measurement.update(created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                       model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                       scaler_sha256=hashlib.sha256(scaler_path.read_bytes()).hexdigest())
    output=HERE/'cpu_power_benchmark.json';save_cpu_energy(output,measurement)
    print('\n'.join(power_summary('cpu',measurement['completed_decisions'],measurement.get('elapsed_seconds',0),measurement=measurement)))
    print(f'Power results saved to: {output}')
    return 0 if measurement['status']=='measured' else 1


def _scalability_initialize(model_path, scaler_path, barrier):
    """Independent process state; model loading and warm-up are not timed."""
    global _scalability_model, _scalability_scaler
    _scalability_model = joblib.load(model_path).booster_
    _scalability_scaler = joblib.load(scaler_path)
    warm = scale_features(_scalability_scaler, np.zeros(27)).reshape(1, -1)
    for _ in range(20):
        _scalability_model.predict(warm, raw_score=True, num_threads=1)
    barrier.wait(timeout=180)


def _scalability_ready():
    return True


def _scalability_one(model, scaler, value, threads):
    begin = time.perf_counter_ns()
    scaled = scale_features(scaler, value).reshape(1, -1)
    scores = model.predict(scaled, raw_score=True, num_threads=threads)[0]
    prediction = int(np.argmax(scores))
    end = time.perf_counter_ns()
    return prediction, end - begin, scores.tolist()


def _scalability_worker(task):
    index, value = task
    return index, _scalability_one(_scalability_model, _scalability_scaler, value, 1)


def scalability_main():
    """Repeatable historical batch-one throughput, service latency and load tests."""
    import concurrent.futures
    import hashlib
    import json
    import multiprocessing
    import os
    import platform
    import random
    import lightgbm
    parser = argparse.ArgumentParser(description=scalability_main.__doc__)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--replays', type=int, default=3, help='Full held-out-set passes per timed run')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    available = os.cpu_count() or 1
    levels = sorted(set(n for n in args.levels if 1 <= n <= available))
    if args.runs < 1 or args.replays < 1 or not levels or 1 not in levels:
        parser.error('Use positive runs/replays and levels including 1 within available logical CPUs')
    model_path = ROOT / 'models/saved/lgbm.pkl'
    scaler_path = ROOT / 'models/saved/scaler.pkl'
    data_path = ROOT / 'data/processed/features.csv'
    paths = [model_path, scaler_path, data_path]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frame = pd.read_csv(data_path, index_col=0)
    start = int(len(frame) * SPLIT['train_frac']) + int(len(frame) * SPLIT['val_frac'])
    frame = frame.iloc[start:]
    values = frame[FEATURES['feature_cols']].to_numpy(dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Expected a finite, nonempty held-out input set')
    model = joblib.load(model_path).booster_
    scaler = joblib.load(scaler_path)
    reference = np.asarray([model.predict(scale_features(scaler, v).reshape(1, -1),
                                         raw_score=True, num_threads=1)[0] for v in values])
    expected = reference.argmax(axis=1)
    n = len(values)
    print(f'\n[CPU SCALABILITY] {n} held-out inputs | batch 1 | {available} logical CPUs', flush=True)
    print(f'[CPU SCALABILITY] {args.runs} runs x {args.replays} complete replays per configuration', flush=True)
    print('[CPU SCALABILITY] Timed service = scaling + prediction API + argmax; throughput includes dispatch/bookkeeping.', flush=True)
    print('[CPU SCALABILITY] Loading and 20 warm-up calls per worker/configuration excluded. No live requests or FPGA required.', flush=True)
    configurations = [('inference_threads', k) for k in levels] + [('process_workers', k) for k in levels]
    random.Random(42).shuffle(configurations)
    runs, summaries, offered = [], [], []
    ctx = multiprocessing.get_context('spawn')
    all_latencies = {}
    for mode, level in configurations:
        pool = None
        if mode == 'process_workers':
            barrier = ctx.Barrier(level)
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=level, mp_context=ctx,
                initializer=_scalability_initialize, initargs=(str(model_path), str(scaler_path), barrier))
            print(f'[CPU SCALABILITY] Preparing {level} independent worker(s)...', flush=True)
            futures = [pool.submit(_scalability_ready) for _ in range(level)]
            for future in futures:
                future.result(timeout=240)
        else:
            for _ in range(20):
                model.predict(scale_features(scaler, values[0]).reshape(1, -1), raw_score=True, num_threads=level)
        stage_times = []
        try:
            for run in range(1, args.runs + 1):
                samples = n * args.replays
                latencies, correct, max_error = [], 0, 0.0
                begin = time.perf_counter_ns()
                if pool is None:
                    results = ((i % n, _scalability_one(model, scaler, values[i % n], level)) for i in range(samples))
                else:
                    results = pool.map(_scalability_worker, ((i % n, values[i % n]) for i in range(samples)), chunksize=1)
                for index, (prediction, latency, scores) in results:
                    correct += int(prediction == expected[index])
                    max_error = max(max_error, float(np.max(np.abs(reference[index] - scores))))
                    latencies.append(latency)
                elapsed_ns = time.perf_counter_ns() - begin
                stage_times.extend(latencies)
                row = dict(mode=mode, level=level, run=run, samples=samples, elapsed_ns=elapsed_ns,
                           throughput_per_s=samples * 1e9 / elapsed_ns,
                           mean_service_us=float(np.mean(latencies) / 1000),
                           p95_service_us=float(np.percentile(latencies, 95) / 1000),
                           correct=correct, max_score_error=max_error)
                runs.append(row)
                print(f'  {mode}={level} run {run}/{args.runs}: {row["throughput_per_s"]:.2f}/s | '
                      f'mean {row["mean_service_us"]:.3f} us | p95 {row["p95_service_us"]:.3f} us | {correct}/{samples} match', flush=True)
                if correct != samples or max_error > 1e-9:
                    raise RuntimeError('Scalability correctness check failed; no speedup should be claimed')
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        group = [r for r in runs if r['mode'] == mode and r['level'] == level]
        all_latencies[(mode, level)] = stage_times
        summaries.append(dict(mode=mode, level=level, runs=len(group), samples=sum(r['samples'] for r in group),
            throughput_median_per_s=float(np.median([r['throughput_per_s'] for r in group])),
            throughput_min_per_s=min(r['throughput_per_s'] for r in group),
            throughput_max_per_s=max(r['throughput_per_s'] for r in group),
            mean_service_us=float(np.mean(stage_times) / 1000),
            p95_service_us=float(np.percentile(stage_times, 95) / 1000),
            correct=sum(r['correct'] for r in group), max_score_error=max(r['max_score_error'] for r in group)))
    for summary in summaries:
        baseline = next(s for s in summaries if s['mode'] == summary['mode'] and s['level'] == 1)
        summary['throughput_speedup_vs_same_mode_1'] = summary['throughput_median_per_s'] / baseline['throughput_median_per_s']
    base = next(s for s in summaries if s['mode'] == 'inference_threads' and s['level'] == 1)
    rates = [round(base['throughput_median_per_s'] * factor, 3) for factor in (.5, 1., 2.)]
    print('[CPU SCALABILITY] Scheduled offered-load replay: one inference thread, one consumer.', flush=True)
    for rate in rates:
        for run in range(1, args.runs + 1):
            waits, stages, response_times = [], [], []
            correct = 0
            epoch = time.perf_counter_ns()
            for index, value in enumerate(values):
                arrival = epoch + int(index * 1e9 / rate)
                remaining = arrival - time.perf_counter_ns()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
                actual_start = time.perf_counter_ns()
                prediction, latency, _ = _scalability_one(model, scaler, value, 1)
                end = time.perf_counter_ns()
                waits.append(max(0, actual_start - arrival))
                response_times.append(end - arrival)
                stages.append(latency)
                correct += int(prediction == expected[index])
            elapsed_ns = time.perf_counter_ns() - epoch
            offered.append(dict(offered_per_s=rate, run=run, samples=n, elapsed_ns=elapsed_ns,
                achieved_per_s=n * 1e9 / elapsed_ns, mean_service_us=float(np.mean(stages) / 1000),
                p95_service_us=float(np.percentile(stages, 95) / 1000),
                max_external_wait_us=max(waits) / 1000,
                p95_external_wait_us=float(np.percentile(waits, 95) / 1000),
                p95_scheduled_response_us=float(np.percentile(response_times, 95) / 1000), correct=correct))
            if correct != n:
                raise RuntimeError('Offered-load correctness failed')
    if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError('An input artifact changed during the benchmark')
    summaries.sort(key=lambda s: (s['mode'], s['level']))
    report = dict(created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        hardware=dict(processor=platform.processor(), logical_cpus=available),
        software=dict(python=platform.python_version(), lightgbm=lightgbm.__version__, numpy=np.__version__,
                      pandas=pd.__version__, platform=platform.platform()),
        artifact_sha256=hashes, input_rows=n, input_first=str(frame.index[0]), input_last=str(frame.index[-1]),
        configuration_order=configurations, warmup_calls=20, batch_size=1, summaries=summaries,
        runs=runs, offered_load=offered,
        limitations=['Fixed configuration order shuffled with seed 42; ordinary Windows scheduling, no CPU affinity or frequency lock.',
          'Service latency excludes process queue/IPC; measured throughput includes submission, IPC, result collection and verification.',
          'Worker creation/model loading/warm-up excluded; one independent model and one inference thread per worker.',
          'Offered-load arrivals are scheduled in this Python harness; wait includes OS scheduling lateness and backlog, not pure queue residence.',
          'Historical replay excludes market-data waits, full feature building, model loading and printing. No power measurement.',
          'CPU and FPGA measurements are from different runs and boundaries; do not treat ratios as a matched end-to-end comparison.'])
    (HERE / 'cpu_scalability_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(runs).to_csv(HERE / 'cpu_scalability_runs.csv', index=False)
    pd.DataFrame(offered).to_csv(HERE / 'cpu_scalability_offered_load.csv', index=False)
    output = pd.DataFrame(summaries)
    print('\nCPU SCALABILITY RESULTS', flush=True)
    print(output[['mode','level','throughput_median_per_s','mean_service_us','p95_service_us',
                  'throughput_speedup_vs_same_mode_1','correct','samples']].to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nSCHEDULED OFFERED-LOAD RESULTS (median rates; worst wait across runs)')
    load = pd.DataFrame(offered).groupby('offered_per_s').agg(achieved_per_s=('achieved_per_s','median'),
              max_external_wait_us=('max_external_wait_us','max'))
    print(load.to_string(float_format=lambda v:f'{v:.3f}'))
    print('\nResults: cpu_scalability_results.json, cpu_scalability_runs.csv, cpu_scalability_offered_load.csv')
    print('Default live pipeline, model, scaler and original historical/live CSVs were not changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
