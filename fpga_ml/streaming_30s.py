"""Observed Yahoo streaming prices -> automatic 30-second bars.
Existing five-minute model is reused experimentally; no retraining is claimed.
"""
from pathlib import Path
import argparse, contextlib, csv, hashlib, io, json, logging, math, queue, threading, time
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config.config import DATA, FEATURES
from features.feature_engineering import build_features
from fpga_ml.preprocessing import scale_features

ROOT=Path(__file__).resolve().parent.parent
NAMES=['BUY','HOLD','SELL']

class CandleBuilder:
    """Only observed event-time prices; never fill empty or initial partial bars."""
    def __init__(self, connected_at, lateness_seconds=2):
        stamp=pd.Timestamp(connected_at).tz_convert('UTC')
        self.first_full=stamp.ceil('30s')
        self.bars={}; self.closed_before=self.first_full
        self.lateness=pd.Timedelta(seconds=lateness_seconds)
        self.seen=set()

    def add(self, message, received_at):
        if message.get('id')!=DATA['ticker']:
            return False
        try:
            stamp=pd.Timestamp(float(message['time']),unit='ms',tz='UTC')
            price=float(message['price'])
        except (KeyError,ValueError,TypeError,OverflowError):
            return False
        now=pd.Timestamp(received_at).tz_convert('UTC')
        local=stamp.tz_convert('Asia/Kolkata'); minute=local.hour*60+local.minute
        if not math.isfinite(price) or price<=0 or (now-stamp).total_seconds()>15 or stamp>now+pd.Timedelta(seconds=2):
            return False
        if local.dayofweek>=5 or not 555<=minute<930:
            return False
        bucket=stamp.floor('30s')
        if bucket<self.first_full or bucket<self.closed_before:
            return False
        key=(stamp.value,price)
        if key in self.seen:
            return False
        self.seen.add(key)
        bar=self.bars.get(bucket)
        if bar is None:
            self.bars[bucket]={'open':price,'high':price,'low':price,'close':price,'volume':0.0,'tick_count':1,'first_tick':stamp,'last_tick':stamp}
        else:
            bar['high']=max(bar['high'],price);bar['low']=min(bar['low'],price)
            if stamp<bar['first_tick']:bar['open']=price;bar['first_tick']=stamp
            if stamp>=bar['last_tick']:bar['close']=price;bar['last_tick']=stamp
            bar['tick_count']+=1
        return True

    def close(self, now):
        now=pd.Timestamp(now).tz_convert('UTC')
        cutoff=(now-self.lateness).floor('30s')
        ready=[]
        for stamp in sorted(list(self.bars)):
            if stamp<cutoff:
                ready.append((stamp,self.bars.pop(stamp)))
        self.closed_before=max(self.closed_before,cutoff)
        self.seen={k for k in self.seen if k[0]>=cutoff.value}
        return ready


def run_completed_stream(backend, argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-samples',type=int,default=30,help='Predictions before stopping; 0 continues')
    parser.add_argument('--max-runtime-seconds',type=float,default=0,help='Optional bounded connection test')
    parser.add_argument('--port',default='COM7')
    args=parser.parse_args(argv)
    if args.max_samples<0 or args.max_runtime_seconds<0:parser.error('Limits must be nonnegative')
    folder=ROOT/('fpga_ml' if backend=='fpga' else 'low_latency')
    cache=folder/'stream_30s_candles.csv'
    log=folder/('live_'+backend+'_30s_results.csv')
    state_file=folder/('stream_30s_'+backend+'_state.json')
    prefix=f'[{backend.upper()} 30S]'
    print(f'\n{prefix} Yahoo streaming {DATA["ticker"]} | Automatic 30-second observed-price candles',flush=True)
    print(f'{prefix} EXPERIMENTAL: current model/scaler were trained on five-minute inputs; 30-second predictive accuracy is unvalidated.',flush=True)
    print(f'{prefix} No synthetic candles or historical-candle splitting. Fresh stream and indicator warm-up are required.',flush=True)
    print(f'{prefix} Index volume is zero (NIFTY is an index). Prices reflect only updates delivered by Yahoo, not every exchange tick.',flush=True)
    scaler=joblib.load(ROOT/'models/saved/scaler.pkl')
    model=joblib.load(ROOT/'models/saved/lgbm.pkl') if backend=='cpu' else None
    identity=hashlib.sha256((ROOT/'models/saved/lgbm.pkl').read_bytes()+(ROOT/'models/saved/scaler.pkl').read_bytes()).hexdigest()
    last=None
    if state_file.exists():
        state=json.loads(state_file.read_text())
        if state.get('model_scaler_sha256')==identity:last=pd.Timestamp(state['last_bar'])
    history=pd.DataFrame()
    if cache.exists():
        history=pd.read_csv(cache,index_col=0)
        history.index=pd.to_datetime(history.index,utc=True)
        # Use a contiguous same-session suffix, not history across feed gaps.
        if len(history):
            gaps=history.index.to_series().diff()!=pd.Timedelta(seconds=30)
            history=history.iloc[np.flatnonzero(gaps.to_numpy())[-1]:]
    events=queue.Queue(maxsize=8192);stop=threading.Event();overflow=threading.Event()
    active={'socket':None};count=0;results=[];port=None
    def put(kind,value):
        try:events.put_nowait((kind,value,pd.Timestamp.now(tz='UTC')))
        except queue.Full:overflow.set()
    class StreamLogFilter(logging.Filter):
        def filter(self, record):
            # The worker emits concise errors itself; normal socket shutdown is not an error.
            return not stop.is_set()
    shutdown_filter=StreamLogFilter()
    def worker():
        while not stop.is_set():
            try:
                with yf.WebSocket(verbose=False) as ws:
                    active['socket']=ws
                    ws.logger.addFilter(shutdown_filter)
                    ws.subscribe([DATA['ticker']]);put('connected',None)
                    ws.listen(lambda msg:put('tick',msg))
                    if not stop.is_set():put('error','Stream ended; reconnecting. Partial candle discarded.')
            except Exception as exc:
                if not stop.is_set():put('error',str(exc))
            finally:
                if active['socket'] is not None:
                    active['socket'].logger.removeFilter(shutdown_filter)
                active['socket']=None
            stop.wait(5)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    start=time.monotonic();last_notice=-30;builder=None;last_tick=None
    try:
        while True:
            if overflow.is_set():raise RuntimeError('Tick queue overflow; stopping instead of silently losing updates')
            if args.max_runtime_seconds and time.monotonic()-start>=args.max_runtime_seconds:break
            try:kind,value,received=events.get(timeout=.5)
            except queue.Empty:kind=None
            if kind=='connected':
                builder=CandleBuilder(received)
                print(f'{prefix} Connected. Initial partial candle is skipped; waiting for fresh {DATA["ticker"]} updates.',flush=True)
            elif kind=='error':
                builder=None
                print(f'{prefix} Feed: {value}',flush=True)
            elif kind=='tick' and builder is not None:
                if builder.add(value,received):last_tick=received
            # Finalize only after all currently queued events were processed.
            if builder is not None and events.empty():
                for stamp,bar in builder.close(pd.Timestamp.now(tz='UTC')):
                    if last is not None and stamp<=last:continue
                    if len(history) and stamp-history.index[-1]!=pd.Timedelta(seconds=30):
                        history=pd.DataFrame()
                        print(f'{prefix} Feed gap/session change: indicator history reset.',flush=True)
                    item={k:bar[k] for k in ['open','high','low','close','volume','tick_count']}
                    history=pd.concat([history,pd.DataFrame([item],index=pd.DatetimeIndex([stamp]))])
                    history.index.name='time';history.to_csv(cache)
                    print(f'{prefix} Candle {stamp.tz_convert("Asia/Kolkata")} | Close={bar["close"]:.2f} | Observed updates={bar["tick_count"]}',flush=True)
                    if len(history)<50:
                        print(f'{prefix} Indicator warm-up: {len(history)}/50 real contiguous candles.',flush=True)
                        continue
                    feature_begin=time.perf_counter_ns()
                    with contextlib.redirect_stdout(io.StringIO()):features=build_features(history,include_labels=False)
                    if stamp not in features.index:continue
                    vector=features.loc[stamp,FEATURES['feature_cols']].to_numpy(dtype=np.float64)
                    feature_ns=time.perf_counter_ns()-feature_begin
                    begin=time.perf_counter_ns();scaled=scale_features(scaler,vector)
                    infer_begin=time.perf_counter_ns()
                    if backend=='cpu':
                        scores=model.booster_.predict(scaled.reshape(1,-1),raw_score=True,num_threads=1)[0]
                        pred=int(np.argmax(scores));infer_end=time.perf_counter_ns()
                        extra={'inference_latency_ns':infer_end-infer_begin,'latency_ns':infer_end-begin}
                    else:
                        from fpga_ml.live_fpga import request_prediction
                        import serial
                        if port is None:
                            port=serial.Serial(args.port,115200,timeout=3,write_timeout=3);port.reset_input_buffer()
                        pred,cycles,integer_scores,roundtrip=request_prediction(port,scaled)
                        scores=np.asarray(integer_scores,dtype=float)/(1<<24)
                        extra={'fpga_cycles':cycles,'fpga_inference_ns':cycles*8,'uart_roundtrip_ns':roundtrip}
                    count+=1
                    record={'source':'yahoo_websocket_observed','model_status':'experimental_5m_model_on_30s_inputs','ticker':DATA['ticker'],'time':str(stamp.tz_convert('Asia/Kolkata')),'bar_close_ist':str((stamp+pd.Timedelta(seconds=30)).tz_convert('Asia/Kolkata')),'close':bar['close'],'score_buy':float(scores[0]),'score_hold':float(scores[1]),'score_sell':float(scores[2]),'signal':NAMES[pred],'tick_count':bar['tick_count'],'scaling_ns':infer_begin-begin,'feature_rebuild_batch_ns':feature_ns,**extra}
                    exists=log.exists() and log.stat().st_size>0
                    with log.open('a',newline='',encoding='utf-8') as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(record))
                        if not exists:writer.writeheader()
                        writer.writerow(record)
                    temp=state_file.with_suffix('.tmp');temp.write_text(json.dumps({'model_scaler_sha256':identity,'last_bar':str(stamp)},indent=2));temp.replace(state_file)
                    last=stamp;results.append(record)
                    if backend=='cpu':print(pd.DataFrame([record])[['time','close','score_buy','score_hold','score_sell','signal','latency_ns']].to_string(index=False,header=count==1),flush=True)
                    else:print(f'[{count:03d}] {record["time"]} | Close={bar["close"]:.2f} | Latency={cycles} cycles ({cycles*8} ns / {cycles*.008:.3f} us) | Signal={NAMES[pred]}',flush=True)
                    if args.max_samples and count>=args.max_samples:break
            if args.max_samples and count>=args.max_samples:break
            if time.monotonic()-start-last_notice>=30:
                if last_tick is None or (pd.Timestamp.now(tz='UTC')-last_tick).total_seconds()>30:
                    print(f'{prefix} No fresh usable NIFTY updates. No candle or decision is fabricated.',flush=True)
                last_notice=time.monotonic()-start
    except KeyboardInterrupt:print(f'\n{prefix} Stopped.')
    finally:
        stop.set()
        if active['socket'] is not None:
            with contextlib.suppress(Exception):active['socket'].close()
        thread.join(timeout=2)
        if port is not None:port.close()
    print(f'\n{prefix} Samples processed: {count}')
    if results:
        column='latency_ns' if backend=='cpu' else 'fpga_inference_ns'
        times=pd.Series([r[column] for r in results])
        print(f'Average Latency (ns): {times.mean():.2f}\nMinimum Latency (ns): {times.min():.2f}\nMaximum Latency (ns): {times.max():.2f}')
        print('Signal Counts:');print(pd.Series([r['signal'] for r in results]).value_counts().to_string())
    print(f'{prefix} Results: {log}')
    return 0


def run_stream(backend, argv=None):
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    if '--completed-30s' in arguments:
        arguments.remove('--completed-30s')
        return run_completed_stream(backend,arguments)
    from fpga_ml.provisional_live import run_provisional
    return run_provisional(backend,arguments)
