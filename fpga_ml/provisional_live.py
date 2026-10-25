"""Five-minute feature history with provisional live decisions every 30 seconds."""
from pathlib import Path
import argparse, contextlib, csv, hashlib, io, json, logging, math, queue, threading, time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from config.config import DATA, FEATURES
from features.feature_engineering import build_features
from fpga_ml.preprocessing import scale_features
from fpga_ml.power_reporting import (add_power_arguments, validate_power_arguments, power_summary,
                                    start_cpu_energy, save_cpu_energy)

ROOT=Path(__file__).resolve().parent.parent
NAMES=['BUY','HOLD','SELL']
OHLCV=['open','high','low','close','volume']

class ProvisionalHistory:
    """Keep completed historical bars separate from the unfinished live bar."""
    def __init__(self):
        self.history=pd.DataFrame(columns=OHLCV)
        self.seeds={};self.current=None;self.bucket=None;self.last_tick=None;self.seen=set()

    def refresh(self, frame, now):
        from fpga_ml.live_fpga import clean_bars
        clean=clean_bars(frame)
        now=pd.Timestamp(now).tz_convert('UTC')
        completed=clean[clean.index+pd.Timedelta(minutes=5)<=now]
        if self.history.empty:self.history=completed.copy()
        else:
            self.history=pd.concat([self.history,completed])
            self.history=self.history[~self.history.index.duplicated(keep='last')].sort_index()
        self.seeds={stamp:row.to_dict() for stamp,row in clean.iterrows() if stamp<=now<stamp+pd.Timedelta(minutes=5)}

    def add(self, message, received_at):
        if message.get('id')!=DATA['ticker']:return False
        try:
            stamp=pd.Timestamp(float(message['time']),unit='ms',tz='UTC');price=float(message['price'])
        except (KeyError,ValueError,TypeError,OverflowError):return False
        now=pd.Timestamp(received_at).tz_convert('UTC');local=stamp.tz_convert('Asia/Kolkata')
        minutes=local.hour*60+local.minute
        if not math.isfinite(price) or price<=0 or (now-stamp).total_seconds()>15 or stamp>now+pd.Timedelta(seconds=2):return False
        if local.dayofweek>=5 or not 555<=minutes<930:return False
        bucket=stamp.floor('5min')
        if bucket!=now.floor('5min'):return False
        if self.bucket is not None and bucket<self.bucket:return False
        key=(stamp.value,price)
        if key in self.seen:return False
        if bucket!=self.bucket:
            self.bucket=bucket;self.seen=set()
            seed=self.seeds.get(bucket)
            self.current=dict(seed) if seed is not None else dict(open=price,high=price,low=price,close=price,volume=0.)
            self.current['volume']=0.
            self.last_tick=None
        self.seen.add(key)
        self.current['high']=max(self.current['high'],price)
        self.current['low']=min(self.current['low'],price)
        if self.last_tick is None or stamp>=self.last_tick:
            self.current['close']=price;self.last_tick=stamp
        return True

    def feature_frame(self, now):
        now=pd.Timestamp(now).tz_convert('UTC')
        if self.bucket is None or self.bucket!=now.floor('5min') or self.last_tick is None or (now-self.last_tick).total_seconds()>15:
            raise ValueError('Waiting for fresh stream prices for the current candle')
        history=self.history[self.history.index<self.bucket]
        local=self.bucket.tz_convert('Asia/Kolkata')
        if local.hour*60+local.minute>555 and self.bucket-pd.Timedelta(minutes=5) not in history.index:
            raise ValueError('Waiting for Yahoo to publish the preceding completed five-minute candle')
        if len(history)<50:raise ValueError('Yahoo history has fewer than 50 completed five-minute bars')
        # Exactly one unfinished bar replaces the current bucket, never many 30-second rows.
        return pd.concat([history,pd.DataFrame([self.current],index=pd.DatetimeIndex([self.bucket]))])[OHLCV]


def format_live_row(record, backend, header=False):
    """Stable terminal widths shared by headers and every streamed row."""
    columns=[('time',26),('close',12),('signal',8)]
    columns+= [('fpga_cycles',12),('latency_ns',14)] if backend=='fpga' else [('latency_ns',14)]
    fields=[]
    for name,width in columns:
        if header:
            value=name
        elif name=='time':
            value=pd.Timestamp(record[name]).isoformat(sep=' ',timespec='seconds')
        elif name=='close':
            value=f'{record[name]:.2f}'
        elif name in ('fpga_cycles','latency_ns'):
            value=str(int(record[name]))
        else:
            value=str(record[name])
        fields.append(f'{value:<{width}}' if name in ('time','signal') else f'{value:>{width}}')
    return '  '.join(fields).rstrip()


def throughput_summary(results, elapsed_seconds, backend):
    """Separate observed wall-clock throughput from reciprocal latency estimates."""
    lines=[f'Run duration (s): {elapsed_seconds:.2f}']
    if elapsed_seconds>0:
        lines.append(f'Live throughput (includes feed wait): {len(results)/elapsed_seconds:.4f} decisions/s')
    else:
        lines.append('Live throughput: unavailable (no elapsed time)')
    if not results:
        lines.append('Model compute rate estimate: unavailable (no predictions)')
        return lines
    key='inference_latency_ns' if backend=='cpu' else 'latency_ns'
    total_ns=sum(float(row[key]) for row in results)
    if total_ns>0:
        lines.append(f'Model compute rate estimate (1 / mean inference time): {len(results)*1e9/total_ns:.2f} inferences/s')
    if backend=='fpga':
        uart_ns=sum(float(row['uart_roundtrip_ns']) for row in results)
        if uart_ns>0:lines.append(f'UART transaction rate estimate (1 / mean round-trip): {len(results)*1e9/uart_ns:.2f} transactions/s')
    lines.append('Compute/transaction rates are latency-derived estimates, not measured saturated throughput.')
    return lines


def run_provisional(backend, argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-samples',type=int,default=0,help='Optional decision limit; 0 uses only the time window')
    parser.add_argument('--max-runtime-seconds',type=float,default=600,help='Window from launch, including startup; default 600 seconds, 0 unlimited')
    parser.add_argument('--port',default='COM7')
    add_power_arguments(parser)
    args=parser.parse_args(argv)
    validate_power_arguments(parser,args)
    if args.max_samples<0 or args.max_runtime_seconds<0:parser.error('Limits must be nonnegative')
    start=time.monotonic()
    launch_time=pd.Timestamp.now(tz='UTC')
    deadline=launch_time+pd.Timedelta(seconds=args.max_runtime_seconds) if args.max_runtime_seconds else None
    prefix=f'[{backend.upper()} LIVE]'
    folder=ROOT/('fpga_ml' if backend=='fpga' else 'low_latency')
    cache=folder/'provisional_5m_history.csv'
    log=folder/('live_'+backend+'.csv')
    state_file=folder/('provisional_'+backend+'_state.json')
    print(f'\n{prefix} {DATA["ticker"]} | PROVISIONAL LIVE DECISIONS | Updates every 30 seconds',flush=True)
    if deadline is not None:
        print(f'{prefix} Output window: {launch_time.tz_convert("Asia/Kolkata")} to {deadline.tz_convert("Asia/Kolkata")}',flush=True)
    print(f'{prefix} Loading real five-minute history for indicators; no 50-new-candle wait.',flush=True)
    print(f'{prefix} Current candle is unfinished. Signals may change before close; provisional prediction quality is unvalidated.',flush=True)
    scaler=joblib.load(ROOT/'models/saved/scaler.pkl')
    model=joblib.load(ROOT/'models/saved/lgbm.pkl') if backend=='cpu' else None
    if model is not None:
        for _ in range(20):model.booster_.predict(np.zeros((1,27),dtype=np.float32),raw_score=True,num_threads=1)
    identity=hashlib.sha256((ROOT/'models/saved/lgbm.pkl').read_bytes()+(ROOT/'models/saved/scaler.pkl').read_bytes()).hexdigest()
    last_slot=None
    if state_file.exists():
        saved=json.loads(state_file.read_text())
        if saved.get('model_scaler_sha256')==identity:last_slot=pd.Timestamp(saved['last_update_slot'])
    if backend=='fpga':
        from serial.tools import list_ports
        if args.port.upper() not in {p.device.upper() for p in list_ports.comports()}:
            print(f'{prefix} {args.port} is unavailable. Connect the powered, ML-programmed ZCU104 and check its UART port; use --port if different.',flush=True)
            return 1
    state=ProvisionalHistory();source=yf.Ticker(DATA['ticker'])
    if cache.exists():
        saved=pd.read_csv(cache,index_col=0);saved.index=pd.to_datetime(saved.index,utc=True)
        state.refresh(saved,pd.Timestamp.now(tz='UTC'))
    history_ns=0
    while True:
        if args.max_runtime_seconds and time.monotonic()-start>=args.max_runtime_seconds:
            print(f'{prefix} Output window ended before history initialization completed.',flush=True)
            return 0
        try:
            begin=time.perf_counter_ns()
            state.refresh(source.history(period='1mo',interval='5m',auto_adjust=False,actions=False,timeout=15),pd.Timestamp.now(tz='UTC'))
            history_ns=time.perf_counter_ns()-begin
            state.history.index.name='time';state.history.to_csv(cache)
            if len(state.history)<50:raise ValueError('Insufficient Yahoo history to initialize the model features')
            break
        except Exception as exc:
            print(f'{prefix} History unavailable: {exc}; retrying.',flush=True)
            if args.max_runtime_seconds and time.monotonic()-start>=args.max_runtime_seconds:return 1
            time.sleep(5)
    print(f'{prefix} Indicators initialized from {len(state.history)} completed historical bars. Waiting for a fresh live price.',flush=True)
    events=queue.Queue(maxsize=8192);stop=threading.Event();overflow=threading.Event();active={'socket':None}
    class ShutdownFilter(logging.Filter):
        def filter(self,record):return not stop.is_set()
    log_filter=ShutdownFilter()
    def put(kind,value):
        try:events.put_nowait((kind,value,pd.Timestamp.now(tz='UTC')))
        except queue.Full:overflow.set()
    def worker():
        while not stop.is_set():
            try:
                with yf.WebSocket(verbose=False) as ws:
                    active['socket']=ws;ws.logger.addFilter(log_filter)
                    ws.subscribe([DATA['ticker']]);put('connected',None)
                    ws.listen(lambda msg:put('tick',msg))
                    if not stop.is_set():put('error','Stream disconnected; reconnecting')
            except Exception as exc:
                if not stop.is_set():put('error',str(exc))
            finally:
                if active['socket'] is not None:active['socket'].logger.removeFilter(log_filter)
                active['socket']=None
            stop.wait(5)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    refreshed_bucket=pd.Timestamp.now(tz='UTC').floor('5min');next_refresh_attempt=0
    count=0;results=[];port=None;last_status=None;last_notice=-30;connected=False;failed=False
    energy_meter=start_cpu_energy(args) if backend=='cpu' else None
    if energy_meter is not None:
        print(f'{prefix} CPU energy sensor: '+(energy_meter.reason or 'Windows RAPL package counter active'),flush=True)
    energy_measurement=None
    try:
        while not (args.max_runtime_seconds and time.monotonic()-start>=args.max_runtime_seconds):
            if overflow.is_set():raise RuntimeError('Stream queue overflow; stopping instead of silently discarding prices')
            try:kind,value,received=events.get(timeout=.5)
            except queue.Empty:kind=None
            if kind=='connected':
                connected=True;state.current=None;state.bucket=None;state.last_tick=None;state.seen=set()
                refreshed_bucket=None
                print(f'{prefix} Connected to Yahoo streaming prices.',flush=True)
            elif kind=='error':
                connected=False;state.current=None;state.bucket=None;state.last_tick=None
                print(f'{prefix} Feed: {value}',flush=True)
            elif kind=='tick' and connected:state.add(value,received)
            now=pd.Timestamp.now(tz='UTC');slot=now.floor('30s')
            if deadline is not None and now>=deadline:break
            due=last_slot is None or slot>last_slot
            fresh=state.last_tick is not None and (now-state.last_tick).total_seconds()<=15
            if connected and due and fresh and events.empty():
                bucket=now.floor('5min')
                if refreshed_bucket!=bucket:
                    if time.monotonic()<next_refresh_attempt:continue
                    try:
                        begin=time.perf_counter_ns()
                        state.refresh(source.history(period='5d',interval='5m',auto_adjust=False,actions=False,timeout=15),pd.Timestamp.now(tz='UTC'))
                        history_ns=time.perf_counter_ns()-begin
                        # Enrich observed partial high/low/open using Yahoo's current snapshot.
                        seed=state.seeds.get(state.bucket)
                        if seed is not None and state.current is not None:
                            state.current['open']=seed['open']
                            state.current['high']=max(state.current['high'],seed['high'])
                            state.current['low']=min(state.current['low'],seed['low'])
                        state.history.index.name='time';state.history.to_csv(cache)
                        refreshed_bucket=bucket
                    except Exception as exc:
                        print(f'{prefix} History refresh unavailable: {exc}',flush=True)
                        next_refresh_attempt=time.monotonic()+5;continue
                    # Consume queued ticks before choosing the live close after HTTP retrieval.
                    continue
                try:
                    frame=state.feature_frame(now)
                    feature_begin=time.perf_counter_ns()
                    with contextlib.redirect_stdout(io.StringIO()):features=build_features(frame,include_labels=False)
                    vector=features.loc[state.bucket,FEATURES['feature_cols']].to_numpy(dtype=np.float64)
                    feature_ns=time.perf_counter_ns()-feature_begin
                    if not np.isfinite(vector).all():raise ValueError('Non-finite provisional features')
                except (ValueError,KeyError) as exc:
                    status=str(exc)
                    if status!=last_status:print(f'{prefix} {status}',flush=True);last_status=status
                    # Retry official completed-bar download if it was not published yet.
                    if 'preceding completed' in status:
                        refreshed_bucket=None;next_refresh_attempt=time.monotonic()+5
                    continue
                if deadline is not None and pd.Timestamp.now(tz='UTC')>=deadline:break
                if backend=='fpga' and port is None:
                    import serial
                    port=serial.Serial(args.port,115200,timeout=3,write_timeout=3);port.reset_input_buffer()
                begin=time.perf_counter_ns();scaled=scale_features(scaler,vector);infer_begin=time.perf_counter_ns()
                if backend=='cpu':
                    scores=model.booster_.predict(scaled.reshape(1,-1),raw_score=True,num_threads=1)[0]
                    prediction=int(np.argmax(scores));end=time.perf_counter_ns()
                    extra={'latency_ns':end-begin,'inference_latency_ns':end-infer_begin}
                else:
                    from fpga_ml.live_fpga import request_prediction
                    prediction,cycles,raw_scores,roundtrip=request_prediction(port,scaled)
                    scores=np.asarray(raw_scores,dtype=float)/(1<<24)
                    extra={'fpga_cycles':cycles,'latency_ns':cycles*8,'uart_roundtrip_ns':roundtrip}
                if deadline is not None and pd.Timestamp.now(tz='UTC')>=deadline:break
                count+=1
                record={'source':'yahoo_history_plus_live_stream','decision_status':'provisional_unfinished_5m_candle','ticker':DATA['ticker'],'time':str(now.tz_convert('Asia/Kolkata')),'update_slot_ist':str(slot.tz_convert('Asia/Kolkata')),'bar_open_ist':str(state.bucket.tz_convert('Asia/Kolkata')),'bar_close_ist':str((state.bucket+pd.Timedelta(minutes=5)).tz_convert('Asia/Kolkata')),'last_tick_ist':str(state.last_tick.tz_convert('Asia/Kolkata')),'tick_age_seconds':(now-state.last_tick).total_seconds(),'open':state.current['open'],'high':state.current['high'],'low':state.current['low'],'close':state.current['close'],'score_buy':float(scores[0]),'score_hold':float(scores[1]),'score_sell':float(scores[2]),'signal':NAMES[prediction],'scaling_ns':infer_begin-begin,'feature_rebuild_batch_ns':feature_ns,'last_history_download_ns':history_ns,**extra}
                exists=log.exists() and log.stat().st_size>0
                with log.open('a',newline='',encoding='utf-8') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(record))
                    if not exists:writer.writeheader()
                    writer.writerow(record)
                temp=state_file.with_suffix('.tmp');temp.write_text(json.dumps({'model_scaler_sha256':identity,'last_update_slot':str(slot)},indent=2));temp.replace(state_file)
                last_slot=slot;results.append(record);last_status=None
                if count==1:
                    print(f'\n{prefix} Initial candle: {record["bar_open_ist"]} | Initial Open={record["open"]:.2f} | PROVISIONAL',flush=True)
                    print(format_live_row(record,backend,header=True),flush=True)
                print(format_live_row(record,backend),flush=True)
                if args.max_samples and count>=args.max_samples:break
            if time.monotonic()-start-last_notice>=30:
                if not fresh:print(f'{prefix} Waiting for a fresh live price; no stale-data prediction.',flush=True)
                last_notice=time.monotonic()-start
    except KeyboardInterrupt:print(f'\n{prefix} Stopped.')
    except (OSError,RuntimeError) as exc:
        failed=True
        print(f'{prefix} Inference stopped: {exc}',flush=True)
    finally:
        run_elapsed=time.monotonic()-start
        if energy_meter is not None:
            energy_measurement=energy_meter.stop(len(results),'Live event loop: includes feed waits, features, inference and logging; excludes initial history/model loading')
            save_cpu_energy(folder/'live_cpu_power.json',energy_measurement)
        stop.set()
        if active['socket'] is not None:
            with contextlib.suppress(Exception):active['socket'].close()
        thread.join(timeout=2)
        if port is not None:port.close()
    print(f'\n{prefix} PROVISIONAL LIVE LATENCY RESULTS\nSamples processed: {count}')
    if results:
        column='latency_ns';times=pd.Series([r[column] for r in results])
        print(f'Average Latency (ns): {times.mean():.2f}\nMinimum Latency (ns): {times.min():.2f}\nMaximum Latency (ns): {times.max():.2f}')
    print('\nTHROUGHPUT RESULTS')
    for line in throughput_summary(results,run_elapsed,backend):print(line)
    print()
    power_lines=power_summary(backend,len(results),run_elapsed,args,measurement=energy_measurement)
    for line in power_lines:print(line)
    summary_path=folder/f'live_{backend}_summary.txt'
    summary_path.write_text('\n'.join([f'{backend.upper()} LIVE RUN SUMMARY',
                                      f'Samples processed: {count}',
                                      *throughput_summary(results,run_elapsed,backend),
                                      *power_lines])+'\n',encoding='utf-8')
    print(f'{prefix} Run / throughput / power summary saved to: {summary_path}')
    if energy_measurement is not None:print(f'{prefix} Power results saved to: {folder / "live_cpu_power.json"}')
    print(f'{prefix} Results saved to: {log}')
    return 1 if failed else 0
