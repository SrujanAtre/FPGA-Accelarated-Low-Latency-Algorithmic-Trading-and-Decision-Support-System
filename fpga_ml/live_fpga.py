"""Default: seeded five-minute features -> provisional FPGA decisions every 30 seconds.
Use --five-minute for the original verified candle pipeline.

Run from fpga_ml: python live_fpga.py
Stop with Ctrl+C. --once processes the latest fresh completed bar and exits.
--replay-file explicitly tests the same pipeline with an offline OHLCV CSV.
"""
from pathlib import Path
import argparse,contextlib,csv,hashlib,io,json,struct,time,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib,serial,yfinance as yf
from config.config import DATA,FEATURES
from features.feature_engineering import build_features
from fpga_ml.preprocessing import scale_features,encode_features

COLS=['open','high','low','close','volume']
NAMES=['BUY','HOLD','SELL']
CACHE=HERE/'live_market_cache.csv'
STATE=HERE/'live_fpga_state.json'
LOG=HERE/'live_fpga_results.csv'

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

def request_prediction(port,scaled):
    packet=b'\xa5'+encode_features(scaled).astype('<i4').tobytes()
    begin=time.perf_counter_ns();port.write(packet);port.flush()
    response=port.read(30);elapsed=time.perf_counter_ns()-begin
    if len(response)!=30 or response[0]!=0x5a or response[1]>2:
        raise RuntimeError(f'Invalid FPGA ML response: received {len(response)}/30 bytes, prefix={response[:8].hex() or "empty"}. Reprogram fpga_ml_top.bit after board power-up or an interrupted UART packet, and close other programs using the port.')
    cycles=struct.unpack_from('<I',response,2)[0]
    scores=struct.unpack_from('<qqq',response,6)
    if response[1]!=int(np.argmax(scores)):raise RuntimeError('FPGA class/score response is inconsistent')
    return response[1],cycles,scores,elapsed

def print_latency_summary(results):
    print('\n======================================')
    print(' FPGA ML LIVE MARKET LATENCY RESULTS')
    print('======================================')
    print(f'Samples processed        : {len(results)}')
    if not results:
        print('No new fresh completed bars were processed in this run.')
        return
    table=pd.DataFrame(results)
    print(f'Average Latency (ns)     : {table.fpga_latency_ns.mean():.2f}')
    print(f'Minimum Latency (ns)     : {table.fpga_latency_ns.min():.2f}')
    print(f'Maximum Latency (ns)     : {table.fpga_latency_ns.max():.2f}')
    print(f'Average Latency (cycles) : {table.fpga_latency_cycles.mean():.2f}')
    print(f'Average Latency (us)     : {table.fpga_latency_ns.mean()/1000:.3f}')
    print('\nSignal Counts:')
    print(table.fpga_signal.value_counts().to_string())
    print('\nSample FPGA Results:')
    print(table.head().to_string(index=False,float_format=lambda v:f'{v:.2f}'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--max-age-minutes',type=float,default=15)
    parser.add_argument('--port',default='COM7')
    parser.add_argument('--max-polls',type=int,default=0,help='Stop after N polls; 0 runs until Ctrl+C')
    parser.add_argument('--replay-file',type=Path)
    parser.add_argument('--replay-samples',type=int,default=3)
    args=parser.parse_args()
    if args.poll_seconds<10 or args.max_age_minutes<=0 or args.replay_samples<1 or args.max_polls<0:
        parser.error('Poll interval must be >=10 seconds; age and replay sample count must be positive')
    ticker=DATA['ticker']
    model_hash=hashlib.sha256((ROOT/'models/saved/lgbm.pkl').read_bytes()).hexdigest()
    scaler=joblib.load(ROOT/'models/saved/scaler.pkl')
    scaler_hash=hashlib.sha256((ROOT/'models/saved/scaler.pkl').read_bytes()).hexdigest()
    identity={'ticker':ticker,'model_sha256':model_hash,'scaler_sha256':scaler_hash}
    last=None
    if STATE.exists() and not args.replay_file:
        state=json.loads(STATE.read_text())
        if all(state.get(k)==v for k,v in identity.items()):last=pd.Timestamp(state['last_bar_utc'])
    history=None
    if CACHE.exists() and not args.replay_file:
        history=clean_bars(pd.read_csv(CACHE,index_col=0))
    source=yf.Ticker(ticker)
    print(f'\n[FPGA LIVE] {ticker} | Completed 5-minute bars | Trained LightGBM | 125 MHz')
    print('[FPGA LIVE] Mode: '+('OFFLINE REPLAY' if args.replay_file else 'YAHOO CURRENT MARKET DATA'))
    print('[FPGA LIVE] Press Ctrl+C to stop. Waiting/stale bars do not generate signals.\n')
    count=0;first_fetch=True;port=None;polls=0;results=[];last_status=None
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
                print(f'[FPGA LIVE] Data unavailable: {exc}',flush=True)
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
                if port is None:
                    port=serial.Serial(args.port,115200,timeout=3,write_timeout=3)
                    port.reset_input_buffer()
                begin=time.perf_counter_ns()
                scaled=scale_features(scaler,row[FEATURES['feature_cols']].to_numpy(dtype=np.float64))
                preprocessing_ns=time.perf_counter_ns()-begin
                prediction,cycles,scores,roundtrip_ns=request_prediction(port,scaled)
                count+=1;processed=True
                record={'source':'offline_replay' if args.replay_file else 'yahoo','ticker':ticker,'bar_open_ist':str(stamp.tz_convert('Asia/Kolkata')),'bar_close_ist':str(close_time.tz_convert('Asia/Kolkata')),'observed_at_utc':str(now),'high':float(history.loc[stamp,'high']),'low':float(history.loc[stamp,'low']),'close':float(history.loc[stamp,'close']),'signal':NAMES[prediction],'fpga_cycles':cycles,'fpga_inference_ns':cycles*8,'input_bar_age_seconds':age,'download_ns':download_ns,'feature_rebuild_batch_ns':feature_ns,'scaling_ns':preprocessing_ns,'uart_roundtrip_ns':roundtrip_ns,'score_buy':scores[0],'score_hold':scores[1],'score_sell':scores[2]}
                results.append({'time':record['bar_open_ist'],'high':record.pop('high'),'low':record.pop('low'),'close':record['close'],'fpga_latency_cycles':cycles,'fpga_latency_ns':cycles*8,'fpga_signal':NAMES[prediction]})
                exists=LOG.exists() and LOG.stat().st_size>0
                with LOG.open('a',newline='',encoding='utf-8') as log:
                    writer=csv.DictWriter(log,fieldnames=list(record))
                    if not exists:writer.writeheader()
                    writer.writerow(record)
                if not args.replay_file:
                    state={**identity,'last_bar_utc':str(stamp)}
                    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)
                    last=stamp
                print(f'[{count:03d}] {record["bar_open_ist"]} | Close={record["close"]:.2f} | Latency={cycles} cycles ({cycles*8} ns / {cycles*.008:.3f} us) | Signal={NAMES[prediction]}',flush=True)
            if not processed:
                latest=history.index[-1].tz_convert('Asia/Kolkata')
                status=f'[FPGA LIVE] Waiting for a new fresh completed 5-minute bar. Latest available: {latest}'
                if status!=last_status:
                    print(status,flush=True)
                    last_status=status
            else:
                last_status=None
            if args.once or args.replay_file or (args.max_polls and polls>=args.max_polls):break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print('\n[FPGA LIVE] Stopped.')
    finally:
        if port is not None:port.close()
    print_latency_summary(results)
    print(f'[FPGA LIVE] Results saved to: {LOG}')
    return 0

if __name__=='__main__':
    if '--five-minute' in sys.argv or '--replay-file' in sys.argv:
        if '--five-minute' in sys.argv:sys.argv.remove('--five-minute')
        raise SystemExit(main())
    from fpga_ml.streaming_30s import run_stream
    raise SystemExit(run_stream('fpga'))
