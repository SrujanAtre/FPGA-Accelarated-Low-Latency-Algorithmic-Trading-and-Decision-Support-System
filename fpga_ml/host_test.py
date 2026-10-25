"""Measure actual COM7 inference; no fabricated hardware results."""
from pathlib import Path
import time,struct,json
import numpy as np
import pandas as pd
import serial,joblib,sys
from preprocessing import encode_features,scale_features
ROOT=Path(__file__).resolve().parent
a=np.load(ROOT/'reports/test_inputs.npz')
rows=[]
project=ROOT.parent
sys.path.insert(0,str(project))
from config.config import FEATURES,SPLIT
scaler=joblib.load(project/'models/saved/scaler.pkl')
df=pd.read_csv(project/'data/processed/features.csv')
start=int(len(df)*SPLIT['train_frac'])+int(len(df)*SPLIT['val_frac'])
raw=df.iloc[start:][FEATURES['feature_cols']]
raw_values=raw.to_numpy(dtype=np.float64)
names=['BUY','HOLD','SELL']
market=pd.read_csv(project/'data/raw/nifty_1m.csv',index_col=0)
market.index=pd.to_datetime(market.index,utc=True)
timestamps=pd.to_datetime(df.iloc[start:,0],utc=True)
market_rows=market.loc[pd.DatetimeIndex(timestamps)]
print('\n[FPGA ML] Loading historical market data...')
print('\n[FULL TEST SET]')
print('Model: trained LightGBM | Clock: 125 MHz')
print(f'Filtered Rows: {len(raw_values)}')
print(f'From: {timestamps.iloc[0].tz_convert("Asia/Kolkata")}')
print(f'To  : {timestamps.iloc[-1].tz_convert("Asia/Kolkata")}')
print('\n[FPGA ML] Opening COM7 at 115200 baud...')
with serial.Serial('COM7',115200,timeout=3,write_timeout=3) as port:
    port.reset_input_buffer()
    print('[FPGA ML] Serial connection established.')
    print('\n[FPGA ML] Processing ALL held-out samples...\n')
    run_begin=time.perf_counter_ns()
    for i,q in enumerate(a['quantized']):
        total_begin=time.perf_counter_ns()
        scaled=scale_features(scaler,raw_values[i])
        encoded=encode_features(scaled)
        assert np.array_equal(encoded,q)
        packet=b'\xa5'+encoded.astype('<i4').tobytes()
        begin=time.perf_counter_ns();port.write(packet);port.flush();sent=time.perf_counter_ns()
        reply=port.read(30);end=time.perf_counter_ns()
        if len(reply)!=30 or reply[0]!=0x5a:
            raise RuntimeError(f'Invalid ML response at sample {i}: {reply.hex()}')
        pred=reply[1];cycles=struct.unpack_from('<I',reply,2)[0]
        scores=struct.unpack_from('<qqq',reply,6)
        if tuple(a['scores'][i])!=scores:raise RuntimeError(f'Score mismatch {i}')
        rows.append(dict(sample=i,software=int(a['software'][i]),fpga=pred,cycles=cycles,preprocessing_ns=begin-total_begin,input_write_ns=sent-begin,uart_roundtrip_ns=end-begin,end_to_end_ns=end-total_begin,score0=scores[0],score1=scores[1],score2=scores[2]))
        local_time=str(timestamps.iloc[i].tz_convert('Asia/Kolkata'))
        row=market_rows.iloc[i]
        match=pred==int(a['software'][i])
        rows[-1].update(time=local_time,high=float(row.high),low=float(row.low),close=float(row.close),fpga_latency_cycles=cycles,fpga_latency_ns=cycles*8,fpga_signal=names[pred],software_signal=names[int(a['software'][i])],prediction_match=match)
        print(f'[{i+1:03d}] {local_time} | Close={row.close:.2f} | '
              f'Latency={cycles} cycles ({cycles*8:.0f} ns / {cycles*.008:.3f} us) | '
              f'Signal={names[pred]}',flush=True)
    run_end=time.perf_counter_ns()
pd.DataFrame(rows).to_csv(ROOT/'reports/hardware_results.csv',index=False)
table=pd.DataFrame(rows)
output=ROOT/'reports/full_test_set_latency.csv'
display_cols=['time','high','low','close','fpga_latency_cycles','fpga_latency_ns','fpga_signal']
table[display_cols].to_csv(output,index=False)
summary={'samples':len(rows),'agreement':float(np.mean([r['software']==r['fpga'] for r in rows])),'wall_clock_elapsed_ns':run_end-run_begin,'actual_wall_clock_sustained_per_second':len(rows)*1e9/(run_end-run_begin),'reciprocal_summed_per_bar_latency_per_second':len(rows)/(sum(r['end_to_end_ns'] for r in rows)/1e9),'status':'Measured board run including loop overhead and host scaling/encoding; excludes initial CSV loading and full-history feature generation'}
(ROOT/'reports/hardware_pipeline_summary.json').write_text(json.dumps(summary,indent=2))
print('\n[FPGA ML] Serial connection closed.')
print('\n======================================')
print(' FPGA ML FULL TEST SET LATENCY RESULTS')
print('======================================')
print(f'Samples processed        : {len(table)}')
print(f'Average Latency (ns)     : {table.fpga_latency_ns.mean():.2f}')
print(f'Minimum Latency (ns)     : {table.fpga_latency_ns.min():.2f}')
print(f'Maximum Latency (ns)     : {table.fpga_latency_ns.max():.2f}')
print(f'Average Latency (cycles) : {table.cycles.mean():.2f}')
print(f'Average Latency (us)     : {table.fpga_latency_ns.mean()/1000:.3f}')
print(f'Classification accuracy  : {np.mean(table.fpga.values==a["labels"])*100:.2f}%')
print(f'Mean end-to-end time (ms): {table.end_to_end_ns.mean()/1e6:.3f}')
print(f'Sustained pipeline rate  : {summary["actual_wall_clock_sustained_per_second"]:.2f} inferences/s')
print('\nSignal Counts:')
print(table.fpga_signal.value_counts().to_string())
print('\nSample FPGA Results:')
print(table[display_cols].head().to_string(index=False,float_format=lambda v:f'{v:.2f}'))
print('\n[FPGA ML] Latency above is FPGA model inference; end-to-end includes host preprocessing and UART/USB.')
print('[FPGA ML] Sustained rate includes printing every sample in this run.')
print(f'[FPGA ML] Results saved to: {output}')
