"""Short jury demonstration: three actual board inferences, one per class."""
from pathlib import Path
import json,struct
import numpy as np
import serial
import pandas as pd
ROOT=Path(__file__).resolve().parent
a=np.load(ROOT/'reports/test_inputs.npz');names=['BUY','HOLD','SELL'];results=[]
project=ROOT.parent
feature_rows=pd.read_csv(project/'data/processed/features.csv')
test_start=len(feature_rows)-len(a['software'])
market=pd.read_csv(project/'data/raw/nifty_1m.csv',index_col=0)
market.index=pd.to_datetime(market.index,utc=True)
print('\n[FPGA ML] Loading historical market data...')
print('[FPGA ML] Model: trained LightGBM | Clock: 125 MHz')
print('\n[FPGA ML] Opening COM7 at 115200 baud...')
with serial.Serial('COM7',115200,timeout=3,write_timeout=3) as port:
 port.reset_input_buffer()
 print('[FPGA ML] Serial connection established.')
 print('\n[FPGA ML] Processing BUY / HOLD / SELL demonstration samples...\n')
 for cls,name in enumerate(names):
  sample=int(np.flatnonzero(a['software']==cls)[0])
  port.write(b'\xa5'+a['quantized'][sample].astype('<i4').tobytes());port.flush()
  response=port.read(30)
  if len(response)!=30 or response[0]!=0x5a:raise RuntimeError('Invalid FPGA response')
  prediction=response[1];cycles=struct.unpack_from('<I',response,2)[0]
  scores=struct.unpack_from('<qqq',response,6)
  assert prediction==cls and tuple(scores)==tuple(a['scores'][sample])
  timestamp=pd.to_datetime(feature_rows.iloc[test_start+sample,0],utc=True)
  close=float(market.loc[timestamp,'close'])
  local_time=str(timestamp.tz_convert('Asia/Kolkata'))
  result={'historical_test_sample':sample,'time':local_time,'close':close,'software':name,'physical_fpga':names[prediction],'measured_cycles':cycles,'inference_us_at_125MHz':cycles*.008,'raw_class_scores':[v/16777216 for v in scores]}
  results.append(result)
  print(f'[{cls+1:03d}] {local_time} | Close={close:.2f} | '
        f'Latency={cycles} cycles ({cycles*8:.0f} ns / {cycles*.008:.3f} us) | '
        f'Signal={names[prediction]}',flush=True)
(ROOT/'reports/demo_results.json').write_text(json.dumps(results,indent=2))
table=pd.DataFrame([{'time':r['time'],'close':r['close'],
 'fpga_latency_cycles':r['measured_cycles'],
 'fpga_latency_ns':r['measured_cycles']*8,'fpga_signal':r['physical_fpga']} for r in results])
output=ROOT/'reports/demo_latency.csv'
table.to_csv(output,index=False)
print('\n[FPGA ML] Serial connection closed.')
print('\n======================================')
print(' FPGA ML LATENCY RESULTS')
print('======================================')
print(f'Samples processed        : {len(table)}')
print(f'Average Latency (ns)     : {table.fpga_latency_ns.mean():.2f}')
print(f'Minimum Latency (ns)     : {table.fpga_latency_ns.min():.2f}')
print(f'Maximum Latency (ns)     : {table.fpga_latency_ns.max():.2f}')
print(f'Average Latency (cycles) : {table.fpga_latency_cycles.mean():.2f}')
print('\nSignal Counts:')
print(table.fpga_signal.value_counts().to_string())
print('\nSample FPGA Results:')
print(table.to_string(index=False))
print('\n[FPGA ML] These are three selected class examples, not the full test-set benchmark.')
print(f'[FPGA ML] Results saved to: {output}')
