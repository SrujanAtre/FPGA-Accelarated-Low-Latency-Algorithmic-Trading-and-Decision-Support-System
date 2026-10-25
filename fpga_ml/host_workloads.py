"""Actual board workload replay with pacing and exact score checking."""
from pathlib import Path
import time,struct,json
import numpy as np
import pandas as pd
import serial
ROOT=Path(__file__).resolve().parent
a=np.load(ROOT/'reports/test_inputs.npz');rows=[];summaries=[]
with serial.Serial('COM7',115200,timeout=3,write_timeout=3) as port:
 port.reset_input_buffer()
 for name,n,period in [('low',16,.1),('medium',64,.04),('continuous',256,0)]:
  beginning=time.perf_counter_ns()
  for i in range(n):
   if period:
    remaining=beginning/1e9+i*period-time.perf_counter()
    if remaining>0:time.sleep(remaining)
   start=time.perf_counter_ns()
   port.write(b'\xa5'+a['quantized'][i].astype('<i4').tobytes());port.flush()
   reply=port.read(30);end=time.perf_counter_ns()
   if len(reply)!=30 or reply[0]!=0x5a:raise RuntimeError(f'{name} framing failure {i}')
   cycles=struct.unpack_from('<I',reply,2)[0];scores=struct.unpack_from('<qqq',reply,6)
   assert reply[1]==a['software'][i] and tuple(scores)==tuple(a['scores'][i])
   rows.append(dict(workload=name,sample=i,cycles=cycles,roundtrip_ns=end-start,start_ns=start,end_ns=end))
  elapsed=time.perf_counter_ns()-beginning
  g=pd.DataFrame([r for r in rows if r['workload']==name])
  summaries.append(dict(workload=name,samples=n,offered_rate=1/period if period else 'stop-and-wait saturation',elapsed_ns=elapsed,sustained_per_second=n*1e9/elapsed,roundtrip_ns=g.roundtrip_ns.describe(percentiles=[.5,.95,.99]).to_dict(),min_cycles=int(g.cycles.min()),max_cycles=int(g.cycles.max()),agreement=1.0))
  print(json.dumps(summaries[-1]),flush=True)
pd.DataFrame(rows).to_csv(ROOT/'reports/hardware_workloads.csv',index=False)
(ROOT/'reports/hardware_workloads.json').write_text(json.dumps(summaries,indent=2))
