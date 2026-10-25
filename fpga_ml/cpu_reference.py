"""Actual local-PC single-bar reference, not teammates' optimized CPU pipeline."""
from pathlib import Path
import time,json,platform
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent;project=ROOT.parent
model=joblib.load(project/'models/saved/lgbm.pkl')
a=np.load(ROOT/'reports/test_inputs.npz');x=a['features']
for _ in range(20):model.booster_.predict(x[:1],raw_score=True,num_threads=1)
rows=[]
for repeat in range(3):
 for i in range(len(x)):
  begin=time.perf_counter_ns()
  scores=model.booster_.predict(x[i:i+1],raw_score=True,num_threads=1)
  pred=int(np.argmax(scores));elapsed=time.perf_counter_ns()-begin
  assert pred==a['software'][i]
  rows.append(dict(repeat=repeat,sample=i,latency_ns=elapsed,prediction=pred))
pd.DataFrame(rows).to_csv(ROOT/'reports/local_cpu_reference.csv',index=False)
report={'description':'Newly measured local PC Python LightGBM Booster, batch=1, num_threads=1; includes Python prediction API and argmax overhead; excludes feature engineering, scaler, and UART; not teammates optimized CPU pipeline','platform':platform.platform(),'processor':platform.processor(),'samples':len(rows),'latency_ns':pd.Series([r['latency_ns'] for r in rows]).describe(percentiles=[.5,.95,.99]).to_dict(),'sustained_inferences_per_second_excluding_loop_overhead':len(rows)*1e9/sum(r['latency_ns'] for r in rows),'cpu_power':'unavailable'}
(ROOT/'reports/local_cpu_reference.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
