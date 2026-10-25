from pathlib import Path
import sys,time,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
project=ROOT.parent;sys.path.insert(0,str(project))
from features.feature_engineering import build_features
from config.config import FEATURES
raw=pd.read_csv(project/'data/raw/nifty_1m.csv',index_col=0,parse_dates=True)
begin=time.perf_counter_ns();generated=build_features(raw);elapsed=time.perf_counter_ns()-begin
saved=pd.read_csv(project/'data/processed/features.csv',index_col=0,parse_dates=True)
cols=FEATURES['feature_cols']
assert len(generated)==len(saved)
error=np.max(np.abs(generated[cols].values-saved[cols].values),axis=0)
report={'raw_rows':len(raw),'feature_rows':len(generated),'median_bar_interval_seconds':float(raw.index.to_series().diff().dt.total_seconds().median()),'feature_generation_batch_ns':elapsed,'amortized_ns_per_raw_bar':elapsed/len(raw),'max_absolute_difference_by_feature':dict(zip(cols,error.tolist())),'labels_equal':bool(np.array_equal(generated.label.values,saved.label.values)),'status':'measured host batch feature generation; amortized time is not streaming latency'}
(ROOT/'reports/historical_preprocessing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
