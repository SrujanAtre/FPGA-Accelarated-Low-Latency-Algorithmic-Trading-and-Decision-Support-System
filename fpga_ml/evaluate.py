"""Consolidate evidence with explicit measurement provenance."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
audit=json.loads((ROOT/'reports/model_audit.json').read_text())
rtl=pd.read_csv(ROOT/'rtl/rtl_results.csv');inputs=np.load(ROOT/'reports/test_inputs.npz')
assert len(rtl)==audit['samples']
assert np.array_equal(rtl.prediction.values,inputs['software'])
assert np.array_equal(rtl[['score0','score1','score2']].values,inputs['scores'])
work=pd.read_csv(ROOT/'rtl/workload_sim_results.csv')
workloads=[]
for period,g in work.groupby('period_cycles',sort=False):
 workloads.append({'offered_per_second':125e6/period,'samples':len(g),'achieved_per_second':len(g)*125e6/(g.completion_cycle.iloc[-1]-g.accepted_cycle.iloc[0]),'min_latency_cycles':int(g.latency_cycles.min()),'max_latency_cycles':int(g.latency_cycles.max()),'max_external_queue_delay_us':float(g.queue_delay_cycles.max()*0.008),'status':'RTL simulation; pending requests held by testbench, not a hardware FIFO'})
summary={'validation':{'samples':len(rtl),'matching':len(rtl),'agreement_percent':100,'max_score_error':audit['max_raw_score_error'],'classification_accuracy':audit['accuracy'],'status':'RTL simulation compared with original saved LightGBM'},'inference':{'frequency_hz':125e6,'min_cycles':int(rtl.cycles.min()),'mean_cycles':float(rtl.cycles.mean()),'max_cycles':int(rtl.cycles.max()),'min_ns':int(rtl.cycles.min()*8),'mean_ns':float(rtl.cycles.mean()*8),'max_ns':int(rtl.cycles.max()*8),'status':'RTL simulation at specified clock; physical measurement unavailable until hardware test'},'uart_simulation':pd.read_csv(ROOT/'rtl/uart_sim_results.csv').to_dict('records'),'workload_simulation':workloads,'rom_bytes':len((ROOT/'rtl/nodes.mem').read_text().splitlines())*8,'hardware_results_available':(ROOT/'reports/hardware_results.csv').exists(),'artifact_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'rtl/nodes.mem',ROOT/'rtl/roots.mem',ROOT/'rtl/ml_core.v',ROOT/'rtl/fpga_ml_top.v']}}
(ROOT/'reports/performance_summary.json').write_text(json.dumps(summary,indent=2))
if (ROOT/'reports/hardware_results.csv').exists():
 hw=pd.read_csv(ROOT/'reports/hardware_results.csv')
 assert np.array_equal(hw.fpga.values,inputs['software'])
 assert np.array_equal(hw.cycles.values,rtl.cycles.values)
 assert np.array_equal(hw[['score0','score1','score2']].values,inputs['scores'])
 summary['physical']={'samples':len(hw),'matching':len(hw),'agreement_percent':100,'classification_accuracy':float(np.mean(hw.fpga.values==inputs['labels'])),'core_cycles':hw.cycles.describe(percentiles=[.5,.95,.99]).to_dict(),'host_stages_ns':hw[['preprocessing_ns','input_write_ns','uart_roundtrip_ns','end_to_end_ns']].describe(percentiles=[.5,.95,.99]).to_dict(),'sustained_per_second_excluding_between_iteration_overhead':len(hw)*1e9/hw.end_to_end_ns.sum(),'status':'Measured programmed ZCU104 over COM7; cycle counters match RTL; host end-to-end timing uses perf_counter_ns'}
 if (ROOT/'reports/hardware_workloads.json').exists():summary['physical_workloads']=json.loads((ROOT/'reports/hardware_workloads.json').read_text())
 if (ROOT/'reports/hardware_pipeline_summary.json').exists():summary['physical_pipeline']=json.loads((ROOT/'reports/hardware_pipeline_summary.json').read_text())
 summary['inference']['status']='Measured FPGA core cycle counters on 643 physical ZCU104 inferences; time calculated using implemented 125 MHz clock, identical to RTL simulation'
 (ROOT/'reports/performance_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
