"""Export the existing trained LightGBM; never train or overwrite artifacts."""
from pathlib import Path
import sys, json, hashlib, collections
import numpy as np
import pandas as pd
import joblib
import onnx
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from config.config import FEATURES, SPLIT
cols = FEATURES['feature_cols']
model = joblib.load(PROJECT/'models/saved/lgbm.pkl')
scaler = joblib.load(PROJECT/'models/saved/scaler.pkl')
dump = model.booster_.dump_model()
assert model.n_features_in_ == 27 and list(model.classes_) == [0,1,2]
nodes, roots, depths = [], [], []
def key(v):
    if v==0:return 0
    bits=int(np.asarray(np.float32(v)).view(np.uint32))
    return bits if bits<0x80000000 else (~(bits&0x7fffffff))
def visit(n, depth=0):
    idx=len(nodes); nodes.append(None); depths.append(depth)
    if 'leaf_value' in n:
        v=int(np.rint(n['leaf_value']*16777216))
        assert -(1<<31)<=v<(1<<31)
        nodes[idx]=(1<<63)|(v & 0xffffffff)
    else:
        assert n['decision_type']=='<=' and n['missing_type'] in ('None','NaN')
        left=visit(n['left_child'],depth+1); right=visit(n['right_child'],depth+1)
        threshold=np.float32(n['threshold'])
        if float(threshold)>n['threshold']:
            threshold=np.nextafter(threshold,np.float32(-np.inf))
        v=key(threshold)
        assert -(1<<31)<=v<(1<<31)
        nodes[idx]=(n['split_feature']<<58)|(left<<45)|(right<<32)|(v&0xffffffff)
    return idx
for t in dump['tree_info']: roots.append(visit(t['tree_structure']))
assert len(nodes)<8192
for name, vals, width in [('nodes',nodes,16),('roots',roots,4)]:
    (ROOT/'rtl'/f'{name}.mem').write_text(''.join(f'{v:0{width}x}\n' for v in vals))
df=pd.read_csv(PROJECT/'data/processed/features.csv')
start=int(len(df)*SPLIT['train_frac'])+int(len(df)*SPLIT['val_frac'])
test=df.iloc[start:]
x=scaler.transform(test[cols]).astype(np.float32)
assert np.isfinite(x).all()
q=np.array([[key(v) for v in row] for row in x],dtype=np.int64)
assert ((q>=-(1<<31))&(q<(1<<31))).all(), 'Feature overflow: choose wider representation'
def predict(row):
    scores=np.zeros(3,dtype=np.int64); visits=0
    for tree, root in enumerate(roots):
        addr=root
        while True:
            visits+=1; n=nodes[addr]; value=n&0xffffffff
            if value>=1<<31: value-=1<<32
            if n>>63: scores[tree%3]+=value; break
            feature=(n>>58)&31
            addr=((n>>45)&8191) if row[feature]<=value else ((n>>32)&8191)
    return scores,visits
results=[predict(r) for r in q]
scores=np.array([r[0] for r in results]); pred=scores.argmax(1)
software=model.predict(x); raw=model.booster_.predict(x,raw_score=True)
pd.DataFrame({'software':software,'fixed':pred,'label':test.label.astype(int).values,'node_visits':[r[1] for r in results]}).to_csv(ROOT/'reports/validation.csv',index=False)
(ROOT/'rtl/vectors.mem').write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for row in q for v in row))
(ROOT/'rtl/expected.mem').write_text(''.join(f'{int(v):x}\n' for v in pred))
np.savez(ROOT/'reports/test_inputs.npz',features=x,quantized=q,labels=test.label.values,software=software,scores=scores)
graphs={}
for name in ['lstm','cnn']:
    g=onnx.load(str(PROJECT/f'models/saved/{name}.onnx'))
    onnx.checker.check_model(g)
    graphs[name]={'operators':dict(collections.Counter(n.op_type for n in g.graph.node)), 'inputs':[[d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim] for t in g.graph.input], 'outputs':[[d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim] for t in g.graph.output], 'initializer_elements':sum(int(np.prod(t.dims)) for t in g.graph.initializer)}
conf=np.zeros((3,3),int)
for a,b in zip(test.label.astype(int),pred): conf[a,b]+=1
report={'model_sha256':hashlib.sha256((PROJECT/'models/saved/lgbm.pkl').read_bytes()).hexdigest(),'trees':len(roots),'nodes':len(nodes),'max_depth':max(depths),'features':cols,'samples':len(x),'matching':int((software==pred).sum()),'agreement_percent':float((software==pred).mean()*100),'accuracy':float((test.label.values==pred).mean()),'confusion_matrix':conf.tolist(),'max_raw_score_error':float(np.max(abs(raw-scores/16777216))),'max_feature_quantization_error':0.0,'onnx':graphs,'scaler':type(scaler).__name__,'data_first':str(test.iloc[0,0]),'data_last':str(test.iloc[-1,0]),'status':'software fixed-point representation validation, not RTL or board validation'}
(ROOT/'reports/model_audit.json').write_text(json.dumps(report,indent=2))
(ROOT/'rtl/model_config.vh').write_text(f'`define N_TREES {len(roots)}\n`define N_NODES {len(nodes)}\n`define N_SAMPLES {len(x)}\n')
print(json.dumps(report,indent=2))


