"""Evaluate recent-data LightGBM candidates without changing the deployed model.
Time-ordered train/validation/test with lookahead purges and a frozen test selection.
"""
from pathlib import Path
import sys,contextlib,io,json,hashlib
import joblib,numpy as np,pandas as pd,yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from config.config import FEATURES,LABELS,DATA
from features.feature_engineering import build_features
from fpga_ml.live_fpga import clean_bars
from fpga_ml.preprocessing import scale_features

def metrics(y,p):
    return {'samples':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),
    'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),'prediction_counts':{name:int((p==i).sum()) for i,name in enumerate(['BUY','HOLD','SELL'])},
    'per_class':classification_report(y,p,labels=[0,1,2],target_names=['BUY','HOLD','SELL'],output_dict=True,zero_division=0)}

def main():
    active=ROOT/'models/saved/lgbm.pkl';scaler_path=ROOT/'models/saved/scaler.pkl'
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [active,scaler_path]]
    raw=clean_bars(yf.Ticker(DATA['ticker']).history(period='1mo',interval='5m',auto_adjust=False,actions=False,timeout=20))
    raw=raw[raw.index+pd.Timedelta(minutes=5)<=pd.Timestamp.now(tz='UTC')]
    raw.to_csv(ROOT/'training/recent_model_market_data.csv')
    with contextlib.redirect_stdout(io.StringIO()):frame=build_features(raw)
    cols=FEATURES['feature_cols'];n=len(frame);a=int(n*.65);b=int(n*.82);gap=LABELS['lookahead_bars']
    train=frame.iloc[:a-gap];val=frame.iloc[a:b-gap];test=frame.iloc[b:]
    if min(len(train),len(val),len(test))<100:raise RuntimeError('Insufficient recent labeled bars for an independent assessment')
    assert (train.label.value_counts().reindex([0,1,2],fill_value=0)>0).all()
    scaler=RobustScaler().fit(train[cols].to_numpy(dtype=np.float64))
    def matrix(part):return np.stack([scale_features(scaler,row) for row in part[cols].to_numpy(dtype=np.float64)])
    xtrain=matrix(train);xval=matrix(val);xtest=matrix(test)
    ytrain=train.label.to_numpy(dtype=int);yval=val.label.to_numpy(dtype=int);ytest=test.label.to_numpy(dtype=int)
    candidates=[];best=None
    # All candidates have at most 72 trees and bounded tree depth for the existing ROM core.
    for depth,leaves in [(3,7),(4,15),(5,31)]:
        for weight in [None,'balanced']:
            for lr in [.05,.15]:
                cfg=dict(n_estimators=24,max_depth=depth,num_leaves=leaves,learning_rate=lr,class_weight=weight,min_child_samples=25,reg_lambda=2.,verbosity=-1,n_jobs=1,random_state=42,deterministic=True,force_col_wise=True)
                model=LGBMClassifier(**cfg).fit(xtrain,ytrain)
                pred=model.booster_.predict(xval,raw_score=True,num_threads=1).argmax(axis=1)
                m=metrics(yval,pred);entry={'parameters':cfg,'validation':m};candidates.append(entry)
                rank=(m['macro_f1'],m['accuracy'])
                if best is None or rank>best[0]:best=(rank,model,entry)
    # Model selection ends here: no candidate is selected or tuned using test outcomes.
    model=best[1]
    prediction=model.booster_.predict(xtest,raw_score=True,num_threads=1).argmax(axis=1)
    baseline=joblib.load(active);old_scaler=joblib.load(scaler_path)
    base_x=np.stack([scale_features(old_scaler,row) for row in test[cols].to_numpy(dtype=np.float64)])
    base_pred=baseline.booster_.predict(base_x,raw_score=True,num_threads=1).argmax(axis=1)
    result=metrics(ytest,prediction);baseline_result=metrics(ytest,base_pred)
    dump=model.booster_.dump_model()
    def count(node):return 1 if 'leaf_value' in node else 1+count(node['left_child'])+count(node['right_child'])
    nodes=sum(count(tree['tree_structure']) for tree in dump['tree_info'])
    improves=result['accuracy']>baseline_result['accuracy'] and result['macro_f1']>baseline_result['macro_f1']
    majority=int(train.label.value_counts().idxmax());majority_result=metrics(ytest,np.full(len(test),majority))
    report={'status':'candidate evaluated, NOT activated on CPU or FPGA','input_interval_seconds':300,'live_provisional_validation':'not established; completed-bar data only',
    'data_source':'fresh Yahoo five-minute completed OHLCV','raw_bars':len(raw),'labeled_rows':n,'data_from':str(raw.index[0]),'data_to':str(raw.index[-1]),
    'split':{'train':[str(train.index[0]),str(train.index[-1]),len(train)],'validation':[str(val.index[0]),str(val.index[-1]),len(val)],'test':[str(test.index[0]),str(test.index[-1]),len(test)],'purged_bars_per_boundary':gap},
    'selection_rule':'highest validation macro F1, accuracy tie-break; frozen before test evaluation','selected_parameters':best[2]['parameters'],'candidates':candidates,
    'candidate_test':result,'deployed_model_same_test':baseline_result,'training_majority_baseline_test':majority_result,'improves_accuracy_and_macro_f1_on_same_test':bool(improves),
    'hardware_shape':{'trees':len(dump['tree_info']),'nodes':nodes,'rom_8192_node_capacity_ok':nodes<8192},'active_model_sha256':before[0],'active_scaler_sha256':before[1],
    'limitations':['Single recent time holdout, not proof of future profitability or accuracy','Recent data may already have been inspected during live development; this is not a pristine external audit','No labeled provisional-candle training snapshots are available','Any candidate deployment requires FPGA re-export, rebuild, fixed-point validation and physical matching before switching both pipelines']}
    joblib.dump(model,ROOT/'models/saved/lgbm_candidate.pkl');joblib.dump(scaler,ROOT/'models/saved/scaler_candidate.pkl')
    (ROOT/'training/model_improvement_report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame({'time':test.index.astype(str),'label':ytest,'deployed_prediction':base_pred,'candidate_prediction':prediction}).to_csv(ROOT/'training/model_candidate_test_results.csv',index=False)
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in [active,scaler_path]]
    print(json.dumps({k:report[k] for k in ['status','candidate_test','deployed_model_same_test','improves_accuracy_and_macro_f1_on_same_test','hardware_shape']},indent=2))
    return report

if __name__=='__main__':main()
