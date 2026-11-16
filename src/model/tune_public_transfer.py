"""Public-data hyperparameter search, then fresh fitting on real Rana data only.

No public weights or examples are loaded into final estimators. Run after
train_rana_new.py. Existing test blocks have already been inspected: this is a
reused-test comparison, not a new confirmatory estimate.
"""
import json, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,iirnotch,tf2sos
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import joblib
import train_rana_new as base

ROOT=base.ROOT
OUT=ROOT/'reports/rana_public_hyperparameters'
DATA=ROOT/'data/model/public_ep10_tuning'
SEED=20261116

def configs():
 result=[]
 def add(family,params): result.append({'id':f'{family}_{len(result):02d}','family':family,'params':params})
 for c,gamma in itertools.product([.1,1,10,100],[.003,.03,.3]): add('svm',{'C':c,'gamma':gamma})
 for leaf,depth in itertools.product([2,8,16],[10,None]): add('extra_trees',{'min_samples_leaf':leaf,'max_depth':depth,'n_estimators':200,'max_features':.7})
 for leaf,reg,lr in itertools.product([7,15],[1,10],[.05,.1]): add('gradient_boosting',{'max_leaf_nodes':leaf,'l2_regularization':reg,'learning_rate':lr,'max_iter':200})
 for c in [.01,.1,1]: add('logistic',{'C':c})
 return result

def model_for(config):
 p=config['params']; name=config['family']
 if name=='svm': return make_pipeline(StandardScaler(),SVC(**p,probability=False,random_state=SEED))
 if name=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(**p,max_iter=2000,random_state=SEED))
 if name=='extra_trees': return ExtraTreesClassifier(**p,n_jobs=4,random_state=SEED)
 return HistGradientBoostingClassifier(**p,early_stopping=False,random_state=SEED)

def score(model,x):
 return model.predict_proba(x)[:,1] if hasattr(model,'predict_proba') else expit(model.decision_function(x))

def prepare_public():
 DATA.mkdir(parents=True,exist_ok=True)
 prov=json.loads((ROOT/'data/derived/rana/session1/provenance.json').read_text())
 for f in ['eye-events.tsv','eye-samples.tsv']:
  assert base.sha(ROOT/'data/raw/rana/session1'/f)==prov['source_files'][f]['sha256']
 raw=pd.read_csv(ROOT/'data/eeg/derived/rana_recording1.csv')['A3 raw (ADC)'].to_numpy(dtype=float)
 event=pd.read_csv(ROOT/'data/raw/rana/session1/eye-events.tsv',sep='\t',header=None,names=['onset','duration','type','value','sample'])
 blink=event[event.type=='blink'].copy(); peaks=(blink.onset+blink.duration/2).to_numpy()
 gaze=pd.read_csv(ROOT/'data/raw/rana/session1/eye-samples.tsv',sep='\t',header=None).to_numpy()
 valid=np.isfinite(gaze).all(axis=1)&(gaze[:,3]>0)&(gaze[:,1]>0)&(gaze[:,2]>0)
 rows=[]; features=[]
 for minute in range(6):
  a=minute*60000; b=min((minute+1)*60000,len(raw)); x=raw[a:b]
  if len(x)<6000: continue
  broad=sosfilt(butter(3,[.5,30],fs=1000,btype='bandpass',output='sos'),sosfilt(tf2sos(*iirnotch(50,30,fs=1000)),x))
  low=sosfilt(butter(3,[.5,12],fs=1000,btype='bandpass',output='sos'),x)
  for center in np.arange(a/1000+2.8,b/1000-2.8,.1):
   distance=float(np.min(abs(peaks-center))); y=int(distance<=.15)
   # Independent gaze validity only for backgrounds. Blinks cause pupil loss.
   ga,gb=np.searchsorted(gaze[:,0],[center-.3,center+.3])
   background=distance>=.45 and gb>ga and valid[ga:gb].all()
   if not (y or background): continue
   u=int(round((center-.75)*1000))-a; v=u+1500
   if v>len(x): continue
   rows.append({'source':'EEGEyeNet_EP10','minute':minute,'fold':minute%3,'center_s':center,'label':y,'start_sample':a+u,'end_sample':a+v-1})
   features.append(base.features(x[u:v],low[u:v],broad[u:v]))
 df=pd.DataFrame(rows); X=np.array(features,dtype=np.float32)
 df.to_csv(DATA/'windows.csv',index=False); np.savez_compressed(DATA/'features.npz',features=X,labels=df.label.to_numpy(),feature_names=np.array(base.FEATURES))
 base.save_json(DATA/'provenance.json',{'source':prov,'used_for':'hyperparameter selection only, not final model fitting','target':'eye-tracker interval midpoint; NOT earlier EEG-derived peak','raw_derived_sha256':base.sha(ROOT/'data/eeg/derived/rana_recording1.csv'),'public_blinks':len(blink),'folds':'minute modulo 3, two-second plus half-window guards; filters independent per minute','rows':len(df),'positives':int(df.label.sum())})
 return df,X

def balanced_indices(y,pool,seed):
 rng=np.random.default_rng(seed); pos=pool[y[pool]==1]; neg=pool[y[pool]==0]
 return np.sort(np.r_[pos,rng.choice(neg,min(len(neg),3*len(pos)),replace=False)])

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=configs()
 base.save_json(OUT/'search_plan.json',{'seed':SEED,'configs':cfg,'public_objective':'mean average precision over 3 held-out time-block folds','shortlist':'top two configs per family by public CV average precision','rana_objective':'event F1 on existing validation blocks; threshold grid 0.05:0.025:0.95','test_policy':'evaluate once after validation selection; test previously exposed in baseline','public_data_not_in_final_fit':True,'timing':'unchanged provisional host-clock alignment; no tuning on test blinks'})
 pub,PX=prepare_public(); y=pub.label.to_numpy(); records=[]
 for i,c in enumerate(cfg):
  fold=[]
  for k in range(3):
   train=balanced_indices(y,np.flatnonzero(pub.fold.to_numpy()!=k),SEED+k)
   valid=np.flatnonzero(pub.fold.to_numpy()==k)
   assert len(set(y[train]))==2 and len(set(y[valid]))==2
   model=model_for(c); model.fit(PX[train],y[train]); p=score(model,PX[valid])
   fold.append(float(average_precision_score(y[valid],p)))
  r={**c,'public_fold_ap':fold,'public_mean_ap':float(np.mean(fold)),'public_std_ap':float(np.std(fold))}; records.append(r)
  base.save_json(OUT/'public_search.json',records)
  print(f"Public search {i+1}/{len(cfg)} {c['id']} AP={r['public_mean_ap']:.3f}",flush=True)
 shortlist=[]
 for family in sorted({c['family'] for c in cfg}): shortlist.extend(sorted([r for r in records if r['family']==family],key=lambda z:z['public_mean_ap'],reverse=True)[:2])
 base.save_json(OUT/'shortlist.json',shortlist)
 d=pd.read_csv(base.OUT/'windows.csv'); archive=np.load(base.OUT/'windows.npz'); X=archive['features']; y=d.label.to_numpy()
 assert set(d.session)==set(base.SESSIONS)
 assert X.shape[1]==PX.shape[1]
 assert np.array_equal(archive['feature_names'],np.load(DATA/'features.npz')['feature_names'])
 fit=np.flatnonzero(d.selected_for_fit); assert (d.iloc[fit].split=='train').all()
 ev=pd.read_csv(base.OUT/'events.csv'); validation=d.split=='validation'; test=d.split=='test'
 candidates=[]; fitted={}
 for c in shortlist:
  # Fresh constructor: the public fitted estimators are deliberately not reused.
  model=model_for(c); model.fit(X[fit],y[fit]); p=score(model,X[validation])
  trials=[(base.event_metrics(d[validation],p,ev,float(th)),float(th)) for th in np.linspace(.05,.95,37)]
  m,threshold=max(trials,key=lambda z:(z[0]['f1'],z[0]['precision']))
  r={**c,'threshold':threshold,'validation_event':m}; candidates.append(r); fitted[c['id']]=model
  print(c['id'],'Rana validation F1',m['f1'],flush=True)
 winner=max(candidates,key=lambda r:(r['validation_event']['f1'],r['validation_event']['precision']))['id']
 # Persist selection BEFORE touching test features/predictions.
 base.save_json(OUT/'frozen_selection.json',{'winner':winner,'candidates':candidates,'final_fit_sessions':list(base.SESSIONS),'training_rows':len(fit),'training_windows_sha256':base.sha(base.OUT/'windows.csv'),'test_used_for_selection':False})
 for r in candidates:
  name=r['id']; model=fitted[name]; p=score(model,X[test]); th=r['threshold']
  keep=d.loc[test,'classification_eligible'].to_numpy()
  r['test_window']=base.metrics(y[test][keep],p[keep],th)
  r['test_event'],detected=base.event_metrics(d[test],p,ev,th,True)
  table=d[test].copy(); table['score']=p; table.to_csv(OUT/f'{name}_test_windows.csv',index=False)
  pd.DataFrame(detected).to_csv(OUT/f'{name}_test_events.csv',index=False)
  artifact={'model':model,'threshold':th,'config':r['params'],'features':archive['feature_names'].tolist(),'source':'fresh fit on real Rana training rows only; settings shortlisted using public EP10','window_s':1.5,'lookahead_s':.75,'nms_seconds':.35,'sync_status':'provisional'}
  joblib.dump(artifact,OUT/f'{name}.joblib')
  if name==winner: joblib.dump(artifact,OUT/'selected_model.joblib')
  restored=joblib.load(OUT/f'{name}.joblib'); assert np.allclose(score(restored['model'],X[test]),p)
  print(name,'TEST',r['test_event'],flush=True)
 results={'selected_model':winner,'public_configurations':len(cfg),'public_fits':len(cfg)*3,'fresh_rana_fits':len(candidates),'experiments':candidates,'public_data_used_for_final_training':False,'test_is_reused':True}
 base.save_json(OUT/'results.json',results)
 pd.DataFrame([{'id':r['id'],'family':r['family'],'parameters':json.dumps(r['params']),'public_cv_ap':r['public_mean_ap'],'validation_event_f1':r['validation_event']['f1'],'window_accuracy':r['test_window']['accuracy'],**{'event_'+k:v for k,v in r['test_event'].items()},'selected':r['id']==winner} for r in candidates]).to_csv(OUT/'results.csv',index=False)
 render(results)

def render(result):
 import matplotlib.pyplot as plt
 rs=result['experiments']; winner=next(r for r in rs if r['id']==result['selected_model'])
 fig,ax=plt.subplots(figsize=(11,5)); xs=np.arange(len(rs)); width=.25
 for i,key in enumerate(['precision','recall','f1']): ax.bar(xs+(i-1)*width,[r['test_event'][key] for r in rs],width,label=key)
 ax.set_xticks(xs,[r['id']+('*' if r['id']==winner['id'] else '') for r in rs],rotation=25,ha='right'); ax.set_ylim(0,1); ax.legend(); ax.set_title('Public-tuned settings, fresh Rana-only models — reused test\n* selected by validation, not by test'); fig.tight_layout(); fig.savefig(OUT/'results.png',dpi=180); plt.close(fig)
 lines=['# Public-data hyperparameters → fresh Rana-only training','',f"Validation-selected model: **{winner['id']}**.",'', '29 configurations across SVM, extra trees, gradient boosting and logistic regression were compared using three public time-block folds (87 public fits). The top two per family were instantiated afresh and fitted only on the existing real Rana training split (8 fresh fits). Validation event F1 selected thresholds and the final model. Public trained weights were not transferred.','', 'Public source: EEGEyeNet EP10, OpenNeuro ds007338 v1.0.0, https://openneuro.org/datasets/ds007338/versions/1.0.0 . The historical folder alias `rana1` is an external public import, NOT a recording of Rana. EEG E22 originally 500Hz, resampled to 1000Hz and mapped to ADC in the prior import. Target times here are independent eye-tracker blink interval midpoints, not EEG-derived peaks.','', 'The public set has only 68 blink annotations from one recording, so the search provides limited evidence about transferable settings. “Best” means best among this explicit search, not a global optimum.','', '## Frozen selection','', '```json',json.dumps({'hyperparameters':winner['params'],'threshold':winner['threshold']},indent=2),'```','', '## Reused-test comparison','', '| Configuration | Public CV AP | Validation event F1 | Test window accuracy | Event precision | Event recall | Event F1 |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rs: lines.append(f"| {r['id']} | {r['public_mean_ap']:.1%} | {r['validation_event']['f1']:.1%} | {r['test_window']['accuracy']:.1%} | {r['test_event']['precision']:.1%} | {r['test_event']['recall']:.1%} | {r['test_event']['f1']:.1%} |")
 lines+=['','## Interpretation','', 'The model settings and thresholds were frozen before this run evaluated test predictions. However, these same test blocks were already viewed during the baseline experiment, so this is not an untouched confirmatory test. Do not choose a different winner by inspecting this test table. A new recording is needed for a fresh final evaluation.','', 'The 60/20/20 time-block split, automatic labels, guard gaps, causal filtering, features, 350ms suppression, 250ms event matching and provisional host-clock synchronization are unchanged from the baseline. All final models use only real Rana recordings. The 1.5-second centered window requires 750ms lookahead. Classifier inputs are EEG-only; face-validity and closure exclusions limit the evaluated intervals.','', 'No additional synchronization correction was applied in this experiment. The known uncertainty in labels and timing remains. Sustained closures are excluded rather than silently converted to no-blink.','', 'No guarantee of improvement: compare the validation-selected model to the original SVM, not whichever test row looks best. Baseline SVM: event precision 82.4%, recall 56.3%, F1 66.9%, window accuracy 77.2%.','', '## Reproduce','', '`python3 src/model/tune_public_transfer.py` after `python3 src/model/train_rana_new.py`; dependencies: `src/model/requirements-rana-new.txt`. All outputs remain local.','', '![Results](results.png)']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
