"""CNN/XGBoost comparison on real Rana sessions; validation selection, reused test.
On macOS run src/model/Run_Rana_CNN_XGB.command (sets bundled OpenMP path).
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/blinkit-mpl')
import copy, itertools, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import average_precision_score
import train_rana_new as base

OUT=base.ROOT/'reports/rana_cnn_xgb'
SEED=20261116

class BlinkCNN(nn.Module):
 def __init__(self,width=16,dropout=.2):
  super().__init__()
  self.layers=nn.Sequential(nn.Conv1d(2,width,9,padding=4),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(width,width*2,7,padding=3),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(width*2,width*2,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(12),nn.Flatten(),nn.Linear(width*2*12,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
 def forward(self,x): return self.layers(x).squeeze(1)

@torch.no_grad()
def predict_cnn(model,x):
 model.eval()
 return np.concatenate([model(chunk).sigmoid().numpy() for chunk in x.split(512)])

def normalize(wave,scale):
 centered=wave-np.mean(wave,axis=1,keepdims=True)
 return torch.from_numpy(np.stack([centered/scale,centered/np.maximum(np.std(centered,axis=1,keepdims=True),scale*.05)],axis=1).astype(np.float32))

def choose_threshold(d,p,events):
 trials=[(base.event_metrics(d,p,events,float(t)),float(t)) for t in np.linspace(.05,.95,37)]
 m,t=max(trials,key=lambda r:(r[0]['f1'],r[0]['precision']))
 return t,m

def train_cnn(config,seed,signal,y,fit,valid):
 torch.manual_seed(seed); random.seed(seed); rng=np.random.default_rng(seed)
 model=BlinkCNN(config['width'],config['dropout'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.001)
 lossfn=nn.BCEWithLogitsLoss()
 target=torch.from_numpy(y.astype(np.float32)); best=-1.; state=None; patience=0; history=[]
 for epoch in range(1,51):
  model.train(); total=0.
  for ix in np.array_split(rng.permutation(fit),int(np.ceil(len(fit)/128))):
   optimizer.zero_grad(); loss=lossfn(model(signal[ix]),target[ix]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=loss.item()*len(ix)
  p=predict_cnn(model,signal[valid]); ap=float(average_precision_score(y[valid],p))
  history.append({'epoch':epoch,'train_loss':total/len(fit),'validation_ap':ap})
  if ap>best+1e-4: best=ap; state=copy.deepcopy(model.state_dict()); patience=0; best_epoch=epoch
  else: patience+=1
  if epoch%10==0: print('CNN',config,'seed',seed,'epoch',epoch,'validation AP',round(ap,3),flush=True)
  if patience>=8: break
 model.load_state_dict(state); model.eval()
 return model,history,best_epoch

def main():
 OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
 xconfigs=[{'max_depth':depth,'min_child_weight':child,'learning_rate':lr} for depth,child,lr in itertools.product([2,3,5],[1,5],[.03,.1])]
 cconfigs=[{'width':8,'dropout':.2,'lr':.001},{'width':16,'dropout':.2,'lr':.001},{'width':16,'dropout':.35,'lr':.003},{'width':32,'dropout':.35,'lr':.001}]
 base.save_json(OUT/'plan.json',{'xgboost_configurations':xconfigs,'cnn_configurations':cconfigs,'cnn_seeds':[SEED,SEED+1],'cnn_max_epochs':50,'cnn_patience':8,'cnn_selection':'average precision for epoch; two-seed mean score per configuration; event F1 for config/threshold','xgboost_max_rounds':600,'xgboost_patience':40,'xgboost_selection':'validation aucpr for tree count; event F1 for config/threshold','postprocessing':'same 350ms suppression and 250ms event matching as baseline','ensemble_xgb_weights':[.25,.5,.75],'test':'already viewed; reused test; never early stopping or model selection','alignment':'unchanged provisional shared host clock','public_data_used':False})
 d=pd.read_csv(base.OUT/'windows.csv'); a=np.load(base.OUT/'windows.npz'); X=a['features']; waves=a['waveforms_128hz']; y=d.label.to_numpy(); events=pd.read_csv(base.OUT/'events.csv')
 assert set(d.session)==set(base.SESSIONS) and np.isfinite(X).all() and np.isfinite(waves).all()
 fit=np.flatnonzero(d.selected_for_fit); val=np.flatnonzero(d.split=='validation'); eligible_val=np.flatnonzero((d.split=='validation')&d.classification_eligible); test=np.flatnonzero(d.split=='test')
 assert (d.iloc[fit].split=='train').all()
 assert not set(fit)&set(test)
 scale=float(np.std(waves[fit]-np.mean(waves[fit],axis=1,keepdims=True))); signal=normalize(waves,scale)
 base.save_json(OUT/'input_manifest.json',{'windows_sha256':base.sha(base.OUT/'windows.csv'),'arrays_sha256':base.sha(base.OUT/'windows.npz'),'fit_rows':fit.tolist(),'normalization_scale_training_only':scale,'features':a['feature_names'].tolist(),'versions':{'torch':torch.__version__,'xgboost':xgboost.__version__}})
 xrows=[]; xmodels={}; xval={}
 for i,c in enumerate(xconfigs):
  name=f'xgb_{i:02d}'
  m=XGBClassifier(**c,n_estimators=600,early_stopping_rounds=40,eval_metric='aucpr',tree_method='hist',subsample=.8,colsample_bytree=.8,reg_lambda=5,n_jobs=2,random_state=SEED)
  m.fit(X[fit],y[fit],eval_set=[(X[eligible_val],y[eligible_val])],verbose=False)
  p=m.predict_proba(X[val])[:,1]; t,met=choose_threshold(d.iloc[val],p,events)
  r={'name':name,'family':'xgboost','config':c,'best_iteration':int(m.best_iteration),'threshold':t,'validation_event':met}
  m.save_model(OUT/f'{name}.ubj'); xrows.append(r); xmodels[name]=m; xval[name]=p
  base.save_json(OUT/'xgboost_validation.json',xrows)
  print(name,'validation event F1',round(met['f1'],4),'trees',m.best_iteration+1,flush=True)
 bestx=max(xrows,key=lambda r:(r['validation_event']['f1'],r['validation_event']['precision']))
 crows=[]; cmodels={}; cval={}
 for i,c in enumerate(cconfigs):
  name=f'cnn_{i:02d}'; members=[]; histories=[]; best_epochs=[]
  for seed in [SEED,SEED+1]:
   m,h,epoch=train_cnn(c,seed,signal,y,fit,eligible_val); members.append(m); histories.append(h); best_epochs.append(epoch)
   torch.save({'state_dict':m.state_dict(),'config':c,'scale':scale,'best_epoch':epoch,'seed':seed},OUT/f'{name}_{seed}.pt')
  p=np.mean([predict_cnn(m,signal[val]) for m in members],axis=0); t,met=choose_threshold(d.iloc[val],p,events)
  r={'name':name,'family':'cnn','config':c,'best_epochs':best_epochs,'threshold':t,'validation_event':met}
  crows.append(r); cmodels[name]=members; cval[name]=p
  base.save_json(OUT/f'{name}_training_history.json',histories); base.save_json(OUT/'cnn_validation.json',crows)
  print(name,'validation event F1',round(met['f1'],4),flush=True)
 bestc=max(crows,key=lambda r:(r['validation_event']['f1'],r['validation_event']['precision']))
 ensemble=[]
 for weight in [.25,.5,.75]:
  p=weight*xval[bestx['name']]+(1-weight)*cval[bestc['name']]; t,met=choose_threshold(d.iloc[val],p,events)
  ensemble.append({'name':f'ensemble_xgb_{weight}','family':'ensemble','xgb_weight':weight,'threshold':t,'validation_event':met,'members':[bestx['name'],bestc['name']]})
 beste=max(ensemble,key=lambda r:(r['validation_event']['f1'],r['validation_event']['precision']))
 finalists=[bestx,bestc,beste]; winner=max(finalists,key=lambda r:(r['validation_event']['f1'],r['validation_event']['precision']))['name']
 base.save_json(OUT/'frozen_selection.json',{'winner':winner,'finalists':finalists,'ensemble_validation':ensemble,'test_not_used_for_selection':True})
 # Test prediction happens only after all selection is frozen above.
 px=xmodels[bestx['name']].predict_proba(X[test])[:,1]
 pc=np.mean([predict_cnn(m,signal[test]) for m in cmodels[bestc['name']]],axis=0)
 scores={bestx['name']:px,bestc['name']:pc,beste['name']:beste['xgb_weight']*px+(1-beste['xgb_weight'])*pc}
 # Verify saved artifacts reproduce scores.
 xm=XGBClassifier(); xm.load_model(OUT/f"{bestx['name']}.ubj"); assert np.allclose(px,xm.predict_proba(X[test])[:,1])
 restored=[]
 for seed in [SEED,SEED+1]:
  ck=torch.load(OUT/f"{bestc['name']}_{seed}.pt",weights_only=True); m=BlinkCNN(ck['config']['width'],ck['config']['dropout']); m.load_state_dict(ck['state_dict']); restored.append(predict_cnn(m,normalize(waves[test],ck['scale'])))
 assert np.allclose(pc,np.mean(restored,axis=0),atol=1e-6)
 for r in finalists:
  p=scores[r['name']]; t=r['threshold']; keep=d.iloc[test].classification_eligible.to_numpy()
  r['test_window']=base.metrics(y[test][keep],p[keep],t)
  r['test_event'],pred=base.event_metrics(d.iloc[test],p,events,t,True)
  table=d.iloc[test].copy(); table['score']=p; table['predicted_positive']=p>=t; table.to_csv(OUT/f"{r['name']}_test_windows.csv",index=False)
  pd.DataFrame(pred).to_csv(OUT/f"{r['name']}_test_events.csv",index=False)
  r['test_by_session']={}
  for session in base.SESSIONS:
   ix=d.iloc[test].session.to_numpy()==session; r['test_by_session'][session]=base.event_metrics(d.iloc[test][ix],p[ix],events,t)
  print(r['name'],'TEST',r['test_event'],flush=True)
 result={'winner_by_validation':winner,'xgboost_fits':12,'cnn_fits':8,'cnn_configurations':4,'ensemble_weight_trials':3,'public_data_used':False,'test_is_reused':True,'experiments':finalists,'saved_prediction_verification_passed':True}
 base.save_json(OUT/'results.json',result)
 pd.DataFrame([{'model':r['name'],'validation_f1':r['validation_event']['f1'],'window_accuracy':r['test_window']['accuracy'],**r['test_event']} for r in finalists]).to_csv(OUT/'results.csv',index=False)
 report(result)

def report(result):
 import matplotlib.pyplot as plt
 old=json.loads((base.REPORT/'results.json').read_text()); baseline=next(r for r in old['experiments'] if r['model']==old['selected_model_from_validation'])
 rows=[{'name':'baseline SVM',**baseline}]+result['experiments']
 fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(rows)); width=.25
 for j,key in enumerate(['precision','recall','f1']): ax.bar(x+(j-1)*width,[r['test_event'][key] for r in rows],width,label=key)
 ax.set_xticks(x,[r['name'] for r in rows]); ax.set_ylim(0,1); ax.legend(); ax.set_title('EEG-only blink detection — same reused test blocks'); fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
 lines=['# CNN and XGBoost on the two real Rana recordings','',f"Validation-selected winner: **{result['winner_by_validation']}**.",'', 'Twelve XGBoost hyperparameter configurations; four CNN configurations each trained with two seeds (eight CNN runs). CNN seeds are averaged per configuration rather than selecting a lucky seed. Three averaging weights combine the two validation-selected family models. No public data or pretrained weights were used.','', 'XGBoost uses the same 58 EEG features as the baseline. CNN receives the saved 192-point, low-pass-filtered waveform per 1.5-second window as two channels: training-scale normalization and within-window normalization. Three convolution blocks retain coarse position with adaptive pooling and a small classification head. Training uses AdamW and binary cross entropy; epoch selection uses validation average precision with early stopping.','', 'All models use the existing 60/20/20 time-block split and original selected training rows. Normalization is fitted on training only. XGBoost tree count and CNN epoch count use validation only. Detection thresholds and ensemble weights maximize validation event F1. Test predictions are produced after frozen_selection.json is written. Saved weights were reloaded and reproduced test scores.','', '| Model | Validation event F1 | Test window accuracy | Event precision | Event recall | Event F1 | False alarms/min |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  w=r['test_window']; e=r['test_event']; lines.append(f"| {r['name']} | {r['validation_event']['f1']:.1%} | {w['accuracy']:.1%} | {e['precision']:.1%} | {e['recall']:.1%} | {e['f1']:.1%} | {e['false_alarms_per_minute']:.2f} |")
 lines+=['','## Limits','', 'These test blocks were already inspected in earlier experiments: this is a reused-test development comparison, not an untouched confirmatory estimate. Automatic webcam labels remain unreviewed. Synchronization is the unchanged provisional host-clock map; no blink-delay correction was introduced. Sustained closures and invalid tracking are excluded from scoring. Clipping is retained and flagged. Only same-person time-block generalization is assessed. Centered windows require 750ms lookahead; this is not yet a validated real-time detector.','', '## Reproduce','', '`src/model/Run_Rana_CNN_XGB.command` on macOS (or `python src/model/train_rana_cnn_xgb.py` on other platforms) after generating `data/model/rana_new` with `train_rana_new.py`. Requirements are in `src/model/requirements-rana-cnn-xgb.txt`. Model files, validation searches, training histories, scores, and frozen decisions are in this folder. Raw files and baseline results are unchanged; nothing was pushed.','', '![Comparison](comparison.png)']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
