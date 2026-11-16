"""Reproducible EEG-only pilot on the two real November 16 Rana recordings.

Run: python3 src/model/train_rana_new.py
Never reads the historical/public 'rana1' dataset. Test minutes are fixed before
fitting. Outputs are relative to the repository; raw inputs are never rewritten.
"""
from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfiltfilt, iirnotch, tf2sos, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
import sklearn, scipy, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/acquisition'))
from record_dataset import clock_map
SESSIONS=('rana_01_clean','rana_02')
OUT=ROOT/'data/model/rana_new'
REPORT=ROOT/'reports/rana_new_training'
SEED=20261116
# Minute numbers are zero based. Two-second purge at every minute boundary.
SPLITS={0:'train',1:'train',2:'validation',3:'train',4:'test',5:'train',6:'train',7:'validation',8:'train',9:'test'}
FEATURES=[]

def save_json(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def features(raw,low,broad):
 vals=[]; names=[]
 for label,x in [('low',low),('broad',broad)]:
  d=np.diff(x); sd=np.std(x)
  v=[np.mean(x),sd,np.ptp(x),np.min(x),np.max(x),np.sqrt(np.mean(x*x)),np.mean(abs(d)),np.std(d),skew(x) if sd>1e-8 else 0,kurtosis(x) if sd>1e-8 else 0,np.argmin(x)/len(x),np.argmax(x)/len(x)]
  n=['mean','std','ptp','min','max','rms','line_length','diff_std','skew','kurtosis','argmin','argmax']
  chunks=np.array_split(x,15)
  v += [np.mean(z) for z in chunks]; n += [f'bin{i}' for i in range(15)]
  vals.extend(v); names.extend([label+'_'+z for z in n])
 vals += [np.mean(raw),np.std(raw),np.ptp(raw),np.mean((raw<=1)|(raw>=1022))]
 names += ['raw_mean','raw_std','raw_ptp','raw_rail_fraction']
 global FEATURES
 FEATURES=names
 return np.nan_to_num(vals)

def prepare():
 rows=[]; X=[]; waves=[]; audits={}; all_events=[]; plots={}; sources={}
 for session in SESSIONS:
  folder=ROOT/'data/raw/datasets'/session
  sources[session]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
  meta=json.loads((folder/'session.json').read_text())
  eeg=pd.read_csv(folder/'eeg.csv'); frames=pd.read_csv(folder/'webcam_frames.csv'); anchors=pd.read_csv(folder/'eeg_clock.csv')
  events=pd.read_csv(folder/'eeg_events_auto.csv').sort_values('peak_frame').reset_index(drop=True)
  assert meta['eeg_sampling_rate_hz']==1000 and meta['eeg_channel']=='A4'
  idx=eeg.sample_index.to_numpy(); raw=eeg.adc.to_numpy(dtype=float)
  assert np.all(np.diff(idx)==1), 'Sample gap requires segment-specific processing'
  assert np.all((np.diff(eeg.sequence.to_numpy())%16)==1)
  scale,intercept,jitter,plausible=clock_map(anchors.to_dict('records'))
  assert plausible
  t=idx/1000*scale+intercept
  ft=frames.session_elapsed_s.to_numpy()
  assert np.all(np.diff(ft)>=0)
  eye=frames[['ear_left','ear_right']].to_numpy()
  eye[frames.face_found.to_numpy()!=1]=np.nan
  # Estimate open-eye reference only from training minutes.
  fm=np.floor(ft/60).astype(int)
  train_frames=np.isin(fm,[k for k,v in SPLITS.items() if v=='train'])
  threshold=float(np.nanpercentile(np.mean(eye[train_frames],axis=1),80)*meta['ear_threshold_ratio'])
  events['event_id']=np.arange(1,len(events)+1)
  events['condition']='unscripted_unknown'
  events['label_source']='mediapipe_ear_auto_unreviewed'
  events['original_protocol_section']=events.section
  events['section']='unscripted_unknown'; events['phase']='unknown'
  for key in ['onset','peak','offset']:
   inds=events[key+'_frame'].to_numpy(dtype=int)
   assert np.all((inds>=0)&(inds<len(frames)))
   times=ft[inds]
   j=np.clip(np.searchsorted(t,times),1,len(t)-1)
   j-= (abs(t[j-1]-times)<abs(t[j]-times)).astype(int)
   events[key+'_session_s']=times
   events[key+'_sample']=idx[j]
   events[key+'_eeg_s']=t[j]
  flags=events['flags'].fillna('').astype(str)
  reliable=~flags.str.contains('short_candidate|reopening_not_observed|recording_boundary')
  events['usable_auto_label']=reliable
  events['use_for_training']=False # set by explicit split/QC below, never inherited
  events['split']=[SPLITS.get(int(x//60),'excluded') for x in events.peak_session_s]
  events['within_split_interior']=[(x%60>=2.75 and x%60<=57.25) for x in events.peak_session_s]
  events['use_for_training']=(events.split=='train') & events.within_split_interior & reliable & (events.event_type=='blink')
  dest=OUT/session; dest.mkdir(parents=True,exist_ok=True)
  events.to_csv(dest/'aligned_events.csv',index=False)
  np.savez_compressed(dest/'aligned_eeg.npz',adc=raw.astype(np.int16),sample_index=idx,session_elapsed_s=t,nominal_hz=1000)
  all_events.append(events.assign(session=session))
  blink=events.loc[(events.event_type=='blink')&reliable,'peak_session_s'].to_numpy()
  excluded=events[(events.event_type!='blink')|~reliable]
  # Diagnostic zero-phase copy only; never supplied to the training models.
  diagnostic=sosfiltfilt(butter(3,[.5,12],fs=1000,btype='bandpass',output='sos'),raw)
  # Choose polarity on training labels only; map itself remains host-clock based.
  extrema={s:t[find_peaks(s*diagnostic,distance=200,prominence=8)[0]] for s in [-1,1]}
  train_blink=np.array([b for b in blink if SPLITS.get(int(b//60))=='train' and 3<b%60<57])
  def residuals(bp,pol):
   return np.array([extrema[pol][np.argmin(abs(extrema[pol]-b))]-b for b in bp])
  polarity=max([-1,1],key=lambda s:np.mean(abs(residuals(train_blink,s))<=.15))
  alignment=[]
  for _,e in events[events.event_type=='blink'].iterrows():
   b=e.peak_session_s; a=extrema[polarity]; near=a[abs(a-b)<=.15]
   alignment.append({'event_id':e.event_id,'split':e['split'],'peak_session_s':b,'nearest_extremum_residual_ms':float(residuals([b],polarity)[0]*1000),'extrema_within_150ms':len(near),'ambiguous':len(near)!=1})
  align=pd.DataFrame(alignment); align.to_csv(dest/'alignment_checks.csv',index=False)
  audits[session]={'source_metadata':meta,'rows':len(raw),'missing_samples':int(np.sum(np.diff(idx)-1)), 'adc_min':int(raw.min()),'adc_max':int(raw.max()),'rail_fraction':float(np.mean((raw<=1)|(raw>=1022))),'clock_scale':scale,'clock_intercept_s':intercept,'batch_residual_p95_ms':jitter*1000,'mapping':'session_s = original_sample_index / 1000 * clock_scale + clock_intercept_s','hardware_sync_verified':False,'mapping_status':'provisional; host receipt clocks, no hardware latency measurement','webcam_duplicate_times':int(np.sum(np.diff(ft)==0)),'webcam_frame_interval_median_ms':float(np.median(np.diff(ft))*1000),'threshold_for_no_blink':threshold,'blink_events':int((events.event_type=='blink').sum()),'sustained_closures':int((events.event_type=='sustained_closure').sum()),'alignment_polarity_selected_on_training':polarity,'alignment_checks':{}}
  for split,g in align.groupby('split'):
   r=g.nearest_extremum_residual_ms.to_numpy()
   audits[session]['alignment_checks'][split]={'n':len(g),'within_150ms':int(np.sum(abs(r)<=150)),'median_signed_ms':float(np.median(r)),'median_absolute_ms':float(np.median(abs(r))),'p95_absolute_ms':float(np.percentile(abs(r),95)),'ambiguous':int(g.ambiguous.sum())}
  plots[session]=(t[::20],diagnostic[::20],ft,np.mean(eye,axis=1),blink)
  # Independent filters in each minute avoid signal crossing split boundaries.
  for minute,split in SPLITS.items():
   a,b=np.searchsorted(t,[minute*60,(minute+1)*60]); local=raw[a:b]
   notch=tf2sos(*iirnotch(50,30,fs=1000))
   broad=sosfilt(butter(3,[.5,30],fs=1000,btype='bandpass',output='sos'),sosfilt(notch,local))
   low=sosfilt(butter(3,[.5,12],fs=1000,btype='bandpass',output='sos'),local)
   for center in np.arange(minute*60+2.8,(minute+1)*60-2.8,.1):
    left,right=np.searchsorted(t,[center-.75,center+.75]); u,v=left-a,right-a
    if u<0 or v>len(local) or right-left<1495: continue
    fa,fb=np.searchsorted(ft,[center-.3,center+.3])
    good=fb-fa>=10 and np.all(np.isfinite(eye[fa:fb])) and np.max(np.diff(ft[fa:fb]),initial=0)<.15
    overlap=any(e.onset_session_s-.25<=center<=e.offset_session_s+.25 for _,e in excluded.iterrows())
    if not good or overlap: continue
    dist=float(np.min(abs(blink-center))) if len(blink) else 999.
    y=int(dist<=.15)
    # Boundary examples omitted from clip metrics/training, but retained for continuous testing.
    clean=y==1 or (dist>=.45 and np.all(eye[fa:fb]>=threshold))
    clipped=float(np.mean((local[u:v]<=1)|(local[u:v]>=1022)))
    row={'session':session,'minute':minute,'split':split,'center_s':center,'start_s':float(t[left]),'end_s':float(t[right-1]),'start_sample':int(idx[left]),'end_sample':int(idx[right-1]),'label':y,'classification_eligible':clean,'rail_fraction':clipped,'condition':'unscripted_unknown'}
    rows.append(row); X.append(features(local[u:v],low[u:v],broad[u:v]))
    waves.append(np.interp(np.linspace(0,1,192),np.linspace(0,1,v-u),low[u:v]))
  print('Prepared',session,flush=True)
 df=pd.DataFrame(rows); X=np.array(X,dtype=np.float32)
 df.to_csv(OUT/'windows.csv',index=False)
 np.savez_compressed(OUT/'windows.npz',features=X,waveforms_128hz=np.array(waves,dtype=np.float32),labels=df.label.to_numpy(),feature_names=np.array(FEATURES))
 ev=pd.concat(all_events,ignore_index=True); ev.to_csv(OUT/'events.csv',index=False)
 save_json(REPORT/'data_quality_and_alignment.json',audits); save_json(REPORT/'input_sha256.json',sources)
 save_json(OUT/'split_manifest.json',{'seed':SEED,'sessions':SESSIONS,'minutes_zero_based':SPLITS,'purge_seconds':2,'window_seconds':1.5,'stride_seconds':.1,'center_tolerance_seconds':.15,'note':'Only these real sessions; protocol sections ignored. No prior/public recordings.'})
 return df,X,ev,plots

def metrics(y,p,threshold):
 pred=p>=threshold
 return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,p)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'n':len(y),'positive_windows':int(np.sum(y))}

def event_metrics(df,p,ev,threshold,return_predictions=False):
 detected=[]; tp=fp=fn=0; evaluated_seconds=0.; eligible_total=0
 for session,g in df.assign(probability=p).groupby('session'):
  for minute,block in g.groupby('minute'):
   times=block.center_s.to_numpy(); probs=block.probability.to_numpy()
   # Non-max suppression on a fixed time grid, no webcam blink times used to predict.
   candidates=np.flatnonzero(probs>=threshold)
   chosen=[]
   for i in candidates[np.argsort(probs[candidates])[::-1]]:
    if all(abs(times[i]-times[j])>=.35 for j in chosen): chosen.append(i)
   predictions=sorted(times[chosen]); evaluated_seconds+=len(times)*.1
   ref=ev[(ev.session==session)&(ev.event_type=='blink')&ev.usable_auto_label & (ev.peak_session_s>=minute*60+2.8)&(ev.peak_session_s<(minute+1)*60-2.8)].peak_session_s.to_numpy()
   # Score only references with usable face/closure coverage on the evaluation grid.
   ref=np.array([x for x in ref if np.min(abs(times-x))<=.15]); eligible_total+=len(ref)
   pairs=sorted((abs(x-y),i,j) for i,x in enumerate(predictions) for j,y in enumerate(ref) if abs(x-y)<=.25)
   usedp=set(); usedr=set()
   for _,i,j in pairs:
    if i not in usedp and j not in usedr: usedp.add(i); usedr.add(j)
   tp+=len(usedp); fp+=len(predictions)-len(usedp); fn+=len(ref)-len(usedr)
   detected.extend({'session':session,'minute':int(minute),'predicted_peak_s':x,'matched_auto_label':i in usedp} for i,x in enumerate(predictions))
 precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
 result={'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'tp':tp,'fp':fp,'fn':fn,'false_alarms_per_minute':fp/max(evaluated_seconds/60,1e-9),'evaluated_minutes':evaluated_seconds/60,'reference_events':eligible_total,'matching_tolerance_ms':250}
 return (result,detected) if return_predictions else result

def main():
 OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
 df,X,events,plots=prepare()
 tr=(df.split=='train')&df.classification_eligible
 rng=np.random.default_rng(SEED)
 pos=np.flatnonzero(tr&(df.label==1)); neg=np.flatnonzero(tr&(df.label==0))
 selected=np.sort(np.r_[pos,rng.choice(neg,min(len(neg),3*len(pos)),replace=False)])
 df['selected_for_fit']=False; df.loc[selected,'selected_for_fit']=True
 df.to_csv(OUT/'windows.csv',index=False)
 assert set(df.loc[selected,'split'])=={'train'}
 for (session,minute),g in df.groupby(['session','minute']): assert g.split.nunique()==1
 models={
  'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=SEED)),
  'svm_rbf':make_pipeline(StandardScaler(),SVC(C=3,gamma='scale',probability=False,random_state=SEED)),
  'extra_trees':ExtraTreesClassifier(n_estimators=300,min_samples_leaf=4,max_features=.7,n_jobs=4,random_state=SEED),
  'gradient_boosting':HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=SEED),
 }
 def probability(model,x):
  if hasattr(model,'predict_proba'): return model.predict_proba(x)[:,1]
  from scipy.special import expit
  return expit(model.decision_function(x)) # score monotonicity; not calibrated probability
 results=[]; fitted={}; thresholds={}; val=df.split=='validation'; test=df.split=='test'
 # All selection is completed on validation before test is evaluated.
 for name,model in models.items():
  model.fit(X[selected],df.label.to_numpy()[selected]); p=probability(model,X[val])
  trials=[(event_metrics(df[val],p,events,float(th)),float(th)) for th in np.linspace(.05,.95,37)]
  m,th=max(trials,key=lambda z:(z[0]['f1'],z[0]['precision']))
  results.append({'model':name,'threshold':th,'validation_event':m})
  fitted[name]=model; thresholds[name]=th
  print(name,'validation',m,flush=True)
 winner=max(results,key=lambda x:(x['validation_event']['f1'],x['validation_event']['precision']))['model']
 for r in results:
  name=r['model']; model=fitted[name]; th=r['threshold']; p=probability(model,X[test])
  eligible=df.loc[test,'classification_eligible'].to_numpy()
  r['test_window']=metrics(df.loc[test,'label'].to_numpy()[eligible],p[eligible],th)
  r['test_event'],pred=event_metrics(df[test],p,events,th,True)
  pd.DataFrame(pred).to_csv(REPORT/f'{name}_test_detected_events.csv',index=False)
  table=df[test].copy(); table['score']=p; table['predicted_positive']=p>=th
  table.to_csv(REPORT/f'{name}_test_window_predictions.csv',index=False)
  r['test_by_session']={}
  for s in SESSIONS:
   subset=df[test].session.to_numpy()==s
   r['test_by_session'][s]=event_metrics(df[test][subset],p[subset],events,th)
  r['test_by_time']={}
  for label,minutes in [('early',[4]),('late',[9])]:
   subset=df[test].minute.isin(minutes).to_numpy()
   r['test_by_time'][label]=event_metrics(df[test][subset],p[subset],events,th)
  joblib.dump({'model':model,'threshold':th,'features':FEATURES,'window_s':1.5,'lookahead_s':.75,'note':'offline pilot; score not necessarily calibrated; host-clock sync provisional'},REPORT/f'{name}.joblib')
  print(name,'TEST',r['test_event'],flush=True)
 summary={'selected_model_from_validation':winner,'seed':SEED,'training_windows':len(selected),'training_positive_windows':len(pos),'training_negative_windows':len(selected)-len(pos),'split_counts':df.groupby(['split','label']).size().to_dict(),'experiments':results,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'limitations':['Same-person held-out time blocks, not unseen people or sessions.','Automatic webcam labels are proxy truth; errors are unreviewed.','Clock alignment is provisional; one millisecond sample spacing is not synchronization accuracy.','Scripted protocol was not followed; no condition-specific labels inferred.','Sustained closures and tracking-invalid intervals are excluded; no separate closure classifier trained.','No test-driven model selection. Test reused across predeclared models only for descriptive reporting.','Centered 1.5-second windows require 750ms lookahead; not a validated live detector.']}
 summary['split_counts']={f'{a}/{b}':int(v) for (a,b),v in summary['split_counts'].items()}
 save_json(REPORT/'results.json',summary)
 records=[]
 for r in results:
  records.append({'model':r['model'],'selected':r['model']==winner,**{'window_'+k:v for k,v in r['test_window'].items() if isinstance(v,(int,float))},**{'event_'+k:v for k,v in r['test_event'].items()},'threshold':r['threshold'],'validation_event_f1':r['validation_event']['f1']})
 pd.DataFrame(records).to_csv(REPORT/'results.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 names=[r['model'] for r in results]; x=np.arange(len(names)); width=.25
 for j,key in enumerate(['precision','recall','f1']): axes[0].bar(x+(j-1)*width,[r['test_event'][key] for r in results],width,label=key)
 axes[0].set_xticks(x,names,rotation=15); axes[0].set_ylim(0,1); axes[0].set_title('Held-out event detection (automatic labels)'); axes[0].legend()
 chosen=next(r for r in results if r['model']==winner)
 cm=np.array(chosen['test_window']['confusion_matrix']); axes[1].imshow(cm,cmap='Blues')
 for i in range(2):
  for j in range(2): axes[1].text(j,i,str(cm[i,j]),ha='center',va='center')
 axes[1].set_xticks([0,1],['No blink','Blink']); axes[1].set_yticks([0,1],['No blink','Blink']); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Automatic reference'); axes[1].set_title(winner+' — eligible-window confusion matrix')
 fig.tight_layout(); fig.savefig(REPORT/'results.png',dpi=180); plt.close(fig)
 fig,axes=plt.subplots(4,1,figsize=(14,9))
 for j,s in enumerate(SESSIONS):
  t,x,ft,ear,blinks=plots[s]; axes[2*j].plot(t,x,lw=.35); axes[2*j+1].plot(ft,ear,lw=.4)
  for a in axes[2*j:2*j+2]:
   a.set_xlim(0,600)
   for b in blinks: a.axvline(b,color='red',alpha=.12,lw=.5)
  axes[2*j].set_title(s+' — diagnostic 0.5–12Hz EEG; red = automatic blink peaks')
  axes[2*j].set_ylabel('ADC (filtered)'); axes[2*j+1].set_ylabel('EAR'); axes[2*j+1].set_xlabel('Shared session seconds')
 fig.tight_layout(); fig.savefig(REPORT/'alignment_overview.png',dpi=160); plt.close(fig)
 write_report(summary,chosen)

def write_report(summary,chosen):
 lines=['# New Rana recordings: training pilot','',f"Selected using validation only: **{summary['selected_model_from_validation']}**.",'', 'Only `rana_01_clean` and `rana_02` recorded on November 16, 2026 are included. All older and public recordings are excluded, without deleting them. Raw source files remain unchanged.','', '## Split and task','', 'Each recording: training 0–120, 180–240, 300–420, 480–540 seconds; validation 120–180 and 420–480; test 240–300 and 540–600. Nominal 60/20/20; two-second guards plus half-window margins remove boundaries. Early and late are time ranges, not verified activity classes.','', 'Input: EEG A4 only, 1,000Hz ADC; independent causal filters per minute (0.5–12Hz and 0.5–30Hz with 50Hz notch), 1.5-second context every 100ms. Shape/amplitude/slope features. No EAR, timestamps, session identity, protocol or label metadata enters the model. Training keeps up to three negatives per positive. Validation/test retain their natural usable prevalence.','', 'Target: a blink peak within 150ms of the window center. Ambiguous near-blink windows are excluded from window classification metrics, but included in continuous event scoring. Sustained closures, incomplete labels and tracking-invalid centers are excluded. Clipped EEG windows are retained and flagged, so movement-related difficulty is not silently discarded. Event scoring uses nonmaximum suppression and one-to-one matching within 250ms.','', '## Results','', '| Model | Window accuracy | Event precision | Event recall | Event F1 | False alarms/min |','|---|---:|---:|---:|---:|---:|']
 for r in summary['experiments']:
  w=r['test_window']; e=r['test_event']; lines.append(f"| {r['model']} | {w['accuracy']:.1%} | {e['precision']:.1%} | {e['recall']:.1%} | {e['f1']:.1%} | {e['false_alarms_per_minute']:.2f} |")
 lines+=['','Accuracy counts correct eligible windows; event precision is how many detections match automatic labels; recall is how many labeled blinks are detected. F1 balances precision and recall. Window accuracy is easier than continuous detection and must not be presented as event accuracy.','', '## Limitations','']+['- '+x for x in summary['limitations']]
 lines+=['','The original session state says partial because Windows failed decoding the review HTML template. Both raw recordings span about 600 seconds. This pipeline does not overwrite that provenance, trusts actual raw counts, and reads the already-written automatic event boundaries by frame index. The recorder template now explicitly uses UTF-8.','', 'Clock diagnostics and extrema matching are in `data_quality_and_alignment.json` and each session’s `alignment_checks.csv`. Extrema coincidence is not proof of correct labeling; dense EEG extrema can match by chance. No blink-derived shift was fitted to the test data.','', 'Reproduce: `python3 src/model/train_rana_new.py` from the repo, using dependencies in `src/model/requirements.txt` plus pandas/matplotlib/joblib. Model artifacts are offline pilot models, not a final retrained deployment model.','', 'Validation practices: https://scikit-learn.org/stable/common_pitfalls.html and https://scikit-learn.org/stable/modules/cross_validation.html','', '![Results](results.png)','', '![Alignment overview](alignment_overview.png)']
 (REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
