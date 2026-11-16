"""Add real Noor4 recording, preserve baseline, test person transfer and pooled CNN."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/blinkit-mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_rana_new as base
from train_rana_cnn_xgb import BlinkCNN,normalize,predict_cnn,train_cnn,choose_threshold

ROOT=base.ROOT
OUT=ROOT/'reports/rana_noor4_cnn'
DATA=ROOT/'data/model/rana_noor4'
CONFIG={'width':16,'dropout':.2,'lr':.001}
SEEDS=[20261116,20261117]

def prepare():
 # Reuse the exact preparation pipeline, with outputs isolated from all earlier runs.
 base.SESSIONS=('noor4-clean',); base.OUT=ROOT/'data/model/noor4_new'; base.REPORT=ROOT/'reports/noor4_preparation'
 base.OUT.mkdir(parents=True,exist_ok=True); base.REPORT.mkdir(parents=True,exist_ok=True)
 nd,_,ne,_=base.prepare()
 rd=pd.read_csv(ROOT/'data/model/rana_new/windows.csv'); re=pd.read_csv(ROOT/'data/model/rana_new/events.csv')
 d=pd.concat([rd,nd],ignore_index=True); d['person']=np.where(d.session=='noor4-clean','Noor','Rana')
 waves=np.concatenate([np.load(ROOT/'data/model/rana_new/windows.npz')['waveforms_128hz'],np.load(base.OUT/'windows.npz')['waveforms_128hz']])
 events=pd.concat([re,ne],ignore_index=True)
 rng=np.random.default_rng(SEEDS[0]); d['selected_for_fit']=False
 for session,g in d.groupby('session'):
  pool=g[(g.split=='train')&g.classification_eligible]
  pos=pool.index[pool.label==1].to_numpy(); neg=pool.index[pool.label==0].to_numpy()
  fit=np.r_[pos,rng.choice(neg,min(len(neg),3*len(pos)),replace=False)]
  d.loc[fit,'selected_for_fit']=True
 DATA.mkdir(parents=True,exist_ok=True); d.to_csv(DATA/'windows.csv',index=False); events.to_csv(DATA/'events.csv',index=False)
 np.savez_compressed(DATA/'waveforms.npz',waveforms=waves,labels=d.label.to_numpy())
 assert (d.loc[d.selected_for_fit,'split']=='train').all()
 for _,g in d.groupby('session'):
  spans=g.groupby('minute').agg(start=('start_sample','min'),end=('end_sample','max')).sort_values('start')
  assert np.all(spans.start.to_numpy()[1:]>spans.end.to_numpy()[:-1])
 return d,waves,events

def evaluate(name,d,p,events,threshold):
 eligible=d.classification_eligible.to_numpy()
 result={'name':name,'threshold':float(threshold),'window':base.metrics(d.label.to_numpy()[eligible],p[eligible],threshold)}
 result['event'],det=base.event_metrics(d,p,events,threshold,True)
 result['by_person']={}
 for who,g in d.groupby('person'):
  mask=d.person.to_numpy()==who
  result['by_person'][who]=base.event_metrics(g,p[mask],events,threshold)
 table=d.copy(); table['score']=p; table.to_csv(OUT/f'{name}_predictions.csv',index=False)
 pd.DataFrame(det).to_csv(OUT/f'{name}_events.csv',index=False)
 print(name,result['event'],flush=True)
 return result

def fit_model(name,d,waves,events,source_person=None):
 pool=d.selected_for_fit.to_numpy().copy()
 if source_person: pool&=(d.person.to_numpy()==source_person)
 fit=np.flatnonzero(pool); vm=(d.split=='validation').to_numpy()
 if source_person: vm&=(d.person.to_numpy()==source_person)
 val=np.flatnonzero(vm); ve=np.flatnonzero(vm & d.classification_eligible.to_numpy())
 scale=float(np.std(waves[fit]-waves[fit].mean(axis=1,keepdims=True)))
 signal=normalize(waves,scale); members=[]; histories=[]; epochs=[]
 for seed in SEEDS:
  model,history,epoch=train_cnn(CONFIG,seed,signal,d.label.to_numpy(),fit,ve)
  members.append(model); histories.append(history); epochs.append(epoch)
  torch.save({'state_dict':model.state_dict(),'config':CONFIG,'scale':scale,'seed':seed,'best_epoch':epoch,'source_people':[source_person] if source_person else ['Rana','Noor']},OUT/f'{name}_{seed}.pt')
 p=np.mean([predict_cnn(m,signal[val]) for m in members],axis=0); threshold,met=choose_threshold(d.iloc[val],p,events)
 decision={'name':name,'source_people':[source_person] if source_person else ['Rana','Noor'],'config':CONFIG,'epochs':epochs,'scale':scale,'threshold':threshold,'validation_event':met,'fit_indices':fit.tolist(),'validation_indices':val.tolist()}
 base.save_json(OUT/f'{name}_frozen_selection.json',decision); base.save_json(OUT/f'{name}_history.json',histories)
 return members,scale,threshold,decision

def main():
 OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
 base.save_json(OUT/'plan.json',{'new_recording':'noor4-clean','source_archive_sha256':base.sha(Path.home()/'Downloads/noor4-clean.zip') if (Path.home()/'Downloads/noor4-clean.zip').exists() else None,'cnn_config':CONFIG,'seeds':SEEDS,'experiments':['frozen_Rana_to_Noor_all','frozen_Rana_to_Noor_test_blocks','pooled_Rana_Noor_test_blocks','Noor_to_Rana_all'],'split':base.SPLITS,'notes':['Use only two real Rana sessions plus new Noor4; no historical Noor or public data.','Raw timestamps aligned using same provisional clock mapping.','Protocol labels ignored as unverified; no activity-specific claims.','Rana results are reused-test; pooled held-out times are familiar-person tests.','Noor-only threshold and epoch selected using Noor validation only.','Current live model is not replaced.']})
 d,waves,events=prepare(); nm=d.person.to_numpy()=='Noor'; rm=~nm
 # Frozen existing CNN: do not tune its threshold, scaler, epochs, or weights on Noor.
 predictions=[]
 for seed in SEEDS:
  ck=torch.load(ROOT/f'reports/rana_cnn_xgb/cnn_01_{seed}.pt',map_location='cpu',weights_only=True)
  model=BlinkCNN(ck['config']['width'],ck['config']['dropout']); model.load_state_dict(ck['state_dict'])
  predictions.append(predict_cnn(model,normalize(waves[nm],ck['scale'])))
 frozen=np.mean(predictions,axis=0)
 results=[evaluate('frozen_Rana_to_Noor_all',d[nm],frozen,events,.8)]
 nt=d[nm].split.to_numpy()=='test'
 results.append(evaluate('frozen_Rana_to_Noor_test_blocks',d[nm][nt],frozen[nt],events,.8))
 base.save_json(OUT/'frozen_transfer_results.json',results)
 # Training uses no test windows, irrespective of frozen model diagnostics above.
 pooled,scale,threshold,decision=fit_model('pooled',d,waves,events)
 tm=d.split.to_numpy()=='test'; p=np.mean([predict_cnn(m,normalize(waves[tm],scale)) for m in pooled],axis=0)
 results.append(evaluate('pooled_Rana_Noor_test_blocks',d[tm],p,events,threshold))
 # Verify that serialized models reproduce the pooled predictions.
 restored=[]
 for seed in SEEDS:
  ck=torch.load(OUT/f'pooled_{seed}.pt',weights_only=True); m=BlinkCNN(16,.2); m.load_state_dict(ck['state_dict']); restored.append(predict_cnn(m,normalize(waves[tm],ck['scale'])))
 assert np.allclose(p,np.mean(restored,axis=0),atol=1e-6)
 noor,scale,threshold,ndecision=fit_model('noor_only',d,waves,events,'Noor')
 p=np.mean([predict_cnn(m,normalize(waves[rm],scale)) for m in noor],axis=0)
 results.append(evaluate('Noor_to_Rana_all',d[rm],p,events,threshold))
 summary={'experiments':results,'pooled_validation':decision['validation_event'],'noor_only_validation':ndecision['validation_event'],'pooled_threshold':decision['threshold'],'noor_only_threshold':ndecision['threshold'],'training_counts':d[d.selected_for_fit].groupby('person').size().to_dict(),'saved_model_check_passed':True,'live_model_replaced':False,'limitations':['Two people only; no third-person validation after pooled training.','Rana data already used in architecture development, so reverse transfer is exploratory.','Automatic labels and host-clock alignment remain provisional.','Same online deployment caveats as earlier CNN; offline event suppression.']}
 base.save_json(OUT/'results.json',summary)
 pd.DataFrame([{'experiment':r['name'],'window_accuracy':r['window']['accuracy'],**r['event']} for r in results]).to_csv(OUT/'results.csv',index=False)
 report(summary)

def report(summary):
 import matplotlib.pyplot as plt
 lines=['# Adding Noor4 to the real-data CNN','', 'Only `rana_01_clean`, `rana_02`, and the new `noor4-clean` recording are used. Historical Noor/Rahma and public recordings are excluded. Raw data and earlier results are preserved.','', 'Architecture is the previously chosen 1D CNN (width 16, dropout 0.2, learning rate 0.001), averaged across two seeds. Pooled and Noor-only models are trained afresh with early stopping on their source validation data. The same purged 60/20/20 minute split is retained. Training uses up to three eligible negatives per positive separately per session. Validation chooses epochs and threshold; the configuration is not searched again.','', '| Experiment | Window accuracy | Event precision | Event recall | Event F1 | False alarms/min |','|---|---:|---:|---:|---:|---:|']
 for r in summary['experiments']:
  e=r['event']; lines.append(f"| {r['name']} | {r['window']['accuracy']:.1%} | {e['precision']:.1%} | {e['recall']:.1%} | {e['f1']:.1%} | {e['false_alarms_per_minute']:.2f} |")
 lines+=['','## Pooled model per person','', '| Person | Precision | Recall | F1 |','|---|---:|---:|---:|']
 pooled=next(r for r in summary['experiments'] if r['name']=='pooled_Rana_Noor_test_blocks')
 for who,e in pooled['by_person'].items(): lines.append(f"| {who} | {e['precision']:.1%} | {e['recall']:.1%} | {e['f1']:.1%} |")
 lines+=['','## Interpretation','', 'Frozen Rana→Noor is the strongest unseen-person check here: no Noor blinks trained that model or selected its threshold. Compare its test-block subset to the pooled Noor test-block result to assess adaptation on the same interval. The pooled result is a familiar-person held-out-time result, NOT evidence of generalization to a third person. Noor→Rana is exploratory because earlier architecture development already used Rana. Different full-recording and test-block rows cover different events and should not be directly ranked.','', 'No source protocol activity labels are treated as true conditions. See `reports/noor4_preparation/data_quality_and_alignment.json` for clock/signal checks. The Windows partial-state export error is preserved in source metadata; actual sample and frame data are processed.','', 'Existing live detector weights are unchanged. New weights and frozen threshold files are in this report folder.','', 'Reproduce with the training environment and bundled Torch OpenMP library: `src/model/Run_Noor4_Training.command`.','', '![Results](comparison.png)']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 fig,ax=plt.subplots(figsize=(12,5)); rows=summary['experiments']; x=np.arange(len(rows)); width=.25
 for j,key in enumerate(['precision','recall','f1']): ax.bar(x+(j-1)*width,[r['event'][key] for r in rows],width,label=key)
 ax.set_xticks(x,['Rana → Noor (all)','Rana → Noor (test blocks)','Rana + Noor (test blocks)','Noor → Rana (all)'],rotation=12); ax.set_ylim(0,1); ax.legend(); ax.set_title('New-person transfer and pooled training — automatic labels'); fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)

if __name__=='__main__': main()
