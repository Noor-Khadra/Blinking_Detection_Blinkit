"""Verify saved artifacts and report baseline/QC strata without changing models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from train_rana_new import ROOT, OUT, REPORT, metrics

def main():
 d=pd.read_csv(OUT/'windows.csv'); arr=np.load(OUT/'windows.npz'); x=arr['features']
 assert len(x)==len(d) and np.isfinite(x).all()
 assert set(d.session)=={'rana_01_clean','rana_02'}
 assert (d.loc[d.selected_for_fit,'split']=='train').all()
 # Assert raw sample intervals do not overlap between different splits.
 for _,g in d.groupby('session'):
  spans=g.groupby(['minute','split']).agg(start=('start_sample','min'),end=('end_sample','max')).reset_index().sort_values('start')
  assert np.all(spans.start.to_numpy()[1:]>spans.end.to_numpy()[:-1])
 r=json.loads((REPORT/'results.json').read_text()); audit={'checks_passed':['finite inputs','new real sessions only','training selection excludes validation/test','no raw sample overlap across minute blocks','saved model predictions reproduce output'], 'models':{}}
 mask=d.split=='test'; y=d.loc[mask,'label'].to_numpy(); eligible=d.loc[mask,'classification_eligible'].to_numpy()
 audit['always_no_blink_window_accuracy']=float(np.mean(y[eligible]==0))
 audit['window_counts']=d.groupby('split').size().to_dict()
 audit['eligible_test_windows']=int(eligible.sum())
 for result in r['experiments']:
  name=result['model']; saved=joblib.load(REPORT/f'{name}.joblib'); model=saved['model']
  p=model.predict_proba(x[mask])[:,1] if hasattr(model,'predict_proba') else expit(model.decision_function(x[mask]))
  original=pd.read_csv(REPORT/f'{name}_test_window_predictions.csv')
  assert np.allclose(p,original.score,rtol=1e-6,atol=1e-8)
  groups={}
  clipped=d.loc[mask,'rail_fraction'].to_numpy()>.01
  for label,sub in [('rail_over_1pct',clipped),('rail_at_most_1pct',~clipped)]:
   keep=sub & eligible
   if keep.any(): groups[label]=metrics(y[keep],p[keep],saved['threshold'])
  audit['models'][name]=groups
 (REPORT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
