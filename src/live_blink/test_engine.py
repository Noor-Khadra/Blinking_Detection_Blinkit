"""Hardware-free checks; replay results are NOT live-device validation."""
import json,sys,unittest
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt
from engine import Predictor,StreamDetector,PeakGate

ROOT=Path(__file__).resolve().parents[2]
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.predictor=Predictor()
 def test_saved_prediction_parity(self):
  import csv
  with (ROOT/'data/model/rana_new/windows.csv').open() as f: rows=list(csv.DictReader(f))
  ix=np.array([i for i,r in enumerate(rows) if r['split']=='test'])[:64]
  waves=np.load(ROOT/'data/model/rana_new/windows.npz')['waveforms_128hz'][ix]
  with (ROOT/'reports/rana_cnn_xgb/cnn_01_test_windows.csv').open() as f: expected=np.array([float(r['score']) for r in csv.DictReader(f)])[:64]
  np.testing.assert_allclose(self.predictor.score_waveforms(waves),expected,atol=2e-6)
 def test_filter_window_and_batching(self):
  rng=np.random.default_rng(5); raw=512+30*rng.normal(size=4000); rows=np.c_[np.arange(4000)%16,raw]
  d=StreamDetector(self.predictor); _,scores,_=d.push(rows)
  filtered=sosfilt(d.sos,raw)
  expected=[]
  for end in range(3500,4001,100):
   wave=np.interp(np.linspace(0,1,192),np.linspace(0,1,1500),filtered[end-1500:end]); expected.append(self.predictor.score_waveforms(wave)[0])
  np.testing.assert_allclose([s['score'] for s in scores],expected,atol=2e-6)
  d2=StreamDetector(self.predictor); other=[]
  for chunk in np.array_split(rows,80): other.extend(d2.push(chunk)[1])
  np.testing.assert_allclose([s['score'] for s in other],expected,atol=2e-6)
  self.assertEqual(d2.gaps,0)
  d2.push([[2,512]]) # expected next sequence 0; two missing packets
  self.assertEqual(d2.gaps,2); self.assertEqual(d2.since_reset,1); self.assertEqual(len(d2.gate.points),0)
 def test_peak_single_and_delay(self):
  g=PeakGate(); events=[]
  for i,s in enumerate([.1,.2,.3,.95,.7,.4,.1,.1,.1]):
   e=g.push(i*.1,s)
   if e: events.append((i*.1,e))
  self.assertEqual(len(events),1); self.assertAlmostEqual(events[0][0]-events[0][1]['peak_s'],.3)
  g=PeakGate(); self.assertTrue(all(g.push(i*.1,.95,False) is None for i in range(20)))
 def test_connection_failure_cleanup(self):
  from tempfile import TemporaryDirectory
  from types import SimpleNamespace
  from threading import Event
  from unittest.mock import patch
  import app
  messages=[]; calls=[]
  class FakeBoard:
   def __init__(self,port,timeout): pass
   def start(self,rate,channels): calls.append('start')
   def read(self,n): raise RuntimeError('Injected stream failure')
   def stop(self): calls.append('stop')
   def close(self): calls.append('close')
  fake=SimpleNamespace(stop=Event(),post=lambda k,v:messages.append((k,v)))
  with TemporaryDirectory() as tmp, patch.object(app,'ROOT',Path(tmp)), patch.dict(sys.modules,{'bitalino':SimpleNamespace(BITalino=FakeBoard)}):
   app.App.worker(fake,'MOCK',4,None)
   self.assertEqual(calls,['start','stop','close'])
   self.assertTrue(any(k=='error' and 'Injected' in v for k,v in messages))
   metadata=json.loads(next(Path(tmp).rglob('session.json')).read_text())
   self.assertEqual(metadata['status'],'error')
   self.assertEqual(messages[-1][0],'done')
 def test_real_recording_stream(self):
  raw=np.loadtxt(ROOT/'data/raw/datasets/rana_01_clean/eeg.csv',delimiter=',',skiprows=1,max_rows=15000,usecols=(1,6))
  d=StreamDetector(self.predictor); scores=[]; alerts=[]
  for batch in np.array_split(raw,300):
   _,s,a=d.push(batch); scores.extend(s); alerts.extend(a)
  self.assertEqual(d.gaps,0); self.assertGreater(len(scores),100)
  self.assertTrue(all(np.isfinite(s['score']) for s in scores))
  self.assertTrue(all(abs(a['algorithm_delay_s']-1.0495)<.001 for a in alerts))
  out=ROOT/'reports/live_blink'; out.mkdir(parents=True,exist_ok=True)
  (out/'replay_smoke_test.json').write_text(json.dumps({'source':'REPLAY only: first 15 seconds rana_01_clean','scores':len(scores),'alerts':alerts,'gaps':d.gaps,'hardware_verified':False},indent=2))

if __name__=='__main__': unittest.main()
