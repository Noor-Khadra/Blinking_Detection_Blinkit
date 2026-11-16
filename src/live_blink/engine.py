"""CPU streaming inference for the saved cnn_01 two-seed model, no camera."""
from collections import deque
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
import torch
from torch import nn

class BlinkCNN(nn.Module):
 def __init__(self,width=16,dropout=.2):
  super().__init__()
  self.layers=nn.Sequential(nn.Conv1d(2,width,9,padding=4),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(width,width*2,7,padding=3),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(width*2,width*2,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(12),nn.Flatten(),nn.Linear(width*2*12,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
 def forward(self,x): return self.layers(x).squeeze(1)

class Predictor:
 def __init__(self,folder=None):
  torch.set_num_threads(2)
  folder=Path(folder or Path(__file__).parent/'models'); self.models=[]
  for seed in (20261116,20261117):
   ck=torch.load(folder/f'cnn_01_{seed}.pt',map_location='cpu',weights_only=True)
   m=BlinkCNN(ck['config']['width'],ck['config']['dropout']); m.load_state_dict(ck['state_dict']); m.eval()
   self.models.append((m,float(ck['scale'])))
 @torch.no_grad()
 def score_waveforms(self,wave):
  wave=np.asarray(wave,dtype=np.float32).reshape(-1,192)
  centered=wave-wave.mean(axis=1,keepdims=True); scores=[]
  for m,scale in self.models:
   x=np.stack([centered/scale,centered/np.maximum(centered.std(axis=1,keepdims=True),scale*.05)],axis=1).astype(np.float32)
   scores.append(m(torch.from_numpy(x)).sigmoid().numpy())
  return np.mean(scores,axis=0)

class PeakGate:
 """Confirm local maximum after 300ms, suppress alerts closer than 350ms."""
 def __init__(self,threshold=.8): self.threshold=threshold; self.points=deque(maxlen=7); self.last=-1e9
 def push(self,t,score,usable=True):
  self.points.append((t,score,usable))
  if len(self.points)<7: return None
  a=list(self.points); ct,cs,ok=a[3]
  if ok and cs>=self.threshold and cs>max(x[1] for x in a[:3]) and cs>=max(x[1] for x in a[4:]) and ct-self.last>=.35:
   self.last=ct
   return {'peak_s':ct,'score':cs}
  return None

class StreamDetector:
 def __init__(self,predictor=None):
  self.predictor=predictor or Predictor(); self.sos=butter(3,[.5,12],fs=1000,btype='bandpass',output='sos'); self.sample=-1; self.previous=None; self.gaps=0; self.reset()
 def reset(self):
  self.zi=np.zeros((len(self.sos),2)); self.raw=deque(maxlen=1500); self.filtered=deque(maxlen=1500); self.since_reset=0; self.gate=PeakGate()
 def push(self,rows):
  """Rows contain [sequence, ADC]. Returns plot points, scores, alerts."""
  traces=[]; scores=[]; alerts=[]
  # Split filtering at sequence gaps; the 4-bit counter cannot expose losses of exactly 16 samples.
  for seq,adc in np.asarray(rows):
   seq=int(seq); step=1 if self.previous is None else ((seq-self.previous)%16 or 16)
   if step!=1: self.gaps+=step-1; self.reset()
   self.sample+=step; self.previous=seq
   value,self.zi=sosfilt(self.sos,[float(adc)],zi=self.zi)
   self.raw.append(float(adc)); self.filtered.append(float(value[0])); self.since_reset+=1
   if self.sample%10==0: traces.append((self.sample/1000,float(value[0])))
   if self.since_reset>=3500 and self.sample%100==99:
    wave=np.interp(np.linspace(0,1,192),np.linspace(0,1,len(self.filtered)),self.filtered)
    score=float(self.predictor.score_waveforms(wave)[0]); center=(self.sample-749.5)/1000
    raw=np.array(self.raw); flat=np.std(raw)<1.; rail=float(np.mean((raw<=1)|(raw>=1022)))
    scores.append({'center_s':center,'score':score,'rail_fraction':rail,'flat':bool(flat)})
    alert=self.gate.push(center,score,not flat)
    if alert: alerts.append({**alert,'emitted_s':self.sample/1000,'algorithm_delay_s':self.sample/1000-alert['peak_s']})
  return traces,scores,alerts
