"""BlinkIt desktop live EEG detector. Real device is the default source."""
import csv,json,queue,sys,time,uuid
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
from threading import Thread,Event
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np
from engine import StreamDetector

ROOT=Path(__file__).resolve().parents[2]
BG='#101827'; CARD='#182438'; TEXT='#e7edf7'; CYAN='#56dfd2'; MUTED='#a1b0c5'

def ports():
 from serial.tools.list_ports import comports
 if sys.platform=='darwin': return sorted(str(p) for p in Path('/dev').glob('cu.*') if 'Bluetooth-Incoming' not in p.name)
 return [p.device for p in comports()]

class App:
 def __init__(self,root):
  self.root=root; root.title('BlinkIt · Live EEG blinks'); root.geometry('1080x780'); root.configure(bg=BG)
  self.messages=queue.Queue(maxsize=200); self.stop=Event(); self.thread=None; self.trace=deque(maxlen=1200); self.scores=deque(maxlen=120); self.markers=deque(maxlen=100); self.count=0; self.last_data=None; self.closing=False; self.flash_until=0
  self.port=tk.StringVar(); self.channel=tk.StringVar(value='4'); self.sound=tk.BooleanVar(value=True); self.mode='LIVE EEG'; self.folder=None
  tk.Label(root,text='BlinkIt',font=('Helvetica',28,'bold'),bg=BG,fg=TEXT).pack(anchor='w',padx=24,pady=(18,0))
  tk.Label(root,text='EEG blink detector · saved 1D CNN · no camera needed',font=('Helvetica',12),bg=BG,fg=MUTED).pack(anchor='w',padx=24)
  bar=tk.Frame(root,bg=BG); bar.pack(fill='x',padx=24,pady=16)
  tk.Label(bar,text='BITalino port',bg=BG,fg=TEXT).pack(side='left'); self.combo=ttk.Combobox(bar,textvariable=self.port,width=28); self.combo.pack(side='left',padx=8)
  ttk.Button(bar,text='Refresh',command=self.refresh).pack(side='left')
  tk.Label(bar,text='EEG channel A',bg=BG,fg=TEXT).pack(side='left',padx=(18,3)); self.channel_box=ttk.Combobox(bar,textvariable=self.channel,values=['1','2','3','4'],width=3,state='readonly'); self.channel_box.pack(side='left')
  self.connect=ttk.Button(bar,text='Connect & detect',command=self.start_live); self.connect.pack(side='left',padx=12)
  self.disconnect=ttk.Button(bar,text='Stop',command=self.request_stop,state='disabled'); self.disconnect.pack(side='left')
  cards=tk.Frame(root,bg=CARD); cards.pack(fill='x',padx=24)
  self.indicator=tk.Label(cards,text='READY',font=('Helvetica',26,'bold'),bg=CARD,fg=CYAN); self.indicator.pack(side='left',padx=20,pady=18)
  self.counter=tk.Label(cards,text='0 blinks',font=('Helvetica',22),bg=CARD,fg=TEXT); self.counter.pack(side='right',padx=20)
  self.status=tk.Label(root,text='Choose the paired port. Close OpenSignals before connecting.',bg=BG,fg=MUTED,anchor='w',wraplength=1000); self.status.pack(fill='x',padx=24,pady=10)
  self.graph=tk.Canvas(root,height=230,bg=CARD,highlightthickness=0); self.graph.pack(fill='both',expand=True,padx=24,pady=5)
  self.scoregraph=tk.Canvas(root,height=120,bg=CARD,highlightthickness=0); self.scoregraph.pack(fill='x',padx=24,pady=5)
  self.detail=tk.Label(root,text='Model threshold 0.80 · ~1.05 s algorithm delay + Bluetooth delivery · experimental',bg=BG,fg=MUTED); self.detail.pack(anchor='w',padx=24,pady=8)
  bottom=tk.Frame(root,bg=BG); bottom.pack(fill='x',padx=24,pady=10)
  ttk.Checkbutton(bottom,text='Blink sound',variable=self.sound).pack(side='left'); ttk.Button(bottom,text='Test sound',command=self.beep).pack(side='left',padx=8)
  self.replay_button=ttk.Button(bottom,text='Optional: replay an EEG CSV',command=self.start_replay); self.replay_button.pack(side='right')
  self.saved=tk.Label(root,text='Live streams and detections are saved automatically when connected.',bg=BG,fg=MUTED,wraplength=1000); self.saved.pack(anchor='w',padx=24,pady=(0,16))
  self.refresh(); root.protocol('WM_DELETE_WINDOW',self.close); root.after(50,self.poll)
 def refresh(self):
  names=ports(); self.combo['values']=names
  if not self.port.get() and names: self.port.set(next((p for p in names if 'bitalino' in p.lower()),names[0]))
 def beep(self):
  if sys.platform=='win32':
   def play():
    import winsound
    winsound.Beep(1050,120)
   Thread(target=play,daemon=True).start()
  else: self.root.bell()
 def start_live(self):
  if not self.port.get().strip(): self.status.config(text='Choose a BITalino serial port first.'); return
  self.start(None)
 def start_replay(self):
  path=filedialog.askopenfilename(title='REPLAY ONLY — choose recorder eeg.csv',filetypes=[('EEG CSV','*.csv')])
  if path: self.start(Path(path))
 def start(self,replay):
  if self.thread and self.thread.is_alive(): return
  try: channel=int(self.channel.get()); assert 1<=channel<=4
  except (ValueError,AssertionError): self.status.config(text='Select an EEG channel A1–A4.'); return
  self.mode='REPLAY — NOT LIVE' if replay else 'LIVE EEG'; self.stop.clear(); self.trace.clear(); self.scores.clear(); self.markers.clear(); self.count=0; self.last_data=None
  self.counter.config(text='0 blinks'); self.indicator.config(text='CONNECTING',fg=CYAN); self.status.config(text=self.mode+' · Loading the saved CNN…')
  for button in [self.connect,self.replay_button]: button.config(state='disabled')
  self.combo.config(state='disabled'); self.channel_box.config(state='disabled'); self.disconnect.config(state='normal')
  self.thread=Thread(target=self.worker,args=(self.port.get().strip(),channel,replay),daemon=True); self.thread.start()
 def post(self,kind,value):
  self.messages.put((kind,value),timeout=2)
 def worker(self,port,channel,replay):
  board=None; started=False; files=[]; folder=None; error=None; meta={}
  try:
   detector=StreamDetector()
   if replay:
    source=open(replay,newline='',encoding='utf-8-sig'); files.append(source); reader=csv.DictReader(source)
    if not {'sequence','adc'}.issubset(reader.fieldnames or []): raise ValueError('Replay requires a recorder eeg.csv with sequence and adc columns.')
   else:
    from bitalino import BITalino
    self.post('status','LIVE EEG · Opening the port and checking firmware…')
    # Retain the object even if the constructor handshake fails, to release serial resources.
    board=BITalino.__new__(BITalino); board.__init__(port,timeout=3)
    board.timeout=2; board.start(1000,[channel-1]); started=True
   folder=ROOT/'data/raw/live_blinks'/('replay' if replay else 'live')/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]); folder.mkdir(parents=True)
   def writer(name,fields):
    f=open(folder/name,'w',newline='',encoding='utf-8',buffering=1); files.append(f); w=csv.writer(f); w.writerow(fields); return w
   rawlog=writer('eeg.csv',['sample_index','sequence','adc','batch_received_elapsed_s'])
   scorelog=writer('scores.csv',['center_s','score','rail_fraction','flat'])
   eventlog=writer('blinks.csv',['peak_s','score','emitted_s','algorithm_delay_s','host_alert_elapsed_s'])
   meta={'source':'replay' if replay else 'live_bitalino','source_file':str(replay) if replay else None,'port':None if replay else port,'channel':channel,'sampling_hz':1000,'model':'cnn_01_two_seed_mean','threshold':.8,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','timing':'relative to first sample; not hardware-synchronized','online_postprocessing':'local maximum +/- 300ms; minimum 350ms between alerts'}
   (folder/'session.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); self.post('folder',str(folder))
   start=time.monotonic(); received=0; previous_seq=None; original_index=-1
   while not self.stop.is_set():
    if replay:
     batch=[]
     for _ in range(50):
      row=next(reader,None)
      if row is None: break
      batch.append([int(row['sequence']),float(row['adc'])])
     if not batch: break
     self.stop.wait(max(0,received/1000-(time.monotonic()-start)))
     if self.stop.is_set(): break
    else:
     values=board.read(50); batch=np.column_stack([values[:,0],values[:,5]])
    now=time.monotonic()-start
    for seq,adc in batch:
     seq=int(seq); step=1 if previous_seq is None else ((seq-previous_seq)%16 or 16); original_index+=step; previous_seq=seq
     rawlog.writerow([original_index,seq,int(adc),f'{now:.6f}'])
    traces,scores,alerts=detector.push(batch); received+=len(batch)
    for s in scores: scorelog.writerow([s[k] for k in ['center_s','score','rail_fraction','flat']])
    for a in alerts: eventlog.writerow([a[k] for k in ['peak_s','score','emitted_s','algorithm_delay_s']]+[time.monotonic()-start])
    self.post('data',{'trace':traces,'scores':scores,'alerts':alerts,'gaps':detector.gaps,'elapsed':detector.sample/1000,'warming':detector.since_reset<3500})
   meta.update(status='stopped' if self.stop.is_set() else 'finished',samples_received=received,estimated_missing_samples=detector.gaps)
  except Exception as exc:
   error=str(exc)
   try: self.post('error',error)
   except queue.Full: pass
  finally:
   if board is not None:
    try:
     if started: board.stop()
    except Exception: pass
    try: board.close()
    except Exception:
     try: board.socket.close()
     except Exception: pass
   for f in files: f.close()
   if folder:
    if error: meta.update(status='error',error=error)
    (folder/'session.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
   try: self.post('done',error)
   except queue.Full: pass
 def request_stop(self): self.stop.set(); self.status.config(text='Stopping and saving…'); self.disconnect.config(state='disabled')
 def close(self):
  if self.thread and self.thread.is_alive(): self.closing=True; self.request_stop()
  else: self.root.destroy()
 def poll(self):
  try:
   while True:
    kind,value=self.messages.get_nowait()
    if kind=='status': self.status.config(text=value)
    elif kind=='folder': self.saved.config(text='Saving to: '+value)
    elif kind=='error': self.status.config(text='Connection stopped: '+value); self.indicator.config(text='DISCONNECTED',fg='#ff9191')
    elif kind=='done':
     self.connect.config(state='normal'); self.replay_button.config(state='normal'); self.combo.config(state='normal'); self.channel_box.config(state='readonly'); self.disconnect.config(state='disabled'); self.last_data=None
     self.indicator.config(text='STOPPED' if not value else 'DISCONNECTED',bg=CARD)
     if not value: self.status.config(text=self.mode+' · Stopped. Files saved.')
     if self.closing: self.root.destroy(); return
    elif kind=='data':
     self.last_data=time.monotonic(); self.trace.extend(value['trace']); self.scores.extend(value['scores'])
     msg=f"{self.mode} · {value['elapsed']:.1f}s · estimated missing samples: {value['gaps']}"
     if value['warming']: msg+=' · Warming up / recovering signal…'
     elif self.scores:
      s=self.scores[-1]; msg+=f" · score {s['score']:.2f}"
      if s['flat']: msg+=' · FLAT SIGNAL — alerts paused'
      elif s['rail_fraction']>.01: msg+=' · CLIPPING — detections may be unreliable'
     self.status.config(text=msg)
     for a in value['alerts']:
      self.count+=1; self.markers.append(a['peak_s']); self.flash_until=time.monotonic()+.35
      if self.sound.get(): self.beep()
     self.counter.config(text=f'{self.count} blinks')
  except queue.Empty: pass
  if self.last_data:
   stale=time.monotonic()-self.last_data>1
   flashing=time.monotonic()<self.flash_until and not stale
   self.indicator.config(text='BLINK!' if flashing else ('NO DATA' if stale else self.mode),fg='#ffffff' if flashing else CYAN,bg='#178f83' if flashing else CARD)
   if stale: self.status.config(text=self.mode+' · Waiting for EEG data; no new detections.')
  self.draw(); self.root.after(50,self.poll)
 def draw(self):
  end=self.trace[-1][0] if self.trace else 10; start=max(0,end-10)
  def chart(canvas,points,label,limits=None):
   canvas.delete('all'); w=max(canvas.winfo_width(),100); h=max(canvas.winfo_height(),80); pad=35
   canvas.create_text(14,14,text=label,anchor='w',fill=TEXT,font=('Helvetica',11))
   pts=[(t,v) for t,v in points if start<=t<=end]
   lo,hi=limits or ((min(v for _,v in pts),max(v for _,v in pts)) if pts else (-1,1)); hi=max(hi,lo+1)
   xy=lambda t,v:(pad+(t-start)/10*(w-pad-12),h-22-(v-lo)/(hi-lo)*(h-55))
   for j in range(6):
    t=start+j*2; x,_=xy(t,lo); canvas.create_line(x,30,x,h-22,fill='#25354c'); canvas.create_text(x,h-10,text=f'{t:.0f}s',fill=MUTED)
   if limits:
    _,y=xy(start,.8); canvas.create_line(pad,y,w-12,y,fill='#e4b25f',dash=(4,4))
   coords=[z for t,v in pts for z in xy(t,v)]
   if len(coords)>=4: canvas.create_line(*coords,fill=CYAN,width=2)
   for t in self.markers:
    if start<=t<=end:
     x,_=xy(t,lo); canvas.create_line(x,30,x,h-22,fill='#ffbd69',width=2)
  chart(self.graph,self.trace,'Filtered EEG · ADC counts · orange lines = detected blink times')
  chart(self.scoregraph,[(s['center_s'],s['score']) for s in self.scores],'CNN score · threshold 0.80 · not a calibrated probability',(0,1))

if __name__=='__main__': App(tk.Tk()); tk.mainloop()
