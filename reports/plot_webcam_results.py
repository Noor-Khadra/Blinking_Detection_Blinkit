from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator

REPO=Path(__file__).resolve().parents[1]
root=REPO/'data/derived/noor/session1'
OUTPUT=REPO/'reports/figures'
OUTPUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((root/'labels.csv').open()))
measure=list(csv.DictReader((root/'webcam_ear.csv').open()))
blinks=[r for r in rows if r['review_status']=='accepted' and r['event_type']=='blink']
closures=[r for r in rows if r['review_status']=='accepted' and r['event_type']=='sustained_closure']
rejected=[r for r in rows if r['review_status']=='rejected']
time=np.array([float(r['video_time_s']) for r in measure]);left=np.array([float(r['ear_left']) for r in measure]);right=np.array([float(r['ear_right']) for r in measure])
durations=np.array([float(r['offset_s'])-float(r['onset_s']) for r in blinks]);cd=np.array([float(r['offset_s'])-float(r['onset_s']) for r in closures])
summary=dict(total_entries=len(rows),accepted_blinks=len(blinks),accepted_sustained_closures=len(closures),rejected_proposals=len(rejected),frames=len(measure),last_frame_s=float(time[-1]),median_blink_duration_ms=float(np.median(durations)*1000),blink_duration_range_ms=[float(durations.min()*1000),float(durations.max()*1000)],sustained_duration_range_s=[float(cd.min()),float(cd.max())],annotated_blinks_per_recording_minute=len(blinks)/(time[-1]/60))
(OUTPUT/'results_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#10283f','axes.edgecolor':'#cbd5e1','xtick.color':'#536579','ytick.color':'#536579','figure.facecolor':'#f5f8fb','axes.facecolor':'#ffffff','savefig.facecolor':'#f5f8fb','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(16,11.5));gs=fig.add_gridspec(3,2,height_ratios=[.9,1,1.05],hspace=.65,wspace=.38,left=.15,right=.97,top=.80,bottom=.09)
fig.text(.15,.953,'BlinkIt | Noor webcam results',fontsize=25,weight='bold')
fig.text(.15,.916,'Human-reviewed video annotations · 20,520 frames · 11 min 16 s',fontsize=13,color='#536579')
fig.text(.15,.875,f'62 blinks     /     7 sustained closures     /     14 rejected proposals',fontsize=14,weight='bold',color='#127c83')
teal='#148c91';orange='#df9b42';gray='#a7b3c2'
ax=fig.add_subplot(gs[0,0]);cats=['Accepted blinks','Sustained closures','Rejected proposals'];vals=[62,7,14]
ax.barh(cats[::-1],vals[::-1],color=[gray,orange,teal],height=.55)
for y,v in enumerate(vals[::-1]):ax.text(v+.7,y,str(v),va='center',weight='bold')
ax.set_xlim(0,70);ax.set_xlabel('Number of event entries');ax.set_title('A   Final review decisions',loc='left',pad=14);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=fig.add_subplot(gs[0,1]);bins=np.arange(0,1001,100);n,_,_=ax.hist(durations*1000,bins=bins,color=teal,rwidth=.85)
ax.axvline(np.median(durations)*1000,color='#10283f',ls='--',lw=1.4,label=f'Median: {np.median(durations)*1000:.0f} ms');ax.legend(frameon=False,fontsize=10,loc='upper right');ax.set_xlabel('Annotated blink duration (ms)');ax.set_ylabel('Blink count');ax.set_title('B   Duration of the 62 accepted blinks',loc='left',pad=14);ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=fig.add_subplot(gs[1,:]);peaks=[float(r['peak_s']) for r in blinks];ax.vlines(peaks,.8,1.2,color=teal,lw=1.4)
for r in closures:ax.broken_barh([(float(r['onset_s']),float(r['offset_s'])-float(r['onset_s']))],(-.2,.4),facecolors=orange)
ax.set_yticks([0,1],['Sustained closure','Blink peak']);ax.set_ylim(-.6,1.6);ax.set_xlim(0,time[-1]);ax.set_xlabel('Video time (min:sec)');ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{int(x)//60}:{int(x)%60:02d}'));ax.set_xticks(np.arange(0,661,60));ax.set_title('C   When the reviewed events occur',loc='left',pad=14);ax.grid(axis='x',alpha=.15)
# A specific corrected event, showing real frame-level measurements.
event=next(r for r in rows if r['event_id']=='76');start,peak,end=[float(event[k]) for k in ['onset_s','peak_s','offset_s']];mask=(time>=start-.45)&(time<=end+.45)
ax=fig.add_subplot(gs[2,0]);ax.plot(time[mask],left[mask],color=teal,lw=1.8,label='Left eye');ax.plot(time[mask],right[mask],color='#7764ad',lw=1.6,label='Right eye');ax.axvspan(start,end,color=teal,alpha=.11);ax.axvline(peak,color='#10283f',ls='--',lw=1.1,label='Reviewed peak');ax.set_xlabel('Video time (seconds)');ax.set_ylabel('Eye aspect ratio (EAR)');ax.set_title('D   Example: corrected blink #76',loc='left',pad=14);ax.legend(frameon=False,fontsize=9,ncols=2,loc='lower right');ax.grid(alpha=.12);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
ax=fig.add_subplot(gs[2,1]);edges=np.arange(0,721,60);counts,_=np.histogram(peaks,bins=edges);labels=[str(i) for i in range(1,len(counts)+1)];ax.bar(np.arange(len(counts)),counts,color=teal,width=.7);ax.set_xticks(np.arange(len(counts)),labels);ax.set_xlabel('Recording minute (12 is a partial minute)');ax.set_ylabel('Annotated blink count');ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));ax.set_title('E   Blink counts across the recording',loc='left',pad=14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.15,.029,'Video timestamps only; EEG alignment is still required. Counts describe reviewed annotations, not detector accuracy.',fontsize=10,color='#536579')
fig.savefig(OUTPUT/'webcam_results.png',dpi=190)
plt.close(fig)
