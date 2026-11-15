"""Generate deliberately inclusive, UNREVIEWED blink/closure candidates.
This is a proposal generator, not a validated classifier or final ground truth.
"""
import csv,json,numpy as np
from pathlib import Path
root=Path(__file__).parent
rows=list(csv.DictReader(open(root/'eye_measurements.csv')))
t=np.array([float(r['video_time_s']) for r in rows]);found=np.array([r['face_found']=='1' for r in rows])
def col(k):return np.array([float(r[k]) if r[k] else np.nan for r in rows])
l=col('ear_left');r=col('ear_right');bl=col('blink_left');br=col('blink_right')
# Robust per-session open-eye reference. These thresholds must be checked visually.
base_l=float(np.nanpercentile(l,80));base_r=float(np.nanpercentile(r,80))
nl=l/base_l;nr=r/base_r
# Include unilateral and one-frame dips for review rather than silently deleting them.
strong=found & ((nl<.80)|(nr<.80)|((bl>.4)&(br>.4)))
weak=found & ((nl<.85)|(nr<.85)|((bl>.2)&(br>.2)))
# Seed each strong segment and extend through surrounding partially closed frames.
segments=[]
for i in np.flatnonzero(strong & ~np.r_[False,strong[:-1]]):
 a=i;b=i
 while a>0 and weak[a-1] and t[a]-t[a-1]<.15:a-=1
 while b+1<len(t) and weak[b+1] and t[b+1]-t[b]<.15:b+=1
 if segments and a<=segments[-1][1]:segments[-1][1]=max(segments[-1][1],b)
 else:segments.append([a,b])
fields=['event_id','onset_s','peak_s','offset_s','onset_frame','peak_frame','offset_frame','event_type','review_status','flags','notes']
events=[]
for a,b in segments:
 # Use both eyes for peak; flag unilateral closure for review.
 score=(nl[a:b+1]+nr[a:b+1])/2
 p=a+int(np.nanargmin(score));d=t[b]-t[a]
 flags=[]
 if min(np.nanmin(nl[a:b+1]),np.nanmin(nr[a:b+1]))>=.68:flags.append('shallow_dip_review_carefully')
 if b-a<2:flags.append('short_candidate')
 if d>.8:flags.append('long_closure_or_tracking_error')
 if (np.nanmin(nl[a:b+1])<.68)!=(np.nanmin(nr[a:b+1])<.68):flags.append('eye_disagreement')
 if a==0 or b==len(t)-1:flags.append('recording_boundary')
 if np.any(np.diff(t[max(0,a-1):min(len(t),b+2)])>.1):flags.append('video_time_gap')
 events.append(dict(event_id=len(events)+1,onset_s=round(t[a],6),peak_s=round(t[p],6),offset_s=round(t[b],6),onset_frame=int(a),peak_frame=int(p),offset_frame=int(b),event_type='closure_candidate',review_status='unreviewed',flags=';'.join(flags),notes=''))
with open(root/'blink_candidates_UNREVIEWED.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(events)
# Explicitly list untracked intervals, which must never be treated as no-blink labels.
missing=[]
for a in np.flatnonzero(~found & ~np.r_[False,(~found)[:-1]]):
 b=a
 while b+1<len(t) and not found[b+1]:b+=1
 missing.append(dict(onset_s=float(t[a]),offset_s=float(t[b]),start_frame=int(a),end_frame=int(b)))
summary=dict(frames=len(t),face_found_frames=int(found.sum()),face_found_fraction=float(found.mean()),first_video_timestamp_s=float(t[0]),last_video_timestamp_s=float(t[-1]),candidate_events=len(events),open_reference_ear_left=base_l,open_reference_ear_right=base_r,proposal_threshold_ratio=.80,proposal_boundary_ratio=.85,missing_face_intervals=missing,timestamp_adjustments_for_mediapipe=sum(int(x['timestamp_adjusted']) for x in rows),status='All candidates unreviewed; frame-level visibility and missed events require human review; no EEG alignment applied.')
(root/'processing_summary.json').write_text(json.dumps(summary,indent=2))
data=dict(events=events,fields=fields,summary=summary,trace=[[round(t[i],6),round(l[i],4) if found[i] else None,round(r[i],4) if found[i] else None] for i in range(len(t))])
(root/'review_data.json').write_text(json.dumps(data,separators=(',',':')))
print(json.dumps(summary,indent=2))
