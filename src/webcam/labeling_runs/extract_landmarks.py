"""Extract both-eye EAR and blink blendshapes, without dropping video frames.
Requires: mediapipe, opencv-python-headless, numpy, ffprobe on PATH.
Usage: python extract_landmarks.py VIDEO MODEL OUTPUT_CSV
Timestamps are presentation times in the encoded source, not capture-clock times.
"""
import csv,json,subprocess,sys,time
import cv2,mediapipe as mp,numpy as np
from mediapipe.tasks.python import vision
video,model,out=sys.argv[1:4]
if len(sys.argv)>4:
 times=[float(line.strip().split(',')[0]) for line in open(sys.argv[4]) if line.strip().split(',')[0]]
else:
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',video]))
 times=[float(x['best_effort_timestamp_time']) for x in probe['frames']]
cap=cv2.VideoCapture(video)
opts=vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),running_mode=vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True)
# Subject's right and left eyes; pixel distances correct for non-square frames.
ri=[33,160,158,133,153,144];le=[362,385,387,263,373,380]
def ear(pts,ids):
 p=pts[ids];return (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*np.linalg.norm(p[0]-p[3]))
fields=['frame_index','video_time_s','model_timestamp_ms','timestamp_adjusted','face_found','ear_right','ear_left','blink_right','blink_left','right_eye_width_px','left_eye_width_px','eye_mid_x','eye_mid_y']
n=0;last=-1;start=time.monotonic()
with open(out,'w',newline='') as f,vision.FaceLandmarker.create_from_options(opts) as detector:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 while True:
  ok,frame=cap.read()
  if not ok:break
  if n>=len(times):raise RuntimeError('More decoded frames than timestamps')
  t=times[n];ms=max(round(t*1000),last+1);last=ms
  rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ms)
  row=dict(frame_index=n,video_time_s=f'{t:.6f}',model_timestamp_ms=ms,timestamp_adjusted=int(ms!=round(t*1000)),face_found=int(bool(result.face_landmarks)))
  if result.face_landmarks:
   h,ww=frame.shape[:2];p=np.array([(v.x*ww,v.y*h) for v in result.face_landmarks[0]])
   b={v.category_name:v.score for v in result.face_blendshapes[0]}
   row.update(ear_right=ear(p,ri),ear_left=ear(p,le),blink_right=b.get('eyeBlinkRight',''),blink_left=b.get('eyeBlinkLeft',''),right_eye_width_px=np.linalg.norm(p[33]-p[133]),left_eye_width_px=np.linalg.norm(p[362]-p[263]),eye_mid_x=float(np.mean(p[[33,133,362,263],0])),eye_mid_y=float(np.mean(p[[33,133,362,263],1])))
  w.writerow(row);n+=1
  if n%1000==0:print(f'{n}/{len(times)} frames; {time.monotonic()-start:.1f}s',flush=True)
cap.release()
print(json.dumps(dict(decoded_frames=n,timestamp_frames=len(times),runtime_seconds=time.monotonic()-start)),flush=True)
if n!=len(times):raise RuntimeError('Frame-count mismatch; inspect before using labels')
