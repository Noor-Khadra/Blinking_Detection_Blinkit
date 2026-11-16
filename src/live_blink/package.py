"""Build portable Windows bundle; includes weights, never participant recordings."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[2]
path=root/'dist/BlinkIt_Live_CNN.zip'; path.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for f in (root/'src/live_blink').rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and f.name not in {'test_engine.py','package.py'}:
   z.write(f,'BlinkIt_Live_CNN/'+f.relative_to(root).as_posix())
 z.writestr('BlinkIt_Live_CNN/START_HERE.txt','Windows: open src/live_blink/Setup_Windows.cmd once, then Run_Live_Blinks.cmd.\nPair BITalino, close OpenSignals, select outgoing COM port and A4, then Connect & detect.\nSee src/live_blink/README.md for details. Model weights included.\n')
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 assert len([n for n in z.namelist() if n.endswith('.pt')])==2
 print(path)
