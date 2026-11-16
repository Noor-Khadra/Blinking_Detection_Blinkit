"""Restore checksummed recording release ZIPs without overwriting differing data."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('archive_dir',type=Path); args=parser.parse_args()
 manifest=json.loads((ROOT/'data/recording_release.json').read_text()); dest=ROOT/manifest['extract_to']
 for asset in manifest['assets']:
  path=args.archive_dir/asset['asset']
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=asset['sha256']: raise ValueError(f'Checksum mismatch: {path}')
  with zipfile.ZipFile(path) as z:
   for entry in z.infolist():
    target=(dest/entry.filename).resolve()
    if not target.is_relative_to(dest.resolve()): raise ValueError('Invalid archive path')
    if entry.is_dir(): continue
    if target.exists() and target.read_bytes()!=z.read(entry): raise ValueError(f'Refusing to replace differing file: {target}')
   z.extractall(dest)
  print(f'Restored {asset["asset"]} to {dest}')

if __name__=='__main__': main()
