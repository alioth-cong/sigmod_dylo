"""Unpack anonymous bundles without overwriting existing observations or replay outputs."""
import hashlib,json,tarfile,sys
from pathlib import Path
from bundle_io import open_bundle
R=Path(__file__).resolve().parent
if __name__=='__main__':
 names=sys.argv[1:] or ['inputs-and-plans.tar.gz']
 for name in names:
  p=R/'bundles'/name
  with open_bundle(name) as tf:
   for m in tf:
    if not m.isfile():raise ValueError('Unexpected non-file member')
    dst=(R/m.name).resolve();assert dst.is_relative_to(R) and not Path(m.name).is_absolute()
    data=tf.extractfile(m).read()
    if dst.exists():assert dst.read_bytes()==data,'Refusing to overwrite '+str(dst)
    else:dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  print('unpacked',name)
