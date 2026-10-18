"""Stream a complete gzip archive or its byte-identical numbered volumes."""
from contextlib import contextmanager
from pathlib import Path
import io,tarfile
R=Path(__file__).resolve().parent
class Volumes(io.RawIOBase):
 def __init__(self,paths):super().__init__();self.paths=iter(paths);self.current=None
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.current is None:
    p=next(self.paths,None)
    if p is None:return 0
    self.current=p.open('rb')
   n=self.current.readinto(b)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current:self.current.close()
  super().close()
def volume_paths(name):
 p=R/'bundles'/name
 if p.exists():return [p]
 parts=sorted(p.parent.glob(p.name+'.part*'))
 assert parts and [x.name for x in parts]==[p.name+f'.part{i:03d}' for i in range(1,len(parts)+1)],'Missing archive volume: '+name
 return parts
@contextmanager
def open_bundle(name):
 with io.BufferedReader(Volumes(volume_paths(name))) as stream:
  with tarfile.open(fileobj=stream,mode='r|gz') as tf:yield tf
