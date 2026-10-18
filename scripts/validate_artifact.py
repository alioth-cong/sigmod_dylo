"""Validate the release file inventory, archive hashes and source syntax offline."""
from pathlib import Path
import ast,hashlib,json,tarfile,sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'experiments/case_study'))
from bundle_io import open_bundle
def sha(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':
 checks=json.loads((R/'CHECKSUMS.json').read_text())
 for name,digest in checks.items():
  p=R/name;assert p.is_file(),name;assert sha(p.read_bytes())==digest,'Changed file: '+name
  assert p.stat().st_size<100*1024**2,'GitHub individual file limit: '+name
  if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
 provenance=json.loads((R/'provenance.json').read_text())
 archives=0;members=0
 for entry in provenance:
  if 'members' not in entry:continue
  expected={m['path']:m['published_sha256'] for m in entry['members']};seen=set()
  with open_bundle(Path(entry['path']).name) as tf:
   for m in tf:
    assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    assert m.uid==m.gid==m.mtime==0 and not m.uname and not m.gname
    assert m.name in expected and m.name not in seen
    assert sha(tf.extractfile(m).read())==expected[m.name],m.name
    seen.add(m.name)
  assert seen==set(expected);archives+=1;members+=len(seen)
 assert (R/'harness/cpp/dylo/driver_std4.cxx').exists()
 assert not list((R/'harness/cpp/dylo').glob('driver_std[567]*.cxx'))
 assert len(json.loads((R/'ARTIFACT_SNAPSHOT.json').read_text())['methods'])==8
 print(json.dumps(dict(status='passed',published_files=len(checks),archives=archives,archive_members=members,standard_implementation='std4',methods=8),indent=2))
