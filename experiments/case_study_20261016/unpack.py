"""Verify and unpack the frozen case-study inputs and observed records."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
def unpack():
    manifest=json.loads((ROOT/'bundles/manifest.json').read_text())
    for entry in manifest:
        path=ROOT/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],str(path)
        with tarfile.open(path,'r:gz') as archive:
            members=archive.getmembers()
            assert len(members)==entry['members']
            for member in members:
                target=(ROOT/member.name).resolve()
                assert target.is_relative_to(ROOT) and member.isfile(),member.name
            archive.extractall(ROOT,members=members,filter='data' if hasattr(tarfile,'data_filter') else None)
        print('Verified and unpacked',path.name,len(members),'files')
    return ROOT
if __name__=='__main__':unpack()
