"""Check the packaged source/evidence inventory without requesting models."""
from pathlib import Path
import hashlib,json,ast,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    checks=json.loads((ROOT/'CHECKSUMS.json').read_text());missing=[];bad=[]
    for name,digest in checks.items():
        p=ROOT/name
        if not p.is_file():missing.append(name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
    assert not missing and not bad,{'missing':missing,'changed':bad}
    for path in (ROOT/'experiments').rglob('*.py'):
        if any(x in path.parts for x in ['observed','cache','__pycache__']):continue
        ast.parse(path.read_text(),filename=str(path))
    print('Artifact inventory verified:',len(checks),'files; Python syntax passed; no model calls.')
if __name__=='__main__':main()
