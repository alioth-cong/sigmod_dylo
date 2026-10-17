"""Build the original std4 and the observed baseline adapters from source."""
from pathlib import Path
import os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def run(cmd):print(' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True)
def build(standalone=False):
    source=ROOT/'native';target=ROOT/'work/fast-summary-20261016/native';target.mkdir(parents=True,exist_ok=True)
    cxx=os.environ.get('CXX','g++');flags=['-O3','-march=native','-std=c++17']
    for name,src in [('dylo_driver','driver_std4.cxx'),('hit_driver','hit_leiden_driver_inc.cxx')]:run([cxx,*flags,'-I'+str(source),source/src,'-o',target/name])
    original=ROOT/'work/table5-std4-20261016/native';original.mkdir(parents=True,exist_ok=True)
    run([cxx,*flags,source/'original_std4/driver_std4.cxx','-o',original/'dylo_driver_std4'])
    extra=ROOT/'work/table5-all-baselines-20261016/native';extra.mkdir(exist_ok=True)
    shutil.copy2(original/'dylo_driver_std4',extra/'dylo_driver_std4')
    if standalone:return
    external=REPO/'external/puzzlef-dynamic'
    if not external.exists():raise RuntimeError('Fetch/build the baseline dependencies with tools/fetch_baselines.sh first.')
    run([cxx,*flags,'-fopenmp','-DTYPE=float','-I'+str(external),source/'df_driver.cxx','-o',target/'df_driver'])
    cp=(REPO/'harness/java/build/.classpath').read_text().strip()
    java=target/'java';java.mkdir(exist_ok=True)
    run([os.environ.get('JAVAC','javac'),'-cp',cp,'-d',java,*sorted((source/'java').rglob('*.java'))])
if __name__=='__main__':build('--standalone' in sys.argv)
