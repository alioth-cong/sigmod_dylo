"""Compile the single paper standard and its measured baseline adapters."""
from pathlib import Path
import os,subprocess,sys
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def run(*args):subprocess.run(list(map(str,args)),check=True)
if __name__=='__main__':
 cxx=os.environ.get('CXX','g++');flags=['-O3','-march=native','-std=c++17']
 dst=R/'work/qwen32-std4-20261017/native';dst.mkdir(parents=True,exist_ok=True)
 run(cxx,*flags,REPO/'harness/cpp/dylo/driver_std4.cxx','-o',dst/'dylo_driver_std4')
 target=R/'work/fast-summary-20261016/native';target.mkdir(parents=True,exist_ok=True)
 run(cxx,*flags,'-I'+str(R/'native'),R/'native/hit_leiden_driver_inc.cxx','-o',target/'hit_driver')
 if '--standalone' not in sys.argv:
  dep=REPO/'external/puzzlef-dynamic';assert dep.exists(),'Run tools/fetch_baselines.sh first'
  run(cxx,*flags,'-fopenmp','-DTYPE=float','-I'+str(dep),R/'native/df_driver.cxx','-o',target/'df_driver')
  cp=(REPO/'harness/java/build/.classpath').read_text().strip();java=target/'java';java.mkdir(exist_ok=True)
  run(os.environ.get('JAVAC','javac'),'-cp',cp,'-d',java,*sorted((R/'native/java').rglob('*.java')))
