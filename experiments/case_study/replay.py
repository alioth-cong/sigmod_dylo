"""Optional fresh maintenance replay; observed results are stored separately."""
from pathlib import Path
import sys,os,asyncio,importlib.util
R=Path(__file__).resolve().parent;W=R/'work/qwen32-std4-20261017'
if __name__=='__main__':
 arm=sys.argv[1];assert arm in ['dylo','batch','rebuild','hit-leiden','dynamo','qca','df-louvain','ds-louvain']
 assert os.environ.get('E1_GPU_UUID'),'Set E1_GPU_UUID to the GPU used by the model services'
 os.environ.setdefault('QWEN32_CONCURRENCY','4')
 sys.path.insert(0,str(W));import run,all_planning
 run.build=all_planning.build
 asyncio.run(run.arm_run(arm))
