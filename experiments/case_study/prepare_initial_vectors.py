"""Recreate omitted vector cache from the supplied frozen 32B initial reports."""
from pathlib import Path
import asyncio,sys
R=Path(__file__).resolve().parent/'work/qwen32-std4-20261017'
if __name__=='__main__':
 sys.path.insert(0,str(R));import runtime
 assert not (R/'initial/index').exists(),'Refusing to overwrite an initial index'
 asyncio.run(runtime.materialize(runtime.load('base'),R/'initial/index'))
