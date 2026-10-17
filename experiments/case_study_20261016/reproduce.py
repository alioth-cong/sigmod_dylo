"""Opt-in generation replay. Does not launch or stop any model service."""
from pathlib import Path
import argparse,asyncio,importlib.util,json,os,shutil,sys
ROOT=Path(__file__).resolve().parent
ARMS=['dylo','batch','rebuild','hit-leiden','dynamo','qca','df-louvain','ds-louvain']
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=ARMS,required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--execute',action='store_true',help='Explicitly permit new local model requests')
    args=parser.parse_args()
    if not args.plan_only and not args.execute:parser.error('Choose --plan-only or --execute')
    from unpack import unpack
    unpack()
    group='table5-std4-20261016' if args.method in ARMS[:3] else 'table5-all-baselines-20261016'
    first=ROOT/'work/table5-std4-20261016';work=ROOT/'work'/group
    if group!='table5-std4-20261016' and not (work/'data').exists():(work/'data').symlink_to(first/'data',target_is_directory=True)
    fast=ROOT/'work/fast-summary-20261016'
    (fast/'data').mkdir(exist_ok=True)
    if not (fast/'data/base').exists():(fast/'data/base').symlink_to(first/'data/base',target_is_directory=True)
    # Fresh output paths never overwrite the archived observations under observed/.
    output=ROOT/'outputs'/group;output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(work))
    planning=load_module('planning',work/'planning.py')
    import pandas as pd
    tables=lambda tag:{n:pd.read_parquet(first/'data'/tag/(n+'.parquet'))
                       for n in ['documents','entities','relationships','text_units','communities','community_reports']}
    base,merged,delta=[tables(t) for t in ['base','merged','delta']]
    prepared=work/'plans'/args.method
    if not prepared.exists():planning.build(args.method,base,merged,prepared)
    delta_ctx=work/'plans/delta_contexts.json'
    if not delta_ctx.exists():
        c=planning.contexts(delta['communities'],delta)
        planning.dump(delta_ctx,[(int(level),frame.to_dict('records')) for level,frame in c])
    if args.plan_only:
        print('CPU-only plan complete; no generation requested:',prepared);return
    if not os.environ.get('E1_GPU_UUID') or not os.environ.get('E1_OWNED_GPU_PIDS'):
        raise RuntimeError('Set E1_GPU_UUID and E1_OWNED_GPU_PIDS for the GPU serving this run.')
    q=planning.q
    cache=fast/'initial/index'
    if not (cache/'published.json').exists():
        # Initial cache construction is separate from every maintenance timer.
        asyncio.run(q.f.e.materialize(base,cache))
    run=load_module('case_runner',work/'run.py')
    result=asyncio.run(run.arm_run(args.method))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
