"""Seed only fixed questions/rubric for a new quality run, without model requests."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    dest=ROOT/'outputs/table5-all-baselines-20261016/quality';dest.mkdir(parents=True,exist_ok=True)
    for name in ['questions.json','criteria.json','judge_prompt_from_paper.txt']:
        source=ROOT/'results/quality'/name;target=dest/name
        if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('Conflicting quality input: '+str(target))
        shutil.copyfile(source,target)
    print('Fixed quality inputs prepared; no model calls; archived answers/checkpoints/calibration not copied.')
