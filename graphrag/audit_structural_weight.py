"""Recompute the appendix's retained/rebuilt token-to-structural-weight fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress,spearmanr
R=Path(__file__).resolve().parents[1]
def fit(x,y):
 z=linregress(x,y)
 return dict(r2=z.rvalue**2,rho=float(spearmanr(x,y).statistic),slope=z.slope,intercept=z.intercept)
if __name__=='__main__':
 expected={'norecl327':(1035,944,.851,.380,.222,.675,.746,.913,3.41),'rebuild327_fresh':(1164,1002,.859,.437,.143,.738,.623,.931,4.43)}
 output={}
 for name,ref in expected.items():
  d=pd.read_csv(R/'results/structural_weight'/f'{name}.csv');leaf=d[d.leaf==1];internal=d[d.leaf==0]
  a=fit(leaf.w,leaf.tokens);b=fit(internal.w,internal.tokens);raw=fit(d.w,d.tokens);ratio=b['slope']/a['slope'];scaled=fit(d.w*np.where(d.leaf==1,1,ratio),d.tokens)
  vals=[len(d),len(leaf),a['r2'],b['r2'],raw['r2'],scaled['r2'],raw['rho'],scaled['rho'],ratio]
  for i,(x,y) in enumerate(zip(vals,ref)):assert abs(x-y)<(0.005 if i==8 else .00051 if i>=2 else .1),(name,i,x,y)
  output[name]=dict(samples=len(d),leaf=a,internal=b,unscaled=raw,rescaled=scaled,internal_multiplier=ratio)
 print(json.dumps(dict(status='passed',fits=output),indent=2))
