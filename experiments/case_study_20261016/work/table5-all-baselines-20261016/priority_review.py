"""Export measured priority60 judging with query-bootstrap uncertainty and maintenance costs."""
import json,csv,html,time
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];O=W/'outputs/table5-all-baselines-20261016';Q=O/'quality'
DIM=[('comprehensiveness','全面性'),('diversity','多样性'),('empowerment','赋能性'),('directness','直接性')]
NAMES={'dylo':'DyLo','batch':'Batch','rebuild':'Rebuild','hit-leiden':'HIT-Leiden','dynamo':'DynaMo','qca':'QCA','df-louvain':'DF-Louvain','ds-louvain':'Δ-screen'}
def read(p):return json.loads(p.read_text())
def table(rows):
 if not rows:return '<p>等待完整60题双顺序裁决；不填入估计分数。</p>'
 keys=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
def main():
 score_rows=[];raw=[];diagnostics=[]
 for arm in ['batch','rebuild','hit-leiden','dynamo','qca']:
  p=Q/f'quality-full-{arm}.json'
  if not p.exists():continue
  result=read(p)
  if result['status']!='complete':continue
  row={'对比对象':NAMES[arm]}
  for dim,label in DIM:
   r=next(r for r in result['rows'] if r['dimension']==dim);assert r['question_count']==60 and r['missing_question_pairs']==0
   lo,hi=r['bootstrap95_percent'];row[label+'得分〔95%区间〕']=f'{r["dylo_preference_score"]:.2f} [{lo:.2f}, {hi:.2f}]';raw.append(r)
   diagnostics.append({'对比对象':NAMES[arm],'维度':label,'顺序分歧题数':r['position_disagreement_count'],'题数':60})
  score_rows.append(row)
 measured=[]
 for r in read(O/'all_results.json')['rows']:
  measured.append({'方法':NAMES[r['method']],'重摘要耗时（秒）':round(r['net_summary_s'],2),'增量维护耗时（秒）':round(r['net_graph_to_queryable_s'],2),'成功摘要输入token':r['successful_input_tokens'],'成功摘要输出token':r['successful_output_tokens'],'成功摘要总token':r['successful_total_tokens'],'失败调用':r['failed_calls'],'失败token另列':r['failed_tokens']})
 if raw:
  with (Q/'priority60_review.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 (Q/'priority60_review.json').write_text(json.dumps({'question_count':60,'complete_baseline_comparisons':len(score_rows),'quality_rows':raw,'maintenance':measured,'score':'DyLo win100/tie50/loss0; average both answer orders within question, then average60 questions','CI':'10000 query bootstrap resamples, seed17, per-dimension95% intervals','scope':'Original q0..q59, all retained including no-evidence responses; C0 global map-reduce, repaired English Qwen generation; independent local gpt-oss120b judge; single-run generation; prior15 answer and verdict reuse retained','limitations':['The C0 report exposure differs from archived Table5 matched453 level<=1 reports and250-word response template.','Graph-ready maintenance timing excludes raw-document extraction.','Preference intervals do not establish a prespecified noninferiority claim.'],'updated_at':time.strftime('%Y-%m-%d %H:%M:%S%z')},ensure_ascii=False,indent=2))
 content='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>正文候选结果：60题优先裁决</title><style>body{font:15px/1.75 sans-serif;margin:30px;max-width:1500px}table{border-collapse:collapse;margin:15px 0}th,td{border:1px solid #bcc;padding:8px}th{background:#eef}</style><h1>60题优先裁决：供正文取舍</h1><p>原实验q0–q59，全部保留；DyLo与每个baseline按四维、两种答案顺序评审，无标准答案。分数为胜100、平50、负0，先按题平均顺序，再跨60题平均；50表示持平。独立本地gpt-oss-120b裁判已通过既定校验。</p><h2>质量偏好及按题bootstrap95%区间</h2>'+table(score_rows)
 content+='<p>区间跨过50时，当前样本无法明确区分偏好方向；这不等同于证明非劣效。直接性独立作为控制维度。</p><h2>对应维护实测</h2>'+table(measured)
 content+='<p>维护token只来自成功的社区摘要更新调用，旧报告补修和失败token另列。上述时间从已抽取图增量开始，不包含原文抽取、初始304篇建库或评测。实测墙钟包含维护重试。</p><h2>顺序分歧</h2>'+table(diagnostics)
 content+='<p>当前查询使用C0全局map–reduce。旧Table5质量实验使用level≤1、各方法453份报告的匹配曝光及250词答案模板；当前结果应标为新的评测，不能直接当作旧Table5名次的复现。</p><p>优先裁决期间，DF-Louvain余下28题及Δ-screen维护／回答暂停；优先结果发布后接续原已授权实验。7种已验收维护结果和索引保留。更新时间：'+time.strftime('%Y-%m-%d %H:%M:%S%z')+'</p></html>'
 (Q/'priority60_review.html').write_text(content)
 print('Completed60-question comparisons:',len(score_rows));print(json.dumps(score_rows,ensure_ascii=False))
if __name__=='__main__':main()
