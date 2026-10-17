"""Actual Table5-style generation on one frozen graph, with initial debt separate."""
import os,sys,json,time,asyncio,hashlib,traceback,csv,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];F=W/'work/fast-summary-20261016';O=W/'outputs/table5-std4-20261016'
sys.path.insert(0,str(F));import quick_run as q
import pandas as pd
from isolation import Guard
from planning import build
TABLES=['documents','entities','relationships','text_units','communities','community_reports']
def load(tag):return {n:pd.read_parquet(R/'data'/tag/f'{n}.parquet') for n in TABLES}
def contexts(path):return [(int(level),pd.DataFrame(records)) for level,records in json.loads(path.read_text())]
def ledger(dest):return q.requests(dest/'requests.jsonl')
def dump(p,value):q.f.e.dump(p,value)
def stage_progress(arm,stage,**kwargs):dump(O/'progress.json',{'method':arm,'stage':stage,'updated_at':time.time(),**kwargs})
async def reports(comms,tables,ctx,dest,arm,stage):
 dest.mkdir(parents=True);stage_progress(arm,stage,planned=len(comms));times={};start=time.perf_counter()
 if len(comms):fresh,planned,failed=await q.f.b.reports(comms,tables['entities'],tables['relationships'],(F/'community_report_graph.txt').read_text(),dest,17,times,checkpoint=False,contexts=ctx)
 else:fresh=pd.DataFrame()
 elapsed=time.perf_counter()-start
 return fresh,elapsed,q.totals(ledger(dest))
async def arm_run(arm):
 dest=R/'runs'/arm
 if (dest/'verified.json').exists():return json.loads((dest/'result.json').read_text())
 assert not dest.exists(),'Refusing to resume a partial timing trial: '+str(dest)
 dest.mkdir(parents=True);cache=q.f.e.load_vector_cache();q.f.b.CONCURRENCY=8
 guard=Guard();guard.start();maintenance_started=time.perf_counter();load_started=time.perf_counter()
 base=load('base');merged=load('merged');delta=load('delta');load_s=time.perf_counter()-load_started
 stage_progress(arm,'community_update_or_rebuild')
 comms,ctx,plan,inherited,parts=build(arm,base,merged,dest/'actual_plan')
 frozen=json.loads((R/'plans'/arm/'refresh_plan.json').read_text())
 assert set(plan['changed'])==set(frozen['changed']) and set(plan['initial_missing_repairs'])==set(frozen['initial_missing_repairs'])
 assert q.f.b.context_hashes(ctx)==json.loads((R/'plans'/arm/'context_hashes.json').read_text()),'Actual contexts diverged from validated plan'
 whole=time.perf_counter()
 dump(dest/'timing_start.json',{'started_at':time.time(),'planned_merged':len(plan['changed']),'delta':26 if arm!='rebuild' else 0})
 try:
  repair,repair_s,repair_tokens=await reports(comms[comms.community.isin(plan['initial_missing_repairs'])],merged,ctx,dest/'initial_missing_repairs',arm,'initial_missing_report_repairs_excluded_from_primary')
  net_started=time.perf_counter();delta_s=0;delta_tokens=q.totals([])
  if arm!='rebuild':
   _,delta_s,delta_tokens=await reports(delta['communities'],delta,contexts(R/'plans/delta_contexts.json'),dest/'delta_summaries',arm,'new_document_delta_summaries')
  fresh,merged_s,merged_tokens=await reports(comms[comms.community.isin(plan['net_update_targets'])],merged,ctx,dest/'merged_refresh',arm,'merged_graph_member_only_refresh')
  net_wall=time.perf_counter()-net_started;whole_wall=time.perf_counter()-whole
  actual=[]
  for folder in ['initial_missing_repairs','delta_summaries','merged_refresh']:actual.extend(ledger(dest/folder))
  net_calls=ledger(dest/'delta_summaries')+ledger(dest/'merged_refresh');tokens=q.totals(actual);primary=q.totals(net_calls)
  assert tokens['successful']['calls']==len(plan['changed'])+(26 if arm!='rebuild' else 0)
  combined=pd.concat([x for x in [inherited,repair,fresh] if len(x)],ignore_index=True)
  assert combined.community.is_unique and combined.id.is_unique and set(combined.community)==set(comms.community)
  stage_progress(arm,'publishing_new_index');pub_start=time.perf_counter()
  vectors=await q.f.e.materialize({**{k:merged[k] for k in ['documents','entities','relationships','text_units']},'communities':comms,'community_reports':combined},dest/'index',cache)
  publication_s=time.perf_counter()-pub_start;maintenance_all_s=time.perf_counter()-maintenance_started
  parts.update(data_load_s=load_s,delta_summary_s=delta_s,merged_refresh_s=merged_s,embedding_write_publish_s=publication_s)
  parts['other_s']=maintenance_all_s-repair_s-sum(parts.values());assert parts['other_s']>=-0.001
  net_maintenance_s=maintenance_all_s-repair_s
  shares={k:100*v/net_maintenance_s for k,v in parts.items()};assert abs(sum(shares.values())-100)<0.0001
  isolated=guard.finish(dest/'resource_isolation.json');assert isolated,'Resource contention invalidates timing'
  row={'status':'completed','method':arm,'summary_wall_s':whole_wall,'net_update_summary_wall_s':net_wall,'initial_missing_repair_wall_s':repair_s,'delta_summary_wall_s':delta_s,'merged_refresh_wall_s':merged_s,'index_publication_s':publication_s,'tokens_table5_style_all_tasks':tokens,'tokens_net_update_primary':primary,'tokens_initial_missing_repair':repair_tokens,'tokens_delta':delta_tokens,'tokens_merged_refresh':merged_tokens,'merged_refresh_targets':len(plan['changed']),'table5_style_targets':tokens['successful']['calls'],'reused_reports':len(inherited),'final_community_reports':len(combined),'queryable':True,'concurrency':8,'generation_seed':17,'graph_semantic_sha256':q.f.semantic_hash(merged),'initial_index_setup_included':False,'extraction_included':False,'failure_tokens_in_primary':False,'retry_time_in_wall_clock':True,'model':'Qwen/Qwen2.5-7B-Instruct','no_response_cache':True,'no_prefix_cache':True,'refresh_rule':'same entity_id set and old report exists; no input-change refresh','original_std4_config':arm=='dylo','vector_tables':vectors,'timing_scope':'Actual summary generation wall time, model load/extraction/initial304build/CPUplan/publication excluded. Initial missing report repair measured separately. All tasks including repair also have one whole measured wall clock.'}
  row.update(maintenance_wall_including_initial_repair_s=maintenance_all_s,net_update_maintenance_wall_s=net_maintenance_s,phase_times_s=parts,phase_percent_of_net_maintenance=shares,
    resummary_fraction_of_net_maintenance=(delta_s+merged_s)/net_maintenance_s,raw_document_extraction='not measured: common frozen archived graph input',native_bootstrap='Original native process bootstrap is included in community-update phase; no initial GraphRAG304-document index build is performed')
  dump(dest/'result.json',row);dump(dest/'verified.json',{'status':'passed','complete_summary_coverage':True,'queryable':True,'resource_isolation':'passed'});dump(O/f'{arm}.json',row);return row
 finally:
  if not guard.stop.is_set():guard.finish(dest/'resource_isolation.json')
def export():
 rows=[];phase_rows=[]
 for arm in ['dylo','batch','rebuild']:
  p=O/f'{arm}.json'
  if not p.exists():continue
  d=json.loads(p.read_text());a=d['tokens_table5_style_all_tasks'];n=d['tokens_net_update_primary']
  phase_rows.extend({'method':arm,'phase':name,'seconds':seconds,'percent_of_net_maintenance':d['phase_percent_of_net_maintenance'][name],'denominator_s':d['net_update_maintenance_wall_s'],'initial_missing_report_repair_excluded_s':d['initial_missing_repair_wall_s']} for name,seconds in d['phase_times_s'].items())
  rows.append({'method':arm,'archived_style_targets':d['table5_style_targets'],'merged_targets':d['merged_refresh_targets'],'net_update_targets':n['successful']['calls'],'table5_style_summary_wall_s':d['summary_wall_s'],'net_update_summary_wall_s':d['net_update_summary_wall_s'],'successful_input_tokens':n['successful']['input_tokens'],'successful_output_tokens':n['successful']['output_tokens'],'successful_total_tokens':n['successful']['total_tokens'],'failed_calls':n['failed']['calls'],'failed_tokens':n['failed']['total_tokens'],'archived_style_failed_calls':a['failed']['calls'],'archived_style_failed_tokens':a['failed']['total_tokens'],'initial_missing_repair_successful_tokens':d['tokens_initial_missing_repair']['successful']['total_tokens'],'archived_style_successful_tokens':a['successful']['total_tokens'],'queryable':d['queryable']})
 dump(O/'results.json',{'scope':'All measured rows only; no imputed time/token','rows':rows})
 if rows:
  with (O/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  with (O/'phase_times.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(phase_rows[0]));w.writeheader();w.writerows(phase_rows)
 subprocess.run([sys.executable,str(R/'report.py')],check=True)
async def main():
 protocol=json.loads((O/'protocol.json').read_text());assert protocol['status']=='CPU_validated'
 for path,digest in {**protocol['unchanged_sources'],**protocol['source_sha256']}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 failures=[]
 for arm in protocol['methods']:
  try:await arm_run(arm)
  except BaseException as exc:
   if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
   failures.append({'method':arm,'error':str(exc),'traceback':traceback.format_exc()});dump(O/'arm_failures.json',failures)
  export()
 subprocess.run([sys.executable,str(R/'report.py')],check=True)
 if failures:raise RuntimeError('Incomplete trials retained separately; no quality score claimed for missing index: '+str(failures))
 stage_progress('all','summary_comparison_completed_quality_queued')
if __name__=='__main__':asyncio.run(main())
