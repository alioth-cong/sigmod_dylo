"""CPU regression checks on real failed responses and bounded English output."""
import os,asyncio,json,tempfile,re
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
import common as c,generate as g
from judge import preference
from budget import pack_reports
from jsonschema import ValidationError
old=c.R/'fixtures'
good=json.loads((old/'safe_pattern_failed_context.json').read_text());allowed=good['request']['response_format']['json_schema']['schema']['properties']['points']['items']['properties']['report_ids']['items']['enum']
assert '[Data: Reports (' in g.render_valid(good['response']['choices'][0]['message']['content'],g.schema_for(set(allowed),True),True)
assert '{1,450}' in g.schema_for({'1'},True)['properties']['points']['items']['properties']['description']['pattern']
assert '{1,650}' in g.schema_for({'1'},False)['properties']['paragraphs']['items']['properties']['text']['pattern']
duplicate='{}'
duplicate=json.dumps({'points':[{'description':'Supported insight','score':80,'report_ids':['1','1']}]})
assert json.loads(g.render_valid(duplicate,g.schema_for({'1'},True),True))['points'][0]['description']=='Supported insight [Data: Reports (1)]'
bad=json.loads((old/'original_broad_pattern_control.json').read_text())['response']['choices'][0]['message']['content']
try:g.render_valid(bad,g.schema_for(set(allowed),True),True);raise AssertionError('Original failure accepted')
except (ValueError,ValidationError):pass
class Response:
 def __init__(self,content,finish='stop'):self.content=content;self.finish=finish
 def raise_for_status(self):pass
 def json(self):return {'usage':{'prompt_tokens':10,'completion_tokens':10},'choices':[{'finish_reason':self.finish,'message':{'content':self.content}}]}
class Fake:
 def __init__(self,content,finish='stop'):self.content=content;self.finish=finish;self.calls=0
 async def post(self,*a,**kw):self.calls+=1;return Response(self.content,self.finish)
async def checks():
 with tempfile.TemporaryDirectory(prefix='sensemaking-repair-tests-') as td:
  previousR,previousO=g.R,g.O;g.R=c.Path(td)/'work';g.O=c.Path(td)/'out';g.O.mkdir()
  history=[{'role':'system','content':'---Data tables---\nid|title|content\n1|Community|Evidence\n---Goal---'}]
  valid='{"points":[{"description":"Supported insight","score":80,"report_ids":["1"]}]}'
  client=Fake(valid);m=g.Model(client,'test',1,['1']);await m.call('Q',history,{},True);await m.call('Q',history,{},True);assert client.calls==1 and m.cache_hits==1
  for i,(text,finish) in enumerate([(valid+'<tool_call>','stop'),(valid,'length'),(valid.replace('["1"]','["2"]'),'stop')]):
   client=Fake(text,finish);m=g.Model(client,'test',i+10,['1'])
   try:await m.call('Q',history,{},True);raise AssertionError('Invalid response accepted')
   except RuntimeError:pass
   assert client.calls==3 and len(m.exhausted)==1
  g.R,g.O=previousR,previousO
asyncio.run(checks())
valid_final=json.dumps({'paragraphs':[{'text':'A supported interpretation.','report_ids':['1']} for _ in range(3)]});assert g.render_valid(valid_final,g.schema_for({'1'},False),False).count('[Data: Reports (1)]')==3
assert preference({'status':'error','winner':None},True) is None
assert preference({'status':'ok','winner':1},True)==preference({'status':'ok','winner':2},False)
c.dump(c.O/'cpu_validation.json',{'status':'passed','protocol':g.PROTOCOL,'gpu_calls':0,'real_model_calls':0,'frozen_questions_sha256':c.sha(c.O/'questions.json'),'pilot_count':15,
 'checks':['real safe-regex probe response validates','real original broad-regex failure rejected','truncated finish rejected despite complete-lookingJSON','tool_call marker rejected without stripping','unknown sourceID fails all bounded retries','validated checkpoints only','structured final paragraphs mechanically cite only allowed evidence','failed judge call is never a tie'],
 'code_sha256':{p.name:c.sha(p) for p in c.R.glob('*.py')}})
print('JSONrepairv2 CPU regression checks passed',flush=True)
