import copy,json
from report_validation import validate,context_ids,usage_rows
context='-----Entities-----\nhuman_readable_id,title,description\n1,A,Fact\n2,B,Fact\n\n-----Relationships-----\nhuman_readable_id,source,target\n7,A,B\n'
allowed=context_ids(context);assert allowed['Entities']=={1,2} and allowed['Relationships']=={7}
r={'title':'A and B','summary':'Facts','rating':2,'rating_explanation':'Connected','findings':[{'summary':'Connected','explanation':'A is connected to B [Data: Entities (1, 2); Relationships (7)].'}]}
validate(json.dumps(r),allowed)
for content in ['```json\n'+json.dumps(r)+'\n```',json.dumps(r)[:-1],json.dumps(r).replace('Relationships (7)','Relationships (999)'),json.dumps(r).replace('Entities (1, 2)','Entities (1, 2, 3, 4, 5, 6)'),json.dumps(r).replace('Relationships (7)]','Relationships (7)'),json.dumps(r).replace('[Data: Entities (1, 2); Relationships (7)]','No references')]:
 try:validate(content,allowed)
 except (ValueError,KeyError):pass
 else:raise AssertionError('invalid report accepted')
records=[{'category':c,'status':status,'usage':{'prompt_tokens':10,'completion_tokens':5}} for c,status in [('initial_repair','ok'),('structural_refresh','error'),('structural_refresh','ok'),('evidence_refresh','ok'),('query_answer','ok')]]
u=usage_rows(records);assert u['paper_successful_update_tokens']=={'input':20,'output':10} and u['all_attempt_resource_tokens']=={'input':50,'output':25}
projection=usage_rows([{'category':'projection_refresh','status':'ok','usage':{'prompt_tokens':8,'completion_tokens':4}}, {'category':'evidence_refresh','status':'error'}])
assert projection['paper_successful_update_tokens']=={'input':8,'output':4}
assert projection['unknown_usage_attempts']==1 and not projection['resource_token_totals_complete']
print('PASS report publication: literal JSON, context citation IDs, truncation/grounding rejection, separate token scopes')
