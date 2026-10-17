"""Validate publication schema and citation IDs; never silently repair content.
This checks syntactic grounding. It does not judge whether prose follows evidence.
"""
import csv,io,json,re,math
from pathlib import Path
LABELS={'entities':'Entities','relationships':'Relationships','reports':'Reports','claims':'Claims'}
def context_ids(context):
 if '# Real Data' in context:context=context.split('# Real Data',1)[1].split('# Grounding Rules',1)[0]
 parts=re.split(r'(?im)^\s*-{2,}\s*(Entities|Relationships|Reports|Claims)\s*-{2,}\s*$',context)
 ids={name:set() for name in LABELS.values()}
 for i in range(1,len(parts),2):
  name=LABELS[parts[i].lower()];table=parts[i+1].strip()
  for row in csv.DictReader(io.StringIO(table)):
   value=row.get('human_readable_id',row.get('id',row.get('community','')))
   if str(value).strip().isdigit():ids[name].add(int(value))
 return ids

def validate(text,allowed,max_words=2000):
 # Literal JSON is required: a fenced or length-truncated answer is not published.
 data=json.loads(text)
 if not isinstance(data,dict) or set(data)!={'title','summary','rating','rating_explanation','findings'}:raise ValueError('invalid report keys')
 for k in ['title','summary','rating_explanation']:
  if not isinstance(data[k],str) or not data[k].strip():raise ValueError('empty report field '+k)
 rating=data['rating']
 if isinstance(rating,bool) or not isinstance(rating,(int,float)) or not math.isfinite(rating) or not 0<=rating<=10:raise ValueError('invalid rating')
 findings=data['findings']
 if not isinstance(findings,list) or not 1<=len(findings)<=10:raise ValueError('invalid findings')
 strings=[data['title'],data['summary'],data['rating_explanation']];references=0
 for f in findings:
  if not isinstance(f,dict) or set(f)!={'summary','explanation'} or any(not isinstance(f[k],str) or not f[k].strip() for k in f):raise ValueError('invalid finding')
  if '[Data:' not in f['explanation']:raise ValueError('finding lacks evidence reference')
  strings.extend([f['summary'],f['explanation']])
 for s in strings:
  spans=list(re.finditer(r'\[Data:\s*([^\[\]]+)\]',s))
  if s.count('[Data:')!=len(spans):raise ValueError('unclosed data citation')
  for span in spans:
   citation=span.group(1);refs=list(re.finditer(r'(Entities|Relationships|Reports|Claims)\s*\(([^()]*)\)',citation))
   remainder=re.sub(r'(Entities|Relationships|Reports|Claims)\s*\([^()]*\)','',citation)
   if not refs or remainder.strip(' ;,\t\n'):raise ValueError('malformed data citation')
   for ref in refs:
    name,values=ref.groups();items=[x.strip() for x in values.split(',')];numbers=[]
    for item in items:
     if item=='+more':continue
     if not item.isdigit():raise ValueError('invalid citation ID')
     numbers.append(int(item))
    if not 1<=len(numbers)<=5 or len(set(numbers))!=len(numbers):raise ValueError('invalid citation cardinality')
    if any(n not in allowed.get(name,set()) for n in numbers):raise ValueError('citation ID absent from actual context')
    references+=len(numbers)
 if sum(len(s.split()) for s in strings)>max_words:raise ValueError('report exceeds word budget')
 return data,{'id_references':references,'findings':len(findings)}

def usage_rows(requests):
 # Successful UPDATE reports only for paper totals; all attempts for resource accounting.
 paper={'input':0,'output':0};all_attempts={'input':0,'output':0};categories={};unknown=0
 for r in requests:
  usage=r.get('usage') or r;unknown+=not ('prompt_tokens' in usage and 'completion_tokens' in usage);category=r['category'];i,o=int(usage.get('prompt_tokens',0)),int(usage.get('completion_tokens',0))
  all_attempts['input']+=i;all_attempts['output']+=o
  c=categories.setdefault(category,{'input':0,'output':0,'successful_reports':0,'failed_attempts':0})
  if r.get('status')=='ok':
   c['input']+=i;c['output']+=o;c['successful_reports']+=1
   if category in ['structural_refresh','evidence_refresh','projection_refresh']:paper['input']+=i;paper['output']+=o
  else:c['failed_attempts']+=1
 return {'paper_successful_update_tokens':paper,'all_attempt_resource_tokens':all_attempts,'unknown_usage_attempts':unknown,'resource_token_totals_complete':unknown==0,'categories':categories,'scope':'successful structural, evidence, and SDK projection updates included in actual maintenance tokens; initial repair, failed requests, and queries excluded; pure structural native cost remains solely in ledger.tsv'}
