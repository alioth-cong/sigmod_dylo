"""Enforce evidence budgets on serialized data, preserving complete report rows."""
import csv,io,re
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext

def pack_reports(chunks,tokenizer,budget=8000,delimiter='|'):
    rows=[];fields=None
    for chunk in chunks:
        reader=csv.DictReader(io.StringIO(chunk),delimiter=delimiter)
        if fields is None:fields=reader.fieldnames
        assert fields==reader.fieldnames
        rows.extend(list(reader))
    def render(batch):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,delimiter=delimiter,lineterminator='\n')
        writer.writeheader();writer.writerows(batch);return stream.getvalue()
    packed=[];batch=[]
    for row in rows:
        candidate=render(batch+[row])
        if tokenizer.num_tokens(candidate)>budget:
            assert batch,'A complete report exceeds the evidence budget'
            packed.append(render(batch));batch=[row]
            assert tokenizer.num_tokens(render(batch))<=budget,'A complete report exceeds the evidence budget'
        else:batch.append(row)
    if batch:packed.append(render(batch))
    assert [r for chunk in packed for r in csv.DictReader(io.StringIO(chunk),delimiter=delimiter)]==rows
    return packed

class BudgetedGlobalContext(GlobalCommunityContext):
    async def build_context(self,*args,**kwargs):
        result=await super().build_context(*args,**kwargs)
        result.context_chunks=pack_reports(result.context_chunks,self.tokenizer,kwargs.get('max_context_tokens',8000),kwargs.get('column_delimiter','|'))
        return result

def bounded_reduce(prompt,tokenizer,budget=8000):
    start=prompt.index('---Analyst Reports---')+len('---Analyst Reports---')
    end=prompt.index('---Goal---',start)
    evidence=prompt[start:end].strip()
    blocks=re.split(r'\n\n(?=----Analyst \d+----)',evidence)
    while tokenizer.num_tokens('\n\n'.join(blocks))>budget:
        assert len(blocks)>1,'One analyst point exceeds the evidence budget'
        blocks.pop()
    evidence='\n\n'.join(blocks)
    return prompt[:start]+'\n\n'+evidence+'\n\n\n'+prompt[end:],evidence
