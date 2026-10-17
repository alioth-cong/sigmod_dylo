from pathlib import Path
from tempfile import TemporaryDirectory
from graphrag_adapter import Report,plan,apply,require_native_coverage,read_native

a=(0,1);b=(0,2);r=(1,3)
old={a:Report('a',frozenset({'x','y'}),{'x':'old','y':'old'},a,1),b:Report('b',frozenset({'z'}),{'z':'old'},b,2),r:Report('r',frozenset({'x','y','z'}),{'x':'old','y':'old','z':'old'},r,3)}
members={a:frozenset({'y'}),b:frozenset({'z'}),r:frozenset({'y','z'})}
evidence={a:{'y':'old'},b:{'z':'old'},r:{'y':'old','z':'old'}}
mapping={x:x for x in old};parents={a:r,b:r,r:None}
deferred={a:(.1,.102,1)}
tasks=plan(old,members,evidence,mapping,{r:4},deferred,parents)
assert {t.node:t.regenerate for t in tasks}=={a:False,b:False,r:True}
assert require_native_coverage(tasks,mapping,{r:4})==set()
tasks_bad=plan(old,members,evidence,mapping,{r:4},{a:(.1,.102,99)},parents)
assert next(t for t in tasks_bad if t.node==a).regenerate
assert a in require_native_coverage(tasks_bad,mapping,{r:4})
changed=dict(evidence);changed[b]={'z':'changed'}
tasks_changed=plan(old,members,changed,mapping,{r:4},deferred,parents)
assert next(t for t in tasks_changed if t.node==b).regenerate
with TemporaryDirectory() as d:
 p=Path(d)/'index.json';p.write_text('old-index')
 def fail(*args):raise RuntimeError('model failure')
 try:apply(old,tasks,members,evidence,mapping,{r:4},fail,p)
 except RuntimeError:pass
 else:raise AssertionError('failure expected')
 assert p.read_text()=='old-index'
 result=apply(old,tasks,members,evidence,mapping,{r:4},lambda *args:'new-report',p)
 assert result[a].represented_members==frozenset({'x','y'})
 assert result[r].represented_members==frozenset({'y','z'}) and result[r].native_version==4
 assert p.read_text()!='old-index'
print('PASS adapter: native deferral, report epoch, evidence, ancestor propagation, failed-generation rollback, atomic publication')

# GraphRAG can number roots before leaves; native levels use the opposite order.
graph_root=(0,30);graph_child=(1,10)
native_child=(0,10);native_root=(1,30)
mapping={graph_root:native_root,graph_child:native_child}
members={graph_root:frozenset({'x'}),graph_child:frozenset({'x'})}
evidence={node:{'x':'evidence'} for node in members}
refresh={native_root:9,native_child:8}
tasks=plan({},members,evidence,mapping,refresh,{},
           {graph_child:graph_root,graph_root:None})
order=[]
def generate(node,current,evidence,staged):
 if node==graph_root:assert staged[graph_child].text=='child-report'
 order.append(node)
 return 'child-report' if node==graph_child else 'root-report'
with TemporaryDirectory() as d:
 apply({},tasks,members,evidence,mapping,refresh,generate,Path(d)/'index.json')
assert order==[graph_child,graph_root]
print('PASS adapter: reversed GraphRAG levels generate native children before parents')

members={a:frozenset({'y'}),b:frozenset({'z'}),r:frozenset({'y','z'})}
evidence={a:{'y':'old'},b:{'z':'old'},r:{'y':'old','z':'old'}}
# Evidence refresh has its own epoch and must not settle structural liabilities.
with TemporaryDirectory() as d:
 result=apply(old,tasks_changed,members,changed,{x:x for x in old},{r:4},lambda *args:'validated-report',Path(d)/'index.json',supplemental={b:1})
 assert result[b].native_version==old[b].native_version
 assert result[b].evidence_version==1
 assert result[a]==old[a]
 try:apply(old,tasks_bad,members,evidence,{x:x for x in old},{r:4},lambda *args:'wrong',Path(d)/'bad.json',supplemental={a:1})
 except ValueError:pass
 else:raise AssertionError('evidence epoch cannot cover unscheduled structural change')
print('PASS adapter: evidence epoch separate from structural epoch, unscheduled structure rejected')

# An unchanged-native parent can retain the same certified source removal.
parent_tasks=plan(old,members,evidence,{x:x for x in old},{},deferred,parents={a:r,b:r,r:None})
assert all(not t.regenerate for t in parent_tasks)
assert next(t.reason for t in parent_tasks if t.node==r)=='native_deferred_ancestor'
# Evidence materialization must not silently remove the deferred member.
new_evidence={a:{'x':'old','y':'changed'},b:{'z':'old'},r:{'x':'old','y':'changed','z':'old'}}
update_tasks=plan(old,members,new_evidence,{x:x for x in old},{},deferred,{a:r,b:r,r:None})
seen={}
def evidence_generate(node,current,evidence,staged):
 seen[node]=current
 return 'evidence-only-report'
with TemporaryDirectory() as d:
 staged=apply(old,update_tasks,members,new_evidence,{x:x for x in old},{},evidence_generate,Path(d)/'index.json',supplemental={a:1,r:1})
assert seen[a]==old[a].represented_members and seen[r]==old[r].represented_members
assert staged[a].native_version==old[a].native_version and staged[a].evidence_version==1
print('PASS adapter: certified source ancestors retained; evidence updates preserve deferred member basis and structural epoch')
parent=(2,99)
reference=frozenset({'A','B','C'});current=frozenset({'A','B'})
previous={parent:Report('parent report',reference,{},(2,99),900)}
views={parent:current};mapping={parent:(2,99)};src=(0,7)
covered=plan(previous,views,{parent:{}},mapping,{}, {src:(1.,1.04,111)}, {parent:None},native_source_bases={src:(reference,current,111)})
assert not covered[0].regenerate and covered[0].reason=='native_deferred_ancestor'
bad=plan(previous,views,{parent:{}},mapping,{}, {src:(1.,1.04,111)}, {parent:None},native_source_bases={src:(reference,current,112)})
assert bad[0].regenerate
print('PASS adapter: projected ancestor can retain certified unmapped source leaf; source epoch mismatch rejected')
