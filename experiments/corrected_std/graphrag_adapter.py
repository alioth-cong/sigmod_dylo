"""Consume the corrected native refresh plan without inferring it from churn.

The caller supplies an explicit GraphRAG-to-native node mapping and evidence
records from its actual context builder. Unmapped nodes refresh conservatively.
The adapter does not make LLM calls or estimate tokens. Publication is atomic:
failed generation leaves the old report index and epoch metadata untouched.
"""
from dataclasses import dataclass
from typing import Callable,Mapping
import csv,json,math,os,tempfile
from pathlib import Path

Node=tuple[int,int]
@dataclass(frozen=True)
class Report:
    text:str
    represented_members:frozenset[str]
    evidence:Mapping[str,str]
    native_node:Node|None
    native_version:int|None
    evidence_version:int=0

@dataclass(frozen=True)
class Task:
    node:Node
    regenerate:bool
    reason:str

def read_native(directory:Path,batch:int):
    refresh={}
    with (directory/'refresh_plan.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if int(r['batch'])==batch:refresh[(int(r['level']),int(r['community']))]=int(r['valuation_version'])
    deferred={}
    with (directory/'source_state.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if int(r['batch'])==batch and int(r['deferred']):
                deferred[(0,int(r['source']))]=(float(r['q_ref']),float(r['q_now']),int(r['report_version']))
    return refresh,deferred

def plan(old:Mapping[Node,Report],members:Mapping[Node,frozenset[str]],evidence:Mapping[Node,Mapping[str,str]],native_mapping:Mapping[Node,Node],refresh:Mapping[Node,int],deferred:Mapping[Node,tuple[float,float,int]],parents:Mapping[Node,Node|None],epsilon:float=.05,native_source_bases=None):
    tasks={}
    # Parent reports may retain removals certified at their deferred source
    # leaves. Keep the report's member basis until native consolidation.
    current_by_native={native_mapping[n]:m for n,m in members.items() if n in native_mapping}
    deferred_sources=[]
    # SDK ragged/LCC projection may omit the native source leaf report. Its
    # logical reference basis must then come from native snapshots/epochs,
    # rather than substituting the ancestor's different epoch number.
    for native,(reference_basis,current_basis,reference_version) in (native_source_bases or {}).items():
        if native not in deferred:continue
        reference,now,version=deferred[native]
        drift=abs(now-reference)/abs(reference) if reference else (0 if not now else math.inf)
        if native[0]==0 and drift<epsilon and version==reference_version and current_basis<reference_basis:
            deferred_sources.append((reference_basis,reference_basis-current_basis))
    for previous in old.values():
        native=previous.native_node
        if native not in deferred or native not in current_by_native:continue
        reference,now,version=deferred[native]
        drift=abs(now-reference)/abs(reference) if reference else (0 if not now else math.inf)
        current=current_by_native[native]
        if drift<epsilon and version==previous.native_version and current<previous.represented_members:
            deferred_sources.append((previous.represented_members,previous.represented_members-current))
    for node,current in members.items():
        previous=old.get(node);native=native_mapping.get(node)
        if previous is None:reason='new_report'
        elif native is None or previous.native_node!=native:reason='unmapped_native_report'
        elif native in refresh:reason='native_refresh_plan'
        else:
            source_deferred=False;ancestor_deferred=False
            if current<previous.represented_members and native in deferred:
                reference,now,version=deferred[native]
                drift=abs(now-reference)/abs(reference) if reference else (0 if not now else math.inf)
                if drift>=epsilon or version!=previous.native_version:
                    tasks[node]=Task(node,True,'source_drift_or_report_epoch_mismatch');continue
                source_deferred=True
            elif current<previous.represented_members and native[0]>0:
                removals=set()
                for source_basis,lost in deferred_sources:
                    if source_basis<=previous.represented_members:removals.update(lost)
                ancestor_deferred=(previous.represented_members-current)<=removals
            if any(previous.evidence.get(k)!=v for k,v in evidence[node].items()) or (current==previous.represented_members and dict(previous.evidence)!=dict(evidence[node])):
                reason='evidence_changed'
            elif current==previous.represented_members:
                tasks[node]=Task(node,False,'unchanged_members_and_evidence');continue
            elif source_deferred or ancestor_deferred:
                tasks[node]=Task(node,False,'native_deferred_source' if source_deferred else 'native_deferred_ancestor');continue
            else:reason='membership_change_not_native_deferred'
        tasks[node]=Task(node,True,reason)
    # A child report affects a parent consuming it, even if logical member
    # ids did not change. Extra conservative tasks are visible to the caller
    # and must not be claimed as native planned execution in cost accounting.
    for node,task in list(tasks.items()):
        if not task.regenerate:continue
        parent=parents.get(node)
        seen=set()
        while parent is not None:
            if parent in seen:raise ValueError('cyclic report hierarchy')
            seen.add(parent)
            if parent in tasks and not tasks[parent].regenerate:tasks[parent]=Task(parent,True,'changed_child_report')
            parent=parents.get(parent)
    return list(tasks.values())

def require_native_coverage(tasks:list[Task],native_mapping:Mapping[Node,Node],refresh:Mapping[Node,int]):
    """Return reports outside the native structural plan for separate auditing.

    Evidence-only refreshes must NOT be fed back into structural cost planning.
    """
    needed=set()
    for task in tasks:
        if not task.regenerate:continue
        native=native_mapping.get(task.node)
        if native is None:raise ValueError('explicit native mapping required for ledger-aligned publication')
        if native not in refresh:needed.add(native)
    return needed

def apply(old:Mapping[Node,Report],tasks:list[Task],members:Mapping[Node,frozenset[str]],evidence:Mapping[Node,Mapping[str,str]],native_mapping:Mapping[Node,Node],refresh:Mapping[Node,int],generate:Callable[[Node,frozenset[str],Mapping[str,str],Mapping[Node,Report]],str],publish:Path,supplemental:Mapping[Node,int]|None=None):
    needed=require_native_coverage(tasks,native_mapping,refresh)
    supplemental=supplemental or {}
    reasons={native_mapping[t.node]:t.reason for t in tasks if t.regenerate}
    for native in needed:
        if native not in supplemental or reasons[native] not in {'evidence_changed','changed_child_report'}:
            raise ValueError(f'unscheduled structural refresh or missing separate evidence epoch: {native}')
    if len(set(native_mapping.values()))!=len(native_mapping):raise ValueError('native report mapping must be one-to-one')
    for node in members:
        native=native_mapping[node]
        previous=old.get(node)
        if native in supplemental and supplemental[native]<=(previous.evidence_version if previous else 0):
            raise ValueError('evidence epoch must increase')
    staged={node:old[node] for node in members if node in old}
    # Native level zero is finest, so children complete before their parents.
    for task in sorted(tasks,key=lambda t:(native_mapping[t.node],t.node)):
        if not task.regenerate:continue
        previous=old.get(task.node)
        report_members=members[task.node]
        if previous is not None and native_mapping[task.node] not in refresh and task.reason in {'evidence_changed','changed_child_report'} and report_members<previous.represented_members:
            report_members=previous.represented_members
        text=generate(task.node,report_members,evidence[task.node],staged)
        if not isinstance(text,str) or not text.strip():raise ValueError('report generation failed')
        native=native_mapping.get(task.node)
        previous=old.get(task.node)
        structural_epoch=refresh.get(native,previous.native_version if previous else None)
        evidence_epoch=supplemental.get(native,previous.evidence_version if previous else 0)
        staged[task.node]=Report(text,report_members,dict(evidence[task.node]),native,structural_epoch,evidence_epoch)
    payload=[]
    for node,report in sorted(staged.items()):
        payload.append({'node':node,'text':report.text,'represented_members':sorted(report.represented_members),'evidence':dict(report.evidence),'native_node':report.native_node,'native_version':report.native_version,'evidence_version':report.evidence_version})
    publish.parent.mkdir(parents=True,exist_ok=True)
    fd,path=tempfile.mkstemp(prefix='.report-index-',dir=publish.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(payload,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(path,publish)
    finally:
        if os.path.exists(path):os.unlink(path)
    return staged
