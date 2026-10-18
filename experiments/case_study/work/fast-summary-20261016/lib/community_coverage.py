"""Preserve edge-free communities without changing clustering or adding edges."""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import hashlib
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5
import pandas as pd

def restore_empty_communities(communities, clusters, entities, relationships):
    """Restore every raw cluster omitted by GraphRAG's internal-edge inner join."""
    result = communities.copy()
    present = set(map(int, result.community))
    by_title = {title: frame for (title, frame) in entities.groupby('title', sort=False)}
    additions = []
    for (level, cid, parent, names) in clusters:
        cid = int(cid)
        if cid in present:
            continue
        names = set(names)
        if not names or not names <= set(by_title):
            raise ValueError(f'Unresolved entity membership in community {cid}')
        internal = relationships.source.isin(names) & relationships.target.isin(names)
        if internal.any():
            raise ValueError(f'Community {cid} was lost despite having internal edges')
        rows = pd.concat([by_title[n] for n in sorted(names)])
        additions.append(dict(id=uuid5(NAMESPACE_URL, f'edge-free-community/{level}/{cid}').hex, human_readable_id=cid, community=cid, level=int(level), parent=int(parent), title=f'Community {cid}', entity_ids=sorted(set(rows.id)), relationship_ids=[], text_unit_ids=sorted({t for ids in rows.text_unit_ids for t in ids}), children=[], size=len(rows), period=datetime.now(timezone.utc).date().isoformat()))
        present.add(cid)
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True)
    children = result.groupby('parent').community.apply(list).to_dict()
    result['children'] = result.community.map(lambda c: sorted(children.get(c, [])))
    if not result.community.is_unique:
        raise ValueError('Community IDs must be unique')
    if not set(result.parent) - {-1} <= set(result.community):
        raise ValueError('Unresolved community parent')
    return result.loc[:, communities.columns]

def create_communities(entities, relationships, max_cluster_size, use_lcc, seed=None):
    """Wrap the installed workflow, keeping its existing rows unchanged."""
    import graphrag.index.workflows.create_communities as workflow
    original = workflow.cluster_graph
    captured = []

    def capture(*args, **kwargs):
        clusters = original(*args, **kwargs)
        captured.extend(clusters)
        return clusters
    workflow.cluster_graph = capture
    try:
        result = workflow.create_communities(entities, relationships, max_cluster_size, use_lcc, seed)
    finally:
        workflow.cluster_graph = original
    return restore_empty_communities(result, captured, entities, relationships)

def complete_archived_root_coverage(communities, entities, relationships):
    """Add entity-only roots for uncovered LCC vertices in the shared old index.

    The archived pre-materialization hierarchy is unavailable. Preserve its
    existing partition and explicitly treat uncovered vertices as singleton
    roots; this is initial preparation, never document-update work.
    """
    from graphrag.index.operations.create_graph import create_graph
    from graphrag.index.utils.stable_lcc import stable_largest_connected_component
    graph = stable_largest_connected_component(create_graph(relationships, edge_attr=['weight']))
    names = dict(zip(entities.id, entities.title))
    covered = {names[i] for ids in communities.loc[communities.level == 0, 'entity_ids'] for i in ids}
    used = set(map(int, communities.community))
    clusters = []
    for title in sorted(set(graph) - covered):
        cid = int.from_bytes(hashlib.blake2b(('initial-edge-free-root|' + title).encode(), digest_size=6).digest(), 'big')
        if cid in used:
            raise ValueError('Restored initial community ID collision')
        clusters.append((0, cid, -1, [title]))
    return restore_empty_communities(communities, clusters, entities, relationships)

def entity_only_context(row, entities, tokenizer, budget=4000):
    """Include real entity evidence even when the edge-driven builder is empty."""
    selected = entities.set_index('id').loc[list(row.entity_ids)].sort_values('human_readable_id')
    records = [{'id': int(r.human_readable_id), 'entity': str(r.title), 'description': str(r.description), 'degree': int(r.degree)} for r in selected.itertuples()]

    def render(values):
        return '-----Entities-----\n' + pd.DataFrame(values).to_csv(index=False)
    text = render(records)
    while tokenizer.num_tokens(text) > budget:
        longest = max(records, key=lambda r: len(r['description']))
        if not longest['description']:
            raise ValueError('Entity identifiers alone exceed the context budget')
        longest['description'] = longest['description'][:len(longest['description']) // 2]
        text = render(records)
    return text
