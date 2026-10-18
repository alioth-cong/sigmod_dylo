"""Source-constrained structured references, mechanically rendered as citations.

Plain report strings have no regex/length bounds. Selecting a real source ID
does not establish whether a claim is actually supported by that record.
"""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import copy
from citations import allowed_ids

def source_schema(schema, prompt):
    schema = copy.deepcopy(schema)
    choices = []
    for (name, ids) in sorted(allowed_ids(prompt).items()):
        if not ids:
            continue
        choices.append({'type': 'object', 'properties': {'dataset': {'type': 'string', 'const': name}, 'ids': {'type': 'array', 'items': {'type': 'integer', 'enum': sorted(ids)}, 'minItems': 1, 'maxItems': 5}, 'more': {'type': 'boolean'}}, 'required': ['dataset', 'ids', 'more'], 'additionalProperties': False})
    finding = schema['$defs']['FindingModel']
    finding['properties']['citations'] = {'type': 'array', 'items': {'anyOf': choices}, 'minItems': 1, 'maxItems': 2} if choices else {'type': 'array', 'maxItems': 0}
    finding['required'].append('citations')
    return schema

def render_report(value):
    """Preserve model prose and model-chosen IDs; only format the references."""
    result = copy.deepcopy(value)
    for finding in result['findings']:
        references = finding.pop('citations')
        parts = []
        for reference in references:
            ids = ', '.join((str(i) for i in reference['ids']))
            if reference['more']:
                ids += ', +more'
            parts.append(reference['dataset'] + ' (' + ids + ')')
        if parts:
            finding['explanation'] += ' [Data: ' + '; '.join(parts) + ']'
    return result
