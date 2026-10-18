"""Check citations against actual prompt CSV records, excluding examples.

This is a structural and ID-grounding gate, not a semantic factuality judge.
Long references to real IDs are recorded as format warnings.
"""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import csv
import io
import re
from collections import Counter

def allowed_ids(prompt):
    real = prompt.rsplit('# Real Data', 1)[-1]
    sections = list(re.finditer('-----([A-Za-z_ ]+)-----\\s*\\n', real))
    ids = {}
    for (index, match) in enumerate(sections):
        name = match.group(1).strip()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(real)
        text = real[match.end():end].split('\nThe report should include')[0]
        rows = csv.DictReader(io.StringIO(text.strip()))
        ids[name] = {int(row.get('human_readable_id', row.get('id', ''))) for row in rows if str(row.get('human_readable_id', row.get('id', ''))).isdigit()}
    return ids

def audit(parsed, prompt):
    ids = allowed_ids(prompt)
    strings = [parsed.title, parsed.summary, parsed.rating_explanation]
    for finding in parsed.findings:
        strings += [finding.summary, finding.explanation]
    (errors, warnings, mentions) = ([], [], [])
    for text in strings:
        citations = re.findall('\\[Data:\\s*([^\\]]*)\\]', text, flags=re.I)
        if len(citations) != len(re.findall('\\[Data:', text, flags=re.I)):
            errors.append('unclosed_data_citation')
        for citation in citations:
            refs = list(re.finditer('([A-Za-z_][A-Za-z_ ]*)\\s*\\(([^()]*)\\)', citation))
            if not refs:
                errors.append('malformed_data_citation')
                continue
            rest = citation
            for ref in reversed(refs):
                rest = rest[:ref.start()] + rest[ref.end():]
            if rest.replace('+more', '').strip(' ,;\n\t'):
                errors.append('unparsed_data_citation:' + rest[:100])
            for ref in refs:
                (name, values) = (ref.group(1).strip(), ref.group(2))
                if name not in ids:
                    errors.append('absent_dataset:' + name)
                    continue
                normalized = re.sub('[‑–—−]', '-', values)
                if ';' in normalized:
                    normalized = normalized.replace(';', ',')
                    warnings.append('semicolon_id_separator')
                atom = '\\d+(?:\\s*-\\s*\\d+)?'
                if not re.fullmatch('\\s*' + atom + '(?:\\s*,\\s*' + atom + ')*(?:\\s*,?\\s*\\+more)?\\s*', normalized):
                    errors.append('malformed_record_ids:' + values[:100])
                    continue
                ns = []
                for (first, last) in re.findall('(\\d+)(?:\\s*-\\s*(\\d+))?', normalized):
                    first = int(first)
                    if last:
                        last = int(last)
                        if last < first or last - first > 10000:
                            errors.append('invalid_id_range')
                            continue
                        ns.extend(range(first, last + 1))
                        warnings.append('range_reference')
                    else:
                        ns.append(first)
                mentions += [(name, n) for n in ns]
                unsupported = sorted(set(ns) - ids[name])
                if unsupported:
                    errors.append(f'unsupported_ids:{name}:{unsupported[:20]}')
                if len(ns) > 5:
                    warnings.append('overlong_reference')
                if len(ns) >= 20 and max(Counter(ns).values(), default=0) >= 5:
                    errors.append('citation_number_loop')
        if re.search('((?:\\d+\\s*,\\s*){2,12})\\1\\1\\1', text):
            errors.append('repeated_number_chain')
    return {'valid': not errors, 'errors': sorted(set(errors)), 'warnings': dict(Counter(warnings)), 'id_mentions': len(mentions), 'input_id_counts': {name: len(values) for (name, values) in ids.items()}}
