"""Shared request-ledger accounting from the measured case study."""
import json
import full_update as f
def requests(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result

def totals(rows):
    return {status: {'calls': len((z := [r for r in rows if (r['status'] == 'ok') == (status == 'successful')])), 'input_tokens': sum((r.get('prompt_tokens', 0) for r in z)), 'output_tokens': sum((r.get('completion_tokens', 0) for r in z)), 'total_tokens': sum((r.get('prompt_tokens', 0) + r.get('completion_tokens', 0) for r in z))} for status in ['successful', 'failed']}
