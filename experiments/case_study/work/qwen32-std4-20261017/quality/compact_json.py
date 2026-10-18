"""Keep the same JSON schema while bounding whitespace on technical retries."""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import xgrammar as xgr

def compact_json_retry(request):
    result = dict(request)
    schema = result.pop('response_format')['json_schema']['schema']
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False, separators=(',', ':'))
    result['structured_outputs'] = {'grammar': str(grammar)}
    return result
