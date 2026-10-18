"""English-only Qwen report generation without prose truncation or ID repair."""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import asyncio
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
import httpx
from jsonschema import Draft202012Validator
from citations import audit, allowed_ids
from citation_schema import source_schema, render_report
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'outputs/qwen-repair-20261014'
TEMPERATURES = [0, 0.3, 0.7, 0.3, 0.7, 1.0, 0.2, 0.5]
ASCII_PROSE = '^[ !#-Z^-~]*$'
NUMERIC_RUN = '[, ]*(?:[0-9]+[, ]+){0,18}(?:[0-9]+[, ]*)?'
BOUNDED_PROSE = '^(?:' + NUMERIC_RUN + '[!#-+./:-Z^-~-])*' + NUMERIC_RUN + '$'
ASCII_ENGLISH = '^[ !#-Z' + re.escape('[]') + '^-~]*$'
ENGLISH = 'Write the entire report in English. Keep the original comprehensive report task, the original maximum of 2000 words, and five to ten substantive findings. Use English spellings or transliterations for names. Do not switch languages. Use plain ASCII punctuation and single quotes for quotations. Do not use double quotes, backslashes or line breaks inside prose fields.'
STRUCTURED = ENGLISH + " Citation formatting override: choose source references ONLY in each finding's structured citations array. Select relevant human_readable_id values from the actual source CSV. The schema limits selections to those supplied IDs. Never write citation lists, record IDs, square brackets, or [Data: ...] in prose. Use plain English ASCII text and punctuation in every prose field. Explain evidence and relationships in natural language. Do not enumerate numbers or repeat text. Never place a comma-separated source-number list in a prose field. The application will format your selected references; all other original report requirements continue to apply."

def schema_for(original, prompt):
    schema = source_schema(original, prompt)
    for props in (schema['properties'], schema['$defs']['FindingModel']['properties']):
        for prop in props.values():
            if prop.get('type') == 'string':
                prop['pattern'] = BOUNDED_PROSE
    schema['properties']['findings'].update(minItems=5, maxItems=10)
    return schema

def control_schema(original):
    schema = copy.deepcopy(original)
    for props in (schema['properties'], schema['$defs']['FindingModel']['properties']):
        for prop in props.values():
            if prop.get('type') == 'string':
                prop['pattern'] = ASCII_ENGLISH
    schema['properties']['findings'].update(minItems=5, maxItems=10)
    return schema

def normalized_prompt(prompt):
    """Replace contradictory inline-citation instructions; keep evidence intact."""
    rules = '# Grounding Rules\n\nUse only evidence provided in the Real Data source tables. Do not invent facts. Select relevant human_readable_id values in the structured citations array of each finding. References must appear only in that array, never as record-number lists in prose. Choose at most five IDs in each reference group; use the more flag where appropriate. Illustrative examples describe report content; the current response schema governs its format. Keep the original comprehensive report requirements and five to ten findings. Limit the total report to 2000 words. Return complete English JSON.\n\n'
    headings = list(re.finditer('(?m)^# [^\\n]+$', prompt))
    spans = []
    for (index, heading) in enumerate(headings):
        if heading.group(0).strip() == '# Grounding Rules':
            end = headings[index + 1].start() if index + 1 < len(headings) else len(prompt)
            spans.append((heading.start(), end))
    assert spans, 'Expected original grounding instructions'
    result = prompt
    for (start, end) in reversed(spans):
        result = result[:start] + rules + result[end:]
    assert allowed_ids(result) == allowed_ids(prompt), 'Source ID tables changed'
    original_data = prompt.rsplit('# Real Data', 1)[1].split('\nThe report should include', 1)[0]
    changed_data = result.rsplit('# Real Data', 1)[1].split('\nThe report should include', 1)[0]
    assert original_data == changed_data, 'Actual source evidence changed'
    return result

def prose(value):
    return [value['title'], value['summary'], value['rating_explanation']] + [f[k] for f in value['findings'] for k in ('summary', 'explanation')]

def inspect(value, original_schema, prompt, mode):
    Draft202012Validator(original_schema).validate(value)
    if not 5 <= len(value['findings']) <= 10:
        raise ValueError('Original task requires five to ten findings')
    if mode == 'structured_english':
        for text in prose(value):
            if not re.fullmatch(ASCII_PROSE, text):
                raise ValueError('Non-ASCII text or inline square-bracket reference in prose')
            if re.search('\\d+(?:\\s*,\\s*\\d+){19,}', text):
                raise ValueError('Long numeric enumeration in prose')
    elif re.search('[\\u3400-\\u9fff\\uac00-\\ud7af\\u3040-\\u30ff]', '\n'.join(prose(value))):
        raise ValueError('Non-English script in control report')
    parsed = SimpleNamespace(**{**value, 'findings': [SimpleNamespace(**f) for f in value['findings']]})
    check = audit(parsed, prompt)
    if not check['valid']:
        raise ValueError('; '.join(check['errors']))
    return check

async def generate(client, case, dest, log, seed, mode, retries=True):
    source = json.loads(Path(case['input_file']).read_text())
    assert hashlib.sha256(Path(case['input_file']).read_bytes()).hexdigest() == case['input_sha256']
    prompt = source['messages'][0]['content']
    model_prompt = normalized_prompt(prompt) if mode == 'structured_english' else prompt
    schema = schema_for(source['schema'], prompt) if mode == 'structured_english' else control_schema(source['schema'])
    feedback = ''
    for attempt in range(len(TEMPERATURES) if retries else 1):
        messages = [{'role': 'system', 'content': STRUCTURED if mode == 'structured_english' else ENGLISH}]
        if feedback:
            permitted = '; '.join((name + ': ' + ', '.join(map(str, sorted(ids))) for (name, ids) in sorted(allowed_ids(prompt).items())))
            messages.append({'role': 'system', 'content': 'The previous response failed validation. Return complete English JSON for the original report task. Do not shorten or cut the report to evade validation. Validation error: ' + feedback + '. Actual permitted source IDs: ' + permitted})
        messages.append({'role': 'user', 'content': model_prompt})
        body = {'model': 'Qwen/Qwen2.5-7B-Instruct', 'messages': messages, 'max_tokens': 8192, 'temperature': TEMPERATURES[attempt], 'top_p': 1, 'top_k': -1, 'repetition_penalty': 1.05, 'frequency_penalty': 0, 'presence_penalty': 0, 'seed': seed, 'n': 1, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'CommunityReportResponse', 'schema': schema, 'strict': True}}}
        record = {'arm': case['arm'], 'community': case['community'], 'level': case['level'], 'attempt': attempt, 'mode': mode, 'scope': case['scope'], 'seed': seed, 'original_prompt_sha256': case['prompt_sha256'], 'model_prompt_sha256': hashlib.sha256(model_prompt.encode()).hexdigest(), 'grounding_prompt_normalized': mode == 'structured_english'}
        started = time.perf_counter()
        response = None
        report = None
        try:
            reply = await client.post('http://127.0.0.1:18618/v1/chat/completions', json=body)
            reply.raise_for_status()
            response = reply.json()
            record['http_status'] = reply.status_code
            record.update(response.get('usage', {}))
            choice = response['choices'][0]
            record['finish_reason'] = choice['finish_reason']
            text = choice['message']['content']
            record['long_numeric_chain'] = bool(re.search('\\d+(?:\\s*,\\s*\\d+){50,}', text))
            if choice['finish_reason'] == 'length':
                raise ValueError('output_budget_exhausted')
            value = json.loads(text)
            Draft202012Validator(schema).validate(value)
            if mode == 'structured_english':
                for string in prose(value):
                    if not re.fullmatch(ASCII_PROSE, string) or re.search('\\d+(?:\\s*,\\s*\\d+){19,}', string):
                        raise ValueError('Invalid prose script, brackets or numeric enumeration')
                report = render_report(value)
                record['citation_audit'] = inspect(report, source['schema'], prompt, 'rendered')
                record['english_ascii_prose'] = True
            else:
                report = value
                record['citation_audit'] = inspect(value, source['schema'], prompt, mode)
            record['findings'] = len(report['findings'])
            record['status'] = 'ok'
        except Exception as exc:
            record['status'] = 'error'
            record['error'] = str(exc)[:500]
            feedback = str(exc)[:1000]
        record['request_s'] = time.perf_counter() - started
        raw = dest / 'raw_requests' / f"{case['arm']}-{case['community']}-a{attempt}.json"
        raw.parent.mkdir(exist_ok=True, parents=True)
        record['raw_file'] = str(raw)
        raw.write_text(json.dumps({'request': body, 'response': response, 'report': report, 'record': record}, ensure_ascii=False))
        log.write(json.dumps(record) + '\n')
        log.flush()
        if record['status'] == 'ok':
            path = dest / 'reports' / f"{case['arm']}-{case['community']}.json"
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(json.dumps(report, indent=2))
            return record
    return record

async def batch(cases, dest, seed, mode='structured_english', concurrency=64, retries=True):
    dest.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        with (dest / 'requests.jsonl').open('w') as log:

            async def one(case):
                nonlocal done
                async with sem:
                    row = await generate(client, case, dest, log, seed, mode, retries)
                    done += 1
                    if done % 25 == 0 or done == len(cases):
                        print(time.strftime('%H:%M:%S'), dest.name, f'{done}/{len(cases)}', flush=True)
                    return row
            results = []
            times = []
            started = time.perf_counter()
            for level in sorted({c['level'] for c in cases}, reverse=True):
                group = [c for c in cases if c['level'] == level]
                t = time.perf_counter()
                results.extend(await asyncio.gather(*(one(c) for c in group)))
                times.append({'level': level, 'tasks': len(group), 'wall_s': time.perf_counter() - t})
    requests = [json.loads(s) for s in (dest / 'requests.jsonl').read_text().splitlines()]
    good = [r for r in requests if r['status'] == 'ok']
    bad = [r for r in requests if r['status'] != 'ok']
    result = {'name': dest.name, 'tasks': len(cases), 'successful_reports': len(good), 'failed_reports': sum((r['status'] != 'ok' for r in results)), 'requests': len(requests), 'failed_attempts': len(bad), 'length_attempts': sum((r.get('finish_reason') == 'length' for r in requests)), 'successful_input_tokens': sum((r.get('prompt_tokens', 0) for r in good)), 'successful_output_tokens': sum((r.get('completion_tokens', 0) for r in good)), 'failed_input_tokens': sum((r.get('prompt_tokens', 0) for r in bad)), 'failed_output_tokens': sum((r.get('completion_tokens', 0) for r in bad)), 'generation_wall_s': time.perf_counter() - started, 'level_times': times, 'mode': mode, 'concurrency': concurrency, 'seed': seed, 'scope': 'Frozen original community-report prompts; not a graph-to-index end-to-end timing trial'}
    (dest / 'result.json').write_text(json.dumps(result, indent=2))
    return result
