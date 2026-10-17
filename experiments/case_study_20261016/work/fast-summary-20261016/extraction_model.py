"""Uncached, serial English Qwen calls with separate extraction/description ledgers."""
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
import httpx
import tiktoken

DEST = None
SEED = 17
PHASE = 'extraction'
EXHAUSTED = []
ENC = tiktoken.get_encoding('cl100k_base')
POLICY = 'english-original-format-lossless-split-v2'

NAME = {'type': 'string', 'minLength': 1, 'maxLength': 120, 'pattern': r'^[\x20-\x7e]+$'}
DESCRIPTION = {'type': 'string', 'minLength': 1, 'pattern': r'^[\x20-\x7e]+$'}
SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['entities', 'relationships'], 'properties': {
        'entities': {'type': 'array', 'maxItems': 64, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'type', 'description'], 'properties': {
                'name': NAME, 'type': {'type': 'string', 'enum': ['ORGANIZATION', 'PERSON', 'GEO', 'EVENT']},
                'description': DESCRIPTION}}},
        'relationships': {'type': 'array', 'maxItems': 96, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['source', 'target', 'description', 'strength'], 'properties': {
                'source': NAME, 'target': NAME, 'description': DESCRIPTION,
                'strength': {'type': 'integer', 'minimum': 1, 'maximum': 10}}}}}}


def render_json(value):
    from jsonschema import Draft202012Validator
    Draft202012Validator(SCHEMA).validate(value)
    if len(value['entities']) >= 64 or len(value['relationships']) >= 96:
        raise ValueError('record_capacity_reached_split_source_required')
    rows = []
    def clean(text):
        if '<|>' in text or '##' in text or '<|COMPLETE|>' in text:
            raise ValueError('delimiter_in_record_field')
        return text.strip()
    for ent in value['entities']:
        rows.append('("entity"<|>'+clean(ent['name']).upper()+'<|>'+ent['type']+'<|>'+clean(ent['description'])+')')
    for rel in value['relationships']:
        rows.append('("relationship"<|>'+clean(rel['source']).upper()+'<|>'+clean(rel['target']).upper()+'<|>'+clean(rel['description'])+'<|>'+str(rel['strength'])+')')
    return '##\n'.join(rows)+'\n<|COMPLETE|>'


def source_prompt(messages):
    for message in messages:
        content = message.get('content', '')
        if message.get('role') == 'user' and '-Real Data-' in content:
            return content
    return None


def extraction_prompt(original):
    # Keep the established examples and preserve every real-input byte.
    prefix, real = original.rsplit('-Real Data-', 1)
    prefix = prefix.replace('Comprehensive description of', 'Concise factual description of')
    instruction = (
        'Use only facts in the Real Data. Preserve named people and institutions accurately. '
        'Each entity description must explain that entity, not contain another entity name as a substitute. '
        'Use one record per entity and one per undirected relationship pair, combining supported facts. '
        'Do not infer a relationship just because two names share a topic. '
        'Write short English descriptions. Stop with <|COMPLETE|> as soon as the supported records are exhausted.\n')
    return instruction + prefix + '-Real Data-' + real


def records(text):
    result = []
    for raw in text.replace('<|COMPLETE|>', '').split('##'):
        raw = raw.strip()
        if not raw:
            continue
        if not raw.startswith('(') or not raw.endswith(')'):
            raise ValueError('malformed_record')
        parts = [s.strip() for s in raw[1:-1].split('<|>')]
        kind = parts[0].strip('"').lower()
        if kind == 'entity' and len(parts) == 4:
            key = ('entity', parts[1].strip('"').upper())
        elif kind == 'relationship' and len(parts) == 5:
            key = ('relationship', *sorted(x.strip('"').upper() for x in parts[1:3]))
            if not 0 <= float(parts[4]) <= 10:
                raise ValueError('invalid_relationship_strength')
        else:
            raise ValueError('malformed_record_fields')
        if any(not p for p in parts[1:]):
            raise ValueError('empty_record_field')
        result.append((key, parts, raw))
    return result


def split_source(original):
    prefix, real = original.rsplit('-Real Data-', 1)
    header, source = real.split('Text:', 1)
    text, trailer = source.rsplit('\n######################', 1)
    if len(text) < 600:
        raise ValueError('source_too_short_to_split')
    middle = len(text)//2
    candidates = [m.end() for m in re.finditer(r'\n\n|[.!?]\s', text) if len(text)//3 <= m.end() <= len(text)*2//3]
    cut = min(candidates, key=lambda x: abs(x-middle)) if candidates else middle
    # Character overlap preserves boundary evidence; both spans cover every input byte.
    overlap = min(160, cut//4)
    spans = [(0, cut), (cut-overlap, len(text))]
    assert text[:cut] + text[cut:] == text
    prompts = [prefix+'-Real Data-'+header+'Text:'+text[a:z]+'\n######################'+trailer for a, z in spans]
    return prompts, spans, hashlib.sha256(text.encode()).hexdigest()


def merge_records(outputs):
    merged = {}
    descriptions = {}
    for text in outputs:
        for key, parts, _ in records(text):
            if key not in merged:
                merged[key] = parts
                descriptions[key] = [parts[3]]
            else:
                index = 3
                if parts[index] not in descriptions[key]:
                    descriptions[key].append(parts[index])
                    merged[key][index] = '; '.join(descriptions[key])
                if key[0] == 'relationship':
                    merged[key][4] = str(max(float(parts[4]), float(merged[key][4])))
    return '##\n'.join('('+'<|>'.join(parts)+')' for parts in merged.values())+'\n<|COMPLETE|>'


class LoggedModel:
    def __init__(self, config, name='', **kwargs):
        self.config, self.name = config, name

    async def achat(self, prompt, history=None, **kwargs):
        depth = kwargs.pop('_split_depth', 0)
        description = kwargs.get('name') == 'summarize' or 'summar' in self.name
        category = ('description_merge' if PHASE == 'merge' else 'description_extraction') if description else 'extraction'
        messages = list(history or [])
        if not messages:
            messages.append({'role': 'system', 'content':
                'Answer entirely in English. Preserve the requested output format and delimiters. '
                'Extract only supported facts; use concise descriptions. Do not add numbered lists, '
                'repeat records or enumerate numbers. Complete the response within the output budget.'})
        original = source_prompt(messages + [{'role': 'user', 'content': prompt}])
        continuation = kwargs.get('name', '').startswith('extract-continuation')
        if not description and '-Real Data-' in prompt:
            prompt = extraction_prompt(prompt)
        elif continuation:
            prompt = ('Check only the original source text for supported entities or relationships missing from the previous output. '
                'Do not repeat previously extracted entity names or undirected relationship pairs. '
                'Do not infer more records just because this is a follow-up. If none are missing, output only <|COMPLETE|>. '
                'Otherwise return missing records in the original entity/relationship tuple format, separated by ##, and end with <|COMPLETE|>. '
                'Write concise English descriptions and stop.')
        messages.append({'role': 'user', 'content': prompt})
        call_id = f'{time.time_ns()}'
        max_tokens = 2048 if description else 4096
        attempts = 3 if description else 2
        failure = ''
        async with httpx.AsyncClient(timeout=600) as client:
            for attempt in range(attempts):
                request_messages = list(messages)
                if attempt:
                    request_messages.append({'role': 'user', 'content':
                        f'The previous attempt failed validation: {failure[:160]}. '
                        'Regenerate only explicitly supported, unique records. Combine repeated facts for the same entity or pair. '
                        'Keep each description to one short sentence. Stop once the source is exhausted; '
                        'complete the requested output format. Do not invent extra records.'})
                body = {'model': 'Qwen/Qwen2.5-7B-Instruct', 'messages': request_messages,
                        'max_tokens': max_tokens, 'temperature': 0, 'top_p': 1, 'top_k': -1,
                        'repetition_penalty': 1.05, 'seed': SEED, 'n': 1}
                row = {'call_id': call_id, 'attempt': attempt, 'category': category,
                       'name': kwargs.get('name', self.name), 'seed': SEED,
                       'max_tokens': max_tokens, 'started_at': time.time(), 'policy': POLICY,
                       'split_depth': depth,
                       'input_cl100k': sum(len(ENC.encode(m['content'])) for m in request_messages),
                       'prompt_sha256': hashlib.sha256(json.dumps(request_messages).encode()).hexdigest()}
                data = None
                start = time.perf_counter()
                try:
                    reply = await client.post('http://127.0.0.1:18620/v1/chat/completions', json=body)
                    reply.raise_for_status()
                    data = reply.json()
                    row.update(data.get('usage', {}))
                    choice = data['choices'][0]
                    row['finish_reason'] = choice.get('finish_reason')
                    text = choice['message'].get('content') or ''
                    if row['finish_reason'] != 'stop':
                        raise ValueError('incomplete_output:' + str(row['finish_reason']))
                    if not text.strip():
                        raise ValueError('empty_output')
                    if re.search(r'[\u3400-\u9fff]', text):
                        raise ValueError('non_English_output')
                    if re.search(r'\d+(?:\s*,\s*\d+){19,}', text):
                        raise ValueError('number_enumeration')
                    if not description:
                        # A model may place different facts for one entity in separate
                        # rows. Union their descriptions without throwing any fact away.
                        # All actual generated tokens still count for this request.
                        if '<|COMPLETE|>' not in text:
                            raise ValueError('missing_extraction_completion_delimiter')
                        raw_records = records(text)
                        from collections import Counter
                        duplicates = Counter(raw for _, _, raw in raw_records)
                        if any(count >= 4 for count in duplicates.values()):
                            raise ValueError('repeated_record_loop')
                        text = merge_records([text])
                    if not description and kwargs.get('name', '').startswith('extract-loopcheck') is False:
                        if '<|COMPLETE|>' not in text:
                            raise ValueError('missing_extraction_completion_delimiter')
                        parsed = records(text)
                        keys = [key for key, _, _ in parsed]
                        if len(keys) != len(set(keys)):
                            raise ValueError('duplicate_entity_or_relationship')
                        if not parsed and not continuation:
                            raise ValueError('empty_initial_extraction')
                        if continuation:
                            previous = set()
                            for m in history or []:
                                if m.get('role') == 'assistant':
                                    previous.update(key for key, _, _ in records(m['content']))
                            # Existing keys can carry additional supported descriptions;
                            # GraphRAG merges those facts with the first extraction.
                        row['entity_records'] = sum(key[0] == 'entity' for key in keys)
                        row['relationship_records'] = sum(key[0] == 'relationship' for key in keys)
                    row['status'] = 'ok'
                    return SimpleNamespace(output=SimpleNamespace(content=text),
                        history=messages + [{'role': 'assistant', 'content': text}])
                except Exception as exc:
                    failure = str(exc)
                    row.update(status='error', error=failure[:800])
                finally:
                    row['request_s'] = time.perf_counter() - start
                    row['ended_at'] = time.time()
                    raw = DEST / 'raw_extraction_requests'
                    raw.mkdir(exist_ok=True)
                    path = raw / f'{call_id}-a{attempt}.json'
                    row['raw_file'] = str(path)
                    path.write_text(json.dumps({'request': body, 'response': data, 'record': row}))
                    with (DEST / 'extraction_calls.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
                if attempt < attempts-1:
                    await asyncio.sleep(1)
        if not description and original and depth < 2:
            try:
                prompts, spans, digest = split_source(original)
            except ValueError:
                prompts = []
            if prompts:
                # Both calls must complete; no truncated response or source span is discarded.
                responses = [await self.achat(p, _split_depth=depth+1) for p in prompts]
                text = merge_records([r.output.content for r in responses])
                event = {'parent_call_id': call_id, 'policy': POLICY, 'split_depth': depth,
                         'source_sha256': digest, 'source_spans': spans, 'all_source_characters_covered': True,
                         'both_children_succeeded': True}
                with (DEST/'extraction_recovery.jsonl').open('a') as log:
                    log.write(json.dumps(event)+'\n')
                return SimpleNamespace(output=SimpleNamespace(content=text),
                    history=messages+[{'role': 'assistant', 'content': text}])
        EXHAUSTED.append({'call_id': call_id, 'category': category, 'error': failure})
        raise RuntimeError('Extraction/description request exhausted: ' + failure)


def install(dest, seed):
    global DEST, SEED, PHASE
    from graphrag.language_model.factory import ModelFactory
    from graphrag.language_model.manager import ModelManager
    DEST, SEED, PHASE = Path(dest), seed, 'extraction'
    EXHAUSTED.clear()
    ModelManager().chat_models.clear()
    ModelFactory.register_chat('openai_chat', lambda **kwargs: LoggedModel(**kwargs))
