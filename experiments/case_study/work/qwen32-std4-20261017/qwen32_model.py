def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import hashlib, json, re, time, os
from pathlib import Path
from types import SimpleNamespace
from jsonschema import Draft202012Validator
import qwen_rules as rules
from citation_schema import render_report
from citations import audit

class ActualModel:

    def __init__(self, client, log, community, level, attempt, seed, feedback=''):
        (self.client, self.log, self.community, self.level, self.attempt, self.seed) = (client, log, community, level, attempt, seed)
        self.feedback = feedback
        self.last_error = ''

    async def achat(self, prompt, **kwargs):
        from benchmark import CommunityReportResponse, ENC
        schema = rules.schema_for(CommunityReportResponse.model_json_schema(), prompt)
        schema['properties']['findings'].update(minItems=5, maxItems=5)
        instruction = 'Write an English community report from the supplied graph evidence only. Return complete JSON within 2048 output tokens. Use a specific title, a concise executive summary, an impact rating from zero to ten and its one-sentence explanation, and exactly five substantive findings. Each finding has a short heading and one or two concise explanatory sentences. Prioritize the most relevant supported facts and relationships; do not invent facts. Select up to five relevant source human_readable_id values per citation group in the structured citations arrays only. Never write source numbers, reference lists or Data citations inside prose. Use printable English ASCII, transliterated names and single quotes. No double quotes, backslashes, square brackets or line breaks inside prose strings. Do not repeat text or enumerate numbers. Output compact JSON without indentation. The maximum of 2000 words is a ceiling, not a target.'
        if self.attempt:
            instruction += ' Previous validation error: ' + self.feedback[:250] + '. Correct the error and complete the same five-finding report within 2048 output tokens.'
        evidence = prompt.rsplit('# Real Data', 1)[1].split('\nThe report should include', 1)[0]
        messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': '# Real Data' + evidence}]
        body = {'model': 'Qwen/Qwen2.5-32B-Instruct', 'messages': messages, 'max_tokens': 2048, 'temperature': 0, 'top_p': 1, 'top_k': -1, 'repetition_penalty': 1.05, 'seed': self.seed, 'n': 1, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'CommunityReportResponse', 'schema': schema, 'strict': True}}}
        stamp = time.time()
        start = time.perf_counter()
        data = None
        record = {'community': int(self.community), 'level': int(self.level), 'attempt': self.attempt, 'seed': self.seed, 'submitted_at': stamp, 'started_at': stamp, 'input_cl100k': sum((len(ENC.encode(m['content'])) for m in messages)), 'prompt_sha256': hashlib.sha256(json.dumps(messages).encode()).hexdigest(), 'model': 'Qwen/Qwen2.5-32B-Instruct', 'generation_repair': bool(self.attempt), 'max_tokens': 2048, 'temperature': 0}
        try:
            response = await self.client.post(os.environ.get('QWEN_SUMMARY_URL', 'http://127.0.0.1:18750/v1/chat/completions'), json=body)
            response.raise_for_status()
            data = response.json()
            record.update(data.get('usage', {}))
            choice = data['choices'][0]
            record['finish_reason'] = choice.get('finish_reason')
            text = choice['message']['content']
            record['output_cl100k'] = len(ENC.encode(text))
            if choice['finish_reason'] == 'length':
                raise ValueError('output_budget_exhausted_2048')
            value = json.loads(text)
            Draft202012Validator(schema).validate(value)
            assert all((re.fullmatch(rules.ASCII_PROSE, s) and (not re.search('\\d+(?:\\s*,\\s*\\d+){19,}', s)) for s in rules.prose(value)))
            report = render_report(value)
            parsed = CommunityReportResponse.model_validate(report)
            check = audit(parsed, prompt)
            if not check['valid']:
                raise ValueError(str(check['errors']))
            record['citation_audit'] = check
            record['status'] = 'ok'
            return SimpleNamespace(parsed_response=parsed)
        except Exception as exc:
            self.last_error = str(exc)
            record.update(status='error', error=str(exc)[:600])
            raise
        finally:
            record['ended_at'] = time.time()
            record['request_s'] = time.perf_counter() - start
            raw = Path(self.log.name).parent / 'raw_requests'
            raw.mkdir(exist_ok=True)
            file = raw / f'{self.community}-a{self.attempt}.json'
            record['raw_file'] = str(file)
            file.write_text(json.dumps({'request': body, 'response': data, 'record': record}, ensure_ascii=False))
            self.log.write(json.dumps(record) + '\n')
            self.log.flush()
