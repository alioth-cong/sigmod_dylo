#!/usr/bin/env python
"""Uniform accounting: the payable cost of every community-report generation, regardless of cache hits.

Why it must be recorded before the cache: GraphRAG's cache hits on prompt content. In the full-rebuild arm, if a
community's content equals that of an earlier run, the LLM call is absorbed by the cache and the booked cost becomes 0 -- but
rebuild by definition rewrites everything, so the measured denominator would be wrong and the arms incomparable.

The `input` of run_graph_intelligence(community, input, level, ...) is the community context,
and the function runs before the cache lookup, so tokens recorded here are "payable", not "paid".

Each line: arm batch community level context_tokens prompt_template_tokens
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os
import sys

_LOG = os.environ.get("COST_LOG", DYLO_SCRATCH + "/grag_cost.tsv")
_ARM = os.environ.get("COST_ARM", "?")
_BATCH = os.environ.get("DYLO_BATCH", os.environ.get("COST_BATCH", "?"))

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _ntok(s: str) -> int:
        return len(_ENC.encode(s or ""))
except Exception:  # fall back to characters/4 when tiktoken is unavailable
    def _ntok(s: str) -> int:
        return len(s or "") // 4


def install() -> None:
    import graphrag.index.operations.summarize_communities.strategies as S

    orig = S.run_graph_intelligence

    async def metered(community, input, level, callbacks, cache, args):  # noqa: A002
        tmpl = args.get("extraction_prompt") or ""
        try:
            with open(_LOG, "a") as f:
                f.write(f"{_ARM}\t{_BATCH}\t{community}\t{level}\t"
                        f"{_ntok(input)}\t{_ntok(tmpl)}\n")
        except Exception as e:  # accounting failures must not break indexing
            print(f"[cost_meter] failed to write log: {e}", file=sys.stderr)
        return await orig(community, input, level, callbacks, cache, args)

    S.run_graph_intelligence = metered
    print(f"[cost_meter] installed arm={_ARM} batch={_BATCH} -> {_LOG}",
          file=sys.stderr, flush=True)
