#!/usr/bin/env python
"""Exact re-summarization token cost of rebuild (full rebuild every batch, the GraphRAG paper's official approach).

No LLM calls: contexts are built through GraphRAG's own explode_communities -> _prep_nodes/_prep_edges ->
build_local_context -> build_level_context chain and counted with the same tokenizer.
Same accounting as cost_meter.py (context tokens + the prompt template resent with every call).

This works because rebuild by definition rewrites every community of the state: which reports are written is fixed
and independent of any LLM output, so the cost can be computed exactly offline instead of running for hours.

Usage: calc_rebuild_cost.py <snapshot dir> [label]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import sys
from pathlib import Path

import pandas as pd


def _patch_litellm():
    import litellm.types.utils as U
    from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock as B

    U.ChatCompletionReasoningSummaryTextBlock = B
    for n in ("Message", "Delta", "Choices", "StreamingChoices",
              "ModelResponse", "ModelResponseStream"):
        c = getattr(U, n, None)
        if c is not None and hasattr(c, "model_rebuild"):
            try:
                c.model_rebuild()
            except Exception:
                pass


def main(snap: str, label: str) -> None:
    _patch_litellm()
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    from graphrag.config.load_config import load_config
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.index.operations.summarize_communities.explode_communities import (
        explode_communities,
    )
    from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
        build_local_context,
        build_level_context,
    )
    from graphrag.index.workflows.create_community_reports import _prep_edges, _prep_nodes
    from graphrag.tokenizer.get_tokenizer import get_tokenizer
    import graphrag.data_model.schemas as schemas

    d = Path(snap)
    # the config directory can be set via CFG_ROOT so one script serves several datasets
    import os
    cfg = load_config(Path(os.environ.get("CFG_ROOT",
                                          DYLO_ROOT + "/graphrag")))
    entities = pd.read_parquet(d / "entities.parquet")
    communities = pd.read_parquet(d / "communities.parquet")
    edges_input = pd.read_parquet(d / "relationships.parquet")

    m = cfg.get_language_model_config(cfg.community_reports.model_id)
    strategy = cfg.community_reports.resolved_strategy(cfg.root_dir, m)
    strategy["extraction_prompt"] = strategy["graph_prompt"]
    tokenizer = get_tokenizer(LanguageModelConfig(**strategy["llm"]))
    max_input_length = strategy.get("max_input_length", 8000)

    nodes = _prep_nodes(explode_communities(communities, entities))
    edges = _prep_edges(edges_input)
    cb = NoopWorkflowCallbacks()

    local_contexts = build_local_context(nodes, edges, None, tokenizer, cb,
                                         max_input_length)

    tmpl_tok = len(tokenizer.encode(strategy["extraction_prompt"]))
    levels = sorted(nodes[schemas.COMMUNITY_LEVEL].unique(), reverse=True)

    total_ctx = 0
    n_reports = 0
    per_level = {}
    reports_so_far = pd.DataFrame()

    for level in levels:
        level_ctx = build_level_context(
            report_df=reports_so_far,
            community_hierarchy_df=communities,
            local_context_df=local_contexts,
            level=level,
            tokenizer=tokenizer,
            max_context_tokens=max_input_length,
        )
        if level_ctx is None or len(level_ctx) == 0:
            continue
        ctx_col = schemas.CONTEXT_STRING
        toks = level_ctx[ctx_col].map(lambda s: len(tokenizer.encode(str(s))))
        per_level[int(level)] = (len(level_ctx), int(toks.sum()))
        total_ctx += int(toks.sum())
        n_reports += len(level_ctx)
        # upper levels take lower-level reports as input; a placeholder keeps the level chain intact
        reports_so_far = pd.concat([reports_so_far, level_ctx], ignore_index=True)

    total_tmpl = n_reports * tmpl_tok
    total = total_ctx + total_tmpl
    price = 0.15 / 1_000_000

    print(f"=== {label} ({d}) ===")
    print(f"template tokens/call: {tmpl_tok}")
    print(f"{'level':<8}{'reports':>9}{'context tokens':>14}")
    for lv in sorted(per_level, reverse=True):
        c, t = per_level[lv]
        print(f"{lv:<8}{c:>9,}{t:>14,}")
    print(f"{'total':<8}{n_reports:>9,}{total_ctx:>14,}")
    print(f"template tokens total: {total_tmpl:,}")
    print(f"input tokens total: {total:,}")
    print(f"cost (GPT-4o-mini input price): ${total*price:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rebuild")
