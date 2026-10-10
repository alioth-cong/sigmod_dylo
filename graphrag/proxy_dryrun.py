#!/usr/bin/env python
"""Tier-A dry run: rebuild offline the real prompt of each community's summary call and map it exactly to the cache.

Method: reuse GraphRAG 2.7.2's create_community_reports data preparation and context construction unchanged
(explode_communities / _prep_nodes / _prep_edges / build_local_context /
build_level_context), replacing run_graph_intelligence with a stub that assembles the prompt
as CommunityReportsExtractor does, computes the cache key by the litellm wrapper's rule
(get_cache_key: sha256 of parameters + response_format + messages), and then reads that call's actual usage
from cache/community_reporting/ -- without calling the LLM.

A key hit means the rebuilt prompt is byte-identical to the one sent to the model, so the hit rate itself
proves the dry run is exact; it replaces the older title-based lookup of proxy_fidelity.py (which matched only 80-90%).

Usage:
  proxy_dryrun.py <index_dir> [--only-communities ids.txt] --out rows.csv

One output row per community:
  community level is_leaf n_children n_ent (entities in the community at its level, i.e. subtree entities)
  n_edges_ctx (relationships actually in the context) n_nodes_ctx (entities actually in the context)
  ctx_tok (context tokens, GraphRAG's own tokenizer) exceed (whether max_input_length truncation fired)
  raw_tok (full context tokens without truncation) prompt_tok (full prompt, same tokenizer)
  hit usage_prompt usage_completion finish_reason
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-communities", default="",
                    help="pass only these communities to create_community_reports (reproduces an incremental arm's call set)")
    ap.add_argument("--changed-from", default="",
                    help="the output directory before the update (update_output/<ts>/previous); following dylo_patch's"
                         " rule, keep only communities whose member set changed, reproducing that update's call set")
    ap.add_argument("--cache-dir", default="",
                    help="cache directory, default <index_dir>/cache/community_reporting")
    args = ap.parse_args()

    root = Path(args.index_dir).resolve()
    os.chdir(root)
    from dotenv import load_dotenv
    load_dotenv(root / ".env")

    import graphrag.data_model.schemas as schemas
    from graphrag.config.load_config import load_config
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.index.operations.summarize_communities.explode_communities import (
        explode_communities,
    )
    from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
        build_level_context,
        build_local_context,
    )
    from graphrag.index.operations.summarize_communities.graph_context.sort_context import (
        sort_context,
    )
    from graphrag.index.operations.summarize_communities.community_reports_extractor import (
        CommunityReportResponse,
    )
    import graphrag.index.operations.summarize_communities.summarize_communities as SC
    from graphrag.index.workflows.create_community_reports import _prep_edges, _prep_nodes
    from graphrag.language_model.providers.litellm.get_cache_key import get_cache_key
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    from graphrag.tokenizer.get_tokenizer import get_tokenizer

    cfg = load_config(root)
    mcfg = cfg.get_language_model_config(cfg.community_reports.model_id)
    strategy = cfg.community_reports.resolved_strategy(str(root), mcfg)
    strategy["extraction_prompt"] = strategy["graph_prompt"]
    model_config = LanguageModelConfig(**strategy["llm"])
    tokenizer = get_tokenizer(model_config)
    max_input = strategy.get("max_input_length")
    max_report = strategy.get("max_report_length") or 1500
    tmpl = strategy["extraction_prompt"]
    cache_dir = Path(args.cache_dir) if args.cache_dir else root / "cache/community_reporting"
    print(f"[dryrun] {root.name}: tokenizer={type(tokenizer).__name__} "
          f"max_input_length={max_input} max_report_length={max_report} "
          f"model={model_config.model} max_tokens={model_config.max_tokens}",
          file=sys.stderr)

    out_dir = root / "output"
    entities = pd.read_parquet(out_dir / "entities.parquet")
    edges_in = pd.read_parquet(out_dir / "relationships.parquet")
    communities = pd.read_parquet(out_dir / "communities.parquet")
    if args.only_communities:
        keep = {int(x) for x in Path(args.only_communities).read_text().split()}
        communities = communities[communities["community"].isin(keep)]
    if args.changed_from:
        # same rule as dylo_patch.dylo_update_community_reports: a member set that appears in the pre-update
        # community table -> inherit the old report; otherwise -> rewritten by create_community_reports
        prev = pd.read_parquet(Path(args.changed_from) / "communities.parquet")
        old_sets = {frozenset() if e is None else frozenset(e) for e in prev["entity_ids"]}
        changed = [frozenset() if e is None else frozenset(e) not in old_sets
                   for e in communities["entity_ids"]]
        communities = communities[changed]
        print(f"[dryrun] rewrite set recomputed by the patch rule: {len(communities)} communities",
              file=sys.stderr)

    nodes = _prep_nodes(explode_communities(communities, entities))
    edges = _prep_edges(edges_in)
    local = build_local_context(nodes, edges, None, tokenizer,
                                NoopWorkflowCallbacks(), max_input)

    # full context length without truncation (= raw(subtree) in the design), used to test the saturation rule
    raw_tok = {}
    n_edges_ctx, n_nodes_ctx = {}, {}
    for _, r in local.iterrows():
        cid = int(r[schemas.COMMUNITY_ID])
        full = sort_context(r[schemas.ALL_CONTEXT], tokenizer, max_context_tokens=None)
        raw_tok[cid] = tokenizer.num_tokens(full)
        # edges / nodes actually present in the context (untruncated)
        es = {int(e[schemas.SHORT_ID]) for rec in r[schemas.ALL_CONTEXT]
              for e in rec.get(schemas.EDGE_DETAILS, []) if isinstance(e, dict)}
        n_edges_ctx[cid] = len(es)
        n_nodes_ctx[cid] = len(r[schemas.ALL_CONTEXT])

    rows: dict[int, dict] = {}

    async def stub(community, input, level, callbacks, cache, a):  # noqa: A002
        prompt = tmpl.format(input_text=input, max_report_length=str(max_report))
        key = get_cache_key(
            model_config=model_config, prefix="chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False, name="create_community_report",
            response_format=CommunityReportResponse,
        )
        rec = dict(community=int(community), level=int(level),
                   ctx_tok=tokenizer.num_tokens(input),
                   prompt_tok=tokenizer.num_tokens(prompt), hit=0,
                   usage_prompt="", usage_completion="", finish_reason="")
        f = cache_dir / key
        if f.exists():
            resp = json.load(f.open())["result"]["response"]
            rec.update(hit=1, usage_prompt=resp["usage"]["prompt_tokens"],
                       usage_completion=resp["usage"]["completion_tokens"],
                       finish_reason=resp["choices"][0]["finish_reason"])
        rows[int(community)] = rec
        return None

    SC.load_strategy = lambda _t: stub
    asyncio.run(SC.summarize_communities(
        nodes, communities, local, build_level_context, NoopWorkflowCallbacks(),
        None, strategy, tokenizer=tokenizer, max_input_length=max_input,
    ))

    kids = {int(c): (list(ch) if ch is not None else [])
            for c, ch in zip(communities["community"], communities["children"])}
    nent = {int(c): (0 if e is None else len(e))
            for c, e in zip(communities["community"], communities["entity_ids"])}
    exceed = {int(c): bool(x) for c, x in zip(local[schemas.COMMUNITY_ID],
                                               local[schemas.CONTEXT_EXCEED_FLAG])}
    cols = ["community", "level", "is_leaf", "n_children", "n_ent", "n_nodes_ctx",
            "n_edges_ctx", "raw_tok", "ctx_tok", "exceed", "prompt_tok", "hit",
            "usage_prompt", "usage_completion", "finish_reason"]
    with open(args.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for cid, rec in sorted(rows.items()):
            ch = [x for x in kids.get(cid, []) if x is not None]
            w.writerow(dict(rec, is_leaf=int(len(ch) == 0), n_children=len(ch),
                            n_ent=nent.get(cid, 0), n_nodes_ctx=n_nodes_ctx.get(cid, 0),
                            n_edges_ctx=n_edges_ctx.get(cid, 0),
                            raw_tok=raw_tok.get(cid, ""), exceed=int(exceed.get(cid, False))))
    hits = sum(r["hit"] for r in rows.values())
    print(f"[dryrun] {root.name}: {len(rows)} communities, cache-key hits {hits} "
          f"({hits / max(len(rows), 1):.1%}) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
