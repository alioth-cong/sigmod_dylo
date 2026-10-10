#!/usr/bin/env python
"""Replace two steps of the official update pipeline with DyLo's approach; everything else is unchanged.

The official `graphrag update` runs the full pipeline on the new documents alone (an independent
community partition on the delta graph, reports for the delta communities), then appends the delta communities with ids offset by old_max
to the old table -- it **never re-clusters the merged graph**, so communities of old and new documents are never reconciled.

DyLo's approach:
  1) update_communities   -> incremental partition of the **merged** graph (DyLo, without singleton reclamation)
  2) update_community_reports -> rewrite reports only for communities whose member set changed; inherit the rest

Extraction, chunking, entity resolution, embedding, and retrieval all keep the official implementation.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_RESUM_LOG = os.environ.get("DYLO_RESUM_LOG", DYLO_SCRATCH + "/dylo_resum.tsv")


def _members(df: pd.DataFrame) -> dict[int, frozenset]:
    """community -> its set of entity ids. The member set is the only criterion for "rewrite the summary or not".

    entity_ids is a numpy array, so `x or []` cannot be used: the truth value of a multi-element array raises
    ValueError ("truth value of an array ... is ambiguous"). Emptiness must be checked explicitly.
    """
    out: dict[int, frozenset] = {}
    for cid, ents in zip(df["community"], df["entity_ids"]):
        out[int(cid)] = frozenset() if ents is None else frozenset(ents)
    return out


def install() -> None:
    import graphrag.index.workflows.update_communities as uc
    import graphrag.index.workflows.update_community_reports as ucr
    from graphrag.index.run.utils import get_update_storages
    from graphrag.index.typing.workflow import WorkflowFunctionOutput
    from graphrag.index.workflows.create_communities import create_communities
    from graphrag.index.workflows.create_community_reports import (
        create_community_reports,
    )
    from graphrag.utils.storage import (
        load_table_from_storage,
        storage_has_table,
        write_table_to_storage,
    )

    import dylo_cluster

    dylo_cluster.install()

    async def dylo_update_communities(config, context):
        logger.info("Workflow started: update_communities [DyLo]")
        output_storage, previous_storage, _ = get_update_storages(
            config, context.state["update_timestamp"]
        )
        entities = context.state["incremental_update_merged_entities"]
        relationships = context.state["incremental_update_merged_relationships"]

        old_communities = await load_table_from_storage(
            "communities", previous_storage
        )
        # DyLo re-partitions the merged graph (batch number given by DYLO_BATCH)
        new_communities = create_communities(
            entities,
            relationships,
            max_cluster_size=config.cluster_graph.max_cluster_size,
            use_lcc=config.cluster_graph.use_lcc,
            seed=config.cluster_graph.seed,
        )
        await write_table_to_storage(new_communities, "communities", output_storage)

        context.state["dylo_old_communities"] = old_communities
        context.state["dylo_new_communities"] = new_communities
        # later official steps read this mapping; DyLo renumbers, so it is left empty
        context.state["incremental_update_community_id_mapping"] = {}
        logger.info(
            "Workflow completed: update_communities [DyLo] %d -> %d communities",
            len(old_communities), len(new_communities),
        )
        return WorkflowFunctionOutput(result=None)

    async def dylo_update_community_reports(config, context):
        logger.info("Workflow started: update_community_reports [DyLo]")
        output_storage, previous_storage, _ = get_update_storages(
            config, context.state["update_timestamp"]
        )
        old_comms = context.state["dylo_old_communities"]
        new_comms = context.state["dylo_new_communities"]
        old_reports = await load_table_from_storage(
            "community_reports", previous_storage
        )

        old_mem = _members(old_comms)
        new_mem = _members(new_comms)
        # old member set -> old community id; decides which communities can inherit their report
        by_set: dict[frozenset, int] = {}
        for cid, ms in old_mem.items():
            by_set.setdefault(ms, cid)

        # an inherited report must also get its level rewritten: after DyLo re-partitions, the same member set can sit on another level,
        # and changing only the community id would leave the report in the table with its old level. Retrieval's read_indexer_reports
        # filters by the report's own level, so a wrong level selects reports that are not actually on that level;
        # those communities get no entities and zero occurrence weight, and once dynamic_community_selection
        # picks them, max_weight in _compute_community_weights is 0, raising
        # ZeroDivisionError. This bug made all 60 questions of the norecl arm fail (1036/1300 reports had the wrong level).
        new_level = {int(c): int(l) for c, l in
                     zip(new_comms["community"], new_comms["level"])}

        rep_by_comm = {int(r["community"]): r for _, r in old_reports.iterrows()}
        inherit_rows, changed = [], []
        for cid, ms in new_mem.items():
            src = by_set.get(ms)
            if src is not None and src in rep_by_comm:
                row = rep_by_comm[src].copy()
                row["community"] = cid          # keep the report text, switch to the new community id
                row["human_readable_id"] = cid
                if "level" in row.index:
                    row["level"] = new_level.get(cid, row["level"])
                inherit_rows.append(row)
            else:
                changed.append(cid)

        logger.info(
            "[DyLo] %d communities, %d inherited, %d to rewrite",
            len(new_mem), len(inherit_rows), len(changed),
        )
        with open(_RESUM_LOG, "a") as f:
            f.write(f"{os.environ.get('DYLO_BATCH','?')}\t{len(new_mem)}\t"
                    f"{len(inherit_rows)}\t{len(changed)}\n")

        if changed:
            edges = await load_table_from_storage("relationships", output_storage)
            entities = await load_table_from_storage("entities", output_storage)
            claims = None
            if config.extract_claims.enabled and await storage_has_table(
                "covariates", output_storage
            ):
                claims = await load_table_from_storage("covariates", output_storage)
            m = config.get_language_model_config(config.community_reports.model_id)
            fresh = await create_community_reports(
                edges_input=edges,
                entities=entities,
                communities=new_comms[new_comms["community"].isin(changed)],
                claims_input=claims,
                callbacks=context.callbacks,
                cache=context.cache,
                summarization_strategy=config.community_reports.resolved_strategy(
                    config.root_dir, m
                ),
                async_mode=m.async_mode,
                num_threads=m.concurrent_requests,
            )
        else:
            fresh = pd.DataFrame(columns=old_reports.columns)

        parts = [p for p in (pd.DataFrame(inherit_rows), fresh) if len(p)]
        merged = pd.concat(parts, ignore_index=True) if parts else old_reports.iloc[:0]
        await write_table_to_storage(
            merged, "community_reports", output_storage
        )
        context.state["incremental_update_merged_community_reports"] = merged
        logger.info("Workflow completed: update_community_reports [DyLo]")
        return WorkflowFunctionOutput(result=None)

    uc.run_workflow = dylo_update_communities
    ucr.run_workflow = dylo_update_community_reports

    # workflows are looked up by name in a registry bound to the old functions at import time, so replace them there too
    from graphrag.index.workflows.factory import PipelineFactory

    PipelineFactory.workflows["update_communities"] = dylo_update_communities
    PipelineFactory.workflows["update_community_reports"] = dylo_update_community_reports
    logger.info("[DyLo] patch installed")
