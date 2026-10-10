#!/usr/bin/env python
"""Self-correcting retry when community-report generation fails.

GraphRAG logs "No report found" and drops the community when JSON parsing fails, without retrying.
At temperature 0 generation is deterministic -- litellm's 10 retries only help with connection errors and are
useless for parse failures: the same prompt always yields the same bad JSON. Measured, this silently loses hundreds
of reports out of 8,550 communities, biased toward large communities, which directly distorts the index.

This wraps the strategy function: when it returns None, retry at a higher temperature so randomness escapes the bad output.
Both failures and successful retries are logged for later reconciliation.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import logging
import os

logger = logging.getLogger(__name__)

_TEMPS = [0.3, 0.7]
_LOG = os.environ.get("RETRY_LOG", DYLO_SCRATCH + "/grag_retry.tsv")
_ARM = os.environ.get("COST_ARM", "?")


def install() -> None:
    import graphrag.index.operations.summarize_communities.strategies as S

    orig = S.run_graph_intelligence

    async def with_retry(community, input, level, callbacks, cache, args):  # noqa: A002
        out = await orig(community, input, level, callbacks, cache, args)
        if out is not None:
            return out
        for t in _TEMPS:
            a = dict(args)
            llm = dict(a.get("llm", {}))
            llm["temperature"] = t
            a["llm"] = llm
            logger.info("[retry] community %s failed to parse, retrying with temperature=%s", community, t)
            try:
                out = await orig(community, input, level, callbacks, cache, a)
            except Exception as e:  # noqa: BLE001
                logger.warning("[retry] community %s raised at temperature=%s: %s", community, t, e)
                out = None
            if out is not None:
                _record(community, level, t, "ok")
                return out
        _record(community, level, -1, "give_up")
        logger.warning("[retry] community %s failed three times, giving up", community)
        return None

    S.run_graph_intelligence = with_retry
    logger.info("[report_retry] installed, retry temperatures %s", _TEMPS)


def _record(community, level, temp, status) -> None:
    try:
        with open(_LOG, "a") as f:
            f.write(f"{_ARM}\t{community}\t{level}\t{temp}\t{status}\n")
    except Exception:
        pass
