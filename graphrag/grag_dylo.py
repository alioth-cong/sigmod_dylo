#!/usr/bin/env python
"""graphrag CLI wrapper for the DyLo arm: litellm fix + the two DyLo patches, then the official CLI.

Apart from community partitioning and report refresh, everything (chunking, extraction, resolution, embedding, retrieval) is the official implementation.
The batch number is passed to dylo_cluster through the DYLO_BATCH environment variable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
            except Exception as e:
                print(f"[patch] {n}.model_rebuild failed: {e}", file=sys.stderr)
    U.ModelResponse()
    print("[patch] litellm forward reference fixed", file=sys.stderr, flush=True)


if __name__ == "__main__":
    if "DYLO_BATCH" not in os.environ:
        sys.exit("DYLO_BATCH must be set")
    _patch_litellm()
    if os.environ.get("COST_ARM"):
        import cost_meter

        cost_meter.install()
        import report_retry

        report_retry.install()
    import dylo_patch

    dylo_patch.install()
    from graphrag.cli.main import app

    app()
