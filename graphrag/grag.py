#!/usr/bin/env python
"""graphrag CLI wrapper with a litellm fix.

litellm 1.97.0's types/utils.py imports ChatCompletionReasoningItem but not
ChatCompletionReasoningSummaryTextBlock, which its fields reference forward; pydantic 2.11 cannot resolve it in that module's namespace
when building Message, so every completion raises PydanticUserError.
sitecustomize is too early (litellm is not importable yet), so we fix it explicitly here and then enter the CLI.

Usage is the same as the official one:  grag.py index --root .   /   grag.py update --root .
"""
import os
import sys


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
    U.ModelResponse()  # validate immediately so a failure surfaces now rather than halfway through a run
    print("[patch] litellm forward reference fixed", file=sys.stderr, flush=True)


if __name__ == "__main__":
    _patch_litellm()
    if os.environ.get("COST_ARM"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import cost_meter

        cost_meter.install()
        import report_retry

        report_retry.install()
    from graphrag.cli.main import app
    app()
