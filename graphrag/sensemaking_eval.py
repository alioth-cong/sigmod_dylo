#!/usr/bin/env python
"""Sensemaking evaluation following GraphRAG's official protocol: global search + pairwise LLM judging.

Why not F1/EM: GraphRAG targets corpus-level synthesized narratives; answers are multi-paragraph text rather than short spans,
with no gold answers to compare against. Its paper has an LLM compare two answers pairwise on criteria taken from
the paper: comprehensiveness / diversity / empowerment / directness.

Judging is done in both orders (A first once, B first once); a win counts only if both agree, otherwise it is a tie -- this
cancels the LLM's position preference.

Usage:
  sensemaking_eval.py answer <arm> <index dir>     # generate this arm's answers
  sensemaking_eval.py judge  <reference arm> <other arms...>  # pairwise judging
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import asyncio
import collections
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = Path(DYLO_SCRATCH + "")
ANS = TMP / "sense_ans"
ANS.mkdir(parents=True, exist_ok=True)
# the question set is switchable: each corpus must use its own generated sensemaking questions,
# otherwise "novel character" questions get asked of a news corpus and the answers are necessarily empty.
QS = json.load(open(os.environ.get("SENSE_QS", str(TMP / "sensemaking.json"))))
API = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# a single question already issues dozens of map calls against the same GPU; outer concurrency only makes them queue: measured,
# with concurrency 3 a question degrades from 22 seconds to over 5 minutes. Answering is serial; only judging is concurrent.
CONC = int(os.environ.get("SENSE_CONC", "1"))
JUDGE_CONC = int(os.environ.get("JUDGE_CONC", "8"))
# response_type is spliced by GraphRAG directly into the reduce-stage prompt; it is free text.
# In the exposure-matched experiment all eight arms share the same string, including a word limit: the judges' length coefficients
# beta=+4.76/+5.84 far exceed any method effect, so without a length constraint "better content" cannot be separated
# from "longer writing" (after length adjustment the GPT lower bound is only +0.054).
RESP_TYPE = os.environ.get("SENSE_RESP", "Multiple paragraphs")

DIMS = {
    "comprehensiveness": "How much detail does the answer provide to cover all "
                         "aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different "
                 "perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make "
                   "informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}


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


async def answer(arm: str, root: str) -> None:
    _patch_litellm()
    from graphrag.config.load_config import load_config
    import graphrag.api as api

    # two input forms are supported: a full project directory (with output/) or a snapshot directory with parquet files only.
    # global search uses only the entities/communities/community_reports tables and does not load
    # the vector store, so any batch's snapshot can be evaluated directly without waiting for the arm to finish all batches.
    root_p = Path(root)
    out = root_p / "output" if (root_p / "output").is_dir() else root_p
    # the config directory must follow the corpus: MultiHop-RAG and HotpotQA have different settings.yaml files
    cfg = load_config(Path(os.environ.get("CFG_ROOT",
                                          DYLO_ROOT + "/graphrag")))
    rd = lambda n: pd.read_parquet(out / f"{n}.parquet")  # noqa: E731
    ents, comms, reps = rd("entities"), rd("communities"), rd("community_reports")
    print(f"[{arm}] source {out}", flush=True)
    print(f"[{arm}] entities {len(ents)}, communities {len(comms)}, reports {len(reps)}", flush=True)

    # global_search chains many calls internally for dynamic_community_selection;
    # without a timeout one stuck connection hung the whole gather forever -- measured: stuck for
    # 2h19m at 0% CPU, and the 10 finished questions were lost because they were only written after everything finished.
    # Now every question has its own timeout and results are written as they arrive, so a run can be interrupted and resumed safely.
    TIMEOUT_S = int(os.environ.get("SENSE_TIMEOUT", "420"))
    f = ANS / f"{arm}.json"
    done: dict[int, dict] = {}
    # per-question retry limit: questions that still hit the timeout while the backend is alive (measured for questions 0/4, while the neighboring
    # questions 1/2/3 returned normally) show that the query makes global_search's map stage explode, not a service failure.
    # Without a limit the outer loop "until all 120 questions are valid" would spin forever.
    MAX_TRY = int(os.environ.get("SENSE_MAX_TRY", "3"))
    tries: dict[int, int] = {}
    if f.exists():
        for r in json.load(open(f)):
            if not r:
                continue
            tries[r["i"]] = int(r.get("tries", 1))
            # only answers with content count: an empty answer is a failure artifact and must be redone (see done.pop below).
            # but one still empty after MAX_TRY attempts is put into done so the arm can finish, and is counted and reported separately.
            if str(r.get("answer", "")).strip() or tries[r["i"]] >= MAX_TRY:
                done[r["i"]] = r
        giveup = sum(1 for i, r in done.items() if not str(r.get("answer", "")).strip())
        print(f"[{arm}] {len(done)} questions already done ({giveup} given up), resuming the rest", flush=True)

    # consecutive-empty counter (kept in a list so the closure can modify it). When the backend is stuck every question
    # waits the full SENSE_TIMEOUT, and 120 questions could burn 14 hours, so after consecutive failures the arm is aborted at once
    # and the outer script resumes after the service recovers.
    FAIL_FAST = int(os.environ.get("SENSE_FAIL_FAST", "5"))
    nonlocal_fail = [0]
    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()

    def _heal() -> bool:
        """Restart the backend in place. This machine's GPU monitor warns after GPU utilization
        stays below 1% for 600 s and sends SIGTERM 120 s later; the gap between arms (reading parquet, initializing
        GraphRAG) falls exactly into that window, so the backend was reclaimed before the arm started.
        Previously the question waited out SENSE_TIMEOUT before aborting the arm, costing 15 minutes each time."""
        import subprocess
        print("  [heal] backend unavailable, restarting in place", flush=True)
        subprocess.run(["bash", DYLO_ROOT + "/graphrag/health.sh", "heal"],
                       capture_output=True, timeout=900)
        for _ in range(30):
            if _backend_alive():
                print("  [heal] recovered", flush=True)
                return True
            time.sleep(10)
        return False

    async def one(i, q):
        if i in done:
            return
        async with sem:
            # with SENSE_TRACE=1, log each step: locate where the contradiction "backend healthy, 10 seconds when run alone,
            # yet times out on this path" actually happens, instead of guessing.
            trace = os.environ.get("SENSE_TRACE") == "1"
            if trace:
                print(f"  [trace {i}] entered semaphore {time.strftime('%H:%M:%S')}", flush=True)
            # confirm the backend is alive before each question; if it is dead, repair first instead of running into timeouts.
            # wrapped in to_thread: requests is blocking and calling it in the coroutine would stall the event loop.
            if os.environ.get("SENSE_PRECHECK", "1") == "1":
                alive = await asyncio.to_thread(_backend_alive)
                if trace:
                    print(f"  [trace {i}] alive={alive} {time.strftime('%H:%M:%S')}", flush=True)
                if not alive and not await asyncio.to_thread(_heal):
                    raise RuntimeError("backend cannot be started, aborting this arm")
            if trace:
                print(f"  [trace {i}] issuing global_search {time.strftime('%H:%M:%S')}", flush=True)
            t0 = time.time()
            try:
                a, _ = await asyncio.wait_for(api.global_search(
                    config=cfg, entities=ents, communities=comms,
                    community_reports=reps, community_level=1,
                    dynamic_community_selection=True,
                    response_type=RESP_TYPE, query=q["question"]),
                    timeout=TIMEOUT_S)
            except TimeoutError:
                print(f"  [{i}] timeout ({TIMEOUT_S}s)", flush=True)
                a = ""
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] failed {type(e).__name__}: {str(e)[:70]}", flush=True)
                a = ""
            # this GPU is shared, and the backend can be pushed out at any time (measured: at question 8
            # another job started a 32B model occupying 75G and our backend went OOM). After a failure, probe liveness once:
            # if the backend is dead, exit immediately instead of running into TIMEOUT_S per question -- otherwise a single
            # preemption burns FAIL_FAST x TIMEOUT_S (measured 35 minutes).
            if not str(a).strip() and not _backend_alive():
                raise RuntimeError("backend unavailable (completion probe failed), aborting this arm")
            tries[i] = tries.get(i, 0) + 1
            r = {"i": i, "question": q["question"], "user": q["user"],
                 "answer": str(a), "sec": round(time.time() - t0, 1),
                 "tries": tries[i]}
            async with lock:
                done[i] = r
                json.dump([done.get(j) for j in range(len(QS))], open(f, "w"),
                          ensure_ascii=False)
                # when the service is stuck every question times out and records an empty answer; if empty answers counted as done,
                # resuming would skip them and one failure would be frozen permanently -- measured: 65 of 120
                # questions of one arm became empty this way and restarts could not recover them. Empty answers are not counted as done.
                if not str(a).strip():
                    if tries[i] < MAX_TRY:
                        done.pop(i, None)      # retries remain, not done
                    nonlocal_fail[0] += 1
                else:
                    nonlocal_fail[0] = 0
            if nonlocal_fail[0] >= FAIL_FAST:
                raise RuntimeError(
                    f"{nonlocal_fail[0]} consecutive empty answers, backend deemed unavailable, aborting this arm")
            if len(done) % 10 == 0:
                print(f"  progress {len(done)}/{len(QS)}", flush=True)

    t0 = time.time()
    try:
        await asyncio.gather(*(one(i, q) for i, q in enumerate(QS)))
    except RuntimeError as e:
        print(f"[{arm}] aborted early: {e}", flush=True)
    miss = [i for i in range(len(QS)) if i not in done]
    print(f"[{arm}] finished {len(done)}/{len(QS)} questions, {time.time()-t0:.0f}s, "
          f"{len(miss)} pending -> {f}")
    if miss:
        sys.exit(1)   # the outer script uses this to decide whether to wait for the service and resume


def _backend_alive() -> bool:
    """Whether the backend can actually complete a request.

    Checking /v1/models is not enough: 8000 is the proxy, which can still return 200 or
    "Internal Server Error" when the backend is dead, neither of which is a JSON completion. A real choices response is required here.

    The check must be lenient: a question that hits the timeout has often filled the backend's map queue itself (measured: normal
    questions take 8-14 seconds, some hang for 420 seconds), and the liveness probe cannot get through either. A single
    25-second check used to misclassify "backend busy" as "backend dead" and aborted arms at questions 7-11.
    Now two attempts of 60 seconds each, with a 15-second pause in between to let the queue drain.
    """
    for k in range(2):
        try:
            r = requests.post(API, json={
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4, "temperature": 0}, timeout=60)
            if "choices" in r.json():
                return True
        except Exception:  # noqa: BLE001
            pass
        if k == 0:
            time.sleep(15)
    return False


def _ask(prompt: str) -> str:
    r = requests.post(API, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0}, timeout=300)
    return r.json()["choices"][0]["message"]["content"].strip().upper()


def judge(base: str, others: list[str]) -> None:
    from concurrent.futures import ThreadPoolExecutor
    A = json.load(open(ANS / f"{base}.json"))
    for other in others:
        B = json.load(open(ANS / f"{other}.json"))
        print(f"\n=== {base} vs {other} ===")
        for dim, desc in DIMS.items():
            def pair(k):
                a, b = A[k]["answer"], B[k]["answer"]
                if not a.strip() and not b.strip():
                    return "TIE"
                if not a.strip():
                    return other
                if not b.strip():
                    return base
                q = A[k]["question"]
                def ask(x, y):
                    p = (f"Question: {q}\n\nAnswer A:\n{x[:3000]}\n\nAnswer B:\n{y[:3000]}\n\n"
                         f"Criterion - {dim}: {desc}\n"
                         f"Which answer is better on this criterion? "
                         f"Reply with exactly one token: A, B, or TIE.")
                    o = _ask(p)
                    return "A" if o.startswith("A") else ("B" if o.startswith("B") else "TIE")
                # ask in both orders to cancel position preference; disagreement is a tie
                p1, p2 = ask(a, b), ask(b, a)
                if p1 == "A" and p2 == "B":
                    return base
                if p1 == "B" and p2 == "A":
                    return other
                return "TIE"

            with ThreadPoolExecutor(JUDGE_CONC) as ex:
                out = list(ex.map(pair, range(len(A))))
            c = collections.Counter(out)
            n = c[base] + c[other]
            rate = c[base] / n if n else 0.5
            print(f"  {dim:<18} {base} {c[base]:>3} / {other} {c[other]:>3} / tie {c['TIE']:>3}"
                  f"   win rate {rate:.1%}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "answer":
        asyncio.run(answer(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "judge":
        judge(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
