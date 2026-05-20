#!/usr/bin/env python3
"""
Re-run Phase 2 (answer + judge) on a cached fact_experiment_v2 result with a cloud model.

Reads the existing v2 JSON (which has retrieved_facts already cached), re-runs
the answer generation + LLM judge with configurable cloud-tier models, and writes
a new JSON with the cloud answer/verdict.

This skips the ~6h Phase 1 ingestion (already done) and only spends time on
Phase 2 (~180 questions × 2 LLM calls each).

Supports asymmetric judging (different model for answer vs judge) to mitigate
same-class judge bias — see https://github.com/peterjohannmedina/heurchain-benchmarks/blob/main/results/COMPARISON_v2_cloud_judge.md
for the methodology finding that motivated this feature.

Each output record gains a `fact_contains_gold` diagnostic flag that does a
normalized substring match of the gold answer against the retrieved_facts.
This separates "extraction failure" from "answerer failure" in the audit trail.

Usage:
  # Same model for answer + judge (current default, equivalent to v1):
  python3 judge_rerun_cloud.py \
    --input  results/facts_v2_knowledge-update_max30.json \
    --output results/facts_v2cloud-deepseek_knowledge-update_max30.json \
    --model  deepseek-v3.1:671b-cloud

  # Cross-judging: DeepSeek answers, Kimi judges (removes self-bias):
  python3 judge_rerun_cloud.py \
    --input  results/facts_v2_knowledge-update_max30.json \
    --output results/facts_v2cross-ds-ans-kimi-judge_knowledge-update_max30.json \
    --answer-model deepseek-v3.1:671b-cloud \
    --judge-model  kimi-k2.6:cloud

  # Mixed API: local model answers, Anthropic judges:
  python3 judge_rerun_cloud.py \
    --input  results/facts_v2_knowledge-update_max30.json \
    --output results/facts_v2cross-local-ans-sonnet-judge_knowledge-update_max30.json \
    --answer-model medina-14b:latest \
    --answer-base-url http://localhost:11434/v1 \
    --judge-model claude-sonnet-4-6 \
    --judge-api anthropic
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx


# ─── prompts (copied verbatim from fact_experiment.py to keep eval identical) ───

ANSWER_PROMPT = (
    "You are a knowledge assistant. Answer the question using ONLY the FACTS below, "
    "which were extracted from the user's past conversations.\n\n"
    "Each fact is tagged with [Session N] — HIGHER N = LATER in time. "
    "If two facts about the same topic conflict (e.g., user's count, status, score, "
    "preference, or location changes over time), ALWAYS trust the fact from the "
    "highest Session number — it represents the user's most current state.\n\n"
    "FACTS:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "Give a direct, concrete answer in one short sentence. "
    "If no fact answers the question, reply: I don't know.\n\n"
    "ANSWER:"
)


def get_anscheck_prompt(task_type, question, answer, response):
    if task_type in ("single-session-user", "single-session-assistant", "multi-session"):
        tmpl = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task_type == "temporal-reasoning":
        tmpl = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "Do not penalize off-by-one errors for the number of days/weeks/months.\n\n"
            "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task_type == "knowledge-update":
        tmpl = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered correct as long as the updated answer is the required answer.\n\n"
            "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task_type == "single-session-preference":
        tmpl = (
            "I will give you a question, a rubric for desired personalized response, and a response from a model. "
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric.\n\n"
            "Question: {q}\n\nRubric: {a}\n\nModel Response: {r}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        tmpl = "Question: {q}\nGold: {a}\nResponse: {r}\nIs response correct? yes or no only."
    return tmpl.format(q=question, a=answer, r=response)


def parse_verdict(text):
    t = text.lower().strip()
    for m in ("</think>", "answer:", "verdict:"):
        if m in t:
            t = t.split(m)[-1]
    t = t.strip()
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        if last.startswith("yes") or last in ("yes", "yes."):
            return 1
        if last.startswith("no") or last in ("no", "no."):
            return 0
    tokens = t.replace(".", " ").split()
    if "yes" in tokens and "no" not in tokens:
        return 1
    if "no" in tokens and "yes" not in tokens:
        return 0
    return 1 if t.rfind("yes") > t.rfind("no") else 0


# ─── cloud chat dispatcher ───

def openai_compat_chat(base_url, model, prompt, max_tokens=300, timeout=300):
    """Call OpenAI-compatible chat completion (Ollama, vLLM, llama-server, etc).

    Some models (Kimi K2.6, etc.) emit a separate `reasoning` field and may
    have empty `content` if max_tokens was exhausted during reasoning. This
    function returns content if present; otherwise extracts the final answer
    from the reasoning trace (best-effort: the last short paragraph).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if content:
        return content
    # Fallback: reasoning trace's last paragraph
    reasoning = (msg.get("reasoning") or "").strip()
    if reasoning:
        paras = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
        if paras:
            return paras[-1].split("\n")[-1].strip()
        return reasoning.split("\n")[-1].strip()
    return ""


def anthropic_chat(model, prompt, max_tokens=2000, timeout=300):
    """Call Anthropic Messages API. Reads API key from ANTHROPIC_API_KEY env var."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY env var not set")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json=payload,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    blocks = r.json().get("content", [])
    for b in blocks:
        if b.get("type") == "text":
            return b.get("text", "").strip()
    return ""


def cloud_chat(base_url, model, prompt, max_tokens=300, timeout=300, api="openai"):
    """Dispatch to the right API based on api flag."""
    if api == "anthropic":
        return anthropic_chat(model, prompt, max_tokens=max_tokens, timeout=timeout)
    return openai_compat_chat(base_url, model, prompt, max_tokens=max_tokens, timeout=timeout)


def build_fact_context_from_records(retrieved_facts, max_chars=8000):
    """Reproduce build_fact_context with the format used in the original."""
    pieces = []
    used = 0
    for i, fact in enumerate(retrieved_facts, 1):
        line = f"[Session {i}] {fact}"
        if used + len(line) > max_chars:
            break
        pieces.append(line)
        used += len(line) + 1
    return "\n".join(pieces)


# ─── NEW: fact_contains_gold diagnostic ──────────────────────────────────────
#
# Tells us whether the retrieved facts plausibly contain the answer.
# This separates extraction failures (facts don't contain the answer, so no
# answerer can succeed) from answerer failures (facts contain the answer, but
# the model couldn't synthesize it). Reading these two failure modes per
# category is the credibility-critical insight from the cross-judge analysis.

_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")
_STOP = frozenset(
    "a an the and or but of to in on at for from with by is are was were be been "
    "being have has had do does did will would shall should may might must can could "
    "this that these those it its they them their there here also as if then than i "
    "you we my your our me us he she his her him".split()
)


def _content_tokens(s):
    """Lowercase content-word tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOP and len(t) > 1]


def fact_contains_gold_score(gold, retrieved_facts):
    """Return (bool_contains, jaccard_overlap_score).

    Heuristic: take content-word tokens from gold, check what fraction appear
    in concatenated retrieved_facts. >= 0.7 → True (high confidence the answer
    is recoverable from these facts). The score is also returned for analysis.

    Why this matters: the cross-judge analysis showed that even at Hit@10=100%,
    the retrieved FACTS might not contain the answer-bearing detail (extraction
    summarizes too aggressively). This flag exposes that without manual audit.
    """
    if not gold or not retrieved_facts:
        return False, 0.0
    gold_toks = set(_content_tokens(gold))
    if not gold_toks:
        return False, 0.0
    facts_text = " ".join(retrieved_facts).lower()
    facts_toks = set(_content_tokens(facts_text))
    overlap = gold_toks & facts_toks
    score = len(overlap) / len(gold_toks)
    return score >= 0.7, round(score, 3)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Re-run Phase 2 (answer + judge) on a cached fact_experiment result.",
    )
    p.add_argument("--input", required=True, help="cached v2 fact-extraction JSON")
    p.add_argument("--output", required=True, help="output JSON for cloud-judge run")

    # Back-compat: --model + --base-url + --api configure BOTH answer and judge.
    p.add_argument("--model", default=None,
                   help="model for BOTH answer + judge (shortcut). Overridden by --answer-model / --judge-model")
    p.add_argument("--base-url", default="http://192.168.1.242:11434/v1",
                   help="OpenAI-compatible endpoint for the shortcut --model (ignored if --api=anthropic)")
    p.add_argument("--api", default="openai", choices=["openai", "anthropic"],
                   help="API protocol for the shortcut --model")

    # Asymmetric: answer and judge can be different models / endpoints / APIs.
    # This removes same-class judge bias documented in COMPARISON_v2_cloud_judge.md.
    p.add_argument("--answer-model", default=None,
                   help="model used to ANSWER (overrides --model for the answer call)")
    p.add_argument("--answer-base-url", default=None,
                   help="endpoint for answer model (overrides --base-url)")
    p.add_argument("--answer-api", default=None, choices=[None, "openai", "anthropic"],
                   help="API protocol for answer model (overrides --api)")

    p.add_argument("--judge-model", default=None,
                   help="model used to JUDGE (overrides --model for the judge call)")
    p.add_argument("--judge-base-url", default=None,
                   help="endpoint for judge model (overrides --base-url)")
    p.add_argument("--judge-api", default=None, choices=[None, "openai", "anthropic"],
                   help="API protocol for judge model (overrides --api)")

    p.add_argument("--limit", type=int, default=0, help="cap on records (0 = all)")
    args = p.parse_args()

    # Resolve answer/judge config: explicit flag → shortcut --model fallback
    answer_model = args.answer_model or args.model
    answer_base_url = args.answer_base_url or args.base_url
    answer_api = args.answer_api or args.api
    judge_model = args.judge_model or args.model
    judge_base_url = args.judge_base_url or args.base_url
    judge_api = args.judge_api or args.api

    if not answer_model or not judge_model:
        print("FAIL: must provide --model (shortcut for both) OR both --answer-model and --judge-model",
              file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"FAIL: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text())
    records = data["records"]
    if args.limit > 0:
        records = records[:args.limit]

    cross_judging = (answer_model != judge_model) or (answer_api != judge_api)

    print(f"Re-judging {len(records)} records from {input_path.name}")
    print(f"  Answer model: {answer_model}  (api={answer_api})")
    if answer_api == "openai":
        print(f"    endpoint:   {answer_base_url}")
    print(f"  Judge model:  {judge_model}  (api={judge_api})")
    if judge_api == "openai":
        print(f"    endpoint:   {judge_base_url}")
    if cross_judging:
        print(f"  Cross-judging: ON (different model for answer vs judge — removes self-bias)")
    print(f"  Category:     {data['question_type']}")
    sys.stdout.flush()

    new_records = []
    correct_count = 0
    contains_gold_count = 0
    refusal_count = 0
    t_start = time.time()
    fail_count = 0

    for i, rec in enumerate(records):
        question = rec["question"]
        gold = rec["gold_answer"]
        question_type = rec["question_type"]
        retrieved_facts = rec["retrieved_facts"]

        # NEW: diagnostic — do the retrieved facts plausibly contain the gold answer?
        contains_gold, overlap_score = fact_contains_gold_score(gold, retrieved_facts)
        if contains_gold:
            contains_gold_count += 1

        ctx = build_fact_context_from_records(retrieved_facts)
        ans_prompt = ANSWER_PROMPT.format(context=ctx, question=question)

        try:
            response = cloud_chat(answer_base_url, answer_model, ans_prompt,
                                  max_tokens=2000, api=answer_api)
        except Exception as e:
            response = f"[GEN ERROR: {e}]"
            fail_count += 1

        # NEW: track "I don't know" refusal rate (calibration signal)
        if "i don't know" in (response or "").lower() or response.strip() == "":
            refusal_count += 1

        judge_prompt = get_anscheck_prompt(question_type, question, gold, response)
        try:
            verdict = cloud_chat(judge_base_url, judge_model, judge_prompt,
                                 max_tokens=800, api=judge_api)
        except Exception as e:
            verdict = f"[JUDGE ERROR: {e}]"
            fail_count += 1

        correct = parse_verdict(verdict)
        correct_count += correct

        new_records.append({
            **rec,
            "model_response": response,
            "judge_verdict": verdict,
            "correct": correct,
            # NEW diagnostic fields
            "fact_contains_gold": contains_gold,
            "fact_gold_overlap_score": overlap_score,
            "answer_model": answer_model,
            "judge_model": judge_model,
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            running = correct_count / (i + 1)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(records)}] QA={running*100:.1f}%  "
                  f"contains_gold={contains_gold_count/(i+1)*100:.0f}%  "
                  f"refusals={refusal_count/(i+1)*100:.0f}%  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s, fails={fail_count})",
                  flush=True)

    n = len(new_records) or 1
    overall_acc = correct_count / n
    contains_gold_rate = contains_gold_count / n
    refusal_rate = refusal_count / n
    out = {
        **data,
        "n_evaluated": n,
        "n_correct": correct_count,
        "overall_qa_acc": round(overall_acc, 4),
        "records": new_records,
        "cloud_judge": {
            "answer_model": answer_model,
            "answer_api": answer_api,
            "answer_base_url": answer_base_url if answer_api == "openai" else None,
            "judge_model": judge_model,
            "judge_api": judge_api,
            "judge_base_url": judge_base_url if judge_api == "openai" else None,
            "cross_judging": cross_judging,
            "rerun_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_time_s": round(time.time() - t_start, 1),
            "fail_count": fail_count,
            # NEW: extraction-quality + calibration summary stats
            "fact_contains_gold_rate": round(contains_gold_rate, 3),
            "refusal_rate": round(refusal_rate, 3),
        },
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    label = answer_model if not cross_judging else f"{answer_model} → {judge_model}"
    print(f"\n=== {label} on {data['question_type']} ===")
    print(f"QA accuracy:           {overall_acc*100:.2f}%  ({correct_count}/{n})")
    print(f"Hit@10 (cached):       {data['retrieval_recall_at_k']*100:.2f}%")
    print(f"Facts contain gold:    {contains_gold_rate*100:.1f}%  (extraction quality)")
    print(f"Refusal rate:          {refusal_rate*100:.1f}%  (\"I don't know\" / empty)")
    print(f"Wall time:             {(time.time() - t_start)/60:.1f} min")
    print(f"Failures:              {fail_count}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
