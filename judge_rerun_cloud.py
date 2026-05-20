#!/usr/bin/env python3
"""
Re-run Phase 2 (answer + judge) on a cached fact_experiment_v2 result with a cloud model.

Reads the existing v2 JSON (which has retrieved_facts already cached), re-runs
the answer generation + LLM judge with a configurable cloud-tier model via the
Ollama REST endpoint on ClawBaby, and writes a new JSON with the cloud
answer/verdict.

This skips the ~6h Phase 1 ingestion (already done) and only spends time on
Phase 2 (~180 questions × 2 LLM calls each).

Usage:
  python3 judge_rerun_cloud.py \
    --input  ~/heurchain-bench/results/facts_v2_knowledge-update_max30.json \
    --output ~/heurchain-bench/results/facts_v2cloud-deepseek_knowledge-update_max30.json \
    --model  deepseek-v3.1:671b-cloud \
    --base-url http://192.168.1.242:11434/v1
"""

import argparse
import json
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


# ─── cloud chat (OpenAI-compatible Ollama endpoint) ───

def cloud_chat(base_url, model, prompt, max_tokens=300, timeout=300):
    """Call OpenAI-compatible chat completion. Handles reasoning-mode models.

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
    # Fallback: reasoning trace's last line/paragraph (reasoning-mode model with truncated content)
    reasoning = (msg.get("reasoning") or "").strip()
    if reasoning:
        # Take last non-empty paragraph; collapse to single line
        paras = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
        if paras:
            return paras[-1].split("\n")[-1].strip()
        return reasoning.split("\n")[-1].strip()
    return ""


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="cached v2 fact-extraction JSON")
    p.add_argument("--output", required=True, help="output JSON for cloud-judge run")
    p.add_argument("--model", required=True, help="model name on the cloud endpoint")
    p.add_argument("--base-url", default="http://192.168.1.242:11434/v1",
                   help="OpenAI-compatible endpoint")
    p.add_argument("--limit", type=int, default=0, help="cap on records (0 = all)")
    args = p.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"FAIL: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text())
    records = data["records"]
    if args.limit > 0:
        records = records[:args.limit]

    print(f"Re-judging {len(records)} records from {input_path.name}")
    print(f"  Model:     {args.model}")
    print(f"  Endpoint:  {args.base_url}")
    print(f"  Category:  {data['question_type']}")
    sys.stdout.flush()

    new_records = []
    correct_count = 0
    t_start = time.time()
    fail_count = 0

    for i, rec in enumerate(records):
        question = rec["question"]
        gold = rec["gold_answer"]
        question_type = rec["question_type"]
        retrieved_facts = rec["retrieved_facts"]

        ctx = build_fact_context_from_records(retrieved_facts)
        ans_prompt = ANSWER_PROMPT.format(context=ctx, question=question)

        try:
            # max_tokens=2000 leaves room for reasoning-mode models (Kimi K2.6)
            # to emit content after their reasoning trace; non-reasoning models
            # stop naturally well below this cap.
            response = cloud_chat(args.base_url, args.model, ans_prompt, max_tokens=2000)
        except Exception as e:
            response = f"[GEN ERROR: {e}]"
            fail_count += 1

        judge_prompt = get_anscheck_prompt(question_type, question, gold, response)
        try:
            # Higher judge budget for reasoning models that "think" before yes/no.
            verdict = cloud_chat(args.base_url, args.model, judge_prompt, max_tokens=800)
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
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            running = correct_count / (i + 1)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(records)}] running QA acc: {running*100:.1f}%  "
                  f"({elapsed:.0f}s elapsed, {eta:.0f}s ETA, fails={fail_count})",
                  flush=True)

    overall_acc = correct_count / max(1, len(new_records))
    out = {
        **data,
        "n_evaluated": len(new_records),
        "n_correct": correct_count,
        "overall_qa_acc": round(overall_acc, 4),
        "records": new_records,
        "cloud_judge": {
            "model": args.model,
            "base_url": args.base_url,
            "rerun_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_time_s": round(time.time() - t_start, 1),
            "fail_count": fail_count,
        },
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== {args.model} on {data['question_type']} ===")
    print(f"QA accuracy:   {overall_acc*100:.2f}%  ({correct_count}/{len(new_records)})")
    print(f"Hit@10 (cached): {data['retrieval_recall_at_k']*100:.2f}%")
    print(f"Wall time:     {(time.time() - t_start)/60:.1f} min")
    print(f"Failures:      {fail_count}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
