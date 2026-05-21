#!/usr/bin/env python3
"""
Test answerer against haystack-wide v4 facts.

Reads the output of haystack_reextract.py (which has ALL haystack sessions
re-extracted with v4 prompt, not just retrieved ones), and asks an answerer
to use them. This is the v4c upper bound: if the right event session was in
the haystack at all, v4 extracted it; if we pass ALL haystack v4 facts to
the answerer, can it answer correctly?

If YES → v4c (event index) is justified: a smarter retriever could give the
answerer just the right subset.

If NO → the bottleneck is the answerer itself even with full information,
and we need a more capable answer model.

Three modes:
  --strategy all      : pass all facts (truncated to context window)
  --strategy top-k    : do a simple lexical re-rank, take top-k
  --strategy event-idx: use build_event_index.py to query relevant events,
                        pass index hits + a few top-k facts

Usage:
  python3 test_haystack_answer.py \
    --haystack-input results/haystack_v4_temporal_reasoning.json \
    --new-prompt answer_prompt_v4b.py \
    --judge-model deepseek-v3.1:671b-cloud \
    --strategy event-idx \
    --output results/v4c_haystack_eventidx_test.json
"""
import argparse, json, re, sys, time
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_rerun_cloud import (
    get_anscheck_prompt, parse_verdict,
    cloud_chat, build_fact_context_from_records,
)
from targeted_reanswer import load_prompt_template


# Lightweight inline event-index helpers (parallel to build_event_index.py but
# in-memory per-question — no SQLite needed for this test).
ISO_RE = re.compile(r"\[date:\s*(\d{4}-\d{2}-\d{2})\]")
SESSID_RE = re.compile(r"\[SessionID\s+(\S+?)\]")


def parse_iso(fact):
    m = ISO_RE.search(fact)
    return m.group(1) if m else None


def lexical_topk(question, facts, k=10):
    """Bag-of-words overlap score; cheap stand-in for proper retrieval."""
    q_words = set(w.lower() for w in re.findall(r"\b\w{3,}\b", question))
    stop = set("how many days weeks months years ago between since when did the and that they i my our".split())
    q_words -= stop
    scored = []
    for f in facts:
        f_words = set(w.lower() for w in re.findall(r"\b\w{3,}\b", f))
        scored.append((len(q_words & f_words), f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:k]]


def event_idx_strategy(question, facts, k=10):
    """For each candidate event noun phrase in the question, find facts with
    matching keywords AND an ISO date tag. Returns up to 2k facts total."""
    # Pull candidate event keywords from question: capitalized words + 'theNoun'
    q_keywords = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", question)
    q_keywords += re.findall(r"\b(?:visit|meet|attend|buy|start|finish|watch|complete|cancel|receive)[a-z]*\b", question.lower())
    # All facts with ISO dates
    dated = [f for f in facts if parse_iso(f)]
    # Score each dated fact by keyword overlap with question
    scored = []
    for f in dated:
        f_low = f.lower()
        hits = sum(1 for kw in q_keywords if kw.lower() in f_low)
        if hits > 0:
            scored.append((hits, parse_iso(f), f))
    scored.sort(key=lambda x: (-x[0], x[1]))  # most keyword hits, then earliest date
    out = [f for _, _, f in scored[:k]]
    # Pad with lexical top-k if fewer than k dated hits
    if len(out) < k:
        remaining = [f for f in lexical_topk(question, facts, k=k*2) if f not in out]
        out += remaining[:k - len(out)]
    return out


def get_question_date(qid, instances):
    for inst in instances:
        if inst["question_id"] == qid:
            return inst.get("question_date", "")
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--haystack-input", required=True,
                   help="output of haystack_reextract.py")
    p.add_argument("--new-prompt", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", default="http://192.168.1.242:11434/v1")
    p.add_argument("--judge-api", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--strategy", default="event-idx",
                   choices=["all", "top-k", "event-idx"])
    p.add_argument("--top-k", type=int, default=20,
                   help="number of facts to pass to answerer (per question)")
    p.add_argument("--longmemeval-data",
                   default=str(Path.home() / "heurchain-bench" / "bench_data"))
    p.add_argument("--output", required=True)
    args = p.parse_args()

    haystack = json.loads(Path(args.haystack_input).expanduser().read_text())
    records = haystack["records"]

    prompt_tmpl = load_prompt_template(args.new_prompt)
    prompt_name = Path(args.new_prompt).stem
    needs_qdate = "{question_date}" in prompt_tmpl

    if needs_qdate:
        sys.path.insert(0, args.longmemeval_data)
        from load_longmemeval import load_split
        instances = load_split(haystack.get("split", "s"))
    else:
        instances = []

    print(f"Test haystack v4 answerer: {len(records)} questions")
    print(f"  Strategy:        {args.strategy} (top-{args.top_k} facts)")
    print(f"  Answer prompt:   {prompt_name}")
    print(f"  Judge:           {args.judge_model}")
    sys.stdout.flush()

    new_records = []
    correct_count = 0
    baseline_correct = 0
    refusal_count = 0
    t_start = time.time()

    for i, r in enumerate(records):
        question = r["question"]
        gold = r["gold_answer"]
        qid = r["question_id"]
        all_facts = r["haystack_facts"]
        baseline_corr = int(r.get("baseline_correct", 0))
        baseline_correct += baseline_corr

        # Apply chosen retrieval strategy on the haystack facts
        if args.strategy == "all":
            # Truncate to context budget
            facts_for_answer = all_facts[:args.top_k * 3]
        elif args.strategy == "top-k":
            facts_for_answer = lexical_topk(question, all_facts, k=args.top_k)
        elif args.strategy == "event-idx":
            facts_for_answer = event_idx_strategy(question, all_facts, k=args.top_k)

        ctx = build_fact_context_from_records(facts_for_answer)
        fmt_kwargs = {"context": ctx, "question": question}
        if needs_qdate:
            fmt_kwargs["question_date"] = get_question_date(qid, instances) or "(unknown)"
        ans_prompt = prompt_tmpl.format(**fmt_kwargs)

        try:
            response = cloud_chat(args.judge_base_url, args.judge_model, ans_prompt,
                                  max_tokens=2000, api=args.judge_api)
        except Exception as e:
            response = f"[GEN ERROR: {e}]"
        if "i don't know" in (response or "").lower() or not response.strip():
            refusal_count += 1

        judge_prompt = get_anscheck_prompt(r["question_type"], question, gold, response)
        try:
            verdict = cloud_chat(args.judge_base_url, args.judge_model, judge_prompt,
                                 max_tokens=800, api=args.judge_api)
        except Exception as e:
            verdict = f"[JUDGE ERROR: {e}]"
        correct = parse_verdict(verdict)
        correct_count += correct

        new_records.append({
            "question_id": qid,
            "question_type": r["question_type"],
            "question": question,
            "gold_answer": gold,
            "n_haystack_facts": len(all_facts),
            "n_facts_passed": len(facts_for_answer),
            "facts_passed": facts_for_answer,
            "response": response,
            "verdict": verdict,
            "correct": correct,
            "baseline_correct": baseline_corr,
            "delta": correct - baseline_corr,
        })
        if (i + 1) % 3 == 0 or (i + 1) == len(records):
            wins = sum(1 for r in new_records if r["delta"] > 0)
            losses = sum(1 for r in new_records if r["delta"] < 0)
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(records)}] wins={wins} losses={losses} refusals={refusal_count}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t_start
    wins = sum(1 for r in new_records if r["delta"] > 0)
    losses = sum(1 for r in new_records if r["delta"] < 0)
    n = len(new_records)
    print()
    print(f"=== test_haystack_answer ({args.strategy}, top-{args.top_k}) on n={n} ===")
    print(f"Baseline correct:        {baseline_correct}/{n} ({baseline_correct/max(1,n)*100:.1f}%)")
    print(f"Haystack + {args.strategy}:  {correct_count}/{n} ({correct_count/max(1,n)*100:.1f}%)")
    print(f"  Wins (0→1):           {wins}")
    print(f"  Losses (1→0):         {losses}")
    print(f"  Net lift:             {correct_count - baseline_correct:+d}")
    print(f"  Refusal rate:         {refusal_count/max(1,n)*100:.1f}%")
    print(f"Wall time:               {elapsed/60:.1f} min")

    out = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "haystack_input": str(args.haystack_input),
        "answer_prompt": str(args.new_prompt),
        "n_records": n,
        "baseline_correct": baseline_correct,
        "new_correct": correct_count,
        "wins": wins,
        "losses": losses,
        "wall_time_s": round(elapsed, 1),
        "records": new_records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
