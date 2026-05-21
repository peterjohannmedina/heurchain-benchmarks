#!/usr/bin/env python3
"""
Inner-loop tool for fast ANSWER prompt iteration.

Sibling to targeted_reextract.py — same idea but iterates the ANSWER_PROMPT
instead of the EXTRACTION_PROMPT. Since facts are already cached in the
baseline JSON, this skips extraction entirely.

Wall time: ~3-5 min for 22 questions vs ~10 min for targeted_reextract.

Accepts result JSONs in TWO shapes:
  (a) cross-judge format: records[].retrieved_facts (the standard shape from
      judge_rerun_cloud.py)
  (b) targeted_reextract output: records[].new_facts (re-extracted with a
      candidate prompt) — lets us test answer prompts against the latest
      extraction iteration without re-extracting.

The new ANSWER_PROMPT may reference {question_date}, {context}, {question}.
question_date is auto-pulled from the LongMemEval instance.

Usage:
  # Test v4a answer prompt on the v4 extraction output:
  python3 targeted_reanswer.py \
    --baseline /tmp/v4_targeted_tr_extraction.json \
    --new-prompt ~/heurchain-bench/answer_prompt_v4a.py \
    --judge-model deepseek-v3.1:671b-cloud \
    --judge-base-url http://192.168.1.242:11434/v1

  # Test v4a on a standard cross-judge result:
  python3 targeted_reanswer.py \
    --baseline results/facts_v3cloud-deepseek_temporal-reasoning_max30.json \
    --new-prompt answer_prompt_v4a.py \
    --judge-model deepseek-v3.1:671b-cloud
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_rerun_cloud import (
    get_anscheck_prompt, parse_verdict,
    cloud_chat, build_fact_context_from_records, fact_contains_gold_score,
)


def load_prompt_template(prompt_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_candidate_answer", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        v = getattr(mod, name)
        if name.startswith("ANSWER_PROMPT") and isinstance(v, str):
            return v
    raise ValueError(f"No ANSWER_PROMPT_* string found in {prompt_path}")


def get_facts(record):
    """Pull retrieved_facts from a record — handles both shapes."""
    if "new_facts" in record:
        return record["new_facts"], "new_facts"
    if "retrieved_facts" in record:
        return record["retrieved_facts"], "retrieved_facts"
    return [], None


def get_baseline_correct(record):
    """Return the baseline correctness — handles both shapes."""
    if "new_correct" in record:
        # targeted_reextract output — baseline_correct is the v3 result
        return int(record.get("baseline_correct", 0))
    return int(record.get("correct", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="result JSON (cross-judge or targeted_reextract output)")
    p.add_argument("--new-prompt", required=True,
                   help=".py file defining ANSWER_PROMPT_* string")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", default="http://192.168.1.242:11434/v1")
    p.add_argument("--judge-api", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--longmemeval-data",
                   default=str(Path.home() / "heurchain-bench" / "bench_data"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).expanduser().read_text())
    records = baseline.get("records", [])
    if args.limit > 0:
        records = records[:args.limit]

    prompt_tmpl = load_prompt_template(args.new_prompt)
    prompt_name = Path(args.new_prompt).stem
    needs_qdate = "{question_date}" in prompt_tmpl

    # Load LongMemEval for question_date lookup
    if needs_qdate:
        sys.path.insert(0, args.longmemeval_data)
        from load_longmemeval import load_split  # noqa
        # Need raw instances (load_longmemeval doesn't expose question_date via
        # iter_retrieval_tasks). Pull from the raw JSON.
        split = baseline.get("split", "s")
        instances = load_split(split)
        qdate_by_qid = {inst["question_id"]: inst.get("question_date", "")
                        for inst in instances}
    else:
        qdate_by_qid = {}

    cat = baseline.get("question_type") or records[0].get("question_type", "?")
    print(f"Targeted re-answer: {len(records)} records from {Path(args.baseline).name}")
    print(f"  Category:        {cat}")
    print(f"  New ANSWER:      {prompt_name}")
    print(f"  Uses {'{'+'question_date'+'}'}: {needs_qdate}")
    print(f"  Judge:           {args.judge_model}")
    sys.stdout.flush()

    new_records = []
    correct_count = 0
    baseline_correct_count = 0
    refusal_count = 0
    t_start = time.time()

    for i, r in enumerate(records):
        question = r.get("question") or r.get("query") or ""
        gold = r.get("gold_answer") or r.get("answer") or ""
        question_type = r.get("question_type", cat)
        facts, facts_field = get_facts(r)
        if not facts:
            print(f"  WARN: no facts in record {i}, skipping"); continue

        qid = r.get("question_id", f"rec{i}")
        baseline_corr = get_baseline_correct(r)
        baseline_correct_count += baseline_corr

        ctx = build_fact_context_from_records(facts)
        fmt_kwargs = {"context": ctx, "question": question}
        if needs_qdate:
            fmt_kwargs["question_date"] = qdate_by_qid.get(qid, "(unknown)")
        ans_prompt = prompt_tmpl.format(**fmt_kwargs)

        try:
            response = cloud_chat(args.judge_base_url, args.judge_model, ans_prompt,
                                  max_tokens=2000, api=args.judge_api)
        except Exception as e:
            response = f"[GEN ERROR: {e}]"
        if "i don't know" in (response or "").lower() or not response.strip():
            refusal_count += 1

        judge_prompt = get_anscheck_prompt(question_type, question, gold, response)
        try:
            verdict = cloud_chat(args.judge_base_url, args.judge_model, judge_prompt,
                                 max_tokens=800, api=args.judge_api)
        except Exception as e:
            verdict = f"[JUDGE ERROR: {e}]"
        correct = parse_verdict(verdict)
        correct_count += correct

        contains_gold, overlap = fact_contains_gold_score(gold, facts)
        new_records.append({
            "question_id": qid,
            "question_type": question_type,
            "question": question,
            "gold_answer": str(gold),
            "question_date": qdate_by_qid.get(qid, ""),
            "facts_source": facts_field,
            "facts": facts,
            "baseline_correct": baseline_corr,
            "new_response": response,
            "new_verdict": verdict,
            "new_correct": correct,
            "delta": correct - baseline_corr,
            "fact_contains_gold": contains_gold,
            "fact_gold_overlap_score": overlap,
        })

        wins = sum(1 for rr in new_records if rr["delta"] > 0)
        losses = sum(1 for rr in new_records if rr["delta"] < 0)
        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(records)}] wins={wins} losses={losses}  "
                  f"refusals={refusal_count}  ({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t_start
    wins = sum(1 for r in new_records if r["delta"] > 0)
    losses = sum(1 for r in new_records if r["delta"] < 0)
    contains_gold_n = sum(1 for r in new_records if r["fact_contains_gold"])
    n = len(new_records)
    print()
    print(f"=== {prompt_name} on {cat} (n={n}) ===")
    print(f"Baseline correct:           {baseline_correct_count}/{n} ({baseline_correct_count/max(1,n)*100:.1f}%)")
    print(f"With new ANSWER prompt:     {correct_count}/{n} ({correct_count/max(1,n)*100:.1f}%)")
    print(f"  Wins (0→1):              {wins}")
    print(f"  Losses (1→0):            {losses}")
    print(f"  Net lift:                {correct_count - baseline_correct_count:+d}")
    print(f"  Refusal rate:            {refusal_count/max(1,n)*100:.1f}%")
    print(f"  Facts contain gold:      {contains_gold_n}/{n} ({contains_gold_n/max(1,n)*100:.1f}%)")
    print(f"Wall time:                  {elapsed/60:.1f} min")

    out = {
        "baseline_path": str(args.baseline),
        "answer_prompt_path": str(args.new_prompt),
        "prompt_name": prompt_name,
        "category": cat,
        "n_records": n,
        "baseline_correct": baseline_correct_count,
        "new_correct": correct_count,
        "wins": wins,
        "losses": losses,
        "refusal_rate": round(refusal_count / max(1, n), 3),
        "wall_time_s": round(elapsed, 1),
        "records": new_records,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/answer_{prompt_name}_{cat}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
