#!/usr/bin/env python3
"""
Inner-loop tool for fast prompt iteration.

Takes a list of question_ids (the EXTRACTION failures from triage_failures.py),
re-extracts JUST the relevant sessions for those questions using a candidate
new prompt, then re-runs the answer+judge step on the new facts.

Wall time: ~3-5 min for 10 questions vs ~6h for a full category re-extract.
This is the difference between a Karpathy cycle and a glacial one.

Reads the cached v2/v3 fact-extraction JSON to know which sessions are
relevant per question. Doesn't need the full dataset on disk.

Usage:
  # Test a new extraction prompt on the EXTRACTION failures of one category:
  python3 targeted_reextract.py \
    --baseline results/facts_v3cloud-deepseek_temporal-reasoning_max30.json \
    --new-prompt extraction_prompt_v4.py \
    --judge-model deepseek-v3.1:671b-cloud \
    --judge-base-url http://192.168.1.242:11434/v1 \
    --extract-base-url http://localhost:8080/v1 \
    --extract-model none  # uses llama-server local model, ignores this arg
    --question-ids q1,q2,q3   # or omit to use ALL EXTRACTION failures
"""
import argparse, json, re, sys, time
from pathlib import Path
import httpx


# Reuse the helpers from judge_rerun_cloud (lives next to this file)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_rerun_cloud import (
    ANSWER_PROMPT, get_anscheck_prompt, parse_verdict,
    cloud_chat, build_fact_context_from_records, fact_contains_gold_score,
)


def load_prompt_template(prompt_path):
    """Import a Python file and pull the first EXTRACTION_PROMPT_* string."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_candidate", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("EXTRACTION_PROMPT") and isinstance(getattr(mod, name), str):
            return getattr(mod, name)
    raise ValueError(f"No EXTRACTION_PROMPT_* string found in {prompt_path}")


def reextract_session(session_text, prompt_template, extract_base_url, extract_model, max_tokens=2400):
    """Re-extract facts from a single session with a candidate prompt."""
    prompt = prompt_template.format(session_text=session_text[:6000])
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extract_model and extract_model != "none":
        payload["model"] = extract_model
    r = httpx.post(f"{extract_base_url}/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    # JSON-list parse (same as fact_extraction.py)
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"```\s*$", "", content)
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="result JSON to triage — failures from here will be re-tested")
    p.add_argument("--new-prompt", required=True,
                   help=".py file defining EXTRACTION_PROMPT_* string")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-base-url", default="http://192.168.1.242:11434/v1")
    p.add_argument("--judge-api", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--extract-base-url", default="http://localhost:8080/v1",
                   help="extraction endpoint (llama-server)")
    p.add_argument("--extract-model", default="none")
    p.add_argument("--question-ids", default=None,
                   help="comma-sep question_ids; omit to use all EXTRACTION failures")
    p.add_argument("--mode", default="EXTRACTION",
                   choices=["EXTRACTION", "ANSWERER", "all_failures"],
                   help="which subset of records to re-test")
    p.add_argument("--longmemeval-data",
                   default=str(Path.home() / "heurchain-bench" / "bench_data"),
                   help="path containing LongMemEval data (for raw session text)")
    p.add_argument("--output", default=None,
                   help="output JSON; default derived from --new-prompt filename")
    args = p.parse_args()

    # Load baseline + extract failed question_ids
    baseline = json.loads(Path(args.baseline).expanduser().read_text())
    records_by_qid = {r["question_id"]: r for r in baseline["records"]}

    if args.question_ids:
        target_qids = [q.strip() for q in args.question_ids.split(",") if q.strip()]
    else:
        # Pick failures based on --mode
        target_qids = []
        for r in baseline["records"]:
            if r["correct"]:
                continue
            if args.mode == "all_failures":
                target_qids.append(r["question_id"])
            elif args.mode == "EXTRACTION":
                if r.get("retrieval_hit", 0) and r.get("fact_contains_gold") is False:
                    target_qids.append(r["question_id"])
            elif args.mode == "ANSWERER":
                if r.get("retrieval_hit", 0) and r.get("fact_contains_gold") is True:
                    target_qids.append(r["question_id"])
    if not target_qids:
        print(f"No {args.mode} failures found in baseline."); sys.exit(0)

    # Load the candidate prompt
    prompt_tmpl = load_prompt_template(args.new_prompt)
    prompt_name = Path(args.new_prompt).stem

    # Load LongMemEval data to access raw session text
    sys.path.insert(0, args.longmemeval_data)
    from load_longmemeval import load_split, iter_retrieval_tasks, session_to_text  # noqa
    instances = load_split(baseline.get("split", "s"))
    tasks_by_qid = {t["question_id"]: t for t in iter_retrieval_tasks(instances)}

    cat = baseline["question_type"]
    print(f"Targeted re-extract: {len(target_qids)} questions from {cat}")
    print(f"  Mode:           {args.mode}")
    print(f"  Baseline judge: {(baseline.get('cloud_judge') or {}).get('judge_model') or 'local'}")
    print(f"  New prompt:     {prompt_name}")
    print(f"  Judge model:    {args.judge_model}")
    print(f"  Extract via:    {args.extract_base_url}")
    sys.stdout.flush()

    new_records = []
    correct_count = 0
    baseline_correct_on_these = 0
    t_start = time.time()

    for i, qid in enumerate(target_qids):
        baseline_rec = records_by_qid.get(qid)
        if not baseline_rec:
            print(f"  WARN: {qid} not in baseline"); continue
        baseline_correct_on_these += int(baseline_rec["correct"])
        task = tasks_by_qid.get(qid)
        if not task:
            print(f"  WARN: {qid} not in dataset"); continue

        # Re-extract facts from each relevant session with the new prompt
        new_facts = []
        for sess in task["sessions"]:
            if sess["session_id"] not in task["relevant_session_ids"]:
                continue  # only re-extract from sessions that retrieval already found relevant
            text = session_to_text(sess["turns"])
            try:
                facts = reextract_session(text, prompt_tmpl, args.extract_base_url, args.extract_model)
                new_facts.extend(facts)
            except Exception as e:
                print(f"  WARN extract failed for {qid}/{sess['session_id']}: {e}")

        # Truncate to same top-k the baseline used
        top_k = baseline.get("top_k", 10)
        new_facts = new_facts[:top_k]

        # Build the answer prompt + re-run answer + judge
        ctx = build_fact_context_from_records(new_facts)
        ans_prompt = ANSWER_PROMPT.format(context=ctx, question=task["query"])
        try:
            response = cloud_chat(args.judge_base_url, args.judge_model, ans_prompt,
                                  max_tokens=2000, api=args.judge_api)
        except Exception as e:
            response = f"[GEN ERROR: {e}]"

        judge_prompt = get_anscheck_prompt(task["question_type"], task["query"],
                                            task["answer"], response)
        try:
            verdict = cloud_chat(args.judge_base_url, args.judge_model, judge_prompt,
                                 max_tokens=800, api=args.judge_api)
        except Exception as e:
            verdict = f"[JUDGE ERROR: {e}]"
        correct = parse_verdict(verdict)
        correct_count += correct

        contains_gold, overlap = fact_contains_gold_score(task["answer"], new_facts)
        new_records.append({
            "question_id": qid,
            "question_type": task["question_type"],
            "question": task["query"],
            "gold_answer": str(task["answer"]),
            "baseline_correct": int(baseline_rec["correct"]),
            "baseline_facts": baseline_rec["retrieved_facts"],
            "baseline_response": baseline_rec.get("model_response", ""),
            "new_facts": new_facts,
            "new_response": response,
            "new_verdict": verdict,
            "new_correct": correct,
            "delta": correct - int(baseline_rec["correct"]),
            "new_fact_contains_gold": contains_gold,
            "new_fact_gold_overlap_score": overlap,
        })
        wins = sum(1 for r in new_records if r["delta"] > 0)
        losses = sum(1 for r in new_records if r["delta"] < 0)
        if (i + 1) % 5 == 0 or (i + 1) == len(target_qids):
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(target_qids)}] wins={wins} losses={losses}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t_start
    wins = sum(1 for r in new_records if r["delta"] > 0)
    losses = sum(1 for r in new_records if r["delta"] < 0)
    contains_gold_n = sum(1 for r in new_records if r["new_fact_contains_gold"])

    print()
    print(f"=== {prompt_name} on {cat} {args.mode} failures (n={len(new_records)}) ===")
    print(f"Baseline correct on these:  {baseline_correct_on_these}/{len(new_records)} "
          f"({baseline_correct_on_these/max(1,len(new_records))*100:.1f}%)")
    print(f"New prompt correct:         {correct_count}/{len(new_records)} "
          f"({correct_count/max(1,len(new_records))*100:.1f}%)")
    print(f"  Wins (0→1):              {wins}")
    print(f"  Losses (1→0):            {losses}")
    print(f"  Facts contain gold now:  {contains_gold_n}/{len(new_records)} "
          f"({contains_gold_n/max(1,len(new_records))*100:.1f}%)")
    print(f"Wall time:                  {elapsed/60:.1f} min")

    out = {
        "baseline_path": str(args.baseline),
        "prompt_path": str(args.new_prompt),
        "prompt_name": prompt_name,
        "category": cat,
        "mode_filter": args.mode,
        "n_records": len(new_records),
        "baseline_correct": baseline_correct_on_these,
        "new_correct": correct_count,
        "wins": wins,
        "losses": losses,
        "contains_gold_n": contains_gold_n,
        "wall_time_s": round(elapsed, 1),
        "records": new_records,
    }
    out_path = Path(args.output) if args.output else Path(
        f"results/targeted_{prompt_name}_{cat}_{args.mode.lower()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
