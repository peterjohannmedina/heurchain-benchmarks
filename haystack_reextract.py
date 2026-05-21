#!/usr/bin/env python3
"""
Haystack-wide re-extraction — for each failed question, re-extract v4 facts
from ALL sessions in that question's haystack (not just the ones v3 retrieval
already surfaced). Used to test v4c (event index) properly: gives the event
index a fair chance to contain dates for sessions that v3 retrieval missed.

Wall time: significantly higher than targeted_reextract.py because we process
ALL haystack sessions per question, not just relevant ones. ~30-50 sessions
per question × 15s per extraction ≈ 7-12 min per question.

For 12 retrieval-miss failures × ~10 min each ≈ 2-3 hours.

Usage:
  python3 haystack_reextract.py \
    --baseline /tmp/v4_v4b_combined.json \
    --new-prompt ~/heurchain-bench/extraction_prompt_v4.py \
    --question-ids gpt4_59149c77,gpt4_e072b769,...  # or omit for all retrieval-miss
    --output results/haystack_v4_temporal_reasoning.json
"""
import argparse, json, re, sys, time
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_rerun_cloud import build_fact_context_from_records, fact_contains_gold_score


def load_prompt_template(prompt_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cand", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        v = getattr(mod, name)
        if name.startswith("EXTRACTION_PROMPT") and isinstance(v, str):
            return v
    raise ValueError(f"No EXTRACTION_PROMPT_* in {prompt_path}")


def extract(session_text, prompt_tmpl, base_url, model, session_date=None,
            max_tokens=2400, timeout=300):
    fmt = {"session_text": session_text[:6000]}
    if "{session_date}" in prompt_tmpl:
        fmt["session_date"] = session_date or "(unknown)"
    prompt = prompt_tmpl.format(**fmt)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if model and model != "none":
        payload["model"] = model
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
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
    p.add_argument("--baseline", required=True, help="result JSON containing the failed records")
    p.add_argument("--new-prompt", required=True)
    p.add_argument("--extract-base-url", default="http://localhost:8080/v1")
    p.add_argument("--extract-model", default="none")
    p.add_argument("--question-ids", default=None,
                   help="comma-sep qids; if omitted, picks records where new_correct=0 AND retrieval-shape failure")
    p.add_argument("--mode", default="retrieval_miss",
                   choices=["retrieval_miss", "all_failures"],
                   help="auto-pick failures to test")
    p.add_argument("--longmemeval-data",
                   default=str(Path.home() / "heurchain-bench" / "bench_data"))
    p.add_argument("--output", required=True)
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).expanduser().read_text())
    records_by_qid = {r["question_id"]: r for r in baseline["records"]}

    if args.question_ids:
        qids = [q.strip() for q in args.question_ids.split(",") if q.strip()]
    else:
        # Heuristic: of records where new_correct=0, pick those whose response is
        # "I don't know" (suggests no useful facts, likely retrieval miss).
        qids = []
        for r in baseline["records"]:
            new_correct = r.get("new_correct", r.get("correct", 0))
            if new_correct:
                continue
            resp_low = (r.get("new_response") or r.get("model_response") or "").lower()
            if args.mode == "all_failures" or "i don" in resp_low:
                qids.append(r["question_id"])
    if not qids:
        print("No failures to re-test."); sys.exit(0)

    prompt_tmpl = load_prompt_template(args.new_prompt)
    prompt_name = Path(args.new_prompt).stem

    sys.path.insert(0, args.longmemeval_data)
    from load_longmemeval import load_split, iter_retrieval_tasks, session_to_text  # noqa
    instances = load_split(baseline.get("split", "s"))
    tasks_by_qid = {t["question_id"]: t for t in iter_retrieval_tasks(instances)}
    session_dates = {}
    for inst in instances:
        for sid, sdate in zip(inst.get("haystack_session_ids", []),
                              inst.get("haystack_dates", [])):
            session_dates[sid] = sdate

    print(f"Haystack-wide v4 re-extraction: {len(qids)} questions")
    print(f"  Prompt:      {prompt_name}")
    print(f"  Extract via: {args.extract_base_url}")
    sys.stdout.flush()

    out_records = []
    t_start = time.time()
    for qi, qid in enumerate(qids):
        task = tasks_by_qid.get(qid)
        if not task:
            print(f"  WARN {qid} not in dataset"); continue
        baseline_rec = records_by_qid.get(qid, {})

        # Extract from ALL haystack sessions (not just relevant ones)
        all_sessions = task["sessions"]
        n_sessions = len(all_sessions)
        all_facts = []
        per_session_meta = []
        for si, sess in enumerate(all_sessions):
            text = session_to_text(sess["turns"])
            sdate = session_dates.get(sess["session_id"])
            try:
                facts = extract(text, prompt_tmpl, args.extract_base_url,
                                args.extract_model, session_date=sdate)
            except Exception as e:
                facts = []
                print(f"  WARN extract failed {qid}/{sess['session_id']}: {e}")
            for f in facts:
                # Tag with the original session_id so the event index can group
                all_facts.append(f"[SessionID {sess['session_id']}] {f}")
            per_session_meta.append({
                "session_id": sess["session_id"],
                "session_date": sdate,
                "is_relevant": sess["session_id"] in task["relevant_session_ids"],
                "n_facts": len(facts),
            })
            if (si + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"    [{qi+1}/{len(qids)}] {qid}: {si+1}/{n_sessions} sessions, "
                      f"{len(all_facts)} facts total  ({elapsed:.0f}s elapsed)", flush=True)

        contains, score = fact_contains_gold_score(task["answer"], all_facts)
        out_records.append({
            "question_id": qid,
            "question_type": task["question_type"],
            "question": task["query"],
            "gold_answer": str(task["answer"]),
            "n_haystack_sessions": n_sessions,
            "n_relevant_sessions": len(task["relevant_session_ids"]),
            "n_facts_total": len(all_facts),
            "haystack_facts": all_facts,
            "per_session_meta": per_session_meta,
            "fact_contains_gold": contains,
            "fact_gold_overlap_score": score,
            "baseline_correct": int(baseline_rec.get("new_correct", baseline_rec.get("correct", 0))),
        })
        elapsed = time.time() - t_start
        print(f"  [{qi+1}/{len(qids)}] done {qid}: {len(all_facts)} facts from "
              f"{n_sessions} sessions, contains_gold={contains}, score={score}  "
              f"({elapsed:.0f}s elapsed)", flush=True)

    out = {
        "baseline_path": str(args.baseline),
        "prompt_path": str(args.new_prompt),
        "prompt_name": prompt_name,
        "split": baseline.get("split", "s"),
        "n_records": len(out_records),
        "mode": args.mode,
        "wall_time_s": round(time.time() - t_start, 1),
        "records": out_records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    contains_n = sum(1 for r in out_records if r["fact_contains_gold"])
    print()
    print(f"=== haystack v4 re-extraction (n={len(out_records)}) ===")
    print(f"Facts now contain gold:  {contains_n}/{len(out_records)} "
          f"({contains_n/max(1,len(out_records))*100:.1f}%)")
    print(f"Wall time:                {(time.time() - t_start)/60:.1f} min")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
