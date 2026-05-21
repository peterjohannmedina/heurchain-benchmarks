#!/usr/bin/env python3
"""
Triage failed questions from a cross-judge results JSON.

Reads a results JSON produced by judge_rerun_cloud.py (with fact_contains_gold
diagnostic fields), classifies each failed question into one of these failure
modes, and ranks them by how confidently fixable they are:

  EXTRACTION   — Hit@10 high but fact_contains_gold=False
                 (retrieval found the right session; extraction lost the answer
                  — this is the high-confidence "fix the prompt" pile)
  ANSWERER     — Hit@10 high AND fact_contains_gold=True but correct=0
                 (facts contain the answer; the model failed to synthesize)
  RETRIEVAL    — Hit@10 = False
                 (wrong session retrieved; extraction can't help)
  UNANSWERABLE — Hit@10 high, fact_contains_gold=False, gold itself is short
                 (rare/edge case)

Outputs a markdown report with per-mode counts + the top-N questions per mode.

Usage:
  python3 triage_failures.py --input results/facts_v3cloud-deepseek_multi-session_max30.json
  python3 triage_failures.py --input <file> --mode EXTRACTION --top 10
  python3 triage_failures.py --inputs 'results/facts_v3cloud-*.json' --mode EXTRACTION
"""
import argparse
import glob
import json
from pathlib import Path


def classify(record):
    """Return failure mode string, or None if record is correct."""
    if record["correct"]:
        return None
    hit = record.get("retrieval_hit", 0)
    contains = record.get("fact_contains_gold", None)
    if not hit:
        return "RETRIEVAL"
    if contains is False:
        return "EXTRACTION"
    if contains is True:
        return "ANSWERER"
    # contains is None (older records without the diagnostic) — best guess
    return "UNCLASSIFIED"


def triage_one(path):
    d = json.loads(Path(path).expanduser().read_text())
    cat = d.get("question_type", "unknown")
    records = d["records"]
    by_mode = {"EXTRACTION": [], "ANSWERER": [], "RETRIEVAL": [], "UNCLASSIFIED": []}
    correct_count = 0
    for r in records:
        mode = classify(r)
        if mode is None:
            correct_count += 1
        else:
            by_mode[mode].append(r)
    return {
        "path": str(path),
        "category": cat,
        "n_records": len(records),
        "n_correct": correct_count,
        "by_mode": by_mode,
        "judge": (d.get("cloud_judge") or {}).get("judge_model")
                 or (d.get("cloud_judge") or {}).get("model")
                 or "local-14B",
    }


def emit_one(triage, top_n=5, mode_filter=None):
    print(f"## {triage['category']}  (judge: `{triage['judge']}`)")
    print()
    print(f"Total records: {triage['n_records']}, correct: {triage['n_correct']} "
          f"({triage['n_correct']/triage['n_records']*100:.1f}%)")
    print()
    print("### Failure breakdown")
    print()
    print("| Mode | Count | Share of failures | Actionable signal |")
    print("|---|---:|---:|---|")
    total_failures = sum(len(v) for v in triage["by_mode"].values())
    if total_failures == 0:
        print("| (all correct) | 0 | — | — |")
        print(); return
    actions = {
        "EXTRACTION": "fix the extraction prompt — answer is in the session but missing from facts",
        "ANSWERER":   "facts have the answer; answerer model isn't synthesizing (try larger / different model)",
        "RETRIEVAL":  "wrong session in top-k — bm25/dense weights, chunk size, or filtering issue",
        "UNCLASSIFIED": "diagnostic missing (re-run with newer harness)",
    }
    for mode in ["EXTRACTION", "ANSWERER", "RETRIEVAL", "UNCLASSIFIED"]:
        items = triage["by_mode"][mode]
        n = len(items)
        share = n / total_failures * 100 if total_failures else 0
        print(f"| {mode} | {n} | {share:.1f}% | {actions[mode]} |")
    print()

    # Top-N per mode (or just one mode if filtered)
    modes_to_show = [mode_filter] if mode_filter else ["EXTRACTION", "ANSWERER", "RETRIEVAL"]
    for mode in modes_to_show:
        items = triage["by_mode"].get(mode, [])
        if not items:
            continue
        print(f"### Top {min(top_n, len(items))} `{mode}` failures")
        print()
        # Rank: EXTRACTION → high overlap_score is "barely missed" (closest to fixable)
        if mode == "EXTRACTION":
            items.sort(key=lambda r: -r.get("fact_gold_overlap_score", 0))
        for i, r in enumerate(items[:top_n], 1):
            qid = r.get("question_id", "?")
            q = r.get("question", "")[:140]
            gold = str(r.get("gold_answer", ""))[:120]
            resp = str(r.get("model_response", ""))[:120]
            score = r.get("fact_gold_overlap_score", "n/a")
            print(f"**[{i}] {qid}** (overlap={score})")
            print(f"- Q: {q}")
            print(f"- Gold: {gold}")
            print(f"- Response: {resp!r}")
            print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="single results JSON")
    p.add_argument("--inputs", help="glob of results JSONs")
    p.add_argument("--mode", choices=["EXTRACTION", "ANSWERER", "RETRIEVAL"],
                   help="show only this mode in the top-N listing")
    p.add_argument("--top", type=int, default=5, help="top-N failures per mode")
    args = p.parse_args()

    paths = []
    if args.input:
        paths.append(args.input)
    if args.inputs:
        paths.extend(sorted(glob.glob(args.inputs)))
    if not paths:
        p.error("provide --input or --inputs")

    print("# Failure triage report\n")
    print(f"Loaded {len(paths)} result file(s).\n")
    print("---\n")

    for path in paths:
        try:
            t = triage_one(path)
            emit_one(t, top_n=args.top, mode_filter=args.mode)
            print("---\n")
        except Exception as e:
            print(f"  WARN: could not triage {path}: {e}\n")


if __name__ == "__main__":
    main()
