#!/usr/bin/env python3
"""
Compute per-question inter-judge agreement across multiple result JSONs.

Why: a single LLM judge produces a single QA accuracy number. Two judges from
independent model families produce a per-question AGREEMENT rate, which is a
much stronger defensibility signal than any individual judge's QA number. If
two frontier judges agree on 87.8% of per-question verdicts (as ours did in
the May 2026 cross-judge run), the published QA number is defensible.

Usage:
  # Across 2 or more results for the same category:
  python3 compute_agreement.py \
    --category single-session-assistant \
    --runs results/facts_v2_*_max30.json results/facts_v2cloud-deepseek_*_max30.json \
                results/facts_v2cloud-kimi_*_max30.json

  # Or compare all 6 categories across all runs:
  python3 compute_agreement.py --all-categories \
    --runs results/facts_v2_*.json results/facts_v2cloud-*.json

Outputs a markdown table to stdout with QA accuracy per run and pairwise
per-question verdict agreement.
"""
import argparse
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


CATEGORIES = [
    "single-session-assistant",
    "temporal-reasoning",
    "single-session-user",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
]


def load_run(path):
    """Return {(question_id): correct_bool} from one result JSON, plus label."""
    p = Path(path).expanduser()
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    cat = d.get("question_type", "?")
    # Label: prefer cloud_judge.answer/judge model; fall back to filename
    cj = d.get("cloud_judge") or {}
    if cj.get("answer_model") and cj.get("judge_model"):
        am = cj["answer_model"]
        jm = cj["judge_model"]
        label = f"{am}→{jm}" if am != jm else am
    elif cj.get("model"):
        label = cj["model"]
    else:
        # Local-only run (no cloud_judge meta) — use stable label so cross-category aggregation works
        label = "local-14B"
    verdicts = {r["question_id"]: bool(r["correct"]) for r in d["records"]}
    return label, verdicts, cat, d.get("overall_qa_acc", 0)


def pairwise_agreement(a, b):
    """Fraction of shared question_ids where a and b's verdicts match."""
    common = set(a) & set(b)
    if not common:
        return 0.0, 0
    matches = sum(1 for qid in common if a[qid] == b[qid])
    return matches / len(common), len(common)


def categorize_runs(paths):
    """Group runs by category: {category: [(label, verdicts, qa), ...]}"""
    by_cat = defaultdict(list)
    for p in paths:
        loaded = load_run(p)
        if loaded[0] is None:
            print(f"  WARN: could not load {p}", file=sys.stderr)
            continue
        label, verdicts, cat, qa = loaded
        by_cat[cat].append((label, verdicts, qa))
    return by_cat


def emit_for_category(cat, runs):
    print(f"\n## Category: {cat}\n")
    if len(runs) < 2:
        print(f"_(need ≥2 runs for agreement, got {len(runs)})_\n")
        return
    # Per-run QA accuracy
    print("### Per-judge QA accuracy\n")
    print("| Judge config | QA accuracy | n |")
    print("|---|---:|---:|")
    for label, verdicts, qa in runs:
        n = len(verdicts)
        print(f"| `{label}` | {qa*100:.2f}% | {n} |")
    # Pairwise agreement
    print("\n### Pairwise per-question agreement\n")
    print("| Judge A | Judge B | Agreement | n_common |")
    print("|---|---|---:|---:|")
    for (la, va, _), (lb, vb, _) in itertools.combinations(runs, 2):
        agree, n_common = pairwise_agreement(va, vb)
        print(f"| `{la}` | `{lb}` | {agree*100:.2f}% | {n_common} |")
    # All-N agreement (if 3+ runs)
    if len(runs) >= 3:
        labels = [r[0] for r in runs]
        all_verdicts = [r[1] for r in runs]
        common = set.intersection(*(set(v) for v in all_verdicts))
        if common:
            all_agree = sum(1 for qid in common
                            if len({v[qid] for v in all_verdicts}) == 1)
            print(f"\n**All {len(runs)} judges agree on {all_agree}/{len(common)} ({all_agree/len(common)*100:.2f}%) of common questions.**")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="result JSON files (globs OK). Can mix categories — they'll be grouped.")
    p.add_argument("--category", default=None,
                   help="restrict to one category (e.g. single-session-assistant)")
    p.add_argument("--all-categories", action="store_true",
                   help="emit a section per category found across the runs")
    args = p.parse_args()

    # Expand globs
    paths = []
    for pat in args.runs:
        matched = sorted(glob.glob(str(Path(pat).expanduser())))
        if matched:
            paths.extend(matched)
        else:
            paths.append(pat)

    by_cat = categorize_runs(paths)
    if not by_cat:
        print("No runs loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"# Inter-judge agreement report\n")
    print(f"Loaded {sum(len(v) for v in by_cat.values())} run(s) across {len(by_cat)} categor(ies).")

    if args.category:
        if args.category not in by_cat:
            print(f"\nNo runs found for category '{args.category}'.", file=sys.stderr)
            sys.exit(1)
        emit_for_category(args.category, by_cat[args.category])
    elif args.all_categories or len(by_cat) == 1:
        for cat in CATEGORIES:
            if cat in by_cat:
                emit_for_category(cat, by_cat[cat])
        # Mean across categories
        if len(by_cat) > 1:
            print("\n## Cross-category summary\n")
            # Aggregate verdicts across categories per label
            label_to_qa = defaultdict(list)
            for cat_runs in by_cat.values():
                for label, verdicts, qa in cat_runs:
                    label_to_qa[label].append(qa)
            print("| Judge config | Mean QA across categories | n_categories |")
            print("|---|---:|---:|")
            for label, qas in label_to_qa.items():
                if qas:
                    print(f"| `{label}` | {sum(qas)/len(qas)*100:.2f}% | {len(qas)} |")
    else:
        # Default: just the first category
        first_cat = next(iter(by_cat))
        emit_for_category(first_cat, by_cat[first_cat])
        print(f"\n_(use --all-categories to see all {len(by_cat)} categories.)_")


if __name__ == "__main__":
    main()
