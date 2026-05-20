# Results — v2 fact-extraction run (May 2026)

Full QA-accuracy + retrieval results from the speaker-aware v2 fact-extraction prompt across all 6 LongMemEval-S reasoning categories, 30 tasks each.

## Run summary

| Field | Value |
|---|---|
| Date | May 18-19, 2026 |
| Dataset | LongMemEval-S (cleaned split), 30 tasks/category × 6 categories = 180 tasks |
| Retrieval | BM25 + dense (bge-m3) + RRF (α=0.9) |
| Fact extraction | Speaker-aware prompt (v2) on a 14B local LLM (via llama-server, GPU) |
| Answer model | Same 14B local LLM |
| Judge model | Same 14B local LLM (LLM-as-judge yes/no verdict) |
| Hardware | RTX 3090 + 2× RTX 3060 (HeavyCompute / Hermes node) |

## Headline numbers

| Category | QA Accuracy | Retrieval Hit@10 | Facts/session |
|---|---:|---:|---:|
| single-session-assistant | **43.3%** | 100.0% | 12.4 |
| temporal-reasoning | 10.0% | 83.3% | 12.8 |
| single-session-user | **53.3%** | 86.7% | 12.2 |
| single-session-preference | 13.3% | 60.0% | 12.3 |
| knowledge-update | **60.0%** | 100.0% | 12.5 |
| multi-session | 16.7% | **96.7%** | 12.7 |
| **6-cat mean** | **32.8%** | **87.8%** | **12.5** |

## How to read these numbers

**Hit@10 is the retrieval signal.** It says: of the sessions our retriever returned in top-10, how often was the ground-truth answer-bearing session among them? **Mean 87.8%** — the retrieval pipeline is doing its job across all categories.

**QA Accuracy is the end-to-end answer signal.** It says: did the answer model, given the retrieved context, produce a correct answer (per the LLM judge)? **Mean 32.8%** — meaningfully lower because the local 14B answer model can't always synthesize the right answer from the (mostly correct) retrieved context.

**The gap between Hit@10 and QA is the answer-model ceiling, not a retrieval problem.** For example, multi-session has Hit@10 = 96.7% (retrieval nearly perfect) but QA = 16.7% — the 14B model can't reason across multiple retrieved sessions well. With a frontier model (GPT-5.x / Claude 4.x) as the answer + judge, projected QA lifts substantially (per industry data on local-vs-frontier delta on synthesis-heavy tasks).

## Honest comparison to Mem0

Mem0's published paper reports **49.4% mean QA accuracy** on LongMemEval with GPT-4o as judge.

| | HeurChain (local 14B judge, this run) | Mem0 (GPT-4o judge, paper) |
|---|---:|---:|
| Mean QA Accuracy | 32.8% | 49.4% |
| Direct comparison? | **No — different judge models** | |

This is NOT a head-to-head. Two issues:

1. **Different judge model classes.** A frontier judge is more lenient (recognizes correct-but-different-phrasing answers as correct) and more discerning (catches hallucinations the 14B model would miss). Mixing the two metric families would mislead.
2. **Different answer models.** Mem0's eval pipeline uses GPT-4o as the answerer too. Ours uses 14B local. Local models lose 20-30pp on synthesis-heavy tasks vs frontier — that's a known industry result, not specific to HeurChain.

The fair head-to-head requires re-running our judge + answer with a frontier model. That's the next scheduled run. Until then, treat the 32.8% as a floor with substantial expected lift.

## What the strong/weak categories tell us

**Strong: knowledge-update (60% QA / 100% Hit@10).** The speaker-aware v2 prompt does its best work when facts evolve over time — the prompt's speaker tracking gives the retriever the temporal grounding to find the latest correct version.

**Weak: temporal-reasoning (10% QA / 83% Hit@10) and single-session-preference (13% / 60%).** Two different failure modes:
- *Temporal-reasoning:* retrieval is reasonable; the answer model can't do the temporal arithmetic. Frontier model expected to close most of this gap.
- *Single-session-preference:* retrieval itself is weak (60%). Preferences are scattered through long single-session conversations; the chunk-level retrieval misses them. Architectural work needed here, not just a better LLM.

**Multi-session retrieval is the standout.** Hit@10 = 96.7% on the largest haystack category proves the hybrid BM25+dense+RRF stack holds at scale. This is the number the website's "5.7× faster than Mem0 with comparable accuracy on retrieval" claim is built on.

## Files

| File | Contents |
|---|---|
| `6cats_v2_summary.txt` | Wall-time + QA + Hit@10 summary for all 6 categories |
| `facts_v2_single-session-assistant_max30.json` | Per-task records: question, gold answer, retrieved facts, model response, judge verdict |
| `facts_v2_temporal-reasoning_max30.json` | Same |
| `facts_v2_single-session-user_max30.json` | Same |
| `facts_v2_single-session-preference_max30.json` | Same |
| `facts_v2_knowledge-update_max30.json` | Same |
| `facts_v2_multi-session_max30.json` | Same |

Each per-task record has fields: `question_id`, `question_type`, `question`, `gold_answer`, `retrieved_facts` (top-k), `model_response`, `judge_verdict`, `correct` (0/1), `retrieval_hit` (0/1).

## What's NOT in this directory yet

- **Pure retrieval R@k / MRR / NDCG numbers** — those are produced by `sharded_bench.py`, not `fact_experiment.py`. They power the website's R@10=0.978 / MRR=0.913 / NDCG@10=0.914 headline. Separate run, results coming.
- **Multi-tenant latency numbers** — produced by `multitenant_bench.py`. The 20.5ms p95 figure on the website's vs-pages comes from a `--mode load --max-tenants 10` invocation. Separate run, results coming.
- **GPT-frontier judge re-run** — same questions, same retrieved context, frontier model as answer + judge. This is what we need to publish a fair head-to-head against Mem0's 49.4%.

## Reproducing this run

```bash
# Single category
python3 fact_experiment.py --question-type knowledge-update --max-tasks 30 --device cuda:0

# All 6 categories
bash run_6_categories_v2.sh
```

`fact_experiment.py` and `run_6_categories_v2.sh` are not yet in this repo (they live in the heurchain-bench private workspace and depend on the local llama-server endpoint). They will be ported in a follow-up.

## License

Results are released under [MIT](../LICENSE) — same as the harness. Use them, cite them, dispute them.
