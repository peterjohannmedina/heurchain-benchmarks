# heurchain-benchmarks

The benchmark harness behind the numbers on **[heurchain.com](https://heurchain.com)** and the [HeurChain vs Mem0](https://heurchain.com/vs-mem0), [vs Zep](https://heurchain.com/vs-zep), [vs Letta](https://heurchain.com/vs-letta), [vs Cognee](https://heurchain.com/vs-cognee) comparison pages.

Published so you can reproduce them, dispute them, or run your own on top of them.

> **Main project:** [peterjohannmedina/heurchain](https://github.com/peterjohannmedina/heurchain) — HeurChain is a persistent vector memory broker for AI agents (MIT licensed, multi-tenant, MCP-native).

---

## What this measures

These scripts evaluate HeurChain on the **[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)** dataset (ICLR 2025): 500 questions across 6 reasoning categories, each backed by a "haystack" of multi-session conversations. The task is, given a question, to retrieve the sessions that actually contain the answer.

### Headline numbers (measured 2026-05, hardware: 1 vCPU container, M-class CPU)

| Metric | Value | Notes |
|---|---|---|
| **R@10** (hybrid α=0.9) | **0.978** | dense bge-m3 + BM25 with Reciprocal Rank Fusion |
| **R@10** (dense only) | 0.972 | bge-m3 dense retrieval |
| **MRR** (hybrid α=0.8) | **0.913** | asymmetric RRF outperforms symmetric (+1.5pp vs α=0.5) |
| **NDCG@10** (hybrid α=0.9) | 0.914 | |
| **P95 latency** (in-process) | 35 ms | algorithm-only; no network |
| **P95 latency** (multi-tenant Docker, 10 tenants concurrent) | **20.5 ms** | closest to production SaaS topology |
| **P95 latency** (BM25 only) | 4.6 ms | keyword-only path |
| **Multi-tenant isolation** | 0 leaks / 90 probe queries | per-tenant namespace + agent_id sub-isolation |

These are reproducible from this harness against the broker in the [main HeurChain repo](https://github.com/peterjohannmedina/heurchain).

### Full per-task results

The complete v2 fact-extraction run across all 6 LongMemEval-S categories — including per-task records, judge verdicts, and a mean QA accuracy of 32.8% (local 14B judge) / 87.8% retrieval Hit@10 — is published in **[`results/`](results/)**. That directory's [README](results/README.md) walks through the headline numbers, honest comparison to Mem0's 49.4%, and what the gap between Hit@10 and QA accuracy actually means.

---

## Honest bias disclosure

This harness was written by the HeurChain team. **Of course it favors what we built well.**

What that does and doesn't mean:

- The numbers above are real — measured against the actual code in the [main repo](https://github.com/peterjohannmedina/heurchain).
- The methodology is standard for retrieval evaluation: Recall@k, MRR, NDCG, p50/p95 latency on a public dataset (LongMemEval-S).
- But: we chose retrieval R@k as our headline metric, and other systems (Mem0, Zep) headline different metrics (end-to-end QA accuracy with an LLM judge). Different metric families are not directly comparable, and we don't claim to win on metrics we haven't published.
- If you're evaluating multiple systems, **run the harness on your own data.** Public benchmarks correlate with real workloads but they're not the same thing.
- If you can't reproduce a number here on your hardware, [open an issue](https://github.com/peterjohannmedina/heurchain-benchmarks/issues) — we'll fix or correct it.

---

## Scripts

### `sharded_bench.py` — distributed correctness invariance

Sharded LongMemEval across N tenants on the local broker. Verifies that **per-tenant retrieval metrics under concurrent load match the single-tenant in-process baseline.** Any degradation = isolation or contention bug.

```bash
python3 sharded_bench.py --url http://127.0.0.1:3012 --n-tenants 10 --max-tasks 500
```

Pass criterion: per-tenant R@10 ≈ 0.972, MRR ≈ 0.911 — same as the single-tenant baseline.

### `multitenant_bench.py` — concurrent throughput + canary correctness

Three modes:

```bash
# Cross-tenant isolation canary (correctness — does tenant A ever see tenant B's data?)
python3 multitenant_bench.py --mode canary --n-tenants 10

# Sharded LongMemEval (same as sharded_bench but lighter setup)
python3 multitenant_bench.py --mode sharded --n-tenants 10

# Concurrent load — sweeps N tenants to find throughput ceiling
python3 multitenant_bench.py --mode load --max-tenants 50 --qps-per-tenant 5
```

The published 20.5 ms p95 multi-tenant latency comes from `--mode load --max-tenants 10 --qps-per-tenant 5` against the broker's Docker compose stack.

---

## Setup — fastest path

### 1. Run a local HeurChain broker

The harness needs a broker listening on `http://127.0.0.1:3012` with the `HEURCHAIN_TEST_MODE=1` environment variable set (this enables the deterministic `testkey-{tid}` API keys the harness uses).

```bash
git clone https://github.com/peterjohannmedina/heurchain.git
cd heurchain
# Follow the README there to start the broker via Docker compose
# Make sure HEURCHAIN_TEST_MODE=1 is set
```

### 2. Install Python deps

```bash
git clone https://github.com/peterjohannmedina/heurchain-benchmarks.git
cd heurchain-benchmarks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Get the LongMemEval-S dataset

Clone the official dataset repo into `bench_data/LongMemEval/`:

```bash
git clone https://github.com/xiaowu0162/LongMemEval.git bench_data/LongMemEval
```

The loader expects `bench_data/LongMemEval/data/longmemeval_s_cleaned.json` to exist (the cleaned S-split is the default for our reported numbers). Other splits (`oracle`, `m`) can be selected via `--split`.

### 4. Run

```bash
python3 sharded_bench.py --n-tenants 10
```

Results land in `results/sharded_<timestamp>.json` with per-tenant metrics.

---

## Reproducibility expectations

- **The harness is deterministic at the embedding + retrieval layer.** Same dataset + same broker version + same embedding model (`BAAI/bge-m3`) = the same R@k / MRR / NDCG numbers, modulo floating-point.
- **Latency numbers will vary by hardware.** Our 20.5 ms p95 came from a 4-vCPU Docker host with HeurChain's broker + Redis + SQLite in-container. On a smaller box expect proportionally higher latency; on a bigger box, lower.
- **The first run is slow** because `sentence-transformers` downloads the bge-m3 model (~2.3 GB). Subsequent runs use the cached model.
- **If your retrieval numbers diverge materially from ours**, that's interesting and we want to know — likely a broker config or embedding model mismatch. File an issue with your config + results JSON.

---

## When NOT to use this harness

- **Comparing HeurChain to Mem0 / Zep / Letta on their own published metrics.** Their papers use task-completion QA accuracy with an LLM judge, not retrieval R@k. Mixing those metric families is misleading. We'll publish an LLM-judge run on our end eventually; until then, treat cross-system comparisons as architectural, not quantitative.
- **Evaluating any system other than HeurChain.** This harness assumes a HeurChain broker on the other end of the HTTP calls; it doesn't have adapters for other memory layers. If you want a multi-system comparison, the [vectorize.io "AI Agent Memory Systems" article](https://vectorize.io/best-ai-agent-memory-systems-in-2025/) is a starting point — and we'd love to be in it.
- **Production load testing.** This is a correctness + latency benchmark, not a load generator. It won't tell you how HeurChain behaves at 10k QPS.

---

## Methodology notes (for the skeptics)

- **Dataset:** LongMemEval-S "cleaned" split, 500 questions across 6 reasoning categories (single-session preference, single-session reasoning, multi-session, knowledge update, temporal reasoning, abstention).
- **Embedding model:** `BAAI/bge-m3` (1024-dim, multilingual, free, BSD-licensed). Same model used in our retrieval index.
- **Hybrid retrieval:** BM25 + dense cosine, fused via Reciprocal Rank Fusion (RRF) with tunable α. Default α=0.9 (dense-weighted); α=0.8 is optimal on LongMemEval-S MRR (+1.5 pp vs symmetric α=0.5 — the default in some peer systems). See [the blog post on asymmetric RRF](https://heurchain.com/blog/the-512-token-chunk-pattern) for the methodology.
- **Relevance:** ground-truth `answer_session_ids` from the dataset. A retrieved session is "relevant" iff its session ID is in this set.
- **What we DO NOT measure here:** end-to-end QA accuracy (no LLM in the loop), generation quality, multi-hop reasoning over retrieved context. Those are downstream of retrieval, and are interesting but different metrics.

---

## Citation

If you reference these numbers in a paper or blog post, please link:

- This repo for reproduction: `https://github.com/peterjohannmedina/heurchain-benchmarks`
- The main HeurChain project: `https://github.com/peterjohannmedina/heurchain`
- The LongMemEval-S dataset: Wu et al., ICLR 2025, [github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)

---

## License

[MIT](LICENSE).
