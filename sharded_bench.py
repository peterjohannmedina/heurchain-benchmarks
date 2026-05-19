#!/usr/bin/env python3
"""
Mode 2: Sharded LongMemEval across N tenants on the local HeurChain broker.

Distributes the 500 LongMemEval-S tasks across N tenants (e.g., 50/tenant × 10),
ingests sessions via the broker REST API (client-side bge-m3 embeddings),
then runs retrieval queries CONCURRENTLY across tenants and computes per-tenant
Recall@k + MRR + NDCG.

Pass criterion: per-tenant metrics must match the single-tenant in-process
baseline (R@10 = 0.972, MRR = 0.911 on 500 tasks). Any per-tenant degradation
under concurrent load is an isolation or contention bug.
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from bench_data.load_longmemeval import load_split, iter_retrieval_tasks, session_to_text  # noqa

DEFAULT_URL = "http://127.0.0.1:3012"
EMBED_MODEL = "BAAI/bge-m3"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def tenant_id(i: int) -> str:
    return f"t{i:02d}"


def headers_for(tid: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-HeurChain-API-Key": f"testkey-{tid}",
        "X-HeurChain-Tenant": tid,
    }


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def mrr(retrieved, relevant):
    rel = set(relevant)
    for i, d in enumerate(retrieved, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    rel = set(relevant)
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# Broker hits look like one of two shapes depending on broker version:
#   {"results": [{"id":..., "score":..., "text":...}, ...]}
# or {"matches": [...]}
def extract_ids(resp_json):
    hits = resp_json.get("results") or resp_json.get("matches") or []
    return [h.get("id") for h in hits if h.get("id")]


async def ingest_tenant(client, base_url, tid, tasks_for_tenant, embedder, semaphore):
    """Sequentially ingest all sessions for this tenant's tasks. Returns mapping
    {question_id: list_of_stored_session_ids_for_relevance_matching}."""
    relevance_map = {}
    for task in tasks_for_tenant:
        qid = task["question_id"]
        # Doc IDs need to be per-tenant unique AND mappable back to the original
        # session_id for relevance scoring. Prefix with question_id to namespace
        # across tasks within a tenant.
        stored_ids = []
        relevant_doc_ids = []
        sess_texts = []
        sess_doc_ids = []
        for sess in task["sessions"]:
            text = session_to_text(sess["turns"])
            if not text.strip():
                continue
            sid = sess["session_id"]
            doc_id = f"{qid}-{sid}"
            sess_texts.append(text)
            sess_doc_ids.append(doc_id)
            stored_ids.append(doc_id)
            if sid in task["relevant_session_ids"]:
                relevant_doc_ids.append(doc_id)

        if not sess_texts:
            continue

        # Batch embed all sessions for this task at once (32 at a time on GPU)
        embs = embedder.encode(sess_texts, normalize_embeddings=True, batch_size=32)

        for doc_id, text, emb in zip(sess_doc_ids, sess_texts, embs):
            async with semaphore:
                try:
                    await client.post(f"{base_url}/store",
                                      headers=headers_for(tid),
                                      json={"id": doc_id, "text": text,
                                            "agent_id": qid,  # sub-tenant isolation per LongMemEval task
                                            "embedding": emb.astype("float32").tolist()},
                                      timeout=15)
                except Exception as e:
                    print(f"  store err tid={tid} doc={doc_id[:30]}: {str(e)[:60]}")

        relevance_map[qid] = relevant_doc_ids
    return relevance_map


async def query_tenant(client, base_url, tid, tasks_for_tenant, relevance_map, embedder):
    """Run queries for this tenant's tasks. Returns list of per-task metrics."""
    rows = []
    for task in tasks_for_tenant:
        qid = task["question_id"]
        rel = relevance_map.get(qid, [])
        if not rel:
            continue
        qvec = embedder.encode([task["query"]], normalize_embeddings=True)[0]
        try:
            r = await client.post(f"{base_url}/query",
                                  headers=headers_for(tid),
                                  json={"embedding": qvec.astype("float32").tolist(),
                                        "agent_id": qid,  # restrict to this task's docs
                                        "top_k": 10},
                                  timeout=15)
            r.raise_for_status()
            retrieved = extract_ids(r.json())
        except Exception as e:
            rows.append({"qid": qid, "qtype": task["question_type"], "err": str(e)[:80]})
            continue

        rows.append({
            "qid": qid,
            "qtype": task["question_type"],
            "tenant": tid,
            "n_retrieved": len(retrieved),
            "n_relevant": len(rel),
            "recall@1": recall_at_k(retrieved, rel, 1),
            "recall@5": recall_at_k(retrieved, rel, 5),
            "recall@10": recall_at_k(retrieved, rel, 10),
            "mrr": mrr(retrieved, rel),
            "ndcg@10": ndcg_at_k(retrieved, rel, 10),
        })
    return rows


def aggregate(rows):
    if not rows:
        return {}
    keys = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    return {k: round(sum(r[k] for r in rows if k in r) / max(1, sum(1 for r in rows if k in r)), 4) for k in keys}


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--n-tenants", type=int, default=10)
    p.add_argument("--max-tasks", type=int, default=100,
                   help="Total tasks across all tenants. Default 100 (10/tenant).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="s")
    p.add_argument("--store-concurrency", type=int, default=8,
                   help="Max concurrent /store calls per tenant during ingest")
    args = p.parse_args()

    # Health check
    async with httpx.AsyncClient() as c:
        h = await c.get(f"{args.url}/health", timeout=5)
        h.raise_for_status()
        print(f"Broker health: {h.json()}")

    print(f"Loading {EMBED_MODEL} on {args.device}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=args.device)

    print(f"Loading LongMemEval-{args.split}...")
    instances = load_split(args.split)
    tasks = iter_retrieval_tasks(instances)[:args.max_tasks]
    print(f"Using {len(tasks)} tasks across {args.n_tenants} tenants "
          f"= {len(tasks) // args.n_tenants} tasks/tenant\n")

    # Distribute tasks round-robin to tenants
    shards = {tenant_id(i + 1): [] for i in range(args.n_tenants)}
    for i, task in enumerate(tasks):
        tid = tenant_id((i % args.n_tenants) + 1)
        shards[tid].append(task)

    # Phase 1: CONCURRENT ingest per tenant (each tenant ingests its own shard
    # in parallel — this is the key multi-tenant stress test for store path)
    print("=== Phase 1: concurrent ingest across all tenants ===")
    t0 = time.perf_counter()
    relevance_maps = {}
    sem_per_tenant = {tid: asyncio.Semaphore(args.store_concurrency) for tid in shards}
    async with httpx.AsyncClient(timeout=30) as client:
        async def tenant_job(tid):
            relevance_maps[tid] = await ingest_tenant(
                client, args.url, tid, shards[tid], embedder, sem_per_tenant[tid])
            print(f"  {tid}: ingested {sum(len(v) for v in relevance_maps[tid].values())} relevant docs "
                  f"({len(shards[tid])} tasks)")

        await asyncio.gather(*[tenant_job(tid) for tid in shards])
    ingest_s = time.perf_counter() - t0
    print(f"Phase 1 done in {ingest_s:.1f}s")

    # Phase 2: CONCURRENT query — every tenant queries simultaneously
    print("\n=== Phase 2: concurrent query across all tenants ===")
    t0 = time.perf_counter()
    all_rows_by_tenant = {}
    async with httpx.AsyncClient(timeout=15) as client:
        async def query_job(tid):
            all_rows_by_tenant[tid] = await query_tenant(
                client, args.url, tid, shards[tid], relevance_maps[tid], embedder)

        await asyncio.gather(*[query_job(tid) for tid in shards])
    query_s = time.perf_counter() - t0
    print(f"Phase 2 done in {query_s:.1f}s")

    # Aggregate per-tenant and overall
    print("\n=== Per-tenant metrics ===")
    print(f"{'tenant':>8} {'tasks':>6} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'NDCG@10':>8}")
    by_tenant_summary = {}
    all_rows = []
    for tid in sorted(shards):
        rows = [r for r in all_rows_by_tenant[tid] if "recall@10" in r]
        agg = aggregate(rows)
        by_tenant_summary[tid] = {"n_tasks": len(rows), **agg}
        all_rows.extend(rows)
        if rows:
            print(f"{tid:>8} {len(rows):>6} {agg['recall@1']:>7.4f} {agg['recall@5']:>7.4f} "
                  f"{agg['recall@10']:>7.4f} {agg['mrr']:>7.4f} {agg['ndcg@10']:>8.4f}")

    overall = aggregate(all_rows)
    print(f"{'OVERALL':>8} {len(all_rows):>6} {overall['recall@1']:>7.4f} {overall['recall@5']:>7.4f} "
          f"{overall['recall@10']:>7.4f} {overall['mrr']:>7.4f} {overall['ndcg@10']:>8.4f}")
    print()
    print("Baseline reference (in-process Dense, single tenant, 500 tasks):")
    print(f"  R@10 = 0.9722, MRR = 0.9112")

    out = {
        "mode": "sharded",
        "n_tenants": args.n_tenants,
        "n_tasks_total": len(tasks),
        "ingest_time_s": round(ingest_s, 1),
        "query_time_s": round(query_s, 1),
        "by_tenant": by_tenant_summary,
        "overall": overall,
        "baseline_reference": {
            "system": "dense-faiss in-process",
            "split": args.split,
            "n_tasks": 500,
            "recall@10": 0.9722,
            "mrr": 0.9112,
        },
    }
    path = RESULTS_DIR / "multitenant_sharded.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    asyncio.run(amain())
