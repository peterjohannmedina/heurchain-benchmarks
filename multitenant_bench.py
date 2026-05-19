#!/usr/bin/env python3
"""
Multi-tenant benchmark for the local HeurChain broker.

Three modes:
  1. canary    — cross-tenant isolation test (correctness boolean)
  2. sharded   — LongMemEval distributed across N tenants (accuracy invariance)
  3. load      — concurrent QPS test, sweeps N tenants to find throughput ceiling

Connects to the Docker stack at http://127.0.0.1:3012 by default.
Uses bge-m3 client-side for embedding (no external embedding service required).

Usage:
  python multitenant_bench.py --mode canary --n-tenants 10
  python multitenant_bench.py --mode load --max-tenants 50 --qps-per-tenant 5
"""
import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_URL = "http://127.0.0.1:3012"
EMBED_MODEL = "BAAI/bge-m3"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def tenant_id(i: int) -> str:
    return f"t{i:02d}"


def api_key_for(tid: str) -> str:
    return f"testkey-{tid}"


def headers_for(tid: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-HeurChain-API-Key": api_key_for(tid),
        "X-HeurChain-Tenant": tid,
    }


# ---------------------------------------------------------------------------
# Embedding (client-side)
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, device: str = "cuda:0"):
        print(f"Loading {EMBED_MODEL} on {device}...")
        self.model = SentenceTransformer(EMBED_MODEL, device=device)

    def embed(self, text: str) -> list[float]:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return v.astype("float32").tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        v = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [row.astype("float32").tolist() for row in v]


# ---------------------------------------------------------------------------
# Broker API wrappers
# ---------------------------------------------------------------------------
async def broker_store(client: httpx.AsyncClient, base_url: str, tid: str,
                       text: str, embedding: list[float], doc_id: str = None) -> dict:
    payload = {"text": text, "embedding": embedding}
    if doc_id:
        payload["id"] = doc_id
    r = await client.post(f"{base_url}/store", headers=headers_for(tid), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


async def broker_query(client: httpx.AsyncClient, base_url: str, tid: str,
                       embedding: list[float], top_k: int = 10) -> dict:
    payload = {"embedding": embedding, "top_k": top_k}
    r = await client.post(f"{base_url}/query", headers=headers_for(tid), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Mode 1: Canary isolation
# ---------------------------------------------------------------------------
async def mode_canary(base_url: str, n_tenants: int, embedder: Embedder) -> dict:
    """
    Each tenant stores a unique CANARY-{tid}-{uuid} doc. Then EVERY tenant queries
    for EVERY OTHER tenant's canary string. Any result that contains another
    tenant's canary substring is a leak. Required: zero leaks across all probes.
    """
    print(f"\n=== MODE: canary isolation | n_tenants={n_tenants} ===")
    tenants = [tenant_id(i + 1) for i in range(n_tenants)]
    canaries = {t: f"CANARY-{t}-{uuid.uuid4().hex[:12]}" for t in tenants}

    async with httpx.AsyncClient(http2=False) as client:
        # 1. Each tenant stores its own canary
        print("Storing canaries...")
        for t in tenants:
            text = canaries[t]
            emb = embedder.embed(text)
            res = await broker_store(client, base_url, t, text, emb)
            print(f"  {t}: stored id={res.get('id', '?')[:12]}...")

        # 2. Every tenant queries for every OTHER tenant's canary
        print("Cross-tenant probe queries...")
        leaks = []
        total_probes = 0
        for querier in tenants:
            for target in tenants:
                if querier == target:
                    continue
                # Embed the OTHER tenant's canary string — semantically near-identical
                # to what's stored, so dense retrieval would surface it if isolation
                # leaked. The tenant filter MUST prevent it.
                target_text = canaries[target]
                qvec = embedder.embed(target_text)
                results = await broker_query(client, base_url, querier, qvec, top_k=5)
                total_probes += 1
                hits = results.get("results", []) or results.get("matches", [])
                for h in hits:
                    text = h.get("text", "") or ""
                    if f"CANARY-{target}" in text:
                        leaks.append({
                            "querier": querier, "target": target,
                            "leaked_text": text[:200],
                        })

        # 3. Each tenant queries for its OWN canary — should always succeed
        print("Self-probe sanity checks...")
        self_hits = 0
        for t in tenants:
            qvec = embedder.embed(canaries[t])
            results = await broker_query(client, base_url, t, qvec, top_k=5)
            hits = results.get("results", []) or results.get("matches", [])
            if any(canaries[t] in (h.get("text", "") or "") for h in hits):
                self_hits += 1

    out = {
        "mode": "canary",
        "n_tenants": n_tenants,
        "total_cross_probes": total_probes,
        "n_leaks": len(leaks),
        "leaks": leaks[:20],  # sample for diagnosis
        "self_hits": self_hits,
        "self_hit_rate": self_hits / n_tenants,
        "isolation_passed": len(leaks) == 0,
    }

    print(f"\n--- CANARY RESULTS ---")
    print(f"Cross-tenant probes: {total_probes}")
    print(f"Leaks detected:      {len(leaks)}  {'✓ PASS' if len(leaks) == 0 else '✗ FAIL'}")
    print(f"Self-hit rate:       {self_hits}/{n_tenants} ({self_hits/n_tenants*100:.0f}%)")
    if leaks:
        print(f"First leak: {leaks[0]}")
    return out


# ---------------------------------------------------------------------------
# Mode 3: Concurrent load + throughput ceiling
# ---------------------------------------------------------------------------
async def mode_load(base_url: str, max_tenants: int, qps_per_tenant: float,
                    duration_s: int, embedder: Embedder) -> dict:
    """
    Each tenant first stores 10 seed docs, then a worker per tenant fires
    queries at the target QPS for `duration_s` seconds. Measures p50/p95/p99
    latency and effective QPS. Sweeps tenant count: 1, 5, 10, ..., max_tenants.
    """
    print(f"\n=== MODE: load | max_tenants={max_tenants} | qps/tenant={qps_per_tenant} | duration={duration_s}s ===")

    # Seed phase — populate each tenant's namespace once
    print("Seeding tenant corpora...")
    seed_texts = [
        f"The user's favorite color is {c}." for c in ["blue", "red", "green", "yellow", "purple"]
    ] + [
        f"The user lives in {city}." for city in ["Boston", "Seattle", "Austin", "Portland", "Denver"]
    ]
    seed_embeddings = embedder.embed_batch(seed_texts)

    async with httpx.AsyncClient(timeout=10) as client:
        for i in range(max_tenants):
            t = tenant_id(i + 1)
            for text, emb in zip(seed_texts, seed_embeddings):
                try:
                    await broker_store(client, base_url, t, text, emb)
                except Exception as e:
                    print(f"  seed error for {t}: {e}")
                    break
        print(f"  seeded {max_tenants} tenants × {len(seed_texts)} docs")

    # Pre-compute a few query embeddings to recycle (avoid embedding during load test)
    query_texts = [
        "what color does the user like?",
        "where does the user live?",
        "user preferences",
        "personal information about the user",
    ]
    query_embeddings = embedder.embed_batch(query_texts)

    async def worker(tid: str, deadline: float, latencies: list, errors: list):
        async with httpx.AsyncClient(timeout=10) as client:
            interval = 1.0 / qps_per_tenant if qps_per_tenant > 0 else 0
            while time.monotonic() < deadline:
                qvec = random.choice(query_embeddings)
                t0 = time.perf_counter()
                try:
                    await broker_query(client, base_url, tid, qvec, top_k=5)
                    latencies.append(time.perf_counter() - t0)
                except Exception as e:
                    errors.append(str(e)[:80])
                if interval:
                    await asyncio.sleep(interval)

    # Sweep tenant counts
    sweep_results = []
    sweep_steps = [n for n in [1, 5, 10, 25, 50, 100] if n <= max_tenants]
    if max_tenants not in sweep_steps:
        sweep_steps.append(max_tenants)

    for n in sweep_steps:
        print(f"\n  Load step: {n} concurrent tenants × {qps_per_tenant} qps for {duration_s}s")
        latencies: list[float] = []
        errors: list[str] = []
        deadline = time.monotonic() + duration_s
        tasks = [worker(tenant_id(i + 1), deadline, latencies, errors) for i in range(n)]
        t_start = time.monotonic()
        await asyncio.gather(*tasks)
        t_elapsed = time.monotonic() - t_start

        if latencies:
            s = sorted(latencies)
            stats = {
                "n_tenants": n,
                "n_queries": len(latencies),
                "n_errors": len(errors),
                "qps_actual": round(len(latencies) / t_elapsed, 2),
                "p50_ms": round(s[len(s) // 2] * 1000, 1),
                "p95_ms": round(s[int(len(s) * 0.95)] * 1000, 1),
                "p99_ms": round(s[int(len(s) * 0.99)] * 1000, 1),
                "mean_ms": round(statistics.mean(s) * 1000, 1),
                "sample_errors": errors[:3],
            }
            sweep_results.append(stats)
            print(f"    qps={stats['qps_actual']} | p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms p99={stats['p99_ms']}ms | errors={stats['n_errors']}")

    return {"mode": "load", "sweep": sweep_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["canary", "load"])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--n-tenants", type=int, default=10)
    p.add_argument("--max-tenants", type=int, default=10, help="(load) max in sweep")
    p.add_argument("--qps-per-tenant", type=float, default=5.0, help="(load)")
    p.add_argument("--duration", type=int, default=20, help="(load) seconds per step")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    # Health check
    async with httpx.AsyncClient() as c:
        h = await c.get(f"{args.url}/health", timeout=5)
        h.raise_for_status()
        print(f"Broker health: {h.json()}")

    embedder = Embedder(device=args.device)

    if args.mode == "canary":
        result = await mode_canary(args.url, args.n_tenants, embedder)
    elif args.mode == "load":
        result = await mode_load(args.url, args.max_tenants, args.qps_per_tenant,
                                 args.duration, embedder)

    out = RESULTS_DIR / f"multitenant_{args.mode}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(amain())
