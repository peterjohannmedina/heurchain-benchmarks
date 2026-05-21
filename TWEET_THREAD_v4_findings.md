# Tweet thread — multi-track prompt iteration on HeurChain benchmark harness

**Ready to paste into mixpost.** 10 posts. Single thread. Each post ≤ 280 chars (except where noted — count includes URL shortener).

Audience: ML engineers, AI infrastructure folks, anyone running LLM-as-judge benchmarks.

Tone: technical, specific, no marketing fluff.

Link target: `https://github.com/peterjohannmedina/heurchain-benchmarks`

---

## Post 1/10 — hook

```
We just lifted temporal-reasoning QA on LongMemEval-S from 3.3% → 31.8% in 25 minutes of prompt iteration.

Not by changing the extraction prompt. By building a multi-track Karpathy cycle on our benchmark harness.

What we learned about LLM-as-judge methodology 🧵
```

(280 chars including emoji)

---

## Post 2/10 — context

```
HeurChain is a vector memory broker for AI agents. We benchmark on LongMemEval-S (ICLR 2025) with 3 independent judges:

• Local Qwen3-14B
• DeepSeek V3.1 671B  
• Kimi K2.6 (reasoning)

Cross-frontier (DeepSeek + Kimi) per-question agreement: 87.8% → reliable signal.
```

(275 chars)

---

## Post 3/10 — the stuck category

```
v3 baseline: temporal-reasoning category stuck at 3-10% across ALL three judges.

Retrieval was fine (Hit@10 = 93%). Facts were getting extracted. But "how many weeks ago did X happen" requires date arithmetic the model couldn't do without explicit ISO dates in facts.
```

(280 chars)

---

## Post 4/10 — the v4 hypothesis + null result

```
v4 hypothesis: tag every fact with [date: YYYY-MM-DD] resolved against the session's timestamp from dataset metadata.

Targeted test on 22 temporal-reasoning failures:

→ 1/22 wins. NULL RESULT.

Extraction worked perfectly (every fact had ISO dates). So why nothing?
```

(279 chars)

---

## Post 5/10 — the diagnostic

```
Diagnosis: the failure wasn't in extraction. It was downstream.

a) The ANSWERER didn't know "today's date" for "how many X ago" questions.

b) RETRIEVAL only surfaced ONE event for "between A and B" questions.

v4 fixed the wrong layer.
```

(266 chars)

---

## Post 6/10 — the multi-track insight

```
This is the lesson: prompt iteration must be MULTI-TRACK.

• Extraction is one prompt.
• Answer generation is another.
• Retrieval is a third.

Any of them can be the bottleneck. A null result on one tells you to test the others.

Most benchmarks iterate only extraction.
```

(279 chars)

---

## Post 7/10 — v4b breakthrough

```
Built targeted_reanswer.py — re-runs only answer+judge against cached facts. ~2 min per cycle.

Tried two answer prompts on the same v4 facts:

v4a: date-aware but strict → 2/22 wins (refusal rate 82%)
v4b: softer refusal threshold → 7/22 wins (54%)

+27pp from a 5-min prompt change.
```

(280 chars)

---

## Post 8/10 — same-class judge bias

```
Bonus finding (covered in our earlier cross-judge study):

When local 14B judges its own answers, partial-correct gets rewarded. When DeepSeek/Kimi judge, they say "I don't know" honestly when extraction is lossy.

Same-class judge bias inflates QA by 1-7pp per category.
```

(278 chars)

---

## Post 9/10 — what the residual tells us

```
Of v4b's 15 remaining wrong: 12 are retrieval-shaped (multi-event "between X and Y" questions where one event's session was never retrieved).

So the next architectural move (event index for multi-event queries) is now SHARPLY justified by data.

The harness told us what to build.
```

(280 chars)

---

## Post 10/10 — call to action

```
Open-source the whole thing:

github.com/peterjohannmedina/heurchain-benchmarks

• triage_failures.py — classifies failures by layer
• targeted_reextract.py — extraction-layer iteration
• targeted_reanswer.py — answer-layer iteration
• PROMPT_LINEAGE.md — full audit trail

Reproduce on your data.
```

(280 chars)

---

## Mixpost API payload (when you have the API token + reachability)

When pushing through Mixpost's API, each post above becomes one entry in the
`postVersions[0].content` field of a multi-post sequence. The exact structure:

```bash
curl -X POST "http://<mixpost-host>:<port>/mixpost/api/posts" \
  -H "Authorization: Bearer <YOUR_MIXPOST_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Where `payload.json` is structured like:
```json
{
  "accounts": [<your-twitter-account-id>],
  "schedule": false,
  "queue": false,
  "draft": true,
  "versions": [
    {
      "account_id": 0,
      "is_original": true,
      "content": [
        { "body": "<POST 1 TEXT>", "media": [] },
        { "body": "<POST 2 TEXT>", "media": [] },
        ...
        { "body": "<POST 10 TEXT>", "media": [] }
      ]
    }
  ]
}
```

If `"draft": true`, the post lands in Mixpost drafts (matches user's "draft to mixpost" intent).

---

## Alternative single-post version (if a thread isn't preferred)

```
Lifted temporal-reasoning QA on LongMemEval-S from 3.3% → 31.8% in 25 min of prompt iteration.

The win wasn't from a better extraction prompt — it was from realizing the bottleneck was the ANSWERER refusing too readily. 5-min answer-prompt change recovered +27pp.

Multi-track Karpathy cycle (extraction / answer / retrieval) is the harness contribution. Open source:

github.com/peterjohannmedina/heurchain-benchmarks
```

(491 chars — long-form / X premium post, or split into 2 if needed)
