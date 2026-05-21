# Extraction prompt lineage

Append-only audit trail of fact-extraction prompt versions tested against LongMemEval-S.

For each version: what it does differently, what's known to lift, what regressed (if anything), measured numbers on the cross-judge harness (Local 14B + DeepSeek V3.1 + Kimi K2.6 — see [results/COMPARISON_v2_cloud_judge.md](results/COMPARISON_v2_cloud_judge.md) for methodology).

---

## v2 — speaker-aware + verbatim-preserving (May 2026, baseline)

**File:** [`systems/fact_extraction.py`](https://github.com/peterjohannmedina/heurchain) at v2 git tag (also archived locally as `fact_extraction.py.v2bak`)

**What it asked for:**
- SPEAKER-AWARE facts (don't conflate user/assistant statements)
- VERBATIM-PRESERVING for dates, durations, numbers, named entities
- UPDATES rule (current state, with prior value if relevant)

**Measured (6 categories × 30 tasks, cross-judge):**

| Category | Local 14B | DeepSeek | Kimi |
|---|---:|---:|---:|
| single-session-assistant | 43.33% | 50.00% | 36.67% |
| temporal-reasoning | 10.00% | 3.33% | 3.33% |
| single-session-user | 53.33% | 53.33% | 53.33% |
| single-session-preference | 13.33% | 23.33% | 3.33% |
| knowledge-update | 60.00% | 53.33% | 53.33% |
| multi-session | 16.67% | 6.67% | 20.00% |
| **6-cat mean** | **32.78%** | **31.66%** | **28.33%** |

**Known failure mode (smoking gun):** when the assistant emits a markdown table / list / schedule, v2 summarized it as "the assistant created a rotation sheet" instead of unpacking each row into a per-row fact. The canonical Admon shift question had Hit@10=100% but the extracted facts didn't contain the answer at all.

---

## v3 — structured-content unpacking (May 2026)

**File:** [`extraction_prompt_v3.py`](extraction_prompt_v3.py), deployed in `systems/fact_extraction.py` ≥ 2026-05-20.

**Diff from v2:**
- **NEW** `STRUCTURED-CONTENT UNPACKING` rule: when assistant produces a table / list / roster / schedule, extract each row as a SEPARATE fact instead of meta-summarizing the action.
- **NEW** markdown table specifics: column headers + row labels = keys; cell contents = values; reconstruct each (row, column) → value mapping as a complete sentence.
- **NEW** explicit anti-meta-summary directive: do NOT extract "the assistant created/provided X" — extract the CONTENT it produced.
- **NEW** question-anticipation framing: imagine future "who was assigned to X? when did Y happen?" questions.
- **NEW** revision rule: when assistant emits multiple drafts, extract only the FINAL version's individual assignments.
- Examples updated to include per-row unpacking patterns.

**Measured (weak 3 categories × 30 tasks, cross-judge):**

| Category | v2→v3 Local | v2→v3 DeepSeek | v2→v3 Kimi |
|---|---:|---:|---:|
| multi-session | +20.00 pp (16.67% → 36.67%) | +10.00 pp (6.67% → 16.67%) | ±0.00 pp (20.00%) |
| knowledge-update | ±0.00 pp (60.00%) | +13.34 pp (53.33% → 66.67%) | +3.34 pp (53.33% → 56.67%) |
| temporal-reasoning | −3.33 pp (10.00% → 6.67%) | ±0.00 pp (3.33%) | +3.34 pp (3.33% → 6.67%) |
| **Weak-3 mean** | **+5.56 pp** | **+7.78 pp** | **+2.23 pp** |

**Cross-frontier agreement on v3 facts: 87.8%** — identical to v2. The judges remain independently reliable; v3 didn't break anything diagnostic.

**Other observable signals:**
- Facts per session: 12.5 (v2) → 19.9 (v3), +59% — confirms structured-content unpacking is active.
- Hit@10 marginally improved on multi-session (96.67% → 100.00%) — more granular facts → better retrieval matching.
- DeepSeek `fact_contains_gold_rate` on v3: multi-session 16.7%, knowledge-update 53.3%, temporal-reasoning 20.0%.
  → multi-session still has lots of headroom; temporal-reasoning's bottleneck is NOT extraction (it's cross-session date arithmetic).

**Confirmed v3 strengths:**
- Tables with named-entity assignments (the Admon shift case): 100% gold-overlap recovery.
- Multi-session entity tracking lifts at frontier judges (+10-13pp on knowledge-update + multi-session).

**Confirmed v3 limitations:**
- Temporal-reasoning: stuck at 3-7% across all judges. Needs different intervention (v4 ISO date tagging, or temporal-aware retrieval, or a graph-lite event index — TBD).
- Single-session-preference: not re-tested in v3 weak-3 sweep; was 13-23% in v2.

---

## v4 — TBD: ISO date tagging

**Status:** planned, not yet built.

**Hypothesis:** the temporal-reasoning failure mode is that v3 facts say "the user visited MoMA today" instead of "the user visited MoMA on 2024-03-15". With explicit ISO dates in facts, the answerer can perform date arithmetic.

**Planned diff from v3:**
- NEW rule: tag each fact with an explicit ISO date (`[date: YYYY-MM-DD]`) when the conversation contains an absolute or inferable date.
- NEW: when a session's date is known from session metadata (e.g. session timestamp), thread it into relative-date facts ("yesterday" → explicit date).
- KEEP all v3 rules unchanged.

**Test plan:** use `targeted_reextract.py` with `extraction_prompt_v4.py` on the temporal-reasoning EXTRACTION failures from v3. ~10 min decision loop. If lift, full v4 overnight re-extract.

---

## How to use this lineage

When proposing a new prompt version (vN):

1. Run `triage_failures.py --input <baseline-result> --mode EXTRACTION --top 10` to see what's failing at the extraction layer in the current baseline.
2. Draft `extraction_prompt_vN.py` with the hypothesized fix.
3. Run `targeted_reextract.py --baseline <baseline> --new-prompt extraction_prompt_vN.py --judge-model deepseek-v3.1:671b-cloud` — measures vN against vN-1 on just the failures, ~10 min.
4. If lift: deploy vN to `systems/fact_extraction.py`, full overnight re-extract.
5. Append the result to this lineage with measured numbers from the cross-judge harness.

Append-only — never delete a version's entry. If we abandon vN, leave the entry with an explanation of why.
