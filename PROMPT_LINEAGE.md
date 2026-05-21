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

## v4 — ISO date tagging (extraction layer)

**File:** [`extraction_prompt_v4.py`](extraction_prompt_v4.py)

**Diff from v3:** receives the SESSION_DATE from dataset metadata (LongMemEval's `haystack_dates`); requires each fact to carry an explicit `[date: YYYY-MM-DD]` ISO tag, resolving relative phrases ("today", "yesterday", "3 weeks ago") against the session date.

**Targeted measurement** (`targeted_reextract.py` on the 22 temporal-reasoning EXTRACTION failures from v3 DeepSeek):

| Metric | Value |
|---|---:|
| Wins (0→1) | **1/22 (4.5%)** |
| Losses (1→0) | 0 |
| Facts contain gold (token-overlap heuristic) | 0/22 |

**Findings:** extraction is doing its job (every fact gets an ISO date), but the lift didn't materialize because **the answerer doesn't know today's date** (for "how many X ago" questions) and **retrieval doesn't surface multiple events** (for "between X and Y" questions). v4 alone is necessary but insufficient.

---

## v4a — date-aware answer prompt (FIRST answer-layer iteration)

**File:** [`answer_prompt_v4a.py`](answer_prompt_v4a.py)

**Diff from baseline ANSWER_PROMPT:** receives the `{question_date}` from the dataset's `question_date` metadata; instructs the answerer to compute `(question_date − event_date)` using ISO tags in facts. Retains the existing "if no fact answers... reply: I don't know" rule.

**Targeted measurement** (`targeted_reanswer.py` on v4 facts of the same 22 failures):

| Metric | Value |
|---|---:|
| Wins | **2/22 (9.1%)** vs v4 alone's 1/22 |
| Refusal rate | 81.8% |

**Findings:** correct cases now do real date arithmetic (e.g. "You met Emma 9 days ago." for gold "9 days ago"). But refusal rate stays high because the strict "I don't know" instruction trips even when the relevant event IS in facts with a date.

---

## v4b — softer answerer (SECOND answer-layer iteration)

**File:** [`answer_prompt_v4b.py`](answer_prompt_v4b.py)

**Diff from v4a:** narrowed the refusal threshold. Only refuse if (a) the relevant event is NOT mentioned in any fact, OR (b) the relevant fact has NO ISO date AND no anchor. Otherwise, attempt the arithmetic and round to the nearest unit the question asks for.

**Targeted measurement** (same 22 failures, same v4 facts):

| Metric | Value |
|---|---:|
| **Wins** | **7/22 (31.8%)** ← +6 over v4a |
| Losses | 0 |
| Refusal rate | 54.5% (down from 81.8%) |
| Wall time | 1.7 min |

**Findings:** ~25 minutes of answer-prompt iteration produced a +27.3pp recovery on the 22 v3 EXTRACTION failures. The "still refusing" cases (5/15 remaining) are all "between A and B" multi-event questions where one event's session wasn't retrieved.

---

## v4c — event index (PROPOSED, not yet built)

**Justification from the v4b triage:** of v4b's 15 remaining wrong answers, 7 are pure retrieval misses (relevant session not in top-10) and 5 are multi-event "between" questions where one of two needed events is missing from facts. **80%+ of the residual is retrieval / multi-event-retrieval shaped.**

**Proposed design:**
- At extraction time, parse v4 facts for `[date: YYYY-MM-DD]` tags + named-event phrases → populate a side `events` SQLite table: `(event_phrase, iso_date, session_id, fact_id)`.
- At query time, detect "between/order/ago" patterns → query the events table in parallel with regular retrieval → pass both as context to the answerer.
- Targeted test the same way: `targeted_reretrieve.py` (sibling track 3 of the Karpathy cycle, also not yet built).

Expected lift: addresses the ~12/15 remaining failures. If even half land, full v4+v4b+event-index temporal-reasoning could reach 40-55% (vs v3's 3.33%).

---

## Multi-track iteration matrix (what we have now)

|                  | Answer v0 (baseline) | Answer v4a | **Answer v4b** |
|---|---:|---:|---:|
| Extraction v3 (baseline) | 3.33% (full cat) | not tested | not tested |
| **Extraction v4 (ISO dates)** | 4.5% (on failures) | 9.1% (on failures) | **31.8% (on failures)** |

Each cell is one targeted run. The +27.3pp move from v4-alone to v4+v4b was a ~7 min cycle. **This is the value of separating extraction + answer iteration tracks** — without `targeted_reanswer.py` we'd have wrongly attributed v4's null result to the prompt itself and moved on.

---

## How to use this lineage

When proposing a new prompt version (vN):

1. Run `triage_failures.py --input <baseline-result> --mode EXTRACTION --top 10` to see what's failing at the extraction layer in the current baseline.
2. Draft `extraction_prompt_vN.py` with the hypothesized fix.
3. Run `targeted_reextract.py --baseline <baseline> --new-prompt extraction_prompt_vN.py --judge-model deepseek-v3.1:671b-cloud` — measures vN against vN-1 on just the failures, ~10 min.
4. If lift: deploy vN to `systems/fact_extraction.py`, full overnight re-extract.
5. Append the result to this lineage with measured numbers from the cross-judge harness.

Append-only — never delete a version's entry. If we abandon vN, leave the entry with an explanation of why.
