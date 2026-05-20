#!/usr/bin/env python3
"""
Smoke-test v2 vs v3 extraction prompts on a single session.

Runs both prompts against the same model (local llama-server) and prints
both fact lists side-by-side for visual inspection.

Usage:
  python3 smoke_extraction_v2_vs_v3.py <session-text-file> [--gold "<gold answer>"]
"""
import argparse, json, re, sys, time
from pathlib import Path
import httpx

LLAMA_URL = "http://localhost:8080/v1"

V2 = """Extract atomic facts from this conversation that would be useful for answering future questions about ANY aspect of it (user statements, assistant responses, recommendations, dates, durations, named items).

Each fact should be:
- A complete, self-contained statement (the future reader does NOT have this conversation).
- SPEAKER-AWARE. For user statements: "The user said/did/likes/has...". For assistant statements: "The assistant told the user that...", "The assistant recommended...", "The assistant explained...". NEVER attribute assistant content to the user.
- VERBATIM-PRESERVING. Keep all specific dates, durations, numbers, locations, named entities, brand names, titles, and direct quotes EXACTLY as stated. Do not paraphrase numbers or dates.
- For preferences and opinions, preserve the user's natural-language phrasing in quotes when the exact wording matters.
- One sentence each.

If a fact UPDATES or REPLACES an earlier fact (e.g., the user moved cities, changed jobs, finished a count, set a new record), phrase it explicitly as the current state and reference the prior value if known.

CONVERSATION:
{session_text}

Output ONLY a JSON array of strings. No commentary, no preamble. Examples of good facts:
["The user graduated with a degree in Business Administration on May 14, 2024.", "The user's current personal best 5K time is 25:50, beating their prior best of 27:12.", "The assistant recommended the book 'Atomic Habits' by James Clear for habit-formation.", "The user said they 'prefer dark roast coffee, especially Stumptown's Hair Bender blend'.", "The user has visited 4 Korean restaurants in Boston as of June 2024."]

FACTS:"""

V3 = """Extract atomic facts from this conversation that would be useful for answering future questions about ANY aspect of it (user statements, assistant responses, recommendations, dates, durations, named items, assignments, schedules, lists, configurations).

Each fact should be:
- A complete, self-contained statement (the future reader does NOT have this conversation).
- SPEAKER-AWARE. For user statements: "The user said/did/likes/has...". For assistant statements: "The assistant told the user that...", "The assistant recommended...", "The assistant explained...". NEVER attribute assistant content to the user.
- VERBATIM-PRESERVING. Keep all specific dates, durations, numbers, locations, named entities, brand names, titles, and direct quotes EXACTLY as stated. Do not paraphrase numbers or dates.
- For preferences and opinions, preserve the user's natural-language phrasing in quotes when the exact wording matters.
- One sentence each.

═══════════════════════════════════════════════════════════════════════════
STRUCTURED-CONTENT UNPACKING (most important — v2 failed at this)
═══════════════════════════════════════════════════════════════════════════

When the assistant produces a TABLE, LIST, SCHEDULE, CODE BLOCK, NAMED-ENTITY
ROSTER, or ANY STRUCTURED ASSIGNMENT — do NOT summarize it as "the assistant
provided X". Instead, EXTRACT EACH ROW / ITEM / ASSIGNMENT AS A SEPARATE FACT.

Bad (meta-summary): "The assistant provided a shift rotation sheet for 7 agents."
Good (unpacked):    "Admon was assigned to the 8am-4pm Day Shift on Sundays.",
                    "Magdy was assigned to the 12pm-8pm Afternoon Shift on Sundays.",
                    ...etc, one fact per assignment.

Bad (meta-summary): "The assistant recommended several restaurants in Tokyo."
Good (unpacked):    "The assistant recommended Sukiyabashi Jiro for sushi in Tokyo's Ginza district.",
                    "The assistant recommended Narisawa for innovative kaiseki in Tokyo's Aoyama district.",
                    ...etc, one fact per recommendation.

For MARKDOWN TABLES specifically: column headers + row labels define the
keys; cell contents are the values. Reconstruct each (row, column) → value
mapping as a complete sentence.

═══════════════════════════════════════════════════════════════════════════
UPDATES AND REVISIONS
═══════════════════════════════════════════════════════════════════════════

If a fact UPDATES or REPLACES an earlier fact (e.g., the user changed an
answer, the assistant revised a recommendation, a count increased, a status
flipped) — extract the CURRENT/LATEST version explicitly as the current
state, and reference the prior value if the change itself is informative.

If the assistant emits multiple REVISIONS of the same structured content
(e.g., three drafts of a schedule with different agent names), extract only
the FINAL/LATEST version's individual assignments — earlier drafts are
superseded.

═══════════════════════════════════════════════════════════════════════════
QUESTION ANTICIPATION
═══════════════════════════════════════════════════════════════════════════

Before extracting, imagine the future questions a user might ask:
  - "Who was assigned to X?"
  - "When did Y happen?"
  - "What was my Z?"
  - "What did the assistant recommend for W?"
  - "How did <named entity>'s <attribute> change?"

Your facts must answer questions like these directly, without paraphrase.

═══════════════════════════════════════════════════════════════════════════
NO META-COMMENTARY
═══════════════════════════════════════════════════════════════════════════

Do NOT extract facts about what the assistant *did* in the abstract —
extract the CONTENT it produced. Skip phrases like "the assistant created",
"the assistant provided", "the assistant updated" UNLESS the action itself
(without the content) is what matters.

═══════════════════════════════════════════════════════════════════════════

CONVERSATION:
{session_text}

Output ONLY a JSON array of strings. No commentary, no preamble. Examples of good facts:
["The user graduated with a degree in Business Administration on May 14, 2024.",
 "Admon was assigned to the 8am-4pm Day Shift on Sundays.",
 "Magdy was assigned to the 12pm-8pm Afternoon Shift on Sundays.",
 "The assistant recommended Sukiyabashi Jiro for sushi in Tokyo's Ginza district.",
 "The user said they 'prefer dark roast coffee, especially Stumptown's Hair Bender blend'."]

FACTS:"""


def extract(prompt_template, session_text, max_tokens=2400):
    prompt = prompt_template.format(session_text=session_text[:6000])
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = httpx.post(f"{LLAMA_URL}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    elapsed = time.time() - t0
    text = r.json()["choices"][0]["message"]["content"].strip()
    # Parse JSON list from response
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [], elapsed
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()], elapsed
    except json.JSONDecodeError:
        return [], elapsed
    return [], elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_file")
    p.add_argument("--gold", default=None, help="gold answer to check fact-containment for")
    args = p.parse_args()

    session_text = Path(args.session_file).read_text()
    print(f"Session: {args.session_file}  ({len(session_text)} chars)")
    if args.gold:
        print(f"Gold:    {args.gold}")
    print()

    print("=" * 70)
    print("RUNNING v2 prompt...")
    print("=" * 70)
    v2_facts, v2_time = extract(V2, session_text)
    print(f"v2: {len(v2_facts)} facts in {v2_time:.1f}s")
    for i, f in enumerate(v2_facts, 1):
        print(f"  v2[{i:2}] {f}")

    print()
    print("=" * 70)
    print("RUNNING v3 prompt...")
    print("=" * 70)
    v3_facts, v3_time = extract(V3, session_text)
    print(f"v3: {len(v3_facts)} facts in {v3_time:.1f}s")
    for i, f in enumerate(v3_facts, 1):
        print(f"  v3[{i:2}] {f}")

    if args.gold:
        print()
        print("=" * 70)
        print("ANSWER-CONTAINMENT CHECK")
        print("=" * 70)
        gold_lower = args.gold.lower()
        # Simple: any fact substring-overlap-rich with gold?
        gold_toks = set(re.findall(r"\b\w+\b", gold_lower))
        gold_toks = {t for t in gold_toks if len(t) > 2 and t not in {"the","and","was","were","that","this"}}

        def best_match(facts):
            best_score, best_fact = 0, None
            for f in facts:
                f_toks = set(re.findall(r"\b\w+\b", f.lower()))
                overlap = len(gold_toks & f_toks) / max(1, len(gold_toks))
                if overlap > best_score:
                    best_score, best_fact = overlap, f
            return best_score, best_fact

        v2_score, v2_match = best_match(v2_facts)
        v3_score, v3_match = best_match(v3_facts)
        print(f"v2 best fact overlap with gold:  {v2_score:.2f}")
        print(f"     fact: {v2_match}")
        print(f"v3 best fact overlap with gold:  {v3_score:.2f}")
        print(f"     fact: {v3_match}")
        print()
        if v3_score > v2_score + 0.2:
            print(f"  ✓ v3 substantially better ({v3_score:.2f} vs {v2_score:.2f})")
        elif v3_score > v2_score:
            print(f"  ~ v3 marginally better")
        elif v3_score == v2_score:
            print(f"  = tied")
        else:
            print(f"  ✗ v3 worse — investigate")


if __name__ == "__main__":
    main()
