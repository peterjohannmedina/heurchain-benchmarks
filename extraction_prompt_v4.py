"""
v4 fact-extraction prompt — explicit ISO date tagging.

Diff from v3:
  + NEW: extraction now receives the SESSION_DATE as context. Each fact that
    mentions a relative date ("today", "yesterday", "last week", "3 days ago")
    MUST resolve it to an explicit ISO date using SESSION_DATE as the anchor.
  + NEW: facts mentioning absolute dates keep them verbatim.
  + Examples updated to show ISO-date facts.

Why: triage on v3 DeepSeek temporal-reasoning showed 75.9% of failures are
EXTRACTION-mode (Hit@10 high but facts don't contain a usable date for
the date-arithmetic question). LongMemEval-S provides haystack_dates per
session in the dataset metadata — we just weren't passing them to the
extractor. v4 closes that loop.

Template variables: {session_date}, {session_text}
"""

EXTRACTION_PROMPT_V4 = """Extract atomic facts from this conversation that would be useful for answering future questions about ANY aspect of it (user statements, assistant responses, recommendations, dates, durations, named items, assignments, schedules, lists, configurations).

═══════════════════════════════════════════════════════════════════════════
THIS CONVERSATION TOOK PLACE ON: {session_date}
═══════════════════════════════════════════════════════════════════════════

Each fact should be:
- A complete, self-contained statement (the future reader does NOT have this conversation).
- SPEAKER-AWARE. For user statements: "The user said/did/likes/has...". For assistant statements: "The assistant told the user that...", "The assistant recommended...", "The assistant explained...". NEVER attribute assistant content to the user.
- VERBATIM-PRESERVING. Keep all specific dates, durations, numbers, locations, named entities, brand names, titles, and direct quotes EXACTLY as stated. Do not paraphrase numbers or dates.
- For preferences and opinions, preserve the user's natural-language phrasing in quotes when the exact wording matters.
- One sentence each.

═══════════════════════════════════════════════════════════════════════════
ISO DATE TAGGING (NEW IN v4 — most important)
═══════════════════════════════════════════════════════════════════════════

For every fact that involves an event, action, or state at a specific time,
include an explicit ISO date in brackets at the end:  [date: YYYY-MM-DD]

How to derive the date:
- If the text mentions an absolute date ("on March 15"), use that.
- If the text uses a RELATIVE term ("today", "yesterday", "this morning",
  "3 days ago", "last week", "a month ago"), RESOLVE it against the
  conversation date stated above and write the explicit ISO date.
- If the text says "today" → use the conversation date.
- If the text says "yesterday" → conversation date − 1 day.
- If the text says "last week" → conversation date − 7 days (or the most
  recent specific weekday if mentioned).
- If the date is genuinely ambiguous, OMIT the date tag rather than guess.

Examples:
- Conversation date 2023-05-20. User says "I visited MoMA today and saw a
  modern art exhibit." Fact: "The user visited the Museum of Modern Art
  and saw a modern art exhibit [date: 2023-05-20]."
- Conversation date 2023-06-15. User says "I started using Ibotta about
  three weeks ago." Fact: "The user started using the Ibotta cashback app
  [date: 2023-05-25]."
- User says "On March 14, 2024 I bought a new smoker." Fact: "The user
  bought a new smoker [date: 2024-03-14]."

These ISO date tags are critical for answering questions like "how many
days between X and Y" or "in what order did events happen" — the future
reader does arithmetic on these dates.

═══════════════════════════════════════════════════════════════════════════
STRUCTURED-CONTENT UNPACKING (carried over from v3)
═══════════════════════════════════════════════════════════════════════════

When the assistant produces a TABLE, LIST, SCHEDULE, CODE BLOCK, NAMED-ENTITY
ROSTER, or ANY STRUCTURED ASSIGNMENT — do NOT summarize it as "the assistant
provided X". Instead, EXTRACT EACH ROW / ITEM / ASSIGNMENT AS A SEPARATE FACT.

For MARKDOWN TABLES specifically: column headers + row labels define the
keys; cell contents are the values. Reconstruct each (row, column) → value
mapping as a complete sentence.

═══════════════════════════════════════════════════════════════════════════
UPDATES AND REVISIONS
═══════════════════════════════════════════════════════════════════════════

If a fact UPDATES or REPLACES an earlier fact, extract the CURRENT/LATEST
version explicitly as the current state, and reference the prior value if
the change itself is informative.

If the assistant emits multiple REVISIONS of the same structured content,
extract only the FINAL/LATEST version's individual assignments.

═══════════════════════════════════════════════════════════════════════════
NO META-COMMENTARY
═══════════════════════════════════════════════════════════════════════════

Do NOT extract facts about what the assistant *did* in the abstract —
extract the CONTENT it produced. Skip phrases like "the assistant created",
"the assistant provided", "the assistant updated" UNLESS the action itself
is what matters.

═══════════════════════════════════════════════════════════════════════════

CONVERSATION:
{session_text}

Output ONLY a JSON array of strings. No commentary, no preamble. Examples of good facts (note the ISO date tags):
["The user visited the Museum of Modern Art and saw a modern art exhibit [date: 2023-05-20].",
 "The user started using the Ibotta cashback app [date: 2023-05-25].",
 "Admon was assigned to the 8am-4pm Day Shift on Sundays.",
 "The user's current 5K personal best is 25:50, improved from a prior best of 27:12 [date: 2024-04-12]."]

FACTS:"""
