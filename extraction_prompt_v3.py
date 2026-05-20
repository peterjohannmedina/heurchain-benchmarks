"""
v3 fact-extraction prompt — preserves structured content.

Diff from v2:
  + NEW rules for tables, lists, schedules, code, and assignments
  + Explicit instruction to UNPACK each row/cell of structured assistant
    output into its own fact (the v2 failure mode: "the assistant created
    a rotation sheet" instead of "Admon was assigned to the 8am-4pm Day
    Shift on Sundays")
  + Stronger anti-meta-summary directive: do not describe WHAT the
    assistant DID; extract the CONTENT it produced
  + Question-anticipation framing: imagine the future questions a user
    would ask about this conversation and write facts that answer them
"""

EXTRACTION_PROMPT_V3 = """Extract atomic facts from this conversation that would be useful for answering future questions about ANY aspect of it (user statements, assistant responses, recommendations, dates, durations, named items, assignments, schedules, lists, configurations).

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

Bad (meta-summary): "The user listed their preferences."
Good (unpacked):    "The user prefers dark roast coffee.",
                    "The user dislikes hazelnut flavoring.",
                    "The user's favorite brand is Stumptown Hair Bender.",
                    ...etc, one fact per preference.

For MARKDOWN TABLES specifically: column headers + row labels define the
keys; cell contents are the values. Reconstruct each (row, column) → value
mapping as a complete sentence.

═══════════════════════════════════════════════════════════════════════════
UPDATES AND REVISIONS
═══════════════════════════════════════════════════════════════════════════

If a fact UPDATES or REPLACES an earlier fact in this same conversation
(e.g., the user changed an answer, the assistant revised a recommendation,
a count increased, a status flipped) — extract the CURRENT/LATEST version
explicitly as the current state, and reference the prior value if the
change itself is informative.

Example: "The user's current 5K personal best is 25:50 (improved from a
prior best of 27:12)."

If the assistant emits multiple REVISIONS of the same structured content
(e.g., three drafts of a schedule with different agent names), extract only
the FINAL/LATEST version's individual assignments — earlier drafts are
superseded.

═══════════════════════════════════════════════════════════════════════════
QUESTION ANTICIPATION
═══════════════════════════════════════════════════════════════════════════

Before extracting, imagine the future questions a user might ask about
this conversation:
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
"the assistant provided", "the assistant updated", "the user requested" UNLESS
the action itself (without the content) is what matters.

Bad:  "The assistant created a 4-week shift rotation."
Good: "The shift rotation covers a 4-week period.",
      "The shift rotation has 7 agents.",
      [plus the per-row assignments unpacked as above]

═══════════════════════════════════════════════════════════════════════════

CONVERSATION:
{session_text}

Output ONLY a JSON array of strings. No commentary, no preamble. Examples of good facts:
["The user graduated with a degree in Business Administration on May 14, 2024.",
 "The user's current personal best 5K time is 25:50, improved from a prior best of 27:12.",
 "Admon was assigned to the 8am-4pm Day Shift on Sundays.",
 "Magdy was assigned to the 12pm-8pm Afternoon Shift on Sundays.",
 "The assistant recommended Sukiyabashi Jiro for sushi in Tokyo's Ginza district.",
 "The user said they 'prefer dark roast coffee, especially Stumptown's Hair Bender blend'."]

FACTS:"""
