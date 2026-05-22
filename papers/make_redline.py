#!/usr/bin/env python3
"""
Generate a v4c-redline version of the paper.

The v4c update added two whole new subsections (5.6 + 6.5) + Appendix A.4,
and surgically inserted v4c content into Abstract, §1, §5.5, §6.3, §6.4,
§8, §9, §10. This script marks those additions with <mark class="new"> tags
so pandoc renders them as highlighted-yellow inline marks in the HTML.

Read paths:
  papers/v4_findings_draft.md     — current (v4c) version
Write paths:
  papers/v4_findings_draft_REDLINE.md   — markdown w/ <mark> tags
  /c/Users/NM2/Dropbox/heurchain/v4_findings_draft_REDLINE.html  — rendered
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "v4_findings_draft.md"
REDLINE_MD = Path(__file__).resolve().parent / "v4_findings_draft_REDLINE.md"

# Whole sections that are NEW in v4c (the agent created these from scratch)
WHOLE_NEW_SECTIONS = {
    "5.6",   # Event Index Bridges the Retrieval-Miss Gap
    "6.5",   # More context does not help — selection matters
    "A.4",   # Event index schema sketch
}

# v4c-specific content patterns to flag inline (within modified sections)
V4C_KEYWORDS = [
    r"\bv4c\b",
    r"\bhaystack[_\s]+re-?extract\w*\b",
    r"\bevent[_\s\-]+(?:idx|index)\b",
    r"\bevent-idx\b",
    r"\b45\.5\s*%\b",          # The cascade endpoint
    r"\b\+\s*33\s*pp\b",       # The cascade total
    r"\b36\.6?7?\s*%\b",       # Full-category projection
    r"\bselection over recall\b",
    r"\bselection matters\b",
    r"\bmore context (?:does not|doesn't) help\b",
    r"\bevent index\b",
    r"\bbuild_event_index\.py\b",
    r"\bhaystack_reextract\.py\b",
    r"\btest_haystack_answer\.py\b",
    r"\b66\.7\s*%\b",          # contains_gold recovery rate
    r"\b8/12\b",               # also recovery rate
    r"\b3/12\b",               # event-idx wins
    r"\bfour contributions?\b",
    r"\bthree iteration tracks?\b",
    r"\bthree[\s-]+layer(?:ed)?\s+cascade\b",
    r"\bv4c stage[\s-]+\d\b",
    r"\bcascade\b",            # heavily v4c-associated
    r"\bTable 7\b",
    r"\bTable 8\b",
    r"\bTable 9\b",
]

KEYWORD_RE = re.compile("|".join(f"({p})" for p in V4C_KEYWORDS), re.IGNORECASE)


def mark_section_heading(line):
    """Detect a section/subsection/appendix heading and return its number+title, or None."""
    # markdown headings like "## 5.6 Title" or "### 6.5 Title" or "## Appendix A.4 ..."
    m = re.match(r"^#+\s+(?:Appendix\s+)?([0-9A-Z]+(?:\.[0-9A-Z]+)*)\s", line)
    if m:
        return m.group(1)
    return None


def is_subordinate(section_id, parent_id):
    """Is section_id a sub-section of parent_id (or equal)?"""
    return section_id == parent_id or section_id.startswith(parent_id + ".")


def main():
    src_text = SRC.read_text(encoding="utf-8")
    lines = src_text.split("\n")
    out_lines = []

    in_whole_new_section = False
    current_section = ""

    for line in lines:
        # Section boundary?
        heading_id = mark_section_heading(line)
        if heading_id:
            current_section = heading_id
            # Are we entering one of the whole-new sections?
            in_whole_new_section = any(
                is_subordinate(heading_id, new_sec) for new_sec in WHOLE_NEW_SECTIONS
            )
            # But check if this heading IS the new section starting — mark its title too
            if in_whole_new_section:
                out_lines.append(f"<mark class='new-block'>{line}</mark>")
                continue
            # If exiting a whole-new section into a new section, just emit the heading normally
            out_lines.append(line)
            continue

        if in_whole_new_section:
            # Wrap entire line as a new-block
            if line.strip():
                out_lines.append(f"<mark class='new-block'>{line}</mark>")
            else:
                out_lines.append(line)
            continue

        # Inline keyword marking: wrap matched spans with <mark class="new-inline">
        def repl(m):
            return f"<mark class='new-inline'>{m.group(0)}</mark>"
        marked = KEYWORD_RE.sub(repl, line)
        out_lines.append(marked)

    REDLINE_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {REDLINE_MD}")
    print(f"  Original lines: {len(lines)}")
    print(f"  Whole-new section lines marked: {sum(1 for l in out_lines if 'new-block' in l)}")
    print(f"  Inline marks: {sum(out_lines[i].count('new-inline') for i in range(len(out_lines)))}")


if __name__ == "__main__":
    main()
