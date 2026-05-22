#!/usr/bin/env python3
"""
Generate a v4c-redline version of the paper using pandoc's NATIVE
fenced-div + span syntax (NOT raw <mark> tags). This preserves markdown
parsing inside marked blocks — critical for tables, lists, code, etc.

Pandoc fenced div:    ::: {.new-block} ... :::  →  <div class="new-block">
Pandoc inline span:   [text]{.new-inline}        →  <span class="new-inline">

CSS in the rendering header targets `.new-block` and `.new-inline` classes
regardless of element type.

Read/write paths:
  in : papers/v4_findings_draft.md  (current v4c version)
  out: papers/v4_findings_draft_REDLINE.md  (markdown w/ pandoc div/span)
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "v4_findings_draft.md"
REDLINE_MD = Path(__file__).resolve().parent / "v4_findings_draft_REDLINE.md"

# Whole sections that are NEW in v4c (the agent created these from scratch)
WHOLE_NEW_SECTIONS = {
    "5.6",
    "6.5",
    "A.4",
}

# v4c-specific content patterns to flag inline within modified sections
V4C_KEYWORDS = [
    r"\bv4c\b",
    r"\bhaystack[_\s]+re-?extract\w*\b",
    r"\bevent[_\s\-]+(?:idx|index)\b",
    r"\bevent-idx\b",
    r"\b45\.5\s*%\b",
    r"\b\+\s*33\s*pp\b",
    r"\b36\.6?7?\s*%\b",
    r"\bselection over recall\b",
    r"\bselection matters\b",
    r"\bmore context (?:does not|doesn't) help\b",
    r"\bevent index\b",
    r"\bbuild_event_index\.py\b",
    r"\bhaystack_reextract\.py\b",
    r"\btest_haystack_answer\.py\b",
    r"\b66\.7\s*%\b",
    r"\b8/12\b",
    r"\b3/12\b",
    r"\bfour contributions?\b",
    r"\bthree iteration tracks?\b",
    r"\bthree[\s-]+layer(?:ed)?\s+cascade\b",
    r"\bv4c stage[\s-]+\d\b",
    r"\bTable 7\b",
    r"\bTable 8\b",
    r"\bTable 9\b",
]

KEYWORD_RE = re.compile("|".join(f"({p})" for p in V4C_KEYWORDS), re.IGNORECASE)


def mark_section_heading(line):
    """Detect a section/subsection/appendix heading; return its dotted ID or None."""
    m = re.match(r"^#+\s+(?:Appendix\s+)?([0-9A-Z]+(?:\.[0-9A-Z]+)*)\s", line)
    if m:
        return m.group(1)
    return None


def is_subordinate(section_id, parent_id):
    return section_id == parent_id or section_id.startswith(parent_id + ".")


def mark_inline(line):
    """Wrap matched v4c keywords in pandoc span syntax: [text]{.new-inline}"""
    def repl(m):
        return f"[{m.group(0)}]{{.new-inline}}"
    return KEYWORD_RE.sub(repl, line)


def main():
    src_text = SRC.read_text(encoding="utf-8")
    lines = src_text.split("\n")
    out_lines = []

    in_whole_new_section = False
    current_new_section = None  # Track which new-block we're inside (to close it cleanly)
    in_code_fence = False

    for line in lines:
        # Track code fences to avoid marking content inside them
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            out_lines.append(line)
            continue

        heading_id = mark_section_heading(line)
        if heading_id and not in_code_fence:
            # We're at a section boundary. If currently inside a new-block, close it.
            if in_whole_new_section:
                out_lines.append(":::")
                out_lines.append("")
                in_whole_new_section = False
                current_new_section = None

            # Is THIS new heading the start of a whole-new section?
            is_new = any(is_subordinate(heading_id, ns) for ns in WHOLE_NEW_SECTIONS)
            if is_new:
                out_lines.append("")
                out_lines.append("::: {.new-block}")
                out_lines.append(line)  # The heading itself, inside the div
                in_whole_new_section = True
                current_new_section = heading_id
                continue
            else:
                out_lines.append(line)
                continue

        # Inside whole-new section: emit as-is (no inline marking needed; everything is new)
        if in_whole_new_section:
            out_lines.append(line)
            continue

        # Outside new-block: do inline keyword marking (unless inside code)
        if in_code_fence:
            out_lines.append(line)
        else:
            out_lines.append(mark_inline(line))

    # Close trailing new-block if file ends inside one
    if in_whole_new_section:
        out_lines.append(":::")

    REDLINE_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {REDLINE_MD}")
    new_blocks = sum(1 for l in out_lines if l.strip() == "::: {.new-block}")
    inline_marks = sum(l.count("{.new-inline}") for l in out_lines)
    print(f"  new-block sections opened: {new_blocks}")
    print(f"  inline marks: {inline_marks}")


if __name__ == "__main__":
    main()
