#!/usr/bin/env python3
"""
v4c event-index builder — extracts (event_phrase, iso_date, session_id, fact_id)
tuples from v4 facts that carry [date: YYYY-MM-DD] tags. Side-store for
temporal-reasoning queries; complements (does not replace) passage retrieval.

Design philosophy: graph-LITE, not full graph. No new database; SQLite
in-process. No new LLM calls at index time — purely parses the v4 facts
the extractor already produced. Cheap to maintain, cheap to query.

Builds:
  results/event_index_v4.sqlite

Schema:
  events (id, event_phrase, iso_date, session_id, fact_id, fact_text, category)
  indexes on iso_date and event_phrase tokens

Usage:
  # Build event index from all v4 fact JSONs:
  python3 build_event_index.py \
    --inputs 'results/facts_v3_*_max30.json' \
    --output results/event_index_v4.sqlite

  # Query the index:
  python3 build_event_index.py --query "MoMA visit" --db results/event_index_v4.sqlite
  python3 build_event_index.py --between "MoMA" "Met exhibit" --db results/event_index_v4.sqlite
"""
import argparse
import glob
import json
import re
import sqlite3
from pathlib import Path

ISO_RE = re.compile(r"\[date:\s*(\d{4}-\d{2}-\d{2})\]")
SESSION_RE = re.compile(r"\[Session\s+(\d+)\]")
# Strip [Session N] and [date: ...] from fact text to extract the event phrase
TAG_STRIP_RE = re.compile(r"\[(?:Session\s+\d+|date:\s*\d{4}-\d{2}-\d{2})\]")


def parse_event(fact_text):
    """Return (event_phrase, iso_date) or (None, None) if fact has no ISO date."""
    m = ISO_RE.search(fact_text)
    if not m:
        return None, None
    iso = m.group(1)
    # Strip the tags from fact text to get the event phrase
    phrase = TAG_STRIP_RE.sub("", fact_text).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase, iso


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS events;
        CREATE TABLE events (
          id           INTEGER PRIMARY KEY AUTOINCREMENT,
          event_phrase TEXT NOT NULL,
          iso_date     TEXT NOT NULL,
          session_id   INTEGER,
          fact_id      TEXT,
          category     TEXT,
          source_file  TEXT
        );
        CREATE INDEX idx_events_date ON events(iso_date);
        CREATE INDEX idx_events_session ON events(session_id);
        CREATE VIRTUAL TABLE events_fts USING fts5(
          event_phrase, iso_date, content='events', content_rowid='id'
        );
    """)
    return conn


def build(inputs, output):
    db_path = Path(output).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(str(db_path))
    cur = conn.cursor()

    paths = []
    for pat in inputs:
        paths.extend(sorted(glob.glob(str(Path(pat).expanduser()))))
    if not paths:
        print("No input files found.")
        return

    total_events = 0
    total_facts_scanned = 0
    files_processed = 0
    for path in paths:
        try:
            d = json.loads(Path(path).read_text())
        except Exception as e:
            print(f"  WARN skip {path}: {e}")
            continue
        category = d.get("question_type", "?")
        records = d.get("records", [])
        for r in records:
            # Use retrieved_facts (also handles targeted_reextract output's new_facts)
            facts = r.get("retrieved_facts") or r.get("new_facts") or []
            for i, f in enumerate(facts):
                total_facts_scanned += 1
                phrase, iso = parse_event(f)
                if not phrase or not iso:
                    continue
                session_match = SESSION_RE.search(f)
                session_id = int(session_match.group(1)) if session_match else None
                fact_id = f"{r.get('question_id','?')}:{i}"
                cur.execute(
                    "INSERT INTO events (event_phrase, iso_date, session_id, fact_id, category, source_file)"
                    " VALUES (?,?,?,?,?,?)",
                    (phrase, iso, session_id, fact_id, category, Path(path).name),
                )
                total_events += 1
        files_processed += 1

    # Populate FTS5 from base table
    cur.execute("INSERT INTO events_fts (rowid, event_phrase, iso_date) "
                "SELECT id, event_phrase, iso_date FROM events")
    conn.commit()
    conn.close()
    print(f"Built {db_path}")
    print(f"  Files processed:   {files_processed}")
    print(f"  Facts scanned:     {total_facts_scanned}")
    print(f"  Events indexed:    {total_events}")
    print(f"  Events/fact ratio: {total_events/max(1,total_facts_scanned)*100:.1f}%")


def query_event(db_path, query, limit=10):
    """FTS search the event_phrase. Returns list of (phrase, iso_date, session_id, fact_id)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Use FTS5 with prefix matching for partial words
    safe_query = " OR ".join(f"{w}*" for w in re.findall(r"\b\w+\b", query) if len(w) > 2)
    if not safe_query:
        safe_query = query
    cur.execute(
        f"SELECT e.event_phrase, e.iso_date, e.session_id, e.fact_id "
        f"FROM events_fts f JOIN events e ON e.id = f.rowid "
        f"WHERE events_fts MATCH ? ORDER BY rank LIMIT ?",
        (safe_query, limit),
    )
    return cur.fetchall()


def query_between(db_path, event_a, event_b, top_per_event=3):
    """Find dates for two events, compute interval. Used by 'between A and B' queries."""
    from datetime import date as _date
    results_a = query_event(db_path, event_a, limit=top_per_event)
    results_b = query_event(db_path, event_b, limit=top_per_event)
    if not results_a or not results_b:
        return {"event_a_hits": results_a, "event_b_hits": results_b, "interval_days": None}

    def best(hits):
        # Pick first (highest FTS rank)
        return hits[0]
    pa = best(results_a); pb = best(results_b)
    try:
        da = _date.fromisoformat(pa[1]); db_ = _date.fromisoformat(pb[1])
        delta = (db_ - da).days
    except Exception:
        delta = None
    return {
        "event_a_hits": results_a,
        "event_b_hits": results_b,
        "interval_days": delta,
        "best_a_date": pa[1] if pa else None,
        "best_b_date": pb[1] if pb else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", help="result JSONs (globs OK)")
    p.add_argument("--output", default="results/event_index_v4.sqlite",
                   help="output SQLite path")
    p.add_argument("--query", help="FTS search the event_phrase")
    p.add_argument("--between", nargs=2, metavar=("EVENT_A", "EVENT_B"),
                   help="find dates for both events + compute interval")
    p.add_argument("--db", default="results/event_index_v4.sqlite",
                   help="path to existing SQLite (for --query / --between)")
    args = p.parse_args()

    if args.inputs:
        build(args.inputs, args.output)
    elif args.query:
        results = query_event(args.db, args.query)
        for phrase, iso, sess, fid in results:
            print(f"  {iso}  [sess {sess}]  {phrase[:120]}")
    elif args.between:
        result = query_between(args.db, args.between[0], args.between[1])
        print(f"Event A ({args.between[0]}):")
        for hit in result["event_a_hits"]:
            print(f"  {hit[1]}  [sess {hit[2]}]  {hit[0][:100]}")
        print(f"Event B ({args.between[1]}):")
        for hit in result["event_b_hits"]:
            print(f"  {hit[1]}  [sess {hit[2]}]  {hit[0][:100]}")
        print(f"\nInterval (days): {result['interval_days']}")
    else:
        p.error("provide --inputs (to build) or --query / --between (to query)")


if __name__ == "__main__":
    main()
