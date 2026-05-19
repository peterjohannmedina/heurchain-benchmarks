"""
LongMemEval dataset loader.

Field mapping in the JSON:
  haystack_session_ids  — list of session ID strings
  haystack_sessions     — list of sessions; each session is a list of turn dicts
                          {role, content, has_answer}
  answer_session_ids    — ground-truth relevant session IDs (subset of haystack_session_ids)
  question              — eval query string
  question_type         — temporal-reasoning | multi-session | ...
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "LongMemEval" / "data"

_SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def load_split(split: str = "oracle") -> list[dict]:
    fname = _SPLIT_FILES.get(split)
    if not fname:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(_SPLIT_FILES)}")
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def session_to_text(turns: list[dict]) -> str:
    """Flatten a session's turns into a single text blob."""
    parts = []
    for t in turns:
        role = t.get("role", "")
        content = t.get("content", "")
        if content:
            parts.append(f"{role}: {content}" if role else content)
    return " ".join(parts)


def iter_retrieval_tasks(instances: list[dict]) -> list[dict]:
    """
    Convert raw LongMemEval instances into retrieval tasks.

    Each task:
      question_id         — unique ID
      query               — evaluation question
      answer              — ground truth answer (for future LLM-judge phase)
      relevant_session_ids — set of session IDs that contain the answer
      sessions            — list of {"session_id": str, "turns": list[dict]}
      question_type       — question category
    """
    tasks = []
    for inst in instances:
        session_ids: list[str] = inst.get("haystack_session_ids", [])
        raw_sessions: list[list[dict]] = inst.get("haystack_sessions", [])
        answer_ids: list[str] = inst.get("answer_session_ids", [])

        sessions = [
            {"session_id": sid, "turns": turns}
            for sid, turns in zip(session_ids, raw_sessions)
        ]

        tasks.append({
            "question_id": inst["question_id"],
            "query": inst["question"],
            "answer": inst.get("answer", ""),
            "relevant_session_ids": set(answer_ids),
            "sessions": sessions,
            "question_type": inst.get("question_type", "unknown"),
        })
    return tasks
