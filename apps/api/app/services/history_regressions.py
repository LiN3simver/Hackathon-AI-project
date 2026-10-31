"""Keep private evidence snapshots for review of incorrect model summaries.

Two sources feed the regression set, each saved with the exact evidence the model saw:

- ``verifier``: a sentence the verifier refused while answering.
- ``feedback``: an AI summary an operator marked 👎.

Cases can be exported through the authenticated history API for a separately
managed evaluation workflow. They contain operational evidence and must remain
outside the public source repository.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from app.services.history_llm import EvidencePack

logger = logging.getLogger("api")
SNAPSHOT_LIMIT = 2000

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS history_regressions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        source TEXT NOT NULL,
        question TEXT NOT NULL,
        sentences_json TEXT NOT NULL,
        reason TEXT,
        pack_json TEXT NOT NULL,
        context_text TEXT NOT NULL,
        expected TEXT NOT NULL DEFAULT '',
        comment TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS history_answer_snapshots (
        answer_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        question TEXT NOT NULL,
        sentences_json TEXT NOT NULL,
        pack_json TEXT NOT NULL,
        context_text TEXT NOT NULL
    )""",
)


def pack_to_dict(pack: EvidencePack) -> dict:
    return {key: sorted(value) if isinstance(value, set) else value for key, value in vars(pack).items()}


def pack_from_dict(data: dict) -> EvidencePack:
    sets = {"names", "confirmed_names", "plates", "known_names", "forbidden", "facts", "periods"}
    return EvidencePack(**{key: set(value) if key in sets else value for key, value in data.items()})


def context_text(messages: list[dict]) -> str:
    """What SentenceVerifier is built from: every message after the system prompt."""
    return "\n".join(message["content"] for message in messages[1:])


def _ensure(connection) -> None:
    for statement in _SCHEMA:
        connection.execute(statement)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _save_case(connection, *, source: str, question: str, sentences: list[str], reason: str | None,
               pack_json: str, context: str, expected: str = "", comment: str = "") -> str:
    # The same sentence on the same evidence is one case, however often the model repeats it.
    digest = hashlib.sha256("\n".join([source, *sentences, pack_json]).encode()).hexdigest()[:16]
    case_id = f"REG-{digest.upper()}"
    connection.execute(
        """INSERT INTO history_regressions (id, created_at, source, question, sentences_json, reason, pack_json,
        context_text, expected, comment) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET expected = excluded.expected, comment = excluded.comment""",
        (case_id, _now(), source, question, json.dumps(sentences, ensure_ascii=False), reason, pack_json,
         context, expected, comment))
    return case_id


def remember_summary(store, question: str, pack: EvidencePack, messages: list[dict], sentences: list[str],
                     rejected: str | None, reason: str | None) -> str | None:
    """Keep the evidence behind a summary (for a later 👎) and record a refused sentence. Never raises."""
    try:
        pack_json = json.dumps(pack_to_dict(pack), ensure_ascii=False)
        context = context_text(messages)
        answer_id = f"ANS-{uuid4().hex[:16].upper()}" if sentences else None
        with store._connect() as connection:
            _ensure(connection)
            if rejected:
                _save_case(connection, source="verifier", question=question, sentences=[rejected], reason=reason,
                           pack_json=pack_json, context=context)
            if answer_id:
                connection.execute(
                    "INSERT INTO history_answer_snapshots VALUES (?, ?, ?, ?, ?, ?)",
                    (answer_id, _now(), question, json.dumps(sentences, ensure_ascii=False), pack_json, context))
                connection.execute(
                    """DELETE FROM history_answer_snapshots WHERE answer_id NOT IN (
                    SELECT answer_id FROM history_answer_snapshots ORDER BY created_at DESC LIMIT ?)""",
                    (SNAPSHOT_LIMIT,))
        return answer_id
    except Exception:  # bookkeeping must never cost the operator their answer
        logger.exception("history_regression_store_failed")
        return None


def record_feedback(connection, answer_id: str | None, rating: int, expected: str, comment: str) -> str | None:
    """A 👎 on an AI summary becomes a regression case with the evidence that summary was written from."""
    if rating != -1 or not answer_id:
        return None
    _ensure(connection)
    row = connection.execute("SELECT * FROM history_answer_snapshots WHERE answer_id = ?", (answer_id,)).fetchone()
    if row is None:
        return None
    return _save_case(connection, source="feedback", question=row["question"],
                      sentences=json.loads(row["sentences_json"]), reason=None, pack_json=row["pack_json"],
                      context=row["context_text"], expected=expected, comment=comment)


def list_cases(store) -> list[dict]:
    with store._connect() as connection:
        _ensure(connection)
        rows = connection.execute("SELECT * FROM history_regressions ORDER BY created_at, id").fetchall()
    return [{"id": row["id"], "source": row["source"], "question": row["question"],
             "sentences": json.loads(row["sentences_json"]), "reason": row["reason"],
             "pack": json.loads(row["pack_json"]), "context_text": row["context_text"],
             "expected": row["expected"], "comment": row["comment"], "accepted": False} for row in rows]


def export_cases(store) -> str:
    lines = [json.dumps(case, ensure_ascii=False) for case in list_cases(store)]
    return "\n".join(lines) + ("\n" if lines else "")
