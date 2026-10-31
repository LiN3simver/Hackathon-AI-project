"""Private operator feedback for a separately managed answer-evaluation workflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.history import HistoryFeedback, HistoryFeedbackInput
from app.services.activity_audit import install_activity_audit
from app.services.history_regressions import record_feedback

_SCHEMA = """CREATE TABLE IF NOT EXISTS history_feedback (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL,
    expected TEXT NOT NULL,
    intent TEXT,
    generated_by TEXT,
    context_json TEXT,
    warnings_json TEXT NOT NULL,
    cited_event_ids_json TEXT NOT NULL
)"""


def _ensure(connection) -> None:
    connection.execute(_SCHEMA)
    install_activity_audit(connection)


def save_feedback(store, request: HistoryFeedbackInput, actor: str) -> HistoryFeedback:
    item = HistoryFeedback(**request.model_dump(), id=f"FB-{uuid4().hex[:12].upper()}",
                           created_at=datetime.now(UTC), created_by=actor)
    with store._connect() as connection:
        _ensure(connection)
        connection.execute(
            """INSERT INTO history_feedback (id, created_at, created_by, question, answer, rating, comment,
            expected, intent, generated_by, context_json, warnings_json, cited_event_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item.id, item.created_at.isoformat(), actor, item.question, item.answer, item.rating, item.comment,
             item.expected, item.intent, item.generated_by,
             item.context.model_dump_json() if item.context else None,
             json.dumps(item.warnings, ensure_ascii=False), json.dumps(item.cited_event_ids)),
        )
        record_feedback(connection, item.answer_id, item.rating, item.expected, item.comment)
    return item


def list_feedback(store, *, rating: int | None = None, limit: int = 200) -> list[HistoryFeedback]:
    with store._connect() as connection:
        _ensure(connection)
        query = "SELECT * FROM history_feedback"
        params: list[object] = []
        if rating is not None:
            query += " WHERE rating = ?"
            params.append(rating)
        rows = connection.execute(query + " ORDER BY created_at DESC LIMIT ?", (*params, limit)).fetchall()
    return [HistoryFeedback(
        id=row["id"], created_at=datetime.fromisoformat(row["created_at"]), created_by=row["created_by"],
        question=row["question"], answer=row["answer"], rating=row["rating"], comment=row["comment"],
        expected=row["expected"], intent=row["intent"], generated_by=row["generated_by"],
        context=json.loads(row["context_json"]) if row["context_json"] else None,
        warnings=json.loads(row["warnings_json"]), cited_event_ids=json.loads(row["cited_event_ids_json"]),
    ) for row in rows]


def export_golden(store) -> str:
    """JSON Lines in the golden-set format; 👍 answers pin their intent, 👎 answers carry the reviewer's note."""
    lines = []
    for item in reversed(list_feedback(store, limit=5000)):
        expect = {"intent": item.intent} if item.rating == 1 and item.intent else {}
        lines.append(json.dumps({
            "id": item.id, "source": "feedback", "rating": item.rating,
            "turns": [{"question": item.question,
                       "context": item.context.model_dump(mode="json") if item.context else None,
                       "expect": expect}],
            "previous_answer": item.answer, "expected_note": item.expected, "comment": item.comment,
        }, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")
