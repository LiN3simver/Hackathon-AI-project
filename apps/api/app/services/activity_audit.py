"""Transactional row history, including workers and maintenance scripts.

Snapshots are labelled as recovered current state, never as invented past actions.
SQL triggers use no Python functions, so writes from other SQLite clients work too.
"""

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

TABLES = {
    "alerts": ("alert", "id"),
    "work_orders": ("work_order", "id"),
    "notification_outbox": ("notification", "id"),
    "damage_cases": ("damage_case", "id"),
    "damage_pair_recognition_runs": ("damage_recognition", "id"),
    "learning_samples": ("learning_sample_data", "id"),
    "training_runs": ("training_run", "id"),
    "inspection_sessions": ("inspection_session", "id"),
    "inspection_photos": ("inspection_photo", "id"),
    "inspection_analysis_jobs": ("analysis_job", "id"),
    "rental_events": ("rental_event", "id"),
    "automation_decisions": ("automation_decision", "id"),
    "history_blacklist": ("history_blacklist", "user_id"),
    "auth_accounts": ("account", "subject"),
    "runtime_settings": ("runtime_setting", "key"),
    "reference_guides": ("reference_guide", "view"),
    "history_feedback": ("history_feedback", "id"),
}
PRIVATE_ENTITY_TYPES = (
    "learning_sample",
    "learning_sample_data",
    "training_run",
    "account",
    "runtime_setting",
    "reference_guide",
    "admin_request",
)
_SECRET_COLUMNS = {"password_hash", "demo_password", "password", "access_token"}


def import_legacy_http_logs(connection: sqlite3.Connection, directory: Path) -> int:
    """Recover only requests actually evidenced in the old rotating API logs, once."""
    migration = "legacy_api_logs_v1"
    if connection.execute("SELECT 1 FROM activity_audit_migrations WHERE table_name = ?", (migration,)).fetchone():
        return 0
    pattern = re.compile(
        r"^(\S+) .*?\b[\w-]+\.api request(_failed)? method=(\S+) path=(\S+)"
        r"(?: status=(\d+))?(?: elapsed_ms=([\d.]+))?"
    )
    count = 0
    for path in sorted(directory.glob("api.log*")):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as source:
            for number, line in enumerate(source, start=1):
                match = pattern.search(line)
                if not match:
                    continue
                timestamp, failure, method, url_path, code, elapsed = match.groups()
                try:
                    created_at = datetime.fromisoformat(timestamp).isoformat()
                except ValueError:
                    continue
                url_path = url_path.split("?", 1)[0]
                status = int(code) if code else 500
                request_match = re.search(r"\brequest_id=([a-f0-9]{32})\b", line)
                if request_match:
                    request_id = request_match[1]
                    if connection.execute(
                        """SELECT 1 FROM operations_activity WHERE entity_id = ? AND action = 'request'
                        UNION ALL SELECT 1 FROM operations_activity_archive
                        WHERE entity_id = ? AND action = 'request' LIMIT 1""",
                        (request_id, request_id),
                    ).fetchone():
                        continue
                    record_id = "LOG-" + request_id
                else:
                    record_id = "LOG-" + hashlib.sha256(f"{path.name}:{number}:{line}".encode()).hexdigest()
                metadata = {
                    "source": "legacy_api_log",
                    "file": path.name,
                    "line": number,
                    "method": method,
                    "path": url_path,
                    "status_code": status,
                    "elapsed_ms": float(elapsed) if elapsed else None,
                    "outcome": "failed" if failure or status >= 400 else "success",
                    "notice": "由既有 API 日誌還原；舊日誌未記錄操作人或資料前後內容",
                }
                restricted = url_path.startswith(("/api/v1/admin/", "/api/v1/learning/"))
                result = connection.execute(
                    """INSERT OR IGNORE INTO operations_activity
                    (id, entity_type, entity_id, action, actor, summary, metadata_json, created_at)
                    VALUES (?, ?, ?, 'historical_request', 'unknown:legacy_log', ?, ?, ?)""",
                    (
                        record_id,
                        "admin_request" if restricted else "api_request",
                        record_id,
                        f"歷史請求：{method} {url_path} → {status}",
                        json.dumps(metadata, ensure_ascii=False),
                        created_at,
                    ),
                )
                count += result.rowcount
    connection.execute("INSERT INTO activity_audit_migrations (table_name) VALUES (?)", (migration,))
    return count


def install_activity_audit(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS operations_activity (
        id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
        action TEXT NOT NULL, actor TEXT NOT NULL, summary TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)""")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_operations_activity_created ON operations_activity(created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_order ON operations_activity(julianday(created_at) DESC, id DESC)"
    )
    connection.execute("""CREATE TABLE IF NOT EXISTS operations_activity_archive (
        id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
        action TEXT NOT NULL, actor TEXT NOT NULL, summary TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)""")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_archive_created ON operations_activity_archive(created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_archive_order "
        "ON operations_activity_archive(julianday(created_at) DESC, id DESC)"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS activity_audit_migrations (table_name TEXT PRIMARY KEY)")
    available = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table, (entity_type, key) in TABLES.items():
        if table not in available:
            continue
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
        safe = [column for column in columns if column not in _SECRET_COLUMNS]

        def snapshot(prefix: str, safe=safe) -> str:
            values = ", ".join(f"'{column}', {prefix}.\"{column}\"" for column in safe)
            return f"json_object({values})"

        now = "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
        for action, source, before, after, label in (
            ("INSERT", "NEW", "NULL", snapshot("NEW"), "新增資料"),
            ("UPDATE", "NEW", snapshot("OLD"), snapshot("NEW"), "更新資料"),
            ("DELETE", "OLD", snapshot("OLD"), "NULL", "刪除資料"),
        ):
            # Existing triggers survive restarts; migrations must not cause false updates.
            changed = ""
            if action == "UPDATE":
                changed = " WHEN " + " OR ".join(f'OLD."{c}" IS NOT NEW."{c}"' for c in columns)
            connection.execute(f"""CREATE TRIGGER IF NOT EXISTS audit_{table}_{action.lower()}
                AFTER {action} ON "{table}"{changed}
                BEGIN
                    INSERT INTO operations_activity
                    (id, entity_type, entity_id, action, actor, summary, metadata_json, created_at)
                    VALUES ('DB-' || hex(randomblob(16)), '{entity_type}',
                        CAST({source}."{key}" AS TEXT), 'data_{action.lower()}',
                        'system:database', '{label}：{table}',
                        json_object('source', 'database_trigger', 'table', '{table}',
                                    'before', {before}, 'after', {after}), {now});
                END""")
        if connection.execute("SELECT 1 FROM activity_audit_migrations WHERE table_name = ?", (table,)).fetchone():
            continue
        timestamp = next((c for c in ("created_at", "updated_at") if c in columns), None)
        recorded_at = f'COALESCE(row."{timestamp}", {now})' if timestamp else now
        connection.execute(f"""INSERT INTO operations_activity
            (id, entity_type, entity_id, action, actor, summary, metadata_json, created_at)
            SELECT 'SNAP-' || hex(randomblob(16)), '{entity_type}', CAST(row."{key}" AS TEXT),
                'historical_snapshot', 'system:history_backfill', '歷史資料快照：{table}',
                json_object('source', 'historical_snapshot', 'table', '{table}',
                    'notice', '現存資料快照，無法還原未保存的歷次操作',
                    'recovered_at', {now}, 'after', {snapshot("row")}), {recorded_at}
            FROM "{table}" AS row""")
        connection.execute("INSERT INTO activity_audit_migrations (table_name) VALUES (?)", (table,))
