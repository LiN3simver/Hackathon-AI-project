import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import RLock
from uuid import uuid4

from PIL import Image, ImageOps, ImageStat

from app.core.config import POWER_AUTOMATE_PROVIDER, get_settings
from app.schemas.cleanliness import CleanlinessAssessment
from app.schemas.damage import (
    DamageCase,
    DamageComparisonResult,
    DamageComparisonStatus,
    DamageObservation,
    DamageReviewDecision,
)
from app.schemas.history import (
    AutomationDecision,
    AutomationSummary,
    RentalEvent,
    UserHistoryListItem,
    UserHistoryProfile,
)
from app.schemas.inspection import (
    InspectionAnalysisJob,
    InspectionAnalysisJobStatus,
    InspectionAnalysisStatus,
    InspectionAnalysisTask,
    InspectionPhoto,
    InspectionPhotoStatus,
    InspectionSession,
    InspectionSessionStatus,
    InspectionStage,
)
from app.schemas.operations import (
    Alert,
    AlertStatus,
    CleanlinessReviewDecision,
    InspectionPhotoRecord,
    LearningRentalSummary,
    LearningSample,
    LearningStatus,
    LearningTask,
    NotificationDeliveryStatus,
    NotificationOutboxItem,
    OperationsActivity,
    RecognitionJobProgress,
    RecognitionProgress,
    TrainingRun,
    TrainingRunStatus,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderStep,
)
from app.schemas.plate import PlatePrediction
from app.schemas.quality import ExpectedView, QualityStatus
from app.services.activity_audit import PRIVATE_ENTITY_TYPES, install_activity_audit
from app.services.runtime_settings import get_runtime_recognition_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


WORK_ORDER_BUSINESS_DAYS = 5
_TAIPEI = timezone(timedelta(hours=8))


def _add_business_days(start: datetime, days: int) -> datetime:
    """Count Monday to Friday in Taipei time; the deadline keeps the same time of day."""
    current = start.astimezone(_TAIPEI)
    remaining = days
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current.astimezone(timezone.utc)


_VIEW_LABELS = {"left_front": "左前方", "right_front": "右前方", "left_rear": "左後方",
                "right_rear": "右後方"}


def _seat_from_title(title: str) -> str:
    return next((seat for seat in ("後座", "前座") if seat in title), "車內")


PICKUP_REQUIRED_SLOTS = [
    ExpectedView.LEFT_FRONT,
    ExpectedView.RIGHT_FRONT,
    ExpectedView.LEFT_REAR,
    ExpectedView.RIGHT_REAR,
]
RETURN_REQUIRED_SLOTS = [
    *PICKUP_REQUIRED_SLOTS,
    ExpectedView.INTERIOR_FRONT,
    ExpectedView.INTERIOR_REAR,
]
MAX_DAMAGE_DETAIL_PHOTOS = 4
# A formal-flow photo reaches the back office only once its own capture is
# finished and the rental's return capture has been completed.
_RELEASED_TO_OPS_SQL = """sessions.status = 'capture_completed' AND EXISTS (
    SELECT 1 FROM inspection_sessions AS returns
    WHERE returns.rental_id = sessions.rental_id
      AND returns.vehicle_id = sessions.vehicle_id
      AND returns.stage = 'return' AND returns.status = 'capture_completed')"""
OPERATIONS_ACTIVITY_RETENTION_DAYS = 180
DAMAGE_PAIR_TRAINING_LABELS = ("confirmed_new_damage", "no_new_damage")
DAMAGE_REVIEW_LABELS = {
    DamageReviewDecision.CONFIRMED_DAMAGE: "confirmed_new_damage",
    DamageReviewDecision.DISMISSED: "no_new_damage",
    DamageReviewDecision.RETRY_PHOTOS: None,
}


class IncompleteInspectionError(ValueError):
    def __init__(self, missing_slots: list[ExpectedView]) -> None:
        self.missing_slots = missing_slots
        super().__init__("仍有必要拍攝槽位未完成")


class ReturnInspectionNotReadyError(ValueError):
    def __init__(self, message: str = "請先完成並送出本租單的借車照片，才能建立還車拍攝流程") -> None:
        super().__init__(message)


class AnalysisQueueFullError(RuntimeError):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        super().__init__(f"AI 分析佇列已滿（上限 {capacity} 件），照片已保存但尚未完成還車，請稍後再按完成")


class OperationsStore:
    def __init__(
        self,
        database_path: Path | None = None,
        learning_storage_path: Path | None = None,
        plate_check_enabled: bool | None = None,
        analysis_queue_capacity: int | None = None,
    ) -> None:
        settings = get_settings()
        self.database_path = database_path or settings.database_path
        if learning_storage_path is not None:
            self.learning_storage_path = learning_storage_path
        elif database_path is not None:
            self.learning_storage_path = database_path.parent / "active_learning_images"
        else:
            self.learning_storage_path = settings.active_learning_storage_path
        self.plate_check_enabled = plate_check_enabled
        self.analysis_queue_capacity = (
            analysis_queue_capacity
            if analysis_queue_capacity is not None
            else settings.inspection_analysis_queue_capacity
        )

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    cancelled_at TEXT,
                    review_decision TEXT,
                    reviewer_notes TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS work_orders (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL REFERENCES alerts(id),
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assignee TEXT,
                    vehicle_id TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    due_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_work_order_status
                    ON work_orders(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS operations_activity (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operations_activity_created
                    ON operations_activity(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_operations_activity_entity
                    ON operations_activity(entity_type, entity_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS rental_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    stage TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    outcome TEXT,
                    severity TEXT,
                    confidence REAL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    disputed INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_type, source_id, event_type)
                );
                CREATE INDEX IF NOT EXISTS idx_rental_events_user
                    ON rental_events(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_rental_events_rental
                    ON rental_events(rental_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_rental_events_type
                    ON rental_events(event_type, verified, created_at DESC);
                CREATE TABLE IF NOT EXISTS history_blacklist (
                    user_id TEXT PRIMARY KEY,
                    blacklisted INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_decisions (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES inspection_analysis_jobs(id)
                        ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    recommended_route TEXT NOT NULL,
                    effective_route TEXT NOT NULL,
                    shadow_mode INTEGER NOT NULL,
                    confidence REAL,
                    reason_codes_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_automation_decisions_route
                    ON automation_decisions(recommended_route, created_at DESC);
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    id TEXT PRIMARY KEY,
                    event_key TEXT NOT NULL UNIQUE,
                    channel TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    last_error TEXT,
                    next_attempt_at TEXT,
                    handed_off_at TEXT,
                    sent_at TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notification_outbox_delivery
                    ON notification_outbox(status, next_attempt_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_notification_outbox_source
                    ON notification_outbox(source_type, source_id);
                CREATE TABLE IF NOT EXISTS damage_cases (
                    id TEXT PRIMARY KEY,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    expected_view TEXT NOT NULL,
                    pickup_filename TEXT NOT NULL,
                    return_filename TEXT NOT NULL,
                    comparison_json TEXT NOT NULL,
                    review_decision TEXT NOT NULL,
                    reviewer_notes TEXT NOT NULL,
                    alert_id TEXT REFERENCES alerts(id),
                    learning_sample_id TEXT REFERENCES learning_samples(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_damage_case_review
                    ON damage_cases(review_decision, created_at DESC);
                CREATE TABLE IF NOT EXISTS damage_pair_recognition_runs (
                    id TEXT PRIMARY KEY,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_damage_pair_recognition_status
                    ON damage_pair_recognition_runs(status, started_at DESC);
                CREATE TABLE IF NOT EXISTS learning_samples (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_reasons_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    primary_image_path TEXT NOT NULL,
                    secondary_image_path TEXT,
                    primary_image_hash TEXT,
                    secondary_image_hash TEXT,
                    primary_filename TEXT NOT NULL,
                    secondary_filename TEXT,
                    rental_id TEXT,
                    vehicle_id TEXT,
                    source_id TEXT,
                    expected_label TEXT,
                    predicted_label TEXT,
                    corrected_label TEXT,
                    confidence REAL,
                    score INTEGER,
                    lighting TEXT NOT NULL,
                    auto_category TEXT NOT NULL,
                    model_version TEXT,
                    sample_kind TEXT NOT NULL DEFAULT 'hard_case',
                    training_eligible INTEGER NOT NULL DEFAULT 0,
                    reference_enabled INTEGER NOT NULL DEFAULT 0,
                    pending_inbox_hidden INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    reviewer_notes TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    trained_at TEXT,
                    UNIQUE(task, content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_learning_review
                    ON learning_samples(status, task, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_learning_source
                    ON learning_samples(source_id, task);
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    approved_samples INTEGER NOT NULL,
                    manifest_path TEXT,
                    candidate_path TEXT,
                    report_path TEXT,
                    log_path TEXT,
                    baseline_metrics_json TEXT NOT NULL,
                    candidate_metrics_json TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_training_run_status
                    ON training_runs(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS inspection_sessions (
                    id TEXT PRIMARY KEY,
                    rental_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required_slots_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    capture_completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_inspection_session_context
                    ON inspection_sessions(rental_id, stage, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_inspection_session_status
                    ON inspection_sessions(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS inspection_photos (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES inspection_sessions(id)
                        ON DELETE CASCADE,
                    slot TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    quality_request_id TEXT NOT NULL UNIQUE,
                    quality_status TEXT NOT NULL,
                    quality_score INTEGER NOT NULL,
                    capture_status TEXT NOT NULL,
                    analysis_status TEXT NOT NULL,
                    object_key TEXT,
                    content_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_inspection_photo_session
                    ON inspection_photos(session_id, slot, created_at);
                CREATE INDEX IF NOT EXISTS idx_inspection_photo_analysis
                    ON inspection_photos(analysis_status, created_at);
                CREATE TABLE IF NOT EXISTS inspection_analysis_jobs (
                    id TEXT PRIMARY KEY,
                    photo_id TEXT NOT NULL REFERENCES inspection_photos(id)
                        ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES inspection_sessions(id)
                        ON DELETE CASCADE,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    damage_case_id TEXT REFERENCES damage_cases(id) ON DELETE SET NULL,
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    next_attempt_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    UNIQUE(photo_id, task)
                );
                CREATE INDEX IF NOT EXISTS idx_inspection_analysis_queue
                    ON inspection_analysis_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_inspection_analysis_session
                    ON inspection_analysis_jobs(session_id, created_at);
                """
            )
            self._ensure_column(
                connection,
                "learning_samples",
                "sample_kind",
                "TEXT NOT NULL DEFAULT 'hard_case'",
            )
            self._ensure_column(
                connection,
                "learning_samples",
                "training_eligible",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "inspection_photos", "content_hash", "TEXT")
            self._ensure_column(connection, "inspection_photos", "plate_prediction_json", "TEXT")
            self._ensure_column(connection, "inspection_photos", "damage_view", "TEXT")
            self._ensure_column(
                connection, "learning_samples", "pending_inbox_hidden", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(connection, "learning_samples", "primary_image_hash", "TEXT")
            self._ensure_column(connection, "learning_samples", "secondary_image_hash", "TEXT")
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_learning_primary_image_hash
                ON learning_samples(primary_image_hash)"""
            )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_learning_secondary_image_hash
                ON learning_samples(secondary_image_hash)"""
            )
            # Migrate existing databases: repeated capture photos are allowed.
            connection.execute("DROP INDEX IF EXISTS idx_inspection_photo_content_hash")
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_inspection_photo_content_hash
                ON inspection_photos(content_hash)
                WHERE content_hash IS NOT NULL AND object_key IS NOT NULL"""
            )
            self._ensure_column(
                connection,
                "training_runs",
                "task",
                f"TEXT NOT NULL DEFAULT '{LearningTask.EXTERIOR_VIEW.value}'",
            )
            self._ensure_column(connection, "work_orders", "due_at", "TEXT")
            self._ensure_column(connection, "work_orders", "completed_at", "TEXT")
            # Open orders from the old two-hour rule move to the five-business-day deadline.
            for legacy in connection.execute(
                """SELECT id, created_at FROM work_orders
                WHERE status IN ('queued', 'assigned', 'in_progress')
                  AND due_at IS NOT NULL AND julianday(due_at) - julianday(created_at) < 1"""
            ).fetchall():
                connection.execute(
                    "UPDATE work_orders SET due_at = ? WHERE id = ?",
                    (
                        _add_business_days(
                            datetime.fromisoformat(legacy["created_at"]), WORK_ORDER_BUSINESS_DAYS
                        ).isoformat(),
                        legacy["id"],
                    ),
                )
            self._ensure_column(connection, "alerts", "cancelled_at", "TEXT")
            self._ensure_column(connection, "alerts", "image_url", "TEXT")
            self._ensure_column(connection, "alerts", "review_decision", "TEXT")
            self._ensure_column(
                connection, "alerts", "reviewer_notes", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(connection, "alerts", "reviewed_at", "TEXT")
            self._ensure_column(connection, "notification_outbox", "handed_off_at", "TEXT")
            self._ensure_column(
                connection,
                "damage_cases",
                "learning_sample_id",
                "TEXT REFERENCES learning_samples(id) ON DELETE SET NULL",
            )
            self._ensure_column(
                connection,
                "inspection_analysis_jobs",
                "next_attempt_at",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "inspection_analysis_jobs",
                "damage_case_id",
                "TEXT REFERENCES damage_cases(id) ON DELETE SET NULL",
            )
            self._ensure_column(
                connection,
                "inspection_analysis_jobs",
                "reviewed_at",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "inspection_analysis_jobs",
                "reviewed_by",
                "TEXT",
            )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_inspection_analysis_damage_case
                ON inspection_analysis_jobs(damage_case_id)"""
            )
            legacy_damage_jobs = connection.execute(
                """SELECT id, result_json FROM inspection_analysis_jobs
                WHERE task = ? AND damage_case_id IS NULL""",
                (InspectionAnalysisTask.DAMAGE_COMPARISON.value,),
            ).fetchall()
            for legacy_job in legacy_damage_jobs:
                try:
                    damage_case_id = json.loads(legacy_job["result_json"]).get("damage_case_id")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    damage_case_id = None
                if damage_case_id:
                    connection.execute(
                        """UPDATE inspection_analysis_jobs SET damage_case_id = ?
                        WHERE id = ?""",
                        (damage_case_id, legacy_job["id"]),
                    )
            stale_review_jobs = connection.execute(
                """SELECT jobs.*, cases.review_decision, cases.reviewer_notes,
                          cases.reviewed_at AS case_reviewed_at
                FROM inspection_analysis_jobs AS jobs
                JOIN damage_cases AS cases ON cases.id = jobs.damage_case_id
                WHERE jobs.task = ? AND jobs.status = ?
                  AND cases.review_decision != ?""",
                (
                    InspectionAnalysisTask.DAMAGE_COMPARISON.value,
                    InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                    DamageReviewDecision.PENDING.value,
                ),
            ).fetchall()
            for stale_job in stale_review_jobs:
                decision = DamageReviewDecision(stale_job["review_decision"])
                corrected_label = DAMAGE_REVIEW_LABELS.get(decision)
                sample = connection.execute(
                    """SELECT status, corrected_label FROM learning_samples
                    WHERE source_id = ? AND task = ?
                    ORDER BY created_at DESC LIMIT 1""",
                    (
                        stale_job["damage_case_id"],
                        LearningTask.DAMAGE_PAIR.value,
                    ),
                ).fetchone()
                if (
                    corrected_label is None
                    or sample is None
                    or sample["status"]
                    not in {
                        LearningStatus.APPROVED.value,
                        LearningStatus.TRAINED.value,
                    }
                    or sample["corrected_label"] != corrected_label
                ):
                    continue
                reviewer_activity = connection.execute(
                    """SELECT actor FROM operations_activity
                    WHERE entity_type = 'damage_case' AND entity_id = ?
                      AND action = 'reviewed'
                    ORDER BY created_at DESC LIMIT 1""",
                    (stale_job["damage_case_id"],),
                ).fetchone()
                reviewed_by = (
                    str(reviewer_activity["actor"])
                    if reviewer_activity is not None
                    else "legacy:unknown"
                )
                reviewed_at = stale_job["case_reviewed_at"] or stale_job["updated_at"]
                job_result = json.loads(stale_job["result_json"])
                job_result["human_review"] = {
                    "decision": decision.value,
                    "corrected_label": corrected_label,
                    "reviewer_notes": stale_job["reviewer_notes"],
                    "reviewed_by": reviewed_by,
                    "reviewed_at": reviewed_at,
                    "reconciled_from_legacy_state": True,
                }
                connection.execute(
                    """UPDATE inspection_analysis_jobs
                    SET status = ?, result_json = ?, updated_at = ?,
                        reviewed_at = ?, reviewed_by = ?
                    WHERE id = ? AND status = ?""",
                    (
                        InspectionAnalysisJobStatus.RESOLVED.value,
                        json.dumps(job_result, ensure_ascii=False),
                        reviewed_at,
                        reviewed_at,
                        reviewed_by,
                        stale_job["id"],
                        InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                    ),
                )
                self._refresh_photo_analysis_status(
                    connection, str(stale_job["photo_id"]), str(reviewed_at)
                )
                self._record_activity(
                    connection,
                    entity_type="analysis_job",
                    entity_id=stale_job["id"],
                    action="legacy_review_reconciled",
                    actor="system:migration",
                    summary="依已覆核車損案件與人工真值補齊 job 結案狀態",
                    metadata={
                        "damage_case_id": stale_job["damage_case_id"],
                        "reviewed_by": reviewed_by,
                        "corrected_label": corrected_label,
                    },
                )
            self._ensure_column(
                connection,
                "learning_samples",
                "reference_enabled",
                "INTEGER NOT NULL DEFAULT 0",
            )
            # Earlier retry decisions incorrectly approved unusable photos as human truth.
            connection.execute(
                """UPDATE learning_samples AS samples
                SET status = ?, corrected_label = NULL, training_eligible = 0,
                    reference_enabled = 0
                WHERE samples.task = ? AND samples.status = ?
                  AND samples.corrected_label = 'cannot_assess'
                  AND EXISTS (
                      SELECT 1 FROM damage_cases AS cases
                      WHERE cases.review_decision = ?
                        AND (cases.learning_sample_id = samples.id
                             OR cases.id = samples.source_id)
                  )""",
                (
                    LearningStatus.REJECTED.value,
                    LearningTask.DAMAGE_PAIR.value,
                    LearningStatus.APPROVED.value,
                    DamageReviewDecision.RETRY_PHOTOS.value,
                ),
            )
            connection.execute(
                """UPDATE inspection_photos SET analysis_status = ?
                WHERE object_key IS NULL AND analysis_status IN (?, ?)""",
                (
                    InspectionAnalysisStatus.NOT_REQUESTED.value,
                    InspectionAnalysisStatus.QUEUED.value,
                    InspectionAnalysisStatus.PROCESSING.value,
                ),
            )
            connection.execute(
                """UPDATE alerts
                SET image_url = (
                    SELECT '/api/v1/inspections/' || photos.session_id ||
                           '/photos/' || photos.id || '/image'
                    FROM inspection_photos AS photos
                    JOIN inspection_sessions AS sessions ON sessions.id = photos.session_id
                    WHERE photos.filename = alerts.image_filename
                      AND sessions.rental_id = alerts.rental_id
                      AND sessions.vehicle_id = alerts.vehicle_id
                    ORDER BY photos.created_at DESC
                    LIMIT 1
                )
                WHERE image_url IS NULL AND type = 'damage_observation'"""
            )
            self._backfill_rental_events(connection)
            self._reconcile_history_user_ids(connection)
            install_activity_audit(connection)

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _alert(row: sqlite3.Row) -> Alert:
        payload = dict(row)
        payload["evidence"] = json.loads(payload.pop("evidence_json"))
        return Alert.model_validate(payload)

    @staticmethod
    def _work_order(row: sqlite3.Row) -> WorkOrder:
        return WorkOrder.model_validate(dict(row))

    @staticmethod
    def _operations_activity(row: sqlite3.Row) -> OperationsActivity:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        recorded = datetime.fromisoformat(payload["created_at"])
        # SQLite CURRENT_TIMESTAMP is UTC but lacks an explicit timezone suffix.
        if recorded.tzinfo is None:
            recorded = recorded.replace(tzinfo=timezone.utc)
        payload["created_at"] = recorded.astimezone(timezone.utc).isoformat()
        return OperationsActivity.model_validate(payload)

    @staticmethod
    def _rental_event(row: sqlite3.Row) -> RentalEvent:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        if payload["metadata"].get("synthetic") and payload["title"].startswith("[假資料]"):
            payload["title"] = "[模擬資料]" + payload["title"][5:]
        payload["verified"] = bool(payload["verified"])
        payload["disputed"] = bool(payload["disputed"])
        return RentalEvent.model_validate(payload)

    @staticmethod
    def _automation_decision(row: sqlite3.Row) -> AutomationDecision:
        payload = dict(row)
        payload["reason_codes"] = json.loads(payload.pop("reason_codes_json"))
        payload["evidence"] = json.loads(payload.pop("evidence_json"))
        payload["shadow_mode"] = bool(payload["shadow_mode"])
        return AutomationDecision.model_validate(payload)

    @staticmethod
    def _resolve_user_id(connection: sqlite3.Connection, rental_id: str) -> str:
        table = connection.execute(
            """SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'auth_accounts'"""
        ).fetchone()
        if table is None:
            return rental_id
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(auth_accounts)").fetchall()
        }
        if "customer_id" not in columns:
            return rental_id
        row = connection.execute(
            """SELECT customer_id FROM auth_accounts
            WHERE kind = 'rental' AND rental_id = ?""",
            (rental_id,),
        ).fetchone()
        return str(row["customer_id"]) if row is not None and row["customer_id"] else rental_id

    @classmethod
    def _insert_rental_event(
        cls,
        connection: sqlite3.Connection,
        *,
        rental_id: str,
        vehicle_id: str,
        event_type: str,
        source_type: str,
        source_id: str,
        title: str,
        summary: str,
        session_id: str | None = None,
        stage: str | None = None,
        outcome: str | None = None,
        severity: str | None = None,
        confidence: float | None = None,
        verified: bool = False,
        disputed: bool = False,
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> None:
        connection.execute(
            """INSERT OR IGNORE INTO rental_events
            (id, user_id, rental_id, vehicle_id, session_id, event_type, stage,
             source_type, source_id, title, summary, outcome, severity, confidence,
             verified, disputed, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"EVT-{uuid4().hex[:12].upper()}",
                cls._resolve_user_id(connection, rental_id),
                rental_id,
                vehicle_id,
                session_id,
                event_type,
                stage,
                source_type,
                source_id,
                title,
                summary,
                outcome,
                severity,
                confidence,
                int(verified),
                int(disputed),
                json.dumps(metadata or {}, ensure_ascii=False),
                created_at or _now(),
            ),
        )

    @classmethod
    def _backfill_rental_events(cls, connection: sqlite3.Connection) -> None:
        connection.execute(
            """DELETE FROM rental_events AS legacy
            WHERE legacy.source_type = 'alert' AND legacy.event_type = 'alert_created'
              AND EXISTS (
                  SELECT 1 FROM rental_events AS specific
                  WHERE specific.source_type = 'alert'
                    AND specific.source_id = legacy.source_id
                    AND specific.event_type IN (
                        'cleanliness_alert_created', 'damage_alert_created',
                        'damage_observation_alert_created'
                    )
              )"""
        )
        for row in connection.execute("SELECT * FROM inspection_sessions").fetchall():
            cls._insert_rental_event(
                connection,
                rental_id=str(row["rental_id"]),
                vehicle_id=str(row["vehicle_id"]),
                session_id=str(row["id"]),
                event_type="inspection_started",
                stage=str(row["stage"]),
                source_type="inspection_session",
                source_id=str(row["id"]),
                title="建立借還車檢查流程",
                summary=f"{row['stage']} 階段檢查流程已建立。",
                outcome=str(row["status"]),
                created_at=str(row["created_at"]),
            )
            if row["capture_completed_at"]:
                cls._insert_rental_event(
                    connection,
                    rental_id=str(row["rental_id"]),
                    vehicle_id=str(row["vehicle_id"]),
                    session_id=str(row["id"]),
                    event_type="capture_completed",
                    stage=str(row["stage"]),
                    source_type="inspection_capture",
                    source_id=str(row["id"]),
                    title="完成借還車拍攝",
                    summary=f"{row['stage']} 階段必要照片已完成。",
                    outcome="completed",
                    verified=True,
                    created_at=str(row["capture_completed_at"]),
                )
        for row in connection.execute("SELECT * FROM damage_cases").fetchall():
            if row["review_decision"] == DamageReviewDecision.PENDING.value:
                continue
            confirmed = row["review_decision"] == DamageReviewDecision.CONFIRMED_DAMAGE.value
            cls._insert_rental_event(
                connection,
                rental_id=str(row["rental_id"]),
                vehicle_id=str(row["vehicle_id"]),
                event_type="damage_confirmed" if confirmed else "damage_reviewed",
                source_type="damage_case_review",
                source_id=str(row["id"]),
                title="人工確認新增車損" if confirmed else "車損案件已覆核",
                summary=str(row["reviewer_notes"] or row["review_decision"]),
                outcome=str(row["review_decision"]),
                severity="high" if confirmed else "none",
                verified=True,
                created_at=str(row["reviewed_at"] or row["created_at"]),
            )
        for row in connection.execute("SELECT * FROM alerts").fetchall():
            if connection.execute(
                """SELECT 1 FROM rental_events
                WHERE source_type = 'alert' AND source_id = ? LIMIT 1""",
                (row["id"],),
            ).fetchone():
                continue
            cls._insert_rental_event(
                connection,
                rental_id=str(row["rental_id"]),
                vehicle_id=str(row["vehicle_id"]),
                event_type="alert_created",
                source_type="alert",
                source_id=str(row["id"]),
                title=str(row["title"]),
                summary=str(row["summary"]),
                outcome=str(row["status"]),
                severity=str(row["severity"]),
                confidence=float(row["confidence"]),
                verified=False,
                metadata={"alert_type": row["type"]},
                created_at=str(row["created_at"]),
            )

    @classmethod
    def _reconcile_history_user_ids(cls, connection: sqlite3.Connection) -> None:
        if not cls._has_customer_id(connection):
            return
        for table in ("rental_events", "automation_decisions"):
            connection.execute(
                f"""UPDATE {table}
                SET user_id = (SELECT accounts.customer_id FROM auth_accounts AS accounts
                               WHERE accounts.kind = 'rental'
                                 AND accounts.rental_id = {table}.rental_id)
                WHERE EXISTS (SELECT 1 FROM auth_accounts AS accounts
                              WHERE accounts.kind = 'rental'
                                AND accounts.rental_id = {table}.rental_id
                                AND accounts.customer_id IS NOT NULL
                                AND accounts.customer_id != {table}.user_id)"""
            )

    @staticmethod
    def _notification_outbox(row: sqlite3.Row) -> NotificationOutboxItem:
        return NotificationOutboxItem.model_validate(dict(row))

    @staticmethod
    def _record_activity(
        connection: sqlite3.Connection,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str,
        summary: str,
        metadata: dict | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO operations_activity
            (id, entity_type, entity_id, action, actor, summary, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"ACT-{uuid4().hex[:10].upper()}",
                entity_type,
                entity_id,
                action,
                actor,
                summary,
                json.dumps(metadata or {}, ensure_ascii=False),
                _now(),
            ),
        )

    @staticmethod
    def _enqueue_alert_notification(
        connection: sqlite3.Connection,
        *,
        alert_id: str,
        alert_type: str,
        rental_id: str,
        vehicle_id: str,
        title: str,
        summary: str,
        confidence: float,
        created_at: str,
        body_text: str | None = None,
        event: str = "created",
        recipient: str | None = None,
    ) -> None:
        settings = get_settings()
        recipient = (recipient or settings.notification_email_recipient).strip()
        if not settings.notification_email_enabled or not recipient:
            return
        body_text = body_text or OperationsStore._staff_email_body(
            headline=f"{vehicle_id} 有狀況需要處理。",
            vehicle_id=vehicle_id,
            found_at=created_at,
            items_title="狀況：",
            items=[summary],
            action="請登入營運中心查看照片並安排處理。",
        )
        connection.execute(
            """INSERT OR IGNORE INTO notification_outbox
            (id, event_key, channel, provider, recipient, subject, body_text,
             status, attempts, max_attempts, next_attempt_at, source_type,
             source_id, created_at, updated_at)
            VALUES (?, ?, 'email', ?, ?, ?, ?, ?, 0, ?, ?, 'alert', ?, ?, ?)""",
            (
                f"NTF-{uuid4().hex[:10].upper()}",
                f"alert:{alert_id}:{event}",
                POWER_AUTOMATE_PROVIDER,
                recipient,
                f"[Guardian] {title}",
                body_text,
                NotificationDeliveryStatus.PENDING.value,
                settings.notification_max_attempts,
                created_at,
                alert_id,
                created_at,
                created_at,
            ),
        )

    @staticmethod
    def _cleaning_email_body(
        *, vehicle_id: str, seat: str, evidence: list[dict], found_at: str, confirmed: bool,
    ) -> str:
        """Plain words for cleaning staff: which car, which seats, where, what to do."""
        spots = [str(item.get("description", "")).strip() for item in evidence
                 if item.get("visible") and item.get("bounding_box") and item.get("description")]
        return OperationsStore._staff_email_body(
            headline=f"{vehicle_id} {seat}需要清潔。" if confirmed else f"{vehicle_id} {seat}可能有髒汙。",
            vehicle_id=vehicle_id,
            found_at=found_at,
            area=seat,
            items_title="髒汙位置：",
            items=spots or [f"請檢查{seat}座椅與腳踏處"],
            action=("工作人員已看過照片，確認有髒汙，請安排清潔。" if confirmed
                    else "請到營運中心「異常警示」查看照片，確認後按「有髒汙」或「沒髒汙」；"
                         "按「有髒汙」會通知清潔人員。"),
        )

    @staticmethod
    def _staff_email_body(
        *, headline: str, vehicle_id: str, found_at: str, items_title: str, items: list[str],
        action: str, area: str | None = None,
    ) -> str:
        """Email text for front-line staff: no IDs, scores or model names, just car, place and task."""
        found = datetime.fromisoformat(found_at).astimezone(timezone(timedelta(hours=8)))
        lines = [headline, "", f"車號：{vehicle_id}"]
        if area:
            lines.append(f"區域：{area}")
        lines += [f"發現時間：{found:%Y/%m/%d %H:%M}", "", items_title]
        lines += [f"・{item}" for item in items[:5] if item]
        lines += ["", action]
        return "\n".join(lines)

    @staticmethod
    def _damage_case(row: sqlite3.Row) -> DamageCase:
        payload = dict(row)
        payload["comparison"] = json.loads(payload.pop("comparison_json"))
        payload.pop("learning_sample_id", None)
        return DamageCase.model_validate(payload)

    @staticmethod
    def _learning_sample(row: sqlite3.Row) -> LearningSample:
        payload = dict(row)
        payload["trigger_reasons"] = json.loads(payload.pop("trigger_reasons_json"))
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        payload["has_secondary_image"] = bool(payload.pop("secondary_image_path"))
        payload.pop("primary_image_path")
        payload.pop("content_hash")
        return LearningSample.model_validate(payload)

    @staticmethod
    def _training_run(row: sqlite3.Row) -> TrainingRun:
        payload = dict(row)
        payload["baseline_metrics"] = json.loads(payload.pop("baseline_metrics_json"))
        payload["candidate_metrics"] = json.loads(payload.pop("candidate_metrics_json"))
        return TrainingRun.model_validate(payload)

    @staticmethod
    def _inspection_photo(row: sqlite3.Row) -> InspectionPhoto:
        payload = dict(row)
        payload["image_available"] = bool(payload.pop("object_key"))
        payload.pop("content_hash", None)
        return InspectionPhoto.model_validate(payload)

    @staticmethod
    def _inspection_analysis_job(row: sqlite3.Row) -> InspectionAnalysisJob:
        payload = dict(row)
        payload["result"] = json.loads(payload.pop("result_json"))
        payload["duration_ms"] = OperationsStore._job_duration_ms(row)
        return InspectionAnalysisJob.model_validate(payload)

    @staticmethod
    def _job_duration_ms(row: sqlite3.Row) -> float | None:
        result = json.loads(row["result_json"])
        if isinstance(result.get("duration_ms"), int | float):
            return max(0, float(result["duration_ms"])) + (
                max(0, float(result.get("active_duration_ms") or 0))
                if row["status"] == "processing" else 0
            )
        if row["started_at"] and row["completed_at"]:
            return max(0, (datetime.fromisoformat(row["completed_at"])
                           - datetime.fromisoformat(row["started_at"])).total_seconds() * 1000)
        captured = result.get("capture_prediction") or {}
        if isinstance(captured.get("latency_ms"), int | float):
            return max(0, float(captured["latency_ms"]))
        return None

    @staticmethod
    def _recognition_job_progress(row: sqlite3.Row) -> RecognitionJobProgress:
        return RecognitionJobProgress(
            **{key: row[key] for key in ("id", "photo_id", "task", "status", "filename", "slot",
                                         "damage_view", "started_at", "completed_at")},
            heartbeat_at=row["updated_at"] if row["status"] == "processing" else None,
            duration_ms=OperationsStore._job_duration_ms(row),
            model_pending=bool(json.loads(row["result_json"]).get("model_pending")),
        )

    @classmethod
    def _inspection_session(
        cls,
        row: sqlite3.Row,
        photo_rows: list[sqlite3.Row],
    ) -> InspectionSession:
        payload = dict(row)
        required_slots = [
            ExpectedView(slot) for slot in json.loads(payload.pop("required_slots_json"))
        ]
        photos = [cls._inspection_photo(photo_row) for photo_row in photo_rows]
        accepted_slots = {
            photo.damage_view or photo.slot
            for photo in photos
            if photo.capture_status == InspectionPhotoStatus.ACCEPTED
        }
        payload["required_slots"] = required_slots
        payload["completed_slots"] = [slot for slot in required_slots if slot in accepted_slots]
        payload["missing_slots"] = [slot for slot in required_slots if slot not in accepted_slots]
        payload["photos"] = photos
        return InspectionSession.model_validate(payload)

    @classmethod
    def _inspection_session_from_connection(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> InspectionSession:
        photo_rows = connection.execute(
            """SELECT * FROM inspection_photos
            WHERE session_id = ? ORDER BY created_at, id""",
            (row["id"],),
        ).fetchall()
        return cls._inspection_session(row, photo_rows)

    @staticmethod
    def _image_profile(content: bytes) -> tuple[str, str]:
        with Image.open(BytesIO(content)) as source:
            image_format = (source.format or "JPEG").upper()
            image = ImageOps.exif_transpose(source).convert("L")
            mean = float(ImageStat.Stat(image).mean[0])
            histogram = image.histogram()
            pixels = max(1, image.width * image.height)
            dark_ratio = sum(histogram[:48]) / pixels
            bright_ratio = sum(histogram[232:]) / pixels
        if mean < 68 or dark_ratio > 0.52:
            lighting = "night"
        elif mean < 105 or dark_ratio > 0.32:
            lighting = "low_light"
        elif mean > 215 or bright_ratio > 0.45:
            lighting = "overexposed"
        else:
            lighting = "daytime"
        suffix = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "BMP": ".bmp",
        }.get(image_format, ".img")
        return lighting, suffix

    def create_inspection_session(
        self,
        *,
        rental_id: str,
        vehicle_id: str,
        stage: InspectionStage,
        enforce_stage_order: bool = False,
        resume_existing: bool = False,
    ) -> InspectionSession:
        session_id = f"INS-{uuid4().hex[:10].upper()}"
        now = _now()
        required_slots = (
            PICKUP_REQUIRED_SLOTS if stage == InspectionStage.PICKUP else RETURN_REQUIRED_SLOTS
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._rental_inspection_state(connection, rental_id.strip(), vehicle_id.strip())
            if enforce_stage_order and stage == InspectionStage.RETURN:
                if state["pickup"] is None or state["pickup"]["status"] != "capture_completed":
                    raise ReturnInspectionNotReadyError()
                if state["completed_return"] is not None:
                    raise ReturnInspectionNotReadyError("本輪還車拍攝已完成，請重新開始借車流程")
            if resume_existing:
                existing = state["completed_return"] or state["active"]
                if existing is not None:
                    self._close_obsolete_inspections(connection, rental_id=rental_id.strip(), keep_id=existing["id"])
                    return self._inspection_session_from_connection(connection, existing)
                if (stage == InspectionStage.PICKUP and state["pickup"] is not None
                        and state["pickup"]["status"] == "capture_completed"):
                    stage = InspectionStage.RETURN
                    required_slots = RETURN_REQUIRED_SLOTS
            if enforce_stage_order:
                self._close_obsolete_inspections(connection, rental_id=rental_id.strip())
            connection.execute(
                """INSERT INTO inspection_sessions
                (id, rental_id, vehicle_id, stage, status, required_slots_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    rental_id.strip(),
                    vehicle_id.strip(),
                    stage.value,
                    InspectionSessionStatus.DRAFT.value,
                    json.dumps([slot.value for slot in required_slots]),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            assert row is not None
            self._insert_rental_event(
                connection,
                rental_id=rental_id.strip(),
                vehicle_id=vehicle_id.strip(),
                session_id=session_id,
                event_type="inspection_started",
                stage=stage.value,
                source_type="inspection_session",
                source_id=session_id,
                title="建立借還車檢查流程",
                summary=f"{stage.value} 階段檢查流程已建立。",
                outcome=InspectionSessionStatus.DRAFT.value,
                created_at=now,
            )
            return self._inspection_session_from_connection(connection, row)

    def _rental_inspection_state(self, connection: sqlite3.Connection, rental_id: str, vehicle_id: str) -> dict:
        pickup = connection.execute(
            """SELECT rowid AS sequence, * FROM inspection_sessions
            WHERE rental_id = ? AND vehicle_id = ? AND stage = 'pickup'
            ORDER BY rowid DESC LIMIT 1""", (rental_id, vehicle_id),
        ).fetchone()
        completed_return = None
        active = None
        if pickup is not None and pickup["status"] != "cancelled":
            if pickup["status"] == "capture_completed":
                completed_return = connection.execute(
                    """SELECT * FROM inspection_sessions WHERE rental_id = ? AND vehicle_id = ?
                    AND stage = 'return' AND status = 'capture_completed' AND rowid > ?
                    ORDER BY rowid DESC LIMIT 1""", (rental_id, vehicle_id, pickup["sequence"]),
                ).fetchone()
                if completed_return is None:
                    active = connection.execute(
                        """SELECT sessions.* FROM inspection_sessions AS sessions
                        WHERE rental_id = ? AND vehicle_id = ? AND stage = 'return'
                        AND status IN ('draft', 'in_progress') AND sessions.rowid > ?
                        ORDER BY (SELECT COUNT(*) FROM inspection_photos AS photos
                            WHERE photos.session_id = sessions.id AND capture_status = 'accepted') DESC,
                        sessions.rowid DESC LIMIT 1""", (rental_id, vehicle_id, pickup["sequence"]),
                    ).fetchone()
            else:
                active = pickup
        return {"pickup": pickup, "active": active, "completed_return": completed_return}

    def _close_obsolete_inspections(self, connection: sqlite3.Connection, *, rental_id: str,
                                    vehicle_id: str | None = None, keep_id: str | None = None) -> int:
        clauses = ["rental_id = ?", "status IN ('draft', 'in_progress')"]
        params = [rental_id]
        if vehicle_id is not None:
            clauses.append("vehicle_id = ?")
            params.append(vehicle_id)
        if keep_id is not None:
            clauses.append("id != ?")
            params.append(keep_id)
        obsolete = connection.execute(
            "SELECT * FROM inspection_sessions WHERE " + " AND ".join(clauses), params,
        ).fetchall()
        now = _now()
        for row in obsolete:
            connection.execute(
                "UPDATE inspection_sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            self._insert_rental_event(
                connection, rental_id=row["rental_id"], vehicle_id=row["vehicle_id"],
                session_id=row["id"], event_type="inspection_cancelled", stage=row["stage"],
                source_type="inspection_session", source_id=row["id"], title="關閉過期拍攝流程",
                summary="已由目前流程取代；原照片與歷史紀錄保留。", outcome="cancelled", created_at=now,
            )
        return len(obsolete)

    def reconcile_inspection_sessions(self) -> int:
        """Close obsolete drafts for every rental, keeping one valid active cycle."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rentals = connection.execute(
                "SELECT DISTINCT rental_id FROM inspection_sessions WHERE status IN ('draft', 'in_progress')",
            ).fetchall()
            has_accounts = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'auth_accounts'",
            ).fetchone() is not None
            closed = 0
            for rental in rentals:
                vehicle = connection.execute(
                    "SELECT vehicle_id FROM auth_accounts WHERE rental_id = ? AND kind = 'rental'",
                    (rental["rental_id"],),
                ).fetchone() if has_accounts else None
                if vehicle is None:
                    vehicle = connection.execute(
                        "SELECT vehicle_id FROM inspection_sessions WHERE rental_id = ? AND stage = 'pickup' "
                        "ORDER BY rowid DESC LIMIT 1",
                        (rental["rental_id"],),
                    ).fetchone()
                state = (self._rental_inspection_state(connection, rental["rental_id"], vehicle["vehicle_id"])
                         if vehicle else {"active": None})
                closed += self._close_obsolete_inspections(
                    connection, rental_id=rental["rental_id"],
                    keep_id=state["active"]["id"] if state["active"] is not None else None,
                )
        return closed

    def get_rental_inspection_state(self, *, rental_id: str, vehicle_id: str) -> dict:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._rental_inspection_state(connection, rental_id.strip(), vehicle_id.strip())
            self._close_obsolete_inspections(
                connection, rental_id=rental_id.strip(),
                keep_id=state["active"]["id"] if state["active"] is not None else None,
            )
            pickup_completed = state["pickup"] is not None and state["pickup"]["status"] == "capture_completed"
            return {
                "pickup_completed": pickup_completed,
                "return_allowed": pickup_completed and state["completed_return"] is None,
                "completed_return_session_id": (
                    state["completed_return"]["id"] if state["completed_return"] is not None else None
                ),
                "active_session_id": state["active"]["id"] if state["active"] is not None else None,
            }

    def has_completed_pickup(self, *, rental_id: str, vehicle_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT status FROM inspection_sessions
                WHERE rental_id = ? AND vehicle_id = ? AND stage = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (
                    rental_id.strip(),
                    vehicle_id.strip(),
                    InspectionStage.PICKUP.value,
                ),
            ).fetchone()
        return row is not None and row["status"] == InspectionSessionStatus.CAPTURE_COMPLETED.value

    def get_inspection_session(self, session_id: str) -> InspectionSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            return self._inspection_session_from_connection(connection, row)

    def list_inspection_sessions(
        self,
        *,
        stage: InspectionStage | None = None,
        status: InspectionSessionStatus | None = None,
        rental_id: str | None = None,
        limit: int = 50,
    ) -> list[InspectionSession]:
        clauses: list[str] = []
        params: list[object] = []
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if rental_id is not None:
            clauses.append("rental_id = ?")
            params.append(rental_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM inspection_sessions{where}
                ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()
            return [self._inspection_session_from_connection(connection, row) for row in rows]

    def record_inspection_photo(
        self,
        session_id: str,
        *,
        slot: ExpectedView,
        filename: str,
        quality_request_id: str,
        quality_status: QualityStatus,
        quality_score: int,
        object_key: str | None,
        content_hash: str | None = None,
        plate_prediction: PlatePrediction | None = None,
        damage_view: ExpectedView | None = None,
    ) -> InspectionSession | None:
        if damage_view is not None and (
            slot != ExpectedView.DAMAGE_DETAIL or damage_view not in PICKUP_REQUIRED_SLOTS
        ):
            raise ValueError("車損角度只能是左前、右前、左後或右後，且只用於車損特寫")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session_row is None:
                return None

            existing = connection.execute(
                """SELECT * FROM inspection_photos
                WHERE quality_request_id = ?""",
                (quality_request_id,),
            ).fetchone()
            if existing is not None:
                same_record = (
                    existing["session_id"] == session_id
                    and existing["slot"] == slot.value
                    and existing["filename"] == filename
                    and existing["quality_status"] == quality_status.value
                    and int(existing["quality_score"]) == quality_score
                    and existing["object_key"] == object_key
                    and existing["content_hash"] == content_hash
                    and existing["damage_view"] == (damage_view.value if damage_view else None)
                )
                if not same_record:
                    raise ValueError("quality_request_id 已用於不同的拍照結果")
                return self._inspection_session_from_connection(connection, session_row)

            if session_row["status"] in {
                InspectionSessionStatus.CAPTURE_COMPLETED.value,
                InspectionSessionStatus.CANCELLED.value,
            }:
                raise ValueError("此租還車檢查流程已關閉，不能再新增照片")

            accepted = quality_status in {QualityStatus.PASS, QualityStatus.WARNING}
            if slot == ExpectedView.DAMAGE_DETAIL and accepted and object_key:
                damage_detail_count = connection.execute(
                    """SELECT COUNT(*) AS count FROM inspection_photos
                    WHERE session_id = ? AND slot = ? AND capture_status = ?
                      AND object_key IS NOT NULL""",
                    (
                        session_id,
                        ExpectedView.DAMAGE_DETAIL.value,
                        InspectionPhotoStatus.ACCEPTED.value,
                    ),
                ).fetchone()
                if int(damage_detail_count["count"]) >= MAX_DAMAGE_DETAIL_PHOTOS:
                    raise ValueError(f"車損特寫一次最多 {MAX_DAMAGE_DETAIL_PHOTOS} 張")
            capture_status = (
                InspectionPhotoStatus.ACCEPTED if accepted else InspectionPhotoStatus.RETRY_REQUIRED
            )
            # Formal-flow photos stay private to the rental until the return is
            # completed; background recognition is queued then, never on upload.
            analysis_status = InspectionAnalysisStatus.NOT_REQUESTED
            photo_id = f"PIC-{uuid4().hex[:10].upper()}"
            now = _now()
            connection.execute(
                """INSERT INTO inspection_photos
                (id, session_id, slot, filename, quality_request_id, quality_status,
                 quality_score, capture_status, analysis_status, object_key,
                 content_hash, plate_prediction_json, damage_view, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    photo_id,
                    session_id,
                    slot.value,
                    filename,
                    quality_request_id,
                    quality_status.value,
                    quality_score,
                    capture_status.value,
                    analysis_status.value,
                    object_key,
                    content_hash,
                    plate_prediction.model_dump_json() if plate_prediction is not None else None,
                    damage_view.value if damage_view else None,
                    now,
                    now,
                ),
            )
            connection.execute(
                """UPDATE inspection_sessions
                SET status = ?, updated_at = ? WHERE id = ?""",
                (InspectionSessionStatus.IN_PROGRESS.value, now, session_id),
            )
            self._insert_rental_event(
                connection,
                rental_id=str(session_row["rental_id"]),
                vehicle_id=str(session_row["vehicle_id"]),
                session_id=session_id,
                event_type="photo_accepted" if accepted else "photo_retry_required",
                stage=str(session_row["stage"]),
                source_type="inspection_photo",
                source_id=photo_id,
                title="照片已接受" if accepted else "照片需要重拍",
                summary=f"{slot.value} 照片品質 {quality_status.value}，分數 {quality_score}。",
                outcome=capture_status.value,
                confidence=quality_score / 100,
                verified=accepted,
                metadata={
                    "slot": slot.value,
                    "damage_view": damage_view.value if damage_view else None,
                    "quality_status": quality_status.value,
                    "quality_score": quality_score,
                    "filename": filename,
                },
                created_at=now,
            )
            session_row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            assert session_row is not None
            return self._inspection_session_from_connection(connection, session_row)

    def has_inspection_photo_content_hash(self, content_hash: str) -> bool:
        with self._connect() as connection:
            inspection_match = connection.execute(
                """SELECT 1 FROM inspection_photos
                WHERE content_hash = ? AND object_key IS NOT NULL LIMIT 1""",
                (content_hash,),
            ).fetchone()
            if inspection_match is not None:
                return True
            sample_match = connection.execute(
                """SELECT 1 FROM learning_samples
                WHERE primary_image_hash = ? OR secondary_image_hash = ?
                   OR content_hash = ? LIMIT 1""",
                (content_hash, content_hash, content_hash),
            ).fetchone()
            if sample_match is not None:
                return True
            legacy_samples = connection.execute(
                """SELECT id, primary_image_path, secondary_image_path,
                          primary_image_hash, secondary_image_hash
                FROM learning_samples
                WHERE primary_image_hash IS NULL
                   OR (secondary_image_path IS NOT NULL AND secondary_image_hash IS NULL)"""
            ).fetchall()
            storage_root = self.learning_storage_path.resolve()
            for sample in legacy_samples:
                hashes: list[str | None] = []
                for path_value, existing_hash in (
                    (sample["primary_image_path"], sample["primary_image_hash"]),
                    (sample["secondary_image_path"], sample["secondary_image_hash"]),
                ):
                    if existing_hash or not path_value:
                        hashes.append(existing_hash)
                        continue
                    path = Path(path_value).resolve()
                    if storage_root not in path.parents:
                        hashes.append(None)
                        continue
                    try:
                        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                    except OSError:
                        hashes.append(None)
                connection.execute(
                    """UPDATE learning_samples
                    SET primary_image_hash = ?, secondary_image_hash = ? WHERE id = ?""",
                    (hashes[0], hashes[1], sample["id"]),
                )
                if content_hash in hashes:
                    return True
            return False

    @staticmethod
    def _analysis_tasks_for_slot(slot: ExpectedView) -> list[InspectionAnalysisTask]:
        if slot in {ExpectedView.INTERIOR_FRONT, ExpectedView.INTERIOR_REAR}:
            return [InspectionAnalysisTask.INTERIOR_CLEANLINESS]
        if slot == ExpectedView.DAMAGE_DETAIL:
            return []  # Close-ups are supporting evidence for human review.
        # Four standard exterior views are evaluated as pickup/return pairs.
        return []

    def _queue_rental_analysis(
        self, connection: sqlite3.Connection, return_row: sqlite3.Row, created_at: str
    ) -> None:
        """Hand a finished rental to the back office: queue every stored photo
        of the latest completed pickup and of this return for recognition."""
        plate_check_enabled = (
            self.plate_check_enabled
            if self.plate_check_enabled is not None
            else get_runtime_recognition_settings().inspection_plate_check_enabled
        )
        pickup_row = connection.execute(
            """SELECT id FROM inspection_sessions
            WHERE rental_id = ? AND vehicle_id = ? AND stage = ? AND status = ?
            ORDER BY capture_completed_at DESC, rowid DESC LIMIT 1""",
            (
                return_row["rental_id"],
                return_row["vehicle_id"],
                InspectionStage.PICKUP.value,
                InspectionSessionStatus.CAPTURE_COMPLETED.value,
            ),
        ).fetchone()
        session_ids = [return_row["id"]] if pickup_row is None else [pickup_row["id"], return_row["id"]]
        for session_id in session_ids:
            photos = connection.execute(
                """SELECT id, slot, plate_prediction_json FROM inspection_photos
                WHERE session_id = ? AND capture_status = ? AND object_key IS NOT NULL
                ORDER BY created_at, rowid""",
                (session_id, InspectionPhotoStatus.ACCEPTED.value),
            ).fetchall()
            for photo in photos:
                self._create_analysis_jobs(
                    connection,
                    photo_id=photo["id"],
                    session_id=session_id,
                    slot=ExpectedView(photo["slot"]),
                    created_at=created_at,
                    plate_check_enabled=plate_check_enabled,
                    plate_prediction=(
                        PlatePrediction.model_validate_json(photo["plate_prediction_json"])
                        if photo["plate_prediction_json"]
                        else None
                    ),
                )

    def _create_analysis_jobs(
        self,
        connection: sqlite3.Connection,
        *,
        photo_id: str,
        session_id: str,
        slot: ExpectedView,
        created_at: str,
        plate_check_enabled: bool,
        plate_prediction: PlatePrediction | None = None,
    ) -> None:
        tasks = self._analysis_tasks_for_slot(slot)
        if (plate_check_enabled or plate_prediction is not None) and slot in PICKUP_REQUIRED_SLOTS:
            existing_plate_job = connection.execute(
                """SELECT 1 FROM inspection_analysis_jobs
                WHERE session_id = ? AND task = ? LIMIT 1""",
                (session_id, InspectionAnalysisTask.PLATE_RECOGNITION.value),
            ).fetchone()
            if existing_plate_job is None or plate_prediction is not None:
                tasks.append(InspectionAnalysisTask.PLATE_RECOGNITION)

        if not tasks:
            connection.execute(
                """UPDATE inspection_photos SET analysis_status = ? WHERE id = ?""",
                (InspectionAnalysisStatus.NOT_REQUESTED.value, photo_id),
            )
            return

        active_jobs = connection.execute(
            """SELECT COUNT(*) AS count FROM inspection_analysis_jobs
            WHERE status IN (?, ?)""",
            (
                InspectionAnalysisJobStatus.QUEUED.value,
                InspectionAnalysisJobStatus.PROCESSING.value,
            ),
        ).fetchone()
        if int(active_jobs["count"]) + len(tasks) > self.analysis_queue_capacity:
            raise AnalysisQueueFullError(self.analysis_queue_capacity)

        for task in tasks:
            connection.execute(
                """INSERT OR IGNORE INTO inspection_analysis_jobs
                (id, photo_id, session_id, task, status, attempts, max_attempts,
                 result_json, error_message, created_at, updated_at, next_attempt_at)
                VALUES (?, ?, ?, ?, ?, 0, 3, ?, '', ?, ?, ?)""",
                (
                    f"JOB-{uuid4().hex[:10].upper()}",
                    photo_id,
                    session_id,
                    task.value,
                    InspectionAnalysisJobStatus.QUEUED.value,
                    json.dumps({"capture_prediction": plate_prediction.model_dump(mode="json")}, ensure_ascii=False)
                    if plate_prediction is not None and task == InspectionAnalysisTask.PLATE_RECOGNITION else "{}",
                    created_at,
                    created_at,
                    created_at,
                ),
            )
        connection.execute(
            """UPDATE inspection_photos SET analysis_status = ?, updated_at = ? WHERE id = ?""",
            (InspectionAnalysisStatus.QUEUED.value, created_at, photo_id),
        )

    def get_inspection_photo(self, session_id: str, photo_id: str) -> InspectionPhoto | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM inspection_photos
                WHERE id = ? AND session_id = ?""",
                (photo_id, session_id),
            ).fetchone()
            return self._inspection_photo(row) if row is not None else None

    def get_inspection_photo_object_key(self, session_id: str, photo_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT object_key FROM inspection_photos
                WHERE id = ? AND session_id = ?""",
                (photo_id, session_id),
            ).fetchone()
            return str(row["object_key"]) if row is not None and row["object_key"] else None

    def list_inspection_photo_records(
        self,
        *,
        limit: int = 200,
    ) -> list[InspectionPhotoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT photos.id, photos.session_id, sessions.rental_id,
                          sessions.vehicle_id, sessions.stage,
                          sessions.status AS session_status, photos.slot,
                          photos.damage_view, photos.filename, photos.quality_status,
                          photos.quality_score, photos.capture_status,
                          photos.analysis_status, photos.object_key,
                          COALESCE(json_extract(plate_jobs.result_json, '$.expected_plate'),
                                   json_extract(photos.plate_prediction_json, '$.expected_plate'))
                            AS expected_plate,
                          COALESCE(json_extract(plate_jobs.result_json, '$.recognized_plate'),
                                   json_extract(plate_jobs.result_json, '$.capture_prediction.normalized_plate'),
                                   json_extract(photos.plate_prediction_json, '$.normalized_plate'))
                            AS recognized_plate,
                          json_extract(plate_jobs.result_json, '$.human_verified_plate')
                            AS human_verified_plate,
                          COALESCE(json_extract(plate_jobs.result_json, '$.model_uncertain'),
                                   json_extract(plate_jobs.result_json, '$.capture_prediction.model_uncertain'),
                                   json_extract(photos.plate_prediction_json, '$.model_uncertain'), 0)
                            AS plate_model_uncertain,
                          photos.created_at
                   FROM inspection_photos AS photos
                   JOIN inspection_sessions AS sessions
                     ON sessions.id = photos.session_id
                   LEFT JOIN inspection_analysis_jobs AS plate_jobs
                     ON plate_jobs.photo_id = photos.id
                    AND plate_jobs.task = 'plate_recognition'
                   WHERE """ + _RELEASED_TO_OPS_SQL + """
                   ORDER BY photos.created_at DESC, photos.id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            jobs_by_photo: dict[str, list[RecognitionJobProgress]] = {}
            if rows:
                photo_ids = [row["id"] for row in rows]
                placeholders = ",".join("?" for _ in photo_ids)
                job_rows = connection.execute(
                    f"""SELECT jobs.*, photos.filename, photos.slot, photos.damage_view
                        FROM inspection_analysis_jobs jobs
                        JOIN inspection_photos photos ON photos.id=jobs.photo_id
                        WHERE jobs.photo_id IN ({placeholders}) ORDER BY jobs.created_at, jobs.rowid""",
                    photo_ids,
                ).fetchall()
                for job_row in job_rows:
                    jobs_by_photo.setdefault(job_row["photo_id"], []).append(self._recognition_job_progress(job_row))
        return [
            InspectionPhotoRecord.model_validate(
                {
                    **dict(row),
                    "analysis_jobs": jobs_by_photo.get(row["id"], []),
                    "image_available": bool(row["object_key"]),
                    "image_url": (
                        f"/api/v1/inspections/{row['session_id']}/photos/{row['id']}/image"
                        if row["object_key"]
                        else None
                    ),
                }
            )
            for row in rows
        ]

    def delete_inspection_photo_record(self, photo_id: str, *, actor: str) -> str | None | bool:
        """Delete one database record and return its private object key.

        ``False`` means the record did not exist; ``None`` means it existed but
        had no stored image (for example, a failed quality capture).
        """
        with self._connect() as connection:
            row = connection.execute(
                """SELECT photos.object_key, photos.session_id, photos.slot,
                          photos.filename, sessions.rental_id, sessions.vehicle_id,
                          sessions.stage
                   FROM inspection_photos AS photos
                   JOIN inspection_sessions AS sessions
                     ON sessions.id = photos.session_id
                   WHERE photos.id = ?""",
                (photo_id,),
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM inspection_photos WHERE id = ?", (photo_id,))
            connection.execute(
                "UPDATE alerts SET image_url = NULL WHERE image_url LIKE ?",
                (f"%/photos/{photo_id}/image",),
            )
            self._record_activity(
                connection,
                entity_type="inspection_photo",
                entity_id=photo_id,
                action="deleted",
                actor=actor,
                summary=f"刪除 {row['vehicle_id']} 的租還車照片",
                metadata={
                    "session_id": row["session_id"],
                    "rental_id": row["rental_id"],
                    "vehicle_id": row["vehicle_id"],
                    "stage": row["stage"],
                    "slot": row["slot"],
                    "filename": row["filename"],
                    "had_stored_image": bool(row["object_key"]),
                },
            )
            return str(row["object_key"]) if row["object_key"] else None

    def clear_inspection_photo_records(self, *, actor: str) -> tuple[int, list[str]]:
        with self._connect() as connection:
            # Only the archive the back office can see; rentals still being
            # captured keep their photos so the customer flow is not broken.
            released = connection.execute(
                """SELECT photos.id, photos.session_id, photos.object_key
                FROM inspection_photos AS photos
                JOIN inspection_sessions AS sessions ON sessions.id = photos.session_id
                WHERE """ + _RELEASED_TO_OPS_SQL
            ).fetchall()
            rows = [row for row in released if row["object_key"]]
            connection.executemany(
                "DELETE FROM inspection_photos WHERE id = ?",
                [(row["id"],) for row in released],
            )
            connection.executemany(
                "UPDATE alerts SET image_url = NULL WHERE image_url = ?",
                [(f"/api/v1/inspections/{row['session_id']}/photos/{row['id']}/image",)
                 for row in released],
            )
            deleted_records = len(released)
            if deleted_records:
                self._record_activity(
                    connection,
                    entity_type="inspection_photo_archive",
                    entity_id="all",
                    action="bulk_deleted",
                    actor=actor,
                    summary=f"清除全部 {deleted_records} 筆租還車照片紀錄",
                    metadata={
                        "deleted_records": deleted_records,
                        "stored_images": len(rows),
                    },
                )
        return deleted_records, [str(row["object_key"]) for row in rows]

    def find_pickup_reference_photos(
        self,
        *,
        rental_id: str,
        vehicle_id: str,
        slot: ExpectedView,
        completed_before: str,
    ) -> list[tuple[InspectionPhoto, str]]:
        """Return the accepted pickup photos of one angle for the exact rental and car.

        Candidates are the standard photo of that angle and any damage close-up
        tagged with it, all from the latest completed pickup that has one;
        the standard photo comes first.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT photos.* FROM inspection_photos AS photos
                JOIN inspection_sessions AS sessions ON sessions.id = photos.session_id
                WHERE sessions.rental_id = ? AND sessions.vehicle_id = ?
                  AND sessions.stage = ? AND sessions.status = ?
                  AND sessions.capture_completed_at <= ?
                  AND (photos.slot = ? OR (photos.slot = ? AND photos.damage_view = ?))
                  AND photos.capture_status = ?
                  AND photos.object_key IS NOT NULL
                ORDER BY sessions.capture_completed_at DESC, sessions.rowid DESC,
                         photos.slot = ? DESC, photos.created_at DESC, photos.id DESC""",
                (
                    rental_id.strip(),
                    vehicle_id.strip(),
                    InspectionStage.PICKUP.value,
                    InspectionSessionStatus.CAPTURE_COMPLETED.value,
                    completed_before,
                    slot.value,
                    ExpectedView.DAMAGE_DETAIL.value,
                    slot.value,
                    InspectionPhotoStatus.ACCEPTED.value,
                    slot.value,
                ),
            ).fetchall()
        if not rows:
            return []
        latest_session = rows[0]["session_id"]
        session_rows = [row for row in rows if row["session_id"] == latest_session]
        standard = [row for row in session_rows if row["slot"] == slot.value][:1]
        closeups = [row for row in session_rows if row["slot"] != slot.value]
        return [
            (self._inspection_photo(row), str(row["object_key"]))
            for row in [*standard, *closeups]
        ]

    def has_pickup_photo_content(
        self, *, rental_id: str, vehicle_id: str, content_hash: str
    ) -> bool:
        """True when this exact file was already saved as a pickup photo of the rental."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM inspection_photos AS photos
                JOIN inspection_sessions AS sessions ON sessions.id = photos.session_id
                WHERE sessions.rental_id = ? AND sessions.vehicle_id = ?
                  AND sessions.stage = ? AND photos.content_hash = ?
                  AND photos.object_key IS NOT NULL
                LIMIT 1""",
                (
                    rental_id.strip(),
                    vehicle_id.strip(),
                    InspectionStage.PICKUP.value,
                    content_hash,
                ),
            ).fetchone()
        return row is not None

    def list_inspection_analysis_jobs(
        self,
        *,
        session_id: str | None = None,
        status: InspectionAnalysisJobStatus | None = None,
        limit: int = 100,
    ) -> list[InspectionAnalysisJob]:
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM inspection_analysis_jobs{where}
                ORDER BY created_at, rowid LIMIT ?""",
                params,
            ).fetchall()
            return [self._inspection_analysis_job(row) for row in rows]

    def start_damage_pair_recognition(self, *, rental_id: str, vehicle_id: str) -> str:
        run_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO damage_pair_recognition_runs
                   (id, rental_id, vehicle_id, status, started_at)
                   VALUES (?, ?, ?, 'processing', ?)""",
                (run_id, rental_id, vehicle_id, _now()),
            )
        return run_id

    def finish_damage_pair_recognition(self, run_id: str, *, failed: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE damage_pair_recognition_runs
                   SET status = ?, finished_at = ? WHERE id = ?""",
                ("failed" if failed else "completed", _now(), run_id),
            )

    def list_recognition_progress(self, *, limit: int = 5) -> list[RecognitionProgress]:
        """Active work plus recent completed sessions; timings remain in the archive."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT jobs.session_id, sessions.rental_id, sessions.vehicle_id,
                          sessions.stage, COUNT(*) AS total,
                          SUM(CASE WHEN jobs.status = 'queued' THEN 1 ELSE 0 END) AS queued,
                          SUM(CASE WHEN jobs.status = 'processing' THEN 1 ELSE 0 END) AS processing,
                          SUM(CASE WHEN jobs.status IN ('completed', 'needs_review', 'resolved')
                              THEN 1 ELSE 0 END) AS completed,
                          SUM(CASE WHEN jobs.status = 'failed' THEN 1 ELSE 0 END) AS failed,
                          MIN(jobs.created_at) AS started_at,
                          MAX(jobs.completed_at) AS finished_at,
                          MAX(jobs.updated_at) AS last_updated_at
                   FROM inspection_analysis_jobs AS jobs
                   JOIN inspection_sessions AS sessions ON sessions.id = jobs.session_id
                   GROUP BY jobs.session_id
                   ORDER BY (queued + processing > 0) DESC, last_updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            damage_pair_rows = connection.execute(
                """SELECT id, rental_id, vehicle_id, status, started_at, finished_at
                   FROM damage_pair_recognition_runs
                   ORDER BY (status = 'processing') DESC, started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            jobs_by_session: dict[str, list[RecognitionJobProgress]] = {}
            if rows:
                session_ids = [row["session_id"] for row in rows]
                placeholders = ",".join("?" for _ in session_ids)
                job_rows = connection.execute(
                    f"""SELECT jobs.*, photos.filename, photos.slot, photos.damage_view
                        FROM inspection_analysis_jobs jobs
                        JOIN inspection_photos photos ON photos.id=jobs.photo_id
                        WHERE jobs.session_id IN ({placeholders}) ORDER BY jobs.created_at, jobs.rowid""",
                    session_ids,
                ).fetchall()
                for job_row in job_rows:
                    jobs_by_session.setdefault(job_row["session_id"], []).append(
                        self._recognition_job_progress(job_row)
                    )
        progress = [
            RecognitionProgress.model_validate(
                {**{key: row[key] for key in row.keys() if key != "last_updated_at"},
                 "jobs": jobs_by_session.get(row["session_id"], [])}
            )
            for row in rows
        ]
        progress.extend(
            RecognitionProgress(
                session_id=row["id"],
                rental_id=row["rental_id"],
                vehicle_id=row["vehicle_id"],
                stage="damage_pair",
                total=1,
                queued=0,
                processing=int(row["status"] == "processing"),
                completed=int(row["status"] == "completed"),
                failed=int(row["status"] == "failed"),
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in damage_pair_rows
        )
        progress.sort(
            key=lambda item: (
                item.queued + item.processing > 0,
                item.finished_at or item.started_at,
            ),
            reverse=True,
        )
        return progress[:limit]

    def inspection_load_summary(self) -> dict[str, int | bool]:
        with self._connect() as connection:
            job_rows = connection.execute(
                """SELECT status, COUNT(*) AS count FROM inspection_analysis_jobs
                WHERE status IN (?, ?) GROUP BY status""",
                (
                    InspectionAnalysisJobStatus.QUEUED.value,
                    InspectionAnalysisJobStatus.PROCESSING.value,
                ),
            ).fetchall()
            active_inspections = connection.execute(
                """SELECT COUNT(*) AS count FROM inspection_sessions
                WHERE status IN (?, ?)""",
                (
                    InspectionSessionStatus.DRAFT.value,
                    InspectionSessionStatus.IN_PROGRESS.value,
                ),
            ).fetchone()
        counts = {row["status"]: int(row["count"]) for row in job_rows}
        queued = counts.get(InspectionAnalysisJobStatus.QUEUED.value, 0)
        processing = counts.get(InspectionAnalysisJobStatus.PROCESSING.value, 0)
        return {
            "active_inspection_count": int(active_inspections["count"]),
            "queued_analysis_job_count": queued,
            "processing_analysis_job_count": processing,
            "analysis_queue_capacity": self.analysis_queue_capacity,
            "analysis_queue_at_capacity": (queued + processing >= self.analysis_queue_capacity),
        }

    @staticmethod
    def _attempt_elapsed_ms(row: sqlite3.Row, result: dict, now: str) -> float:
        """Callers without their own clock use wall time since this attempt started."""
        if not row["started_at"]:
            return float(result.get("active_duration_ms") or 0)
        return max(0, (datetime.fromisoformat(now)
                       - datetime.fromisoformat(row["started_at"])).total_seconds() * 1000)

    def heartbeat_inspection_analysis_job(self, job_id: str, *, started_at: str, elapsed_ms: float) -> bool:
        """Persist monotonic active time only while this attempt owns the job."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result_json FROM inspection_analysis_jobs "
                "WHERE id = ? AND status = 'processing' AND started_at = ?",
                (job_id, started_at),
            ).fetchone()
            if row is None:
                return False
            result = json.loads(row["result_json"])
            result["active_duration_ms"] = round(max(0, elapsed_ms), 1)
            connection.execute(
                "UPDATE inspection_analysis_jobs SET result_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(result, ensure_ascii=False), _now(), job_id),
            )
        return True

    def recover_interrupted_inspection_jobs(self, *, stale_seconds: float = 60) -> int:
        """Recover expired attempts without counting downtime as recognition time."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE status = 'processing' AND updated_at < ?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                result = json.loads(row["result_json"])
                result["duration_ms"] = round(float(result.get("duration_ms") or 0)
                                              + float(result.pop("active_duration_ms", 0) or 0), 1)
                result["interrupted"] = True
                retry = int(row["attempts"]) < int(row["max_attempts"])
                connection.execute(
                    """UPDATE inspection_analysis_jobs SET status = ?, result_json = ?,
                    started_at = NULL, updated_at = ?, completed_at = ?, next_attempt_at = NULL,
                    error_message = ? WHERE id = ?""",
                    ("queued" if retry else "failed", json.dumps(result, ensure_ascii=False),
                     now, None if retry else now,
                     "背景辨識中斷，已重新排隊；停機時間不計入辨識耗時" if retry
                     else "背景辨識中斷且已達重試上限，請人工處理", row["id"]),
                )
                self._refresh_photo_analysis_status(connection, row["photo_id"], now)
        return len(rows)

    def claim_next_inspection_analysis_job(self) -> InspectionAnalysisJob | None:
        self.recover_interrupted_inspection_jobs()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM inspection_analysis_jobs
                WHERE status = ? AND attempts < max_attempts
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY CASE task
                    WHEN 'plate_recognition' THEN 0
                    WHEN 'damage_comparison' THEN 1
                    WHEN 'interior_cleanliness' THEN 2
                    ELSE 3 END,
                    created_at, rowid LIMIT 1""",
                (InspectionAnalysisJobStatus.QUEUED.value, _now()),
            ).fetchone()
            if row is None:
                return None

            now = _now()
            timing = json.loads(row["result_json"])
            timing["duration_ms"] = float(timing.get("duration_ms") or 0)
            timing["active_duration_ms"] = 0
            connection.execute(
                "UPDATE inspection_analysis_jobs SET result_json = ? WHERE id = ?",
                (json.dumps(timing, ensure_ascii=False), row["id"]),
            )
            connection.execute(
                """UPDATE inspection_analysis_jobs
                SET status = ?, attempts = attempts + 1, updated_at = ?,
                    started_at = ?, completed_at = NULL, next_attempt_at = NULL
                WHERE id = ? AND status = ?""",
                (
                    InspectionAnalysisJobStatus.PROCESSING.value,
                    now,
                    now,
                    row["id"],
                    InspectionAnalysisJobStatus.QUEUED.value,
                ),
            )
            connection.execute(
                """UPDATE inspection_photos SET analysis_status = ?, updated_at = ?
                WHERE id = ?""",
                (InspectionAnalysisStatus.PROCESSING.value, now, row["photo_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE id = ?", (row["id"],)
            ).fetchone()
            assert claimed is not None
            return self._inspection_analysis_job(claimed)

    def complete_inspection_analysis_job(
        self,
        job_id: str,
        *,
        result: dict,
        needs_review: bool,
        elapsed_ms: float | None = None,
        expected_started_at: str | None = None,
    ) -> InspectionAnalysisJob | None:
        next_status = (
            InspectionAnalysisJobStatus.NEEDS_REVIEW
            if needs_review
            else InspectionAnalysisJobStatus.COMPLETED
        )
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            damage_case_id = (
                result.get("damage_case_id")
                if row["task"] == InspectionAnalysisTask.DAMAGE_COMPARISON.value
                else None
            )
            previous = json.loads(row["result_json"])
            if expected_started_at is not None and (row["status"] != "processing"
                    or row["started_at"] != expected_started_at):
                return None
            if elapsed_ms is None:
                elapsed_ms = OperationsStore._attempt_elapsed_ms(row, previous, now)
            result.pop("active_duration_ms", None)
            result = {**result, "duration_ms": round(
                float(previous.get("duration_ms") or 0) + max(elapsed_ms, float(result.get("inference_ms") or 0)), 1,
            )}
            session_row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (row["session_id"],)
            ).fetchone()
            connection.execute(
                """UPDATE inspection_analysis_jobs
                SET status = ?, result_json = ?, damage_case_id = ?,
                    error_message = '', updated_at = ?, completed_at = ?,
                    reviewed_at = NULL, reviewed_by = NULL
                WHERE id = ?""",
                (
                    next_status.value,
                    json.dumps(result, ensure_ascii=False),
                    damage_case_id,
                    now,
                    now,
                    job_id,
                ),
            )
            self._refresh_photo_analysis_status(connection, row["photo_id"], now)
            automation = result.get("automation")
            if session_row is not None and isinstance(automation, dict):
                decision_id = f"AUT-{uuid4().hex[:12].upper()}"
                connection.execute(
                    """INSERT OR IGNORE INTO automation_decisions
                    (id, job_id, session_id, user_id, rental_id, vehicle_id, task,
                     recommended_route, effective_route, shadow_mode, confidence,
                     reason_codes_json, evidence_json, policy_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        decision_id,
                        job_id,
                        row["session_id"],
                        self._resolve_user_id(connection, str(session_row["rental_id"])),
                        session_row["rental_id"],
                        session_row["vehicle_id"],
                        row["task"],
                        automation.get("recommended_route", "manual_review"),
                        automation.get("effective_route", "manual_review"),
                        int(bool(automation.get("shadow_mode"))),
                        automation.get("confidence"),
                        json.dumps(automation.get("reason_codes") or [], ensure_ascii=False),
                        json.dumps(automation.get("evidence") or {}, ensure_ascii=False),
                        automation.get("policy_version", "unknown"),
                        now,
                    ),
                )
                self._insert_rental_event(
                    connection,
                    rental_id=str(session_row["rental_id"]),
                    vehicle_id=str(session_row["vehicle_id"]),
                    session_id=str(row["session_id"]),
                    event_type="automation_decision",
                    stage=str(session_row["stage"]),
                    source_type="analysis_job",
                    source_id=job_id,
                    title="AI 自動化路由完成",
                    summary=(
                        f"{row['task']} 建議 {automation.get('recommended_route')}，"
                        f"實際 {automation.get('effective_route')}。"
                    ),
                    outcome=str(automation.get("effective_route") or "manual_review"),
                    confidence=automation.get("confidence"),
                    verified=False,
                    metadata={
                        "task": row["task"],
                        "recommended_route": automation.get("recommended_route"),
                        "effective_route": automation.get("effective_route"),
                        "shadow_mode": bool(automation.get("shadow_mode")),
                        "reason_codes": automation.get("reason_codes") or [],
                    },
                    created_at=now,
                )
            completed = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            assert completed is not None
            return self._inspection_analysis_job(completed)

    def fail_inspection_analysis_job(
        self, job_id: str, *, error_message: str, elapsed_ms: float | None = None,
        expected_started_at: str | None = None,
    ) -> InspectionAnalysisJob | None:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            retry_available = int(row["attempts"]) < int(row["max_attempts"])
            result = json.loads(row["result_json"])
            if expected_started_at is not None and (row["status"] != "processing"
                    or row["started_at"] != expected_started_at):
                return None
            if elapsed_ms is None:
                elapsed_ms = OperationsStore._attempt_elapsed_ms(row, result, now)
            result.pop("active_duration_ms", None)
            result["duration_ms"] = round(float(result.get("duration_ms") or 0) + max(0, elapsed_ms), 1)
            next_status = (
                InspectionAnalysisJobStatus.QUEUED
                if retry_available
                else InspectionAnalysisJobStatus.FAILED
            )
            completed_at = None if retry_available else now
            next_attempt_at = None
            if retry_available:
                delay_seconds = min(300, 30 * (2 ** max(0, int(row["attempts"]) - 1)))
                next_attempt_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                ).isoformat()
            connection.execute(
                """UPDATE inspection_analysis_jobs
                SET status = ?, error_message = ?, updated_at = ?, completed_at = ?,
                    next_attempt_at = ?, result_json = ?
                WHERE id = ?""",
                (
                    next_status.value,
                    error_message[:1000],
                    now,
                    completed_at,
                    next_attempt_at,
                    json.dumps(result, ensure_ascii=False),
                    job_id,
                ),
            )
            self._refresh_photo_analysis_status(connection, row["photo_id"], now)
            failed = connection.execute(
                "SELECT * FROM inspection_analysis_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            assert failed is not None
            return self._inspection_analysis_job(failed)

    @staticmethod
    def _refresh_photo_analysis_status(
        connection: sqlite3.Connection, photo_id: str, updated_at: str
    ) -> None:
        rows = connection.execute(
            "SELECT status FROM inspection_analysis_jobs WHERE photo_id = ?", (photo_id,)
        ).fetchall()
        statuses = {str(row["status"]) for row in rows}
        if InspectionAnalysisJobStatus.PROCESSING.value in statuses:
            photo_status = InspectionAnalysisStatus.PROCESSING
        elif InspectionAnalysisJobStatus.QUEUED.value in statuses:
            photo_status = InspectionAnalysisStatus.QUEUED
        elif InspectionAnalysisJobStatus.FAILED.value in statuses:
            photo_status = InspectionAnalysisStatus.FAILED
        elif InspectionAnalysisJobStatus.NEEDS_REVIEW.value in statuses:
            photo_status = InspectionAnalysisStatus.NEEDS_REVIEW
        else:
            photo_status = InspectionAnalysisStatus.COMPLETED
        connection.execute(
            """UPDATE inspection_photos SET analysis_status = ?, updated_at = ?
            WHERE id = ?""",
            (photo_status.value, updated_at, photo_id),
        )

    def complete_inspection_capture(self, session_id: str) -> InspectionSession | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session_row is None:
                return None
            if session_row["status"] == InspectionSessionStatus.CAPTURE_COMPLETED.value:
                return self._inspection_session_from_connection(connection, session_row)
            if session_row["status"] == InspectionSessionStatus.CANCELLED.value:
                raise ValueError("此租還車檢查流程已取消")

            session = self._inspection_session_from_connection(connection, session_row)
            if session.missing_slots:
                raise IncompleteInspectionError(session.missing_slots)

            now = _now()
            connection.execute(
                """UPDATE inspection_sessions
                SET status = ?, updated_at = ?, capture_completed_at = ?
                WHERE id = ?""",
                (
                    InspectionSessionStatus.CAPTURE_COMPLETED.value,
                    now,
                    now,
                    session_id,
                ),
            )
            if session.stage == InspectionStage.RETURN:
                self._close_obsolete_inspections(connection, rental_id=session.rental_id)
                self._queue_rental_analysis(connection, session_row, now)
                comparison_photos = []
                for slot in PICKUP_REQUIRED_SLOTS:
                    return_photo = connection.execute(
                        """SELECT id FROM inspection_photos
                        WHERE session_id = ? AND slot = ? AND capture_status = ?
                          AND object_key IS NOT NULL
                        ORDER BY created_at DESC, id DESC LIMIT 1""",
                        (
                            session_id,
                            slot.value,
                            InspectionPhotoStatus.ACCEPTED.value,
                        ),
                    ).fetchone()
                    if return_photo is not None:
                        comparison_photos.append(return_photo)
                # Angle-tagged close-ups are compared with the same pickup angle.
                comparison_photos.extend(connection.execute(
                    """SELECT id FROM inspection_photos
                    WHERE session_id = ? AND slot = ? AND damage_view IS NOT NULL
                      AND capture_status = ? AND object_key IS NOT NULL
                    ORDER BY created_at, id""",
                    (
                        session_id,
                        ExpectedView.DAMAGE_DETAIL.value,
                        InspectionPhotoStatus.ACCEPTED.value,
                    ),
                ).fetchall())
                for return_photo in comparison_photos:
                    connection.execute(
                        """INSERT OR IGNORE INTO inspection_analysis_jobs
                        (id, photo_id, session_id, task, status, attempts, max_attempts,
                         result_json, error_message, created_at, updated_at, next_attempt_at)
                        VALUES (?, ?, ?, ?, ?, 0, 3, '{}', '', ?, ?, ?)""",
                        (
                            f"JOB-{uuid4().hex[:10].upper()}",
                            return_photo["id"],
                            session_id,
                            InspectionAnalysisTask.DAMAGE_COMPARISON.value,
                            InspectionAnalysisJobStatus.QUEUED.value,
                            now,
                            now,
                            now,
                        ),
                    )
                    connection.execute(
                        """UPDATE inspection_photos
                        SET analysis_status = ?, updated_at = ? WHERE id = ?""",
                        (
                            InspectionAnalysisStatus.QUEUED.value,
                            now,
                            return_photo["id"],
                        ),
                    )
            self._insert_rental_event(
                connection,
                rental_id=session.rental_id,
                vehicle_id=session.vehicle_id,
                session_id=session_id,
                event_type="capture_completed",
                stage=session.stage.value,
                source_type="inspection_capture",
                source_id=session_id,
                title="完成借還車拍攝",
                summary=f"{session.stage.value} 階段必要照片已完成。",
                outcome="completed",
                verified=True,
                created_at=now,
            )
            session_row = connection.execute(
                "SELECT * FROM inspection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            assert session_row is not None
            return self._inspection_session_from_connection(connection, session_row)

    def create_cleanliness_alert(
        self,
        *,
        rental_id: str,
        vehicle_id: str,
        filename: str,
        assessment: CleanlinessAssessment,
    ) -> Alert:
        alert_id = f"ALT-{uuid4().hex[:8].upper()}"
        created_at = _now()
        title = f"{vehicle_id} 車內疑似髒汙／雜物待覆核"
        evidence = [item.model_dump(mode="json") for item in assessment.evidence]
        if assessment.specialist is not None:
            seat = "後座" if getattr(assessment.view, "value", assessment.view) == "rear" else "前座"
            title = f"{vehicle_id} {seat}疑似髒汙待覆核"
            evidence.append({"category": "other", "severity": 0, "visible": False,
                             "description": f"{seat}髒汙模型達到覆核提示門檻，待工作人員確認。",
                             **assessment.specialist.model_dump(mode="json")})
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO alerts
                (id, type, severity, status, rental_id, vehicle_id, title, summary,
                 evidence_json, confidence, image_filename, created_at)
                VALUES (?, 'cleanliness', 'high', 'open', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert_id,
                    rental_id,
                    vehicle_id,
                    title,
                    assessment.summary,
                    json.dumps(
                        evidence,
                        ensure_ascii=False,
                    ),
                    assessment.confidence,
                    filename,
                    created_at,
                ),
            )
            self._record_activity(
                connection,
                entity_type="alert",
                entity_id=alert_id,
                action="created",
                actor="ai-system",
                summary=f"AI 建立 {vehicle_id} 車內清潔警示",
                metadata={"rental_id": rental_id, "vehicle_id": vehicle_id},
            )
            self._insert_rental_event(
                connection,
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                event_type="cleanliness_alert_created",
                source_type="alert",
                source_id=alert_id,
                title=title,
                summary=assessment.summary,
                outcome=AlertStatus.OPEN.value,
                severity="high",
                confidence=assessment.confidence,
                verified=False,
                metadata={"alert_type": "cleanliness", "specialist": (
                    assessment.specialist.model_dump(mode="json") if assessment.specialist else None)},
                created_at=created_at,
            )
            # Only email when a model boxed the dirt; unlocated suspicions stay in the review queue.
            if any(item.visible and item.bounding_box is not None for item in assessment.evidence):
                self._enqueue_alert_notification(
                    connection,
                    alert_id=alert_id,
                    alert_type="cleanliness",
                    rental_id=rental_id,
                    vehicle_id=vehicle_id,
                    title=title,
                    summary=assessment.summary,
                    confidence=assessment.confidence,
                    created_at=created_at,
                    body_text=self._cleaning_email_body(
                        vehicle_id=vehicle_id, evidence=evidence,
                        seat={"rear": "後座", "front": "前座"}.get(
                            getattr(assessment.view, "value", assessment.view), "車內"),
                        found_at=created_at, confirmed=False,
                    ),
                )
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(row)

    def create_damage_case(
        self,
        *,
        rental_id: str,
        vehicle_id: str,
        expected_view: ExpectedView,
        pickup_filename: str,
        return_filename: str,
        comparison: DamageComparisonResult,
    ) -> DamageCase:
        case_id = f"DMG-{uuid4().hex[:8].upper()}"
        created_at = _now()
        if comparison.status in {
            DamageComparisonStatus.REVIEW_REQUIRED,
            DamageComparisonStatus.INCOMPATIBLE,
        }:
            review_decision = DamageReviewDecision.PENDING.value
            reviewer_notes = ""
            reviewed_at = None
        elif comparison.status == DamageComparisonStatus.NO_CHANGE:
            review_decision = DamageReviewDecision.DISMISSED.value
            reviewer_notes = "雙階段模型未確認新增車損，未加入人工覆核佇列。"
            reviewed_at = created_at
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO damage_cases
                (id, rental_id, vehicle_id, expected_view, pickup_filename,
                 return_filename, comparison_json, review_decision, reviewer_notes,
                 created_at, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    case_id,
                    rental_id,
                    vehicle_id,
                    expected_view.value,
                    pickup_filename,
                    return_filename,
                    comparison.model_dump_json(),
                    review_decision,
                    reviewer_notes,
                    created_at,
                    reviewed_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
        return self._damage_case(row)

    def list_damage_cases(self) -> list[DamageCase]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT cases.* FROM damage_cases AS cases
                WHERE NOT (
                    cases.review_decision = ? AND EXISTS (
                        SELECT 1 FROM learning_samples AS samples
                        WHERE samples.task = ?
                          AND samples.status IN (?, ?)
                          AND (samples.source_id = cases.id
                               OR samples.id = cases.learning_sample_id)
                    )
                )
                -- Cases made before the same-photo guard compared a photo with
                -- itself; there is nothing for a person to review.
                AND NOT (
                    cases.review_decision = ? AND EXISTS (
                        SELECT 1 FROM learning_samples AS samples
                        WHERE samples.task = ?
                          AND (samples.source_id = cases.id
                               OR samples.id = cases.learning_sample_id)
                          AND samples.primary_image_hash = samples.secondary_image_hash
                    )
                )
                ORDER BY cases.created_at DESC""",
                (
                    DamageReviewDecision.PENDING.value,
                    LearningTask.DAMAGE_PAIR.value,
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                    DamageReviewDecision.PENDING.value,
                    LearningTask.DAMAGE_PAIR.value,
                ),
            ).fetchall()
        return [self._damage_case(row) for row in rows]

    def has_damage_pair_images(self, pickup_content: bytes, return_content: bytes) -> bool:
        """Detect an exact photo pair even though sample hashes include case IDs."""
        digest = hashlib.sha256()
        digest.update(pickup_content)
        digest.update(b"\x00PAIR\x00")
        digest.update(return_content)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_id, content_hash FROM learning_samples WHERE task = ?",
                (LearningTask.DAMAGE_PAIR.value,),
            ).fetchall()
        for row in rows:
            candidate = digest.copy()
            if row["source_id"]:
                candidate.update(b"\x00CASE\x00")
                candidate.update(str(row["source_id"]).encode("utf-8"))
            if candidate.hexdigest() == row["content_hash"]:
                return True
        return False

    def get_damage_case(self, case_id: str) -> DamageCase | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
        return self._damage_case(row) if row else None

    def attach_damage_case_learning_sample(self, case_id: str, sample_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE damage_cases SET learning_sample_id = ? WHERE id = ?",
                (sample_id, case_id),
            )

    def archive_automatic_reference(self, sample_id: str) -> None:
        """Keep paired evidence without asking an operator to label a clear result."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE learning_samples SET status = ?, training_eligible = 0,
                    reference_enabled = 0, reviewer_notes = ?
                WHERE id = ? AND status = ? AND sample_kind = 'good_reference'""",
                (
                    LearningStatus.REJECTED.value,
                    "系統保留原圖供案件查閱；無需人工真值。",
                    sample_id,
                    LearningStatus.PENDING_REVIEW.value,
                ),
            )

    def get_damage_case_image_file(self, case_id: str, slot: str) -> Path | None:
        column = "primary_image_path" if slot == "pickup" else "secondary_image_path"
        with self._connect() as connection:
            row = connection.execute(
                f"""SELECT samples.{column}
                FROM damage_cases AS cases
                JOIN learning_samples AS samples
                  ON samples.id = cases.learning_sample_id
                  OR (
                    cases.learning_sample_id IS NULL
                    AND samples.source_id = cases.id
                    AND samples.task = ?
                  )
                WHERE cases.id = ?
                ORDER BY CASE WHEN samples.id = cases.learning_sample_id THEN 0 ELSE 1 END,
                         samples.created_at DESC
                LIMIT 1""",
                (LearningTask.DAMAGE_PAIR.value, case_id),
            ).fetchone()
        if row is None or row[column] is None:
            return None
        candidate = Path(row[column]).resolve()
        root = self.learning_storage_path.resolve()
        if root not in candidate.parents or not candidate.is_file():
            return None
        return candidate

    def replace_damage_case_comparison(
        self,
        case_id: str,
        comparison: DamageComparisonResult,
        actor: str = "ops-prototype",
    ) -> DamageCase | None:
        """Store a re-run comparison on a pending case and its pending pair sample.

        The case stays pending even when the new run finds no change: an operator
        asked for it and still records the human decision.
        """
        predicted_label = (
            comparison.verification.decision.value
            if comparison.verification is not None
            else "no_new_damage"
            if comparison.status == DamageComparisonStatus.NO_CHANGE
            else comparison.status.value
        )
        confidence = (
            comparison.verification.confidence
            if comparison.verification is not None
            else comparison.risk_score
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
            if current is None:
                return None
            if current["review_decision"] != DamageReviewDecision.PENDING.value:
                raise ValueError("此案件已完成覆核，不能重新辨識")
            previous = json.loads(current["comparison_json"])
            connection.execute(
                "UPDATE damage_cases SET comparison_json = ? WHERE id = ?",
                (comparison.model_dump_json(), case_id),
            )
            connection.execute(
                """UPDATE learning_samples
                SET predicted_label = ?, confidence = ?, model_version = ?, metadata_json = ?
                WHERE task = ? AND status = ?
                  AND (id = ? OR source_id = ?)""",
                (
                    predicted_label,
                    confidence,
                    comparison.model_version,
                    json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False),
                    LearningTask.DAMAGE_PAIR.value,
                    LearningStatus.PENDING_REVIEW.value,
                    current["learning_sample_id"],
                    case_id,
                ),
            )
            self._record_activity(
                connection,
                entity_type="damage_case",
                entity_id=case_id,
                action="reanalyzed",
                actor=actor,
                summary=f"重新辨識車損：{previous.get('status')} → {comparison.status.value}",
                metadata={
                    "previous_status": previous.get("status"),
                    "status": comparison.status.value,
                    "candidates": len(comparison.candidates),
                    "model_version": comparison.model_version,
                },
            )
            row = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
        return self._damage_case(row)

    def review_damage_case(
        self,
        case_id: str,
        decision: DamageReviewDecision,
        reviewer_notes: str,
        actor: str = "ops-prototype",
    ) -> DamageCase | None:
        if decision == DamageReviewDecision.PENDING:
            raise ValueError("覆核結果不能仍為待確認")
        corrected_label = DAMAGE_REVIEW_LABELS[decision]
        reviewed_at = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
            if current is None:
                return None
            if current["review_decision"] != DamageReviewDecision.PENDING.value:
                raise ValueError("此案件已完成覆核，不能覆寫稽核紀錄")

            sample = connection.execute(
                """SELECT * FROM learning_samples
                WHERE source_id = ? AND task = ?
                ORDER BY created_at DESC LIMIT 1""",
                (case_id, LearningTask.DAMAGE_PAIR.value),
            ).fetchone()
            if decision == DamageReviewDecision.DISMISSED and sample is None:
                raise ValueError("找不到此案件的雙圖樣本，無法送入人工真值")

            connection.execute(
                """UPDATE damage_cases
                SET review_decision = ?, reviewer_notes = ?, reviewed_at = ?
                WHERE id = ? AND review_decision = ?""",
                (
                    decision.value,
                    reviewer_notes,
                    reviewed_at,
                    case_id,
                    DamageReviewDecision.PENDING.value,
                ),
            )
            if not connection.execute("SELECT changes()").fetchone()[0]:
                raise ValueError("此案件已完成覆核，不能覆寫稽核紀錄")

            if (
                sample is not None
                and sample["status"] != LearningStatus.PENDING_REVIEW.value
                and (
                    sample["status"]
                    not in (
                        {LearningStatus.REJECTED.value}
                        if decision == DamageReviewDecision.RETRY_PHOTOS
                        else {LearningStatus.APPROVED.value, LearningStatus.TRAINED.value}
                    )
                    or sample["corrected_label"] != corrected_label
                )
            ):
                raise ValueError("車損案件與現有人工真值不一致，已保留待查，未執行結案")
            if sample is not None and sample["status"] == LearningStatus.PENDING_REVIEW.value:
                retry_photos = decision == DamageReviewDecision.RETRY_PHOTOS
                training_eligible = int(corrected_label in DAMAGE_PAIR_TRAINING_LABELS)
                sample_status = (
                    LearningStatus.REJECTED if retry_photos else LearningStatus.APPROVED
                )
                connection.execute(
                    """UPDATE learning_samples
                    SET status = ?, corrected_label = ?, reviewer_notes = ?,
                        reviewed_at = ?, training_eligible = ?, reference_enabled = 0
                    WHERE id = ? AND status = ?""",
                    (
                        sample_status.value,
                        corrected_label,
                        reviewer_notes,
                        reviewed_at,
                        training_eligible,
                        sample["id"],
                        LearningStatus.PENDING_REVIEW.value,
                    ),
                )
                self._record_activity(
                    connection,
                    entity_type="learning_sample",
                    entity_id=sample["id"],
                    action="retry_requested" if retry_photos else "approved",
                    actor=actor,
                    summary=(
                        "車損照片需重拍，未建立人工真值"
                        if retry_photos else f"車損案件人工真值為 {corrected_label}"
                    ),
                    metadata={
                        "task": LearningTask.DAMAGE_PAIR.value,
                        "predicted_label": sample["predicted_label"],
                        "corrected_label": corrected_label,
                        "training_eligible": bool(training_eligible),
                        "reviewer_notes": reviewer_notes,
                        "damage_case_id": case_id,
                    },
                )

            analysis_job = connection.execute(
                """SELECT * FROM inspection_analysis_jobs
                WHERE damage_case_id = ? AND task = ?
                ORDER BY created_at DESC LIMIT 1""",
                (case_id, InspectionAnalysisTask.DAMAGE_COMPARISON.value),
            ).fetchone()
            resolved_job_id: str | None = None
            if (
                analysis_job is not None
                and analysis_job["status"] == InspectionAnalysisJobStatus.NEEDS_REVIEW.value
            ):
                job_result = json.loads(analysis_job["result_json"])
                job_result["human_review"] = {
                    "decision": decision.value,
                    "corrected_label": corrected_label,
                    "reviewer_notes": reviewer_notes,
                    "reviewed_by": actor,
                    "reviewed_at": reviewed_at,
                }
                connection.execute(
                    """UPDATE inspection_analysis_jobs
                    SET status = ?, result_json = ?, updated_at = ?,
                        reviewed_at = ?, reviewed_by = ?
                    WHERE id = ? AND status = ?""",
                    (
                        InspectionAnalysisJobStatus.RESOLVED.value,
                        json.dumps(job_result, ensure_ascii=False),
                        reviewed_at,
                        reviewed_at,
                        actor,
                        analysis_job["id"],
                        InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                    ),
                )
                resolved_job_id = str(analysis_job["id"])
                self._refresh_photo_analysis_status(
                    connection, str(analysis_job["photo_id"]), reviewed_at
                )

            reviewed_row = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
            assert reviewed_row is not None
            reviewed = self._damage_case(reviewed_row)
            if decision == DamageReviewDecision.CONFIRMED_DAMAGE:
                self._create_damage_alert_in_connection(connection, reviewed, actor=actor)

            final_row = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ?", (case_id,)
            ).fetchone()
            assert final_row is not None
            final_case = self._damage_case(final_row)
            self._record_activity(
                connection,
                entity_type="damage_case",
                entity_id=case_id,
                action="reviewed",
                actor=actor,
                summary=f"人工覆核結果：{decision.value}",
                metadata={
                    "reviewer_notes": reviewer_notes,
                    "corrected_label": corrected_label,
                    "learning_sample_id": str(sample["id"]) if sample else None,
                    "analysis_job_id": resolved_job_id,
                    "alert_id": final_case.alert_id,
                },
            )
            self._insert_rental_event(
                connection,
                rental_id=final_case.rental_id,
                vehicle_id=final_case.vehicle_id,
                event_type=(
                    "damage_confirmed"
                    if decision == DamageReviewDecision.CONFIRMED_DAMAGE
                    else "damage_reviewed"
                ),
                source_type="damage_case_review",
                source_id=case_id,
                title=(
                    "人工確認新增車損"
                    if decision == DamageReviewDecision.CONFIRMED_DAMAGE
                    else "車損案件已覆核"
                ),
                summary=reviewer_notes or f"人工覆核結果：{decision.value}",
                outcome=decision.value,
                severity=("high" if decision == DamageReviewDecision.CONFIRMED_DAMAGE else "none"),
                confidence=final_case.comparison.risk_score,
                verified=True,
                metadata={
                    "corrected_label": corrected_label,
                    "reviewed_by": actor,
                    "alert_id": final_case.alert_id,
                },
                created_at=reviewed_at,
            )
        return final_case

    def _create_damage_alert_in_connection(
        self,
        connection: sqlite3.Connection,
        damage_case: DamageCase,
        *,
        actor: str,
    ) -> Alert:
        if damage_case.alert_id:
            existing = connection.execute(
                "SELECT * FROM alerts WHERE id = ?", (damage_case.alert_id,)
            ).fetchone()
            if existing is not None:
                return self._alert(existing)

        alert_id = f"ALT-{uuid4().hex[:8].upper()}"
        created_at = _now()
        evidence = [
            {
                "category": "visual_change",
                "severity": 3,
                "description": candidate.description,
                "visible": True,
                "region_id": candidate.id,
                "bounding_box": candidate.bounding_box,
                "change_score": candidate.change_score,
            }
            for candidate in damage_case.comparison.candidates
        ]
        connection.execute(
            """INSERT INTO alerts
            (id, type, severity, status, rental_id, vehicle_id, title, summary,
             evidence_json, confidence, image_filename, created_at)
            VALUES (?, 'damage', 'high', 'open', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                alert_id,
                damage_case.rental_id,
                damage_case.vehicle_id,
                f"{damage_case.vehicle_id} 疑似新增車損待檢查",
                (
                    f"人工已確認 {len(evidence)} 個疑似新增變化區域，請安排驗車。"
                    if evidence
                    else "人工比對取還車照片確認新增車損，請安排驗車。"
                ),
                json.dumps(evidence, ensure_ascii=False),
                damage_case.comparison.risk_score,
                damage_case.return_filename,
                created_at,
            ),
        )
        connection.execute(
            "UPDATE damage_cases SET alert_id = ? WHERE id = ?",
            (alert_id, damage_case.id),
        )
        self._record_activity(
            connection,
            entity_type="alert",
            entity_id=alert_id,
            action="created",
            actor=actor,
            summary=f"人工確認 {damage_case.vehicle_id} 疑似新增車損並建立警示",
            metadata={"damage_case_id": damage_case.id},
        )
        self._insert_rental_event(
            connection,
            rental_id=damage_case.rental_id,
            vehicle_id=damage_case.vehicle_id,
            event_type="damage_alert_created",
            source_type="alert",
            source_id=alert_id,
            title=f"{damage_case.vehicle_id} 新增車損警示",
            summary="人工已確認新增車損，系統已建立營運警示。",
            outcome=AlertStatus.OPEN.value,
            severity="high",
            confidence=damage_case.comparison.risk_score,
            verified=True,
            metadata={"damage_case_id": damage_case.id, "actor": actor},
            created_at=created_at,
        )
        self._enqueue_alert_notification(
            connection,
            alert_id=alert_id,
            alert_type="damage",
            rental_id=damage_case.rental_id,
            vehicle_id=damage_case.vehicle_id,
            title=f"{damage_case.vehicle_id} 有新的車損，請安排驗車",
            summary=f"人工已確認 {len(evidence)} 個疑似新增變化區域，請安排驗車。",
            confidence=damage_case.comparison.risk_score,
            created_at=created_at,
            body_text=self._staff_email_body(
                headline=f"{damage_case.vehicle_id} 有新的車損，請安排驗車。",
                vehicle_id=damage_case.vehicle_id,
                found_at=created_at,
                area=_VIEW_LABELS.get(damage_case.expected_view.value),
                items_title="狀況：",
                items=[f"還車照片比借車時多了 {len(evidence)} 處損傷" if evidence
                       else "還車照片和借車時不一樣"],
                action="工作人員已比對借車與還車照片並確認，請安排驗車並拍照記錄。",
            ),
        )
        row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        assert row is not None
        return self._alert(row)

    def create_damage_alert(self, damage_case: DamageCase, actor: str = "ops-prototype") -> Alert:
        with self._connect() as connection:
            return self._create_damage_alert_in_connection(connection, damage_case, actor=actor)

    def create_damage_observation_alert(
        self,
        *,
        rental_id: str,
        vehicle_id: str,
        filename: str,
        assessment: DamageObservation,
        image_url: str | None = None,
    ) -> Alert:
        alert_id = f"ALT-{uuid4().hex[:8].upper()}"
        created_at = _now()
        evidence = [
            {
                "category": item.damage_type.value,
                "severity": assessment.severity.value,
                "description": f"{item.location}：{item.description}",
                "visible": True,
            }
            for item in assessment.evidence
        ]
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO alerts
                (id, type, severity, status, rental_id, vehicle_id, title, summary,
                 evidence_json, confidence, image_filename, image_url, created_at)
                VALUES (?, 'damage_observation', 'high', 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert_id,
                    rental_id,
                    vehicle_id,
                    f"{vehicle_id} 可見車損待人工確認",
                    assessment.summary,
                    json.dumps(evidence, ensure_ascii=False),
                    assessment.confidence,
                    filename,
                    image_url,
                    created_at,
                ),
            )
            self._record_activity(
                connection,
                entity_type="alert",
                entity_id=alert_id,
                action="created",
                actor="ai-system",
                summary=f"AI 建立 {vehicle_id} 可見車損警示",
                metadata={"rental_id": rental_id, "vehicle_id": vehicle_id},
            )
            self._insert_rental_event(
                connection,
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                event_type="damage_observation_alert_created",
                source_type="alert",
                source_id=alert_id,
                title=f"{vehicle_id} 可見車損待確認",
                summary=assessment.summary,
                outcome=AlertStatus.OPEN.value,
                severity=assessment.severity.value,
                confidence=assessment.confidence,
                verified=False,
                metadata={"alert_type": "damage_observation"},
                created_at=created_at,
            )
            self._enqueue_alert_notification(
                connection,
                alert_id=alert_id,
                alert_type="damage_observation",
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                title=f"{vehicle_id} 車身可能有損傷，請確認",
                summary=assessment.summary,
                confidence=assessment.confidence,
                created_at=created_at,
                body_text=self._staff_email_body(
                    headline=f"{vehicle_id} 車身可能有損傷。",
                    vehicle_id=vehicle_id,
                    found_at=created_at,
                    items_title="可能的損傷：",
                    items=[f"{item.location}：{item.description}" for item in assessment.evidence]
                    or [assessment.summary],
                    action="請到營運中心「異常警示」查看照片，確認是否需要安排驗車。",
                ),
            )
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(row)

    @staticmethod
    def _learning_content_hash(
        task: LearningTask,
        primary_content: bytes,
        secondary_content: bytes | None,
        source_id: str | None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(primary_content)
        if secondary_content is not None:
            digest.update(b"\x00PAIR\x00")
            digest.update(secondary_content)
        # A damage review belongs to one case. Reusing a sample across cases
        # can make one operator's decision silently resolve another case's truth.
        if task == LearningTask.DAMAGE_PAIR and source_id:
            digest.update(b"\x00CASE\x00")
            digest.update(source_id.encode("utf-8"))
        return digest.hexdigest()

    def find_learning_sample_by_content(
        self,
        *,
        task: LearningTask,
        primary_content: bytes,
        secondary_content: bytes | None = None,
        source_id: str | None = None,
    ) -> LearningSample | None:
        content_hash = self._learning_content_hash(
            task, primary_content, secondary_content, source_id
        )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_samples WHERE task = ? AND content_hash = ?",
                (task.value, content_hash),
            ).fetchone()
        return self._learning_sample(row) if row else None

    def existing_learning_content_hashes(
        self, *, task: LearningTask, content_hashes: list[str]
    ) -> set[str]:
        """Return which single-photo SHA-256 digests already exist in one dataset."""
        wanted = sorted({value.lower() for value in content_hashes})
        found: set[str] = set()
        with self._connect() as connection:
            # Stay well below SQLite's bound-parameter limit.
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""SELECT content_hash FROM learning_samples
                    WHERE task = ? AND content_hash IN ({placeholders})""",
                    (task.value, *chunk),
                ).fetchall()
                found.update(row["content_hash"] for row in rows)
        return found

    def collect_learning_sample(
        self,
        *,
        task: LearningTask,
        primary_content: bytes,
        primary_filename: str,
        trigger_reasons: list[str],
        secondary_content: bytes | None = None,
        secondary_filename: str | None = None,
        rental_id: str | None = None,
        vehicle_id: str | None = None,
        source_id: str | None = None,
        expected_label: str | None = None,
        predicted_label: str | None = None,
        confidence: float | None = None,
        score: int | None = None,
        model_version: str | None = None,
        metadata: dict | None = None,
        sample_kind: str = "hard_case",
        training_eligible: bool = False,
    ) -> LearningSample:
        """Store a deduplicated hard sample for later human truthing.

        Predictions and workflow expectations are suggestions only.  A sample is
        never eligible for training until a reviewer supplies a corrected label.
        """
        content_hash = self._learning_content_hash(
            task, primary_content, secondary_content, source_id
        )
        now = _now()
        clean_reasons = sorted({reason.strip() for reason in trigger_reasons if reason.strip()})
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM learning_samples WHERE task = ? AND content_hash = ?",
                (task.value, content_hash),
            ).fetchone()
            if existing is not None:
                merged_reasons = sorted(
                    set(json.loads(existing["trigger_reasons_json"])) | set(clean_reasons)
                )
                next_status = existing["status"]
                reviewed_at = existing["reviewed_at"]
                if existing["status"] == LearningStatus.TRAINED.value:
                    next_status = LearningStatus.PENDING_REVIEW.value
                    reviewed_at = None
                    merged_reasons = sorted(set(merged_reasons) | {"failed_after_training"})
                existing_confidence = existing["confidence"]
                next_confidence = confidence
                if existing_confidence is not None and confidence is not None:
                    next_confidence = min(float(existing_confidence), confidence)
                elif existing_confidence is not None:
                    next_confidence = float(existing_confidence)
                existing_score = existing["score"]
                next_score = score
                if existing_score is not None and score is not None:
                    next_score = min(int(existing_score), score)
                elif existing_score is not None:
                    next_score = int(existing_score)
                next_training_eligible = (
                    False
                    if existing["status"] == LearningStatus.REJECTED.value
                    else bool(existing["training_eligible"] or training_eligible)
                )
                connection.execute(
                    """UPDATE learning_samples
                    SET trigger_reasons_json = ?, occurrence_count = occurrence_count + 1,
                        last_seen_at = ?, confidence = ?, score = ?,
                        source_id = COALESCE(source_id, ?),
                        rental_id = COALESCE(rental_id, ?), vehicle_id = COALESCE(vehicle_id, ?),
                        status = ?, reviewed_at = ?, model_version = COALESCE(?, model_version),
                        metadata_json = ?,
                        sample_kind = CASE
                            WHEN sample_kind = 'hard_case' OR ? = 'hard_case'
                            THEN 'hard_case' ELSE 'good_reference' END,
                        training_eligible = ?
                    WHERE id = ?""",
                    (
                        json.dumps(merged_reasons, ensure_ascii=False),
                        now,
                        next_confidence,
                        next_score,
                        source_id,
                        rental_id,
                        vehicle_id,
                        next_status,
                        reviewed_at,
                        model_version,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        sample_kind,
                        int(next_training_eligible),
                        existing["id"],
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM learning_samples WHERE id = ?", (existing["id"],)
                ).fetchone()
                return self._learning_sample(row)

        lighting, primary_suffix = self._image_profile(primary_content)
        secondary_suffix = None
        if secondary_content is not None:
            _, secondary_suffix = self._image_profile(secondary_content)
        sample_id = f"LRN-{uuid4().hex[:10].upper()}"
        sample_directory = (self.learning_storage_path / task.value / sample_id).resolve()
        sample_directory.mkdir(parents=True, exist_ok=False)
        primary_path = sample_directory / f"primary{primary_suffix}"
        primary_path.write_bytes(primary_content)
        secondary_path: Path | None = None
        if secondary_content is not None and secondary_suffix is not None:
            secondary_path = sample_directory / f"secondary{secondary_suffix}"
            secondary_path.write_bytes(secondary_content)
        category_label = expected_label or predicted_label or "unclassified"
        auto_category = f"{task.value}/{lighting}/{category_label}"
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO learning_samples
                (id, task, status, trigger_reasons_json, content_hash,
                 primary_image_path, secondary_image_path, primary_filename,
                 secondary_filename, rental_id, vehicle_id, source_id, expected_label,
                 predicted_label, confidence, score, lighting, auto_category,
                 model_version, sample_kind, training_eligible, reference_enabled,
                 metadata_json, reviewer_notes, occurrence_count,
                 created_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        0, ?, '', 1, ?, ?)""",
                (
                    sample_id,
                    task.value,
                    LearningStatus.PENDING_REVIEW.value,
                    json.dumps(clean_reasons, ensure_ascii=False),
                    content_hash,
                    str(primary_path),
                    str(secondary_path) if secondary_path else None,
                    primary_filename,
                    secondary_filename,
                    rental_id,
                    vehicle_id,
                    source_id,
                    expected_label,
                    predicted_label,
                    confidence,
                    score,
                    lighting,
                    auto_category,
                    model_version,
                    sample_kind,
                    int(training_eligible),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.execute(
                """UPDATE learning_samples
                SET primary_image_hash = ?, secondary_image_hash = ? WHERE id = ?""",
                (
                    hashlib.sha256(primary_content).hexdigest(),
                    hashlib.sha256(secondary_content).hexdigest()
                    if secondary_content is not None else None,
                    sample_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return self._learning_sample(row)

    def list_learning_samples(
        self,
        *,
        status: LearningStatus | None = None,
        task: LearningTask | None = None,
        limit: int = 100,
        offset: int = 0,
        pending_inbox_only: bool = False,
    ) -> list[LearningSample]:
        clauses: list[str] = []
        params: list[str | int] = []
        if pending_inbox_only:
            clauses.extend(["samples.status = 'pending_review'", "samples.pending_inbox_hidden = 0"])
        if status is not None:
            clauses.append("samples.status = ?")
            params.append(status.value)
        if task is not None:
            clauses.append("samples.task = ?")
            params.append(task.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        params.append(offset)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT samples.*,
                    COALESCE(job_sessions.stage, quality_sessions.stage) AS inspection_stage,
                    COALESCE(
                        job_photos.slot, quality_photos.slot, cases.expected_view,
                        json_extract(samples.metadata_json, '$.inspection_slot')
                    ) AS inspection_slot
                FROM learning_samples AS samples
                LEFT JOIN inspection_analysis_jobs AS jobs ON jobs.id = samples.source_id
                LEFT JOIN inspection_photos AS job_photos ON job_photos.id = jobs.photo_id
                LEFT JOIN inspection_sessions AS job_sessions ON job_sessions.id = jobs.session_id
                LEFT JOIN inspection_photos AS quality_photos
                    ON quality_photos.quality_request_id = samples.source_id AND jobs.id IS NULL
                LEFT JOIN inspection_sessions AS quality_sessions
                    ON quality_sessions.id = quality_photos.session_id
                LEFT JOIN damage_cases AS cases
                    ON cases.id = samples.source_id AND samples.task = 'damage_pair'
                {where}
                ORDER BY samples.created_at DESC, samples.id DESC LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
        return [self._learning_sample(row) for row in rows]

    def list_learning_rentals(self) -> list[LearningRentalSummary]:
        """List only visible pending rental/vehicle groups, oldest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT TRIM(rental_id) AS rental_id,
                          TRIM(vehicle_id) AS vehicle_id,
                          COUNT(*) AS sample_count,
                          MIN(created_at) AS oldest_created_at,
                          MAX(created_at) AS newest_created_at
                   FROM learning_samples
                   WHERE rental_id IS NOT NULL AND TRIM(rental_id) <> ''
                     AND vehicle_id IS NOT NULL AND TRIM(vehicle_id) <> ''
                     AND status = 'pending_review' AND pending_inbox_hidden = 0
                   GROUP BY TRIM(rental_id), TRIM(vehicle_id)
                   ORDER BY oldest_created_at ASC, rental_id ASC, vehicle_id ASC"""
            ).fetchall()
        return [LearningRentalSummary.model_validate(dict(row)) for row in rows]

    def list_recent_learning_approvals(self, *, limit: int = 5) -> list[LearningSample]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM learning_samples
                WHERE status IN (?, ?) AND corrected_label IS NOT NULL
                ORDER BY reviewed_at DESC, id DESC LIMIT ?""",
                (
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                    limit,
                ),
            ).fetchall()
        return [self._learning_sample(row) for row in rows]

    def get_learning_sample(self, sample_id: str) -> LearningSample | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return self._learning_sample(row) if row else None

    def get_learning_sample_file(self, sample_id: str, slot: str) -> Path | None:
        column = "primary_image_path" if slot == "primary" else "secondary_image_path"
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {column} FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        if row is None or row[column] is None:
            return None
        candidate = Path(row[column]).resolve()
        root = self.learning_storage_path.resolve()
        if root not in candidate.parents or not candidate.is_file():
            return None
        return candidate

    def get_reference_sample_file(self, sample_id: str) -> Path | None:
        """Return only a reviewed exterior image explicitly enabled as a user guide."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT primary_image_path FROM learning_samples
                WHERE id = ? AND task = ? AND status IN (?, ?)
                  AND reference_enabled = 1""",
                (
                    sample_id,
                    LearningTask.EXTERIOR_VIEW.value,
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                ),
            ).fetchone()
        if row is None or row["primary_image_path"] is None:
            return None
        candidate = Path(row["primary_image_path"]).resolve()
        root = self.learning_storage_path.resolve()
        if root not in candidate.parents or not candidate.is_file():
            return None
        return candidate

    def delete_learning_sample(self, sample_id: str) -> int | None:
        """Delete one dataset record and source images owned by learning storage."""
        if self.has_active_training_run():
            raise ValueError("已有訓練工作正在執行，請完成後再刪除樣本")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT status, task, source_id, primary_image_path, secondary_image_path
                FROM learning_samples WHERE id = ?""",
                (sample_id,),
            ).fetchone()
            if row is None:
                return None
            image_paths = [row["primary_image_path"], row["secondary_image_path"]]
            if row["task"] == LearningTask.LICENSE_PLATE.value and row["source_id"]:
                self._write_plate_job_truth(connection, row["source_id"], None, None)
            connection.execute("DELETE FROM learning_samples WHERE id = ?", (sample_id,))

        storage_root = self.learning_storage_path.resolve()
        deleted_images = 0
        for raw_path in image_paths:
            if not raw_path:
                continue
            candidate = Path(raw_path).resolve()
            if storage_root not in candidate.parents or not candidate.is_file():
                continue
            candidate.unlink()
            deleted_images += 1
        return deleted_images

    def delete_learning_rental(
        self, *, rental_id: str, vehicle_id: str, expected_sample_count: int,
        sample_ids: list[str],
    ) -> tuple[int, int] | None:
        """Remove selected pending inbox entries without deleting any data or files."""
        rental_id = rental_id.strip()
        vehicle_id = vehicle_id.strip()
        if not rental_id or not vehicle_id:
            raise ValueError("請指定完整租單編號與車牌")
        if len(sample_ids) != expected_sample_count or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("待處理照片清單已變更，請重新查看清單後再移除")
        with self._connect() as connection:
            # Lock before validating so a concurrent approval cannot be hidden.
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT id FROM learning_samples
                   WHERE TRIM(rental_id) = ? AND TRIM(vehicle_id) = ?
                     AND status = 'pending_review' AND pending_inbox_hidden = 0""",
                (rental_id, vehicle_id),
            ).fetchall()
            if not rows:
                return None
            if {row["id"] for row in rows} != set(sample_ids):
                raise ValueError("待處理照片清單已變更，請重新查看清單後再移除")
            connection.execute(
                """UPDATE learning_samples SET pending_inbox_hidden = 1
                   WHERE TRIM(rental_id) = ? AND TRIM(vehicle_id) = ?
                     AND status = 'pending_review' AND pending_inbox_hidden = 0""",
                (rental_id, vehicle_id),
            )
        return len(rows), 0

    @staticmethod
    def _write_plate_job_truth(
        connection: sqlite3.Connection,
        source_id: str,
        plate: str | None,
        actor: str | None,
    ) -> None:
        row = connection.execute(
            "SELECT result_json FROM inspection_analysis_jobs WHERE id = ? AND task = ?",
            (source_id, InspectionAnalysisTask.PLATE_RECOGNITION.value),
        ).fetchone()
        if row is None:
            return
        result = json.loads(row["result_json"])
        if plate is None:
            result.pop("human_verified_plate", None)
            result.pop("human_verified_by", None)
        else:
            result["human_verified_plate"] = plate
            result["human_verified_by"] = actor
        connection.execute(
            "UPDATE inspection_analysis_jobs SET result_json = ? WHERE id = ?",
            (json.dumps(result, ensure_ascii=False), source_id),
        )

    def review_learning_sample(
        self,
        sample_id: str,
        *,
        approved: bool,
        corrected_label: str | None,
        reviewer_notes: str,
        training_eligible: bool | None = None,
        reference_enabled: bool = False,
        review_metadata: dict | None = None,
        actor: str = "system",
    ) -> LearningSample | None:
        status = LearningStatus.APPROVED if approved else LearningStatus.REJECTED
        eligibility_sql = "training_eligible" if training_eligible is None else "?"
        eligibility_values: list[object] = (
            [] if training_eligible is None else [int(training_eligible)]
        )
        with self._connect() as connection:
            current = connection.execute(
                """SELECT status, task, source_id, expected_label, predicted_label, metadata_json
                FROM learning_samples WHERE id = ?""",
                (sample_id,),
            ).fetchone()
            if current is None:
                return None
            if current["status"] == LearningStatus.REJECTED.value:
                raise ValueError("這筆樣本已永久排除，不能再次變更")
            if not approved and not corrected_label:
                raise ValueError("排除樣本前必須提供人工真值")
            if not approved and current["status"] == LearningStatus.TRAINED.value:
                raise ValueError("這筆樣本已納入模型，不能再排除")
            if (
                not approved
                and current["status"] == LearningStatus.APPROVED.value
                and self.has_active_training_run()
            ):
                raise ValueError("已有訓練工作正在執行，請完成後再排除樣本")
            metadata = json.loads(current["metadata_json"])
            if review_metadata is not None:
                metadata["human_review"] = {
                    **review_metadata,
                    "corrected_label": corrected_label,
                    "reviewed_by": actor,
                    "reviewed_at": _now(),
                }
            connection.execute(
                f"""UPDATE learning_samples
                SET status = ?, corrected_label = ?, reviewer_notes = ?, reviewed_at = ?,
                    training_eligible = {eligibility_sql}, reference_enabled = ?,
                    metadata_json = ?
                WHERE id = ?""",
                (
                    status.value,
                    corrected_label,
                    reviewer_notes,
                    _now(),
                    *eligibility_values,
                    int(reference_enabled and approved),
                    json.dumps(metadata, ensure_ascii=False),
                    sample_id,
                ),
            )
            if (
                approved
                and current["task"] == LearningTask.LICENSE_PLATE.value
                and current["source_id"]
            ):
                self._write_plate_job_truth(
                    connection, current["source_id"], corrected_label, actor
                )
            row = connection.execute(
                "SELECT * FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
            self._record_activity(
                connection,
                entity_type="learning_sample",
                entity_id=sample_id,
                action="approved" if approved else "rejected",
                actor=actor,
                summary=(
                    f"人工確認訓練真值為 {corrected_label}"
                    if approved
                    else f"人工排除樣本，真值為 {corrected_label}"
                ),
                metadata={
                    "task": current["task"],
                    "expected_label": current["expected_label"],
                    "predicted_label": current["predicted_label"],
                    "corrected_label": corrected_label,
                    "training_eligible": bool(row["training_eligible"]),
                    "reference_enabled": bool(row["reference_enabled"]),
                    "reviewer_notes": reviewer_notes,
                    "review_metadata": review_metadata or {},
                },
            )
        return self._learning_sample(row) if row else None

    def cancel_learning_sample_approval(
        self,
        sample_id: str,
        *,
        actor: str,
    ) -> LearningSample | None:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
            if current is None:
                return None
            # Trained truths may be edited too: the photos and the previous truth
            # stay with the sample, and the next training run uses the new one.
            if current["status"] not in {
                LearningStatus.APPROVED.value,
                LearningStatus.TRAINED.value,
            }:
                raise ValueError("只有已確認的人工真值可以重新標註")
            active_training = connection.execute(
                "SELECT 1 FROM training_runs WHERE status IN (?, ?) LIMIT 1",
                (TrainingRunStatus.QUEUED.value, TrainingRunStatus.RUNNING.value),
            ).fetchone()
            if active_training is not None:
                raise ValueError("已有訓練工作正在排隊或執行，完成後才能取消確認")

            damage_case = None
            analysis_job = None
            if current["task"] == LearningTask.DAMAGE_PAIR.value and current["source_id"]:
                damage_case = connection.execute(
                    "SELECT * FROM damage_cases WHERE id = ?",
                    (current["source_id"],),
                ).fetchone()
                if damage_case is not None and damage_case["alert_id"]:
                    raise ValueError("這筆車損真值已建立營運警示，不能從訓練資料區直接取消")
                analysis_job = connection.execute(
                    """SELECT * FROM inspection_analysis_jobs
                    WHERE damage_case_id = ? AND task = ?
                    ORDER BY created_at DESC LIMIT 1""",
                    (
                        current["source_id"],
                        InspectionAnalysisTask.DAMAGE_COMPARISON.value,
                    ),
                ).fetchone()

            metadata = json.loads(current["metadata_json"])
            previous_review = metadata.pop("human_review", None)
            # Keep the last truth so the review form can start from it; nothing a
            # reviewer entered is discarded by reopening a sample.
            metadata["previous_human_review"] = {
                **(previous_review if isinstance(previous_review, dict) else {}),
                "corrected_label": current["corrected_label"],
                "reviewer_notes": current["reviewer_notes"],
                "reviewed_at": current["reviewed_at"],
                "status": current["status"],
                "trained_at": current["trained_at"],
            }
            if current["task"] == LearningTask.LICENSE_PLATE.value and current["source_id"]:
                self._write_plate_job_truth(connection, current["source_id"], None, None)
            connection.execute(
                """UPDATE learning_samples
                SET status = ?, corrected_label = NULL, reviewer_notes = '',
                    reviewed_at = NULL, trained_at = NULL,
                    training_eligible = 0, reference_enabled = 0,
                    metadata_json = ?
                WHERE id = ?""",
                (
                    LearningStatus.PENDING_REVIEW.value,
                    json.dumps(metadata, ensure_ascii=False),
                    sample_id,
                ),
            )
            if (
                damage_case is not None
                and damage_case["review_decision"] != DamageReviewDecision.PENDING.value
            ):
                connection.execute(
                    """UPDATE damage_cases
                    SET review_decision = ?, reviewer_notes = '', reviewed_at = NULL
                    WHERE id = ?""",
                    (DamageReviewDecision.PENDING.value, damage_case["id"]),
                )
                self._record_activity(
                    connection,
                    entity_type="damage_case",
                    entity_id=damage_case["id"],
                    action="review_reopened",
                    actor=actor,
                    summary="取消人工真值，車損案件退回待覆核",
                    metadata={"learning_sample_id": sample_id},
                )
            if (
                analysis_job is not None
                and analysis_job["status"] == InspectionAnalysisJobStatus.RESOLVED.value
            ):
                job_result = json.loads(analysis_job["result_json"])
                job_result.pop("human_review", None)
                now = _now()
                connection.execute(
                    """UPDATE inspection_analysis_jobs
                    SET status = ?, result_json = ?, updated_at = ?,
                        reviewed_at = NULL, reviewed_by = NULL
                    WHERE id = ? AND status = ?""",
                    (
                        InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                        json.dumps(job_result, ensure_ascii=False),
                        now,
                        analysis_job["id"],
                        InspectionAnalysisJobStatus.RESOLVED.value,
                    ),
                )
                self._refresh_photo_analysis_status(connection, str(analysis_job["photo_id"]), now)
            reopened = connection.execute(
                "SELECT * FROM learning_samples WHERE id = ?", (sample_id,)
            ).fetchone()
            self._record_activity(
                connection,
                entity_type="learning_sample",
                entity_id=sample_id,
                action="approval_cancelled",
                actor=actor,
                summary=f"重新標註人工真值 {current['corrected_label']}，退回待處理並保留上次真值",
                metadata={
                    "task": current["task"],
                    "previous_status": current["status"],
                    "cancelled_corrected_label": current["corrected_label"],
                    "previous_reviewer_notes": current["reviewer_notes"],
                    "previous_training_eligible": bool(current["training_eligible"]),
                    "previous_reference_enabled": bool(current["reference_enabled"]),
                },
            )
        return self._learning_sample(reopened) if reopened else None

    def approve_learning_sample_by_source(
        self,
        source_id: str,
        *,
        corrected_label: str,
        reviewer_notes: str,
    ) -> None:
        damage_pair_training_eligible = int(corrected_label in DAMAGE_PAIR_TRAINING_LABELS)
        with self._connect() as connection:
            connection.execute(
                """UPDATE learning_samples
                SET status = ?, corrected_label = ?, reviewer_notes = ?, reviewed_at = ?,
                    training_eligible = CASE
                        WHEN task = ? THEN ?
                        ELSE training_eligible
                    END
                WHERE source_id = ? AND status = ?""",
                (
                    LearningStatus.APPROVED.value,
                    corrected_label,
                    reviewer_notes,
                    _now(),
                    LearningTask.DAMAGE_PAIR.value,
                    damage_pair_training_eligible,
                    source_id,
                    LearningStatus.PENDING_REVIEW.value,
                ),
            )

    def count_learning_samples(
        self,
        *,
        task: LearningTask | None = None,
        status: LearningStatus | None = None,
        pending_inbox_only: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[str] = []
        if pending_inbox_only:
            clauses.extend(["status = 'pending_review'", "pending_inbox_hidden = 0"])
        if task is not None:
            clauses.append("task = ?")
            params.append(task.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            return int(
                connection.execute(
                    f"SELECT COUNT(*) FROM learning_samples{where}", params
                ).fetchone()[0]
            )

    def learning_counts(self, column: str, *, include_rejected: bool = True) -> dict[str, int]:
        if column not in {"task", "lighting", "status"}:
            raise ValueError("Unsupported learning count column")
        where = "" if include_rejected else " WHERE status != ?"
        params: list[object] = [] if include_rejected else [LearningStatus.REJECTED.value]
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT {column}, COUNT(*) AS count
                FROM learning_samples{where} GROUP BY {column}""",
                params,
            ).fetchall()
        return {str(row[column]): int(row["count"]) for row in rows}

    def exterior_training_samples(self) -> list[tuple[LearningSample, Path]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM learning_samples
                WHERE task = ? AND status IN (?, ?) AND training_eligible = 1
                  AND corrected_label IN (?, ?, ?, ?)
                ORDER BY created_at""",
                (
                    LearningTask.EXTERIOR_VIEW.value,
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                    "left_front",
                    "right_front",
                    "left_rear",
                    "right_rear",
                ),
            ).fetchall()
        output: list[tuple[LearningSample, Path]] = []
        for row in rows:
            path = Path(row["primary_image_path"]).resolve()
            if path.is_file():
                output.append((self._learning_sample(row), path))
        return output

    def damage_pair_training_samples(
        self,
    ) -> list[tuple[LearningSample, Path, Path]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM learning_samples
                WHERE task = ? AND status IN (?, ?) AND training_eligible = 1
                  AND corrected_label IN (?, ?)
                ORDER BY created_at""",
                (
                    LearningTask.DAMAGE_PAIR.value,
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                    *DAMAGE_PAIR_TRAINING_LABELS,
                ),
            ).fetchall()
        output: list[tuple[LearningSample, Path, Path]] = []
        for row in rows:
            if not row["primary_image_path"] or not row["secondary_image_path"]:
                continue
            primary_path = Path(row["primary_image_path"]).resolve()
            secondary_path = Path(row["secondary_image_path"]).resolve()
            if not primary_path.is_file() or not secondary_path.is_file():
                continue
            output.append((self._learning_sample(row), primary_path, secondary_path))
        return output

    def count_approved_damage_pair_trainable(self) -> int:
        return sum(
            sample.status == LearningStatus.APPROVED
            for sample, _primary, _secondary in self.damage_pair_training_samples()
        )

    def task_training_samples(
        self, task: LearningTask, *, labels: tuple[str, ...] | None = None
    ) -> list[tuple[LearningSample, Path, Path | None]]:
        """Human-confirmed samples of one dataset whose image files still exist."""
        clauses = ["task = ?", "status IN (?, ?)", "training_eligible = 1",
                   "corrected_label IS NOT NULL"]
        params: list[str] = [
            task.value, LearningStatus.APPROVED.value, LearningStatus.TRAINED.value,
        ]
        if labels is not None:
            clauses.append(f"corrected_label IN ({','.join('?' for _ in labels)})")
            params.extend(labels)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM learning_samples WHERE {' AND '.join(clauses)}
                ORDER BY created_at""",
                params,
            ).fetchall()
        output: list[tuple[LearningSample, Path, Path | None]] = []
        for row in rows:
            primary_path = Path(row["primary_image_path"]).resolve()
            if not primary_path.is_file():
                continue
            secondary_path = (
                Path(row["secondary_image_path"]).resolve()
                if row["secondary_image_path"]
                else None
            )
            if secondary_path is not None and not secondary_path.is_file():
                secondary_path = None
            output.append((self._learning_sample(row), primary_path, secondary_path))
        return output

    def count_approved_exterior_trainable(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM learning_samples
                    WHERE task = ? AND status = ?
                      AND training_eligible = 1
                      AND corrected_label IN (?, ?, ?, ?)""",
                    (
                        LearningTask.EXTERIOR_VIEW.value,
                        LearningStatus.APPROVED.value,
                        "left_front",
                        "right_front",
                        "left_rear",
                        "right_rear",
                    ),
                ).fetchone()[0]
            )

    def list_reference_samples(
        self, *, expected_label: str, limit: int = 6
    ) -> list[LearningSample]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM learning_samples
                WHERE task = ? AND status IN (?, ?) AND reference_enabled = 1
                  AND corrected_label = ?
                ORDER BY reviewed_at DESC LIMIT ?""",
                (
                    LearningTask.EXTERIOR_VIEW.value,
                    LearningStatus.APPROVED.value,
                    LearningStatus.TRAINED.value,
                    expected_label,
                    limit,
                ),
            ).fetchall()
        return [self._learning_sample(row) for row in rows]

    def mark_learning_samples_trained(self, sample_ids: list[str]) -> None:
        if not sample_ids:
            return
        placeholders = ",".join("?" for _ in sample_ids)
        with self._connect() as connection:
            connection.execute(
                f"""UPDATE learning_samples SET status = ?, trained_at = ?
                WHERE id IN ({placeholders}) AND status = ?""",
                [
                    LearningStatus.TRAINED.value,
                    _now(),
                    *sample_ids,
                    LearningStatus.APPROVED.value,
                ],
            )

    def create_training_run(
        self,
        *,
        trigger: str,
        approved_samples: int,
        status: TrainingRunStatus,
        task: LearningTask = LearningTask.EXTERIOR_VIEW,
        message: str | None = None,
    ) -> TrainingRun:
        run_id = f"TRN-{uuid4().hex[:10].upper()}"
        now = _now()
        if message is None:
            message = (
                "等待背景訓練" if status == TrainingRunStatus.QUEUED else "樣本不足，略過訓練"
            )
        completed_at = now if status == TrainingRunStatus.SKIPPED else None
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO training_runs
                (id, task, status, trigger, approved_samples, baseline_metrics_json,
                 candidate_metrics_json, message, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, '{}', '{}', ?, ?, ?)""",
                (
                    run_id,
                    task.value,
                    status.value,
                    trigger,
                    approved_samples,
                    message,
                    now,
                    completed_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row)

    def update_training_run(self, run_id: str, **changes: object) -> TrainingRun | None:
        json_columns = {"baseline_metrics", "candidate_metrics"}
        allowed = {
            "status",
            "manifest_path",
            "candidate_path",
            "report_path",
            "log_path",
            "baseline_metrics",
            "candidate_metrics",
            "message",
            "started_at",
            "completed_at",
        }
        assignments: list[str] = []
        values: list[object] = []
        for key, value in changes.items():
            if key not in allowed:
                raise ValueError(f"Unsupported training-run field: {key}")
            column = f"{key}_json" if key in json_columns else key
            if isinstance(value, TrainingRunStatus):
                value = value.value
            if key in json_columns:
                value = json.dumps(value, ensure_ascii=False)
            assignments.append(f"{column} = ?")
            values.append(value)
        if assignments:
            values.append(run_id)
            with self._connect() as connection:
                connection.execute(
                    f"UPDATE training_runs SET {', '.join(assignments)} WHERE id = ?", values
                )
                row = connection.execute(
                    "SELECT * FROM training_runs WHERE id = ?", (run_id,)
                ).fetchone()
        else:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM training_runs WHERE id = ?", (run_id,)
                ).fetchone()
        return self._training_run(row) if row else None

    def list_training_runs(
        self, limit: int = 20, *, task: LearningTask | None = None
    ) -> list[TrainingRun]:
        with self._connect() as connection:
            if task is None:
                rows = connection.execute(
                    "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM training_runs WHERE task = ?
                    ORDER BY created_at DESC LIMIT ?""",
                    (task.value, limit),
                ).fetchall()
        return [self._training_run(row) for row in rows]

    def has_active_training_run(self, *, task: LearningTask | None = None) -> bool:
        query = "SELECT 1 FROM training_runs WHERE status IN (?, ?)"
        params: list[str] = [TrainingRunStatus.QUEUED.value, TrainingRunStatus.RUNNING.value]
        if task is not None:
            query += " AND task = ?"
            params.append(task.value)
        with self._connect() as connection:
            return bool(connection.execute(f"{query} LIMIT 1", params).fetchone())

    def next_queued_training_run(self) -> TrainingRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM training_runs WHERE status = ?
                ORDER BY created_at LIMIT 1""",
                (TrainingRunStatus.QUEUED.value,),
            ).fetchone()
        return self._training_run(row) if row else None

    def get_training_run(self, run_id: str) -> TrainingRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row) if row else None

    def list_alerts(self, status: str | None = None) -> list[Alert]:
        # Every alert shows its photo as evidence: alerts that only kept the
        # filename are matched to the stored photo of the same rental, preferring
        # the return photo because pickup and return files may share a name.
        query = """SELECT alerts.*, (
                SELECT '/api/v1/inspections/' || photos.session_id ||
                       '/photos/' || photos.id || '/image'
                FROM inspection_photos AS photos
                JOIN inspection_sessions AS sessions ON sessions.id = photos.session_id
                WHERE photos.filename = alerts.image_filename
                  AND sessions.rental_id = alerts.rental_id
                  AND sessions.vehicle_id = alerts.vehicle_id
                  AND photos.object_key IS NOT NULL
                  AND photos.created_at <= alerts.created_at
                ORDER BY sessions.stage = ? DESC, photos.created_at DESC
                LIMIT 1
            ) AS matched_image_url FROM alerts"""
        params: tuple[str, ...] = (InspectionStage.RETURN.value,)
        if status:
            query += " WHERE status = ?"
            params += (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        alerts = []
        for row in rows:
            alert = self._alert(row)
            if alert.image_url is None and row["matched_image_url"]:
                alert.image_url = row["matched_image_url"]
            alerts.append(alert)
        return alerts

    def list_operations_activity(
        self, *, limit: int | None = 50, entity_type: str | None = None, offset: int = 0,
        exclude_learning_samples: bool = False, include_archived: bool = False,
        before_created_at: str | None = None, before_id: str | None = None, actor: str | None = None,
    ) -> list[OperationsActivity]:
        source = "operations_activity"
        if include_archived:
            source = "(SELECT * FROM operations_activity UNION ALL SELECT * FROM operations_activity_archive)"
        conditions: list[str] = []
        params: list[str | int] = []
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor)
        if exclude_learning_samples:
            conditions.append("entity_type NOT IN (" + ",".join("?" for _ in PRIVATE_ENTITY_TYPES) + ")")
            params.extend(PRIVATE_ENTITY_TYPES)
        if before_created_at is not None and before_id is not None:
            conditions.append(
                "(julianday(created_at) < julianday(?) OR (julianday(created_at) = julianday(?) AND id < ?))"
            )
            params.extend([before_created_at, before_created_at, before_id])
        query = f"SELECT * FROM {source}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY julianday(created_at) DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit if limit is not None else -1, offset])
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._operations_activity(row) for row in rows]

    def clear_operations_activity_older_than(
        self,
        *,
        days: int,
        now: datetime | None = None,
    ) -> int:
        """Keep the fixed 180-day active timeline cleanup, archiving history atomically first."""
        if days < 1:
            raise ValueError("保留天數至少要是 1 天")
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO operations_activity_archive
                SELECT * FROM operations_activity WHERE julianday(created_at) < julianday(?)""",
                (cutoff.isoformat(),),
            )
            result = connection.execute(
                "DELETE FROM operations_activity WHERE julianday(created_at) < julianday(?)",
                (cutoff.isoformat(),),
            )
        return result.rowcount

    def record_request_activity(self, *, request_id: str, actor: str, summary: str,
                                metadata: dict, restricted: bool = False) -> None:
        with self._connect() as connection:
            self._record_activity(connection, entity_type="admin_request" if restricted else "api_request",
                                  entity_id=request_id, action="request", actor=actor,
                                  summary=summary, metadata=metadata)

    def list_notification_outbox(
        self,
        *,
        limit: int | None = 100,
        status: NotificationDeliveryStatus | None = None,
    ) -> list[NotificationOutboxItem]:
        query = "SELECT * FROM notification_outbox"
        params: list[str | int] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._notification_outbox(row) for row in rows]

    def claim_notification_outbox(
        self,
        *,
        limit: int = 20,
        provider: str,
    ) -> list[NotificationOutboxItem]:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT id FROM notification_outbox
                WHERE status IN (?, ?) AND attempts < max_attempts
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY created_at ASC LIMIT ?""",
                (
                    NotificationDeliveryStatus.PENDING.value,
                    NotificationDeliveryStatus.FAILED.value,
                    now,
                    limit,
                ),
            ).fetchall()
            claimed: list[NotificationOutboxItem] = []
            for row in rows:
                connection.execute(
                    """UPDATE notification_outbox
                    SET status = ?, provider = ?, attempts = attempts + 1,
                        updated_at = ?
                    WHERE id = ? AND status IN (?, ?)""",
                    (
                        NotificationDeliveryStatus.PROCESSING.value,
                        provider,
                        now,
                        row["id"],
                        NotificationDeliveryStatus.PENDING.value,
                        NotificationDeliveryStatus.FAILED.value,
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0]:
                    claimed_row = connection.execute(
                        "SELECT * FROM notification_outbox WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    if claimed_row is not None:
                        claimed.append(self._notification_outbox(claimed_row))
        return claimed

    def mark_notification_handed_off(
        self, notification_id: str
    ) -> NotificationOutboxItem | None:
        """Mark a notification as handed to Power Automate, not email-delivered."""
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE notification_outbox
                SET status = ?, handed_off_at = ?, last_error = NULL,
                    next_attempt_at = NULL, updated_at = ?
                WHERE id = ? AND status = ?""",
                (
                    NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED.value,
                    now,
                    now,
                    notification_id,
                    NotificationDeliveryStatus.PROCESSING.value,
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_outbox WHERE id = ?", (notification_id,)
            ).fetchone()
        return self._notification_outbox(row) if row else None

    def mark_power_automate_result(
        self, notification_id: str, *, sent: bool
    ) -> NotificationOutboxItem | None:
        """Mirror a Power Automate file moved to 已寄出／已寄 or 失敗."""
        now = _now()
        next_status = (
            NotificationDeliveryStatus.SENT
            if sent
            else NotificationDeliveryStatus.POWER_AUTOMATE_FAILED
        )
        eligible_statuses = (
            (
                NotificationDeliveryStatus.PROCESSING.value,
                NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED.value,
                NotificationDeliveryStatus.POWER_AUTOMATE_FAILED.value,
            )
            if sent
            else (
                NotificationDeliveryStatus.PROCESSING.value,
                NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED.value,
            )
        )
        placeholders = ",".join("?" for _ in eligible_statuses)
        with self._connect() as connection:
            connection.execute(
                f"""UPDATE notification_outbox
                SET status = ?, sent_at = ?, last_error = ?,
                    next_attempt_at = NULL, updated_at = ?
                WHERE id = ? AND provider = ?
                  AND status IN ({placeholders})""",
                (
                    next_status.value,
                    now if sent else None,
                    None if sent else "Power Automate 已將通知移到失敗資料夾",
                    now,
                    notification_id,
                    POWER_AUTOMATE_PROVIDER,
                    *eligible_statuses,
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_outbox WHERE id = ?", (notification_id,)
            ).fetchone()
        return self._notification_outbox(row) if row else None

    def mark_notification_failed(
        self,
        notification_id: str,
        *,
        error_message: str,
    ) -> NotificationOutboxItem | None:
        settings = get_settings()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT attempts FROM notification_outbox WHERE id = ?",
                (notification_id,),
            ).fetchone()
            if current is None:
                return None
            delay_seconds = settings.notification_retry_base_seconds * (
                2 ** max(0, int(current["attempts"]) - 1)
            )
            now = datetime.now(timezone.utc)
            connection.execute(
                """UPDATE notification_outbox
                SET status = ?, last_error = ?, next_attempt_at = ?, updated_at = ?
                WHERE id = ? AND status = ?""",
                (
                    NotificationDeliveryStatus.FAILED.value,
                    error_message[:1000],
                    (now + timedelta(seconds=delay_seconds)).isoformat(),
                    now.isoformat(),
                    notification_id,
                    NotificationDeliveryStatus.PROCESSING.value,
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_outbox WHERE id = ?",
                (notification_id,),
            ).fetchone()
        return self._notification_outbox(row) if row else None

    def retry_notification(self, notification_id: str) -> NotificationOutboxItem | None:
        now = _now()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT status FROM notification_outbox WHERE id = ?",
                (notification_id,),
            ).fetchone()
            if current is None:
                return None
            if current["status"] == NotificationDeliveryStatus.SENT.value:
                raise ValueError("已寄送成功的 Email 不需要重試")
            if current["status"] == NotificationDeliveryStatus.PROCESSING.value:
                raise ValueError("通知正在派送，請先確認外部寄送結果")
            if current["status"] == NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED.value:
                raise ValueError(
                    "通知已交給 Power Automate，請在 OneDrive 的失敗資料夾重試"
                )
            if current["status"] == NotificationDeliveryStatus.POWER_AUTOMATE_FAILED.value:
                raise ValueError("請將通知檔移回 OneDrive 待寄資料夾，由 Power Automate 重試")
            connection.execute(
                """UPDATE notification_outbox
                SET status = ?, attempts = 0, last_error = NULL,
                    next_attempt_at = ?, sent_at = NULL, updated_at = ?
                WHERE id = ?""",
                (
                    NotificationDeliveryStatus.PENDING.value,
                    now,
                    now,
                    notification_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_outbox WHERE id = ?",
                (notification_id,),
            ).fetchone()
        return self._notification_outbox(row)

    def get_alert(self, alert_id: str) -> Alert | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(row) if row else None

    def set_alert_status(
        self, alert_id: str, status: str, actor: str = "ops-prototype"
    ) -> Alert | None:
        allowed_transitions = {
            "open": {"acknowledged", "resolved", "cancelled"},
            "acknowledged": {"resolved", "cancelled"},
            "resolved": set(),
            "cancelled": set(),
        }
        if status not in {"acknowledged", "resolved", "cancelled"}:
            raise ValueError("不支援的警示狀態")
        timestamp_column = {
            "acknowledged": "acknowledged_at",
            "resolved": "resolved_at",
            "cancelled": "cancelled_at",
        }[status]
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if current is None:
                return None
            if current["status"] == status:
                return self._alert(current)
            if status not in allowed_transitions.get(current["status"], set()):
                raise ValueError(f"警示不能由 {current['status']} 直接改為 {status}")
            connection.execute(
                f"UPDATE alerts SET status = ?, {timestamp_column} = ? WHERE id = ?",
                (status, _now(), alert_id),
            )
            status_summary = {
                "acknowledged": "警示已確認",
                "resolved": "警示已結案",
                "cancelled": "警示已取消",
            }
            self._record_activity(
                connection,
                entity_type="alert",
                entity_id=alert_id,
                action="status_changed",
                actor=actor,
                summary=status_summary[status],
                metadata={"from": current["status"], "to": status},
            )
            row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(row) if row else None

    def review_cleanliness_alert(
        self,
        alert_id: str,
        *,
        decision: CleanlinessReviewDecision,
        reviewer_notes: str = "",
        actor: str = "ops-prototype",
    ) -> Alert | None:
        """Record a human cleanliness verdict without treating an AI alert as truth."""
        reviewed_at = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if current is None:
                return None
            if current["type"] != "cleanliness":
                raise ValueError("這個按鈕只用來確認車內髒污")
            if current["review_decision"]:
                if current["review_decision"] == decision.value:
                    return self._alert(current)
                raise ValueError("這筆髒污警示已經確認過，不能改成相反結果")
            if current["status"] in {
                AlertStatus.RESOLVED.value,
                AlertStatus.CANCELLED.value,
            }:
                raise ValueError("這筆警示已經結束，不能再確認")

            if decision == CleanlinessReviewDecision.DISMISSED:
                active_order = connection.execute(
                    """SELECT 1 FROM work_orders
                    WHERE alert_id = ? AND status NOT IN (?, ?) LIMIT 1""",
                    (
                        alert_id,
                        WorkOrderStatus.COMPLETED.value,
                        WorkOrderStatus.CANCELLED.value,
                    ),
                ).fetchone()
                if active_order is not None:
                    raise ValueError("請先取消進行中的清潔工單")

            confirmed = decision == CleanlinessReviewDecision.CONFIRMED
            next_status = (
                AlertStatus.ACKNOWLEDGED.value
                if confirmed
                else AlertStatus.CANCELLED.value
            )
            timestamp_column = "acknowledged_at" if confirmed else "cancelled_at"
            notes = reviewer_notes or (
                "工作人員看過照片後，確認車內有髒污。"
                if confirmed
                else "工作人員看過照片後，確認車內沒有髒污。"
            )
            connection.execute(
                f"""UPDATE alerts
                SET status = ?, {timestamp_column} = COALESCE({timestamp_column}, ?),
                    review_decision = ?, reviewer_notes = ?, reviewed_at = ?
                WHERE id = ?""",
                (
                    next_status,
                    reviewed_at,
                    decision.value,
                    notes,
                    reviewed_at,
                    alert_id,
                ),
            )
            self._record_activity(
                connection,
                entity_type="alert",
                entity_id=alert_id,
                action="cleanliness_reviewed",
                actor=actor,
                summary="人工確認車內有髒污" if confirmed else "人工確認車內沒有髒污",
                metadata={"decision": decision.value, "reviewer_notes": notes},
            )
            self._insert_rental_event(
                connection,
                rental_id=str(current["rental_id"]),
                vehicle_id=str(current["vehicle_id"]),
                event_type=("cleanliness_confirmed" if confirmed else "cleanliness_reviewed"),
                source_type="alert_review",
                source_id=alert_id,
                title="工作人員確認車內有髒污" if confirmed else "工作人員確認車內沒有髒污",
                summary=notes,
                outcome=decision.value,
                severity=str(current["severity"]) if confirmed else None,
                verified=True,
                metadata={"actor": actor},
                created_at=reviewed_at,
            )
            if confirmed:
                # Staff saw the photo and said dirty, so the cleaners get a plain-language job email.
                vehicle_id = str(current["vehicle_id"])
                self._enqueue_alert_notification(
                    connection,
                    alert_id=alert_id,
                    alert_type="cleanliness",
                    rental_id=str(current["rental_id"]),
                    vehicle_id=vehicle_id,
                    title=f"{vehicle_id} {_seat_from_title(current['title'])}需要清潔",
                    summary=notes,
                    confidence=float(current["confidence"]),
                    created_at=reviewed_at,
                    body_text=self._cleaning_email_body(
                        vehicle_id=vehicle_id, seat=_seat_from_title(current["title"]),
                        evidence=json.loads(current["evidence_json"]),
                        found_at=str(current["created_at"]), confirmed=True,
                    ),
                    event="cleaning_confirmed",
                    recipient=get_settings().notification_cleaning_email_recipient,
                )
            analysis_job = connection.execute(
                """SELECT * FROM inspection_analysis_jobs
                WHERE task = ? AND json_extract(result_json, '$.alert_id') = ?
                ORDER BY created_at DESC LIMIT 1""",
                (InspectionAnalysisTask.INTERIOR_CLEANLINESS.value, alert_id),
            ).fetchone()
            if (
                analysis_job is not None
                and analysis_job["status"] == InspectionAnalysisJobStatus.NEEDS_REVIEW.value
            ):
                job_result = json.loads(analysis_job["result_json"])
                corrected_label = "dirty" if confirmed else "clean"
                job_result["human_review"] = {
                    "decision": decision.value,
                    "corrected_label": corrected_label,
                    "reviewer_notes": notes,
                    "reviewed_by": actor,
                    "reviewed_at": reviewed_at,
                }
                connection.execute(
                    """UPDATE inspection_analysis_jobs
                    SET status = ?, result_json = ?, updated_at = ?,
                        reviewed_at = ?, reviewed_by = ?
                    WHERE id = ? AND status = ?""",
                    (
                        InspectionAnalysisJobStatus.RESOLVED.value,
                        json.dumps(job_result, ensure_ascii=False),
                        reviewed_at,
                        reviewed_at,
                        actor,
                        analysis_job["id"],
                        InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                    ),
                )
                self._refresh_photo_analysis_status(
                    connection, str(analysis_job["photo_id"]), reviewed_at
                )
                connection.execute(
                    """UPDATE learning_samples
                    SET status = ?, corrected_label = ?, reviewer_notes = ?,
                        reviewed_at = ?, training_eligible = 1
                    WHERE source_id = ? AND task = ? AND status = ?""",
                    (
                        LearningStatus.APPROVED.value,
                        corrected_label,
                        notes,
                        reviewed_at,
                        analysis_job["id"],
                        LearningTask.INTERIOR_CLEANLINESS.value,
                        LearningStatus.PENDING_REVIEW.value,
                    ),
                )
                self._insert_rental_event(
                    connection,
                    rental_id=str(current["rental_id"]),
                    vehicle_id=str(current["vehicle_id"]),
                    session_id=str(analysis_job["session_id"]),
                    event_type="manual_review_completed",
                    source_type="analysis_job_review",
                    source_id=str(analysis_job["id"]),
                    title="工作人員已看完這次照片",
                    summary=notes,
                    outcome=decision.value,
                    verified=True,
                    metadata={"actor": actor},
                    created_at=reviewed_at,
                )
            row = connection.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        reviewed_alert = self._alert(row) if row else None
        if reviewed_alert is not None and confirmed:
            self.create_work_order(
                reviewed_alert,
                notes="工作人員已確認車內有髒污，請安排清潔。",
                actor=actor,
            )
            from app.services.history_records import apply_blacklist_rule

            with self._connect() as connection:
                reviewed_user_id = self._resolve_user_id(connection, reviewed_alert.rental_id)
            apply_blacklist_rule(self, reviewed_user_id)
        return reviewed_alert

    def create_work_order(
        self,
        alert: Alert,
        assignee: str | None = None,
        notes: str = "",
        actor: str = "ops-prototype",
    ) -> WorkOrder:
        work_order_id = f"WO-{uuid4().hex[:8].upper()}"
        now = _now()
        due_at = _add_business_days(
            datetime.now(timezone.utc), WORK_ORDER_BUSINESS_DAYS
        ).isoformat()
        assignee = assignee or None
        status = "assigned" if assignee else "queued"
        kind = "vehicle_inspection" if alert.type.startswith("damage") else "cleaning"
        default_notes = "請依候選區域完成車身檢查並回報" if kind == "vehicle_inspection" else ""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM work_orders WHERE alert_id = ? ORDER BY created_at LIMIT 1",
                (alert.id,),
            ).fetchone()
            if existing is not None:
                return self._work_order(existing)
            connection.execute(
                """INSERT INTO work_orders
                (id, alert_id, kind, status, priority, assignee, vehicle_id, notes,
                 due_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'urgent', ?, ?, ?, ?, ?, ?)""",
                (
                    work_order_id,
                    alert.id,
                    kind,
                    status,
                    assignee,
                    alert.vehicle_id,
                    notes or default_notes,
                    due_at,
                    now,
                    now,
                ),
            )
            self._record_activity(
                connection,
                entity_type="work_order",
                entity_id=work_order_id,
                action="created",
                actor=actor,
                summary=(
                    f"建立 {alert.vehicle_id} "
                    f"{'驗車' if kind == 'vehicle_inspection' else '清潔'}工單"
                ),
                metadata={
                    "alert_id": alert.id,
                    "assignee": assignee,
                    "due_at": due_at,
                },
            )
            row = connection.execute(
                "SELECT * FROM work_orders WHERE id = ?", (work_order_id,)
            ).fetchone()
        return self._work_order(row)

    def list_work_orders(self, status: str | None = None) -> list[WorkOrder]:
        query = "SELECT * FROM work_orders"
        params: tuple[str, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            steps: dict[str, list[WorkOrderStep]] = {}
            # Only the people-facing steps; row-level audit triggers record data_* actions too.
            for step in connection.execute(
                """SELECT entity_id, action, actor, summary, created_at FROM operations_activity
                WHERE entity_type = 'work_order' AND action IN ('created', 'updated')
                ORDER BY julianday(created_at), id"""
            ).fetchall():
                steps.setdefault(str(step["entity_id"]), []).append(
                    WorkOrderStep(
                        action=step["action"],
                        actor=step["actor"],
                        summary=step["summary"],
                        created_at=step["created_at"],
                    )
                )
        return [
            self._work_order(row).model_copy(update={"history": steps.get(str(row["id"]), [])})
            for row in rows
        ]

    def delete_work_order(self, work_order_id: str, *, actor: str) -> bool:
        """Remove the order; its source alert stays so a new order can be opened later."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM work_orders WHERE id = ?", (work_order_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM work_orders WHERE id = ?", (work_order_id,))
            self._record_activity(
                connection,
                entity_type="work_order",
                entity_id=work_order_id,
                action="deleted",
                actor=actor,
                summary=f"刪除 {row['vehicle_id']} 工單",
                metadata={"alert_id": row["alert_id"], "status": row["status"]},
            )
        return True

    def update_work_order(
        self,
        work_order_id: str,
        *,
        status: str | None = None,
        assignee: str | None = None,
        notes: str | None = None,
        actor: str = "ops-prototype",
    ) -> WorkOrder | None:
        allowed_transitions = {
            "queued": {"assigned", "cancelled"},
            "assigned": {"in_progress", "cancelled"},
            "in_progress": {"completed", "cancelled"},
            "completed": set(),
            "cancelled": set(),
        }
        with self._connect() as connection:
            current_row = connection.execute(
                "SELECT * FROM work_orders WHERE id = ?", (work_order_id,)
            ).fetchone()
            if current_row is None:
                return None

            current = self._work_order(current_row)
            target_status = status or current.status.value
            next_assignee = assignee or current.assignee
            if status and status != current.status.value:
                if status not in allowed_transitions[current.status.value]:
                    raise ValueError(f"工單不能由 {current.status.value} 直接改為 {status}")
                if status in {"assigned", "in_progress"} and not next_assignee:
                    raise ValueError("請先指派負責人員")
            elif not status and assignee and current.status.value == "queued":
                target_status = "assigned"

            changes: list[str] = []
            values: list[str | None] = []
            changed_fields: list[str] = []
            if target_status != current.status.value:
                changes.append("status = ?")
                values.append(target_status)
                changed_fields.append("status")
            if assignee is not None and assignee != current.assignee:
                changes.append("assignee = ?")
                values.append(assignee)
                changed_fields.append("assignee")
            if notes is not None and notes != current.notes:
                changes.append("notes = ?")
                values.append(notes)
                changed_fields.append("notes")
            if target_status == "completed" and current.completed_at is None:
                changes.append("completed_at = ?")
                values.append(_now())

            if changes:
                changes.append("updated_at = ?")
                values.append(_now())
                values.append(work_order_id)
                connection.execute(
                    f"UPDATE work_orders SET {', '.join(changes)} WHERE id = ?", values
                )
                labels = {
                    "queued": "待指派",
                    "assigned": "已指派",
                    "in_progress": "處理中",
                    "completed": "已完成",
                    "cancelled": "已取消",
                }
                summary_parts: list[str] = []
                if "status" in changed_fields:
                    summary_parts.append(f"狀態改為{labels[target_status]}")
                if "assignee" in changed_fields:
                    summary_parts.append(f"指派給 {assignee}")
                if "notes" in changed_fields:
                    summary_parts.append("更新處理備註")
                self._record_activity(
                    connection,
                    entity_type="work_order",
                    entity_id=work_order_id,
                    action="updated",
                    actor=actor,
                    summary="、".join(summary_parts) or "更新工單",
                    metadata={
                        "changed_fields": changed_fields,
                        "from_status": current.status.value,
                        "to_status": target_status,
                    },
                )

                if target_status == "completed":
                    alert_row = connection.execute(
                        "SELECT * FROM alerts WHERE id = ?", (current.alert_id,)
                    ).fetchone()
                    if alert_row is not None and alert_row["status"] != "resolved":
                        connection.execute(
                            "UPDATE alerts SET status = 'resolved', resolved_at = ? WHERE id = ?",
                            (_now(), current.alert_id),
                        )
                        self._record_activity(
                            connection,
                            entity_type="alert",
                            entity_id=current.alert_id,
                            action="auto_resolved",
                            actor=actor,
                            summary=f"工單 {work_order_id} 完成，來源警示自動結案",
                            metadata={"work_order_id": work_order_id},
                        )

            row = connection.execute(
                "SELECT * FROM work_orders WHERE id = ?", (work_order_id,)
            ).fetchone()
        return self._work_order(row) if row else None

    def list_automation_decisions(self, *, limit: int = 100) -> list[AutomationDecision]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM automation_decisions
                ORDER BY created_at DESC, id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._automation_decision(row) for row in rows]

    def automation_summary(self, *, limit: int = 50) -> AutomationSummary:
        settings = get_settings()
        with self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM automation_decisions").fetchone()[0]
            )
            recommended_rows = connection.execute(
                """SELECT recommended_route, COUNT(*) AS count
                FROM automation_decisions GROUP BY recommended_route"""
            ).fetchall()
            effective_rows = connection.execute(
                """SELECT effective_route, COUNT(*) AS count
                FROM automation_decisions GROUP BY effective_route"""
            ).fetchall()
        recommended = {str(row["recommended_route"]): int(row["count"]) for row in recommended_rows}
        effective = {str(row["effective_route"]): int(row["count"]) for row in effective_rows}
        recommended_manual = recommended.get("manual_review", 0)
        effective_manual = effective.get("manual_review", 0)
        reduction = (
            max(0.0, (effective_manual - recommended_manual) / effective_manual)
            if effective_manual
            else 0.0
        )
        return AutomationSummary(
            enabled=settings.automation_routing_enabled,
            shadow_mode=settings.automation_shadow_mode,
            policy_version=settings.automation_policy_version,
            total=total,
            recommended=recommended,
            effective=effective,
            simulated_review_reduction_rate=round(reduction, 4),
            effective_manual_review_rate=(round(effective_manual / total, 4) if total else 0.0),
            decisions=self.list_automation_decisions(limit=limit),
        )

    @staticmethod
    def _has_customer_id(connection: sqlite3.Connection) -> bool:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'auth_accounts'"
        ).fetchone()
        if table is None:
            return False
        return "customer_id" in {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(auth_accounts)").fetchall()
        }

    @classmethod
    def _history_rental_ids(cls, connection: sqlite3.Connection, user_id: str) -> list[str]:
        rental_ids: set[str] = set()
        if cls._has_customer_id(connection):
            rows = connection.execute(
                """SELECT rental_id FROM auth_accounts
                WHERE kind = 'rental'
                  AND (customer_id = ? OR rental_id = ? OR subject = ?)
                  AND rental_id IS NOT NULL""",
                (user_id, user_id, user_id),
            ).fetchall()
            rental_ids.update(str(row["rental_id"]) for row in rows)
        rows = connection.execute(
            """SELECT DISTINCT rental_id FROM rental_events
            WHERE user_id = ? OR rental_id = ?""",
            (user_id, user_id),
        ).fetchall()
        rental_ids.update(str(row["rental_id"]) for row in rows)
        return sorted(rental_ids)

    def list_rental_events(
        self,
        *,
        user_id: str | None,
        limit: int | None = 200,
        since: str | None = None,
        until: str | None = None,
        event_types: list[str] | None = None,
        event_ids: list[str] | None = None,
    ) -> list[RentalEvent]:
        with self._connect() as connection:
            rental_ids = self._history_rental_ids(connection, user_id) if user_id else []
            clauses = ["(user_id = ? OR rental_id = ?"] if user_id else ["1 = 1"]
            params: list[object] = [user_id, user_id] if user_id else []
            if user_id and rental_ids:
                placeholders = ",".join("?" for _ in rental_ids)
                clauses[0] += f" OR rental_id IN ({placeholders})"
                params.extend(rental_ids)
            if user_id:
                clauses[0] += ")"
            clauses.append("event_type != 'cleaning_liability_recorded'")
            account_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(auth_accounts)")
            }
            rental_date_sql = (
                "(SELECT " + ("CASE WHEN rental_events.stage = 'pickup' THEN rental_date "
                "ELSE COALESCE(return_date, rental_date) END"
                if "return_date" in account_columns else "rental_date") +
                " FROM auth_accounts WHERE kind = 'rental' "
                "AND auth_accounts.rental_id = rental_events.rental_id)"
                if "rental_date" in account_columns else "NULL"
            )
            date_sql = (
                "COALESCE(json_extract(metadata_json, '$.occurred_on'), "
                "CASE WHEN event_type IN ('damage_confirmed', 'cleanliness_confirmed', "
                "'cleanliness_alert_created') THEN "
                "(SELECT json_extract(linked.metadata_json, '$.occurred_on') "
                "FROM rental_events AS linked WHERE linked.source_id = rental_events.source_id "
                "AND linked.rental_id = rental_events.rental_id AND "
                "((rental_events.event_type = 'damage_confirmed' "
                "AND linked.source_type = 'damage_history_record') OR "
                "(rental_events.event_type IN ('cleanliness_confirmed', 'cleanliness_alert_created') "
                "AND linked.source_type = 'litter_history_record')) LIMIT 1) END, "
                f"{rental_date_sql}, date(created_at, '+8 hours'))"
            )
            if since:
                clauses.append(f"{date_sql} >= ?")
                params.append(since[:10])
            if until:
                clauses.append(f"{date_sql} < ?")
                params.append(until[:10])
            if event_types:
                placeholders = ",".join("?" for _ in event_types)
                clauses.append(f"event_type IN ({placeholders})")
                params.extend(event_types)
            if event_ids is not None:
                # Re-reads events named by a client, keeping the computed date and litter flags.
                clauses.append(f"id IN ({','.join('?' for _ in event_ids) or 'NULL'})")
                params.extend(event_ids)
            if limit is not None:
                params.append(limit)
            rows = connection.execute(
                f"""SELECT *, {date_sql} AS effective_date,
                (SELECT evidence_json FROM alerts WHERE alerts.id = rental_events.source_id
                    AND alerts.type = 'cleanliness') AS litter_evidence_json,
                (SELECT review_decision FROM alerts WHERE alerts.id = rental_events.source_id
                    AND alerts.type = 'cleanliness') AS litter_review_decision
                FROM rental_events
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at DESC, id DESC {"LIMIT ?" if limit is not None else ""}""",
                params,
            ).fetchall()
        events = []
        for row in rows:
            event = self._rental_event(row)
            metadata = dict(event.metadata)
            if row["litter_evidence_json"] is not None:
                metadata["litter_present"] = any(
                    item.get("category") == "trash" and item.get("visible") is True
                    for item in json.loads(row["litter_evidence_json"])
                )
                metadata["litter_dismissed"] = row["litter_review_decision"] == "dismissed"
            events.append(event.model_copy(update={
                "occurred_on": date.fromisoformat(row["effective_date"]), "metadata": metadata,
            }))
        return events

    def get_user_history_profile(
        self, user_id: str, *, event_limit: int = 200
    ) -> UserHistoryProfile | None:
        all_events = self.list_rental_events(user_id=user_id, limit=None)
        events = all_events[:event_limit]
        with self._connect() as connection:
            rental_ids = self._history_rental_ids(connection, user_id)
            display_name = user_id
            if self._has_customer_id(connection):
                row = connection.execute(
                    """SELECT display_name, customer_id FROM auth_accounts
                    WHERE kind = 'rental'
                      AND (customer_id = ? OR rental_id = ? OR subject = ?)
                    ORDER BY updated_at DESC, rental_id LIMIT 1""",
                    (user_id, user_id, user_id),
                ).fetchone()
                if row is not None:
                    display_name = str(row["display_name"] or "使用者")
                    user_id = str(row["customer_id"] or user_id)
                else:
                    # Retain historical events, but deleted accounts are no longer selectable.
                    return None
            if not events and not rental_ids:
                return None
            scope_rentals = rental_ids or [user_id]
            scope_placeholders = ",".join("?" for _ in scope_rentals)
            scope_sql = (f"(user_id = ? OR rental_id IN ({scope_placeholders})) "
                         "AND event_type != 'cleaning_liability_recorded'")
            scope_params = (user_id, *scope_rentals)
            totals = connection.execute(
                f"""SELECT COUNT(*) AS event_count,
                    SUM(CASE WHEN event_type = 'damage_confirmed' AND verified = 1
                             THEN 1 ELSE 0 END) AS confirmed_damage,
                    SUM(CASE WHEN event_type = 'cleanliness_confirmed' AND verified = 1
                             THEN 1 ELSE 0 END) AS confirmed_cleanliness,
                    SUM(CASE WHEN disputed = 1 THEN 1 ELSE 0 END) AS disputed,
                    COUNT(DISTINCT CASE WHEN event_type = 'capture_completed'
                                        THEN rental_id END) AS completed_rentals
                FROM rental_events WHERE {scope_sql}""",
                scope_params,
            ).fetchone()
            vehicle_ids = [
                str(row["vehicle_id"])
                for row in connection.execute(
                    f"""SELECT DISTINCT vehicle_id FROM rental_events
                    WHERE {scope_sql} ORDER BY vehicle_id""",
                    scope_params,
                ).fetchall()
            ]
            history_rental_ids = sorted(set(rental_ids) | {user_id})
            placeholders = ",".join("?" for _ in history_rental_ids)
            pending_review = int(
                connection.execute(
                    f"""SELECT COUNT(*) FROM inspection_analysis_jobs AS jobs
                    JOIN inspection_sessions AS sessions ON sessions.id = jobs.session_id
                    WHERE jobs.status = ?
                      AND sessions.rental_id IN ({placeholders})""",
                    (
                        InspectionAnalysisJobStatus.NEEDS_REVIEW.value,
                        *history_rental_ids,
                    ),
                ).fetchone()[0]
            )
        all_rentals = rental_ids
        confirmed_damage = len({
            event.metadata.get("damage_case_id") or event.source_id
            for event in all_events
            if event.verified and event.event_type in
            {"damage_confirmed", "damage_liability_recorded"}
        })
        confirmed_cleanliness = len({
            event.metadata.get("cleanliness_alert_id") or event.source_id
            for event in all_events
            if event.verified and event.event_type in
            {"cleanliness_confirmed", "litter_recorded"}
        })
        disputed = int(totals["disputed"] or 0)
        incident_count = confirmed_damage + confirmed_cleanliness
        exposure = max(int(totals["completed_rentals"] or 0), len(all_rentals))
        risk_reasons: list[str] = []
        if exposure < 2:
            risk_level = "insufficient_data"
            risk_reasons.append("完成租借樣本不足，不進行風險分級")
        elif incident_count == 0:
            risk_level = "low"
            risk_reasons.append("目前沒有已確認的新增車損或髒污事件")
        elif incident_count >= 2 or incident_count / exposure >= 0.5:
            risk_level = "high"
            risk_reasons.append("已確認事件相對租借次數偏高，需檢視個別證據")
        else:
            risk_level = "watch"
            risk_reasons.append("存在已確認事件，建議查看租單與證據")
        if disputed:
            risk_reasons.append("存在爭議事件，爭議結案前不得作不利處分")
        return UserHistoryProfile(
            user_id=user_id,
            display_name=display_name,
            rental_ids=all_rentals,
            vehicle_ids=vehicle_ids,
            rental_count=exposure,
            event_count=int(totals["event_count"]),
            confirmed_damage_count=confirmed_damage,
            confirmed_cleanliness_count=confirmed_cleanliness,
            pending_review_count=pending_review,
            disputed_count=disputed,
            last_event_at=events[0].created_at if events else None,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            events=events,
            **self.damage_history_statistics(user_id, events=all_events),
        )

    def damage_history_statistics(
        self, user_id: str, *, events: list[RentalEvent] | None = None,
    ) -> dict:
        from app.services.history_records import compensation_statistics, litter_statistics

        events = events if events is not None else self.list_rental_events(user_id=user_id, limit=None)
        statistics = {**compensation_statistics(events), **litter_statistics(events)}
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM history_blacklist WHERE user_id = ?", (user_id,)
            ).fetchone()
        statistics.update(
            blacklisted=bool(row["blacklisted"]) if row else False,
            blacklist_reason=str(row["reason"]) if row else "",
        )
        return statistics

    def list_history_users(
        self, *, search: str | None = None, limit: int | None = 50
    ) -> list[UserHistoryListItem]:
        candidates: set[str] = set()
        with self._connect() as connection:
            if self._has_customer_id(connection):
                query = """SELECT DISTINCT COALESCE(customer_id, rental_id) AS user_id
                    FROM auth_accounts WHERE kind = 'rental'"""
                params: tuple[object, ...] = ()
                if search:
                    query += (
                        " AND (display_name LIKE ? OR customer_id LIKE ? "
                        "OR rental_id LIKE ? OR vehicle_id LIKE ?)"
                    )
                    pattern = f"%{search}%"
                    params = (pattern, pattern, pattern, pattern)
                rows = connection.execute(query, params).fetchall()
                candidates.update(str(row["user_id"]) for row in rows if row["user_id"])
            if self._has_customer_id(connection):
                event_query = """SELECT DISTINCT
                    COALESCE(accounts.customer_id, accounts.rental_id) AS user_id
                    FROM rental_events AS events
                    JOIN auth_accounts AS accounts
                      ON accounts.kind = 'rental' AND accounts.rental_id = events.rental_id"""
                event_columns = "events.user_id", "events.rental_id", "events.vehicle_id"
            else:
                event_query = "SELECT DISTINCT user_id FROM rental_events"
                event_columns = "user_id", "rental_id", "vehicle_id"
            event_params: tuple[object, ...] = ()
            if search:
                event_query += " WHERE " + " OR ".join(
                    f"{column} LIKE ?" for column in event_columns
                )
                pattern = f"%{search}%"
                event_params = (pattern, pattern, pattern)
            rows = connection.execute(event_query, event_params).fetchall()
            candidates.update(str(row["user_id"]) for row in rows)
        profiles = [
            self.get_user_history_profile(candidate, event_limit=1) for candidate in candidates
        ]
        items = [
            UserHistoryListItem.model_validate(
                profile.model_dump(
                    exclude={"events", "disputed_count", "risk_level", "risk_reasons"}
                )
            )
            for profile in profiles
            if profile is not None
        ]
        items.sort(
            key=lambda item: item.last_event_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return items[:limit]

    def clear_damage_review_duplicates(self) -> dict[str, int]:
        """Remove review cases whose photo pair already has confirmed human truth."""
        with self._connect() as connection:
            cases = connection.execute(
                "SELECT id, learning_sample_id FROM damage_cases"
            ).fetchall()
            case_ids = {str(row["id"]) for row in cases}
            cases_by_sample = {
                str(row["learning_sample_id"]): str(row["id"])
                for row in cases if row["learning_sample_id"]
            }
            samples = connection.execute(
                """SELECT id, source_id, status, primary_image_path, secondary_image_path
                FROM learning_samples WHERE task = ?""",
                (LearningTask.DAMAGE_PAIR.value,),
            ).fetchall()
            confirmed_statuses = {
                LearningStatus.APPROVED.value, LearningStatus.TRAINED.value
            }
            storage_root = self.learning_storage_path.resolve()
            sample_pairs: list[tuple[sqlite3.Row, str]] = []
            confirmed_pairs: set[str] = set()
            targets: set[str] = set()
            for sample in samples:
                if sample["status"] in confirmed_statuses:
                    if sample["source_id"] in case_ids:
                        targets.add(str(sample["source_id"]))
                    if sample["id"] in cases_by_sample:
                        targets.add(cases_by_sample[str(sample["id"])])
                if not sample["secondary_image_path"]:
                    continue
                primary = Path(sample["primary_image_path"]).resolve()
                secondary = Path(sample["secondary_image_path"]).resolve()
                if storage_root not in primary.parents or storage_root not in secondary.parents:
                    continue
                try:
                    digest = hashlib.sha256()
                    digest.update(primary.read_bytes())
                    digest.update(b"\x00PAIR\x00")
                    digest.update(secondary.read_bytes())
                except OSError:
                    continue
                pair_hash = digest.hexdigest()
                sample_pairs.append((sample, pair_hash))
                if sample["status"] in confirmed_statuses:
                    confirmed_pairs.add(pair_hash)
            for sample, pair_hash in sample_pairs:
                if pair_hash not in confirmed_pairs:
                    continue
                if sample["source_id"] in case_ids:
                    targets.add(str(sample["source_id"]))
                if sample["id"] in cases_by_sample:
                    targets.add(cases_by_sample[str(sample["id"])])
            deleted_jobs = self._delete_damage_cases(connection, targets)
        return {
            "deleted_damage_cases": len(targets),
            "deleted_analysis_jobs": deleted_jobs,
        }

    def clear_pending_damage_review(self) -> dict[str, int]:
        """Remove only the cases listed in the pending review queue.

        Learning samples, source photos, alerts, work orders and human truth
        stay untouched; only each case's own comparison job goes with it.
        """
        targets = {
            case.id
            for case in self.list_damage_cases()
            if case.review_decision == DamageReviewDecision.PENDING
        }
        with self._connect() as connection:
            deleted_jobs = self._delete_damage_cases(connection, targets)
        return {
            "deleted_damage_cases": len(targets),
            "deleted_analysis_jobs": deleted_jobs,
        }

    def _delete_damage_cases(
        self, connection: sqlite3.Connection, targets: set[str]
    ) -> int:
        """Delete damage cases with their comparison jobs; return the job count.

        Jobs are removed rather than detached because startup re-links
        detached jobs from ``result_json`` to the now-missing case id.
        """
        if not targets:
            return 0
        affected_photos: set[str] = set()
        deleted_jobs = 0
        jobs = connection.execute(
            """SELECT id, photo_id, damage_case_id, result_json
            FROM inspection_analysis_jobs WHERE task = ?""",
            (InspectionAnalysisTask.DAMAGE_COMPARISON.value,),
        ).fetchall()
        for job in jobs:
            try:
                result_case_id = json.loads(job["result_json"]).get("damage_case_id")
            except (json.JSONDecodeError, TypeError, AttributeError):
                result_case_id = None
            if job["damage_case_id"] not in targets and result_case_id not in targets:
                continue
            connection.execute(
                "DELETE FROM inspection_analysis_jobs WHERE id = ?", (job["id"],)
            )
            affected_photos.add(str(job["photo_id"]))
            deleted_jobs += 1
        connection.executemany(
            "DELETE FROM damage_cases WHERE id = ?",
            ((case_id,) for case_id in targets),
        )
        updated_at = _now()
        for photo_id in affected_photos:
            self._refresh_photo_analysis_status(connection, photo_id, updated_at)
        return deleted_jobs

    def clear_test_data(self) -> dict[str, int]:
        """Clear prototype operations records in foreign-key-safe order."""
        with self._connect() as connection:
            work_orders = int(connection.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0])
            damage_cases = int(
                connection.execute("SELECT COUNT(*) FROM damage_cases").fetchone()[0]
            )
            alerts = int(connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
            notification_events = int(
                connection.execute("SELECT COUNT(*) FROM notification_outbox").fetchone()[0]
            )
            connection.execute("DELETE FROM notification_outbox")
            connection.execute("DELETE FROM work_orders")
            connection.execute("DELETE FROM damage_cases")
            connection.execute("DELETE FROM alerts")
        return {
            "deleted_alerts": alerts,
            "deleted_work_orders": work_orders,
            "deleted_damage_cases": damage_cases,
            "deleted_activity_events": 0,
            "deleted_notification_events": notification_events,
        }


_store: OperationsStore | None = None
_store_lock = RLock()


def get_operations_store() -> OperationsStore:
    global _store
    with _store_lock:
        if _store is None:
            candidate = OperationsStore()
            candidate.initialize()
            _store = candidate
    return _store
