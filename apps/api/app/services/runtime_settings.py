from __future__ import annotations

import os
import shutil
import sqlite3
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock

from PIL import Image

from app.core.config import get_settings
from app.schemas.admin import (
    ModelQueueSettings,
    RecognitionSettings,
    RecognitionSettingsResponse,
    ReferenceGuide,
)
from app.schemas.quality import ExpectedView
from app.services.activity_audit import install_activity_audit

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VIEW_MODEL_ID = "exterior_view_classifier.pt"
DEFAULT_GUIDES = {
    ExpectedView.LEFT_FRONT: (
        PROJECT_ROOT / "apps" / "web" / "public" / "reference-views" / "left-front.jfif",
        "左前方正確拍攝範本",
        "地下停車場完整拍入車頭與左側車身。",
    ),
    ExpectedView.RIGHT_FRONT: (
        PROJECT_ROOT / "apps" / "web" / "public" / "reference-views" / "right-front.jfif",
        "右前方正確拍攝範本",
        "雨天仍須完整拍入車頭與右側車身。",
    ),
    ExpectedView.LEFT_REAR: (
        PROJECT_ROOT / "apps" / "web" / "public" / "reference-views" / "left-rear.png",
        "左後方正確拍攝範本",
        "完整拍入車尾與左側車身，避免裁切。",
    ),
    ExpectedView.RIGHT_REAR: (
        PROJECT_ROOT / "apps" / "web" / "public" / "reference-views" / "right-rear.png",
        "右後方正確拍攝範本",
        "完整拍入車尾與右側車身，避免裁切。",
    ),
}


def default_recognition_settings() -> RecognitionSettings:
    settings = get_settings()
    return RecognitionSettings(
        active_learning_enabled=settings.active_learning_enabled,
        active_learning_score_threshold=settings.active_learning_score_threshold,
        active_learning_confidence_threshold=settings.active_learning_confidence_threshold,
        active_learning_min_training_score=settings.active_learning_min_training_score,
        active_learning_collect_good_samples=settings.active_learning_collect_good_samples,
        active_learning_auto_train=settings.active_learning_auto_train,
        active_learning_min_approved=settings.active_learning_min_approved,
        active_learning_max_metric_drop=settings.active_learning_max_metric_drop,
        active_learning_min_test_macro_f1=settings.active_learning_min_test_macro_f1,
        active_learning_min_accepted_accuracy=settings.active_learning_min_accepted_accuracy,
        active_learning_max_coverage_drop=settings.active_learning_max_coverage_drop,
        inspection_plate_check_enabled=settings.inspection_plate_check_enabled,
        plate_recognition_confidence_threshold=settings.plate_recognition_confidence_threshold,
        automation_auto_clear_confidence=settings.automation_auto_clear_confidence,
        automation_auto_case_confidence=settings.automation_auto_case_confidence,
    )


class RuntimeSettingsStore:
    def __init__(
        self,
        database_path: Path | None = None,
        guide_storage_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        raw_database = database_path or settings.database_path
        self.database_path = (
            raw_database if raw_database.is_absolute() else PROJECT_ROOT / raw_database
        )
        self.guide_storage_path = (
            guide_storage_path or PROJECT_ROOT / "data" / "reference_guides"
        ).resolve()
        self._lock = RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.guide_storage_path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._initialized:
                return
            connection = self._connect()
            try:
                self._initialize_connection(connection)
                with connection:
                    install_activity_audit(connection)
            finally:
                connection.close()
            self._initialized = True

    def _initialize_connection(self, connection: sqlite3.Connection) -> None:
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reference_guides (
                    view TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    image_path TEXT,
                    filename TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT NOT NULL
                );
                """
            )
            for view, (source, title, description) in DEFAULT_GUIDES.items():
                if not source.is_file():
                    continue
                destination = (self.guide_storage_path / f"{view.value}{source.suffix}").resolve()
                if not destination.is_file():
                    shutil.copy2(source, destination)
                connection.execute(
                    """INSERT OR IGNORE INTO reference_guides
                    (view, title, description, image_path, filename, updated_at, updated_by)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'system-default')""",
                    (
                        view.value,
                        title,
                        description,
                        str(destination),
                        source.name,
                    ),
                )

    def get_recognition_settings(self) -> RecognitionSettingsResponse:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_settings WHERE key = 'recognition'"
            ).fetchone()
        if row is None:
            return RecognitionSettingsResponse(settings=default_recognition_settings())
        try:
            current = RecognitionSettings.model_validate_json(row["payload_json"])
        except (ValueError, TypeError):
            current = default_recognition_settings()
        return RecognitionSettingsResponse(
            settings=current,
            updated_at=row["updated_at"],
            updated_by=row["updated_by"],
        )

    def update_recognition_settings(
        self, settings: RecognitionSettings, *, actor: str
    ) -> RecognitionSettingsResponse:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO runtime_settings (key, payload_json, updated_at, updated_by)
                VALUES ('recognition', ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(key) DO UPDATE SET payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP, updated_by = excluded.updated_by""",
                (settings.model_dump_json(), actor),
            )
        return self.get_recognition_settings()

    def get_model_queue(
        self,
    ) -> tuple[ModelQueueSettings, str | None, str | None]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_settings WHERE key = 'view_model_queue'"
            ).fetchone()
        if row is None:
            return ModelQueueSettings(model_ids=[DEFAULT_VIEW_MODEL_ID]), None, None
        try:
            queue = ModelQueueSettings.model_validate_json(row["payload_json"])
        except (ValueError, TypeError):
            queue = ModelQueueSettings(model_ids=[DEFAULT_VIEW_MODEL_ID])
        return queue, row["updated_at"], row["updated_by"]

    def update_model_queue(
        self, queue: ModelQueueSettings, *, actor: str
    ) -> tuple[ModelQueueSettings, str | None, str | None]:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO runtime_settings (key, payload_json, updated_at, updated_by)
                VALUES ('view_model_queue', ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(key) DO UPDATE SET payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP, updated_by = excluded.updated_by""",
                (queue.model_dump_json(), actor),
            )
        return self.get_model_queue()

    @staticmethod
    def _guide_from_row(row: sqlite3.Row) -> ReferenceGuide:
        image_available = bool(
            row["image_path"] and Path(row["image_path"]).is_file()
        )
        view = ExpectedView(row["view"])
        return ReferenceGuide(
            view=view,
            title=row["title"],
            description=row["description"],
            image_available=image_available,
            image_url=(
                f"/api/v1/admin/reference-guides/{view.value}/image"
                if image_available
                else None
            ),
            filename=row["filename"],
            updated_at=row["updated_at"],
            updated_by=row["updated_by"],
        )

    def list_reference_guides(self) -> list[ReferenceGuide]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reference_guides ORDER BY view"
            ).fetchall()
        return [self._guide_from_row(row) for row in rows]

    def get_reference_guide(self, view: ExpectedView) -> ReferenceGuide | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reference_guides WHERE view = ?", (view.value,)
            ).fetchone()
        return self._guide_from_row(row) if row else None

    def get_reference_guide_path(self, view: ExpectedView) -> Path | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_path FROM reference_guides WHERE view = ?", (view.value,)
            ).fetchone()
        if row is None or not row["image_path"]:
            return None
        candidate = Path(row["image_path"]).resolve()
        if self.guide_storage_path not in candidate.parents or not candidate.is_file():
            return None
        return candidate

    def update_reference_guide(
        self,
        view: ExpectedView,
        *,
        title: str,
        description: str,
        actor: str,
        image_content: bytes | None = None,
        filename: str | None = None,
    ) -> ReferenceGuide:
        self.initialize()
        current_path = self.get_reference_guide_path(view)
        next_path = current_path
        clean_filename = filename
        if image_content is not None:
            with Image.open(BytesIO(image_content)) as image:
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
                    image.format or "", ".jpg"
                )
            next_path = (self.guide_storage_path / f"{view.value}{suffix}").resolve()
            temporary_path = next_path.with_suffix(next_path.suffix + ".tmp")
            temporary_path.write_bytes(image_content)
            os.replace(temporary_path, next_path)
            if current_path and current_path != next_path and current_path.is_file():
                current_path.unlink()

        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO reference_guides
                (view, title, description, image_path, filename, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(view) DO UPDATE SET title = excluded.title,
                    description = excluded.description,
                    image_path = COALESCE(excluded.image_path, reference_guides.image_path),
                    filename = COALESCE(excluded.filename, reference_guides.filename),
                    updated_at = CURRENT_TIMESTAMP, updated_by = excluded.updated_by""",
                (
                    view.value,
                    title.strip(),
                    description.strip(),
                    str(next_path) if next_path else None,
                    clean_filename,
                    actor,
                ),
            )
        guide = self.get_reference_guide(view)
        assert guide is not None
        return guide


@lru_cache
def get_runtime_settings_store() -> RuntimeSettingsStore:
    return RuntimeSettingsStore()


def get_runtime_recognition_settings() -> RecognitionSettings:
    return get_runtime_settings_store().get_recognition_settings().settings
