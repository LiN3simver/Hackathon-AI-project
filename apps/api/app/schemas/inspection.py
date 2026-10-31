from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.quality import ExpectedView, QualityResponse, QualityStatus


class InspectionStage(str, Enum):
    PICKUP = "pickup"
    RETURN = "return"


class InspectionSessionStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    CAPTURE_COMPLETED = "capture_completed"
    CANCELLED = "cancelled"


class InspectionPhotoStatus(str, Enum):
    ACCEPTED = "accepted"
    RETRY_REQUIRED = "retry_required"


class InspectionAnalysisStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class InspectionAnalysisTask(str, Enum):
    PLATE_RECOGNITION = "plate_recognition"
    VISIBLE_DAMAGE = "visible_damage"
    DAMAGE_COMPARISON = "damage_comparison"
    INTERIOR_CLEANLINESS = "interior_cleanliness"


class InspectionAnalysisJobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    RESOLVED = "resolved"
    FAILED = "failed"


class CreateInspectionSessionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    stage: InspectionStage
    restart: bool = False


class ActiveRentalContext(BaseModel):
    rental_id: str
    vehicle_id: str
    vehicle_label: str
    license_plate: str | None = None
    source: str
    pickup_completed: bool = False
    return_allowed: bool = False
    completed_return_session_id: str | None = None
    active_session_id: str | None = None


class InspectionPhoto(BaseModel):
    id: str
    session_id: str
    slot: ExpectedView
    # A damage close-up tagged with an exterior angle stands in for that
    # angle's standard photo and is compared against the same pickup angle.
    damage_view: ExpectedView | None = None
    filename: str
    quality_request_id: str
    quality_status: QualityStatus
    quality_score: int = Field(ge=0, le=100)
    capture_status: InspectionPhotoStatus
    analysis_status: InspectionAnalysisStatus
    image_available: bool = False
    created_at: datetime
    updated_at: datetime


class InspectionAnalysisJob(BaseModel):
    duration_ms: float | None = Field(default=None, ge=0)
    id: str
    photo_id: str
    session_id: str
    task: InspectionAnalysisTask
    status: InspectionAnalysisJobStatus
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    result: dict[str, Any] = Field(default_factory=dict)
    damage_case_id: str | None = None
    error_message: str = ""
    created_at: datetime
    updated_at: datetime
    next_attempt_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None


class InspectionSession(BaseModel):
    id: str
    rental_id: str
    vehicle_id: str
    stage: InspectionStage
    status: InspectionSessionStatus
    required_slots: list[ExpectedView]
    completed_slots: list[ExpectedView]
    missing_slots: list[ExpectedView]
    photos: list[InspectionPhoto]
    created_at: datetime
    updated_at: datetime
    capture_completed_at: datetime | None = None


class InspectionCaptureResponse(BaseModel):
    quality: QualityResponse
    session: InspectionSession
