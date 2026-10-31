from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.annotation_geometry import validate_polygon


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class CleanlinessReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class WorkOrderStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class LearningTask(str, Enum):
    EXTERIOR_VIEW = "exterior_view"
    VEHICLE_COVERAGE = "vehicle_coverage"
    INTERIOR_CAPTURE = "interior_capture"
    INTERIOR_CLEANLINESS = "interior_cleanliness"
    VISIBLE_DAMAGE = "visible_damage"
    DAMAGE_PAIR = "damage_pair"
    LICENSE_PLATE = "license_plate"
    CAPTURE_SHARPNESS = "capture_sharpness"


class LearningStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    TRAINED = "trained"


class TrainingRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PROMOTED = "promoted"
    # A trial checkpoint was saved for review; nothing in production changed.
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrainingRunner(str, Enum):
    """How a dataset's automatic training reaches production."""

    DEPLOY = "deploy"  # gated against a fixed test set, then promoted
    CANDIDATE = "candidate"  # saved as a trial model for an administrator
    NONE = "none"  # pretrained model; confirmations are records only


class NotificationDeliveryStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    POWER_AUTOMATE_QUEUED = "power_automate_queued"
    POWER_AUTOMATE_FAILED = "power_automate_failed"
    SENT = "sent"
    FAILED = "failed"


class Alert(BaseModel):
    id: str
    type: str
    severity: str
    status: AlertStatus
    rental_id: str
    vehicle_id: str
    title: str
    summary: str
    evidence: list[dict]
    confidence: float
    image_filename: str
    image_url: str | None = None
    created_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    cancelled_at: datetime | None = None
    review_decision: CleanlinessReviewDecision | None = None
    reviewer_notes: str = ""
    reviewed_at: datetime | None = None


class ReviewCleanlinessAlertRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    decision: CleanlinessReviewDecision
    reviewer_notes: str = Field(default="", max_length=500)


class WorkOrderStep(BaseModel):
    action: str
    actor: str
    summary: str
    created_at: datetime


class WorkOrder(BaseModel):
    id: str
    alert_id: str
    kind: str
    status: WorkOrderStatus
    priority: str
    assignee: str | None = None
    vehicle_id: str
    notes: str
    due_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    # What people have done on this order so far, oldest first.
    history: list[WorkOrderStep] = Field(default_factory=list)


class CreateWorkOrderRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    assignee: str | None = Field(default=None, max_length=80)
    notes: str = Field(default="", max_length=500)


class UpdateWorkOrderRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    status: WorkOrderStatus | None = None
    assignee: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=500)


class OperationsActivity(BaseModel):
    id: str
    entity_type: str
    entity_id: str
    action: str
    actor: str
    summary: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class ClearOperationsActivityResponse(BaseModel):
    deleted_activity_events: int
    older_than_days: int
    message: str


class NotificationOutboxItem(BaseModel):
    id: str
    event_key: str
    channel: str
    provider: str
    recipient: str
    subject: str
    body_text: str
    status: NotificationDeliveryStatus
    attempts: int
    max_attempts: int
    last_error: str | None = None
    next_attempt_at: datetime | None = None
    handed_off_at: datetime | None = None
    sent_at: datetime | None = None
    source_type: str
    source_id: str
    created_at: datetime
    updated_at: datetime


class NotificationDispatchResponse(BaseModel):
    processed: int
    handed_off: int = 0
    failed: int
    message: str


class RecognitionJobProgress(BaseModel):
    heartbeat_at: datetime | None = None
    model_pending: bool = False
    id: str
    photo_id: str
    task: str
    status: str
    filename: str
    slot: str
    damage_view: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)


class InspectionPhotoRecord(BaseModel):
    analysis_jobs: list[RecognitionJobProgress] = Field(default_factory=list)
    id: str
    session_id: str
    rental_id: str
    vehicle_id: str
    stage: str
    session_status: str
    slot: str
    # Exterior angle an angle-tagged damage close-up stands in for.
    damage_view: str | None = None
    filename: str
    quality_status: str
    quality_score: int = Field(ge=0, le=100)
    capture_status: str
    analysis_status: str
    image_available: bool
    image_url: str | None = None
    expected_plate: str | None = None
    recognized_plate: str | None = None
    human_verified_plate: str | None = None
    plate_model_uncertain: bool = False
    created_at: datetime


class RecognitionProgress(BaseModel):
    jobs: list[RecognitionJobProgress] = Field(default_factory=list)
    session_id: str
    rental_id: str
    vehicle_id: str
    stage: str
    total: int
    queued: int
    processing: int
    completed: int
    failed: int
    started_at: datetime
    finished_at: datetime | None = None


class DeleteInspectionPhotosResponse(BaseModel):
    deleted_records: int
    deleted_images: int
    message: str


class LearningSample(BaseModel):
    id: str
    task: LearningTask
    status: LearningStatus
    trigger_reasons: list[str]
    primary_filename: str
    secondary_filename: str | None = None
    has_secondary_image: bool = False
    rental_id: str | None = None
    vehicle_id: str | None = None
    source_id: str | None = None
    inspection_stage: str | None = None
    inspection_slot: str | None = None
    expected_label: str | None = None
    predicted_label: str | None = None
    corrected_label: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    score: int | None = Field(default=None, ge=0, le=100)
    lighting: str
    auto_category: str
    model_version: str | None = None
    sample_kind: str = "hard_case"
    training_eligible: bool = False
    reference_enabled: bool = False
    metadata: dict = Field(default_factory=dict)
    reviewer_notes: str = ""
    occurrence_count: int = 1
    created_at: datetime
    last_seen_at: datetime
    reviewed_at: datetime | None = None
    trained_at: datetime | None = None


class LearningAnnotationKind(str, Enum):
    ISSUE = "issue"
    HARD_NEGATIVE = "hard_negative"
    REFERENCE = "reference"


class LearningAnnotationImageSlot(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class LearningAnnotationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)


class LearningRegionGeometry(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    shape: Literal["rectangle", "polygon"] = "rectangle"
    points: list[LearningAnnotationPoint] = Field(default_factory=list, max_length=257)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LearningRegionGeometry":
        if self.shape == "polygon":
            vertices = validate_polygon([(p.x, p.y) for p in self.points])
            self.points = [LearningAnnotationPoint(x=x, y=y) for x, y in vertices]
            xs, ys = zip(*vertices, strict=True)
            self.x, self.y = min(xs), min(ys)
            self.width, self.height = max(xs) - self.x, max(ys) - self.y
        elif self.points:
            raise ValueError("矩形標註不能包含輪廓頂點")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("標記框不能超出影像邊界")
        return self


class LearningOutlineRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original: LearningRegionGeometry
    method: str = Field(max_length=80)
    removed_area_ratio: float = Field(ge=0, le=1)


class LearningRegionAnnotation(LearningRegionGeometry):
    image_slot: LearningAnnotationImageSlot
    kind: LearningAnnotationKind
    label: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=200)
    normal_reference: bool = False
    auto_refinement: LearningOutlineRefinement | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "LearningRegionAnnotation":
        if self.normal_reference and self.kind != LearningAnnotationKind.REFERENCE:
            raise ValueError("只有部位輪廓可以確認為正常外觀參考")
        if self.auto_refinement is not None and self.label not in {
            "vehicle_body", "normal_vehicle_body", "normal_body_line",
        }:
            raise ValueError("自動車身輪廓只適用於車身範圍")
        return self


class LearningVehicleType(str, Enum):
    PASSENGER_CAR = "passenger_car"
    SUV = "suv"
    VAN = "van"


class LearningVehicleIdentity(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    make: str = Field(default="", max_length=80)
    vehicle_type: LearningVehicleType | None = None
    model: str = Field(default="", max_length=80)
    generation: str = Field(default="", max_length=80)
    seat_material: str = Field(default="", max_length=80)
    seat_style: str = Field(default="", max_length=80)
    seat_pattern: str = Field(default="", max_length=80)


class LearningReviewMethod(str, Enum):
    REGION_ANNOTATION = "region_annotation"
    MANUAL_PAIR = "manual_pair"
    MANUAL_SINGLE = "manual_single"


class PairAngleResult(str, Enum):
    SAME_VIEW = "same_view"
    INCOMPATIBLE = "incompatible"


class LearningTrainingApprovalRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    corrected_label: str = Field(min_length=1, max_length=80)
    reviewer_notes: str = Field(default="", max_length=500)
    annotations: list[LearningRegionAnnotation] = Field(
        default_factory=list, max_length=20
    )
    vehicle_identity: LearningVehicleIdentity | None = None
    review_method: LearningReviewMethod | None = None
    pair_angle_result: PairAngleResult | None = None


class ManualTruthDestination(str, Enum):
    TRUTH = "truth"
    PENDING = "pending"


class ManualTruthUploadStatus(str, Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class ManualTruthUploadItem(BaseModel):
    filename: str
    status: ManualTruthUploadStatus
    sample_id: str | None = None
    message: str = ""


class ManualTruthUploadResponse(BaseModel):
    task: LearningTask
    corrected_label: str
    destination: ManualTruthDestination
    inspection_slot: str | None = None
    created: int
    results: list[ManualTruthUploadItem]


class ManualTruthDuplicateCheckRequest(BaseModel):
    task: LearningTask
    # SHA-256 hex digests of single photos, computed in the browser.
    hashes: list[str] = Field(max_length=2000)


class ManualTruthDuplicateCheckResponse(BaseModel):
    duplicates: list[str]


class TrainingRun(BaseModel):
    id: str
    task: LearningTask = LearningTask.EXTERIOR_VIEW
    status: TrainingRunStatus
    trigger: str
    approved_samples: int
    manifest_path: str | None = None
    candidate_path: str | None = None
    report_path: str | None = None
    log_path: str | None = None
    baseline_metrics: dict = Field(default_factory=dict)
    candidate_metrics: dict = Field(default_factory=dict)
    message: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class LearningStats(BaseModel):
    total: int
    pending_review: int
    approved: int
    rejected: int
    trained: int
    by_task: dict[str, int]
    by_lighting: dict[str, int]
    approved_exterior_untrained: int
    next_training_in: int
    auto_train_enabled: bool
    min_approved_for_training: int
    last_training_run: TrainingRun | None = None
    approved_untrained: int = 0
    next_training_task: LearningTask | None = None
    datasets: list["DatasetTrainingStatus"] = Field(default_factory=list)


class DatasetTrainingStatus(BaseModel):
    task: LearningTask
    runner: TrainingRunner
    approved_untrained: int
    trained: int
    threshold: int
    remaining: int
    last_run: TrainingRun | None = None


class DeleteLearningSampleResponse(BaseModel):
    sample_id: str
    deleted_images: int
    message: str


class LearningRentalSummary(BaseModel):
    rental_id: str
    vehicle_id: str
    sample_count: int
    oldest_created_at: datetime
    newest_created_at: datetime


class DeleteLearningRentalResponse(BaseModel):
    rental_id: str
    vehicle_id: str
    deleted_samples: int
    deleted_images: int
    message: str


class ClearTestDataResponse(BaseModel):
    deleted_alerts: int
    deleted_work_orders: int
    deleted_damage_cases: int
    deleted_activity_events: int = 0
    deleted_notification_events: int = 0
    message: str


class ClearDamageReviewDuplicatesResponse(BaseModel):
    deleted_damage_cases: int
    deleted_analysis_jobs: int
    message: str
