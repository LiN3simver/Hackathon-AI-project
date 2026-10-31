from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.quality import ExpectedView


class DamageComparisonStatus(str, Enum):
    NO_CHANGE = "no_change"
    REVIEW_REQUIRED = "review_required"
    INCOMPATIBLE = "incompatible"


class DamageReviewDecision(str, Enum):
    PENDING = "pending"
    CONFIRMED_DAMAGE = "confirmed_damage"
    DISMISSED = "dismissed"
    RETRY_PHOTOS = "retry_photos"


class DamagePairDecision(str, Enum):
    CONFIRMED_NEW_DAMAGE = "confirmed_new_damage"
    NO_NEW_DAMAGE = "no_new_damage"
    CANNOT_ASSESS = "cannot_assess"


class VisibleDamageType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    BROKEN_PART = "broken_part"
    PAINT_TRANSFER = "paint_transfer"
    DEFORMATION = "deformation"
    PANEL_MISALIGNMENT = "panel_misalignment"
    OTHER = "other"


class VisibleDamageSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CANNOT_ASSESS = "cannot_assess"


class DamageObservationEvidence(BaseModel):
    damage_type: VisibleDamageType
    location: str
    description: str


class DamageObservation(BaseModel):
    is_vehicle_exterior: bool
    image_usable: bool
    damage_visible: bool
    damage_types: list[VisibleDamageType]
    severity: VisibleDamageSeverity
    confidence: float = Field(ge=0, le=1)
    affected_area: str
    safety_issue_possible: bool
    summary: str
    evidence: list[DamageObservationEvidence]


class DamageInspectionResponse(BaseModel):
    request_id: str
    rental_id: str
    vehicle_id: str
    filename: str
    assessment: DamageObservation
    model: str
    provider: str = "ollama_local"
    latency_ms: float
    alert_id: str | None = None
    human_review_required: bool = True


class DamageCandidate(BaseModel):
    id: str
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    area_ratio: float = Field(ge=0, le=1)
    change_score: float = Field(ge=0, le=1)
    description: str


class DamagePairVerification(BaseModel):
    decision: DamagePairDecision
    confidence: float = Field(ge=0, le=1)
    same_vehicle_and_view: bool
    confirmed_region_ids: list[str]
    summary: str
    evidence: str
    model: str
    latency_ms: float = Field(ge=0)


class DamagePairSpecialistResult(BaseModel):
    """Learned before/after damage model; a detection only routes to review."""

    detected: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    pickup_match_score: float = Field(ge=0, le=1)
    regions: list[DamageCandidate] = Field(default_factory=list)
    annotated_return_image: str | None = None
    model_version: str
    latency_ms: float = Field(ge=0)


class AlignmentResult(BaseModel):
    success: bool
    method: str
    matched_features: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    overlap_ratio: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    message: str


class DamageComparisonResult(BaseModel):
    status: DamageComparisonStatus
    alignment: AlignmentResult
    candidates: list[DamageCandidate]
    # Pixel-difference boxes kept for audit when the damage-pair model's regions
    # replace them as the reviewer-facing candidates.
    geometric_candidates: list[DamageCandidate] = Field(default_factory=list)
    change_area_ratio: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    summary: str
    annotated_return_image: str | None = None
    difference_mask_image: str | None = None
    model_version: str
    latency_ms: float
    verification: DamagePairVerification | None = None
    specialist: DamagePairSpecialistResult | None = None


class DamageCase(BaseModel):
    id: str
    rental_id: str
    vehicle_id: str
    expected_view: ExpectedView
    pickup_filename: str
    return_filename: str
    comparison: DamageComparisonResult
    review_decision: DamageReviewDecision
    reviewer_notes: str
    alert_id: str | None = None
    created_at: datetime
    reviewed_at: datetime | None = None


class DamageReviewRequest(BaseModel):
    decision: DamageReviewDecision
    reviewer_notes: str = Field(default="", max_length=500)
