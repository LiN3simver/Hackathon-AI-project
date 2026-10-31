from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CleanlinessBox(BaseModel):
    """Coordinates relative to the EXIF-corrected image, from 0 to 1."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid cleanliness region")
        return self


class CleanlinessLevel(str, Enum):
    DIRTY = "dirty"
    ORDINARY = "ordinary"
    CLEAN = "clean"
    CANNOT_ASSESS = "cannot_assess"


class InteriorView(str, Enum):
    FRONT = "front"
    REAR = "rear"
    UNKNOWN = "unknown"


class EvidenceCategory(str, Enum):
    TRASH = "trash"
    STAIN = "stain"
    LIQUID = "liquid"
    FOOD = "food"
    PET_HAIR = "pet_hair"
    CLUTTER = "clutter"
    FLOOR = "floor"
    SEAT = "seat"
    OTHER = "other"


class CleanlinessEvidence(BaseModel):
    category: EvidenceCategory
    severity: int = Field(ge=0, le=3)
    description: str
    visible: bool
    bounding_box: CleanlinessBox | None = None


class CleanlinessScreening(BaseModel):
    model_version: str
    score: float | None = None
    threshold: float
    suspected_dirty: bool
    human_review_required: bool
    score_is_probability: bool = False


# Preserve the front model's existing Python imports and response fields.
FrontCleanlinessScreening = CleanlinessScreening


class CleanlinessAssessment(BaseModel):
    is_vehicle_interior: bool
    image_usable: bool
    view: InteriorView
    level: CleanlinessLevel
    confidence: float = Field(ge=0, le=1)
    immediate_cleaning_required: bool
    summary: str
    evidence: list[CleanlinessEvidence] = Field(default_factory=list)
    specialist: CleanlinessScreening | None = None
    localization_status: Literal["not_requested", "completed", "unavailable"] = "not_requested"
    localization_model: str | None = None


class InteriorAnalysisResponse(BaseModel):
    request_id: str
    rental_id: str
    vehicle_id: str
    filename: str
    assessment: CleanlinessAssessment
    model: str
    latency_ms: float
    alert_id: str | None = None
    human_review_required: bool
