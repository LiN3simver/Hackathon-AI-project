from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.plate import PlatePrediction


class QualityStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class ExpectedView(str, Enum):
    LEFT_FRONT = "left_front"
    RIGHT_FRONT = "right_front"
    LEFT_REAR = "left_rear"
    RIGHT_REAR = "right_rear"
    INTERIOR_FRONT = "interior_front"
    INTERIOR_REAR = "interior_rear"
    DAMAGE_DETAIL = "damage_detail"


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    PENDING_MODEL = "pending_model"


class QualityCheck(BaseModel):
    key: str
    label: str
    status: CheckStatus
    measured_value: float | str | None = None
    threshold: str | None = None
    message: str


class ImageMetrics(BaseModel):
    width: int
    height: int
    megapixels: float
    blur_variance: float
    brightness: float
    dynamic_range: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    orientation: str


class ViewPrediction(BaseModel):
    predicted_view: ExpectedView
    confidence: float = Field(ge=0, le=1)
    confidence_threshold: float = Field(ge=0, le=1)
    is_confident: bool
    expected_view: ExpectedView | None = None
    matches_expected: bool | None = None
    latency_ms: float
    model_version: str


class CoveragePrediction(BaseModel):
    car_detected: bool
    confidence: float = Field(ge=0, le=1)
    bounding_box: list[float]
    area_ratio: float = Field(ge=0, le=1)
    width_ratio: float = Field(ge=0, le=1)
    height_ratio: float = Field(ge=0, le=1)
    minimum_edge_margin: float = Field(ge=0, le=1)
    status: CheckStatus
    latency_ms: float
    model_version: str


class InteriorCapturePrediction(BaseModel):
    scene_label: str
    scene_confidence: float = Field(ge=0, le=1)
    scene_confidence_threshold: float = Field(ge=0, le=1)
    scene_confident: bool
    matches_expected: bool
    specialist_probability: float | None = Field(default=None, ge=0, le=1)
    specialist_threshold: float | None = Field(default=None, ge=0, le=1)
    rotation_probability: float | None = Field(default=None, ge=0, le=1)
    rotation_threshold: float | None = Field(default=None, ge=0, le=1)
    rotation_vetoed: bool = False
    latency_ms: float
    model_version: str


class ExteriorScenePrediction(BaseModel):
    front_interior_probability: float = Field(ge=0, le=1)
    rear_interior_probability: float = Field(ge=0, le=1)
    interior_probability: float = Field(ge=0, le=1)
    veto_threshold: float = Field(ge=0, le=1)
    is_interior: bool
    latency_ms: float
    model_version: str


class BlurAssessment(BaseModel):
    """Hand-shake blur model score for exterior four-corner photos."""

    probability: float = Field(ge=0, le=1)
    fail_threshold: float = Field(ge=0, le=1)
    warning_threshold: float = Field(ge=0, le=1)
    latency_ms: float
    model_version: str


class QualityResponse(BaseModel):
    recognition_ms: float | None = Field(default=None, ge=0)
    request_id: str
    filename: str
    expected_view: ExpectedView | None = None
    status: QualityStatus
    score: int = Field(ge=0, le=100)
    summary: str
    retake_required: bool
    checks: list[QualityCheck]
    metrics: ImageMetrics
    model_checks_ready: bool = False
    view_prediction: ViewPrediction | None = None
    coverage_prediction: CoveragePrediction | None = None
    interior_capture_prediction: InteriorCapturePrediction | None = None
    exterior_scene_prediction: ExteriorScenePrediction | None = None
    plate_prediction: PlatePrediction | None = None
    blur_assessment: BlurAssessment | None = None
