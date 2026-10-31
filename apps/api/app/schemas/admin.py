from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.quality import ExpectedView


class RecognitionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_min_short_edge: int = Field(default=540, ge=240, le=4096)
    quality_blur_fail: float = Field(default=35.0, ge=0, le=1000)
    quality_blur_warning: float = Field(default=80.0, ge=0, le=2000)
    quality_brightness_low_fail: float = Field(default=25.0, ge=0, le=255)
    quality_brightness_low_warning: float = Field(default=45.0, ge=0, le=255)
    quality_brightness_high_warning: float = Field(default=220.0, ge=0, le=255)
    quality_brightness_high_fail: float = Field(default=242.0, ge=0, le=255)
    quality_min_dynamic_range: float = Field(default=30.0, ge=0, le=255)
    quality_max_clipped_ratio: float = Field(default=0.45, ge=0, le=1)
    # Hand-shake blur model on exterior four-corner photos (probability 0-1).
    blur_model_enabled: bool = True
    blur_model_fail_probability: float = Field(default=0.75, ge=0, le=1)
    blur_model_warning_probability: float = Field(default=0.50, ge=0, le=1)

    view_confidence_threshold: float = Field(default=0.90, ge=0, le=1)
    vehicle_detection_confidence_threshold: float = Field(default=0.50, ge=0, le=1)
    vehicle_coverage_low_fail: float = Field(default=0.07, ge=0, le=1)
    vehicle_coverage_low_warning: float = Field(default=0.13, ge=0, le=1)
    vehicle_coverage_high_warning: float = Field(default=0.82, ge=0, le=1)
    vehicle_coverage_high_fail: float = Field(default=0.90, ge=0, le=1)
    vehicle_edge_margin_warning: float = Field(default=0.002, ge=0, le=0.25)
    vehicle_edge_margin_fail: float = Field(default=0.001, ge=0, le=0.25)

    interior_review_confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    interior_alert_confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    damage_alert_confidence_threshold: float = Field(default=0.85, ge=0, le=1)
    damage_review_confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    damage_pair_confirm_confidence_threshold: float = Field(default=0.95, ge=0, le=1)
    damage_pair_no_change_confidence_threshold: float = Field(default=0.95, ge=0, le=1)
    plate_recognition_confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    inspection_plate_check_enabled: bool = True
    automation_auto_clear_confidence: float = Field(default=0.95, ge=0, le=1)
    automation_auto_case_confidence: float = Field(default=0.90, ge=0, le=1)

    interior_scene_veto_confidence: float = Field(default=0.95, ge=0.5, le=1)
    # Colour-anomaly fast lane: boxes litter in ~50 ms before any vision-language grounding.
    cleanliness_fast_lane_enabled: bool = True
    cleanliness_fast_lane_color_threshold: float = Field(default=20.0, ge=5, le=60)
    cleanliness_fast_lane_min_area_ratio: float = Field(default=0.004, ge=0.001, le=0.1)
    # Ask the slow grounding model only when the seat model already suspects dirt.
    cleanliness_vlm_only_when_suspected: bool = True
    plate_hint_min_width_px: int = Field(default=80, ge=10, le=1000)
    plate_hint_min_height_px: int = Field(default=20, ge=5, le=400)
    plate_hint_min_sharpness: float = Field(default=35.0, ge=0, le=1000)

    # Seconds each model may take before the photo is marked as timed out.
    timeout_view_seconds: float = Field(default=20, ge=1, le=600)
    timeout_vehicle_detection_seconds: float = Field(default=20, ge=1, le=600)
    timeout_interior_capture_seconds: float = Field(default=20, ge=1, le=600)
    timeout_plate_seconds: float = Field(default=5, ge=1, le=600)
    timeout_front_cleanliness_seconds: float = Field(default=120, ge=1, le=600)
    timeout_rear_cleanliness_seconds: float = Field(default=120, ge=1, le=600)
    timeout_damage_pair_seconds: float = Field(default=60, ge=1, le=600)


    active_learning_enabled: bool = True
    active_learning_score_threshold: int = Field(default=90, ge=0, le=100)
    active_learning_confidence_threshold: float = Field(default=0.90, ge=0, le=1)
    active_learning_min_training_score: int = Field(default=90, ge=0, le=100)
    active_learning_collect_good_samples: bool = False
    active_learning_auto_train: bool = True
    # Human truths a dataset needs before it trains itself. The original field
    # stays the exterior-view threshold so saved settings keep validating.
    active_learning_min_approved: int = Field(default=20, ge=1, le=10000)
    auto_train_min_vehicle_coverage: int = Field(default=20, ge=1, le=10000)
    auto_train_min_interior_capture: int = Field(default=20, ge=1, le=10000)
    auto_train_min_damage_pair: int = Field(default=20, ge=1, le=10000)
    active_learning_max_metric_drop: float = Field(default=0.005, ge=0, le=1)
    active_learning_min_test_macro_f1: float = Field(default=0.85, ge=0, le=1)
    active_learning_min_accepted_accuracy: float = Field(default=0.90, ge=0, le=1)
    active_learning_max_coverage_drop: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_retired_model_settings(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        previous_timeout = values.pop("timeout_interior_cleanliness_seconds", None)
        if previous_timeout is not None:
            values.setdefault("timeout_front_cleanliness_seconds", previous_timeout)
            values.setdefault("timeout_rear_cleanliness_seconds", previous_timeout)
        for field in (
            "vision_keep_alive_minutes", "vision_max_output_tokens",
            "cleanliness_image_max_edge", "damage_image_max_edge",
            "auto_train_min_interior_cleanliness", "auto_train_min_visible_damage",
            "timeout_visible_damage_seconds", "timeout_plate_background_seconds",
        ):
            values.pop(field, None)
        return values

    @model_validator(mode="after")
    def validate_ranges(self) -> "RecognitionSettings":
        if self.quality_blur_fail > self.quality_blur_warning:
            raise ValueError("清晰度失敗門檻不可高於警告門檻")
        if self.blur_model_warning_probability > self.blur_model_fail_probability:
            raise ValueError("手晃模糊提醒門檻不可高於要求重拍門檻")
        if not (
            self.quality_brightness_low_fail
            <= self.quality_brightness_low_warning
            < self.quality_brightness_high_warning
            <= self.quality_brightness_high_fail
        ):
            raise ValueError("曝光門檻順序必須為低失敗、低警告、高警告、高失敗")
        if not (
            self.vehicle_coverage_low_fail
            <= self.vehicle_coverage_low_warning
            < self.vehicle_coverage_high_warning
            <= self.vehicle_coverage_high_fail
        ):
            raise ValueError("車身覆蓋率門檻順序不正確")
        if self.vehicle_edge_margin_fail > self.vehicle_edge_margin_warning:
            raise ValueError("貼邊失敗門檻不可高於貼邊警告門檻")
        return self


class RecognitionSettingsResponse(BaseModel):
    settings: RecognitionSettings
    updated_at: datetime | None = None
    updated_by: str | None = None


class ManagedModelSource(str, Enum):
    DEPLOYED = "deployed"
    AUTO_TRAINED = "auto_trained"
    MANUAL_TRAINED = "manual_trained"
    HISTORY = "history"


class ManagedModelKind(str, Enum):
    EXTERIOR_VIEW = "exterior_view"
    INTERIOR_FRONT = "interior_front"
    INTERIOR_REAR = "interior_rear"
    INTERIOR_FRONT_CLEANLINESS = "interior_front_cleanliness"
    INTERIOR_REAR_CLEANLINESS = "interior_rear_cleanliness"
    DAMAGE_PAIR = "damage_pair"
    VEHICLE_COVERAGE = "vehicle_coverage"
    INTERIOR_CAPTURE = "interior_capture"


class ManagedModelStage(str, Enum):
    DEPLOYED = "deployed"
    FIRST_VERSION = "first_version"
    TRIAL = "trial"
    HISTORY = "history"


class ManagedModelMetric(BaseModel):
    """One evaluation score shown beside a model, with the target it is judged against."""

    key: str
    label: str
    value: float
    unit: Literal["percent", "seconds", "score"]
    target: float | None = None
    higher_is_better: bool = True
    passed: bool | None = None


class ManagedViewModel(BaseModel):
    id: str
    version: str
    architecture: str
    kind: ManagedModelKind
    stage: ManagedModelStage
    source: ManagedModelSource
    is_current: bool
    quality_score: float | None = Field(default=None, ge=0, le=1)
    minimum_quality_score: float | None = Field(default=None, ge=0, le=1)
    quality_metric: str | None = None
    label_source: str | None = None
    production_approved: bool = False
    # Mean wall time per recognition on the deployment machine, when benchmarked.
    average_latency_ms: float | None = None
    latency_breakdown_ms: dict[str, float] | None = None
    # Damage-pair models: every recorded score, the backbones used, and whether the
    # file is what online inference currently loads (a candidate may be a copy of it).
    evaluation: list[ManagedModelMetric] = Field(default_factory=list)
    backbone_summary: str | None = None
    in_use: bool = False
    can_activate: bool = False
    # Reviewer ranking stored in the checkpoint (1 = recommended first) and why.
    recommendation_rank: int | None = None
    recommendation_note: str | None = None
    file_size_bytes: int
    modified_at: datetime
    sha256: str


class ModelQueueSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_ids: list[str] = Field(min_length=1, max_length=3)

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("模型順位不可為空")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("同一個模型不可重複排入候選隊列")
        return cleaned


class ModelManagementResponse(BaseModel):
    models: list[ManagedViewModel]
    queue: ModelQueueSettings
    updated_at: datetime | None = None
    updated_by: str | None = None


class DeleteManagedModelRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    confirmation_model_id: str = Field(min_length=1, max_length=500)
    admin_password: str = Field(min_length=1, max_length=120)


class DeleteManagedModelResponse(BaseModel):
    deleted_model_id: str
    message: str
    management: ModelManagementResponse


class ActivateManagedModelRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    confirmation_model_id: str = Field(min_length=1, max_length=500)
    admin_password: str = Field(min_length=1, max_length=120)


class ActivateManagedModelResponse(BaseModel):
    activated_model_id: str
    backup_model_id: str | None
    message: str
    management: ModelManagementResponse


class ReferenceGuide(BaseModel):
    view: ExpectedView
    title: str
    description: str
    image_available: bool
    image_url: str | None = None
    filename: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
