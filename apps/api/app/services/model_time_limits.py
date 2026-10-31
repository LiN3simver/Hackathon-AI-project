"""Per-model recognition time limits configured in the admin settings.

A limit bounds how long a caller waits. In-process PyTorch inference cannot be
interrupted, so a timed-out call keeps running in its worker thread and its
result is discarded; network models (Ollama) and the OCR subprocess are cut off
at the source instead.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TypeVar

from app.services.runtime_settings import get_runtime_recognition_settings

T = TypeVar("T")


class RecognitionModel(str, Enum):
    VIEW = "view"
    VEHICLE_DETECTION = "vehicle_detection"
    INTERIOR_CAPTURE = "interior_capture"
    PLATE = "plate"
    FRONT_CLEANLINESS = "front_cleanliness"
    REAR_CLEANLINESS = "rear_cleanliness"
    DAMAGE_PAIR = "damage_pair"


MODEL_LABELS = {
    RecognitionModel.VIEW: "四角拍攝角度",
    RecognitionModel.VEHICLE_DETECTION: "車身位置偵測",
    RecognitionModel.INTERIOR_CAPTURE: "車內前後座檢查",
    RecognitionModel.PLATE: "車牌辨識",
    RecognitionModel.FRONT_CLEANLINESS: "前座髒污",
    RecognitionModel.REAR_CLEANLINESS: "後座髒污",
    RecognitionModel.DAMAGE_PAIR: "取還車車損比對",
}
SETTING_FIELDS = {
    RecognitionModel.VIEW: "timeout_view_seconds",
    RecognitionModel.VEHICLE_DETECTION: "timeout_vehicle_detection_seconds",
    RecognitionModel.INTERIOR_CAPTURE: "timeout_interior_capture_seconds",
    RecognitionModel.PLATE: "timeout_plate_seconds",
    RecognitionModel.FRONT_CLEANLINESS: "timeout_front_cleanliness_seconds",
    RecognitionModel.REAR_CLEANLINESS: "timeout_rear_cleanliness_seconds",
    RecognitionModel.DAMAGE_PAIR: "timeout_damage_pair_seconds",
}

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="model-time-limit")
_limit_overrides: ContextVar[dict[RecognitionModel, float] | None] = ContextVar("model_limit_overrides", default=None)


@contextmanager
def override_time_limits(limits: dict[RecognitionModel, float]):
    """Override limits for this request without changing persisted admin settings."""
    token = _limit_overrides.set({**(_limit_overrides.get() or {}), **limits})
    try:
        yield
    finally:
        _limit_overrides.reset(token)


class ModelTimeoutError(RuntimeError):
    def __init__(self, model: RecognitionModel, seconds: float) -> None:
        self.model = model
        self.seconds = seconds
        super().__init__(
            f"{MODEL_LABELS[model]}辨識超過 {seconds:g} 秒的時間上限，已停止等待並改由人工確認。"
        )


def time_limit_seconds(model: RecognitionModel) -> float:
    overrides = _limit_overrides.get()
    if overrides is not None and model in overrides:
        return overrides[model]
    return float(getattr(get_runtime_recognition_settings(), SETTING_FIELDS[model]))


def run_with_time_limit(
    model: RecognitionModel, function: Callable[..., T], *args: object
) -> T:
    seconds = time_limit_seconds(model)
    future = _executor.submit(function, *args)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError as exc:
        raise ModelTimeoutError(model, seconds) from exc
