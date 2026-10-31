from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from threading import Lock
from time import perf_counter

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import pil_to_tensor

from app.schemas.quality import CheckStatus, CoveragePrediction
from app.services.runtime_settings import get_runtime_recognition_settings

EDGE_NAMES = ("左側", "上方", "右側", "下方")


def touched_edges(edge_margins: Sequence[float], margin: float) -> list[str]:
    """Name the photo edges (left, top, right, bottom) the vehicle box touches."""
    return [name for name, distance in zip(EDGE_NAMES, edge_margins, strict=False) if distance < margin]


def is_cropped(edges: Sequence[str]) -> bool:
    """A vehicle is cropped when it touches two or more edges, except both sides.

    A three-quarter view in a portrait photo is often wider than the frame,
    so reaching the left and right edges together still shows the whole end
    of the car; the required plate proves that end is in the photo.
    """
    return len(edges) >= 2 and set(edges) != {"左側", "右側"}


def edge_margins_from_box(box: Sequence[float]) -> tuple[float, ...]:
    if len(box) != 4:
        return ()
    x1, y1, x2, y2 = box
    return (x1, y1, 1 - x2, 1 - y2)


def classify_vehicle_coverage(
    *,
    area_ratio: float,
    edge_margins: Sequence[float],
) -> CheckStatus:
    """Reject a vehicle only when it is cut off (see ``is_cropped``)."""
    settings = get_runtime_recognition_settings()
    if area_ratio < settings.vehicle_coverage_low_fail:
        return CheckStatus.FAIL
    if area_ratio < settings.vehicle_coverage_low_warning:
        return CheckStatus.WARNING
    if (
        area_ratio > settings.vehicle_coverage_high_fail
        or is_cropped(touched_edges(edge_margins, settings.vehicle_edge_margin_fail))
    ):
        return CheckStatus.FAIL
    if (
        area_ratio > settings.vehicle_coverage_high_warning
        or is_cropped(touched_edges(edge_margins, settings.vehicle_edge_margin_warning))
    ):
        return CheckStatus.WARNING
    return CheckStatus.PASS


class VehicleCoverageDetector:
    def __init__(self) -> None:
        self._model = None
        self._categories: list[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            # Small 320px detections become much slower with excessive CPU threads.
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self._categories = list(weights.meta["categories"])
            self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            self._model.to(self._device).eval()

    def predict(self, content: bytes) -> CoveragePrediction:
        settings = get_runtime_recognition_settings()
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image for vehicle coverage detection") from exc

        self._load()
        width, height = image.size
        tensor = pil_to_tensor(image).float().to(self._device) / 255.0
        started = perf_counter()
        with self._inference_lock, torch.inference_mode():
            output = self._model([tensor])[0]
            if self._device.type == "cuda":
                torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000

        candidates: list[tuple[float, float, torch.Tensor]] = []
        for box, label_index, score in zip(
            output["boxes"], output["labels"], output["scores"], strict=True
        ):
            confidence = float(score.item())
            if confidence < settings.vehicle_detection_confidence_threshold:
                continue
            label = self._categories[int(label_index.item())]
            if label not in {"car", "truck"}:
                continue
            box_width = max(0.0, float(box[2] - box[0]))
            box_height = max(0.0, float(box[3] - box[1]))
            candidates.append((box_width * box_height * confidence, confidence, box))

        if not candidates:
            return CoveragePrediction(
                car_detected=False,
                confidence=0.0,
                bounding_box=[],
                area_ratio=0.0,
                width_ratio=0.0,
                height_ratio=0.0,
                minimum_edge_margin=0.0,
                status=CheckStatus.FAIL,
                latency_ms=round(elapsed_ms, 2),
                model_version="fasterrcnn-mobilenetv3-320-coco-v1",
            )

        _, confidence, box = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = [float(value) for value in box]
        width_ratio = max(0.0, min(1.0, (x2 - x1) / width))
        height_ratio = max(0.0, min(1.0, (y2 - y1) / height))
        area_ratio = width_ratio * height_ratio
        margins = [x1 / width, y1 / height, (width - x2) / width, (height - y2) / height]
        minimum_margin = max(0.0, min(margins))

        status = classify_vehicle_coverage(area_ratio=area_ratio, edge_margins=margins)

        return CoveragePrediction(
            car_detected=True,
            confidence=round(confidence, 4),
            bounding_box=[
                round(x1 / width, 4),
                round(y1 / height, 4),
                round(x2 / width, 4),
                round(y2 / height, 4),
            ],
            area_ratio=round(area_ratio, 4),
            width_ratio=round(width_ratio, 4),
            height_ratio=round(height_ratio, 4),
            minimum_edge_margin=round(minimum_margin, 4),
            status=status,
            latency_ms=round(elapsed_ms, 2),
            model_version="fasterrcnn-mobilenetv3-320-coco-v1",
        )


_detector: VehicleCoverageDetector | None = None


def get_vehicle_coverage_detector() -> VehicleCoverageDetector:
    global _detector
    if _detector is None:
        _detector = VehicleCoverageDetector()
    return _detector
