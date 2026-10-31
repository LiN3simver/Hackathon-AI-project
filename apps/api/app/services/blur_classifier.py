"""Hand-shake blur classifier for exterior four-corner photos.

A global Laplacian score is easily fooled at night: sensor noise, street lights
and parking lines keep it high while the car itself is smeared. This model
measures sharpness inside the detected vehicle box, edge steepness only where
edges exist (so smooth white panels do not count as blur), per-direction
softness (one-way hand shake) and an echo term for doubled edges, then combines
them with a logistic regression trained on synthetic blur
(scripts/train_blur_detector.py). Inference needs only OpenCV and numpy.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT / "models" / "blur_detector.json"
WORKING_SHORT_EDGE = 720
MIN_CROP_PIXELS = 64
MIN_EDGE_PIXELS = 150

BASE_FEATURE_NAMES = (
    "lap_raw", "lap_den", "lap_half", "lap_drop", "tile_p50", "tile_p90", "tile_max",
    "cr_v", "cr_h", "cr_max", "cr_aniso", "coherence", "edge_slope", "noise", "brightness", "echo",
)
EDGE_FEATURE_NAMES = (
    "acut_med", "acut_p25", "acut_p75", "acut_dir_min", "acut_dir_max", "acut_dir_ratio",
    "lapnorm_med", "edge_density", "acut_top10",
)
# Whole-image features kept next to the vehicle-box features.
GLOBAL_BASE_KEEP = (0, 1, 6, 13, 14, 15)
GLOBAL_EDGE_KEEP = (0, 5, 7, 8)
FEATURE_NAMES = (
    tuple(f"veh_{name}" for name in BASE_FEATURE_NAMES)
    + tuple(f"all_{BASE_FEATURE_NAMES[i]}" for i in GLOBAL_BASE_KEEP)
    + tuple(f"veh_{name}" for name in EDGE_FEATURE_NAMES)
    + tuple(f"all_{EDGE_FEATURE_NAMES[i]}" for i in GLOBAL_EDGE_KEEP)
)


def decode_working_image(content: bytes) -> np.ndarray:
    """Decode (EXIF-rotated) and shrink so the short edge is at most 720 px."""
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("無法解析圖片")
    return resize_to_working(image)


def resize_to_working(image: np.ndarray) -> np.ndarray:
    scale = WORKING_SHORT_EDGE / min(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def crop_to_box(image: np.ndarray, box: Sequence[float] | None) -> np.ndarray:
    if not box or len(box) != 4:
        return image
    height, width = image.shape[:2]
    x1, y1 = int(box[0] * width), int(box[1] * height)
    x2, y2 = int(box[2] * width), int(box[3] * height)
    if (x2 - x1) < MIN_CROP_PIXELS or (y2 - y1) < MIN_CROP_PIXELS:
        return image
    return image[y1:y2, x1:x2]


def _crete_roffet(gray: np.ndarray) -> tuple[float, float]:
    """No-reference blur metric per direction: re-blurring a blurry photo changes little."""
    values = gray.astype(np.float32)
    result = []
    for axis in (0, 1):
        kernel = np.ones((9, 1), np.float32) / 9 if axis == 0 else np.ones((1, 9), np.float32) / 9
        blurred = cv2.filter2D(values, -1, kernel)
        d_original = np.abs(np.diff(values, axis=axis))
        d_blurred = np.abs(np.diff(blurred, axis=axis))
        variation = np.maximum(0, d_original - d_blurred)
        total = d_original.sum()
        result.append(float((total - variation.sum()) / max(total, 1e-6)))
    return result[0], result[1]


def _echo(gray: np.ndarray, min_radius: int = 3, max_radius: int = 24) -> float:
    """Secondary autocorrelation peak of the high-pass image: doubled (ghost) edges."""
    values = gray.astype(np.float32)
    high_pass = values - cv2.GaussianBlur(values, (0, 0), 2.0)
    high_pass -= high_pass.mean()
    autocorr = np.fft.irfft2(np.abs(np.fft.rfft2(high_pass)) ** 2, s=high_pass.shape)
    autocorr = np.fft.fftshift(autocorr) / (autocorr[0, 0] + 1e-9)
    cy, cx = np.array(autocorr.shape) // 2
    window = autocorr[cy - max_radius:cy + max_radius + 1, cx - max_radius:cx + max_radius + 1]
    yy, xx = np.mgrid[-max_radius:max_radius + 1, -max_radius:max_radius + 1]
    radius = np.sqrt(yy ** 2 + xx ** 2)
    best = 0.0
    for r in range(min_radius, max_radius + 1):
        ring = window[(radius >= r - 0.5) & (radius < r + 0.5)]
        if ring.size:
            best = max(best, float(ring.max() - np.median(ring)))
    return best


def base_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (0, 0), 1.0)
    lap_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_den = cv2.Laplacian(denoised, cv2.CV_64F).var()
    half = cv2.resize(denoised, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lap_half = cv2.Laplacian(half, cv2.CV_64F).var()
    tiles = np.array([
        cv2.Laplacian(tile, cv2.CV_64F).var()
        for row in np.array_split(denoised, 6, 0) for tile in np.array_split(row, 6, 1)
    ])
    cr_v, cr_h = _crete_roffet(gray)
    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    strong = magnitude > np.percentile(magnitude, 90)
    jxx, jyy, jxy = (gx * gx)[strong].sum(), (gy * gy)[strong].sum(), (gx * gy)[strong].sum()
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6))
    p99 = float(np.percentile(magnitude, 99))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5)) + 1
    noise = float(np.median(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3))))
    return np.array([
        np.log1p(lap_raw), np.log1p(lap_den), np.log1p(lap_half),
        np.log1p(lap_den) - np.log1p(lap_half),
        np.log1p(np.percentile(tiles, 50)), np.log1p(np.percentile(tiles, 90)), np.log1p(tiles.max()),
        cr_v, cr_h, max(cr_v, cr_h), abs(cr_v - cr_h),
        coherence, np.log1p(p99 / contrast * 100), noise, gray.mean() / 255.0, _echo(gray),
    ], np.float32)


def edge_features(image: np.ndarray) -> np.ndarray:
    """Edge steepness measured only on edge pixels, so smooth panels do not read as blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    high = max(float(np.percentile(magnitude, 97)), 20.0)
    edges = cv2.Canny(denoised, 0.4 * high, high, L2gradient=True) > 0
    density = float(edges.mean())
    if edges.sum() < MIN_EDGE_PIXELS:
        return np.array([0, 0, 0, 0, 0, 1, 0, density, 0], np.float32)
    kernel = np.ones((7, 7), np.uint8)
    contrast = cv2.dilate(denoised, kernel).astype(np.float32) - cv2.erode(denoised, kernel).astype(np.float32)
    acutance = magnitude[edges] / (contrast[edges] + 8.0)
    angle = (np.degrees(np.arctan2(gy[edges], gx[edges])) + 180.0) % 180.0
    bins = np.minimum((angle / 45.0).astype(int), 3)
    per_direction = [float(np.median(acutance[bins == b])) for b in range(4) if (bins == b).sum() >= 30]
    dir_min, dir_max = (min(per_direction), max(per_direction)) if per_direction else (0.0, 0.0)
    laplacian = cv2.Laplacian(denoised.astype(np.float32), cv2.CV_32F)
    near = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lap_norm = np.abs(laplacian[near]) / (contrast[near] + 8.0)
    top = acutance[acutance >= np.percentile(acutance, 90)]
    return np.array([
        float(np.median(acutance)), float(np.percentile(acutance, 25)), float(np.percentile(acutance, 75)),
        dir_min, dir_max, dir_min / max(dir_max, 1e-6),
        float(np.median(lap_norm)), density, float(top.mean()),
    ], np.float32)


def blur_features(image: np.ndarray, vehicle_box: Sequence[float] | None) -> np.ndarray:
    """Feature vector in FEATURE_NAMES order for a working-size BGR image."""
    vehicle = crop_to_box(image, vehicle_box)
    whole_base = base_features(image)
    whole_edge = edge_features(image)
    return np.concatenate([
        base_features(vehicle), whole_base[list(GLOBAL_BASE_KEEP)],
        edge_features(vehicle), whole_edge[list(GLOBAL_EDGE_KEEP)],
    ]).astype(np.float32)


@dataclass(frozen=True)
class BlurPrediction:
    probability: float
    model_version: str
    latency_ms: float


class BlurClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: dict | None = None
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self) -> dict:
        with self._lock:
            if self._model is None:
                model = json.loads(self.model_path.read_text(encoding="utf-8"))
                if tuple(model["features"]) != FEATURE_NAMES:
                    raise ValueError("模糊模型的特徵順序與程式不一致，請重新訓練")
                self._model = model
            return self._model

    def score_features(self, features: np.ndarray) -> float:
        model = self._load()
        z = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
        logit = float(z @ np.asarray(model["coef"]) + model["intercept"])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict(self, content: bytes, vehicle_box: Sequence[float] | None = None) -> BlurPrediction:
        started = perf_counter()
        features = blur_features(decode_working_image(content), vehicle_box)
        probability = self.score_features(features)
        return BlurPrediction(
            probability=round(probability, 4),
            model_version=str(self._load().get("version", "unknown")),
            latency_ms=round((perf_counter() - started) * 1000, 1),
        )


_classifier: BlurClassifier | None = None


def get_blur_classifier() -> BlurClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BlurClassifier(MODEL_PATH)
    return _classifier
