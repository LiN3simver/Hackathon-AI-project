"""Production CPU front-seat dirt screening; final dirt verdict stays human."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.nn import functional as F

from app.schemas.cleanliness import CleanlinessAssessment, CleanlinessLevel, FrontCleanlinessScreening, InteriorView
from app.schemas.quality import ExpectedView
from app.services.cleanliness import InvalidModelResponseError, ModelUnavailableError
from app.services.cleanliness_features import clip_embedding, clip_pixels, color_features, fused_features
from app.services.interior_capture import get_interior_capture_classifier
from app.services.runtime_settings import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models/interior_front_cleanliness_classifier.pt"
ARCHITECTURE = "clip_color_ridge_review"


class FrontCleanlinessAnalyzer:
    def __init__(self, model_path: Path = MODEL_PATH):
        torch.set_num_threads(4)
        self.model_path = model_path
        self._lock = RLock()
        self._signature = None
        self._checkpoint = None
        self._model = None
        self._processor = None

    def _load(self):
        stat = self.model_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != ARCHITECTURE or not checkpoint.get("production_approved"):
            raise ValueError("前座髒汙模型尚未正式啟用或架構不相容。")
        backbone = (PROJECT_ROOT / checkpoint["backbone_path"]).resolve()
        if PROJECT_ROOT not in backbone.parents:
            raise ValueError("前座髒汙骨幹路徑不合法。")
        if self._model is None or self._checkpoint.get("backbone_path") != checkpoint["backbone_path"]:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(backbone, local_files_only=True).eval().to("cpu")
            processor = CLIPProcessor.from_pretrained(backbone, local_files_only=True)
            self._model, self._processor = model, processor
        self._checkpoint = checkpoint
        self._signature = signature

    def analyze(self, content: bytes, expected_view: str | None = "front") -> tuple[CleanlinessAssessment, float]:
        if expected_view != "front":
            raise ValueError("前座髒汙模型僅供前座照片使用。")
        started = perf_counter()
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidModelResponseError("前座照片無法解析。") from exc
        try:
            scene = get_interior_capture_classifier().predict(content, ExpectedView.INTERIOR_FRONT)
            usable = scene.matches_expected and scene.scene_confident and not scene.rotation_vetoed
            with self._lock:
                self._load()
                version = str(self._checkpoint["model_version"])
                threshold = float(self._checkpoint["threshold"])
                score = None
                if usable:
                    with torch.inference_mode():
                        embedding = F.normalize(
                            clip_embedding(
                                self._model.get_image_features(pixel_values=clip_pixels(self._processor, image))
                            ),
                            dim=1,
                        )[0].numpy()
                    color, _ = color_features(image)
                    values = fused_features(embedding, color)
                    ridge = self._checkpoint["ridge_state"]
                    score = float(values @ ridge["coef"].numpy() + float(ridge["intercept"]))
        except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
            raise ModelUnavailableError(f"前座髒汙模型目前無法執行，請人工覆核：{exc}") from exc
        detected = score is not None and score >= threshold
        screening = FrontCleanlinessScreening(
            model_version=version,
            score=score,
            threshold=threshold,
            suspected_dirty=detected,
            human_review_required=True,
        )
        summary = (
            "前座照片範圍、方向或清晰度不足，請人工確認。"
            if not usable
            else "前座髒汙模型提示疑似髒汙，需要人工覆核。"
            if detected
            else "前座髒汙檢查已完成，目前未觸發髒汙覆核提示。"
        )
        return CleanlinessAssessment(
            is_vehicle_interior=bool(scene.matches_expected),
            image_usable=bool(usable),
            view=InteriorView.FRONT,
            level=CleanlinessLevel.CANNOT_ASSESS
            if not usable
            else CleanlinessLevel.DIRTY
            if detected
            else CleanlinessLevel.ORDINARY,
            confidence=0.0,
            immediate_cleaning_required=False,
            summary=summary,
            evidence=[],
            specialist=screening,
        ), (perf_counter() - started) * 1000


@lru_cache
def get_front_cleanliness_analyzer() -> FrontCleanlinessAnalyzer:
    return FrontCleanlinessAnalyzer()
