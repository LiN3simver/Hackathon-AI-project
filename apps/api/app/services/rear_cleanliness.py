"""Deployed CPU rear-seat screening, with every verdict retained for human review."""

import hashlib
import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.nn import functional as F

from app.schemas.cleanliness import CleanlinessAssessment, CleanlinessLevel, CleanlinessScreening, InteriorView
from app.schemas.quality import ExpectedView
from app.services.cleanliness import InvalidModelResponseError, ModelUnavailableError
from app.services.interior_capture import get_interior_capture_classifier
from app.services.rear_cleanliness_features import rear_patch_input, rear_patch_score
from app.services.runtime_settings import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models/interior_rear_cleanliness_classifier.pt"
ARCHITECTURE = "dino_box_ridge_0.001_top5"
BACKBONE_REPOSITORY = PROJECT_ROOT / ".cache/torch/facebookresearch_dinov2_main"


def load_rear_checkpoint(path: Path) -> tuple[dict, Path]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("architecture") != ARCHITECTURE
        or checkpoint.get("task") != "rear_ordinary_vs_dirty"
        or checkpoint.get("classes") != ["not_dirty", "dirty"]
        or not checkpoint.get("production_approved")
    ):
        raise ValueError("後座髒污模型尚未啟用或架構不相容。")
    spec = checkpoint["method_spec"]
    ridge = checkpoint["ridge_state"]
    if (
        spec.get("kind") != "patch_ridge"
        or spec.get("top_k") != 5
        or ridge["coef"].shape != (384,)
        or ridge["intercept"].numel() != 1
        or not torch.isfinite(ridge["coef"]).all()
        or not torch.isfinite(ridge["intercept"]).all()
        or not math.isfinite(float(checkpoint["threshold"]))
    ):
        raise ValueError("後座髒污模型參數不相容。")
    backbone = (PROJECT_ROOT / checkpoint["backbone"]["weights_path"]).resolve()
    if PROJECT_ROOT not in backbone.parents or not backbone.is_file() or not BACKBONE_REPOSITORY.is_dir():
        raise ValueError("後座髒污模型缺少本機骨幹。")
    if hashlib.sha256(backbone.read_bytes()).hexdigest() != checkpoint["backbone"]["sha256"]:
        raise ValueError("後座骨幹與訓練權重不一致。")
    return checkpoint, backbone


class RearCleanlinessAnalyzer:
    def __init__(self, model_path: Path = MODEL_PATH):
        torch.set_num_threads(4)
        self.model_path = model_path
        self._lock = RLock()
        self._signature = None
        self._checkpoint = None
        self._model = None

    def _load(self):
        stat = self.model_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return
        checkpoint, backbone = load_rear_checkpoint(self.model_path)
        if self._model is None or self._checkpoint["backbone"] != checkpoint["backbone"]:
            model = torch.hub.load(str(BACKBONE_REPOSITORY), "dinov2_vits14", source="local", pretrained=False)
            model.load_state_dict(torch.load(backbone, map_location="cpu", weights_only=True))
            self._model = model.eval().to("cpu")
        self._checkpoint, self._signature = checkpoint, signature

    def analyze(self, content: bytes, expected_view: str | None = "rear") -> tuple[CleanlinessAssessment, float]:
        if expected_view != "rear":
            raise InvalidModelResponseError("後座髒污模型僅供後座照片使用。")
        started = perf_counter()
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidModelResponseError("後座照片無法解析。") from exc
        try:
            scene = get_interior_capture_classifier().predict(content, ExpectedView.INTERIOR_REAR)
            usable = scene.matches_expected and scene.scene_confident and not scene.rotation_vetoed
            with self._lock:
                self._load()
                version, threshold = str(self._checkpoint["model_version"]), float(self._checkpoint["threshold"])
                score = None
                if usable:
                    values, valid = rear_patch_input(image)
                    with torch.inference_mode():
                        result = self._model.forward_features(values[None])
                        tokens = F.normalize(result["x_norm_patchtokens"], dim=2)[0, valid]
                        score = rear_patch_score(tokens, self._checkpoint)
                    if not math.isfinite(score):
                        raise ValueError("後座模型回傳無效分數。")
        except (OSError, RuntimeError, ValueError, ImportError, KeyError, TypeError) as exc:
            raise ModelUnavailableError(f"後座髒污模型目前無法執行，請人工覆核：{exc}") from exc
        detected = score is not None and score >= threshold
        return CleanlinessAssessment(
            is_vehicle_interior=bool(scene.matches_expected),
            image_usable=bool(usable),
            view=InteriorView.REAR,
            level=CleanlinessLevel.CANNOT_ASSESS
            if not usable
            else CleanlinessLevel.DIRTY
            if detected
            else CleanlinessLevel.ORDINARY,
            confidence=0.0,
            immediate_cleaning_required=False,
            summary="後座照片範圍、方向或清晰度不足，請人工確認。"
            if not usable
            else "後座模型提示疑似髒污或垃圾，需要人工覆核。"
            if detected
            else "後座檢查已完成，目前未觸發髒污覆核提示。",
            evidence=[],
            specialist=CleanlinessScreening(
                model_version=version,
                score=score,
                threshold=threshold,
                suspected_dirty=detected,
                human_review_required=True,
            ),
        ), (perf_counter() - started) * 1000


@lru_cache
def get_rear_cleanliness_analyzer() -> RearCleanlinessAnalyzer:
    return RearCleanlinessAnalyzer()
