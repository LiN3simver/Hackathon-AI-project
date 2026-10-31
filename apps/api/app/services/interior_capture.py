from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F

from app.schemas.quality import (
    ExpectedView,
    ExteriorScenePrediction,
    InteriorCapturePrediction,
)
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.view_classifier import (
    PROJECT_ROOT,
    _letterbox_transform,
    _load_dinov2_backbone,
)

FRONT_SCENE_MODEL_PATH = PROJECT_ROOT / "models" / "interior_front_classifier.pt"
REAR_SCENE_MODEL_PATH = PROJECT_ROOT / "models" / "interior_rear_classifier.pt"
SCENE_MODEL_PATH = PROJECT_ROOT / "models" / "interior_view_classifier.pt"
CURATED_SCENE_MODEL_PATH = PROJECT_ROOT / "models" / "curated_interior_view_classifier.pt"
ROTATION_MODEL_PATH = PROJECT_ROOT / "models" / "interior_rotation_veto.pt"
INTERIOR_LABELS = {
    ExpectedView.INTERIOR_FRONT: "interior_front",
    ExpectedView.INTERIOR_REAR: "interior_rear",
}


class SpecialistProbe(nn.Module):
    def __init__(self, feature_dim: int, classes: int, architecture: str) -> None:
        super().__init__()
        if architecture.endswith("_linear_probe"):
            self.network = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, classes),
            )
        elif architecture.endswith("_mlp_probe"):
            self.network = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 128),
                nn.GELU(),
                nn.Dropout(0.12),
                nn.Linear(128, classes),
            )
        else:
            raise ValueError(f"Unsupported interior specialist architecture: {architecture}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class LoadedProbe:
    checkpoint: dict[str, Any]
    head: SpecialistProbe
    mtime_ns: int


class InteriorCaptureClassifier:
    """Conservative scene and rotation veto for front/rear interior uploads."""

    def __init__(
        self,
        scene_path: Path = SCENE_MODEL_PATH,
        front_scene_path: Path = FRONT_SCENE_MODEL_PATH,
        rear_scene_path: Path = REAR_SCENE_MODEL_PATH,
        rotation_path: Path = ROTATION_MODEL_PATH,
        curated_scene_path: Path = CURATED_SCENE_MODEL_PATH,
    ) -> None:
        self.scene_path = scene_path
        self.front_scene_path = front_scene_path
        self.rear_scene_path = rear_scene_path
        self.rotation_path = rotation_path
        self.curated_scene_path = curated_scene_path
        self._backbone: nn.Module | None = None
        self._scene_cache: dict[Path, LoadedProbe] = {}
        self._rotation: LoadedProbe | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def available(self) -> bool:
        return (
            self.front_scene_path.exists() and self.rear_scene_path.exists()
        ) or self.scene_path.exists()

    @property
    def exterior_veto_available(self) -> bool:
        return self.front_scene_path.exists() or self.rear_scene_path.exists()

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_probe(
        self,
        path: Path,
        cached: LoadedProbe | None,
        device: torch.device,
    ) -> LoadedProbe | None:
        if not path.exists():
            return None
        mtime_ns = path.stat().st_mtime_ns
        if cached is not None and cached.mtime_ns == mtime_ns:
            return cached
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        head = SpecialistProbe(
            int(checkpoint["feature_dim"]),
            len(checkpoint["classes"]),
            str(checkpoint["architecture"]),
        )
        head.load_state_dict(checkpoint["head_state_dict"])
        head.eval().to(device)
        return LoadedProbe(checkpoint=checkpoint, head=head, mtime_ns=mtime_ns)

    def _load(
        self, expected_view: ExpectedView
    ) -> tuple[nn.Module, LoadedProbe, LoadedProbe | None, LoadedProbe | None, torch.device]:
        device = self._device()
        with self._load_lock:
            if self._backbone is None:
                self._backbone = _load_dinov2_backbone().eval().to(device)
            preferred_path = (
                self.front_scene_path
                if expected_view == ExpectedView.INTERIOR_FRONT
                else self.rear_scene_path
            )
            scene_path = preferred_path if preferred_path.exists() else self.scene_path
            scene = self._load_probe(
                scene_path,
                self._scene_cache.get(scene_path),
                device,
            )
            if scene is not None:
                self._scene_cache[scene_path] = scene
            curated = self._load_probe(
                self.curated_scene_path,
                self._scene_cache.get(self.curated_scene_path),
                device,
            )
            if curated is not None:
                self._scene_cache[self.curated_scene_path] = curated
                if not bool(curated.checkpoint.get("production_approved", False)):
                    curated = None
            self._rotation = self._load_probe(
                self.rotation_path,
                self._rotation,
                device,
            )
            if self._rotation is not None and not bool(
                self._rotation.checkpoint.get("production_approved", False)
            ):
                self._rotation = None
            if scene is None:
                raise FileNotFoundError(scene_path)
            return self._backbone, scene, curated, self._rotation, device

    def _load_exterior_veto(
        self,
    ) -> tuple[nn.Module, list[tuple[str, LoadedProbe]], torch.device]:
        device = self._device()
        with self._load_lock:
            if self._backbone is None:
                self._backbone = _load_dinov2_backbone().eval().to(device)
            probes: list[tuple[str, LoadedProbe]] = []
            for positive_label, path in (
                ("interior_front", self.front_scene_path),
                ("interior_rear", self.rear_scene_path),
            ):
                probe = self._load_probe(path, self._scene_cache.get(path), device)
                if probe is None:
                    continue
                self._scene_cache[path] = probe
                probes.append((positive_label, probe))
            if not probes:
                raise FileNotFoundError("No interior specialist is available for exterior veto")
            return self._backbone, probes, device

    @staticmethod
    def _probabilities(features: torch.Tensor, probe: LoadedProbe) -> torch.Tensor:
        temperature = float(probe.checkpoint.get("temperature", 1.0))
        return (probe.head(features) / temperature).softmax(dim=1)[0]

    def predict(
        self,
        content: bytes,
        expected_view: ExpectedView,
    ) -> InteriorCapturePrediction:
        if expected_view not in INTERIOR_LABELS:
            raise ValueError("Interior capture classifier only supports interior views")
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image for interior capture classification") from exc

        backbone, scene, curated, rotation, device = self._load(expected_view)
        tensor = _letterbox_transform()(image).unsqueeze(0).to(device)
        started = perf_counter()
        with self._inference_lock, torch.inference_mode():
            features = F.normalize(backbone(tensor).float(), dim=1)
            scene_probabilities = self._probabilities(features, scene)
            scene_confidence, scene_index = scene_probabilities.max(dim=0)
            scene_label = str(scene.checkpoint["classes"][scene_index.item()])
            expected_label = INTERIOR_LABELS[expected_view]
            expert_classes = list(scene.checkpoint["classes"])
            specialist_probability = (
                float(scene_probabilities[expert_classes.index(expected_label)])
                if expected_label in expert_classes else 0.0
            )
            specialist_threshold = float(
                scene.checkpoint.get("predicted_class_confidence_policies", {})
                .get(expected_label, {}).get("threshold", 0.95)
            )
            if curated is not None:
                curated_probabilities = self._probabilities(features, curated)
                scene_confidence, scene_index = curated_probabilities.max(dim=0)
                scene_label = str(curated.checkpoint["classes"][scene_index.item()])
            rotation_probability: float | None = None
            rotation_threshold: float | None = None
            rotation_vetoed = False
            rotation_version = ""
            if rotation is not None:
                rotation_probabilities = self._probabilities(features, rotation)
                rotation_classes = list(rotation.checkpoint["classes"])
                rotated_index = rotation_classes.index("rotated")
                rotation_probability = float(rotation_probabilities[rotated_index])
                rotation_threshold = float(rotation.checkpoint["rotation_veto_threshold"])
                rotation_vetoed = rotation_probability >= rotation_threshold
                rotation_version = str(
                    rotation.checkpoint.get("model_version", "rotation-veto")
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        scene_confidence_value = float(scene_confidence)
        if curated is not None:
            scene_threshold = float(
                curated.checkpoint["acceptance_thresholds"][expected_label]
            )
            scene_confident = (
                scene_label == expected_label
                and scene_confidence_value >= scene_threshold
                and specialist_probability >= specialist_threshold
            )
            scene_version = (
                f"{curated.checkpoint.get('model_version', 'curated-interior')}+"
                f"{scene.checkpoint.get('model_version', 'interior-specialist')}"
            )
        else:
            scene_threshold = get_runtime_recognition_settings().interior_scene_veto_confidence
            scene_confident = scene_confidence_value >= scene_threshold
            scene_version = str(scene.checkpoint.get("model_version", "interior-scene"))
        model_version = scene_version
        if rotation_version:
            model_version = f"{scene_version}+{rotation_version}"
        return InteriorCapturePrediction(
            scene_label=scene_label,
            scene_confidence=round(scene_confidence_value, 4),
            scene_confidence_threshold=scene_threshold,
            scene_confident=scene_confident,
            matches_expected=scene_label == expected_label,
            specialist_probability=round(specialist_probability, 4),
            specialist_threshold=specialist_threshold,
            rotation_probability=(
                round(rotation_probability, 4)
                if rotation_probability is not None
                else None
            ),
            rotation_threshold=rotation_threshold,
            rotation_vetoed=rotation_vetoed,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            model_version=model_version,
        )

    def predict_exterior_scene(self, content: bytes) -> ExteriorScenePrediction:
        """Reject a confidently recognized cabin photo from an exterior slot."""
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image for exterior scene veto") from exc

        backbone, probes, device = self._load_exterior_veto()
        tensor = _letterbox_transform()(image).unsqueeze(0).to(device)
        started = perf_counter()
        probabilities = {"interior_front": 0.0, "interior_rear": 0.0}
        versions: list[str] = []
        with self._inference_lock, torch.inference_mode():
            features = F.normalize(backbone(tensor).float(), dim=1)
            for positive_label, probe in probes:
                classes = list(probe.checkpoint["classes"])
                if positive_label not in classes:
                    continue
                result = self._probabilities(features, probe)
                probabilities[positive_label] = float(result[classes.index(positive_label)])
                versions.append(str(probe.checkpoint.get("model_version", positive_label)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        interior_probability = max(probabilities.values())
        threshold = get_runtime_recognition_settings().interior_scene_veto_confidence
        return ExteriorScenePrediction(
            front_interior_probability=round(probabilities["interior_front"], 4),
            rear_interior_probability=round(probabilities["interior_rear"], 4),
            interior_probability=round(interior_probability, 4),
            veto_threshold=threshold,
            is_interior=interior_probability >= threshold,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            model_version="+".join(versions),
        )


_classifier: InteriorCaptureClassifier | None = None


def get_interior_capture_classifier() -> InteriorCaptureClassifier:
    global _classifier
    if _classifier is None:
        _classifier = InteriorCaptureClassifier()
    return _classifier
