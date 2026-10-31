from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from app.schemas.quality import ExpectedView, ViewPrediction
from app.services.model_management import get_model_management_service
from app.services.runtime_settings import get_runtime_recognition_settings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelState:
    backbone: nn.Module
    head: nn.Module
    transform: transforms.Compose
    classes: list[str]
    threshold: float
    temperature: float
    architecture: str
    model_version: str
    device: torch.device
    reference_features: torch.Tensor
    reference_labels: torch.Tensor
    knn_neighbors: int
    knn_temperature: float = 1.0
    linear_temperature: float = 1.0
    ensemble_alpha: float = 1.0
    night_head: nn.Module | None = None
    night_temperature: float = 1.0
    night_routing_domains: tuple[str, ...] = ()
    router_temperature: float = 1.0


class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ResidualCosineHead(nn.Module):
    def __init__(self, feature_dim: int, classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.class_weight = nn.Parameter(torch.empty(classes, feature_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.residual_scale.sigmoid() * self.adapter(features)
        adapted = F.normalize(adapted, dim=1)
        weights = F.normalize(self.class_weight, dim=1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * adapted @ weights.T


class LetterboxSquare:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        contained = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(
            contained,
            ((self.size - contained.width) // 2, (self.size - contained.height) // 2),
        )
        return canvas


def _imagenet_transform(resize: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _letterbox_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            LetterboxSquare(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _lighting_domain(image: Image.Image) -> str:
    grayscale = np.asarray(image.convert("L").resize((256, 256)), dtype=np.uint8)
    mean = float(grayscale.mean())
    dark_ratio = float((grayscale < 48).mean())
    bright_ratio = float((grayscale > 235).mean())
    if mean < 68 or dark_ratio > 0.52:
        return "night"
    if mean < 105 or dark_ratio > 0.32:
        return "low_light"
    if mean > 215 or bright_ratio > 0.45:
        return "overexposed"
    return "daytime"


def _load_dinov2_backbone() -> nn.Module:
    cache_dir = PROJECT_ROOT / ".cache" / "torch"
    torch.hub.set_dir(str(cache_dir))
    local_repository = cache_dir / "facebookresearch_dinov2_main"
    if local_repository.exists():
        return torch.hub.load(
            str(local_repository),
            "dinov2_vits14",
            source="local",
            pretrained=True,
        )
    return torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        trust_repo=True,
        pretrained=True,
    )


def _knn_probabilities(features: torch.Tensor, state: ModelState) -> torch.Tensor:
    neighbor_count = min(state.knn_neighbors, len(state.reference_features))
    similarities = features @ state.reference_features.T
    top_similarities, top_indices = similarities.topk(neighbor_count, dim=1)
    neighbor_labels = state.reference_labels[top_indices]
    weights = torch.exp(
        (top_similarities - top_similarities[:, :1]) / state.knn_temperature
    )
    scores = torch.zeros(
        (len(features), len(state.classes)), device=features.device, dtype=torch.float32
    )
    scores.scatter_add_(1, neighbor_labels, weights)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


class ViewClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._state: ModelState | None = None
        self._loaded_mtime_ns: int | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def _load(self) -> ModelState:
        current_mtime_ns = self.checkpoint_path.stat().st_mtime_ns
        if self._state is not None and self._loaded_mtime_ns == current_mtime_ns:
            return self._state
        with self._load_lock:
            current_mtime_ns = self.checkpoint_path.stat().st_mtime_ns
            if self._state is not None and self._loaded_mtime_ns == current_mtime_ns:
                return self._state
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(self.checkpoint_path)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            classes = list(checkpoint["classes"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            architecture = str(checkpoint.get("architecture", ""))
            if architecture not in {
                "dinov2_vits14_knn_ensemble",
                "dinov2_vits14_knn_ensemble_night_adapter",
            }:
                raise ValueError(f"Unsupported view-classifier architecture: {architecture}")
            backbone = _load_dinov2_backbone().to(device).eval()
            head = LinearProbe(int(checkpoint["feature_dim"]), len(classes))
            head.load_state_dict(checkpoint["head_state_dict"])
            head.to(device).eval()
            night_head: nn.Module | None = None
            if "night_head_state_dict" in checkpoint:
                night_head = ResidualCosineHead(
                    int(checkpoint["feature_dim"]),
                    len(classes),
                    hidden_dim=int(checkpoint.get("night_adapter_dim", 64)),
                )
                night_head.load_state_dict(checkpoint["night_head_state_dict"])
                night_head.to(device).eval()
            transform = (
                _letterbox_transform()
                if checkpoint.get("preprocess") == "aspect_preserving_letterbox"
                else _imagenet_transform(252)
            )
            self._state = ModelState(
                backbone=backbone,
                head=head,
                transform=transform,
                classes=classes,
                threshold=float(checkpoint["confidence_threshold"]),
                temperature=float(checkpoint["temperature"]),
                architecture=architecture,
                model_version=str(checkpoint["model_version"]),
                device=device,
                reference_features=checkpoint["reference_features"].to(
                    device=device, dtype=torch.float32
                ),
                reference_labels=checkpoint["reference_labels"].to(device),
                knn_neighbors=int(checkpoint["knn_neighbors"]),
                knn_temperature=float(checkpoint["knn_temperature"]),
                linear_temperature=float(checkpoint["linear_temperature"]),
                ensemble_alpha=float(checkpoint["ensemble_alpha"]),
                night_head=night_head,
                night_temperature=float(checkpoint.get("night_temperature", 1.0)),
                night_routing_domains=tuple(
                    str(value)
                    for value in checkpoint.get("night_routing_domains", [])
                ),
                router_temperature=float(checkpoint.get("router_temperature", 1.0)),
            )
            self._loaded_mtime_ns = current_mtime_ns
            return self._state

    def predict(self, content: bytes, expected_view: ExpectedView | None) -> ViewPrediction:
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image for view classification") from exc

        state = self._load()
        lighting_domain = _lighting_domain(image)
        tensor = state.transform(image).unsqueeze(0).to(state.device)
        started = perf_counter()
        with self._inference_lock, torch.inference_mode():
            features = F.normalize(state.backbone(tensor).float(), dim=1)
            linear_probabilities = (
                state.head(features) / state.linear_temperature
            ).softmax(dim=1)
            knn_probabilities = _knn_probabilities(features, state)
            ensemble_probabilities = (
                state.ensemble_alpha * linear_probabilities
                + (1 - state.ensemble_alpha) * knn_probabilities
            )
            probabilities = (
                ensemble_probabilities.clamp_min(1e-9).log() / state.temperature
            ).softmax(dim=1)
            if (
                state.night_head is not None
                and lighting_domain in state.night_routing_domains
            ):
                probabilities = (
                    state.night_head(features) / state.night_temperature
                ).softmax(dim=1)
            probabilities = (
                probabilities.clamp_min(1e-9).log() / state.router_temperature
            ).softmax(dim=1)[0]
            confidence, index = probabilities.max(dim=0)
            if state.device.type == "cuda":
                torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000
        predicted = ExpectedView(state.classes[index.item()])
        confidence_value = confidence.item()
        expected_supported = expected_view is not None and expected_view.value in state.classes
        return ViewPrediction(
            predicted_view=predicted,
            confidence=round(confidence_value, 4),
            confidence_threshold=state.threshold,
            is_confident=confidence_value >= state.threshold,
            expected_view=expected_view,
            matches_expected=predicted == expected_view if expected_supported else None,
            latency_ms=round(elapsed_ms, 2),
            model_version=state.model_version,
        )


class CascadingViewClassifier:
    def __init__(self) -> None:
        self._classifiers: dict[Path, ViewClassifier] = {}
        self._lock = Lock()

    def _selected_classifiers(self) -> list[ViewClassifier]:
        classifiers: list[ViewClassifier] = []
        for path in get_model_management_service().selected_paths():
            with self._lock:
                classifier = self._classifiers.get(path)
                if classifier is None:
                    classifier = ViewClassifier(path)
                    self._classifiers[path] = classifier
            classifiers.append(classifier)
        return classifiers

    @property
    def available(self) -> bool:
        return any(classifier.available for classifier in self._selected_classifiers())

    def predict(self, content: bytes, expected_view: ExpectedView | None) -> ViewPrediction:
        threshold = get_runtime_recognition_settings().view_confidence_threshold
        best_prediction: ViewPrediction | None = None
        failures: list[str] = []
        for classifier in self._selected_classifiers()[:3]:
            try:
                prediction = classifier.predict(content, expected_view)
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f"{classifier.checkpoint_path.name}: {exc}")
                logger.warning(
                    "view_model_fallback model=%s reason=%s",
                    classifier.checkpoint_path,
                    exc,
                )
                continue
            if best_prediction is None or prediction.confidence > best_prediction.confidence:
                best_prediction = prediction
            if prediction.confidence >= threshold:
                return prediction
        if best_prediction is not None:
            return best_prediction
        detail = "; ".join(failures) or "no configured model is available"
        raise RuntimeError(f"All configured view models failed: {detail}")


_classifier: CascadingViewClassifier | None = None


def get_view_classifier() -> CascadingViewClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CascadingViewClassifier()
    return _classifier
