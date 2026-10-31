"""Train a small-data exterior-view classifier on frozen DINOv2 features.

The trusted local manifest is split by vehicle.  The script never uses the test
split for checkpoint selection or confidence calibration; test is evaluated
once after the best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import truststore
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ["left_front", "right_front", "left_rear", "right_rear"]
HORIZONTAL_LABEL_MAP = {
    "left_front": "right_front",
    "right_front": "left_front",
    "left_rear": "right_rear",
    "right_rear": "left_rear",
}


class RandomLightingDomain:
    """Simulate daylight, low-light, and night exposure without changing geometry."""

    def __init__(self, probability: float = 0.7) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        if random.random() < 0.62:
            brightness = random.uniform(0.32, 0.76)
            contrast = random.uniform(0.78, 1.28)
            saturation = random.uniform(0.62, 1.02)
        else:
            brightness = random.uniform(0.86, 1.28)
            contrast = random.uniform(0.82, 1.22)
            saturation = random.uniform(0.82, 1.15)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return ImageEnhance.Color(image).enhance(saturation)


class ForcedNightDomain:
    """Create a hard day-to-night view for domain-expert training."""

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.18, 0.52))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.35))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.48, 0.92))
        return image


class LetterboxSquare:
    """Preserve the complete vehicle instead of cropping the long image axis."""

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


class ViewDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        transform: transforms.Compose,
        classes: list[str],
        *,
        mirror: bool = False,
        swap_mirrored_label: bool = False,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.classes = classes
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.mirror = mirror
        self.swap_mirrored_label = swap_mirrored_label

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["local_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        label = row["label"]
        if self.mirror:
            image = ImageOps.mirror(image)
            if self.swap_mirrored_label:
                label = HORIZONTAL_LABEL_MAP[label]
        return self.transform(image), self.class_to_index[label]


class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ResidualCosineHead(nn.Module):
    """A small residual feature adapter with a normalized cosine classifier."""

    def __init__(self, feature_dim: int, classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.class_weight = nn.Parameter(torch.empty(classes, feature_dim))
        nn.init.normal_(self.class_weight, std=0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.residual_scale.sigmoid() * self.adapter(features)
        adapted = F.normalize(adapted, dim=1)
        weights = F.normalize(self.class_weight, dim=1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * adapted @ weights.T


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path, classes: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] in classes and Path(row["local_path"]).exists():
                grouped[row["split"]].append(row)
    for split in ("train", "val", "test"):
        if not grouped[split]:
            raise ValueError(f"Manifest has no valid {split} images")
    car_sets = {
        split: {row["car_no"] for row in split_rows}
        for split, split_rows in grouped.items()
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = car_sets[first] & car_sets[second]
        if overlap:
            raise ValueError(f"Vehicle leakage between {first}/{second}: {sorted(overlap)}")
    return grouped


def build_transforms(
    *, aspect_preserving: bool = False
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_geometry: list[object]
    evaluation_geometry: list[object]
    if aspect_preserving:
        train_geometry = [LetterboxSquare(238)]
        evaluation_geometry = [LetterboxSquare(224)]
    else:
        train_geometry = [
            transforms.Resize(252, antialias=True),
            transforms.CenterCrop(238),
        ]
        evaluation_geometry = [
            transforms.Resize(252, antialias=True),
            transforms.CenterCrop(224),
        ]
    train_transform = transforms.Compose(
        [
            *train_geometry,
            transforms.RandomResizedCrop(
                224,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                antialias=True,
            ),
            RandomLightingDomain(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.16,
                        contrast=0.14,
                        saturation=0.10,
                        hue=0.01,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            *evaluation_geometry,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def build_synthetic_night_transform(*, aspect_preserving: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    geometry: list[object]
    if aspect_preserving:
        geometry = [LetterboxSquare(238)]
    else:
        geometry = [
            transforms.Resize(252, antialias=True),
            transforms.CenterCrop(238),
        ]
    return transforms.Compose(
        [
            *geometry,
            transforms.RandomResizedCrop(
                224,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                antialias=True,
            ),
            ForcedNightDomain(),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_lighting(path: str) -> str:
    with Image.open(path) as source:
        grayscale = np.asarray(
            ImageOps.exif_transpose(source).convert("L").resize((256, 256)),
            dtype=np.uint8,
        )
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


def load_backbone(cache_dir: Path, *, pretrained: bool = True) -> nn.Module:
    torch.hub.set_dir(str(cache_dir))
    local_repository = cache_dir / "facebookresearch_dinov2_main"
    if local_repository.exists():
        return torch.hub.load(
            str(local_repository),
            "dinov2_vits14",
            source="local",
            pretrained=pretrained,
        )
    return torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        trust_repo=True,
        pretrained=pretrained,
    )


@torch.inference_mode()
def extract_features(
    backbone: nn.Module,
    rows: list[dict[str, str]],
    transform: transforms.Compose,
    classes: list[str],
    device: torch.device,
    batch_size: int,
    *,
    mirror: bool = False,
    swap_mirrored_label: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        ViewDataset(
            rows,
            transform,
            classes,
            mirror=mirror,
            swap_mirrored_label=swap_mirrored_label,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    backbone.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        embeddings = F.normalize(backbone(images).float(), dim=1)
        features.append(embeddings.cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def build_training_features(
    backbone: nn.Module,
    rows: list[dict[str, str]],
    transform: transforms.Compose,
    classes: list[str],
    device: torch.device,
    batch_size: int,
    augmentations: int,
    mirrored_augmentation: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for repeat in range(augmentations):
        seed_everything(seed + repeat)
        features, labels = extract_features(
            backbone,
            rows,
            transform,
            classes,
            device,
            batch_size,
            mirror=mirrored_augmentation and repeat % 2 == 1,
            swap_mirrored_label=mirrored_augmentation and repeat % 2 == 1,
        )
        all_features.append(features)
        all_labels.append(labels)
        print(
            json.dumps(
                {
                    "feature_pass": repeat + 1,
                    "feature_passes": augmentations,
                    "mirrored": mirrored_augmentation and repeat % 2 == 1,
                }
            )
        )
    return torch.cat(all_features), torch.cat(all_labels)


def horizontal_remap(classes: list[str]) -> list[int]:
    class_to_index = {label: index for index, label in enumerate(classes)}
    return [class_to_index[HORIZONTAL_LABEL_MAP[label]] for label in classes]


@torch.inference_mode()
def prediction_logits(
    backbone: nn.Module,
    probe: LinearProbe,
    rows: list[dict[str, str]],
    transform: transforms.Compose,
    classes: list[str],
    device: torch.device,
    batch_size: int,
    *,
    horizontal_tta: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = extract_features(
        backbone, rows, transform, classes, device, batch_size
    )
    logits = probe(features.to(device)).cpu()
    if not horizontal_tta:
        return logits, labels
    mirrored_features, mirrored_labels = extract_features(
        backbone,
        rows,
        transform,
        classes,
        device,
        batch_size,
        mirror=True,
        swap_mirrored_label=False,
    )
    if not torch.equal(labels, mirrored_labels):
        raise RuntimeError("TTA labels changed unexpectedly")
    mirrored_logits = probe(mirrored_features.to(device)).cpu()
    mirrored_logits = mirrored_logits[:, horizontal_remap(classes)]
    return (logits + mirrored_logits) / 2, labels


def classification_metrics(
    labels: torch.Tensor, predictions: torch.Tensor, classes: list[str]
) -> dict[str, object]:
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for expected, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[expected, predicted] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for index, label in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum()) - true_positive
        false_negative = int(matrix[index, :].sum()) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(matrix[index, :].sum()),
        }
    accuracy = float((predictions == labels).float().mean())
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_order": classes,
    }


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # prediction_logits is intentionally produced under inference mode.  Clone
    # it into ordinary tensors before LBFGS needs to retain values for autograd.
    logits = logits.clone()
    labels = labels.clone()
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return round(float(log_temperature.exp().clamp(0.05, 10.0).detach()), 4)


def knn_probabilities(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    reference_labels: torch.Tensor,
    classes: int,
    *,
    neighbors: int,
    temperature: float,
) -> torch.Tensor:
    neighbor_count = min(neighbors, len(reference_features))
    similarities = query_features @ reference_features.T
    top_similarities, top_indices = similarities.topk(neighbor_count, dim=1)
    neighbor_labels = reference_labels[top_indices]
    # Subtracting the row maximum keeps the exponential stable without
    # changing the normalized class probabilities.
    weights = torch.exp(
        (top_similarities - top_similarities[:, :1]) / temperature
    )
    scores = torch.zeros((len(query_features), classes), dtype=torch.float32)
    scores.scatter_add_(1, neighbor_labels, weights)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


def select_knn_hyperparameters(
    features: torch.Tensor,
    labels: torch.Tensor,
    cars: list[str],
    classes: list[str],
    seed: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    unique_cars = sorted(set(cars))
    random.Random(seed).shuffle(unique_cars)
    fold_by_car = {car: index % 5 for index, car in enumerate(unique_cars)}
    candidates: list[dict[str, float | int]] = []
    for neighbors in (1, 3, 5, 7, 11, 15):
        for temperature in (0.04, 0.07, 0.10, 0.20):
            fold_predictions: list[torch.Tensor] = []
            fold_labels: list[torch.Tensor] = []
            for fold in range(5):
                reference_mask = torch.tensor(
                    [fold_by_car[car] != fold for car in cars], dtype=torch.bool
                )
                query_mask = ~reference_mask
                probabilities = knn_probabilities(
                    features[query_mask],
                    features[reference_mask],
                    labels[reference_mask],
                    len(classes),
                    neighbors=neighbors,
                    temperature=temperature,
                )
                fold_predictions.append(probabilities.argmax(dim=1))
                fold_labels.append(labels[query_mask])
            metrics = classification_metrics(
                torch.cat(fold_labels), torch.cat(fold_predictions), classes
            )
            candidates.append(
                {
                    "neighbors": neighbors,
                    "temperature": temperature,
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            float(item["macro_f1"]),
            float(item["accuracy"]),
            -int(item["neighbors"]),
        ),
    )
    return selected, candidates


def select_ensemble_alpha(
    linear_probabilities: torch.Tensor,
    knn_probabilities_value: torch.Tensor,
    labels: torch.Tensor,
    classes: list[str],
) -> tuple[float, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    for step in range(11):
        alpha = step / 10
        probabilities = (
            alpha * linear_probabilities + (1 - alpha) * knn_probabilities_value
        )
        metrics = classification_metrics(labels, probabilities.argmax(dim=1), classes)
        candidates.append(
            {
                "linear_alpha": alpha,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -abs(item["linear_alpha"] - 0.5),
        ),
    )
    return selected["linear_alpha"], candidates


def confidence_policy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    target_accuracy: float = 0.97,
) -> dict[str, float | int]:
    probabilities = (logits / temperature).softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    candidates: list[dict[str, float | int]] = []
    for value in range(50, 100):
        threshold = value / 100
        accepted = confidences >= threshold
        accepted_count = int(accepted.sum())
        if accepted_count < 8:
            continue
        accuracy = float((predictions[accepted] == labels[accepted]).float().mean())
        coverage = accepted_count / len(labels)
        candidates.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "coverage": round(coverage, 4),
                "accepted_accuracy": round(accuracy, 4),
            }
        )

    if not candidates:
        return {
            "threshold": 0.99,
            "accepted": 0,
            "coverage": 0.0,
            "accepted_accuracy": 0.0,
        }

    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["accepted_accuracy"]) >= target_accuracy
    ]
    if eligible:
        return max(
            eligible,
            key=lambda candidate: (
                float(candidate["coverage"]),
                float(candidate["accepted_accuracy"]),
                float(candidate["threshold"]),
            ),
        )

    # A small calibration set may contain no threshold that reaches the ideal
    # target.  Return the best *measured* policy instead of the old synthetic
    # fallback (0 accepted at 0.95), which made the report contradict the
    # evaluation performed immediately afterwards.
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["accepted_accuracy"]),
            float(candidate["coverage"]),
            float(candidate["threshold"]),
        ),
    )


def evaluate_split(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: list[str],
    temperature: float,
    threshold: float,
) -> dict[str, object]:
    probabilities = (logits / temperature).softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    metrics = classification_metrics(labels, predictions, classes)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum())
    metrics["mean_confidence"] = round(float(confidences.mean()), 4)
    metrics["confidence_policy"] = {
        "threshold": threshold,
        "accepted": accepted_count,
        "rejected": len(labels) - accepted_count,
        "coverage": round(accepted_count / len(labels), 4),
        "accepted_accuracy": round(
            float((predictions[accepted] == labels[accepted]).float().mean())
            if accepted_count
            else 0.0,
            4,
        ),
    }
    return metrics


def train_residual_cosine_head(
    training_features: torch.Tensor,
    training_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    classes: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[ResidualCosineHead, dict[str, object]]:
    head = ResidualCosineHead(
        training_features.shape[1],
        len(classes),
        hidden_dim=args.night_adapter_dim,
    ).to(device)
    counts = torch.bincount(training_labels, minlength=len(classes)).float()
    class_weights = counts.sum() / (len(classes) * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=min(args.label_smoothing, 0.03),
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.night_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.night_epochs,
        eta_min=args.night_learning_rate * 0.02,
    )
    loader = DataLoader(
        TensorDataset(training_features, training_labels),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 9000),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.night_epochs + 1):
        head.train()
        total_loss = 0.0
        seen = 0
        correct = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss) * labels.size(0)
            seen += labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
        scheduler.step()
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_features.to(device)).cpu()
        metrics = classification_metrics(
            validation_labels, validation_logits.argmax(dim=1), classes
        )
        result = {
            "epoch": epoch,
            "train_loss": round(total_loss / seen, 5),
            "train_accuracy": round(correct / seen, 4),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(result)
        score = float(metrics["macro_f1"])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"night_expert": result}))
        if epoch >= args.min_epochs and without_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Night expert training did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.to(device).eval()
    return head, {
        "head_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "class_counts": {
            label: int(counts[index]) for index, label in enumerate(classes)
        },
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required for the DINOv2 training profile")
    classes = [value.strip() for value in args.classes.split(",") if value.strip()]
    if set(classes) != set(DEFAULT_CLASSES) or len(classes) != 4:
        raise ValueError("DINOv2 exterior training requires the four exterior-view labels")
    rows = read_manifest(args.manifest, classes)
    train_transform, evaluation_transform = build_transforms(
        aspect_preserving=args.aspect_preserving
    )
    cache_dir = args.cache_dir.resolve()
    backbone = load_backbone(cache_dir).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    started = time.perf_counter()
    training_features, training_labels = build_training_features(
        backbone,
        rows["train"],
        train_transform,
        classes,
        device,
        args.feature_batch_size,
        args.augmentations,
        args.mirrored_augmentation,
        args.seed,
    )
    validation_features, validation_labels = extract_features(
        backbone,
        rows["val"],
        evaluation_transform,
        classes,
        device,
        args.feature_batch_size,
    )
    feature_seconds = time.perf_counter() - started

    probe = LinearProbe(training_features.shape[1], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.02
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        TensorDataset(training_features, training_labels),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        seen = 0
        correct = 0
        for features, labels in training_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
        scheduler.step()
        probe.eval()
        with torch.inference_mode():
            val_logits = probe(validation_features.to(device)).cpu()
        val_predictions = val_logits.argmax(dim=1)
        val_metrics = classification_metrics(validation_labels, val_predictions, classes)
        epoch_result = {
            "epoch": epoch,
            "train_loss": round(total_loss / seen, 5),
            "train_accuracy": round(correct / seen, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_result)
        val_macro_f1 = float(val_metrics["macro_f1"])
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(epoch_result))
        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    probe.load_state_dict(best_state)
    probe.to(device).eval()

    ensemble_report: dict[str, object] | None = None
    checkpoint_extras: dict[str, object] = {}
    architecture = "dinov2_vits14_linear_probe"
    model_version = args.model_version or "exterior-view-dinov2-v2"
    if args.knn_ensemble:
        if args.horizontal_tta:
            raise ValueError("--knn-ensemble does not combine with --horizontal-tta")
        reference_features, reference_labels = extract_features(
            backbone,
            rows["train"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
        )
        validation_features, validation_labels = extract_features(
            backbone,
            rows["val"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
        )
        with torch.inference_mode():
            validation_linear_logits = probe(validation_features.to(device)).cpu()
        linear_temperature = calibrate_temperature(
            validation_linear_logits, validation_labels
        )
        selected_knn, knn_candidates = select_knn_hyperparameters(
            reference_features,
            reference_labels,
            [row["car_no"] for row in rows["train"]],
            classes,
            args.seed,
        )
        validation_knn_probabilities = knn_probabilities(
            validation_features,
            reference_features,
            reference_labels,
            len(classes),
            neighbors=int(selected_knn["neighbors"]),
            temperature=float(selected_knn["temperature"]),
        )
        validation_linear_probabilities = (
            validation_linear_logits / linear_temperature
        ).softmax(dim=1)
        ensemble_alpha, ensemble_candidates = select_ensemble_alpha(
            validation_linear_probabilities,
            validation_knn_probabilities,
            validation_labels,
            classes,
        )
        validation_probabilities = (
            ensemble_alpha * validation_linear_probabilities
            + (1 - ensemble_alpha) * validation_knn_probabilities
        )
        validation_logits = validation_probabilities.clamp_min(1e-9).log()
        temperature = calibrate_temperature(validation_logits, validation_labels)
        policy = confidence_policy(validation_logits, validation_labels, temperature)
        threshold = float(policy["threshold"])

        test_started = time.perf_counter()
        test_features, test_labels = extract_features(
            backbone,
            rows["test"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
        )
        with torch.inference_mode():
            test_linear_logits = probe(test_features.to(device)).cpu()
        test_knn_probabilities = knn_probabilities(
            test_features,
            reference_features,
            reference_labels,
            len(classes),
            neighbors=int(selected_knn["neighbors"]),
            temperature=float(selected_knn["temperature"]),
        )
        test_probabilities = (
            ensemble_alpha
            * (test_linear_logits / linear_temperature).softmax(dim=1)
            + (1 - ensemble_alpha) * test_knn_probabilities
        )
        test_logits = test_probabilities.clamp_min(1e-9).log()
        architecture = "dinov2_vits14_knn_ensemble"
        model_version = args.model_version or "exterior-view-dinov2-ensemble-v2"
        ensemble_report = {
            "knn_selected_by_vehicle_group_cross_validation": selected_knn,
            "knn_candidates": knn_candidates,
            "linear_temperature": linear_temperature,
            "linear_alpha_selected_on_validation": ensemble_alpha,
            "ensemble_candidates": ensemble_candidates,
        }
        checkpoint_extras = {
            "reference_features": reference_features.to(torch.float16),
            "reference_labels": reference_labels,
            "reference_cars": [row["car_no"] for row in rows["train"]],
            "knn_neighbors": int(selected_knn["neighbors"]),
            "knn_temperature": float(selected_knn["temperature"]),
            "linear_temperature": linear_temperature,
            "ensemble_alpha": ensemble_alpha,
        }
    else:
        validation_logits, validation_labels = prediction_logits(
            backbone,
            probe,
            rows["val"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
            horizontal_tta=args.horizontal_tta,
        )
        temperature = calibrate_temperature(validation_logits, validation_labels)
        policy = confidence_policy(validation_logits, validation_labels, temperature)
        threshold = float(policy["threshold"])
        test_started = time.perf_counter()
        test_logits, test_labels = prediction_logits(
            backbone,
            probe,
            rows["test"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
            horizontal_tta=args.horizontal_tta,
        )
    main_temperature = temperature
    main_policy = policy
    main_validation = evaluate_split(
        validation_logits,
        validation_labels,
        classes,
        main_temperature,
        float(main_policy["threshold"]),
    )
    main_test = evaluate_split(
        test_logits,
        test_labels,
        classes,
        main_temperature,
        float(main_policy["threshold"]),
    )
    night_expert_report: dict[str, object] | None = None
    router_temperature = 1.0
    if args.night_expert:
        routing_domains = {"night", "low_light"}
        train_domains = [image_lighting(row["local_path"]) for row in rows["train"]]
        validation_domains = [
            image_lighting(row["local_path"]) for row in rows["val"]
        ]
        test_domains = [image_lighting(row["local_path"]) for row in rows["test"]]
        real_night_rows = [
            row
            for row, domain in zip(rows["train"], train_domains, strict=True)
            if domain in routing_domains
        ]
        night_validation_mask = torch.tensor(
            [domain in routing_domains for domain in validation_domains],
            dtype=torch.bool,
        )
        night_test_mask = torch.tensor(
            [domain in routing_domains for domain in test_domains], dtype=torch.bool
        )
        if (
            len(real_night_rows) < 16
            or int(night_validation_mask.sum()) < 8
            or int(night_test_mask.sum()) < 8
        ):
            raise RuntimeError("Insufficient independent night/low-light domain data")
        real_features, real_labels = build_training_features(
            backbone,
            real_night_rows,
            train_transform,
            classes,
            device,
            args.feature_batch_size,
            args.night_augmentations,
            False,
            args.seed + 1000,
        )
        synthetic_transform = build_synthetic_night_transform(
            aspect_preserving=args.aspect_preserving
        )
        synthetic_features, synthetic_labels = build_training_features(
            backbone,
            rows["train"],
            synthetic_transform,
            classes,
            device,
            args.feature_batch_size,
            args.night_synthetic_augmentations,
            False,
            args.seed + 3000,
        )
        expert_training_features = torch.cat(
            [real_features, synthetic_features], dim=0
        )
        expert_training_labels = torch.cat([real_labels, synthetic_labels], dim=0)
        expert_validation_features, expert_validation_labels = extract_features(
            backbone,
            rows["val"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
        )
        if not torch.equal(expert_validation_labels, validation_labels):
            raise RuntimeError("Night validation feature order changed")
        night_head, night_training = train_residual_cosine_head(
            expert_training_features,
            expert_training_labels,
            expert_validation_features[night_validation_mask],
            validation_labels[night_validation_mask],
            classes,
            device,
            args,
        )
        with torch.inference_mode():
            night_validation_logits = night_head(
                expert_validation_features.to(device)
            ).cpu()
        expert_test_features, expert_test_labels = extract_features(
            backbone,
            rows["test"],
            evaluation_transform,
            classes,
            device,
            args.feature_batch_size,
        )
        if not torch.equal(expert_test_labels, test_labels):
            raise RuntimeError("Night test feature order changed")
        with torch.inference_mode():
            night_test_logits = night_head(expert_test_features.to(device)).cpu()
        night_temperature = calibrate_temperature(
            night_validation_logits[night_validation_mask],
            validation_labels[night_validation_mask],
        )
        night_policy = confidence_policy(
            night_validation_logits[night_validation_mask],
            validation_labels[night_validation_mask],
            night_temperature,
        )
        routed_validation_probabilities = (
            validation_logits / main_temperature
        ).softmax(dim=1)
        routed_test_probabilities = (test_logits / main_temperature).softmax(dim=1)
        routed_validation_probabilities[night_validation_mask] = (
            night_validation_logits[night_validation_mask] / night_temperature
        ).softmax(dim=1)
        routed_test_probabilities[night_test_mask] = (
            night_test_logits[night_test_mask] / night_temperature
        ).softmax(dim=1)
        validation_logits = routed_validation_probabilities.clamp_min(1e-9).log()
        test_logits = routed_test_probabilities.clamp_min(1e-9).log()
        router_temperature = calibrate_temperature(
            validation_logits, validation_labels
        )
        temperature = router_temperature
        policy = confidence_policy(validation_logits, validation_labels, temperature)
        threshold = float(policy["threshold"])
        architecture = f"{architecture}_night_adapter"
        checkpoint_extras.update(
            {
                "night_head_state_dict": night_training["head_state_dict"],
                "night_adapter_dim": args.night_adapter_dim,
                "night_temperature": night_temperature,
                "night_routing_domains": sorted(routing_domains),
                "router_temperature": router_temperature,
            }
        )
        night_training_report = {
            key: value
            for key, value in night_training.items()
            if key != "head_state_dict"
        }
        night_only_mask = torch.tensor(
            [domain == "night" for domain in test_domains], dtype=torch.bool
        )
        night_expert_report = {
            "architecture": "residual_cosine_feature_adapter",
            "routing_domains": sorted(routing_domains),
            "real_training_images": len(real_night_rows),
            "synthetic_day_to_night_images": len(rows["train"]),
            "training_feature_rows": len(expert_training_labels),
            "validation_images": int(night_validation_mask.sum()),
            "test_images": int(night_test_mask.sum()),
            "night_only_test_images": int(night_only_mask.sum()),
            "temperature": night_temperature,
            "confidence_policy_selected_on_domain_validation": night_policy,
            "training": night_training_report,
            "domain_validation": evaluate_split(
                night_validation_logits[night_validation_mask],
                validation_labels[night_validation_mask],
                classes,
                night_temperature,
                float(night_policy["threshold"]),
            ),
            "domain_test": evaluate_split(
                night_test_logits[night_test_mask],
                test_labels[night_test_mask],
                classes,
                night_temperature,
                float(night_policy["threshold"]),
            ),
            "night_only_test": evaluate_split(
                night_test_logits[night_only_mask],
                test_labels[night_only_mask],
                classes,
                night_temperature,
                float(night_policy["threshold"]),
            ),
        }
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_seconds = time.perf_counter() - test_started
    validation = evaluate_split(
        validation_logits, validation_labels, classes, temperature, threshold
    )
    test = evaluate_split(test_logits, test_labels, classes, temperature, threshold)
    test["inference_ms_per_image"] = round(test_seconds * 1000 / len(test_labels), 3)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "backbone_source": "facebookresearch/dinov2 (LVD-142M public weights)",
        "classes": classes,
        "device": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU experiment"
        ),
        "dataset": {
            split: {
                "images": len(split_rows),
                "vehicles": len({row["car_no"] for row in split_rows}),
            }
            for split, split_rows in rows.items()
        },
        "split_policy": "Vehicle-disjoint train/val/test; test excluded from selection and calibration",
        "seed": args.seed,
        "augmentations": args.augmentations,
        "lighting_domain_augmentation": True,
        "aspect_preserving_letterbox": args.aspect_preserving,
        "mirrored_augmentation": args.mirrored_augmentation,
        "horizontal_tta": args.horizontal_tta,
        "feature_extraction_seconds": round(feature_seconds, 2),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "temperature": temperature,
        "main_temperature": main_temperature,
        "router_temperature": router_temperature,
        "ensemble": ensemble_report,
        "main_general_without_domain_expert": {
            "confidence_policy_selected_on_validation": main_policy,
            "validation": main_validation,
            "test": main_test,
        },
        "night_expert": night_expert_report,
        "confidence_policy_selected_on_validation": policy,
        "validation": validation,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "classes": classes,
            "input_size": 224,
            "preprocess": (
                "aspect_preserving_letterbox"
                if args.aspect_preserving
                else "center_crop"
            ),
            "feature_dim": training_features.shape[1],
            "head_state_dict": best_state,
            "confidence_threshold": threshold,
            "temperature": main_temperature,
            "horizontal_tta": args.horizontal_tta,
            "model_version": model_version,
            "report": report,
            **checkpoint_extras,
        },
        args.output,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/view_dataset/local_damage_manifest.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/exterior_view_dinov2_candidate.pt")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/view_dataset/exterior_dinov2_training_report.json"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "torch"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow slow experiment-only training when CUDA PyTorch is unavailable",
    )
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--mirrored-augmentation", action="store_true")
    parser.add_argument("--horizontal-tta", action="store_true")
    parser.add_argument("--knn-ensemble", action="store_true")
    parser.add_argument("--aspect-preserving", action="store_true")
    parser.add_argument("--night-expert", action="store_true")
    parser.add_argument("--night-augmentations", type=int, default=8)
    parser.add_argument("--night-synthetic-augmentations", type=int, default=2)
    parser.add_argument("--night-adapter-dim", type=int, default=64)
    parser.add_argument("--night-learning-rate", type=float, default=2e-3)
    parser.add_argument("--night-epochs", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
