"""Learned before/after damage specialist trained on operator human truths.

Both photos are cropped to the detected vehicle and encoded by a frozen DINOv2
backbone (``BACKBONE`` at ``INPUT_SIZE`` px, recorded in every checkpoint) into a
grid of patch tokens. A logistic head scores
each return-photo patch for visible physical damage; the pair score is the
highest 3x3-smoothed patch score. Training uses the reviewers' red boxes on the
return photo as positives, every other return patch and the blue "normal"
boxes on the pickup photo as negatives, plus patches from no-claim rental pairs.

Scoring does not need the two photos to be geometrically aligned, which is why
it still works when the ORB/homography stage reports "incompatible". A
detection only routes the pair to human review; it never confirms damage.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError

from app.core.config import get_settings
from app.schemas.damage import DamageCandidate, DamagePairSpecialistResult
from app.schemas.operations import LearningSample, LearningStatus, LearningTask, TrainingRunStatus
from app.services.dataset_paths import resolve_dataset_path
from app.services.operations_store import DAMAGE_PAIR_TRAINING_LABELS, OperationsStore
from app.services.runtime_settings import PROJECT_ROOT

ARCHITECTURE = "dinov2_patch_damage_probe"
# First checkpoints (2026-09-27) were written before the backbone became configurable.
ACCEPTED_ARCHITECTURES = {ARCHITECTURE, "dinov2_vits14_patch_damage_probe"}
CLASSES = ["no_new_damage", "confirmed_new_damage"]
# Members trained for new runs: (backbone, input px, positive top fraction, weight,
# feature mode). Inference follows whatever members the loaded checkpoint recorded.
# On the refined labels (2026-09-27, 10-fold CV, 166 held-out no-claim pairs)
# averaging ViT-S (all box patches) with ViT-B (top 30%) gave AUROC 0.88,
# localisation 90% and test FPR 6% versus 0.86 / 70% / 12% for the best single
# model, at about 3 s per pair on the deployment CPU. 644 px inputs were no better.
# With 9 boxed damages (2026-09-28) adding the pickup contrast raised CV recall at
# the 10% calibration from 44% to 78%: every dent and the small scratch had been
# missed by return-only features.
ENSEMBLE: tuple[tuple[str, int, float, float, str], ...] = (
    ("dinov2_vits14", 448, 1.0, 0.5, "contrast"),
    ("dinov2_vitb14", 448, 0.3, 0.5, "contrast"),
)
BACKBONE, INPUT_SIZE = ENSEMBLE[0][0], ENSEMBLE[0][1]
PATCH = 14
GRID = INPUT_SIZE // PATCH
CROP_MARGIN = 0.04
MODEL_PATH = PROJECT_ROOT / "models" / "damage_pair_specialist.pt"
CACHE_ROOT = PROJECT_ROOT / "data" / "active_learning" / "damage_pair_features"
RUN_ROOT = PROJECT_ROOT / "data" / "active_learning" / "runs"
# Fixed test set. ``claim_folder`` points at the reviewer-curated claim folder:
# sub-folders ending in _TEST are test pairs (new damage), _BAD are excluded
# from everything. Test pairs never train, not even as weak negatives, and every
# run reports how the new model scores them. Nothing here records a plate.
HOLDOUT_FILE = PROJECT_ROOT / "data" / "active_learning" / "damage_pair_holdout.json"
ENTERPRISE_MANIFEST = PROJECT_ROOT / "data" / "view_dataset" / "local_damage_manifest.csv"
SUPPLEMENTAL_PAIRS = (
    PROJECT_ROOT / "outputs" / "supplemental_500_20260923" / "matched_damage_pair_review_queue.csv"
)
# The pair threshold is the score that only this share of held-out no-claim
# pairs reach, so roughly one normal return in ten is still sent to a human.
TARGET_CALIBRATION_FPR = 0.10
L2 = 0.01
AUGMENTED_VARIANTS = 3
NEGATIVE_PATCHES_PER_IMAGE = 48
# A member's positive top fraction below 1.0 means a second fit keeps only that
# share of each red box's most damage-like patches: loose boxes are mostly paint.
MAX_FOLDS = 10
# Repeated CV: (fold shuffle seed, patch sampling seed) per repeat. The first
# repeat is the single CV used before 2026-09-28.
CV_REPEATS = ((7, 20260927), (11, 11), (23, 23))
# Absolute floor for a first deployment and every later candidate.
MIN_RECALL = 0.60
MIN_AUROC = 0.75
MIN_LOCALIZATION = 0.50
MAX_TEST_FPR = 0.25
# Operator targets shown next to every model in model management; a model that
# meets recall, accuracy and time together is picked by damage_pair_autolearn.py.
TARGET_ACCURACY = 0.95
MAX_INSPECTION_SECONDS = 30.0
PAIRS_PER_INSPECTION = 4  # one return inspection compares the four corner views
# Candidates an administrator can switch online inference to (model management).
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "damage_pair"
# Relative guard against the deployed model on the same fixed test negatives.
MAX_RECALL_DROP = 0.10
MAX_FPR_RISE = 0.05
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EncodedImage:
    tokens: np.ndarray  # (grid*grid, dim) L2-normalised float32
    centers: np.ndarray  # (grid*grid, 2) patch centres, normalised original coords
    valid: np.ndarray  # (grid*grid,) patch lies on the photo, not letterbox padding
    half_extent: tuple[float, float]  # half a patch, normalised original coords


def _load_backbone(name: str):
    import torch

    cache_dir = PROJECT_ROOT / ".cache" / "torch"
    torch.hub.set_dir(str(cache_dir))
    local_repository = cache_dir / "facebookresearch_dinov2_main"
    if local_repository.exists():
        return torch.hub.load(str(local_repository), name, source="local", pretrained=True)
    return torch.hub.load("facebookresearch/dinov2", name, trust_repo=True, pretrained=True)


def backbone_stride(name: str) -> int:
    return PATCH if name.startswith("dinov2_") else 16


class _Extractor:
    """Frozen backbone: [0, 1] pixels in, L2-normalised patch tokens out.

    ``dinov2_*`` loads through torch.hub; ``timm:<name>`` and ``hf-vit:<repo>``
    (checkpoints from scripts/damage_pair_autolearn.py) download from Hugging
    Face into ``HF_HOME``."""

    def __init__(self, name: str) -> None:
        mean, std = MEAN, STD
        self.prefix = 0
        if name.startswith("dinov2_"):
            model, self.kind = _load_backbone(name), "dinov2"
        elif name.startswith("timm:"):
            import timm

            arch = name.split(":", 1)[1]
            if arch.startswith("convnext"):
                model = timm.create_model(
                    arch, pretrained=True, features_only=True, out_indices=(2,)
                )
                self.kind = "cnn"
            else:
                model = timm.create_model(arch, pretrained=True, num_classes=0)
                self.kind, self.prefix = "vit", model.num_prefix_tokens
            mean = np.array(model.pretrained_cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
            std = np.array(model.pretrained_cfg["std"], dtype=np.float32).reshape(3, 1, 1)
        elif name.startswith("hf-vit:"):
            from transformers import AutoImageProcessor, ViTModel

            repo = name.split(":", 1)[1]
            model = ViTModel.from_pretrained(repo, add_pooling_layer=False)
            processor = AutoImageProcessor.from_pretrained(repo)
            mean = np.array(processor.image_mean, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(processor.image_std, dtype=np.float32).reshape(3, 1, 1)
            self.kind = "hf-vit"
        else:
            raise ValueError(f"車損比對模型不支援骨幹 {name}")
        self.model = model.eval()
        self.mean, self.std = mean, std

    def tokens(self, pixels: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy((pixels - self.mean) / self.std)[None]
        with torch.inference_mode():
            if self.kind == "dinov2":
                tokens = self.model.forward_features(tensor)["x_norm_patchtokens"][0]
            elif self.kind == "cnn":
                tokens = self.model(tensor)[0][0].flatten(1).T
            elif self.kind == "vit":
                tokens = self.model.forward_features(tensor)[0, self.prefix:]
            else:
                output = self.model(pixel_values=tensor, interpolate_pos_encoding=True)
                tokens = output.last_hidden_state[0, 1:]
            return torch.nn.functional.normalize(tokens.float(), dim=1).numpy().astype(np.float32)


class PatchEncoder:
    """Vehicle crop + frozen-backbone patch tokens, shared by training and inference."""

    def __init__(self, backbone: str = BACKBONE, input_size: int = INPUT_SIZE) -> None:
        self.backbone = backbone
        self.input_size = input_size
        self._backbone = None
        self._lock = Lock()

    @property
    def tag(self) -> str:
        # Cache folder name: timm:/hf-vit: names carry characters Windows paths reject.
        safe = self.backbone.replace(":", "_").replace("/", "_")
        return f"{safe}_{self.input_size}c"

    def _model(self) -> _Extractor:
        if self._backbone is None:
            with self._lock:
                if self._backbone is None:
                    self._backbone = _Extractor(self.backbone)
        return self._backbone

    @staticmethod
    def _vehicle_box(image: Image.Image) -> tuple[float, float, float, float]:
        from urllib.error import URLError

        from app.services.vehicle_coverage import get_vehicle_coverage_detector

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        try:
            prediction = get_vehicle_coverage_detector().predict(buffer.getvalue())
        except (RuntimeError, ValueError, OSError, URLError):
            return (0.0, 0.0, 1.0, 1.0)
        if not prediction.car_detected:
            return (0.0, 0.0, 1.0, 1.0)
        x1, y1, x2, y2 = prediction.bounding_box
        return (
            max(0.0, x1 - CROP_MARGIN),
            max(0.0, y1 - CROP_MARGIN),
            min(1.0, x2 + CROP_MARGIN),
            min(1.0, y2 + CROP_MARGIN),
        )

    def encode(self, image: Image.Image, box: tuple[float, float, float, float] | None = None) -> EncodedImage:
        bx0, by0, bx1, by1 = box or self._vehicle_box(image)
        full_width, full_height = image.size
        crop = image.crop(
            (
                round(bx0 * full_width),
                round(by0 * full_height),
                max(round(bx0 * full_width) + 1, round(bx1 * full_width)),
                max(round(by0 * full_height) + 1, round(by1 * full_height)),
            )
        )
        size = self.input_size
        scale = size / max(crop.size)
        resized = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.BICUBIC,
        )
        canvas = Image.new("RGB", (size, size))
        left = (size - resized.width) // 2
        top = (size - resized.height) // 2
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tokens = self._model().tokens(array)
        stride = backbone_stride(self.backbone)
        steps = (np.arange(size // stride) + 0.5) * stride
        cx, cy = np.meshgrid(steps, steps)
        nx = ((cx - left) / resized.width).ravel()
        ny = ((cy - top) / resized.height).ravel()
        valid = (nx >= 0) & (nx <= 1) & (ny >= 0) & (ny <= 1)
        crop_width, crop_height = bx1 - bx0, by1 - by0
        centers = np.stack([bx0 + nx * crop_width, by0 + ny * crop_height], axis=1)
        half = (
            stride / 2 / resized.width * crop_width,
            stride / 2 / resized.height * crop_height,
        )
        return EncodedImage(tokens, centers.astype(np.float32), valid, half)


_encoders: dict[tuple[str, int], PatchEncoder] = {}
_encoders_lock = Lock()


def get_encoder(backbone: str = BACKBONE, input_size: int = INPUT_SIZE) -> PatchEncoder:
    key = (backbone, input_size)
    with _encoders_lock:
        if key not in _encoders:
            _encoders[key] = PatchEncoder(backbone, input_size)
        return _encoders[key]


def _open(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("車損比對模型無法讀取照片") from exc


def _grid(values: np.ndarray) -> int:
    return int(round(np.sqrt(len(values))))


def _pool(scores: np.ndarray) -> np.ndarray:
    size = _grid(scores)
    grid = np.pad(scores.reshape(size, size), 1)
    return (sum(grid[i : i + size, j : j + size] for i in range(3) for j in range(3)) / 9).ravel()


def _valid_mode(mode: str) -> bool:
    return mode in {"plain", "contrast"} or (mode.startswith("local") and mode[5:].isdigit())


def _local_window(mode: str) -> int | None:
    return int(mode[5:]) if mode.startswith("local") else None


def _features(
    query: EncodedImage,
    reference: EncodedImage | None,
    mode: str,
    nearest: np.ndarray | None = None,
    flip: bool = False,
) -> np.ndarray:
    """Patch features for one head (same as scripts/damage_pair_autolearn.py).

    ``contrast`` appends each patch minus its most similar patch anywhere on the
    other photo of the pair, so marks already visible at pickup cancel out.
    ``local<N>`` only matches patches within N grid cells of the same spot, which
    compares a panel with the same panel instead of any clean paint elsewhere."""
    if mode == "plain":
        return query.tokens
    if reference is None:
        raise ValueError("pickup comparison features need the other photo of the pair")
    if nearest is None:
        nearest = _nearest(query, reference, _local_window(mode), flip)
    return np.concatenate([query.tokens, query.tokens - reference.tokens[nearest]], axis=1)


def _nearest(
    query: EncodedImage, reference: EncodedImage, window: int | None = None, flip: bool = False
) -> np.ndarray:
    """Index into ``reference.tokens`` of each query patch's most similar valid patch;
    ``flip`` means the query is a mirrored augmentation of the reference's view."""
    similarity = query.tokens @ reference.tokens.T
    similarity[:, ~reference.valid] = -np.inf
    best = similarity.argmax(1)
    if window is not None:
        size = _grid(query.tokens)
        index = np.arange(size * size)
        rows, cols = index // size, index % size
        query_cols = size - 1 - cols if flip else cols
        far = (np.abs(rows[:, None] - rows[None]) > window) | (np.abs(query_cols[:, None] - cols[None]) > window)
        local = np.where(far, -np.inf, similarity)
        best = np.where(np.isfinite(local.max(1)), local.argmax(1), best)
    return best


def _patch_probabilities(
    weights: np.ndarray, bias: float, encoded: EncodedImage, features: np.ndarray | None = None
) -> np.ndarray:
    logits = (encoded.tokens if features is None else features) @ weights + bias
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    probabilities[~encoded.valid] = 0.0
    return probabilities


def _pair_score(weights: np.ndarray, bias: float, encoded: EncodedImage) -> tuple[float, int, np.ndarray]:
    pooled = _pool(_patch_probabilities(weights, bias, encoded))
    top = int(np.argmax(pooled))
    return float(pooled[top]), top, pooled


def _regions(
    pooled: np.ndarray, encoded: EncodedImage, threshold: float, limit: int = 3
) -> list[tuple[list[float], float]]:
    import cv2

    size = _grid(pooled)
    mask = (pooled.reshape(size, size) >= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    regions: list[tuple[list[float], float]] = []
    half_x, half_y = encoded.half_extent
    for label in range(1, count):
        members = np.where(labels.ravel() == label)[0]
        members = members[encoded.valid[members]]
        if not len(members):
            continue
        xs, ys = encoded.centers[members, 0], encoded.centers[members, 1]
        box = [
            round(float(max(0.0, xs.min() - half_x)), 4),
            round(float(max(0.0, ys.min() - half_y)), 4),
            round(float(min(1.0, xs.max() + half_x)), 4),
            round(float(min(1.0, ys.max() + half_y)), 4),
        ]
        regions.append((box, float(pooled[members].max())))
    return sorted(regions, key=lambda item: item[1], reverse=True)[:limit]


def _annotate(image: Image.Image, regions: list[tuple[list[float], float]]) -> str:
    preview = image.copy()
    preview.thumbnail((1280, 1280))
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    line = max(3, round(min(width, height) / 220))
    for index, (box, _score) in enumerate(regions, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline=(255, 140, 0), width=line)
        draw.text((x1 * width + line + 2, max(0, y1 * height - 16)), f"AI-{index:02d}", fill=(255, 140, 0))
    output = BytesIO()
    preview.save(output, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@dataclass(frozen=True)
class Member:
    """One backbone's patch head; the pair map is the weighted mean of members."""

    backbone: str
    input_size: int
    weights: np.ndarray
    bias: float
    weight: float
    positive_top_fraction: float = 1.0
    feature_mode: str = "plain"

    @property
    def encoder(self) -> PatchEncoder:
        return get_encoder(self.backbone, self.input_size)


def _mixed_map(
    members: list[Member],
    encodings: list[EncodedImage],
    references: list[EncodedImage] | None = None,
) -> np.ndarray:
    """Weighted member map for ``encodings``; ``references`` are the pair's other
    photo per member, required only by contrast members."""
    total = sum(member.weight for member in members)
    references = references or [None] * len(members)
    return sum(
        member.weight / total * _pool(
            _patch_probabilities(
                member.weights, member.bias, encoded, _features(encoded, reference, member.feature_mode)
            )
        )
        for member, encoded, reference in zip(members, encodings, references, strict=True)
    )


def _members_from_checkpoint(checkpoint: dict) -> list[Member]:
    if "members" in checkpoint:
        return [
            Member(
                backbone=str(item["backbone"]),
                input_size=int(item["input_size"]),
                weights=item["weights"].numpy().astype(np.float32),
                bias=float(item["bias"]),
                weight=float(item["weight"]),
                positive_top_fraction=float(item.get("positive_top_fraction", 1.0)),
                feature_mode=str(item.get("feature_mode", "plain")),
            )
            for item in checkpoint["members"]
        ]
    # Single-backbone checkpoints written before ensembles existed.
    return [
        Member(
            backbone=str(checkpoint.get("backbone", "dinov2_vits14")),
            input_size=int(checkpoint.get("input_size", 448)),
            weights=checkpoint["weights"].numpy().astype(np.float32),
            bias=float(checkpoint["bias"]),
            weight=1.0,
        )
    ]


def _load_checkpoint(path: Path) -> dict | None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in ACCEPTED_ARCHITECTURES:
        return None
    modes = {item.get("feature_mode", "plain") for item in checkpoint.get("members", [])}
    if not all(_valid_mode(str(mode)) for mode in modes):
        return None
    return checkpoint


def _encode_members(
    members: list[Member], image: Image.Image, box: tuple[float, float, float, float]
) -> list[EncodedImage]:
    """One encoding per distinct backbone; members sharing it (plain + contrast) reuse it."""
    cache: dict[tuple[str, int], EncodedImage] = {}
    for member in members:
        key = (member.backbone, member.input_size)
        if key not in cache:
            cache[key] = member.encoder.encode(image, box)
    return [cache[(member.backbone, member.input_size)] for member in members]


@dataclass(frozen=True)
class LoadedSpecialist:
    members: list[Member]
    threshold: float
    model_version: str
    mtime_ns: int


class DamagePairSpecialist:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._loaded: LoadedSpecialist | None = None
        self._lock = Lock()

    def _load(self) -> LoadedSpecialist | None:
        if not self.model_path.is_file():
            return None
        mtime_ns = self.model_path.stat().st_mtime_ns
        with self._lock:
            if self._loaded is None or self._loaded.mtime_ns != mtime_ns:
                checkpoint = _load_checkpoint(self.model_path)
                if checkpoint is None:
                    return None
                self._loaded = LoadedSpecialist(
                    members=_members_from_checkpoint(checkpoint),
                    threshold=float(checkpoint["threshold"]),
                    model_version=str(checkpoint["model_version"]),
                    mtime_ns=mtime_ns,
                )
        return self._loaded

    def available(self) -> bool:
        return self._load() is not None

    def predict(self, pickup_content: bytes, return_content: bytes) -> DamagePairSpecialistResult | None:
        loaded = self._load()
        if loaded is None:
            return None
        started = perf_counter()
        pickup_image = _open(pickup_content)
        return_image = _open(return_content)
        return_box = PatchEncoder._vehicle_box(return_image)
        returned = _encode_members(loaded.members, return_image, return_box)
        pickup_box = PatchEncoder._vehicle_box(pickup_image)
        pickup = _encode_members(loaded.members, pickup_image, pickup_box)
        pooled = _mixed_map(loaded.members, returned, pickup)
        top = int(np.argmax(pooled))
        score = float(pooled[top])
        detected = score >= loaded.threshold
        regions = _regions(pooled, returned[0], loaded.threshold) if detected else []
        # Evidence for the reviewer: how damage-like the most similar pickup
        # patch looks. High values suggest the mark was already there.
        pickup_pooled = _mixed_map(loaded.members, pickup, returned)
        valid = np.where(pickup[0].valid)[0]
        similar = int(np.argmax(pickup[0].tokens[valid] @ returned[0].tokens[top]))
        pickup_score = float(pickup_pooled[valid[similar]])
        return DamagePairSpecialistResult(
            detected=detected,
            score=round(score, 4),
            threshold=round(loaded.threshold, 4),
            pickup_match_score=round(pickup_score, 4),
            regions=[
                DamageCandidate(
                    id=f"AI-{index:02d}",
                    bounding_box=box,
                    area_ratio=round((box[2] - box[0]) * (box[3] - box[1]), 5),
                    change_score=round(region_score, 4),
                    description="車損比對模型判斷此區域可能有新增損傷，需人工放大確認。",
                )
                for index, (box, region_score) in enumerate(regions, start=1)
            ],
            annotated_return_image=_annotate(return_image, regions) if regions else None,
            model_version=loaded.model_version,
            latency_ms=round((perf_counter() - started) * 1000, 1),
        )


_specialist: DamagePairSpecialist | None = None


def get_damage_pair_specialist() -> DamagePairSpecialist:
    global _specialist
    if _specialist is None:
        _specialist = DamagePairSpecialist()
    return _specialist


# --------------------------------------------------------------------- training


_hash_cache: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_cache:
        _hash_cache[key] = _hash_file(path)
    return _hash_cache[key]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _augment(image: Image.Image, seed: str) -> tuple[Image.Image, bool]:
    rng = random.Random(seed)
    flip = rng.random() < 0.5
    if flip:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    return image, flip


def _cached_encode(
    path: Path, variant: int = 0, *, pair_seed: str = "", encoder: PatchEncoder | None = None
) -> tuple[EncodedImage, bool]:
    """Encode one photo (or one augmented copy) through a per-content disk cache."""
    encoder = encoder or get_encoder()
    digest = _sha256(path)
    key = f"{digest}_v{variant}_{hashlib.sha1(pair_seed.encode()).hexdigest()[:8]}" if variant else f"{digest}_v0"
    cache_path = CACHE_ROOT / encoder.tag / f"{key}.npz"
    if cache_path.is_file():
        data = np.load(cache_path)
        return (
            EncodedImage(
                data["tokens"].astype(np.float32),
                data["centers"],
                data["valid"].astype(bool),
                (float(data["half"][0]), float(data["half"][1])),
            ),
            bool(data["flip"]),
        )
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    flip = False
    if variant:
        image, flip = _augment(image, f"{pair_seed}:{variant}")
    encoded = encoder.encode(image)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        tokens=encoded.tokens.astype(np.float16),
        centers=encoded.centers,
        valid=encoded.valid,
        half=np.array(encoded.half_extent, dtype=np.float32),
        flip=np.array(flip),
    )
    temporary.replace(cache_path)
    return encoded, flip


def _in_boxes(encoded: EncodedImage, boxes: list[dict], flip: bool) -> np.ndarray:
    from app.core.annotation_geometry import points_in_regions

    return points_in_regions(encoded.centers, boxes, flip=flip) & encoded.valid


def _annotations(sample: LearningSample) -> list[dict]:
    review = sample.metadata.get("human_review")
    if not isinstance(review, dict):
        return []
    return [item for item in review.get("annotations") or [] if isinstance(item, dict)]


@dataclass
class HumanPair:
    sample: LearningSample
    pickup: Path
    returned: Path
    positive: bool
    issue_boxes: list[dict]
    normal_boxes: list[dict]


@dataclass
class WeakPair:
    key: str
    pickup: Path
    returned: Path
    split: str  # train | val | test
    source: str


@dataclass
class HoldoutPair:
    key: str  # sha256 prefix of the return photo; never a plate
    pickup: Path
    returned: Path
    truth: str | None  # confirmed_new_damage | no_new_damage | None (not yet confirmed)
    issue_boxes: list[dict]
    note: str
    sample_id: str | None = None


ANGLE_WORDS = {"左前": "左前", "右前": "右前", "左後": "左後", "右後": "右後"}


def _holdout_config() -> dict:
    if not HOLDOUT_FILE.is_file():
        return {}
    return json.loads(HOLDOUT_FILE.read_text(encoding="utf-8"))


def _folder_pairs(folder: Path) -> list[tuple[Path, Path, str]]:
    """(pickup, return, angle) for every 借/還 photo pair in one case folder."""
    photos = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".jfif", ".webp"}]
    pairs = []
    for angle in ANGLE_WORDS:
        pickups = [path for path in photos if angle in path.stem and "借" in path.stem]
        returns = sorted(path for path in photos if angle in path.stem and "還" in path.stem)
        if pickups:
            pairs.extend((pickups[0], returned, angle) for returned in returns)
    return pairs


def excluded_folder_hashes() -> set[str]:
    """Photos in _TEST and _BAD claim folders; neither may ever train."""
    root = _holdout_config().get("claim_folder")
    if not root or not Path(root).is_dir():
        return set()
    return {
        _sha256(path)
        for folder in Path(root).iterdir()
        if folder.is_dir() and folder.name.endswith(("_TEST", "_BAD"))
        for pair in _folder_pairs(folder)
        for path in pair[:2]
    }


def holdout_pairs(store: OperationsStore) -> list[HoldoutPair]:
    config = _holdout_config()
    pairs: list[HoldoutPair] = []
    root = config.get("claim_folder")
    if root and Path(root).is_dir():
        for folder in sorted(Path(root).iterdir()):
            if not (folder.is_dir() and folder.name.endswith("_TEST")):
                continue
            for pickup, returned, angle in _folder_pairs(folder):
                pairs.append(
                    HoldoutPair(
                        key=_sha256(returned)[:12],
                        pickup=pickup,
                        returned=returned,
                        truth="confirmed_new_damage",
                        issue_boxes=[],
                        note=f"TEST 資料夾・{angle}",
                    )
                )
    # Reviewer truths (boxes) for test photos that also exist as inbox samples.
    by_hash = {pair.key: pair for pair in pairs}
    if by_hash:
        with store._connect() as connection:
            rows = connection.execute(
                "SELECT id, secondary_image_hash FROM learning_samples WHERE task = ?",
                (LearningTask.DAMAGE_PAIR.value,),
            ).fetchall()
        for row in rows:
            pair = by_hash.get(str(row["secondary_image_hash"] or "").lower()[:12])
            sample = store.get_learning_sample(row["id"]) if pair else None
            if pair and sample:
                review = sample.metadata.get("human_review") or sample.metadata.get("previous_human_review") or {}
                pair.sample_id = sample.id
                pair.issue_boxes = [
                    item for item in review.get("annotations") or []
                    if isinstance(item, dict) and item.get("image_slot") == "secondary" and item.get("kind") == "issue"
                ]
                if sample.corrected_label in DAMAGE_PAIR_TRAINING_LABELS:
                    pair.truth = sample.corrected_label
    return pairs


def _human_pairs(store: OperationsStore, excluded_hashes: set[str] | None = None) -> list[HumanPair]:
    excluded_hashes = excluded_hashes if excluded_hashes is not None else excluded_folder_hashes()
    pairs: list[HumanPair] = []
    for sample, primary, secondary in store.task_training_samples(
        LearningTask.DAMAGE_PAIR, labels=DAMAGE_PAIR_TRAINING_LABELS
    ):
        if secondary is None:
            continue
        if _sha256(primary) in excluded_hashes or _sha256(secondary) in excluded_hashes:
            continue  # a _TEST or _BAD claim photo
        annotations = _annotations(sample)
        positive = sample.corrected_label == "confirmed_new_damage"
        issue = [
            item for item in annotations
            if item.get("image_slot") == "secondary" and item.get("kind") == "issue"
        ]
        if positive and not issue:
            continue  # a positive without a red box cannot teach where damage is
        pairs.append(
            HumanPair(
                sample=sample,
                pickup=primary,
                returned=secondary,
                positive=positive,
                issue_boxes=issue,
                normal_boxes=[
                    item for item in annotations
                    if item.get("image_slot") == "primary" and item.get("kind") == "hard_negative"
                ],
            )
        )
    return pairs


def _supplemental_path(value: str) -> Path | None:
    root = get_settings().supplemental_photo_root
    if root is None or "NEW_Pic500" not in value:
        return None
    return Path(str(root) + value.split("NEW_Pic500", 1)[1])


def _weak_pairs(excluded_hashes: set[str]) -> tuple[list[WeakPair], list[str]]:
    """No-claim rental pairs used as negatives; any pair a human reviewed is dropped."""
    notes: list[str] = []
    pairs: list[WeakPair] = []
    if ENTERPRISE_MANIFEST.is_file():
        with ENTERPRISE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
            grouped: dict[str, dict[str, object]] = {}
            for row in csv.DictReader(handle):
                if row.get("claim_label") != "negative":
                    continue
                group = grouped.setdefault(row["pair_key"], {"split": row["split"]})
                group[row["stage"]] = resolve_dataset_path(row["local_path"])
        missing = 0
        for pair_key, group in sorted(grouped.items()):
            pickup, returned = group.get("borrow"), group.get("return")
            if not isinstance(pickup, Path) or not isinstance(returned, Path):
                continue
            if not pickup.is_file() or not returned.is_file():
                missing += 1
                continue
            pairs.append(WeakPair(f"enterprise:{pair_key}", pickup, returned, str(group["split"]), "enterprise"))
        if missing:
            notes.append(f"外部無理賠配對有 {missing} 組找不到照片（請設定 GUARDIAN_COMPETITION_DATA_ROOT）")
    if SUPPLEMENTAL_PAIRS.is_file():
        found = missing = 0
        with SUPPLEMENTAL_PAIRS.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                pickup = _supplemental_path(row["pickup_external_path"])
                returned = _supplemental_path(row["return_external_path"])
                if pickup is None or returned is None or not pickup.is_file() or not returned.is_file():
                    missing += 1
                    continue
                split = {"validation": "val"}.get(row["split"], row["split"])
                if split == "train":
                    # Experiments (2026-09-27) showed these unlabelled rentals lower
                    # recall when used as training negatives, so they only calibrate
                    # the threshold and measure the false-positive rate.
                    continue
                pairs.append(
                    WeakPair(
                        f"supplemental:{row['case_id']}:{row['predicted_actual_view']}:{index}",
                        pickup, returned, split, "supplemental",
                    )
                )
                found += 1
        if missing:
            notes.append(f"補充 500 組有 {missing} 組配對照片找不到（請設定 GUARDIAN_SUPPLEMENTAL_PHOTO_ROOT）")
    kept = [
        pair for pair in pairs
        if _sha256(pair.pickup) not in excluded_hashes and _sha256(pair.returned) not in excluded_hashes
    ]
    if len(kept) != len(pairs):
        notes.append(f"{len(pairs) - len(kept)} 組弱負樣本已有人工真值，改以人工結果為準")
    return kept, notes


def training_blocker(store: OperationsStore) -> str | None:
    pairs = _human_pairs(store)
    if not any(pair.positive for pair in pairs):
        return "前後車損比對至少需要一筆在還車照框出紅色車損區域的人工真值"
    return None


class _Dataset:
    """Encodes training material once so every fold reuses the same tokens."""

    def __init__(
        self,
        humans: list[HumanPair],
        weak: list[WeakPair],
        progress,
        encoder: PatchEncoder | None = None,
        feature_mode: str = "plain",
    ) -> None:
        encoder = encoder or get_encoder()
        self.encoder = encoder
        self.feature_mode = feature_mode
        # Contrast features are rebuilt per fit; only the nearest-patch indices
        # are kept (full features for every photo would not fit in memory).
        self._nearest: dict[tuple[int, int], np.ndarray] = {}
        self.humans = humans
        self.weak = weak
        self.human_variants: dict[str, list[tuple[EncodedImage, bool]]] = {}
        self.human_pickup: dict[str, EncodedImage] = {}
        self.weak_encoded: dict[str, tuple[EncodedImage, EncodedImage]] = {}
        total = len(humans) * (1 + AUGMENTED_VARIANTS) + len(weak)
        done = 0
        for pair in humans:
            variants = []
            count = 1 + (AUGMENTED_VARIANTS if pair.positive else 0)
            for variant in range(count):
                encoded, flip = _cached_encode(
                    pair.returned, variant, pair_seed=pair.sample.id, encoder=encoder
                )
                variants.append((encoded, flip))
                done += 1
            self.human_variants[pair.sample.id] = variants
            self.human_pickup[pair.sample.id] = _cached_encode(pair.pickup, encoder=encoder)[0]
            progress(done, total)
        for pair in weak:
            self.weak_encoded[pair.key] = (
                _cached_encode(pair.pickup, encoder=encoder)[0],
                _cached_encode(pair.returned, encoder=encoder)[0],
            )
            done += 1
            if done % 25 == 0:
                progress(done, total)

    def features(self, query: EncodedImage, reference: EncodedImage, flip: bool = False) -> np.ndarray:
        if self.feature_mode == "plain":
            return query.tokens
        key = (id(query), id(reference))
        if key not in self._nearest:
            self._nearest[key] = _nearest(query, reference, _local_window(self.feature_mode), flip)
        return _features(query, reference, self.feature_mode, self._nearest[key])

    def matrix(
        self,
        human_ids: set[str],
        rng: np.random.Generator,
        refine: tuple[np.ndarray, float] | None = None,
        fraction: float = 1.0,
    ):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ws: list[np.ndarray] = []

        def add(tokens: np.ndarray, label: float, mass: float) -> None:
            if len(tokens):
                xs.append(tokens)
                ys.append(np.full(len(tokens), label))
                ws.append(np.full(len(tokens), mass / len(tokens)))

        for pair in self.humans:
            if pair.sample.id not in human_ids:
                continue
            pickup = self.human_pickup[pair.sample.id]
            for encoded, flip in self.human_variants[pair.sample.id]:
                features = self.features(encoded, pickup, flip)
                issue = _in_boxes(encoded, pair.issue_boxes, flip)
                background = np.where(encoded.valid & ~issue)[0]
                background = rng.choice(background, min(len(background), 160), replace=False)
                positives = np.where(issue)[0]
                if refine is not None and fraction < 1.0 and len(positives):
                    scores = features[positives] @ refine[0] + refine[1]
                    keep = max(1, round(len(positives) * fraction))
                    positives = positives[np.argsort(scores)[-keep:]]
                add(features[positives], 1.0, 1.0)
                add(features[background], 0.0, 1.0)
            if pair.normal_boxes:
                features = self.features(pickup, self.human_variants[pair.sample.id][0][0])
                add(features[_in_boxes(pickup, pair.normal_boxes, False)], 0.0, 0.5)
        for pair in self.weak:
            if pair.split != "train":
                continue
            pickup, returned = self.weak_encoded[pair.key]
            for encoded, reference in ((pickup, returned), (returned, pickup)):
                features = self.features(encoded, reference)
                index = np.where(encoded.valid)[0]
                chosen = rng.choice(index, min(len(index), NEGATIVE_PATCHES_PER_IMAGE), replace=False)
                add(features[chosen], 0.0, 1.0)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        w[y == 1] /= max(1e-12, w[y == 1].sum())
        w[y == 0] /= max(1e-12, w[y == 0].sum())
        return x, y, w


@dataclass
class _Ensemble:
    """One encoded dataset per member backbone, sharing the same pairs."""

    datasets: list[_Dataset]
    fractions: list[float]
    mix: list[float]

    @property
    def weak(self) -> list[WeakPair]:
        return self.datasets[0].weak

    def fit(self, human_ids: set[str], rng: np.random.Generator) -> list[Member]:
        members = []
        for dataset, fraction, weight in zip(self.datasets, self.fractions, self.mix, strict=True):
            weights, bias = _fit(*dataset.matrix(human_ids, rng))
            if fraction < 1.0:
                weights, bias = _fit(
                    *dataset.matrix(human_ids, rng, refine=(weights, bias), fraction=fraction)
                )
            members.append(
                Member(
                    dataset.encoder.backbone, dataset.encoder.input_size,
                    weights, bias, weight, fraction, dataset.feature_mode,
                )
            )
        return members

    def _map(self, members: list[Member], pairs: list[tuple[EncodedImage, EncodedImage]]) -> np.ndarray:
        """Weighted member map; ``pairs`` holds (query, reference) per member dataset."""
        total = sum(member.weight for member in members)
        return sum(
            member.weight / total * _pool(
                _patch_probabilities(member.weights, member.bias, query, dataset.features(query, reference))
            )
            for member, dataset, (query, reference) in zip(members, self.datasets, pairs, strict=True)
        )

    def weak_scores(self, members: list[Member], split: str) -> list[float]:
        return [
            float(self._map(members, [(d.weak_encoded[pair.key][1], d.weak_encoded[pair.key][0])
                                      for d in self.datasets]).max())
            for pair in self.weak
            if pair.split == split
        ]

    def human_map(self, members: list[Member], sample_id: str) -> tuple[np.ndarray, EncodedImage, bool]:
        encoded, flip = self.datasets[0].human_variants[sample_id][0]
        pooled = self._map(
            members,
            [(d.human_variants[sample_id][0][0], d.human_pickup[sample_id]) for d in self.datasets],
        )
        return pooled, encoded, flip


def _fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    import torch

    mean = x.mean(0)
    std = x.std(0) + 1e-6
    features = torch.tensor((x - mean) / std, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)
    weights = torch.tensor(w, dtype=torch.float32)
    torch.manual_seed(0)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                head(features).squeeze(1), labels, reduction="none"
            )
            * weights
        ).sum() + L2 * head.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    raw = head.weight.detach().numpy()[0]
    # Fold the standardisation into the head so inference is one dot product.
    return (raw / std).astype(np.float32), float(head.bias.item() - (raw * mean / std).sum())


def _threshold(calibration: list[float]) -> float:
    return float(np.quantile(np.asarray(calibration), 1 - TARGET_CALIBRATION_FPR))


def _auroc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        return 0.0
    pos = np.asarray(positives)[:, None]
    neg = np.asarray(negatives)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def evaluate_and_train(
    humans: list[HumanPair], ensemble: _Ensemble, progress
) -> tuple[dict, list[Member]]:
    """Repeated grouped cross-validation on the human truths, then a final fit on everything.

    With a handful of boxed damages one pair moves recall by about ten points, so
    every CV metric is the mean over ``CV_REPEATS`` different fold shuffles."""
    positives = [pair for pair in humans if pair.positive]
    negatives = [pair for pair in humans if not pair.positive]
    folds = max(2, min(MAX_FOLDS, len(positives)))
    held_scores: list[bool] = []
    held_hits: list[bool] = []
    held_negative_flags: list[bool] = []
    fold_fprs: list[float] = []
    fold_aurocs: list[float] = []
    for repeat, (shuffle_seed, rng_seed) in enumerate(CV_REPEATS):
        rng = np.random.default_rng(rng_seed)
        order = list(positives)
        random.Random(shuffle_seed).shuffle(order)
        fold_of = {pair.sample.id: index % folds for index, pair in enumerate(order)}
        for index, pair in enumerate(negatives):
            fold_of[pair.sample.id] = index % folds
        for fold in range(folds):
            train_ids = {pair.sample.id for pair in humans if fold_of[pair.sample.id] != fold}
            members = ensemble.fit(train_ids, rng)
            threshold = _threshold(ensemble.weak_scores(members, "val"))
            test_negatives = ensemble.weak_scores(members, "test")
            fold_fprs.append(float(np.mean([score >= threshold for score in test_negatives])))
            fold_positive_scores = []
            for pair in humans:
                if fold_of[pair.sample.id] != fold:
                    continue
                pooled, encoded, flip = ensemble.human_map(members, pair.sample.id)
                top = int(np.argmax(pooled))
                score = float(pooled[top])
                if pair.positive:
                    held_scores.append(score >= threshold)
                    held_hits.append(bool(_in_boxes(encoded, pair.issue_boxes, flip)[top]))
                    fold_positive_scores.append(score)
                else:
                    held_negative_flags.append(score >= threshold)
            if fold_positive_scores:
                fold_aurocs.append(_auroc(fold_positive_scores, test_negatives))
            progress(repeat * folds + fold + 1, len(CV_REPEATS) * folds)
    rng = np.random.default_rng(CV_REPEATS[0][1])
    members = ensemble.fit({pair.sample.id for pair in humans}, rng)
    threshold = _threshold(ensemble.weak_scores(members, "val"))
    final_test = ensemble.weak_scores(members, "test")
    metrics = {
        "human_positive_pairs": len(positives),
        "human_negative_pairs": len(negatives),
        "weak_negative_pairs": {
            split: sum(pair.split == split for pair in ensemble.weak) for split in ("train", "val", "test")
        },
        "members": [
            f"{m.backbone}@{m.input_size} {m.feature_mode} "
            f"top{m.positive_top_fraction:.0%} ×{m.weight}"
            for m in members
        ],
        "folds": folds,
        "cv_repeats": len(CV_REPEATS),
        "cv_recall": float(np.mean(held_scores)) if held_scores else 0.0,
        "cv_localization": float(np.mean(held_hits)) if held_hits else 0.0,
        "cv_test_fpr": float(np.mean(fold_fprs)) if fold_fprs else 1.0,
        "cv_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else 0.0,
        "cv_human_negative_fpr": float(np.mean(held_negative_flags)) if held_negative_flags else None,
        "final_test_fpr": float(np.mean([score >= threshold for score in final_test])) if final_test else 1.0,
        "threshold": threshold,
        "target_calibration_fpr": TARGET_CALIBRATION_FPR,
    }
    return metrics, members


BENCHMARK_PAIRS = 8


def benchmark_members(members: list[Member], pairs: list[tuple[Path, Path]]) -> dict:
    """Mean wall time per pair on this machine: vehicle detection once per photo,
    then each member encodes both photos. Scoring itself is negligible."""
    from app.schemas.damage import DamagePairSpecialistResult  # noqa: F401  (warm import)

    for member in members:
        member.encoder._model()
    detection: list[float] = []
    distinct = list({(member.backbone, member.input_size): member for member in members}.values())
    per_member: dict[str, list[float]] = {member.backbone: [] for member in distinct}
    for pickup_path, return_path in pairs[:BENCHMARK_PAIRS]:
        images = [_open(pickup_path.read_bytes()), _open(return_path.read_bytes())]
        started = perf_counter()
        boxes = [PatchEncoder._vehicle_box(image) for image in images]
        detection.append(perf_counter() - started)
        for member in distinct:
            started = perf_counter()
            for image, box in zip(images, boxes, strict=True):
                member.encoder.encode(image, box)
            per_member[member.backbone].append(perf_counter() - started)
    if not detection:
        return {}
    members_ms = {name: round(float(np.mean(times)) * 1000, 1) for name, times in per_member.items()}
    detection_ms = round(float(np.mean(detection)) * 1000, 1)
    return {
        "average_ms": round(detection_ms + sum(members_ms.values()), 1),
        "vehicle_detection_ms": detection_ms,
        "members_ms": members_ms,
        "pairs": len(detection),
        "measured_at": _now(),
    }


def evaluate_holdout(holdout: list[HoldoutPair], members: list[Member], threshold: float) -> dict:
    """Score the fixed test set with a model that never saw it."""
    rows = []
    for pair in holdout:
        encodings = [_cached_encode(pair.returned, encoder=member.encoder)[0] for member in members]
        pickups = [_cached_encode(pair.pickup, encoder=member.encoder)[0] for member in members]
        pooled = _mixed_map(members, encodings, pickups)
        top = int(np.argmax(pooled))
        score = float(pooled[top])
        hit = bool(_in_boxes(encodings[0], pair.issue_boxes, False)[top]) if pair.issue_boxes else None
        rows.append(
            {
                "key": pair.key,
                "sample_id": pair.sample_id,
                "truth": pair.truth,
                "score": round(score, 4),
                "detected": score >= threshold,
                "top_region_in_red_box": hit,
                "note": pair.note,
            }
        )
    damage = [row for row in rows if row["truth"] == "confirmed_new_damage"]
    clean = [row for row in rows if row["truth"] == "no_new_damage"]
    located = [row for row in damage if row["top_region_in_red_box"] is not None]
    return {
        "holdout": rows,
        "holdout_recall": (sum(row["detected"] for row in damage) / len(damage)) if damage else None,
        "holdout_fpr": (sum(row["detected"] for row in clean) / len(clean)) if clean else None,
        "holdout_localization": (
            sum(row["top_region_in_red_box"] for row in located) / len(located) if located else None
        ),
    }


def _deployed_test_fpr(weak: list[WeakPair]) -> tuple[dict | None, float | None]:
    """The deployed model's false-positive rate on today's fixed test pairs."""
    if not MODEL_PATH.is_file():
        return None, None
    checkpoint = _load_checkpoint(MODEL_PATH)
    if checkpoint is None:
        return None, None
    members = _members_from_checkpoint(checkpoint)
    scores = [
        float(
            _mixed_map(
                members,
                [_cached_encode(pair.returned, encoder=member.encoder)[0] for member in members],
                [_cached_encode(pair.pickup, encoder=member.encoder)[0] for member in members],
            ).max()
        )
        for pair in weak
        if pair.split == "test"
    ]
    threshold = float(checkpoint["threshold"])
    report = json.loads(str(checkpoint.get("report_json", "{}")))
    return report, float(np.mean([score >= threshold for score in scores])) if scores else None


def _promotion_gate(candidate: dict, baseline: dict | None, baseline_fpr: float | None) -> tuple[bool, str]:
    if candidate["cv_recall"] < MIN_RECALL:
        return False, f"交叉驗證召回率 {candidate['cv_recall']:.0%} 未達 {MIN_RECALL:.0%}，禁止發布"
    if candidate["cv_auroc"] < MIN_AUROC:
        return False, f"交叉驗證 AUROC {candidate['cv_auroc']:.2f} 未達 {MIN_AUROC:.2f}，禁止發布"
    if candidate["cv_localization"] < MIN_LOCALIZATION:
        return False, f"車損位置命中率 {candidate['cv_localization']:.0%} 未達 {MIN_LOCALIZATION:.0%}，禁止發布"
    if candidate["final_test_fpr"] > MAX_TEST_FPR:
        return False, f"固定測試集誤報率 {candidate['final_test_fpr']:.0%} 高於 {MAX_TEST_FPR:.0%}，禁止發布"
    if baseline is not None and candidate["cv_recall"] + MAX_RECALL_DROP < float(baseline.get("cv_recall", 0.0)):
        return False, "召回率比現行模型退步過多，禁止發布"
    if baseline_fpr is not None and candidate["final_test_fpr"] > baseline_fpr + MAX_FPR_RISE:
        return False, "固定測試集誤報率比現行模型升高過多，禁止發布"
    return True, "候選模型通過交叉驗證與固定測試集防退步門檻"


def run_damage_pair_training(store: OperationsStore, run_id: str) -> None:
    import torch

    run_directory = RUN_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    candidate_path = run_directory / "damage_pair_candidate.pt"
    report_path = run_directory / "training_report.json"
    store.update_training_run(
        run_id,
        status=TrainingRunStatus.RUNNING,
        started_at=_now(),
        candidate_path=str(candidate_path),
        report_path=str(report_path),
        message="正在整理前後車損比對人工真值",
    )
    holdout = holdout_pairs(store)
    excluded = excluded_folder_hashes()
    humans = _human_pairs(store, excluded)
    if not any(pair.positive for pair in humans):
        raise ValueError(training_blocker(store) or "沒有可訓練的車損比對人工真值")
    reviewed_hashes = excluded | {
        _sha256(path)
        for pair in humans
        for path in (pair.pickup, pair.returned)
    }
    weak, notes = _weak_pairs(reviewed_hashes)
    if sum(pair.split == "val" for pair in weak) < 10 or sum(pair.split == "test" for pair in weak) < 10:
        raise FileNotFoundError(
            "校準／測試用的無理賠配對不足 10 組，無法評估誤報率；" + "；".join(notes)
        )

    def extraction_progress(done: int, total: int) -> None:
        store.update_training_run(run_id, message=f"正在抽取車身特徵 {done}/{total}（已抽取的照片會快取）")

    ensemble = _Ensemble(
        datasets=[
            _Dataset(humans, weak, extraction_progress, get_encoder(backbone, size), mode)
            for backbone, size, _fraction, _weight, mode in ENSEMBLE
        ],
        fractions=[fraction for _b, _s, fraction, _w, _m in ENSEMBLE],
        mix=[weight for _b, _s, _f, weight, _m in ENSEMBLE],
    )

    def fold_progress(done: int, total: int) -> None:
        store.update_training_run(run_id, message=f"交叉驗證 {done}/{total} 折")

    metrics, members = evaluate_and_train(humans, ensemble, fold_progress)
    metrics.update(evaluate_holdout(holdout, members, metrics["threshold"]))
    baseline, baseline_fpr = _deployed_test_fpr(weak)
    metrics["notes"] = notes
    allowed, gate_message = _promotion_gate(metrics, baseline, baseline_fpr)
    report = {"candidate": metrics, "baseline": baseline, "baseline_test_fpr": baseline_fpr}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = {
        "schema_version": 2,
        "task": LearningTask.DAMAGE_PAIR.value,
        "architecture": ARCHITECTURE,
        "backbone": "+".join(member.backbone for member in members),
        "classes": CLASSES,
        "input_size": members[0].input_size,
        "preprocess": "vehicle_crop_letterbox",
        "members": [
            {
                "backbone": member.backbone,
                "input_size": member.input_size,
                "weights": torch.from_numpy(member.weights),
                "bias": member.bias,
                "weight": member.weight,
                "positive_top_fraction": member.positive_top_fraction,
                "feature_mode": member.feature_mode,
            }
            for member in members
        ],
        "threshold": metrics["threshold"],
        "model_version": f"auto-damage-pair-{run_id.lower()}",
        "management_stage": "deployed" if allowed else "trial",
        "management_metric": "human_truth_cv_recall",
        "management_score": round(metrics["cv_recall"], 4),
        "minimum_management_score": MIN_RECALL,
        "production_approved": allowed,
        "label_source": "營運人員框選的取還車車損人工真值＋無理賠配對弱負樣本",
        "sample_ids": [pair.sample.id for pair in humans],
        "report": {"test": {"macro_f1": metrics["cv_recall"], "accuracy": 1 - metrics["final_test_fpr"]}},
        "report_json": json.dumps(
            {key: value for key, value in metrics.items() if key != "holdout"}, ensure_ascii=False
        ),
        "created_at": _now(),
    }
    checkpoint["benchmark"] = benchmark_members(
        members, [(pair.pickup, pair.returned) for pair in weak if pair.split == "test"]
    )
    torch.save(checkpoint, candidate_path)
    summary = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in metrics.items()
        if key not in {"notes", "holdout"}
    }
    if not allowed:
        store.update_training_run(
            run_id,
            status=TrainingRunStatus.REJECTED,
            candidate_metrics=summary,
            baseline_metrics={"test_fpr": baseline_fpr} if baseline_fpr is not None else {},
            message=gate_message,
            completed_at=_now(),
        )
        return
    history = MODEL_PATH.parent / "history"
    if MODEL_PATH.is_file():
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, history / f"damage_pair_before_{run_id}.pt")
    temporary = MODEL_PATH.with_suffix(".next.pt")
    shutil.copy2(candidate_path, temporary)
    temporary.replace(MODEL_PATH)
    store.mark_learning_samples_trained(
        [pair.sample.id for pair in humans if pair.sample.status == LearningStatus.APPROVED]
    )
    store.update_training_run(
        run_id,
        status=TrainingRunStatus.PROMOTED,
        candidate_metrics=summary,
        baseline_metrics={"test_fpr": baseline_fpr} if baseline_fpr is not None else {},
        message=(
            f"{gate_message}，已自動發布：交叉驗證召回 {metrics['cv_recall']:.0%}、"
            f"位置命中 {metrics['cv_localization']:.0%}、固定測試集誤報 {metrics['final_test_fpr']:.0%}"
        ),
        completed_at=_now(),
    )
