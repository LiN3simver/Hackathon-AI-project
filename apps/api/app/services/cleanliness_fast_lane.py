"""Millisecond colour-anomaly lane: flags objects that clash with the cabin's own palette.

Car interiors are dominated by a few low-key colours (black/grey/beige trim). A
foreign object such as a red rope, a shopping bag or a drink cup forms a compact
region whose Lab colour is far from every dominant colour. Finding those regions
takes ~20 ms on CPU, so a hit can skip the minutes-long vision-language grounding.
Dust and footprints share the cabin's colours and stay with the slower models.
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.cleanliness import CleanlinessBox, CleanlinessEvidence, EvidenceCategory

MODEL_NAME = "color-anomaly-v1"
ANALYSIS_SIDE = 320
# Seats and footwells sit below the roof line, windscreen and side windows.
ROI_TOP = 0.30
# Shadows and leather highlights change lightness far more than hue.
LIGHTNESS_WEIGHT = 0.35
PALETTE_SIZE = 4
# A colour covering at least this share of the cabin is trim, not a foreign object.
MIN_PALETTE_SHARE = 0.10
# Larger blobs are sunlight, seat covers or exterior paint rather than litter.
MAX_AREA_RATIO = 0.20
# White sills, grey body paint and leather glare are far in lightness but colourless;
# litter worth flagging is saturated (90th-percentile Lab chroma of the region).
MIN_REGION_CHROMA = 30.0
MAX_FINDINGS = 5


def _color_name(rgb: np.ndarray) -> str:
    hue, saturation, value = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    if saturation < 50:
        return "白色" if value > 170 else "黑色" if value < 60 else "灰色"
    degrees = int(hue) * 2
    for limit, name in ((15, "紅色"), (40, "橘色"), (70, "黃色"), (165, "綠色"), (200, "青色"),
                        (255, "藍色"), (290, "紫色"), (340, "粉紅色")):
        if degrees < limit:
            return name
    return "紅色"


def describe_location(box: CleanlinessBox) -> str:
    x, y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    horizontal = "左側" if x < 1 / 3 else "中央" if x < 2 / 3 else "右側"
    vertical = "中段" if y < .55 else "下方" if y < .8 else "底部"
    return f"照片{horizontal}{vertical}"


def detect_color_anomalies(
    content: bytes, *, color_threshold: float = 20.0, min_area_ratio: float = 0.004,
) -> list[CleanlinessEvidence]:
    """Return boxed evidence for compact regions far from every dominant cabin colour."""
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return []
    image.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    top = int(height * ROI_TOP)
    if height - top < 8 or width < 8:
        return []
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    features = np.stack([lab[..., 0] * (100 / 255) * LIGHTNESS_WEIGHT, lab[..., 1] - 128, lab[..., 2] - 128], -1)

    # Seed clusters by lightness rank so results never depend on OpenCV's global RNG.
    roi = np.ascontiguousarray(features[top:].reshape(-1, 3))
    seeds = (np.argsort(np.argsort(roi[:, 0])) * PALETTE_SIZE // len(roi)).astype(np.int32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(roi, PALETTE_SIZE, seeds, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS)
    share = np.bincount(labels.ravel(), minlength=PALETTE_SIZE) / len(labels)
    palette = centers[share >= MIN_PALETTE_SHARE]
    distance = np.linalg.norm(features[:, :, None, :] - palette[None, None], axis=-1).min(axis=-1)

    mask = (distance > color_threshold).astype(np.uint8)
    mask[:top] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        ratio = area / (height * width)
        if not min_area_ratio <= ratio <= MAX_AREA_RATIO:
            continue
        # Regions cut by the frame are door sills, body paint or the ground outside.
        if x <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1:
            continue
        pixels = components == index
        if np.percentile(np.hypot(features[..., 1][pixels], features[..., 2][pixels]), 90) < MIN_REGION_CHROMA:
            continue
        candidates.append((ratio * float(distance[pixels].mean()), CleanlinessBox(
            x1=x / width, y1=y / height, x2=(x + box_width) / width, y2=(y + box_height) / height,
        ), rgb[pixels].mean(axis=0)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        CleanlinessEvidence(
            category=EvidenceCategory.CLUTTER, severity=1, visible=True, bounding_box=box,
            description=f"{describe_location(box)}有{_color_name(color)}異物，與車內主色差異明顯",
        )
        for _, box, color in candidates[:MAX_FINDINGS]
    ]
