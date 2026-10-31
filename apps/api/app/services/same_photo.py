"""Detect a return photo that is only the pickup photo uploaded again."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SAME_AS_PICKUP_MESSAGE = "這張就是取車照，請拍還車當下的新照片。"
# Thumbnails of one photo re-saved or resized differ by about 1 grey level on
# average and at most a few levels anywhere; two handheld shots of the same
# corner differ by 30+ on average, and one new scratch on a copy shows up as a
# local spike, so both limits must hold.
_THUMBNAIL_SIZE = (128, 128)
_MAX_MEAN_DIFFERENCE = 2.0
_MAX_LOCAL_DIFFERENCE = 20.0
_MAX_ASPECT_DIFFERENCE = 0.02


def _thumbnail(content: bytes) -> tuple[np.ndarray, float] | None:
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            aspect = image.width / image.height
            pixels = np.asarray(image.resize(_THUMBNAIL_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError, ZeroDivisionError):
        return None
    return pixels, aspect


def is_same_photo(first: bytes, second: bytes) -> bool:
    """True for identical files or the same picture re-encoded or resized."""
    if first == second:
        return True
    left, right = _thumbnail(first), _thumbnail(second)
    if left is None or right is None:
        return False
    if abs(left[1] - right[1]) > _MAX_ASPECT_DIFFERENCE * max(left[1], right[1]):
        return False
    difference = np.abs(left[0] - right[0])
    return (
        float(difference.mean()) < _MAX_MEAN_DIFFERENCE
        and float(np.percentile(difference, 99.5)) < _MAX_LOCAL_DIFFERENCE
    )
