"""Validate uploaded images by their decoded bytes, not browser MIME metadata."""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


async def read_validated_image(image: UploadFile, max_upload_mb: int) -> bytes:
    limit = max_upload_mb * 1024 * 1024
    content = await image.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"圖片不可超過 {max_upload_mb} MB",
        )
    if not content:
        raise HTTPException(status_code=400, detail="圖片內容不可為空")

    try:
        with Image.open(BytesIO(content)) as decoded:
            image_format = decoded.format
            decoded.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="只接受有效的 JPEG、PNG 或 WebP 圖片",
        ) from exc
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="只接受 JPEG、PNG 或 WebP 圖片",
        )
    return content
