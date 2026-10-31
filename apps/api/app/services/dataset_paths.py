"""Resolve dataset manifest paths that were recorded on another machine."""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


def resolve_dataset_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    settings = get_settings()
    root = settings.competition_data_root
    source_root = settings.dataset_source_root
    if root is None or source_root is None:
        return path
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return path
    candidate = root / relative
    if candidate.is_file():
        return candidate
    return path
