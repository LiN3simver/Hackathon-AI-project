from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.config import get_settings


class InvalidInspectionObjectKey(ValueError):
    pass


class InspectionImageStorage:
    """Private filesystem storage; images are never exposed as static files."""

    def __init__(self, root_path: Path | None = None) -> None:
        self.root_path = (root_path or get_settings().inspection_storage_path).resolve()

    @staticmethod
    def _suffix(content: bytes) -> str:
        with Image.open(BytesIO(content)) as image:
            image_format = (image.format or "JPEG").upper()
        return {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }.get(image_format, ".img")

    def save(self, session_id: str, request_id: str, content: bytes) -> str:
        safe_session_id = session_id.replace("/", "").replace("\\", "")
        safe_request_id = request_id.replace("/", "").replace("\\", "")
        if not safe_session_id or not safe_request_id:
            raise InvalidInspectionObjectKey("照片儲存識別碼無效")

        object_key = f"{safe_session_id}/{safe_request_id}{self._suffix(content)}"
        target = self.resolve(object_key, require_exists=False)
        if target is None:
            raise InvalidInspectionObjectKey("照片儲存路徑無效")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return object_key

    def resolve(self, object_key: str, *, require_exists: bool = True) -> Path | None:
        key_path = Path(object_key)
        if key_path.is_absolute() or ".." in key_path.parts:
            raise InvalidInspectionObjectKey("照片儲存路徑無效")
        target = (self.root_path / key_path).resolve()
        if not target.is_relative_to(self.root_path):
            raise InvalidInspectionObjectKey("照片儲存路徑超出允許範圍")
        if require_exists and not target.is_file():
            return None
        return target

    def delete(self, object_key: str) -> None:
        target = self.resolve(object_key)
        if target is not None:
            target.unlink(missing_ok=True)


_storage: InspectionImageStorage | None = None


def get_inspection_storage() -> InspectionImageStorage:
    global _storage
    if _storage is None:
        _storage = InspectionImageStorage()
    return _storage
