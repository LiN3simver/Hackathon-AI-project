import json
import os
import platform
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.schemas.readiness import ReadinessReport
from app.services.auth import require_admin
from app.services.readiness import build_readiness_report
from app.services.view_classifier import PROJECT_ROOT

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "guardian-api",
        "python": platform.python_version(),
    }


@router.get("/readiness", response_model=ReadinessReport)
def readiness(_: object = Depends(require_admin)) -> ReadinessReport:
    return build_readiness_report()


@router.get("/capabilities")
def capabilities(_: object = Depends(require_admin)) -> dict[str, object]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_available = False
        gpu_name = None

    local_app_data = os.environ.get("LOCALAPPDATA")
    installed_ollama = shutil.which("ollama")
    if not installed_ollama and local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        try:
            installed_ollama = str(candidate) if candidate.exists() else None
        except OSError:
            # A restricted service account may not be allowed to inspect a
            # per-user installation. Capabilities must remain observable.
            installed_ollama = None

    ollama_models: list[str] = []
    ollama_service_ready = False
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=0.5) as response:  # noqa: S310
            payload = json.load(response)
            ollama_models = [model["name"] for model in payload.get("models", [])]
            ollama_service_ready = True
    except (OSError, URLError, ValueError, KeyError):
        pass

    configured_view_model = get_settings().view_model_path
    view_model_path = (
        configured_view_model
        if configured_view_model.is_absolute()
        else PROJECT_ROOT / configured_view_model
    )
    qwen_ready = any(model.startswith("qwen3-vl:4b") for model in ollama_models)

    return {
        "opencv": {"ready": True, "version": cv2.__version__},
        "pytorch": {
            "ready": torch_version is not None,
            "version": torch_version,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
        },
        "ollama": {
            "installed": installed_ollama is not None,
            "service_ready": ollama_service_ready,
            "models": ollama_models,
            "model_ready": qwen_ready,
        },
        "quality_engine": {"ready": True, "mode": "opencv_baseline"},
        "vehicle_view_model": {
            "ready": view_model_path.exists(),
            "mode": "dinov2_vits14_knn_ensemble" if view_model_path.exists() else "missing",
            "scope": "four exterior corners",
        },
        "plate_model": {
            "ready": ollama_service_ready and qwen_ready,
            "mode": "qwen3_vl_non_blocking" if qwen_ready else "unavailable",
        },
        "vehicle_coverage_model": {
            "ready": True,
            "mode": "fasterrcnn_mobilenet_v3_320_coco",
            "scope": "exterior car bounding box and framing",
        },
    }
