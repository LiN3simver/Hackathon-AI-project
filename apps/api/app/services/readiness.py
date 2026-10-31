from datetime import datetime, timezone

import httpx
import torch

from app.core.config import get_settings
from app.schemas.readiness import ReadinessCheck, ReadinessLevel, ReadinessReport
from app.services.front_cleanliness import ARCHITECTURE as FRONT_CLEANLINESS_ARCHITECTURE
from app.services.front_cleanliness import MODEL_PATH as FRONT_CLEANLINESS_PATH
from app.services.interior_capture import (
    FRONT_SCENE_MODEL_PATH,
    REAR_SCENE_MODEL_PATH,
    SCENE_MODEL_PATH,
)
from app.services.operations_store import get_operations_store
from app.services.rapid_plate_recognition import model_info as plate_model_info
from app.services.rear_cleanliness import MODEL_PATH as REAR_CLEANLINESS_PATH
from app.services.rear_cleanliness import load_rear_checkpoint
from app.services.view_classifier import PROJECT_ROOT


def _check(
    key: str,
    label: str,
    ready: bool,
    message: str,
    *,
    required: bool = True,
    detail: str | None = None,
    warning: bool = False,
) -> ReadinessCheck:
    status = (
        ReadinessLevel.READY
        if ready
        else ReadinessLevel.WARNING
        if warning or not required
        else ReadinessLevel.ERROR
    )
    return ReadinessCheck(
        key=key,
        label=label,
        status=status,
        required=required,
        message=message,
        detail=detail,
    )


def build_readiness_report() -> ReadinessReport:
    settings = get_settings()
    checks: list[ReadinessCheck] = []

    try:
        store = get_operations_store()
        store.initialize()
        with store._connect() as connection:  # Internal readiness probe.
            connection.execute("SELECT 1").fetchone()
        checks.append(_check("database", "營運資料庫", True, "SQLite 可讀寫"))
    except Exception as exc:  # pragma: no cover - defensive system probe
        checks.append(_check("database", "營運資料庫", False, "SQLite 無法讀寫", detail=str(exc)))

    configured_model = settings.view_model_path
    view_model_path = (
        configured_model if configured_model.is_absolute() else PROJECT_ROOT / configured_model
    )
    view_model_ready = view_model_path.exists()
    view_model_message = "缺少視角模型權重"
    view_model_detail = str(view_model_path)
    if view_model_ready:
        try:
            checkpoint = torch.load(view_model_path, map_location="cpu", weights_only=True)
            architecture = str(checkpoint.get("architecture", "unknown"))
            if architecture == "dinov2_vits14_knn_ensemble":
                cache_dir = PROJECT_ROOT / ".cache" / "torch"
                repository = cache_dir / "facebookresearch_dinov2_main"
                backbone_weights = (
                    cache_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
                )
                view_model_ready = repository.exists() and backbone_weights.exists()
                view_model_message = (
                    "DINOv2 v2 權重與骨幹快取已就緒"
                    if view_model_ready
                    else "缺少 DINOv2 骨幹快取，請執行 download_model_weights.py"
                )
                view_model_detail = (
                    f"checkpoint={view_model_path}; backbone={backbone_weights}"
                )
            else:
                view_model_message = f"{architecture} 權重已就緒"
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            view_model_ready = False
            view_model_message = "視角模型 checkpoint 無法讀取"
            view_model_detail = f"{view_model_path}: {exc}"
    checks.append(
        _check(
            "view_model",
            "車外視角模型",
            view_model_ready,
            view_model_message,
            detail=view_model_detail,
        )
    )

    specialist_specs = (
        (FRONT_SCENE_MODEL_PATH, {"interior_front", "not_interior_front"}),
        (REAR_SCENE_MODEL_PATH, {"interior_rear", "not_interior_rear"}),
    )
    interior_model_ready = all(path.exists() for path, _ in specialist_specs)
    interior_model_message = "缺少前座或後座獨立場景模型權重"
    interior_model_detail = "; ".join(str(path) for path, _ in specialist_specs)
    if interior_model_ready:
        try:
            details: list[str] = []
            for path, required_classes in specialist_specs:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                classes = set(checkpoint.get("classes", []))
                architecture = str(checkpoint.get("architecture", ""))
                if classes != required_classes or not architecture.endswith(
                    ("_linear_probe", "_mlp_probe")
                ):
                    interior_model_ready = False
                details.append(
                    f"checkpoint={path}; architecture={architecture}; classes={sorted(classes)}"
                )
            interior_model_message = (
                "前座與後座獨立場景權重已就緒"
                if interior_model_ready
                else "前座或後座 checkpoint 類別或架構不相容"
            )
            interior_model_detail = "; ".join(details)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            interior_model_ready = False
            interior_model_message = "前座或後座 checkpoint 無法讀取"
            interior_model_detail = str(exc)
    elif SCENE_MODEL_PATH.exists():
        interior_model_ready = True
        interior_model_message = "使用舊版前／後座合併場景模型；建議補齊獨立模型"
        interior_model_detail = str(SCENE_MODEL_PATH)
    checks.append(
        _check(
            "interior_capture_model",
            "車內拍攝場景模型",
            interior_model_ready,
            interior_model_message,
            detail=interior_model_detail,
        )
    )

    front_ready = False
    front_detail = str(FRONT_CLEANLINESS_PATH)
    try:
        front_checkpoint = torch.load(FRONT_CLEANLINESS_PATH, map_location="cpu", weights_only=True)
        front_backbone = (PROJECT_ROOT / front_checkpoint["backbone_path"]).resolve()
        front_ready = (front_checkpoint.get("architecture") == FRONT_CLEANLINESS_ARCHITECTURE
                       and bool(front_checkpoint.get("production_approved"))
                       and PROJECT_ROOT in front_backbone.parents
                       and all((front_backbone / name).is_file() for name in
                               ("model.safetensors", "config.json", "tokenizer.json"))
                       # transformers 5 saves the image preprocessing inside processor_config.json.
                       and any((front_backbone / name).is_file() for name in
                               ("preprocessor_config.json", "processor_config.json")))
        front_detail = str(front_checkpoint.get("model_version", ""))
    except (OSError, RuntimeError, KeyError, ValueError, TypeError):
        pass
    checks.append(_check("front_cleanliness_model", "前座髒汙模型", front_ready,
                         "CLIP＋色調 CPU 模型已正式啟用" if front_ready else "前座髒汙模型或骨幹尚未就緒",
                         detail=front_detail))

    try:
        response = httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags",
            timeout=httpx.Timeout(1.2),
        )
        response.raise_for_status()
        model_names = [item.get("name", "") for item in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError, TypeError):
        model_names = []
    rear_ready, rear_detail = False, str(REAR_CLEANLINESS_PATH)
    try:
        rear_checkpoint, _ = load_rear_checkpoint(REAR_CLEANLINESS_PATH)
        rear_ready, rear_detail = True, str(rear_checkpoint["model_version"])
    except (OSError, RuntimeError, KeyError, ValueError, TypeError):
        pass
    checks.append(_check(
        "rear_cleanliness_model", "後座髒汙模型", rear_ready,
        "DINOv2 局部 CPU 模型已啟用，疑似髒污送人工覆核" if rear_ready else "後座髒污模型或骨幹尚未就緒",
        detail=rear_detail,
    ))
    # The assistant answers without it (deterministic card only), so a missing model only warns.
    history_model = settings.history_rag_model or settings.ollama_vision_model
    history_ready = history_model in model_names or f"{history_model}:latest" in model_names
    checks.append(
        _check(
            "history_llm",
            "借還車AI助理摘要模型",
            history_ready,
            f"{history_model} 可用" if history_ready else f"未找到 {history_model}，助理只顯示系統計算的答案卡",
            warning=True,
            detail=history_model,
        )
    )

    # Missing OCR degrades rather than blocks: plate jobs fall back to manual entry.
    plate_model = plate_model_info()
    checks.append(
        _check(
            "plate_ocr",
            "車牌辨識模型",
            plate_model["available"],
            f"{plate_model['name']} 已就緒" if plate_model["available"] else plate_model["message"],
            warning=True,
            detail=f"{plate_model['version']}; python={plate_model['python']}",
        )
    )

    if settings.notification_email_enabled:
        outbox_path = settings.power_automate_outbox_path
        power_automate_ready = bool(outbox_path and outbox_path.is_dir())
        checks.append(
            _check(
                "power_automate_outbox",
                "Power Automate 通知資料夾",
                power_automate_ready,
                "OneDrive 待寄資料夾已設定"
                if power_automate_ready
                else "請設定並建立 OneDrive 的待寄資料夾",
                required=False,
                detail=str(outbox_path) if outbox_path else None,
            )
        )

    required_checks = [item for item in checks if item.required]
    if any(item.status == ReadinessLevel.ERROR for item in required_checks):
        overall = "blocked"
    elif any(item.status == ReadinessLevel.WARNING for item in required_checks):
        overall = "degraded"
    else:
        overall = "ready"
    return ReadinessReport(
        overall=overall,
        checked_at=datetime.now(timezone.utc),
        checks=checks,
        ready_count=sum(item.status == ReadinessLevel.READY for item in checks),
        total_count=len(checks),
    )
