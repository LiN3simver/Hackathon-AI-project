from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch

from app.schemas.admin import (
    ManagedModelKind,
    ManagedModelMetric,
    ManagedModelSource,
    ManagedModelStage,
    ManagedViewModel,
    ModelManagementResponse,
    ModelQueueSettings,
)
from app.services import runtime_settings
from app.services.runtime_settings import (
    DEFAULT_VIEW_MODEL_ID,
    PROJECT_ROOT,
    RuntimeSettingsStore,
)

SUPPORTED_EXTERIOR_ARCHITECTURES = {
    "dinov2_vits14_knn_ensemble",
    "dinov2_vits14_knn_ensemble_night_adapter",
}
SUPPORTED_SPECIALIST_ARCHITECTURES = {
    "dinov2_vits14_linear_probe",
    "dinov2_vits14_mlp_probe",
}
DAMAGE_PAIR_ARCHITECTURE = "dinov2_patch_damage_probe"
SUPPORTED_MANUAL_ARCHITECTURES = {
    "torchvision_resnet18_transfer",
    "torchvision_mobilenet_v3_small_transfer",
    "torchvision_efficientnet_b0_transfer",
    "torchvision_convnext_tiny_transfer",
}
MODEL_SPECS = {
    ManagedModelKind.EXTERIOR_VIEW: {
        "classes": {"left_front", "right_front", "left_rear", "right_rear"},
        "architectures": SUPPORTED_EXTERIOR_ARCHITECTURES | SUPPORTED_MANUAL_ARCHITECTURES,
        "default_id": "exterior_view_classifier.pt",
    },
    ManagedModelKind.INTERIOR_FRONT: {
        "classes": {"interior_front", "not_interior_front"},
        "architectures": SUPPORTED_SPECIALIST_ARCHITECTURES | SUPPORTED_MANUAL_ARCHITECTURES,
        "default_id": "interior_front_classifier.pt",
    },
    ManagedModelKind.INTERIOR_REAR: {
        "classes": {"interior_rear", "not_interior_rear"},
        "architectures": SUPPORTED_SPECIALIST_ARCHITECTURES | SUPPORTED_MANUAL_ARCHITECTURES,
        "default_id": "interior_rear_classifier.pt",
    },
    ManagedModelKind.INTERIOR_FRONT_CLEANLINESS: {
        "classes": {"not_dirty", "dirty"},
        "architectures": {"clip_color_ridge_review"},
        "default_id": "interior_front_cleanliness_classifier.pt",
    },
    ManagedModelKind.INTERIOR_REAR_CLEANLINESS: {
        "classes": {"not_dirty", "dirty"},
        "architectures": {"dino_box_ridge_0.001_top5"},
        "default_id": "interior_rear_cleanliness_classifier.pt",
    },
    ManagedModelKind.DAMAGE_PAIR: {
        "classes": {"no_new_damage", "confirmed_new_damage"},
        "architectures": {DAMAGE_PAIR_ARCHITECTURE, "dinov2_vits14_patch_damage_probe"},
        "default_id": "damage_pair_specialist.pt",
    },
}
# Generic cleanliness, visible-damage, vehicle-coverage and interior-capture checkpoints are no longer
# managed; anything outside MODEL_SPECS is left off the registry.
MODEL_KIND_ORDER = {
    ManagedModelKind.EXTERIOR_VIEW: 0,
    ManagedModelKind.INTERIOR_FRONT: 1,
    ManagedModelKind.INTERIOR_REAR: 2,
    ManagedModelKind.INTERIOR_FRONT_CLEANLINESS: 3,
    ManagedModelKind.INTERIOR_REAR_CLEANLINESS: 4,
    ManagedModelKind.DAMAGE_PAIR: 5,
}


BACKBONE_LABELS = {
    "dinov2_vits14": "DINOv2 ViT-S",
    "dinov2_vitb14": "DINOv2 ViT-B",
    "dinov2_vitl14": "DINOv2 ViT-L",
    "dinov2_vitg14": "DINOv2 ViT-g",
    "timm:vit_base_patch16_siglip_512.v2_webli": "SigLIP2 ViT-B",
    "timm:convnext_base.fb_in22k": "ConvNeXt-B",
    "hf-vit:beingamit99/car_damage_detection": "CarDD 車損 ViT-B",
}


def _feature_label(mode: str) -> str:
    if mode == "plain":
        return "只看還車照"
    if mode == "contrast":
        return "整張取車照比對"
    if mode.startswith("local"):
        return f"取車照同位置 ±{mode[5:]} 格比對"
    return mode


def _backbone_summary(checkpoint: dict) -> str | None:
    members = checkpoint.get("members")
    if not isinstance(members, list) or not members:
        # Single-backbone checkpoints written before ensembles existed.
        name = str(checkpoint.get("backbone") or "dinov2_vits14")
        return f"{BACKBONE_LABELS.get(name, name)}（{_feature_label('plain')}）"
    parts = []
    for member in members:
        name = str(member.get("backbone", ""))
        label = BACKBONE_LABELS.get(name, name)
        parts.append(f"{label}（{_feature_label(str(member.get('feature_mode', 'plain')))}）")
    return " ＋ ".join(parts)


def _damage_pair_metrics(checkpoint: dict) -> list[ManagedModelMetric]:
    """Every score the training report recorded, judged against the operator targets.

    Autolearn checkpoints carry recall/accuracy over all held-out pairs; API-trained
    ones carry cross-validated recall and the fixed-test false-positive rate."""
    from app.services import damage_pair_specialist as dps

    try:
        report = json.loads(str(checkpoint.get("report_json") or "{}"))
    except json.JSONDecodeError:
        report = {}
    metrics: list[ManagedModelMetric] = []

    def add(key: str, label: str, value, unit: str, target: float | None = None, higher: bool = True) -> None:
        if value is None:
            return
        value = float(value)
        passed = None if target is None else (value >= target if higher else value <= target)
        metrics.append(ManagedModelMetric(
            key=key, label=label, value=value, unit=unit, target=target, higher_is_better=higher, passed=passed,
        ))

    if "recall" in report:
        add("recall", "召回率（交叉驗證＋固定測試集）", report["recall"], "percent", dps.MIN_RECALL)
        add("accuracy", "準確度", report.get("accuracy"), "percent", dps.TARGET_ACCURACY)
    else:
        add("recall", "交叉驗證召回率", report.get("cv_recall"), "percent", dps.MIN_RECALL)
    review = report.get("human_review_rate", report.get("final_test_fpr"))
    add("human_review_rate", "正常還車送人工比例", review, "percent", dps.MAX_TEST_FPR, higher=False)
    add("holdout_recall", "固定測試集（_TEST）召回", report.get("holdout_recall"), "percent")
    add("cv_localization", "車損位置命中", report.get("cv_localization"), "percent", dps.MIN_LOCALIZATION)
    add("cv_auroc", "AUROC（分辨力）", report.get("cv_auroc"), "score", dps.MIN_AUROC)
    fixed = report.get("at_10pct_calibration")
    if isinstance(fixed, dict):
        add("recall_at_10pct", "10% 校準門檻下召回", fixed.get("recall"), "percent")
        add("review_at_10pct", "10% 校準門檻下送人工", fixed.get("human_review_rate"), "percent", higher=False)
    benchmark = checkpoint.get("benchmark")
    if isinstance(benchmark, dict) and benchmark.get("average_ms") is not None:
        seconds = float(benchmark["average_ms"]) * dps.PAIRS_PER_INSPECTION / 1000
        add("inspection_seconds", "一次還車檢查（4 角度，本機）", seconds, "seconds",
            dps.MAX_INSPECTION_SECONDS, higher=False)
    return metrics


class ManagedModelNotFoundError(ValueError):
    pass


class ManagedModelDeleteConflict(ValueError):
    pass


class ModelManagementService:
    def __init__(
        self,
        model_root: Path | None = None,
        runtime_store: RuntimeSettingsStore | None = None,
    ) -> None:
        self.model_root = (model_root or PROJECT_ROOT / "models").resolve()
        self.runtime_store = runtime_store

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()

    def _inspect_model(self, path: Path) -> ManagedViewModel | None:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            architecture = str(checkpoint["architecture"])
            classes = {str(value) for value in checkpoint["classes"]}
            version = str(checkpoint["model_version"])
            kind = next(
                (
                    candidate
                    for candidate, spec in MODEL_SPECS.items()
                    if architecture in spec["architectures"]
                    and classes == spec["classes"]
                ),
                None,
            )
            declared_kind = checkpoint.get("managed_kind")
            if declared_kind is not None:
                declared = ManagedModelKind(str(declared_kind))
                spec = MODEL_SPECS.get(declared)
                kind = declared if spec and architecture in spec["architectures"] else None
            if kind is None:
                return None
            relative_id = path.relative_to(self.model_root).as_posix()
            default_id = str(MODEL_SPECS.get(kind, {}).get("default_id", ""))
            is_current = relative_id == default_id
            if version.startswith(("exterior-view-active-", "auto-", "autolearn-")):
                source = ManagedModelSource.AUTO_TRAINED
            elif version.startswith("manual-"):
                source = ManagedModelSource.MANUAL_TRAINED
            elif is_current:
                source = ManagedModelSource.DEPLOYED
            else:
                source = ManagedModelSource.HISTORY
            default_stage = (
                ManagedModelStage.DEPLOYED
                if kind == ManagedModelKind.EXTERIOR_VIEW and is_current
                else ManagedModelStage.FIRST_VERSION
                if is_current
                else ManagedModelStage.HISTORY
            )
            try:
                stage = ManagedModelStage(
                    str(checkpoint.get("management_stage", default_stage.value))
                )
            except ValueError:
                stage = default_stage
            report = checkpoint.get("report", {})
            test_metrics = report.get("test", {}) if isinstance(report, dict) else {}
            quality_score = checkpoint.get(
                "management_score",
                test_metrics.get("macro_f1") if isinstance(test_metrics, dict) else None,
            )
            stat = path.stat()
            evaluation = []
            backbone_summary = None
            if kind == ManagedModelKind.DAMAGE_PAIR:
                evaluation = _damage_pair_metrics(checkpoint)
                backbone_summary = _backbone_summary(checkpoint)
            elif kind == ManagedModelKind.INTERIOR_FRONT_CLEANLINESS:
                evaluation = [ManagedModelMetric.model_validate(metric)
                              for metric in checkpoint.get("review_metrics", [])]
                backbone_summary = "CLIP ViT-B/32 ＋局部 Lab 色調 · CPU"
            elif kind == ManagedModelKind.INTERIOR_REAR_CLEANLINESS:
                evaluation = [ManagedModelMetric.model_validate(metric)
                              for metric in checkpoint.get("review_metrics", [])]
                backbone_summary = "DINOv2 ViT-S/14 · 448px 局部 patch ＋框選 ridge · CPU"
            return ManagedViewModel(
                id=relative_id,
                version=version,
                architecture=architecture,
                kind=kind,
                stage=stage,
                source=source,
                is_current=is_current,
                quality_score=(float(quality_score) if quality_score is not None else None),
                minimum_quality_score=(
                    float(checkpoint["minimum_management_score"])
                    if checkpoint.get("minimum_management_score") is not None
                    else None
                ),
                quality_metric=(
                    str(checkpoint["management_metric"])
                    if checkpoint.get("management_metric")
                    else None
                ),
                label_source=(
                    str(checkpoint["label_source"])
                    if checkpoint.get("label_source")
                    else None
                ),
                production_approved=bool(checkpoint.get("production_approved", False)),
                average_latency_ms=(
                    float(checkpoint["benchmark"]["average_ms"])
                    if isinstance(checkpoint.get("benchmark"), dict)
                    and checkpoint["benchmark"].get("average_ms") is not None
                    else None
                ),
                latency_breakdown_ms=(
                    {
                        "vehicle_detection": float(checkpoint["benchmark"].get("vehicle_detection_ms", 0.0)),
                        **{
                            str(name): float(value)
                            for name, value in (checkpoint["benchmark"].get("members_ms") or {}).items()
                        },
                    }
                    if isinstance(checkpoint.get("benchmark"), dict)
                    else None
                ),
                evaluation=evaluation,
                backbone_summary=backbone_summary,
                recommendation_rank=(
                    int(checkpoint["management_rank"]) if checkpoint.get("management_rank") is not None else None
                ),
                recommendation_note=(
                    str(checkpoint["management_note"]) if checkpoint.get("management_note") else None
                ),
                file_size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                sha256=self._sha256(path),
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def list_models(self) -> list[ManagedViewModel]:
        if not self.model_root.is_dir():
            return []
        models = [
            model
            for path in self.model_root.rglob("*.pt")
            if not path.name.endswith(".next.pt")
            if (model := self._inspect_model(path)) is not None
        ]
        # A damage-pair candidate is "in use" when online inference loads an identical copy.
        deployed = next(
            (m.sha256 for m in models if m.kind == ManagedModelKind.DAMAGE_PAIR and m.is_current), None
        )
        for model in models:
            if model.kind in {ManagedModelKind.INTERIOR_FRONT_CLEANLINESS, ManagedModelKind.INTERIOR_REAR_CLEANLINESS}:
                model.in_use = model.is_current and model.production_approved
            if model.kind == ManagedModelKind.DAMAGE_PAIR:
                model.in_use = model.sha256 == deployed
                model.can_activate = not model.in_use
        # The deployed file is only listed on its own when no candidate is an identical copy.
        if any(m.kind == ManagedModelKind.DAMAGE_PAIR and m.in_use and not m.is_current for m in models):
            models = [m for m in models if not (m.kind == ManagedModelKind.DAMAGE_PAIR and m.is_current)]

        def order(model: ManagedViewModel) -> tuple:
            if model.kind == ManagedModelKind.DAMAGE_PAIR:
                # Recommended ranking first, then what is online, then newest.
                rank = model.recommendation_rank if model.recommendation_rank is not None else 10_000
                return (MODEL_KIND_ORDER[model.kind], rank, not model.in_use, -model.modified_at.timestamp())
            return (MODEL_KIND_ORDER[model.kind], 0, not model.is_current, -model.modified_at.timestamp())

        return sorted(models, key=order)

    def resolve_model(self, model_id: str) -> Path | None:
        candidate = (self.model_root / Path(model_id)).resolve()
        if candidate != self.model_root and self.model_root not in candidate.parents:
            return None
        if not candidate.is_file() or candidate.suffix.lower() != ".pt":
            return None
        return candidate

    def selected_paths(self) -> list[Path]:
        store = self.runtime_store or runtime_settings.get_runtime_settings_store()
        queue, _, _ = store.get_model_queue()
        exterior_ids = {
            model.id
            for model in self.list_models()
            if model.kind == ManagedModelKind.EXTERIOR_VIEW
        }
        paths = [
            path
            for model_id in queue.model_ids
            if model_id in exterior_ids
            if (path := self.resolve_model(model_id)) is not None
        ]
        if not paths:
            default_path = self.resolve_model(DEFAULT_VIEW_MODEL_ID)
            if default_path is not None:
                paths.append(default_path)
        return paths[:3]

    def get_management(self) -> ModelManagementResponse:
        store = self.runtime_store or runtime_settings.get_runtime_settings_store()
        queue, updated_at, updated_by = store.get_model_queue()
        models = self.list_models()
        available_ids = {
            model.id
            for model in models
            if model.kind == ManagedModelKind.EXTERIOR_VIEW
        }
        selected = [model_id for model_id in queue.model_ids if model_id in available_ids]
        if not selected and DEFAULT_VIEW_MODEL_ID in available_ids:
            selected = [DEFAULT_VIEW_MODEL_ID]
        return ModelManagementResponse(
            models=models,
            queue=ModelQueueSettings(model_ids=selected or queue.model_ids),
            updated_at=updated_at,
            updated_by=updated_by,
        )

    def update_queue(self, queue: ModelQueueSettings, *, actor: str) -> ModelManagementResponse:
        available_ids = {
            model.id
            for model in self.list_models()
            if model.kind == ManagedModelKind.EXTERIOR_VIEW
        }
        missing = [model_id for model_id in queue.model_ids if model_id not in available_ids]
        if missing:
            raise ValueError(f"找不到或不支援模型：{', '.join(missing)}")
        store = self.runtime_store or runtime_settings.get_runtime_settings_store()
        store.update_model_queue(queue, actor=actor)
        return self.get_management()

    def activate_damage_pair_model(self, model_id: str) -> tuple[ModelManagementResponse, str | None]:
        """Make online damage-pair inference use ``model_id``.

        The file is copied over the deployed ``damage_pair_specialist.pt`` (the path
        the specialist and automatic training compare against); the previous file
        goes to ``history/`` first. The specialist reloads on the next comparison."""
        model = next((item for item in self.list_models() if item.id == model_id), None)
        if model is None or model.kind != ManagedModelKind.DAMAGE_PAIR:
            raise ManagedModelNotFoundError("找不到可啟用的前後車損比對模型")
        if model.in_use:
            raise ManagedModelDeleteConflict("這個模型已經是使用中的版本")
        source = self.resolve_model(model_id)
        if source is None:
            raise ManagedModelNotFoundError("模型檔案已不存在")
        target = self.model_root / str(MODEL_SPECS[ManagedModelKind.DAMAGE_PAIR]["default_id"])
        backup_id = None
        if target.is_file():
            history = self.model_root / "history"
            history.mkdir(parents=True, exist_ok=True)
            backup = history / f"damage_pair_before_activate_{datetime.now():%Y%m%d%H%M%S}.pt"
            shutil.copy2(target, backup)
            backup_id = backup.relative_to(self.model_root).as_posix()
        temporary = target.with_suffix(".next.pt")
        # copyfile, not copy2: a fresh mtime is what makes the specialist reload.
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        return self.get_management(), backup_id

    def delete_model(self, model_id: str) -> ModelManagementResponse:
        model = next((item for item in self.list_models() if item.id == model_id), None)
        if model is None:
            raise ManagedModelNotFoundError("找不到可刪除的模型")
        if model.is_current or model.in_use:
            raise ManagedModelDeleteConflict("現行正式模型不可刪除")
        store = self.runtime_store or runtime_settings.get_runtime_settings_store()
        queue, _, _ = store.get_model_queue()
        if model_id in queue.model_ids:
            raise ManagedModelDeleteConflict("模型仍在候選隊列中，請先取消勾選並儲存順位")
        path = self.resolve_model(model_id)
        if path is None:
            raise ManagedModelNotFoundError("模型檔案已不存在")
        path.unlink()
        return self.get_management()


@lru_cache
def get_model_management_service() -> ModelManagementService:
    return ModelManagementService()
