from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.schemas.admin import RecognitionSettings
from app.schemas.operations import (
    DatasetTrainingStatus,
    LearningStatus,
    LearningTask,
    TrainingRun,
    TrainingRunner,
    TrainingRunStatus,
)
from app.services.dataset_paths import resolve_dataset_path
from app.services.operations_store import DAMAGE_PAIR_TRAINING_LABELS, OperationsStore
from app.services.runtime_settings import get_runtime_recognition_settings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_MANIFEST = PROJECT_ROOT / "data" / "view_dataset" / "local_damage_manifest.csv"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_dinov2_view_classifier.py"
DEPLOYED_MODEL = PROJECT_ROOT / "models" / "exterior_view_classifier.pt"
EXTERIOR_LABELS = ("left_front", "right_front", "left_rear", "right_rear")

# Each dataset trains on its own once its untrained human truths reach the
# threshold stored under this settings field.
TASK_THRESHOLD_FIELDS: dict[LearningTask, str] = {
    LearningTask.EXTERIOR_VIEW: "active_learning_min_approved",
    LearningTask.VEHICLE_COVERAGE: "auto_train_min_vehicle_coverage",
    LearningTask.INTERIOR_CAPTURE: "auto_train_min_interior_capture",
    LearningTask.DAMAGE_PAIR: "auto_train_min_damage_pair",
}
TASK_RUNNERS: dict[LearningTask, TrainingRunner] = {
    LearningTask.EXTERIOR_VIEW: TrainingRunner.DEPLOY,
    LearningTask.DAMAGE_PAIR: TrainingRunner.DEPLOY,
    LearningTask.VEHICLE_COVERAGE: TrainingRunner.CANDIDATE,
    LearningTask.INTERIOR_CAPTURE: TrainingRunner.CANDIDATE,
    LearningTask.INTERIOR_CLEANLINESS: TrainingRunner.NONE,
    LearningTask.VISIBLE_DAMAGE: TrainingRunner.NONE,
    # RapidOCR is pretrained; plate confirmations are kept as records only.
    LearningTask.LICENSE_PLATE: TrainingRunner.NONE,
    # Trained offline with scripts/train_blur_detector.py, which gates deployment.
    LearningTask.CAPTURE_SHARPNESS: TrainingRunner.NONE,
}
TASK_NAMES: dict[LearningTask, str] = {
    LearningTask.EXTERIOR_VIEW: "四角視角",
    LearningTask.VEHICLE_COVERAGE: "車身構圖",
    LearningTask.INTERIOR_CAPTURE: "車內拍攝有效性",
    LearningTask.INTERIOR_CLEANLINESS: "車內清潔",
    LearningTask.VISIBLE_DAMAGE: "單張車損特寫",
    LearningTask.DAMAGE_PAIR: "前後車損比對",
    LearningTask.LICENSE_PLATE: "車牌辨識",
    LearningTask.CAPTURE_SHARPNESS: "車外照片清晰度",
}

# One worker at a time drains the queue; runs for other datasets wait their turn.
_WORKER_LOCK = Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def training_threshold(settings: RecognitionSettings, task: LearningTask) -> int:
    return int(getattr(settings, TASK_THRESHOLD_FIELDS[task]))


def trainable_labels(task: LearningTask) -> tuple[str, ...] | None:
    if task == LearningTask.EXTERIOR_VIEW:
        return EXTERIOR_LABELS
    if task == LearningTask.DAMAGE_PAIR:
        return DAMAGE_PAIR_TRAINING_LABELS
    return None


def count_untrained(store: OperationsStore, task: LearningTask) -> int:
    """Approved human truths of one dataset that no training run has used yet."""
    if task == LearningTask.EXTERIOR_VIEW:
        return store.count_approved_exterior_trainable()
    if TASK_RUNNERS[task] == TrainingRunner.NONE:
        return 0
    return sum(
        sample.status == LearningStatus.APPROVED
        for sample, _primary, _secondary in store.task_training_samples(
            task, labels=trainable_labels(task)
        )
    )


def _dataset_blocker(store: OperationsStore, task: LearningTask) -> str | None:
    """Why this dataset cannot train yet even with enough samples, if anything."""
    if task == LearningTask.EXTERIOR_VIEW:
        if _curated_exterior_is_active():
            return (
                "目前四角正式模型使用人工挑選樣本；舊自動訓練流程仍混用未經逐張確認的"
                "檔名標籤資料，已暫停自動覆蓋，待改用人工真值資料集。"
            )
        return None
    if task == LearningTask.DAMAGE_PAIR:
        from app.services.damage_pair_specialist import training_blocker

        return training_blocker(store)
    from app.services.specialist_training import candidate_blocker

    return candidate_blocker(store, task)


def _curated_exterior_is_active() -> bool:
    if not DEPLOYED_MODEL.is_file():
        return False
    import torch

    checkpoint = torch.load(DEPLOYED_MODEL, map_location="cpu", weights_only=True)
    return checkpoint.get("label_source") == "human_selected_standard_angle_folders"


def request_training_run(
    store: OperationsStore,
    *,
    trigger: str,
    force: bool = False,
    task: LearningTask = LearningTask.EXTERIOR_VIEW,
) -> TrainingRun | None:
    """Queue one dataset's training run, or record why it was skipped.

    Returns None while that dataset already has a queued or running job.
    """
    if store.has_active_training_run(task=task):
        return None
    name = TASK_NAMES[task]
    if TASK_RUNNERS[task] == TrainingRunner.NONE:
        return store.create_training_run(
            trigger=trigger,
            approved_samples=0,
            status=TrainingRunStatus.SKIPPED,
            task=task,
            message=(f"{name}模型已移除，僅保留人工真值，不再訓練通用或單張模型"
                     if task in {LearningTask.INTERIOR_CLEANLINESS, LearningTask.VISIBLE_DAMAGE}
                     else f"{name}使用預訓練模型，人工確認只保留紀錄，不在本系統訓練"),
        )
    settings = get_runtime_recognition_settings()
    approved = count_untrained(store, task)
    threshold = training_threshold(settings, task)
    # A forced run may retrain on truths already used, e.g. after the test set or
    # the model recipe changed; the exterior runner keeps its original rule.
    available = (
        len(store.task_training_samples(task, labels=trainable_labels(task)))
        if force and task != LearningTask.EXTERIOR_VIEW
        else approved
    )
    if available == 0 or (approved < threshold and not force):
        return store.create_training_run(
            trigger=trigger,
            approved_samples=approved,
            status=TrainingRunStatus.SKIPPED,
            task=task,
            message=f"{name}待訓練人工真值 {approved} 筆，未達 {threshold} 筆門檻，略過訓練",
        )
    blocker = _dataset_blocker(store, task)
    if blocker:
        return store.create_training_run(
            trigger=trigger,
            approved_samples=approved,
            status=TrainingRunStatus.SKIPPED,
            task=task,
            message=blocker,
        )
    return store.create_training_run(
        trigger=trigger,
        approved_samples=approved,
        status=TrainingRunStatus.QUEUED,
        task=task,
        message=f"{name}：等待背景訓練",
    )


def queue_ready_training(
    store: OperationsStore, *, trigger: str, force: bool = False
) -> list[TrainingRun]:
    """Queue every dataset that is ready (or, with force, has any new truth)."""
    settings = get_runtime_recognition_settings()
    queued: list[TrainingRun] = []
    for task, runner in TASK_RUNNERS.items():
        if runner == TrainingRunner.NONE:
            continue
        untrained = count_untrained(store, task)
        if untrained == 0 or (untrained < training_threshold(settings, task) and not force):
            continue
        run = request_training_run(store, trigger=trigger, force=force, task=task)
        if run is not None and run.status == TrainingRunStatus.QUEUED:
            queued.append(run)
    return queued


def dataset_training_statuses(
    store: OperationsStore, settings: RecognitionSettings
) -> list[DatasetTrainingStatus]:
    statuses: list[DatasetTrainingStatus] = []
    for task, runner in TASK_RUNNERS.items():
        runs = store.list_training_runs(limit=1, task=task)
        if runner == TrainingRunner.NONE:
            statuses.append(
                DatasetTrainingStatus(
                    task=task,
                    runner=runner,
                    approved_untrained=0,
                    trained=0,
                    threshold=0,
                    remaining=0,
                    last_run=runs[0] if runs else None,
                )
            )
            continue
        samples = store.task_training_samples(task, labels=trainable_labels(task))
        untrained = count_untrained(store, task)
        threshold = training_threshold(settings, task)
        statuses.append(
            DatasetTrainingStatus(
                task=task,
                runner=runner,
                approved_untrained=untrained,
                trained=sum(sample.status == LearningStatus.TRAINED for sample, *_ in samples),
                threshold=threshold,
                remaining=max(0, threshold - untrained),
                last_run=runs[0] if runs else None,
            )
        )
    return statuses


def _build_manifest(
    store: OperationsStore, run_directory: Path
) -> tuple[Path, list[str], list[str]]:
    run_directory.mkdir(parents=True, exist_ok=True)
    with BASE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"path", "local_path", "car_no", "label", "stage", "split"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Base manifest is missing columns: {sorted(required - set(fieldnames))}")
    for row in rows:
        # The committed manifest keeps the machine it was written on; the trainer
        # silently drops rows whose file is missing, which would empty the test set.
        row["local_path"] = str(resolve_dataset_path(row["local_path"]))
    missing = sum(not Path(row["local_path"]).is_file() for row in rows)
    if missing:
        raise FileNotFoundError(
            f"固定訓練/測試集有 {missing} 張照片找不到，請設定 GUARDIAN_COMPETITION_DATA_ROOT"
        )
    protected_cars = {
        row["car_no"] for row in rows if row["split"] in {"val", "test"}
    }
    used_sample_ids: list[str] = []
    skipped_sample_ids: list[str] = []
    for sample, image_path in store.exterior_training_samples():
        group_id = sample.vehicle_id or sample.rental_id or sample.id
        if group_id in protected_cars:
            skipped_sample_ids.append(sample.id)
            continue
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "path": str(image_path),
                "local_path": str(image_path),
                "car_no": group_id,
                "label": sample.corrected_label or "",
                "stage": "active_learning",
                "split": "train",
            }
        )
        if "claim_label" in row:
            row["claim_label"] = "human_verified"
        if "pair_key" in row:
            row["pair_key"] = sample.id
        rows.append(row)
        used_sample_ids.append(sample.id)
    if not used_sample_ids:
        raise ValueError("No leakage-safe approved exterior samples are available")
    manifest_path = run_directory / "training_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path, used_sample_ids, skipped_sample_ids


def _checkpoint_metrics(path: Path) -> dict[str, float]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    test = checkpoint.get("report", {}).get("test", {})
    policy = test.get("confidence_policy", {})
    return {
        "accuracy": float(test.get("accuracy", 0.0)),
        "macro_f1": float(test.get("macro_f1", 0.0)),
        "accepted_accuracy": float(policy.get("accepted_accuracy", 0.0)),
        "coverage": float(policy.get("coverage", 0.0)),
    }


def _report_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    test = report["test"]
    policy = test.get("confidence_policy", {})
    return {
        "accuracy": float(test["accuracy"]),
        "macro_f1": float(test["macro_f1"]),
        "accepted_accuracy": float(policy.get("accepted_accuracy", 0.0)),
        "coverage": float(policy.get("coverage", 0.0)),
    }


def _passes_promotion_gate(
    baseline: dict[str, float], candidate: dict[str, float]
) -> tuple[bool, str]:
    settings = get_runtime_recognition_settings()
    if candidate["macro_f1"] < settings.active_learning_min_test_macro_f1:
        return False, "候選模型固定測試集 macro F1 未達最低門檻"
    allowed_drop = settings.active_learning_max_metric_drop
    if candidate["macro_f1"] + allowed_drop < baseline["macro_f1"]:
        return False, "候選模型固定測試集 macro F1 退步，禁止發布"
    if candidate["accuracy"] + allowed_drop < baseline["accuracy"]:
        return False, "候選模型固定測試集 accuracy 退步，禁止發布"
    if candidate["accepted_accuracy"] < settings.active_learning_min_accepted_accuracy:
        return False, (
            "候選模型高信心樣本正確率未達"
            f" {settings.active_learning_min_accepted_accuracy:.0%}，禁止發布"
        )
    if (
        candidate["coverage"] + settings.active_learning_max_coverage_drop
        < baseline["coverage"]
    ):
        return False, "候選模型高信心覆蓋率下降過多，禁止發布"
    return True, "候選模型通過固定測試集防退步門檻"


def _run_exterior_training(store: OperationsStore, run_id: str) -> None:
    blocker = _dataset_blocker(store, LearningTask.EXTERIOR_VIEW)
    if blocker:
        raise RuntimeError(blocker)
    run_root = PROJECT_ROOT / "data" / "active_learning" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    candidate_path = run_root / "exterior_view_candidate.pt"
    report_path = run_root / "training_report.json"
    log_path = run_root / "training.log"
    store.update_training_run(
        run_id,
        status=TrainingRunStatus.RUNNING,
        started_at=utc_now(),
        candidate_path=str(candidate_path),
        report_path=str(report_path),
        log_path=str(log_path),
        message="正在建立主動學習資料集",
    )
    manifest_path, used_sample_ids, skipped_sample_ids = _build_manifest(store, run_root)
    baseline = _checkpoint_metrics(DEPLOYED_MODEL)
    store.update_training_run(
        run_id,
        manifest_path=str(manifest_path),
        baseline_metrics=baseline,
        message=f"正在以 {len(used_sample_ids)} 筆已覆核困難樣本訓練候選模型",
    )
    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--output",
        str(candidate_path),
        "--report",
        str(report_path),
        "--model-version",
        f"exterior-view-active-{run_id.lower()}",
        "--augmentations",
        "8",
        "--knn-ensemble",
        "--aspect-preserving",
        "--night-expert",
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Training process exited with code {completed.returncode}")
    candidate = _report_metrics(report_path)
    allowed, gate_message = _passes_promotion_gate(baseline, candidate)
    if not allowed:
        store.update_training_run(
            run_id,
            status=TrainingRunStatus.REJECTED,
            candidate_metrics=candidate,
            message=gate_message,
            completed_at=utc_now(),
        )
        return

    history_directory = DEPLOYED_MODEL.parent / "history"
    history_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEPLOYED_MODEL, history_directory / f"before_{run_id}.pt")
    temporary_deployment = DEPLOYED_MODEL.with_suffix(".next.pt")
    shutil.copy2(candidate_path, temporary_deployment)
    os.replace(temporary_deployment, DEPLOYED_MODEL)
    newly_trained_ids = [
        sample.id
        for sample, _ in store.exterior_training_samples()
        if sample.status == LearningStatus.APPROVED and sample.id in used_sample_ids
    ]
    store.mark_learning_samples_trained(newly_trained_ids)
    skip_note = (
        f"；另有 {len(skipped_sample_ids)} 筆因車輛與固定驗證/測試集重疊而隔離"
        if skipped_sample_ids
        else ""
    )
    store.update_training_run(
        run_id,
        status=TrainingRunStatus.PROMOTED,
        candidate_metrics=candidate,
        message=f"{gate_message}，已自動發布並保留上一版{skip_note}",
        completed_at=utc_now(),
    )


def _execute_run(store: OperationsStore, run_id: str) -> None:
    run = store.get_training_run(run_id)
    if run is None or run.status != TrainingRunStatus.QUEUED:
        return
    try:
        if run.task in {LearningTask.INTERIOR_CLEANLINESS, LearningTask.VISIBLE_DAMAGE}:
            store.update_training_run(
                run_id, status=TrainingRunStatus.SKIPPED,
                message="通用整潔與單張車損模型已移除，不再執行訓練。", completed_at=utc_now(),
            )
            return
        if run.task == LearningTask.EXTERIOR_VIEW:
            _run_exterior_training(store, run_id)
        elif run.task == LearningTask.DAMAGE_PAIR:
            from app.services.damage_pair_specialist import run_damage_pair_training

            run_damage_pair_training(store, run_id)
        else:
            from app.services.specialist_training import run_candidate_training

            run_candidate_training(store, run_id, run.task)
    except Exception as exc:  # background job must persist its failure for operators
        store.update_training_run(
            run_id,
            status=TrainingRunStatus.FAILED,
            message=str(exc),
            completed_at=utc_now(),
        )


def run_training_cycle(
    run_id: str,
    database_path: str,
    learning_storage_path: str,
) -> None:
    """Run one queued job, then any other datasets queued behind it."""
    store = OperationsStore(Path(database_path), Path(learning_storage_path))
    store.initialize()
    with _WORKER_LOCK:
        _execute_run(store, run_id)
        while (queued := store.next_queued_training_run()) is not None:
            _execute_run(store, queued.id)
