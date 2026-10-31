"""Automatic trial models for single-image datasets.

These datasets have no fixed human-truth test set to gate a production swap, so
their automatic runs stop at a trial checkpoint that model management lists for
an administrator; nothing already deployed changes.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from app.schemas.admin import ManagedModelKind
from app.schemas.operations import LearningTask, TrainingRunStatus
from app.services.operations_store import OperationsStore
from app.services.runtime_settings import PROJECT_ROOT

MIN_PER_LABEL = 2
MINIMUM_MANAGEMENT_SCORE = 0.60
MANAGED_KINDS = {
    LearningTask.VEHICLE_COVERAGE: ManagedModelKind.VEHICLE_COVERAGE,
    LearningTask.INTERIOR_CAPTURE: ManagedModelKind.INTERIOR_CAPTURE,
}
MODEL_ROOT = PROJECT_ROOT / "models"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label_counts(store: OperationsStore, task: LearningTask) -> Counter[str]:
    return Counter(
        str(sample.corrected_label) for sample, _, _ in store.task_training_samples(task)
    )


def candidate_blocker(store: OperationsStore, task: LearningTask) -> str | None:
    if task not in MANAGED_KINDS:
        return "此通用或單張模型已移除，請使用專用模型訓練流程"
    counts = _label_counts(store, task)
    usable = [label for label, count in counts.items() if count >= MIN_PER_LABEL]
    if len(usable) >= 2:
        return None
    summary = "、".join(f"{label} {count} 筆" for label, count in counts.items()) or "0 筆"
    return (
        f"至少需要兩種人工真值、每種至少 {MIN_PER_LABEL} 筆才能訓練分類器"
        f"（目前：{summary}）"
    )


def _macro_f1(predictions: list[int], truths: list[int], classes: int) -> float:
    scores = []
    for index in range(classes):
        tp = sum(p == index and t == index for p, t in zip(predictions, truths, strict=False))
        fp = sum(p == index and t != index for p, t in zip(predictions, truths, strict=False))
        fn = sum(p != index and t == index for p, t in zip(predictions, truths, strict=False))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return sum(scores) / max(1, len(scores))


def _fit_head(features, labels, classes: int, feature_dim: int, seed: int):
    import torch

    from app.services.interior_capture import SpecialistProbe

    torch.manual_seed(seed)
    head = SpecialistProbe(feature_dim, classes, "dinov2_vits14_linear_probe")
    counts = torch.bincount(labels, minlength=classes).float().clamp(min=1)
    criterion = torch.nn.CrossEntropyLoss(weight=counts.sum() / (classes * counts))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
    head.train()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(features), labels)
        loss.backward()
        optimizer.step()
    return head.eval()


def run_candidate_training(store: OperationsStore, run_id: str, task: LearningTask) -> None:
    import torch

    from app.services.view_classifier import _letterbox_transform, _load_dinov2_backbone

    store.update_training_run(
        run_id,
        status=TrainingRunStatus.RUNNING,
        started_at=_now(),
        message="正在抽取人工真值照片特徵",
    )
    samples = store.task_training_samples(task)
    counts = Counter(str(sample.corrected_label) for sample, _, _ in samples)
    classes = sorted(label for label, count in counts.items() if count >= MIN_PER_LABEL)
    if len(classes) < 2:
        raise ValueError(candidate_blocker(store, task) or "人工真值類別不足")
    used = [(sample, path) for sample, path, _ in samples if sample.corrected_label in classes]
    class_to_index = {label: index for index, label in enumerate(classes)}

    backbone = _load_dinov2_backbone().eval()
    transform = _letterbox_transform()
    features: list[torch.Tensor] = []
    flipped: list[torch.Tensor] = []
    with torch.inference_mode():
        for _sample, path in used:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            batch = torch.stack([transform(image), transform(ImageOps.mirror(image))])
            output = torch.nn.functional.normalize(backbone(batch), dim=1)
            features.append(output[0])
            flipped.append(output[1])
    x = torch.stack(features)
    x_flip = torch.stack(flipped)
    y = torch.tensor([class_to_index[str(sample.corrected_label)] for sample, _ in used])
    feature_dim = int(x.shape[1])

    # Stratified cross-validation on the human truths is the only honest score a
    # dataset this small has; the flipped copy only ever joins its own fold.
    folds = max(2, min(5, min(counts[label] for label in classes)))
    generator = random.Random(20260927)
    fold_of = [0] * len(used)
    for label in classes:
        members = [index for index, (sample, _) in enumerate(used) if sample.corrected_label == label]
        generator.shuffle(members)
        for position, index in enumerate(members):
            fold_of[index] = position % folds
    predictions = [0] * len(used)
    for fold in range(folds):
        train_index = [i for i in range(len(used)) if fold_of[i] != fold]
        test_index = [i for i in range(len(used)) if fold_of[i] == fold]
        head = _fit_head(
            torch.cat([x[train_index], x_flip[train_index]]),
            torch.cat([y[train_index], y[train_index]]),
            len(classes),
            feature_dim,
            seed=fold,
        )
        with torch.no_grad():
            for index, predicted in zip(test_index, head(x[test_index]).argmax(1).tolist(), strict=False):
                predictions[index] = predicted
    truths = y.tolist()
    macro_f1 = _macro_f1(predictions, truths, len(classes))
    accuracy = sum(p == t for p, t in zip(predictions, truths, strict=False)) / len(truths)

    store.update_training_run(run_id, message="正在以全部人工真值訓練試行模型")
    head = _fit_head(torch.cat([x, x_flip]), torch.cat([y, y]), len(classes), feature_dim, seed=0)
    model_directory = MODEL_ROOT / "auto" / task.value
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / f"auto-{task.value}-{run_id.lower()}.pt"
    checkpoint = {
        "schema_version": 1,
        "task": task.value,
        "managed_kind": MANAGED_KINDS[task].value,
        "architecture": "dinov2_vits14_linear_probe",
        "backbone": "dinov2_vits14",
        "feature_dim": feature_dim,
        "classes": classes,
        "preprocess": "aspect_preserving_letterbox",
        "input_size": 224,
        "head_state_dict": {key: value.detach().clone() for key, value in head.state_dict().items()},
        "temperature": 1.0,
        "model_version": f"auto-{task.value}-{run_id.lower()}",
        "management_stage": "trial",
        "management_metric": "human_truth_cv_macro_f1",
        "management_score": round(macro_f1, 4),
        "minimum_management_score": MINIMUM_MANAGEMENT_SCORE,
        "meets_minimum_standard": macro_f1 >= MINIMUM_MANAGEMENT_SCORE,
        "production_approved": False,
        "label_source": "營運人員確認的主動學習人工真值",
        "class_counts": {label: counts[label] for label in classes},
        "report": {"test": {"macro_f1": macro_f1, "accuracy": accuracy, "folds": folds}},
        "sample_ids": [sample.id for sample, _ in used],
        "created_at": _now(),
    }
    temporary = model_path.with_suffix(".next.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(model_path)
    store.mark_learning_samples_trained([sample.id for sample, _ in used])
    metrics = {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "samples": len(used),
        "folds": folds,
    }
    store.update_training_run(
        run_id,
        status=TrainingRunStatus.CANDIDATE,
        candidate_path=str(model_path),
        candidate_metrics=metrics,
        completed_at=_now(),
        message=(
            f"已用 {len(used)} 筆人工真值產生試行模型 {Path(model_path).name}；"
            f"交叉驗證 Macro F1 {macro_f1:.0%}。線上判斷未變更，請在模型管理評估後再部署。"
        ),
    )
