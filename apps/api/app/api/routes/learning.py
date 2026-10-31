import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.operations import (
    DeleteLearningRentalResponse,
    DeleteLearningSampleResponse,
    LearningRentalSummary,
    LearningReviewMethod,
    LearningSample,
    LearningStats,
    LearningStatus,
    LearningTask,
    LearningTrainingApprovalRequest,
    ManualTruthDestination,
    ManualTruthDuplicateCheckRequest,
    ManualTruthDuplicateCheckResponse,
    ManualTruthUploadItem,
    ManualTruthUploadResponse,
    ManualTruthUploadStatus,
    PairAngleResult,
    TrainingRunner,
)
from app.services.active_learning import ALLOWED_CORRECTED_LABELS, EXTERIOR_LABELS
from app.services.auth import require_admin, require_authenticated
from app.services.license_plate import is_valid_plate_label, normalize_plate
from app.services.operations_store import OperationsStore, get_operations_store
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.training_automation import dataset_training_statuses
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/learning", tags=["active learning"])

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")

NON_TRAINING_LABELS = {
    LearningTask.EXTERIOR_VIEW: {"unusable"},
    LearningTask.INTERIOR_CLEANLINESS: {"cannot_assess"},
    LearningTask.VISIBLE_DAMAGE: {"cannot_assess"},
    LearningTask.DAMAGE_PAIR: {"cannot_assess"},
}


def _validated_task_label(
    task: LearningTask, corrected_label: str, *, has_secondary_image: bool
) -> str:
    if task == LearningTask.LICENSE_PLATE:
        if not is_valid_plate_label(corrected_label):
            raise ValueError("車牌人工真值格式無效")
        return normalize_plate(corrected_label) or corrected_label

    allowed = ALLOWED_CORRECTED_LABELS[task]
    if corrected_label not in allowed:
        raise ValueError("人工真值不在此任務允許的分類中")
    if corrected_label in NON_TRAINING_LABELS.get(task, set()):
        raise ValueError(f"{corrected_label} 不是可訓練分類，請清除此筆資料")
    if task == LearningTask.EXTERIOR_VIEW and corrected_label not in EXTERIOR_LABELS:
        raise ValueError("四角視角訓練集只接受四個標準角度")
    if task == LearningTask.DAMAGE_PAIR and not has_secondary_image:
        raise ValueError("車損比對訓練集必須同時有取車與還車照片")
    return corrected_label


def _validated_training_label(sample: LearningSample, corrected_label: str) -> str:
    """Validate the label explicitly confirmed by the authenticated reviewer."""
    return _validated_task_label(
        sample.task,
        corrected_label,
        has_secondary_image=sample.has_secondary_image,
    )


_DIRT_ISSUE_LABELS = {
    # Earlier generic boxes stay valid so saved truths keep validating.
    "dirt_floor",
    "dirt_seat",
    "stain",
    "dust",
    "footprint",
    "mud_sand",
    "leather_seat_stain",
    "fabric_seat_stain",
    "food_crumbs",
    "liquid_spill",
    "dashboard_dust",
    "seatback_pocket",
    "trash",
    "pet_hair",
    "other_dirt",
}
_DAMAGE_ISSUE_LABELS = {
    "scratch",
    "dent",
    "crack",
    "broken_part",
    "paint_transfer",
    "deformation",
    "panel_misalignment",
    "other_damage",
}
_INTERIOR_HARD_NEGATIVE_LABELS = {
    "reflection",
    "shadow",
    "floor_pattern",
    "seat_pattern",
    "leather_texture",
    "normal_wear",
    "other_normal",
}
_DAMAGE_HARD_NEGATIVE_LABELS = {
    "normal_body_line",
    "panel_gap",
    "reflection",
    "water_droplet",
    "surface_dirt",
    "shadow",
    "other_normal",
}
_HARD_NEGATIVE_LABELS = (
    _INTERIOR_HARD_NEGATIVE_LABELS | _DAMAGE_HARD_NEGATIVE_LABELS
)
_INTERIOR_REFERENCE_LABELS = {"seat_cushion", "seat_backrest", "headrest", "floor_mat", "interior_floor", "dashboard"}
_DAMAGE_REFERENCE_LABELS = {"vehicle_body", "normal_vehicle_body", "body_panel", "bumper", "window", "wheel"}
_REFERENCE_LABELS = _INTERIOR_REFERENCE_LABELS | _DAMAGE_REFERENCE_LABELS


def _validated_review_metadata(
    sample: LearningSample,
    request: LearningTrainingApprovalRequest,
    corrected_label: str,
) -> dict[str, object]:
    annotations = []
    for item in request.annotations:
        # Old rectangles retain their wire format; new outlines carry explicit geometry.
        excluded = set()
        if item.shape == "rectangle":
            excluded.update({"shape", "points"})
        if item.kind != "reference":
            excluded.add("normal_reference")
        if item.auto_refinement is None:
            excluded.add("auto_refinement")
        annotations.append(item.model_dump(mode="json", exclude=excluded))
    problem_annotations = [a for a in annotations if a["kind"] != "reference"]
    if request.review_method == LearningReviewMethod.MANUAL_PAIR:
        if sample.task != LearningTask.DAMAGE_PAIR:
            raise ValueError("人工核對角度只適用於取還車雙圖車損")
        if not sample.has_secondary_image:
            raise ValueError("人工核對角度需要取車與還車兩張照片")
        if request.pair_angle_result is None:
            raise ValueError("請先選擇人工角度核對結果")
        if problem_annotations:
            raise ValueError("人工核對角度模式不能同時提交區域框選")
        expected_label = (
            "no_new_damage"
            if request.pair_angle_result == PairAngleResult.SAME_VIEW
            else "cannot_assess"
        )
        if corrected_label != expected_label:
            raise ValueError("人工真值與角度核對結果不一致")
    elif request.review_method == LearningReviewMethod.MANUAL_SINGLE:
        if sample.task not in {
            LearningTask.INTERIOR_CLEANLINESS,
            LearningTask.VISIBLE_DAMAGE,
        } or sample.has_secondary_image:
            raise ValueError("單圖人工覆核只適用於單張車內清潔或可見車損照片")
        if problem_annotations:
            raise ValueError("單圖人工覆核不能同時提交區域框選")
        if request.pair_angle_result is not None:
            raise ValueError("單圖人工覆核不適用角度核對結果")
    elif request.pair_angle_result is not None:
        if sample.task != LearningTask.DAMAGE_PAIR or not sample.has_secondary_image:
            raise ValueError("角度核對結果只適用於取還車雙圖車損")
        if request.pair_angle_result != PairAngleResult.SAME_VIEW:
            raise ValueError("角度不符時請使用人工核對模式記錄無法比對")
    for annotation in annotations:
        if annotation["image_slot"] == "secondary" and not sample.has_secondary_image:
            raise ValueError("單張影像樣本不能標記 secondary 影像")
        label = str(annotation["label"])
        kind = str(annotation["kind"])
        if sample.task == LearningTask.INTERIOR_CLEANLINESS:
            allowed = _DIRT_ISSUE_LABELS | _INTERIOR_HARD_NEGATIVE_LABELS | _INTERIOR_REFERENCE_LABELS
        elif sample.task in {LearningTask.VISIBLE_DAMAGE, LearningTask.DAMAGE_PAIR}:
            allowed = _DAMAGE_ISSUE_LABELS | _DAMAGE_HARD_NEGATIVE_LABELS | _DAMAGE_REFERENCE_LABELS
        else:
            allowed = set()
        if label not in allowed:
            raise ValueError("這個區域標記不適用於此訓練任務")
        expected_kind = "reference" if label in _REFERENCE_LABELS else (
            "hard_negative" if label in _HARD_NEGATIVE_LABELS else "issue"
        )
        if kind != expected_kind:
            raise ValueError("區域類型與標記類別不一致")
        negative_label = corrected_label in {"clean", "ordinary", "no_damage"}
        if corrected_label == "no_new_damage":
            negative_label = not (
                sample.task == LearningTask.DAMAGE_PAIR
                and annotation["image_slot"] == "primary"
            )
        if kind == "issue" and negative_label:
            raise ValueError("人工真值為沒有問題時，不能標記問題區域")

    vehicle_identity = (
        request.vehicle_identity.model_dump(mode="json")
        if request.vehicle_identity is not None
        else None
    )
    if vehicle_identity is not None:
        vehicle_identity = {key: value for key, value in vehicle_identity.items()
                            if key in {"make", "vehicle_type"} or value}
        if vehicle_identity.get("model") and not vehicle_identity.get("make"):
            raise ValueError("記錄實際車款時請一併提供品牌")
    if vehicle_identity is not None and not any(vehicle_identity.values()):
        vehicle_identity = None
    if vehicle_identity is not None and sample.task not in {
        LearningTask.VISIBLE_DAMAGE,
        LearningTask.DAMAGE_PAIR,
        LearningTask.INTERIOR_CLEANLINESS,
    }:
        raise ValueError("車款與座椅款式只適用於車損或車內清潔訓練任務")
    uses_outlines = any(a.get("shape") == "polygon" or a["kind"] == "reference" for a in annotations)
    uses_detailed_identity = bool(vehicle_identity and set(vehicle_identity) - {"make", "vehicle_type"})
    metadata = {
        "schema_version": 2 if uses_outlines or uses_detailed_identity else 1,
        "annotations": annotations,
        "vehicle_identity": vehicle_identity,
        "localization_eligible": bool(problem_annotations),
        "review_method": (
            request.review_method.value if request.review_method else None
        ),
        "pair_angle_result": (
            request.pair_angle_result.value if request.pair_angle_result else None
        ),
    }
    if metadata["schema_version"] == 2:
        metadata["coordinate_space"] = "normalized_image"
        metadata["outline_eligible"] = any(a["kind"] == "reference" for a in annotations)
    return metadata


@router.get("/samples", response_model=list[LearningSample])
def list_learning_samples(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
    status: Annotated[LearningStatus | None, Query()] = None,
    task: Annotated[LearningTask | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    pending_inbox_only: Annotated[bool, Query()] = False,
) -> list[LearningSample]:
    return get_operations_store().list_learning_samples(
        status=status,
        task=task,
        limit=limit,
        offset=offset,
        pending_inbox_only=pending_inbox_only,
    )


@router.get("/rentals", response_model=list[LearningRentalSummary])
def list_learning_rentals(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> list[LearningRentalSummary]:
    return get_operations_store().list_learning_rentals()


@router.delete("/rentals/pending", response_model=DeleteLearningRentalResponse)
@router.delete("/rentals", response_model=DeleteLearningRentalResponse)
def delete_learning_rental(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
    rental_id: Annotated[str, Query(min_length=1)],
    vehicle_id: Annotated[str, Query(min_length=1)],
    expected_sample_count: Annotated[int, Query(ge=1)],
    sample_ids: Annotated[list[str], Body(min_length=1)],
) -> DeleteLearningRentalResponse:
    try:
        deleted = get_operations_store().delete_learning_rental(
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            expected_sample_count=expected_sample_count,
            sample_ids=sample_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if deleted is None:
        raise HTTPException(status_code=404, detail="找不到這組租單的待處理照片")
    deleted_samples, deleted_images = deleted
    return DeleteLearningRentalResponse(
        rental_id=rental_id.strip(),
        vehicle_id=vehicle_id.strip(),
        deleted_samples=deleted_samples,
        deleted_images=deleted_images,
        message="已從待處理清單移除這組租單；人工真值、照片與其他資料保留。",
    )


@router.get("/recent-approvals", response_model=list[LearningSample])
def list_recent_learning_approvals(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[LearningSample]:
    return get_operations_store().list_recent_learning_approvals(limit=limit)


@router.get("/samples/{sample_id}/images/{slot}", response_class=FileResponse)
def get_learning_image(
    sample_id: str,
    slot: str,
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> FileResponse:
    if slot not in {"primary", "secondary"}:
        raise HTTPException(status_code=404, detail="找不到影像欄位")
    path = get_operations_store().get_learning_sample_file(sample_id, slot)
    if path is None:
        raise HTTPException(status_code=404, detail="找不到困難樣本影像")
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@router.delete("/samples/{sample_id}", response_model=DeleteLearningSampleResponse)
def delete_learning_sample(
    sample_id: str,
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteLearningSampleResponse:
    try:
        deleted_images = get_operations_store().delete_learning_sample(sample_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if deleted_images is None:
        raise HTTPException(status_code=404, detail="找不到困難樣本")
    return DeleteLearningSampleResponse(
        sample_id=sample_id,
        deleted_images=deleted_images,
        message="已清除資料集紀錄與來源影像。",
    )


@router.get("/references", response_model=list[LearningSample])
def list_reference_samples(
    _: Annotated[AuthPrincipal, Depends(require_authenticated)],
    view: Annotated[str, Query()],
    limit: Annotated[int, Query(ge=1, le=12)] = 6,
) -> list[LearningSample]:
    if view not in {"left_front", "right_front", "left_rear", "right_rear"}:
        raise HTTPException(status_code=422, detail="拍攝範例只支援四角車身視角")
    return get_operations_store().list_reference_samples(
        expected_label=view,
        limit=limit,
    )


@router.get("/references/{sample_id}/image", response_class=FileResponse)
def get_reference_sample_image(
    sample_id: str,
    _: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> FileResponse:
    path = get_operations_store().get_reference_sample_file(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="找不到已核准的拍攝範例")
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@router.post("/samples/{sample_id}/training-set", response_model=LearningSample)
def add_learning_sample_to_training_set(
    sample_id: str,
    request: LearningTrainingApprovalRequest,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> LearningSample:
    store = get_operations_store()
    current = store.get_learning_sample(sample_id)
    if current is None:
        raise HTTPException(status_code=404, detail="找不到困難樣本")
    try:
        if (
            current.task == LearningTask.DAMAGE_PAIR
            and request.review_method == LearningReviewMethod.MANUAL_PAIR
            and request.pair_angle_result == PairAngleResult.INCOMPATIBLE
            and request.corrected_label == "cannot_assess"
        ):
            dataset_label = "cannot_assess"
        else:
            dataset_label = _validated_training_label(current, request.corrected_label)
        review_metadata = _validated_review_metadata(current, request, dataset_label)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if current.status in {LearningStatus.APPROVED, LearningStatus.TRAINED}:
        if current.corrected_label != dataset_label:
            if (
                current.task != LearningTask.LICENSE_PLATE
                or current.status == LearningStatus.TRAINED
            ):
                raise HTTPException(
                    status_code=409,
                    detail="這筆資料已核准或訓練，不能用不同人工真值重複加入",
                )
        else:
            return current
    settings = get_runtime_recognition_settings()
    reference_enabled = (
        current.task == LearningTask.EXTERIOR_VIEW
        and current.sample_kind == "good_reference"
        and (current.score or 0) >= settings.active_learning_min_training_score
    )
    try:
        reviewed = store.review_learning_sample(
            sample_id,
            approved=True,
            corrected_label=dataset_label,
            reviewer_notes=request.reviewer_notes or "人工確認訓練真值",
            training_eligible=(
                current.task != LearningTask.LICENSE_PLATE
                and dataset_label != "cannot_assess"
            ),
            reference_enabled=reference_enabled,
            review_metadata=review_metadata,
            actor=principal.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reviewed is None:
        raise HTTPException(status_code=404, detail="找不到困難樣本")
    return reviewed


MANUAL_TRUTH_MAX_FILES = 30
# Folder uploads arrive as many batches; more than this many at once get a 429.
MANUAL_TRUTH_MAX_CONCURRENT_BATCHES = 2
MANUAL_TRUTH_RETRY_AFTER_SECONDS = 2
MANUAL_TRUTH_INTERIOR_SLOTS = {"interior_front", "interior_rear"}
MANUAL_SINGLE_REVIEW_TASKS = {
    LearningTask.INTERIOR_CLEANLINESS,
    LearningTask.VISIBLE_DAMAGE,
}
_manual_truth_batches_lock = threading.Lock()
_manual_truth_batches_in_flight = 0
# Serializes the duplicate check and insert so parallel batches cannot both add one photo.
_manual_truth_write_lock = threading.Lock()


@contextmanager
def _manual_truth_batch_slot() -> Iterator[None]:
    global _manual_truth_batches_in_flight
    with _manual_truth_batches_lock:
        if _manual_truth_batches_in_flight >= MANUAL_TRUTH_MAX_CONCURRENT_BATCHES:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="同時上傳的批次過多，請稍後再試",
                headers={"Retry-After": str(MANUAL_TRUTH_RETRY_AFTER_SECONDS)},
            )
        _manual_truth_batches_in_flight += 1
    try:
        yield
    finally:
        with _manual_truth_batches_lock:
            _manual_truth_batches_in_flight -= 1


@router.post(
    "/manual-truths/duplicates", response_model=ManualTruthDuplicateCheckResponse
)
def check_manual_truth_duplicates(
    request: ManualTruthDuplicateCheckRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ManualTruthDuplicateCheckResponse:
    """Tell the uploader which photos are already in a dataset before sending them."""
    hashes = [value.lower() for value in request.hashes if _SHA256_HEX.fullmatch(value)]
    found = get_operations_store().existing_learning_content_hashes(
        task=request.task, content_hashes=hashes
    )
    return ManualTruthDuplicateCheckResponse(duplicates=sorted(found))


@router.post("/manual-truths", response_model=ManualTruthUploadResponse)
async def upload_manual_truths(
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
    task: Annotated[LearningTask, Form()],
    corrected_label: Annotated[str, Form(min_length=1, max_length=80)],
    images: Annotated[list[UploadFile], File()],
    destination: Annotated[
        ManualTruthDestination, Form()
    ] = ManualTruthDestination.TRUTH,
    inspection_slot: Annotated[str | None, Form()] = None,
    secondary_image: Annotated[UploadFile | None, File()] = None,
    reviewer_notes: Annotated[str, Form(max_length=500)] = "",
) -> ManualTruthUploadResponse:
    """Add operator-supplied photos straight to one dataset, bypassing any model."""
    is_pair = task == LearningTask.DAMAGE_PAIR
    slot = (inspection_slot or "").strip() or None
    try:
        if task == LearningTask.LICENSE_PLATE:
            raise ValueError("車牌辨識使用預訓練模型，不接受直接上傳真值")
        label = _validated_task_label(
            task, corrected_label.strip(), has_secondary_image=is_pair
        )
        if task == LearningTask.INTERIOR_CLEANLINESS:
            if slot not in MANUAL_TRUTH_INTERIOR_SLOTS:
                raise ValueError("車內清潔樣本請選擇前座或後座")
        elif slot is not None:
            raise ValueError("只有車內清潔樣本需要選擇前座或後座")
        if is_pair:
            if len(images) != 1 or secondary_image is None:
                raise ValueError("前後車損比對一次上傳一組：取車照與還車照各 1 張")
        elif secondary_image is not None:
            raise ValueError("此任務只接受單張照片")
        elif not 1 <= len(images) <= MANUAL_TRUTH_MAX_FILES:
            raise ValueError(f"一次可上傳 1–{MANUAL_TRUTH_MAX_FILES} 張照片")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _manual_truth_batch_slot():
        settings = get_settings()
        store = get_operations_store()
        secondary_content: bytes | None = None
        secondary_filename: str | None = None
        if is_pair and secondary_image is not None:
            secondary_content = await read_validated_image(
                secondary_image, settings.max_upload_mb
            )
            secondary_filename = secondary_image.filename or "return.jpg"
        notes = reviewer_notes.strip() or "後台直接上傳的人工真值"
        results: list[ManualTruthUploadItem] = []
        for upload in images:
            filename = upload.filename or "manual-upload.jpg"
            try:
                content = await read_validated_image(upload, settings.max_upload_mb)
            except HTTPException as exc:
                results.append(ManualTruthUploadItem(
                    filename=filename,
                    status=ManualTruthUploadStatus.FAILED,
                    message=str(exc.detail),
                ))
                continue
            with _manual_truth_write_lock:
                item, _ = _save_manual_truth(
                    store=store,
                    task=task,
                    label=label,
                    destination=destination,
                    slot=slot,
                    notes=notes,
                    actor=principal.subject,
                    filename=filename,
                    content=content,
                    secondary_content=secondary_content,
                    secondary_filename=secondary_filename,
                )
            results.append(item)

    return ManualTruthUploadResponse(
        task=task,
        corrected_label=label,
        destination=destination,
        inspection_slot=slot,
        created=sum(item.status == ManualTruthUploadStatus.CREATED for item in results),
        results=results,
    )


def _save_manual_truth(
    *,
    store: OperationsStore,
    task: LearningTask,
    label: str,
    destination: ManualTruthDestination,
    slot: str | None,
    notes: str,
    actor: str,
    filename: str,
    content: bytes,
    secondary_content: bytes | None,
    secondary_filename: str | None,
) -> tuple[ManualTruthUploadItem, LearningSample | None]:
    # Re-collecting a known photo would overwrite its saved review, so skip it.
    existing = store.find_learning_sample_by_content(
        task=task,
        primary_content=content,
        secondary_content=secondary_content,
    )
    if existing is not None:
        return ManualTruthUploadItem(
            filename=filename,
            status=ManualTruthUploadStatus.DUPLICATE,
            sample_id=existing.id,
            message="這張照片已在此資料集中，未重複加入",
        ), None
    metadata: dict[str, object] = {
        "source": "manual_upload",
        "uploaded_by": actor,
    }
    if slot is not None:
        metadata["inspection_slot"] = slot
    truth: LearningSample | None = None
    try:
        sample: LearningSample | None = store.collect_learning_sample(
            task=task,
            primary_content=content,
            primary_filename=filename,
            secondary_content=secondary_content,
            secondary_filename=secondary_filename,
            trigger_reasons=["manual_upload"],
            expected_label=label,
            metadata=metadata,
        )
        if destination == ManualTruthDestination.TRUTH and sample is not None:
            sample = store.review_learning_sample(
                sample.id,
                approved=True,
                corrected_label=label,
                reviewer_notes=notes,
                training_eligible=True,
                review_metadata={
                    "schema_version": 1,
                    "annotations": [],
                    "vehicle_identity": None,
                    "localization_eligible": False,
                    "review_method": (
                        LearningReviewMethod.MANUAL_SINGLE.value
                        if task in MANUAL_SINGLE_REVIEW_TASKS
                        else None
                    ),
                    "pair_angle_result": None,
                },
                actor=actor,
            )
            truth = sample
    except (OSError, ValueError, sqlite3.Error) as exc:
        return ManualTruthUploadItem(
            filename=filename,
            status=ManualTruthUploadStatus.FAILED,
            message=f"無法保存：{exc}",
        ), None
    return ManualTruthUploadItem(
        filename=filename,
        status=ManualTruthUploadStatus.CREATED,
        sample_id=sample.id if sample else None,
    ), truth


@router.delete("/samples/{sample_id}/approval", response_model=LearningSample)
def cancel_learning_sample_approval(
    sample_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> LearningSample:
    try:
        reopened = get_operations_store().cancel_learning_sample_approval(
            sample_id,
            actor=principal.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reopened is None:
        raise HTTPException(status_code=404, detail="找不到人工真值樣本")
    return reopened


@router.get("/stats", response_model=LearningStats)
def get_learning_stats(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> LearningStats:
    store = get_operations_store()
    settings = get_runtime_recognition_settings()
    status_counts = store.learning_counts("status")
    approved_exterior = store.count_approved_exterior_trainable()
    runs = store.list_training_runs(limit=1)
    datasets = dataset_training_statuses(store, settings)
    trainable = [item for item in datasets if item.runner != TrainingRunner.NONE]
    upcoming = min(trainable, key=lambda item: item.remaining, default=None)
    return LearningStats(
        total=sum(status_counts.values()),
        pending_review=store.count_learning_samples(pending_inbox_only=True),
        approved=status_counts.get(LearningStatus.APPROVED.value, 0),
        rejected=status_counts.get(LearningStatus.REJECTED.value, 0),
        trained=status_counts.get(LearningStatus.TRAINED.value, 0),
        by_task=store.learning_counts("task"),
        # Rejected records remain auditable in the database, but must not be
        # presented as samples still available to the active-learning loop.
        by_lighting=store.learning_counts("lighting", include_rejected=False),
        approved_exterior_untrained=approved_exterior,
        next_training_in=max(0, settings.active_learning_min_approved - approved_exterior),
        auto_train_enabled=settings.active_learning_auto_train,
        min_approved_for_training=settings.active_learning_min_approved,
        last_training_run=runs[0] if runs else None,
        approved_untrained=sum(item.approved_untrained for item in trainable),
        next_training_task=upcoming.task if upcoming else None,
        datasets=datasets,
    )
