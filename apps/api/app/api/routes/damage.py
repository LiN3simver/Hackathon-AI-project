from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes.quality import EXTERIOR_VIEWS
from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.damage import (
    DamageCase,
    DamageComparisonStatus,
    DamageReviewDecision,
    DamageReviewRequest,
)
from app.schemas.quality import ExpectedView
from app.services.active_learning import (
    collect_damage_pair_sample,
)
from app.services.auth import require_ops, require_user
from app.services.damage_comparison import (
    InvalidDamageImageError,
)
from app.services.damage_pair_workflow import evaluate_damage_pair
from app.services.email_notifications import dispatch_notification_outbox
from app.services.operations_store import get_operations_store
from app.services.same_photo import SAME_AS_PICKUP_MESSAGE, is_same_photo
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/damage", tags=["new damage comparison"])


async def _read_image(image: UploadFile) -> bytes:
    return await read_validated_image(image, get_settings().max_upload_mb)


async def _compare_and_store_damage_pair(
    *,
    pickup_content: bytes,
    return_content: bytes,
    pickup_filename: str,
    return_filename: str,
    rental_id: str,
    vehicle_id: str,
    expected_view: ExpectedView,
) -> DamageCase:
    """Run one comparison and persist the case plus its paired learning sample."""
    store = get_operations_store()
    if store.has_damage_pair_images(pickup_content, return_content):
        raise HTTPException(
            status_code=409,
            detail="這組取還車照片已存在人工真值或覆核資料，請使用不同照片",
        )
    run_id = store.start_damage_pair_recognition(
        rental_id=rental_id, vehicle_id=vehicle_id
    )
    succeeded = False
    try:
        try:
            comparison = await run_in_threadpool(
                evaluate_damage_pair, pickup_content, return_content
            )
        except InvalidDamageImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stored_case = store.create_damage_case(
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            expected_view=expected_view,
            pickup_filename=pickup_filename,
            return_filename=return_filename,
            comparison=comparison,
        )
        sample_id = collect_damage_pair_sample(
            pickup_content=pickup_content,
            return_content=return_content,
            pickup_filename=pickup_filename,
            return_filename=return_filename,
            damage_case=stored_case,
            store=store,
        )
        if sample_id:
            store.attach_damage_case_learning_sample(stored_case.id, sample_id)
        succeeded = True
        return stored_case.model_copy(update={"comparison": comparison})
    finally:
        store.finish_damage_pair_recognition(run_id, failed=not succeeded)


@router.post("/compare", response_model=DamageCase, status_code=201)
async def compare_damage(
    pickup_image: Annotated[UploadFile, File()],
    return_image: Annotated[UploadFile, File()],
    rental_id: Annotated[str, Form(min_length=1, max_length=80)],
    vehicle_id: Annotated[str, Form(min_length=1, max_length=40)],
    expected_view: Annotated[ExpectedView, Form()],
    _: Annotated[AuthPrincipal, Depends(require_ops)],
) -> DamageCase:
    if expected_view not in EXTERIOR_VIEWS:
        raise HTTPException(status_code=422, detail="新車損比對只接受四個車外標準視角")
    pickup_content = await _read_image(pickup_image)
    return_content = await _read_image(return_image)
    return await _compare_and_store_damage_pair(
        pickup_content=pickup_content,
        return_content=return_content,
        pickup_filename=pickup_image.filename or "pickup.jpg",
        return_filename=return_image.filename or "return.jpg",
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        expected_view=expected_view,
    )


@router.post("/compare-user", response_model=DamageCase, status_code=201)
async def compare_damage_for_user(
    pickup_image: Annotated[UploadFile, File()],
    return_image: Annotated[UploadFile, File()],
    expected_view: Annotated[ExpectedView, Form()],
    principal: Annotated[AuthPrincipal, Depends(require_user)],
) -> DamageCase:
    """Compare a user-owned pair without trusting rental identity from multipart data."""
    if expected_view not in EXTERIOR_VIEWS:
        raise HTTPException(status_code=422, detail="新車損比對只接受四個車外標準視角")
    if not principal.rental_id or not principal.vehicle_id:
        raise HTTPException(status_code=409, detail="登入資訊缺少有效租單或車輛")

    pickup_content = await _read_image(pickup_image)
    return_content = await _read_image(return_image)
    return await _compare_and_store_damage_pair(
        pickup_content=pickup_content,
        return_content=return_content,
        pickup_filename=pickup_image.filename or "pickup.jpg",
        return_filename=return_image.filename or "return.jpg",
        rental_id=principal.rental_id,
        vehicle_id=principal.vehicle_id,
        expected_view=expected_view,
    )


@router.get("/cases", response_model=list[DamageCase])
def list_damage_cases(
    _: Annotated[AuthPrincipal, Depends(require_ops)],
) -> list[DamageCase]:
    return get_operations_store().list_damage_cases()


@router.get("/cases/{case_id}/images/{slot}", response_class=FileResponse)
def get_damage_case_image(
    case_id: str,
    slot: str,
    _: Annotated[AuthPrincipal, Depends(require_ops)],
) -> FileResponse:
    if slot not in {"pickup", "return"}:
        raise HTTPException(status_code=404, detail="找不到車損照片欄位")
    path = get_operations_store().get_damage_case_image_file(case_id, slot)
    if path is None:
        raise HTTPException(status_code=404, detail="找不到車損案件原始照片")
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@router.post("/cases/{case_id}/reanalyze", response_model=DamageCase)
async def reanalyze_damage_case(
    case_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> DamageCase:
    """Re-run the two-stage comparison on the stored photos of a pending case."""
    store = get_operations_store()
    damage_case = store.get_damage_case(case_id)
    if damage_case is None:
        raise HTTPException(status_code=404, detail="找不到車損比對案件")
    if damage_case.review_decision != DamageReviewDecision.PENDING:
        raise HTTPException(status_code=409, detail="此案件已完成覆核，不能重新辨識")
    pickup_path = store.get_damage_case_image_file(case_id, "pickup")
    return_path = store.get_damage_case_image_file(case_id, "return")
    if pickup_path is None or return_path is None:
        raise HTTPException(status_code=404, detail="找不到車損案件原始照片，無法重新辨識")
    pickup_content, return_content = pickup_path.read_bytes(), return_path.read_bytes()
    if is_same_photo(pickup_content, return_content):
        raise HTTPException(status_code=409, detail=SAME_AS_PICKUP_MESSAGE)
    try:
        comparison = await run_in_threadpool(
            evaluate_damage_pair, pickup_content, return_content
        )
    except InvalidDamageImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        updated = store.replace_damage_case_comparison(
            case_id, comparison, actor=principal.subject
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="找不到車損比對案件")
    return updated


@router.patch("/cases/{case_id}/review", response_model=DamageCase)
def review_damage_case(
    case_id: str,
    request: DamageReviewRequest,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
    background_tasks: BackgroundTasks,
) -> DamageCase:
    store = get_operations_store()
    damage_case = store.get_damage_case(case_id)
    if damage_case is None:
        raise HTTPException(status_code=404, detail="找不到車損比對案件")
    if damage_case.review_decision != DamageReviewDecision.PENDING:
        raise HTTPException(status_code=409, detail="此案件已完成覆核，不能覆寫稽核紀錄")
    # A miss of the models on aligned photos is exactly what the reviewer checks,
    # so only photos that could not be aligned cannot be confirmed as damage.
    if (
        request.decision == DamageReviewDecision.CONFIRMED_DAMAGE
        and damage_case.comparison.status != DamageComparisonStatus.REVIEW_REQUIRED
        and not damage_case.comparison.alignment.success
    ):
        raise HTTPException(status_code=409, detail="兩張照片對不上，請要求重拍")

    actor = principal.subject
    try:
        reviewed = store.review_damage_case(
            case_id, request.decision, request.reviewer_notes, actor=actor
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reviewed is None:
        raise HTTPException(status_code=404, detail="找不到車損比對案件")
    if reviewed.alert_id:
        background_tasks.add_task(dispatch_notification_outbox, store=store)
    return reviewed
