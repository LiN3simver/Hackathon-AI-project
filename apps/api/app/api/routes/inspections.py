import hashlib
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes.quality import evaluate_quality_capture
from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.capture_test import CaptureTestResponse
from app.schemas.inspection import (
    ActiveRentalContext,
    CreateInspectionSessionRequest,
    InspectionAnalysisJob,
    InspectionAnalysisJobStatus,
    InspectionCaptureResponse,
    InspectionSession,
    InspectionSessionStatus,
    InspectionStage,
)
from app.schemas.quality import (
    CheckStatus,
    ExpectedView,
    QualityCheck,
    QualityResponse,
    QualityStatus,
)
from app.services.auth import require_authenticated, require_ops, require_user
from app.services.image_quality import InvalidImageError
from app.services.inspection_storage import (
    InvalidInspectionObjectKey,
    get_inspection_storage,
)
from app.services.operations_store import (
    AnalysisQueueFullError,
    IncompleteInspectionError,
    ReturnInspectionNotReadyError,
    get_operations_store,
)
from app.services.rental_context import get_active_rental_context
from app.services.same_photo import SAME_AS_PICKUP_MESSAGE
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/inspections", tags=["inspection sessions"])


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_pickup_copy(
    quality: QualityResponse,
    session: InspectionSession,
    content_hash: str,
) -> QualityResponse:
    """A return photo must be new; the pickup file uploaded again fails."""
    if session.stage != InspectionStage.RETURN or not get_operations_store().has_pickup_photo_content(
        rental_id=session.rental_id,
        vehicle_id=session.vehicle_id,
        content_hash=content_hash,
    ):
        return quality
    check = QualityCheck(
        key="same_as_pickup",
        label="還車新照片",
        status=CheckStatus.FAIL,
        message=SAME_AS_PICKUP_MESSAGE,
    )
    return quality.model_copy(update={
        "status": QualityStatus.FAIL,
        "retake_required": True,
        "summary": SAME_AS_PICKUP_MESSAGE,
        "checks": [check, *quality.checks],
    })


def _require_session_access(
    session: InspectionSession,
    principal: AuthPrincipal,
) -> None:
    if principal.audience.value == "ops":
        return
    if (
        session.rental_id != principal.rental_id
        or session.vehicle_id != principal.vehicle_id
    ):
        # Do not reveal that another rental account's private session exists.
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")


@router.post("", response_model=InspectionSession, status_code=201)
def create_inspection_session(
    request: CreateInspectionSessionRequest,
    principal: Annotated[AuthPrincipal, Depends(require_user)],
) -> InspectionSession:
    rental_context = get_active_rental_context(principal)
    if request.restart and request.stage != InspectionStage.PICKUP:
        raise HTTPException(status_code=422, detail="重新開始只能建立借車流程")
    try:
        return get_operations_store().create_inspection_session(
            rental_id=rental_context.rental_id,
            vehicle_id=rental_context.vehicle_id,
            stage=request.stage,
            enforce_stage_order=True,
            resume_existing=not request.restart,
        )
    except ReturnInspectionNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/context", response_model=ActiveRentalContext)
def get_inspection_rental_context(
    principal: Annotated[AuthPrincipal, Depends(require_user)],
) -> ActiveRentalContext:
    context = get_active_rental_context(principal)
    state = get_operations_store().get_rental_inspection_state(
        rental_id=context.rental_id, vehicle_id=context.vehicle_id,
    )
    return context.model_copy(update=state)


@router.get("", response_model=list[InspectionSession])
def list_inspection_sessions(
    _: Annotated[AuthPrincipal, Depends(require_ops)],
    stage: Annotated[InspectionStage | None, Query()] = None,
    session_status: Annotated[InspectionSessionStatus | None, Query(alias="status")] = None,
    rental_id: Annotated[str | None, Query(max_length=80)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[InspectionSession]:
    return get_operations_store().list_inspection_sessions(
        stage=stage,
        status=session_status,
        rental_id=rental_id,
        limit=limit,
    )


@router.get("/{session_id}", response_model=InspectionSession)
def get_inspection_session(
    session_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> InspectionSession:
    session = get_operations_store().get_inspection_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    _require_session_access(session, principal)
    return session


@router.post(
    "/{session_id}/photos/check",
    response_model=CaptureTestResponse,
)
async def check_inspection_photo(
    session_id: str,
    image: Annotated[UploadFile, File()],
    slot: Annotated[ExpectedView, Form()],
    principal: Annotated[AuthPrincipal, Depends(require_user)],
) -> CaptureTestResponse:
    store = get_operations_store()
    existing_session = store.get_inspection_session(session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    _require_session_access(existing_session, principal)
    if existing_session.status in {
        InspectionSessionStatus.CAPTURE_COMPLETED,
        InspectionSessionStatus.CANCELLED,
    }:
        raise HTTPException(status_code=409, detail="此租還車檢查流程已關閉，不能再新增照片")
    if slot not in existing_session.required_slots and slot != ExpectedView.DAMAGE_DETAIL:
        raise HTTPException(status_code=422, detail="此租還車階段不需要這個拍攝位置")

    started = perf_counter()
    content = await read_validated_image(image, get_settings().max_upload_mb)
    raw_filename = image.filename or "capture.jpg"
    filename = raw_filename.replace("\\", "/").rsplit("/", 1)[-1][:255] or "capture.jpg"
    try:
        quality = await run_in_threadpool(
            evaluate_quality_capture,
            content=content,
            filename=filename,
            expected_view=slot,
            rental_id=existing_session.rental_id,
            vehicle_id=existing_session.vehicle_id,
            collect_samples=False,
            plate_scope=f"inspection:{session_id}",
            rental_plate=principal.license_plate,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    quality = _reject_pickup_copy(quality, existing_session, _content_hash(content))
    quality = CaptureTestResponse(**quality.model_dump())
    # The formal flow shows the same recognition time as test mode.
    quality.recognition_ms = round((perf_counter() - started) * 1000, 1)
    return quality


@router.post(
    "/{session_id}/photos",
    response_model=InspectionCaptureResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_inspection_photo(
    session_id: str,
    image: Annotated[UploadFile, File()],
    slot: Annotated[ExpectedView, Form()],
    principal: Annotated[AuthPrincipal, Depends(require_user)],
    damage_view: Annotated[ExpectedView | None, Form()] = None,
) -> InspectionCaptureResponse:
    store = get_operations_store()
    existing_session = store.get_inspection_session(session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    _require_session_access(existing_session, principal)
    if existing_session.status in {
        InspectionSessionStatus.CAPTURE_COMPLETED,
        InspectionSessionStatus.CANCELLED,
    }:
        raise HTTPException(status_code=409, detail="此租還車檢查流程已關閉，不能再新增照片")
    if slot not in existing_session.required_slots and slot != ExpectedView.DAMAGE_DETAIL:
        raise HTTPException(status_code=422, detail="此租還車階段不需要這個拍攝位置")
    if damage_view is not None and (
        slot != ExpectedView.DAMAGE_DETAIL
        or damage_view not in existing_session.required_slots
        or damage_view in {ExpectedView.INTERIOR_FRONT, ExpectedView.INTERIOR_REAR}
    ):
        raise HTTPException(
            status_code=422, detail="車損角度只能是左前、右前、左後或右後，且只用於車損特寫"
        )

    content = await read_validated_image(image, get_settings().max_upload_mb)
    content_hash = _content_hash(content)
    raw_filename = image.filename or "capture.jpg"
    filename = raw_filename.replace("\\", "/").rsplit("/", 1)[-1][:255] or "capture.jpg"
    try:
        quality = await run_in_threadpool(
            evaluate_quality_capture,
            content=content,
            filename=filename,
            expected_view=slot,
            rental_id=existing_session.rental_id,
            vehicle_id=existing_session.vehicle_id,
            collect_samples=True,
            plate_scope=f"inspection:{session_id}",
            rental_plate=principal.license_plate,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    quality = _reject_pickup_copy(quality, existing_session, content_hash)

    object_key = None
    storage = get_inspection_storage()
    if quality.status in {QualityStatus.PASS, QualityStatus.WARNING}:
        try:
            object_key = storage.save(session_id, quality.request_id, content)
        except (OSError, InvalidInspectionObjectKey) as exc:
            raise HTTPException(status_code=507, detail="私有照片儲存失敗，請稍後重試") from exc

    try:
        session = store.record_inspection_photo(
            session_id,
            slot=slot,
            filename=filename,
            quality_request_id=quality.request_id,
            quality_status=quality.status,
            quality_score=quality.score,
            object_key=object_key,
            content_hash=(content_hash if object_key else None),
            plate_prediction=quality.plate_prediction,
            damage_view=damage_view,
        )
    except ValueError as exc:
        if object_key:
            storage.delete(object_key)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if session is None:
        if object_key:
            storage.delete(object_key)
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    return InspectionCaptureResponse(quality=quality, session=session)


@router.get(
    "/{session_id}/photos/{photo_id}/image",
    response_class=FileResponse,
)
def get_inspection_photo_image(
    session_id: str,
    photo_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> FileResponse:
    store = get_operations_store()
    session = store.get_inspection_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="找不到可讀取的檢查照片")
    _require_session_access(session, principal)
    photo = store.get_inspection_photo(session_id, photo_id)
    object_key = store.get_inspection_photo_object_key(session_id, photo_id)
    if photo is None or object_key is None:
        raise HTTPException(status_code=404, detail="找不到可讀取的檢查照片")
    try:
        path = get_inspection_storage().resolve(object_key)
    except InvalidInspectionObjectKey as exc:
        raise HTTPException(status_code=404, detail="照片儲存路徑無效") from exc
    if path is None:
        raise HTTPException(status_code=404, detail="檢查照片檔案不存在")
    return FileResponse(
        path,
        filename=photo.filename,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{session_id}/analysis-jobs", response_model=list[InspectionAnalysisJob])
def list_inspection_analysis_jobs(
    session_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
    job_status: Annotated[
        InspectionAnalysisJobStatus | None, Query(alias="status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[InspectionAnalysisJob]:
    store = get_operations_store()
    session = store.get_inspection_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    _require_session_access(session, principal)
    return store.list_inspection_analysis_jobs(
        session_id=session_id,
        status=job_status,
        limit=limit,
    )


@router.post("/{session_id}/complete", response_model=InspectionSession)
def complete_inspection_capture(
    session_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_user)],
) -> InspectionSession:
    store = get_operations_store()
    existing_session = store.get_inspection_session(session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    _require_session_access(existing_session, principal)
    try:
        session = store.complete_inspection_capture(session_id)
    except IncompleteInspectionError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "仍有必要拍攝槽位未通過即時防呆",
                "missing_slots": [slot.value for slot in exc.missing_slots],
            },
        ) from exc
    except AnalysisQueueFullError as exc:
        # Return completion queues the whole rental; nothing is saved, retry later.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "15"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail="找不到租還車檢查流程")
    return session
