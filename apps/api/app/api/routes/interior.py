from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.cleanliness import CleanlinessLevel, InteriorAnalysisResponse, InteriorView
from app.services.active_learning import collect_interior_sample
from app.services.auth import require_user
from app.services.cleanliness import (
    InvalidModelResponseError,
    ModelUnavailableError,
    get_cleanliness_analyzer,
)
from app.services.email_notifications import dispatch_notification_outbox
from app.services.model_time_limits import ModelTimeoutError
from app.services.operations_store import get_operations_store
from app.services.rental_context import get_principal_rental_identity
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/interior", tags=["interior cleanliness"])


@router.post("/analyze", response_model=InteriorAnalysisResponse)
async def analyze_interior(
    image: Annotated[UploadFile, File()],
    principal: Annotated[AuthPrincipal, Depends(require_user)],
    background_tasks: BackgroundTasks,
    expected_view: Annotated[InteriorView | None, Form()] = None,
    test_mode: Annotated[bool, Form()] = False,
) -> InteriorAnalysisResponse:
    try:
        rental_id, vehicle_id = get_principal_rental_identity(principal)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    settings = get_settings()
    content = await read_validated_image(image, settings.max_upload_mb)

    model_name = f"{expected_view.value}_cleanliness" if expected_view else "interior_cleanliness"
    try:
        assessment, latency_ms = await run_in_threadpool(
            get_cleanliness_analyzer().analyze,
            content,
            expected_view.value if expected_view else None,
        )
    except ModelTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except (ModelUnavailableError, InvalidModelResponseError) as local_error:
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if isinstance(local_error, ModelUnavailableError)
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=response_status, detail=str(local_error)) from local_error

    filename = image.filename or "interior.jpg"
    if assessment.specialist is not None:
        model_name = assessment.specialist.model_version
    elif assessment.localization_model:
        model_name = assessment.localization_model
    request_id = str(uuid4())
    alert_id = None
    runtime = get_runtime_recognition_settings()
    should_alert = (
        assessment.is_vehicle_interior
        and assessment.image_usable
        and assessment.level == CleanlinessLevel.DIRTY
        and assessment.immediate_cleaning_required
        and assessment.confidence >= runtime.interior_alert_confidence_threshold
    )
    if assessment.specialist is not None:
        should_alert = assessment.image_usable and assessment.specialist.suspected_dirty
    should_alert = should_alert or (assessment.image_usable and bool(assessment.evidence))
    if should_alert and not test_mode:
        store = get_operations_store()
        alert = store.create_cleanliness_alert(
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            filename=filename,
            assessment=assessment,
        )
        alert_id = alert.id
        background_tasks.add_task(dispatch_notification_outbox, store=store)

    human_review_required = (
        assessment.level == CleanlinessLevel.CANNOT_ASSESS
        or assessment.confidence
        < runtime.interior_review_confidence_threshold
    )
    if assessment.specialist is not None:
        human_review_required = assessment.specialist.human_review_required
    if not test_mode:
        collect_interior_sample(
            content=content,
            filename=filename,
            request_id=request_id,
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            assessment=assessment,
            model_version=model_name,
            human_review_required=human_review_required,
        )
    return InteriorAnalysisResponse(
        request_id=request_id,
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        filename=filename,
        assessment=assessment,
        model=model_name,
        latency_ms=round(latency_ms, 1),
        alert_id=alert_id,
        human_review_required=human_review_required,
    )
