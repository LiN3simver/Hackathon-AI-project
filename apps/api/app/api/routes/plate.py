from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.plate import PlateModelInfo, PlateRecognitionResponse
from app.services.active_learning import collect_plate_sample
from app.services.auth import require_admin, require_user
from app.services.model_time_limits import ModelTimeoutError, RecognitionModel, run_with_time_limit
from app.services.rapid_plate_recognition import (
    get_plate_recognizer,
    model_info,
    timed_out_plate_prediction,
)
from app.services.rental_context import get_principal_rental_identity
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/plate", tags=["license plate recognition"])


@router.get("/model-info", response_model=PlateModelInfo)
def active_plate_model(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> PlateModelInfo:
    return PlateModelInfo(**model_info())


@router.post("/recognize", response_model=PlateRecognitionResponse)
async def recognize_plate(
    principal: Annotated[AuthPrincipal, Depends(require_user)],
    image: Annotated[UploadFile, File()],
) -> PlateRecognitionResponse:
    try:
        rental_id, vehicle_id = get_principal_rental_identity(principal)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    content = await read_validated_image(image, get_settings().max_upload_mb)
    request_id = str(uuid4())
    filename = image.filename or "plate.jpg"
    try:
        prediction = await run_in_threadpool(
            run_with_time_limit, RecognitionModel.PLATE, get_plate_recognizer().recognize, content,
        )
    except ModelTimeoutError as exc:
        prediction = timed_out_plate_prediction(exc.seconds)
    collect_plate_sample(
        content=content, filename=filename, request_id=request_id,
        rental_id=rental_id, vehicle_id=vehicle_id,
        prediction=prediction, force_review=True,
    )
    return PlateRecognitionResponse(
        request_id=request_id, filename=filename, prediction=prediction,
    )
