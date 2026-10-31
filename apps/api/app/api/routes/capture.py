"""Transient camera hints. Preview frames never enter storage or training."""

import asyncio
import logging
from time import monotonic, perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.routes.quality import evaluate_quality_capture
from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.capture_preview import CapturePreviewResponse
from app.schemas.capture_test import CaptureTestAnalysis, CaptureTestResponse
from app.schemas.history import AutomationRoute
from app.schemas.inspection import InspectionAnalysisTask, InspectionStage
from app.schemas.quality import ExpectedView, QualityStatus
from app.services.auth import require_user
from app.services.decision_routing import route_analysis_decision
from app.services.image_quality import InvalidImageError
from app.services.inspection_analysis import (
    _best_aligned_damage_pair,
    _interior_result,
    _view_guarded_damage_comparison,
)
from app.services.model_time_limits import RecognitionModel, override_time_limits, run_with_time_limit
from app.services.same_photo import SAME_AS_PICKUP_MESSAGE, is_same_photo
from app.services.upload_validation import read_validated_image
from app.services.vehicle_coverage import get_vehicle_coverage_detector

router = APIRouter(prefix="/capture", tags=["camera preview"])
logger = logging.getLogger(__name__)
# One standard photo of the angle plus up to four close-ups tagged with it.
MAX_PICKUP_DETAILS = 5
DAMAGE_VIEWS = {
    ExpectedView.LEFT_FRONT, ExpectedView.RIGHT_FRONT,
    ExpectedView.LEFT_REAR, ExpectedView.RIGHT_REAR,
}
# Whole test capture (quality checks plus follow-up analysis) gives up after this.
TEST_RECOGNITION_LIMIT_SECONDS = 250.0
ABANDONED_MESSAGE = f"辨識超過 {TEST_RECOGNITION_LIMIT_SECONDS:g} 秒，已放棄辨識，請重拍。"


@router.post("/test", response_model=CaptureTestResponse)
async def test_capture(
    _: Annotated[AuthPrincipal, Depends(require_user)],
    image: Annotated[UploadFile, File()],
    expected_view: Annotated[ExpectedView, Form()],
    stage: Annotated[InspectionStage, Form()] = InspectionStage.PICKUP,
    pickup_image: Annotated[UploadFile | None, File()] = None,
    pickup_details: Annotated[list[UploadFile] | None, File()] = None,
    damage_view: Annotated[ExpectedView | None, Form()] = None,
    capture_flow_id: Annotated[str | None, Form(max_length=128)] = None,
) -> CaptureTestResponse:
    """Run the production capture checks without sessions, storage or training."""
    if damage_view is not None and (
        expected_view != ExpectedView.DAMAGE_DETAIL or damage_view not in DAMAGE_VIEWS
    ):
        raise HTTPException(status_code=422, detail="車損角度只能是左前、右前、左後或右後，且只用於車損特寫")
    started = perf_counter()
    deadline = monotonic() + TEST_RECOGNITION_LIMIT_SECONDS
    content = await read_validated_image(image, get_settings().max_upload_mb)
    filename = (image.filename or "capture.jpg").replace("\\", "/").rsplit("/", 1)[-1][:255] or "capture.jpg"
    pickup_content = (
        await read_validated_image(pickup_image, get_settings().max_upload_mb)
        if pickup_image is not None else None
    )
    pickup_detail_contents = [
        await read_validated_image(detail, get_settings().max_upload_mb)
        for detail in (pickup_details or [])[:MAX_PICKUP_DETAILS]
    ]
    if expected_view == ExpectedView.DAMAGE_DETAIL:
        # Test close-ups are accepted directly; only the damage pair model runs.
        response = CaptureTestResponse(
            request_id=str(uuid4()), filename=filename, expected_view=expected_view,
            status=QualityStatus.PASS, summary="", retake_required=False,
            checks=[], quality_skipped=True,
        )
    else:
        try:
            # The worker thread cannot be interrupted; we only stop waiting for it.
            quality = await asyncio.wait_for(run_in_threadpool(
                evaluate_quality_capture, content=content, filename=filename,
                expected_view=expected_view, collect_samples=False,
                plate_scope=(
                    f"test:{_.subject}:{_.rental_id}:{_.vehicle_id}:{stage.value}:{capture_flow_id}"
                    if capture_flow_id else None
                ),
                rental_plate=_.license_plate,
            ), timeout=max(0.0, deadline - monotonic()))
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TimeoutError:
            logger.warning("capture_test_abandoned phase=quality view=%s", expected_view.value)
            response = CaptureTestResponse(
                request_id=str(uuid4()), filename=filename, expected_view=expected_view,
                status=QualityStatus.FAIL, summary=ABANDONED_MESSAGE, retake_required=True,
                checks=[], recognition_abandoned=True,
            )
            response.recognition_ms = round((perf_counter() - started) * 1000, 1)
            return response
        response = CaptureTestResponse(**quality.model_dump())
    def timed_response():
        response.recognition_ms = round((perf_counter() - started) * 1000, 1)
        return response
    if response.status == QualityStatus.FAIL:
        return timed_response()
    task = None
    if expected_view in {ExpectedView.INTERIOR_FRONT, ExpectedView.INTERIOR_REAR}:
        if stage == InspectionStage.RETURN:
            task = InspectionAnalysisTask.INTERIOR_CLEANLINESS
    elif expected_view == ExpectedView.DAMAGE_DETAIL:
        # Close-ups are never judged alone; a return close-up is only compared
        # against the pickup photos of its angle taken in the same test run.
        if stage == InspectionStage.RETURN and (damage_view or pickup_detail_contents):
            task = InspectionAnalysisTask.DAMAGE_COMPARISON
    elif stage == InspectionStage.RETURN:
        task = InspectionAnalysisTask.DAMAGE_COMPARISON
    if task is None:
        return timed_response()

    def analyze():
        if any(is_same_photo(pickup, content) for pickup in [
            *([pickup_content] if pickup_content is not None else []), *pickup_detail_contents,
        ]):
            raise ValueError(SAME_AS_PICKUP_MESSAGE)
        if task == InspectionAnalysisTask.INTERIOR_CLEANLINESS:
            return _interior_result(
                None, content, filename=filename, rental_id="", vehicle_id="",
                expected_view="front" if expected_view == ExpectedView.INTERIOR_FRONT else "rear",
                store=None, persist=False,
            )
        matched_detail = None
        if expected_view == ExpectedView.DAMAGE_DETAIL or (pickup_content is None and pickup_detail_contents):
            if not pickup_detail_contents:
                raise ValueError("缺少同角度的借車測試照片；請先在借車測試拍攝此角度，再測試還車車損比較。")
            matched_detail, comparison = _best_aligned_damage_pair(pickup_detail_contents, content)
        elif pickup_content is None:
            raise ValueError("缺少同角度的借車測試照片；請先完成借車測試，再測試還車車損比較。")
        else:
            comparison = _view_guarded_damage_comparison(
                pickup_content, content, expected_view=expected_view,
            )
        return {
            "comparison_status": comparison.status.value,
            "risk_score": comparison.risk_score,
            "summary": comparison.summary,
            "verification": comparison.verification.model_dump(mode="json") if comparison.verification else None,
            "specialist": (
                comparison.specialist.model_dump(mode="json", exclude={"annotated_return_image", "regions"})
                if comparison.specialist else None
            ),
            "model": comparison.model_version,
            "matched_pickup_detail": matched_detail,
            "automated": True,
        }, comparison.status.value != "no_change"

    def analyze_with_test_limits():
        with override_time_limits({
            RecognitionModel.FRONT_CLEANLINESS: TEST_RECOGNITION_LIMIT_SECONDS,
            RecognitionModel.REAR_CLEANLINESS: TEST_RECOGNITION_LIMIT_SECONDS,
            RecognitionModel.DAMAGE_PAIR: TEST_RECOGNITION_LIMIT_SECONDS,
        }):
            return analyze()

    try:
        result, needs_review = await asyncio.wait_for(
            run_in_threadpool(analyze_with_test_limits), timeout=max(0.0, deadline - monotonic()),
        )
        routing = route_analysis_decision(task, result, legacy_needs_review=needs_review)
        result["automation"] = routing.as_result_payload()
        response.analyses.append(CaptureTestAnalysis(
            task=task, result=result,
            needs_review=routing.effective_route != AutomationRoute.AUTO_CLEAR,
        ))
    except TimeoutError:
        logger.warning("capture_test_abandoned phase=analysis task=%s", task.value)
        response.recognition_abandoned = True
        response.analyses.append(CaptureTestAnalysis(
            task=task, needs_review=True, error_message=ABANDONED_MESSAGE,
        ))
    except Exception as exc:
        logger.warning("capture_test_analysis_unavailable task=%s error=%s", task.value, type(exc).__name__)
        response.analyses.append(CaptureTestAnalysis(task=task, needs_review=True, error_message=str(exc)))
    return timed_response()


@router.post("/preview", response_model=CapturePreviewResponse)
async def preview_capture(
    _: Annotated[AuthPrincipal, Depends(require_user)],
    image: Annotated[UploadFile, File()],
) -> CapturePreviewResponse:
    """Live framing hint only; the plate is read once from the captured photo."""
    content = await read_validated_image(image, 2)
    response = CapturePreviewResponse()
    try:
        response.coverage_prediction = await run_in_threadpool(
            run_with_time_limit,
            RecognitionModel.VEHICLE_DETECTION, get_vehicle_coverage_detector().predict, content,
        )
    except Exception as exc:
        logger.warning("capture_preview_unavailable model=coverage_prediction error=%s", type(exc).__name__)
        response.messages.append("車身偵測暫時無法使用，請依輪廓對齊。")
    return response
