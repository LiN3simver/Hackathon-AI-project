from threading import Event, Thread
from time import monotonic
from typing import Any

from app.schemas.cleanliness import CleanlinessLevel
from app.schemas.damage import (
    AlignmentResult,
    DamageComparisonResult,
    DamageComparisonStatus,
)
from app.schemas.history import AutomationRoute
from app.schemas.inspection import (
    InspectionAnalysisJob,
    InspectionAnalysisTask,
)
from app.schemas.plate import PlatePrediction
from app.schemas.quality import ExpectedView
from app.services.active_learning import (
    collect_damage_pair_sample,
    collect_interior_sample,
    collect_plate_sample,
)
from app.services.cleanliness import (
    InvalidModelResponseError,
    ModelUnavailableError,
    get_cleanliness_analyzer,
)
from app.services.damage_comparison import get_damage_comparator
from app.services.damage_pair_workflow import evaluate_damage_pair
from app.services.decision_routing import route_analysis_decision
from app.services.email_notifications import dispatch_notification_outbox
from app.services.inspection_storage import (
    InspectionImageStorage,
    InvalidInspectionObjectKey,
    get_inspection_storage,
)
from app.services.model_time_limits import (
    ModelTimeoutError,
    RecognitionModel,
    run_with_time_limit,
)
from app.services.operations_store import OperationsStore, get_operations_store
from app.services.rapid_plate_recognition import MODEL_VERSION as PLATE_MODEL_VERSION
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.same_photo import SAME_AS_PICKUP_MESSAGE, is_same_photo
from app.services.view_classifier import get_view_classifier


class InspectionAnalysisContextError(RuntimeError):
    pass


def _incompatible_view_pair(summary: str) -> DamageComparisonResult:
    return DamageComparisonResult(
        status=DamageComparisonStatus.INCOMPATIBLE,
        alignment=AlignmentResult(
            success=False,
            method="view-consistency-guard",
            matched_features=0,
            inlier_ratio=0,
            overlap_ratio=0,
            score=0,
            message=summary,
        ),
        candidates=[],
        change_area_ratio=0,
        risk_score=0,
        summary=summary,
        model_version="view-consistency-guard-v1",
        latency_ms=0,
    )


def _view_guarded_damage_comparison(
    pickup_content: bytes,
    return_content: bytes,
    *,
    expected_view: ExpectedView,
) -> DamageComparisonResult:
    classifier = get_view_classifier()
    if not classifier.available:
        return _incompatible_view_pair(
            "視角模型目前不可用，系統未執行前後車損比較，請人工確認或重拍。"
        )

    threshold = get_runtime_recognition_settings().view_confidence_threshold
    try:
        pickup = run_with_time_limit(
            RecognitionModel.VIEW, classifier.predict, pickup_content, expected_view
        )
        returned = run_with_time_limit(
            RecognitionModel.VIEW, classifier.predict, return_content, expected_view
        )
    except ModelTimeoutError as exc:
        return _incompatible_view_pair(f"{exc}系統未執行前後車損比較。")
    except (OSError, RuntimeError, ValueError) as exc:
        return _incompatible_view_pair(
            f"視角模型無法確認前後照片是否同角度，系統未執行車損比較：{exc}"
        )

    if pickup.confidence < threshold or returned.confidence < threshold:
        return _incompatible_view_pair(
            "借車或還車照片的視角信心不足，系統未執行前後車損比較，請重拍。"
        )
    if pickup.predicted_view != expected_view or returned.predicted_view != expected_view:
        return _incompatible_view_pair(
            "借車或還車照片不符合指定的左前、右前、左後或右後槽位，"
            "系統未執行車損比較，請依同一指定角度重拍。"
        )
    return evaluate_damage_pair(pickup_content, return_content)


def _best_aligned_damage_pair(
    pickup_contents: list[bytes], return_content: bytes,
) -> tuple[int, DamageComparisonResult]:
    """Pair a close-up with its best-aligned pickup photo, then run the full model.

    Close-ups do not show a complete four-corner view, so the view guard does
    not apply; the cheap geometric alignment picks the pickup photo of the
    same spot among the candidates of that angle.
    """
    best = 0
    if len(pickup_contents) > 1:
        comparator = get_damage_comparator()
        scores = [
            comparator.compare(pickup, return_content).alignment.score
            for pickup in pickup_contents
        ]
        best = max(range(len(scores)), key=scores.__getitem__)
    return best, evaluate_damage_pair(pickup_contents[best], return_content)


def _plate_result(
    job: InspectionAnalysisJob,
    content: bytes,
    *,
    filename: str,
    rental_id: str,
    vehicle_id: str,
    store: OperationsStore,
) -> tuple[dict[str, Any], bool]:
    # Plates are read only while the photo is captured, and a capture without
    # a readable plate is blocked there. Ops records that result; it never
    # runs OCR again.
    captured = job.result.get("capture_prediction")
    prediction = (
        PlatePrediction.model_validate(captured) if captured else PlatePrediction(
            plate_detected=False, confidence=0, is_confident=False, image_usable=True,
            message="拍攝時沒有留下車牌辨識結果，請人工確認車牌。",
            model_version=PLATE_MODEL_VERSION, latency_ms=0, model_uncertain=True,
        )
    )
    collect_plate_sample(
        content=content,
        filename=filename,
        request_id=job.id,
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        prediction=prediction,
        store=store,
        force_review=True,
    )
    decision = "awaiting_human_confirmation"
    return {
        "prediction": prediction.model_dump(mode="json"),
        "expected_plate": prediction.expected_plate,
        "recognized_plate": prediction.normalized_plate,
        "inference_ms": prediction.latency_ms,
        "model_uncertain": prediction.model_uncertain,
        "model": prediction.model_version,
        "decision": decision,
        "non_blocking": True,
        "capture_prediction": captured,
    }, False


def _interior_result(
    job: InspectionAnalysisJob | None,
    content: bytes,
    *,
    filename: str,
    rental_id: str,
    vehicle_id: str,
    expected_view: str,
    store: OperationsStore | None,
    persist: bool = True,
) -> tuple[dict[str, Any], bool]:
    runtime = get_runtime_recognition_settings()
    model_name = f"{expected_view}_cleanliness"
    assessment, latency_ms = get_cleanliness_analyzer().analyze(content, expected_view)
    if assessment.specialist is not None:
        model_name = assessment.specialist.model_version
    elif assessment.localization_model:
        model_name = assessment.localization_model

    alert_id = None
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
    if should_alert and persist:
        alert_id = store.create_cleanliness_alert(
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            filename=filename,
            assessment=assessment,
        ).id
    needs_review = (
        assessment.level == CleanlinessLevel.CANNOT_ASSESS
        or assessment.confidence < runtime.interior_review_confidence_threshold
        or should_alert
    )
    if assessment.specialist is not None:
        needs_review = assessment.specialist.human_review_required
    if persist:
        collect_interior_sample(
            content=content,
            filename=filename,
            request_id=job.id,
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            assessment=assessment,
            model_version=model_name,
            human_review_required=needs_review,
            store=store,
        )
    return {
        "assessment": assessment.model_dump(mode="json"),
        "model": model_name,
        "latency_ms": round(latency_ms, 1),
        "alert_id": alert_id,
    }, needs_review


def _damage_comparison_result(
    job: InspectionAnalysisJob,
    return_content: bytes,
    *,
    return_filename: str,
    rental_id: str,
    vehicle_id: str,
    expected_view: ExpectedView,
    return_is_closeup: bool,
    return_session_created_at: str,
    store: OperationsStore,
    storage: InspectionImageStorage,
) -> tuple[dict[str, Any], bool]:
    references = store.find_pickup_reference_photos(
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        slot=expected_view,
        completed_before=return_session_created_at,
    )
    if not references:
        raise InspectionAnalysisContextError("找不到同租單、同車且同視角的借車基準照片")
    candidates = []
    for reference_photo, object_key in references:
        path = storage.resolve(object_key)
        if path is not None:
            candidates.append((reference_photo, path.read_bytes()))
    if not candidates:
        raise InspectionAnalysisContextError("借車基準照片檔案不存在")
    # Comparing a photo with itself proves nothing, so it never becomes a case.
    candidates = [item for item in candidates if not is_same_photo(item[1], return_content)]
    if not candidates:
        raise InspectionAnalysisContextError(SAME_AS_PICKUP_MESSAGE)
    if not return_is_closeup and candidates[0][0].slot == expected_view:
        # Two standard photos of the same angle keep the view-consistency guard.
        pickup_photo, pickup_content = candidates[0]
        comparison = _view_guarded_damage_comparison(
            pickup_content,
            return_content,
            expected_view=expected_view,
        )
    else:
        best, comparison = _best_aligned_damage_pair(
            [content for _, content in candidates], return_content,
        )
        pickup_photo, pickup_content = candidates[best]
    damage_case = store.create_damage_case(
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        expected_view=expected_view,
        pickup_filename=pickup_photo.filename,
        return_filename=return_filename,
        comparison=comparison,
    )
    collect_damage_pair_sample(
        pickup_content=pickup_content,
        return_content=return_content,
        pickup_filename=pickup_photo.filename,
        return_filename=return_filename,
        damage_case=damage_case,
        store=store,
    )
    needs_review = comparison.status != DamageComparisonStatus.NO_CHANGE
    return {
        "damage_case_id": damage_case.id,
        "comparison_status": comparison.status.value,
        "review_decision": damage_case.review_decision.value,
        "risk_score": comparison.risk_score,
        "verification": (
            comparison.verification.model_dump(mode="json") if comparison.verification else None
        ),
        "specialist": (
            comparison.specialist.model_dump(
                mode="json", exclude={"annotated_return_image", "regions"}
            )
            if comparison.specialist
            else None
        ),
        "pickup_photo_id": pickup_photo.id,
        "return_photo_id": job.photo_id,
        "expected_view": expected_view.value,
        "return_is_closeup": return_is_closeup,
        "automated": True,
    }, needs_review


def process_next_inspection_job(
    *,
    store: OperationsStore | None = None,
    storage: InspectionImageStorage | None = None,
) -> InspectionAnalysisJob | None:
    store = store or get_operations_store()
    storage = storage or get_inspection_storage()
    job = store.claim_next_inspection_analysis_job()
    if job is None:
        return None

    attempt_start = monotonic()
    attempt_token = job.started_at.isoformat()
    heartbeat_stop = Event()

    def elapsed_ms() -> float:
        return max(0, (monotonic() - attempt_start) * 1000)

    def heartbeat() -> None:
        while not heartbeat_stop.wait(5):
            if not store.heartbeat_inspection_analysis_job(
                job.id, started_at=attempt_token, elapsed_ms=elapsed_ms(),
            ):
                return

    heartbeat_thread = Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        session = store.get_inspection_session(job.session_id)
        photo = store.get_inspection_photo(job.session_id, job.photo_id)
        object_key = store.get_inspection_photo_object_key(job.session_id, job.photo_id)
        if session is None or photo is None or object_key is None:
            raise InspectionAnalysisContextError("AI 工作找不到對應的租單或私有照片")
        image_path = storage.resolve(object_key)
        if image_path is None:
            raise InspectionAnalysisContextError("AI 工作的私有照片檔案不存在")
        content = image_path.read_bytes()

        if job.task == InspectionAnalysisTask.PLATE_RECOGNITION:
            result, needs_review = _plate_result(
                job,
                content,
                filename=photo.filename,
                rental_id=session.rental_id,
                vehicle_id=session.vehicle_id,
                store=store,
            )
        elif job.task == InspectionAnalysisTask.INTERIOR_CLEANLINESS:
            expected_view = "front" if photo.slot.value.endswith("front") else "rear"
            result, needs_review = _interior_result(
                job,
                content,
                filename=photo.filename,
                rental_id=session.rental_id,
                vehicle_id=session.vehicle_id,
                expected_view=expected_view,
                store=store,
            )
        elif job.task == InspectionAnalysisTask.DAMAGE_COMPARISON:
            return_is_closeup = photo.slot == ExpectedView.DAMAGE_DETAIL
            if return_is_closeup and photo.damage_view is None:
                raise InspectionAnalysisContextError("車損特寫未標示車損角度，無法進行同角度比對")
            result, needs_review = _damage_comparison_result(
                job,
                content,
                return_filename=photo.filename,
                rental_id=session.rental_id,
                vehicle_id=session.vehicle_id,
                expected_view=photo.damage_view or photo.slot,
                return_is_closeup=return_is_closeup,
                return_session_created_at=session.created_at.isoformat(),
                store=store,
                storage=storage,
            )
        else:
            # Existing single-image jobs may remain in the queue after an upgrade.
            # Never run a retired model; retain a clear review result for old jobs.
            result, needs_review = {
                "model_retired": True,
                "manual_review_reason": "單張車損辨識已移除；請使用同角度借還車雙圖比對。",
            }, True
        routing = route_analysis_decision(
            job.task,
            result,
            legacy_needs_review=needs_review,
        )
        result["automation"] = routing.as_result_payload()
        completed = store.complete_inspection_analysis_job(
            job.id, elapsed_ms=elapsed_ms(), expected_started_at=attempt_token,
            result=result,
            needs_review=(routing.effective_route != AutomationRoute.AUTO_CLEAR),
        )
        dispatch_notification_outbox(store=store)
        return completed
    except ModelTimeoutError as exc:
        return store.complete_inspection_analysis_job(
            job.id, elapsed_ms=elapsed_ms(), expected_started_at=attempt_token,
            result={
                "model_uncertain": True,
                "timed_out": True,
                "time_limit_seconds": exc.seconds,
                "manual_review_reason": str(exc),
            },
            needs_review=True,
        )
    except (ModelUnavailableError, InvalidModelResponseError) as exc:
        return store.complete_inspection_analysis_job(
            job.id, elapsed_ms=elapsed_ms(), expected_started_at=attempt_token,
            result={"model_uncertain": True, "manual_review_reason": str(exc)},
            needs_review=True,
        )
    except (
        InspectionAnalysisContextError,
        InvalidInspectionObjectKey,
        OSError,
    ) as exc:
        return store.fail_inspection_analysis_job(
            job.id, error_message=str(exc), elapsed_ms=elapsed_ms(), expected_started_at=attempt_token,
        )
    except Exception as exc:  # Keep the daemon alive and preserve an auditable error.
        return store.fail_inspection_analysis_job(
            job.id,
            error_message=f"{type(exc).__name__}: {exc}",
            elapsed_ms=elapsed_ms(), expected_started_at=attempt_token,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=6)
