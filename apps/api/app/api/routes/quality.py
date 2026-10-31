import logging
from collections import OrderedDict
from hashlib import sha256
from threading import Lock
from time import monotonic, perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.plate import PlatePrediction
from app.schemas.quality import (
    BlurAssessment,
    CheckStatus,
    ExpectedView,
    QualityResponse,
    QualityStatus,
)
from app.services.active_learning import collect_quality_samples
from app.services.auth import require_user
from app.services.blur_classifier import get_blur_classifier
from app.services.cleanliness import InvalidModelResponseError
from app.services.image_quality import InvalidImageError, Thresholds, analyze_image
from app.services.interior_capture import get_interior_capture_classifier
from app.services.license_plate import normalize_plate
from app.services.model_time_limits import (
    ModelTimeoutError,
    RecognitionModel,
    run_with_time_limit,
)
from app.services.plate_capture import (
    PlateCaptureAssessment,
    assess_plate_capture,
    complete_plate_visible,
    is_complete_plate_text,
    ordered_plate_match,
    plate_inside_vehicle,
    plate_touches_edge,
    shared_plate_run,
)
from app.services.rapid_plate_recognition import (
    get_plate_recognizer,
    timed_out_plate_prediction,
)
from app.services.rental_context import get_principal_rental_identity
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.upload_validation import read_validated_image
from app.services.vehicle_coverage import (
    edge_margins_from_box,
    get_vehicle_coverage_detector,
    touched_edges,
)
from app.services.view_classifier import get_view_classifier

router = APIRouter(prefix="/quality", tags=["capture quality"])
logger = logging.getLogger(__name__)

EXTERIOR_VIEWS = {
    ExpectedView.LEFT_FRONT,
    ExpectedView.RIGHT_FRONT,
    ExpectedView.LEFT_REAR,
    ExpectedView.RIGHT_REAR,
}
VIEW_LABELS = {
    ExpectedView.LEFT_FRONT: "左前方",
    ExpectedView.RIGHT_FRONT: "右前方",
    ExpectedView.LEFT_REAR: "左後方",
    ExpectedView.RIGHT_REAR: "右後方",
    ExpectedView.INTERIOR_FRONT: "前座車內",
    ExpectedView.INTERIOR_REAR: "後座車內",
    ExpectedView.DAMAGE_DETAIL: "車損特寫",
}

SCENE_LABELS = {"interior_front": "前座車內", "interior_rear": "後座車內"}

DAMAGE_DETAIL_THRESHOLDS = Thresholds(
    min_short_edge=360,
    min_blur_fail=25.0,
    min_blur_warn=60.0,
    brightness_low_fail=18.0,
    brightness_low_warn=38.0,
    brightness_high_warn=225.0,
    brightness_high_fail=248.0,
    min_dynamic_range=24.0,
    max_clipped_ratio=0.55,
)

# The formal flow checks a photo and later saves the same bytes. Reusing the
# accepted verdict keeps both calls identical, so an OCR or model timeout on
# the second run cannot turn an accepted photo into a different result.
ACCEPTED_VERDICT_TTL_SECONDS = 30 * 60
ACCEPTED_VERDICT_LIMIT = 256
_accepted_verdicts: OrderedDict[tuple[str, str, str], tuple[float, QualityResponse]] = OrderedDict()
_accepted_verdicts_lock = Lock()
_verified_plates: OrderedDict[str, tuple[float, str]] = OrderedDict()


def clear_accepted_verdicts() -> None:
    with _accepted_verdicts_lock:
        _accepted_verdicts.clear()
        _verified_plates.clear()


def _verified_plate(scope: str | None) -> str | None:
    if not scope:
        return None
    with _accepted_verdicts_lock:
        entry = _verified_plates.get(scope)
        if entry is None:
            return None
        if monotonic() - entry[0] > ACCEPTED_VERDICT_TTL_SECONDS:
            del _verified_plates[scope]
            return None
        return entry[1]


def _remember_plate(scope: str | None, result: QualityResponse) -> None:
    prediction = result.plate_prediction
    if (not scope or result.status == QualityStatus.FAIL or prediction is None
            or prediction.fast_matched or not prediction.is_confident or prediction.model_uncertain
            or not is_complete_plate_text(prediction.normalized_plate)):
        return
    with _accepted_verdicts_lock:
        _verified_plates[scope] = (monotonic(), prediction.normalized_plate)
        _verified_plates.move_to_end(scope)
        while len(_verified_plates) > ACCEPTED_VERDICT_LIMIT:
            _verified_plates.popitem(last=False)


def _cached_verdict(key: tuple[str, str, str]) -> QualityResponse | None:
    with _accepted_verdicts_lock:
        entry = _accepted_verdicts.get(key)
        if entry is None:
            return None
        stored_at, verdict = entry
        if monotonic() - stored_at > ACCEPTED_VERDICT_TTL_SECONDS:
            del _accepted_verdicts[key]
            return None
        _accepted_verdicts.move_to_end(key)
        return verdict


def _remember_verdict(key: tuple[str, str, str], result: QualityResponse) -> None:
    if result.status == QualityStatus.FAIL:
        return  # A blocked photo is always evaluated again on retry.
    with _accepted_verdicts_lock:
        _accepted_verdicts[key] = (monotonic(), result.model_copy(deep=True))
        _accepted_verdicts.move_to_end(key)
        while len(_accepted_verdicts) > ACCEPTED_VERDICT_LIMIT:
            _accepted_verdicts.popitem(last=False)


def evaluate_quality_capture(
    *,
    content: bytes,
    filename: str,
    expected_view: ExpectedView | None,
    rental_id: str | None = None,
    vehicle_id: str | None = None,
    request_id: str | None = None,
    collect_samples: bool = True,
    plate_scope: str | None = None,
    rental_plate: str | None = None,
) -> QualityResponse:
    """Evaluate one verified upload and attach a server-issued request ID.

    Each corner must contain plate evidence. The plate set on the rental order
    is the reference from the first photo, so a neighbouring car's plate never
    passes; without it, the first full read in this flow becomes the reference.
    Four matching characters can then confirm a photo without crop retries.
    """
    runtime = get_runtime_recognition_settings()
    order_plate = normalize_plate(rental_plate)
    if not is_complete_plate_text(order_plate):
        order_plate = None
    verified = order_plate or _verified_plate(plate_scope)
    image_sha256 = sha256(content).hexdigest()
    verdict_key = (
        image_sha256,
        expected_view.value if expected_view else "",
        sha256(
            (runtime.model_dump_json() + (plate_scope or "") + (order_plate or "")).encode()
        ).hexdigest(),
    )
    cached = _cached_verdict(verdict_key)
    if cached is not None:
        result = cached.model_copy(
            deep=True, update={"request_id": request_id or str(uuid4()), "filename": filename}
        )
        _remember_plate(plate_scope, result)
        _finish_capture(
            result, content, filename, image_sha256, expected_view,
            rental_id=rental_id, vehicle_id=vehicle_id, collect_samples=collect_samples,
        )
        return result
    standard_thresholds = Thresholds(
        min_short_edge=runtime.quality_min_short_edge,
        min_blur_fail=runtime.quality_blur_fail,
        min_blur_warn=runtime.quality_blur_warning,
        brightness_low_fail=runtime.quality_brightness_low_fail,
        brightness_low_warn=runtime.quality_brightness_low_warning,
        brightness_high_warn=runtime.quality_brightness_high_warning,
        brightness_high_fail=runtime.quality_brightness_high_fail,
        min_dynamic_range=runtime.quality_min_dynamic_range,
        max_clipped_ratio=runtime.quality_max_clipped_ratio,
    )
    result = analyze_image(
        content=content,
        filename=filename,
        request_id=request_id or str(uuid4()),
        expected_view=expected_view,
        thresholds=(
            DAMAGE_DETAIL_THRESHOLDS
            if expected_view == ExpectedView.DAMAGE_DETAIL
            else standard_thresholds
        ),
    )
    if expected_view == ExpectedView.DAMAGE_DETAIL:
        _apply_damage_detail_mode(result)
    else:
        _apply_plate_capture_hint(result, content)
        if result.status != QualityStatus.FAIL:
            _apply_view_prediction(result, content)
        if result.status != QualityStatus.FAIL:
            _apply_exterior_scene_prediction(result, content)
        if result.status != QualityStatus.FAIL:
            _apply_interior_capture_prediction(result, content)
        if result.status != QualityStatus.FAIL:
            _apply_coverage_prediction(result, content)
            _apply_blur_prediction(result, content)
            # Coverage failure is not evidence that OCR failed. Read the plate
            # independently so a framing issue does not hide a readable plate.
            _apply_required_plate_prediction(
                result, content, verified_plate=verified, from_rental=order_plate is not None,
            )
    _remember_verdict(verdict_key, result)
    _remember_plate(plate_scope, result)
    _finish_capture(
        result, content, filename, image_sha256, expected_view,
        rental_id=rental_id, vehicle_id=vehicle_id, collect_samples=collect_samples,
    )
    return result


def _finish_capture(
    result: QualityResponse,
    content: bytes,
    filename: str,
    image_sha256: str,
    expected_view: ExpectedView | None,
    *,
    rental_id: str | None,
    vehicle_id: str | None,
    collect_samples: bool,
) -> None:
    if collect_samples:
        collect_quality_samples(
            content=content,
            filename=filename,
            result=result,
            rental_id=rental_id,
            vehicle_id=vehicle_id,
        )
    logger.info(
        "capture_quality request_id=%s image_sha256=%s view=%s size=%sx%s status=%s "
        "failed_checks=%s coverage_box=%s plate_detected=%s plate_timed_out=%s",
        result.request_id, image_sha256,
        expected_view.value if expected_view else None,
        result.metrics.width, result.metrics.height, result.status.value,
        [check.key for check in result.checks if check.status == CheckStatus.FAIL],
        result.coverage_prediction.bounding_box if result.coverage_prediction else None,
        result.plate_prediction.plate_detected if result.plate_prediction else None,
        result.plate_prediction.timed_out if result.plate_prediction else None,
    )


def _mark_model_timeout(
    result: QualityResponse, check_key: str, error: ModelTimeoutError
) -> None:
    """A slow model never blocks the upload; the check is flagged instead."""
    logger.warning("model_time_limit_exceeded model=%s seconds=%s", error.model.value, error.seconds)
    check = next(item for item in result.checks if item.key == check_key)
    check.status = CheckStatus.WARNING
    check.measured_value = f"逾時（超過 {error.seconds:g} 秒）"
    check.message = f"{error}此項暫不判斷，不影響上傳。"
    if result.status == QualityStatus.PASS:
        result.status = QualityStatus.WARNING
        result.summary = "部分 AI 檢查逾時，照片仍可上傳，後台會再人工確認。"


def _fail_required_check(
    result: QualityResponse, check_key: str, message: str, summary: str,
    *, score_cap: int | None = None,
) -> None:
    check = next(item for item in result.checks if item.key == check_key)
    check.status = CheckStatus.FAIL
    check.message = message
    result.score = min(result.score, score_cap) if score_cap is not None else max(0, result.score - 30)
    result.status = QualityStatus.FAIL
    result.retake_required = True
    result.summary = summary


def _apply_damage_detail_mode(result: QualityResponse) -> None:
    messages = {
        "vehicle_view": "車損特寫不要求完整四角視角，請讓受損部位與周邊板件一起入鏡。",
        "plate_presence": "車損特寫不強制車牌入鏡；案件由租單與車輛 ID 配對。",
        "vehicle_coverage": "車損特寫允許局部車身，不套用完整車身覆蓋率門檻。",
    }
    for check in result.checks:
        if check.key in messages:
            check.status = CheckStatus.PASS
            check.message = messages[check.key]
            check.measured_value = "特寫模式"
            check.threshold = "不適用完整車身門檻"
    result.model_checks_ready = True
    if result.status == QualityStatus.PASS:
        result.summary = "車損特寫品質通過，可以進行可見損傷判讀。"


def _apply_view_prediction(result: QualityResponse, content: bytes) -> None:
    if result.expected_view not in EXTERIOR_VIEWS:
        return
    classifier = get_view_classifier()
    if not classifier.available:
        _fail_required_check(
            result, "vehicle_view", "角度辨識模型無法使用，請聯絡客服。",
            "尚未能確認車外角度，需要重新檢查。",
        )
        return

    try:
        prediction = run_with_time_limit(
            RecognitionModel.VIEW, classifier.predict, content, result.expected_view
        )
    except ModelTimeoutError:
        _fail_required_check(
            result, "vehicle_view", "角度辨識逾時，請重拍。",
            "尚未能確認車外角度，需要重新檢查。",
        )
        return
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("exterior_view_model_unavailable reason=%s", exc)
        _fail_required_check(
            result, "vehicle_view", "角度辨識模型無法使用，請聯絡客服。",
            "尚未能確認車外角度，需要重新檢查。",
        )
        return
    configured_threshold = (
        get_runtime_recognition_settings().view_confidence_threshold
    )
    prediction = prediction.model_copy(
        update={
            "confidence_threshold": configured_threshold,
            "is_confident": prediction.confidence >= configured_threshold,
        }
    )
    result.view_prediction = prediction
    result.model_checks_ready = True
    view_check = next(check for check in result.checks if check.key == "vehicle_view")
    predicted_label = VIEW_LABELS[prediction.predicted_view]
    expected_label = VIEW_LABELS[result.expected_view]
    confidence_percent = prediction.confidence * 100
    view_check.measured_value = f"{predicted_label} ({confidence_percent:.1f}%)"
    view_check.threshold = f"信心門檻 {prediction.confidence_threshold:.0%}"

    if not prediction.is_confident:
        view_check.status = CheckStatus.FAIL
        view_check.message = f"無法辨認拍攝角度，請對照{expected_label}範本重拍。"
        result.score = max(0, result.score - 30)
        result.status = QualityStatus.FAIL
        result.retake_required = True
        result.summary = "無法確認車外拍攝角度，照片不得用 90 分警告狀態放行。"
        return

    if prediction.matches_expected:
        view_check.status = CheckStatus.PASS
        view_check.message = f"拍攝位置符合要求：{expected_label}。"
        return

    view_check.status = CheckStatus.FAIL
    view_check.message = f"拍到的是{predicted_label}，此欄位需要{expected_label}，請重拍。"
    result.score = max(0, result.score - 25)
    result.status = QualityStatus.FAIL
    result.retake_required = True
    result.summary = "拍攝角度不符合指定槽位，請依左前、右前、左後、右後精確重拍。"


def _apply_exterior_scene_prediction(result: QualityResponse, content: bytes) -> None:
    if result.expected_view not in EXTERIOR_VIEWS:
        return
    classifier = get_interior_capture_classifier()
    if not classifier.exterior_veto_available:
        _fail_required_check(
            result, "vehicle_view", "車內外辨識模型無法使用，請聯絡客服。",
            "尚未能確認照片是車外場景，需要重新檢查。",
        )
        return
    try:
        prediction = run_with_time_limit(
            RecognitionModel.INTERIOR_CAPTURE,
            classifier.predict_exterior_scene,
            content,
        )
    except ModelTimeoutError:
        _fail_required_check(
            result, "vehicle_view", "車內外辨識逾時，請重拍。",
            "尚未能確認照片是車外場景，需要重新檢查。",
        )
        return
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("exterior_scene_model_unavailable reason=%s", exc)
        _fail_required_check(
            result, "vehicle_view", "車內外辨識模型無法使用，請聯絡客服。",
            "尚未能確認照片是車外場景，需要重新檢查。",
        )
        return
    result.exterior_scene_prediction = prediction
    if not prediction.is_interior:
        return

    view_check = next(check for check in result.checks if check.key == "vehicle_view")
    view_check.status = CheckStatus.FAIL
    view_check.measured_value = f"偵測到車內場景 ({prediction.interior_probability:.1%})"
    view_check.threshold = f"車內否決門檻 {prediction.veto_threshold:.0%}"
    view_check.message = (
        f"拍到的是車內，此欄位需要{VIEW_LABELS[result.expected_view]}車外照，請重拍。"
    )
    result.score = max(0, result.score - 35)
    result.status = QualityStatus.FAIL
    result.retake_required = True
    result.summary = "車外四角欄位偵測到車內照片，需要重拍。"


def _apply_plate_capture_hint(
    result: QualityResponse, content: bytes
) -> PlateCaptureAssessment | None:
    if result.expected_view not in EXTERIOR_VIEWS:
        return None
    check = next(item for item in result.checks if item.key == "plate_presence")
    check.label = "完整車牌"
    runtime = get_runtime_recognition_settings()
    check.threshold = (
        f"需辨識完整車牌；候選區至少 {runtime.plate_hint_min_width_px}×"
        f"{runtime.plate_hint_min_height_px}px"
    )
    try:
        assessment = assess_plate_capture(content)
    except InvalidModelResponseError:
        check.status = CheckStatus.PENDING_MODEL
        check.measured_value = "無法解析候選區"
        check.message = "車牌候選區暫時無法解析，將由完整車牌 OCR 檢查。"
        return None

    if not assessment.plate_visible:
        check.status = CheckStatus.PENDING_MODEL
        check.measured_value = "未找到明顯車牌候選區"
        check.message = "尚未找到車牌候選區，將由完整車牌 OCR 檢查。"
        return assessment

    check.measured_value = (
        f"候選區 {assessment.width_pixels}×{assessment.height_pixels}px；"
        f"清晰度 {assessment.sharpness:.1f}"
    )
    if assessment.likely_readable:
        check.status = CheckStatus.PENDING_MODEL
        check.message = "找到車牌候選區，仍需讀出完整車牌才能通過。"
    else:
        check.status = CheckStatus.PENDING_MODEL
        check.message = "車牌候選區偏小或偏模糊，仍需通過完整車牌 OCR。"
    return assessment


def _plate_failure_reason(
    prediction: PlatePrediction,
    width: int,
    height: int,
    vehicle_box: list[float] | None,
) -> str:
    """One short reason taken only from what OCR actually returned.

    The OpenCV plate hint is never quoted: it often boxes a taillight or a
    logo, so its size says nothing about whether the plate is complete.
    """
    if prediction.timed_out:
        return "車牌辨識逾時，請讓整張車牌清楚入鏡後重拍。"
    if not prediction.plate_detected or prediction.bounding_box is None:
        return "讀不到完整車牌，請讓整張車牌入鏡後重拍。"
    if not prediction.is_confident:
        return "模型尚未可靠辨識車牌，請核對照片後重拍。"
    # Only a plate that could not be read completely may be called cut off.
    if not is_complete_plate_text(prediction.normalized_plate) and plate_touches_edge(
        prediction.bounding_box, width, height
    ):
        return "車牌被畫面邊緣切到，請讓整張車牌入鏡後重拍。"
    if not is_complete_plate_text(prediction.normalized_plate):
        return "模型讀出的車牌格式不完整，請核對照片後重拍。"
    return "車牌位置資料異常，請重新辨識或重拍。"


RENTAL_PLATE_MIN_SHARED = 4
# The rental order is authoritative, so an ordered match against it on the
# car may pass below the usual OCR threshold (RD-O27 read at 0.62).
RENTAL_PLATE_RELAXED_CONFIDENCE = 0.6
# The renter only needs to know the plate was found; details stay in measured_value.
PLATE_RECOGNIZED_MESSAGE = "已辨識到車牌。"


def _apply_required_plate_prediction(
    result: QualityResponse, content: bytes,
    *, verified_plate: str | None = None, from_rental: bool = False,
) -> None:
    """Require a full read or four characters matching the reference plate.

    The reference is the rental order's plate when it is set, otherwise the
    plate already read in this flow.
    """
    if result.expected_view not in EXTERIOR_VIEWS:
        return
    vehicle_box = (
        result.coverage_prediction.bounding_box or None
        if result.coverage_prediction is not None else None
    )
    try:
        recognize = get_plate_recognizer().recognize
        if verified_plate:
            from functools import partial

            recognize = partial(recognize, verified_plate=verified_plate, vehicle_box=vehicle_box)
        prediction = run_with_time_limit(
            RecognitionModel.PLATE, recognize, content
        )
    except ModelTimeoutError as exc:
        prediction = timed_out_plate_prediction(exc.seconds)
    result.plate_prediction = prediction
    result.model_checks_ready = True
    check = next(item for item in result.checks if item.key == "plate_presence")
    check.label = "完整車牌"
    reference = "租單車牌" if from_rental else "本次已辨識車牌"
    check.threshold = (
        f"與{reference} {verified_plate} 相同的完整車牌可讀，"
        "或位於車身內且至少 4 個連續字元相符；信心需達門檻"
        + (
            f"；或英文字頭相同且依序至少 4 字相符、位於車身內，信心達 "
            f"{RENTAL_PLATE_RELAXED_CONFIDENCE:.0%}"
            if from_rental else ""
        )
        if verified_plate else "完整車牌字元可讀且信心達門檻"
    )
    # No detected car yields an empty box; treat it the same as no box.
    vehicle_box = (
        result.coverage_prediction.bounding_box or None
        if result.coverage_prediction is not None
        else None
    )
    consecutive = (
        shared_plate_run(prediction.normalized_plate, verified_plate)
        if verified_plate and prediction.normalized_plate else ""
    )
    if not is_complete_plate_text(prediction.normalized_plate):
        consecutive = ""
    ordered = (
        ordered_plate_match(prediction.normalized_plate, verified_plate)
        if from_rental and verified_plate and prediction.normalized_plate else ""
    )
    shared = consecutive if len(consecutive) >= RENTAL_PLATE_MIN_SHARED else ordered
    confident_match = bool(
        shared and prediction.is_confident and prediction.normalized_plate != verified_plate
    )
    relaxed_match = bool(
        ordered and not prediction.is_confident and not prediction.timed_out
        and prediction.confidence >= RENTAL_PLATE_RELAXED_CONFIDENCE
        and prediction.bounding_box is not None
        and plate_inside_vehicle(prediction.bounding_box, vehicle_box)
    )
    if from_rental and (confident_match or relaxed_match):
        # The rental order is authoritative; OCR often swaps one character
        # (visually similar letters) or drops some characters, so four
        # consecutive characters, or the same leading letters plus four in
        # order, confirm the car; the ordered match on the car body may pass
        # at a lower confidence. plate_text keeps the raw OCR read.
        result.plate_prediction = prediction.model_copy(update={
            "normalized_plate": verified_plate, "expected_plate": verified_plate,
            "matches_expected": True, "matched_characters": shared,
        })
        check.status = CheckStatus.PASS
        check.measured_value = (
            f"本張 {prediction.normalized_plate}；租單車牌 {verified_plate}；相符 {shared}"
            + (f"（信心 {prediction.confidence:.1%}）" if relaxed_match else "")
        )
        check.message = PLATE_RECOGNIZED_MESSAGE
        return
    if (verified_plate and prediction.is_confident
            and is_complete_plate_text(prediction.normalized_plate)
            and prediction.normalized_plate != verified_plate):
        check.measured_value = f"本張 {prediction.normalized_plate}；{reference} {verified_plate}"
        _fail_required_check(
            result, "plate_presence", f"本張車牌與{reference}不同，請核對車輛後重拍。",
            f"本張車牌與{reference}不同。",
        )
        return
    if complete_plate_visible(
        prediction, result.metrics.width, result.metrics.height, vehicle_box
    ):
        check.status = CheckStatus.PASS
        check.measured_value = (
            f"快速核對 {prediction.matched_characters}（{prediction.confidence:.1%}）"
            if prediction.fast_matched else f"完整車牌可讀（{prediction.confidence:.1%}）"
        )
        check.message = PLATE_RECOGNIZED_MESSAGE
        return
    check.measured_value = (
        "車牌辨識逾時" if prediction.timed_out else
        "無法讀出完整車牌" if not prediction.plate_detected else
        f"辨識信心 {prediction.confidence:.1%} 或車牌不完整"
    )
    _fail_required_check(
        result, "plate_presence",
        _plate_failure_reason(prediction, result.metrics.width, result.metrics.height, vehicle_box),
        "車牌辨識逾時，需要重新檢查。" if prediction.timed_out else "未能確認完整車牌，需要重拍。",
        score_cap=75,
    )


def _apply_interior_capture_prediction(result: QualityResponse, content: bytes) -> None:
    if result.expected_view not in {
        ExpectedView.INTERIOR_FRONT,
        ExpectedView.INTERIOR_REAR,
    }:
        return
    classifier = get_interior_capture_classifier()
    if not classifier.available:
        _fail_required_check(
            result, "vehicle_view", "座位辨識模型無法使用，請聯絡客服。",
            "尚未能確認前／後座，照片需要重新檢查。",
        )
        return
    try:
        prediction = run_with_time_limit(
            RecognitionModel.INTERIOR_CAPTURE,
            classifier.predict,
            content,
            result.expected_view,
        )
    except ModelTimeoutError:
        _fail_required_check(
            result, "vehicle_view", "座位辨識逾時，請重拍。",
            "尚未能確認前／後座，照片需要重新檢查。",
        )
        return
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("interior_capture_model_unavailable reason=%s", exc)
        _fail_required_check(
            result, "vehicle_view", "座位辨識模型無法使用，請聯絡客服。",
            "尚未能確認前／後座，照片需要重新檢查。",
        )
        return
    result.interior_capture_prediction = prediction
    result.model_checks_ready = True
    view_check = next(check for check in result.checks if check.key == "vehicle_view")
    view_check.label = "車內場景與方向"
    coverage_check = next(check for check in result.checks if check.key == "vehicle_coverage")
    coverage_check.status = CheckStatus.PASS
    coverage_check.measured_value = "車內模式"
    coverage_check.threshold = "不套用車外車身覆蓋率"
    coverage_check.message = "車內照片不要求完整車身入鏡。"
    rotation_text = (
        f"；旋轉機率 {prediction.rotation_probability:.1%}"
        if prediction.rotation_probability is not None
        else ""
    )
    view_check.measured_value = (
        f"{prediction.scene_label} ({prediction.scene_confidence:.1%}){rotation_text}"
    )
    view_check.threshold = (
        f"座位信心門檻 {prediction.scene_confidence_threshold:.0%}"
        + (
            f"；專家門檻 {prediction.specialist_threshold:.0%}"
            if prediction.specialist_threshold is not None
            else ""
        )
        + (
            f"；旋轉門檻 {prediction.rotation_threshold:.0%}"
            if prediction.rotation_threshold is not None
            else ""
        )
    )

    if prediction.rotation_vetoed:
        view_check.status = CheckStatus.FAIL
        view_check.message = "照片方向歪斜，請轉正手機重拍。"
        result.score = max(0, result.score - 28)
        result.status = QualityStatus.FAIL
        result.retake_required = True
        result.summary = "車內照片方向錯誤，需要轉正後重拍。"
        return
    if prediction.matches_expected and prediction.scene_confident:
        view_check.status = CheckStatus.PASS
        view_check.message = "前／後座模型與座位專家均確認符合此拍攝欄位。"
        return
    expected_label = VIEW_LABELS[result.expected_view]
    seen = SCENE_LABELS.get(prediction.scene_label, "車外")
    _fail_required_check(
        result, "vehicle_view",
        f"拍到的是{seen}，此欄位需要{expected_label}，請重拍。"
        if not prediction.matches_expected
        else f"無法辨認為{expected_label}，請對照範本重拍。",
        "前／後座未通過確認，需要重新拍攝。",
    )


def _apply_coverage_prediction(result: QualityResponse, content: bytes) -> None:
    if result.expected_view not in EXTERIOR_VIEWS:
        return
    runtime = get_runtime_recognition_settings()
    try:
        prediction = run_with_time_limit(
            RecognitionModel.VEHICLE_DETECTION,
            get_vehicle_coverage_detector().predict,
            content,
        )
    except ModelTimeoutError as exc:
        _mark_model_timeout(result, "vehicle_coverage", exc)
        return
    result.coverage_prediction = prediction
    coverage_check = next(check for check in result.checks if check.key == "vehicle_coverage")
    coverage_check.status = prediction.status
    coverage_check.measured_value = f"車身佔畫面 {prediction.area_ratio:.0%}"
    coverage_check.threshold = (
        f"建議 {runtime.vehicle_coverage_low_warning:.0%}–"
        f"{runtime.vehicle_coverage_high_warning:.0%}；車身同時貼到兩側以上邊界才需重拍"
    )

    area = f"{prediction.area_ratio:.0%}"
    if not prediction.car_detected:
        coverage_check.message = "找不到車輛，請對準車身重拍。"
    elif prediction.area_ratio < runtime.vehicle_coverage_low_warning:
        coverage_check.message = f"車身太小（佔畫面 {area}），請靠近重拍。"
    elif prediction.status == CheckStatus.PASS:
        coverage_check.message = "車身距離與畫面覆蓋率合適。"
    elif prediction.area_ratio > runtime.vehicle_coverage_high_warning:
        coverage_check.message = f"車身太大（佔畫面 {area}），請後退重拍。"
    else:
        edges = touched_edges(
            edge_margins_from_box(prediction.bounding_box),
            runtime.vehicle_edge_margin_warning,
        )
        coverage_check.message = f"車身{'與'.join(edges) or '邊緣'}被切到，請後退重拍。"

    if prediction.status in {CheckStatus.FAIL, CheckStatus.WARNING}:
        coverage_check.status = CheckStatus.FAIL
        result.score = min(result.score, 75)
        result.status = QualityStatus.FAIL
        result.retake_required = True
        result.summary = "車身距離或構圖不符合辨識需求，需要重拍。"


@router.post("/check", response_model=QualityResponse)
async def check_quality(
    image: Annotated[UploadFile, File()],
    principal: Annotated[AuthPrincipal, Depends(require_user)],
    expected_view: Annotated[ExpectedView | None, Form()] = None,
    capture_flow_id: Annotated[str | None, Form(max_length=128)] = None,
) -> QualityResponse:
    try:
        rental_id, vehicle_id = get_principal_rental_identity(principal)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    started = perf_counter()
    settings = get_settings()
    content = await read_validated_image(image, settings.max_upload_mb)

    try:
        result = await run_in_threadpool(
            evaluate_quality_capture,
            content=content,
            filename=image.filename or "capture.jpg",
            expected_view=expected_view,
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            plate_scope=(
                f"quality:{principal.subject}:{rental_id}:{vehicle_id}:{capture_flow_id}" if capture_flow_id else None
            ),
            rental_plate=principal.license_plate,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result.recognition_ms = round((perf_counter() - started) * 1000, 1)
    return result


def _apply_blur_prediction(result: QualityResponse, content: bytes) -> None:
    """Catch hand-shake blur that the global sharpness score misses at night.

    Street lights, parking lines and sensor noise keep the Laplacian score high
    while the car itself is smeared, so the model looks inside the vehicle box.
    It only ever tightens the verdict; an unavailable model never blocks upload.
    """
    if result.expected_view not in EXTERIOR_VIEWS:
        return
    runtime = get_runtime_recognition_settings()
    if not runtime.blur_model_enabled:
        return
    classifier = get_blur_classifier()
    if not classifier.available:
        logger.warning("blur_model_unavailable path=%s", classifier.model_path)
        return
    coverage = result.coverage_prediction
    vehicle_box = coverage.bounding_box if coverage is not None and coverage.car_detected else None
    try:
        prediction = classifier.predict(content, vehicle_box)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("blur_model_failed reason=%s", exc)
        return
    result.blur_assessment = BlurAssessment(
        probability=prediction.probability,
        fail_threshold=runtime.blur_model_fail_probability,
        warning_threshold=runtime.blur_model_warning_probability,
        latency_ms=prediction.latency_ms,
        model_version=prediction.model_version,
    )
    check = next(item for item in result.checks if item.key == "sharpness")
    probability = prediction.probability
    if probability >= runtime.blur_model_fail_probability:
        check.measured_value = f"手晃模糊機率 {probability:.0%}"
        check.threshold = f"手晃模糊機率需低於 {runtime.blur_model_fail_probability:.0%}"
        if check.status == CheckStatus.FAIL:
            return
        _fail_required_check(
            result, "sharpness", "車身或車牌因手晃而模糊，請停穩手機、對焦後重拍。",
            "照片手晃模糊，需要重拍。", score_cap=72,
        )
        return
    if probability >= runtime.blur_model_warning_probability and check.status == CheckStatus.PASS:
        check.status = CheckStatus.WARNING
        check.measured_value = f"手晃模糊機率 {probability:.0%}"
        check.threshold = f"建議低於 {runtime.blur_model_warning_probability:.0%}"
        check.message = "照片可能有點手晃，建議停穩手機後重拍。"
        result.score = max(0, result.score - 10)
        if result.status == QualityStatus.PASS:
            result.status = QualityStatus.WARNING
            result.summary = "照片可辨識但可能有點手晃，建議重拍以降低誤判。"
