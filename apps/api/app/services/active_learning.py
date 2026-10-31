from __future__ import annotations

import logging
import sqlite3

from app.schemas.cleanliness import CleanlinessAssessment
from app.schemas.damage import (
    DamageCase,
    DamageComparisonStatus,
    DamageReviewDecision,
)
from app.schemas.operations import LearningSample, LearningTask
from app.schemas.plate import PlatePrediction
from app.schemas.quality import CheckStatus, ExpectedView, QualityResponse
from app.services.operations_store import OperationsStore, get_operations_store
from app.services.runtime_settings import get_runtime_recognition_settings
from app.services.vehicle_coverage import edge_margins_from_box, touched_edges

logger = logging.getLogger(__name__)

EXTERIOR_LABELS = {"left_front", "right_front", "left_rear", "right_rear"}
ALLOWED_CORRECTED_LABELS: dict[LearningTask, set[str]] = {
    LearningTask.EXTERIOR_VIEW: EXTERIOR_LABELS | {"unusable"},
    LearningTask.VEHICLE_COVERAGE: {
        "acceptable",
        "too_far",
        "too_close",
        "not_vehicle",
        "unusable",
    },
    LearningTask.INTERIOR_CAPTURE: {
        "valid_front",
        "valid_rear",
        "wrong_scene",
        "blurred",
        "rotated",
        "unusable_other",
    },
    LearningTask.INTERIOR_CLEANLINESS: {
        "clean",
        "ordinary",
        "dirty",
        "cannot_assess",
    },
    LearningTask.VISIBLE_DAMAGE: {"damage", "no_damage", "cannot_assess"},
    LearningTask.DAMAGE_PAIR: {
        "confirmed_new_damage",
        "no_new_damage",
        "cannot_assess",
    },
    LearningTask.LICENSE_PLATE: set(),
    LearningTask.CAPTURE_SHARPNESS: {"sharp", "blurred", "unusable"},
}


def _safe_collect(store: OperationsStore, **kwargs: object) -> LearningSample | None:
    try:
        return store.collect_learning_sample(**kwargs)  # type: ignore[arg-type]
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.warning("Unable to collect active-learning sample: %s", exc)
        return None


def collect_quality_samples(
    *,
    content: bytes,
    filename: str,
    result: QualityResponse,
    rental_id: str | None = None,
    vehicle_id: str | None = None,
    store: OperationsStore | None = None,
) -> list[str]:
    settings = get_runtime_recognition_settings()
    if not settings.active_learning_enabled:
        return []
    store = store or get_operations_store()
    shared_reasons: list[str] = []
    if result.score < settings.active_learning_score_threshold:
        shared_reasons.append(f"score_below_{settings.active_learning_score_threshold}")
    if result.retake_required:
        shared_reasons.append("retake_required")
    if result.status.value == "fail":
        shared_reasons.append("quality_failed")

    created: list[str] = []
    prediction = result.view_prediction
    if result.expected_view in {
        ExpectedView.INTERIOR_FRONT,
        ExpectedView.INTERIOR_REAR,
    }:
        expected_label = (
            "valid_front"
            if result.expected_view == ExpectedView.INTERIOR_FRONT
            else "valid_rear"
        )
        interior_reasons = list(shared_reasons)
        if result.status.value == "warning":
            interior_reasons.append("quality_warning")
        sample_kind = "hard_case"
        predicted_label = "unusable_other"
        interior_prediction = result.interior_capture_prediction
        sharpness_flagged = any(
            check.key == "sharpness" and check.status != CheckStatus.PASS
            for check in result.checks
        )
        if sharpness_flagged:
            predicted_label = "blurred"
        elif interior_prediction is not None and interior_prediction.rotation_vetoed:
            predicted_label = "rotated"
        elif (
            interior_prediction is not None
            and interior_prediction.matches_expected is False
        ):
            predicted_label = "wrong_scene"
        if interior_prediction is not None:
            if interior_prediction.scene_confident is False:
                interior_reasons.append("interior_scene_low_confidence")
            if interior_prediction.matches_expected is False:
                interior_reasons.append("interior_scene_mismatch")
            if interior_prediction.rotation_vetoed:
                interior_reasons.append("interior_rotation_veto")
        if (
            not interior_reasons
            and settings.active_learning_collect_good_samples
            and result.status.value == "pass"
            and result.score >= settings.active_learning_min_training_score
        ):
            interior_reasons.append("interior_reference_candidate")
            sample_kind = "good_reference"
            predicted_label = expected_label
        if interior_reasons:
            sample = _safe_collect(
                store,
                task=LearningTask.INTERIOR_CAPTURE,
                primary_content=content,
                primary_filename=filename,
                trigger_reasons=interior_reasons,
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                source_id=result.request_id,
                expected_label=expected_label,
                predicted_label=predicted_label,
                confidence=(
                    interior_prediction.scene_confidence
                    if interior_prediction is not None
                    else result.score / 100
                ),
                score=result.score,
                model_version=(
                    interior_prediction.model_version
                    if interior_prediction is not None
                    else None
                ),
                metadata=result.model_dump(mode="json"),
                sample_kind=sample_kind,
                training_eligible=False,
            )
            if sample is not None:
                created.append(sample.id)
    if result.expected_view and result.expected_view.value in EXTERIOR_LABELS:
        view_reasons = list(shared_reasons)
        if prediction is not None:
            if prediction.confidence < settings.active_learning_confidence_threshold:
                view_reasons.append("view_low_confidence")
            if prediction.matches_expected is False:
                view_reasons.append("view_mismatch")
        if view_reasons:
            sample = _safe_collect(
                store,
                task=LearningTask.EXTERIOR_VIEW,
                primary_content=content,
                primary_filename=filename,
                trigger_reasons=view_reasons,
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                source_id=result.request_id,
                expected_label=result.expected_view.value,
                predicted_label=prediction.predicted_view.value if prediction else None,
                confidence=prediction.confidence if prediction else result.score / 100,
                score=result.score,
                model_version=prediction.model_version if prediction else None,
                metadata=result.model_dump(mode="json"),
                training_eligible=(
                    result.score >= settings.active_learning_min_training_score
                    and not result.retake_required
                ),
            )
            if sample is not None:
                created.append(sample.id)

    blur = result.blur_assessment
    if blur is not None and blur.probability >= blur.warning_threshold:
        # The uncertain and blocked bands are where human truths teach the
        # blur model most, including sharp white panels it mistakes for blur.
        sample = _safe_collect(
            store,
            task=LearningTask.CAPTURE_SHARPNESS,
            primary_content=content,
            primary_filename=filename,
            trigger_reasons=[
                "blur_model_blocked" if blur.probability >= blur.fail_threshold
                else "blur_model_uncertain"
            ],
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            source_id=result.request_id,
            expected_label="sharp",
            predicted_label="blurred",
            confidence=blur.probability,
            score=result.score,
            model_version=blur.model_version,
            metadata=result.model_dump(mode="json"),
            training_eligible=False,
        )
        if sample is not None:
            created.append(sample.id)

    coverage = result.coverage_prediction
    is_good_reference = (
        settings.active_learning_collect_good_samples
        and result.expected_view is not None
        and result.expected_view.value in EXTERIOR_LABELS
        and result.status.value == "pass"
        and result.score >= settings.active_learning_min_training_score
        and prediction is not None
        and prediction.matches_expected is True
        and prediction.confidence >= settings.active_learning_confidence_threshold
        and coverage is not None
        and coverage.status == CheckStatus.PASS
        and coverage.car_detected
    )
    if is_good_reference and result.expected_view is not None and prediction is not None:
        sample = _safe_collect(
            store,
            task=LearningTask.EXTERIOR_VIEW,
            primary_content=content,
            primary_filename=filename,
            trigger_reasons=["high_quality_reference_candidate"],
            rental_id=rental_id,
            vehicle_id=vehicle_id,
            source_id=result.request_id,
            expected_label=result.expected_view.value,
            predicted_label=prediction.predicted_view.value,
            confidence=prediction.confidence,
            score=result.score,
            model_version=prediction.model_version,
            metadata=result.model_dump(mode="json"),
            sample_kind="good_reference",
            training_eligible=True,
        )
        if sample is not None:
            created.append(sample.id)

    if coverage is not None and not (
        coverage.car_detected
        and not result.retake_required
        and result.score >= 90
    ):
        coverage_reasons: list[str] = []
        if coverage.status != CheckStatus.PASS:
            coverage_reasons.append(f"coverage_{coverage.status.value}")
        if coverage.confidence < settings.active_learning_confidence_threshold:
            coverage_reasons.append("coverage_low_confidence")
        if result.retake_required and not coverage_reasons:
            coverage_reasons.append("retake_required")
        if coverage_reasons:
            if not coverage.car_detected:
                predicted_label = "not_vehicle"
            elif coverage.area_ratio < settings.vehicle_coverage_low_warning:
                predicted_label = "too_far"
            elif (
                coverage.area_ratio > settings.vehicle_coverage_high_warning
                or len(touched_edges(
                    edge_margins_from_box(coverage.bounding_box),
                    settings.vehicle_edge_margin_warning,
                )) >= 2
            ):
                predicted_label = "too_close"
            else:
                predicted_label = "acceptable"
            sample = _safe_collect(
                store,
                task=LearningTask.VEHICLE_COVERAGE,
                primary_content=content,
                primary_filename=filename,
                trigger_reasons=coverage_reasons,
                rental_id=rental_id,
                vehicle_id=vehicle_id,
                source_id=result.request_id,
                expected_label="acceptable",
                predicted_label=predicted_label,
                confidence=coverage.confidence,
                score=result.score,
                model_version=coverage.model_version,
                metadata=result.model_dump(mode="json"),
            )
            if sample is not None:
                created.append(sample.id)

    return created


def collect_interior_sample(
    *,
    content: bytes,
    filename: str,
    request_id: str,
    rental_id: str,
    vehicle_id: str,
    assessment: CleanlinessAssessment,
    model_version: str,
    human_review_required: bool,
    store: OperationsStore | None = None,
) -> str | None:
    settings = get_runtime_recognition_settings()
    if not settings.active_learning_enabled:
        return None
    reasons: list[str] = []
    if assessment.specialist is None and assessment.confidence < settings.active_learning_confidence_threshold:
        reasons.append("confidence_below_90")
    if human_review_required and (
        assessment.specialist is None or assessment.specialist.suspected_dirty or not assessment.image_usable
    ):
        reasons.append("human_review_required")
    if not assessment.image_usable:
        reasons.append("image_unusable")
    if not reasons:
        return None
    sample = _safe_collect(
        store or get_operations_store(),
        task=LearningTask.INTERIOR_CLEANLINESS,
        primary_content=content,
        primary_filename=filename,
        trigger_reasons=reasons,
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        source_id=request_id,
        predicted_label=assessment.level.value,
        confidence=assessment.confidence,
        model_version=model_version,
        metadata=assessment.model_dump(mode="json"),
    )
    return sample.id if sample else None


def collect_damage_pair_sample(
    *,
    pickup_content: bytes,
    return_content: bytes,
    pickup_filename: str,
    return_filename: str,
    damage_case: DamageCase,
    store: OperationsStore | None = None,
) -> str | None:
    settings = get_runtime_recognition_settings()
    # Pending damage cases need their paired photos to record a review truth,
    # even when automatic learning collection is disabled.
    if (
        not settings.active_learning_enabled
        and damage_case.review_decision != DamageReviewDecision.PENDING
    ):
        return None
    comparison = damage_case.comparison
    reasons: list[str] = []
    if comparison.status == DamageComparisonStatus.INCOMPATIBLE:
        reasons.append("pair_incompatible_retake")
    if comparison.status == DamageComparisonStatus.REVIEW_REQUIRED:
        reasons.append("model_requested_review")
    sample_kind = "hard_case"
    if comparison.status == DamageComparisonStatus.NO_CHANGE:
        # A candidate-free pair is valuable negative evidence, but remains a
        # model suggestion until a reviewer supplies the human truth.
        reasons.append("model_no_change_candidate")
        sample_kind = "good_reference"
    confidence = comparison.risk_score
    predicted_label = comparison.status.value
    if comparison.status == DamageComparisonStatus.NO_CHANGE:
        predicted_label = "no_new_damage"
    if comparison.specialist is not None and comparison.specialist.detected:
        confidence = comparison.specialist.score
        predicted_label = "confirmed_new_damage"
    if comparison.verification is not None:
        confidence = comparison.verification.confidence
        predicted_label = comparison.verification.decision.value
        if confidence < settings.active_learning_confidence_threshold:
            reasons.append("confidence_below_90")
    if not reasons:
        return None
    collect_store = store or get_operations_store()
    collect = (
        collect_store.collect_learning_sample
        if damage_case.review_decision == DamageReviewDecision.PENDING
        else lambda **kwargs: _safe_collect(collect_store, **kwargs)
    )
    sample = collect(
        task=LearningTask.DAMAGE_PAIR,
        primary_content=pickup_content,
        secondary_content=return_content,
        primary_filename=pickup_filename,
        secondary_filename=return_filename,
        trigger_reasons=reasons,
        rental_id=damage_case.rental_id,
        vehicle_id=damage_case.vehicle_id,
        source_id=damage_case.id,
        expected_label=damage_case.expected_view.value,
        predicted_label=predicted_label,
        confidence=confidence,
        model_version=comparison.model_version,
        metadata=comparison.model_dump(mode="json"),
        sample_kind=sample_kind,
        training_eligible=False,
    )
    if (
        sample
        and comparison.status == DamageComparisonStatus.NO_CHANGE
        and not settings.active_learning_collect_good_samples
    ):
        collect_store.archive_automatic_reference(sample.id)
    return sample.id if sample else None


def collect_plate_sample(
    *,
    content: bytes,
    filename: str,
    request_id: str,
    rental_id: str | None,
    vehicle_id: str | None,
    prediction: PlatePrediction,
    store: OperationsStore | None = None,
    force_review: bool = False,
) -> str | None:
    settings = get_runtime_recognition_settings()
    if not settings.active_learning_enabled and not force_review:
        return None
    reasons: list[str] = []
    if prediction.timed_out:
        reasons.append("plate_timeout")
    if prediction.model_uncertain:
        reasons.append("plate_model_uncertain")
    if not prediction.plate_detected:
        reasons.append("plate_not_detected")
    if prediction.normalized_plate is None:
        reasons.append("plate_text_unreadable")
    if not prediction.is_confident:
        reasons.append("plate_low_confidence")
    if prediction.matches_expected is False:
        reasons.append("plate_mismatch")
    if not reasons:
        reasons.append("human_review_required")
    sample = _safe_collect(
        store or get_operations_store(),
        task=LearningTask.LICENSE_PLATE,
        primary_content=content,
        primary_filename=filename,
        trigger_reasons=reasons,
        rental_id=rental_id,
        vehicle_id=vehicle_id,
        source_id=request_id,
        expected_label=prediction.expected_plate,
        predicted_label=prediction.normalized_plate,
        confidence=prediction.confidence,
        model_version=prediction.model_version,
        metadata=prediction.model_dump(mode="json"),
        sample_kind="hard_case",
        training_eligible=False,
    )
    return sample.id if sample else None
