from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.admin import RecognitionSettings
from app.schemas.history import AutomationRoute
from app.schemas.inspection import InspectionAnalysisTask
from app.services.runtime_settings import get_runtime_recognition_settings


@dataclass(frozen=True)
class RoutingDecision:
    recommended_route: AutomationRoute
    effective_route: AutomationRoute
    shadow_mode: bool
    confidence: float | None
    reason_codes: list[str]
    evidence: dict[str, Any]
    policy_version: str

    def as_result_payload(self) -> dict[str, Any]:
        return {
            "recommended_route": self.recommended_route.value,
            "effective_route": self.effective_route.value,
            "shadow_mode": self.shadow_mode,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "evidence": self.evidence,
            "policy_version": self.policy_version,
        }


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def _recommend_plate(
    result: dict[str, Any], thresholds: RecognitionSettings
) -> tuple[AutomationRoute, float | None, list[str], dict[str, Any]]:
    prediction = result.get("prediction") or {}
    confidence = _number(prediction.get("confidence"))
    decision = str(result.get("decision") or "not_confirmed")
    evidence = {
        "decision": decision,
        "matches_expected": prediction.get("matches_expected"),
        "recognized_plate": prediction.get("normalized_plate"),
        "model_uncertain": prediction.get("model_uncertain", False),
        "timed_out": prediction.get("timed_out", False),
    }
    if prediction.get("model_uncertain") or prediction.get("timed_out"):
        return AutomationRoute.MANUAL_REVIEW, confidence, ["plate_model_uncertain"], evidence
    if (
        decision == "matched"
        and confidence is not None
        and confidence >= thresholds.automation_auto_clear_confidence
    ):
        return AutomationRoute.AUTO_CLEAR, confidence, ["plate_high_confidence_match"], evidence
    if (
        decision == "mismatch_review"
        and confidence is not None
        and confidence >= thresholds.automation_auto_case_confidence
    ):
        return AutomationRoute.AUTO_CASE, confidence, ["plate_high_confidence_mismatch"], evidence
    return AutomationRoute.MANUAL_REVIEW, confidence, ["plate_not_conclusive"], evidence


def _recommend_interior(
    result: dict[str, Any], thresholds: RecognitionSettings
) -> tuple[AutomationRoute, float | None, list[str], dict[str, Any]]:
    assessment = result.get("assessment") or {}
    confidence = _number(assessment.get("confidence"))
    level = str(assessment.get("level") or "cannot_assess")
    usable = bool(assessment.get("image_usable"))
    immediate = bool(assessment.get("immediate_cleaning_required"))
    evidence = {
        "level": level,
        "image_usable": usable,
        "immediate_cleaning_required": immediate,
        "alert_id": result.get("alert_id"),
    }
    screening = assessment.get("specialist")
    if isinstance(screening, dict):
        seat = "rear" if assessment.get("view") == "rear" else "front"
        evidence[f"{seat}_cleanliness_screening"] = screening
        return AutomationRoute.MANUAL_REVIEW, None, [f"{seat}_cleanliness_human_verdict_required"], evidence
    if not usable or level == "cannot_assess":
        return AutomationRoute.MANUAL_REVIEW, confidence, ["interior_not_assessable"], evidence
    if (
        level == "dirty"
        and immediate
        and confidence is not None
        and confidence >= thresholds.automation_auto_case_confidence
    ):
        return AutomationRoute.AUTO_CASE, confidence, ["interior_high_confidence_dirty"], evidence
    if (
        level in {"clean", "ordinary"}
        and confidence is not None
        and confidence >= thresholds.automation_auto_clear_confidence
    ):
        return AutomationRoute.AUTO_CLEAR, confidence, ["interior_high_confidence_clear"], evidence
    return (
        AutomationRoute.MANUAL_REVIEW,
        confidence,
        ["interior_confidence_or_policy_exception"],
        evidence,
    )


def _recommend_visible_damage(
    result: dict[str, Any], thresholds: RecognitionSettings
) -> tuple[AutomationRoute, float | None, list[str], dict[str, Any]]:
    assessment = result.get("assessment") or {}
    damage_probability = _number(assessment.get("confidence"))
    visible = bool(assessment.get("damage_visible"))
    confidence = (
        damage_probability if visible else 1 - damage_probability
    ) if damage_probability is not None else None
    usable = bool(assessment.get("image_usable"))
    evidence = {
        "damage_visible": visible,
        "image_usable": usable,
        "damage_probability": damage_probability,
        "severity": assessment.get("severity"),
        "alert_id": result.get("alert_id"),
    }
    if not usable:
        return AutomationRoute.MANUAL_REVIEW, confidence, ["damage_image_not_usable"], evidence
    if (
        visible
        and confidence is not None
        and confidence >= thresholds.automation_auto_case_confidence
    ):
        return AutomationRoute.AUTO_CASE, confidence, ["visible_damage_high_confidence"], evidence
    if (
        not visible
        and confidence is not None
        and confidence >= thresholds.automation_auto_clear_confidence
    ):
        return (
            AutomationRoute.AUTO_CLEAR,
            confidence,
            ["visible_damage_high_confidence_clear"],
            evidence,
        )
    return AutomationRoute.MANUAL_REVIEW, confidence, ["visible_damage_not_conclusive"], evidence


def _recommend_damage_pair(
    result: dict[str, Any], thresholds: RecognitionSettings
) -> tuple[AutomationRoute, float | None, list[str], dict[str, Any]]:
    status = str(result.get("comparison_status") or "incompatible")
    verification = result.get("verification") or {}
    confidence = _number(verification.get("confidence"))
    verification_decision = verification.get("decision")
    evidence = {
        "comparison_status": status,
        "verification_decision": verification_decision,
        "same_vehicle_and_view": verification.get("same_vehicle_and_view"),
        "risk_score": result.get("risk_score"),
        "damage_case_id": result.get("damage_case_id"),
    }
    if (
        status == "no_change"
        and verification_decision == "no_new_damage"
        and verification.get("same_vehicle_and_view") is True
        and confidence is not None
        and confidence >= thresholds.automation_auto_clear_confidence
    ):
        return AutomationRoute.AUTO_CLEAR, confidence, ["damage_pair_no_change"], evidence
    if (
        status == "review_required"
        and verification_decision == "confirmed_new_damage"
        and verification.get("same_vehicle_and_view") is True
        and confidence is not None
        and confidence >= thresholds.automation_auto_case_confidence
    ):
        return (
            AutomationRoute.AUTO_CASE,
            confidence,
            ["damage_pair_high_confidence_change"],
            evidence,
        )
    if status == "no_change":
        return (
            AutomationRoute.MANUAL_REVIEW,
            confidence,
            ["damage_pair_clear_not_verified"],
            evidence,
        )
    if status == "incompatible":
        return AutomationRoute.MANUAL_REVIEW, confidence, ["damage_pair_incompatible"], evidence
    return AutomationRoute.MANUAL_REVIEW, confidence, ["damage_pair_model_disagreement"], evidence


def route_analysis_decision(
    task: InspectionAnalysisTask,
    result: dict[str, Any],
    *,
    legacy_needs_review: bool,
    settings: Settings | None = None,
    thresholds: RecognitionSettings | None = None,
) -> RoutingDecision:
    """Route one analysis result.

    Routing switches come from the environment; confidence thresholds come from the
    admin recognition settings so changes apply to the next job without a restart.
    """
    settings = settings or get_settings()
    thresholds = thresholds or get_runtime_recognition_settings()
    if task == InspectionAnalysisTask.PLATE_RECOGNITION:
        recommended, confidence, reasons, evidence = _recommend_plate(result, thresholds)
    elif task == InspectionAnalysisTask.INTERIOR_CLEANLINESS:
        recommended, confidence, reasons, evidence = _recommend_interior(result, thresholds)
    elif task == InspectionAnalysisTask.DAMAGE_COMPARISON:
        recommended, confidence, reasons, evidence = _recommend_damage_pair(result, thresholds)
    else:
        recommended, confidence, reasons, evidence = _recommend_visible_damage(result, thresholds)

    legacy_route = (
        AutomationRoute.MANUAL_REVIEW if legacy_needs_review else AutomationRoute.AUTO_CLEAR
    )
    shadow_mode = settings.automation_routing_enabled and settings.automation_shadow_mode
    if not settings.automation_routing_enabled:
        recommended = legacy_route
        reasons = ["automation_routing_disabled"]
        effective = legacy_route
    elif shadow_mode:
        effective = legacy_route
    else:
        effective = recommended
        if effective == AutomationRoute.AUTO_CASE and not (
            result.get("alert_id") or result.get("damage_case_id")
        ):
            effective = AutomationRoute.MANUAL_REVIEW
            reasons = [*reasons, "auto_case_has_no_case_record"]
    return RoutingDecision(
        recommended_route=recommended,
        effective_route=effective,
        shadow_mode=shadow_mode,
        confidence=confidence,
        reason_codes=reasons,
        evidence=evidence,
        policy_version=settings.automation_policy_version,
    )
