"""Shared before/after evaluation used by manual diagnostics and inspection jobs."""

from __future__ import annotations

import logging

from app.schemas.damage import (
    DamageComparisonResult,
    DamageComparisonStatus,
    DamagePairSpecialistResult,
)
from app.services.damage_comparison import get_damage_comparator
from app.services.damage_pair_specialist import get_damage_pair_specialist
from app.services.model_time_limits import (
    ModelTimeoutError,
    RecognitionModel,
    run_with_time_limit,
)

logger = logging.getLogger(__name__)


def _run_specialist(
    pickup_content: bytes, return_content: bytes
) -> tuple[DamagePairSpecialistResult | None, str]:
    """Return the specialist result, or None plus a note for the reviewer."""
    try:
        return run_with_time_limit(
            RecognitionModel.DAMAGE_PAIR,
            get_damage_pair_specialist().predict,
            pickup_content,
            return_content,
        ), ""
    except ModelTimeoutError as exc:
        logger.warning("Damage-pair specialist timed out after %ss", exc.seconds)
        return None, f"車損比對模型超過 {exc.seconds:g} 秒時間上限未完成，"
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Damage-pair specialist unavailable: %s", exc)
        return None, ""


def _flag_for_review(comparison: DamageComparisonResult) -> DamageComparisonResult:
    """Route a specialist detection to a human; it never confirms damage itself."""
    specialist = comparison.specialist
    assert specialist is not None
    comparison.status = DamageComparisonStatus.REVIEW_REQUIRED
    comparison.risk_score = round(max(comparison.risk_score, specialist.score * 0.85), 4)
    comparison.summary = (
        f"車損比對模型在還車照偵測到疑似新增車損（分數 {specialist.score:.0%}，"
        f"門檻 {specialist.threshold:.0%}），已標示候選區並送交後台覆核。"
    )
    return comparison


def evaluate_damage_pair(
    pickup_content: bytes,
    return_content: bytes,
) -> DamageComparisonResult:
    """Run geometric change detection plus the learned damage-pair specialist.

    Neither stage may clear a pair on its own: the specialist's recall is not
    high enough to trust a miss, so every pair without a confident detection
    still goes to a human, exactly as when the old visual-language stage timed
    out. The Qwen3-VL stage was removed because it needed 158-203 s per pair on
    the deployment CPU and never finished inside its 45 s budget.
    """
    comparison = get_damage_comparator().compare(pickup_content, return_content)
    comparison.specialist, timeout_note = _run_specialist(pickup_content, return_content)
    if comparison.specialist is not None:
        comparison.latency_ms = round(comparison.latency_ms + comparison.specialist.latency_ms, 1)
    if comparison.specialist is not None and comparison.specialist.detected:
        if comparison.specialist.regions:
            # The learned regions decided the review, so they are what the reviewer
            # sees; pixel-difference boxes (often ground, shadow or background)
            # stay on record but no longer replace them.
            comparison.geometric_candidates = list(comparison.candidates)
            comparison.candidates = list(comparison.specialist.regions)
            comparison.annotated_return_image = (
                comparison.specialist.annotated_return_image
                or comparison.annotated_return_image
            )
        elif not comparison.candidates and comparison.specialist.annotated_return_image:
            comparison.annotated_return_image = comparison.specialist.annotated_return_image
        return _flag_for_review(comparison)

    specialist_note = (
        "車損比對模型未達門檻，" if comparison.specialist is not None else timeout_note
    )
    if comparison.status == DamageComparisonStatus.REVIEW_REQUIRED:
        comparison.summary = (
            f"幾何差異模型找到局部新增變化；{specialist_note}"
            "候選區證據已保留並送交後台覆核。"
        )
    elif comparison.status == DamageComparisonStatus.INCOMPATIBLE:
        comparison.summary = (
            f"照片幾何對位不足；{specialist_note}"
            "案件已送交營運人員判斷是否需要重拍。"
        )
    else:
        comparison.status = DamageComparisonStatus.INCOMPATIBLE
        comparison.summary = (
            f"幾何模型未找到明顯變化；{specialist_note}"
            "仍請人工檢查取還車雙圖。"
        )
    return comparison
