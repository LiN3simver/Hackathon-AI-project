from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.quality import (
    CheckStatus,
    ExpectedView,
    ImageMetrics,
    QualityCheck,
    QualityResponse,
    QualityStatus,
)


class InvalidImageError(ValueError):
    pass


REGIONAL_BLUR_GRID_SIZE = 4


@dataclass(frozen=True)
class Thresholds:
    min_short_edge: int = 540
    min_blur_fail: float = 35.0
    min_blur_warn: float = 80.0
    brightness_low_fail: float = 25.0
    brightness_low_warn: float = 45.0
    brightness_high_warn: float = 220.0
    brightness_high_fail: float = 242.0
    min_dynamic_range: float = 30.0
    max_clipped_ratio: float = 0.45


def _regional_blur_is_failure(gray: np.ndarray, fail_threshold: float) -> bool:
    """Catch broad blur that a few sharp edges can hide in a global score."""
    height, width = gray.shape
    if height < REGIONAL_BLUR_GRID_SIZE or width < REGIONAL_BLUR_GRID_SIZE:
        return False

    tile_blurs = np.asarray(
        [
            float(cv2.Laplacian(tile, cv2.CV_64F).var())
            for row in np.array_split(gray, REGIONAL_BLUR_GRID_SIZE, axis=0)
            for tile in np.array_split(row, REGIONAL_BLUR_GRID_SIZE, axis=1)
        ],
        dtype=np.float64,
    )
    at_least_half_below_fail = (
        int(np.count_nonzero(tile_blurs < fail_threshold)) * 2 >= tile_blurs.size
    )
    return bool(
        at_least_half_below_fail
        and float(np.median(tile_blurs)) < fail_threshold
    )


def _check(
    key: str,
    label: str,
    status: CheckStatus,
    message: str,
    measured_value: float | str | None = None,
    threshold: str | None = None,
) -> QualityCheck:
    return QualityCheck(
        key=key,
        label=label,
        status=status,
        message=message,
        measured_value=measured_value,
        threshold=threshold,
    )


def analyze_image(
    content: bytes,
    filename: str,
    request_id: str,
    expected_view: ExpectedView | None = None,
    thresholds: Thresholds | None = None,
) -> QualityResponse:
    t = thresholds or Thresholds()
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("檔案不是可解析的 JPEG、PNG 或 WebP 圖片") from exc

    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    short_edge = min(width, height)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)
    dark_ratio = float(np.mean(gray <= 15))
    bright_ratio = float(np.mean(gray >= 245))
    orientation = "landscape" if width >= height else "portrait"

    checks: list[QualityCheck] = []

    if short_edge < t.min_short_edge:
        checks.append(
            _check(
                "resolution",
                "解析度",
                CheckStatus.FAIL,
                f"解析度不足（短邊 {short_edge}px，需 {t.min_short_edge}px），請用原圖重拍。",
                short_edge,
                f"短邊至少 {t.min_short_edge}px",
            )
        )
    else:
        checks.append(
            _check(
                "resolution",
                "解析度",
                CheckStatus.PASS,
                "解析度足夠進行後續辨識。",
                short_edge,
                f"短邊至少 {t.min_short_edge}px",
            )
        )

    regional_blur_failure = _regional_blur_is_failure(gray, t.min_blur_fail)
    if blur < t.min_blur_fail or regional_blur_failure:
        blur_status, blur_message = (
            CheckStatus.FAIL,
            "照片模糊，請停穩手機重拍。",
        )
    elif blur < t.min_blur_warn:
        blur_status, blur_message = CheckStatus.WARNING, "照片略微模糊，建議停穩手機後重拍。"
    else:
        blur_status, blur_message = CheckStatus.PASS, "影像清晰度正常。"
    checks.append(
        _check(
            "sharpness",
            "清晰度",
            blur_status,
            blur_message,
            round(blur, 1),
            f"建議 >= {t.min_blur_warn:.0f}",
        )
    )

    if brightness < t.brightness_low_fail:
        exposure_status, exposure_message = (
            CheckStatus.FAIL,
            "照片太暗，請補光重拍。",
        )
    elif brightness < t.brightness_low_warn:
        exposure_status, exposure_message = CheckStatus.WARNING, "照片偏暗，建議增加環境光源。"
    elif brightness > t.brightness_high_fail:
        exposure_status, exposure_message = CheckStatus.FAIL, "照片過曝，請避開強光重拍。"
    elif brightness > t.brightness_high_warn:
        exposure_status, exposure_message = CheckStatus.WARNING, "照片偏亮，建議避開直射光。"
    else:
        exposure_status, exposure_message = CheckStatus.PASS, "整體曝光正常。"
    checks.append(
        _check(
            "exposure",
            "曝光",
            exposure_status,
            exposure_message,
            round(brightness, 1),
            f"建議 {t.brightness_low_warn:.0f}–{t.brightness_high_warn:.0f}",
        )
    )

    clipped_ratio = max(dark_ratio, bright_ratio)
    if clipped_ratio > t.max_clipped_ratio or dynamic_range < t.min_dynamic_range:
        range_status, range_message = CheckStatus.WARNING, "明暗細節不足，請調整角度並避免逆光。"
    else:
        range_status, range_message = CheckStatus.PASS, "明暗細節可辨識。"
    checks.append(
        _check(
            "dynamic_range",
            "明暗細節",
            range_status,
            range_message,
            round(dynamic_range, 1),
            f">= {t.min_dynamic_range:.0f}",
        )
    )

    checks.append(
        _check(
            "orientation",
            "拍攝方向",
            CheckStatus.PASS,
            "已正確讀取照片方向；是否完整涵蓋車身將由構圖模型判斷。",
            orientation,
        )
    )

    checks.extend(
        [
            _check(
                "vehicle_view",
                "車輛視角",
                CheckStatus.PENDING_MODEL,
                "車外視角模型將在基礎品質通過後判斷左前、右前、左後與右後。",
            ),
            _check(
                "plate_presence",
                "完整車牌" if expected_view in {
                    ExpectedView.LEFT_FRONT, ExpectedView.RIGHT_FRONT,
                    ExpectedView.LEFT_REAR, ExpectedView.RIGHT_REAR,
                } else "車牌（車內不適用）",
                CheckStatus.PENDING_MODEL if expected_view in {
                    ExpectedView.LEFT_FRONT, ExpectedView.RIGHT_FRONT,
                    ExpectedView.LEFT_REAR, ExpectedView.RIGHT_REAR,
                } else CheckStatus.PASS,
                "車外四角照片必須讀出完整車牌；車內照片不要求車牌。",
                "待車牌檢查" if expected_view in {
                    ExpectedView.LEFT_FRONT, ExpectedView.RIGHT_FRONT,
                    ExpectedView.LEFT_REAR, ExpectedView.RIGHT_REAR,
                } else "不適用",
                "車外四角必填",
            ),
            _check(
                "vehicle_coverage",
                "車身覆蓋率",
                CheckStatus.PENDING_MODEL,
                "車體分割模型尚未載入；目前不將此項納入通過與否。",
            ),
        ]
    )

    operational = [item for item in checks if item.status != CheckStatus.PENDING_MODEL]
    fail_count = sum(item.status == CheckStatus.FAIL for item in operational)
    warning_count = sum(item.status == CheckStatus.WARNING for item in operational)
    score = max(0, 100 - fail_count * 28 - warning_count * 10)
    if fail_count:
        status = QualityStatus.FAIL
        summary = "照片需要重拍，請依下方指引調整。"
    elif warning_count:
        status = QualityStatus.WARNING
        summary = "照片可辨識但品質有改善空間，建議重拍以降低誤判。"
    else:
        status = QualityStatus.PASS
        summary = "基礎影像品質通過，可以進入 AI 車況辨識。"

    metrics = ImageMetrics(
        width=width,
        height=height,
        megapixels=round(width * height / 1_000_000, 2),
        blur_variance=round(blur, 2),
        brightness=round(brightness, 2),
        dynamic_range=round(dynamic_range, 2),
        dark_pixel_ratio=round(dark_ratio, 4),
        bright_pixel_ratio=round(bright_ratio, 4),
        orientation=orientation,
    )
    return QualityResponse(
        request_id=request_id,
        filename=filename,
        expected_view=expected_view,
        status=status,
        score=score,
        summary=summary,
        retake_required=status == QualityStatus.FAIL,
        checks=checks,
        metrics=metrics,
    )
