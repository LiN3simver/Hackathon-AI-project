from __future__ import annotations

import base64
from io import BytesIO
from time import perf_counter
from urllib.error import URLError

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.damage import (
    AlignmentResult,
    DamageCandidate,
    DamageComparisonResult,
    DamageComparisonStatus,
)
from app.services.vehicle_coverage import get_vehicle_coverage_detector


class InvalidDamageImageError(ValueError):
    pass


def _decode(content: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            array = np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidDamageImageError("無法讀取車況照片") from exc
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _fit_long_side(image: np.ndarray, long_side: int = 1280) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _resize_pair(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downscale each photo on its own, keeping its aspect ratio.

    The two vehicle crops rarely share an aspect ratio; forcing the return photo
    to the pickup dimensions stretched it before alignment. Geometry differences
    are left to the homography, which maps onto the reference frame anyway.
    """
    return _fit_long_side(reference), _fit_long_side(current)


def _homography_is_plausible(
    homography: np.ndarray, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> bool:
    """Reject warps that shear, fold or collapse the photo.

    Mismatched features on glossy white paint can still pass RANSAC and produce
    a homography that flips or wedges the image. A same-angle retake should map
    the photo's corners onto a convex quadrilateral of similar size and shape.
    """
    if not np.all(np.isfinite(homography)) or abs(homography[2, 2]) < 1e-9:
        return False
    matrix = homography / homography[2, 2]
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    # Perspective terms, normalised by image size, stay small for handheld retakes.
    if abs(matrix[2, 0]) * source_width > 0.6 or abs(matrix[2, 1]) * source_height > 0.6:
        return False
    corners = np.float32(
        [[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]]
    ).reshape(-1, 1, 2)
    denominators = matrix[2, 0] * corners[:, 0, 0] + matrix[2, 1] * corners[:, 0, 1] + 1
    if np.any(denominators <= 0.05):
        return False
    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not cv2.isContourConvex(projected.astype(np.float32)):
        return False
    # Every corner must keep the source photo's winding; a mirrored or folded
    # warp flips it.
    edges = np.roll(projected, -1, axis=0) - projected
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if not np.all(turns > 0):
        return False
    source_area = float(source_width * source_height)
    projected_area = float(cv2.contourArea(projected.astype(np.float32)))
    target_area = float(target_width * target_height)
    # Scale relative to the source photo and to the reference frame.
    if not 0.25 <= projected_area / source_area <= 4.0:
        return False
    if projected_area < 0.3 * target_area:
        return False
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() <= 1:
        return False
    # Opposite sides should keep a similar length (no wedge-shaped warp).
    if max(lengths[0], lengths[2]) / min(lengths[0], lengths[2]) > 2.0:
        return False
    if max(lengths[1], lengths[3]) / min(lengths[1], lengths[3]) > 2.0:
        return False
    # Corner angles should stay near 90 degrees (no heavy shear).
    for index in range(4):
        incoming = -edges[index - 1]
        outgoing = edges[index]
        cosine = float(
            np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        )
        if abs(cosine) > 0.5:  # beyond 60-120 degrees
            return False
    return True


def _crop_vehicle(image: np.ndarray, bounding_box: list[float]) -> np.ndarray:
    """Crop to the detected primary vehicle while retaining a small context margin."""
    if len(bounding_box) != 4:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounding_box
    pad_x = 0.025
    pad_y = 0.025
    left = max(0, round((x1 - pad_x) * width))
    top = max(0, round((y1 - pad_y) * height))
    right = min(width, round((x2 + pad_x) * width))
    bottom = min(height, round((y2 + pad_y) * height))
    if right - left < 240 or bottom - top < 180:
        return image
    return image[top:bottom, left:right]


def _prepare_vehicle_pair(
    pickup_content: bytes, return_content: bytes
) -> tuple[np.ndarray, np.ndarray]:
    pickup = _decode(pickup_content)
    returned = _decode(return_content)
    try:
        detector = get_vehicle_coverage_detector()
        pickup_detection = detector.predict(pickup_content)
        return_detection = detector.predict(return_content)
        if pickup_detection.car_detected and return_detection.car_detected:
            pickup = _crop_vehicle(pickup, pickup_detection.bounding_box)
            returned = _crop_vehicle(returned, return_detection.bounding_box)
    except (RuntimeError, ValueError, OSError, URLError):
        # The comparison still has a deterministic full-frame fallback when the
        # optional object detector is unavailable.
        pass
    return _resize_pair(pickup, returned)


def _feature_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _align(
    reference: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray, AlignmentResult]:
    reference_features = _feature_image(reference)
    current_features = _feature_image(current)
    orb = cv2.ORB_create(nfeatures=4500, fastThreshold=8)
    keypoints_reference, descriptors_reference = orb.detectAndCompute(
        reference_features, None
    )
    keypoints_current, descriptors_current = orb.detectAndCompute(current_features, None)
    failure = AlignmentResult(
        success=False,
        method="ORB + RANSAC homography",
        matched_features=0,
        inlier_ratio=0,
        overlap_ratio=0,
        score=0,
        message="兩張照片缺少足夠的共同特徵，請依相同視角重新拍攝。",
    )
    if descriptors_reference is None or descriptors_current is None:
        return current, np.zeros(reference.shape[:2], dtype=np.uint8), failure

    raw_matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        descriptors_current, descriptors_reference, k=2
    )
    good_matches = [
        first
        for pair in raw_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.76 * second.distance
    ]
    if len(good_matches) < 12:
        failure.matched_features = len(good_matches)
        return current, np.zeros(reference.shape[:2], dtype=np.uint8), failure

    source_points = np.float32(
        [keypoints_current[item.queryIdx].pt for item in good_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [keypoints_reference[item.trainIdx].pt for item in good_matches]
    ).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(
        source_points, destination_points, cv2.RANSAC, 4.0
    )
    if homography is None or inliers is None:
        failure.matched_features = len(good_matches)
        return current, np.zeros(reference.shape[:2], dtype=np.uint8), failure
    if not _homography_is_plausible(homography, current.shape, reference.shape):
        failure.matched_features = int(inliers.sum())
        failure.message = "兩張照片的拍攝角度或距離差異過大，對齊結果會變形，請依相同視角重新拍攝。"
        return current, np.zeros(reference.shape[:2], dtype=np.uint8), failure

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(current, homography, (width, height))
    valid_mask = cv2.warpPerspective(
        np.full(current.shape[:2], 255, dtype=np.uint8), homography, (width, height)
    )
    overlap_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(good_matches)
    score = min(1.0, inlier_count / 45) * min(1.0, inlier_ratio / 0.6)
    success = (
        inlier_count >= 10 and inlier_ratio >= 0.25 and overlap_ratio >= 0.55
    ) or (
        inlier_count >= 20 and inlier_ratio >= 0.35 and overlap_ratio >= 0.45
    )
    message = (
        "影像已對齊，可檢查局部新增變化。"
        if success
        else "拍攝位置差異過大，無法可靠比較，請依同一角度重拍。"
    )
    return aligned, valid_mask, AlignmentResult(
        success=success,
        method="ORB + RANSAC homography",
        matched_features=inlier_count,
        inlier_ratio=round(inlier_ratio, 4),
        overlap_ratio=round(overlap_ratio, 4),
        score=round(score, 4),
        message=message,
    )


def _data_url(image: np.ndarray, extension: str) -> str:
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return ""
    mime = "image/png" if extension == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}"


def _find_candidates(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    alignment_score: float,
) -> tuple[list[DamageCandidate], np.ndarray, float]:
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    aligned_lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)

    valid_pixels = valid_mask >= 250
    # Match broad exposure and white-balance differences before comparing. This
    # intentionally preserves local scratches/dents while removing day/night or
    # camera auto-exposure shifts that previously became false damage regions.
    normalized_aligned = aligned_lab.astype(np.float32)
    reference_float = reference_lab.astype(np.float32)
    for channel in range(3):
        reference_values = reference_float[:, :, channel][valid_pixels]
        aligned_values = normalized_aligned[:, :, channel][valid_pixels]
        if not reference_values.size or not aligned_values.size:
            continue
        ref_p10, ref_median, ref_p90 = np.percentile(reference_values, [10, 50, 90])
        cur_p10, cur_median, cur_p90 = np.percentile(aligned_values, [10, 50, 90])
        ref_span = max(12.0, float(ref_p90 - ref_p10))
        cur_span = max(12.0, float(cur_p90 - cur_p10))
        scale = float(np.clip(ref_span / cur_span, 0.72, 1.38))
        normalized_aligned[:, :, channel] = (
            normalized_aligned[:, :, channel] - cur_median
        ) * scale + ref_median
    normalized_aligned = np.clip(normalized_aligned, 0, 255).astype(np.uint8)

    reference_smooth = cv2.GaussianBlur(reference_lab, (5, 5), 0)
    aligned_smooth = cv2.GaussianBlur(normalized_aligned, (5, 5), 0)
    difference = cv2.absdiff(reference_smooth, aligned_smooth).astype(np.float32)

    # Remove slow illumination gradients from luminance. Chroma remains useful
    # for paint transfer, while local luminance catches dents and scratches.
    ref_luminance = reference_smooth[:, :, 0].astype(np.float32)
    cur_luminance = aligned_smooth[:, :, 0].astype(np.float32)
    ref_local = ref_luminance - cv2.GaussianBlur(ref_luminance, (0, 0), 15)
    cur_local = cur_luminance - cv2.GaussianBlur(cur_luminance, (0, 0), 15)
    luminance = np.abs(ref_local - cur_local)
    chroma = np.maximum(difference[:, :, 1], difference[:, :, 2])
    change_map = np.maximum(luminance * 0.92, chroma * 1.05)
    change_map[valid_mask < 250] = 0

    # Most residuals around an existing panel/headlight/background edge are
    # sub-pixel alignment error, not new damage. Downweight those zones rather
    # than drawing red boxes around every contour of the car.
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_edges = cv2.Canny(reference_gray, 70, 150)
    reference_edges = cv2.dilate(
        reference_edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    change_map[reference_edges > 0] *= 0.28

    height, width = change_map.shape
    border = max(5, round(min(height, width) * 0.025))
    change_map[:border, :] = 0
    change_map[-border:, :] = 0
    change_map[:, :border] = 0
    change_map[:, -border:] = 0

    valid_values = change_map[valid_pixels]
    if valid_values.size:
        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        robust_threshold = median + 7.0 * max(1.0, 1.4826 * mad)
        tail_threshold = float(np.percentile(valid_values, 99.35))
        threshold = max(34.0, robust_threshold, tail_threshold)
    else:
        threshold = 255.0
    binary = np.where(change_map >= threshold, 255, 0).astype(np.uint8)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = height * width
    minimum_area = max(180, image_area * 0.00055)
    candidates: list[DamageCandidate] = []
    accepted_mask = np.zeros_like(binary)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in sorted_contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area or area > image_area * 0.045:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 10 or box_height < 10:
            continue
        region_mask = np.zeros_like(binary)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
        mean_change = float(cv2.mean(change_map, mask=region_mask)[0])
        region_values = change_map[region_mask > 0]
        strong_density = float(np.mean(region_values >= threshold + 5.0))
        peak_change = float(np.percentile(region_values, 90))
        if strong_density < 0.28 or peak_change < threshold + 10.0:
            continue
        area_ratio = area / image_area
        change_score = min(
            1.0,
            (mean_change / 82) * 0.62
            + min(1.0, area_ratio / 0.009) * 0.18
            + strong_density * 0.12
            + alignment_score * 0.08,
        )
        if change_score < 0.52:
            continue
        cv2.drawContours(accepted_mask, [contour], -1, 255, thickness=-1)
        candidates.append(
            DamageCandidate(
                id=f"REGION-{len(candidates) + 1:02d}",
                bounding_box=[
                    round(x / width, 4),
                    round(y / height, 4),
                    round((x + box_width) / width, 4),
                    round((y + box_height) / height, 4),
                ],
                area_ratio=round(area_ratio, 5),
                change_score=round(change_score, 4),
                description="此區域與取車照片有明顯局部視覺變化，需人工放大確認。",
            )
        )
        if len(candidates) == 5:
            break

    change_area_ratio = float(np.count_nonzero(accepted_mask) / image_area)
    return candidates, accepted_mask, change_area_ratio


class DamageComparator:
    model_version = "opencv-vehicle-roi-photometric-change-v2"

    def compare(self, pickup_content: bytes, return_content: bytes) -> DamageComparisonResult:
        started = perf_counter()
        pickup, returned = _prepare_vehicle_pair(pickup_content, return_content)
        aligned, valid_mask, alignment = _align(pickup, returned)
        if not alignment.success:
            return DamageComparisonResult(
                status=DamageComparisonStatus.INCOMPATIBLE,
                alignment=alignment,
                candidates=[],
                change_area_ratio=0,
                risk_score=0,
                summary="兩張照片無法可靠對齊，系統不進行車損推論，請重拍。",
                model_version=self.model_version,
                latency_ms=round((perf_counter() - started) * 1000, 1),
            )

        candidates, mask, change_area_ratio = _find_candidates(
            pickup, aligned, valid_mask, alignment.score
        )
        annotated = aligned.copy()
        overlay = np.zeros_like(annotated)
        overlay[:, :, 2] = mask
        annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.28, 0)
        height, width = annotated.shape[:2]
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bounding_box
            point_one = (round(x1 * width), round(y1 * height))
            point_two = (round(x2 * width), round(y2 * height))
            cv2.rectangle(annotated, point_one, point_two, (32, 54, 230), 3)
            cv2.putText(
                annotated,
                candidate.id,
                (point_one[0], max(24, point_one[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (32, 54, 230),
                2,
                cv2.LINE_AA,
            )

        status = (
            DamageComparisonStatus.REVIEW_REQUIRED
            if candidates
            else DamageComparisonStatus.NO_CHANGE
        )
        risk_score = min(
            1.0,
            sum(item.change_score for item in candidates) / 2.2
            + min(0.2, change_area_ratio * 4),
        )
        summary = (
            f"找到 {len(candidates)} 個疑似新增變化區域，請營運人員人工覆核。"
            if candidates
            else "未找到達門檻的局部新增變化；結果仍保留人工抽查。"
        )
        return DamageComparisonResult(
            status=status,
            alignment=alignment,
            candidates=candidates,
            change_area_ratio=round(change_area_ratio, 5),
            risk_score=round(risk_score, 4),
            summary=summary,
            annotated_return_image=_data_url(annotated, ".jpg"),
            difference_mask_image=_data_url(mask, ".png"),
            model_version=self.model_version,
            latency_ms=round((perf_counter() - started) * 1000, 1),
        )


_comparator: DamageComparator | None = None


def get_damage_comparator() -> DamageComparator:
    global _comparator
    if _comparator is None:
        _comparator = DamageComparator()
    return _comparator
