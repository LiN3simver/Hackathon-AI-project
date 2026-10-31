"""Fast OpenCV plate-candidate check for the live capture step.

Plate OCR itself runs in the background via rapid_plate_recognition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.plate import PlatePrediction
from app.services.cleanliness import InvalidModelResponseError
from app.services.runtime_settings import get_runtime_recognition_settings

PLATE_EDGE_MARGIN = 0.01
PLATE_VEHICLE_SLACK = 0.02


@dataclass(frozen=True)
class PlateCaptureAssessment:
    plate_visible: bool
    likely_readable: bool
    candidate_count: int
    width_pixels: int = 0
    height_pixels: int = 0
    sharpness: float = 0.0


def is_complete_plate_text(plate: str | None) -> bool:
    return bool(plate) and re.fullmatch(
        r"(?:[A-Z]{2,3}-[0-9]{3,4}|[0-9]{3,4}-[A-Z]{2,3})", plate
    ) is not None


def shared_plate_run(text: str, reference: str) -> str:
    """Longest run of consecutive characters both texts share, dashes ignored."""
    left, right = text.replace("-", ""), reference.replace("-", "")
    best = ""
    for start in range(len(left)):
        for end in range(start + len(best) + 1, len(left) + 1):
            if left[start:end] not in right:
                break
            best = left[start:end]
    return best


def _is_subsequence(text: str, reference: str) -> bool:
    remaining = iter(reference)
    return all(char in remaining for char in text)


def ordered_plate_match(text: str, reference: str, minimum: int = 4) -> str:
    """Characters OCR read in order from the reference plate, or "".

    A tilted plate often loses characters, which
    breaks every consecutive run. The read still counts when its letters
    start the reference's letters (at least two) and its digits, after
    O/I/Z fixes, appear in order among the reference's digits.
    """
    ocr = re.sub(r"[^A-Z0-9]", "", text.upper())
    ref = reference.replace("-", "")
    ref_letters = "".join(char for char in ref if char.isalpha())
    ref_digits = "".join(char for char in ref if char.isdigit())
    if len(ocr) < minimum or len(ref_letters) < 2:
        return ""
    letters_first = ref[0].isalpha()
    for split in range(2, len(ocr) - 1):
        letters, digits = (ocr[:split], ocr[split:]) if letters_first else (
            ocr[len(ocr) - split:], ocr[:len(ocr) - split]
        )
        digits = digits.translate(str.maketrans("OIZ", "012"))
        if (letters.isalpha() and digits.isdigit()
                and ref_letters.startswith(letters)
                and _is_subsequence(digits, ref_digits)):
            return letters + digits if letters_first else digits + letters
    return ""


def complete_plate_visible(
    prediction: PlatePrediction,
    width: int,
    height: int,
    vehicle_box: list[float] | None,
) -> bool:
    """Require complete, confidently read plate text.

    A full plate-format read already proves no character is cut, so a plate
    whose frame touches the photo edge still counts as complete. The vehicle
    box is not consulted: the car detector often boxes a neighbouring car or
    stops short of the bumper, and a full plate read is stronger evidence.
    Four-character fragment matches are still held to the vehicle box when
    they are produced.
    """
    if (
        not prediction.plate_detected
        or not prediction.is_confident
        or not prediction.normalized_plate
        or prediction.bounding_box is None
        or prediction.timed_out
        or not is_complete_plate_text(prediction.normalized_plate)
    ):
        return False
    x1, y1, x2, y2 = prediction.bounding_box
    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def rectified_plate_candidates(content: bytes, entries: list[dict]) -> tuple[list[bytes], list[list[float]]]:
    """Deskew detected text using OCR's quadrilateral, before generic retries.

    Bounding boxes returned for these crops remain in original image space.
    EXIF orientation must match the OCR worker and quality metrics.
    """
    from io import BytesIO

    from PIL import Image, ImageOps

    from app.services.license_plate import normalize_ocr_plate

    relevant = [entry for entry in entries if normalize_ocr_plate(entry["text"]) and entry.get("polygon")]
    if not relevant:
        return [], []
    with Image.open(BytesIO(content)) as source:
        image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(source).convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    crops, boxes = [], []
    for entry in sorted(relevant, key=lambda item: float(item["confidence"]), reverse=True)[:2]:
        points = np.asarray(entry["polygon"], dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        points *= np.asarray([width, height], dtype=np.float32)
        tl, tr, br, bl = points
        crop_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        crop_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        if crop_width < 1 or crop_height < 1:
            continue
        target_width = 900
        target_height = max(24, min(900, round(target_width * crop_height / crop_width)))
        destination = np.float32([
            [10, 10], [target_width - 11, 10],
            [target_width - 11, target_height - 11], [10, target_height - 11],
        ])
        crop = cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(points, destination),
            (target_width, target_height), borderMode=cv2.BORDER_REPLICATE,
        )
        # Avoid sharpening already compressed characters: it can turn a 6 into a 5.
        ok, encoded = cv2.imencode(".png", crop)
        if ok:
            crops.append(encoded.tobytes())
            boxes.append(entry["bbox"])
    return crops, boxes


def plate_touches_edge(box: list[float], width: int, height: int) -> bool:
    """Characters running into the photo edge may be cut.

    The plate text already has to match a complete plate format, so only a
    thin border is required; a corner photo often places the plate near the
    edge of the frame.
    """
    x1, y1, x2, y2 = box
    margin_x = max(2 / width, PLATE_EDGE_MARGIN)
    margin_y = max(2 / height, PLATE_EDGE_MARGIN)
    return x1 < margin_x or y1 < margin_y or x2 > 1 - margin_x or y2 > 1 - margin_y


def plate_inside_vehicle(box: list[float], vehicle_box: list[float] | None) -> bool:
    """The detector box often stops just short of the bumper corner."""
    if not vehicle_box:
        return False
    x1, y1, x2, y2 = box
    car_x1, car_y1, car_x2, car_y2 = vehicle_box
    slack = PLATE_VEHICLE_SLACK
    return (
        car_x1 - slack <= x1 and x2 <= car_x2 + slack
        and car_y1 - slack <= y1 and y2 <= car_y2 + slack
    )


def _encode(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    if width < 900:
        scale = min(6.0, 900 / max(1, width))
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    image = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    image = cv2.addWeighted(image, 1.65, blurred, -0.65, 0)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise InvalidModelResponseError("車牌候選影像無法編碼。")
    return encoded.tobytes()


def _plate_candidates(content: bytes) -> tuple[list[bytes], list[list[float]]]:
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidModelResponseError("車牌照片無法解析。")
    original_height, original_width = image.shape[:2]
    scale = min(1.0, 1500 / max(original_height, original_width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7)),
    )
    gradient = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gradient = np.absolute(gradient)
    maximum = float(gradient.max())
    if maximum > 0:
        gradient = (255 * (gradient / maximum)).astype("uint8")
    else:
        gradient = np.zeros_like(gray)
    gradient = cv2.morphologyEx(
        gradient,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
    )
    _, threshold = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
    )
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked: list[tuple[float, tuple[int, int, int, int]]] = []
    image_area = max(1, height * width)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_height < 12 or box_width < 50:
            continue
        aspect = box_width / max(1, box_height)
        area_ratio = box_width * box_height / image_area
        if not 1.7 <= aspect <= 7.5 or not 0.00025 <= area_ratio <= 0.08:
            continue
        center_distance = abs((x + box_width / 2) / width - 0.5)
        lower_bonus = (y + box_height / 2) / height
        score = (
            area_ratio * 18
            + min(aspect, 5) * 0.08
            + lower_bonus * 0.16
            - center_distance * 0.12
        )
        ranked.append((score, (x, y, box_width, box_height)))
    ranked.sort(reverse=True)
    crops: list[bytes] = []
    boxes: list[list[float]] = []
    for _, (x, y, box_width, box_height) in ranked[:4]:
        pad_x = int(box_width * 0.14)
        pad_y = int(box_height * 0.45)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size:
            crops.append(_encode(crop))
            boxes.append(
                [
                    round(x1 / width, 4),
                    round(y1 / height, 4),
                    round(x2 / width, 4),
                    round(y2 / height, 4),
                ]
            )
    return crops, boxes


def assess_plate_capture(content: bytes) -> PlateCaptureAssessment:
    """Fast, local capture hint; OCR remains the background source of truth."""
    crops, boxes = _plate_candidates(content)
    if not boxes:
        return PlateCaptureAssessment(False, False, 0)

    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise InvalidModelResponseError("車牌照片無法解析。")
    height, width = image.shape
    x1, y1, x2, y2 = boxes[0]
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    candidate = image[top:bottom, left:right]
    candidate_width = right - left
    candidate_height = bottom - top
    sharpness = (
        float(cv2.Laplacian(candidate, cv2.CV_64F).var())
        if candidate.size
        else 0.0
    )
    runtime = get_runtime_recognition_settings()
    likely_readable = (
        candidate_width >= runtime.plate_hint_min_width_px
        and candidate_height >= runtime.plate_hint_min_height_px
        and sharpness >= runtime.plate_hint_min_sharpness
    )
    return PlateCaptureAssessment(
        plate_visible=True,
        likely_readable=likely_readable,
        candidate_count=len(crops),
        width_pixels=candidate_width,
        height_pixels=candidate_height,
        sharpness=round(sharpness, 1),
    )
