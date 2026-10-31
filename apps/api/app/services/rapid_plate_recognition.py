"""Production plate OCR using a warm, isolated RapidOCR ONNX CPU process."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from app.schemas.plate import PlatePrediction
from app.services.license_plate import normalize_ocr_plate, normalize_plate
from app.services.model_time_limits import (
    ModelTimeoutError,
    RecognitionModel,
    time_limit_seconds,
)
from app.services.runtime_settings import PROJECT_ROOT, get_runtime_recognition_settings

MODEL_VERSION = "rapidocr-3.9.2-ppocrv6-small-onnx"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv-rapidocr" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
# Startup shares the same per-photo deadline as inference and crop retries.
MODEL_FILES = (
    "PP-OCRv6_det_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "PP-OCRv6_rec_small.onnx",
)


def _python_path() -> Path:
    return Path(os.environ.get("GUARDIAN_RAPIDOCR_PYTHON") or DEFAULT_PYTHON)


def _models_available(python: Path) -> bool:
    environment = python.parent.parent
    candidates = (
        [environment / "Lib" / "site-packages" / "rapidocr" / "models"]
        if os.name == "nt"
        else list(environment.glob("lib/python3*/site-packages/rapidocr/models"))
    )
    return any(all((models / name).is_file() for name in MODEL_FILES) for models in candidates)


def model_info() -> dict:
    python = _python_path()
    available = python.is_file() and _models_available(python)
    return {
        "name": "RapidOCR PP-OCRv6 small",
        "version": MODEL_VERSION,
        "engine": "ONNX Runtime CPU",
        "status": "active" if available else "unavailable",
        "available": available,
        "message": (
            "預訓練模型已就緒；辨識結果只作為人工確認的預填值"
            if available
            else "RapidOCR CPU 環境或模型檔不存在，請執行 scripts/setup_plate_pretrained.ps1"
        ),
        "python": str(python),
        "model_files": list(MODEL_FILES),
        "source": "https://github.com/RapidAI/RapidOCR",
        "license": "Apache-2.0",
        "requires_training": False,
        "human_confirmation": True,
    }


class RapidPlateRecognizer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict | None] = queue.Queue()
        self._ready = threading.Event()

    def _stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.wait(timeout=5)
            self._process = None

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ModelTimeoutError(RecognitionModel.PLATE, time_limit_seconds(RecognitionModel.PLATE))
        return remaining

    def _start(self, deadline: float) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        python = _python_path()
        if not python.is_file():
            raise FileNotFoundError("RapidOCR CPU 環境尚未安裝")
        environment = os.environ.copy()
        environment.update({
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONIOENCODING": "utf-8",
        })
        self._responses = queue.Queue()
        self._ready = threading.Event()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [str(python), "-u", str(PROJECT_ROOT / "scripts" / "serve_plate_rapidocr.py")],
            cwd=PROJECT_ROOT, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, creationflags=flags,
        )
        responses = self._responses
        ready = self._ready
        process = self._process

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith("PLATE_READY"):
                    ready.set()
                elif line.startswith("PLATE_RESULT\t"):
                    try:
                        responses.put(json.loads(line.split("\t", 1)[1]))
                    except json.JSONDecodeError:
                        responses.put({"error": "OCR 回傳格式錯誤"})
            ready.set()
            responses.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        try:
            remaining = self._remaining(deadline)
        except ModelTimeoutError:
            self._stop()
            raise
        if not ready.wait(remaining):
            self._stop()
            raise ModelTimeoutError(RecognitionModel.PLATE, time_limit_seconds(RecognitionModel.PLATE))

    def _infer(self, content: bytes, deadline: float) -> list[dict]:
        if not self._lock.acquire(timeout=self._remaining(deadline)):
            raise ModelTimeoutError(RecognitionModel.PLATE, time_limit_seconds(RecognitionModel.PLATE))
        try:
            self._start(deadline)
            assert self._process is not None and self._process.stdin is not None
            with tempfile.TemporaryDirectory(prefix="guardian-rapidocr-") as temporary:
                path = Path(temporary) / "plate-image"
                path.write_bytes(content)
                try:
                    self._remaining(deadline)
                    self._process.stdin.write(
                        json.dumps({"image_path": str(path)}, ensure_ascii=False) + "\n"
                    )
                    self._process.stdin.flush()
                    response = self._responses.get(timeout=self._remaining(deadline))
                except queue.Empty as exc:
                    # A late answer would be read by the next photo, so restart.
                    self._stop()
                    raise ModelTimeoutError(
                        RecognitionModel.PLATE, time_limit_seconds(RecognitionModel.PLATE)
                    ) from exc
                except ModelTimeoutError:
                    self._stop()
                    raise
                except (BrokenPipeError, OSError) as exc:
                    self._stop()
                    raise RuntimeError("RapidOCR 程序中斷") from exc
                if response is None:
                    self._stop()
                    raise RuntimeError("RapidOCR 程序意外結束")
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                return response["entries"]
        finally:
            self._lock.release()

    def _infer_with_plate_crops(
        self, content: bytes, entries: list[dict], deadline: float
    ) -> list[dict]:
        """Retry hard photos on enhanced candidate crops, preserving image coordinates."""
        self._remaining(deadline)
        threshold = get_runtime_recognition_settings().plate_recognition_confidence_threshold
        if any(normalize_ocr_plate(entry["text"]) and
               float(entry["confidence"]) >= threshold for entry in entries):
            return entries
        try:
            from app.services.plate_capture import _plate_candidates, rectified_plate_candidates

            targeted_crops, targeted_boxes = rectified_plate_candidates(content, entries)
            crops, boxes = ([], []) if targeted_crops else _plate_candidates(content)
        except (OSError, ValueError):
            return entries
        self._remaining(deadline)
        retries = [(crop, box, True) for crop, box in zip(targeted_crops, targeted_boxes, strict=True)]
        retries += [(crop, box, False) for crop, box in zip(crops[:2], boxes[:2], strict=True)]
        for crop, (left, top, right, bottom), rectified in retries:
            try:
                crop_entries = self._infer(crop, deadline)
            except ModelTimeoutError:
                raise
            except (OSError, RuntimeError, ValueError):
                continue
            for entry in crop_entries:
                x1, y1, x2, y2 = entry["bbox"]
                entries.append({
                    **entry,
                    "bbox": [left, top, right, bottom] if rectified else [
                        left + x1 * (right - left),
                        top + y1 * (bottom - top),
                        left + x2 * (right - left),
                        top + y2 * (bottom - top),
                    ],
                })
            if any(normalize_ocr_plate(entry["text"]) and
                   float(entry["confidence"]) >= threshold for entry in entries):
                break
        return entries

    def recognize(
        self, content: bytes, expected_plate: str | None = None,
        *, timeout_seconds: float | None = None,
        verified_plate: str | None = None,
        vehicle_box: list[float] | None = None,
    ) -> PlatePrediction:
        started = time.perf_counter()
        seconds = time_limit_seconds(RecognitionModel.PLATE) if timeout_seconds is None else timeout_seconds
        deadline = started + seconds
        expected = normalize_plate(expected_plate)
        reference = normalize_plate(verified_plate)
        try:
            entries = self._infer(content, deadline)
            self._remaining(deadline)
            if verified_plate and vehicle_box:
                fast = self._match_verified_plate(entries, verified_plate, vehicle_box, started)
                if fast is not None:
                    return fast
            entries = self._infer_with_plate_crops(content, entries, deadline)
            self._remaining(deadline)
            candidates = []
            for entry in entries:
                plate = normalize_ocr_plate(entry["text"])
                if plate:
                    box = entry["bbox"]
                    # The reference plate wins over a neighbouring car's plate.
                    rank = (
                        (1.0 if plate == reference else 0.0)
                        + float(entry["confidence"])
                        + 0.2 * (box[1] + box[3]) / 2
                        + 0.05 * (box[2] - box[0])
                    )
                    candidates.append((rank, plate, entry))
            best = max(candidates, default=None, key=lambda item: item[0])
            self._remaining(deadline)
            if best is None:
                return PlatePrediction(
                    plate_detected=False, confidence=0, is_confident=False,
                    image_usable=True, expected_plate=expected, matches_expected=None,
                    message="未讀出完整車牌，請人工查看照片並輸入。",
                    model_version=MODEL_VERSION,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                    model_uncertain=True,
                )
            _, plate, entry = best
            confidence = float(entry["confidence"])
            threshold = get_runtime_recognition_settings().plate_recognition_confidence_threshold
            matches = (plate.replace("-", "") == expected.replace("-", "")) if expected else None
            return PlatePrediction(
                plate_detected=True, plate_text=entry["text"], normalized_plate=plate,
                confidence=confidence, is_confident=confidence >= threshold,
                bounding_box=entry["bbox"], image_usable=True,
                expected_plate=expected, matches_expected=matches,
                message=f"模型預填 {plate}；請核對照片後確認或修改。",
                model_version=MODEL_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                model_uncertain=confidence < threshold or len({
                    value for _, value, candidate in candidates
                    if float(candidate["confidence"]) >= threshold
                }) > 1,
            )
        except ModelTimeoutError:
            return timed_out_plate_prediction(
                seconds,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                expected_plate=expected,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return PlatePrediction(
                plate_detected=False, confidence=0, is_confident=False,
                image_usable=True, expected_plate=expected, matches_expected=None,
                message=f"車牌辨識暫不可用，請人工查看照片：{exc}",
                model_version=MODEL_VERSION,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                model_uncertain=True,
            )

    @staticmethod
    def _match_verified_plate(
        entries: list[dict], verified_plate: str, vehicle_box: list[float], started: float,
    ) -> PlatePrediction | None:
        from app.services.plate_capture import (
            is_complete_plate_text,
            plate_inside_vehicle,
            shared_plate_run,
        )

        verified = normalize_plate(verified_plate)
        if not is_complete_plate_text(verified):
            return None
        compact = verified.replace("-", "")
        threshold = get_runtime_recognition_settings().plate_recognition_confidence_threshold
        # A different full plate on this vehicle must never be rescued by a
        # shared numeric fragment.
        for entry in entries:
            plate = normalize_ocr_plate(entry["text"])
            if (is_complete_plate_text(plate) and plate != verified
                    and plate_inside_vehicle(entry["bbox"], vehicle_box)):
                return None
        matches = []
        for entry in entries:
            raw = entry["text"].strip().upper()
            if not re.fullmatch(r"[A-Z0-9\s:·.\-]+", raw):
                continue
            # A tilted plate's dash or screw is often read as an extra
            # character, so compare the longest run
            # shared with the reference instead of requiring a clean substring.
            fragment = shared_plate_run(re.sub(r"[^A-Z0-9]", "", raw), compact)
            confidence = float(entry["confidence"])
            if (len(fragment) >= 4 and confidence >= threshold
                    and plate_inside_vehicle(entry["bbox"], vehicle_box)):
                matches.append((confidence, fragment, entry))
        if not matches:
            return None
        confidence, fragment, entry = max(matches, key=lambda item: (item[0], len(item[1])))
        return PlatePrediction(
            plate_detected=True, plate_text=entry["text"], normalized_plate=verified,
            confidence=confidence, is_confident=True, bounding_box=entry["bbox"],
            image_usable=True, message=f"已與本次車牌 {verified} 核對 {len(fragment)} 個字元，快速通過。",
            model_version=MODEL_VERSION, latency_ms=round((time.perf_counter() - started) * 1000, 1),
            fast_matched=True, matched_characters=fragment,
        )


def timed_out_plate_prediction(
    seconds: float, *, latency_ms: float | None = None, expected_plate: str | None = None
) -> PlatePrediction:
    return PlatePrediction(
        plate_detected=False, confidence=0, is_confident=False,
        image_usable=True, expected_plate=expected_plate, matches_expected=None,
        message=f"車牌辨識超過 {seconds:g} 秒，已放棄本次辨識；請重拍清晰車牌。",
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms if latency_ms is not None else seconds * 1000, 1),
        model_uncertain=True, timed_out=True,
    )


_recognizer = RapidPlateRecognizer()


def get_plate_recognizer() -> RapidPlateRecognizer:
    return _recognizer
