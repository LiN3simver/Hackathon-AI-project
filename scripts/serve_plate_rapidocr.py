"""Persistent, isolated RapidOCR CPU process for production plate inference."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from PIL import Image, ImageOps  # noqa: E402
from rapidocr import RapidOCR  # noqa: E402


def main() -> None:
    # OMP_NUM_THREADS does not configure ONNX Runtime's own thread pools.
    # Unbounded pools in all three OCR models compete with the capture models.
    engine = RapidOCR(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    # Tells the API the models are loaded, so startup never counts as recognition time.
    sys.stdout.write("PLATE_READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            path = Path(request["image_path"])
            # RapidOCR applies EXIF orientation before detection, so boxes must be
            # normalized against the rotated size, not the raw sensor size.
            with Image.open(path) as image:
                width, height = ImageOps.exif_transpose(image).size
            result = engine(str(path))
            boxes = result.boxes if result.boxes is not None else []
            entries = []
            for text, score, box in zip(
                result.txts or (), result.scores or (), boxes, strict=False
            ):
                x_values = [float(point[0]) for point in box]
                y_values = [float(point[1]) for point in box]
                entries.append({
                    "text": str(text), "confidence": float(score),
                    "polygon": [[float(point[0]) / width, float(point[1]) / height]
                                for point in box],
                    "bbox": [
                        max(0.0, min(x_values) / width),
                        max(0.0, min(y_values) / height),
                        min(1.0, max(x_values) / width),
                        min(1.0, max(y_values) / height),
                    ],
                })
            payload = {"entries": entries}
        except Exception as exc:
            payload = {"error": str(exc)[:300]}
        sys.stdout.write("PLATE_RESULT\t" + json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
