"""Train the hand-shake blur classifier used on exterior four-corner photos.

Real blurry photos are scarce, so the model learns from curated sharp photos
and copies of them degraded with synthetic hand shake (motion blur), defocus
and doubled "ghost" edges, plus night darkening, sensor noise and JPEG on both
classes so that noise is not mistaken for sharpness. Human truths from the
learning inbox (task capture_sharpness) are added with a higher weight.

The hand-collected real blur folder is never trained on: together with a set
of normal four-corner photos it is the fixed exam. With --deploy the candidate
replaces models/blur_detector.json only when it blocks at least as many real
blurry photos and does not block more normal photos than the deployed model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.schemas.operations import LearningTask  # noqa: E402
from app.services.blur_classifier import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_PATH,
    BlurClassifier,
    blur_features,
    resize_to_working,
)

DEFAULT_REMAP: list[str] = []
DEFAULT_BOX_CACHE = ROOT / "data" / "active_learning" / "blur_vehicle_boxes.json"
FOUR_CORNER = re.compile(r"_0[1-4]\.(png|jpe?g)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path: Path) -> np.ndarray | None:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else resize_to_working(image)


class VehicleBoxes:
    """Vehicle boxes keyed by file hash, cached because detection is the slow part."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.boxes: dict[str, list[float] | None] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        )
        self._detector = None

    def get(self, path: Path) -> list[float] | None:
        key = sha256(path)
        if key not in self.boxes:
            if self._detector is None:
                from app.services.vehicle_coverage import get_vehicle_coverage_detector
                self._detector = get_vehicle_coverage_detector()
            self.boxes[key] = self._detector.predict(path.read_bytes()).bounding_box or None
        return self.boxes[key]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.boxes), encoding="utf-8")


# ---------- synthetic degradations ----------
class Degrader:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _darken(self, image: np.ndarray) -> np.ndarray:
        gamma = self.rng.uniform(1.4, 2.4)
        out = 255 * (image / 255.0) ** gamma * self.rng.uniform(0.6, 1.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _noise_jpeg(self, image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32) + self.rng.normal(0, self.rng.uniform(0, 7), image.shape)
        quality = int(self.rng.integers(45, 92))
        ok, buffer = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _motion_kernel(self) -> np.ndarray:
        length = int(self.rng.integers(5, 30))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), self.rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        return kernel / max(kernel.sum(), 1e-6)

    def _defocus_kernel(self) -> np.ndarray:
        radius = int(self.rng.integers(2, 6))
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        return kernel / kernel.sum()

    def _ghost_kernel(self) -> np.ndarray:
        """A twitch during exposure: 2-3 discrete positions give doubled edges."""
        size = 31
        kernel = np.zeros((size, size), np.float32)
        angle = self.rng.uniform(0, np.pi)
        points = [(0.0, 0.0)]
        for _ in range(int(self.rng.integers(2, 4)) - 1):
            distance = self.rng.uniform(3, 14)
            direction = angle + self.rng.normal(0, 0.3)
            points.append((distance * np.cos(direction), distance * np.sin(direction)))
        for index, (dx, dy) in enumerate(points):
            weight = 1.0 if index == 0 else self.rng.uniform(0.35, 0.9)
            kernel[int(round(size // 2 + dy)), int(round(size // 2 + dx))] += weight
        kernel = cv2.GaussianBlur(kernel, (0, 0), self.rng.uniform(0.4, 1.0))
        return kernel / kernel.sum()

    def blurred(self, image: np.ndarray) -> np.ndarray:
        pick = self.rng.random()
        kernel = (self._ghost_kernel() if pick < 0.35
                  else self._motion_kernel() if pick < 0.8 else self._defocus_kernel())
        out = cv2.filter2D(image, -1, kernel)
        if self.rng.random() < 0.5:
            out = self._darken(out)
        return self._noise_jpeg(out)

    def sharp(self, image: np.ndarray) -> np.ndarray:
        out = self._darken(image) if self.rng.random() < 0.5 else image
        return self._noise_jpeg(out)


# ---------- data ----------
def remap_path(value: str, remaps: list[tuple[str, str]]) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for old, new in remaps:
        if value.startswith(old):
            candidate = Path(new + value[len(old):])
            if candidate.exists():
                return candidate
    return path


def unique_images(paths: list[Path], skip_hashes: set[str]) -> list[tuple[Path, str]]:
    seen = set(skip_hashes)
    output = []
    for path in paths:
        digest = sha256(path)
        if digest not in seen:
            seen.add(digest)
            output.append((path, digest))
    return output


def human_truths() -> list[tuple[str, Path, int]]:
    from app.services.operations_store import get_operations_store
    try:
        samples = get_operations_store().task_training_samples(
            LearningTask.CAPTURE_SHARPNESS, labels=("sharp", "blurred"),
        )
    except Exception as exc:  # the training script must still run without the database
        print(f"[warn] 讀不到人工真值：{exc}")
        return []
    return [(sample.id, path, int(sample.corrected_label == "blurred")) for sample, path, _ in samples]


def evaluate(model: BlurClassifier, features: np.ndarray, labels: np.ndarray,
             fail: float, warning: float) -> dict:
    scores = np.array([model.score_features(row) for row in features])
    blur, normal = scores[labels == 1], scores[labels == 0]
    return {
        "real_blur": int(len(blur)),
        "normal": int(len(normal)),
        "blocked_real_blur": int((blur >= fail).sum()),
        "flagged_real_blur": int((blur >= warning).sum()),
        "blocked_normal": int((normal >= fail).sum()),
        "flagged_normal": int((normal >= warning).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sharp-manifest", type=Path, required=True,
                        help="Authorized sharp-image manifest with a local_path column")
    parser.add_argument("--remap", action="append", default=None,
                        help="OLD=NEW prefix replacement for moved dataset folders")
    parser.add_argument("--real-blur-dir", type=Path, required=True,
                        help="Private held-out blurred-image directory")
    parser.add_argument("--normal-test-dir", type=Path, required=True,
                        help="Private held-out sharp-image directory")
    parser.add_argument("--box-cache", type=Path, default=DEFAULT_BOX_CACHE)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / f"blur_detector_{datetime.now():%Y%m%d}")
    parser.add_argument("--human-weight", type=float, default=3.0)
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fail", type=float, default=0.75)
    parser.add_argument("--warning", type=float, default=0.50)
    parser.add_argument("--deploy", action="store_true", help="replace models/blur_detector.json if the gate passes")
    parser.add_argument("--force", action="store_true", help="deploy even if the gate fails")
    args = parser.parse_args()
    remaps = [tuple(item.split("=", 1)) for item in (args.remap or DEFAULT_REMAP)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boxes = VehicleBoxes(args.box_cache)
    degrade = Degrader(args.seed)

    real_blur = unique_images(
        sorted(p for p in args.real_blur_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), set(),
    )
    exam_hashes = {digest for _, digest in real_blur}
    with open(args.sharp_manifest, encoding="utf-8-sig") as handle:
        curated = [remap_path(row["local_path"], remaps) for row in csv.DictReader(handle)]
    curated = unique_images([p for p in curated if p.exists()], exam_hashes)
    curated_hashes = {digest for _, digest in curated}
    normal_test = unique_images(
        sorted(p for p in args.normal_test_dir.rglob("*")
               if FOUR_CORNER.search(p.name) and "糊" not in str(p)),
        exam_hashes | curated_hashes,
    )
    humans = human_truths()
    print(f"sharp train={len(curated)} human truths={len(humans)} | "
          f"exam real blur={len(real_blur)} normal={len(normal_test)}", flush=True)

    rows, labels, weights = [], [], []
    for index, (path, _) in enumerate(curated):
        image = read_image(path)
        if image is None:
            continue
        box = boxes.get(path)
        for variant, label in ((degrade.sharp(image), 0), (image, 0),
                               (degrade.blurred(image), 1), (degrade.blurred(image), 1)):
            rows.append(blur_features(variant, box)); labels.append(label); weights.append(1.0)
        if index % 100 == 0:
            print(f"  features {index}/{len(curated)}", flush=True)
    for _, path, label in humans:
        image = read_image(path)
        if image is not None:
            rows.append(blur_features(image, boxes.get(path))); labels.append(label)
            weights.append(args.human_weight)
    x, y, w = np.stack(rows), np.array(labels), np.array(weights)

    scaler = StandardScaler().fit(x, sample_weight=w)
    logreg = LogisticRegression(C=args.C, max_iter=5000).fit(scaler.transform(x), y, sample_weight=w)
    version = f"blur-logreg-{datetime.now():%Y%m%d-%H%M}"
    candidate = {
        "version": version,
        "features": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training": {"curated_sharp": len(curated), "synthetic_per_photo": 4,
                     "human_truths": len(humans), "human_weight": args.human_weight, "C": args.C},
    }
    candidate_path = args.output_dir / f"{version}.json"
    candidate_path.write_text(json.dumps(candidate, indent=1), encoding="utf-8")

    exam_rows, exam_labels, exam_paths = [], [], []
    for group, label in ((real_blur, 1), (normal_test, 0)):
        for path, _ in group:
            image = read_image(path)
            if image is not None:
                exam_rows.append(blur_features(image, boxes.get(path)))
                exam_labels.append(label); exam_paths.append(str(path))
    boxes.save()
    exam_x, exam_y = np.stack(exam_rows), np.array(exam_labels)

    candidate_model = BlurClassifier(candidate_path)
    report = {"candidate": version,
              "thresholds": {"fail": args.fail, "warning": args.warning},
              "candidate_exam": evaluate(candidate_model, exam_x, exam_y, args.fail, args.warning)}
    deployed = BlurClassifier(MODEL_PATH)
    if deployed.available:
        try:
            report["deployed_exam"] = evaluate(deployed, exam_x, exam_y, args.fail, args.warning)
        except ValueError as exc:  # feature layout changed: the old model cannot be compared
            report["deployed_error"] = str(exc)
    report["per_image"] = [
        {"path": path, "label": "blur" if label else "normal",
         "probability": round(candidate_model.score_features(row), 4)}
        for path, label, row in zip(exam_paths, exam_labels, exam_rows, strict=True)
    ]

    new, old = report["candidate_exam"], report.get("deployed_exam")
    tolerance = max(1, int(0.005 * new["normal"]))
    gate_passed = old is None or (
        new["blocked_real_blur"] >= old["blocked_real_blur"]
        and new["blocked_normal"] <= old["blocked_normal"] + tolerance
    )
    report["gate_passed"] = gate_passed
    deployed_now = args.deploy and (gate_passed or args.force)
    report["deployed"] = deployed_now
    if deployed_now:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(candidate, indent=1), encoding="utf-8")
        if humans:
            from app.services.operations_store import get_operations_store
            get_operations_store().mark_learning_samples_trained([sample_id for sample_id, _, _ in humans])
    (args.output_dir / f"{version}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    def line(name: str, stats: dict) -> str:
        return (f"{name}: 擋下真實模糊 {stats['blocked_real_blur']}/{stats['real_blur']}，"
                f"至少提醒 {stats['flagged_real_blur']}/{stats['real_blur']}；"
                f"誤擋一般照 {stats['blocked_normal']}/{stats['normal']}，"
                f"提醒 {stats['flagged_normal']}/{stats['normal']}")
    print(line("候選模型", new))
    if old:
        print(line("線上模型", old))
    print(f"把關{'通過' if gate_passed else '未通過'}；{'已部署到 ' + str(MODEL_PATH) if deployed_now else '未部署'}")
    print(f"報告：{args.output_dir / (version + '_report.json')}")


if __name__ == "__main__":
    main()
