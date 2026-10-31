"""Export the damage-pair training material as one zip for GPU training elsewhere.

The zip holds exactly what ``run_damage_pair_training`` would train and test on:
human-truth pairs (red/blue boxes), the no-claim weak pairs with their fixed
train/val/test split, and the _TEST claim folder. Photos are renamed to their
SHA-256 so no file name carries a plate, and each photo's vehicle crop box is
computed here with the same detector the API uses, so a model trained on the
export sees the same crops as online inference.

    python scripts/export_damage_pair_dataset.py
    # -> outputs/damage_pair_dataset/damage_pair_dataset_<date>.zip

The photos themselves still show plates and are external source data: keep the zip
in storage your team is allowed to use.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.services import damage_pair_specialist as dps  # noqa: E402
from app.services.operations_store import get_operations_store  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "damage_pair_dataset" / f"damage_pair_dataset_{date.today():%Y%m%d}.zip"
BOX_CACHE = PROJECT_ROOT / "data" / "active_learning" / "damage_pair_vehicle_boxes.json"
BOX_KEYS = ("x", "y", "width", "height", "label", "shape", "points", "normal_reference", "image_slot", "kind", "notes", "auto_refinement")


def _boxes(items: list[dict]) -> list[dict]:
    return [{key: item[key] for key in BOX_KEYS if key in item} for item in items]


class _Photos:
    """Collects photos by content hash and their vehicle crop boxes."""

    def __init__(self) -> None:
        self.paths: dict[str, Path] = {}
        self.cache: dict[str, list[float]] = (
            json.loads(BOX_CACHE.read_text(encoding="utf-8")) if BOX_CACHE.is_file() else {}
        )

    def add(self, path: Path) -> str:
        digest = dps._sha256(path)
        name = f"images/{digest[:24]}{path.suffix.lower()}"
        self.paths[name] = path
        if digest not in self.cache:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            self.cache[digest] = [round(value, 5) for value in dps.PatchEncoder._vehicle_box(image)]
        return name

    def box(self, name: str) -> list[float]:
        return self.cache[dps._sha256(self.paths[name])]

    def save_cache(self) -> None:
        BOX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BOX_CACHE.write_text(json.dumps(self.cache), encoding="utf-8")


def _supplemental_train_pairs(excluded: set[str]) -> list[tuple[Path, Path]]:
    """Supplemental 0921 train-split pairs: no claim record, not reviewed by a human."""
    import csv

    if not dps.SUPPLEMENTAL_PAIRS.is_file():
        return []
    found = []
    with dps.SUPPLEMENTAL_PAIRS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                continue
            pickup = dps._supplemental_path(row["pickup_external_path"])
            returned = dps._supplemental_path(row["return_external_path"])
            if pickup is None or returned is None or not pickup.is_file() or not returned.is_file():
                continue
            if dps._sha256(pickup) in excluded or dps._sha256(returned) in excluded:
                continue
            found.append((pickup, returned))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extra-negatives",
        action="store_true",
        help="Also export the unreviewed supplemental train pairs as weak split 'extra' "
        "(the API does not train on them; damage_pair_autolearn.py --extra-negatives can)",
    )
    args = parser.parse_args()

    store = get_operations_store()
    excluded = dps.excluded_folder_hashes()
    humans = dps._human_pairs(store, excluded)
    holdout = dps.holdout_pairs(store)
    reviewed = excluded | {dps._sha256(path) for pair in humans for path in (pair.pickup, pair.returned)}
    weak, notes = dps._weak_pairs(reviewed)
    photos = _Photos()
    pairs: list[dict] = []
    total = len(humans) + len(holdout) + len(weak)

    def progress() -> None:
        done = len(pairs)
        if done % 25 == 0 or done == total:
            print(f"{done}/{total} pairs", flush=True)
            photos.save_cache()

    for pair in humans:
        sample = pair.sample
        pairs.append(
            {
                "id": sample.id,
                "group": "human",
                "label": sample.corrected_label,
                "pickup": photos.add(pair.pickup),
                "return": photos.add(pair.returned),
                "issue_boxes": _boxes(pair.issue_boxes),
                "normal_boxes": _boxes(pair.normal_boxes),
                "return_normal_boxes": _boxes(
                    [
                        item for item in dps._annotations(sample)
                        if item.get("image_slot") == "secondary" and item.get("kind") == "hard_negative"
                    ]
                ),
                "angle": sample.expected_label,
                "lighting": sample.lighting,
                "vehicle_identity": (sample.metadata.get("human_review") or {}).get("vehicle_identity"),
                "reference_regions": _boxes([item for item in dps._annotations(sample) if item.get("kind") == "reference"]),
            }
        )
        progress()
    for pair in holdout:
        pairs.append(
            {
                "id": f"holdout:{pair.key}",
                "group": "holdout",
                "label": pair.truth,
                "pickup": photos.add(pair.pickup),
                "return": photos.add(pair.returned),
                "issue_boxes": _boxes(pair.issue_boxes),
                "note": pair.note,
            }
        )
        progress()
    for pair in weak:
        pairs.append(
            {
                "id": f"weak:{dps._sha256(pair.returned)[:12]}",
                "group": "weak",
                "label": "no_new_damage",
                "split": pair.split,
                "source": pair.source,
                "pickup": photos.add(pair.pickup),
                "return": photos.add(pair.returned),
            }
        )
        progress()
    if args.extra_negatives:
        excluded_extra = reviewed | {dps._sha256(p.returned) for p in weak}
        for pair in _supplemental_train_pairs(excluded_extra):
            pairs.append(
                {
                    "id": f"weak:{dps._sha256(pair[1])[:12]}",
                    "group": "weak",
                    "label": "no_new_damage",
                    "split": "extra",
                    "source": "supplemental_train",
                    "pickup": photos.add(pair[0]),
                    "return": photos.add(pair[1]),
                }
            )
            if len(pairs) % 50 == 0:
                print(f"{len(pairs)} pairs (extra)", flush=True)
                photos.save_cache()
    photos.save_cache()
    for pair in pairs:
        pair["pickup_vehicle_box"] = photos.box(pair["pickup"])
        pair["return_vehicle_box"] = photos.box(pair["return"])

    manifest = {
        "schema": "guardian_damage_pair_dataset_v2",
        "coordinate_space": "normalized_image",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "human = 人工真值（issue_boxes 為還車照紅框、normal_boxes 為取車照藍框、"
            "return_normal_boxes 為還車照藍框：正常車身、陰影、反光、接縫等）；"
            "holdout = _TEST 固定測試集，只能評估不能訓練；"
            "weak = 無理賠配對弱負樣本，train 可訓練、val 校準門檻、test 量測誤報。"
            "框座標為原圖（EXIF 轉正後）的 0–1 比例；vehicle_box 為 API 同一偵測器裁切框。"
        ),
        "preprocess": {"crop_margin": dps.CROP_MARGIN, "patch": dps.PATCH, "letterbox": "centered_black"},
        "current_ensemble": [list(member) for member in dps.ENSEMBLE],
        "notes": notes,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        for name, path in sorted(photos.paths.items()):
            archive.write(path, name)
    counts = {group: sum(pair["group"] == group for pair in pairs) for group in ("human", "holdout", "weak")}
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(json.dumps({"output": str(args.output), "pairs": counts, "photos": len(photos.paths),
                      "size_mb": round(size_mb, 1), "notes": notes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
