"""Prepare auditable guardian view-classification manifests and image samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import ssl
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from PIL import Image
import truststore

CORE_TYPES = {
    1: "left_front",
    2: "right_front",
    3: "left_rear",
    4: "right_rear",
    10: "interior_front",
    11: "interior_rear",
}
VIEW_NAMES = {
    "左前": "left_front",
    "右前": "right_front",
    "左後": "left_rear",
    "右後": "right_rear",
}
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _column_name(reference: str) -> str:
    match = re.match(r"([A-Z]+)", reference)
    return match.group(1) if match else reference


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.itertext()) for node in root.findall(f"{{{SHEET_NS}}}si")]


def read_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read values without modifying the source workbook or evaluating formulas."""
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    decoded_rows: list[dict[str, str]] = []
    for row in root.findall(f".//{{{SHEET_NS}}}row"):
        decoded: dict[str, str] = {}
        for cell in row.findall(f"{{{SHEET_NS}}}c"):
            column = _column_name(cell.attrib.get("r", ""))
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{{{SHEET_NS}}}v")
            if cell_type == "inlineStr":
                value = "".join(cell.itertext())
            elif value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared[int(value_node.text or 0)]
            else:
                value = value_node.text or ""
            decoded[column] = value.strip()
        decoded_rows.append(decoded)
    return decoded_rows


def stable_split(car_no: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{car_no}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def build_remote_manifest(
    rows: list[dict[str, str]], seed: int
) -> list[dict[str, str]]:
    headers = rows[0]
    header_to_column = {value: column for column, value in headers.items()}
    required = {"order_number", "CarNo", "ImageType", "下載網址"}
    missing = required - set(header_to_column)
    if missing:
        raise ValueError(f"Workbook is missing columns: {sorted(missing)}")

    records: list[dict[str, str]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        try:
            image_type = int(float(row.get(header_to_column["ImageType"], "")))
        except ValueError:
            continue
        label = CORE_TYPES.get(image_type)
        if not label:
            continue
        car_no = row.get(header_to_column["CarNo"], "").strip()
        image_url = row.get(header_to_column["下載網址"], "").strip()
        if not car_no or not image_url:
            continue
        records.append(
            {
                "row_number": str(row_number),
                "order_number": row.get(header_to_column["order_number"], ""),
                "car_no": car_no,
                "image_type": str(image_type),
                "label": label,
                "image_url": image_url,
                "split": stable_split(car_no, seed),
            }
        )
    return records


def build_local_manifest(folder: Path, seed: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    car_pattern = re.compile(r"([A-Z]{3})[- ]?(\d{4})", re.IGNORECASE)
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".jfif", ".png", ".webp"}:
            continue
        label = next(
            (value for key, value in VIEW_NAMES.items() if key in path.stem), None
        )
        car_match = car_pattern.search(path.stem)
        if not label or not car_match:
            continue
        car_no = f"{car_match.group(1).upper()}-{car_match.group(2)}"
        stage = (
            "borrow"
            if "借" in path.stem
            else "return"
            if "還" in path.stem
            else "unknown"
        )
        ancestor_names = {parent.name for parent in path.parents}
        if "進行索賠資料" in ancestor_names:
            claim = "positive"
        elif "沒有進行索賠" in ancestor_names:
            claim = "negative"
        else:
            claim = "unknown"
        records.append(
            {
                "path": str(path.resolve()),
                "local_path": str(path.resolve()),
                "car_no": car_no,
                "label": label,
                "stage": stage,
                "claim_label": claim,
                "pair_key": f"{car_no}:{label}",
            }
        )
    car_status = {record["car_no"]: record["claim_label"] for record in records}
    split_by_car: dict[str, str] = {}
    for status_index, status in enumerate(("negative", "positive")):
        cars = sorted(car for car, value in car_status.items() if value == status)
        random.Random(seed + status_index).shuffle(cars)
        train_end = round(len(cars) * 0.70)
        val_end = train_end + round(len(cars) * 0.15)
        for index, car_no in enumerate(cars):
            split_by_car[car_no] = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
    for record in records:
        record["split"] = split_by_car[record["car_no"]]
    return records


def balanced_sample(
    records: list[dict[str, str]], per_class: int, seed: int
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["label"], record["split"])].append(record)

    rng = random.Random(seed)
    targets = {
        "train": round(per_class * 0.70),
        "val": round(per_class * 0.15),
        "test": per_class - round(per_class * 0.70) - round(per_class * 0.15),
    }
    selected: list[dict[str, str]] = []
    for label in CORE_TYPES.values():
        for split, target in targets.items():
            candidates = grouped[(label, split)]
            rng.shuffle(candidates)
            if len(candidates) < target:
                raise ValueError(
                    f"Not enough {label}/{split}: {len(candidates)} < {target}"
                )
            selected.extend(candidates[:target])
    return sorted(
        selected, key=lambda item: (item["split"], item["label"], item["car_no"])
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _download(
    record: dict[str, str], images_dir: Path
) -> tuple[dict[str, str], str | None]:
    digest = hashlib.sha256(record["image_url"].encode()).hexdigest()[:20]
    destination = images_dir / record["split"] / record["label"] / f"{digest}.png"
    enriched = dict(record)
    enriched["local_path"] = str(destination.resolve())
    if destination.exists():
        return enriched, None

    destination.parent.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    for attempt in range(3):
        try:
            request = Request(
                record["image_url"], headers={"User-Agent": "Guardian/0.2"}
            )
            with urlopen(request, timeout=30, context=SSL_CONTEXT) as response:  # noqa: S310
                content = response.read(20 * 1024 * 1024)
            with Image.open(BytesIO(content)) as image:
                image.convert("RGB").save(destination, format="PNG", optimize=True)
            return enriched, None
        except (HTTPError, URLError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    return enriched, error


def download_selected(
    selected: list[dict[str, str]], images_dir: Path, workers: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download, row, images_dir): row for row in selected}
        for index, future in enumerate(as_completed(futures), start=1):
            enriched, error = future.result()
            if error:
                failures.append({**enriched, "error": error})
            else:
                completed.append(enriched)
            if index % 50 == 0 or index == len(selected):
                print(f"Downloaded {index}/{len(selected)} (failures={len(failures)})")
    return completed, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, type=Path)
    parser.add_argument("--local-images", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/view_dataset"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    workbook_rows = read_first_sheet(args.xlsx)
    remote = build_remote_manifest(workbook_rows, args.seed)
    local = build_local_manifest(args.local_images, args.seed)
    selected = balanced_sample(remote, args.per_class, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "remote_core_manifest.csv", remote)
    write_csv(args.output_dir / "local_damage_manifest.csv", local)
    write_csv(args.output_dir / "selected_manifest.csv", selected)

    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    if args.download:
        completed, failures = download_selected(
            selected, args.output_dir / "images", args.workers
        )
        write_csv(args.output_dir / "downloaded_manifest.csv", completed)
        write_csv(args.output_dir / "download_failures.csv", failures)

    summary = {
        "source_workbook": str(args.xlsx.resolve()),
        "workbook_rows_including_header": len(workbook_rows),
        "core_remote_rows": len(remote),
        "core_type_counts": Counter(row["label"] for row in remote),
        "unique_orders": len({row["order_number"] for row in remote}),
        "unique_cars": len({row["car_no"] for row in remote}),
        "selected_rows": len(selected),
        "selected_counts": {
            split: Counter(row["label"] for row in selected if row["split"] == split)
            for split in ("train", "val", "test")
        },
        "local_labeled_images": len(local),
        "local_pair_keys": len({row["pair_key"] for row in local}),
        "downloaded": len(completed),
        "download_failures": len(failures),
        "seed": args.seed,
        "split_policy": "SHA256(seed:car_no), 70/15/15; same car never crosses splits",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
