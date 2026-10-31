# -*- coding: utf-8 -*-
"""guardian 前後車損比對：自主學習訓練與模型比較

本地（CPU 即可，有 CUDA 會自動使用）：
    python scripts/export_damage_pair_dataset.py            # 產生訓練資料 zip
    python scripts/damage_pair_autolearn.py --compare        # 每個候選模型各測一次，列出差異
    python scripts/damage_pair_autolearn.py                  # 自主學習，最多 5 輪
    python scripts/import_damage_pair_model.py outputs/damage_pair_autolearn/damage_pair_specialist.pt

    --models "S+B contrast,L contrast" 只測指定模型；ViT-g 在 CPU 太慢，預設只在有 GPU 時納入。

說明：docs/damage-pair-autolearn.md。

自主學習機制：
    * 每一輪 = 一次完整測試（分組交叉驗證 + 固定測試集 + 一次還車檢查計時）。
    * 每輪把失敗原因寫進 <work>/failure_memory.json，下一輪與下次執行都記得：
        missed_low_score        紅框內有反應但分數不到門檻 → 該樣本與該車損類型加權
        missed_wrong_location   最高分落在紅框外 → 錯誤位置記為困難負樣本，樣本加權
        detected_wrong_location 有抓到但位置錯 → 錯誤位置記為困難負樣本
        false_alarm             人工確認無新增車損卻被抓 → 最高分位置記為困難負樣本
        too_many_false_alarms   無理賠配對誤報太多 → 從「訓練用」無理賠配對挖困難負樣本
        holdout_missed          固定測試集漏抓 → 換更強的模型（測試集永遠不拿來訓練）
        too_slow                超過時間上限 → 這個骨幹永久封鎖，不再嘗試
      困難負樣本以「原圖座標」記錄，換骨幹或解析度也能沿用；同一模型＋同一組補強失敗過
      就不會再原樣重跑；補強讓同一模型變差時會撤銷並記為有害，之後不再加回。
    * 門檻自動挑「召回率達標下最少送人工審核」的值，只用交叉驗證與校準集挑，
      不偷看測試集。
    * 同時達成 召回率 ≥ 60%、準確度 ≥ 95%、一次還車檢查（4 個角度）≤ 30 秒
      就立刻選它並停止；5 輪都沒達成則選最好的一輪當試行模型並標註未達標項目。

指標定義（全部是模型沒看過的樣本）：
    召回率 = (交叉驗證留出的人工車損 + _TEST 固定測試集) 中被抓到的比例
    準確度 = 上述車損 + 交叉驗證留出的人工無車損 + 無理賠 test 配對，整體判對比例
    人工審核率 = 無理賠 test 配對被送人工的比例
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else None
LOCAL_HUB = PROJECT_ROOT / ".cache" / "torch" if PROJECT_ROOT else None

# ============================================================== 目標
TARGET_RECALL = 0.60
TARGET_ACCURACY = 0.95
MAX_INSPECTION_SECONDS = 30.0  # 一次還車檢查 = 4 個角度的取還車配對
PAIRS_PER_INSPECTION = 4
MAX_ROUNDS = 5

# 候選模型（皆為免費開源、可商用授權）。feature: plain = 只看還車照；contrast = 還車照
# 區塊再減去取車照最相似的區塊，原本就有的痕跡、反光會被抵銷。
# backbone 命名：dinov2_* = Meta DINOv2（Apache-2.0，torch.hub）；timm:<名稱> = timm
# 模型（Hugging Face 下載）；hf-vit:<repo> = transformers ViT。所有成員的區塊格數都是
# 32×32（DINOv2 448/14、其他 512/16），才能在同一張圖上平均。
def _member(backbone: str, size: int, feature: str = "contrast", top: float = 1.0, weight: float = 1.0) -> dict:
    return {"backbone": backbone, "input_size": size, "feature": feature, "top_fraction": top, "weight": weight}


CARDD_VIT = "hf-vit:beingamit99/car_damage_detection"      # MIT；CarDD 車損六類微調的 ViT-B/16
CONVNEXT = "timm:convnext_base.fb_in22k"                   # Apache-2.0；ImageNet-22k CNN
SIGLIP2 = "timm:vit_base_patch16_siglip_512.v2_webli"      # Apache-2.0；Google SigLIP2 圖文預訓練
# 順序＝自主學習嘗試順序：2026-09-28 本機比較（3 次重複交叉驗證）由強到弱排列；
# local4（只跟取車照同位置 ±4 格比對）明顯優於整張比對。
# 現行做法放第一個只當基準，不會被自主學習選到（START_RUNG 從 1 開始）。
ALL_MODELS = [
    {"name": "S+B plain (現行做法)", "members": [
        _member("dinov2_vits14", 448, "plain", 1.0, 0.5), _member("dinov2_vitb14", 448, "plain", 0.3, 0.5)]},
    {"name": "SigLIP2+S+B local4", "members": [
        _member(SIGLIP2, 512, "local4", weight=0.5), _member("dinov2_vits14", 448, "local4", top=1.0, weight=0.25),
        _member("dinov2_vitb14", 448, "local4", top=0.3, weight=0.25)]},
    {"name": "SigLIP2 local4", "members": [_member(SIGLIP2, 512, "local4")]},
    {"name": "SigLIP2 plain+local4", "members": [
        _member(SIGLIP2, 512, "plain", weight=0.5), _member(SIGLIP2, 512, "local4", weight=0.5)]},
    {"name": "SigLIP2 local8", "members": [_member(SIGLIP2, 512, "local8")]},
    {"name": "SigLIP2 plain+contrast", "members": [
        _member(SIGLIP2, 512, "plain", weight=0.5), _member(SIGLIP2, 512, "contrast", weight=0.5)]},
    {"name": "SigLIP2 plain", "members": [_member(SIGLIP2, 512, "plain")]},
    {"name": "SigLIP2+S+B contrast", "members": [
        _member(SIGLIP2, 512, weight=0.5), _member("dinov2_vits14", 448, top=1.0, weight=0.25),
        _member("dinov2_vitb14", 448, top=0.3, weight=0.25)]},
    {"name": "SigLIP2 contrast", "members": [_member(SIGLIP2, 512)]},
    {"name": "SigLIP2+ConvNeXt contrast", "members": [
        _member(SIGLIP2, 512, weight=0.5), _member(CONVNEXT, 512, weight=0.5)]},
    {"name": "S+B contrast", "members": [
        _member("dinov2_vits14", 448, top=1.0, weight=0.5), _member("dinov2_vitb14", 448, top=0.3, weight=0.5)]},
    {"name": "L contrast", "members": [_member("dinov2_vitl14", 448)]},
    {"name": "ConvNeXt contrast", "members": [_member(CONVNEXT, 512)]},
    {"name": "L+ConvNeXt contrast", "members": [
        _member("dinov2_vitl14", 448, weight=0.5), _member(CONVNEXT, 512, weight=0.5)]},
    {"name": "B+CarDD contrast", "members": [
        _member("dinov2_vitb14", 448, top=0.3, weight=0.5), _member(CARDD_VIT, 512, weight=0.5)]},
    {"name": "CarDD-ViT contrast", "members": [_member(CARDD_VIT, 512)]},
    {"name": "g contrast", "members": [_member("dinov2_vitg14", 448)]},
]
LADDER = list(ALL_MODELS)  # main() 依 --models 與有無 GPU 篩選
START_RUNG = 1
CPU_ONLY_SKIP = {"g contrast"}  # ViT-g 在 CPU 抽特徵要數小時，沒有 GPU 時預設不測

# 與 API 的 damage_pair_specialist.py 相同，請勿修改，否則線上推論會對不上。
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
AUGMENTED_VARIANTS = 3
NEGATIVE_PATCHES_PER_IMAGE = 48
BACKGROUND_PATCHES = 160
MAX_FOLDS = 10
# 重複交叉驗證：(分折洗牌種子, 取樣種子)。第一組與 API 自動訓練相同。
CV_REPEATS = ((7, 20260927), (11, 11), (23, 23))
L2 = 0.01
CALIBRATION_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
MAX_POSITIVE_BOOST = 8.0
MAX_TYPE_BOOST = 3.0
HARD_NEGATIVE_MASS = 1.0
# 還車照藍框（正常車身、陰影、反光、接縫）每個框當一組困難負樣本的權重；0 = 不用（2026-10-11 前的做法）。
# 框越小，框內每個區塊分到的權重越大，所以陰影、接縫這類小框會被特別記住。
RETURN_NORMAL_MASS = 0.0
# 這個面積以上的「正常車身」藍框視為整台車的範圍，用來量測誤報是否落在車身外。
BODY_BOX_MIN_AREA = 0.15
HARD_NEGATIVE_RADIUS = 1.5  # 以區塊為單位
WEAK_MINING_PER_ROUND = 15
# ==============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configure() 設定下列路徑。
DATASET_ZIP: Path = Path()
WORK: Path = Path()
DATA: Path = Path()
FEATURES: Path = Path()
MEMORY_PATH: Path = Path()


def configure(dataset_zip, work_dir, data_dir=None, features_dir=None) -> None:
    """dataset_zip: 匯出的 zip；work_dir: 記憶、報告、模型輸出；data/features 預設放在 work_dir 下。"""
    global DATASET_ZIP, WORK, DATA, FEATURES, MEMORY_PATH
    DATASET_ZIP = Path(dataset_zip)
    WORK = Path(work_dir)
    DATA = Path(data_dir) if data_dir else WORK / "data"
    FEATURES = Path(features_dir) if features_dir else WORK / "features"
    MEMORY_PATH = WORK / "failure_memory.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# ------------------------------------------------------------------ 資料
EXTRA_NEGATIVES = 0.0  # --extra-negatives：未人工確認的補充配對，取多少比例當訓練負樣本


def load_manifest() -> dict:
    if not (DATA / "manifest.json").is_file():
        DATA.mkdir(parents=True, exist_ok=True)
        log(f"解壓縮 {DATASET_ZIP}")
        with zipfile.ZipFile(DATASET_ZIP) as archive:
            archive.extractall(DATA)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    extra = sorted((p for p in manifest["pairs"] if p.get("split") == "extra"), key=lambda p: p["id"])
    keep = extra[: round(len(extra) * EXTRA_NEGATIVES)]
    for pair in keep:
        pair["split"] = "train"
    if extra:
        log(f"補充無損配對：共 {len(extra)} 組，本次取 {len(keep)} 組當訓練負樣本")
    pairs = [p for p in manifest["pairs"] if p.get("split") != "extra"] + keep
    for pair in pairs:
        pair.setdefault("return_normal_boxes", [])  # 2026-10-11 前匯出的 zip 沒有還車照藍框
    groups = {group: [p for p in pairs if p["group"] == group] for group in ("human", "holdout", "weak")}
    log("資料：人工真值 {}（車損 {}）、固定測試集 {}、無理賠配對 {}".format(
        len(groups["human"]), sum(p["label"] == "confirmed_new_damage" for p in groups["human"]),
        len(groups["holdout"]), len(groups["weak"])))
    return groups


def open_image(name: str) -> Image.Image:
    with Image.open(DATA / name) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def augment(image: Image.Image, seed: str) -> tuple[Image.Image, bool]:
    rng = random.Random(seed)
    flip = rng.random() < 0.5
    if flip:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    return image, flip


def backbone_stride(name: str) -> int:
    return 14 if name.startswith("dinov2_") else 16


def letterbox(image: Image.Image, box: list[float], size: int, stride: int):
    """Same crop / resize / centred black canvas as PatchEncoder.encode; pixels in [0, 1]."""
    bx0, by0, bx1, by1 = box
    width, height = image.size
    x0, y0 = round(bx0 * width), round(by0 * height)
    crop = image.crop((x0, y0, max(x0 + 1, round(bx1 * width)), max(y0 + 1, round(by1 * height))))
    scale = size / max(crop.size)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size))
    left, top = (size - resized.width) // 2, (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    steps = (np.arange(size // stride) + 0.5) * stride
    cx, cy = np.meshgrid(steps, steps)
    nx = ((cx - left) / resized.width).ravel()
    ny = ((cy - top) / resized.height).ravel()
    valid = (nx >= 0) & (nx <= 1) & (ny >= 0) & (ny <= 1)
    centers = np.stack([bx0 + nx * (bx1 - bx0), by0 + ny * (by1 - by0)], axis=1).astype(np.float32)
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return array, centers, valid


class Encoded:
    __slots__ = ("tokens", "centers", "valid", "flip")

    def __init__(self, tokens, centers, valid, flip):
        self.tokens, self.centers, self.valid, self.flip = tokens, centers, valid, flip


class Backbone:
    """Frozen feature extractor: [0, 1] RGB batch in, L2-normalised patch tokens out."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stride = backbone_stride(name)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if name.startswith("dinov2_"):
            repository = LOCAL_HUB / "facebookresearch_dinov2_main" if LOCAL_HUB else None
            if repository is not None and repository.exists():
                torch.hub.set_dir(str(LOCAL_HUB))
                model = torch.hub.load(str(repository), name, source="local", pretrained=True)
            else:
                model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True, trust_repo=True)
            self.kind, self.dim = "dinov2", model.embed_dim
        elif name.startswith("timm:"):
            import timm

            arch = name.split(":", 1)[1]
            if arch.startswith("convnext"):
                model = timm.create_model(arch, pretrained=True, features_only=True, out_indices=(2,))
                self.kind, self.dim = "cnn", model.feature_info.channels()[0]
            else:
                model = timm.create_model(arch, pretrained=True, num_classes=0)
                self.kind, self.dim, self.prefix = "vit", model.embed_dim, model.num_prefix_tokens
            mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
        elif name.startswith("hf-vit:"):
            from transformers import AutoImageProcessor, ViTModel

            repo = name.split(":", 1)[1]
            model = ViTModel.from_pretrained(repo, add_pooling_layer=False)
            processor = AutoImageProcessor.from_pretrained(repo)
            mean, std = processor.image_mean, processor.image_std
            self.kind, self.dim = "hf-vit", model.config.hidden_size
        else:
            raise ValueError(f"unknown backbone {name}")
        self.model = model.eval().to(DEVICE)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels.to(DEVICE) - self.mean) / self.std
        with torch.inference_mode():
            if self.kind == "dinov2":
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]
            elif self.kind == "cnn":
                tokens = self.model(x)[0].flatten(2).transpose(1, 2)
            elif self.kind == "vit":
                tokens = self.model.forward_features(x)[:, self.prefix:]
            else:
                tokens = self.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
            return torch.nn.functional.normalize(tokens.float(), dim=2)


_backbones: dict[str, Backbone] = {}


def backbone(name: str) -> Backbone:
    if name not in _backbones:
        log(f"載入 {name}")
        _backbones[name] = Backbone(name)
    return _backbones[name]


def image_jobs(groups: dict) -> list[tuple[str, str, int, str, list[float]]]:
    """(key, photo, variant, seed, box) for every image any recipe needs."""
    jobs = {}
    for pair in groups["human"] + groups["holdout"] + groups["weak"]:
        for slot in ("pickup", "return"):
            key = f"{pair[slot]}|v0"
            jobs[key] = (key, pair[slot], 0, "", pair[f"{slot}_vehicle_box"])
        if pair["group"] == "human" and pair["label"] == "confirmed_new_damage":
            for variant in range(1, AUGMENTED_VARIANTS + 1):
                key = f"{pair['return']}|v{variant}|{pair['id']}"
                jobs[key] = (key, pair["return"], variant, f"{pair['id']}:{variant}", pair["return_vehicle_box"])
    return list(jobs.values())


def encode_all(groups: dict, name: str, size: int) -> dict[str, Encoded]:
    """Patch tokens for every image, kept as a float16 memmap on disk.

    Rows already encoded are reused; a
    second run in the same session reuses the local files."""
    folder = FEATURES / f"{name.replace(':', '_').replace('/', '_')}_{size}"
    jobs = image_jobs(groups)
    keys = [job[0] for job in jobs]
    index_path = folder / "keys.json"
    old_keys = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    if old_keys != keys:
        old_rows = {key: row for row, key in enumerate(old_keys)}
        if old_rows:
            for suffix in ("tokens", "centers", "valid", "flip"):
                (folder / f"{suffix}.npy").replace(folder / f"{suffix}.old.npy")
            old_tokens = np.load(folder / "tokens.old.npy", mmap_mode="r")
            old_centers = np.load(folder / "centers.old.npy")
            old_valid = np.load(folder / "valid.old.npy")
            old_flip = np.load(folder / "flip.old.npy")
        model = backbone(name)
        grid = (size // model.stride) ** 2
        dim = model.dim
        folder.mkdir(parents=True, exist_ok=True)
        log(f"{name}@{size}：抽取 {sum(k not in old_rows for k in keys)} 張照片特徵（{len(old_rows)} 張沿用快取）")
        tokens = np.lib.format.open_memmap(folder / "tokens.npy", mode="w+", dtype=np.float16, shape=(len(jobs), grid, dim))
        centers = np.zeros((len(jobs), grid, 2), dtype=np.float32)
        valid = np.zeros((len(jobs), grid), dtype=bool)
        flips = np.zeros(len(jobs), dtype=bool)
        batch: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            output = model(torch.from_numpy(np.stack([item[1] for item in batch]))).cpu().numpy().astype(np.float16)
            for (row, _array), value in zip(batch, output):
                tokens[row] = value
            batch.clear()

        for row, (key, photo, variant, seed, box) in enumerate(jobs):
            if key in old_rows:
                old = old_rows[key]
                tokens[row], centers[row], valid[row], flips[row] = old_tokens[old], old_centers[old], old_valid[old], old_flip[old]
                continue
            image = open_image(photo)
            if variant:
                image, flips[row] = augment(image, seed)
                if flips[row]:
                    box = [1 - box[2], box[1], 1 - box[0], box[3]]
            array, centers[row], valid[row] = letterbox(image, box, size, model.stride)
            batch.append((row, array))
            if len(batch) == 8:
                flush()
            if (row + 1) % 200 == 0:
                log(f"  {row + 1}/{len(jobs)}")
        if batch:
            flush()
        tokens.flush()
        del tokens
        np.save(folder / "centers.npy", centers)
        np.save(folder / "valid.npy", valid)
        np.save(folder / "flip.npy", flips)
        index_path.write_text(json.dumps(keys), encoding="utf-8")
        if old_rows:
            del old_tokens
            for suffix in ("tokens", "centers", "valid", "flip"):
                (folder / f"{suffix}.old.npy").unlink()
    tokens = np.load(folder / "tokens.npy", mmap_mode="r")
    centers = np.load(folder / "centers.npy")
    valid = np.load(folder / "valid.npy")
    flips = np.load(folder / "flip.npy")
    return {key: Encoded(tokens[row], centers[row], valid[row], bool(flips[row])) for row, key in enumerate(keys)}


# ------------------------------------------------------------------ 特徵與模型
def local_window(mode: str) -> int | None:
    """``contrast`` = anywhere on the other photo; ``local<N>`` = within N grid cells of the same spot."""
    return int(mode[5:]) if mode.startswith("local") else None


def nearest_patches(query: Encoded, reference: Encoded, window: int | None = None) -> np.ndarray:
    """Index (into reference.tokens) of each query patch's most similar valid reference patch.

    With ``window`` only reference patches within that many cells of the same grid
    position count, so a dent is compared with the same panel at pickup instead of
    any clean paint elsewhere. Both photos are vehicle-cropped, so grid positions
    roughly line up; a mirrored augmentation mirrors the columns."""
    q = torch.from_numpy(np.asarray(query.tokens, dtype=np.float32)).to(DEVICE)
    r = torch.from_numpy(np.asarray(reference.tokens, dtype=np.float32)).to(DEVICE)
    similarity = q @ r.T
    similarity[:, torch.from_numpy(~reference.valid).to(DEVICE)] = -torch.inf
    best = similarity.argmax(1)
    if window is not None:
        size = int(round(np.sqrt(len(query.tokens))))
        index = torch.arange(size * size, device=DEVICE)
        rows, cols = index // size, index % size
        query_cols = size - 1 - cols if query.flip != reference.flip else cols
        far = ((rows[:, None] - rows[None]).abs() > window) | ((query_cols[:, None] - cols[None]).abs() > window)
        local = similarity.masked_fill(far, -torch.inf)
        found = torch.isfinite(local.max(1).values)
        best = torch.where(found, local.argmax(1), best)
    return best.cpu().numpy().astype(np.int32)


def features(query: Encoded, reference: Encoded, mode: str, nearest: np.ndarray | None = None) -> np.ndarray:
    """Same as the API's _features: plain tokens, or tokens + (token - matched other-photo token)."""
    tokens = np.asarray(query.tokens, dtype=np.float32)
    if mode == "plain":
        return tokens
    if nearest is None:
        nearest = nearest_patches(query, reference, local_window(mode))
    return np.concatenate([tokens, tokens - np.asarray(reference.tokens, dtype=np.float32)[nearest]], axis=1)


def in_boxes(encoded: Encoded, boxes: list[dict]) -> np.ndarray:
    hit = np.zeros(len(encoded.centers), dtype=bool)
    x = 1 - encoded.centers[:, 0] if encoded.flip else encoded.centers[:, 0]
    y = encoded.centers[:, 1]
    for box in boxes:
        if box.get("shape") == "polygon":
            # Keep this script standalone for GPU/Colab runs without the API package.
            points = box["points"]
            inside, boundary = np.zeros(len(x), bool), np.zeros(len(x), bool)
            for a, b in zip(points, points[1:] + points[:1]):
                cross = (b["x"] - a["x"]) * (y - a["y"]) - (b["y"] - a["y"]) * (x - a["x"])
                boundary |= ((np.abs(cross) <= 1e-7)
                             & (x >= min(a["x"], b["x"]) - 1e-7) & (x <= max(a["x"], b["x"]) + 1e-7)
                             & (y >= min(a["y"], b["y"]) - 1e-7) & (y <= max(a["y"], b["y"]) + 1e-7))
                if a["y"] != b["y"]:
                    inside ^= ((a["y"] > y) != (b["y"] > y)) & (
                        x < (b["x"] - a["x"]) * (y - a["y"]) / (b["y"] - a["y"]) + a["x"])
            hit |= inside | boundary
        else:
            hit |= ((x >= box["x"]) & (x <= box["x"] + box["width"])
                    & (y >= box["y"]) & (y <= box["y"] + box["height"]))
    return hit & encoded.valid


def body_boxes(pair: dict) -> list[dict]:
    """Prefer explicit vehicle outlines; retain older broad normal-body boxes."""
    references = [box for box in pair.get("reference_regions", [])
                  if box.get("image_slot") == "secondary" and box.get("label") in {"vehicle_body", "normal_vehicle_body"}]
    return references or [box for box in pair.get("return_normal_boxes", [])
            if box.get("label") == "normal_body_line" and box["width"] * box["height"] >= BODY_BOX_MIN_AREA]


def on_body(pair: dict, peak: list[float]) -> bool | None:
    """最高分位置是否在人工框的車身範圍內（原圖座標）；沒有車身框時為 None。"""
    boxes = body_boxes(pair)
    if not boxes:
        return None
    point = Encoded(None, np.asarray([peak]), np.ones(1, bool), False)
    return bool(in_boxes(point, boxes)[0])


def near(encoded: Encoded, x: float, y: float) -> np.ndarray:
    if encoded.flip:
        x = 1 - x
    spacing = np.abs(np.diff(np.unique(encoded.centers[:, 0]))).min() if len(encoded.centers) > 1 else 0.02
    radius = HARD_NEGATIVE_RADIUS * spacing
    distance = np.hypot(encoded.centers[:, 0] - x, (encoded.centers[:, 1] - y))
    return (distance <= radius) & encoded.valid


def pool3(scores: np.ndarray) -> np.ndarray:
    size = int(round(np.sqrt(len(scores))))
    grid = np.pad(scores.reshape(size, size), 1)
    return (sum(grid[i:i + size, j:j + size] for i in range(3) for j in range(3)) / 9).ravel()


def fit_head(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Same weighted logistic head as the API (_fit), run on the GPU."""
    mean = x.mean(0)
    std = x.std(0) + 1e-6
    feats = torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)
    labels = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    torch.manual_seed(0)
    head = torch.nn.Linear(x.shape[1], 1).to(DEVICE)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(head(feats).squeeze(1), labels, reduction="none")
                * weights).sum() + L2 * head.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    raw = head.weight.detach().cpu().numpy()[0]
    return (raw / std).astype(np.float32), float(head.bias.item() - (raw * mean / std).sum())


def patch_map(weights: np.ndarray, bias: float, feats: np.ndarray, valid: np.ndarray) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(feats @ weights + bias, -30, 30)))
    probabilities[~valid] = 0.0
    return pool3(probabilities)


class Trainer:
    """One recipe on top of the encoded photos; remedies come from the memory."""

    def __init__(self, groups: dict, recipe: dict, memory: dict) -> None:
        self.groups = groups
        self.recipe = recipe
        self.memory = recipe_remedies(memory, recipe["name"])
        self.encoded = {}
        for member in recipe["members"]:
            tag = (member["backbone"], member["input_size"])
            if tag not in self.encoded:
                self.encoded[tag] = encode_all(groups, *tag)
        self.humans = groups["human"]
        self.positives = [p for p in self.humans if p["label"] == "confirmed_new_damage"]
        self.negatives = [p for p in self.humans if p["label"] != "confirmed_new_damage"]
        # Only the nearest-patch indices are cached; full contrast features for
        # every photo would not fit in memory with ViT-L/g.
        self._nearest: dict = {}

    # ---- lookups
    def enc(self, member: dict, photo: str, variant: int = 0, pair_id: str = "") -> Encoded:
        key = f"{photo}|v0" if variant == 0 else f"{photo}|v{variant}|{pair_id}"
        return self.encoded[(member["backbone"], member["input_size"])][key]

    def feats(self, member: dict, query: Encoded, reference: Encoded, cache_key) -> np.ndarray:
        if member["feature"] == "plain":
            return features(query, reference, "plain")
        key = (member["backbone"], member["input_size"], member["feature"], cache_key)
        if key not in self._nearest:
            self._nearest[key] = nearest_patches(query, reference, local_window(member["feature"]))
        return features(query, reference, member["feature"], self._nearest[key])

    def pair_map(self, members: list[dict], heads: list, pair: dict) -> tuple[np.ndarray, Encoded]:
        total = sum(m["weight"] for m in members)
        mixed = 0.0
        first = None
        for member, (weights, bias) in zip(members, heads):
            returned = self.enc(member, pair["return"])
            pickup = self.enc(member, pair["pickup"])
            feats = self.feats(member, returned, pickup, (pair["id"], "return", 0))
            mixed = mixed + member["weight"] / total * patch_map(weights, bias, feats, returned.valid)
            first = first or returned
        return mixed, first

    # ---- training matrix
    def matrix(self, member: dict, train_ids: set[str], rng, refine=None):
        xs, ys, ws = [], [], []

        def add(values, label, mass):
            if len(values):
                xs.append(values)
                ys.append(np.full(len(values), label, dtype=np.float32))
                ws.append(np.full(len(values), mass / len(values), dtype=np.float32))

        boosts = self.memory["positive_boost"]
        type_boost = self.memory["type_boost"]
        hard = self.memory["hard_negatives"]
        for pair in self.humans:
            if pair["id"] not in train_ids:
                continue
            positive = pair["label"] == "confirmed_new_damage"
            pickup = self.enc(member, pair["pickup"])
            variants = range(1 + (AUGMENTED_VARIANTS if positive else 0))
            mass = 1.0
            if positive:
                mass = boosts.get(pair["id"], 1.0) * max(
                    [type_boost.get(box.get("label", ""), 1.0) for box in pair["issue_boxes"]] or [1.0])
            for variant in variants:
                returned = self.enc(member, pair["return"], variant, pair["id"])
                feats = self.feats(member, returned, pickup, (pair["id"], "return", variant))
                issue = in_boxes(returned, pair["issue_boxes"]) if positive else np.zeros(len(returned.valid), bool)
                background = np.where(returned.valid & ~issue)[0]
                background = rng.choice(background, min(len(background), BACKGROUND_PATCHES), replace=False)
                chosen = np.where(issue)[0]
                if refine is not None and member["top_fraction"] < 1.0 and len(chosen):
                    scores = feats[chosen] @ refine[0] + refine[1]
                    keep = max(1, round(len(chosen) * member["top_fraction"]))
                    chosen = chosen[np.argsort(scores)[-keep:]]
                add(feats[chosen], 1.0, mass)
                add(feats[background], 0.0, 1.0)
                if RETURN_NORMAL_MASS > 0:
                    for box in pair["return_normal_boxes"]:
                        add(feats[in_boxes(returned, [box]) & ~issue], 0.0, RETURN_NORMAL_MASS)
                for spot in hard.get(pair["id"], []):
                    if spot["slot"] == "return":
                        add(feats[near(returned, spot["x"], spot["y"]) & ~issue], 0.0, HARD_NEGATIVE_MASS)
            if pair["normal_boxes"]:
                reference = self.enc(member, pair["return"])
                feats = self.feats(member, pickup, reference, (pair["id"], "pickup", 0))
                add(feats[in_boxes(pickup, pair["normal_boxes"])], 0.0, 0.5)
        for pair in self.groups["weak"]:
            if pair["split"] != "train":
                continue
            returned = self.enc(member, pair["return"])
            pickup = self.enc(member, pair["pickup"])
            for query, reference, slot in ((returned, pickup, "return"), (pickup, returned, "pickup")):
                feats = self.feats(member, query, reference, (pair["id"], slot, 0))
                index = np.where(query.valid)[0]
                add(feats[rng.choice(index, min(len(index), NEGATIVE_PATCHES_PER_IMAGE), replace=False)], 0.0, 1.0)
                if slot == "return":
                    for spot in hard.get(pair["id"], []):
                        add(feats[near(query, spot["x"], spot["y"])], 0.0, HARD_NEGATIVE_MASS)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        w[y == 1] /= max(1e-12, w[y == 1].sum())
        w[y == 0] /= max(1e-12, w[y == 0].sum())
        return x, y, w

    def fit(self, train_ids: set[str], rng) -> list:
        heads = []
        for member in self.recipe["members"]:
            head = fit_head(*self.matrix(member, train_ids, rng))
            if member["top_fraction"] < 1.0:
                head = fit_head(*self.matrix(member, train_ids, rng, refine=head))
            heads.append(head)
        return heads

    def scores(self, heads: list, pairs: list[dict]) -> list[float]:
        return [float(self.pair_map(self.recipe["members"], heads, pair)[0].max()) for pair in pairs]

    # ---- one full test
    def _cross_validate(self, shuffle_seed: int, rng_seed: int, val_pairs: list[dict]) -> dict:
        """Grouped CV once: out-of-fold rows for every human pair and each fold's calibration scores."""
        rng = np.random.default_rng(rng_seed)
        members = self.recipe["members"]
        folds = max(2, min(MAX_FOLDS, len(self.positives)))
        order = list(self.positives)
        random.Random(shuffle_seed).shuffle(order)
        fold_of = {pair["id"]: index % folds for index, pair in enumerate(order)}
        for index, pair in enumerate(self.negatives):
            fold_of[pair["id"]] = index % folds
        oof: dict[str, dict] = {}
        fold_val: list[np.ndarray] = []
        for fold in range(folds):
            heads = self.fit({p["id"] for p in self.humans if fold_of[p["id"]] != fold}, rng)
            fold_val.append(np.asarray(self.scores(heads, val_pairs)))
            for pair in self.humans:
                if fold_of[pair["id"]] != fold:
                    continue
                pooled, encoded = self.pair_map(members, heads, pair)
                top = int(pooled.argmax())
                positive = pair["label"] == "confirmed_new_damage"
                oof[pair["id"]] = {
                    "positive": positive, "fold": fold, "score": float(pooled[top]),
                    "top_in_box": bool(in_boxes(encoded, pair["issue_boxes"])[top]) if positive else None,
                    "peak": [float(encoded.centers[top, 0]), float(encoded.centers[top, 1])],
                    "labels": sorted({b.get("label", "") for b in pair["issue_boxes"]}),
                }
                oof[pair["id"]]["peak_on_body"] = on_body(pair, oof[pair["id"]]["peak"])
        return {"oof": oof, "fold_val": fold_val}

    @staticmethod
    def _flags(run: dict, rate: float) -> dict[str, bool]:
        thresholds = [float(np.quantile(values, 1 - rate)) for values in run["fold_val"]]
        return {key: row["score"] >= thresholds[row["fold"]] for key, row in run["oof"].items()}

    def evaluate(self) -> dict:
        """Repeated grouped CV (CV_REPEATS shuffles) plus the fixed test set.

        With only a handful of boxed damages one pair moves recall by ~10 points,
        so every out-of-fold number is the mean over the repeats."""
        val_pairs = [p for p in self.groups["weak"] if p["split"] == "val"]
        test_pairs = [p for p in self.groups["weak"] if p["split"] == "test"]
        runs = []
        for index, (shuffle_seed, rng_seed) in enumerate(CV_REPEATS):
            runs.append(self._cross_validate(shuffle_seed, rng_seed, val_pairs))
            log(f"  重複交叉驗證 {index + 1}/{len(CV_REPEATS)}")
        heads = self.fit({p["id"] for p in self.humans}, np.random.default_rng(CV_REPEATS[0][1]))
        val = np.asarray(self.scores(heads, val_pairs))
        test = np.asarray(self.scores(heads, test_pairs))
        holdout_pairs = [p for p in self.groups["holdout"] if p["label"] == "confirmed_new_damage"]
        holdout = dict(zip([p["id"] for p in holdout_pairs], self.scores(heads, holdout_pairs)))
        positive_ids = [p["id"] for p in self.positives]
        negative_ids = [p["id"] for p in self.negatives]

        def operating_point(rate: float) -> dict:
            flags = [self._flags(run, rate) for run in runs]
            threshold = float(np.quantile(val, 1 - rate))
            cv_found = float(np.mean([sum(f[k] for k in positive_ids) for f in flags]))
            hold_found = sum(score >= threshold for score in holdout.values())
            cv_false = float(np.mean([sum(f[k] for k in negative_ids) for f in flags]))
            test_flags = test >= threshold
            n_pos = len(positive_ids) + len(holdout)
            total = n_pos + len(negative_ids) + len(test)
            correct = cv_found + hold_found + (len(negative_ids) - cv_false) + int((~test_flags).sum())
            return {
                "rate": rate, "threshold": threshold, "flags": flags,
                "recall": (cv_found + hold_found) / n_pos if n_pos else 0.0,
                "cv_recall": cv_found / len(positive_ids) if positive_ids else 0.0,
                "holdout_recall": hold_found / len(holdout) if holdout else None,
                "accuracy": correct / total,
                "human_review_rate": float(test_flags.mean()),
                "cv_human_negative_fpr": cv_false / len(negative_ids) if negative_ids else None,
                "positives_found": f"{cv_found + hold_found:.1f}/{n_pos}",
            }

        # 門檻：召回率達標下，人工審核（無理賠校準集誤報）最少的校準比例；達不到目標時
        # 取能達到的最高召回率中送審最少的，不為了沒有增加的召回多送人工。只看交叉驗證
        # 與校準集，不看測試集。
        options = [(min(operating_point(rate)["cv_recall"], TARGET_RECALL), -rate, rate) for rate in CALIBRATION_GRID]
        rate = max(options)[2]
        chosen = operating_point(rate)
        fixed = operating_point(0.10)
        # 每組人工真值：重複交叉驗證中被抓到的比例；多數決當作這一輪的判定。
        oof = runs[0]["oof"]
        for key, row in oof.items():
            row["detect_rate"] = float(np.mean([flags[key] for flags in chosen["flags"]]))
            row["detected"] = row["detect_rate"] >= 0.5
        holdout_rows = {key: {"score": score, "detected": score >= chosen["threshold"]} for key, score in holdout.items()}
        pos_scores = np.asarray([run["oof"][k]["score"] for run in runs for k in positive_ids])
        auroc = float(((pos_scores[:, None] > test[None]) + 0.5 * (pos_scores[:, None] == test[None])).mean()) if len(pos_scores) else 0.0
        localization = float(np.mean([run["oof"][k]["top_in_box"] for run in runs for k in positive_ids])) if positive_ids else 0.0
        # 人工確認無車損卻被抓的組數（重複交叉驗證平均），以及其中最高分落在人工車身框外的組數。
        flagged_negatives = [(run["oof"][k], flags[k]) for run, flags in zip(runs, chosen["flags"]) for k in negative_ids]
        off_body = [row["peak_on_body"] is False for row, flagged in flagged_negatives if flagged]
        metrics = {key: chosen[key] for key in ("recall", "cv_recall", "holdout_recall", "accuracy", "human_review_rate",
                                                 "cv_human_negative_fpr", "positives_found")}
        metrics.update({
            "cv_localization": localization,
            "cv_auroc": auroc,
            "cv_repeats": len(runs),
            "human_negatives_flagged_mean": float(np.mean([sum(f[k] for k in negative_ids) for f in chosen["flags"]]))
            if negative_ids else 0.0,
            "flagged_negatives_off_body": float(np.mean(off_body)) if off_body else 0.0,
            "at_10pct_calibration": {key: fixed[key] for key in ("recall", "cv_recall", "holdout_recall", "accuracy",
                                                                 "human_review_rate", "cv_human_negative_fpr")},
        })
        return {
            "heads": heads,
            "threshold": chosen["threshold"],
            "calibration_rate": rate,
            "oof": oof,
            "holdout": holdout_rows,
            "weak_test_scores": test.tolist(),
            "metrics": metrics,
        }

    def mine_weak_train(self, heads: list) -> list[tuple[str, list[float]]]:
        """Peaks of the most damage-like no-claim *train* pairs (never val/test)."""
        rows = []
        for pair in self.groups["weak"]:
            if pair["split"] != "train":
                continue
            pooled, encoded = self.pair_map(self.recipe["members"], heads, pair)
            top = int(pooled.argmax())
            rows.append((float(pooled[top]), pair["id"], [float(encoded.centers[top, 0]), float(encoded.centers[top, 1])]))
        rows.sort(reverse=True)
        return [(pair_id, peak) for _score, pair_id, peak in rows[:WEAK_MINING_PER_ROUND]]


# ------------------------------------------------------------------ GPU 計時
_detector = None


def vehicle_detector():
    global _detector
    if _detector is None:
        if LOCAL_HUB is not None:
            torch.hub.set_dir(str(LOCAL_HUB))
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        _detector = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).eval().to(DEVICE)
    return _detector


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def time_inspection(groups: dict, recipe: dict, heads: list) -> float:
    """Wall seconds for one return inspection (4 pairs): detect, encode both photos, score."""
    pairs = [p for p in groups["weak"] if p["split"] == "test"][:PAIRS_PER_INSPECTION]
    detector = vehicle_detector()

    def run_once() -> float:
        started = time.perf_counter()
        for pair in pairs:
            photos = {}
            for slot in ("pickup", "return"):
                image = open_image(pair[slot])
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0).to(DEVICE)
                with torch.inference_mode():
                    detector([tensor])
                photos[slot] = image
            total = sum(m["weight"] for m in recipe["members"])
            mixed = 0.0
            shared: dict[tuple[str, int], dict] = {}  # 同一骨幹的成員共用一次特徵
            for member, (weights, bias) in zip(recipe["members"], heads):
                tag = (member["backbone"], member["input_size"])
                if tag not in shared:
                    shared[tag] = {}
                    for slot, image in photos.items():
                        model = backbone(member["backbone"])
                        array, centers, valid = letterbox(image, pair[f"{slot}_vehicle_box"], member["input_size"], model.stride)
                        tokens = model(torch.from_numpy(array)[None])[0].cpu().numpy()
                        shared[tag][slot] = Encoded(tokens, centers, valid, False)
                encoded = shared[tag]
                feats = features(encoded["return"], encoded["pickup"], member["feature"])
                mixed = mixed + member["weight"] / total * patch_map(weights, bias, feats, encoded["return"].valid)
            float(np.max(mixed))
        sync()
        return time.perf_counter() - started

    run_once()  # warm-up (model load, cuDNN autotune)
    return min(run_once() for _ in range(2))


# ------------------------------------------------------------------ 記憶
def empty_memory() -> dict:
    return {
        "schema": "damage_pair_failure_memory_v2",
        # 每個模型各自的補強：A 模型學到的困難負樣本與加權對 B 模型是雜訊（2026-09-28
        # 實測把整張比對模型的補強套到同位置比對模型，AUROC 由 0.889 掉到 0.791）。
        "remedies": {},          # 模型名稱 -> recipe_remedies() 的內容
        "banned_backbones": {},  # backbone@size -> 原因
        "tried": {},             # recipe 指紋 -> 結果摘要（相同設定不再重跑）
        "sample_history": {},    # sample id -> [{run, round, reason}]
        "runs": [],
    }


def recipe_remedies(memory: dict, name: str) -> dict:
    """The remedies one model has learned (created empty on first use)."""
    remedies = memory.setdefault("remedies", {}).setdefault(name, {})
    remedies.setdefault("positive_boost", {})   # sample id -> 權重倍數
    remedies.setdefault("type_boost", {})       # 車損類型 -> 權重倍數
    remedies.setdefault("hard_negatives", {})   # pair id -> [{slot, x, y, reason, round}]
    remedies.setdefault("remedy_log", [])       # 每批補強的內容，變差時用來撤銷
    remedies.setdefault("rejected", {"spots": [], "boost_samples": [], "types": []})  # 證實有害、不再加回
    return remedies


def load_memory() -> dict:
    if MEMORY_PATH.is_file():
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        legacy = {key: memory.pop(key) for key in ("positive_boost", "type_boost", "hard_negatives",
                                                   "remedy_log", "rejected") if key in memory}
        if legacy:
            # v1 shared one set of remedies across models; keep it for reference only.
            memory.setdefault("remedies", {})["_v1_shared_unused"] = legacy
            memory["schema"] = "damage_pair_failure_memory_v2"
        for key, value in empty_memory().items():
            memory.setdefault(key, value)
        learned = {name: r for name, r in memory["remedies"].items() if not name.startswith("_")}
        log(f"讀取失敗記憶：{len(memory['runs'])} 次執行、{len(learned)} 個模型有補強"
            f"（困難負樣本 {sum(len(v) for r in learned.values() for v in r.get('hard_negatives', {}).values())} 個）、"
            f"封鎖骨幹 {list(memory['banned_backbones'])}")
        return memory
    return empty_memory()


def save_memory(memory: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    temporary = MEMORY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(memory, ensure_ascii=False, indent=1, default=to_json), encoding="utf-8")
    temporary.replace(MEMORY_PATH)


def fingerprint(recipe: dict, memory: dict) -> str:
    own = recipe_remedies(memory, recipe["name"])
    remedies = {key: own[key] for key in ("positive_boost", "type_boost", "hard_negatives")}
    text = json.dumps([recipe["members"], remedies] + ([RETURN_NORMAL_MASS] if RETURN_NORMAL_MASS else []), sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def member_tags(recipe: dict) -> list[str]:
    return [f"{m['backbone']}@{m['input_size']}" for m in recipe["members"]]


def rung_allowed(rung: int, memory: dict) -> bool:
    return not any(tag in memory["banned_backbones"] for tag in member_tags(LADDER[rung]))


def learn_from(result: dict, trainer: Trainer, memory: dict, run_id: str, round_no: int, seconds: float) -> list[str]:
    """Turn this round's failures into remedies the next round (and next run) keeps."""
    reasons: list[str] = []

    batch = f"{run_id}/{round_no}"
    remedies = recipe_remedies(memory, trainer.recipe["name"])
    rejected = remedies["rejected"]

    def remember(sample_id: str, reason: str) -> None:
        memory["sample_history"].setdefault(sample_id, []).append({"run": run_id, "round": round_no, "reason": reason})
        reasons.append(reason)

    def same_spot(a: dict, pair_id: str, peak: list[float]) -> bool:
        return a.get("pair_id", pair_id) == pair_id and abs(a["x"] - peak[0]) < 0.02 and abs(a["y"] - peak[1]) < 0.02

    def hard_negative(pair_id: str, peak: list[float], reason: str) -> None:
        spots = remedies["hard_negatives"].get(pair_id, [])
        if any(same_spot(s, pair_id, peak) for s in spots + rejected["spots"]):
            return
        remedies["hard_negatives"][pair_id] = spots
        spots.append({"slot": "return", "x": round(peak[0], 4), "y": round(peak[1], 4), "reason": reason, "round": batch})
        remedies["remedy_log"].append({"batch": batch, "kind": "spot", "pair_id": pair_id,
                                     "x": round(peak[0], 4), "y": round(peak[1], 4)})

    def boost(kind: str, key: str, value: float) -> None:
        table = remedies["positive_boost" if kind == "boost" else "type_boost"]
        if key in rejected["boost_samples" if kind == "boost" else "types"]:
            return
        remedies["remedy_log"].append({"batch": batch, "kind": kind, "key": key, "before": table.get(key)})
        table[key] = value

    for sample_id, row in result["oof"].items():
        if row["positive"]:
            if not row["detected"]:
                reason = "missed_low_score" if row["top_in_box"] else "missed_wrong_location"
                boost("boost", sample_id, min(MAX_POSITIVE_BOOST, remedies["positive_boost"].get(sample_id, 1.0) * 2))
                for label in row["labels"]:
                    boost("type", label, min(MAX_TYPE_BOOST, remedies["type_boost"].get(label, 1.0) + 0.5))
                if not row["top_in_box"]:
                    hard_negative(sample_id, row["peak"], reason)
                remember(sample_id, reason)
            elif not row["top_in_box"]:
                hard_negative(sample_id, row["peak"], "detected_wrong_location")
                remember(sample_id, "detected_wrong_location")
        elif row["detected"]:
            hard_negative(sample_id, row["peak"], "false_alarm")
            remember(sample_id, "false_alarm")
    for key, row in result["holdout"].items():
        if not row["detected"]:
            remember(key, "holdout_missed")
    metrics = result["metrics"]
    if metrics["accuracy"] < TARGET_ACCURACY:
        for pair_id, peak in trainer.mine_weak_train(result["heads"]):
            hard_negative(pair_id, peak, "too_many_false_alarms")
        reasons.append("too_many_false_alarms")
    if seconds > MAX_INSPECTION_SECONDS:
        for tag in member_tags(trainer.recipe):
            memory["banned_backbones"][tag] = f"一次檢查 {seconds:.1f} 秒超過 {MAX_INSPECTION_SECONDS:.0f} 秒"
        reasons.append("too_slow")
    return reasons


def roll_back(memory: dict, recipe_name: str, batch: str) -> int:
    """Undo one batch of remedies that made the same model worse, and never re-add it."""
    remedies = recipe_remedies(memory, recipe_name)
    undone = 0
    for item in [i for i in remedies["remedy_log"] if i["batch"] == batch][::-1]:
        if item["kind"] == "spot":
            spots = remedies["hard_negatives"].get(item["pair_id"], [])
            spots[:] = [s for s in spots if s["round"] != batch]
            if not spots:
                remedies["hard_negatives"].pop(item["pair_id"], None)
            remedies["rejected"]["spots"].append({"pair_id": item["pair_id"], "x": item["x"], "y": item["y"]})
        else:
            table = remedies["positive_boost" if item["kind"] == "boost" else "type_boost"]
            if item["before"] is None:
                table.pop(item["key"], None)
            else:
                table[item["key"]] = item["before"]
            remedies["rejected"]["boost_samples" if item["kind"] == "boost" else "types"].append(item["key"])
        undone += 1
    remedies["remedy_log"] = [i for i in remedies["remedy_log"] if i["batch"] != batch]
    return undone


def next_rung(current: int, stay: bool, memory: dict) -> int | None:
    """Stay on the model to validate new remedies, otherwise take the next untested model.

    A model plus the exact remedies that were already tested is never run again."""
    size = len(LADDER)
    order = ([current] if stay else []) + [(current + step) % size for step in range(1, size)]
    for rung in order:
        if rung_allowed(rung, memory) and fingerprint(LADDER[rung], memory) not in memory["tried"]:
            return rung
    return None


# ------------------------------------------------------------------ 匯出
def selection_key(entry: dict) -> tuple:
    """Recall target first; among rounds that reach it, the most accurate (fewest reviews),
    otherwise the highest recall."""
    metrics = entry["metrics"]
    meets_recall = metrics["recall"] >= TARGET_RECALL
    return (meets_recall, metrics["accuracy"] >= TARGET_ACCURACY,
            metrics["accuracy"] if meets_recall else metrics["recall"], metrics["recall"])


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def export_checkpoint(groups: dict, recipe: dict, result: dict, seconds: float, run_id: str, passed: bool,
                      path: Path | None = None) -> Path:
    metrics = result["metrics"]
    device = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
    checkpoint = {
        "schema_version": 3,
        "task": "damage_pair",
        "architecture": "dinov2_patch_damage_probe",
        "backbone": "+".join(m["backbone"] for m in recipe["members"]),
        "classes": ["no_new_damage", "confirmed_new_damage"],
        "input_size": recipe["members"][0]["input_size"],
        "preprocess": "vehicle_crop_letterbox",
        "members": [
            {
                "backbone": m["backbone"],
                "input_size": m["input_size"],
                "weights": torch.from_numpy(weights),
                "bias": float(bias),
                "weight": float(m["weight"]),
                "positive_top_fraction": float(m["top_fraction"]),
                "feature_mode": m["feature"],
            }
            for m, (weights, bias) in zip(recipe["members"], result["heads"])
        ],
        "threshold": float(result["threshold"]),
        "model_version": f"autolearn-{slug(recipe['name'])}-{run_id}",
        "management_stage": "deployed" if passed else "trial",
        "management_metric": "human_truth_cv_recall",
        "management_score": round(metrics["cv_recall"], 4),
        "minimum_management_score": TARGET_RECALL,
        "production_approved": passed,
        "label_source": "營運人員框選的取還車車損人工真值＋無理賠配對弱負樣本（自主學習）",
        "sample_ids": [p["id"] for p in groups["human"]],
        "report": {"test": {"macro_f1": metrics["cv_recall"], "accuracy": metrics["accuracy"]}},
        "report_json": json.dumps({
            "recipe": recipe["name"], **metrics,
            "calibration_rate": result["calibration_rate"],
            "target_calibration_fpr": result["calibration_rate"],
            "final_test_fpr": metrics["human_review_rate"],
            "inspection_seconds": round(seconds, 2),
            "inspection_device": device,
            "return_normal_mass": RETURN_NORMAL_MASS,
        }, ensure_ascii=False),
        "created_at": now(),
        # average_ms is per pair, like the API benchmark; import_damage_pair_model.py
        # re-measures it on the deployment machine.
        "benchmark": {"average_ms": round(seconds / PAIRS_PER_INSPECTION * 1000, 1),
                      "inspection_seconds": round(seconds, 2), "pairs": PAIRS_PER_INSPECTION,
                      "device": device, "measured_at": now()},
    }
    path = path or WORK / "damage_pair_specialist.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


# ------------------------------------------------------------------ 主流程
def to_json(value):
    return value.item() if hasattr(value, "item") else str(value)


def select_models(names: list[str] | None) -> None:
    if names:
        unknown = set(names) - {recipe["name"] for recipe in ALL_MODELS}
        if unknown:
            raise SystemExit(f"沒有這些模型：{sorted(unknown)}；可用：{[r['name'] for r in ALL_MODELS]}")
        chosen = [recipe for recipe in ALL_MODELS if recipe["name"] in names]
    else:
        chosen = [recipe for recipe in ALL_MODELS if DEVICE.type == "cuda" or recipe["name"] not in CPU_ONLY_SKIP]
    LADDER[:] = chosen
    log(f"裝置 {DEVICE}；候選模型：{', '.join(r['name'] for r in LADDER)}")


def compare(groups: dict) -> list[dict]:
    """Every candidate once, without remedies, so the models differ only by themselves."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"compare_{run_id}.json"
    repeats = len(CV_REPEATS)
    for recipe in LADDER:
        log(f"===== 比較：{recipe['name']}（{', '.join(member_tags(recipe))}）")
        trainer = Trainer(groups, recipe, empty_memory())
        result = trainer.evaluate()
        seconds = time_inspection(groups, recipe, result["heads"])
        metrics = result["metrics"]
        passed = (metrics["recall"] >= TARGET_RECALL and metrics["accuracy"] >= TARGET_ACCURACY
                  and seconds <= MAX_INSPECTION_SECONDS)
        saved = export_checkpoint(groups, recipe, result, seconds, run_id, passed,
                                  WORK / "models" / f"{slug(recipe['name'])}.pt")
        log(f"  模型已存：{saved}")
        # 每組車損：交叉驗證 N 次中抓到幾次（固定測試集只測一次）。
        pairs = {k: (round(r["score"], 3), round(r["detect_rate"] * repeats), repeats)
                 for k, r in result["oof"].items() if r["positive"]}
        pairs.update({k: (round(r["score"], 3), int(r["detected"]), 1) for k, r in result["holdout"].items()})
        rows.append({"model": recipe["name"], **metrics, "seconds": seconds, "threshold": result["threshold"],
                     "calibration_rate": result["calibration_rate"], "damage_pairs": pairs,
                     "human_negatives_flagged": sorted(k for k, r in result["oof"].items()
                                                       if not r["positive"] and r["detected"]),
                     "return_normal_mass": RETURN_NORMAL_MASS})
        fixed = metrics["at_10pct_calibration"]
        log("  自動門檻：召回率 {:.0%}（{}）、準確度 {:.1%}、人工審核率 {:.1%}；10% 校準：召回率 {:.0%}、準確度 {:.1%}、"
            "人工審核率 {:.1%}；位置命中 {:.0%}、AUROC {:.3f}、一次檢查 {:.1f} 秒".format(
                metrics["recall"], metrics["positives_found"], metrics["accuracy"], metrics["human_review_rate"],
                fixed["recall"], fixed["accuracy"], fixed["human_review_rate"],
                metrics["cv_localization"], metrics["cv_auroc"], seconds))
        del trainer
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=to_json), encoding="utf-8")
    print("\n| 模型 | 召回率 | 準確度 | 人工審核率 | 10%校準 召回率 | 10%校準 準確度 | 10%校準 審核率 | 位置命中 | AUROC | 一次檢查秒數 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        fixed = row["at_10pct_calibration"]
        print(f"| {row['model']} | {row['recall']:.0%} | {row['accuracy']:.1%} | {row['human_review_rate']:.1%} | "
              f"{fixed['recall']:.0%} | {fixed['accuracy']:.1%} | {fixed['human_review_rate']:.1%} | "
              f"{row['cv_localization']:.0%} | {row['cv_auroc']:.3f} | {row['seconds']:.1f} |")
    print("\n| 模型 | 人工無車損誤報（組／比例） | 誤報落在車身框外 | 無理賠 test 送審率 |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print(f"| {row['model']} | {row['human_negatives_flagged_mean']:.1f}／{row['cv_human_negative_fpr'] or 0:.1%} | "
              f"{row['flagged_negatives_off_body']:.0%} | {row['human_review_rate']:.1%} |")
    keys = list(rows[0]["damage_pairs"]) if rows else []
    print("\n| 車損配對（抓到次數/測試次數 最高分） | " + " | ".join(row["model"] for row in rows) + " |")
    print("| --- |" + " --- |" * len(rows))
    for key in keys:
        cells = [f"{row['damage_pairs'][key][1]}/{row['damage_pairs'][key][2]} {row['damage_pairs'][key][0]:.2f}"
                 for row in rows]
        print(f"| {key} | " + " | ".join(cells) + " |")
    log(f"比較結果：{path}")
    return rows


def learning_curve(groups: dict, sizes: tuple[int, ...] = (2, 4, 6), shuffles: int = 3) -> list[dict]:
    """How much each model gains from more boxed damages.

    For each shuffle, EVAL_HELD boxed damages plus the _TEST folder are held out
    as a fixed exam; the model trains on the first k of the remaining damages
    (k in ``sizes``) and every human-confirmed normal pair, and is scored on the
    same exam each time, so the curve shows the effect of k alone."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    positives = [p for p in groups["human"] if p["label"] == "confirmed_new_damage"]
    negatives = [p for p in groups["human"] if p["label"] != "confirmed_new_damage"]
    held = max(1, len(positives) - max(sizes))
    sizes = tuple(k for k in sizes if k <= len(positives) - held)
    val_pairs = [p for p in groups["weak"] if p["split"] == "val"]
    test_pairs = [p for p in groups["weak"] if p["split"] == "test"]
    holdout = [p for p in groups["holdout"] if p["label"] == "confirmed_new_damage"]
    rows = []
    for recipe in LADDER:
        log(f"===== 學習曲線：{recipe['name']}")
        trainer = Trainer(groups, recipe, empty_memory())
        curve = {k: {"auroc": [], "recall": [], "review": []} for k in sizes}
        for shuffle in range(shuffles):
            order = list(positives)
            random.Random(1000 + shuffle).shuffle(order)
            exam, pool = order[:held], order[held:]
            exam_pairs = exam + holdout
            for k in sizes:
                ids = {p["id"] for p in pool[:k]} | {p["id"] for p in negatives}
                heads = trainer.fit(ids, np.random.default_rng(shuffle * 100 + k))
                threshold = float(np.quantile(trainer.scores(heads, val_pairs), 0.90))
                test = np.asarray(trainer.scores(heads, test_pairs))
                found = np.asarray(trainer.scores(heads, exam_pairs))
                curve[k]["auroc"].append(float(((found[:, None] > test[None]) + 0.5 * (found[:, None] == test[None])).mean()))
                curve[k]["recall"].append(float((found >= threshold).mean()))
                curve[k]["review"].append(float((test >= threshold).mean()))
            log(f"  打散 {shuffle + 1}/{shuffles}")
        summary = {k: {name: float(np.mean(values)) for name, values in metrics.items()} for k, metrics in curve.items()}
        first, last = sizes[0], sizes[-1]
        rows.append({
            "model": recipe["name"], "curve": summary, "exam_damages": held + len(holdout),
            "auroc_gain": summary[last]["auroc"] - summary[first]["auroc"],
            "recall_gain": summary[last]["recall"] - summary[first]["recall"],
        })
        del trainer
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"learning_curve_{run_id}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=to_json), encoding="utf-8")
    header = " | ".join(f"{k} 組 AUROC／召回" for k in sizes)
    print(f"\n| 模型 | {header} | AUROC 增幅 | 召回增幅 |")
    print("| --- |" + " --- |" * (len(sizes) + 2))
    for row in rows:
        cells = " | ".join(f"{row['curve'][k]['auroc']:.3f}／{row['curve'][k]['recall']:.0%}" for k in sizes)
        print(f"| {row['model']} | {cells} | {row['auroc_gain']:+.3f} | {row['recall_gain']:+.0%} |")
    log(f"學習曲線：{path}（每次考題：{rows[0]['exam_damages'] if rows else 0} 組車損＋{len(test_pairs)} 組無理賠）")
    return rows


def main(models: list[str] | None = None, rounds: int = MAX_ROUNDS, run_compare: bool = False,
         run_curve: bool = False) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    select_models(models)
    groups = load_manifest()
    if run_compare:
        compare(groups)
        return
    if run_curve:
        learning_curve(groups)
        return
    memory = load_memory()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    history: list[dict] = []
    names = [recipe["name"] for recipe in LADDER]
    best_known = max(
        (t for t in memory["tried"].values() if t.get("recipe") in names),
        key=lambda t: (t["passed"], t["recall"] >= TARGET_RECALL, t["accuracy"], t["recall"]),
        default=None,
    )
    baseline = ALL_MODELS[0]["name"]
    first = next((index for index, name in enumerate(names) if name != baseline), 0)
    preferred = names.index(best_known["recipe"]) if best_known else first
    if best_known:
        log(f"從上次最好的模型「{best_known['recipe']}」接續")
    if rung_allowed(preferred, memory) and fingerprint(LADDER[preferred], memory) not in memory["tried"]:
        rung = preferred
    else:
        rung = next_rung(preferred, False, memory)
    selected = None
    for round_no in range(1, rounds + 1):
        if rung is None:
            log("所有可用模型搭配目前的補強都已測過，沒有新的組合可試，提前結束")
            break
        recipe = copy.deepcopy(LADDER[rung])
        key = fingerprint(recipe, memory)
        log(f"===== 第 {round_no}/{rounds} 輪：{recipe['name']}（{', '.join(member_tags(recipe))}）")
        trainer = Trainer(groups, recipe, memory)
        result = trainer.evaluate()
        seconds = time_inspection(groups, recipe, result["heads"])
        metrics = result["metrics"]
        passed = (metrics["recall"] >= TARGET_RECALL and metrics["accuracy"] >= TARGET_ACCURACY
                  and seconds <= MAX_INSPECTION_SECONDS)
        log("召回率 {:.0%}（{}）、準確度 {:.1%}、人工審核率 {:.1%}、位置命中 {:.0%}、AUROC {:.3f}、一次檢查 {:.1f} 秒 → {}".format(
            metrics["recall"], metrics["positives_found"], metrics["accuracy"], metrics["human_review_rate"],
            metrics["cv_localization"], metrics["cv_auroc"], seconds, "達標" if passed else "未達標"))
        missed = [k for k, r in result["oof"].items() if r["positive"] and not r["detected"]]
        missed += [k for k, r in result["holdout"].items() if not r["detected"]]
        if missed:
            log(f"  漏抓：{', '.join(missed)}")
        entry = {"round": round_no, "rung": rung, "recipe": recipe["name"], "metrics": metrics,
                 "threshold": result["threshold"], "seconds": seconds, "passed": passed, "missed": missed,
                 "result": result}
        history.append(entry)
        # 每輪模型都留一份，事後可以改選別輪。
        export_checkpoint(groups, recipe, result, seconds, run_id, passed,
                          WORK / "rounds" / f"{run_id}_round{round_no}.pt")
        summary = {"recipe": recipe["name"], "recall": metrics["recall"], "accuracy": metrics["accuracy"],
                   "passed": passed, "run": run_id}
        if passed:
            memory["tried"][key] = {**summary, "reasons": []}
            selected = entry
            log("三項條件同時達成，選用此模型並停止。")
            break
        memory["tried"][key] = {**summary, "reasons": []}
        previous = history[-2] if len(history) >= 2 else None
        if previous is not None and previous["rung"] == rung and selection_key(entry) < selection_key(previous):
            # 上一輪的補強讓同一模型變差：撤銷、記為有害，不從這個較差的結果學，直接換模型。
            undone = roll_back(memory, recipe["name"], f"{run_id}/{previous['round']}")
            entry["reasons"] = ["remedies_harmful"]
            save_memory(memory)
            log(f"  第 {previous['round']} 輪加入的補強讓同一模型變差，已撤銷 {undone} 項並記為有害，之後不再使用；換下一個模型")
            rung = next_rung(rung, False, memory)
            continue
        batch = f"{run_id}/{round_no}"
        reasons = sorted(set(learn_from(result, trainer, memory, run_id, round_no, seconds)))
        memory["tried"][key]["reasons"] = reasons
        entry["reasons"] = reasons
        added = sum(item["batch"] == batch for item in recipe_remedies(memory, recipe["name"])["remedy_log"])
        save_memory(memory)
        stay = added > 0 and "too_slow" not in reasons
        log(f"  失敗原因：{', '.join(reasons)}；新增 {added} 項補強已寫入記憶，"
            + ("下一輪先用同一模型驗證補強" if stay else "下一輪換模型"))
        rung = next_rung(rung, stay, memory)
    if not history:
        raise SystemExit("沒有執行任何一輪；若要重新開始，請刪除 " + str(MEMORY_PATH))
    if selected is None:
        # 沒有同時達標：時間內的優先，再看召回率是否達標、準確度、召回率。
        in_time = [h for h in history if h["seconds"] <= MAX_INSPECTION_SECONDS] or history
        selected = max(in_time, key=selection_key)
        log(f"沒有一輪同時達標，選用第 {selected['round']} 輪（{selected['recipe']}）當試行模型")
    path = export_checkpoint(groups, LADDER[selected["rung"]], selected["result"], selected["seconds"], run_id,
                             selected["passed"])
    rounds_report = [{k: v for k, v in h.items() if k != "result"} for h in history]
    report = {
        "run": run_id, "selected_round": selected["round"], "passed": selected["passed"],
        "targets": {"recall": TARGET_RECALL, "accuracy": TARGET_ACCURACY, "seconds": MAX_INSPECTION_SECONDS},
        "rounds": rounds_report,
        "missed_now": selected["missed"],
    }
    report_path = WORK / f"report_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=to_json), encoding="utf-8")
    memory["runs"].append({"run": run_id, "selected": selected["recipe"], "passed": selected["passed"],
                           "metrics": selected["metrics"], "rounds": len(history)})
    save_memory(memory)
    log(f"模型：{path}")
    log(f"報告：{report_path}")


def cli() -> None:
    default_zip = max((PROJECT_ROOT / "outputs" / "damage_pair_dataset").glob("damage_pair_dataset_*.zip"), default=None)
    parser = argparse.ArgumentParser(description="前後車損比對自主學習／模型比較")
    parser.add_argument("--dataset", type=Path, default=default_zip, help="export_damage_pair_dataset.py 產生的 zip")
    parser.add_argument("--work", type=Path, default=PROJECT_ROOT / "outputs" / "damage_pair_autolearn")
    parser.add_argument("--models", help="逗號分隔的模型名稱；預設全部（沒有 GPU 時略過 CPU_ONLY_SKIP）")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS, help=f"最多幾輪，上限 {MAX_ROUNDS}")
    parser.add_argument("--compare", action="store_true", help="每個模型各測一次（不加補強）並列出差異")
    parser.add_argument("--learning-curve", action="store_true",
                        help="用 2／4／6 組車損訓練、固定考題，看哪個模型樣本變多時進步最多")
    parser.add_argument("--return-normal-mass", type=float, default=0.0,
                        help="還車照藍框（正常車身、陰影、反光、接縫）每框當困難負樣本的權重；0 = 不用")
    parser.add_argument("--extra-negatives", type=float, default=0.0,
                        help="0–1：未人工確認的補充配對取多少比例當訓練負樣本（需以 --extra-negatives 匯出）")
    args = parser.parse_args()
    if args.dataset is None:
        raise SystemExit("找不到訓練資料 zip，請先執行 scripts/export_damage_pair_dataset.py")
    configure(args.dataset, args.work)
    global EXTRA_NEGATIVES, RETURN_NORMAL_MASS
    EXTRA_NEGATIVES = args.extra_negatives
    RETURN_NORMAL_MASS = args.return_normal_mass
    models = [name.strip() for name in args.models.split(",")] if args.models else None
    main(models, min(args.rounds, MAX_ROUNDS), args.compare, args.learning_curve)


if __name__ == "__main__" and PROJECT_ROOT is not None and "ipykernel" not in sys.modules:
    cli()
