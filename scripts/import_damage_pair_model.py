"""Verify a damage-pair checkpoint trained by scripts/damage_pair_autolearn.py and install it.

The checkpoint is re-scored here on the fixed test set (_TEST claim folder and
the no-claim test pairs) with this machine's encoders, so a mismatch between the
GPU training environment and online inference shows up before it is deployed.
Its recognition time is re-measured on this machine as well.

    python scripts/import_damage_pair_model.py downloads/damage_pair_specialist.pt
    python scripts/import_damage_pair_model.py model.pt --dry-run   # verify only
    python scripts/import_damage_pair_model.py model.pt --candidate siglip2_local4

The current model is copied to models/history/ first. A checkpoint that did not
pass its own targets (production_approved=false) needs --allow-trial.
``--candidate NAME`` instead stores it as models/damage_pair/NAME.pt, where model
management lists it with its scores and an administrator can switch to it;
failed checks are printed but do not block a candidate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.services import damage_pair_specialist as dps  # noqa: E402
from app.services.operations_store import get_operations_store  # noqa: E402

# GPU and CPU tokens differ slightly; the local fixed-test recall may not fall
# further than this below what training reported.
MAX_LOCAL_RECALL_DROP = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Verify without installing")
    parser.add_argument("--allow-trial", action="store_true", help="Install a checkpoint that missed its targets")
    parser.add_argument("--candidate", metavar="NAME",
                        help="Store as models/damage_pair/NAME.pt for model management instead of installing")
    args = parser.parse_args()
    import torch

    checkpoint = dps._load_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise SystemExit(f"{args.checkpoint} is not a damage-pair checkpoint this API can load")
    members = dps._members_from_checkpoint(checkpoint)
    threshold = float(checkpoint["threshold"])
    trained = json.loads(str(checkpoint.get("report_json", "{}")))
    print(f"{checkpoint['model_version']}: {', '.join(f'{m.backbone}@{m.input_size}/{m.feature_mode}' for m in members)}")

    store = get_operations_store()
    holdout = dps.evaluate_holdout(dps.holdout_pairs(store), members, threshold)
    weak, notes = dps._weak_pairs(dps.excluded_folder_hashes())
    test = [pair for pair in weak if pair.split == "test"]
    scores = [
        float(
            dps._mixed_map(
                members,
                [dps._cached_encode(pair.returned, encoder=m.encoder)[0] for m in members],
                [dps._cached_encode(pair.pickup, encoder=m.encoder)[0] for m in members],
            ).max()
        )
        for pair in test
    ]
    local_fpr = float(np.mean([score >= threshold for score in scores])) if scores else None
    training_benchmark = checkpoint.get("benchmark")
    checkpoint["benchmark"] = dps.benchmark_members(members, [(pair.pickup, pair.returned) for pair in test])
    if isinstance(training_benchmark, dict):
        checkpoint["training_benchmark"] = training_benchmark
    checkpoint["local_verification"] = {
        "holdout_recall": holdout["holdout_recall"], "test_fpr": local_fpr, "verified_at": datetime.now().isoformat(),
    }
    summary = {
        "trained_recall": trained.get("recall"),
        "trained_accuracy": trained.get("accuracy"),
        "trained_human_review_rate": trained.get("human_review_rate"),
        "local_holdout_recall": holdout["holdout_recall"],
        "local_holdout": [(row["note"], row["score"], row["detected"]) for row in holdout["holdout"]],
        "local_test_fpr": local_fpr,
        "local_cpu_ms_per_pair": checkpoint["benchmark"].get("average_ms"),
        "production_approved": bool(checkpoint.get("production_approved")),
        "notes": notes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    problems = []
    if not checkpoint.get("production_approved") and not args.allow_trial and not args.candidate:
        problems.append("訓練時未同時達成召回率／準確度／時間目標（可用 --allow-trial 當試行模型安裝）")
    trained_holdout = trained.get("holdout_recall")
    if (
        trained_holdout is not None
        and holdout["holdout_recall"] is not None
        and holdout["holdout_recall"] + MAX_LOCAL_RECALL_DROP < float(trained_holdout)
    ):
        problems.append(
            f"本機固定測試集召回 {holdout['holdout_recall']:.0%} 明顯低於訓練回報的 {float(trained_holdout):.0%}，"
            "前處理可能與線上不一致"
        )
    if local_fpr is not None and local_fpr > dps.MAX_TEST_FPR:
        problems.append(f"本機無理賠測試誤報 {local_fpr:.0%} 高於 {dps.MAX_TEST_FPR:.0%}")
    if args.candidate:
        if problems:
            print("注意（仍存為候選，由管理員決定是否使用）：" + "；".join(problems))
        target = dps.CANDIDATE_ROOT / f"{args.candidate}.pt"
        if args.dry_run:
            print(f"驗證完成（--dry-run，未存到 {target}）")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".next.pt")
        torch.save(checkpoint, temporary)
        temporary.replace(target)
        print(f"已存為候選模型 {target}；到「模型管理 → 前後車損比對模型」可設為使用中")
        return
    if problems:
        raise SystemExit("不安裝：" + "；".join(problems))
    if args.dry_run:
        print("驗證通過（--dry-run，未安裝）")
        return
    if dps.MODEL_PATH.is_file():
        history = dps.MODEL_PATH.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dps.MODEL_PATH, history / f"damage_pair_before_import_{datetime.now():%Y%m%d%H%M%S}.pt")
    temporary = dps.MODEL_PATH.with_suffix(".next.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(dps.MODEL_PATH)
    print(f"已安裝到 {dps.MODEL_PATH}；API 會在下一次比對時自動重新載入")


if __name__ == "__main__":
    main()
