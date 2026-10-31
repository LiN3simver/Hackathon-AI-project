"""Download and verify third-party model weights used by the local API."""

from pathlib import Path

import torch
import truststore

truststore.inject_into_ssl()

from torchvision.models.detection import (  # noqa: E402
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
print("Torchvision vehicle detector weights are ready.")

dinov2_cache = PROJECT_ROOT / ".cache" / "torch"
torch.hub.set_dir(str(dinov2_cache))
local_repository = dinov2_cache / "facebookresearch_dinov2_main"
if local_repository.exists():
    torch.hub.load(
        str(local_repository),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
else:
    torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        trust_repo=True,
        pretrained=True,
    )
print("DINOv2 ViT-S/14 view-classifier backbone weights are ready.")
