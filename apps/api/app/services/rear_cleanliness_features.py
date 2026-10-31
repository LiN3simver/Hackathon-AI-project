"""Frozen rear-seat patch preprocessing, identical to the training experiment."""

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def rear_patch_input(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    size = 448
    width, height = image.size
    ratio = size / max(width, height)
    rw, rh = round(width * ratio), round(height * ratio)
    left, top = (size - rw) // 2, (size - rh) // 2
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
    values = TF.normalize(TF.to_tensor(canvas), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    yy, xx = np.mgrid[: size // 14, : size // 14]
    x, y = ((xx + 0.5) * 14 - left) / rw, ((yy + 0.5) * 14 - top) / rh
    valid = (x >= 0) & (x < 1) & (y >= 0.2) & (y < 1)
    return values, torch.from_numpy(valid.ravel())


def rear_patch_score(tokens: torch.Tensor, checkpoint: dict) -> float:
    ridge = checkpoint["ridge_state"]
    scores = tokens @ ridge["coef"] + ridge["intercept"]
    return float(scores.topk(min(checkpoint["method_spec"]["top_k"], len(scores))).values.mean())
