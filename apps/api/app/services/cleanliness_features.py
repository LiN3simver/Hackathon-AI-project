"""Frozen CLIP preprocessing and per-image Lab features used by the deployed head."""

import cv2
import numpy as np
import torch
from PIL import Image


def color_features(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR))
    lab_u8 = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
    lab = lab_u8.astype(np.float32)
    median = cv2.medianBlur(lab_u8, 21).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:] - median[:, :, 1:], axis=2)
    light = np.abs(lab[:, :, 0] - median[:, :, 0])
    features = []
    heat = np.zeros((256, 256), np.float32)
    for y0, y1 in ((0.25, 0.65), (0.60, 1.0)):
        for x0, x1 in ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0)):
            ys, xs = slice(round(y0 * 256), round(y1 * 256)), slice(round(x0 * 256), round(x1 * 256))
            region = lab[ys, xs]
            ab = region[:, :, 1:].reshape(-1, 2)
            palette = np.median(ab, axis=0)
            mad = np.maximum(np.median(np.abs(ab - palette), axis=0) * 1.4826, 4.0)
            deviation = np.linalg.norm((region[:, :, 1:] - palette) / mad, axis=2)
            local_chroma, local_light = chroma[ys, xs], light[ys, xs]
            score = 0.5 * deviation + local_chroma / 12.0 + local_light / 50.0
            heat[ys, xs] = np.maximum(heat[ys, xs], score)
            features.extend(
                [
                    *palette / 255.0,
                    *mad / 255.0,
                    *np.quantile(local_chroma, [0.5, 0.9, 0.99]) / 40.0,
                    *np.quantile(local_light, [0.5, 0.9, 0.99]) / 100.0,
                    float((deviation > 3).mean()),
                    float((local_chroma > 15).mean()),
                    float((local_light > 45).mean()),
                ]
            )
    return np.asarray(features, np.float32), heat


def clip_pixels(processor, image: Image.Image) -> torch.Tensor:
    width, height = image.size
    ratio = 224 / max(width, height)
    resized = image.resize((round(width * ratio), round(height * ratio)), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), (127, 127, 127))
    canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return processor(images=canvas, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]


def clip_embedding(value) -> torch.Tensor:
    return value if torch.is_tensor(value) else value.pooler_output


def fused_features(embedding: np.ndarray, color: np.ndarray) -> np.ndarray:
    def unit(values):
        return values / max(float(np.linalg.norm(values)), 1e-8)

    return unit(np.concatenate([unit(embedding), unit(color)]))
