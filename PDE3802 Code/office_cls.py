"""Shared helpers for the office-item classifier.

Kept deliberately small: the model wrapper, temperature scaling (a few lines of
NumPy, so no scipy/sklearn dependency) and image listing/reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str = "PDE3802 Code/Office_items_dataset") -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def read_image(path):
    """BGR image or None. imdecode (not imread) so non-ASCII Windows paths work."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Temperature scaling from probabilities.

    log(p) equals the logits minus a per-sample constant, and that constant
    cancels inside softmax, so softmax(log(p) / T) == softmax(logits / T).
    That lets us calibrate a model that only exposes softmax outputs.
    """
    if T == 1.0:
        return probs
    return softmax(np.log(np.clip(probs, 1e-12, 1.0)) / T)


class OfficeClassifier:
    """Thin wrapper around an Ultralytics classification model (.pt, .onnx, NCNN dir...)."""

    def __init__(self, weights, imgsz: int = 224, temperature: float | None = None, device=None):
        from ultralytics import YOLO  # imported here so `--help` stays fast

        self.weights = Path(weights)
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(str(weights), task="classify")
        names = self.model.names
        self.names = [names[i] for i in sorted(names)]
        self.temperature = temperature if temperature is not None else self._load_temperature()
        self.probs(np.zeros((imgsz, imgsz, 3), np.uint8))  # warm-up

    def _load_temperature(self) -> float:
        f = self.weights.parent / "calibration.json"
        if f.exists():
            return float(json.loads(f.read_text())["temperature"])
        return 1.0

    def probs(self, bgr: np.ndarray, calibrated: bool = True) -> np.ndarray:
        kw = {"imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(bgr, **kw)[0]
        p = r.probs.data.cpu().numpy().astype(np.float64)
        p = p / p.sum()
        return apply_temperature(p, self.temperature) if calibrated else p

    def describe(self, p: np.ndarray, topk: int = 3, min_conf: float = 0.0) -> dict:
        order = np.argsort(p)[::-1][:topk]
        top = [(self.names[i], float(p[i])) for i in order]
        label, conf = top[0]
        if conf < min_conf:
            label = "unsure"
        return {"class": label, "confidence": conf, "top": top}

    def classify(self, bgr: np.ndarray, topk: int = 3, min_conf: float = 0.0) -> dict:
        return self.describe(self.probs(bgr), topk, min_conf)
