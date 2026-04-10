from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import exifread
import numpy as np
from PIL import Image

from .ocr_engine import OCRWord, extract_words


def _safe_norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
    return gray


def ela_score(image_bgr: np.ndarray) -> float:
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mem = io.BytesIO()
    pil.save(mem, format="JPEG", quality=90)
    mem.seek(0)
    recompressed = Image.open(mem).convert("RGB")
    original = np.array(pil, dtype=np.float32)
    edited = np.array(recompressed, dtype=np.float32)
    diff = np.abs(original - edited)
    diff_gray = cv2.cvtColor(np.uint8(np.clip(diff * 10.0, 0, 255)), cv2.COLOR_RGB2GRAY)
    return float(np.mean(diff_gray))


def blockiness_score(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    v_edges = np.abs(np.diff(gray.astype(np.float32), axis=1))
    h_edges = np.abs(np.diff(gray.astype(np.float32), axis=0))
    v_lines = v_edges[:, 7::8]
    h_lines = h_edges[7::8, :]
    return float((v_lines.mean() + h_lines.mean()) / 2.0)


def noise_inconsistency(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    patch = 18
    step = 9
    vals: List[float] = []
    h, w = gray.shape
    for y in range(0, max(1, h - patch), step):
        for x in range(0, max(1, w - patch), step):
            vals.append(float(np.var(lap[y : y + patch, x : x + patch])))
    if not vals:
        return 0.0
    vals_arr = np.array(vals)
    if np.mean(vals_arr) <= 1e-6:
        return 0.0
    return float(np.std(vals_arr) / (np.mean(vals_arr) + 1e-6))


def hotspot_ratio(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blur)
    thr = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return float(np.sum(thr > 0) / thr.size)


def _layout_stats(words: List[OCRWord], shape: Tuple[int, int]) -> Dict[str, float]:
    if not words:
        return {
            "layout_irregularity": 1.0,
            "spacing_irregularity": 1.0,
            "font_inconsistency": 1.0,
            "text_density": 0.0,
            "ocr_conf_mean": 0.0,
            "ocr_conf_std": 1.0,
        }
    h, w = shape
    ys = np.array([word.y + word.h / 2 for word in words], dtype=np.float32)
    ws = np.array([word.w for word in words], dtype=np.float32)
    hs = np.array([word.h for word in words], dtype=np.float32)
    conf = np.array([word.conf for word in words], dtype=np.float32)
    ys_sorted = np.sort(ys)
    spacing = np.diff(ys_sorted) if len(ys_sorted) > 1 else np.array([0.0])
    area = float(np.sum(ws * hs))
    text_density = float(area / max(1.0, h * w))
    return {
        "layout_irregularity": float(np.std(ys) / max(1.0, h)),
        "spacing_irregularity": float(np.std(spacing) / max(1.0, np.mean(spacing) + 1e-6)),
        "font_inconsistency": float(np.std(hs) / max(1.0, np.mean(hs) + 1e-6)),
        "text_density": text_density,
        "ocr_conf_mean": float(np.mean(conf) / 100.0),
        "ocr_conf_std": float(np.std(conf) / 100.0),
    }


def metadata_risk(file_path: Path) -> float:
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        software = str(tags.get("Image Software", "")).lower()
        if not software:
            return 0.25
        bad = ["photoshop", "gimp", "adobe", "canva", "paint", "pixlr", "snapseed"]
        return 1.0 if any(token in software for token in bad) else 0.1
    except Exception:
        return 0.25


def extract_features(image_bgr: np.ndarray, file_path: Path) -> Dict[str, float]:
    gray = preprocess(image_bgr)
    words = extract_words(image_bgr)
    layout = _layout_stats(words, gray.shape)
    feats = {
        "ela_mean": _safe_norm(ela_score(image_bgr), 8.0, 65.0),
        "blockiness": _safe_norm(blockiness_score(gray), 2.0, 26.0),
        "noise_inconsistency": _safe_norm(noise_inconsistency(gray), 0.1, 2.2),
        "hotspot_ratio": _safe_norm(hotspot_ratio(gray), 0.005, 0.2),
        "layout_irregularity": _safe_norm(layout["layout_irregularity"], 0.008, 0.09),
        "spacing_irregularity": _safe_norm(layout["spacing_irregularity"], 0.12, 1.5),
        "font_inconsistency": _safe_norm(layout["font_inconsistency"], 0.08, 0.9),
        "metadata_risk": float(np.clip(metadata_risk(file_path), 0.0, 1.0)),
        "text_density": float(np.clip(layout["text_density"], 0.0, 1.0)),
        "ocr_conf_mean": float(np.clip(layout["ocr_conf_mean"], 0.0, 1.0)),
        "ocr_conf_std": float(np.clip(layout["ocr_conf_std"], 0.0, 1.0)),
        "word_count_norm": _safe_norm(float(len(words)), 10.0, 450.0),
    }
    return feats
