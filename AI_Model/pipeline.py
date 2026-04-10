from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_EVALUATION_CSV, DEFAULT_OUTPUT_CSV, HEURISTIC_WEIGHTS, STATUS_THRESHOLDS
from .features import extract_features
from .io_utils import list_inputs, load_document_pages
from .model import model_exists, predict_probability, train_model
from .schemas import FeatureSample, PipelineResult, VerificationResult


def _heuristic_probability(features: Dict[str, float]) -> float:
    total_w = float(sum(HEURISTIC_WEIGHTS.values()))
    if total_w <= 0:
        return 0.0
    score = 0.0
    for k, w in HEURISTIC_WEIGHTS.items():
        score += float(features.get(k, 0.0)) * w
    score += max(0.0, 0.5 - float(features.get("ocr_conf_mean", 0.0))) * 0.15
    score += max(0.0, 0.12 - float(features.get("text_density", 0.0))) * 1.8
    return float(np.clip(score / (total_w + 0.4), 0.0, 1.0))


def _status_from_probability(prob: float) -> str:
    if prob < STATUS_THRESHOLDS["authentic"]:
        return "Authentic Certificate"
    if prob < STATUS_THRESHOLDS["suspicious"]:
        return "Suspicious"
    return "Fraudulent"


def _reasons(features: Dict[str, float], final_prob: float) -> List[str]:
    reasons: List[str] = []
    if features.get("font_inconsistency", 0.0) > 0.55:
        reasons.append("Font mismatch pattern")
    if features.get("spacing_irregularity", 0.0) > 0.6:
        reasons.append("Inconsistent text alignment and spacing")
    if features.get("hotspot_ratio", 0.0) > 0.35:
        reasons.append("Editing hotspots detected")
    if features.get("ela_mean", 0.0) > 0.5:
        reasons.append("Image manipulation signature")
    if features.get("noise_inconsistency", 0.0) > 0.55:
        reasons.append("Noise and artifact inconsistency")
    if features.get("metadata_risk", 0.0) >= 0.9:
        reasons.append("Editing software signature in metadata")
    if features.get("ocr_conf_mean", 0.0) < 0.45:
        reasons.append("Unusual OCR confidence profile")
    if final_prob >= 0.7 and not reasons:
        reasons.append("Composite tampering risk above threshold")
    return reasons


def _aggregate_page_features(page_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for f in page_features for k in f.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([f.get(k, 0.0) for f in page_features]))
    return out


def _write_csv(rows: List[VerificationResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File", "Tampering Status", "Score", "Details"])
        for row in rows:
            w.writerow([row.file_name, row.status, f"{row.score:.2f}", row.details])


def _write_evaluation_csv(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File", "Expected", "Predicted", "Score", "Details"])
        for row in rows:
            w.writerow([row["file"], row["expected"], row["predicted"], f"{row['score']:.2f}", row["details"]])


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float | List[List[int]]]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = max(1, len(y_true))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _collect_training_samples(dataset_root: Path) -> List[FeatureSample]:
    classes = {
        "authentic": 0,
        "real": 0,
        "genuine": 0,
        "fraudulent": 1,
        "fake": 1,
        "tampered": 1,
        "suspicious": 1,
    }
    samples: List[FeatureSample] = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        label = classes.get(folder.name.lower())
        if label is None:
            continue
        files = list_inputs(str(folder), recursive=True)
        for file_path in files:
            pages = load_document_pages(file_path)
            feats = [extract_features(img, file_path) for _, img in pages]
            if not feats:
                continue
            agg = _aggregate_page_features(feats)
            samples.append(FeatureSample(file_name=file_path.name, label=label, features=agg))
    return samples


def _augment_sample(features: Dict[str, float], copies: int = 4) -> List[Dict[str, float]]:
    augmented = [dict(features)]
    rng = np.random.default_rng(42)
    numeric_keys = [
        "ela_mean",
        "blockiness",
        "noise_inconsistency",
        "hotspot_ratio",
        "layout_irregularity",
        "spacing_irregularity",
        "font_inconsistency",
        "metadata_risk",
        "text_density",
        "ocr_conf_mean",
        "ocr_conf_std",
        "word_count_norm",
    ]
    for _ in range(copies):
        perturbed = dict(features)
        for key in numeric_keys:
            value = float(perturbed.get(key, 0.0))
            noise = rng.normal(0.0, 0.03 if key != "metadata_risk" else 0.015)
            perturbed[key] = float(np.clip(value + noise, 0.0, 1.0))
        augmented.append(perturbed)
    return augmented


def train_pipeline(dataset_path: str, artifacts_dir: str | None = None, epochs: int = 120, lr: float = 1e-3) -> Dict[str, float]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    samples = _collect_training_samples(dataset_root)
    if len(samples) < 6:
        raise ValueError("Training dataset is too small. Provide at least 6 labeled certificates total.")

    X: List[Dict[str, float]] = []
    y: List[int] = []
    for sample in samples:
        for variant in _augment_sample(sample.features, copies=3):
            X.append(variant)
            y.append(sample.label)

    artifacts = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else DEFAULT_ARTIFACTS_DIR.resolve()
    meta = train_model(X, y, artifacts, epochs=epochs, lr=lr)
    return {
        **meta,
        "artifacts_dir": str(artifacts),
        "samples": len(samples),
        "positive_ratio": float(np.mean(y)),
    }


def evaluate_labeled_dataset(
    dataset_path: str,
    artifacts_dir: str | None = None,
    output_csv: str | None = None,
    recursive: bool = True,
) -> Dict[str, float]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    artifacts = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else DEFAULT_ARTIFACTS_DIR.resolve()
    out = Path(output_csv).expanduser().resolve() if output_csv else DEFAULT_EVALUATION_CSV.resolve()

    class_map = {
        "authentic": 0,
        "real": 0,
        "genuine": 0,
        "fraudulent": 1,
        "fake": 1,
        "tampered": 1,
        "suspicious": 1,
    }
    rows: List[dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        label = class_map.get(folder.name.lower())
        if label is None:
            continue
        for file_path in list_inputs(str(folder), recursive=recursive):
            pages = load_document_pages(file_path)
            feats = [extract_features(img, file_path) for _, img in pages]
            if not feats:
                continue
            merged = _aggregate_page_features(feats)
            heuristic_prob = _heuristic_probability(merged)
            deep_prob = predict_probability(merged, artifacts) if model_exists(artifacts) else 0.0
            final_prob = heuristic_prob * 0.55 + deep_prob * 0.45 if model_exists(artifacts) else heuristic_prob
            predicted = 1 if final_prob >= STATUS_THRESHOLDS["suspicious"] else 0
            y_true.append(label)
            y_pred.append(predicted)
            y_score.append(final_prob)
            rows.append(
                {
                    "file": file_path.name,
                    "expected": "Fraudulent" if label else "Authentic Certificate",
                    "predicted": "Fraudulent" if predicted else "Authentic Certificate",
                    "score": (1.0 - final_prob) * 100.0,
                    "details": "; ".join(_reasons(merged, final_prob)) or "No suspicious anomaly detected",
                }
            )

    _write_evaluation_csv(rows, out)
    if not y_true:
        raise ValueError("No labeled files found under the provided dataset path")

    metrics = {"samples": len(y_true), **_binary_metrics(y_true, y_pred), "evaluation_csv": str(out)}
    return metrics


def verify_pipeline(
    input_path: str,
    output_csv: str | None = None,
    artifacts_dir: str | None = None,
    recursive: bool = True,
) -> PipelineResult:
    files = list_inputs(input_path, recursive=recursive)
    artifacts = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else DEFAULT_ARTIFACTS_DIR.resolve()
    rows: List[VerificationResult] = []

    use_model = model_exists(artifacts)

    for file_path in files:
        pages = load_document_pages(file_path)
        feats = [extract_features(img, file_path) for _, img in pages]
        if not feats:
            continue
        merged = _aggregate_page_features(feats)

        heuristic_prob = _heuristic_probability(merged)
        deep_prob = predict_probability(merged, artifacts) if use_model else 0.0
        final_prob = heuristic_prob * 0.55 + deep_prob * 0.45 if use_model else heuristic_prob

        status = _status_from_probability(final_prob)
        score = float(np.clip((1.0 - final_prob) * 100.0, 0.0, 100.0))
        reasons = _reasons(merged, final_prob)

        rows.append(
            VerificationResult(
                file_name=file_path.name,
                status=status,
                score=score,
                details="; ".join(reasons) if reasons else "No suspicious anomaly detected",
            )
        )

    out = Path(output_csv).expanduser().resolve() if output_csv else DEFAULT_OUTPUT_CSV.resolve()
    _write_csv(rows, out)
    return PipelineResult(rows=rows, output_csv=str(out))
