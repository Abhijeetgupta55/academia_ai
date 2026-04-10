from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_OUTPUT_CSV = Path("tampering_report.csv")
DEFAULT_EVALUATION_CSV = Path("tampering_evaluation.csv")

HEURISTIC_WEIGHTS = {
    "ela_mean": 0.14,
    "noise_inconsistency": 0.11,
    "blockiness": 0.09,
    "layout_irregularity": 0.15,
    "spacing_irregularity": 0.11,
    "font_inconsistency": 0.14,
    "hotspot_ratio": 0.14,
    "metadata_risk": 0.12,
}

STATUS_THRESHOLDS = {
    "authentic": 0.42,
    "suspicious": 0.7,
}
