from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from .config import IMAGE_EXTENSIONS, PDF_EXTENSIONS


def list_inputs(input_path: str, recursive: bool = True) -> List[Path]:
    p = Path(input_path).expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    pattern = "**/*" if recursive else "*"
    files = [f for f in p.glob(pattern) if f.is_file()]
    allowed = IMAGE_EXTENSIONS | PDF_EXTENSIONS
    return [f for f in files if f.suffix.lower() in allowed]


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_document_pages(path: Path, dpi: int = 220) -> List[Tuple[str, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Unable to read image: {path}")
        return [(path.name, img)]
    if suffix in PDF_EXTENSIONS:
        pages = convert_from_path(str(path), dpi=dpi)
        return [(f"{path.name}::page_{idx + 1}", _pil_to_bgr(page)) for idx, page in enumerate(pages)]
    raise ValueError(f"Unsupported file type: {path}")


def flatten_feature_keys(samples: Iterable[dict]) -> List[str]:
    keys = set()
    for sample in samples:
        keys.update(sample.keys())
    return sorted(keys)
