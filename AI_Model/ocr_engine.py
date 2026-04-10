from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class OCRWord:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


def extract_words(image_bgr: np.ndarray, min_conf: float = 35.0) -> List[OCRWord]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    words: List[OCRWord] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf_raw = str(data["conf"][i]).strip()
        try:
            conf = float(conf_raw)
        except ValueError:
            conf = -1.0
        if not text or conf < min_conf:
            continue
        words.append(
            OCRWord(
                text=text,
                conf=conf,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
            )
        )
    return words
