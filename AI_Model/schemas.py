from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VerificationResult:
    file_name: str
    status: str
    score: float
    details: str


@dataclass
class FeatureSample:
    file_name: str
    label: int
    features: Dict[str, float]


@dataclass
class PipelineResult:
    rows: List[VerificationResult]
    output_csv: str