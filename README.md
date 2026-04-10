# Academic Certificate Authenticator

Academic Certificate Authenticator is an OCR + forensic + deep-learning pipeline for identifying tampered academic certificates.

It detects anomalies such as font mismatch patterns, spacing inconsistencies, editing hotspots, metadata risk, and image artifacts, then combines heuristic and neural signals into a final authenticity decision.

## What Is Included

- OCR extraction using `pytesseract`
- Document preprocessing with OpenCV
- Artifact analysis with ELA, blockiness, and noise inconsistency
- Layout analysis for line spacing, font consistency, and OCR confidence shifts
- Deep tabular neural model (PyTorch) trained on extracted forensic features
- CSV reporting with per-document status, score, and anomaly details

## Project Structure

```text
academia_ai-1/
├── AI_Model/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── io_utils.py
│   ├── main.py
│   ├── model.py
│   ├── ocr_engine.py
│   ├── pipeline.py
│   └── types.py
├── requirements.txt
├── tampering_report.csv
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

You also need a local Tesseract installation available on PATH.

## Dataset Format For Training

Create a dataset root folder with class folders:

```text
data/train/
├── authentic/
│   ├── cert_001.png
│   └── cert_002.pdf
└── fraudulent/
	├── cert_101.png
	└── cert_102.pdf
```

Accepted class folder names:
- `authentic`, `real`, `genuine` → label `0`
- `fraudulent`, `fake`, `tampered`, `suspicious` → label `1`

## Train With A Small Dataset

Yes, you can start with about 20 images total.

For a very small dataset, use only the images you already know are real or fake and place them into class folders. The current training flow is designed to work as a demo-level bootstrap on small labeled sets, but it is still limited by data size.

Example:

```text
data/my_set/
├── authentic/
│   ├── real_1.png
│   ├── real_2.png
│   └── real_3.png
└── fraudulent/
	├── fake_1.png
	├── fake_2.png
	└── fake_3.png
```

If you only have 20 total images, that is enough to test the pipeline and build a basic classifier, but it is not enough for a reliable production detector.

## Train Deep Model

```bash
python AI_Model/main.py train --dataset data/train --artifacts artifacts --epochs 120 --lr 0.001
```

This generates:
- `artifacts/deep_model.pt`
- `artifacts/feature_stats.npz`
- `artifacts/model_meta.json`

## Evaluate Labeled Images

If you already know which images are real and fake, use evaluation mode directly:

```bash
python AI_Model/main.py evaluate --dataset data/my_set --artifacts artifacts --output tampering_evaluation.csv
```

This writes a labeled report with expected vs predicted results, plus accuracy, precision, recall, F1, and confusion matrix.

## Verify Certificates

Verify a single file:

```bash
python AI_Model/main.py verify --input data/sample_fake.pdf --output tampering_report.csv --artifacts artifacts
```

Verify a folder recursively:

```bash
python AI_Model/main.py verify --input data/incoming_certificates --output tampering_report.csv --artifacts artifacts
```

Backward-compatible shorthand is also supported:

```bash
python AI_Model/main.py --input data/incoming_certificates
```

## Output

`tampering_report.csv` columns:

- `File`
- `Tampering Status` (`Authentic Certificate`, `Suspicious`, `Fraudulent`)
- `Score` (trust score out of `100`)
- `Details` (human-readable anomaly reasons)

Example row:

```text
certificate_17.png,Suspicious,61.24,Font mismatch pattern; Editing hotspots detected
```

## Design Notes

- If trained artifacts exist, final risk uses both heuristic and neural probabilities.
- If no artifacts exist, the verifier runs fully in heuristic mode.
- PDFs are supported and evaluated page-wise before document-level aggregation.

## Next Expansion Ideas

- Add certificate template matching by institution
- Add signature/stamp verification modules
- Add web API and dashboard for batch uploads
