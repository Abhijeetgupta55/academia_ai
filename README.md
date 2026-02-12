## AI Academic Certificate  Authenticator
# Overview

The AI Certificate Verifier is a tool designed to detect fraudulent or forged academic certificates using OCR (Optical Character Recognition), OpenCV image analysis, and machine learning models.

The system identifies signs of tampering such as:

- Font mismatches

- Inconsistent text alignment and spacing

- Editing hotspots (regions where modifications are likely)

- Signs of image manipulation from editing software

- Noise and artifact analysis

#  Features

- OCR Extraction using pytesseract

- Preprocessing with OpenCV (thresholding, denoising, contour analysis)

- Forgery Detection using handcrafted + AI-based techniques

- Noise & Artifact Analysis

- Modular Pipeline (easy to extend with new detection techniques)

#  Repository Structure
```
ai-certificate-verifier/
├── AI_Model/               # Core source code for OCR + forgery detection pipeline
│   └── main.py             # Entry point for running the verifier
├── tampering_report.csv     # output report (file status, score, details)
├── requirements.txt         # Project dependencies
|___README.md                # Project documentation
```
# Getting Started
1. Clone the repository
```
git clone <repository-url>
cd academia_ai
```

# 2. Install dependencies
```
pip install -r requirements.txt
```
# 3. Run the pipeline
```
python AI_Model/main.py --input data/sample_fake.pdf
```

# Dependencies

- Python 3.8+

- OpenCV

- pytesseract

- numpy, pandas

- scikit-learn / tensorflow / pytorch (if ML models are used)

- matplotlib / seaborn (for visualization)

# Example Usage
```
from src.main import run_verification

result = run_verification("data/sample_fake.pdf")
print(result)
# Output: {"status": "fraudulent", "reasons": ["Font mismatch", "Editing hotspots detected"]}
```
# Sample Results
The verifier outputs results in a tabular format summarizing each certificate’s tampering status, score, and details of anomalies detected.
| File      | Tampering Status | Score | Details |
|-----------|-----------------|-------|---------|
| 1.img1    | Authentic Certificate |       |         |
| 2.img2    | Suspicious            |       |         |
| 3.img3    | Suspicious            |       |         |

# Future Work

- Deep learning–based forgery detection

- Larger dataset for improved model accuracy
