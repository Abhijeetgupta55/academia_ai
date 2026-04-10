from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


class TabularNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _to_matrix(samples: List[Dict[str, float]], keys: List[str]) -> np.ndarray:
    m = np.zeros((len(samples), len(keys)), dtype=np.float32)
    for i, s in enumerate(samples):
        for j, k in enumerate(keys):
            m[i, j] = float(s.get(k, 0.0))
    return m


def _save_stats(path: Path, mean: np.ndarray, std: np.ndarray, keys: List[str]) -> None:
    np.savez(path, mean=mean, std=std, keys=np.array(keys))


def _load_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    keys = [str(k) for k in data["keys"].tolist()]
    return data["mean"], data["std"], keys


def train_model(
    feature_samples: List[Dict[str, float]],
    labels: List[int],
    artifacts_dir: Path,
    epochs: int = 100,
    lr: float = 1e-3,
) -> Dict[str, float]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise RuntimeError("PyTorch is required for training the deep model")
    keys = sorted({k for sample in feature_samples for k in sample.keys()})
    X = _to_matrix(feature_samples, keys)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    model_path = artifacts_dir / "deep_model.pt"
    stats_path = artifacts_dir / "feature_stats.npz"
    meta_path = artifacts_dir / "model_meta.json"
    _save_stats(stats_path, mean, std, keys)

    model = TabularNet(Xn.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(Xn, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)

    with torch.no_grad():
        probs = torch.sigmoid(model(x_t)).detach().cpu().numpy().reshape(-1)
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == np.array(labels)).mean())
    auc = float("nan")
    if len(set(labels)) > 1:
        pos = np.array(labels) == 1
        neg = np.array(labels) == 0
        auc = float(np.mean([float(p > n) for p in probs[pos] for n in probs[neg]])) if pos.any() and neg.any() else float("nan")

    meta = {
        "epochs": epochs,
        "learning_rate": lr,
        "feature_count": len(keys),
        "train_samples": len(labels),
        "train_accuracy": acc,
        "train_auc": auc,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def model_exists(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "deep_model.pt").exists() and (artifacts_dir / "feature_stats.npz").exists()


def predict_probability(feature_map: Dict[str, float], artifacts_dir: Path) -> float:
    model_path = artifacts_dir / "deep_model.pt"
    stats_path = artifacts_dir / "feature_stats.npz"
    if not model_path.exists() or not stats_path.exists():
        return 0.0

    if torch is None:
        return 0.0

    mean, std, keys = _load_stats(stats_path)
    x = np.array([float(feature_map.get(k, 0.0)) for k in keys], dtype=np.float32)
    x = (x - mean) / (std + 1e-6)

    model = TabularNet(len(keys))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        t = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        prob = torch.sigmoid(model(t)).cpu().numpy().reshape(-1)[0]
    return float(prob)
