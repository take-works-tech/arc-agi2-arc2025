"""ProgramScorer 学習用トレーナ

generate_program_scorer_data.py で生成した JSONL
  {"features": [...], "label": float, "metadata": {...}}
を読み込み、ProgramScorer モデルを学習・保存する。
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ProgramScorer
from .dataset import ProgramScorerDataset


def load_program_scorer_samples(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """JSONL から ProgramScorer 用サンプルを読み込む"""
    samples: List[Dict[str, Any]] = []
    path = Path(jsonl_path)
    if not path.exists():
        print(f"[program_scorer_trainer] JSONL が見つかりません: {jsonl_path}")
        return samples

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "features" not in obj or "label" not in obj:
                continue
            samples.append(obj)

    print(f"[program_scorer_trainer] 読み込んだサンプル数: {len(samples)}")
    return samples


def train_program_scorer(
    jsonl_path: str,
    model_out_path: str,
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> None:
    """ProgramScorer モデルを学習するメイントレーニングループ"""
    samples = load_program_scorer_samples(jsonl_path, max_samples=max_samples)
    if not samples:
        print(f"[program_scorer_trainer] サンプルが空です: {jsonl_path}")
        return

    dataset = ProgramScorerDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 特徴次元を推定
    first_feats = samples[0]["features"]
    in_dim = len(first_feats)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ProgramScorer(in_dim=in_dim, hidden_dim=128).to(device)
    # ラベルは 0〜1 のスカラーなので BCE / MSE どちらでも良いが、ここでは MSE を使用
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[program_scorer_trainer] start training: "
        f"samples={len(dataset)}, in_dim={in_dim}, device={device}"
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for batch in dataloader:
            features = batch["features"].to(device)  # [N, in_dim]
            labels = batch["label"].to(device)       # [N]

            optimizer.zero_grad()
            pred = model(features).squeeze(-1)       # [N]
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            batch_size_eff = features.size(0)
            epoch_loss += loss.item() * batch_size_eff
            count += batch_size_eff

        avg_loss = epoch_loss / max(1, count)
        print(f"[program_scorer_trainer] epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4f}")

    # モデルを保存
    out_path = Path(model_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[program_scorer_trainer] モデルを保存しました: {out_path}")
