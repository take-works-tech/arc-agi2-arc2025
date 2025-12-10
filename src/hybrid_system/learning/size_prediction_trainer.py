"""
出力グリッドサイズ予測タスク用のトレーナ

MLP を使って
  features -> [output_height, output_width]
を回帰する実装。
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .size_prediction_dataset import OutputSizePredictionDataset


class OutputSizeRegressor(nn.Module):
    """features -> (height, width) の MLP（本格実装）"""

    def __init__(self, in_dim: int = 9, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_output_size_regressor(
    output_root: str,
    batch_size: int = 64,
    num_epochs: int = 3,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> None:
    """
    出力サイズ予測モデルを学習するトレーニングループ（本格実装）

    Args:
        output_root: outputs ディレクトリ（grid_batch_*.json 群がある場所）
        batch_size: バッチサイズ
        num_epochs: エポック数（小さなテスト用途なのでデフォルト3）
        lr: 学習率
        device: "cuda" など（None の場合は自動判定）
    """
    dataset = OutputSizePredictionDataset(output_root)
    if len(dataset) == 0:
        print(f"[size_prediction_trainer] データセットが空です: {output_root}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = OutputSizeRegressor(in_dim=5, hidden_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[size_prediction_trainer] start training: samples={len(dataset)}, device={device}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for batch in dataloader:
            features = batch["features"].to(device)
            target = batch["target"].to(device).float()  # [N, 2]

            optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            batch_size_eff = features.size(0)
            epoch_loss += loss.item() * batch_size_eff
            count += batch_size_eff

        avg_loss = epoch_loss / max(1, count)
        print(f"[size_prediction_trainer] epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4f}")

    print("[size_prediction_trainer] training finished")
