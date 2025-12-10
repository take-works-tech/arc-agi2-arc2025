"""ProgramScorer モデル

候補プログラムの特徴ベクトルから「良さスコア」を予測する単純な MLP。
将来的には、オブジェクト列やプログラムトークン列を直接入力する
よりリッチなアーキテクチャに差し替えることを想定している。
"""

from typing import Optional

import torch
from torch import nn


class ProgramScorer(nn.Module):
    """
    features -> score の回帰モデル

    現段階では単純な MLP とし、特徴設計は呼び出し側に委ねる。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # スコアを 0.0〜1.0 に正規化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特徴テンソル [batch_size, in_dim]

        Returns:
            score: [batch_size, 1] のスコア（0.0〜1.0）
        """
        return self.net(x)
