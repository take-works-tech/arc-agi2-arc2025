"""
Contrastive Loss モジュール

対照学習の損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """対照学習の基本損失関数"""

    def __init__(self, temperature: float = 0.07):
        """
        初期化

        Args:
            temperature: 温度パラメータ（デフォルト: 0.07）
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        対照学習損失を計算

        Args:
            anchor: アンカー特徴量 [batch, dim]
            positive: 正例特徴量 [batch, dim]
            negatives: 負例特徴量 [batch, num_negatives, dim]（オプション）

        Returns:
            loss: 損失値
        """
        # 正規化
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # 正例との類似度
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)  # [batch]
        pos_sim = pos_sim / self.temperature

        if negatives is not None:
            # 負例との類似度
            negatives = F.normalize(negatives, p=2, dim=2)  # [batch, num_negatives, dim]
            anchor_expanded = anchor.unsqueeze(1)  # [batch, 1, dim]
            neg_sim = torch.bmm(anchor_expanded, negatives.transpose(1, 2)).squeeze(1)  # [batch, num_negatives]
            neg_sim = neg_sim / self.temperature

            # InfoNCE損失
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + num_negatives]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # シンプルな対照学習損失（正例のみ）
            loss = -torch.log(torch.sigmoid(pos_sim)).mean()

        return loss


class InfoNCE(nn.Module):
    """InfoNCE損失（対照学習の標準的な損失関数）"""

    def __init__(self, temperature: float = 0.07):
        """
        初期化

        Args:
            temperature: 温度パラメータ（デフォルト: 0.07）
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE損失を計算

        Args:
            query: クエリ特徴量 [batch, dim]
            positive: 正例特徴量 [batch, dim]
            negatives: 負例特徴量 [batch, num_negatives, dim]

        Returns:
            loss: 損失値
        """
        # 正規化
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=2)

        # 正例との類似度
        pos_sim = F.cosine_similarity(query, positive, dim=1)  # [batch]
        pos_sim = pos_sim / self.temperature

        # 負例との類似度
        query_expanded = query.unsqueeze(1)  # [batch, 1, dim]
        neg_sim = torch.bmm(query_expanded, negatives.transpose(1, 2)).squeeze(1)  # [batch, num_negatives]
        neg_sim = neg_sim / self.temperature

        # InfoNCE損失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + num_negatives]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class SimCLRLoss(nn.Module):
    """SimCLR損失（対照学習の標準的な損失関数）"""

    def __init__(self, temperature: float = 0.07):
        """
        初期化

        Args:
            temperature: 温度パラメータ（デフォルト: 0.07）
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        SimCLR損失を計算（バッチ内の他のサンプルを負例として使用）

        Args:
            z1: 拡張1の特徴量 [batch, dim]
            z2: 拡張2の特徴量 [batch, dim]

        Returns:
            loss: 損失値
        """
        batch_size = z1.size(0)

        # 正規化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # すべての特徴量を結合
        all_features = torch.cat([z1, z2], dim=0)  # [2*batch, dim]

        # 類似度行列を計算
        similarity_matrix = torch.matmul(all_features, all_features.t()) / self.temperature  # [2*batch, 2*batch]

        # 対角成分をマスク（自己類似度を除外）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # 正例のマスク（z1[i]とz2[i]が正例ペア）
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2*batch]

        # 損失を計算
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
