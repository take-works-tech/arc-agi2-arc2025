"""
Vision Transformer (ViT) Grid Encoder

ViTアーキテクチャを使用したグリッドエンコーダー
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """パッチ埋め込みモジュール"""

    def __init__(
        self,
        patch_size: int = 2,
        embed_dim: int = 256,
        input_channels: int = 10
    ):
        """
        初期化

        Args:
            patch_size: パッチサイズ（デフォルト: 2）
            embed_dim: 埋め込み次元
            input_channels: 入力チャネル数（色数）
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # パッチを埋め込みに変換
        # 各パッチは patch_size x patch_size のピクセル
        patch_dim = input_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        グリッドをパッチに分割して埋め込みに変換

        Args:
            grid: グリッド [batch, height, width]

        Returns:
            patch_embeddings: パッチ埋め込み [batch, num_patches, embed_dim]
        """
        batch_size, height, width = grid.shape

        # パッチサイズで割り切れるようにパディング
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            grid = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)
            height += pad_h
            width += pad_w

        # パッチに分割
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # グリッドをパッチに分割
        patches = grid.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # [batch, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.contiguous().view(batch_size, num_patches, self.patch_size * self.patch_size)
        # [batch, num_patches, patch_size * patch_size]

        # 色の埋め込みを適用（各ピクセルの色を埋め込みに変換）
        # 簡易版: パッチ内のピクセル値を直接使用
        # より高度な実装では、色の埋め込みを使用可能
        patch_features = patches.float()  # [batch, num_patches, patch_size * patch_size]

        # 埋め込みに投影
        patch_embeddings = self.projection(patch_features)  # [batch, num_patches, embed_dim]

        return patch_embeddings, (num_patches_h, num_patches_w)


class ViTGridEncoder(nn.Module):
    """
    Vision Transformer Grid Encoder

    ViTアーキテクチャを使用してARCグリッドをエンコード
    """

    def __init__(
        self,
        input_channels: int = 10,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 2,
        dropout: float = 0.1,
        max_num_patches: int = 256
    ):
        """
        初期化

        Args:
            input_channels: 入力チャネル数（色数）
            embed_dim: 埋め込み次元
            num_layers: Transformerレイヤー数
            num_heads: アテンションヘッド数
            patch_size: パッチサイズ
            dropout: ドロップアウト率
            max_num_patches: 最大パッチ数
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches

        # パッチ埋め込み
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            input_channels=input_channels
        )

        # クラストークン（オプション、グリッド全体の表現として使用）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 位置エンコーディング
        self.position_embedding = nn.Parameter(torch.randn(1, max_num_patches + 1, embed_dim))

        # Transformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            grid: グリッド [batch, height, width]

        Returns:
            encoded: エンコードされた表現 [batch, num_patches+1, embed_dim]
        """
        batch_size = grid.size(0)

        # パッチ埋め込み
        patch_embeddings, (num_patches_h, num_patches_w) = self.patch_embedding(grid)
        # [batch, num_patches, embed_dim]

        num_patches = patch_embeddings.size(1)

        # クラストークンを追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # [batch, num_patches+1, embed_dim]

        # 位置エンコーディングを追加
        embeddings = embeddings + self.position_embedding[:, :num_patches+1, :]

        # ドロップアウト
        embeddings = self.dropout(embeddings)

        # Transformerエンコーダー
        encoded = self.transformer(embeddings)  # [batch, num_patches+1, embed_dim]

        # レイヤー正規化
        encoded = self.layer_norm(encoded)

        return encoded

    def get_cls_representation(self, grid: torch.Tensor) -> torch.Tensor:
        """
        クラストークンの表現を取得（グリッド全体の表現）

        Args:
            grid: グリッド [batch, height, width]

        Returns:
            cls_representation: クラストークンの表現 [batch, embed_dim]
        """
        encoded = self.forward(grid)  # [batch, num_patches+1, embed_dim]
        cls_representation = encoded[:, 0, :]  # [batch, embed_dim]
        return cls_representation

    def get_patch_representations(self, grid: torch.Tensor) -> torch.Tensor:
        """
        パッチ表現を取得（クラストークンを除く）

        Args:
            grid: グリッド [batch, height, width]

        Returns:
            patch_representations: パッチ表現 [batch, num_patches, embed_dim]
        """
        encoded = self.forward(grid)  # [batch, num_patches+1, embed_dim]
        patch_representations = encoded[:, 1:, :]  # [batch, num_patches, embed_dim]
        return patch_representations
