"""
グリッドエンコーダ

ARCグリッドをエンコードするトランスフォーマーベースのエンコーダ
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import math


class GridEncoder(nn.Module):
    """
    グリッドエンコーダ
    
    ARCグリッド（H x W）を受け取り、エンコードされた表現を出力
    """
    
    def __init__(
        self,
        input_channels: int = 10,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_grid_size: int = 30
    ):
        """初期化"""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_grid_size = max_grid_size
        
        # 色の埋め込み（0-9の10色）
        self.color_embedding = nn.Embedding(input_channels, embed_dim)
        
        # 位置エンコーディング
        self.row_embedding = nn.Embedding(max_grid_size, embed_dim)
        self.col_embedding = nn.Embedding(max_grid_size, embed_dim)
        
        # トランスフォーマーエンコーダ
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
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """順伝播
        
        Args:
            grid: グリッド [batch, height, width]
        
        Returns:
            encoded: エンコードされた表現 [batch, height*width, embed_dim]
        """
        batch_size, height, width = grid.shape
        
        # 色の埋め込み
        color_embed = self.color_embedding(grid)  # [batch, height, width, embed_dim]
        
        # 位置エンコーディング
        row_indices = torch.arange(height, device=grid.device).unsqueeze(1).expand(height, width)
        col_indices = torch.arange(width, device=grid.device).unsqueeze(0).expand(height, width)
        
        row_embed = self.row_embedding(row_indices)  # [height, width, embed_dim]
        col_embed = self.col_embedding(col_indices)  # [height, width, embed_dim]
        
        # 埋め込みを結合
        embed = color_embed + row_embed.unsqueeze(0) + col_embed.unsqueeze(0)  # [batch, height, width, embed_dim]
        
        # シーケンスに変換
        seq = embed.reshape(batch_size, height * width, self.embed_dim)  # [batch, height*width, embed_dim]
        
        # トランスフォーマーエンコーダ
        encoded = self.transformer(seq)  # [batch, height*width, embed_dim]
        
        # レイヤー正規化
        encoded = self.layer_norm(encoded)
        
        return encoded
    
    def encode_with_mask(
        self,
        grid: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """マスク付きエンコード
        
        Args:
            grid: グリッド
            mask: マスク（Trueの位置をマスク）
        
        Returns:
            encoded: エンコードされた表現
            attention_mask: アテンションマスク
        """
        encoded = self.forward(grid)
        
        if mask is not None:
            # マスクをシーケンス形式に変換
            batch_size, height, width = grid.shape
            attention_mask = mask.reshape(batch_size, height * width)
        else:
            attention_mask = None
        
        return encoded, attention_mask
