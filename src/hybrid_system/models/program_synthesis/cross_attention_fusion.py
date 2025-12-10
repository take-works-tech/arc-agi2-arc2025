"""
Cross-Attention between Input/Output の強化

入出力の融合を強化するCross-Attentionモジュール
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合モジュール"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        """
        初期化

        Args:
            embed_dim: 埋め込み次元
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            layer_norm_eps: LayerNormのepsilon
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 出力投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            query: Queryテンソル [batch, seq_len_q, embed_dim]（出力側）
            key: Keyテンソル [batch, seq_len_k, embed_dim]（入力側）
            value: Valueテンソル [batch, seq_len_k, embed_dim]（入力側）
            key_padding_mask: Keyのパディングマスク [batch, seq_len_k]
            attn_mask: アテンションマスク [seq_len_q, seq_len_k]

        Returns:
            torch.Tensor: 融合されたテンソル [batch, seq_len_q, embed_dim]
        """
        # 残差接続用に保存
        residual = query

        # LayerNorm
        query = self.norm1(query)
        key = self.norm1(key)
        value = self.norm1(value)

        # Query, Key, Value投影
        Q = self.q_proj(query)  # [batch, seq_len_q, embed_dim]
        K = self.k_proj(key)    # [batch, seq_len_k, embed_dim]
        V = self.v_proj(value)  # [batch, seq_len_k, embed_dim]

        # Multi-head attention
        batch_size, seq_len_q, _ = Q.shape
        seq_len_k = K.shape[1]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len_k, head_dim]
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len_k, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, num_heads, seq_len_q, seq_len_k]

        # アテンションマスクを適用
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Key padding maskを適用
        if key_padding_mask is not None:
            # [batch, seq_len_k] -> [batch, 1, 1, seq_len_k]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len_q, seq_len_k]
        attn_weights = self.dropout(attn_weights)

        # アテンション適用
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len_q, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len_q, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)  # [batch, seq_len_q, embed_dim]

        # 出力投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 残差接続
        output = residual + attn_output

        # Feed-forward network
        residual = output
        output = self.norm2(output)
        output = self.ffn(output)
        output = residual + output

        return output


class InputOutputFusion(nn.Module):
    """入出力融合モジュール（複数のCross-Attention層をスタック）"""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Cross-Attention層をスタック
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionFusion(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        input_embed: torch.Tensor,
        output_embed: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            input_embed: 入力埋め込み [batch, seq_len_input, embed_dim]
            output_embed: 出力埋め込み [batch, seq_len_output, embed_dim]
            input_mask: 入力マスク [batch, seq_len_input]
            output_mask: 出力マスク [batch, seq_len_output]

        Returns:
            torch.Tensor: 融合された出力埋め込み [batch, seq_len_output, embed_dim]
        """
        # 出力側をQuery、入力側をKey/ValueとしてCross-Attention
        fused_output = output_embed

        for cross_attn in self.cross_attention_layers:
            fused_output = cross_attn(
                query=fused_output,
                key=input_embed,
                value=input_embed,
                key_padding_mask=input_mask
            )

        return fused_output


class BidirectionalInputOutputFusion(nn.Module):
    """双方向入出力融合モジュール"""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 入力→出力方向のCross-Attention
        self.input_to_output = InputOutputFusion(embed_dim, num_layers, num_heads, dropout)

        # 出力→入力方向のCross-Attention
        self.output_to_input = InputOutputFusion(embed_dim, num_layers, num_heads, dropout)

        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        input_embed: torch.Tensor,
        output_embed: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        フォワードパス

        Args:
            input_embed: 入力埋め込み [batch, seq_len_input, embed_dim]
            output_embed: 出力埋め込み [batch, seq_len_output, embed_dim]
            input_mask: 入力マスク [batch, seq_len_input]
            output_mask: 出力マスク [batch, seq_len_output]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (
                融合された入力埋め込み [batch, seq_len_input, embed_dim],
                融合された出力埋め込み [batch, seq_len_output, embed_dim]
            )
        """
        # 入力→出力方向
        output_attended = self.input_to_output(input_embed, output_embed, input_mask, output_mask)

        # 出力→入力方向
        input_attended = self.output_to_input(output_embed, input_embed, output_mask, input_mask)

        # 双方向の情報を融合（本格実装）
        # 元の埋め込みとアテンション結果を適切に融合
        input_fused = self._fuse_embeddings(input_embed, input_attended)
        output_fused = self._fuse_embeddings(output_embed, output_attended)

        return input_fused, output_fused

    def _fuse_embeddings(
        self,
        original: torch.Tensor,
        attended: torch.Tensor
    ) -> torch.Tensor:
        """
        埋め込みを融合（本格実装）

        元の埋め込みとアテンション結果を適切に融合

        Args:
            original: 元の埋め込み [batch, seq_len, embed_dim]
            attended: アテンション結果 [batch, seq_len, embed_dim]

        Returns:
            torch.Tensor: 融合された埋め込み [batch, seq_len, embed_dim]
        """
        # 既存のfusion_layerを使用して融合
        # 元の埋め込みとアテンション結果を連結
        concat = torch.cat([original, attended], dim=-1)  # [batch, seq_len, embed_dim * 2]

        # 融合層を通して融合
        fused = self.fusion_layer(concat)  # [batch, seq_len, embed_dim]

        # 残差接続を追加（元の埋め込みとの重み付き和）
        fused = 0.5 * original + 0.5 * fused

        return fused
