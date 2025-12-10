"""
グリッドデコーダ

エンコードされたグリッド表現からグリッドを復元するU-Netスタイルのデコーダ
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridDecoder(nn.Module):
    """
    グリッドデコーダ

    U-Netスタイルのデコーダで、エンコードされたグリッド表現からグリッドを復元
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_colors: int = 10,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_grid_size: int = 30
    ):
        """初期化

        Args:
            embed_dim: 埋め込み次元
            num_colors: 色の数（0-9の10色）
            num_layers: デコーダレイヤー数
            dropout: ドロップアウト率
            max_grid_size: 最大グリッドサイズ
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_colors = num_colors
        self.max_grid_size = max_grid_size

        # デコーダレイヤー（U-Netスタイル）
        self.decoder_layers = nn.ModuleList()

        # 各レイヤーで次元を調整しながらグリッドを復元
        current_dim = embed_dim
        for i in range(num_layers):
            # アップサンプリング + 畳み込み
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_dim,
                        current_dim // 2 if i < num_layers - 1 else num_colors,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(current_dim // 2 if i < num_layers - 1 else num_colors),
                    nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                    nn.Dropout2d(dropout) if i < num_layers - 1 else nn.Identity()
                )
            )
            current_dim = current_dim // 2 if i < num_layers - 1 else num_colors

        # 最終的な色予測層
        self.color_predictor = nn.Sequential(
            nn.Conv2d(num_colors, num_colors, kernel_size=1),
            nn.Softmax(dim=1)  # 各ピクセルで色の確率分布を出力
        )

    def forward(
        self,
        encoded: torch.Tensor,
        target_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """順伝播

        Args:
            encoded: エンコードされた表現 [batch, seq_len, embed_dim]
            target_shape: 目標グリッドサイズ (height, width)（オプション）

        Returns:
            predicted_grid: 予測されたグリッド [batch, height, width, num_colors]
        """
        batch_size, seq_len, embed_dim = encoded.shape

        # 目標形状を決定
        if target_shape is None:
            # seq_lenからグリッドサイズを推論（平方根を取る）
            grid_size = int(seq_len ** 0.5)
            height = width = grid_size
        else:
            height, width = target_shape

        # シーケンスをグリッド形状に変換
        # [batch, seq_len, embed_dim] -> [batch, height, width, embed_dim]
        if seq_len == height * width:
            grid_encoded = encoded.reshape(batch_size, height, width, embed_dim)
        else:
            # パディングまたはトリミング
            if seq_len < height * width:
                # パディング
                padding_size = height * width - seq_len
                padding = torch.zeros(batch_size, padding_size, embed_dim, device=encoded.device)
                encoded = torch.cat([encoded, padding], dim=1)
            else:
                # トリミング
                encoded = encoded[:, :height * width, :]
            grid_encoded = encoded.reshape(batch_size, height, width, embed_dim)

        # [batch, height, width, embed_dim] -> [batch, embed_dim, height, width]
        x = grid_encoded.permute(0, 3, 1, 2)

        # デコーダレイヤーを適用
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        # 色予測
        color_logits = self.color_predictor(x)  # [batch, num_colors, height, width]

        # [batch, num_colors, height, width] -> [batch, height, width, num_colors]
        color_logits = color_logits.permute(0, 2, 3, 1)

        return color_logits
