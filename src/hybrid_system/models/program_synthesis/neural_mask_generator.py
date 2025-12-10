"""
Neural Mask Generator モジュール

プログラム探索前処理としてマスク生成（補助専用）
前景/背景マスク、対称性マップ、オブジェクトヒートマップを生成
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralMaskGenerator(nn.Module):
    """ニューラルマスク生成器"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            input_dim: 入力特徴量の次元（グリッドエンコーダーの出力次元）
            hidden_dim: 隠れ層の次元
            num_layers: レイヤー数
            dropout: ドロップアウト率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # エンコーダー（グリッド特徴量をエンコード）
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # マスク生成ヘッド
        # 前景/背景マスク: 各ピクセルが前景か背景かを予測
        self.foreground_mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 対称性マップ: 対称性の強度を予測
        self.symmetry_map_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # オブジェクトヒートマップ: オブジェクトの存在確率を予測
        self.object_heatmap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        grid_features: torch.Tensor,
        grid_shape: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        フォワードパス

        Args:
            grid_features: グリッド特徴量 [batch, H*W, input_dim] または [batch, input_dim]
            grid_shape: グリッド形状 (H, W)

        Returns:
            masks: {
                'foreground_mask': [batch, H, W],
                'symmetry_map': [batch, H, W],
                'object_heatmap': [batch, H, W]
            }
        """
        H, W = grid_shape
        batch_size = grid_features.shape[0]

        # グリッド特徴量をエンコード
        if grid_features.dim() == 2:
            # [batch, input_dim] -> [batch, H*W, hidden_dim]
            # グリッド全体の特徴量を各ピクセルに展開
            encoded = self.encoder(grid_features)  # [batch, hidden_dim]
            encoded = encoded.unsqueeze(1).expand(-1, H * W, -1)  # [batch, H*W, hidden_dim]
        else:
            # [batch, H*W, input_dim] -> [batch, H*W, hidden_dim]
            encoded = self.encoder(grid_features)

        # マスクを生成
        foreground_mask = self.foreground_mask_head(encoded)  # [batch, H*W, 1]
        symmetry_map = self.symmetry_map_head(encoded)  # [batch, H*W, 1]
        object_heatmap = self.object_heatmap_head(encoded)  # [batch, H*W, 1]

        # 形状を [batch, H, W] に変換
        foreground_mask = foreground_mask.squeeze(-1).view(batch_size, H, W)
        symmetry_map = symmetry_map.squeeze(-1).view(batch_size, H, W)
        object_heatmap = object_heatmap.squeeze(-1).view(batch_size, H, W)

        return {
            'foreground_mask': foreground_mask,
            'symmetry_map': symmetry_map,
            'object_heatmap': object_heatmap
        }

    def _extract_detailed_grid_features(self, grid: np.ndarray) -> torch.Tensor:
        """
        グリッドから詳細な特徴量を抽出（本格実装）

        Args:
            grid: 入力グリッド [H, W]

        Returns:
            features: 特徴量テンソル [1, H*W, feature_dim]
        """
        H, W = grid.shape
        features_list = []

        # 各ピクセルごとの特徴量を計算
        for i in range(H):
            for j in range(W):
                pixel_features = []

                # 1. 基本特徴（色、位置）
                color = float(grid[i, j])
                normalized_x = j / max(W - 1, 1)  # 正規化されたX座標
                normalized_y = i / max(H - 1, 1)  # 正規化されたY座標
                pixel_features.extend([color, normalized_x, normalized_y])

                # 2. 近傍特徴（3x3近傍の色分布）
                neighbor_colors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            neighbor_colors.append(float(grid[ni, nj]))
                        else:
                            neighbor_colors.append(0.0)  # 境界外は0

                # 近傍の色の統計（平均、分散、最大、最小）
                neighbor_colors_array = np.array(neighbor_colors)
                neighbor_mean = float(np.mean(neighbor_colors_array))
                neighbor_std = float(np.std(neighbor_colors_array))
                neighbor_max = float(np.max(neighbor_colors_array))
                neighbor_min = float(np.min(neighbor_colors_array))
                pixel_features.extend([neighbor_mean, neighbor_std, neighbor_max, neighbor_min])

                # 3. エッジ特徴（色の変化）
                edge_strength = 0.0
                if i > 0:
                    edge_strength += abs(float(grid[i, j]) - float(grid[i - 1, j]))
                if i < H - 1:
                    edge_strength += abs(float(grid[i, j]) - float(grid[i + 1, j]))
                if j > 0:
                    edge_strength += abs(float(grid[i, j]) - float(grid[i, j - 1]))
                if j < W - 1:
                    edge_strength += abs(float(grid[i, j]) - float(grid[i, j + 1]))
                edge_strength = edge_strength / 4.0  # 正規化
                pixel_features.append(edge_strength)

                # 4. テクスチャ特徴（局所的な色の変化）
                # 5x5近傍の色の分散
                texture_variance = 0.0
                texture_colors = []
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            texture_colors.append(float(grid[ni, nj]))
                if len(texture_colors) > 0:
                    texture_variance = float(np.var(texture_colors))
                pixel_features.append(texture_variance)

                # 5. グローバル統計（グリッド全体の色分布との関係）
                # グリッド全体の色のヒストグラム
                grid_flat = grid.flatten()
                color_hist = np.bincount(grid_flat.astype(int), minlength=10) / len(grid_flat)
                # 現在のピクセルの色がどの程度一般的か
                color_frequency = color_hist[int(color)] if int(color) < len(color_hist) else 0.0
                pixel_features.append(color_frequency)

                # 6. 位置特徴（中心からの距離、対称性）
                center_x = W / 2.0
                center_y = H / 2.0
                dist_from_center = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                normalized_dist = dist_from_center / max(max_dist, 1.0)
                pixel_features.append(normalized_dist)

                # X軸対称位置の色
                symmetric_x_j = W - 1 - j
                symmetric_x_color = float(grid[i, symmetric_x_j]) if 0 <= symmetric_x_j < W else 0.0
                x_symmetry_score = 1.0 if abs(color - symmetric_x_color) < 0.5 else 0.0
                pixel_features.append(x_symmetry_score)

                # Y軸対称位置の色
                symmetric_y_i = H - 1 - i
                symmetric_y_color = float(grid[symmetric_y_i, j]) if 0 <= symmetric_y_i < H else 0.0
                y_symmetry_score = 1.0 if abs(color - symmetric_y_color) < 0.5 else 0.0
                pixel_features.append(y_symmetry_score)

                features_list.append(pixel_features)

        # 特徴量をテンソルに変換
        features_array = np.array(features_list, dtype=np.float32)  # [H*W, feature_dim]
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)  # [1, H*W, feature_dim]

        return features_tensor

    def generate_masks(
        self,
        grid: np.ndarray,
        grid_encoder: Optional[nn.Module] = None
    ) -> Dict[str, np.ndarray]:
        """
        グリッドからマスクを生成（推論用）

        Args:
            grid: 入力グリッド [H, W]
            grid_encoder: グリッドエンコーダー（Noneの場合は簡易特徴量を使用）

        Returns:
            masks: {
                'foreground_mask': [H, W],
                'symmetry_map': [H, W],
                'object_heatmap': [H, W]
            }
        """
        self.eval()
        H, W = grid.shape

        with torch.no_grad():
            if grid_encoder is not None:
                # グリッドエンコーダーを使用
                grid_tensor = torch.tensor(grid, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                grid_features = grid_encoder(grid_tensor)  # [1, H*W, embed_dim] または [1, embed_dim]
                if grid_features.dim() == 4:
                    # [1, embed_dim, H, W] -> [1, H*W, embed_dim]
                    grid_features = grid_features.view(1, -1, H * W).transpose(1, 2)
                elif grid_features.dim() == 2:
                    # [1, embed_dim] -> そのまま使用
                    pass
                else:
                    # [1, H*W, embed_dim] -> そのまま使用
                    pass
            else:
                # 本格実装: グリッドエンコーダーがない場合の詳細特徴量抽出
                grid_features = self._extract_detailed_grid_features(grid)  # [1, H*W, feature_dim]
                # 入力次元に合わせて投影
                if grid_features.shape[-1] != self.input_dim:
                    if not hasattr(self, '_simple_proj'):
                        self._simple_proj = nn.Linear(grid_features.shape[-1], self.input_dim).to(grid_features.device)
                    grid_features = self._simple_proj(grid_features)  # [1, H*W, input_dim]

            # マスクを生成
            masks = self.forward(grid_features, (H, W))

            # NumPy配列に変換
            result = {
                'foreground_mask': masks['foreground_mask'][0].cpu().numpy(),
                'symmetry_map': masks['symmetry_map'][0].cpu().numpy(),
                'object_heatmap': masks['object_heatmap'][0].cpu().numpy()
            }

        return result


class MaskBasedProgramGuider:
    """マスクベースプログラムガイダー（プログラム探索の前処理）"""

    def __init__(
        self,
        mask_generator: Optional[NeuralMaskGenerator] = None,
        foreground_threshold: float = 0.5,
        symmetry_threshold: float = 0.6,
        object_threshold: float = 0.5
    ):
        """
        初期化

        Args:
            mask_generator: マスク生成器
            foreground_threshold: 前景マスクの閾値
            symmetry_threshold: 対称性マップの閾値
            object_threshold: オブジェクトヒートマップの閾値
        """
        self.mask_generator = mask_generator
        self.foreground_threshold = foreground_threshold
        self.symmetry_threshold = symmetry_threshold
        self.object_threshold = object_threshold

    def guide_program_search(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        grid_encoder: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        プログラム探索をガイドする情報を生成

        Args:
            input_grid: 入力グリッド [H, W]
            output_grid: 出力グリッド [H, W]
            grid_encoder: グリッドエンコーダー

        Returns:
            guidance: {
                'foreground_regions': List[Tuple[int, int]],  # 前景領域の座標
                'symmetry_regions': List[Tuple[int, int]],  # 対称性領域の座標
                'object_regions': List[Tuple[int, int]],  # オブジェクト領域の座標
                'suggested_operations': List[str],  # 推奨される操作
            }
        """
        if self.mask_generator is None:
            # マスク生成器がない場合は空のガイダンスを返す
            return {
                'foreground_regions': [],
                'symmetry_regions': [],
                'object_regions': [],
                'suggested_operations': []
            }

        # 入力グリッドからマスクを生成
        input_masks = self.mask_generator.generate_masks(input_grid, grid_encoder)

        # 出力グリッドからマスクを生成
        output_masks = self.mask_generator.generate_masks(output_grid, grid_encoder)

        # マスクから領域を抽出
        foreground_regions = self._extract_regions(
            input_masks['foreground_mask'], self.foreground_threshold
        )
        symmetry_regions = self._extract_regions(
            input_masks['symmetry_map'], self.symmetry_threshold
        )
        object_regions = self._extract_regions(
            input_masks['object_heatmap'], self.object_threshold
        )

        # 推奨操作を生成
        suggested_operations = self._suggest_operations(
            foreground_regions, symmetry_regions, object_regions
        )

        return {
            'foreground_regions': foreground_regions,
            'symmetry_regions': symmetry_regions,
            'object_regions': object_regions,
            'suggested_operations': suggested_operations,
            'input_masks': input_masks,
            'output_masks': output_masks
        }

    def _extract_regions(
        self,
        mask: np.ndarray,
        threshold: float
    ) -> List[Tuple[int, int]]:
        """
        マスクから領域を抽出

        Args:
            mask: マスク [H, W]
            threshold: 閾値

        Returns:
            regions: 領域の座標リスト
        """
        regions = []
        H, W = mask.shape

        for i in range(H):
            for j in range(W):
                if mask[i, j] >= threshold:
                    regions.append((i, j))

        return regions

    def _suggest_operations(
        self,
        foreground_regions: List[Tuple[int, int]],
        symmetry_regions: List[Tuple[int, int]],
        object_regions: List[Tuple[int, int]]
    ) -> List[str]:
        """
        領域情報から推奨操作を生成

        Args:
            foreground_regions: 前景領域
            symmetry_regions: 対称性領域
            object_regions: オブジェクト領域

        Returns:
            suggested_operations: 推奨操作のリスト
        """
        operations = []

        # 対称性領域がある場合、対称性操作を推奨
        if len(symmetry_regions) > 0:
            operations.append('MIRROR_X')
            operations.append('MIRROR_Y')
            operations.append('ROTATE')

        # オブジェクト領域がある場合、オブジェクト操作を推奨
        if len(object_regions) > 0:
            operations.append('GET_ALL_OBJECTS')
            operations.append('FILTER')
            operations.append('MOVE')

        # 前景領域がある場合、前景操作を推奨
        if len(foreground_regions) > 0:
            operations.append('SET_COLOR')
            operations.append('FILL')

        return list(set(operations))  # 重複を除去
