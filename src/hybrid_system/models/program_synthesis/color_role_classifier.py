"""
色役割分類モジュール

色の役割（前景/背景/構造色など）を分類
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from src.data_systems.data_models.core.object import Object


class ColorRoleClassifier(nn.Module):
    """色役割分類器"""

    def __init__(
        self,
        feature_dim: int = 20,
        hidden_dim: int = 64,
        num_roles: int = 4,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            feature_dim: 特徴量の次元
            hidden_dim: 隠れ層の次元
            num_roles: 色役割の数（background, foreground, structure, other）
            dropout: ドロップアウト率
        """
        super().__init__()
        self.num_roles = num_roles

        # 色役割の定義
        self.role_names = ['background', 'foreground', 'structure', 'other']

        # 特徴量投影
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # 色役割分類ネットワーク
        self.role_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_roles)
        )

    def forward(
        self,
        color_features: torch.Tensor
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            color_features: 色特徴量 [batch, feature_dim] または [num_colors, feature_dim]

        Returns:
            torch.Tensor: 色役割分類スコア [batch, num_roles] または [num_colors, num_roles]
        """
        x = self.feature_proj(color_features)
        role_logits = self.role_mlp(x)
        return role_logits

    def classify_color_roles(
        self,
        color_features: torch.Tensor
    ) -> Dict[int, str]:
        """
        色の役割を分類

        Args:
            color_features: 色特徴量 [num_colors, feature_dim]

        Returns:
            Dict[int, str]: {color_id: role_name, ...}
        """
        self.eval()
        with torch.no_grad():
            role_logits = self.forward(color_features)  # [num_colors, num_roles]
            role_probs = F.softmax(role_logits, dim=-1)  # [num_colors, num_roles]

        roles = {}
        for i in range(color_features.size(0)):
            probs = role_probs[i].cpu().numpy()
            role_id = np.argmax(probs)
            roles[i] = self.role_names[role_id]

        return roles


class ColorFeatureExtractor:
    """色特徴量抽出器"""

    def __init__(self, max_grid_size: int = 30):
        """
        初期化

        Args:
            max_grid_size: 最大グリッドサイズ
        """
        self.max_grid_size = max_grid_size

    def extract_color_features(
        self,
        grid: np.ndarray,
        objects: Optional[List[Object]] = None
    ) -> Dict[int, np.ndarray]:
        """
        色特徴量を抽出

        Args:
            grid: グリッド [H, W]
            objects: オブジェクトリスト（オプション）

        Returns:
            Dict[int, np.ndarray]: {color_id: feature_vector, ...}
        """
        height, width = grid.shape
        unique_colors = np.unique(grid)
        color_features = {}

        for color in unique_colors:
            color = int(color)
            features = self._extract_single_color_features(
                grid, color, objects, height, width
            )
            color_features[color] = features

        return color_features

    def _extract_single_color_features(
        self,
        grid: np.ndarray,
        color: int,
        objects: Optional[List[Object]],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        単一の色の特徴量を抽出

        Args:
            grid: グリッド [H, W]
            color: 色ID
            objects: オブジェクトリスト（オプション）
            height: グリッド高さ
            width: グリッド幅

        Returns:
            np.ndarray: 特徴量ベクトル [feature_dim]
        """
        features = []

        # 1. 頻度特徴量
        total_pixels = height * width
        color_pixels = np.sum(grid == color)
        frequency = color_pixels / total_pixels
        features.append(frequency)

        # 2. エッジ特徴量
        edge_ratio = self._compute_edge_ratio(grid, color, height, width)
        features.append(edge_ratio)

        # 3. 位置特徴量
        center_x, center_y = self._compute_color_center(grid, color, height, width)
        normalized_center_x = center_x / width
        normalized_center_y = center_y / height
        features.extend([normalized_center_x, normalized_center_y])

        # 4. 分散特徴量
        variance_x, variance_y = self._compute_color_variance(grid, color, height, width)
        normalized_variance_x = variance_x / (width ** 2)
        normalized_variance_y = variance_y / (height ** 2)
        features.extend([normalized_variance_x, normalized_variance_y])

        # 5. 連結成分特徴量
        if objects:
            connected_components = self._count_connected_components_for_color(objects, color)
            features.append(connected_components / max(1, len(objects)))
        else:
            features.append(0.0)

        # 6. サイズ特徴量
        if objects:
            avg_size = self._compute_avg_object_size_for_color(objects, color)
            features.append(avg_size / total_pixels)
        else:
            features.append(0.0)

        # 7. 形状特徴量
        if objects:
            compactness = self._compute_avg_compactness_for_color(objects, color)
            features.append(compactness)
        else:
            features.append(0.0)

        # 8. 周囲特徴量（周囲の色の多様性）
        neighbor_diversity = self._compute_neighbor_color_diversity(grid, color, height, width)
        features.append(neighbor_diversity)

        # 9. 対称性特徴量
        symmetry_score = self._compute_symmetry_score(grid, color, height, width)
        features.append(symmetry_score)

        # 10. グリッドサイズ正規化特徴量
        normalized_height = height / self.max_grid_size
        normalized_width = width / self.max_grid_size
        features.extend([normalized_height, normalized_width])

        # 特徴量を正規化（0-1の範囲に）
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, 0.0, 1.0)

        return features

    def _compute_edge_ratio(
        self,
        grid: np.ndarray,
        color: int,
        height: int,
        width: int
    ) -> float:
        """エッジに占める色の割合を計算"""
        if height == 0 or width == 0:
            return 0.0

        edge_pixels = []
        # 上端と下端
        edge_pixels.extend(grid[0, :].tolist())
        if height > 1:
            edge_pixels.extend(grid[height - 1, :].tolist())
        # 左端と右端
        edge_pixels.extend(grid[:, 0].tolist())
        if width > 1:
            edge_pixels.extend(grid[:, width - 1].tolist())

        if not edge_pixels:
            return 0.0

        edge_color_count = sum(1 for c in edge_pixels if c == color)
        return edge_color_count / len(edge_pixels)

    def _compute_color_center(
        self,
        grid: np.ndarray,
        color: int,
        height: int,
        width: int
    ) -> Tuple[float, float]:
        """色の中心座標を計算"""
        color_positions = np.argwhere(grid == color)
        if len(color_positions) == 0:
            return width / 2.0, height / 2.0

        center_y = np.mean(color_positions[:, 0])
        center_x = np.mean(color_positions[:, 1])
        return float(center_x), float(center_y)

    def _compute_color_variance(
        self,
        grid: np.ndarray,
        color: int,
        height: int,
        width: int
    ) -> Tuple[float, float]:
        """色の分散を計算"""
        color_positions = np.argwhere(grid == color)
        if len(color_positions) == 0:
            return 0.0, 0.0

        variance_y = np.var(color_positions[:, 0])
        variance_x = np.var(color_positions[:, 1])
        return float(variance_x), float(variance_y)

    def _count_connected_components_for_color(
        self,
        objects: List[Object],
        color: int
    ) -> int:
        """色に対応する連結成分数をカウント"""
        count = 0
        for obj in objects:
            if obj.dominant_color == color:
                count += 1
        return count

    def _compute_avg_object_size_for_color(
        self,
        objects: List[Object],
        color: int
    ) -> float:
        """色に対応するオブジェクトの平均サイズを計算"""
        sizes = []
        for obj in objects:
            if obj.dominant_color == color:
                sizes.append(obj.pixel_count)
        return np.mean(sizes) if sizes else 0.0

    def _compute_avg_compactness_for_color(
        self,
        objects: List[Object],
        color: int
    ) -> float:
        """色に対応するオブジェクトの平均コンパクトネスを計算"""
        compactness_scores = []
        for obj in objects:
            if obj.dominant_color == color:
                # コンパクトネス = 面積 / (周囲長^2)
                if hasattr(obj, 'perimeter') and obj.perimeter > 0:
                    compactness = obj.pixel_count / (obj.perimeter ** 2)
                else:
                    # 周囲長がない場合は、bbox面積に対する実際の面積の比を使用
                    bbox_area = obj.bbox_width * obj.bbox_height
                    compactness = obj.pixel_count / (bbox_area + 1e-6)
                compactness_scores.append(compactness)
        return np.mean(compactness_scores) if compactness_scores else 0.0

    def _compute_neighbor_color_diversity(
        self,
        grid: np.ndarray,
        color: int,
        height: int,
        width: int
    ) -> float:
        """周囲の色の多様性を計算"""
        color_positions = np.argwhere(grid == color)
        if len(color_positions) == 0:
            return 0.0

        neighbor_colors = set()
        for y, x in color_positions:
            # 4近傍をチェック
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_color = grid[ny, nx]
                    if neighbor_color != color:
                        neighbor_colors.add(neighbor_color)

        # 多様性 = 異なる色の数 / 最大色数（9）
        diversity = len(neighbor_colors) / 9.0
        return diversity

    def _compute_symmetry_score(
        self,
        grid: np.ndarray,
        color: int,
        height: int,
        width: int
    ) -> float:
        """対称性スコアを計算"""
        color_mask = (grid == color).astype(float)
        if np.sum(color_mask) == 0:
            return 0.0

        # X軸対称性
        x_symmetry = 0.0
        if height > 1:
            flipped_y = np.flipud(color_mask)
            x_symmetry = np.sum(color_mask * flipped_y) / np.sum(color_mask)

        # Y軸対称性
        y_symmetry = 0.0
        if width > 1:
            flipped_x = np.fliplr(color_mask)
            y_symmetry = np.sum(color_mask * flipped_x) / np.sum(color_mask)

        # 平均対称性
        symmetry = (x_symmetry + y_symmetry) / 2.0
        return symmetry


class EnhancedBackgroundColorInferencer:
    """拡張背景色推論器（色役割分類統合）"""

    def __init__(
        self,
        color_role_classifier: Optional[ColorRoleClassifier] = None,
        use_color_role_classification: bool = True
    ):
        """
        初期化

        Args:
            color_role_classifier: 色役割分類器（Noneの場合は自動生成）
            use_color_role_classification: 色役割分類を使用するか
        """
        self.use_color_role_classification = use_color_role_classification
        self.color_feature_extractor = ColorFeatureExtractor()

        if use_color_role_classification:
            if color_role_classifier is None:
                self.color_role_classifier = ColorRoleClassifier(
                    feature_dim=20,
                    hidden_dim=64,
                    num_roles=4,
                    dropout=0.1
                )
            else:
                self.color_role_classifier = color_role_classifier
        else:
            self.color_role_classifier = None

    def infer_background_color_with_roles(
        self,
        grid: np.ndarray,
        objects: Optional[List[Object]] = None
    ) -> Dict[str, any]:
        """
        背景色と色役割を推論

        Args:
            grid: グリッド [H, W]
            objects: オブジェクトリスト（オプション）

        Returns:
            Dict: {
                'background_color': int,
                'confidence': float,
                'color_roles': Dict[int, str],
                'color_features': Dict[int, np.ndarray]
            }
        """
        # 色特徴量を抽出
        color_features = self.color_feature_extractor.extract_color_features(grid, objects)

        # 色役割を分類
        color_roles = {}
        if self.use_color_role_classification and self.color_role_classifier:
            # 特徴量をテンソルに変換
            unique_colors = sorted(color_features.keys())
            feature_matrix = torch.tensor(
                [color_features[c] for c in unique_colors],
                dtype=torch.float32
            )

            # 色役割を分類
            color_roles_dict = self.color_role_classifier.classify_color_roles(feature_matrix)
            # 色IDにマッピング
            for i, color in enumerate(unique_colors):
                color_roles[color] = color_roles_dict[i]

        # 背景色を推論（色役割を考慮）
        background_color = self._infer_background_color_with_roles(
            grid, color_roles, color_features
        )

        # 信頼度を計算
        confidence = self._compute_confidence(
            grid, background_color, color_roles, color_features
        )

        return {
            'background_color': background_color,
            'confidence': confidence,
            'color_roles': color_roles,
            'color_features': color_features
        }

    def _infer_background_color_with_roles(
        self,
        grid: np.ndarray,
        color_roles: Dict[int, str],
        color_features: Dict[int, np.ndarray]
    ) -> int:
        """色役割を考慮して背景色を推論"""
        # 色役割が'background'の色を優先
        background_candidates = [
            color for color, role in color_roles.items()
            if role == 'background'
        ]

        if background_candidates:
            # 背景色候補の中から、頻度が最も高いものを選択
            color_frequencies = {
                color: color_features[color][0]  # 最初の特徴量は頻度
                for color in background_candidates
            }
            return max(color_frequencies.items(), key=lambda x: x[1])[0]

        # 色役割分類がない場合、従来の方法を使用
        # エッジ特徴量が高い色を優先
        edge_ratios = {
            color: color_features[color][1]  # 2番目の特徴量はエッジ比率
            for color in color_features.keys()
        }
        return max(edge_ratios.items(), key=lambda x: x[1])[0]

    def _compute_confidence(
        self,
        grid: np.ndarray,
        background_color: int,
        color_roles: Dict[int, str],
        color_features: Dict[int, np.ndarray]
    ) -> float:
        """信頼度を計算"""
        if background_color not in color_features:
            return 0.5

        features = color_features[background_color]

        # 頻度が高いほど信頼度が高い
        frequency = features[0]

        # エッジ比率が高いほど信頼度が高い
        edge_ratio = features[1]

        # 色役割が'background'の場合、信頼度を高く
        role_bonus = 0.2 if color_roles.get(background_color) == 'background' else 0.0

        # 統合信頼度
        confidence = (frequency * 0.4 + edge_ratio * 0.4 + role_bonus)
        return min(1.0, confidence)
