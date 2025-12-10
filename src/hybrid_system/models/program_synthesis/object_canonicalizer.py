"""
オブジェクト正規化モジュール

色のランダムリマップ、位置の正規化、サイズの正規化、形状の正規化を行う
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import cv2
from dataclasses import dataclass

from src.data_systems.data_models.core.object import Object


@dataclass
class CanonicalizedObject:
    """正規化されたオブジェクト"""
    original_object: Object
    remapped_color: int
    normalized_position: Tuple[float, float]  # (x, y) 正規化座標 [0, 1]
    normalized_size: Tuple[float, float]  # (width, height) 正規化サイズ [0, 1]
    shape_embedding: np.ndarray  # [5-10次元] 形状埋め込み


class ObjectCanonicalizer:
    """オブジェクト正規化クラス"""

    def __init__(
        self,
        enable_color_remap: bool = True,
        enable_position_normalize: bool = True,
        enable_size_normalize: bool = True,
        enable_shape_embedding: bool = True,
        shape_embedding_dim: int = 8
    ):
        """
        初期化

        Args:
            enable_color_remap: 色のランダムリマップを有効化
            enable_position_normalize: 位置の正規化を有効化
            enable_size_normalize: サイズの正規化を有効化
            enable_shape_embedding: 形状埋め込みを有効化
            shape_embedding_dim: 形状埋め込みの次元（5-10）
        """
        self.enable_color_remap = enable_color_remap
        self.enable_position_normalize = enable_position_normalize
        self.enable_size_normalize = enable_size_normalize
        self.enable_shape_embedding = enable_shape_embedding
        self.shape_embedding_dim = shape_embedding_dim

        # 色リマップマップ（実行時に生成）
        self.color_remap_map: Optional[Dict[int, int]] = None

    def canonicalize(
        self,
        objects: List[Object],
        grid_width: int,
        grid_height: int,
        color_remap_map: Optional[Dict[int, int]] = None
    ) -> List[CanonicalizedObject]:
        """
        オブジェクトリストを正規化

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅
            grid_height: グリッド高さ
            color_remap_map: 色リマップマップ（指定しない場合は自動生成）

        Returns:
            List[CanonicalizedObject]: 正規化されたオブジェクトリスト
        """
        if not objects:
            return []

        # 色リマップマップを生成または使用
        if self.enable_color_remap:
            if color_remap_map is None:
                self.color_remap_map = self._generate_color_remap_map(objects)
            else:
                self.color_remap_map = color_remap_map
        else:
            self.color_remap_map = {i: i for i in range(10)}

        canonicalized = []
        for obj in objects:
            # 色のリマップ
            remapped_color = self.color_remap_map.get(obj.dominant_color, obj.dominant_color)

            # 位置の正規化
            if self.enable_position_normalize:
                # 原点を左上に移動（相対座標に変換）
                normalized_x = obj.center_x / grid_width
                normalized_y = obj.center_y / grid_height
                normalized_position = (normalized_x, normalized_y)
            else:
                normalized_position = (obj.center_x, obj.center_y)

            # サイズの正規化
            if self.enable_size_normalize:
                normalized_width = obj.bbox_width / grid_width
                normalized_height = obj.bbox_height / grid_height
                normalized_size = (normalized_width, normalized_height)
            else:
                normalized_size = (obj.bbox_width, obj.bbox_height)

            # 形状埋め込み
            if self.enable_shape_embedding:
                shape_embedding = self._compute_shape_embedding(obj, grid_width, grid_height)
            else:
                shape_embedding = np.zeros(self.shape_embedding_dim)

            canonicalized.append(CanonicalizedObject(
                original_object=obj,
                remapped_color=remapped_color,
                normalized_position=normalized_position,
                normalized_size=normalized_size,
                shape_embedding=shape_embedding
            ))

        return canonicalized

    def _generate_color_remap_map(self, objects: List[Object]) -> Dict[int, int]:
        """
        色リマップマップを生成

        使用されている色をランダムにリマップする

        Args:
            objects: オブジェクトリスト

        Returns:
            Dict[int, int]: 色リマップマップ {original_color: remapped_color}
        """
        # 使用されている色を収集
        used_colors = set()
        for obj in objects:
            used_colors.add(obj.dominant_color)

        # 使用されている色をランダムにリマップ
        used_colors_list = sorted(list(used_colors))
        remapped_colors = used_colors_list.copy()
        np.random.shuffle(remapped_colors)

        color_remap_map = {}
        for original_color in range(10):
            if original_color in used_colors:
                # 使用されている色はランダムにリマップ
                idx = used_colors_list.index(original_color)
                color_remap_map[original_color] = remapped_colors[idx]
            else:
                # 使用されていない色はそのまま
                color_remap_map[original_color] = original_color

        return color_remap_map

    def _compute_shape_embedding(
        self,
        obj: Object,
        grid_width: int,
        grid_height: int
    ) -> np.ndarray:
        """
        形状埋め込みを計算

        Huモーメント、周囲長/面積比、アスペクト比などを使用

        Args:
            obj: オブジェクト
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            np.ndarray: [shape_embedding_dim] 形状埋め込み
        """
        features = []

        # 1. アスペクト比
        if obj.bbox_height > 0:
            aspect_ratio = obj.bbox_width / obj.bbox_height
        else:
            aspect_ratio = 1.0
        features.append(aspect_ratio)

        # 2. 面積比（bbox面積に対する実際の面積）
        bbox_area = obj.bbox_width * obj.bbox_height
        if bbox_area > 0:
            area_ratio = obj.pixel_count / bbox_area
        else:
            area_ratio = 0.0
        features.append(area_ratio)

        # 3. 周囲長/面積比
        if hasattr(obj, 'perimeter') and obj.perimeter > 0:
            perimeter_area_ratio = obj.perimeter / (obj.pixel_count + 1e-6)
        else:
            # 周囲長を推定（bboxの周囲長）
            perimeter = 2 * (obj.bbox_width + obj.bbox_height)
            perimeter_area_ratio = perimeter / (obj.pixel_count + 1e-6)
        features.append(perimeter_area_ratio)

        # 4. Huモーメント（形状の不変特徴量）
        if obj.pixels:
            # オブジェクトのマスクを作成
            mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
            for y, x in obj.pixels:
                if 0 <= y < grid_height and 0 <= x < grid_width:
                    mask[y, x] = 1

            # Huモーメントを計算
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                # 対数変換（値が非常に小さいため）
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                features.extend(hu_moments[:4].tolist())  # 最初の4つのHuモーメント
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)

        # 5. 密度
        if hasattr(obj, 'density'):
            features.append(obj.density)
        else:
            features.append(area_ratio)

        # 6. 穴の数
        if hasattr(obj, 'hole_count'):
            features.append(obj.hole_count / 10.0)  # 正規化
        else:
            features.append(0.0)

        # 次元を調整
        if len(features) < self.shape_embedding_dim:
            features.extend([0.0] * (self.shape_embedding_dim - len(features)))
        elif len(features) > self.shape_embedding_dim:
            features = features[:self.shape_embedding_dim]

        return np.array(features, dtype=np.float32)

    def get_color_remap_map(self) -> Optional[Dict[int, int]]:
        """色リマップマップを取得"""
        return self.color_remap_map
