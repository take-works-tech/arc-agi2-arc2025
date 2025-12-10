"""
類似度計算モジュール

オブジェクト間の類似度を計算（通常版と回転・反転考慮版）
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from functools import lru_cache

from .data_structures import ObjectInfo
from .similarity_utils import calculate_position_similarity, normalize_pixels, calculate_shape_similarity, get_pixels_set


class SimilarityCalculator:
    """類似度計算器"""

    def __init__(self, config: Any):
        self.config = config
        # パフォーマンス最適化: 類似度計算のキャッシュ
        self._similarity_cache: Dict[Tuple[int, int], Tuple[float, Optional[Dict[str, Any]]]] = {}

    def calculate_object_similarity(
        self, obj1: ObjectInfo, obj2: ObjectInfo, use_rotation_flip: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        オブジェクト間の類似度を計算

        Args:
            obj1: 入力オブジェクト
            obj2: 出力オブジェクト
            use_rotation_flip: 回転・反転を考慮するか

        Returns:
            (similarity, transformation_info):
            - similarity: 類似度
            - transformation_info: 変換情報（回転角度、反転タイプなど）
        """
        # パフォーマンス最適化: キャッシュをチェック
        cache_key = (id(obj1), id(obj2), use_rotation_flip)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        similarities = []
        transformation_info = None

        # 色の類似度
        if self.config.enable_color_matching:
            color_similarity = 1.0 if obj1.color == obj2.color else 0.0
            similarities.append(color_similarity)

        # 形状の類似度
        if self.config.enable_shape_matching:
            if use_rotation_flip and self.config.enable_rotation_flip_matching:
                shape_similarity, rotation, flip = self._calculate_shape_similarity_with_rotation_flip(obj1, obj2)
                transformation_info = {
                    'rotation': rotation,
                    'flip': flip
                }
            else:
                shape_similarity = calculate_shape_similarity(obj1, obj2)
            similarities.append(shape_similarity)

        # 位置の類似度（回転・反転を考慮しない）
        if self.config.enable_position_matching:
            position_similarity = calculate_position_similarity(obj1, obj2)
            similarities.append(position_similarity)

        # サイズの類似度
        size_similarity = self._calculate_size_similarity(obj1, obj2)
        similarities.append(size_similarity)

        # 重み付き平均
        weights = [0.3, 0.3, 0.2, 0.2]  # 色、形状、位置、サイズの重み
        # 有効な類似度のみを使用
        valid_similarities = [s for s in similarities if s is not None]
        valid_weights = weights[:len(valid_similarities)]
        if sum(valid_weights) > 0:
            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            weighted_similarity = sum(s * w for s, w in zip(valid_similarities, normalized_weights))
        else:
            weighted_similarity = 0.0

        result = (weighted_similarity, transformation_info)

        # パフォーマンス最適化: キャッシュに保存（キャッシュサイズを制限）
        if len(self._similarity_cache) < 1000:  # 最大1000エントリまでキャッシュ
            self._similarity_cache[cache_key] = result

        return result

    def _calculate_shape_similarity_with_rotation_flip(
        self, obj1: ObjectInfo, obj2: ObjectInfo
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """
        回転・反転を考慮した形状類似度を計算

        Returns:
            (similarity, best_rotation, best_flip):
            - similarity: 最高類似度
            - best_rotation: 最良の回転角度（0, 90, 180, 270）
            - best_flip: 最良の反転タイプ（None, "X", "Y"）
        """
        pixels1 = get_pixels_set(obj1)
        pixels2 = get_pixels_set(obj2)

        if not pixels1 or not pixels2:
            return 0.0, None, None

        # 重心基準に正規化（共通実装を使用）
        pixels1_norm = normalize_pixels(pixels1)

        best_similarity = 0.0
        best_rotation = None
        best_flip = None

        # すべての回転・反転パターンを試す
        rotations = [0, 90, 180, 270]
        flips = [None, "X", "Y"]

        for rotation in rotations:
            for flip in flips:
                # 回転・反転を適用
                pixels2_transformed = self._transform_pixels(pixels2, rotation, flip)
                pixels2_norm = normalize_pixels(pixels2_transformed)

                # Jaccard係数を計算
                intersection = len(pixels1_norm & pixels2_norm)
                union = len(pixels1_norm | pixels2_norm)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_rotation = rotation
                    best_flip = flip

        return best_similarity, best_rotation, best_flip

    def _transform_pixels(
        self, pixels: set, rotation: int, flip: Optional[str]
    ) -> set:
        """
        ピクセル座標を回転・反転変換

        Args:
            pixels: ピクセル座標のセット
            rotation: 回転角度（0, 90, 180, 270）
            flip: 反転タイプ（None, "X", "Y")

        Returns:
            変換後のピクセル座標のセット
        """
        transformed = set()

        for y, x in pixels:
            # 回転
            if rotation == 90:
                new_y, new_x = x, -y
            elif rotation == 180:
                new_y, new_x = -y, -x
            elif rotation == 270:
                new_y, new_x = -x, y
            else:  # rotation == 0
                new_y, new_x = y, x

            # 反転
            if flip == "X":
                new_y = -new_y
            elif flip == "Y":
                new_x = -new_x

            transformed.add((new_y, new_x))

        return transformed


    def _calculate_size_similarity(self, obj1: ObjectInfo, obj2: ObjectInfo) -> float:
        """サイズの類似度を計算"""
        if obj1.size == 0 and obj2.size == 0:
            return 1.0

        if obj1.size == 0 or obj2.size == 0:
            return 0.0

        size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        return size_ratio
