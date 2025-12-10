"""
Abstract Object Patterns モジュール

色無視/相対座標/サイズだけ抽象化によるパターン理解の堅牢性向上
"""

from typing import List, Optional, Dict, Tuple
import torch
import numpy as np

from src.data_systems.data_models.core.object import Object


class AbstractObjectPattern:
    """抽象オブジェクトパターン"""

    COLOR_AGNOSTIC = "color_agnostic"  # 色無視
    RELATIVE_COORDINATES = "relative_coordinates"  # 相対座標
    SIZE_ONLY = "size_only"  # サイズのみ
    FULL = "full"  # 完全な特徴量（抽象化なし）


class AbstractObjectPatternExtractor:
    """抽象オブジェクトパターン抽出器"""

    def __init__(
        self,
        enable_color_agnostic: bool = True,
        enable_relative_coordinates: bool = True,
        enable_size_only: bool = True
    ):
        """
        初期化

        Args:
            enable_color_agnostic: 色無視パターンを有効化
            enable_relative_coordinates: 相対座標パターンを有効化
            enable_size_only: サイズのみパターンを有効化
        """
        self.enable_color_agnostic = enable_color_agnostic
        self.enable_relative_coordinates = enable_relative_coordinates
        self.enable_size_only = enable_size_only

    def extract_patterns(
        self,
        objects: List[Object],
        pattern_type: str = AbstractObjectPattern.FULL,
        grid_width: int = 30,
        grid_height: int = 30,
        reference_object: Optional[Object] = None
    ) -> List[Dict[str, float]]:
        """
        オブジェクトから抽象パターンを抽出

        Args:
            objects: オブジェクトリスト
            pattern_type: パターンタイプ（COLOR_AGNOSTIC, RELATIVE_COORDINATES, SIZE_ONLY, FULL）
            grid_width: グリッド幅（正規化用）
            grid_height: グリッド高さ（正規化用）
            reference_object: 相対座標の基準となるオブジェクト（Noneの場合は最初のオブジェクト）

        Returns:
            抽象化された特徴量のリスト
        """
        if not objects:
            return []

        # 基準オブジェクトを決定
        if reference_object is None and pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
            reference_object = objects[0] if objects else None

        patterns = []
        for obj in objects:
            pattern = self._extract_single_pattern(
                obj, pattern_type, grid_width, grid_height, reference_object
            )
            patterns.append(pattern)

        return patterns

    def _extract_single_pattern(
        self,
        obj: Object,
        pattern_type: str,
        grid_width: int,
        grid_height: int,
        reference_object: Optional[Object] = None
    ) -> Dict[str, float]:
        """
        単一オブジェクトから抽象パターンを抽出

        Args:
            obj: オブジェクト
            pattern_type: パターンタイプ
            grid_width: グリッド幅
            grid_height: グリッド高さ
            reference_object: 相対座標の基準となるオブジェクト

        Returns:
            抽象化された特徴量辞書
        """
        # 基本特徴量を取得
        color = obj.dominant_color if hasattr(obj, 'dominant_color') else obj.color if hasattr(obj, 'color') else 0
        bbox = obj.bbox if hasattr(obj, 'bbox') else (0, 0, 0, 0)
        x1, y1, x2, y2 = bbox
        width = obj.bbox_width if hasattr(obj, 'bbox_width') else (x2 - x1 + 1)
        height = obj.bbox_height if hasattr(obj, 'bbox_height') else (y2 - y1 + 1)
        area = obj.area if hasattr(obj, 'area') else (width * height)
        center_x = obj.center_x if hasattr(obj, 'center_x') else ((x1 + x2) / 2.0)
        center_y = obj.center_y if hasattr(obj, 'center_y') else ((y1 + y2) / 2.0)
        density = obj.density if hasattr(obj, 'density') else 0.0
        perimeter = obj.perimeter if hasattr(obj, 'perimeter') else 0

        pattern = {}

        if pattern_type == AbstractObjectPattern.COLOR_AGNOSTIC:
            # 色無視パターン: 色情報を除外
            pattern = {
                'bbox_left': x1 / max(grid_width, 1),
                'bbox_top': y1 / max(grid_height, 1),
                'bbox_right': x2 / max(grid_width, 1),
                'bbox_bottom': y2 / max(grid_height, 1),
                'width': width / max(grid_width, 1),
                'height': height / max(grid_height, 1),
                'area': area / max(grid_width * grid_height, 1),
                'center_x': center_x / max(grid_width, 1),
                'center_y': center_y / max(grid_height, 1),
                'density': density,
                'perimeter': perimeter / max(grid_width + grid_height, 1)
            }
        elif pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
            # 相対座標パターン: 基準オブジェクトからの相対位置
            if reference_object:
                ref_center_x = reference_object.center_x if hasattr(reference_object, 'center_x') else 0
                ref_center_y = reference_object.center_y if hasattr(reference_object, 'center_y') else 0

                # 相対座標を計算
                rel_center_x = (center_x - ref_center_x) / max(grid_width, 1)
                rel_center_y = (center_y - ref_center_y) / max(grid_height, 1)

                pattern = {
                    'color': color / 9.0,  # 正規化
                    'relative_center_x': rel_center_x,
                    'relative_center_y': rel_center_y,
                    'width': width / max(grid_width, 1),
                    'height': height / max(grid_height, 1),
                    'area': area / max(grid_width * grid_height, 1),
                    'density': density,
                    'perimeter': perimeter / max(grid_width + grid_height, 1)
                }
            else:
                # 基準オブジェクトがない場合は通常の特徴量
                pattern = {
                    'color': color / 9.0,
                    'bbox_left': x1 / max(grid_width, 1),
                    'bbox_top': y1 / max(grid_height, 1),
                    'bbox_right': x2 / max(grid_width, 1),
                    'bbox_bottom': y2 / max(grid_height, 1),
                    'width': width / max(grid_width, 1),
                    'height': height / max(grid_height, 1),
                    'area': area / max(grid_width * grid_height, 1),
                    'center_x': center_x / max(grid_width, 1),
                    'center_y': center_y / max(grid_height, 1),
                    'density': density,
                    'perimeter': perimeter / max(grid_width + grid_height, 1)
                }
        elif pattern_type == AbstractObjectPattern.SIZE_ONLY:
            # サイズのみパターン: サイズ情報のみを使用
            pattern = {
                'width': width / max(grid_width, 1),
                'height': height / max(grid_height, 1),
                'area': area / max(grid_width * grid_height, 1),
                'aspect_ratio': width / max(height, 1),
                'density': density,
                'perimeter': perimeter / max(grid_width + grid_height, 1)
            }
        else:
            # FULL: 完全な特徴量（抽象化なし）
            pattern = {
                'color': color / 9.0,
                'bbox_left': x1 / max(grid_width, 1),
                'bbox_top': y1 / max(grid_height, 1),
                'bbox_right': x2 / max(grid_width, 1),
                'bbox_bottom': y2 / max(grid_height, 1),
                'width': width / max(grid_width, 1),
                'height': height / max(grid_height, 1),
                'area': area / max(grid_width * grid_height, 1),
                'center_x': center_x / max(grid_width, 1),
                'center_y': center_y / max(grid_height, 1),
                'density': density,
                'perimeter': perimeter / max(grid_width + grid_height, 1)
            }

        return pattern

    def extract_multiple_patterns(
        self,
        objects: List[Object],
        grid_width: int = 30,
        grid_height: int = 30
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        複数の抽象パターンを同時に抽出

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            パターンタイプをキーとする辞書
        """
        patterns = {
            AbstractObjectPattern.FULL: self.extract_patterns(
                objects, AbstractObjectPattern.FULL, grid_width, grid_height
            )
        }

        if self.enable_color_agnostic:
            patterns[AbstractObjectPattern.COLOR_AGNOSTIC] = self.extract_patterns(
                objects, AbstractObjectPattern.COLOR_AGNOSTIC, grid_width, grid_height
            )

        if self.enable_relative_coordinates:
            patterns[AbstractObjectPattern.RELATIVE_COORDINATES] = self.extract_patterns(
                objects, AbstractObjectPattern.RELATIVE_COORDINATES, grid_width, grid_height
            )

        if self.enable_size_only:
            patterns[AbstractObjectPattern.SIZE_ONLY] = self.extract_patterns(
                objects, AbstractObjectPattern.SIZE_ONLY, grid_width, grid_height
            )

        return patterns


class AbstractPatternEncoder:
    """抽象パターンエンコーダー（ニューラルネットワーク用）"""

    def __init__(
        self,
        pattern_type: str = AbstractObjectPattern.FULL,
        feature_dim: int = 256,
        hidden_dim: int = 128
    ):
        """
        初期化

        Args:
            pattern_type: パターンタイプ
            feature_dim: 出力特徴量の次元
            hidden_dim: 隠れ層の次元
        """
        import torch.nn as nn

        self.pattern_type = pattern_type
        self.extractor = AbstractObjectPatternExtractor()

        # パターンタイプに応じた入力次元を決定
        if pattern_type == AbstractObjectPattern.COLOR_AGNOSTIC:
            input_dim = 11  # bbox(4) + size(2) + area(1) + center(2) + density(1) + perimeter(1)
        elif pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
            input_dim = 8  # color(1) + relative_center(2) + size(2) + area(1) + density(1) + perimeter(1)
        elif pattern_type == AbstractObjectPattern.SIZE_ONLY:
            input_dim = 6  # size(2) + area(1) + aspect_ratio(1) + density(1) + perimeter(1)
        else:  # FULL
            input_dim = 12  # color(1) + bbox(4) + size(2) + area(1) + center(2) + density(1) + perimeter(1)

        # エンコーダーネットワーク
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def encode(
        self,
        objects: List[Object],
        grid_width: int = 30,
        grid_height: int = 30
    ) -> torch.Tensor:
        """
        オブジェクトリストを抽象パターンにエンコード

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            エンコードされた特徴量テンソル [num_objects, feature_dim]
        """
        # 抽象パターンを抽出
        patterns = self.extractor.extract_patterns(
            objects, self.pattern_type, grid_width, grid_height
        )

        if not patterns:
            return torch.zeros(1, self.encoder[-1].out_features)

        # パターンをテンソルに変換
        pattern_values = []
        for pattern in patterns:
            # パターンタイプに応じた順序で値を取得
            if self.pattern_type == AbstractObjectPattern.COLOR_AGNOSTIC:
                values = [
                    pattern['bbox_left'], pattern['bbox_top'],
                    pattern['bbox_right'], pattern['bbox_bottom'],
                    pattern['width'], pattern['height'],
                    pattern['area'], pattern['center_x'], pattern['center_y'],
                    pattern['density'], pattern['perimeter']
                ]
            elif self.pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
                values = [
                    pattern['color'],
                    pattern['relative_center_x'], pattern['relative_center_y'],
                    pattern['width'], pattern['height'],
                    pattern['area'], pattern['density'], pattern['perimeter']
                ]
            elif self.pattern_type == AbstractObjectPattern.SIZE_ONLY:
                values = [
                    pattern['width'], pattern['height'],
                    pattern['area'], pattern['aspect_ratio'],
                    pattern['density'], pattern['perimeter']
                ]
            else:  # FULL
                values = [
                    pattern['color'],
                    pattern['bbox_left'], pattern['bbox_top'],
                    pattern['bbox_right'], pattern['bbox_bottom'],
                    pattern['width'], pattern['height'],
                    pattern['area'], pattern['center_x'], pattern['center_y'],
                    pattern['density'], pattern['perimeter']
                ]
            pattern_values.append(values)

        # テンソルに変換
        pattern_tensor = torch.tensor(pattern_values, dtype=torch.float32)

        # エンコード
        encoded = self.encoder(pattern_tensor)

        return encoded
