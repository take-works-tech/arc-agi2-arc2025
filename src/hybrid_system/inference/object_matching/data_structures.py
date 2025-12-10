"""
オブジェクトマッチング用データ構造

設計書に基づいたデータ構造の定義
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


def get_object_signature(obj: 'ObjectInfo') -> Optional[Tuple[int, frozenset]]:
    """
    オブジェクトの一意な識別子を生成（ピクセル座標と色を使用）

    Args:
        obj: ObjectInfoオブジェクト

    Returns:
        シグネチャ（色、ピクセルセット）のタプル、またはNone

    注意:
        - ピクセルの順序は無視されます（frozensetを使用）
        - 同じピクセルと色を持つオブジェクトは同じシグネチャになります
    """
    if not hasattr(obj, 'pixels') or not hasattr(obj, 'color'):
        return None
    pixels_set = frozenset(obj.pixels) if obj.pixels else frozenset()
    return (obj.color, pixels_set)


@dataclass
class ObjectInfo:
    """オブジェクト情報"""
    # 基本情報
    pixels: List[Tuple[int, int]]  # ピクセル座標
    color: int  # 色
    size: int  # ピクセル数

    # 形状情報
    bbox: Tuple[int, int, int, int]  # (min_i, min_j, max_i, max_j)
    center: Tuple[float, float]  # (center_y, center_x)
    width: int  # 幅
    height: int  # 高さ
    aspect_ratio: float  # アスペクト比

    # グリッド情報
    grid_index: int  # どの入力グリッドから抽出されたか
    is_train: bool  # 訓練ペアかどうか
    index: int = 0  # オブジェクトのインデックス（対応関係検出で使用）
    hole_count: Optional[int] = None  # 穴の数（COUNT_HOLES）

    # 対応関係の種類（訓練ペアのみ）
    correspondence_type: Optional[str] = None  # 'one_to_one', 'one_to_many', 'many_to_one', 'one_to_zero', 'zero_to_one'

    # 変換パターン（訓練ペアのみ、対応関係の種類とは独立）
    transformation_pattern: Optional[Dict[str, Any]] = None  # {'color_change': bool, 'position_change': bool, 'shape_change': bool, 'disappearance': bool}

    # 対応関係（訓練ペアのみ）
    matched_output_object: Optional['ObjectInfo'] = None  # 1対1対応の場合
    matched_output_objects: Optional[List['ObjectInfo']] = None  # 分割の場合（複数対応）

    # ベース特徴量ベクトル（色、形状、位置のみ、重み付けなし）
    # 57次元: 色(10) + 形状(9) + 位置(2) + パッチハッシュ3×3(8) + パッチハッシュ2×2(4) +
    #         ダウンサンプリング4×4(16) + 境界方向ヒストグラム(4) + スケルトン化特徴(2) + 形状中心位置(2)
    base_feature_vector: Optional[np.ndarray] = None

    # キャッシュフィールド（計算コスト削減のため）
    _perimeter_cache: Optional[int] = None  # 周囲長のキャッシュ
    _hu_moment_cache: Optional[float] = None  # Huモーメントのキャッシュ
    _symmetry_score_cache: Optional[Dict[str, float]] = None  # 対称性スコアのキャッシュ {'X': float, 'Y': float}
    _pixels_set_cache: Optional[set] = None  # ピクセル集合のキャッシュ

    @classmethod
    def from_object(cls, obj: Any, grid_index: int, is_train: bool, index: int = 0) -> 'ObjectInfo':
        """既存のObjectクラスからObjectInfoを作成"""
        # Objectクラスの属性を取得
        pixels = obj.pixels if hasattr(obj, 'pixels') else []

        # dominant_colorはプロパティとして提供されている可能性がある
        try:
            color = obj.dominant_color if hasattr(obj, 'dominant_color') else 0
        except:
            color = 0

        # pixel_countはプロパティとして提供されている可能性がある
        try:
            size = obj.pixel_count if hasattr(obj, 'pixel_count') else len(pixels)
        except:
            size = len(pixels)

        # bboxは(min_i, min_j, max_i, max_j)形式に変換
        # Objectクラスのbboxは(x1, y1, x2, y2)形式の可能性がある
        if hasattr(obj, 'bbox') and obj.bbox:
            if len(obj.bbox) == 4:
                # (x1, y1, x2, y2)形式を(min_i, min_j, max_i, max_j)形式に変換
                x1, y1, x2, y2 = obj.bbox
                bbox = (y1, x1, y2, x2)  # (min_i, min_j, max_i, max_j)
            else:
                bbox = (0, 0, 0, 0)
        else:
            bbox = (0, 0, 0, 0)

        # centerは(center_y, center_x)形式
        if hasattr(obj, 'center_position') and obj.center_position:
            center_x, center_y = obj.center_position
            center = (float(center_y), float(center_x))
        elif hasattr(obj, 'center_y') and hasattr(obj, 'center_x'):
            center = (float(obj.center_y), float(obj.center_x))
        else:
            # bboxから計算
            min_i, min_j, max_i, max_j = bbox
            center = ((min_i + max_i) / 2.0, (min_j + max_j) / 2.0)

        # width, height
        min_i, min_j, max_i, max_j = bbox
        width = max_j - min_j + 1
        height = max_i - min_i + 1

        # aspect_ratio
        aspect_ratio = width / height if height > 0 else 1.0

        # hole_count
        try:
            hole_count = obj.hole_count if hasattr(obj, 'hole_count') else None
        except:
            hole_count = None

        return cls(
            pixels=pixels,
            color=color,
            size=size,
            bbox=bbox,
            center=center,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            hole_count=hole_count,
            grid_index=grid_index,
            is_train=is_train,
            index=index
        )


@dataclass
class CategoryInfo:
    """カテゴリ情報"""
    category_id: int
    objects: List[ObjectInfo]

    # 代表特徴
    representative_color: Optional[List[int]] = None  # カテゴリ内のすべてのオブジェクトの色の種類（リスト）
    shape_info: Optional[Dict[str, Any]] = None  # 形状情報（最大最小値）
    position_info: Optional[Dict[str, Any]] = None  # 位置情報（最大最小値）
    representative_correspondence_type: Optional[str] = None  # 代表対応関係の種類

    # 統計情報
    object_count_per_grid: List[int] = field(default_factory=list)  # 各入力グリッドでのオブジェクト数
    total_objects: int = 0  # 総オブジェクト数

    # 対応関係の種類の統計
    disappearance_ratio: float = 0.0  # 消失オブジェクトの割合（correspondence_type == 'one_to_zero'）
    appearance_ratio: float = 0.0  # 新規出現オブジェクトの割合（将来的に使用）
    split_ratio: float = 0.0  # 分割オブジェクトの割合（correspondence_type == 'one_to_many'）
    merge_ratio: float = 0.0  # 統合オブジェクトの割合（correspondence_type == 'many_to_one'）


@dataclass
class BackgroundColorInfo:
    """背景色情報"""
    grid_index: int
    inferred_color: int
    confidence: float
    method: str  # 'frequency', 'edge', 'enhanced_with_color_roles', etc.

    # 統計情報
    color_frequency: Dict[int, int] = field(default_factory=dict)  # 色の頻度
    edge_colors: List[int] = field(default_factory=list)  # エッジ色

    # 拡張情報（色役割分類統合時）
    color_roles: Dict[int, str] = field(default_factory=dict)  # 色役割 {color_id: role_name}
    color_features: Dict[int, np.ndarray] = field(default_factory=dict)  # 色特徴量 {color_id: feature_vector}
