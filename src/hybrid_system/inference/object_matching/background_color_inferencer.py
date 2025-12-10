"""
背景色推論モジュール

タスク内のすべての入力グリッドに対して背景色を推論
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

from .data_structures import BackgroundColorInfo

# 拡張背景色推論器をインポート（オプション）
try:
    from src.hybrid_system.models.program_synthesis.color_role_classifier import (
        EnhancedBackgroundColorInferencer,
        ColorRoleClassifier
    )
    ENHANCED_BG_INFERENCE_AVAILABLE = True
except ImportError:
    ENHANCED_BG_INFERENCE_AVAILABLE = False
    EnhancedBackgroundColorInferencer = None
    ColorRoleClassifier = None


class BackgroundColorInferencer:
    """背景色推論器"""

    def __init__(
        self,
        use_enhanced_inference: bool = False,
        color_role_classifier: Optional[Any] = None
    ):
        """
        初期化

        Args:
            use_enhanced_inference: 拡張背景色推論（色役割分類統合）を使用するか
            color_role_classifier: 色役割分類器（Noneの場合は自動生成）
        """
        self.use_enhanced_inference = use_enhanced_inference and ENHANCED_BG_INFERENCE_AVAILABLE

        if self.use_enhanced_inference:
            self.enhanced_inferencer = EnhancedBackgroundColorInferencer(
                color_role_classifier=color_role_classifier,
                use_color_role_classification=True
            )
        else:
            self.enhanced_inferencer = None

    def infer_background_colors(self, task: Any) -> Dict[int, BackgroundColorInfo]:
        """
        タスク内のすべての入力グリッド（訓練+テスト）に対して背景色を推論

        Args:
            task: Taskオブジェクト

        Returns:
            {
                grid_index: BackgroundColorInfo,
                ...
            }
        """
        all_input_grids = task.get_all_inputs()
        background_colors = {}

        for grid_index, input_grid in enumerate(all_input_grids):
            input_grid_np = np.array(input_grid)

            # 拡張推論を使用する場合
            if self.use_enhanced_inference and self.enhanced_inferencer:
                # オブジェクトを抽出（色役割分類に使用）
                try:
                    from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
                    from src.data_systems.config.config import ExtractionConfig
                    from src.data_systems.data_models.base import ObjectType

                    extractor = IntegratedObjectExtractor(ExtractionConfig())
                    result = extractor.extract_objects_by_type(input_grid_np, input_image_index=grid_index)
                    objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, []) if result.success else []
                except Exception:
                    objects = []

                # 拡張推論を実行
                enhanced_result = self.enhanced_inferencer.infer_background_color_with_roles(
                    input_grid_np, objects
                )

                # BackgroundColorInfoに変換
                bg_info = BackgroundColorInfo(
                    grid_index=grid_index,
                    inferred_color=enhanced_result['background_color'],
                    confidence=enhanced_result['confidence'],
                    method='enhanced_with_color_roles',
                    color_frequency={},  # 拡張推論では詳細情報を保持
                    edge_colors=[],
                    color_roles=enhanced_result.get('color_roles', {}),  # 色役割情報を追加
                    color_features=enhanced_result.get('color_features', {})  # 色特徴量を追加
                )
            else:
                # 従来の推論を使用
                bg_info = self._infer_single_background_color(input_grid_np, grid_index)

            background_colors[grid_index] = bg_info

        return background_colors

    def _infer_single_background_color(
        self, grid: np.ndarray, grid_index: int
    ) -> BackgroundColorInfo:
        """
        単一のグリッドから背景色を推論

        Args:
            grid: グリッド（numpy配列）
            grid_index: グリッドのインデックス

        Returns:
            BackgroundColorInfo
        """
        # 色の頻度を計算
        color_counts = Counter(grid.flatten())
        color_frequency = dict(color_counts)

        # 最頻出色を基本候補とする
        most_common_color = color_counts.most_common(1)[0][0] if color_counts else 0

        # エッジ色を考慮
        edge_colors = self._get_edge_colors(grid)

        # エッジに多く出現する色を優先
        edge_color_counts = Counter(edge_colors)
        if edge_color_counts:
            most_common_edge_color = edge_color_counts.most_common(1)[0][0]
            # エッジ色と最頻出色が一致する場合、信頼度を高く
            if most_common_edge_color == most_common_color:
                inferred_color = most_common_color
                confidence = 0.9
                method = 'frequency_and_edge'
            else:
                # エッジ色を優先（背景色はエッジに多く出現する可能性が高い）
                inferred_color = most_common_edge_color
                confidence = 0.7
                method = 'edge'
        else:
            inferred_color = most_common_color
            confidence = 0.8
            method = 'frequency'

        return BackgroundColorInfo(
            grid_index=grid_index,
            inferred_color=inferred_color,
            confidence=confidence,
            method=method,
            color_frequency=color_frequency,
            edge_colors=edge_colors
        )

    def _get_edge_colors(self, grid: np.ndarray) -> List[int]:
        """グリッドのエッジ色を取得"""
        if grid.size == 0:
            return []

        height, width = grid.shape
        edge_colors = []

        # 上端と下端
        edge_colors.extend(grid[0, :].tolist())
        if height > 1:
            edge_colors.extend(grid[height - 1, :].tolist())

        # 左端と右端
        edge_colors.extend(grid[:, 0].tolist())
        if width > 1:
            edge_colors.extend(grid[:, width - 1].tolist())

        return edge_colors

    def calculate_color_consistency(self, input_grids: List[List[List[int]]]) -> float:
        """
        入力グリッドごとに使われている色の一致度を計算

        Args:
            input_grids: 入力グリッドのリスト

        Returns:
            一致度（0.0～1.0）
        """
        if not input_grids:
            return 0.0

        # 各グリッドで使われている色の集合
        color_sets = [set(np.array(grid).flatten()) for grid in input_grids]

        # すべてのグリッドで共通の色
        common_colors = set.intersection(*color_sets) if color_sets else set()

        if not color_sets:
            return 0.0

        # 各グリッドの色数の平均
        avg_color_count = sum(len(color_set) for color_set in color_sets) / len(color_sets)

        if avg_color_count == 0:
            return 0.0

        # 一致度 = 共通色の数 / 平均色数
        consistency = len(common_colors) / avg_color_count
        return consistency

    def calculate_bg_color_consistency(self, inferred_bg_colors: List[int]) -> float:
        """
        推論した背景色の一致度を計算

        Args:
            inferred_bg_colors: 各入力グリッドから推論した背景色のリスト
                               （例: [0, 0, 1, 0] → 4つの入力グリッドの背景色）

        Returns:
            一致度（0.0～1.0）
        """
        if not inferred_bg_colors:
            return 0.0

        # 最頻出背景色
        most_common_bg = Counter(inferred_bg_colors).most_common(1)[0][0]
        most_common_count = Counter(inferred_bg_colors).most_common(1)[0][1]

        # 一致度 = 最頻出背景色の出現回数 / 総入力グリッド数
        # inferred_bg_colorsの長さは総入力グリッド数に等しい
        consistency = most_common_count / len(inferred_bg_colors)
        return consistency

    def check_color_ratio_consistency(
        self, all_input_grids: List[List[List[int]]],
        color: int,
        max_variance: float = 0.4
    ) -> bool:
        """
        その色のピクセル数の割合がグリッド間で変動しすぎていないかチェック

        Args:
            all_input_grids: すべての入力グリッド
            color: チェックする色
            max_variance: 最大変動係数（標準偏差 / 平均）

        Returns:
            変動が許容範囲内ならTrue、そうでなければFalse
        """
        if not all_input_grids:
            return False

        ratios = []
        for grid in all_input_grids:
            grid_np = np.array(grid)
            total_pixels = grid_np.size
            color_pixels = np.sum(grid_np == color)
            ratio = color_pixels / total_pixels if total_pixels > 0 else 0.0
            ratios.append(ratio)

        if not ratios:
            return False

        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        # 平均が0の場合はFalse
        if mean_ratio == 0:
            return False

        # 変動係数（標準偏差 / 平均）を計算
        coefficient_of_variation = std_ratio / mean_ratio

        # 変動係数が閾値以下ならTrue
        return coefficient_of_variation <= max_variance
