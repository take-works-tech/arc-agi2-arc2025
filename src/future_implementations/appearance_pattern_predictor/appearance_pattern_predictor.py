"""
新規出現パターン予測モジュール

新規出現オブジェクトの特徴を分析し、予測する機能
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter

from src.hybrid_system.inference.object_matching.data_structures import ObjectInfo, CategoryInfo


@dataclass
class AppearancePattern:
    """新規出現パターン"""
    # 出現オブジェクトの特徴
    color: int
    size: int
    width: int
    height: int
    aspect_ratio: float
    position: Tuple[int, int]  # (center_x, center_y)

    # 出現パターンの統計
    frequency: int = 1  # 出現頻度
    confidence: float = 0.0  # 予測信頼度

    # 関連情報
    related_categories: List[int] = None  # 関連するカテゴリID
    transformation_hints: List[str] = None  # 変換のヒント


class AppearancePatternPredictor:
    """新規出現パターン予測器"""

    def __init__(self, config: Optional[Any] = None):
        """
        初期化

        Args:
            config: 設定（オプション）
        """
        self.config = config
        self.appearance_patterns: List[AppearancePattern] = []

    def analyze_appearance_patterns(
        self,
        transformation_patterns: Dict[int, List[Dict[str, Any]]],
        categories: List[CategoryInfo]
    ) -> List[AppearancePattern]:
        """
        新規出現パターンを分析

        Args:
            transformation_patterns: 変換パターン情報
            categories: カテゴリ情報

        Returns:
            appearance_patterns: 新規出現パターンのリスト
        """
        appearance_patterns = []

        # 各ペアの新規出現オブジェクトを分析
        for connectivity, pair_patterns in transformation_patterns.items():
            for pair_pattern in pair_patterns:
                appeared_objects = pair_pattern.get('appeared_objects', [])

                for obj in appeared_objects:
                    # 新規出現オブジェクトの特徴を抽出
                    pattern = self._extract_appearance_pattern(obj, categories)
                    if pattern:
                        appearance_patterns.append(pattern)

        # パターンを集約（類似パターンを統合）
        aggregated_patterns = self._aggregate_patterns(appearance_patterns)

        self.appearance_patterns = aggregated_patterns
        return aggregated_patterns

    def _extract_appearance_pattern(
        self,
        obj: ObjectInfo,
        categories: List[CategoryInfo]
    ) -> Optional[AppearancePattern]:
        """
        新規出現オブジェクトからパターンを抽出

        Args:
            obj: 新規出現オブジェクト
            categories: カテゴリ情報

        Returns:
            pattern: 新規出現パターン
        """
        if not obj:
            return None

        # 基本特徴を抽出
        color = obj.color if obj.color is not None else 0
        size = obj.size if obj.size is not None else 0
        width = obj.width if obj.width is not None else 0
        height = obj.height if obj.height is not None else 0
        aspect_ratio = obj.aspect_ratio if obj.aspect_ratio is not None else 1.0

        # 位置を抽出
        if obj.bbox and len(obj.bbox) == 4:
            min_i, min_j, max_i, max_j = obj.bbox
            center_x = (min_j + max_j) / 2
            center_y = (min_i + max_i) / 2
            position = (int(center_x), int(center_y))
        else:
            position = (0, 0)

        # 関連するカテゴリを特定（類似特徴を持つカテゴリ）
        related_categories = self._find_related_categories(obj, categories)

        # 変換のヒントを生成
        transformation_hints = self._generate_transformation_hints(obj, categories)

        pattern = AppearancePattern(
            color=color,
            size=size,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            position=position,
            frequency=1,
            confidence=0.0,
            related_categories=related_categories,
            transformation_hints=transformation_hints
        )

        return pattern

    def _find_related_categories(
        self,
        obj: ObjectInfo,
        categories: List[CategoryInfo]
    ) -> List[int]:
        """
        新規出現オブジェクトに関連するカテゴリを特定

        Args:
            obj: 新規出現オブジェクト
            categories: カテゴリ情報

        Returns:
            related_category_ids: 関連するカテゴリIDのリスト
        """
        related_category_ids = []

        for category in categories:
            # カテゴリの代表特徴と比較
            similarity = self._calculate_category_similarity(obj, category)

            # 類似度が閾値以上のカテゴリを関連カテゴリとして追加
            if similarity > 0.5:
                related_category_ids.append(category.category_id)

        return related_category_ids

    def _calculate_category_similarity(
        self,
        obj: ObjectInfo,
        category: CategoryInfo
    ) -> float:
        """
        オブジェクトとカテゴリの類似度を計算

        Args:
            obj: オブジェクト
            category: カテゴリ情報

        Returns:
            similarity: 類似度（0.0-1.0）
        """
        if not category.objects:
            return 0.0

        similarities = []

        # 色の類似度
        if category.representative_color and obj.color is not None:
            if obj.color in category.representative_color:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

        # サイズの類似度
        if category.shape_info:
            min_size = category.shape_info.get('min_size')
            max_size = category.shape_info.get('max_size')
            if min_size is not None and max_size is not None and obj.size is not None:
                if min_size <= obj.size <= max_size:
                    size_sim = 1.0
                else:
                    # 範囲外の場合、距離に基づく類似度
                    if obj.size < min_size:
                        size_sim = obj.size / min_size if min_size > 0 else 0.0
                    else:
                        size_sim = max_size / obj.size if obj.size > 0 else 0.0
                    size_sim = max(0.0, min(1.0, size_sim))
                similarities.append(size_sim)

        # 位置の類似度
        if category.position_info and obj.bbox and len(obj.bbox) == 4:
            min_i, min_j, max_i, max_j = obj.bbox
            obj_center_x = (min_j + max_j) / 2
            obj_center_y = (min_i + max_i) / 2

            min_x = category.position_info.get('min_x')
            max_x = category.position_info.get('max_x')
            min_y = category.position_info.get('min_y')
            max_y = category.position_info.get('max_y')

            if min_x is not None and max_x is not None and min_y is not None and max_y is not None:
                if min_x <= obj_center_x <= max_x and min_y <= obj_center_y <= max_y:
                    similarities.append(1.0)
                else:
                    # 範囲外の場合、距離に基づく類似度
                    dist_x = min(abs(obj_center_x - min_x), abs(obj_center_x - max_x))
                    dist_y = min(abs(obj_center_y - min_y), abs(obj_center_y - max_y))
                    max_dist = max(max_x - min_x, max_y - min_y, 1)
                    pos_sim = 1.0 - min(1.0, (dist_x + dist_y) / max_dist)
                    similarities.append(pos_sim)

        # 平均類似度を返す
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _generate_transformation_hints(
        self,
        obj: ObjectInfo,
        categories: List[CategoryInfo]
    ) -> List[str]:
        """
        変換のヒントを生成

        Args:
            obj: 新規出現オブジェクト
            categories: カテゴリ情報

        Returns:
            hints: 変換のヒントのリスト
        """
        hints = []

        # 関連するカテゴリから変換パターンを推測
        for category in categories:
            if category.category_id in (obj.related_categories if hasattr(obj, 'related_categories') else []):
                # カテゴリの変換パターンを分析
                if hasattr(category, 'disappearance_ratio') and category.disappearance_ratio > 0.5:
                    hints.append("disappearance_to_appearance")

                if hasattr(category, 'split_ratio') and category.split_ratio > 0.5:
                    hints.append("split_result")

                if hasattr(category, 'merge_ratio') and category.merge_ratio > 0.5:
                    hints.append("merge_result")

        return hints

    def _aggregate_patterns(
        self,
        patterns: List[AppearancePattern]
    ) -> List[AppearancePattern]:
        """
        類似パターンを集約

        Args:
            patterns: パターンのリスト

        Returns:
            aggregated_patterns: 集約されたパターンのリスト
        """
        if not patterns:
            return []

        # 類似パターンをグループ化
        groups = []
        used = set()

        for i, pattern1 in enumerate(patterns):
            if i in used:
                continue

            group = [pattern1]
            used.add(i)

            for j, pattern2 in enumerate(patterns[i+1:], start=i+1):
                if j in used:
                    continue

                # 類似度を計算
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)

                if similarity > 0.7:
                    group.append(pattern2)
                    used.add(j)

            groups.append(group)

        # グループを集約
        aggregated_patterns = []
        for group in groups:
            if len(group) == 1:
                aggregated_patterns.append(group[0])
            else:
                # グループ内のパターンを統合
                aggregated = self._merge_patterns(group)
                aggregated_patterns.append(aggregated)

        return aggregated_patterns

    def _calculate_pattern_similarity(
        self,
        pattern1: AppearancePattern,
        pattern2: AppearancePattern
    ) -> float:
        """
        2つのパターンの類似度を計算

        Args:
            pattern1: パターン1
            pattern2: パターン2

        Returns:
            similarity: 類似度（0.0-1.0）
        """
        similarities = []

        # 色の類似度
        if pattern1.color == pattern2.color:
            similarities.append(1.0)
        else:
            similarities.append(0.0)

        # サイズの類似度
        if pattern1.size > 0 and pattern2.size > 0:
            size_sim = min(pattern1.size, pattern2.size) / max(pattern1.size, pattern2.size)
            similarities.append(size_sim)

        # アスペクト比の類似度
        if pattern1.aspect_ratio > 0 and pattern2.aspect_ratio > 0:
            aspect_sim = min(pattern1.aspect_ratio, pattern2.aspect_ratio) / max(pattern1.aspect_ratio, pattern2.aspect_ratio)
            similarities.append(aspect_sim)

        # 位置の類似度（距離ベース）
        pos1 = pattern1.position
        pos2 = pattern2.position
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        max_dist = 30.0  # 最大距離（仮定）
        pos_sim = 1.0 - min(1.0, dist / max_dist)
        similarities.append(pos_sim)

        # 平均類似度を返す
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _merge_patterns(
        self,
        patterns: List[AppearancePattern]
    ) -> AppearancePattern:
        """
        複数のパターンを統合

        Args:
            patterns: パターンのリスト

        Returns:
            merged_pattern: 統合されたパターン
        """
        if not patterns:
            return None

        if len(patterns) == 1:
            return patterns[0]

        # 最頻出の色
        colors = [p.color for p in patterns]
        color_counts = Counter(colors)
        most_common_color = color_counts.most_common(1)[0][0]

        # 平均サイズ、幅、高さ
        avg_size = int(np.mean([p.size for p in patterns]))
        avg_width = int(np.mean([p.width for p in patterns]))
        avg_height = int(np.mean([p.height for p in patterns]))
        avg_aspect_ratio = np.mean([p.aspect_ratio for p in patterns])

        # 平均位置
        avg_x = int(np.mean([p.position[0] for p in patterns]))
        avg_y = int(np.mean([p.position[1] for p in patterns]))
        avg_position = (avg_x, avg_y)

        # 関連カテゴリを統合
        all_related_categories = []
        for p in patterns:
            if p.related_categories:
                all_related_categories.extend(p.related_categories)
        unique_related_categories = list(set(all_related_categories))

        # 変換ヒントを統合
        all_hints = []
        for p in patterns:
            if p.transformation_hints:
                all_hints.extend(p.transformation_hints)
        unique_hints = list(set(all_hints))

        merged = AppearancePattern(
            color=most_common_color,
            size=avg_size,
            width=avg_width,
            height=avg_height,
            aspect_ratio=avg_aspect_ratio,
            position=avg_position,
            frequency=len(patterns),
            confidence=min(1.0, len(patterns) / 10.0),  # 頻度に基づく信頼度
            related_categories=unique_related_categories,
            transformation_hints=unique_hints
        )

        return merged

    def predict_appearance(
        self,
        input_objects: List[ObjectInfo],
        categories: List[CategoryInfo]
    ) -> List[AppearancePattern]:
        """
        新規出現を予測

        Args:
            input_objects: 入力オブジェクトのリスト
            categories: カテゴリ情報

        Returns:
            predicted_patterns: 予測された新規出現パターンのリスト
        """
        predicted_patterns = []

        # 既存のパターンから予測
        for pattern in self.appearance_patterns:
            # 入力オブジェクトとパターンの関連カテゴリを比較
            input_category_ids = set()
            for obj in input_objects:
                # オブジェクトがどのカテゴリに属するかを判定
                for category in categories:
                    if self._calculate_category_similarity(obj, category) > 0.5:
                        input_category_ids.add(category.category_id)

            # 関連カテゴリが一致する場合、新規出現を予測
            if pattern.related_categories and any(
                cat_id in input_category_ids for cat_id in pattern.related_categories
            ):
                # 信頼度を調整（入力オブジェクトとの関連性に基づく）
                adjusted_confidence = pattern.confidence * 0.8  # 予測なので信頼度を下げる
                predicted = AppearancePattern(
                    color=pattern.color,
                    size=pattern.size,
                    width=pattern.width,
                    height=pattern.height,
                    aspect_ratio=pattern.aspect_ratio,
                    position=pattern.position,
                    frequency=pattern.frequency,
                    confidence=adjusted_confidence,
                    related_categories=pattern.related_categories,
                    transformation_hints=pattern.transformation_hints
                )
                predicted_patterns.append(predicted)

        # 信頼度順にソート
        predicted_patterns.sort(key=lambda p: p.confidence, reverse=True)

        return predicted_patterns
