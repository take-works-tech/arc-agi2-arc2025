"""
汎化能力評価器

テスト範囲包含性と変換パターン一貫性を評価
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from core.data_structures import Task, DataPair


@dataclass
class GeneralizationConfig:
    """汎化評価設定"""
    test_range_threshold: float = 0.8
    pattern_consistency_threshold: float = 0.7
    color_range_tolerance: float = 0.1
    size_range_tolerance: float = 0.2
    enable_detailed_analysis: bool = True


class GeneralizationEvaluator:
    """汎化能力評価器"""

    def __init__(self, config: Optional[GeneralizationConfig] = None):
        """初期化"""
        self.config = config or GeneralizationConfig()

        # 評価統計
        self.evaluation_stats = {
            'total_tasks_evaluated': 0,
            'tasks_passed_test_range': 0,
            'tasks_passed_pattern_consistency': 0,
            'tasks_passed_both': 0,
            'average_test_range_score': 0.0,
            'average_pattern_consistency_score': 0.0
        }

    def evaluate_task(self, task: Task) -> Dict[str, Any]:
        """タスクの汎化能力を評価

        Args:
            task: 評価するタスク

        Returns:
            評価結果
        """
        self.evaluation_stats['total_tasks_evaluated'] += 1

        # 1. テスト範囲包含性の評価
        test_range_result = self._evaluate_test_range_inclusion(task)

        # 2. 変換パターン一貫性の評価
        pattern_consistency_result = self._evaluate_pattern_consistency(task)

        # 3. 総合評価
        overall_score = (test_range_result['score'] + pattern_consistency_result['score']) / 2.0
        overall_passed = (test_range_result['passed'] and pattern_consistency_result['passed'])

        # 統計更新
        if test_range_result['passed']:
            self.evaluation_stats['tasks_passed_test_range'] += 1
        if pattern_consistency_result['passed']:
            self.evaluation_stats['tasks_passed_pattern_consistency'] += 1
        if overall_passed:
            self.evaluation_stats['tasks_passed_both'] += 1

        # 平均スコア更新
        total = self.evaluation_stats['total_tasks_evaluated']
        self.evaluation_stats['average_test_range_score'] = (
            (self.evaluation_stats['average_test_range_score'] * (total - 1) + test_range_result['score']) / total
        )
        self.evaluation_stats['average_pattern_consistency_score'] = (
            (self.evaluation_stats['average_pattern_consistency_score'] * (total - 1) + pattern_consistency_result['score']) / total
        )

        return {
            'task_id': task.task_id,
            'overall_score': overall_score,
            'overall_passed': overall_passed,
            'test_range_inclusion': test_range_result,
            'pattern_consistency': pattern_consistency_result,
            'detailed_analysis': self._perform_detailed_analysis(task) if self.config.enable_detailed_analysis else None
        }

    def _evaluate_test_range_inclusion(self, task: Task) -> Dict[str, Any]:
        """テスト範囲包含性を評価"""
        train_inputs = task.get_train_inputs()
        test_inputs = task.get_test_inputs()

        # 1. 色の範囲チェック
        color_score = self._check_color_range_inclusion(train_inputs, test_inputs)

        # 2. グリッドサイズの範囲チェック
        size_score = self._check_size_range_inclusion(train_inputs, test_inputs)

        # 3. オブジェクト数の範囲チェック
        object_count_score = self._check_object_count_range_inclusion(train_inputs, test_inputs)

        # 4. 空間配置の範囲チェック
        spatial_score = self._check_spatial_range_inclusion(train_inputs, test_inputs)

        # 総合スコア
        overall_score = (color_score + size_score + object_count_score + spatial_score) / 4.0
        passed = overall_score >= self.config.test_range_threshold

        return {
            'score': overall_score,
            'passed': passed,
            'color_score': color_score,
            'size_score': size_score,
            'object_count_score': object_count_score,
            'spatial_score': spatial_score,
            'threshold': self.config.test_range_threshold
        }

    def _evaluate_pattern_consistency(self, task: Task) -> Dict[str, Any]:
        """変換パターン一貫性を評価"""
        train_pairs = task.train
        test_pairs = task.test

        # 1. 変換パターンの抽出
        train_patterns = [self._extract_transformation_pattern(pair) for pair in train_pairs]
        test_patterns = [self._extract_transformation_pattern(pair) for pair in test_pairs]

        # 2. パターンの一貫性チェック
        consistency_score = self._calculate_pattern_consistency(train_patterns, test_patterns)

        # 3. 変換の複雑度一貫性チェック
        complexity_score = self._check_complexity_consistency(train_pairs, test_pairs)

        # 4. 変換の方向性一貫性チェック
        direction_score = self._check_direction_consistency(train_pairs, test_pairs)

        # 総合スコア
        overall_score = (consistency_score + complexity_score + direction_score) / 3.0
        passed = overall_score >= self.config.pattern_consistency_threshold

        return {
            'score': overall_score,
            'passed': passed,
            'consistency_score': consistency_score,
            'complexity_score': complexity_score,
            'direction_score': direction_score,
            'threshold': self.config.pattern_consistency_threshold,
            'train_patterns': train_patterns,
            'test_patterns': test_patterns
        }

    def _check_color_range_inclusion(self, train_inputs: List, test_inputs: List) -> float:
        """色の範囲包含性をチェック"""
        # 訓練データの色範囲
        train_colors = set()
        for grid in train_inputs:
            for row in grid:
                train_colors.update(row)

        # テストデータの色範囲
        test_colors = set()
        for grid in test_inputs:
            for row in grid:
                test_colors.update(row)

        # テストの色が訓練の範囲内にあるか
        if not train_colors:
            return 1.0 if not test_colors else 0.0

        included_colors = test_colors.intersection(train_colors)
        total_test_colors = len(test_colors)

        if total_test_colors == 0:
            return 1.0

        inclusion_ratio = len(included_colors) / total_test_colors

        # 許容範囲内かチェック
        if inclusion_ratio >= (1.0 - self.config.color_range_tolerance):
            return 1.0
        else:
            return inclusion_ratio

    def _check_size_range_inclusion(self, train_inputs: List, test_inputs: List) -> float:
        """グリッドサイズの範囲包含性をチェック"""
        # 訓練データのサイズ範囲
        train_sizes = []
        for grid in train_inputs:
            if grid:
                train_sizes.append((len(grid), len(grid[0]) if grid[0] else 0))

        # テストデータのサイズ範囲
        test_sizes = []
        for grid in test_inputs:
            if grid:
                test_sizes.append((len(grid), len(grid[0]) if grid[0] else 0))

        if not train_sizes or not test_sizes:
            return 1.0

        # サイズ範囲を計算
        train_min_h, train_min_w = min(train_sizes)
        train_max_h, train_max_w = max(train_sizes)

        # テストサイズが訓練範囲内にあるか
        included_count = 0
        for h, w in test_sizes:
            if (train_min_h <= h <= train_max_h and
                train_min_w <= w <= train_max_w):
                included_count += 1

        inclusion_ratio = included_count / len(test_sizes)

        # 許容範囲内かチェック
        if inclusion_ratio >= (1.0 - self.config.size_range_tolerance):
            return 1.0
        else:
            return inclusion_ratio

    def _check_object_count_range_inclusion(self, train_inputs: List, test_inputs: List) -> float:
        """オブジェクト数の範囲包含性をチェック"""
        # 訓練データのオブジェクト数範囲
        train_object_counts = []
        for grid in train_inputs:
            unique_colors = set()
            for row in grid:
                unique_colors.update(row)
            # 背景色（通常0）を除く
            train_object_counts.append(len(unique_colors - {0}))

        # テストデータのオブジェクト数範囲
        test_object_counts = []
        for grid in test_inputs:
            unique_colors = set()
            for row in grid:
                unique_colors.update(row)
            test_object_counts.append(len(unique_colors - {0}))

        if not train_object_counts or not test_object_counts:
            return 1.0

        # オブジェクト数範囲を計算
        train_min_objects = min(train_object_counts)
        train_max_objects = max(train_object_counts)

        # テストオブジェクト数が訓練範囲内にあるか
        included_count = 0
        for count in test_object_counts:
            if train_min_objects <= count <= train_max_objects:
                included_count += 1

        inclusion_ratio = included_count / len(test_object_counts)
        return inclusion_ratio

    def _check_spatial_range_inclusion(self, train_inputs: List, test_inputs: List) -> float:
        """空間配置の範囲包含性をチェック（本格実装）"""
        if not train_inputs or not test_inputs:
            return 1.0

        # 訓練データの空間配置範囲を計算
        train_spatial_ranges = []
        for grid in train_inputs:
            if not grid:
                continue
            # 非ゼロピクセルの位置を取得
            non_zero_positions = []
            for i, row in enumerate(grid):
                for j, val in enumerate(row):
                    if val != 0:
                        non_zero_positions.append((i, j))

            if non_zero_positions:
                # バウンディングボックスを計算
                min_y = min(pos[0] for pos in non_zero_positions)
                max_y = max(pos[0] for pos in non_zero_positions)
                min_x = min(pos[1] for pos in non_zero_positions)
                max_x = max(pos[1] for pos in non_zero_positions)

                # グリッドサイズで正規化
                h, w = len(grid), len(grid[0]) if grid[0] else 0
                if h > 0 and w > 0:
                    train_spatial_ranges.append({
                        'min_y': min_y / h,
                        'max_y': max_y / h,
                        'min_x': min_x / w,
                        'max_x': max_x / w,
                        'center_y': (min_y + max_y) / (2 * h),
                        'center_x': (min_x + max_x) / (2 * w)
                    })

        # テストデータの空間配置範囲を計算
        test_spatial_ranges = []
        for grid in test_inputs:
            if not grid:
                continue
            # 非ゼロピクセルの位置を取得
            non_zero_positions = []
            for i, row in enumerate(grid):
                for j, val in enumerate(row):
                    if val != 0:
                        non_zero_positions.append((i, j))

            if non_zero_positions:
                # バウンディングボックスを計算
                min_y = min(pos[0] for pos in non_zero_positions)
                max_y = max(pos[0] for pos in non_zero_positions)
                min_x = min(pos[1] for pos in non_zero_positions)
                max_x = max(pos[1] for pos in non_zero_positions)

                # グリッドサイズで正規化
                h, w = len(grid), len(grid[0]) if grid[0] else 0
                if h > 0 and w > 0:
                    test_spatial_ranges.append({
                        'min_y': min_y / h,
                        'max_y': max_y / h,
                        'min_x': min_x / w,
                        'max_x': max_x / w,
                        'center_y': (min_y + max_y) / (2 * h),
                        'center_x': (min_x + max_x) / (2 * w)
                    })

        if not train_spatial_ranges or not test_spatial_ranges:
            return 1.0

        # 訓練データの範囲を計算
        train_min_y = min(r['min_y'] for r in train_spatial_ranges)
        train_max_y = max(r['max_y'] for r in train_spatial_ranges)
        train_min_x = min(r['min_x'] for r in train_spatial_ranges)
        train_max_x = max(r['max_x'] for r in train_spatial_ranges)

        # テストデータが訓練データの範囲内にあるかチェック
        included_count = 0
        for test_range in test_spatial_ranges:
            if (train_min_y <= test_range['min_y'] <= train_max_y and
                train_min_y <= test_range['max_y'] <= train_max_y and
                train_min_x <= test_range['min_x'] <= train_max_x and
                train_min_x <= test_range['max_x'] <= train_max_x):
                included_count += 1

        inclusion_ratio = included_count / len(test_spatial_ranges) if test_spatial_ranges else 1.0
        return inclusion_ratio

    def _extract_transformation_pattern(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """変換パターンを抽出"""
        input_grid = pair['input']
        output_grid = pair['output']

        if not input_grid or not output_grid:
            return {'type': 'empty', 'confidence': 0.0}

        # 基本的な変換パターンを識別
        input_size = len(input_grid) * len(input_grid[0]) if input_grid else 0
        output_size = len(output_grid) * len(output_grid[0]) if output_grid else 0

        # 色の変化
        input_colors = set()
        output_colors = set()

        for row in input_grid:
            input_colors.update(row)
        for row in output_grid:
            output_colors.update(row)

        color_changes = len(output_colors - input_colors)

        # パターンタイプを決定
        if input_size == output_size and color_changes == 0:
            pattern_type = 'identity'
        elif input_size == output_size and color_changes > 0:
            pattern_type = 'color_change'
        elif output_size > input_size:
            pattern_type = 'expansion'
        elif output_size < input_size:
            pattern_type = 'reduction'
        else:
            pattern_type = 'complex'

        return {
            'type': pattern_type,
            'size_change': output_size - input_size,
            'color_changes': color_changes,
            'confidence': 1.0
        }

    def _calculate_pattern_consistency(self, train_patterns: List, test_patterns: List) -> float:
        """パターンの一貫性を計算"""
        if not train_patterns or not test_patterns:
            return 1.0

        # 訓練パターンのタイプ分布
        train_types = [pattern['type'] for pattern in train_patterns]
        train_type_counts = defaultdict(int)
        for pattern_type in train_types:
            train_type_counts[pattern_type] += 1

        # テストパターンが訓練パターンに含まれているか
        consistent_count = 0
        for test_pattern in test_patterns:
            test_type = test_pattern['type']
            if test_type in train_type_counts:
                consistent_count += 1

        consistency_ratio = consistent_count / len(test_patterns)
        return consistency_ratio

    def _check_complexity_consistency(self, train_pairs: List, test_pairs: List) -> float:
        """変換の複雑度一貫性をチェック（本格実装）"""
        if not train_pairs or not test_pairs:
            return 1.0

        # 訓練ペアの複雑度を計算
        train_complexities = []
        for pair in train_pairs:
            input_grid = pair.get('input', [])
            output_grid = pair.get('output', [])

            if not input_grid or not output_grid:
                continue

            # 複雑度の指標:
            # 1. 色の変化数
            input_colors = set(c for row in input_grid for c in row)
            output_colors = set(c for row in output_grid for c in row)
            color_changes = len(output_colors - input_colors)

            # 2. サイズの変化率
            input_size = len(input_grid) * len(input_grid[0]) if input_grid[0] else 0
            output_size = len(output_grid) * len(output_grid[0]) if output_grid[0] else 0
            size_change_ratio = abs(output_size - input_size) / max(input_size, 1)

            # 3. 非ゼロピクセルの変化率
            input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
            output_nonzero = sum(1 for row in output_grid for c in row if c != 0)
            nonzero_change_ratio = abs(output_nonzero - input_nonzero) / max(input_nonzero, 1)

            # 総合複雑度
            complexity = color_changes * 0.3 + size_change_ratio * 0.4 + nonzero_change_ratio * 0.3
            train_complexities.append(complexity)

        # テストペアの複雑度を計算
        test_complexities = []
        for pair in test_pairs:
            input_grid = pair.get('input', [])
            output_grid = pair.get('output', [])

            if not input_grid or not output_grid:
                continue

            # 同じ複雑度計算
            input_colors = set(c for row in input_grid for c in row)
            output_colors = set(c for row in output_grid for c in row)
            color_changes = len(output_colors - input_colors)

            input_size = len(input_grid) * len(input_grid[0]) if input_grid[0] else 0
            output_size = len(output_grid) * len(output_grid[0]) if output_grid[0] else 0
            size_change_ratio = abs(output_size - input_size) / max(input_size, 1)

            input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
            output_nonzero = sum(1 for row in output_grid for c in row if c != 0)
            nonzero_change_ratio = abs(output_nonzero - input_nonzero) / max(input_nonzero, 1)

            complexity = color_changes * 0.3 + size_change_ratio * 0.4 + nonzero_change_ratio * 0.3
            test_complexities.append(complexity)

        if not train_complexities or not test_complexities:
            return 1.0

        # 複雑度の範囲を計算
        train_min_complexity = min(train_complexities)
        train_max_complexity = max(train_complexities)
        train_avg_complexity = sum(train_complexities) / len(train_complexities)

        # テスト複雑度が訓練複雑度の範囲内にあるか
        included_count = 0
        for test_complexity in test_complexities:
            # 平均からの偏差が許容範囲内か
            if abs(test_complexity - train_avg_complexity) <= (train_max_complexity - train_min_complexity) * 0.5:
                included_count += 1

        consistency_ratio = included_count / len(test_complexities) if test_complexities else 1.0
        return consistency_ratio

    def _check_direction_consistency(self, train_pairs: List, test_pairs: List) -> float:
        """変換の方向性一貫性をチェック（本格実装）"""
        if not train_pairs or not test_pairs:
            return 1.0

        # 訓練ペアの方向性を計算
        train_directions = []
        for pair in train_pairs:
            input_grid = pair.get('input', [])
            output_grid = pair.get('output', [])

            if not input_grid or not output_grid:
                continue

            # 方向性の指標:
            # 1. サイズの変化方向（拡大/縮小/維持）
            input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid[0] else 0
            output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid[0] else 0

            size_direction = 'maintain'
            if output_h > input_h or output_w > input_w:
                size_direction = 'expand'
            elif output_h < input_h or output_w < input_w:
                size_direction = 'shrink'

            # 2. 非ゼロピクセル数の変化方向
            input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
            output_nonzero = sum(1 for row in output_grid for c in row if c != 0)

            pixel_direction = 'maintain'
            if output_nonzero > input_nonzero:
                pixel_direction = 'increase'
            elif output_nonzero < input_nonzero:
                pixel_direction = 'decrease'

            # 3. 色の変化方向（増加/減少/維持）
            input_colors = set(c for row in input_grid for c in row)
            output_colors = set(c for row in output_grid for c in row)

            color_direction = 'maintain'
            if len(output_colors) > len(input_colors):
                color_direction = 'increase'
            elif len(output_colors) < len(input_colors):
                color_direction = 'decrease'

            train_directions.append({
                'size': size_direction,
                'pixel': pixel_direction,
                'color': color_direction
            })

        # テストペアの方向性を計算
        test_directions = []
        for pair in test_pairs:
            input_grid = pair.get('input', [])
            output_grid = pair.get('output', [])

            if not input_grid or not output_grid:
                continue

            input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid[0] else 0
            output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid[0] else 0

            size_direction = 'maintain'
            if output_h > input_h or output_w > input_w:
                size_direction = 'expand'
            elif output_h < input_h or output_w < input_w:
                size_direction = 'shrink'

            input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
            output_nonzero = sum(1 for row in output_grid for c in row if c != 0)

            pixel_direction = 'maintain'
            if output_nonzero > input_nonzero:
                pixel_direction = 'increase'
            elif output_nonzero < input_nonzero:
                pixel_direction = 'decrease'

            input_colors = set(c for row in input_grid for c in row)
            output_colors = set(c for row in output_grid for c in row)

            color_direction = 'maintain'
            if len(output_colors) > len(input_colors):
                color_direction = 'increase'
            elif len(output_colors) < len(input_colors):
                color_direction = 'decrease'

            test_directions.append({
                'size': size_direction,
                'pixel': pixel_direction,
                'color': color_direction
            })

        if not train_directions or not test_directions:
            return 1.0

        # 訓練方向性の分布を計算
        train_size_dist = defaultdict(int)
        train_pixel_dist = defaultdict(int)
        train_color_dist = defaultdict(int)

        for direction in train_directions:
            train_size_dist[direction['size']] += 1
            train_pixel_dist[direction['pixel']] += 1
            train_color_dist[direction['color']] += 1

        # テスト方向性が訓練方向性に一致するか
        consistent_count = 0
        for test_direction in test_directions:
            # 各方向性が訓練分布に含まれているか
            size_match = test_direction['size'] in train_size_dist
            pixel_match = test_direction['pixel'] in train_pixel_dist
            color_match = test_direction['color'] in train_color_dist

            # すべて一致している場合
            if size_match and pixel_match and color_match:
                consistent_count += 1

        consistency_ratio = consistent_count / len(test_directions) if test_directions else 1.0
        return consistency_ratio

    def _perform_detailed_analysis(self, task: Task) -> Dict[str, Any]:
        """詳細分析を実行"""
        return {
            'color_analysis': self._analyze_color_patterns(task),
            'size_analysis': self._analyze_size_patterns(task),
            'spatial_analysis': self._analyze_spatial_patterns(task),
            'transformation_analysis': self._analyze_transformation_patterns(task)
        }

    def _analyze_color_patterns(self, task: Task) -> Dict[str, Any]:
        """色パターンを分析"""
        train_colors = set()
        test_colors = set()

        for grid in task.get_train_inputs():
            for row in grid:
                train_colors.update(row)

        for grid in task.get_test_inputs():
            for row in grid:
                test_colors.update(row)

        return {
            'train_colors': list(train_colors),
            'test_colors': list(test_colors),
            'color_overlap': len(train_colors.intersection(test_colors)),
            'color_diversity': len(train_colors.union(test_colors))
        }

    def _analyze_size_patterns(self, task: Task) -> Dict[str, Any]:
        """サイズパターンを分析"""
        train_sizes = []
        test_sizes = []

        for grid in task.get_train_inputs():
            if grid:
                train_sizes.append((len(grid), len(grid[0]) if grid[0] else 0))

        for grid in task.get_test_inputs():
            if grid:
                test_sizes.append((len(grid), len(grid[0]) if grid[0] else 0))

        return {
            'train_size_range': (min(train_sizes), max(train_sizes)) if train_sizes else None,
            'test_size_range': (min(test_sizes), max(test_sizes)) if test_sizes else None,
            'size_consistency': len(set(train_sizes).intersection(set(test_sizes))) / max(len(set(train_sizes)), 1)
        }

    def _analyze_spatial_patterns(self, task: Task) -> Dict[str, Any]:
        """空間パターンを分析（本格実装）"""
        train_inputs = task.get_train_inputs()
        test_inputs = task.get_test_inputs()

        # 訓練データの空間パターンを分析
        train_spatial_features = []
        for grid in train_inputs:
            if not grid:
                continue
            # 非ゼロピクセルの位置を取得
            non_zero_positions = [(i, j) for i, row in enumerate(grid)
                                 for j, val in enumerate(row) if val != 0]

            if non_zero_positions:
                h, w = len(grid), len(grid[0]) if grid[0] else 0
                if h > 0 and w > 0:
                    # 中心位置
                    center_y = sum(pos[0] for pos in non_zero_positions) / len(non_zero_positions) / h
                    center_x = sum(pos[1] for pos in non_zero_positions) / len(non_zero_positions) / w

                    # 分散
                    var_y = sum((pos[0] / h - center_y) ** 2 for pos in non_zero_positions) / len(non_zero_positions)
                    var_x = sum((pos[1] / w - center_x) ** 2 for pos in non_zero_positions) / len(non_zero_positions)

                    train_spatial_features.append({
                        'center_y': center_y,
                        'center_x': center_x,
                        'var_y': var_y,
                        'var_x': var_x,
                        'count': len(non_zero_positions)
                    })

        # テストデータの空間パターンを分析
        test_spatial_features = []
        for grid in test_inputs:
            if not grid:
                continue
            non_zero_positions = [(i, j) for i, row in enumerate(grid)
                                 for j, val in enumerate(row) if val != 0]

            if non_zero_positions:
                h, w = len(grid), len(grid[0]) if grid[0] else 0
                if h > 0 and w > 0:
                    center_y = sum(pos[0] for pos in non_zero_positions) / len(non_zero_positions) / h
                    center_x = sum(pos[1] for pos in non_zero_positions) / len(non_zero_positions) / w

                    var_y = sum((pos[0] / h - center_y) ** 2 for pos in non_zero_positions) / len(non_zero_positions)
                    var_x = sum((pos[1] / w - center_x) ** 2 for pos in non_zero_positions) / len(non_zero_positions)

                    test_spatial_features.append({
                        'center_y': center_y,
                        'center_x': center_x,
                        'var_y': var_y,
                        'var_x': var_x,
                        'count': len(non_zero_positions)
                    })

        return {
            'train_spatial_features': train_spatial_features,
            'test_spatial_features': test_spatial_features,
            'train_avg_center': (
                sum(f['center_y'] for f in train_spatial_features) / len(train_spatial_features) if train_spatial_features else 0,
                sum(f['center_x'] for f in train_spatial_features) / len(train_spatial_features) if train_spatial_features else 0
            ),
            'test_avg_center': (
                sum(f['center_y'] for f in test_spatial_features) / len(test_spatial_features) if test_spatial_features else 0,
                sum(f['center_x'] for f in test_spatial_features) / len(test_spatial_features) if test_spatial_features else 0
            )
        }

    def _analyze_transformation_patterns(self, task: Task) -> Dict[str, Any]:
        """変換パターンを分析"""
        train_patterns = [self._extract_transformation_pattern(pair) for pair in task.train]
        test_patterns = [self._extract_transformation_pattern(pair) for pair in task.test]

        return {
            'train_pattern_types': [p['type'] for p in train_patterns],
            'test_pattern_types': [p['type'] for p in test_patterns],
            'pattern_overlap': len(set(p['type'] for p in train_patterns).intersection(set(p['type'] for p in test_patterns)))
        }

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """評価統計を取得"""
        stats = dict(self.evaluation_stats)

        if stats['total_tasks_evaluated'] > 0:
            stats['test_range_pass_rate'] = stats['tasks_passed_test_range'] / stats['total_tasks_evaluated']
            stats['pattern_consistency_pass_rate'] = stats['tasks_passed_pattern_consistency'] / stats['total_tasks_evaluated']
            stats['overall_pass_rate'] = stats['tasks_passed_both'] / stats['total_tasks_evaluated']
        else:
            stats['test_range_pass_rate'] = 0.0
            stats['pattern_consistency_pass_rate'] = 0.0
            stats['overall_pass_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.evaluation_stats = {
            'total_tasks_evaluated': 0,
            'tasks_passed_test_range': 0,
            'tasks_passed_pattern_consistency': 0,
            'tasks_passed_both': 0,
            'average_test_range_score': 0.0,
            'average_pattern_consistency_score': 0.0
        }
