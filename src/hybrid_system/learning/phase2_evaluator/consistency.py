"""
一貫性チェッカー

タスクの一貫性をチェックする機能
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from core.data_structures import Task


@dataclass
class ConsistencyConfig:
    """一貫性チェック設定"""
    color_consistency_threshold: float = 0.8
    size_consistency_threshold: float = 0.7
    pattern_consistency_threshold: float = 0.6
    enable_detailed_analysis: bool = True


class ConsistencyChecker:
    """一貫性チェッカー"""

    def __init__(self, config: Optional[ConsistencyConfig] = None):
        """初期化"""
        self.config = config or ConsistencyConfig()

        # チェック統計
        self.check_stats = {
            'total_tasks_checked': 0,
            'color_consistency_passed': 0,
            'size_consistency_passed': 0,
            'pattern_consistency_passed': 0,
            'overall_consistency_passed': 0
        }

    def check_task_consistency(self, task: Task) -> Dict[str, Any]:
        """タスクの一貫性をチェック

        Args:
            task: チェックするタスク

        Returns:
            一貫性チェック結果
        """
        self.check_stats['total_tasks_checked'] += 1

        # 1. 色の一貫性チェック
        color_consistency = self._check_color_consistency(task)

        # 2. サイズの一貫性チェック
        size_consistency = self._check_size_consistency(task)

        # 3. パターンの一貫性チェック
        pattern_consistency = self._check_pattern_consistency(task)

        # 4. 総合一貫性
        overall_consistency = (
            color_consistency['score'] >= self.config.color_consistency_threshold and
            size_consistency['score'] >= self.config.size_consistency_threshold and
            pattern_consistency['score'] >= self.config.pattern_consistency_threshold
        )

        # 統計更新
        if color_consistency['score'] >= self.config.color_consistency_threshold:
            self.check_stats['color_consistency_passed'] += 1
        if size_consistency['score'] >= self.config.size_consistency_threshold:
            self.check_stats['size_consistency_passed'] += 1
        if pattern_consistency['score'] >= self.config.pattern_consistency_threshold:
            self.check_stats['pattern_consistency_passed'] += 1
        if overall_consistency:
            self.check_stats['overall_consistency_passed'] += 1

        return {
            'task_id': task.task_id,
            'overall_consistency': overall_consistency,
            'color_consistency': color_consistency,
            'size_consistency': size_consistency,
            'pattern_consistency': pattern_consistency,
            'detailed_analysis': self._perform_detailed_consistency_analysis(task) if self.config.enable_detailed_analysis else None
        }

    def _check_color_consistency(self, task: Task) -> Dict[str, Any]:
        """色の一貫性をチェック"""
        train_colors = set()
        test_colors = set()

        # 訓練データの色を収集
        for pair in task.train:
            for row in pair['input']:
                train_colors.update(row)
            for row in pair['output']:
                train_colors.update(row)

        # テストデータの色を収集
        for pair in task.test:
            for row in pair['input']:
                test_colors.update(row)
            for row in pair['output']:
                test_colors.update(row)

        # 色の重複率を計算
        if not train_colors and not test_colors:
            overlap_ratio = 1.0
        elif not train_colors or not test_colors:
            overlap_ratio = 0.0
        else:
            overlap_ratio = len(train_colors.intersection(test_colors)) / len(train_colors.union(test_colors))

        # 色の分布の一貫性をチェック
        train_color_counts = self._count_colors_in_pairs(task.train)
        test_color_counts = self._count_colors_in_pairs(task.test)

        distribution_consistency = self._calculate_distribution_consistency(train_color_counts, test_color_counts)

        # 総合スコア
        overall_score = (overlap_ratio + distribution_consistency) / 2.0

        return {
            'score': overall_score,
            'overlap_ratio': overlap_ratio,
            'distribution_consistency': distribution_consistency,
            'train_colors': list(train_colors),
            'test_colors': list(test_colors),
            'passed': overall_score >= self.config.color_consistency_threshold
        }

    def _check_size_consistency(self, task: Task) -> Dict[str, Any]:
        """サイズの一貫性をチェック"""
        train_sizes = []
        test_sizes = []

        # 訓練データのサイズを収集
        for pair in task.train:
            input_size = (len(pair['input']), len(pair['input'][0]) if pair['input'] else 0)
            output_size = (len(pair['output']), len(pair['output'][0]) if pair['output'] else 0)
            train_sizes.extend([input_size, output_size])

        # テストデータのサイズを収集
        for pair in task.test:
            input_size = (len(pair['input']), len(pair['input'][0]) if pair['input'] else 0)
            output_size = (len(pair['output']), len(pair['output'][0]) if pair['output'] else 0)
            test_sizes.extend([input_size, output_size])

        # サイズの一貫性を計算
        if not train_sizes or not test_sizes:
            consistency_score = 1.0
        else:
            train_size_set = set(train_sizes)
            test_size_set = set(test_sizes)

            # テストサイズが訓練サイズに含まれている割合
            if not test_size_set:
                consistency_score = 1.0
            else:
                consistency_score = len(test_size_set.intersection(train_size_set)) / len(test_size_set)

        return {
            'score': consistency_score,
            'train_sizes': train_sizes,
            'test_sizes': test_sizes,
            'size_overlap': len(set(train_sizes).intersection(set(test_sizes))) if train_sizes and test_sizes else 0,
            'passed': consistency_score >= self.config.size_consistency_threshold
        }

    def _check_pattern_consistency(self, task: Task) -> Dict[str, Any]:
        """パターンの一貫性をチェック"""
        train_patterns = []
        test_patterns = []

        # 訓練データのパターンを抽出
        for pair in task.train:
            pattern = self._extract_transformation_pattern(pair)
            train_patterns.append(pattern)

        # テストデータのパターンを抽出
        for pair in task.test:
            pattern = self._extract_transformation_pattern(pair)
            test_patterns.append(pattern)

        # パターンの一貫性を計算
        if not train_patterns or not test_patterns:
            consistency_score = 1.0
        else:
            train_pattern_types = [p['type'] for p in train_patterns]
            test_pattern_types = [p['type'] for p in test_patterns]

            # テストパターンが訓練パターンに含まれている割合
            if not test_pattern_types:
                consistency_score = 1.0
            else:
                consistency_score = len(set(test_pattern_types).intersection(set(train_pattern_types))) / len(set(test_pattern_types))

        return {
            'score': consistency_score,
            'train_patterns': train_patterns,
            'test_patterns': test_patterns,
            'pattern_overlap': len(set([p['type'] for p in train_patterns]).intersection(set([p['type'] for p in test_patterns]))) if train_patterns and test_patterns else 0,
            'passed': consistency_score >= self.config.pattern_consistency_threshold
        }

    def _count_colors_in_pairs(self, pairs: List[Dict[str, Any]]) -> Dict[int, int]:
        """ペアの色数をカウント（共通実装を使用）"""
        from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import (
            get_color_distribution
        )

        color_counts = defaultdict(int)

        for pair in pairs:
            # 入力グリッドの色（共通実装を使用）
            input_dist = get_color_distribution(pair['input'])
            for color, count in input_dist.items():
                color_counts[color] += count

            # 出力グリッドの色（共通実装を使用）
            output_dist = get_color_distribution(pair['output'])
            for color, count in output_dist.items():
                color_counts[color] += count

        return dict(color_counts)

    def _calculate_distribution_consistency(self, train_counts: Dict[int, int], test_counts: Dict[int, int]) -> float:
        """分布の一貫性を計算"""
        if not train_counts or not test_counts:
            return 1.0

        # 共通の色の分布の一貫性をチェック
        common_colors = set(train_counts.keys()).intersection(set(test_counts.keys()))

        if not common_colors:
            return 0.0

        consistency_scores = []
        for color in common_colors:
            train_ratio = train_counts[color] / sum(train_counts.values())
            test_ratio = test_counts[color] / sum(test_counts.values())

            # 比率の差が小さいほど一貫性が高い
            ratio_diff = abs(train_ratio - test_ratio)
            consistency_scores.append(1.0 - ratio_diff)

        return np.mean(consistency_scores) if consistency_scores else 0.0

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

    def _perform_detailed_consistency_analysis(self, task: Task) -> Dict[str, Any]:
        """詳細な一貫性分析を実行"""
        return {
            'color_analysis': self._analyze_color_consistency(task),
            'size_analysis': self._analyze_size_consistency(task),
            'pattern_analysis': self._analyze_pattern_consistency(task)
        }

    def _analyze_color_consistency(self, task: Task) -> Dict[str, Any]:
        """色の一貫性を詳細分析"""
        train_colors = set()
        test_colors = set()

        for pair in task.train:
            for row in pair['input']:
                train_colors.update(row)
            for row in pair['output']:
                train_colors.update(row)

        for pair in task.test:
            for row in pair['input']:
                test_colors.update(row)
            for row in pair['output']:
                test_colors.update(row)

        return {
            'train_color_count': len(train_colors),
            'test_color_count': len(test_colors),
            'color_overlap': len(train_colors.intersection(test_colors)),
            'color_diversity': len(train_colors.union(test_colors))
        }

    def _analyze_size_consistency(self, task: Task) -> Dict[str, Any]:
        """サイズの一貫性を詳細分析"""
        train_sizes = []
        test_sizes = []

        for pair in task.train:
            input_size = (len(pair['input']), len(pair['input'][0]) if pair['input'] else 0)
            output_size = (len(pair['output']), len(pair['output'][0]) if pair['output'] else 0)
            train_sizes.extend([input_size, output_size])

        for pair in task.test:
            input_size = (len(pair['input']), len(pair['input'][0]) if pair['input'] else 0)
            output_size = (len(pair['output']), len(pair['output'][0]) if pair['output'] else 0)
            test_sizes.extend([input_size, output_size])

        return {
            'train_size_count': len(set(train_sizes)),
            'test_size_count': len(set(test_sizes)),
            'size_overlap': len(set(train_sizes).intersection(set(test_sizes))),
            'size_diversity': len(set(train_sizes).union(set(test_sizes)))
        }

    def _analyze_pattern_consistency(self, task: Task) -> Dict[str, Any]:
        """パターンの一貫性を詳細分析"""
        train_patterns = [self._extract_transformation_pattern(pair) for pair in task.train]
        test_patterns = [self._extract_transformation_pattern(pair) for pair in task.test]

        return {
            'train_pattern_count': len(set([p['type'] for p in train_patterns])),
            'test_pattern_count': len(set([p['type'] for p in test_patterns])),
            'pattern_overlap': len(set([p['type'] for p in train_patterns]).intersection(set([p['type'] for p in test_patterns]))),
            'pattern_diversity': len(set([p['type'] for p in train_patterns]).union(set([p['type'] for p in test_patterns])))
        }

    def get_consistency_statistics(self) -> Dict[str, Any]:
        """一貫性チェック統計を取得"""
        stats = dict(self.check_stats)

        if stats['total_tasks_checked'] > 0:
            stats['color_consistency_pass_rate'] = stats['color_consistency_passed'] / stats['total_tasks_checked']
            stats['size_consistency_pass_rate'] = stats['size_consistency_passed'] / stats['total_tasks_checked']
            stats['pattern_consistency_pass_rate'] = stats['pattern_consistency_passed'] / stats['total_tasks_checked']
            stats['overall_consistency_pass_rate'] = stats['overall_consistency_passed'] / stats['total_tasks_checked']
        else:
            stats['color_consistency_pass_rate'] = 0.0
            stats['size_consistency_pass_rate'] = 0.0
            stats['pattern_consistency_pass_rate'] = 0.0
            stats['overall_consistency_pass_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.check_stats = {
            'total_tasks_checked': 0,
            'color_consistency_passed': 0,
            'size_consistency_passed': 0,
            'pattern_consistency_passed': 0,
            'overall_consistency_passed': 0
        }
