"""
一貫性チェッカー

プログラムがすべての訓練ペアで一貫して動作するかチェック
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

try:
    from src.hybrid_system.core.data_structures.task import Task
except ImportError:
    from core.data_structures import Task


@dataclass
class ConsistencyConfig:
    """一貫性チェック設定"""
    consistency_threshold: float = 0.8
    enable_detailed_analysis: bool = True
    strict_mode: bool = True


class ConsistencyChecker:
    """一貫性チェッカー"""

    def __init__(self, config: Optional[ConsistencyConfig] = None):
        """初期化"""
        self.config = config or ConsistencyConfig()

        # チェック統計
        self.check_stats = {
            'total_checks': 0,
            'consistent_programs': 0,
            'inconsistent_programs': 0,
            'average_consistency_score': 0.0
        }

    def check_consistency(self, program: str, task: Task) -> float:
        """プログラムの一貫性をチェック

        Args:
            program: チェックするプログラム
            task: タスク

        Returns:
            一貫性スコア（0.0-1.0）
        """
        self.check_stats['total_checks'] += 1

        if not program or not task.train:
            self.check_stats['inconsistent_programs'] += 1
            return 0.0

        # 各訓練ペアでプログラムの動作をチェック
        consistency_scores = []

        for i, train_pair in enumerate(task.train):
            pair_score = self._check_pair_consistency(program, train_pair)
            consistency_scores.append(pair_score)

        # 平均一貫性スコアを計算
        overall_score = np.mean(consistency_scores) if consistency_scores else 0.0

        # 統計更新
        if overall_score >= self.config.consistency_threshold:
            self.check_stats['consistent_programs'] += 1
        else:
            self.check_stats['inconsistent_programs'] += 1

        # 平均スコア更新
        total_checks = self.check_stats['total_checks']
        self.check_stats['average_consistency_score'] = (
            (self.check_stats['average_consistency_score'] * (total_checks - 1) + overall_score) / total_checks
        )

        return overall_score

    def _check_pair_consistency(
        self,
        program: str,
        pair: Dict[str, Any],
        execution_cache: Optional[Any] = None
    ) -> float:
        """単一ペアでの一貫性をチェック

        Args:
            program: チェックするプログラム
            pair: 訓練ペア
            execution_cache: プログラム実行結果のキャッシュ（オプション）

        Returns:
            一貫性スコア（0.0-1.0）
        """
        input_grid = pair['input']
        expected_output = pair['output']

        # プログラムを実行して出力を取得（キャッシュ付き）
        actual_output = self._execute_program(program, input_grid, execution_cache)

        if actual_output is None:
            return 0.0

        # 出力の一致度を計算
        consistency_score = self._calculate_output_consistency(expected_output, actual_output)

        return consistency_score

    def _execute_program(
        self,
        program: str,
        input_grid: List[List[int]],
        execution_cache: Optional[Any] = None
    ) -> Optional[List[List[int]]]:
        """プログラムを実行

        Args:
            program: 実行するプログラム
            input_grid: 入力グリッド
            execution_cache: プログラム実行結果のキャッシュ（オプション）

        Returns:
            実行結果のグリッド（失敗時はNone）
        """
        if not program or not input_grid:
            return None

        # キャッシュから取得を試みる
        if execution_cache:
            import numpy as np
            input_array = np.array(input_grid)
            cached_result = execution_cache.get_execution_result(program, input_array)
            if cached_result is not None:
                return cached_result.tolist()

        try:
            # ExecutorCoreを使用して実際にプログラムを実行
            from src.core_systems.executor.core import ExecutorCore
            import numpy as np

            executor = ExecutorCore()
            input_array = np.array(input_grid)
            output_array = executor.execute_program(program, input_array)

            if output_array is None:
                return None

            return output_array.tolist() if isinstance(output_array, np.ndarray) else output_array

        except Exception as e:
            # 実行エラーの場合はNone
            return None

    def _calculate_output_consistency(self, expected: List[List[int]], actual: List[List[int]]) -> float:
        """出力の一致度を計算

        Args:
            expected: 期待される出力
            actual: 実際の出力

        Returns:
            一致度スコア（0.0-1.0）
        """
        if not expected or not actual:
            return 0.0

        if len(expected) != len(actual) or len(expected[0]) != len(actual[0]):
            return 0.0

        # ピクセル単位での一致度を計算
        total_pixels = len(expected) * len(expected[0])
        matching_pixels = 0

        for i in range(len(expected)):
            for j in range(len(expected[0])):
                if expected[i][j] == actual[i][j]:
                    matching_pixels += 1

        consistency_score = matching_pixels / total_pixels if total_pixels > 0 else 0.0

        return consistency_score

    def check_detailed_consistency(self, program: str, task: Task) -> Dict[str, Any]:
        """詳細な一貫性チェック

        Args:
            program: チェックするプログラム
            task: タスク

        Returns:
            詳細な一貫性チェック結果
        """
        if not program or not task.train:
            return {
                'overall_score': 0.0,
                'pair_scores': [],
                'detailed_analysis': None
            }

        pair_scores = []
        detailed_analysis = []

        for i, train_pair in enumerate(task.train):
            pair_score = self._check_pair_consistency(program, train_pair)
            pair_scores.append(pair_score)

            # 詳細分析
            if self.config.enable_detailed_analysis:
                analysis = self._analyze_pair_consistency(program, train_pair, pair_score)
                detailed_analysis.append(analysis)

        overall_score = np.mean(pair_scores) if pair_scores else 0.0

        return {
            'overall_score': overall_score,
            'pair_scores': pair_scores,
            'detailed_analysis': detailed_analysis,
            'consistency_variance': np.var(pair_scores) if pair_scores else 0.0,
            'min_pair_score': min(pair_scores) if pair_scores else 0.0,
            'max_pair_score': max(pair_scores) if pair_scores else 0.0
        }

    def _analyze_pair_consistency(self, program: str, pair: Dict[str, Any], consistency_score: float) -> Dict[str, Any]:
        """ペアの一貫性を詳細分析

        Args:
            program: チェックするプログラム
            pair: 訓練ペア
            consistency_score: 一貫性スコア

        Returns:
            詳細分析結果
        """
        input_grid = pair['input']
        expected_output = pair['output']
        actual_output = self._execute_program(program, input_grid)

        analysis = {
            'consistency_score': consistency_score,
            'input_analysis': self._analyze_grid(input_grid),
            'expected_output_analysis': self._analyze_grid(expected_output),
            'actual_output_analysis': self._analyze_grid(actual_output) if actual_output else None,
            'output_differences': self._analyze_output_differences(expected_output, actual_output) if actual_output else None
        }

        return analysis

    def _analyze_grid(self, grid: List[List[int]]) -> Dict[str, Any]:
        """グリッドを分析

        Args:
            grid: 分析するグリッド

        Returns:
            分析結果
        """
        if not grid:
            return {'error': 'Empty grid'}

        # 色の分析
        colors = set()
        color_counts = defaultdict(int)

        for row in grid:
            for color in row:
                colors.add(color)
                color_counts[color] += 1

        # サイズの分析
        height = len(grid)
        width = len(grid[0]) if grid else 0

        return {
            'size': (height, width),
            'unique_colors': len(colors),
            'color_distribution': dict(color_counts),
            'total_pixels': height * width
        }

    def _analyze_output_differences(self, expected: List[List[int]], actual: List[List[int]]) -> Dict[str, Any]:
        """出力の違いを分析

        Args:
            expected: 期待される出力
            actual: 実際の出力

        Returns:
            違いの分析結果
        """
        if not expected or not actual:
            return {'error': 'Empty grids'}

        if len(expected) != len(actual) or len(expected[0]) != len(actual[0]):
            return {'error': 'Size mismatch'}

        differences = []
        total_pixels = len(expected) * len(expected[0])
        matching_pixels = 0

        for i in range(len(expected)):
            for j in range(len(expected[0])):
                if expected[i][j] != actual[i][j]:
                    differences.append({
                        'position': (i, j),
                        'expected': expected[i][j],
                        'actual': actual[i][j]
                    })
                else:
                    matching_pixels += 1

        return {
            'total_differences': len(differences),
            'matching_pixels': matching_pixels,
            'difference_rate': len(differences) / total_pixels if total_pixels > 0 else 0.0,
            'differences': differences[:10]  # 最初の10個の違いのみ
        }

    def get_consistency_statistics(self) -> Dict[str, Any]:
        """一貫性チェック統計を取得"""
        stats = dict(self.check_stats)

        if stats['total_checks'] > 0:
            stats['consistency_rate'] = stats['consistent_programs'] / stats['total_checks']
        else:
            stats['consistency_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.check_stats = {
            'total_checks': 0,
            'consistent_programs': 0,
            'inconsistent_programs': 0,
            'average_consistency_score': 0.0
        }
