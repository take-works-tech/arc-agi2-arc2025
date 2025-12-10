"""
ARC評価器

ARCタスクの評価機能を提供
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from src.hybrid_system.core.data_structures.task import Task


@dataclass
class ARCEvaluationConfig:
    """ARC評価設定"""
    enable_accuracy_evaluation: bool = True
    enable_consistency_evaluation: bool = True
    enable_completeness_evaluation: bool = True
    accuracy_threshold: float = 0.8
    consistency_threshold: float = 0.7
    completeness_threshold: float = 0.6


class ARCEvaluator:
    """ARC評価器"""

    def __init__(self, config: Optional[ARCEvaluationConfig] = None):
        """初期化"""
        self.config = config or ARCEvaluationConfig()

        # 評価統計
        self.evaluation_stats = {
            'total_evaluations': 0,
            'passed_evaluations': 0,
            'failed_evaluations': 0,
            'average_accuracy': 0.0,
            'average_consistency': 0.0,
            'average_completeness': 0.0
        }

    def evaluate_task(self, task: Task) -> Dict[str, Any]:
        """タスクを評価

        Args:
            task: 評価するタスク

        Returns:
            評価結果
        """
        self.evaluation_stats['total_evaluations'] += 1

        evaluation_results = {}

        # 1. 精度評価
        if self.config.enable_accuracy_evaluation:
            accuracy_result = self._evaluate_accuracy(task)
            evaluation_results['accuracy'] = accuracy_result

        # 2. 一貫性評価
        if self.config.enable_consistency_evaluation:
            consistency_result = self._evaluate_consistency(task)
            evaluation_results['consistency'] = consistency_result

        # 3. 完全性評価
        if self.config.enable_completeness_evaluation:
            completeness_result = self._evaluate_completeness(task)
            evaluation_results['completeness'] = completeness_result

        # 総合評価
        overall_score = self._calculate_overall_score(evaluation_results)
        overall_passed = self._check_overall_pass(evaluation_results)

        if overall_passed:
            self.evaluation_stats['passed_evaluations'] += 1
        else:
            self.evaluation_stats['failed_evaluations'] += 1

        # 統計更新
        self._update_statistics(evaluation_results)

        return {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'evaluation_results': evaluation_results,
            'task_id': task.task_id
        }

    def _evaluate_accuracy(self, task: Task) -> Dict[str, Any]:
        """精度を評価"""
        if not task.train or not task.test:
            return {
                'score': 0.0,
                'passed': False,
                'details': 'No train or test data'
            }

        # 訓練ペアの精度を評価
        train_accuracy_scores = []
        for train_pair in task.train:
            accuracy = self._calculate_pair_accuracy(train_pair)
            train_accuracy_scores.append(accuracy)

        # テストペアの精度を評価
        test_accuracy_scores = []
        for test_pair in task.test:
            accuracy = self._calculate_pair_accuracy(test_pair)
            test_accuracy_scores.append(accuracy)

        # 平均精度を計算
        avg_train_accuracy = np.mean(train_accuracy_scores) if train_accuracy_scores else 0.0
        avg_test_accuracy = np.mean(test_accuracy_scores) if test_accuracy_scores else 0.0
        overall_accuracy = (avg_train_accuracy + avg_test_accuracy) / 2.0

        return {
            'score': overall_accuracy,
            'passed': overall_accuracy >= self.config.accuracy_threshold,
            'train_accuracy': avg_train_accuracy,
            'test_accuracy': avg_test_accuracy,
            'details': f'Train: {avg_train_accuracy:.3f}, Test: {avg_test_accuracy:.3f}'
        }

    def _evaluate_consistency(self, task: Task) -> Dict[str, Any]:
        """一貫性を評価"""
        if not task.train:
            return {
                'score': 0.0,
                'passed': False,
                'details': 'No train data'
            }

        # 訓練ペア間の一貫性を評価
        consistency_scores = []
        for i in range(len(task.train)):
            for j in range(i + 1, len(task.train)):
                consistency = self._calculate_pair_consistency(task.train[i], task.train[j])
                consistency_scores.append(consistency)

        # 平均一貫性を計算
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        return {
            'score': avg_consistency,
            'passed': avg_consistency >= self.config.consistency_threshold,
            'details': f'Average consistency: {avg_consistency:.3f}'
        }

    def _evaluate_completeness(self, task: Task) -> Dict[str, Any]:
        """完全性を評価"""
        if not task.train or not task.test:
            return {
                'score': 0.0,
                'passed': False,
                'details': 'No train or test data'
            }

        # 訓練データの完全性を評価
        train_completeness = self._calculate_data_completeness(task.train)

        # テストデータの完全性を評価
        test_completeness = self._calculate_data_completeness(task.test)

        # 総合完全性を計算
        overall_completeness = (train_completeness + test_completeness) / 2.0

        return {
            'score': overall_completeness,
            'passed': overall_completeness >= self.config.completeness_threshold,
            'train_completeness': train_completeness,
            'test_completeness': test_completeness,
            'details': f'Train: {train_completeness:.3f}, Test: {test_completeness:.3f}'
        }

    def _calculate_pair_accuracy(self, pair: Dict[str, Any]) -> float:
        """ペアの精度を計算"""
        input_grid = pair['input']
        output_grid = pair['output']

        if not input_grid or not output_grid:
            return 0.0

        # グリッドサイズの一致をチェック
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return 0.0

        # ピクセル単位での一致度を計算
        total_pixels = len(input_grid) * len(input_grid[0])
        matching_pixels = 0

        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                if input_grid[i][j] == output_grid[i][j]:
                    matching_pixels += 1

        accuracy = matching_pixels / total_pixels if total_pixels > 0 else 0.0

        return accuracy

    def _calculate_pair_consistency(self, pair1: Dict[str, Any], pair2: Dict[str, Any]) -> float:
        """ペア間の一貫性を計算（本格実装）"""
        # 2つのペアが同じ変換規則に従っているかをチェック
        input1 = pair1.get('input', [])
        output1 = pair1.get('output', [])
        input2 = pair2.get('input', [])
        output2 = pair2.get('output', [])

        if not input1 or not output1 or not input2 or not output2:
            return 0.0

        consistency_scores = []

        # 1. サイズ変換の一貫性
        input1_h, input1_w = len(input1), len(input1[0]) if input1 else 0
        output1_h, output1_w = len(output1), len(output1[0]) if output1 else 0
        input2_h, input2_w = len(input2), len(input2[0]) if input2 else 0
        output2_h, output2_w = len(output2), len(output2[0]) if output2 else 0

        if input1_h > 0 and input1_w > 0 and input2_h > 0 and input2_w > 0:
            ratio1_h = output1_h / input1_h
            ratio1_w = output1_w / input1_w
            ratio2_h = output2_h / input2_h
            ratio2_w = output2_w / input2_w

            # サイズ変換の比率が一致しているか
            if abs(ratio1_h - ratio2_h) < 0.1 and abs(ratio1_w - ratio2_w) < 0.1:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)

        # 2. 色の変換の一貫性
        input1_colors = set(c for row in input1 for c in row if c != 0)
        output1_colors = set(c for row in output1 for c in row if c != 0)
        input2_colors = set(c for row in input2 for c in row if c != 0)
        output2_colors = set(c for row in output2 for c in row if c != 0)

        # 入力色の集合が一致しているか
        if input1_colors == input2_colors:
            consistency_scores.append(1.0)
        else:
            # 部分的に一致
            overlap = len(input1_colors & input2_colors)
            union = len(input1_colors | input2_colors)
            if union > 0:
                consistency_scores.append(overlap / union)
            else:
                consistency_scores.append(0.0)

        # 出力色の集合が一致しているか
        if output1_colors == output2_colors:
            consistency_scores.append(1.0)
        else:
            # 部分的に一致
            overlap = len(output1_colors & output2_colors)
            union = len(output1_colors | output2_colors)
            if union > 0:
                consistency_scores.append(overlap / union)
            else:
                consistency_scores.append(0.0)

        # 3. 非ゼロピクセル数の変換の一貫性
        input1_nonzero = sum(1 for row in input1 for c in row if c != 0)
        output1_nonzero = sum(1 for row in output1 for c in row if c != 0)
        input2_nonzero = sum(1 for row in input2 for c in row if c != 0)
        output2_nonzero = sum(1 for row in output2 for c in row if c != 0)

        if input1_nonzero > 0 and input2_nonzero > 0:
            ratio1 = output1_nonzero / input1_nonzero
            ratio2 = output2_nonzero / input2_nonzero
            if abs(ratio1 - ratio2) < 0.1:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)
        else:
            consistency_scores.append(1.0 if input1_nonzero == input2_nonzero else 0.5)

        # 平均を返す
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

    def _calculate_data_completeness(self, data: List[Dict[str, Any]]) -> float:
        """データの完全性を計算"""
        if not data:
            return 0.0

        completeness_scores = []

        for pair in data:
            # 入力と出力の存在をチェック
            has_input = bool(pair.get('input'))
            has_output = bool(pair.get('output'))

            # グリッドの妥当性をチェック
            input_valid = self._is_grid_valid(pair.get('input', []))
            output_valid = self._is_grid_valid(pair.get('output', []))

            # 完全性スコアを計算
            completeness = (has_input + has_output + input_valid + output_valid) / 4.0
            completeness_scores.append(completeness)

        return np.mean(completeness_scores) if completeness_scores else 0.0

    def _is_grid_valid(self, grid: List[List[int]]) -> bool:
        """グリッドの妥当性をチェック"""
        if not grid:
            return False

        # 基本的な妥当性チェック
        if not isinstance(grid, list):
            return False

        if not grid[0] or not isinstance(grid[0], list):
            return False

        # 色の妥当性をチェック
        for row in grid:
            if not isinstance(row, list):
                return False
            for color in row:
                if not isinstance(color, int) or color < 0 or color > 9:
                    return False

        return True

    def _calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """総合スコアを計算"""
        scores = []

        for result in evaluation_results.values():
            if isinstance(result, dict) and 'score' in result:
                scores.append(result['score'])

        return np.mean(scores) if scores else 0.0

    def _check_overall_pass(self, evaluation_results: Dict[str, Any]) -> bool:
        """総合合格をチェック"""
        for result in evaluation_results.values():
            if isinstance(result, dict) and 'passed' in result:
                if not result['passed']:
                    return False

        return True

    def _update_statistics(self, evaluation_results: Dict[str, Any]):
        """統計を更新"""
        # 精度統計の更新
        if 'accuracy' in evaluation_results:
            accuracy_score = evaluation_results['accuracy']['score']
            total = self.evaluation_stats['total_evaluations']
            self.evaluation_stats['average_accuracy'] = (
                (self.evaluation_stats['average_accuracy'] * (total - 1) + accuracy_score) / total
            )

        # 一貫性統計の更新
        if 'consistency' in evaluation_results:
            consistency_score = evaluation_results['consistency']['score']
            total = self.evaluation_stats['total_evaluations']
            self.evaluation_stats['average_consistency'] = (
                (self.evaluation_stats['average_consistency'] * (total - 1) + consistency_score) / total
            )

        # 完全性統計の更新
        if 'completeness' in evaluation_results:
            completeness_score = evaluation_results['completeness']['score']
            total = self.evaluation_stats['total_evaluations']
            self.evaluation_stats['average_completeness'] = (
                (self.evaluation_stats['average_completeness'] * (total - 1) + completeness_score) / total
            )

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """評価統計を取得"""
        stats = dict(self.evaluation_stats)

        if stats['total_evaluations'] > 0:
            stats['pass_rate'] = stats['passed_evaluations'] / stats['total_evaluations']
        else:
            stats['pass_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'passed_evaluations': 0,
            'failed_evaluations': 0,
            'average_accuracy': 0.0,
            'average_consistency': 0.0,
            'average_completeness': 0.0
        }
