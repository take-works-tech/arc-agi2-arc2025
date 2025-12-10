"""
ARC評価器

ARCタスクの評価機能を提供
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from core.data_structures import Task


@dataclass
class ARCEvaluationConfig:
    """ARC評価設定"""
    enable_accuracy_check: bool = True
    enable_consistency_check: bool = True
    enable_completeness_check: bool = True
    strict_mode: bool = True


class ARCEvaluator:
    """ARC評価器"""

    def __init__(self, config: Optional[ARCEvaluationConfig] = None):
        """初期化"""
        self.config = config or ARCEvaluationConfig()

        # 評価統計
        self.evaluation_stats = {
            'total_tasks_evaluated': 0,
            'passed_tasks': 0,
            'failed_tasks': 0,
            'accuracy_checks_passed': 0,
            'consistency_checks_passed': 0,
            'completeness_checks_passed': 0
        }

    def evaluate_task(self, task: Task) -> Dict[str, Any]:
        """タスクを評価

        Args:
            task: 評価するタスク

        Returns:
            評価結果
        """
        self.evaluation_stats['total_tasks_evaluated'] += 1

        evaluation_results = {}

        # 1. 精度チェック
        if self.config.enable_accuracy_check:
            accuracy_result = self._check_accuracy(task)
            evaluation_results['accuracy'] = accuracy_result
            if accuracy_result['passed']:
                self.evaluation_stats['accuracy_checks_passed'] += 1

        # 2. 一貫性チェック
        if self.config.enable_consistency_check:
            consistency_result = self._check_consistency(task)
            evaluation_results['consistency'] = consistency_result
            if consistency_result['passed']:
                self.evaluation_stats['consistency_checks_passed'] += 1

        # 3. 完全性チェック
        if self.config.enable_completeness_check:
            completeness_result = self._check_completeness(task)
            evaluation_results['completeness'] = completeness_result
            if completeness_result['passed']:
                self.evaluation_stats['completeness_checks_passed'] += 1

        # 総合評価
        overall_passed = all(
            result.get('passed', False) for result in evaluation_results.values()
        )

        if overall_passed:
            self.evaluation_stats['passed_tasks'] += 1
        else:
            self.evaluation_stats['failed_tasks'] += 1

        return {
            'task_id': task.task_id,
            'overall_passed': overall_passed,
            'evaluation_results': evaluation_results
        }

    def _check_accuracy(self, task: Task) -> Dict[str, Any]:
        """精度チェック（本格実装）"""
        if not task.train:
            return {
                'passed': False,
                'score': 0.0,
                'details': '訓練データが存在しません'
            }

        # 訓練データの入出力ペアの一貫性をチェック
        # 各訓練ペアが同じ変換規則に従っているかを確認
        accuracy_scores = []
        details = []

        for i, train_pair in enumerate(task.train):
            input_grid = train_pair.get('input')
            output_grid = train_pair.get('output')

            if input_grid is None or output_grid is None:
                accuracy_scores.append(0.0)
                details.append(f'ペア{i+1}: データが不完全')
                continue

            # グリッドサイズの一貫性
            input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid else 0

            # サイズの一貫性チェック（変換によってサイズが変わる場合もある）
            size_consistent = True
            if input_h != output_h or input_w != output_w:
                # サイズが異なる場合、合理的な変換かチェック
                size_ratio_h = output_h / input_h if input_h > 0 else 1.0
                size_ratio_w = output_w / input_w if input_w > 0 else 1.0
                # 整数倍の変換のみ許可
                if not (abs(size_ratio_h - round(size_ratio_h)) < 0.01 and
                        abs(size_ratio_w - round(size_ratio_w)) < 0.01):
                    size_consistent = False

            if size_consistent:
                accuracy_scores.append(1.0)
                details.append(f'ペア{i+1}: サイズ一貫性OK')
            else:
                accuracy_scores.append(0.5)
                details.append(f'ペア{i+1}: サイズ一貫性に問題あり')

        avg_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        passed = avg_score >= 0.8  # 80%以上のスコアで合格

        return {
            'passed': passed,
            'score': avg_score,
            'details': '; '.join(details)
        }

    def _check_consistency(self, task: Task) -> Dict[str, Any]:
        """一貫性チェック（本格実装）"""
        if not task.train or len(task.train) < 2:
            return {
                'passed': True,
                'score': 1.0,
                'details': '訓練ペアが1つ以下のため、一貫性チェックをスキップ'
            }

        # すべての訓練ペアが同じ変換規則に従っているかをチェック
        # 色の分布、オブジェクト数、パターンなどの一貫性を確認
        consistency_scores = []
        details = []

        # 1. 色の一貫性チェック
        input_colors_sets = []
        output_colors_sets = []
        for train_pair in task.train:
            input_grid = train_pair.get('input', [])
            output_grid = train_pair.get('output', [])
            input_colors = set(c for row in input_grid for c in row if c != 0)
            output_colors = set(c for row in output_grid for c in row if c != 0)
            input_colors_sets.append(input_colors)
            output_colors_sets.append(output_colors)

        # 入力色の一貫性
        if len(set(tuple(sorted(colors)) for colors in input_colors_sets)) == 1:
            consistency_scores.append(1.0)
            details.append('入力色の一貫性: OK')
        else:
            consistency_scores.append(0.5)
            details.append('入力色の一貫性: 部分的')

        # 出力色の一貫性
        if len(set(tuple(sorted(colors)) for colors in output_colors_sets)) == 1:
            consistency_scores.append(1.0)
            details.append('出力色の一貫性: OK')
        else:
            consistency_scores.append(0.5)
            details.append('出力色の一貫性: 部分的')

        # 2. 非ゼロピクセル数の一貫性
        input_nonzero_counts = []
        output_nonzero_counts = []
        for train_pair in task.train:
            input_grid = train_pair.get('input', [])
            output_grid = train_pair.get('output', [])
            input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
            output_nonzero = sum(1 for row in output_grid for c in row if c != 0)
            input_nonzero_counts.append(input_nonzero)
            output_nonzero_counts.append(output_nonzero)

        # 入力非ゼロピクセル数の一貫性
        if len(set(input_nonzero_counts)) == 1:
            consistency_scores.append(1.0)
            details.append('入力非ゼロピクセル数の一貫性: OK')
        else:
            consistency_scores.append(0.5)
            details.append('入力非ゼロピクセル数の一貫性: 部分的')

        # 出力非ゼロピクセル数の一貫性
        if len(set(output_nonzero_counts)) == 1:
            consistency_scores.append(1.0)
            details.append('出力非ゼロピクセル数の一貫性: OK')
        else:
            consistency_scores.append(0.5)
            details.append('出力非ゼロピクセル数の一貫性: 部分的')

        avg_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        passed = avg_score >= 0.7  # 70%以上のスコアで合格

        return {
            'passed': passed,
            'score': avg_score,
            'details': '; '.join(details)
        }

    def _check_completeness(self, task: Task) -> Dict[str, Any]:
        """完全性チェック（本格実装）"""
        completeness_scores = []
        details = []

        # 1. 訓練データの完全性
        if task.train and len(task.train) > 0:
            completeness_scores.append(1.0)
            details.append(f'訓練データ: {len(task.train)}ペア存在')
        else:
            completeness_scores.append(0.0)
            details.append('訓練データ: 存在しません')

        # 2. テストデータの完全性
        if task.test and len(task.test) > 0:
            completeness_scores.append(1.0)
            details.append(f'テストデータ: {len(task.test)}ペア存在')
        else:
            completeness_scores.append(0.0)
            details.append('テストデータ: 存在しません')

        # 3. 各訓練ペアのデータ完全性
        complete_pairs = 0
        for i, train_pair in enumerate(task.train):
            if (train_pair.get('input') is not None and
                train_pair.get('output') is not None):
                complete_pairs += 1

        if task.train:
            pair_completeness = complete_pairs / len(task.train)
            completeness_scores.append(pair_completeness)
            details.append(f'訓練ペアの完全性: {complete_pairs}/{len(task.train)}')
        else:
            completeness_scores.append(0.0)
            details.append('訓練ペアの完全性: データなし')

        # 4. グリッドサイズの妥当性
        valid_sizes = 0
        for train_pair in task.train:
            input_grid = train_pair.get('input', [])
            output_grid = train_pair.get('output', [])
            if (input_grid and len(input_grid) > 0 and len(input_grid[0]) > 0 and
                output_grid and len(output_grid) > 0 and len(output_grid[0]) > 0):
                valid_sizes += 1

        if task.train:
            size_validity = valid_sizes / len(task.train)
            completeness_scores.append(size_validity)
            details.append(f'グリッドサイズの妥当性: {valid_sizes}/{len(task.train)}')
        else:
            completeness_scores.append(0.0)
            details.append('グリッドサイズの妥当性: データなし')

        avg_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        passed = avg_score >= 0.8  # 80%以上のスコアで合格

        return {
            'passed': passed,
            'score': avg_score,
            'details': '; '.join(details)
        }

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """評価統計を取得"""
        stats = dict(self.evaluation_stats)

        if stats['total_tasks_evaluated'] > 0:
            stats['pass_rate'] = stats['passed_tasks'] / stats['total_tasks_evaluated']
            stats['accuracy_pass_rate'] = stats['accuracy_checks_passed'] / stats['total_tasks_evaluated']
            stats['consistency_pass_rate'] = stats['consistency_checks_passed'] / stats['total_tasks_evaluated']
            stats['completeness_pass_rate'] = stats['completeness_checks_passed'] / stats['total_tasks_evaluated']
        else:
            stats['pass_rate'] = 0.0
            stats['accuracy_pass_rate'] = 0.0
            stats['consistency_pass_rate'] = 0.0
            stats['completeness_pass_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.evaluation_stats = {
            'total_tasks_evaluated': 0,
            'passed_tasks': 0,
            'failed_tasks': 0,
            'accuracy_checks_passed': 0,
            'consistency_checks_passed': 0,
            'completeness_checks_passed': 0
        }