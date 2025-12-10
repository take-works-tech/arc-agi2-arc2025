"""
タスクアセンブラー

DataPairをTaskに統合する機能を提供
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import random
import uuid

# プロジェクトルートをパスに追加
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(project_root)

from src.hybrid_system.core.data_structures import DataPair, Task
from learning.phase2_evaluator.generalization import GeneralizationEvaluator


@dataclass
class AssemblyConfig:
    """アセンブリ設定"""
    num_train: int = 3
    num_test: int = 1
    min_pairs_per_program: int = 4
    max_assembly_attempts_per_program: int = 10
    ensure_input_uniqueness: bool = True
    apply_manual_rules: bool = True
    enable_quality_check: bool = True


@dataclass
class AssemblyStatistics:
    """アセンブリ統計"""
    successful_assemblies: int = 0
    failed_assemblies: int = 0
    reasons_for_failure: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class TaskAssembler:
    """DataPairをARC形式のTaskに統合"""

    def __init__(self, seed: Optional[int] = None, config: Optional[AssemblyConfig] = None):
        """初期化"""
        self.rng = random.Random(seed)
        self.config = config or AssemblyConfig()
        self.generalization_evaluator = GeneralizationEvaluator()
        self.assembly_stats = AssemblyStatistics()

    def assemble_tasks_from_pairs(self, data_pairs: List[DataPair]) -> List[Task]:
        """DataPairをプログラムごとにグループ化してTaskに変換

        Args:
            data_pairs: DataPairのリスト

        Returns:
            Taskのリスト
        """
        self.assembly_stats = AssemblyStatistics()  # 統計をリセット

        # プログラムごとにグループ化
        pairs_by_program = defaultdict(list)
        for pair in data_pairs:
            pairs_by_program[pair.program].append(pair)

        tasks = []
        for program, pairs in pairs_by_program.items():
            if len(pairs) < self.config.min_pairs_per_program:
                self.assembly_stats.reasons_for_failure['not_enough_pairs_for_program'] += 1
                continue

            task = self._attempt_assemble_single_task(program, pairs)
            if task:
                tasks.append(task)
                self.assembly_stats.successful_assemblies += 1
            else:
                self.assembly_stats.failed_assemblies += 1

        return tasks

    def assemble_tasks_with_validation(
        self,
        data_pairs: List[DataPair],
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
        apply_manual_rules: Optional[bool] = None
    ) -> List[Task]:
        """手動作成ルールを適用してタスクを作成

        Args:
            data_pairs: DataPairのリスト
            num_train: 訓練ペア数
            num_test: テストペア数
            apply_manual_rules: 手動ルールを適用するか

        Returns:
            検証済みTaskのリスト
        """
        # 設定の更新
        if num_train is not None:
            self.config.num_train = num_train
        if num_test is not None:
            self.config.num_test = num_test
        if apply_manual_rules is not None:
            self.config.apply_manual_rules = apply_manual_rules

        # タスクをアセンブリ
        tasks = self.assemble_tasks_from_pairs(data_pairs)

        if not self.config.apply_manual_rules:
            return tasks

        # 手動ルールによる検証
        validated_tasks = []
        for task in tasks:
            if self._validate_task_against_manual_rules(task):
                validated_tasks.append(task)
            else:
                self.assembly_stats.reasons_for_failure['manual_rules_violation'] += 1

        return validated_tasks

    def _attempt_assemble_single_task(self, program: str, all_pairs_for_program: List[DataPair]) -> Optional[Task]:
        """単一プログラムから1つのタスクを組み立てる試行

        Args:
            program: プログラム
            all_pairs_for_program: そのプログラムのすべてのペア

        Returns:
            組み立てられたTask（失敗時はNone）
        """
        required_pairs = self.config.num_train + self.config.num_test

        if len(all_pairs_for_program) < required_pairs:
            self.assembly_stats.reasons_for_failure['not_enough_pairs_for_task_assembly'] += 1
            return None

        for attempt in range(self.config.max_assembly_attempts_per_program):
            # シャッフルして異なる組み合わせを試す
            self.rng.shuffle(all_pairs_for_program)
            selected = all_pairs_for_program[:required_pairs]

            train_pairs = selected[:self.config.num_train]
            test_pairs = selected[self.config.num_train:]

            # 入力の重複チェック
            if self.config.ensure_input_uniqueness:
                if not self._check_unique_inputs(train_pairs + test_pairs):
                    continue  # 重複があれば再試行

            # Taskを作成
            task = Task(
                train=[{'input': p.input, 'output': p.output} for p in train_pairs],
                test=[{'input': p.input, 'output': p.output} for p in test_pairs],
                program=program,
                metadata={
                    'num_train': len(train_pairs),
                    'num_test': len(test_pairs),
                    'source_pair_ids': [p.pair_id for p in selected],
                    'assembly_attempt': attempt + 1
                },
                task_id=str(uuid.uuid4())
            )

            return task

        # 最大試行回数に達した場合
        self.assembly_stats.reasons_for_failure['max_attempts_reached'] += 1
        return None

    def _check_unique_inputs(self, pairs: List[DataPair]) -> bool:
        """入力グリッドの重複チェック

        Args:
            pairs: チェックするペアのリスト

        Returns:
            重複がない場合True
        """
        seen = set()
        for pair in pairs:
            input_hash = hash(str(pair.input))
            if input_hash in seen:
                return False
            seen.add(input_hash)
        return True

    def _validate_task_against_manual_rules(self, task: Task) -> bool:
        """手動作成ルールに基づいてタスクを検証

        Args:
            task: 検証するタスク

        Returns:
            ルールに適合する場合True
        """
        # 1. テスト範囲の包含性
        test_range_result = self.generalization_evaluator._evaluate_test_range_inclusion(task)
        if not test_range_result['passed']:
            return False

        # 2. 変換パターンの一貫性
        pattern_consistency_result = self.generalization_evaluator._evaluate_pattern_consistency(task)
        if not pattern_consistency_result['passed']:
            return False

        # 3. 入力の重複なし（assemble_tasks_from_pairsで既にチェック済み）
        # 4. その他の手動ルール
        if not self._check_additional_manual_rules(task):
            return False

        return True

    def _check_additional_manual_rules(self, task: Task) -> bool:
        """追加の手動ルールをチェック

        Args:
            task: チェックするタスク

        Returns:
            ルールに適合する場合True
        """
        # 1. 色数の制限（最大8色）
        all_colors = task.get_unique_colors()
        if len(all_colors) > 8:
            return False

        # 2. グリッドサイズの制限（3x3以上、30x30以下）
        min_size, max_size = task.get_grid_size_range()
        if min_size[0] < 3 or min_size[1] < 3:
            return False
        if max_size[0] > 30 or max_size[1] > 30:
            return False

        # 3. プログラムの妥当性
        if not task.program or len(task.program.strip()) < 10:
            return False

        # 4. 訓練ペア数の制限（2以上）
        if len(task.train) < 2:
            return False

        # 5. テストペア数の制限（1以上）
        if len(task.test) < 1:
            return False

        # 6. 空のoutput（0ピクセル）のチェック
        if self._has_empty_output(task):
            self.assembly_stats.reasons_for_failure['empty_output'] += 1
            return False

        return True

    def _has_empty_output(self, task: Task) -> bool:
        """タスクに空のoutput（0ピクセル）があるかチェック

        Args:
            task: チェックするタスク

        Returns:
            空のoutputがある場合True
        """
        # trainペアをチェック
        for pair in task.train:
            if isinstance(pair, dict) and 'output' in pair:
                output = pair['output']
                if self._is_empty_output(output):
                    return True

        # testペアをチェック
        for pair in task.test:
            if isinstance(pair, dict) and 'output' in pair:
                output = pair['output']
                if self._is_empty_output(output):
                    return True

        return False

    def _is_empty_output(self, output: Any) -> bool:
        """outputが空（0ピクセル）かどうかを判定

        Args:
            output: 出力グリッド（リストのリストまたはリスト）

        Returns:
            空の場合はTrue
        """
        if not output:
            return True

        if not isinstance(output, list):
            return False

        # 空配列の場合
        if len(output) == 0:
            return True

        # すべての行が空配列の場合
        if all(isinstance(row, list) and len(row) == 0 for row in output):
            return True

        # 総ピクセル数を計算
        total_pixels = 0
        for row in output:
            if isinstance(row, list):
                total_pixels += len(row)
            else:
                # 行がリストでない場合は無効
                return False

        return total_pixels == 0

    def get_assembly_statistics(self) -> Dict[str, Any]:
        """アセンブリ統計を取得

        Returns:
            統計情報
        """
        stats = {
            'successful_assemblies': self.assembly_stats.successful_assemblies,
            'failed_assemblies': self.assembly_stats.failed_assemblies,
            'reasons_for_failure': dict(self.assembly_stats.reasons_for_failure),
            'total_attempts': self.assembly_stats.successful_assemblies + self.assembly_stats.failed_assemblies
        }

        if stats['total_attempts'] > 0:
            stats['success_rate'] = self.assembly_stats.successful_assemblies / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.assembly_stats = AssemblyStatistics()
