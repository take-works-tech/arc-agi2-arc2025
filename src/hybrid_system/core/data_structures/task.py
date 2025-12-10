"""
Task実装

フェーズ2で使用するARC形式のタスク（複数ペア + 共通プログラム）を定義
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
from datetime import datetime

from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import (
    get_color_distribution as get_single_grid_color_distribution
)


@dataclass
class Task:
    """
    ARC形式のタスク（複数ペア + 共通プログラム）

    フェーズ2で生成される最終形式。
    評価時に汎化能力をテストするために使用。
    """
    train: List[Dict[str, List[List[int]]]]  # [{'input': ..., 'output': ...}, ...]
    test: List[Dict[str, List[List[int]]]]
    program: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'task_id': self.task_id,
            'train': self.train,
            'test': self.test,
            'program': self.program,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """辞書から復元"""
        return cls(
            train=data['train'],
            test=data['test'],
            program=data.get('program', ''),
            metadata=data.get('metadata', {}),
            task_id=data.get('task_id', str(uuid.uuid4())),
            created_at=data.get('created_at', datetime.now().isoformat())
        )

    def to_arc_format(self) -> Dict[str, Any]:
        """ARC-AGI2互換形式に変換（program情報は除外）"""
        return {
            'train': self.train,
            'test': self.test
        }

    def get_num_train_pairs(self) -> int:
        """訓練ペア数を取得"""
        return len(self.train)

    def get_num_test_pairs(self) -> int:
        """テストペア数を取得"""
        return len(self.test)

    def get_all_inputs(self) -> List[List[List[int]]]:
        """すべての入力グリッドを取得"""
        all_inputs = []
        for pair in self.train:
            all_inputs.append(pair['input'])
        for pair in self.test:
            all_inputs.append(pair['input'])
        return all_inputs

    def get_all_outputs(self) -> List[List[List[int]]]:
        """すべての出力グリッドを取得"""
        all_outputs = []
        for pair in self.train:
            all_outputs.append(pair['output'])
        for pair in self.test:
            all_outputs.append(pair['output'])
        return all_outputs

    def get_train_inputs(self) -> List[List[List[int]]]:
        """訓練入力グリッドを取得"""
        return [pair['input'] for pair in self.train]

    def get_train_outputs(self) -> List[List[List[int]]]:
        """訓練出力グリッドを取得"""
        return [pair['output'] for pair in self.train]

    def get_test_inputs(self) -> List[List[List[int]]]:
        """テスト入力グリッドを取得"""
        return [pair['input'] for pair in self.test]

    def get_test_outputs(self) -> List[List[List[int]]]:
        """テスト出力グリッドを取得"""
        return [pair['output'] for pair in self.test]

    def get_color_distribution(self) -> Dict[int, int]:
        """色の分布を取得（共通実装を使用）"""
        color_count = {}

        # 入力グリッドの色（共通実装を使用）
        for input_grid in self.get_all_inputs():
            input_dist = get_single_grid_color_distribution(input_grid)
            for color, count in input_dist.items():
                color_count[color] = color_count.get(color, 0) + count

        # 出力グリッドの色（共通実装を使用）
        for output_grid in self.get_all_outputs():
            output_dist = get_single_grid_color_distribution(output_grid)
            for color, count in output_dist.items():
                color_count[color] = color_count.get(color, 0) + count

        return color_count

    def get_unique_colors(self) -> set:
        """使用されている色の集合を取得"""
        colors = set()

        for input_grid in self.get_all_inputs():
            for row in input_grid:
                colors.update(row)

        for output_grid in self.get_all_outputs():
            for row in output_grid:
                colors.update(row)

        return colors

    def get_grid_size_range(self) -> tuple:
        """グリッドサイズの範囲を取得"""
        all_inputs = self.get_all_inputs()
        if not all_inputs:
            return (0, 0), (0, 0)

        sizes = [(len(grid), len(grid[0]) if grid else 0) for grid in all_inputs]
        min_size = min(sizes)
        max_size = max(sizes)

        return min_size, max_size

    def validate_consistency(self) -> bool:
        """タスクの一貫性を検証"""
        # 基本的な検証
        if not self.train or not self.test:
            return False

        if not self.program:
            return False

        # 入力グリッドの重複チェック
        all_inputs = self.get_all_inputs()
        seen_inputs = set()
        for input_grid in all_inputs:
            input_hash = hash(str(input_grid))
            if input_hash in seen_inputs:
                return False
            seen_inputs.add(input_hash)

        return True

    def get_program_complexity(self) -> int:
        """プログラムの複雑度を推定"""
        if not self.program:
            return 0

        complexity = 0
        complexity += self.program.count('FOR ')
        complexity += self.program.count('IF ')
        complexity += self.program.count('WHILE ')
        complexity += self.program.count('FILTER')
        complexity += self.program.count('GET_')

        return complexity

    def is_valid(self) -> bool:
        """基本的な妥当性チェック"""
        if not self.train or not self.test:
            return False

        if not self.program:
            return False

        return self.validate_consistency()

    def __str__(self) -> str:
        return f"Task(id={self.task_id[:8]}, train={len(self.train)}, test={len(self.test)})"

    def __repr__(self) -> str:
        return self.__str__()
