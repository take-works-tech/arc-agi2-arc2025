"""
DataPair実装

フェーズ1で使用する単一の入出力ペア（プログラム付き）を定義
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
from datetime import datetime

from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import (
    get_color_distribution as get_single_grid_color_distribution
)


@dataclass
class DataPair:
    """
    単一の入出力ペア（プログラム付き）

    フェーズ1で生成される基本単位。
    訓練時に「Input, Output → Program」の学習に使用。
    """
    input: List[List[int]]
    output: List[List[int]]
    program: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pair_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'pair_id': self.pair_id,
            'input': self.input,
            'output': self.output,
            'program': self.program,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPair':
        """辞書から復元"""
        return cls(
            input=data['input'],
            output=data['output'],
            program=data['program'],
            metadata=data.get('metadata', {}),
            pair_id=data.get('pair_id', str(uuid.uuid4())),
            created_at=data.get('created_at', datetime.now().isoformat())
        )

    def get_grid_size(self) -> tuple:
        """グリッドサイズを取得"""
        if not self.input:
            return (0, 0)
        return (len(self.input), len(self.input[0]) if self.input else 0)

    def get_program_complexity(self) -> int:
        """プログラムの複雑度を推定"""
        if not self.program:
            return 0

        # 簡単な複雑度計算
        complexity = 0
        complexity += self.program.count('FOR ')
        complexity += self.program.count('IF ')
        complexity += self.program.count('WHILE ')
        complexity += self.program.count('FILTER')
        complexity += self.program.count('GET_')

        return complexity

    def get_color_distribution(self) -> Dict[int, int]:
        """色の分布を取得（共通実装を使用）"""
        color_count = {}

        # 入力グリッドの色（共通実装を使用）
        input_dist = get_single_grid_color_distribution(self.input)
        for color, count in input_dist.items():
            color_count[color] = color_count.get(color, 0) + count

        # 出力グリッドの色（共通実装を使用）
        output_dist = get_single_grid_color_distribution(self.output)
        for color, count in output_dist.items():
            color_count[color] = color_count.get(color, 0) + count

        return color_count

    def get_unique_colors(self) -> set:
        """使用されている色の集合を取得"""
        colors = set()

        for row in self.input + self.output:
            colors.update(row)

        return colors

    def is_valid(self) -> bool:
        """基本的な妥当性チェック"""
        if not self.input or not self.output:
            return False

        if not self.program:
            return False

        # ARC-AGI2では入出力のサイズが異なる場合があるため、サイズチェックは除外
        # 基本的な構造のチェックのみ行う
        if len(self.input) == 0 or len(self.input[0]) == 0:
            return False

        if len(self.output) == 0 or len(self.output[0]) == 0:
            return False

        return True

    def __str__(self) -> str:
        return f"DataPair(id={self.pair_id[:8]}, program={self.program[:50]}...)"

    def __repr__(self) -> str:
        return self.__str__()
