"""
プログラム生成器

DSLプログラムの生成機能を提供
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# 既存のプログラム生成器をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_systems.generator.program_generator import UnifiedProgramGenerator


@dataclass
class GenerationConfig:
    """プログラム生成設定"""
    complexity_distribution: Dict[int, float] = None
    command_constraints: Dict[str, Any] = None
    ensure_diversity: bool = True
    avoid_similar_programs: bool = True
    max_attempts: int = 10
    seed: Optional[int] = None

    def __post_init__(self):
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                1: 0.4,  # 40% 簡単
                2: 0.3,  # 30% 中程度
                3: 0.2,  # 20% 複雑
                4: 0.1   # 10% 非常に複雑
            }


class ProgramGenerator:
    """プログラム生成器"""

    def __init__(self, config: Optional[GenerationConfig] = None):
        """初期化"""
        self.config = config or GenerationConfig()
        self.rng = random.Random(self.config.seed)

        # 新システム（v2.0）を使用
        self.complete_generator = UnifiedProgramGenerator()
        self._use_legacy = False  # 新システムを使用

        # 生成統計
        self.generation_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'complexity_distribution': {1: 0, 2: 0, 3: 0, 4: 0}
        }

    def generate_program(self, complexity: Optional[int] = None) -> Optional[str]:
        """単一のプログラムを生成

        Args:
            complexity: 複雑度（Noneの場合はランダム）

        Returns:
            生成されたプログラム（失敗時はNone）
        """
        try:
            if complexity is None:
                # 複雑度分布に基づいてランダム選択
                complexity = self._select_complexity()

            program = self.complete_generator.generate_program(complexity=complexity)

            if program and self._validate_program(program):
                self.generation_stats['successful_generations'] += 1
                self.generation_stats['complexity_distribution'][complexity] += 1
                return program
            else:
                self.generation_stats['failed_generations'] += 1
                return None

        except Exception as e:
            print(f"プログラム生成エラー: {e}")
            self.generation_stats['failed_generations'] += 1
            return None
        finally:
            self.generation_stats['total_generated'] += 1

    def generate_programs_batch(
        self,
        num_programs: int,
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """複数のプログラムをバッチ生成

        Args:
            num_programs: 生成するプログラム数
            config: 生成設定（Noneの場合はデフォルト設定を使用）

        Returns:
            生成されたプログラムのリスト
        """
        if config:
            self.config = config

        programs = []
        seen_programs = set()

        # 複雑度ごとの生成数
        complexity_counts = self._calculate_complexity_counts(num_programs)

        for complexity, count in complexity_counts.items():
            if count > 0:
                print(f"複雑度 {complexity} のプログラム {count} 個を生成中...")

                for _ in range(count):
                    attempts = 0
                    while attempts < self.config.max_attempts:
                        program = self.generate_program(complexity)

                        if program and program not in seen_programs:
                            programs.append(program)
                            seen_programs.add(program)
                            break

                        attempts += 1

        print(f"プログラム生成完了: {len(programs)}/{num_programs} プログラム生成")
        return programs

    def generate_diverse_programs(
        self,
        num_programs: int,
        diversity_threshold: float = 0.8
    ) -> List[str]:
        """多様性を重視したプログラム生成

        Args:
            num_programs: 生成するプログラム数
            diversity_threshold: 多様性の閾値

        Returns:
            多様性のあるプログラムのリスト
        """
        programs = []
        seen_programs = set()

        while len(programs) < num_programs:
            program = self.generate_program()

            if program and self._check_diversity(program, seen_programs, diversity_threshold):
                programs.append(program)
                seen_programs.add(program)

        return programs

    def _select_complexity(self) -> int:
        """複雑度分布に基づいて複雑度を選択"""
        rand = self.rng.random()
        cumulative = 0.0

        for complexity, probability in self.config.complexity_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return complexity

        return 1  # デフォルト

    def _calculate_complexity_counts(self, num_programs: int) -> Dict[int, int]:
        """複雑度ごとの生成数を計算"""
        counts = {}
        for complexity, probability in self.config.complexity_distribution.items():
            counts[complexity] = int(num_programs * probability)

        # 不足分を調整
        total_allocated = sum(counts.values())
        if total_allocated < num_programs:
            # 最も確率の高い複雑度に不足分を追加
            max_complexity = max(self.config.complexity_distribution.keys(),
                               key=lambda k: self.config.complexity_distribution[k])
            counts[max_complexity] += (num_programs - total_allocated)

        return counts

    def _validate_program(self, program: str) -> bool:
        """プログラムの妥当性をチェック"""
        if not program or len(program.strip()) < 10:
            return False

        # 基本的な構文チェック
        if not any(cmd in program for cmd in ['GET_', 'SET_', 'FOR ', 'IF ']):
            return False

        # 制約チェック
        if self.config.command_constraints:
            for constraint, value in self.config.command_constraints.items():
                if not self._check_constraint(program, constraint, value):
                    return False

        return True

    def _check_constraint(self, program: str, constraint: str, value: Any) -> bool:
        """制約をチェック"""
        if constraint == 'max_length':
            return len(program) <= value
        elif constraint == 'min_length':
            return len(program) >= value
        elif constraint == 'max_commands':
            command_count = sum(program.count(cmd) for cmd in ['GET_', 'SET_', 'FOR ', 'IF '])
            return command_count <= value
        elif constraint == 'required_commands':
            return all(cmd in program for cmd in value)
        elif constraint == 'forbidden_commands':
            return not any(cmd in program for cmd in value)

        return True

    def _check_diversity(self, program: str, seen_programs: set, threshold: float) -> bool:
        """多様性をチェック"""
        if not seen_programs:
            return True

        # 類似性チェック（本格実装）
        # 複数の類似度指標を使用: Jaccard類似度、編集距離、構造的類似度
        import re

        # プログラムをトークン化（より詳細な分割）
        def tokenize_program(prog):
            # コマンド、変数、数値、括弧などを適切に分割
            tokens = re.findall(r'[A-Z_]+|\(|\)|\[|\]|,|;|\d+|\$[a-zA-Z0-9_]+|[a-zA-Z_][a-zA-Z0-9_]*', prog)
            return set(tokens)

        program_tokens = tokenize_program(program)

        for seen_program in list(seen_programs)[-50:]:  # 最近の50個のみチェック
            seen_tokens = tokenize_program(seen_program)

            # Jaccard類似度
            intersection = len(program_tokens & seen_tokens)
            union = len(program_tokens | seen_tokens)

            if union > 0:
                jaccard_similarity = intersection / union

                # 編集距離も考慮（短いプログラムの場合はより厳密に）
                if len(program) < 100 or len(seen_program) < 100:
                    # 短いプログラムの場合は、文字列の類似度も考慮
                    from difflib import SequenceMatcher
                    char_similarity = SequenceMatcher(None, program, seen_program).ratio()
                    # 両方の類似度を組み合わせ
                    combined_similarity = (jaccard_similarity * 0.7 + char_similarity * 0.3)
                else:
                    combined_similarity = jaccard_similarity

                if combined_similarity > threshold:
                    return False

        return True

    def is_available(self) -> bool:
        """生成器が利用可能かチェック"""
        return self._use_legacy

    def get_generation_statistics(self) -> Dict[str, Any]:
        """生成統計を取得"""
        stats = dict(self.generation_stats)

        if stats['total_generated'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_generated']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.generation_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'complexity_distribution': {1: 0, 2: 0, 3: 0, 4: 0}
        }
