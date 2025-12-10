"""
複雑度正則化器

プログラムの複雑度を評価し、正則化する機能
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from collections import defaultdict

from src.hybrid_system.inference.program_synthesis.command_complexity_config import (
    build_command_complexity_weights,
)


@dataclass
class ComplexityConfig:
    """複雑度設定"""
    max_complexity: int = 20
    complexity_weights: Dict[str, float] = None
    enable_penalty: bool = True
    penalty_factor: float = 0.1

    def __post_init__(self):
        if self.complexity_weights is None:
            self.complexity_weights = {
                'control_structures': 2.0,
                'function_calls': 1.0,
                'variables': 0.5,
                'nested_depth': 3.0,
                'program_length': 0.1
            }


class ComplexityRegularizer:
    """複雑度正則化器"""

    def __init__(self, config: Optional[ComplexityConfig] = None):
        """初期化"""
        self.config = config or ComplexityConfig()

        # コマンドごとの複雑度重み（人間の直感的な「説明コスト」に近づけるため）
        # カテゴリごとのベース重みから、全コマンドに対して重みを付与する
        self.command_complexity_weights: Dict[str, float] = self._build_command_complexity_weights()

        # 複雑度統計
        self.complexity_stats = {
            'total_evaluations': 0,
            'average_complexity': 0.0,
            'complexity_distribution': defaultdict(int),
            'penalty_applied_count': 0
        }

    def calculate_complexity_score(self, program: str) -> float:
        """プログラムの複雑度スコアを計算

        Args:
            program: 評価するプログラム

        Returns:
            複雑度スコア（0.0-1.0、高いほど複雑）
        """
        self.complexity_stats['total_evaluations'] += 1

        if not program:
            return 0.0

        # 各複雑度要素を計算
        complexity_components = self._calculate_complexity_components(program)

        # 重み付き複雑度スコアを計算
        weighted_score = 0.0
        for component, value in complexity_components.items():
            weight = self.config.complexity_weights.get(component, 1.0)
            weighted_score += value * weight

        # 正規化（0.0-1.0の範囲に）
        normalized_score = min(weighted_score / self.config.max_complexity, 1.0)

        # 統計更新
        self.complexity_stats['average_complexity'] = (
            (self.complexity_stats['average_complexity'] * (self.complexity_stats['total_evaluations'] - 1) + normalized_score) /
            self.complexity_stats['total_evaluations']
        )

        # 複雑度分布を更新
        complexity_level = self._get_complexity_level(normalized_score)
        self.complexity_stats['complexity_distribution'][complexity_level] += 1

        return normalized_score

    def _calculate_complexity_components(self, program: str) -> Dict[str, float]:
        """複雑度要素を計算

        Args:
            program: 評価するプログラム

        Returns:
            複雑度要素の辞書
        """
        components = {}

        # 1. 制御構造の複雑度
        components['control_structures'] = self._calculate_control_structure_complexity(program)

        # 2. 関数呼び出しの複雑度
        components['function_calls'] = self._calculate_function_call_complexity(program)

        # 3. 変数の複雑度
        components['variables'] = self._calculate_variable_complexity(program)

        # 4. ネスト深度の複雑度
        components['nested_depth'] = self._calculate_nested_depth_complexity(program)

        # 5. プログラム長の複雑度
        components['program_length'] = self._calculate_program_length_complexity(program)

        return components

    def _calculate_control_structure_complexity(self, program: str) -> float:
        """制御構造の複雑度を計算"""
        complexity = 0.0

        # FOR文の複雑度
        for_count = program.count('FOR ')
        complexity += for_count * 2.0

        # IF文の複雑度
        if_count = program.count('IF ')
        complexity += if_count * 1.5

        # WHILE文の複雑度
        while_count = program.count('WHILE ')
        complexity += while_count * 3.0

        # ネストした制御構造の複雑度
        nested_complexity = self._calculate_nested_control_complexity(program)
        complexity += nested_complexity

        return complexity

    def _calculate_function_call_complexity(self, program: str) -> float:
        """関数呼び出しの複雑度を計算"""
        complexity = 0.0

        # 関数呼び出しのパターンを検索し、「コマンドごとの重み」を用いて複雑度を積算する
        # 例: MOVE(...), FILTER(...), SET_COLOR(...) など
        command_pattern = re.compile(r'\b([A-Z_]+)\s*\(')
        for match in command_pattern.finditer(program):
            command_name = match.group(1)
            weight = self.command_complexity_weights.get(command_name, 1.0)
            complexity += weight

        return complexity

    def _calculate_variable_complexity(self, program: str) -> float:
        """変数の複雑度を計算"""
        # 変数の数をカウント
        variables = re.findall(r'\$obj\d+', program)
        unique_variables = len(set(variables))

        return unique_variables * 0.5

    def _calculate_nested_depth_complexity(self, program: str) -> float:
        """ネスト深度の複雑度を計算"""
        max_depth = 0
        current_depth = 0

        lines = program.split('\n')
        for line in lines:
            stripped = line.strip()

            # 制御構造の開始
            if stripped.startswith('FOR ') or stripped.startswith('IF ') or stripped.startswith('WHILE '):
                current_depth += 1
                max_depth = max(max_depth, current_depth)

            # 制御構造の終了
            elif stripped.startswith('END') or stripped.startswith('}'):
                current_depth = max(0, current_depth - 1)

        return max_depth * 3.0

    def _calculate_program_length_complexity(self, program: str) -> float:
        """プログラム長の複雑度を計算"""
        # 行数と文字数で複雑度を計算
        lines = program.split('\n')
        line_count = len([line for line in lines if line.strip()])
        char_count = len(program)

        # 行数と文字数の重み付き平均
        length_complexity = (line_count * 0.5 + char_count * 0.01) * 0.1

        return length_complexity

    def _calculate_nested_control_complexity(self, program: str) -> float:
        """ネストした制御構造の複雑度を計算"""
        complexity = 0.0
        current_depth = 0

        lines = program.split('\n')
        for line in lines:
            stripped = line.strip()

            # 制御構造の開始
            if stripped.startswith('FOR ') or stripped.startswith('IF ') or stripped.startswith('WHILE '):
                current_depth += 1
                # ネストが深いほど複雑度が高くなる
                complexity += current_depth * 0.5

            # 制御構造の終了
            elif stripped.startswith('END') or stripped.startswith('}'):
                current_depth = max(0, current_depth - 1)

        return complexity

    def _build_command_complexity_weights(self) -> Dict[str, float]:
        """
        コマンドごとの複雑度重みテーブルを構築

        NOTE:
            - すべてのDSLコマンドに対して何らかの重みを持つ
            - カテゴリごとの「認知的コスト」と、一部コマンドの個別補正を反映する
        """
        return build_command_complexity_weights()

    def _get_complexity_level(self, complexity_score: float) -> str:
        """複雑度レベルを取得"""
        if complexity_score < 0.2:
            return 'low'
        elif complexity_score < 0.5:
            return 'medium'
        elif complexity_score < 0.8:
            return 'high'
        else:
            return 'very_high'

    def apply_complexity_penalty(self, program: str, base_score: float) -> float:
        """複雑度ペナルティを適用

        Args:
            program: 評価するプログラム
            base_score: ベーススコア

        Returns:
            ペナルティ適用後のスコア
        """
        if not self.config.enable_penalty:
            return base_score

        complexity_score = self.calculate_complexity_score(program)

        # 複雑度が高いほどペナルティが大きくなる
        penalty = complexity_score * self.config.penalty_factor

        # ペナルティを適用
        penalized_score = base_score - penalty

        # スコアが負にならないようにする
        final_score = max(penalized_score, 0.0)

        if penalty > 0:
            self.complexity_stats['penalty_applied_count'] += 1

        return final_score

    def get_complexity_analysis(self, program: str) -> Dict[str, Any]:
        """プログラムの複雑度分析を取得

        Args:
            program: 分析するプログラム

        Returns:
            複雑度分析結果
        """
        if not program:
            return {'error': 'Empty program'}

        # 複雑度要素を計算
        components = self._calculate_complexity_components(program)

        # 総合複雑度スコアを計算
        total_score = self.calculate_complexity_score(program)

        # 複雑度レベルを取得
        complexity_level = self._get_complexity_level(total_score)

        # 推奨事項を生成
        recommendations = self._generate_recommendations(components, total_score)

        return {
            'total_complexity_score': total_score,
            'complexity_level': complexity_level,
            'components': components,
            'recommendations': recommendations,
            'program_length': len(program),
            'line_count': len([line for line in program.split('\n') if line.strip()])
        }

    def score_candidates(
        self,
        programs: List[str],
        base_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        複数の候補プログラムに対して、複雑度ペナルティ込みのスコアを計算しソートして返す

        Args:
            programs: 候補プログラム文字列のリスト
            base_scores: データ適合度などのベーススコア（高いほど良い）

        Returns:
            各候補について以下を含む辞書のリスト（final_score 降順ソート済み）
                - index: 元の候補インデックス
                - program: プログラム文字列
                - base_score: ベーススコア
                - complexity_score: 0.0-1.0 の複雑度スコア
                - final_score: 複雑度ペナルティ適用後のスコア
        """
        results: List[Dict[str, Any]] = []
        for idx, (prog, base) in enumerate(zip(programs, base_scores)):
            if not prog:
                # 空プログラムはスコア0扱い
                results.append({
                    'index': idx,
                    'program': prog,
                    'base_score': base,
                    'complexity_score': 0.0,
                    'final_score': 0.0,
                })
                continue

            complexity_score = self.calculate_complexity_score(prog)
            penalty = complexity_score * self.config.penalty_factor if self.config.enable_penalty else 0.0
            penalized = max(base - penalty, 0.0)

            if penalty > 0:
                self.complexity_stats['penalty_applied_count'] += 1

            results.append({
                'index': idx,
                'program': prog,
                'base_score': base,
                'complexity_score': complexity_score,
                'final_score': penalized,
            })

        # final_score の高い順にソート（マルチ仮説候補のランキングに利用）
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results

    def _generate_recommendations(self, components: Dict[str, float], total_score: float) -> List[str]:
        """複雑度改善の推奨事項を生成

        Args:
            components: 複雑度要素
            total_score: 総合複雑度スコア

        Returns:
            推奨事項のリスト
        """
        recommendations = []

        # 制御構造の推奨事項
        if components['control_structures'] > 5.0:
            recommendations.append("制御構造が多すぎます。ロジックを簡素化することを検討してください。")

        # 関数呼び出しの推奨事項
        if components['function_calls'] > 10.0:
            recommendations.append("関数呼び出しが多すぎます。共通処理をまとめることを検討してください。")

        # ネスト深度の推奨事項
        if components['nested_depth'] > 6.0:
            recommendations.append("ネストが深すぎます。関数に分割することを検討してください。")

        # プログラム長の推奨事項
        if components['program_length'] > 2.0:
            recommendations.append("プログラムが長すぎます。機能を分割することを検討してください。")

        # 総合的な推奨事項
        if total_score > 0.8:
            recommendations.append("全体的に複雑度が高すぎます。プログラムの構造を見直すことを推奨します。")
        elif total_score < 0.2:
            recommendations.append("複雑度が低く、シンプルなプログラムです。")

        return recommendations

    def get_complexity_statistics(self) -> Dict[str, Any]:
        """複雑度統計を取得"""
        stats = dict(self.complexity_stats)

        # 複雑度分布を通常の辞書に変換
        stats['complexity_distribution'] = dict(stats['complexity_distribution'])

        if stats['total_evaluations'] > 0:
            stats['penalty_application_rate'] = stats['penalty_applied_count'] / stats['total_evaluations']
        else:
            stats['penalty_application_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.complexity_stats = {
            'total_evaluations': 0,
            'average_complexity': 0.0,
            'complexity_distribution': defaultdict(int),
            'penalty_applied_count': 0
        }
