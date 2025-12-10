"""
候補生成器（統合クラス）

各候補生成方法を別々のファイルに分割した後の統合クラス
"""

from typing import List, Dict, Any, Optional
import random
import torch
import numpy as np

from src.hybrid_system.inference.program_synthesis.complexity_regularizer import (
    ComplexityRegularizer,
    ComplexityConfig,
)

from .config import CandidateConfig
from .candidate_generators import (
    NeuralCandidateGenerator,
    NeuralObjectCandidateGenerator,
)
from .candidate_generators.common_helpers import (
    add_render_grid_if_needed,
    get_background_color,
    get_grid_size,
)

class CandidateGenerator:
    """候補プログラム生成器（統合クラス）"""

    def __init__(self, config: Optional[CandidateConfig] = None,
                 neural_model=None, tokenizer=None,
                 neural_object_model=None):
        """
        初期化

        Args:
            config: 設定
            neural_model: ProgramSynthesisModel（オプション）
            tokenizer: ProgramTokenizer（オプション）
            neural_object_model: ObjectBasedProgramSynthesisModel（オプション）
        """
        self.config = config or CandidateConfig()
        self.neural_model = neural_model
        self.tokenizer = tokenizer
        self.neural_object_model = neural_object_model

        # シード管理
        self.current_task_seed = self.config.task_seed
        self.current_pair_index = 0

        # 生成統計
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'neural_generations': 0,
            'neural_object_generations': 0,
            'seeds_used': 0
        }

        # 各生成器を初期化
        self.neural_generator = NeuralCandidateGenerator(
            neural_model,
            tokenizer,
            enable_ngps=self.config.enable_ngps,
            enable_dsl_selector=self.config.enable_dsl_selector,
            dsl_selector_weight=self.config.ngps_dsl_selector_weight,
            token_prob_weight=self.config.ngps_token_prob_weight,
            exploration_bonus_weight=self.config.ngps_exploration_bonus_weight,
            dsl_filter_threshold=self.config.ngps_dsl_filter_threshold
        )
        self.neural_object_generator = NeuralObjectCandidateGenerator(
            neural_object_model,
            tokenizer,
            enable_object_canonicalization=self.config.enable_object_canonicalization,
            enable_object_graph=self.config.enable_object_graph,
            enable_relation_classifier=self.config.enable_relation_classifier,
            graph_encoder_type=self.config.graph_encoder_type,
            relation_classifier_threshold=self.config.relation_classifier_threshold
        )

        # 複雑度正則化器
        self.complexity_regularizer = ComplexityRegularizer(ComplexityConfig())

    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        max_candidates: Optional[int] = None,
        pair_index: Optional[int] = None,
        partial_program: Optional[str] = None,
        matching_result: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """候補プログラムを生成

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            max_candidates: 最大候補数
            pair_index: ペアのインデックス（シード計算用）
            partial_program: 部分プログラム（オブジェクトマッチングから生成された場合）
            matching_result: オブジェクトマッチング結果

        Returns:
            候補プログラムのリスト
        """
        max_candidates = max_candidates or self.config.max_candidates
        self.generation_stats['total_generations'] += 1

        all_candidates = []

        # 部分プログラムがある場合、それを拡張して完全なプログラム候補を生成
        if partial_program:
            extended_program = self._extend_partial_program(
                partial_program, matching_result, input_grid, output_grid, pair_index
            )
            if extended_program:
                all_candidates.append(extended_program)
            # 元の部分プログラムも候補として追加
            all_candidates.append(partial_program)

        # シード変動が有効な場合、複数のシードで生成を試行
        num_seeds = self.config.num_seeds_per_pair if self.config.enable_seed_variation else 1

        for seed_offset in range(num_seeds):
            try:
                # シードを設定
                if self.config.enable_seed_variation and pair_index is not None:
                    current_seed = self._get_seed_for_pair(pair_index, seed_offset)
                    random.seed(current_seed)
                    np.random.seed(current_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(current_seed)
                    torch.manual_seed(current_seed)
                    self.generation_stats['seeds_used'] += 1

                candidates = []

                # 1. ニューラルモデルベースの生成（④深層学習ベース）
                if self.config.enable_neural_generation and self.neural_model is not None:
                    # 部分プログラムがある場合: 部分プログラムを使う候補と使わない候補の両方を生成
                    if partial_program:
                        # 部分プログラムを使う候補
                        neural_candidates_with_partial = self.neural_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_candidates_with_partial,
                            partial_program=partial_program
                        )
                        candidates.extend(neural_candidates_with_partial)
                        self.generation_stats['neural_generations'] += len(neural_candidates_with_partial)

                        # 部分プログラムを使わない候補
                        neural_candidates_without_partial = self.neural_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_candidates_without_partial,
                            partial_program=None
                        )
                        candidates.extend(neural_candidates_without_partial)
                        self.generation_stats['neural_generations'] += len(neural_candidates_without_partial)
                    else:
                        # 部分プログラムがない場合: 部分プログラムを使わない候補のみ
                        neural_candidates = self.neural_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_candidates_without_partial,
                            partial_program=None
                        )
                        candidates.extend(neural_candidates)
                        self.generation_stats['neural_generations'] += len(neural_candidates)

                # 2. オブジェクトベースのニューラルモデル生成（⑤深層学習ベース２）
                if self.config.enable_neural_object_generation and self.neural_object_model is not None:
                    # 部分プログラムがある場合: 部分プログラムを使う候補と使わない候補の両方を生成
                    if partial_program:
                        # 部分プログラムを使う候補
                        neural_object_candidates_with_partial = self.neural_object_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_object_candidates_with_partial,
                            partial_program=partial_program,
                            matching_result=matching_result
                        )
                        candidates.extend(neural_object_candidates_with_partial)
                        self.generation_stats['neural_object_generations'] = self.generation_stats.get('neural_object_generations', 0) + len(neural_object_candidates_with_partial)

                        # 部分プログラムを使わない候補
                        neural_object_candidates_without_partial = self.neural_object_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_object_candidates_without_partial,
                            partial_program=None,
                            matching_result=matching_result
                        )
                        candidates.extend(neural_object_candidates_without_partial)
                        self.generation_stats['neural_object_generations'] = self.generation_stats.get('neural_object_generations', 0) + len(neural_object_candidates_without_partial)
                    else:
                        # 部分プログラムがない場合: 部分プログラムを使わない候補のみ
                        neural_object_candidates = self.neural_object_generator.generate_candidates(
                            input_grid, output_grid,
                            beam_width=self.config.num_neural_object_candidates_without_partial,
                            partial_program=None,
                            matching_result=matching_result
                        )
                        candidates.extend(neural_object_candidates)
                        self.generation_stats['neural_object_generations'] = self.generation_stats.get('neural_object_generations', 0) + len(neural_object_candidates)

                all_candidates.extend(candidates)

            except Exception as e:
                print(f"候補生成エラー (seed offset {seed_offset}): {e}")
                continue

        # 重複を除去
        unique_candidates = list(set(all_candidates))

        # プログラム検証レイヤーによる事前フィルタリング（強化版）
        # 無効な候補を事前に除外して、実行時間を削減
        filtered_candidates = self._filter_invalid_candidates(unique_candidates, input_grid, output_grid)

        # 最大候補数に制限（複雑度によるランキングで選出）
        if len(filtered_candidates) > max_candidates:
            # 複雑度によるランキングを実行
            ranked_results = self.rank_candidates_by_complexity(filtered_candidates)
            # final_scoreの高い順にソート済みなので、上位max_candidates個を取得
            unique_candidates = [result['program'] for result in ranked_results[:max_candidates]]
        elif len(filtered_candidates) > 0:
            # 候補数がmax_candidates以下の場合でも、複雑度によるランキングを実行（順序を整えるため）
            ranked_results = self.rank_candidates_by_complexity(filtered_candidates)
            unique_candidates = [result['program'] for result in ranked_results]
        else:
            # フィルタリング後に候補がなくなった場合は、元の候補を返す
            unique_candidates = unique_candidates

        self.generation_stats['successful_generations'] += 1

        return unique_candidates

    def _filter_invalid_candidates(
        self,
        candidates: List[str],
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> List[str]:
        """無効な候補を事前にフィルタリング（プログラム検証レイヤーの強化）

        Args:
            candidates: 候補プログラムのリスト
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            フィルタリングされた候補プログラムのリスト
        """
        try:
            from src.hybrid_system.utils.validation.enhanced_program_validator import EnhancedProgramValidator

            validator = EnhancedProgramValidator()
            # グリッドサイズを設定（境界チェック用）
            if input_grid:
                grid_height = len(input_grid)
                grid_width = len(input_grid[0]) if input_grid else 0
                validator.grid_size = (grid_height, grid_width)

            filtered = []
            for program in candidates:
                # 基本的な構文チェックのみ（高速化のため）
                # 完全な検証は後で実行されるため、ここでは明らかに無効なもののみを除外
                validation_result = validator.validate_program(program, task=None)

                # エラーがない、または警告のみの場合は通過
                if validation_result.is_valid or len(validation_result.errors) == 0:
                    filtered.append(program)
                # エラーがある場合でも、検証ペナルティが低い場合は通過（後でランキングで処理）
                else:
                    # 拡張検証を実行してペナルティを計算
                    enhanced_result = validator.validate_program_enhanced(program, task=None)
                    # ペナルティが0.5未満の場合は通過（軽微な問題のみ）
                    if enhanced_result.validation_penalty < 0.5:
                        filtered.append(program)

            return filtered
        except Exception as e:
            # 検証に失敗した場合は、すべての候補を返す
            print(f"候補フィルタリングエラー: {e}")
            return candidates

    def rank_candidates_by_complexity(
        self,
        programs: List[str],
        base_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成された候補プログラムを、複雑さペナルティ込みのスコアでランキングする

        Args:
            programs: 候補プログラム文字列のリスト
            base_scores: データ適合度などのベーススコア（省略時は全て1.0）

        Returns:
            ComplexityRegularizer.score_candidates と同じ形式の結果リスト
        """
        if not programs:
            return []

        if base_scores is None:
            base_scores = [1.0 for _ in programs]

        # プログラム検証レイヤーのペナルティを計算（強化版）
        validation_penalties: List[float] = []
        try:
            from src.hybrid_system.utils.validation.enhanced_program_validator import EnhancedProgramValidator

            validator = EnhancedProgramValidator()
            for program in programs:
                validation_result = validator.validate_program_enhanced(program, task=None)
                validation_penalties.append(validation_result.validation_penalty)
        except Exception as e:
            # 検証に失敗した場合はペナルティなし
            validation_penalties = [0.0] * len(programs)

        # ComplexityRegularizerを使用してランキング
        ranked_results = self.complexity_regularizer.score_candidates(programs, base_scores)

        # 検証ペナルティを最終スコアに組み込む（強化版）
        validation_penalty_weight = 0.2  # 検証ペナルティの重み
        for idx, result in enumerate(ranked_results):
            validation_penalty = validation_penalties[idx] * validation_penalty_weight
            result['final_score'] = result['final_score'] - validation_penalty
            result['validation_penalty'] = validation_penalties[idx]

        # 最終スコアで再ソート
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)

        return ranked_results

    def get_generation_statistics(self) -> Dict[str, Any]:
        """生成統計を取得"""
        stats = dict(self.generation_stats)

        if stats['total_generations'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_generations']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'neural_generations': 0,
            'neural_object_generations': 0,
            'seeds_used': 0
        }

    def _get_seed_for_pair(self, pair_index: int, seed_offset: int) -> int:
        """ペア用のシードを計算"""
        base_seed = self.current_task_seed if self.current_task_seed is not None else 42
        return base_seed * 10000 + pair_index * 100 + seed_offset

    def _extend_partial_program(
        self,
        partial_program: str,
        matching_result: Optional[Dict[str, Any]],
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        pair_index: Optional[int] = None
    ) -> Optional[str]:
        """部分プログラムを拡張して完全なプログラムを生成"""
        if not partial_program:
            return None

        try:
            # マッチング結果から変換パターンを取得（個別パラメータから生成）
            transformation_patterns = []
            if matching_result and matching_result.get('success'):
                categories = matching_result.get('categories', [])
                for category in categories:
                    if category.representative_transformation_pattern:
                        pattern = category.representative_transformation_pattern
                        # 個別パラメータから変換パターンタイプを決定
                        pattern_types = []
                        if pattern.get('color_change', False):
                            pattern_types.append('color_change')
                        if pattern.get('position_change', False):
                            pattern_types.append('position_change')
                        if pattern.get('shape_change', False):
                            pattern_types.append('shape_change')

                        # 各変換パターンタイプに対してパターンを追加
                        for pattern_type in pattern_types:
                            transformation_patterns.append({
                                'type': pattern_type,
                                'count': category.total_objects  # カテゴリ内のオブジェクト数を使用
                            })

            # 変換操作を生成
            transformation_code = self._generate_transformation_code(
                transformation_patterns, input_grid, output_grid
            )

            # RENDER_GRIDを追加（キャッシュから取得）
            grid_info_cache = matching_result.get('grid_info_cache', {}) if matching_result else {}
            if pair_index is not None and pair_index in grid_info_cache:
                grid_info = grid_info_cache[pair_index]
                output_h, output_w = grid_info['output_size']
                bg_color = grid_info['output_bg']
            else:
                # フォールバック: 直接計算
                output_h, output_w = get_grid_size(output_grid)
                bg_color = get_background_color(output_grid)

            # 部分プログラムに変換操作とRENDER_GRIDを追加
            extended_program = f"{partial_program}\n"
            if transformation_code:
                extended_program += f"{transformation_code}\n"
            extended_program += f"RENDER_GRID(objects, {bg_color}, {output_w}, {output_h})"

            return extended_program

        except Exception as e:
            print(f"部分プログラム拡張エラー: {e}")
            return None

    def _generate_transformation_code(
        self,
        transformation_patterns: List[Dict[str, Any]],
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> str:
        """変換パターンから変換操作のコードを生成"""
        if not transformation_patterns:
            return ""

        # 最も頻繁な変換パターンを選択
        pattern_counts = {}
        for pattern in transformation_patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + pattern.get('count', 1)

        if not pattern_counts:
            return ""

        # 最も頻繁なパターンを選択
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]

        # パターンに応じて変換コードを生成
        if dominant_pattern == 'color_change':
            target_color = self._infer_target_color_from_output(output_grid)
            return (
                f"FOR i LEN(objects) DO\n"
                f"    objects[i] = SET_COLOR(objects[i], {target_color})\n"
                f"END"
            )
        elif dominant_pattern == 'position_change':
            return (
                "FOR i LEN(objects) DO\n"
                "    objects[i] = MOVE(objects[i], 1, 0)\n"
                "END"
            )
        elif dominant_pattern == 'shape_change':
            return (
                "FOR i LEN(objects) DO\n"
                "    objects[i] = SCALE(objects[i], 2)\n"
                "END"
            )
        else:
            return ""

    def _infer_target_color_from_output(self, output_grid: List[List[int]]) -> int:
        """出力グリッドから代表的な色を推定する"""
        from collections import Counter
        flat_colors = [c for row in output_grid for c in row if c != 0]
        if not flat_colors:
            return 1
        counter = Counter(flat_colors)
        most_common_color, _ = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))
        return int(most_common_color)
