"""
ARCソルバー

ARC問題を解決するメインソルバー
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time

from core.data_structures import Task
# 循環インポートを回避するため、遅延インポートを使用
# from src.hybrid_system.inference.program_synthesis import ProgramSynthesisEngine
from .pattern_recognizer import PatternRecognizer


@dataclass
class SolverConfig:
    """ソルバー設定"""
    enable_program_synthesis: bool = True
    enable_pattern_recognition: bool = True
    enable_object_matching: bool = False  # 従来のオブジェクトマッチングは削除済み
    max_solving_time: float = 60.0
    confidence_threshold: float = 0.7


class ARCSolver:
    """ARCソルバー"""

    def __init__(self, config: Optional[SolverConfig] = None):
        """初期化"""
        self.config = config or SolverConfig()

        # コンポーネントの初期化（循環インポートを回避するため、遅延初期化）
        self.program_synthesis_engine = None
        if self.config.enable_program_synthesis:
            # 遅延インポート
            from src.hybrid_system.inference.program_synthesis import ProgramSynthesisEngine
            self.program_synthesis_engine = ProgramSynthesisEngine()
        self.pattern_recognizer = PatternRecognizer() if self.config.enable_pattern_recognition else None
        self.object_matcher = None  # 従来のオブジェクトマッチングは削除済み

        # ソルバー統計
        self.solver_stats = {
            'total_solving_attempts': 0,
            'successful_solutions': 0,
            'failed_solutions': 0,
            'average_solving_time': 0.0,
            'method_usage': {
                'program_synthesis': 0,
                'pattern_recognition': 0,
                'object_matching': 0
            }
        }

    def solve_task(self, task: Task) -> Dict[str, Any]:
        """タスクを解決

        Args:
            task: 解決するタスク

        Returns:
            解決結果
        """
        start_time = time.time()
        self.solver_stats['total_solving_attempts'] += 1

        try:
            # 複数の解決方法を試す
            solutions = []

            # 1. プログラム合成による解決
            if self.config.enable_program_synthesis and self.program_synthesis_engine:
                print("プログラム合成による解決を試行中...")
                synthesis_result = self._solve_with_program_synthesis(task)
                if synthesis_result['success']:
                    solutions.append(synthesis_result)
                    self.solver_stats['method_usage']['program_synthesis'] += 1

            # 2. パターン認識による解決
            if self.config.enable_pattern_recognition and self.pattern_recognizer:
                print("パターン認識による解決を試行中...")
                pattern_result = self._solve_with_pattern_recognition(task)
                if pattern_result['success']:
                    solutions.append(pattern_result)
                    self.solver_stats['method_usage']['pattern_recognition'] += 1

            # 3. オブジェクトマッチングによる解決（削除済み）

            # 最適な解決策を選択
            if solutions:
                best_solution = self._select_best_solution(solutions)

                solving_time = time.time() - start_time
                self.solver_stats['successful_solutions'] += 1
                self.solver_stats['average_solving_time'] = (
                    (self.solver_stats['average_solving_time'] * (self.solver_stats['successful_solutions'] - 1) + solving_time) /
                    self.solver_stats['successful_solutions']
                )

                return {
                    'success': True,
                    'solution': best_solution,
                    'solving_time': solving_time,
                    'all_solutions': solutions
                }
            else:
                print("すべての解決方法が失敗しました")
                self.solver_stats['failed_solutions'] += 1
                return {
                    'success': False,
                    'solution': None,
                    'solving_time': time.time() - start_time,
                    'all_solutions': []
                }

        except Exception as e:
            print(f"タスク解決エラー: {e}")
            self.solver_stats['failed_solutions'] += 1
            return {
                'success': False,
                'solution': None,
                'solving_time': time.time() - start_time,
                'error': str(e)
            }

    def _solve_with_program_synthesis(self, task: Task) -> Dict[str, Any]:
        """プログラム合成による解決"""
        try:
            # プログラムを合成
            synthesized_program = self.program_synthesis_engine.synthesize_program(task)

            if not synthesized_program:
                return {'success': False, 'method': 'program_synthesis'}

            # テスト入力にプログラムを適用
            test_input = task.test[0]['input']
            predicted_output = self._execute_program(synthesized_program, test_input)

            if predicted_output is None:
                return {'success': False, 'method': 'program_synthesis'}

            # 信頼度を計算
            confidence = self._calculate_confidence(synthesized_program, task)

            return {
                'success': True,
                'method': 'program_synthesis',
                'program': synthesized_program,
                'predicted_output': predicted_output,
                'confidence': confidence
            }

        except Exception as e:
            print(f"プログラム合成エラー: {e}")
            return {'success': False, 'method': 'program_synthesis', 'error': str(e)}

    def _solve_with_pattern_recognition(self, task: Task) -> Dict[str, Any]:
        """パターン認識による解決"""
        try:
            # パターンを認識
            patterns = self.pattern_recognizer.recognize_patterns(task)

            if not patterns:
                return {'success': False, 'method': 'pattern_recognition'}

            # 最適なパターンを選択
            best_pattern = max(patterns, key=lambda p: p['confidence'])

            # テスト入力にパターンを適用
            test_input = task.test[0]['input']
            predicted_output = self._apply_pattern(best_pattern, test_input)

            if predicted_output is None:
                return {'success': False, 'method': 'pattern_recognition'}

            return {
                'success': True,
                'method': 'pattern_recognition',
                'pattern': best_pattern,
                'predicted_output': predicted_output,
                'confidence': best_pattern['confidence']
            }

        except Exception as e:
            print(f"パターン認識エラー: {e}")
            return {'success': False, 'method': 'pattern_recognition', 'error': str(e)}

    def _solve_with_object_matching(self, task: Task) -> Dict[str, Any]:
        """オブジェクトマッチングによる解決（削除済み）"""
        return {'success': False, 'method': 'object_matching', 'error': '従来のオブジェクトマッチングは削除されました'}

    def _select_best_solution(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適な解決策を選択"""
        if not solutions:
            return None

        # 信頼度でソート
        solutions.sort(key=lambda s: s.get('confidence', 0.0), reverse=True)

        return solutions[0]

    def _execute_program(self, program: str, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """プログラムを実行"""
        if not program or not input_grid:
            return None

        try:
            from src.core_systems.executor.core import ExecutorCore
            import numpy as np

            executor = ExecutorCore()
            input_array = np.array(input_grid)
            output_array = executor.execute_program(program, input_array)

            if output_array is None:
                return None

            return output_array.tolist() if isinstance(output_array, np.ndarray) else output_array

        except Exception:
            return None

    def _calculate_confidence(self, program: str, task: Task) -> float:
        """信頼度を計算（本格実装）"""
        if not program or not task.train:
            return 0.0

        confidence_scores = []

        # 1. 訓練データでの一貫性チェック
        consistency_score = self._check_training_consistency(program, task)
        confidence_scores.append(consistency_score * 0.4)  # 重み: 40%

        # 2. プログラムの複雑度（シンプルなプログラムほど信頼度が高い）
        complexity_score = self._calculate_complexity_score(program)
        confidence_scores.append(complexity_score * 0.2)  # 重み: 20%

        # 3. プログラムの実行可能性
        executability_score = self._check_executability(program)
        confidence_scores.append(executability_score * 0.2)  # 重み: 20%

        # 4. 訓練ペア数による信頼度（多いほど信頼度が高い）
        training_pairs_score = min(1.0, len(task.train) / 5.0)  # 5ペアで最大
        confidence_scores.append(training_pairs_score * 0.2)  # 重み: 20%

        # 総合信頼度
        total_confidence = sum(confidence_scores)
        return max(0.0, min(1.0, total_confidence))

    def _check_training_consistency(self, program: str, task: Task) -> float:
        """訓練データでの一貫性をチェック"""
        if not task.train:
            return 0.0

        correct_count = 0
        for train_pair in task.train:
            try:
                predicted = self._execute_program(program, train_pair['input'])
                expected = train_pair['output']
                if predicted == expected:
                    correct_count += 1
            except Exception:
                continue

        return correct_count / len(task.train) if task.train else 0.0

    def _calculate_complexity_score(self, program: str) -> float:
        """プログラムの複雑度スコアを計算（シンプルなほど高い）"""
        if not program:
            return 0.0

        # トークン数
        token_count = len(program.split())

        # ネスト深度（FOR/IFの数）
        nest_depth = program.count('FOR ') + program.count('IF ')

        # 複雑度スコア（シンプルなほど高い）
        # トークン数が少なく、ネストが浅いほど高スコア
        token_score = max(0.0, 1.0 - (token_count - 10) / 50.0)  # 10トークンで1.0、60トークンで0.0
        nest_score = max(0.0, 1.0 - nest_depth / 5.0)  # ネスト0で1.0、5で0.0

        return (token_score + nest_score) / 2.0

    def _check_executability(self, program: str) -> float:
        """プログラムの実行可能性をチェック"""
        if not program:
            return 0.0

        score = 0.0

        # 基本的な構文チェック
        if any(cmd in program for cmd in ['GET_', 'SET_', 'FOR ', 'IF ']):
            score += 0.3

        # 括弧のバランス
        if program.count('(') == program.count(')'):
            score += 0.2

        # ループ構造のチェック
        for_count = program.count('FOR ')
        do_count = program.count(' DO')
        end_count = program.count('END')
        if for_count > 0 and do_count >= for_count and end_count >= for_count:
            score += 0.3
        elif for_count == 0:
            score += 0.2  # ループがない場合も有効

        # 変数代入のチェック
        if '=' in program:
            score += 0.2

        return min(1.0, score)

    def _apply_pattern(self, pattern: Dict[str, Any], input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """パターンを適用（本格実装）"""
        pattern_type = pattern.get('type', 'unknown')
        pattern_params = pattern.get('parameters', {})

        try:
            import numpy as np
            from src.core_systems.executor.core import ExecutorCore

            executor = ExecutorCore()
            input_array = np.array(input_grid, dtype=np.int64)

            # パターンタイプに応じてプログラムを生成して実行
            if pattern_type == 'identity':
                program = "GET_ALL_OBJECTS(4)"
            elif pattern_type == 'color_change':
                target_color = pattern_params.get('target_color', 1)
                program = (
                    f"objects = GET_ALL_OBJECTS(4)\n"
                    f"FOR i LEN(objects) DO\n"
                    f"    objects[i] = SET_COLOR(objects[i], {target_color})\n"
                    f"END"
                )
            elif pattern_type == 'flip_horizontal':
                program = (
                    "objects = GET_ALL_OBJECTS(4)\n"
                    "FOR i LEN(objects) DO\n"
                    "    objects[i] = FLIP(objects[i], \"X\")\n"
                    "END"
                )
            elif pattern_type == 'flip_vertical':
                program = (
                    "objects = GET_ALL_OBJECTS(4)\n"
                    "FOR i LEN(objects) DO\n"
                    "    objects[i] = FLIP(objects[i], \"Y\")\n"
                    "END"
                )
            elif pattern_type == 'rotate':
                angle = pattern_params.get('angle', 90)
                program = (
                    f"objects = GET_ALL_OBJECTS(4)\n"
                    f"FOR i LEN(objects) DO\n"
                    f"    objects[i] = ROTATE(objects[i], {angle})\n"
                    f"END"
                )
            elif pattern_type == 'scale':
                factor = pattern_params.get('factor', 2)
                program = (
                    f"objects = GET_ALL_OBJECTS(4)\n"
                    f"FOR i LEN(objects) DO\n"
                    f"    objects[i] = SCALE(objects[i], {factor})\n"
                    f"END"
                )
            elif pattern_type == 'move':
                dx = pattern_params.get('dx', 0)
                dy = pattern_params.get('dy', 0)
                program = (
                    f"objects = GET_ALL_OBJECTS(4)\n"
                    f"FOR i LEN(objects) DO\n"
                    f"    objects[i] = MOVE(objects[i], {dx}, {dy})\n"
                    f"END"
                )
            else:
                # 未知のパターンタイプの場合、恒等変換を返す
                program = "GET_ALL_OBJECTS(4)"

            # プログラムを実行
            output_array = executor.execute_program(program, input_array)
            if output_array is None:
                return None

            return output_array.tolist() if isinstance(output_array, np.ndarray) else output_array

        except Exception:
            # エラーが発生した場合、フォールバックとして恒等変換を返す
            return input_grid

    def _apply_object_matching(self, matching_result: Dict[str, Any], input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """オブジェクトマッチング結果を適用（削除済み）"""
        return None

    def get_solver_statistics(self) -> Dict[str, Any]:
        """ソルバー統計を取得"""
        stats = dict(self.solver_stats)

        if stats['total_solving_attempts'] > 0:
            stats['success_rate'] = stats['successful_solutions'] / stats['total_solving_attempts']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.solver_stats = {
            'total_solving_attempts': 0,
            'successful_solutions': 0,
            'failed_solutions': 0,
            'average_solving_time': 0.0,
            'method_usage': {
                'program_synthesis': 0,
                'pattern_recognition': 0,
                'object_matching': 0
            }
        }
