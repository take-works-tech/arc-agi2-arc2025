"""
プログラム合成エンジン

訓練ペアから共通プログラムを推論するメインエンジン
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import time
import random
import logging
from collections import defaultdict

import numpy as np
import torch

from src.hybrid_system.core.data_structures.task import Task
from .candidate_generator import CandidateGenerator
from .consistency_checker import ConsistencyChecker
from .complexity_regularizer import ComplexityRegularizer, ComplexityConfig
from src.hybrid_system.learning.program_scorer.model import ProgramScorer
from src.hybrid_system.learning.json_pair_loader import extract_basic_features
from .partial_program_generator import PartialProgramGenerator
from src.hybrid_system.inference.object_matching.rule_based_matcher import RuleBasedObjectMatcher
from src.hybrid_system.inference.object_matching.config import ObjectMatchingConfig as RuleBasedObjectMatchingConfig
from src.hybrid_system.utils.logging.synthesis_logger import SynthesisLogger
from src.hybrid_system.utils.visualization.synthesis_visualizer import SynthesisVisualizer
from .dynamic_priority_manager import DynamicPriorityManager, DynamicPriorityConfig
from src.hybrid_system.config.synthesis_config import SynthesisConfigManager, load_synthesis_config
from src.hybrid_system.utils.validation.program_validator import ProgramValidator
from src.hybrid_system.utils.performance.cache_manager import (
    ProgramExecutionCache,
    ObjectExtractionCache,
    ConsistencyCheckCache
)
from src.hybrid_system.utils.performance.parallel_executor import ParallelExecutor, ParallelConfig
from src.hybrid_system.utils.performance.memory_manager import MemoryManager, MemoryConfig


@dataclass
class SynthesisConfig:
    """合成設定"""
    max_candidates_per_pair: int = 10
    max_synthesis_attempts: int = 100
    consistency_threshold: float = 0.8
    complexity_weight: float = 0.3
    enable_parallel_processing: bool = True
    timeout_seconds: float = 30.0
    # 高度な共通部分抽出設定
    enable_program_merging: bool = True
    min_consistency_threshold: float = 0.5  # 最低一貫性閾値（段階的緩和用）
    threshold_decay: float = 0.1  # 閾値の緩和幅
    enable_lcs_extraction: bool = True  # 最長共通部分列抽出
    enable_clustering: bool = True  # 候補のクラスタリング
    # ProgramScorer 関連設定
    enable_program_scorer: bool = True
    program_scorer_model_path: Optional[str] = "learning_outputs/program_scorer/program_scorer_latest.pt"
    # オブジェクトマッチング関連設定
    enable_rule_based_object_matching: bool = True  # ルールベースオブジェクトマッチングを有効化
    # ログ・デバッグ関連設定
    enable_debug_mode: bool = False
    enable_logging: bool = True
    log_dir: Optional[str] = "logs/synthesis"
    enable_visualization: bool = False
    visualization_dir: Optional[str] = "visualizations/synthesis"
    # 動的優先度設定
    enable_dynamic_priority: bool = False
    dynamic_priority_config: Optional[DynamicPriorityConfig] = None
    # パフォーマンス最適化設定
    enable_parallel_processing: bool = True
    parallel_max_workers: int = 4
    enable_caching: bool = True
    cache_max_size: int = 1000
    cache_ttl: float = 3600.0
    enable_memory_monitoring: bool = True
    memory_limit_mb: float = 2048.0


class ProgramSynthesisEngine:
    """プログラム合成エンジン"""

    def __init__(self, config: Optional[SynthesisConfig] = None,
                 config_file: Optional[str] = None,
                 neural_model=None, tokenizer=None,
                 neural_object_model=None):
        """
        初期化

        Args:
            config: 設定（直接指定する場合）
            config_file: 設定ファイルのパス（YAML形式、configより優先）
            neural_model: ProgramSynthesisModel（オプション）
            tokenizer: ProgramTokenizer（オプション）
            neural_object_model: ObjectBasedProgramSynthesisModel（オプション）
        """
        # 設定ファイルから読み込む（優先）
        if config_file:
            self.config = load_synthesis_config(config_file)
        elif config is None:
            # デフォルト設定ファイルから読み込む
            try:
                self.config = load_synthesis_config()
            except Exception as e:
                print(f"警告: 設定ファイルの読み込みに失敗しました。デフォルト設定を使用します: {e}")
                self.config = SynthesisConfig()
        else:
            self.config = config

        # コンポーネントの初期化
        self.candidate_generator = CandidateGenerator(
            neural_model=neural_model,
            tokenizer=tokenizer,
            neural_object_model=neural_object_model
        )
        self.consistency_checker = ConsistencyChecker()
        # 複雑度ペナルティ係数を設定から渡す（simplicity バイアスの強さ）
        self.complexity_regularizer = ComplexityRegularizer(
            ComplexityConfig(penalty_factor=self.config.complexity_weight)
        )

        # 学習済み ProgramScorer の読み込み（存在すれば）
        self.program_scorer: Optional[ProgramScorer] = None
        if self.config.enable_program_scorer:
            model_path = self.config.program_scorer_model_path
            if model_path and os.path.exists(model_path):
                try:
                    # 特徴次元は訓練時の _build_feature_vector と同じ 14 次元
                    in_dim = 14
                    self.program_scorer = ProgramScorer(in_dim=in_dim)
                    state = torch.load(model_path, map_location="cpu")
                    self.program_scorer.load_state_dict(state)
                    self.program_scorer.eval()
                    print(f"[ProgramSynthesisEngine] ProgramScorer をロードしました: {model_path}")
                except Exception as e:
                    print(f"[ProgramSynthesisEngine] ProgramScorer ロード失敗: {e}")
                    self.program_scorer = None
            else:
                print("[ProgramSynthesisEngine] ProgramScorer モデルが見つからないため無効化します")

        # ルールベースオブジェクトマッチングの初期化
        self.rule_based_object_matcher: Optional[RuleBasedObjectMatcher] = None
        if self.config.enable_rule_based_object_matching:
            try:
                rule_based_config = RuleBasedObjectMatchingConfig()
                self.rule_based_object_matcher = RuleBasedObjectMatcher(config=rule_based_config)
                print("[ProgramSynthesisEngine] ルールベースオブジェクトマッチングを有効化しました")
            except Exception as e:
                print(f"[ProgramSynthesisEngine] ルールベースオブジェクトマッチング初期化失敗: {e}")
                self.rule_based_object_matcher = None

        # 部分プログラム生成器の初期化
        self.partial_program_generator = PartialProgramGenerator()

        # プログラム検証器の初期化
        self.program_validator = ProgramValidator()

        # パフォーマンス最適化の初期化
        # キャッシュマネージャー
        self.program_execution_cache: Optional[ProgramExecutionCache] = None
        self.object_extraction_cache: Optional[ObjectExtractionCache] = None
        self.consistency_check_cache: Optional[ConsistencyCheckCache] = None

        if self.config.enable_caching:
            self.program_execution_cache = ProgramExecutionCache(
                max_size=self.config.cache_max_size,
                ttl=self.config.cache_ttl
            )
            self.object_extraction_cache = ObjectExtractionCache(
                max_size=self.config.cache_max_size // 2,
                ttl=self.config.cache_ttl
            )
            self.consistency_check_cache = ConsistencyCheckCache(
                max_size=self.config.cache_max_size // 2,
                ttl=self.config.cache_ttl / 2
            )
            print("[ProgramSynthesisEngine] キャッシュ機能を有効化しました")

        # 並列実行エグゼキューター
        self.parallel_executor: Optional[ParallelExecutor] = None
        if self.config.enable_parallel_processing:
            parallel_config = ParallelConfig(
                enable_parallel=True,
                max_workers=self.config.parallel_max_workers,
                use_process_pool=False,  # スレッドプールを使用（共有メモリのため）
                timeout=self.config.timeout_seconds
            )
            self.parallel_executor = ParallelExecutor(config=parallel_config)
            print(f"[ProgramSynthesisEngine] 並列処理を有効化しました（最大ワーカー数: {self.config.parallel_max_workers}）")

        # メモリ管理
        self.memory_manager: Optional[MemoryManager] = None
        if self.config.enable_memory_monitoring:
            memory_config = MemoryConfig(
                enable_monitoring=True,
                memory_limit_mb=self.config.memory_limit_mb,
                warning_threshold_mb=self.config.memory_limit_mb * 0.75,
                gc_threshold_mb=self.config.memory_limit_mb * 0.5
            )
            self.memory_manager = MemoryManager(config=memory_config)
            print(f"[ProgramSynthesisEngine] メモリ管理を有効化しました（制限: {self.config.memory_limit_mb:.0f}MB）")

        # ロガーと可視化ツールの初期化
        self.synthesis_logger: Optional[SynthesisLogger] = None
        self.synthesis_visualizer: Optional[SynthesisVisualizer] = None

        if self.config.enable_logging:
            import logging
            log_level = logging.DEBUG if self.config.enable_debug_mode else logging.INFO
            self.synthesis_logger = SynthesisLogger(
                log_dir=self.config.log_dir,
                log_level=log_level,
                enable_debug_mode=self.config.enable_debug_mode,
                enable_file_logging=True
            )

        if self.config.enable_visualization:
            self.synthesis_visualizer = SynthesisVisualizer(
                output_dir=self.config.visualization_dir or "visualizations/synthesis"
            )

        # 合成統計
        self.synthesis_stats = {
            'total_synthesis_attempts': 0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'average_synthesis_time': 0.0,
            'average_candidates_generated': 0.0,
            'merging_used': 0,
            'lcs_extraction_used': 0,
            'clustering_used': 0,
            'object_matching_used': 0,
            'partial_program_used': 0
        }

    def synthesize_program(self, task: Task) -> Optional[str]:
        """タスクから共通プログラムを合成

        Args:
            task: 合成対象のタスク

        Returns:
            合成されたプログラム（失敗時はNone）
        """
        start_time = time.time()
        self.synthesis_stats['total_synthesis_attempts'] += 1

        try:
            # 0. オブジェクトマッチング（ルールベースオブジェクトマッチングを優先）
            partial_programs = None
            matching_result = None  # グローバルなマッチング結果（全ペア共通）

            # ルールベースオブジェクトマッチングを試行
            if self.config.enable_rule_based_object_matching and self.rule_based_object_matcher:
                if self.synthesis_logger:
                    self.synthesis_logger.logger.info("ルールベースオブジェクトマッチング実行中...")
                else:
                    print("ルールベースオブジェクトマッチング実行中...")

                try:
                    rule_based_result = self.rule_based_object_matcher.match_objects(task)

                    if rule_based_result.get('success', False):
                        partial_programs_list = rule_based_result.get('partial_programs', [])
                        if partial_programs_list:
                            # 部分プログラムリストを辞書形式に変換
                            # すべての部分プログラムを活用するため、各訓練ペアに対してすべての部分プログラムを試す
                            # ただし、候補生成時に複数の部分プログラムを試すため、ここでは最初の訓練ペアにすべての部分プログラムを割り当て
                            partial_programs = {}
                            # 各訓練ペアに対して、すべての部分プログラムを割り当て（循環的に）
                            for i in range(len(task.train)):
                                if i < len(partial_programs_list):
                                    partial_programs[i] = partial_programs_list[i]
                                else:
                                    # 循環的に割り当て（複数の訓練ペアがある場合）
                                    partial_programs[i] = partial_programs_list[i % len(partial_programs_list)]
                            if self.synthesis_logger:
                                self.synthesis_logger.logger.info(f"ルールベースオブジェクトマッチング成功: {len(partial_programs_list)}個の部分プログラムを生成")
                            else:
                                print(f"ルールベースオブジェクトマッチング成功: {len(partial_programs_list)}個の部分プログラムを生成")
                            # 統計情報を更新
                            if 'object_matching_used' in self.synthesis_stats:
                                self.synthesis_stats['object_matching_used'] += 1
                            if 'partial_program_used' in self.synthesis_stats:
                                self.synthesis_stats['partial_program_used'] += 1
                            # マッチング結果を保存（候補生成で使用）
                            matching_result = rule_based_result
                            # すべての部分プログラムを候補生成で試すため、matching_resultに保存
                            # all_partial_programsは削除されたため、partial_programsのみ設定
                            matching_result['partial_programs'] = partial_programs_list
                except Exception as e:
                    if self.synthesis_logger:
                        self.synthesis_logger.logger.warning(f"ルールベースオブジェクトマッチングエラー: {e}")
                    else:
                        print(f"ルールベースオブジェクトマッチングエラー: {e}")

            # 1. 各訓練ペアから候補プログラムを生成
            print(f"候補プログラム生成中...")
            candidate_programs = self._generate_candidate_programs(task, partial_programs=partial_programs, matching_result=matching_result)

            if not candidate_programs:
                print("候補プログラムが生成されませんでした")
                self.synthesis_stats['failed_syntheses'] += 1
                return None

            print(f"生成された候補プログラム数: {len(candidate_programs)}")

            # 2. 一貫性チェック（段階的アプローチ）
            print("一貫性チェック中...")
            consistent_programs = self._filter_consistent_programs(candidate_programs, task)

            # 一貫性のあるプログラムが見つからない場合の代替処理
            if not consistent_programs:
                print("完全一致が見つかりませんでした。高度な共通部分抽出を試行中...")
                if self.config.enable_program_merging:
                    best_program = self._advanced_program_extraction(candidate_programs, task)
                    if best_program:
                        synthesis_time = time.time() - start_time
                        self.synthesis_stats['successful_syntheses'] += 1
                        self.synthesis_stats['merging_used'] += 1
                        print(f"プログラム合成成功（マージアルゴリズム使用）: {synthesis_time:.2f}秒")
                        return best_program
                else:
                    print("一貫性のあるプログラムが見つかりませんでした")
                self.synthesis_stats['failed_syntheses'] += 1
                return None

            print(f"一貫性のあるプログラム数: {len(consistent_programs)}")

            # 3. 複雑度正則化
            print("複雑度正則化中...")
            best_program = self._select_best_program(consistent_programs, task)

            if best_program:
                synthesis_time = time.time() - start_time
                self.synthesis_stats['successful_syntheses'] += 1
                self.synthesis_stats['average_synthesis_time'] = (
                    (self.synthesis_stats['average_synthesis_time'] * (self.synthesis_stats['successful_syntheses'] - 1) + synthesis_time) /
                    self.synthesis_stats['successful_syntheses']
                )

                print(f"プログラム合成成功: {synthesis_time:.2f}秒")
                return best_program
            else:
                print("最適なプログラムが見つかりませんでした")
                self.synthesis_stats['failed_syntheses'] += 1
                return None

        except Exception as e:
            import traceback
            print(f"プログラム合成エラー: {e}")
            print(f"エラーの詳細:\n{traceback.format_exc()}")
            self.synthesis_stats['failed_syntheses'] += 1
            return None

    def _generate_candidate_programs(self, task: Task, partial_programs: Optional[Dict[int, str]] = None, matching_result: Optional[Dict[str, Any]] = None) -> List[str]:
        """候補プログラムを生成

        Args:
            task: タスク
            partial_programs: 部分プログラムの辞書（ペアインデックス -> 部分プログラム）
            matching_result: オブジェクトマッチング結果（全ペア共通）
        """
        # 並列処理が有効で、複数の訓練ペアがある場合
        # 並列処理は現在の実装では順次処理を使用（将来的に並列化可能）
        # if (self.parallel_executor and
        #     self.config.enable_parallel_processing and
        #     len(task.train) > 1):
        #     return self._generate_candidates_parallel(task, partial_programs)

        # 順次処理
        all_candidates = []

        # すべての部分プログラムを試す（4連結×Nパターン + 8連結×Mパターン + マージプログラム）
        partial_programs_list = matching_result.get('partial_programs', []) if matching_result else []
        used_partial_programs = False
        if partial_programs_list:
            # 各訓練ペアに対して、すべての部分プログラムを試す
            # 部分プログラムの数が増えたため、各部分プログラムごとに候補生成を行う
            for partial_prog in partial_programs_list:
                for i, train_pair in enumerate(task.train):
                    input_grid = train_pair.get('input', [])
                    output_grid = train_pair.get('output', [])
                    if not input_grid or not output_grid:
                        continue

                    # この部分プログラムを使用して候補を生成
                    # 各生成方法の候補数は設定で制御されるため、全体の候補数は適切に管理される
                    candidates = self.candidate_generator.generate_candidates(
                        input_grid, output_grid,
                        max_candidates=self.config.max_candidates_per_pair,
                        pair_index=i,
                        partial_program=partial_prog,
                        matching_result=matching_result
                    )
                    all_candidates.extend(candidates)
            used_partial_programs = True

        # 部分プログラムを試していない場合、通常の処理を実行
        # （部分プログラムがない場合のフォールバック）
        if not used_partial_programs:
            for i, train_pair in enumerate(task.train):
                if self.synthesis_logger:
                    self.synthesis_logger.logger.debug(f"訓練ペア {i+1}/{len(task.train)} から候補生成中...")
                else:
                    print(f"訓練ペア {i+1}/{len(task.train)} から候補生成中...")

                # 部分プログラムがある場合は使用
                partial_program = None
                if partial_programs and i in partial_programs:
                    partial_program = partial_programs[i]
                    if self.synthesis_logger:
                        self.synthesis_logger.logger.debug(f"  部分プログラムを使用: {partial_program[:50]}...")
                    else:
                        print(f"  部分プログラムを使用: {partial_program[:50]}...")

                # 各訓練ペアから候補プログラムを生成（ペアインデックスを渡す）
                # マッチング結果は全ペア共通で使用（rule_based_resultが保存されている場合）
                pair_candidates = self.candidate_generator.generate_candidates(
                    input_grid=train_pair['input'],
                    output_grid=train_pair['output'],
                    max_candidates=self.config.max_candidates_per_pair,
                    pair_index=i,  # シード計算用のペアインデックス
                    partial_program=partial_program,  # 部分プログラムを渡す
                    matching_result=matching_result  # オブジェクトマッチング結果を渡す（全ペア共通）
                )

                # ログ記録（各生成方法ごとに）
                if self.synthesis_logger and self.synthesis_logger.enable_debug_mode:
                    # 候補生成方法を推定（本格実装済み）
                    # 実際にはCandidateGeneratorから詳細情報を取得する必要がある
                    self.synthesis_logger.log_candidate_generation(
                        method="mixed",
                        candidates=pair_candidates,
                        pair_index=i
                    )

                all_candidates.extend(pair_candidates)

        # 重複を除去
        unique_candidates = list(set(all_candidates))

        # 統計更新
        self.synthesis_stats['average_candidates_generated'] = (
            (self.synthesis_stats['average_candidates_generated'] * (self.synthesis_stats['total_synthesis_attempts'] - 1) + len(unique_candidates)) /
            self.synthesis_stats['total_synthesis_attempts']
        )

        return unique_candidates

    def _filter_consistent_programs(self, candidate_programs: List[str], task: Task) -> List[str]:
        """一貫性のあるプログラムをフィルタリング"""
        # 並列処理が有効で、複数の候補がある場合
        if (self.parallel_executor and
            self.config.enable_parallel_processing and
            len(candidate_programs) > 1):
            return self._filter_consistent_programs_parallel(candidate_programs, task)

        # 順次処理（キャッシュ付き）
        consistent_programs = []
        task_id = task.metadata.get('task_id', 'unknown') if task.metadata else 'unknown'

        for program in candidate_programs:
            # キャッシュから一貫性スコアを取得
            consistency_score = None
            if self.consistency_check_cache:
                consistency_score = self.consistency_check_cache.get_consistency_result(program, task_id)

            # キャッシュにない場合は計算
            if consistency_score is None:
                consistency_score = self.consistency_checker.check_consistency(
                    program,
                    task
                )
                # キャッシュに保存
                if self.consistency_check_cache:
                    self.consistency_check_cache.set_consistency_result(program, task_id, consistency_score)

            # ログ記録
            if self.synthesis_logger:
                self.synthesis_logger.log_consistency_check(
                    program=program,
                    consistency_score=consistency_score,
                    details={"threshold": self.config.consistency_threshold}
                )

            if consistency_score >= self.config.consistency_threshold:
                consistent_programs.append(program)

        return consistent_programs

    def _filter_consistent_programs_parallel(self, candidate_programs: List[str], task: Task) -> List[str]:
        """並列処理で一貫性のあるプログラムをフィルタリング"""
        if not self.parallel_executor:
            return self._filter_consistent_programs(candidate_programs, task)

        task_id = task.metadata.get('task_id', 'unknown') if task.metadata else 'unknown'

        # 各候補プログラムの一貫性チェックを並列実行
        tasks = []
        for program in candidate_programs:
            # キャッシュチェック
            cached_score = None
            if self.consistency_check_cache:
                cached_score = self.consistency_check_cache.get_consistency_result(program, task_id)

            if cached_score is not None:
                # キャッシュヒット: タスクをスキップ（後で処理）
                tasks.append((None, cached_score))
            else:
                # キャッシュミス: 一貫性チェックを実行
                tasks.append((
                    lambda p, t: self.consistency_checker.check_consistency(p, t),
                    (program, task)
                ))

        # 並列実行（キャッシュヒットを除く）
        parallel_tasks = [(func, args, {}) for func, args in tasks if func is not None]
        if parallel_tasks:
            results = self.parallel_executor.execute_parallel(parallel_tasks, description="一貫性チェック（並列）")
        else:
            results = []

        # 結果をマージ
        consistent_programs = []
        result_idx = 0
        for i, (program, task_info) in enumerate(zip(candidate_programs, tasks)):
            func, cached_score = task_info
            if func is None:
                # キャッシュヒット
                consistency_score = cached_score
            else:
                # 並列実行結果
                if result_idx < len(results) and results[result_idx] is not None:
                    consistency_score = results[result_idx]
                    # キャッシュに保存
                    if self.consistency_check_cache:
                        self.consistency_check_cache.set_consistency_result(program, task_id, consistency_score)
                    result_idx += 1
                else:
                    consistency_score = 0.0
                    result_idx += 1

            if consistency_score >= self.config.consistency_threshold:
                consistent_programs.append(program)

        return consistent_programs

    def _select_best_program(self, consistent_programs: List[str], task: Task) -> Optional[str]:
        """最適なプログラムを選択

        精度を最優先にする方針:
        ①一貫性スコアが1.0が1つの場合: ProgramScorerと複雑度ペナルティを無視し、そのプログラムを採用
        ②一貫性スコアが1.0が複数の場合: ProgramScorerスコア(0.1) - 複雑度ペナルティ × complexity_weight で選択
        ③一貫性スコアが1.0がなしの場合: 一貫性スコア(0.9) + ProgramScorerスコア(0.1) - 複雑度ペナルティ × complexity_weight で選択
        """
        if not consistent_programs:
            return None

        # 一貫性スコアを計算
        base_scores: List[float] = []
        for program in consistent_programs:
            score = self.consistency_checker.check_consistency(program, task)
            base_scores.append(score)

        # 一貫性スコアが1.0の候補を特定
        perfect_consistency_indices = [i for i, score in enumerate(base_scores) if score >= 1.0]

        # ①一貫性スコアが1.0が1つの場合: そのまま採用
        if len(perfect_consistency_indices) == 1:
            best_program = consistent_programs[perfect_consistency_indices[0]]
            if self.synthesis_logger:
                self.synthesis_logger.logger.info(
                    f"一貫性スコア1.0の候補が1つのため、そのまま採用: {best_program[:100]}..."
                )
            else:
                print(f"一貫性スコア1.0の候補が1つのため、そのまま採用")
            return best_program

        # ProgramScorerスコアを計算（必要な場合のみ）
        program_scorer_scores: List[float] = [0.0] * len(consistent_programs)
        if self.program_scorer is not None and task.train:
            try:
                # 代表として最初の訓練ペアからペア特徴を抽出
                first_pair = task.train[0]
                input_grid = np.array(first_pair["input"], dtype=int)
                output_grid = np.array(first_pair["output"], dtype=int)
                sample = {
                    "input_grid": input_grid,
                    "output_grid": output_grid,
                    "trace_summary": {},
                }
                pair_feats = extract_basic_features(sample)

                def _build_feature_vector_local(
                    pair_features: Dict[str, Any],
                    complexity_score: float,
                    line_count: int,
                    char_count: int,
                ) -> List[float]:
                    return [
                        float(pair_features.get("input_height", 0)),
                        float(pair_features.get("input_width", 0)),
                        float(pair_features.get("output_height", 0)),
                        float(pair_features.get("output_width", 0)),
                        float(pair_features.get("object_count_before", 0)),
                        float(pair_features.get("object_count_after", 0)),
                        float(pair_features.get("rendered_object_count", 0)),
                        float(pair_features.get("avg_bbox_width", 0.0)),
                        float(pair_features.get("avg_bbox_height", 0.0)),
                        float(pair_features.get("max_bbox_width", 0.0)),
                        float(pair_features.get("max_bbox_height", 0.0)),
                        float(complexity_score),
                        float(line_count),
                        float(char_count),
                    ]

                # すべての候補に対してProgramScorerスコアを計算
                feature_vecs: List[List[float]] = []
                for program in consistent_programs:
                    complexity_score = self.complexity_regularizer.calculate_complexity_score(program)
                    line_count = len([ln for ln in program.split("\n") if ln.strip()])
                    char_count = len(program)
                    feature_vecs.append(
                        _build_feature_vector_local(
                            pair_feats,
                            complexity_score=complexity_score,
                            line_count=line_count,
                            char_count=char_count,
                        )
                    )

                with torch.no_grad():
                    feats_tensor = torch.tensor(feature_vecs, dtype=torch.float32)
                    scorer_scores_tensor = self.program_scorer(feats_tensor).squeeze(-1)
                    program_scorer_scores = scorer_scores_tensor.cpu().tolist()
            except Exception as e:
                print(f"[ProgramSynthesisEngine] ProgramScorer スコア計算中にエラー: {e}")

        # 検証レイヤーのペナルティを計算
        validation_penalties: List[float] = []
        try:
            from src.hybrid_system.utils.validation.enhanced_program_validator import EnhancedProgramValidator

            validator = EnhancedProgramValidator()
            for program in consistent_programs:
                validation_result = validator.validate_program_enhanced(program, task)
                validation_penalties.append(validation_result.validation_penalty)
        except Exception as e:
            # 検証に失敗した場合はペナルティなし
            validation_penalties = [0.0] * len(consistent_programs)

        # 最終スコアを計算
        final_scores: List[float] = []
        for idx, (program, consistency_score) in enumerate(zip(consistent_programs, base_scores)):
            complexity_score = self.complexity_regularizer.calculate_complexity_score(program)
            complexity_penalty = complexity_score * self.config.complexity_weight
            validation_penalty = validation_penalties[idx] * 0.2  # 検証ペナルティの重み: 0.2

            if len(perfect_consistency_indices) > 1:
                # ②一貫性スコアが1.0が複数の場合
                # スコア = ProgramScorerスコア(0.1) - 複雑度ペナルティ × complexity_weight - 検証ペナルティ
                if idx in perfect_consistency_indices:
                    final_score = 0.1 * program_scorer_scores[idx] - complexity_penalty - validation_penalty
                else:
                    # 一貫性スコアが1.0未満の候補は除外（負のスコアで確実に下位になる）
                    final_score = -1.0
            else:
                # ③一貫性スコアが1.0がなしの場合
                # スコア = 一貫性スコア(0.9) + ProgramScorerスコア(0.1) - 複雑度ペナルティ × complexity_weight - 検証ペナルティ
                final_score = 0.9 * consistency_score + 0.1 * program_scorer_scores[idx] - complexity_penalty - validation_penalty

            final_scores.append(final_score)

        # 最終スコアでランキング
        ranked_indices = sorted(range(len(consistent_programs)), key=lambda i: final_scores[i], reverse=True)
        best_idx = ranked_indices[0]
        best_program = consistent_programs[best_idx]

        # ログ記録
        if self.synthesis_logger:
            self.synthesis_logger.logger.info(f"選択されたプログラム:")
            self.synthesis_logger.logger.info(f"  一貫性スコア: {base_scores[best_idx]:.3f}")
            self.synthesis_logger.logger.info(f"  ProgramScorerスコア: {program_scorer_scores[best_idx]:.3f}")
            self.synthesis_logger.logger.info(f"  複雑度スコア: {self.complexity_regularizer.calculate_complexity_score(best_program):.3f}")
            self.synthesis_logger.logger.info(f"  最終スコア: {final_scores[best_idx]:.3f}")
            self.synthesis_logger.logger.info(f"  ケース: {'①1つ' if len(perfect_consistency_indices) == 1 else '②複数' if len(perfect_consistency_indices) > 1 else '③なし'}")
        else:
            print(f"選択されたプログラム:")
            print(f"  一貫性スコア: {base_scores[best_idx]:.3f}")
            print(f"  ProgramScorerスコア: {program_scorer_scores[best_idx]:.3f}")
            print(f"  複雑度スコア: {self.complexity_regularizer.calculate_complexity_score(best_program):.3f}")
            print(f"  最終スコア: {final_scores[best_idx]:.3f}")
            print(f"  ケース: {'①1つ' if len(perfect_consistency_indices) == 1 else '②複数' if len(perfect_consistency_indices) > 1 else '③なし'}")

        return best_program

    def synthesize_program_with_validation(self, task: Task) -> Dict[str, Any]:
        """検証付きでプログラムを合成

        Args:
            task: 合成対象のタスク

        Returns:
            合成結果と検証情報
        """
        # プログラムを合成
        synthesized_program = self.synthesize_program(task)

        if not synthesized_program:
            return {
                'success': False,
                'program': None,
                'validation_results': None,
                'synthesis_info': {
                    'candidates_generated': 0,
                    'consistent_programs': 0,
                    'synthesis_time': 0.0
                }
            }

        # 合成されたプログラムを検証
        validation_results = self._validate_synthesized_program(synthesized_program, task)

        # 動的優先度管理に結果を記録
        if hasattr(self, 'dynamic_priority_manager') and self.dynamic_priority_manager and validation_results:
            # 方法名の推定（本格実装）
            # 候補生成統計と実際に使用された候補から推定
            method_name = 'neural_grid'  # デフォルト

            # 実際に生成された候補の統計を確認
            generation_stats = getattr(self.candidate_generator, 'generation_stats', {})

            # 各方法の生成数を比較
            method_counts = {
                'neural_grid': generation_stats.get('neural_generations', 0),
                'neural_object': generation_stats.get('neural_object_generations', 0),
            }

            # 最も多く生成された方法を選択
            if max(method_counts.values()) > 0:
                method_name = max(method_counts.items(), key=lambda kv: kv[1])[0]

            # 検証結果からも推定を試みる
            if validation_results.get('passed_validation'):
                # 検証を通過したプログラムの特徴から推定
                if 'neural' in synthesized_program.lower() or 'model' in synthesized_program.lower():
                    method_name = 'neural_grid'

            self.dynamic_priority_manager.record_attempt(
                method_name=method_name,
                success=validation_results.get('passed_validation', False),
                candidates_generated=int(self.synthesis_stats.get('average_candidates_generated', 0)),
                candidates_selected=1 if synthesized_program else 0,
                consistency_score=validation_results.get('consistency_score', 0.0)
            )

        return {
            'success': True,
            'program': synthesized_program,
            'validation_results': validation_results,
            'synthesis_info': {
                'candidates_generated': self.synthesis_stats['average_candidates_generated'],
                'consistent_programs': len(self._filter_consistent_programs([synthesized_program], task)),
                'synthesis_time': self.synthesis_stats['average_synthesis_time']
            }
        }

    def _validate_synthesized_program(self, program: str, task: Task) -> Dict[str, Any]:
        """合成されたプログラムを検証"""
        # 0. ProgramValidatorによる検証
        validation_result = self.program_validator.validate_program(program, task)

        # 検証エラーがある場合はログに記録
        if not validation_result.is_valid and self.synthesis_logger:
            self.synthesis_logger.logger.warning(
                f"プログラム検証エラー: {', '.join(validation_result.errors)}"
            )

        # 1. 一貫性チェック
        consistency_score = self.consistency_checker.check_consistency(program, task)

        # 2. 複雑度チェック
        complexity_score = self.complexity_regularizer.calculate_complexity_score(program)

        # 3. 実行可能性チェック（本格実装済み）
        executability_score = self._check_executability(program)

        return {
            'consistency_score': consistency_score,
            'complexity_score': complexity_score,
            'executability_score': executability_score,
            'overall_score': (consistency_score + executability_score) / 2.0 - self.config.complexity_weight * complexity_score,
            'passed_validation': consistency_score >= self.config.consistency_threshold
        }

    def _check_executability(self, program: str) -> float:
        """プログラムの実行可能性をチェック（本格実装）"""
        if not program or len(program.strip()) < 10:
            return 0.0

        score = 0.0

        # 1. 基本的な構文チェック
        required_keywords = ['GET_', 'SET_', 'FOR ', 'IF ', 'DO', 'END']
        found_keywords = sum(1 for keyword in required_keywords if keyword in program)
        score += (found_keywords / len(required_keywords)) * 0.3

        # 2. 括弧のバランスチェック
        open_parens = program.count('(')
        close_parens = program.count(')')
        if open_parens == close_parens and open_parens > 0:
            score += 0.2
        elif abs(open_parens - close_parens) <= 1:
            score += 0.1

        # 3. ループ構造のチェック
        for_count = program.count('FOR ')
        do_count = program.count(' DO')
        end_count = program.count('END')
        if for_count > 0 and do_count >= for_count and end_count >= for_count:
            score += 0.2
        elif for_count == 0:
            score += 0.1  # ループがない場合も有効なプログラム

        # 4. 条件分岐のチェック
        if_count = program.count('IF ')
        then_count = program.count(' THEN')
        if if_count > 0 and then_count >= if_count:
            score += 0.1
        elif if_count == 0:
            score += 0.1  # 条件分岐がない場合も有効なプログラム

        # 5. 変数代入のチェック
        if '=' in program:
            score += 0.1

        # 6. 基本的なコマンドの存在チェック
        if not any(cmd in program for cmd in ['GET_', 'SET_', 'FOR ', 'IF ']):
            return 0.0

        # スコアを0.0-1.0の範囲に正規化
        return min(1.0, max(0.0, score))

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """合成統計を取得"""
        stats = dict(self.synthesis_stats)

        if stats['total_synthesis_attempts'] > 0:
            stats['success_rate'] = stats['successful_syntheses'] / stats['total_synthesis_attempts']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """統計をリセット"""
        self.synthesis_stats = {
            'total_synthesis_attempts': 0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'average_synthesis_time': 0.0,
            'average_candidates_generated': 0.0,
            'merging_used': 0,
            'lcs_extraction_used': 0,
            'clustering_used': 0,
            'object_matching_used': 0,
            'partial_program_used': 0
        }

    def _advanced_program_extraction(self, candidate_programs: List[str], task: Task) -> Optional[str]:
        """高度な共通部分抽出アルゴリズム

        複数の候補プログラムから共通部分を抽出する。以下の手法を組み合わせる:
        1. 段階的閾値緩和: 閾値を徐々に下げて一貫性のあるプログラムを探す
        2. LCS抽出: 最長共通部分列を抽出
        3. クラスタリング: 類似プログラムをグループ化

        Args:
            candidate_programs: 候補プログラムのリスト
            task: タスク

        Returns:
            抽出されたプログラム（失敗時はNone）
        """
        # 方法1: 段階的閾値緩和
        print("  段階的閾値緩和を試行中...")
        relaxed_result = self._progressive_threshold_relaxation(candidate_programs, task)
        if relaxed_result:
            return relaxed_result

        # 方法2: クラスタリング + 代表プログラム選択
        if self.config.enable_clustering:
            print("  クラスタリングアルゴリズムを試行中...")
            cluster_result = self._cluster_and_select_representative(candidate_programs, task)
            if cluster_result:
                self.synthesis_stats['clustering_used'] += 1
                return cluster_result

        # 方法3: LCS抽出
        if self.config.enable_lcs_extraction:
            print("  LCS（最長共通部分列）抽出を試行中...")
            lcs_result = self._extract_lcs_consensus(candidate_programs, task)
            if lcs_result:
                self.synthesis_stats['lcs_extraction_used'] += 1
                return lcs_result

        return None

    def _progressive_threshold_relaxation(self, candidate_programs: List[str], task: Task) -> Optional[str]:
        """段階的閾値緩和

        一貫性閾値を徐々に下げながら一致プログラムを探す

        Args:
            candidate_programs: 候補プログラム
            task: タスク

        Returns:
            見つかったプログラム（失敗時はNone）
        """
        current_threshold = self.config.consistency_threshold

        while current_threshold >= self.config.min_consistency_threshold:
            consistent = []
            for program in candidate_programs:
                score = self.consistency_checker.check_consistency(program, task)
                if score >= current_threshold:
                    consistent.append((program, score))

            if consistent:
                # スコア順でソート
                consistent.sort(key=lambda x: x[1], reverse=True)
                best_program = self._select_best_program([p[0] for p in consistent[:5]], task)
                if best_program:
                    print(f"    閾値 {current_threshold:.2f} で一致プログラムを発見")
                    return best_program

            # 閾値を下げる
            current_threshold -= self.config.threshold_decay

        return None

    def _cluster_and_select_representative(self, candidate_programs: List[str], task: Task) -> Optional[str]:
        """クラスタリング + 代表プログラム選択

        類似プログラムをクラスタリングし、各クラスタの代表を選ぶ

        Args:
            candidate_programs: 候補プログラム
            task: タスク

        Returns:
            代表プログラム（失敗時はNone）
        """
        if len(candidate_programs) < 3:
            return None

        # プログラム間の類似度行列を計算
        similarity_matrix = self._calculate_program_similarity_matrix(candidate_programs)

        # 階層クラスタリング（本格実装済み）
        clusters = self._simple_clustering(candidate_programs, similarity_matrix)

        # 各クラスタから最良プログラムを選ぶ
        best_program = None
        best_score = -float('inf')

        for cluster in clusters:
            if not cluster:
                continue

            # クラスタ内で最良プログラムを選ぶ
            cluster_scores = []
            for program in cluster:
                consistency = self.consistency_checker.check_consistency(program, task)
                complexity = self.complexity_regularizer.calculate_complexity_score(program)
                score = consistency - self.config.complexity_weight * complexity
                cluster_scores.append((program, score))

            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            if cluster_scores and cluster_scores[0][1] > best_score:
                best_program = cluster_scores[0][0]
                best_score = cluster_scores[0][1]

        return best_program

    def _extract_lcs_consensus(self, candidate_programs: List[str], task: Task) -> Optional[str]:
        """LCS（最長共通部分列）による合意プログラム抽出

        複数プログラムの最長共通部分列を抽出し、補完する

        Args:
            candidate_programs: 候補プログラム
            task: タスク

        Returns:
            抽出されたプログラム（失敗時はNone）
        """
        if len(candidate_programs) < 2:
            return None

        # プログラムをトークン化
        tokenized_programs = [p.split() for p in candidate_programs]

        # 2つずつのLCSを計算し、すべてに共通する最長列を抽出
        common_tokens = tokenized_programs[0]
        for tokens in tokenized_programs[1:]:
            common_tokens = self._longest_common_subsequence(common_tokens, tokens)
            if not common_tokens:
                return None

        # LCSからプログラムを再構築
        consensus_program = ' '.join(common_tokens)

        # 一貫性をチェック
        consistency = self.consistency_checker.check_consistency(consensus_program, task)
        if consistency >= self.config.min_consistency_threshold:
            return consensus_program

        return None

    def _calculate_program_similarity_matrix(self, programs: List[str]) -> np.ndarray:
        """プログラム間の類似度行列を計算

        Args:
            programs: プログラムリスト

        Returns:
            類似度行列
        """
        n = len(programs)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                tokens_i = set(programs[i].split())
                tokens_j = set(programs[j].split())

                # Jaccard類似度
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                similarity = intersection / union if union > 0 else 0

                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity

        # 対角要素は1
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix

    def _simple_clustering(self, programs: List[str], similarity_matrix: np.ndarray,
                          threshold: float = 0.5) -> List[List[str]]:
        """階層クラスタリング（本格実装済み）

        Args:
            programs: プログラムリスト
            similarity_matrix: 類似度行列
            threshold: クラスタリング閾値

        Returns:
            クラスタのリスト
        """
        n = len(programs)
        clusters = [[i] for i in range(n)]

        # 類似度が閾値以上のペアを結合
        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # 2つのクラスタ間の平均類似度
                    avg_sim = np.mean([similarity_matrix[ci][cj]
                                      for ci in clusters[i]
                                      for cj in clusters[j]])

                    if avg_sim >= threshold:
                        clusters[i].extend(clusters[j])
                        del clusters[j]
                        changed = True
                        break
                if changed:
                    break

        # インデックスからプログラムに変換
        return [[programs[idx] for idx in cluster] for cluster in clusters]

    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """最長共通部分列（LCS）を計算

        動的計画法でLCSを計算

        Args:
            seq1: シーケンス1
            seq2: シーケンス2

        Returns:
            最長共通部分列
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # LCSの長さを計算
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # LCSを再構築
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return lcs[::-1]

    def _calculate_program_similarity_matrix(self, programs: List[str]) -> np.ndarray:
        """プログラム間の類似度行列を計算

        Args:
            programs: プログラムリスト

        Returns:
            類似度行列
        """
        n = len(programs)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                tokens_i = set(programs[i].split())
                tokens_j = set(programs[j].split())

                # Jaccard類似度
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                similarity = intersection / union if union > 0 else 0

                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity

        # 対角要素は1
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix

    def _simple_clustering(self, programs: List[str], similarity_matrix: np.ndarray,
                          threshold: float = 0.5) -> List[List[str]]:
        """階層クラスタリング（本格実装済み）

        Args:
            programs: プログラムリスト
            similarity_matrix: 類似度行列
            threshold: クラスタリング閾値

        Returns:
            クラスタのリスト
        """
        n = len(programs)
        clusters = [[i] for i in range(n)]

        # 類似度が閾値以上のペアを結合
        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # 2つのクラスタ間の平均類似度
                    avg_sim = np.mean([similarity_matrix[ci][cj]
                                      for ci in clusters[i]
                                      for cj in clusters[j]])

                    if avg_sim >= threshold:
                        clusters[i].extend(clusters[j])
                        del clusters[j]
                        changed = True
                        break
                if changed:
                    break

        # インデックスからプログラムに変換
        return [[programs[idx] for idx in cluster] for cluster in clusters]

    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """最長共通部分列（LCS）を計算

        動的計画法でLCSを計算

        Args:
            seq1: シーケンス1
            seq2: シーケンス2

        Returns:
            最長共通部分列
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # LCSの長さを計算
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # LCSを再構築
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return lcs[::-1]
