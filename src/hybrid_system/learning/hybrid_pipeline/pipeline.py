"""
ハイブリッド学習パイプライン

フェーズ1→2の統合パイプラインを提供
"""

import os
import sys
import time
import json
import gc
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# プロジェクトルートをパスに追加
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(project_root)

from src.hybrid_system.core.data_structures import DataPair, Task, DatasetStatistics
from src.hybrid_system.core.program_synthesis import ProgramPoolManager
from .assembler import TaskAssembler
from .io import DatasetIO
from src.data_systems.generator.file_buffer_manager import FileBufferManager
from src.data_systems.generator.program_serializer import tokenize_program, analyze_program_statistics
from src.data_systems.generator.program_generator.generation.code_generator import generate_code
from src.data_systems.generator.program_executor.node_analyzer.variable_index_normalizer import normalize_variable_indices
from src.data_systems.generator.config import get_config


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    base_dir: str = "data/generated/hybrid"
    num_programs: int = 100
    pairs_per_program: int = 5
    num_train: int = 3
    num_test: int = 1
    apply_manual_rules: bool = True
    max_attempts_per_pair: int = 50
    max_task_attempts: int = 10
    seed: Optional[int] = None


class HybridLearningPipeline:
    """ハイブリッド学習パイプライン"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """初期化"""
        self.config = config or PipelineConfig()

        # ログ出力制御設定を読み込み
        generator_config = get_config()
        self.enable_debug_logs = generator_config.enable_debug_logs

        # コンポーネントの初期化
        self.program_pool_manager = ProgramPoolManager(seed=self.config.seed)
        self.task_assembler = TaskAssembler()
        self.dataset_io = DatasetIO(self.config.base_dir)

        # FileBufferManagerを初期化（PNG/txt保存用）
        # base_dirの親ディレクトリ（タイムスタンプ付きディレクトリ）を使用
        # base_dirが "outputs/YYYYMMDD_HHMMSS/hybrid" の場合、親は "outputs/YYYYMMDD_HHMMSS"
        buffer_base_dir = os.path.dirname(self.config.base_dir) if os.path.dirname(self.config.base_dir) else self.config.base_dir
        self.buffer_manager = FileBufferManager(base_output_dir=buffer_base_dir, auto_flush=False)
        # バッチサイズを1000に設定（10万タスク = 100バッチ）
        self.buffer_manager.BATCH_SIZE = 1000
        self.task_counter = 0  # タスクカウンター

        # NeuralTrainingDataGeneratorを初期化（推論パイプライン用学習データ生成）
        from src.data_systems.generator.neural_training_data_generator import NeuralTrainingDataGenerator
        self.neural_data_generator = NeuralTrainingDataGenerator(output_dir=buffer_base_dir)
        # BATCH_SIZEはNeuralTrainingDataGenerator内で設定されているため、ここでは設定しない

        # 統計情報
        self.pipeline_stats = {
            'phase1_start_time': None,
            'phase1_end_time': None,
            'phase2_start_time': None,
            'phase2_end_time': None,
            'total_programs_generated': 0,
            'total_pairs_generated': 0,
            'total_tasks_created': 0,
            'successful_assemblies': 0,
            'failed_assemblies': 0
        }

        # 詳細な統計情報（main.pyと同じ）
        from src.data_systems.generator.main import TimingStatistics
        self.timing_stats = TimingStatistics()

        # 破棄原因別統計
        self.rejection_reason_stats = {
            'program_data_none': 0,
            'nodes_empty': 0,
            'program_code_empty': 0,
            'silent_exception': 0,
            'execution_timeout': 0,
            'result_nodes_none': 0,
            'input_grid_none': 0,
            'output_grid_none': 0,
            'object_count_limit': 0,
            'index_error': 0,
            'program_error': 0,
            'validation_error': 0,
            'other_value_error': 0,
            'validate_result_none': 0,
            'validate_timeout': 0,
            'validate_exception': 0,
            'other_error': 0,
            'total': 0
        }

        # 条件別の破棄統計
        self.condition_rejection_stats = {
            'condition1': 0,
            'condition2': 0,
            'condition3': 0,
            'condition4': 0,
            'condition5': 0,
            'condition6': 0,
            'multiple': 0,
            'total': 0
        }

    def run_phase1_pair_generation(
        self,
        num_programs: Optional[int] = None,
        pairs_per_program: Optional[int] = None,
        max_attempts_per_pair: int = 50
    ) -> List[DataPair]:
        """フェーズ1: 大量のペア生成

        Args:
            num_programs: 生成するプログラム数
            pairs_per_program: 各プログラムから生成するペア数
            max_attempts_per_pair: 各ペアの最大試行回数

        Returns:
            DataPairのリスト（program情報含む）
        """
        # 設定の取得
        num_programs = num_programs or self.config.num_programs
        pairs_per_program = pairs_per_program or self.config.pairs_per_program

        print(f"フェーズ1開始: {num_programs}プログラム × {pairs_per_program}ペア = {num_programs * pairs_per_program}ペア生成")

        self.pipeline_stats['phase1_start_time'] = time.time()

        # プロファイラーをインポート
        from src.data_systems.generator.program_executor.performance_profiler import get_profiler
        profiler = get_profiler()

        # 1. プログラムプールを生成（main.pyと同じ方法でノードも生成）
        print("プログラム生成中...")
        program_pool_gen_start = time.time()
        # main.pyと同じ方法でプログラムノードとコードを生成
        from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
        from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
        from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
        from src.data_systems.generator.command_usage_statistics import get_global_statistics, reset_global_statistics
        from src.data_systems.generator.main import save_weight_adjustments, load_weight_adjustments, normalize_ratios, select_complexity
        from src.data_systems.generator.config import get_config as get_generator_config

        _generator_config = get_generator_config()
        BATCH_SIZE = self.buffer_manager.BATCH_SIZE

        # コマンド使用統計をリセット
        reset_global_statistics()
        statistics = get_global_statistics()

        # 複雑度比率を正規化
        normalized_complexity_ratios = normalize_ratios(_generator_config.complexity_ratios)

        generator = UnifiedProgramGenerator()

        # 既存の重み調整ファイルがあれば読み込み
        buffer_base_dir = os.path.dirname(self.config.base_dir) if os.path.dirname(self.config.base_dir) else self.config.base_dir
        existing_adjustments = load_weight_adjustments(buffer_base_dir)

        # 関係性生成コマンドの初期重み調整を適用
        if _generator_config.enable_relationship_command_weights:
            # 既存の重み調整と統合（既存の重み調整がある場合は、それに初期重み調整を乗算）
            if existing_adjustments:
                # 既存の重み調整に初期重み調整を統合
                for cmd, weight in _generator_config.relationship_command_weights.items():
                    if cmd in existing_adjustments:
                        # 既存の重み調整がある場合は、初期重み調整を乗算
                        existing_adjustments[cmd] = existing_adjustments[cmd] * weight
                    else:
                        # 既存の重み調整がない場合は、初期重み調整を追加
                        existing_adjustments[cmd] = weight
                generator.command_weight_adjustments = existing_adjustments
                print(f"  読み込んだ重み調整に初期重み調整を統合: {len(existing_adjustments)}個のコマンド")
            else:
                # 既存の重み調整がない場合は、初期重み調整のみを適用
                generator.command_weight_adjustments = _generator_config.relationship_command_weights.copy()
                print(f"  関係性生成コマンドの初期重み調整を適用: {len(_generator_config.relationship_command_weights)}個のコマンド")
                if _generator_config.enable_verbose_logging:
                    for cmd, weight in _generator_config.relationship_command_weights.items():
                        print(f"    {cmd}: {weight:.2f}x")
        else:
            # 関係性生成コマンドの重み調整が無効な場合は、既存の重み調整のみを適用
            if existing_adjustments:
                generator.command_weight_adjustments = existing_adjustments
                print(f"  読み込んだ重み調整を適用: {len(existing_adjustments)}個のコマンド")

        # main.pyと同じヘルパー関数をインポート
        from src.data_systems.generator.main import (
            generate_program, generate_program_with_partial_program_flow,
            validate_execution_results, validate_grid_data,
            collect_commands_to_check, increment_rejection_stat,
            generate_pairs_with_retry, check_all_pairs_conditions, extract_first_pair_data
        )
        from src.data_systems.generator.program_executor.node_analyzer import is_assignment_node, get_commands_sorted_by_depth
        from src.data_systems.generator.program_executor.node_analyzer.execution_helpers import replace_command_with_fallback
        from src.data_systems.generator.program_executor.core_executor import CoreExecutor
        from src.data_systems.generator.program_executor.node_validator_output import check_output_conditions, extract_all_commands_from_nodes
        from src.data_systems.generator.program_executor.node_analyzer.scope_removal_helper import find_scope_pairs, remove_scope_from_nodes
        from src.core_systems.executor.core import SilentException
        import numpy as np
        import random

        program_pool_gen_elapsed = time.time() - program_pool_gen_start
        profiler.record_timing("program_pool_initialization", program_pool_gen_elapsed, "phase1")
        print(f"プログラムプール初期化完了 (処理時間: {program_pool_gen_elapsed:.3f}秒)")

        # 2. 各プログラムから複数のペアを生成（バッチ処理対応 - main.pyと同じ構造）
        print("ペア生成中...")
        data_pairs = []
        self.task_counter = 0  # タスクカウンターをリセット

        # バッチサイズ
        batch_size = self.buffer_manager.BATCH_SIZE
        batch_data_pairs = []  # バッチごとのDataPairリスト（中間保存用）
        batch_number = 0  # バッチ番号（0始まり）

        # バッチ進捗の読み込み（再開機能）
        from src.data_systems.generator.main import load_batch_progress, save_batch_progress
        buffer_base_dir = os.path.dirname(self.config.base_dir) if os.path.dirname(self.config.base_dir) else self.config.base_dir
        batch_progress = load_batch_progress(buffer_base_dir)
        start_task_index = 0
        start_batch_num = 0
        if batch_progress:
            start_task_index = batch_progress['next_start_task']
            batch_number = batch_progress['next_batch']
            start_batch_num = start_task_index // batch_size
            print(f"\n【バッチ再開モード】")
            print(f"  前回完了: バッチ{batch_progress['current_batch']}, {batch_progress['completed_tasks']}タスク")
            print(f"  再開位置: バッチ{batch_progress['next_batch']}から開始（タスク{start_task_index}から）")

        # main.pyのrejection_reason_statsとcondition_rejection_statsを初期化
        from collections import defaultdict
        rejection_reason_stats = defaultdict(int)
        condition_rejection_stats = defaultdict(int)

        # 最大再生成試行回数
        max_regeneration_attempts = 1000  # main.pyと同じ

        # execution_resultsを初期化（main.pyと同じ）
        self.execution_results = []

        # 総バッチ数を計算
        total_tasks = num_programs
        num_batches = (total_tasks + batch_size - 1) // batch_size  # 切り上げ

        print(f"\nプログラム生成・実行を開始...")
        print(f"  総タスク数: {total_tasks}")
        print(f"  バッチ数: {num_batches} (1バッチあたり最大{batch_size}タスク)")
        if batch_progress and start_task_index > 0:
            print(f"  再開: バッチ{start_batch_num + 1}から開始（タスク{start_task_index + 1}から）")

        # リソース監視用の変数
        prev_threads_count = 0
        prev_memory_mb = 0

        # バッチごとに処理（main.pyと同じ構造）
        for batch_num in range(start_batch_num, num_batches):
            batch_start_idx = batch_num * batch_size
            batch_end_idx = min((batch_num + 1) * batch_size, total_tasks)
            batch_tasks = batch_end_idx - batch_start_idx

            # バッチ内の開始タスクインデックスを決定（再開時はstart_task_indexから）
            batch_internal_start = 0
            if batch_num == start_batch_num and start_task_index > 0:
                batch_internal_start = start_task_index - batch_start_idx

            # リソース監視（main.pyと同じ）
            threads_before = threading.active_count()
            threadpool_threads = []
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                memory_before = 0
            except Exception:
                memory_before = 0

            print(f"\n{'='*80}")
            print(f"【バッチ {batch_num + 1}/{num_batches} 処理開始】")
            print(f"  タスク範囲: {batch_start_idx + 1} ～ {batch_end_idx} ({batch_tasks}個)")
            print(f"  [リソース測定] スレッド数: {threads_before}, ThreadPoolExecutor系: {len(threadpool_threads)}, メモリ: {memory_before:.1f}MB")
            if batch_num > start_batch_num:
                thread_diff = threads_before - prev_threads_count
                memory_diff = memory_before - prev_memory_mb
                if thread_diff > 0:
                    print(f"    [警告] スレッド数が前回バッチから増加: +{thread_diff} (前回: {prev_threads_count} → 現在: {threads_before})", flush=True)
                if memory_diff > 50:
                    print(f"    [警告] メモリ使用量が前回バッチから大幅増加: +{memory_diff:.1f}MB (前回: {prev_memory_mb:.1f}MB → 現在: {memory_before:.1f}MB)", flush=True)
            print(f"{'='*80}")

            # ===== このバッチのタスクを生成 =====
            print(f"\n[バッチ {batch_num + 1} プログラム生成開始]")
            batch_program_gen_start = time.time()
            batch_programs = []  # main.pyと同じ形式: (nodes, program_code, filename, grid_width, grid_height)または(nodes, program_code, filename)
            batch_task_grid_sizes = []
            batch_complexities = []

            # このバッチで生成するタスクの範囲を決定
            for i in range(batch_start_idx, batch_end_idx):
                # 再開時は、開始位置より前のタスクはスキップ
                if i < start_task_index:
                    continue

                complexity = select_complexity(normalized_complexity_ratios)
                batch_complexities.append(complexity)

                grid_width, grid_height = None, None
                try:
                    # グリッドサイズを決定（プログラム生成前に決定）
                    grid_width, grid_height = decide_grid_size()
                    batch_task_grid_sizes.append((grid_width, grid_height))

                    # プログラム生成（環境変数で部分プログラムフローを制御可能、デフォルト: true）
                    use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true').lower() in ('true', '1', 'yes')
                    if use_partial_program_flow:
                        result = generate_program_with_partial_program_flow(generator, complexity, buffer_base_dir, i+1, grid_width, grid_height, self.buffer_manager, self.timing_stats)
                    else:
                        result = generate_program(generator, complexity, buffer_base_dir, i+1, grid_width, grid_height, self.buffer_manager, self.timing_stats)
                    batch_programs.append(result)
                except Exception as e:
                    print(f"[ERROR] タスク{i+1}の生成に失敗: {type(e).__name__}: {e}", flush=True)
                    if (grid_width, grid_height) not in batch_task_grid_sizes:
                        batch_task_grid_sizes.append((None, None))
                    # エラー時はNoneを追加してスキップ
                    batch_programs.append(None)

            batch_program_gen_elapsed = time.time() - batch_program_gen_start
            profiler.record_timing("batch_program_generation", batch_program_gen_elapsed, f"batch_{batch_num + 1}")
            print(f"[バッチ {batch_num + 1} プログラム生成完了] {len([p for p in batch_programs if p is not None])}/{len(batch_programs)}個のタスクを生成 (処理時間: {batch_program_gen_elapsed:.3f}秒)")

            # このバッチ内の破棄タスクリスト
            rejected_task_indices = []  # バッチ内でのローカルインデックス（1始まり）

            # 条件を満たしたタスクのIDセット（バッチ終了時にフィルタリング用）
            valid_task_ids = set()  # {"task_001", "task_002", ...} の形式

            # エラーカウンター（バッチごとにリセット）
            consecutive_errors = 0

            # タスクごとの処理時間を記録するリスト（バッチごとに初期化）
            # 注意: このリストはバッチごとにリセットされるため、全バッチの統計を表示するには
            # クラス変数として管理する必要がある
            if not hasattr(self, '_all_task_processing_times'):
                self._all_task_processing_times = []
            batch_task_processing_times = []

            # 初回実行ループ（このバッチのみ）
            # batch_internal_startから開始（再開時は該当位置から）
            for local_idx_offset, program_data in enumerate(batch_programs[batch_internal_start:], batch_internal_start + 1):
                local_idx = local_idx_offset
                global_idx = batch_start_idx + local_idx  # グローバルなタスクインデックス
                task_idx_global = start_task_index + global_idx if start_task_index > 0 else global_idx

                # タスク全体の処理時間を計測
                task_loop_start_time = time.time()

                # 100タスクごとにガベージコレクションを実行（定期クリーンアップ）
                if global_idx % 100 == 0:
                    gc_start = time.time()
                    gc.collect()
                    gc_time = time.time() - gc_start
                    profiler.record_timing("garbage_collection", gc_time, f"task_{global_idx}")
                    if gc_time > 0.1:
                        print(f"  [計測] 定期GC (タスク{global_idx}): {gc_time:.3f}秒", flush=True)

                # Noneの場合はスキップ
                if program_data is None:
                    print(f"  [警告] タスク{global_idx}: プログラムデータがNoneです。スキップします。", flush=True)
                    increment_rejection_stat(rejection_reason_stats, 'program_data_none')
                    rejected_task_indices.append(local_idx)
                    continue

                # データの展開（generate_program系の関数は5要素のタプルを返す）
                # 形式: (nodes, program_code, grid_width, grid_height, temporary_input_grid)
                # 注: filenameは含まれていないため、後で生成する
                if len(program_data) == 5:
                    nodes, program_code, grid_width_from_data, grid_height_from_data, temporary_input_grid = program_data
                    # grid_width, grid_heightがプログラム生成時に決定されている場合はそれを使用
                    if grid_width_from_data is not None and grid_height_from_data is not None:
                        grid_width, grid_height = grid_width_from_data, grid_height_from_data
                    elif local_idx - 1 < len(batch_task_grid_sizes):
                        grid_width, grid_height = batch_task_grid_sizes[local_idx - 1]
                    filename = f"task_{global_idx:03d}.json"
                else:
                    # 後方互換性のため、旧形式もサポート（通常は使用されない）
                    nodes, program_code, filename, grid_width, grid_height = program_data
                    temporary_input_grid = None

                # データの妥当性チェック
                if not nodes or len(nodes) == 0:
                    print(f"  [警告] タスク{global_idx}: プログラムノードが空です。スキップします。", flush=True)
                    increment_rejection_stat(rejection_reason_stats, 'nodes_empty')
                    rejected_task_indices.append(local_idx)
                    continue
                if not program_code or not program_code.strip():
                    print(f"  [警告] タスク{global_idx}: プログラムコードが空です。スキップします。", flush=True)
                    increment_rejection_stat(rejection_reason_stats, 'program_code_empty')
                    rejected_task_indices.append(local_idx)
                    continue

                # pairs_dataを初期化（tryブロックの外で定義することでスコープ問題を回避）
                pairs_data = []
                validated_nodes = None

                try:
                    execution_start_time = time.time()
                    task_processing_start = time.time()

                    # 複数ペア生成: 1つのプログラムから3-10個の入出力ペアを生成
                    num_pairs = random.randint(_generator_config.min_pairs_per_program, _generator_config.max_pairs_per_program)
                    print(f"タスク {global_idx}/{total_tasks} - 複数ペア生成開始 ({num_pairs}個のペア)", flush=True)

                    # すべてのペアで検証を有効化（ステップ3-4を実行）
                    # ただし、置き換え判定（ステップ5）はスキップ（後で全ペアの結果を使用して実行）
                    enable_replacement = False  # ステップ5はスキップ（後で全ペアの結果を使用）

                    # 共通関数でペア生成
                    # temporary_input_gridが提供されている場合は、最初のペアで使用される
                    pairs_data, validated_nodes = generate_pairs_with_retry(
                        nodes, num_pairs, grid_width, grid_height, global_idx,
                        generator, enable_replacement, is_regeneration=False,
                        temporary_input_grid=temporary_input_grid
                    )

                    # すべてのペアが生成できなかった場合、タスクを破棄
                    if len(pairs_data) == 0:
                        print(f"  [破棄] タスク{global_idx}: すべてのペアの生成に失敗しました", flush=True)
                        rejected_task_indices.append(local_idx)
                        continue

                    # ペア数が最小要件（MIN_PAIRS_PER_PROGRAM）未満の場合: タスク廃棄（学習データとして不十分）
                    if len(pairs_data) < _generator_config.min_pairs_per_program:
                        print(f"  [破棄] タスク{global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {_generator_config.min_pairs_per_program}個）", flush=True)
                        rejected_task_indices.append(local_idx)
                        continue

                    # 最初のペアの結果を使用
                    result_nodes, input_grid, output_grid, exec_wrapper_elapsed = extract_first_pair_data(pairs_data)

                    # 検証されたnodesを保存（最初のペアで検証されたnodes）
                    if validated_nodes:
                        result_nodes = validated_nodes

                    # None値チェックを強化
                    validate_execution_results(
                        result_nodes, input_grid, output_grid, global_idx, rejection_reason_stats, is_regeneration=False
                    )

                    # 入力グリッドと出力グリッドの妥当性チェック
                    validate_grid_data(
                        input_grid, output_grid, global_idx, rejection_reason_stats,
                        is_regeneration=False, raise_on_error=True
                    )

                    # 正常実行できた場合はエラーカウンターをリセット
                    consecutive_errors = 0

                    # 進捗メッセージを表示（タスクの処理時間も記録）
                    task_total_time = time.time() - task_loop_start_time
                    task_processing_time = time.time() - task_processing_start
                    profiler.record_timing("task_processing", task_processing_time, f"task_{global_idx}")
                    batch_task_processing_times.append(task_total_time)
                    self._all_task_processing_times.append(task_total_time)
                    print(f"タスク {global_idx}/{total_tasks} - 実行完了 (処理時間: {task_total_time:.2f}秒)", flush=True)

                except Exception as exec_error:
                    # SilentException（タイムアウト）を最優先で捕捉
                    exec_error_elapsed = time.time() - execution_start_time
                    total_task_error_elapsed = time.time() - task_loop_start_time
                    batch_task_processing_times.append(total_task_error_elapsed)  # エラー時も処理時間を記録
                    self._all_task_processing_times.append(total_task_error_elapsed)  # 全タスク統計にも追加
                    error_msg = str(exec_error).replace("SilentException: ", "").replace("src.core_systems.executor.core.SilentException: ", "")

                    if isinstance(exec_error, SilentException):
                        consecutive_errors += 1
                        print(f"  [エラー] タスク{global_idx} 実行エラー: {error_msg[:100]} (実行時間: {exec_error_elapsed:.3f}秒, タスク全体: {total_task_error_elapsed:.3f}秒)", flush=True)
                        increment_rejection_stat(rejection_reason_stats, 'silent_exception')
                        self.timing_stats.record_step("プログラム実行", 0, success=False)
                        rejected_task_indices.append(local_idx)
                        # execution_resultsに追加（main.pyと同じ）
                        self.execution_results.append((None, None, None, filename))
                        continue  # 即座に次のタスクへ

                    # ValueError（プログラム実行エラーによりタスク破棄）を優先処理
                    if isinstance(exec_error, ValueError) and "タスクが破棄されました" in str(exec_error):
                        execution_time = time.time() - execution_start_time
                        self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                        # オブジェクト数上限の場合は特別なメッセージを表示
                        if "オブジェクト数上限" in str(exec_error):
                            print(f"  [再生成対象] タスク{global_idx}: オブジェクト数が上限を超えたため、タスクを破棄して再生成します", flush=True)
                            increment_rejection_stat(rejection_reason_stats, 'object_count_limit')
                        else:
                            error_str = str(exec_error)
                            if "validate_nodes_and_adjust_objectsの結果がNone" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの結果がNoneによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_result_none')
                            elif "validate_nodes_and_adjust_objectsのタイムアウト" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsのタイムアウトによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_timeout')
                            elif "validate_nodes_and_adjust_objectsの例外" in error_str:
                                exception_detail = ""
                                if "例外: " in error_str:
                                    exception_detail = error_str.split("例外: ")[-1].rstrip("）")
                                if exception_detail:
                                    print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの例外によりタスクを破棄して再生成します（例外: {exception_detail}）", flush=True)
                                else:
                                    print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの例外によりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_exception')
                            else:
                                print(f"  [再生成対象] タスク{global_idx}: プログラム実行エラーによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'other_value_error')
                        rejected_task_indices.append(local_idx)
                        continue  # 即座に次のタスクへ（再生成ループに回す）
                except IndexError as index_error:
                        # 空のリストや範囲外アクセスのエラーは、プログラム生成の問題なので再生成対象とする
                        execution_time = time.time() - execution_start_time
                        error_msg = str(index_error)
                        if "空のリスト" in error_msg:
                            print(f"  [再生成対象] タスク{global_idx} リストアクセスエラー: {error_msg} → 再生成します", flush=True)
                            increment_rejection_stat(rejection_reason_stats, 'index_error')
                            self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                            rejected_task_indices.append(local_idx)
                            continue  # 再生成対象として次へ
                        else:
                            # その他のIndexErrorは致命的なエラーとして扱う
                            print(f"  [廃棄] タスク{global_idx} 実行中にプログラムエラーが発生: {type(index_error).__name__}: {index_error} → 即座に廃棄", flush=True)
                            self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                            rejected_task_indices.append(local_idx)
                        continue
                except (TypeError, AttributeError, KeyError, ZeroDivisionError) as exec_error:
                        # プログラム実行エラー（致命的なエラー）の場合は即座に廃棄
                        execution_time = time.time() - execution_start_time
                        print(f"  [廃棄] タスク{global_idx} 実行中にプログラムエラーが発生: {type(exec_error).__name__}: {exec_error} → 即座に廃棄", flush=True)
                        increment_rejection_stat(rejection_reason_stats, 'program_error')
                        self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                        rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                        # execution_resultsに追加（main.pyと同じ）
                        self.execution_results.append((None, None, None, filename))
                        continue  # 即座に廃棄して次へ
                except ValueError as validation_error:
                        # ValueError（プログラム実行エラーによりタスク破棄）を優先チェック
                        if "タスクが破棄されました" in str(validation_error):
                            execution_time = time.time() - execution_start_time
                            self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                            # オブジェクト数上限の場合は特別なメッセージを表示
                            if "オブジェクト数上限" in str(validation_error):
                                print(f"  [再生成対象] タスク{global_idx}: オブジェクト数が上限を超えたため、タスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'object_count_limit')
                            else:
                                error_str = str(validation_error)
                                if "validate_nodes_and_adjust_objectsの結果がNone" in error_str:
                                    print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの結果がNoneによりタスクを破棄して再生成します", flush=True)
                                    increment_rejection_stat(rejection_reason_stats, 'validate_result_none')
                                elif "validate_nodes_and_adjust_objectsのタイムアウト" in error_str:
                                    print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsのタイムアウトによりタスクを破棄して再生成します", flush=True)
                                    increment_rejection_stat(rejection_reason_stats, 'validate_timeout')
                                elif "validate_nodes_and_adjust_objectsの例外" in error_str:
                                    exception_detail = ""
                                    if "例外: " in error_str:
                                        exception_detail = error_str.split("例外: ")[-1].rstrip("）")
                                    if exception_detail:
                                        print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの例外によりタスクを破棄して再生成します（例外: {exception_detail}）", flush=True)
                                    else:
                                        print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの例外によりタスクを破棄して再生成します", flush=True)
                                    increment_rejection_stat(rejection_reason_stats, 'validate_exception')
                                else:
                                    print(f"  [再生成対象] タスク{global_idx}: プログラム実行エラーによりタスクを破棄して再生成します", flush=True)
                                    increment_rejection_stat(rejection_reason_stats, 'other_value_error')
                            rejected_task_indices.append(local_idx)
                            continue  # 再生成対象として次へ
                        # バリデーションエラー（条件該当等）の場合は再生成対象にする
                        execution_time = time.time() - execution_start_time
                        consecutive_errors += 1
                        print(f"  [再生成対象] タスク{global_idx} バリデーションエラー: {type(validation_error).__name__}: {validation_error} → 再生成します", flush=True)
                        increment_rejection_stat(rejection_reason_stats, 'validation_error')
                        self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                        rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                        continue  # 再生成対象として次へ
                except Exception as exec_error:
                        # その他の予期しないエラーの場合も再生成対象にする
                        execution_time = time.time() - execution_start_time
                        total_task_error_elapsed = time.time() - task_loop_start_time
                        consecutive_errors += 1
                        print(f"  [再生成対象] タスク{global_idx} 実行中に予期しないエラーが発生: {type(exec_error).__name__}: {exec_error} → 再生成します (実行時間: {execution_time:.3f}秒, タスク全体: {total_task_error_elapsed:.3f}秒)", flush=True)
                        increment_rejection_stat(rejection_reason_stats, 'other_error')
                        self.timing_stats.record_step("プログラム実行", execution_time, success=False)
                        rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                        # execution_resultsに追加（main.pyと同じ）
                        self.execution_results.append((None, None, None, filename))
                        continue  # 再生成対象として次へ

                # タスクフォルダーのパスを取得
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 複数ペアの条件チェック: すべてのペアが条件を満たしている場合のみ保存
                # 改善: すべてのペアで条件に該当する場合のみタスク廃棄、一部のペアのみの場合はペアスキップ
                if _generator_config.enable_output_condition_check:
                    invalid_pairs, valid_pairs_data = check_all_pairs_conditions(
                        pairs_data, global_idx, condition_rejection_stats, rejection_reason_stats, is_regeneration=False
                    )

                    # すべてのペアで条件に該当する場合: タスク廃棄（プログラムの問題）
                    if len(invalid_pairs) == len(pairs_data):
                        print(f"  [破棄] タスク{global_idx}: すべてのペアが条件に該当するためタスクを破棄します（プログラムの問題）", flush=True)
                        rejected_task_indices.append(local_idx)
                        continue

                    # 一部のペアのみ条件に該当する場合: そのペアをスキップ（入力グリッドの問題）
                    elif len(invalid_pairs) > 0:
                        print(f"  [スキップ] タスク{global_idx}: {len(invalid_pairs)}個のペアが条件に該当するため、それらのペアをスキップします（入力グリッドの問題）", flush=True)
                        # 条件に該当するペアをpairs_dataから削除
                        pairs_data = valid_pairs_data

                        # すべてのペアがスキップされた場合: タスク廃棄
                        if len(pairs_data) == 0:
                            print(f"  [破棄] タスク{global_idx}: すべてのペアがスキップされたためタスクを破棄します", flush=True)
                            rejected_task_indices.append(local_idx)
                            continue

                        # ペア数が最小要件（MIN_PAIRS_PER_PROGRAM）未満の場合: タスク廃棄（学習データとして不十分）
                        if len(pairs_data) < _generator_config.min_pairs_per_program:
                            print(f"  [破棄] タスク{global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {_generator_config.min_pairs_per_program}個）", flush=True)
                            rejected_task_indices.append(local_idx)
                            continue

                        print(f"  [続行] タスク{global_idx}: {len(pairs_data)}個のペアが残りました（スキップ: {len(invalid_pairs)}個）", flush=True)

                    # 条件に該当しない場合：有効なタスクとして記録
                    valid_task_ids.add(f"task_{global_idx:03d}")

                    # ステップ5: スコープ削除検証 → コマンド置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                    # すべてのペアの結果を使用して置き換え判定を実行
                    if validated_nodes and len(pairs_data) > 0:
                        try:
                            # 置き換え判定用のexecutorを準備
                            replacement_executor = CoreExecutor(preserve_indentation=True)

                            # プログラムコードを生成（置き換え判定で使用）
                            task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                            context_for_replacement = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                            try:
                                program_code_for_replacement = generator._generate_code(validated_nodes, context_for_replacement, preserve_indentation=True)
                            except TypeError:
                                program_code_for_replacement = generator._generate_code(validated_nodes, context_for_replacement)

                            # ステップ5-1: スコープ削除検証（一番深いスコープから順に）
                            updated_nodes_for_scope = validated_nodes
                            scope_pairs = find_scope_pairs(validated_nodes)

                            # 一番深いスコープから順に検証
                            for scope_info in scope_pairs:
                                # スコープを削除したノードリストを作成
                                nodes_without_scope = remove_scope_from_nodes(updated_nodes_for_scope, scope_info)

                                # スコープ削除後のプログラムコードを生成
                                try:
                                    program_code_without_scope = replacement_executor.program_generator._generate_code(
                                        nodes_without_scope, context_for_replacement, preserve_indentation=True
                                    )
                                except (TypeError, Exception):
                                    try:
                                        program_code_without_scope = replacement_executor.program_generator._generate_code(
                                            nodes_without_scope, context_for_replacement
                                        )
                                    except Exception:
                                        continue

                                # すべてのペアで条件チェックを実行
                                all_pairs_unnecessary_scope = True

                                for pair_data in pairs_data:
                                    pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                    pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                    try:
                                        # 元のコードの出力（既に取得済み）
                                        output_grid_original = pair_output_grid

                                        # スコープ削除後のコードを実行
                                        try:
                                            output_grid_without_scope, _, _, _ = replacement_executor.execute_program_string(
                                                program_code=program_code_without_scope,
                                                input_grid=pair_input_grid,
                                                input_objects=None,
                                                input_image_index=0,
                                                background_color=None
                                            )
                                        except Exception:
                                            all_pairs_unnecessary_scope = False
                                            break

                                        if output_grid_without_scope is None:
                                            all_pairs_unnecessary_scope = False
                                            break

                                        # スコープ削除前のoutput_gridが条件に該当するかチェック
                                        input_unique, input_counts = np.unique(pair_input_grid, return_counts=True)
                                        inferred_background_color = input_unique[np.argmax(input_counts)] if len(input_unique) > 0 else None

                                        condition1_original, condition2_original, condition3_original, condition4_original, condition5_original, condition6_original = check_output_conditions(
                                            pair_input_grid, output_grid_original, background_color=inferred_background_color
                                        )
                                        original_meets_condition = condition1_original or condition2_original or condition3_original or condition5_original or condition6_original

                                        # スコープ削除前後のoutput_gridが完全一致かチェック
                                        outputs_match = np.array_equal(output_grid_original, output_grid_without_scope)

                                        # original_meets_conditionがFalseの場合、スコープ削除前後の完全一致が必要
                                        if not original_meets_condition:
                                            if not outputs_match:
                                                # このペアでは不要と判断されない
                                                all_pairs_unnecessary_scope = False
                                                break

                                    except Exception:
                                        all_pairs_unnecessary_scope = False
                                        break

                                # すべてのペアで不要と判断された場合のみスコープ削除を適用
                                if all_pairs_unnecessary_scope:
                                    # スコープ削除後のプログラムにGET_ALL_OBJECTSが含まれているか確認
                                    if 'GET_ALL_OBJECTS' not in program_code_without_scope:
                                        continue

                                    # すべてのペアで検証（エラーが発生しないことを確認）
                                    scope_removal_valid = True
                                    for pair_data in pairs_data:
                                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                        try:
                                            test_output_grid, _, test_execution_time, _ = replacement_executor.execute_program_string(
                                                program_code=program_code_without_scope,
                                                input_grid=pair_input_grid,
                                                input_objects=None,
                                                input_image_index=0,
                                                background_color=None
                                            )

                                            # エラー検出: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致している場合はエラー
                                            if test_execution_time == 0.0 and np.array_equal(pair_input_grid, test_output_grid):
                                                scope_removal_valid = False
                                                break
                                        except Exception:
                                            scope_removal_valid = False
                                            break

                                    # 検証が成功した場合のみ、スコープ削除を適用
                                    if scope_removal_valid:
                                        updated_nodes_for_scope = nodes_without_scope
                                        # 一度スコープ削除を行ったら、次のスコープの検証には更新後のnodesを使用
                                        # インデックスが変わるため、スコープペアを再計算
                                        scope_pairs = find_scope_pairs(updated_nodes_for_scope)

                            # スコープ削除後のnodesをvalidated_nodesに反映
                            validated_nodes = updated_nodes_for_scope

                            # ステップ5-2: コマンド置き換え判定
                            # 置き換え対象のコマンド情報を収集
                            commands_to_check = collect_commands_to_check(validated_nodes)

                            # 各コマンドを検証して置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                            updated_nodes = validated_nodes
                            for cmd_info in commands_to_check:
                                node_index = cmd_info['node_index']
                                cmd_name = cmd_info['command']
                                cmd_node = cmd_info['node']

                                # 置き換え候補を取得
                                nodes_after_current = updated_nodes[node_index+1:]
                                replace_result = replace_command_with_fallback(
                                    cmd_name, cmd_node, updated_nodes[:node_index+1],
                                    grid_width=grid_width, grid_height=grid_height
                                )

                                if replace_result is None:
                                    # 置き換え候補がない場合はスキップ
                                    continue

                                nodes_replaced_up_to_current, fallback_code = replace_result
                                nodes_with_fallback = nodes_replaced_up_to_current + nodes_after_current

                                # 置き換え後のプログラムコードを生成
                                if replacement_executor.program_generator is None:
                                    continue

                                try:
                                    context_fallback = ProgramContext(complexity=1)
                                    try:
                                        program_code_fallback = replacement_executor.program_generator._generate_code(
                                            nodes_with_fallback, context_fallback, preserve_indentation=True
                                        )
                                    except TypeError:
                                        program_code_fallback = replacement_executor.program_generator._generate_code(
                                            nodes_with_fallback, context_fallback
                                        )
                                except Exception:
                                    continue

                                # すべてのペアで条件チェックを実行
                                all_pairs_unnecessary = True

                                for pair_data in pairs_data:
                                    pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                    pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                    try:
                                        # 元のコードの出力（既に取得済み）
                                        output_grid_original = pair_output_grid

                                        # 置き換え後のコードを実行
                                        try:
                                            output_grid_fallback, _, _, _ = replacement_executor.execute_program_string(
                                                program_code=program_code_fallback,
                                                input_grid=pair_input_grid,
                                                input_objects=None,
                                                input_image_index=0,
                                                background_color=None
                                            )
                                        except Exception:
                                            all_pairs_unnecessary = False
                                            break

                                        if output_grid_fallback is None:
                                            all_pairs_unnecessary = False
                                            break

                                        # 置き換え前のoutput_gridが条件に該当するかチェック
                                        # 入力グリッドから背景色を推論
                                        input_unique, input_counts = np.unique(pair_input_grid, return_counts=True)
                                        inferred_background_color = input_unique[np.argmax(input_counts)] if len(input_unique) > 0 else None

                                        condition1_original, condition2_original, condition3_original, condition4_original, condition5_original, condition6_original = check_output_conditions(
                                            pair_input_grid, output_grid_original, background_color=inferred_background_color
                                        )
                                        original_meets_condition = condition1_original or condition2_original or condition3_original or condition5_original or condition6_original

                                        # 置き換え前後のoutput_gridが完全一致かチェック
                                        outputs_match = np.array_equal(output_grid_original, output_grid_fallback)

                                        # original_meets_conditionがFalseの場合、置き換え前後の完全一致が必要
                                        if not original_meets_condition:
                                            if not outputs_match:
                                                # このペアでは不要と判断されない
                                                all_pairs_unnecessary = False
                                                break

                                        # すべてのペアで条件チェックを続行

                                    except Exception:
                                        # エラーが発生した場合はスキップ
                                        all_pairs_unnecessary = False
                                        break

                                # すべてのペアで不要と判断された場合のみ置き換えを適用
                                if all_pairs_unnecessary:
                                    # 置き換え後のプログラムにGET_ALL_OBJECTSが含まれているか確認
                                    try:
                                        if 'GET_ALL_OBJECTS' not in program_code_fallback:
                                            # GET_ALL_OBJECTSがない場合はスキップ
                                            continue

                                        # すべてのペアで検証（エラーが発生しないことを確認）
                                        replacement_valid = True
                                        for pair_data in pairs_data:
                                            pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                            try:
                                                test_output_grid, _, test_execution_time, _ = replacement_executor.execute_program_string(
                                                    program_code=program_code_fallback,
                                                    input_grid=pair_input_grid,
                                                    input_objects=None,
                                                    input_image_index=0,
                                                    background_color=None
                                                )

                                                # エラー検出: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致している場合はエラー
                                                if test_execution_time == 0.0 and np.array_equal(pair_input_grid, test_output_grid):
                                                    replacement_valid = False
                                                    break
                                            except Exception:
                                                replacement_valid = False
                                                break

                                        # 検証が成功した場合のみ、置き換えを適用
                                        if replacement_valid:
                                            updated_nodes = nodes_with_fallback
                                            result_nodes = updated_nodes
                                            validated_nodes = updated_nodes
                                            # 一度置き換えを行ったら、次のコマンドの検証には更新後のnodesを使用

                                    except Exception:
                                        # エラーが発生した場合はスキップ
                                        continue

                            # 置き換えが行われた場合、result_nodesを更新
                            if updated_nodes is not validated_nodes:
                                result_nodes = updated_nodes
                                # 最初のペアのデータも更新（既存コードとの互換性のため）
                                if len(pairs_data) > 0:
                                    pairs_data[0]['nodes'] = updated_nodes

                        except Exception as replacement_error:
                            # 置き換え判定でエラーが発生した場合は、元のnodesを使用
                            pass

                    # すべてのペアのグリッドデータをバッファに追加（条件を満たしたタスクのみ）
                    # 注: バッチ終了時にフィルタリングするため、ここでは追加のみ

                    # PNGデータはすべてのペアを追加（複数ペア対応）
                    # すべてのペア生成後に、すべてのペアデータを渡す
                    _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                    first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                    first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                    self.buffer_manager.add_png_data(
                        task_index=global_idx,
                        input_grid=first_pair_input_grid,
                        output_grid=first_pair_output_grid,
                        timestamp=timestamp,
                        pairs_data=pairs_data  # すべてのペアデータを渡す
                    )

                    # JSONデータはすべてのペアを追加（複数ペア対応）
                    for pair_data in pairs_data:
                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                        pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                        trace_results = pair_data.get('trace_results')
                        self.buffer_manager.add_grid_json(
                            task_index=global_idx,
                            input_grid=pair_input_grid,
                            output_grid=pair_output_grid,
                            trace_results=trace_results
                        )

                    # プログラムJSONを追加（txtファイル生成に使用）
                    # 注: 最初のペアが生成された時点で、プログラムJSONを追加する必要がある
                    # 複雑度を取得（batch_complexitiesから）
                    task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                    program_json_data = {
                        "task_id": f"task_{global_idx:03d}",
                        "timestamp": timestamp,
                        "complexity": task_complexity,
                        "grid_size": {
                            "width": grid_width,
                            "height": grid_height
                        },
                        "program_code": program_code,
                        "program_length": len(program_code),
                        "node_count": len(nodes) if nodes else 0,
                        "statistics": {
                            "line_count": program_code.count('\n') + 1,
                            "character_count": len(program_code),
                            "word_count": len(program_code.split())
                        }
                    }
                    self.buffer_manager.add_program_json(global_idx, program_json_data)

                    # トークンJSONを追加
                    tokens = tokenize_program(program_code)
                    tokens_data = {
                        "task_id": f"task_{global_idx:03d}",
                        "timestamp": timestamp,
                        "complexity": task_complexity,
                        "grid_size": {
                            "width": grid_width,
                            "height": grid_height
                        },
                        "tokens": tokens,
                        "token_count": len(tokens),
                        "vocabulary": list(set(tokens))
                    }
                    self.buffer_manager.add_tokens_json(global_idx, tokens_data)

                    # 統計JSONを追加
                    stats = analyze_program_statistics(program_code, nodes if nodes else [])
                    stats_data = {
                        "task_id": f"task_{global_idx:03d}",
                        "timestamp": timestamp,
                        "complexity": task_complexity,
                        "grid_size": {
                            "width": grid_width,
                            "height": grid_height
                        },
                        "statistics": stats
                    }
                    self.buffer_manager.add_stats_json(global_idx, stats_data)

                    # 置き換え後のnodesからprogram_codeを再生成して更新
                    if result_nodes:
                        try:
                            # 複雑度を取得（batch_complexitiesから）
                            task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                            context = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                            updated_program_code = generate_code(result_nodes, context)

                            # 変数インデックスを正規化
                            updated_program_code = normalize_variable_indices(updated_program_code)

                            # GET_ALL_OBJECTSが含まれているか確認
                            if 'GET_ALL_OBJECTS' not in updated_program_code:
                                print(f"  [破棄] タスク{global_idx}: プログラムコードにGET_ALL_OBJECTSが含まれていないため破棄", flush=True)
                                rejected_task_indices.append(local_idx)
                                continue

                            # batch_programsのprogram_codeを更新
                            if len(batch_programs[local_idx - 1]) == 5:
                                batch_programs[local_idx - 1] = (result_nodes, updated_program_code, filename, grid_width, grid_height)
                            else:
                                batch_programs[local_idx - 1] = (result_nodes, updated_program_code, filename)
                            # JSONバッファのprogram_codeも更新
                            if self.buffer_manager:
                                self.buffer_manager.update_program_json(global_idx, updated_program_code)
                            program_code = updated_program_code  # 後で使用するため更新
                        except Exception as e:
                            if self.enable_debug_logs:
                                print(f"[DEBUG] タスク{global_idx} (updated_program_code): 例外発生: {type(e).__name__}: {e}", flush=True)
                            pass

                    # コマンド使用統計に追加
                    if result_nodes:
                        commands = extract_all_commands_from_nodes(result_nodes)
                        statistics.add_program_commands(commands)

                    # 1000タスクごとに重み調整を適用・保存
                    if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                        adjustments = statistics.get_weight_adjustments()
                        if adjustments:
                            generator.command_weight_adjustments = adjustments
                            print(f"\n【1000タスク到達: 重み調整を適用】")
                            print(f"  調整対象: {len(adjustments)}個のコマンド")
                            statistics.print_statistics()

                            # 重み調整をファイルに保存
                            save_weight_adjustments(buffer_base_dir, adjustments, statistics.total_programs)

                            # バッチ進捗を保存
                            current_batch = statistics.total_programs // BATCH_SIZE
                            save_batch_progress(buffer_base_dir, statistics.total_programs, current_batch)

                    # ニューラルモデル用学習データを生成（有効なタスクのみ）
                    if f"task_{global_idx:03d}" in valid_task_ids:
                        try:
                            # 置き換え後のprogram_codeを取得
                            task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                            context = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                            final_program_code = generate_code(result_nodes, context)

                            # 変数インデックスを正規化
                            final_program_code = normalize_variable_indices(final_program_code)

                            # 各ペアごとに学習データを生成
                            for pair_idx, pair_data in enumerate(pairs_data):
                                pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                self.neural_data_generator.generate_from_generator_output(
                                    task_id=f"task_{global_idx:03d}",
                                    program_code=final_program_code,
                                    input_grid=pair_input_grid,
                                    output_grid=pair_output_grid,
                                    nodes=result_nodes,
                                    complexity=task_complexity,
                                    pair_index=pair_idx
                                )
                        except Exception as e:
                            # 学習データ生成でエラーが発生しても処理を継続
                            pass

                    # 成功したペアをDataPairオブジェクトに変換して追加
                    for pair_data in pairs_data:
                        pair = DataPair(
                            input=pair_data['input_grid'].tolist(),
                            output=pair_data['output_grid'].tolist(),
                            program=program_code,
                            metadata={'generated_by': 'hybrid_learning_pipeline', 'generation_attempt': 1}
                        )
                        data_pairs.append(pair)
                        batch_data_pairs.append(pair)
                        self.task_counter += 1

                    # 保存完了メッセージを表示
                    print(f"タスク {global_idx}/{total_tasks} - 保存完了", flush=True)

                    # execution_resultsに追加（main.pyと同じ - tryブロック内）
                    self.execution_results.append((result_nodes, input_grid, output_grid, filename))
                else:
                    # 条件チェックが無効な場合、すべて有効として扱う（複数ペア対応）
                    valid_task_ids.add(f"task_{global_idx:03d}")
                    print(f"  [デバッグ] タスク{global_idx}: 条件チェック無効 - valid_task_idsに追加しました (現在の数: {len(valid_task_ids)})", flush=True)

                    # PNGデータはすべてのペアを追加（複数ペア対応）
                    # すべてのペア生成後に、すべてのペアデータを渡す
                    _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                    first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                    first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                    self.buffer_manager.add_png_data(
                        task_index=global_idx,
                        input_grid=first_pair_input_grid,
                        output_grid=first_pair_output_grid,
                        timestamp=timestamp,
                        pairs_data=pairs_data  # すべてのペアデータを渡す
                    )

                    # JSONデータはすべてのペアを追加（複数ペア対応）
                    for pair_data in pairs_data:
                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                        pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                        trace_results = pair_data.get('trace_results')
                        self.buffer_manager.add_grid_json(
                            task_index=global_idx,
                            input_grid=pair_input_grid,
                            output_grid=pair_output_grid,
                            trace_results=trace_results
                        )

                    # 置き換え後のnodesからprogram_codeを再生成して更新
                    if result_nodes:
                        try:
                            # 複雑度を取得（batch_complexitiesから）
                            task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                            context = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                            try:
                                updated_program_code = generator._generate_code(result_nodes, context, preserve_indentation=True)
                            except TypeError:
                                updated_program_code = generator._generate_code(result_nodes, context)

                            # GET_ALL_OBJECTSが含まれているか確認
                            if 'GET_ALL_OBJECTS' not in updated_program_code:
                                print(f"  [破棄] タスク{global_idx}: プログラムコードにGET_ALL_OBJECTSが含まれていないため破棄", flush=True)
                                rejected_task_indices.append(local_idx)
                                continue

                            # batch_programsのprogram_codeを更新
                            if len(batch_programs[local_idx - 1]) == 5:
                                batch_programs[local_idx - 1] = (result_nodes, updated_program_code, filename, grid_width, grid_height)
                            else:
                                batch_programs[local_idx - 1] = (result_nodes, updated_program_code, filename)
                            # JSONバッファのprogram_codeも更新
                            if self.buffer_manager:
                                self.buffer_manager.update_program_json(global_idx, updated_program_code)
                            program_code = updated_program_code  # 後で使用するため更新

                            # ニューラルモデル用学習データを生成（条件チェック無効の場合）
                            try:
                                # 各ペアごとに学習データを生成
                                for pair_idx, pair_data in enumerate(pairs_data):
                                    pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                    pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                    self.neural_data_generator.generate_from_generator_output(
                                        task_id=f"task_{global_idx:03d}",
                                        program_code=updated_program_code,
                                        input_grid=pair_input_grid,
                                        output_grid=pair_output_grid,
                                        nodes=result_nodes,
                                        complexity=task_complexity,
                                        pair_index=pair_idx
                                    )
                            except Exception as e:
                                # 学習データ生成でエラーが発生しても処理を継続
                                pass
                        except Exception as e:
                            pass

                    # コマンド使用統計に追加
                    if result_nodes:
                        commands = extract_all_commands_from_nodes(result_nodes)
                        statistics.add_program_commands(commands)
                        print(f"  [デバッグ] タスク{global_idx}: 条件チェック無効 - statistics.add_program_commandsを呼び出しました (total_programs: {statistics.total_programs})", flush=True)

                        # 1000タスクごとに重み調整を適用・保存
                        if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                            adjustments = statistics.get_weight_adjustments()
                            if adjustments:
                                generator.command_weight_adjustments = adjustments
                                print(f"\n【1000タスク到達: 重み調整を適用】")
                                print(f"  調整対象: {len(adjustments)}個のコマンド")
                                statistics.print_statistics()

                                # 重み調整をファイルに保存
                                save_weight_adjustments(buffer_base_dir, adjustments, statistics.total_programs)

                                # バッチ進捗を保存
                                current_batch = statistics.total_programs // BATCH_SIZE
                                save_batch_progress(buffer_base_dir, statistics.total_programs, current_batch)
                    else:
                        print(f"  [警告] タスク{global_idx}: 条件チェック無効 - result_nodesがNoneのため、statisticsに追加しませんでした", flush=True)

                    # 成功したペアをDataPairオブジェクトに変換して追加
                    for pair_data in pairs_data:
                        pair = DataPair(
                            input=pair_data['input_grid'].tolist(),
                            output=pair_data['output_grid'].tolist(),
                            program=program_code,
                            metadata={'generated_by': 'hybrid_learning_pipeline', 'generation_attempt': 1}
                        )
                        data_pairs.append(pair)
                        batch_data_pairs.append(pair)
                        self.task_counter += 1

                    # 保存完了メッセージを表示
                    print(f"タスク {global_idx}/{total_tasks} - 保存完了", flush=True)

                    # execution_resultsに追加（main.pyと同じ - tryブロック内）
                    self.execution_results.append((result_nodes, input_grid, output_grid, filename))

            # このバッチ内の破棄されたタスクを再生成（ループで繰り返し）
            if rejected_task_indices:
                print(f"\n[バッチ {batch_num + 1} 初回ループ完了] {len(rejected_task_indices)}個のタスクが条件に該当したため破棄されました")
                print(f"  破棄されたタスク（バッチ内ローカルインデックス）: {rejected_task_indices}")

            regeneration_attempt = 0
            # 連続エラーが発生した場合の早期終了カウンター（無効化: 早期終了しない）
            consecutive_regen_failures = 0
            max_consecutive_regen_failures = float('inf')  # 早期終了を無効化（無限大に設定）

            while rejected_task_indices and regeneration_attempt < max_regeneration_attempts:
                regeneration_attempt += 1
                loop_start_time = time.time()
                print(f"\n[バッチ {batch_num + 1} 再生成ループ {regeneration_attempt}] 破棄された{len(rejected_task_indices)}個のタスクを再生成・実行します", flush=True)
                rejected_task_indices.sort()
                new_rejected_indices = []  # このループで新たに破棄されたタスク

                # 前回のループですべて失敗した場合の早期終了（無効化: 早期終了しない）
                if max_consecutive_regen_failures != float('inf') and consecutive_regen_failures >= max_consecutive_regen_failures:
                    print(f"  [警告] 再生成が{consecutive_regen_failures}回連続で失敗したため、早期終了します（処理速度向上のため）", flush=True)
                    print(f"  残りの破棄タスク数: {len(rejected_task_indices)}")
                    print(f"  これらのタスクは条件を満たさなかったため、保存されません")
                    break

                # 最大再生成試行回数に近づいている場合は警告
                if regeneration_attempt >= max_regeneration_attempts * 0.8:
                    print(f"  [警告] 再生成ループ {regeneration_attempt}/{max_regeneration_attempts}: 最大試行回数に近づいています", flush=True)

                # 再生成フェーズと実行フェーズを統合して、1つのループで処理（処理速度向上のため）
                regen_start = time.time()
                total_regen_tasks = len(rejected_task_indices)
                print(f"  → 再生成ループ {regeneration_attempt}: {total_regen_tasks}個のタスクを処理します", flush=True)

                for task_idx, rejected_local_idx in enumerate(rejected_task_indices, 1):
                    rejected_global_idx = batch_start_idx + rejected_local_idx  # グローバルインデックスに変換
                    task_start_time = time.time()
                    if task_idx % 10 == 0 or task_idx == 1:
                        print(f"  → 再生成ループ {regeneration_attempt} 進捗: {task_idx}/{total_regen_tasks} (タスク{rejected_global_idx})", flush=True)

                    # ステップ1: プログラム再生成
                    regen_gen_start_time = time.time()
                    try:
                        # 再生成時は必ず新しいパラメータをランダムに選択（同じパラメータで再生成すると同じようなプログラムが生成されるため）
                        # 元の複雑度とは異なる複雑度を選択
                        original_complexity = batch_complexities[rejected_local_idx - 1] if (rejected_local_idx - 1) < len(batch_complexities) else 1
                        available_complexities = [c for c in normalized_complexity_ratios.keys() if c != original_complexity]
                        if available_complexities:
                            new_complexity = random.choice(available_complexities)
                        else:
                            # 利用可能な複雑度がない場合は、比率に基づいてランダム選択
                            new_complexity = select_complexity(normalized_complexity_ratios)

                        # 新しいグリッドサイズをランダムに決定（通常の決定ロジックを使用）
                        grid_width, grid_height = decide_grid_size()

                        print(f"  [再生成] タスク{rejected_global_idx}: パラメータを変更 (複雑度: {original_complexity}→{new_complexity}, サイズ: {grid_width}x{grid_height})", flush=True)
                        regen_complexity = new_complexity

                        # 再生成（タイムアウトやエラーが発生した場合はスキップ）
                        try:
                            generate_start_time = time.time()
                            # 再生成時も部分プログラムフローを使用可能（環境変数で制御）
                            use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true').lower() in ('true', '1', 'yes')
                            if use_partial_program_flow:
                                result = generate_program_with_partial_program_flow(generator, regen_complexity, buffer_base_dir, rejected_global_idx, grid_width, grid_height, self.buffer_manager, self.timing_stats)
                            else:
                                result = generate_program(generator, regen_complexity, buffer_base_dir, rejected_global_idx, grid_width, grid_height, self.buffer_manager, self.timing_stats)
                            generate_elapsed = time.time() - generate_start_time
                            batch_programs[rejected_local_idx - 1] = result  # バッチ内のリストを更新
                            # 再生成時のパラメータを保存（次回の再生成で使用）
                            batch_complexities[rejected_local_idx - 1] = regen_complexity
                            batch_task_grid_sizes[rejected_local_idx - 1] = (grid_width, grid_height)
                        except Exception as regen_error:
                            error_type = type(regen_error).__name__
                            error_msg = str(regen_error)
                            # 致命的なエラーは即座に破棄
                            if error_type == 'AttributeError' and 'generate' in error_msg:
                                continue  # 再生成を試みない
                            new_rejected_indices.append(rejected_local_idx)
                            continue  # 次のタスクへ
                    except Exception as e:
                        new_rejected_indices.append(rejected_local_idx)
                        continue  # 次のタスクへ

                    # ステップ2: 再生成されたプログラムを即座に実行（2フェーズに分けない）
                    program_data = batch_programs[rejected_local_idx - 1]

                    # Noneの場合はスキップ（再生成ループに回す）
                    if program_data is None:
                        print(f"  [再生成失敗] タスク{rejected_global_idx}: プログラムデータがNoneです。再生成ループに回します。", flush=True)
                        new_rejected_indices.append(rejected_local_idx)
                        continue

                    # データの展開（generate_program系の関数は5要素のタプルを返す）
                    # 形式: (nodes, program_code, grid_width, grid_height, temporary_input_grid)
                    if len(program_data) == 5:
                        nodes, program_code, grid_width_from_data, grid_height_from_data, temporary_input_grid = program_data
                        # grid_width, grid_heightがプログラム生成時に決定されている場合はそれを使用
                        if grid_width_from_data is not None and grid_height_from_data is not None:
                            grid_width, grid_height = grid_width_from_data, grid_height_from_data
                        elif rejected_local_idx - 1 < len(batch_task_grid_sizes):
                            grid_width, grid_height = batch_task_grid_sizes[rejected_local_idx - 1]
                        filename = f"task_{rejected_global_idx:03d}.json"
                    else:
                        # 後方互換性のため、旧形式もサポート（通常は使用されない）
                        nodes, program_code, filename, grid_width, grid_height = program_data
                        temporary_input_grid = None

                    # 再生成データの妥当性チェック
                    if not nodes or len(nodes) == 0 or not program_code or not program_code.strip():
                        print(f"  [再生成失敗] タスク{rejected_global_idx}: プログラムノードまたはコードが空です。再生成ループに回します。", flush=True)
                        new_rejected_indices.append(rejected_local_idx)
                        continue  # 次のタスクへ

                    # ステップ2: 再生成されたプログラムを即座に実行（2フェーズに分けない = 処理速度向上）
                    regen_gen_elapsed = time.time() - regen_gen_start_time
                    regen_exec_start = time.time()
                    try:
                        # 複数ペア生成: 1つのプログラムから3-10個の入出力ペアを生成
                        num_pairs = random.randint(_generator_config.min_pairs_per_program, _generator_config.max_pairs_per_program)
                        print(f"再生成タスク {rejected_global_idx}/{total_tasks} - 複数ペア生成開始 ({num_pairs}個のペア)", flush=True)

                        # すべてのペアで検証を有効化（ステップ3-4を実行）
                        # ただし、置き換え判定（ステップ5）はスキップ（後で全ペアの結果を使用して実行）
                        enable_replacement = False  # ステップ5はスキップ（後で全ペアの結果を使用）

                        # 共通関数でペア生成
                        # 再生成時もtemporary_input_gridを使用可能（部分プログラムフロー使用時）
                        pairs_data, validated_nodes = generate_pairs_with_retry(
                            nodes, num_pairs, grid_width, grid_height, rejected_global_idx,
                            generator, enable_replacement, is_regeneration=True,
                            temporary_input_grid=temporary_input_grid  # 再生成時もtemporary_input_gridを使用可能
                        )

                        # すべてのペアが生成できなかった場合、タスクを破棄
                        if len(pairs_data) == 0:
                            print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: すべてのペアの生成に失敗しました", flush=True)
                            new_rejected_indices.append(rejected_local_idx)
                            continue

                        # ペア数が最小要件（MIN_PAIRS_PER_PROGRAM）未満の場合: タスク廃棄（学習データとして不十分）
                        if len(pairs_data) < _generator_config.min_pairs_per_program:
                            print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {_generator_config.min_pairs_per_program}個）", flush=True)
                            new_rejected_indices.append(rejected_local_idx)
                            continue

                        # 最初のペアの結果を使用
                        first_pair = pairs_data[0]
                        result_nodes = first_pair['nodes']
                        input_grid = first_pair['input_grid']
                        output_grid = first_pair['output_grid']
                        exec_elapsed = first_pair['execution_time']

                        # 検証されたnodesを保存（最初のペアで検証されたnodes）
                        if validated_nodes:
                            result_nodes = validated_nodes

                        # None値チェックを強化
                        validate_execution_results(
                            result_nodes, input_grid, output_grid, rejected_global_idx, rejection_reason_stats, is_regeneration=True
                        )

                        # 進捗メッセージを表示（重要なログなので常に表示）
                        print(f"再生成タスク {rejected_global_idx}/{total_tasks} - 複数ペア実行完了 ({len(pairs_data)}個のペア)", flush=True)
                        regen_exec_time = time.time() - regen_exec_start

                        # 結果の妥当性チェック（既にvalidate_execution_resultsでチェック済み、ここではグリッドデータの妥当性のみチェック）

                        # 入力グリッドと出力グリッドの妥当性チェック
                        if not validate_grid_data(
                            input_grid, output_grid, rejected_global_idx, rejection_reason_stats,
                            is_regeneration=True, raise_on_error=False
                        ):
                            new_rejected_indices.append(rejected_local_idx)
                            self.timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                            continue  # 次のタスクへ

                        self.timing_stats.record_step("再生成実行", regen_exec_time, success=True)
                    except Exception as exec_error:
                        # SilentException（タイムアウト）を最優先で捕捉
                        if isinstance(exec_error, SilentException):
                            regen_exec_time = time.time() - regen_exec_start
                            # SilentExceptionの場合は最小限の処理のみ
                            increment_rejection_stat(rejection_reason_stats, 'silent_exception')
                            self.timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                            new_rejected_indices.append(rejected_local_idx)
                            continue  # 即座に次のタスクへ
                        # ValueError（プログラム実行エラーによりタスク破棄、または異常に長い実行時間）を優先処理
                        if isinstance(exec_error, ValueError):
                            error_msg = str(exec_error)
                            regen_exec_time = time.time() - regen_exec_start
                            self.timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                            # 異常に長い実行時間で破棄された場合は、再生成を中止（同じ問題が繰り返される可能性が高いため）
                            if "実行時間が異常に長い" in error_msg:
                                print(f"  [破棄確定] 再生成タスク{rejected_global_idx} は実行時間が異常に長いため、再生成を中止します。", flush=True)
                                # 再生成を中止（new_rejected_indicesに追加しない = これ以上再生成しない）
                                continue  # 即座に次のタスクへ（このタスクは再生成対象から除外）
                            # "実行時間がタイムアウト値を超えた"メッセージも再生成を中止（タイムアウト処理が機能していない場合）
                            if "実行時間がタイムアウト値を超えた" in error_msg:
                                print(f"  [破棄確定] 再生成タスク{rejected_global_idx} はタイムアウト値を超えたため、再生成を中止します。", flush=True)
                                # 再生成を中止（new_rejected_indicesに追加しない = これ以上再生成しない）
                                continue  # 即座に次のタスクへ（このタスクは再生成対象から除外）
                            # その他のValueError（タスクが破棄されましたなど）は再生成を継続
                            if "タスクが破棄されました" in error_msg:
                                if rejected_local_idx not in new_rejected_indices:
                                    new_rejected_indices.append(rejected_local_idx)
                                continue  # 即座に次のタスクへ（次の再生成ループで再試行）
                        # その他の例外は下のexcept節で処理
                        raise
                    except (IndexError, TypeError, AttributeError, KeyError, ZeroDivisionError) as exec_error:
                        # プログラム実行エラーは即座に次のタスクへ（ログ出力を削減して処理速度向上）
                        regen_exec_time = time.time() - regen_exec_start
                        increment_rejection_stat(rejection_reason_stats, 'program_error')
                        self.timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                        new_rejected_indices.append(rejected_local_idx)
                        continue  # 即座に次のタスクへ
                    except Exception as regen_exec_error:
                        # その他のエラーの場合は再生成を継続（SilentExceptionは既に処理済み）
                        regen_exec_time = time.time() - regen_exec_start
                        increment_rejection_stat(rejection_reason_stats, 'other_error')
                        self.timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                        new_rejected_indices.append(rejected_local_idx)
                        continue  # 再生成ループを継続

                    # 再検証（パフォーマンス最適化：スキップ可能）
                    # 注: バッファへの追加・削除はバッチ終了時に行うため、ここでは条件チェックのみ
                    # 条件チェックが無効な場合や、input_grid/output_gridがNoneの場合はスキップ
                    if not _generator_config.enable_output_condition_check:
                        pass
                    elif input_grid is None or output_grid is None:
                        print(f"  [警告] 再生成タスク{rejected_global_idx}: input_gridまたはoutput_gridがNoneのため、条件チェックをスキップ", flush=True)
                        # 条件チェックができない場合は、再度破棄
                        new_rejected_indices.append(rejected_local_idx)
                        continue
                    else:
                        pass

                    if _generator_config.enable_output_condition_check and input_grid is not None and output_grid is not None:
                        # 複数ペアの条件チェック: すべてのペアが条件を満たしている場合のみ保存
                        invalid_pairs, valid_pairs_data = check_all_pairs_conditions(
                            pairs_data, rejected_global_idx, condition_rejection_stats, rejection_reason_stats, is_regeneration=True
                        )

                        # すべてのペアで条件に該当する場合: タスク廃棄（プログラムの問題）
                        if len(invalid_pairs) == len(pairs_data):
                            print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: すべてのペアが条件に該当するためタスクを破棄します（プログラムの問題）", flush=True)
                            new_rejected_indices.append(rejected_local_idx)
                            continue

                        # 一部のペアのみ条件に該当する場合: そのペアをスキップ（入力グリッドの問題）
                        elif len(invalid_pairs) > 0:
                            print(f"  [スキップ] 再生成タスク{rejected_global_idx}: {len(invalid_pairs)}個のペアが条件に該当するため、それらのペアをスキップします（入力グリッドの問題）", flush=True)
                            # 条件に該当するペアをpairs_dataから削除
                            pairs_data = valid_pairs_data

                            # すべてのペアがスキップされた場合: タスク廃棄
                            if len(pairs_data) == 0:
                                print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: すべてのペアがスキップされたためタスクを破棄します", flush=True)
                                new_rejected_indices.append(rejected_local_idx)
                                continue

                            # ペア数が最小要件（MIN_PAIRS_PER_PROGRAM）未満の場合: タスク廃棄（学習データとして不十分）
                            if len(pairs_data) < _generator_config.min_pairs_per_program:
                                print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {_generator_config.min_pairs_per_program}個）", flush=True)
                                new_rejected_indices.append(rejected_local_idx)
                                continue

                            print(f"  [続行] 再生成タスク{rejected_global_idx}: {len(pairs_data)}個のペアが残りました（スキップ: {len(invalid_pairs)}個）", flush=True)

                        # すべてのペアが条件を満たしている場合: 保存処理に進む（既存のコードをそのまま使用）

                        # 条件に該当しない場合：有効なタスクとして記録
                        valid_task_ids.add(f"task_{rejected_global_idx:03d}")

                        # ステップ5: スコープ削除検証 → コマンド置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                        # すべてのペアの結果を使用して置き換え判定を実行
                        if validated_nodes and len(pairs_data) > 0 and _generator_config.enable_output_condition_check:
                            try:
                                # 置き換え判定用のexecutorを準備
                                replacement_executor = CoreExecutor(preserve_indentation=True)

                                # プログラムコードを生成（置き換え判定で使用）
                                task_complexity = regen_complexity if 'regen_complexity' in locals() else batch_complexities[rejected_local_idx - 1] if (rejected_local_idx - 1) < len(batch_complexities) else 1
                                context_for_replacement = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                                try:
                                    program_code_for_replacement = generator._generate_code(validated_nodes, context_for_replacement, preserve_indentation=True)
                                except TypeError:
                                    program_code_for_replacement = generator._generate_code(validated_nodes, context_for_replacement)

                                # ステップ5-1: スコープ削除検証（一番深いスコープから順に）
                                updated_nodes_for_scope = validated_nodes
                                scope_pairs = find_scope_pairs(validated_nodes)

                                # 一番深いスコープから順に検証
                                for scope_info in scope_pairs:
                                    # スコープを削除したノードリストを作成
                                    nodes_without_scope = remove_scope_from_nodes(updated_nodes_for_scope, scope_info)

                                    # スコープ削除後のプログラムコードを生成
                                    try:
                                        program_code_without_scope = replacement_executor.program_generator._generate_code(
                                            nodes_without_scope, context_for_replacement, preserve_indentation=True
                                        )
                                    except (TypeError, Exception):
                                        try:
                                            program_code_without_scope = replacement_executor.program_generator._generate_code(
                                                nodes_without_scope, context_for_replacement
                                            )
                                        except Exception:
                                            continue

                                    # すべてのペアで条件チェックを実行
                                    all_pairs_unnecessary_scope = True

                                    for pair_data in pairs_data:
                                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                        pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                        try:
                                            # 元のコードの出力（既に取得済み）
                                            output_grid_original = pair_output_grid

                                            # スコープ削除後のコードを実行
                                            try:
                                                output_grid_without_scope, _, _, _ = replacement_executor.execute_program_string(
                                                    program_code=program_code_without_scope,
                                                    input_grid=pair_input_grid,
                                                    input_objects=None,
                                                    input_image_index=0,
                                                    background_color=None
                                                )
                                            except Exception:
                                                all_pairs_unnecessary_scope = False
                                                break

                                            if output_grid_without_scope is None:
                                                all_pairs_unnecessary_scope = False
                                                break

                                            # スコープ削除前のoutput_gridが条件に該当するかチェック
                                            input_unique, input_counts = np.unique(pair_input_grid, return_counts=True)
                                            inferred_background_color = input_unique[np.argmax(input_counts)] if len(input_unique) > 0 else None

                                            condition1_original, condition2_original, condition3_original, condition4_original, condition5_original, condition6_original = check_output_conditions(
                                                pair_input_grid, output_grid_original, background_color=inferred_background_color
                                            )
                                            original_meets_condition = condition1_original or condition2_original or condition3_original or condition5_original or condition6_original

                                            # スコープ削除前後のoutput_gridが完全一致かチェック
                                            outputs_match = np.array_equal(output_grid_original, output_grid_without_scope)

                                            # original_meets_conditionがFalseの場合、スコープ削除前後の完全一致が必要
                                            if not original_meets_condition:
                                                if not outputs_match:
                                                    # このペアでは不要と判断されない
                                                    all_pairs_unnecessary_scope = False
                                                    break

                                        except Exception:
                                            all_pairs_unnecessary_scope = False
                                            break

                                    # すべてのペアで不要と判断された場合のみスコープ削除を適用
                                    if all_pairs_unnecessary_scope:
                                        # スコープ削除後のプログラムにGET_ALL_OBJECTSが含まれているか確認
                                        if 'GET_ALL_OBJECTS' not in program_code_without_scope:
                                            continue

                                        # すべてのペアで検証（エラーが発生しないことを確認）
                                        scope_removal_valid = True
                                        for pair_data in pairs_data:
                                            pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                            try:
                                                test_output_grid, _, test_execution_time, _ = replacement_executor.execute_program_string(
                                                    program_code=program_code_without_scope,
                                                    input_grid=pair_input_grid,
                                                    input_objects=None,
                                                    input_image_index=0,
                                                    background_color=None
                                                )

                                                # エラー検出: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致している場合はエラー
                                                if test_execution_time == 0.0 and np.array_equal(pair_input_grid, test_output_grid):
                                                    scope_removal_valid = False
                                                    break
                                            except Exception:
                                                scope_removal_valid = False
                                                break

                                        # 検証が成功した場合のみ、スコープ削除を適用
                                        if scope_removal_valid:
                                            updated_nodes_for_scope = nodes_without_scope
                                            # 一度スコープ削除を行ったら、次のスコープの検証には更新後のnodesを使用
                                            # インデックスが変わるため、スコープペアを再計算
                                            scope_pairs = find_scope_pairs(updated_nodes_for_scope)

                                # スコープ削除後のnodesをvalidated_nodesに反映
                                validated_nodes = updated_nodes_for_scope

                                # ステップ5-2: コマンド置き換え判定
                                # 置き換え対象のコマンド情報を収集
                                commands_to_check = []
                                for i, node in enumerate(validated_nodes):
                                    # InitializationNodeの場合はスキップ
                                    node_type_name = type(node).__name__
                                    if node_type_name == 'InitializationNode':
                                        continue
                                    # 代入式でない場合はスキップ
                                    if not is_assignment_node(node):
                                        continue

                                    # 一番浅いネストのコマンドを取得
                                    commands_with_depth = get_commands_sorted_by_depth(node, reverse=False)
                                    if not commands_with_depth:
                                        continue

                                    # 一番浅い深度のコマンドを取得
                                    cmd_info = commands_with_depth[0]
                                    cmd_name = cmd_info['command']
                                    cmd_node = cmd_info['node']

                                    commands_to_check.append({
                                        'node_index': i,
                                        'command': cmd_name,
                                        'node': cmd_node,
                                        'assignment_node': node
                                    })

                                # 各コマンドを検証して置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                                updated_nodes = validated_nodes
                                for cmd_info in commands_to_check:
                                    node_index = cmd_info['node_index']
                                    cmd_name = cmd_info['command']
                                    cmd_node = cmd_info['node']

                                    # 置き換え候補を取得
                                    nodes_after_current = updated_nodes[node_index+1:]
                                    replace_result = replace_command_with_fallback(
                                        cmd_name, cmd_node, updated_nodes[:node_index+1],
                                        grid_width=grid_width, grid_height=grid_height
                                    )

                                    if replace_result is None:
                                        # 置き換え候補がない場合はスキップ
                                        continue

                                    nodes_replaced_up_to_current, fallback_code = replace_result
                                    nodes_with_fallback = nodes_replaced_up_to_current + nodes_after_current

                                    # 置き換え後のプログラムコードを生成
                                    if replacement_executor.program_generator is None:
                                        continue

                                    try:
                                        context_fallback = ProgramContext(complexity=1)
                                        try:
                                            program_code_fallback = replacement_executor.program_generator._generate_code(
                                                nodes_with_fallback, context_fallback, preserve_indentation=True
                                            )
                                        except TypeError:
                                            program_code_fallback = replacement_executor.program_generator._generate_code(
                                                nodes_with_fallback, context_fallback
                                            )
                                    except Exception:
                                        continue

                                    # すべてのペアで条件チェックを実行
                                    all_pairs_unnecessary = True

                                    for pair_data in pairs_data:
                                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                        pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                        try:
                                            # 元のコードの出力（既に取得済み）
                                            output_grid_original = pair_output_grid

                                            # 置き換え後のコードを実行
                                            try:
                                                output_grid_fallback, _, _, _ = replacement_executor.execute_program_string(
                                                    program_code=program_code_fallback,
                                                    input_grid=pair_input_grid,
                                                    input_objects=None,
                                                    input_image_index=0,
                                                    background_color=None
                                                )
                                            except Exception:
                                                all_pairs_unnecessary = False
                                                break

                                            if output_grid_fallback is None:
                                                all_pairs_unnecessary = False
                                                break

                                            # 置き換え前のoutput_gridが条件に該当するかチェック
                                            # 入力グリッドから背景色を推論
                                            input_unique, input_counts = np.unique(pair_input_grid, return_counts=True)
                                            inferred_background_color = input_unique[np.argmax(input_counts)] if len(input_unique) > 0 else None

                                            condition1_original, condition2_original, condition3_original, condition4_original, condition5_original, condition6_original = check_output_conditions(
                                                pair_input_grid, output_grid_original, background_color=inferred_background_color
                                            )
                                            original_meets_condition = condition1_original or condition2_original or condition3_original or condition5_original or condition6_original

                                            # 置き換え前後のoutput_gridが完全一致かチェック
                                            outputs_match = np.array_equal(output_grid_original, output_grid_fallback)

                                            # original_meets_conditionがFalseの場合、置き換え前後の完全一致が必要
                                            if not original_meets_condition:
                                                if not outputs_match:
                                                    # このペアでは不要と判断されない
                                                    all_pairs_unnecessary = False
                                                    break

                                            # すべてのペアで条件チェックを続行

                                        except Exception:
                                            # エラーが発生した場合はスキップ
                                            all_pairs_unnecessary = False
                                            break

                                    # すべてのペアで不要と判断された場合のみ置き換えを適用
                                    if all_pairs_unnecessary:
                                        # 置き換え後のプログラムにGET_ALL_OBJECTSが含まれているか確認
                                        try:
                                            if 'GET_ALL_OBJECTS' not in program_code_fallback:
                                                # GET_ALL_OBJECTSがない場合はスキップ
                                                continue

                                            # すべてのペアで検証（エラーが発生しないことを確認）
                                            replacement_valid = True
                                            for pair_data in pairs_data:
                                                pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                                try:
                                                    test_output_grid, _, test_execution_time, _ = replacement_executor.execute_program_string(
                                                        program_code=program_code_fallback,
                                                        input_grid=pair_input_grid,
                                                        input_objects=None,
                                                        input_image_index=0,
                                                        background_color=None
                                                    )

                                                    # エラー検出: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致している場合はエラー
                                                    if test_execution_time == 0.0 and np.array_equal(pair_input_grid, test_output_grid):
                                                        replacement_valid = False
                                                        break
                                                except Exception:
                                                    replacement_valid = False
                                                    break

                                            # 検証が成功した場合のみ、置き換えを適用
                                            if replacement_valid:
                                                updated_nodes = nodes_with_fallback
                                                result_nodes = updated_nodes
                                                validated_nodes = updated_nodes
                                                # 一度置き換えを行ったら、次のコマンドの検証には更新後のnodesを使用
                                                validated_nodes = updated_nodes

                                        except Exception:
                                            # エラーが発生した場合はスキップ
                                            continue

                                # 置き換えが行われた場合、result_nodesを更新
                                if updated_nodes is not validated_nodes:
                                    result_nodes = updated_nodes
                                    # 最初のペアのデータも更新（既存コードとの互換性のため）
                                    if len(pairs_data) > 0:
                                        pairs_data[0]['nodes'] = updated_nodes

                            except Exception as replacement_error:
                                # 置き換え判定でエラーが発生した場合は、元のnodesを使用
                                pass

                        # 条件を満たしている場合：有効なタスクとして記録
                        valid_task_ids.add(f"task_{rejected_global_idx:03d}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # PNGデータはすべてのペアを追加（複数ペア対応）
                        first_pair = pairs_data[0]
                        first_pair_input_grid = np.asarray(first_pair['input_grid'], dtype=int)
                        first_pair_output_grid = np.asarray(first_pair['output_grid'], dtype=int)
                        self.buffer_manager.add_png_data(
                            task_index=rejected_global_idx,
                            input_grid=first_pair_input_grid,
                            output_grid=first_pair_output_grid,
                            timestamp=timestamp,
                            pairs_data=pairs_data  # すべてのペアデータを渡す
                        )

                        # JSONデータはすべてのペアを追加（複数ペア対応）
                        for pair_data in pairs_data:
                            pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                            pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                            trace_results = pair_data.get('trace_results')
                            self.buffer_manager.add_grid_json(
                                task_index=rejected_global_idx,
                                input_grid=pair_input_grid,
                                output_grid=pair_output_grid,
                                trace_results=trace_results
                            )

                        # プログラムJSONを追加（txtファイル生成に使用）
                        # 注: 再生成ループでも、プログラムJSONを追加する必要がある
                        # 最初にプログラムコードを取得（置き換え前）
                        program_code_for_json = program_code  # 初期値
                        if result_nodes:
                            try:
                                context = ProgramContext(original_complexity, grid_width=grid_width, grid_height=grid_height)
                                try:
                                    program_code_for_json = generator._generate_code(result_nodes, context, preserve_indentation=True)
                                except TypeError:
                                    program_code_for_json = generator._generate_code(result_nodes, context)
                            except Exception as e:
                                # エラーが発生した場合は、元のprogram_codeを使用
                                pass

                        # プログラムJSONを追加（_upsert_json_bufferにより既存データがあれば置き換えられる）
                        program_json_data = {
                            "task_id": f"task_{rejected_global_idx:03d}",
                            "timestamp": timestamp,
                            "complexity": original_complexity,
                            "grid_size": {
                                "width": grid_width,
                                "height": grid_height
                            },
                            "program_code": program_code_for_json,
                            "program_length": len(program_code_for_json),
                            "node_count": len(result_nodes) if result_nodes else 0,
                            "statistics": {
                                "line_count": program_code_for_json.count('\n') + 1,
                                "character_count": len(program_code_for_json),
                                "word_count": len(program_code_for_json.split())
                            }
                        }
                        self.buffer_manager.add_program_json(rejected_global_idx, program_json_data)

                        # トークンJSONを追加
                        tokens = tokenize_program(program_code_for_json)
                        tokens_data = {
                            "task_id": f"task_{rejected_global_idx:03d}",
                            "timestamp": timestamp,
                            "complexity": original_complexity,
                            "grid_size": {
                                "width": grid_width,
                                "height": grid_height
                            },
                            "tokens": tokens,
                            "token_count": len(tokens),
                            "vocabulary": list(set(tokens))
                        }
                        self.buffer_manager.add_tokens_json(rejected_global_idx, tokens_data)

                        # 統計JSONを追加
                        stats = analyze_program_statistics(program_code_for_json, result_nodes if result_nodes else [])
                        stats_data = {
                            "task_id": f"task_{rejected_global_idx:03d}",
                            "timestamp": timestamp,
                            "complexity": original_complexity,
                            "grid_size": {
                                "width": grid_width,
                                "height": grid_height
                            },
                            "statistics": stats
                        }
                        self.buffer_manager.add_stats_json(rejected_global_idx, stats_data)

                        # 置き換え後のnodesからprogram_codeを再生成して更新
                        if result_nodes:
                            try:
                                context = ProgramContext(original_complexity, grid_width=grid_width, grid_height=grid_height)
                                try:
                                    updated_program_code = generator._generate_code(result_nodes, context, preserve_indentation=True)
                                except TypeError:
                                    updated_program_code = generator._generate_code(result_nodes, context)
                                # batch_programsのprogram_codeを更新
                                if len(batch_programs[rejected_local_idx - 1]) == 5:
                                    batch_programs[rejected_local_idx - 1] = (result_nodes, updated_program_code, filename, grid_width, grid_height)
                                else:
                                    batch_programs[rejected_local_idx - 1] = (result_nodes, updated_program_code, filename)
                                # JSONバッファのprogram_codeも更新
                                if self.buffer_manager:
                                    self.buffer_manager.update_program_json(rejected_global_idx, updated_program_code)
                                program_code = updated_program_code  # 後で使用するため更新
                            except Exception as e:
                                pass

                        # 成功したペアをDataPairオブジェクトに変換して追加
                        for pair_data in pairs_data:
                            pair = DataPair(
                                input=pair_data['input_grid'].tolist(),
                                output=pair_data['output_grid'].tolist(),
                                program=program_code,
                                metadata={'generated_by': 'hybrid_learning_pipeline', 'generation_attempt': regeneration_attempt + 1}
                            )
                            data_pairs.append(pair)
                            batch_data_pairs.append(pair)
                            self.task_counter += 1

                        # execution_resultsに追加（main.pyと同じ）
                        self.execution_results.append((result_nodes, input_grid, output_grid, filename))

                        # コマンド使用統計に追加
                        if result_nodes:
                            commands = extract_all_commands_from_nodes(result_nodes)
                            statistics.add_program_commands(commands)

                            # 1000タスクごとに重み調整を適用・保存
                            if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                                adjustments = statistics.get_weight_adjustments()
                                if adjustments:
                                    generator.command_weight_adjustments = adjustments
                                    print(f"\n【1000タスク到達: 重み調整を適用】")
                                    print(f"  調整対象: {len(adjustments)}個のコマンド")
                                    statistics.print_statistics()

                                    # 重み調整をファイルに保存
                                    save_weight_adjustments(buffer_base_dir, adjustments, statistics.total_programs)

                                    # バッチ進捗を保存
                                    current_batch = statistics.total_programs // BATCH_SIZE
                                    save_batch_progress(buffer_base_dir, statistics.total_programs, current_batch)

                        task_elapsed = time.time() - task_start_time
                        # 再生成成功時の処理時間も記録
                        self._all_task_processing_times.append(task_elapsed)
                        print(f"  [再生成成功] タスク{rejected_global_idx}: 条件を満たしているため保存 (処理時間: {task_elapsed:.2f}秒)")
                        # 保存完了メッセージを表示
                        print(f"タスク {rejected_global_idx}/{total_tasks} - 保存完了", flush=True)
                    else:
                        # 条件チェックが無効な場合、すべて有効として扱う（複数ペア対応）
                        valid_task_ids.add(f"task_{rejected_global_idx:03d}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # PNGデータはすべてのペアを追加（複数ペア対応）
                        _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                        first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                        first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                        self.buffer_manager.add_png_data(
                            task_index=rejected_global_idx,
                            input_grid=first_pair_input_grid,
                            output_grid=first_pair_output_grid,
                            timestamp=timestamp,
                            pairs_data=pairs_data  # すべてのペアデータを渡す
                        )

                        # JSONデータはすべてのペアを追加（複数ペア対応）
                        for pair_data in pairs_data:
                            pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                            pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                            trace_results = pair_data.get('trace_results')
                            self.buffer_manager.add_grid_json(
                                task_index=rejected_global_idx,
                                input_grid=pair_input_grid,
                                output_grid=pair_output_grid,
                                trace_results=trace_results
                            )

                        # 置き換え後のnodesからprogram_codeを再生成して更新
                        if result_nodes:
                            try:
                                context = ProgramContext(original_complexity, grid_width=grid_width, grid_height=grid_height)
                                try:
                                    updated_program_code = generator._generate_code(result_nodes, context, preserve_indentation=True)
                                except TypeError:
                                    updated_program_code = generator._generate_code(result_nodes, context)
                                # batch_programsのprogram_codeを更新
                                if len(batch_programs[rejected_local_idx - 1]) == 5:
                                    batch_programs[rejected_local_idx - 1] = (result_nodes, updated_program_code, filename, grid_width, grid_height)
                                else:
                                    batch_programs[rejected_local_idx - 1] = (result_nodes, updated_program_code, filename)
                                # JSONバッファのprogram_codeも更新
                                if self.buffer_manager:
                                    self.buffer_manager.update_program_json(rejected_global_idx, updated_program_code)
                                program_code = updated_program_code  # 後で使用するため更新
                            except Exception as e:
                                pass

                        # 成功したペアをDataPairオブジェクトに変換して追加
                        for pair_data in pairs_data:
                            pair = DataPair(
                                input=pair_data['input_grid'].tolist(),
                                output=pair_data['output_grid'].tolist(),
                                program=program_code,
                                metadata={'generated_by': 'hybrid_learning_pipeline', 'generation_attempt': regeneration_attempt + 1}
                            )
                            data_pairs.append(pair)
                            batch_data_pairs.append(pair)
                            self.task_counter += 1

                        # execution_resultsに追加（main.pyと同じ）
                        self.execution_results.append((result_nodes, input_grid, output_grid, filename))

                        # コマンド使用統計に追加
                        if result_nodes:
                            commands = extract_all_commands_from_nodes(result_nodes)
                            statistics.add_program_commands(commands)

                            # 1000タスクごとに重み調整を適用・保存
                            if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                                adjustments = statistics.get_weight_adjustments()
                                if adjustments:
                                    generator.command_weight_adjustments = adjustments
                                    print(f"\n【1000タスク到達: 重み調整を適用】")
                                    print(f"  調整対象: {len(adjustments)}個のコマンド")
                                    statistics.print_statistics()

                                    # 重み調整をファイルに保存
                                    save_weight_adjustments(buffer_base_dir, adjustments, statistics.total_programs)

                                    # バッチ進捗を保存
                                    current_batch = statistics.total_programs // BATCH_SIZE
                                    save_batch_progress(buffer_base_dir, statistics.total_programs, current_batch)

                        task_elapsed = time.time() - task_start_time
                        # 再生成成功時の処理時間も記録
                        self._all_task_processing_times.append(task_elapsed)
                        print(f"  [再生成成功] タスク{rejected_global_idx}: 条件を満たしているため保存 (処理時間: {task_elapsed:.2f}秒)")
                        # 保存完了メッセージを表示
                        print(f"タスク {rejected_global_idx}/{total_tasks} - 保存完了", flush=True)

                # 次のループで再生成するタスクリストを更新
                regen_elapsed = time.time() - regen_start
                previous_rejected_count = len(rejected_task_indices)
                rejected_task_indices = new_rejected_indices
                current_rejected_count = len(rejected_task_indices)

                print(f"  → 再生成ループ {regeneration_attempt} 完了: {current_rejected_count}個のタスクが残りました (処理時間: {regen_elapsed:.2f}秒)", flush=True)

                # 連続失敗カウンターを更新（処理速度向上のため）
                if current_rejected_count >= previous_rejected_count:
                    # タスク数が減っていない（または増えている）場合は失敗とみなす
                    consecutive_regen_failures += 1
                    print(f"  [警告] 再生成ループ {regeneration_attempt}: タスク数が減りませんでした ({previous_rejected_count} → {current_rejected_count})", flush=True)
                else:
                    # タスク数が減っている場合は成功とみなしてカウンターをリセット
                    consecutive_regen_failures = 0
                    print(f"  [成功] 再生成ループ {regeneration_attempt}: タスク数が減少しました ({previous_rejected_count} → {current_rejected_count})", flush=True)

                if not rejected_task_indices:
                    print(f"\n[バッチ {batch_num + 1} 再生成完了] すべてのタスクが条件を満たしました")
                    break

                # 再生成ループごとにガベージコレクション（処理速度向上のため）
                if regeneration_attempt % 3 == 0:  # 3回ごとに実行
                    gc_start = time.time()
                    print(f"  → 再生成ループ {regeneration_attempt}: ガベージコレクション実行中...", flush=True)
                    gc.collect()
                    gc_time = time.time() - gc_start
                    print(f"  [計測] 再生成ループ定期GC (試行{regeneration_attempt}): {gc_time:.3f}秒", flush=True)

            # 再生成ループ終了理由をログに出力
            if rejected_task_indices:
                # ループが終了したが、まだ破棄されたタスクが残っている場合
                if regeneration_attempt >= max_regeneration_attempts:
                    print(f"\n[警告] バッチ {batch_num + 1} 再生成ループ: 最大試行回数（{max_regeneration_attempts}回）に達しました")
                    print(f"  残りの破棄タスク数: {len(rejected_task_indices)}")
                    print(f"  これらのタスクは条件を満たさなかったため、保存されません")
                else:
                    print(f"\n[警告] バッチ {batch_num + 1} 再生成ループ: 予期しない終了（試行回数: {regeneration_attempt}/{max_regeneration_attempts}）")
                    print(f"  残りの破棄タスク数: {len(rejected_task_indices)}")
            else:
                print(f"\n[バッチ {batch_num + 1} 再生成完了] すべてのタスクが条件を満たしました")

            # バッチ終了時にバッファをフラッシュ（条件を満たしたタスクのみ保存）
            # バッチ終了時にフィルタリング（valid_task_idsを使用）
            # このバッチ範囲内のデータのみを処理
            batch_start_task = batch_start_idx + 1  # 1始まりに変換
            batch_end_task = batch_end_idx  # 1始まり

            # 条件を満たしたタスクのみをフィルタリング（バッチ範囲内かつvalid_task_idsに含まれる）
            # JSONバッファから条件を満たしたタスクのみを抽出してフラッシュ（ソート済み）
            print(f"  → JSONバッファのフラッシュ開始 (バッチ {batch_num + 1})...", flush=True)
            for json_type in ["program", "tokens", "stats", "grid"]:
                print(f"    → {json_type} JSONバッファを処理中...", flush=True)
                buffer = getattr(self.buffer_manager, f"{json_type}_json_buffer")
                original_buffer = getattr(self.buffer_manager, f"{json_type}_json_buffer")
                # バッチ範囲内で、かつ条件を満たしたタスクのみを抽出
                batch_items = [
                    item for item in buffer
                    if (batch_start_task <= int(item.get("task_id", "task_000").split("_")[-1]) <= batch_end_task
                        and item.get("task_id") in valid_task_ids)
                ]
                if batch_items:
                    # task_idでソート（task_001, task_002, ...の順序）
                    batch_items = sorted(batch_items, key=lambda x: int(x.get("task_id", "task_000").split("_")[-1]))
                    # 一時的にバッファを置き換えてフラッシュ
                    setattr(self.buffer_manager, f"{json_type}_json_buffer", batch_items)
                    self.buffer_manager._flush_json(json_type, batch_num)

                # バッチ範囲内のすべてのデータをバッファから削除（条件を満たさないタスクも含む）
                # これにより、条件を満たさないタスクのデータがバッファに残らないようにする
                remaining_items = [
                    item for item in original_buffer
                    if not (batch_start_task <= int(item.get("task_id", "task_000").split("_")[-1]) <= batch_end_task)
                ]
                setattr(self.buffer_manager, f"{json_type}_json_buffer", remaining_items)

            # PNGバッファから条件を満たしたタスクのみを抽出してフラッシュ（ソート済み）
            print(f"  → PNGバッファのフラッシュ開始 (バッチ {batch_num + 1})...", flush=True)
            batch_png_data = [
                task_data for task_data in self.buffer_manager.png_buffer
                if (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task
                    and f"task_{task_data.get('task_index', 0):03d}" in valid_task_ids)
            ]
            if batch_png_data:
                # task_indexでソート（1, 2, 3, ...の順序）
                batch_png_data = sorted(batch_png_data, key=lambda x: x.get("task_index", 0))
                # 一時的にバッファを置き換えてフラッシュ
                original_buffer = self.buffer_manager.png_buffer.copy()
                self.buffer_manager.png_buffer = batch_png_data
                self.buffer_manager._flush_png(batch_num)
                # フラッシュ済みのデータをバッファから削除（バッチ範囲内のすべてのデータを削除）
                self.buffer_manager.png_buffer = [
                    task_data for task_data in original_buffer
                    if not (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task)
                ]
            print(f"[バッチ {batch_num + 1} バッファフラッシュ完了]")

            # NeuralTrainingDataGeneratorのバッファをフラッシュ（推論パイプライン用学習データ）
            try:
                self.neural_data_generator.flush_batch(batch_index=batch_num)
            except Exception as e:
                print(f"  [警告] バッチ {batch_num + 1} の学習データ保存でエラーが発生しました: {e}", flush=True)

            # バッチサイズに達したら中間保存
            if len(batch_data_pairs) >= batch_size:
                batch_number += 1
                print(f"  バッチ {batch_number} を保存中... ({len(batch_data_pairs)}タスク)")
                try:
                    # バッチごとにDataPairを保存（最初のバッチは上書き、以降は追記）
                    append_mode = (batch_number > 1)
                    save_start = time.time()
                    self.dataset_io.save_data_pairs(batch_data_pairs, format='jsonl', compress=True, append=append_mode)
                    save_elapsed = time.time() - save_start
                    profiler.record_timing("save_data_pairs", save_elapsed, f"batch_{batch_number}")
                    print(f"  バッチ {batch_number} 保存完了 (処理時間: {save_elapsed:.3f}秒)")

                    # バッチ進捗を保存
                    completed_tasks = self.task_counter
                    save_batch_progress(buffer_base_dir, completed_tasks, batch_number)

                    print(f"  バッチ {batch_number} 保存完了")
                    batch_data_pairs = []  # バッチをクリア
                except Exception as e:
                    # エラーが発生しても処理を継続（バッチは保持）
                    print(f"  警告: バッチ {batch_number} の保存中にエラーが発生しました: {e}")
                    print(f"  バッチ {batch_number} は次のバッチサイズ到達時に再試行されます")
                    # バッチをクリアせずに保持（次のバッチサイズ到達時に再試行）
                    # ただし、メモリリークを防ぐため、バッチサイズの2倍を超えた場合は強制的にクリア
                    if len(batch_data_pairs) >= batch_size * 2:
                        print(f"  警告: バッチサイズが2倍を超えたため、強制的にクリアします")
                        batch_data_pairs = []

            # バッチ完了時の処理（再生成も含めてすべて完了してから）
            print(f"\n【バッチ {batch_num + 1} 完了】")
            print(f"  完了タスク数: {statistics.total_programs} (累積)")
            if rejected_task_indices:
                print(f"  再生成ループ試行回数: {regeneration_attempt}/{max_regeneration_attempts}")
                print(f"  最終的に破棄されたタスク数: {len(rejected_task_indices)}")

            # 条件別の破棄数集計結果を表示
            if condition_rejection_stats['total'] > 0:
                total_rejections = condition_rejection_stats['total']
                print(f"\n【条件別破棄数集計（バッチ {batch_num + 1}）】")
                print(f"  総破棄数: {total_rejections}")
                print(f"  条件①（入力と出力が完全一致）: {condition_rejection_stats['condition1']} ({condition_rejection_stats['condition1'] / total_rejections * 100:.1f}%)")
                print(f"  条件②（トリミング一致）: {condition_rejection_stats['condition2']} ({condition_rejection_stats['condition2'] / total_rejections * 100:.1f}%)")
                print(f"  条件③（入力が単色）: {condition_rejection_stats['condition3']} ({condition_rejection_stats['condition3'] / total_rejections * 100:.1f}%)")
                print(f"  条件④（オブジェクトピクセル数5%未満かつ総色数2以下）: {condition_rejection_stats['condition4']} ({condition_rejection_stats['condition4'] / total_rejections * 100:.1f}%)")
                print(f"  条件⑤（空のoutput）: {condition_rejection_stats['condition5']} ({condition_rejection_stats['condition5'] / total_rejections * 100:.1f}%)")
                print(f"  条件⑥（出力が単色）: {condition_rejection_stats['condition6']} ({condition_rejection_stats['condition6'] / total_rejections * 100:.1f}%)")
                print(f"  複数条件に該当: {condition_rejection_stats['multiple']} ({condition_rejection_stats['multiple'] / total_rejections * 100:.1f}%)")
            else:
                print(f"  条件による破棄なし")

            # 詳細な破棄原因別統計を表示
            if rejection_reason_stats['total'] > 0:
                total_rejections = rejection_reason_stats['total']
                print(f"\n【詳細な破棄原因別統計（バッチ {batch_num + 1}）】")
                print(f"  総破棄数: {total_rejections}")
                print(f"  プログラムデータがNone: {rejection_reason_stats['program_data_none']} ({rejection_reason_stats['program_data_none'] / total_rejections * 100:.1f}%)")
                print(f"  プログラムノードが空: {rejection_reason_stats['nodes_empty']} ({rejection_reason_stats['nodes_empty'] / total_rejections * 100:.1f}%)")
                print(f"  プログラムコードが空: {rejection_reason_stats['program_code_empty']} ({rejection_reason_stats['program_code_empty'] / total_rejections * 100:.1f}%)")
                print(f"  SilentException（タイムアウト）: {rejection_reason_stats['silent_exception']} ({rejection_reason_stats['silent_exception'] / total_rejections * 100:.1f}%)")
                print(f"  プログラム実行エラー: {rejection_reason_stats['program_error']} ({rejection_reason_stats['program_error'] / total_rejections * 100:.1f}%)")
                print(f"  バリデーションエラー: {rejection_reason_stats['validation_error']} ({rejection_reason_stats['validation_error'] / total_rejections * 100:.1f}%)")
                print(f"  その他のエラー: {rejection_reason_stats['other_error']} ({rejection_reason_stats['other_error'] / total_rejections * 100:.1f}%)")
            else:
                print(f"  破棄なし")

            # 条件別集計をリセット（次のバッチ用）
            condition_rejection_stats = {
                'condition1': 0,
                'condition2': 0,
                'condition3': 0,
                'condition4': 0,
                'condition5': 0,
                'condition6': 0,
                'multiple': 0,
                'total': 0
            }

            # 詳細な破棄原因統計をリセット（次のバッチ用）
            rejection_reason_stats = {
                'program_data_none': 0,
                'nodes_empty': 0,
                'program_code_empty': 0,
                'silent_exception': 0,
                'execution_timeout': 0,
                'result_nodes_none': 0,
                'input_grid_none': 0,
                'output_grid_none': 0,
                'object_count_limit': 0,
                'index_error': 0,
                'program_error': 0,
                'validation_error': 0,
                'other_value_error': 0,
                'validate_result_none': 0,
                'validate_timeout': 0,
                'validate_exception': 0,
                'other_error': 0,
                'regeneration_result_none': 0,
                'grid_not_array': 0,
                'grid_empty': 0,
                'condition1': 0,
                'condition2': 0,
                'condition3': 0,
                'condition4': 0,
                'condition5': 0,
                'condition6': 0,
                'total': 0
            }

            # メモリ使用量削減のため、このバッチのexecution_resultsのグリッド配列データを削除
            # （filenameのみ保持して、グリッド配列はファイルに保存済みなので削除）
            batch_start_global = batch_start_idx + 1
            batch_end_global = batch_end_idx
            for i in range(len(self.execution_results) - 1, -1, -1):
                result_nodes, input_grid, output_grid, filename = self.execution_results[i]
                # ファイル名からタスクインデックスを抽出
                try:
                    task_idx = int(filename.split("task_")[-1].split(".")[0])
                    if batch_start_global <= task_idx <= batch_end_global:
                        # このバッチのデータはグリッド配列をNoneにしてメモリを解放（filenameは保持）
                        self.execution_results[i] = (result_nodes, None, None, filename)
                except (ValueError, IndexError):
                    pass  # ファイル名からタスクインデックスを抽出できなかった場合はスキップ

            # バッチ処理完了後のメモリ解放（バッチごとに生成するため、バッチ終了時に明示的に削除）
            memory_cleanup_start = time.time()
            # このバッチのプログラムデータを削除（メモリ解放）
            batch_programs = None
            batch_task_grid_sizes = None
            batch_complexities = None

            # ガベージコレクションを実行してメモリを解放
            gc.collect()

            memory_cleanup_time = time.time() - memory_cleanup_start
            if _generator_config.enable_verbose_logging:
                print(f"  [計測] バッチ分データ削除とメモリ解放: {memory_cleanup_time:.3f}秒", flush=True)

            # メモリ使用量を測定（解放後）
            memory_after = 0
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_freed = memory_before - memory_after
                if memory_freed > 0:
                    print(f"  [メモリ解放] バッチ処理完了後: {memory_before:.1f}MB → {memory_after:.1f}MB (解放: {memory_freed:.1f}MB)", flush=True)
            except (ImportError, Exception):
                pass

            prev_threads_count = threads_before
            prev_memory_mb = memory_after

        print(f"生成されたペア数: {len(data_pairs)}")
        self.pipeline_stats['total_pairs_generated'] = len(data_pairs)

        # 統計情報を出力
        print(f"\n=== 統計情報 ===")
        print(f"破棄原因別統計: {dict(rejection_reason_stats)}")
        print(f"条件別破棄統計: {dict(condition_rejection_stats)}")

        # コマンド使用統計を出力（最終統計）
        if statistics.total_programs > 0:
            print(f"\n【最終統計と重み調整】")
            statistics.print_statistics()
            adjustments = statistics.get_weight_adjustments()
            if adjustments:
                generator.command_weight_adjustments = adjustments
                print(f"  調整対象: {len(adjustments)}個のコマンド")
                # 最終的な重み調整も保存
                save_weight_adjustments(buffer_base_dir, adjustments, statistics.total_programs)
                # バッチ進捗を保存
                current_batch = statistics.total_programs // BATCH_SIZE
                save_batch_progress(buffer_base_dir, statistics.total_programs, current_batch)

        # タスクごとの処理時間の統計を表示
        if hasattr(self, '_all_task_processing_times') and self._all_task_processing_times:
            task_times = self._all_task_processing_times
            avg_task_time = sum(task_times) / len(task_times)
            max_task_time = max(task_times)
            min_task_time = min(task_times)
            print(f"\n{'='*80}")
            print("【タスクごとの処理時間統計（全バッチ合計）】")
            print(f"{'='*80}")
            print(f"処理されたタスク数: {len(task_times)}")
            print(f"平均処理時間: {avg_task_time:.3f}秒 ({avg_task_time/60:.2f}分)")
            print(f"最大処理時間: {max_task_time:.3f}秒 ({max_task_time/60:.2f}分)")
            print(f"最小処理時間: {min_task_time:.3f}秒 ({min_task_time/60:.2f}分)")
            print(f"{'='*80}\n")

        timing_stats_dict = self.timing_stats.get_statistics()
        if timing_stats_dict:
            print(f"タイミング統計:")
            for step_name, stats in timing_stats_dict.items():
                print(f"  {step_name}: 平均={stats['average']:.3f}秒, 最大={stats['max']:.3f}秒, カウント={stats['count']}, 失敗={stats['failures']}")

        # 3. 残りのバッファを保存
        if batch_data_pairs:
            batch_number += 1
            print(f"  最終バッチ {batch_number} を保存中... ({len(batch_data_pairs)}タスク)")
            # バッチ進捗を保存
            completed_tasks = self.task_counter
            save_batch_progress(buffer_base_dir, completed_tasks, batch_number - 1)
            try:
                append_mode = (batch_number > 1)
                save_start = time.time()
                self.dataset_io.save_data_pairs(batch_data_pairs, format='jsonl', compress=True, append=append_mode)
                save_elapsed = time.time() - save_start
                profiler.record_timing("save_data_pairs", save_elapsed, f"batch_{batch_number}")
                # NeuralTrainingDataGeneratorのバッファをフラッシュ（推論パイプライン用学習データ）
                self.neural_data_generator.flush_batch(batch_index=batch_number - 1)
                print(f"  最終バッチ {batch_number} 保存完了 (処理時間: {save_elapsed:.3f}秒)")
            except Exception as e:
                print(f"  警告: 最終バッチ {batch_number} の保存中にエラーが発生しました: {e}")
                # エラーが発生しても処理を継続（データは失われるが、既に保存済みのデータは保持される）

        # 4. FileBufferManagerのバッファをフラッシュ（PNG/txt保存）
        print("PNG/txtファイルを保存中...")
        try:
            flush_start = time.time()
            self.buffer_manager.flush_all()
            flush_elapsed = time.time() - flush_start
            profiler.record_timing("buffer_flush", flush_elapsed, "all_batches")

            summary_start = time.time()
            self.buffer_manager.generate_summary_txt_files()
            summary_elapsed = time.time() - summary_start
            profiler.record_timing("generate_summary_txt", summary_elapsed, "all_batches")
            print(f"PNG/txtファイル保存完了 (flush: {flush_elapsed:.3f}秒, summary: {summary_elapsed:.3f}秒)")
        except Exception as e:
            print(f"警告: PNG/txtファイルの保存中にエラーが発生しました: {e}")
            # エラーが発生しても処理を継続

        # 5. データセット保存（全データを再保存する必要はないが、統計用に保持）
        # 注: バッチごとに既に保存済みなので、ここでは統計情報のみ
        if data_pairs:
            # 最新のファイルパスを取得（既に保存済み）
            saved_path = self.dataset_io.get_latest_file('phase1')
            if saved_path:
                print(f"データセット保存完了: {saved_path}")
            else:
                # フォールバック: 全データを保存
                print("フェーズ1データを保存中...")
                save_start = time.time()
                saved_path = self.dataset_io.save_data_pairs(data_pairs, format='jsonl', compress=True)
                save_elapsed = time.time() - save_start
                profiler.record_timing("save_data_pairs", save_elapsed, "fallback_all")
                print(f"保存完了: {saved_path} (処理時間: {save_elapsed:.3f}秒)")

        self.pipeline_stats['phase1_end_time'] = time.time()

        # 実行時間ログを保存
        start_time = self.pipeline_stats['phase1_start_time']
        end_time = self.pipeline_stats['phase1_end_time']
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60

        time_log_path = os.path.join(buffer_base_dir, "execution_time_log.json")
        avg_time_per_task = total_seconds / statistics.total_programs if statistics.total_programs > 0 else 0
        time_log = {
            'start_datetime': datetime.fromtimestamp(start_time).isoformat(),
            'end_datetime': datetime.fromtimestamp(end_time).isoformat(),
            'total_seconds': total_seconds,
            'total_minutes': total_minutes,
            'total_hours': total_hours,
            'total_tasks': statistics.total_programs,
            'avg_time_per_task': avg_time_per_task,
            'estimated_100k_hours': (avg_time_per_task * 100000 / 3600) if statistics.total_programs > 0 else 0,
            'estimated_100k_days': (avg_time_per_task * 100000 / 86400) if statistics.total_programs > 0 else 0,
            'task_count': num_programs,
            'batch_size': BATCH_SIZE,
        }
        with open(time_log_path, 'w', encoding='utf-8') as f:
            json.dump(time_log, f, indent=2, ensure_ascii=False)
        print(f"\n実行時間ログを保存: {time_log_path}")

        return data_pairs

    def run_phase2_task_assembly(
        self,
        data_pairs: Optional[List[DataPair]] = None,
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
        apply_manual_rules: Optional[bool] = None
    ) -> List[Task]:
        """フェーズ2: タスク統合

        Args:
            data_pairs: DataPairのリスト（Noneの場合は最新のファイルから読み込み）
            num_train: 訓練ペア数
            num_test: テストペア数
            apply_manual_rules: 手動ルールを適用するか

        Returns:
            Taskのリスト
        """
        print("フェーズ2開始: タスク統合")

        self.pipeline_stats['phase2_start_time'] = time.time()

        # データの取得
        if data_pairs is None:
            print("最新のフェーズ1データを読み込み中...")
            data_pairs = self._load_latest_phase1_data()

        if not data_pairs:
            raise ValueError("フェーズ1データが見つかりません")

        # 設定の取得
        num_train = num_train or self.config.num_train
        num_test = num_test or self.config.num_test
        apply_manual_rules = apply_manual_rules if apply_manual_rules is not None else self.config.apply_manual_rules

        # 1. ペアをタスクに統合
        print("ペアをタスクに統合中...")
        tasks = self.task_assembler.assemble_tasks_with_validation(
            data_pairs,
            num_train=num_train,
            num_test=num_test,
            apply_manual_rules=apply_manual_rules
        )

        print(f"統合されたタスク数: {len(tasks)}")
        self.pipeline_stats['total_tasks_created'] = len(tasks)

        # 2. 統計情報の更新
        assembly_stats = self.task_assembler.get_assembly_statistics()
        self.pipeline_stats['successful_assemblies'] = assembly_stats['successful_assemblies']
        self.pipeline_stats['failed_assemblies'] = assembly_stats['failed_assemblies']

        # 3. 保存
        print("フェーズ2データを保存中...")
        saved_path = self.dataset_io.save_tasks(tasks, format='json', compress=True, arc_compatible=True)
        print(f"保存完了: {saved_path}")

        self.pipeline_stats['phase2_end_time'] = time.time()

        return tasks

    def run_full_pipeline(
        self,
        num_programs: Optional[int] = None,
        pairs_per_program: Optional[int] = None,
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
        apply_manual_rules: Optional[bool] = None
    ) -> Dict[str, Any]:
        """フェーズ1→2を一括実行

        Args:
            num_programs: 生成するプログラム数
            pairs_per_program: 各プログラムから生成するペア数
            num_train: 訓練ペア数
            num_test: テストペア数
            apply_manual_rules: 手動ルールを適用するか

        Returns:
            結果レポート
        """
        print("=== ハイブリッドデータセット生成パイプライン開始 ===")
        start_time = time.time()

        # フェーズ1
        print("\n--- フェーズ1: ペア生成 ---")
        data_pairs = self.run_phase1_pair_generation(
            num_programs=num_programs,
            pairs_per_program=pairs_per_program
        )

        # フェーズ2
        print("\n--- フェーズ2: タスク統合 ---")
        tasks = self.run_phase2_task_assembly(
            data_pairs=data_pairs,
            num_train=num_train,
            num_test=num_test,
            apply_manual_rules=apply_manual_rules
        )

        # 統計レポート
        end_time = time.time()
        total_time = end_time - start_time

        report = {
            'pipeline_summary': {
                'total_time_seconds': total_time,
                'phase1_time_seconds': self.pipeline_stats['phase1_end_time'] - self.pipeline_stats['phase1_start_time'],
                'phase2_time_seconds': self.pipeline_stats['phase2_end_time'] - self.pipeline_stats['phase2_start_time']
            },
            'phase1_results': {
                'total_programs': self.pipeline_stats['total_programs_generated'],
                'total_pairs': self.pipeline_stats['total_pairs_generated'],
                'unique_programs': len(set(pair.program for pair in data_pairs))
            },
            'phase2_results': {
                'total_tasks': self.pipeline_stats['total_tasks_created'],
                'successful_assemblies': self.pipeline_stats['successful_assemblies'],
                'failed_assemblies': self.pipeline_stats['failed_assemblies'],
                'avg_train_pairs': num_train or self.config.num_train,
                'avg_test_pairs': num_test or self.config.num_test
            },
            'data_statistics': self._calculate_dataset_statistics(data_pairs, tasks)
        }

        # レポートを保存
        report_path = os.path.join(self.config.base_dir, "statistics", f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n=== パイプライン完了 ===")
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"生成されたタスク数: {len(tasks)}")
        print(f"レポート保存先: {report_path}")

        return report

    def _generate_multiple_pairs_with_nodes(
        self,
        program_code: str,
        nodes: List[Any],
        max_attempts: int = 50,
        task_index: int = 1
    ) -> List[DataPair]:
        """複数ペアを生成（main.pyのgenerate_pairs_with_retryを使用）"""
        import random
        import numpy as np
        from src.data_systems.generator.config import get_config
        from src.data_systems.generator.main import generate_pairs_with_retry, check_all_pairs_conditions, extract_first_pair_data
        from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
        from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size

        _config = get_config()
        MIN_PAIRS_PER_PROGRAM = _config.min_pairs_per_program
        MAX_PAIRS_PER_PROGRAM = _config.max_pairs_per_program

        # グリッドサイズを決定
        grid_width, grid_height = decide_grid_size()

        # 生成するペア数を決定（3-10の乱数）
        num_pairs = random.randint(MIN_PAIRS_PER_PROGRAM, MAX_PAIRS_PER_PROGRAM)

        generator = UnifiedProgramGenerator()
        enable_replacement = _config.enable_output_condition_check  # 条件チェックが有効な場合のみ置き換え判定を実行

        # 複数ペア生成
        pairs_data, validated_nodes = generate_pairs_with_retry(
            nodes, num_pairs, grid_width, grid_height, task_index,
            generator, enable_replacement, is_regeneration=False,
            temporary_input_grid=None  # この関数ではtemporary_input_gridは未使用
        )

        # すべてのペアが生成できなかった場合
        if len(pairs_data) == 0:
            return []

        # ペア数が最小要件未満の場合
        if len(pairs_data) < MIN_PAIRS_PER_PROGRAM:
            return []

        # 条件チェック（簡易版：すべてのペアを有効とみなす）
        # 注: 詳細な条件チェックはmain.pyのcheck_all_pairs_conditionsを使用可能だが、
        #     ここでは簡易版として、すべてのペアを有効とみなす
        valid_pairs = []
        for pair_data in pairs_data:
            input_grid = pair_data['input_grid']
            output_grid = pair_data['output_grid']

            if input_grid is None or output_grid is None:
                continue

            input_grid_np = np.asarray(input_grid, dtype=int) if not isinstance(input_grid, np.ndarray) else input_grid
            output_grid_np = np.asarray(output_grid, dtype=int) if not isinstance(output_grid, np.ndarray) else output_grid

            # 入力と出力が同じ場合はスキップ
            if np.array_equal(input_grid_np, output_grid_np):
                continue

            # 出力が単色の場合はスキップ
            unique_colors = np.unique(output_grid_np)
            if len(unique_colors) <= 1:
                continue

            pair = DataPair(
                input=input_grid_np.tolist(),
                output=output_grid_np.tolist(),
                program=program_code,
                metadata={'generated_by': 'hybrid_learning_pipeline', 'pair_index': len(valid_pairs)}
            )

            valid_pairs.append(pair)

        # 最小要件を満たす場合のみ返す
        if len(valid_pairs) >= MIN_PAIRS_PER_PROGRAM:
            return valid_pairs
        else:
            return []

    def _select_complexity(self) -> int:
        """複雑度分布に基づいて複雑度を選択（main.pyと同じ方法）"""
        import random
        # デフォルトの複雑度分布（ARC-AGI2準拠）
        complexity_distribution = {
            1: 0.4,  # 40% シンプル
            2: 0.3,  # 30% 中程度
            3: 0.2,  # 20% 複雑
            4: 0.1   # 10% 非常に複雑
        }
        rand = random.random()
        cumulative = 0.0
        for complexity, probability in complexity_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return complexity
        return 1  # デフォルト

    def _generate_single_pair_with_nodes(
        self,
        program_code: str,
        nodes: List[Any],
        max_attempts: int = 50,
        task_index: int = 1
    ) -> Optional[DataPair]:
        """単一プログラムから1ペアを生成（設計B）- main.pyのcore_executor_mainを使用"""
        import numpy as np
        from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
        from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
        from src.data_systems.generator.program_executor.node_validator_output import extract_all_commands_from_nodes
        from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
        from src.data_systems.generator.program_generator.generation.program_context import ProgramContext

        # main.pyと同じ設定
        from src.data_systems.generator.config import get_config as get_generator_config
        _generator_config = get_generator_config()
        MAX_EXECUTION_TIME = 10.0  # 最大実行時間（秒）
        enable_replacement = _generator_config.enable_output_condition_check  # 条件チェックが有効な場合のみ置き換え判定を実行

        for attempt in range(max_attempts):
            try:
                # グリッドサイズを決定
                grid_width, grid_height = decide_grid_size()

                # core_executor_mainを使用してプログラムを実行（main.pyと同じ方法）
                try:
                    # 最初のペアではコマンドとプログラムコードを抽出
                    all_commands = None
                    saved_program_code = None
                    if attempt == 0:
                        try:
                            all_commands = extract_all_commands_from_nodes(nodes)
                            generator = UnifiedProgramGenerator()
                            context = ProgramContext(complexity=1)
                            try:
                                saved_program_code = generator._generate_code(nodes, context, preserve_indentation=True)
                            except TypeError:
                                saved_program_code = generator._generate_code(nodes, context)
                        except Exception:
                            pass  # エラーは無視

                    # core_executor_mainを呼び出し
                    result_nodes, input_grid, output_grid, trace_results = core_executor_main(
                        nodes,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        task_index=task_index,
                        enable_replacement=enable_replacement,
                        all_commands=all_commands,
                        program_code=saved_program_code or program_code,
                        is_first_pair=True
                    )

                    if input_grid is None or output_grid is None:
                        continue

                    input_grid_np = np.asarray(input_grid, dtype=int) if not isinstance(input_grid, np.ndarray) else input_grid
                    output_grid_np = np.asarray(output_grid, dtype=int) if not isinstance(output_grid, np.ndarray) else output_grid

                    # 入力と出力が同じ場合はスキップ
                    if np.array_equal(input_grid_np, output_grid_np):
                        continue

                    # 出力が単色の場合はスキップ
                    unique_colors = np.unique(output_grid_np)
                    if len(unique_colors) <= 1:
                        continue

                    pair = DataPair(
                        input=input_grid_np.tolist(),
                        output=output_grid_np.tolist(),
                        program=program_code,
                        metadata={'generated_by': 'hybrid_learning_pipeline', 'generation_attempt': attempt + 1}
                    )

                    # FileBufferManagerにPNG/txtデータを追加
                    self._add_to_buffer_manager(task_index, program_code, input_grid_np, output_grid_np, pair)

                    # NeuralTrainingDataGeneratorに推論パイプライン用学習データを追加
                    self._add_to_neural_data_generator(task_index, program_code, input_grid_np, output_grid_np, pair)

                    return pair

                except Exception as exec_error:
                    # 実行エラーは無視して継続
                    if attempt < 3:
                        print(f"  警告: プログラム実行エラー（試行 {attempt + 1}）: {type(exec_error).__name__}")
                    continue

            except KeyboardInterrupt:
                # ユーザー中断
                raise
            except Exception as e:
                # その他のエラーは無視して継続
                if attempt < 3:
                    error_msg = str(e) if str(e) else type(e).__name__
                    print(f"  警告: ペア生成試行 {attempt + 1} でエラー: {type(e).__name__}: {error_msg}")
                continue
        return None

    def _add_to_buffer_manager(
        self,
        task_index: int,
        program: str,
        input_grid,
        output_grid,
        pair: DataPair
    ):
        """FileBufferManagerにPNG/txtデータを追加"""
        from datetime import datetime
        import numpy as np

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 複雑度を取得（metadataから、またはデフォルト値）
        complexity = pair.metadata.get('complexity', 1) if pair.metadata else 1

        # グリッドサイズを取得
        if len(input_grid.shape) == 2:
            grid_height, grid_width = input_grid.shape
        else:
            # 1次元配列の場合
            grid_height = len(input_grid) if hasattr(input_grid, '__len__') else 0
            grid_width = len(input_grid[0]) if grid_height > 0 and hasattr(input_grid[0], '__len__') else 0

        # PNGデータを追加
        pairs_data = [{
            "pair_index": 0,
            "input_grid": input_grid,
            "output_grid": output_grid
        }]
        self.buffer_manager.add_png_data(
            task_index=task_index,
            input_grid=input_grid,
            output_grid=output_grid,
            timestamp=timestamp,
            pairs_data=pairs_data
        )

        # グリッドJSONを追加
        self.buffer_manager.add_grid_json(
            task_index=task_index,
            input_grid=input_grid,
            output_grid=output_grid
        )

        # プログラムJSONを追加（txtファイル生成に使用）
        program_data = {
            "task_id": f"task_{task_index:03d}",
            "timestamp": timestamp,
            "complexity": complexity,
            "grid_size": {
                "width": grid_width,
                "height": grid_height
            },
            "program_code": program,
            "program_length": len(program),
            "node_count": 0,  # nodes情報がない場合は0
            "statistics": {
                "line_count": program.count('\n') + 1,
                "character_count": len(program),
                "word_count": len(program.split())
            }
        }
        self.buffer_manager.add_program_json(task_index, program_data)

        # トークンJSONを追加
        tokens = tokenize_program(program)
        tokens_data = {
            "task_id": f"task_{task_index:03d}",
            "timestamp": timestamp,
            "complexity": complexity,
            "grid_size": {
                "width": grid_width,
                "height": grid_height
            },
            "tokens": tokens,
            "token_count": len(tokens),
            "vocabulary": list(set(tokens))
        }
        self.buffer_manager.add_tokens_json(task_index, tokens_data)

        # 統計JSONを追加
        stats = analyze_program_statistics(program, [])  # nodes情報がない場合は空リスト
        stats_data = {
            "task_id": f"task_{task_index:03d}",
            "timestamp": timestamp,
            "complexity": complexity,
            "grid_size": {
                "width": grid_width,
                "height": grid_height
            },
            "statistics": stats
        }
        self.buffer_manager.add_stats_json(task_index, stats_data)

    def _add_to_neural_data_generator(
        self,
        task_index: int,
        program_code: str,
        input_grid,
        output_grid,
        pair: DataPair
    ):
        """NeuralTrainingDataGeneratorに推論パイプライン用学習データを追加"""
        import numpy as np

        try:
            # 複雑度を取得（metadataから、またはデフォルト値）
            complexity = pair.metadata.get('complexity', 1) if pair.metadata else 1

            # numpy配列に変換
            input_grid_np = np.asarray(input_grid, dtype=int) if not isinstance(input_grid, np.ndarray) else input_grid
            output_grid_np = np.asarray(output_grid, dtype=int) if not isinstance(output_grid, np.ndarray) else output_grid

            # task_idを生成
            task_id = f"task_{task_index:03d}"

            # NeuralTrainingDataGeneratorにデータを追加
            # 注意: nodes情報はHybridLearningPipelineでは利用できないため、空リストを渡す
            self.neural_data_generator.generate_from_generator_output(
                task_id=task_id,
                program_code=program_code,
                input_grid=input_grid_np,
                output_grid=output_grid_np,
                nodes=[],  # nodes情報は利用不可
                complexity=complexity,
                pair_index=0  # 設計B: 1プログラム=1ペアなので、pair_indexは常に0
            )
        except Exception as e:
            # エラーが発生しても処理を継続（ログ出力のみ）
            print(f"警告: NeuralTrainingDataGeneratorへのデータ追加でエラーが発生しました: {e}")

    def _load_latest_phase1_data(self) -> List[DataPair]:
        """最新のフェーズ1データを読み込み"""
        phase1_files = self.dataset_io.list_saved_files('phase1')

        if not phase1_files.get('phase1'):
            return []

        # 最新のファイルを選択
        latest_file = max(phase1_files['phase1'], key=os.path.getmtime)

        return self.dataset_io.load_data_pairs(latest_file)

    def _calculate_dataset_statistics(
        self,
        data_pairs: List[DataPair],
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """データセット統計を計算"""
        stats = DatasetStatistics()
        stats.update_from_pairs(data_pairs)
        stats.update_from_tasks(tasks)

        return stats.to_dict()

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """パイプライン統計情報を取得"""
        return dict(self.pipeline_stats)

    def reset_statistics(self):
        """統計情報をリセット"""
        self.pipeline_stats = {
            'phase1_start_time': None,
            'phase1_end_time': None,
            'phase2_start_time': None,
            'phase2_end_time': None,
            'total_programs_generated': 0,
            'total_pairs_generated': 0,
            'total_tasks_created': 0,
            'successful_assemblies': 0,
            'failed_assemblies': 0
        }
        self.task_assembler.reset_statistics()
