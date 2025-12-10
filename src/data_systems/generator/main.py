"""
プログラム生成と実行のメインエントリーポイント
"""
import os
import json
import random
import argparse
import time
import sys
import gc
import math
import re
import glob
import traceback
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import Counter, defaultdict
from pathlib import Path

# プロジェクトルートをパスに追加（直接実行時に対応）
# main.pyのパス: src/data_systems/generator/main.py
# プロジェクトルートはmain.pyから4階層上
_file_path = Path(__file__).resolve()
project_root = _file_path.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_executor.core_executor import main as core_executor_main, CoreExecutor

# 部分プログラムフローのインポート（必須）
from src.data_systems.generator.partial_program_helper import (
    generate_partial_program_from_input_grid,
    parse_partial_program_to_nodes
)
from src.data_systems.generator.program_executor.node_validator_output import check_output_conditions, extract_all_commands_from_nodes
from src.data_systems.generator.program_executor.node_analyzer import is_assignment_node, get_commands_sorted_by_depth
from src.data_systems.generator.program_executor.node_analyzer.scope_removal_helper import find_scope_pairs, remove_scope_from_nodes
from src.data_systems.generator.program_executor.node_analyzer.variable_index_normalizer import normalize_variable_indices
from src.data_systems.generator.program_executor.node_analyzer.execution_helpers import replace_command_with_fallback
import numpy as np
from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
# grid_visualizerは遅延インポート（matplotlib依存を回避）
# from src.data_systems.generator.grid_visualizer import save_grids_to_png, save_single_grid_to_png
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.input_grid_generator.background_decider import decide_background_color
from src.data_systems.generator.program_serializer import save_program_for_training, tokenize_program, analyze_program_statistics
from src.data_systems.generator.file_buffer_manager import FileBufferManager, set_global_buffer_manager
from src.data_systems.generator.command_usage_statistics import get_global_statistics, reset_global_statistics
from src.data_systems.generator.program_generator.generation.code_generator import generate_code
# NeuralTrainingDataGeneratorは遅延インポート（モデル登録ログを避けるため）
from src.data_systems.generator.grid_visualizer import save_single_grid_to_png

# ========================================
# 設定（config.pyから読み込み）
# ========================================
try:
    from .config import get_config
except ImportError:
    # 直接実行時（python src/data_systems/generator/main.py）は絶対インポートを使用
    from src.data_systems.generator.config import get_config

# グローバル設定を取得
_config = get_config()

# 設定から定数を取得
WEIGHT_ADJUSTMENT_DIR = _config.weight_adjustment_dir
WEIGHT_ADJUSTMENT_FILENAME = _config.weight_adjustment_filename
BATCH_PROGRESS_FILENAME = _config.batch_progress_filename
BATCH_SIZE = _config.batch_size
TASK_COUNT = _config.task_count

ENABLE_DEBUG_OUTPUT = _config.enable_debug_output
ENABLE_VERBOSE_OUTPUT = _config.enable_verbose_output
ENABLE_VERBOSE_LOGGING = _config.enable_verbose_logging
ENABLE_ALL_LOGS = _config.enable_all_logs

ENABLE_OUTPUT_CONDITION_CHECK = _config.enable_output_condition_check

CONDITION4_KEEP_PROBABILITY = _config.condition4_keep_probability
CONDITION6_KEEP_PROBABILITY = _config.condition6_keep_probability

MIN_PAIRS_PER_PROGRAM = _config.min_pairs_per_program
MAX_PAIRS_PER_PROGRAM = _config.max_pairs_per_program

MAX_PAIR_RETRIES = _config.max_pair_retries

MAX_GRID_REGENERATION_ATTEMPTS = _config.max_grid_regeneration_attempts
MAX_CONSECUTIVE_CONTINUES = _config.max_consecutive_continues

# ノード生成リトライ設定（constants.pyから読み込み）
# 注意: FILTER/SORT_BY/MATCH_PAIRSの再生成は最終チェック関数内で行われるため、
# ここでの再試行設定は不要

COMPLEXITY_RATIOS = _config.complexity_ratios

MAX_EXECUTION_TIME = _config.max_execution_time

# ========================================
# ヘルパー関数
# ========================================
def normalize_ratios(ratios: Dict[int, float]) -> Dict[int, float]:
    """比率を正規化して合計を100%にする"""
    total = sum(ratios.values())
    if total == 0:
        raise ValueError("複雑度比率の合計が0です")
    return {k: (v / total) * 100.0 for k, v in ratios.items()}


def select_complexity(ratios: Dict[int, float]) -> int:
    """比率に基づいて複雑度を選択"""
    return random.choices(list(ratios.keys()), weights=list(ratios.values()), k=1)[0]

def run_with_timeout(func: Callable, timeout: float, *args, **kwargs):
    """関数を直接実行（タイムアウト処理は削除）

    Args:
        func: 実行する関数
        timeout: タイムアウト時間（秒、使用しない）
        *args: 関数に渡す位置引数
        **kwargs: 関数に渡すキーワード引数

    Returns:
        関数の戻り値
    """
    return func(*args, **kwargs)


class TimingStatistics:
    """タイミング統計を管理するクラス"""

    def __init__(self):
        self.step_times: Dict[str, List[float]] = defaultdict(list)
        self.step_counts: Dict[str, int] = defaultdict(int)
        self.step_failures: Dict[str, int] = defaultdict(int)

    def record_step(self, step_name: str, elapsed_time: float, success: bool = True):
        """ステップの実行時間を記録

        Args:
            step_name: ステップ名
            elapsed_time: 実行時間（秒）
            success: 成功したかどうか
        """
        if success:
            self.step_times[step_name].append(elapsed_time)
            self.step_counts[step_name] += 1
        else:
            self.step_failures[step_name] += 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """統計情報を取得

        Returns:
            各ステップの統計情報（平均、最大、最小、合計、カウント）
        """
        stats = {}
        for step_name, times in self.step_times.items():
            if times:
                stats[step_name] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'max': max(times),
                    'min': min(times),
                    'failures': self.step_failures.get(step_name, 0)
                }
            else:
                stats[step_name] = {
                    'count': 0,
                    'total': 0.0,
                    'average': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'failures': self.step_failures.get(step_name, 0)
                }
        return stats

    def print_statistics(self):
        """統計情報を出力"""
        stats = self.get_statistics()
        if not stats:
            print("\n【タイミング統計】データがありません")
            return

        print("\n" + "="*80)
        print("【タイミング統計 - 各ステップの実行時間】")
        print("="*80)

        # ステップ名でソート
        sorted_steps = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

        for step_name, step_stats in sorted_steps:
            if step_stats['count'] > 0:
                print(f"\n【{step_name}】")
                print(f"  実行回数: {step_stats['count']}")
                print(f"  合計時間: {step_stats['total']:.3f}秒")
                print(f"  平均時間: {step_stats['average']:.3f}秒")
                print(f"  最大時間: {step_stats['max']:.3f}秒")
                print(f"  最小時間: {step_stats['min']:.3f}秒")
                if step_stats['failures'] > 0:
                    print(f"  失敗回数: {step_stats['failures']}")
            else:
                if step_stats['failures'] > 0:
                    print(f"\n【{step_name}】")
                    print(f"  実行回数: 0")
                    print(f"  失敗回数: {step_stats['failures']}")

        print("\n" + "="*80)

        # 最大時間のステップを表示
        max_step = max(stats.items(), key=lambda x: x[1]['max'] if x[1]['count'] > 0 else 0)
        if max_step[1]['max'] > 0:
            print(f"【最長時間のステップ】")
            print(f"  {max_step[0]}: {max_step[1]['max']:.3f}秒")

        print("="*80)


def generate_program_with_partial_program_flow(
    generator: UnifiedProgramGenerator,
    complexity: int,
    output_dir: str,
    task_index: int,
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None,
    buffer_manager: Optional[Any] = None,
    timing_stats: Optional[TimingStatistics] = None
) -> Tuple:
    """
    新しいフローでプログラムを生成（部分プログラムを使用）

    1. グリッドサイズ決定
    2. 仮のインプットグリッド生成（プログラムなし）
    3. 部分プログラム生成
    4. プログラム生成器で続きを生成

    Args:
        generator: プログラムジェネレータ
        complexity: 複雑度
        output_dir: 出力ディレクトリ
        task_index: タスクインデックス
        grid_width: グリッド幅（Noneの場合は決定）
        grid_height: グリッド高さ（Noneの場合は決定）
        buffer_manager: バッファマネージャー
        timing_stats: タイミング統計

    Returns:
        (nodes, program_code, grid_width, grid_height) のタプル
    """
    prog_gen_start = time.time()

    # 1. グリッドサイズ決定
    if grid_width is None or grid_height is None:
        size_decide_start = time.time()
        try:
            grid_width, grid_height = decide_grid_size()
            elapsed = time.time() - size_decide_start
            if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[TIME] タスク{task_index} グリッドサイズ決定: {elapsed:.3f}秒")
            if timing_stats:
                timing_stats.record_step("グリッドサイズ決定", elapsed, success=True)
        except Exception as e:
            elapsed = time.time() - size_decide_start
            if timing_stats:
                timing_stats.record_step("グリッドサイズ決定", elapsed, success=False)
            raise

    # 2. 仮のインプットグリッド生成（プログラムなし）
    input_grid_gen_start = time.time()
    try:
        # 背景色を決定（入力グリッド生成時に使用される背景色と同じロジック）
        original_background_color = decide_background_color(nodes=None, default=0)

        # 決定した背景色をcore_executor_main()に渡す（確実に同じ背景色を使用）
        _, input_grid, _, _ = core_executor_main(
            nodes=None,
            grid_width=grid_width,
            grid_height=grid_height,
            enable_replacement=False,
            background_color=original_background_color  # 決定した背景色を渡す
        )
        elapsed = time.time() - input_grid_gen_start
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} インプットグリッド生成: {elapsed:.3f}秒")
        if timing_stats:
            timing_stats.record_step("インプットグリッド生成", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - input_grid_gen_start
        if timing_stats:
            timing_stats.record_step("インプットグリッド生成", elapsed, success=False)
        # インプットグリッド生成に失敗した場合はエラーを発生
        error_msg = f"タスク{task_index} インプットグリッド生成に失敗: {e}"
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg) from e

    # 3. 部分プログラム生成
    partial_program_gen_start = time.time()
    try:
        partial_program, category_var_mapping = generate_partial_program_from_input_grid(
            input_grid=input_grid,
            grid_width=grid_width,
            grid_height=grid_height,
            original_background_color=original_background_color  # データセット生成時限定：背景色情報を渡す
        )
        elapsed = time.time() - partial_program_gen_start
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} 部分プログラム生成: {elapsed:.3f}秒")
        if timing_stats:
            timing_stats.record_step("部分プログラム生成", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - partial_program_gen_start
        if timing_stats:
            timing_stats.record_step("部分プログラム生成", elapsed, success=False)
        # 部分プログラム生成に失敗した場合はエラーを発生
        error_msg = f"タスク{task_index} 部分プログラム生成に失敗: {e}"
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg) from e

    if partial_program is None:
        # 部分プログラムがNoneの場合はエラーを発生
        error_msg = f"タスク{task_index} 部分プログラムがNone（オブジェクト抽出またはカテゴリ分類に失敗）"
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 4. 部分プログラムをNodeリストに変換
    parse_start = time.time()
    try:
        # 変数追跡システムをリセット（前のプログラムの情報が残らないように）
        # 注意: parse_partial_program_to_nodesで部分プログラムの変数が登録される前に実行する
        from src.data_systems.generator.program_generator.metadata.variable_manager import variable_manager
        variable_manager.tracker.reset_variable_tracking()

        # ProgramContextを作成（部分プログラムの変数を登録するため）
        context = ProgramContext(complexity, grid_width=grid_width, grid_height=grid_height)
        partial_nodes, category_var_mapping = parse_partial_program_to_nodes(
            partial_program=partial_program,
            category_var_mapping=category_var_mapping or {},
            context=context
        )
        elapsed = time.time() - parse_start
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} 部分プログラムパース: {elapsed:.3f}秒")
        if timing_stats:
            timing_stats.record_step("部分プログラムパース", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - parse_start
        if timing_stats:
            timing_stats.record_step("部分プログラムパース", elapsed, success=False)
        # パースに失敗した場合はエラーを発生
        error_msg = f"タスク{task_index} 部分プログラムパースに失敗: {e}"
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg) from e

    # 5. プログラム生成器で続きを生成
    node_gen_start = time.time()
    try:
        # contextは部分プログラムパーサーで既に初期化され、変数が登録済み
        nodes = generator.generate_program_nodes_from_partial(
            partial_nodes=partial_nodes,
            category_var_mapping=category_var_mapping or {},
            context=context  # 部分プログラムの変数が登録済みのcontextを渡す
        )
        elapsed = time.time() - node_gen_start

        # ノードの妥当性チェック
        if not nodes or len(nodes) == 0:
            error_msg = f"タスク{task_index}: プログラムノードが空です"
            statistics = get_global_statistics()
            statistics.record_generation_failure('empty_nodes', error_msg)
            raise ValueError(error_msg)

        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} プログラムノード生成（部分プログラムから）: {elapsed:.3f}秒", flush=True)
        if timing_stats:
            timing_stats.record_step("プログラムノード生成（部分プログラムから）", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - node_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムノード生成（部分プログラムから）", elapsed, success=False)
        # 生成に失敗した場合はエラーを発生
        error_msg = f"タスク{task_index} プログラムノード生成に失敗: {e}"
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg) from e

    # 6. プログラムコードを生成
    code_gen_start = time.time()
    try:
        program_code = generate_code(nodes, context)
        elapsed = time.time() - code_gen_start
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} プログラムコード生成: {elapsed:.3f}秒")
        if timing_stats:
            timing_stats.record_step("プログラムコード生成", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - code_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムコード生成", elapsed, success=False)
        raise

    prog_gen_elapsed = time.time() - prog_gen_start
    if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
        print(f"[TIME] タスク{task_index} プログラム生成（部分プログラムフロー）合計: {prog_gen_elapsed:.3f}秒")

    return nodes, program_code, grid_width, grid_height, input_grid


def generate_program(generator: UnifiedProgramGenerator, complexity: int,
                     output_dir: str, task_index: int, grid_width: Optional[int] = None, grid_height: Optional[int] = None, buffer_manager: Optional[Any] = None, timing_stats: Optional[TimingStatistics] = None) -> Tuple:
    """プログラムを生成して保存

    Args:
        generator: プログラムジェネレータ
        complexity: 複雑度
        output_dir: 出力ディレクトリ
        task_index: タスクインデックス
        grid_width: グリッド幅（Noneの場合は決定）
        grid_height: グリッド高さ（Noneの場合は決定）
    """
    prog_gen_start = time.time()

    # グリッドサイズを決定（まだ決定していない場合）
    if grid_width is None or grid_height is None:
        size_decide_start = time.time()
        try:
            grid_width, grid_height = decide_grid_size()
            elapsed = time.time() - size_decide_start
            if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[TIME] タスク{task_index} グリッドサイズ決定: {elapsed:.3f}秒")
            if timing_stats:
                timing_stats.record_step("グリッドサイズ決定", elapsed, success=True)
        except Exception as e:
            elapsed = time.time() - size_decide_start
            if timing_stats:
                timing_stats.record_step("グリッドサイズ決定", elapsed, success=False)
            raise

    # ProgramContextを作成（grid_width/grid_heightを渡す）
    context_start = time.time()
    try:
        context = ProgramContext(complexity, grid_width=grid_width, grid_height=grid_height)
        elapsed = time.time() - context_start
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} ProgramContext作成: {elapsed:.3f}秒", flush=True)
        if timing_stats:
            timing_stats.record_step("ProgramContext作成", elapsed, success=True)
    except Exception as e:
        elapsed = time.time() - context_start
        if timing_stats:
            timing_stats.record_step("ProgramContext作成", elapsed, success=False)
        raise

    # プログラムを生成
    node_gen_start = time.time()
    try:
        nodes = generator.generate_program_nodes(complexity=complexity, grid_width=grid_width, grid_height=grid_height)
        elapsed = time.time() - node_gen_start

        # ノードの妥当性チェック
        if not nodes or len(nodes) == 0:
            error_msg = f"タスク{task_index}: プログラムノードが空です"
            statistics = get_global_statistics()
            statistics.record_generation_failure('empty_nodes', error_msg)
            raise ValueError(error_msg)

        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} プログラムノード生成: {elapsed:.3f}秒", flush=True)
        if timing_stats:
            timing_stats.record_step("プログラムノード生成", elapsed, success=True)
    except RuntimeError as e:
        elapsed = time.time() - node_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムノード生成", elapsed, success=False)
        error_msg = str(e)
        statistics = get_global_statistics()
        if "すべてのノード生成試行が失敗しました" in error_msg:
            statistics.record_generation_failure('runtime_error', error_msg)
        else:
            statistics.record_generation_failure('other', error_msg)
        raise
    except ValueError as e:
        elapsed = time.time() - node_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムノード生成", elapsed, success=False)
        error_msg = str(e)
        statistics = get_global_statistics()
        if "プログラムノードが空です" in error_msg:
            statistics.record_generation_failure('empty_nodes', error_msg)
        elif "CONDITION_ARGでリテラル値生成に失敗しました" in error_msg:
            statistics.record_generation_failure('condition_arg_failed', error_msg)
        elif "リテラル値生成に失敗しました。変数生成は禁止されています" in error_msg:
            statistics.record_generation_failure('literal_generation_failed', error_msg)
        else:
            statistics.record_generation_failure('other', error_msg)
        raise
    except Exception as e:
        elapsed = time.time() - node_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムノード生成", elapsed, success=False)
        error_msg = f"{type(e).__name__}: {str(e)}"
        # AttributeError: 'list' object has no attribute 'add'の場合はスタックトレースを出力
        if "AttributeError" in error_msg and "'list' object has no attribute 'add'" in error_msg:
            print(f"[ERROR] スタックトレース:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        statistics = get_global_statistics()
        statistics.record_generation_failure('other', error_msg)
        raise


    code_gen_start = time.time()
    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
        print(f"[DEBUG] main: generate_code開始 (タスク{task_index})", flush=True)
    try:
        # 統一されたコード生成関数を使用（UnifiedProgramGeneratorのメソッドから分離）
        program_code = generate_code(nodes, context)
        elapsed = time.time() - code_gen_start
        if ENABLE_DEBUG_OUTPUT:
            print(f"[DEBUG] main: _generate_code完了 (タスク{task_index})", flush=True)

        # プログラムコードの妥当性チェック
        if not program_code or not program_code.strip():
            error_msg = f"タスク{task_index}: プログラムコードが空です"
            statistics = get_global_statistics()
            statistics.record_generation_failure('empty_code', error_msg)
            raise ValueError(error_msg)

        # GET_ALL_OBJECTSが含まれているか確認（必須）
        if 'GET_ALL_OBJECTS' not in program_code:
            error_msg = f"タスク{task_index}: プログラムコードにGET_ALL_OBJECTSが含まれていません（必須）"
            statistics = get_global_statistics()
            statistics.record_generation_failure('missing_get_all_objects', error_msg)
            raise ValueError(error_msg)

        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] タスク{task_index} プログラムコード生成: {elapsed:.3f}秒", flush=True)
        if timing_stats:
            timing_stats.record_step("プログラムコード生成", elapsed, success=True)
    except ValueError as e:
        elapsed = time.time() - code_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムコード生成", elapsed, success=False)
        error_msg = str(e)
        statistics = get_global_statistics()
        if "プログラムコードが空です" in error_msg:
            statistics.record_generation_failure('empty_code', error_msg)
        elif "GET_ALL_OBJECTSが含まれていません" in error_msg:
            statistics.record_generation_failure('missing_get_all_objects', error_msg)
        else:
            statistics.record_generation_failure('other', error_msg)
        raise
    except Exception as e:
        elapsed = time.time() - code_gen_start
        if timing_stats:
            timing_stats.record_step("プログラムコード生成", elapsed, success=False)
        error_msg = f"{type(e).__name__}: {str(e)}"
        statistics = get_global_statistics()
        statistics.record_generation_failure('other', error_msg)
        raise

    timestamp_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ENABLE_VERBOSE_OUTPUT:
        print(f"[TIME] タスク{task_index} timestamp生成: {time.time() - timestamp_start:.3f}秒", flush=True)

    if ENABLE_DEBUG_OUTPUT:
        print(f"[DEBUG] main: プログラム生成完了ログ前 (タスク{task_index})", flush=True)

    if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
        print(f"[TIME] タスク{task_index} プログラム生成完了 (合計: {time.time() - prog_gen_start:.3f}秒)", flush=True)

    # バッチ保存用にプログラムデータを追加（JSON形式）
    if buffer_manager:
        json_build_start = time.time()
        json_data = {
            "task_id": f"task_{task_index:03d}",
            "timestamp": timestamp,
            "complexity": complexity,
            "grid_size": {
                "width": grid_width,
                "height": grid_height
            },
            "program_code": program_code,
            "program_length": len(program_code),
            "node_count": len(nodes),
            "statistics": {
                "line_count": program_code.count('\n') + 1,
                "character_count": len(program_code),
                "word_count": len(program_code.split())
            }
        }
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} JSONデータ構築: {time.time() - json_build_start:.3f}秒", flush=True)

        buffer_add_start = time.time()
        buffer_manager.add_program_json(task_index, json_data)
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} buffer_manager.add_program_json: {time.time() - buffer_add_start:.3f}秒", flush=True)

        # tokens_jsonも追加（batch_0000に保存される）
        tokenize_start = time.time()
        tokens = tokenize_program(program_code)
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} tokenize_program: {time.time() - tokenize_start:.3f}秒", flush=True)

        vocab_start = time.time()
        vocabulary = list(set(tokens))
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} set(tokens)重複除去: {time.time() - vocab_start:.3f}秒", flush=True)

        tokens_build_start = time.time()
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
            "vocabulary": vocabulary
        }
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} tokens_data構築: {time.time() - tokens_build_start:.3f}秒", flush=True)

        buffer_add_start = time.time()
        buffer_manager.add_tokens_json(task_index, tokens_data)
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} buffer_manager.add_tokens_json: {time.time() - buffer_add_start:.3f}秒", flush=True)

        # stats_jsonも追加（batch_0000に保存される）
        stats_analyze_start = time.time()
        stats = analyze_program_statistics(program_code, nodes)
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} analyze_program_statistics: {time.time() - stats_analyze_start:.3f}秒", flush=True)

        stats_build_start = time.time()
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
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} stats_data構築: {time.time() - stats_build_start:.3f}秒", flush=True)

        buffer_add_start = time.time()
        buffer_manager.add_stats_json(task_index, stats_data)
        if ENABLE_VERBOSE_OUTPUT:
            print(f"[TIME] タスク{task_index} buffer_manager.add_stats_json: {time.time() - buffer_add_start:.3f}秒", flush=True)
        if ENABLE_DEBUG_OUTPUT:
            print(f"[DEBUG] main: buffer_managerへのデータ追加完了 (タスク{task_index})", flush=True)

    # ファイル名を生成（batch_0000への保存は残すため）
    filename = f"program_complexity{complexity}_{timestamp}.txt"

    if ENABLE_DEBUG_OUTPUT:
        print(f"[DEBUG] main: タスク情報表示前 (タスク{task_index})", flush=True)
    if ENABLE_VERBOSE_OUTPUT or task_index % 10 == 0:  # 10タスクごとに進捗表示
        print(f"タスク {task_index}/{TASK_COUNT} (複雑度={complexity}, グリッドサイズ={grid_width}x{grid_height}): "
              f"Node数={len(nodes)}, コード長={len(program_code)}文字", flush=True)

    return nodes, program_code, grid_width, grid_height, None


def generate_input_grids_only(output_dir: str, timing_stats: Optional[TimingStatistics] = None) -> None:
    """インプットグリッドのみを生成してPNGとして保存

    Args:
        output_dir: 出力ディレクトリ
        timing_stats: タイミング統計（オプション）
    """
    # 出力ディレクトリを作成
    png_output_dir = Path(output_dir) / "input_grids"
    png_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"インプットグリッドPNG出力ディレクトリ: {png_output_dir.absolute()}")

    # 統計情報
    success_count = 0
    failure_count = 0
    total_time = 0.0

    print(f"\nインプットグリッド生成を開始します（{TASK_COUNT}タスク）...\n")

    for task_index in range(1, TASK_COUNT + 1):
        print(f"\n{'='*80}")
        print(f"タスク {task_index}/{TASK_COUNT}")
        print(f"{'='*80}")

        task_start_time = time.time()

        try:
            # 1. グリッドサイズ決定
            size_decide_start = time.time()
            grid_width, grid_height = decide_grid_size()
            size_decide_elapsed = time.time() - size_decide_start
            print(f"[成功] グリッドサイズ決定: {grid_width}x{grid_height} (処理時間: {size_decide_elapsed:.3f}秒)")
            if timing_stats:
                timing_stats.record_step("グリッドサイズ決定", size_decide_elapsed, success=True)

            # 2. 仮のインプットグリッド生成（プログラムなし）
            input_grid_gen_start = time.time()
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )
            input_grid_gen_elapsed = time.time() - input_grid_gen_start

            # インプットグリッドの情報を表示
            if input_grid is not None:
                unique_colors = np.unique(input_grid).tolist()
                print(f"[成功] インプットグリッド生成完了 (処理時間: {input_grid_gen_elapsed:.3f}秒)")
                print(f"  グリッドサイズ: {input_grid.shape[1]}x{input_grid.shape[0]} (幅x高さ)")
                print(f"  使用されている色: {unique_colors}")
                print(f"  色の種類数: {len(unique_colors)}")

                if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"  グリッドの内容:")
                    for y in range(min(10, input_grid.shape[0])):  # 最大10行まで表示
                        row_str = " ".join(str(input_grid[y, x]) for x in range(min(10, input_grid.shape[1])))
                        if input_grid.shape[1] > 10:
                            row_str += " ..."
                        print(f"    {row_str}")
                    if input_grid.shape[0] > 10:
                        print(f"    ... ({input_grid.shape[0] - 10}行省略)")

                if timing_stats:
                    timing_stats.record_step("インプットグリッド生成", input_grid_gen_elapsed, success=True)

                # PNGとして保存（遅延インポート）
                png_path = png_output_dir / f"task_{task_index:03d}_input_grid_{grid_width}x{grid_height}.png"
                png_title = f"Task {task_index} - Input Grid ({grid_width}x{grid_height})"
                save_start = time.time()
                try:

                    save_success = save_single_grid_to_png(
                        grid=input_grid,
                        output_path=str(png_path),
                        title=png_title,
                        show_grid=True,
                        dpi=100
                    )
                except Exception as e:
                    print(f"  [警告] PNG保存に失敗しました（matplotlib依存の問題の可能性）: {e}")
                    save_success = False
                save_elapsed = time.time() - save_start

                if save_success:
                    print(f"  [PNG保存] {png_path.name} (処理時間: {save_elapsed:.3f}秒)")
                    if timing_stats:
                        timing_stats.record_step("PNG保存", save_elapsed, success=True)
                else:
                    print(f"  [警告] PNG保存に失敗しました: {png_path.name}")
                    if timing_stats:
                        timing_stats.record_step("PNG保存", save_elapsed, success=False)
            else:
                print(f"[警告] インプットグリッドがNoneです")
                if timing_stats:
                    timing_stats.record_step("インプットグリッド生成", input_grid_gen_elapsed, success=False)
                raise ValueError(f"タスク{task_index}: インプットグリッドがNoneです")

            task_elapsed = time.time() - task_start_time
            total_time += task_elapsed
            success_count += 1
            print(f"[完了] タスク{task_index} 成功 (総処理時間: {task_elapsed:.3f}秒)")

        except Exception as e:
            task_elapsed = time.time() - task_start_time
            total_time += task_elapsed
            failure_count += 1
            print(f"[失敗] タスク{task_index} エラー: {type(e).__name__}: {e} (処理時間: {task_elapsed:.3f}秒)")
            if timing_stats:
                timing_stats.record_step("タスク生成（失敗）", task_elapsed, success=False)
            if ENABLE_ALL_LOGS:
                traceback.print_exc()

    # 統計情報を表示
    print(f"\n{'='*80}")
    print(f"【実行完了】")
    print(f"{'='*80}")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"成功: {success_count}/{TASK_COUNT}")
    print(f"失敗: {failure_count}/{TASK_COUNT}")
    if success_count > 0:
        avg_time = total_time / TASK_COUNT
        print(f"1タスクあたりの平均時間: {avg_time:.3f}秒")
    print(f"PNG出力ディレクトリ: {png_output_dir.absolute()}")


def print_settings():
    """設定を表示"""
    print(f"\n設定: タスク数={TASK_COUNT}, 複雑度比率=", end="")
    ratios_str = ", ".join([f"{c}:{COMPLEXITY_RATIOS[c]}%" for c in sorted(COMPLEXITY_RATIOS.keys())])
    print(ratios_str)


def save_weight_adjustments(output_dir: str, adjustments: Dict[str, float], total_programs: int):
    """重み調整をJSONファイルに保存

    Args:
        output_dir: 出力ディレクトリ
        adjustments: コマンド名をキー、調整倍率を値とする辞書
        total_programs: 総プログラム数
    """
    weight_dir = os.path.join(output_dir, WEIGHT_ADJUSTMENT_DIR.replace("outputs/", ""))
    os.makedirs(weight_dir, exist_ok=True)

    # ファイル名に総プログラム数を含める（例: command_weight_adjustments_1000.json）
    batch_number = (total_programs // 1000) * 1000
    filename = f"command_weight_adjustments_{batch_number}.json"
    filepath = os.path.join(weight_dir, filename)

    # メタデータを含めて保存
    data = {
        'total_programs': total_programs,
        'batch_number': batch_number,
        'timestamp': datetime.now().isoformat(),
        'adjustments': adjustments
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  [重み調整保存] {filepath} (総プログラム数: {total_programs})")

    # 最新の重み調整を常に同じファイル名でも保存（再開時に読み込むため）
    latest_filepath = os.path.join(weight_dir, WEIGHT_ADJUSTMENT_FILENAME)
    with open(latest_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if ENABLE_ALL_LOGS:
        print(f"  [重み調整保存] {latest_filepath} (最新版)")


def load_weight_adjustments(output_dir: str) -> Optional[Dict[str, float]]:
    """最新の重み調整ファイルを読み込む

    Args:
        output_dir: 出力ディレクトリ

    Returns:
        重み調整辞書（ファイルが存在しない場合はNone）
    """
    weight_dir = os.path.join(output_dir, WEIGHT_ADJUSTMENT_DIR.replace("outputs/", ""))
    latest_filepath = os.path.join(weight_dir, WEIGHT_ADJUSTMENT_FILENAME)

    if not os.path.exists(latest_filepath):
        return None

    try:
        with open(latest_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        adjustments = data.get('adjustments', {})
        total_programs = data.get('total_programs', 0)
        timestamp = data.get('timestamp', '')

        print(f"\n【重み調整ファイルを読み込み】")
        print(f"  ファイル: {latest_filepath}")
        print(f"  総プログラム数: {total_programs}")
        print(f"  保存時刻: {timestamp}")
        print(f"  調整対象コマンド数: {len(adjustments)}")

        return adjustments
    except Exception as e:
        print(f"  [警告] 重み調整ファイルの読み込みに失敗: {e}")
        return None


def save_batch_progress(output_dir: str, completed_tasks: int, current_batch: int):
    """バッチ進捗をJSONファイルに保存

    Args:
        output_dir: 出力ディレクトリ
        completed_tasks: 完了したタスク数（累積）
        current_batch: 現在のバッチ番号（0始まり）
    """
    weight_dir = os.path.join(output_dir, WEIGHT_ADJUSTMENT_DIR.replace("outputs/", ""))
    os.makedirs(weight_dir, exist_ok=True)

    filepath = os.path.join(weight_dir, BATCH_PROGRESS_FILENAME)

    data = {
        'completed_tasks': completed_tasks,
        'current_batch': current_batch,
        'batch_size': BATCH_SIZE,
        'last_batch_end_task': current_batch * BATCH_SIZE,
        'timestamp': datetime.now().isoformat()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  [バッチ進捗保存] 完了タスク: {completed_tasks}, バッチ: {current_batch}")


def detect_latest_batch_progress(output_dir: str) -> Optional[Dict[str, int]]:
    """実際のバッチファイルから最新の進捗を検出

    Args:
        output_dir: 出力ディレクトリ

    Returns:
        進捗情報の辞書（バッチファイルが見つからない場合はNone）
        {
            'completed_tasks': int,  # 完了したタスク数
            'current_batch': int,   # 最後に完了したバッチ番号
            'next_batch': int,      # 次に処理すべきバッチ番号
            'next_start_task': int  # 次に処理を開始すべきタスク番号
        }
    """
    # バッチディレクトリを検索
    batch_pattern = os.path.join(output_dir, "batch_*")
    batch_dirs = glob.glob(batch_pattern)

    if not batch_dirs:
        return None

    max_batch_num = -1
    max_completed_tasks = 0

    # 各バッチディレクトリから最大のタスク番号を検出
    for batch_dir in batch_dirs:
        # バッチ番号を抽出（例: batch_0018 -> 18）
        batch_match = re.search(r'batch_(\d+)', os.path.basename(batch_dir))
        if not batch_match:
            continue

        batch_num = int(batch_match.group(1))

        # バッチディレクトリ内のprogram_batch_*.jsonファイルを検索
        program_pattern = os.path.join(batch_dir, "program_batch_*.json")
        program_files = glob.glob(program_pattern)

        if not program_files:
            continue

        # ファイル名からタスク番号を抽出（例: program_batch_0018_tasks_18001_to_19000_*.json）
        for program_file in program_files:
            task_match = re.search(r'tasks_(\d+)_to_(\d+)', os.path.basename(program_file))
            if task_match:
                end_task = int(task_match.group(2))
                if end_task > max_completed_tasks:
                    max_completed_tasks = end_task
                    max_batch_num = batch_num

    if max_batch_num < 0:
        return None

    # 次のバッチを計算
    next_batch = max_batch_num + 1
    batch_size = BATCH_SIZE  # デフォルトのバッチサイズを使用
    next_start_task = next_batch * batch_size

    print(f"\n【実際のバッチファイルから進捗を検出】")
    print(f"  検出された完了タスク数: {max_completed_tasks}")
    print(f"  検出された完了バッチ: {max_batch_num} (0始まり)")
    print(f"  次バッチ: {next_batch}")
    print(f"  次開始タスク番号: {next_start_task}")

    return {
        'completed_tasks': max_completed_tasks,
        'current_batch': max_batch_num,
        'next_batch': next_batch,
        'next_start_task': next_start_task,
        'batch_size': batch_size
    }


def load_batch_progress(output_dir: str) -> Optional[Dict[str, int]]:
    """バッチ進捗ファイルを読み込み、必要に応じて実際のバッチファイルから進捗を検出

    Args:
        output_dir: 出力ディレクトリ

    Returns:
        進捗情報の辞書（ファイルが存在しない場合はNone）
        {
            'completed_tasks': int,  # 完了したタスク数
            'current_batch': int,   # 最後に完了したバッチ番号
            'next_batch': int,      # 次に処理すべきバッチ番号
            'next_start_task': int  # 次に処理を開始すべきタスク番号
        }
    """
    weight_dir = os.path.join(output_dir, WEIGHT_ADJUSTMENT_DIR.replace("outputs/", ""))
    filepath = os.path.join(weight_dir, BATCH_PROGRESS_FILENAME)

    # 実際のバッチファイルから進捗を検出
    detected_progress = detect_latest_batch_progress(output_dir)

    # batch_progress.jsonが存在する場合
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_completed_tasks = data.get('completed_tasks', 0)
            file_current_batch = data.get('current_batch', -1)
            batch_size = data.get('batch_size', BATCH_SIZE)
            timestamp = data.get('timestamp', '')

            # ファイルの進捗と実際のファイルから検出した進捗を比較
            if detected_progress:
                detected_completed_tasks = detected_progress['completed_tasks']
                detected_current_batch = detected_progress['current_batch']

                # 実際のファイルから検出した進捗が、batch_progress.jsonの進捗を超えている場合
                # これは、途中から再開した場合に、新しいバッチが生成されたことを意味する
                # 実際のファイルの進捗を使用する（実際のファイルには絶対的なタスク番号が記録されている）
                if detected_completed_tasks > file_completed_tasks:
                    print(f"\n【警告】batch_progress.jsonより実際のバッチファイルの方が進んでいます")
                    print(f"  batch_progress.json: {file_completed_tasks}タスク (バッチ{file_current_batch})")
                    print(f"  実際のファイル: {detected_completed_tasks}タスク (バッチ{detected_current_batch})")
                    print(f"  実際のファイルの進捗を使用します")

                    # batch_progress.jsonを更新
                    save_batch_progress(output_dir, detected_completed_tasks, detected_current_batch)

                    # detected_progressにbatch_sizeを追加（返り値の一貫性のため）
                    detected_progress['batch_size'] = batch_size
                    return detected_progress
                else:
                    # batch_progress.jsonの方が進んでいる、または同じ場合
                    # batch_progress.jsonの内容を使用
                    pass  # 下の共通処理に進む
            else:
                # ファイルは存在するが、実際のバッチファイルが見つからない場合は、ファイルの内容を使用
                pass  # 下の共通処理に進む

            # 共通処理: batch_progress.jsonの内容を使用
            next_batch = file_current_batch + 1
            next_start_task = next_batch * batch_size

            print(f"\n【バッチ進捗ファイルを読み込み】")
            print(f"  ファイル: {filepath}")
            print(f"  完了タスク数: {file_completed_tasks}")
            print(f"  完了バッチ: {file_current_batch} (0始まり)")
            print(f"  次バッチ: {next_batch}")
            print(f"  次開始タスク番号: {next_start_task}")
            print(f"  保存時刻: {timestamp}")

            return {
                'completed_tasks': file_completed_tasks,
                'current_batch': file_current_batch,
                'next_batch': next_batch,
                'next_start_task': next_start_task,
                'batch_size': batch_size
            }
        except Exception as e:
            print(f"  [警告] バッチ進捗ファイルの読み込みに失敗: {e}")
            # ファイルの読み込みに失敗した場合は、実際のファイルから検出を試みる
            if detected_progress:
                print(f"  実際のバッチファイルから進捗を検出します")
                return detected_progress
            return None
    else:
        # batch_progress.jsonが存在しない場合は、実際のファイルから検出
        if detected_progress:
            print(f"\n【batch_progress.jsonが存在しないため、実際のバッチファイルから進捗を検出】")
            # 検出した進捗をbatch_progress.jsonに保存
            save_batch_progress(
                output_dir,
                detected_progress['completed_tasks'],
                detected_progress['current_batch']
            )
            return detected_progress
        else:
            print(f"\n【バッチ進捗情報が見つかりません】新規開始します")
        return None


def load_statistics_from_file(output_dir: str, statistics) -> int:
    """統計ファイルから状態を復元（オプション）

    現在は重み調整のみ復元するが、将来的に統計状態も復元可能にする

    Args:
        output_dir: 出力ディレクトリ
        statistics: CommandUsageStatisticsインスタンス

    Returns:
        復元された総プログラム数（復元できない場合は0）
    """
    weight_dir = os.path.join(output_dir, WEIGHT_ADJUSTMENT_DIR.replace("outputs/", ""))
    latest_filepath = os.path.join(weight_dir, WEIGHT_ADJUSTMENT_FILENAME)

    if not os.path.exists(latest_filepath):
        return 0

    try:
        with open(latest_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_programs = data.get('total_programs', 0)
        # 統計状態の復元は将来的な拡張として残しておく
        # 現時点では総プログラム数のみ返す
        return total_programs
    except Exception:
        return 0


def convert_grid_to_list(grid: Any) -> List[List[int]]:
    """グリッドをリスト形式に変換"""
    if grid is None:
        return []
    if isinstance(grid, np.ndarray):
        return grid.tolist()
    return [[int(cell) for cell in row] for row in grid]


# ========================================
# 共通ヘルパー関数（重複削減）
# ========================================

def update_rejection_stats(
    condition1: bool,
    condition2: bool,
    condition3: bool,
    condition4: bool,
    condition5: bool,
    condition6: bool,
    condition_rejection_stats: Dict[str, int],
    rejection_reason_stats: Dict[str, int]
) -> None:
    """統計情報を更新

    Args:
        condition1-6: 各条件の判定結果
        condition_rejection_stats: 条件別の統計
        rejection_reason_stats: 破棄原因別の統計
    """
    condition_rejection_stats['total'] += 1
    condition_count = sum([condition1, condition2, condition3, condition4, condition5, condition6])
    if condition_count > 1:
        condition_rejection_stats['multiple'] += 1
    else:
        if condition1:
            condition_rejection_stats['condition1'] += 1
            rejection_reason_stats['condition1'] += 1
        if condition2:
            condition_rejection_stats['condition2'] += 1
            rejection_reason_stats['condition2'] += 1
        if condition3:
            condition_rejection_stats['condition3'] += 1
            rejection_reason_stats['condition3'] += 1
        if condition4:
            condition_rejection_stats['condition4'] += 1
            rejection_reason_stats['condition4'] += 1
        if condition5:
            condition_rejection_stats['condition5'] += 1
            rejection_reason_stats['condition5'] += 1
        if condition6:
            condition_rejection_stats['condition6'] += 1
            rejection_reason_stats['condition6'] += 1
    rejection_reason_stats['total'] += 1


def validate_execution_results(
    result_nodes: Any,
    input_grid: Any,
    output_grid: Any,
    task_idx: int,
    rejection_reason_stats: Dict[str, int],
    is_regeneration: bool = False
) -> None:
    """実行結果の妥当性をチェックし、Noneの場合は例外を発生

    Args:
        result_nodes: 実行結果のノード
        input_grid: 入力グリッド
        output_grid: 出力グリッド
        task_idx: タスクインデックス
        rejection_reason_stats: 破棄原因別の統計
        is_regeneration: 再生成ループかどうか

    Raises:
        ValueError: None値が検出された場合
    """
    task_prefix = f"再生成タスク{task_idx}" if is_regeneration else f"タスク{task_idx}"

    if result_nodes is None:
        error_msg = f"{task_prefix}: result_nodesがNoneです。プログラム実行に失敗しました。"
        print(f"  [警告] {error_msg}", flush=True)
        if not is_regeneration:
            rejection_reason_stats['result_nodes_none'] += 1
            rejection_reason_stats['total'] += 1
        raise ValueError(error_msg)
    if input_grid is None:
        error_msg = f"{task_prefix}: input_gridがNoneです。プログラム実行に失敗しました。"
        print(f"  [警告] {error_msg}", flush=True)
        if not is_regeneration:
            rejection_reason_stats['input_grid_none'] += 1
            rejection_reason_stats['total'] += 1
        raise ValueError(error_msg)
    if output_grid is None:
        error_msg = f"{task_prefix}: output_gridがNoneです。プログラム実行に失敗しました。"
        print(f"  [警告] {error_msg}", flush=True)
        if not is_regeneration:
            rejection_reason_stats['output_grid_none'] += 1
            rejection_reason_stats['total'] += 1
        raise ValueError(error_msg)


def handle_pair_generation_error(
    error: Exception,
    pair_idx: int,
    num_pairs: int,
    retry_count: int,
    max_retries: int,
    is_first_pair: bool
) -> Tuple[bool, bool]:
    """ペア生成エラーを処理し、リトライまたはスキップを決定

    Args:
        error: 発生したエラー
        pair_idx: ペアのインデックス（0始まり）
        num_pairs: 総ペア数
        retry_count: 現在のリトライ回数
        max_retries: 最大リトライ回数
        is_first_pair: 最初のペアかどうか（pair_idx == 0）

    Returns:
        (should_retry, pair_generated): リトライすべきか、ペアが生成されたか
        - (True, True): リトライする（continue）
        - (False, False): ペアをスキップする（break）
        - 例外を再発生: 最初のペアでエラーが発生した場合

    Raises:
        元の例外: 最初のペア（pair_idx == 0）でエラーが発生した場合
    """
    # SilentException（タイムアウト）を最優先で処理
    # SilentExceptionはタスク破棄を意味するため、最初のペアでない場合でもタスクを破棄する必要がある
    try:
        from src.core_systems.executor.core import SilentException
        if isinstance(error, SilentException):
            # SilentExceptionが発生した場合、最初のペアでない場合でもタスクを破棄する
            # 同じプログラムでリトライしても解決しないため
            if is_first_pair:
                # 最初のペアの場合は、タスク廃棄の例外として再発生
                raise
            else:
                # 最初以外のペアの場合でも、タスクを破棄する（同じプログラムでリトライしても解決しない）
                error_msg = str(error).replace("SilentException: ", "").replace("src.core_systems.executor.core.SilentException: ", "")
                print(f"  [タスク破棄] ペア {pair_idx+1}/{num_pairs} でSilentException（タイムアウト）が発生しました。同じプログラムでリトライしても解決しないため、タスクを破棄します: {error_msg[:200]}", flush=True)
                # タスク破棄の例外として再発生（上位のエラーハンドリングで処理される）
                raise ValueError(f"プログラム実行エラーによりタスクが破棄されました（{error_msg[:200]}）")
    except ImportError:
        # SilentExceptionが利用できない場合は、型名で判定
        if type(error).__name__ == 'SilentException':
            if is_first_pair:
                raise
            else:
                error_msg = str(error)
                print(f"  [タスク破棄] ペア {pair_idx+1}/{num_pairs} でSilentException（タイムアウト）が発生しました。同じプログラムでリトライしても解決しないため、タスクを破棄します: {error_msg[:200]}", flush=True)
                raise ValueError(f"プログラム実行エラーによりタスクが破棄されました（{error_msg[:200]}）")

    error_str = str(error)
    error_type = type(error).__name__

    # プログラム実行エラーのパターン（リトライしても解決しないエラー）
    program_execution_error_patterns = [
        "未定義の変数:",
        "'int' object is not iterable",
        "'str' object is not iterable",
        "'float' object is not iterable",
        "TypeError:",
        "NameError:",
        "AttributeError:",
        "IndexError:",
        "KeyError:",
        "プログラム実行エラー: プログラム実行エラー:",
        "ピクセル数が上限を超えました",  # ピクセル数上限もリトライ不可能
        "オブジェクト数が上限を超えました",  # オブジェクト数上限もリトライ不可能（変数、FORループ、WHILEループ、EXCLUDEなど）
        "EXCLUDE: オブジェクト数が上限を超えています"  # EXCLUDE操作の上限もリトライ不可能
    ]

    # プログラム実行エラーかどうかを判定
    is_program_execution_error = any(pattern in error_str for pattern in program_execution_error_patterns)

    # 最初のペアで発生した場合は、タスク廃棄の例外として再発生
    if is_first_pair:
        raise

    # ValueError（ペアスキップ例外）の処理
    if isinstance(error, ValueError):
        # プログラム実行エラーを優先チェック（「ペアスキップ:」で始まっていてもプログラム実行エラーの場合は即座にスキップ）
        # 「execution_timeが0.0」はプログラム実行エラーが原因であることを示す
        is_execution_time_zero_error = "execution_timeが0.0" in error_str and "入力グリッドと出力グリッドが完全一致" in error_str
        if is_program_execution_error or is_execution_time_zero_error or ("ペアスキップ: SilentException発生 - プログラム実行エラー" in error_str and any(pattern in error_str for pattern in ["未定義の変数", "'int' object is not iterable", "TypeError", "NameError", "ピクセル数が上限", "オブジェクト数が上限", "FORループ実行エラー", "WHILEループ実行エラー"])):
            # プログラム実行エラー（リトライ不可能）: 即座にスキップ
            print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} を即座にスキップします（プログラム実行エラーのためリトライ不可能）: {error_str[:200]}", flush=True)
            return False, False  # 即座にスキップ（リトライしない）
        elif error_str.startswith("ペアスキップ:"):
            # ペアスキップの例外（リトライ可能）
            if retry_count < max_retries:
                print(f"  [リトライ] ペア {pair_idx+1}/{num_pairs} をリトライします（{retry_count+1}/{max_retries}回目）: {error_str}", flush=True)
                return True, True  # リトライ
            else:
                # リトライ上限に達した場合、ペアスキップ
                print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} のリトライ上限に達しました。このペアをスキップします: {error_str}", flush=True)
                return False, False  # スキップ
        # 注意: is_program_execution_errorのチェックは1257行目で既に実施済みのため、ここでのチェックは不要（到達不可能なコード）
        else:
            # その他のValueError（リトライ可能な可能性がある）
            if retry_count < max_retries:
                print(f"  [リトライ] ペア {pair_idx+1}/{num_pairs} をリトライします（{retry_count+1}/{max_retries+1}回目）: {error_type}: {error_str}", flush=True)
                return True, True  # リトライ
            else:
                # リトライ上限に達した場合、ペアスキップ
                print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} のリトライ上限に達しました。このペアをスキップします: {error_type}: {error_str}", flush=True)
                return False, False  # スキップ

    # TypeError、NameErrorなど（プログラム実行エラー）の処理
    if is_program_execution_error or error_type in ('TypeError', 'NameError', 'AttributeError', 'IndexError', 'KeyError'):
        # プログラム実行エラー: 即座にスキップ
        print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} を即座にスキップします（プログラム実行エラーのためリトライ不可能）: {error_type}: {error_str[:200]}", flush=True)
        return False, False  # 即座にスキップ（リトライしない）

    # その他の例外
    if retry_count < max_retries:
        print(f"  [リトライ] ペア {pair_idx+1}/{num_pairs} をリトライします（{retry_count+1}/{max_retries+1}回目）: {error_type}: {error_str}", flush=True)
        return True, True  # リトライ
    else:
        # リトライ上限に達した場合、ペアスキップ
        print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} のリトライ上限に達しました。このペアをスキップします: {error_type}: {error_str}", flush=True)
        return False, False  # スキップ


def generate_pairs_with_retry(
    nodes: List[Any],
    num_pairs: int,
    grid_width: Optional[int],
    grid_height: Optional[int],
    task_idx: int,
    generator: Any,
    enable_replacement: bool,
    is_regeneration: bool = False,
    temporary_input_grid: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    """複数ペアを生成し、リトライロジックを適用

    Args:
        nodes: プログラムノード
        num_pairs: 生成するペア数
        grid_width: グリッド幅（Noneの場合は各ペアで決定）
        grid_height: グリッド高さ（Noneの場合は各ペアで決定）
        task_idx: タスクインデックス
        generator: プログラムジェネレータ
        enable_replacement: 置き換えを有効化するか
        is_regeneration: 再生成ループかどうか
        temporary_input_grid: 仮のインプットグリッド（1つ目のペアで使用、Noneの場合は通常通り生成）

    Returns:
        (pairs_data, validated_nodes): ペアデータのリストと検証済みノード（最初のペアのもの）
    """
    pairs_data = []
    validated_nodes = None
    saved_all_commands = None
    saved_program_code = None
    task_prefix = f"再生成タスク{task_idx}" if is_regeneration else f"タスク{task_idx}"

    def select_size_with_proximity_bias(original_size: int, size_seed: int = None) -> int:
        """元のサイズに近い値を高い確率で選択する（1-30の範囲内）

        Args:
            original_size: 元のグリッドサイズ
            size_seed: サイズ決定用のシード（Noneの場合は現在のrandom状態を使用）

        Returns:
            選択されたサイズ（1-30の範囲内）
        """
        # シードが指定されている場合は設定
        if size_seed is not None:
            temp_state = random.getstate()
            random.seed(size_seed)

        try:
            # 固定の標準偏差を使用（少し大きめの分散で多様性を確保）
            std_dev = 5.0

            # 1-30の範囲で重み付きランダム選択
            # 元のサイズに近い値ほど高い重みを持つ
            candidates = list(range(1, 31))
            weights = []
            for size in candidates:
                # 距離を計算（元のサイズからの差分）
                distance = abs(size - original_size)
                # 重み = exp(-(distance^2) / (2 * std_dev^2)) （正規分布的な重み）
                weight = math.exp(-(distance ** 2) / (2 * std_dev ** 2))
                weights.append(weight)

            # 重み付きランダム選択
            selected_size = random.choices(candidates, weights=weights, k=1)[0]

            return selected_size
        finally:
            # シードが指定されていた場合は元の状態に戻す
            if size_seed is not None:
                random.setstate(temp_state)

    for pair_idx in range(num_pairs):
        # 1番目のペアでは、グリッドサイズは固定（grid_width, grid_heightを使用）
        # 2番目以降のペアでは、リトライループ内で最初のグリッドサイズに近い値を選択（すべての試行で）
        pair_grid_width = grid_width
        pair_grid_height = grid_height
        # 最初以外のペアでは、最初のペアで抽出したコマンドとプログラムコードを再利用
        pair_all_commands = saved_all_commands if pair_idx > 0 else None
        pair_program_code = saved_program_code if pair_idx > 0 else None

        # ペア生成成功フラグ
        pair_generated = False

        # 最初以外のペアでリトライループ（最大MAX_PAIR_RETRIES回）を実装
        # 最初のペアの場合はリトライなし（タスク廃棄のため）
        max_retries = MAX_PAIR_RETRIES if pair_idx > 0 else 0

        for retry_count in range(max_retries + 1):
            try:
                # リトライごとにシードを変更（異なる入力グリッドを生成）
                # シードは task_idx * 10000 + pair_idx * 100 + retry_count として一意にする
                # リトライ時は追加のランダム性を加えて多様性を向上（リトライ回数が高い場合により大きなランダム性を加える）
                base_seed = task_idx * 10000 + pair_idx * 100 + retry_count
                if retry_count > 0:
                    # リトライ時は、シードにランダムオフセットを追加（リトライ回数が増えるほど大きなオフセット）
                    # リトライ回数に応じて、より大きなランダム性を加える（最大10000）
                    # オフセット生成用の一時的なシードを設定（base_seedをベースに一意なシードを生成）
                    temp_seed = base_seed * 31 + 17  # ハッシュ関数のようなもの
                    random.seed(temp_seed)
                    np.random.seed(temp_seed % (2**32))
                    random_offset = random.randint(0, min(10000, retry_count * 100))
                    pair_seed = base_seed + random_offset
                else:
                    # 最初の試行は決定論的シードを使用（再現性を保つ）
                    pair_seed = base_seed
                random.seed(pair_seed)
                np.random.seed(pair_seed % (2**32))  # numpy用に32bitに制限

                # 2番目以降のペアでは、すべての試行（最初の試行も含む）で最初のグリッドサイズに近い値を高い確率で選択（固定の分散）
                retry_grid_width = pair_grid_width
                retry_grid_height = pair_grid_height
                if pair_idx > 0 and grid_width is not None and grid_height is not None:
                    # すべての試行（最初の試行も含む）で、元のサイズ（grid_width, grid_height）に近い値を高い確率で選択
                    retry_size_seed = pair_seed * 7 + 13  # グリッドサイズ決定用の一意なシード
                    retry_grid_width = select_size_with_proximity_bias(grid_width, size_seed=retry_size_seed)
                    retry_grid_height = select_size_with_proximity_bias(grid_height, size_seed=retry_size_seed + 1)

                    if retry_count == 0:
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [グリッドサイズ] ペア{pair_idx+1}/{num_pairs}の最初の試行: {retry_grid_width}x{retry_grid_height} (元: {grid_width}x{grid_height})", flush=True)
                    else:
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [グリッドサイズ変更] リトライ{retry_count}回目: {pair_grid_width}x{pair_grid_height} → {retry_grid_width}x{retry_grid_height} (元: {grid_width}x{grid_height})", flush=True)

                if pair_idx > 0 and retry_count > 0:
                    print(f"  [リトライ] ペア {pair_idx+1}/{num_pairs} をリトライします（{retry_count}/{max_retries}回目、シード={pair_seed}、オフセット={pair_seed - base_seed}）", flush=True)

                # 最初のペアで temporary_input_grid が提供されている場合は使用する
                # ただし、条件に合致しない場合は他のペアと同様に廃棄される
                use_temporary_input_grid = (pair_idx == 0 and temporary_input_grid is not None and retry_count == 0)
                input_grid_for_first_pair = temporary_input_grid if use_temporary_input_grid else None

                if use_temporary_input_grid:
                    print(f"  [部分プログラムフロー] 最初のペア（1/{num_pairs}）で部分プログラム生成に使用したインプットグリッドを使用します", flush=True)

                # 実行時間を監視して無限ループを検出
                exec_wrapper_start = time.time()
                result_nodes, input_grid, output_grid, trace_results = core_executor_main(
                    nodes,
                    grid_width=retry_grid_width,
                    grid_height=retry_grid_height,
                    task_index=task_idx,
                    enable_replacement=enable_replacement,
                    all_commands=pair_all_commands,
                    program_code=pair_program_code,
                    is_first_pair=(pair_idx == 0),
                    input_grid=input_grid_for_first_pair
                )
                exec_wrapper_elapsed = time.time() - exec_wrapper_start

                # 最初のペアで検証されたnodesとコマンド・プログラムコードを保存（後続のペアでも使用）
                if pair_idx == 0:
                    validated_nodes = result_nodes
                    # 最初のペアのnodesを後続のペアでも使用
                    nodes = validated_nodes

                    # 最初のペアで抽出したコマンドとプログラムコードを保存（最適化のため）
                    try:
                        saved_all_commands = extract_all_commands_from_nodes(validated_nodes)
                        # プログラムコードを生成
                        context = ProgramContext(complexity=1)
                        try:
                            saved_program_code = generator._generate_code(validated_nodes, context, preserve_indentation=True)
                        except TypeError:
                            saved_program_code = generator._generate_code(validated_nodes, context)
                    except Exception as e:
                        # エラーが発生した場合は、次回も抽出するようにNoneのままにする
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [警告] {task_prefix}: 最初のペアでコマンド・プログラムコード抽出に失敗: {e}", flush=True)

                # 実行時間チェック
                if exec_wrapper_elapsed > MAX_EXECUTION_TIME:
                    if pair_idx == 0:
                        # 最初のペアで実行時間が上限を超えた場合、タスク廃棄
                        error_msg = f"{task_prefix}: 最初のペアの実行時間が異常に長いため破棄します（{exec_wrapper_elapsed:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）。無限ループの可能性があります。"
                        print(f"  [警告] {error_msg}", flush=True)
                        raise ValueError(error_msg)
                    else:
                        # 最初以外のペアで実行時間が上限を超えた場合、リトライ
                        if retry_count < max_retries:
                            print(f"  [リトライ] ペア {pair_idx+1}/{num_pairs} の実行時間が上限を超えました（{exec_wrapper_elapsed:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）。リトライします（{retry_count+1}/{max_retries+1}回目）。", flush=True)
                            continue  # リトライ
                        else:
                            # リトライ上限に達した場合、ペアスキップ
                            print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} の実行時間が上限を超えました（{exec_wrapper_elapsed:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）。リトライ上限に達しました。このペアをスキップします。", flush=True)
                            pair_generated = False
                            break  # リトライループを終了

                # 重複入力グリッドのチェック（2番目以降のペアのみ）
                has_duplicate = False
                if pair_idx > 0:
                    # 既存のペアの入力グリッドと比較
                    input_grid_bytes = input_grid.tobytes()
                    for existing_pair in pairs_data:
                        existing_input = existing_pair.get('input_grid')
                        if existing_input is not None:
                            existing_input_bytes = existing_input.tobytes()
                            if input_grid_bytes == existing_input_bytes:
                                has_duplicate = True
                                break  # 重複が見つかったので、内側のforループを終了

                    if has_duplicate:
                        # 重複が見つかった場合、リトライ
                        if retry_count < max_retries:
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [重複検出] ペア {pair_idx+1}/{num_pairs} の入力グリッドが既存のペアと重複しています。リトライします（{retry_count+1}/{max_retries+1}回目、シード={pair_seed}）。", flush=True)
                            continue  # リトライ
                        else:
                            # リトライ上限に達した場合、ペアスキップ
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [スキップ] ペア {pair_idx+1}/{num_pairs} の入力グリッドが重複しています。リトライ上限に達しました。このペアをスキップします。", flush=True)
                            pair_generated = False
                            break  # リトライループを終了

                # 重複チェックをパスした場合（または最初のペアの場合）、ペアを追加
                # 各ペアのデータを保存（置き換え判定で使用するため）
                pairs_data.append({
                    'pair_index': pair_idx,
                    'input_grid': input_grid,
                    'output_grid': output_grid,
                    'nodes': result_nodes,
                    'execution_time': exec_wrapper_elapsed,
                    # データセット用のトレース（ENABLE_TRACE_FOR_DATASET=True の場合のみ有効）
                    'trace_results': trace_results,
                })

                print(f"  ペア {pair_idx+1}/{num_pairs} 生成完了" + (f"（リトライ{retry_count}回目）" if retry_count > 0 else ""), flush=True)

                # ペア生成成功
                pair_generated = True
                break  # リトライループを終了

            except (ValueError, Exception) as pair_error:
                # ペア生成エラーの処理
                should_retry, pair_generated = handle_pair_generation_error(
                    pair_error, pair_idx, num_pairs, retry_count, max_retries, is_first_pair=(pair_idx == 0)
                )
                if should_retry:
                    continue  # リトライ
                else:
                    break  # リトライループを終了（ペアスキップ）

        # ペアが生成できなかった場合、スキップして次のペアへ
        if not pair_generated:
            continue  # 次のペアへ

        # 注: 各ペアでシードを明示的に設定しているため、random.seed()のリセットは不要
        # 次のペアのループ開始時に、pair_seedで再度シードを設定するため

    return pairs_data, validated_nodes


def extract_first_pair_data(pairs_data: List[Dict[str, Any]]) -> Tuple[Any, Any, Any, float]:
    """最初のペアデータを抽出

    Args:
        pairs_data: ペアデータのリスト

    Returns:
        (result_nodes, input_grid, output_grid, execution_time): 最初のペアのデータ
    """
    first_pair = pairs_data[0]
    return (
        first_pair['nodes'],
        first_pair['input_grid'],
        first_pair['output_grid'],
        first_pair['execution_time']
    )


def validate_grid_data(
    input_grid: Any,
    output_grid: Any,
    task_idx: int,
    rejection_reason_stats: Dict[str, int],
    is_regeneration: bool = False,
    raise_on_error: bool = False
) -> bool:
    """グリッドデータの妥当性をチェック

    Args:
        input_grid: 入力グリッド
        output_grid: 出力グリッド
        task_idx: タスクインデックス
        rejection_reason_stats: 破棄原因別の統計
        is_regeneration: 再生成ループかどうか
        raise_on_error: Trueの場合、エラー時に例外を発生（メインループ用）

    Returns:
        True: 妥当、False: 無効（raise_on_error=Falseの場合）

    Raises:
        ValueError: raise_on_error=Trueでエラーが検出された場合
    """
    task_prefix = f"再生成タスク{task_idx}" if is_regeneration else f"タスク{task_idx}"

    # 形状チェック
    if not isinstance(input_grid, np.ndarray) or not isinstance(output_grid, np.ndarray):
        error_msg = f"{task_prefix}: input_gridまたはoutput_gridがnumpy配列ではありません（型: input={type(input_grid)}, output={type(output_grid)}）"
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            print(f"  [再生成失敗] {error_msg}", flush=True)
            increment_rejection_stat(rejection_reason_stats, 'grid_not_array')
            return False

    # 空のグリッドチェック
    if input_grid.size == 0 or output_grid.size == 0:
        error_msg = f"{task_prefix}: input_gridまたはoutput_gridが空です（サイズ: input={input_grid.size}, output={output_grid.size}）"
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            print(f"  [再生成失敗] {error_msg}", flush=True)
            increment_rejection_stat(rejection_reason_stats, 'grid_empty')
            return False

    return True


def collect_commands_to_check(validated_nodes: List[Any]) -> List[Dict[str, Any]]:
    """置き換え対象のコマンド情報を収集

    Args:
        validated_nodes: 検証済みノードリスト

    Returns:
        commands_to_check: コマンド情報のリスト
    """
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

    return commands_to_check


def increment_rejection_stat(
    rejection_reason_stats: Dict[str, int],
    stat_key: str,
    increment_total: bool = True
) -> None:
    """統計情報を1つ増やす（ヘルパー関数）

    Args:
        rejection_reason_stats: 破棄原因別の統計
        stat_key: 統計キー名
        increment_total: Trueの場合、totalも増やす
    """
    rejection_reason_stats[stat_key] += 1
    if increment_total:
        rejection_reason_stats['total'] += 1


def check_all_pairs_conditions(
    pairs_data: List[Dict[str, Any]],
    task_idx: int,
    condition_rejection_stats: Dict[str, int],
    rejection_reason_stats: Dict[str, int],
    is_regeneration: bool = False
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """すべてのペアの条件をチェックし、無効なペアを返す

    Args:
        pairs_data: ペアデータのリスト
        task_idx: タスクインデックス
        condition_rejection_stats: 条件別の統計
        rejection_reason_stats: 破棄原因別の統計
        is_regeneration: 再生成ループかどうか

    Returns:
        (invalid_pairs, valid_pairs_data): 無効なペアのインデックスリストと有効なペアデータ
    """
    invalid_pairs = []  # 条件に該当するペアのインデックスを収集
    task_prefix = f"再生成タスク{task_idx}" if is_regeneration else f"タスク{task_idx}"

    for pair_data in pairs_data:
        pair_input_grid = pair_data['input_grid']
        pair_output_grid = pair_data['output_grid']

        if ENABLE_OUTPUT_CONDITION_CHECK and pair_input_grid is not None and pair_output_grid is not None:
            # 各ペアの条件チェック
            pair_input_grid_array = np.asarray(pair_input_grid, dtype=int)
            pair_output_grid_array = np.asarray(pair_output_grid, dtype=int)

            # 入力グリッドから背景色を推論
            pair_input_unique, pair_input_counts = np.unique(pair_input_grid_array, return_counts=True)
            pair_inferred_background_color = pair_input_unique[np.argmax(pair_input_counts)] if len(pair_input_unique) > 0 else None

            pair_condition1, pair_condition2, pair_condition3, pair_condition4_original, pair_condition5, pair_condition6_original = check_output_conditions(
                pair_input_grid_array,
                pair_output_grid_array,
                background_color=pair_inferred_background_color
            )

            # 条件④と条件⑥の確率的な破棄
            pair_condition4 = pair_condition4_original
            pair_condition4_kept_probabilistically = False
            if pair_condition4_original:
                if random.random() < CONDITION4_KEEP_PROBABILITY:
                    pair_condition4 = False
                    pair_condition4_kept_probabilistically = True
            pair_condition6 = pair_condition6_original
            pair_condition6_kept_probabilistically = False
            if pair_condition6_original:
                if random.random() < CONDITION6_KEEP_PROBABILITY:
                    pair_condition6 = False
                    pair_condition6_kept_probabilistically = True

            # 条件に該当するかどうかを判定
            pair_meets_condition = pair_condition1 or pair_condition2 or pair_condition3 or pair_condition4 or pair_condition5 or pair_condition6

            # 確率的に残された場合でも、他の条件（①、②、③、⑤）に該当する場合は無効とする
            if pair_condition4_kept_probabilistically or pair_condition6_kept_probabilistically:
                pair_other_conditions = pair_condition1 or pair_condition2 or pair_condition3 or pair_condition5
                if pair_other_conditions:
                    # 他の条件に該当する場合は、このペアを無効とマーク
                    invalid_pairs.append(pair_data['pair_index'])
                    # 統計情報を集計
                    update_rejection_stats(
                        pair_condition1, pair_condition2, pair_condition3, pair_condition4, pair_condition5, pair_condition6,
                        condition_rejection_stats, rejection_reason_stats
                    )
                    print(f"  [無効] {task_prefix}: ペア{pair_data['pair_index']+1}が確率的に残されたが、他の条件（①、②、③、⑤）に該当するため無効", flush=True)
                else:
                    # 他の条件に該当しない場合は、このペアは通過（統計には記録）
                    condition_rejection_stats['total'] += 1
                    rejection_reason_stats['total'] += 1
                    if pair_condition4_kept_probabilistically:
                        condition_rejection_stats['condition4'] += 1
                        rejection_reason_stats['condition4'] += 1
                    if pair_condition6_kept_probabilistically:
                        condition_rejection_stats['condition6'] += 1
                        rejection_reason_stats['condition6'] += 1
            elif pair_meets_condition:
                # 確率的に残されず、条件に該当する場合は、このペアを無効とマーク
                invalid_pairs.append(pair_data['pair_index'])
                # 統計情報を集計
                update_rejection_stats(
                    pair_condition1, pair_condition2, pair_condition3, pair_condition4, pair_condition5, pair_condition6,
                    condition_rejection_stats, rejection_reason_stats
                )
                print(f"  [無効] {task_prefix}: ペア{pair_data['pair_index']+1}が条件に該当するため無効", flush=True)

    # 有効なペアデータを返す
    valid_pairs_data = [p for p in pairs_data if p['pair_index'] not in invalid_pairs]
    return invalid_pairs, valid_pairs_data


def save_grids_to_json(execution_results: List[Tuple], output_path: str):
    """実行結果をJSON形式で保存（arc-agi_evaluation_challenges.jsonと同じ形式）"""
    challenges_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for i, (result_nodes, input_grid, output_grid, filename) in enumerate(execution_results, 1):
        if input_grid is None or output_grid is None:
            continue

        task_id = f"task_{i:03d}"
        input_list = convert_grid_to_list(input_grid)
        output_list = convert_grid_to_list(output_grid)

        challenges_dict[task_id] = {
            "train": [
                {
                    "input": input_list,
                    "output": output_list
                }
            ],
            "test": []
        }

    # JSONファイルとして保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(challenges_dict, f, ensure_ascii=False, indent=2)

    print(f"グリッドデータをJSON形式で保存: {output_path} ({len(challenges_dict)}個のタスク)")


def main(mode: str = "full", output_dir: Optional[str] = None):
    """メイン関数

    Args:
        mode: 実行モード
            - "full": プログラム生成と実行（デフォルト）
            - "program-only": プログラム生成のみを行い実行はしない
            - "input-grid-only": インプットグリッドのみを生成してPNGとして保存
        output_dir: 出力ディレクトリ（指定しない場合は新しいタイムスタンプ付きディレクトリを作成）
    """
    # モードの検証
    valid_modes = ["full", "program-only", "input-grid-only"]
    if mode not in valid_modes:
        raise ValueError(f"無効なモード: {mode}。有効なモード: {', '.join(valid_modes)}")

    # ログ出力制御: 環境変数を設定（すべてのモジュールで使用）
    # デフォルトで詳細ログを無効化（工程やタスク数などの重要なログのみ出力）
    os.environ['ENABLE_VERBOSE_LOGGING'] = 'false'
    os.environ['ENABLE_ALL_LOGS'] = 'false'

    # 実行時間計測開始
    start_time = time.time()
    start_datetime = datetime.now()

    # モード判定
    mode_display_names = {
        "full": "プログラム生成と実行",
        "program-only": "プログラム生成のみ",
        "input-grid-only": "インプットグリッド生成のみ"
    }
    mode_str = mode_display_names.get(mode, mode)

    print(f"{mode_str}を開始")
    print(f"開始時刻: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TASK_COUNT: {TASK_COUNT}")

    # インプットグリッドのみ生成モードの場合は、専用処理を実行して終了
    if mode == "input-grid-only":
        # 出力ディレクトリの設定
        if output_dir is None:
            # 新しいタイムスタンプ付きディレクトリを作成
            base_output_dir = "outputs/input_grid_only"
            timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_output_dir, timestamp_dir)
            print(f"新規出力ディレクトリを作成: {output_dir}")
        else:
            # 既存のディレクトリを使用
            if not os.path.exists(output_dir):
                raise ValueError(f"指定された出力ディレクトリが存在しません: {output_dir}")
            print(f"既存の出力ディレクトリを使用: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")

        # タイミング統計を初期化
        timing_stats = TimingStatistics()

        # インプットグリッド生成のみを実行
        generate_input_grids_only(output_dir, timing_stats)

        # タイミング統計を出力
        timing_stats.print_statistics()

        # 実行時間計測終了
        end_time = time.time()
        end_datetime = datetime.now()
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60

        print(f"\n{'='*80}")
        print(f"【実行完了】")
        print(f"{'='*80}")
        print(f"開始時刻: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"終了時刻: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {total_seconds:.2f}秒 ({total_minutes:.2f}分 / {total_hours:.2f}時間)")

        return

    # 比率を正規化
    normalized_ratios = normalize_ratios(COMPLEXITY_RATIOS)
    original_total = sum(COMPLEXITY_RATIOS.values())

    # 設定確認
    print_settings()
    if abs(original_total - 100.0) > 0.01:
        print(f"注: 比率を自動正規化しました（元の合計: {original_total}%）")

    # 出力ディレクトリの設定
    if output_dir is None:
        # 新しいタイムスタンプ付きディレクトリを作成（ルールベースパイプライン）
        base_output_dir = "outputs/rule_based"
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp_dir)
        print(f"新規出力ディレクトリを作成（ルールベースパイプライン）: {output_dir}")
    else:
        # 既存のディレクトリを使用（続きから実行）
        if not os.path.exists(output_dir):
            raise ValueError(f"指定された出力ディレクトリが存在しません: {output_dir}")
        print(f"既存の出力ディレクトリを使用: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")

    # コマンド使用統計をリセット
    reset_global_statistics()
    statistics = get_global_statistics()

    # タイミング統計を初期化
    timing_stats = TimingStatistics()

    generator = UnifiedProgramGenerator()

    # 既存の重み調整ファイルがあれば読み込み
    existing_adjustments = load_weight_adjustments(output_dir)

    # 関係性生成コマンドの初期重み調整を適用
    if _config.enable_relationship_command_weights:
        # 既存の重み調整と統合（既存の重み調整がある場合は、それに初期重み調整を乗算）
        if existing_adjustments:
            # 既存の重み調整に初期重み調整を統合
            for cmd, weight in _config.relationship_command_weights.items():
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
            generator.command_weight_adjustments = _config.relationship_command_weights.copy()
            print(f"  関係性生成コマンドの初期重み調整を適用: {len(_config.relationship_command_weights)}個のコマンド")
            if ENABLE_VERBOSE_LOGGING:
                for cmd, weight in _config.relationship_command_weights.items():
                    print(f"    {cmd}: {weight:.2f}x")
    else:
        # 関係性生成コマンドの重み調整が無効な場合は、既存の重み調整のみを適用
        if existing_adjustments:
            generator.command_weight_adjustments = existing_adjustments
            print(f"  読み込んだ重み調整を適用: {len(existing_adjustments)}個のコマンド")

        # 統計の総プログラム数を復元（参考情報として）
        restored_total = load_statistics_from_file(output_dir, statistics)
        if restored_total > 0:
            print(f"  前回の総プログラム数: {restored_total}（統計は新規開始）")

    # バッチ進捗を読み込み
    batch_progress = load_batch_progress(output_dir)
    start_task_index = 0
    if batch_progress:
        start_task_index = batch_progress['next_start_task']
        print(f"\n【バッチ再開モード】")
        print(f"  前回完了: バッチ{batch_progress['current_batch']}, {batch_progress['completed_tasks']}タスク")
        print(f"  再開位置: バッチ{batch_progress['next_batch']}から開始（タスク{start_task_index}から）")

        # TASK_COUNTを調整（再開位置から残りのタスク数を計算）
        if start_task_index >= TASK_COUNT:
            print(f"\n  [警告] 再開位置({start_task_index})がTASK_COUNT({TASK_COUNT})を超えています")
            print(f"  バッチ単位で処理を継続します")

    # ファイルバッファマネージャーを初期化
    # 自動フラッシュを無効化し、バッチ完了時に手動でフラッシュする
    test_batch_size = min(BATCH_SIZE, max(10, TASK_COUNT // 10 + 1))  # テスト時は適切なサイズに調整（最低10、最大BATCH_SIZE）
    if TASK_COUNT < BATCH_SIZE:
        print(f"\n【テストモード】バッチサイズを調整: {BATCH_SIZE} → {test_batch_size}")
        buffer_manager = FileBufferManager(output_dir, auto_flush=False)  # 自動フラッシュ無効
        buffer_manager.BATCH_SIZE = test_batch_size  # テスト用にバッチサイズを調整
    else:
        buffer_manager = FileBufferManager(output_dir, auto_flush=False)  # 自動フラッシュ無効
    # グローバルに設定（node_validator_variable.pyなどからアクセス可能に）
    set_global_buffer_manager(buffer_manager)

    # ニューラルモデル用学習データ生成器を初期化（遅延インポート）
    from src.data_systems.generator.neural_training_data_generator import NeuralTrainingDataGenerator
    neural_data_generator = NeuralTrainingDataGenerator(output_dir)

    # バッチごとにループ（各バッチ内で生成→実行→再生成→保存を完結）
    execution_results = []
    max_regeneration_attempts = 1000  # 最大再生成試行回数

    # 条件別の破棄数集計（バッチ全体で集計）
    condition_rejection_stats = {
        'condition1': 0,  # 条件①: 入力と出力が完全一致
        'condition2': 0,  # 条件②: トリミング一致
        'condition3': 0,  # 条件③: 入力が単色
        'condition4': 0,  # 条件④: オブジェクトピクセル数5%未満かつ出力グリッドの総色数が2以下
        'condition5': 0,  # 条件⑤: 空のoutput（0ピクセル）
        'condition6': 0,  # 条件⑥: 出力が単色
        'multiple': 0,    # 複数条件に該当
        'total': 0        # 総破棄数
    }

    # すべてのタスク破棄原因を集計（詳細な原因別統計）
    rejection_reason_stats = {
        # データ不備
        'program_data_none': 0,  # プログラムデータがNone
        'nodes_empty': 0,  # プログラムノードが空
        'program_code_empty': 0,  # プログラムコードが空
        # 実行エラー
        'silent_exception': 0,  # SilentException（タイムアウト）
        'execution_timeout': 0,  # 実行時間が異常に長い
        'result_nodes_none': 0,  # result_nodesがNone
        'input_grid_none': 0,  # input_gridがNone
        'output_grid_none': 0,  # output_gridがNone
        'object_count_limit': 0,  # オブジェクト数上限超過
        'index_error': 0,  # IndexError（リストアクセスエラー）
        'program_error': 0,  # プログラム実行エラー（TypeError, AttributeError, KeyError, ZeroDivisionError）
        'validation_error': 0,  # バリデーションエラー
        'other_value_error': 0,  # その他のValueError
        'validate_result_none': 0,  # validate_nodes_and_adjust_objectsの結果がNone
        'validate_timeout': 0,  # validate_nodes_and_adjust_objectsのタイムアウト
        'validate_exception': 0,  # validate_nodes_and_adjust_objectsの例外
        'other_error': 0,  # その他の予期しないエラー
        # 再生成時のエラー
        'regeneration_result_none': 0,  # 再生成時の結果がNone
        'grid_not_array': 0,  # グリッドがnumpy配列ではない
        'grid_empty': 0,  # グリッドが空
        # 出力条件
        'condition1': 0,  # 条件①（入力と出力が完全一致）
        'condition2': 0,  # 条件②（トリミング一致）
        'condition3': 0,  # 条件③（入力が単色）
        'condition6': 0,  # 条件⑥（出力が単色）
        'condition4': 0,  # 条件④（オブジェクトピクセル数5%未満かつ出力グリッドの総色数が2以下）
        'condition5': 0,  # 条件⑤（空のoutput）
        'total': 0  # 総破棄数
    }

    # 総バッチ数を計算
    total_tasks = TASK_COUNT
    num_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE  # 切り上げ

    print(f"\nプログラム生成・実行を開始...")
    print(f"  総タスク数: {total_tasks}")
    print(f"  バッチ数: {num_batches} (1バッチあたり最大{BATCH_SIZE}タスク)")
    if batch_progress and start_task_index > 0:
        start_batch_num = start_task_index // BATCH_SIZE
        print(f"  再開: バッチ{start_batch_num + 1}から開始（タスク{start_task_index + 1}から）")

    # リソース測定用の変数（バッチ間で比較するため）
    prev_threads_count = 0
    prev_memory_mb = 0

    # 開始バッチ番号を決定（再開時は該当バッチから開始）
    start_batch_num = 0
    if batch_progress and start_task_index > 0:
        start_batch_num = start_task_index // BATCH_SIZE

    for batch_num in range(start_batch_num, num_batches):
        batch_start_idx = batch_num * BATCH_SIZE
        batch_end_idx = min((batch_num + 1) * BATCH_SIZE, total_tasks)
        batch_tasks = batch_end_idx - batch_start_idx

        # バッチ内の開始タスクインデックスを決定（再開時はstart_task_indexから）
        batch_internal_start = 0
        if batch_num == start_batch_num and start_task_index > 0:
            batch_internal_start = start_task_index - batch_start_idx

        # バッチ開始時のリソース測定（蓄積状況を確認）
        import threading

        threads_before = threading.active_count()
        thread_names_before = [t.name for t in threading.enumerate()]
        # ThreadPoolExecutorのスレッドをカウント
        threadpool_threads = [t for t in threading.enumerate() if 'ThreadPoolExecutor' in str(t) or 'Thread-' in t.name]

        memory_before = 0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # psutilがインストールされていない場合はスキップ
            pass
        except Exception:
            pass

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
        batch_programs = []
        batch_task_grid_sizes = []
        batch_complexities = []

        # このバッチで生成するタスクの範囲を決定
        for i in range(batch_start_idx, batch_end_idx):
            # 再開時は、開始位置より前のタスクはスキップ
            if i < start_task_index:
                continue

            complexity = select_complexity(normalized_ratios)
            batch_complexities.append(complexity)

            grid_width, grid_height = None, None
            # タスク処理開始時間を計測
            task_start_time = time.time()
            try:
                # グリッドサイズを決定（プログラム生成前に決定）
                grid_size_start = time.time()
                try:
                    grid_width, grid_height = decide_grid_size()
                    elapsed = time.time() - grid_size_start
                    if ENABLE_VERBOSE_OUTPUT:
                        print(f"[TIME] タスク{i+1} decide_grid_size: {elapsed:.3f}秒", flush=True)
                    batch_task_grid_sizes.append((grid_width, grid_height))
                    timing_stats.record_step("グリッドサイズ決定（バッチ生成）", elapsed, success=True)
                except Exception as e:
                    elapsed = time.time() - grid_size_start
                    print(f"[警告] タスク{i+1} グリッドサイズ決定エラー: {e}", flush=True)
                    batch_task_grid_sizes.append((None, None))
                    timing_stats.record_step("グリッドサイズ決定（バッチ生成）", elapsed, success=False)
                    raise

                # プログラム生成
                # 環境変数で新しいフロー（部分プログラム使用）を有効化可能（デフォルト: true）
                use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true').lower() in ('true', '1', 'yes')
                if use_partial_program_flow:
                    nodes, program_code, grid_width, grid_height, temporary_input_grid = generate_program_with_partial_program_flow(generator, complexity, output_dir, i+1, grid_width, grid_height, buffer_manager, timing_stats)
                    # 仮のインプットグリッドをバッファに追加（部分プログラムフローの場合のみ）
                    if temporary_input_grid is not None and buffer_manager is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        buffer_manager.add_temporary_input_grid(i+1, temporary_input_grid, timestamp)
                else:
                    nodes, program_code, grid_width, grid_height, temporary_input_grid = generate_program(generator, complexity, output_dir, i+1, grid_width, grid_height, buffer_manager, timing_stats)
                batch_programs.append((nodes, program_code, grid_width, grid_height, temporary_input_grid))

                # タスク処理成功時の時間を計測
                task_elapsed = time.time() - task_start_time
                timing_stats.record_step("タスク生成（成功）", task_elapsed, success=True)
                if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[TIME] タスク{i+1} 生成完了（成功）: {task_elapsed:.3f}秒", flush=True)
            except Exception as e:
                # タスク処理失敗時の時間を計測
                task_elapsed = time.time() - task_start_time
                timing_stats.record_step("タスク生成（失敗）", task_elapsed, success=False)
                print(f"[ERROR] タスク{i+1}の生成に失敗: {type(e).__name__}: {e} (処理時間: {task_elapsed:.3f}秒)", flush=True)
                if (grid_width, grid_height) not in batch_task_grid_sizes:
                    batch_task_grid_sizes.append((None, None))
                # エラー時はNoneを追加してスキップ
                batch_programs.append(None)

        print(f"[バッチ {batch_num + 1} プログラム生成完了] {len([p for p in batch_programs if p is not None])}/{len(batch_programs)}個のタスクを生成")

        # プログラム生成のみモードの場合は、このバッチの生成後にバッファをフラッシュして次のバッチへ
        if mode == "program-only":
            # プログラムJSONバッファをフラッシュ
            buffer_manager.flush_all()
            # メモリ解放
            batch_programs = None
            batch_task_grid_sizes = None
            batch_complexities = None
            gc.collect()
            continue

        # このバッチ内の破棄タスクリスト
        rejected_task_indices = []  # バッチ内でのローカルインデックス（1始まり）

        # 条件を満たしたタスクのIDセット（バッチ終了時にフィルタリング用）
        valid_task_ids = set()  # {"task_001", "task_002", ...} の形式

        # エラーカウンター（バッチごとにリセット）
        consecutive_errors = 0


        # バッチ処理開始時はクリーンアップなし（処理速度向上のため、定期実行に統一）

        # 初回実行ループ（このバッチのみ）
        # batch_internal_startから開始（再開時は該当位置から）
        for local_idx_offset, program_data in enumerate(batch_programs[batch_internal_start:], batch_internal_start + 1):
            local_idx = local_idx_offset
            global_idx = batch_start_idx + local_idx  # グローバルなタスクインデックス

            # program_data を展開
            if program_data is None:
                continue
            nodes, program_code, grid_width, grid_height, temporary_input_grid = program_data

            # タスク全体の処理時間を計測
            task_loop_start_time = time.time()
            # デフォルトで無効化（詳細ログ）
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [デバッグ] バッチ {batch_num + 1}: タスク{global_idx}の処理開始 ({local_idx}/{len(batch_programs)})", flush=True)

            # Noneの場合はスキップ（既に展開済み）
            # program_data は既に展開されているため、ここではスキップチェックのみ
            if nodes is None or program_code is None:
                print(f"  [警告] タスク{global_idx}: プログラムデータがNoneです。スキップします。", flush=True)
                increment_rejection_stat(rejection_reason_stats, 'program_data_none')
                rejected_task_indices.append(local_idx)
                continue

            # filename を取得
            if isinstance(program_data, tuple) and len(program_data) >= 3:
                filename = program_data[2] if len(program_data) > 2 else f"task_{global_idx:03d}.json"
            else:
                filename = f"task_{global_idx:03d}.json"

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

            if ENABLE_ALL_LOGS:
                print(f"\n[実行開始] タスク{global_idx} (バッチ内: {local_idx}/{batch_tasks}): {os.path.basename(filename)}", flush=True)
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and grid_width and grid_height:
                print(f"  グリッドサイズ: {grid_width}x{grid_height}", flush=True)

            try:
                if ENABLE_DEBUG_OUTPUT:
                    print(f"[DEBUG] main: core_executor_main呼び出し前 (タスク{global_idx})", flush=True)
                execution_start_time = time.time()

                # core_executor_mainにグリッドサイズを渡す（タイムアウト付き）
                try:
                    if ENABLE_DEBUG_OUTPUT:
                        print(f"[DEBUG] main: run_with_timeout呼び出し直前 (タスク{global_idx}, nodes数={len(nodes)}, grid_size={grid_width}x{grid_height})", flush=True)
                    # プログラム実行をタイムアウト付きで実行（異常に長い実行時間を防ぐ）
                    if ENABLE_DEBUG_OUTPUT:
                        print(f"[DEBUG] main: lambda関数作成完了 (タスク{global_idx})", flush=True)

                    # 複数ペア生成: 1つのプログラムから3-10個の入出力ペアを生成
                    num_pairs = random.randint(MIN_PAIRS_PER_PROGRAM, MAX_PAIRS_PER_PROGRAM)
                    print(f"タスク {global_idx}/{TASK_COUNT} - 複数ペア生成開始 ({num_pairs}個のペア)", flush=True)

                    # すべてのペアで検証を有効化（ステップ3-4を実行）
                    # ただし、置き換え判定（ステップ5）はスキップ（後で全ペアの結果を使用して実行）
                    enable_replacement = False  # ステップ5はスキップ（後で全ペアの結果を使用）

                    # 共通関数でペア生成
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
                    # 注: このチェックは早期の破棄のため。後続の条件チェックでも同様のチェックを実施
                    if len(pairs_data) < MIN_PAIRS_PER_PROGRAM:
                        print(f"  [破棄] タスク{global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {MIN_PAIRS_PER_PROGRAM}個）", flush=True)
                        rejected_task_indices.append(local_idx)
                        continue

                    # 最初のペアの結果を使用
                    result_nodes, input_grid, output_grid, exec_wrapper_elapsed = extract_first_pair_data(pairs_data)

                    # 検証されたnodesを保存（最初のペアで検証されたnodes）
                    if validated_nodes:
                        result_nodes = validated_nodes

                    # 注: 最初のペアの実行時間チェックはgenerate_pairs_with_retry内で既に実施済み
                    # 実行時間が上限を超えた場合は、generate_pairs_with_retry内でValueErrorが発生し、タスクが破棄される

                    # None値チェックを強化
                    validate_execution_results(
                        result_nodes, input_grid, output_grid, global_idx, rejection_reason_stats, is_regeneration=False
                    )

                    # 入力グリッドと出力グリッドが同じ場合（信頼度0.0が返された場合）も検出
                    if isinstance(output_grid, np.ndarray) and isinstance(input_grid, np.ndarray):
                        if np.array_equal(input_grid, output_grid):
                            print(f"  [警告] タスク{global_idx}: input_gridとoutput_gridが同一です。プログラムが正しく実行されていません。", flush=True)

                    # 進捗メッセージを表示
                    print(f"タスク {global_idx}/{TASK_COUNT} - 実行完了", flush=True)
                    if ENABLE_DEBUG_OUTPUT:
                        print(f"[DEBUG] main: run_with_timeout呼び出し完了 (タスク{global_idx}, result_nodes={'None' if result_nodes is None else 'not None'})", flush=True)
                    if ENABLE_DEBUG_OUTPUT:
                        print(f"[DEBUG] main: core_executor_main呼び出し完了 (タスク{global_idx})", flush=True)

                    # 結果の妥当性チェック（既にvalidate_execution_resultsでチェック済み、ここではグリッドデータの妥当性のみチェック）
                    # 入力グリッドと出力グリッドの妥当性チェック
                    validate_grid_data(
                        input_grid, output_grid, global_idx, rejection_reason_stats,
                        is_regeneration=False, raise_on_error=True
                    )

                    # 正常実行できた場合はエラーカウンターをリセット
                    consecutive_errors = 0
                except Exception as exec_error:
                    # SilentException（タイムアウト）を最優先で捕捉（スタックトレース抑制のため）
                    from src.core_systems.executor.core import SilentException
                    exec_error_elapsed = time.time() - execution_start_time
                    total_task_error_elapsed = time.time() - task_loop_start_time
                    # ここで一度メッセージを整形しておく（型に関わらず使用）
                    error_msg = str(exec_error).replace("SilentException: ", "").replace("src.core_systems.executor.core.SilentException: ", "")
                    if isinstance(exec_error, SilentException):
                        consecutive_errors += 1
                        print(f"  [エラー] タスク{global_idx} 実行エラー: {error_msg[:100]} (実行時間: {exec_error_elapsed:.3f}秒, タスク全体: {total_task_error_elapsed:.3f}秒)", flush=True)
                        increment_rejection_stat(rejection_reason_stats, 'silent_exception')
                        timing_stats.record_step("プログラム実行", 0, success=False)
                        rejected_task_indices.append(local_idx)
                        continue  # 即座に次のタスクへ
                    # ValueError（プログラム実行エラーによりタスク破棄）を優先処理
                    if isinstance(exec_error, ValueError) and "タスクが破棄されました" in str(exec_error):
                        execution_time = time.time() - execution_start_time
                        timing_stats.record_step("プログラム実行", execution_time, success=False)
                        # オブジェクト数上限の場合は特別なメッセージを表示
                        if "オブジェクト数上限" in str(exec_error):
                            print(f"  [再生成対象] タスク{global_idx}: オブジェクト数が上限を超えたため、タスクを破棄して再生成します", flush=True)
                            increment_rejection_stat(rejection_reason_stats, 'object_count_limit')
                        else:
                            # エラーメッセージから詳細な原因を特定
                            error_str = str(exec_error)
                            if "validate_nodes_and_adjust_objectsの結果がNone" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの結果がNoneによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_result_none')
                            elif "validate_nodes_and_adjust_objectsのタイムアウト" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsのタイムアウトによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_timeout')
                            elif "validate_nodes_and_adjust_objectsの例外" in error_str:
                                # エラーメッセージから例外の詳細を抽出
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
                        timing_stats.record_step("プログラム実行", execution_time, success=False)
                        rejected_task_indices.append(local_idx)
                        continue  # 再生成対象として次へ
                    else:
                        # その他のIndexErrorは致命的なエラーとして扱う
                        print(f"  [廃棄] タスク{global_idx} 実行中にプログラムエラーが発生: {type(index_error).__name__}: {index_error} → 即座に廃棄", flush=True)
                        timing_stats.record_step("プログラム実行", execution_time, success=False)
                        rejected_task_indices.append(local_idx)
                    continue
                except (TypeError, AttributeError, KeyError, ZeroDivisionError) as exec_error:
                    # プログラム実行エラー（致命的なエラー）の場合は即座に廃棄
                    execution_time = time.time() - execution_start_time
                    print(f"  [廃棄] タスク{global_idx} 実行中にプログラムエラーが発生: {type(exec_error).__name__}: {exec_error} → 即座に廃棄", flush=True)
                    increment_rejection_stat(rejection_reason_stats, 'program_error')
                    timing_stats.record_step("プログラム実行", execution_time, success=False)
                    rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                    continue  # 即座に廃棄して次へ
                except ValueError as validation_error:
                    # ValueError（プログラム実行エラーによりタスク破棄）を優先チェック
                    if "タスクが破棄されました" in str(validation_error):
                        execution_time = time.time() - execution_start_time
                        timing_stats.record_step("プログラム実行", execution_time, success=False)
                        # オブジェクト数上限の場合は特別なメッセージを表示
                        if "オブジェクト数上限" in str(validation_error):
                            print(f"  [再生成対象] タスク{global_idx}: オブジェクト数が上限を超えたため、タスクを破棄して再生成します", flush=True)
                            increment_rejection_stat(rejection_reason_stats, 'object_count_limit')
                        else:
                            # エラーメッセージから詳細な原因を特定
                            error_str = str(validation_error)
                            if "validate_nodes_and_adjust_objectsの結果がNone" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsの結果がNoneによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_result_none')
                            elif "validate_nodes_and_adjust_objectsのタイムアウト" in error_str:
                                print(f"  [再生成対象] タスク{global_idx}: validate_nodes_and_adjust_objectsのタイムアウトによりタスクを破棄して再生成します", flush=True)
                                increment_rejection_stat(rejection_reason_stats, 'validate_timeout')
                            elif "validate_nodes_and_adjust_objectsの例外" in error_str:
                                # エラーメッセージから例外の詳細を抽出
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
                    timing_stats.record_step("プログラム実行", execution_time, success=False)
                    rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                    # ガベージコレクションは定期実行に統一（処理速度向上のため）
                    continue  # 再生成対象として次へ
                except Exception as exec_error:
                    # その他の予期しないエラーの場合も再生成対象にする
                    execution_time = time.time() - execution_start_time
                    total_task_error_elapsed = time.time() - task_loop_start_time
                    consecutive_errors += 1
                    print(f"  [再生成対象] タスク{global_idx} 実行中に予期しないエラーが発生: {type(exec_error).__name__}: {exec_error} → 再生成します (実行時間: {execution_time:.3f}秒, タスク全体: {total_task_error_elapsed:.3f}秒)", flush=True)
                    increment_rejection_stat(rejection_reason_stats, 'other_error')
                    timing_stats.record_step("プログラム実行", execution_time, success=False)
                    rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                    # ガベージコレクションは定期実行に統一（処理速度向上のため）
                    continue  # 再生成対象として次へ

                # タスク全体の処理完了ログ
                total_task_elapsed = time.time() - task_loop_start_time
                if total_task_elapsed > 5.0:  # 5秒を超えた場合は警告
                    print(f"  [警告] タスク{global_idx}: タスク全体の処理時間が長いです（{total_task_elapsed:.3f}秒）", flush=True)

                # タスクフォルダーのパスを取得
                # task_dir = os.path.dirname(filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 複数ペア生成時は、最初のペアだけでタスクを破棄しない
                # すべてのペアをチェックしてから判定する（以下の「すべてのペアの条件チェック」を使用）

                # 複数ペアの条件チェック: すべてのペアが条件を満たしている場合のみ保存
                # 改善: すべてのペアで条件に該当する場合のみタスク廃棄、一部のペアのみの場合はペアスキップ
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
                    if len(pairs_data) < MIN_PAIRS_PER_PROGRAM:
                        print(f"  [破棄] タスク{global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {MIN_PAIRS_PER_PROGRAM}個）", flush=True)
                        rejected_task_indices.append(local_idx)
                        continue

                    print(f"  [続行] タスク{global_idx}: {len(pairs_data)}個のペアが残りました（スキップ: {len(invalid_pairs)}個）", flush=True)

                    # すべてのペアが条件を満たしている場合: 保存処理に進む（既存のコードをそのまま使用）

                    # 条件に該当しない場合：有効なタスクとして記録
                    if ENABLE_ALL_LOGS:
                        print(f"  [保存] タスク{global_idx}: すべてのペアが条件を満たしているため保存 ({len(pairs_data)}個のペア)", flush=True)
                    valid_task_ids.add(f"task_{global_idx:03d}")
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] タスク{global_idx}: valid_task_idsに追加しました (現在の数: {len(valid_task_ids)})", flush=True)

                    # ステップ5: コマンド置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                    # すべてのペアの結果を使用して置き換え判定を実行
                    if validated_nodes and len(pairs_data) > 0 and ENABLE_OUTPUT_CONDITION_CHECK:
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
                                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                                print(f"  [置き換え適用] タスク{global_idx}: コマンド'{cmd_name}'を'{fallback_code}'に置き換えました（すべてのペアで不要と判断されました）", flush=True)
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
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [警告] タスク{global_idx}: 置き換え判定でエラーが発生しました: {type(replacement_error).__name__}: {replacement_error}", flush=True)

                    # すべてのペアのグリッドデータをバッファに追加（条件を満たしたタスクのみ）
                    # 注: バッチ終了時にフィルタリングするため、ここでは追加のみ

                    # PNGデータはすべてのペアを追加（複数ペア対応）
                    # すべてのペア生成後に、すべてのペアデータを渡す
                    _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                    first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                    first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                    buffer_manager.add_png_data(
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
                        buffer_manager.add_grid_json(
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
                            if buffer_manager:
                                buffer_manager.update_program_json(global_idx, updated_program_code)
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [置き換え適用] タスク{global_idx}: 置き換え後のプログラムコードを更新しました (長さ={len(updated_program_code)}文字)", flush=True)
                        except Exception as e:
                            if ENABLE_ALL_LOGS:
                                print(f"  [警告] タスク{global_idx}: 置き換え後のプログラムコード生成に失敗: {e}", flush=True)

                    # コマンド使用統計に追加
                    if result_nodes:
                        commands = extract_all_commands_from_nodes(result_nodes)
                        statistics.add_program_commands(commands)
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] タスク{global_idx}: statistics.add_program_commandsを呼び出しました (total_programs: {statistics.total_programs})", flush=True)
                    else:
                        print(f"  [警告] タスク{global_idx}: result_nodesがNoneのため、statisticsに追加しませんでした", flush=True)

                    # ニューラルモデル用学習データを生成（有効なタスクのみ）
                    if f"task_{global_idx:03d}" in valid_task_ids:
                        try:
                            # 置き換え後のprogram_codeを取得
                            task_complexity = batch_complexities[local_idx - 1] if (local_idx - 1) < len(batch_complexities) else 1
                            context = ProgramContext(task_complexity, grid_width=grid_width, grid_height=grid_height)
                            try:
                                final_program_code = generator._generate_code(result_nodes, context, preserve_indentation=True)
                            except TypeError:
                                final_program_code = generator._generate_code(result_nodes, context)

                            # 変数インデックスを正規化
                            final_program_code = normalize_variable_indices(final_program_code)

                            # 各ペアごとに学習データを生成
                            for pair_idx, pair_data in enumerate(pairs_data):
                                pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                neural_data_generator.generate_from_generator_output(
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
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [警告] タスク{global_idx}: 学習データ生成でエラーが発生しました: {e}", flush=True)

                    # 保存完了メッセージを表示
                    print(f"タスク {global_idx}/{TASK_COUNT} - 保存完了", flush=True)
                else:
                    # 条件チェックが無効な場合、すべて有効として扱う（複数ペア対応）
                    valid_task_ids.add(f"task_{global_idx:03d}")
                    print(f"  [デバッグ] タスク{global_idx}: 条件チェック無効 - valid_task_idsに追加しました (現在の数: {len(valid_task_ids)})", flush=True)

                    # PNGデータはすべてのペアを追加（複数ペア対応）
                    # すべてのペア生成後に、すべてのペアデータを渡す
                    _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                    first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                    first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                    buffer_manager.add_png_data(
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
                        buffer_manager.add_grid_json(
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
                            if buffer_manager:
                                buffer_manager.update_program_json(global_idx, updated_program_code)

                            # ニューラルモデル用学習データを生成（条件チェック無効の場合）
                            try:
                                # 各ペアごとに学習データを生成
                                for pair_idx, pair_data in enumerate(pairs_data):
                                    pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                                    pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)

                                    neural_data_generator.generate_from_generator_output(
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
                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                    print(f"  [警告] タスク{global_idx}: 学習データ生成でエラーが発生しました: {e}", flush=True)

                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [置き換え適用] タスク{global_idx}: 条件チェック無効 - 置き換え後のプログラムコードを更新しました (長さ={len(updated_program_code)}文字)", flush=True)
                        except Exception as e:
                            if ENABLE_ALL_LOGS:
                                print(f"  [警告] タスク{global_idx}: 条件チェック無効 - 置き換え後のプログラムコード生成に失敗: {e}", flush=True)

                    # コマンド使用統計に追加
                    if result_nodes:
                        commands = extract_all_commands_from_nodes(result_nodes)
                        statistics.add_program_commands(commands)
                        print(f"  [デバッグ] タスク{global_idx}: 条件チェック無効 - statistics.add_program_commandsを呼び出しました (total_programs: {statistics.total_programs})", flush=True)

                        # 1000タスクごとに重み調整を適用・保存（100タスクの場合はスキップ）
                        if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                            if ENABLE_ALL_LOGS:
                                print(f"\n【1000タスク到達: 重み調整を適用】")
                            adjustments = statistics.get_weight_adjustments()
                            if adjustments:
                                generator.command_weight_adjustments = adjustments
                                if ENABLE_ALL_LOGS:
                                    print(f"  調整対象: {len(adjustments)}個のコマンド")
                                    statistics.print_statistics()

                                # 重み調整をファイルに保存
                                save_weight_adjustments(output_dir, adjustments, statistics.total_programs)

                                # バッチ進捗を保存
                                current_batch = statistics.total_programs // BATCH_SIZE
                                save_batch_progress(output_dir, statistics.total_programs, current_batch)
                            else:
                                if ENABLE_ALL_LOGS:
                                    print(f"  調整対象のコマンドなし")
                    else:
                        print(f"  [警告] タスク{global_idx}: 条件チェック無効 - result_nodesがNoneのため、statisticsに追加しませんでした", flush=True)

                    # 保存完了メッセージを表示
                    print(f"タスク {global_idx}/{TASK_COUNT} - 保存完了", flush=True)

                execution_results.append((result_nodes, input_grid, output_grid, filename))
            except Exception as e:
                # SilentExceptionは既に処理済みなので、スタックトレースを出力しない
                from src.core_systems.executor.core import SilentException
                if isinstance(e, SilentException):
                    # SilentExceptionの場合は最小限の処理のみ（処理速度向上のため即座に次へ）
                    execution_time = time.time() - execution_start_time
                    consecutive_errors += 1
                    # rejected_task_indicesに追加（まだ追加されていない場合）
                    if local_idx not in rejected_task_indices:
                        rejected_task_indices.append(local_idx)
                    timing_stats.record_step("プログラム実行", execution_time, success=False)
                    # ガベージコレクションは定期実行に統一（処理速度向上のため）
                    execution_results.append((None, None, None, filename))
                    continue  # 即座に次のタスクへ（処理速度向上）

                # その他の例外の場合のみスタックトレースを出力
                execution_time = time.time() - execution_start_time
                consecutive_errors += 1
                print(f"[ERROR] 実行に失敗: {type(e).__name__}: {e} (経過時間: {execution_time:.1f}秒)", flush=True)
                # スタックトレースは出力しない（処理速度向上のため）
                # エラーが発生したタスクも再生成対象にする
                print(f"  [破棄] タスク{global_idx}: 実行エラーが発生したため破棄して再生成", flush=True)
                rejection_reason_stats['other_error'] += 1
                rejection_reason_stats['total'] += 1
                rejected_task_indices.append(local_idx)  # バッチ内のローカルインデックス
                # ガベージコレクションは定期実行に統一（処理速度向上のため）
                execution_results.append((None, None, None, filename))

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
            # 早期終了機能は無効化されています（max_consecutive_regen_failures = float('inf')）
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

                # デバッグ: ループの開始を確認（最初の5タスクと20タスクごと）
                # デフォルトで無効化（詳細ログ）
                if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and (task_idx <= 5 or task_idx % 20 == 0):
                    print(f"  [デバッグ] 再生成ループ {regeneration_attempt}: タスク{rejected_global_idx}の処理開始 ({task_idx}/{total_regen_tasks})", flush=True)

                # ステップ1: プログラム再生成
                # デフォルトで無効化（詳細ログ）
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: プログラム再生成開始", flush=True)
                regen_gen_start_time = time.time()
                try:
                    # 再生成時は必ず新しいパラメータをランダムに選択（同じパラメータで再生成すると同じようなプログラムが生成されるため）
                    # 元の複雑度とは異なる複雑度を選択
                    original_complexity = batch_complexities[rejected_local_idx - 1]
                    available_complexities = [c for c in normalized_ratios.keys() if c != original_complexity]
                    if available_complexities:
                        new_complexity = random.choice(available_complexities)
                    else:
                        # 利用可能な複雑度がない場合は、比率に基づいてランダム選択
                        new_complexity = select_complexity(normalized_ratios)

                    # 新しいグリッドサイズをランダムに決定（通常の決定ロジックを使用）
                    grid_width, grid_height = decide_grid_size()

                    print(f"  [再生成] タスク{rejected_global_idx}: パラメータを変更 (複雑度: {original_complexity}→{new_complexity}, サイズ: {grid_width}x{grid_height})", flush=True)
                    regen_complexity = new_complexity

                    # 再生成（タイムアウトやエラーが発生した場合はスキップ）
                    try:
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: generate_program呼び出し前", flush=True)
                        generate_start_time = time.time()
                        # 環境変数で新しいフロー（部分プログラム使用）を有効化可能（デフォルト: true）
                        use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true').lower() in ('true', '1', 'yes')
                        if use_partial_program_flow:
                            nodes, program_code, grid_width, grid_height, temporary_input_grid = generate_program_with_partial_program_flow(generator, regen_complexity, output_dir, rejected_global_idx, grid_width, grid_height, buffer_manager, timing_stats)
                        else:
                            nodes, program_code, grid_width, grid_height, temporary_input_grid = generate_program(generator, regen_complexity, output_dir, rejected_global_idx, grid_width, grid_height, buffer_manager, timing_stats)
                        generate_elapsed = time.time() - generate_start_time
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: generate_program完了 (処理時間: {generate_elapsed:.3f}秒)", flush=True)
                        batch_programs[rejected_local_idx - 1] = (nodes, program_code, grid_width, grid_height, temporary_input_grid)  # バッチ内のリストを更新
                        # 再生成時のパラメータを保存（次回の再生成で使用）
                        batch_complexities[rejected_local_idx - 1] = regen_complexity
                        batch_task_grid_sizes[rejected_local_idx - 1] = (grid_width, grid_height)
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: プログラムデータ更新完了", flush=True)
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
                    continue

                # program_data を展開
                nodes, program_code, grid_width, grid_height, temporary_input_grid = program_data

                # Noneの場合はスキップ（再生成ループに回す）
                if nodes is None or program_code is None:
                    print(f"  [再生成失敗] タスク{rejected_global_idx}: プログラムデータがNoneです。再生成ループに回します。", flush=True)
                    new_rejected_indices.append(rejected_local_idx)
                    continue

                filename = f"task_{rejected_global_idx:03d}.json"

                # 再生成データの妥当性チェック
                if not nodes or len(nodes) == 0 or not program_code or not program_code.strip():
                    print(f"  [再生成失敗] タスク{rejected_global_idx}: プログラムノードまたはコードが空です。再生成ループに回します。", flush=True)
                    new_rejected_indices.append(rejected_local_idx)
                    continue  # 次のタスクへ

                # ステップ2: 再生成されたプログラムを即座に実行（2フェーズに分けない = 処理速度向上）
                regen_gen_elapsed = time.time() - regen_gen_start_time
                # デフォルトで無効化（詳細ログ）
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: プログラム再生成完了 (総処理時間: {regen_gen_elapsed:.3f}秒)", flush=True)
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: プログラム実行開始", flush=True)
                regen_exec_start = time.time()
                try:
                    # 複数ペア生成: 1つのプログラムから3-10個の入出力ペアを生成
                    num_pairs = random.randint(MIN_PAIRS_PER_PROGRAM, MAX_PAIRS_PER_PROGRAM)
                    print(f"再生成タスク {rejected_global_idx}/{TASK_COUNT} - 複数ペア生成開始 ({num_pairs}個のペア)", flush=True)

                    # すべてのペアで検証を有効化（ステップ3-4を実行）
                    # ただし、置き換え判定（ステップ5）はスキップ（後で全ペアの結果を使用して実行）
                    enable_replacement = False  # ステップ5はスキップ（後で全ペアの結果を使用）

                    # 共通関数でペア生成
                    pairs_data, validated_nodes = generate_pairs_with_retry(
                        nodes, num_pairs, grid_width, grid_height, rejected_global_idx,
                        generator, enable_replacement, is_regeneration=True
                    )

                    # すべてのペアが生成できなかった場合、タスクを破棄
                    if len(pairs_data) == 0:
                        print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: すべてのペアの生成に失敗しました", flush=True)
                        new_rejected_indices.append(rejected_local_idx)
                        continue

                    # ペア数が最小要件（MIN_PAIRS_PER_PROGRAM）未満の場合: タスク廃棄（学習データとして不十分）
                    # 注: このチェックは早期の破棄のため。後続の条件チェックでも同様のチェックを実施
                    if len(pairs_data) < MIN_PAIRS_PER_PROGRAM:
                        print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {MIN_PAIRS_PER_PROGRAM}個）", flush=True)
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

                    # 注: 最初のペアの実行時間チェックはgenerate_pairs_with_retry内で既に実施済み
                    # 実行時間が上限を超えた場合は、generate_pairs_with_retry内でValueErrorが発生し、タスクが破棄される

                    # None値チェックを強化
                    validate_execution_results(
                        result_nodes, input_grid, output_grid, rejected_global_idx, rejection_reason_stats, is_regeneration=True
                    )

                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 複数ペア生成完了 (処理時間: {exec_elapsed:.3f}秒)", flush=True)
                    # 進捗メッセージを表示（重要なログなので常に表示）
                    print(f"再生成タスク {rejected_global_idx}/{TASK_COUNT} - 複数ペア実行完了 ({len(pairs_data)}個のペア)", flush=True)
                    regen_exec_time = time.time() - regen_exec_start
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 実行完了、結果の妥当性チェック開始 (実行時間: {regen_exec_time:.2f}秒)", flush=True)

                    # 結果の妥当性チェック（既にvalidate_execution_resultsでチェック済み、ここではグリッドデータの妥当性のみチェック）

                    # 入力グリッドと出力グリッドの妥当性チェック
                    if not validate_grid_data(
                        input_grid, output_grid, rejected_global_idx, rejection_reason_stats,
                        is_regeneration=True, raise_on_error=False
                    ):
                        new_rejected_indices.append(rejected_local_idx)
                        timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                        continue  # 次のタスクへ

                    timing_stats.record_step("再生成実行", regen_exec_time, success=True)
                except Exception as exec_error:
                    # SilentException（タイムアウト）を最優先で捕捉（スタックトレース抑制のため）
                    from src.core_systems.executor.core import SilentException
                    if isinstance(exec_error, SilentException):
                        regen_exec_time = time.time() - regen_exec_start
                        # SilentExceptionの場合は最小限の処理のみ
                        rejection_reason_stats['silent_exception'] += 1
                        rejection_reason_stats['total'] += 1
                        timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                        new_rejected_indices.append(rejected_local_idx)
                        continue  # 即座に次のタスクへ
                    # ValueError（プログラム実行エラーによりタスク破棄、または異常に長い実行時間）を優先処理
                    if isinstance(exec_error, ValueError):
                        error_msg = str(exec_error)
                        regen_exec_time = time.time() - regen_exec_start
                        timing_stats.record_step("再生成実行", regen_exec_time, success=False)
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
                    rejection_reason_stats['program_error'] += 1
                    rejection_reason_stats['total'] += 1
                    timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                    new_rejected_indices.append(rejected_local_idx)
                    continue  # 即座に次のタスクへ
                except Exception as regen_exec_error:
                    # その他のエラーの場合は再生成を継続（SilentExceptionは既に処理済み）
                    regen_exec_time = time.time() - regen_exec_start
                    from src.core_systems.executor.core import SilentException
                    # SilentExceptionの場合は何も出力しない（処理速度向上のため）
                    rejection_reason_stats['other_error'] += 1
                    rejection_reason_stats['total'] += 1
                    timing_stats.record_step("再生成実行", regen_exec_time, success=False)
                    new_rejected_indices.append(rejected_local_idx)
                    continue  # 再生成ループを継続

                # 再検証（パフォーマンス最適化：スキップ可能）
                # 注: バッファへの追加・削除はバッチ終了時に行うため、ここでは条件チェックのみ
                # 条件チェックが無効な場合や、input_grid/output_gridがNoneの場合はスキップ
                if not ENABLE_OUTPUT_CONDITION_CHECK:
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 条件チェック無効 - 保存処理開始", flush=True)
                elif input_grid is None or output_grid is None:
                    print(f"  [警告] 再生成タスク{rejected_global_idx}: input_gridまたはoutput_gridがNoneのため、条件チェックをスキップ", flush=True)
                    # 条件チェックができない場合は、再度破棄
                    new_rejected_indices.append(rejected_local_idx)
                    continue
                else:
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 条件チェック開始", flush=True)

                if ENABLE_OUTPUT_CONDITION_CHECK and input_grid is not None and output_grid is not None:
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 複数ペアの条件チェック開始", flush=True)

                        # 複数ペアの条件チェック: すべてのペアが条件を満たしている場合のみ保存
                        # 改善: すべてのペアで条件に該当する場合のみタスク廃棄、一部のペアのみの場合はペアスキップ
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
                            if len(pairs_data) < MIN_PAIRS_PER_PROGRAM:
                                print(f"  [再生成失敗] 再生成タスク{rejected_global_idx}: ペア数が最小要件未満のためタスクを破棄します（{len(pairs_data)}個 < {MIN_PAIRS_PER_PROGRAM}個）", flush=True)
                                new_rejected_indices.append(rejected_local_idx)
                                continue

                            print(f"  [続行] 再生成タスク{rejected_global_idx}: {len(pairs_data)}個のペアが残りました（スキップ: {len(invalid_pairs)}個）", flush=True)

                        # すべてのペアが条件を満たしている場合: 保存処理に進む（既存のコードをそのまま使用）

                        # 条件に該当しない場合：有効なタスクとして記録
                        if ENABLE_ALL_LOGS:
                            print(f"  [保存] 再生成タスク{rejected_global_idx}: すべてのペアが条件を満たしているため保存 ({len(pairs_data)}個のペア)", flush=True)
                        valid_task_ids.add(f"task_{rejected_global_idx:03d}")
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: valid_task_idsに追加しました (現在の数: {len(valid_task_ids)})", flush=True)

                        # ステップ5: コマンド置き換え判定（すべてのペアで不要と判断された場合のみ置き換え）
                        # すべてのペアの結果を使用して置き換え判定を実行
                        if validated_nodes and len(pairs_data) > 0 and ENABLE_OUTPUT_CONDITION_CHECK:
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
                                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                                    print(f"  [置き換え適用] 再生成タスク{rejected_global_idx}: コマンド'{cmd_name}'を'{fallback_code}'に置き換えました（すべてのペアで不要と判断されました）", flush=True)
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
                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                    print(f"  [警告] 再生成タスク{rejected_global_idx}: 置き換え判定でエラーが発生しました: {type(replacement_error).__name__}: {replacement_error}", flush=True)

                        # 条件を満たしている場合：有効なタスクとして記録
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 条件を満たしているため保存処理開始", flush=True)
                        valid_task_ids.add(f"task_{rejected_global_idx:03d}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: PNGデータ追加前", flush=True)

                        # PNGデータはすべてのペアを追加（複数ペア対応）
                        # すべてのペア生成後に、すべてのペアデータを渡す
                        first_pair = pairs_data[0]
                        first_pair_input_grid = np.asarray(first_pair['input_grid'], dtype=int)
                        first_pair_output_grid = np.asarray(first_pair['output_grid'], dtype=int)
                        buffer_manager.add_png_data(
                            task_index=rejected_global_idx,
                            input_grid=first_pair_input_grid,
                            output_grid=first_pair_output_grid,
                            timestamp=timestamp,
                            pairs_data=pairs_data  # すべてのペアデータを渡す
                        )
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: PNGデータ追加完了、グリッドJSON追加前", flush=True)

                        # JSONデータはすべてのペアを追加（複数ペア対応）
                        for pair_data in pairs_data:
                            pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                            pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                            trace_results = pair_data.get('trace_results')
                            buffer_manager.add_grid_json(
                                task_index=rejected_global_idx,
                                input_grid=pair_input_grid,
                                output_grid=pair_output_grid,
                                trace_results=trace_results
                            )
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: グリッドJSON追加完了、execution_results追加前", flush=True)

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
                                if buffer_manager:
                                    buffer_manager.update_program_json(rejected_global_idx, updated_program_code)
                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                    print(f"  [置き換え適用] 再生成タスク{rejected_global_idx}: 置き換え後のプログラムコードを更新しました (長さ={len(updated_program_code)}文字)", flush=True)
                            except Exception as e:
                                if ENABLE_ALL_LOGS:
                                    print(f"  [警告] 再生成タスク{rejected_global_idx}: 置き換え後のプログラムコード生成に失敗: {e}", flush=True)

                        execution_results.append((result_nodes, input_grid, output_grid, filename))
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: execution_results追加完了", flush=True)

                        # コマンド使用統計に追加
                        stats_start_time = time.time()
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: statistics.add_program_commands呼び出し前", flush=True)
                        if result_nodes:
                            commands = extract_all_commands_from_nodes(result_nodes)
                            # デフォルトで無効化（詳細ログ）
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: extract_all_commands_from_nodes完了 ({len(commands)}個のコマンド)", flush=True)
                            statistics.add_program_commands(commands)
                            stats_elapsed = time.time() - stats_start_time
                            # デフォルトで無効化（詳細ログ）
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: statistics.add_program_commands完了 (total_programs: {statistics.total_programs}, 処理時間: {stats_elapsed:.3f}秒)", flush=True)

                            # 100タスクごとに重み調整を適用・保存（100タスクの場合はスキップ）
                            if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                                weight_start_time = time.time()
                                # デフォルトで無効化（詳細ログ）
                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 重み調整チェック開始 (total_programs: {statistics.total_programs})", flush=True)
                                if ENABLE_ALL_LOGS:
                                    print(f"\n【100タスク到達: 重み調整を適用】")
                                adjustments = statistics.get_weight_adjustments()
                                # デフォルトで無効化（詳細ログ）
                                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: get_weight_adjustments完了 ({len(adjustments) if adjustments else 0}個の調整)", flush=True)
                                if adjustments:
                                    generator.command_weight_adjustments = adjustments
                                    if ENABLE_ALL_LOGS:
                                        print(f"  調整対象: {len(adjustments)}個のコマンド")
                                        statistics.print_statistics()

                                    # 重み調整をファイルに保存
                                    save_start_time = time.time()
                                    save_weight_adjustments(output_dir, adjustments, statistics.total_programs)
                                    save_elapsed = time.time() - save_start_time
                                    # デフォルトで無効化（詳細ログ）
                                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: save_weight_adjustments完了 (処理時間: {save_elapsed:.3f}秒)", flush=True)

                                    # バッチ進捗を保存
                                    progress_start_time = time.time()
                                    current_batch = statistics.total_programs // BATCH_SIZE
                                    save_batch_progress(output_dir, statistics.total_programs, current_batch)
                                    progress_elapsed = time.time() - progress_start_time
                                    # デフォルトで無効化（詳細ログ）
                                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: save_batch_progress完了 (処理時間: {progress_elapsed:.3f}秒)", flush=True)

                                    weight_elapsed = time.time() - weight_start_time
                                    # デフォルトで無効化（詳細ログ）
                                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 重み調整保存完了 (総処理時間: {weight_elapsed:.3f}秒)", flush=True)
                                else:
                                    if ENABLE_ALL_LOGS:
                                        print(f"  調整対象のコマンドなし")
                        else:
                            print(f"  [警告] 再生成タスク{rejected_global_idx}: result_nodesがNoneのため、statisticsに追加しませんでした", flush=True)

                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 保存処理の最後のステップ開始", flush=True)
                        task_elapsed = time.time() - task_start_time
                        print(f"  [再生成成功] タスク{rejected_global_idx}: 条件を満たしているため保存 (処理時間: {task_elapsed:.2f}秒)")
                        # 保存完了メッセージを表示
                        print(f"タスク {rejected_global_idx}/{TASK_COUNT} - 保存完了", flush=True)
                        # デフォルトで無効化（詳細ログ）
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 保存処理完了、次のタスクへ", flush=True)
                else:
                    # 条件チェックが無効な場合、すべて有効として扱う（複数ペア対応）
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 条件チェック無効 - 保存処理開始 ({len(pairs_data)}個のペア)", flush=True)
                    valid_task_ids.add(f"task_{rejected_global_idx:03d}")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: PNGデータ追加前", flush=True)

                    # PNGデータはすべてのペアを追加（複数ペア対応）
                    # すべてのペア生成後に、すべてのペアデータを渡す
                    _, first_pair_input_grid, first_pair_output_grid, _ = extract_first_pair_data(pairs_data)
                    first_pair_input_grid = np.asarray(first_pair_input_grid, dtype=int)
                    first_pair_output_grid = np.asarray(first_pair_output_grid, dtype=int)
                    buffer_manager.add_png_data(
                        task_index=rejected_global_idx,
                        input_grid=first_pair_input_grid,
                        output_grid=first_pair_output_grid,
                        timestamp=timestamp,
                        pairs_data=pairs_data  # すべてのペアデータを渡す
                    )
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: PNGデータ追加完了、グリッドJSON追加前", flush=True)

                    # JSONデータはすべてのペアを追加（複数ペア対応）
                    for pair_data in pairs_data:
                        pair_input_grid = np.asarray(pair_data['input_grid'], dtype=int)
                        pair_output_grid = np.asarray(pair_data['output_grid'], dtype=int)
                        trace_results = pair_data.get('trace_results')
                        buffer_manager.add_grid_json(
                            task_index=rejected_global_idx,
                            input_grid=pair_input_grid,
                            output_grid=pair_output_grid,
                            trace_results=trace_results
                        )
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: グリッドJSON追加完了、execution_results追加前", flush=True)

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
                            if buffer_manager:
                                buffer_manager.update_program_json(rejected_global_idx, updated_program_code)
                            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                                print(f"  [置き換え適用] 再生成タスク{rejected_global_idx}: 条件チェック無効 - 置き換え後のプログラムコードを更新しました (長さ={len(updated_program_code)}文字)", flush=True)
                        except Exception as e:
                            if ENABLE_ALL_LOGS:
                                print(f"  [警告] 再生成タスク{rejected_global_idx}: 条件チェック無効 - 置き換え後のプログラムコード生成に失敗: {e}", flush=True)

                    execution_results.append((result_nodes, input_grid, output_grid, filename))
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: execution_results追加完了", flush=True)
                    task_elapsed = time.time() - task_start_time
                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 保存処理完了 (処理時間: {task_elapsed:.2f}秒)", flush=True)

                    # コマンド使用統計に追加
                    stats_start_time = time.time()
                    print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: statistics.add_program_commands呼び出し前", flush=True)
                    if result_nodes:
                        commands = extract_all_commands_from_nodes(result_nodes)
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: extract_all_commands_from_nodes完了 ({len(commands)}個のコマンド)", flush=True)
                        statistics.add_program_commands(commands)
                        stats_elapsed = time.time() - stats_start_time
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: statistics.add_program_commands完了 (total_programs: {statistics.total_programs}, 処理時間: {stats_elapsed:.3f}秒)", flush=True)

                        # 100タスクごとに重み調整を適用・保存（100タスクの場合はスキップ）
                        if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
                            weight_start_time = time.time()
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 重み調整チェック開始 (total_programs: {statistics.total_programs})", flush=True)
                            if ENABLE_ALL_LOGS:
                                print(f"\n【100タスク到達: 重み調整を適用】")
                            adjustments = statistics.get_weight_adjustments()
                            print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: get_weight_adjustments完了 ({len(adjustments) if adjustments else 0}個の調整)", flush=True)
                            if adjustments:
                                generator.command_weight_adjustments = adjustments
                                if ENABLE_ALL_LOGS:
                                    print(f"  調整対象: {len(adjustments)}個のコマンド")
                                statistics.print_statistics()

                                # 重み調整をファイルに保存
                                save_start_time = time.time()
                                save_weight_adjustments(output_dir, adjustments, statistics.total_programs)
                                save_elapsed = time.time() - save_start_time
                                print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: save_weight_adjustments完了 (処理時間: {save_elapsed:.3f}秒)", flush=True)

                                # バッチ進捗を保存
                                progress_start_time = time.time()
                                current_batch = statistics.total_programs // BATCH_SIZE
                                save_batch_progress(output_dir, statistics.total_programs, current_batch)
                                progress_elapsed = time.time() - progress_start_time
                                print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: save_batch_progress完了 (処理時間: {progress_elapsed:.3f}秒)", flush=True)

                                weight_elapsed = time.time() - weight_start_time
                                print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 重み調整保存完了 (総処理時間: {weight_elapsed:.3f}秒)", flush=True)
                            else:
                                if ENABLE_ALL_LOGS:
                                    print(f"  調整対象のコマンドなし")
                    else:
                        print(f"  [警告] 再生成タスク{rejected_global_idx}: result_nodesがNoneのため、statisticsに追加しませんでした", flush=True)
                    new_rejected_indices.append(rejected_local_idx)

                    # デフォルトで無効化（詳細ログ）
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [デバッグ] 再生成タスク{rejected_global_idx}: 保存処理完了、次のタスクへ", flush=True)

            # 次のループで再生成するタスクリストを更新
            # デフォルトで無効化（詳細ログ）
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [デバッグ] 再生成ループ {regeneration_attempt}: ループ終了処理開始", flush=True)
            regen_elapsed = time.time() - regen_start
            previous_rejected_count = len(rejected_task_indices)
            rejected_task_indices = new_rejected_indices
            current_rejected_count = len(rejected_task_indices)
            # デフォルトで無効化（詳細ログ）
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [デバッグ] 再生成ループ {regeneration_attempt}: タスクリスト更新完了 ({previous_rejected_count} → {current_rejected_count})", flush=True)

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
            # すべてのタスクが条件を満たした場合（このメッセージは既に出力されている）
            pass

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

            # 最も多い条件を特定
            single_conditions = {
                '条件①': condition_rejection_stats['condition1'],
                '条件②': condition_rejection_stats['condition2'],
                '条件③': condition_rejection_stats['condition3'],
                '条件④': condition_rejection_stats['condition4'],
                '条件⑤': condition_rejection_stats['condition5'],
                '条件⑥': condition_rejection_stats['condition6']
            }
            max_condition = max(single_conditions.items(), key=lambda x: x[1])
            if max_condition[1] > 0:
                print(f"  → 最も多い破棄条件: {max_condition[0]} ({max_condition[1]}回, {max_condition[1] / total_rejections * 100:.1f}%)")
        else:
            print(f"  条件による破棄なし")

        # 詳細な破棄原因別統計を表示
        if rejection_reason_stats['total'] > 0:
            total_rejections = rejection_reason_stats['total']
            print(f"\n【詳細な破棄原因別統計（バッチ {batch_num + 1}）】")
            print(f"  総破棄数: {total_rejections}")

            # データ不備
            print(f"\n【データ不備】")
            print(f"  プログラムデータがNone: {rejection_reason_stats['program_data_none']} ({rejection_reason_stats['program_data_none'] / total_rejections * 100:.1f}%)")
            print(f"  プログラムノードが空: {rejection_reason_stats['nodes_empty']} ({rejection_reason_stats['nodes_empty'] / total_rejections * 100:.1f}%)")
            print(f"  プログラムコードが空: {rejection_reason_stats['program_code_empty']} ({rejection_reason_stats['program_code_empty'] / total_rejections * 100:.1f}%)")

            # 実行エラー
            print(f"\n【実行エラー】")
            print(f"  SilentException（タイムアウト）: {rejection_reason_stats['silent_exception']} ({rejection_reason_stats['silent_exception'] / total_rejections * 100:.1f}%)")
            print(f"  実行時間が異常に長い: {rejection_reason_stats['execution_timeout']} ({rejection_reason_stats['execution_timeout'] / total_rejections * 100:.1f}%)")
            print(f"  result_nodesがNone: {rejection_reason_stats['result_nodes_none']} ({rejection_reason_stats['result_nodes_none'] / total_rejections * 100:.1f}%)")
            print(f"  input_gridがNone: {rejection_reason_stats['input_grid_none']} ({rejection_reason_stats['input_grid_none'] / total_rejections * 100:.1f}%)")
            print(f"  output_gridがNone: {rejection_reason_stats['output_grid_none']} ({rejection_reason_stats['output_grid_none'] / total_rejections * 100:.1f}%)")
            print(f"  オブジェクト数上限超過: {rejection_reason_stats['object_count_limit']} ({rejection_reason_stats['object_count_limit'] / total_rejections * 100:.1f}%)")
            print(f"  IndexError（リストアクセスエラー）: {rejection_reason_stats['index_error']} ({rejection_reason_stats['index_error'] / total_rejections * 100:.1f}%)")
            print(f"  プログラム実行エラー: {rejection_reason_stats['program_error']} ({rejection_reason_stats['program_error'] / total_rejections * 100:.1f}%)")
            print(f"  バリデーションエラー: {rejection_reason_stats['validation_error']} ({rejection_reason_stats['validation_error'] / total_rejections * 100:.1f}%)")
            print(f"  その他のValueError: {rejection_reason_stats['other_value_error']} ({rejection_reason_stats['other_value_error'] / total_rejections * 100:.1f}%)")
            print(f"  validate_nodes_and_adjust_objectsの結果がNone: {rejection_reason_stats['validate_result_none']} ({rejection_reason_stats['validate_result_none'] / total_rejections * 100:.1f}%)")
            print(f"  validate_nodes_and_adjust_objectsのタイムアウト: {rejection_reason_stats['validate_timeout']} ({rejection_reason_stats['validate_timeout'] / total_rejections * 100:.1f}%)")
            print(f"  validate_nodes_and_adjust_objectsの例外: {rejection_reason_stats['validate_exception']} ({rejection_reason_stats['validate_exception'] / total_rejections * 100:.1f}%)")
            print(f"  その他の予期しないエラー: {rejection_reason_stats['other_error']} ({rejection_reason_stats['other_error'] / total_rejections * 100:.1f}%)")

            # 再生成時のエラー
            print(f"\n【再生成時のエラー】")
            print(f"  再生成時の結果がNone: {rejection_reason_stats['regeneration_result_none']} ({rejection_reason_stats['regeneration_result_none'] / total_rejections * 100:.1f}%)")
            print(f"  グリッドがnumpy配列ではない: {rejection_reason_stats['grid_not_array']} ({rejection_reason_stats['grid_not_array'] / total_rejections * 100:.1f}%)")
            print(f"  グリッドが空: {rejection_reason_stats['grid_empty']} ({rejection_reason_stats['grid_empty'] / total_rejections * 100:.1f}%)")

            # 出力条件
            print(f"\n【出力条件】")
            print(f"  条件①（入力と出力が完全一致）: {rejection_reason_stats['condition1']} ({rejection_reason_stats['condition1'] / total_rejections * 100:.1f}%)")
            print(f"  条件②（トリミング一致）: {rejection_reason_stats['condition2']} ({rejection_reason_stats['condition2'] / total_rejections * 100:.1f}%)")
            print(f"  条件③（入力が単色）: {rejection_reason_stats['condition3']} ({rejection_reason_stats['condition3'] / total_rejections * 100:.1f}%)")
            print(f"  条件④（オブジェクトピクセル数5%未満かつ総色数2以下）: {rejection_reason_stats['condition4']} ({rejection_reason_stats['condition4'] / total_rejections * 100:.1f}%)")
            print(f"  条件⑤（空のoutput）: {rejection_reason_stats['condition5']} ({rejection_reason_stats['condition5'] / total_rejections * 100:.1f}%)")
            print(f"  条件⑥（出力が単色）: {rejection_reason_stats['condition6']} ({rejection_reason_stats['condition6'] / total_rejections * 100:.1f}%)")

            # 最も多い破棄原因を特定
            reason_categories = {
                'データ不備': rejection_reason_stats['program_data_none'] + rejection_reason_stats['nodes_empty'] + rejection_reason_stats['program_code_empty'],
                '実行エラー': rejection_reason_stats['silent_exception'] + rejection_reason_stats['execution_timeout'] + rejection_reason_stats['result_nodes_none'] + rejection_reason_stats['input_grid_none'] + rejection_reason_stats['output_grid_none'] + rejection_reason_stats['object_count_limit'] + rejection_reason_stats['index_error'] + rejection_reason_stats['program_error'] + rejection_reason_stats['validation_error'] + rejection_reason_stats['other_value_error'] + rejection_reason_stats['validate_result_none'] + rejection_reason_stats['validate_timeout'] + rejection_reason_stats['validate_exception'] + rejection_reason_stats['other_error'],
                '再生成時のエラー': rejection_reason_stats['regeneration_result_none'] + rejection_reason_stats['grid_not_array'] + rejection_reason_stats['grid_empty'],
                '出力条件': rejection_reason_stats['condition1'] + rejection_reason_stats['condition2'] + rejection_reason_stats['condition3'] + rejection_reason_stats['condition4'] + rejection_reason_stats['condition5'] + rejection_reason_stats['condition6']
            }
            max_category = max(reason_categories.items(), key=lambda x: x[1])
            if max_category[1] > 0:
                print(f"\n  → 最も多い破棄カテゴリ: {max_category[0]} ({max_category[1]}回, {max_category[1] / total_rejections * 100:.1f}%)")

            # 個別の破棄原因で最も多いもの
            all_reasons = {k: v for k, v in rejection_reason_stats.items() if k != 'total'}
            max_reason = max(all_reasons.items(), key=lambda x: x[1])
            if max_reason[1] > 0:
                reason_name_map = {
                    'program_data_none': 'プログラムデータがNone',
                    'nodes_empty': 'プログラムノードが空',
                    'program_code_empty': 'プログラムコードが空',
                    'silent_exception': 'SilentException（タイムアウト）',
                    'execution_timeout': '実行時間が異常に長い',
                    'result_nodes_none': 'result_nodesがNone',
                    'input_grid_none': 'input_gridがNone',
                    'output_grid_none': 'output_gridがNone',
                    'object_count_limit': 'オブジェクト数上限超過',
                    'index_error': 'IndexError（リストアクセスエラー）',
                    'program_error': 'プログラム実行エラー',
                    'validation_error': 'バリデーションエラー',
                    'other_value_error': 'その他のValueError',
                    'validate_result_none': 'validate_nodes_and_adjust_objectsの結果がNone',
                    'validate_timeout': 'validate_nodes_and_adjust_objectsのタイムアウト',
                    'validate_exception': 'validate_nodes_and_adjust_objectsの例外',
                    'other_error': 'その他の予期しないエラー',
                    'regeneration_result_none': '再生成時の結果がNone',
                    'grid_not_array': 'グリッドがnumpy配列ではない',
                    'grid_empty': 'グリッドが空',
                    'condition1': '条件①（入力と出力が完全一致）',
                    'condition2': '条件②（トリミング一致）',
                    'condition3': '条件③（入力が単色）',
                    'condition6': '条件⑥（出力が単色）',
                    'condition4': '条件④（オブジェクトピクセル数5%未満かつ総色数2以下）',
                    'condition5': '条件⑤（空のoutput）'
                }
                reason_display_name = reason_name_map.get(max_reason[0], max_reason[0])
                print(f"  → 最も多い破棄原因: {reason_display_name} ({max_reason[1]}回, {max_reason[1] / total_rejections * 100:.1f}%)")

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
        for i in range(len(execution_results) - 1, -1, -1):
            result_nodes, input_grid, output_grid, filename = execution_results[i]
            # ファイル名からタスクインデックスを抽出
            try:
                task_idx = int(filename.split("task_")[-1].split(".")[0])
                if batch_start_global <= task_idx <= batch_end_global:
                    # このバッチのデータはグリッド配列をNoneにしてメモリを解放（filenameは保持）
                    execution_results[i] = (result_nodes, None, None, filename)
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
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
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

        # ニューラルモデル用学習データを保存（バッチごと）
        try:
            neural_data_generator.flush_batch(batch_num)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  → バッチ {batch_num + 1} のニューラルモデル用学習データを保存しました", flush=True)
        except Exception as e:
            print(f"  [警告] バッチ {batch_num + 1} の学習データ保存でエラーが発生しました: {e}", flush=True)

        # このバッチのバッファをフラッシュ（プログラムJSON、グリッドJSON、PNGを保存）
        # 再生成も含めてすべてのタスクが完了した後で一括保存
        # バッチ終了時に、条件を満たしたタスクのみをフィルタリングして保存
        print(f"  → バッチ {batch_num + 1} のバッファをフラッシュ中...", flush=True)

        # このバッチ範囲内のデータのみを処理
        batch_start_task = batch_start_idx + 1  # 1始まりに変換
        batch_end_task = batch_end_idx  # 1始まり

        # 条件を満たしたタスクのみをフィルタリング（バッチ範囲内かつvalid_task_idsに含まれる）
        # JSONバッファから条件を満たしたタスクのみを抽出してフラッシュ（ソート済み）
        print(f"  → JSONバッファのフラッシュ開始 (バッチ {batch_num + 1})...", flush=True)
        for json_type in ["program", "tokens", "stats", "grid"]:
            print(f"    → {json_type} JSONバッファを処理中...", flush=True)
            buffer = getattr(buffer_manager, f"{json_type}_json_buffer")
            original_buffer = getattr(buffer_manager, f"{json_type}_json_buffer")
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
                setattr(buffer_manager, f"{json_type}_json_buffer", batch_items)
                buffer_manager._flush_json(json_type, batch_num)

            # バッチ範囲内のすべてのデータをバッファから削除（条件を満たさないタスクも含む）
            # これにより、条件を満たさないタスクのデータがバッファに残らないようにする
            remaining_items = [
                item for item in original_buffer
                if not (batch_start_task <= int(item.get("task_id", "task_000").split("_")[-1]) <= batch_end_task)
            ]
            setattr(buffer_manager, f"{json_type}_json_buffer", remaining_items)

        # PNGバッファから条件を満たしたタスクのみを抽出してフラッシュ（ソート済み）
        print(f"  → PNGバッファのフラッシュ開始 (バッチ {batch_num + 1})...", flush=True)
        batch_png_data = [
            task_data for task_data in buffer_manager.png_buffer
            if (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task
                and f"task_{task_data.get('task_index', 0):03d}" in valid_task_ids)
        ]
        if batch_png_data:
            # task_indexでソート（1, 2, 3, ...の順序）
            batch_png_data = sorted(batch_png_data, key=lambda x: x.get("task_index", 0))
            # 一時的にバッファを置き換えてフラッシュ
            original_buffer = buffer_manager.png_buffer.copy()
            buffer_manager.png_buffer = batch_png_data
            buffer_manager._flush_png(batch_num)
            # フラッシュ済みのデータをバッファから削除（バッチ範囲内のすべてのデータを削除）
            buffer_manager.png_buffer = [
                task_data for task_data in original_buffer
                if not (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task)
            ]

        # 仮のインプットグリッドPNGバッファのフラッシュ（部分プログラムフロー用）
        print(f"  → 仮のインプットグリッドPNGバッファのフラッシュ開始 (バッチ {batch_num + 1})...", flush=True)
        batch_temporary_input_grid_data = [
            task_data for task_data in buffer_manager.temporary_input_grid_buffer
            if (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task
                and f"task_{task_data.get('task_index', 0):03d}" in valid_task_ids)
        ]
        if batch_temporary_input_grid_data:
            # task_indexでソート（1, 2, 3, ...の順序）
            batch_temporary_input_grid_data = sorted(batch_temporary_input_grid_data, key=lambda x: x.get("task_index", 0))
            # 一時的にバッファを置き換えてフラッシュ
            original_temporary_buffer = buffer_manager.temporary_input_grid_buffer.copy()
            buffer_manager.temporary_input_grid_buffer = batch_temporary_input_grid_data
            buffer_manager._flush_temporary_input_grid_png(batch_num)
            # フラッシュ済みのデータをバッファから削除（バッチ範囲内のすべてのデータを削除）
            buffer_manager.temporary_input_grid_buffer = [
                task_data for task_data in original_temporary_buffer
                if not (batch_start_task <= task_data.get("task_index", 0) <= batch_end_task)
            ]

        # バッチ処理完了時には常にバッチ進捗を保存（再開用）（100タスクの場合は最小限のみ）
        # 現在のバッチ番号を計算（バッチ処理ループで処理したバッチ番号を使用）
        current_batch = batch_num
        if ENABLE_ALL_LOGS:
            save_batch_progress(output_dir, statistics.total_programs, current_batch)
            print(f"  → バッチ {current_batch} 進捗を保存（完了タスク: {statistics.total_programs}）")
        else:
            save_batch_progress(output_dir, statistics.total_programs, current_batch)

        # バッチ処理完了時にTXTファイルを生成（バッチごとの効率的な処理）
        print(f"  → バッチ {batch_num + 1} のTXTファイルを生成中...", flush=True)
        try:
            # 現在のバッチディレクトリ内のJSONファイルに対応するTXTファイルを生成
            batch_dir = os.path.join(output_dir, f"batch_{batch_num:04d}")
            if os.path.exists(batch_dir):
                # このバッチのprogram JSONファイルを探す
                program_json_files = [f for f in os.listdir(batch_dir)
                                    if f.startswith("program_batch_") and f.endswith(".json")]
                program_json_files.sort()

                for json_file in program_json_files:
                    json_path = os.path.join(batch_dir, json_file)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            programs = json.load(f)

                        if not programs:
                            continue

                        # task_idでソート（task_001, task_002, ...の順）
                        sorted_programs = sorted(programs, key=lambda x: int(x.get('task_id', 'task_000').split('_')[-1]) if '_' in x.get('task_id', '') else 0)

                        # JSONファイル名からTXTファイル名を生成
                        txt_filename = json_file.replace('.json', '.txt')
                        txt_path = os.path.join(batch_dir, txt_filename)

                        # 既にTXTファイルが存在する場合はスキップ（重複防止）
                        if os.path.exists(txt_path):
                            if ENABLE_ALL_LOGS:
                                print(f"    [TXT] スキップ（既存）: {txt_path}")
                            continue

                        with open(txt_path, "w", encoding="utf-8") as f:
                            for program in sorted_programs:
                                f.write(f"# Task {program.get('task_id', 'unknown')}\n")
                                f.write(f"# Complexity: {program.get('complexity', 'N/A')}\n")
                                f.write(f"# Timestamp: {program.get('timestamp', 'N/A')}\n")
                                f.write("# ========================================\n")
                                f.write(program.get("program_code", ""))
                                f.write("\n\n")

                        print(f"    [TXT] バッチファイル生成: {txt_path} ({len(sorted_programs)}タスク)")
                    except Exception as e:
                        print(f"    [WARNING] TXTファイル生成エラー: {json_path} - {e}")
                        continue
        except Exception as e:
            print(f"  [WARNING] バッチ {batch_num + 1} のTXTファイル生成中にエラーが発生しました: {e}", flush=True)

        # 1000タスク（1バッチ）完了時に重み調整も保存（100タスクの場合はスキップ）
        if statistics.total_programs % BATCH_SIZE == 0 and statistics.total_programs >= BATCH_SIZE:
            if ENABLE_ALL_LOGS:
                print(f"  → バッチ {current_batch} 完了: 重み調整を保存")
            adjustments = statistics.get_weight_adjustments()
            if adjustments:
                save_weight_adjustments(output_dir, adjustments, statistics.total_programs)

        # バッチ処理完了時にリソースクリーンアップを実行（各バッチ終了時に実行）
        # プログラム実行以降のすべての処理やデータを完全にリセット（状態蓄積防止）
        try:
            # 1. ExecutorCoreの永続状態をクリア（モジュールを再インポートして強制的にリセット）
            # ExecutorCoreとExecutorのモジュールをリロードして状態をクリア
            if 'src.core_systems.executor.core' in sys.modules:
                # モジュールのグローバル変数をクリア
                core_module = sys.modules['src.core_systems.executor.core']
                # ExecutorCoreクラスのインスタンス変数をクリアできないため、モジュールレベルで対応

            # 2. ガベージコレクションを複数回実行して確実にクリア
            batch_cleanup_start = time.time()
            for _ in range(3):
                gc.collect()
            batch_cleanup_time = time.time() - batch_cleanup_start
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"  [計測] バッチ終了時ガベージコレクション: {batch_cleanup_time:.3f}秒", flush=True)

            # 3. スレッドプールやバックグラウンド処理のリソースもクリア・測定
            import threading

            threads_after = threading.active_count()
            thread_names_after = [t.name for t in threading.enumerate()]
            threadpool_threads_after = [t for t in threading.enumerate() if 'ThreadPoolExecutor' in str(t) or 'Thread-' in t.name]

            memory_after = 0
            memory_diff = 0
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
            except ImportError:
                # psutilがインストールされていない場合はスキップ
                pass
            except Exception:
                pass

            # リソース測定結果を強制出力（警告の有無に関わらず）
            print(f"  [リソース測定] バッチ {batch_num + 1} 完了後: スレッド数: {threads_after} (開始時: {threads_before}), ThreadPoolExecutor系: {len(threadpool_threads_after)} (開始時: {len(threadpool_threads)}), メモリ: {memory_after:.1f}MB (開始時: {memory_before:.1f}MB, 差分: {memory_diff:+.1f}MB)", flush=True)

            # リソース増加の警告
            if threads_after > threads_before:
                print(f"  [警告] バッチ {batch_num + 1} 完了後: スレッド数が増加しています ({threads_before} → {threads_after}, 増加: {threads_after - threads_before})", flush=True)
                if ENABLE_ALL_LOGS:
                    new_threads = [t.name for t in threading.enumerate() if t.name not in thread_names_before]
                    print(f"    新規スレッド: {new_threads}", flush=True)

            if len(threadpool_threads_after) > len(threadpool_threads):
                print(f"  [警告] バッチ {batch_num + 1} 完了後: ThreadPoolExecutorスレッドが増加しています ({len(threadpool_threads)} → {len(threadpool_threads_after)})", flush=True)

            if memory_diff > 50:  # 50MB以上の増加
                print(f"  [警告] バッチ {batch_num + 1} 完了後: メモリ使用量が大幅に増加しています ({memory_before:.1f}MB → {memory_after:.1f}MB, 増加: {memory_diff:.1f}MB)", flush=True)

            # すべてのスレッドが終了していることを確認（タイムアウト付き）
            # ThreadPoolExecutorのスレッドが確実に終了するまで短時間待機
            thread_wait_start = time.time()
            max_thread_wait = 0.2  # 最大200ms待機
            initial_alive = len([t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()])
            while time.time() - thread_wait_start < max_thread_wait:
                alive_threads = [t for t in threading.enumerate()
                               if t != threading.current_thread() and t.is_alive()]
                if not alive_threads:
                    break
                # ThreadPoolExecutorのスレッドのみをチェック
                threadpool_alive = [t for t in alive_threads
                                  if 'ThreadPoolExecutor' in str(t) or 'Thread-' in t.name]
                if not threadpool_alive:
                    break
                time.sleep(0.01)  # 10ms待機
            thread_wait_time = time.time() - thread_wait_start
            final_alive = len([t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()])
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"  [計測] ThreadPoolExecutorスレッド待機: {thread_wait_time:.3f}秒 (初期: {initial_alive}, 終了: {final_alive})", flush=True)

            for thread in threading.enumerate():
                if thread != threading.current_thread() and thread.is_alive():
                    # バックグラウンドスレッドが残っている場合は警告
                    if ENABLE_ALL_LOGS:
                        print(f"    [情報] 残存スレッド: {thread.name} (is_alive={thread.is_alive()})", flush=True)
                    # 強制終了はしない（タイムアウト処理に任せる）

            # 次のバッチとの比較用に現在の値を保存
            prev_threads_count = threads_after
            prev_memory_mb = memory_after
        except Exception as e:
            # リセット処理でエラーが発生しても処理を継続
            if ENABLE_ALL_LOGS:
                print(f"  [警告] バッチ {batch_num + 1} リソースクリーンアップ中にエラー: {e}", flush=True)
            # エラーが発生しても最低限のガベージコレクションは実行
            gc_start = time.time()
            gc.collect()
            gc_time = time.time() - gc_start
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"  [計測] エラー時ガベージコレクション: {gc_time:.3f}秒", flush=True)

        # バッチ終了時にバッファ内のデータも明示的にクリア（メモリ使用量削減）
        # フラッシュ済みのデータは既に削除されているが、念のため再度確認
        # buffer_managerのバッファは既にフラッシュ済みデータを削除している（1499-1503行目、1518-1521行目）

    print(f"\nすべてのバッチ処理完了: {len(execution_results)}個のプログラム（条件を満たしているもののみ）")

    # すべてのバッチ処理完了後にもリソースクリーンアップを実行
    gc.collect()

    # コマンド使用統計を出力（最終統計）（100タスクの場合は最小限のみ）
    if statistics.total_programs > 0:
        # 1000タスクの倍数でない場合は最終的な統計を出力（100タスクの場合はスキップ）
        if statistics.total_programs % BATCH_SIZE != 0 and ENABLE_ALL_LOGS:
            print(f"\n【最終統計と重み調整】")
            statistics.print_statistics()
            adjustments = statistics.get_weight_adjustments()
            if adjustments:
                generator.command_weight_adjustments = adjustments
                print(f"  調整対象: {len(adjustments)}個のコマンド")
                # 最終的な重み調整も保存
                save_weight_adjustments(output_dir, adjustments, statistics.total_programs)
                # バッチ進捗を保存
                current_batch = statistics.total_programs // BATCH_SIZE
                save_batch_progress(output_dir, statistics.total_programs, current_batch)
        elif statistics.total_programs >= BATCH_SIZE:
            # 1000タスクの倍数の場合は既に統計が出力されているので、調整のみ確認
            adjustments = statistics.get_weight_adjustments()
            if adjustments and not generator.command_weight_adjustments:
                # まだ調整が適用されていない場合は適用
                generator.command_weight_adjustments = adjustments
                if ENABLE_ALL_LOGS:
                    print(f"\n【最終重み調整を適用】")
                    print(f"  調整対象: {len(adjustments)}個のコマンド")
                # 重み調整を保存
                save_weight_adjustments(output_dir, adjustments, statistics.total_programs)
                # バッチ進捗を保存
                current_batch = statistics.total_programs // BATCH_SIZE
                save_batch_progress(output_dir, statistics.total_programs, current_batch)

    # すべてのバッファをフラッシュ（条件に関係なく実行）
    if ENABLE_ALL_LOGS:
        print(f"\nバッファをフラッシュ中...")
    flush_start = time.time()
    try:
        buffer_manager.flush_all()
        elapsed = time.time() - flush_start
        if timing_stats:
            timing_stats.record_step("バッファフラッシュ", elapsed, success=True)
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] バッファフラッシュ: {elapsed:.3f}秒", flush=True)
    except Exception as e:
        elapsed = time.time() - flush_start
        if timing_stats:
            timing_stats.record_step("バッファフラッシュ", elapsed, success=False)
        print(f"  [エラー] バッファのフラッシュ中にエラーが発生しました: {e}", flush=True)

    # JSONからサマリーTXTファイルを生成（常に実行）
    print(f"\nサマリーTXTファイルを生成中...", flush=True)
    summary_start = time.time()
    try:
        buffer_manager.generate_summary_txt_files()
        elapsed = time.time() - summary_start
        if timing_stats:
            timing_stats.record_step("サマリーTXTファイル生成", elapsed, success=True)
        if ENABLE_VERBOSE_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[TIME] サマリーTXTファイル生成: {elapsed:.3f}秒", flush=True)
        print(f"サマリーTXTファイルの生成が完了しました。", flush=True)
    except Exception as e:
        elapsed = time.time() - summary_start
        if timing_stats:
            timing_stats.record_step("サマリーTXTファイル生成", elapsed, success=False)
        print(f"  [エラー] サマリーTXTファイルの生成中にエラーが発生しました: {e}", flush=True)
        traceback.print_exc()

    # タイミング統計を出力
    timing_stats.print_statistics()

    # 実行時間計測終了
    end_time = time.time()
    end_datetime = datetime.now()
    total_seconds = end_time - start_time
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    # 計測できている時間を計算
    timing_stats_dict = timing_stats.get_statistics()
    measured_time_total = sum(stat['total'] for stat in timing_stats_dict.values())

    # 試行タスク数を計算（成功+失敗）
    attempted_tasks = statistics.total_programs
    failed_tasks = timing_stats.step_failures.get("タスク生成（失敗）", 0)
    attempted_tasks = attempted_tasks + failed_tasks  # 成功数 + 失敗数

    # 計測できていない時間を計算
    unmeasured_time = total_seconds - measured_time_total

    print(f"\n{'='*80}")
    print(f"【実行完了】")
    print(f"{'='*80}")
    print(f"開始時刻: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"終了時刻: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総実行時間: {total_seconds:.2f}秒 ({total_minutes:.2f}分 / {total_hours:.2f}時間)")
    print(f"試行タスク数: {attempted_tasks} (成功: {statistics.total_programs}, 失敗: {failed_tasks})")
    print(f"処理タスク数: {statistics.total_programs}")

    if attempted_tasks > 0:
        avg_time_per_attempted_task = total_seconds / attempted_tasks
        print(f"1タスクあたりの平均時間（試行ベース）: {avg_time_per_attempted_task:.3f}秒")

    if statistics.total_programs > 0:
        avg_time_per_task = total_seconds / statistics.total_programs
        print(f"1タスクあたりの平均時間（成功ベース）: {avg_time_per_task:.3f}秒")
        print(f"10万タスクの推定時間: {avg_time_per_task * 100000 / 3600:.1f}時間 ({avg_time_per_task * 100000 / 86400:.1f}日)")

    # 計測できている時間と計測できていない時間を表示
    print(f"\n【時間計測の内訳】")
    print(f"計測できている時間: {measured_time_total:.2f}秒 ({measured_time_total/total_seconds*100:.1f}%)")
    print(f"計測できていない時間: {unmeasured_time:.2f}秒 ({unmeasured_time/total_seconds*100:.1f}%)")
    if unmeasured_time > 0:
        print(f"  内訳（推定）:")
        print(f"    - バッファフラッシュ")
        print(f"    - メモリ解放・ガベージコレクション")
        print(f"    - 統計情報の出力")
        print(f"    - ファイルの保存")
        print(f"    - その他のオーバーヘッド")

    # 実行時間をファイルに保存
    time_log_path = os.path.join(output_dir, "execution_time_log.json")
    avg_time_per_task = total_seconds / statistics.total_programs if statistics.total_programs > 0 else 0
    avg_time_per_attempted_task = total_seconds / attempted_tasks if attempted_tasks > 0 else 0
    time_log = {
        'start_datetime': start_datetime.isoformat(),
        'end_datetime': end_datetime.isoformat(),
        'total_seconds': total_seconds,
        'total_minutes': total_minutes,
        'total_hours': total_hours,
        'attempted_tasks': attempted_tasks,
        'successful_tasks': statistics.total_programs,
        'failed_tasks': failed_tasks,
        'total_tasks': statistics.total_programs,
        'avg_time_per_task': avg_time_per_task,
        'avg_time_per_attempted_task': avg_time_per_attempted_task,
        'measured_time_total': measured_time_total,
        'unmeasured_time': unmeasured_time,
        'estimated_100k_hours': (avg_time_per_task * 100000 / 3600) if statistics.total_programs > 0 else 0,
        'estimated_100k_days': (avg_time_per_task * 100000 / 86400) if statistics.total_programs > 0 else 0,
        'task_count': TASK_COUNT,
        'batch_size': BATCH_SIZE,
    }
    with open(time_log_path, 'w', encoding='utf-8') as f:
        json.dump(time_log, f, indent=2, ensure_ascii=False)
    print(f"\n実行時間ログを保存: {time_log_path}")

    print(f"\nすべてのファイル処理が完了しました")


if __name__ == "__main__":
    # 直接実行時は相対インポートを絶対インポートに変更
    # モジュールとして実行される場合は既にインポート済み
    try:
        from .config import get_config
    except ImportError:
        # 直接実行時は絶対インポートを使用
        from src.data_systems.generator.config import get_config
        # グローバル設定を再取得（直接実行時）
        _config = get_config()
        # 設定から定数を取得
        WEIGHT_ADJUSTMENT_DIR = _config.weight_adjustment_dir
        WEIGHT_ADJUSTMENT_FILENAME = _config.weight_adjustment_filename
        BATCH_PROGRESS_FILENAME = _config.batch_progress_filename
        BATCH_SIZE = _config.batch_size
        TASK_COUNT = _config.task_count
        ENABLE_DEBUG_OUTPUT = _config.enable_debug_output
        ENABLE_VERBOSE_OUTPUT = _config.enable_verbose_output
        ENABLE_VERBOSE_LOGGING = _config.enable_verbose_logging
        ENABLE_ALL_LOGS = _config.enable_all_logs
        ENABLE_OUTPUT_CONDITION_CHECK = _config.enable_output_condition_check
        CONDITION4_KEEP_PROBABILITY = _config.condition4_keep_probability
        CONDITION6_KEEP_PROBABILITY = _config.condition6_keep_probability
        MIN_PAIRS_PER_PROGRAM = _config.min_pairs_per_program
        MAX_PAIRS_PER_PROGRAM = _config.max_pairs_per_program
        MAX_PAIR_RETRIES = _config.max_pair_retries
        MAX_GRID_REGENERATION_ATTEMPTS = _config.max_grid_regeneration_attempts
        MAX_CONSECUTIVE_CONTINUES = _config.max_consecutive_continues
        COMPLEXITY_RATIOS = _config.complexity_ratios
        MAX_EXECUTION_TIME = _config.max_execution_time

    parser = argparse.ArgumentParser(description="プログラム生成と実行")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "program-only", "input-grid-only"],
        help="実行モードを指定 (default: full)"
            "\n  - full: プログラム生成と実行"
            "\n  - program-only: プログラム生成のみを行い、実行はしない"
            "\n  - input-grid-only: インプットグリッドのみを生成してPNGとして保存"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（指定しない場合は新しいタイムスタンプ付きディレクトリを作成）。続きから実行する場合は既存のディレクトリを指定してください（例: outputs/20251105_050641）"
    )
    args = parser.parse_args()

    main(mode=args.mode, output_dir=args.output_dir)
