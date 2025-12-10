"""
中央実行エンジン

プログラムのNodeリストを受け取り、文字列化して実行する
"""
import os
import numpy as np
import random
import time
from typing import List, Optional, Tuple, Any, Dict, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# ログ出力制御（config.pyから読み込み）
from src.data_systems.generator.config import get_config

_config = get_config()

ENABLE_VERBOSE_LOGGING = _config.enable_verbose_logging
ENABLE_ALL_LOGS = _config.enable_all_logs
ENABLE_TRACE_FOR_DATASET = _config.enable_trace_for_dataset

from src.core_systems.executor.executor import Executor
from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
from src.data_systems.generator.program_generator.generation.code_generator import generate_code
from src.data_systems.generator.input_grid_generator.background_decider import decide_background_color, generate_background_grid_pattern
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.input_grid_generator.builders import generate_objects_from_conditions, modify_objects
from src.data_systems.generator.input_grid_generator.managers import build_grid, set_position
from .node_analyzer import is_assignment_node, is_if_node, is_for_node, is_end_node, get_commands_sorted_by_depth
from .node_analyzer.variable_analyzer import remove_unused_variables, remove_self_assignments
from .node_analyzer.execution_helpers import (
    complete_incomplete_control_structures,
    execute_and_get_variable_state,
    replace_command_with_fallback,
    compare_variable_info,
    _add_test_objects_tracking
)
# SilentExceptionをモジュールレベルでインポート（スコープ問題を回避）
from src.core_systems.executor.core import SilentException
from .performance_profiler import profile_code_block

# ========================================
# 定数
# ========================================
DEFAULT_COMPLEXITY = _config.default_complexity
DEFAULT_GRID_SIZE = _config.default_grid_size  # フォールバック用（decide_grid_sizeを使う場合は不要）

# ========================================
# ヘルパー関数
# ========================================
def get_modify_objects_kwargs(
    cmd_name: str,
    change_color: bool,
    change_shape: bool,
    change_position: bool,
    shape_change_count: int = 0
) -> Dict[str, Any]:
    """コマンドの種類に応じて、modify_objectsに渡す追加のkwargsを返す

    Args:
        cmd_name: 検証中のコマンド名
        change_color: 色を変更するか
        change_shape: 形を変更するか
        change_position: 位置を変更するか
        shape_change_count: change_shapeがTrueになった回数（ROTATEなどの場合に使用）

    Returns:
        modify_objectsに渡す追加のkwargs辞書
    """
    kwargs = {}

    # ROTATEコマンドの場合
    if cmd_name == 'ROTATE' or cmd_name == 'FLIP':
        if change_shape:
            # change_shapeがTrueになるたびに、symmetry_constraintを順番に適用
            symmetry_types = ['vertical', 'horizontal', 'both']
            symmetry_index = shape_change_count % len(symmetry_types)
            kwargs['symmetry_constraint'] = symmetry_types[symmetry_index]
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [{cmd_name}] change_shape={change_shape}, symmetry_constraint={kwargs['symmetry_constraint']} (shape_change_count={shape_change_count})")

    # SCALE_DOWNコマンドの場合: 2x2以上のオブジェクトが必要
    elif cmd_name == 'SCALE_DOWN':
        if change_shape:
            kwargs['min_bbox_size'] = 2  # 2x2以上のオブジェクト
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [SCALE_DOWN] change_shape={change_shape}, min_bbox_size=2")

    # FILL_HOLESコマンドの場合: 穴が1つ以上のオブジェクトが必要
    elif cmd_name == 'FILL_HOLES':
        if change_shape:
            kwargs['allow_holes'] = True
            kwargs['min_holes'] = 1  # 穴が1つ以上
            kwargs['max_holes'] = 5  # 最大穴数
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [FILL_HOLES] change_shape={change_shape}, allow_holes=True, min_holes=1")

    # HOLLOWコマンドの場合: 3x3以上のオブジェクトが必要
    elif cmd_name == 'HOLLOW':
        if change_shape:
            kwargs['min_bbox_size'] = 3  # 3x3以上のオブジェクト
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [HOLLOW] change_shape={change_shape}, min_bbox_size=3")

    # BBOXコマンドの場合: 密度が100%未満、2x2以上のオブジェクトが必要
    elif cmd_name == 'BBOX':
        if change_shape:
            kwargs['min_bbox_size'] = 2  # 2x2以上のオブジェクト
            kwargs['max_density'] = 0.99  # 密度が100%未満（99%以下）
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [BBOX] change_shape={change_shape}, min_bbox_size=2, max_density=0.99")

    # EXTRACT_RECTSコマンドの場合: 矩形生成
    elif cmd_name == 'EXTRACT_RECTS':
        if change_shape:
            kwargs['shape_type'] = 'rectangle'
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [EXTRACT_RECTS] change_shape={change_shape}, shape_type='rectangle'")

    # EXTRACT_LINESコマンドの場合: 線生成
    elif cmd_name == 'EXTRACT_LINES':
        if change_shape:
            kwargs['shape_type'] = 'line'
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [EXTRACT_LINES] change_shape={change_shape}, shape_type='line'")

    # EXTRACT_HOLLOW_RECTSコマンドの場合: 中空矩形生成
    elif cmd_name == 'EXTRACT_HOLLOW_RECTS':
        if change_shape:
            kwargs['shape_type'] = 'hollow_rectangle'
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"    [EXTRACT_HOLLOW_RECTS] change_shape={change_shape}, shape_type='hollow_rectangle'")

    # EXTEND_PATTERNコマンドの場合: 既存オブジェクトから複製
    # 注意: modify_objectsにはduplicate_modeがないため、shape_typeは設定しない（Noneのまま、形状制約なし）
    elif cmd_name == 'EXTEND_PATTERN':
        # 既存オブジェクトから形状を複製する場合、形状制約なしで通常の生成ロジックを使用
        # 実際の複製はgenerate_objects_from_conditionsの既存ロジック（SAME_COLOR_AND_SHAPE_PROBABILITY等）に依存
        # shape_typeは設定しない（Noneのまま）
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"    [EXTEND_PATTERN] change_shape={change_shape}, 形状制約なし（既存オブジェクトから複製）")

    # MATCH_PAIRSコマンドの場合: 既存オブジェクトから複製
    elif cmd_name == 'MATCH_PAIRS':
        # 既存オブジェクトから形状を複製する場合、形状制約なしで通常の生成ロジックを使用
        # shape_typeは設定しない（Noneのまま）
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"    [MATCH_PAIRS] change_shape={change_shape}, 形状制約なし（既存オブジェクトから複製）")

    return kwargs




# ========================================
# CoreExecutor クラス
# ========================================
class CoreExecutor:
    """Nodeリストからプログラムを実行する中央実行エンジン"""

    def __init__(self, preserve_indentation: bool = True, enable_execution_cache: bool = None):
        """初期化

        Args:
            preserve_indentation: インデントを保持するか
            enable_execution_cache: プログラム実行キャッシュを有効にするか（Noneの場合は環境変数から読み込み）
        """
        self.preserve_indentation = preserve_indentation
        self.executor = Executor()
        self.program_generator = UnifiedProgramGenerator()

        # プログラム実行キャッシュの初期化
        if enable_execution_cache is None:
            enable_execution_cache = os.environ.get('ENABLE_PROGRAM_EXECUTION_CACHE', 'true').lower() in ('true', '1', 'yes')

        self.enable_execution_cache = enable_execution_cache
        self.execution_cache = None
        if self.enable_execution_cache:
            from src.hybrid_system.utils.performance.cache_manager import ProgramExecutionCache
            # キャッシュサイズとTTLは環境変数から設定可能（デフォルト: 1000エントリ、1時間）
            cache_max_size = int(os.environ.get('PROGRAM_EXECUTION_CACHE_MAX_SIZE', '1000'))
            cache_ttl = float(os.environ.get('PROGRAM_EXECUTION_CACHE_TTL', '3600.0'))
            self.execution_cache = ProgramExecutionCache(max_size=cache_max_size, ttl=cache_ttl)

    def _nodes_to_string(self, nodes: List[Any], complexity: int = DEFAULT_COMPLEXITY,
                         grid_width: Optional[int] = None, grid_height: Optional[int] = None) -> str:
        """Nodeリストをプログラム文字列に変換"""
        try:
            context = ProgramContext(complexity=complexity, grid_width=grid_width, grid_height=grid_height)
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

        # generate_code関数を直接使用
        try:
            result = generate_code(nodes, context)
            return result
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

    def execute_program_string(
        self,
        program_code: str,
        input_grid: np.ndarray,
        input_objects: Optional[List] = None,
        input_image_index: int = 0,
        background_color: Optional[int] = None
    ) -> Tuple[np.ndarray, List, float, Dict[str, Any]]:
        """プログラム文字列を直接実行（キャッシュ対応）"""
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[execute_program_string] 関数開始", flush=True)
        input_grid = np.asarray(input_grid, dtype=int)
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[execute_program_string] input_grid変換完了 (shape={input_grid.shape})", flush=True)

        # キャッシュから取得を試みる（input_objectsがNoneの場合のみ、キャッシュのキーが一致するため）
        cached_output_grid = None
        if self.enable_execution_cache and self.execution_cache is not None and input_objects is None:
            try:
                cached_output_grid = self.execution_cache.get_execution_result(program_code, input_grid)
                if cached_output_grid is not None:
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"[execute_program_string] キャッシュヒット: プログラム実行をスキップ", flush=True)
                    # キャッシュから取得した場合は、execution_timeを0.0として返す（実際の実行時間ではないため）
                    # final_objectsはNoneとして返す（キャッシュには保存していないため）
                    metadata = {
                        'program_code': program_code,
                        'execution_time': 0.0,
                        'from_string': True,
                        'from_cache': True
                    }
                    return cached_output_grid, [], 0.0, metadata
            except Exception as e:
                # キャッシュ取得エラーは無視して続行（通常の実行にフォールバック）
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"[execute_program_string] キャッシュ取得エラー（無視して続行）: {e}", flush=True)

        try:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[execute_program_string] self.executor.execute_program呼び出し前", flush=True)
            # background_colorがNoneの場合は、ExecutorCoreで自動推論される
            output_grid, final_objects, execution_time = self.executor.execute_program(
                program_code=program_code,
                input_grid=input_grid,
                input_objects=input_objects,
                input_image_index=input_image_index,
                background_color=background_color
            )
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[execute_program_string] self.executor.execute_program呼び出し完了 (execution_time={execution_time:.3f}秒)", flush=True)

            # キャッシュに保存（成功した場合のみ、input_objectsがNoneの場合のみ）
            if self.enable_execution_cache and self.execution_cache is not None and input_objects is None and output_grid is not None:
                try:
                    self.execution_cache.set_execution_result(program_code, input_grid, output_grid)
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"[execute_program_string] 実行結果をキャッシュに保存", flush=True)
                except Exception as e:
                    # キャッシュ保存エラーは無視して続行
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"[execute_program_string] キャッシュ保存エラー（無視して続行）: {e}", flush=True)
        except Exception as e:
            # エラーログは常に出力（重要な情報）
            print(f"[execute_program_string] エラー発生: {type(e).__name__}: {e}", flush=True)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                import traceback
                traceback.print_exc()
            raise

        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[execute_program_string] metadata作成開始", flush=True)
        metadata = {
            'program_code': program_code,
            'execution_time': execution_time,
            'from_string': True,
            'from_cache': False
        }
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[execute_program_string] 関数終了", flush=True)

        return output_grid, final_objects, execution_time, metadata

    def reset_executor(self):
        """Executorをリセット"""
        if hasattr(self.executor, 'reset_context'):
            self.executor.reset_context()


# ========================================
# メイン関数
# ========================================
# 実行時間の上限（秒） - 無限ループ検出用（main.pyと同期）
# 環境変数から取得可能（デフォルトは3.0秒）
MAX_EXECUTION_TIME = _config.max_execution_time  # 3秒を超える場合は無限ループの可能性

def main(nodes: Optional[List[Any]] = None, grid_width: Optional[int] = None, grid_height: Optional[int] = None,
         task_index: Optional[int] = None, enable_replacement: bool = True,
         all_commands: Optional[Set[str]] = None, program_code: Optional[str] = None,
         is_first_pair: bool = True, input_grid: Optional[np.ndarray] = None,
         background_color: Optional[int] = None) -> Tuple[Optional[List[Any]], np.ndarray, np.ndarray, Optional[List[Any]]]:
    """メイン関数 - サンプル実行

    Args:
        nodes: プログラムのNodeリスト（Noneの場合はプログラムなしで入力グリッドのみ生成）
        grid_width: グリッド幅（Noneの場合は自動決定）
        grid_height: グリッド高さ（Noneの場合は自動決定）
        task_index: タスクインデックス（進捗表示用、Noneの場合は表示しない）
        enable_replacement: Trueの場合、コマンド置き換え判定を実行（ステップ5）
        all_commands: 抽出済みのコマンドセット（提供されている場合はステップ1をスキップ、最初以外のペア用）
        program_code: 生成済みのプログラムコード（提供されている場合はステップ2をスキップ、最初以外のペア用）
        is_first_pair: Trueの場合、最初のペア（タスク廃棄）、Falseの場合、最初以外のペア（ペアスキップ）
        input_grid: 既存のインプットグリッド（提供されている場合、インプットグリッド生成をスキップ）
        background_color: 背景色（指定された場合はそれを使用、Noneの場合は自動決定）

    Returns:
        (nodes, input_grid, output_grid, trace_results)のタプル

        trace_results:
            - ENABLE_TRACE_FOR_DATASET=True の場合:
                ExecutorCoreのexecution_context['results']をベースにしたトレース（リスト）
            - それ以外の場合:
                None
    """
    main_start_time = time.time()  # 全体の実行時間を計測

    # input_grid が提供されている場合、インプットグリッド生成をスキップ
    if input_grid is not None:
        # インプットグリッド生成をスキップ
        # validate_nodes_and_adjust_objects は実行しない（インプットグリッドが既に決定されているため）

        # プログラムがない場合は入力グリッドのみを返す
        if nodes is None:
            output_grid = input_grid.copy()
            trace_results = None
            return None, input_grid, output_grid, trace_results

        # プログラムコード生成
        executor = CoreExecutor(preserve_indentation=True)
        if program_code is None:
            try:
                program_code = executor._nodes_to_string(nodes, complexity=DEFAULT_COMPLEXITY, grid_width=grid_width, grid_height=grid_height)
            except Exception as e:
                if ENABLE_VERBOSE_LOGGING:
                    import traceback
                    traceback.print_exc()
                raise

        if not program_code.strip():
            if ENABLE_ALL_LOGS:
                print("❌ 空のプログラムが生成されました")
            # エラー時は入力グリッドと同じサイズの空のグリッドを返す
            if grid_width is None or grid_height is None:
                grid_width, grid_height = input_grid.shape[1], input_grid.shape[0]
            empty_grid = np.zeros((grid_height, grid_width), dtype=int)
            return nodes, input_grid, empty_grid, None

        # プログラム実行
        try:
            output_grid, final_objects, execution_time, execution_info = executor.execute_program_string(
                program_code=program_code,
                input_grid=input_grid,
                input_objects=None,
                input_image_index=0,
                background_color=None
            )
        except SilentException:
            # SilentExceptionの場合はそのまま再発生（エラーハンドリングで処理される）
            raise
        except Exception as e:
            raise ValueError(f"プログラム実行エラー: {str(e)}")

        # トレース結果の取得（オプション）
        trace_results: Optional[List[Any]] = None
        if ENABLE_TRACE_FOR_DATASET:
            try:
                trace_results = executor.executor.core.execution_context.get('results', [])
            except Exception:
                trace_results = None

        return nodes, input_grid, output_grid, trace_results

    # 通常の処理（インプットグリッド生成）

    # グローバルなgrid_size_contextをリセット（状態汚染を防止）
    try:
        from src.core_systems.executor.grid import grid_size_context
        # GridSizeContextはinitializeでリセット（状態汚染を防止）
        if grid_width is not None and grid_height is not None:
            grid_size_context.initialize((grid_width, grid_height))
        else:
            # グリッドサイズが不明な場合はNoneでリセット
            grid_size_context.input_grid_size = None
            grid_size_context._initialized = False
    except Exception as e:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        pass  # リセットに失敗しても無視（処理速度優先）
    executor = CoreExecutor(preserve_indentation=True)
    # バッチごとの完全リセットのため、ExecutorCoreの永続状態をクリア
    if hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'clear_persistent_data'):
        executor.executor.core.clear_persistent_data()
    # execution_contextも明示的にクリア
    if hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'execution_context'):
        executor.executor.core.execution_context = {
            'objects': {},
            'variables': {},
            'arrays': {},
            'results': [],
            'input_image_index': 0,
            'background_color': 0,
            'object_type_filter': None,
            'object_type_set': False
        }
    # grid_contextもリセット（状態汚染を防止）
    if hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'grid_context'):
        try:
            grid_ctx = executor.executor.core.grid_context
            if grid_width is not None and grid_height is not None:
                grid_ctx.initialize((grid_width, grid_height))
            else:
                grid_ctx.input_grid_size = None
                grid_ctx._initialized = False
        except Exception:
            pass  # リセットに失敗しても無視

    # プログラムコード生成（nodesがNoneの場合はスキップ）
    program_code = None
    if nodes is not None:
        try:
            program_code = executor._nodes_to_string(nodes, complexity=DEFAULT_COMPLEXITY, grid_width=grid_width, grid_height=grid_height)
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

        if not program_code.strip():
            if ENABLE_ALL_LOGS:
                print("❌ 空のプログラムが生成されました")
            # エラー時は空のグリッドを返す
            if grid_width is None or grid_height is None:
                grid_width, grid_height = decide_grid_size()
            empty_grid = np.zeros((grid_height, grid_width), dtype=int)
            return nodes, empty_grid, empty_grid

    # 背景色を決定（指定されていない場合のみ自動決定）
    if background_color is None:
        try:
            background_color = decide_background_color(nodes if nodes is not None else [], default=0)
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

    # グリッドサイズを決定（引数で渡されていない場合は統計に基づく）
    if grid_width is None or grid_height is None:
        try:
            grid_width, grid_height = decide_grid_size()
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

    # 背景グリッドパターンを生成
    try:
        background_grid_pattern = generate_background_grid_pattern(
            background_color=background_color,
            grid_width=grid_width,
            grid_height=grid_height
        )
    except Exception as e:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        raise
    all_objects = []
    # 全てのノードの検証が完了したら、実際のオブジェクト生成とグリッド構築を行う
    # ARC-AGI2統計に基づいてオブジェクト数を決定（改善版）
    from .node_validator_output import decide_num_objects_by_arc_statistics
    from src.data_systems.generator.input_grid_generator.builders.color_distribution import decide_object_color_count, select_object_colors
    num_objects = decide_num_objects_by_arc_statistics(
        grid_width=grid_width,
        grid_height=grid_height,
        all_commands=set()  # この時点ではコマンド情報がないため空セット
    )

    # プログラムなしの場合も、プログラムありと同じようにグリッドサイズ情報を活用
    # decide_num_objects_by_arc_statisticsが既にグリッドサイズに基づいて調整しているため、
    # 追加の調整は不要（プログラムありの場合と同じロジックを使用）
    # 以前の「カテゴリ分類のため」の追加調整は削除し、統一されたロジックを使用
    if nodes is None:
        # グリッドサイズに基づく調整は decide_num_objects_by_arc_statistics 内で既に実行済み
        # プログラムありの場合と同じロジックを使用するため、追加の調整は行わない
        pass

    # 色の決定（ARC-AGI2統計に基づく）
    target_color_count = decide_object_color_count(
        existing_colors=set(),
        background_color=background_color
    )

    object_colors = select_object_colors(
        background_color=background_color,
        target_color_count=target_color_count,
        existing_colors=set()
    )
    # オブジェクト生成（別処理）
    try:
        all_objects = generate_objects_from_conditions(
            background_color=background_color,
            object_color=None,  # Noneにすることで統計に基づいて色を決定
            num_objects=num_objects,
            existing_objects=all_objects,  # 既存のオブジェクトが空なので、統計から色を決定
            grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,  # グリッドサイズを渡してオブジェクトサイズを調整
            total_num_objects=num_objects,  # 面積ベースの制約を適用するため、total_num_objectsを指定
            object_colors=object_colors  # 通常の生成と同じ方法で色を指定
        )
    except Exception as e:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        raise
    # all_objectsからグリッドを構築
    try:
        all_objects = set_position(
            width=grid_width,
            height=grid_height,
            background_color=background_color,
            objects=all_objects,
            existing_objects=None
        )
    except Exception as e:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        raise

    # IF/FOR/ENDのカウント用変数
    if_count = 0
    for_count = 0
    end_count = 0

    # ノード検証とオブジェクト調整
    # プログラムがない場合でも、validate_nodes_and_adjust_objects()を呼び出して
    # 入力グリッドの条件チェックと再試行を有効化
    # validate_nodes_and_adjust_objectsを実行（最初のペアのみ）
    # デフォルトは'output_v3'（旧バージョンは削除済み）
    validator_type = os.environ.get('VALIDATOR_TYPE', 'output_v3').lower()
    if validator_type != 'output_v3':
        print(f"[警告] VALIDATOR_TYPE='{validator_type}'は削除されました。'output_v3'を使用します。")
        validator_type = 'output_v3'

    # output_v3バリデータを使用
    try:
        from .node_validator_output import validate_nodes_and_adjust_objects
    except Exception as e:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        raise

    try:
        # 実行時間を監視して無限ループを検出
        validate_start_time = time.time()
        elapsed_before_validate = validate_start_time - main_start_time

        # 既に実行時間が上限を超えている場合は早期終了
        if elapsed_before_validate > MAX_EXECUTION_TIME:
            error_msg = f"タスク{task_index}: validate_nodes_and_adjust_objects呼び出し前に実行時間が上限を超えました（{elapsed_before_validate:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）"
            print(f"  [警告] {error_msg}", flush=True)
            # SilentExceptionはモジュールレベルでインポート済み
            raise SilentException(error_msg)

        with profile_code_block("validate_nodes_output_v3", "core_executor"):
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None:
                print(f"  [タイミング] タスク{task_index}: validate_nodes_and_adjust_objects呼び出し開始", flush=True)
            validate_call_start = time.time()

            # タイムアウトチェック: 残り時間を計算
            elapsed_so_far = time.time() - main_start_time
            remaining_time = max(0.1, MAX_EXECUTION_TIME - elapsed_so_far)  # 最低0.1秒は確保

            # ThreadPoolExecutorを使用してタイムアウト機構を実装
            executor_pool = None
            future = None
            try:
                executor_pool = ThreadPoolExecutor(max_workers=1)
                future = executor_pool.submit(
                    validate_nodes_and_adjust_objects,
                    nodes=nodes,
                    all_objects=all_objects,
                    background_color=background_color,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    background_grid_pattern=background_grid_pattern,
                    executor=executor,
                    if_count=if_count,
                    for_count=for_count,
                    end_count=end_count,
                    enable_replacement=enable_replacement,
                    all_commands=all_commands,
                    program_code=program_code,
                    is_first_pair=is_first_pair
                )
                result = future.result(timeout=remaining_time)
            except FuturesTimeoutError:
                error_msg = f"タスク{task_index}: validate_nodes_and_adjust_objectsがタイムアウトしました（{remaining_time:.2f}秒以内に完了しませんでした）"
                print(f"  [警告] {error_msg}", flush=True)
                # SilentExceptionはモジュールレベルでインポート済み
                raise SilentException(error_msg)
            except KeyboardInterrupt:
                # KeyboardInterruptを確実に捕捉: スレッドをキャンセル試行
                print(f"\n  [中断] タスク{task_index}: validate_nodes_and_adjust_objectsがユーザーによって中断されました", flush=True)
                if future is not None:
                    future.cancel()
                raise
            finally:
                # リソースをクリーンアップ
                if executor_pool is not None:
                    executor_pool.shutdown(wait=True)

            validate_call_elapsed = time.time() - validate_call_start
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None:
                print(f"  [タイミング] タスク{task_index}: validate_nodes_and_adjust_objects完了 (処理時間: {validate_call_elapsed:.3f}秒)", flush=True)

            # 実行時間をチェック
            validate_elapsed = time.time() - validate_start_time
            total_elapsed = time.time() - main_start_time

            # validate_nodes_and_adjust_objectsの実行時間が上限を超えた場合
            if validate_elapsed > MAX_EXECUTION_TIME * 0.8:  # 80%を超えた場合は警告
                print(f"  [警告] タスク{task_index}: validate_nodes_and_adjust_objectsの実行時間が長いです（{validate_elapsed:.2f}秒）", flush=True)

            # 全体の実行時間が上限を超えた場合
            if total_elapsed > MAX_EXECUTION_TIME:
                error_msg = f"タスク{task_index}: validate_nodes_and_adjust_objects完了後に実行時間が上限を超えました（{total_elapsed:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）"
                print(f"  [警告] {error_msg}", flush=True)
                # SilentExceptionはモジュールレベルでインポート済み
                raise SilentException(error_msg)

            # タイムアウトやエラーでタスクが破棄された場合（Noneが返された場合）
            # 注意: nodes=Noneは正常な動作（プログラムなしの場合）なので、result[0] is Noneのチェックは不要
            if result is None:
                if is_first_pair:
                    raise ValueError("プログラム実行エラーによりタスクが破棄されました（validate_nodes_and_adjust_objectsの結果がNone）")
                else:
                    # 最初以外のペアでNoneが返された場合は、ペアスキップの例外を発生
                    raise ValueError("ペアスキップ: validate_nodes_and_adjust_objectsの結果がNone")
            nodes, all_objects, if_count, for_count, end_count = result

            # プログラムがない場合（nodes is None かつ program_code is None）は、入力グリッドを構築して返す
            # program_codeが提供されている場合は、プログラムありとして処理を続行
            if nodes is None and program_code is None:
                # all_objectsが空の場合、単色グリッドになるため失敗として扱う
                if not all_objects or len(all_objects) == 0:
                    raise ValueError(f"プログラムなしモード: all_objectsが空です。単色グリッドを生成できません。")

                # 入力グリッドを構築
                grid_list = build_grid(
                    width=grid_width,
                    height=grid_height,
                    background_grid_pattern=background_grid_pattern,
                    objects=all_objects
                )
                input_grid = np.array(grid_list, dtype=int)

                # 単色グリッドのチェック（オブジェクトが配置されているにもかかわらず単色になっている場合）
                unique_colors = np.unique(input_grid).tolist()
                if len(unique_colors) == 1:
                    raise ValueError(f"プログラムなしモード: 入力グリッドが単色です（色={unique_colors[0]}）。オブジェクトが配置されていません。")

                # 出力グリッドは入力グリッドと同じ（プログラムなしのため）
                output_grid = input_grid.copy()
                trace_results = None
                return None, input_grid, output_grid, trace_results

            # program_codeが提供されているがnodes is Noneの場合、入力グリッドを構築してからプログラムを実行
            if nodes is None and program_code is not None:
                # all_objectsが空の場合、単色グリッドになるため失敗として扱う
                if not all_objects or len(all_objects) == 0:
                    raise ValueError(f"プログラムありモード（program_code提供）: all_objectsが空です。単色グリッドを生成できません。")

                # 入力グリッドを構築
                grid_list = build_grid(
                    width=grid_width,
                    height=grid_height,
                    background_grid_pattern=background_grid_pattern,
                    objects=all_objects
                )
                input_grid = np.array(grid_list, dtype=int)

                # 単色グリッドのチェック
                unique_colors = np.unique(input_grid).tolist()
                if len(unique_colors) == 1:
                    raise ValueError(f"プログラムありモード（program_code提供）: 入力グリッドが単色です（色={unique_colors[0]}）。オブジェクトが配置されていません。")

                # program_codeを実行して出力グリッドを取得
                try:
                    output_grid, _, execution_time, _ = executor.execute_program_string(
                        program_code=program_code,
                        input_grid=input_grid,
                        input_objects=None,
                        input_image_index=0,
                        background_color=None
                    )
                    if output_grid is None:
                        raise ValueError(f"プログラム実行エラー: output_gridがNone")
                    trace_results = None
                    return None, input_grid, output_grid, trace_results
                except Exception as e:
                    raise ValueError(f"プログラム実行エラー: {e}")
    except ValueError as e:
        # ペアスキップの例外を再発生（最初以外のペアで発生した場合）
        if not is_first_pair and str(e).startswith("ペアスキップ:"):
            raise
        # 最初のペアで発生した場合は、タスク廃棄の例外として再発生
        raise
    except (TimeoutError, Exception) as e:
        # 例外の詳細をログに記録（UnboundLocalErrorなどの問題を特定するため）
        exception_type = type(e).__name__
        exception_message = str(e)
        print(f"[core_executor] 例外発生: 例外型: {exception_type}, メッセージ: {exception_message}", flush=True)
        # スタックトレースを出力してエラーの発生箇所を特定
        import traceback
        print(f"[core_executor] スタックトレース:", flush=True)
        traceback.print_exc()

        # SilentException（タイムアウト、オブジェクト数上限など）も捕捉（モジュールレベルでインポート済み）
        # UnboundLocalErrorを防ぐため、グローバルスコープから明示的に参照
        try:
            # SilentExceptionが利用可能か確認
            from src.core_systems.executor.core import SilentException as SE
            is_silent_exception = isinstance(e, SE)
        except (ImportError, NameError) as import_error:
            # SilentExceptionが利用できない場合は、型名で判定
            print(f"[core_executor] SilentExceptionインポートエラー: {type(import_error).__name__}, {import_error}", flush=True)
            is_silent_exception = type(e).__name__ == 'SilentException'
        except Exception as check_error:
            # isinstanceチェック中にエラーが発生した場合
            print(f"[core_executor] isinstanceチェックエラー: {type(check_error).__name__}, {check_error}", flush=True)
            is_silent_exception = type(e).__name__ == 'SilentException'

        is_timeout = isinstance(e, TimeoutError) or (is_silent_exception and "タイムアウト" in str(e))
        is_object_limit = is_silent_exception and "オブジェクト数が上限を超えました" in str(e)
        is_pixel_limit = is_silent_exception and "ピクセル数が上限を超えました" in str(e)
        is_exclude_limit = is_silent_exception and "EXCLUDE: オブジェクト数が上限を超えています" in str(e)

        # ピクセル数上限の場合は、即座にタスク破棄（プログラムに問題があるため）
        if is_pixel_limit:
            raise ValueError("プログラム実行エラーによりタスクが破棄されました（ピクセル数上限）")
        # オブジェクト数上限の場合は、即座にタスク破棄（プログラムに問題があるため）
        # （変数への格納、FORループ、WHILEループ、プログラム実行中のオブジェクト数上限を含む）
        if is_object_limit:
            raise ValueError("プログラム実行エラーによりタスクが破棄されました（オブジェクト数上限）")
        # EXCLUDE操作のオブジェクト数上限の場合は、即座にタスク破棄
        if is_exclude_limit:
            raise ValueError("プログラム実行エラーによりタスクが破棄されました（EXCLUDE操作のオブジェクト数上限）")
        # タイムアウトの場合は、タイムアウトとしてマーク
        if is_timeout:
            raise ValueError("プログラム実行エラーによりタスクが破棄されました（validate_nodes_and_adjust_objectsのタイムアウト）")
        # その他のエラーが発生した場合は、例外としてマーク
        raise ValueError(f"プログラム実行エラーによりタスクが破棄されました（validate_nodes_and_adjust_objectsの例外: {type(e).__name__}）")
    except Exception as profile_error:
        if ENABLE_VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        raise

    # 自己代入の削除
    nodes = remove_self_assignments(nodes)

    # 未使用変数の削除
    nodes = remove_unused_variables(nodes)

    # ARC-AGI2統計に基づいてオブジェクト数を調整（改善版）
    # 現在のall_objectsの数から、統計的な理想的な数との差を計算して追加
    current_object_count = len(all_objects)

    # ARC-AGI2統計: 2-30個が多く、特に10-30個のタスクを増やす必要がある
    from .node_validator_output import decide_num_objects_by_arc_statistics
    ideal_count = decide_num_objects_by_arc_statistics(
        grid_width=grid_width,
        grid_height=grid_height,
        all_commands=set()  # この時点ではコマンド情報がないため空セット
    )

    if current_object_count == 0:
        # オブジェクト数が0の場合は理想的な数を設定
        target_count = ideal_count
    elif current_object_count < ideal_count:
        # 現在の数が理想的な数より少ない場合は、理想的な数まで追加
        # ただし、一度に追加しすぎないように段階的に追加
        additional_count = min(ideal_count - current_object_count, max(1, ideal_count // 3))
        target_count = current_object_count + additional_count
    elif current_object_count <= ideal_count * 1.2:
        # 現在の数が理想的な数の1.2倍以内の場合は、確率的に追加（20%の確率）
        if random.random() < 0.2:
            additional_count = random.randint(1, max(1, ideal_count // 5))
            target_count = min(ideal_count * 2, current_object_count + additional_count)
        else:
            target_count = current_object_count
    else:
        # 現在の数が理想的な数を大幅に超えている場合は追加しない
        target_count = current_object_count

    if target_count > current_object_count:
        additional_objects = target_count - current_object_count

        # 既存のオブジェクトから使用されている色を取得
        existing_colors = set()
        if all_objects:
            for obj in all_objects:
                if isinstance(obj, dict) and 'color' in obj:
                    existing_colors.add(obj['color'])
        # 色の決定（ARC-AGI2統計に基づく、既存の色を考慮）
        from src.data_systems.generator.input_grid_generator.builders.color_distribution import decide_object_color_count, select_object_colors
        target_color_count = decide_object_color_count(
            existing_colors=existing_colors,
            background_color=background_color
        )
        object_colors = select_object_colors(
            background_color=background_color,
            target_color_count=target_color_count,
            existing_colors=existing_colors
        )

        new_objects = generate_objects_from_conditions(
            background_color=background_color,
            object_color=None,
            num_objects=additional_objects,
            existing_objects=all_objects,
            grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,  # グリッドサイズを渡してオブジェクトサイズを調整
            total_num_objects=additional_objects,  # 面積ベースの制約を適用するため、total_num_objectsを指定（追加オブジェクトのみなので、additional_objectsを使用）
            object_colors=object_colors  # 通常の生成と同じ方法で色を指定
        )

        new_objects = set_position(
            width=grid_width,
            height=grid_height,
            background_color=background_color,
            objects=new_objects,
            existing_objects=all_objects
        )
        all_objects.extend(new_objects)

    # all_objectsからグリッドを構築
    grid_list = build_grid(
        width=grid_width,
        height=grid_height,
        background_grid_pattern=background_grid_pattern,
        objects=all_objects
    )
    input_grid = np.array(grid_list, dtype=int)

    # デバッグ: input_gridの検証（最初の10タスクのみ）
    # デフォルトで無効化（詳細ログ）
    if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None and task_index <= 10:
        print(f"  [デバッグ] タスク{task_index}: input_grid生成後の検証")
        print(f"    all_objects数: {len(all_objects)}")
        print(f"    input_grid.shape: {input_grid.shape}")
        input_unique = sorted(np.unique(input_grid).tolist())
        print(f"    input_grid unique値: {input_unique}")
        print(f"    input_gridが単色か: {len(input_unique) == 1}")
        if len(input_unique) == 1:
            print(f"    [警告] タスク{task_index}: input_gridが単色です！（色={input_unique[0]}）")
        # オブジェクトの情報を出力
        if all_objects:
            print(f"    all_objectsの詳細:")
            for i, obj in enumerate(all_objects[:5]):  # 最初の5個のみ
                if hasattr(obj, 'pixels') and obj.pixels:
                    print(f"      オブジェクト{i+1}: color={getattr(obj, 'color', 'N/A')}, pixels数={len(obj.pixels)}")
                elif isinstance(obj, dict):
                    print(f"      オブジェクト{i+1}: color={obj.get('color', 'N/A')}, pixels数={len(obj.get('pixels', []))}")
        else:
            print(f"    [警告] タスク{task_index}: all_objectsが空です！")

    program_code = executor._nodes_to_string(nodes, complexity=DEFAULT_COMPLEXITY, grid_width=grid_width, grid_height=grid_height)

    # プログラム実行
    # background_colorはNoneを渡すことで、本番用の背景色推論（_estimate_background_color）を使用
    # ExecutorCore.execute_program内で、input_objects=Noneの場合に自動的に背景色を推論する
    try:
        # デバッグ: 最初の10タスクのみプログラムコードと実行前の状態を出力
        # デフォルトで無効化（詳細ログ）
        if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None and task_index <= 10:
            print(f"  [デバッグ] タスク{task_index}: プログラム実行前")
            print(f"    プログラムコード (長さ={len(program_code)}文字):")
            print(f"      {program_code[:300]}..." if len(program_code) > 300 else f"      {program_code}")
            print(f"    input_grid.shape={input_grid.shape}, input_grid unique={sorted(np.unique(input_grid).tolist())}")
            # オブジェクト抽出のテスト（GET_ALL_OBJECTS(8)相当の動作を確認）
            try:
                from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
                from src.data_systems.config.config import ExtractionConfig
                from src.data_systems.data_models.base import ObjectType
                extractor = IntegratedObjectExtractor(ExtractionConfig())
                extraction_result = extractor.extract_objects_by_type(input_grid, input_image_index=0)

                if extraction_result.success:
                    # 8連結オブジェクトを取得（GET_ALL_OBJECTS(8)相当）
                    extracted_objects_8way = extraction_result.objects_by_type.get(ObjectType.SINGLE_COLOR_8WAY, [])
                    extracted_objects_4way = extraction_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
                    # 8連結を優先、なければ4連結を使用
                    extracted_objects = extracted_objects_8way if extracted_objects_8way else extracted_objects_4way

                    print(f"    オブジェクト抽出結果: {len(extracted_objects)}個のオブジェクト (8連結: {len(extracted_objects_8way)}, 4連結: {len(extracted_objects_4way)})")
                    if len(extracted_objects) > 0:
                        for i, obj in enumerate(extracted_objects[:3]):  # 最初の3個のみ
                            print(f"      抽出オブジェクト{i+1}: color={obj.dominant_color}, pixels数={obj.pixel_count}")
                    else:
                        print(f"    [警告] オブジェクト抽出結果が空です！")
                else:
                    print(f"    [警告] オブジェクト抽出に失敗しました")
            except Exception as e:
                print(f"    [警告] オブジェクト抽出テストでエラー: {e}")

        # デバッグ用に環境変数を設定（最初の10タスクのみ）
        # デフォルトで無効化（詳細ログ）
        if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None and task_index <= 10:
            os.environ['DEBUG_TASK_INDEX'] = str(task_index)
            print(f"  [デバッグ] 環境変数設定: DEBUG_TASK_INDEX={task_index}", flush=True)

        # 実行時間をチェック
        elapsed_before_exec = time.time() - main_start_time
        if elapsed_before_exec > MAX_EXECUTION_TIME:
            error_msg = f"タスク{task_index}: execute_program_string呼び出し前に実行時間が上限を超えました（{elapsed_before_exec:.2f}秒 > {MAX_EXECUTION_TIME:.2f}秒）"
            print(f"  [警告] {error_msg}", flush=True)
            # SilentExceptionはモジュールレベルでインポート済み
            raise SilentException(error_msg)

        exec_program_start = time.time()

        # タイムアウトチェック: 残り時間を計算
        elapsed_so_far = time.time() - main_start_time
        remaining_time = max(0.1, MAX_EXECUTION_TIME - elapsed_so_far)  # 最低0.1秒は確保

        # ThreadPoolExecutorを使用してタイムアウト機構を実装
        executor_pool = None
        future = None
        try:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [タイムアウト] execute_program_string開始: 残り時間={remaining_time:.2f}秒", flush=True)
            executor_pool = ThreadPoolExecutor(max_workers=1)
            future = executor_pool.submit(
                executor.execute_program_string,
                program_code=program_code,
                input_grid=input_grid,
                input_image_index=0,
                background_color=None  # Noneにすることで本番用の背景色推論を使用
            )
            output_grid, final_objects, execution_time, metadata = future.result(timeout=remaining_time)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [タイムアウト] execute_program_string完了 (残り時間={remaining_time:.2f}秒)", flush=True)
        except FuturesTimeoutError:
            error_msg = f"タスク{task_index}: execute_program_stringがタイムアウトしました（{remaining_time:.2f}秒以内に完了しませんでした）"
            print(f"  [タイムアウト] {error_msg}", flush=True)
            # SilentExceptionはモジュールレベルでインポート済み
            raise SilentException(error_msg)
        except KeyboardInterrupt:
            # KeyboardInterruptを確実に捕捉: スレッドをキャンセル試行
            print(f"\n  [中断] タスク{task_index}: execute_program_stringがユーザーによって中断されました", flush=True)
            if future is not None:
                future.cancel()
            raise
        finally:
            # リソースをクリーンアップ
            if executor_pool is not None:
                executor_pool.shutdown(wait=True)

        exec_program_elapsed = time.time() - exec_program_start
        total_elapsed = time.time() - main_start_time

        # execute_program_stringの実行時間が上限を超えた場合
        if exec_program_elapsed > MAX_EXECUTION_TIME * 0.8:  # 80%を超えた場合は警告
            print(f"  [警告] タスク{task_index}: execute_program_stringの実行時間が長いです（{exec_program_elapsed:.2f}秒）", flush=True)

        # デバッグ用の環境変数をクリア
        if task_index is not None and task_index <= 10:
            if 'DEBUG_TASK_INDEX' in os.environ:
                del os.environ['DEBUG_TASK_INDEX']
                # デフォルトで無効化（詳細ログ）
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [デバッグ] 環境変数クリア完了", flush=True)

        # デバッグ: 最初の10タスクのみ実行結果を出力
        # デフォルトで無効化（詳細ログ）
        if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and task_index is not None and task_index <= 10:
            print(f"  [デバッグ] タスク{task_index}: プログラム実行後")
            print(f"    output_grid.shape={output_grid.shape}, output_grid unique={sorted(np.unique(output_grid).tolist())}")
            print(f"    input_grid == output_grid: {np.array_equal(input_grid, output_grid)}")
            if np.array_equal(input_grid, output_grid):
                print(f"    [警告] タスク{task_index}: プログラム実行後もinput_gridとoutput_gridが一致しています！")
                print(f"    [警告] プログラムが何も変更していない可能性があります。")
                # 実行されたオブジェクト数を確認
                if final_objects:
                    print(f"    final_objects数: {len(final_objects)}")
                else:
                    print(f"    [警告] final_objectsが空またはNoneです")

        # プロファイリング統計を出力
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            from .performance_profiler import print_profiling_statistics
            print_profiling_statistics()

        # 結果の妥当性チェック
        if nodes is None:
            raise ValueError("core_executor: nodesがNoneです")
        if input_grid is None:
            raise ValueError("core_executor: input_gridがNoneです")
        if output_grid is None:
            raise ValueError("core_executor: output_gridがNoneです")

        # トレース（results）の取得（オプション）
        trace_results: Optional[List[Any]] = None
        if ENABLE_TRACE_FOR_DATASET:
            try:
                if hasattr(executor, 'executor') and hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'get_persistent_state'):
                    state = executor.executor.core.get_persistent_state()
                    trace_results = state.get('results', [])
            except Exception:
                trace_results = None

        # バッチごとの完全リセットのため、処理完了時にExecutorCoreの永続状態をクリア
        if hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'clear_persistent_data'):
            try:
                executor.executor.core.clear_persistent_data()
            except Exception:
                pass  # クリア処理でエラーが発生しても無視
        # grid_contextもリセット（状態汚染を防止）
        if hasattr(executor.executor, 'core') and hasattr(executor.executor.core, 'grid_context'):
            if hasattr(executor.executor.core.grid_context, 'reset_to_input_size'):
                try:
                    executor.executor.core.grid_context.reset_to_input_size()
                except Exception:
                    pass  # リセットに失敗しても無視
        # グローバルなgrid_size_contextもリセット（状態汚染を防止）
        try:
            from src.core_systems.executor.grid import grid_size_context
            # GridSizeContextはNoneでリセット（状態汚染を防止）
            grid_size_context.input_grid_size = None
            grid_size_context._initialized = False
        except Exception:
            pass  # リセットに失敗しても無視

        return nodes, input_grid, output_grid, trace_results

    except Exception as e:
        # 例外の詳細をログに記録（UnboundLocalErrorなどの問題を特定するため）
        exception_type = type(e).__name__
        exception_message = str(e)
        print(f"[core_executor] 最終例外発生: 例外型: {exception_type}, メッセージ: {exception_message}", flush=True)

        # SilentExceptionの場合はスタックトレースを出力しない（処理速度向上のため）
        # UnboundLocalErrorを防ぐため、グローバルスコープから明示的に参照
        try:
            # SilentExceptionが利用可能か確認
            from src.core_systems.executor.core import SilentException as SE
            if isinstance(e, SE):
                raise  # 即座に再発生して処理を中断
        except (ImportError, NameError) as import_error:
            # SilentExceptionが利用できない場合は、型名で判定
            print(f"[core_executor] SilentExceptionインポートエラー: {type(import_error).__name__}, {import_error}", flush=True)
            if type(e).__name__ == 'SilentException':
                raise  # 即座に再発生して処理を中断
        except Exception as check_error:
            # isinstanceチェック中にエラーが発生した場合
            print(f"[core_executor] isinstanceチェックエラー: {type(check_error).__name__}, {check_error}", flush=True)
            if type(e).__name__ == 'SilentException':
                raise  # 即座に再発生して処理を中断

        # SilentException以外のエラーはログ出力（デバッグ用）
        if ENABLE_VERBOSE_LOGGING:
            print(f"[core_executor] [実行3 エラー] 最終プログラム実行中にエラー: {e}", flush=True)
            import traceback
            traceback.print_exc()
        # エラー時は例外を再発生させて、呼び出し元で適切に処理されるようにする
        raise