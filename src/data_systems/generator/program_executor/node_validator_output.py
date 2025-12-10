"""
ノード検証とオブジェクト調整のロジック - Output_grid比照版 v3
新しいプログラム改良方法:
1. すべてのコマンドを抽出
2. ループ: オブジェクト生成 → 実行 → 条件チェック（条件に応じて再生）
3. ループ終了後、各コマンドを置き換え判定（完全一致のみ置き換え）
"""
import numpy as np
import random
from typing import List, Optional, Tuple, Any, Dict, Set
import os
import time

# ログ出力制御（config.pyから読み込み）
from src.data_systems.generator.config import get_config

_config = get_config()

ENABLE_VERBOSE_LOGGING = _config.enable_verbose_logging
ENABLE_ALL_LOGS = _config.enable_all_logs

# 色数調査ログの制御（デフォルト: 無効）
ENABLE_COLOR_INVESTIGATION_LOGS = os.environ.get('ENABLE_COLOR_INVESTIGATION_LOGS', 'false').lower() in ('true', '1', 'yes')

# 詳細タイミングログの制御（デフォルト: 無効）- 処理時間の詳細分析用
ENABLE_DETAILED_TIMING_LOGS = _config.enable_detailed_timing_logs
# DEBUGログの制御（デフォルト: 無効）
ENABLE_DEBUG_LOGS = _config.enable_debug_logs

# 置き換え検証ループ制御
ENABLE_REPLACEMENT_VERIFICATION = _config.enable_replacement_verification
MAX_REPLACEMENT_COMMANDS = _config.max_replacement_commands

# グリッド再生成制御
# MAX_CONSECUTIVE_CONTINUES: 連続して条件に該当し続ける場合の早期終了（通常はこちらが機能）
# MAX_GRID_REGENERATION_ATTEMPTS: 絶対的な上限（whileループの条件として必須、通常はMAX_CONSECUTIVE_CONTINUESで終了するため使われない）
MAX_CONSECUTIVE_CONTINUES = _config.max_consecutive_continues
MAX_GRID_REGENERATION_ATTEMPTS = _config.max_grid_regeneration_attempts

from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
from src.data_systems.generator.input_grid_generator.builders import generate_objects_from_conditions
from src.data_systems.generator.input_grid_generator.builders.color_distribution import decide_object_color_count, select_object_colors
from src.data_systems.generator.input_grid_generator.managers import build_grid
from .node_analyzer import is_assignment_node, extract_commands_from_node, get_commands_sorted_by_depth
from .node_analyzer.execution_helpers import replace_command_with_fallback
from .performance_profiler import profile_code_block
# SilentExceptionをモジュールレベルでインポート（スコープ問題を回避）
from src.core_systems.executor.core import SilentException


def extract_all_commands_from_nodes(nodes: List[Any]) -> Set[str]:
    """すべてのノードから使用されているコマンドを抽出

    Args:
        nodes: プログラムのNodeリスト
    Returns:
        使用されているコマンド名のセット（重複なし）
    """
    all_commands = set()

    for node in nodes:
        commands = extract_commands_from_node(node)
        all_commands.update(commands)

    return all_commands


def get_duplicate_mode_kwargs_from_commands(all_commands: Set[str]) -> Dict[str, Any]:
    """コマンドの種類に応じて、generate_objects_from_conditionsの複製に関するkwargsを返す

    Args:
        all_commands: プログラムで使用されているコマンド名のセット

    Returns:
        複製に関するkwargs辞書（duplicate_mode等）
    """
    kwargs = {}

    # MATCH_PAIRSコマンドがある場合: 色と形状を複製
    if 'MATCH_PAIRS' in all_commands:
        kwargs['duplicate_mode'] = 'exact'  # 色と形状を100%完全複製

    # EXTEND_PATTERNコマンドがある場合: 50%の確率で色と形状を複製
    elif 'EXTEND_PATTERN' in all_commands:
        if random.random() < 0.5:
            kwargs['duplicate_mode'] = 'exact'  # 色と形状を100%完全複製
        else:
            kwargs['duplicate_mode'] = 'shape_only'  # 形だけ複製

    # FIT_SHAPE_COLORコマンドがある場合: 色と形状を複製
    elif 'FIT_SHAPE_COLOR' in all_commands:
        kwargs['duplicate_mode'] = 'exact'  # 色と形状を100%完全複製

    # FIT_SHAPEコマンドがある場合: 50%の確率で色と形状を複製
    elif 'FIT_SHAPE' in all_commands:
        if random.random() < 0.5:
            kwargs['duplicate_mode'] = 'exact'  # 色と形状を100%完全複製
        else:
            kwargs['duplicate_mode'] = 'shape_only'  # 形だけ複製

    return kwargs


def get_generate_objects_kwargs_from_commands(all_commands: Set[str]) -> Dict[str, Any]:
    """コマンドの種類に応じて、generate_objects_from_conditionsに渡すkwargsを返す（複製以外）

    Args:
        all_commands: プログラムで使用されているコマンド名のセット

    Returns:
        generate_objects_from_conditionsに渡すkwargs辞書（shape_type, symmetry_constraint, min_spacing等）
    """
    kwargs = {}

    # EXTRACT_RECTSコマンドがある場合: 矩形生成を優先
    if 'EXTRACT_RECTS' in all_commands:
        kwargs['shape_type'] = 'rectangle'
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] EXTRACT_RECTS検出: shape_type='rectangle'")

    # EXTRACT_LINESコマンドがある場合: 線生成を優先
    elif 'EXTRACT_LINES' in all_commands:
        kwargs['shape_type'] = 'line'
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] EXTRACT_LINES検出: shape_type='line'")

    # EXTRACT_HOLLOW_RECTSコマンドがある場合: 中空矩形生成を優先
    elif 'EXTRACT_HOLLOW_RECTS' in all_commands:
        kwargs['shape_type'] = 'hollow_rectangle'
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] EXTRACT_HOLLOW_RECTS検出: shape_type='hollow_rectangle'")

    # FILL_HOLESコマンドがある場合: 穴が1つ以上のオブジェクトが必要
    if 'FILL_HOLES' in all_commands:
        kwargs['allow_holes'] = True
        kwargs['min_holes'] = 1
        kwargs['max_holes'] = 5
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] FILL_HOLES検出: allow_holes=True, min_holes=1, max_holes=5")

    # HOLLOWコマンドがある場合: 3x3以上のオブジェクトが必要
    if 'HOLLOW' in all_commands:
        kwargs['min_bbox_size'] = 3
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] HOLLOW検出: min_bbox_size=3")

    # SCALE_DOWNコマンドがある場合: 2x2以上のオブジェクトが必要
    if 'SCALE_DOWN' in all_commands:
        if 'min_bbox_size' not in kwargs:  # HOLLOWで既に設定されている場合は上書きしない
            kwargs['min_bbox_size'] = 2
            if ENABLE_VERBOSE_LOGGING:
                print(f"    [生成条件] SCALE_DOWN検出: min_bbox_size=2")

    # BBOXコマンドがある場合: 密度が100%未満、Ex2以上のオブジェクトが必要
    if 'BBOX' in all_commands:
        if 'min_bbox_size' not in kwargs:
            kwargs['min_bbox_size'] = 2
        kwargs['max_density'] = 0.99
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] BBOX検出: min_bbox_size=2, max_density=0.99")

    # ROTATEまたはFLIPコマンドがある場合: 対称性制約を適用
    if 'ROTATE' in all_commands or 'FLIP' in all_commands:
        # 対称性のある形状を生成しやすくするため、symmetry_constraintを指定
        symmetry_types = ['vertical', 'horizontal', 'both']
        symmetry_constraint = random.choice(symmetry_types)
        kwargs['symmetry_constraint'] = symmetry_constraint
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] ROTATE/FLIP検出: symmetry_constraint='{symmetry_constraint}'")

    # EXTEND_PATTERNまたはMATCH_PAIRSコマンドがある場合: 形状制約を緩和（複製は別関数で処理）
    # shape_typeは設定しない（Noneのまま、形状制約なし）
    if ENABLE_VERBOSE_LOGGING:
        if 'EXTEND_PATTERN' in all_commands or 'MATCH_PAIRS' in all_commands:
            print(f"    [生成条件] EXTEND_PATTERN/MATCH_PAIRS検出: 形状制約なし（複製は別関数で処理）")

    # 最小スペース設定（必要に応じてコマンドに応じて自動設定可能）
    # FLOW、LAY、FIT_ADJACENTコマンドがある場合: オブジェクト間の最小スペースを1に設定
    # 注意: デフォルト値（0）は設定しない。空のコマンドセットの場合は空の辞書を返し、
    # 一括生成（elseブロック）が実行されるようにする
    if 'FLOW' in all_commands or 'LAY' in all_commands or 'FIT_ADJACENT' in all_commands:
        kwargs['min_spacing'] = 1
        if ENABLE_VERBOSE_LOGGING:
            print(f"    [生成条件] FLOW/LAY/FIT_ADJACENT検出: min_spacing=1")

    # デフォルト値は設定しない（generate_objects_from_conditions内で処理される）
    # これにより、空のコマンドセットの場合は空の辞書が返され、一括生成が実行される

    return kwargs


def decide_num_objects_by_arc_statistics(
    grid_width: int = None,
    grid_height: int = None,
    all_commands: set = None
) -> int:
    """ARC-AGI2統計に基づく確率分布でオブジェクト数を決定（グリッド面積に基づいて直接決定）

    ARC-AGI2実データ分析結果（3232サンプル）に基づく:
    - グリッド面積 < 100: 平均=7.87, 標準偏差=6.64, 範囲=1-65
    - グリッド面積 100-200: 平均=10.68, 標準偏差=10.31, 範囲=1-109
    - グリッド面積 200-300: 平均=19.72, 標準偏差=28.82, 範囲=1-206
    - グリッド面積 300-400: 平均=29.08, 標準偏差=45.44, 範囲=2-298
    - グリッド面積 400-600: 平均=44.98, 標準偏差=71.63, 範囲=2-500
    - グリッド面積 >= 600: 平均=109.93, 標準偏差=165.88, 範囲=2-785

    Args:
        grid_width: グリッド幅（オプション、指定時は面積に基づいて決定）
        grid_height: グリッド高さ（オプション、指定時は面積に基づいて決定）
        all_commands: 使用されているコマンドのセット（オプション、MATCH_PAIRSがある場合は4以上になる）

    Returns:
        決定されたオブジェクト数（グリッド面積に基づいて決定、最小2個）
    """
    import math

    # グリッド面積に基づいて直接オブジェクト数を決定
    if grid_width is not None and grid_height is not None:
        grid_area = grid_width * grid_height

        # ARC-AGI2実データに基づく分布パラメータ
        # 対数正規分布を使用して実データの分布を近似
        if grid_area < 100:
            # 平均=7.87, 標準偏差=6.64, 範囲=1-65
            mean = 7.87
            std = 6.64
            min_val = 1
            max_val = 65
        elif grid_area < 200:
            # 平均=10.68, 標準偏差=10.31, 範囲=1-109
            mean = 10.68
            std = 10.31
            min_val = 1
            max_val = 109
        elif grid_area < 300:
            # 平均=19.72, 標準偏差=28.82, 範囲=1-206
            mean = 19.72
            std = 28.82
            min_val = 1
            max_val = 206
        elif grid_area < 400:
            # 平均=29.08, 標準偏差=45.44, 範囲=2-298
            mean = 29.08
            std = 45.44
            min_val = 2
            max_val = 298
        elif grid_area < 600:
            # 平均=44.98, 標準偏差=71.63, 範囲=2-500
            mean = 44.98
            std = 71.63
            min_val = 2
            max_val = 500
        else:
            # 平均=109.93, 標準偏差=165.88, 範囲=2-785
            # ただし、パフォーマンスを考慮して上限を制限（MAX_OBJECTS_PER_GENERATION=25を考慮）
            mean = 109.93
            std = 165.88
            min_val = 2
            # 実データの最大値は785だが、パフォーマンスを考慮して100に制限
            # 実際の生成時にはMAX_OBJECTS_PER_GENERATION=25でさらに制限される
            max_val = min(100, 785)

        # 対数正規分布のパラメータを計算
        # log_mean = ln(mean^2 / sqrt(std^2 + mean^2))
        # log_std = sqrt(ln(1 + (std/mean)^2))
        if mean > 0 and std > 0:
            log_mean = math.log(mean ** 2 / math.sqrt(std ** 2 + mean ** 2))
            log_std = math.sqrt(math.log(1 + (std / mean) ** 2))
        else:
            # フォールバック: 平均値を使用
            log_mean = math.log(max(mean, 1))
            log_std = 0.5

        # 対数正規分布からサンプル（最大10回試行）
        num_objects = None
        for _ in range(10):
            sample = random.lognormvariate(log_mean, log_std)
            num_objects = int(sample)
            if min_val <= num_objects <= max_val:
                break

        # 範囲内に収まらない場合は、範囲内のランダムな値を使用
        if num_objects is None or num_objects < min_val or num_objects > max_val:
            num_objects = random.randint(min_val, max_val)

        adjusted_num = num_objects
    else:
        # グリッドサイズが指定されていない場合: 全体の平均（18.27）に基づく分布を使用
        # 平均=18.27, 標準偏差≈15（推定）
        mean = 18.27
        std = 15.0
        log_mean = math.log(mean ** 2 / math.sqrt(std ** 2 + mean ** 2))
        log_std = math.sqrt(math.log(1 + (std / mean) ** 2))

        num_objects = None
        for _ in range(10):
            sample = random.lognormvariate(log_mean, log_std)
            num_objects = int(sample)
            if 2 <= num_objects <= 100:
                break

        if num_objects is None or num_objects < 2 or num_objects > 100:
            num_objects = random.randint(2, 100)

        adjusted_num = num_objects

    # MATCH_PAIRSがある場合は4以上にする
    if all_commands and 'MATCH_PAIRS' in all_commands:
        if adjusted_num < 4:
            adjusted_num = random.randint(4, min(30, adjusted_num + 10))
            if ENABLE_VERBOSE_LOGGING:
                print(f"    [MATCH_PAIRS調整] MATCH_PAIRS検出により、オブジェクト数を{adjusted_num}個に調整")

    # 最小2個を保証
    adjusted_num = max(2, adjusted_num)

    # 最大値を設定（グリッドサイズに応じて）
    if grid_width is not None and grid_height is not None:
        grid_area = grid_width * grid_height
        # グリッド面積の約1/3を上限とする（オブジェクトが小さすぎないように）
        # ただし、パフォーマンスを考慮して100を上限とする
        # 実際の生成時にはMAX_OBJECTS_PER_GENERATION=25でさらに制限される
        max_objects = min(100, max(50, grid_area // 3))
        adjusted_num = min(adjusted_num, max_objects)

    return adjusted_num


def _is_empty_output(output_grid: np.ndarray) -> bool:
    """outputが空（0ピクセル）かどうかを判定

    Args:
        output_grid: 出力グリッド

    Returns:
        空の場合はTrue
    """
    if output_grid is None:
        return True

    if not isinstance(output_grid, np.ndarray):
        return False

    # 空配列の場合（size=0）
    if output_grid.size == 0:
        return True

    # 2次元配列の場合、すべての行が空配列かチェック
    if output_grid.ndim == 2:
        # shape[0]が0（行数が0）の場合
        if output_grid.shape[0] == 0:
            return True
        # shape[1]が0（列数が0、つまりすべての行が空配列）の場合
        if output_grid.shape[1] == 0:
            return True
        # 各行が空配列かチェック（通常はshape[1]==0でカバーされるが、念のため）
        if all(len(row) == 0 for row in output_grid):
            return True

    # 1次元配列で空の場合
    if output_grid.ndim == 1 and len(output_grid) == 0:
        return True

    # 総ピクセル数が0の場合
    # 通常は上記の条件でカバーされるが、念のため
    return output_grid.size == 0


def check_output_conditions(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    background_color: int = None,
    early_exit: bool = True
) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """出力グリッドの条件をチェック（最適化版: 軽い条件を先にチェック）

    Args:
        input_grid: 入力グリッド
        output_grid: 出力グリッド
        background_color: 入力グリッドの背景色（条件②の判定に使用）
        early_exit: Trueの場合、軽い条件を先にチェックして早期終了（デフォルト: True）

    Returns:
        (条件①一致, 条件②トリミング一致, 条件③入力が単色, 条件④オブジェクトピクセル数5%未満かつ出力グリッドの総色数が2以下, 条件⑤空のoutput, 条件⑥出力が単色)のタプル
    """
    if output_grid is None:
        # 条件⑤: Noneの場合は空として扱う
        # 条件③: 入力が単色かチェック（出力がNoneでも入力が単色なら条件③に該当）
        input_unique_colors = np.unique(input_grid)
        condition3 = len(input_unique_colors) == 1
        # 条件⑥: 出力がNoneの場合はFalse
        condition6 = False
        return False, False, condition3, False, True, condition6

    # 最適化: 軽い条件を先にチェック（早期終了）
    condition3 = False
    condition5 = False
    condition6 = False

    if early_exit:
        # 条件③: インプットグリッドがすべて単色（軽い: O(n)）
        input_unique_colors = np.unique(input_grid)
        condition3 = len(input_unique_colors) == 1

        # 条件⑤: アウトプットグリッドが空（軽い: O(1)）
        condition5 = _is_empty_output(output_grid)

        # 条件⑥: アウトプットグリッドがすべて単色（軽い: O(n)）
        output_unique_colors = np.unique(output_grid)
        condition6 = len(output_unique_colors) == 1

        # 軽い条件で早期終了できる場合は、重い条件をスキップ
        if condition3 or condition5 or condition6:
            # 条件①と条件②は重いので、必要に応じて計算
            # ただし、早期終了する場合は計算をスキップ
            condition1 = False
            condition2 = False
            condition4 = False
            return condition1, condition2, condition3, condition4, condition5, condition6

    # 条件①: アウトグリッドとインプットグリッドが一致（重い: O(n)）
    condition1 = np.array_equal(input_grid, output_grid)

    # 条件②: インプットグリッドとアウトプットグリッドの左上をそろえて重ねたときに、
    # 重なった部分で、入力グリッドの背景色部分以外が一致
    # 入力グリッドの背景色位置に対応する出力位置を重なっていない部分に含める
    # 重なっていない部分がすべて1つの単色ならTrue
    condition2 = False
    # 重なり部分のサイズを計算
    overlap_h = min(input_grid.shape[0], output_grid.shape[0])
    overlap_w = min(input_grid.shape[1], output_grid.shape[1])
    if overlap_h > 0 and overlap_w > 0:
        # 左上(0,0)を基準に重なった部分を抽出
        input_overlap = input_grid[:overlap_h, :overlap_w]
        output_overlap = output_grid[:overlap_h, :overlap_w]

        # 重なった部分で、入力グリッドの背景色部分以外が一致しているかチェック
        if background_color is not None:
            # 入力グリッドの背景色部分をマスク
            input_bg_mask = (input_overlap == background_color)

            # 背景色部分以外が一致しているかチェック
            non_bg_matches = np.array_equal(
                input_overlap[~input_bg_mask],
                output_overlap[~input_bg_mask]
            )

            if non_bg_matches:
                # 入力グリッドの背景色位置に対応する出力位置を取得（重なっていない部分として扱う）
                output_bg_positions = output_overlap[input_bg_mask]

                # 通常の重なっていない部分を取得（出力グリッドが大きい場合のみ）
                # 注意: 入力グリッドが大きい場合の重なっていない部分は判定に含めない
                non_overlap_parts = []
                if output_grid.shape[0] > input_grid.shape[0]:
                    # 出力グリッドの下の余分な部分
                    bottom_non_overlap = output_grid[input_grid.shape[0]:, :]
                    non_overlap_parts.append(bottom_non_overlap.flatten())
                if output_grid.shape[1] > input_grid.shape[1]:
                    # 出力グリッドの右側の余分な部分（重なった部分の右側）
                    right_non_overlap = output_grid[:overlap_h, input_grid.shape[1]:]
                    non_overlap_parts.append(right_non_overlap.flatten())

                # 入力グリッドの背景色位置に対応する出力位置と通常の重なっていない部分を結合
                all_non_overlap_colors = list(output_bg_positions.flatten())
                for part in non_overlap_parts:
                    all_non_overlap_colors.extend(part)

                # すべてが1つの単色かチェック
                if len(all_non_overlap_colors) > 0:
                    unique_colors = np.unique(all_non_overlap_colors)
                    condition2 = len(unique_colors) == 1
                else:
                    # 重なっていない部分がない場合は、non_bg_matchesのみで判定（True）
                    condition2 = True
            else:
                condition2 = False
        else:
            # 背景色が指定されていない場合は、従来通り完全一致をチェック
            overlap_matches = np.array_equal(input_overlap, output_overlap)

            if overlap_matches:
                # 重なっていない部分をチェック（アウトプットグリッドが大きい場合）
                non_overlap_parts = []
                if output_grid.shape[0] > input_grid.shape[0]:
                    # 下の余分な部分
                    bottom_non_overlap = output_grid[input_grid.shape[0]:, :]
                    non_overlap_parts.append(bottom_non_overlap)
                if output_grid.shape[1] > input_grid.shape[1]:
                    # 右側の余分な部分（重なった部分の右側）
                    right_non_overlap = output_grid[:overlap_h, input_grid.shape[1]:]
                    non_overlap_parts.append(right_non_overlap)

                # 重なっていない部分がすべて単色かチェック
                all_monochrome = True
                for part in non_overlap_parts:
                    unique_colors = np.unique(part)
                    if len(unique_colors) != 1:
                        all_monochrome = False
                        break

                # 重なった部分が一致 AND 重なっていない部分がすべて単色
                condition2 = all_monochrome
            else:
                condition2 = False

    # 条件③: インプットグリッドがすべて単色（early_exitがFalseの場合のみ再計算）
    if not early_exit or not condition3:
        input_unique_colors = np.unique(input_grid)
        input_is_single_color = len(input_unique_colors) == 1
        condition3 = input_is_single_color

    # 条件⑥: アウトプットグリッドがすべて単色（early_exitがFalseの場合のみ再計算）
    if not early_exit or not condition6:
        output_unique_colors = np.unique(output_grid)
        output_is_single_color = len(output_unique_colors) == 1
        condition6 = output_is_single_color

    # 条件④: オブジェクトピクセル数5%未満かつ出力グリッドの総色数が2以下
    condition4 = False
    total_pixels = output_grid.size  # 総ピクセル数

    if total_pixels > 0:
        # 出力グリッドの背景色を推論（最も頻繁に出現する色）
        output_unique, output_counts = np.unique(output_grid, return_counts=True)
        output_background_color = output_unique[np.argmax(output_counts)] if len(output_unique) > 0 else None
        output_total_colors = len(output_unique)  # 出力グリッドの総色数

        if output_background_color is not None:
            # オブジェクトピクセル（背景色以外のピクセル）の数をカウント
            object_pixels = np.sum(output_grid != output_background_color)
            object_pixel_ratio = object_pixels / total_pixels

            # オブジェクトピクセル数が5%未満かつ出力グリッドの総色数が2以下の場合、条件④をTrueにする
            if object_pixel_ratio < 0.05 and output_total_colors <= 2:
                condition4 = True

    # 条件⑤: アウトプットグリッドが空（0ピクセル）
    condition5 = _is_empty_output(output_grid)

    return condition1, condition2, condition3, condition4, condition5, condition6


def should_continue_loop(condition1: bool, condition2: bool, condition3: bool, condition4: bool = False, condition5: bool = False, condition6: bool = False) -> bool:
    """ループを継続すべきかを判定
    条件①、②、③、④、⑤、⑥のいずれかに該当する場合は継続

    Args:
        condition1: 入力と出力が一致
        condition2: トリミングで一致
        condition3: 入力が単色
        condition4: オブジェクトピクセル数5%未満（オプション、デフォルト: False）
        condition5: 空のoutput（0ピクセル）（オプション、デフォルト: False）
        condition6: 出力が単色（オプション、デフォルト: False）

    Returns:
        True: ループ継続、False: ループ終了
    """
    return condition1 or condition2 or condition3 or condition4 or condition5 or condition6


def validate_nodes_and_adjust_objects(
    nodes: Optional[List[Any]],
    all_objects: List[Dict],
    background_color: int,
    grid_width: int,
    grid_height: int,
    background_grid_pattern: np.ndarray,
    executor: Any,
    if_count: int = 0,
    for_count: int = 0,
    end_count: int = 0,
    enable_replacement: bool = True,
    all_commands: Optional[Set[str]] = None,
    program_code: Optional[str] = None,
    is_first_pair: bool = True,
    force_mode: Optional[str] = None
) -> Tuple[Optional[List[Any]], List[Dict], int, int, int]:
    """ノードを検証し、必要に応じてオブジェクトを調整・追加する - v3版

    Args:
        nodes: プログラムのNodeリスト（Noneの場合はプログラムなしで入力グリッドのみ生成）
        all_objects: オブジェクトリスト（変更される）
        background_color: 背景色
        grid_width: グリッド幅
        grid_height: グリッド高さ
        background_grid_pattern: 背景グリッドパターン
        executor: CoreExecutorインスタンス
        if_count: IF文のカウント（デフォルト値: 0）
        for_count: FORループのカウント（デフォルト値: 0）
        end_count: ENDのカウント（デフォルト値: 0）
        enable_replacement: Trueの場合、コマンド置き換え判定を実行（ステップ5）
        all_commands: 抽出済みのコマンドセット（提供されている場合はステップ1をスキップ）
        program_code: 生成済みのプログラムコード（提供されている場合はステップ2をスキップ）

    Returns:
        (更新されたnodes, 更新されたall_objects, if_count, for_count, end_count)のタプル
    """
    # SilentExceptionはモジュールレベルでインポート済み
    # ログ出力の最適化: 条件を事前にチェックして文字列フォーマット処理を削減
    _log_enabled = ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] 関数開始", flush=True)

    # グリッド面積を事前計算（ループ内で使用される定数）
    if grid_width is not None and grid_height is not None:
        grid_area = grid_width * grid_height
    else:
        grid_area = 0  # デフォルト値

    # プログラムがない場合の初期化（ただし、program_codeが提供されている場合はプログラムありとして扱う）
    if nodes is None and program_code is None:
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] プログラムなしモード: 入力グリッド生成", flush=True)
        # コマンド抽出をスキップ
        if all_commands is None:
            all_commands = set()

    # 1. プログラムで使われているコマンドをすべて洗い出す
    # all_commandsが提供されている場合はスキップ（最初以外のペア用の最適化）
    if all_commands is None:
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] ステップ1: コマンド抽出開始", flush=True)
        try:
            if nodes is not None:
                # nodesからコマンドを抽出
                all_commands = extract_all_commands_from_nodes(nodes)
            elif program_code is not None:
                # program_codeからコマンドを抽出
                import re
                commands = set()
                # 一般的なコマンドパターンを抽出
                command_pattern = r'\b([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]*)?)\s*\('
                matches = re.findall(command_pattern, program_code)
                commands.update(matches)
                # FOR文のパターン
                if re.search(r'\bFOR\s+', program_code, re.IGNORECASE):
                    commands.add('FOR i LEN')
                # IF文のパターン
                if re.search(r'\bIF\s+', program_code, re.IGNORECASE):
                    commands.add('IF')
                # 条件演算子のパターン
                condition_operators = ['AND', 'OR', 'NOT', 'EQUAL', 'NOT_EQUAL', 'IS_INSIDE', 'IS_SAME_SHAPE']
                for op in condition_operators:
                    if op in program_code:
                        commands.add(op)
                all_commands = commands
            else:
                # nodesもprogram_codeもない場合は空セット
                all_commands = set()

            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] ステップ1完了: コマンド数={len(all_commands)}", flush=True)
        except Exception as e:
            # エラーログは常に出力（重要な情報）
            print(f"[validate_nodes_and_adjust_objects] ステップ1エラー: {e}", flush=True)
            if _log_enabled:
                import traceback
                traceback.print_exc()
            raise

        if _log_enabled:
            print(f"[コマンド抽出] 使用されているコマンド数: {len(all_commands)}")
            print(f"[コマンド抽出] コマンド一覧: {sorted(all_commands)}")
    else:
        # all_commandsが提供されている場合（最初以外のペア）
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] ステップ1: コマンド抽出をスキップ（提供済み、コマンド数={len(all_commands)}）", flush=True)

    # 完全なプログラムコードを生成
    # program_codeが提供されている場合はスキップ（最初以外のペア用の最適化）
    # プログラムがない場合（nodes is None）もスキップ
    if program_code is None and nodes is not None:
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] ステップ2: プログラムコード生成開始", flush=True)
        try:
            context = ProgramContext(complexity=1)
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] ステップ2-1完了: ProgramContext作成", flush=True)
        except Exception as e:
            # エラーログは常に出力（重要な情報）
            print(f"[validate_nodes_and_adjust_objects] ステップ2-1エラー: {e}", flush=True)
            if _log_enabled:
                import traceback
                traceback.print_exc()
            raise

        if executor.program_generator is None:
            # エラーログは常に出力（重要な情報）
            print(f"[validate_nodes_and_adjust_objects] エラー: executor.program_generator is None", flush=True)
            raise ValueError("executor.program_generator is None - プログラム生成が不可能です")

        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] ステップ2-2: _generate_code呼び出し開始", flush=True)
        try:
            try:
                program_code = executor.program_generator._generate_code(nodes, context, preserve_indentation=True)
            except TypeError:
                program_code = executor.program_generator._generate_code(nodes, context)
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] ステップ2完了: プログラムコード生成完了 (長さ={len(program_code) if program_code else 0})", flush=True)
        except Exception as e:
            # エラーログは常に出力（重要な情報）
            print(f"[validate_nodes_and_adjust_objects] ステップ2エラー: {e}", flush=True)
            if _log_enabled:
                import traceback
                traceback.print_exc()
            raise
    else:
        # program_codeが提供されている場合（最初以外のペア）
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] ステップ2: プログラムコード生成をスキップ（提供済み、長さ={len(program_code) if program_code else 0}）", flush=True)

    # 2. ループ: オブジェクト生成 → 実行 → 条件チェック
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] ステップ3: ループ初期化開始", flush=True)
    max_attempts = MAX_GRID_REGENERATION_ATTEMPTS  # 最大試行回数（環境変数から設定可能）
    attempt = 0
    consecutive_continue_count = 0  # 連続してcontinueした回数
    max_consecutive_continues = MAX_CONSECUTIVE_CONTINUES  # 連続continueの最大回数（環境変数から設定可能）
    best_objects = None  # 条件に該当しなかった最後のオブジェクトセットを保持
    loop_start_time = time.time()  # ループ全体の開始時刻
    MAX_LOOP_TIME = 30.0  # ループ全体の最大実行時間（秒）- 無限ループ防止

    # タイムアウト無効化フラグをチェック
    DISABLE_TIMEOUTS = os.environ.get('DISABLE_TIMEOUTS', 'false').lower() in ('true', '1', 'yes')
    max_attempt_time = 0.8 if not DISABLE_TIMEOUTS else float('inf')  # 各試行の最大実行時間（秒）

    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] ステップ3完了: ループ初期化完了 (max_attempts={max_attempts}, max_attempt_time={max_attempt_time})", flush=True)
        print(f"[node_validator_output] バリデーションループ開始 (max_attempts={max_attempts}, max_attempt_time={max_attempt_time}, DISABLE_TIMEOUTS={DISABLE_TIMEOUTS})", flush=True)
        print(f"[validate_nodes_and_adjust_objects] ステップ4: whileループ開始", flush=True)
    while attempt < max_attempts:
        # ループ全体のタイムアウトチェック
        elapsed_from_loop_start = time.time() - loop_start_time
        if elapsed_from_loop_start > MAX_LOOP_TIME:
            if ENABLE_VERBOSE_LOGGING:
                print(f"[タイムアウト] validate_nodes_and_adjust_objectsループが{MAX_LOOP_TIME}秒を超えました。ループを終了します。", flush=True)
            if best_objects is not None:
                all_objects = best_objects
                break
            # タイムアウトでタスクを破棄
            return None, None, None, None, None

        attempt += 1
        attempt_start_time = time.time()
        if ENABLE_DETAILED_TIMING_LOGS:
            print(f"[TIMING] 試行{attempt}: 試行開始 (ループ経過時間: {elapsed_from_loop_start:.4f}秒)", flush=True)
        # elapsed_from_loop_startはループ開始時に既に計算済み（上記で計算）

        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}/{max_attempts} 開始 (ループ経過時間: {elapsed_from_loop_start:.1f}秒)", flush=True)
        if ENABLE_VERBOSE_LOGGING:
            print(f"[入力グリッド再生] 試行{attempt}/{max_attempts}] 開始 (ループ経過時間: {elapsed_from_loop_start:.1f}秒)", flush=True)

        # グリッド内の色数を決定（各試行ごとに独立に決定）
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 色数決定開始", flush=True)
        color_decide_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
        target_color_count = decide_object_color_count(
            existing_colors=set(),
            background_color=background_color
        )

        # 色数がグリッド面積を超えないように制限（grid_areaはループ外で事前計算済み）
        target_color_count = min(target_color_count, grid_area)

        selected_colors = select_object_colors(
            background_color=background_color,
            target_color_count=target_color_count,
            existing_colors=set()
        )
        grid_color_list = selected_colors.copy() if selected_colors else [1 if background_color == 0 else 0]
        if ENABLE_DETAILED_TIMING_LOGS and color_decide_start is not None:
            color_decide_elapsed = time.time() - color_decide_start
            print(f"[TIMING] 試行{attempt}: 色数決定: {color_decide_elapsed:.4f}秒 (色数={target_color_count})", flush=True)

        # 詳細ログ: 色数決定と選択された色リスト
        if ENABLE_COLOR_INVESTIGATION_LOGS:
            print(f"[色数調査] 試行{attempt}: 背景色={background_color}, 決定色数={target_color_count}, 選択色リスト={selected_colors}, grid_color_list={grid_color_list}", flush=True)

        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 色数決定完了 (色数={target_color_count}, グリッド面積={grid_area})", flush=True)
        if ENABLE_VERBOSE_LOGGING:
            print(f"  [試行{attempt}] [色数決定] ARC統計に基づいて{target_color_count}色を決定（色リスト: {selected_colors}）", flush=True)

        # オブジェクトリストをリセット
        all_objects = []

        # ARC-AGI2の統計に合わせてオブジェクトを作成
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: オブジェクト数決定開始", flush=True)
        num_objects_decide_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
        num_objects = decide_num_objects_by_arc_statistics(
            grid_width=grid_width,
            grid_height=grid_height,
            all_commands=all_commands
        )
        # オブジェクト数が色数以上になることを保証（オブジェクト数 < 色数 の場合、すべての色を使用できないため）
        num_objects = max(num_objects, target_color_count)
        # オブジェクト数がグリッド面積を超えないように制限（grid_areaはループ外で事前計算済み）
        num_objects = min(num_objects, grid_area)

        # 大量オブジェクト生成時の制限（パフォーマンス向上）
        # 25個以上のオブジェクト生成は時間がかかりすぎるため制限（100→25に削減）
        # 統計: num_objects=30で平均0.158秒、num_objects=82で0.524秒かかっている
        MAX_OBJECTS_PER_GENERATION = 25
        if num_objects > MAX_OBJECTS_PER_GENERATION:
            if _log_enabled:
                print(f"[最適化] オブジェクト数が多すぎるため制限: {num_objects} -> {MAX_OBJECTS_PER_GENERATION}", flush=True)
            num_objects = MAX_OBJECTS_PER_GENERATION
        if ENABLE_DETAILED_TIMING_LOGS and num_objects_decide_start is not None:
            num_objects_decide_elapsed = time.time() - num_objects_decide_start
            print(f"[TIMING] 試行{attempt}: オブジェクト数決定: {num_objects_decide_elapsed:.4f}秒 (num_objects={num_objects})", flush=True)

        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: オブジェクト数決定完了 (num_objects={num_objects}, target_color_count={target_color_count}, グリッド面積={grid_area})", flush=True)

        # コマンドの種類によって作成するオブジェクトの条件を調整
        # プログラムなしモードでは、generate_kwargsを空にする
        if nodes is None and program_code is None:
            duplicate_kwargs = {}
            generate_kwargs = {}
        else:
            duplicate_kwargs = get_duplicate_mode_kwargs_from_commands(all_commands)
            generate_kwargs = get_generate_objects_kwargs_from_commands(all_commands)

        if ENABLE_VERBOSE_LOGGING:
            print(f"  [試行{attempt}] コマンド条件解析完了 duplicate_kwargs={bool(duplicate_kwargs)}, generate_kwargs={bool(generate_kwargs)}", flush=True)

        # オブジェクト生成方法を決定（統合版）
        use_individual_reference = False  # False: normal（全既存参照）、True: copy（個別参照）

        # 検証用: force_modeが指定されている場合は強制的にモードを設定
        if force_mode is not None:
            if force_mode == 'normal':
                use_individual_reference = False
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] 検証モード: normalモードを強制", flush=True)
            elif force_mode == 'copy':
                use_individual_reference = True
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] 検証モード: copyモードを強制", flush=True)
            else:
                raise ValueError(f"force_mode must be 'normal' or 'copy', got: {force_mode}")
        elif generate_kwargs or duplicate_kwargs:
            # コマンド条件がある場合: normal or copyを選択
            rand_val1 = random.random()
            rand_val2 = random.random()
            if (duplicate_kwargs and rand_val1 < 0.95) or (rand_val2 < 0.1):
                use_individual_reference = True  # copy mode
            else:
                use_individual_reference = False  # normal mode

            if ENABLE_VERBOSE_LOGGING:
                method_name = 'copy' if use_individual_reference else 'normal'
                print(f"  [試行{attempt}] 生成方法決定完了 method={method_name}", flush=True)

        # force_modeが指定されているか、コマンド条件がある場合は分割生成を使用
        if (force_mode is not None) or (generate_kwargs or duplicate_kwargs):
            # num_normalとnum_conditionalを決定
            # 最適化: num_conditionalの上限を設定（処理時間短縮のため）
            # 統計: num_conditional=15, 17で時間がかかるため、20個に制限（30→20に削減）
            MAX_CONDITIONAL_OBJECTS = 20

            if use_individual_reference:
                # copy mode: 少数の元オブジェクトから複製生成
                if 'MATCH_PAIRS' in all_commands and num_objects >= 4:
                    num_normal = 2
                else:
                    num_normal = random.randint(1, 3)
                    if num_objects < 4:
                        num_normal = 1
                    elif num_objects < 6:
                        num_normal = random.randint(1, 2)
                num_conditional = max(1, (num_objects // num_normal) - 1)  # 最低1を保証
                # 上限を適用
                num_conditional = min(num_conditional, MAX_CONDITIONAL_OBJECTS)
            else:
                # normal mode: 約50%ずつ分割
                min_conditional = max(1, num_objects // 2)  # 最低50%、ただし最低1つ
                max_conditional = max(min_conditional, num_objects - 1)  # 最大: remaining_objectsが1つ
                num_conditional = random.randint(min_conditional, max_conditional)
                # 上限を適用
                num_conditional = min(num_conditional, MAX_CONDITIONAL_OBJECTS)
                num_normal = num_objects - num_conditional

            # 2段階生成の場合、全体で1つのarea_ratioを決定して各段階で共有
            # これにより、合計の利用可能面積がtotal_area * area_ratioになり、密度の一貫性が向上
            shared_area_ratio = None
            if grid_width is not None and grid_height is not None:
                # 乱数生成器を初期化（シードは既存のrandomから取得）
                try:
                    rng_state = random.getstate()
                    seed_value = rng_state[1][0] if len(rng_state) > 1 and len(rng_state[1]) > 0 else None
                    np_rng = np.random.default_rng(seed_value)
                except:
                    np_rng = np.random.default_rng()
                # ARC-AGI2統計に基づいて利用可能面積の割合を決定（0.08から0.95までの範囲）
                # 平均値を下げるため、beta分布のパラメータを調整: alpha=0.2, beta=2.5で平均≈0.074 → 0.08 + 0.074 * 0.87 ≈ 0.144
                beta_sample = np_rng.beta(0.2, 2.5)
                shared_area_ratio = 0.08 + beta_sample * 0.87  # 0.08-0.95範囲にスケール

            # copy mode用の変数を初期化
            actual_normal_count = 0
            normal_start_index = 0

            # 通常のオブジェクトを先に生成（既存オブジェクトリストが空の状態）
            if num_normal > 0:
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] [ステップ: 通常オブジェクト生成] {num_normal}個を生成開始..", flush=True)

                try:
                    normal_gen_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                    with profile_code_block("generate_objects_from_conditions (normal)", f"attempt_{attempt}_normal"):
                        normal_kwargs = {}
                        if generate_kwargs and use_individual_reference:
                            # copy modeの場合のみgenerate_kwargsを適用
                            normal_kwargs.update(generate_kwargs)

                        normal_kwargs_with_colors = normal_kwargs.copy() if normal_kwargs else {}
                        normal_kwargs_with_colors['object_colors'] = grid_color_list
                        # 2段階生成の場合、全体で決定したarea_ratioを共有
                        if shared_area_ratio is not None:
                            normal_kwargs_with_colors['area_ratio'] = shared_area_ratio
                        normal_objects = generate_objects_from_conditions(
                            background_color=background_color,
                            num_objects=num_normal,
                            existing_objects=[],
                            grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                            total_num_objects=num_objects,
                            **normal_kwargs_with_colors
                        )
                    if ENABLE_DETAILED_TIMING_LOGS and normal_gen_start is not None:
                        normal_gen_elapsed = time.time() - normal_gen_start
                        print(f"[TIMING] 試行{attempt}: 通常オブジェクト生成: {normal_gen_elapsed:.4f}秒 (num_normal={num_normal}, 生成数={len(normal_objects) if normal_objects else 0})", flush=True)
                        if not normal_objects:
                            if best_objects is not None:
                                all_objects = best_objects
                                break
                            continue
                        all_objects.extend(normal_objects)
                        # copy mode用の変数を設定
                        if use_individual_reference:
                            actual_normal_count = len(normal_objects)  # 実際に生成されたオブジェクト数
                            normal_start_index = len(all_objects) - actual_normal_count  # normalオブジェクトの開始インデックス
                except Exception as e:
                    if ENABLE_VERBOSE_LOGGING:
                        print(f"    [エラー] 通常オブジェクト生成でエラーが発生しました: {e}")
                    if best_objects is not None:
                        all_objects = best_objects
                        break
                    continue

            # 条件付きオブジェクトを生成
            if num_conditional > 0:
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] [ステップ: 条件付きオブジェクト生成] {num_conditional}個を生成開始..", flush=True)

                if use_individual_reference:
                    # copy mode: 最適化 - 大量オブジェクト生成時の効率化
                    # 個別生成が非効率なため、normalオブジェクト全体を参照してバッチ生成に変更
                    # ただし、少数のオブジェクトの場合のみ個別生成を使用（動作保証のため）

                    # 最適化: num_conditionalが大きい場合、またはactual_normal_countが多い場合はバッチ処理
                    # 個別生成は最大2回までに制限（パフォーマンス向上）
                    MAX_INDIVIDUAL_GENERATIONS = 2
                    # バッチ処理の適用条件をさらに緩和: num_conditional > 1 または actual_normal_count > 1
                    # バッチ処理の方が効率的であるため、より積極的に適用（num_conditional > 2 → > 1に変更）
                    USE_BATCH_FOR_CONDITIONAL = (num_conditional > 1) or (actual_normal_count > MAX_INDIVIDUAL_GENERATIONS)

                    if USE_BATCH_FOR_CONDITIONAL:
                        # バッチ処理: すべてのnormalオブジェクトを参照して一度に生成
                        conditional_batch_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                        with profile_code_block("generate_objects_from_conditions (conditional)", f"attempt_{attempt}_conditional_batch"):
                            conditional_kwargs = {}
                            if generate_kwargs:
                                conditional_kwargs.update(generate_kwargs)
                            if duplicate_kwargs:
                                conditional_kwargs.update(duplicate_kwargs)

                            # normalオブジェクトのみを参照（全オブジェクトではなく）
                            normal_objects_ref = all_objects[normal_start_index:normal_start_index + actual_normal_count]

                            # 最適化: existing_refsの数を制限（処理時間短縮のため）
                            # 注: 形状シグネチャは必要時に1個ずつ取得するため、サンプリングは不要
                            # ただし、get_object_with_shape_signatureの試行効率化のため、大量の場合は制限
                            MAX_REFERENCE_OBJECTS = 20  # 形状シグネチャ取得の試行回数削減のため
                            if len(normal_objects_ref) > MAX_REFERENCE_OBJECTS:
                                # ランダムにサンプリングして参照数を削減
                                # 注意: randomはファイル先頭でインポート済み
                                sample_size = min(MAX_REFERENCE_OBJECTS, len(normal_objects_ref))
                                if sample_size > 0:
                                    normal_objects_ref = random.sample(normal_objects_ref, sample_size)
                                else:
                                    normal_objects_ref = []

                            conditional_kwargs_with_colors = conditional_kwargs.copy() if conditional_kwargs else {}
                            conditional_kwargs_with_colors['object_colors'] = grid_color_list
                            # 2段階生成の場合、全体で決定したarea_ratioを共有
                            if shared_area_ratio is not None:
                                conditional_kwargs_with_colors['area_ratio'] = shared_area_ratio
                            conditional_objects = generate_objects_from_conditions(
                                background_color=background_color,
                                num_objects=num_conditional,
                                existing_objects=normal_objects_ref,  # normalオブジェクトを参照（制限済み）
                                grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                                total_num_objects=num_objects,
                                **conditional_kwargs_with_colors
                            )
                            if conditional_objects:
                                all_objects.extend(conditional_objects)
                        if ENABLE_DETAILED_TIMING_LOGS and conditional_batch_start is not None:
                            conditional_batch_elapsed = time.time() - conditional_batch_start
                            print(f"[TIMING] 試行{attempt}: 条件付きオブジェクト生成(バッチ): {conditional_batch_elapsed:.4f}秒 (num_conditional={num_conditional}, normal_refs={len(normal_objects_ref)}, 生成数={len(conditional_objects) if conditional_objects else 0})", flush=True)
                    else:
                        # 個別生成: 少数の場合のみ（動作保証のため）
                        conditional_individual_total_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                        for i in range(min(actual_normal_count, MAX_INDIVIDUAL_GENERATIONS)):
                            try:
                                conditional_individual_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                                with profile_code_block("generate_objects_from_conditions (conditional)", f"attempt_{attempt}_conditional_{i}"):
                                    conditional_kwargs = {}
                                    if generate_kwargs:
                                        conditional_kwargs.update(generate_kwargs)
                                    if duplicate_kwargs and random.random() < 0.9:
                                        conditional_kwargs.update(duplicate_kwargs)
                                    else:
                                        rand = random.random()
                                        if rand < 0.33:
                                            conditional_kwargs['duplicate_mode'] = 'shape_only'
                                        elif rand < 0.67:
                                            conditional_kwargs['duplicate_mode'] = 'exact'
                                        else:
                                            conditional_kwargs['duplicate_mode'] = None

                                    conditional_kwargs['duplicate_num'] = 1
                                    obj_index = normal_start_index + i
                                    if 0 <= obj_index < len(all_objects):
                                        conditional_kwargs_with_colors = conditional_kwargs.copy() if conditional_kwargs else {}
                                        conditional_kwargs_with_colors['object_colors'] = grid_color_list
                                        # 2段階生成の場合、全体で決定したarea_ratioを共有
                                        if shared_area_ratio is not None:
                                            conditional_kwargs_with_colors['area_ratio'] = shared_area_ratio
                                        conditional_objects = generate_objects_from_conditions(
                                            background_color=background_color,
                                            num_objects=num_conditional,
                                            existing_objects=[all_objects[obj_index]],  # 個別参照（1個のみ）
                                            grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                                            total_num_objects=num_objects,
                                            **conditional_kwargs_with_colors
                                        )
                                        if conditional_objects:
                                            all_objects.extend(conditional_objects)
                                if ENABLE_DETAILED_TIMING_LOGS and conditional_individual_start is not None:
                                    conditional_individual_elapsed = time.time() - conditional_individual_start
                                    print(f"[TIMING] 試行{attempt}: 条件付きオブジェクト生成(個別{i}): {conditional_individual_elapsed:.4f}秒 (num_conditional={num_conditional}, 生成数={len(conditional_objects) if conditional_objects else 0})", flush=True)
                            except Exception as e:
                                if ENABLE_VERBOSE_LOGGING:
                                    print(f"    [エラー] 条件付きオブジェクト生成（copy, i={i}）でエラーが発生しました: {e}")
                                continue
                        if ENABLE_DETAILED_TIMING_LOGS and conditional_individual_total_start is not None:
                            conditional_individual_total_elapsed = time.time() - conditional_individual_total_start
                            print(f"[TIMING] 試行{attempt}: 条件付きオブジェクト生成(個別合計): {conditional_individual_total_elapsed:.4f}秒 (繰り返し数={min(actual_normal_count, MAX_INDIVIDUAL_GENERATIONS)})", flush=True)
                else:
                    # normal mode: 全既存オブジェクトを参照して生成
                    conditional_normal_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                    with profile_code_block("generate_objects_from_conditions (conditional)", f"attempt_{attempt}_conditional"):
                        conditional_kwargs = {}
                        if generate_kwargs:
                            conditional_kwargs.update(generate_kwargs)
                        if duplicate_kwargs:
                            conditional_kwargs.update(duplicate_kwargs)

                        conditional_kwargs_with_colors = conditional_kwargs.copy() if conditional_kwargs else {}
                        conditional_kwargs_with_colors['object_colors'] = grid_color_list
                        # 2段階生成の場合、全体で決定したarea_ratioを共有
                        if shared_area_ratio is not None:
                            conditional_kwargs_with_colors['area_ratio'] = shared_area_ratio

                        # 最適化: existing_refsの数を制限（処理時間短縮のため）
                        # 注: 形状シグネチャは必要時に1個ずつ取得するため、サンプリングは不要
                        # ただし、get_object_with_shape_signatureの試行効率化のため、大量の場合は制限
                        MAX_REFERENCE_OBJECTS = 20  # 形状シグネチャ取得の試行回数削減のため
                        existing_objects_ref = all_objects
                        if len(all_objects) > MAX_REFERENCE_OBJECTS:
                            # ランダムにサンプリングして参照数を削減
                            # 注意: randomはファイル先頭でインポート済み
                            sample_size = min(MAX_REFERENCE_OBJECTS, len(all_objects))
                            if sample_size > 0:
                                existing_objects_ref = random.sample(all_objects, sample_size)
                            else:
                                existing_objects_ref = []

                        conditional_objects = generate_objects_from_conditions(
                            background_color=background_color,
                            num_objects=num_conditional,
                            existing_objects=existing_objects_ref,  # 全既存参照（制限済み）
                            grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                            total_num_objects=num_objects,
                            **conditional_kwargs_with_colors
                        )
                        if conditional_objects:
                            all_objects.extend(conditional_objects)
                    if ENABLE_DETAILED_TIMING_LOGS and conditional_normal_start is not None:
                        conditional_normal_elapsed = time.time() - conditional_normal_start
                        print(f"[TIMING] 試行{attempt}: 条件付きオブジェクト生成(normal mode): {conditional_normal_elapsed:.4f}秒 (num_conditional={num_conditional}, existing_refs={len(all_objects)}, 生成数={len(conditional_objects) if conditional_objects else 0})", flush=True)
        else:
            # プログラムなしでforce_modeも指定されていない場合: 通常通り全オブジェクトを一括生成
            # 一括生成の方が、object_colorsが均等に使用される可能性が高い
            if ENABLE_VERBOSE_LOGGING:
                print(f"  [試行{attempt}] [ステップ: 一括生成] {num_objects}個を生成開始..", flush=True)

            # 詳細ログ: 一括生成で渡されるパラメータ
            if ENABLE_COLOR_INVESTIGATION_LOGS:
                print(f"[色数調査] 試行{attempt}: 一括生成開始 - background_color={background_color}, num_objects={num_objects}, object_colors={grid_color_list}, len(object_colors)={len(grid_color_list)}", flush=True)

            batch_gen_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
            with profile_code_block("generate_objects_from_conditions (batch)", f"attempt_{attempt}_batch"):
                batch_kwargs = {
                    'object_colors': grid_color_list,
                    'duplicate_mode': None,  # ARC-AGI2統計に基づいた複製ロジックを有効化（selected_color優先は既に実装済み）
                }
                all_objects = generate_objects_from_conditions(
                    background_color=background_color,
                    num_objects=num_objects,
                    existing_objects=[],
                    grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                    total_num_objects=num_objects,
                    **batch_kwargs
                )
            if ENABLE_DETAILED_TIMING_LOGS and batch_gen_start is not None:
                batch_gen_elapsed = time.time() - batch_gen_start
                print(f"[TIMING] 試行{attempt}: 一括オブジェクト生成: {batch_gen_elapsed:.4f}秒 (num_objects={num_objects}, 生成数={len(all_objects) if all_objects else 0})", flush=True)

            # 詳細ログ: 生成されたオブジェクトの色
            # Noneを除外して有効なオブジェクトのみを取得
            valid_objects = [obj for obj in all_objects if obj is not None] if all_objects else []

            if ENABLE_COLOR_INVESTIGATION_LOGS:
                if valid_objects:
                    actual_colors = [obj.get('color') for obj in valid_objects if obj.get('color') is not None]
                    unique_colors = sorted(set(actual_colors))
                    print(f"[色数調査] 試行{attempt}: 一括生成完了 - 生成オブジェクト数={len(all_objects)}, 有効オブジェクト数={len(valid_objects)}, None数={len(all_objects) - len(valid_objects)}, 使用色={unique_colors}, 使用色数={len(unique_colors)}, オブジェクト色一覧={actual_colors}", flush=True)
                else:
                    print(f"[色数調査] 試行{attempt}: 一括生成完了 - 生成オブジェクト数={len(all_objects)}, 有効オブジェクト数=0, None数={len(all_objects) if all_objects else 0}", flush=True)

            # Noneを除外してall_objectsを更新
            all_objects = valid_objects

            if not all_objects:
                if best_objects is not None:
                    all_objects = best_objects
                    break
                continue

        # オブジェクト生成完了後の共通処理
        if _log_enabled:
            print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: オブジェクト生成完了 (len={len(all_objects) if all_objects else 0})", flush=True)

        # 特定コマンドの場合、ノイズパターンのオブジェクトを追加
        noise_object_commands = ['CROP', 'EXTRACT_RECTS', 'EXTRACT_LINES', 'EXTRACT_HOLLOW_RECTS']
        has_noise_command = any(cmd in all_commands for cmd in noise_object_commands)

        if has_noise_command:
            add_noise_probability = 0.5  # 50%
        else:
            add_noise_probability = 0.01  # 1%（非常に低確率）

        if random.random() < add_noise_probability:
            # ノイズパターンのオブジェクト数を乱数で決定（1-3個）
            num_noise_objects = random.randint(1, 3)
            with profile_code_block("generate_objects_from_conditions (noise)", f"attempt_{attempt}_noise"):
                noise_objects = generate_objects_from_conditions(
                    background_color=background_color,
                    num_objects=num_noise_objects,
                    existing_objects=all_objects,
                    shape_type='noise',
                    grid_size=(grid_width, grid_height) if grid_width is not None and grid_height is not None else None,
                    total_num_objects=num_noise_objects,  # 面積ベースの制約を適用するため、total_num_objectsを指定（ノイズオブジェクトのみなので、num_noise_objectsを使用）
                    object_colors=grid_color_list
                )
                if noise_objects:
                    from src.data_systems.generator.input_grid_generator.managers.grid_builder import set_position
                    noise_objects = set_position(
                        width=grid_width,
                        height=grid_height,
                        background_color=background_color,
                        objects=noise_objects,
                        existing_objects=all_objects
                    )
                    all_objects.extend(noise_objects)

        # all_objectsが空の場合はチェック（オブジェクト生成に失敗した場合）
        # 注意: この時点でall_objectsが空の場合は、オブジェクト生成に失敗している
        # 次の試行に進むか、best_objectsを使用する
        if not all_objects or len(all_objects) == 0:
            if best_objects is not None:
                all_objects = best_objects
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] all_objectsが空のため、best_objectsを使用します (len={len(best_objects)})", flush=True)
                break
            if ENABLE_VERBOSE_LOGGING:
                print(f"  [試行{attempt}] all_objectsが空のため、次の試行に進みます", flush=True)
            continue

        min_spacing = 0  # generate_kwargsから取得する場合は: generate_kwargs.get('min_spacing', 0) if generate_kwargs else 0

        # グリッドを構築
        if ENABLE_VERBOSE_LOGGING:
            print(f"  [試行{attempt}] [ステップ: グリッド構築] 開始..", flush=True)

        try:
            build_grid_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
            with profile_code_block("build_grid", f"attempt_{attempt}"):
                temp_grid_list = build_grid(
                    width=grid_width,
                    height=grid_height,
                    background_grid_pattern=background_grid_pattern,
                    objects=all_objects,
                    min_spacing=min_spacing
                )
                temp_input_grid = np.array(temp_grid_list, dtype=int)
            if ENABLE_DETAILED_TIMING_LOGS and build_grid_start is not None:
                build_grid_elapsed = time.time() - build_grid_start
                print(f"[TIMING] 試行{attempt}: グリッド構築: {build_grid_elapsed:.4f}秒 (grid_size={grid_width}x{grid_height}, objects={len(all_objects)})", flush=True)
        except Exception as e:
            if ENABLE_VERBOSE_LOGGING:
                import traceback
                traceback.print_exc()
            raise

        if ENABLE_VERBOSE_LOGGING:
            print(f"  [試行{attempt}] [ステップ完了] グリッド構築 (サイズ={temp_input_grid.shape})", flush=True)

        # タイムアウトチェック: 各試行の実行時間が上限を超えた場合
        attempt_elapsed = time.time() - attempt_start_time
        if attempt_elapsed > max_attempt_time:
            if best_objects is not None:
                all_objects = best_objects
                break
            continue

        # ループ全体のタイムアウトチェック（プログラム実行前）
        # elapsed_from_loop_startは既に計算済みなので、再利用
        elapsed_from_loop_start_before_exec = elapsed_from_loop_start
        if elapsed_from_loop_start_before_exec > MAX_LOOP_TIME:
            if ENABLE_VERBOSE_LOGGING:
                print(f"[タイムアウト] validate_nodes_and_adjust_objectsループが{MAX_LOOP_TIME}秒を超えました。ループを終了します。", flush=True)
            if best_objects is not None:
                all_objects = best_objects
                break
            # タイムアウトでタスクを破棄
            return None, None, None, None, None

        # プログラムがない場合は、プログラム実行と条件チェックをスキップして簡易版の条件チェックを使用
        # ただし、program_codeが提供されている場合はプログラムありとして扱う
        # nodes is None かつ program_code is None の場合のみプログラムなしモード
        if nodes is None and program_code is None:
            # 入力グリッド検証（プログラムがない場合の条件チェック）
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 入力グリッド検証開始", flush=True)

            condition_check_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None

            input_unique_colors = np.unique(temp_input_grid)
            condition3 = len(input_unique_colors) == 1  # 入力が単色

            # 詳細ログ: グリッド構築後の色
            if ENABLE_COLOR_INVESTIGATION_LOGS:
                print(f"[色数調査] 試行{attempt}: グリッド構築後 - 背景色={background_color}, グリッド内の色={sorted(input_unique_colors.tolist())}, 色数={len(input_unique_colors)}, 決定色数={target_color_count}, 選択色リスト={selected_colors}", flush=True)

            # オブジェクトピクセル数を計算（condition4は削除：ARC-AGI2データセットに存在する有効なタスク（5.29%）を生成できるようにするため）
            object_pixels = np.sum(temp_input_grid != background_color)
            total_pixels = grid_width * grid_height
            # condition4は削除（condition3とcondition5で十分カバーできているため）

            # 入力グリッドが空でないか（オブジェクトが存在するか）
            condition5 = object_pixels == 0  # 空の入力（オブジェクトなし）

            # プログラムがない場合の条件チェック（condition1, condition2, condition4, condition6は使用しない）
            condition1 = False
            condition2 = False
            condition4 = False  # 削除：ARC-AGI2に存在する有効なタスク（3%未満のオブジェクト）を生成可能にする
            condition6 = False

            # ループ継続判定（プログラムがない場合）- condition4は削除
            should_continue = condition3 or condition5

            if ENABLE_DETAILED_TIMING_LOGS and condition_check_start is not None:
                condition_check_elapsed = time.time() - condition_check_start
                print(f"[TIMING] 試行{attempt}: 入力グリッド検証(プログラムなし): {condition_check_elapsed:.4f}秒 (should_continue={should_continue})", flush=True)

            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 入力グリッド検証完了 (③入力が単色={condition3}, ⑤空の入力={condition5})", flush=True)

            if ENABLE_VERBOSE_LOGGING:
                print(f"  [試行{attempt}] [ステップ完了] 条件チェック (③入力が単色={condition3}, ⑤空の入力={condition5})", flush=True)

            # プログラムがない場合の条件を設定（condition1, condition2, condition6は使用しない）
            condition1 = False
            condition2 = False
            condition6 = False
        else:
            # プログラムを実行
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: プログラム実行開始", flush=True)
            exec_start = time.time()
            try:
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] [ステップ: プログラム実行] 開始..", flush=True)

                # プログラム実行のタイムアウトチェック（実行時間がmax_attempt_time * 0.8を超える場合は中断）
                if not DISABLE_TIMEOUTS and elapsed_from_loop_start_before_exec > max_attempt_time * 0.8:
                    # タイムアウトは重要な情報なので常に出力
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: タイムアウトによりスキップ (経過時間: {elapsed_from_loop_start_before_exec:.2f}秒)", flush=True)
                    if _log_enabled:
                        print(f"  [タイムアウト] プログラム実行をスキップ（ループ経過時間が長すぎます: {elapsed_from_loop_start_before_exec:.2f}秒）", flush=True)
                    if best_objects is not None:
                        all_objects = best_objects
                        break
                    continue

                if _log_enabled:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: executor.execute_program_string呼び出し前", flush=True)
                exec_program_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
                with profile_code_block("execute_program_string", f"attempt_{attempt}"):
                    output_grid, _, execution_time, _ = executor.execute_program_string(
                        program_code=program_code,
                        input_grid=temp_input_grid,
                        input_objects=None,  # オブジェクト自動抽出
                        input_image_index=0,
                        background_color=None  # 本番用の背景色推論を使用
                    )
                if ENABLE_DETAILED_TIMING_LOGS and exec_program_start is not None:
                    exec_program_elapsed = time.time() - exec_program_start
                    print(f"[TIMING] 試行{attempt}: プログラム実行: {exec_program_elapsed:.4f}秒 (execution_time={execution_time}, grid_size={temp_input_grid.shape})", flush=True)
                if _log_enabled:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: executor.execute_program_string呼び出し完了", flush=True)

                exec_elapsed = time.time() - exec_start
                if _log_enabled:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 実行時間計算完了 (exec_elapsed={exec_elapsed:.3f}秒, execution_time={execution_time})", flush=True)
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: output_gridチェック開始 (output_grid is None={output_grid is None})", flush=True)
                if output_grid is None:
                    # Noneチェックは重要な情報なので常に出力
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: output_gridがNoneのためcontinue", flush=True)
                    if _log_enabled:
                        print(f"  [試行{attempt}] [ステップ失敗] プログラム実行エラー: output_gridがNone", flush=True)
                    continue

                # エラー検出: execution_timeが0.0の場合は、エラーで入力グリッドが返された可能性がある
                # ただし、入力グリッドと出力グリッドが完全一致している場合は、エラーとして扱う
                if execution_time == 0.0:
                    if np.array_equal(temp_input_grid, output_grid):
                        error_msg = f"プログラム実行エラー: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致しています（エラーで入力グリッドが返された可能性があります）"
                        # エラーログは常に出力（重要な情報）
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: {error_msg}", flush=True)
                        # SilentExceptionを投げてタスクを廃棄
                        from src.core_systems.executor.core import SilentException
                        raise SilentException(error_msg)

                if _log_enabled:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: output_gridチェック完了 (shape={output_grid.shape if hasattr(output_grid, 'shape') else 'N/A'})", flush=True)
                if ENABLE_VERBOSE_LOGGING:
                    print(f"  [試行{attempt}] [ステップ完了] プログラム実行 {exec_elapsed:.3f}秒 (出力サイズ={output_grid.shape})", flush=True)

            except (TimeoutError, Exception) as timeout_e:
                # SilentExceptionは関数の先頭でインポート済み
                # UnboundLocalErrorを防ぐため、グローバルスコープから明示的に参照
                try:
                    # SilentExceptionが利用可能か確認
                    from src.core_systems.executor.core import SilentException as SE
                    is_silent_exception = isinstance(timeout_e, SE)
                except (ImportError, NameError):
                    # SilentExceptionが利用できない場合は、型名で判定
                    is_silent_exception = type(timeout_e).__name__ == 'SilentException'

                # SilentExceptionの場合は即座にタスクを破棄
                if is_silent_exception:
                    if is_first_pair:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: SilentException発生によりタスクを破棄: {timeout_e}", flush=True)
                        return None, None, None, None, None
                    else:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: SilentException発生によりペアをスキップ: {timeout_e}", flush=True)
                        raise ValueError(f"ペアスキップ: SilentException発生 - {timeout_e}")

                is_timeout = isinstance(timeout_e, TimeoutError) or (is_silent_exception and "タイムアウト" in str(timeout_e))

                if is_timeout:
                    # タイムアウトエラー: 最初のペアで発生した場合はタスク廃棄、最初以外のペアで発生した場合はペアスキップ
                    if is_first_pair:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: タイムアウト発生によりタスクを破棄", flush=True)
                        return None, None, None, None, None
                    else:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: タイムアウト発生によりペアをスキップ", flush=True)
                        raise ValueError(f"ペアスキップ: タイムアウト発生")
                raise
            except Exception as e:
                # プログラム実行エラーが1回発生したら即座にタスクを破棄
                # SilentExceptionの可能性もチェック（関数の先頭でインポート済み）
                # UnboundLocalErrorを防ぐため、グローバルスコープから明示的に参照
                try:
                    # SilentExceptionが利用可能か確認
                    from src.core_systems.executor.core import SilentException as SE
                    is_silent_exception = isinstance(e, SE)
                except (ImportError, NameError):
                    # SilentExceptionが利用できない場合は、型名で判定
                    is_silent_exception = type(e).__name__ == 'SilentException'

                if is_silent_exception:
                    if is_first_pair:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: SilentException発生によりタスクを破棄: {e}", flush=True)
                        return None, None, None, None, None
                    else:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: SilentException発生によりペアをスキップ: {e}", flush=True)
                        raise ValueError(f"ペアスキップ: SilentException発生 - {e}")
                # 例外の詳細をログに記録
                exception_type = type(e).__name__
                exception_message = str(e)
                # 最初のペアで発生した場合はタスク廃棄、最初以外のペアで発生した場合はペアスキップ
                if is_first_pair:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 例外発生によりタスクを破棄 - 例外型: {exception_type}, メッセージ: {exception_message}", flush=True)
                    return None, None, None, None, None
                else:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 例外発生によりペアをスキップ - 例外型: {exception_type}, メッセージ: {exception_message}", flush=True)
                    raise ValueError(f"ペアスキップ: 例外発生 - {exception_type}: {exception_message}")

            # 条件をチェック（最適化: 早期終了を有効化）
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 条件チェック開始 (temp_input_grid shape={temp_input_grid.shape if hasattr(temp_input_grid, 'shape') else 'N/A'}, output_grid shape={output_grid.shape if hasattr(output_grid, 'shape') else 'N/A'})", flush=True)
            condition_check_start = time.time() if ENABLE_DETAILED_TIMING_LOGS else None
            condition1, condition2, condition3, condition4, condition5, condition6 = check_output_conditions(temp_input_grid, output_grid, background_color=background_color, early_exit=True)
            if ENABLE_DETAILED_TIMING_LOGS and condition_check_start is not None:
                condition_check_elapsed = time.time() - condition_check_start
                print(f"[TIMING] 試行{attempt}: 条件チェック: {condition_check_elapsed:.4f}秒", flush=True)
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 条件チェック完了 (①入力一致={condition1}, ②トリミング一致={condition2}, ③入力が単色={condition3}, ④オブジェクトピクセル数5%未満={condition4}, ⑤空のoutput={condition5}, ⑥出力が単色={condition6})", flush=True)

            if ENABLE_VERBOSE_LOGGING:
                print(f"  [試行{attempt}] [ステップ完了] 条件チェック (①入力一致={condition1}, ②トリミング一致={condition2}, ③入力が単色={condition3}, ④オブジェクトピクセル数5%未満={condition4}, ⑤空のoutput={condition5}, ⑥出力が単色={condition6})", flush=True)

            # ループを継続すべきかを判定（プログラムありの場合）
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: ループ継続判定開始", flush=True)
            should_continue = should_continue_loop(condition1, condition2, condition3, condition4, condition5, condition6)

            if ENABLE_DETAILED_TIMING_LOGS and condition_check_start is not None:
                condition_check_total_elapsed = time.time() - condition_check_start
                print(f"[TIMING] 試行{attempt}: 条件チェック+判定: {condition_check_total_elapsed:.4f}秒 (should_continue={should_continue})", flush=True)
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: ループ継続判定完了 (should_continue={should_continue})", flush=True)

        # ループ継続判定の処理（プログラムあり/なしの共通処理）
        if should_continue:
            consecutive_continue_count += 1
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: ループ継続 (consecutive_continue_count={consecutive_continue_count})", flush=True)

            # 連続して条件に該当し続ける場合、早期にタスクを破棄（パフォーマンス向上のため）
            if consecutive_continue_count >= max_consecutive_continues:
                # デフォルトで無効化（詳細ログ）
                if _log_enabled:
                    print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 連続continue上限に達しました (best_objects is None={best_objects is None}, all_objects len={len(all_objects)})", flush=True)
                if best_objects is None:
                    # best_objectsがない場合、タスク/ペアを破棄（条件に該当するall_objectsを使用しない）
                    if is_first_pair:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 連続continue上限に達し、best_objectsなしによりタスクを破棄", flush=True)
                        return None, None, None, None, None
                    else:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: 連続continue上限に達し、best_objectsなしによりペアをスキップ", flush=True)
                        raise ValueError("ペアスキップ: 連続continue上限に達し、best_objectsなし")
                else:
                    # best_objectsがある場合、それを使用してループを終了
                    all_objects = best_objects
                    if _log_enabled:
                        print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: break (best_objects使用)", flush=True)
                    break

            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: continue (次のループへ)", flush=True)
            continue  # 次のループへ
        else:
            consecutive_continue_count = 0  # 条件に該当しなかった場合はリセット
            # 条件に該当しないオブジェクトセットを保持（最良の結果として）
            best_objects = all_objects.copy() if all_objects else None
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] 試行{attempt}: ループ終了 (best_objects保存完了, len={len(best_objects) if best_objects else 0})", flush=True)
            break  # ループ終了

    # ループ終了後の処理（最大試行回数に達した場合）
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] ループ終了後の処理開始 (attempt={attempt}, max_attempts={max_attempts})", flush=True)
    if attempt >= max_attempts:
        # 最大試行回数到達は重要な情報なので常に出力
        print(f"[validate_nodes_and_adjust_objects] 最大試行回数に達しました", flush=True)
        if _log_enabled:
            print(f"[警告] 最大試行回数({max_attempts})に達しました")
        # best_objectsがあれば使用、なければタスク/ペアを破棄（条件に該当する可能性があるall_objectsを使用しない）
        if best_objects is not None:
            all_objects = best_objects
            if _log_enabled:
                print(f"[validate_nodes_and_adjust_objects] best_objectsを使用 (オブジェクト数={len(all_objects)})", flush=True)
        else:
            # best_objectsがない場合、タスク/ペアを破棄（より保守的に）
            if is_first_pair:
                print(f"[validate_nodes_and_adjust_objects] 最大試行回数に達し、best_objectsなしによりタスクを破棄", flush=True)
                return None, None, None, None, None
            else:
                print(f"[validate_nodes_and_adjust_objects] 最大試行回数に達し、best_objectsなしによりペアをスキップ", flush=True)
                raise ValueError("ペアスキップ: 最大試行回数に達し、best_objectsなし")

    # 3. ループ終了後、置き換えのループ（パフォーマンス最適化：スキップ可能）
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] 置き換え検証チェック開始 (ENABLE_REPLACEMENT_VERIFICATION={ENABLE_REPLACEMENT_VERIFICATION}, enable_replacement={enable_replacement})", flush=True)
    # nodes is Noneでもprogram_codeが提供されている場合は、コマンド置き換え処理をスキップ（nodesが必要なため）
    if not ENABLE_REPLACEMENT_VERIFICATION or not enable_replacement or nodes is None:
        if _log_enabled:
            skip_reason = "ENABLE_REPLACEMENT_VERIFICATION=False" if not ENABLE_REPLACEMENT_VERIFICATION else \
                         "enable_replacement=False" if not enable_replacement else \
                         "nodes is None (program_codeから生成されたため)"
            print(f"[validate_nodes_and_adjust_objects] 置き換えループをスキップ ({skip_reason})", flush=True)
            print(f"[置き換えループ] スキップ（{skip_reason}）")
            print(f"[validate_nodes_and_adjust_objects] 関数終了（置き換えループスキップ）", flush=True)
        return nodes, all_objects, if_count, for_count, end_count

    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] 置き換えループ開始", flush=True)
        print(f"\n[置き換えループ開始]")

    # 置き換えループでもgenerate_kwargsを使用するため、再度取得
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] 置き換えループ: kwargs取得開始", flush=True)
    duplicate_kwargs_for_replacement = get_duplicate_mode_kwargs_from_commands(all_commands)
    generate_kwargs_for_replacement = get_generate_objects_kwargs_from_commands(all_commands)
    generate_kwargs_for_replacement.update(duplicate_kwargs_for_replacement)
    if _log_enabled:
        print(f"[validate_nodes_and_adjust_objects] 置き換えループ: kwargs取得完了", flush=True)

    # 置き換え対象のコマンド情報を収集
    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
        print(f"[validate_nodes_and_adjust_objects] 置き換えループ: コマンド情報収集開始 (nodes数={len(nodes) if nodes else 0})", flush=True)
    commands_to_check = []
    for i, node in enumerate(nodes):
        if i % 5 == 0 and (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS):  # 5ノードごとに進捗ログ
            print(f"[validate_nodes_and_adjust_objects] 置き換えループ: ノード{i}/{len(nodes)}処理中", flush=True)

        # InitializationNodeの場合はスキップ
        node_type_name = type(node).__name__
        if node_type_name == 'InitializationNode':
            continue
        # 代入式でない場合はスキップ
        if not is_assignment_node(node):
            continue

        # 一番浅いネストのコマンドを取得
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[validate_nodes_and_adjust_objects] 置き換えループ: ノード{i} - get_commands_sorted_by_depth呼び出し前", flush=True)
        commands_with_depth = get_commands_sorted_by_depth(node, reverse=False)
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[validate_nodes_and_adjust_objects] 置き換えループ: ノード{i} - get_commands_sorted_by_depth呼び出し完了 (commands数={len(commands_with_depth) if commands_with_depth else 0})", flush=True)
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
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[validate_nodes_and_adjust_objects] 置き換えループ: ノード{i} - コマンド '{cmd_name}' を追加", flush=True)

    # 検証対象コマンド数を制限（パフォーマンス最適化）
    if MAX_REPLACEMENT_COMMANDS > 0 and len(commands_to_check) > MAX_REPLACEMENT_COMMANDS:
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[置き換えループ] 検証対象コマンド数を{MAX_REPLACEMENT_COMMANDS}個に制限")
        commands_to_check = commands_to_check[:MAX_REPLACEMENT_COMMANDS]
    else:
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[置き換えループ] 検証対象コマンド数: {len(commands_to_check)}")

    # 各コマンドを検証して置き換え
    for cmd_info in commands_to_check:
        node_index = cmd_info['node_index']
        cmd_name = cmd_info['command']
        cmd_node = cmd_info['node']

        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"\n[置き換え検証] コマンド: {cmd_name} (Node {node_index})")

        # 現在のオブジェクトとグリッドを使用（ループ終了後の状態）
        min_spacing_for_replacement = 0

        temp_grid_list = build_grid(
            width=grid_width,
            height=grid_height,
            background_grid_pattern=background_grid_pattern,
            objects=all_objects,
            min_spacing=min_spacing_for_replacement
        )
        temp_input_grid = np.array(temp_grid_list, dtype=int)

        # 1. 元のコードでoutput_grid取得
        try:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [実行] 元のコードを実行中..")

            with profile_code_block("execute_program_string (original)", f"cmd_{cmd_name}_original"):
                output_grid_original, _, _, _ = executor.execute_program_string(
                    program_code=program_code,
                    input_grid=temp_input_grid,
                    input_objects=None,  # オブジェクト自動抽出
                    input_image_index=0,
                    background_color=None  # 本番用の背景色推論を使用
                )

            if output_grid_original is None:
                if ENABLE_ALL_LOGS:
                    print(f"  [実行 エラー] output_gridがNone")
                continue

            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [実行] 完了 サイズ={output_grid_original.shape}")

        except Exception as e:
            if ENABLE_ALL_LOGS:
                print(f"  [実行 エラー] {e}")
                import traceback
                traceback.print_exc()
            continue

        # 2. 置き換え後のコードでoutput_grid取得
        nodes_after_current = nodes[node_index+1:]
        replace_result = replace_command_with_fallback(
            cmd_name, cmd_node, nodes[:node_index+1],
            grid_width=grid_width, grid_height=grid_height
        )

        if replace_result is None:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [置き換え] スキップ（置き換え候補なし）")
            continue

        nodes_replaced_up_to_current, fallback_code = replace_result
        nodes_with_fallback = nodes_replaced_up_to_current + nodes_after_current

        # 置き換え後のプログラムコードを生成
        try:
            if executor.program_generator is None:
                if ENABLE_ALL_LOGS:
                    print(f"  [実行 エラー] executor.program_generator is None")
                continue
            context_fallback = ProgramContext(complexity=1)
            try:
                program_code_fallback = executor.program_generator._generate_code(
                    nodes_with_fallback, context_fallback, preserve_indentation=True
                )
            except TypeError:
                program_code_fallback = executor.program_generator._generate_code(
                    nodes_with_fallback, context_fallback
                )
        except Exception as e:
            if ENABLE_ALL_LOGS:
                print(f"  [実行 エラー] コード生成失敗 {e}")
            continue

        try:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [実行] 置き換え後のコードを実行中..")

            with profile_code_block("execute_program_string (fallback)", f"cmd_{cmd_name}_fallback"):
                output_grid_fallback, _, _, _ = executor.execute_program_string(
                    program_code=program_code_fallback,
                    input_grid=temp_input_grid,
                    input_objects=None,  # オブジェクト自動抽出
                    input_image_index=0,
                    background_color=None  # 本番用の背景色推論を使用
                )

            if output_grid_fallback is None:
                if ENABLE_ALL_LOGS:
                    print(f"  [実行 エラー] output_gridがNone")
                continue

            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [実行] 完了 サイズ={output_grid_fallback.shape}")

        except Exception as e:
            if ENABLE_ALL_LOGS:
                print(f"  [実行 エラー] {e}")
                import traceback
                traceback.print_exc()
            continue

        # 3. 置き換え前のoutput_gridが条件に該当するかチェック
        condition1_original, condition2_original, condition3_original, condition4_original, condition5_original, condition6_original = check_output_conditions(
            temp_input_grid, output_grid_original, background_color=background_color
        )
        original_meets_condition = condition1_original or condition2_original or condition3_original or condition5_original or condition6_original

        # 4. 置き換え前後のoutput_gridが完全一致かチェック
        outputs_match = np.array_equal(output_grid_original, output_grid_fallback)

        # original_meets_conditionがFalseの場合、置き換え前後の完全一致が必要
        if not original_meets_condition:
            if not outputs_match:
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [判定] 置き換え前のoutput_gridが条件に該当せず、かつ置き換え前後のoutput_gridが一致しない→置き換えスキップ")
                    print(f"  [詳細] 条件①入力一致={condition1_original}, 条件②重なり一致={condition2_original}, 条件③入力が単色={condition3_original}, 条件④オブジェクトピクセル数5%未満={condition4_original}, 条件⑤空のoutput={condition5_original}, 条件⑥出力が単色={condition6_original}")
                    print(f"  [詳細] 置き換え前後のoutput_grid一致={outputs_match}")
                continue
            else:
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [詳細] 条件①入力一致={condition1_original}, 条件②重なり一致={condition2_original}, 条件③入力が単色={condition3_original}, 条件④オブジェクトピクセル数5%未満={condition4_original}, 条件⑤空のoutput={condition5_original}, 条件⑥出力が単色={condition6_original}")
                    print(f"  [詳細] 置き換え前後のoutput_grid一致={outputs_match}")
        else:
            # original_meets_conditionがTrueなら、置き換え前後が一致しなくても置き換えを実行
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [詳細] 条件①入力一致={condition1_original}, 条件②重なり一致={condition2_original}, 条件③入力が単色={condition3_original}, 条件④オブジェクトピクセル数5%未満={condition4_original}, 条件⑤空のoutput={condition5_original}, 条件⑥出力が単色={condition6_original}")
                print(f"  [詳細] 置き換え前後のoutput_grid一致={outputs_match}")

        # 置き換えを適用
        nodes_after_current = nodes[node_index+1:]
        replace_result = replace_command_with_fallback(
            cmd_name, cmd_node, nodes[:node_index+1],
            grid_width=grid_width, grid_height=grid_height
        )

        if replace_result is not None:
            nodes_replaced_up_to_current, fallback_code = replace_result
            nodes_with_fallback = nodes_replaced_up_to_current + nodes_after_current

            # 更新後のプログラムコードを再生成
            try:
                if executor.program_generator is None:
                    if ENABLE_ALL_LOGS:
                        print(f"  [警告] executor.program_generator is None - コードの生成をスキップ")
                    continue
                else:
                    context = ProgramContext(complexity=1)
                    try:
                        program_code_after_replace = executor.program_generator._generate_code(
                            nodes_with_fallback, context, preserve_indentation=True
                        )
                    except TypeError:
                        program_code_after_replace = executor.program_generator._generate_code(nodes_with_fallback, context)
            except Exception as e:
                if ENABLE_ALL_LOGS:
                    print(f"  [警告] コードの生成失敗 {e}")
                continue

            # 置き換え後のプログラムにGET_ALL_OBJECTSが含まれているか確認
            if 'GET_ALL_OBJECTS' not in program_code_after_replace:
                error_msg = f"フォールバック置換後のプログラムにGET_ALL_OBJECTSが含まれていません"
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [検証失敗] {error_msg}")
                # 置き換えをスキップ
                continue

            # 置き換え後のプログラムを再実行して、エラーが発生しないことを確認
            # 実際の入力グリッド（temp_input_grid）を使用して、未定義変数エラーを検出
            try:
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [検証] 置き換え後のコードを再実行して検証中..")

                # 実際の入力グリッドを使用（空のグリッドでは未定義変数エラーが検出できないため）
                # temp_input_gridは既に構築されている（上記1053-1060行目で構築済み）

                with profile_code_block("execute_program_string (verification)", f"cmd_{cmd_name}_verification"):
                    test_output_grid, _, test_execution_time, _ = executor.execute_program_string(
                        program_code=program_code_after_replace,
                        input_grid=temp_input_grid,  # 実際の入力グリッドを使用
                        input_objects=None,  # オブジェクト自動抽出
                        input_image_index=0,
                        background_color=None  # 本番用の背景色推論を使用
                    )

                # エラー検出: execution_timeが0.0で、入力グリッドと出力グリッドが完全一致している場合はエラー
                if test_execution_time == 0.0 and np.array_equal(temp_input_grid, test_output_grid):
                    error_msg = f"フォールバック置換後のプログラムがエラーを発生させました（execution_time=0.0、入力と出力が一致）"
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [検証失敗] {error_msg}")
                    # 置き換えをスキップ
                    continue

                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"  [検証成功] 置き換え後のプログラムは正常に実行されました")

            except Exception as e:
                # 再実行でエラーが発生した場合は、置き換えをスキップ
                # 未定義変数エラーやその他の実行エラーを検出
                error_type = type(e).__name__
                error_message = str(e)

                # SilentExceptionの場合は、内部で既にログ出力されているため、簡潔にログ出力
                from src.core_systems.executor.core import SilentException
                if isinstance(e, SilentException) or error_type == 'SilentException':
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [検証失敗] 置き換え後のプログラムの再実行でSilentExceptionが発生しました（未定義変数エラーなどの可能性）")
                else:
                    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                        print(f"  [検証失敗] 置き換え後のプログラムの再実行でエラーが発生しました: {error_type}: {error_message}")
                        # 未定義変数エラーの場合は特に明示
                        if "未定義" in error_message or "undefined" in error_message.lower() or "NameError" in error_type:
                            print(f"  [検証失敗] 未定義変数エラーが検出されました。置き換えをスキップします。")
                continue

            # 検証が成功した場合のみ、置き換えを適用
            nodes = nodes_with_fallback
            program_code = program_code_after_replace

            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"  [置き換え適用] {cmd_name}を{fallback_code}に置き換えました")

        else:
            if ENABLE_ALL_LOGS:
                print(f"  [警告] 置き換えの実行に失敗しました")

    if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
        print(f"[validate_nodes_and_adjust_objects] 置き換えループ: 各コマンド検証完了", flush=True)
        print(f"\n[置き換えループ終了]")
        print(f"[validate_nodes_and_adjust_objects] 関数終了", flush=True)
    return nodes, all_objects, if_count, for_count, end_count
