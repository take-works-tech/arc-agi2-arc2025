"""
定数定義（簡素化版）

実際に使用されている定数のみを保持
"""
import random
from typing import Dict, Callable, Tuple, List


# ============================================================
# グローバル確率設定（使用中）
# ============================================================

# グリッドサイズ保持
GRID_SIZE_PRESERVATION_PROB = 0.55  # 入力と同じサイズで出力する確率（AGI2学習データの多様性向上のため0.80 → 0.55に変更）

# EMPTY_ARRAYノード追加確率（EMPTY_ARRAYノードが存在しない場合）
EMPTY_ARRAY_ADDITION_PROB = 0.14  # EMPTY_ARRAYノードが存在しない場合に追加する確率

# 依存関係解決の最大反復回数範囲
RESOLVE_DEPENDENCIES_MIN_ITERATIONS = 1  # 最小反復回数
RESOLVE_DEPENDENCIES_MAX_ITERATIONS = 3  # 最大反復回数

# ============================================================
# ノード生成リトライ設定
# ============================================================
# 注意: FILTER/SORT_BY/MATCH_PAIRSの再生成は最終チェック関数内で行われるため、
# ここでの再試行設定は不要（MAX_FILTER_CONDITION_REGENERATION_RETRIES等を使用）

# ============================================================
# 部分プログラム生成の最大試行回数
# ============================================================
# プログラム生成フローで部分プログラムを生成する際の最大試行回数
# 各試行で異なるpattern_idxをランダムに選択（重複なし）
# 最適化: 100回に削減（ほとんどのケースで1回目で成功するため、処理時間を大幅に削減）
MAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTS = 100  # 3071 → 100に削減

# ============================================================
# MERGEのみの部分プログラム生成確率（データセット生成時限定）
# ============================================================
# カテゴリ分けなしでMERGEのみの部分プログラムを生成する確率
# 推論パイプラインとの一貫性を保つため、低確率でMERGEのみのパターンも生成
# AGI2実データ分析結果（9.07%）を参考に、6%に設定（保守的な値）
MERGE_ONLY_PATTERN_PROBABILITY = 0.06  # 6%

# ============================================================
# 引数生成の最大試行回数（new_rules_summary.mdに基づく）
# ============================================================
# コマンド生成時の再生成最大試行回数（docs/new_rules_summary.md 行204）
# コマンド選択後、引数生成と最終チェックを繰り返す最大回数
MAX_COMMAND_GENERATION_RETRIES = 10

# FILTERの第2引数再生成最大試行回数（docs/new_rules_summary.md 行233）
# FILTERの第2引数内で$objが使用されていない場合の再生成最大回数
MAX_FILTER_CONDITION_REGENERATION_RETRIES = 10


# MATCH_PAIRSの第3引数再生成最大試行回数（docs/new_rules_summary.md 行251）
# MATCH_PAIRSの第3引数内で$obj1,$obj2が最低一回ずつ使用されていない場合の再生成最大回数
MAX_MATCH_PAIRS_CONDITION_REGENERATION_RETRIES = 10


# SORT_BYの第2引数再生成最大試行回数（docs/new_rules_summary.md 行267）
# SORT_BYの第2引数内で$objが使用されていない場合の再生成最大回数
MAX_SORT_BY_KEY_EXPR_REGENERATION_RETRIES = 10


# 重複構造チェック時の再生成最大試行回数（docs/new_rules_summary.md 行315）
# コマンドの引数内にまったく同じ構造の引数がある場合の再生成最大回数
MAX_DUPLICATE_STRUCTURE_REGENERATION_RETRIES = 10

# SemanticType選択時の最大再試行回数（動的計算のベース値）
# 比較演算・算術演算・通常の引数で、SemanticType選択時に重みがすべて0.0の場合の再試行回数
# 実際の値は len(compatible_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER で計算
SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER = 2

# ============================================================
# FORループ外でのコマンド重み調整設定
# ============================================================
# FORのネスト数が0の時（FORループ外）に、確率を下げるコマンドとその倍率
# キー: コマンド名, 値: 重み倍率（1.0未満で確率を下げる）
# 引数にオブジェクト単体を含むコマンドすべてを設定
# 注意: プレースホルダー変数を使うコマンド（FILTER、MATCH_PAIRS、SORT_BY）は
#       この重み調整を無視します（command_generators.pyで処理）


COMMANDS_TO_REDUCE_OUTSIDE_FOR_LOOP: Dict[str, float] = {
    # 基本操作
    'MOVE': 0.25,
    'TELEPORT': 0.25,
    'ROTATE': 0.25,
    'FLIP': 0.25,
    'SCALE': 0.25,
    'SCALE_DOWN': 0.25,
    'EXPAND': 0.25,
    'FILL_HOLES': 0.25,
    'SET_COLOR': 0.25,
    'ALIGN': 0.25,

    # 高度操作
    'OUTLINE': 0.25,
    'HOLLOW': 0.25,
    'BBOX': 0.25,
    'INTERSECTION': 0.1,
    'SUBTRACT': 0.1,
    'FLOW': 0.25,
    'DRAW': 0.25,
    'LAY': 0.25,
    'SLIDE': 0.25,
    'PATHFIND': 0.25,

    # オブジェクト生成
    'APPEND': 0.25,
    'TILE': 0.25,

    # オブジェクト分割/抽出
    'SPLIT_CONNECTED': 0.25,
    'CROP': 0.25,
    'EXTRACT_RECTS': 0.25,
    'EXTRACT_HOLLOW_RECTS': 0.25,
    'EXTRACT_LINES': 0.25,

    # フィッティング
    'FIT_SHAPE': 0.1,
    'FIT_SHAPE_COLOR': 0.1,
    'FIT_ADJACENT': 0.1,

    # GET関数（オブジェクト単体を引数に取る）
    'GET_ASPECT_RATIO': 0.25,
    'GET_CENTER_X': 0.25,
    'GET_CENTER_Y': 0.25,
    'GET_CENTROID': 0.25,
    'GET_COLOR': 0.25,
    'GET_COLORS': 0.25,
    'GET_DENSITY': 0.25,
    'GET_DIRECTION': 0.25,
    'GET_DISTANCE': 0.25,
    'GET_HEIGHT': 0.25,
    'GET_LINE_TYPE': 0.25,
    'GET_MAX_X': 0.25,
    'GET_MAX_Y': 0.25,
    'GET_NEAREST': 0.25,
    'GET_RECTANGLE_TYPE': 0.25,
    'GET_SIZE': 0.25,
    'GET_SYMMETRY_SCORE': 0.25,
    'GET_WIDTH': 0.25,
    'GET_X': 0.25,
    'GET_X_DISTANCE': 0.1,
    'GET_Y': 0.25,
    'GET_Y_DISTANCE': 0.1,

    # 判定関数
    'IS_IDENTICAL': 0.1,
    'IS_INSIDE': 0.25,
    'IS_SAME_SHAPE': 0.1,
    'IS_SAME_STRUCT': 0.1,

    # カウント関数
    'COUNT_ADJACENT': 0.1,
    'COUNT_HOLES': 0.1,
    'COUNT_OVERLAP': 0.1,
}

# ============================================================
# 出力グリッドサイズの決定（使用中）
# ============================================================

OUTPUT_SIZE_PATTERNS: Dict[str, Dict] = {
    'same': {
        'probability': 0.50,  # AGI2学習データの多様性向上のため0.654 → 0.50に変更
        'generator': lambda width=12, height=12: (width, height)  # 入力と同じ
    },
    'scale': {
        'probability': 0.20,  # AGI2学習データの多様性向上のため0.081 → 0.20に変更
        'generator': lambda width=12, height=12: _generate_scaled_size(width, height)
    },
    'random': {
        'probability': 0.30,  # AGI2学習データの多様性向上のため0.265 → 0.30に変更
        'generator': lambda width=12, height=12: _generate_random_size(width, height)
    }
}

def _generate_random_size(grid_width: int = 12, grid_height: int = 12) -> Tuple[int, int]:
    """ランダムサイズを生成（グリッドサイズに基づく）"""
    from .types import TypeSystem, SemanticType
    # SIZE型の範囲を取得
    size_range = TypeSystem.get_default_range(SemanticType.SIZE, grid_width, grid_height)
    if size_range:
        min_size, max_size = size_range
        min_size = max(2, min_size)  # 最小2を保証
        # 90%の確率で大きめのサイズ、10%の確率で小さめのサイズ
        if random.random() < 0.9:
            if random.random() < 0.8:
                # 大きめのサイズ（min_size以上）
                size_range_for_random = (max(min_size, min(10, max_size)), max_size)
                if size_range_for_random[0] <= size_range_for_random[1]:
                    width = random.randint(*size_range_for_random)
                    height = random.randint(*size_range_for_random)
                else:
                    width = random.randint(min_size, max_size)
                    height = random.randint(min_size, max_size)
            else:
                width = random.randint(10, 30) if random.random() < 0.9 else random.randint(1, 9)
                height = random.randint(10, 30) if random.random() < 0.9 else random.randint(1, 9)
        else:
            # 小さめのサイズ（1-9の範囲、ただしmin_size以上）
            small_max = max(9, min_size)
            width = random.randint(max(1, min_size), small_max)
            height = random.randint(max(1, min_size), small_max)


    else:
        # フォールバック
        width = random.randint(10, 30) if random.random() < 0.9 else random.randint(1, 9)
        height = random.randint(10, 30) if random.random() < 0.9 else random.randint(1, 9)

    return (width, height)

def _generate_scaled_size(grid_width: int = 12, grid_height: int = 12) -> Tuple[int, int]:
    """スケールされたサイズを生成（グリッドサイズに基づく）"""
    scale_choice = random.random()
    if scale_choice < 0.6:
        # 拡大（60%）
        scale_factor = random.choice([2, 3])
        return (grid_width * scale_factor, grid_height * scale_factor)
    else:
        # 縮小（40%）
        scale_divisor = random.choice([2, 3])
        return (grid_width // scale_divisor, grid_height // scale_divisor)

# ============================================================
# 複雑度システム（使用中）
# ============================================================

# 複雑度 → ステップ数範囲（ARC-AGI2実分布に基づく調整）
# 実分布: 平均5.9、p10=2、p25=3、p50=4、p75=8、p90=12、p95=15、p99=20、最大60
# 形式: (最小ステップ数, 最大ステップ数)
# 注意: min_nodesとSTEPS_BY_COMPLEXITYの両方の条件を満たすまで生成を継続
STEPS_BY_COMPLEXITY: Dict[int, Tuple[int, int]] = {
    1: (1, 2),      # 最小1、最大2ステップ（ARC-AGI2想定: シンプル、p10=2）
    2: (1, 3),      # 最小2、最大3ステップ（ARC-AGI2想定: ややシンプル、p25=3）
    3: (1, 3),      # 最小3、最大5ステップ（ARC-AGI2想定: 中程度、p50=4を超える範囲）
    4: (2, 5),      # 最小4、最大8ステップ（ARC-AGI2想定: 中程度、p50=4からp75=8まで）
    5: (3, 7),     # 最小5、最大12ステップ（ARC-AGI2想定: 複雑、p75=8からp90=12まで）
    6: (3, 8),     # 最小6、最大15ステップ（ARC-AGI2想定: やや複雑、p90=12からp95=15まで）
    7: (4, 8),     # 最小7、最大20ステップ（ARC-AGI2想定: 複雑、p90=12からp99=20まで）
    8: (5, 10),     # 最小8、最大30ステップ（ARC-AGI2想定: 非常に複雑、p99=20を超えて最大60に近づく）
}

# 複雑度 → ノード数範囲（ARC-AGI2実分布に基づく調整）
# 実分布: 平均14.6ノード、p50=14、p75=18、p90=22、p95=26、p99=34、最大78
# 形式: (最小ノード数, 最大ノード数)
NODES_BY_COMPLEXITY: Dict[int, Tuple[int, int]] = {
    1: (2, 20),     # 2-30ノード（ARC-AGI2想定: Very Simple）
    2: (4, 30),     # 4-40ノード（ARC-AGI2想定: Simple）
    3: (10, 40),    # 10-50ノード（ARC-AGI2想定: Medium）
    4: (12, 50),    # 12-60ノード（ARC-AGI2想定: Complex）
    5: (15, 60),    # 18-70ノード（ARC-AGI2想定: Very Complex）
    6: (15, 70),    # 22-80ノード（ARC-AGI2想定: 超複雑）
    7: (20, 80),    # 25-90ノード（ARC-AGI2想定: 超複雑）
    8: (20, 120),   # 27-100ノード（ARC-AGI2想定: 超複雑）
}


# ============================================================
# 構造的ネスト深度制限（FOR/IFのネスト用）
# ============================================================
# ネスト深度制限の有効化フラグ
ENABLE_NESTING_DEPTH_LIMIT = True

# ============================================================
# コマンドネスト深度制限（コマンド引数内のネスト用）
# ============================================================
#
# 用途: コマンド引数内でのネスト深度を制限
# 使用箇所: ProgramContext.max_command_nesting_depth, get_command_nesting_depth()
# 例: FILTER(objects, EQUAL(GET_COLOR($obj), 3)) のような引数内のネスト
#
# 複雑度別のコマンド最大ネスト深度
COMMAND_NESTING_DEPTH_BY_COMPLEXITY: Dict[int, int] = {
    1: 1,    # Simple: 浅いコマンドネスト
    2: 1,    # Simple: 浅いコマンドネスト
    3: 2,    # Simple: 中程度のコマンドネスト
    4: 2,    # Medium: 中程度のコマンドネスト
    5: 2,    # Medium: 深いコマンドネスト
    6: 3,    # Medium: 深いコマンドネスト
    7: 3,    # Complex: 非常に深いコマンドネスト
    8: 3,    # Complex: 非常に深いコマンドネスト
}

# ============================================================
# 複雑度自動選択（使用中）
# ============================================================

def select_complexity() -> int:
    """複雑度を自動選択（5段階細分化版）"""
    rand = random.random()
    if rand < 0.15:
        return 1  # 15%: Very Simple
    elif rand < 0.35:
        return 2  # 20%: Simple
    elif rand < 0.60:
        return 3  # 25%: Medium
    elif rand < 0.85:
        return 4  # 25%: Complex
    else:
        return 5  # 15%: Very Complex

# ============================================================
# 出力グリッドサイズ生成（使用中）
# ============================================================

def generate_output_grid_size(input_width: int = 12, input_height: int = 12) -> Tuple[int, int]:
    """出力グリッドサイズを生成（入力グリッドサイズに基づく）"""
    rand = random.random()

    if rand < 0.50:
        # 入力と同じ（50%）- AGI2学習データの多様性向上のため0.654 → 0.50に変更
        return OUTPUT_SIZE_PATTERNS['same']['generator'](input_width, input_height)

    elif rand < 0.50 + 0.20:
        # 整数倍スケール（20%）- AGI2学習データの多様性向上のため0.081 → 0.20に変更
        return OUTPUT_SIZE_PATTERNS['scale']['generator'](input_width, input_height)

    else:
        # ランダムサイズ（30%）- AGI2学習データの多様性向上のため0.265 → 0.30に変更
        return OUTPUT_SIZE_PATTERNS['random']['generator'](input_width, input_height)

# ============================================================
# コマンドカテゴリの定義
COMMAND_CATEGORIES = {
    'getter': ['GET_X', 'GET_Y', 'GET_WIDTH', 'GET_HEIGHT', 'GET_SIZE', 'GET_COLOR',
               'GET_COLORS', 'GET_BACKGROUND_COLOR', 'GET_DISTANCE', 'GET_X_DISTANCE',
               'GET_Y_DISTANCE', 'GET_DIRECTION', 'GET_NEAREST', 'GET_SYMMETRY_SCORE', 'GET_LINE_TYPE', 'GET_RECTANGLE_TYPE',
               'GET_ASPECT_RATIO', 'GET_DENSITY', 'GET_CENTROID',
               'GET_CENTER_X', 'GET_CENTER_Y', 'GET_MAX_X', 'GET_MAX_Y',
               'GET_INPUT_GRID_SIZE', 'COUNT_HOLES', 'COUNT_ADJACENT', 'COUNT_OVERLAP', 'LEN'],
    'comparison': ['IS_SAME_SHAPE', 'IS_IDENTICAL', 'IS_INSIDE', 'IS_SAME_STRUCT'],
    'arithmetic': ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD'],
    'proportional': ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS'],
    'transform': ['MOVE', 'TELEPORT', 'SLIDE', 'PATHFIND', 'ROTATE', 'FLIP', 'SCALE', 'SCALE_DOWN',
                  'EXPAND', 'FILL_HOLES', 'SET_COLOR', 'OUTLINE', 'HOLLOW', 'BBOX',
                  'INTERSECTION', 'SUBTRACT', 'FLOW', 'DRAW', 'LAY', 'ALIGN', 'CROP', 'FIT_SHAPE',
                  'FIT_SHAPE_COLOR', 'FIT_ADJACENT'],
    'create': ['CREATE_LINE', 'CREATE_RECT', 'MERGE', 'TILE'],
    'array': ['FILTER', 'MATCH_PAIRS', 'SORT_BY', 'EXTEND_PATTERN', 'EXCLUDE',
              'ARRANGE_GRID', 'CONCAT', 'APPEND', 'REVERSE', 'EXTRACT_LINES', 'EXTRACT_RECTS',
              'EXTRACT_HOLLOW_RECTS', 'GET_ALL_OBJECTS', 'SPLIT_CONNECTED']
}

# ============================================================
# プログラム複雑さ制御（改善案2）
# ============================================================

# 複雑さレベルの定義（5段階細分化版）
#
# すべての複雑度関連設定を1箇所に集約
# 注意: max_nodesとmin_nodesはNODES_BY_COMPLEXITYから取得
COMPLEXITY_LEVELS = {
    # 数値キー版（ProgramContext用、ARC-AGI2実分布に基づく調整）
    # 実分布: 平均14.6ノード、p50=14、p75=18、p90=22、p95=26、p99=34、最大78
    1: {
        'max_nesting_depth': 1,  # 構造的ネスト深度（FOR/IFのスコープネスト）
        'max_for_loops': 4,       # FORループ許可（最大4）
        'max_if_statements': 1,   # IF文最大数
        'probability': 0.15,     # 15%の確率
        'max_array_operations': 1,
        'max_for_array_assignments': 1  # FORループ内で使用可能な配列代入ノードの最大数
    },
    2: {
        'max_nesting_depth': 2,   # 構造的ネスト深度
        'max_for_loops': 5,       # 最大5個のFORループ
        'max_if_statements': 2,   # 最大2個のIF文
        'probability': 0.20,     # 20%の確率
        'max_array_operations': 2,
        'max_for_array_assignments': 1  # FORループ内で使用可能な配列代入ノードの最大数
    },
    3: {
        'max_nesting_depth': 2,   # 構造的ネスト深度
        'max_for_loops': 6,       # 最大6個のFORループ
        'max_if_statements': 2,   # 最大2個のIF文
        'probability': 0.25,     # 25%の確率
        'max_array_operations': 4,
        'max_for_array_assignments': 1  # FORループ内で使用可能な配列代入ノードの最大数
    },
    4: {
        'max_nesting_depth': 3,   # 構造的ネスト深度
        'max_for_loops': 5,       # 最大5個のFORループ
        'max_if_statements': 4,   # 最大4個のIF文
        'probability': 0.25,     # 25%の確率
        'max_array_operations': 4,
        'max_for_array_assignments': 1  # FORループ内で使用可能な配列代入ノードの最大数
    },
    5: {
        'max_nesting_depth': 3,   # 構造的ネスト深度
        'max_for_loops': 8,       # 最大8個のFORループ
        'max_if_statements': 5,   # 最大5個のIF文
        'probability': 0.15,     # 15%の確率
        'max_array_operations': 6,
        'max_for_array_assignments': 2  # FORループ内で使用可能な配列代入ノードの最大数
    },
    6: {
        'max_nesting_depth': 3,   # 構造的ネスト深度
        'max_for_loops': 9,       # 最大9個のFORループ
        'max_if_statements': 6,   # 最大6個のIF文
        'probability': 0.10,     # 自動選択ではほぼ使用しない
        'max_array_operations': 8,
        'max_for_array_assignments': 2  # FORループ内で使用可能な配列代入ノードの最大数
    },
    7: {
        'max_nesting_depth': 4,   # 構造的ネスト深度
        'max_for_loops': 10,      # 最大10個のFORループ
        'max_if_statements': 7,   # 最大7個のIF文
        'probability': 0.05,     # 自動選択ではほぼ使用しない
        'max_array_operations': 10,
        'max_for_array_assignments': 3  # FORループ内で使用可能な配列代入ノードの最大数
    },
    8: {
        'max_nesting_depth': 4,   # 構造的ネスト深度
        'max_for_loops': 11,      # 最大11個のFORループ
        'max_if_statements': 8,   # 最大8個のIF文
        'probability': 0.05,     # 自動選択ではほぼ使用しない
        'max_array_operations': 10,
        'max_for_array_assignments': 3  # FORループ内で使用可能な配列代入ノードの最大数
    }
}

# 複雑さ制御の有効化フラグ
ENABLE_COMPLEXITY_CONTROL = True
