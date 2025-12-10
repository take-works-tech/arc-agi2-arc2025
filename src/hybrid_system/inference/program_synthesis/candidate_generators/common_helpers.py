"""
共通ヘルパー関数

すべての候補生成器で使用される共通のヘルパー関数
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def add_render_grid_if_needed(
    program: str,
    output_grid: List[List[int]],
    matching_result: Optional[Dict[str, Any]] = None,
    pair_index: Optional[int] = None
) -> str:
    """プログラムにRENDER_GRIDを追加（必要な場合のみ）

    Args:
        program: プログラムコード
        output_grid: 出力グリッド
        matching_result: オブジェクトマッチング結果（キャッシュ取得用、オプション）
        pair_index: 訓練ペアのインデックス（キャッシュ取得用、オプション）

    Returns:
        RENDER_GRIDが追加されたプログラム
    """
    if 'RENDER_GRID' in program:
        return program

    if is_empty_grid(output_grid):
        return program

    # キャッシュから取得（なければ計算）
    grid_info_cache = matching_result.get('grid_info_cache', {}) if matching_result else {}
    if pair_index is not None and pair_index in grid_info_cache:
        grid_info = grid_info_cache[pair_index]
        output_h, output_w = grid_info['output_size']
        bg_color = grid_info['output_bg']
    else:
        # フォールバック: 直接計算
        output_h, output_w = get_grid_size(output_grid)
        if output_h == 0 or output_w == 0:
            return program
        bg_color = get_background_color(output_grid)

    # objects変数が定義されているかチェック
    if 'objects =' in program or 'objects=' in program:
        return f"{program}\nRENDER_GRID(objects, {bg_color}, {output_w}, {output_h})"
    else:
        if 'GET_ALL_OBJECTS' in program:
            program = program.replace('GET_ALL_OBJECTS', 'objects = GET_ALL_OBJECTS', 1)
        else:
            program = f"objects = GET_ALL_OBJECTS(4)\n{program}"
        return f"{program}\nRENDER_GRID(objects, {bg_color}, {output_w}, {output_h})"


def get_background_color(grid: List[List[int]], use_edge_heuristic: bool = False) -> int:
    """グリッドの背景色を取得（共通実装、改善版）

    Args:
        grid: グリッド
        use_edge_heuristic: エッジ色を考慮するか（デフォルト: False、シンプルな実装）

    Returns:
        背景色（通常は0）
    """
    if is_empty_grid(grid):
        return 0

    flat_colors = [c for row in grid for c in row]
    if not flat_colors:
        return 0

    counter = Counter(flat_colors)
    # 最頻出色を基本候補とする
    most_common_color, _ = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))

    if use_edge_heuristic:
        # エッジ色を考慮（より詳細な実装）
        edge_colors = _get_edge_colors(grid)
        if edge_colors:
            edge_color_counts = Counter(edge_colors)
            most_common_edge_color = edge_color_counts.most_common(1)[0][0]
            # エッジ色と最頻出色が一致する場合、信頼度が高い
            if most_common_edge_color == most_common_color:
                return int(most_common_color)
            else:
                # エッジ色を優先（背景色はエッジに多く出現する可能性が高い）
                return int(most_common_edge_color)

    # シンプルな実装: 最頻出色を返す（同率の場合は数値が小さいものを優先）
    return int(most_common_color)


def _get_edge_colors(grid: List[List[int]]) -> List[int]:
    """グリッドのエッジ色を取得（内部ヘルパー関数）

    Args:
        grid: グリッド

    Returns:
        エッジ色のリスト
    """
    if is_empty_grid(grid):
        return []

    h, w = get_grid_size(grid)
    edge_colors = []

    # 上端と下端
    edge_colors.extend(grid[0])
    if h > 1:
        edge_colors.extend(grid[h - 1])

    # 左端と右端
    for i in range(h):
        edge_colors.append(grid[i][0])
        if w > 1:
            edge_colors.append(grid[i][w - 1])

    return edge_colors


def is_empty_grid(grid: List[List[int]]) -> bool:
    """グリッドが空かチェック

    Args:
        grid: グリッド

    Returns:
        空の場合True
    """
    return not grid or not grid[0] or len(grid) == 0 or len(grid[0]) == 0


def get_grid_size(grid: List[List[int]]) -> tuple[int, int]:
    """グリッドサイズを取得

    Args:
        grid: グリッド

    Returns:
        (height, width) のタプル
    """
    if is_empty_grid(grid):
        return (0, 0)
    return (len(grid), len(grid[0]))


# ============================================================================
# 共通パラメータ推定メソッド（重複実装の統合）
# ============================================================================

def infer_target_color_candidates_from_output(output_grid: List[List[int]], max_candidates: int = 3) -> List[int]:
    """出力グリッドから代表的な色の候補を推定（複数候補を返す、共通実装）

    Args:
        output_grid: 出力グリッド
        max_candidates: 最大候補数（デフォルト: 3）

    Returns:
        色候補のリスト
    """
    flat_colors = [c for row in output_grid for c in row if c != 0]
    if not flat_colors:
        return [1]
    counter = Counter(flat_colors)

    # 頻度順にソート（頻度が高い順、同頻度の場合は色番号が小さい順）
    sorted_colors = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    # 上位max_candidates色を候補として返す
    candidates = [int(color) for color, _ in sorted_colors[:max_candidates]]

    # 改善: 背景色（0）以外で、頻度が低い色も候補に追加（多様性を確保）
    if len(candidates) < max_candidates:
        additional_colors = [int(color) for color, _ in sorted_colors[max_candidates:max_candidates*2] if color not in candidates]
        candidates.extend(additional_colors[:max_candidates - len(candidates)])

    return candidates[:max_candidates]


def infer_target_color_from_output(output_grid: List[List[int]]) -> int:
    """出力グリッドから代表的な色を推定する（共通実装）

    Args:
        output_grid: 出力グリッド

    Returns:
        代表的な色（候補の最初の色）
    """
    candidates = infer_target_color_candidates_from_output(output_grid, max_candidates=1)
    return candidates[0] if candidates else 1


def estimate_movement_parameters(
    input_grid: List[List[int]],
    output_grid: List[List[int]],
    max_candidates: int = 3
) -> List[Tuple[int, int]]:
    """移動量パラメータを推定（共通実装）

    Args:
        input_grid: 入力グリッド
        output_grid: 出力グリッド
        max_candidates: 最大候補数（デフォルト: 3）

    Returns:
        移動量候補のリスト [(dx, dy), ...]（可能性の高い順）
    """
    if is_empty_grid(input_grid) or is_empty_grid(output_grid):
        return []

    input_h, input_w = get_grid_size(input_grid)
    output_h, output_w = get_grid_size(output_grid)

    if input_h != output_h or input_w != output_w:
        return []

    # 非ゼロピクセルの位置を取得
    input_positions = []
    output_positions = []

    for i in range(input_h):
        for j in range(input_w):
            if input_grid[i][j] != 0:
                input_positions.append((i, j, input_grid[i][j]))
            if output_grid[i][j] != 0:
                output_positions.append((i, j, output_grid[i][j]))

    if not input_positions or not output_positions:
        return []

    # 色が一致するピクセルペアを探す
    movement_counts = {}

    for in_i, in_j, in_color in input_positions:
        for out_i, out_j, out_color in output_positions:
            if in_color == out_color:
                dx = out_j - in_j
                dy = out_i - in_i
                movement = (dx, dy)
                movement_counts[movement] = movement_counts.get(movement, 0) + 1

    if not movement_counts:
        return []

    # 最も頻繁な移動量をソート
    sorted_movements = sorted(
        movement_counts.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    # 上位max_candidates個の移動量を返す
    return [movement for movement, count in sorted_movements[:max_candidates]]


def estimate_scale_factor_from_grids(
    input_grid: List[List[int]],
    output_grid: List[List[int]]
) -> Optional[int]:
    """グリッドからスケール倍率を推定（共通実装）

    Args:
        input_grid: 入力グリッド
        output_grid: 出力グリッド

    Returns:
        推定されたスケール倍率（推定できない場合はNone）
    """
    if is_empty_grid(input_grid) or is_empty_grid(output_grid):
        return None

    input_h, input_w = get_grid_size(input_grid)
    output_h, output_w = get_grid_size(output_grid)

    # グリッドサイズの比率からスケール倍率を推定
    h_ratio = output_h / input_h if input_h > 0 else 1.0
    w_ratio = output_w / input_w if input_w > 0 else 1.0

    # 比率が整数に近い場合、その値を返す
    h_scale = int(round(h_ratio))
    w_scale = int(round(w_ratio))

    # 高さと幅のスケールが一致し、かつ1より大きい場合
    if h_scale == w_scale and h_scale > 1:
        # 実際の比率との差が小さい場合のみ採用
        if abs(h_ratio - h_scale) < 0.2 and abs(w_ratio - w_scale) < 0.2:
            return h_scale

    return None


def estimate_scale_factor(input_size: int, output_size: int) -> float:
    """サイズからスケール倍率を推定（共通実装）

    Args:
        input_size: 入力サイズ
        output_size: 出力サイズ

    Returns:
        スケール倍率
    """
    if input_size == 0:
        return 1.0
    return float(output_size) / float(input_size)


# ============================================================================
# オブジェクトユーティリティ関数（重複実装の統合）
# ============================================================================

def get_object_center(obj: Any) -> Tuple[int, int]:
    """オブジェクトの中心座標を取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        (center_x, center_y) のタプル
    """
    # Objectインスタンスの場合
    if hasattr(obj, 'center_position') and obj.center_position:
        return obj.center_position
    if hasattr(obj, 'center') and obj.center:
        return tuple(obj.center) if isinstance(obj.center, (list, tuple)) else obj.center

    # Dictの場合
    if isinstance(obj, dict):
        if 'center' in obj:
            center = obj['center']
            return tuple(center) if isinstance(center, (list, tuple)) else center
        if 'center_position' in obj:
            center = obj['center_position']
            return tuple(center) if isinstance(center, (list, tuple)) else center

    # bboxから計算
    bbox = get_object_bbox(obj)
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # pixelsから計算
    if hasattr(obj, 'pixels') and obj.pixels:
        pixels = obj.pixels
    elif isinstance(obj, dict) and 'pixels' in obj:
        pixels = obj['pixels']
    else:
        return (0, 0)

    if not pixels:
        return (0, 0)

    # pixelsが(x, y)形式か(y, x)形式かを判定
    first_pixel = pixels[0]
    if isinstance(first_pixel, (list, tuple)) and len(first_pixel) >= 2:
        # 通常は(x, y)形式
        xs = [p[0] for p in pixels if len(p) >= 2]
        ys = [p[1] for p in pixels if len(p) >= 2]
        if xs and ys:
            return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    return (0, 0)


def get_object_size(obj: Any) -> int:
    """オブジェクトのサイズ（ピクセル数）を取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        ピクセル数
    """
    # Objectインスタンスの場合
    if hasattr(obj, 'size') and obj.size:
        return obj.size
    if hasattr(obj, 'area') and obj.area:
        return obj.area

    # Dictの場合
    if isinstance(obj, dict):
        if 'size' in obj:
            return obj['size']
        if 'area' in obj:
            return obj['area']

    # pixelsから計算
    if hasattr(obj, 'pixels') and obj.pixels:
        return len(obj.pixels)
    if isinstance(obj, dict) and 'pixels' in obj:
        return len(obj['pixels'])

    return 0


def get_object_width(obj: Any) -> int:
    """オブジェクトの幅を取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        幅
    """
    bbox = get_object_bbox(obj)
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        return x2 - x1 + 1
    return 0


def get_object_height(obj: Any) -> int:
    """オブジェクトの高さを取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        高さ
    """
    bbox = get_object_bbox(obj)
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        return y2 - y1 + 1
    return 0


def get_object_bbox(obj: Any) -> Optional[Tuple[int, int, int, int]]:
    """オブジェクトのバウンディングボックスを取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        (x1, y1, x2, y2) のタプル、またはNone
    """
    # Objectインスタンスの場合
    if hasattr(obj, 'bbox') and obj.bbox:
        bbox = obj.bbox
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return tuple(bbox[:4])

    # Dictの場合
    if isinstance(obj, dict):
        if 'bbox' in obj and obj['bbox']:
            bbox = obj['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return tuple(bbox[:4])

    return None


def get_object_color(obj: Any) -> int:
    """オブジェクトの色を取得（共通実装）

    Args:
        obj: オブジェクト（ObjectインスタンスまたはDict）

    Returns:
        色（デフォルト: 0）
    """
    # Objectインスタンスの場合
    if hasattr(obj, 'color') and obj.color is not None:
        return int(obj.color)
    if hasattr(obj, 'dominant_color') and obj.dominant_color is not None:
        return int(obj.dominant_color)

    # Dictの場合
    if isinstance(obj, dict):
        if 'color' in obj and obj['color'] is not None:
            return int(obj['color'])
        if 'dominant_color' in obj and obj['dominant_color'] is not None:
            return int(obj['dominant_color'])

    return 0


# ============================================================================
# グリッドユーティリティ関数（重複実装の統合）
# ============================================================================

def get_unique_colors(grid: List[List[int]], exclude_zero: bool = False) -> List[int]:
    """グリッド内のユニークな色を取得（共通実装）

    Args:
        grid: グリッド
        exclude_zero: 背景色（0）を除外するか

    Returns:
        ユニークな色のリスト
    """
    if is_empty_grid(grid):
        return []

    flat_colors = [c for row in grid for c in row]
    if exclude_zero:
        flat_colors = [c for c in flat_colors if c != 0]

    unique_colors = sorted(set(flat_colors))
    return unique_colors


def get_color_distribution(grid: List[List[int]]) -> Dict[int, int]:
    """グリッド内の色の分布を取得（共通実装）

    Args:
        grid: グリッド

    Returns:
        色の分布（色: 出現回数）
    """
    if is_empty_grid(grid):
        return {}

    flat_colors = [c for row in grid for c in row]
    return dict(Counter(flat_colors))


def calculate_euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """2点間のユークリッド距離を計算（共通実装）

    Args:
        pos1: 位置1 (x, y)
        pos2: 位置2 (x, y)

    Returns:
        ユークリッド距離
    """
    import math
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


# ============================================================================
# グリッド操作関数（重複実装の統合）
# ============================================================================

def pad_grid(grid: List[List[int]], pad_top: int = 0, pad_bottom: int = 0,
             pad_left: int = 0, pad_right: int = 0, pad_value: int = 0) -> List[List[int]]:
    """グリッドにパディングを追加（共通実装）

    Args:
        grid: グリッド
        pad_top: 上側のパディング
        pad_bottom: 下側のパディング
        pad_left: 左側のパディング
        pad_right: 右側のパディング
        pad_value: パディングの値（デフォルト: 0）

    Returns:
        パディングされたグリッド
    """
    if is_empty_grid(grid):
        return grid

    h, w = get_grid_size(grid)
    new_h = h + pad_top + pad_bottom
    new_w = w + pad_left + pad_right

    # 新しいグリッドを作成
    new_grid = [[pad_value for _ in range(new_w)] for _ in range(new_h)]

    # 元のグリッドを配置
    for i in range(h):
        for j in range(w):
            new_grid[i + pad_top][j + pad_left] = grid[i][j]

    return new_grid


def crop_grid(grid: List[List[int]], start_row: int, start_col: int,
              height: int, width: int) -> List[List[int]]:
    """グリッドをクロップ（共通実装）

    Args:
        grid: グリッド
        start_row: 開始行
        start_col: 開始列
        height: 高さ
        width: 幅

    Returns:
        クロップされたグリッド
    """
    if is_empty_grid(grid):
        return grid

    h, w = get_grid_size(grid)

    # 境界チェック
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    end_row = min(start_row + height, h)
    end_col = min(start_col + width, w)

    if start_row >= end_row or start_col >= end_col:
        return [[0]]

    return [row[start_col:end_col] for row in grid[start_row:end_row]]


def resize_grid(grid: List[List[int]], new_height: int, new_width: int,
                default_value: int = 0) -> List[List[int]]:
    """グリッドサイズを変更（共通実装）

    Args:
        grid: グリッド
        new_height: 新しい高さ
        new_width: 新しい幅
        default_value: デフォルト値（デフォルト: 0）

    Returns:
        リサイズされたグリッド
    """
    if is_empty_grid(grid):
        return [[default_value for _ in range(new_width)] for _ in range(new_height)]

    h, w = get_grid_size(grid)
    new_grid = [[default_value for _ in range(new_width)] for _ in range(new_height)]

    # 元のグリッドをコピー
    for i in range(min(h, new_height)):
        for j in range(min(w, new_width)):
            new_grid[i][j] = grid[i][j]

    return new_grid


# ============================================================================
# 部分プログラム解析ヘルパー関数
# ============================================================================

def parse_partial_program(partial_program: str) -> Dict[str, Any]:
    """部分プログラムを解析して構造とパラメータを抽出

    Args:
        partial_program: 部分プログラムコード

    Returns:
        解析結果の辞書:
        - 'connectivity': GET_ALL_OBJECTSのconnectivity引数（4 or 8）
        - 'pattern_type': パターンタイプ（'color_change', 'position_change', 'shape_change', 'identity', 'categorized', 'unknown'）
        - 'parameters': パラメータの辞書（色、移動量、スケール倍率など）
        - 'has_loop': ループがあるかどうか
        - 'is_valid': 有効な部分プログラムかどうか
        - 'has_categories': カテゴリ分けがあるかどうか（objects1, objects2など）
        - 'category_vars': カテゴリ変数のリスト（['objects1', 'objects2', ...]）
    """
    if not partial_program or not partial_program.strip():
        return {
            'connectivity': 4,
            'pattern_type': 'unknown',
            'parameters': {},
            'has_loop': False,
            'is_valid': False,
            'has_categories': False,
            'category_vars': []
        }

    import re

    result = {
        'connectivity': 4,  # デフォルト
        'pattern_type': 'unknown',
        'parameters': {},
        'has_loop': 'FOR' in partial_program and 'DO' in partial_program,
        'is_valid': True,
        'has_categories': False,
        'category_vars': []
    }

    # カテゴリ分けの検出（objects1, objects2, ...）
    category_var_matches = re.findall(r'objects(\d+)\s*=', partial_program)
    if category_var_matches:
        result['has_categories'] = True
        result['category_vars'] = [f"objects{i}" for i in category_var_matches]
        result['pattern_type'] = 'categorized'

    # GET_ALL_OBJECTSのconnectivity引数を抽出
    match = re.search(r'GET_ALL_OBJECTS\((\d+)\)', partial_program)
    if match:
        result['connectivity'] = int(match.group(1))

    # カテゴリ分けがない場合のみ、パターンタイプとパラメータを抽出
    if not result['has_categories']:
        if 'SET_COLOR' in partial_program:
            result['pattern_type'] = 'color_change'
            # 色パラメータを抽出
            match = re.search(r'SET_COLOR\([^,]+,\s*(\d+)\)', partial_program)
            if match:
                result['parameters']['target_color'] = int(match.group(1))
        elif 'MOVE' in partial_program:
            result['pattern_type'] = 'position_change'
            # 移動量パラメータを抽出
            match = re.search(r'MOVE\([^,]+,\s*(-?\d+),\s*(-?\d+)\)', partial_program)
            if match:
                result['parameters']['dx'] = int(match.group(1))
                result['parameters']['dy'] = int(match.group(2))
        elif 'SCALE' in partial_program:
            result['pattern_type'] = 'shape_change'
            # スケール倍率パラメータを抽出
            match = re.search(r'SCALE\([^,]+,\s*(\d+)\)', partial_program)
            if match:
                result['parameters']['scale'] = int(match.group(1))
        elif 'ROTATE' in partial_program:
            result['pattern_type'] = 'rotation'
            # 回転角度パラメータを抽出
            match = re.search(r'ROTATE\([^,]+,\s*(\d+)\)', partial_program)
            if match:
                result['parameters']['angle'] = int(match.group(1))
        elif 'GET_ALL_OBJECTS' in partial_program and not result['has_loop']:
            result['pattern_type'] = 'identity'

    return result


def extend_partial_program(
    partial_program: str,
    transformation_code: str,
    output_grid: List[List[int]],
    improved_params: Optional[Dict[str, Any]] = None,
    matching_result: Optional[Dict[str, Any]] = None,
    pair_index: Optional[int] = None
) -> str:
    """部分プログラムを拡張して完全なプログラムを生成（カテゴリ分け対応）

    Args:
        partial_program: 部分プログラムコード
        transformation_code: 追加する変換コード（ループ内の変換操作）
        output_grid: 出力グリッド（RENDER_GRID用）
        improved_params: 改善されたパラメータ（オプション）
        matching_result: オブジェクトマッチング結果（キャッシュ取得用、オプション）
        pair_index: 訓練ペアのインデックス（キャッシュ取得用、オプション）

    Returns:
        拡張された完全なプログラム
    """
    if not partial_program:
        return ""

    # 部分プログラムの解析結果をキャッシュから取得（なければ計算）
    partial_program_parsed_cache = matching_result.get('partial_program_parsed_cache', {}) if matching_result else {}
    if partial_program in partial_program_parsed_cache:
        parsed = partial_program_parsed_cache[partial_program]
    else:
        parsed = parse_partial_program(partial_program)
    if not parsed['is_valid']:
        return ""

    import re

    # カテゴリ分けを含む部分プログラムの場合
    if parsed['has_categories'] and parsed['category_vars']:
        # カテゴリ分けを含む部分プログラムを拡張
        # 各カテゴリ（objects1, objects2, ...）に対して変換を適用
        program_lines = [partial_program]  # 元の部分プログラム（カテゴリ分けを含む）

        # 各カテゴリに対して変換を適用
        all_objects_list = []
        for category_var in parsed['category_vars']:
            # 変換コードをカテゴリ変数に適用
            category_transformation = transformation_code.replace('objects[i]', f'{category_var}[i]')

            # 改善されたパラメータがある場合、変換コードを更新
            if improved_params:
                if 'target_color' in improved_params and 'SET_COLOR' in category_transformation:
                    category_transformation = re.sub(
                        r'SET_COLOR\([^,]+,\s*\d+\)',
                        f"SET_COLOR({category_var}[i], {improved_params['target_color']})",
                        category_transformation
                    )
                elif 'dx' in improved_params and 'dy' in improved_params and 'MOVE' in category_transformation:
                    category_transformation = re.sub(
                        r'MOVE\([^,]+,\s*-?\d+,\s*-?\d+\)',
                        f"MOVE({category_var}[i], {improved_params['dx']}, {improved_params['dy']})",
                        category_transformation
                    )
                elif 'scale' in improved_params and 'SCALE' in category_transformation:
                    category_transformation = re.sub(
                        r'SCALE\([^,]+,\s*\d+\)',
                        f"SCALE({category_var}[i], {improved_params['scale']})",
                        category_transformation
                    )
                elif 'angle' in improved_params and 'ROTATE' in category_transformation:
                    category_transformation = re.sub(
                        r'ROTATE\([^,]+,\s*\d+\)',
                        f"ROTATE({category_var}[i], {improved_params['angle']})",
                        category_transformation
                    )

            # カテゴリごとの変換を追加
            program_lines.append(f"FOR i LEN({category_var}) DO")
            program_lines.append(f"    {category_transformation}")
            program_lines.append("END")

            # 変換後のカテゴリをリストに追加
            all_objects_list.append(category_var)

        # すべてのカテゴリを統合
        if len(all_objects_list) > 1:
            program_lines.append(f"objects = {all_objects_list[0]}")
            for category_var in all_objects_list[1:]:
                program_lines.append(f"FOR i LEN({category_var}) DO")
                program_lines.append(f"    objects = APPEND(objects, {category_var}[i])")
                program_lines.append("END")
        elif len(all_objects_list) == 1:
            program_lines.append(f"objects = {all_objects_list[0]}")
        else:
            program_lines.append("objects = []")

        program = "\n".join(program_lines)

        # RENDER_GRIDを追加
        return add_render_grid_if_needed(program, output_grid, matching_result, pair_index)

    # カテゴリ分けがない場合（従来の処理）
    # 部分プログラムからオブジェクト取得部分を抽出
    objects_line_match = re.search(r'objects\s*=\s*GET_ALL_OBJECTS\(\d+\)', partial_program)
    if not objects_line_match:
        # GET_ALL_OBJECTSが見つからない場合、デフォルトを使用
        objects_line = f"objects = GET_ALL_OBJECTS({parsed['connectivity']})"
    else:
        objects_line = objects_line_match.group(0)

    # 改善されたパラメータがある場合、変換コードを更新
    if improved_params:
        if 'target_color' in improved_params and 'SET_COLOR' in transformation_code:
            transformation_code = re.sub(
                r'SET_COLOR\([^,]+,\s*\d+\)',
                f"SET_COLOR(objects[i], {improved_params['target_color']})",
                transformation_code
            )
        elif 'dx' in improved_params and 'dy' in improved_params and 'MOVE' in transformation_code:
            transformation_code = re.sub(
                r'MOVE\([^,]+,\s*-?\d+,\s*-?\d+\)',
                f"MOVE(objects[i], {improved_params['dx']}, {improved_params['dy']})",
                transformation_code
            )
        elif 'scale' in improved_params and 'SCALE' in transformation_code:
            transformation_code = re.sub(
                r'SCALE\([^,]+,\s*\d+\)',
                f"SCALE(objects[i], {improved_params['scale']})",
                transformation_code
            )
        elif 'angle' in improved_params and 'ROTATE' in transformation_code:
            transformation_code = re.sub(
                r'ROTATE\([^,]+,\s*\d+\)',
                f"ROTATE(objects[i], {improved_params['angle']})",
                transformation_code
            )

    # 完全なプログラムを構築
    program = f"{objects_line}\n"
    program += f"FOR i LEN(objects) DO\n"
    program += f"    {transformation_code}\n"
    program += f"END"

    # RENDER_GRIDを追加
    return add_render_grid_if_needed(program, output_grid, matching_result, pair_index)
