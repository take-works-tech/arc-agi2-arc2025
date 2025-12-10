"""
オブジェクト生成関数

条件を受け取ってオブジェクトを生成する関数群
"""
from typing import List, Dict, Optional, Set, Tuple
import random
import time
import os
import numpy as np
from .core_object_builder import CoreObjectBuilder
from ...program_executor.performance_profiler import get_profiler

# 色数調査ログの制御（デフォルト: 無効）
ENABLE_COLOR_INVESTIGATION_LOGS = os.environ.get('ENABLE_COLOR_INVESTIGATION_LOGS', 'false').lower() in ('true', '1', 'yes')

# ログ出力制御（パフォーマンス最適化：デフォルトですべてのログを無効化）
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')
from .color_distribution import (
    SAME_COLOR_AND_SHAPE_PROBABILITY,
    SAME_SHAPE_PROBABILITY
)
from .shape_utils import copy_object_with_new_color
from ..managers.auto_placement import _find_best_position


def _count_holes_in_object(obj: Dict) -> int:
    """オブジェクトの穴の数をカウント（COUNT_HOLESと同じロジック）

    Args:
        obj: オブジェクト辞書（pixels, width, heightを含む）

    Returns:
        穴の数（完全に囲まれた空白領域の数）
    """
    if not obj or 'pixels' not in obj:
        return 0

    pixels = obj.get('pixels', [])
    if not pixels:
        return 0

    # バウンディングボックスを計算
    x_coords = [p[0] for p in pixels]
    y_coords = [p[1] for p in pixels]

    if not x_coords or not y_coords:
        return 0

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    pixel_set = set((p[0], p[1]) for p in pixels)

    # BFS/DFSで外側から到達可能な空白ピクセルをマーク
    visited = set()
    queue = []

    # 境界の空白ピクセルをキューに追加（外側に接している空白）
    for x in range(min_x, max_x + 1):
        for y in [min_y, max_y]:
            if (x, y) not in pixel_set and (x, y) not in visited:
                queue.append((x, y))
                visited.add((x, y))
    for y in range(min_y + 1, max_y):
        for x in [min_x, max_x]:
            if (x, y) not in pixel_set and (x, y) not in visited:
                queue.append((x, y))
                visited.add((x, y))

    # BFSで外側から到達可能な空白を全てマーク
    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (min_x <= nx <= max_x and min_y <= ny <= max_y and
                (nx, ny) not in pixel_set and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))

    # 残りの空白ピクセル（外側から到達不可能）が穴
    hole_pixels = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) not in pixel_set and (x, y) not in visited:
                hole_pixels.add((x, y))

    # 穴の連結成分の数をカウント
    hole_count = 0
    hole_visited = set()

    for hole_pixel in hole_pixels:
        if hole_pixel in hole_visited:
            continue

        # 新しい穴を発見
        hole_count += 1
        queue = [hole_pixel]
        hole_visited.add(hole_pixel)

        # この穴に属するすべてのピクセルをマーク
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in hole_pixels and (nx, ny) not in hole_visited:
                    hole_visited.add((nx, ny))
                    queue.append((nx, ny))

    return hole_count


def _get_object_center(obj: Dict) -> Optional[Tuple[float, float]]:
    """オブジェクトの中心座標を取得

    Args:
        obj: オブジェクト辞書

    Returns:
        (center_x, center_y)のタプル、またはNone（座標情報がない場合）
    """
    x = obj.get('x')
    y = obj.get('y')

    if x is None or y is None:
        position = obj.get('position')
        if position:
            x, y = position[0], position[1]

    if x is None or y is None:
        pixels = obj.get('pixels', [])
        if pixels:
            xs = [px for px, py in pixels]
            ys = [py for px, py in pixels]
            if xs and ys:
                x = (min(xs) + max(xs)) / 2.0
                y = (min(ys) + max(ys)) / 2.0
            else:
                return None
        else:
            return None

    return (float(x), float(y))


def _calculate_distance(obj1: Dict, obj2: Dict) -> float:
    """2つのオブジェクト間のユークリッド距離を計算

    Args:
        obj1: オブジェクト1
        obj2: オブジェクト2

    Returns:
        距離（座標情報がない場合はfloat('inf')）
    """
    center1 = _get_object_center(obj1)
    center2 = _get_object_center(obj2)

    if center1 is None or center2 is None:
        return float('inf')

    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    return np.sqrt(dx * dx + dy * dy)


def _select_nearby_objects_group(
    base_obj: Dict,
    existing_objects: List[Dict],
    duplicate_num: int,
    rng
) -> List[Dict]:
    """基準オブジェクトに近いオブジェクトを選択してグループを形成

    Args:
        base_obj: 基準オブジェクト
        existing_objects: 既存オブジェクトのリスト
        duplicate_num: グループに含めるオブジェクト数（基準オブジェクトを含む）
        rng: 乱数生成器

    Returns:
        選択されたオブジェクトのリスト（基準オブジェクトを含む）
    """
    if duplicate_num <= 1 or len(existing_objects) < duplicate_num:
        return [base_obj] if base_obj in existing_objects else []

    # 基準オブジェクトを除いた既存オブジェクトのリスト
    other_objects = [obj for obj in existing_objects if obj != base_obj]

    if len(other_objects) < duplicate_num - 1:
        return [base_obj] + other_objects

    # 各オブジェクトと基準オブジェクトの距離を計算
    distances = []
    for obj in other_objects:
        dist = _calculate_distance(base_obj, obj)
        distances.append((dist, obj))

    # 距離が近い順にソート
    distances.sort(key=lambda x: x[0])

    # 近いオブジェクトを選択（距離が近いほど確率が高い）
    # 距離の逆数を重みとして使用
    selected = [base_obj]
    remaining = distances.copy()

    for _ in range(duplicate_num - 1):
        if not remaining:
            break

        # 距離の逆数を重みとして使用（近いほど確率が高い）
        # ただし、完全にランダムな要素も加える（多様性のため）
        if len(remaining) <= 3:
            # 残りが少ない場合は、距離が近い順に選択
            selected.append(remaining[0][1])
            remaining.pop(0)
        else:
            # 距離の逆数を重みとして使用（近いほど確率が高い）
            weights = []
            for dist, _ in remaining:
                if dist > 0:
                    weight = 1.0 / (dist + 1.0)  # 距離が0の場合は1.0
                else:
                    weight = 1.0
                weights.append(weight)

            # 重み付きランダム選択
            total_weight = sum(weights)
            if total_weight > 0:
                rand_val = rng.random() * total_weight
                cumulative = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if rand_val <= cumulative:
                        selected.append(remaining[i][1])
                        remaining.pop(i)
                        break
            else:
                # 重みが0の場合はランダムに選択
                idx = rng.randint(0, len(remaining) - 1)
                selected.append(remaining[idx][1])
                remaining.pop(idx)

    return selected


def _get_existing_object_positions(existing_objects: List[Dict]) -> List[Tuple[int, int, int, int]]:
    """既存オブジェクトの位置情報（x, y, width, height）を取得

    Args:
        existing_objects: 既存オブジェクトのリスト

    Returns:
        各オブジェクトの(x, y, width, height)のリスト
    """
    positions = []
    if existing_objects is None:
        return positions
    for obj in existing_objects:
        if not isinstance(obj, dict):
            continue

        # x, y座標を取得
        x = obj.get('x')
        y = obj.get('y')

        # positionフィールドから取得（x, yがない場合）
        if x is None or y is None:
            position = obj.get('position')
            if position:
                x, y = position[0], position[1]

        # 座標情報がない場合はスキップ
        if x is None or y is None:
            continue

        # バウンディングボックスサイズを取得
        pixels = obj.get('pixels', [])
        if not pixels:
            continue

        # ピクセルからバウンディングボックスを計算
        min_px = min(px for px, py in pixels)
        max_px = max(px for px, py in pixels)
        min_py = min(py for px, py in pixels)
        max_py = max(py for px, py in pixels)
        width = max_px - min_px + 1
        height = max_py - min_py + 1

        positions.append((x, y, width, height))

    return positions


def _calculate_max_object_size_for_available_space(
    grid_width: int,
    grid_height: int,
    existing_positions: List[Tuple[int, int, int, int]],
    min_spacing: int = 0,
    default_max_size: int = 10
) -> Tuple[int, int]:
    """利用可能スペースを考慮した最大オブジェクトサイズを計算

    Args:
        grid_width: グリッド幅
        grid_height: グリッド高さ
        existing_positions: 既存オブジェクトの位置リスト
        min_spacing: 最小スペース
        default_max_size: デフォルト最大サイズ

    Returns:
        (max_width, max_height)のタプル
    """
    if not existing_positions:
        # 既存オブジェクトがない場合、グリッドサイズに応じた最大サイズを返す
        max_w = min(default_max_size, grid_width)
        max_h = min(default_max_size, grid_height)
        return max_w, max_h

    # 既存オブジェクトの領域を考慮
    # 既存オブジェクトが占める領域を考慮し、残りスペースから推定
    # 計算コストと精度のバランスを考慮した効率的な方法を使用

    # 既存オブジェクトの総占有面積を計算
    occupied_area = sum(w * h for _, _, w, h in existing_positions)
    total_area = grid_width * grid_height

    # 利用可能面積の推定（min_spacingも考慮）
    available_area = total_area - occupied_area - (len(existing_positions) * min_spacing * 4)
    available_area = max(1, available_area)

    # 利用可能面積から最大サイズを推定（正方形近似を使用）
    max_size = int(np.sqrt(available_area))
    max_size = min(max_size, default_max_size)
    max_size = max(2, max_size)  # 最小2x2

    return max_size, max_size


def generate_objects_from_conditions(
    background_color: int = 0,
    num_objects: int = 1,
    seed: Optional[int] = None,
    existing_objects: Optional[List[Dict]] = None,
    allow_same_color_and_shape: bool = True,
    allow_same_shape: bool = True,
    shape_type: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """条件を受け取ってオブジェクトのリストを生成

    Args:
        background_color: 背景色
        num_objects: 生成するオブジェクト数
        seed: 乱数シード
        existing_objects: 既存のオブジェクトリスト（背景色以外の色を確認するために使用）
        allow_same_color_and_shape: 同じ色と形状のオブジェクトを許可（デフォルト: True）
        allow_same_shape: 同じ形状のオブジェクトを許可（デフォルト: True）
        shape_type: 形状タイプ（None, 'rectangle', 'line', 'hollow_rectangle', 'cross', 't_shape', 'diagonal_45', 'u_shape', 'h_shape', 'z_shape', 'noise', 'random_pattern'など）
            - None: 制約なし（既存ロジック）
            - 'rectangle': 矩形のみ
            - 'line': 線のみ
            - 'hollow_rectangle': 中空矩形のみ
            - 'cross': 十字形
            - 't_shape': T字形
            - 'diagonal_45': 45度斜め線
            - 'u_shape', 'h_shape', 'z_shape': それぞれの形状
            - 'noise': ノイズパターン（グリッド全体に1ピクセルをランダム配置、連結性や重複を気にしない）
            - 'random_pattern': ランダムパターン（指定サイズ内でランダムにピクセルで埋め、連結性を保証）
            - その他: builder._generate_{shape_type}メソッドが存在する場合は自動的に使用
        **kwargs: その他の条件（将来の拡張用）
            - object_spec: オブジェクト仕様（Dict形式）
            - complexity: オブジェクト生成の複雑度（互換性のため残すが、使用しない）
            - grid_size: グリッドサイズ（タプル形式、適応的生成に使用）
            - total_num_objects: 作成予定の総オブジェクト数（num_objectsと組み合わせてサイズ調整に使用）
                - 指定されていない場合は、num_objectsをtotal_num_objectsとして使用
                - 指定された場合：今回作成するオブジェクト数の割合 = num_objects / total_num_objects
                - 割合 × 総面積 × 0.90で今回作成するオブジェクト全体が使える面積を計算
                - 各オブジェクトごとにランダムに面積を割り当て、1から面積までの乱数でサイズを決定
            - connectivity_constraint: 連結性制約（None, '4-connected', '8-connected'）
                - None: デフォルト（50%で8連結、50%で4連結）
                - '4-connected': 4連結のみ
                - '8-connected': 50%で8連結、50%で4連結（デフォルトと同じ）
            - allow_holes: 穴ありオブジェクトを優先的に作成（100%で作成、デフォルト: False）
            - min_bbox_size: 最小バウンディングボックスサイズ（幅=高さ、0の場合は制約なし、デフォルト: 0）
            - max_bbox_size: 最大バウンディングボックスサイズ（幅=高さ、0の場合は制約なし、デフォルト: 0）
            - min_density: 最小密度（0.0-1.0、0.0の場合は制約なし、デフォルト: 0.0）
            - max_density: 最大密度（0.0-1.0、1.0の場合は制約なし、デフォルト: 1.0）
            - symmetry_constraint: 対称性制約（None, 'none', 'vertical', 'horizontal', 'both'）
                - None/'none': 制約なし（デフォルト）
                - 'vertical': 垂直対称（左右対称、Y軸）
                - 'horizontal': 水平対称（上下対称、X軸）
                - 'both': 両対称（垂直と水平の両方）
            - duplicate_mode: 複製制御（None, 'exact', 'color_only', 'shape_only', 'forbidden'）
                - None: 既存の統計ベースロジック（SAME_COLOR_AND_SHAPE_PROBABILITY使用）
                - 'exact': 既存オブジェクトと完全一致を強制（色・形状・位置）
                - 'color_only': 既存オブジェクトの色のみ複製
                - 'shape_only': 既存オブジェクトの形状のみ複製
                - 'forbidden': 既存オブジェクトの複製を禁止
            - duplicate_num: グループ複製用のオブジェクト数（デフォルト: 1 = 単一オブジェクト複製）
                - 1: 単一オブジェクト複製（通常の動作）
                - 2以上: グループ複製（基準オブジェクトに近いオブジェクトを選択してグループを形成）
                - duplicate_mode == 'exact'の場合: グループ全体を同じ色で複製（元の色を保持）
                - duplicate_mode == 'shape_only'の場合: グループ内の各オブジェクトの形状は同じだが、色は個別に決定
                - duplicate_mode == Noneの場合: 統計ベース（元の色を保持）
            - min_spacing: オブジェクト間の最小スペース（ピクセル単位、0の場合は制約なし、デフォルト: 0）
                - この値は配置時に使用されるため、set_positionやbuild_gridに渡される
                - バウンディングボックス間の距離（マンハッタン距離の最大値）がmin_spacing以上になるように配置される

    Returns:
        オブジェクトのリスト
    """
    gen_start_time = time.time()
    num_objects_param = num_objects

    builder = CoreObjectBuilder(seed=seed)
    rng = builder.rng
    result_objects = []

    # duplicate_modeを事前に取得（形状シグネチャの取得が必要かどうかを判定するため）
    duplicate_mode = kwargs.get('duplicate_mode')

    # object_colorsがkwargsに指定されている場合は、ループ内で色を選択する
    # object_colorsが指定されていない場合は、背景色以外からランダムに選択（統計は既にobject_colors決定時に適用済み）
    object_colors = kwargs.get('object_colors')

    # 既存のオブジェクトから色を取得（object_colorsが指定されていない場合のみ）
    # 最適化: object_colorsが指定されている場合は色の抽出をスキップ
    existing_colors: Set[int] = set()
    needs_color_extraction = (object_colors is None or len(object_colors) == 0)

    if existing_objects and len(existing_objects) > 0 and needs_color_extraction:
        for obj in existing_objects:
            if isinstance(obj, dict) and 'color' in obj:
                existing_colors.add(obj['color'])
            elif hasattr(obj, 'color'):
                existing_colors.add(obj.color)

    # 最適化: 形状シグネチャは必要時に1個ずつ取得（事前に全て取得しない）
    # 複製が決定されたときに、1個のオブジェクトを選んで形状シグネチャを取得する
    # これにより、形状シグネチャの取得回数を大幅に削減（10回 → 1-2回程度）

    # duplicate_numを取得（グループ複製の場合は複数必要）
    duplicate_num = kwargs.get('duplicate_num', 1)

    # 最適化: 既存オブジェクトのフィルタリング結果をキャッシュ（O(n × m) → O(n + m)）
    # ループの外で一度だけ実行して、ループ内ではキャッシュを使用
    valid_existing_objects_for_duplication = None
    if existing_objects and len(existing_objects) > 0:
        valid_existing_objects_for_duplication = [
            obj for obj in existing_objects
            if isinstance(obj, dict) and obj.get('pixels') and isinstance(obj.get('pixels'), list) and len(obj.get('pixels', [])) > 0
        ]

    # 最適化: 事前計算可能な値をキャッシュ（結果を変えずに計算を削減）
    # available_colorsの事前計算（object_colorsが指定されていない場合のみ）
    precomputed_available_colors = None
    if not object_colors:
        precomputed_available_colors = [c for c in range(10) if c != background_color]

    # existing_colors_listの事前計算（duplicate_mode='shape_only'で使用）
    precomputed_existing_colors_list = None
    if existing_colors and background_color is not None:
        precomputed_existing_colors_list = [c for c in existing_colors if c != background_color]

    def get_valid_object_for_duplication(existing_objs, max_size=None, target_area=None):
        """既存オブジェクトから複製可能なオブジェクトを1個取得（面積ベース制約対応）

        Args:
            existing_objs: 既存オブジェクトのリスト（combined_existing_objectsを渡すことを想定）
            max_size: 最大サイズ制約（Noneの場合は制約なし、後方互換性のため残す）
            target_area: 目標面積（指定された場合、この面積以下のオブジェクトを優先的に選択）

        Returns:
            pixelsが存在するオブジェクト、またはNone
        """
        # existing_objsから有効なオブジェクトをフィルタリング（キャッシュではなく、渡されたリストを使用）
        if existing_objs is None or len(existing_objs) == 0:
            return None

        valid_objects = [
            obj for obj in existing_objs
            if isinstance(obj, dict) and obj.get('pixels') and isinstance(obj.get('pixels'), list) and len(obj.get('pixels', [])) > 0
        ]

        if len(valid_objects) == 0:
            return None

        # 面積ベースの制約を優先（target_areaが指定されている場合）
        if target_area is not None and target_area > 0:
            # 目標面積以下のオブジェクトを優先的に選択
            area_constrained_objects = []
            for obj in valid_objects:
                obj_area = obj.get('area', 0)
                if obj_area > 0 and obj_area <= target_area:
                    area_constrained_objects.append(obj)

            # 面積制約に収まるオブジェクトがある場合は、その中から選択
            if len(area_constrained_objects) > 0:
                return rng.choice(area_constrained_objects)
            # 面積制約に収まるオブジェクトがない場合は、すべてのオブジェクトから選択（フォールバック）

        # サイズ制約がある場合、そのサイズに収まるオブジェクトを優先的に選択（後方互換性のため）
        if max_size is not None and max_size > 0:
            # オブジェクトのサイズを計算（width × height または area）
            size_constrained_objects = []
            for obj in valid_objects:
                obj_width = obj.get('width', 0)
                obj_height = obj.get('height', 0)
                obj_area = obj.get('area', 0)

                # サイズ判定: widthとheightの最大値がmax_size以下、またはareaがmax_size^2以下
                if obj_width > 0 and obj_height > 0:
                    obj_max_dim = max(obj_width, obj_height)
                    if obj_max_dim <= max_size:
                        size_constrained_objects.append(obj)
                elif obj_area > 0:
                    # areaから推定される最大次元がmax_size以下
                    estimated_max_dim = int(obj_area ** 0.5) + 1
                    if estimated_max_dim <= max_size:
                        size_constrained_objects.append(obj)

            # サイズ制約に収まるオブジェクトがある場合は、その中から選択
            if len(size_constrained_objects) > 0:
                return rng.choice(size_constrained_objects)
            # サイズ制約に収まるオブジェクトがない場合は、すべてのオブジェクトから選択（フォールバック）

        # ランダムに1個を選ぶ（フィルタリング済みリストから選択）
        return rng.choice(valid_objects)

    # グループ複製用のヘルパー関数（複数のオブジェクトを取得）
    def get_valid_objects_for_group_duplication(existing_objs, needed_count):
        """グループ複製用に、複製可能なオブジェクトを複数個取得（キャッシュ版）

        Args:
            existing_objs: 既存オブジェクトのリスト（実際にはキャッシュを使用）
            needed_count: 必要なオブジェクト数

        Returns:
            pixelsが存在するオブジェクトのリスト
        """
        # 最適化: キャッシュを使用（フィルタリング済み）
        if valid_existing_objects_for_duplication is None or len(valid_existing_objects_for_duplication) == 0 or needed_count <= 0:
            return []

        # 必要な数だけランダムに選択（重複なし）
        needed_count = min(needed_count, len(valid_existing_objects_for_duplication))
        if needed_count > 0 and len(valid_existing_objects_for_duplication) > 0:
            selected_objects = rng.sample(valid_existing_objects_for_duplication, needed_count)
        else:
            selected_objects = []

        return selected_objects

    if not object_colors:
        # 背景色以外の色を選択（0-9の範囲から背景色を除く）
        available_colors = [c for c in range(10) if c != background_color]
        if available_colors:
            # ランダムに1色を選択（builder.rngを使用）
            selected_color = rng.choice(available_colors)
            object_colors = [selected_color]
        else:
            # 背景色が0以外の場合は0を、それ以外の場合は1を使用
            object_colors = [1 if background_color == 0 else 0]

    # object_colorsがkwargsで指定されている場合は、それで上書き（上記の条件分岐で設定されていない場合）
    object_colors_from_kwargs = kwargs.get('object_colors')
    if object_colors_from_kwargs:
        object_colors = object_colors_from_kwargs
    # オブジェクト仕様を取得（将来の拡張用）
    object_spec = kwargs.get('object_spec')
    # 複雑度（オブジェクト生成では使用しないが、互換性のため残す）
    complexity = kwargs.get('complexity', None)
    # shape_typeを取得（引数で指定されている場合は優先、kwargsでも指定可能）
    if shape_type is None:
        shape_type = kwargs.get('shape_type')
    grid_size = kwargs.get('grid_size')
    connectivity_constraint = kwargs.get('connectivity_constraint')
    allow_holes = kwargs.get('allow_holes', False)
    min_bbox_size = kwargs.get('min_bbox_size', 0)
    max_bbox_size = kwargs.get('max_bbox_size', 0)
    min_density = kwargs.get('min_density', 0.0)
    max_density = kwargs.get('max_density', 1.0)
    symmetry_constraint = kwargs.get('symmetry_constraint')
    # duplicate_modeは上記で既に取得済み（419行目）
    # duplicate_numは上記で既に取得済み（443行目）
    min_spacing = kwargs.get('min_spacing', 0)

    # 既存オブジェクトの位置情報を取得
    existing_positions = []
    if existing_objects and grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
        existing_positions = _get_existing_object_positions(existing_objects)

    # 早期リターンチェック
    if num_objects <= 0:
        return result_objects

    # グリッドサイズを取得
    grid_width = None
    grid_height = None
    if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
        grid_width, grid_height = grid_size[0], grid_size[1]

    # 連結性を決定（connectivity_constraintに基づく）
    def _determine_connectivity(builder_rng, constraint):
        """連結性を決定"""
        if constraint == '4-connected':
            return 4
        elif constraint == '8-connected':
            # 50%の確率で8連結
            return 8 if builder_rng.random() < 0.5 else 4
        else:  # None: デフォルト（50%で8連結、50%で4連結）
            return 8 if builder_rng.random() < 0.5 else 4

    # グリッドサイズに応じたデフォルトサイズ範囲を計算
    def _calculate_default_size_range(grid_w, grid_h, default_min=2, default_max=6):
        """グリッドサイズに応じたデフォルトサイズ範囲を計算"""
        if grid_w is None or grid_h is None:
            return default_min, default_max

        # グリッドサイズに応じたオブジェクトサイズを決定（ARC-AGI2統計に合わせて調整）
        # 多様性を保つため、最大サイズを緩和
        min_grid_dim = min(grid_w, grid_h)
        # 最小サイズ: グリッドサイズの6%以上、ただし最小2（8%→6%に緩和して多様性を向上）
        adjusted_min = max(default_min, int(min_grid_dim * 0.06))
        # 最大サイズ: グリッドサイズの50%以下、ただし最小値以上（40%→50%に緩和して多様性を向上）
        adjusted_max = max(adjusted_min, min(default_max, int(min_grid_dim * 0.50)))

        # サイズが大きすぎる場合は制限
        if min_grid_dim < 5:
            adjusted_max = min(adjusted_max, min_grid_dim - 1)

        return adjusted_min, adjusted_max

    def _calculate_size_with_probability_distribution(
        grid_w, grid_h, total_num_objects, rng,
        min_size=1, max_size=None, shape_type=None
    ):
        """確率分布に基づいてオブジェクトサイズを決定（shape_type対応版、面積制限対応）

        Args:
            grid_w: グリッド幅
            grid_h: グリッド高さ
            total_num_objects: 総オブジェクト数
            rng: 乱数生成器
            min_size: 最小サイズ（デフォルト: 1）
            max_size: 最大サイズ（Noneの場合はグリッドサイズから計算）
            shape_type: 形状タイプ（'rectangle', 'line'など、確率分布の調整に使用）

        Returns:
            確率分布に基づいて決定されたオブジェクトサイズ
        """
        import random
        if rng is None:
            rng = random.Random()

        # 最大サイズを決定（グリッドサイズを考慮）
        if max_size is None:
            if grid_w is not None and grid_h is not None:
                max_size = min(grid_w, grid_h) - 1
            else:
                max_size = 10  # デフォルト
        # max_sizeがNoneでないことを保証
        if max_size is None:
            max_size = 10  # フォールバック
        max_size = max(max_size, min_size)
        # 最終的なmax_sizeがNoneでないことを保証
        if max_size is None:
            max_size = 1

        # shape_typeに応じた確率分布の重みを調整（1ピクセルを下げ、2-5ピクセルを上げる）
        # rectangle系は少し大きめのサイズも許容、line系は小さめを優先
        size_weights = {}  # デフォルトを空辞書に初期化
        # ARC-AGI2統計に基づく確率分布（ドキュメント参照）
        # 1ピクセルを下げ、2-5ピクセルの確率を上げる
        # 基本確率: 1ピクセル約45-50%、2-5ピクセル約35-40%、6ピクセル以上約5-10%
        if shape_type in ['rectangle', 'hollow_rectangle', 'square']:
            # rectangle系: 1ピクセルをさらに下げ、2-5ピクセルをさらに上げる（密度を下げる）
            size_weights = {1: 0.30, 2: 0.32, 3: 0.22, 4: 0.13, 5: 0.02, 6: 0.01}
        elif shape_type in ['line', 'diagonal_45', 'diagonal']:
            # line系: 1ピクセルをさらに下げ、2-5ピクセルをさらに上げる（密度を下げる）
            size_weights = {1: 0.38, 2: 0.34, 3: 0.18, 4: 0.08, 5: 0.02, 6: 0.00}
        else:
            # その他: 1ピクセルをさらに下げ、2-5ピクセルをさらに上げる（密度を下げる）
            size_weights = {1: 0.32, 2: 0.32, 3: 0.20, 4: 0.13, 5: 0.02, 6: 0.01}

        # size_weightsがNoneでないことを保証
        if size_weights is None:
            size_weights = {1: 0.52, 2: 0.24, 3: 0.14, 4: 0.06, 5: 0.03, 6: 0.01}

        # グリッドサイズに応じて確率分布を調整（密度をさらに下げるため、1ピクセルを優先）
        if grid_w is not None and grid_h is not None:
            min_grid_dim = min(grid_w, grid_h)
            # グリッドサイズに応じて確率分布を調整（多様性を保つため、大きなオブジェクトの制限を緩和）
            if min_grid_dim <= 10:
                size_weights[1] = min(0.70, size_weights[1] * 1.25)
                size_weights[6] *= 0.5  # 大きなオブジェクトの制限を緩和（0.3 → 0.5）
            elif min_grid_dim <= 20:
                size_weights[1] = min(0.65, size_weights[1] * 1.2)
                size_weights[6] *= 0.7  # 大きなオブジェクトの制限を緩和（0.5 → 0.7）
            # 大きいグリッドでは大きなオブジェクトも許容（多様性を保つため）
            elif min_grid_dim >= 30:
                size_weights[1] = min(0.60, size_weights[1] * 1.1)
                size_weights[6] *= 0.9  # 大きなオブジェクトの制限を緩和（0.7 → 0.9）

        # total_num_objectsに応じて確率分布を調整（_calculate_size_from_areaと同じロジック）
        if total_num_objects is not None and total_num_objects > 10:
            small_size_max = min(5, max_size)

            if total_num_objects >= 51:
                # 51個以上: ARC-AGI2統計に基づく（1ピクセル58.6%）、1ピクセルをさらに下げ、2-5ピクセルをさらに上げる（密度を下げる）
                rand = rng.random()
                if rand < 0.40:  # ARC-AGI2統計: 58.6% → 40%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    size = 1
                elif rand < 0.92:  # 2-5ピクセル: 52%（合計92%、45% → 52%に増加）
                    # 2-5ピクセル
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    size = min(size, small_size_max) if small_size_max >= 2 else 1
                else:
                    # 6ピクセル以上（重み付きランダム選択）
                    if max_size is not None and max_size >= 6:
                        try:
                            sizes = list(range(6, max_size + 1))
                            if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                                weights = [size_weights.get(s, 0.01) for s in sizes]
                                if sum(weights) > 0:
                                    size = rng.choices(sizes, weights=weights)[0]
                                else:
                                    size = rng.randint(6, max_size)
                            else:
                                size = 1
                        except (TypeError, ValueError):
                            size = 1
                    else:
                        size = 1
            elif total_num_objects >= 31:
                # 31-50個: ARC-AGI2統計に基づく（1ピクセル48.0%）、2-5ピクセルの確率を上げる
                rand = rng.random()
                if rand < 0.45:  # ARC-AGI2統計: 48.0% → 45%に調整（1ピクセルを下げ、2-5ピクセルを上げる）
                    size = 1
                elif rand < 0.82:  # 2-5ピクセル: 37%（合計82%、33% → 37%に増加）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    size = min(size, small_size_max) if small_size_max >= 2 else 1
                else:
                    # 6ピクセル以上（重み付きランダム選択）
                    if max_size is not None and max_size >= 6:
                        try:
                            sizes = list(range(6, max_size + 1))
                            if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                                weights = [size_weights.get(s, 0.01) for s in sizes]
                                if sum(weights) > 0:
                                    size = rng.choices(sizes, weights=weights)[0]
                                else:
                                    size = rng.randint(6, max_size)
                            else:
                                size = 1
                        except (TypeError, ValueError):
                            size = 1
                    else:
                        size = 1
            elif total_num_objects >= 21:
                # 21-30個: ARC-AGI2統計に基づく（1ピクセル44.8%）、2-5ピクセルの確率を上げる
                rand = rng.random()
                if rand < 0.42:  # ARC-AGI2統計: 44.8% → 42%に調整（1ピクセルを下げ、2-5ピクセルを上げる）
                    size = 1
                elif rand < 0.80:  # 2-5ピクセル: 38%（合計80%、37% → 38%に増加）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    size = min(size, small_size_max) if small_size_max >= 2 else 1
                else:
                    # 6ピクセル以上（重み付きランダム選択）
                    if max_size is not None and max_size >= 6:
                        try:
                            sizes = list(range(6, max_size + 1))
                            if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                                weights = [size_weights.get(s, 0.01) for s in sizes]
                                if sum(weights) > 0:
                                    size = rng.choices(sizes, weights=weights)[0]
                                else:
                                    size = rng.randint(6, max_size)
                            else:
                                size = 1
                        except (TypeError, ValueError):
                            size = 1
                    else:
                        size = 1
            elif total_num_objects >= 11:
                # 11-20個: ARC-AGI2統計に基づく（1ピクセル39.3%）、2-5ピクセルの確率を上げる
                rand = rng.random()
                if rand < 0.36:  # ARC-AGI2統計: 39.3% → 36%に調整（1ピクセルを下げ、2-5ピクセルを上げる）
                    size = 1
                elif rand < 0.78:  # 2-5ピクセル: 42%（合計78%、40% → 42%に増加）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    size = min(size, small_size_max) if small_size_max >= 2 else 1
                else:
                    # 6ピクセル以上（重み付きランダム選択）
                    if max_size is not None and max_size >= 6:
                        try:
                            sizes = list(range(6, max_size + 1))
                            if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                                weights = [size_weights.get(s, 0.01) for s in sizes]
                                if sum(weights) > 0:
                                    size = rng.choices(sizes, weights=weights)[0]
                                else:
                                    size = rng.randint(6, max_size)
                            else:
                                size = 1
                        except (TypeError, ValueError):
                            size = 1
                    else:
                        size = 1
            else:
                # 10個以下: ARC-AGI2統計に基づく（1ピクセル55.0%）、密度を下げるため1ピクセルを少し増やす
                rand = rng.random()
                if rand < 0.60:  # ARC-AGI2統計: 55.0% → 60%に微調整（密度を下げるため）
                    size = 1
                elif rand < 0.90:  # 2-5ピクセル: 30%（合計90%）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    size = min(size, min(5, max_size)) if min(5, max_size) >= 2 else 1
                else:
                    # 6ピクセル以上（重み付きランダム選択）
                    if max_size is not None and max_size >= 6:
                        try:
                            sizes = list(range(6, max_size + 1))
                            if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                                weights = [size_weights.get(s, 0.01) for s in sizes]
                                if sum(weights) > 0:
                                    size = rng.choices(sizes, weights=weights)[0]
                                else:
                                    size = rng.randint(6, max_size)
                            else:
                                size = 1
                        except (TypeError, ValueError):
                            size = 1
                    else:
                        size = 1
        else:
            # total_num_objectsが指定されていない場合: ARC-AGI2統計に基づく（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）（密度を下げる）
            rand = rng.random()
            if rand < 0.30:  # ARC-AGI2統計: 50% → 30%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                size = 1
            elif rand < 0.90:  # 2-5ピクセル: 60%（合計90%、53% → 60%に増加）
                rand_size = rng.random()
                if rand_size < 0.407:
                    size = 2
                elif rand_size < 0.603:
                    size = 3
                elif rand_size < 0.898:
                    size = 4
                else:
                    size = 5
                size = min(size, min(5, max_size)) if min(5, max_size) >= 2 else 1
            else:
                # 6ピクセル以上（重み付きランダム選択）
                if max_size is not None and max_size >= 6:
                    try:
                        sizes = list(range(6, max_size + 1))
                        if sizes and size_weights is not None:  # sizesが空でなく、size_weightsがNoneでないことを確認
                            weights = [size_weights.get(s, 0.01) for s in sizes]
                            if sum(weights) > 0:
                                size = rng.choices(sizes, weights=weights)[0]
                            else:
                                size = rng.randint(6, max_size)
                        else:
                            size = 1
                    except (TypeError, ValueError):
                        size = 1
                else:
                    size = 1

        # 範囲チェック
        size = max(min_size, min(size, max_size))
        return size

    # num_objects、total_num_objects、grid_sizeに基づいて各オブジェクトごとの面積を計算
    def _calculate_object_areas_from_total(
        grid_w, grid_h, num_objects, total_num_objects, rng=None, area_ratio=None
    ):
        """num_objects、total_num_objects、grid_sizeに基づいて各オブジェクトごとの面積を計算

        Args:
            grid_w: グリッド幅
            grid_h: グリッド高さ
            num_objects: 今回作成するオブジェクト数
            total_num_objects: 作成予定の総オブジェクト数
            rng: 乱数生成器（Noneの場合はrandomを使用）
            area_ratio: 利用可能面積の割合（Noneの場合は自動決定、0.08-0.95の範囲）

        Returns:
            各オブジェクトごとの面積のリスト、またはNone（不適切な場合）
        """
        import random
        import numpy as np
        if rng is None:
            rng = random.Random()
            # numpyの乱数生成器も初期化（rngがNoneの場合）
            np_rng = np.random.default_rng()
        else:
            # rngが提供されている場合、シードを設定してnumpyの乱数生成器を作成
            # random.Randomのシードを取得してnumpyに設定
            try:
                rng_state = rng.getstate()
                # getstate()はタプルを返す: (version, state_tuple, gauss_next)
                # state_tupleの最初の要素をシードとして使用
                seed_value = rng_state[1][0] if len(rng_state) > 1 and len(rng_state[1]) > 0 else None
                np_rng = np.random.default_rng(seed_value)
            except:
                # フォールバック: rngから直接シードを取得できない場合は、ランダムなシードを使用
                np_rng = np.random.default_rng()

        if grid_w is None or grid_h is None or num_objects is None or num_objects <= 0:
            return None
        if total_num_objects is None or total_num_objects <= 0:
            return None

        # グリッドの総面積
        total_area = grid_w * grid_h

        # 今回作成するオブジェクト数の割合（%）
        ratio = num_objects / total_num_objects

        # ARC-AGI2統計に基づいて利用可能面積の割合を決定（0.08から0.95までの範囲）
        # area_ratioが指定されていない場合のみ、自動決定
        if area_ratio is None:
            # ARC-AGI2のオブジェクトピクセル比: 平均=0.291, 標準偏差=0.196
            # 利用可能面積の割合を多様化し、統計に合わせて調整
            # 平均値を下げるため、beta分布のパラメータを調整: alpha=0.2, beta=2.5で平均≈0.074 → 0.08 + 0.074 * 0.87 ≈ 0.144
            # これにより、0.08-0.95範囲で平均約0.144、より低い密度を実現
            beta_sample = np_rng.beta(0.2, 2.5)
            area_ratio = 0.08 + beta_sample * 0.87  # 0.08-0.95範囲にスケール
        # area_ratioが指定されている場合は、その値を使用（2段階生成で全体で1つのarea_ratioを共有する場合）

        # 割合に総面積とarea_ratioをかける
        available_area_for_this_batch = round(total_area * ratio * area_ratio)

        # 最小1にする
        available_area_for_this_batch = max(1, available_area_for_this_batch)

        # 適切性チェック
        # available_area_for_this_batch < num_objectsの場合、各オブジェクトに最低1ピクセルを割り当てる
        # これにより、面積ベースの制約を維持しつつ、オブジェクト生成を続行できる
        if available_area_for_this_batch < num_objects:
            # 各オブジェクトに1ピクセルを割り当て（面積ベースの制約を維持）
            object_areas = [1] * num_objects
            return object_areas

        # ランダムな割合で今回作成するオブジェクト数に分割
        # 各オブジェクトにランダムに面積を割り当て（最小1）
        object_areas = []
        remaining_area = available_area_for_this_batch

        # 最後の1つ以外はランダムに割り当て（残りを最後の1つに確保）
        for i in range(num_objects - 1):
            # 残りのオブジェクト数
            remaining_objects = num_objects - i
            # 残りのオブジェクトすべてに最低1を確保する必要がある
            min_for_this = 1
            max_for_this = remaining_area - (remaining_objects - 1)

            if max_for_this < min_for_this:
                max_for_this = min_for_this

            # ランダムに割り当て
            area = _safe_randint(rng, min_for_this, max_for_this, default=min_for_this)
            object_areas.append(area)
            remaining_area -= area

        # 最後の1つは残り全部
        object_areas.append(remaining_area)

        # 小さい順にソート（小さいオブジェクトから生成することで、複製時に適切なサイズのオブジェクトが選択されやすくなる）
        object_areas.sort()

        return object_areas

    # 面積からオブジェクトサイズ（1ピクセルから面積まで）を計算
    def _calculate_size_from_area(area, grid_w, grid_h, rng=None, total_num_objects=None):
        """面積からオブジェクトサイズを計算（1ピクセルから面積まで）

        ARC-AGI2の分析結果に基づき、オブジェクト数が多い場合に小さいサイズが
        選ばれる確率を上げる。

        Args:
            area: オブジェクトの使用可能面積
            grid_w: グリッド幅
            grid_h: グリッド高さ
            rng: 乱数生成器
            total_num_objects: 総オブジェクト数（Noneの場合は一様分布を使用）

        Returns:
            オブジェクトサイズ（幅と高さの平均、正方形近似を使用）
        """
        import random
        if rng is None:
            rng = random.Random()

        if area <= 0:
            return 1

        # 1ピクセルから面積までの乱数でサイズを決定
        # 面積からサイズを推定（正方形近似と長方形も考慮して多様性を向上）
        # 正方形近似: sqrt(area)
        # 長方形も考慮: 面積が大きい場合、より大きなサイズも可能
        # 例: 面積16の場合、4x4だけでなく、3x6や2x8も可能
        # より柔軟に: 面積の1/3から面積までの範囲でサイズを決定可能
        max_size_from_area_square = int(np.sqrt(area))
        # 長方形を考慮しない（密度を下げるため）
        # 面積20の場合、10x2=20や5x4=20は可能だが、10x10=100は不可能
        # 長方形を考慮すると、実際の面積が割り当てられた面積を超えてしまう可能性がある
        # より保守的に: 正方形近似のみを使用（width * height <= areaを保証）
        max_size_from_area = max_size_from_area_square

        # グリッドサイズを超えないようにする
        max_size = min(max_size_from_area, min(grid_w, grid_h) - 1)
        max_size = max(max_size, 1)  # 最低でも1

        # オブジェクト数に基づいてサイズ選択の確率を調整
        # ARC-AGI2の分析結果:
        # - 51+個: 1ピクセルが58.6%、小さいオブジェクト（1-5ピクセル）が86.5%
        # - 31-50個: 1ピクセルが48.0%、小さいオブジェクトが71.4%
        # - 21-30個: 1ピクセルが44.8%、小さいオブジェクトが69.0%
        # - 11-20個: 1ピクセルが39.3%、小さいオブジェクトが64.5%
        # - 6-10個: 1ピクセルが34.7%、小さいオブジェクトが53.0%
        # - 2-5個: 1ピクセルが6.1%、小さいオブジェクトが30.8%
        if total_num_objects is not None and total_num_objects > 10:
            # 小さいオブジェクトの範囲を定義（1-5ピクセル）
            small_size_max = min(5, max_size)

            if total_num_objects >= 51:
                # 51個以上: 1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = rng.random()
                if rand < 0.40:  # ARC-AGI2統計: 58.6% → 40%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    # 1ピクセルを優先
                    size = 1
                elif rand < 0.92:  # 2-5ピクセル: 52%（合計92%、45% → 52%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく: 2px=40.7%, 3px=19.6%, 4px=29.5%, 5px=10.2%）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:  # 40.7% + 19.6%
                        size = 3
                    elif rand_size < 0.898:  # 40.7% + 19.6% + 29.5%
                        size = 4
                    else:
                        size = 5
                    # 範囲チェック
                    if size > small_size_max:
                        size = small_size_max if small_size_max >= 2 else 1
                    elif size < 2:
                        size = 2
                else:
                    # 6ピクセル以上
                    size = rng.randint(small_size_max + 1, max_size) if small_size_max < max_size else rng.randint(1, max_size)
            elif total_num_objects >= 31:
                # 31-50個: 1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = rng.random()
                if rand < 0.30:  # ARC-AGI2統計: 48.0% → 30%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    size = 1
                elif rand < 0.90:  # 2-5ピクセル: 60%（合計90%、53% → 60%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    if size > small_size_max:
                        size = small_size_max if small_size_max >= 2 else 1
                    elif size < 2:
                        size = 2
                else:
                    size = rng.randint(small_size_max + 1, max_size) if small_size_max < max_size else rng.randint(1, max_size)
            elif total_num_objects >= 21:
                # 21-30個: 1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = rng.random()
                if rand < 0.28:  # ARC-AGI2統計: 44.8% → 28%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    size = 1
                elif rand < 0.90:  # 2-5ピクセル: 62%（合計90%、55% → 62%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    if size > small_size_max:
                        size = small_size_max if small_size_max >= 2 else 1
                    elif size < 2:
                        size = 2
                else:
                    # 15%の確率で6ピクセル以上（25% → 15%に減少）
                    size = rng.randint(small_size_max + 1, max_size) if small_size_max < max_size else rng.randint(1, max_size)
            elif total_num_objects >= 11:
                # 11-20個: 1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = rng.random()
                if rand < 0.25:  # ARC-AGI2統計: 39.3% → 25%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    size = 1
                elif rand < 0.88:  # 2-5ピクセル: 63%（合計88%、57% → 63%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    if size > small_size_max:
                        size = small_size_max if small_size_max >= 2 else 1
                    elif size < 2:
                        size = 2
                else:
                    # 15%の確率で6ピクセル以上（25% → 15%に減少）
                    size = rng.randint(small_size_max + 1, max_size) if small_size_max < max_size else rng.randint(1, max_size)
            else:
                # 10個以下: ARC-AGI2の分布に基づく（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）（密度を下げる）
                rand = rng.random()
                if rand < 0.18:  # 18%の確率で1ピクセル（20% → 18%にさらに下げ、2-5ピクセルをさらに上げる）
                    size = 1
                elif rand < 0.88:  # 70%の確率で小さいサイズ（2-5px）（65% → 70%にさらに増加）
                    # 小さいサイズ内での分布（ARC-AGI2に基づく）
                    rand_size = rng.random()
                    if rand_size < 0.407:
                        size = 2
                    elif rand_size < 0.603:
                        size = 3
                    elif rand_size < 0.898:
                        size = 4
                    else:
                        size = 5
                    if size > min(5, max_size):
                        size = min(5, max_size) if min(5, max_size) >= 2 else 1
                    elif size < 2:
                        size = 2
                else:
                    # 10%の確率で6ピクセル以上（8% → 10%に増加して多様性を向上）
                    size = rng.randint(6, max_size) if max_size >= 6 else rng.randint(1, max_size)
        else:
            # total_num_objectsが指定されていない場合も、ARC-AGI2の分布に基づく
            # 全体の傾向: 1px=52.9%, 2-5px=27.5%, 6+=19.6%
            rand = rng.random()
            if rand < 0.529:  # 52.9%の確率で1ピクセル
                size = 1
            elif rand < 0.804:  # 27.5%の確率で小さいサイズ（2-5px）
                rand_size = rng.random()
                if rand_size < 0.407:
                    size = 2
                elif rand_size < 0.603:
                    size = 3
                elif rand_size < 0.898:
                    size = 4
                else:
                    size = 5
                if size > min(5, max_size):
                    size = min(5, max_size) if min(5, max_size) >= 2 else 1
                elif size < 2:
                    size = 2
            else:
                # 19.6%の確率で6ピクセル以上
                size = rng.randint(6, max_size) if max_size >= 6 else rng.randint(1, max_size)

        return size

    # バウンディングボックスサイズを決定
    def _safe_randint(rng, min_val, max_val, default=None):
        """安全なrandint呼び出し（min <= maxをチェック）

        Args:
            rng: 乱数ジェネレータ
            min_val: 最小値
            max_val: 最大値
            default: min > maxの場合のデフォルト値（Noneの場合はmin_valを使用）

        Returns:
            ランダムな整数値、またはデフォルト値
        """
        if min_val > max_val:
            return default if default is not None else min_val
        return rng.randint(min_val, max_val)

    def _determine_bbox_size(builder_rng, min_size, max_size, default_min=2, default_max=6, grid_size=None):
        """バウンディングボックスサイズを決定（幅=高さ）"""
        # グリッドサイズが指定されている場合、デフォルト範囲を調整
        if grid_size is not None:
            grid_w, grid_h = grid_size
            adjusted_min, adjusted_max = _calculate_default_size_range(grid_w, grid_h, default_min, default_max)
            default_min, default_max = adjusted_min, adjusted_max

        if min_size > 0 or max_size > 0:
            if min_size > 0 and max_size > 0:
                if min_size == max_size:
                    return min_size
                if min_size > max_size:
                    # min_sizeがmax_sizeより大きい場合は、min_sizeを返す（安全なフォールバック）
                    return min_size
                return _safe_randint(builder_rng, min_size, max_size, default=min_size)
            elif min_size > 0:
                if min_size > default_max:
                    return min_size
                return _safe_randint(builder_rng, min_size, default_max, default=min_size)
            elif max_size > 0:
                if default_min > max_size:
                    return max_size
                return _safe_randint(builder_rng, default_min, max_size, default=max_size)
        # 制約がない場合、グリッドサイズに応じたデフォルト範囲から選択
        if grid_size is not None:
            return _safe_randint(builder_rng, default_min, default_max, default=default_min)
        return None  # 制約なし

    # ノイズパターンの色を決定（3パターンから確率的に選択）
    def _determine_noise_color(builder_rng, bg_color, existing_colors_set, default_color):
        """ノイズパターンの色を決定

        ①背景色以外で完全ランダム（33%）
        ②既存のオブジェクトで使用されている背景色以外の色の中でランダム（33%）
        ③既存のオブジェクトで使用されていない背景色以外の色の中でランダム（34%）
        """
        rand = builder_rng.random()

        # 背景色以外の全色
        all_non_bg_colors = [c for c in range(10) if c != bg_color]

        if rand < 0.33:
            # ①背景色以外で完全ランダム
            if all_non_bg_colors:
                return builder_rng.choice(all_non_bg_colors)
        elif rand < 0.66:
            # ②既存のオブジェクトで使用されている背景色以外の色の中でランダム
            used_colors = [c for c in existing_colors_set if c != bg_color]
            if used_colors:
                return builder_rng.choice(used_colors)
            # 既存の色がない場合は、全色から選択
            elif all_non_bg_colors:
                return builder_rng.choice(all_non_bg_colors)
        else:
            # ③既存のオブジェクトで使用されていない背景色以外の色の中でランダム
            unused_colors = [c for c in all_non_bg_colors if c not in existing_colors_set]
            if unused_colors:
                return builder_rng.choice(unused_colors)
            # 未使用の色がない場合は、全色から選択
            elif all_non_bg_colors:
                return builder_rng.choice(all_non_bg_colors)

        # フォールバック
        return default_color if default_color is not None else (1 if bg_color == 0 else 0)

    # num_objects、total_num_objects、grid_sizeに基づいて各オブジェクトごとの面積を計算
    object_areas = None
    # total_num_objectsが指定されていない場合は、num_objectsをtotal_num_objectsとして使用
    total_num_objects_from_kwargs = kwargs.get('total_num_objects', None)
    total_num_objects = total_num_objects_from_kwargs if total_num_objects_from_kwargs is not None else num_objects
    # area_ratioが指定されている場合は、それを使用（2段階生成で全体で1つのarea_ratioを共有する場合）
    shared_area_ratio = kwargs.get('area_ratio', None)

    # grid_width/grid_heightがNoneの場合とtotal_num_objectsが未指定の場合の処理

    # grid_width/grid_heightがNoneの場合でも、area_ratioが指定されていればデフォルトのグリッドサイズを仮定
    # これにより、面積ベースの制約を常に適用できるようにする
    if num_objects > 0 and total_num_objects and total_num_objects > 0:
        if grid_width and grid_height:
            # 通常ケース: グリッドサイズが指定されている場合
            object_areas = _calculate_object_areas_from_total(
                grid_width, grid_height, num_objects, total_num_objects, rng, area_ratio=shared_area_ratio
            )
        elif shared_area_ratio is not None:
            # グリッドサイズがNoneでも、area_ratioが指定されている場合はデフォルトのグリッドサイズを仮定
            # デフォルトのグリッドサイズ: 10x10（一般的な最小サイズを想定）
            default_grid_size = 10
            object_areas = _calculate_object_areas_from_total(
                default_grid_size, default_grid_size, num_objects, total_num_objects, rng, area_ratio=shared_area_ratio
            )
    # object_colorsが指定されている場合、色の使用状況をトラッキング
    color_usage_count = {}  # 各色の使用回数を記録
    if object_colors:
        for color in object_colors:
            color_usage_count[color] = 0
        # 既存オブジェクトの色もカウント
        if existing_objects:
            for obj in existing_objects:
                if isinstance(obj, dict) and 'color' in obj:
                    obj_color = obj['color']
                    if obj_color in color_usage_count:
                        color_usage_count[obj_color] = color_usage_count.get(obj_color, 0) + 1

    # グループ複製の事前チェック（duplicate_num >= 2の場合）
    # 最適化: 事前に形状シグネチャを取得せず、必要時に取得する
    group_duplication_applied = False
    if duplicate_num >= 2 and existing_objects and len(existing_objects) >= duplicate_num:
        # グループ複製に必要な数のオブジェクトを取得
        candidate_objects = get_valid_objects_for_group_duplication(existing_objects, duplicate_num)

        if len(candidate_objects) >= 2:
            # 基準オブジェクトを選択
            base_obj = rng.choice(candidate_objects)

            # 近いオブジェクトを選択してグループを形成
            group_objects = _select_nearby_objects_group(
                base_obj, candidate_objects, duplicate_num, rng
            )

        if len(group_objects) >= 2:
            # グループ全体の相対位置を計算
            base_center = _get_object_center(base_obj)
            if base_center is not None:
                # グループ内の各オブジェクトの相対位置を計算
                relative_positions = []
                for group_obj in group_objects:
                    obj_center = _get_object_center(group_obj)
                    if obj_center is not None:
                        rel_x = obj_center[0] - base_center[0]
                        rel_y = obj_center[1] - base_center[1]
                        relative_positions.append((rel_x, rel_y, group_obj))

                # グループ全体を複製（num_objects回）
                group_result = []
                for group_idx in range(num_objects):
                    # 新しい基準位置をランダムに決定（既存オブジェクトの位置を避ける）
                    # 簡易実装：既存オブジェクトの位置から十分離れた位置を選択
                    new_base_x = rng.randint(0, grid_width - 1) if grid_width else 0
                    new_base_y = rng.randint(0, grid_height - 1) if grid_height else 0

                    # グループ内の各オブジェクトを複製
                    for rel_x, rel_y, source_obj in relative_positions:
                        new_obj = source_obj.copy()
                        if 'pixels' in new_obj and isinstance(new_obj['pixels'], list):
                            new_obj['pixels'] = [(px, py) for px, py in new_obj['pixels']]

                        # 新しい位置を計算
                        new_x = int(new_base_x + rel_x)
                        new_y = int(new_base_y + rel_y)

                        # グリッド範囲内に収める
                        if grid_width:
                            new_x = max(0, min(grid_width - 1, new_x))
                        if grid_height:
                            new_y = max(0, min(grid_height - 1, new_y))

                        # 位置情報を更新
                        new_obj['x'] = new_x
                        new_obj['y'] = new_y
                        new_obj['position'] = (new_x, new_y)

                        # duplicate_modeに応じて色を決定
                        if duplicate_mode == 'exact':
                            # 元の色を保持
                            pass  # 色は既にコピーされている
                        elif duplicate_mode == 'shape_only':
                            # 各オブジェクトの色を個別に決定
                            if object_colors:
                                new_color = rng.choice(object_colors)
                            else:
                                existing_colors_list = [c for c in existing_colors if c != background_color]
                                source_color = source_obj.get('color')
                                available_colors = [c for c in existing_colors_list if c != source_color]
                                if available_colors:
                                    new_color = rng.choice(available_colors)
                                else:
                                    new_color = source_color if source_color is not None else 1
                            new_obj['color'] = new_color
                        # duplicate_mode == None の場合は統計ベース（元の色を保持）

                        group_result.append(new_obj)

                # グループ複製の結果を返す
                result_objects.extend(group_result)
                group_duplication_applied = True

    # グループ複製が適用されなかった場合のみ、通常のループ処理を実行
    if not group_duplication_applied:
        # 詳細ログ: ループ開始
        if ENABLE_COLOR_INVESTIGATION_LOGS:
            print(f"[色数調査] generate_objects_from_conditions: ループ開始 - num_objects={num_objects}, object_colors={object_colors}, existing_objects数={len(existing_objects) if existing_objects else 0}", flush=True)

        loop_start_time = time.time()
        object_generation_times = []

        for i in range(num_objects):
            object_gen_start = time.time()
            obj = None
            should_generate_new = True

            # object_specを取得（リストの場合はi番目の要素を使用）
            current_object_spec = None
            if isinstance(object_spec, list):
                if i < len(object_spec):
                    current_object_spec = object_spec[i]
            elif object_spec is not None:
                current_object_spec = object_spec

            # 各反復の最初で、should_generate_newとobjを初期化
            should_generate_new = True  # デフォルトは新規生成
            obj = None  # デフォルトはNone（生成後、設定される）

            # 統計ベースの複製を適用するため、既に生成されたオブジェクトも参照対象に含める
            # 一括生成や第1段階でも、生成されたオブジェクト同士で複製を適用できるようにする
            # existing_objects（外部から渡された既存オブジェクト）とresult_objects（このループで生成されたオブジェクト）を結合
            combined_existing_objects = []
            if existing_objects:
                combined_existing_objects.extend(existing_objects)
            if result_objects:
                combined_existing_objects.extend(result_objects)

            # 2.1 色の選択（object_colorsから均等に選択）
            # object_colorsが指定されている場合、色を選択
            # 既存オブジェクトの色も含めて、各色が1回ずつ使われていたら、あとはランダム
            if object_colors:
                # まだ使用されていない色を確認
                unused_colors = [c for c in object_colors if color_usage_count.get(c, 0) == 0]
                if unused_colors:
                    # 未使用の色からランダムに選択
                    selected_color = rng.choice(unused_colors)
                else:
                    # すべての色が1回ずつ使われた場合は、object_colorsからランダムに選択
                    selected_color = rng.choice(object_colors)
                # 選択した色をcurrent_object_colorに設定（このオブジェクト生成時に使用）
                current_object_color = selected_color
                # 使用回数を更新
                color_usage_count[selected_color] = color_usage_count.get(selected_color, 0) + 1

                # 詳細ログ: 色選択の詳細（最初の数回のみ出力）
                if ENABLE_COLOR_INVESTIGATION_LOGS and (i < 5 or len(object_colors) > 1):  # 最初の5回、または複数色の場合は詳細ログ
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, object_colors={object_colors}, unused_colors={unused_colors}, selected_color={selected_color}, color_usage_count={dict(color_usage_count)}", flush=True)
            else:
                # object_colorsが指定されていない場合、背景色以外からランダムに選択
                # 最適化: 事前計算済みのavailable_colorsを使用（結果は同じ）
                if precomputed_available_colors:
                    current_object_color = rng.choice(precomputed_available_colors)
                else:
                    # フォールバック（通常は発生しない）
                    current_object_color = 1 if background_color == 0 else 0

            # 2.2 サイズ計算（object_specを最優先）
            # object_specにサイズが指定されている場合は、それを最優先で使用
            calculated_min_size_for_this = None
            calculated_max_size_for_this = None

            if current_object_spec:
                spec_type = current_object_spec.get('type')
                spec_value = current_object_spec.get('value')
                operator = current_object_spec.get('operator', 'greater')

                if spec_type == 'size':
                    # サイズ指定（面積ベース）
                    if operator == 'equal' and spec_value is not None:
                        # 等しい場合は、その値を使用
                        calculated_max_size_for_this = int(rng.randint(1, int(spec_value ** 0.5)) if spec_value > 1 else 1)
                        calculated_min_size_for_this = calculated_max_size_for_this
                    elif operator == 'greater' and spec_value is not None:
                        # より大きい場合は、spec_valueより大きい値
                        calculated_min_size_for_this = max(1, int(spec_value ** 0.5) + 1)
                        calculated_max_size_for_this = min(grid_width, grid_height) if grid_width and grid_height else calculated_min_size_for_this + 10
                    elif operator == 'less' and spec_value is not None:
                        # より小さい場合は、spec_valueより小さい値
                        calculated_max_size_for_this = max(1, int(spec_value ** 0.5) - 1)
                        calculated_min_size_for_this = 1
                elif spec_type == 'width':
                    # 幅指定
                    if operator == 'equal' and spec_value is not None:
                        calculated_max_size_for_this = spec_value
                        calculated_min_size_for_this = spec_value
                    elif operator == 'greater' and spec_value is not None:
                        calculated_min_size_for_this = spec_value + 1
                        calculated_max_size_for_this = grid_width if grid_width else calculated_min_size_for_this + 10
                    elif operator == 'less' and spec_value is not None:
                        calculated_max_size_for_this = max(1, spec_value - 1)
                        calculated_min_size_for_this = 1
                elif spec_type == 'height':
                    # 高さ指定
                    if operator == 'equal' and spec_value is not None:
                        calculated_max_size_for_this = spec_value
                        calculated_min_size_for_this = spec_value
                    elif operator == 'greater' and spec_value is not None:
                        calculated_min_size_for_this = spec_value + 1
                        calculated_max_size_for_this = grid_height if grid_height else calculated_min_size_for_this + 10
                    elif operator == 'less' and spec_value is not None:
                        calculated_max_size_for_this = max(1, spec_value - 1)
                        calculated_min_size_for_this = 1

            # object_specにサイズが指定されていない場合は、従来通りobject_areasから計算
            if calculated_min_size_for_this is None and calculated_max_size_for_this is None:
                if object_areas is None:
                    # object_areasがNoneの場合はエラーを出す（フォールバックを廃止）
                    raise ValueError(
                        f"[面積ベース制約エラー] object_areasがNoneです: "
                        f"grid_width={grid_width}, grid_height={grid_height}, "
                        f"num_objects={num_objects}, total_num_objects={kwargs.get('total_num_objects', num_objects)}, "
                        f"i={i}, area_ratio={shared_area_ratio}"
                    )
                elif i >= len(object_areas):
                    # object_areasのインデックスが範囲外の場合はエラーを出す（フォールバックを廃止）
                    raise ValueError(
                        f"[面積ベース制約エラー] object_areasのインデックスが範囲外です: "
                        f"i={i}, len(object_areas)={len(object_areas)}, "
                        f"object_areas={object_areas[:5] if len(object_areas) >= 5 else object_areas}"
                    )
                else:
                    area = object_areas[i]
                    # total_num_objectsを渡して、オブジェクト数が多い場合に小さいサイズが選ばれる確率を上げる
                    total_num_objects = kwargs.get('total_num_objects', num_objects)
                    calculated_max_size_for_this = _calculate_size_from_area(area, grid_width, grid_height, rng, total_num_objects)
                    calculated_min_size_for_this = 1  # 常に最小1

            # duplicate_modeによる複製制御
            # 詳細ログ: 複製制御の判定
            if i < 5:  # 最初の5回のみ詳細ログ
                if ENABLE_COLOR_INVESTIGATION_LOGS:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, duplicate_mode判定開始 - duplicate_mode={duplicate_mode}, existing_objects数={len(existing_objects) if existing_objects else 0}", flush=True)

            # duplicate_modeがNoneで既存オブジェクトがない場合、should_generate_newをTrueに設定（明示的に）
            # ただし、combined_existing_objects（既に生成されたオブジェクトを含む）がある場合は統計ベースの複製を適用
            if duplicate_mode is None and (not combined_existing_objects or len(combined_existing_objects) == 0):
                should_generate_new = True
                if i < 5:  # 詳細ログ
                    if ENABLE_COLOR_INVESTIGATION_LOGS:
                        print(f"[色数調査] generate_objects_from_conditions: i={i}, duplicate_mode=Noneかつ既存オブジェクトなし → should_generate_new=Trueに設定", flush=True)

            if duplicate_mode == 'forbidden':
                # 複製禁止：常に新規生成
                should_generate_new = True
            elif duplicate_mode == 'exact' and combined_existing_objects and len(combined_existing_objects) > 0:
                # 完全一致を強制（単一オブジェクト複製）
                # 最適化: 形状シグネチャ取得は不要、そのままオブジェクトをコピー可能
                # 面積ベースの制約: 割り当てられた面積以下のオブジェクトを優先的に選択
                target_area_for_duplication = object_areas[i] if object_areas is not None and i < len(object_areas) else None
                source_obj = get_valid_object_for_duplication(combined_existing_objects, target_area=target_area_for_duplication)
                if source_obj is None:
                    should_generate_new = True
                else:
                    # ARC-AGI2の分析結果: 0.5%が変換あり（回転0.4%、反転0.1%）
                    original_color = source_obj.get('color', 0)
                    # 変換を適用するかどうか（確率0.5%）
                    apply_transformation = rng.random() < 0.005
                    obj = copy_object_with_new_color(source_obj, original_color, apply_transformation=apply_transformation, rng=rng)
                    should_generate_new = False
            elif duplicate_mode == 'color_only' and combined_existing_objects and len(combined_existing_objects) > 0:
                # 色のみ複製（形状は新規生成）
                # 最適化: 必要時に1個のオブジェクトを選ぶ（形状シグネチャは不要だが、色を取得するため）
                source_obj = rng.choice(combined_existing_objects) if combined_existing_objects else None
                if source_obj is None:
                    should_generate_new = True
                else:
                    source_color = source_obj.get('color')
                    if source_color is not None:
                        current_object_color = source_color
                    should_generate_new = True  # 形状は新規生成
            elif duplicate_mode == 'shape_only' and combined_existing_objects and len(combined_existing_objects) > 0:
                # 形状のみ複製（色は新規、オプションで反転・回転を適用）
                # 最適化: 形状シグネチャ取得は不要、そのままオブジェクトをコピー可能
                # 面積ベースの制約: 割り当てられた面積以下のオブジェクトを優先的に選択
                target_area_for_duplication = object_areas[i] if object_areas is not None and i < len(object_areas) else None
                source_obj = get_valid_object_for_duplication(combined_existing_objects, target_area=target_area_for_duplication)
                if source_obj is None:
                    should_generate_new = True
                else:
                    if object_colors:
                        new_color = rng.choice(object_colors)
                    else:
                        # 最適化: 事前計算済みのexisting_colors_listを使用（結果は同じ）
                        source_color = source_obj.get('color')
                        if precomputed_existing_colors_list:
                            available_colors = [c for c in precomputed_existing_colors_list if c != source_color]
                            if available_colors:
                                new_color = rng.choice(available_colors)
                            else:
                                new_color = source_color if source_color is not None else 1
                        else:
                            # フォールバック（通常は発生しない）
                            new_color = source_color if source_color is not None else 1
                    # ARC-AGI2の分析結果: 0.5%が変換あり（回転0.4%、反転0.1%）
                    # 変換を適用するかどうか（確率0.5%）
                    apply_transformation = rng.random() < 0.005
                    obj = copy_object_with_new_color(source_obj, new_color, apply_transformation=apply_transformation, rng=rng)
                    should_generate_new = False
            # 既存のオブジェクトがある場合、統計に基づいて同じ形状/色を生成（duplicate_mode=Noneの場合）
            # 一括生成や第1段階でも、既に生成されたオブジェクトを参照して統計ベースの複製を適用
            elif duplicate_mode is None and combined_existing_objects and len(combined_existing_objects) > 0:
                # 形状や色に関する制約があるかチェック
                has_constraints = (
                    allow_holes or  # 穴ありオブジェクトの制約
                    current_object_spec is not None or  # オブジェクト仕様（現在のオブジェクト）
                    shape_type is not None or  # 形状タイプ
                    (symmetry_constraint is not None and symmetry_constraint != 'none')  # 対称性制約
                )

                # 制約がある場合は新規生成を優先、制約がない場合のみ確率でコピー
                if has_constraints:
                    should_generate_new = True
                    rand_val = rng.random()  # 制約がある場合もrand_valを定義（後続の条件チェックで使用される可能性があるため）
                else:
                    # 制約がない場合、確率でコピー or 新規生成
                    rand_val = rng.random()

                # 同じ色と形状を生成（許可されている場合）
                # 修正: selected_color（current_object_color）を優先的に使用して色の多様性を保つ
                if allow_same_color_and_shape and rand_val < SAME_COLOR_AND_SHAPE_PROBABILITY:
                    # 最適化: 形状シグネチャ取得は不要、そのままオブジェクトをコピー可能
                    # 面積ベースの制約: 割り当てられた面積以下のオブジェクトを優先的に選択
                    target_area_for_duplication = object_areas[i] if object_areas is not None and i < len(object_areas) else None
                    source_obj = get_valid_object_for_duplication(combined_existing_objects, target_area=target_area_for_duplication)
                    if source_obj is None:
                        should_generate_new = True
                    else:
                        # 形状をコピーするが、色はselected_color（current_object_color）を使用
                        # ARC-AGI2の分析結果: 0.5%が変換あり（回転0.4%、反転0.1%）
                        # 変換を適用するかどうか（確率0.5%）
                        apply_transformation = rng.random() < 0.005
                        # current_object_colorを優先的に使用（selected_colorが指定されている場合）
                        obj = copy_object_with_new_color(source_obj, current_object_color, apply_transformation=apply_transformation, rng=rng)
                        should_generate_new = False

                # 同じ形状（色は異なる）を生成（許可されている場合）
                # 修正: selected_color（current_object_color）を優先的に使用
                elif allow_same_shape and rand_val < (SAME_COLOR_AND_SHAPE_PROBABILITY + SAME_SHAPE_PROBABILITY):
                    # 最適化: 形状シグネチャ取得は不要、そのままオブジェクトをコピー可能
                    # 面積ベースの制約: 割り当てられた面積以下のオブジェクトを優先的に選択
                    target_area_for_duplication = object_areas[i] if object_areas is not None and i < len(object_areas) else None
                    source_obj = get_valid_object_for_duplication(combined_existing_objects, target_area=target_area_for_duplication)
                    if source_obj is None:
                        should_generate_new = True
                    else:
                        # 形状をコピーして新しい色を設定
                        # current_object_colorを優先的に使用（selected_colorが指定されている場合）
                        if object_colors and current_object_color in object_colors:
                            # selected_colorがobject_colorsに含まれている場合は優先的に使用
                            new_color = current_object_color
                        elif object_colors:
                            # selected_colorがobject_colorsに含まれていない場合は、object_colorsから選択
                            new_color = rng.choice(object_colors)
                        else:
                            # 既存の色から選択（元の色と異なる色）
                            existing_colors_list = [c for c in existing_colors if c != background_color]
                            source_color = source_obj.get('color')
                            available_colors = [c for c in existing_colors_list if c != source_color]
                            if available_colors:
                                new_color = rng.choice(available_colors)
                            else:
                                # 利用可能な色がない場合はselected_colorを使用
                                new_color = current_object_color

                        # ARC-AGI2の分析結果: 0.5%が変換あり（回転0.4%、反転0.1%）
                        # 変換を適用するかどうか（確率0.5%）
                        apply_transformation = rng.random() < 0.005
                        obj = copy_object_with_new_color(source_obj, new_color, apply_transformation=apply_transformation, rng=rng)
                        should_generate_new = False
                else:
                    # 確率で新規生成に該当する場合
                    should_generate_new = True

            # 新しいオブジェクトを生成（既存のオブジェクトを使用しない場合）
            # 詳細ログ: オブジェクト生成の判定
            if i < 5:  # 最初の5回のみ詳細ログ
                if ENABLE_COLOR_INVESTIGATION_LOGS:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, should_generate_new={should_generate_new}, obj is None={obj is None}", flush=True)

            if should_generate_new:
                # このオブジェクト専用のサイズ範囲を使用（object_areasベースの計算結果がある場合）
                calculated_min_size = calculated_min_size_for_this
                calculated_max_size = calculated_max_size_for_this

                # 利用可能スペースを考慮したサイズ制約を計算
                # 既存オブジェクトの位置情報がある場合、利用可能スペースを考慮
                adjusted_max_bbox_size = max_bbox_size
                if grid_width and grid_height and existing_positions:
                    max_w, max_h = _calculate_max_object_size_for_available_space(
                        grid_width, grid_height, existing_positions, min_spacing, default_max_size=10
                    )
                    # max_bbox_sizeが指定されている場合は、より小さい方を採用
                    if max_bbox_size > 0:
                        adjusted_max_bbox_size = min(max_bbox_size, max(max_w, max_h))
                    else:
                        adjusted_max_bbox_size = max(max_w, max_h)

                # num_objectsベースの計算結果を適用
                if calculated_max_size is not None:
                    # calculated_max_sizeが既存の制約より小さい場合は、そちらを優先
                    if adjusted_max_bbox_size > 0:
                        adjusted_max_bbox_size = min(adjusted_max_bbox_size, calculated_max_size)
                    else:
                        adjusted_max_bbox_size = calculated_max_size

                    # min_bbox_sizeも更新（計算結果がある場合）
                    if calculated_min_size is not None and min_bbox_size < calculated_min_size:
                        min_bbox_size = calculated_min_size

                # オブジェクト生成（再生成ループは廃止）
                obj = None

                # 形状タイプが指定されている場合（最優先）
                if shape_type:
                    # object_areasから計算されたサイズ制約を優先的に使用
                    if calculated_max_size is not None:
                        # object_areasから計算されたサイズ制約がある場合は、それを最大サイズとして使用
                        max_size_for_dist = calculated_max_size
                    else:
                        # object_areasから計算されたサイズ制約がない場合は、従来通り確率分布を使用
                        total_num_objects = kwargs.get('total_num_objects', num_objects)
                        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                            grid_w = grid_size[0]
                            grid_h = grid_size[1]
                        else:
                            grid_w = None
                            grid_h = None

                        # 最大サイズを決定（bbox_size制約を考慮）
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if effective_max_bbox_size > 0:
                            max_size_for_dist = min(effective_max_bbox_size, min(grid_w, grid_h) - 1 if grid_w and grid_h else 10)
                        else:
                            max_size_for_dist = min(grid_w, grid_h) - 1 if grid_w and grid_h else 10

                    # 確率分布に基づいてサイズを決定（max_size_for_distを制約として使用）
                    total_num_objects = kwargs.get('total_num_objects', num_objects)
                    if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                        grid_w = grid_size[0]
                        grid_h = grid_size[1]
                    else:
                        grid_w = None
                        grid_h = None

                    size = _calculate_size_with_probability_distribution(
                        grid_w, grid_h, total_num_objects, rng,
                        min_size=1, max_size=max_size_for_dist, shape_type=shape_type
                    )

                    # bbox_size制約がある場合は調整
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if effective_max_bbox_size > 0:
                        size = min(size, effective_max_bbox_size)

                    if shape_type == 'rectangle':
                        # サイズは既に確率分布で決定済み

                        # バリエーション: 正方形と長方形の両方を生成
                        # 60%の確率で正方形、40%の確率で長方形
                        if rng.random() < 0.6:
                            # 正方形
                            obj = builder._generate_rectangle(current_object_color, size, size, filled=True)
                        else:
                            # 長方形（幅と高さを異なる値にする）
                            width = size
                            # 高さはサイズの0.5倍から2倍の範囲で確率分布に基づいて決定
                            height_min = max(1, size // 2)
                            height_max = min(max_size_for_dist, size * 2)
                            height = _safe_randint(rng, height_min, min(height_max, size + 3), default=height_min)
                            obj = builder._generate_rectangle(current_object_color, width, height, filled=True)
                elif shape_type == 'line':
                    # サイズは既に確率分布で決定済み（上で計算されたsizeを使用）
                    length = size

                    # バリエーション: 水平、垂直、斜め45度、斜め-45度のすべてを生成可能
                    # 60%で水平/垂直、40%で斜め
                    if rng.random() < 0.6:
                        direction = rng.choice(['horizontal', 'vertical'])
                        obj = builder._generate_line(current_object_color, length, direction)
                    else:
                        # 斜め線を生成（diagonal_45を使用、sizeパラメータにlengthを使用）
                        diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        obj = builder._generate_diagonal_45(current_object_color, length, diagonal_direction)
                elif shape_type == 'hollow_rectangle':
                    # サイズは既に確率分布で決定済み（上で計算されたsizeを使用）

                    # バリエーション: 正方形と長方形の両方を生成
                    # 60%の確率で正方形、40%の確率で長方形
                    if rng.random() < 0.6:
                        # 正方形の中空矩形
                        obj = builder._generate_rectangle(current_object_color, size, size, filled=False)
                    else:
                        # 長方形の中空矩形（幅と高さを異なる値にする）
                        width = size
                        # 高さはサイズの0.5倍から2倍の範囲で確率分布に基づいて決定
                        height_min = max(1, size // 2)
                        height_max = min(max_size_for_dist if 'max_size_for_dist' in locals() else size * 2, size * 2)
                        height = _safe_randint(rng, height_min, min(height_max, size + 3), default=height_min)
                        obj = builder._generate_rectangle(current_object_color, width, height, filled=False)
                elif shape_type == 'noise':
                    # ノイズパターン：グリッド全体の一定パーセンテージの1ピクセルをランダムに配置
                    # 連結性や重複を気にせず、ランダムに配置

                    # ノイズパターンの色選択（3パターンから確率的に選択、全体で1つの色を決定）
                    # ①背景色以外で完全ランダム（33%）
                    # ②既存のオブジェクトで使用されている背景色以外の色の中でランダム（33%）
                    # ③既存のオブジェクトで使用されていない背景色以外の色の中でランダム（34%）
                    noise_color = _determine_noise_color(rng, background_color, existing_colors, current_object_color)

                    # グリッドサイズからノイズピクセル数を決定（最適化版：処理速度向上）
                    if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                        grid_width_noise, grid_height_noise = grid_size[0], grid_size[1]
                        total_cells = grid_width_noise * grid_height_noise

                        # ノイズパーセンテージを決定（グリッドサイズに応じて調整）
                        # 大きいグリッドでは密度を下げて処理速度を向上
                        if total_cells <= 100:  # 10x10以下
                            noise_percentage_min = 0.05  # 5%
                            noise_percentage_max = 0.25  # 25%
                        elif total_cells <= 400:  # 20x20以下
                            noise_percentage_min = 0.03  # 3%
                            noise_percentage_max = 0.15  # 15%
                        else:  # それ以上
                            noise_percentage_min = 0.02  # 2%
                            noise_percentage_max = 0.10  # 10%

                        noise_percentage = noise_percentage_min + rng.random() * (noise_percentage_max - noise_percentage_min)

                        # ノイズピクセル数を計算（グリッドサイズの面積が自然な上限）
                        num_noise_pixels = max(1, int(total_cells * noise_percentage))

                        # グリッド内のランダムな位置にピクセルを生成（処理速度最適化版）
                        # 効率性を考慮して、num_noise_pixelsの割合に応じて方法を選択
                        noise_ratio = num_noise_pixels / total_cells if total_cells > 0 else 0

                        # num_noise_pixelsがtotal_cellsの50%以上の場合、または小さいグリッド（400セル以下）の場合、
                        # random.sampleを使用（効率的）
                        if grid_width_noise is not None and grid_height_noise is not None:
                            if total_cells <= 400 or noise_ratio >= 0.5:
                                # 全候補を生成してrandom.sampleを使用
                                if num_noise_pixels >= total_cells:
                                    all_positions = [(x, y) for x in range(grid_width_noise) for y in range(grid_height_noise)]
                                    noise_pixels = all_positions
                                else:
                                    all_positions = [(x, y) for x in range(grid_width_noise) for y in range(grid_height_noise)]
                                    sample_size = min(num_noise_pixels, len(all_positions))
                                    if sample_size > 0 and len(all_positions) > 0:
                                        noise_pixels = rng.sample(all_positions, sample_size)
                                    else:
                                        noise_pixels = []
                        else:
                            # 大きいグリッドでnum_noise_pixelsが少ない場合、直接ランダム生成（メモリ効率を優先）
                            # used_positionsを使った方法で重複を避けながら生成
                            if grid_width_noise is not None and grid_height_noise is not None:
                                noise_pixels = []
                                used_positions = set()
                                # 効率性を考慮して、max_attemptsを適切に設定
                                # num_noise_pixelsが小さい場合、試行回数を制限
                                max_attempts = min(num_noise_pixels * 10, total_cells * 2)
                                attempts = 0

                                while len(noise_pixels) < num_noise_pixels and attempts < max_attempts:
                                    x = rng.randint(0, grid_width_noise - 1)
                                    y = rng.randint(0, grid_height_noise - 1)
                                    pos = (x, y)
                                    if pos not in used_positions:
                                        noise_pixels.append(pos)
                                        used_positions.add(pos)
                                    attempts += 1

                                # 必要な数に達しなかった場合、残りを追加
                                if len(noise_pixels) < num_noise_pixels:
                                    remaining = num_noise_pixels - len(noise_pixels)
                                    all_positions = [(x, y) for x in range(grid_width_noise) for y in range(grid_height_noise) if (x, y) not in used_positions]
                                    if all_positions:
                                        sample_size = min(remaining, len(all_positions))
                                        if sample_size > 0:
                                            additional = rng.sample(all_positions, sample_size)
                                            noise_pixels.extend(additional)
                            else:
                                noise_pixels = []

                        # バウンディングボックスを計算
                        if noise_pixels:
                            min_x = min(px for px, py in noise_pixels)
                            max_x = max(px for px, py in noise_pixels)
                            min_y = min(py for px, py in noise_pixels)
                            max_y = max(py for px, py in noise_pixels)

                            # オフセットを0,0にする
                            normalized_pixels = [(x - min_x, y - min_y) for x, y in noise_pixels]
                            width = max_x - min_x + 1
                            height = max_y - min_y + 1

                            # 1つのオブジェクトとして生成（複数のピクセルを持つ）
                            from .core_object_builder import _add_lock_flags
                            obj = _add_lock_flags({
                                'pixels': normalized_pixels,
                                'color': noise_color,
                                'width': width,
                                'height': height,
                                'area': len(normalized_pixels),
                                'shape_type': 'noise_pattern',
                                'x': min_x,  # 元の位置情報を保持
                                'y': min_y
                            })

                        else:
                            # ピクセルが生成できなかった場合は1ピクセルをフォールバック
                            obj = builder._generate_single_pixel(noise_color)
                    else:
                        # グリッドサイズが指定されていない場合は1ピクセルを生成
                        obj = builder._generate_single_pixel(noise_color)
                elif shape_type == 'cross':
                    # adjusted_max_bbox_sizeを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(7, effective_max_bbox_size) if effective_max_bbox_size > 0 else 7
                        size = _safe_randint(rng, 3, max_val, default=3)
                    obj = builder._generate_cross(current_object_color, size)
                elif shape_type == 't_shape':
                    # adjusted_max_bbox_sizeを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                        size = _safe_randint(rng, 3, max_val, default=3)
                    rotation = rng.choice([0, 90, 180, 270])
                    obj = builder._generate_t_shape(current_object_color, size, rotation)
                elif shape_type == 'diagonal_45':
                    # adjusted_max_bbox_sizeを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                        size = _safe_randint(rng, 3, max_val, default=3)
                    direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                    obj = builder._generate_diagonal_45(current_object_color, size, direction)
                elif shape_type == 'circle':
                    # 円はサイズパラメータがないため、そのまま生成
                    obj = builder._generate_circle(current_object_color)
                elif shape_type == 'l_shape':
                    # L字形: サイズパラメータを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        adjusted_min_l = calculated_min_size
                        adjusted_max_l = calculated_max_size
                    else:
                        # グリッドサイズに応じたデフォルト範囲を計算
                        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                            adjusted_min_l, adjusted_max_l = _calculate_default_size_range(
                                grid_size[0], grid_size[1], 2, 6
                            )
                        else:
                            adjusted_min_l, adjusted_max_l = (2, 6)

                    effective_max_l = min(adjusted_max_l, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_l
                    width = _safe_randint(rng, adjusted_min_l, effective_max_l, default=adjusted_min_l)
                    height = _safe_randint(rng, adjusted_min_l, effective_max_l, default=adjusted_min_l)
                    obj = builder._generate_l_shape(current_object_color, width, height)
                elif shape_type == 'triangle':
                    # 三角形: サイズパラメータを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        adjusted_min_tri = calculated_min_size
                        adjusted_max_tri = calculated_max_size
                    else:
                        # グリッドサイズに応じたデフォルト範囲を計算
                        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                            adjusted_min_tri, adjusted_max_tri = _calculate_default_size_range(
                                grid_size[0], grid_size[1], 3, 6
                            )
                        else:
                            adjusted_min_tri, adjusted_max_tri = (3, 6)

                    effective_max_tri = min(adjusted_max_tri, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_tri
                    size = _safe_randint(rng, adjusted_min_tri, effective_max_tri, default=adjusted_min_tri)
                    obj = builder._generate_triangle(current_object_color, size)
                elif shape_type == 'single_pixel':
                    # 1ピクセル点はサイズパラメータがないため、そのまま生成
                    obj = builder._generate_single_pixel(current_object_color)
                elif shape_type == 'checkerboard':
                    # チェッカーボード
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        pattern_size = max(3, min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size)
                    else:
                        pattern_size = max(3, min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6)
                    obj = builder._generate_checkerboard_shape(current_object_color, pattern_size, pattern_size)
                elif shape_type == 'diagonal':
                    # 対角線はサイズパラメータがないため、そのまま生成
                    obj = builder._generate_diagonal(current_object_color)
                elif shape_type == 'diagonal_connected_shape':
                    # 斜め連結形状
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                        size = _safe_randint(rng, 3, max_val, default=3)
                    obj = builder._generate_diagonal_connected_shape(current_object_color, size)
                elif shape_type == 'arrow':
                    # 矢印型
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(8, effective_max_bbox_size) if effective_max_bbox_size > 0 else 8
                        size = _safe_randint(rng, 3, max_val, default=3)
                    direction = rng.choice(['up', 'down', 'left', 'right'])
                    obj = builder._generate_arrow(current_object_color, size, direction)
                elif shape_type == 'diamond':
                    # ダイヤモンド型
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                        size = _safe_randint(rng, 2, max_val, default=2)
                    filled = rng.random() < 0.7  # 70%の確率で塗りつぶし
                    obj = builder._generate_diamond(current_object_color, size, filled)
                elif shape_type == 'stairs':
                    # 階段型
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        steps = _safe_randint(rng, 2, min(5, max_val // 2), default=2)
                    else:
                        max_val = min(8, effective_max_bbox_size) if effective_max_bbox_size > 0 else 8
                        steps = _safe_randint(rng, 2, min(5, max_val // 2), default=2)
                    step_size = _safe_randint(rng, 2, 4, default=2)
                    obj = builder._generate_stairs(current_object_color, steps, step_size)
                elif shape_type == 'zigzag':
                    # ギザギザ型
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if calculated_min_size is not None and calculated_max_size is not None:
                        max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                        length = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                    else:
                        max_val = min(10, effective_max_bbox_size) if effective_max_bbox_size > 0 else 10
                        length = _safe_randint(rng, 4, max_val, default=4)
                    amplitude = _safe_randint(rng, 1, 3, default=2)
                    obj = builder._generate_zigzag(current_object_color, length, amplitude)
                elif shape_type == 'simple_object':
                    # シンプルオブジェクトはサイズパラメータがないため、そのまま生成
                    # グリッドサイズと総オブジェクト数情報を渡して動的サイズ範囲を適用
                    obj = builder._generate_simple_object(current_object_color, grid_size, total_num_objects, max_size=calculated_max_size)
                elif shape_type == 'random_pattern':
                    # ランダムパターン：指定サイズ内でランダムにピクセルで埋め、連結性を保証
                    # num_objectsベースのサイズ範囲がある場合はそれを優先、なければグリッドサイズに応じたデフォルト範囲を計算
                    if calculated_min_size is not None and calculated_max_size is not None:
                        adjusted_min_size = calculated_min_size
                        adjusted_max_size = calculated_max_size
                    else:
                        # グリッドサイズに応じたデフォルト範囲を計算
                        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                            adjusted_min_size, adjusted_max_size = _calculate_default_size_range(
                                grid_size[0], grid_size[1], 3, 8
                            )
                        else:
                            adjusted_min_size, adjusted_max_size = (3, 8)

                    # adjusted_max_bbox_sizeを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    effective_max_size = min(adjusted_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_size

                    # サイズを決定
                    width = _safe_randint(rng, adjusted_min_size, effective_max_size, default=adjusted_min_size)
                    height = _safe_randint(rng, adjusted_min_size, effective_max_size, default=adjusted_min_size)

                    # 密度を決定（20%〜90%）
                    # 計算速度を優先する場合は密度指定なし（None）にして、メソッド内のデフォルト処理に任せることも可能
                    density = 0.2 + rng.random() * 0.7

                    # ランダムパターンを生成
                    obj = builder._generate_random_pattern(current_object_color, width, height, density)

                    # 連結性を保証（最適化: 小さいオブジェクトではスキップ）
                    # 1-5ピクセルは常に連結なので、連結性チェックをスキップ
                    if obj:
                        pixels = obj.get('pixels', [])
                        num_pixels = len(pixels) if pixels else 0

                        # 小さいオブジェクト（1-5ピクセル）では連結性チェックをスキップ
                        if num_pixels <= 5:
                            # 連結性チェックをスキップ（常に連結のため）
                            pass
                        else:
                            # 5ピクセル以上のオブジェクトのみ連結性チェック
                            width_obj = obj.get('width', 1)
                            height_obj = obj.get('height', 1)
                            if width_obj > 0 and height_obj > 0:
                                if connectivity_constraint is not None:
                                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                                    connected_pixels = builder._ensure_connected(pixels, width_obj, height_obj, connectivity)
                                    obj['pixels'] = connected_pixels
                                    obj['area'] = len(connected_pixels)
                                else:
                                    # 連結性制約が指定されていない場合でも、デフォルトで4連結を保証
                                    connected_pixels = builder._ensure_connected(pixels, width_obj, height_obj, 4)
                                    obj['pixels'] = connected_pixels
                                    obj['area'] = len(connected_pixels)
                elif shape_type in ['u_shape', 'h_shape', 'z_shape']:
                    # adjusted_max_bbox_sizeを考慮
                    effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                    if hasattr(builder, f'_generate_{shape_type}'):
                        method = getattr(builder, f'_generate_{shape_type}')
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                            size = _safe_randint(rng, 3, max_val, default=3)
                        obj = method(current_object_color, size)
                else:
                    # その他の形状タイプ（hasattrでチェックしてから生成）
                    if hasattr(builder, f'_generate_{shape_type}'):
                        method = getattr(builder, f'_generate_{shape_type}')
                        obj = method(current_object_color)
                        # 動的メソッド呼び出しで生成されたオブジェクトのサイズをチェック（警告は削除済み）

                # オブジェクト生成（shape_typeが指定されていない場合、またはshape_typeで生成できなかった場合）
                # 注意: 対称性制約と穴ありオブジェクトの適用は後処理で行う（後処理部分を参照）
                # 注意: object_specはサイズ計算段階（2.2）で既に影響を与えているため、ここでは生成しない
                if obj is None:
                    # shape_typeが指定されていない場合、またはshape_typeで生成できなかった場合
                    # calculated_max_sizeをmax_sizeパラメータとして渡すことで、サイズ制約を適用
                    rand = builder.rng.random()
                    if rand < 0.33:  # 33%の確率でARC統計生成
                        obj = builder.generate_object_by_arc_stats(current_object_color, grid_size, max_size=calculated_max_size)
                    elif rand < 0.66:  # 33%の確率で合成オブジェクト
                        obj = builder.generate_synthesized_object(current_object_color, complexity, max_size=calculated_max_size)
                    else:  # 34%の確率で複合オブジェクト
                        obj = builder.generate_composite_object(current_object_color, grid_size, max_size=calculated_max_size)
                else:
                    # 25%の確率でshape_typeからランダムに選択して生成
                    # フォールバック用のshape_typeリスト（確率分布を保持）
                    # 各形状タイプの重み（フォールバック処理の確率分布に基づく）
                    # 注意: noiseは除外（フォールバックでは使用しない）
                    # 最適化: random_patternと複雑な形状の確率を削減（処理速度向上）
                    # 確率分布を微調整: 基本形状をやや削減、中程度の形状をやや増加
                    # 合計が1.00になるように調整（0.12+0.10+0.09*5+0.05+0.02+0.03+0.01*2+0.03*5+0.05 = 1.00）
                    shape_type_weights = [
                        ('rectangle', 0.11),  # 0.12 → 0.11（新形状追加のため微調整）
                        ('line', 0.09),  # 0.10 → 0.09（新形状追加のため微調整）
                        ('cross', 0.08),  # 0.09 → 0.08（新形状追加のため微調整）
                        ('circle', 0.08),  # 0.09 → 0.08（新形状追加のため微調整）
                        ('l_shape', 0.08),  # 0.09 → 0.08（新形状追加のため微調整）
                        ('triangle', 0.08),  # 0.09 → 0.08（新形状追加のため微調整）
                        ('single_pixel', 0.08),  # 0.09 → 0.08（新形状追加のため微調整）
                        ('arrow', 0.04),  # 新規追加（最高優先度）
                        ('diamond', 0.04),  # 新規追加（最高優先度）
                        ('stairs', 0.03),  # 新規追加（高優先度）
                        ('zigzag', 0.03),  # 新規追加（高優先度）
                        ('random_pattern', 0.04),  # 0.05 → 0.04（新形状追加のため微調整）
                        ('u_shape', 0.02),  # 維持
                        ('t_shape', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('h_shape', 0.01),  # 維持
                        ('z_shape', 0.01),  # 維持
                        ('diagonal_45', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('hollow_rectangle', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('checkerboard', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('diagonal', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('diagonal_connected_shape', 0.02),  # 0.03 → 0.02（新形状追加のため微調整）
                        ('simple_object', 0.05),  # 0.06 → 0.05（新形状追加のため微調整）
                    ]

                    # noiseが含まれている場合は除外（念のため）
                    shape_type_weights = [(st, w) for st, w in shape_type_weights if st != 'noise']

                    # 重みに基づいてランダムに選択
                    rand_shape = builder.rng.random()
                    cumulative = 0.0
                    selected_shape_type = None
                    for st, weight in shape_type_weights:
                        cumulative += weight
                        if rand_shape < cumulative:
                            selected_shape_type = st
                            break

                    # 選択されたshape_typeがNoneの場合はsimple_objectをデフォルトとして使用
                    if selected_shape_type is None:
                        selected_shape_type = 'simple_object'

                    # 選択されたshape_typeで生成（既存のshape_type処理ロジックを再利用）
                    # shape_typeを一時的に変更して、既存のshape_type処理ロジックを再利用
                    original_shape_type = shape_type
                    shape_type = selected_shape_type

                    # サイズ範囲を計算（フォールバック用、既に計算済みの場合は再計算しない）
                    # 密度調整版2: デフォルトサイズ範囲を小さくする
                    if calculated_max_size is not None and calculated_max_size > 0:
                        adjusted_min_rect = calculated_min_size
                        adjusted_max_rect = calculated_max_size
                        adjusted_min_line = calculated_min_size
                        adjusted_max_line = calculated_max_size
                    else:
                        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                            adjusted_min_rect, adjusted_max_rect = _calculate_default_size_range(
                                grid_size[0], grid_size[1], 1, 4
                            )
                            adjusted_min_line, adjusted_max_line = _calculate_default_size_range(
                                grid_size[0], grid_size[1], 2, 5
                            )
                        else:
                            adjusted_min_rect, adjusted_max_rect = (1, 4)
                            adjusted_min_line, adjusted_max_line = (2, 5)

                    # 既存のshape_type処理部分と同じロジックを実行（noiseは除外）
                    if shape_type == 'rectangle':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        effective_max_rect = min(adjusted_max_rect, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_rect
                        bbox_size = _determine_bbox_size(rng, min_bbox_size, effective_max_bbox_size, adjusted_min_rect, effective_max_rect, grid_size)
                        size = bbox_size if bbox_size else _safe_randint(rng, adjusted_min_rect, effective_max_rect, default=adjusted_min_rect)
                        if rng.random() < 0.6:
                            obj = builder._generate_rectangle(current_object_color, size, size, filled=True)
                        else:
                            width = size
                            height_range_min = max(adjusted_min_rect, size // 2) if adjusted_min_rect > 0 else max(2, size // 2)
                            height_range_max = min(effective_max_rect, size * 2) if effective_max_rect > 0 else size * 2
                            height_range_max_val = min(height_range_max, size + 3)
                            height = _safe_randint(rng, height_range_min, height_range_max_val, default=height_range_min)
                            obj = builder._generate_rectangle(current_object_color, width, height, filled=True)
                    elif shape_type == 'line':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        effective_max_line = min(adjusted_max_line, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_line
                        bbox_size = _determine_bbox_size(rng, min_bbox_size, effective_max_bbox_size, adjusted_min_line, effective_max_line, grid_size)
                        length = bbox_size if bbox_size else _safe_randint(rng, adjusted_min_line, effective_max_line, default=adjusted_min_line)
                        if rng.random() < 0.6:
                            direction = rng.choice(['horizontal', 'vertical'])
                            obj = builder._generate_line(current_object_color, length, direction)
                        else:
                            diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                            obj = builder._generate_diagonal_45(current_object_color, length, diagonal_direction)
                    elif shape_type == 'hollow_rectangle':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        effective_max_line = min(adjusted_max_line, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_line
                        bbox_size = _determine_bbox_size(rng, min_bbox_size, effective_max_bbox_size, adjusted_min_line, effective_max_line, grid_size)
                        size = bbox_size if bbox_size else _safe_randint(rng, adjusted_min_line, effective_max_line, default=adjusted_min_line)
                        if rng.random() < 0.6:
                            obj = builder._generate_rectangle(current_object_color, size, size, filled=False)
                        else:
                            width = size
                            height_range_min = max(adjusted_min_line, size // 2) if adjusted_min_line > 0 else max(2, size // 2)
                            height_range_max = min(effective_max_line, size * 2) if effective_max_line > 0 else size * 2
                            height_range_max_val = min(height_range_max, size + 3)
                            height = _safe_randint(rng, height_range_min, height_range_max_val, default=height_range_min)
                            obj = builder._generate_rectangle(current_object_color, width, height, filled=False)
                    elif shape_type == 'cross':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(7, effective_max_bbox_size) if effective_max_bbox_size > 0 else 7
                            size = _safe_randint(rng, 3, max_val, default=3)
                        obj = builder._generate_cross(current_object_color, size)
                    elif shape_type == 't_shape':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                            size = _safe_randint(rng, 3, max_val, default=3)
                        rotation = rng.choice([0, 90, 180, 270])
                        obj = builder._generate_t_shape(current_object_color, size, rotation)
                    elif shape_type == 'diagonal_45':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                            size = _safe_randint(rng, 3, max_val, default=3)
                        direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        obj = builder._generate_diagonal_45(current_object_color, size, direction)
                    elif shape_type == 'circle':
                        obj = builder._generate_circle(current_object_color)
                    elif shape_type == 'l_shape':
                        # L字形: サイズパラメータを考慮
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            adjusted_min_l = calculated_min_size
                            adjusted_max_l = calculated_max_size
                        else:
                            if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                                adjusted_min_l, adjusted_max_l = _calculate_default_size_range(
                                    grid_size[0], grid_size[1], 2, 6
                                )
                            else:
                                adjusted_min_l, adjusted_max_l = (2, 6)

                        effective_max_l = min(adjusted_max_l, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_l
                        width = _safe_randint(rng, adjusted_min_l, effective_max_l, default=adjusted_min_l)
                        height = _safe_randint(rng, adjusted_min_l, effective_max_l, default=adjusted_min_l)
                        obj = builder._generate_l_shape(current_object_color, width, height)
                    elif shape_type == 'triangle':
                        # 三角形: サイズパラメータを考慮
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            adjusted_min_tri = calculated_min_size
                            adjusted_max_tri = calculated_max_size
                        else:
                            if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                                adjusted_min_tri, adjusted_max_tri = _calculate_default_size_range(
                                    grid_size[0], grid_size[1], 3, 6
                                )
                            else:
                                adjusted_min_tri, adjusted_max_tri = (3, 6)

                        effective_max_tri = min(adjusted_max_tri, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_tri
                        size = _safe_randint(rng, adjusted_min_tri, effective_max_tri, default=adjusted_min_tri)
                        obj = builder._generate_triangle(current_object_color, size)
                    elif shape_type == 'single_pixel':
                        obj = builder._generate_single_pixel(current_object_color)
                    elif shape_type == 'checkerboard':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            pattern_size = max(3, min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size)
                        else:
                            pattern_size = max(3, min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6)
                        obj = builder._generate_checkerboard_shape(current_object_color, pattern_size, pattern_size)
                    elif shape_type == 'diagonal':
                        obj = builder._generate_diagonal(current_object_color)
                    elif shape_type == 'diagonal_connected_shape':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                            size = _safe_randint(rng, 3, max_val, default=3)
                        obj = builder._generate_diagonal_connected_shape(current_object_color, size)
                    elif shape_type == 'arrow':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(8, effective_max_bbox_size) if effective_max_bbox_size > 0 else 8
                            size = _safe_randint(rng, 3, max_val, default=3)
                        direction = rng.choice(['up', 'down', 'left', 'right'])
                        obj = builder._generate_arrow(current_object_color, size, direction)
                    elif shape_type == 'diamond':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                            size = _safe_randint(rng, 2, max_val, default=2)
                        filled = rng.random() < 0.7  # 70%の確率で塗りつぶし
                        obj = builder._generate_diamond(current_object_color, size, filled)
                    elif shape_type == 'stairs':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            steps = _safe_randint(rng, 2, min(5, max_val // 2), default=2)
                        else:
                            max_val = min(8, effective_max_bbox_size) if effective_max_bbox_size > 0 else 8
                            steps = _safe_randint(rng, 2, min(5, max_val // 2), default=2)
                        step_size = _safe_randint(rng, 2, 4, default=2)
                        obj = builder._generate_stairs(current_object_color, steps, step_size)
                    elif shape_type == 'zigzag':
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if calculated_min_size is not None and calculated_max_size is not None:
                            max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                            length = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                        else:
                            max_val = min(10, effective_max_bbox_size) if effective_max_bbox_size > 0 else 10
                            length = _safe_randint(rng, 4, max_val, default=4)
                        amplitude = _safe_randint(rng, 1, 3, default=2)
                        obj = builder._generate_zigzag(current_object_color, length, amplitude)
                    elif shape_type == 'simple_object':
                        obj = builder._generate_simple_object(current_object_color, grid_size, total_num_objects, max_size=calculated_max_size)
                    elif shape_type == 'random_pattern':
                        if calculated_min_size is not None and calculated_max_size is not None:
                            adjusted_min_size = calculated_min_size
                            adjusted_max_size = calculated_max_size
                        else:
                            if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                                adjusted_min_size, adjusted_max_size = _calculate_default_size_range(
                                    grid_size[0], grid_size[1], 3, 8
                                )
                            else:
                                adjusted_min_size, adjusted_max_size = (3, 8)
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        effective_max_size = min(adjusted_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else adjusted_max_size
                        width = _safe_randint(rng, adjusted_min_size, effective_max_size, default=adjusted_min_size)
                        height = _safe_randint(rng, adjusted_min_size, effective_max_size, default=adjusted_min_size)
                        # 密度を決定（20%〜90%）
                        # 計算速度を優先する場合は密度指定なし（None）にして、メソッド内のデフォルト処理に任せることも可能
                        density = 0.2 + rng.random() * 0.7
                        obj = builder._generate_random_pattern(current_object_color, width, height, density)
                        if obj and connectivity_constraint is not None:
                            connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                            pixels = obj.get('pixels', [])
                            width_obj = obj.get('width', 1)
                            height_obj = obj.get('height', 1)
                            if pixels and width_obj > 0 and height_obj > 0:
                                connected_pixels = builder._ensure_connected(pixels, width_obj, height_obj, connectivity)
                                obj['pixels'] = connected_pixels
                                obj['area'] = len(connected_pixels)
                        elif obj:
                            pixels = obj.get('pixels', [])
                            width_obj = obj.get('width', 1)
                            height_obj = obj.get('height', 1)
                            if pixels and width_obj > 0 and height_obj > 0:
                                connected_pixels = builder._ensure_connected(pixels, width_obj, height_obj, 4)
                                obj['pixels'] = connected_pixels
                                obj['area'] = len(connected_pixels)
                    elif shape_type in ['u_shape', 'h_shape', 'z_shape']:
                        effective_max_bbox_size = adjusted_max_bbox_size if adjusted_max_bbox_size > 0 else max_bbox_size
                        if hasattr(builder, f'_generate_{shape_type}'):
                            method = getattr(builder, f'_generate_{shape_type}')
                            if calculated_min_size is not None and calculated_max_size is not None:
                                max_val = min(calculated_max_size, effective_max_bbox_size) if effective_max_bbox_size > 0 else calculated_max_size
                                size = _safe_randint(rng, calculated_min_size, max_val, default=calculated_min_size)
                            else:
                                max_val = min(6, effective_max_bbox_size) if effective_max_bbox_size > 0 else 6
                                size = _safe_randint(rng, 3, max_val, default=3)
                            obj = method(current_object_color, size)
                    else:
                        if hasattr(builder, f'_generate_{shape_type}'):
                            method = getattr(builder, f'_generate_{shape_type}')
                            obj = method(current_object_color)
                            # 動的メソッド呼び出しで生成されたオブジェクトのサイズをチェック
                            if obj is not None and calculated_max_size is not None and calculated_max_size > 0:
                                obj_max_dim = max(obj.get('width', 0), obj.get('height', 0))
                        else:
                            obj = builder._generate_simple_object(current_object_color, grid_size, total_num_objects, max_size=calculated_max_size)

                    # shape_typeを元に戻す
                    shape_type = original_shape_type

            # オブジェクト生成後の検証ログ（常に出力）
            # 詳細ログ: オブジェクト生成後の状態
            if obj is None:
                if ENABLE_COLOR_INVESTIGATION_LOGS:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, オブジェクト生成失敗（obj is None） - shape_type={shape_type}, color={current_object_color}, grid_size={grid_size}", flush=True)
            elif not isinstance(obj, dict):
                if ENABLE_COLOR_INVESTIGATION_LOGS:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, オブジェクトが辞書型ではない（無効化） - type={type(obj)}, shape_type={shape_type}", flush=True)
                obj = None  # 無効なデータとして扱う
            else:
                # 必須フィールドのチェック
                required_fields = ['pixels', 'color', 'width', 'height']
                missing_fields = [field for field in required_fields if field not in obj]
                if missing_fields:
                    if ENABLE_COLOR_INVESTIGATION_LOGS:
                        print(f"[色数調査] generate_objects_from_conditions: i={i}, 必須フィールド欠落（無効化） - 欠落={missing_fields}, shape_type={shape_type}", flush=True)
                    obj = None  # 無効なデータとして扱う
                elif not isinstance(obj.get('pixels'), list) or len(obj.get('pixels', [])) == 0:
                    if ENABLE_COLOR_INVESTIGATION_LOGS:
                        print(f"[色数調査] generate_objects_from_conditions: i={i}, pixelsが空（無効化） - pixels={obj.get('pixels')}, shape_type={shape_type}", flush=True)
                    obj = None  # 無効なデータとして扱う
                elif obj.get('width', 0) <= 0 or obj.get('height', 0) <= 0:
                    if ENABLE_COLOR_INVESTIGATION_LOGS:
                        print(f"[色数調査] generate_objects_from_conditions: i={i}, サイズが無効（無効化） - width={obj.get('width')}, height={obj.get('height')}, shape_type={shape_type}", flush=True)
                    obj = None  # 無効なデータとして扱う
                else:
                    # 正常に生成された場合のログ
                    pixels_count = len(obj.get('pixels', []))
                    if ENABLE_COLOR_INVESTIGATION_LOGS:
                        print(f"[色数調査] generate_objects_from_conditions: i={i}, オブジェクト生成成功 - shape_type={shape_type}, color={obj.get('color')}, size={obj.get('width')}x{obj.get('height')}, pixels={pixels_count}", flush=True)

            # オブジェクトが生成された場合の後処理（対称性、穴、連結性）
            if obj:
                # duplicate_mode='forbidden'の場合、色を強制的にcurrent_object_colorに設定
                # オブジェクト生成メソッド内部で色が変更される可能性があるため
                if duplicate_mode == 'forbidden' and object_colors and current_object_color is not None:
                    # オブジェクトの色がcurrent_object_colorと異なる場合は修正
                    if obj.get('color') != current_object_color:
                        old_color = obj.get('color')
                        obj['color'] = current_object_color
                        if i < 5:  # 最初の5回のみ詳細ログ
                            if ENABLE_COLOR_INVESTIGATION_LOGS:
                                print(f"[色数調査] generate_objects_from_conditions: i={i}, 色を修正 - old_color={old_color}, new_color={current_object_color} (duplicate_mode='forbidden')", flush=True)
                # 対称性制約の適用（後処理、ノイズパターン以外）
                # ノイズパターンと対称性は矛盾するため、ノイズパターンの場合はスキップ
                # 既に対称性が適用されているオブジェクト（shape_typeに'symmetric'が含まれる）はスキップ
                if obj and symmetry_constraint and symmetry_constraint != 'none' and shape_type != 'noise':
                    pixels = obj.get('pixels', [])
                    width = obj.get('width', 1)
                    height = obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        # 既に対称性が適用されているかチェック（obj_shape_typeに'symmetric'が含まれているか）
                        obj_shape_type = obj.get('shape_type', '')
                        if 'symmetric' not in obj_shape_type:
                            # 対称性を適用
                            symmetric_pixels = builder._make_symmetric(pixels, width, height, symmetry_constraint)
                            # バウンディングボックスを再計算
                            if symmetric_pixels:
                                min_x = min(x for x, y in symmetric_pixels)
                                max_x = max(x for x, y in symmetric_pixels)
                                min_y = min(y for x, y in symmetric_pixels)
                                max_y = max(y for x, y in symmetric_pixels)
                                obj['width'] = max_x - min_x + 1
                                obj['height'] = max_y - min_y + 1
                                obj['pixels'] = [(x - min_x, y - min_y) for x, y in symmetric_pixels]
                                obj['area'] = len(obj['pixels'])
                                obj['shape_type'] = f'symmetric_{symmetry_constraint}'

                # 穴ありオブジェクトの生成（後処理、対称性処理の後、ノイズパターン以外）
                if allow_holes and shape_type != 'noise':
                    # 既に穴があるかチェック
                    current_hole_count = _count_holes_in_object(obj)
                    if current_hole_count == 0:
                        # 穴がない場合、穴を作成
                        pixels = obj.get('pixels', [])
                        width = obj.get('width', 1)
                        height = obj.get('height', 1)
                        obj_color = obj.get('color', current_object_color)

                        if pixels and width > 1 and height > 1:
                            # 穴の数を決定（1-3のランダム）
                            hole_count = rng.randint(1, 3)

                            # 対称性が適用されているかチェック
                            obj_shape_type = obj.get('shape_type', '')
                            applied_symmetry = None
                            if 'symmetric' in obj_shape_type:
                                # symmetric_vertical, symmetric_horizontal, symmetric_bothから抽出
                                if 'symmetric_vertical' in obj_shape_type:
                                    applied_symmetry = 'vertical'
                                elif 'symmetric_horizontal' in obj_shape_type:
                                    applied_symmetry = 'horizontal'
                                elif 'symmetric_both' in obj_shape_type:
                                    applied_symmetry = 'both'
                                else:
                                    # symmetry_constraintが指定されている場合はそれを使用
                                    applied_symmetry = symmetry_constraint if symmetry_constraint and symmetry_constraint != 'none' else None
                            else:
                                # 対称性が適用されていない場合
                                applied_symmetry = symmetry_constraint if symmetry_constraint and symmetry_constraint != 'none' else None

                            # 内部領域（境界を除く）にピクセルがあるかチェック
                            inner_pixels = [(x, y) for x, y in pixels
                                           if 1 <= x < width - 1 and 1 <= y < height - 1]

                            if inner_pixels:
                                # 穴が作成可能（既存ピクセルから削除）
                                new_pixels = builder._add_holes_with_symmetry(
                                    pixels, width, height, applied_symmetry, hole_count, obj_color
                                )
                            else:
                                # 穴が作成不可能（ピクセルを追加してから穴を作成）
                                new_pixels = builder._add_pixels_for_holes_with_symmetry(
                                    pixels, width, height, applied_symmetry, hole_count, obj_color
                                )

                            # バウンディングボックスを再計算
                            if new_pixels:
                                min_x = min(x for x, y in new_pixels)
                                max_x = max(x for x, y in new_pixels)
                                min_y = min(y for x, y in new_pixels)
                                max_y = max(y for x, y in new_pixels)
                                obj['width'] = max_x - min_x + 1
                                obj['height'] = max_y - min_y + 1
                                obj['pixels'] = [(x - min_x, y - min_y) for x, y in new_pixels]
                                obj['area'] = len(obj['pixels'])
                                obj['holes'] = hole_count
                                if 'hollow' not in obj.get('shape_type', ''):
                                    obj['shape_type'] = obj.get('shape_type', '') + '_with_holes'

                # 連結性制約の適用（ノイズパターン以外、すべての生成メソッドに適用）
                # random_patternは既に連結性が保証されているためスキップ
                if shape_type not in ['noise', 'random_pattern'] and connectivity_constraint is not None:
                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                    pixels = obj.get('pixels', [])
                    width = obj.get('width', 1)
                    height = obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        obj['pixels'] = builder._ensure_connected(pixels, width, height, connectivity)

                # 後処理後の色チェック（念のため）
                # object_colorsが指定されている場合、色がobject_colorsに含まれていない場合はcurrent_object_colorに修正
                if object_colors and current_object_color is not None:
                    obj_color = obj.get('color')
                    if obj_color is not None and obj_color not in object_colors:
                        old_color = obj_color
                        obj['color'] = current_object_color
                        if i < 5:  # 最初の5回のみ詳細ログ
                            if ENABLE_COLOR_INVESTIGATION_LOGS:
                                print(f"[色数調査] generate_objects_from_conditions: i={i}, 後処理後の色修正 - old_color={old_color} (object_colorsに不在), new_color={current_object_color}", flush=True)

            # 位置情報を設定（オブジェクトが生成された場合のみ）
            if obj:
                # 位置情報を設定（スコア制配置システムを使用、重なっても最適な位置に配置）
                if grid_width and grid_height:
                    obj_width = obj.get('width', 1)
                    obj_height = obj.get('height', 1)
                    obj_color = obj.get('color', background_color)

                    # スコア制配置システム用のデータ構造を準備
                    # 既に配置されたピクセルを追跡（スコア計算に必要）
                    placed_pixels: Set[Tuple[int, int]] = set()
                    placed_colors_map: Dict[int, Set[Tuple[int, int]]] = {}
                    placed_bboxes = existing_positions.copy()

                    # 既存オブジェクトのピクセルを収集
                    for existing_obj in result_objects:
                        if existing_obj is None:
                            continue
                        if 'x' in existing_obj and 'y' in existing_obj:
                            ex = existing_obj.get('x', 0)
                            ey = existing_obj.get('y', 0)
                            ex_color = existing_obj.get('color', background_color)
                            ex_pixels = existing_obj.get('pixels', [])
                            for px, py in ex_pixels:
                                gx, gy = ex + px, ey + py
                                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                                    placed_pixels.add((gx, gy))
                                    if ex_color not in placed_colors_map:
                                        placed_colors_map[ex_color] = set()
                                    placed_colors_map[ex_color].add((gx, gy))

                    # グリッドリストを作成（スコア計算に必要、背景色で初期化）
                    if grid_width is not None and grid_height is not None:
                        grid_list = [[background_color for _ in range(grid_width)] for _ in range(grid_height)]
                    else:
                        grid_list = []  # デフォルトを空リストに設定

                    # 既存オブジェクトをグリッドに配置（スコア計算の精度向上）
                    for existing_obj in result_objects:
                        if existing_obj is None:
                            continue
                        if 'x' in existing_obj and 'y' in existing_obj:
                            ex = existing_obj.get('x', 0)
                            ey = existing_obj.get('y', 0)
                            ex_color = existing_obj.get('color', background_color)
                            ex_pixels = existing_obj.get('pixels', [])
                            for px, py in ex_pixels:
                                gx, gy = ex + px, ey + py
                                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                                    grid_list[gy][gx] = ex_color

                    # スコア制配置システムで最適な位置を探索（重なっても配置可能）
                    best_position = _find_best_position(
                        grid_list=grid_list,
                        obj=obj,
                        width=grid_width,
                        height=grid_height,
                        background_color=background_color,
                        placed_pixels=placed_pixels,
                        placed_colors_map=placed_colors_map,
                        placed_bboxes=placed_bboxes,
                        obj_color=obj_color,
                        rng=rng,
                        min_spacing=min_spacing
                    )

                    if best_position:
                        x, y = best_position
                        obj['x'] = x
                        obj['y'] = y
                        obj['position'] = (x, y)
                        # 既存オブジェクトリストに位置情報を追加（次のオブジェクト生成時に考慮）
                        existing_positions.append((x, y, obj_width, obj_height))
                    else:
                        # フォールバック: ランダムな位置に配置（重なっても配置）
                        x = rng.randint(0, max(1, grid_width - obj_width))
                        y = rng.randint(0, max(1, grid_height - obj_height))
                        obj['x'] = x
                        obj['y'] = y
                        obj['position'] = (x, y)
                        existing_positions.append((x, y, obj_width, obj_height))

            # 詳細ログ: オブジェクト追加前の確認
            if ENABLE_COLOR_INVESTIGATION_LOGS and (i < 5 or obj is not None):  # i<5の場合は常に、それ以外はobjがNoneでない場合のみ
                if obj is None:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, obj is None - result_objectsには追加しません（ただしresult_objects.append(obj)は実行されます）", flush=True)
                else:
                    obj_color = obj.get('color') if obj else None
                    obj_pixels_count = len(obj.get('pixels', [])) if obj else 0
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, obj追加前 - color={obj_color}, pixels={obj_pixels_count}, result_objects現在の数={len(result_objects)}", flush=True)

            result_objects.append(obj)

            # 詳細ログ: オブジェクト追加後の確認
            if ENABLE_COLOR_INVESTIGATION_LOGS and (i < 5 or obj is not None):  # i<5の場合は常に、それ以外はobjがNoneでない場合のみ
                if obj is not None:
                    print(f"[色数調査] generate_objects_from_conditions: i={i}, obj追加後 - result_objects現在の数={len(result_objects)}", flush=True)

            # 最適化: 形状シグネチャの事前取得を削除（必要時に取得する方式に変更済み）
            # 生成されたオブジェクトを既存オブジェクトとして使用する場合は、
            # 次回のオブジェクト生成時に必要に応じて形状シグネチャを取得する

            # オブジェクト生成の処理時間を記録
            object_gen_elapsed = time.time() - object_gen_start
            object_generation_times.append(object_gen_elapsed)

            # 0.1秒以上の場合は警告ログを出力
            if object_gen_elapsed > 0.1:
                print(f"[警告] generate_objects_from_conditions: i={i} の処理時間が長い: {object_gen_elapsed:.3f}秒", flush=True)

    # ループ全体の処理時間を記録
    loop_elapsed = time.time() - loop_start_time
    profiler = get_profiler()
    profiler.record_timing("generate_objects_from_conditions_loop", loop_elapsed, f"num_objects={num_objects}")

    # オブジェクト生成時間の統計を記録
    if object_generation_times:
        avg_obj_time = sum(object_generation_times) / len(object_generation_times)
        max_obj_time = max(object_generation_times)
        profiler.record_timing("generate_objects_from_conditions_per_object", avg_obj_time, "average")
        profiler.record_timing("generate_objects_from_conditions_per_object", max_obj_time, "max")

    # 詳細ログ: ループ終了後の結果
    valid_objects_count = len([obj for obj in result_objects if obj is not None])
    none_objects_count = len([obj for obj in result_objects if obj is None])
    if ENABLE_COLOR_INVESTIGATION_LOGS:
        print(f"[色数調査] generate_objects_from_conditions: ループ終了 - 要求数={num_objects}, 生成数={len(result_objects)}, 有効オブジェクト数={valid_objects_count}, None数={none_objects_count}", flush=True)

    gen_elapsed = time.time() - gen_start_time
    # プロファイラーに記録
    profiler = get_profiler()
    context_str = f"num_objects={num_objects}, colors={len(object_colors) if object_colors else 0}"
    profiler.record_timing("generate_objects_from_conditions", gen_elapsed, context_str)

    # 1秒以上の場合は警告ログを出力
    if gen_elapsed > 1.0:
        print(f"[警告] generate_objects_from_conditions の処理時間が長い: {gen_elapsed:.3f}秒 (num_objects={num_objects})", flush=True)

    return result_objects
