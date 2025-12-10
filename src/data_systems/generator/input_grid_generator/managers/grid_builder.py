"""
グリッド構築関数

条件を受け取ってグリッドを構築する関数群
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from .core_placement_manager import CorePlacementManager
from .auto_placement import build_grid_with_auto_placement, _find_best_position, _place_object_pixels


def build_grid_from_conditions(
    width: int,
    height: int,
    background_color: int = 0,
    objects: Optional[List[Dict]] = None,
    placement_position: Optional[Tuple[int, int]] = None,
    placement_pattern: str = 'random',
    seed: Optional[int] = None,
    **kwargs
) -> List[List[int]]:
    """条件を受け取ってグリッドリストを生成

    Args:
        width: グリッド幅
        height: グリッド高さ
        background_color: 背景色
        objects: 配置するオブジェクトのリスト（Noneの場合は空のグリッド）
        placement_position: 配置位置（Noneの場合は自動配置）
            - (x, y) のタプル：指定位置に配置を試行
            - None: 配置パターンに従って配置
        placement_pattern: 配置パターン（placement_positionがNoneの場合のみ有効）
            - 'random': ランダム配置
            - 'grid': グリッド配置
            - 'spiral': 螺旋配置
            - 'cluster': クラスタ配置
            - 'border': 境界配置
            - 'center': 中央配置
            - 'symmetry': 対称配置
            - 'arc_pattern': ARCパターン配置
            - 'structured': 構造化配置
        seed: 乱数シード
        **kwargs: その他の条件（将来の拡張用）
            - allow_overlap: 重複を許可するか
            - adjacency_avoidance: 隣接回避率

    Returns:
        グリッドリスト（List[List[int]]）
    """
    # 空のグリッドを作成（背景色で埋める）
    grid = np.full((height, width), background_color, dtype=int)
    grid_list = grid.tolist()

    # オブジェクトがない場合は空のグリッドを返す
    if not objects:
        return grid_list

    # 配置マネージャーを初期化
    placement_manager = CorePlacementManager(seed=seed)

    # 配置位置が指定されている場合
    if placement_position is not None:
        x, y = placement_position
        for obj in objects:
            # オブジェクトのサイズを取得
            obj_width = obj.get('width', 1)
            obj_height = obj.get('height', 1)

            # 配置可能かチェック
            if placement_manager._can_place_object(grid_list, obj, x, y):
                placement_manager._place_object_at_position(grid_list, obj, x, y)
            else:
                # 配置できない場合は、最初の空き位置を探す
                placed = False
                for search_y in range(height - obj_height + 1):
                    for search_x in range(width - obj_width + 1):
                        if placement_manager._can_place_object(grid_list, obj, search_x, search_y):
                            placement_manager._place_object_at_position(grid_list, obj, search_x, search_y)
                            placed = True
                            break
                    if placed:
                        break
    else:
        # 配置パターンに従って配置
        placement_manager.place_objects(grid_list, objects, pattern=placement_pattern)

    return grid_list


def build_grid(
    width: int,
    height: int,
    background_grid_pattern: Optional[List[List[int]]] = None,
    objects: List[Dict] = None,
    seed: Optional[int] = None,
    min_spacing: int = 0
) -> List[List[int]]:
    """オブジェクトを配置したグリッドを生成

    Args:
        width: グリッド幅
        height: グリッド高さ
        background_grid_pattern: 背景グリッドパターン（2次元の色情報、Noneの場合は単色の背景）
            - 例: [[0]] → 単色の背景（色0）
            - 例: [[0, 1], [1, 0]] → チェッカーボードパターン
            - パターンは座標(0,0)から左右上下に繰り返され、width x heightで切り出される
        objects: 配置するオブジェクトのリスト（各オブジェクトはx, y座標情報を持つことがある）
        seed: 乱数シード

    Returns:
        グリッドリスト（List[List[int]]）
    """
    import random
    rng = random.Random(seed)

    # 背景グリッドを作成
    if background_grid_pattern is None:
        # パターンが指定されていない場合はデフォルト（色0の単色）
        background_grid_pattern = [[0]]

    # パターンをnumpy配列に変換
    pattern_array = np.array(background_grid_pattern, dtype=int)
    pattern_height, pattern_width = pattern_array.shape

    # パターンを繰り返して、width x heightの背景グリッドを作成
    # numpyのtileを使用して繰り返し、その後必要な部分を切り出す
    # 繰り返し回数を計算（余裕を持たせる）
    repeat_y = (height // pattern_height) + 2
    repeat_x = (width // pattern_width) + 2

    # パターンを繰り返す
    tiled_pattern = np.tile(pattern_array, (repeat_y, repeat_x))

    # width x heightの範囲を切り出す（座標0,0から）
    grid = tiled_pattern[:height, :width].copy()
    grid_list = grid.tolist()

    if not objects:
        return grid_list

    # 背景色を取得（パターンの最も頻繁な色、またはパターンが単色の場合はその色）
    # スコア計算のために使用（背景パターンがある場合も、主要な背景色を推論）
    background_color = int(np.bincount(pattern_array.flatten()).argmax()) if pattern_array.size > 0 else 0

    # 既に配置されたオブジェクトを記録
    placed_pixels: set = set()  # 配置されたピクセルのセット
    placed_colors_map: Dict[int, set] = {}  # 色ごとの配置ピクセル
    placed_bboxes: List[Tuple[int, int, int, int]] = []  # 配置されたバウンディングボックスのリスト (x, y, width, height)

    # オブジェクトを順番に配置
    for obj in objects:
        # Noneチェック
        if obj is None:
            continue
        pixels = obj.get('pixels', [])
        if not pixels:
            continue

        obj_color = obj.get('color', 1)

        # オブジェクトがx, y座標を持っている場合、それを使用
        obj_x = obj.get('x')
        obj_y = obj.get('y')

        if obj_x is not None and obj_y is not None:
            # 指定された座標に配置
            _place_object_pixels(grid_list, obj, obj_x, obj_y, obj_color)
            # 注: positionフィールドは削除（x, yフィールドが優先的に使用されるため不要）

            # 配置されたピクセルを記録
            for px, py in pixels:
                gx, gy = obj_x + px, obj_y + py
                if 0 <= gx < width and 0 <= gy < height:
                    placed_pixels.add((gx, gy))
                    if obj_color not in placed_colors_map:
                        placed_colors_map[obj_color] = set()
                    placed_colors_map[obj_color].add((gx, gy))

            # バウンディングボックスを記録
            min_px = min(px for px, py in pixels)
            max_px = max(px for px, py in pixels)
            min_py = min(py for px, py in pixels)
            max_py = max(py for px, py in pixels)
            bbox_width = max_px - min_px + 1
            bbox_height = max_py - min_py + 1
            placed_bboxes.append((obj_x, obj_y, bbox_width, bbox_height))
        else:
            # 座標情報がない場合は自動配置（スコア形式の位置決め）
            # 注意: 背景パターンがある場合でも、主要な背景色を推論して使用
            best_position = _find_best_position(
                grid_list=grid_list,
                obj=obj,
                width=width,
                height=height,
                background_color=background_color,
                placed_pixels=placed_pixels,
                placed_colors_map=placed_colors_map,
                placed_bboxes=placed_bboxes,
                obj_color=obj_color,
                rng=rng,
                min_spacing=min_spacing
            )

            if best_position is not None:
                x, y = best_position
                # オブジェクトを配置
                _place_object_pixels(grid_list, obj, x, y, obj_color)
                # 注: positionフィールドは削除（x, yフィールドが優先的に使用されるため不要）
                obj['x'] = x  # x座標を保存
                obj['y'] = y  # y座標を保存

                # 配置されたピクセルを記録
                for px, py in pixels:
                    gx, gy = x + px, y + py
                    if 0 <= gx < width and 0 <= gy < height:
                        placed_pixels.add((gx, gy))
                        if obj_color not in placed_colors_map:
                            placed_colors_map[obj_color] = set()
                        placed_colors_map[obj_color].add((gx, gy))

                # バウンディングボックスを記録
                min_px = min(px for px, py in pixels)
                max_px = max(px for px, py in pixels)
                min_py = min(py for px, py in pixels)
                max_py = max(py for px, py in pixels)
                bbox_width = max_px - min_px + 1
                bbox_height = max_py - min_py + 1
                placed_bboxes.append((x, y, bbox_width, bbox_height))

    return grid_list


def set_position(
    width: int,
    height: int,
    background_color: int,
    objects: List[Dict],
    existing_objects: Optional[List[Dict]] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    seed: Optional[int] = None,
    min_spacing: int = 0
) -> List[Dict]:
    """既存オブジェクトを配置し、その後新しいオブジェクトを配置して座標情報を追加

    Args:
        width: グリッド幅
        height: グリッド高さ
        background_color: 背景色
        objects: 配置するオブジェクトのリスト
        existing_objects: 既に配置済みの既存オブジェクトのリスト（x, y座標情報を持つ）
        x: 配置X座標（指定された場合、objectsの全オブジェクトに適用）
        y: 配置Y座標（指定された場合、objectsの全オブジェクトに適用）
        seed: 乱数シード

    Returns:
        配置後のXY座標情報が追加されたobjectsリスト
    """
    import random
    rng = random.Random(seed)

    # 空のグリッドを作成
    grid = np.full((height, width), background_color, dtype=int)
    grid_list = grid.tolist()

    # 既に配置されたオブジェクトを記録
    placed_pixels: set = set()  # 配置されたピクセルのセット
    placed_colors_map: Dict[int, set] = {}  # 色ごとの配置ピクセル
    placed_bboxes: List[Tuple[int, int, int, int]] = []  # 配置されたバウンディングボックスのリスト (x, y, width, height)

    # まず、既存オブジェクトを配置
    if existing_objects:
        for existing_obj in existing_objects:
            # 既存オブジェクトがx, y座標を持っている場合、それを使用
            # またはpositionフィールドから取得（既存データとの互換性のため）
            existing_x = existing_obj.get('x')
            existing_y = existing_obj.get('y')

            # positionフィールドから取得（x, yがない場合）
            if existing_x is None or existing_y is None:
                position = existing_obj.get('position')
                if position:
                    existing_x, existing_y = position[0], position[1]

            if existing_x is not None and existing_y is not None:
                obj_color = existing_obj.get('color', 1)
                pixels = existing_obj.get('pixels', [])

                # 既存オブジェクトを配置
                _place_object_pixels(grid_list, existing_obj, existing_x, existing_y, obj_color)
                # 注: positionフィールドは削除（x, yフィールドが優先的に使用されるため不要）
                existing_obj['x'] = existing_x  # x座標を保存
                existing_obj['y'] = existing_y  # y座標を保存

                # 配置されたピクセルを記録
                for px, py in pixels:
                    gx, gy = existing_x + px, existing_y + py
                    if 0 <= gx < width and 0 <= gy < height:
                        placed_pixels.add((gx, gy))
                        if obj_color not in placed_colors_map:
                            placed_colors_map[obj_color] = set()
                        placed_colors_map[obj_color].add((gx, gy))

                # バウンディングボックスを記録
                min_px = min(px for px, py in pixels)
                max_px = max(px for px, py in pixels)
                min_py = min(py for px, py in pixels)
                max_py = max(py for px, py in pixels)
                bbox_width = max_px - min_px + 1
                bbox_height = max_py - min_py + 1
                placed_bboxes.append((existing_x, existing_y, bbox_width, bbox_height))

    # 次に、新しいオブジェクトを配置
    result_objects = []
    for obj in objects:
        if obj is None:
            result_objects.append(obj)
            continue
        pixels = obj.get('pixels', [])
        if not pixels:
            result_objects.append(obj)
            continue

        obj_color = obj.get('color', 1)

        # 引数x, yが指定されている場合は優先して使用
        if x is not None and y is not None:
            # 指定された座標に配置
            _place_object_pixels(grid_list, obj, x, y, obj_color)
            # 注: positionフィールドは削除（x, yフィールドが優先的に使用されるため不要）
            obj['x'] = x  # x座標を保存
            obj['y'] = y  # y座標を保存

            # 配置されたピクセルを記録
            for px, py in pixels:
                gx, gy = x + px, y + py
                if 0 <= gx < width and 0 <= gy < height:
                    placed_pixels.add((gx, gy))
                    if obj_color not in placed_colors_map:
                        placed_colors_map[obj_color] = set()
                    placed_colors_map[obj_color].add((gx, gy))

            # バウンディングボックスを記録
            min_px = min(px for px, py in pixels)
            max_px = max(px for px, py in pixels)
            min_py = min(py for px, py in pixels)
            max_py = max(py for px, py in pixels)
            bbox_width = max_px - min_px + 1
            bbox_height = max_py - min_py + 1
            placed_bboxes.append((x, y, bbox_width, bbox_height))

            result_objects.append(obj)
        else:
            # 引数x, yが指定されていない場合、オブジェクトのx, y情報を無視して自動配置
            best_position = _find_best_position(
                grid_list=grid_list,
                obj=obj,
                width=width,
                height=height,
                background_color=background_color,
                placed_pixels=placed_pixels,
                placed_colors_map=placed_colors_map,
                placed_bboxes=placed_bboxes,
                obj_color=obj_color,
                rng=rng,
                min_spacing=min_spacing
            )

            if best_position is not None:
                placement_x, placement_y = best_position
                # オブジェクトを配置
                _place_object_pixels(grid_list, obj, placement_x, placement_y, obj_color)
                # 注: positionフィールドは削除（x, yフィールドが優先的に使用されるため不要）
                obj['x'] = placement_x  # x座標を保存
                obj['y'] = placement_y  # y座標を保存

                # 配置されたピクセルを記録
                for px, py in pixels:
                    gx, gy = placement_x + px, placement_y + py
                    if 0 <= gx < width and 0 <= gy < height:
                        placed_pixels.add((gx, gy))
                        if obj_color not in placed_colors_map:
                            placed_colors_map[obj_color] = set()
                        placed_colors_map[obj_color].add((gx, gy))

                # バウンディングボックスを記録
                min_px = min(px for px, py in pixels)
                max_px = max(px for px, py in pixels)
                min_py = min(py for px, py in pixels)
                max_py = max(py for px, py in pixels)
                bbox_width = max_px - min_px + 1
                bbox_height = max_py - min_py + 1
                placed_bboxes.append((placement_x, placement_y, bbox_width, bbox_height))

            result_objects.append(obj)

    return result_objects
