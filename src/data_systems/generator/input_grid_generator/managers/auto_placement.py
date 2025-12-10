"""
自動配置機能

優先順位に基づいてオブジェクトを最適な位置に自動配置する
"""
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import random


def build_grid_with_auto_placement(
    width: int,
    height: int,
    background_color: int,
    objects: List[Dict],
    existing_objects: Optional[List[Dict]] = None,
    seed: Optional[int] = None
) -> List[List[int]]:
    """優先順位に基づいてオブジェクトを自動配置

    優先順位:
    1. グリッドサイズ内の位置に配置（一部はみ出してもいい）
    2. 他のオブジェクトとのピクセル単位での重なりを最小にする
    3. 他のオブジェクトとのピクセル単位で重ならない
    4. 同じ色のオブジェクトとピクセル単位で隣接しない
    5. グリッドサイズ内にすべての形状が収まる

    Args:
        width: グリッド幅
        height: グリッド高さ
        background_color: 背景色
        objects: 配置するオブジェクトのリスト
        existing_objects: 既に配置済みの既存オブジェクトのリスト（position情報が必要）
        seed: 乱数シード

    Returns:
        グリッドリスト（List[List[int]]）
    """
    import random
    rng = random.Random(seed)

    # 空のグリッドを作成
    grid = np.full((height, width), background_color, dtype=int)
    grid_list = grid.tolist()

    # 既に配置されたオブジェクトを記録
    placed_objects: List[Dict] = []
    placed_pixels: Set[Tuple[int, int]] = set()  # 配置されたピクセルのセット
    placed_colors_map: Dict[int, Set[Tuple[int, int]]] = {}  # 色ごとの配置ピクセル
    placed_bboxes: List[Tuple[int, int, int, int]] = []  # 配置されたバウンディングボックスのリスト (x, y, width, height)

    # 既存オブジェクトを先に配置
    if existing_objects:
        for existing_obj in existing_objects:
            if 'position' not in existing_obj:
                continue

            x, y = existing_obj['position']
            obj_color = existing_obj.get('color', 1)
            pixels = existing_obj.get('pixels', [])

            # 既存オブジェクトを配置
            _place_object_pixels(grid_list, existing_obj, x, y, obj_color)
            placed_objects.append(existing_obj)

            # 配置されたピクセルを記録
            for px, py in pixels:
                gx, gy = x + px, y + py
                if 0 <= gx < width and 0 <= gy < height:
                    placed_pixels.add((gx, gy))
                    if obj_color not in placed_colors_map:
                        placed_colors_map[obj_color] = set()
                    placed_colors_map[obj_color].add((gx, gy))

            # バウンディングボックスを記録
            if pixels:
                min_px = min(px for px, py in pixels)
                max_px = max(px for px, py in pixels)
                min_py = min(py for px, py in pixels)
                max_py = max(py for px, py in pixels)
                bbox_width = max_px - min_px + 1
                bbox_height = max_py - min_py + 1
                placed_bboxes.append((x, y, bbox_width, bbox_height))

    if not objects:
        return grid_list

    # 新しいオブジェクトを順番に配置
    # 注意: 各オブジェクトの配置時には、既存オブジェクト（existing_objects）と
    #       既に配置済みの新しいオブジェクト（objects内の先に配置されたもの）の
    #       両方が考慮される（placed_pixelsとplaced_colors_mapが更新されるため）
    for obj in objects:
        if 'position' in obj:
            # 既に配置位置が決まっている場合はスキップ（既に配置済みとして記録）
            placed_objects.append(obj)
            # 既に配置済みのピクセルも記録
            pixels = obj.get('pixels', [])
            x, y = obj['position']
            obj_color = obj.get('color', 1)
            for px, py in pixels:
                gx, gy = x + px, y + py
                if 0 <= gx < width and 0 <= gy < height:
                    placed_pixels.add((gx, gy))
                    if obj_color not in placed_colors_map:
                        placed_colors_map[obj_color] = set()
                    placed_colors_map[obj_color].add((gx, gy))
            # バウンディングボックスを記録
            if pixels:
                min_px = min(px for px, py in pixels)
                max_px = max(px for px, py in pixels)
                min_py = min(py for px, py in pixels)
                max_py = max(py for px, py in pixels)
                bbox_width = max_px - min_px + 1
                bbox_height = max_py - min_py + 1
                placed_bboxes.append((x, y, bbox_width, bbox_height))
            continue

        pixels = obj.get('pixels', [])
        if not pixels:
            continue

        obj_color = obj.get('color', 1)

        # 最適な配置位置を探索
        # placed_pixelsとplaced_colors_mapには、既存オブジェクトと
        # 既に配置された新しいオブジェクトの両方の情報が含まれている
        best_position = _find_best_position(
            grid_list=grid_list,
            obj=obj,
            width=width,
            height=height,
            background_color=background_color,
            placed_pixels=placed_pixels,  # 既存 + 既に配置された新規オブジェクトのピクセル
            placed_colors_map=placed_colors_map,  # 既存 + 既に配置された新規オブジェクトの色情報
            placed_bboxes=placed_bboxes,  # 既存 + 既に配置された新規オブジェクトのバウンディングボックス
            obj_color=obj_color,
            rng=rng,
            min_spacing=0  # build_grid_with_auto_placementではデフォルト0（必要に応じて拡張可能）
        )

        if best_position is not None:
            x, y = best_position
            # オブジェクトを配置
            _place_object_pixels(grid_list, obj, x, y, obj_color)
            obj['position'] = (x, y)
            placed_objects.append(obj)

            # 配置されたピクセルを記録（次のオブジェクトの配置時に考慮される）
            for px, py in pixels:
                gx, gy = x + px, y + py
                if 0 <= gx < width and 0 <= gy < height:
                    placed_pixels.add((gx, gy))
                    if obj_color not in placed_colors_map:
                        placed_colors_map[obj_color] = set()
                    placed_colors_map[obj_color].add((gx, gy))

            # バウンディングボックスを記録（次のオブジェクトの配置時に考慮される）
            min_px = min(px for px, py in pixels)
            max_px = max(px for px, py in pixels)
            min_py = min(py for px, py in pixels)
            max_py = max(py for px, py in pixels)
            bbox_width = max_px - min_px + 1
            bbox_height = max_py - min_py + 1
            placed_bboxes.append((x, y, bbox_width, bbox_height))

    return grid_list


def _calculate_bbox_distance(
    x1: int, y1: int, w1: int, h1: int,
    x2: int, y2: int, w2: int, h2: int
) -> int:
    """2つのバウンディングボックス間の最小距離を計算

    Args:
        x1, y1, w1, h1: 最初のバウンディングボックス（x, y, width, height）
        x2, y2, w2, h2: 2番目のバウンディングボックス（x, y, width, height）

    Returns:
        バウンディングボックス間の最小距離（重複している場合は0）
    """
    # 各バウンディングボックスの境界を計算
    x1_min, x1_max = x1, x1 + w1 - 1
    y1_min, y1_max = y1, y1 + h1 - 1
    x2_min, x2_max = x2, x2 + w2 - 1
    y2_min, y2_max = y2, y2 + h2 - 1

    # 重複している場合は距離0
    if (x1_max >= x2_min and x1_min <= x2_max and
        y1_max >= y2_min and y1_min <= y2_max):
        return 0

    # X方向の距離
    if x1_max < x2_min:
        dx = x2_min - x1_max - 1
    elif x2_max < x1_min:
        dx = x1_min - x2_max - 1
    else:
        dx = 0

    # Y方向の距離
    if y1_max < y2_min:
        dy = y2_min - y1_max - 1
    elif y2_max < y1_min:
        dy = y1_min - y2_max - 1
    else:
        dy = 0

    # マンハッタン距離を返す（チェビシェフ距離でも可）
    return max(dx, dy)


def _find_best_position(
    grid_list: List[List[int]],
    obj: Dict,
    width: int,
    height: int,
    background_color: int,
    placed_pixels: Set[Tuple[int, int]],
    placed_colors_map: Dict[int, Set[Tuple[int, int]]],
    placed_bboxes: List[Tuple[int, int, int, int]],
    obj_color: int,
    rng,
    min_spacing: int = 0
) -> Optional[Tuple[int, int]]:
    """最適な配置位置を探索（最適化版：サンプリング）

    優先順位に基づいてスコアを計算し、最適な位置を返す。
    全候補位置を評価するのではなく、ランダムサンプリングで候補位置を削減する。
    """
    pixels = obj.get('pixels', [])
    if not pixels:
        return None

    # オブジェクトのバウンディングボックスを計算
    min_x = min(x for x, y in pixels)
    max_x = max(x for x, y in pixels)
    min_y = min(y for x, y in pixels)
    max_y = max(y for x, y in pixels)
    obj_width = max_x - min_x + 1
    obj_height = max_y - min_y + 1

    best_position: Optional[Tuple[int, int]] = None
    best_score: float = float('inf')

    # 探索範囲を拡大（一部はみ出してもいいため、少し外側も探索）
    margin = 2
    search_min_x = -margin
    search_max_x = width + margin - obj_width
    search_min_y = -margin
    search_max_y = height + margin - obj_height

    # サンプリング数を決定（グリッドサイズに応じて調整）
    # グリッド面積に基づいてサンプリング数を決定
    grid_area = width * height

    # グリッドサイズに応じたサンプリング数（削減版）:
    # - 小さいグリッド（面積 < 100）: 10-15個
    # - 中程度のグリッド（100 <= 面積 < 400）: 15-25個
    # - 大きいグリッド（400 <= 面積 < 900）: 25-35個
    # - 非常に大きいグリッド（面積 >= 900）: 35-45個
    # 早期終了があるため、少ないサンプル数でも十分な可能性が高い
    if grid_area < 100:
        sample_count = max(10, min(15, grid_area // 6))
    elif grid_area < 400:
        sample_count = max(15, min(25, grid_area // 16))
    elif grid_area < 900:
        sample_count = max(25, min(35, grid_area // 26))
    else:
        sample_count = max(35, min(45, grid_area // 30))

    # 全候補位置数を計算（上限チェック用）
    total_candidates = (search_max_x - search_min_x + 1) * (search_max_y - search_min_y + 1)

    # 全候補数がサンプリング数以下の場合は全て使用
    if total_candidates <= sample_count:
        sample_count = total_candidates

    # ランダムサンプリング: 全候補位置からランダムに選択
    all_candidates = []
    for y in range(search_min_y, search_max_y + 1):
        for x in range(search_min_x, search_max_x + 1):
            all_candidates.append((x, y))

    if len(all_candidates) <= sample_count:
        # 候補数が少ない場合は全て使用
        candidates_list = all_candidates
    else:
        # ランダムサンプリング
        sample_size = min(sample_count, len(all_candidates))
        if sample_size > 0 and len(all_candidates) > 0:
            candidates_list = rng.sample(all_candidates, sample_size)
        else:
            candidates_list = []

    # ランダムに並び替え（多様性のため）
    rng.shuffle(candidates_list)

    # 十分良いスコアの閾値（早期終了用）
    # score < -50.0 は重なりなし + グリッド内収まり良好 + ボーナス
    early_exit_threshold = -50.0

    # サンプリングされた候補位置を評価
    for x, y in candidates_list:
        score = _calculate_placement_score(
            grid_list=grid_list,
            obj=obj,
            x=x,
            y=y,
            width=width,
            height=height,
            background_color=background_color,
            placed_pixels=placed_pixels,
            placed_colors_map=placed_colors_map,
            placed_bboxes=placed_bboxes,
            obj_color=obj_color,
            min_spacing=min_spacing,
            rng=rng  # ランダム化のためにrngを渡す
        )

        if score < best_score:
            best_score = score
            best_position = (x, y)

            # 早期終了: 十分良い候補が見つかったら探索を終了
            if score < early_exit_threshold:
                break

    return best_position


def _calculate_placement_score(
    grid_list: List[List[int]],
    obj: Dict,
    x: int,
    y: int,
    width: int,
    height: int,
    background_color: int,
    placed_pixels: Set[Tuple[int, int]],
    placed_colors_map: Dict[int, Set[Tuple[int, int]]],
    placed_bboxes: List[Tuple[int, int, int, int]],
    obj_color: int,
    min_spacing: int = 0,
    rng=None
) -> float:
    """配置スコアを計算（小さいほど良い）

    優先順位:
    1. グリッドサイズ内の位置に配置（一部はみ出してもいい）
    2. 他のオブジェクトとピクセル単位での重なりを最小にする
    3. 他のオブジェクトとピクセル単位で重ならない
    4. 同じ色のオブジェクトとピクセル単位で隣接しない
    5. グリッドサイズ内にすべての形状が収まる
    """
    pixels = obj.get('pixels', [])
    if not pixels:
        return float('inf')

    # rngがNoneの場合はランダム性を無効化（後方互換性のため）
    use_randomization = rng is not None
    if not use_randomization:
        import random
        rng = random.Random()

    score = 0.0

    # オブジェクトのバウンディングボックスを計算
    obj_max_x = max(px for px, py in pixels)
    obj_max_y = max(py for px, py in pixels)
    obj_min_x = min(px for px, py in pixels)
    obj_min_y = min(py for px, py in pixels)
    obj_bbox_width = obj_max_x - obj_min_x + 1
    obj_bbox_height = obj_max_y - obj_min_y + 1

    # 最適化: 近傍範囲内のバウンディングボックスのみを事前フィルタリング
    # オブジェクトのバウンディングボックスの境界を計算
    obj_bbox_left = x
    obj_bbox_right = x + obj_bbox_width
    obj_bbox_top = y
    obj_bbox_bottom = y + obj_bbox_height

    # 拡張範囲（マージンを追加）を計算
    # マージン = オブジェクトサイズの最大値 + 一定値（例: 最大20ピクセル）
    # これにより、重なりうるオブジェクトを確実に含める
    margin = min(max(obj_bbox_width, obj_bbox_height) + 20, max(width, height) // 2)
    search_left = obj_bbox_left - margin
    search_right = obj_bbox_right + margin
    search_top = obj_bbox_top - margin
    search_bottom = obj_bbox_bottom + margin

    # 近傍範囲内のバウンディングボックスのみをフィルタリング
    nearby_bboxes = []
    for placed_x, placed_y, placed_w, placed_h in placed_bboxes:
        placed_right = placed_x + placed_w
        placed_bottom = placed_y + placed_h

        # バウンディングボックスが近傍範囲内にあるかチェック
        # 重なりうる可能性がある場合（範囲が重なる場合）のみを含める
        if (placed_right >= search_left and placed_x <= search_right and
            placed_bottom >= search_top and placed_y <= search_bottom):
            nearby_bboxes.append((placed_x, placed_y, placed_w, placed_h))

    # バウンディングボックスの重なりと最小スペースをチェック（フィルタリング済み）
    bbox_overlap_count = 0
    min_spacing_violation_count = 0
    for placed_x, placed_y, placed_w, placed_h in nearby_bboxes:
        # バウンディングボックスの重なりを計算
        overlap_x = max(0, min(x + obj_bbox_width, placed_x + placed_w) - max(x, placed_x))
        overlap_y = max(0, min(y + obj_bbox_height, placed_y + placed_h) - max(y, placed_y))
        if overlap_x > 0 and overlap_y > 0:
            bbox_overlap_count += 1
            # ピクセル単位の重なり（100.0）より低いペナルティ
            # 階層的ランダム化: 高優先度（±5%の変動）
            # 低確率で無視（1%）または軽減（2%）
            if use_randomization:
                rand_val = rng.random()
                if rand_val < 0.01:  # 1%の確率で完全無視
                    bbox_penalty = 0.0
                elif rand_val < 0.03:  # 2%の確率で軽減（半減）
                    base_penalty = 50.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    bbox_penalty = base_penalty * (1.0 + variation) * 0.5
                else:
                    base_penalty = 50.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    bbox_penalty = base_penalty * (1.0 + variation)
            else:
                bbox_penalty = 50.0
            score += bbox_penalty
        elif min_spacing > 0:
            # 最小スペースチェック（重複していない場合のみ）
            distance = _calculate_bbox_distance(
                x, y, obj_bbox_width, obj_bbox_height,
                placed_x, placed_y, placed_w, placed_h
            )
            if distance < min_spacing:
                min_spacing_violation_count += 1
                # 最小スペース違反は大きなペナルティ（重なりよりは小さいが、かなり大きい）
                # 階層的ランダム化: 高優先度（±5%の変動）
                # 低確率で無視（1%）または軽減（2%）
                if use_randomization:
                    rand_val = rng.random()
                    if rand_val < 0.01:  # 1%の確率で完全無視
                        spacing_penalty = 0.0
                    elif rand_val < 0.03:  # 2%の確率で軽減（半減）
                        base_penalty = 80.0
                        variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                        spacing_penalty = base_penalty * (1.0 + variation) * 0.5
                    else:
                        base_penalty = 80.0
                        variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                        spacing_penalty = base_penalty * (1.0 + variation)
                else:
                    spacing_penalty = 80.0
                score += spacing_penalty

    # オブジェクトの各ピクセルを評価
    inside_count = 0  # グリッド内のピクセル数
    outside_count = 0  # グリッド外のピクセル数
    overlap_count = 0  # 重なりピクセル数
    adjacent_same_color_count = 0  # 同じ色と隣接するピクセル数

    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for px, py in pixels:
        gx, gy = x + px, y + py

        # 優先順位1: グリッドサイズ内かどうか（一部はみ出してもいい）
        is_inside = (0 <= gx < width and 0 <= gy < height)
        if is_inside:
            inside_count += 1
        else:
            outside_count += 1
            # はみ出しは軽微なペナルティ（優先順位1）
            score += 1.0

        if not is_inside:
            continue

        # 優先順位2, 3: 他のオブジェクトとの重なり
        # 最適化: バウンディングボックスの重なりがある場合のみチェック
        if bbox_overlap_count > 0 and (gx, gy) in placed_pixels:
            overlap_count += 1
            # 重なりは大きなペナルティ（優先順位2, 3）
            # 階層的ランダム化: 高優先度（±5%の変動）
            # 低確率で無視（1%）または軽減（2%）
            if use_randomization:
                rand_val = rng.random()
                if rand_val < 0.01:  # 1%の確率で完全無視
                    overlap_penalty = 0.0
                elif rand_val < 0.03:  # 2%の確率で軽減（半減）
                    base_penalty = 100.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    overlap_penalty = base_penalty * (1.0 + variation) * 0.5
                else:
                    base_penalty = 100.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    overlap_penalty = base_penalty * (1.0 + variation)
            else:
                overlap_penalty = 100.0
            score += overlap_penalty
        elif bbox_overlap_count > 0 and grid_list[gy][gx] != background_color:
            # 既存の非背景色ピクセルとの重なり（バウンディングボックスの重なりがある場合のみチェック）
            overlap_count += 1
            # 階層的ランダム化: 高優先度（±5%の変動）
            # 低確率で無視（1%）または軽減（2%）
            if use_randomization:
                rand_val = rng.random()
                if rand_val < 0.01:  # 1%の確率で完全無視
                    overlap_penalty = 0.0
                elif rand_val < 0.03:  # 2%の確率で軽減（半減）
                    base_penalty = 100.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    overlap_penalty = base_penalty * (1.0 + variation) * 0.5
                else:
                    base_penalty = 100.0
                    variation = rng.uniform(-0.05, 0.05)  # ±5%の変動
                    overlap_penalty = base_penalty * (1.0 + variation)
            else:
                overlap_penalty = 100.0
            score += overlap_penalty

        # 優先順位4: 同じ色のオブジェクトと隣接（最適化版）
        # 最適化: 近傍範囲内の同じ色ピクセルのみをチェック（全件チェックを回避）
        if obj_color in placed_colors_map:
            same_color_pixels = placed_colors_map[obj_color]
            if same_color_pixels:  # 空でない場合のみ
                # 最適化: 近傍範囲（±2ピクセル）内の同じ色ピクセルのみをチェック
                # 全件チェックを回避することで計算量を大幅に削減
                nearby_same_color_pixels = set()  # SetにすることでO(1)のlookupが可能
                for same_color_px, same_color_py in same_color_pixels:
                    # 現在のピクセル（gx, gy）との距離を計算
                    dist_x = abs(same_color_px - gx)
                    dist_y = abs(same_color_py - gy)
                    # 近傍範囲内（±2ピクセル、マンハッタン距離 <= 2）のみを対象
                    if dist_x <= 2 and dist_y <= 2:
                        nearby_same_color_pixels.add((same_color_px, same_color_py))

                # 近傍範囲内のピクセルが多すぎる場合はサンプリング（最大10個）
                MAX_NEARBY_CHECK = 10
                if len(nearby_same_color_pixels) > MAX_NEARBY_CHECK:
                    nearby_list = list(nearby_same_color_pixels)
                    sample_size = min(MAX_NEARBY_CHECK, len(nearby_list))
                    if sample_size > 0 and len(nearby_list) > 0:
                        if use_randomization:
                            nearby_sampled = rng.sample(nearby_list, sample_size)
                        else:
                            nearby_sampled = random.sample(nearby_list, sample_size)
                    else:
                        nearby_sampled = []
                    nearby_same_color_pixels = set(nearby_sampled)

                # 隣接チェック（4方向のみ）
                # 最適化: 早期終了（1つの隣接が見つかったらそのピクセルのチェックを終了）
                found_adjacent = False
                for dx, dy in adjacent_offsets:
                    if found_adjacent:
                        break  # 早期終了
                    adj_x, adj_y = gx + dx, gy + dy
                    # 近傍範囲内のピクセルのみをチェック（Set lookup: O(1)）
                    if (adj_x, adj_y) in nearby_same_color_pixels:
                        adjacent_same_color_count += 1
                        found_adjacent = True  # 1つ見つかったら終了
                        # 同じ色との隣接は中程度のペナルティ（優先順位4）
                        # 階層的ランダム化: 低優先度（±50%の変動）
                        # 低確率で無視（2%）または軽減（3%）
                        if use_randomization:
                            rand_val = rng.random()
                            if rand_val < 0.02:  # 2%の確率で完全無視
                                adjacent_penalty = 0.0
                            elif rand_val < 0.05:  # 3%の確率で軽減（半減）
                                base_penalty = 10.0
                                variation = rng.uniform(-0.5, 0.5)  # ±50%の変動
                                adjacent_penalty = base_penalty * (1.0 + variation) * 0.5
                            else:
                                base_penalty = 10.0
                                variation = rng.uniform(-0.5, 0.5)  # ±50%の変動
                                adjacent_penalty = base_penalty * (1.0 + variation)
                        else:
                            adjacent_penalty = 10.0
                        score += adjacent_penalty

    # 優先順位5: グリッドサイズ内にすべての形状が収まる（ボーナス）
    # 階層的ランダム化: 中優先度（±10%の変動）
    if outside_count == 0:
        if use_randomization:
            base_bonus = -50.0
            variation = rng.uniform(-0.10, 0.10)  # ±10%の変動
            score += base_bonus * (1.0 + variation)
        else:
            score -= 50.0  # 大きなボーナス

    # バウンディングボックスの重なりがないほど良い
    # 階層的ランダム化: 中優先度（±10%の変動）
    if bbox_overlap_count == 0:
        if use_randomization:
            base_bonus = -10.0
            variation = rng.uniform(-0.10, 0.10)  # ±10%の変動
            score += base_bonus * (1.0 + variation)
        else:
            score -= 10.0  # バウンディングボックス重なりなしボーナス

    # 最小スペース制約を満たしている場合のボーナス
    # 階層的ランダム化: 中優先度（±10%の変動）
    if min_spacing > 0 and min_spacing_violation_count == 0:
        if use_randomization:
            base_bonus = -15.0
            variation = rng.uniform(-0.10, 0.10)  # ±10%の変動
            score += base_bonus * (1.0 + variation)
        else:
            score -= 15.0  # 最小スペース制約満足ボーナス

    # 重なりが少ないほど良い（優先順位2, 3を強調）
    # 階層的ランダム化: 中優先度（±10%の変動）
    if overlap_count == 0:
        if use_randomization:
            base_bonus = -20.0
            variation = rng.uniform(-0.10, 0.10)  # ±10%の変動
            score += base_bonus * (1.0 + variation)
        else:
            score -= 20.0  # 重なりなしボーナス

    # 同じ色との隣接がないほど良い（優先順位4を強調）
    # 階層的ランダム化: 低優先度（±40%の変動）
    if adjacent_same_color_count == 0:
        if use_randomization:
            base_bonus = -5.0
            variation = rng.uniform(-0.40, 0.40)  # ±40%の変動
            score += base_bonus * (1.0 + variation)
        else:
            score -= 5.0  # 隣接なしボーナス

    # 無意味なランダムボーナス: 低確率（0.3%）で大きな変動を追加
    # スコアに予測不可能な変動を加えることで多様性を確保
    if use_randomization:
        if rng.random() < 0.003:  # 0.3%の確率
            meaningless_bonus = rng.uniform(-100.0, 0.0)  # -100.0 〜 0.0の範囲
            score += meaningless_bonus

    return score


def _place_object_pixels(grid_list: List[List[int]], obj: Dict, x: int, y: int, color: int):
    """オブジェクトのピクセルをグリッドに配置"""
    pixels = obj.get('pixels', [])
    height = len(grid_list)
    width = len(grid_list[0]) if grid_list else 0

    for px, py in pixels:
        gx, gy = x + px, y + py
        if 0 <= gx < width and 0 <= gy < height:
            grid_list[gy][gx] = color
