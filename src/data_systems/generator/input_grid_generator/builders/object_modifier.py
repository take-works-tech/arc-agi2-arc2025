"""
オブジェクト変更関数

既存のオブジェクトの色、形、位置を変更する関数群
"""
from typing import List, Dict, Optional
from .core_object_builder import CoreObjectBuilder
from .shape_utils import copy_object_with_new_color


def modify_objects(
    objects: List[Dict],
    background_color: int = 0,
    change_color: bool = False,
    change_shape: bool = False,
    change_position: bool = False,
    seed: Optional[int] = None,
    **kwargs
) -> List[Dict]:
    """既存のオブジェクトの色、形、位置を変更

    Args:
        objects: 変更するオブジェクトのリスト
        background_color: 背景色
        change_color: 色を変更するか（Trueの場合、色を変更。lock_color_change=Falseの場合のみ）
        change_shape: 形を変更するか（Trueの場合、形を変更。lock_shape_change=Falseの場合のみ）
        change_position: 位置を変更するか（Trueの場合、位置を変更。lock_position_change=Falseの場合のみ）
        seed: 乱数シード
        **kwargs: その他の条件（将来の拡張用）
            - complexity: 複雑度（デフォルト: 3）
            - grid_size: グリッドサイズ（タプル形式、適応的生成に使用）
            - remaining_space: 残りスペース（適応的生成に使用）
            - num_objects_remaining: 残りオブジェクト数（適応的生成に使用）
            - connectivity_constraint: 連結性制約（None, '4-connected', '8-connected'）
                - None: デフォルト（50%で8連結、50%で4連結）
                - '4-connected': 4連結のみ（change_shape=Falseの場合は適用不可）
                - '8-connected': 50%で8連結、50%で4連結（デフォルトと同じ）
            - allow_holes: 穴ありオブジェクトを優先的に作成（90%の確率で作成、change_shape=Falseの場合は適用不可、デフォルト: False）
            - min_holes: 最小穴数（allow_holes=Trueの場合、デフォルト: 0）
            - max_holes: 最大穴数（allow_holes=Trueの場合、デフォルト: 3）
            - min_bbox_size: 最小バウンディングボックスサイズ（change_shape=Falseの場合は適用不可、幅=高さ、0の場合は制約なし、デフォルト: 0）
            - max_bbox_size: 最大バウンディングボックスサイズ（change_shape=Falseの場合は適用不可、幅=高さ、0の場合は制約なし、デフォルト: 0）
            - shape_type: 形状指定（change_shape=Falseの場合は適用不可、None, 'rectangle', 'line', 'hollow_rectangle', 'noise'）
                - None: 制約なし（既存ロジック）
                - 'rectangle': 矩形のみ
                - 'line': 線のみ
                - 'hollow_rectangle': 中空矩形のみ
                - 'noise': ノイズパターンのみ（random_pattern）
            - min_density: 最小密度（change_shape=Falseの場合は適用不可、0.0-1.0、0.0の場合は制約なし、デフォルト: 0.0）
            - max_density: 最大密度（change_shape=Falseの場合は適用不可、0.0-1.0、1.0の場合は制約なし、デフォルト: 1.0）
            - symmetry_constraint: 対称性制約（change_shape=Falseの場合は適用不可、None, 'none', 'vertical', 'horizontal', 'both'）
                - None/'none': 制約なし（デフォルト）
                - 'vertical': 垂直対称（左右対称、Y軸）
                - 'horizontal': 水平対称（上下対称、X軸）
                - 'both': 両対称（垂直と水平の両方）

    Returns:
        変更されたオブジェクトのリスト
    """
    builder = CoreObjectBuilder(seed=seed)
    rng = builder.rng
    result_objects = []

    # オブジェクト仕様を取得（将来の拡張用）
    complexity = kwargs.get('complexity', 3)
    grid_size = kwargs.get('grid_size')
    remaining_space = kwargs.get('remaining_space')
    num_objects_remaining = kwargs.get('num_objects_remaining')
    connectivity_constraint = kwargs.get('connectivity_constraint')
    allow_holes = kwargs.get('allow_holes', False)
    min_holes = kwargs.get('min_holes', 0)
    max_holes = kwargs.get('max_holes', 3)
    min_bbox_size = kwargs.get('min_bbox_size', 0)
    max_bbox_size = kwargs.get('max_bbox_size', 0)
    shape_type = kwargs.get('shape_type')
    min_density = kwargs.get('min_density', 0.0)
    max_density = kwargs.get('max_density', 1.0)
    symmetry_constraint = kwargs.get('symmetry_constraint')

    # 連結性を決定（connectivity_constraintに基づく）
    # 注意: change_shape=Falseの場合は連結性制約は適用しない
    def _determine_connectivity(builder_rng, constraint):
        """連結性を決定"""
        if constraint == '4-connected':
            return 4
        elif constraint == '8-connected':
            # 50%の確率で8連結
            return 8 if builder_rng.random() < 0.5 else 4
        else:  # None: デフォルト（50%で8連結、50%で4連結）
            return 8 if builder_rng.random() < 0.5 else 4

    # バウンディングボックスサイズを決定
    def _determine_bbox_size(builder_rng, min_size, max_size, default_min=2, default_max=6):
        """バウンディングボックスサイズを決定（幅=高さ）"""
        if min_size > 0 or max_size > 0:
            if min_size > 0 and max_size > 0:
                return builder_rng.randint(min_size, max_size)
            elif min_size > 0:
                return builder_rng.randint(min_size, default_max)
            elif max_size > 0:
                return builder_rng.randint(default_min, max_size)
        return None  # 制約なし

    # 密度を決定
    def _determine_density(builder_rng, min_dens, max_dens):
        """密度を決定"""
        if min_dens > 0.0 or max_dens < 1.0:
            if min_dens > 0.0 and max_dens < 1.0:
                return min_dens + builder_rng.random() * (max_dens - min_dens)
            elif min_dens > 0.0:
                return min_dens + builder_rng.random() * (1.0 - min_dens)
            elif max_dens < 1.0:
                return builder_rng.random() * max_dens
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

    # 形状生成ヘルパー関数（重複コードを削減）
    def _generate_shape_for_complexity(builder, color, complexity, grid_size=None, connectivity=None, max_size=None):
        """複雑度に応じた形状生成（ARC統計生成も選択肢に含める）"""
        if complexity <= 3:
            # 30%の確率でARC統計生成を使用
            if builder.rng.random() < 0.3:
                return builder.generate_object_by_arc_stats(color, grid_size, max_size=max_size)
            else:
                return builder._generate_simple_object(color, max_size=max_size)
        elif complexity <= 6:
            rand = builder.rng.random()
            if rand < 0.3:
                return builder.generate_object_by_arc_stats(color, grid_size, max_size=max_size)
            elif rand < 0.7:
                return builder.generate_synthesized_object(color, max_size=max_size)
            else:
                return builder._generate_simple_object(color, max_size=max_size)
        else:
            rand = builder.rng.random()
            if rand < 0.2:
                return builder.generate_object_by_arc_stats(color, grid_size, max_size=max_size)
            elif rand < 0.6:
                return builder.generate_synthesized_object(color, max_size=max_size)
            elif rand < 0.8:
                return builder.generate_composite_object(color, grid_size, max_size=max_size)
            else:
                return builder._generate_simple_object(color, max_size=max_size)

    for obj in objects:
        # 変更禁止フラグをチェック
        # オブジェクトが変更禁止フラグを持っている場合、引数でTrueが指定されていても変更しない
        lock_color_change = obj.get('lock_color_change', False)
        lock_shape_change = obj.get('lock_shape_change', False)
        lock_position_change = obj.get('lock_position_change', False)

        # 変更する属性を決定
        # 変更禁止フラグがTrueの場合、引数がTrueでも変更しない（Falseになる）
        should_change_color = change_color and not lock_color_change
        should_change_shape = change_shape and not lock_shape_change
        should_change_position = change_position and not lock_position_change

        # 何も変更しない場合は元のオブジェクトをそのまま返す
        if not should_change_color and not should_change_shape and not should_change_position:
            result_objects.append(obj)
            continue

        # 元のオブジェクトの属性を取得（固定する属性）
        original_color = obj.get('color', 1)
        original_x = obj.get('x')
        original_y = obj.get('y')

        new_obj = None

        # 色のみ変更する場合：形と位置を固定
        if should_change_color and not should_change_shape and not should_change_position:
            # 新しい色を決定（背景色以外の色からランダム選択）
            available_colors = [c for c in range(10) if c != background_color and c != original_color]
            if available_colors:
                new_color = rng.choice(available_colors)
            else:
                new_color = 1 if background_color == 0 else 0

            # 形と位置を固定して色のみ変更
            new_obj = copy_object_with_new_color(obj, new_color)
            # x, y座標を保持
            if original_x is not None:
                new_obj['x'] = original_x
            if original_y is not None:
                new_obj['y'] = original_y
            if original_x is not None and original_y is not None:
                new_obj['position'] = (original_x, original_y)

        # 形のみ変更する場合：色と位置を固定
        elif should_change_shape and not should_change_color and not should_change_position:
            # 新しい形を生成（元の色を使用）
            # 穴ありオブジェクトの場合（優先度：最高、90%の確率で確実に作成）
            if allow_holes:
                if rng.random() < 0.9:  # 90%の確率で穴ありオブジェクト
                    hole_count = rng.randint(min_holes, max_holes)
                    new_obj = builder._generate_object_with_holes(original_color, hole_count)
            # 形状制約がある場合
            if new_obj is None and shape_type:
                if shape_type == 'rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 2, 6)
                    size = bbox_size if bbox_size else rng.randint(2, 6)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(original_color, size, size, filled=True)
                    else:
                        # 長方形（幅と高さを異なる値にする）
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(original_color, width, height, filled=True)
                elif shape_type == 'line':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    length = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 水平、垂直、斜め45度のすべてを生成可能
                    if rng.random() < 0.6:
                        direction = rng.choice(['horizontal', 'vertical'])
                        new_obj = builder._generate_line(original_color, length, direction)
                    else:
                        # 斜め線を生成
                        diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        new_obj = builder._generate_diagonal_45(original_color, length, diagonal_direction)
                elif shape_type == 'hollow_rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    size = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(original_color, size, size, filled=False)
                    else:
                        # 長方形の中空矩形
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(original_color, width, height, filled=False)
                elif shape_type == 'noise':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                    density = _determine_density(rng, min_density, max_density)
                    # ノイズパターンの色選択（既存オブジェクトから色を取得）
                    existing_colors_in_objects = {obj.get('color') for obj in objects if isinstance(obj, dict) and 'color' in obj}
                    noise_color = _determine_noise_color(rng, background_color, existing_colors_in_objects, original_color)
                    # ノイズパターンでは連結性制約を適用しない
                    new_obj = builder._generate_random_pattern(noise_color, width, height, density, None)
            # 対称性制約がある場合（形状制約がない、または形状制約が'any'の場合）
            if new_obj is None and symmetry_constraint and symmetry_constraint != 'none':
                # バウンディングボックスサイズ制約がある場合は使用
                if min_bbox_size > 0 or max_bbox_size > 0:
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                else:
                    width = None
                    height = None
                new_obj = builder._generate_symmetric_object(
                    original_color,
                    base_shape=None,
                    width=width,
                    height=height,
                    symmetry_type=symmetry_constraint
                )
            # 適応的生成が可能な場合
            if new_obj is None and grid_size and remaining_space is not None and num_objects_remaining is not None:
                new_obj = builder.generate_object_adaptive(
                    original_color, grid_size, remaining_space, num_objects_remaining
                )
            elif new_obj is None:
                max_size_for_shape = max_bbox_size if max_bbox_size > 0 else None
                new_obj = _generate_shape_for_complexity(builder, original_color, complexity, grid_size, max_size=max_size_for_shape)

            # 生成後の制約チェック
            if new_obj:
                # 対称性制約の適用（形状制約で生成されなかった場合）
                if new_obj and symmetry_constraint and symmetry_constraint != 'none':
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        # 既に対称性が適用されているかチェック（shape_typeに'symmetric'が含まれているか）
                        shape_type = new_obj.get('shape_type', '')
                        if 'symmetric' not in shape_type:
                            # 対称性を適用
                            symmetric_pixels = builder._make_symmetric(pixels, width, height, symmetry_constraint)
                            # バウンディングボックスを再計算
                            if symmetric_pixels:
                                min_x = min(x for x, y in symmetric_pixels)
                                max_x = max(x for x, y in symmetric_pixels)
                                min_y = min(y for x, y in symmetric_pixels)
                                max_y = max(y for x, y in symmetric_pixels)
                                new_obj['width'] = max_x - min_x + 1
                                new_obj['height'] = max_y - min_y + 1
                                new_obj['pixels'] = [(x - min_x, y - min_y) for x, y in symmetric_pixels]
                                new_obj['area'] = len(new_obj['pixels'])
                                new_obj['shape_type'] = f'symmetric_{symmetry_constraint}'
                # 連結性制約の適用（ノイズパターン以外、形状変更時のみ）
                if shape_type != 'noise' and connectivity_constraint is not None:
                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        new_obj['pixels'] = builder._ensure_connected(pixels, width, height, connectivity)

                # バウンディングボックスサイズ制約のチェック
                bbox_size_valid = True
                if min_bbox_size > 0 or max_bbox_size > 0:
                    obj_width = new_obj.get('width', 0)
                    obj_height = new_obj.get('height', 0)
                    obj_size = max(obj_width, obj_height)  # サイズ判定用に最大寸法を使用

                    if min_bbox_size > 0 and obj_size < min_bbox_size:
                        bbox_size_valid = False
                    if max_bbox_size > 0 and obj_size > max_bbox_size:
                        bbox_size_valid = False

                # 密度制約のチェック（ノイズパターン以外）
                density_valid = True
                if 'density' not in new_obj and shape_type != 'noise' and (min_density > 0.0 or max_density < 1.0):
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if width > 0 and height > 0:
                        actual_density = len(pixels) / (width * height)
                        if (min_density > 0.0 and actual_density < min_density) or \
                           (max_density < 1.0 and actual_density > max_density):
                            density_valid = False

                # すべての制約を満たしていない場合は再生成（最大試行回数: 10回）
                if not (bbox_size_valid and density_valid):
                    # 制約を満たさない場合でも受け入れる（呼び出し側で再生成を制御）
                    # 将来的には、ここで再生成ループを追加することも可能
                    pass

            if new_obj:
                # x, y座標を保持
                if original_x is not None:
                    new_obj['x'] = original_x
                if original_y is not None:
                    new_obj['y'] = original_y
                if original_x is not None and original_y is not None:
                    new_obj['position'] = (original_x, original_y)

        # 位置のみ変更する場合：色と形を固定
        elif should_change_position and not should_change_color and not should_change_shape:
            # 新しい位置を決定（ランダム、またはスコア形式）
            # 位置は呼び出し側の配置ロジックで設定されるため、ここでは位置情報をクリア
            new_obj = obj.copy()
            # pixelsもディープコピー
            if 'pixels' in new_obj and isinstance(new_obj['pixels'], list):
                new_obj['pixels'] = [pixel for pixel in new_obj['pixels']]
            # x, yをNoneに設定（後で配置ロジックで設定される）
            new_obj['x'] = None
            new_obj['y'] = None
            if 'position' in new_obj:
                del new_obj['position']

        # 色と形を変更する場合：位置を固定
        elif should_change_color and should_change_shape and not should_change_position:
            # 新しい色を決定
            available_colors = [c for c in range(10) if c != background_color and c != original_color]
            if available_colors:
                new_color = rng.choice(available_colors)
            else:
                new_color = 1 if background_color == 0 else 0

            # 新しい形を生成（新しい色を使用）
            # 穴ありオブジェクトの場合（優先度：最高、90%の確率で確実に作成）
            if allow_holes:
                if rng.random() < 0.9:  # 90%の確率で穴ありオブジェクト
                    hole_count = rng.randint(min_holes, max_holes)
                    new_obj = builder._generate_object_with_holes(new_color, hole_count)
            # 形状制約がある場合
            if new_obj is None and shape_type:
                if shape_type == 'rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 2, 6)
                    size = bbox_size if bbox_size else rng.randint(2, 6)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(new_color, size, size, filled=True)
                    else:
                        # 長方形（幅と高さを異なる値にする）
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(new_color, width, height, filled=True)
                elif shape_type == 'line':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    length = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 水平、垂直、斜め45度のすべてを生成可能
                    if rng.random() < 0.6:
                        direction = rng.choice(['horizontal', 'vertical'])
                        new_obj = builder._generate_line(new_color, length, direction)
                    else:
                        # 斜め線を生成
                        diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        new_obj = builder._generate_diagonal_45(new_color, length, diagonal_direction)
                elif shape_type == 'hollow_rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    size = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(new_color, size, size, filled=False)
                    else:
                        # 長方形の中空矩形
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(new_color, width, height, filled=False)
                elif shape_type == 'noise':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                    density = _determine_density(rng, min_density, max_density)
                    # ノイズパターンの色選択（既存オブジェクトから色を取得）
                    existing_colors_in_objects = {obj.get('color') for obj in objects if isinstance(obj, dict) and 'color' in obj}
                    noise_color = _determine_noise_color(rng, background_color, existing_colors_in_objects, new_color)
                    # ノイズパターンでは連結性制約を適用しない
                    new_obj = builder._generate_random_pattern(noise_color, width, height, density, None)
            # 対称性制約がある場合（形状制約がない、または形状制約が'any'の場合）
            if new_obj is None and symmetry_constraint and symmetry_constraint != 'none':
                # バウンディングボックスサイズ制約がある場合は使用
                if min_bbox_size > 0 or max_bbox_size > 0:
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                else:
                    width = None
                    height = None
                new_obj = builder._generate_symmetric_object(
                    new_color,
                    base_shape=None,
                    width=width,
                    height=height,
                    symmetry_type=symmetry_constraint
                )
            # 適応的生成が可能な場合
            if new_obj is None and grid_size and remaining_space is not None and num_objects_remaining is not None:
                new_obj = builder.generate_object_adaptive(
                    new_color, grid_size, remaining_space, num_objects_remaining
                )
            elif new_obj is None:
                max_size_for_shape = max_bbox_size if max_bbox_size > 0 else None
                new_obj = _generate_shape_for_complexity(builder, new_color, complexity, grid_size, max_size=max_size_for_shape)

            # 生成後の制約チェック
            if new_obj:
                # 対称性制約の適用（形状制約で生成されなかった場合）
                if new_obj and symmetry_constraint and symmetry_constraint != 'none':
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        # 既に対称性が適用されているかチェック（shape_typeに'symmetric'が含まれているか）
                        shape_type = new_obj.get('shape_type', '')
                        if 'symmetric' not in shape_type:
                            # 対称性を適用
                            symmetric_pixels = builder._make_symmetric(pixels, width, height, symmetry_constraint)
                            # バウンディングボックスを再計算
                            if symmetric_pixels:
                                min_x = min(x for x, y in symmetric_pixels)
                                max_x = max(x for x, y in symmetric_pixels)
                                min_y = min(y for x, y in symmetric_pixels)
                                max_y = max(y for x, y in symmetric_pixels)
                                new_obj['width'] = max_x - min_x + 1
                                new_obj['height'] = max_y - min_y + 1
                                new_obj['pixels'] = [(x - min_x, y - min_y) for x, y in symmetric_pixels]
                                new_obj['area'] = len(new_obj['pixels'])
                                new_obj['shape_type'] = f'symmetric_{symmetry_constraint}'
                # 連結性制約の適用（ノイズパターン以外、形状変更時のみ）
                if shape_type != 'noise' and connectivity_constraint is not None:
                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        new_obj['pixels'] = builder._ensure_connected(pixels, width, height, connectivity)

                # バウンディングボックスサイズ制約のチェック
                bbox_size_valid = True
                if min_bbox_size > 0 or max_bbox_size > 0:
                    obj_width = new_obj.get('width', 0)
                    obj_height = new_obj.get('height', 0)
                    obj_size = max(obj_width, obj_height)  # サイズ判定用に最大寸法を使用

                    if min_bbox_size > 0 and obj_size < min_bbox_size:
                        bbox_size_valid = False
                    if max_bbox_size > 0 and obj_size > max_bbox_size:
                        bbox_size_valid = False

                # 密度制約のチェック（ノイズパターン以外）
                density_valid = True
                if 'density' not in new_obj and shape_type != 'noise' and (min_density > 0.0 or max_density < 1.0):
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if width > 0 and height > 0:
                        actual_density = len(pixels) / (width * height)
                        if (min_density > 0.0 and actual_density < min_density) or \
                           (max_density < 1.0 and actual_density > max_density):
                            density_valid = False

                # すべての制約を満たしていない場合は再生成（最大試行回数: 10回）
                if not (bbox_size_valid and density_valid):
                    # 制約を満たさない場合でも受け入れる（呼び出し側で再生成を制御）
                    # 将来的には、ここで再生成ループを追加することも可能
                    pass

            if new_obj:
                # x, y座標を保持
                if original_x is not None:
                    new_obj['x'] = original_x
                if original_y is not None:
                    new_obj['y'] = original_y
                if original_x is not None and original_y is not None:
                    new_obj['position'] = (original_x, original_y)

        # 色と位置を変更する場合：形を固定
        elif should_change_color and should_change_position and not should_change_shape:
            # 新しい色を決定
            available_colors = [c for c in range(10) if c != background_color and c != original_color]
            if available_colors:
                new_color = rng.choice(available_colors)
            else:
                new_color = 1 if background_color == 0 else 0

            # 形を固定して色のみ変更
            new_obj = copy_object_with_new_color(obj, new_color)
            # x, yをNoneに設定（後で配置ロジックで設定される）
            new_obj['x'] = None
            new_obj['y'] = None
            if 'position' in new_obj:
                del new_obj['position']

        # 形と位置を変更する場合：色を固定
        elif should_change_shape and should_change_position and not should_change_color:
            # 新しい形を生成（元の色を使用）
            # 穴ありオブジェクトの場合（優先度：最高、90%の確率で確実に作成）
            if allow_holes:
                if rng.random() < 0.9:  # 90%の確率で穴ありオブジェクト
                    hole_count = rng.randint(min_holes, max_holes)
                    new_obj = builder._generate_object_with_holes(original_color, hole_count)
            # 形状制約がある場合
            if new_obj is None and shape_type:
                if shape_type == 'rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 2, 6)
                    size = bbox_size if bbox_size else rng.randint(2, 6)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(original_color, size, size, filled=True)
                    else:
                        # 長方形（幅と高さを異なる値にする）
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(original_color, width, height, filled=True)
                elif shape_type == 'line':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    length = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 水平、垂直、斜め45度のすべてを生成可能
                    if rng.random() < 0.6:
                        direction = rng.choice(['horizontal', 'vertical'])
                        new_obj = builder._generate_line(original_color, length, direction)
                    else:
                        # 斜め線を生成
                        diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        new_obj = builder._generate_diagonal_45(original_color, length, diagonal_direction)
                elif shape_type == 'hollow_rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    size = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(original_color, size, size, filled=False)
                    else:
                        # 長方形の中空矩形
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(original_color, width, height, filled=False)
                elif shape_type == 'noise':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                    density = _determine_density(rng, min_density, max_density)
                    # ノイズパターンの色選択（既存オブジェクトから色を取得）
                    existing_colors_in_objects = {obj.get('color') for obj in objects if isinstance(obj, dict) and 'color' in obj}
                    noise_color = _determine_noise_color(rng, background_color, existing_colors_in_objects, original_color)
                    # ノイズパターンでは連結性制約を適用しない
                    new_obj = builder._generate_random_pattern(noise_color, width, height, density, None)
            # 対称性制約がある場合（形状制約がない、または形状制約が'any'の場合）
            if new_obj is None and symmetry_constraint and symmetry_constraint != 'none':
                # バウンディングボックスサイズ制約がある場合は使用
                if min_bbox_size > 0 or max_bbox_size > 0:
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                else:
                    width = None
                    height = None
                new_obj = builder._generate_symmetric_object(
                    original_color,
                    base_shape=None,
                    width=width,
                    height=height,
                    symmetry_type=symmetry_constraint
                )
            # 適応的生成が可能な場合
            if new_obj is None and grid_size and remaining_space is not None and num_objects_remaining is not None:
                new_obj = builder.generate_object_adaptive(
                    original_color, grid_size, remaining_space, num_objects_remaining
                )
            elif new_obj is None:
                max_size_for_shape = max_bbox_size if max_bbox_size > 0 else None
                new_obj = _generate_shape_for_complexity(builder, original_color, complexity, grid_size, max_size=max_size_for_shape)

            # 生成後の制約チェック
            if new_obj:
                # 連結性制約の適用（ノイズパターン以外、形状変更時のみ）
                if shape_type != 'noise' and connectivity_constraint is not None:
                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        new_obj['pixels'] = builder._ensure_connected(pixels, width, height, connectivity)

                # バウンディングボックスサイズ制約のチェック
                bbox_size_valid = True
                if min_bbox_size > 0 or max_bbox_size > 0:
                    obj_width = new_obj.get('width', 0)
                    obj_height = new_obj.get('height', 0)
                    obj_size = max(obj_width, obj_height)  # サイズ判定用に最大寸法を使用

                    if min_bbox_size > 0 and obj_size < min_bbox_size:
                        bbox_size_valid = False
                    if max_bbox_size > 0 and obj_size > max_bbox_size:
                        bbox_size_valid = False

                # 密度制約のチェック（ノイズパターン以外）
                density_valid = True
                if 'density' not in new_obj and shape_type != 'noise' and (min_density > 0.0 or max_density < 1.0):
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if width > 0 and height > 0:
                        actual_density = len(pixels) / (width * height)
                        if (min_density > 0.0 and actual_density < min_density) or \
                           (max_density < 1.0 and actual_density > max_density):
                            density_valid = False

                # すべての制約を満たしていない場合は再生成（最大試行回数: 10回）
                if not (bbox_size_valid and density_valid):
                    # 制約を満たさない場合でも受け入れる（呼び出し側で再生成を制御）
                    # 将来的には、ここで再生成ループを追加することも可能
                    pass

            if new_obj:
                # x, yをNoneに設定（後で配置ロジックで設定される）
                new_obj['x'] = None
                new_obj['y'] = None
                if 'position' in new_obj:
                    del new_obj['position']

        # すべて変更する場合
        elif should_change_color and should_change_shape and should_change_position:
            # 新しい色を決定
            available_colors = [c for c in range(10) if c != background_color and c != original_color]
            if available_colors:
                new_color = rng.choice(available_colors)
            else:
                new_color = 1 if background_color == 0 else 0

            # 新しい形を生成（新しい色を使用）
            # 穴ありオブジェクトの場合（優先度：最高、90%の確率で確実に作成）
            if allow_holes:
                if rng.random() < 0.9:  # 90%の確率で穴ありオブジェクト
                    hole_count = rng.randint(min_holes, max_holes)
                    new_obj = builder._generate_object_with_holes(new_color, hole_count)
            # 形状制約がある場合
            if new_obj is None and shape_type:
                if shape_type == 'rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 2, 6)
                    size = bbox_size if bbox_size else rng.randint(2, 6)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(new_color, size, size, filled=True)
                    else:
                        # 長方形（幅と高さを異なる値にする）
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(new_color, width, height, filled=True)
                elif shape_type == 'line':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    length = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 水平、垂直、斜め45度のすべてを生成可能
                    if rng.random() < 0.6:
                        direction = rng.choice(['horizontal', 'vertical'])
                        new_obj = builder._generate_line(new_color, length, direction)
                    else:
                        # 斜め線を生成
                        diagonal_direction = rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                        new_obj = builder._generate_diagonal_45(new_color, length, diagonal_direction)
                elif shape_type == 'hollow_rectangle':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    size = bbox_size if bbox_size else rng.randint(3, 8)
                    # バリエーション: 正方形と長方形の両方を生成
                    if rng.random() < 0.6:
                        new_obj = builder._generate_rectangle(new_color, size, size, filled=False)
                    else:
                        # 長方形の中空矩形
                        width = size
                        height = rng.randint(max(2, size // 2), min(size * 2, size + 3))
                        new_obj = builder._generate_rectangle(new_color, width, height, filled=False)
                elif shape_type == 'noise':
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                    density = _determine_density(rng, min_density, max_density)
                    # ノイズパターンの色選択（既存オブジェクトから色を取得）
                    existing_colors_in_objects = {obj.get('color') for obj in objects if isinstance(obj, dict) and 'color' in obj}
                    noise_color = _determine_noise_color(rng, background_color, existing_colors_in_objects, new_color)
                    # ノイズパターンでは連結性制約を適用しない
                    new_obj = builder._generate_random_pattern(noise_color, width, height, density, None)
            # 対称性制約がある場合（形状制約がない、または形状制約が'any'の場合）
            if new_obj is None and symmetry_constraint and symmetry_constraint != 'none':
                # バウンディングボックスサイズ制約がある場合は使用
                if min_bbox_size > 0 or max_bbox_size > 0:
                    bbox_size = _determine_bbox_size(rng, min_bbox_size, max_bbox_size, 3, 8)
                    width = bbox_size if bbox_size else None
                    height = bbox_size if bbox_size else None
                else:
                    width = None
                    height = None
                new_obj = builder._generate_symmetric_object(
                    new_color,
                    base_shape=None,
                    width=width,
                    height=height,
                    symmetry_type=symmetry_constraint
                )
            # 適応的生成が可能な場合
            if new_obj is None and grid_size and remaining_space is not None and num_objects_remaining is not None:
                new_obj = builder.generate_object_adaptive(
                    new_color, grid_size, remaining_space, num_objects_remaining
                )
            elif new_obj is None:
                max_size_for_shape = max_bbox_size if max_bbox_size > 0 else None
                new_obj = _generate_shape_for_complexity(builder, new_color, complexity, grid_size, max_size=max_size_for_shape)

            # 生成後の制約チェック
            if new_obj:
                # 対称性制約の適用（形状制約で生成されなかった場合）
                if new_obj and symmetry_constraint and symmetry_constraint != 'none':
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        # 既に対称性が適用されているかチェック（shape_typeに'symmetric'が含まれているか）
                        shape_type = new_obj.get('shape_type', '')
                        if 'symmetric' not in shape_type:
                            # 対称性を適用
                            symmetric_pixels = builder._make_symmetric(pixels, width, height, symmetry_constraint)
                            # バウンディングボックスを再計算
                            if symmetric_pixels:
                                min_x = min(x for x, y in symmetric_pixels)
                                max_x = max(x for x, y in symmetric_pixels)
                                min_y = min(y for x, y in symmetric_pixels)
                                max_y = max(y for x, y in symmetric_pixels)
                                new_obj['width'] = max_x - min_x + 1
                                new_obj['height'] = max_y - min_y + 1
                                new_obj['pixels'] = [(x - min_x, y - min_y) for x, y in symmetric_pixels]
                                new_obj['area'] = len(new_obj['pixels'])
                                new_obj['shape_type'] = f'symmetric_{symmetry_constraint}'
                # 連結性制約の適用（ノイズパターン以外、形状変更時のみ）
                if shape_type != 'noise' and connectivity_constraint is not None:
                    connectivity = _determine_connectivity(builder.rng, connectivity_constraint)
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if pixels and width > 0 and height > 0:
                        new_obj['pixels'] = builder._ensure_connected(pixels, width, height, connectivity)

                # バウンディングボックスサイズ制約のチェック
                bbox_size_valid = True
                if min_bbox_size > 0 or max_bbox_size > 0:
                    obj_width = new_obj.get('width', 0)
                    obj_height = new_obj.get('height', 0)
                    obj_size = max(obj_width, obj_height)  # サイズ判定用に最大寸法を使用

                    if min_bbox_size > 0 and obj_size < min_bbox_size:
                        bbox_size_valid = False
                    if max_bbox_size > 0 and obj_size > max_bbox_size:
                        bbox_size_valid = False

                # 密度制約のチェック（ノイズパターン以外）
                density_valid = True
                if 'density' not in new_obj and shape_type != 'noise' and (min_density > 0.0 or max_density < 1.0):
                    pixels = new_obj.get('pixels', [])
                    width = new_obj.get('width', 1)
                    height = new_obj.get('height', 1)
                    if width > 0 and height > 0:
                        actual_density = len(pixels) / (width * height)
                        if (min_density > 0.0 and actual_density < min_density) or \
                           (max_density < 1.0 and actual_density > max_density):
                            density_valid = False

                # すべての制約を満たしていない場合は再生成（最大試行回数: 10回）
                if not (bbox_size_valid and density_valid):
                    # 制約を満たさない場合でも受け入れる（呼び出し側で再生成を制御）
                    # 将来的には、ここで再生成ループを追加することも可能
                    pass

            if new_obj:
                # x, yをNoneに設定（後で配置ロジックで設定される）
                new_obj['x'] = None
                new_obj['y'] = None
                if 'position' in new_obj:
                    del new_obj['position']

        # 新しいオブジェクトが生成された場合は追加、そうでない場合は元のオブジェクトを追加
        if new_obj:
            # 変更禁止フラグを引き継ぐ
            new_obj['lock_color_change'] = obj.get('lock_color_change', False)
            new_obj['lock_shape_change'] = obj.get('lock_shape_change', False)
            new_obj['lock_position_change'] = obj.get('lock_position_change', False)
            result_objects.append(new_obj)
        else:
            result_objects.append(obj)

    return result_objects
