"""
部分プログラム生成モジュール

カテゴリ分けの結果に基づいて部分プログラムを生成
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from itertools import combinations

from .data_structures import CategoryInfo, BackgroundColorInfo, ObjectInfo
from .config import ObjectMatchingConfig
from .similarity_utils import calculate_symmetry_score
from src.data_systems.generator.program_generator.metadata.variable_naming import variable_naming_system
from src.data_systems.generator.program_generator.metadata.types import TypeInfo, ReturnType, SemanticType


class PartialProgramGenerator:
    """部分プログラム生成器"""

    def __init__(self, config: Optional[ObjectMatchingConfig] = None):
        """
        初期化

        Args:
            config: オブジェクトマッチング設定
        """
        self.config = config or ObjectMatchingConfig()
        # デバッグ情報を保存（部分プログラム生成失敗の原因分析用）
        self.last_failure_debug_info = []

    def generate_partial_program(
        self, connectivity: int, bg_strategy: Dict[str, Any],
        grid_sizes: List[Tuple[int, int]], valid_categories: List[CategoryInfo],
        pattern_idx: int = 0
    ) -> Tuple[str, Dict[str, str], Optional[str]]:
        """
        部分プログラムを生成

        Args:
            connectivity: 連結性（4または8）
            bg_strategy: 背景色戦略
            grid_sizes: グリッドサイズのリスト
            valid_categories: 有効なカテゴリのリスト
            pattern_idx: パターンインデックス（デバッグ用）

        Returns:
            (部分プログラム文字列, カテゴリ変数マッピング, リトライフラグ)
        """
        # 初期化
        partial_program = f"objects = GET_ALL_OBJECTS({connectivity})"
        category_var_mapping = {}

        # 背景色フィルタリング（設定で有効化されている場合のみ追加）
        if self.config.enable_background_filtering_in_partial_program:
            if bg_strategy['type'] == 'unified':
                bg_color = bg_strategy.get('color', 0)
                partial_program += f"\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), {bg_color}))"
            elif bg_strategy['type'] == 'per_grid':
                partial_program += "\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"

        # カテゴリ数が0または1の場合、背景色フィルタリングまでの部分プログラムを返す
        num_categories = len(valid_categories)
        if num_categories == 0:
            # カテゴリが0個の場合（オブジェクトが0個、再試行が上限に達した場合など）、
            # 背景色フィルタリングまでの部分プログラムを返す
            category_var_mapping = {}
            return partial_program, category_var_mapping, None

        if num_categories == 1:
            # カテゴリが1つだけの場合、FILTERによる分類に失敗したとみなす
            # objects1 = objects を追加せず、背景色フィルタリングまでの部分プログラムを返す
            category_var_mapping = {}
            return partial_program, category_var_mapping, None

        # すべてのオブジェクトを取得
        all_objects = []
        for category in valid_categories:
            all_objects.extend(category.objects)

        # 変数名システムを初期化
        used_vars = set()
        object_array_type = TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True)

        # デバッグ情報を保存（部分プログラム生成失敗の原因分析用）
        self.last_failure_debug_info = []
        failure_debug_info = []

        # ループ1: 有効なカテゴリ数でループ（各カテゴリに対して）
        last_category_failed = False
        for category_id, category in enumerate(valid_categories):
            is_last_category = (category_id == len(valid_categories) - 1)

            # カテゴリ固有の条件を生成
            combined_condition, success, condition_debug_info = self._generate_category_unique_condition(
                category, valid_categories, category_id, all_objects,
                max_parameter_combinations=2
            )

            # 失敗した場合、デバッグ情報を記録
            if not success and condition_debug_info:
                failure_debug_info.append({
                    'category_id': category_id,
                    'target_category_id': category.category_id if hasattr(category, 'category_id') else category_id,
                    'total_objects': len(category.objects),
                    'object_count_per_grid': category.object_count_per_grid if hasattr(category, 'object_count_per_grid') else [],
                    'debug_info': condition_debug_info
                })

            if not success:
                # 最後のカテゴリのFILTER条件生成に失敗した場合、EXCLUDEでフォールバック
                if is_last_category and len(valid_categories) > 1:
                    last_category_failed = True
                    # 最後のカテゴリの変数名を生成
                    preferred_name = f"objects{category_id + 1}"
                    category_var_name = preferred_name
                    used_vars.add(category_var_name)
                    category_var_mapping[category.category_id] = category_var_name
                    # EXCLUDEで最後のカテゴリを変数に代入する処理は後で追加
                    break
                else:
                    # 最後のカテゴリ以外、またはカテゴリが1個のみの場合は失敗とする
                    self.last_failure_debug_info = failure_debug_info
                    return None, {}, 'RETRY_CATEGORY'

            # 4. 成功した場合: FILTER条件を部分プログラムに追加
            # 変数名を生成
            preferred_name = f"objects{category_id + 1}"
            # preferred_nameが使用されていない場合、そのまま使用
            # preferred_nameは常に優先して使用（used_varsに含まれていても使用）
            category_var_name = preferred_name
            used_vars.add(category_var_name)
            category_var_mapping[category.category_id] = category_var_name
            partial_program += f"\n{category_var_name} = FILTER(objects, {combined_condition})"

            # FILTER条件で使用された特徴量の範囲をコメントとして追加
            used_parameter_info = self._extract_used_params_from_condition(combined_condition, category)
            if used_parameter_info:
                for param_name, param_data in used_parameter_info.items():
                    min_val = param_data.get('min_val')
                    max_val = param_data.get('max_val')
                    if min_val is not None and max_val is not None:
                        # パラメータ名に応じたコメントを追加
                        if param_name == 'size':
                            partial_program += f"\n# {category_var_name} size range: {min_val} - {max_val}"
                        elif param_name == 'width':
                            partial_program += f"\n# {category_var_name} width range: {min_val} - {max_val}"
                        elif param_name == 'height':
                            partial_program += f"\n# {category_var_name} height range: {min_val} - {max_val}"
                        elif param_name == 'hole_count':
                            partial_program += f"\n# {category_var_name} hole_count range: {min_val} - {max_val}"
                        elif param_name == 'x':
                            partial_program += f"\n# {category_var_name} X range: {min_val} - {max_val}"
                        elif param_name == 'y':
                            partial_program += f"\n# {category_var_name} Y range: {min_val} - {max_val}"
                    elif param_name == 'color':
                        # 色の場合は範囲ではなく色のリストを表示
                        target_colors = param_data.get('colors', [])
                        if target_colors:
                            partial_program += f"\n# {category_var_name} colors: {target_colors}"

            # 消失パターンが高いカテゴリの場合、特別な処理を追加
            if hasattr(category, 'disappearance_ratio') and category.disappearance_ratio > 0.5:
                if category.disappearance_ratio > 0.7:
                    # 消失パターンが非常に高い場合: 条件付きで除外
                    # 訓練ペアに存在する場合のみ使用、存在しない場合は空配列を返す
                    partial_program += f"\n# {category_var_name} has very high disappearance ratio: {category.disappearance_ratio:.2f}"
                    # 条件付き処理: 訓練ペアに存在する場合のみ使用
                    # 注: 実際の実装では、IF文や条件分岐を使用する必要があるが、
                    # 現在のDSLでは直接サポートされていないため、コメントとして残す
                    # 将来的には、条件付き処理をサポートするDSL拡張が必要
                    partial_program += f"\n# Consider excluding {category_var_name} if not present in training pairs"
                    # 代替案: 消失パターンが高いカテゴリは、部分プログラムから除外する
                    # ただし、これはカテゴリ分類の結果に依存するため、ここではコメントのみ
                else:
                    partial_program += f"\n# {category_var_name} has high disappearance ratio: {category.disappearance_ratio:.2f}"
                    # 消失パターンが中程度の場合: 警告のみ（処理は通常通り）

        # 最後のカテゴリのFILTER条件生成に失敗した場合、EXCLUDEでフォールバック
        if last_category_failed and len(valid_categories) > 1:
            last_category_id = valid_categories[-1].category_id
            last_var_name = category_var_mapping.get(last_category_id)

            if last_var_name:
                # 最後のカテゴリ以外の変数名をEXCLUDEで削除して、最後のカテゴリを取得
                # objectsを上書きせず、中間変数を使用する
                # objectsは既に使用されているため、used_varsに追加して、get_next_variable_nameが"objects"を返さないようにする
                if "objects" not in used_vars:
                    used_vars.add("objects")
                temp_var = "objects"
                num_excludes = len(valid_categories) - 1
                for category_id, category in enumerate(valid_categories[:-1]):
                    var_name = category_var_mapping.get(category.category_id)
                    if var_name:
                        # 最後のEXCLUDEの場合は、最後のカテゴリの変数名に直接代入
                        # それ以外は中間変数を使用（objectsを上書きしない）
                        if category_id == num_excludes - 1:
                            # 最後のEXCLUDE（最後のカテゴリの変数に代入）
                            partial_program += f"\n{last_var_name} = EXCLUDE({temp_var}, {var_name})"
                        else:
                            # 中間変数を使用（objectsを上書きしない）
                            next_temp_var = variable_naming_system.get_next_variable_name(
                                object_array_type,
                                used_vars
                            )
                            used_vars.add(next_temp_var)
                            partial_program += f"\n{next_temp_var} = EXCLUDE({temp_var}, {var_name})"
                            temp_var = next_temp_var

        return partial_program, category_var_mapping, None

    def _extract_used_params_from_condition(
        self, condition_str: str, category: CategoryInfo
    ) -> Dict[str, Dict[str, Any]]:
        """
        FILTER条件から使用されたパラメータ情報を抽出

        Args:
            condition_str: FILTER条件の文字列（例: "GREATER(GET_SIZE($obj), 15)"）
            category: カテゴリ情報

        Returns:
            使用されたパラメータ情報の辞書 {param_name: {min_val, max_val, colors, ...}}
        """
        used_params = {}

        # GET_COLORが使用されている場合
        if 'GET_COLOR' in condition_str:
            target_colors = category.representative_color or []
            if target_colors:
                used_params['color'] = {'colors': target_colors}

        # GET_SIZEが使用されている場合
        if 'GET_SIZE' in condition_str:
            if category.shape_info:
                min_size = category.shape_info.get('min_size')
                max_size = category.shape_info.get('max_size')
                if min_size is not None and max_size is not None:
                    used_params['size'] = {'min_val': min_size, 'max_val': max_size}

        # GET_WIDTHが使用されている場合
        if 'GET_WIDTH' in condition_str:
            if category.shape_info:
                min_width = category.shape_info.get('min_width')
                max_width = category.shape_info.get('max_width')
                if min_width is not None and max_width is not None:
                    used_params['width'] = {'min_val': min_width, 'max_val': max_width}

        # GET_HEIGHTが使用されている場合
        if 'GET_HEIGHT' in condition_str:
            if category.shape_info:
                min_height = category.shape_info.get('min_height')
                max_height = category.shape_info.get('max_height')
                if min_height is not None and max_height is not None:
                    used_params['height'] = {'min_val': min_height, 'max_val': max_height}

        # COUNT_HOLESが使用されている場合
        if 'COUNT_HOLES' in condition_str:
            if category.shape_info:
                min_hole_count = category.shape_info.get('min_hole_count')
                max_hole_count = category.shape_info.get('max_hole_count')
                if min_hole_count is not None and max_hole_count is not None:
                    used_params['hole_count'] = {'min_val': min_hole_count, 'max_val': max_hole_count}

        # GET_Xが使用されている場合
        if 'GET_X' in condition_str:
            if category.position_info:
                min_x = category.position_info.get('min_x')
                max_x = category.position_info.get('max_x')
                if min_x is not None and max_x is not None:
                    used_params['x'] = {'min_val': min_x, 'max_val': max_x}

        # GET_Yが使用されている場合
        if 'GET_Y' in condition_str:
            if category.position_info:
                min_y = category.position_info.get('min_y')
                max_y = category.position_info.get('max_y')
                if min_y is not None and max_y is not None:
                    used_params['y'] = {'min_val': min_y, 'max_val': max_y}

        # GET_RECTANGLE_TYPEが使用されている場合
        if 'GET_RECTANGLE_TYPE' in condition_str:
            if category.shape_info:
                rectangle_types = category.shape_info.get('rectangle_types', [])
                if rectangle_types:
                    used_params['rectangle_type'] = {'types': rectangle_types}

        # GET_LINE_TYPEが使用されている場合
        if 'GET_LINE_TYPE' in condition_str:
            if category.shape_info:
                line_types = category.shape_info.get('line_types', [])
                if line_types:
                    used_params['line_type'] = {'types': line_types}

        return used_params

    def _analyze_parameter_overlap(
        self, target_category: CategoryInfo, all_categories: List[CategoryInfo],
        target_category_id: int, all_objects: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        対象カテゴリの各パラメータの範囲内に他のカテゴリのオブジェクトがいくつ属するかを調べる

        Args:
            target_category: 対象カテゴリ
            all_categories: すべてのカテゴリ
            target_category_id: 対象カテゴリのID
            all_objects: すべてのオブジェクト

        Returns:
            パラメータ情報の辞書 {param_name: {overlap_count, overlap_object_ids, min_val, max_val, ...}}
        """
        parameter_info = {}

        # 対象カテゴリの代表色を取得
        target_colors = target_category.representative_color or []

        # 背景色を除外（念のため、安全策として）
        # objects_dataは既に背景色オブジェクトが除外されているはずなので、
        # representative_colorにも背景色は含まれないはずだが、念のため確認
        # bg_strategyは利用できないため、この時点では背景色の明示的な除外は行わない
        # （objects_dataは既に背景色オブジェクトが除外されているため、不要）

        # 他のカテゴリのすべてのオブジェクトを取得
        other_category_objects = []
        for other_id, other_category in enumerate(all_categories):
            if other_id != target_category_id:
                other_category_objects.extend(other_category.objects)

        # 色の分析
        if target_colors:
            overlap_object_ids = []
            for obj in other_category_objects:
                # objects_dataは既に背景色オブジェクトが除外されているはずなので、
                # obj.colorに背景色が含まれることはないはず
                if obj.color in target_colors:
                    overlap_object_ids.append(obj.index)  # オブジェクトID（識別可能な情報）
            parameter_info['color'] = {
                'overlap_count': len(overlap_object_ids),
                'overlap_object_ids': overlap_object_ids,
            }

        # 形状パラメータの分析（size, width, height, hole_count, symmetry_x, symmetry_y）
        # 対象カテゴリの形状情報を取得（存在する場合）
        target_shape_info = target_category.shape_info if hasattr(target_category, 'shape_info') else None

        # shape_infoが存在しない場合、オブジェクトから直接取得
        if target_shape_info is None:
            # shape_infoが存在しない場合、オブジェクトから直接計算
            shape_info = {}
            for obj in target_category.objects:
                if obj.size is not None:
                    shape_info.setdefault('sizes', []).append(obj.size)
                if obj.width is not None:
                    shape_info.setdefault('widths', []).append(obj.width)
                if obj.height is not None:
                    shape_info.setdefault('heights', []).append(obj.height)
                if obj.aspect_ratio is not None:
                    # GET_ASPECT_RATIOは(幅/高さ) * 100をintで返す
                    aspect_ratio_int = int(obj.aspect_ratio * 100)
                    shape_info.setdefault('aspect_ratios', []).append(aspect_ratio_int)
                # 密度を計算（GET_DENSITYは(ピクセル数/bbox面積) * 100をintで返す）
                if obj.width > 0 and obj.height > 0:
                    density = obj.size / (obj.width * obj.height)
                    density_int = int(density * 100)
                    shape_info.setdefault('densities', []).append(density_int)
                if obj.hole_count is not None:
                    shape_info.setdefault('hole_counts', []).append(obj.hole_count)
                if hasattr(obj, 'pixels') and obj.pixels:
                    sym_x = calculate_symmetry_score(obj, 'X')
                    if sym_x is not None:
                        shape_info.setdefault('symmetry_x_scores', []).append(sym_x)
                    sym_y = calculate_symmetry_score(obj, 'Y')
                    if sym_y is not None:
                        shape_info.setdefault('symmetry_y_scores', []).append(sym_y)
                    # 矩形タイプと線タイプを計算
                    rectangle_type = self._get_rectangle_type(obj)
                    line_type = self._get_line_type(obj)
                    shape_info.setdefault('rectangle_types', []).append(rectangle_type)
                    shape_info.setdefault('line_types', []).append(line_type)

            # min/maxを計算
            target_shape_info = {}
            if 'sizes' in shape_info:
                target_shape_info['min_size'] = min(shape_info['sizes'])
                target_shape_info['max_size'] = max(shape_info['sizes'])
            if 'widths' in shape_info:
                target_shape_info['min_width'] = min(shape_info['widths'])
                target_shape_info['max_width'] = max(shape_info['widths'])
            if 'heights' in shape_info:
                target_shape_info['min_height'] = min(shape_info['heights'])
                target_shape_info['max_height'] = max(shape_info['heights'])
            if 'aspect_ratios' in shape_info:
                target_shape_info['min_aspect_ratio'] = min(shape_info['aspect_ratios'])
                target_shape_info['max_aspect_ratio'] = max(shape_info['aspect_ratios'])
            if 'densities' in shape_info:
                target_shape_info['min_density'] = min(shape_info['densities'])
                target_shape_info['max_density'] = max(shape_info['densities'])
            if 'hole_counts' in shape_info:
                target_shape_info['min_hole_count'] = min(shape_info['hole_counts'])
                target_shape_info['max_hole_count'] = max(shape_info['hole_counts'])
            if 'symmetry_x_scores' in shape_info:
                target_shape_info['min_symmetry_x'] = min(shape_info['symmetry_x_scores'])
                target_shape_info['max_symmetry_x'] = max(shape_info['symmetry_x_scores'])
            if 'symmetry_y_scores' in shape_info:
                target_shape_info['min_symmetry_y'] = min(shape_info['symmetry_y_scores'])
                target_shape_info['max_symmetry_y'] = max(shape_info['symmetry_y_scores'])
            if 'rectangle_types' in shape_info:
                # ユニークな矩形タイプのリストを保存
                target_shape_info['rectangle_types'] = list(set(shape_info['rectangle_types']))
            if 'line_types' in shape_info:
                # ユニークな線タイプのリストを保存
                target_shape_info['line_types'] = list(set(shape_info['line_types']))
            if 'centroids' in shape_info:
                # ユニークな重心位置方向のリストを保存
                target_shape_info['centroids'] = list(set(shape_info['centroids']))

        shape_params = [
            ('size', 'min_size', 'max_size', 'GET_SIZE'),
            ('width', 'min_width', 'max_width', 'GET_WIDTH'),
            ('height', 'min_height', 'max_height', 'GET_HEIGHT'),
            ('aspect_ratio', 'min_aspect_ratio', 'max_aspect_ratio', 'GET_ASPECT_RATIO'),
            ('density', 'min_density', 'max_density', 'GET_DENSITY'),
            ('hole_count', 'min_hole_count', 'max_hole_count', 'COUNT_HOLES'),
            ('symmetry_x', 'min_symmetry_x', 'max_symmetry_x', 'GET_SYMMETRY_SCORE'),
            ('symmetry_y', 'min_symmetry_y', 'max_symmetry_y', 'GET_SYMMETRY_SCORE'),
        ]

        for param_name, min_key, max_key, command in shape_params:
            if target_shape_info and min_key in target_shape_info and max_key in target_shape_info:
                min_val = target_shape_info[min_key]
                max_val = target_shape_info[max_key]
            else:
                # shape_infoに存在しない場合、スキップ
                continue

            overlap_object_ids = []
            below_min_count = 0
            above_max_count = 0

            for obj in other_category_objects:
                # パラメータ値を取得（_get_param_valueを使用して一貫性を保つ）
                obj_val = self._get_param_value(obj, param_name)

                if obj_val is None:
                    continue

                if min_val <= obj_val <= max_val:
                    # 範囲内
                    overlap_object_ids.append(obj.index)
                elif obj_val < min_val:
                    # minより小さい
                    below_min_count += 1
                elif obj_val > max_val:
                    # maxより大きい
                    above_max_count += 1

            parameter_info[param_name] = {
                'overlap_count': len(overlap_object_ids),
                'overlap_object_ids': overlap_object_ids,
                'below_min_count': below_min_count,
                'above_max_count': above_max_count,
                'min_val': min_val,
                'max_val': max_val,
                'command': command,
            }

        # 位置パラメータの分析（x, y, center_x, center_y, max_x, max_y）
        if target_category.objects:
            position_info = {}
            x_coords = []
            y_coords = []
            center_x_coords = []
            center_y_coords = []
            max_x_coords = []
            max_y_coords = []
            for obj in target_category.objects:
                if obj.bbox and len(obj.bbox) == 4:
                    min_i, min_j, max_i, max_j = obj.bbox
                    x_coords.append(min_j)  # GET_Xはbbox_left
                    y_coords.append(min_i)  # GET_Yはbbox_top
                    max_x_coords.append(max_j)  # GET_MAX_Xはbbox_right
                    max_y_coords.append(max_i)  # GET_MAX_Yはbbox_bottom
                if obj.center and len(obj.center) == 2:
                    center_y, center_x = obj.center  # centerは(center_y, center_x)形式
                    center_x_coords.append(int(center_x))  # GET_CENTER_X
                    center_y_coords.append(int(center_y))  # GET_CENTER_Y

            if x_coords:
                position_info['min_x'] = min(x_coords)
                position_info['max_x'] = max(x_coords)
            if y_coords:
                position_info['min_y'] = min(y_coords)
                position_info['max_y'] = max(y_coords)
            if center_x_coords:
                position_info['min_center_x'] = min(center_x_coords)
                position_info['max_center_x'] = max(center_x_coords)
            if center_y_coords:
                position_info['min_center_y'] = min(center_y_coords)
                position_info['max_center_y'] = max(center_y_coords)
            if max_x_coords:
                position_info['min_max_x'] = min(max_x_coords)
                position_info['max_max_x'] = max(max_x_coords)
            if max_y_coords:
                position_info['min_max_y'] = min(max_y_coords)
                position_info['max_max_y'] = max(max_y_coords)

            position_params = [
                ('x', 'min_x', 'max_x', 'GET_X'),
                ('y', 'min_y', 'max_y', 'GET_Y'),
                ('center_x', 'min_center_x', 'max_center_x', 'GET_CENTER_X'),
                ('center_y', 'min_center_y', 'max_center_y', 'GET_CENTER_Y'),
                ('max_x', 'min_max_x', 'max_max_x', 'GET_MAX_X'),
                ('max_y', 'min_max_y', 'max_max_y', 'GET_MAX_Y'),
            ]

            for param_name, min_key, max_key, command in position_params:
                if min_key in position_info and max_key in position_info:
                    min_val = position_info[min_key]
                    max_val = position_info[max_key]

                    overlap_object_ids = []
                    below_min_count = 0
                    above_max_count = 0

                    for obj in other_category_objects:
                        obj_val = None
                        if param_name == 'x':
                            if obj.bbox and len(obj.bbox) == 4:
                                min_i, min_j, max_i, max_j = obj.bbox
                                obj_val = min_j  # GET_Xはbbox_left
                        elif param_name == 'y':
                            if obj.bbox and len(obj.bbox) == 4:
                                min_i, min_j, max_i, max_j = obj.bbox
                                obj_val = min_i  # GET_Yはbbox_top
                        elif param_name == 'center_x':
                            if obj.center and len(obj.center) == 2:
                                center_y, center_x = obj.center
                                obj_val = int(center_x)
                        elif param_name == 'center_y':
                            if obj.center and len(obj.center) == 2:
                                center_y, center_x = obj.center
                                obj_val = int(center_y)
                        elif param_name == 'max_x':
                            if obj.bbox and len(obj.bbox) == 4:
                                min_i, min_j, max_i, max_j = obj.bbox
                                obj_val = max_j  # GET_MAX_Xはbbox_right
                        elif param_name == 'max_y':
                            if obj.bbox and len(obj.bbox) == 4:
                                min_i, min_j, max_i, max_j = obj.bbox
                                obj_val = max_i  # GET_MAX_Yはbbox_bottom

                        if obj_val is None:
                            continue

                        if min_val <= obj_val <= max_val:
                            overlap_object_ids.append(obj.index)
                        elif obj_val < min_val:
                            below_min_count += 1
                        elif obj_val > max_val:
                            above_max_count += 1

                    parameter_info[param_name] = {
                        'overlap_count': len(overlap_object_ids),
                        'overlap_object_ids': overlap_object_ids,
                        'below_min_count': below_min_count,
                        'above_max_count': above_max_count,
                        'min_val': min_val,
                        'max_val': max_val,
                        'command': command,
                    }

        # 矩形タイプと線タイプの分析（文字列値、EQUAL条件のみ）
        if target_shape_info:
            # 矩形タイプ
            if 'rectangle_types' in target_shape_info:
                rectangle_types = target_shape_info['rectangle_types']
                if rectangle_types:
                    # 他のカテゴリのオブジェクトで同じタイプを持つものをチェック
                    overlap_object_ids = []
                    for obj in other_category_objects:
                        obj_rectangle_type = self._get_rectangle_type(obj)
                        if obj_rectangle_type in rectangle_types:
                            overlap_object_ids.append(obj.index)

                    parameter_info['rectangle_type'] = {
                        'overlap_count': len(overlap_object_ids),
                        'overlap_object_ids': overlap_object_ids,
                        'types': rectangle_types,
                        'command': 'GET_RECTANGLE_TYPE',
                    }

            # 線タイプ
            if 'line_types' in target_shape_info:
                line_types = target_shape_info['line_types']
                if line_types:
                    # 他のカテゴリのオブジェクトで同じタイプを持つものをチェック
                    overlap_object_ids = []
                    for obj in other_category_objects:
                        obj_line_type = self._get_line_type(obj)
                        if obj_line_type in line_types:
                            overlap_object_ids.append(obj.index)

                    parameter_info['line_type'] = {
                        'overlap_count': len(overlap_object_ids),
                        'overlap_object_ids': overlap_object_ids,
                        'types': line_types,
                        'command': 'GET_LINE_TYPE',
                    }

            # 重心位置方向
            if 'centroids' in target_shape_info:
                centroids = target_shape_info['centroids']
                if centroids:
                    # 他のカテゴリのオブジェクトで同じ重心位置方向を持つものをチェック
                    overlap_object_ids = []
                    for obj in other_category_objects:
                        obj_centroid = self._get_centroid(obj)
                        if obj_centroid in centroids:
                            overlap_object_ids.append(obj.index)

                    parameter_info['centroid'] = {
                        'overlap_count': len(overlap_object_ids),
                        'overlap_object_ids': overlap_object_ids,
                        'types': centroids,
                        'command': 'GET_CENTROID',
                    }

        return parameter_info

    def _generate_category_unique_condition(
        self, target_category: CategoryInfo, all_categories: List[CategoryInfo],
        target_category_id: int, all_objects: List[Any],
        max_parameter_combinations: int = 2
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        カテゴリ固有の条件を生成

        Args:
            max_parameter_combinations: 最大パラメータ条件併用数（デフォルト値2）

        Returns:
            (条件文字列, 成功フラグ, デバッグ情報)
            成功した場合: (条件文字列, True, {})
            失敗した場合: (None, False, デバッグ情報)
        """
        # デバッグ情報を初期化
        debug_info = {
            'tried_parameters': [],
            'tried_combinations': [],
            'failure_reason': None,
            'overlap_objects': []
        }
        # 1. 対象カテゴリの各パラメータの範囲内に他のカテゴリのオブジェクトがいくつ属するかを調べる
        parameter_info = self._analyze_parameter_overlap(
            target_category, all_categories, target_category_id, all_objects
        )

        # 2. 他のカテゴリのオブジェクト数が少ない順にパラメータを並べる
        # 同じ数の場合は、パラメータの優先順位で決定
        def _get_parameter_priority(param_name: str) -> int:
            """
            パラメータの優先順位を取得（数値が小さいほど優先度が高い）

            優先順位:
            1. 色情報（color）- 視覚的に明確
            2. 基本形状情報（size, width, height, aspect_ratio）- 構造的特徴
            3. 位置情報（x, y, center_x, center_y, max_x, max_y）- 具体的で有用
            4. 詳細形状情報（hole_count, density）- 詳細な構造的特徴
            5. 特殊な形状情報（centroid）- 重心位置方向
            6. 対称性情報（symmetry_x, symmetry_y）- 特殊な特徴
            """
            priority_map = {
                # 色情報（最優先）
                'color': 10,
                # 基本形状情報
                'size': 20,
                'width': 21,
                'height': 22,
                'aspect_ratio': 23,  # アスペクト比（形状の基本情報）
                # 位置情報
                'x': 25,
                'y': 26,
                'center_x': 27,  # 中心X座標
                'center_y': 28,  # 中心Y座標
                'max_x': 29,  # 最大X座標
                'max_y': 30,  # 最大Y座標
                # 詳細形状情報
                'hole_count': 35,
                'density': 36,  # 密度（詳細な形状情報）
                # 特殊な形状情報
                'centroid': 38,  # 重心位置方向
                # 対称性情報
                'symmetry_x': 40,
                'symmetry_y': 41,
            }
            return priority_map.get(param_name, 100)  # 未知のパラメータは最後

        sorted_parameters = sorted(
            parameter_info.items(),
            key=lambda x: (
                x[1].get('overlap_count', float('inf')),  # 第一キー: 他のカテゴリのオブジェクト数
                _get_parameter_priority(x[0])  # 第二キー: パラメータの優先順位
            )
        )

        # ループ2: 他のカテゴリのオブジェクト数が少ない順にパラメータを調べる
        for param_name, param_data in sorted_parameters:
            object_conditions = []  # オブジェクト条件のリスト
            success = False

            # デバッグ情報: 試したパラメータを記録
            param_debug = {
                'param_name': param_name,
                'overlap_count': param_data.get('overlap_count', 0),
                'tried_combinations': []
            }

            # 条件リストに条件を追加
            condition_str = self._generate_parameter_condition(
                param_name, param_data, target_category
            )
            if condition_str is None:
                # 条件を生成できない場合（例: 色が3つ以上）
                debug_info['tried_parameters'].append({
                    'param_name': param_name,
                    'overlap_count': param_data.get('overlap_count', 0),
                    'reason': 'condition_generation_failed'
                })
                continue

            object_conditions.append(condition_str)

            # 対象パラメータの他のカテゴリのオブジェクト数が0の場合
            if param_data.get('overlap_count', 0) == 0:
                success = True
                debug_info['tried_parameters'].append({
                    'param_name': param_name,
                    'overlap_count': 0,
                    'success': True
                })
                break

            # オブジェクト数が0でない場合: 他のパラメータと組み合わせる
            # 他のパラメータを、重複する他のカテゴリのオブジェクトIDが少ない順に並び替えたリストを作る
            other_params = [
                (name, data) for name, data in sorted_parameters
                if name != param_name
            ]
            other_params_sorted = sorted(
                other_params,
                key=lambda x: len(set(x[1].get('overlap_object_ids', [])) &
                                 set(param_data.get('overlap_object_ids', [])))
            )

            # ループ3: 最大パラメータ条件併用数-1でループ
            # 目的: 最初のパラメータに追加のパラメータを組み合わせて、
            # すべての他のカテゴリのオブジェクトを除外できる条件を見つける
            for i in range(min(max_parameter_combinations - 1, len(other_params_sorted))):
                other_param_name, other_param_data = other_params_sorted[i]

                # 条件リストに条件を追加
                other_condition_str = self._generate_parameter_condition(
                    other_param_name, other_param_data, target_category
                )
                if other_condition_str is None:
                    continue

                # 条件リストに追加
                object_conditions.append(other_condition_str)

                # ステップ1: 現在の条件リスト（最初のパラメータ + これまでに追加したパラメータ）に属する
                # 他のカテゴリのオブジェクトIDをすべて収集（和集合）
                # これにより、現在の条件の組み合わせで影響を受けるすべての他のカテゴリのオブジェクトを特定
                all_overlap_ids = set(param_data.get('overlap_object_ids', []))
                # ループ3でこれまでに追加したパラメータ（インデックス0からiまで）のオーバーラップIDも追加
                for j in range(i + 1):
                    all_overlap_ids.update(other_params_sorted[j][1].get('overlap_object_ids', []))

                # ステップ2: 成功条件のチェック
                # 成功条件: 条件に属する他のカテゴリのオブジェクトそれぞれすべてが、
                # 最低１つ以上属していないパラメータの条件を持つこと
                #
                # つまり、AND条件で結合した場合:
                # - すべての他のカテゴリのオブジェクトが、最低1つ以上の条件に当てはまらない
                # - これにより、AND条件で結合すると、すべての他のカテゴリのオブジェクトが除外される
                # - 結果として、対象カテゴリのオブジェクトのみが残る
                success = True
                for overlap_obj_id in all_overlap_ids:
                    # このオブジェクトが、現在の条件リストのすべてのパラメータに属しているかチェック
                    obj_in_all_params = True

                    # 最初のパラメータをチェック
                    if overlap_obj_id not in param_data.get('overlap_object_ids', []):
                        # このオブジェクトは、最初のパラメータの条件に属していない
                        # つまり、最初の条件で除外される
                        obj_in_all_params = False
                    else:
                        # ループ3で追加したパラメータ（インデックス0からiまで）をチェック
                        for j in range(i + 1):
                            if overlap_obj_id not in other_params_sorted[j][1].get('overlap_object_ids', []):
                                # このオブジェクトは、このパラメータの条件に属していない
                                # つまり、この条件で除外される
                                obj_in_all_params = False
                                break

                    if obj_in_all_params:
                        # このオブジェクトは、すべての条件に属している
                        # つまり、AND条件で結合しても除外されない
                        # この場合、成功条件を満たさない
                        success = False
                        # デバッグ情報: 失敗したオブジェクトIDを記録
                        debug_info['overlap_objects'].append({
                            'obj_id': overlap_obj_id,
                            'all_conditions': [param_name] + [other_params_sorted[j][0] for j in range(i + 1)]
                        })
                        break

                # デバッグ情報: 試した組み合わせを記録
                param_debug['tried_combinations'].append({
                    'parameters': [param_name] + [other_params_sorted[j][0] for j in range(i + 1)],
                    'conditions': object_conditions.copy(),
                    'success': success,
                    'all_overlap_ids_count': len(all_overlap_ids)
                })

                if success:
                    # すべての他のカテゴリのオブジェクトが、最低1つ以上の条件に属していない
                    # つまり、AND条件で結合すると、すべての他のカテゴリのオブジェクトが除外される
                    # 成功: 対象カテゴリ固有の条件が見つかった
                    debug_info['tried_parameters'].append(param_debug)
                    break

            # デバッグ情報: 試したパラメータを記録（成功しなかった場合）
            if not success:
                debug_info['tried_parameters'].append(param_debug)

            if success:
                break

        if success:
            # 条件をANDで結合
            if len(object_conditions) == 1:
                combined_condition = object_conditions[0]
            elif len(object_conditions) == 2:
                combined_condition = f"AND({object_conditions[0]}, {object_conditions[1]})"
            else:
                # 3つ以上の条件をANDで結合: AND(AND(条件1, 条件2), 条件3)
                combined_condition = object_conditions[0]
                for cond in object_conditions[1:]:
                    combined_condition = f"AND({combined_condition}, {cond})"

            # 成功した場合、デバッグ情報は空で返す
            return combined_condition, True, {}
        else:
            # AND条件で失敗した場合、OR条件を試す
            if self.config.enable_or_condition:
                # 優先順位2: OR条件 - 単一パラメータの領域の組み合わせ
                or_condition, or_success, or_debug_info = self._generate_or_condition_single_param(
                    target_category, all_categories, target_category_id, all_objects,
                    self.config.max_or_region_combinations
                )

                if or_success:
                    debug_info['tried_or_conditions'] = [{
                        'type': 'single_param',
                        'success': True,
                        'condition': or_condition
                    }]
                    return or_condition, True, {}

                # 優先順位3: OR条件 - 2つのパラメータの領域の組み合わせ
                or_condition, or_success, or_debug_info = self._generate_or_condition_multi_param(
                    target_category, all_categories, target_category_id, all_objects,
                    self.config.max_or_region_combinations
                )

                if or_success:
                    debug_info['tried_or_conditions'] = [{
                        'type': 'multi_param',
                        'success': True,
                        'condition': or_condition
                    }]
                    return or_condition, True, {}

                # OR条件でも失敗した場合、デバッグ情報を記録
                debug_info['tried_or_conditions'] = or_debug_info.get('tried_params', []) + or_debug_info.get('tried_param_combinations', [])

            # すべての条件生成方法で失敗した場合
            debug_info['failure_reason'] = 'no_unique_condition_found'
            if not debug_info['tried_parameters']:
                debug_info['failure_reason'] = 'no_parameters_available'
            return None, False, debug_info

    def _generate_parameter_condition(
        self, param_name: str, param_data: Dict[str, Any],
        target_category: CategoryInfo
    ) -> Optional[str]:
        """
        パラメータから条件文字列を生成

        Returns:
            条件文字列、またはNone（条件を生成できない場合）
        """
        if param_name == 'color':
            # 色の場合
            target_colors = target_category.representative_color or []
            if len(target_colors) == 0:
                return None
            elif len(target_colors) == 1:
                return f"EQUAL(GET_COLOR($obj), {target_colors[0]})"
            elif len(target_colors) == 2:
                return f"OR(EQUAL(GET_COLOR($obj), {target_colors[0]}), EQUAL(GET_COLOR($obj), {target_colors[1]}))"
            else:
                # 色が3つ以上の場合は条件に使えない
                return None

        elif param_name == 'centroid':
            # 重心位置方向の場合（文字列値、EQUAL条件のみ）
            if target_category.shape_info and 'centroids' in target_category.shape_info:
                centroids = target_category.shape_info['centroids']
            else:
                # shape_infoに存在しない場合、オブジェクトから直接計算
                centroids = []
                for obj in target_category.objects:
                    if isinstance(obj, ObjectInfo):
                        centroid = self._get_centroid(obj)
                        if centroid != "none":
                            centroids.append(centroid)
                centroids = list(set(centroids))

            if not centroids:
                return None

            # ユニークな重心位置方向のみを使用
            unique_centroids = list(set(centroids))
            if len(unique_centroids) == 1:
                return f'EQUAL(GET_CENTROID($obj), "{unique_centroids[0]}")'
            elif len(unique_centroids) == 2:
                return f'OR(EQUAL(GET_CENTROID($obj), "{unique_centroids[0]}"), EQUAL(GET_CENTROID($obj), "{unique_centroids[1]}"))'
            else:
                # 3つ以上の場合は条件に使えない（OR条件で処理される可能性がある）
                return None

        elif param_name == 'rectangle_type':
            # 矩形タイプの場合（文字列値、EQUAL条件のみ）
            types = param_data.get('types', [])
            if not types:
                return None

            # ユニークなタイプのみを使用
            unique_types = list(set(types))
            if len(unique_types) == 1:
                return f'EQUAL(GET_RECTANGLE_TYPE($obj), "{unique_types[0]}")'
            elif len(unique_types) == 2:
                return f'OR(EQUAL(GET_RECTANGLE_TYPE($obj), "{unique_types[0]}"), EQUAL(GET_RECTANGLE_TYPE($obj), "{unique_types[1]}"))'
            else:
                # 3つ以上の場合は条件に使えない（OR条件で処理される可能性がある）
                return None

        elif param_name == 'line_type':
            # 線タイプの場合（文字列値、EQUAL条件のみ）
            types = param_data.get('types', [])
            if not types:
                return None

            # ユニークなタイプのみを使用
            unique_types = list(set(types))
            if len(unique_types) == 1:
                return f'EQUAL(GET_LINE_TYPE($obj), "{unique_types[0]}")'
            elif len(unique_types) == 2:
                return f'OR(EQUAL(GET_LINE_TYPE($obj), "{unique_types[0]}"), EQUAL(GET_LINE_TYPE($obj), "{unique_types[1]}"))'
            else:
                # 3つ以上の場合は条件に使えない（OR条件で処理される可能性がある）
                return None

        else:
            # 色以外のパラメータ（size, width, height, hole_count, symmetry_x, symmetry_y, x, y）
            min_val = param_data.get('min_val')
            max_val = param_data.get('max_val')
            below_min_count = param_data.get('below_min_count', 0)
            above_max_count = param_data.get('above_max_count', 0)
            command = param_data.get('command')

            if min_val is None or max_val is None or command is None:
                return None

            # コマンド式を構築（_build_command_exprを使用して一貫性を保つ）
            # GET_SYMMETRY_SCOREの場合は引数が2つなので、_build_command_exprで適切に処理される
            command_expr = self._build_command_expr(param_name)

            if below_min_count > 0 and above_max_count > 0:
                # minより小さくて範囲外 AND maxより大きくて範囲外
                return f"AND(GREATER({command_expr}, {min_val - 1}), LESS({command_expr}, {max_val + 1}))"
            elif below_min_count == 0 and above_max_count > 0:
                # maxより大きくて範囲外のみ
                return f"LESS({command_expr}, {max_val + 1})"
            elif below_min_count > 0 and above_max_count == 0:
                # minより小さくて範囲外のみ
                return f"GREATER({command_expr}, {min_val - 1})"
            else:
                # 範囲外のオブジェクトがない場合（すべて範囲内）
                # この場合は条件として使えない（他のカテゴリのオブジェクトもすべて含まれる）
                return None

    def _analyze_category_features(self, category: CategoryInfo) -> Dict[str, Any]:
        """
        カテゴリ内のすべてのオブジェクトの特徴量を分析

        Args:
            category: カテゴリ情報

        Returns:
            特徴量の辞書:
            - colors: 色のリスト
            - sizes: サイズのリスト
            - widths: 幅のリスト
            - heights: 高さのリスト
            - aspect_ratios: アスペクト比のリスト
            - hole_counts: 穴の数のリスト
            - symmetry_x_scores: X軸対称性スコアのリスト
            - symmetry_y_scores: Y軸対称性スコアのリスト
            - positions: 中心座標のリスト
            - x_coords: X座標のリスト（GET_X、bboxから取得）
            - y_coords: Y座標のリスト（GET_Y、bboxから取得）
        """
        features = {
            'colors': [],
            'sizes': [],
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'densities': [],  # 密度を追加
            'hole_counts': [],
            'symmetry_x_scores': [],
            'symmetry_y_scores': [],
            'positions': [],
            'x_coords': [],
            'y_coords': [],
            'center_x_coords': [],  # 中心X座標を追加
            'center_y_coords': [],  # 中心Y座標を追加
            'max_x_coords': [],  # 最大X座標を追加
            'max_y_coords': [],  # 最大Y座標を追加
            'centroids': [],  # 重心位置方向を追加
            'rectangle_types': [],
            'line_types': []
        }

        if not category.objects:
            return features

        # すべてのオブジェクトの特徴量を収集
        for obj in category.objects:
            if obj.color is not None:
                features['colors'].append(obj.color)
            if obj.size is not None:
                features['sizes'].append(obj.size)
            if obj.width is not None:
                features['widths'].append(obj.width)
            if obj.height is not None:
                features['heights'].append(obj.height)
            if obj.aspect_ratio is not None:
                # GET_ASPECT_RATIOは(幅/高さ) * 100をintで返す
                aspect_ratio_int = int(obj.aspect_ratio * 100) if obj.aspect_ratio is not None else None
                if aspect_ratio_int is not None:
                    features['aspect_ratios'].append(aspect_ratio_int)
            if obj.hole_count is not None:
                features['hole_counts'].append(obj.hole_count)
            if obj.center is not None:
                features['positions'].append(obj.center)
            if obj.bbox and len(obj.bbox) == 4:
                min_i, min_j, max_i, max_j = obj.bbox
                # GET_Xはbbox_left（min_j）、GET_Yはbbox_top（min_i）
                features['x_coords'].append(min_j)
                features['y_coords'].append(min_i)
                # GET_MAX_Xはbbox_right（max_j）、GET_MAX_Yはbbox_bottom（max_i）
                features['max_x_coords'].append(max_j)
                features['max_y_coords'].append(max_i)

            # 密度を計算（GET_DENSITYは(ピクセル数/bbox面積) * 100をintで返す）
            if obj.width > 0 and obj.height > 0:
                density = obj.size / (obj.width * obj.height)
                density_int = int(density * 100)
                features['densities'].append(density_int)

            # 中心座標を取得
            if obj.center and len(obj.center) == 2:
                center_y, center_x = obj.center  # centerは(center_y, center_x)形式
                features['center_x_coords'].append(int(center_x))
                features['center_y_coords'].append(int(center_y))
            # 対称性スコアを計算
            if isinstance(obj, ObjectInfo):
                sym_x = calculate_symmetry_score(obj, 'X')
                sym_y = calculate_symmetry_score(obj, 'Y')
                features['symmetry_x_scores'].append(sym_x)
                features['symmetry_y_scores'].append(sym_y)

                # 矩形タイプ、線タイプ、重心位置方向を計算
                rectangle_type = self._get_rectangle_type(obj)
                line_type = self._get_line_type(obj)
                centroid = self._get_centroid(obj)
                features['rectangle_types'].append(rectangle_type)
                features['line_types'].append(line_type)
                if centroid != "none":
                    features['centroids'].append(centroid)
            else:
                # ObjectInfo以外の場合は0.0を設定
                features['symmetry_x_scores'].append(0.0)
                features['symmetry_y_scores'].append(0.0)
                features['rectangle_types'].append("none")
                features['line_types'].append("none")

        return features

    # ============================================================
    # OR条件生成メソッド
    # ============================================================

    def _find_exclusive_regions(
        self, param_name: str, target_category: CategoryInfo, all_categories: List[CategoryInfo]
    ) -> List[Dict[str, Any]]:
        """
        各パラメータで「そのカテゴリのオブジェクトだけが含まれる領域」を見つける

        Args:
            param_name: パラメータ名（'color', 'x', 'y', 'size', etc.）
            target_category: 対象カテゴリ
            all_categories: すべてのカテゴリ

        Returns:
            領域のリスト（各領域は 'param_name', 'condition', 'object_indices' を持つ）
        """
        if param_name == 'color':
            return self._find_exclusive_regions_for_discrete(param_name, target_category, all_categories)
        elif param_name in ['rectangle_type', 'line_type', 'centroid']:
            return self._find_exclusive_regions_for_discrete(param_name, target_category, all_categories)
        elif param_name in ['x', 'y', 'size', 'width', 'height', 'aspect_ratio', 'density', 'hole_count', 'symmetry_x', 'symmetry_y', 'center_x', 'center_y', 'max_x', 'max_y']:
            return self._find_exclusive_regions_for_continuous(param_name, target_category, all_categories)
        else:
            return []

    def _find_exclusive_regions_for_discrete(
        self, param_name: str, target_category: CategoryInfo, all_categories: List[CategoryInfo]
    ) -> List[Dict[str, Any]]:
        """
        離散値パラメータ（colorなど）で、そのカテゴリのオブジェクトだけが含まれる領域を見つける
        """
        # そのカテゴリの値のセット
        target_values = set()
        for obj in target_category.objects:
                if param_name == 'color' and obj.color is not None:
                    target_values.add(obj.color)
                elif param_name == 'rectangle_type':
                    rectangle_type = self._get_rectangle_type(obj)
                    if rectangle_type != "none":
                        target_values.add(rectangle_type)
                elif param_name == 'line_type':
                    line_type = self._get_line_type(obj)
                    if line_type != "none":
                        target_values.add(line_type)
                elif param_name == 'centroid':
                    centroid = self._get_centroid(obj)
                    if centroid != "none":
                        target_values.add(centroid)

        if not target_values:
            return []

        # 他のカテゴリの値のセット
        other_values = set()
        for cat in all_categories:
            if cat.category_id != target_category.category_id:
                for obj in cat.objects:
                    if param_name == 'color' and obj.color is not None:
                        other_values.add(obj.color)
                    elif param_name == 'rectangle_type':
                        rectangle_type = self._get_rectangle_type(obj)
                        if rectangle_type != "none":
                            other_values.add(rectangle_type)
                    elif param_name == 'line_type':
                        line_type = self._get_line_type(obj)
                        if line_type != "none":
                            other_values.add(line_type)
                    elif param_name == 'centroid':
                        centroid = self._get_centroid(obj)
                        if centroid != "none":
                            other_values.add(centroid)

        # そのカテゴリだけに含まれる値
        exclusive_values = target_values - other_values

        regions = []
        for value in exclusive_values:
            # その領域に含まれるオブジェクトのインデックスを取得
            object_indices = []
            for obj in target_category.objects:
                if param_name == 'color' and obj.color == value:
                    object_indices.append(obj.index)
                elif param_name == 'rectangle_type':
                    rectangle_type = self._get_rectangle_type(obj)
                    if rectangle_type == value:
                        object_indices.append(obj.index)
                elif param_name == 'line_type':
                    line_type = self._get_line_type(obj)
                    if line_type == value:
                        object_indices.append(obj.index)
                elif param_name == 'centroid':
                    centroid = self._get_centroid(obj)
                    if centroid == value:
                        object_indices.append(obj.index)

            if object_indices:
                # 離散値パラメータ（color, rectangle_type, line_type, centroid）の場合
                if param_name == 'color':
                    regions.append({
                        'param_name': param_name,
                        'condition': f"EQUAL(GET_COLOR($obj), {value})",
                        'object_indices': object_indices
                    })
                elif param_name == 'rectangle_type':
                    regions.append({
                        'param_name': param_name,
                        'condition': f'EQUAL(GET_RECTANGLE_TYPE($obj), "{value}")',
                        'object_indices': object_indices
                    })
                elif param_name == 'line_type':
                    regions.append({
                        'param_name': param_name,
                        'condition': f'EQUAL(GET_LINE_TYPE($obj), "{value}")',
                        'object_indices': object_indices
                    })
                elif param_name == 'centroid':
                    regions.append({
                        'param_name': param_name,
                        'condition': f'EQUAL(GET_CENTROID($obj), "{value}")',
                        'object_indices': object_indices
                    })

        return regions

    def _find_exclusive_regions_for_continuous(
        self, param_name: str, target_category: CategoryInfo, all_categories: List[CategoryInfo]
    ) -> List[Dict[str, Any]]:
        """
        連続値パラメータ（x, y, sizeなど）で、そのカテゴリのオブジェクトだけが含まれる領域を見つける
        """
        # そのカテゴリの値のリスト
        target_values = []
        value_to_objects = {}  # 値からオブジェクトのインデックスへのマッピング

        for obj in target_category.objects:
            obj_val = self._get_param_value(obj, param_name)
            if obj_val is not None:
                target_values.append(obj_val)
                if obj_val not in value_to_objects:
                    value_to_objects[obj_val] = []
                value_to_objects[obj_val].append(obj.index)

        if not target_values:
            return []

        target_values = sorted(set(target_values))

        # 他のカテゴリの値のセット
        other_values = set()
        for cat in all_categories:
            if cat.category_id != target_category.category_id:
                for obj in cat.objects:
                    obj_val = self._get_param_value(obj, param_name)
                    if obj_val is not None:
                        other_values.add(obj_val)

        regions = []

        # コマンド式を一度だけ計算（すべての領域タイプで再利用）
        command_expr = self._build_command_expr(param_name)

        # 単一値領域を見つける
        for value in target_values:
            if value not in other_values:
                object_indices = value_to_objects.get(value, [])
                if object_indices:
                    regions.append({
                        'param_name': param_name,
                        'condition': f"EQUAL({command_expr}, {value})",
                        'object_indices': object_indices
                    })

        # 範囲領域を見つける
        for i in range(len(target_values) - 1):
            val1 = target_values[i]
            val2 = target_values[i + 1]

            # 範囲内の値が他のカテゴリに含まれないかチェック
            range_values = [v for v in target_values if val1 < v < val2]
            if not range_values or all(v not in other_values for v in range_values):
                if val1 not in other_values and val2 not in other_values:
                    # 範囲内のオブジェクトのインデックスを取得
                    object_indices = []
                    for obj in target_category.objects:
                        obj_val = self._get_param_value(obj, param_name)
                        if obj_val is not None and val1 < obj_val < val2:
                            object_indices.append(obj.index)

                    if object_indices:
                        regions.append({
                            'param_name': param_name,
                            'condition': f"AND(GREATER({command_expr}, {val1}), LESS({command_expr}, {val2}))",
                            'object_indices': object_indices
                        })

        # 以上領域（x > 15）や以下領域（x < 1）を見つける
        min_target = min(target_values)
        max_target = max(target_values)

        if min_target not in other_values:
            object_indices = value_to_objects.get(min_target, [])
            if object_indices:
                regions.append({
                    'param_name': param_name,
                    'condition': f"LESS({command_expr}, {min_target + 1})",
                    'object_indices': object_indices
                })

        if max_target not in other_values:
            object_indices = value_to_objects.get(max_target, [])
            if object_indices:
                regions.append({
                    'param_name': param_name,
                    'condition': f"GREATER({command_expr}, {max_target - 1})",
                    'object_indices': object_indices
                })

        return regions

    def _get_param_value(self, obj: Any, param_name: str) -> Optional[float]:
        """
        オブジェクトからパラメータ値を取得

        Args:
            obj: オブジェクト
            param_name: パラメータ名

        Returns:
            パラメータ値、またはNone
        """
        if param_name == 'x':
            if obj.bbox and len(obj.bbox) == 4:
                return float(obj.bbox[1])  # GET_Xはbbox_left（min_j）
        elif param_name == 'y':
            if obj.bbox and len(obj.bbox) == 4:
                return float(obj.bbox[0])  # GET_Yはbbox_top（min_i）
        elif param_name == 'size':
            return float(obj.size) if obj.size is not None else None
        elif param_name == 'width':
            return float(obj.width) if obj.width is not None else None
        elif param_name == 'height':
            return float(obj.height) if obj.height is not None else None
        elif param_name == 'hole_count':
            return float(obj.hole_count) if obj.hole_count is not None else None
        elif param_name == 'symmetry_x':
            return calculate_symmetry_score(obj, 'X')
        elif param_name == 'symmetry_y':
            return calculate_symmetry_score(obj, 'Y')
        elif param_name == 'aspect_ratio':
            # GET_ASPECT_RATIOは(幅/高さ) * 100をintで返す
            if obj.aspect_ratio is not None:
                return float(int(obj.aspect_ratio * 100))
            return None
        elif param_name == 'density':
            # GET_DENSITYは(ピクセル数/bbox面積) * 100をintで返す
            if obj.width > 0 and obj.height > 0:
                density = obj.size / (obj.width * obj.height)
                return float(int(density * 100))
            return None
        elif param_name == 'center_x':
            if obj.center and len(obj.center) == 2:
                center_y, center_x = obj.center  # centerは(center_y, center_x)形式
                return float(center_x)
        elif param_name == 'center_y':
            if obj.center and len(obj.center) == 2:
                center_y, center_x = obj.center  # centerは(center_y, center_x)形式
                return float(center_y)
        elif param_name == 'max_x':
            if obj.bbox and len(obj.bbox) == 4:
                min_i, min_j, max_i, max_j = obj.bbox
                return float(max_j)  # GET_MAX_Xはbbox_right
        elif param_name == 'max_y':
            if obj.bbox and len(obj.bbox) == 4:
                min_i, min_j, max_i, max_j = obj.bbox
                return float(max_i)  # GET_MAX_Yはbbox_bottom
        elif param_name == 'rectangle_type':
            # 文字列を返すが、比較のために数値に変換する必要がある場合がある
            # ここでは文字列のまま返す（EQUAL条件でのみ使用）
            return None  # 文字列値は別途処理
        elif param_name == 'line_type':
            # 文字列を返すが、比較のために数値に変換する必要がある場合がある
            # ここでは文字列のまま返す（EQUAL条件でのみ使用）
            return None  # 文字列値は別途処理

        return None

    def _get_rectangle_type(self, obj: ObjectInfo) -> str:
        """矩形の種類を判定

        Args:
            obj: オブジェクト

        Returns:
            "filled" - 塗りつぶされた矩形
            "hollow" - 中空矩形（枠線のみ、厚さ1ピクセル）
            "none"   - 矩形ではない
        """
        try:
            if not obj.pixels or len(obj.pixels) < 4:
                return "none"

            # bboxを取得
            if not obj.bbox or len(obj.bbox) != 4:
                return "none"

            min_i, min_j, max_i, max_j = obj.bbox
            width = max_j - min_j + 1
            height = max_i - min_i + 1

            # 最小サイズチェック
            if width < 2 or height < 2:
                return "none"

            pixel_set = set(obj.pixels)

            # 塗りつぶし矩形かチェック（すべてのピクセルが存在）
            expected_pixels = width * height
            if len(obj.pixels) == expected_pixels:
                # すべてのピクセルが矩形範囲内に存在するか確認
                all_present = True
                for y in range(height):
                    for x in range(width):
                        if (min_j + x, min_i + y) not in pixel_set:
                            all_present = False
                            break
                    if not all_present:
                        break

                if all_present:
                    return "filled"

            # 中空矩形かチェック（3x3以上、厚さ1ピクセル）
            if width >= 3 and height >= 3:
                # 4辺がすべて存在するか（ExecutorCoreの実装に合わせる）
                has_all_edges = True

                # 上辺
                for x in range(width):
                    if (min_j + x, min_i) not in pixel_set:
                        has_all_edges = False
                        break

                # 下辺
                if has_all_edges:
                    for x in range(width):
                        if (min_j + x, min_i + height - 1) not in pixel_set:
                            has_all_edges = False
                            break

                # 左辺
                if has_all_edges:
                    for y in range(height):
                        if (min_j, min_i + y) not in pixel_set:
                            has_all_edges = False
                            break

                # 右辺
                if has_all_edges:
                    for y in range(height):
                        if (min_j + width - 1, min_i + y) not in pixel_set:
                            has_all_edges = False
                            break

                # 4辺がすべて存在し、かつ内部にピクセルがない場合
                if has_all_edges:
                    # 内部ピクセルをチェック（外周以外のピクセルが存在しないか）
                    inner_pixels = pixel_set.copy()
                    for y in range(height):
                        inner_pixels.discard((min_j, min_i + y))  # 左端
                        inner_pixels.discard((min_j + width - 1, min_i + y))  # 右端
                    for x in range(width):
                        inner_pixels.discard((min_j + x, min_i))  # 上端
                        inner_pixels.discard((min_j + x, min_i + height - 1))  # 下端

                    if not inner_pixels:  # 内部ピクセルが存在しない
                        return "hollow"

            return "none"
        except Exception:
            return "none"

    def _get_line_type(self, obj: ObjectInfo) -> str:
        """線の種類を判定

        Args:
            obj: オブジェクト

        Returns:
            "X"    - 横線（X方向、すべてのピクセルが同じY座標）
            "Y"    - 縦線（Y方向、すべてのピクセルが同じX座標）
            "XY"   - 45度の対角線（slope = 1.0）
            "-XY"  - 135度の対角線（slope = -1.0）
            "none" - 線ではない
        """
        try:
            if not obj.pixels or len(obj.pixels) < 2:
                return "none"

            # ピクセル座標から判定
            x_coords = [x for x, y in obj.pixels]
            y_coords = [y for x, y in obj.pixels]

            # 横線（X方向、すべてのピクセルが同じY座標）
            if len(set(y_coords)) == 1:
                return "X"

            # 縦線（Y方向、すべてのピクセルが同じX座標）
            if len(set(x_coords)) == 1:
                return "Y"

            # 斜め線のチェック（ExecutorCoreの実装に合わせる）
            # _is_line相当のチェック: すべての点が一直線上にあるか
            if len(obj.pixels) >= 2:
                # 最初の2点から傾きを計算
                x1, y1 = obj.pixels[0][0], obj.pixels[0][1]
                x2, y2 = obj.pixels[1][0], obj.pixels[1][1]

                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)

                    # すべての点が同じ直線上にあるか確認（_is_line相当）
                    is_line = True
                    for x, y in obj.pixels[2:]:
                        expected_y = y1 + slope * (x - x1)
                        if abs(y - expected_y) > 0.5:  # 許容誤差
                            is_line = False
                            break

                    if is_line:
                        # 厳密な45度判定（slope = 1.0）
                        if abs(slope - 1.0) < 0.01:
                            return "XY"
                        # 厳密な135度判定（slope = -1.0）
                        elif abs(slope - (-1.0)) < 0.01:
                            return "-XY"
                        # その他の斜め線は線として認識しない
                        else:
                            return "none"
                else:
                    # x座標が同じ場合は縦線（既に判定されているはず）
                    return "Y"

            return "none"
        except Exception:
            return "none"

    def _get_centroid(self, obj: ObjectInfo) -> str:
        """重心位置方向を判定（ExecutorCoreの実装に合わせる）

        Args:
            obj: オブジェクト

        Returns:
            方向文字列（"C", "X", "Y", "-X", "-Y", "XY", "-XY", "-X-Y", "X-Y"など）
        """
        try:
            if not obj.pixels or obj.size == 0:
                return "C"

            if not obj.bbox or len(obj.bbox) != 4:
                return "C"

            min_i, min_j, max_i, max_j = obj.bbox
            width = max_j - min_j + 1
            height = max_i - min_i + 1

            if width <= 0 or height <= 0:
                return "C"

            # 重心を計算（ピクセルの平均位置）
            sum_x = 0.0
            sum_y = 0.0
            for y, x in obj.pixels:
                sum_x += x
                sum_y += y

            centroid_x = sum_x / obj.size
            centroid_y = sum_y / obj.size

            # バウンディングボックス内で正規化（0.0-1.0）
            local_x = (centroid_x - min_j) / width if width > 0 else 0.5
            local_y = (centroid_y - min_i) / height if height > 0 else 0.5

            # 中心（0.5）からの距離を計算
            # 計算誤差を考慮して、ぴったり中心（0.5）に近い場合のみ"C"と判定
            # 浮動小数点の計算誤差を考慮して、非常に小さな閾値（0.001）を使用
            # 0.499-0.501の範囲を中心と判定（約±0.1%、非常に厳密）
            center_threshold = 0.001
            center_min = 0.5 - center_threshold  # 0.499
            center_max = 0.5 + center_threshold  # 0.501

            # 方向を決定
            dir_x = ""
            dir_y = ""

            if local_x < center_min:
                dir_x = "-X"  # 左に偏り
            elif local_x > center_max:
                dir_x = "X"  # 右に偏り
            # center_min <= local_x <= center_max の場合は dir_x = ""（ぴったり中心）

            if local_y < center_min:
                dir_y = "-Y"  # 上に偏り
            elif local_y > center_max:
                dir_y = "Y"  # 下に偏り
            # center_min <= local_y <= center_max の場合は dir_y = ""（ぴったり中心）

            # 両方ともぴったり中心の場合のみ"C"を返す
            if dir_x == "" and dir_y == "":
                return "C"

            # 方向を結合
            if dir_x and dir_y:
                # 両方の方向がある場合（例: "X-Y", "-XY"）
                return f"{dir_x}{dir_y}" if dir_x.startswith("-") else f"{dir_x}-{dir_y}"
            elif dir_x:
                return dir_x
            elif dir_y:
                return dir_y
            else:
                return "C"
        except Exception:
            return "C"

    def _get_param_command(self, param_name: str) -> str:
        """
        パラメータ名からコマンドを取得

        Args:
            param_name: パラメータ名

        Returns:
            コマンド文字列（例: 'GET_X', 'GET_COLOR'など）
        """
        command_map = {
            'x': 'GET_X',
            'y': 'GET_Y',
            'size': 'GET_SIZE',
            'width': 'GET_WIDTH',
            'height': 'GET_HEIGHT',
            'hole_count': 'COUNT_HOLES',
            'symmetry_x': 'GET_SYMMETRY_SCORE',
            'symmetry_y': 'GET_SYMMETRY_SCORE',
            'rectangle_type': 'GET_RECTANGLE_TYPE',
            'line_type': 'GET_LINE_TYPE',
            # 新規追加コマンド
            'aspect_ratio': 'GET_ASPECT_RATIO',
            'density': 'GET_DENSITY',
            'center_x': 'GET_CENTER_X',
            'center_y': 'GET_CENTER_Y',
            'max_x': 'GET_MAX_X',
            'max_y': 'GET_MAX_Y',
            'centroid': 'GET_CENTROID',  # 重心位置方向
            'direction': 'GET_DIRECTION',
        }
        return command_map.get(param_name, f"GET_{param_name.upper()}")

    def _build_command_expr(self, param_name: str) -> str:
        """
        パラメータ名からコマンド式を構築

        Args:
            param_name: パラメータ名

        Returns:
            コマンド式文字列（例: "GET_SIZE($obj)", "GET_SYMMETRY_SCORE($obj, 'X')"）
        """
        command = self._get_param_command(param_name)
        if command == 'GET_SYMMETRY_SCORE':
            axis = param_name[-1].upper()  # 'X' または 'Y'
            return f"GET_SYMMETRY_SCORE($obj, '{axis}')"
        elif command == 'GET_DIRECTION':
            # GET_DIRECTIONは2つのオブジェクトが必要なので、$objと$obj2を使用
            return f"GET_DIRECTION($obj, $obj2)"
        else:
            return f"{command}($obj)"

    def _generate_region_combinations(
        self, regions: List[Dict[str, Any]], num_regions: int
    ) -> List[List[Dict[str, Any]]]:
        """
        領域の組み合わせを生成（組み合わせ数: num_regions）

        Args:
            regions: 領域のリスト
            num_regions: 組み合わせ数

        Returns:
            領域の組み合わせのリスト
        """
        if num_regions == 1:
            return [[r] for r in regions]
        elif num_regions == 2:
            return list(combinations(regions, 2))
        else:
            return list(combinations(regions, num_regions))

    def _all_objects_covered(
        self, region_combination: List[Dict[str, Any]], target_category: CategoryInfo
    ) -> bool:
        """
        領域の組み合わせで、カテゴリ内のすべてのオブジェクトを含めることができるかチェック

        Args:
            region_combination: 領域の組み合わせ（各領域は 'object_indices' キーを持つ）
            target_category: 対象カテゴリ

        Returns:
            カテゴリ内のすべてのオブジェクトが領域の組み合わせに含まれるかどうか
        """
        covered_object_indices = set()

        # 各領域に含まれるオブジェクトのインデックスを収集
        for region in region_combination:
            covered_object_indices.update(region.get('object_indices', []))

        # 対象カテゴリ内のすべてのオブジェクトのインデックス
        target_object_indices = set(obj.index for obj in target_category.objects)

        # 対象カテゴリ内のすべてのオブジェクトが領域の組み合わせに含まれるかチェック
        return target_object_indices.issubset(covered_object_indices)

    def _create_or_condition_from_regions(
        self, region_combination: List[Dict[str, Any]]
    ) -> str:
        """
        領域の組み合わせからOR条件を生成（DSLプログラムの記述に合わせる）

        Args:
            region_combination: 領域の組み合わせ（各領域は 'condition' キーを持つ）

        Returns:
            OR条件の文字列
        """
        if len(region_combination) == 1:
            return region_combination[0]['condition']
        elif len(region_combination) == 2:
            return f"OR({region_combination[0]['condition']}, {region_combination[1]['condition']})"
        else:
            # 3つ以上の場合は、ネストしたOR条件を作成（AND条件と同じ方式）
            # 例: OR(OR(EQUAL(GET_COLOR($obj), 1), EQUAL(GET_COLOR($obj), 2)), EQUAL(GET_COLOR($obj), 4))
            result = region_combination[0]['condition']
            for region in region_combination[1:]:
                result = f"OR({result}, {region['condition']})"
            return result

    def _generate_or_condition_single_param(
        self, target_category: CategoryInfo, all_categories: List[CategoryInfo],
        target_category_id: int, all_objects: List[Any],
        max_or_region_combinations: int = 2
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        OR条件 - 単一パラメータの領域の組み合わせでカテゴリ固有の条件を生成

        Args:
            target_category: 対象カテゴリ
            all_categories: すべてのカテゴリ
            target_category_id: 対象カテゴリのID
            all_objects: すべてのオブジェクト
            max_or_region_combinations: OR条件の最大領域組み合わせ数（デフォルト: 2）

        Returns:
            (条件文字列, 成功フラグ, デバッグ情報)
        """
        debug_info = {
            'tried_params': [],
            'failure_reason': None
        }

        # 各パラメータで領域を見つける
        param_regions_map = {}
        param_names = ['x', 'y', 'color', 'size', 'width', 'height', 'aspect_ratio', 'density', 'hole_count', 'symmetry_x', 'symmetry_y', 'center_x', 'center_y', 'max_x', 'max_y', 'centroid']

        for param_name in param_names:
            regions = self._find_exclusive_regions(param_name, target_category, all_categories)
            if regions:
                param_regions_map[param_name] = regions

        # 領域数の少ないパラメータ順にソート
        sorted_params = sorted(
            param_regions_map.items(),
            key=lambda x: len(x[1])
        )

        # ループ①: 領域数の少ないパラメータ順にループ
        for param_name, regions in sorted_params:
            param_debug = {
                'param_name': param_name,
                'num_regions': len(regions),
                'tried_combinations': []
            }

            # そのパラメータの領域の組み合わせで全オブジェクトを含めることができるか試す
            # 組み合わせ数は1から max_or_region_combinations まで
            for num_regions in range(1, max_or_region_combinations + 1):
                region_combinations = self._generate_region_combinations(regions, num_regions)

                for combination in region_combinations:
                    if self._all_objects_covered(combination, target_category):
                        # 成功: OR条件で結合
                        or_condition = self._create_or_condition_from_regions(combination)
                        param_debug['tried_combinations'].append({
                            'num_regions': num_regions,
                            'success': True,
                            'condition': or_condition
                        })
                        debug_info['tried_params'].append(param_debug)
                        return or_condition, True, {}

                    param_debug['tried_combinations'].append({
                        'num_regions': num_regions,
                        'success': False
                    })

            debug_info['tried_params'].append(param_debug)

        # すべてのパラメータと組み合わせを試したが失敗
        debug_info['failure_reason'] = 'No unique OR condition found for single parameter'
        return None, False, debug_info

    def _generate_or_condition_multi_param(
        self, target_category: CategoryInfo, all_categories: List[CategoryInfo],
        target_category_id: int, all_objects: List[Any],
        max_or_region_combinations: int = 2
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        OR条件 - 2つのパラメータの領域の組み合わせでカテゴリ固有の条件を生成

        Args:
            target_category: 対象カテゴリ
            all_categories: すべてのカテゴリ
            target_category_id: 対象カテゴリのID
            all_objects: すべてのオブジェクト
            max_or_region_combinations: OR条件の最大領域組み合わせ数（デフォルト: 2）

        Returns:
            (条件文字列, 成功フラグ, デバッグ情報)
        """
        debug_info = {
            'tried_param_combinations': [],
            'failure_reason': None
        }

        # 各パラメータで領域を見つける
        param_regions_map = {}
        param_names = ['x', 'y', 'color', 'size', 'width', 'height', 'aspect_ratio', 'density', 'hole_count', 'symmetry_x', 'symmetry_y', 'center_x', 'center_y', 'max_x', 'max_y', 'centroid']

        for param_name in param_names:
            regions = self._find_exclusive_regions(param_name, target_category, all_categories)
            if regions:
                param_regions_map[param_name] = regions

        # 領域数の少ないパラメータ順にソート
        sorted_params = sorted(
            param_regions_map.items(),
            key=lambda x: len(x[1])
        )

        tried_combinations = set()

        # ループ①: 領域数の少ないパラメータ順にループ
        for param1_name, param1_regions in sorted_params:
            # ループ②: ループ①以外のパラメータで領域数の少ないパラメータ順にループ
            for param2_name, param2_regions in sorted_params:
                if param1_name == param2_name:
                    continue

                # 一度行ったパラメータの組み合わせはスキップ
                combination_key = tuple(sorted([param1_name, param2_name]))
                if combination_key in tried_combinations:
                    continue

                tried_combinations.add(combination_key)

                param_combination_debug = {
                    'param1': param1_name,
                    'param2': param2_name,
                    'num_regions_param1': len(param1_regions),
                    'num_regions_param2': len(param2_regions),
                    'tried_combinations': []
                }

                # ループ①②のパラメータの領域を組み合わせる
                all_regions = param1_regions + param2_regions

                # 組み合わせ数は2から max_or_region_combinations まで
                for num_regions in range(2, max_or_region_combinations + 1):
                    region_combinations = self._generate_region_combinations(all_regions, num_regions)

                    for combination in region_combinations:
                        if self._all_objects_covered(combination, target_category):
                            # 成功: OR条件で結合
                            or_condition = self._create_or_condition_from_regions(combination)
                            param_combination_debug['tried_combinations'].append({
                                'num_regions': num_regions,
                                'success': True,
                                'condition': or_condition
                            })
                            debug_info['tried_param_combinations'].append(param_combination_debug)
                            return or_condition, True, {}

                        param_combination_debug['tried_combinations'].append({
                            'num_regions': num_regions,
                            'success': False
                        })

                debug_info['tried_param_combinations'].append(param_combination_debug)

        # すべてのパラメータの組み合わせを試したが失敗
        debug_info['failure_reason'] = 'No unique OR condition found for multi parameter'
        return None, False, debug_info
