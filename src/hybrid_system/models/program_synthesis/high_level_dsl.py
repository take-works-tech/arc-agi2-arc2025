"""
高レベルDSL APIモジュール

低レベルのDSLコマンドを高レベルの意図ベースのAPIに変換
"""

from typing import List, Dict, Optional, Tuple, Any
import re
from dataclasses import dataclass

# Objectクラスは型ヒントのみで使用（実際のインポートは必要に応じて）
try:
    from src.data_systems.data_models.core.object import Object
except ImportError:
    Object = None  # 型ヒントのみの使用を許可


@dataclass
class HighLevelDSLCommand:
    """高レベルDSLコマンド"""
    name: str
    intent: str
    parameters: Dict[str, Any]
    low_level_commands: List[str]  # 変換後の低レベルコマンド


class HighLevelDSLConverter:
    """高レベルDSLコンバーター"""

    def __init__(self):
        """初期化"""
        # 高レベルDSLコマンドの定義
        self.high_level_commands = {
            'align_objects': {
                'intent': 'オブジェクトを整列',
                'parameters': ['target_obj', 'direction'],
                'low_level_mapping': {
                    'left': 'TELEPORT(obj, 0, GET_Y(obj))',
                    'right': 'TELEPORT(obj, GET_INPUT_GRID_SIZE()[1] - GET_WIDTH(obj), GET_Y(obj))',
                    'top': 'TELEPORT(obj, GET_X(obj), 0)',
                    'bottom': 'TELEPORT(obj, GET_X(obj), GET_INPUT_GRID_SIZE()[0] - GET_HEIGHT(obj))',
                    'center_x': 'TELEPORT(obj, (GET_INPUT_GRID_SIZE()[1] - GET_WIDTH(obj)) / 2, GET_Y(obj))',
                    'center_y': 'TELEPORT(obj, GET_X(obj), (GET_INPUT_GRID_SIZE()[0] - GET_HEIGHT(obj)) / 2)',
                }
            },
            'tile_pattern': {
                'intent': 'パターンをタイル状に配置',
                'parameters': ['pattern_id', 'grid_size'],
                'low_level_mapping': None  # 複雑な変換が必要
            },
            'recolor_by_role': {
                'intent': '役割に基づいて色を変更',
                'parameters': ['role', 'new_color'],
                'low_level_mapping': {
                    'background': 'SET_COLOR(obj, new_color) WHERE GET_COLOR(obj) == background_color',
                    'foreground': 'SET_COLOR(obj, new_color) WHERE GET_COLOR(obj) != background_color',
                }
            },
            'mirror_objects': {
                'intent': 'オブジェクトを鏡映',
                'parameters': ['axis'],
                'low_level_mapping': {
                    'X': 'FLIP(obj, "X")',
                    'Y': 'FLIP(obj, "Y")',
                }
            },
            'rotate_objects': {
                'intent': 'オブジェクトを回転',
                'parameters': ['angle'],
                'low_level_mapping': {
                    '90': 'ROTATE(obj, 90)',
                    '180': 'ROTATE(obj, 180)',
                    '270': 'ROTATE(obj, 270)',
                }
            },
            'arrange_in_grid': {
                'intent': 'オブジェクトをグリッド状に配置',
                'parameters': ['rows', 'cols', 'spacing'],
                'low_level_mapping': None  # 複雑な変換が必要
            },
            'group_by_color': {
                'intent': '色でグループ化',
                'parameters': [],
                'low_level_mapping': None  # 複雑な変換が必要
            },
            'extract_pattern': {
                'intent': 'パターンを抽出',
                'parameters': ['pattern_type'],
                'low_level_mapping': None  # 複雑な変換が必要
            },
        }

    def convert_to_high_level(
        self,
        low_level_program: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        低レベルDSLプログラムを高レベルDSLに変換

        Args:
            low_level_program: 低レベルDSLプログラム
            context: コンテキスト情報（オブジェクト、背景色など）

        Returns:
            str: 高レベルDSLプログラム
        """
        # パターンマッチングで高レベルコマンドを検出
        high_level_program = self._detect_high_level_patterns(low_level_program, context)
        return high_level_program

    def _detect_high_level_patterns(
        self,
        program: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """高レベルパターンを検出して変換"""
        lines = program.split('\n')
        converted_lines = []

        for line in lines:
            converted_line = self._convert_line(line, context)
            converted_lines.append(converted_line)

        return '\n'.join(converted_lines)

    def _convert_line(
        self,
        line: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """1行を変換"""
        line = line.strip()
        if not line or line.startswith('#'):
            return line

        # パターン1: 整列パターン
        # TELEPORT(obj, x, y) のパターンを検出
        align_pattern = r'TELEPORT\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)'
        match = re.search(align_pattern, line)
        if match:
            obj, x, y = match.groups()
            # 整列パターンを検出
            if '0' in x or 'GET_INPUT_GRID_SIZE' in x:
                # 左端または右端への配置
                if '0' in x:
                    return f"ALIGN_OBJECTS({obj}, 'left')"
                else:
                    return f"ALIGN_OBJECTS({obj}, 'right')"
            elif '0' in y or 'GET_INPUT_GRID_SIZE' in y:
                # 上端または下端への配置
                if '0' in y:
                    return f"ALIGN_OBJECTS({obj}, 'top')"
                else:
                    return f"ALIGN_OBJECTS({obj}, 'bottom')"

        # パターン2: 鏡映パターン
        # FLIP(obj, "X") または FLIP(obj, "Y")
        flip_pattern = r'FLIP\s*\(\s*(\w+)\s*,\s*["\']([XY])["\']\s*\)'
        match = re.search(flip_pattern, line)
        if match:
            obj, axis = match.groups()
            return f"MIRROR_OBJECTS({obj}, '{axis}')"

        # パターン3: 回転パターン
        # ROTATE(obj, angle)
        rotate_pattern = r'ROTATE\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)'
        match = re.search(rotate_pattern, line)
        if match:
            obj, angle = match.groups()
            return f"ROTATE_OBJECTS({obj}, {angle})"

        # パターン4: 色変更パターン（役割ベース）
        # SET_COLOR(obj, color) WHERE condition
        if 'SET_COLOR' in line and context:
            # 背景色に基づく色変更を検出
            background_color = context.get('background_color', 0)
            if f'GET_COLOR({background_color})' in line or f'== {background_color}' in line:
                return f"RECOLOR_BY_ROLE({obj}, 'background', {color})"

        return line

    def expand_high_level_to_low_level(
        self,
        high_level_program: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        高レベルDSLプログラムを低レベルDSLに展開

        Args:
            high_level_program: 高レベルDSLプログラム
            context: コンテキスト情報

        Returns:
            str: 低レベルDSLプログラム
        """
        lines = high_level_program.split('\n')
        expanded_lines = []

        for line in lines:
            expanded_line = self._expand_line(line, context)
            expanded_lines.append(expanded_line)

        return '\n'.join(expanded_lines)

    def _expand_line(
        self,
        line: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """1行を展開"""
        line = line.strip()
        if not line or line.startswith('#'):
            return line

        # ALIGN_OBJECTS(obj, direction)
        align_pattern = r'ALIGN_OBJECTS\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*\)'
        match = re.search(align_pattern, line)
        if match:
            obj, direction = match.groups()
            cmd_def = self.high_level_commands.get('align_objects')
            if cmd_def and cmd_def['low_level_mapping']:
                low_level = cmd_def['low_level_mapping'].get(direction)
                if low_level:
                    return low_level.replace('obj', obj)

        # MIRROR_OBJECTS(obj, axis)
        mirror_pattern = r'MIRROR_OBJECTS\s*\(\s*(\w+)\s*,\s*["\']([XY])["\']\s*\)'
        match = re.search(mirror_pattern, line)
        if match:
            obj, axis = match.groups()
            cmd_def = self.high_level_commands.get('mirror_objects')
            if cmd_def and cmd_def['low_level_mapping']:
                low_level = cmd_def['low_level_mapping'].get(axis)
                if low_level:
                    return low_level.replace('obj', obj)

        # ROTATE_OBJECTS(obj, angle)
        rotate_pattern = r'ROTATE_OBJECTS\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)'
        match = re.search(rotate_pattern, line)
        if match:
            obj, angle = match.groups()
            cmd_def = self.high_level_commands.get('rotate_objects')
            if cmd_def and cmd_def['low_level_mapping']:
                low_level = cmd_def['low_level_mapping'].get(angle)
                if low_level:
                    return low_level.replace('obj', obj)

        # RECOLOR_BY_ROLE(obj, role, new_color)
        recolor_pattern = r'RECOLOR_BY_ROLE\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*(\d+)\s*\)'
        match = re.search(recolor_pattern, line)
        if match:
            obj, role, new_color = match.groups()
            cmd_def = self.high_level_commands.get('recolor_by_role')
            if cmd_def and cmd_def['low_level_mapping']:
                low_level = cmd_def['low_level_mapping'].get(role)
                if low_level:
                    return low_level.replace('obj', obj).replace('new_color', new_color)

        return line

    def get_high_level_intent(
        self,
        low_level_program: str
    ) -> str:
        """
        低レベルプログラムから高レベル意図を抽出

        Args:
            low_level_program: 低レベルDSLプログラム

        Returns:
            str: 高レベル意図（例: "align_objects", "mirror_objects"）
        """
        # パターンマッチングで意図を検出
        if 'TELEPORT' in low_level_program and ('0' in low_level_program or 'GET_INPUT_GRID_SIZE' in low_level_program):
            return 'align_objects'
        elif 'FLIP' in low_level_program:
            return 'mirror_objects'
        elif 'ROTATE' in low_level_program:
            return 'rotate_objects'
        elif 'SET_COLOR' in low_level_program:
            return 'recolor_by_role'
        elif 'EXTEND_PATTERN' in low_level_program or 'REPEAT' in low_level_program:
            return 'tile_pattern'
        else:
            return 'unknown'


class HighLevelDSLGenerator:
    """高レベルDSL生成器（プログラム生成時に使用）"""

    def __init__(self, converter: Optional[HighLevelDSLConverter] = None):
        """
        初期化

        Args:
            converter: 高レベルDSLコンバーター
        """
        self.converter = converter or HighLevelDSLConverter()

    def generate_high_level_program(
        self,
        intent: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        高レベル意図からプログラムを生成

        Args:
            intent: 高レベル意図（例: "align_objects", "mirror_objects"）
            parameters: パラメータ
            context: コンテキスト情報

        Returns:
            str: 高レベルDSLプログラム
        """
        if intent == 'align_objects':
            target_obj = parameters.get('target_obj', 'obj')
            direction = parameters.get('direction', 'left')
            return f"ALIGN_OBJECTS({target_obj}, '{direction}')"

        elif intent == 'mirror_objects':
            target_obj = parameters.get('target_obj', 'obj')
            axis = parameters.get('axis', 'X')
            return f"MIRROR_OBJECTS({target_obj}, '{axis}')"

        elif intent == 'rotate_objects':
            target_obj = parameters.get('target_obj', 'obj')
            angle = parameters.get('angle', 90)
            return f"ROTATE_OBJECTS({target_obj}, {angle})"

        elif intent == 'recolor_by_role':
            target_obj = parameters.get('target_obj', 'obj')
            role = parameters.get('role', 'foreground')
            new_color = parameters.get('new_color', 1)
            return f"RECOLOR_BY_ROLE({target_obj}, '{role}', {new_color})"

        elif intent == 'tile_pattern':
            pattern_id = parameters.get('pattern_id', 'pattern')
            grid_size = parameters.get('grid_size', (10, 10))
            return f"TILE_PATTERN({pattern_id}, {grid_size})"

        else:
            return ""

    def suggest_high_level_intent(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        objects: Optional[List[Object]] = None
    ) -> List[Tuple[str, float]]:
        """
        入力と出力から高レベル意図を提案

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            objects: オブジェクトリスト（オプション）

        Returns:
            List[Tuple[str, float]]: [(intent, confidence), ...]
        """
        suggestions = []

        # パターン1: 整列
        if self._detect_alignment_pattern(input_grid, output_grid):
            suggestions.append(('align_objects', 0.8))

        # パターン2: 鏡映
        if self._detect_mirror_pattern(input_grid, output_grid):
            suggestions.append(('mirror_objects', 0.9))

        # パターン3: 回転
        if self._detect_rotation_pattern(input_grid, output_grid):
            suggestions.append(('rotate_objects', 0.85))

        # パターン4: 色変更
        if self._detect_recolor_pattern(input_grid, output_grid):
            suggestions.append(('recolor_by_role', 0.7))

        # パターン5: タイル
        if self._detect_tile_pattern(input_grid, output_grid):
            suggestions.append(('tile_pattern', 0.75))

        return suggestions

    def _detect_alignment_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> bool:
        """
        整列パターンを検出（本格実装）

        オブジェクトが端に配置されているか、整列しているかをチェック

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            bool: 整列パターンが検出された場合True
        """
        import numpy as np
        from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
        from src.data_systems.config.config import ExtractionConfig

        try:
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            height, width = input_array.shape

            # オブジェクト抽出器を使用
            extractor = IntegratedObjectExtractor(ExtractionConfig())

            # 入力グリッドからオブジェクトを抽出
            input_result = extractor.extract_objects_by_type(input_array)
            input_objects = []
            for obj_list in input_result.objects_by_type.values():
                input_objects.extend(obj_list)

            # 出力グリッドからオブジェクトを抽出
            output_result = extractor.extract_objects_by_type(output_array)
            output_objects = []
            for obj_list in output_result.objects_by_type.values():
                output_objects.extend(obj_list)

            if not input_objects or not output_objects:
                return False

            # オブジェクトが端に配置されているかチェック
            edge_aligned_count = 0
            for obj in output_objects:
                # 上端、下端、左端、右端に接しているかチェック
                is_top_edge = obj.bbox_top == 0
                is_bottom_edge = obj.bbox_bottom >= height - 1
                is_left_edge = obj.bbox_left == 0
                is_right_edge = obj.bbox_right >= width - 1

                if is_top_edge or is_bottom_edge or is_left_edge or is_right_edge:
                    edge_aligned_count += 1

            # 50%以上のオブジェクトが端に配置されている場合、整列パターンと判定
            if edge_aligned_count / len(output_objects) >= 0.5:
                return True

            # 水平・垂直整列をチェック
            # オブジェクトの中心座標を取得
            centers_x = [obj.center_x for obj in output_objects]
            centers_y = [obj.center_y for obj in output_objects]

            # 水平整列: 同じY座標のオブジェクトが複数ある
            unique_y = set(centers_y)
            horizontal_aligned = any(centers_y.count(y) >= 2 for y in unique_y)

            # 垂直整列: 同じX座標のオブジェクトが複数ある
            unique_x = set(centers_x)
            vertical_aligned = any(centers_x.count(x) >= 2 for x in unique_x)

            return horizontal_aligned or vertical_aligned

        except Exception:
            # エラーが発生した場合はFalseを返す
            return False

    def _detect_mirror_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> bool:
        """
        鏡映パターンを検出（本格実装）

        X軸対称、Y軸対称、回転対称性を検出

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            bool: 鏡映パターンが検出された場合True
        """
        import numpy as np
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        height, width = input_array.shape

        # X軸対称（上下反転）
        if height > 1:
            flipped_x = np.flipud(input_array)
            if np.array_equal(output_array, flipped_x):
                return True

        # Y軸対称（左右反転）
        if width > 1:
            flipped_y = np.fliplr(input_array)
            if np.array_equal(output_array, flipped_y):
                return True

        # 180度回転対称
        if height > 1 and width > 1:
            rotated_180 = np.rot90(input_array, 2)
            if np.array_equal(output_array, rotated_180):
                return True

        # 90度回転対称（正方形の場合のみ）
        if height == width and height > 1:
            rotated_90 = np.rot90(input_array, 1)
            if np.array_equal(output_array, rotated_90):
                return True

            rotated_270 = np.rot90(input_array, 3)
            if np.array_equal(output_array, rotated_270):
                return True

        # 対角線対称（正方形の場合のみ）
        if height == width and height > 1:
            # 主対角線対称
            diagonal_main = input_array.T
            if np.array_equal(output_array, diagonal_main):
                return True

            # 副対角線対称
            diagonal_anti = np.fliplr(np.flipud(input_array)).T
            if np.array_equal(output_array, diagonal_anti):
                return True

        return False

    def _detect_rotation_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> bool:
        """回転パターンを検出"""
        import numpy as np
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)

        # 90度回転
        if np.array_equal(output_array, np.rot90(input_array, 1)):
            return True
        # 180度回転
        if np.array_equal(output_array, np.rot90(input_array, 2)):
            return True
        # 270度回転
        if np.array_equal(output_array, np.rot90(input_array, 3)):
            return True

        return False

    def _detect_recolor_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> bool:
        """色変更パターンを検出"""
        import numpy as np
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)

        # 形状が同じで色だけが異なる場合
        input_binary = (input_array > 0).astype(int)
        output_binary = (output_array > 0).astype(int)

        if np.array_equal(input_binary, output_binary):
            # 色が異なる
            if not np.array_equal(input_array, output_array):
                return True

        return False

    def _detect_tile_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> bool:
        """タイルパターンを検出"""
        import numpy as np
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)

        # 入力が出力の一部として繰り返されているかチェック
        input_h, input_w = input_array.shape
        output_h, output_w = output_array.shape

        if output_h % input_h == 0 and output_w % input_w == 0:
            # タイル状に繰り返されている可能性
            return True

        return False
