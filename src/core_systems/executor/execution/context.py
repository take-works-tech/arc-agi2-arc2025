#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラム実行器コンテキスト
実行コンテキスト管理とグローバル関数を提供
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
import logging

logger = logging.getLogger("ProgramExecutorContext")

class ExecutionContext:
    """実行コンテキスト管理"""
    
    def __init__(self):
        self.context = {
            'objects': [],
            'input_image_index': 0,
            'background_color': 0,
            'selected_objects': [],
            'variables': {},
            'arrays': {},
            'execution_stack': []
        }
        
        logger.info("ExecutionContext初期化完了")
    
    def set_objects(self, objects: List[Object]):
        """オブジェクトを設定"""
        self.context['objects'] = objects
    
    def get_objects(self) -> List[Object]:
        """オブジェクトを取得"""
        return self.context['objects']
    
    def set_selected_objects(self, objects: List[Object]):
        """選択されたオブジェクトを設定"""
        self.context['selected_objects'] = objects
    
    def get_selected_objects(self) -> List[Object]:
        """選択されたオブジェクトを取得"""
        return self.context['selected_objects']
    
    def set_variable(self, name: str, value: Any):
        """変数を設定"""
        self.context['variables'][name] = value
    
    def get_variable(self, name: str) -> Any:
        """変数を取得"""
        return self.context['variables'].get(name)
    
    def set_array(self, name: str, array: List[Any]):
        """配列を設定"""
        self.context['arrays'][name] = array
    
    def get_array(self, name: str) -> List[Any]:
        """配列を取得"""
        return self.context['arrays'].get(name, [])
    
    def push_execution_stack(self, frame: Dict[str, Any]):
        """実行スタックにプッシュ"""
        self.context['execution_stack'].append(frame)
    
    def pop_execution_stack(self) -> Dict[str, Any]:
        """実行スタックからポップ"""
        return self.context['execution_stack'].pop() if self.context['execution_stack'] else {}
    
    def get_current_frame(self) -> Dict[str, Any]:
        """現在のフレームを取得"""
        return self.context['execution_stack'][-1] if self.context['execution_stack'] else {}

class GlobalFunctionManager:
    """グローバル関数管理"""
    
    def __init__(self, execution_context: ExecutionContext):
        self.context = execution_context
        
    def setup_global_functions(self):
        """グローバル関数を設定"""
        try:
            import builtins
            
            # グリッド情報取得関数
            builtins.get_input_grid_size = self._get_input_grid_size
            builtins.get_current_grid_size = self._get_current_grid_size
            builtins.get_grid_size_difference = self._get_grid_size_difference
            builtins.get_grid_size_ratio = self._get_grid_size_ratio
            builtins.reset_grid_size = self._reset_grid_size
            builtins.get_grid_size_statistics = self._get_grid_size_statistics
            
            # オブジェクト操作関数
            builtins.get_objects_by_type = self._get_objects_by_type
            builtins.get_object_count = self._get_object_count
            builtins.get_objects_in_area = self._get_objects_in_area
            builtins.get_objects_by_color = self._get_objects_by_color
            builtins.get_objects_by_size = self._get_objects_by_size
            builtins.get_nearest_object = self._get_nearest_object
            builtins.get_objects_at_position = self._get_objects_at_position
            
            # ピクセル・色情報関数
            builtins.get_pixel_color = self._get_pixel_color
            builtins.get_color_count = self._get_color_count
            builtins.get_unique_colors = self._get_unique_colors
            builtins.get_color_distribution = self._get_color_distribution
            builtins.get_grid_statistics = self._get_grid_statistics
            
            # 空間関係関数
            builtins.get_distance = self._get_distance
            builtins.get_angle = self._get_angle
            builtins.is_overlapping = self._is_overlapping
            builtins.is_adjacent = self._is_adjacent
            builtins.get_spatial_relationships = self._get_spatial_relationships
            
            # 変数・配列操作関数
            builtins.set_var = self._set_variable
            builtins.get_var = self._get_variable
            builtins.create_array = self._create_array
            builtins.get_array = self._get_array
            builtins.append_to_array = self._append_to_array
            builtins.remove_from_array = self._remove_from_array
            builtins.get_array_length = self._get_array_length
            
            # 型チェック関数
            builtins.is_int = self._is_int
            builtins.is_bool = self._is_bool
            builtins.is_object = self._is_object
            builtins.to_int = self._to_int
            builtins.to_bool = self._to_bool
            
            # 算術演算関数
            builtins.add = self._add
            builtins.subtract = self._subtract
            builtins.multiply = self._multiply
            builtins.divide = self._divide
            builtins.modulo = self._modulo
            
            # 比較演算関数
            builtins.equal = self._equal
            builtins.not_equal = self._not_equal
            builtins.less_than = self._less_than
            builtins.greater_than = self._greater_than
            builtins.less_equal = self._less_equal
            builtins.greater_equal = self._greater_equal
            
            # 論理演算関数
            builtins.and_op = self._and_op
            builtins.or_op = self._or_op
            builtins.not_op = self._not_op
            
            logger.info("グローバル関数設定完了")
            
        except Exception as e:
            logger.error(f"グローバル関数設定エラー: {e}")
    
    # =============================================================================
    # グリッド情報取得関数
    # =============================================================================
    
    def _get_input_grid_size(self) -> Tuple[int, int]:
        """入力グリッドサイズを取得"""
        try:
            from ..grid.grid_size_management import grid_size_context
            return grid_size_context.get_input_size() or (30, 30)
        except Exception:
            return (30, 30)
    
    def _get_current_grid_size(self) -> Tuple[int, int]:
        """現在のグリッドサイズを取得"""
        try:
            from ..grid.grid_size_management import grid_size_context
            return grid_size_context.get_input_grid_size() or (30, 30)
        except Exception:
            return (30, 30)
    
    def _get_grid_size_difference(self) -> Tuple[int, int]:
        """グリッドサイズの差を取得"""
        current = self._get_current_grid_size()
        input_size = self._get_input_grid_size()
        return (current[0] - input_size[0], current[1] - input_size[1])
    
    def _get_grid_size_ratio(self) -> Tuple[int, int]:
        """グリッドサイズの比率を取得（整数ベース：100=1.0倍）"""
        current = self._get_current_grid_size()
        input_size = self._get_input_grid_size()
        height_ratio = int((current[0] / input_size[0]) * 100) if input_size[0] > 0 else 100
        width_ratio = int((current[1] / input_size[1]) * 100) if input_size[1] > 0 else 100
        return (height_ratio, width_ratio)
    
    def _reset_grid_size(self):
        """グリッドサイズをリセット"""
        try:
            from ..grid.grid_size_management import grid_size_context
            grid_size_context.reset_to_input_size()
        except Exception as e:
            logger.error(f"グリッドサイズリセットエラー: {e}")
    
    def _get_grid_size_statistics(self) -> Dict[str, Any]:
        """グリッドサイズ統計を取得"""
        return {
            'input_size': self._get_input_grid_size(),
            'current_size': self._get_current_grid_size(),
            'difference': self._get_grid_size_difference(),
            'ratio': self._get_grid_size_ratio()
        }
    
    # =============================================================================
    # オブジェクト操作関数
    # =============================================================================
    
    def _get_objects_by_type(self, image_index: int, object_type: str) -> List[Object]:
        """オブジェクトタイプでオブジェクトを取得"""
        try:
            objects = []
            for obj in self.context.get_objects():
                if obj.object_type.value == object_type.lower():
                    objects.append(obj)
            return objects
            
        except Exception as e:
            logger.error(f"オブジェクトタイプ取得エラー: {e}")
            return []
    
    def _get_object_count(self) -> int:
        """オブジェクト数を取得"""
        return len(self.context.get_objects())
    
    def _get_objects_in_area(self, x1: int, y1: int, x2: int, y2: int) -> List[Object]:
        """指定エリア内のオブジェクトを取得"""
        try:
            objects = []
            for obj in self.context.get_objects():
                if obj.bbox_left >= x1 and obj.bbox_right <= x2 and \
                   obj.bbox_top >= y1 and obj.bbox_bottom <= y2:
                    objects.append(obj)
            return objects
            
        except Exception as e:
            logger.error(f"エリア内オブジェクト取得エラー: {e}")
            return []
    
    def _get_objects_by_color(self, color: int) -> List[Object]:
        """色でオブジェクトを取得"""
        return [obj for obj in self.context.get_objects() if obj.color == color]
    
    def _get_objects_by_size(self, min_area: int, max_area: int) -> List[Object]:
        """サイズでオブジェクトを取得"""
        try:
            objects = []
            for obj in self.context.get_objects():
                area = obj.bbox_width * obj.bbox_height
                if min_area <= area <= max_area:
                    objects.append(obj)
            return objects
            
        except Exception as e:
            logger.error(f"サイズ別オブジェクト取得エラー: {e}")
            return []
    
    def _get_nearest_object(self, x: int, y: int) -> Optional[Object]:
        """最も近いオブジェクトを取得"""
        try:
            objects = self.context.get_objects()
            if not objects:
                return None
            
            nearest_obj = None
            min_distance = float('inf')
            
            for obj in objects:
                center_x = (obj.bbox_left + obj.bbox_right) / 2
                center_y = (obj.bbox_top + obj.bbox_bottom) / 2
                
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_obj = obj
            
            return nearest_obj
            
        except Exception as e:
            logger.error(f"最近オブジェクト取得エラー: {e}")
            return None
    
    def _get_objects_at_position(self, x: int, y: int) -> List[Object]:
        """指定位置のオブジェクトを取得"""
        try:
            objects = []
            for obj in self.context.get_objects():
                if (x, y) in obj.pixels:
                    objects.append(obj)
            return objects
            
        except Exception as e:
            logger.error(f"位置別オブジェクト取得エラー: {e}")
            return []
    
    # =============================================================================
    # ピクセル・色情報関数
    # =============================================================================
    
    def _get_pixel_color(self, x: int, y: int) -> int:
        """ピクセルの色を取得"""
        try:
            objects = self._get_objects_at_position(x, y)
            if objects:
                # 最も前面のオブジェクトの色を返す
                frontmost = max(objects, key=lambda obj: obj.layer)
                return frontmost.color
            else:
                # 背景色を返す
                frame = self.context.get_current_frame()
                return frame.get('background_color', 0)
                
        except Exception as e:
            logger.error(f"ピクセル色取得エラー: {e}")
            return 0
    
    def _get_color_count(self, color: int) -> int:
        """色の数を取得"""
        return sum(1 for obj in self.context.get_objects() if obj.color == color)
    
    def _get_unique_colors(self) -> List[int]:
        """ユニークな色のリストを取得"""
        colors = set(obj.color for obj in self.context.get_objects())
        frame = self.context.get_current_frame()
        colors.add(frame.get('background_color', 0))
        return list(colors)
    
    def _get_color_distribution(self) -> Dict[int, int]:
        """色の分布を取得"""
        distribution = {}
        for obj in self.context.get_objects():
            distribution[obj.color] = distribution.get(obj.color, 0) + len(obj.pixels)
        
        # 背景色も追加
        frame = self.context.get_current_frame()
        background_color = frame.get('background_color', 0)
        distribution[background_color] = distribution.get(background_color, 0)
        
        return distribution
    
    def _get_grid_statistics(self) -> Dict[str, Any]:
        """グリッド統計を取得"""
        return {
            'size': self._get_current_grid_size(),
            'object_count': self._get_object_count(),
            'color_distribution': self._get_color_distribution(),
            'unique_colors': self._get_unique_colors()
        }
    
    # =============================================================================
    # 空間関係関数
    # =============================================================================
    
    def _get_distance(self, obj1: Object, obj2: Object) -> float:
        """オブジェクト間の距離を取得"""
        try:
            center1_x = (obj1.bbox_left + obj1.bbox_right) / 2
            center1_y = (obj1.bbox_top + obj1.bbox_bottom) / 2
            center2_x = (obj2.bbox_left + obj2.bbox_right) / 2
            center2_y = (obj2.bbox_top + obj2.bbox_bottom) / 2
            
            return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
            
        except Exception as e:
            logger.error(f"距離計算エラー: {e}")
            return 0.0
    
    def _get_angle(self, obj1: Object, obj2: Object) -> float:
        """オブジェクト間の角度を取得"""
        try:
            import math
            
            center1_x = (obj1.bbox_left + obj1.bbox_right) / 2
            center1_y = (obj1.bbox_top + obj1.bbox_bottom) / 2
            center2_x = (obj2.bbox_left + obj2.bbox_right) / 2
            center2_y = (obj2.bbox_top + obj2.bbox_bottom) / 2
            
            dx = center2_x - center1_x
            dy = center2_y - center1_y
            
            angle = math.atan2(dy, dx)
            return math.degrees(angle)
            
        except Exception as e:
            logger.error(f"角度計算エラー: {e}")
            return 0.0
    
    def _is_overlapping(self, obj1: Object, obj2: Object) -> bool:
        """オブジェクトが重複しているかチェック"""
        try:
            pixels1 = set(obj1.pixels)
            pixels2 = set(obj2.pixels)
            return bool(pixels1 & pixels2)
            
        except Exception as e:
            logger.error(f"重複チェックエラー: {e}")
            return False
    
    def _is_adjacent(self, obj1: Object, obj2: Object) -> bool:
        """オブジェクトが隣接しているかチェック"""
        try:
            # バウンディングボックスで隣接チェック
            return (obj1.bbox_right + 1 == obj2.bbox_left or 
                    obj2.bbox_right + 1 == obj1.bbox_left or
                    obj1.bbox_bottom + 1 == obj2.bbox_top or 
                    obj2.bbox_bottom + 1 == obj1.bbox_top)
            
        except Exception as e:
            logger.error(f"隣接チェックエラー: {e}")
            return False
    
    def _get_spatial_relationships(self) -> Dict[str, List[Tuple[str, str]]]:
        """空間関係を取得"""
        try:
            relationships = {
                'overlapping': [],
                'adjacent': [],
                'distant': []
            }
            
            objects = self.context.get_objects()
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    if self._is_overlapping(obj1, obj2):
                        relationships['overlapping'].append((obj1.object_id, obj2.object_id))
                    elif self._is_adjacent(obj1, obj2):
                        relationships['adjacent'].append((obj1.object_id, obj2.object_id))
                    else:
                        relationships['distant'].append((obj1.object_id, obj2.object_id))
            
            return relationships
            
        except Exception as e:
            logger.error(f"空間関係取得エラー: {e}")
            return {'overlapping': [], 'adjacent': [], 'distant': []}
    
    # =============================================================================
    # 変数・配列操作関数
    # =============================================================================
    
    def _set_variable(self, name: str, value: Any):
        """変数を設定"""
        self.context.set_variable(name, value)
    
    def _get_variable(self, name: str) -> Any:
        """変数を取得"""
        return self.context.get_variable(name)
    
    def _create_array(self, name: str, initial_values: List[Any] = None):
        """配列を作成"""
        self.context.set_array(name, initial_values or [])
    
    def _get_array(self, name: str) -> List[Any]:
        """配列を取得"""
        return self.context.get_array(name)
    
    def _append_to_array(self, name: str, value: Any):
        """配列に要素を追加"""
        array = self.context.get_array(name)
        array.append(value)
        self.context.set_array(name, array)
    
    def _remove_from_array(self, name: str, value: Any):
        """配列から要素を削除"""
        array = self.context.get_array(name)
        if value in array:
            array.remove(value)
            self.context.set_array(name, array)
    
    def _get_array_length(self, name: str) -> int:
        """配列の長さを取得"""
        return len(self.context.get_array(name))
    
    # =============================================================================
    # 型チェック関数
    # =============================================================================
    
    def _is_int(self, value: Any) -> bool:
        """値が整数かどうかチェック"""
        return isinstance(value, int)
    
    def _is_bool(self, value: Any) -> bool:
        """値がブール値かどうかチェック"""
        return isinstance(value, bool)
    
    def _is_object(self, value: Any) -> bool:
        """値がオブジェクトかどうかチェック"""
        return isinstance(value, Object)
    
    def _to_int(self, value: Any) -> int:
        """値を整数に変換"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _to_bool(self, value: Any) -> bool:
        """値をブール値に変換"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return value != 0
        elif isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'on']
        else:
            return bool(value)
    
    # =============================================================================
    # 算術演算関数
    # =============================================================================
    
    def _add(self, a: Any, b: Any) -> Any:
        """加算"""
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a + b
            elif isinstance(a, str) or isinstance(b, str):
                return str(a) + str(b)
            else:
                return self._to_int(a) + self._to_int(b)
        except Exception:
            return 0
    
    def _subtract(self, a: Any, b: Any) -> Any:
        """減算"""
        try:
            return self._to_int(a) - self._to_int(b)
        except Exception:
            return 0
    
    def _multiply(self, a: Any, b: Any) -> Any:
        """乗算"""
        try:
            return self._to_int(a) * self._to_int(b)
        except Exception:
            return 0
    
    def _divide(self, a: Any, b: Any) -> Any:
        """除算"""
        try:
            b_val = self._to_int(b)
            if b_val == 0:
                return 0
            return self._to_int(a) // b_val
        except Exception:
            return 0
    
    def _modulo(self, a: Any, b: Any) -> Any:
        """剰余"""
        try:
            b_val = self._to_int(b)
            if b_val == 0:
                return 0
            return self._to_int(a) % b_val
        except Exception:
            return 0
    
    # =============================================================================
    # 比較演算関数
    # =============================================================================
    
    def _equal(self, a: Any, b: Any) -> bool:
        """等価比較"""
        return a == b
    
    def _not_equal(self, a: Any, b: Any) -> bool:
        """不等価比較"""
        return a != b
    
    def _less_than(self, a: Any, b: Any) -> bool:
        """小なり比較"""
        try:
            return self._to_int(a) < self._to_int(b)
        except Exception:
            return False
    
    def _greater_than(self, a: Any, b: Any) -> bool:
        """大なり比較"""
        try:
            return self._to_int(a) > self._to_int(b)
        except Exception:
            return False
    
    def _less_equal(self, a: Any, b: Any) -> bool:
        """小なりイコール比較"""
        try:
            return self._to_int(a) <= self._to_int(b)
        except Exception:
            return False
    
    def _greater_equal(self, a: Any, b: Any) -> bool:
        """大なりイコール比較"""
        try:
            return self._to_int(a) >= self._to_int(b)
        except Exception:
            return False
    
    # =============================================================================
    # 論理演算関数
    # =============================================================================
    
    def _and_op(self, a: Any, b: Any) -> bool:
        """論理AND"""
        return self._to_bool(a) and self._to_bool(b)
    
    def _or_op(self, a: Any, b: Any) -> bool:
        """論理OR"""
        return self._to_bool(a) or self._to_bool(b)
    
    def _not_op(self, a: Any) -> bool:
        """論理NOT"""
        return not self._to_bool(a)
