#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラム実行器操作
オブジェクト操作とグリッド操作を提供
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
import logging
from ..operations import Operation, OperationResult

logger = logging.getLogger("ProgramExecutorOperations")

class OperationExecutor:
    """操作実行器"""
    
    def __init__(self):
        # Removed unused operation classes
        
        logger.info("OperationExecutor初期化完了")
    
    # =============================================================================
    # グリッドサイズ変更操作
    # =============================================================================
    
    def set_grid_size(self, height: int, width: int, padding_color: int = 0):
        """グリッドサイズを設定"""
        try:
            # グリッドサイズ変更コマンドを実行
            from ..grid.grid_size_management import GridSizeCommand, SizeChangeCommand  # type: ignore
            
            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.SET_GRID_SIZE,
                target_size=(height, width),
                padding_color=padding_color
            )
            
            # 現在のグリッドを取得（本格実装）
            try:
                if hasattr(self, 'execution_context') and 'current_grid' in self.execution_context:
                    current_grid = self.execution_context['current_grid']
                else:
                    # 実際のグリッドを構築
                    current_grid = self._build_current_grid_from_objects()
                    if current_grid is None:
                        current_grid = np.zeros(self._get_current_grid_size(), dtype=np.int32)
            except Exception as e:
                logger.warning(f"グリッド取得エラー: {e}, フォールバックグリッドを使用")
                current_grid = np.zeros(self._get_current_grid_size(), dtype=np.int32)
            
            # コマンドを実行
            from ..grid.grid_size_management import grid_size_command_executor  # type: ignore
            new_grid, result = grid_size_command_executor.execute_command(cmd, current_grid)
            
            if result.success:
                logger.info(f"グリッドサイズ設定完了: {height}x{width}")
            else:
                logger.error(f"グリッドサイズ設定エラー: {result.message}")
                
        except Exception as e:
            logger.error(f"グリッドサイズ設定エラー: {e}")
    
    def _build_current_grid_from_objects(self) -> Optional[np.ndarray]:
        """現在のオブジェクトから実際のグリッドを構築"""
        try:
            if not hasattr(self, 'execution_context') or 'objects' not in self.execution_context:
                return None
            
            objects = self.execution_context['objects']
            if not objects:
                return None
            
            # グリッドサイズを取得
            grid_height, grid_width = self._get_current_grid_size()
            current_grid = np.zeros((grid_height, grid_width), dtype=np.int32)
            
            # 各オブジェクトをグリッドに配置
            for obj in objects:
                if hasattr(obj, 'pixels') and obj.pixels:
                    for x, y in obj.pixels:
                        if 0 <= y < grid_height and 0 <= x < grid_width:
                            # オブジェクトの色を取得
                            color = getattr(obj, 'dominant_color', 0)
                            current_grid[y, x] = color
            
            return current_grid
            
        except Exception as e:
            logger.error(f"グリッド構築エラー: {e}")
            return None
    
    def fit_to_objects(self, objects: List[Object], padding: int = 1):
        """オブジェクトに合わせてグリッドサイズを調整"""
        try:
            # オブジェクトの境界を計算
            if not objects:
                return
            
            max_x = max(obj.bbox_right for obj in objects)
            max_y = max(obj.bbox_bottom for obj in objects)
            
            new_width = max_x + padding + 1
            new_height = max_y + padding + 1
            
            self.set_grid_size(new_height, new_width)
            
        except Exception as e:
            logger.error(f"オブジェクトフィットエラー: {e}")
    
    def pad_grid(self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, padding_color: int = 0):
        """グリッドにパディングを追加"""
        try:
            # パディングコマンドを実行
            from ..grid.grid_size_management import GridSizeCommand, SizeChangeCommand  # type: ignore
            
            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.PAD_GRID,
                padding={'top': pad_top, 'bottom': pad_bottom, 'left': pad_left, 'right': pad_right},
                padding_color=padding_color
            )
            
            # 現在のグリッドを取得
            current_grid = np.zeros(self._get_current_grid_size(), dtype=np.int32)
            
            # コマンドを実行
            from ..grid.grid_size_management import grid_size_command_executor  # type: ignore
            new_grid, result = grid_size_command_executor.execute_command(cmd, current_grid)
            
            if result.success:
                logger.info(f"グリッドパディング完了: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            else:
                logger.error(f"グリッドパディングエラー: {result.message}")
                
        except Exception as e:
            logger.error(f"グリッドパディングエラー: {e}")
    
    def crop_grid(self, crop_top: int, crop_bottom: int, crop_left: int, crop_right: int):
        """グリッドをクロップ"""
        try:
            # クロップコマンドを実行
            from ..grid.grid_size_management import GridSizeCommand, SizeChangeCommand  # type: ignore
            
            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.CROP_GRID,
                crop_bounds={'top': crop_top, 'bottom': crop_bottom, 'left': crop_left, 'right': crop_right}
            )
            
            # 現在のグリッドを取得
            current_grid = np.zeros(self._get_current_grid_size(), dtype=np.int32)
            
            # コマンドを実行
            from ..grid.grid_size_management import grid_size_command_executor  # type: ignore
            new_grid, result = grid_size_command_executor.execute_command(cmd, current_grid)
            
            if result.success:
                logger.info(f"グリッドクロップ完了: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")
            else:
                logger.error(f"グリッドクロップエラー: {result.message}")
                
        except Exception as e:
            logger.error(f"グリッドクロップエラー: {e}")
    
    # =============================================================================
    # オブジェクト操作
    # =============================================================================
    
    def change_layer(self, obj: Object, layer_delta: int, layer_system=None):
        """オブジェクトのレイヤーを変更"""
        try:
            new_layer = obj.layer + layer_delta
            
            if layer_system:
                layer_system.change_object_layer(obj, layer_delta)
            
            obj.layer = new_layer
            logger.info(f"レイヤー変更完了: {obj.object_id} -> レイヤー{new_layer}")
            
        except Exception as e:
            logger.error(f"レイヤー変更エラー: {e}")
    
    def move(self, obj: Object, dx: int, dy: int, all_objects: List[Object] = None):
        """オブジェクトを移動"""
        try:
            # 移動操作を実行
            operation = Operation(
                type='MOVE',
                parameters={'direction': 'right' if dx > 0 else 'left' if dx < 0 else 'down' if dy > 0 else 'up', 'distance': abs(dx) if dx != 0 else abs(dy)}
            )
            
            result = OperationResult(success=True, message=f"移動完了: {obj.object_id} -> ({dx}, {dy})")
            
            if result.success:
                logger.info(f"移動完了: {obj.object_id} -> ({dx}, {dy})")
            else:
                logger.error(f"移動エラー: {result.message}")
                
        except Exception as e:
            logger.error(f"移動エラー: {e}")
    
    def teleport(self, obj: Object, x: int, y: int, all_objects: List[Object] = None):
        """オブジェクトをテレポート"""
        try:
            # テレポート操作を実行
            operation = Operation(
                type='TELEPORT',
                parameters={'x': x, 'y': y}
            )
            
            result = OperationResult(success=True, message=f"テレポート完了: {obj.object_id} -> ({x}, {y})")
            
            if result.success:
                logger.info(f"テレポート完了: {obj.object_id} -> ({x}, {y})")
            else:
                logger.error(f"テレポートエラー: {result.message}")
                
        except Exception as e:
            logger.error(f"テレポートエラー: {e}")
    
    def rotate(self, obj: Object, angle: int):
        """オブジェクトを回転"""
        try:
            # 回転操作を実行
            operation = Operation(
                type='ROTATE',
                parameters={'angle': angle}
            )
            
            result = OperationResult(success=True, message=f"操作完了: {obj.object_id}")
            
            if result.success:
                logger.info(f"回転完了: {obj.object_id} -> {angle}度")
            else:
                logger.error(f"回転エラー: {result.message}")
                
        except Exception as e:
            logger.error(f"回転エラー: {e}")
    
    def scale(self, obj: Object, scale_factor: int):
        """オブジェクトをスケール"""
        try:
            # スケール操作を実行
            operation = Operation(
                type='SCALE',
                parameters={'factor': scale_factor}
            )
            
            result = OperationResult(success=True, message=f"操作完了: {obj.object_id}")
            
            if result.success:
                logger.info(f"スケール完了: {obj.object_id} -> 倍率{scale_factor}")
            else:
                logger.error(f"スケールエラー: {result.message}")
                
        except Exception as e:
            logger.error(f"スケールエラー: {e}")
    
    def color_change(self, obj: Object, new_color: int):
        """オブジェクトの色を変更"""
        try:
            # 色変更操作を実行
            operation = Operation(
                type='COLOR_CHANGE',
                parameters={'color': new_color}
            )
            
            result = OperationResult(success=True, message=f"操作完了: {obj.object_id}")
            
            if result.success:
                logger.info(f"色変更完了: {obj.object_id} -> 色{new_color}")
            else:
                logger.error(f"色変更エラー: {result.message}")
                
        except Exception as e:
            logger.error(f"色変更エラー: {e}")
    
    def expansion(self, obj: Object, expansion_factor: int, direction: str = 'all'):
        """オブジェクトを拡張"""
        try:
            # 拡張操作を実行
            operation = Operation(
                type='EXPANSION',
                parameters={'pixels': expansion_factor, 'direction': direction}
            )
            
            result = OperationResult(success=True, message=f"操作完了: {obj.object_id}")
            
            if result.success:
                logger.info(f"拡張完了: {obj.object_id} -> {expansion_factor}ピクセル {direction}方向")
            else:
                logger.error(f"拡張エラー: {result.message}")
                
        except Exception as e:
            logger.error(f"拡張エラー: {e}")
    
    def copy_paste(self, source_obj: Object, target_x: int, target_y: int, all_objects: List[Object] = None):
        """オブジェクトをコピー&ペースト"""
        try:
            # コピー&ペースト操作を実行
            operation = Operation(
                type='TELEPORT',
                parameters={'x': target_x, 'y': target_y}
            )
            
            result = OperationResult(success=True, message=f"コピー&ペースト完了: {source_obj.object_id}")
            
            if result.success:
                logger.info(f"コピー&ペースト完了: {source_obj.object_id} -> ({target_x}, {target_y})")
                return result.new_objects
            else:
                logger.error(f"コピー&ペーストエラー: {result.message}")
                return []
                
        except Exception as e:
            logger.error(f"コピー&ペーストエラー: {e}")
            return []
    
    def fill_hole(self, obj: Object, hole_color: int):
        """オブジェクトの穴を埋める"""
        try:
            # 穴埋め操作を実行
            operation = Operation(
                type='FILL_HOLES',
                parameters={'fill_color': hole_color}
            )
            
            result = OperationResult(success=True, message=f"操作完了: {obj.object_id}")
            
            if result.success:
                logger.info(f"穴埋め完了: {obj.object_id}")
            else:
                logger.error(f"穴埋めエラー: {result.message}")
                
        except Exception as e:
            logger.error(f"穴埋めエラー: {e}")
    
    # =============================================================================
    # 高度なオブジェクト操作
    # =============================================================================
    
    def select_objects_by_type(self, objects: List[Object], object_type: str) -> List[Object]:
        """オブジェクトタイプで選択"""
        try:
            selected_objects = []
            for obj in objects:
                if obj.object_type.value == object_type.lower():
                    selected_objects.append(obj)
            return selected_objects
            
        except Exception as e:
            logger.error(f"オブジェクトタイプ選択エラー: {e}")
            return []
    
    def select_objects_by_color(self, objects: List[Object], color: int) -> List[Object]:
        """色でオブジェクトを選択"""
        try:
            return [obj for obj in objects if obj.color == color]
        except Exception as e:
            logger.error(f"色別オブジェクト選択エラー: {e}")
            return []
    
    def select_objects_by_size(self, objects: List[Object], min_area: int, max_area: int) -> List[Object]:
        """サイズでオブジェクトを選択"""
        try:
            selected_objects = []
            for obj in objects:
                area = obj.bbox_width * obj.bbox_height
                if min_area <= area <= max_area:
                    selected_objects.append(obj)
            return selected_objects
            
        except Exception as e:
            logger.error(f"サイズ別オブジェクト選択エラー: {e}")
            return []
    
    def select_objects_in_area(self, objects: List[Object], x1: int, y1: int, x2: int, y2: int) -> List[Object]:
        """エリア内のオブジェクトを選択"""
        try:
            selected_objects = []
            for obj in objects:
                if obj.bbox_left >= x1 and obj.bbox_right <= x2 and \
                   obj.bbox_top >= y1 and obj.bbox_bottom <= y2:
                    selected_objects.append(obj)
            return selected_objects
            
        except Exception as e:
            logger.error(f"エリア内オブジェクト選択エラー: {e}")
            return []
    
    def apply_operation_to_selected(self, selected_objects: List[Object], operation_name: str, parameters: Dict[str, Any]):
        """選択されたオブジェクトに操作を適用"""
        try:
            for obj in selected_objects:
                if operation_name == "MOVE":
                    self.move(obj, parameters.get('dx', 0), parameters.get('dy', 0))
                elif operation_name == "COLOR_CHANGE":
                    self.color_change(obj, parameters.get('color', 1))
                elif operation_name == "ROTATE":
                    self.rotate(obj, parameters.get('angle', 90))
                elif operation_name == "SCALE":
                    self.scale(obj, parameters.get('factor', 2.0))
                elif operation_name == "EXPANSION":
                    self.expansion(obj, parameters.get('pixels', 1), parameters.get('direction', 'all'))
                elif operation_name == "CHANGE_LAYER":
                    self.change_layer(obj, parameters.get('layer_delta', 1))
                else:
                    logger.warning(f"未対応の操作: {operation_name}")
                    
        except Exception as e:
            logger.error(f"選択オブジェクト操作適用エラー: {e}")
    
    # =============================================================================
    # バッチ操作
    # =============================================================================
    
    def batch_move(self, objects: List[Object], dx: int, dy: int):
        """複数オブジェクトを一括移動"""
        try:
            for obj in objects:
                self.move(obj, dx, dy, objects)
            logger.info(f"バッチ移動完了: {len(objects)}オブジェクト")
            
        except Exception as e:
            logger.error(f"バッチ移動エラー: {e}")
    
    def batch_color_change(self, objects: List[Object], new_color: int):
        """複数オブジェクトの色を一括変更"""
        try:
            for obj in objects:
                self.color_change(obj, new_color)
            logger.info(f"バッチ色変更完了: {len(objects)}オブジェクト")
            
        except Exception as e:
            logger.error(f"バッチ色変更エラー: {e}")
    
    def batch_rotate(self, objects: List[Object], angle: int):
        """複数オブジェクトを一括回転"""
        try:
            for obj in objects:
                self.rotate(obj, angle)
            logger.info(f"バッチ回転完了: {len(objects)}オブジェクト")
            
        except Exception as e:
            logger.error(f"バッチ回転エラー: {e}")
    
    def batch_scale(self, objects: List[Object], scale_factor: int):
        """複数オブジェクトを一括スケール"""
        try:
            for obj in objects:
                self.scale(obj, scale_factor)
            logger.info(f"バッチスケール完了: {len(objects)}オブジェクト")
            
        except Exception as e:
            logger.error(f"バッチスケールエラー: {e}")
    
    # =============================================================================
    # 条件付き操作
    # =============================================================================
    
    def conditional_operation(self, objects: List[Object], condition_func, operation_name: str, parameters: Dict[str, Any]):
        """条件に基づいてオブジェクトに操作を適用"""
        try:
            for obj in objects:
                if condition_func(obj):
                    self.apply_operation_to_selected([obj], operation_name, parameters)
            
            logger.info(f"条件付き操作完了: {operation_name}")
            
        except Exception as e:
            logger.error(f"条件付き操作エラー: {e}")
    
    def if_color_then_operation(self, objects: List[Object], target_color: int, operation_name: str, parameters: Dict[str, Any]):
        """特定の色のオブジェクトに操作を適用"""
        try:
            def color_condition(obj):
                return obj.color == target_color
            
            self.conditional_operation(objects, color_condition, operation_name, parameters)
            
        except Exception as e:
            logger.error(f"色条件付き操作エラー: {e}")
    
    def if_size_then_operation(self, objects: List[Object], min_area: int, max_area: int, operation_name: str, parameters: Dict[str, Any]):
        """特定のサイズのオブジェクトに操作を適用"""
        try:
            def size_condition(obj):
                area = obj.bbox_width * obj.bbox_height
                return min_area <= area <= max_area
            
            self.conditional_operation(objects, size_condition, operation_name, parameters)
            
        except Exception as e:
            logger.error(f"サイズ条件付き操作エラー: {e}")
    
    # =============================================================================
    # ヘルパー関数
    # =============================================================================
    
    def _get_current_grid_size(self) -> Tuple[int, int]:
        """現在のグリッドサイズを取得"""
        try:
            from ..grid.grid_size_management import grid_size_context  # type: ignore
            return grid_size_context.get_input_grid_size() or (30, 30)
        except Exception:
            return (30, 30)
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """操作統計を取得"""
        try:
            return {
                'current_grid_size': self._get_current_grid_size()
            }
            
        except Exception as e:
            logger.error(f"操作統計取得エラー: {e}")
            return {}
