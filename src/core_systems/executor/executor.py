#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラム実行器（統合版）
分割されたコンポーネントを統合してプログラムを実行
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
import logging

from src.core_systems.executor.core import ExecutorCore, SilentException
from src.core_systems.executor.execution import ExecutionContext, GlobalFunctionManager, OperationExecutor
from src.core_systems.executor.debug import DebugExecutor, ExecutionSnapshot

logger = logging.getLogger("ProgramExecutor")

class Executor:
    """プログラム実行器（統合版）"""

    def __init__(self):
        # コンポーネントを初期化
        self.core = ExecutorCore()
        self.context = ExecutionContext()
        self.global_functions = GlobalFunctionManager(self.context)
        self.operations = OperationExecutor()

        # コンテキストをコアに設定
        self.core.context = self.context

        # デバッグ実行器を初期化
        self.debug_executor = DebugExecutor(self.core)

        logger.info("ProgramExecutor初期化完了")

    def execute_program(self, program_code: str, input_grid: np.ndarray,
                       input_objects: List[Object] = None,
                       input_image_index: int = 0,
                       background_color: int = None) -> Tuple[np.ndarray, List[Object], float]:
        """プログラムを実行"""
        try:
            # コンテキストを設定
            self.context.set_objects(input_objects or [])

            # グローバル関数を設定
            self.global_functions.setup_global_functions()

            # コア実行器でプログラムを実行
            output_grid, final_objects, execution_time = self.core.execute_program(
                program_code, input_grid, input_objects, input_image_index, background_color
            )

            return output_grid, final_objects, execution_time

        except SilentException:
            # SilentExceptionの場合はそのまま再発生（早期タスク廃棄のため）
            raise
        except Exception as e:
            logger.error(f"プログラム実行エラー: {e}")
            # エラー時は入力グリッドを返す
            return input_grid, input_objects or [], 0.0

    def get_execution_statistics(self) -> Dict[str, Any]:
        """実行統計を取得"""
        try:
            core_stats = self.core.get_execution_statistics()
            operation_stats = self.operations.get_operation_statistics()

            return {
                **core_stats,
                **operation_stats,
                'context_objects': len(self.context.get_objects()),
                'selected_objects': len(self.context.get_selected_objects()),
                'variables_count': len(self.context.context['variables']),
                'arrays_count': len(self.context.context['arrays'])
            }

        except Exception as e:
            logger.error(f"実行統計取得エラー: {e}")
            return {}

    def set_objects(self, objects: List[Object]):
        """オブジェクトを設定"""
        self.context.set_objects(objects)

    def get_objects(self) -> List[Object]:
        """オブジェクトを取得"""
        return self.context.get_objects()

    def set_selected_objects(self, objects: List[Object]):
        """選択されたオブジェクトを設定"""
        self.context.set_selected_objects(objects)

    def get_selected_objects(self) -> List[Object]:
        """選択されたオブジェクトを取得"""
        return self.context.get_selected_objects()

    def set_variable(self, name: str, value: Any):
        """変数を設定"""
        self.context.set_variable(name, value)

    def get_variable(self, name: str) -> Any:
        """変数を取得"""
        return self.context.get_variable(name)

    def set_array(self, name: str, array: List[Any]):
        """配列を設定"""
        self.context.set_array(name, array)

    def get_array(self, name: str) -> List[Any]:
        """配列を取得"""
        return self.context.get_array(name)

    # =============================================================================
    # 操作の直接実行
    # =============================================================================

    def execute_move(self, objects: List[Object], dx: int, dy: int):
        """移動操作を実行"""
        self.operations.batch_move(objects, dx, dy)

    def execute_color_change(self, objects: List[Object], new_color: int):
        """色変更操作を実行"""
        self.operations.batch_color_change(objects, new_color)

    def execute_rotate(self, objects: List[Object], angle: int):
        """回転操作を実行"""
        self.operations.batch_rotate(objects, angle)

    def execute_scale(self, objects: List[Object], scale_factor: int):
        """スケール操作を実行"""
        self.operations.batch_scale(objects, scale_factor)

    def execute_expansion(self, objects: List[Object], pixels: int, direction: str = 'all'):
        """拡張操作を実行"""
        for obj in objects:
            self.operations.expansion(obj, pixels, direction)

    def execute_fill(self, objects: List[Object], fill_color: int):
        """塗りつぶし操作を実行"""
        for obj in objects:
            self.operations.fill(obj, fill_color)

    def execute_copy_paste(self, source_objects: List[Object], target_x: int, target_y: int):
        """コピー&ペースト操作を実行"""
        new_objects = []
        for source_obj in source_objects:
            copied_objects = self.operations.copy_paste(source_obj, target_x, target_y, self.get_objects())
            new_objects.extend(copied_objects)

        # 新しいオブジェクトを追加
        if new_objects:
            current_objects = self.get_objects()
            current_objects.extend(new_objects)
            self.set_objects(current_objects)

    def execute_conditional_operation(self, condition_type: str, condition_params: Dict[str, Any],
                                    operation_name: str, operation_params: Dict[str,Any]):
        """条件付き操作を実行"""
        objects = self.get_objects()

        if condition_type == "color":
            self.operations.if_color_then_operation(
                objects, condition_params['color'], operation_name, operation_params
            )
        elif condition_type == "size":
            self.operations.if_size_then_operation(
                objects, condition_params['min_area'], condition_params['max_area'],
                operation_name, operation_params
            )
        else:
            logger.warning(f"未対応の条件タイプ: {condition_type}")

    # =============================================================================
    # グリッドサイズ操作
    # =============================================================================

    def execute_set_grid_size(self, height: int, width: int, padding_color: int = 0):
        """グリッドサイズ設定を実行"""
        self.operations.set_grid_size(height, width, padding_color)

    def execute_fit_to_objects(self, padding: int = 1):
        """オブジェクトに合わせてグリッドサイズを調整"""
        self.operations.fit_to_objects(self.get_objects(), padding)

    def execute_pad_grid(self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, padding_color: int = 0):
        """グリッドパディングを実行"""
        self.operations.pad_grid(pad_top, pad_bottom, pad_left, pad_right, padding_color)

    def execute_crop_grid(self, crop_top: int, crop_bottom: int, crop_left: int, crop_right: int):
        """グリッドクロップを実行"""
        self.operations.crop_grid(crop_top, crop_bottom, crop_left, crop_right)

    # =============================================================================
    # 選択操作
    # =============================================================================

    def select_objects_by_type(self, object_type: str):
        """オブジェクトタイプで選択"""
        selected = self.operations.select_objects_by_type(self.get_objects(), object_type)
        self.set_selected_objects(selected)
        return selected

    def select_objects_by_color(self, color: int):
        """色でオブジェクトを選択"""
        selected = self.operations.select_objects_by_color(self.get_objects(), color)
        self.set_selected_objects(selected)
        return selected

    def select_objects_by_size(self, min_area: int, max_area: int):
        """サイズでオブジェクトを選択"""
        selected = self.operations.select_objects_by_size(self.get_objects(), min_area, max_area)
        self.set_selected_objects(selected)
        return selected

    def select_objects_in_area(self, x1: int, y1: int, x2: int, y2: int):
        """エリア内のオブジェクトを選択"""
        selected = self.operations.select_objects_in_area(self.get_objects(), x1, y1, x2, y2)
        self.set_selected_objects(selected)
        return selected

    def apply_operation_to_selected(self, operation_name: str, parameters: Dict[str, Any]):
        """選択されたオブジェクトに操作を適用"""
        selected_objects = self.get_selected_objects()
        self.operations.apply_operation_to_selected(selected_objects, operation_name, parameters)

    # =============================================================================
    # デバッグ・情報取得
    # =============================================================================

    def get_context_info(self) -> Dict[str, Any]:
        """コンテキスト情報を取得"""
        try:
            return {
                'total_objects': len(self.get_objects()),
                'selected_objects': len(self.get_selected_objects()),
                'variables': dict(self.context.context['variables']),
                'arrays': {name: len(array) for name, array in self.context.context['arrays'].items()},
                'execution_stack_depth': len(self.context.context['execution_stack'])
            }

        except Exception as e:
            logger.error(f"コンテキスト情報取得エラー: {e}")
            return {}

    def reset_context(self):
        """コンテキストをリセット"""
        try:
            self.context = ExecutionContext()
            self.global_functions = GlobalFunctionManager(self.context)
            self.core.context = self.context

            logger.info("コンテキストリセット完了")

        except Exception as e:
            logger.error(f"コンテキストリセットエラー: {e}")

    def validate_execution(self) -> List[str]:
        """実行の妥当性をチェック"""
        errors = []

        try:
            # オブジェクトの妥当性チェック
            for obj in self.get_objects():
                if not obj.pixels:
                    errors.append(f"オブジェクト {obj.object_id} にピクセルがありません")

                if hasattr(obj, 'layer') and obj.layer < 0:
                    errors.append(f"オブジェクト {obj.object_id} のレイヤーが負の値です: {obj.layer}")

            # 選択されたオブジェクトの妥当性チェック
            selected = self.get_selected_objects()
            all_objects = self.get_objects()

            for obj in selected:
                if obj not in all_objects:
                    errors.append(f"選択されたオブジェクト {obj.object_id} がオブジェクトリストに存在しません")

            # 変数の妥当性チェック
            for var_name, var_value in self.context.context['variables'].items():
                if var_name.startswith('_'):
                    errors.append(f"変数名 {var_name} は内部使用のため使用できません")

        except Exception as e:
            errors.append(f"妥当性チェックエラー: {e}")

        return errors

    # =============================================================================
    # デバッグ実行機能
    # =============================================================================

    def execute_program_until_line(
        self,
        program_code: str,
        input_grid: np.ndarray,
        max_line: int,
        input_objects: List[Object] = None,
        input_image_index: int = 0,
        background_color: int = 0,
        create_snapshots: bool = True
    ) -> Tuple[np.ndarray, List[Object], float, List[ExecutionSnapshot]]:
        """指定した行数までプログラムを実行

        Args:
            program_code: プログラムコード
            input_grid: 入力グリッド
            max_line: 実行を停止する行数（0から始まる）
            input_objects: 入力オブジェクト
            input_image_index: 入力画像インデックス
            background_color: 背景色
            create_snapshots: スナップショットを作成するか

        Returns:
            (出力グリッド, オブジェクトリスト, 実行時間, スナップショットリスト)
        """
        return self.debug_executor.execute_program_until_line(
            program_code=program_code,
            input_grid=input_grid,
            max_line=max_line,
            input_objects=input_objects,
            input_image_index=input_image_index,
            background_color=background_color,
            create_snapshots=create_snapshots
        )

    def get_state_at_line(
        self,
        program_code: str,
        input_grid: np.ndarray,
        line_number: int,
        input_objects: List[Object] = None,
        input_image_index: int = 0,
        background_color: int = 0
    ) -> Optional[ExecutionSnapshot]:
        """指定した行数の状態を取得

        Args:
            program_code: プログラムコード
            input_grid: 入力グリッド
            line_number: 取得したい行数（0から始まる）
            input_objects: 入力オブジェクト
            input_image_index: 入力画像インデックス
            background_color: 背景色

        Returns:
            スナップショット（該当する行がない場合はNone）
        """
        return self.debug_executor.get_state_at_line(
            program_code=program_code,
            input_grid=input_grid,
            line_number=line_number,
            input_objects=input_objects,
            input_image_index=input_image_index,
            background_color=background_color
        )
