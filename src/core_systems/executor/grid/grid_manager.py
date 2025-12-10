#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グリッドサイズ管理システム
グリッドサイズの変更、コンテキスト管理、コマンド実行を統合
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
from src.data_systems.data_models.core.object import Object

logger = logging.getLogger("GridSizeManagement")

# =============================================================================
# データ構造
# =============================================================================

@dataclass
class SizeChangeRecord:
    """サイズ変更記録"""
    operation: str  # 'INPUT', 'UPDATE', 'EXPAND', 'SHRINK', 'SET', 'PAD', 'CROP'
    old_size: Optional[Tuple[int, int]]
    new_size: Tuple[int, int]
    timestamp: float
    parameters: Optional[Dict[str, Any]] = None

class SizeChangeCommand(Enum):
    """サイズ変更コマンド"""
    SET_GRID_SIZE = "SET_GRID_SIZE"
    FIT_TO_OBJECTS = "FIT_TO_OBJECTS"

class SizeChangeType(Enum):
    """サイズ変更タイプ"""
    EXPAND = "expand"
    SHRINK = "shrink"
    RESHAPE = "reshape"
    MAINTAIN = "maintain"

@dataclass
class GridSizeCommand:
    """グリッドサイズ変更コマンド"""
    command_type: SizeChangeCommand
    target_size: Optional[Tuple[int, int]] = None
    padding: Optional[Dict[str, int]] = None  # {'top': 1, 'bottom': 1, 'left': 1, 'right': 1}
    crop_bounds: Optional[Dict[str, int]] = None  # {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    padding_color: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SizeChangeOperation:
    """サイズ変更操作"""
    operation_type: SizeChangeType
    target_size: Optional[Tuple[int, int]]
    padding: Optional[Dict[str, int]]
    crop_bounds: Optional[Dict[str, int]]
    padding_color: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SizeChangeResult:
    """サイズ変更結果"""
    success: bool
    old_size: Tuple[int, int]
    new_size: Tuple[int, int]
    operation_applied: SizeChangeOperation
    execution_time: float
    message: str = ""

# =============================================================================
# グリッドサイズコンテキスト
# =============================================================================

class GridSizeContext:
    """グリッドサイズ管理コンテキスト"""
    
    def __init__(self):
        self.input_grid_size: Optional[Tuple[int, int]] = None
        self._initialized = False
        
        logger.info("GridSizeContext初期化完了")
    
    def initialize(self, input_size: Tuple[int, int]) -> None:
        """初期化（入力サイズを設定）"""
        self.input_grid_size = input_size
        self._initialized = True
        
        logger.info(f"GridSizeContext初期化: 入力サイズ={input_size}")
    
    def get_input_grid_size(self) -> Optional[Tuple[int, int]]:
        """入力グリッドサイズを取得"""
        return self.input_grid_size
    
    def is_initialized(self) -> bool:
        """初期化されているかチェック"""
        return self._initialized

# グローバルインスタンス
grid_size_context = GridSizeContext()

# =============================================================================
# グリッドサイズ変更操作
# =============================================================================

class SizeChangeOperations:
    """サイズ変更操作システム"""
    
    def __init__(self):
        logger.info("SizeChangeOperations初期化完了")
    
    def apply_size_change(self, input_grid: np.ndarray, 
                         operation: SizeChangeOperation) -> SizeChangeResult:
        """サイズ変更を適用"""
        start_time = time.time()
        old_size = input_grid.shape
        
        try:
            if operation.operation_type == SizeChangeType.EXPAND:
                result_grid = self._expand_grid(input_grid, operation)
            elif operation.operation_type == SizeChangeType.SHRINK:
                result_grid = self._shrink_grid(input_grid, operation)
            elif operation.operation_type == SizeChangeType.RESHAPE:
                result_grid = self._reshape_grid(input_grid, operation)
            else:
                result_grid = input_grid.copy()
            
            new_size = result_grid.shape
            execution_time = time.time() - start_time
            
            return SizeChangeResult(
                success=True,
                old_size=old_size,
                new_size=new_size,
                operation_applied=operation,
                execution_time=execution_time,
                message="サイズ変更成功"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"サイズ変更エラー: {e}")
            
            return SizeChangeResult(
                success=False,
                old_size=old_size,
                new_size=old_size,
                operation_applied=operation,
                execution_time=execution_time,
                message=f"サイズ変更エラー: {e}"
            )
    
    def _expand_grid(self, input_grid: np.ndarray, operation: SizeChangeOperation) -> np.ndarray:
        """グリッドを拡張"""
        target_size = operation.target_size
        if not target_size:
            return input_grid.copy()
        
        height, width = input_grid.shape
        target_height, target_width = target_size
        
        if target_height < height or target_width < width:
            logger.warning("拡張サイズが現在のサイズより小さいです")
            return input_grid.copy()
        
        # 新しいグリッドを作成
        expanded_grid = np.full(target_size, operation.padding_color, dtype=input_grid.dtype)
        
        # 元のグリッドを左上に配置
        expanded_grid[:height, :width] = input_grid
        
        return expanded_grid
    
    def _shrink_grid(self, input_grid: np.ndarray, operation: SizeChangeOperation) -> np.ndarray:
        """グリッドを縮小"""
        target_size = operation.target_size
        if not target_size:
            return input_grid.copy()
        
        height, width = input_grid.shape
        target_height, target_width = target_size
        
        if target_height > height or target_width > width:
            logger.warning("縮小サイズが現在のサイズより大きいです")
            return input_grid.copy()
        
        # 指定されたサイズでクロップ
        return input_grid[:target_height, :target_width].copy()
    
    def _reshape_grid(self, input_grid: np.ndarray, operation: SizeChangeOperation) -> np.ndarray:
        """グリッドをリシェイプ"""
        target_size = operation.target_size
        if not target_size:
            return input_grid.copy()
        
        # リシェイプ（データの再配置は行わない）
        return input_grid.reshape(target_size).copy()
    
    @staticmethod
    def create_operation(operation_type: SizeChangeType, 
                        target_size: Optional[Tuple[int, int]] = None,
                        padding: Optional[Dict[str, int]] = None,
                        crop_bounds: Optional[Dict[str, int]] = None,
                        padding_color: int = 0,
                        **kwargs) -> SizeChangeOperation:
        """サイズ変更操作を作成"""
        return SizeChangeOperation(
            operation_type=operation_type,
            target_size=target_size,
            padding=padding,
            crop_bounds=crop_bounds,
            padding_color=padding_color,
            parameters=kwargs
        )

# =============================================================================
# グリッドサイズコマンド実行
# =============================================================================

class GridSizeCommandExecutor:
    """グリッドサイズコマンド実行器"""
    
    def __init__(self):
        self.size_change_operations = SizeChangeOperations()
        logger.info("GridSizeCommandExecutor初期化完了")
    
    def execute_command(self, command: GridSizeCommand, 
                       input_grid: np.ndarray,
                       objects: List[Object] = None) -> Tuple[np.ndarray, SizeChangeResult]:
        """コマンドを実行"""
        try:
            # コマンドタイプに応じて操作を作成
            if command.command_type == SizeChangeCommand.SET_GRID_SIZE:
                operation = SizeChangeOperation(
                    operation_type=SizeChangeType.RESHAPE,
                    target_size=command.target_size,
                    parameters=command.parameters
                )
            elif command.command_type == SizeChangeCommand.FIT_TO_OBJECTS:
                if objects:
                    target_size = self._calculate_fit_size(objects)
                    operation = SizeChangeOperation(
                        operation_type=SizeChangeType.EXPAND,
                        target_size=target_size,
                        parameters=command.parameters
                    )
                else:
                    operation = SizeChangeOperation(
                        operation_type=SizeChangeType.MAINTAIN,
                        parameters=command.parameters
                    )
            else:
                operation = SizeChangeOperation(
                    operation_type=SizeChangeType.MAINTAIN,
                    parameters=command.parameters
                )
            
            # 操作を実行
            result = self.size_change_operations.apply_size_change(input_grid, operation)
            
            # コンテキストを更新
            if result.success:
                grid_size_context.update_size(
                    result.new_size, 
                    command.command_type.value,
                    command.parameters
                )
            
            return result.new_size if result.success else input_grid, result
            
        except Exception as e:
            logger.error(f"コマンド実行エラー: {e}")
            return input_grid, SizeChangeResult(
                success=False,
                old_size=input_grid.shape,
                new_size=input_grid.shape,
                operation_applied=SizeChangeOperation(
                    operation_type=SizeChangeType.MAINTAIN
                ),
                execution_time=0.0,
                message=f"コマンド実行エラー: {e}"
            )
    
    def _calculate_fit_size(self, objects: List[Object]) -> Tuple[int, int]:
        """オブジェクトに合わせたサイズを計算"""
        if not objects:
            return (10, 10)  # デフォルトサイズ
        
        max_x = max(obj.bbox_right for obj in objects if hasattr(obj, 'bbox_right'))
        max_y = max(obj.bbox_bottom for obj in objects if hasattr(obj, 'bbox_bottom'))
        
        # 少し余裕を持たせる
        return (max_y + 2, max_x + 2)

# =============================================================================
# グリッドサイズコマンド生成
# =============================================================================

class GridSizeCommandGenerator:
    """グリッドサイズコマンド生成器"""
    
    def __init__(self):
        logger.info("GridSizeCommandGenerator初期化完了")
    
    def generate_size_change_commands(self, current_size: Tuple[int, int],
                                    target_size: Tuple[int, int],
                                    objects: List[Any]) -> List[GridSizeCommand]:
        """サイズ変更コマンドを生成"""
        commands = []
        
        # サイズ変更が必要な場合
        if current_size != target_size:
            commands.append(GridSizeCommand(
                command_type=SizeChangeCommand.SET_GRID_SIZE,
                target_size=target_size
            ))
        
        # オブジェクトに合わせた調整
        if objects:
            commands.append(GridSizeCommand(
                command_type=SizeChangeCommand.FIT_TO_OBJECTS,
                parameters={'object_count': len(objects)}
            ))
        
        return commands
    
    def generate_padding_commands(self, padding_config: Dict[str, Any]) -> List[GridSizeCommand]:
        """パディングコマンドを生成"""
        commands = []
        
        if 'uniform' in padding_config:
            padding = padding_config['uniform']
            commands.append(GridSizeCommand(
                command_type=SizeChangeCommand.PAD_GRID,
                padding={
                    'top': padding,
                    'bottom': padding,
                    'left': padding,
                    'right': padding
                },
                padding_color=padding_config.get('color', 0)
            ))
        
        if 'asymmetric' in padding_config:
            padding = padding_config['asymmetric']
            commands.append(GridSizeCommand(
                command_type=SizeChangeCommand.PAD_GRID,
                padding=padding,
                padding_color=padding_config.get('color', 0)
            ))
        
        return commands
    
    def generate_crop_commands(self, crop_config: Dict[str, Any]) -> List[GridSizeCommand]:
        """クロップコマンドを生成"""
        commands = []
        
        if 'bounds' in crop_config:
            commands.append(GridSizeCommand(
                command_type=SizeChangeCommand.CROP_GRID,
                crop_bounds=crop_config['bounds']
            ))
        
        return commands

# =============================================================================
# グローバルインスタンス
# =============================================================================

# グローバルインスタンス
size_change_operations = SizeChangeOperations()
grid_size_command_executor = GridSizeCommandExecutor()
grid_size_command_generator = GridSizeCommandGenerator()

# =============================================================================
# テスト用のメイン関数
# =============================================================================

if __name__ == "__main__":
    # テスト用のグリッド
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # グリッドサイズコンテキストを初期化
    grid_size_context.initialize(test_grid.shape)
    
    # サイズ変更操作をテスト
    operation = SizeChangeOperation(
        operation_type=SizeChangeType.EXPAND,
        target_size=(5, 5),
        padding_color=0
    )
    
    result = size_change_operations.apply_size_change(test_grid, operation)
    
    print(f"元のサイズ: {test_grid.shape}")
    print(f"新しいサイズ: {result.new_size}")
    print(f"成功: {result.success}")
    print(f"実行時間: {result.execution_time:.4f}秒")
    
    # コマンド実行をテスト
    command = GridSizeCommand(
        command_type=SizeChangeCommand.SET_GRID_SIZE,
        target_size=(4, 4)
    )
    
    result_grid, command_result = grid_size_command_executor.execute_command(command, test_grid)
    
    print(f"\nコマンド実行結果:")
    print(f"結果サイズ: {result_grid.shape}")
    print(f"成功: {command_result.success}")
    print(f"実行時間: {command_result.execution_time:.4f}秒")
