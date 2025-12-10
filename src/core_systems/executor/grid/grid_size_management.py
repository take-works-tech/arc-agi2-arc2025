#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グリッドサイズ管理（互換性レイヤー）

Note: このファイルはgrid_manager.pyへの互換性レイヤーです。
実際の実装はgrid_manager.pyにあります。
"""

from .grid_manager import (
    GridSizeContext,
    GridSizeCommand,
    SizeChangeCommand,
    GridSizeCommandExecutor,
    grid_size_context,
    grid_size_command_executor
)

__all__ = [
    'GridSizeContext',
    'GridSizeCommand', 
    'SizeChangeCommand',
    'GridSizeCommandExecutor',
    'grid_size_context',
    'grid_size_command_executor'
]

