"""
メタデータパッケージ

コマンド・パターン・型の定義を一元管理
"""
from .types import SemanticType, ReturnType, TypeSystem
from .argument_schema import ArgumentSchema
from .constants import (
    GRID_SIZE_PRESERVATION_PROB,
    select_complexity,
    generate_output_grid_size,
)
from .commands import CommandMetadata, COMMAND_METADATA

__all__ = [
    # 型システム
    'SemanticType',
    'ReturnType',
    'TypeSystem',
    # 引数スキーマ
    'ArgumentSchema',
    # 定数
    'GRID_SIZE_PRESERVATION_PROB',
    'select_complexity',
    'generate_output_grid_size',
    # コマンドメタデータ
    'CommandMetadata',
    'COMMAND_METADATA',
]
