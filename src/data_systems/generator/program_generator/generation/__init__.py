"""
プログラム生成システム

統一的なプログラム生成システム
"""
from ..metadata.types import SemanticType, ReturnType, TypeInfo

# 共通の型情報定義
OBJECT_ARRAY_TYPE = TypeInfo(
    semantic_type=SemanticType.OBJECT,
    is_array=True,
    return_type=ReturnType.OBJECT
)

OBJECT_TYPE = TypeInfo(
    semantic_type=SemanticType.OBJECT,
    is_array=False,
    return_type=ReturnType.OBJECT
)

from .unified_program_generator import UnifiedProgramGenerator
from .program_context import ProgramContext

__all__ = [
    'UnifiedProgramGenerator',
    'ProgramContext',
    'OBJECT_ARRAY_TYPE',
    'OBJECT_TYPE',
]
