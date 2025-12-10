"""
プログラム生成器パッケージ

統一的なプログラム生成システム
"""

# 実際に存在するモジュールをインポート
try:
    from .generation import UnifiedProgramGenerator, ProgramContext
except ImportError:
    UnifiedProgramGenerator = None
    ProgramContext = None

try:
    from .metadata import COMMAND_METADATA
except ImportError:
    COMMAND_METADATA = None

__all__ = [
    # プログラム生成
    'UnifiedProgramGenerator',
    'ProgramContext',
    # メタデータ
    'COMMAND_METADATA',
]
