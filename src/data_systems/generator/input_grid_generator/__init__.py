"""
Input Grid Generator - 入力グリッド生成専用モジュール

オブジェクト生成、配置、グリッド構築のみを行う
プログラム実行・検証・修正は program_executor モジュールに分離
"""

# オブジェクト生成系モジュール
try:
    from .builders import (
        CoreObjectBuilder
    )
except ImportError as e:
    print(f"Warning: Failed to import builders: {e}")
    CoreObjectBuilder = None

# 管理系モジュール
try:
    from .managers import (
        CorePlacementManager
    )
except ImportError as e:
    print(f"Warning: Failed to import managers: {e}")
    CorePlacementManager = None

# 生成器系モジュール
# ProgramAwareGeneratorは削除されました（未使用のため）

__all__ = [
    'CoreObjectBuilder',
    'CorePlacementManager'
]
