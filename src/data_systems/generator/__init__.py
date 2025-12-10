#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラム生成システムパッケージ

統一的プログラム生成システム v2.0（77コマンド対応）
"""

# 実際に存在するモジュールのみをインポート
try:
    from .program_generator import UnifiedProgramGenerator, ProgramContext
except ImportError:
    UnifiedProgramGenerator = None
    ProgramContext = None

__all__ = [
    # プログラム生成
    'UnifiedProgramGenerator',
    'ProgramContext',
]
