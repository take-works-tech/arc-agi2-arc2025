"""
オブジェクトマッチングモジュール

ルールベースでオブジェクトの役割分類を行い、深層学習モデルに渡す部分プログラムを生成する機能
"""

from .rule_based_matcher import RuleBasedObjectMatcher
from .config import ObjectMatchingConfig
from .data_structures import ObjectInfo, CategoryInfo, BackgroundColorInfo

__all__ = [
    'RuleBasedObjectMatcher',
    'ObjectMatchingConfig',
    'ObjectInfo',
    'CategoryInfo',
    'BackgroundColorInfo',
]
