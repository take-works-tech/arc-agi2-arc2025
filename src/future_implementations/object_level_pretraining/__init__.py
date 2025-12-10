"""
Object-level Pretraining モジュール

オブジェクトレベルの事前学習パイプライン
"""

from .object_pretrainer import ObjectLevelPretrainer, ObjectPretrainingConfig

__all__ = [
    'ObjectLevelPretrainer',
    'ObjectPretrainingConfig',
]
