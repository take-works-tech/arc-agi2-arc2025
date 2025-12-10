"""
システム統合器

推論システム全体を統合する機能を提供
"""

from .evaluator import ARCEvaluator, ARCEvaluationConfig

__all__ = [
    'ARCEvaluator',
    'ARCEvaluationConfig'
]
