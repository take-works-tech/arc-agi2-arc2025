"""
候補生成器モジュール

各候補生成方法を別々のファイルに分割
"""

from .neural_generator import NeuralCandidateGenerator
from .neural_object_generator import NeuralObjectCandidateGenerator

__all__ = [
    'NeuralCandidateGenerator',
    'NeuralObjectCandidateGenerator',
]
