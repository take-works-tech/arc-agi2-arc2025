"""
ObjectBasedProgramSynthesisModel用の訓練モジュール
"""

from .trainer import ObjectBasedTrainer
from .dataset import ObjectBasedDataset, collate_fn

__all__ = [
    'ObjectBasedTrainer',
    'ObjectBasedDataset',
    'collate_fn'
]
