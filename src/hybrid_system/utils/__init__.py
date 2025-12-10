"""
ユーティリティモジュール

ロギング、メトリクス、可視化、トークナイザーなどの共通機能を提供
"""

from .logging import Logger, TrainingLogger
from .metrics import MetricsCalculator
# visualizationは遅延インポート（matplotlib依存を回避）
# from .visualization import GridVisualizer, TrainingVisualizer
from .tokenizer import ProgramTokenizer

__all__ = [
    'Logger',
    'TrainingLogger',
    'MetricsCalculator',
    # 'GridVisualizer',  # 遅延インポート
    # 'TrainingVisualizer',  # 遅延インポート
    'ProgramTokenizer'
]

# visualizationモジュールの遅延インポート関数
def get_grid_visualizer():
    """GridVisualizerを遅延インポート"""
    from .visualization import GridVisualizer
    return GridVisualizer

def get_training_visualizer():
    """TrainingVisualizerを遅延インポート"""
    from .visualization import TrainingVisualizer
    return TrainingVisualizer
