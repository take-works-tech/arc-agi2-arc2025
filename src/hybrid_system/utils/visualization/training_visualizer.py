"""訓練可視化"""

import matplotlib.pyplot as plt
from typing import List, Dict


class TrainingVisualizer:
    """訓練可視化クラス"""
    
    @staticmethod
    def plot_loss(train_losses: List[float], val_losses: List[float] = None) -> plt.Figure:
        """損失をプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss')
        if val_losses:
            ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)
        return fig
    
    @staticmethod
    def plot_metrics(metrics_history: Dict[str, List[float]]) -> plt.Figure:
        """メトリクスをプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, values in metrics_history.items():
            ax.plot(values, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics History')
        ax.legend()
        ax.grid(True)
        return fig
