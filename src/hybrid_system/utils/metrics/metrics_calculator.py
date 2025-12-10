"""
メトリクス計算器

各種評価メトリクスを計算
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch


class MetricsCalculator:
    """
    メトリクス計算器
    
    分類、回帰、シーケンスなどの評価メトリクスを計算
    """
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """正解率を計算
        
        Args:
            predictions: 予測値
            targets: 正解値
        
        Returns:
            正解率
        """
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def pixel_accuracy(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
        """ピクセル単位の正解率を計算
        
        Args:
            pred_grid: 予測グリッド [batch, height, width]
            target_grid: 正解グリッド [batch, height, width]
        
        Returns:
            ピクセル正解率
        """
        return MetricsCalculator.accuracy(pred_grid, target_grid)
    
    @staticmethod
    def grid_accuracy(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
        """グリッド単位の正解率を計算
        
        Args:
            pred_grid: 予測グリッド
            target_grid: 正解グリッド
        
        Returns:
            グリッド正解率
        """
        # すべてのピクセルが一致するかチェック
        correct_grids = (pred_grid == target_grid).all(dim=(1, 2))
        return correct_grids.float().mean().item()
    
    @staticmethod
    def token_accuracy(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, ignore_index: int = 0) -> float:
        """トークン単位の正解率を計算
        
        Args:
            pred_tokens: 予測トークン
            target_tokens: 正解トークン
            ignore_index: 無視するインデックス（パディングなど）
        
        Returns:
            トークン正解率
        """
        mask = target_tokens != ignore_index
        if mask.sum().item() == 0:
            return 0.0
        
        correct = (pred_tokens == target_tokens)[mask].sum().item()
        total = mask.sum().item()
        
        return correct / total
    
    @staticmethod
    def sequence_accuracy(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
        """シーケンス単位の正解率を計算
        
        Args:
            pred_tokens: 予測トークン [batch, seq_len]
            target_tokens: 正解トークン [batch, seq_len]
        
        Returns:
            シーケンス正解率
        """
        correct_sequences = (pred_tokens == target_tokens).all(dim=-1)
        return correct_sequences.float().mean().item()
    
    @staticmethod
    def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """平均絶対誤差（MAE）を計算
        
        Args:
            predictions: 予測値
            targets: 正解値
        
        Returns:
            MAE
        """
        return (predictions - targets).abs().mean().item()
    
    @staticmethod
    def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """平均二乗誤差（MSE）を計算
        
        Args:
            predictions: 予測値
            targets: 正解値
        
        Returns:
            MSE
        """
        return ((predictions - targets) ** 2).mean().item()
    
    @staticmethod
    def root_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """平方根平均二乗誤差（RMSE）を計算
        
        Args:
            predictions: 予測値
            targets: 正解値
        
        Returns:
            RMSE
        """
        mse = MetricsCalculator.mean_squared_error(predictions, targets)
        return np.sqrt(mse)
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """パープレキシティを計算
        
        Args:
            loss: クロスエントロピー損失
        
        Returns:
            パープレキシティ
        """
        return np.exp(loss)
    
    @staticmethod
    def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Top-K正解率を計算
        
        Args:
            logits: logits [batch, num_classes]
            targets: 正解ラベル [batch]
            k: Top-K
        
        Returns:
            Top-K正解率
        """
        _, top_k_pred = logits.topk(k, dim=-1)
        correct = (top_k_pred == targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()
    
    @staticmethod
    def compute_all_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metric_types: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """複数のメトリクスを一度に計算
        
        Args:
            predictions: 予測値
            targets: 正解値
            metric_types: メトリクスタイプのリスト
            **kwargs: 追加パラメータ
        
        Returns:
            メトリクス辞書
        """
        metrics = {}
        
        for metric_type in metric_types:
            if metric_type == 'accuracy':
                metrics['accuracy'] = MetricsCalculator.accuracy(predictions, targets)
            elif metric_type == 'pixel_accuracy':
                metrics['pixel_accuracy'] = MetricsCalculator.pixel_accuracy(predictions, targets)
            elif metric_type == 'grid_accuracy':
                metrics['grid_accuracy'] = MetricsCalculator.grid_accuracy(predictions, targets)
            elif metric_type == 'token_accuracy':
                metrics['token_accuracy'] = MetricsCalculator.token_accuracy(
                    predictions, targets, kwargs.get('ignore_index', 0)
                )
            elif metric_type == 'sequence_accuracy':
                metrics['sequence_accuracy'] = MetricsCalculator.sequence_accuracy(predictions, targets)
            elif metric_type == 'mae':
                metrics['mae'] = MetricsCalculator.mean_absolute_error(predictions, targets)
            elif metric_type == 'mse':
                metrics['mse'] = MetricsCalculator.mean_squared_error(predictions, targets)
            elif metric_type == 'rmse':
                metrics['rmse'] = MetricsCalculator.root_mean_squared_error(predictions, targets)
        
        return metrics
