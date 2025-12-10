"""
ベースモデル

すべてのモデルの基底クラス
"""

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from dataclasses import dataclass
import json
import os


@dataclass
class ModelConfig:
    """モデル設定"""
    model_name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 1000  # DSL語彙に最適化（従来: 10000）
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "models/checkpoints"
    log_dir: str = "logs"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """辞書から復元"""
        return cls(**data)

    def save(self, path: str):
        """設定を保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """設定を読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class BaseModel(nn.Module, ABC):
    """
    すべてのモデルの基底クラス

    共通の機能:
    - モデルの保存・読み込み
    - デバイス管理
    - 学習・推論モードの切り替え
    - 統計情報の管理
    """

    def __init__(self, config: ModelConfig):
        """初期化"""
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # 統計情報
        self.training_stats = {
            'total_steps': 0,
            'total_epochs': 0,
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'best_epoch': 0
        }

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """順伝播（サブクラスで実装）"""
        raise NotImplementedError("forward() must be implemented by subclasses")

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """損失計算（サブクラスで実装）"""
        raise NotImplementedError("compute_loss() must be implemented by subclasses")

    def train_step(self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer) -> float:
        """1ステップの訓練

        Args:
            batch: バッチデータ
            optimizer: オプティマイザ

        Returns:
            損失値
        """
        self.train()
        optimizer.zero_grad()

        loss = self.compute_loss(**batch)
        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()

        self.training_stats['total_steps'] += 1
        self.training_stats['total_loss'] += loss.item()

        return loss.item()

    def eval_step(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """1ステップの評価

        Args:
            batch: バッチデータ

        Returns:
            損失値と評価メトリクス
        """
        self.eval()

        with torch.no_grad():
            loss = self.compute_loss(**batch)
            metrics = self.compute_metrics(**batch)

        return loss.item(), metrics

    def compute_metrics(self, *args, **kwargs) -> Dict[str, float]:
        """評価メトリクスの計算（サブクラスでオーバーライド可能）"""
        return {}

    def save_checkpoint(self, path: str, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None):
        """チェックポイントを保存

        Args:
            path: 保存先パス
            epoch: エポック数
            optimizer: オプティマイザ（オプション）
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config.to_dict()
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """チェックポイントを読み込み

        Args:
            path: 読み込み元パス
            optimizer: オプティマイザ（オプション）

        Returns:
            エポック数
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_stats = checkpoint['training_stats']

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']

    def count_parameters(self) -> int:
        """学習可能なパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            'model_name': self.config.model_name,
            'num_parameters': self.count_parameters(),
            'device': str(self.device),
            'training_stats': self.training_stats,
            'config': self.config.to_dict()
        }

    def to_device(self, device: Optional[torch.device] = None):
        """デバイスに移動"""
        if device is None:
            device = self.device
        return self.to(device)

    def freeze(self):
        """モデルのパラメータを凍結"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """モデルのパラメータの凍結を解除"""
        for param in self.parameters():
            param.requires_grad = True

    def reset_stats(self):
        """統計情報をリセット"""
        self.training_stats = {
            'total_steps': 0,
            'total_epochs': 0,
            'total_loss': 0.0,
            'best_loss': float('inf'),
            'best_epoch': 0
        }

    def update_best_model(self, loss: float, epoch: int):
        """最良モデルを更新"""
        if loss < self.training_stats['best_loss']:
            self.training_stats['best_loss'] = loss
            self.training_stats['best_epoch'] = epoch
            return True
        return False

    def get_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """現在の学習率を取得"""
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def set_learning_rate(self, optimizer: torch.optim.Optimizer, lr: float):
        """学習率を設定"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.config.model_name}, "
            f"num_parameters={self.count_parameters():,}, "
            f"device={self.device})"
        )

    def __repr__(self) -> str:
        """表現文字列"""
        return self.__str__()
