"""
Contrastive Pretraining Trainer

対照学習による事前学習トレーナー
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from src.hybrid_system.models.program_synthesis import ProgramSynthesisModel, ObjectBasedProgramSynthesisModel
from src.hybrid_system.models.program_synthesis.grid_encoder import GridEncoder
from src.hybrid_system.models.program_synthesis.object_encoder import ObjectEncoder
from .contrastive_loss import ContrastiveLoss, InfoNCE, SimCLRLoss


@dataclass
class ContrastivePretrainingConfig:
    """対照学習事前学習設定"""
    # モデル設定
    model_type: str = "grid"  # "grid" or "object"
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # 学習設定
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    temperature: float = 0.07
    loss_type: str = "simclr"  # "simclr", "infonce", "contrastive"

    # データ拡張設定
    enable_augmentation: bool = True
    augmentation_prob: float = 0.5

    # その他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_clip: float = 1.0
    save_checkpoint_interval: int = 10


class ContrastivePretrainer:
    """対照学習事前学習トレーナー"""

    def __init__(
        self,
        config: ContrastivePretrainingConfig,
        model: Optional[nn.Module] = None
    ):
        """
        初期化

        Args:
            config: 設定
            model: 事前学習するモデル（Noneの場合は新規作成）
        """
        self.config = config
        self.device = torch.device(config.device)

        # モデルの初期化
        if model is None:
            if config.model_type == "grid":
                self.model = GridEncoder(
                    input_channels=10,
                    embed_dim=config.embed_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    dropout=config.dropout
                )
            elif config.model_type == "object":
                self.model = ObjectEncoder(
                    embed_dim=config.embed_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    dropout=config.dropout
                )
            else:
                raise ValueError(f"Unknown model_type: {config.model_type}")
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # 投影ヘッド（対照学習用）
        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim // 2)
        ).to(self.device)

        # 損失関数
        if config.loss_type == "simclr":
            self.criterion = SimCLRLoss(temperature=config.temperature)
        elif config.loss_type == "infonce":
            self.criterion = InfoNCE(temperature=config.temperature)
        else:
            self.criterion = ContrastiveLoss(temperature=config.temperature)

        # オプティマイザー
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr=config.learning_rate
        )

        # 学習履歴
        self.training_history = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        事前学習を実行

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー（オプション）

        Returns:
            学習結果
        """
        self.model.train()
        self.projection_head.train()

        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            # 訓練
            train_loss = self._train_epoch(train_loader, epoch)

            # 検証
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)

                # ベストモデルの保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, "best")

            # チェックポイントの保存
            if (epoch + 1) % self.config.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch+1}")

            # 履歴の記録
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f if val_loss is not None else 'N/A'}")

        return {
            'status': 'completed',
            'num_epochs': self.config.num_epochs,
            'training_history': self.training_history,
            'best_val_loss': best_val_loss
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """1エポックの訓練"""
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # バッチデータの処理
            if self.config.model_type == "grid":
                # グリッドベースの対照学習
                input_grids = batch['input_grid'].to(self.device)  # [batch, H, W]
                output_grids = batch['output_grid'].to(self.device)  # [batch, H, W]

                # データ拡張（有効な場合）
                if self.config.enable_augmentation:
                    input_grids_aug = self._augment_grid(input_grids)
                    output_grids_aug = self._augment_grid(output_grids)
                else:
                    input_grids_aug = input_grids
                    output_grids_aug = output_grids

                # エンコード
                input_encoded = self.model(input_grids)  # [batch, H*W, dim]
                input_encoded_aug = self.model(input_grids_aug)  # [batch, H*W, dim]

                # プーリング
                input_features = input_encoded.mean(dim=1)  # [batch, dim]
                input_features_aug = input_encoded_aug.mean(dim=1)  # [batch, dim]

                # 投影
                z1 = self.projection_head(input_features)  # [batch, dim//2]
                z2 = self.projection_head(input_features_aug)  # [batch, dim//2]

            else:
                # オブジェクトベースの対照学習
                input_objects = batch['input_objects']
                output_objects = batch['output_objects']

                # エンコード
                input_encoded = self.model(
                    input_objects,
                    batch['input_background_color'].to(self.device),
                    batch['input_grid_width'].to(self.device),
                    batch['input_grid_height'].to(self.device)
                )  # [1, num_objects, dim]

                output_encoded = self.model(
                    output_objects,
                    batch['output_background_color'].to(self.device),
                    batch['output_grid_width'].to(self.device),
                    batch['output_grid_height'].to(self.device)
                )  # [1, num_objects, dim]

                # プーリング
                input_features = input_encoded.mean(dim=1)  # [1, dim]
                output_features = output_encoded.mean(dim=1)  # [1, dim]

                # 投影
                z1 = self.projection_head(input_features)  # [1, dim//2]
                z2 = self.projection_head(output_features)  # [1, dim//2]

            # 損失を計算
            if self.config.loss_type == "simclr":
                loss = self.criterion(z1, z2)
            else:
                # InfoNCEまたはContrastiveLossの場合、負例が必要
                # バッチ内の他のサンプルを負例として使用
                batch_size = z1.size(0)
                if batch_size > 1:
                    # 負例を生成（バッチ内の他のサンプル）
                    neg_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                    negatives = z2[neg_indices].unsqueeze(1)  # [batch, 1, dim//2]
                    loss = self.criterion(z1, z2, negatives)
                else:
                    # バッチサイズが1の場合はSimCLR損失を使用
                    loss = self.criterion(z1, z2)

            # バックプロパゲーション
            self.optimizer.zero_grad()
            loss.backward()

            # 勾配クリッピング
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.projection_head.parameters()),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """1エポックの検証"""
        self.model.eval()
        self.projection_head.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.config.model_type == "grid":
                    input_grids = batch['input_grid'].to(self.device)
                    output_grids = batch['output_grid'].to(self.device)

                    if self.config.enable_augmentation:
                        input_grids_aug = self._augment_grid(input_grids)
                        output_grids_aug = self._augment_grid(output_grids)
                    else:
                        input_grids_aug = input_grids
                        output_grids_aug = output_grids

                    input_encoded = self.model(input_grids)
                    input_encoded_aug = self.model(input_grids_aug)

                    input_features = input_encoded.mean(dim=1)
                    input_features_aug = input_encoded_aug.mean(dim=1)

                    z1 = self.projection_head(input_features)
                    z2 = self.projection_head(input_features_aug)
                else:
                    input_encoded = self.model(
                        batch['input_objects'],
                        batch['input_background_color'].to(self.device),
                        batch['input_grid_width'].to(self.device),
                        batch['input_grid_height'].to(self.device)
                    )
                    output_encoded = self.model(
                        batch['output_objects'],
                        batch['output_background_color'].to(self.device),
                        batch['output_grid_width'].to(self.device),
                        batch['output_grid_height'].to(self.device)
                    )

                    input_features = input_encoded.mean(dim=1)
                    output_features = output_encoded.mean(dim=1)

                    z1 = self.projection_head(input_features)
                    z2 = self.projection_head(output_features)

                if self.config.loss_type == "simclr":
                    loss = self.criterion(z1, z2)
                else:
                    batch_size = z1.size(0)
                    if batch_size > 1:
                        neg_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                        negatives = z2[neg_indices].unsqueeze(1)
                        loss = self.criterion(z1, z2, negatives)
                    else:
                        loss = self.criterion(z1, z2)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        self.projection_head.train()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _augment_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        グリッドにデータ拡張を適用

        注意: Contrastive Pretrainingでは、プログラムが提供されていない
        （ラベルなし）データのみを使用するため、グリッドのみの拡張で問題ありません。
        プログラムが提供されているデータに適用する場合は、プログラムも変換する必要があります。

        Args:
            grid: グリッド [batch, H, W]

        Returns:
            augmented_grid: 拡張されたグリッド [batch, H, W]
        """
        if np.random.random() > self.config.augmentation_prob:
            return grid

        # ランダムな拡張を適用
        # 注意: プログラムが提供されているデータには適用しない（プログラム変換が必要なため）
        augmented = grid.clone()

        # 回転（90度単位）
        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            augmented = torch.rot90(augmented, k, dims=[1, 2])

        # 反転
        if np.random.random() < 0.5:
            if np.random.random() < 0.5:
                augmented = torch.flip(augmented, dims=[1])  # 上下反転
            else:
                augmented = torch.flip(augmented, dims=[2])  # 左右反転

        return augmented

    def _save_checkpoint(self, epoch: int, suffix: str):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        import os
        checkpoint_dir = "checkpoints/contrastive_pretraining"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
