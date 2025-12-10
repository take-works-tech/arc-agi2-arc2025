"""
ObjectBasedProgramSynthesisModel用のTrainerクラス

PyTorch DataLoaderを使った標準的な訓練ループ
"""

import os
import math
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from collections import deque

try:
    from torch.amp import autocast, GradScaler
except ImportError:  # 互換性確保（古いPyTorch向け）
    from torch.cuda.amp import autocast, GradScaler

from src.hybrid_system.models.program_synthesis.object_based_program_synthesis_model import ObjectBasedProgramSynthesisModel
from src.hybrid_system.utils.logging import TrainingLogger
from src.hybrid_system.ir.serialization import template_string_to_sequence
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl


class ObjectBasedTrainer:
    """
    ObjectBasedProgramSynthesisModel用の訓練器

    DataLoaderを使った標準的な訓練ループ
    """

    def __init__(
        self,
        model: ObjectBasedProgramSynthesisModel,
        config: Dict[str, Any],
        logger: Optional[TrainingLogger] = None
    ):
        """初期化"""
        self.model = model
        self.config = config
        self.logger = logger
        self.device = model.device

        # オプティマイザー
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )

        # 学習率スケジューラー
        self.scheduler, self.scheduler_step_mode = self._build_scheduler()
        self._last_logged_lr = self.optimizer.param_groups[0]['lr']

        # Early Stopping
        self.best_val_loss = float('inf')
        self.patience = config.get('early_stopping_patience', 10)
        self.patience_counter = 0

        # 勾配累積
        self.gradient_accumulation_steps = max(1, int(config.get('gradient_accumulation_steps', 1)))
        self.gradient_clip_value = max(float(config.get('gradient_clip', 1.0)), 1e-6)

        # ログ出力間隔
        self.log_interval = max(1, int(config.get('log_interval', 10)))

        # AMP設定
        amp_requested = bool(config.get('use_amp', False))
        device_str = str(self.device)
        self.use_amp = amp_requested and device_str.startswith('cuda') and torch.cuda.is_available()

        amp_dtype_key = str(config.get('amp_dtype', 'float16')).lower()
        if amp_dtype_key in ('bfloat16', 'bf16'):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        self.amp_device_type = 'cuda' if device_str.startswith('cuda') else 'cpu'
        self.grad_scaler = GradScaler(enabled=self.use_amp)

        # 保存ディレクトリ
        self.save_dir = config.get('save_dir', 'models/checkpoints/object_based')
        os.makedirs(self.save_dir, exist_ok=True)

    def _build_scheduler(self) -> Tuple[Optional[Any], Optional[str]]:
        """学習率スケジューラーを構築"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'none').lower()

        if scheduler_type == 'none' or not scheduler_type:
            return None, None

        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
            return scheduler, 'epoch'

        elif scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            return scheduler, 'epoch'

        elif scheduler_type == 'plateau':
            mode = scheduler_config.get('mode', 'min')
            factor = scheduler_config.get('factor', 0.5)
            patience = scheduler_config.get('patience', 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, patience=patience
            )
            return scheduler, 'metric'

        return None, None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        start_epoch: int = 1
    ) -> Dict[str, Any]:
        """訓練を実行

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            start_epoch: 開始エポック（チェックポイントから再開する場合）

        Returns:
            訓練結果の辞書
        """
        self.model.train()

        train_losses = []
        val_losses = []

        for epoch in range(start_epoch, num_epochs + 1):
            # 訓練
            epoch_train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(epoch_train_loss)

            # 検証
            if val_loader is not None:
                epoch_val_loss = self._validate_epoch(val_loader, epoch)
                val_losses.append(epoch_val_loss)

                # Early Stopping
                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.patience_counter = 0
                    # ベストモデルを保存
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        if self.logger:
                            self.logger.info(f"Early stopping at epoch {epoch}")
                        break

                # スケジューラー更新（metricベース）
                if self.scheduler and self.scheduler_step_mode == 'metric':
                    self.scheduler.step(epoch_val_loss)
            else:
                # スケジューラー更新（epochベース）
                if self.scheduler and self.scheduler_step_mode == 'epoch':
                    self.scheduler.step()

            # チェックポイント保存
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # データをデバイスに移動
            input_objects_list = batch['input_objects']
            output_objects_list = batch['output_objects']
            input_bg_color = batch['input_background_color'].to(self.device)
            output_bg_color = batch['output_background_color'].to(self.device)
            input_w = batch['input_grid_width'].to(self.device)
            input_h = batch['input_grid_height'].to(self.device)
            output_w = batch['output_grid_width'].to(self.device)
            output_h = batch['output_grid_height'].to(self.device)
            program_tokens = batch['program_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            batch_size = len(input_objects_list)

            # 勾配をリセット
            self.optimizer.zero_grad()

            # Forward pass（バッチ内の各サンプルに対して損失を計算）
            total_loss = 0.0
            with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                for i in range(batch_size):
                    sample_loss = self.model.compute_loss(
                        input_objects=input_objects_list[i],
                        output_objects=output_objects_list[i],
                        input_background_color=input_bg_color[i].item(),
                        output_background_color=output_bg_color[i].item(),
                        input_grid_width=input_w[i].item(),
                        input_grid_height=input_h[i].item(),
                        output_grid_width=output_w[i].item(),
                        output_grid_height=output_h[i].item(),
                        program_tokens=program_tokens[i:i+1],
                        target_tokens=target_tokens[i:i+1]
                    )
                    total_loss += sample_loss

                # バッチ平均
                loss = total_loss / batch_size

                # 勾配累積で正規化
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # 勾配累積
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 勾配クリッピング
                if self.use_amp:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    self.optimizer.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # ログ出力
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.6f}"
                })

                if self.logger:
                    self.logger.log_metric('train_loss', loss.item() * self.gradient_accumulation_steps, step=epoch * len(train_loader) + batch_idx)
                    if current_lr != self._last_logged_lr:
                        self.logger.log_metric('learning_rate', current_lr, step=epoch * len(train_loader) + batch_idx)
                        self._last_logged_lr = current_lr

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                # データをデバイスに移動
                input_objects_list = batch['input_objects']
                output_objects_list = batch['output_objects']
                input_bg_color = batch['input_background_color'].to(self.device)
                output_bg_color = batch['output_background_color'].to(self.device)
                input_w = batch['input_grid_width'].to(self.device)
                input_h = batch['input_grid_height'].to(self.device)
                output_w = batch['output_grid_width'].to(self.device)
                output_h = batch['output_grid_height'].to(self.device)
                program_tokens = batch['program_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)

                batch_size = len(input_objects_list)

                # Forward pass（バッチ内の各サンプルに対して損失を計算）
                batch_loss = 0.0
                with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    for i in range(batch_size):
                        sample_loss = self.model.compute_loss(
                            input_objects=input_objects_list[i],
                            output_objects=output_objects_list[i],
                            input_background_color=input_bg_color[i].item(),
                            output_background_color=output_bg_color[i].item(),
                            input_grid_width=input_w[i].item(),
                            input_grid_height=input_h[i].item(),
                            output_grid_width=output_w[i].item(),
                            output_grid_height=output_h[i].item(),
                            program_tokens=program_tokens[i:i+1],
                            target_tokens=target_tokens[i:i+1]
                        )
                        batch_loss += sample_loss.item()

                    # バッチ平均
                    loss = batch_loss / batch_size

                total_loss += loss
                num_batches += 1

                pbar.set_postfix({'loss': f"{loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if self.logger:
            self.logger.log_metric('val_loss', avg_loss, step=epoch)

        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint['grad_scaler_state_dict'] = self.grad_scaler.state_dict()

        # 通常のチェックポイント
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # 最新チェックポイント
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # ベストモデル
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """チェックポイントを読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'grad_scaler_state_dict' in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

        return checkpoint['epoch']
