"""
完全な訓練ループ実装

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

from src.hybrid_system.models.program_synthesis import ProgramSynthesisModel
from src.hybrid_system.utils.logging import TrainingLogger
from src.hybrid_system.ir.serialization import template_string_to_sequence
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl


class FullPhase1Trainer:
    """
    完全なPhase1訓練器

    DataLoaderを使った標準的な訓練ループ
    """

    def __init__(
        self,
        model: ProgramSynthesisModel,
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
        self.gradient_clip_settings = config.get('gradient_clip', 1.0)
        self.gradient_clip_mode = 'static'
        self.gradient_clip_base = 1.0
        self.gradient_clip_min = 1.0
        self.gradient_clip_max = 1.0
        self.gradient_clip_percentile = 90.0
        self.gradient_clip_history_size = 256
        self.gradient_clip_warmup = 25
        self.gradient_clip_smoothing = 0.5
        self.gradient_clip_norm_type = 2.0
        self.gradient_clip_history: Optional[deque] = None

        if isinstance(self.gradient_clip_settings, dict):
            self.gradient_clip_mode = str(self.gradient_clip_settings.get('mode', 'static')).lower()
            self.gradient_clip_base = float(self.gradient_clip_settings.get('base', self.gradient_clip_settings.get('value', 1.0)))
            self.gradient_clip_min = float(self.gradient_clip_settings.get('min', self.gradient_clip_base))
            self.gradient_clip_max = float(self.gradient_clip_settings.get('max', max(self.gradient_clip_base, self.gradient_clip_min)))
            self.gradient_clip_percentile = float(self.gradient_clip_settings.get('percentile', 90.0))
            self.gradient_clip_history_size = int(self.gradient_clip_settings.get('history_size', 256))
            self.gradient_clip_warmup = int(self.gradient_clip_settings.get('warmup_steps', 25))
            self.gradient_clip_smoothing = float(self.gradient_clip_settings.get('smoothing', 0.5))
            self.gradient_clip_norm_type = float(self.gradient_clip_settings.get('norm_type', 2.0))
            self.gradient_clip_min = max(self.gradient_clip_min, 1e-6)
            self.gradient_clip_max = max(self.gradient_clip_max, self.gradient_clip_min)
            self.gradient_clip_base = min(max(self.gradient_clip_base, self.gradient_clip_min), self.gradient_clip_max)
            self.gradient_clip_history = deque(maxlen=self.gradient_clip_history_size)
            self.gradient_clip_value = self.gradient_clip_base
        else:
            self.gradient_clip_value = max(float(self.gradient_clip_settings), 1e-6)
            self.gradient_clip_base = self.gradient_clip_value
            self.gradient_clip_min = self.gradient_clip_value
            self.gradient_clip_max = self.gradient_clip_value

        self.last_grad_norm: Optional[float] = None
        self.last_clip_value: float = float(self.gradient_clip_value)
        self.last_found_inf_flag: float = 0.0
        self.total_found_inf_events: int = 0

        spike_config = config.get('gradient_spike', {})
        self.grad_spike_enabled = bool(spike_config.get('enable', False))
        self.grad_spike_threshold = float(spike_config.get('grad_norm_threshold', 8.0))
        self.grad_spike_lr_scale = float(spike_config.get('lr_scale', 0.5))
        self.grad_spike_recovery_steps = max(1, int(spike_config.get('recovery_steps', 5)))
        self.grad_spike_cooldown_steps = max(0, int(spike_config.get('cooldown_steps', 20)))
        self.grad_spike_min_lr = float(spike_config.get('min_lr', 0.0))
        self.grad_spike_state = {
            'active': False,
            'remaining_steps': 0,
            'cooldown': 0,
            'original_lrs': None,
            'spike_count': 0
        }
        self.grad_spike_events_total = 0

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

        resume_config = config.get('resume', {})
        self.reset_optimizer_on_resume = bool(resume_config.get('reset_optimizer_state', False))
        self.reset_grad_scaler_on_resume = bool(resume_config.get('reset_grad_scaler', False))
        self.reset_scheduler_on_resume = bool(resume_config.get('reset_scheduler_state', False))

        # TF32設定
        self.allow_tf32 = bool(config.get('allow_tf32', False))
        if self.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True

        validation_monitor_config = config.get('validation_monitor', {})
        self.validation_monitor_enabled = bool(validation_monitor_config.get('enable', False))
        self.validation_monitor_threshold = float(validation_monitor_config.get('loss_increase_threshold', 0.05))
        self._last_validation_loss = None

        # 保存ディレクトリ
        self.save_dir = config.get('save_dir', 'models/checkpoints/phase1')
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_execution_eval_params(self, epoch: int) -> Dict[str, Any]:
        """エポックに応じた実行ベース評価パラメータを取得"""
        validation_config = self.config.get('validation', {})

        default_params = {
            'interval': validation_config.get('execution_eval_interval', 10),
            'samples_per_batch': validation_config.get('execution_eval_samples_per_batch', 2),
            'batch_frequency': validation_config.get('execution_eval_batch_frequency', None)
        }

        schedule = validation_config.get('execution_eval_schedule', [])
        for entry in schedule:
            start = entry.get('epoch_start', 1)
            end = entry.get('epoch_end', None)

            if epoch < start:
                continue
            if end is not None and epoch > end:
                continue

            return {
                'interval': entry.get('interval', default_params['interval']),
                'samples_per_batch': entry.get('samples_per_batch', default_params['samples_per_batch']),
                'batch_frequency': entry.get('batch_frequency', default_params['batch_frequency'])
            }

        return default_params

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        val_loader_own: Optional[DataLoader] = None,
        val_loader_eval: Optional[DataLoader] = None,
        num_epochs: int = 100,
        start_epoch: int = 1
    ) -> Dict[str, Any]:
        """訓練を実行

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー（val_loader_ownとval_loader_evalがNoneの場合に使用）
            val_loader_own: 検証データローダー（自作データ）
            val_loader_eval: 検証データローダー（評価データ）
            num_epochs: エポック数
            start_epoch: 開始エポック（チェックポイントから再開する場合）

        Returns:
            訓練結果
        """
        print(f"\n{'='*60}")
        print(f"Phase1訓練開始")
        if start_epoch > 1:
            print(f"チェックポイントから再開: エポック {start_epoch} から")
        print(f"エポック数: {num_epochs} (開始: {start_epoch})")
        print(f"バッチサイズ: {self.config.get('batch_size', 32)}")
        print(f"学習率: {self.config.get('learning_rate', 1e-4)}")
        print(f"{'='*60}\n")

        last_val_loss = None

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # 訓練
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)

            # 検証（複数の検証データセットに対応）
            val_losses = {}
            val_metrics_all = {}

            # 検証間隔をチェック
            eval_interval = self.config.get('eval_interval', 1)
            should_validate = (epoch % eval_interval == 0) or (epoch == num_epochs)

            if should_validate:
                # 検証設定を取得
                validation_config = self.config.get('validation', {})
                validate_own = validation_config.get('validate_own_data', True)
                validate_eval = validation_config.get('validate_eval_data', True)

                # 自作データの検証
                if val_loader_own and validate_own:
                    val_loss_own, val_metrics_own = self._validate_epoch(val_loader_own, epoch, label="検証（自作データ）")
                    val_losses['own'] = val_loss_own
                    val_metrics_all['own'] = val_metrics_own

                # 評価データの検証
                if val_loader_eval and validate_eval:
                    val_loss_eval, val_metrics_eval = self._validate_epoch(val_loader_eval, epoch, label="検証（評価データ）")
                    val_losses['eval'] = val_loss_eval
                    val_metrics_all['eval'] = val_metrics_eval

                # val_loader_ownとval_loader_evalがNoneの場合、val_loaderを使用
                if val_loader and not val_loader_own and not val_loader_eval:
                    val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
                    val_losses['default'] = val_loss
                    val_metrics_all['default'] = val_metrics
            else:
                # 検証をスキップ（lossベースの評価のみ、前回の値を保持）
                if val_loader_own:
                    val_losses['own'] = None
                if val_loader_eval:
                    val_losses['eval'] = None

            # ログ記録
            if self.logger:
                self.logger.log_epoch(epoch, {'loss': train_loss, **train_metrics}, mode='train')
                for label, val_loss_val in val_losses.items():
                    if val_loss_val is not None:
                        self.logger.log_epoch(epoch, {'loss': val_loss_val, **val_metrics_all.get(label, {})}, mode=f'val_{label}')

            # 学習率調整（評価データの検証lossを優先）
            primary_val_loss = val_losses.get('eval') or val_losses.get('own') or val_losses.get('default')
            self._step_scheduler(primary_val_loss, epoch)

            # Early Stopping & Best Model保存（評価データの検証lossを使用）
            primary_val_loss = val_losses.get('eval') or val_losses.get('own') or val_losses.get('default')
            if primary_val_loss is not None and primary_val_loss < self.best_val_loss:
                self.best_val_loss = primary_val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"[BEST] 新しいベストモデル保存: val_loss={primary_val_loss:.4f}")
            elif primary_val_loss is not None:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n[EARLY STOP] {self.patience}エポック改善なし。訓練を終了します。")
                    break

            # 定期保存
            if epoch % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)

            # 最後の検証損失を保持
            last_val_loss = primary_val_loss

        # 最終結果（最後の検証損失を使用）
        result = {
            'total_epochs': epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': last_val_loss
        }

        print(f"\n{'='*60}")
        print(f"Phase1訓練完了")
        print(f"最良検証損失: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        return result

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """1エポックの訓練"""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        grad_norm_sum = 0.0
        clip_value_sum = 0.0
        grad_norm_count = 0
        epoch_found_inf = 0
        epoch_grad_spike_events = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        num_batches = len(train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            self.last_found_inf_flag = 0.0
            pair_ids: List[str] = batch.get('pair_ids', []) if isinstance(batch, dict) else []

            # データをデバイスに移動
            input_grid = batch['input_grid'].to(self.device)
            output_grid = batch['output_grid'].to(self.device)
            program_tokens = batch['program_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)

            # 順伝播
            with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.model.compute_loss(
                    input_grid, output_grid, program_tokens, target_tokens
                )

            loss_value = loss.detach().item()
            loss_to_optimize = loss / self.gradient_accumulation_steps

            # 逆伝播とパラメータ更新
            if self.use_amp:
                self.grad_scaler.scale(loss_to_optimize).backward()
            else:
                loss_to_optimize.backward()

            should_step = (
                ((step + 1) % self.gradient_accumulation_steps == 0) or
                (step + 1 == num_batches)
            )

            if should_step:
                found_inf_in_step = False
                grad_norm = None

                if self.use_amp:
                    previous_scale = self.grad_scaler.get_scale()
                    self.grad_scaler.unscale_(self.optimizer)

                grad_norm = self._compute_grad_norm()
                clip_value = self._update_gradient_clip(grad_norm)
                if self.grad_spike_enabled:
                    triggered = self._maybe_trigger_grad_spike(
                        grad_norm=grad_norm,
                        epoch=epoch,
                        step=step,
                        pair_ids=pair_ids
                    )
                    if triggered:
                        epoch_grad_spike_events += 1
                try:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=clip_value,
                        error_if_nonfinite=False
                    )
                except TypeError:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=clip_value
                    )

                if grad_norm is None:
                    grad_norm_for_log = float('nan')
                else:
                    grad_norm_for_log = float(grad_norm)

                if math.isnan(grad_norm_for_log) and not self.use_amp:
                    found_inf_in_step = True

                self.last_grad_norm = grad_norm_for_log
                self.last_clip_value = float(clip_value)

                if math.isfinite(grad_norm_for_log):
                    grad_norm_sum += grad_norm_for_log
                    clip_value_sum += float(clip_value)
                    grad_norm_count += 1

                if self.use_amp:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    current_scale = self.grad_scaler.get_scale()
                    if current_scale < previous_scale:
                        found_inf_in_step = True
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self._update_grad_spike_state()

                if found_inf_in_step:
                    self.last_found_inf_flag = 1.0
                    epoch_found_inf += 1
                    self.total_found_inf_events += 1

            # メトリクス計算
            with torch.no_grad():
                metrics = self.model.compute_metrics(
                    input_grid, output_grid, program_tokens, target_tokens
                )
                total_correct += metrics.get('accuracy', 0) * target_tokens.numel()
                total_tokens += target_tokens.numel()

            total_loss += loss_value

            # プログレスバー更新
            postfix = {
                'loss': f"{loss_value:.4f}",
                'acc': f"{metrics.get('accuracy', 0):.4f}"
            }
            if self.last_grad_norm is not None and math.isfinite(self.last_grad_norm):
                postfix['grad'] = f"{self.last_grad_norm:.3f}"
            pbar.set_postfix(postfix)

            metrics['grad_norm'] = float(self.last_grad_norm) if self.last_grad_norm is not None else float('nan')
            metrics['clip_value'] = float(self.last_clip_value)
            metrics['found_inf_flag'] = self.last_found_inf_flag
            metrics['grad_spike_active'] = 1.0 if self.grad_spike_state['active'] else 0.0

            # ステップログ
            if self.logger and step % self.log_interval == 0:
                self.logger.log_step(
                    epoch=epoch,
                    step=step,
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    additional_metrics=metrics
                )

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        if grad_norm_count > 0:
            avg_grad_norm = grad_norm_sum / grad_norm_count
            avg_clip_value = clip_value_sum / grad_norm_count
        else:
            avg_grad_norm = float(self.last_grad_norm) if self.last_grad_norm is not None and math.isfinite(self.last_grad_norm) else 0.0
            avg_clip_value = float(self.last_clip_value) if self.last_clip_value is not None else 0.0

        metrics = {
            'accuracy': avg_accuracy,
            'avg_grad_norm': avg_grad_norm,
            'avg_clip_value': avg_clip_value,
            'found_inf_events': float(epoch_found_inf),
            'grad_spike_events': float(epoch_grad_spike_events)
        }

        print(
            f"Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
            f"AvgGradNorm: {avg_grad_norm:.4f}, Clip: {avg_clip_value:.4f}, "
            f"FoundInf: {epoch_found_inf}, GradSpike: {epoch_grad_spike_events}"
        )

        return avg_loss, metrics

    def _validate_epoch(self, val_loader: DataLoader, epoch: int, label: str = "検証", force_full_eval: bool = False) -> tuple:
        """1エポックの検証

        Args:
            val_loader: 検証データローダー
            epoch: エポック数
            label: 検証ラベル（表示用）
            force_full_eval: Trueの場合、実行ベース評価を強制的に全サンプルで実行（最終評価用）
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # 実行ベースの評価用統計
        execution_stats = {
            'total_samples': 0,
            'programs_generated': 0,
            'programs_executed': 0,
            'execution_successful': 0,
            'grid_matches': 0,
            'grid_exact_matches': 0
        }

        # Executorを初期化（実行ベースの評価用）
        executor = None
        try:
            from src.data_systems.generator.program_executor.core_executor import CoreExecutor
            executor = CoreExecutor()
        except Exception as e:
            print(f"警告: Executorの初期化に失敗しました（実行ベース評価はスキップ）: {e}")

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"{label} Epoch {epoch}")

            for batch_idx, batch in enumerate(pbar):
                # データをデバイスに移動
                input_grid = batch['input_grid'].to(self.device)
                output_grid = batch['output_grid'].to(self.device)
                program_tokens = batch['program_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)

                # pair_idsを取得
                pair_ids: List[str] = batch.get('pair_ids', []) if isinstance(batch, dict) else []

                # 順伝播（損失計算用）
                with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self.model.compute_loss(
                        input_grid, output_grid, program_tokens, target_tokens
                    )

                loss_value = loss.detach().item()

                # メトリクス計算（トークンレベル）
                with autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    metrics = self.model.compute_metrics(
                        input_grid, output_grid, program_tokens, target_tokens
                    )

                total_correct += metrics.get('accuracy', 0) * target_tokens.numel()
                total_tokens += target_tokens.numel()
                total_loss += loss_value

                # 実行ベースの評価（設定に基づいて制御）
                validation_config = self.config.get('validation', {})
                use_execution_eval = validation_config.get('use_execution_eval', True)

                if force_full_eval:
                    # 最終評価モード: 全サンプルで実行ベース評価を実行
                    should_do_execution_eval = use_execution_eval and executor is not None
                    execution_eval_samples = input_grid.size(0)  # 全サンプル
                    execution_eval_batch_freq = 1  # 全バッチ
                else:
                    # 通常の検証モード: 設定またはスケジュールに基づいて制御
                    eval_params = self._get_execution_eval_params(epoch)
                    execution_eval_interval = eval_params['interval']
                    execution_eval_samples = eval_params['samples_per_batch']
                    execution_eval_batch_freq = eval_params['batch_frequency']

                    # 実行ベース評価を実行するかチェック
                    should_do_execution_eval = (
                        use_execution_eval and
                        executor is not None and
                        (epoch % execution_eval_interval == 0 or epoch == self.config.get('num_epochs', 100))
                    )

                    # バッチ頻度を計算（デフォルト: バッチ数の10分の1）
                    if execution_eval_batch_freq is None:
                        execution_eval_batch_freq = max(1, len(val_loader) // 10)

                    # サンプル数を制限（検証時間を短縮）
                    execution_eval_samples = min(input_grid.size(0), execution_eval_samples)

                if should_do_execution_eval and batch_idx % execution_eval_batch_freq == 0:
                    num_samples = execution_eval_samples

                    for i in range(num_samples):
                        execution_stats['total_samples'] += 1

                        # 入力と期待出力を取得
                        input_grid_np = input_grid[i].cpu().numpy()
                        expected_output_np = output_grid[i].cpu().numpy()

                        # モデルでプログラムを生成（beam_search）
                        try:
                            candidates = self.model.beam_search(
                                input_grid=input_grid[i:i+1],
                                output_grid=output_grid[i:i+1],
                                beam_width=3,
                                max_length=self.model.program_config.max_program_length
                            )

                            if candidates and len(candidates) > 0:
                                execution_stats['programs_generated'] += 1

                                # 最良の候補を使用（(tokens, score)のタプル）
                                best_candidate = candidates[0]
                                if isinstance(best_candidate, (list, tuple)) and len(best_candidate) >= 1:
                                    program_tokens_tensor = best_candidate[0]  # torch.Tensor

                                    # Tensorをリストに変換
                                    if isinstance(program_tokens_tensor, torch.Tensor):
                                        program_tokens_generated = program_tokens_tensor.squeeze().cpu().tolist()
                                        if not isinstance(program_tokens_generated, list):
                                            program_tokens_generated = [program_tokens_generated]
                                    else:
                                        program_tokens_generated = program_tokens_tensor

                                    # トークンをプログラム文字列にデコード
                                    if hasattr(val_loader.dataset, 'tokenizer'):
                                        tokenizer = val_loader.dataset.tokenizer
                                    else:
                                        # フォールバック: モデルのtokenizerを使用
                                        from src.hybrid_system.utils.tokenizer import ProgramTokenizer
                                        tokenizer = ProgramTokenizer()

                                    template_sequence_str = tokenizer.decode(program_tokens_generated, skip_special_tokens=True)
                                    template_sequence_str = template_sequence_str.strip()

                                    # 空のテンプレートはスキップ
                                    if not template_sequence_str:
                                        continue

                                    try:
                                        sequence = template_string_to_sequence(
                                            template_sequence_str,
                                            task_id=pair_ids[i] if pair_ids else "",
                                        )
                                        generated_program = sequence_to_dsl(sequence)
                                    except Exception:
                                        continue

                                    if not generated_program.strip():
                                        continue

                                    # プログラムを実行
                                    try:
                                        execution_stats['programs_executed'] += 1

                                        output_grid_result, _, _, _ = executor.execute_program_string(
                                            program_code=generated_program,
                                            input_grid=input_grid_np,
                                            background_color=None
                                        )

                                        if output_grid_result is not None:
                                            execution_stats['execution_successful'] += 1

                                            # グリッドを比較
                                            output_grid_result = output_grid_result.tolist() if hasattr(output_grid_result, 'tolist') else output_grid_result
                                            expected_output = expected_output_np.tolist()

                                            # グリッド一致率を計算
                                            match_score = self._compare_grids(expected_output, output_grid_result)

                                            if match_score >= 1.0:  # 100%一致
                                                execution_stats['grid_exact_matches'] += 1
                                            if match_score > 0.5:  # 50%以上一致
                                                execution_stats['grid_matches'] += 1

                                    except Exception as exec_error:
                                        # 実行エラーは無視（統計のみ）
                                        pass

                        except Exception as gen_error:
                            # 生成エラーは無視（統計のみ）
                            pass

                # プログレスバー更新
                pbar.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'acc': f"{metrics.get('accuracy', 0):.4f}"
                })

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        # 実行ベースのメトリクスを計算
        execution_metrics = {}
        if execution_stats['total_samples'] > 0:
            execution_metrics['program_generation_rate'] = execution_stats['programs_generated'] / execution_stats['total_samples']
            if execution_stats['programs_generated'] > 0:
                execution_metrics['execution_success_rate'] = execution_stats['execution_successful'] / execution_stats['programs_generated']
            else:
                execution_metrics['execution_success_rate'] = 0.0

            if execution_stats['execution_successful'] > 0:
                execution_metrics['grid_exact_match_rate'] = execution_stats['grid_exact_matches'] / execution_stats['execution_successful']
                execution_metrics['grid_match_rate'] = execution_stats['grid_matches'] / execution_stats['execution_successful']
            else:
                execution_metrics['grid_exact_match_rate'] = 0.0
                execution_metrics['grid_match_rate'] = 0.0

        metrics = {
            'accuracy': avg_accuracy,
            **execution_metrics
        }

        print(f"{label} Loss: {avg_loss:.4f}, Token Accuracy: {avg_accuracy:.4f}")
        if execution_stats['total_samples'] > 0:
            print(f"{label} - 実行ベース評価: サンプル数={execution_stats['total_samples']}, "
                  f"生成成功率={execution_metrics.get('program_generation_rate', 0):.2%}, "
                  f"実行成功率={execution_metrics.get('execution_success_rate', 0):.2%}, "
                  f"グリッド完全一致率(100%)={execution_metrics.get('grid_exact_match_rate', 0):.2%}, "
                  f"グリッド一致率(50%+)=>{execution_metrics.get('grid_match_rate', 0):.2%}")

        return avg_loss, metrics

    def _compare_grids(self, expected: list, actual: list) -> float:
        """グリッドの一致度を計算

        Args:
            expected: 期待される出力グリッド
            actual: 実際の出力グリッド

        Returns:
            一致度スコア（0.0-1.0）
        """
        if not expected or not actual:
            return 0.0

        try:
            # グリッドを正規化（サイズを合わせる）
            expected_np = np.array(expected, dtype=int)
            actual_np = np.array(actual, dtype=int)

            # サイズが異なる場合は0.0を返す
            if expected_np.shape != actual_np.shape:
                return 0.0

            # ピクセル単位での一致率を計算
            matches = (expected_np == actual_np)
            match_rate = matches.sum() / matches.size

            return float(match_rate)

        except Exception:
            return 0.0

    def _build_scheduler(self) -> Tuple[Optional[Any], str]:
        """学習率スケジューラーを構築"""
        default_params = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'threshold': 1e-4,
            'threshold_mode': 'rel',
            'cooldown': 0,
            'min_lr': 0.0,
            'eps': 1e-8,
            'verbose': True
        }

        scheduler_config = self.config.get('learning_rate_scheduler')
        if not scheduler_config:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **default_params
            )
            return scheduler, 'metric'

        scheduler_type = str(scheduler_config.get('type', 'ReduceLROnPlateau')).lower()
        if scheduler_type in ('none', 'off', 'disabled'):
            return None, 'none'

        if scheduler_type in ('reduceonplateau', 'reducelronplateau'):
            params = scheduler_config.get('params', {})
            merged_params = default_params.copy()
            merged_params.update(params)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **merged_params
            )
            step_mode = str(scheduler_config.get('step_on', 'metric')).lower()
            if step_mode not in ('metric', 'epoch'):
                step_mode = 'metric'
            return scheduler, step_mode

        print(f"警告: 未対応のスケジューラータイプ '{scheduler_type}' が指定されました。ReduceLROnPlateauを使用します。")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **default_params
        )
        return scheduler, 'metric'

    def _step_scheduler(self, primary_val_loss: Optional[float], epoch: int):
        """スケジューラーのステップ処理"""
        if not self.scheduler:
            return

        if primary_val_loss is not None and self.validation_monitor_enabled:
            if self._last_validation_loss is not None:
                baseline = max(abs(self._last_validation_loss), 1e-12)
                relative_change = (primary_val_loss - self._last_validation_loss) / baseline
                if relative_change > self.validation_monitor_threshold:
                    print(
                        f"[VAL WARN] Epoch {epoch}: val_lossが {self._last_validation_loss:.4f} "
                        f"→ {primary_val_loss:.4f}（+{relative_change * 100:.2f}%）"
                    )
            self._last_validation_loss = primary_val_loss

        if self.scheduler_step_mode == 'metric':
            if primary_val_loss is not None:
                self.scheduler.step(primary_val_loss)
                self._check_learning_rate_update(epoch)
        elif self.scheduler_step_mode == 'epoch':
            self.scheduler.step()
            self._check_learning_rate_update(epoch)

    def _check_learning_rate_update(self, epoch: int):
        """学習率の変化を監視してログを出力"""
        current_lr = self.optimizer.param_groups[0]['lr']
        if not hasattr(self, '_last_logged_lr'):
            self._last_logged_lr = current_lr
            return

        if abs(current_lr - self._last_logged_lr) > 1e-12:
            print(f"[LR] Epoch {epoch}: 学習率更新 -> {current_lr:.6e}")
            if self.logger:
                self.logger.log_epoch(epoch, {'learning_rate': current_lr}, mode='lr')
            self._last_logged_lr = current_lr

    def _maybe_trigger_grad_spike(
        self,
        grad_norm: Optional[float],
        epoch: int,
        step: int,
        pair_ids: Optional[List[str]] = None
    ) -> bool:
        """勾配スパイク検知時の学習率低減処理"""
        if not self.grad_spike_enabled:
            return False

        if grad_norm is None or not math.isfinite(grad_norm):
            return False

        if grad_norm < self.grad_spike_threshold:
            return False

        state = self.grad_spike_state

        if state['active']:
            state['remaining_steps'] = max(state['remaining_steps'], self.grad_spike_recovery_steps)
            state['cooldown'] = self.grad_spike_cooldown_steps
            return False

        if state['cooldown'] > 0:
            return False

        original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        scaled_lrs = []
        for lr in original_lrs:
            scaled_lr = max(lr * self.grad_spike_lr_scale, self.grad_spike_min_lr)
            scaled_lrs.append(scaled_lr)

        for param_group, lr in zip(self.optimizer.param_groups, scaled_lrs):
            param_group['lr'] = lr

        state['active'] = True
        state['remaining_steps'] = self.grad_spike_recovery_steps
        state['cooldown'] = self.grad_spike_cooldown_steps
        state['original_lrs'] = original_lrs
        state['spike_count'] += 1
        self.grad_spike_events_total += 1

        pair_info = ','.join(pair_ids) if pair_ids else 'N/A'
        print(
            f"[GRAD SPIKE] Epoch {epoch} Step {step}: grad_norm={grad_norm:.4f}, "
            f"lr -> {scaled_lrs[0]:.6e} (scale={self.grad_spike_lr_scale}), pairs={pair_info}"
        )

        return True

    def _update_grad_spike_state(self):
        """学習率低減状態を管理"""
        if not self.grad_spike_enabled:
            return

        state = self.grad_spike_state

        if state['active']:
            state['remaining_steps'] -= 1
            if state['remaining_steps'] <= 0:
                if state['original_lrs'] is not None:
                    scheduler_lrs: Optional[List[float]] = None
                    if hasattr(self.scheduler, 'get_last_lr'):
                        try:
                            scheduler_lrs = [float(v) for v in self.scheduler.get_last_lr()]
                        except Exception:
                            scheduler_lrs = None
                    restored_lrs = []
                    for idx, (param_group, original_lr) in enumerate(zip(self.optimizer.param_groups, state['original_lrs'])):
                        target_lr = float(original_lr)
                        if scheduler_lrs is not None and idx < len(scheduler_lrs):
                            target_lr = min(target_lr, scheduler_lrs[idx])
                        param_group['lr'] = target_lr
                        restored_lrs.append(target_lr)
                    if restored_lrs:
                        self._last_logged_lr = restored_lrs[0]
                state['active'] = False
                state['original_lrs'] = None
                print(f"[GRAD SPIKE] 学習率を復元。クールダウン開始 {state['cooldown']} steps")

        if not state['active'] and state['cooldown'] > 0:
            state['cooldown'] -= 1
            if state['cooldown'] < 0:
                state['cooldown'] = 0

    def _compute_grad_norm(self) -> Optional[float]:
        """勾配ノルムを計算"""
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return None

        norm_type = self.gradient_clip_norm_type
        if math.isinf(norm_type):
            max_norm = max(p.grad.data.abs().max().item() for p in parameters)
            return float(max_norm)

        total = 0.0
        for param in parameters:
            param_norm = param.grad.data.norm(norm_type)
            if not torch.isfinite(param_norm):
                return float('nan')
            total += param_norm.item() ** norm_type

        return float(total ** (1.0 / norm_type))

    def _update_gradient_clip(self, grad_norm: Optional[float]) -> float:
        """勾配クリップ閾値を更新"""
        if self.gradient_clip_mode != 'adaptive' or self.gradient_clip_history is None:
            return float(self.gradient_clip_value)

        if grad_norm is None or not math.isfinite(grad_norm):
            return float(self.gradient_clip_value)

        self.gradient_clip_history.append(float(grad_norm))

        if len(self.gradient_clip_history) < max(1, self.gradient_clip_warmup):
            self.gradient_clip_value = self.gradient_clip_base
            return float(self.gradient_clip_value)

        percentile_val = float(np.percentile(self.gradient_clip_history, self.gradient_clip_percentile))
        target_clip = max(self.gradient_clip_min, min(self.gradient_clip_max, percentile_val))

        smoothing = max(0.0, min(1.0, self.gradient_clip_smoothing))
        if smoothing > 0:
            self.gradient_clip_value = smoothing * self.gradient_clip_value + (1.0 - smoothing) * target_clip
        else:
            self.gradient_clip_value = target_clip

        self.gradient_clip_value = max(self.gradient_clip_min, min(self.gradient_clip_max, self.gradient_clip_value))
        return float(self.gradient_clip_value)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """チェックポイントを保存"""
        # エポック単位のチェックポイントパス
        epoch_filename = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:04d}.pt')

        # モデルのチェックポイントを保存（Early Stoppingの状態も含める）
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.model.training_stats,
            'config': self.model.config.to_dict(),
            # Early Stoppingの状態
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            # 学習率スケジューラーの状態（可能な範囲で）
            'scheduler_state': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'grad_scaler_state': self.grad_scaler.state_dict() if (self.use_amp and self.grad_scaler) else None
        }

        os.makedirs(self.save_dir, exist_ok=True)

        # 各エポックのチェックポイント
        torch.save(checkpoint_data, epoch_filename)

        # 最新チェックポイントを更新（常に上書き）
        latest_filename = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint_data, latest_filename)

        # ベストモデルが更新された場合は別ファイルにも保存
        if is_best:
            best_filename = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint_data, best_filename)
            if self.logger:
                self.logger.log_best_model(epoch, {'val_loss': self.best_val_loss})

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """チェックポイントを読み込んで状態を復元

        Args:
            checkpoint_path: チェックポイントファイルのパス

        Returns:
            エポック数
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # モデルの状態を復元（torch.compile で保存された場合は _orig_mod. を除去）
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            stripped_state = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '', 1) if key.startswith('_orig_mod.') else key
                stripped_state[new_key] = value
            state_dict = stripped_state

        self.model.load_state_dict(state_dict, strict=False)
        if 'training_stats' in checkpoint:
            self.model.training_stats = checkpoint['training_stats']

        # オプティマイザーの状態を復元
        if 'optimizer_state_dict' in checkpoint and not getattr(self, 'reset_optimizer_on_resume', False):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # チェックポイント側の学習率よりも現在の設定値を優先して反映
            configured_lr = float(self.config.get('learning_rate', self.optimizer.param_groups[0]['lr']))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = configured_lr
            self._last_logged_lr = configured_lr
        else:
            configured_lr = float(self.config.get('learning_rate', self.optimizer.param_groups[0]['lr']))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = configured_lr
            self._last_logged_lr = configured_lr

        # Early Stoppingの状態を復元
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']

        # 学習率スケジューラーの状態を復元（可能な範囲で）
        if (
            'scheduler_state' in checkpoint
            and checkpoint['scheduler_state'] is not None
            and not getattr(self, 'reset_scheduler_on_resume', False)
        ):
            if hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        elif getattr(self, 'reset_scheduler_on_resume', False) and hasattr(self.scheduler, 'base_lrs'):
            base_lrs = list(self.scheduler.base_lrs)
            for param_group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                param_group['lr'] = base_lr
            self._last_logged_lr = self.optimizer.param_groups[0]['lr']

        if getattr(self, 'reset_grad_scaler_on_resume', False):
            self.grad_scaler = GradScaler(enabled=self.use_amp)
        else:
            grad_scaler_state = checkpoint.get('grad_scaler_state')
            if grad_scaler_state and self.use_amp and self.grad_scaler:
                try:
                    self.grad_scaler.load_state_dict(grad_scaler_state)
                except Exception:
                    self.grad_scaler = GradScaler(enabled=self.use_amp)

        return checkpoint['epoch']
