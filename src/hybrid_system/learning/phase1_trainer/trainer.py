"""
フェーズ1学習器

プログラム予測学習の統合学習器
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np

from core.data_structures import DataPair
from .program_predictor import ProgramPredictor, ProgramPredictorTrainer, PredictorConfig


@dataclass
class Phase1TrainingConfig:
    """フェーズ1学習設定"""
    predictor_config: PredictorConfig = None
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    enable_program_synthesis: bool = True

    def __post_init__(self):
        if self.predictor_config is None:
            self.predictor_config = PredictorConfig()


class Phase1Trainer:
    """フェーズ1学習器"""

    def __init__(self, config: Optional[Phase1TrainingConfig] = None):
        """初期化"""
        self.config = config or Phase1TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 学習器の初期化
        self.predictor_trainer = ProgramPredictorTrainer(self.config.predictor_config)

        # 学習履歴
        self.training_history = []

    def train(self, train_loader=None, val_loader=None, num_epochs=None, data_pairs: List[DataPair] = None) -> Dict[str, Any]:
        """学習を実行

        Args:
            train_loader: 訓練データローダー（DataLoaderまたはNone）
            val_loader: 検証データローダー（DataLoaderまたはNone）
            num_epochs: エポック数（DataLoaderを使う場合）
            data_pairs: 学習データ（train_loaderがNoneの場合に使用）

        Returns:
            学習結果
        """
        # DataLoaderを使う場合（新しい実装）
        if train_loader is not None:
            print(f"フェーズ1学習開始: DataLoader使用")

            # エポック数が指定されていない場合はデフォルト値を使用
            if num_epochs is None:
                num_epochs = self.config.num_epochs

            # 訓練ループを実行
            training_results = self._train_with_dataloader(train_loader, val_loader, num_epochs)

            result = {
                'mode': 'dataloader',
                'status': 'completed',
                'num_epochs': num_epochs,
                'training_results': training_results
            }
            return result

        # train_loaderがNoneの場合、data_pairsを使用
        if data_pairs is None:
            raise ValueError("train_loaderまたはdata_pairsのいずれかが必要です")

        print(f"フェーズ1学習開始: {len(data_pairs)}ペア")

        # データを分割
        train_pairs, val_pairs = self._split_data(data_pairs)

        # プログラム予測器の学習
        print("プログラム予測器の学習中...")
        predictor_result = self.predictor_trainer.train(train_pairs)

        # 検証
        if val_pairs:
            print("検証中...")
            validation_result = self._validate(val_pairs)
        else:
            validation_result = None

        # 結果をまとめる
        result = {
            'predictor_training': predictor_result,
            'validation': validation_result,
            'total_pairs': len(data_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs)
        }

        self.training_history.append(result)

        print("フェーズ1学習完了")
        return result

    def predict_program(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """プログラムを予測

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            予測されたプログラム
        """
        return self.predictor_trainer.predict(input_grid, output_grid)

    def _split_data(self, data_pairs: List[DataPair]) -> tuple:
        """データを訓練・検証に分割

        Args:
            data_pairs: データペアのリスト

        Returns:
            (訓練データ, 検証データ)
        """
        if self.config.validation_split <= 0:
            return data_pairs, []

        # ランダムに分割
        np.random.seed(42)
        indices = np.random.permutation(len(data_pairs))
        split_idx = int(len(data_pairs) * (1 - self.config.validation_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_pairs = [data_pairs[i] for i in train_indices]
        val_pairs = [data_pairs[i] for i in val_indices]

        return train_pairs, val_pairs

    def _validate(self, val_pairs: List[DataPair]) -> Dict[str, Any]:
        """検証を実行

        Args:
            val_pairs: 検証データ

        Returns:
            検証結果
        """
        correct_predictions = 0
        total_predictions = len(val_pairs)

        for pair in val_pairs:
            predicted_program = self.predict_program(
                np.array(pair.input),
                np.array(pair.output)
            )

            # プログラム一致チェック
            if predicted_program == pair.program:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

    def get_training_history(self) -> List[Dict[str, Any]]:
        """学習履歴を取得

        Returns:
            学習履歴のリスト
        """
        return self.training_history

    def save_model(self, filepath: str):
        """モデルを保存

        Args:
            filepath: 保存先ファイルパス
        """
        torch.save({
            'model_state_dict': self.predictor_trainer.model.state_dict(),
            'optimizer_state_dict': self.predictor_trainer.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)

    def load_model(self, filepath: str):
        """モデルを読み込み

        Args:
            filepath: 読み込み元ファイルパス
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.predictor_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        self.predictor_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

    def _train_with_dataloader(self, train_loader, val_loader=None, num_epochs=10) -> Dict[str, Any]:
        """DataLoaderを使った訓練ループ

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー（オプション）
            num_epochs: エポック数

        Returns:
            訓練結果
        """
        results = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'best_val_accuracy': 0.0,
            'best_epoch': 0
        }

        for epoch in range(num_epochs):
            print(f"エポック {epoch + 1}/{num_epochs}")

            # 訓練フェーズ
            train_loss, train_accuracy = self._train_epoch(train_loader)
            results['train_losses'].append(train_loss)
            results['train_accuracies'].append(train_accuracy)

            # 検証フェーズ
            if val_loader is not None:
                val_loss, val_accuracy = self._validate_epoch(val_loader)
                results['val_losses'].append(val_loss)
                results['val_accuracies'].append(val_accuracy)

                # 最良のモデルを保存
                if val_accuracy > results['best_val_accuracy']:
                    results['best_val_accuracy'] = val_accuracy
                    results['best_epoch'] = epoch + 1

                print(f"訓練損失: {train_loss:.4f}, 訓練精度: {train_accuracy:.4f}")
                print(f"検証損失: {val_loss:.4f}, 検証精度: {val_accuracy:.4f}")
            else:
                print(f"訓練損失: {train_loss:.4f}, 訓練精度: {train_accuracy:.4f}")

        return results

    def _train_epoch(self, train_loader) -> tuple:
        """1エポックの訓練"""
        self.predictor_trainer.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # バッチデータの処理
            if isinstance(batch, dict):
                inputs = batch.get('input', batch.get('inputs'))
                targets = batch.get('target', batch.get('targets'))
            else:
                inputs, targets = batch

            # デバイスに移動
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 勾配をリセット
            self.predictor_trainer.optimizer.zero_grad()

            # フォワードパス
            outputs = self.predictor_trainer.model(inputs)
            loss = self.predictor_trainer.criterion(outputs, targets)

            # バックワードパス
            loss.backward()
            self.predictor_trainer.optimizer.step()

            # 統計更新
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader) -> tuple:
        """1エポックの検証"""
        self.predictor_trainer.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # バッチデータの処理
                if isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('inputs'))
                    targets = batch.get('target', batch.get('targets'))
                else:
                    inputs, targets = batch

                # デバイスに移動
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # フォワードパス
                outputs = self.predictor_trainer.model(inputs)
                loss = self.predictor_trainer.criterion(outputs, targets)

                # 統計更新
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
