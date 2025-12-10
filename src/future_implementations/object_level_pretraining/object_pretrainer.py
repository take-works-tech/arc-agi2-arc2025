"""
Object-level Pretraining Trainer

オブジェクトレベルの事前学習トレーナー
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.hybrid_system.models.program_synthesis.object_encoder import ObjectEncoder
from src.hybrid_system.models.program_synthesis.object_graph_encoder import ObjectGraphEncoder
from src.hybrid_system.models.program_synthesis.relation_classifier import RelationClassifier


@dataclass
class ObjectPretrainingConfig:
    """オブジェクトレベル事前学習設定"""
    # モデル設定
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # 学習設定
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100

    # タスク設定
    enable_object_embedding: bool = True  # オブジェクト埋め込みの事前学習
    enable_graph_embedding: bool = True  # グラフ埋め込みの事前学習
    enable_relation_classification: bool = True  # 関係分類の事前学習

    # その他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_clip: float = 1.0
    save_checkpoint_interval: int = 10


class ObjectLevelPretrainer:
    """オブジェクトレベル事前学習トレーナー"""

    def __init__(
        self,
        config: ObjectPretrainingConfig,
        object_encoder: Optional[ObjectEncoder] = None,
        graph_encoder: Optional[ObjectGraphEncoder] = None,
        relation_classifier: Optional[RelationClassifier] = None
    ):
        """
        初期化

        Args:
            config: 設定
            object_encoder: オブジェクトエンコーダー（Noneの場合は新規作成）
            graph_encoder: グラフエンコーダー（Noneの場合は新規作成）
            relation_classifier: 関係分類器（Noneの場合は新規作成）
        """
        self.config = config
        self.device = torch.device(config.device)

        # モデルの初期化
        if object_encoder is None:
            self.object_encoder = ObjectEncoder(
                embed_dim=config.embed_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            self.object_encoder = object_encoder

        if graph_encoder is None and config.enable_graph_embedding:
            self.graph_encoder = ObjectGraphEncoder(
                node_dim=config.embed_dim,
                edge_dim=config.embed_dim // 2,
                embed_dim=config.embed_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            self.graph_encoder = graph_encoder

        if relation_classifier is None and config.enable_relation_classification:
            self.relation_classifier = RelationClassifier(
                object_dim=config.embed_dim,
                num_relation_types=10,  # 関係タイプ数（仮定）
                hidden_dim=config.embed_dim,
                dropout=config.dropout
            )
        else:
            self.relation_classifier = relation_classifier

        # モデルをデバイスに移動
        self.object_encoder = self.object_encoder.to(self.device)
        if self.graph_encoder:
            self.graph_encoder = self.graph_encoder.to(self.device)
        if self.relation_classifier:
            self.relation_classifier = self.relation_classifier.to(self.device)

        # 損失関数
        self.object_loss_fn = nn.MSELoss()  # オブジェクト再構成損失
        self.relation_loss_fn = nn.BCEWithLogitsLoss()  # 関係分類損失

        # オプティマイザー
        params = list(self.object_encoder.parameters())
        if self.graph_encoder:
            params.extend(list(self.graph_encoder.parameters()))
        if self.relation_classifier:
            params.extend(list(self.relation_classifier.parameters()))

        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

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
        self.object_encoder.train()
        if self.graph_encoder:
            self.graph_encoder.train()
        if self.relation_classifier:
            self.relation_classifier.train()

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
            loss = 0.0

            # オブジェクト埋め込みの事前学習
            if self.config.enable_object_embedding:
                object_loss = self._train_object_embedding(batch)
                loss += object_loss

            # グラフ埋め込みの事前学習
            if self.config.enable_graph_embedding and self.graph_encoder:
                graph_loss = self._train_graph_embedding(batch)
                loss += graph_loss

            # 関係分類の事前学習
            if self.config.enable_relation_classification and self.relation_classifier:
                relation_loss = self._train_relation_classification(batch)
                loss += relation_loss

            # バックプロパゲーション
            self.optimizer.zero_grad()
            loss.backward()

            # 勾配クリッピング
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.object_encoder.parameters()) +
                    (list(self.graph_encoder.parameters()) if self.graph_encoder else []) +
                    (list(self.relation_classifier.parameters()) if self.relation_classifier else []),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _train_object_embedding(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        オブジェクト埋め込みの事前学習

        Args:
            batch: バッチデータ

        Returns:
            loss: 損失値
        """
        # オブジェクトをエンコード
        input_objects = batch['input_objects']
        input_encoded = self.object_encoder(
            input_objects,
            batch['input_background_color'].to(self.device),
            batch['input_grid_width'].to(self.device),
            batch['input_grid_height'].to(self.device)
        )  # [1, num_objects, dim]

        # 再構成タスク（オブジェクト特徴量を再構成）
        # 簡易版: オブジェクト特徴量を再構成
        object_features = self._extract_object_features(input_objects)
        reconstructed_features = self._reconstruct_features(input_encoded)

        # 再構成損失
        loss = self.object_loss_fn(reconstructed_features, object_features)

        return loss

    def _train_graph_embedding(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        グラフ埋め込みの事前学習

        Args:
            batch: バッチデータ

        Returns:
            loss: 損失値
        """
        # グラフを構築してエンコード
        if 'graph' not in batch:
            return torch.tensor(0.0, device=self.device)

        graph = batch['graph']
        graph_encoded = self.graph_encoder(
            graph.node_features.to(self.device),
            graph.edge_index.to(self.device),
            graph.edge_attr.to(self.device) if graph.edge_attr is not None else None
        )  # [1, num_nodes, dim]

        # グラフ再構成タスク（簡易版）
        # 実際の実装では、より高度な再構成タスクを使用
        loss = torch.tensor(0.0, device=self.device)  # プレースホルダー

        return loss

    def _train_relation_classification(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        関係分類の事前学習

        Args:
            batch: バッチデータ

        Returns:
            loss: 損失値
        """
        # オブジェクトペアの関係を分類
        if 'object_pairs' not in batch or 'relation_labels' not in batch:
            return torch.tensor(0.0, device=self.device)

        object_pairs = batch['object_pairs']
        relation_labels = batch['relation_labels'].to(self.device)

        # オブジェクトをエンコード
        obj1_encoded = self.object_encoder(
            [pair[0] for pair in object_pairs],
            batch['input_background_color'].to(self.device),
            batch['input_grid_width'].to(self.device),
            batch['input_grid_height'].to(self.device)
        )

        obj2_encoded = self.object_encoder(
            [pair[1] for pair in object_pairs],
            batch['input_background_color'].to(self.device),
            batch['input_grid_width'].to(self.device),
            batch['input_grid_height'].to(self.device)
        )

        # 関係を分類
        relation_logits = self.relation_classifier(
            obj1_encoded.mean(dim=1),  # [batch, dim]
            obj2_encoded.mean(dim=1)   # [batch, dim]
        )  # [batch, num_relation_types]

        # 関係分類損失
        loss = self.relation_loss_fn(relation_logits, relation_labels)

        return loss

    def _extract_object_features(self, objects: List[Any]) -> torch.Tensor:
        """
        オブジェクト特徴量を抽出

        Args:
            objects: オブジェクトのリスト

        Returns:
            features: 特徴量テンソル
        """
        # 簡易版: オブジェクトの基本特徴量を抽出
        features = []
        for obj in objects:
            obj_features = [
                obj.color if hasattr(obj, 'color') else 0,
                obj.size if hasattr(obj, 'size') else 0,
                obj.width if hasattr(obj, 'width') else 0,
                obj.height if hasattr(obj, 'height') else 0,
            ]
            features.append(obj_features)

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _reconstruct_features(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        エンコードされた特徴量から元の特徴量を再構成

        Args:
            encoded: エンコードされた特徴量 [1, num_objects, dim]

        Returns:
            reconstructed: 再構成された特徴量
        """
        # 簡易版: 線形投影で再構成
        if not hasattr(self, '_reconstruction_head'):
            self._reconstruction_head = nn.Linear(
                self.config.embed_dim,
                4  # 基本特徴量の数（color, size, width, height）
            ).to(self.device)

        reconstructed = self._reconstruction_head(encoded.mean(dim=0))  # [num_objects, 4]
        return reconstructed

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """1エポックの検証"""
        self.object_encoder.eval()
        if self.graph_encoder:
            self.graph_encoder.eval()
        if self.relation_classifier:
            self.relation_classifier.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = 0.0

                if self.config.enable_object_embedding:
                    object_loss = self._train_object_embedding(batch)
                    loss += object_loss

                if self.config.enable_graph_embedding and self.graph_encoder:
                    graph_loss = self._train_graph_embedding(batch)
                    loss += graph_loss

                if self.config.enable_relation_classification and self.relation_classifier:
                    relation_loss = self._train_relation_classification(batch)
                    loss += relation_loss

                total_loss += loss.item()
                num_batches += 1

        self.object_encoder.train()
        if self.graph_encoder:
            self.graph_encoder.train()
        if self.relation_classifier:
            self.relation_classifier.train()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save_checkpoint(self, epoch: int, suffix: str):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'object_encoder_state_dict': self.object_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        if self.graph_encoder:
            checkpoint['graph_encoder_state_dict'] = self.graph_encoder.state_dict()
        if self.relation_classifier:
            checkpoint['relation_classifier_state_dict'] = self.relation_classifier.state_dict()

        import os
        checkpoint_dir = "checkpoints/object_level_pretraining"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
