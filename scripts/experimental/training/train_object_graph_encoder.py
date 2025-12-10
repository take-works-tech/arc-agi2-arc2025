"""
Object Graph + GNN 訓練スクリプト

Object Graph Encoder（Graphormer/EGNN）を訓練

使い方:
    python scripts/production/training/train_object_graph_encoder.py \\
        <train_data_jsonl> \\
        [--output-dir OUTPUT_DIR] \\
        [--encoder-type {graphormer,egnn}] \\
        [--epochs N] \\
        [--batch-size N] \\
        [--learning-rate LR]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_system.models.program_synthesis.object_graph_encoder import ObjectGraphEncoder
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder, ObjectGraph
from src.hybrid_system.inference.object_matching.object_extractor import ObjectExtractor
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType


class ObjectGraphDataset(Dataset):
    """Object Graph訓練データセット"""

    def __init__(self, jsonl_path: str):
        """
        初期化

        Args:
            jsonl_path: JSONLファイルのパス
        """
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # グラフ特徴量を取得
        graph_features = sample['graph_features']

        # テンソルに変換
        node_features = torch.tensor(graph_features['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_features['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph_features['edge_attr'], dtype=torch.float32)

        # ObjectGraphオブジェクトを再構築（簡易版）
        # 実際の実装では、オブジェクト情報も保存・復元する必要がある
        nodes = []  # 簡易版：空のリスト
        edges = []  # 簡易版：空のリスト
        edge_type_map = {'adjacent': 0, 'spatial': 1}

        graph = ObjectGraph(
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type_map=edge_type_map
        )

        # プログラムの存在をラベルとして使用（簡易版）
        has_program = 1.0 if sample.get('program') else 0.0

        return {
            'graph': graph,
            'has_program': torch.tensor(has_program, dtype=torch.float32),
            'task_id': sample.get('task_id', ''),
            'pair_index': sample.get('pair_index', 0)
        }


def collate_fn(batch):
    """バッチ処理用のコレート関数"""
    graphs = [item['graph'] for item in batch]
    has_program = torch.stack([item['has_program'] for item in batch])
    task_ids = [item['task_id'] for item in batch]
    pair_indices = [item['pair_index'] for item in batch]

    return {
        'graphs': graphs,
        'has_program': has_program,
        'task_ids': task_ids,
        'pair_indices': pair_indices
    }


class ObjectGraphTrainer:
    """Object Graph Encoder訓練クラス"""

    def __init__(
        self,
        model: ObjectGraphEncoder,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        初期化

        Args:
            model: ObjectGraphEncoderモデル
            device: デバイス
            learning_rate: 学習率
            weight_decay: 重み減衰
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()  # 簡易版：回帰タスクとして扱う

    def train_epoch(self, dataloader: DataLoader) -> float:
        """1エポック訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            graphs = batch['graphs']
            targets = batch['has_program'].to(self.device)

            # バッチ内の各グラフを処理
            batch_outputs = []
            for graph in graphs:
                # グラフをデバイスに移動
                graph.node_features = graph.node_features.to(self.device)
                graph.edge_index = graph.edge_index.to(self.device)
                graph.edge_attr = graph.edge_attr.to(self.device)

                # フォワードパス
                output, _ = self.model(graph)

                # グラフ全体の表現を取得（平均プーリング）
                graph_embedding = output.mean(dim=1)  # [batch, embed_dim]
                batch_outputs.append(graph_embedding)

            # バッチをスタック
            if batch_outputs:
                outputs = torch.cat(batch_outputs, dim=0)  # [batch_size, embed_dim]

                # 簡易版：線形層でスカラーに変換
                if not hasattr(self, 'output_projection'):
                    self.output_projection = nn.Linear(outputs.size(1), 1).to(self.device)
                    self.optimizer.add_param_group({'params': self.output_projection.parameters()})

                predictions = self.output_projection(outputs).squeeze(-1)  # [batch_size]

                # 損失計算
                loss = self.criterion(predictions, targets)

                # バックプロパゲーション
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, dataloader: DataLoader) -> float:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                graphs = batch['graphs']
                targets = batch['has_program'].to(self.device)

                batch_outputs = []
                for graph in graphs:
                    graph.node_features = graph.node_features.to(self.device)
                    graph.edge_index = graph.edge_index.to(self.device)
                    graph.edge_attr = graph.edge_attr.to(self.device)

                    output, _ = self.model(graph)
                    graph_embedding = output.mean(dim=1)
                    batch_outputs.append(graph_embedding)

                if batch_outputs:
                    outputs = torch.cat(batch_outputs, dim=0)

                    if hasattr(self, 'output_projection'):
                        predictions = self.output_projection(outputs).squeeze(-1)
                        loss = self.criterion(predictions, targets)
                        total_loss += loss.item()
                        num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Object Graph + GNN訓練')
    parser.add_argument('train_data', type=str, help='訓練データJSONLファイルのパス')
    parser.add_argument('--output-dir', type=str, default='learning_outputs/object_graph_encoder', help='出力ディレクトリ')
    parser.add_argument('--encoder-type', type=str, choices=['graphormer', 'egnn'], default='graphormer', help='エンコーダータイプ')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=8, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学習率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='重み減衰')
    parser.add_argument('--val-split', type=float, default=0.2, help='検証データの割合')
    parser.add_argument('--node-feature-dim', type=int, default=12, help='ノード特徴量の次元')
    parser.add_argument('--embed-dim', type=int, default=256, help='埋め込み次元')
    parser.add_argument('--num-layers', type=int, default=4, help='レイヤー数')
    parser.add_argument('--num-heads', type=int, default=8, help='アテンションヘッド数（Graphormerのみ）')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')

    args = parser.parse_args()

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # データセット読み込み
    print(f"データセット読み込み: {args.train_data}")
    dataset = ObjectGraphDataset(args.train_data)
    print(f"データセットサイズ: {len(dataset)}")

    # 訓練/検証分割
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # モデル初期化
    print(f"モデル初期化: {args.encoder_type}")
    model = ObjectGraphEncoder(
        encoder_type=args.encoder_type,
        node_feature_dim=args.node_feature_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )

    # 訓練器初期化
    trainer = ObjectGraphTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練ループ
    print(f"\n訓練開始: {args.epochs}エポック")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nエポック {epoch}/{args.epochs}")

        # 訓練
        train_loss = trainer.train_epoch(train_loader)
        print(f"訓練損失: {train_loss:.4f}")

        # 検証
        val_loss = trainer.validate(val_loader)
        print(f"検証損失: {val_loss:.4f}")

        # チェックポイント保存
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'encoder_type': args.encoder_type,
            'config': vars(args)
        }

        # 最新チェックポイント
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')

        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"ベストモデルを保存: val_loss={val_loss:.4f}")

        # エポックごとのチェックポイント
        if epoch % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')

    print(f"\n訓練完了")
    print(f"ベスト検証損失: {best_val_loss:.4f}")
    print(f"モデル保存先: {output_dir}")


if __name__ == "__main__":
    main()
