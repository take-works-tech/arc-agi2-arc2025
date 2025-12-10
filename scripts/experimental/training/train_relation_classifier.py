"""
Relation Classifier 訓練スクリプト

Relation Classifierを訓練

使い方:
    python scripts/production/training/train_relation_classifier.py \\
        <train_data_jsonl> \\
        [--output-dir OUTPUT_DIR] \\
        [--epochs N] \\
        [--batch-size N]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_system.models.program_synthesis.relation_classifier import RelationClassifier


class RelationClassifierDataset(Dataset):
    """Relation Classifier訓練データセット"""

    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'obj1_features': torch.tensor(sample['obj1_features'], dtype=torch.float32),
            'obj2_features': torch.tensor(sample['obj2_features'], dtype=torch.float32),
            'relative_features': torch.tensor(sample['relative_features'], dtype=torch.float32),
            'relation_labels': torch.tensor(sample['relation_labels'], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'obj1_features': torch.stack([item['obj1_features'] for item in batch]),
        'obj2_features': torch.stack([item['obj2_features'] for item in batch]),
        'relative_features': torch.stack([item['relative_features'] for item in batch]),
        'relation_labels': torch.stack([item['relation_labels'] for item in batch]),
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        obj1 = batch['obj1_features'].to(device)  # [batch, 8]
        obj2 = batch['obj2_features'].to(device)  # [batch, 8]
        relative = batch['relative_features'].to(device)  # [batch, 4]
        targets = batch['relation_labels'].to(device)  # [batch, num_relation_types]

        # ノード特徴量を投影
        obj1_embed = model.node_projection(obj1)  # [batch, embed_dim]
        obj2_embed = model.node_projection(obj2)  # [batch, embed_dim]

        # 結合して予測
        combined = torch.cat([obj1_embed, obj2_embed, relative], dim=1)  # [batch, embed_dim * 2 + 4]
        predictions = model.relation_mlp(combined)  # [batch, num_relation_types]

        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            obj1 = batch['obj1_features'].to(device)  # [batch, 8]
            obj2 = batch['obj2_features'].to(device)  # [batch, 8]
            relative = batch['relative_features'].to(device)  # [batch, 4]
            targets = batch['relation_labels'].to(device)  # [batch, num_relation_types]

            # ノード特徴量を投影
            obj1_embed = model.node_projection(obj1)  # [batch, embed_dim]
            obj2_embed = model.node_projection(obj2)  # [batch, embed_dim]

            # 結合して予測
            combined = torch.cat([obj1_embed, obj2_embed, relative], dim=1)  # [batch, embed_dim * 2 + 4]
            predictions = model.relation_mlp(combined)  # [batch, num_relation_types]
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Relation Classifier訓練')
    parser.add_argument('train_data', type=str, help='訓練データJSONLファイルのパス')
    parser.add_argument('--output-dir', type=str, default='learning_outputs/relation_classifier', help='出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学習率')
    parser.add_argument('--val-split', type=float, default=0.2, help='検証データの割合')
    parser.add_argument('--node-feature-dim', type=int, default=8, help='ノード特徴量の次元（obj1_features/obj2_featuresの次元）')
    parser.add_argument('--embed-dim', type=int, default=128, help='埋め込み次元')
    parser.add_argument('--num-relation-types', type=int, default=8, help='関係タイプ数')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    dataset = RelationClassifierDataset(args.train_data)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = RelationClassifier(
        node_feature_dim=args.node_feature_dim,
        embed_dim=args.embed_dim,
        num_relation_types=args.num_relation_types,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\nエポック {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"訓練損失: {train_loss:.4f}, 検証損失: {val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"ベストモデルを保存: val_loss={val_loss:.4f}")

    print(f"\n訓練完了: ベスト検証損失={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
