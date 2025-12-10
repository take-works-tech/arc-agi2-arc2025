"""
Color Role Classifier 訓練スクリプト

Color Role Classifierを訓練

使い方:
    python scripts/production/training/train_color_role_classifier.py \\
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

from src.hybrid_system.models.program_synthesis.color_role_classifier import ColorRoleClassifier


class ColorRoleDataset(Dataset):
    """Color Role Classifier訓練データセット"""

    def __init__(self, jsonl_path: str):
        self.samples = []
        self.role_to_id = {'background': 0, 'foreground': 1, 'structure': 2, 'other': 3}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        role_id = self.role_to_id.get(sample['color_role'], 3)
        return {
            'color_features': torch.tensor(sample['color_features'], dtype=torch.float32),
            'role_id': torch.tensor(role_id, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        'color_features': torch.stack([item['color_features'] for item in batch]),
        'role_id': torch.stack([item['role_id'] for item in batch]),
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['color_features'].to(device)
        targets = batch['role_id'].to(device)

        predictions = model(features)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            features = batch['color_features'].to(device)
            targets = batch['role_id'].to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return total_loss / len(dataloader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Color Role Classifier訓練')
    parser.add_argument('train_data', type=str, help='訓練データJSONLファイルのパス')
    parser.add_argument('--output-dir', type=str, default='learning_outputs/color_role_classifier', help='出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学習率')
    parser.add_argument('--val-split', type=float, default=0.2, help='検証データの割合')
    parser.add_argument('--feature-dim', type=int, default=13, help='特徴量の次元（データ生成スクリプトで生成される特徴量の次元）')
    parser.add_argument('--hidden-dim', type=int, default=64, help='隠れ層の次元')
    parser.add_argument('--num-roles', type=int, default=4, help='色役割の数')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    dataset = ColorRoleDataset(args.train_data)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ColorRoleClassifier(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_roles=args.num_roles,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nエポック {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"訓練: 損失={train_loss:.4f}, 精度={train_acc:.2f}%")
        print(f"検証: 損失={val_loss:.4f}, 精度={val_acc:.2f}%")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"ベストモデルを保存: val_acc={val_acc:.2f}%")

    print(f"\n訓練完了: ベスト検証精度={best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
