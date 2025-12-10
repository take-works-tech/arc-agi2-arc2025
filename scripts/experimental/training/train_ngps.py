"""
NGPS / DSL Selector 訓練スクリプト

DSL Selectorを訓練

使い方:
    python scripts/production/training/train_ngps.py \\
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

from src.hybrid_system.models.program_synthesis.dsl_selector import DSLSelector


class NGPSDataset(Dataset):
    """NGPS / DSL Selector訓練データセット"""

    def __init__(self, jsonl_path: str, dsl_vocab: Dict[str, int] = None):
        self.samples = []
        self.dsl_vocab = dsl_vocab or {}
        self.vocab_size = len(self.dsl_vocab) if self.dsl_vocab else 100

        # データを読み込んでDSL語彙を構築
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
                    # DSL語彙を更新
                    if 'dsl_probabilities' in sample:
                        for dsl_cmd in sample['dsl_probabilities'].keys():
                            if dsl_cmd not in self.dsl_vocab:
                                self.dsl_vocab[dsl_cmd] = len(self.dsl_vocab)

        self.vocab_size = max(len(self.dsl_vocab), 100)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # グリッド特徴量を取得（簡易版：固定次元のベクトルに変換）
        grid_features = sample.get('grid_features', {})
        feature_vector = [
            grid_features.get('input_size', 0),
            grid_features.get('output_size', 0),
            grid_features.get('input_unique_colors', 0),
            grid_features.get('output_unique_colors', 0),
            grid_features.get('input_mean', 0.0),
            grid_features.get('output_mean', 0.0),
        ]
        # 256次元に拡張（簡易版）
        while len(feature_vector) < 256:
            feature_vector.append(0.0)
        feature_vector = feature_vector[:256]

        # DSL確率をベクトル化
        dsl_probs = sample.get('dsl_probabilities', {})
        dsl_vector = np.zeros(self.vocab_size)
        for dsl_cmd, prob in dsl_probs.items():
            if dsl_cmd in self.dsl_vocab:
                dsl_vector[self.dsl_vocab[dsl_cmd]] = prob

        return {
            'grid_features': torch.tensor(feature_vector, dtype=torch.float32),
            'dsl_probabilities': torch.tensor(dsl_vector, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'grid_features': torch.stack([item['grid_features'] for item in batch]),
        'dsl_probabilities': torch.stack([item['dsl_probabilities'] for item in batch]),
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['grid_features'].to(device)
        targets = batch['dsl_probabilities'].to(device)

        predictions = model(features)
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
            features = batch['grid_features'].to(device)
            targets = batch['dsl_probabilities'].to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='NGPS / DSL Selector訓練')
    parser.add_argument('train_data', type=str, help='訓練データJSONLファイルのパス')
    parser.add_argument('--output-dir', type=str, default='learning_outputs/ngps', help='出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学習率')
    parser.add_argument('--val-split', type=float, default=0.2, help='検証データの割合')
    parser.add_argument('--input-dim', type=int, default=256, help='入力特徴量の次元')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隠れ層の次元')
    parser.add_argument('--num-dsl-commands', type=int, default=100, help='DSLコマンド数')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # データセットを読み込んで語彙サイズを決定
    temp_dataset = NGPSDataset(args.train_data)
    vocab_size = temp_dataset.vocab_size
    dsl_vocab = temp_dataset.dsl_vocab

    print(f"DSL語彙サイズ: {vocab_size}")

    dataset = NGPSDataset(args.train_data, dsl_vocab=dsl_vocab)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = DSLSelector(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_dsl_commands=max(vocab_size, args.num_dsl_commands),
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DSL語彙を保存
    with open(output_dir / 'dsl_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(dsl_vocab, f, ensure_ascii=False, indent=2)

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
            'dsl_vocab': dsl_vocab,
        }
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"ベストモデルを保存: val_loss={val_loss:.4f}")

    print(f"\n訓練完了: ベスト検証損失={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
