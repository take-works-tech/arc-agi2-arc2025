"""
Contrastive Pretraining 学習スクリプト

ARC問題の訓練ペア（ラベルなし）を使用して対照学習による事前学習を実行
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
import os
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from src.hybrid_system.learning.contrastive_pretraining import (
    ContrastivePretrainer,
    ContrastivePretrainingConfig
)
from src.hybrid_system.models.program_synthesis.grid_encoder import GridEncoder
from src.hybrid_system.models.program_synthesis.object_encoder import ObjectEncoder


class ARCGridPairDataset(Dataset):
    """ARC問題のグリッドペアデータセット（ラベルなし）"""

    def __init__(
        self,
        data_path: str,
        model_type: str = "grid",
        max_samples: Optional[int] = None
    ):
        """
        初期化

        Args:
            data_path: データファイルのパス（JSON形式）
            model_type: モデルタイプ（"grid" or "object"）
            max_samples: 最大サンプル数（Noneの場合はすべて使用）
        """
        self.model_type = model_type
        self.samples = []

        # データを読み込む
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                # JSONL形式
                for line in f:
                    if max_samples and len(self.samples) >= max_samples:
                        break
                    sample = json.loads(line.strip())
                    self.samples.append(sample)
            else:
                # JSON形式
                data = json.load(f)
                if isinstance(data, list):
                    for sample in data:
                        if max_samples and len(self.samples) >= max_samples:
                            break
                        self.samples.append(sample)
                elif isinstance(data, dict):
                    # タスク形式の場合
                    for task_id, task_data in data.items():
                        if max_samples and len(self.samples) >= max_samples:
                            break
                        train_pairs = task_data.get('train', [])
                        for pair in train_pairs:
                            if max_samples and len(self.samples) >= max_samples:
                                break
                            self.samples.append({
                                'input': pair.get('input', []),
                                'output': pair.get('output', [])
                            })

        print(f"読み込んだサンプル数: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """サンプルを取得"""
        sample = self.samples[idx]

        if self.model_type == "grid":
            # グリッド形式
            input_grid = np.array(sample.get('input', []), dtype=np.int32)
            output_grid = np.array(sample.get('output', []), dtype=np.int32)

            return {
                'input_grid': torch.tensor(input_grid, dtype=torch.long),
                'output_grid': torch.tensor(output_grid, dtype=torch.long)
            }
        else:
            # オブジェクト形式（将来的に実装）
            # 現在はグリッド形式のみサポート
            input_grid = np.array(sample.get('input', []), dtype=np.int32)
            output_grid = np.array(sample.get('output', []), dtype=np.int32)

            return {
                'input_grid': torch.tensor(input_grid, dtype=torch.long),
                'output_grid': torch.tensor(output_grid, dtype=torch.long),
                'input_objects': [],  # オブジェクト抽出は将来的に実装
                'output_objects': [],
                'input_background_color': torch.tensor(0, dtype=torch.long),
                'output_background_color': torch.tensor(0, dtype=torch.long),
                'input_grid_width': torch.tensor(input_grid.shape[1], dtype=torch.long),
                'input_grid_height': torch.tensor(input_grid.shape[0], dtype=torch.long),
                'output_grid_width': torch.tensor(output_grid.shape[1], dtype=torch.long),
                'output_grid_height': torch.tensor(output_grid.shape[0], dtype=torch.long)
            }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """バッチを結合"""
    if not batch:
        return {}

    # グリッド形式の場合
    if 'input_grid' in batch[0] and 'output_grid' in batch[0]:
        # 最大サイズを取得
        max_h = max(sample['input_grid'].shape[0] for sample in batch)
        max_w = max(sample['input_grid'].shape[1] for sample in batch)

        # パディング
        input_grids = []
        output_grids = []
        for sample in batch:
            input_grid = sample['input_grid']
            output_grid = sample['output_grid']

            # パディング
            pad_h = max_h - input_grid.shape[0]
            pad_w = max_w - input_grid.shape[1]
            input_padded = torch.nn.functional.pad(
                input_grid, (0, pad_w, 0, pad_h), value=0
            )
            output_padded = torch.nn.functional.pad(
                output_grid, (0, pad_w, 0, pad_h), value=0
            )

            input_grids.append(input_padded)
            output_grids.append(output_padded)

        return {
            'input_grid': torch.stack(input_grids),
            'output_grid': torch.stack(output_grids)
        }

    return {}


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Contrastive Pretraining')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='データファイルのパス（JSONまたはJSONL形式）'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='grid',
        choices=['grid', 'object'],
        help='モデルタイプ（grid or object）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/contrastive_pretraining',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=256,
        help='埋め込み次元'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=4,
        help='レイヤー数'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='アテンションヘッド数'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='ドロップアウト率'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='学習率'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='バッチサイズ'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='エポック数'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='温度パラメータ'
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='simclr',
        choices=['simclr', 'infonce', 'contrastive'],
        help='損失関数タイプ'
    )
    parser.add_argument(
        '--enable_augmentation',
        action='store_true',
        help='データ拡張を有効にする'
    )
    parser.add_argument(
        '--augmentation_prob',
        type=float,
        default=0.5,
        help='データ拡張の確率'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大サンプル数（Noneの場合はすべて使用）'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='検証データの割合'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='デバイス'
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)

    # データセットを読み込む
    print(f"データを読み込み中: {args.data_path}")
    dataset = ARCGridPairDataset(
        data_path=args.data_path,
        model_type=args.model_type,
        max_samples=args.max_samples
    )

    # 訓練データと検証データに分割
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows環境では0を推奨
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Windows環境では0を推奨
    )

    print(f"訓練データ: {len(train_dataset)}サンプル")
    print(f"検証データ: {len(val_dataset)}サンプル")

    # 設定を作成
    config = ContrastivePretrainingConfig(
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        loss_type=args.loss_type,
        enable_augmentation=args.enable_augmentation,
        augmentation_prob=args.augmentation_prob,
        device=args.device
    )

    # トレーナーを初期化
    print("トレーナーを初期化中...")
    trainer = ContrastivePretrainer(config)

    # 学習を実行
    print("学習を開始...")
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # 結果を保存
    result_path = os.path.join(args.output_dir, 'training_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"学習完了: {result_path}")
    print(f"ベスト検証損失: {result.get('best_val_loss', 'N/A')}")


if __name__ == "__main__":
    main()
