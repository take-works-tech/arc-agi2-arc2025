"""
ObjectBasedProgramSynthesisModel 訓練スクリプト

オブジェクトベースプログラム合成モデルを訓練
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import glob
import yaml
import torch
import os
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, random_split

from src.hybrid_system.models.program_synthesis.object_based_program_synthesis_model import ObjectBasedProgramSynthesisModel
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
from src.hybrid_system.learning.object_based_trainer.trainer import ObjectBasedTrainer
from src.hybrid_system.learning.object_based_trainer.dataset import ObjectBasedDataset, collate_fn
from src.hybrid_system.data_management.io.data_io import DataIO
from src.hybrid_system.utils.logging import Logger, TrainingLogger
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from datetime import datetime


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """ディレクトリ内で最新のチェックポイントを探索"""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
    candidates = glob.glob(pattern)

    def extract_epoch(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1

    candidates = [c for c in candidates if extract_epoch(c) >= 0]
    if candidates:
        candidates.sort(key=extract_epoch, reverse=True)
        return candidates[0]

    # フォールバックとして最新チェックポイントを確認
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        return latest_path

    return None


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Train ObjectBasedProgramSynthesisModel')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--data', type=str, required=True, help='Path to training data JSONL file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    # 設定を読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化
    log_dir = config.get('logging', {}).get('log_dir', 'logs/object_based')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logger = Logger(log_file)
    training_logger = TrainingLogger(log_dir)

    logger.info(f"訓練開始: {timestamp}")
    logger.info(f"設定ファイル: {args.config}")
    logger.info(f"データファイル: {args.data}")
    logger.info(f"デバイス: {args.device}")

    # データを読み込み
    data_io = DataIO()
    data_pairs = data_io.load_data_pairs(args.data)
    logger.info(f"読み込んだデータペア数: {len(data_pairs)}")

    # データセットを分割
    train_size = int(len(data_pairs) * config.get('data', {}).get('train_ratio', 0.8))
    val_size = len(data_pairs) - train_size
    train_pairs, val_pairs = random_split(data_pairs, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    logger.info(f"訓練データ: {len(train_pairs)} ペア")
    logger.info(f"検証データ: {len(val_pairs)} ペア")

    # トークナイザーを初期化
    tokenizer = ProgramTokenizer()
    vocab_size = tokenizer.vocab_size()
    logger.info(f"語彙サイズ: {vocab_size}")

    # データセットを初期化
    train_dataset = ObjectBasedDataset(
        data_pairs=train_pairs,
        tokenizer=tokenizer,
        max_program_length=config.get('data', {}).get('max_program_length', 512),
        use_ir_templates=config.get('data', {}).get('use_ir_templates', True),
        connectivity=config.get('data', {}).get('connectivity', 4)
    )

    val_dataset = ObjectBasedDataset(
        data_pairs=val_pairs,
        tokenizer=tokenizer,
        max_program_length=config.get('data', {}).get('max_program_length', 512),
        use_ir_templates=config.get('data', {}).get('use_ir_templates', True),
        connectivity=config.get('data', {}).get('connectivity', 4)
    )

    # データローダーを初期化
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('training', {}).get('num_workers', 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('training', {}).get('num_workers', 0)
    )

    # モデルを初期化
    model_config_dict = config.get('models', {}).get('object_based_program_synthesis', {})
    model_config_dict['vocab_size'] = vocab_size
    model_config = ProgramSynthesisConfig(**model_config_dict)
    model_config.device = args.device

    model = ObjectBasedProgramSynthesisModel(model_config)
    logger.info(f"モデル作成完了: {model.count_parameters():,} パラメータ")

    # トレーナーを初期化
    trainer = ObjectBasedTrainer(
        model=model,
        config=config.get('training', {}).get('object_based', {}),
        logger=training_logger
    )

    # チェックポイントから再開
    start_epoch = 1
    training_config = config.get('training', {}).get('object_based', {})
    resume_checkpoint = training_config.get('resume_checkpoint', None)
    auto_resume = training_config.get('auto_resume', False)

    if auto_resume or (isinstance(resume_checkpoint, str) and resume_checkpoint.lower() == "latest"):
        latest_checkpoint = find_latest_checkpoint(model_config.save_dir)
        if latest_checkpoint:
            resume_checkpoint = latest_checkpoint
            logger.info(f"最新チェックポイントを検出: {resume_checkpoint}")
        else:
            logger.info("自動再開: 有効なチェックポイントが見つかりません。新規学習を開始します")
            resume_checkpoint = None

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"チェックポイントから再開: {resume_checkpoint}")
        start_epoch = trainer.load_checkpoint(resume_checkpoint) + 1
        logger.info(f"エポック {start_epoch} から再開します")

    # 訓練を実行
    num_epochs = training_config.get('num_epochs', 100)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch
    )

    logger.info(f"訓練完了")
    logger.info(f"ベスト検証損失: {results['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
