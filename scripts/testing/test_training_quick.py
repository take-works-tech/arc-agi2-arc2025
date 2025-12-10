"""
小規模学習テストスクリプト

1エポックのみ実行して、学習パイプラインが正常に動作するか確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import torch
import os
from torch.utils.data import DataLoader, random_split

from src.hybrid_system.models.program_synthesis import ProgramSynthesisModel
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
from src.hybrid_system.learning.phase1_trainer.full_trainer import FullPhase1Trainer
from src.hybrid_system.learning.phase1_trainer.dataset import DataPairDataset, collate_fn
from src.hybrid_system.data_management.io import DataIO
from src.hybrid_system.utils.logging import Logger, TrainingLogger
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from src.hybrid_system.ir.parser import RelabelTransformer, RelabelTransformerConfig
from src.hybrid_system.ir.serialization import sequence_to_template_string


def main():
    """メイン処理"""
    # 設定を読み込み
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化
    logger = Logger.get_logger("phase1_training_test", config['logging']['log_dir'])
    logger.info("Phase 1 小規模学習テスト開始（1エポック）")

    # 訓練ロガーを初期化
    training_logger = TrainingLogger(
        config['logging']['log_dir'],
        config['logging']['experiment_name'] + "_phase1_test"
    )
    training_logger.log_config(config)

    # デバイス設定
    device = config['device']['type']
    logger.info(f"使用デバイス: {device}")

    # トークナイザーを初期化
    tokenizer = ProgramTokenizer()
    logger.info("トークナイザー初期化完了")

    # データを読み込み
    logger.info("データ読み込み中...")
    data_io = DataIO(config['data']['generated_dir'])
    all_data_pairs = []

    try:
        # 1. outputsから変換したデータ
        converted_data_path = os.path.join('phase1_pairs', 'data_pairs_from_outputs.jsonl.gz')
        converted_data_full_path = os.path.join(config['data']['generated_dir'], converted_data_path)
        if os.path.exists(converted_data_full_path):
            logger.info(f"outputsから変換したDataPair読み込み中...")
            converted_pairs = data_io.load_data_pairs(converted_data_path)
            logger.info(f"outputsから変換したDataPair読み込み完了: {len(converted_pairs)}個")
            # テスト用にサブセット（最初の1000ペアのみ）
            all_data_pairs.extend(converted_pairs[:1000])
            logger.info(f"テスト用に1000ペアに制限")

        # 2. Pipelineで生成したデータ
        pipeline_data_path = os.path.join('phase1_pairs', 'data_pairs.jsonl.gz')
        pipeline_data_full_path = os.path.join(config['data']['generated_dir'], pipeline_data_path)
        if os.path.exists(pipeline_data_full_path):
            logger.info(f"Pipelineで生成したDataPair読み込み中...")
            pipeline_pairs = data_io.load_data_pairs(pipeline_data_path)
            logger.info(f"Pipelineで生成したDataPair読み込み完了: {len(pipeline_pairs)}個")
            all_data_pairs.extend(pipeline_pairs[:100])  # テスト用に100ペア

        # 3. 手動プログラムから変換したデータ
        manual_data_path = os.path.join('phase1_pairs', 'data_pairs_manual.jsonl.gz')
        manual_data_full_path = os.path.join(config['data']['generated_dir'], manual_data_path)
        if os.path.exists(manual_data_full_path):
            logger.info(f"手動プログラムから変換したDataPair読み込み中...")
            manual_pairs = data_io.load_data_pairs(manual_data_path)
            logger.info(f"手動プログラムから変換したDataPair読み込み完了: {len(manual_pairs)}個")
            all_data_pairs.extend(manual_pairs)

        # 4. オブジェクト操作コマンドから変換したデータ
        object_ops_data_path = os.path.join('phase1_pairs', 'data_pairs_object_operations.jsonl.gz')
        object_ops_data_full_path = os.path.join(config['data']['generated_dir'], object_ops_data_path)
        if os.path.exists(object_ops_data_full_path):
            logger.info(f"オブジェクト操作コマンドから変換したDataPair読み込み中...")
            object_ops_pairs = data_io.load_data_pairs(object_ops_data_path)
            logger.info(f"オブジェクト操作コマンドから変換したDataPair読み込み完了: {len(object_ops_pairs)}個")
            all_data_pairs.extend(object_ops_pairs[:100])  # テスト用に100ペア

        data_pairs = all_data_pairs
        logger.info(f"合計DataPair読み込み完了: {len(data_pairs)}個（テスト用サブセット）")

        # テンプレートシーケンスへ変換
        logger.info("テンプレートシーケンスへ変換中...")
        transformer = RelabelTransformer(RelabelTransformerConfig())
        converted_count = 0
        for dp in data_pairs:
            if not dp.program:
                continue
            try:
                sequence = transformer.transform(
                    dp.program,
                    task_id=dp.pair_id,
                    sequence_metadata={
                        "pair_id": dp.pair_id,
                        "source_metadata": dp.metadata,
                    },
                )
                dp.metadata = dp.metadata or {}
                dp.metadata["original_program"] = dp.program
                dp.metadata["template_sequence"] = sequence.to_dict()
                dp.program = sequence_to_template_string(sequence)
                converted_count += 1
            except Exception as transform_error:
                logger.warning(f"テンプレート変換に失敗: pair_id={dp.pair_id} error={transform_error}")
        logger.info(f"テンプレート変換完了: {converted_count}/{len(data_pairs)}")

        # トークナイザーの語彙を構築（全テンプレートをスキャン）
        logger.info("トークナイザーの語彙を構築中...")
        for dp in data_pairs:
            tokenizer.encode(dp.program, add_special_tokens=False)
        logger.info(f"語彙サイズ: {tokenizer.vocab_size()}")

        # データセットを作成
        full_dataset = DataPairDataset(
            data_pairs=data_pairs,
            tokenizer=tokenizer,
            max_program_length=config['models']['program_synthesis']['max_program_length'],
            max_grid_size=config['data']['generation']['max_grid_size']
        )

        # 訓練/検証データに分割
        val_ratio = config['evaluation']['validation_split']
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"データ分割: 訓練={train_size}, 検証={val_size}")

        # データローダーを作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['phase1']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['phase1']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        logger.info("データローダー作成完了")

    except FileNotFoundError as e:
        logger.error(f"データファイルが見つかりません: {e}")
        return
    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # 語彙サイズを更新
    vocab_size = tokenizer.vocab_size()
    config['models']['program_synthesis']['vocab_size'] = vocab_size
    logger.info(f"語彙サイズ: {vocab_size}")

    # モデル設定
    model_config = ProgramSynthesisConfig(
        **config['models']['program_synthesis']
    )
    model_config.device = device

    # モデルを作成
    model = ProgramSynthesisModel(model_config)
    logger.info(f"モデル作成完了: {model.count_parameters():,} パラメータ")

    # トレーナーを作成
    trainer = FullPhase1Trainer(
        model=model,
        config=config['training']['phase1'],
        logger=training_logger
    )

    logger.info("訓練準備完了")

    # テスト実行（1エポックのみ）
    logger.info("小規模学習テスト開始（1エポック）...")
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1  # テスト用に1エポックのみ
    )

    logger.info(f"テスト結果: {result}")

    # テスト用チェックポイント保存
    test_checkpoint_path = os.path.join(model_config.save_dir, 'test_model.pt')
    model.save_checkpoint(test_checkpoint_path, 1, trainer.optimizer)
    logger.info(f"テスト用モデルを保存: {test_checkpoint_path}")

    logger.info("小規模学習テスト完了")
    training_logger.close()


if __name__ == "__main__":
    main()
