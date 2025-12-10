"""
推論スクリプト

訓練済みモデルで推論を実行
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import torch
import json

from src.hybrid_system.models.program_synthesis import ProgramSynthesisModel
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
# SystemIntegrator は削除されたため、直接 ProgramSynthesisEngine を使用
from src.hybrid_system.inference.program_synthesis.synthesis_engine import ProgramSynthesisEngine
from src.hybrid_system.utils.logging import Logger
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from src.hybrid_system.core.data_structures import Task
from typing import List, Dict, Any
import os


def load_arc_test_data(file_path: str) -> List[Dict[str, Any]]:
    """
    ARC-AGI2テストデータを読み込み

    Args:
        file_path: JSONファイルパス

    Returns:
        テストタスクのリスト
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = []
    for task_id, task_data in data.items():
        tasks.append({
            'id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        })

    return tasks


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    推論結果を保存

    Args:
        results: 推論結果のリスト
        output_path: 出力ファイルパス
    """
    # ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # JSON形式で保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """メイン処理"""
    # 設定を読み込み
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化
    logger = Logger.get_logger("inference", config['logging']['log_dir'])
    logger.info("推論開始")

    # デバイス設定
    device = config['device']['type']
    logger.info(f"使用デバイス: {device}")

    # モデルを読み込み
    logger.info("モデル読み込み中...")

    # チェックポイントパスを取得（設定から、またはデフォルト）
    checkpoint_path = config.get('inference', {}).get('checkpoint_path', 'models/checkpoints/program_synthesis/best_model.pt')
    vocab_path = config.get('inference', {}).get('vocab_path', 'models/checkpoints/program_synthesis/vocab.json')

    # トークナイザーを初期化
    if os.path.exists(vocab_path):
        logger.info(f"語彙を読み込み: {vocab_path}")
        tokenizer = ProgramTokenizer.load_vocab(vocab_path)
        logger.info(f"語彙サイズ: {tokenizer.vocab_size()}")
    else:
        logger.warning(f"語彙ファイルが見つかりません: {vocab_path}")
        logger.info("空のトークナイザーを使用します")
        tokenizer = ProgramTokenizer()

    # プログラム合成モデル
    ps_config = ProgramSynthesisConfig(**config['models']['program_synthesis'])
    ps_model = ProgramSynthesisModel(ps_config)

    # チェックポイントを読み込み
    if os.path.exists(checkpoint_path):
        logger.info(f"チェックポイントを読み込み: {checkpoint_path}")
        ps_model.load_checkpoint(checkpoint_path)
    else:
        logger.warning(f"チェックポイントが見つかりません: {checkpoint_path}")
        logger.info("初期化されたモデルを使用します")

    ps_model.eval()

    logger.info("モデル読み込み完了")

    # プログラム合成エンジンを初期化（ニューラルモデルとトークナイザーを渡す）
    synthesis_engine = ProgramSynthesisEngine(
        config_file='configs/synthesis_config.yaml',
        neural_model=ps_model,
        tokenizer=tokenizer
    )

    logger.info("推論準備完了")

    # テストデータの読み込み
    test_data_path = config.get('data', {}).get('test_data_path', 'data/arc_agi2/arc-agi_test_challenges.json')

    try:
        logger.info(f"テストデータ読み込み中: {test_data_path}")
        test_tasks = load_arc_test_data(test_data_path)
        logger.info(f"テストタスク数: {len(test_tasks)}")
    except FileNotFoundError:
        logger.warning(f"テストデータが見つかりません: {test_data_path}")
        logger.info("デモモードで実行します")
        test_tasks = []

    # 推論実行
    if test_tasks:
        results = []
        for i, task in enumerate(test_tasks[:10]):  # 最初の10タスクのみ
            logger.info(f"タスク {i+1}/{min(10, len(test_tasks))}: {task['id']} を推論中...")

            try:
                # Task オブジェクトを作成
                task_obj = Task(
                    train=task['train'],
                    test=task['test'],
                    task_id=task['id']
                )

                # ProgramSynthesisEngineで推論
                program = synthesis_engine.synthesize_program(task_obj)

                # 結果を取得
                if program:
                    # プログラムを実行して予測を取得
                    # 注: 実際の実装では、プログラムを実行してテストグリッドの出力を生成する必要があります
                    predictions = program  # 暫定的にプログラムを返す
                    success = True
                else:
                    predictions = None
                    success = False

                results.append({
                    'task_id': task['id'],
                    'predictions': predictions,
                    'success': success
                })
                logger.info(f"タスク {task['id']}: 推論成功" if success else f"タスク {task['id']}: 推論失敗")
            except Exception as e:
                logger.error(f"タスク {task['id']}: 推論失敗 - {e}")
                results.append({
                    'task_id': task['id'],
                    'predictions': None,
                    'success': False,
                    'error': str(e)
                })

        # 結果保存
        output_path = 'results/predictions.json'
        save_results(results, output_path)
        logger.info(f"結果を保存: {output_path}")
        logger.info(f"成功: {sum(1 for r in results if r['success'])}/{len(results)}")
    else:
        logger.info("テストデータがないため推論をスキップ")

    logger.info("推論完了")


if __name__ == "__main__":
    main()
