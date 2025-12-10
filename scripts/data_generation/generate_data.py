"""
データ生成スクリプト

ハイブリッドアプローチのデータセットを生成
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 部分プログラムフローを有効化（環境変数が設定されていない場合のみ）
# 既に設定されている場合はその値を優先（CI/CDなどで外部から制御可能）
if 'USE_PARTIAL_PROGRAM_FLOW' not in os.environ:
    os.environ['USE_PARTIAL_PROGRAM_FLOW'] = 'true'

import yaml
from src.hybrid_system.learning.hybrid_pipeline import HybridLearningPipeline
from src.hybrid_system.utils.logging import Logger


def main():
    """メイン処理"""
    # 設定を読み込み
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化
    logger = Logger.get_logger("data_generation", config['logging']['log_dir'])
    logger.info("データ生成開始")

    # タイムスタンプ付きディレクトリを作成
    base_output_dir = "outputs"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp_dir)
    os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプ付きディレクトリ内にhybridサブディレクトリを作成
    hybrid_dir = os.path.join(output_dir, "hybrid")
    os.makedirs(hybrid_dir, exist_ok=True)

    print(f"新規出力ディレクトリを作成: {output_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 環境変数の設定を表示
    use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'not set')
    print(f"USE_PARTIAL_PROGRAM_FLOW = {use_partial_program_flow}")
    logger.info(f"USE_PARTIAL_PROGRAM_FLOW = {use_partial_program_flow}")

    # パイプライン設定（タイムスタンプ付きディレクトリ内のhybridディレクトリを使用）
    from src.hybrid_system.learning.hybrid_pipeline.pipeline import PipelineConfig
    pipeline_config_obj = PipelineConfig(base_dir=hybrid_dir)
    pipeline = HybridLearningPipeline(pipeline_config_obj)
    logger.info("パイプライン作成完了")

    # Phase1のみ実行（DataPair生成: 設計B - 1プログラム=1ペア）
    num_programs = config['data']['generation']['num_programs']

    logger.info(f"データ生成を開始: {num_programs}プログラム → {num_programs} DataPair（1プログラム=1ペア）")

    # Phase1: DataPair生成（設計B: 1プログラム=1ペア）
    data_pairs = pipeline.run_phase1_pair_generation(
        num_programs=num_programs,
        pairs_per_program=1,  # 設計B: 1プログラム=1ペア
        max_attempts_per_pair=50
    )

    # パイプライン内で保存済みなので、統計のみ計算

    # 統計情報を計算
    unique_programs = len(set(p.program for p in data_pairs))
    complexity_counts = {}
    for pair in data_pairs:
        comp = pair.get_program_complexity()
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1

    # レポートを出力
    logger.info("データ生成完了")
    logger.info(f"生成されたDataPair数: {len(data_pairs)}")
    logger.info(f"ユニークなプログラム数: {unique_programs}")
    logger.info(f"複雑度分布: {complexity_counts}")

    print("\n=== データ生成完了 ===")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"DataPair: {len(data_pairs)}個")
    print(f"ユニークなプログラム: {unique_programs}種類")
    if len(data_pairs) > 0:
        print(f"多様性率: {unique_programs / len(data_pairs) * 100:.1f}%")
    else:
        print("多様性率: N/A (DataPairが0個)")
    print(f"複雑度分布: {complexity_counts}")


if __name__ == "__main__":
    main()
