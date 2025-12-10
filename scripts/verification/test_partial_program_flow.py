"""
部分プログラムフローの検証スクリプト

本番フローでgenerate_program_with_partial_program_flow()を10タスクで検証
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 部分プログラムフローを有効化
os.environ['USE_PARTIAL_PROGRAM_FLOW'] = 'true'

import yaml
from src.hybrid_system.learning.hybrid_pipeline import HybridLearningPipeline
from src.hybrid_system.utils.logging import Logger


def main():
    """検証処理"""
    # 設定を読み込み
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化
    logger = Logger.get_logger("partial_program_flow_verification", config['logging']['log_dir'])
    logger.info("部分プログラムフロー検証開始")

    # タイムスタンプ付きディレクトリを作成
    base_output_dir = "outputs"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, "verification", "partial_program_flow", timestamp_dir)
    os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプ付きディレクトリ内にhybridサブディレクトリを作成
    hybrid_dir = os.path.join(output_dir, "hybrid")
    os.makedirs(hybrid_dir, exist_ok=True)

    print(f"検証出力ディレクトリを作成: {output_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # パイプライン設定
    from src.hybrid_system.learning.hybrid_pipeline.pipeline import PipelineConfig
    pipeline_config_obj = PipelineConfig(base_dir=hybrid_dir)
    pipeline = HybridLearningPipeline(pipeline_config_obj)
    logger.info("パイプライン作成完了")

    # 検証用に10タスクのみ生成
    num_programs = 10

    print(f"\n=== 部分プログラムフロー検証 ===")
    print(f"USE_PARTIAL_PROGRAM_FLOW = {os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'not set')}")
    print(f"生成タスク数: {num_programs}")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    logger.info(f"検証開始: {num_programs}プログラム生成")

    try:
        # Phase1: DataPair生成
        data_pairs = pipeline.run_phase1_pair_generation(
            num_programs=num_programs,
            pairs_per_program=1,  # 設計B: 1プログラム=1ペア
            max_attempts_per_pair=50
        )

        # 統計情報を計算
        unique_programs = len(set(p.program for p in data_pairs)) if data_pairs else 0
        complexity_counts = {}
        for pair in data_pairs:
            comp = pair.get_program_complexity()
            complexity_counts[comp] = complexity_counts.get(comp, 0) + 1

        # レポートを出力
        print("\n=== 検証結果 ===")
        print(f"生成されたDataPair数: {len(data_pairs)}")
        print(f"ユニークなプログラム数: {unique_programs}")
        if len(data_pairs) > 0:
            print(f"多様性率: {unique_programs / len(data_pairs) * 100:.1f}%")
        else:
            print("多様性率: N/A (DataPairが0個)")
        print(f"複雑度分布: {complexity_counts}")
        print(f"出力ディレクトリ: {output_dir}")

        logger.info("検証完了")
        logger.info(f"生成されたDataPair数: {len(data_pairs)}")
        logger.info(f"ユニークなプログラム数: {unique_programs}")
        logger.info(f"複雑度分布: {complexity_counts}")

        # 結果をファイルに保存
        result_file = os.path.join(output_dir, "verification_result.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=== 部分プログラムフロー検証結果 ===\n")
            f.write(f"USE_PARTIAL_PROGRAM_FLOW: {os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'not set')}\n")
            f.write(f"生成タスク数: {num_programs}\n")
            f.write(f"生成されたDataPair数: {len(data_pairs)}\n")
            f.write(f"ユニークなプログラム数: {unique_programs}\n")
            if len(data_pairs) > 0:
                f.write(f"多様性率: {unique_programs / len(data_pairs) * 100:.1f}%\n")
            f.write(f"複雑度分布: {complexity_counts}\n")
            f.write(f"出力ディレクトリ: {output_dir}\n")

        print(f"\n検証結果を保存: {result_file}")

        return len(data_pairs) > 0

    except Exception as e:
        print(f"\n[エラー] 検証中にエラーが発生しました: {e}")
        logger.error(f"検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
