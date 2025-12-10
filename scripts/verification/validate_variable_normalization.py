"""
変数インデックス正規化と未使用変数削除の検証スクリプト

100タスクで検証を実行する
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data_systems.generator.program_executor.node_analyzer.variable_index_normalizer import normalize_variable_indices
from src.data_systems.generator.program_executor.node_analyzer.variable_normalization_validator import validate_variable_normalization
from src.core_systems.executor.executor import Executor
from src.hybrid_system.learning.hybrid_pipeline.pipeline import HybridLearningPipeline


def main():
    # 検証モードを有効化
    os.environ['ENABLE_VARIABLE_NORMALIZATION_VALIDATION'] = 'true'

    print("=" * 80)
    print("変数インデックス正規化と未使用変数削除の検証")
    print("=" * 80)
    print()

    # パイプラインを初期化
    pipeline = HybridLearningPipeline()

    # 実行器を初期化
    executor = Executor()

    # 統計情報
    total_tasks = 0
    successful_validations = 0
    failed_validations = 0
    execution_errors = 0

    failed_cases = []

    print("データ生成を開始して、各タスクのプログラムを検証します...")
    print()

    # データ生成パイプラインを使用してタスクを生成
    try:
        # 10タスクずつ生成して検証
        for batch_idx in range(10):  # 10バッチ × 10タスク = 100タスク
            print(f"[バッチ {batch_idx + 1}/10] データ生成開始...", flush=True)

            # 10タスク生成
            pipeline.run_phase1_pair_generation(num_tasks=10, num_pairs=5)

            # 生成されたタスクを取得して検証
            # 注意: パイプラインから直接タスクを取得する方法が必要
            # ここでは、生成されたJSONファイルから読み込む必要がある

            print(f"[バッチ {batch_idx + 1}/10] 検証は次のステップで実装...", flush=True)

    except KeyboardInterrupt:
        print("\n検証が中断されました", flush=True)
    except Exception as e:
        print(f"\n検証中にエラーが発生しました: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 結果サマリー
    print()
    print("=" * 80)
    print("検証結果サマリー")
    print("=" * 80)
    print(f"総タスク数: {total_tasks}")
    print(f"検証成功: {successful_validations}")
    print(f"検証失敗: {failed_validations}")
    print(f"実行エラー: {execution_errors}")
    print()

    if failed_cases:
        print("失敗したケース:")
        for i, case in enumerate(failed_cases[:10], 1):  # 最初の10件のみ表示
            print(f"  {i}. {case}")
        if len(failed_cases) > 10:
            print(f"  ... 他 {len(failed_cases) - 10} 件")

    print("=" * 80)


if __name__ == "__main__":
    main()

