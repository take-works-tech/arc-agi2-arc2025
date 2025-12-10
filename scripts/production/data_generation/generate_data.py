"""
データ生成スクリプト

ハイブリッドアプローチのデータセットを生成
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
# scripts/production/data_generation/generate_data.py から4階層上
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 部分プログラムフローを有効化（環境変数が設定されていない場合のみ）
# 既に設定されている場合はその値を優先（CI/CDなどで外部から制御可能）
if 'USE_PARTIAL_PROGRAM_FLOW' not in os.environ:
    os.environ['USE_PARTIAL_PROGRAM_FLOW'] = 'true'

import yaml
from src.hybrid_system.learning.hybrid_pipeline import HybridLearningPipeline
from src.data_systems.generator.program_executor.performance_profiler import get_profiler

# Loggerは直接インポート（utils/__init__.pyのvisualization依存を回避）
logger = None
def get_logger(log_dir):
    global logger
    if logger is None:
        try:
            # utils/__init__.pyを経由せず、直接loggingモジュールからインポート
            from src.hybrid_system.utils.logging.logger import Logger
            logger = Logger.get_logger("data_generation", log_dir)
        except Exception as e:
            # Loggerの初期化に失敗した場合はNoneを返す（print文で代替）
            print(f"警告: Loggerの初期化に失敗しました: {e}")
            logger = None
    return logger


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='データ生成スクリプト')
    parser.add_argument(
        '--num-programs',
        type=int,
        default=None,
        help='生成するプログラム数（デフォルト: 設定ファイルから読み込み）'
    )
    args = parser.parse_args()
    """メイン処理"""
    # 設定を読み込み
    config_path = project_root / 'configs' / 'default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ロガーを初期化（オプショナル）
    logger = get_logger(config['logging']['log_dir'])
    if logger:
        logger.info("データ生成開始")
    else:
        print("データ生成開始（Loggerは使用できません）")

    # タイムスタンプ付きディレクトリを作成
    base_output_dir = project_root / "outputs"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ付きディレクトリ内にhybridサブディレクトリを作成
    hybrid_dir = output_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    print(f"新規出力ディレクトリを作成: {output_dir}")
    if logger:
        logger.info(f"出力ディレクトリ: {output_dir}")
    # Pathオブジェクトを文字列に変換（後続の処理で使用）
    output_dir_str = str(output_dir)
    hybrid_dir_str = str(hybrid_dir)

    # 環境変数の設定を表示
    use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'not set')
    print(f"USE_PARTIAL_PROGRAM_FLOW = {use_partial_program_flow}")
    if logger:
        logger.info(f"USE_PARTIAL_PROGRAM_FLOW = {use_partial_program_flow}")

    # パイプライン設定（タイムスタンプ付きディレクトリ内のhybridディレクトリを使用）
    from src.hybrid_system.learning.hybrid_pipeline.pipeline import PipelineConfig
    pipeline_config_obj = PipelineConfig(base_dir=hybrid_dir_str)
    pipeline = HybridLearningPipeline(pipeline_config_obj)
    if logger:
        logger.info("パイプライン作成完了")

    # Phase1のみ実行（DataPair生成: 設計B - 1プログラム=1ペア）
    # コマンドライン引数が指定されている場合はそれを優先
    if args.num_programs is not None:
        num_programs = args.num_programs
        if logger:
            logger.info(f"コマンドライン引数で指定: num_programs={num_programs}")
    else:
        num_programs = config['data']['generation']['num_programs']
        if logger:
            logger.info(f"設定ファイルから読み込み: num_programs={num_programs}")

    if logger:
        logger.info(f"データ生成を開始: {num_programs}プログラム → {num_programs} DataPair（1プログラム=1ペア）")

    # プロファイラーを取得
    profiler = get_profiler()

    # 全体実行時間の計測開始
    overall_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"【データ生成開始】")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

    # Phase1: DataPair生成（設計B: 1プログラム=1ペア）
    phase1_start = time.time()
    data_pairs = pipeline.run_phase1_pair_generation(
        num_programs=num_programs,
        pairs_per_program=1,  # 設計B: 1プログラム=1ペア
        max_attempts_per_pair=50
    )
    phase1_elapsed = time.time() - phase1_start
    profiler.record_timing("phase1_pair_generation", phase1_elapsed, "overall")

    # 全体実行時間の計測終了
    overall_end_time = time.time()
    overall_elapsed = overall_end_time - overall_start_time
    overall_minutes = overall_elapsed / 60
    overall_hours = overall_minutes / 60

    # パイプライン内で保存済みなので、統計のみ計算

    # 統計情報を計算
    unique_programs = len(set(p.program for p in data_pairs))
    complexity_counts = {}
    for pair in data_pairs:
        comp = pair.get_program_complexity()
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1

    # レポートを出力
    if logger:
        logger.info("データ生成完了")
        logger.info(f"生成されたDataPair数: {len(data_pairs)}")
        logger.info(f"ユニークなプログラム数: {unique_programs}")
        logger.info(f"複雑度分布: {complexity_counts}")

    # プロファイラーの統計情報を取得
    profiler = get_profiler()
    profiler_stats = profiler.get_statistics()

    print(f"\n{'='*80}")
    print(f"【データ生成完了】")
    print(f"{'='*80}")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"全体実行時間: {overall_elapsed:.3f}秒 ({overall_minutes:.2f}分 / {overall_hours:.2f}時間)")
    print(f"出力ディレクトリ: {output_dir_str}")
    print(f"DataPair: {len(data_pairs)}個")
    print(f"ユニークなプログラム: {unique_programs}種類")
    if len(data_pairs) > 0:
        print(f"多様性率: {unique_programs / len(data_pairs) * 100:.1f}%")
        print(f"1タスクあたりの平均時間: {overall_elapsed / len(data_pairs):.3f}秒")
    else:
        print("多様性率: N/A (DataPairが0個)")
    print(f"複雑度分布: {complexity_counts}")

    # プロファイラーの統計情報を表示（処理時間が長いものから順に）
    if profiler_stats:
        print(f"\n{'='*80}")
        print(f"【詳細な処理時間統計】")
        print(f"{'='*80}")
        # 合計時間でソート
        sorted_stats = sorted(profiler_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        for name, stats in sorted_stats:
            if stats['count'] > 0:
                print(f"{name}:")
                print(f"  合計: {stats['total']:.3f}秒 ({stats['total']/60:.2f}分)")
                print(f"  平均: {stats['average']:.3f}秒")
                print(f"  最大: {stats['max']:.3f}秒")
                print(f"  最小: {stats['min']:.3f}秒")
                print(f"  回数: {stats['count']}")
        print(f"{'='*80}")

    if logger:
        logger.info(f"全体実行時間: {overall_elapsed:.3f}秒 ({overall_minutes:.2f}分)")


if __name__ == "__main__":
    main()
