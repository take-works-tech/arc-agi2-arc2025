"""
新規実装モジュールの統合訓練スクリプト

すべての新規モジュール（Object Graph + GNN, NGPS, Relation Classifier, Color Role Classifier）を
順次訓練する統合スクリプト

使い方:
    python scripts/production/training/train_all_new_modules.py \\
        [--max-tasks N] \\
        [--skip-data-generation] \\
        [--skip-training MODULE_NAME]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: List[str], description: str) -> bool:
    """コマンドを実行"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"実行コマンド: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[OK] {description} 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[NG] {description} 失敗: {e}")
        return False
    except Exception as e:
        print(f"[NG] {description} エラー: {e}")
        return False


def generate_training_data(max_tasks: Optional[int] = None) -> Dict[str, bool]:
    """訓練データを生成"""
    results = {}

    base_output_dir = project_root / "learning_outputs"

    # Object Graph + GNN
    output_path = base_output_dir / "object_graph" / "train_data.jsonl"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "experimental" / "data_generation" / "generate_object_graph_training_data.py"),
        str(output_path)
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])
    results['object_graph'] = run_command(cmd, "Object Graph + GNN 訓練データ生成")

    # NGPS / DSL Selector
    output_path = base_output_dir / "ngps" / "train_data.jsonl"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "experimental" / "data_generation" / "generate_ngps_training_data.py"),
        str(output_path)
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])
    results['ngps'] = run_command(cmd, "NGPS / DSL Selector 訓練データ生成")

    # Relation Classifier
    output_path = base_output_dir / "relation_classifier" / "train_data.jsonl"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "experimental" / "data_generation" / "generate_relation_classifier_data.py"),
        str(output_path)
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])
    results['relation_classifier'] = run_command(cmd, "Relation Classifier 訓練データ生成")

    # Color Role Classifier
    output_path = base_output_dir / "color_role" / "train_data.jsonl"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "experimental" / "data_generation" / "generate_color_role_data.py"),
        str(output_path)
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])
    results['color_role'] = run_command(cmd, "Color Role Classifier 訓練データ生成")

    return results


def train_models(skip_modules: List[str] = None) -> Dict[str, bool]:
    """モデルを訓練"""
    if skip_modules is None:
        skip_modules = []

    results = {}
    base_output_dir = project_root / "learning_outputs"

    # Object Graph + GNN
    if 'object_graph' not in skip_modules:
        train_data = base_output_dir / "object_graph" / "train_data.jsonl"
        if train_data.exists():
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "experimental" / "training" / "train_object_graph_encoder.py"),
                str(train_data),
                "--output-dir", str(base_output_dir / "object_graph_encoder"),
                "--epochs", "10",  # テスト用に少なめ
                "--batch-size", "4"
            ]
            results['object_graph'] = run_command(cmd, "Object Graph + GNN 訓練")
        else:
            print(f"[SKIP] 訓練データが見つかりません: {train_data}")
            results['object_graph'] = False
    else:
        print("[SKIP] Object Graph + GNN 訓練をスキップ")
        results['object_graph'] = None

    # NGPS / DSL Selector
    if 'ngps' not in skip_modules:
        train_data = base_output_dir / "ngps" / "train_data.jsonl"
        if train_data.exists():
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "experimental" / "training" / "train_ngps.py"),
                str(train_data),
                "--output-dir", str(base_output_dir / "ngps"),
                "--epochs", "10",
                "--batch-size", "16"
            ]
            results['ngps'] = run_command(cmd, "NGPS / DSL Selector 訓練")
        else:
            print(f"[SKIP] 訓練データが見つかりません: {train_data}")
            results['ngps'] = False
    else:
        print("[SKIP] NGPS / DSL Selector 訓練をスキップ")
        results['ngps'] = None

    # Relation Classifier
    if 'relation_classifier' not in skip_modules:
        train_data = base_output_dir / "relation_classifier" / "train_data.jsonl"
        if train_data.exists():
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "experimental" / "training" / "train_relation_classifier.py"),
                str(train_data),
                "--output-dir", str(base_output_dir / "relation_classifier"),
                "--epochs", "10",
                "--batch-size", "16"
            ]
            results['relation_classifier'] = run_command(cmd, "Relation Classifier 訓練")
        else:
            print(f"[SKIP] 訓練データが見つかりません: {train_data}")
            results['relation_classifier'] = False
    else:
        print("[SKIP] Relation Classifier 訓練をスキップ")
        results['relation_classifier'] = None

    # Color Role Classifier
    if 'color_role' not in skip_modules:
        train_data = base_output_dir / "color_role" / "train_data.jsonl"
        if train_data.exists():
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "experimental" / "training" / "train_color_role_classifier.py"),
                str(train_data),
                "--output-dir", str(base_output_dir / "color_role_classifier"),
                "--epochs", "10",
                "--batch-size", "16"
            ]
            results['color_role'] = run_command(cmd, "Color Role Classifier 訓練")
        else:
            print(f"[SKIP] 訓練データが見つかりません: {train_data}")
            results['color_role'] = False
    else:
        print("[SKIP] Color Role Classifier 訓練をスキップ")
        results['color_role'] = None

    return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='新規実装モジュールの統合訓練')
    parser.add_argument('--max-tasks', type=int, default=10, help='最大タスク数（テスト用）')
    parser.add_argument('--skip-data-generation', action='store_true', help='データ生成をスキップ')
    parser.add_argument('--skip-training', nargs='+', choices=['object_graph', 'ngps', 'relation_classifier', 'color_role'],
                        help='訓練をスキップするモジュール')
    parser.add_argument('--data-only', action='store_true', help='データ生成のみ実行')
    parser.add_argument('--train-only', action='store_true', help='訓練のみ実行（データ生成をスキップ）')

    args = parser.parse_args()

    print("="*60)
    print("新規実装モジュールの統合訓練")
    print("="*60)
    print(f"最大タスク数: {args.max_tasks}")
    print(f"データ生成スキップ: {args.skip_data_generation or args.train_only}")
    print(f"訓練スキップ: {args.skip_training or args.data_only}")
    print()

    start_time = time.time()
    results = {}

    # 1. 訓練データ生成
    if not (args.skip_data_generation or args.train_only):
        print("\n" + "="*60)
        print("ステップ1: 訓練データ生成")
        print("="*60)
        data_results = generate_training_data(max_tasks=args.max_tasks)
        results['data_generation'] = data_results

        # 結果サマリー
        success_count = sum(1 for v in data_results.values() if v)
        total_count = len([v for v in data_results.values() if v is not None])
        print(f"\nデータ生成結果: {success_count}/{total_count} 成功")

        if args.data_only:
            print("\nデータ生成のみ実行しました")
            return
    else:
        print("\n[SKIP] 訓練データ生成をスキップします")

    # 2. モデル訓練
    if not args.data_only:
        print("\n" + "="*60)
        print("ステップ2: モデル訓練")
        print("="*60)
        skip_modules = args.skip_training or []
        train_results = train_models(skip_modules=skip_modules)
        results['training'] = train_results

        # 結果サマリー
        success_count = sum(1 for v in train_results.values() if v)
        total_count = len([v for v in train_results.values() if v is not None])
        print(f"\n訓練結果: {success_count}/{total_count} 成功")

    # 最終サマリー
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("最終サマリー")
    print("="*60)
    print(f"実行時間: {elapsed_time:.1f}秒")

    if 'data_generation' in results:
        print("\nデータ生成:")
        for module, success in results['data_generation'].items():
            status = "[OK]" if success else "[NG]"
            print(f"  {status} {module}")

    if 'training' in results:
        print("\n訓練:")
        for module, success in results['training'].items():
            if success is None:
                status = "[SKIP]"
            elif success:
                status = "[OK]"
            else:
                status = "[NG]"
            print(f"  {status} {module}")


if __name__ == "__main__":
    main()
