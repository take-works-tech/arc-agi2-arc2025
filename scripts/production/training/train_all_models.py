"""
統合学習パイプライン

全モデルの学習を一括実行するオプションスクリプト
設定ファイルで学習するモデルを選択可能
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def load_config(config_path: str = 'configs/default_config.yaml') -> Dict[str, Any]:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_script(script_path: str, description: str, check_existence: bool = True) -> bool:
    """
    スクリプトを実行

    Args:
        script_path: 実行するスクリプトのパス
        description: 説明
        check_existence: ファイルの存在をチェックするか

    Returns:
        成功した場合True
    """
    script_file = Path(script_path)
    if check_existence and not script_file.exists():
        print(f"⚠️  スキップ: {description}（スクリプトが見つかりません: {script_path}）")
        return False

    print(f"\n{'='*60}")
    print(f"開始: {description}")
    print(f"スクリプト: {script_path}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=Path(__file__).parent.parent
        )
        print(f"✅ 完了: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {description}")
        print(f"   エラーコード: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ エラー: {description}")
        print(f"   エラー: {e}")
        return False


def run_program_scorer_training(config: Dict[str, Any]) -> bool:
    """
    ProgramScorerの学習を実行

    Args:
        config: 設定

    Returns:
        成功した場合True
    """
    # データ生成スクリプトのパスを確認
    generate_script = Path(__file__).parent.parent / "data_generation" / "generate_program_scorer_data.py"
    if not generate_script.exists():
        print(f"⚠️  スキップ: ProgramScorerデータ生成（スクリプトが見つかりません）")
        return False

    # データ生成
    print(f"\n{'='*60}")
    print(f"開始: ProgramScorerデータ生成")
    print(f"{'='*60}")
    try:
        subprocess.run(
            [sys.executable, str(generate_script)],
            check=True,
            cwd=Path(__file__).parent.parent
        )
        print(f"✅ 完了: ProgramScorerデータ生成")
    except Exception as e:
        print(f"❌ エラー: ProgramScorerデータ生成: {e}")
        return False

    # 学習データのパスを取得（デフォルト値）
    train_data_path = "learning_outputs/program_scorer/train_data_latest.jsonl"
    model_out_path = "learning_outputs/program_scorer/program_scorer_latest.pt"

    # 学習スクリプトを実行
    train_script = Path(__file__).parent / "train_program_scorer.py"
    if not train_script.exists():
        print(f"⚠️  スキップ: ProgramScorer学習（スクリプトが見つかりません）")
        return False

    print(f"\n{'='*60}")
    print(f"開始: ProgramScorer学習")
    print(f"データ: {train_data_path}")
    print(f"出力: {model_out_path}")
    print(f"{'='*60}")

    try:
        subprocess.run(
            [sys.executable, str(train_script), train_data_path, model_out_path],
            check=True,
            cwd=Path(__file__).parent.parent
        )
        print(f"✅ 完了: ProgramScorer学習")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: ProgramScorer学習")
        print(f"   エラーコード: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ エラー: ProgramScorer学習: {e}")
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="統合学習パイプライン: 全モデルの学習を一括実行"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='データ生成をスキップ（既にデータが存在する場合）'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['phase1', 'object_based', 'program_scorer', 'all'],
        default=['all'],
        help='学習するモデルを指定（複数指定可能、デフォルト: all）'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='エラーが発生しても続行'
    )

    args = parser.parse_args()

    # 設定を読み込み
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みに失敗: {e}")
        sys.exit(1)

    print("="*60)
    print("統合学習パイプライン")
    print("="*60)
    print(f"設定ファイル: {args.config}")
    print(f"学習モデル: {', '.join(args.models)}")
    print("="*60)

    # 学習するモデルを決定
    if 'all' in args.models:
        models_to_train = ['phase1', 'object_based', 'program_scorer']
    else:
        models_to_train = args.models

    results: Dict[str, bool] = {}

    # 1. データ生成（スキップされていない場合）
    if not args.skip_data_generation:
        print(f"\n{'='*60}")
        print("ステップ1: データ生成")
        print(f"{'='*60}")
        results['data_generation'] = run_script(
            'scripts/production/data_generation/generate_data.py',
            'データ生成（DataPair生成）'
        )
        if not results['data_generation'] and not args.continue_on_error:
            print("\n❌ データ生成に失敗しました。処理を中断します。")
            sys.exit(1)
    else:
        print("\n⏭️  データ生成をスキップします")
        results['data_generation'] = True

    # 2. 各モデルの学習
    print(f"\n{'='*60}")
    print("ステップ2: モデル学習")
    print(f"{'='*60}")

    scripts_map = {
        'phase1': ('scripts/production/training/train_program_synthesis.py', 'プログラム合成モデル学習（ProgramSynthesisModel）'),
        'object_based': ('scripts/production/training/train_object_based.py', 'オブジェクトベースモデル学習'),
    }

    for model_name in models_to_train:
        if model_name == 'program_scorer':
            # ProgramScorerは特別な処理が必要
            results['program_scorer'] = run_program_scorer_training(config)
        elif model_name in scripts_map:
            script_path, description = scripts_map[model_name]
            results[model_name] = run_script(script_path, description)

            if not results[model_name] and not args.continue_on_error:
                print(f"\n❌ {description}に失敗しました。処理を中断します。")
                sys.exit(1)

    # 結果サマリー
    print(f"\n{'='*60}")
    print("学習結果サマリー")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {name}: {status}")

    # 全体の成功/失敗を判定
    all_success = all(results.values())
    if all_success:
        print(f"\n✅ 全モデルの学習が完了しました！")
        sys.exit(0)
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"\n⚠️  一部のモデル学習が失敗しました: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
