"""
生成されたデータの動作検証スクリプト

既存のgenerate_data.pyで生成されたデータを検証します。
"""

import sys
import os
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.hybrid_system.learning.hybrid_pipeline.io import DatasetIO


def validate_data_pairs(data_pairs: List, expected_count: int = 10) -> Dict[str, Any]:
    """DataPairの検証

    Args:
        data_pairs: DataPairのリスト
        expected_count: 期待されるタスク数

    Returns:
        検証結果の辞書
    """
    results = {
        'success': True,
        'errors': [],
        'warnings': [],
        'task_count': len(data_pairs),
        'valid_tasks': 0,
        'invalid_tasks': 0,
        'complexity_distribution': {},
        'unique_programs': 0
    }

    if len(data_pairs) == 0:
        results['success'] = False
        results['errors'].append("DataPairが0個です")
        return results

    if len(data_pairs) < expected_count:
        results['warnings'].append(
            f"期待されるタスク数({expected_count})より少ないです: {len(data_pairs)}個"
        )

    # 各DataPairの検証
    unique_programs = set()
    for i, pair in enumerate(data_pairs):
        is_valid = True

        # グリッドデータの確認
        if not hasattr(pair, 'input') or pair.input is None:
            results['errors'].append(f"DataPair {i+1}: inputが欠損しています")
            results['success'] = False
            is_valid = False

        if not hasattr(pair, 'output') or pair.output is None:
            results['errors'].append(f"DataPair {i+1}: outputが欠損しています")
            results['success'] = False
            is_valid = False

        # プログラムの確認
        if not hasattr(pair, 'program') or not pair.program:
            results['errors'].append(f"DataPair {i+1}: programが欠損しています")
            results['success'] = False
            is_valid = False
        else:
            unique_programs.add(pair.program)

        # 複雑度の確認
        try:
            complexity = pair.get_program_complexity()
            results['complexity_distribution'][complexity] = results['complexity_distribution'].get(complexity, 0) + 1
        except Exception as e:
            results['warnings'].append(f"DataPair {i+1}: 複雑度の取得に失敗: {e}")

        if is_valid:
            results['valid_tasks'] += 1
        else:
            results['invalid_tasks'] += 1

    results['unique_programs'] = len(unique_programs)

    return results


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description='生成されたデータの動作検証')
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='検証する出力ディレクトリ（例: outputs/20251209_050423）'
    )
    parser.add_argument(
        '--expected-count',
        type=int,
        default=10,
        help='期待されるタスク数（デフォルト: 10）'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"エラー: 出力ディレクトリが存在しません: {output_dir}")
        sys.exit(1)

    hybrid_dir = output_dir / "hybrid"
    if not hybrid_dir.exists():
        print(f"エラー: hybridディレクトリが存在しません: {hybrid_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"【生成データの動作検証】")
    print(f"{'='*80}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"期待されるタスク数: {args.expected_count}")
    print(f"{'='*80}\n")

    # データの読み込み
    try:
        dataset_io = DatasetIO(str(hybrid_dir))
        phase1_files = dataset_io.list_saved_files('phase1')

        if not phase1_files.get('phase1'):
            print("エラー: phase1データファイルが見つかりません")
            sys.exit(1)

        # ファイル名から完全なパスを構築
        phase1_dir = hybrid_dir / "phase1_pairs"
        file_paths = [str(phase1_dir / fname) for fname in phase1_files['phase1']]
        latest_file = max(file_paths, key=os.path.getmtime)
        print(f"データファイル: {latest_file}")

        data_pairs = dataset_io.load_data_pairs(latest_file)
        print(f"読み込んだDataPair数: {len(data_pairs)}\n")

    except Exception as e:
        print(f"エラー: データの読み込みに失敗しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 検証の実行
    validation_results = validate_data_pairs(data_pairs, args.expected_count)

    # 結果の表示
    print(f"{'='*80}")
    print(f"【検証結果】")
    print(f"{'='*80}")
    print(f"成功: {'OK' if validation_results['success'] else 'NG'}")
    print(f"タスク数: {validation_results['task_count']}")
    print(f"有効なタスク: {validation_results['valid_tasks']}")
    print(f"無効なタスク: {validation_results['invalid_tasks']}")
    print(f"ユニークなプログラム数: {validation_results['unique_programs']}")
    print(f"複雑度分布: {validation_results['complexity_distribution']}")

    if validation_results['errors']:
        print(f"\nエラー ({len(validation_results['errors'])}件):")
        for error in validation_results['errors']:
            print(f"  - {error}")

    if validation_results['warnings']:
        print(f"\n警告 ({len(validation_results['warnings'])}件):")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")

    print(f"{'='*80}\n")

    # 検証結果をJSONファイルに保存
    validation_output_file = output_dir / "validation_results.json"
    with open(validation_output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)

    print(f"検証結果を保存: {validation_output_file}\n")

    if not validation_results['success']:
        print("検証が失敗しました。")
        sys.exit(1)
    else:
        print("検証が成功しました。")
        sys.exit(0)


if __name__ == "__main__":
    main()
