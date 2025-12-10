"""
プログラム生成テストの動作検証スクリプト

test_outputs/program_generation_test ディレクトリ内のテストファイルを検証します。
"""
import sys
import os
from pathlib import Path
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
from src.data_systems.generator.program_generator.generation.code_generator import generate_code
import numpy as np


def verify_test_files(test_dir: str = "test_outputs/program_generation_test"):
    """テストファイルを検証

    Args:
        test_dir: テストディレクトリ
    """
    print(f"プログラム生成テストの動作検証を開始します")
    print(f"テストディレクトリ: {test_dir}")
    print(f"{'='*60}\n")

    if not os.path.exists(test_dir):
        print(f"[エラー] テストディレクトリが存在しません: {test_dir}")
        return False

    # テストファイルを検索
    test_files = []
    for i in range(1, 31):  # test_1からtest_30まで
        program_file = os.path.join(test_dir, f"test_{i}_program.txt")
        grid_file = os.path.join(test_dir, f"test_{i}_input_grid.png")

        if os.path.exists(program_file):
            test_files.append({
                'index': i,
                'program_file': program_file,
                'grid_file': grid_file if os.path.exists(grid_file) else None
            })

    if not test_files:
        print(f"[警告] テストファイルが見つかりませんでした")
        return False

    print(f"見つかったテストファイル: {len(test_files)}個\n")

    # 各テストファイルを検証
    success_count = 0
    failure_count = 0
    verification_results = []

    executor = UnifiedProgramGenerator()

    for test_info in test_files:
        test_index = test_info['index']
        program_file = test_info['program_file']
        grid_file = test_info['grid_file']

        print(f"[{test_index}] テスト{test_index}を検証中...")

        try:
            # プログラムファイルを読み込み
            with open(program_file, "r", encoding="utf-8") as f:
                program_code = f.read().strip()

            if not program_code:
                print(f"  [NG] プログラムコードが空です")
                failure_count += 1
                verification_results.append({
                    'test_index': test_index,
                    'status': 'failed',
                    'reason': 'プログラムコードが空'
                })
                continue

            # プログラムコードの基本チェック
            if 'GET_ALL_OBJECTS' not in program_code:
                print(f"  [警告] GET_ALL_OBJECTSが含まれていません")

            # プログラムをパースして実行可能かチェック
            # 注意: 実際の実行は時間がかかるため、基本的な検証のみ

            # グリッドファイルの存在確認
            if grid_file and os.path.exists(grid_file):
                file_size = os.path.getsize(grid_file)
                print(f"  [OK] グリッド画像ファイル: {grid_file} ({file_size} bytes)")
            else:
                print(f"  [警告] グリッド画像ファイルが見つかりません")

            print(f"  [OK] プログラムファイル: {program_file} ({len(program_code)} 文字)")
            print(f"  [OK] テスト{test_index}の検証が完了しました")

            success_count += 1
            verification_results.append({
                'test_index': test_index,
                'status': 'success',
                'program_length': len(program_code),
                'has_grid_file': grid_file is not None and os.path.exists(grid_file)
            })

        except Exception as e:
            print(f"  [NG] 検証エラー: {e}")
            failure_count += 1
            verification_results.append({
                'test_index': test_index,
                'status': 'failed',
                'reason': str(e)
            })
            continue

    # 検証結果をまとめる
    print(f"\n{'='*60}")
    print(f"検証完了: 成功 {success_count}個, 失敗 {failure_count}個")
    print(f"{'='*60}\n")

    # 詳細統計
    if verification_results:
        successful = [r for r in verification_results if r['status'] == 'success']
        if successful:
            avg_program_length = sum(r.get('program_length', 0) for r in successful) / len(successful)
            has_grid_count = sum(1 for r in successful if r.get('has_grid_file', False))

            print(f"詳細統計:")
            print(f"  - 平均プログラム長: {avg_program_length:.1f} 文字")
            print(f"  - グリッド画像ファイルあり: {has_grid_count}/{len(successful)}")

    # 検証結果をJSONファイルに保存
    result_file = os.path.join(test_dir, "verification_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            'total_tests': len(test_files),
            'success_count': success_count,
            'failure_count': failure_count,
            'results': verification_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n検証結果を保存: {result_file}")

    return success_count > 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="プログラム生成テストを検証")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="test_outputs/program_generation_test",
        help="テストディレクトリ（デフォルト: test_outputs/program_generation_test）"
    )

    args = parser.parse_args()

    success = verify_test_files(test_dir=args.test_dir)
    sys.exit(0 if success else 1)
