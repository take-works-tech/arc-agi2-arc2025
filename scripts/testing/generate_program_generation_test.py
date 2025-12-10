"""
プログラム生成テストの再生成スクリプト

test_outputs/program_generation_test ディレクトリ内のテストファイルを再生成します。
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
from src.data_systems.generator.grid_visualizer import save_single_grid_to_png
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_generator.generation.code_generator import generate_code

import numpy as np

# デバッグ用のログ出力フラグ
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')


def generate_test_programs(num_tests: int = 30, output_dir: str = "test_outputs/program_generation_test"):
    """テストプログラムを生成

    Args:
        num_tests: 生成するテスト数（デフォルト: 30）
        output_dir: 出力ディレクトリ
    """
    print(f"プログラム生成テストを開始します（{num_tests}個のテスト）")
    print(f"出力ディレクトリ: {output_dir}")

    # 部分プログラムフローの使用状況を確認
    use_partial_program_flow_env = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true')
    print(f"環境変数 USE_PARTIAL_PROGRAM_FLOW: {use_partial_program_flow_env}")

    # 部分プログラムフローのインポートを試行
    try:
        from src.data_systems.generator.partial_program_helper import (
            generate_partial_program_from_input_grid,
            parse_partial_program_to_nodes
        )
        print(f"✓ 部分プログラムフローのインポートに成功しました")
        partial_program_flow_available = True
    except ImportError as e:
        print(f"✗ 部分プログラムフローのインポートに失敗しました: {e}")
        print(f"  通常フローを使用します")
        partial_program_flow_available = False

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    generator = UnifiedProgramGenerator()

    success_count = 0
    failure_count = 0

    for test_index in range(1, num_tests + 1):
        max_retries = 10  # 最大再試行回数
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"  [再試行 {retry_count}/{max_retries-1}] テスト{test_index}を再生成中...")
                else:
                    print(f"\n[{test_index}/{num_tests}] テスト{test_index}を生成中...")

                # 複雑度を決定
                import random
                complexity = random.randint(1, 3)  # 複雑度をランダムに選択（1-3の範囲）

                # 環境変数で部分プログラムフローを有効化（デフォルト: true）
                # 検証では部分プログラムフローを必ず使用する
                use_partial_program_flow = os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true').lower() in ('true', '1', 'yes')

                if use_partial_program_flow:
                    print(f"  [情報] テスト{test_index}: 部分プログラムフローを使用します（環境変数: USE_PARTIAL_PROGRAM_FLOW={os.environ.get('USE_PARTIAL_PROGRAM_FLOW', 'true')}）")
                    print(f"  [情報] テスト{test_index}: 部分プログラムフローを使用します")
                    # 部分プログラムフローを使用（遅延インポートで依存関係の問題を回避）
                    try:
                        from src.data_systems.generator.partial_program_helper import (
                            generate_partial_program_from_input_grid,
                            parse_partial_program_to_nodes
                        )

                        # 1. グリッドサイズ決定
                        grid_width, grid_height = decide_grid_size()

                        # 2. 仮のインプットグリッド生成（プログラムなし）
                        _, input_grid, _, _ = core_executor_main(
                            nodes=None,
                            grid_width=grid_width,
                            grid_height=grid_height,
                            enable_replacement=False
                        )

                        if input_grid is None:
                            print(f"  [警告] テスト{test_index}: インプットグリッド生成に失敗しました。")
                            retry_count += 1
                            continue

                        # 3. 部分プログラム生成
                        partial_program, category_var_mapping = generate_partial_program_from_input_grid(
                            input_grid=input_grid,
                            grid_width=grid_width,
                            grid_height=grid_height
                        )

                        if partial_program is None:
                            print(f"  [警告] テスト{test_index}: 部分プログラム生成に失敗しました。")
                            retry_count += 1
                            continue

                        # 4. 部分プログラムをNodeリストに変換
                        context = ProgramContext(complexity, grid_width=grid_width, grid_height=grid_height)
                        partial_nodes, category_var_mapping = parse_partial_program_to_nodes(
                            partial_program=partial_program,
                            category_var_mapping=category_var_mapping or {},
                            context=context
                        )

                        # 5. プログラム生成器で続きを生成
                        nodes = generator.generate_program_nodes_from_partial(
                            partial_nodes=partial_nodes,
                            category_var_mapping=category_var_mapping or {},
                            context=context
                        )

                        if not nodes or len(nodes) == 0:
                            print(f"  [警告] テスト{test_index}: プログラムノードが空です。")
                            retry_count += 1
                            continue

                        # 6. プログラムコードを生成
                        program_code = generate_code(nodes, context)

                        if not program_code or not program_code.strip():
                            print(f"  [警告] テスト{test_index}: プログラムコードが空です。")
                            retry_count += 1
                            continue

                        # GET_ALL_OBJECTSが含まれているか確認（必須）
                        if 'GET_ALL_OBJECTS' not in program_code:
                            print(f"  [警告] テスト{test_index}: プログラムコードにGET_ALL_OBJECTSが含まれていません。")
                            retry_count += 1
                            continue

                        # 7. プログラムを実行してアウトプットグリッドを取得
                        result_nodes, _, output_grid, trace_results = core_executor_main(
                            nodes=nodes,
                            grid_width=grid_width,
                            grid_height=grid_height,
                            task_index=test_index,
                            enable_replacement=False,
                            is_first_pair=True
                        )

                        if output_grid is None:
                            print(f"  [警告] テスト{test_index}: アウトプットグリッド生成に失敗しました。")
                            retry_count += 1
                            continue

                    except ImportError as e:
                        print(f"  [警告] テスト{test_index}: 部分プログラムフローのインポートに失敗しました。通常フローにフォールバックします: {e}")
                        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                            import traceback
                            traceback.print_exc()
                        use_partial_program_flow = False
                    except Exception as e:
                        # プログラム生成エラー（再試行可能）
                        error_msg = str(e)
                        print(f"  [エラー] テスト{test_index}: 部分プログラムフローでエラー: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            failure_count += 1
                            break
                        continue

                if not use_partial_program_flow:
                    # 通常フローを使用
                    print(f"  [情報] テスト{test_index}: 通常フローを使用します")
                    # グリッドサイズを決定
                    grid_width, grid_height = decide_grid_size()

                    # ProgramContextを作成
                    context = ProgramContext(complexity, grid_width=grid_width, grid_height=grid_height)

                    # プログラムノードを生成
                    try:
                        nodes = generator.generate_program_nodes(complexity=complexity, grid_width=grid_width, grid_height=grid_height)
                    except Exception as e:
                        # プログラム生成エラー（再試行可能）
                        error_msg = str(e)
                        if "通常の引数生成に失敗しました" in error_msg or "プログラムノードが空です" in error_msg:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"  [エラー] テスト{test_index}: 最大再試行回数に達しました。スキップします。")
                                failure_count += 1
                                break
                            continue
                        else:
                            print(f"  [エラー] テスト{test_index}: プログラム生成エラー: {e}")
                            retry_count += 1
                            if retry_count >= max_retries:
                                failure_count += 1
                                break
                            continue

                    if not nodes or len(nodes) == 0:
                        print(f"  [警告] テスト{test_index}: プログラムノードが空です。")
                        retry_count += 1
                        continue

                    # プログラムコードを生成
                    try:
                        program_code = generate_code(nodes, context)
                    except Exception as e:
                        print(f"  [エラー] テスト{test_index}: プログラムコード生成エラー: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            failure_count += 1
                            break
                        continue

                    if not program_code or not program_code.strip():
                        print(f"  [警告] テスト{test_index}: プログラムコードが空です。")
                        retry_count += 1
                        continue

                    # GET_ALL_OBJECTSが含まれているか確認（必須）
                    if 'GET_ALL_OBJECTS' not in program_code:
                        print(f"  [警告] テスト{test_index}: プログラムコードにGET_ALL_OBJECTSが含まれていません。")
                        retry_count += 1
                        continue

                    # プログラムを実行してインプットグリッドとアウトプットグリッドを取得
                    try:
                        result_nodes, input_grid, output_grid, trace_results = core_executor_main(
                            nodes=nodes,
                            grid_width=grid_width,
                            grid_height=grid_height,
                            task_index=test_index,
                            enable_replacement=False,
                            is_first_pair=True
                        )

                        if input_grid is None or output_grid is None:
                            print(f"  [警告] テスト{test_index}: グリッド生成に失敗しました。")
                            retry_count += 1
                            continue

                    except Exception as e:
                        print(f"  [エラー] テスト{test_index}: プログラム実行エラー: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            failure_count += 1
                            break
                        continue

                # プログラムファイルを保存
                program_file = os.path.join(output_dir, f"test_{test_index}_program.txt")
                with open(program_file, "w", encoding="utf-8") as f:
                    f.write(program_code)
                print(f"  [OK] プログラムファイルを保存: {program_file}")

                # グリッドを可視化（入力グリッドのみ）
                input_grid_file = os.path.join(output_dir, f"test_{test_index}_input_grid.png")
                try:
                    # 入力グリッドのみを可視化
                    save_single_grid_to_png(
                        grid=input_grid,
                        output_path=input_grid_file,
                        title=f"Test {test_index} Input Grid"
                    )
                    print(f"  [OK] インプットグリッド画像を保存: {input_grid_file}")
                except Exception as e:
                    print(f"  [警告] テスト{test_index}: 画像保存エラー: {e}")

                success_count += 1
                print(f"  [完了] テスト{test_index}の生成が完了しました")
                success = True
                break  # 成功したらループを抜ける

            except Exception as e:
                print(f"  [エラー] テスト{test_index}: 予期しないエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    import traceback
                    traceback.print_exc()
                    failure_count += 1
                    break
                continue

        if not success:
            # ループを抜けたが成功しなかった場合
            if retry_count >= max_retries:
                failure_count += 1
            continue

    print(f"\n{'='*60}")
    print(f"生成完了: 成功 {success_count}個, 失敗 {failure_count}個")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="プログラム生成テストを再生成")
    parser.add_argument(
        "--num-tests",
        type=int,
        default=30,
        help="生成するテスト数（デフォルト: 30）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_outputs/program_generation_test",
        help="出力ディレクトリ（デフォルト: test_outputs/program_generation_test）"
    )

    args = parser.parse_args()

    generate_test_programs(num_tests=args.num_tests, output_dir=args.output_dir)
