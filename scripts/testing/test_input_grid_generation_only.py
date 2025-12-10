"""
インプットグリッド生成のみを検証するスクリプト

部分プログラムフローで使用する仮のインプットグリッド生成までを検証
"""
import sys
import os
import warnings
from pathlib import Path
import numpy as np

# joblibエラーを防ぐため、環境変数をインポート前に設定
try:
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
except Exception:
    cpu_count = 4

# 環境変数を設定（joblibの初期化前に設定する必要がある）
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 警告フィルターも設定
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')

# 環境変数を設定（小規模テスト用）
os.environ['TASK_COUNT'] = '10'  # 10タスクのみ生成
os.environ['ENABLE_ALL_LOGS'] = 'true'  # 詳細ログを有効化
os.environ['ENABLE_VERBOSE_LOGGING'] = 'true'  # 詳細ログを有効化
os.environ['ENABLE_VERBOSE_OUTPUT'] = 'true'  # 詳細を表示

print("="*80)
print("インプットグリッド生成のみの検証テスト")
print("="*80)
print(f"\n[設定]")
print(f"  生成タスク数: 10")
print(f"  インプットグリッド生成のみ（プログラム生成はしない）")
print(f"\n[実行] インプットグリッド生成を開始します...\n")

try:
    # 環境変数を再設定
    os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'

    # 必要なモジュールをインポート
    from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
    from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
    from src.data_systems.generator.grid_visualizer import save_single_grid_to_png
    import time
    from pathlib import Path

    # 統計情報
    success_count = 0
    failure_count = 0
    total_time = 0.0

    # タスク数を取得
    task_count = int(os.environ.get('TASK_COUNT', '10'))

    print(f"開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 出力ディレクトリを作成
    output_dir = Path("outputs/input_grid_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir.absolute()}\n")

    for task_index in range(1, task_count + 1):
        print(f"\n{'='*80}")
        print(f"タスク {task_index}/{task_count}")
        print(f"{'='*80}")

        task_start_time = time.time()

        try:
            # 1. グリッドサイズ決定
            size_decide_start = time.time()
            grid_width, grid_height = decide_grid_size()
            size_decide_elapsed = time.time() - size_decide_start
            print(f"[成功] グリッドサイズ決定: {grid_width}x{grid_height} (処理時間: {size_decide_elapsed:.3f}秒)")

            # 2. 仮のインプットグリッド生成（プログラムなし）
            input_grid_gen_start = time.time()
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )
            input_grid_gen_elapsed = time.time() - input_grid_gen_start

            # インプットグリッドの情報を表示
            if input_grid is not None:
                unique_colors = np.unique(input_grid).tolist()

                # 単色グリッドのチェック（失敗として扱う）
                if len(unique_colors) == 1:
                    raise ValueError(f"タスク{task_index}: 入力グリッドが単色です（色={unique_colors[0]}）。オブジェクトが配置されていません。")

                print(f"[成功] インプットグリッド生成完了 (処理時間: {input_grid_gen_elapsed:.3f}秒)")
                print(f"  グリッドサイズ: {input_grid.shape[1]}x{input_grid.shape[0]} (幅x高さ)")
                print(f"  使用されている色: {unique_colors}")
                print(f"  色の種類数: {len(unique_colors)}")
                print(f"  グリッドの内容:")
                for y in range(min(10, input_grid.shape[0])):  # 最大10行まで表示
                    row_str = " ".join(str(input_grid[y, x]) for x in range(min(10, input_grid.shape[1])))
                    if input_grid.shape[1] > 10:
                        row_str += " ..."
                    print(f"    {row_str}")
                if input_grid.shape[0] > 10:
                    print(f"    ... ({input_grid.shape[0] - 10}行省略)")

                # PNGとして保存
                png_path = output_dir / f"task_{task_index:03d}_input_grid_{grid_width}x{grid_height}.png"
                png_title = f"Task {task_index} - Input Grid ({grid_width}x{grid_height})"
                save_success = save_single_grid_to_png(
                    grid=input_grid,
                    output_path=str(png_path),
                    title=png_title,
                    show_grid=True,
                    dpi=100
                )
                if save_success:
                    print(f"  [PNG保存] {png_path.name}")
                else:
                    print(f"  [警告] PNG保存に失敗しました: {png_path.name}")
            else:
                print(f"[警告] インプットグリッドがNoneです")

            task_elapsed = time.time() - task_start_time
            total_time += task_elapsed
            success_count += 1
            print(f"[完了] タスク{task_index} 成功 (総処理時間: {task_elapsed:.3f}秒)")

        except Exception as e:
            task_elapsed = time.time() - task_start_time
            total_time += task_elapsed
            failure_count += 1
            print(f"[失敗] タスク{task_index} エラー: {type(e).__name__}: {e} (処理時間: {task_elapsed:.3f}秒)")
            import traceback
            traceback.print_exc()

    # 統計情報を表示
    print(f"\n{'='*80}")
    print(f"【実行完了】")
    print(f"{'='*80}")
    print(f"終了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"成功: {success_count}/{task_count}")
    print(f"失敗: {failure_count}/{task_count}")
    if success_count > 0:
        avg_time = total_time / task_count
        print(f"1タスクあたりの平均時間: {avg_time:.3f}秒")

except KeyboardInterrupt:
    print("\n[中断] ユーザーによって中断されました。")
    sys.exit(1)
except Exception as e:
    print(f"\n[エラー] エラーが発生しました: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
