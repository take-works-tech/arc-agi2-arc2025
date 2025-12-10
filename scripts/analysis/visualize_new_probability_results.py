"""
新しい確率値で生成された入力グリッドをPNG画像として可視化
"""
import sys
from pathlib import Path
import numpy as np
from collections import Counter

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.grid_visualizer import save_single_grid_to_png
from src.data_systems.generator.input_grid_generator.builders.color_distribution import (
    decide_object_color_count,
    select_object_colors
)

def visualize_input_grids(num_samples=100, output_dir=None):
    """入力グリッドを生成してPNG画像として保存"""

    if output_dir is None:
        output_dir = project_root / "outputs" / "new_probability_visualization"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("新しい確率値で生成された入力グリッドの可視化")
    print("=" * 80)
    print(f"\nサンプル数: {num_samples}")
    print(f"出力ディレクトリ: {output_dir}\n")

    success_count = 0
    failure_count = 0

    # 統計情報
    stats = {
        'decided_color_counts': [],
        'decided_object_counts': [],
        'actual_color_counts': [],
        'actual_object_counts': [],
    }

    for i in range(num_samples):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()
            grid_area = grid_width * grid_height

            # 背景色決定（仮）
            background_color = 0

            # 色数決定
            target_color_count = decide_object_color_count(
                existing_colors=set(),
                background_color=background_color
            )
            stats['decided_color_counts'].append(target_color_count)

            # 色の選択
            selected_colors = select_object_colors(
                background_color=background_color,
                target_color_count=target_color_count,
                existing_colors=set()
            )

            # 入力グリッド生成
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )

            if input_grid is None:
                failure_count += 1
                continue

            # 単色グリッドチェック
            unique_colors = np.unique(input_grid)
            if len(unique_colors) <= 1:
                failure_count += 1
                continue

            # 実際の値を計算
            actual_background_color = Counter(input_grid.flatten()).most_common(1)[0][0]
            actual_color_count = len([c for c in unique_colors if c != actual_background_color])
            stats['actual_color_counts'].append(actual_color_count)

            # オブジェクト数は正確には抽出できないため、簡易的に推定
            # 実際にはログから抽出するか、別途オブジェクト抽出を行う必要がある
            # ここでは簡易的に推定値として使用
            object_pixels = np.sum(input_grid != actual_background_color)
            # 簡易的な推定: オブジェクトピクセル数から推定（正確ではない）
            estimated_object_count = max(1, object_pixels // 10)  # 簡易推定

            # PNG画像として保存
            png_path = output_dir / f"grid_{i+1:03d}_{grid_width}x{grid_height}_colors{actual_color_count}_objects{estimated_object_count}.png"
            title = f"Grid {i+1}: {grid_width}x{grid_height}, Colors={actual_color_count}, Objects≈{estimated_object_count}"

            if save_single_grid_to_png(
                grid=input_grid,
                output_path=str(png_path),
                title=title,
                show_grid=True,
                dpi=100
            ):
                success_count += 1
            else:
                failure_count += 1

            if (i + 1) % 10 == 0:
                print(f"  処理済み: {i+1}/{num_samples} (成功: {success_count}, 失敗: {failure_count})", flush=True)

        except Exception as e:
            print(f"  [エラー] サンプル{i+1}でエラーが発生: {e}", flush=True)
            failure_count += 1
            continue

    # 統計サマリー
    print(f"\n{'='*80}")
    print("可視化完了")
    print(f"{'='*80}\n")
    print(f"成功: {success_count}/{num_samples}")
    print(f"失敗: {failure_count}/{num_samples}")
    print(f"出力ディレクトリ: {output_dir}")

    if stats['decided_color_counts']:
        print(f"\n決定された色数の分布:")
        decided_dist = Counter(stats['decided_color_counts'])
        for color_count in sorted(decided_dist.keys()):
            count = decided_dist[color_count]
            percentage = (count / len(stats['decided_color_counts'])) * 100
            print(f"  {color_count}色: {count}回 ({percentage:.2f}%)")

    if stats['actual_color_counts']:
        print(f"\n実際のオブジェクト色数の分布:")
        actual_dist = Counter(stats['actual_color_counts'])
        for color_count in sorted(actual_dist.keys()):
            count = actual_dist[color_count]
            percentage = (count / len(stats['actual_color_counts'])) * 100
            print(f"  {color_count}色: {count}回 ({percentage:.2f}%)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='新しい確率値で生成された入力グリッドをPNG画像として可視化')
    parser.add_argument('--num-samples', type=int, default=100, help='生成するサンプル数')
    parser.add_argument('--output-dir', type=str, default=None, help='出力ディレクトリ')
    args = parser.parse_args()

    visualize_input_grids(num_samples=args.num_samples, output_dir=args.output_dir)
