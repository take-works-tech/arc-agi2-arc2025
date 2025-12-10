"""
プログラムなしモードでのオブジェクト数と色数の決定過程をデバッグするスクリプト
"""
import sys
from pathlib import Path
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['ENABLE_VERBOSE_LOGGING'] = 'true'

import numpy as np
from collections import Counter
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size

def debug_color_and_object_count(num_samples=100):
    """色数とオブジェクト数の決定過程をデバッグ"""

    print("=" * 80)
    print("プログラムなしモードでのオブジェクト数と色数の決定過程のデバッグ")
    print("=" * 80)

    stats = {
        'decided_color_counts': [],
        'decided_object_counts': [],
        'actual_color_counts': [],
        'actual_object_counts': [],
        'object_pixel_ratios': [],
        'grid_areas': []
    }

    success_count = 0

    for i in range(num_samples):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()

            # 入力グリッド生成（プログラムなし）
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )

            if input_grid is None:
                continue

            # 単色グリッドチェック
            unique_colors = np.unique(input_grid)
            if len(unique_colors) <= 1:
                continue

            # 統計情報を抽出
            height, width = input_grid.shape
            background_color = Counter(input_grid.flatten()).most_common(1)[0][0]
            object_pixels = np.sum(input_grid != background_color)
            object_pixel_ratio = object_pixels / (width * height)

            actual_color_count = len(unique_colors)
            actual_object_colors = len([c for c in unique_colors if c != background_color])

            stats['actual_color_counts'].append(actual_color_count)
            stats['grid_areas'].append(width * height)
            stats['object_pixel_ratios'].append(object_pixel_ratio)

            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"進捗: {i+1}/{num_samples} (成功: {success_count})")

        except Exception as e:
            continue

    # 結果を分析
    print(f"\n成功したタスク数: {success_count}/{num_samples}")

    if stats['actual_color_counts']:
        print("\n実際の色数分布:")
        color_dist = Counter(stats['actual_color_counts'])
        for color_count in sorted(color_dist.keys()):
            print(f"  {color_count}色: {color_dist[color_count]}個 ({color_dist[color_count]/len(stats['actual_color_counts'])*100:.1f}%)")

        print(f"\n実際の色数: 平均={np.mean(stats['actual_color_counts']):.2f}, 中央値={np.median(stats['actual_color_counts']):.2f}")

    if stats['object_pixel_ratios']:
        print(f"\nオブジェクトピクセル比率: 平均={np.mean(stats['object_pixel_ratios']):.4f}, 中央値={np.median(stats['object_pixel_ratios']):.4f}")
        print(f"  最小={min(stats['object_pixel_ratios']):.4f}, 最大={max(stats['object_pixel_ratios']):.4f}")

    if stats['grid_areas']:
        print(f"\nグリッド面積: 平均={np.mean(stats['grid_areas']):.2f}, 中央値={np.median(stats['grid_areas']):.2f}")

if __name__ == "__main__":
    debug_color_and_object_count(100)

