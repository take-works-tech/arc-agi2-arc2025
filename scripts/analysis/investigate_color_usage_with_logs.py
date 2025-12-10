"""
色数の使用状況を詳細ログで調査
"""
import sys
import os
from pathlib import Path
import numpy as np
from collections import Counter

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.input_grid_generator.builders.color_distribution import (
    decide_object_color_count,
    select_object_colors
)

def investigate_color_usage_with_logs(num_samples=100):
    """色数の使用状況を詳細ログで調査"""

    print("=" * 80)
    print("色数の使用状況を詳細ログで調査")
    print("=" * 80)
    print(f"\nサンプル数: {num_samples}\n")

    # 統計情報
    stats = {
        'decided_color_counts': [],
        'selected_colors_counts': [],
        'actual_object_colors': [],
        'final_grid_colors': [],
        'cases': []
    }

    for i in range(num_samples):
        print(f"\n{'='*80}")
        print(f"サンプル {i+1}/{num_samples}")
        print(f"{'='*80}")

        try:
            # 1. グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()
            grid_area = grid_width * grid_height
            print(f"[1] グリッドサイズ: {grid_width}x{grid_height} (面積: {grid_area})")

            # 2. 背景色決定（仮）
            background_color = 0  # 実際には決定されるが、ここでは仮に0
            print(f"[2] 背景色: {background_color}")

            # 3. 色数決定
            target_color_count = decide_object_color_count(
                existing_colors=set(),
                background_color=background_color
            )
            stats['decided_color_counts'].append(target_color_count)
            print(f"[3] 決定された色数: {target_color_count}")

            # 4. 色の選択
            selected_colors = select_object_colors(
                background_color=background_color,
                target_color_count=target_color_count,
                existing_colors=set()
            )
            stats['selected_colors_counts'].append(len(selected_colors))
            print(f"[4] 選択された色リスト: {selected_colors} (数: {len(selected_colors)})")

            # 5. 入力グリッド生成
            print(f"[5] 入力グリッド生成開始...")
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )

            if input_grid is None:
                print(f"  [エラー] 入力グリッド生成に失敗")
                continue

            # 6. 最終的なグリッドの色数
            unique_colors = np.unique(input_grid)
            num_colors_total = len(unique_colors)
            object_colors_in_grid = [c for c in unique_colors if c != background_color]
            num_colors_object = len(object_colors_in_grid)

            stats['final_grid_colors'].append(num_colors_total)
            stats['actual_object_colors'].append(num_colors_object)

            print(f"[6] 最終的なグリッドの色数:")
            print(f"  - 総色数（背景含む）: {num_colors_total}")
            print(f"  - オブジェクト色数（背景除く）: {num_colors_object}")
            print(f"  - オブジェクト色: {object_colors_in_grid}")
            print(f"  - 全体の色: {unique_colors.tolist()}")

            # 7. 色の使用状況を確認
            color_counts = Counter(input_grid.flatten())
            print(f"[7] 色の使用状況:")
            for color, count in sorted(color_counts.items()):
                percentage = (count / grid_area) * 100
                print(f"  - 色{color}: {count}ピクセル ({percentage:.2f}%)")

            # 8. 比較
            print(f"[8] 比較:")
            print(f"  - 決定された色数: {target_color_count}")
            print(f"  - 選択された色数: {len(selected_colors)}")
            print(f"  - 実際のオブジェクト色数: {num_colors_object}")

            if target_color_count != num_colors_object:
                print(f"  [警告] 決定された色数({target_color_count})と実際の色数({num_colors_object})が一致しません")
                if set(object_colors_in_grid) != set(selected_colors):
                    print(f"  [警告] 選択された色({selected_colors})と実際の色({object_colors_in_grid})が一致しません")

            # ケース情報を保存
            stats['cases'].append({
                'sample': i + 1,
                'decided_color_count': target_color_count,
                'selected_colors': selected_colors,
                'actual_object_colors': object_colors_in_grid,
                'final_total_colors': num_colors_total,
                'final_object_colors': num_colors_object,
                'match': target_color_count == num_colors_object
            })

        except Exception as e:
            print(f"  [エラー] サンプル{i+1}でエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 統計の集計
    print(f"\n{'='*80}")
    print("統計集計")
    print(f"{'='*80}\n")

    print(f"決定された色数の分布:")
    decided_dist = Counter(stats['decided_color_counts'])
    for color_count in sorted(decided_dist.keys()):
        count = decided_dist[color_count]
        percentage = (count / len(stats['decided_color_counts'])) * 100
        print(f"  {color_count}色: {count}回 ({percentage:.2f}%)")

    print(f"\n実際のオブジェクト色数の分布:")
    actual_dist = Counter(stats['actual_object_colors'])
    for color_count in sorted(actual_dist.keys()):
        count = actual_dist[color_count]
        percentage = (count / len(stats['actual_object_colors'])) * 100
        print(f"  {color_count}色: {count}回 ({percentage:.2f}%)")

    print(f"\n一致率:")
    matches = sum(1 for case in stats['cases'] if case['match'])
    total = len(stats['cases'])
    match_rate = (matches / total) * 100 if total > 0 else 0
    print(f"  一致: {matches}/{total} ({match_rate:.2f}%)")

    print(f"\n不一致のケース:")
    mismatches = [case for case in stats['cases'] if not case['match']]
    for case in mismatches[:10]:  # 最初の10ケースのみ表示
        print(f"  サンプル{case['sample']}: 決定={case['decided_color_count']}色, "
              f"実際={case['final_object_colors']}色, "
              f"選択色={case['selected_colors']}, "
              f"実際色={case['actual_object_colors']}")

    if len(mismatches) > 10:
        print(f"  ... 他{len(mismatches) - 10}ケース")

if __name__ == '__main__':
    investigate_color_usage_with_logs(num_samples=100)

