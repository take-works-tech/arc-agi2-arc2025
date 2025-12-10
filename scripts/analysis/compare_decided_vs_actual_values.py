"""
決定された値と実際の値を比較して、オブジェクト数と色数のどちらが問題かを特定
"""
import sys
from pathlib import Path
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 詳細ログを無効化（出力を減らすため）
os.environ['ENABLE_VERBOSE_LOGGING'] = 'false'

import numpy as np
from collections import Counter
import json
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.node_validator_output import (
    decide_num_objects_by_arc_statistics,
    decide_object_color_count,
    select_object_colors
)
# オブジェクト抽出は後で実装（importエラー回避）

def compare_decided_vs_actual(num_samples=100):
    """決定された値と実際の値を比較"""

    print("=" * 80)
    print("決定値と実際の値の比較（オブジェクト数 vs 色数）")
    print("=" * 80)

    results = []

    for i in range(num_samples):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()
            grid_area = grid_width * grid_height

            # 決定された値を記録（実際のロジックを再現）
            background_color = 0  # 仮の背景色（実際には決定されるが、簡略化のため0を使用）

            decided_color_count = decide_object_color_count(
                existing_colors=set(),
                background_color=background_color
            )
            selected_colors = select_object_colors(
                background_color=background_color,
                target_color_count=decided_color_count,
                existing_colors=set()
            )
            decided_object_count = decide_num_objects_by_arc_statistics(
                grid_width=grid_width,
                grid_height=grid_height,
                all_commands=None
            )

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

            # 実際の値を計算
            actual_background_color = Counter(input_grid.flatten()).most_common(1)[0][0]
            actual_color_count = len(unique_colors)
            actual_object_colors = len([c for c in unique_colors if c != actual_background_color])

            # オブジェクト数の推定（ピクセル数から推定）
            # 注意: 正確なオブジェクト数を取得するにはオブジェクト抽出が必要だが、
            # ここでは簡易的にピクセル数から推定する
            # 実際のオブジェクト数は、オブジェクト配置の成功率に依存する
            # ここでは、決定されたオブジェクト数と比較するため、グリッド構築時に
            # 実際に配置されたオブジェクト数を記録する必要がある

            # 簡易推定：オブジェクトピクセル数を平均サイズ（4ピクセル）で割る
            # これは正確ではないが、傾向を把握するためには有効
            object_pixels_for_count = np.sum(input_grid != actual_background_color)
            estimated_object_count = max(1, object_pixels_for_count // 4)
            actual_object_count = estimated_object_count

            # より正確な推定：実際のオブジェクト配置ロジックを確認する必要がある
            # 現時点では、決定値との比較が主目的なので、推定値を使用

            object_pixels = np.sum(input_grid != actual_background_color)
            object_pixel_ratio = object_pixels / (grid_width * grid_height)

            result = {
                'grid_size': f"{grid_width}x{grid_height}",
                'grid_area': grid_area,
                'decided_color_count': decided_color_count,
                'decided_object_count': decided_object_count,
                'actual_color_count': actual_color_count,
                'actual_object_color_count': actual_object_colors,
                'actual_object_count': actual_object_count,
                'object_pixel_ratio': object_pixel_ratio,
                'color_count_difference': actual_object_colors - decided_color_count,
                'object_count_difference': actual_object_count - decided_object_count,
                'color_count_ratio': actual_object_colors / decided_color_count if decided_color_count > 0 else 0,
                'object_count_ratio': actual_object_count / decided_object_count if decided_object_count > 0 else 0,
            }

            results.append(result)

            if (i + 1) % 20 == 0:
                print(f"進捗: {i+1}/{num_samples}")

        except Exception as e:
            continue

    if not results:
        print("結果がありません")
        return

    print(f"\n成功したタスク数: {len(results)}/{num_samples}\n")

    # 統計を計算
    color_differences = [r['color_count_difference'] for r in results]
    object_differences = [r['object_count_difference'] for r in results]
    color_ratios = [r['color_count_ratio'] for r in results]
    object_ratios = [r['object_count_ratio'] for r in results]

    print("=" * 80)
    print("1. 色数の比較")
    print("=" * 80)

    print(f"\n決定された色数（平均）: {np.mean([r['decided_color_count'] for r in results]):.2f}")
    print(f"実際のオブジェクト色数（平均）: {np.mean([r['actual_object_color_count'] for r in results]):.2f}")
    print(f"色数の差（平均）: {np.mean(color_differences):.2f}")
    print(f"色数の比率（平均）: {np.mean(color_ratios):.2f}")
    print(f"色数が減少したケース: {sum(1 for d in color_differences if d < 0)}個 ({sum(1 for d in color_differences if d < 0)/len(results)*100:.1f}%)")
    print(f"色数が一致したケース: {sum(1 for d in color_differences if d == 0)}個 ({sum(1 for d in color_differences if d == 0)/len(results)*100:.1f}%)")
    print(f"色数が増加したケース: {sum(1 for d in color_differences if d > 0)}個 ({sum(1 for d in color_differences if d > 0)/len(results)*100:.1f}%)")

    print("\n色数の差の分布:")
    color_diff_dist = Counter(color_differences)
    for diff in sorted(color_diff_dist.keys()):
        print(f"  {diff:+d}色: {color_diff_dist[diff]}個 ({color_diff_dist[diff]/len(results)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("2. オブジェクト数の比較")
    print("=" * 80)

    print(f"\n決定されたオブジェクト数（平均）: {np.mean([r['decided_object_count'] for r in results]):.2f}")
    print(f"実際のオブジェクト数（平均）: {np.mean([r['actual_object_count'] for r in results]):.2f}")
    print(f"オブジェクト数の差（平均）: {np.mean(object_differences):.2f}")
    print(f"オブジェクト数の比率（平均）: {np.mean(object_ratios):.2f}")
    print(f"オブジェクト数が減少したケース: {sum(1 for d in object_differences if d < 0)}個 ({sum(1 for d in object_differences if d < 0)/len(results)*100:.1f}%)")
    print(f"オブジェクト数が一致したケース: {sum(1 for d in object_differences if d == 0)}個 ({sum(1 for d in object_differences if d == 0)/len(results)*100:.1f}%)")
    print(f"オブジェクト数が増加したケース: {sum(1 for d in object_differences if d > 0)}個 ({sum(1 for d in object_differences if d > 0)/len(results)*100:.1f}%)")

    print("\nオブジェクト数の差の分布:")
    object_diff_dist = Counter([int(d) for d in object_differences])
    for diff in sorted(object_diff_dist.keys())[:20]:  # 最初の20個を表示
        print(f"  {diff:+d}個: {object_diff_dist[diff]}個 ({object_diff_dist[diff]/len(results)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("3. 問題の特定")
    print("=" * 80)

    color_decreased_count = sum(1 for d in color_differences if d < 0)
    object_decreased_count = sum(1 for d in object_differences if d < 0)

    print(f"\n色数が減少したケース: {color_decreased_count}個 ({color_decreased_count/len(results)*100:.1f}%)")
    print(f"オブジェクト数が減少したケース: {object_decreased_count}個 ({object_decreased_count/len(results)*100:.1f}%)")

    if color_decreased_count > object_decreased_count:
        print(f"\n→ **色数の問題が主要**: 色数が減少したケースがオブジェクト数より{color_decreased_count - object_decreased_count}個多い")
    elif object_decreased_count > color_decreased_count:
        print(f"\n→ **オブジェクト数の問題が主要**: オブジェクト数が減少したケースが色数より{object_decreased_count - color_decreased_count}個多い")
    else:
        print(f"\n→ 両方の問題が同程度")

    avg_color_ratio = np.mean(color_ratios)
    avg_object_ratio = np.mean(object_ratios)

    print(f"\n平均比率:")
    print(f"  色数: {avg_color_ratio:.2%}（1.0 = 完全一致）")
    print(f"  オブジェクト数: {avg_object_ratio:.2%}（1.0 = 完全一致）")

    if avg_color_ratio < avg_object_ratio:
        print(f"\n→ **色数の問題が主要**: 色数の平均比率がオブジェクト数より{avg_object_ratio - avg_color_ratio:.2%}低い")
    elif avg_object_ratio < avg_color_ratio:
        print(f"\n→ **オブジェクト数の問題が主要**: オブジェクト数の平均比率が色数より{avg_color_ratio - avg_object_ratio:.2%}低い")
    else:
        print(f"\n→ 両方の問題が同程度")

    # 結果を保存
    output_file = Path("outputs/input_grid_comparison_no_program_analysis/decided_vs_actual_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'total_samples': num_samples,
        'success_count': len(results),
        'color_statistics': {
            'decided_mean': float(np.mean([r['decided_color_count'] for r in results])),
            'actual_mean': float(np.mean([r['actual_object_color_count'] for r in results])),
            'difference_mean': float(np.mean(color_differences)),
            'ratio_mean': float(np.mean(color_ratios)),
            'decreased_count': color_decreased_count,
            'decreased_rate': color_decreased_count / len(results) if results else 0
        },
        'object_statistics': {
            'decided_mean': float(np.mean([r['decided_object_count'] for r in results])),
            'actual_mean': float(np.mean([r['actual_object_count'] for r in results])),
            'difference_mean': float(np.mean(object_differences)),
            'ratio_mean': float(np.mean(object_ratios)),
            'decreased_count': object_decreased_count,
            'decreased_rate': object_decreased_count / len(results) if results else 0
        },
        'conclusion': {
            'color_problem_is_primary': color_decreased_count > object_decreased_count and avg_color_ratio < avg_object_ratio,
            'object_problem_is_primary': object_decreased_count > color_decreased_count and avg_object_ratio < avg_color_ratio,
            'both_problems_equal': abs(color_decreased_count - object_decreased_count) <= 5 and abs(avg_color_ratio - avg_object_ratio) < 0.1
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {output_file}")

if __name__ == "__main__":
    compare_decided_vs_actual(100)
