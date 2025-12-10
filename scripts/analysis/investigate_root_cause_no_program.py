"""
プログラムなしモードでのオブジェクト数と色数の根本原因を調査するスクリプト
詳細なログを出力して、決定された値と実際に使用された値を比較
"""
import sys
from pathlib import Path
import os
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 詳細ログを有効化
os.environ['ENABLE_VERBOSE_LOGGING'] = 'true'

import numpy as np
from collections import Counter
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.node_validator_output import (
    decide_num_objects_by_arc_statistics,
    decide_object_color_count,
    select_object_colors
)

def investigate_root_cause(num_samples=50):
    """根本原因を調査"""

    print("=" * 80)
    print("プログラムなしモードでのオブジェクト数と色数の根本原因調査")
    print("=" * 80)

    # 決定された値と実際の値を記録
    statistics = {
        'grid_areas': [],
        'decided_color_counts': [],
        'decided_object_counts': [],
        'actual_color_counts': [],
        'actual_object_color_counts': [],
        'object_pixel_ratios': [],
        'cases': []  # 各ケースの詳細
    }

    success_count = 0

    for i in range(num_samples):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()
            grid_area = grid_width * grid_height

            # 決定された値を記録（実際のロジックを再現）
            background_color = 0  # 仮の背景色（実際には決定される）
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
            object_pixels = np.sum(input_grid != actual_background_color)
            object_pixel_ratio = object_pixels / (grid_width * grid_height)

            # 記録
            statistics['grid_areas'].append(grid_area)
            statistics['decided_color_counts'].append(decided_color_count)
            statistics['decided_object_counts'].append(decided_object_count)
            statistics['actual_color_counts'].append(actual_color_count)
            statistics['actual_object_color_counts'].append(actual_object_colors)
            statistics['object_pixel_ratios'].append(object_pixel_ratio)

            statistics['cases'].append({
                'grid_size': f"{grid_width}x{grid_height}",
                'grid_area': grid_area,
                'decided_color_count': decided_color_count,
                'decided_object_count': decided_object_count,
                'actual_color_count': actual_color_count,
                'actual_object_color_count': actual_object_colors,
                'object_pixel_ratio': object_pixel_ratio,
                'color_difference': actual_object_colors - decided_color_count,
                'object_count_too_low': decided_object_count < decided_color_count
            })

            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"進捗: {i+1}/{num_samples} (成功: {success_count})")

        except Exception as e:
            import traceback
            print(f"エラー: {e}")
            traceback.print_exc()
            continue

    # 分析結果を表示
    print(f"\n成功したタスク数: {success_count}/{num_samples}\n")

    if statistics['decided_color_counts']:
        print("=" * 80)
        print("1. 色数の分析")
        print("=" * 80)

        print(f"\n決定された色数分布:")
        decided_dist = Counter(statistics['decided_color_counts'])
        for color_count in sorted(decided_dist.keys()):
            print(f"  {color_count}色: {decided_dist[color_count]}個 ({decided_dist[color_count]/len(statistics['decided_color_counts'])*100:.1f}%)")

        print(f"\n実際の色数分布:")
        actual_dist = Counter(statistics['actual_color_counts'])
        for color_count in sorted(actual_dist.keys()):
            print(f"  {color_count}色: {actual_dist[color_count]}個 ({actual_dist[color_count]/len(statistics['actual_color_counts'])*100:.1f}%)")

        print(f"\n決定された色数: 平均={np.mean(statistics['decided_color_counts']):.2f}")
        print(f"実際の色数: 平均={np.mean(statistics['actual_color_counts']):.2f}")
        print(f"実際のオブジェクト色数: 平均={np.mean(statistics['actual_object_color_counts']):.2f}")

        # 色数が減少したケースを分析
        color_decreased_cases = [c for c in statistics['cases'] if c['color_difference'] < 0]
        if color_decreased_cases:
            print(f"\n色数が減少したケース: {len(color_decreased_cases)}個")
            for case in color_decreased_cases[:5]:  # 最初の5個を表示
                print(f"  グリッド: {case['grid_size']}, 決定色数: {case['decided_color_count']}, 実際: {case['actual_object_color_count']}, "
                      f"決定オブジェクト数: {case['decided_object_count']}, オブジェクト数不足: {case['object_count_too_low']}")

        # オブジェクト数が色数より少ないケース
        object_count_too_low_cases = [c for c in statistics['cases'] if c['object_count_too_low']]
        if object_count_too_low_cases:
            print(f"\nオブジェクト数が色数より少ないケース: {len(object_count_too_low_cases)}個 ({len(object_count_too_low_cases)/len(statistics['cases'])*100:.1f}%)")
            for case in object_count_too_low_cases[:5]:
                print(f"  グリッド: {case['grid_size']}, 決定色数: {case['decided_color_count']}, 決定オブジェクト数: {case['decided_object_count']}, "
                      f"実際の色数: {case['actual_object_color_count']}")

    if statistics['decided_object_counts']:
        print("\n" + "=" * 80)
        print("2. オブジェクト数の分析")
        print("=" * 80)

        print(f"\n決定されたオブジェクト数分布:")
        decided_obj_dist = Counter(statistics['decided_object_counts'])
        for obj_count in sorted(decided_obj_dist.keys())[:10]:  # 最初の10個を表示
            print(f"  {obj_count}個: {decided_obj_dist[obj_count]}回 ({decided_obj_dist[obj_count]/len(statistics['decided_object_counts'])*100:.1f}%)")

        print(f"\n決定されたオブジェクト数: 平均={np.mean(statistics['decided_object_counts']):.2f}, "
              f"最小={min(statistics['decided_object_counts'])}, 最大={max(statistics['decided_object_counts'])}")

        # グリッドサイズとの関係
        print(f"\nグリッド面積との関係:")
        small_grids = [c for c in statistics['cases'] if c['grid_area'] < 100]
        medium_grids = [c for c in statistics['cases'] if 100 <= c['grid_area'] < 300]
        large_grids = [c for c in statistics['cases'] if c['grid_area'] >= 300]

        if small_grids:
            avg_obj_small = np.mean([c['decided_object_count'] for c in small_grids])
            print(f"  小さいグリッド (面積<100): 平均オブジェクト数={avg_obj_small:.2f} (n={len(small_grids)})")
        if medium_grids:
            avg_obj_medium = np.mean([c['decided_object_count'] for c in medium_grids])
            print(f"  中程度のグリッド (100<=面積<300): 平均オブジェクト数={avg_obj_medium:.2f} (n={len(medium_grids)})")
        if large_grids:
            avg_obj_large = np.mean([c['decided_object_count'] for c in large_grids])
            print(f"  大きいグリッド (面積>=300): 平均オブジェクト数={avg_obj_large:.2f} (n={len(large_grids)})")

    if statistics['object_pixel_ratios']:
        print("\n" + "=" * 80)
        print("3. オブジェクトピクセル比率の分析")
        print("=" * 80)

        print(f"\nオブジェクトピクセル比率: 平均={np.mean(statistics['object_pixel_ratios']):.4f}, "
              f"中央値={np.median(statistics['object_pixel_ratios']):.4f}, "
              f"最小={min(statistics['object_pixel_ratios']):.4f}, 最大={max(statistics['object_pixel_ratios']):.4f}")

        # オブジェクト数とピクセル比率の関係
        print(f"\nオブジェクト数とピクセル比率の関係:")
        for obj_range in [(2, 4), (5, 8), (9, 12), (13, 20)]:
            cases_in_range = [c for c in statistics['cases'] if obj_range[0] <= c['decided_object_count'] <= obj_range[1]]
            if cases_in_range:
                avg_ratio = np.mean([c['object_pixel_ratio'] for c in cases_in_range])
                print(f"  オブジェクト数{obj_range[0]}-{obj_range[1]}個: 平均ピクセル比率={avg_ratio:.4f} (n={len(cases_in_range)})")

    # 結果をJSONに保存
    output_file = Path("outputs/input_grid_comparison_no_program_analysis/root_cause_investigation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    output_data = {
        'summary': {
            'total_samples': num_samples,
            'success_count': success_count,
            'success_rate': success_count / num_samples if num_samples > 0 else 0
        },
        'statistics': {
            'decided_color_counts': {
                'mean': float(np.mean(statistics['decided_color_counts'])) if statistics['decided_color_counts'] else 0,
                'distribution': dict(Counter(statistics['decided_color_counts']))
            },
            'actual_color_counts': {
                'mean': float(np.mean(statistics['actual_color_counts'])) if statistics['actual_color_counts'] else 0,
                'distribution': dict(Counter(statistics['actual_color_counts']))
            },
            'decided_object_counts': {
                'mean': float(np.mean(statistics['decided_object_counts'])) if statistics['decided_object_counts'] else 0,
                'distribution': dict(Counter(statistics['decided_object_counts']))
            },
            'object_pixel_ratios': {
                'mean': float(np.mean(statistics['object_pixel_ratios'])) if statistics['object_pixel_ratios'] else 0,
                'median': float(np.median(statistics['object_pixel_ratios'])) if statistics['object_pixel_ratios'] else 0
            }
        },
        'key_findings': {
            'color_decreased_cases_count': len([c for c in statistics['cases'] if c['color_difference'] < 0]),
            'object_count_too_low_cases_count': len([c for c in statistics['cases'] if c['object_count_too_low']]),
            'object_count_too_low_rate': len([c for c in statistics['cases'] if c['object_count_too_low']]) / len(statistics['cases']) if statistics['cases'] else 0
        },
        'cases': statistics['cases'][:20]  # 最初の20個を保存
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n調査結果を保存: {output_file}")

if __name__ == "__main__":
    investigate_root_cause(100)

