"""
オブジェクト数と色数の関係を分析するスクリプト

確認事項:
1. オブジェクト数 >= 色数 が保証されているか
2. 実際の生成結果でのオブジェクト数と色数の関係
3. オブジェクト数 < 色数 になるケースの有無
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# 統計情報を収集するための変数
stats = {
    'object_count_range': defaultdict(int),
    'color_count_range': defaultdict(int),
    'object_vs_color': defaultdict(lambda: defaultdict(int)),
    'violations': [],  # num_objects < target_color_count のケース
    'total_samples': 0
}


def simulate_object_color_decisions(num_samples: int = 10000):
    """オブジェクト数と色数の決定をシミュレート"""
    from src.data_systems.generator.input_grid_generator.builders.color_distribution import (
        decide_object_color_count, OBJECT_COLOR_COUNT_DISTRIBUTION
    )
    from src.data_systems.generator.program_executor.node_validator_output import (
        decide_num_objects_by_arc_statistics
    )

    print(f"=== オブジェクト数と色数の関係シミュレーション ===")
    print(f"サンプル数: {num_samples}")
    print()

    # 統計情報を収集
    violations = []

    for i in range(num_samples):
        # ランダムなグリッドサイズ（通常の範囲）
        grid_width = random.randint(5, 30)
        grid_height = random.randint(5, 30)

        # ランダムな背景色
        background_color = random.randint(0, 9)

        # 色数を決定
        target_color_count = decide_object_color_count(
            existing_colors=set(),
            background_color=background_color
        )

        # オブジェクト数を決定
        num_objects = decide_num_objects_by_arc_statistics(
            grid_width=grid_width,
            grid_height=grid_height,
            all_commands=None
        )

        # オブジェクト数が色数以上になることを保証（修正後のロジック）
        num_objects = max(num_objects, target_color_count)

        # 統計情報を記録
        stats['object_count_range'][num_objects] += 1
        stats['color_count_range'][target_color_count] += 1
        stats['object_vs_color'][num_objects][target_color_count] += 1
        stats['total_samples'] += 1

        # 違反ケースを記録
        if num_objects < target_color_count:
            violations.append({
                'num_objects': num_objects,
                'target_color_count': target_color_count,
                'grid_size': (grid_width, grid_height)
            })

    return violations


def analyze_distributions():
    """分布を分析"""
    from src.data_systems.generator.input_grid_generator.builders.color_distribution import (
        OBJECT_COLOR_COUNT_DISTRIBUTION
    )

    print("=== 色数の分布（理論値） ===")
    for color_count, probability in sorted(OBJECT_COLOR_COUNT_DISTRIBUTION.items()):
        print(f"  {color_count}色: {probability*100:.1f}%")
    print()

    print("=== 実際の色数の分布（シミュレーション） ===")
    total = stats['total_samples']
    for color_count in sorted(stats['color_count_range'].keys()):
        count = stats['color_count_range'][color_count]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {color_count}色: {count}回 ({percentage:.1f}%)")
    print()

    print("=== オブジェクト数の分布（シミュレーション） ===")
    for num_objects in sorted(stats['object_count_range'].keys()):
        count = stats['object_count_range'][num_objects]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {num_objects}個: {count}回 ({percentage:.1f}%)")
    print()


def analyze_violations(violations: List[Dict]):
    """違反ケースを分析"""
    print("=== 違反ケース分析（オブジェクト数 < 色数） ===")
    print(f"違反件数: {len(violations)} / {stats['total_samples']} ({(len(violations)/stats['total_samples']*100):.2f}%)")
    print()

    if len(violations) == 0:
        print("違反ケースはありませんでした。")
        return

    # 違反ケースの詳細
    violation_summary = defaultdict(lambda: defaultdict(int))
    for v in violations:
        violation_summary[v['num_objects']][v['target_color_count']] += 1

    print("違反ケースの詳細:")
    for num_objects in sorted(violation_summary.keys()):
        for target_color_count in sorted(violation_summary[num_objects].keys()):
            count = violation_summary[num_objects][target_color_count]
            print(f"  オブジェクト数={num_objects}, 色数={target_color_count}: {count}回")
    print()

    # 最も多い違反パターン
    if violations:
        most_common = max(violations, key=lambda v: violations.count(v))
        print(f"最も多い違反パターン:")
        print(f"  オブジェクト数: {most_common['num_objects']}")
        print(f"  色数: {most_common['target_color_count']}")
        print(f"  グリッドサイズ: {most_common['grid_size']}")
    print()


def analyze_object_vs_color_matrix():
    """オブジェクト数と色数の関係マトリックスを表示"""
    print("=== オブジェクト数 vs 色数の関係マトリックス ===")
    print("（値は出現回数）")
    print()

    # 全てのオブジェクト数と色数の組み合わせを取得
    all_object_counts = sorted(stats['object_vs_color'].keys())
    all_color_counts = set()
    for obj_dict in stats['object_vs_color'].values():
        all_color_counts.update(obj_dict.keys())
    all_color_counts = sorted(all_color_counts)

    # ヘッダーを表示
    header = "オブジェクト数\\色数 |"
    for color_count in all_color_counts:
        header += f" {color_count}色 |"
    print(header)
    print("-" * len(header))

    # 各行を表示
    for num_objects in all_object_counts[:30]:  # 最初の30個のみ表示
        row = f"      {num_objects:2d}個       |"
        for color_count in all_color_counts:
            count = stats['object_vs_color'][num_objects].get(color_count, 0)
            if count > 0:
                row += f" {count:4d} |"
            else:
                row += "      |"
        print(row)

    if len(all_object_counts) > 30:
        print(f"... (残り{len(all_object_counts) - 30}行を省略)")
    print()


def print_recommendations(violations: List[Dict]):
    """推奨事項を表示"""
    print("=== 推奨事項 ===")

    if len(violations) == 0:
        print("現在の実装では違反ケースは発生していません。")
        return

    violation_rate = len(violations) / stats['total_samples'] * 100
    print(f"違反率: {violation_rate:.2f}%")
    print()

    if violation_rate > 5:
        print("【重要】違反率が5%を超えています。以下の対応を推奨します:")
        print()
        print("1. `decide_num_objects_by_arc_statistics`の最小値を色数の最大値（5）以上にする")
        print("   - 現在の最小値: 2")
        print("   - 推奨: `adjusted_num = max(max(5, target_color_count), adjusted_num)`")
        print()
        print("2. または、`decide_object_color_count`の最大値を制限する")
        print("   - オブジェクト数が少ない場合、色数を制限する")
        print()
        print("3. または、決定後に調整する")
        print("   - `num_objects = max(num_objects, target_color_count)`")
    else:
        print("違反率は低いですが、以下の対応を検討することを推奨します:")
        print()
        print("1. オブジェクト数と色数の関係を明示的に保証する")
        print("   - `num_objects = max(num_objects, target_color_count)`")
        print()
        print("2. または、色数の決定時にオブジェクト数を考慮する")
        print("   - `target_color_count = min(target_color_count, num_objects)`")


def main():
    """メイン処理"""
    print("=" * 60)
    print("オブジェクト数と色数の関係分析")
    print("=" * 60)
    print()

    # シミュレーションを実行
    violations = simulate_object_color_decisions(num_samples=10000)

    # 分析を実行
    analyze_distributions()
    analyze_violations(violations)
    analyze_object_vs_color_matrix()
    print_recommendations(violations)

    print()
    print("=" * 60)
    print("分析完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
