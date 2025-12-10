"""
decide_object_color_countがどの色数を決定しているかを調査
"""
import sys
import os
from pathlib import Path
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.input_grid_generator.builders.color_distribution import decide_object_color_count

def investigate_color_count_decision(num_samples=10000):
    """decide_object_color_countの決定分布を調査"""

    print("=" * 80)
    print("decide_object_color_count の決定分布調査")
    print("=" * 80)
    print(f"\nサンプル数: {num_samples}")

    # 各色数の決定回数をカウント
    color_count_distribution = {}

    for i in range(num_samples):
        target_color_count = decide_object_color_count(
            existing_colors=set(),
            background_color=0
        )

        color_count_distribution[target_color_count] = color_count_distribution.get(target_color_count, 0) + 1

        if (i + 1) % 1000 == 0:
            print(f"  進捗: {i + 1}/{num_samples}")

    print(f"\n決定された色数の分布:")
    print("-" * 40)
    total = sum(color_count_distribution.values())
    for color_count in sorted(color_count_distribution.keys()):
        count = color_count_distribution[color_count]
        percentage = (count / total) * 100
        print(f"  {color_count}色: {count}回 ({percentage:.2f}%)")

    print(f"\n合計: {total}")

    # 期待される分布と比較
    print("\n期待される分布 (OBJECT_COLOR_COUNT_DISTRIBUTION):")
    print("-" * 40)
    from src.data_systems.generator.input_grid_generator.builders.color_distribution import OBJECT_COLOR_COUNT_DISTRIBUTION
    for color_count, expected_prob in sorted(OBJECT_COLOR_COUNT_DISTRIBUTION.items()):
        expected_count = expected_prob * total
        actual_count = color_count_distribution.get(color_count, 0)
        diff = actual_count - expected_count
        diff_percent = (diff / expected_count * 100) if expected_count > 0 else 0
        print(f"  {color_count}色: 期待={expected_count:.1f}回 ({expected_prob*100:.1f}%), "
              f"実際={actual_count}回 ({actual_count/total*100:.2f}%), "
              f"差分={diff:.1f}回 ({diff_percent:+.2f}%)")

if __name__ == '__main__':
    investigate_color_count_decision(num_samples=10000)

