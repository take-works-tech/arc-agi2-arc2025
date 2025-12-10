"""
現在の実装と提案した実装の比較分析
- 多様性への影響
- 密度改善効果
"""
import random
import numpy as np
from collections import defaultdict

def current_implementation_simulation(num_samples=10000):
    """現在の実装をシミュレート"""
    results_by_area = defaultdict(list)

    area_ranges = [
        (0, 100, "< 100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 400, "300-400"),
        (400, 600, "400-600"),
        (600, float('inf'), ">= 600"),
    ]

    # 各面積範囲でシミュレート
    for min_area, max_area, label in area_ranges:
        # 代表的な面積を選択
        if label == "< 100":
            grid_area = 42  # 平均値
        elif label == "100-200":
            grid_area = 129  # 平均値
        elif label == "200-300":
            grid_area = 243  # 平均値
        elif label == "300-400":
            grid_area = 344  # 平均値
        elif label == "400-600":
            grid_area = 462  # 平均値
        else:  # >= 600
            grid_area = 792  # 平均値

        for _ in range(num_samples):
            # base_numを決定
            rand = random.random()
            if rand < 0.25:
                base_num = random.randint(2, 4)
            elif rand < 0.47:
                base_num = random.randint(5, 8)
            elif rand < 0.67:
                base_num = random.randint(9, 12)
            elif rand < 0.85:
                base_num = random.randint(13, 20)
            elif rand < 0.95:
                base_num = random.randint(21, 30)
            elif rand < 0.98:
                base_num = random.randint(31, 50)
            else:
                base_num = random.randint(51, 100)

            # 面積に応じて調整
            if grid_area < 100:
                if base_num >= 13:
                    adjusted_num = max(8, int(base_num * 0.7))
                else:
                    adjusted_num = max(2, int(base_num * 0.6))
            elif grid_area < 200:
                if base_num >= 13:
                    adjusted_num = max(10, int(base_num * 0.85))
                else:
                    adjusted_num = max(2, int(base_num * 0.8))
            elif grid_area < 300:
                adjusted_num = base_num
            elif grid_area < 400:
                adjusted_num = min(50, int(base_num * 1.1))
            elif grid_area < 600:
                adjusted_num = min(70, int(base_num * 1.3))
            else:
                adjusted_num = min(100, int(base_num * 1.5))

            # 上限チェック
            max_objects = max(50, grid_area // 3)
            adjusted_num = min(adjusted_num, max_objects)
            adjusted_num = max(2, adjusted_num)

            results_by_area[label].append(adjusted_num)

    return results_by_area


def proposed_implementation_simulation(num_samples=10000):
    """提案した実装をシミュレート（ARC-AGI2実データに基づく）"""
    results_by_area = defaultdict(list)

    # ARC-AGI2実データに基づく分布
    # 各面積範囲で、実データの統計（平均、標準偏差、範囲）に基づいて分布を生成
    distributions = {
        "< 100": {
            'mean': 7.87,
            'std': 6.64,
            'min': 1,
            'max': 65,
            'median': 6.0,
        },
        "100-200": {
            'mean': 10.68,
            'std': 10.31,
            'min': 1,
            'max': 109,
            'median': 7.0,
        },
        "200-300": {
            'mean': 19.72,
            'std': 28.82,
            'min': 1,
            'max': 206,
            'median': 10.0,
        },
        "300-400": {
            'mean': 29.08,
            'std': 45.44,
            'min': 2,
            'max': 298,
            'median': 14.0,
        },
        "400-600": {
            'mean': 44.98,
            'std': 71.63,
            'min': 2,
            'max': 500,
            'median': 19.0,
        },
        ">= 600": {
            'mean': 109.93,
            'std': 165.88,
            'min': 2,
            'max': 785,
            'median': 30.5,
        },
    }

    for label, dist in distributions.items():
        for _ in range(num_samples):
            # 対数正規分布を使用して実データの分布を近似
            # 平均と標準偏差から対数正規分布のパラメータを計算
            log_mean = np.log(dist['mean'] ** 2 / np.sqrt(dist['std'] ** 2 + dist['mean'] ** 2))
            log_std = np.sqrt(np.log(1 + (dist['std'] / dist['mean']) ** 2))

            # 対数正規分布からサンプル
            num_objects = int(np.random.lognormal(log_mean, log_std))

            # 範囲内に制限
            num_objects = max(dist['min'], min(dist['max'], num_objects))
            num_objects = max(2, num_objects)  # 最小2個

            results_by_area[label].append(num_objects)

    return results_by_area


def analyze_diversity(results_by_area):
    """多様性を分析（標準偏差、範囲、一意の値の数）"""
    diversity_stats = {}

    for label, values in results_by_area.items():
        if values:
            diversity_stats[label] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'unique_count': len(set(values)),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,  # 変動係数
            }

    return diversity_stats


def analyze_density_alignment(results_by_area, arc_agi2_data):
    """密度のARC-AGI2実データとの整合性を分析"""
    alignment_stats = {}

    area_representatives = {
        "< 100": 42,
        "100-200": 129,
        "200-300": 243,
        "300-400": 344,
        "400-600": 462,
        ">= 600": 792,
    }

    for label, values in results_by_area.items():
        if values and label in arc_agi2_data:
            mean_objects = np.mean(values)
            grid_area = area_representatives[label]
            objects_per_area = mean_objects / grid_area if grid_area > 0 else 0

            arc_mean = arc_agi2_data[label]['mean']
            arc_ratio = arc_agi2_data[label]['objects_per_area']

            alignment_stats[label] = {
                'simulated_mean': mean_objects,
                'arc_mean': arc_mean,
                'difference': mean_objects - arc_mean,
                'difference_pct': ((mean_objects - arc_mean) / arc_mean * 100) if arc_mean > 0 else 0,
                'simulated_ratio': objects_per_area,
                'arc_ratio': arc_ratio,
                'ratio_difference': objects_per_area - arc_ratio,
                'ratio_difference_pct': ((objects_per_area - arc_ratio) / arc_ratio * 100) if arc_ratio > 0 else 0,
            }

    return alignment_stats


def print_comparison(current_diversity, proposed_diversity, current_alignment, proposed_alignment):
    """比較結果を表示"""
    print("\n" + "=" * 80)
    print("【多様性の比較】")
    print("=" * 80)

    print("\n【現在の実装】")
    print("-" * 80)
    for label in ["< 100", "100-200", "200-300", "300-400", "400-600", ">= 600"]:
        if label in current_diversity:
            stats = current_diversity[label]
            print(f"{label}:")
            print(f"  平均={stats['mean']:.2f}, 標準偏差={stats['std']:.2f}, 範囲={stats['range']}, 一意の値={stats['unique_count']}, 変動係数={stats['cv']:.3f}")

    print("\n【提案した実装】")
    print("-" * 80)
    for label in ["< 100", "100-200", "200-300", "300-400", "400-600", ">= 600"]:
        if label in proposed_diversity:
            stats = proposed_diversity[label]
            print(f"{label}:")
            print(f"  平均={stats['mean']:.2f}, 標準偏差={stats['std']:.2f}, 範囲={stats['range']}, 一意の値={stats['unique_count']}, 変動係数={stats['cv']:.3f}")

    print("\n" + "=" * 80)
    print("【密度のARC-AGI2実データとの整合性】")
    print("=" * 80)

    print("\n【現在の実装】")
    print("-" * 80)
    for label in ["< 100", "100-200", "200-300", "300-400", "400-600", ">= 600"]:
        if label in current_alignment:
            stats = current_alignment[label]
            print(f"{label}:")
            print(f"  平均オブジェクト数: シミュレート={stats['simulated_mean']:.2f}, ARC-AGI2={stats['arc_mean']:.2f}, 差={stats['difference']:+.2f} ({stats['difference_pct']:+.1f}%)")
            print(f"  オブジェクト数/面積比: シミュレート={stats['simulated_ratio']:.4f}, ARC-AGI2={stats['arc_ratio']:.4f}, 差={stats['ratio_difference']:+.4f} ({stats['ratio_difference_pct']:+.1f}%)")

    print("\n【提案した実装】")
    print("-" * 80)
    for label in ["< 100", "100-200", "200-300", "300-400", "400-600", ">= 600"]:
        if label in proposed_alignment:
            stats = proposed_alignment[label]
            print(f"{label}:")
            print(f"  平均オブジェクト数: シミュレート={stats['simulated_mean']:.2f}, ARC-AGI2={stats['arc_mean']:.2f}, 差={stats['difference']:+.2f} ({stats['difference_pct']:+.1f}%)")
            print(f"  オブジェクト数/面積比: シミュレート={stats['simulated_ratio']:.4f}, ARC-AGI2={stats['arc_ratio']:.4f}, 差={stats['ratio_difference']:+.4f} ({stats['ratio_difference_pct']:+.1f}%)")

    print("\n" + "=" * 80)
    print("【結論】")
    print("=" * 80)

    # 多様性の比較
    print("\n【多様性について】")
    current_cv_avg = np.mean([stats['cv'] for stats in current_diversity.values()])
    proposed_cv_avg = np.mean([stats['cv'] for stats in proposed_diversity.values()])
    print(f"現在の実装の平均変動係数: {current_cv_avg:.3f}")
    print(f"提案した実装の平均変動係数: {proposed_cv_avg:.3f}")
    if proposed_cv_avg >= current_cv_avg * 0.9:
        print("→ 多様性はほぼ維持される（変動係数の差が10%以内）")
    else:
        print("→ 多様性がやや減少する可能性がある")

    # 密度の整合性
    print("\n【密度の整合性について】")
    current_ratio_diff_avg = np.mean([abs(stats['ratio_difference_pct']) for stats in current_alignment.values()])
    proposed_ratio_diff_avg = np.mean([abs(stats['ratio_difference_pct']) for stats in proposed_alignment.values()])
    print(f"現在の実装の平均比率差: {current_ratio_diff_avg:.1f}%")
    print(f"提案した実装の平均比率差: {proposed_ratio_diff_avg:.1f}%")
    if proposed_ratio_diff_avg < current_ratio_diff_avg:
        print(f"→ 密度の差異が{current_ratio_diff_avg - proposed_ratio_diff_avg:.1f}%ポイント改善される")
    else:
        print("→ 密度の差異は改善されない")


def main():
    # ARC-AGI2実データ
    arc_agi2_data = {
        "< 100": {'mean': 7.87, 'objects_per_area': 0.1895},
        "100-200": {'mean': 10.68, 'objects_per_area': 0.0826},
        "200-300": {'mean': 19.72, 'objects_per_area': 0.0813},
        "300-400": {'mean': 29.08, 'objects_per_area': 0.0845},
        "400-600": {'mean': 44.98, 'objects_per_area': 0.0974},
        ">= 600": {'mean': 109.93, 'objects_per_area': 0.1389},
    }

    print("現在の実装をシミュレート中...")
    current_results = current_implementation_simulation(num_samples=10000)

    print("提案した実装をシミュレート中...")
    proposed_results = proposed_implementation_simulation(num_samples=10000)

    print("\n多様性を分析中...")
    current_diversity = analyze_diversity(current_results)
    proposed_diversity = analyze_diversity(proposed_results)

    print("密度の整合性を分析中...")
    current_alignment = analyze_density_alignment(current_results, arc_agi2_data)
    proposed_alignment = analyze_density_alignment(proposed_results, arc_agi2_data)

    print_comparison(current_diversity, proposed_diversity, current_alignment, proposed_alignment)


if __name__ == '__main__':
    main()

