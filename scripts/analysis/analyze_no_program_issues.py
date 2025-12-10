"""
プログラムなしモードの問題点とARC-AGI2との違いを詳細に分析するスクリプト
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_no_program_issues():
    """プログラムなしモードの問題点を分析"""

    # 生成された統計を読み込み
    generated_file = Path("outputs/input_grid_comparison_no_program_analysis/generated_statistics.json")
    if not generated_file.exists():
        print(f"統計ファイルが見つかりません: {generated_file}")
        return

    with open(generated_file, 'r', encoding='utf-8') as f:
        generated_stats = json.load(f)

    # ARC-AGI2統計を読み込み
    arc_file = Path("outputs/input_grid_comparison_after_condition4_removal/arc_agi2_statistics.json")
    if not arc_file.exists():
        print(f"ARC-AGI2統計ファイルが見つかりません: {arc_file}")
        return

    with open(arc_file, 'r', encoding='utf-8') as f:
        arc_stats = json.load(f)

    print("=" * 80)
    print("プログラムなしモードの問題点とARC-AGI2との違い")
    print("=" * 80)

    # 1. 色数の分析
    print("\n1. 色数の分析")
    print("-" * 80)

    gen_total_colors = [s['num_colors_total'] for s in generated_stats]
    gen_object_colors = [s['num_colors_object'] for s in generated_stats]

    arc_total_colors = [s['num_colors_total'] for s in arc_stats if 'num_colors_total' in s]
    arc_object_colors = [s['num_colors_object'] for s in arc_stats if 'num_colors_object' in s]

    print(f"生成データ (n={len(generated_stats)}):")
    print(f"  総色数: 平均={np.mean(gen_total_colors):.2f}, 中央値={np.median(gen_total_colors):.2f}, 分布={dict(Counter(gen_total_colors))}")
    print(f"  オブジェクト色数: 平均={np.mean(gen_object_colors):.2f}, 中央値={np.median(gen_object_colors):.2f}, 分布={dict(Counter(gen_object_colors))}")

    print(f"\nARC-AGI2 (n={len(arc_stats)}):")
    print(f"  総色数: 平均={np.mean(arc_total_colors):.2f}, 中央値={np.median(arc_total_colors):.2f}, 分布={dict(Counter(arc_total_colors))}")
    print(f"  オブジェクト色数: 平均={np.mean(arc_object_colors):.2f}, 中央値={np.median(arc_object_colors):.2f}, 分布={dict(Counter(arc_object_colors))}")

    # 2. オブジェクトピクセル比率の分析
    print("\n2. オブジェクトピクセル比率の分析")
    print("-" * 80)

    gen_ratios = [s['object_pixel_ratio'] for s in generated_stats]
    arc_ratios = [s['object_pixel_ratio'] for s in arc_stats if 'object_pixel_ratio' in s]

    print(f"生成データ:")
    print(f"  平均={np.mean(gen_ratios):.4f}, 中央値={np.median(gen_ratios):.4f}, 最小={min(gen_ratios):.4f}, 最大={max(gen_ratios):.4f}, 標準偏差={np.std(gen_ratios):.4f}")
    print(f"  分布:")
    ratio_bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
    for low, high in ratio_bins:
        count = sum(1 for r in gen_ratios if low <= r < high)
        pct = count / len(gen_ratios) * 100
        print(f"    [{low:.2f}, {high:.2f}): {count}個 ({pct:.1f}%)")

    print(f"\nARC-AGI2:")
    print(f"  平均={np.mean(arc_ratios):.4f}, 中央値={np.median(arc_ratios):.4f}, 最小={min(arc_ratios):.4f}, 最大={max(arc_ratios):.4f}, 標準偏差={np.std(arc_ratios):.4f}")
    print(f"  分布:")
    for low, high in ratio_bins:
        count = sum(1 for r in arc_ratios if low <= r < high)
        pct = count / len(arc_ratios) * 100
        print(f"    [{low:.2f}, {high:.2f}): {count}個 ({pct:.1f}%)")

    # 3. グリッドサイズの分析
    print("\n3. グリッドサイズの分析")
    print("-" * 80)

    gen_areas = [s['area'] for s in generated_stats]
    gen_widths = [s['width'] for s in generated_stats]
    gen_heights = [s['height'] for s in generated_stats]

    arc_areas = [s['area'] for s in arc_stats if 'area' in s]
    arc_widths = [s['width'] for s in arc_stats if 'width' in s]
    arc_heights = [s['height'] for s in arc_stats if 'height' in s]

    print(f"生成データ:")
    print(f"  面積: 平均={np.mean(gen_areas):.2f}, 中央値={np.median(gen_areas):.2f}")
    print(f"  幅: 平均={np.mean(gen_widths):.2f}, 中央値={np.median(gen_widths):.2f}")
    print(f"  高さ: 平均={np.mean(gen_heights):.2f}, 中央値={np.median(gen_heights):.2f}")

    print(f"\nARC-AGI2:")
    print(f"  面積: 平均={np.mean(arc_areas):.2f}, 中央値={np.median(arc_areas):.2f}")
    print(f"  幅: 平均={np.mean(arc_widths):.2f}, 中央値={np.median(arc_widths):.2f}")
    print(f"  高さ: 平均={np.mean(arc_heights):.2f}, 中央値={np.median(arc_heights):.2f}")

    # 4. 背景色の分析
    print("\n4. 背景色の分析")
    print("-" * 80)

    gen_bg = [s['background_color'] for s in generated_stats]
    arc_bg = [s['background_color'] for s in arc_stats if 'background_color' in s]

    print(f"生成データ: {dict(Counter(gen_bg))}")
    print(f"ARC-AGI2: {dict(Counter(arc_bg))}")

    # 5. 問題点の要約
    print("\n5. 主な問題点")
    print("-" * 80)

    issues = []

    # 色数の問題
    gen_2color_pct = sum(1 for c in gen_total_colors if c == 2) / len(gen_total_colors) * 100
    arc_2color_pct = sum(1 for c in arc_total_colors if c == 2) / len(arc_total_colors) * 100
    if gen_2color_pct > arc_2color_pct * 2:
        issues.append(f"色数が非常に少ない: 2色が{gen_2color_pct:.1f}% (ARC-AGI2: {arc_2color_pct:.1f}%)")

    # オブジェクトピクセル比率の問題
    gen_mean_ratio = np.mean(gen_ratios)
    arc_mean_ratio = np.mean(arc_ratios)
    if gen_mean_ratio < arc_mean_ratio * 0.7:
        issues.append(f"オブジェクトピクセル比率が低い: 平均{gen_mean_ratio:.4f} (ARC-AGI2: {arc_mean_ratio:.4f})")

    # 成功率の問題
    print(f"成功率: {len(generated_stats)}/1000 = {len(generated_stats)/10:.1f}%")
    if len(generated_stats) < 500:
        issues.append(f"成功率が低い: {len(generated_stats)/10:.1f}%")

    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

    # 結果を保存
    output_file = Path("outputs/input_grid_comparison_no_program_analysis/issues_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    analysis = {
        'generated_count': len(generated_stats),
        'arc_count': len(arc_stats),
        'issues': issues,
        'color_statistics': {
            'generated': {
                'total_mean': float(np.mean(gen_total_colors)),
                'total_median': float(np.median(gen_total_colors)),
                'total_distribution': dict(Counter(gen_total_colors)),
                'object_mean': float(np.mean(gen_object_colors)),
                'object_median': float(np.median(gen_object_colors)),
                'object_distribution': dict(Counter(gen_object_colors))
            },
            'arc_agi2': {
                'total_mean': float(np.mean(arc_total_colors)),
                'total_median': float(np.median(arc_total_colors)),
                'total_distribution': dict(Counter(arc_total_colors)),
                'object_mean': float(np.mean(arc_object_colors)),
                'object_median': float(np.median(arc_object_colors)),
                'object_distribution': dict(Counter(arc_object_colors))
            }
        },
        'object_pixel_ratio': {
            'generated': {
                'mean': float(np.mean(gen_ratios)),
                'median': float(np.median(gen_ratios)),
                'min': float(min(gen_ratios)),
                'max': float(max(gen_ratios)),
                'std': float(np.std(gen_ratios))
            },
            'arc_agi2': {
                'mean': float(np.mean(arc_ratios)),
                'median': float(np.median(arc_ratios)),
                'min': float(min(arc_ratios)),
                'max': float(max(arc_ratios)),
                'std': float(np.std(arc_ratios))
            }
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n分析結果を保存: {output_file}")

if __name__ == "__main__":
    analyze_no_program_issues()

