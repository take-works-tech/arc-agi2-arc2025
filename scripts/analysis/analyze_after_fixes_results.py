"""
修正後の結果を分析するスクリプト
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_after_fixes():
    """修正後の結果を分析"""

    # 生成された統計を読み込み
    generated_file = Path("outputs/input_grid_comparison_after_fixes/generated_statistics.json")
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

    # 比較結果を読み込み
    comparison_file = Path("outputs/input_grid_comparison_after_fixes/comparison_result.json")
    if comparison_file.exists():
        with open(comparison_file, 'r', encoding='utf-8') as f:
            comparison = json.load(f)
            print(f"生成タスク数: {comparison['generated_count']}/1000")
            print(f"成功率: {comparison['generated_count']/10:.1f}%\n")

    print("=" * 80)
    print("修正後の結果分析")
    print("=" * 80)

    # 1. 色数の分析
    print("\n1. 色数の分析")
    print("-" * 80)

    gen_total_colors = [s['num_colors_total'] for s in generated_stats]
    gen_object_colors = [s['num_colors_object'] for s in generated_stats]

    arc_total_colors = [s['num_colors_total'] for s in arc_stats if 'num_colors_total' in s]
    arc_object_colors = [s['num_colors_object'] for s in arc_stats if 'num_colors_object' in s]

    print(f"生成データ (n={len(generated_stats)}):")
    print(f"  総色数: 平均={np.mean(gen_total_colors):.2f}, 中央値={np.median(gen_total_colors):.2f}")
    print(f"  分布: {dict(Counter(gen_total_colors))}")
    print(f"  オブジェクト色数: 平均={np.mean(gen_object_colors):.2f}, 中央値={np.median(gen_object_colors):.2f}")
    print(f"  分布: {dict(Counter(gen_object_colors))}")

    print(f"\nARC-AGI2 (n={len(arc_stats)}):")
    print(f"  総色数: 平均={np.mean(arc_total_colors):.2f}, 中央値={np.median(arc_total_colors):.2f}")
    print(f"  分布: {dict(Counter(arc_total_colors))}")
    print(f"  オブジェクト色数: 平均={np.mean(arc_object_colors):.2f}, 中央値={np.median(arc_object_colors):.2f}")
    print(f"  分布: {dict(Counter(arc_object_colors))}")

    # 2. オブジェクトピクセル比率の分析
    print("\n2. オブジェクトピクセル比率の分析")
    print("-" * 80)

    gen_ratios = [s['object_pixel_ratio'] for s in generated_stats]
    arc_ratios = [s['object_pixel_ratio'] for s in arc_stats if 'object_pixel_ratio' in s]

    print(f"生成データ:")
    print(f"  平均={np.mean(gen_ratios):.4f}, 中央値={np.median(gen_ratios):.4f}, 最小={min(gen_ratios):.4f}, 最大={max(gen_ratios):.4f}")

    print(f"\nARC-AGI2:")
    print(f"  平均={np.mean(arc_ratios):.4f}, 中央値={np.median(arc_ratios):.4f}, 最小={min(arc_ratios):.4f}, 最大={max(arc_ratios):.4f}")

    # 3. 改善の比較
    print("\n3. 修正前後の比較")
    print("-" * 80)

    print("\n成功率:")
    print(f"  修正前: 8.6% (86/1000)")
    print(f"  修正後: {len(generated_stats)/10:.1f}% ({len(generated_stats)}/1000)")
    print(f"  改善: +{len(generated_stats)/10 - 8.6:.1f}%")

    print("\n色数:")
    print(f"  修正前: 平均2.01色 (オブジェクト色数: 平均1.01色)")
    print(f"  修正後: 平均{np.mean(gen_total_colors):.2f}色 (オブジェクト色数: 平均{np.mean(gen_object_colors):.2f}色)")

    print("\nオブジェクトピクセル比率:")
    print(f"  修正前: 平均0.1143")
    print(f"  修正後: 平均{np.mean(gen_ratios):.4f}")

    # 4. 残っている問題
    print("\n4. 残っている問題")
    print("-" * 80)

    issues = []

    # 色数の問題
    gen_2color_pct = sum(1 for c in gen_total_colors if c == 2) / len(gen_total_colors) * 100
    arc_2color_pct = sum(1 for c in arc_total_colors if c == 2) / len(arc_total_colors) * 100
    if gen_2color_pct > arc_2color_pct * 1.5:
        issues.append(f"色数がまだ少ない: 2色が{gen_2color_pct:.1f}% (ARC-AGI2: {arc_2color_pct:.1f}%)")

    # オブジェクトピクセル比率の問題
    gen_mean_ratio = np.mean(gen_ratios)
    arc_mean_ratio = np.mean(arc_ratios)
    if gen_mean_ratio < arc_mean_ratio * 0.7:
        issues.append(f"オブジェクトピクセル比率がまだ低い: 平均{gen_mean_ratio:.4f} (ARC-AGI2: {arc_mean_ratio:.4f})")

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("主要な問題は解決されました！")

    # 結果を保存
    output_file = Path("outputs/input_grid_comparison_after_fixes/after_fixes_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    analysis = {
        'generated_count': len(generated_stats),
        'arc_count': len(arc_stats),
        'success_rate_before': 8.6,
        'success_rate_after': len(generated_stats) / 10,
        'improvement': len(generated_stats) / 10 - 8.6,
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
                'max': float(max(gen_ratios))
            },
            'arc_agi2': {
                'mean': float(np.mean(arc_ratios)),
                'median': float(np.median(arc_ratios)),
                'min': float(min(arc_ratios)),
                'max': float(max(arc_ratios))
            }
        },
        'remaining_issues': issues
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n分析結果を保存: {output_file}")

if __name__ == "__main__":
    analyze_after_fixes()

