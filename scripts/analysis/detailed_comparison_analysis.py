"""condition4削除後の生成タスクとARC-AGI2データセットの詳細比較分析"""

import json
from pathlib import Path


def calculate_percentage(value, total):
    """パーセンテージを計算"""
    return (value / total * 100) if total > 0 else 0


def format_distribution(dist_dict, total):
    """分布を整形して表示"""
    items = sorted(dist_dict.items(), key=lambda x: int(x[0]))
    lines = []
    for key, value in items:
        pct = calculate_percentage(value, total)
        lines.append(f"    {key}: {value} ({pct:.2f}%)")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("condition4削除後の生成タスク vs ARC-AGI2データセット 詳細比較分析")
    print("=" * 80)

    # 比較結果を読み込み
    comparison_file = Path("outputs/input_grid_comparison_after_condition4_removal/comparison_result.json")
    with open(comparison_file, 'r', encoding='utf-8') as f:
        comparison = json.load(f)

    gen_count = comparison['generated_count']
    arc_count = comparison['arc_count']
    comps = comparison['comparisons']

    print(f"\n【タスク数】")
    print(f"  生成タスク: {gen_count}")
    print(f"  ARC-AGI2: {arc_count}")
    print(f"  比率: {calculate_percentage(gen_count, arc_count):.2f}%")

    # 1. グリッドサイズの比較
    print("\n" + "=" * 80)
    print("【1. グリッドサイズ】")
    print("=" * 80)

    gen_size = comps['grid_size']['generated']
    arc_size = comps['grid_size']['arc_agi2']

    print(f"\n幅 (width):")
    print(f"  生成: 最小={gen_size['width']['min']}, 最大={gen_size['width']['max']}, "
          f"平均={gen_size['width']['mean']:.2f}, 中央値={gen_size['width']['median']}, "
          f"標準偏差={gen_size['width']['std']:.2f}")
    print(f"  ARC-AGI2: 最小={arc_size['width']['min']}, 最大={arc_size['width']['max']}, "
          f"平均={arc_size['width']['mean']:.2f}, 中央値={arc_size['width']['median']}, "
          f"標準偏差={arc_size['width']['std']:.2f}")
    print(f"  差: 平均{gen_size['width']['mean'] - arc_size['width']['mean']:+.2f}, "
          f"標準偏差{gen_size['width']['std'] - arc_size['width']['std']:+.2f}")

    print(f"\n高さ (height):")
    print(f"  生成: 最小={gen_size['height']['min']}, 最大={gen_size['height']['max']}, "
          f"平均={gen_size['height']['mean']:.2f}, 中央値={gen_size['height']['median']}, "
          f"標準偏差={gen_size['height']['std']:.2f}")
    print(f"  ARC-AGI2: 最小={arc_size['height']['min']}, 最大={arc_size['height']['max']}, "
          f"平均={arc_size['height']['mean']:.2f}, 中央値={arc_size['height']['median']}, "
          f"標準偏差={arc_size['height']['std']:.2f}")
    print(f"  差: 平均{gen_size['height']['mean'] - arc_size['height']['mean']:+.2f}, "
          f"標準偏差{gen_size['height']['std'] - arc_size['height']['std']:+.2f}")

    print(f"\n面積 (area):")
    print(f"  生成: 最小={gen_size['area']['min']}, 最大={gen_size['area']['max']}, "
          f"平均={gen_size['area']['mean']:.2f}, 中央値={gen_size['area']['median']}, "
          f"標準偏差={gen_size['area']['std']:.2f}")
    print(f"  ARC-AGI2: 最小={arc_size['area']['min']}, 最大={arc_size['area']['max']}, "
          f"平均={arc_size['area']['mean']:.2f}, 中央値={arc_size['area']['median']}, "
          f"標準偏差={arc_size['area']['std']:.2f}")
    print(f"  差: 平均{gen_size['area']['mean'] - arc_size['area']['mean']:+.2f} "
          f"({calculate_percentage(gen_size['area']['mean'] - arc_size['area']['mean'], arc_size['area']['mean']):+.2f}%), "
          f"中央値{gen_size['area']['median'] - arc_size['area']['median']:+.0f}")

    # 2. 色数の比較
    print("\n" + "=" * 80)
    print("【2. 色数】")
    print("=" * 80)

    gen_color = comps['color_count']['generated']
    arc_color = comps['color_count']['arc_agi2']

    print(f"\n総色数 (total):")
    print(f"  生成: 最小={gen_color['total']['min']}, 最大={gen_color['total']['max']}, "
          f"平均={gen_color['total']['mean']:.2f}, 中央値={gen_color['total']['median']}")
    print(f"  ARC-AGI2: 最小={arc_color['total']['min']}, 最大={arc_color['total']['max']}, "
          f"平均={arc_color['total']['mean']:.2f}, 中央値={arc_color['total']['median']}")
    print(f"  差: 平均{gen_color['total']['mean'] - arc_color['total']['mean']:+.2f} "
          f"({calculate_percentage(gen_color['total']['mean'] - arc_color['total']['mean'], arc_color['total']['mean']):+.2f}%)")

    print(f"\n  生成タスクの分布:")
    print(format_distribution(gen_color['total']['distribution'], gen_count))
    print(f"\n  ARC-AGI2の分布:")
    print(format_distribution(arc_color['total']['distribution'], arc_count))

    print(f"\nオブジェクト色数 (object):")
    print(f"  生成: 最小={gen_color['object']['min']}, 最大={gen_color['object']['max']}, "
          f"平均={gen_color['object']['mean']:.2f}, 中央値={gen_color['object']['median']}")
    print(f"  ARC-AGI2: 最小={arc_color['object']['min']}, 最大={arc_color['object']['max']}, "
          f"平均={arc_color['object']['mean']:.2f}, 中央値={arc_color['object']['median']}")
    print(f"  差: 平均{gen_color['object']['mean'] - arc_color['object']['mean']:+.2f} "
          f"({calculate_percentage(gen_color['object']['mean'] - arc_color['object']['mean'], arc_color['object']['mean']):+.2f}%)")

    print(f"\n  生成タスクの分布:")
    print(format_distribution(gen_color['object']['distribution'], gen_count))
    print(f"\n  ARC-AGI2の分布:")
    print(format_distribution(arc_color['object']['distribution'], arc_count))

    # 3. 背景色の比較
    print("\n" + "=" * 80)
    print("【3. 背景色の分布】")
    print("=" * 80)

    gen_bg = comps['background_color']['generated']
    arc_bg = comps['background_color']['arc_agi2']

    print(f"\n生成タスク:")
    print(format_distribution(gen_bg, gen_count))
    print(f"\nARC-AGI2:")
    print(format_distribution(arc_bg, arc_count))

    # 主要な背景色（0と7）の比較
    bg0_gen = gen_bg.get('0', 0)
    bg0_arc = arc_bg.get('0', 0)
    bg7_gen = gen_bg.get('7', 0)
    bg7_arc = arc_bg.get('7', 0)

    print(f"\n背景色0:")
    print(f"  生成: {bg0_gen} ({calculate_percentage(bg0_gen, gen_count):.2f}%)")
    print(f"  ARC-AGI2: {bg0_arc} ({calculate_percentage(bg0_arc, arc_count):.2f}%)")
    print(f"  差: {calculate_percentage(bg0_gen, gen_count) - calculate_percentage(bg0_arc, arc_count):+.2f}ポイント")

    print(f"\n背景色7:")
    print(f"  生成: {bg7_gen} ({calculate_percentage(bg7_gen, gen_count):.2f}%)")
    print(f"  ARC-AGI2: {bg7_arc} ({calculate_percentage(bg7_arc, arc_count):.2f}%)")
    print(f"  差: {calculate_percentage(bg7_gen, gen_count) - calculate_percentage(bg7_arc, arc_count):+.2f}ポイント")

    # 4. オブジェクトピクセル比率の比較
    print("\n" + "=" * 80)
    print("【4. オブジェクトピクセル比率】")
    print("=" * 80)

    gen_ratio = comps['object_pixel_ratio']['generated']
    arc_ratio = comps['object_pixel_ratio']['arc_agi2']

    print(f"\n生成タスク:")
    print(f"  最小={gen_ratio['min']:.6f} ({gen_ratio['min']*100:.4f}%)")
    print(f"  最大={gen_ratio['max']:.6f} ({gen_ratio['max']*100:.2f}%)")
    print(f"  平均={gen_ratio['mean']:.6f} ({gen_ratio['mean']*100:.2f}%)")
    print(f"  中央値={gen_ratio['median']:.6f} ({gen_ratio['median']*100:.2f}%)")
    print(f"  標準偏差={gen_ratio['std']:.6f} ({gen_ratio['std']*100:.2f}%)")

    print(f"\nARC-AGI2:")
    print(f"  最小={arc_ratio['min']:.6f} ({arc_ratio['min']*100:.4f}%)")
    print(f"  最大={arc_ratio['max']:.6f} ({arc_ratio['max']*100:.2f}%)")
    print(f"  平均={arc_ratio['mean']:.6f} ({arc_ratio['mean']*100:.2f}%)")
    print(f"  中央値={arc_ratio['median']:.6f} ({arc_ratio['median']*100:.2f}%)")
    print(f"  標準偏差={arc_ratio['std']:.6f} ({arc_ratio['std']*100:.2f}%)")

    print(f"\n差:")
    print(f"  平均: {gen_ratio['mean'] - arc_ratio['mean']:+.6f} "
          f"({calculate_percentage(gen_ratio['mean'] - arc_ratio['mean'], arc_ratio['mean']):+.2f}%)")
    print(f"  中央値: {gen_ratio['median'] - arc_ratio['median']:+.6f} "
          f"({calculate_percentage(gen_ratio['median'] - arc_ratio['median'], arc_ratio['median']):+.2f}%)")
    print(f"  標準偏差: {gen_ratio['std'] - arc_ratio['std']:+.6f} "
          f"({calculate_percentage(gen_ratio['std'] - arc_ratio['std'], arc_ratio['std']):+.2f}%)")

    # まとめと改善提案
    print("\n" + "=" * 80)
    print("【まとめと改善提案】")
    print("=" * 80)

    issues = []
    improvements = []

    # 面積の差
    area_diff_pct = calculate_percentage(gen_size['area']['mean'] - arc_size['area']['mean'], arc_size['area']['mean'])
    if abs(area_diff_pct) > 10:
        issues.append(f"グリッド面積が{'大きい' if area_diff_pct > 0 else '小さい'} ({area_diff_pct:+.1f}%)")
        improvements.append("グリッドサイズ分布の調整が必要")

    # 色数の差
    color_diff_pct = calculate_percentage(gen_color['total']['mean'] - arc_color['total']['mean'], arc_color['total']['mean'])
    if abs(color_diff_pct) > 20:
        issues.append(f"総色数が少ない ({color_diff_pct:+.1f}%)")
        improvements.append("色数分布の調整が必要（特に多色のタスクを増やす）")

    obj_color_diff_pct = calculate_percentage(gen_color['object']['mean'] - arc_color['object']['mean'], arc_color['object']['mean'])
    if abs(obj_color_diff_pct) > 30:
        issues.append(f"オブジェクト色数が少ない ({obj_color_diff_pct:+.1f}%)")
        improvements.append("オブジェクト色数分布の調整が必要")

    # 背景色0の過多
    bg0_diff = calculate_percentage(bg0_gen, gen_count) - calculate_percentage(bg0_arc, arc_count)
    if bg0_diff > 10:
        issues.append(f"背景色0が多すぎる (+{bg0_diff:.1f}ポイント)")
        improvements.append("背景色分布の調整が必要（背景色0を減らし、他の色を増やす）")

    # オブジェクトピクセル比率の差
    ratio_diff_pct = calculate_percentage(gen_ratio['mean'] - arc_ratio['mean'], arc_ratio['mean'])
    if abs(ratio_diff_pct) > 30:
        issues.append(f"オブジェクトピクセル比率が{'高い' if ratio_diff_pct > 0 else '低い'} ({ratio_diff_pct:+.1f}%)")
        improvements.append("オブジェクト密度の調整が必要")

    if issues:
        print("\n【主な問題点】")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    if improvements:
        print("\n【改善提案】")
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")
    else:
        print("\n特に大きな問題は見つかりませんでした。")

    print("=" * 80)


if __name__ == "__main__":
    main()
