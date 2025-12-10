"""生成されたタスクのcondition4関連統計を分析するスクリプト"""

import json
from pathlib import Path


def main():
    # 生成された統計データを読み込み
    stats_file = Path("outputs/input_grid_comparison/generated_statistics.json")
    if not stats_file.exists():
        print(f"エラー: {stats_file} が見つかりません")
        return

    with open(stats_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # オブジェクトピクセル比率を抽出
    ratios = []
    zero_ratios = []
    below_3pct = []
    below_1pct = []
    below_0_5pct = []

    for entry in data:
        if not isinstance(entry, dict):
            continue

        ratio = entry.get('object_pixel_ratio', None)
        if ratio is None:
            continue

        ratios.append(ratio)

        # 分類
        if ratio == 0:
            zero_ratios.append(entry)
        elif 0 < ratio < 0.03:
            below_3pct.append(entry)
        if 0 < ratio < 0.01:
            below_1pct.append(entry)
        if 0 < ratio < 0.005:
            below_0_5pct.append(entry)

    # 統計を出力
    total = len(ratios)
    print("=" * 60)
    print("生成されたタスクのcondition4関連統計分析")
    print("=" * 60)
    print(f"\n総タスク数: {total}")

    if total == 0:
        print("データがありません")
        return

    print(f"\n【0%のタスク】")
    print(f"  数: {len(zero_ratios)} ({len(zero_ratios)/total*100:.2f}%)")

    print(f"\n【1%未満（0より大きい）のタスク】")
    print(f"  数: {len(below_1pct)} ({len(below_1pct)/total*100:.2f}%)")

    print(f"\n【0.5%未満（0より大きい）のタスク】")
    print(f"  数: {len(below_0_5pct)} ({len(below_0_5pct)/total*100:.2f}%)")

    print(f"\n【3%未満（0より大きい）のタスク】")
    print(f"  数: {len(below_3pct)} ({len(below_3pct)/total*100:.2f}%)")

    if ratios:
        non_zero_ratios = [r for r in ratios if r > 0]
        if non_zero_ratios:
            print(f"\n【統計情報】")
            print(f"  最小値（0以外）: {min(non_zero_ratios):.6f} ({min(non_zero_ratios)*100:.4f}%)")
            print(f"  平均値: {sum(ratios)/len(ratios):.6f} ({sum(ratios)/len(ratios)*100:.2f}%)")
            print(f"  中央値（0以外）: {sorted(non_zero_ratios)[len(non_zero_ratios)//2]:.6f} ({sorted(non_zero_ratios)[len(non_zero_ratios)//2]*100:.4f}%)")

    print("\n" + "=" * 60)
    print("【推奨事項】")

    # 1%未満が極端に多い場合は、condition4を緩和することを推奨
    if len(below_1pct) / total > 0.10:  # 10%以上
        print("  condition4の閾値を3%から1%に緩和することを推奨します")
        print(f"  （1%未満のタスクが{len(below_1pct)/total*100:.2f}%と多いため）")
    elif len(below_3pct) / total > 0.15:  # 15%以上
        print("  condition4を削除または閾値を緩和することを推奨します")
        print(f"  （3%未満のタスクが{len(below_3pct)/total*100:.2f}%と多いため）")
    elif len(below_0_5pct) / total < 0.01:  # 1%未満
        print("  condition4の閾値を0.5%に厳格化することを推奨します")
        print(f"  （0.5%未満のタスクが{len(below_0_5pct)/total*100:.2f}%と少ないため）")
    else:
        print("  現状のcondition4（3%未満）を維持することを推奨します")
        print("  （バランスが取れています）")

    print("=" * 60)


if __name__ == "__main__":
    main()

