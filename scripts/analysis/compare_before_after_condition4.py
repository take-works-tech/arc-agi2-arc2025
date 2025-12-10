"""condition4削除前後の比較分析スクリプト"""

import json
from pathlib import Path


def load_statistics(file_path):
    """統計ファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_object_pixel_ratios(data):
    """オブジェクトピクセル比率を分析"""
    ratios = []
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

        if 0 < ratio < 0.03:
            below_3pct.append(entry)
        if 0 < ratio < 0.01:
            below_1pct.append(entry)
        if 0 < ratio < 0.005:
            below_0_5pct.append(entry)

    return ratios, below_3pct, below_1pct, below_0_5pct


def main():
    print("=" * 70)
    print("condition4削除前後の比較分析")
    print("=" * 70)

    # ファイルパス
    before_file = Path("outputs/input_grid_comparison/generated_statistics.json")
    after_file = Path("outputs/input_grid_comparison_after_condition4_removal/generated_statistics.json")

    # データを読み込み
    print("\n【データ読み込み】")
    try:
        before_data = load_statistics(before_file)
        print(f"  削除前: {len(before_data)}タスク")
    except FileNotFoundError:
        print(f"  削除前: ファイルが見つかりません ({before_file})")
        before_data = []

    try:
        after_data = load_statistics(after_file)
        print(f"  削除後: {len(after_data)}タスク")
    except FileNotFoundError:
        print(f"  削除後: ファイルが見つかりません ({after_file})")
        after_data = []

    # 分析
    if before_data:
        before_ratios, before_3pct, before_1pct, before_0_5pct = analyze_object_pixel_ratios(before_data)
        before_total = len(before_ratios)
        before_non_zero = [r for r in before_ratios if r > 0]
    else:
        before_ratios, before_3pct, before_1pct, before_0_5pct = [], [], [], []
        before_total = 0
        before_non_zero = []

    if after_data:
        after_ratios, after_3pct, after_1pct, after_0_5pct = analyze_object_pixel_ratios(after_data)
        after_total = len(after_ratios)
        after_non_zero = [r for r in after_ratios if r > 0]
    else:
        after_ratios, after_3pct, after_1pct, after_0_5pct = [], [], [], []
        after_total = 0
        after_non_zero = []

    # 結果を表示
    print("\n" + "=" * 70)
    print("【オブジェクトピクセル比率の比較】")
    print("=" * 70)

    if before_total > 0:
        print(f"\n【削除前】")
        print(f"  総タスク数: {before_total}")
        print(f"  3%未満（0より大きい）: {len(before_3pct)} ({len(before_3pct)/before_total*100:.2f}%)")
        print(f"  1%未満（0より大きい）: {len(before_1pct)} ({len(before_1pct)/before_total*100:.2f}%)")
        print(f"  0.5%未満（0より大きい）: {len(before_0_5pct)} ({len(before_0_5pct)/before_total*100:.2f}%)")
        if before_non_zero:
            print(f"  最小値（0以外）: {min(before_non_zero):.6f} ({min(before_non_zero)*100:.4f}%)")
            print(f"  平均値: {sum(before_ratios)/len(before_ratios):.6f} ({sum(before_ratios)/len(before_ratios)*100:.2f}%)")
            print(f"  中央値（0以外）: {sorted(before_non_zero)[len(before_non_zero)//2]:.6f} ({sorted(before_non_zero)[len(before_non_zero)//2]*100:.4f}%)")

    if after_total > 0:
        print(f"\n【削除後】")
        print(f"  総タスク数: {after_total}")
        print(f"  3%未満（0より大きい）: {len(after_3pct)} ({len(after_3pct)/after_total*100:.2f}%)")
        print(f"  1%未満（0より大きい）: {len(after_1pct)} ({len(after_1pct)/after_total*100:.2f}%)")
        print(f"  0.5%未満（0より大きい）: {len(after_0_5pct)} ({len(after_0_5pct)/after_total*100:.2f}%)")
        if after_non_zero:
            print(f"  最小値（0以外）: {min(after_non_zero):.6f} ({min(after_non_zero)*100:.4f}%)")
            print(f"  平均値: {sum(after_ratios)/len(after_ratios):.6f} ({sum(after_ratios)/len(after_ratios)*100:.2f}%)")
            print(f"  中央値（0以外）: {sorted(after_non_zero)[len(after_non_zero)//2]:.6f} ({sorted(after_non_zero)[len(after_non_zero)//2]*100:.4f}%)")

    # 変化を表示
    if before_total > 0 and after_total > 0:
        print(f"\n【変化】")
        print(f"  総タスク数: {after_total - before_total:+d} ({((after_total - before_total)/before_total*100):+.2f}%)")
        print(f"  3%未満のタスク: {len(after_3pct) - len(before_3pct):+d} ({((len(after_3pct) - len(before_3pct))/before_total*100 if before_total > 0 else 0):+.2f}ポイント)")
        print(f"  1%未満のタスク: {len(after_1pct) - len(before_1pct):+d} ({((len(after_1pct) - len(before_1pct))/before_total*100 if before_total > 0 else 0):+.2f}ポイント)")
        if before_non_zero and after_non_zero:
            min_change = min(after_non_zero) - min(before_non_zero)
            print(f"  最小値の変化: {min_change:+.6f} ({min_change*100:+.4f}ポイント)")
            mean_change = (sum(after_ratios)/len(after_ratios)) - (sum(before_ratios)/len(before_ratios))
            print(f"  平均値の変化: {mean_change:+.6f} ({mean_change*100:+.2f}ポイント)")

    print("\n" + "=" * 70)
    print("【結論】")
    if before_total > 0 and after_total > 0:
        if len(after_3pct) > len(before_3pct):
            print("  condition4削除により、3%未満のオブジェクトピクセル比率のタスクが生成されるようになりました。")
            print(f"  これはARC-AGI2データセットに存在する有効なタスクタイプです。")
        else:
            print("  condition4削除による影響は限定的でした。")
    print("=" * 70)


if __name__ == "__main__":
    main()

