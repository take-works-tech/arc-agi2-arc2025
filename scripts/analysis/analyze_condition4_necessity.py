"""condition4（オブジェクトピクセル数3%未満）の必要性を分析するスクリプト"""

import json
from pathlib import Path


def main():
    # ARC-AGI2統計データを読み込み
    stats_file = Path("outputs/input_grid_comparison/arc_agi2_statistics.json")
    if not stats_file.exists():
        print(f"エラー: {stats_file} が見つかりません")
        return

    with open(stats_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # オブジェクトピクセル比率を抽出
    ratios = []
    zero_ratios = []
    below_3pct = []
    single_color = []

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

        # 単色チェック（ユニーク色が1つ）
        if entry.get('num_colors_total', 0) == 1:
            single_color.append(entry)

    # 統計を出力
    total = len(ratios)
    print("=" * 60)
    print("condition4（オブジェクトピクセル数3%未満）の必要性分析")
    print("=" * 60)
    print(f"\n総タスク数: {total}")
    print(f"\n【0%のタスク】")
    print(f"  数: {len(zero_ratios)} ({len(zero_ratios)/total*100:.2f}%)")
    print(f"  これはcondition3（単色）またはcondition5（空）でカバーされる")

    print(f"\n【3%未満（0より大きい）のタスク】")
    print(f"  数: {len(below_3pct)} ({len(below_3pct)/total*100:.2f}%)")
    print(f"  これはcondition4でのみフィルタリングされる")

    if below_3pct:
        print(f"\n  サンプル（最小5件）:")
        sorted_below = sorted(below_3pct, key=lambda x: x['object_pixel_ratio'])
        for i, entry in enumerate(sorted_below[:5], 1):
            ratio = entry['object_pixel_ratio']
            task_id = entry.get('task_id', 'N/A')
            area = entry.get('area', 0)
            obj_pixels = entry.get('object_pixel_count', 0)
            print(f"    {i}. タスクID: {task_id}, 比率: {ratio:.6f} ({ratio*100:.4f}%), "
                  f"面積: {area}, オブジェクトピクセル数: {obj_pixels}")

    print(f"\n【単色のタスク（condition3でカバー）】")
    print(f"  数: {len(single_color)} ({len(single_color)/total*100:.2f}%)")

    # 重複チェック: condition3でカバーされていて、かつ3%未満のタスク
    overlap = [e for e in below_3pct if e.get('num_colors_total', 0) == 1]
    if overlap:
        print(f"\n【注意】condition3（単色）でカバーされているのに、3%未満でもあるタスク: {len(overlap)}件")
        print("  これは重複です（condition3で既にフィルタリングされる）")

    # condition4だけがカバーするケース
    condition4_only = [e for e in below_3pct if e.get('num_colors_total', 1) > 1]
    print(f"\n【condition4のみがカバーするケース】")
    print(f"  数: {len(condition4_only)} ({len(condition4_only)/total*100:.2f}%)")
    print(f"  条件: 複数色あり、かつ0% < オブジェクトピクセル比率 < 3%")

    if condition4_only:
        print(f"\n  サンプル（最小5件）:")
        sorted_condition4_only = sorted(condition4_only, key=lambda x: x['object_pixel_ratio'])
        for i, entry in enumerate(sorted_condition4_only[:5], 1):
            ratio = entry['object_pixel_ratio']
            task_id = entry.get('task_id', 'N/A')
            area = entry.get('area', 0)
            obj_pixels = entry.get('object_pixel_count', 0)
            num_colors = entry.get('num_colors_total', 0)
            print(f"    {i}. タスクID: {task_id}, 比率: {ratio:.6f} ({ratio*100:.4f}%), "
                  f"面積: {area}, オブジェクトピクセル数: {obj_pixels}, 色数: {num_colors}")

    print("\n" + "=" * 60)
    print("【結論】")
    if len(condition4_only) > 0:
        print(f"condition4は必要です。")
        print(f"  - {len(condition4_only)}件（{len(condition4_only)/total*100:.2f}%）のタスクが")
        print(f"    condition4でのみフィルタリングされます。")
        print(f"  - これらは非常に少ないオブジェクト（1-2ピクセル程度）を持つ複数色のグリッドです。")
    else:
        print(f"condition4は不要かもしれません。")
        print(f"  - condition3とcondition5で十分カバーできています。")
    print("=" * 60)


if __name__ == "__main__":
    main()

