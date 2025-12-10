"""
色分布統計

ARC-AGI2統計に基づく色数の分布
"""
import random
from typing import Set, List, Optional


# ARC-AGI2統計に基づく背景色以外の色数の分布
# 一般的なARCタスクでは、背景色以外の色数は1-3色が多い
# 複雑なタスクでは4-5色まで使用される場合もある
#
# 注意: この確率は「1つのグリッド」に対して1回だけ適用される
# - 既存のオブジェクトが空の場合のみ統計から色数を決定
# - 既存のオブジェクトがある場合は既存の色を使用（統計に影響しない）
# - これにより、プログラムの色による分岐などにも柔軟に対応可能
OBJECT_COLOR_COUNT_DISTRIBUTION = {
    1: 0.20,  # 背景色以外1色: 20%（30%→20%に減少、ARC-AGI2実データ: 26.1%に近づける）
    2: 0.40,  # 背景色以外2色: 40%（38%→40%に増加）
    3: 0.30,  # 背景色以外3色: 30%（25%→30%に増加）
    4: 0.07,  # 背景色以外4色: 7%（5%→7%に増加）
    5: 0.03,  # 背景色以外5色: 3%（2%→3%に増加）
    # 合計: 100%
    # 1色: 20%（ARC-AGI2実データ: 26.1%に近づける）
    # 2-3色: 70%（ARC-AGI2実データの分布に合わせる）
    # 4-5色: 10%（ARC-AGI2実データの分布に合わせる）
}

# ARC-AGI2統計に基づくオブジェクトの重複確率
# 一般的なARCタスクでは、同じ形状や同じ色と形状のオブジェクトが出現する確率
# ARC-AGI2データセットの実際の分析結果（1000タスク、82,705オブジェクト）に基づく
SAME_COLOR_AND_SHAPE_PROBABILITY = 0.612248  # 同じ色と形状: 61.22%
SAME_SHAPE_PROBABILITY = 0.097140  # 同じ形状（色は異なる）: 9.71%
# 残り29.06%は新規生成（異なる形状と色）


def decide_object_color_count(existing_colors: Set[int], background_color: int, target_count: Optional[int] = None) -> int:
    """統計に基づいて背景色以外の色数を決定

    この関数は、既存のオブジェクトがない場合にのみ統計から色数を決定する。
    既存のオブジェクトがある場合は、既存の色数に合わせる（統計に影響しない）。
    これにより、プログラムの色による分岐などにも柔軟に対応可能。

    Args:
        existing_colors: 既存のオブジェクトの色のセット
        background_color: 背景色
        target_count: 目標色数（Noneの場合は統計から決定）

    Returns:
        背景色以外の色数
    """
    # 既存の色（背景色を除く）をカウント
    existing_count = len([c for c in existing_colors if c != background_color])

    if target_count is not None:
        # 目標色数が指定されている場合
        return max(1, min(5, target_count))

    # 統計に基づいて色数を決定（既存のオブジェクトがない場合のみ適用）
    counts = list(OBJECT_COLOR_COUNT_DISTRIBUTION.keys())
    weights = list(OBJECT_COLOR_COUNT_DISTRIBUTION.values())

    selected_count = random.choices(counts, weights=weights, k=1)[0]

    # 既存の色数より少ない場合は既存の色数を返す
    # これにより、後続のオブジェクト生成でも既存の色が維持される
    return max(existing_count, selected_count)


def select_object_colors(
    background_color: int,
    target_color_count: int,
    existing_colors: Set[int],
    exclude_colors: Set[int] = None
) -> List[int]:
    """背景色以外のオブジェクト色を選択

    Args:
        background_color: 背景色
        target_color_count: 目標色数（背景色以外）
        existing_colors: 既存の色のセット
        exclude_colors: 除外する色のセット

    Returns:
        選択された色のリスト（背景色以外）
    """
    if exclude_colors is None:
        exclude_colors = set()

    # 除外する色に背景色を追加
    exclude_colors = exclude_colors | {background_color}

    # 利用可能な色を決定（0-9から背景色と除外色を除く）
    available_colors = [c for c in range(10) if c not in exclude_colors]

    if not available_colors:
        # 利用可能な色がない場合は、背景色以外の色をランダムに選択
        available_colors = [c for c in range(10) if c != background_color]

    # 既存の色を優先的に使用
    existing_object_colors = [c for c in existing_colors if c != background_color and c in available_colors]

    selected_colors = list(set(existing_object_colors))  # 既存の色から重複を除去

    # 必要な色数に達するまで追加
    while len(selected_colors) < target_color_count and available_colors:
        # まだ選択されていない色からランダムに選択
        remaining_colors = [c for c in available_colors if c not in selected_colors]
        if remaining_colors:
            selected_colors.append(random.choice(remaining_colors))
        else:
            break

    # 目標色数に満たない場合は、既存の色を再利用
    if len(selected_colors) < target_color_count:
        # 既存の色を追加して目標色数に達するまで繰り返す
        while len(selected_colors) < target_color_count:
            if existing_object_colors:
                selected_colors.append(random.choice(existing_object_colors))
            else:
                # 既存の色がない場合は利用可能な色からランダムに選択
                if available_colors:
                    selected_colors.append(random.choice(available_colors))
                else:
                    break

    return selected_colors[:target_color_count]
