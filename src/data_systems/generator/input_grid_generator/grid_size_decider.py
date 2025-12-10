"""
グリッドサイズ決定ロジック

ARC-AGI2統計に基づいてグリッドサイズを決定する
プログラム（Nodeリスト）は引数に持たず、統計から決定
"""
import os
from typing import Tuple
import random

# ログ出力制御（パフォーマンス最適化：デフォルトですべてのログを無効化）
ENABLE_DEBUG_OUTPUT = False  # 常にOFF（環境変数で有効化されても無効）
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')


# ARC-AGI2統計に基づくグリッドサイズの分布
# node_generators.pyの実装を参考:
# - サイズ範囲: 1～30
# - 正方形の確率: 26%（ARC-AGI2実データ分析結果: 26.2%が正方形）
# - 小さなサイズ（3-10）が最も多い傾向

# サイズ範囲の分布（ARC-AGI2実データに合わせて調整：平均面積170.8に近づける）
# グリッドサイズを大きくして、オブジェクト密度を下げる
# サイズ1も含める（最小サイズとして重要）
SIZE_DISTRIBUTION = {
    # 非常に小さいサイズ
    (1, 1): 0.01,   # 1x1: 1%（特殊ケース、維持）
    (2, 2): 0.02,   # 2x2: 2%（特殊ケース、維持）
    # 小さいサイズ（3-8）- 確率をさらに下げる（グリッドサイズを大きくするため）
    (3, 5): 0.10,   # 3-5: 10%（15%→10%に減少）
    (6, 8): 0.15,   # 6-8: 15%（20%→15%に減少）
    # 中サイズ（9-15）- 確率をさらに上げる
    (9, 12): 0.38,  # 9-12: 38%（35%→38%に増加）
    (13, 15): 0.20, # 13-15: 20%（18%→20%に増加）
    # 大きいサイズ（16-20）- 確率を上げる
    (16, 20): 0.10, # 16-20: 10%（7%→10%に増加）
    # 大きなサイズ（21-30）- 確率を上げる
    (21, 30): 0.04, # 21-30: 4%（2%→4%に増加）
}

# 正方形の確率（ARC-AGI2統計に基づく）
# 実データ分析結果: ARC-AGI2実データでは26.2%が正方形
SQUARE_PROBABILITY = 0.26  # 26%の確率で正方形（ARC-AGI2実データに基づく）

# 長方形の場合のサイズ差の分布
# 幅と高さの差のバリエーションを増やし、より多様な形状を生成
RECTANGLE_SIZE_DIFF_PROBABILITIES = {
    1: 0.25,  # 差が1（例: 5x6, 6x5）- 40% → 25%に減少
    2: 0.20,  # 差が2（例: 5x7, 7x5）- 30% → 20%に減少
    3: 0.15,  # 差が3（例: 5x8, 8x5）- 維持
    4: 0.12,  # 差が4（例: 5x9, 9x5）- 8% → 12%に増加
    5: 0.10,  # 差が5（例: 5x10, 10x5）- 4% → 10%に増加
    6: 0.08,  # 差が6（例: 5x11, 11x5）- 2% → 8%に増加
    7: 0.05,  # 差が7（例: 5x12, 12x5）- 1% → 5%に増加
    8: 0.03,  # 差が8（例: 5x13, 13x5）- 新規追加
    9: 0.02,  # 差が9以上（例: 5x14, 14x5など）- 新規追加
}


def decide_grid_size() -> Tuple[int, int]:
    """ARC-AGI2統計に基づいてグリッドサイズを決定

    Returns:
        (width, height)のタプル
    """
    # サイズ範囲を選択
    size_range = _select_size_range()
    min_size, max_size = size_range

    # 正方形か長方形かを決定
    is_square = random.random() < SQUARE_PROBABILITY
    if is_square:
        # 正方形（26%の確率）
        size = random.randint(min_size, max_size)
        return (size, size)
    else:
        # 長方形（74%の確率）
        result = _generate_rectangle_size(min_size, max_size)
        return result


def _select_size_range() -> Tuple[int, int]:
    """サイズ範囲を選択"""
    ranges = list(SIZE_DISTRIBUTION.keys())
    weights = list(SIZE_DISTRIBUTION.values())
    result = random.choices(ranges, weights=weights, k=1)[0]
    return result


def _generate_rectangle_size(min_size: int, max_size: int) -> Tuple[int, int]:
    """長方形のサイズを生成"""
    # min_size == max_size の場合は長方形を生成できないため、正方形を返す
    if min_size == max_size:
        if ENABLE_ALL_LOGS:
            print(f"[WARNING] _generate_rectangle_size: min_size==max_size ({min_size}) のため、正方形を返します", flush=True)
        return (min_size, max_size)

    # バリエーション: 75%の確率で完全にランダムなサイズを生成（より多様性を確保）
    use_random = random.random() < 0.75
    if use_random:
        # 幅と高さを独立にランダムに決定（より広い範囲で多様性を確保）
        # 範囲を拡張してより多様な形状を生成（min_size-2からmax_size+2まで、ただし1-30に制限）
        expanded_min = max(1, min_size - 2)
        expanded_max = min(30, max_size + 2)

        width = random.randint(expanded_min, expanded_max)
        height = random.randint(expanded_min, expanded_max)

        # 同じサイズになった場合は再生成（長方形を保証）
        loop_count = 0
        while width == height:
            loop_count += 1
            if loop_count > 100:
                if ENABLE_ALL_LOGS:
                    print(f"[WARNING] _generate_rectangle_size: 再生成ループが100回を超えました。強制的に差を付けて終了します。", flush=True)
                # 強制的に差を付ける（min_size < max_size は保証されている）
                width = expanded_min
                height = expanded_max
                break
            width = random.randint(expanded_min, expanded_max)
            height = random.randint(expanded_min, expanded_max)
    else:
        # 既存のロジック（基準サイズベース）- より大きな差も許容
        base_size = random.randint(min_size, max_size)

        # サイズ差を選択（より大きな差のバリエーションを含める）
        diff_options = list(RECTANGLE_SIZE_DIFF_PROBABILITIES.keys())
        weights = list(RECTANGLE_SIZE_DIFF_PROBABILITIES.values())
        size_diff = random.choices(diff_options, weights=weights, k=1)[0]

        # 実際の差（1～size_diffの範囲でランダム、より大きい差も許容）
        # 上限が0以下になるとrandint(1, 0)のような不正範囲になってしまうため、安全にクランプ
        max_diff = min(size_diff, max_size - min_size, max(1, 28 - base_size))
        if max_diff <= 0:
            # 理論上ほぼ起こらないが、保険として1を使用
            actual_diff = 1
        else:
            actual_diff = random.randint(1, max_diff)

        # 横長か縦長かを決定（50%ずつ）
        is_wide = random.random() < 0.5
        if is_wide:
            # 横長（width > height）
            # より大きな差を許容しつつ、範囲内に収める
            width = min(30, base_size + actual_diff)
            height = max(1, base_size)
        else:
            # 縦長（height > width）
            width = max(1, base_size)
            height = min(30, base_size + actual_diff)

    # 範囲内に収まるように調整（1-30の範囲に制限）
    width = max(1, min(30, width))
    height = max(1, min(30, height))

    # 最終的に正方形になった場合は、強制的に差を付ける
    if width == height and min_size < max_size:
        # 差を1つ追加
        if width < 30:
            width += 1
        elif height > 1:
            height -= 1
        else:
            # どちらも調整できない場合は、min_sizeとmax_sizeを使用
            width = min_size
            height = max_size

    return (width, height)
