"""
コマンド一覧＋複雑さ重み設定（静的テーブル版）

可読性を最優先し、
  - 各DSLコマンドに対して
      * 生成カテゴリ（参考情報）
      * 複雑さカテゴリ（very_light / light / medium / heavy / very_heavy）
      * コマンド固有の補正係数
      * 最終的な複雑さ重み
  を一元管理する。
"""

from typing import Dict, List, Tuple


# ============================================================
# 「重みづけ専用」の複雑さカテゴリ
# ============================================================

# 複雑さカテゴリごとの基礎重み
# 値が大きいほど「1回の呼び出しが複雑」とみなす
# 情報取得・演算系をかなり軽くし、オブジェクト操作・生成系を強く罰する設定
COMPLEXITY_CATEGORY_WEIGHTS: Dict[str, float] = {
    "very_light": 0.3,   # ほとんどペナルティ無し（単純情報取得・論理演算など）
    "light": 0.6,        # 軽い演算・比較
    "medium": 1.0,       # 通常レベル（配列操作の一部など）
    "heavy": 1.8,        # オブジェクト操作（transform 系）
    "very_heavy": 2.4,   # 構造変換・パターン操作・オブジェクト生成
}


# ============================================================
# コマンドごとの複雑さ設定（静的テーブル）
# ============================================================

# 各要素:
#   "complexity_category": 複雑さカテゴリ
#   "command_multiplier": コマンド固有補正
COMMAND_COMPLEXITY_TABLE: Dict[str, Dict[str, object]] = {
    # getter 系（情報取得・カウント） → very_light に寄せてほぼ無視
    "GET_X": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_Y": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_WIDTH": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_HEIGHT": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_SIZE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_COLOR": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_COLORS": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_BACKGROUND_COLOR": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_DISTANCE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_X_DISTANCE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_Y_DISTANCE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_SYMMETRY_SCORE": {"complexity_category": "light", "command_multiplier": 1.0},
    "GET_LINE_TYPE": {"complexity_category": "light", "command_multiplier": 1.0},
    "GET_RECTANGLE_TYPE": {"complexity_category": "light", "command_multiplier": 1.0},
    "GET_INPUT_GRID_SIZE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_ASPECT_RATIO": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_DENSITY": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_CENTROID": {"complexity_category": "light", "command_multiplier": 1.0},
    "GET_CENTER_X": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_CENTER_Y": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_MAX_X": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_MAX_Y": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GET_DIRECTION": {"complexity_category": "light", "command_multiplier": 1.0},
    "GET_NEAREST": {"complexity_category": "medium", "command_multiplier": 1.0},
    "COUNT_HOLES": {"complexity_category": "light", "command_multiplier": 1.0},
    "COUNT_ADJACENT": {"complexity_category": "light", "command_multiplier": 1.0},
    "COUNT_OVERLAP": {"complexity_category": "light", "command_multiplier": 1.0},
    "LEN": {"complexity_category": "very_light", "command_multiplier": 1.0},

    # comparison 系 → 情報判定として light
    "IS_SAME_SHAPE": {"complexity_category": "light", "command_multiplier": 1.0},
    "IS_IDENTICAL": {"complexity_category": "light", "command_multiplier": 1.0},
    "IS_INSIDE": {"complexity_category": "light", "command_multiplier": 1.0},
    "IS_SAME_STRUCT": {"complexity_category": "light", "command_multiplier": 1.0},

    # arithmetic / proportional 系 → very_light（人間にとってはほぼタダ）
    "ADD": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "SUB": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "MULTIPLY": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "DIVIDE": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "MOD": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "EQUAL": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "NOT_EQUAL": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "GREATER": {"complexity_category": "very_light", "command_multiplier": 1.0},
    "LESS": {"complexity_category": "very_light", "command_multiplier": 1.0},

    # transform 系（オブジェクト操作）
    "MOVE": {"complexity_category": "heavy", "command_multiplier": 0.8},
    "TELEPORT": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "SLIDE": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "PATHFIND": {"complexity_category": "heavy", "command_multiplier": 1.5},
    "ROTATE": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "FLIP": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "SCALE": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "SCALE_DOWN": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "EXPAND": {"complexity_category": "heavy", "command_multiplier": 1.1},
    "FILL_HOLES": {"complexity_category": "heavy", "command_multiplier": 1.1},
    "SET_COLOR": {"complexity_category": "heavy", "command_multiplier": 0.7},
    "OUTLINE": {"complexity_category": "heavy", "command_multiplier": 1.1},
    "HOLLOW": {"complexity_category": "heavy", "command_multiplier": 1.1},
    "BBOX": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "INTERSECTION": {"complexity_category": "heavy", "command_multiplier": 1.2},
    "SUBTRACT": {"complexity_category": "heavy", "command_multiplier": 1.2},
    "FLOW": {"complexity_category": "heavy", "command_multiplier": 1.4},
    "DRAW": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "LAY": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "ALIGN": {"complexity_category": "heavy", "command_multiplier": 0.9},
    "CROP": {"complexity_category": "heavy", "command_multiplier": 1.0},
    "FIT_SHAPE": {"complexity_category": "heavy", "command_multiplier": 1.3},
    "FIT_SHAPE_COLOR": {"complexity_category": "heavy", "command_multiplier": 1.3},
    "FIT_ADJACENT": {"complexity_category": "heavy", "command_multiplier": 1.3},

    # create 系
    "CREATE_LINE": {"complexity_category": "very_heavy", "command_multiplier": 1.0},
    "CREATE_RECT": {"complexity_category": "very_heavy", "command_multiplier": 1.0},
    "MERGE": {"complexity_category": "very_heavy", "command_multiplier": 1.3},
    "TILE": {"complexity_category": "very_heavy", "command_multiplier": 1.2},

    # array / collection 系
    #   - 軽い配列操作: medium
    #   - 強い構造変換: very_heavy / heavy
    "FILTER": {"complexity_category": "medium", "command_multiplier": 0.8},
    "MATCH_PAIRS": {"complexity_category": "very_heavy", "command_multiplier": 1.6},
    "SORT_BY": {"complexity_category": "medium", "command_multiplier": 0.9},
    "EXTEND_PATTERN": {"complexity_category": "very_heavy", "command_multiplier": 1.6},
    "EXCLUDE": {"complexity_category": "heavy", "command_multiplier": 1.3},
    "ARRANGE_GRID": {"complexity_category": "heavy", "command_multiplier": 1.4},
    "CONCAT": {"complexity_category": "heavy", "command_multiplier": 1.2},
    "REVERSE": {"complexity_category": "medium", "command_multiplier": 0.8},
    "APPEND": {"complexity_category": "medium", "command_multiplier": 0.8},
    "EXTRACT_LINES": {"complexity_category": "medium", "command_multiplier": 1.0},
    "EXTRACT_RECTS": {"complexity_category": "medium", "command_multiplier": 1.0},
    "EXTRACT_HOLLOW_RECTS": {"complexity_category": "medium", "command_multiplier": 1.1},
    "GET_ALL_OBJECTS": {"complexity_category": "medium", "command_multiplier": 1.0},
    "SPLIT_CONNECTED": {"complexity_category": "very_heavy", "command_multiplier": 1.6},

    # logical 系
    "AND": {"complexity_category": "light", "command_multiplier": 1.0},
    "OR": {"complexity_category": "light", "command_multiplier": 1.0},

    # その他のグローバル関数
    "RENDER_GRID": {"complexity_category": "very_light", "command_multiplier": 0.5},
}


def build_command_complexity_weights() -> Dict[str, float]:
    """
    コマンドごとの複雑度重みテーブルを構築

    Returns:
        {コマンド名: 複雑さ重み} の辞書
    """
    weights: Dict[str, float] = {}
    for cmd, info in COMMAND_COMPLEXITY_TABLE.items():
        cat = str(info["complexity_category"])
        mul = float(info.get("command_multiplier", 1.0))
        base = COMPLEXITY_CATEGORY_WEIGHTS.get(cat, 1.0)
        weights[cmd] = base * mul
    return weights


def list_commands_with_categories() -> List[Tuple[str, str, float]]:
    """
    可読性のためのヘルパー:
        [(コマンド名, 複雑さカテゴリ, 最終重み)] のリストを返す。
    """
    result: List[Tuple[str, str, float]] = []
    for cmd, info in COMMAND_COMPLEXITY_TABLE.items():
        cat = str(info["complexity_category"])
        mul = float(info.get("command_multiplier", 1.0))
        base = COMPLEXITY_CATEGORY_WEIGHTS.get(cat, 1.0)
        final = base * mul
        result.append((cmd, cat, final))

    result.sort(key=lambda x: x[0])
    return result


def list_commands_full() -> List[Dict[str, float]]:
    """
    全コマンドについて、複雑さ関連の情報をフルで返すヘルパー。

    戻り値の各要素は:
        {
            "command": コマンド名,
            "generator_category": 生成カテゴリ名,
            "complexity_category": 複雑さカテゴリ名,
            "category_weight": 複雑さカテゴリの基礎重み,
            "command_multiplier": コマンド固有の補正係数,
            "final_weight": category_weight * command_multiplier,
        }
    """
    result: List[Dict[str, float]] = []
    for cmd, info in COMMAND_COMPLEXITY_TABLE.items():
        gen_cat = ""  # 生成カテゴリは保持していないので空文字
        comp_cat = str(info["complexity_category"])
        mul = float(info.get("command_multiplier", 1.0))
        base = COMPLEXITY_CATEGORY_WEIGHTS.get(comp_cat, 1.0)
        final = base * mul
        result.append(
            {
                "command": cmd,
                "generator_category": gen_cat,
                "complexity_category": comp_cat,
                "category_weight": base,
                "command_multiplier": mul,
                "final_weight": final,
            }
        )

    result.sort(key=lambda x: x["command"])
    return result
