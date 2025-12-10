#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全コマンドの対応状況を確認"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_system.ir.mapping.command_registry import list_supported_commands

# コマンドクイックリファレンスから抽出した全コマンドリスト（92個 = 80個 + 12個の新規追加）
ALL_COMMANDS = {
    # 変換操作 (24)
    "MOVE", "TELEPORT", "SLIDE", "PATHFIND", "ROTATE", "FLIP", "SCALE", "SCALE_DOWN",
    "EXPAND", "FILL_HOLES", "SET_COLOR", "OUTLINE", "HOLLOW", "BBOX",
    "INTERSECTION", "SUBTRACT", "FLOW", "DRAW", "LAY", "ALIGN", "CROP", "FIT_SHAPE",
    "FIT_SHAPE_COLOR", "FIT_ADJACENT",

    # オブジェクト生成 (4)
    "MERGE", "CREATE_LINE", "CREATE_RECT", "TILE",

    # オブジェクト分割 (1)
    "SPLIT_CONNECTED",

    # オブジェクト抽出 (3)
    "EXTRACT_RECTS", "EXTRACT_HOLLOW_RECTS", "EXTRACT_LINES",

    # 配列操作 (9)
    "APPEND", "LEN", "CONCAT", "FILTER", "SORT_BY", "EXTEND_PATTERN",
    "ARRANGE_GRID", "MATCH_PAIRS", "REVERSE",

    # 配列操作（除外） (1)
    "EXCLUDE",

    # 情報取得 (28)
    "GET_SIZE", "GET_WIDTH", "GET_HEIGHT", "GET_X", "GET_Y", "GET_COLOR",
    "GET_COLORS", "COUNT_HOLES", "GET_SYMMETRY_SCORE", "GET_LINE_TYPE",
    "GET_RECTANGLE_TYPE", "GET_DISTANCE", "GET_X_DISTANCE", "GET_Y_DISTANCE",
    "COUNT_ADJACENT", "COUNT_OVERLAP", "GET_ALL_OBJECTS", "GET_BACKGROUND_COLOR",
    "GET_INPUT_GRID_SIZE", "GET_ASPECT_RATIO", "GET_DENSITY", "GET_CENTROID",
    "GET_CENTER_X", "GET_CENTER_Y", "GET_MAX_X", "GET_MAX_Y", "GET_DIRECTION",
    "GET_NEAREST",

    # 判定関数 (4)
    "IS_INSIDE", "IS_SAME_SHAPE", "IS_SAME_STRUCT", "IS_IDENTICAL",

    # 算術演算 (5)
    "ADD", "SUB", "MULTIPLY", "DIVIDE", "MOD",

    # 比較演算 (4)
    "EQUAL", "NOT_EQUAL", "GREATER", "LESS",

    # 論理演算 (2)
    "AND", "OR",

    # グローバル関数 (1)
    "RENDER_GRID",
}

# 制御構造（別カウント）
CONTROL_STRUCTURES = {"FOR", "WHILE", "IF"}

print("=" * 80)
print("全コマンド対応状況チェック")
print("=" * 80)
print(f"総コマンド数: {len(ALL_COMMANDS)}個")
print(f"制御構造: {len(CONTROL_STRUCTURES)}個")
print(f"合計: {len(ALL_COMMANDS) + len(CONTROL_STRUCTURES)}個")
print()

# CommandRegistryから登録済みコマンドを取得
registered_commands = set(list_supported_commands())

print("=" * 80)
print("登録済みコマンド")
print("=" * 80)
print(f"登録数: {len(registered_commands)}個")
print()

# 対応状況チェック
missing_commands = ALL_COMMANDS - registered_commands
extra_commands = registered_commands - ALL_COMMANDS

print("=" * 80)
print("対応状況")
print("=" * 80)

if not missing_commands:
    print("[OK] すべてのコマンドが登録されています！")
else:
    print(f"[WARNING] 未登録コマンド: {len(missing_commands)}個")
    print()
    for cmd in sorted(missing_commands):
        print(f"  - {cmd}")

print()

if extra_commands:
    print(f"[INFO] 追加コマンド（リファレンスにないもの）: {len(extra_commands)}個")
    print()
    for cmd in sorted(extra_commands):
        print(f"  + {cmd}")
else:
    print("[INFO] 追加コマンドなし")

print()
print("=" * 80)
print("カテゴリ別チェック")
print("=" * 80)

categories = {
    "変換操作": ["MOVE", "TELEPORT", "SLIDE", "ROTATE", "FLIP", "SCALE", "SCALE_DOWN",
                 "EXPAND", "FILL_HOLES", "SET_COLOR", "OUTLINE", "HOLLOW", "BBOX",
                 "INTERSECTION", "SUBTRACT", "FLOW", "DRAW", "LAY", "ALIGN", "CROP", "FIT_SHAPE",
                 "FIT_SHAPE_COLOR", "FIT_ADJACENT"],
    "オブジェクト生成": ["MERGE", "CREATE_LINE", "CREATE_RECT", "TILE"],
    "オブジェクト分割": ["SPLIT_CONNECTED"],
    "オブジェクト抽出": ["EXTRACT_RECTS", "EXTRACT_HOLLOW_RECTS", "EXTRACT_LINES"],
    "配列操作": ["APPEND", "LEN", "CONCAT", "FILTER", "SORT_BY", "EXTEND_PATTERN",
                 "ARRANGE_GRID", "MATCH_PAIRS", "EXCLUDE", "REVERSE"],
    "情報取得": ["GET_SIZE", "GET_WIDTH", "GET_HEIGHT", "GET_X", "GET_Y", "GET_COLOR",
                 "GET_COLORS", "COUNT_HOLES", "GET_SYMMETRY_SCORE", "GET_LINE_TYPE",
                 "GET_RECTANGLE_TYPE", "GET_DISTANCE", "GET_X_DISTANCE", "GET_Y_DISTANCE",
                 "COUNT_ADJACENT", "COUNT_OVERLAP", "GET_ALL_OBJECTS", "GET_BACKGROUND_COLOR",
                 "GET_INPUT_GRID_SIZE", "GET_ASPECT_RATIO", "GET_DENSITY", "GET_CENTROID",
                 "GET_CENTER_X", "GET_CENTER_Y", "GET_MAX_X", "GET_MAX_Y", "GET_DIRECTION",
                 "GET_NEAREST"],
    "判定関数": ["IS_INSIDE", "IS_SAME_SHAPE", "IS_SAME_STRUCT", "IS_IDENTICAL"],
    "算術演算": ["ADD", "SUB", "MULTIPLY", "DIVIDE", "MOD"],
    "比較演算": ["EQUAL", "NOT_EQUAL", "GREATER", "LESS"],
    "論理演算": ["AND", "OR"],
    "グローバル関数": ["RENDER_GRID"],
}

for category, commands in categories.items():
    missing_in_category = [cmd for cmd in commands if cmd not in registered_commands]
    if missing_in_category:
        print(f"[!] {category}: {len(missing_in_category)}/{len(commands)}個が未登録")
        for cmd in missing_in_category:
            print(f"    - {cmd}")
    else:
        print(f"[OK] {category}: {len(commands)}/{len(commands)}個すべて登録済み")

print()
print("=" * 80)
print("結論")
print("=" * 80)

if not missing_commands:
    print("[OK] すべてのコマンドがIRマッピングに登録されています")
    print("[OK] フルデータセット訓練に進んで問題ありません")
else:
    print(f"[WARNING] {len(missing_commands)}個のコマンドが未登録です")
    print("[WARNING] これらのコマンドを含むプログラムはIR変換に失敗します")
    print("[WARNING] 先に未登録コマンドをCommandRegistryに追加してください")
