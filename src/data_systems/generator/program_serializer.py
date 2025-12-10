"""
プログラムの複数形式保存用モジュール

学習用途を想定して、プログラムを複数の形式で保存する
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


def save_program_for_training(
    program_code: str,
    nodes: List[Any],
    complexity: int,
    grid_width: int,
    grid_height: int,
    task_index: int,
    output_dir: str,
    timestamp: str,
    metadata: Optional[Dict[str, Any]] = None,
    buffer_manager: Optional[Any] = None
):
    """プログラムを学習用途に適した複数の形式で保存

    Args:
        program_code: プログラムコード（文字列）
        nodes: プログラムのNodeリスト
        complexity: 複雑度
        grid_width: グリッド幅
        grid_height: グリッド高さ
        task_index: タスクインデックス
        output_dir: 出力ディレクトリ
        timestamp: タイムスタンプ
        metadata: 追加のメタデータ
    """
    task_dir = os.path.join(output_dir, f"task_{task_index:03d}")
    os.makedirs(task_dir, exist_ok=True)

    # 1. テキスト形式（既存の形式）
    txt_filename = os.path.join(task_dir, f"program_complexity{complexity}_{timestamp}.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(program_code)

    # 2. JSON形式（メタデータを含む）
    json_data = {
        "task_id": f"task_{task_index:03d}",
        "timestamp": timestamp,
        "complexity": complexity,
        "grid_size": {
            "width": grid_width,
            "height": grid_height
        },
        "program_code": program_code,
        "program_length": len(program_code),
        "node_count": len(nodes),
        "statistics": {
            "line_count": program_code.count('\n') + 1,
            "character_count": len(program_code),
            "word_count": len(program_code.split())
        }
    }

    # 追加のメタデータがあればマージ
    if metadata:
        json_data["metadata"] = metadata

    json_filename = os.path.join(task_dir, f"program_complexity{complexity}_{timestamp}.json")
    # バッファマネージャーが提供されている場合はバッファに追加、そうでなければ直接書き込み
    if buffer_manager:
        buffer_manager.add_program_json(task_index, json_data)
    else:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    # tokens_jsonとstats_jsonも同じタスクインデックスで追加（バッファマネージャーを使用する場合）
    # これにより、同じタスクのデータが同じバッチに含まれる

    # 3. プログラムをトークン単位で分割した形式（学習用）
    tokens = tokenize_program(program_code)
    tokens_filename = os.path.join(task_dir, f"program_complexity{complexity}_{timestamp}_tokens.json")
    tokens_data = {
        "task_id": f"task_{task_index:03d}",
        "timestamp": timestamp,
        "complexity": complexity,
        "grid_size": {
            "width": grid_width,
            "height": grid_height
        },
        "tokens": tokens,
        "token_count": len(tokens),
        "vocabulary": list(set(tokens))
    }
    # バッファマネージャーが提供されている場合はバッファに追加、そうでなければ直接書き込み
    if buffer_manager:
        buffer_manager.add_tokens_json(task_index, tokens_data)
    else:
        with open(tokens_filename, "w", encoding="utf-8") as f:
            json.dump(tokens_data, f, ensure_ascii=False, indent=2)

    # 4. プログラム統計情報
    stats = analyze_program_statistics(program_code, nodes)
    stats_filename = os.path.join(task_dir, f"program_complexity{complexity}_{timestamp}_stats.json")
    stats_data = {
        "task_id": f"task_{task_index:03d}",
        "timestamp": timestamp,
        "complexity": complexity,
        "grid_size": {
            "width": grid_width,
            "height": grid_height
        },
        "statistics": stats
    }
    # バッファマネージャーが提供されている場合はバッファに追加、そうでなければ直接書き込み
    if buffer_manager:
        buffer_manager.add_stats_json(task_index, stats_data)
    else:
        with open(stats_filename, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

    return {
        "txt": txt_filename,
        "json": json_filename,
        "tokens": tokens_filename,
        "stats": stats_filename
    }


def tokenize_program(program_code: str) -> List[str]:
    """プログラムをトークン単位に分割

    Args:
        program_code: プログラムコード

    Returns:
        トークンのリスト
    """
    tokens = []
    current_token = ""
    in_string = False
    string_char = None

    i = 0
    while i < len(program_code):
        char = program_code[i]

        # 文字列リテラルの処理
        if char in ('"', "'") and (i == 0 or program_code[i-1] != '\\'):
            if in_string and char == string_char:
                # 文字列終了
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                in_string = False
                string_char = None
            elif not in_string:
                # 文字列開始
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                in_string = True
                string_char = char
                current_token = char
            else:
                current_token += char
        elif in_string:
            # 文字列内
            current_token += char
        elif char.isspace():
            # 空白文字
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif char in ('(', ')', '[', ']', ',', '=', ';'):
            # 区切り文字
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            # 通常の文字
            current_token += char

        i += 1

    # 最後のトークン
    if current_token:
        tokens.append(current_token)

    # 空のトークンを削除
    tokens = [t for t in tokens if t.strip()]

    return tokens


def analyze_program_statistics(program_code: str, nodes: List[Any]) -> Dict[str, Any]:
    """プログラムの統計情報を分析

    Args:
        program_code: プログラムコード
        nodes: プログラムのNodeリスト

    Returns:
        統計情報の辞書
    """
    stats = {
        "program_length": len(program_code),
        "line_count": program_code.count('\n') + 1,
        "character_count": len(program_code),
        "node_count": len(nodes),
        "command_count": 0,
        "variable_count": 0,
        "loop_count": 0,
        "condition_count": 0,
        "commands": {},
        "variables": set(),
    }

    # コマンドを抽出
    lines = program_code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # コマンド名を抽出（=の左側または直接コマンド）
        if '=' in line:
            right_side = line.split('=')[1].strip()
            # コマンド名を抽出（最初の'('の前）
            if '(' in right_side:
                cmd_name = right_side.split('(')[0].strip()
                if cmd_name:
                    stats["commands"][cmd_name] = stats["commands"].get(cmd_name, 0) + 1
                    stats["command_count"] += 1
        elif '(' in line:
            cmd_name = line.split('(')[0].strip()
            if cmd_name:
                stats["commands"][cmd_name] = stats["commands"].get(cmd_name, 0) + 1
                stats["command_count"] += 1

        # FORループを検出
        if line.startswith('FOR'):
            stats["loop_count"] += 1

        # IF文を検出
        if line.startswith('IF'):
            stats["condition_count"] += 1

        # 変数を抽出（より詳細なパターンマッチ）
        # =の左側を変数として抽出
        if '=' in line:
            left_side = line.split('=')[0].strip()
            # 配列要素（objects[i]）や複雑な式を除外
            if '[' not in left_side and '(' not in left_side:
                var_name = left_side.split()[0] if left_side.split() else left_side
                if var_name and var_name[0].isalpha():
                    # 予約語を除外
                    reserved = ['FOR', 'IF', 'END', 'DO', 'GET', 'SET', 'CREATE', 'FILTER',
                               'EXCLUDE', 'CONCAT', 'APPEND', 'MERGE', 'NOT', 'EQUAL', 'AND', 'OR']
                    if var_name not in reserved:
                        if '$' not in var_name:  # プレースホルダーを除外
                            stats["variables"].add(var_name)

    stats["variable_count"] = len(stats["variables"])
    stats["variables"] = sorted(list(stats["variables"]))
    stats["commands"] = dict(sorted(stats["commands"].items(), key=lambda x: x[1], reverse=True))

    # Nodeタイプの統計
    node_types = {}
    for node in nodes:
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    stats["node_types"] = dict(sorted(node_types.items(), key=lambda x: x[1], reverse=True))

    return stats
