"""プログラムコード文字列からコマンドを抽出するユーティリティ"""
import re
from typing import Set


def extract_commands_from_program_code(program_code: str) -> Set[str]:
    """プログラムコード文字列から使用されているコマンドを抽出

    Args:
        program_code: プログラムコード文字列

    Returns:
        使用されているコマンドのセット
    """
    commands = set()

    # 一般的なコマンドパターンを抽出
    # 例: GET_ALL_OBJECTS(4), FILTER(objects, ...), RENDER_GRID(...)
    command_pattern = r'\b([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]*)?)\s*\('

    matches = re.findall(command_pattern, program_code)
    commands.update(matches)

    # FOR文のパターン
    if re.search(r'\bFOR\s+', program_code, re.IGNORECASE):
        commands.add('FOR i LEN')

    # IF文のパターン
    if re.search(r'\bIF\s+', program_code, re.IGNORECASE):
        commands.add('IF')

    # 条件演算子のパターン
    condition_operators = ['AND', 'OR', 'NOT', 'EQUAL', 'NOT_EQUAL', 'IS_INSIDE', 'IS_SAME_SHAPE']
    for op in condition_operators:
        if op in program_code:
            commands.add(op)

    return commands

