"""
未使用変数を削除するモジュール

左辺で代入されているが、それより下の行で一度も使用されていない変数の代入式を削除する
"""
import re
from typing import List, Tuple, Set


def remove_unused_variables(program_code: str) -> str:
    """
    未使用変数を削除

    未使用変数：左辺で代入されているが、それより下の行で一度も使用されていない変数

    Args:
        program_code: プログラムコード文字列

    Returns:
        未使用変数を削除したプログラムコード
    """
    if not program_code:
        return program_code

    # 未使用変数がなくなるまでループ
    max_iterations = 100  # 無限ループを防ぐ
    current_code = program_code

    for iteration in range(max_iterations):
        lines = current_code.split('\n')
        if not lines:
            break

        # 各行を解析して、削除すべき行のインデックスを収集
        lines_to_remove = set()

        for i, line in enumerate(lines):
            # 代入式（=）がある行をチェック
            # 注：実際のプログラムでは最後の行はRENDER_GRID（=がない）なので、
            # この時点で処理対象外になる
            if '=' not in line:
                continue

            # GET_ALL_OBJECTSが使われている行は削除しない（初期化処理のため重要）
            if 'GET_ALL_OBJECTS' in line:
                continue

            # 左辺の変数名を抽出
            left_var = _extract_left_hand_variable(line)
            if not left_var:
                continue

            # それより下の行でその変数が使用されているかチェック
            is_used = _is_variable_used_below(lines, i, left_var)

            if not is_used:
                # 未使用変数：この行を削除対象に追加
                lines_to_remove.add(i)

        # 削除すべき行がなければ終了
        if not lines_to_remove:
            break

        # 行を削除
        new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
        current_code = '\n'.join(new_lines)

    return current_code


def _extract_left_hand_variable(line: str) -> str:
    """
    代入式の左辺の変数名を抽出

    Args:
        line: 行の文字列（例：'coordinate5 = GET_POSITION(objects[0])'）

    Returns:
        左辺の変数名（例：'coordinate5'）。抽出できない場合はNone
    """
    # コメントを除去
    line = line.split('#')[0].strip()
    if not line:
        return None

    # 代入式（=）で分割
    parts = line.split('=', 1)
    if len(parts) != 2:
        return None

    left_side = parts[0].strip()
    if not left_side:
        return None

    # 左辺から変数名を抽出
    # 例: "coordinate5", "objects[i]", "result" など
    # 配列アクセス（[i]）がある場合は、変数名部分のみを抽出
    match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(\[.*\])?\s*$', left_side)
    if match:
        var_name = match.group(1)
        return var_name

    return None


def _is_variable_used_below(lines: List[str], start_index: int, var_name: str) -> bool:
    """
    指定された行より下で、変数が使用されているかチェック

    Args:
        lines: 行のリスト
        start_index: 開始行インデックス
        var_name: 変数名

    Returns:
        変数が使用されている場合はTrue
    """
    # 変数名のパターン（単語境界を使用）
    pattern = r'\b' + re.escape(var_name) + r'\b'

    for i in range(start_index + 1, len(lines)):
        line = lines[i]

        # コメントを除去
        line = line.split('#')[0]

        # 変数名が含まれているかチェック
        if re.search(pattern, line, re.IGNORECASE):
            # 左辺の代入先としてのみ使用されている場合は、使用とはみなさない
            # 例: "coordinate5 = ..." のような行では、左辺のcoordinate5は使用とはみなさない

            # 代入式の場合、左辺を除外してチェック
            if '=' in line:
                # 左辺と右辺に分割
                parts = line.split('=', 1)
                right_side = parts[1] if len(parts) > 1 else ''

                # 右辺に変数が含まれているかチェック
                if re.search(pattern, right_side, re.IGNORECASE):
                    return True
            else:
                # 代入式でない場合、そのままチェック
                return True

    return False
