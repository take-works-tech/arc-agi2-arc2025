"""
変数名のインデックスを正規化するモジュール

プログラムコード内の変数名のインデックスが不連続な場合（例：coordinate5があるのにcoordinate1~4がない）
を連続するように正規化する
"""
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from ...program_generator.metadata.variable_naming import VariableNamingSystem, variable_naming_system
from .unused_variable_remover import remove_unused_variables

def normalize_variable_indices(
    program_code: str,
    remove_unused: bool = True
) -> str:
    """
    プログラムコード内の変数名のインデックスを正規化する

    例：
    - coordinate5が使われていて、coordinate1~4がない場合
    - coordinate5をcoordinate1に変更する

    Args:
        program_code: プログラムコード文字列
        remove_unused: 未使用変数を削除するか（デフォルト: True）

    Returns:
        正規化後のプログラムコード
    """

    if not program_code:
        return program_code

    # 変数名を抽出
    variable_names = _extract_variable_names(program_code)

    # ベース名ごとにインデックスをグループ化
    base_to_indices = _group_variables_by_base(variable_names)

    # インデックスの正規化マッピングを作成
    index_mapping = _create_index_mapping(base_to_indices)

    # プログラムコードを置換
    normalized_code = _apply_mapping(program_code, index_mapping)

    # 未使用変数を削除
    if remove_unused:
        normalized_code = remove_unused_variables(normalized_code)

    return normalized_code


def _extract_variable_names(program_code: str) -> Set[str]:
    """
    プログラムコードから変数名を抽出
    variable_naming.pyのベース名パターンに基づいて抽出

    Args:
        program_code: プログラムコード文字列

    Returns:
        抽出された変数名のセット
    """
    variable_names = set()

    # variable_naming.pyからベース名パターンを取得
    base_patterns = variable_naming_system.base_patterns
    all_base_patterns = set()
    for base_names_list in base_patterns.values():
        all_base_patterns.update(base_names_list)
        # 複数形も追加（配列型で使用される）
        for base_name in base_names_list:
            plural_base = base_name + 's'
            if plural_base not in all_base_patterns:
                all_base_patterns.add(plural_base)

    # ベース名パターンを長さの降順でソート（長いパターンを優先）
    # 例: 'x_dist' が 'x' より先にチェックされる
    sorted_patterns = sorted(all_base_patterns, key=len, reverse=True)

    # プレースホルダーを一時的にマスク
    placeholder_pattern = r'\$[a-zA-Z0-9_]+'
    placeholders_map = {}
    placeholder_counter = 0

    def replace_placeholder(match):
        nonlocal placeholder_counter
        placeholder = match.group(0)
        placeholder_id = f"__PLACEHOLDER_{placeholder_counter}__"
        placeholders_map[placeholder_id] = placeholder
        placeholder_counter += 1
        return placeholder_id

    program_without_placeholders = re.sub(placeholder_pattern, replace_placeholder, program_code)

    # コマンド名を一時的にマスク
    command_pattern = r'\b(GET_[A-Z_]+|IS_[A-Z_]+|COUNT_[A-Z_]+|FILTER|MERGE|CONCAT|EXCLUDE|APPEND|SORT_BY|ARRANGE_GRID|MATCH_PAIRS|SPLIT_CONNECTED|CREATE_[A-Z_]+|EXTRACT_[A-Z_]+|FIT_[A-Z_]+|MOVE|TELEPORT|SLIDE|PATHFIND|ROTATE|FLIP|SCALE|EXPAND|FILL_HOLES|SET_COLOR|OUTLINE|HOLLOW|INTERSECTION|SUBTRACT|FLOW|DRAW|LAY|CROP|ALIGN|TILE|REVERSE|FOR|DO|END|IF|THEN|ELSE|LEN|COUNT|RENDER_GRID|True|False)\b'
    commands_map = {}
    command_counter = 0

    def replace_command(match):
        nonlocal command_counter
        command = match.group(0)
        command_id = f"__COMMAND_{command_counter}__"
        commands_map[command_id] = command
        command_counter += 1
        return command_id

    program_without_commands = re.sub(command_pattern, replace_command, program_without_placeholders, flags=re.IGNORECASE)

    # 各ベース名パターンにマッチする変数名を抽出
    found_matches = set()
    for base_pattern in sorted_patterns:
        # ベース名パターン（大文字小文字を区別しない）
        # 1. 完全一致（インデックス0）
        pattern_full = r'\b' + re.escape(base_pattern) + r'\b'
        for match in re.finditer(pattern_full, program_without_commands, re.IGNORECASE):
            var_name = match.group(0).lower()
            found_matches.add((match.start(), match.end(), var_name))

        # 2. ベース名 + 数字（インデックス1以上）
        pattern_with_index = r'\b' + re.escape(base_pattern) + r'\d+\b'
        for match in re.finditer(pattern_with_index, program_without_commands, re.IGNORECASE):
            var_name = match.group(0).lower()
            found_matches.add((match.start(), match.end(), var_name))

    # 重複するマッチを解決（より長いマッチを優先）
    # 位置が重複している場合、より長いマッチを残す
    sorted_matches = sorted(found_matches, key=lambda x: (x[0], -(x[1] - x[0])))
    final_matches = []
    for start, end, var_name in sorted_matches:
        # 既存のマッチと重複していないかチェック
        overlap = False
        for prev_start, prev_end, _ in final_matches:
            if not (end <= prev_start or start >= prev_end):
                # 重複している場合、より長い方を残す
                if (end - start) > (prev_end - prev_start):
                    final_matches.remove((prev_start, prev_end, _))
                    final_matches.append((start, end, var_name))
                overlap = True
                break
        if not overlap:
            final_matches.append((start, end, var_name))

    # 変数名をセットに追加
    for _, _, var_name in final_matches:
        variable_names.add(var_name)

    return variable_names


def _group_variables_by_base(variable_names: Set[str]) -> Dict[str, Set[int]]:
    """
    変数名をベース名ごとにグループ化し、インデックスを収集

    Args:
        variable_names: 変数名のセット

    Returns:
        ベース名 -> インデックスのセットのマッピング
        例: {'coordinate': {0, 5}, 'size': {1, 3, 7}, 'object': {0}, 'objects': {1, 2}}
    """
    base_to_indices: Dict[str, Set[int]] = {}

    # ベース名の候補を取得（variable_naming_systemから）
    base_patterns = variable_naming_system.base_patterns
    all_base_names = set()
    for base_names_list in base_patterns.values():
        all_base_names.update(base_names_list)
        # 複数形も追加（配列型で使用される）
        # 例: object -> objects
        for base_name in base_names_list:
            # 複数形を生成（既に存在しない場合のみ）
            plural_base = base_name + 's'
            if plural_base not in all_base_names:
                # objectsのような複数形もベース名として扱う
                # ただし、元のベース名と区別する
                all_base_names.add(plural_base)

    # 変数名を解析
    for var_name in variable_names:
        base_name, index = _parse_variable_name(var_name, all_base_names)
        if base_name:
            if base_name not in base_to_indices:
                base_to_indices[base_name] = set()
            base_to_indices[base_name].add(index)

    return base_to_indices


def _parse_variable_name(var_name: str, base_names: Set[str]) -> Tuple[str, int]:
    """
    変数名をベース名とインデックスに分解

    Args:
        var_name: 変数名（例：'coordinate5', 'size', 'objects1', 'object2'）
        base_names: ベース名のセット（'object'と'objects'の両方が含まれる可能性がある）

    Returns:
        (ベース名, インデックス) のタプル
        インデックス0は番号なし、1以上は番号付き
        例: ('coordinate', 0), ('coordinate', 5), ('objects', 1), ('object', 2)
    """
    var_name_lower = var_name.lower()

    # ベース名と完全一致する場合（インデックス0）
    if var_name_lower in base_names:
        return var_name_lower, 0

    # ベース名を長さの降順でソート（長いベース名を先にチェック）
    # これにより、'objects'が'object'より先にチェックされ、誤認識を防ぐ
    sorted_base_names = sorted(base_names, key=len, reverse=True)

    # ベース名 + 数字 のパターンをチェック
    for base_name in sorted_base_names:
        # ベース名で始まるかチェック（完全一致は既にチェック済み）
        if var_name_lower.startswith(base_name):
            suffix = var_name_lower[len(base_name):]
            # サフィックスが数字のみの場合
            if suffix.isdigit():
                index = int(suffix)
                if index > 0:  # インデックスは1以上
                    return base_name, index
            # サフィックスが空の場合（完全一致）は既に処理済み
            elif not suffix:
                # このケースは既に処理されているはずだが、念のため
                continue

    # マッチしない場合はNoneを返す
    return None, -1


def _create_index_mapping(base_to_indices: Dict[str, Set[int]]) -> Dict[str, Dict[int, int]]:
    """
    インデックスの正規化マッピングを作成

    不連続なインデックスを連続するように再割り当て
    例: {0, 5, 7} -> {0, 1, 2}

    Args:
        base_to_indices: ベース名 -> インデックスのセット

    Returns:
        ベース名 -> (旧インデックス -> 新インデックス) のマッピング
        例: {'coordinate': {0: 0, 5: 1, 7: 2}}
    """
    mapping: Dict[str, Dict[int, int]] = {}

    for base_name, indices in base_to_indices.items():
        if not indices:
            continue

        # インデックスをソート
        sorted_indices = sorted(indices)

        # インデックス0が含まれているかチェック
        has_index_0 = 0 in sorted_indices

        # 新しいマッピングを作成
        index_mapping: Dict[int, int] = {}
        new_index = 0

        # インデックス0は常に0にマッピング
        if has_index_0:
            index_mapping[0] = 0
            new_index = 1
            # 0以外のインデックスを処理
            for old_index in sorted_indices:
                if old_index != 0:
                    index_mapping[old_index] = new_index
                    new_index += 1
        else:
            # インデックス0がない場合、最初のインデックスから開始
            for old_index in sorted_indices:
                index_mapping[old_index] = new_index
                new_index += 1

        # マッピングが実際に変更が必要な場合のみ追加
        if any(old != new for old, new in index_mapping.items()):
            mapping[base_name] = index_mapping

    return mapping


def _apply_mapping(program_code: str, index_mapping: Dict[str, Dict[int, int]]) -> str:
    """
    インデックスマッピングをプログラムコードに適用

    Args:
        program_code: プログラムコード文字列
        index_mapping: ベース名 -> (旧インデックス -> 新インデックス) のマッピング

    Returns:
        置換後のプログラムコード
    """
    if not index_mapping:
        return program_code

    result = program_code

    # ベース名ごとに置換（長いベース名から順に処理して、部分一致を防ぐ）
    sorted_bases = sorted(index_mapping.keys(), key=len, reverse=True)

    for base_name in sorted_bases:
        mapping = index_mapping[base_name]

        # 置換を正しい順序で実行
        # 問題: coordinate1 -> coordinate の後、coordinate3 の coordinate 部分も置換されてしまう
        # 解決策: すべての番号付き変数を一時的な名前に置換してから、最終的な名前に置換

        # ステップ1: 一時的な名前に置換（すべての番号付き変数）
        # これにより、coordinate1 -> coordinate の置換が coordinate3 に影響しないようにする
        temp_mapping = {}
        for old_index, new_index in mapping.items():
            if old_index == new_index:
                continue
            if old_index != 0:  # 番号付き変数のみ
                temp_name = f"__TEMP_{base_name}_{old_index}__"
                temp_mapping[old_index] = temp_name
                pattern = r'\b' + re.escape(f"{base_name}{old_index}") + r'\b'
                result = re.sub(pattern, temp_name, result)

        # ステップ2: インデックス0がある場合の処理
        if 0 in mapping:
            old_index_0, new_index_0 = 0, mapping[0]
            if old_index_0 != new_index_0:
                if new_index_0 == 0:
                    # 変更不要
                    pass
                else:
                    # ベース名（番号なし）をベース名+新インデックスに変更
                    pattern = r'\b' + re.escape(base_name) + r'\b'
                    replacement = f"{base_name}{new_index_0}"
                    result = re.sub(pattern, replacement, result)

        # ステップ3: 一時的な名前から最終的な名前に置換
        for old_index, temp_name in temp_mapping.items():
            new_index = mapping[old_index]
            if new_index == 0:
                # ベース名（番号なし）に変更
                pattern = r'\b' + re.escape(temp_name) + r'\b'
                replacement = base_name
                result = re.sub(pattern, replacement, result)
            else:
                # ベース名+新インデックスに変更
                pattern = r'\b' + re.escape(temp_name) + r'\b'
                replacement = f"{base_name}{new_index}"
                result = re.sub(pattern, replacement, result)

    return result
