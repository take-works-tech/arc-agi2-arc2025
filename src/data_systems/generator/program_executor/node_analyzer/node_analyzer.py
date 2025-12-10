"""
Node解析ユーティリティ

プログラムのNodeリストを解析する機能を提供
"""
from typing import Any, List
from ..validation.commands import COMMAND_METADATA


def is_assignment_node(node: Any) -> bool:
    """Nodeが代入式（=を含む）かどうかを判定

    Args:
        node: チェックするNodeオブジェクト

    Returns:
        代入式の場合はTrue、そうでなければFalse
    """
    if node is None:
        return False

    node_type = type(node).__name__

    # 代入系のノードタイプをチェック
    assignment_node_types = [
        'AssignmentNode',        # variable = expression
        'ArrayAssignmentNode',   # array[index] = expression
        'FilterNode',            # target = FILTER(...)
        'ExcludeNode',           # target = EXCLUDE(...)
        'ConcatNode',            # target = CONCAT(...)
        'AppendNode',            # target = APPEND(...)
        'MergeNode',             # target = MERGE(...)
        'MatchPairsNode',        # target = MATCH_PAIRS(...)
        'ExtendPatternNode',     # target = EXTEND_PATTERN(...)
        'ArrangeGridNode',       # target = ARRANGE_GRID(...)
        'ExtractShapeNode',      # target = EXTRACT_RECTS/EXTRACT_HOLLOW_RECTS/EXTRACT_LINES(...)
    ]

    if node_type in assignment_node_types:
        return True

    # フォールバック: generate()メソッドの結果に=が含まれているかチェック
    if hasattr(node, 'generate'):
        try:
            generated_code = node.generate()
            if isinstance(generated_code, str) and '=' in generated_code:
                # '=='（等価比較）ではなく単一の'='（代入）を含むかチェック
                # ただし、'=='が含まれている場合は代入ではない
                if ' == ' in generated_code or '!=' in generated_code:
                    return False
                return ' = ' in generated_code or generated_code.strip().startswith('objects =') or generated_code.strip().startswith('array =')
        except Exception:
            pass

    return False


def get_assignment_nodes(nodes: List[Any]) -> List[Any]:
    """Nodeリストから代入式のNodeのみを抽出

    Args:
        nodes: チェックするNodeオブジェクトのリスト

    Returns:
        代入式のNodeのリスト
    """
    return [node for node in nodes if is_assignment_node(node)]


def is_if_node(node: Any) -> bool:
    """NodeがIF文かどうかを判定

    Args:
        node: チェックするNodeオブジェクト

    Returns:
        IF文の場合はTrue、そうでなければFalse
    """
    if node is None:
        return False

    node_type = type(node).__name__

    # IF系のノードタイプをチェック
    if_node_types = [
        'IfStartNode',      # IF文開始
        'IfBranchNode',     # IF分岐
    ]

    return node_type in if_node_types


def is_for_node(node: Any) -> bool:
    """NodeがFORループかどうかを判定

    Args:
        node: チェックするNodeオブジェクト

    Returns:
        FORループの場合はTrue、そうでなければFalse
    """
    if node is None:
        return False

    node_type = type(node).__name__

    # FOR系のノードタイプをチェック
    for_node_types = [
        'ForStartNode',              # FORループ開始
        'ForStartWithCountNode',     # COUNT型変数を使用したFORループ開始
        'ForStartWithConstantNode',  # 定数値を使用したFORループ開始
    ]

    return node_type in for_node_types


def is_end_node(node: Any) -> bool:
    """NodeがEND（終了）ノードかどうかを判定

    Args:
        node: チェックするNodeオブジェクト

    Returns:
        ENDノードの場合はTrue、そうでなければFalse
    """
    if node is None:
        return False

    node_type = type(node).__name__

    # ENDノードタイプをチェック
    return node_type == 'EndNode'


def count_if_and_for_nodes(nodes: List[Any]) -> dict:
    """NodeリストからIF文とFORループの数をカウント

    Args:
        nodes: チェックするNodeオブジェクトのリスト

    Returns:
        {'if_count': int, 'for_count': int} の辞書
    """
    if_count = 0
    for_count = 0

    for node in nodes:
        if is_if_node(node):
            if_count += 1
        elif is_for_node(node):
            for_count += 1

    return {
        'if_count': if_count,
        'for_count': for_count
    }


def extract_commands_from_node(node: Any) -> List[str]:
    """Nodeから再帰的にすべてのコマンド名を抽出

    Args:
        node: 抽出するNodeオブジェクト

    Returns:
        コマンド名のリスト（重複を含む可能性がある）
    """
    commands = []

    if node is None:
        return commands

    node_type = type(node).__name__

    # CommandNodeの場合はコマンド名を追加
    if node_type == 'CommandNode':
        if hasattr(node, 'command'):
            commands.append(node.command)
        # 引数も再帰的に探索
        if hasattr(node, 'arguments') and node.arguments:
            for arg in node.arguments:
                commands.extend(extract_commands_from_node(arg))

    # BinaryOpNodeの場合は左右を探索
    elif node_type == 'BinaryOpNode':
        if hasattr(node, 'left'):
            commands.extend(extract_commands_from_node(node.left))
        if hasattr(node, 'right'):
            commands.extend(extract_commands_from_node(node.right))

    # AssignmentNodeの場合はexpressionを探索
    elif node_type == 'AssignmentNode':
        if hasattr(node, 'expression'):
            commands.extend(extract_commands_from_node(node.expression))

    # ArrayAssignmentNodeの場合はexpressionを探索
    elif node_type == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            commands.extend(extract_commands_from_node(node.expression))

    # FilterNodeの場合はFILTERコマンド自体を追加し、条件も探索
    elif node_type == 'FilterNode':
        # FILTERコマンド自体を追加
        commands.append('FILTER')
        # 条件部分も探索
        if hasattr(node, 'condition'):
            commands.extend(extract_commands_from_node(node.condition))
        if hasattr(node, 'array'):
            # arrayは変数名なのでスキップ
            pass

    # ExcludeNodeの場合はEXCLUDEコマンド自体を追加
    elif node_type == 'ExcludeNode':
        commands.append('EXCLUDE')

    # ConcatNodeの場合はCONCATコマンド自体を追加
    elif node_type == 'ConcatNode':
        commands.append('CONCAT')

    # AppendNodeの場合はAPPENDコマンド自体を追加
    elif node_type == 'AppendNode':
        commands.append('APPEND')

    # MergeNodeの場合はMERGEコマンド自体を追加
    elif node_type == 'MergeNode':
        commands.append('MERGE')

    # MatchPairsNodeの場合はMATCH_PAIRSコマンド自体を追加
    elif node_type == 'MatchPairsNode':
        commands.append('MATCH_PAIRS')

    # ExtendPatternNodeの場合はEXTEND_PATTERNコマンド自体を追加
    elif node_type == 'ExtendPatternNode':
        commands.append('EXTEND_PATTERN')

    # ArrangeGridNodeの場合はARRANGE_GRIDコマンド自体を追加
    elif node_type == 'ArrangeGridNode':
        commands.append('ARRANGE_GRID')

    # ExtractShapeNodeの場合はextract_typeに応じてコマンド名を追加
    elif node_type == 'ExtractShapeNode':
        extract_type = getattr(node, 'extract_type', 'rects')
        if extract_type == 'rects':
            commands.append('EXTRACT_RECTS')
        elif extract_type == 'hollow_rects':
            commands.append('EXTRACT_HOLLOW_RECTS')
        elif extract_type == 'lines':
            commands.append('EXTRACT_LINES')

    # RenderNodeの場合は引数を探索
    elif node_type == 'RenderNode':
        if hasattr(node, 'bg_color'):
            commands.extend(extract_commands_from_node(node.bg_color))
        if hasattr(node, 'width'):
            commands.extend(extract_commands_from_node(node.width))
        if hasattr(node, 'height'):
            commands.extend(extract_commands_from_node(node.height))

    # その他のノードタイプで引数を持つ可能性があるものを探索
    # 一般的なattributesをチェック
    if hasattr(node, 'arguments') and isinstance(getattr(node, 'arguments'), list):
        for arg in node.arguments:
            commands.extend(extract_commands_from_node(arg))

    return commands


def extract_all_node_info(node: Any) -> dict:
    """Nodeから再帰的にすべてのコマンドと引数の情報を抽出

    Args:
        node: 抽出するNodeオブジェクト

    Returns:
        {
            'commands': List[str],  # すべてのコマンド名
            'arguments': List[Any],  # すべての引数Node
            'structure': dict  # ネスト構造の詳細情報
        }
    """
    result = {
        'commands': [],
        'arguments': [],
        'structure': {}
    }

    if node is None:
        return result

    node_type = type(node).__name__
    result['structure']['node_type'] = node_type

    # CommandNodeの場合
    if node_type == 'CommandNode':
        if hasattr(node, 'command'):
            result['commands'].append(node.command)
            result['structure']['command'] = node.command
        if hasattr(node, 'arguments'):
            result['arguments'].extend(node.arguments)
            result['structure']['arguments'] = [
                extract_all_node_info(arg) for arg in node.arguments
            ]

    # BinaryOpNodeの場合
    elif node_type == 'BinaryOpNode':
        if hasattr(node, 'operator'):
            result['structure']['operator'] = node.operator
        if hasattr(node, 'left'):
            left_info = extract_all_node_info(node.left)
            result['commands'].extend(left_info['commands'])
            result['arguments'].extend(left_info['arguments'])
            result['structure']['left'] = left_info['structure']
        if hasattr(node, 'right'):
            right_info = extract_all_node_info(node.right)
            result['commands'].extend(right_info['commands'])
            result['arguments'].extend(right_info['arguments'])
            result['structure']['right'] = right_info['structure']

    # AssignmentNodeの場合
    elif node_type == 'AssignmentNode':
        if hasattr(node, 'variable'):
            result['structure']['variable'] = node.variable
        if hasattr(node, 'expression'):
            expr_info = extract_all_node_info(node.expression)
            result['commands'].extend(expr_info['commands'])
            result['arguments'].extend(expr_info['arguments'])
            result['structure']['expression'] = expr_info['structure']

    # ArrayAssignmentNodeの場合
    elif node_type == 'ArrayAssignmentNode':
        if hasattr(node, 'array'):
            result['structure']['array'] = node.array
        if hasattr(node, 'index'):
            result['structure']['index'] = node.index
        if hasattr(node, 'expression'):
            expr_info = extract_all_node_info(node.expression)
            result['commands'].extend(expr_info['commands'])
            result['arguments'].extend(expr_info['arguments'])
            result['structure']['expression'] = expr_info['structure']

    # FilterNodeの場合
    elif node_type == 'FilterNode':
        if hasattr(node, 'array'):
            result['structure']['array'] = node.array
        if hasattr(node, 'condition'):
            cond_info = extract_all_node_info(node.condition)
            result['commands'].extend(cond_info['commands'])
            result['arguments'].extend(cond_info['arguments'])
            result['structure']['condition'] = cond_info['structure']

    # RenderNodeの場合
    elif node_type == 'RenderNode':
        if hasattr(node, 'array'):
            result['structure']['array'] = node.array
        for attr_name in ['bg_color', 'width', 'height']:
            if hasattr(node, attr_name):
                attr_info = extract_all_node_info(getattr(node, attr_name))
                result['commands'].extend(attr_info['commands'])
                result['arguments'].extend(attr_info['arguments'])
                result['structure'][attr_name] = attr_info['structure']

    # LiteralNodeの場合
    elif node_type == 'LiteralNode':
        if hasattr(node, 'value'):
            result['structure']['value'] = node.value

    # VariableNodeの場合
    elif node_type == 'VariableNode':
        if hasattr(node, 'variable'):
            result['structure']['variable'] = node.variable

    return result


def extract_commands_with_depth(node: Any, current_depth: int = 0) -> List[dict]:
    """Nodeから再帰的にすべてのコマンドとそのネスト深度を抽出

    Args:
        node: 抽出するNodeオブジェクト
        current_depth: 現在のネスト深度（デフォルト: 0）

    Returns:
        コマンド情報のリスト [{'command': str, 'depth': int, 'node': Any}, ...]
    """
    commands = []

    if node is None:
        return commands

    node_type = type(node).__name__

    # CommandNodeの場合はコマンド名を追加
    if node_type == 'CommandNode':
        if hasattr(node, 'command'):
            commands.append({
                'command': node.command,
                'depth': current_depth,
                'node': node
            })
        # 引数も再帰的に探索（深度を1増やす）
        if hasattr(node, 'arguments') and node.arguments:
            for arg in node.arguments:
                commands.extend(extract_commands_with_depth(arg, current_depth + 1))

    # BinaryOpNodeの場合は左右を探索
    elif node_type == 'BinaryOpNode':
        if hasattr(node, 'left'):
            commands.extend(extract_commands_with_depth(node.left, current_depth + 1))
        if hasattr(node, 'right'):
            commands.extend(extract_commands_with_depth(node.right, current_depth + 1))

    # AssignmentNodeの場合はexpressionを探索
    elif node_type == 'AssignmentNode':
        if hasattr(node, 'expression'):
            commands.extend(extract_commands_with_depth(node.expression, current_depth + 1))

    # ArrayAssignmentNodeの場合はexpressionを探索
    elif node_type == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            commands.extend(extract_commands_with_depth(node.expression, current_depth + 1))

    # FilterNodeの場合はFILTERコマンド自体を追加し、条件も探索
    elif node_type == 'FilterNode':
        # FILTERコマンド自体を追加（深度0、つまりFilterNodeと同じ深度）
        commands.append({
            'command': 'FILTER',
            'depth': current_depth,
            'node': node
        })
        # 条件部分も探索（深度を1増やす）
        if hasattr(node, 'condition'):
            commands.extend(extract_commands_with_depth(node.condition, current_depth + 1))

    # ExcludeNodeの場合はEXCLUDEコマンド自体を追加
    elif node_type == 'ExcludeNode':
        commands.append({
            'command': 'EXCLUDE',
            'depth': current_depth,
            'node': node
        })

    # ConcatNodeの場合はCONCATコマンド自体を追加
    elif node_type == 'ConcatNode':
        commands.append({
            'command': 'CONCAT',
            'depth': current_depth,
            'node': node
        })

    # AppendNodeの場合はAPPENDコマンド自体を追加
    elif node_type == 'AppendNode':
        commands.append({
            'command': 'APPEND',
            'depth': current_depth,
            'node': node
        })

    # MergeNodeの場合はMERGEコマンド自体を追加
    elif node_type == 'MergeNode':
        commands.append({
            'command': 'MERGE',
            'depth': current_depth,
            'node': node
        })

    # MatchPairsNodeの場合はMATCH_PAIRSコマンド自体を追加
    elif node_type == 'MatchPairsNode':
        commands.append({
            'command': 'MATCH_PAIRS',
            'depth': current_depth,
            'node': node
        })

    # ExtendPatternNodeの場合はEXTEND_PATTERNコマンド自体を追加
    elif node_type == 'ExtendPatternNode':
        commands.append({
            'command': 'EXTEND_PATTERN',
            'depth': current_depth,
            'node': node
        })

    # ArrangeGridNodeの場合はARRANGE_GRIDコマンド自体を追加
    elif node_type == 'ArrangeGridNode':
        commands.append({
            'command': 'ARRANGE_GRID',
            'depth': current_depth,
            'node': node
        })

    # ExtractShapeNodeの場合はextract_typeに応じてコマンド名を追加
    elif node_type == 'ExtractShapeNode':
        extract_type = getattr(node, 'extract_type', 'rects')
        if extract_type == 'rects':
            commands.append({
                'command': 'EXTRACT_RECTS',
                'depth': current_depth,
                'node': node
            })
        elif extract_type == 'hollow_rects':
            commands.append({
                'command': 'EXTRACT_HOLLOW_RECTS',
                'depth': current_depth,
                'node': node
            })
        elif extract_type == 'lines':
            commands.append({
                'command': 'EXTRACT_LINES',
                'depth': current_depth,
                'node': node
            })

    # RenderNodeの場合は引数を探索
    elif node_type == 'RenderNode':
        if hasattr(node, 'bg_color'):
            commands.extend(extract_commands_with_depth(node.bg_color, current_depth + 1))
        if hasattr(node, 'width'):
            commands.extend(extract_commands_with_depth(node.width, current_depth + 1))
        if hasattr(node, 'height'):
            commands.extend(extract_commands_with_depth(node.height, current_depth + 1))

    # その他のノードタイプで引数を持つ可能性があるものを探索
    if hasattr(node, 'arguments') and isinstance(getattr(node, 'arguments'), list):
        for arg in node.arguments:
            commands.extend(extract_commands_with_depth(arg, current_depth + 1))

    return commands


def get_commands_sorted_by_depth(node: Any, reverse: bool = True, exclude_no_effectiveness: bool = True) -> List[dict]:
    """Nodeからコマンドを抽出し、深度順にソートして返す

    Args:
        node: 抽出するNodeオブジェクト
        reverse: Trueの場合は深度が深い順（デフォルト）、Falseの場合は浅い順
        exclude_no_effectiveness: Trueの場合、validate_effectiveness=Falseのコマンドを除外（デフォルト: True）

    Returns:
        コマンド情報のリスト [{'command': str, 'depth': int, 'node': Any}, ...]
        深度順にソート済み
    """
    commands = extract_commands_with_depth(node, current_depth=0)

    # validate_effectiveness=Falseのコマンドを除外
    if exclude_no_effectiveness:
        filtered_commands = []
        for cmd_info in commands:
            cmd_name = cmd_info['command']
            # COMMAND_METADATAに存在し、validate_effectivenessがFalseの場合は除外
            if cmd_name in COMMAND_METADATA:
                metadata = COMMAND_METADATA[cmd_name]
                if metadata.validate_effectiveness:
                    filtered_commands.append(cmd_info)
            else:
                # メタデータに存在しないコマンドは含める
                filtered_commands.append(cmd_info)
        commands = filtered_commands

    # 深度が深い順（reverse=True）または浅い順（reverse=False）にソート
    commands.sort(key=lambda x: x['depth'], reverse=reverse)
    return commands