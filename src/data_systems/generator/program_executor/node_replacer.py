"""
ノード置き換えユーティリティ

ノードツリー内の特定のCommandNodeをfallback_templateに置き換える機能
"""
from typing import Any, Optional, List
from copy import deepcopy
from src.data_systems.generator.program_generator.generation.nodes import (
    CommandNode, VariableNode, LiteralNode, BinaryOpNode, AssignmentNode
)


def parse_fallback_code(fallback_code: str) -> Any:
    """fallback_code文字列をNodeに変換

    Args:
        fallback_code: fallback_templateの文字列（例: "{arg0}", "SET_COLOR({arg0}, 1)"）

    Returns:
        Nodeインスタンス（VariableNode, LiteralNode, CommandNodeなど）
    """
    fallback_code = fallback_code.strip()

    # コマンド呼び出しの場合 (例: SET_COLOR(object[i], 1))
    if '(' in fallback_code and ')' in fallback_code:
        # コマンド名と引数を抽出
        cmd_name = fallback_code.split('(')[0].strip()
        args_str = fallback_code.split('(', 1)[1].rsplit(')', 1)[0]

        # 引数をパース（カンマで分割、ネストされた括弧に対応）
        args = []
        if args_str.strip():
            current_arg = ""
            depth = 0
            for char in args_str:
                if char == '(':
                    depth += 1
                    current_arg += char
                elif char == ')':
                    depth -= 1
                    current_arg += char
                elif char == ',' and depth == 0:
                    if current_arg.strip():
                        args.append(parse_fallback_code(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            if current_arg.strip():
                args.append(parse_fallback_code(current_arg.strip()))

        return CommandNode(cmd_name, args)

    # 数値リテラルの場合
    try:
        value = int(fallback_code)
        return LiteralNode(value)
    except ValueError:
        pass

    try:
        value = float(fallback_code)
        return LiteralNode(value)
    except ValueError:
        pass

    # ブール値リテラルの場合（後方互換性のため大文字小文字を区別しない）
    if fallback_code.upper() == 'TRUE' or fallback_code == 'True':
        return LiteralNode(True)
    if fallback_code.upper() == 'FALSE' or fallback_code == 'False':
        return LiteralNode(False)

    # その他は変数として扱う
    return VariableNode(fallback_code)


def find_target_node_in_copied_tree(original_node: Any, target_node: Any, copied_node: Any) -> Optional[Any]:
    """deepcopy後のノードツリー内で、元のノードに対応するノードを探す

    Args:
        original_node: 元のノードツリー内の現在のノード
        target_node: 探す対象のノード（元のノードツリー内）
        copied_node: deepcopy後のノードツリー内の現在のノード

    Returns:
        見つかったノード、またはNone
    """
    if original_node is None or copied_node is None:
        return None

    # 同じノードインスタンスか確認
    if original_node is target_node:
        return copied_node

    # ノードタイプが異なる場合はNone
    if type(original_node).__name__ != type(copied_node).__name__:
        return None

    # 各ノードタイプに応じて再帰的に探索
    node_type = type(original_node).__name__

    if node_type == 'CommandNode':
        # コマンド名と引数の構造が同じか確認
        if (hasattr(original_node, 'command') and hasattr(copied_node, 'command') and
            original_node.command == copied_node.command and
            hasattr(original_node, 'arguments') and hasattr(copied_node, 'arguments') and
            len(original_node.arguments) == len(copied_node.arguments)):
            # すべての引数が一致する場合は、このノードが対象の可能性がある
            # ただし、ネストされたノードも確認する必要がある
            found = None
            for orig_arg, copy_arg in zip(original_node.arguments, copied_node.arguments):
                found = find_target_node_in_copied_tree(orig_arg, target_node, copy_arg)
                if found:
                    return found
            # 引数が完全に一致する場合は、このノード自体が対象の可能性
            # （ただし、より正確にはgenerate()の結果で比較する方が確実）
            if original_node is target_node:
                return copied_node

    elif node_type == 'AssignmentNode':
        if (hasattr(original_node, 'variable') and hasattr(copied_node, 'variable') and
            original_node.variable == copied_node.variable and
            hasattr(original_node, 'expression') and hasattr(copied_node, 'expression')):
            found = find_target_node_in_copied_tree(original_node.expression, target_node, copied_node.expression)
            if found:
                return found
            if original_node is target_node:
                return copied_node

    elif node_type == 'BinaryOpNode':
        if (hasattr(original_node, 'operator') and hasattr(copied_node, 'operator') and
            original_node.operator == copied_node.operator and
            hasattr(original_node, 'left') and hasattr(copied_node, 'left') and
            hasattr(original_node, 'right') and hasattr(copied_node, 'right')):
            found = find_target_node_in_copied_tree(original_node.left, target_node, copied_node.left)
            if found:
                return found
            found = find_target_node_in_copied_tree(original_node.right, target_node, copied_node.right)
            if found:
                return found
            if original_node is target_node:
                return copied_node

    return None


def replace_command_node_in_tree(node: Any, target_cmd_node: Any, replacement_node: Any, original_tree: Any = None) -> Any:
    """ノードツリー内の特定のCommandNodeを置き換え

    Args:
        node: 置き換え対象のノードツリー（deepcopy後）
        target_cmd_node: 置き換える対象のCommandNode（元のノードツリー内）
        replacement_node: 置き換え後のノード
        original_tree: 元のノードツリー（deepcopy前、ノード探索用）

    Returns:
        置き換え後のノード
    """
    if node is None:
        return None

    # generate()の結果で比較（より確実な方法）
    # ただし、同じ構造のノードが複数ある場合は最初のマッチを置き換える
    if isinstance(node, CommandNode) and isinstance(target_cmd_node, CommandNode):
        try:
            node_code = node.generate()
            target_code = target_cmd_node.generate()
            if node_code == target_code and node.command == target_cmd_node.command:
                return replacement_node
        except Exception:
            pass

    # 同じCommandNodeインスタンスか確認（IDで判定、deepcopy後はIDが変わる可能性がある）
    if node is target_cmd_node:
        return replacement_node

    # 各ノードタイプに応じて再帰的に置き換え
    node_type = type(node).__name__

    if node_type == 'CommandNode':
        if hasattr(node, 'arguments'):
            new_args = [replace_command_node_in_tree(arg, target_cmd_node, replacement_node)
                       for arg in node.arguments]
            new_node = CommandNode(node.command, new_args, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    elif node_type == 'AssignmentNode':
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_in_tree(node.expression, target_cmd_node, replacement_node)
            new_node = AssignmentNode(node.variable, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    elif node_type == 'BinaryOpNode':
        if hasattr(node, 'left') and hasattr(node, 'right'):
            new_left = replace_command_node_in_tree(node.left, target_cmd_node, replacement_node)
            new_right = replace_command_node_in_tree(node.right, target_cmd_node, replacement_node)
            new_node = BinaryOpNode(node.operator, new_left, new_right, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    elif node_type == 'FilterNode':
        if hasattr(node, 'condition'):
            new_condition = replace_command_node_in_tree(node.condition, target_cmd_node, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import FilterNode
            new_node = FilterNode(node.source_array, node.target_array, new_condition, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    elif node_type == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_in_tree(node.expression, target_cmd_node, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import ArrayAssignmentNode
            new_node = ArrayAssignmentNode(node.array, node.index, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # その他のノードタイプはそのまま返す
    return node


def find_and_replace_command_in_tree(nodes_list: List[Any], target_cmd_node: Any, replacement_node: Any, original_nodes: List[Any] = None) -> List[Any]:
    """ノードリスト内で特定のCommandNodeを見つけて置き換え

    Args:
        nodes_list: ノードリスト（deepcopy後）
        target_cmd_node: 置き換える対象のCommandNode（元のノードツリー内）
        replacement_node: 置き換え後のノード
        original_nodes: 元のノードリスト（deepcopy前、ノード対応付け用）

    Returns:
        置き換え後のノードリスト
    """
    if not isinstance(target_cmd_node, CommandNode):
        return nodes_list

    # 元のノードリストが提供されている場合、IDで対応付けて正確に置き換える
    if original_nodes is not None:
        # 元のノードツリー内でtarget_cmd_nodeのIDを取得
        target_node_id = getattr(target_cmd_node, 'id', None)
        if target_node_id is not None:
            # コピー後のノードツリー内で同じIDを持つノードを探して置き換え
            replaced_nodes = []
            for orig_node, copied_node in zip(original_nodes, nodes_list):
                replaced = replace_command_node_by_id(copied_node, target_node_id, replacement_node)
                replaced_nodes.append(replaced)
            return replaced_nodes

    # フォールバック: generate()の結果で比較（1回だけ置き換え）
    try:
        target_code = target_cmd_node.generate()
        target_cmd_name = target_cmd_node.command
    except Exception:
        return nodes_list

    # 置き換えカウンター（最初の1回だけ置き換える）
    replacement_done = {'count': 0}

    replaced_nodes = []
    for node in nodes_list:
        replaced = replace_command_node_by_code_once(node, target_cmd_name, target_code, replacement_node, replacement_done)
        replaced_nodes.append(replaced)

    return replaced_nodes


def replace_command_node_by_id(node: Any, target_node_id: str, replacement_node: Any) -> Any:
    """IDでCommandNodeを特定して置き換え（正確な方法）

    Args:
        node: 置き換え対象のノード
        target_node_id: 対象ノードのID
        replacement_node: 置き換え後のノード

    Returns:
        置き換え後のノード
    """
    if node is None:
        return None

    # CommandNodeの場合、IDで比較
    if isinstance(node, CommandNode):
        node_id = getattr(node, 'id', None)
        if node_id == target_node_id:
            return replacement_node

        # 引数も再帰的に探索
        if hasattr(node, 'arguments'):
            new_args = [replace_command_node_by_id(arg, target_node_id, replacement_node)
                       for arg in node.arguments]
            new_node = CommandNode(node.command, new_args, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # AssignmentNodeの場合
    elif isinstance(node, AssignmentNode):
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_id(node.expression, target_node_id, replacement_node)
            new_node = AssignmentNode(node.variable, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # BinaryOpNodeの場合
    elif isinstance(node, BinaryOpNode):
        if hasattr(node, 'left') and hasattr(node, 'right'):
            new_left = replace_command_node_by_id(node.left, target_node_id, replacement_node)
            new_right = replace_command_node_by_id(node.right, target_node_id, replacement_node)
            new_node = BinaryOpNode(node.operator, new_left, new_right, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # FilterNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'FilterNode':
        if hasattr(node, 'condition'):
            new_condition = replace_command_node_by_id(node.condition, target_node_id, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import FilterNode
            new_node = FilterNode(node.source_array, node.target_array, new_condition, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # ArrayAssignmentNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_id(node.expression, target_node_id, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import ArrayAssignmentNode
            new_node = ArrayAssignmentNode(node.array, node.index, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # その他のノードタイプはそのまま返す
    return node


def replace_command_node_by_code_once(node: Any, target_cmd_name: str, target_code: str, replacement_node: Any, replacement_done: dict) -> Any:
    """generate()の結果でCommandNodeを特定して置き換え（最初の1回だけ）

    Args:
        node: 置き換え対象のノード
        target_cmd_name: 対象コマンド名
        target_code: 対象ノードのgenerate()結果
        replacement_node: 置き換え後のノード
        replacement_done: 置き換え済みフラグ（{'count': int}）

    Returns:
        置き換え後のノード
    """
    if node is None:
        return None

    # 既に置き換え済みの場合はスキップ
    if replacement_done['count'] > 0:
        return node

    # CommandNodeの場合、generate()の結果で比較
    if isinstance(node, CommandNode):
        try:
            node_code = node.generate()
            if node_code == target_code and node.command == target_cmd_name:
                replacement_done['count'] = 1
                return replacement_node
        except Exception:
            pass

        # 引数も再帰的に探索
        if hasattr(node, 'arguments'):
            new_args = [replace_command_node_by_code_once(arg, target_cmd_name, target_code, replacement_node, replacement_done)
                       for arg in node.arguments]
            new_node = CommandNode(node.command, new_args, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # AssignmentNodeの場合
    elif isinstance(node, AssignmentNode):
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_code_once(node.expression, target_cmd_name, target_code, replacement_node, replacement_done)
            new_node = AssignmentNode(node.variable, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # BinaryOpNodeの場合
    elif isinstance(node, BinaryOpNode):
        if hasattr(node, 'left') and hasattr(node, 'right'):
            new_left = replace_command_node_by_code_once(node.left, target_cmd_name, target_code, replacement_node, replacement_done)
            new_right = replace_command_node_by_code_once(node.right, target_cmd_name, target_code, replacement_node, replacement_done)
            new_node = BinaryOpNode(node.operator, new_left, new_right, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # FilterNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'FilterNode':
        if hasattr(node, 'condition'):
            new_condition = replace_command_node_by_code_once(node.condition, target_cmd_name, target_code, replacement_node, replacement_done)
            from src.data_systems.generator.program_generator.generation.nodes import FilterNode
            new_node = FilterNode(node.source_array, node.target_array, new_condition, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # ArrayAssignmentNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_code_once(node.expression, target_cmd_name, target_code, replacement_node, replacement_done)
            from src.data_systems.generator.program_generator.generation.nodes import ArrayAssignmentNode
            new_node = ArrayAssignmentNode(node.array, node.index, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # その他のノードタイプはそのまま返す
    return node


def replace_command_node_by_code(node: Any, target_cmd_name: str, target_code: str, replacement_node: Any) -> Any:
    """generate()の結果でCommandNodeを特定して置き換え

    Args:
        node: 置き換え対象のノード
        target_cmd_name: 対象コマンド名
        target_code: 対象ノードのgenerate()結果
        replacement_node: 置き換え後のノード

    Returns:
        置き換え後のノード
    """
    if node is None:
        return None

    # CommandNodeの場合、generate()の結果で比較
    if isinstance(node, CommandNode):
        try:
            node_code = node.generate()
            if node_code == target_code and node.command == target_cmd_name:
                return replacement_node
        except Exception:
            pass

        # 引数も再帰的に探索
        if hasattr(node, 'arguments'):
            new_args = [replace_command_node_by_code(arg, target_cmd_name, target_code, replacement_node)
                       for arg in node.arguments]
            new_node = CommandNode(node.command, new_args, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # AssignmentNodeの場合
    elif isinstance(node, AssignmentNode):
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_code(node.expression, target_cmd_name, target_code, replacement_node)
            new_node = AssignmentNode(node.variable, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # BinaryOpNodeの場合
    elif isinstance(node, BinaryOpNode):
        if hasattr(node, 'left') and hasattr(node, 'right'):
            new_left = replace_command_node_by_code(node.left, target_cmd_name, target_code, replacement_node)
            new_right = replace_command_node_by_code(node.right, target_cmd_name, target_code, replacement_node)
            new_node = BinaryOpNode(node.operator, new_left, new_right, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # FilterNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'FilterNode':
        if hasattr(node, 'condition'):
            new_condition = replace_command_node_by_code(node.condition, target_cmd_name, target_code, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import FilterNode
            new_node = FilterNode(node.source_array, node.target_array, new_condition, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # ArrayAssignmentNodeの場合
    elif hasattr(node, '__class__') and node.__class__.__name__ == 'ArrayAssignmentNode':
        if hasattr(node, 'expression'):
            new_expr = replace_command_node_by_code(node.expression, target_cmd_name, target_code, replacement_node)
            from src.data_systems.generator.program_generator.generation.nodes import ArrayAssignmentNode
            new_node = ArrayAssignmentNode(node.array, node.index, new_expr, node.context if hasattr(node, 'context') else None)
            if hasattr(node, 'id'):
                new_node.id = node.id
            return new_node

    # その他のノードタイプはそのまま返す
    return node
