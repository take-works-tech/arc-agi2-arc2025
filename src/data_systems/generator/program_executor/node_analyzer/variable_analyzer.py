"""
変数分析ユーティリティ

変数の定義・使用状況を分析し、未使用変数を検出・削除する
"""
from typing import List, Set, Dict, Any, Optional
import re
from .node_analyzer import is_for_node, is_if_node, is_end_node


def extract_variable_definitions_from_node(node: Any) -> List[str]:
    """ノードから定義される変数を抽出

    Args:
        node: 分析対象のノード

    Returns:
        定義される変数名のリスト
    """
    defined_vars = []
    node_type = type(node).__name__

    # InitializationNode: objects = GET_ALL_OBJECTS(...)
    if node_type == 'InitializationNode':
        # InitializationNodeは常に"objects"を定義
        defined_vars.append('objects')

    # AssignmentNode: variable = expression
    elif node_type == 'AssignmentNode':
        if hasattr(node, 'variable'):
            defined_vars.append(node.variable)

    # ArrayAssignmentNode: array[index] = expression
    # これは配列要素への代入なので、新規変数の定義ではない
    # ただし、配列自体が定義される場合もあるが、通常は既存配列への要素代入

    # FilterNode: target_array = FILTER(...)
    elif node_type == 'FilterNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ExcludeNode: target_array = EXCLUDE(...)
    elif node_type == 'ExcludeNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ConcatNode: target_array = CONCAT(...)
    elif node_type == 'ConcatNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # AppendNode: target_array = APPEND(...)
    elif node_type == 'AppendNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # EmptyArrayNode: array_name = []
    elif node_type == 'EmptyArrayNode':
        if hasattr(node, 'array_name'):
            defined_vars.append(node.array_name)

    # MatchPairsNode: target_array = MATCH_PAIRS(...)
    elif node_type == 'MatchPairsNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ObjectAccessNode: new_object = objects[i] など
    elif node_type == 'ObjectAccessNode':
        if hasattr(node, 'obj_var'):
            defined_vars.append(node.obj_var)
        elif hasattr(node, 'variable'):
            defined_vars.append(node.variable)

    # MergeNode: target_obj = MERGE(...)
    elif node_type == 'MergeNode':
        if hasattr(node, 'target_obj'):
            defined_vars.append(node.target_obj)

    # SingleObjectArrayNode: array_name = [object_name]
    elif node_type == 'SingleObjectArrayNode':
        if hasattr(node, 'array_name'):
            defined_vars.append(node.array_name)

    # SplitConnectedNode: target_array = SPLIT_CONNECTED(...)
    elif node_type == 'SplitConnectedNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ExtractShapeNode: target_array = EXTRACT_RECTS(...) など
    elif node_type == 'ExtractShapeNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ExtendPatternNode: target_array = EXTEND_PATTERN(...)
    elif node_type == 'ExtendPatternNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    # ArrangeGridNode: target_array = ARRANGE_GRID(...)
    elif node_type == 'ArrangeGridNode':
        if hasattr(node, 'target_array'):
            defined_vars.append(node.target_array)

    return defined_vars


def extract_variable_usages_from_node(node: Any) -> Set[str]:
    """ノードから使用される変数を抽出

    Args:
        node: 分析対象のノード

    Returns:
        使用される変数名のセット
    """
    used_vars = set()

    # ノードの属性から変数を抽出（generate()より先に処理）
    node_type = type(node).__name__

    # AssignmentNode: variable = expression
    # 左辺の変数名は「定義」なので、右辺の式からのみ変数を抽出
    if node_type == 'AssignmentNode':
        if hasattr(node, 'expression'):
            used_vars.update(_extract_vars_from_expression(node.expression))
        # generate()からは抽出しない（左辺の変数名が含まれるため）
        return used_vars

    # ノードのgenerate()メソッドで生成されるコードから変数を抽出
    try:
        code = node.generate()
        # コードから変数名を抽出（正規表現を使用）
        # プレースホルダー（$objなど）や関数名、リテラルを除外
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(var_pattern, code)

        # 除外するキーワード
        excluded = {
            # 関数名・コマンド名
            'GET_ALL_OBJECTS', 'GET_COLOR', 'GET_SHAPE', 'GET_POSITION', 'GET_SIZE',
            'FILTER', 'EXCLUDE', 'CONCAT', 'APPEND', 'MERGE', 'SORT',
            'MOVE', 'ROTATE', 'SCALE', 'SET_COLOR', 'SET_SHAPE',
            'EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS', 'GREATER_EQUAL', 'LESS_EQUAL',
            'FOR', 'DO', 'END', 'IF', 'THEN', 'ELSE',
            'LEN', 'COUNT', 'RENDER_GRID',
            # 予約語・定数
            'True', 'False', 'X', 'Y', 'start', 'end', 'first', 'last',
            # プレースホルダー（$で始まる）
        }

        for match in matches:
            # 除外チェック
            if match in excluded:
                continue
            if match.startswith('$'):
                continue
            if match.isdigit():
                continue
            # 変数として使用されている可能性がある
            used_vars.add(match)
    except Exception:
        # generate()が失敗した場合は属性から直接抽出を試みる
        pass

    # ArrayAssignmentNode: array[index] = expression
    if node_type == 'ArrayAssignmentNode':
        if hasattr(node, 'array'):
            used_vars.add(node.array)
        if hasattr(node, 'index'):
            # インデックスが変数の場合
            if isinstance(node.index, str) and not node.index.isdigit():
                used_vars.add(node.index)
        if hasattr(node, 'expression'):
            used_vars.update(_extract_vars_from_expression(node.expression))

    # FilterNode: target_array = FILTER(source_array, condition)
    elif node_type == 'FilterNode':
        if hasattr(node, 'source_array'):
            used_vars.add(node.source_array)

    # ForStartNode: FOR loop_var LEN(array) DO
    if node_type == 'ForStartNode':
        if hasattr(node, 'array'):
            used_vars.add(node.array)

    # ForStartWithCountNode: FOR loop_var count_variable DO
    if node_type == 'ForStartWithCountNode':
        if hasattr(node, 'count_variable'):
            used_vars.add(node.count_variable)

    # ObjectAccessNode: obj = objects[i] または obj = objects[SUB(LEN(objects), 1)]
    if node_type == 'ObjectAccessNode':
        if hasattr(node, 'objects_array'):
            used_vars.add(node.objects_array)
        # generate()で生成されるコードにLEN(objects)が含まれるので、そこからも検出される

    # FilterNode: target_array = FILTER(source_array, condition)
    elif node_type == 'FilterNode':
        if hasattr(node, 'source_array'):
            used_vars.add(node.source_array)
        # condition内の変数も抽出（文字列の場合はgenerate()で処理済み、Nodeの場合は再帰的に抽出）
        if hasattr(node, 'condition'):
            if not isinstance(node.condition, str):
                used_vars.update(_extract_vars_from_expression(node.condition))

    # ExcludeNode: target_array = EXCLUDE(source_array, targets_array)
    elif node_type == 'ExcludeNode':
        if hasattr(node, 'source_array'):
            used_vars.add(node.source_array)
        if hasattr(node, 'targets_array'):
            used_vars.add(node.targets_array)

    # ConcatNode: target_array = CONCAT(array1, array2)
    elif node_type == 'ConcatNode':
        if hasattr(node, 'array1'):
            used_vars.add(node.array1)
        if hasattr(node, 'array2'):
            used_vars.add(node.array2)

    # AppendNode: target_array = APPEND(array, obj)
    elif node_type == 'AppendNode':
        if hasattr(node, 'array'):
            used_vars.add(node.array)
        if hasattr(node, 'obj'):
            used_vars.add(node.obj)

    # MatchPairsNode: target_array = MATCH_PAIRS(array1, array2, ...)
    elif node_type == 'MatchPairsNode':
        if hasattr(node, 'array1'):
            used_vars.add(node.array1)
        if hasattr(node, 'array2'):
            used_vars.add(node.array2)

    # RenderNode: RENDER_GRID(array, ...)
    elif node_type == 'RenderNode':
        if hasattr(node, 'array'):
            used_vars.add(node.array)

    # MergeNode: target_obj = MERGE(objects_array)
    elif node_type == 'MergeNode':
        if hasattr(node, 'objects_array'):
            used_vars.add(node.objects_array)

    # SingleObjectArrayNode: array_name = [object_name]
    elif node_type == 'SingleObjectArrayNode':
        if hasattr(node, 'object_name'):
            used_vars.add(node.object_name)

    # SplitConnectedNode: target_array = SPLIT_CONNECTED(source_object, ...)
    elif node_type == 'SplitConnectedNode':
        if hasattr(node, 'source_object'):
            used_vars.add(node.source_object)

    # ExtractShapeNode: target_array = EXTRACT_RECTS(source_object) など
    elif node_type == 'ExtractShapeNode':
        if hasattr(node, 'source_object'):
            used_vars.add(node.source_object)

    # ExtendPatternNode: target_array = EXTEND_PATTERN(source_array, ...)
    elif node_type == 'ExtendPatternNode':
        if hasattr(node, 'source_array'):
            used_vars.add(node.source_array)

    # ArrangeGridNode: target_array = ARRANGE_GRID(source_array, ...)
    elif node_type == 'ArrangeGridNode':
        if hasattr(node, 'source_array'):
            used_vars.add(node.source_array)

    # IfBranchNode: IF condition THEN ... ELSE ... END
    elif node_type == 'IfBranchNode':
        # condition内の変数を抽出（文字列の場合はgenerate()で処理済み、Nodeの場合は再帰的に抽出）
        if hasattr(node, 'condition'):
            if isinstance(node.condition, str):
                # 文字列の場合は正規表現で変数を抽出
                var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
                matches = re.findall(var_pattern, node.condition)
                excluded = {
                    'IF', 'THEN', 'ELSE', 'END', 'EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS',
                    'GREATER_EQUAL', 'LESS_EQUAL', 'AND', 'OR', 'True', 'False',
                    'GET_COLOR', 'GET_SIZE', 'GET_X', 'GET_Y', 'GET_WIDTH', 'GET_HEIGHT',
                    'IS_INSIDE', 'IS_IDENTICAL', 'IS_SAME_STRUCT', 'GET_BACKGROUND_COLOR',
                    'GET_INPUT_GRID_SIZE', 'LEN', 'COUNT', 'GET_DISTANCE', 'GET_DENSITY',
                    'GET_ASPECT_RATIO', 'GET_CENTER_X', 'GET_CENTER_Y', 'GET_MAX_X', 'GET_MAX_Y',
                    'GET_SYMMETRY_SCORE', 'GET_LINE_TYPE', 'GET_RECTANGLE_TYPE', 'COUNT_HOLES',
                    'COUNT_ADJACENT', 'COUNT_OVERLAP', 'X', 'Y', 'start', 'end', 'first', 'last'
                }
                for match in matches:
                    if match not in excluded and not match.startswith('$') and not match.isdigit():
                        used_vars.add(match)
            else:
                # Nodeの場合は再帰的に抽出
                used_vars.update(_extract_vars_from_expression(node.condition))
        # then_bodyとelse_body内の変数も抽出
        if hasattr(node, 'then_body'):
            for body_node in node.then_body:
                used_vars.update(extract_variable_usages_from_node(body_node))
        if hasattr(node, 'else_body'):
            for body_node in node.else_body:
                used_vars.update(extract_variable_usages_from_node(body_node))

    return used_vars


def _extract_vars_from_expression(expression: Any) -> Set[str]:
    """式ノードから変数を抽出

    Args:
        expression: 式ノード

    Returns:
        使用される変数名のセット
    """
    used_vars = set()

    if expression is None:
        return used_vars

    # VariableNode: 変数参照
    expr_type = type(expression).__name__
    if expr_type == 'VariableNode':
        if hasattr(expression, 'name'):
            used_vars.add(expression.name)

    # CommandNode: コマンド呼び出し（引数から変数を抽出）
    elif expr_type == 'CommandNode':
        if hasattr(expression, 'arguments'):
            for arg in expression.arguments:
                used_vars.update(_extract_vars_from_expression(arg))

    # BinaryOpNode: 二項演算
    elif expr_type == 'BinaryOpNode':
        if hasattr(expression, 'left'):
            used_vars.update(_extract_vars_from_expression(expression.left))
        if hasattr(expression, 'right'):
            used_vars.update(_extract_vars_from_expression(expression.right))

    return used_vars


def remove_self_assignments(nodes: List[Any]) -> List[Any]:
    """同じ変数から同じ変数への代入式を削除

    Args:
        nodes: ノードリスト

    Returns:
        自己代入が削除されたノードリスト
    """
    if not nodes:
        return nodes

    filtered_nodes = []

    for i, node in enumerate(nodes):
        should_remove = False

        try:
            # ノードから生成されるコード文字列を取得
            code = node.generate()

            # パターン1: var = var または var = var[i] のような単純な代入
            # パターン2: array[i] = array[i] のような配列要素代入
            # パターン3: array[i] = array[index] で i == index の場合

            # 正規表現で自己代入を検出
            # Pattern 1: variable = variable
            import re
            simple_assignment = re.match(r'^(\w+)\s*=\s*(\w+)$', code)
            if simple_assignment:
                var1, var2 = simple_assignment.groups()
                if var1 == var2:
                    should_remove = True

            # Pattern 2: array[i] = array[i]
            if not should_remove:
                array_assignment = re.match(r'^(\w+)\[(\w+)\]\s*=\s*(\w+)\[(\w+)\]$', code)
                if array_assignment:
                    var1, idx1, var2, idx2 = array_assignment.groups()
                    if var1 == var2 and idx1 == idx2:
                        should_remove = True

        except Exception:
            # generate()が失敗した場合は残す
            pass

        # 自己代入でない場合は残す
        if not should_remove:
            filtered_nodes.append(node)

    return filtered_nodes


def remove_unused_variables(nodes: List[Any]) -> List[Any]:
    """未使用変数の定義ノードを削除（強化版：再帰的削除と自己代入の削除を含む）

    Args:
        nodes: ノードリスト

    Returns:
        未使用変数の定義が削除されたノードリスト
    """
    if not nodes:
        return nodes

    # 自己代入を先に削除
    nodes = remove_self_assignments(nodes)

    # 複数回のパスで未使用変数を削除（再帰的削除）
    # 変数を削除した後、その変数に依存していた他の変数も未使用になる可能性がある
    max_iterations = 10  # 無限ループ防止
    iteration = 0
    previous_count = len(nodes)

    while iteration < max_iterations:
        # 各ノードで定義される変数とその位置を記録
        variable_definitions: Dict[str, List[int]] = {}  # {変数名: [定義ノードのインデックスのリスト]}

        # 各ノードで使用される変数を収集
        used_variables: Set[str] = set()

        # 最初のパス: 変数の定義と使用を収集
        for i, node in enumerate(nodes):
            # このノードで定義される変数を取得
            defined_vars = extract_variable_definitions_from_node(node)
            for var in defined_vars:
                if var not in variable_definitions:
                    variable_definitions[var] = []
                variable_definitions[var].append(i)

            # このノードで使用される変数を取得
            used_vars = extract_variable_usages_from_node(node)
            used_variables.update(used_vars)

        # 未使用変数の定義ノードのインデックスを特定
        unused_node_indices = set()
        for var_name, def_indices in variable_definitions.items():
            if var_name not in used_variables:
                # grid_sizeが含まれる変数は削除しない
                if 'grid_size' in var_name.lower():
                    continue
                # この変数は定義されているが使用されていない
                # すべての定義を削除（再定義も含む）
                unused_node_indices.update(def_indices)

        # 削除するノードがない場合は終了
        if not unused_node_indices:
            break

        # 未使用変数の定義ノードを削除
        filtered_nodes = []
        removed_vars = []
        for i, node in enumerate(nodes):
            if i not in unused_node_indices:
                filtered_nodes.append(node)
            else:
                defined_vars = extract_variable_definitions_from_node(node)
                if defined_vars:
                    removed_vars.append(defined_vars[0])

        nodes = filtered_nodes
        iteration += 1

        # 削除されたノード数が変わらなかった場合は終了
        if len(nodes) == previous_count:
            break
        previous_count = len(nodes)

    # 空のFOR/IFループを削除
    nodes = remove_empty_control_structures(nodes)

    return nodes


def remove_empty_control_structures(nodes: List[Any]) -> List[Any]:
    """空のFOR/IFループ（FOR/IFとENDの間に有効なコマンドがない）を削除

    FOR i LEN(objects) DO
    END

    や

    IF .....
    END

    のように、FOR/IFとENDが連続している（または間に有効なコマンドがない）場合、
    両方を削除する。ネストされた構造も処理される。

    Args:
        nodes: ノードリスト

    Returns:
        空のFOR/IFループが削除されたノードリスト
    """
    if not nodes:
        return nodes

    # 削除を繰り返し実行（ネストされた空ループも処理されるまで）
    max_iterations = 10  # 無限ループ防止
    iteration = 0

    while iteration < max_iterations:
        # 削除するノードのインデックスを収集
        nodes_to_remove = set()

        # スタックを使用してネストされたFOR/IFを追跡
        # 各要素は (start_index, type) のタプル
        # typeは 'for' または 'if'
        stack = []

        i = 0
        found_empty = False

        while i < len(nodes):
            node = nodes[i]

            if is_for_node(node) or is_if_node(node):
                # FOR/IFノードを見つけたらスタックに追加
                node_type = 'for' if is_for_node(node) else 'if'
                stack.append((i, node_type))
                i += 1

            elif is_end_node(node):
                # ENDノードを見つけたら、対応するFOR/IFを探す
                if stack:
                    start_index, control_type = stack.pop()

                    # FOR/IFとENDの間のノードをチェック
                    # 間に有効なコマンドノード（FOR/IF/END以外）があるか確認
                    has_valid_command = False
                    for j in range(start_index + 1, i):
                        mid_node = nodes[j]
                        # FOR/IF/END以外のノードがあれば有効なコマンド
                        # ただし、既に削除対象となっているノードはスキップ
                        if j in nodes_to_remove:
                            continue
                        if not is_for_node(mid_node) and not is_if_node(mid_node) and not is_end_node(mid_node):
                            has_valid_command = True
                            break

                    # 有効なコマンドがない場合、FOR/IFとENDを削除対象に追加
                    if not has_valid_command:
                        nodes_to_remove.add(start_index)  # FOR/IFノード
                        nodes_to_remove.add(i)  # ENDノード
                        found_empty = True
                i += 1

            else:
                # 通常のノードはスキップ
                i += 1

        # 削除対象がない場合は終了
        if not found_empty:
            break

        # 削除対象のノードを除外
        filtered_nodes = []
        for i, node in enumerate(nodes):
            if i not in nodes_to_remove:
                filtered_nodes.append(node)

        nodes = filtered_nodes
        iteration += 1

    return nodes
