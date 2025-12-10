"""
ノード操作のヘルパー関数
"""
from typing import List, Dict, Optional, Set, Any, Tuple
from .nodes import Node
from .program_context import ProgramContext

# 型定義
NodeWithVarInfo = Tuple[Node, Optional[str], bool]  # (node, variable, is_first_definition)
NodeListWithVarInfo = List[NodeWithVarInfo]


def get_defined_variables_from_node_list(node_list: NodeListWithVarInfo) -> Set[str]:
    """ノードリストから定義された変数を取得"""
    return {var_info[1] for var_info in node_list if var_info[1]}


def get_first_defined_variables(node_list: NodeListWithVarInfo) -> Set[str]:
    """初回定義された変数を取得"""
    return {var_info[1] for var_info in node_list if var_info[2] and var_info[1]}


def get_variable_info(node_list: NodeListWithVarInfo, variable: str) -> Dict[str, Any]:
    """特定の変数の情報を取得"""
    for node, var, is_first in node_list:
        if var == variable:
            return {'node': node, 'is_first': is_first}
    return {}


def is_first_definition(variable_tracker, var_name: str) -> bool:
    """変数が初回定義かどうかを判定"""
    if var_name not in variable_tracker._variable_tracker:
        return True

    var_info = variable_tracker._variable_tracker[var_name]
    return not var_info.get('is_defined', False)


def extend_node_list_with_var_info(target: NodeListWithVarInfo, source: NodeListWithVarInfo) -> None:
    """変数情報付きノードリストを拡張"""
    target.extend(source)


def get_nodes_from_var_info_list(node_list: NodeListWithVarInfo) -> List[Node]:
    """変数情報付きノードリストからノードのみを取得"""
    return [node_info[0] for node_info in node_list]


def add_node_to_list(node_list: NodeListWithVarInfo, node: Node, variable: Optional[str] = None, context: Optional[ProgramContext] = None, is_first_override: Optional[bool] = None) -> None:
    """ノードを変数情報付きリストに追加（変数情報を自動取得）

    Args:
        node_list: ノードリスト
        node: 追加するノード
        variable: 変数名（オプション）
        context: プログラムコンテキスト（オプション）
        is_first_override: 初回定義フラグの上書き値（指定された場合はこれを使用）
    """

    # 文字列が渡された場合は無視（デバッグ用）
    if isinstance(node, str):
        return

    # 初回定義かどうかを判定
    if is_first_override is not None:
        # 上書き値が指定されている場合はそれを使用
        is_first = is_first_override
    else:
        # 上書き値が指定されていない場合は、変数が定義される前に判定
        is_first = False
        if variable is not None and context is not None:
            # ノード生成メソッド内でdefine_variableが呼ばれる前に判定するため、
            # 変数が存在しない、または定義されていない場合は初回定義とみなす
            if variable not in context.variable_manager.tracker._variable_tracker:
                is_first = True
            else:
                var_info = context.variable_manager.tracker._variable_tracker[variable]
                is_first = not var_info.get('is_defined', False)

    node_list.append((node, variable, is_first))

    # ノード履歴を更新（UnifiedProgramGeneratorのインスタンスがある場合）
    if context is not None and hasattr(context, '_generator_instance'):
        context._generator_instance._update_previous_nodes(context, node)


# NodeWithVarInfoの各要素を取得するヘルパー関数
def get_node_from_var_info(node_info: NodeWithVarInfo) -> Node:
    """NodeWithVarInfoからNodeを取得"""
    return node_info[0]


def get_variable_from_var_info(node_info: NodeWithVarInfo) -> str:
    """NodeWithVarInfoから変数名を取得"""
    return node_info[1]


def get_is_first_definition_from_var_info(node_info: NodeWithVarInfo) -> bool:
    """NodeWithVarInfoから初回定義フラグを取得"""
    return node_info[2]


def has_variable_info(node_info: NodeWithVarInfo) -> bool:
    """NodeWithVarInfoに変数情報があるかチェック"""
    return node_info[1] is not None


def get_all_variables_from_list(node_list: NodeListWithVarInfo) -> Set[str]:
    """NodeListWithVarInfoからすべての変数名を取得（None除く）"""
    return {node_info[1] for node_info in node_list if node_info[1] is not None}
