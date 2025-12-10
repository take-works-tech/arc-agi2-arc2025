"""
ノード生成関連のユーティリティ関数
"""
from typing import Optional, Tuple, List
from ..metadata.types import SemanticType, ReturnType, TypeInfo
from .nodes import Node, LiteralNode, VariableNode, CommandNode, PlaceholderNode, AssignmentNode, ArrayAssignmentNode, FilterNode, IfBranchNode, RenderNode, BinaryOpNode
from .program_context import ProgramContext


class NodeUtilities:
    """ノード生成関連のユーティリティ関数を提供するクラス"""

    @staticmethod
    def get_context_dict(context: ProgramContext, ctx_dict: dict) -> dict:
        """コンテキスト辞書を取得"""
        if hasattr(context, 'to_dict'):
            return context.to_dict()
        return ctx_dict if isinstance(ctx_dict, dict) else {}

    @staticmethod
    def get_grid_size(context_dict: dict) -> Tuple[int, int]:
        """グリッドサイズを取得"""
        grid_width = context_dict.get('output_grid_width') or context_dict.get('input_grid_width') or context_dict.get('grid_width')
        grid_height = context_dict.get('output_grid_height') or context_dict.get('input_grid_height') or context_dict.get('grid_height')
        return grid_width, grid_height

    @staticmethod
    def is_color_type(type_info: Optional[TypeInfo], arg_schema) -> bool:
        """COLOR型かどうかを判定"""
        if type_info and type_info.semantic_type == SemanticType.COLOR:
            return True
        if hasattr(arg_schema, 'type_info') and arg_schema.type_info and arg_schema.type_info.semantic_type == SemanticType.COLOR:
            return True
        return False

    @staticmethod
    def is_bool_type(type_info: Optional[TypeInfo], arg_schema) -> bool:
        """BOOL型かどうかを判定"""
        if type_info and type_info.semantic_type == SemanticType.BOOL:
            return True
        if hasattr(arg_schema, 'type_info') and arg_schema.type_info:
            if hasattr(arg_schema.type_info, 'semantic_type') and arg_schema.type_info.semantic_type == SemanticType.BOOL:
                return True
            if hasattr(arg_schema.type_info, 'return_type') and arg_schema.type_info.return_type == ReturnType.BOOL:
                return True
        return False

    @staticmethod
    def check_bool_literal_prob(arg_schema) -> None:
        """BOOL型でliteral_prob=0.0の場合は例外を発生"""
        if hasattr(arg_schema, 'literal_prob') and arg_schema.literal_prob == 0.0:
            raise ValueError("BOOL型でliteral_prob=0.0のため、リテラル値は生成できません")

    @staticmethod
    def clamp_color_range(range_min: int, range_max: int) -> Tuple[int, int]:
        """COLOR型の範囲を0-9に制限"""
        range_min = max(0, range_min)
        range_max = min(9, range_max)
        if range_min > range_max:
            return 0, 9
        return range_min, range_max

    @staticmethod
    def is_special_argument(arg_schema) -> bool:
        """特殊な引数かどうかを判定"""
        from ..metadata.argument_schema import CONDITION_ARG, MATCH_PAIRS_CONDITION_ARG, KEY_EXPR_ARG
        return arg_schema in [CONDITION_ARG, MATCH_PAIRS_CONDITION_ARG, KEY_EXPR_ARG]

    @staticmethod
    def is_proportional_operation(cmd_name: str) -> bool:
        """比例演算かどうかを判定"""
        return cmd_name in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']

    @staticmethod
    def is_constant_only_operation(cmd_name: str, args: List[Node]) -> bool:
        """定数同士の演算かどうかをチェック"""
        # 算術演算と比例演算のコマンドをチェック
        arithmetic_ops = ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']
        proportional_ops = ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']
        logical_ops = ['AND', 'OR']  # 論理演算も追加

        if cmd_name in arithmetic_ops + proportional_ops + logical_ops:
            # すべての引数が定数（LiteralNode）かチェック
            for arg in args:
                if not isinstance(arg, LiteralNode):
                    return False
            return True

        return False

    @staticmethod
    def get_placeholder_variables_in_node(node: Node) -> set:
        """ノード内で使用されているプレースホルダー変数（$obj, $obj1, $obj2など）を取得"""
        placeholders = set()

        if isinstance(node, PlaceholderNode):
            # PlaceholderNodeの場合はplaceholder属性を取得
            placeholders.add(node.placeholder)
        elif isinstance(node, VariableNode):
            # プレースホルダー変数（$で始まる）のみを取得
            if node.name.startswith('$'):
                placeholders.add(node.name)
        elif isinstance(node, CommandNode):
            # コマンドの引数内のプレースホルダーを再帰的に取得
            for arg in node.arguments:
                placeholders.update(NodeUtilities.get_placeholder_variables_in_node(arg))
        elif isinstance(node, (AssignmentNode, ArrayAssignmentNode)):
            placeholders.update(NodeUtilities.get_placeholder_variables_in_node(node.expression))
        elif isinstance(node, FilterNode):
            # FilterNodeの条件文字列からプレースホルダーを抽出
            import re
            condition_placeholders = re.findall(r'\$obj\d*', node.condition)
            placeholders.update(condition_placeholders)

        return placeholders

    @staticmethod
    def get_variables_used_in_node(node: Node) -> set:
        """ノード内で使用されている変数名を取得"""
        variables = set()

        if isinstance(node, VariableNode):
            variables.add(node.name)
        elif isinstance(node, CommandNode):
            # コマンドの引数内の変数を再帰的に取得
            for arg in node.arguments:
                variables.update(NodeUtilities.get_variables_used_in_node(arg))
        elif isinstance(node, (AssignmentNode, ArrayAssignmentNode)):
            # 代入ノードの式内の変数を取得
            if node.expression:
                variables.update(NodeUtilities.get_variables_used_in_node(node.expression))
        elif isinstance(node, FilterNode):
            variables.add(node.source_array)
            variables.add(node.target_array)
            # 条件文字列から変数を抽出（改善版）
            import re
            condition_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', node.condition)

            # 除外すべき予約語・定数・コマンド名
            reserved_words = {
                'True', 'False', 'start', 'end', 'first', 'last',
                'X', 'Y', '-X', '-Y',
                'AND', 'OR', 'EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS',
                'MERGE', 'CONCAT', 'FILTER', 'EXCLUDE', 'SORT_BY', 'MATCH_PAIRS'
            }

            for var in condition_vars:
                if (not var.startswith('$') and
                    not var.startswith('GET_') and
                    not var.startswith('IS_') and
                    var not in reserved_words):
                    variables.add(var)
        elif isinstance(node, IfBranchNode):
            for then_node in node.then_body:
                variables.update(NodeUtilities.get_variables_used_in_node(then_node))
            for else_node in node.else_body:
                variables.update(NodeUtilities.get_variables_used_in_node(else_node))
        elif isinstance(node, RenderNode):
            variables.add(node.array)
        elif isinstance(node, BinaryOpNode):
            variables.update(NodeUtilities.get_variables_used_in_node(node.left))
            variables.update(NodeUtilities.get_variables_used_in_node(node.right))

        return variables

    @staticmethod
    def is_self_reference(var_name: str, node: Node) -> bool:
        """ノードが変数名を自己参照しているかどうかをチェック"""
        from .nodes import BinaryOpNode
        if isinstance(node, VariableNode):
            # 変数ノードの場合、変数名が一致するかチェック
            return node.name == var_name
        elif isinstance(node, CommandNode):
            # コマンドノードの場合、引数を再帰的にチェック
            for arg in node.arguments:
                if NodeUtilities.is_self_reference(var_name, arg):
                    return True
        elif isinstance(node, BinaryOpNode):
            # 二項演算ノードの場合、左右の引数をチェック
            if NodeUtilities.is_self_reference(var_name, node.left):
                return True
            if NodeUtilities.is_self_reference(var_name, node.right):
                return True
        elif isinstance(node, (AssignmentNode, ArrayAssignmentNode)):
            # 代入ノードの式内で自己参照をチェック
            if node.expression:
                return NodeUtilities.is_self_reference(var_name, node.expression)
        elif isinstance(node, LiteralNode):
            # リテラルノードの場合は自己参照ではない
            return False
        return False
