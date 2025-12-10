"""
構造生成関連の機能
"""
import random
from typing import List, Dict, Optional, Any
from ..metadata.types import SemanticType, ReturnType, TypeSystem, TypeInfo
from ..metadata.constants import GRID_SIZE_PRESERVATION_PROB
from ..metadata.commands import COMMAND_METADATA
from .nodes import (
    Node, RenderNode, AssignmentNode, LiteralNode, VariableNode, CommandNode,
    IfStartNode, ForStartNode, ForStartWithCountNode, ForStartWithConstantNode,
    ForStartWithMatchPairsNode, EndNode,
)
from .program_context import ProgramContext
from .node_utilities import NodeUtilities
from . import OBJECT_ARRAY_TYPE


class StructureGenerators:
    """構造生成関連の機能を提供するクラス"""

    def __init__(self):
        self.utilities = NodeUtilities()
        self.node_generators = None  # NodeGeneratorsへの参照（後で設定される）

    def generate_goal_node(self, context: ProgramContext) -> RenderNode:
        """ゴールノード（RENDER_GRID）を生成"""
        from .nodes import RenderNode

        final_array = "objects"

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(final_array, "argument", OBJECT_ARRAY_TYPE)

        # 背景色の決定（AGI2学習データの多様性向上のため0.80 → 0.675に変更）
        if random.random() < 0.675:
            bg_color = CommandNode('GET_BACKGROUND_COLOR', [], context.to_dict())
        else:
            bg_color = LiteralNode(random.randint(0, 9), context.to_dict())

        # グリッドサイズの決定（55%の確率でgrid_size変数、45%で乱数/混合）- AGI2学習データの多様性向上のため確率を調整（80% → 55%）
        if random.random() < GRID_SIZE_PRESERVATION_PROB:
            width = VariableNode("grid_size_x", context.to_dict())
            height = VariableNode("grid_size_y", context.to_dict())
        else:
            # グリッドサイズに基づいた範囲を使用
            context_dict = context.to_dict()
            grid_width = context_dict.get('output_grid_width') or context_dict.get('input_grid_width') or context_dict.get('grid_width')
            grid_height = context_dict.get('output_grid_height') or context_dict.get('input_grid_height') or context_dict.get('grid_height')
            # SIZE型のデフォルト範囲を取得（最小1、最大グリッドサイズ）
            size_range = TypeSystem.get_default_range(SemanticType.SIZE, grid_width, grid_height)
            if size_range:
                min_size, max_size = size_range
                min_size = max(1, min_size)  # 最小1を保証
            else:
                min_size, max_size = 1, 30  # フォールバック

            # 1からmax_sizeの乱数を生成
            if random.random() < 0.3:
                # 完全に乱数（30%）
                if random.random() < 0.7:
                    # 正方形（70%の確率）
                    size = random.randint(1, 30)
                    width = LiteralNode(size, context.to_dict())
                    height = LiteralNode(size, context.to_dict())
                else:
                    # 異なるサイズ（30%の確率）
                    width = LiteralNode(random.randint(min_size, max_size), context.to_dict())
                    height = LiteralNode(random.randint(min_size, max_size), context.to_dict())
            else:
                # 片方だけgrid_size変数（70%）
                if random.random() < 0.5:
                    width = VariableNode("grid_size_x", context.to_dict())
                    height = LiteralNode(random.randint(min_size, max_size), context.to_dict())
                else:
                    width = LiteralNode(random.randint(min_size, max_size), context.to_dict())
                    height = VariableNode("grid_size_y", context.to_dict())

        return RenderNode(
            array=final_array,
            bg_color=bg_color,
            width=width,
            height=height,
            context=context.to_dict()
        )

    def generate_if_branch(self, context: ProgramContext) -> Node:
        """IF分岐開始ノードを生成"""
        # IFネスト深度を1増加
        context.enter_if_nesting()

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['in_if_branch'] = True

        return IfStartNode(context=ctx_dict)

    def generate_for_loop(self, current_array: str, context: ProgramContext) -> Node:
        """FORループ開始ノードを生成"""
        # FORネスト深度を1増加
        context.enter_for_nesting()

        # ループ変数を選択
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        loop_var_index = min(context.get_for_nesting_depth() - 1, len(loop_vars) - 1)
        loop_var = loop_vars[loop_var_index]

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['in_for_loop'] = True
        ctx_dict['current_array'] = current_array
        ctx_dict['loop_var'] = loop_var

        # FORループ内で配列変数を定義
        from ..metadata.types import TypeInfo
        context.variable_manager.register_variable_usage(
            current_array, "argument",
            TypeInfo.create_from_semantic_type(SemanticType.COUNT, is_array=False)
        )

        return ForStartNode(
            loop_var=loop_var,
            array=current_array,
            context=ctx_dict
        )

    def generate_for_loop_with_count(self, count_variable: str, context: ProgramContext) -> Node:
        """カウント変数を使用したFORループ開始ノードを生成"""
        # FORループ深度を1増加
        context.enter_for_nesting()

        # ループ変数を決定
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        loop_idx = len([arr for arr in context.for_arrays if arr is not None])
        loop_var = loop_vars[min(loop_idx, len(loop_vars) - 1)]

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['in_for_loop'] = True
        ctx_dict['loop_var'] = loop_var
        ctx_dict['count_variable'] = count_variable

        # カウント変数が使用されたことを記録
        from ..metadata.types import TypeInfo
        context.variable_manager.register_variable_usage(
            count_variable, "argument",
            TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False)
        )

        return ForStartWithCountNode(
            loop_var=loop_var,
            count_variable=count_variable,
            context=ctx_dict
        )

    def generate_for_loop_with_constant(self, constant_value: int, context: ProgramContext) -> Node:
        """定数値を使用したFORループ開始ノードを生成"""
        # FORループ深度を1増加
        context.enter_for_nesting()

        # ループ変数を決定
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        loop_idx = len([arr for arr in context.for_arrays if arr is not None])
        loop_var = loop_vars[min(loop_idx, len(loop_vars) - 1)]

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['in_for_loop'] = True
        ctx_dict['loop_var'] = loop_var
        ctx_dict['constant_value'] = constant_value

        return ForStartWithConstantNode(
            loop_var=loop_var,
            constant_value=constant_value,
            context=ctx_dict
        )

    def generate_for_loop_with_match_pairs(self, match_pairs_array: str, context: ProgramContext) -> Node:
        """MATCH_PAIRS配列を使用したFORループ開始ノードを生成"""
        # FORループ深度を1増加
        context.enter_for_nesting()

        # ループ変数を決定
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        loop_idx = len([arr for arr in context.for_arrays if arr is not None])
        loop_var = loop_vars[min(loop_idx, len(loop_vars) - 1)]

        # MATCH_PAIRS配列をfor_arraysに追加
        context.for_arrays.append(match_pairs_array)

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['in_for_loop'] = True
        ctx_dict['current_array'] = match_pairs_array
        ctx_dict['loop_var'] = loop_var
        ctx_dict['match_pairs_array'] = match_pairs_array

        # MATCH_PAIRS配列を使用済みとして登録
        from ..metadata.types import TypeInfo
        context.variable_manager.register_variable_usage(
            match_pairs_array, "argument",
            TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True)
        )

        return ForStartWithMatchPairsNode(
            loop_var=loop_var,
            match_pairs_array=match_pairs_array,
            context=ctx_dict
        )

    def generate_end(self, context: ProgramContext) -> Node:
        """ENDするだけのノードを生成"""
        # コンテキスト辞書を生成
        ctx_dict = context.to_dict()

        return EndNode(context=ctx_dict)

    def generate_variable_definition_node_with_literal(self, var_name: str, semantic_type: SemanticType, literal_value: Any, context: ProgramContext) -> Node:
        """リテラル値を使用した変数定義ノードを生成"""
        # コンテキスト辞書を生成
        ctx_dict = context.to_dict()

        # LiteralNodeを作成してexpressionに渡す
        literal_node = LiteralNode(literal_value, ctx_dict)

        return AssignmentNode(
            variable=var_name,
            expression=literal_node,
            context=ctx_dict
        )
