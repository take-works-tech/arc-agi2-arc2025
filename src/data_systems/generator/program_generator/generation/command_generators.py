"""
コマンド生成関連の機能
"""
from typing import List, Dict, Optional
from ..metadata.types import SemanticType, TypeInfo
from .nodes import (
    Node, FilterNode, MatchPairsNode,
    ExcludeNode, ConcatNode, AppendNode, MergeNode, EmptyArrayNode,
    CommandNode, LiteralNode, VariableNode,
)
from .program_context import ProgramContext
from .node_utilities import NodeUtilities
from . import OBJECT_ARRAY_TYPE, OBJECT_TYPE


class CommandGenerators:
    """コマンド生成関連の機能を提供するクラス"""

    def __init__(self):
        self.utilities = NodeUtilities()

    def generate_exclude_node(self, source_array: str, target_array: str, targets_array: str, context: ProgramContext) -> ExcludeNode:
        """EXCLUDEノードを生成"""
        # 配列が使用されたことを記録
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayとtargets_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)
        context.variable_manager.register_variable_usage(targets_array, "argument", OBJECT_ARRAY_TYPE)

        return ExcludeNode(
            source_array=source_array,
            target_array=target_array,
            targets_array=targets_array,
            context=context.to_dict()
        )

    def generate_concat_node(self, array1: str, array2: str, target_array: str, context: ProgramContext) -> ConcatNode:
        """CONCATノードを生成"""
        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(array1, "argument", OBJECT_ARRAY_TYPE)
        context.variable_manager.register_variable_usage(array2, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return ConcatNode(
            array1=array1,
            array2=array2,
            target_array=target_array,
            context=context.to_dict()
        )

    def generate_append_node(self, array: str, obj: str, target_array: str, context: ProgramContext) -> AppendNode:
        """APPENDノードを生成"""
        # 配列とオブジェクトが使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(array, "argument", OBJECT_ARRAY_TYPE)
        context.variable_manager.register_variable_usage(obj, "argument", OBJECT_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return AppendNode(
            array=array,
            obj=obj,
            target_array=target_array,
            context=context.to_dict()
        )

    def generate_merge_node(self, objects_array: str, target_obj: str, context: ProgramContext) -> MergeNode:
        """MERGEノードを生成"""
        # オブジェクト配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(objects_array, "argument", OBJECT_ARRAY_TYPE)
        # target_objは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_obj, SemanticType.OBJECT, is_array=False)
        # スコープ情報を記録
        context.add_scope_variable(target_obj)

        return MergeNode(
            objects_array=objects_array,
            target_obj=target_obj,
            context=context.to_dict()
        )

    def generate_empty_array_node(self, array_name: str, context: ProgramContext) -> EmptyArrayNode:
        """空のオブジェクト配列定義ノードを生成"""
        # 配列変数を定義済みとして登録
        context.variable_manager.define_variable(array_name, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(array_name)

        return EmptyArrayNode(
            array_name=array_name,
            context=context.to_dict()
        )
