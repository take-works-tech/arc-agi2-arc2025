"""
ノード生成関連の機能
"""
from __future__ import annotations

import random
from typing import List, Dict, Optional, Any, Tuple
from ..metadata.types import SemanticType, ReturnType, TypeSystem, TypeInfo
from ..metadata.constants import (
    COMMAND_CATEGORIES, COMPLEXITY_LEVELS, ENABLE_COMPLEXITY_CONTROL,
    COMMAND_NESTING_DEPTH_BY_COMPLEXITY
)

# 共通の型情報定義をインポート
from . import OBJECT_ARRAY_TYPE, OBJECT_TYPE

from ..metadata.variable_manager import variable_manager
from ..metadata.constants import (
    GRID_SIZE_PRESERVATION_PROB,
)
from ..metadata.commands import COMMAND_METADATA, CommandMetadata
from .nodes import (
    Node, StatementNode, ExpressionNode,
    InitializationNode, AssignmentNode, ArrayAssignmentNode,
    FilterNode, IfBranchNode, RenderNode,
    LiteralNode, VariableNode, CommandNode, BinaryOpNode,
    PlaceholderNode,
    ExcludeNode, ConcatNode, AppendNode, MergeNode, EmptyArrayNode,
    SingleObjectArrayNode, SplitConnectedNode, ExtractShapeNode, ExtendPatternNode, ArrangeGridNode,
    ObjectAccessNode, MatchPairsNode,
)
from .program_context import ProgramContext
from .node_utilities import NodeUtilities
from .variable_generators import VariableGenerators
from .command_generators import CommandGenerators
from .structure_generators import StructureGenerators


class NodeGenerators:
    """ノード生成関連の機能を提供するファサードクラス

    このクラスは、以下の専門クラスへの委譲を行います：
    - NodeUtilities: ユーティリティメソッド
    - ArgumentGenerator: 引数生成メソッド（新しいルールに基づく）
    - VariableGenerators: 変数生成メソッド
    - CommandGenerators: コマンド生成メソッド
    - StructureGenerators: 構造生成メソッド

    複雑な依存関係を持つメソッドは、このクラス内に残しています。
    """

    # ============================================================
    # セクション1: 初期化・基本設定
    # ============================================================

    def __init__(self):
        from ..metadata.variable_managers import VariableManagers
        self.variable_managers = VariableManagers()
        self.utilities = NodeUtilities()
        self.variable_generators = VariableGenerators()
        self.command_generators = CommandGenerators()
        self.structure_generators = StructureGenerators()
        # StructureGeneratorsにNodeGeneratorsへの参照を渡す（_generate_value_argumentを使用するため）
        self.structure_generators.node_generators = self
        # ArgumentGeneratorを初期化（引数生成関連の機能を提供）
        from .argument_generator import ArgumentGenerator
        self.argument_generators = ArgumentGenerator(self)

    # ============================================================
    # セクション2: ノード生成関連（基本ノード）
    # ============================================================

    def _generate_goal_node(self, context: ProgramContext) -> RenderNode:
        """ゴールノード（RENDER_GRID）を生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_goal_node(context)

    def _generate_grid_size_definition_nodes(self, context: ProgramContext, uses_grid_size_x: bool = True, uses_grid_size_y: bool = True) -> List[Node]:
        """grid_size関連の定義ノード群を生成（必要な変数のみ）"""
        from .nodes import AssignmentNode, CommandNode, VariableNode, LiteralNode

        nodes = []

        # grid_size_xまたはgrid_size_yが使用される場合のみgrid_sizeを定義
        if uses_grid_size_x or uses_grid_size_y:
            # 1. grid_size = GET_INPUT_GRID_SIZE()
            command_node = CommandNode('GET_INPUT_GRID_SIZE', [], context.to_dict())
            grid_size_node = AssignmentNode(
                variable="grid_size",
                expression=command_node,
                context=context.to_dict()
            )
            nodes.append(grid_size_node)

        # 2. grid_size_x = grid_size[0] (必要な場合のみ生成)
        if uses_grid_size_x:
            expression_choice = random.random()

            if expression_choice < 0.05:
                # スケール式を生成（MULTIPLYまたはDIVIDEを使用）- AGI2学習データの多様性向上のため確率を上げる（2% → 5%）
                scale_choice = random.random()
                if scale_choice < 0.6:
                    # 拡大（60%）- AGI2学習データの多様性向上のため4倍も追加（2, 3 → 2, 3, 4）
                    scale_factor = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1], k=1)[0]
                    grid_size_x_expression = CommandNode('MULTIPLY', [
                        VariableNode("grid_size[0]", context.to_dict()),
                        LiteralNode(scale_factor, context.to_dict())
                    ], context.to_dict())
                else:
                    # 縮小（40%）- AGI2学習データの多様性向上のため4倍も追加（2, 3 → 2, 3, 4）
                    scale_divisor = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1], k=1)[0]
                    grid_size_x_expression = CommandNode('DIVIDE', [
                        VariableNode("grid_size[0]", context.to_dict()),
                        LiteralNode(scale_divisor, context.to_dict())
                    ], context.to_dict())
            elif expression_choice < 0.30:
                # 既存変数から代入元を生成（25%の確率）- AGI2学習データの多様性向上のため確率を上げる（8% → 25%）
                try:
                    from ..metadata.argument_schema import create_argument_schema_with_naming_system

                    size_type = TypeInfo(
                        semantic_type=SemanticType.SIZE,
                        is_array=False,
                        return_type=ReturnType.SIZE
                    )

                    arg_schema = create_argument_schema_with_naming_system(
                        size_type,
                        literal_prob=0.3,
                        variable_prob=0.2,
                        nested_prob=0.5
                    )
                    try:
                        # 複雑度に応じたネスト深度を使用
                        ctx_dict = context.to_dict()
                        grid_size_x_expression = self._generate_argument_node(
                            arg_schema, context, ctx_dict, arg_schema.type_info, no_new_vars_mode=True, max_nesting_depth=context.get_command_nesting_depth()
                        )
                    finally:
                        context.variable_manager.clear_unavailable_variables()
                except Exception:
                    # フォールバック: 直接代入
                    grid_size_x_expression = VariableNode("grid_size[0]", context.to_dict())
            else:
                # 直接代入（70%）- AGI2学習データの多様性向上のため確率を下げる（90% → 70%）
                grid_size_x_expression = VariableNode("grid_size[0]", context.to_dict())

            # expressionがNoneの場合はフォールバック
            if grid_size_x_expression is None:
                grid_size_x_expression = VariableNode("grid_size[0]", context.to_dict())

            grid_size_x_node = AssignmentNode(
                variable="grid_size_x",
                expression=grid_size_x_expression,
                context=context.to_dict()
            )
            nodes.append(grid_size_x_node)

        # 3. grid_size_y = grid_size[1] (必要な場合のみ生成)
        if uses_grid_size_y:
            expression_choice = random.random()

            if expression_choice < 0.05:
                # スケール式を生成（MULTIPLYまたはDIVIDEを使用）- AGI2学習データの多様性向上のため確率を上げる（2% → 5%）
                scale_choice = random.random()
                if scale_choice < 0.6:
                    # 拡大（60%）- AGI2学習データの多様性向上のため4倍も追加（2, 3 → 2, 3, 4）
                    scale_factor = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1], k=1)[0]
                    grid_size_y_expression = CommandNode('MULTIPLY', [
                        VariableNode("grid_size[1]", context.to_dict()),
                        LiteralNode(scale_factor, context.to_dict())
                    ], context.to_dict())
                else:
                    # 縮小（40%）- AGI2学習データの多様性向上のため4倍も追加（2, 3 → 2, 3, 4）
                    scale_divisor = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1], k=1)[0]
                    grid_size_y_expression = CommandNode('DIVIDE', [
                        VariableNode("grid_size[1]", context.to_dict()),
                        LiteralNode(scale_divisor, context.to_dict())
                    ], context.to_dict())
            elif expression_choice < 0.30:
                # 既存変数から代入元を生成（25%の確率）- AGI2学習データの多様性向上のため確率を上げる（8% → 25%）
                try:
                    from ..metadata.argument_schema import create_argument_schema_with_naming_system

                    size_type = TypeInfo(
                        semantic_type=SemanticType.SIZE,
                        is_array=False,
                        return_type=ReturnType.SIZE
                    )

                    arg_schema = create_argument_schema_with_naming_system(
                        size_type,
                        literal_prob=0.3,
                        variable_prob=0.2,
                        nested_prob=0.5
                    )

                    # no_new_vars_modeで既存変数を使用（grid_size_xは利用可能）
                    # 複雑度に応じたネスト深度を使用
                    ctx_dict = context.to_dict()
                    grid_size_y_expression = self._generate_argument_node(
                        arg_schema, context, ctx_dict, arg_schema.type_info, no_new_vars_mode=True, max_nesting_depth=context.get_command_nesting_depth()
                    )
                except Exception:
                    # フォールバック: 直接代入
                    grid_size_y_expression = VariableNode("grid_size[1]", context.to_dict())
            else:
                # 直接代入（70%）- AGI2学習データの多様性向上のため確率を下げる（90% → 70%）
                grid_size_y_expression = VariableNode("grid_size[1]", context.to_dict())

            # expressionがNoneの場合はフォールバック
            if grid_size_y_expression is None:
                grid_size_y_expression = VariableNode("grid_size[1]", context.to_dict())

            grid_size_y_node = AssignmentNode(
                variable="grid_size_y",
                expression=grid_size_y_expression,
                context=context.to_dict()
            )
            nodes.append(grid_size_y_node)

        # 変数を定義済みとして登録（使用される変数のみ）
        if uses_grid_size_x or uses_grid_size_y:
            grid_size_type = TypeInfo(
                semantic_type=SemanticType.COUNT,
                is_array=True,
                return_type=ReturnType.INT
            )

        grid_size_single_type = TypeInfo(
            semantic_type=SemanticType.COUNT,
            is_array=False,
            return_type=ReturnType.INT
        )

        return nodes

    def _generate_create_objects(self, var_name: str, context: ProgramContext, no_new_vars_mode: bool = False) -> Node:
        """オブジェクト生成ノードを生成（限定されたコマンド選択肢で変数定義ノードを生成）"""
        # 限定されたオブジェクト生成コマンドの選択肢で変数定義ノードを生成（MERGEを除外）
        limited_commands = ['CREATE_LINE', 'CREATE_RECT']
        return self._generate_variable_definition_node(
            var_name,
            context,
            limited_commands,
            no_new_vars_mode=no_new_vars_mode,
            provided_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)
        )

    # ============================================================
    # セクション3: 引数生成関連
    # ============================================================

    def _generate_argument_node(self, arg_schema, context: ProgramContext, ctx_dict: Dict, target_type_info: TypeInfo = None, no_new_vars_mode: bool = False, nesting_depth: int = 0, max_nesting_depth: int = None) -> Node:
        """引数ノードを生成（統一された確率的選択システムを使用）

        注意: このメソッドで使用されるnesting_depthは「引数ネスト深度」（コマンド引数内のネスト）であり、
        「スコープネスト深度」（FOR/IFの構造的ネスト）とは別物です。
        - 引数ネスト深度: FILTER(objects, EQUAL(GET_COLOR($obj), 3)) のような引数内のネスト
        - スコープネスト深度: FOR ... DO FOR ... DO ... END END のような構造的ネスト
        """
        # 新しいルールを使用
        from .argument_generator import ArgumentGenerator
        new_generator = ArgumentGenerator(self)

        # 引数のインデックスを取得（contextから）
        arg_index = getattr(context, 'current_argument_index', None)
        if arg_index is None:
            # current_arithmetic_or_proportional_arg_indexから取得を試みる
            arg_index = getattr(context, 'current_arithmetic_or_proportional_arg_index', None)
        parent_command = getattr(context, 'current_command', None)

        # 最大ネスト深度を取得
        if max_nesting_depth is None:
            max_nesting_depth = context.get_command_nesting_depth()

        return new_generator._generate_argument_with_new_rules(
                        arg_schema, context, ctx_dict, target_type_info,
                            nesting_depth, max_nesting_depth, no_new_vars_mode,
            arg_index=arg_index, parent_command=parent_command
        )

    def _get_compatible_nested_commands(self, allowed_commands: List[str], target_type: 'SemanticType', exclude_arrays: bool = False, expected_is_array: bool = None, exclude_arithmetic: bool = False, context: 'ProgramContext' = None) -> List[str]:
        """型互換性のあるネストされたコマンドを取得（ArgumentGeneratorに委譲）"""
        return self.argument_generators._get_compatible_nested_commands(allowed_commands, target_type, exclude_arrays, expected_is_array, exclude_arithmetic, context)

    def _get_expected_argument_type_info(self, cmd_name: str, arg_index: int, arg_schema) -> TypeInfo:
        """コマンドの各引数の実際の期待型情報を取得（ArgumentGeneratorに委譲）"""
        return self.argument_generators._get_expected_argument_type_info(cmd_name, arg_index, arg_schema)

    # ============================================================
    # セクション4: 変数生成関連
    # ============================================================

    def _normalize_variable_name(self, var_name: str) -> Tuple[str, Optional[str]]:
        """変数名を正規化（ベース名とインデックスに分割）（VariableGeneratorsに委譲）"""
        return self.variable_generators.normalize_variable_name(var_name)

    def _generate_variable_argument(
        self,
        target_type_info: TypeInfo,
        arg_schema,
        context: ProgramContext,
        ctx_dict: Dict,
        nesting_depth: int = 0,
        effective_max_nesting_depth: Optional[int] = None,
        no_new_vars_mode: bool = False,
        parent_command: Optional[str] = None
    ) -> Node:
        """変数引数を生成（既存変数の再利用または新規作成）

        新しいルール:
        - オブジェクト単体のSemanticTypeの場合、配列要素（objects[i]）が使えるものがあれば、90%の確率で使う
        - no_new_vars_mode=Trueの場合:
          - 選択されたSemanticTypeに既存変数がなければ、リテラル値選択に切り替え
          - 選択されたSemanticTypeに既存変数がある場合:
            - 未使用変数があれば、未使用既存変数からランダム選択
            - 未使用変数がなければ、使用済み既存変数からランダム選択
        - no_new_vars_mode=Falseの場合:
          - 選択されたSemanticTypeに既存変数がなければ、新規変数作成
          - 選択されたSemanticTypeに既存変数がある場合:
            - 90%: 未使用変数があれば、未使用既存変数からランダム選択
                    未使用変数がなければ、使用済み既存変数からランダム選択
            - 10%: 新規変数作成

        注意: nesting_depthは「引数ネスト深度」（コマンド引数内のネスト）であり、
        「スコープネスト深度」（FOR/IFの構造的ネスト）とは別物です。
        """
        from ..metadata.variable_manager import variable_manager
        from ..metadata.types import TypeSystem
        from .nodes import VariableNode

        selected_type = target_type_info.semantic_type
        is_array = target_type_info.is_array

        # ========================================================================
        # セクション1: 障害物配列の除外処理
        # ========================================================================
        if (selected_type == SemanticType.OBJECT and
            is_array and
            hasattr(context, 'current_command') and context.current_command):
            if self.variable_managers._should_exclude_object_from_obstacle_array(context.current_command, 0, context):
                return self._generate_obstacle_array_with_exclusion(context.current_command, context, ctx_dict)

        # ========================================================================
        # セクション2: FORループ内での配列要素使用
        # ========================================================================
        # オブジェクト単体のSemanticTypeの場合、配列要素（objects[i]）が使えるものがあれば、90%の確率で使う
        if (selected_type == SemanticType.OBJECT and not is_array):
            valid_for_arrays = [arr for arr in context.for_arrays if arr is not None]
            if valid_for_arrays and random.random() < 0.9:
                # 配列要素を使用
                try:
                    array_element_node = self._generate_array_element_node(context, ctx_dict, use_index_zero=False)
                    if array_element_node:
                        return array_element_node
                except ValueError as e:
                    # 「利用可能な配列がありません（重複回避のため）」エラーの場合も、フォールバック処理に進む
                    # 通常の変数生成と同様に、新規変数作成やリテラル値にフォールバックする
                    if "利用可能な配列がありません" in str(e):
                        pass  # フォールバック処理に進む（エラーを再発生させない）
                    else:
                        raise  # その他のValueErrorは再発生
                except Exception:
                    pass  # フォールバック処理に進む

        # ========================================================================
        # セクション3: 既存変数の取得とフィルタリング
        # ========================================================================
        # 既存変数を取得
        existing_vars = self.variable_managers._get_existing_variables_for_argument(target_type_info, context)

        # 除外変数と使用不可変数を除去
        excluded_vars = getattr(context, 'excluded_variables_for_definition', set())
        try:
            unavailable_vars = context.variable_manager.get_unavailable_variables()
        except Exception:
            unavailable_vars = set()
        existing_vars = [var for var in existing_vars if var not in excluded_vars and var not in unavailable_vars]

        # 可視性を考慮してフィルタ（IF文の条件分岐を考慮した変数管理）
        existing_vars = [var for var in existing_vars if context.is_variable_visible(var)]

        # 重複回避ロジックでフィルタ
        used_variables_in_current_command = getattr(context, 'used_variables_in_current_command', set())
        filtered_vars = []
        for var in existing_vars:
            normalized_result = self._normalize_variable_name(var)
            # インデックスも考慮した重複チェックのため、タプル全体を使用
            if isinstance(normalized_result, tuple):
                normalized_var = normalized_result  # タプル全体を使用（ベース名+インデックス）
            else:
                normalized_var = (normalized_result, None)  # 通常変数の場合もタプルに統一
            # 正規化された変数名が使用済みでない場合のみ追加
            if normalized_var not in used_variables_in_current_command:
                # FLOW/LAY/SLIDEコマンドの特殊ケース: 配列要素（例: objects[i]）が使用済みの場合、
                # その配列のベース名（例: objects）も使用できない
                current_cmd = getattr(context, 'current_command', None)
                if current_cmd in ['FLOW', 'LAY', 'SLIDE', 'PATHFIND']:
                    # 配列全体（例: objects）を生成しようとする場合
                    if not isinstance(normalized_result, tuple):
                        # 通常変数（配列全体）の場合
                        base_name = normalized_result
                        # 同じベース名の配列要素が使用済みかチェック
                        has_array_element = any(
                            isinstance(used_var, tuple) and used_var[0] == base_name and used_var[1] is not None
                            for used_var in used_variables_in_current_command
                        )
                        if has_array_element:
                            continue  # 配列要素が使用済みの場合は除外
                filtered_vars.append(var)
        existing_vars = filtered_vars

        # 型互換性でフィルタ
        # 算術演算・比較演算の場合は、専用の互換性チェックを使用
        is_arithmetic = parent_command in ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']
        is_proportional = parent_command in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']

        compatible_existing_vars = []
        for var in existing_vars:
            var_info = context.variable_manager.get_variable_info(var)
            if var_info and var_info.get('type_info'):
                other_ti = var_info['type_info']
                try:
                    compatible = arg_schema.is_compatible_with(
                        other_ti,
                        is_arithmetic_operation=is_arithmetic,
                        is_proportional_operation=is_proportional
                    )
                except AttributeError:
                    compatible = (
                        TypeSystem.is_compatible(target_type_info.semantic_type, other_ti.semantic_type)
                        and target_type_info.is_array == other_ti.is_array
                    )
                if compatible:
                    compatible_existing_vars.append(var)
            else:
                # 型情報が無い場合は一旦許可（従来挙動維持）
                compatible_existing_vars.append(var)
        existing_vars = compatible_existing_vars

        # 重複回避後も使用可能な変数（available_vars）を設定
        available_vars = existing_vars

        # ========================================================================
        # セクション4: 既存変数の使用または新規変数の生成
        # ========================================================================
        # 重複回避後も使用可能な変数がない場合、設計上のフォールバック処理を実行
        # 「選択されたSemanticTypeに既存変数がなければ」に該当する
        if not available_vars:
            if no_new_vars_mode:
                # リテラル値選択に切り替え
                return self._generate_value_argument(arg_schema, context, ctx_dict, target_type_info=target_type_info)
            else:
                # 新規変数作成
                var_name = variable_manager.get_next_variable_name(
                    target_type_info,
                    set(context.variable_manager.get_all_variable_names())
                )
                # 逆算方式では、変数を使用したことを記録するだけで、定義は後で行う
                context.variable_manager.register_variable_usage(var_name, "argument", target_type_info)
                # 型情報を予約（定義ではない）
                self._reserve_variable_type_info(var_name, selected_type, is_array, context)
                return VariableNode(var_name, ctx_dict)

        if no_new_vars_mode:
            # no_new_vars_mode=Trueの場合
            # 未使用変数を取得（すべての未使用変数から、選択された型と互換性のあるものをフィルタ）
            all_unused_vars = context.variable_manager.get_unused_variables()
            unused_vars = []
            for var_name in all_unused_vars:
                if var_name not in available_vars:
                    continue  # 重複回避で除外された変数はスキップ
                var_info = context.variable_manager.get_variable_info(var_name)
                if var_info and 'type_info' in var_info:
                    var_type_info = var_info['type_info']
                    if (var_type_info.semantic_type == selected_type and
                        var_type_info.is_array == is_array):
                        unused_vars.append(var_name)

            if unused_vars:
                # 未使用変数があれば、未使用既存変数からランダム選択
                var_name = random.choice(unused_vars)
            else:
                # 未使用変数がなければ、使用済み既存変数からランダム選択（重複回避後も使用可能な変数から）
                used_vars = context.variable_manager.get_used_variable_names()
                compatible_used_vars = [v for v in used_vars if v in available_vars]
                if compatible_used_vars:
                    var_name = random.choice(compatible_used_vars)
                else:
                    # 使用済み変数もない場合、リテラル値選択に切り替え
                    return self._generate_value_argument(arg_schema, context, ctx_dict, target_type_info=target_type_info)

            # 変数ノードを生成
            context.variable_manager.register_variable_usage(var_name, "argument", target_type_info)
            # 同じコマンド内での重複を防ぐため、正規化形式で記録
            if hasattr(context, 'used_variables_in_current_command'):
                normalized_result = self._normalize_variable_name(var_name)
                if isinstance(normalized_result, tuple):
                    normalized_var = normalized_result
                else:
                    normalized_var = (normalized_result, None)
                context.used_variables_in_current_command.add(normalized_var)
            return VariableNode(var_name, ctx_dict)
        else:
            # no_new_vars_mode=Falseの場合
            if random.random() < 0.9:
                # 90%: 既存変数を使用
                # 未使用変数を取得（すべての未使用変数から、選択された型と互換性のあるものをフィルタ）
                all_unused_vars = context.variable_manager.get_unused_variables()
                unused_vars = []
                for var_name in all_unused_vars:
                    if var_name not in available_vars:
                        continue  # 重複回避で除外された変数はスキップ
                    var_info = context.variable_manager.get_variable_info(var_name)
                    if var_info and 'type_info' in var_info:
                        var_type_info = var_info['type_info']
                        if (var_type_info.semantic_type == selected_type and
                            var_type_info.is_array == is_array):
                            unused_vars.append(var_name)

                if unused_vars:
                    # 未使用変数があれば、未使用既存変数からランダム選択
                    var_name = random.choice(unused_vars)
                else:
                    # 未使用変数がなければ、使用済み既存変数からランダム選択（重複回避後も使用可能な変数から）
                    used_vars = context.variable_manager.get_used_variable_names()
                    compatible_used_vars = [v for v in used_vars if v in available_vars]
                    if compatible_used_vars:
                        var_name = random.choice(compatible_used_vars)
                    else:
                        # 使用済み変数もない場合、新規変数作成
                        var_name = variable_manager.get_next_variable_name(
                            target_type_info,
                            set(context.variable_manager.get_all_variable_names())
                        )
                        context.variable_manager.register_variable_usage(var_name, "argument", target_type_info)
                        self._reserve_variable_type_info(var_name, selected_type, is_array, context)

                # 変数ノードを生成
                context.variable_manager.register_variable_usage(var_name, "argument", target_type_info)
                # 同じコマンド内での重複を防ぐため、正規化形式で記録
                if hasattr(context, 'used_variables_in_current_command'):
                    normalized_result = self._normalize_variable_name(var_name)
                    if isinstance(normalized_result, tuple):
                        normalized_var = normalized_result
                    else:
                        normalized_var = (normalized_result, None)
                    context.used_variables_in_current_command.add(normalized_var)
                return VariableNode(var_name, ctx_dict)
            else:
                # 10%: 新規変数作成
                var_name = variable_manager.get_next_variable_name(
                    target_type_info,
                    set(context.variable_manager.get_all_variable_names())
                )
                context.variable_manager.register_variable_usage(var_name, "argument", target_type_info)
                self._reserve_variable_type_info(var_name, selected_type, is_array, context)
                return VariableNode(var_name, ctx_dict)

    def _get_context_dict(self, context: ProgramContext, ctx_dict: Dict) -> Dict:
        """コンテキスト辞書を取得（NodeUtilitiesに委譲）"""
        return self.utilities.get_context_dict(context, ctx_dict)

    def _get_grid_size(self, context_dict: Dict) -> Tuple[int, int]:
        """グリッドサイズを取得（NodeUtilitiesに委譲）"""
        return self.utilities.get_grid_size(context_dict)

    def _is_color_type(self, type_info: Optional[TypeInfo], arg_schema) -> bool:
        """COLOR型かどうかを判定（NodeUtilitiesに委譲）"""
        return self.utilities.is_color_type(type_info, arg_schema)

    def _is_bool_type(self, type_info: Optional[TypeInfo], arg_schema) -> bool:
        """BOOL型かどうかを判定（NodeUtilitiesに委譲）"""
        return self.utilities.is_bool_type(type_info, arg_schema)

    def _check_bool_literal_prob(self, arg_schema) -> None:
        """BOOL型でliteral_prob=0.0の場合は例外を発生（NodeUtilitiesに委譲）"""
        return self.utilities.check_bool_literal_prob(arg_schema)

    def _clamp_color_range(self, range_min: int, range_max: int) -> Tuple[int, int]:
        """COLOR型の範囲を0-9に制限（NodeUtilitiesに委譲）"""
        return self.utilities.clamp_color_range(range_min, range_max)

    def _generate_value_argument(self, arg_schema, context: ProgramContext, ctx_dict: Dict, target_type_info=None, excluded_values: Optional[List[int]] = None) -> Node:
        """値引数を生成（主にリテラル値）。OBJECT型のリテラル値はtypes.pyのget_literal_choicesで管理（objects[0]またはobjects）。

        Args:
            excluded_values: 除外するリテラル値のリスト（例: [0] で0を除外、[0, 1] で0と1を除外）
        """
        # 型情報の決定: target_type_infoが指定されていればそれを優先、なければarg_schemaから取得
        type_info_to_check = target_type_info if target_type_info else (arg_schema.type_info if hasattr(arg_schema, 'type_info') and arg_schema.type_info else None)

        # ========================================================================
        # セクション1: 選択肢がある場合（arg_schema.choices優先）
        # ========================================================================
        # arg_schema.choicesが定義されている場合、常に優先（target_type_infoに関係なく）
        if arg_schema.choices and len(arg_schema.choices) > 0:
            # BOOL型の場合は確率チェック
            if self._is_bool_type(type_info_to_check, arg_schema):
                self._check_bool_literal_prob(arg_schema)
            selected_value = random.choice(arg_schema.choices)

            # OBJECT型の場合、objects[0]やobjectsは変数参照として扱う
            if type_info_to_check and type_info_to_check.semantic_type == SemanticType.OBJECT:
                from .nodes import VariableNode
                return VariableNode(selected_value, ctx_dict)

            return LiteralNode(selected_value, ctx_dict)

        # ========================================================================
        # セクション2: types.pyのget_literal_choicesを使用（フォールバック）
        # ========================================================================
        # arg_schema.choicesがない場合、types.pyのget_literal_choicesを使用
        if type_info_to_check:
            literal_choices = TypeSystem.get_literal_choices(
                type_info_to_check.semantic_type,
                is_array=type_info_to_check.is_array
            )
            if literal_choices is not None and len(literal_choices) > 0:
                # BOOL型の場合は確率チェック
                if type_info_to_check.semantic_type == SemanticType.BOOL:
                    self._check_bool_literal_prob(arg_schema)
                selected_value = random.choice(literal_choices)

                # OBJECT型の場合、objects[0]やobjectsは変数参照として扱う
                if type_info_to_check.semantic_type == SemanticType.OBJECT:
                    from .nodes import VariableNode
                    return VariableNode(selected_value, ctx_dict)

                return LiteralNode(selected_value, ctx_dict)

        # ========================================================================
        # セクション3: コンテキスト辞書とグリッドサイズの取得
        # ========================================================================
        context_dict = self._get_context_dict(context, ctx_dict)

        # ========================================================================
        # セクション4: custom_generatorが定義されている場合
        # ========================================================================
        if hasattr(arg_schema, 'custom_generator') and arg_schema.custom_generator:
            try:
                value = arg_schema.custom_generator(context_dict)
                if self._is_color_type(type_info_to_check, arg_schema) and isinstance(value, int):
                    value = max(0, min(9, value))
                return LiteralNode(value, ctx_dict)
            except Exception:
                pass  # custom_generatorが失敗した場合は、下の処理に進む

        # ========================================================================
        # セクション5: get_range()を使用してグリッドサイズ対応の範囲を取得
        # ========================================================================
        try:
            range_min, range_max = arg_schema.get_range(context_dict)
            if range_min is not None and range_max is not None:
                if self._is_color_type(type_info_to_check, arg_schema):
                    range_min, range_max = self._clamp_color_range(range_min, range_max)
                # excluded_valuesに含まれる値を除外
                if excluded_values:
                    # 除外する値が範囲内にある場合、それらを除外した範囲から選択
                    valid_values = [v for v in range(range_min, range_max + 1) if v not in excluded_values]
                    if valid_values:
                        return LiteralNode(random.choice(valid_values), ctx_dict)
                    else:
                        # 有効な値がない場合、除外値を含めて範囲を拡張
                        while range_min in excluded_values and range_min <= range_max:
                            range_min += 1
                        if range_min > range_max:
                            range_min = 1
                            range_max = max(1, range_max)
                return LiteralNode(random.randint(range_min, range_max), ctx_dict)
        except Exception:
            pass  # get_range()が失敗した場合は、下のフォールバック処理に進む

        # ========================================================================
        # セクション6: range_min/range_maxを使用（get_range()が失敗した場合のフォールバック）
        # ========================================================================
        if (arg_schema.range_min is not None and arg_schema.range_max is not None and
            not getattr(arg_schema, 'depends_on_grid_size', False)):
            range_min, range_max = arg_schema.range_min, arg_schema.range_max
            if self._is_color_type(type_info_to_check, arg_schema):
                range_min, range_max = self._clamp_color_range(range_min, range_max)
            # excluded_valuesに含まれる値を除外
            if excluded_values:
                valid_values = [v for v in range(range_min, range_max + 1) if v not in excluded_values]
                if valid_values:
                    return LiteralNode(random.choice(valid_values), ctx_dict)
                else:
                    # 有効な値がない場合、除外値を含めて範囲を拡張
                    while range_min in excluded_values and range_min <= range_max:
                        range_min += 1
                    if range_min > range_max:
                        range_min = 1
                        range_max = max(1, range_max)
            return LiteralNode(random.randint(range_min, range_max), ctx_dict)

        # ========================================================================
        # セクション7: 最終フォールバック（デフォルトの整数値）
        # ========================================================================
        grid_width, grid_height = self._get_grid_size(context_dict)
        range_tuple = TypeSystem.get_default_range(SemanticType.SIZE, grid_width, grid_height)
        if range_tuple:
            range_min, range_max = range_tuple
            # excluded_valuesに含まれる値を除外
            if excluded_values:
                valid_values = [v for v in range(range_min, range_max + 1) if v not in excluded_values]
                if valid_values:
                    return LiteralNode(random.choice(valid_values), ctx_dict)
                else:
                    # 有効な値がない場合、除外値を含めて範囲を拡張
                    while range_min in excluded_values and range_min <= range_max:
                        range_min += 1
                    if range_min > range_max:
                        range_min = 1
                        range_max = max(1, range_max)
            return LiteralNode(random.randint(range_min, range_max), ctx_dict)
        else:
            # excluded_valuesに含まれる値を除外
            if excluded_values:
                valid_values = [v for v in range(0, 31) if v not in excluded_values]
                if valid_values:
                    return LiteralNode(random.choice(valid_values), ctx_dict)
                else:
                    return LiteralNode(random.randint(1, 30), ctx_dict)
            else:
                return LiteralNode(random.randint(0, 30), ctx_dict)

    # ============================================================
    # セクション8: ユーティリティ
    # ============================================================

    def _is_special_argument(self, arg_schema) -> bool:
        """特殊な引数かどうかを判定（NodeUtilitiesに委譲）"""
        return self.utilities.is_special_argument(arg_schema)

    def _is_proportional_operation(self, cmd_name: str) -> bool:
        """比例演算かどうかを判定（NodeUtilitiesに委譲）"""
        return self.utilities.is_proportional_operation(cmd_name)

    # ============================================================
    # セクション5: プレースホルダー関連
    # ============================================================

    def _generate_argument_with_validation(self, arg_schema, context: ProgramContext, ctx_dict: Dict, target_type_info: TypeInfo = None, nesting_depth: int = 0, max_nesting_depth: Optional[int] = None, no_new_vars_mode: bool = False) -> Node:
        """引数生成（検証なし、通常の生成処理）

        注意: nesting_depthは「引数ネスト深度」（コマンド引数内のネスト）であり、
        「スコープネスト深度」（FOR/IFの構造的ネスト）とは別物です。
        """
        # 複雑度に応じたコマンドネスト深度を取得（引数ネスト深度の上限）
        if max_nesting_depth is None:
            max_nesting_depth = context.get_command_nesting_depth()
        return self._generate_argument_node(arg_schema, context, ctx_dict, target_type_info, no_new_vars_mode=no_new_vars_mode, nesting_depth=nesting_depth, max_nesting_depth=max_nesting_depth)

    def _is_constant_only_operation(self, cmd_name: str, args: List[Node]) -> bool:
        """定数同士の演算かどうかをチェック（NodeUtilitiesに委譲）"""
        return self.utilities.is_constant_only_operation(cmd_name, args)

    def _generate_placeholder_node(self, placeholder_name: str, context: ProgramContext, ctx_dict: Dict) -> PlaceholderNode:
        """プレースホルダーノードを生成"""
        return PlaceholderNode(placeholder_name, ctx_dict)

    def _get_placeholder_variables_in_node(self, node: Node) -> set:
        """ノード内で使用されているプレースホルダー変数（$obj, $obj1, $obj2など）を取得（NodeUtilitiesに委譲）"""
        return self.utilities.get_placeholder_variables_in_node(node)

    def _get_variables_used_in_node(self, node: Node) -> set:
        """ノード内で使用されている変数名を取得（NodeUtilitiesに委譲）"""
        return self.utilities.get_variables_used_in_node(node)

    # ============================================================
    # セクション6: ノード生成関連（特殊ノード）
    # ============================================================

    def _generate_object_definition_argument(self, context: ProgramContext, ctx_dict: Dict, type_info: TypeInfo, var_name: str = None, no_new_vars_mode: bool = False, nesting_depth: int = 0, max_nesting_depth: Optional[int] = None) -> Node:
        """オブジェクト定義用の引数を生成（配列要素、コマンド、またはコマンドの引数として配列要素）

        注意: nesting_depthは「引数ネスト深度」（コマンド引数内のネスト）であり、
        「スコープネスト深度」（FOR/IFの構造的ネスト）とは別物です。
        """
        # 複雑度に応じたコマンドネスト深度を取得（引数ネスト深度の上限）
        if max_nesting_depth is None:
            max_nesting_depth = context.get_command_nesting_depth()

        # OBJECTを返すコマンドのみを許可
        object_commands = [
            'MOVE', 'TELEPORT', 'SLIDE', 'PATHFIND', 'ROTATE', 'FLIP', 'SCALE', 'SCALE_DOWN', 'EXPAND',
            'FILL_HOLES', 'SET_COLOR', 'OUTLINE', 'HOLLOW', 'BBOX', 'INTERSECTION',
            'SUBTRACT', 'FLOW', 'DRAW', 'LAY', 'ALIGN', 'CROP', 'FIT_SHAPE', 'FIT_SHAPE_COLOR', 'FIT_ADJACENT'
        ]

        from ..metadata.commands import create_argument_schema_with_naming_system
        arg_schema = create_argument_schema_with_naming_system(
            type_info,
            literal_prob=0.0,
            variable_prob=0.0,
            nested_prob=1.0,
            allowed_nested_commands=object_commands  # OBJECTを返すコマンドのみ
        )

        # 自己参照を禁止する場合
        if var_name:
            return self._generate_argument_node_without_self_reference(arg_schema, context, ctx_dict, type_info, var_name, no_new_vars_mode=no_new_vars_mode, nesting_depth=nesting_depth, max_nesting_depth=max_nesting_depth)
        else:
            return self._generate_argument_node(arg_schema, context, ctx_dict, type_info, no_new_vars_mode=no_new_vars_mode, nesting_depth=nesting_depth, max_nesting_depth=max_nesting_depth)

    def _generate_advanced_filter(self, source_array: str, target_array: str, context: ProgramContext) -> FilterNode:
        """FILTERノードを生成（専用条件生成）"""
        from ..metadata.argument_schema import CONDITION_ARG
        from .program_context import PlaceholderTracking

        # プレースホルダー追跡をリセット
        context.used_placeholders_in_current_command = set()
        context.used_variables_in_current_command = set()
        # OBJECT型引数カウントをリセット
        context.filter_sort_object_arg_count = 0

        # プレースホルダー追跡を初期化
        context.placeholder_tracking = PlaceholderTracking('FILTER')

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['source_array'] = source_array
        ctx_dict['target_array'] = target_array

        # FILTER用のプレースホルダーコンテキストを設定
        context.current_placeholder_context = 'FILTER'

        # FILTERコンテキストを型選択時に参照できるようにcurrent_commandを設定
        prev_command_for_filter = context.current_command if hasattr(context, 'current_command') else None
        context.current_command = 'FILTER'

        # CONDITION_ARGを使用して条件ノードを生成（検証付き）
        # SemanticType.BOOLをTypeInfoに変換
        from ..metadata.types import TypeInfo
        bool_type_info = TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False)

        try:
            condition_node = self._generate_argument_with_validation(
                CONDITION_ARG, context, ctx_dict, bool_type_info, nesting_depth=1, max_nesting_depth=context.get_command_nesting_depth()
            )
        except Exception as e:
            import traceback
            raise

        # current_commandを元に戻す
        context.current_command = prev_command_for_filter

        # プレースホルダー追跡をクリア
        context.placeholder_tracking = None

        # FILTERノードの引数生成完了後、カウンターをリセット（次のFILTERノードのために）
        context.filter_sort_object_arg_count = 0

        # 条件ノード内で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(condition_node, context)

        condition = condition_node.generate()

        # 条件が変数名（flagなど）かどうかをチェック
        if isinstance(condition, str):
            # 変数名のパターンをチェック（単純な変数名、コマンドではない）
            import re
            simple_var_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
            if re.match(simple_var_pattern, condition.strip()) and not condition.strip().startswith('$'):
                # 変数が定義されているかチェック
                var_info = context.variable_manager.get_variable_info(condition.strip())

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        # プレースホルダーコンテキストをリセット
        context.current_placeholder_context = None

        filter_node = FilterNode(
            source_array=source_array,
            target_array=target_array,
            condition=condition,
            context=context.to_dict()
        )
        return filter_node


    def _generate_sort_node(self, current_array: str, var_name: str, context: ProgramContext, no_new_vars_mode: bool = False) -> Node:
        """ソートノードを生成（専用キー式生成）"""
        from ..metadata.argument_schema import KEY_EXPR_ARG, ORDER_ARG
        from .program_context import PlaceholderTracking

        # プレースホルダー追跡をリセット
        context.used_placeholders_in_current_command = set()
        context.used_variables_in_current_command = set()
        # OBJECT型引数カウントをリセット
        context.filter_sort_object_arg_count = 0

        # プレースホルダー追跡を初期化
        context.placeholder_tracking = PlaceholderTracking('SORT_BY')

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['current_array'] = current_array
        ctx_dict['target_array'] = var_name

        # SORT_BY用のプレースホルダーコンテキストを設定
        context.current_placeholder_context = 'SORT_BY'

        # SORT_BYコンテキストを型選択時に参照できるようにcurrent_commandを設定
        prev_command_for_sort = context.current_command if hasattr(context, 'current_command') else None
        context.current_command = 'SORT_BY'

        # KEY_EXPR_ARGを使用してキー式ノードを生成（検証付き）
        # SORT_BY_TYPESから重み付きでSemanticTypeを選択
        from ..metadata.types import TypeInfo, TypeSystem
        import random
        sort_by_types_dict = TypeSystem.SORT_BY_TYPES
        sort_by_types = list(sort_by_types_dict.keys())
        sort_by_weights = list(sort_by_types_dict.values())
        selected_type = random.choices(sort_by_types, weights=sort_by_weights, k=1)[0]
        selected_type_info = TypeInfo.create_from_semantic_type(selected_type, is_array=False)

        key_expr_node = self._generate_argument_with_validation(
            KEY_EXPR_ARG, context, ctx_dict, selected_type_info, nesting_depth=1, max_nesting_depth=context.get_command_nesting_depth(), no_new_vars_mode=no_new_vars_mode
        )

        # current_commandを元に戻す
        context.current_command = prev_command_for_sort

        # プレースホルダー追跡をクリア
        context.placeholder_tracking = None

        # キー式ノード内で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(key_expr_node, context)

        # ORDER引数を生成（ORDER_ARGを使用）
        order_arg_node = self._generate_value_argument(ORDER_ARG, context, ctx_dict)

        # SORT_BYコマンドノードを生成
        # key_expr_nodeはCommandNode（例：GET_SIZE($obj)）またはVariableNodeの可能性がある
        # そのまま使用する（VariableNodeに変換しない）
        command_node = CommandNode(
            command='SORT_BY',
            arguments=[
                VariableNode(current_array),
                key_expr_node,  # 生成されたキー式ノードをそのまま使用（プレースホルダー変数を含む可能性がある）
                order_arg_node  # ORDER_ARGから生成されたリテラル値ノード
            ],
            context=ctx_dict
        )

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(current_array, "argument", OBJECT_ARRAY_TYPE)
        # var_name（target_array）は新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(var_name)

        # 代入ノードを作成（左辺: var_name, 右辺: SORT_BYコマンド）
        assignment_node = AssignmentNode(
            variable=var_name,
            expression=command_node,
            context=ctx_dict
        )

        # 変数を登録
        self.variable_managers._register_variables_used_in_node(assignment_node, context)

        # プレースホルダーコンテキストをリセット
        context.current_placeholder_context = None

        return assignment_node


    def _generate_match_pairs_node(self, array1: str, array2: str, target_array: str, context: ProgramContext) -> MatchPairsNode:
        """MATCH_PAIRSノードを生成（専用条件生成）"""
        from .nodes import MatchPairsNode
        from ..metadata.argument_schema import MATCH_PAIRS_CONDITION_ARG
        from .program_context import PlaceholderTracking

        # プレースホルダー追跡をリセット
        context.used_placeholders_in_current_command = set()
        context.used_variables_in_current_command = set()

        # プレースホルダー追跡を初期化
        context.placeholder_tracking = PlaceholderTracking('MATCH_PAIRS')

        # コンテキストを更新
        ctx_dict = context.to_dict()
        ctx_dict['array1'] = array1
        ctx_dict['array2'] = array2
        ctx_dict['target_array'] = target_array

        # MATCH_PAIRS用のプレースホルダーコンテキストを設定
        context.current_placeholder_context = 'MATCH_PAIRS'

        # MATCH_PAIRSコンテキストを型選択時に参照できるようにcurrent_commandを設定
        prev_command_for_match = context.current_command if hasattr(context, 'current_command') else None
        context.current_command = 'MATCH_PAIRS'

        # MATCH_PAIRS_CONDITION_ARGを使用して条件ノードを生成（検証付き）
        # SemanticType.BOOLをTypeInfoに変換
        from ..metadata.types import TypeInfo
        bool_type_info = TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False)

        condition_node = self._generate_argument_with_validation(
            MATCH_PAIRS_CONDITION_ARG, context, ctx_dict, bool_type_info, nesting_depth=1, max_nesting_depth=context.get_command_nesting_depth()
        )

        # current_commandを元に戻す
        context.current_command = prev_command_for_match

        # プレースホルダー追跡をクリア
        context.placeholder_tracking = None

        # 条件ノード内で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(condition_node, context)

        condition = condition_node.generate()

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(array1, "argument", OBJECT_ARRAY_TYPE)
        context.variable_manager.register_variable_usage(array2, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        # プレースホルダーコンテキストをリセット
        context.current_placeholder_context = None

        return MatchPairsNode(
            array1=array1,
            array2=array2,
            target_array=target_array,
            condition=condition,
            context=context.to_dict()
        )


    def _generate_exclude_node(self, source_array: str, target_array: str, targets_array: str, context: ProgramContext) -> ExcludeNode:
        """EXCLUDEノードを生成（CommandGeneratorsに委譲）"""
        return self.command_generators.generate_exclude_node(source_array, target_array, targets_array, context)

    def _generate_concat_node(self, array1: str, array2: str, target_array: str, context: ProgramContext) -> ConcatNode:
        """CONCATノードを生成（CommandGeneratorsに委譲）"""
        return self.command_generators.generate_concat_node(array1, array2, target_array, context)

    def _generate_append_node(self, array: str, obj: str, target_array: str, context: ProgramContext) -> AppendNode:
        """APPENDノードを生成（CommandGeneratorsに委譲）"""
        return self.command_generators.generate_append_node(array, obj, target_array, context)

    def _generate_merge_node(self, objects_array: str, target_obj: str, context: ProgramContext) -> MergeNode:
        """MERGEノードを生成（CommandGeneratorsに委譲）"""
        return self.command_generators.generate_merge_node(objects_array, target_obj, context)

    def _generate_empty_array_node(self, array_name: str, context: ProgramContext) -> EmptyArrayNode:
        """空のオブジェクト配列定義ノードを生成（CommandGeneratorsに委譲）"""
        return self.command_generators.generate_empty_array_node(array_name, context)




    def _generate_if_branch(self, context: ProgramContext) -> Node:
        """IF分岐開始ノードを生成"""
        from .nodes import IfStartNode

        # IFネスト深度を1増加
        context.enter_if_nesting()

        # コンテキストを更新object
        ctx_dict = context.to_dict()
        ctx_dict['in_if_branch'] = True

        # 条件を生成
        condition = self._generate_if_condition(context)

        return IfStartNode(
            condition=condition,
            context=ctx_dict
        )

    def _generate_if_condition(self, context: ProgramContext) -> str:
        """IF条件を生成（統一された引数生成システムを使用）"""
        from ..metadata.argument_schema import BOOL_ARG

        # 統一された引数生成システムを使用
        try:
            ctx_dict = context.to_dict()

            # IF条件は最上位の引数なので、nesting_depth=0から開始
            condition_node = self._generate_argument_node(
                BOOL_ARG, context, ctx_dict, BOOL_ARG.type_info,
                nesting_depth=0, max_nesting_depth=context.get_command_nesting_depth()
            )

            result = condition_node.generate()

            return result

        except Exception as e:
            # 最後の手段として、EQUAL(1, 1)を文字列で返す
            return "EQUAL(1, 1)"

    def _generate_for_loop(self, current_array: str, context: ProgramContext) -> Node:
        """FORループ開始ノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_for_loop(current_array, context)

    def _generate_for_loop_with_count(self, count_variable: str, context: ProgramContext) -> Node:
        """COUNT型変数を使用したFORループ開始ノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_for_loop_with_count(count_variable, context)

    def _generate_for_loop_with_constant(self, constant_value: int, context: ProgramContext) -> Node:
        """定数値を使用したFORループ開始ノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_for_loop_with_constant(constant_value, context)

    def _generate_for_loop_with_match_pairs(self, match_pairs_array: str, context: ProgramContext) -> Node:
        """MATCH_PAIRS配列用のFORループ開始ノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_for_loop_with_match_pairs(match_pairs_array, context)

    def _generate_end(self, context: ProgramContext) -> Node:
        """ENDするだけのノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_end(context)

    def _generate_variable_definition_node_with_literal(self, var_name: str, semantic_type: 'SemanticType', literal_value: Any, context: ProgramContext) -> Node:
        """リテラル値を使用した変数定義ノードを生成（StructureGeneratorsに委譲）"""
        return self.structure_generators.generate_variable_definition_node_with_literal(var_name, semantic_type, literal_value, context)

    def _generate_array_element_assignment(self, for_array: str, loop_idx: int, context: ProgramContext, ctx_dict: Dict) -> Node:
        """配列要素代入を生成"""
        # FORループ内ではプレースホルダーコンテキストをリセット
        context.current_placeholder_context = None

        # コマンド全体を生成するための特別な引数スキーマを作成
        from ..metadata.argument_schema import ArgumentSchema

        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        # 範囲外アクセスを防ぐ
        if loop_idx >= len(loop_vars):
            loop_idx = loop_idx % len(loop_vars)  # 循環的に使用
        loop_var = loop_vars[loop_idx]

        # 利用可能なコマンドを取得（配列を返さないコマンドのみ、MERGEも除外）
        all_commands = list(COMMAND_METADATA.keys())
        array_returning_commands = ['FILTER', 'MATCH_PAIRS', 'SORT_BY', 'EXTEND_PATTERN', 'EXCLUDE', 'ARRANGE_GRID', 'CONCAT', 'APPEND', 'EXTRACT_LINES', 'EXTRACT_RECTS', 'EXTRACT_HOLLOW_RECTS', 'GET_COLORS', 'GET_ALL_OBJECTS', 'REVERSE', 'TILE', 'SPLIT_CONNECTED']
        forbidden_in_for_loop = ['MERGE']  # FORループ内で禁止するコマンド
        available_commands = [cmd for cmd in all_commands if cmd not in array_returning_commands and cmd not in forbidden_in_for_loop]

        # コマンド全体を生成するための引数スキーマ
        command_schema = ArgumentSchema(
            type_info=TypeInfo(
                return_type=ReturnType.OBJECT,
                semantic_type=SemanticType.OBJECT,
                is_array=False
            ),
            literal_prob=0.0,  # リテラルは使わない
            variable_prob=0.0,  # 変数は使わない
            nested_prob=1.0,   # 必ずネストされたコマンドを生成
            allowed_nested_commands=available_commands,  # 配列を返さないコマンドのみを許可
            description="コマンド全体生成用"
        )
        target_type_info = TypeInfo(
            return_type=ReturnType.OBJECT,
            semantic_type=SemanticType.OBJECT,
            is_array=False
        )

        # nested_prob=1.0, variable_prob=0.0, literal_prob=0.0 により、必ずコマンドが生成されるため、
        # 変数や配列要素が生成されることはない。したがって、除外処理は不要。
        # _generate_argument_nodeでコマンド全体を生成
        # 注意: Complexity: 1の場合、max_command_nesting_depth=1のため、nesting_depth=0で呼び出す必要がある
        max_nesting_depth = context.get_command_nesting_depth()
        operation = self._generate_argument_node(
            command_schema, context, ctx_dict, target_type_info,
            nesting_depth=0, max_nesting_depth=max_nesting_depth
        )

        # 代入ノードを作成（逆算ルール適用）

        # FORループ内の配列代入ノード数をカウント
        if context:
            context.increment_for_array_assignment_count()
            # オブジェクト操作コマンドが生成された場合はカウント
            # オブジェクト操作カウントは削除されました（max_object_operations制限を削除したため）

        # MATCH_PAIRS配列の場合は特別なインデックス計算を使用
        if for_array in context.match_pairs_arrays:
            # MULTIPLY(i, 2) または ADD(MULTIPLY(i, 2), 1) をランダムに選択
            if random.random() < 0.5:
                # MULTIPLY(i, 2)
                index_expr = f"MULTIPLY({loop_var}, 2)"
            else:
                # ADD(MULTIPLY(i, 2), 1)
                index_expr = f"ADD(MULTIPLY({loop_var}, 2), 1)"
            index_str = index_expr
        else:
            index_str = loop_var

        assignment_node = ArrayAssignmentNode(
            array=for_array,
            index=index_str,
            expression=operation,
            context=ctx_dict
        )

        return assignment_node




    def _get_array_element_type(self, array_name: str, context: ProgramContext) -> SemanticType:
        """配列要素の型を取得（VariableGeneratorsに委譲）"""
        return self.variable_generators.get_array_element_type(array_name, context)

    def _select_assignment_target(self, target_type: 'SemanticType', context: ProgramContext) -> str:
        """逆算ルールに基づいて代入先を選択（VariableGeneratorsに委譲）"""
        return self.variable_generators.select_assignment_target(target_type, context)

    def _generate_variable_definition_node(self, var_name: str, context: ProgramContext, limited_commands: List[str] = None, no_new_vars_mode: bool = False, provided_type_info: Optional[TypeInfo] = None) -> AssignmentNode:
        """変数定義ノードを生成"""
        # 変数の型情報を取得
        var_info = context.variable_manager.get_variable_info(var_name)

        # 明示型が引数で渡された場合はそれを最優先で使用
        if provided_type_info is not None:
            type_info = provided_type_info
        elif not var_info or 'type_info' not in var_info:
            # 型情報が見つからない場合は変数名から型を推定
            type_info = context.variable_manager.tracker._infer_type_from_variable_name(var_name)
            if not type_info:
                # 推定も失敗した場合はデフォルトのOBJECT型を使用
                type_info = TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)
        else:
            type_info = var_info['type_info']

        # 右辺の引数スキーマを作成（変数定義ノード専用：変数単体のみ禁止）
        from ..metadata.commands import create_argument_schema_with_naming_system

        if type_info.semantic_type == SemanticType.OBJECT and type_info.is_array:
            # オブジェクト配列の場合
            arg_schema = create_argument_schema_with_naming_system(
                type_info,
                literal_prob=0.0,  # 定数は使用しない
                variable_prob=0.0,  # 変数単体禁止
                nested_prob=1.0   # コマンドのみ許可
            )
        elif type_info.semantic_type == SemanticType.OBJECT:
            # オブジェクトの場合（リテラルは許可しない）
            # variable_probは0だが、配列要素は特別に許可
            arg_schema = create_argument_schema_with_naming_system(
                type_info,
                literal_prob=0.0,  # 定数は許可しない（OBJECT型のリテラルは存在しない）
                variable_prob=0.0,  # 変数単体禁止（ただし配列要素は許可）
                nested_prob=1.0   # コマンドのみ許可
            )
            # 代入先の変数を定義として登録（自己参照問題を解決）

            # オブジェクト定義の場合は配列要素も許可
            ctx_dict = context.to_dict()
            try:
                right_side = self._generate_object_definition_argument(context, ctx_dict, type_info, var_name, no_new_vars_mode=no_new_vars_mode)
            except Exception as e:
                # OBJECT型のリテラル値は存在しないため、例外を再発生
                # （OBJECT型以外ではリテラル値にフォールバックするが、OBJECT型では不可能）
                raise

            # 右辺で使用された変数を登録
            self.variable_managers._register_variables_used_in_node(right_side, context)
            context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
            # スコープ情報を記録
            context.add_scope_variable(var_name)

            # 代入ノードを作成
            assignment_node = AssignmentNode(
                variable=var_name,
                expression=right_side,
                context=ctx_dict
            )

            return assignment_node
        else:
            # その他の型の場合（SIZE、COLOR、BOOLを含む）
            # 現在の一番近いネストのタイプと、FORネストの存在をチェック
            current_nest_type = None
            has_for_nest = False
            if context.nest_stack:
                current_nest = context.nest_stack[-1]
                current_nest_type = current_nest.get('type')
                # FORネストが存在するかチェック（nest_stack全体を確認）
                has_for_nest = any(nest.get('type') == 'for' for nest in context.nest_stack)

            # 3つのケースに分ける:
            # 1. 現在の一番近いネストがFORの場合
            # 2. FORネストがあるが、現在の一番近いネストがIFの場合（FOR内のIF）
            # 3. それ以外（IFのネストだけのとき、またはネストなしの時）
            is_in_for_loop = (current_nest_type == 'for')
            is_in_if_within_for = (has_for_nest and current_nest_type == 'if')

            # ネストの状態に応じたnested_prob_valueとliteral_probを決定
            if is_in_for_loop:
                nested_prob_value = 500.0  # FORループ内
            elif is_in_if_within_for:
                nested_prob_value = 1.0  # FOR内のIF
            else:  # is_in_if_or_outside
                nested_prob_value = 0.3  # IFのみまたはネスト外

            # literal_probの決定
            literal_prob = 1.0  # 定数の確率を下げる（0.6→0.5）→ ネストコマンド確率を40%→50%に向上

            arg_schema = create_argument_schema_with_naming_system(
                type_info,
                literal_prob=literal_prob,
                variable_prob=0.0,  # 変数単体禁止
                nested_prob=nested_prob_value
            )

        # 限定されたコマンドが指定されている場合は、それを使用
        if limited_commands:
            # can_use_command()でフィルタリング（max_array_operations制限を適用）
            available_commands = [cmd for cmd in limited_commands if context.can_use_command(cmd)]
            if not available_commands:
                # 使用可能なコマンドがない場合はNoneを返す（呼び出し側で処理）
                return None
            # 最大10回まで再試行（limited_commandsは通常2つ程度で、重複の可能性は低い）
            # 最初の再試行開始時の変数状態をスナップショットとして保存
            previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
            previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

            for attempt in range(10):
                # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
                if attempt > 0:
                    # 現在の変数状態を取得
                    current_variable_names = set(context.variable_manager.get_all_variable_names())
                    current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                    # 前回の試行開始時以降に追加された変数を特定
                    new_variables = current_variable_names - previous_attempt_start_variable_names
                    new_type_reservation_vars = current_type_reservation_vars - previous_attempt_start_type_reservation_vars

                    # 削除対象の変数を決定
                    # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                    vars_to_remove = []
                    for var_name in new_variables:
                        var_info = context.variable_manager.get_variable_info(var_name)
                        if var_info and not var_info.get('is_defined', False) and var_info.get('is_used', False):
                            vars_to_remove.append(var_name)

                    # 2. 前回の試行開始時以降に追加された型情報予約変数
                    vars_to_remove.extend(list(new_type_reservation_vars))

                    # 重複を除去して削除
                    vars_to_remove = list(set(vars_to_remove))
                    if vars_to_remove:
                        context.variable_manager.remove_variables(vars_to_remove)
                        # variable_scope_mapからも削除（スコープ情報の整合性を保つため）
                        for var_name in vars_to_remove:
                            if var_name in context.variable_scope_map:
                                del context.variable_scope_map[var_name]
                        # scope_variablesからも削除
                        context.scope_variables = [v for v in context.scope_variables if v not in vars_to_remove]
                        # nest_stack内のscope_variablesからも削除
                        for nest in context.nest_stack:
                            if 'scope_variables' in nest:
                                nest['scope_variables'] = [v for v in nest['scope_variables'] if v not in vars_to_remove]

                    # used_variables_in_current_commandもリセット
                    context.used_variables_in_current_command = set()

                # 現在の試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
                previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
                previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                selected_command = random.choice(available_commands)
                result = self._generate_variable_definition_node_with_command(var_name, selected_command, context, provided_type_info=type_info, no_new_vars_mode=no_new_vars_mode)
                if result is not None:  # 重複チェックをパスした場合
                    return result
            # 10回失敗した場合は最後の結果を返す（フォールバック）
            return self._generate_variable_definition_node_with_command(var_name, selected_command, context, provided_type_info=type_info, no_new_vars_mode=no_new_vars_mode)


        # 右辺の引数を生成（自己参照を禁止）
        # 変数定義ノードであることをコンテキストに設定（算術演算の重み調整用）
        prev_is_var_def = getattr(context, 'is_variable_definition_context', False)
        context.is_variable_definition_context = True
        ctx_dict = context.to_dict()
        right_side = None
        try:
            right_side = self._generate_argument_node_without_self_reference(arg_schema, context, ctx_dict, type_info, var_name, no_new_vars_mode=no_new_vars_mode, nesting_depth=1)
        except Exception as e:
            # 例外が発生した場合、OBJECT型以外ではリテラル値にフォールバック
            if type_info.semantic_type != SemanticType.OBJECT:
                try:
                    right_side = self._generate_value_argument(arg_schema, context, ctx_dict, target_type_info=type_info)
                except Exception as fallback_error:
                    # リテラル値の生成も失敗した場合、元の例外を再発生
                    raise e from fallback_error
            else:
                # OBJECT型の場合は例外を再発生（OBJECT型のリテラル値は存在しない）
                raise
        finally:
            # コンテキストを元に戻す
            context.is_variable_definition_context = prev_is_var_def

        # right_sideがNoneの場合はリテラル値を生成（フォールバック）
        if right_side is None:
            try:
                right_side = self._generate_value_argument(arg_schema, context, ctx_dict, target_type_info=type_info)
            except Exception as e:
                # 例外が発生した場合は再発生
                raise

        # 代入先の変数を定義として登録（自己参照問題を解決）
        context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
        # スコープ情報を記録
        context.add_scope_variable(var_name)

        # 右辺で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(right_side, context)

        # 代入ノードを作成
        assignment_node = AssignmentNode(
            variable=var_name,
            expression=right_side,
            context=ctx_dict
        )

        return assignment_node

    def _generate_argument_node_without_self_reference(self, arg_schema, context: ProgramContext, ctx_dict: Dict, type_info: TypeInfo, var_name: str, max_retries: int = 10, no_new_vars_mode: bool = False, nesting_depth: int = 0, max_nesting_depth: Optional[int] = None) -> Node:
        """自己参照を禁止した引数ノードを生成

        注意: nesting_depthは「引数ネスト深度」（コマンド引数内のネスト）であり、
        「スコープネスト深度」（FOR/IFの構造的ネスト）とは別物です。
        """
        # 複雑度に応じたコマンドネスト深度を取得（引数ネスト深度の上限）
        if max_nesting_depth is None:
            max_nesting_depth = context.get_command_nesting_depth()

        # 最初の試行開始時の変数状態をスナップショットとして保存
        # これにより、各再試行で追加された変数のみを正確に削除できる
        previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
        previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

        for attempt in range(max_retries):
            # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
            if attempt > 0:
                # 現在の変数状態を取得
                current_variable_names = set(context.variable_manager.get_all_variable_names())
                current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                # 前回の試行開始時以降に追加された変数を特定
                new_variables = current_variable_names - previous_attempt_start_variable_names
                new_type_reservation_vars = current_type_reservation_vars - previous_attempt_start_type_reservation_vars

                # 削除対象の変数を決定
                # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                vars_to_remove = []
                for var_name in new_variables:
                    var_info = context.variable_manager.get_variable_info(var_name)
                    if var_info and not var_info.get('is_defined', False) and var_info.get('is_used', False):
                        vars_to_remove.append(var_name)

                # 2. 前回の試行開始時以降に追加された型情報予約変数
                vars_to_remove.extend(list(new_type_reservation_vars))

                # 重複を除去して削除
                vars_to_remove = list(set(vars_to_remove))
                if vars_to_remove:
                    context.variable_manager.remove_variables(vars_to_remove)
                    # variable_scope_mapからも削除（スコープ情報の整合性を保つため）
                    for var_name in vars_to_remove:
                        if var_name in context.variable_scope_map:
                            del context.variable_scope_map[var_name]
                    # scope_variablesからも削除
                    context.scope_variables = [v for v in context.scope_variables if v not in vars_to_remove]
                    # nest_stack内のscope_variablesからも削除
                    for nest in context.nest_stack:
                        if 'scope_variables' in nest:
                            nest['scope_variables'] = [v for v in nest['scope_variables'] if v not in vars_to_remove]

                # used_variables_in_current_commandもリセット
                context.used_variables_in_current_command = set()

            # 現在の試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
            previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
            previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

            # 通常の引数生成
            try:
                right_side = self._generate_argument_node(arg_schema, context, ctx_dict, type_info, no_new_vars_mode=no_new_vars_mode, nesting_depth=nesting_depth, max_nesting_depth=max_nesting_depth)
            except Exception as e:
                # 引数生成で例外が発生した場合
                # 例外が発生した場合でも、最後の試行でない限り再試行
                if attempt < max_retries - 1:
                    continue
                # 最後の試行でも例外が発生した場合、Noneを返す（呼び出し元でフォールバック処理が実行される）
                return None

            # Noneが返された場合
            if right_side is None:
                # Noneが返された場合でも、最後の試行でない限り再試行
                if attempt < max_retries - 1:
                    continue
                # 最後の試行でもNoneが返された場合、Noneを返す（呼び出し元でフォールバック処理が実行される）
                return None

            # 自己参照チェック
            if self._is_self_reference(var_name, right_side):
                if attempt < max_retries - 1:
                    continue
                # 最後の試行でも自己参照の場合はNoneを返す
                return None

            # 自己参照でない場合は成功
            return right_side

        # 最大試行回数に達した場合の処理
        # リテラル値を生成（フォールバック）
        return self._generate_value_argument(arg_schema, context, ctx_dict, target_type_info=type_info)

    def _is_self_reference(self, var_name: str, node: Node) -> bool:
        """自己参照かどうかをチェック（NodeUtilitiesに委譲）"""
        return self.utilities.is_self_reference(var_name, node)


    def _generate_variable_definition_node_with_command(self, var_name: str, command: str, context: ProgramContext, provided_type_info: Optional[TypeInfo] = None, no_new_vars_mode: bool = False) -> AssignmentNode:
        """指定されたコマンドで変数定義ノードを生成"""

        # 変数の型情報を取得
        var_info = context.variable_manager.get_variable_info(var_name)

        if provided_type_info is not None:
            type_info = provided_type_info
        elif not var_info or 'type_info' not in var_info:
            type_info = context.variable_manager.tracker._infer_type_from_variable_name(var_name)
            if not type_info:
                type_info = TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)
        else:
            type_info = var_info['type_info']

        # 指定されたコマンドで右辺を生成
        ctx_dict = context.to_dict()

        # コマンドの引数を生成（自己参照を禁止）
        if command in COMMAND_METADATA:
            cmd_metadata = COMMAND_METADATA[command]
            # 引数生成時にcurrent_commandを設定（障害物配列の除外ロジックなどで使用）
            prev_command = getattr(context, 'current_command', None)
            context.current_command = command
            arguments = []
            try:
                for i, arg_schema in enumerate(cmd_metadata.arguments):
                    # 引数を生成（自己参照を禁止）
                    arg_node = self._generate_argument_node_without_self_reference(arg_schema, context, ctx_dict, arg_schema.type_info, var_name, no_new_vars_mode=no_new_vars_mode, nesting_depth=1)
                    if arg_node is None:
                        # 引数生成に失敗した場合は再生成を試みる（次の試行へ）
                        return None
                    arguments.append(arg_node)
            finally:
                # current_commandを元に戻す
                if prev_command is not None:
                    context.current_command = prev_command
                else:
                    context.current_command = None
        else:
            # フォールバック: 基本的な引数
            arguments = []

        # 変数重複チェック
        from ..metadata.variable_manager import variable_manager
        if variable_manager.check_variable_duplication(command, arguments, context):
            # 変数重複の場合は再生成（次の試行へ）
            return None

        # 代入先の変数を定義として登録（自己参照問題を解決）
        context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
        # スコープ情報を記録
        context.add_scope_variable(var_name)

        # コマンドノードを作成
        command_node = CommandNode(
            command=command,
            arguments=arguments,
            context=ctx_dict
        )

        # 代入ノードを作成
        assignment_node = AssignmentNode(
            variable=var_name,
            expression=command_node,
            context=ctx_dict
        )

        return assignment_node


    def _reserve_variable_type_info(self, var_name: str, semantic_type: 'SemanticType', is_array: bool, context: ProgramContext):
        """変数の型情報を予約（定義ではない）（VariableGeneratorsに委譲）"""
        return self.variable_generators.reserve_variable_type_info(var_name, semantic_type, is_array, context)

    def _generate_array_element_node(self, context: ProgramContext, ctx_dict: Dict, use_index_zero: bool = False) -> Node:
        """配列要素ノードを生成（VariableGeneratorsに委譲）"""
        return self.variable_generators.generate_array_element_node(context, ctx_dict, use_index_zero)




    def _should_exclude_object_from_obstacle_array(self, cmd_name: str, arg_index: int, context: ProgramContext) -> bool:
        """オブジェクトを障害物配列から除外すべきかどうかを判定"""
        # FLOW, LAY, SLIDE, PATHFINDコマンドの第1引数（操作対象）の場合
        obstacle_commands = ['FLOW', 'LAY', 'SLIDE', 'PATHFIND']
        if cmd_name in obstacle_commands and arg_index == 0:
            return True
        return False

    def _generate_obstacle_array_with_exclusion(self, cmd_name: str, context: ProgramContext, ctx_dict: Dict) -> Node:
        """障害物配列を生成（操作対象オブジェクトを除外）"""
        # 利用可能な配列変数を取得
        available_arrays = []
        for var_name in context.variable_manager.get_all_variable_names():
            var_info = context.variable_manager.get_variable(var_name)
            if (var_info and
                'type_info' in var_info and
                var_info['type_info'].semantic_type == SemanticType.OBJECT and
                var_info['type_info'].is_array):
                available_arrays.append(var_name)

        # 使用禁止/明示除外を反映
        try:
            unavailable_vars = context.variable_manager.get_unavailable_variables()
        except Exception:
            unavailable_vars = set()
        excluded_vars = getattr(context, 'excluded_variables_for_definition', set())
        filtered_arrays = [arr for arr in available_arrays if arr not in unavailable_vars and arr not in excluded_vars]
        if filtered_arrays:
            available_arrays = filtered_arrays

        if not available_arrays:
            # 配列変数がない場合は新しく作成
            array_name = variable_manager.get_next_variable_name(
                TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
                set(context.variable_manager.get_all_variable_names())
            )
            context.variable_manager.define_variable(array_name, SemanticType.OBJECT, is_array=True)
            # スコープ情報を記録
            context.add_scope_variable(array_name)
            return VariableNode(array_name, ctx_dict)

        # 配列を選択
        array_name = random.choice(available_arrays)

        # 操作対象オブジェクトが同じ配列の場合は、EXCLUDEで除外
        if hasattr(context, 'current_object') and context.current_object:
            # 操作対象オブジェクトが配列要素の場合
            if hasattr(context.current_object, 'name') and '[' in context.current_object.name:
                # 操作対象オブジェクトが配列要素の場合
                if context.current_object.name.startswith(array_name):
                    # EXCLUDEで除外した新しい配列を作成
                    excluded_array_name = variable_manager.get_next_variable_name(
                        TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
                set(context.variable_manager.get_all_variable_names())
            )
                    context.variable_manager.define_variable(excluded_array_name, SemanticType.OBJECT, is_array=True)
                    # スコープ情報を記録
                    context.add_scope_variable(excluded_array_name)

                    return VariableNode(excluded_array_name, ctx_dict)

        return VariableNode(array_name, ctx_dict)

    def _generate_object_access_node(self, obj_var: str, objects_array: str, access_type: str, context: ProgramContext) -> ObjectAccessNode:
        """オブジェクトアクセスノードを生成"""
        from .nodes import ObjectAccessNode

        # オブジェクト変数を定義
        context.variable_manager.define_variable(obj_var, SemanticType.OBJECT, is_array=False)
        # スコープ情報を記録
        context.add_scope_variable(obj_var)

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(objects_array, "argument", OBJECT_ARRAY_TYPE)

        return ObjectAccessNode(
            obj_var=obj_var,
            objects_array=objects_array,
            access_type=access_type,
            context=context.to_dict()
        )

    def _generate_single_object_array_node(self, array_name: str, object_name: str, context: ProgramContext) -> SingleObjectArrayNode:
        """単一オブジェクト配列定義ノードを生成"""
        from .nodes import SingleObjectArrayNode

        # 配列変数を定義
        context.variable_manager.define_variable(array_name, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(array_name)

        # オブジェクトが使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(object_name, "argument", OBJECT_TYPE)

        return SingleObjectArrayNode(
            array_name=array_name,
            object_name=object_name,
            context=context.to_dict()
        )

    def _generate_split_connected_node(self, source_object: str, target_array: str, context: ProgramContext, connectivity: int = 4) -> SplitConnectedNode:
        """SPLIT_CONNECTEDノードを生成"""
        from .nodes import SplitConnectedNode

        # オブジェクトと配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(source_object, "argument", OBJECT_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return SplitConnectedNode(
            source_object=source_object,
            target_array=target_array,
            connectivity=connectivity,
            context=context.to_dict()
        )

    def _generate_extract_shape_node(self, source_object: str, target_array: str, extract_type: str, context: ProgramContext) -> ExtractShapeNode:
        """形状抽出ノードを生成"""
        from .nodes import ExtractShapeNode

        # オブジェクトと配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(source_object, "argument", OBJECT_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return ExtractShapeNode(
            source_object=source_object,
            target_array=target_array,
            extract_type=extract_type,
            context=context.to_dict()
        )

    def _generate_extend_pattern_node(self, source_array: str, target_array: str, context: ProgramContext, side: str = "end", count: int = 1) -> ExtendPatternNode:
        """EXTEND_PATTERNノードを生成"""
        from .nodes import ExtendPatternNode

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return ExtendPatternNode(
            source_array=source_array,
            target_array=target_array,
            side=side,
            count=count,
            context=context.to_dict()
        )

    def _generate_arrange_grid_node(self, source_array: str, target_array: str, context: ProgramContext, cols: int = 3, width: int = 10, height: int = 10) -> ArrangeGridNode:
        """ARRANGE_GRIDノードを生成"""
        from .nodes import ArrangeGridNode

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)
        # target_arrayは新しく定義される変数なので、定義として登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(target_array)

        return ArrangeGridNode(
            source_array=source_array,
            target_array=target_array,
            cols=cols,
            width=width,
            height=height,
            context=context.to_dict()
        )

    def _generate_object_operations_node(self, var_name: str, context: ProgramContext) -> Node:
        """オブジェクト操作ノードを生成（オブジェクト操作コマンドのみに限定）"""
        # オブジェクト操作コマンドのみに限定（COMMAND_CATEGORIESから取得）
        object_operation_commands = COMMAND_CATEGORIES['transform'] + COMMAND_CATEGORIES['create']

        # _generate_variable_definition_nodeを使用してオブジェクト操作ノードを生成
        return self._generate_variable_definition_node(
            var_name,
            context,
            object_operation_commands,
            provided_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)
        )

    # ============================================================
    # セクション7: 複雑度制御関連
    # ============================================================

    def _select_complexity_level(self) -> str:
        """複雑さレベルを確率的に選択"""
        if not ENABLE_COMPLEXITY_CONTROL:
            return 'medium'  # デフォルト

        rand_val = random.random()
        cumulative_prob = 0

        for level, config in COMPLEXITY_LEVELS.items():
            cumulative_prob += config['probability']
            if rand_val <= cumulative_prob:
                return level

        return 'medium'  # フォールバック

    def _initialize_complexity_constraints(self, context):
        """複雑さ制約を初期化"""
        if not ENABLE_COMPLEXITY_CONTROL:
            return

        # 既にコンテキストに複雑度レベルが設定されている場合はそれを使用
        if hasattr(context, 'complexity_level') and context.complexity_level:
            level = context.complexity_level
        else:
            level = self._select_complexity_level()
            context.complexity_level = level
        context.complexity_constraints = COMPLEXITY_LEVELS[level].copy()

        # 制約強化フラグ
        context.enforce_strict_constraints = True
        context.constraint_violations = 0
        context.max_constraint_violations = 3  # 最大3回まで制約違反を許可


    def _should_add_structure_based_on_complexity(self, structure_type: str, context) -> bool:
        """複雑さ制約に基づいて構造を追加すべきかチェック（強化版）"""
        if not context.complexity_constraints:
            return True  # 制約なしの場合は追加可能

        constraints = context.complexity_constraints

        # 厳格な制約チェック
        if hasattr(context, 'enforce_strict_constraints') and context.enforce_strict_constraints:
            if structure_type == 'for_loop':
                can_add = context.current_for_loops < constraints['max_for_loops']
                return can_add
            elif structure_type == 'if_statement':
                can_add = context.current_if_statements < constraints['max_if_statements']
                return can_add
        # 通常の制約チェック（緩い）
        if structure_type == 'for_loop':
            return context.current_for_loops < constraints['max_for_loops']
        elif structure_type == 'if_statement':
            return context.current_if_statements < constraints['max_if_statements']

        return True

    def _update_complexity_counters(self, structure_type: str, context, increment: int = 1):
        """複雑さカウンターを更新"""
        if structure_type == 'for_loop':
            context.current_for_loops += increment
        elif structure_type == 'if_statement':
            context.current_if_statements += increment
        elif structure_type == 'line':
            context.current_lines += increment

    def _enforce_complexity_constraints(self, context) -> bool:
        """複雑さ制約を強制的にチェック（制約違反時はプログラム生成を停止）"""
        if not context.complexity_constraints:
            return True

        constraints = context.complexity_constraints

        # 各制約をチェック

        if context.current_for_loops > constraints['max_for_loops']:
            return False

        if context.current_if_statements > constraints['max_if_statements']:
            return False


        return True
