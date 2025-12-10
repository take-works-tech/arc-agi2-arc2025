"""
新しいルールに基づいた引数生成関数

このファイルは、docs/new_rules_summary.mdに記載された新しいルールに基づいて
引数生成を実装します。
"""
from __future__ import annotations

import random
import sys
from typing import List, Dict, Optional, Any, Tuple
from ..metadata.types import SemanticType, ReturnType, TypeSystem, TypeInfo
from ..metadata.argument_schema import ArgumentSchema, CONDITION_ARG, MATCH_PAIRS_CONDITION_ARG
from ..metadata.commands import COMMAND_METADATA, CommandMetadata
from ..metadata.constants import (
    MAX_COMMAND_GENERATION_RETRIES,
    MAX_FILTER_CONDITION_REGENERATION_RETRIES,
    MAX_MATCH_PAIRS_CONDITION_REGENERATION_RETRIES,
    MAX_SORT_BY_KEY_EXPR_REGENERATION_RETRIES,
    MAX_DUPLICATE_STRUCTURE_REGENERATION_RETRIES,
    SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER
)
from ..metadata.variable_manager import variable_manager
from .nodes import Node, LiteralNode, VariableNode, CommandNode, ArrayElementAccessNode
from .program_context import ProgramContext, PlaceholderTracking
from .variable_generators import VariableGenerators
from src.data_systems.generator.config import get_config


class ArgumentGenerator:
    """引数生成クラス"""

    def __init__(self, node_generators):
        """初期化

        Args:
            node_generators: NodeGeneratorsインスタンス（既存のメソッドを使用するため）
        """
        self.node_generators = node_generators
        # ログ出力制御設定を読み込み
        config = get_config()
        self.enable_debug_logs = config.enable_debug_logs

    def _get_compatible_nested_commands(self, allowed_commands: List[str], target_type: SemanticType, exclude_arrays: bool = False, expected_is_array: bool = None, exclude_arithmetic: bool = False, context: 'ProgramContext' = None) -> List[str]:
        """型互換性のあるネストされたコマンドを取得"""
        compatible_commands = []
        # allowed_commands が未指定(None/空)の場合は、全コマンドから型互換で候補を構築
        candidate_space = allowed_commands if allowed_commands else list(COMMAND_METADATA.keys())

        # 算術演算は戻り値の型が第1引数型に依存するため、特別な処理が必要
        # ただし、COMPATIBILITY_MATRIXで型互換性があれば、通常の型互換性チェックで自動的に候補に含まれる
        arith_cmds = ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']

        # 低確率で要求SemanticTypeを互換型へ差し替えて多様性を付与（配列期待やBOOLは除外）
        # ただし、context.disable_type_substitutionがTrueの場合は型差し替えを無効化
        # （算術演算の第1引数生成時など、型不一致を防ぐ必要がある場合）
        effective_target_type = target_type
        try:
            disable_type_substitution = getattr(context, 'disable_type_substitution', False) if context else False
            if not disable_type_substitution and expected_is_array is not True and target_type is not None and target_type in TypeSystem.COMPATIBILITY_MATRIX:
                alt_pool = set(TypeSystem.COMPATIBILITY_MATRIX[target_type])
                if target_type in alt_pool:
                    alt_pool.remove(target_type)
                if alt_pool and random.random() < 0.05:
                    effective_target_type = random.choice(list(alt_pool))
        except Exception:
            pass

        for cmd_name in candidate_space:
            if cmd_name not in COMMAND_METADATA:
                continue
            cmd_metadata = COMMAND_METADATA[cmd_name]

            # 算術演算の場合は、戻り値の型が第1引数型に依存するため、型互換性チェックを緩和
            # COMMAND_METADATAに登録されている型は固定されているが、実際の生成時には動的に型を選択する
            is_arithmetic = cmd_name in arith_cmds
            if is_arithmetic:
                # 算術演算の場合、ターゲット型がARITHMETIC_TYPESに含まれている場合のみ許可
                # （算術演算はARITHMETIC_TYPESの場合だけに制限）
                if target_type is not None and target_type not in TypeSystem.ARITHMETIC_TYPES:
                    continue
                # 算術演算は型互換性チェックをスキップ（後続の配列チェックと候補追加処理に進む）
            else:
                # 算術演算以外の場合は通常の型互換性チェック
                if not TypeSystem.are_compatible(cmd_metadata.return_type_info.semantic_type, effective_target_type):
                    continue

            # 配列/非配列の一致チェック（例外: GET_INPUT_GRID_SIZE は非配列でも許可）
            if expected_is_array is not None:
                if cmd_name != 'GET_INPUT_GRID_SIZE' and cmd_metadata.return_type_info.is_array != expected_is_array:
                    continue

            # 配列返却の除外指定
            if exclude_arrays and cmd_metadata.return_type_info.is_array:
                continue

            # usage_contextsのチェック（GET_ALL_OBJECTSのみ除外）
            # GET_ALL_OBJECTSは初期化専用で、引数生成では使用しない（InitializationNodeで直接生成される）
            if cmd_name == 'GET_ALL_OBJECTS':
                continue

            compatible_commands.append(cmd_name)

        # 例外許容: 算術演算は戻り値が第1引数型に依存するため、
        # ターゲット型がARITHMETIC_TYPESに含まれていて非配列期待なら候補に常に追加
        # （allowed_commandsが指定されていて算術演算が除外される可能性があるため、フォールバックとして追加）
        if expected_is_array is False and target_type is not None:
            # ARITHMETIC_TYPESの場合のみ、算術演算を追加
            if target_type in TypeSystem.ARITHMETIC_TYPES:
                for c in arith_cmds:
                    if c not in compatible_commands and c in COMMAND_METADATA:
                        compatible_commands.append(c)

        # 算術演算の過剰なネストを防ぐ: exclude_arithmeticがTrueの場合は算術演算を除外
        if exclude_arithmetic:
            compatible_commands = [c for c in compatible_commands if c not in ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']]

        return compatible_commands

    def _get_expected_argument_type_info(self, cmd_name: str, arg_index: int, arg_schema, target_type: Optional[SemanticType] = None) -> Optional[TypeInfo]:
        """コマンドの各引数の実際の期待型情報を取得

        Args:
            cmd_name: コマンド名
            arg_index: 引数のインデックス
            arg_schema: 引数スキーマ
            target_type: ターゲット型（算術演算・比較演算の場合は実際のターゲット型を使用）

        Returns:
            型情報（比較演算・算術演算の場合はNoneを返し、argument_generator.pyで動的に決定）
        """
        # 比較演算と算術演算の場合は、Noneを返す（argument_generator.pyで動的に決定）
        # これにより、COMMAND_METADATAの固定値を無視し、実際のターゲット型に依存する
        proportional_cmds = ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']
        arith_cmds = ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']

        if cmd_name in proportional_cmds or cmd_name in arith_cmds:
            # 比較演算と算術演算の引数の型は、argument_generator.pyで動的に決定される
            # COMMAND_METADATAの固定値は無視される
            return None

        # その他のコマンドの場合は、COMMAND_METADATAから型情報を取得
        if cmd_name in COMMAND_METADATA:
            cmd_metadata = COMMAND_METADATA[cmd_name]
            if arg_index < len(cmd_metadata.arguments):
                # 引数スキーマの型情報から実際の期待型情報を取得
                return cmd_metadata.arguments[arg_index].type_info

        # フォールバック: 引数スキーマの型情報を使用
        return arg_schema.type_info

    def _get_nesting_limit_with_relaxation(self, context: ProgramContext, nesting_depth: int, max_nesting_depth: int) -> int:
        """ネスト制限の取得関数（緩和を考慮）

        新しいルール:
        - 複雑度ごとのネスト制限を取得
        - 取得関数実行時の、すべての親のコマンドに、１つでもFILTERまたはSORT_BY、MATCH_PAIRSが含まれていた場合、ネスト制限＋１
        - 最大緩和１

        Args:
            context: プログラムコンテキスト
            nesting_depth: 現在のネスト深度
            max_nesting_depth: 基本の最大ネスト深度

        Returns:
            緩和を考慮した実効的な最大ネスト深度
        """
        relaxation = 0

        # すべての親コマンドを確認
        has_filter_match_sort = False

        # argument_nesting_command_stackからすべての親コマンドを取得
        for depth in range(nesting_depth + 1):
            if depth in context.argument_nesting_command_stack:
                commands = context.argument_nesting_command_stack[depth]
                for cmd in commands:
                    if not has_filter_match_sort and cmd in ['FILTER', 'SORT_BY', 'MATCH_PAIRS']:
                        has_filter_match_sort = True
                        relaxation += 1
                        break
                if has_filter_match_sort:
                    break

        # context.current_commandも確認
        if hasattr(context, 'current_command') and context.current_command:
            current_cmd = context.current_command
            if not has_filter_match_sort and current_cmd in ['FILTER', 'SORT_BY', 'MATCH_PAIRS']:
                has_filter_match_sort = True
                relaxation += 1

        # 最大緩和1
        relaxation = min(relaxation, 1)

        return max_nesting_depth + relaxation

    def _is_in_condition_arg_nesting(self, context: ProgramContext, arg_schema) -> bool:
        """CONDITION_ARGの引数ネスト内かどうかを判定

        Args:
            context: プログラムコンテキスト
            arg_schema: 引数スキーマ

        Returns:
            CONDITION_ARGの引数ネスト内の場合True
        """
        # FILTERのCONDITION_ARGの引数ネスト内かどうか
        # MATCH_PAIRSのCONDITION_ARGの引数ネスト内かどうか
        # SORT_BYのKEY_EXPR_ARGの引数ネスト内かどうか

        # 現在のコマンドスタックを確認
        for depth in range(len(context.argument_nesting_command_stack)):
            if depth in context.argument_nesting_command_stack:
                commands = context.argument_nesting_command_stack[depth]
                for cmd in commands:
                    if cmd in ['FILTER', 'MATCH_PAIRS', 'SORT_BY']:
                        return True

        # context.current_commandも確認
        if hasattr(context, 'current_command') and context.current_command:
            if context.current_command in ['FILTER', 'MATCH_PAIRS', 'SORT_BY']:
                return True

        return False

    def _is_target_type_object_single(self, target_type_info: Optional[TypeInfo]) -> bool:
        """ターゲット型がオブジェクト単体かどうかを判定

        Args:
            target_type_info: ターゲット型情報

        Returns:
            オブジェクト単体の場合True
        """
        if not target_type_info:
            return False
        return (target_type_info.semantic_type == SemanticType.OBJECT and
                not target_type_info.is_array)

    def _generate_argument_with_new_rules(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo = None,
        nesting_depth: int = 0,
        max_nesting_depth: Optional[int] = None,
        no_new_vars_mode: bool = False,
        arg_index: Optional[int] = None,
        parent_command: Optional[str] = None,
        first_arg_type: Optional[SemanticType] = None
    ) -> Node:
        """新しいルールに基づいた引数生成関数

        Args:
            arg_schema: 引数スキーマ
            context: プログラムコンテキスト
            ctx_dict: コンテキスト辞書
            target_type_info: ターゲット型情報
            nesting_depth: 現在のネスト深度
            max_nesting_depth: 最大ネスト深度
            no_new_vars_mode: 新しい変数を生成禁止モード
            arg_index: 引数のインデックス（第1引数=0, 第2引数=1など）
            parent_command: 親コマンド名
            first_arg_type: 第1引数の型（比較演算の第2引数生成時に使用、ネストされた比較演算による上書きを防ぐ）

        Returns:
            生成されたノード
        """
        # 最大ネスト深度を取得
        if max_nesting_depth is None:
            max_nesting_depth = context.get_command_nesting_depth()

        # 緩和を考慮した実効的な最大ネスト深度を計算
        effective_max_nesting_depth = self._get_nesting_limit_with_relaxation(
            context, nesting_depth, max_nesting_depth
        )

        # 親コマンドを取得
        if parent_command is None:
            parent_command = getattr(context, 'current_command', None)

        # 比較演算、算術演算、論理演算の判定
        is_proportional = parent_command in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']
        is_arithmetic = parent_command in ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']
        is_logical = parent_command in ['AND', 'OR']

        # CONDITION_ARGの判定
        is_condition_arg = (arg_schema is CONDITION_ARG or arg_schema is MATCH_PAIRS_CONDITION_ARG)
        is_in_condition_nesting = self._is_in_condition_arg_nesting(context, arg_schema)

        # 引数の種類に応じて処理を分岐
        if is_proportional:
            # 比較演算のルール
            if arg_index == 0:
                # 第1引数
                return self._generate_comparison_first_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    is_in_condition_nesting, parent_command
                )
            elif arg_index == 1:
                # 第2引数
                # 第1引数の型情報を取得（引数で直接指定された場合は優先、それ以外はcontextから取得）
                if first_arg_type is None:
                    first_arg_type = getattr(context, 'current_proportional_first_arg_type', None)
                return self._generate_comparison_second_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command, first_arg_type=first_arg_type
                )
        elif is_arithmetic:
            # 算術演算のルール
            if arg_index == 0:
                # 第1引数
                return self._generate_arithmetic_first_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command=parent_command
                )
            elif arg_index == 1:
                # 第2引数
                # 第1引数の型情報を取得（引数で直接指定された場合は優先、それ以外はcontextから取得）
                first_arg_type = getattr(context, 'current_arithmetic_first_arg_type', None)
                return self._generate_arithmetic_second_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command, first_arg_type=first_arg_type
                )
        elif is_logical:
            # 論理演算のルール
            if arg_index == 0:
                # 第1引数
                return self._generate_logical_first_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    is_in_condition_nesting, parent_command=parent_command
                )
            elif arg_index == 1:
                # 第2引数
                return self._generate_logical_second_arg(
                    arg_schema, context, ctx_dict, target_type_info,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    is_in_condition_nesting, parent_command=parent_command
                )
        else:
            # 通常の引数のルール
            return self._generate_normal_argument(
                arg_schema, context, ctx_dict, target_type_info,
                nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                is_in_condition_nesting, parent_command=parent_command
            )

    def _generate_comparison_first_arg(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: Optional[TypeInfo],
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        is_in_condition_nesting: bool,
        parent_command: str
    ) -> Node:
        """比較演算の第1引数を生成

        新しいルール:
        - SemanticType選択: PROPORTIONAL_TYPESから動的に選択（制約: GREATER/LESSではBOOL型を引数として使用しない）
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
        - 第1引数なので、literal_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら最初に戻り、PROPORTIONAL_TYPESから、他のSemanticType選択（すべてのSemanticTypeで再試行が発生した場合、失敗で④return）
        - 変数、リテラル値、コマンドを重み付き選択。
        - →②に進む
        """
        # arg_schemaからis_arrayフラグを取得
        target_is_array = arg_schema.type_info.is_array if arg_schema and arg_schema.type_info else False

        # PROPORTIONAL_TYPESから選択
        proportional_types = list(TypeSystem.PROPORTIONAL_TYPES)

        # GREATER/LESSではBOOL型を除外
        if parent_command in ['GREATER', 'LESS']:
            proportional_types = [t for t in proportional_types if t != SemanticType.BOOL]

        # 最大再試行回数
        max_retries = len(proportional_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER

        for retry in range(max_retries):
            # SemanticType選択
            selected_type = random.choice(proportional_types)

            # 重みを初期化
            literal_prob = arg_schema.literal_prob
            variable_prob = arg_schema.variable_prob
            nested_prob = arg_schema.nested_prob

            # 第1引数なので、literal_prob *= 0.0
            literal_prob = 0.0

            # CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
            if is_in_condition_nesting:
                variable_prob = 0.0

            # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
            if no_new_vars_mode:
                existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                    selected_type, is_array=False, context=context
                )
                if not existing_vars:
                    variable_prob = 0.0

            # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
            compatible_commands = self._get_compatible_nested_commands(
                arg_schema.allowed_nested_commands,
                selected_type,
                exclude_arrays=False,
                expected_is_array=target_is_array,  # arg_schemaのis_arrayフラグを使用
                exclude_arithmetic=False,
                context=context
            )
            if not compatible_commands:
                nested_prob = 0.0
            else:
                # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
                if self._check_all_commands_have_zero_weight(
                    compatible_commands, context, nesting_depth, effective_max_nesting_depth
                ):
                    nested_prob = 0.0

            # ネスト制限に達している場合、nested_prob *= 0.0
            if nesting_depth >= effective_max_nesting_depth:
                nested_prob = 0.0

            # すべての重みが0.0の場合、別のSemanticTypeを選択
            if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
                continue  # 次のSemanticTypeを試す

            # 重み付き選択
            choice = random.choices(
                ['literal', 'variable', 'nested'],
                weights=[literal_prob, variable_prob, nested_prob],
                k=1
            )[0]

            # 選択に応じて生成（②の処理）
            if choice == 'nested':
                # コマンドを生成
                return self._generate_command_node(
                    selected_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    expected_is_array=target_is_array  # target_type_infoのis_arrayフラグを使用
                )
            elif choice == 'variable':
                # 変数を生成（②の処理: 変数を選択した場合）
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                return self.node_generators._generate_variable_argument(
                    new_target_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode
                )
            else:
                # literal_prob=0.0なので、ここには来ないはず
                continue

        # すべてのSemanticTypeで失敗した場合、④return（失敗）
        raise ValueError("比較演算の第1引数生成に失敗しました（すべてのSemanticTypeで再試行が発生）")

    def _generate_comparison_second_arg(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        parent_command: str,
        first_arg_type: Optional[SemanticType] = None
    ) -> Node:
        """比較演算の第2引数を生成

        新しいルール:
        - SemanticType選択: 第1引数の型と互換性のある型（COMPATIBILITY_MATRIXから取得）
          - 選択確率: 第1引数型と同型: 90%の確率、それ以外の互換型: 10%の確率（均等配分）
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら最初に戻り、COMPATIBILITY_MATRIXから他のSemanticType選択（すべてのSemanticTypeで再試行が発生した場合、失敗で④return）
        - 変数、リテラル値、コマンドを重み付き選択。
        - →②に進む

        Args:
            first_arg_type: 第1引数の型（直接指定された場合は優先使用、Noneの場合はcontextから取得）
        """
        # target_type_infoからis_arrayフラグを取得
        target_is_array = target_type_info.is_array if target_type_info else False

        # 第1引数の型を取得（引数で指定されていない場合、contextから取得）
        if first_arg_type is None:
            first_arg_type = getattr(context, 'current_proportional_first_arg_type', None)

        if first_arg_type is None:
            # contextに保存されていない場合、target_type_infoから取得を試みる
            first_arg_type = target_type_info.semantic_type if target_type_info else None

        if first_arg_type is None:
            # 第1引数の型が不明な場合、PROPORTIONAL_TYPESから選択
            proportional_types_list = list(TypeSystem.PROPORTIONAL_TYPES)
            if not proportional_types_list:
                # PROPORTIONAL_TYPESが空の場合はデフォルト型を使用
                first_arg_type = SemanticType.COORDINATE
            else:
                first_arg_type = random.choice(proportional_types_list)

        # 互換性のある型を取得
        compatible_types = list(TypeSystem.COMPATIBILITY_MATRIX.get(first_arg_type, {first_arg_type}))

        # 最大再試行回数
        max_retries = len(compatible_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER

        for retry in range(max_retries):
            # SemanticType選択（90%で同型、10%で互換型）
            if random.random() < 0.9 and first_arg_type in compatible_types:
                selected_type = first_arg_type
            else:
                # 互換型から選択（同型を除く）
                other_types = [t for t in compatible_types if t != first_arg_type]
                if other_types:
                    selected_type = random.choice(other_types)
                else:
                    selected_type = first_arg_type

            # 重みを初期化
            literal_prob = arg_schema.literal_prob
            variable_prob = arg_schema.variable_prob
            nested_prob = arg_schema.nested_prob

            # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
            if no_new_vars_mode:
                existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                    selected_type, is_array=False, context=context
                )
                if not existing_vars:
                    variable_prob = 0.0

            # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
            compatible_commands = self._get_compatible_nested_commands(
                arg_schema.allowed_nested_commands,
                selected_type,
                exclude_arrays=False,
                expected_is_array=target_is_array,  # target_type_infoのis_arrayフラグを使用
                exclude_arithmetic=False,
                context=context
            )
            if not compatible_commands:
                nested_prob = 0.0
            else:
                # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
                if self._check_all_commands_have_zero_weight(
                    compatible_commands, context, nesting_depth, effective_max_nesting_depth
                ):
                    nested_prob = 0.0

            # ネスト制限に達している場合、nested_prob *= 0.0
            if nesting_depth >= effective_max_nesting_depth:
                nested_prob = 0.0

            # すべての重みが0.0の場合、別のSemanticTypeを選択
            if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
                continue  # 次のSemanticTypeを試す

            # 重み付き選択
            choice = random.choices(
                ['literal', 'variable', 'nested'],
                weights=[literal_prob, variable_prob, nested_prob],
                k=1
            )[0]

            # 選択に応じて生成（②の処理）
            if choice == 'literal':
                # リテラル値を生成（②の処理: リテラル値を選択した場合）
                # リテラル値が選ばれた場合、第1引数と異なるSemanticTypeが選択されていた場合、第1引数型と同型に変更
                literal_type = first_arg_type if selected_type != first_arg_type else selected_type
                new_target_type = TypeInfo.create_from_semantic_type(literal_type, is_array=False)
                return self.node_generators._generate_value_argument(
                    arg_schema, context, ctx_dict, target_type_info=new_target_type
                )
            elif choice == 'variable':
                # 変数を生成（②の処理: 変数を選択した場合）
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                return self.node_generators._generate_variable_argument(
                    new_target_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode
                )
            else:
                # コマンドを生成
                # 第1引数の型と互換性があるコマンドのみを選択するように、_get_compatible_nested_commandsでフィルタリング済み
                # しかし、念のため、生成されたコマンドの戻り値の型が第1引数の型と互換性があるかをチェック
                command_node = self._generate_command_node(
                    selected_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    expected_is_array=target_is_array  # target_type_infoのis_arrayフラグを使用
                )

                # 生成されたコマンドの戻り値の型が第1引数の型と互換性があるかをチェック
                if isinstance(command_node, CommandNode) and first_arg_type is not None:
                    # コマンドのメタデータから戻り値の型を取得
                    cmd_name = command_node.command if hasattr(command_node, 'command') else None
                    if cmd_name:
                        cmd_meta = COMMAND_METADATA.get(cmd_name)
                        if cmd_meta and cmd_meta.return_type_info:
                            return_type = cmd_meta.return_type_info.semantic_type
                            # 互換性をチェック
                            compatible_types = TypeSystem.COMPATIBILITY_MATRIX.get(first_arg_type, {first_arg_type})
                            if return_type not in compatible_types:
                                # 互換性がない場合、再試行（次のSemanticTypeを試す）
                                continue

                return command_node

        # すべてのSemanticTypeで失敗した場合、④return（失敗）
        raise ValueError("比較演算の第2引数生成に失敗しました（すべてのSemanticTypeで再試行が発生）")

    def _generate_arithmetic_first_arg(
        self,
        arg_schema: Optional[ArgumentSchema],
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: Optional[TypeInfo],
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        parent_command: Optional[str] = None
    ) -> Node:
        """算術演算の第1引数を生成

        新しいルール:
        - SemanticType選択: ターゲット型と互換性のある型（COMPATIBILITY_MATRIXから取得）
          - 選択確率: ターゲット型と同型: 90%の確率、それ以外の互換型: 10%の確率（均等配分）
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら最初に戻り、他のSemanticType選択（すべてのSemanticTypeで再試行が発生した場合、失敗で④return）
        - 変数、リテラル値、コマンドを重み付き選択。
        - リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
        - →②に進む
        """
        # target_type_infoからis_arrayフラグを取得
        target_is_array = target_type_info.is_array if target_type_info else False

        # ターゲット型を取得
        target_type = target_type_info.semantic_type if target_type_info else None

        if target_type is None:
            # ターゲット型が不明な場合、ARITHMETIC_TYPESから選択
            # 算術演算で使用可能な型は、ARITHMETIC_TYPESに限定される
            arithmetic_types_list = list(TypeSystem.ARITHMETIC_TYPES)
            if not arithmetic_types_list:
                # ARITHMETIC_TYPESが空の場合はデフォルト型を使用
                target_type = SemanticType.COORDINATE
            else:
                target_type = random.choice(arithmetic_types_list)

        # 互換性のある型を取得（COMPATIBILITY_MATRIXから取得）
        compatible_types = list(TypeSystem.COMPATIBILITY_MATRIX.get(target_type, {target_type}))

        # 最大再試行回数
        max_retries = len(compatible_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER

        for retry in range(max_retries):
            # SemanticType選択:
            # - ターゲット型と同型: 90%の確率
            # - それ以外の互換型（COMPATIBILITY_MATRIXから取得した互換型の中で、ターゲット型以外）: 10%の確率（均等配分）
            # - ターゲット型と同型の1つしか互換性の型がない場合（compatible_types = {target_type}）、ターゲット型と同型100%になる
            if random.random() < 0.9 and target_type in compatible_types:
                selected_type = target_type
            else:
                # 互換型から選択（ターゲット型を除く）
                other_types = [t for t in compatible_types if t != target_type]
                if other_types:
                    # それ以外の互換型から均等配分で選択
                    selected_type = random.choice(other_types)
                else:
                    # ターゲット型と同型の1つしか互換性の型がない場合、ターゲット型を選択（100%）
                    selected_type = target_type

            # 重みを初期化
            literal_prob = arg_schema.literal_prob
            variable_prob = arg_schema.variable_prob
            nested_prob = arg_schema.nested_prob

            # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
            if no_new_vars_mode:
                existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                    selected_type, is_array=False, context=context
                )
                if not existing_vars:
                    variable_prob = 0.0

            # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
            compatible_commands = self._get_compatible_nested_commands(
                arg_schema.allowed_nested_commands,
                selected_type,
                exclude_arrays=False,
                expected_is_array=target_is_array,  # target_type_infoのis_arrayフラグを使用
                exclude_arithmetic=False,
                context=context
            )
            if not compatible_commands:
                nested_prob = 0.0
            else:
                # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
                if self._check_all_commands_have_zero_weight(
                    compatible_commands, context, nesting_depth, effective_max_nesting_depth
                ):
                    nested_prob = 0.0

            # ネスト制限に達している場合、nested_prob *= 0.0
            if nesting_depth >= effective_max_nesting_depth:
                nested_prob = 0.0

            # すべての重みが0.0の場合、別のSemanticTypeを選択
            if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
                continue  # 次のSemanticTypeを試す

            # 重み付き選択
            choice = random.choices(
                ['literal', 'variable', 'nested'],
                weights=[literal_prob, variable_prob, nested_prob],
                k=1
            )[0]

            # 選択に応じて生成（②の処理）
            if choice == 'literal':
                # リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
                if selected_type != target_type:
                    selected_type = target_type
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                # 無駄な処理を避けるため、特定の値を除外
                excluded_values = []
                if parent_command == 'ADD':
                    excluded_values = [0]  # ADD(0, x) は x と同じ
                elif parent_command == 'MULTIPLY':
                    excluded_values = [1]  # MULTIPLY(1, x) は x と同じ
                # 第1引数がリテラル値であることを記録（第2引数生成時にリテラル値同士の演算を防ぐため）
                context.current_arithmetic_first_arg_is_literal = True
                return self.node_generators._generate_value_argument(
                    arg_schema, context, ctx_dict, target_type_info=new_target_type,
                    excluded_values=excluded_values if excluded_values else None
                )
            elif choice == 'variable':
                # 第1引数が変数であることを記録
                context.current_arithmetic_first_arg_is_literal = False
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                return self.node_generators._generate_variable_argument(
                    new_target_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command=parent_command
                )
            else:
                # コマンドを生成（第1引数がコマンドであることを記録）
                context.current_arithmetic_first_arg_is_literal = False
                return self._generate_command_node(
                    selected_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    expected_is_array=target_is_array  # target_type_infoのis_arrayフラグを使用
                )

        # すべてのSemanticTypeで失敗した場合、④return（失敗）
        raise ValueError("算術演算の第1引数生成に失敗しました（すべてのSemanticTypeで再試行が発生）")

    def _generate_arithmetic_second_arg(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        parent_command: str,
        first_arg_type: Optional[SemanticType] = None
    ) -> Node:
        """算術演算の第2引数を生成

        新しいルール:
        - SemanticType選択: 第1引数の型と互換性のある型（COMPATIBILITY_MATRIXから取得）
          - 選択確率: 第1引数型と同型: 90%の確率、それ以外の互換型: 10%の確率（均等配分）
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - 第一引数がリテラル値の場合、literal_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら最初に戻り、他のSemanticType選択（すべてのSemanticTypeで再試行が発生した場合、失敗で④return）
        - 変数、リテラル値、コマンドを重み付き選択。
        - リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
        - →②に進む

        Args:
            first_arg_type: 第1引数の型（直接指定された場合は優先使用、Noneの場合はcontextから取得）
        """
        # target_type_infoからis_arrayフラグを取得
        target_is_array = target_type_info.is_array if target_type_info else False

        # 第1引数の型を取得（引数で指定されていない場合、contextから取得）
        if first_arg_type is None:
            first_arg_type = getattr(context, 'current_arithmetic_first_arg_type', None)

        if first_arg_type is None:
            # contextに保存されていない場合、target_type_infoから取得を試みる
            first_arg_type = target_type_info.semantic_type if target_type_info else None

        if first_arg_type is None:
            # 第1引数の型が不明な場合、ARITHMETIC_TYPESから選択
            # 算術演算で使用可能な型は、ARITHMETIC_TYPESに限定される
            if TypeSystem.ARITHMETIC_TYPES:
                first_arg_type = random.choice(list(TypeSystem.ARITHMETIC_TYPES))
            else:
                # フォールバック: デフォルト型を使用
                first_arg_type = SemanticType.COORDINATE

        # 互換性のある型を取得
        compatible_types = list(TypeSystem.COMPATIBILITY_MATRIX.get(first_arg_type, {first_arg_type}))

        # 最大再試行回数
        max_retries = len(compatible_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER

        for retry in range(max_retries):
            # SemanticType選択（90%で同型、10%で互換型）
            if random.random() < 0.9 and first_arg_type in compatible_types:
                selected_type = first_arg_type
            else:
                # 互換型から選択（同型を除く）
                other_types = [t for t in compatible_types if t != first_arg_type]
                if other_types:
                    selected_type = random.choice(other_types)
                else:
                    selected_type = first_arg_type

            # 重みを初期化
            literal_prob = arg_schema.literal_prob
            variable_prob = arg_schema.variable_prob
            nested_prob = arg_schema.nested_prob

            # 第一引数がリテラル値の場合、literal_prob *= 0.0
            # 第1引数がリテラル値かどうかを確認（contextから取得）
            first_arg_is_literal = getattr(context, 'current_arithmetic_first_arg_is_literal', False)
            if first_arg_is_literal:
                literal_prob = 0.0

            # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
            if no_new_vars_mode:
                existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                    selected_type, is_array=False, context=context
                )
                if not existing_vars:
                    variable_prob = 0.0

            # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
            compatible_commands = self._get_compatible_nested_commands(
                arg_schema.allowed_nested_commands,
                selected_type,
                exclude_arrays=False,
                expected_is_array=False,
                exclude_arithmetic=False,
                context=context
            )
            if not compatible_commands:
                nested_prob = 0.0
            else:
                # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
                if self._check_all_commands_have_zero_weight(
                    compatible_commands, context, nesting_depth, effective_max_nesting_depth
                ):
                    nested_prob = 0.0

            # ネスト制限に達している場合、nested_prob *= 0.0
            if nesting_depth >= effective_max_nesting_depth:
                nested_prob = 0.0

            # すべての重みが0.0の場合、別のSemanticTypeを選択
            if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
                continue  # 次のSemanticTypeを試す

            # 重み付き選択
            choice = random.choices(
                ['literal', 'variable', 'nested'],
                weights=[literal_prob, variable_prob, nested_prob],
                k=1
            )[0]

            # 選択に応じて生成（②の処理）
            if choice == 'literal':
                # リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
                # 注意: ターゲット型は第1引数の型と同じ
                if selected_type != first_arg_type:
                    selected_type = first_arg_type
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                # 無駄な処理を避けるため、特定の値を除外
                excluded_values = []
                if parent_command == 'ADD':
                    excluded_values = [0]  # ADD(x, 0) は x と同じ
                elif parent_command == 'SUB':
                    excluded_values = [0]  # SUB(x, 0) は x と同じ
                elif parent_command in ['DIVIDE', 'MOD']:
                    excluded_values = [0, 1]  # DIVIDE(x, 0) はエラー、DIVIDE(x, 1) は x と同じ
                elif parent_command == 'MULTIPLY':
                    excluded_values = [1]  # MULTIPLY(x, 1) は x と同じ
                return self.node_generators._generate_value_argument(
                    arg_schema, context, ctx_dict, target_type_info=new_target_type,
                    excluded_values=excluded_values if excluded_values else None
                )
            elif choice == 'variable':
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
                return self.node_generators._generate_variable_argument(
                    new_target_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command=parent_command
                )
            else:
                # コマンドを生成
                # 第1引数の型と互換性があるコマンドのみを選択するように、_get_compatible_nested_commandsでフィルタリング済み
                # しかし、念のため、生成されたコマンドの戻り値の型が第1引数の型と互換性があるかをチェック
                command_node = self._generate_command_node(
                    selected_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    expected_is_array=target_is_array  # target_type_infoのis_arrayフラグを使用
                )

                # 生成されたコマンドの戻り値の型が第1引数の型と互換性があるかをチェック
                if isinstance(command_node, CommandNode) and first_arg_type is not None:
                    # コマンドのメタデータから戻り値の型を取得
                    cmd_name = command_node.command if hasattr(command_node, 'command') else None
                    if cmd_name:
                        cmd_meta = COMMAND_METADATA.get(cmd_name)
                        if cmd_meta and cmd_meta.return_type_info:
                            return_type = cmd_meta.return_type_info.semantic_type
                            # 互換性をチェック
                            compatible_types = TypeSystem.COMPATIBILITY_MATRIX.get(first_arg_type, {first_arg_type})
                            if return_type not in compatible_types:
                                # 互換性がない場合、再試行（次のSemanticTypeを試す）
                                continue

                return command_node

        # すべてのSemanticTypeで失敗した場合、④return（失敗）
        raise ValueError("算術演算の第2引数生成に失敗しました（すべてのSemanticTypeで再試行が発生）")

    def _generate_logical_first_arg(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        is_in_condition_nesting: bool,
        parent_command: Optional[str] = None
    ) -> Node:
        """論理演算の第1引数を生成

        新しいルール:
        - SemanticType選択: SemanticType.BOOL型のみ
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - 無意味な論理演算防止のため、literal_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら失敗で④return
        - 変数、リテラル値、コマンドを重み付き選択。
        - →②に進む
        """
        selected_type = SemanticType.BOOL

        # 重みを初期化
        literal_prob = arg_schema.literal_prob
        variable_prob = arg_schema.variable_prob
        nested_prob = arg_schema.nested_prob

        # 無意味な論理演算防止のため、literal_prob *= 0.0
        literal_prob = 0.0

        # CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
        if is_in_condition_nesting:
            variable_prob = 0.0

        # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        if no_new_vars_mode:
            existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                selected_type, is_array=False, context=context
            )
            if not existing_vars:
                variable_prob = 0.0

        # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        compatible_commands = self._get_compatible_nested_commands(
            arg_schema.allowed_nested_commands,
            selected_type,
            exclude_arrays=False,
            expected_is_array=False,
            exclude_arithmetic=False,
            context=context
        )
        if not compatible_commands:
            nested_prob = 0.0
        else:
            # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
            if self._check_all_commands_have_zero_weight(
                compatible_commands, context, nesting_depth, effective_max_nesting_depth
            ):
                nested_prob = 0.0

        # ネスト制限に達している場合、nested_prob *= 0.0
        if nesting_depth >= effective_max_nesting_depth:
            nested_prob = 0.0

        # すべての重みが0.0の場合、失敗で④return
        if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
            raise ValueError("論理演算の第1引数生成に失敗しました（すべての重みが0.0）")

        # 重み付き選択
        choice = random.choices(
            ['literal', 'variable', 'nested'],
            weights=[literal_prob, variable_prob, nested_prob],
            k=1
        )[0]

        # 選択に応じて生成（②の処理）
        if choice == 'nested':
            # コマンドを生成
            return self._generate_command_node(
                selected_type, arg_schema, context, ctx_dict,
                nesting_depth, effective_max_nesting_depth, no_new_vars_mode
            )
        elif choice == 'variable':
            new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
            return self.node_generators._generate_variable_argument(
                new_target_type, arg_schema, context, ctx_dict,
                nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                parent_command=parent_command
            )
        else:
            # literal_prob=0.0なので、ここには来ないはず
            raise ValueError("論理演算の第1引数でリテラル値が選択されました（literal_prob=0.0）")

    def _generate_logical_second_arg(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        is_in_condition_nesting: bool,
        parent_command: Optional[str] = None
    ) -> Node:
        """論理演算の第2引数を生成

        新しいルール:
        - SemanticType選択: SemanticType.BOOL型のみ
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - 無意味な論理演算防止のため、literal_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0
        - CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
        - 変数、リテラル値、コマンドを重みがすべて、0.0なら失敗で④return
        - 変数、リテラル値、コマンドを重み付き選択。
        - →②に進む
        """
        selected_type = SemanticType.BOOL

        # 重みを初期化
        literal_prob = arg_schema.literal_prob
        variable_prob = arg_schema.variable_prob
        nested_prob = arg_schema.nested_prob

        # 無意味な論理演算防止のため、literal_prob *= 0.0
        literal_prob = 0.0

        # CONDITION_ARGの引数ネスト内の場合、variable_prob *= 0.0
        if is_in_condition_nesting:
            variable_prob = 0.0

        # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        if no_new_vars_mode:
            existing_vars = context.variable_manager.get_compatible_variables_for_assignment(
                selected_type, is_array=False, context=context
            )
            if not existing_vars:
                variable_prob = 0.0

        # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        compatible_commands = self._get_compatible_nested_commands(
            arg_schema.allowed_nested_commands,
            selected_type,
            exclude_arrays=False,
            expected_is_array=False,
            exclude_arithmetic=False,
            context=context
        )
        if not compatible_commands:
            nested_prob = 0.0
        else:
            # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
            if self._check_all_commands_have_zero_weight(
                compatible_commands, context, nesting_depth, effective_max_nesting_depth
            ):
                nested_prob = 0.0

        # ネスト制限に達している場合、nested_prob *= 0.0
        if nesting_depth >= effective_max_nesting_depth:
            nested_prob = 0.0

        # すべての重みが0.0の場合、失敗で④return
        if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
            raise ValueError("論理演算の第2引数生成に失敗しました（すべての重みが0.0）")

        # 重み付き選択
        choice = random.choices(
            ['literal', 'variable', 'nested'],
            weights=[literal_prob, variable_prob, nested_prob],
            k=1
        )[0]

        # 選択に応じて生成（②の処理）
        if choice == 'nested':
            # コマンドを生成
            return self._generate_command_node(
                selected_type, arg_schema, context, ctx_dict,
                nesting_depth, effective_max_nesting_depth, no_new_vars_mode
            )
        elif choice == 'variable':
            new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=False)
            return self.node_generators._generate_variable_argument(
                new_target_type, arg_schema, context, ctx_dict,
                nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                parent_command=parent_command
            )
        else:
            # literal_prob=0.0なので、ここには来ないはず
            raise ValueError("論理演算の第2引数でリテラル値が選択されました（literal_prob=0.0）")

    def _generate_normal_argument(
        self,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        target_type_info: TypeInfo,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        is_in_condition_nesting: bool,
        parent_command: Optional[str] = None
    ) -> Node:
        """通常の引数を生成

        新しいルール:
        - SemanticType選択: ターゲット型と互換性のある型（COMPATIBILITY_MATRIXから取得）
          - 選択確率: ターゲット型と同型: 90%の確率、それ以外の互換型: 10%の確率（均等配分）
        - no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
        - 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
        - ネスト制限に達している場合、nested_prob *= 0.0

        分岐:
        - if FILTERのCONDITION_ARGの引数ネスト内でかつターゲット型がオブジェクト単体の場合
          - プレースホルダー変数$objを選択
          - →③に進む
        - else if SORT_BYの第２引数ネスト内でかつターゲット型がオブジェクト単体の場合
          - プレースホルダー変数$objを選択
          - →③に進む
        - else if MATCH_PAIRSの第３引数（CONDITION_ARG）のネスト内でかつターゲット型がオブジェクト単体の場合
          - プレースホルダー変数$obj1または$obj2をランダムに選択
          - →③に進む
        - else
          - 変数、リテラル値、コマンドを重みがすべて、0.0なら最初に戻り、他のSemanticType選択
          - 変数、リテラル値、コマンドを重み付き選択。
          - リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
          - →②に進む
        """
        # ターゲット型を取得
        target_type = target_type_info.semantic_type if target_type_info else None

        if target_type is None:
            # ターゲット型が不明な場合、デフォルト型を使用
            # ここでは、arg_schemaから取得を試みる
            if hasattr(arg_schema, 'type_info') and arg_schema.type_info:
                target_type = arg_schema.type_info.semantic_type

        # 分岐: FILTER/SORT_BY/MATCH_PAIRSのCONDITION_ARGの引数ネスト内でかつターゲット型がオブジェクト単体の場合
        if (is_in_condition_nesting and
            self._is_target_type_object_single(target_type_info)):
            # プレースホルダー変数を選択
            # FILTER/SORT_BYの場合: $obj
            # MATCH_PAIRSの場合: $obj1または$obj2
            if hasattr(context, 'placeholder_tracking'):
                if context.placeholder_tracking.context_type == 'MATCH_PAIRS':
                    # MATCH_PAIRSの場合、$obj1または$obj2をランダムに選択
                    placeholder_name = random.choice(['$obj1', '$obj2'])
                else:
                    # FILTER/SORT_BYの場合、$objを選択
                    placeholder_name = '$obj'

                # プレースホルダーノードを生成
                return self.node_generators._generate_placeholder_node(
                    placeholder_name, context, ctx_dict
                )

        # 通常の処理
        if target_type is None:
            # ターゲット型が不明な場合、デフォルト型を使用
            target_type = SemanticType.OBJECT

        # arg_schemaからis_arrayフラグを取得（LENコマンドなど配列引数の場合に重要）
        target_is_array = arg_schema.type_info.is_array if arg_schema and arg_schema.type_info else False

        # 互換性のある型を取得
        compatible_types = list(TypeSystem.COMPATIBILITY_MATRIX.get(target_type, {target_type}))

        # 最大再試行回数
        max_retries = len(compatible_types) * SEMANTIC_TYPE_SELECTION_RETRY_MULTIPLIER

        # 詳細ログ用: 試行履歴を記録
        attempt_history = []

        for retry in range(max_retries):
            # SemanticType選択（90%で同型、10%で互換型）
            if random.random() < 0.9 and target_type in compatible_types:
                selected_type = target_type
            else:
                # 互換型から選択（同型を除く）
                other_types = [t for t in compatible_types if t != target_type]
                if other_types:
                    selected_type = random.choice(other_types)
                else:
                    selected_type = target_type

            # 重みを初期化
            literal_prob = arg_schema.literal_prob
            variable_prob = arg_schema.variable_prob
            nested_prob = arg_schema.nested_prob

            # no_new_vars_mode=True の場合、選択されたSemanticTypeに既存変数がなければ、variable_prob *= 0.0
            existing_vars_list = []
            if no_new_vars_mode:
                existing_vars_list = context.variable_manager.get_compatible_variables_for_assignment(
                    selected_type, is_array=target_is_array, context=context
                )
                if not existing_vars_list:
                    variable_prob = 0.0

            # 選択されたSemanticTypeに、利用可能なコマンドがなければ、nested_prob *= 0.0
            compatible_commands = self._get_compatible_nested_commands(
                arg_schema.allowed_nested_commands,
                selected_type,
                exclude_arrays=False,
                expected_is_array=target_is_array,  # arg_schemaのis_arrayフラグを使用
                exclude_arithmetic=False,
                context=context
            )
            if not compatible_commands:
                nested_prob = 0.0
            else:
                # 重みが0.0になるコマンドをチェックし、すべてのコマンドの重みが0.0になる可能性がある場合、nested_prob *= 0.0
                if self._check_all_commands_have_zero_weight(
                    compatible_commands, context, nesting_depth, effective_max_nesting_depth
                ):
                    nested_prob = 0.0

            # ネスト制限に達している場合、nested_prob *= 0.0
            if nesting_depth >= effective_max_nesting_depth:
                nested_prob = 0.0

            # 詳細ログ: 試行情報を記録
            attempt_info = {
                'retry': retry,
                'selected_type': selected_type.name if hasattr(selected_type, 'name') else str(selected_type),
                'literal_prob': literal_prob,
                'variable_prob': variable_prob,
                'nested_prob': nested_prob,
                'all_zero': literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0,
                'existing_vars_count': len(existing_vars_list) if no_new_vars_mode else None,
                'existing_vars': existing_vars_list[:5] if no_new_vars_mode and existing_vars_list else None,  # 最初の5個のみ
                'compatible_commands_count': len(compatible_commands) if compatible_commands else 0,
                'compatible_commands': compatible_commands if compatible_commands else None,  # すべて表示
                'nesting_depth': nesting_depth,
                'effective_max_nesting_depth': effective_max_nesting_depth,
                'nesting_limit_reached': nesting_depth >= effective_max_nesting_depth
            }
            attempt_history.append(attempt_info)

            # すべての重みが0.0の場合、別のSemanticTypeを選択
            if literal_prob == 0.0 and variable_prob == 0.0 and nested_prob == 0.0:
                continue  # 次のSemanticTypeを試す

            # 重み付き選択
            choice = random.choices(
                ['literal', 'variable', 'nested'],
                weights=[literal_prob, variable_prob, nested_prob],
                k=1
            )[0]

            # 選択に応じて生成（②の処理）
            if choice == 'literal':
                # リテラル値が選ばれた場合、ターゲット型と異なるSemanticTypeが選択されていた場合、ターゲット型と同型に変更
                if selected_type != target_type:
                    selected_type = target_type
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=target_is_array)
                return self.node_generators._generate_value_argument(
                    arg_schema, context, ctx_dict, target_type_info=new_target_type
                )
            elif choice == 'variable':
                new_target_type = TypeInfo.create_from_semantic_type(selected_type, is_array=target_is_array)
                return self.node_generators._generate_variable_argument(
                    new_target_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    parent_command=parent_command
                )
            else:
                # コマンドを生成
                return self._generate_command_node(
                    selected_type, arg_schema, context, ctx_dict,
                    nesting_depth, effective_max_nesting_depth, no_new_vars_mode,
                    expected_is_array=target_is_array  # arg_schemaのis_arrayフラグを使用
                )

        # すべてのSemanticTypeで失敗した場合、④return（失敗）
        # 詳細ログを生成
        error_details = {
            'parent_command': parent_command,
            'target_type': target_type.name if hasattr(target_type, 'name') else str(target_type),
            'target_is_array': target_is_array,
            'compatible_types': [t.name if hasattr(t, 'name') else str(t) for t in compatible_types],
            'max_retries': max_retries,
            'no_new_vars_mode': no_new_vars_mode,
            'is_in_condition_nesting': is_in_condition_nesting,
            'nesting_depth': nesting_depth,
            'effective_max_nesting_depth': effective_max_nesting_depth,
            'arg_schema_literal_prob': arg_schema.literal_prob if arg_schema else None,
            'arg_schema_variable_prob': arg_schema.variable_prob if arg_schema else None,
            'arg_schema_nested_prob': arg_schema.nested_prob if arg_schema else None,
            'attempt_history': attempt_history
        }

        # エラーメッセージに詳細情報を追加
        error_msg = (
            f"通常の引数生成に失敗しました（すべてのSemanticTypeで再試行が発生）\n"
            f"詳細: parent_command={parent_command}, target_type={target_type.name if hasattr(target_type, 'name') else str(target_type)}, "
            f"target_is_array={target_is_array}, compatible_types={[t.name if hasattr(t, 'name') else str(t) for t in compatible_types]}, "
            f"max_retries={max_retries}, no_new_vars_mode={no_new_vars_mode}, "
            f"nesting_depth={nesting_depth}/{effective_max_nesting_depth}, "
            f"arg_schema_probs=(literal={arg_schema.literal_prob if arg_schema else None}, "
            f"variable={arg_schema.variable_prob if arg_schema else None}, "
            f"nested={arg_schema.nested_prob if arg_schema else None})\n"
            f"試行履歴: {len(attempt_history)}回の試行、すべての重みが0.0になった試行: {sum(1 for a in attempt_history if a['all_zero'])}回"
        )

        # 詳細ログをstderrに出力
        if self.enable_debug_logs:
            print(f"[DEBUG] _generate_normal_argument失敗: {error_msg}", file=sys.stderr, flush=True)
            print(f"[DEBUG] _generate_normal_argument詳細: {error_details}", file=sys.stderr, flush=True)

        raise ValueError(error_msg)

    def _generate_command_node(
        self,
        selected_type: SemanticType,
        arg_schema: ArgumentSchema,
        context: ProgramContext,
        ctx_dict: Dict,
        nesting_depth: int,
        effective_max_nesting_depth: int,
        no_new_vars_mode: bool,
        expected_is_array: bool = False
    ) -> Node:
        """コマンドノードを生成

        新しいルール:
        - そのSemanticTypeで使用可能かコマンド一覧を取得
        - コマンドごとの重み付き選択
          - ネスト制限-2に達している場合、論理演算AND,ORの重みを×０
          - ネスト制限-1に達している場合、比較演算と算術演算の重みを×０
          - FILTERのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
          - MATCH_PAIRSのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
          - SORT_BYのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整

        flag = False
        for 再生成の最大試行回数:
          for コマンドの引数でループ:
            →＜引数作成関数＞に進む
          →＜最終チェック関数＞へ
          最終チェック成功の場合→flag = True break

        if flag == False:
            →最終チェック失敗で④return
        else:
            →③に進む
        """
        # コマンド引数生成開始時に重複回避用のセットをリセット
        # 重複回避の範囲は、そのコマンドの引数での直接使用に限定
        context.used_variables_in_current_command = set()

        # 使用可能なコマンド一覧を取得
        compatible_commands = self._get_compatible_nested_commands(
            arg_schema.allowed_nested_commands,
            selected_type,
            exclude_arrays=False,
            expected_is_array=expected_is_array,  # パラメータとして受け取ったexpected_is_arrayを使用
            exclude_arithmetic=False,
            context=context
        )

        if not compatible_commands:
            raise ValueError(f"使用可能なコマンドがありません（SemanticType: {selected_type}）")

        # 最初の試行開始時の変数状態をスナップショットとして保存
        # これにより、各再試行で追加された変数のみを正確に削除できる
        previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
        previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

        # 最大再試行回数
        for attempt in range(MAX_COMMAND_GENERATION_RETRIES):
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
                # 理由: 引数生成処理では変数定義ノード（InitializationNode, AssignmentNode）が生成されないため、
                # 定義済み変数（is_defined=True）が作成される可能性はゼロ
                # そのため、使用のみの変数（未定義）だけを削除する
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
                    # unavailable_variablesからも削除（未定義変数の追跡から削除）
                    for var_name in vars_to_remove:
                        context.variable_manager.remove_unavailable_variable(var_name)
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

                # argument_nesting_command_stackの更新をリセット（現在のnesting_depthのみ）
                if nesting_depth in context.argument_nesting_command_stack:
                    # 前回の試行で追加されたコマンドを削除（ただし、親のコマンドは保持）
                    # 現在のnesting_depthのスタックをクリア
                    context.argument_nesting_command_stack[nesting_depth] = []
                # used_variables_in_current_commandもリセット
                context.used_variables_in_current_command = set()
                # 比較演算・算術演算の第1引数の型情報もリセット（ケース3対策）
                context.current_proportional_first_arg_type = None
                context.current_arithmetic_first_arg_type = None

            # 現在の試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
            previous_attempt_start_variable_names = set(context.variable_manager.get_all_variable_names())
            previous_attempt_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

            # コマンドを選択（重み付き）
            selected_cmd = self._select_command_with_weights(
                compatible_commands, context, nesting_depth, effective_max_nesting_depth
            )

            if selected_cmd is None:
                if self.enable_debug_logs:
                    print(f"[DEBUG] _generate_command_node: 試行{attempt+1}/{MAX_COMMAND_GENERATION_RETRIES}: コマンド選択失敗（すべての重みが0.0）、compatible_commands={compatible_commands[:5]}...", file=sys.stderr, flush=True)
                continue

            # コマンドのメタデータを取得
            cmd_metadata = COMMAND_METADATA.get(selected_cmd)
            if not cmd_metadata:
                if self.enable_debug_logs:
                    print(f"[DEBUG] _generate_command_node: 試行{attempt+1}/{MAX_COMMAND_GENERATION_RETRIES}: コマンドメタデータが見つかりません、selected_cmd={selected_cmd}", file=sys.stderr, flush=True)
                continue

            # コマンドの引数を生成
            try:
                nested_args = []
                # 比較演算・算術演算の場合、第1引数の型を保存
                first_arg_type_info = None

                # argument_nesting_command_stackを更新（緩和計算用）
                if nesting_depth not in context.argument_nesting_command_stack:
                    context.argument_nesting_command_stack[nesting_depth] = []
                if selected_cmd not in context.argument_nesting_command_stack[nesting_depth]:
                    context.argument_nesting_command_stack[nesting_depth].append(selected_cmd)

                # 比較演算・算術演算の場合、第1引数の型情報を保存（第2引数生成時に使用）
                saved_first_arg_type = None
                saved_arithmetic_first_arg_type = None

                for i, nested_arg_schema in enumerate(cmd_metadata.arguments):
                    nested_target_type_info = self._get_expected_argument_type_info(
                        selected_cmd, i, nested_arg_schema, target_type=selected_type
                    )

                    # 比較演算・算術演算の場合、第1引数の型を保存（第2引数生成時に使用）
                    if i == 0 and (selected_cmd in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS', 'ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']):
                        # 第1引数の型を保存（第2引数生成時に使用）
                        context.current_proportional_first_arg_type = nested_target_type_info.semantic_type if nested_target_type_info else None
                        context.current_arithmetic_first_arg_type = nested_target_type_info.semantic_type if nested_target_type_info else None
                        first_arg_type_info = nested_target_type_info
                        saved_first_arg_type = nested_target_type_info.semantic_type if nested_target_type_info else None
                        saved_arithmetic_first_arg_type = nested_target_type_info.semantic_type if nested_target_type_info else None

                    # 引数を作成
                    # 比較演算・算術演算の第2引数の場合、保存した第1引数の型情報を直接渡す（ネストされた比較演算・算術演算による上書きを防ぐ）
                    first_arg_type_for_arg = None
                    if i == 1 and selected_cmd in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']:
                        first_arg_type_for_arg = saved_first_arg_type
                    elif i == 1 and selected_cmd in ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']:
                        first_arg_type_for_arg = saved_arithmetic_first_arg_type

                    nested_arg_node = self._generate_argument_with_new_rules(
                        nested_arg_schema, context, ctx_dict,
                        nested_target_type_info,
                        nesting_depth + 1, effective_max_nesting_depth,
                        no_new_vars_mode,
                        arg_index=i,
                        parent_command=selected_cmd,
                        first_arg_type=first_arg_type_for_arg
                    )
                    nested_args.append(nested_arg_node)

                    # 比較演算・算術演算の場合、第1引数の型とリテラル値かどうかを更新（実際に生成されたノードから取得）
                    if i == 0 and (selected_cmd in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS', 'ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']):
                        # 生成されたノードから型を取得
                        if isinstance(nested_arg_node, LiteralNode):
                            # リテラルノードの場合
                            context.current_arithmetic_first_arg_is_literal = True
                            context.current_proportional_first_arg_is_literal = True
                            # リテラルノードの型は、実際のターゲット型（nested_target_type_info）を使用
                            # （COMMAND_METADATAのダミー値ではなく、実際に生成されたリテラル値の型を使用）
                            if nested_target_type_info:
                                first_arg_type_info = nested_target_type_info
                                context.current_proportional_first_arg_type = first_arg_type_info.semantic_type
                                context.current_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                                saved_first_arg_type = first_arg_type_info.semantic_type
                                saved_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                            else:
                                # フォールバック: コマンドのメタデータから取得（算術演算・比較演算以外の場合）
                                cmd_meta = COMMAND_METADATA.get(selected_cmd)
                                if cmd_meta and cmd_meta.arguments and len(cmd_meta.arguments) > 0:
                                    first_arg_schema = cmd_meta.arguments[0]
                                    if hasattr(first_arg_schema, 'type_info') and first_arg_schema.type_info:
                                        first_arg_type_info = first_arg_schema.type_info
                                        context.current_proportional_first_arg_type = first_arg_type_info.semantic_type
                                        context.current_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                                        saved_first_arg_type = first_arg_type_info.semantic_type
                                        saved_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                        elif isinstance(nested_arg_node, CommandNode):
                            # コマンドノードの場合、戻り値の型を取得
                            context.current_arithmetic_first_arg_is_literal = False
                            context.current_proportional_first_arg_is_literal = False
                            if hasattr(nested_arg_node, 'return_type_info') and nested_arg_node.return_type_info:
                                first_arg_type_info = nested_arg_node.return_type_info
                                context.current_proportional_first_arg_type = first_arg_type_info.semantic_type
                                context.current_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                                saved_first_arg_type = first_arg_type_info.semantic_type
                                saved_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                            else:
                                # コマンドのメタデータから戻り値の型を取得（nested_arg_nodeのコマンド名を使用）
                                nested_cmd_name = nested_arg_node.command if hasattr(nested_arg_node, 'command') else None
                                if nested_cmd_name:
                                    nested_cmd_meta = COMMAND_METADATA.get(nested_cmd_name)
                                    if nested_cmd_meta and nested_cmd_meta.return_type_info:
                                        first_arg_type_info = nested_cmd_meta.return_type_info
                                        context.current_proportional_first_arg_type = first_arg_type_info.semantic_type
                                        context.current_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                                        saved_first_arg_type = first_arg_type_info.semantic_type
                                        saved_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                        elif isinstance(nested_arg_node, VariableNode):
                            # 変数ノードの場合、変数の型情報を取得
                            context.current_arithmetic_first_arg_is_literal = False
                            context.current_proportional_first_arg_is_literal = False
                            var_info = context.variable_manager.get_variable_info(nested_arg_node.name)
                            if var_info and 'type_info' in var_info:
                                first_arg_type_info = var_info['type_info']
                                context.current_proportional_first_arg_type = first_arg_type_info.semantic_type
                                context.current_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                                saved_first_arg_type = first_arg_type_info.semantic_type
                                saved_arithmetic_first_arg_type = first_arg_type_info.semantic_type
                        else:
                            # その他のノードタイプの場合、型情報を取得できない可能性がある
                            # ケース2対策: ノードから型情報を取得できない場合、再試行を促す
                            if saved_first_arg_type is None:
                                # 型情報を取得できない場合、再試行（次のコマンド選択を試す）
                                raise ValueError(f"第1引数の型情報を取得できません（ノードタイプ: {type(nested_arg_node).__name__}）")

                    # 比較演算の第2引数生成時、保存した第1引数の型情報を直接渡す（ネストされた比較演算による上書きを防ぐ）
                    if i == 1 and selected_cmd in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']:
                        # 第2引数生成時に、保存した第1引数の型情報を使用
                        # ただし、_generate_argument_with_new_rules内で呼び出されるため、
                        # ここではcontextに保存された値を使用する（既に更新済み）
                        pass

                # 最終チェック関数を実行（再生成に必要な情報を渡す）
                final_check_result = self._final_check_function(
                    selected_cmd, nested_args, context,
                    cmd_metadata=cmd_metadata,
                    ctx_dict=ctx_dict,
                    nesting_depth=nesting_depth,
                    effective_max_nesting_depth=effective_max_nesting_depth,
                    no_new_vars_mode=no_new_vars_mode
                )
                if final_check_result:
                    # 最終チェック成功
                    # CommandNodeにreturn_type_infoを設定（COMMAND_METADATAから取得）
                    return_type_info = None
                    if cmd_metadata and cmd_metadata.return_type_info:
                        return_type_info = cmd_metadata.return_type_info
                    command_node = CommandNode(selected_cmd, nested_args, ctx_dict, return_type_info=return_type_info)

                    # GET_INPUT_GRID_SIZEが非配列期待で選択された場合、配列要素アクセスを追加
                    if selected_cmd == 'GET_INPUT_GRID_SIZE' and cmd_metadata.return_type_info.is_array and not expected_is_array:
                        # expected_is_array=Falseの場合、配列要素アクセスを追加
                        # [0]または[1]をランダムに選択
                        index = random.choice([0, 1])
                        return ArrayElementAccessNode(command_node, index, ctx_dict)

                    return command_node
                else:
                    if self.enable_debug_logs:
                        print(f"[DEBUG] _generate_command_node: 試行{attempt+1}/{MAX_COMMAND_GENERATION_RETRIES}: 最終チェック失敗、selected_cmd={selected_cmd}", file=sys.stderr, flush=True)
            except Exception as e:
                # エラーが発生した場合、次の試行へ
                if self.enable_debug_logs:
                    print(f"[DEBUG] _generate_command_node: 試行{attempt+1}/{MAX_COMMAND_GENERATION_RETRIES}: 例外発生、selected_cmd={selected_cmd if 'selected_cmd' in locals() else 'N/A'}, error={type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue

        # すべての試行が失敗した場合、④return（失敗）
        if self.enable_debug_logs:
            print(f"[DEBUG] _generate_command_node: すべての試行が失敗、compatible_commands={compatible_commands[:10]}, selected_type={selected_type}, nesting_depth={nesting_depth}, effective_max_nesting_depth={effective_max_nesting_depth}", file=sys.stderr, flush=True)
        raise ValueError(f"コマンドノード生成に失敗しました（{MAX_COMMAND_GENERATION_RETRIES}回試行）")

    def _calculate_command_weight(
        self,
        cmd: str,
        context: ProgramContext,
        nesting_depth: int,
        effective_max_nesting_depth: int
    ) -> float:
        """コマンドの重みを計算（_select_command_with_weightsと同じロジック）

        Args:
            cmd: コマンド名
            context: プログラムコンテキスト
            nesting_depth: 現在のネスト深度
            effective_max_nesting_depth: 実効的な最大ネスト深度

        Returns:
            計算された重み
        """
        cmd_metadata = COMMAND_METADATA.get(cmd)
        if not cmd_metadata:
            return 0.0

        weight = cmd_metadata.base_weight

        # ネスト制限-2に達している場合、論理演算AND,ORの重みを×０
        if (effective_max_nesting_depth - nesting_depth) <= 2:
            if cmd in ['AND', 'OR']:
                weight = 0.0

        # ネスト制限-1に達している場合、比較演算と算術演算の重みを×０
        if (effective_max_nesting_depth - nesting_depth) <= 1:
            if cmd in ['EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS']:
                weight = 0.0
            if cmd in ['ADD', 'SUB', 'MULTIPLY', 'DIVIDE', 'MOD']:
                weight = 0.0

        # FILTER/MATCH_PAIRS/SORT_BYのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
        # 現在のコマンドスタックを確認
        is_in_filter_condition = False
        is_in_match_pairs_condition = False
        is_in_sort_by_condition = False

        # argument_nesting_command_stackから確認
        for depth in range(nesting_depth + 1):
            if depth in context.argument_nesting_command_stack:
                commands = context.argument_nesting_command_stack[depth]
                if 'FILTER' in commands:
                    is_in_filter_condition = True
                if 'MATCH_PAIRS' in commands:
                    is_in_match_pairs_condition = True
                if 'SORT_BY' in commands:
                    is_in_sort_by_condition = True

        # context.current_commandも確認
        if hasattr(context, 'current_command') and context.current_command:
            if context.current_command == 'FILTER':
                is_in_filter_condition = True
            elif context.current_command == 'MATCH_PAIRS':
                is_in_match_pairs_condition = True
            elif context.current_command == 'SORT_BY':
                is_in_sort_by_condition = True

        # 重み調整（既存の実装を参考に、詳細に実装）
        if is_in_filter_condition:
            # FILTER内での重み調整（既存の実装を参考）
            filter_weight_multipliers = {
                # GET関数（基本情報）
                'GET_COLOR': 3.0, 'GET_SIZE': 3.0, 'GET_WIDTH': 3.0, 'GET_HEIGHT': 3.0,
                # GET関数（位置情報）
                'GET_X': 2.2, 'GET_Y': 2.2, 'GET_CENTER_X': 2.2, 'GET_CENTER_Y': 2.2,
                'GET_MAX_X': 2.2, 'GET_MAX_Y': 2.2,
                # GET関数（形状情報）
                'COUNT_HOLES': 1.5, 'GET_SYMMETRY_SCORE': 1.5, 'GET_ASPECT_RATIO': 1.5,
                'GET_DENSITY': 1.5, 'GET_RECTANGLE_TYPE': 1.5, 'GET_LINE_TYPE': 1.5,
                'GET_CENTROID': 1.5,
                # 比較演算（比例演算）
                'EQUAL': 1.0, 'NOT_EQUAL': 1.0, 'GREATER': 1.0, 'LESS': 1.0,
                # 論理演算
                'AND': 0.5, 'OR': 0.5,
            }
            if cmd in filter_weight_multipliers:
                weight *= filter_weight_multipliers[cmd]
            else:
                # 部分プログラム生成で使えないコマンドは重みを0にする
                weight = 0.0
        elif is_in_match_pairs_condition:
            # MATCH_PAIRS内での重み調整（FILTERと同様）
            match_pairs_weight_multipliers = {
                # GET関数（基本情報）
                'GET_COLOR': 3.0, 'GET_SIZE': 3.0, 'GET_WIDTH': 3.0, 'GET_HEIGHT': 3.0,
                # GET関数（位置情報）
                'GET_X': 2.2, 'GET_Y': 2.2, 'GET_CENTER_X': 2.2, 'GET_CENTER_Y': 2.2,
                'GET_MAX_X': 2.2, 'GET_MAX_Y': 2.2,
                # GET関数（形状情報）
                'COUNT_HOLES': 1.5, 'GET_SYMMETRY_SCORE': 1.5, 'GET_ASPECT_RATIO': 1.5,
                'GET_DENSITY': 1.5, 'GET_RECTANGLE_TYPE': 1.5, 'GET_LINE_TYPE': 1.5,
                'GET_CENTROID': 1.5,
                # GET関数（オブジェクト間の関係）
                'GET_DISTANCE': 3.0, 'GET_X_DISTANCE': 2.5, 'GET_Y_DISTANCE': 2.5,
                'COUNT_ADJACENT': 0.5, 'COUNT_OVERLAP': 0.5, 'GET_DIRECTION': 1.0,
                # 比較演算（形状・色の一致判定）
                'IS_SAME_SHAPE': 0.25, 'IS_SAME_STRUCT': 0.25, 'IS_IDENTICAL': 0.25,
                'IS_INSIDE': 1.5,
                # 算術演算
                'ADD': 1.0, 'SUB': 1.0, 'MULTIPLY': 1.0, 'DIVIDE': 1.0, 'MOD': 1.0,
                # 比較演算（比例演算）
                'EQUAL': 1.0, 'NOT_EQUAL': 1.0, 'GREATER': 1.0, 'LESS': 1.0,
                # 論理演算
                'AND': 0.5, 'OR': 0.5,
            }
            if cmd in match_pairs_weight_multipliers:
                weight *= match_pairs_weight_multipliers[cmd]
            else:
                # 部分プログラム生成で使えないコマンドは重みを0にする
                weight = 0.0
        elif is_in_sort_by_condition:
            # SORT_BY内での重み調整（FILTERと同様）
            sort_by_weight_multipliers = {
                # GET関数（基本情報）
                'GET_COLOR': 3.0, 'GET_SIZE': 3.0, 'GET_WIDTH': 3.0, 'GET_HEIGHT': 3.0,
                # GET関数（位置情報）
                'GET_X': 2.2, 'GET_Y': 2.2, 'GET_CENTER_X': 2.2, 'GET_CENTER_Y': 2.2,
                'GET_MAX_X': 2.2, 'GET_MAX_Y': 2.2,
                # GET関数（形状情報）
                'COUNT_HOLES': 1.5, 'GET_SYMMETRY_SCORE': 1.5, 'GET_ASPECT_RATIO': 1.5,
                'GET_DENSITY': 1.5, 'GET_RECTANGLE_TYPE': 1.5, 'GET_LINE_TYPE': 1.5,
                'GET_CENTROID': 1.5,
                # GET関数（オブジェクト間の関係）
                # 注意: SORT_BYでは$objが1つだけのため、これらのコマンドは通常使用できないが、
                # ネストされたコマンド内で使用される可能性があるため、低めの重みを設定
                'GET_DISTANCE': 1.0, 'GET_X_DISTANCE': 1.0, 'GET_Y_DISTANCE': 1.0,
                'COUNT_ADJACENT': 1.0, 'COUNT_OVERLAP': 1.0,
                # 算術演算
                'ADD': 1.0, 'SUB': 1.0, 'MULTIPLY': 1.0, 'DIVIDE': 1.0, 'MOD': 1.0,
                # 比較演算（比例演算）
                'EQUAL': 1.0, 'NOT_EQUAL': 1.0, 'GREATER': 1.0, 'LESS': 1.0,
                # 論理演算
                'AND': 0.5, 'OR': 0.5,
            }
            if cmd in sort_by_weight_multipliers:
                weight *= sort_by_weight_multipliers[cmd]
            else:
                # 部分プログラム生成で使えないコマンドは重みを0にする
                weight = 0.0

        return weight

    def _check_all_commands_have_zero_weight(
        self,
        compatible_commands: List[str],
        context: ProgramContext,
        nesting_depth: int,
        effective_max_nesting_depth: int
    ) -> bool:
        """すべてのコマンドの重みが0.0になる可能性があるかチェック

        Args:
            compatible_commands: 互換性のあるコマンドのリスト
            context: プログラムコンテキスト
            nesting_depth: 現在のネスト深度
            effective_max_nesting_depth: 実効的な最大ネスト深度

        Returns:
            すべてのコマンドの重みが0.0になる可能性がある場合True
        """
        if not compatible_commands:
            return True

        # すべてのコマンドの重みを計算し、すべてが0.0かどうかをチェック
        for cmd in compatible_commands:
            weight = self._calculate_command_weight(
                cmd, context, nesting_depth, effective_max_nesting_depth
            )
            if weight > 0.0:
                # 重みが0.0でないコマンドが見つかった
                return False

        # すべてのコマンドの重みが0.0の場合、Trueを返す
        return True

    def _select_command_with_weights(
        self,
        compatible_commands: List[str],
        context: ProgramContext,
        nesting_depth: int,
        effective_max_nesting_depth: int
    ) -> Optional[str]:
        """コマンドを重み付きで選択

        新しいルール:
        - ネスト制限-2に達している場合、論理演算AND,ORの重みを×０
        - ネスト制限-1に達している場合、比較演算と算術演算の重みを×０
        - FILTERのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
        - MATCH_PAIRSのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
        - SORT_BYのCONDITION_ARGの引数ネスト内の場合、コマンドの重み調整
        """
        weights = []
        for cmd in compatible_commands:
            weight = self._calculate_command_weight(
                cmd, context, nesting_depth, effective_max_nesting_depth
            )
            weights.append(weight)

        # すべての重みが0.0の場合、Noneを返す
        if all(w == 0.0 for w in weights):
            return None

        # 重み付き選択
        return random.choices(compatible_commands, weights=weights, k=1)[0]

    def _final_check_function(
        self,
        cmd_name: str,
        args: List[Node],
        context: ProgramContext,
        cmd_metadata: Optional[CommandMetadata] = None,
        ctx_dict: Optional[Dict] = None,
        nesting_depth: int = 0,
        effective_max_nesting_depth: int = 0,
        no_new_vars_mode: bool = False
    ) -> bool:
        """最終チェック関数

        新しいルールに基づいた最終チェックを実行します。
        設計ドキュメントに従い、再生成も試みます。

        Args:
            cmd_name: コマンド名
            args: 引数ノードのリスト（変更可能）
            context: プログラムコンテキスト
            cmd_metadata: コマンドのメタデータ（再生成に必要）
            ctx_dict: コンテキスト辞書（再生成に必要）
            nesting_depth: ネスト深度（再生成に必要）
            effective_max_nesting_depth: 実効的な最大ネスト深度（再生成に必要）
            no_new_vars_mode: 新しい変数を生成禁止モード（再生成に必要）

        Returns:
            最終チェックが成功した場合True
        """
        # FILTERのチェック
        if cmd_name == 'FILTER' and len(args) >= 2:
            condition_arg = args[1]  # 第2引数
            placeholders = self.node_generators._get_placeholder_variables_in_node(condition_arg)
            if '$obj' not in placeholders:
                # まず置き換えを試みる
                success, replaced_node = self._try_replace_with_placeholder(condition_arg, '$obj', context)
                if success and replaced_node is not None:
                    # 置き換えが成功した場合、args[1]を更新
                    args[1] = replaced_node
                    return True

                # 置き換えに失敗した場合、再生成を試みる
                if cmd_metadata and ctx_dict is not None and len(cmd_metadata.arguments) > 1:
                    # 再生成処理前にplaceholder_trackingを設定（オブジェクト単体引数の場合にプレースホルダー変数を確実に使用するため）
                    context.placeholder_tracking = PlaceholderTracking('FILTER')
                    # context.current_commandを設定（_is_in_condition_arg_nestingで検出されるようにするため）
                    context.current_command = 'FILTER'
                    # argument_nesting_command_stackにも追加（nesting_depthが分からないため、すべての深度に追加）
                    # 注意: 既存のスタックに追加することで、_is_in_condition_arg_nestingで確実に検出されるようにする
                    for depth in range(len(context.argument_nesting_command_stack) + 1):
                        if depth not in context.argument_nesting_command_stack:
                            context.argument_nesting_command_stack[depth] = []
                        if 'FILTER' not in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].append('FILTER')

                    used_obj = False
                    # 最初の再試行開始時の変数状態をスナップショットとして保存
                    previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                    previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                    for retry in range(MAX_FILTER_CONDITION_REGENERATION_RETRIES):
                        # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
                        if retry > 0:
                            # 現在の変数状態を取得
                            current_variable_names = set(context.variable_manager.get_all_variable_names())
                            current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                            # 前回の試行開始時以降に追加された変数を特定
                            new_variables = current_variable_names - previous_retry_start_variable_names
                            new_type_reservation_vars = current_type_reservation_vars - previous_retry_start_type_reservation_vars

                            # 削除対象の変数を決定
                            # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                            # 理由: 引数生成処理では変数定義ノード（InitializationNode, AssignmentNode）が生成されないため、
                            # 定義済み変数（is_defined=True）が作成される可能性はゼロ
                            # そのため、使用のみの変数（未定義）だけを削除する
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
                                # unavailable_variablesからも削除（未定義変数の追跡から削除）
                                for var_name in vars_to_remove:
                                    context.variable_manager.remove_unavailable_variable(var_name)
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

                        # 現在の再試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
                        previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                        previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                        # 第2引数を再生成
                        arg_schema = cmd_metadata.arguments[1]
                        nested_target_type_info = self._get_expected_argument_type_info(
                            cmd_name, 1, arg_schema
                        )
                        new_condition_arg = self._generate_argument_with_new_rules(
                            arg_schema, context, ctx_dict,
                            nested_target_type_info,
                            nesting_depth + 1, effective_max_nesting_depth,
                            no_new_vars_mode,
                            arg_index=1,
                            parent_command=cmd_name
                        )
                        args[1] = new_condition_arg

                        # $objが使われているか確認
                        placeholders = self.node_generators._get_placeholder_variables_in_node(new_condition_arg)
                        if '$obj' in placeholders:
                            used_obj = True
                            break

                        # オブジェクト単体の変数、リテラル値、コマンドがある場合、置き換えを試みる
                        success, replaced_node = self._try_replace_with_placeholder(new_condition_arg, '$obj', context)
                        if success and replaced_node is not None:
                            # 置き換えが成功した場合、args[1]を更新
                            args[1] = replaced_node
                            used_obj = True
                            break

                    # 再生成処理後にplaceholder_trackingとcurrent_commandをクリア
                    context.placeholder_tracking = None
                    context.current_command = None
                    # argument_nesting_command_stackからFILTERを削除（追加したもののみ）
                    # 注意: 元々あったFILTERも削除される可能性があるが、再生成処理中なので問題ない
                    for depth in context.argument_nesting_command_stack:
                        if 'FILTER' in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].remove('FILTER')

                    if not used_obj:
                        return False
                    return True
                else:
                    return False

        # MATCH_PAIRSのチェック
        if cmd_name == 'MATCH_PAIRS' and len(args) >= 3:
            condition_arg = args[2]  # 第3引数
            placeholders = self.node_generators._get_placeholder_variables_in_node(condition_arg)
            if '$obj1' not in placeholders or '$obj2' not in placeholders:
                # まず置き換えを試みる
                missing = []
                if '$obj1' not in placeholders:
                    missing.append('$obj1')
                if '$obj2' not in placeholders:
                    missing.append('$obj2')
                success, replaced_node = self._try_replace_with_placeholders(condition_arg, missing, context)
                if success and replaced_node is not None:
                    # 置き換えが成功した場合、args[2]を更新
                    args[2] = replaced_node
                    return True

                # 置き換えに失敗した場合、再生成を試みる
                if cmd_metadata and ctx_dict is not None and len(cmd_metadata.arguments) > 2:
                    # 再生成処理前にplaceholder_trackingを設定（オブジェクト単体引数の場合にプレースホルダー変数を確実に使用するため）
                    context.placeholder_tracking = PlaceholderTracking('MATCH_PAIRS')
                    # context.current_commandを設定（_is_in_condition_arg_nestingで検出されるようにするため）
                    context.current_command = 'MATCH_PAIRS'
                    # argument_nesting_command_stackにも追加（nesting_depthが分からないため、すべての深度に追加）
                    # 注意: 既存のスタックに追加することで、_is_in_condition_arg_nestingで確実に検出されるようにする
                    for depth in range(len(context.argument_nesting_command_stack) + 1):
                        if depth not in context.argument_nesting_command_stack:
                            context.argument_nesting_command_stack[depth] = []
                        if 'MATCH_PAIRS' not in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].append('MATCH_PAIRS')

                    used_obj = False
                    # 最初の再試行開始時の変数状態をスナップショットとして保存
                    previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                    previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                    for retry in range(MAX_MATCH_PAIRS_CONDITION_REGENERATION_RETRIES):
                        # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
                        if retry > 0:
                            # 現在の変数状態を取得
                            current_variable_names = set(context.variable_manager.get_all_variable_names())
                            current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                            # 前回の試行開始時以降に追加された変数を特定
                            new_variables = current_variable_names - previous_retry_start_variable_names
                            new_type_reservation_vars = current_type_reservation_vars - previous_retry_start_type_reservation_vars

                            # 削除対象の変数を決定
                            # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                            # 理由: 引数生成処理では変数定義ノード（InitializationNode, AssignmentNode）が生成されないため、
                            # 定義済み変数（is_defined=True）が作成される可能性はゼロ
                            # そのため、使用のみの変数（未定義）だけを削除する
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
                                # unavailable_variablesからも削除（未定義変数の追跡から削除）
                                for var_name in vars_to_remove:
                                    context.variable_manager.remove_unavailable_variable(var_name)
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

                        # 現在の再試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
                        previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                        previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                        # 第3引数を再生成
                        arg_schema = cmd_metadata.arguments[2]
                        nested_target_type_info = self._get_expected_argument_type_info(
                            cmd_name, 2, arg_schema
                        )
                        new_condition_arg = self._generate_argument_with_new_rules(
                            arg_schema, context, ctx_dict,
                            nested_target_type_info,
                            nesting_depth + 1, effective_max_nesting_depth,
                            no_new_vars_mode,
                            arg_index=2,
                            parent_command=cmd_name
                        )
                        args[2] = new_condition_arg

                        # $obj1,$obj2が最低一回ずつ使われているか確認
                        placeholders = self.node_generators._get_placeholder_variables_in_node(new_condition_arg)
                        if '$obj1' in placeholders and '$obj2' in placeholders:
                            used_obj = True
                            break

                        # オブジェクト単体の変数、リテラル値、コマンドがある場合、置き換えを試みる
                        missing = []
                        if '$obj1' not in placeholders:
                            missing.append('$obj1')
                        if '$obj2' not in placeholders:
                            missing.append('$obj2')
                        if missing:
                            success, replaced_node = self._try_replace_with_placeholders(new_condition_arg, missing, context)
                            if success and replaced_node is not None:
                                # 置き換えが成功した場合、args[2]を更新
                                args[2] = replaced_node
                                used_obj = True
                                break

                    # 再生成処理後にplaceholder_trackingとcurrent_commandをクリア
                    context.placeholder_tracking = None
                    context.current_command = None
                    # argument_nesting_command_stackからMATCH_PAIRSを削除（追加したもののみ）
                    # 注意: 元々あったMATCH_PAIRSも削除される可能性があるが、再生成処理中なので問題ない
                    for depth in context.argument_nesting_command_stack:
                        if 'MATCH_PAIRS' in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].remove('MATCH_PAIRS')

                    if not used_obj:
                        return False
                    return True
                else:
                    return False

        # SORT_BYのチェック
        if cmd_name == 'SORT_BY' and len(args) >= 2:
            key_expr_arg = args[1]  # 第2引数
            placeholders = self.node_generators._get_placeholder_variables_in_node(key_expr_arg)
            if '$obj' not in placeholders:
                # まず置き換えを試みる
                success, replaced_node = self._try_replace_with_placeholder(key_expr_arg, '$obj', context)
                if success and replaced_node is not None:
                    # 置き換えが成功した場合、args[1]を更新
                    args[1] = replaced_node
                    return True

                # 置き換えに失敗した場合、再生成を試みる
                if cmd_metadata and ctx_dict is not None and len(cmd_metadata.arguments) > 1:
                    # 再生成処理前にplaceholder_trackingを設定（オブジェクト単体引数の場合にプレースホルダー変数を確実に使用するため）
                    context.placeholder_tracking = PlaceholderTracking('SORT_BY')
                    # context.current_commandを設定（_is_in_condition_arg_nestingで検出されるようにするため）
                    context.current_command = 'SORT_BY'
                    # argument_nesting_command_stackにも追加（nesting_depthが分からないため、すべての深度に追加）
                    # 注意: 既存のスタックに追加することで、_is_in_condition_arg_nestingで確実に検出されるようにする
                    for depth in range(len(context.argument_nesting_command_stack) + 1):
                        if depth not in context.argument_nesting_command_stack:
                            context.argument_nesting_command_stack[depth] = []
                        if 'SORT_BY' not in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].append('SORT_BY')

                    used_obj = False
                    # 最初の再試行開始時の変数状態をスナップショットとして保存
                    previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                    previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                    for retry in range(MAX_SORT_BY_KEY_EXPR_REGENERATION_RETRIES):
                        # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
                        if retry > 0:
                            # 現在の変数状態を取得
                            current_variable_names = set(context.variable_manager.get_all_variable_names())
                            current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                            # 前回の試行開始時以降に追加された変数を特定
                            new_variables = current_variable_names - previous_retry_start_variable_names
                            new_type_reservation_vars = current_type_reservation_vars - previous_retry_start_type_reservation_vars

                            # 削除対象の変数を決定
                            # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                            # 理由: 引数生成処理では変数定義ノード（InitializationNode, AssignmentNode）が生成されないため、
                            # 定義済み変数（is_defined=True）が作成される可能性はゼロ
                            # そのため、使用のみの変数（未定義）だけを削除する
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
                                # unavailable_variablesからも削除（未定義変数の追跡から削除）
                                for var_name in vars_to_remove:
                                    context.variable_manager.remove_unavailable_variable(var_name)
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

                        # 現在の再試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
                        previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                        previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                        # 第2引数を再生成
                        arg_schema = cmd_metadata.arguments[1]
                        nested_target_type_info = self._get_expected_argument_type_info(
                            cmd_name, 1, arg_schema
                        )
                        new_key_expr_arg = self._generate_argument_with_new_rules(
                            arg_schema, context, ctx_dict,
                            nested_target_type_info,
                            nesting_depth + 1, effective_max_nesting_depth,
                            no_new_vars_mode,
                            arg_index=1,
                            parent_command=cmd_name
                        )
                        args[1] = new_key_expr_arg

                        # $objが使われているか確認
                        placeholders = self.node_generators._get_placeholder_variables_in_node(new_key_expr_arg)
                        if '$obj' in placeholders:
                            used_obj = True
                            break

                        # オブジェクト単体の変数、リテラル値、コマンドがある場合、置き換えを試みる
                        success, replaced_node = self._try_replace_with_placeholder(new_key_expr_arg, '$obj', context)
                        if success and replaced_node is not None:
                            # 置き換えが成功した場合、args[1]を更新
                            args[1] = replaced_node
                            used_obj = True
                            break

                    # 再生成処理後にplaceholder_trackingとcurrent_commandをクリア
                    context.placeholder_tracking = None
                    context.current_command = None
                    # argument_nesting_command_stackからSORT_BYを削除（追加したもののみ）
                    # 注意: 元々あったSORT_BYも削除される可能性があるが、再生成処理中なので問題ない
                    for depth in context.argument_nesting_command_stack:
                        if 'SORT_BY' in context.argument_nesting_command_stack[depth]:
                            context.argument_nesting_command_stack[depth].remove('SORT_BY')

                    if not used_obj:
                        return False
                    return True
                else:
                    return False

        # FLOW/LAY/SLIDEのチェック
        if cmd_name in ['FLOW', 'LAY', 'SLIDE'] and len(args) >= 3:
            first_arg = args[0]  # 第1引数（オブジェクト単体）
            third_arg = args[2]  # 第3引数（オブジェクト配列）
            if self._is_same_object_array_reference(first_arg, third_arg, context):
                # 同じオブジェクト配列を参照している場合、置き換えを試みる
                if self._try_replace_flow_lay_slide_args(args, context):
                    return True
                return False

        # PATHFINDのチェック
        if cmd_name == 'PATHFIND' and len(args) >= 4:
            first_arg = args[0]  # 第1引数（オブジェクト単体）
            fourth_arg = args[3]  # 第4引数（オブジェクト配列）
            if self._is_same_object_array_reference(first_arg, fourth_arg, context):
                # 同じオブジェクト配列を参照している場合、置き換えを試みる
                if self._try_replace_pathfind_args(args, context):
                    return True
                return False

        # 重複チェック: まったく同じ構造の引数がないか
        duplicate_indices = self._find_duplicate_structure_indices(args)
        if duplicate_indices:
            # 重複が検出された場合、再生成を試みる
            if cmd_metadata and ctx_dict is not None:
                flag = False
                # 最初の再試行開始時の変数状態をスナップショットとして保存
                previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                for retry in range(MAX_DUPLICATE_STRUCTURE_REGENERATION_RETRIES):
                    # 再試行時は、前回の試行で作成された変数（前回の試行開始時以降に追加された変数）を削除
                    if retry > 0:
                        # 現在の変数状態を取得
                        current_variable_names = set(context.variable_manager.get_all_variable_names())
                        current_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                        # 前回の試行開始時以降に追加された変数を特定
                        new_variables = current_variable_names - previous_retry_start_variable_names
                        new_type_reservation_vars = current_type_reservation_vars - previous_retry_start_type_reservation_vars

                        # 削除対象の変数を決定
                        # 1. 前回の試行開始時以降に追加された使用のみの変数（未定義）
                        # 理由: 引数生成処理では変数定義ノード（InitializationNode, AssignmentNode）が生成されないため、
                        # 定義済み変数（is_defined=True）が作成される可能性はゼロ
                        # そのため、使用のみの変数（未定義）だけを削除する
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
                            # unavailable_variablesからも削除（未定義変数の追跡から削除）
                            for var_name in vars_to_remove:
                                context.variable_manager.remove_unavailable_variable(var_name)
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

                    # 現在の再試行開始時の変数状態をスナップショットとして保存（次の再試行で使用）
                    previous_retry_start_variable_names = set(context.variable_manager.get_all_variable_names())
                    previous_retry_start_type_reservation_vars = set(context.variable_manager.get_variables_by_usage_context("type_reservation"))

                    # 重複している引数を再生成
                    for idx in duplicate_indices:
                        if idx < len(cmd_metadata.arguments):
                            arg_schema = cmd_metadata.arguments[idx]
                            nested_target_type_info = self._get_expected_argument_type_info(
                                cmd_name, idx, arg_schema
                            )
                            new_arg = self._generate_argument_with_new_rules(
                                arg_schema, context, ctx_dict,
                                nested_target_type_info,
                                nesting_depth + 1, effective_max_nesting_depth,
                                no_new_vars_mode,
                                arg_index=idx,
                                parent_command=cmd_name
                            )
                            args[idx] = new_arg

                    # 重複が解消されたか確認
                    if not self._has_duplicate_structure(args):
                        flag = True
                        break

                if not flag:
                    return False
            else:
                return False

        return True

    def _try_replace_with_placeholder(
        self,
        node: Node,
        placeholder: str,
        context: ProgramContext
    ) -> Tuple[bool, Optional[Node]]:
        """ノード内のオブジェクト単体の変数、リテラル値、コマンドをプレースホルダーに置き換え

        Args:
            node: 置き換え対象のノード
            placeholder: 置き換え先のプレースホルダー（$objなど）
            context: プログラムコンテキスト

        Returns:
            (置き換えが成功した場合True, 置き換え後のノード)
        """
        # オブジェクト単体のSemanticTypeの変数、リテラル値、コマンドを探す
        candidates = self._find_object_single_candidates(node, context)
        if not candidates:
            return (False, None)

        # ランダムに1つを選択して置き換え
        target = random.choice(candidates)
        success, replaced_node = self._replace_node_with_placeholder(node, target, placeholder, context)
        return (success, replaced_node)

    def _try_replace_with_placeholders(
        self,
        node: Node,
        placeholders: List[str],
        context: ProgramContext
    ) -> Tuple[bool, Optional[Node]]:
        """ノード内のオブジェクト単体の変数、リテラル値、コマンドをプレースホルダーに置き換え（複数）

        Args:
            node: 置き換え対象のノード
            placeholders: 置き換え先のプレースホルダーのリスト（[$obj1, $obj2]など）
            context: プログラムコンテキスト

        Returns:
            (置き換えが成功した場合True, 置き換え後のノード)
        """
        # オブジェクト単体のSemanticTypeの変数、リテラル値、コマンドを探す
        candidates = self._find_object_single_candidates(node, context)
        if len(candidates) < len(placeholders):
            return (False, None)

        # ランダムに選択して置き換え
        sample_size = min(len(placeholders), len(candidates))
        if sample_size > 0 and len(candidates) > 0:
            selected = random.sample(candidates, sample_size)
        else:
            selected = []
        current_node = node
        for target, placeholder in zip(selected, placeholders):
            success, replaced_node = self._replace_node_with_placeholder(current_node, target, placeholder, context)
            if not success:
                return (False, None)
            current_node = replaced_node
        return (True, current_node)

    def _find_object_single_candidates(self, node: Node, context: ProgramContext = None) -> List[Node]:
        """ノード内のオブジェクト単体のSemanticTypeの変数、リテラル値、コマンドを探す

        Args:
            node: 探索対象のノード
            context: プログラムコンテキスト（型情報取得に必要）

        Returns:
            候補ノードのリスト
        """
        candidates = []

        if isinstance(node, VariableNode):
            # 変数ノードの場合、オブジェクト単体かどうかを確認
            # プレースホルダー変数は除外
            if node.name.startswith('$'):
                return candidates

            # 配列要素は除外
            if '[' in node.name and ']' in node.name:
                return candidates

            # 型情報を確認（contextが利用可能な場合）
            if context is not None:
                var_info = context.variable_manager.get_variable_info(node.name)
                if var_info and 'type_info' in var_info:
                    type_info = var_info['type_info']
                    # オブジェクト単体（OBJECT型でis_array=False）かどうかを確認
                    if (type_info.semantic_type == SemanticType.OBJECT and
                        not type_info.is_array):
                        candidates.append(node)
                else:
                    # 型情報が取得できない場合、変数名から推定
                    # プレースホルダー変数ではなく、配列要素でもない場合は候補とする
                    candidates.append(node)
            else:
                # contextが利用できない場合、変数名から推定
                candidates.append(node)

        elif isinstance(node, LiteralNode):
            # リテラルノードの場合、オブジェクト単体かどうかを確認
            # リテラル値でオブジェクト単体を表すことは通常ないが、念のため
            # 型情報を確認（contextが利用可能な場合）
            if context is not None and hasattr(node, 'type_info') and node.type_info:
                type_info = node.type_info
                if (type_info.semantic_type == SemanticType.OBJECT and
                    not type_info.is_array):
                    candidates.append(node)

        elif isinstance(node, CommandNode):
            # コマンドノードの場合、戻り値がオブジェクト単体かどうかを確認
            # コマンドのメタデータから戻り値の型情報を取得
            cmd_metadata = COMMAND_METADATA.get(node.command)
            if cmd_metadata and cmd_metadata.return_type_info:
                return_type_info = cmd_metadata.return_type_info
                if (return_type_info.semantic_type == SemanticType.OBJECT and
                    not return_type_info.is_array):
                    candidates.append(node)
            else:
                # メタデータが取得できない場合、コマンド名から推定（フォールバック）
                if node.command.startswith('GET_') and 'OBJECT' in node.command.upper():
                    candidates.append(node)

            # 引数も再帰的に探索
            for arg in node.arguments:
                candidates.extend(self._find_object_single_candidates(arg, context))

        return candidates

    def _replace_node_with_placeholder(
        self,
        root_node: Node,
        target_node: Node,
        placeholder: str,
        context: ProgramContext
    ) -> Tuple[bool, Optional[Node]]:
        """ノードツリー内の特定のノードをプレースホルダーに置き換え

        Args:
            root_node: ルートノード
            target_node: 置き換え対象のノード
            placeholder: 置き換え先のプレースホルダー
            context: プログラムコンテキスト

        Returns:
            (置き換えが成功した場合True, 置き換え後のルートノード)
        """
        # プレースホルダーノードを生成
        placeholder_node = self.node_generators._generate_placeholder_node(
            placeholder, context, {}
        )

        # ルートノード自体が置き換え対象の場合
        if self._is_same_node(root_node, target_node):
            # ルートノードをプレースホルダーに置き換え
            return (True, placeholder_node)

        # ノードツリーを走査して置き換え
        if self._replace_node_recursive(root_node, target_node, placeholder_node):
            return (True, root_node)
        else:
            return (False, None)

    def _replace_node_recursive(
        self,
        node: Node,
        target: Node,
        replacement: Node
    ) -> bool:
        """ノードを再帰的に置き換え

        Args:
            node: 現在のノード
            target: 置き換え対象のノード
            replacement: 置き換え後のノード

        Returns:
            置き換えが成功した場合True
        """
        # 同じノードかどうかを確認（構造的に比較）
        if self._is_same_node(node, target):
            # ルートノードの場合は呼び出し側で置き換えを行うため、ここではTrueを返す
            # 実際の置き換えは親ノードで行われる
            return True

        # コマンドノードの場合、引数を再帰的に探索
        if isinstance(node, CommandNode):
            for i, arg in enumerate(node.arguments):
                if self._replace_node_recursive(arg, target, replacement):
                    # 引数を置き換え
                    node.arguments[i] = replacement
                    return True

        return False

    def _is_same_node(self, node1: Node, node2: Node) -> bool:
        """2つのノードが同じノードかどうかを構造的に判定

        Args:
            node1: ノード1
            node2: ノード2

        Returns:
            同じノードの場合True
        """
        # 型が異なる場合はFalse
        if not isinstance(node1, type(node2)):
            return False

        # VariableNodeの場合、変数名を比較
        if isinstance(node1, VariableNode) and isinstance(node2, VariableNode):
            return node1.name == node2.name

        # LiteralNodeの場合、値と型を比較
        if isinstance(node1, LiteralNode) and isinstance(node2, LiteralNode):
            # LiteralNodeの属性を確認（value属性がある場合）
            if hasattr(node1, 'value') and hasattr(node2, 'value'):
                if node1.value != node2.value:
                    return False
            # 型情報を比較（ある場合）
            if hasattr(node1, 'type_info') and hasattr(node2, 'type_info'):
                if node1.type_info != node2.type_info:
                    return False
            # generate()の結果も比較（フォールバック）
            try:
                return node1.generate() == node2.generate()
            except Exception:
                return False

        # CommandNodeの場合、コマンド名と引数を再帰的に比較
        if isinstance(node1, CommandNode) and isinstance(node2, CommandNode):
            if node1.command != node2.command:
                return False
            if len(node1.arguments) != len(node2.arguments):
                return False
            # 引数を再帰的に比較
            for arg1, arg2 in zip(node1.arguments, node2.arguments):
                if not self._is_same_node(arg1, arg2):
                    return False
            return True

        # その他のノードタイプの場合、generate()の結果で比較（フォールバック）
        try:
            return node1.generate() == node2.generate()
        except Exception:
            return False

    def _is_same_object_array_reference(self, obj_node: Node, array_node: Node, context: ProgramContext = None) -> bool:
        """オブジェクト単体とオブジェクト配列が同じオブジェクト配列を参照しているかチェック

        Args:
            obj_node: オブジェクト単体のノード（例: objects[i]）
            array_node: オブジェクト配列のノード（例: objects）
            context: プログラムコンテキスト（型情報取得に必要）

        Returns:
            同じオブジェクト配列を参照している場合True
        """
        # VariableNodeの場合、変数名を比較
        if isinstance(obj_node, VariableNode) and isinstance(array_node, VariableNode):
            obj_name = obj_node.name
            array_name = array_node.name

            # objects[i]とobjectsの場合、objectsが同じかどうかを確認
            if '[' in obj_name and ']' in obj_name:
                obj_base = obj_name.split('[')[0]
                if obj_base == array_name:
                    # 型情報も確認（contextが利用可能な場合）
                    if context is not None:
                        # オブジェクト単体の型情報を確認
                        obj_var_info = context.variable_manager.get_variable_info(obj_name)
                        # 配列の型情報を確認
                        array_var_info = context.variable_manager.get_variable_info(array_name)

                        # 型情報が取得できた場合、整合性を確認
                        if obj_var_info and 'type_info' in obj_var_info:
                            obj_type_info = obj_var_info['type_info']
                            # オブジェクト単体（OBJECT型でis_array=False）であることを確認
                            if not (obj_type_info.semantic_type == SemanticType.OBJECT and
                                    not obj_type_info.is_array):
                                return False

                        if array_var_info and 'type_info' in array_var_info:
                            array_type_info = array_var_info['type_info']
                            # オブジェクト配列（OBJECT型でis_array=True）であることを確認
                            if not (array_type_info.semantic_type == SemanticType.OBJECT and
                                    array_type_info.is_array):
                                return False
                    return True

        # ObjectAccessNodeの場合（配列要素アクセスを表すノード）
        # 注意: ObjectAccessNodeが実装されている場合、ここで処理を追加

        return False

    def _try_replace_flow_lay_slide_args(
        self,
        args: List[Node],
        context: ProgramContext
    ) -> bool:
        """FLOW/LAY/SLIDEの引数を置き換え

        新しいルール:
        - 既存変数に、使用可能な異なるオブジェクト配列の配列要素（objects1[j]）があれば、第1引数を置き換え
        - 既存変数に、使用可能な異なるオブジェクト配列があれば、第3引数を置き換え
        - 既存変数に、使用可能なオブジェクト単体があれば、第１引数を置き換え
        - それ以外の場合、失敗

        Args:
            args: 引数ノードのリスト
            context: プログラムコンテキスト

        Returns:
            置き換えが成功した場合True
        """
        if len(args) < 3:
            return False

        first_arg = args[0]  # 第1引数（オブジェクト単体）
        third_arg = args[2]  # 第3引数（オブジェクト配列）

        # 第3引数の配列名を取得
        third_array_name = None
        if isinstance(third_arg, VariableNode):
            third_array_name = third_arg.name

        # 既存変数から使用可能な変数を取得
        all_vars = context.variable_manager.get_all_variable_names()

        # 1. 異なるオブジェクト配列の配列要素（objects1[j]）があれば、第1引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    var_type_info.is_array and
                    var_name != third_array_name):
                    # 配列要素を使用（既存のメソッドを使用）
                    var_generators = VariableGenerators()
                    try:
                        # 配列要素ノードを生成
                        array_element_node = var_generators.generate_array_element_node(
                            context, {}, use_index_zero=False
                        )
                        if array_element_node:
                            args[0] = array_element_node
                            return True
                    except Exception:
                        pass

        # 2. 異なるオブジェクト配列があれば、第3引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    var_type_info.is_array and
                    var_name != third_array_name):
                    args[2] = VariableNode(var_name, {})
                    return True

        # 3. オブジェクト単体があれば、第1引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    not var_type_info.is_array):
                    args[0] = VariableNode(var_name, {})
                    return True

        return False

    def _try_replace_pathfind_args(
        self,
        args: List[Node],
        context: ProgramContext
    ) -> bool:
        """PATHFINDの引数を置き換え

        新しいルール:
        - 既存変数に、使用可能な異なるオブジェクト配列の配列要素（objects1[j]）があれば、第1引数を置き換え
        - 既存変数に、使用可能な異なるオブジェクト配列があれば、第4引数を置き換え
        - 既存変数に、使用可能なオブジェクト単体があれば、第１引数を置き換え
        - それ以外の場合、失敗

        Args:
            args: 引数ノードのリスト
            context: プログラムコンテキスト

        Returns:
            置き換えが成功した場合True
        """
        if len(args) < 4:
            return False

        first_arg = args[0]  # 第1引数（オブジェクト単体）
        fourth_arg = args[3]  # 第4引数（オブジェクト配列）

        # 第4引数の配列名を取得
        fourth_array_name = None
        if isinstance(fourth_arg, VariableNode):
            fourth_array_name = fourth_arg.name

        # 既存変数から使用可能な変数を取得
        all_vars = context.variable_manager.get_all_variable_names()

        # 1. 異なるオブジェクト配列の配列要素（objects1[j]）があれば、第1引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    var_type_info.is_array and
                    var_name != fourth_array_name):
                    # 配列要素を使用（既存のメソッドを使用）
                    var_generators = VariableGenerators()
                    try:
                        # 配列要素ノードを生成
                        array_element_node = var_generators.generate_array_element_node(
                            context, {}, use_index_zero=False
                        )
                        if array_element_node:
                            args[0] = array_element_node
                            return True
                    except Exception:
                        pass

        # 2. 異なるオブジェクト配列があれば、第4引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    var_type_info.is_array and
                    var_name != fourth_array_name):
                    args[3] = VariableNode(var_name, {})
                    return True

        # 3. オブジェクト単体があれば、第1引数を置き換え
        for var_name in all_vars:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (var_type_info.semantic_type == SemanticType.OBJECT and
                    not var_type_info.is_array):
                    args[0] = VariableNode(var_name, {})
                    return True

        return False

    def _has_duplicate_structure(self, args: List[Node]) -> bool:
        """引数内にまったく同じ構造の引数がないかチェック

        新しいルール:
        - ✖GET_DISTANCE(object, object)
        - ✖GET_DISTANCE(objects[i], objects[i])
        - ○GET_DISTANCE(objects[i], objects[0])
        - ✖GET_DISTANCE(GET_X(object), GET_X(object))
        - ✖GET_DISTANCE(GET_X(objects[i]), GET_X(objects[i]))
        - ○GET_DISTANCE(GET_X(objects2[j]), GET_X(objects[i]))

        Args:
            args: 引数ノードのリスト

        Returns:
            重複がある場合True
        """
        # 各引数のペアを比較
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                if self._is_same_structure(args[i], args[j]):
                    return True
        return False

    def _find_duplicate_structure_indices(self, args: List[Node]) -> List[int]:
        """引数内にまったく同じ構造の引数のインデックスを取得

        Args:
            args: 引数ノードのリスト

        Returns:
            重複している引数のインデックスのリスト（最初に見つかった重複ペアの両方のインデックス）
        """
        # 各引数のペアを比較
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                if self._is_same_structure(args[i], args[j]):
                    return [i, j]
        return []

    def _is_same_structure(self, node1: Node, node2: Node) -> bool:
        """2つのノードがまったく同じ構造かどうかをチェック

        Args:
            node1: ノード1
            node2: ノード2

        Returns:
            同じ構造の場合True
        """
        # generate()の結果で比較
        try:
            return node1.generate() == node2.generate()
        except Exception:
            return False
