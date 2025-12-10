"""
変数管理関連の機能
"""
import random
import re
from typing import List, Dict, Optional, Set
from .types import SemanticType, ReturnType, TypeSystem, TypeInfo
from .variable_manager import variable_manager
from .commands import COMMAND_METADATA, CommandMetadata
from ..generation.nodes import (
    Node, StatementNode, ExpressionNode,
    InitializationNode, AssignmentNode, ArrayAssignmentNode,
    FilterNode, IfBranchNode, RenderNode,
    LiteralNode, VariableNode, CommandNode, BinaryOpNode,
    PlaceholderNode,
)
from ..generation.program_context import ProgramContext


class VariableManagers:
    """変数管理関連の機能を提供するクラス"""

    def __init__(self):
        pass

    def _get_undefined_variables(self, context: ProgramContext, program_nodes: List[Node] = None) -> List[str]:
        """未定義の変数を取得（実際にプログラム内で使用されている変数のみ）"""
        # 引数生成時にmark_variable_usedで記録された変数を使用
        used_vars = context.variable_manager.get_used_variable_names()

        # プレースホルダー変数、数値リテラル、予約語を除外
        filtered_used_vars = []
        import re
        for var_name in used_vars:
            # プレースホルダー変数（$obj1, $obj2など）を除外
            if var_name.startswith('$'):
                continue

            # 数値リテラル（-6, 123など）を除外
            if re.match(r'^-?\d+$', var_name):
                continue

            # 文字列リテラル（"start", "end"など）を除外
            if (var_name.startswith('"') and var_name.endswith('"')) or (var_name.startswith("'") and var_name.endswith("'")):
                continue

            # 予約語・定数を除外
            reserved_words = {'True', 'False', 'start', 'end', 'first', 'last', 'X', 'Y', '-X', '-Y'}
            if var_name in reserved_words:
                continue

            # 関数名を除外
            if var_name.startswith('GET_') or var_name.startswith('IS_') or var_name in ['MERGE', 'CONCAT', 'FILTER', 'EXCLUDE']:
                continue

            filtered_used_vars.append(var_name)

        # 新しい未定義変数検出ロジックを使用（定義されていない変数）
        return context.variable_manager.tracker.get_undefined_variables(filtered_used_vars)

    def _remove_unused_variables(self, program_nodes: List[Node], context: ProgramContext) -> List[Node]:
        """未使用変数の定義ノードを削除（variable_analyzerの実装を使用）"""
        # variable_analyzerのremove_unused_variablesを使用（より確実な実装）
        try:
            from ...program_executor.node_analyzer.variable_analyzer import remove_unused_variables
            # variable_analyzerの実装は、ノードリストを解析して実際に使用されている変数を検出する
            filtered_nodes = remove_unused_variables(program_nodes)
            return filtered_nodes
        except Exception as e:
            # フォールバック: 元の実装を使用
            import sys
            print(f"[WARNING] _remove_unused_variables: variable_analyzerの使用に失敗しました: {e}", file=sys.stderr, flush=True)

            # variable_analyzerのextract_variable_definitions_from_nodeとextract_variable_usages_from_nodeを使用
            try:
                from ...program_executor.node_analyzer.variable_analyzer import (
                    extract_variable_definitions_from_node,
                    extract_variable_usages_from_node,
                    remove_self_assignments,
                    remove_empty_control_structures
                )

                # 自己代入を先に削除
                program_nodes = remove_self_assignments(program_nodes)

                # 複数回のパスで未使用変数を削除（再帰的削除）
                max_iterations = 10  # 無限ループ防止
                iteration = 0
                previous_count = len(program_nodes)

                while iteration < max_iterations:
                    # 各ノードで定義される変数とその位置を記録
                    variable_definitions: Dict[str, List[int]] = {}  # {変数名: [定義ノードのインデックスのリスト]}

                    # 各ノードで使用される変数を収集
                    used_variables: Set[str] = set()

                    # 最初のパス: 変数の定義と使用を収集
                    for i, node in enumerate(program_nodes):
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
                            # FORカウント変数（ForStartWithCountNode）に紐づく変数は保持
                            try:
                                from ..generation.nodes import ForStartWithCountNode
                                if ForStartWithCountNode is not None:
                                    for n in program_nodes:
                                        if isinstance(n, ForStartWithCountNode):
                                            count_var = getattr(n, "count_variable", "")
                                            if var_name == count_var:
                                                continue
                            except Exception:
                                pass
                            # この変数は定義されているが使用されていない
                            # すべての定義を削除（再定義も含む）
                            unused_node_indices.update(def_indices)

                    # 削除するノードがない場合は終了
                    if not unused_node_indices:
                        break

                    # 未使用変数の定義ノードを削除
                    filtered_nodes = []
                    for i, node in enumerate(program_nodes):
                        if i not in unused_node_indices:
                            filtered_nodes.append(node)

                    program_nodes = filtered_nodes
                    iteration += 1

                    # 削除されたノード数が変わらなかった場合は終了
                    if len(program_nodes) == previous_count:
                        break
                    previous_count = len(program_nodes)

                # 空のFOR/IFループを削除
                filtered_nodes = remove_empty_control_structures(program_nodes)

                return filtered_nodes
            except Exception as e2:
                # さらにフォールバック: 簡易実装を使用
                import sys
                print(f"[WARNING] _remove_unused_variables: extract関数の使用にも失敗しました: {e2}", file=sys.stderr, flush=True)
                # 引数生成時にmark_variable_usedで記録された変数を使用
                used_vars = set(context.variable_manager.get_used_variable_names())

                # 未使用変数の定義ノードを削除
                filtered_nodes = []
                # GET_INPUT_GRID_SIZE 系は可読性・後段利用を想定して常に保持
                protected_vars = {"grid_size", "grid_size_x", "grid_size_y"}
                # grid_size系の別名も将来考慮（grid_size* の接頭辞を保護）
                def is_protected_name(name: str) -> bool:
                    return name in protected_vars or name.startswith("grid_size")

                # FORカウント変数（ForStartWithCountNode）に紐づく変数は保持
                try:
                    from ..generation.nodes import ForStartWithCountNode
                except Exception:
                    ForStartWithCountNode = None  # ない場合はスキップ

                if ForStartWithCountNode is not None:
                    for n in program_nodes:
                        if isinstance(n, ForStartWithCountNode):
                            protected_vars.add(getattr(n, "count_variable", ""))

                for node in program_nodes:
                    if isinstance(node, AssignmentNode):
                        # 代入ノードの場合、左辺の変数が実際に使用されているかチェック
                        var_name = node.variable
                        if var_name in used_vars or is_protected_name(var_name):
                            filtered_nodes.append(node)
                        else:
                            pass
                    else:
                        # 代入ノード以外はそのまま保持
                        filtered_nodes.append(node)

                return filtered_nodes


    def _register_variables_used_in_node(self, node: Node, context: ProgramContext, usage_context: str = "argument"):
        """ノード内で使用された変数を登録（統一版）"""
        variables_used = variable_manager.tracker._get_variables_used_in_node(node)
        for var_name in variables_used:
            # 実際の変数名のみを登録（コマンドの結果は除外）
            if self._is_actual_variable_name(var_name):
                # 変数を使用済みとして登録（型情報付き）
                # 変数の型情報を取得
                var_info = context.variable_manager.get_variable_info(var_name)
                if var_info and 'type_info' in var_info:
                    type_info = var_info['type_info']
                else:
                    # 型情報がない場合はデフォルトのOBJECT配列型を使用
                    from .types import TypeInfo, SemanticType, ReturnType
                    type_info = TypeInfo(
                        semantic_type=SemanticType.OBJECT,
                        is_array=True,
                        return_type=ReturnType.OBJECT
                    )
                context.variable_manager.register_variable_usage(var_name, usage_context, type_info)

    def _is_actual_variable_name(self, name: str) -> bool:
        """実際の変数名かどうかを判定（コマンドの結果は除外）"""
        # コマンドの結果（例：GET_Y($obj)）は除外
        if '(' in name and ')' in name:
            return False

        # プレースホルダー（例：$obj）は除外
        if name.startswith('$'):
            return False

        # grid_size_x と grid_size_y は特別な変数として除外
        if name in ['grid_size_x', 'grid_size_y']:
            return False

        # その他は実際の変数名とみなす
        return True


    def _get_existing_variables_for_argument(self, type_info: TypeInfo, context: ProgramContext) -> List[str]:
        """引数で使用可能な既存変数を取得"""
        # 型と配列フラグの両方を考慮して互換性のある変数を取得（可視性も考慮）
        compatible_vars = context.variable_manager.get_compatible_variables_for_assignment(
            type_info.semantic_type,
            type_info.is_array,
            context=context  # 可視性チェック用にcontextを渡す
        )

        # 型互換性チェックを強化
        if type_info.semantic_type == SemanticType.OBJECT and type_info.is_array:
            # OBJECT_ARRAY_ARG用の変数のみを厳密にフィルタリング
            compatible_vars = [var for var in compatible_vars if self._is_object_array_variable(var, context)]

            # 厳密フィルタリング後も変数がない場合は、新しく生成
            if not compatible_vars:
                return []
        elif type_info.semantic_type == SemanticType.COLOR and type_info.is_array:
            # COLOR_ARRAY_ARG用の変数のみを厳密にフィルタリング
            compatible_vars = [var for var in compatible_vars if self._is_color_array_variable(var, context)]

            # 厳密フィルタリング後も変数がない場合は、新しく生成
            if not compatible_vars:
                return []

        # 引数で使用された変数を優先
        argument_used_vars = context.variable_manager.get_variables_used_in_arguments()
        priority_vars = [var for var in compatible_vars if var in argument_used_vars]

        if priority_vars:
            return priority_vars

        # フォールバック: 型互換性のある変数
        return compatible_vars


    def _should_exclude_object_from_obstacle_array(self, cmd_name: str, arg_index: int, context: ProgramContext) -> bool:
        """オブジェクトを障害物配列から除外すべきかどうかを判定"""
        # FLOW, LAY, SLIDE, PATHFINDコマンドの第1引数（操作対象）の場合
        obstacle_commands = ['FLOW', 'LAY', 'SLIDE', 'PATHFIND']
        if cmd_name in obstacle_commands and arg_index == 0:
            return True
        return False

    def _is_object_array_variable(self, var_name: str, context: ProgramContext) -> bool:
        """変数がオブジェクト配列かどうかを厳密にチェック"""
        try:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                type_info = var_info['type_info']
                return (type_info.semantic_type == SemanticType.OBJECT and
                        type_info.is_array == True)
        except:
            pass
        return False

    def _is_color_array_variable(self, var_name: str, context: ProgramContext) -> bool:
        """変数が色配列かどうかを厳密にチェック"""
        try:
            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                type_info = var_info['type_info']
                return (type_info.semantic_type == SemanticType.COLOR and
                        type_info.is_array == True)
        except:
            pass
        return False



    def _is_command_compatible_with_target(self, cmd_name: str, cmd_metadata: CommandMetadata, target_type: SemanticType, context: ProgramContext, node: 'CommandNode' = None) -> bool:
        """コマンドの戻り値が代入先の型と互換性があるかチェック"""
        from ..metadata.types import TypeSystem

        # 算術演算ノードの場合は、ノードのreturn_type_infoを使用（ターゲット型に設定されている）
        if node and hasattr(node, 'return_type_info') and node.return_type_info is not None:
            return_type = node.return_type_info.semantic_type
            return_is_array = node.return_type_info.is_array
        else:
            # 通常のコマンドの場合は、CommandMetadataから取得
            return_type = cmd_metadata.return_type_info.semantic_type
            return_is_array = cmd_metadata.return_type_info.is_array

        # 型互換性をチェック（is_arrayフラグも考慮）
        # FORループ内では配列要素に代入するので、非配列（is_array=False）のコマンドが必要
        return (TypeSystem.are_compatible(return_type, target_type) and
                return_is_array == False)
