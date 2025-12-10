"""
変数生成関連の機能
"""
import random
from typing import List, Dict, Optional, Tuple
from ..metadata.types import SemanticType, TypeInfo
from ..metadata.variable_manager import variable_manager
from .nodes import Node, VariableNode, ArrayAssignmentNode, CommandNode, LiteralNode
from .program_context import ProgramContext
from .node_utilities import NodeUtilities
from . import OBJECT_ARRAY_TYPE


class VariableGenerators:
    """変数生成関連の機能を提供するクラス"""

    def __init__(self):
        self.utilities = NodeUtilities()

    def normalize_variable_name(self, var_name: str) -> Tuple[str, Optional[str]]:
        """変数名を正規化（ベース名とインデックスに分割）

        Args:
            var_name: 変数名（例: "objects[i]", "objects[0]", "objects"）

        Returns:
            Tuple[ベース名, インデックス]: 配列要素の場合は(ベース名, インデックス)、通常変数の場合は(ベース名, None)
        """
        if '[' in var_name and ']' in var_name:
            base = var_name.split('[')[0]
            index_part = var_name.split('[')[1].split(']')[0]
            return (base, index_part)
        else:
            return (var_name, None)

    def reserve_variable_type_info(self, var_name: str, semantic_type: SemanticType, is_array: bool, context: ProgramContext):
        """変数の型情報を予約（定義ではない）"""
        from ..metadata.types import TypeInfo

        # VariableManagerからtrackerを取得
        tracker = context.variable_manager.tracker if hasattr(context.variable_manager, 'tracker') else context.variable_manager

        # 既に型情報が予約されている場合はスキップ
        var_info = tracker.get_variable(var_name)
        if var_info and 'type_info' in var_info:
            return

        # 型情報を予約（定義ではない）
        type_info = TypeInfo.create_from_semantic_type(semantic_type, is_array)
        # register_variable_usageを使用して適切に登録
        context.variable_manager.register_variable_usage(var_name, "type_reservation", type_info)

    def get_array_element_type(self, array_name: str, context: ProgramContext) -> SemanticType:
        """配列要素の型を取得"""
        # 配列の型情報から要素の型を決定
        var_info = context.variable_manager.get_variable(array_name)
        if var_info and 'type_info' in var_info:
            type_info = var_info['type_info']
            # 配列の要素型を取得
            if hasattr(type_info, 'semantic_type'):
                return type_info.semantic_type

        # デフォルトはOBJECT型
        return SemanticType.OBJECT

    def select_assignment_target(self, target_type: SemanticType, context: ProgramContext) -> str:
        """逆算ルールに基づいて代入先を選択"""
        # 1. 引数で使用された変数から型互換性のあるものを取得
        argument_used_vars = context.variable_manager.get_variables_used_in_arguments()
        compatible_vars = context.variable_manager.get_compatible_variables_for_assignment(target_type, is_array=True, context=context)

        # 2. 引数使用履歴と型互換性の両方を満たす変数を選択
        candidate_vars = [var for var in argument_used_vars if var in compatible_vars]

        if candidate_vars:
            # 引数で使用された変数から選択（逆算ルール）
            return random.choice(candidate_vars)

        # 3. フォールバック: 型互換性のある変数から選択
        if compatible_vars:
            return random.choice(compatible_vars)

        # 4. 最終フォールバック: 新しい変数を生成
        type_info = TypeInfo.create_from_semantic_type(target_type, is_array=True)
        new_var_name = variable_manager.get_next_variable_name(
            type_info,
            set(context.variable_manager.get_all_variable_names())
        )
        context.variable_manager.define_variable(new_var_name, target_type, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(new_var_name)
        return new_var_name

    def generate_array_element_node(self, context: ProgramContext, ctx_dict: Dict, use_index_zero: bool = False) -> Node:
        """配列要素ノードを生成（objects[i]形式またはobjects[0]形式）"""
        valid_for_arrays = [arr for arr in context.for_arrays if arr is not None]
        # 使用禁止変数と明示除外を反映
        try:
            unavailable_vars = context.variable_manager.get_unavailable_variables()
        except Exception:
            unavailable_vars = set()
        excluded_vars = getattr(context, 'excluded_variables_for_definition', set())
        filtered_arrays = [arr for arr in valid_for_arrays if arr not in unavailable_vars and arr not in excluded_vars]
        if filtered_arrays:
            valid_for_arrays = filtered_arrays

        # 重複回避: used_variables_in_current_commandを参照
        # 注意: 同じコマンド内で同じ変数を直接2回使用することを禁止
        # 例: GET_DISTANCE(objects6[i], objects6[i]) は禁止
        # 例: GET_DISTANCE(objects6[i], objects6[0]) は許可（異なるインデックス）
        # 例: GET_DISTANCE(objects6[i], object) は許可（異なる変数）
        used_variables_in_current_command = getattr(context, 'used_variables_in_current_command', set())

        # 使用可能な配列をフィルタ（重複回避のため）
        # 同じコマンド内で同じ配列要素（例：objects6[i]）を2回使用することを禁止
        # ただし、異なるインデックス（例：objects6[0]）や配列全体（例：objects6）は許可
        available_arrays = []
        for arr in valid_for_arrays:
            # ループ変数を取得（最後から逆順に検索して、最も内側のFORループのインデックスを見つける）
            try:
                # 最後から逆順に検索して、最も内側のFORループのインデックスを見つける
                loop_idx = len(context.for_arrays) - 1 - context.for_arrays[::-1].index(arr)
            except ValueError:
                continue
            loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
            loop_var = loop_vars[min(loop_idx, len(loop_vars) - 1)]

            # 生成される変数名を予測
            if use_index_zero:
                predicted_var_name = f"{arr}[0]"
            else:
                predicted_var_name = f"{arr}[{loop_var}]"

            # 重複チェック: 同じコマンド内で同じ配列要素を2回使用することを禁止
            normalized_result = self.normalize_variable_name(predicted_var_name)
            # インデックスも考慮した重複チェックのため、タプル全体を使用
            if isinstance(normalized_result, tuple):
                normalized_var = normalized_result  # タプル全体を使用（ベース名+インデックス）
            else:
                normalized_var = (normalized_result, None)  # 通常変数の場合もタプルに統一
            # 同じコマンド内で同じ配列要素（例：objects6[i]）が既に使用されている場合は除外
            # ただし、異なるインデックス（例：objects6[0]）や配列全体（例：objects6）は許可
            # objects[i]とobjectsは区別する（FLOW/LAY/SLIDE/PATHFINDの最終チェック以外）
            if normalized_var not in used_variables_in_current_command:
                available_arrays.append(arr)

        # 使用可能な配列がない場合、エラーを発生（重複回避のため）
        if not available_arrays:
            raise ValueError(f"利用可能な配列がありません（重複回避のため）。used_variables_in_current_command={used_variables_in_current_command}, valid_for_arrays={valid_for_arrays}")

        # ランダムにFORループの配列を選択
        array_name = random.choice(available_arrays)

        # 対応するループ変数を取得（最後から逆順に検索して、最も内側のFORループのインデックスを見つける）
        # これにより、ネストされたループで正しいループ変数が選択される
        loop_idx = len(context.for_arrays) - 1 - context.for_arrays[::-1].index(array_name)
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        loop_var = loop_vars[min(loop_idx, len(loop_vars) - 1)]

        # 配列が使用されたことを記録（型情報付き）
        context.variable_manager.register_variable_usage(array_name, "argument", OBJECT_ARRAY_TYPE)

        # MATCH_PAIRS配列の場合は特別なインデックス計算を使用
        if array_name in context.match_pairs_arrays:
            # MULTIPLY(i, 2) または ADD(MULTIPLY(i, 2), 1) をランダムに選択
            if random.random() < 0.5:
                # MULTIPLY(i, 2)
                index_expr = f"MULTIPLY({loop_var}, 2)"
            else:
                # ADD(MULTIPLY(i, 2), 1)
                index_expr = f"ADD(MULTIPLY({loop_var}, 2), 1)"
            element_var_name = f"{array_name}[{index_expr}]"
        elif use_index_zero:
            # 通常の引数選択では要素0または最後の要素をランダムに選択
            if random.random() < 0.5:
                # objects[0]を使用
                element_var_name = f"{array_name}[0]"
            else:
                # objects[SUB(LEN(objects), 1)]を使用（最後の要素）
                len_cmd = CommandNode('LEN', [VariableNode(array_name, ctx_dict)], ctx_dict)
                sub_cmd = CommandNode('SUB', [len_cmd, LiteralNode(1, ctx_dict)], ctx_dict)
                element_var_name = f"{array_name}[{sub_cmd.generate()}]"
        else:
            element_var_name = f"{array_name}[{loop_var}]"

        # used_variables_in_current_commandに追加
        normalized_result = self.normalize_variable_name(element_var_name)
        # インデックスも考慮した重複チェックのため、タプル全体を使用
        if isinstance(normalized_result, tuple):
            normalized_var = normalized_result  # タプル全体を使用（ベース名+インデックス）
        else:
            normalized_var = (normalized_result, None)  # 通常変数の場合もタプルに統一
        if hasattr(context, 'used_variables_in_current_command'):
            context.used_variables_in_current_command.add(normalized_var)
            # FLOW/LAY/SLIDEコマンドの特殊ケース: 配列要素（例: objects[i]）が使用された場合、
            # その配列のベース名（例: objects）も使用済みとしてマーク
            # これにより、第1引数がobjects[i]の場合、第3引数でobjectsを使用できなくなる
            if isinstance(normalized_result, tuple):
                base_name = normalized_result[0]
                current_cmd = getattr(context, 'current_command', None)
                if current_cmd in ['FLOW', 'LAY', 'SLIDE', 'PATHFIND']:
                    # 配列全体も使用済みとしてマーク
                    context.used_variables_in_current_command.add((base_name, None))

        return VariableNode(element_var_name, ctx_dict)
