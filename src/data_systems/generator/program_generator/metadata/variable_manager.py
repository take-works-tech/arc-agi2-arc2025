"""
統合変数管理システム

変数命名と変数追跡を統合した管理システム
"""
import logging
from typing import Dict, List, Set, Optional, Any
from .types import SemanticType, ReturnType, TypeInfo
from .variable_naming import VariableNamingSystem
from .variable_tracker import VariableTracker

# ログ設定
logger = logging.getLogger(__name__)


class VariableManager:
    """統合変数管理システム

    変数命名と変数追跡を統合した管理システム
    """

    def __init__(self, max_variables_per_type: int = 10):
        """初期化

        Args:
            max_variables_per_type: 型あたりの最大変数数
        """
        self.naming_system = VariableNamingSystem(max_variables_per_type)
        self.tracker = VariableTracker()

        logger.info("VariableManager initialized")

    # ========================================
    # 変数命名機能（VariableNamingSystem委譲）
    # ========================================

    def get_next_variable_name(
        self,
        type_info: TypeInfo,
        used_vars: Set[str],
        preferred_name: Optional[str] = None
    ) -> str:
        """次の変数名を取得"""
        return self.naming_system.get_next_variable_name(type_info, used_vars, preferred_name)

    def get_argument_name(self, type_info: TypeInfo) -> str:
        """引数名を取得"""
        return self.naming_system.get_argument_name(type_info)

    def get_semantic_type_from_variable_name(self, var_name: str) -> Optional[SemanticType]:
        """変数名から意味的型を推定"""
        return self.naming_system.get_semantic_type_from_variable_name(var_name)

    # ========================================
    # 変数追跡機能（VariableTracker委譲）
    # ========================================

    def define_variable(self, name: str, semantic_type: SemanticType, is_array: bool = False, value: Any = None, is_defined: bool = True):
        """変数を定義"""
        self.tracker.define_variable(name, semantic_type, is_array, value, is_defined)

    def get_variable(self, name: str) -> Optional[Dict]:
        """変数を取得"""
        return self.tracker.get_variable(name)

    def register_variable_usage(self, name: str, context: str = "general", type_info=None):
        """変数の使用を登録（未定義の場合は追加、既存の場合は使用マーク）"""
        self.tracker.register_variable_usage(name, context, type_info)

    def unmark_variable_usage(self, var_name: str) -> None:
        """変数の使用マークを解除"""
        self.tracker.unmark_variable_usage(var_name)

    def unmark_variables_usage(self, var_names: List[str]) -> None:
        """複数の変数の使用マークを解除"""
        self.tracker.unmark_variables_usage(var_names)


    def get_all_variable_names(self) -> List[str]:
        """全ての変数名を取得"""
        return self.tracker.get_all_variable_names()

    def get_unused_variables(self) -> List[str]:
        """未使用の変数名を取得"""
        return self.tracker.get_unused_variables()


    def get_variable_type(self, name: str) -> Optional[SemanticType]:
        """変数の型を取得"""
        return self.tracker.get_variable_type(name)

    def get_variable_info(self, variable_name: str) -> Dict:
        """変数の情報を取得"""
        return self.tracker.get_variable_info(variable_name)

    def get_all_variables(self) -> Dict[str, Dict]:
        """すべての変数情報を取得"""
        return self.tracker.get_all_variables()

    def get_variables_by_type(self, semantic_type: SemanticType) -> List[str]:
        """指定された型の変数名リストを取得"""
        return self.tracker.get_variables_by_type(semantic_type)

    def get_variable_usage_count(self, variable_name: str) -> int:
        """変数の使用回数を取得"""
        return self.tracker.get_variable_usage_count(variable_name)

    def get_argument_usage_history(self) -> List[str]:
        """引数で使用された変数の履歴を取得"""
        return self.tracker.get_argument_usage_history()

    def get_variables_used_in_arguments(self) -> List[str]:
        """引数で使用された変数のリストを取得（重複除去）"""
        return self.tracker.get_variables_used_in_arguments()

    def get_variables_by_usage_context(self, context: str) -> List[str]:
        """指定されたコンテキストで使用された変数を取得"""
        return self.tracker.get_variables_by_usage_context(context)

    def get_compatible_variables_for_assignment(self, target_type: SemanticType, is_array: bool = False, context: Optional[Any] = None) -> List[str]:
        """代入先として使用可能な変数を取得（型互換性を考慮、可視性も考慮）

        Args:
            target_type: ターゲット型
            is_array: 配列かどうか
            context: ProgramContext（可視性チェック用、オプショナル）

        Returns:
            List[str]: 使用可能な変数名のリスト
        """
        from .types import SemanticType

        compatible_vars = self.tracker.get_compatible_variables_for_assignment(target_type, is_array)

        # オブジェクト単体の場合、FORループの配列も既存変数として扱う
        # （配列要素参照（objects[i]）が使用可能であることを示す）
        if (not is_array and target_type == SemanticType.OBJECT and
            context is not None and hasattr(context, 'for_arrays')):
            valid_for_arrays = [arr for arr in context.for_arrays if arr is not None]
            for arr_name in valid_for_arrays:
                # FORループの配列は既存変数として扱う（配列要素参照が使用可能）
                # 実際の変数として存在するかどうかに関わらず、FORループで使用されている配列は既存変数として扱う
                if arr_name not in compatible_vars:
                    compatible_vars.append(arr_name)

        # 可視性を考慮（contextが提供された場合）
        if context is not None and hasattr(context, 'is_variable_visible'):
            compatible_vars = [var for var in compatible_vars if context.is_variable_visible(var)]

        return compatible_vars

    def get_defined_variables(self) -> Set[str]:
        """定義された変数のセットを取得"""
        return self.tracker.get_defined_variables()

    def is_defined(self, variable_name: str) -> bool:
        """変数が定義されているかチェック"""
        return self.tracker.is_defined(variable_name)


    def get_used_only_variables(self) -> List[str]:
        """使用のみの変数（未定義）を取得"""
        return self.tracker.get_used_only_variables()

    def get_defined_only_variables(self) -> List[str]:
        """定義のみの変数（未使用）を取得"""
        return self.tracker.get_defined_only_variables()

    def get_defined_and_used_variables(self) -> List[str]:
        """定義済みかつ使用済みの変数を取得"""
        return self.tracker.get_defined_and_used_variables()

    def check_variable_duplication(self, cmd_name: str, arguments: List, context) -> bool:
        """コマンド内での変数重複をチェック（直接引数のみ）"""
        from ..generation.nodes import VariableNode, PlaceholderNode
        from ..metadata.types import SemanticType

        # すべての変数名を収集（直接引数のみ）
        all_variable_names = []
        placeholder_vars = []

        for arg in arguments:
            if isinstance(arg, PlaceholderNode):
                # PlaceholderNodeの場合はplaceholder属性を取得
                placeholder_vars.append(arg.placeholder)
            elif isinstance(arg, VariableNode):
                var_name = arg.name
                if var_name.startswith('$'):
                    # プレースホルダーは常にオブジェクト関連として扱う
                    placeholder_vars.append(var_name)
                else:
                    # すべての変数を収集（型に関係なく）
                    all_variable_names.append(var_name)

        # プレースホルダーの重複チェック
        if len(placeholder_vars) != len(set(placeholder_vars)):
            return True  # プレースホルダー重複あり

        # すべての変数の重複チェック（同じ変数の直接併用を禁止）
        # 配列要素の場合は、ベース名とインデックスの両方を考慮
        normalized_vars = []
        for var_name in all_variable_names:
            if '[' in var_name and ']' in var_name:
                # 配列要素の場合: ベース名とインデックスの両方を正規化
                # 例: objects[0] -> ('objects', '0'), objects[i] -> ('objects', 'i')
                base = var_name.split('[')[0]
                index_part = var_name.split('[')[1].split(']')[0]
                normalized_vars.append((base, index_part))
            else:
                # 通常の変数の場合: ベース名のみ
                normalized_vars.append((var_name, None))

        # 正規化された変数名の重複チェック
        # 同じベース名+インデックスの組み合わせが重複していれば禁止
        if len(normalized_vars) != len(set(normalized_vars)):
            return True  # 変数重複あり

        # 同じベース名の配列要素で、インデックスが異なる場合は許可（既にset()で処理済み）
        # しかし、ベース名のみの変数と配列要素の併用は許可する必要がある
        # （例: objects と objects[0] は併用可能）
        # このチェックは上記のset()チェックで既に処理されている

        # オブジェクト関連変数の重複チェック（配列要素のインデックスを考慮）
        object_variable_names = [name for name in all_variable_names if self._is_object_related_variable(name, context)]
        if self._has_object_variable_duplication(object_variable_names):
            return True  # オブジェクト変数重複あり

        # 特殊ケース: FLOW, LAY, SLIDE, PATHFINDの衝突チェック
        if cmd_name in ['FLOW', 'LAY', 'SLIDE', 'PATHFIND'] and len(arguments) >= 2:
            target_arg = arguments[0]
            # PATHFINDは引数構造が異なる: [obj, target_x, target_y, obstacles]
            # 他のコマンドは: [obj, direction, obstacles]
            if cmd_name == 'PATHFIND':
                obstacles_arg = arguments[3] if len(arguments) > 3 else None
            else:
                obstacles_arg = arguments[2] if len(arguments) > 2 else None

            if isinstance(target_arg, VariableNode) and isinstance(obstacles_arg, VariableNode):
                target_var = target_arg.name
                obstacles_var = obstacles_arg.name

                # 操作対象が配列要素で、obstaclesが同じ配列の場合
                if '[' in target_var and ']' in target_var:
                    target_base = target_var.split('[')[0]
                    if obstacles_var == target_base:
                        return True  # 衝突あり

        # ネストされたコマンドの重複チェック
        nested_signatures = self._collect_nested_command_signatures(arguments)
        if len(nested_signatures) != len(set(nested_signatures)):
            return True  # ネストされたコマンドの重複あり

        return False  # 重複なし

    def _collect_nested_command_signatures(self, arguments: List) -> List[tuple]:
        """ネストされたコマンドのシグネチャ（コマンド名、引数のシグネチャ）を再帰的に収集

        Args:
            arguments: 引数のリスト（Nodeのリスト）

        Returns:
            シグネチャのリスト。各シグネチャは以下の形式：
            - CommandNodeの場合: (コマンド名, (引数のシグネチャのタプル))
            - PlaceholderNodeの場合: ('$placeholder', プレースホルダー名)
            - VariableNodeの場合: ('$variable', 変数名) または ('$placeholder', プレースホルダー名)
            - LiteralNodeの場合: ('$literal', 値)
        """
        from ..generation.nodes import CommandNode, VariableNode, PlaceholderNode, LiteralNode

        signatures = []
        for arg in arguments:
            if isinstance(arg, CommandNode):
                # 引数のシグネチャを再帰的に収集
                arg_signatures = self._collect_nested_command_signatures(arg.arguments)
                # コマンド名と引数のシグネチャの組み合わせを作成
                signature = (arg.command, tuple(arg_signatures))
                signatures.append(signature)
            elif isinstance(arg, PlaceholderNode):
                # PlaceholderNodeの場合はplaceholder属性を取得
                signatures.append(('$placeholder', arg.placeholder))
            elif isinstance(arg, VariableNode):
                var_name = arg.name
                if var_name.startswith('$'):
                    # プレースホルダー変数（$で始まる）
                    signatures.append(('$placeholder', var_name))
                else:
                    # 通常の変数
                    signatures.append(('$variable', var_name))
            elif isinstance(arg, LiteralNode):
                # リテラル値は値も含めて比較
                signatures.append(('$literal', arg.value))
            # その他のノードタイプは無視（BinaryOpNodeなど）

        return signatures

    def _has_object_variable_duplication(self, object_variable_names: List[str]) -> bool:
        """オブジェクト変数の重複をチェック（配列要素のインデックスを考慮）"""
        # 完全に同じ変数名の重複をチェック
        if len(object_variable_names) != len(set(object_variable_names)):
            return True

        # 配列要素のインデックスが異なる場合は別物として扱う
        # 例: objects[i] と objects[j] は別物
        # 例: grid_size[0] と grid_size[1] は別物

        # 配列とその要素の併用は許可
        # 例: objects と objects[i] は併用可能

        return False

    def _is_object_related_variable(self, var_name: str, context) -> bool:
        """変数がオブジェクト関連（オブジェクト配列またはオブジェクト）かチェック"""
        # 変数名のパターンで判定
        if var_name.startswith('object') or var_name.startswith('objects'):
            return True

        # 配列要素の場合
        if '[' in var_name and ']' in var_name:
            base_name = var_name.split('[')[0]
            if base_name.startswith('object'):
                return True

        # コンテキストから変数の型情報を取得して判定
        if hasattr(context, 'variable_manager'):
            try:
                var_info = context.variable_manager.get_variable_info(var_name)
                if var_info and 'type_info' in var_info:
                    type_info = var_info['type_info']
                    semantic_type = type_info.semantic_type
                    # オブジェクト関連の型かチェック
                    return semantic_type in [SemanticType.OBJECT]
            except:
                pass

        return False

    def reset_variable_tracking(self):
        """変数追跡システムをリセット"""
        self.tracker.reset_variable_tracking()

    def clear(self):
        """全ての変数をクリア"""
        self.tracker.clear()

    # ========================================
    # 互換性属性（VariableTracker互換）
    # ========================================

    @property
    def variables(self) -> Dict[str, Dict]:
        """互換性のためのvariables属性"""
        if self.tracker is None:
            logger.error("VariableManager.variables: tracker is None")
            return {}
        result = self.tracker.variables
        if not result:
            logger.debug("VariableManager.variables: tracker.variables is empty")
        else:
            # type_infoの状態を確認
            for name, info in result.items():
                if not isinstance(info, dict):
                    logger.warning(f"VariableManager.variables: Variable '{name}' info is not a dict: {type(info)}")
                elif 'type_info' not in info:
                    logger.warning(f"VariableManager.variables: Variable '{name}' has no 'type_info' key")
                elif info.get('type_info') is None:
                    logger.warning(f"VariableManager.variables: Variable '{name}' has type_info=None")
        return result

    @property
    def used_variable_names(self) -> Set[str]:
        """互換性のためのused_variable_names属性"""
        return set(self.tracker.get_used_variable_names())

    def get_used_variable_names(self) -> List[str]:
        """使用済み変数名のリストを取得"""
        return self.tracker.get_used_variable_names()

    def get_defined_variable_names(self) -> List[str]:
        """定義済み変数名のリストを取得"""
        return self.tracker.get_defined_variable_names()

    def remove_variables(self, variable_names: List[str]):
        """指定された変数を削除"""
        self.tracker.remove_variables(variable_names)

    def remove_variables_by_type(self, semantic_type: SemanticType):
        """指定された型の変数を削除"""
        self.tracker.remove_variables_by_type(semantic_type)

    # 使用不可変数管理の委譲メソッド
    def add_unavailable_variable(self, var_name: str) -> None:
        """使用不可変数を追加"""
        self.tracker.add_unavailable_variable(var_name)

    def remove_unavailable_variable(self, var_name: str) -> None:
        """使用不可変数を削除"""
        self.tracker.remove_unavailable_variable(var_name)

    def clear_unavailable_variables(self) -> None:
        """使用不可変数をすべてクリア"""
        self.tracker.clear_unavailable_variables()

    def get_unavailable_variables(self) -> Set[str]:
        """使用不可変数のセットを取得"""
        return self.tracker.get_unavailable_variables()

    def is_variable_unavailable(self, var_name: str) -> bool:
        """変数が使用不可かどうかを判定"""
        return self.tracker.is_variable_unavailable(var_name)

    def get_available_variables(self, all_variables: Set[str]) -> Set[str]:
        """使用可能な変数のセットを取得（使用不可変数を除外）"""
        return self.tracker.get_available_variables(all_variables)


# グローバルインスタンス
variable_manager = VariableManager()
