"""
変数追跡システム

変数の定義、使用状況、型情報の管理を担当
"""
import logging
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from .types import SemanticType, ReturnType, TypeInfo

# ログ設定
logger = logging.getLogger(__name__)


class VariableTracker:
    """変数追跡システム

    変数の定義、使用状況、型情報の管理を担当
    """

    def __init__(self):
        """初期化"""
        # 変数追跡システム
        self._variable_tracker: Dict[str, Dict] = {}  # 変数名 -> {type_info, position, usage_count, is_defined, is_used, etc.}
        self._position_counter: Dict[SemanticType, int] = {}  # 型ごとの位置カウンター

        # 変数使用履歴の強化
        self._argument_usage_history: List[str] = []  # 引数で使用された変数の履歴
        self._variable_usage_context: Dict[str, List[str]] = {}  # 変数名 -> 使用コンテキストのリスト

        # 使用不可変数の管理
        self._unavailable_variables: Set[str] = set()  # 使用不可変数のセット

        # 定義された変数の追跡は各変数のis_definedフラグで管理

        # VariableTracker互換性のための属性
        self.variables: Dict[str, Dict] = {}  # 互換性のためのvariables属性

        logger.info("VariableTracker initialized")

    def _infer_type_from_variable_name(self, name: str) -> Optional['TypeInfo']:
        """変数名から型情報を推定"""
        from .types import TypeInfo, SemanticType

        # 不正な変数名（関数呼び出しの一部など）をフィルタリング
        if ('(' in name or ')' in name or
            name.startswith('GET_') or name.startswith('IS_') or
            name.startswith('$') or
            name in ['MERGE', 'CONCAT', 'FILTER', 'EXCLUDE']):
            return None

        # 変数名パターンから型を推定
        name_lower = name.lower()

        # オブジェクト配列
        if (name.startswith('objects') or name.endswith('_objects') or
            name in ['filtered_objects', 'sorted_objects']):
            return TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True)

        # 単一オブジェクト
        if (name.startswith('object') or name.endswith('_object') or
            name in ['obj', 'current_object', 'target_object']):
            return TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)

        # ループ変数（i, j, kなど）
        if name_lower in ['i', 'j', 'k', 'l', 'm', 'n']:
            return TypeInfo.create_from_semantic_type(SemanticType.LOOP_INDEX, is_array=False)

        # サイズ関連
        if ('size' in name_lower or 'width' in name_lower or 'height' in name_lower or
            'count' in name_lower or name_lower in ['x', 'y', 'w', 'h']):
            return TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False)

        # 比率関連
        if 'aspect_ratio' in name_lower or 'aspect' in name_lower:
            return TypeInfo.create_from_semantic_type(SemanticType.ASPECT_RATIO, is_array=False)

        if 'density' in name_lower:
            return TypeInfo.create_from_semantic_type(SemanticType.DENSITY, is_array=False)

        # 色関連
        if ('color' in name_lower or name_lower in ['bg_color', 'background_color']):
            return TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=False)

        # フラグ関連
        if ('flag' in name_lower or 'is_' in name_lower or name_lower.startswith('has_')):
            return TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False)

        # 文字列型関連
        if ('align_mode' in name_lower or name_lower == 'mode'):
            return TypeInfo.create_from_semantic_type(SemanticType.ALIGN_MODE, is_array=False)
        if ('direction' in name_lower):
            return TypeInfo.create_from_semantic_type(SemanticType.DIRECTION, is_array=False)
        if (name_lower == 'axis'):
            return TypeInfo.create_from_semantic_type(SemanticType.AXIS, is_array=False)

        # デフォルト: オブジェクト型として推定
        return TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False)

    def _get_next_position_for_type(self, semantic_type: SemanticType) -> int:
        """型ごとの次の位置を取得"""
        if semantic_type not in self._position_counter:
            self._position_counter[semantic_type] = 0
        self._position_counter[semantic_type] += 1
        return self._position_counter[semantic_type]


    def _track_variable(self, variable_name: str, type_info: TypeInfo, position: int, is_defined: bool = False):
        """変数を追跡システムに記録"""
        # デバッグログ: type_infoの状態を確認
        if type_info is None:
            logger.error(f"VariableTracker._track_variable(): type_info is None for variable '{variable_name}'")
            return

        if not isinstance(type_info, TypeInfo):
            logger.error(f"VariableTracker._track_variable(): type_info is not TypeInfo for variable '{variable_name}': {type(type_info)}")
            return

        if variable_name not in self._variable_tracker:
            self._variable_tracker[variable_name] = {
                'type_info': type_info,
                'position': position,
                'usage_count': 0,
                'first_used': True,
                'last_used': False,
                'is_defined': is_defined,
                'is_used': False
            }
            # 互換性のためのvariables属性も更新
            self.variables[variable_name] = self._variable_tracker[variable_name]
            logger.debug(f"VariableTracker._track_variable(): Added new variable '{variable_name}' with type_info={type_info.semantic_type.value}")
        else:
            # 既存の変数の使用回数を増やす
            self._variable_tracker[variable_name]['usage_count'] += 1
            # 互換性のためのvariables属性も更新
            self.variables[variable_name] = self._variable_tracker[variable_name]
            logger.debug(f"VariableTracker._track_variable(): Updated existing variable '{variable_name}' usage_count")

    def define_variable(self, name: str, semantic_type: SemanticType, is_array: bool = False, value: Any = None, is_defined: bool = True):
        """変数を定義（二重定義を防ぐ）"""
        from .types import TypeInfo

        # 既に定義されている場合
        if name in self._variable_tracker:
            # 既存の変数情報を更新
            self._variable_tracker[name]['value'] = value
            self._variable_tracker[name]['is_array'] = is_array
            # 定義フラグを必ず更新（True/Falseに関わらず）
            self._variable_tracker[name]['is_defined'] = is_defined
            return

        # 新規変数の場合
        type_info = TypeInfo.create_from_semantic_type(semantic_type, is_array)
        self._track_variable(name, type_info, self._get_next_position_for_type(semantic_type), is_defined)

        # 値と配列情報を追加
        self._variable_tracker[name]['value'] = value
        self._variable_tracker[name]['is_array'] = is_array

    def get_variable(self, name: str) -> Optional[Dict]:
        """変数を取得"""
        return self._variable_tracker.get(name)


    def register_variable_usage(self, name: str, context: str = "general", type_info: Optional['TypeInfo'] = None):
        """変数の使用を登録（未定義の場合は追加、既存の場合は使用マーク）"""
        # デバッグログ: 引数の状態を確認
        logger.debug(f"VariableTracker.register_variable_usage(): name='{name}', context='{context}', type_info={type_info}")

        # 変数が未定義の場合は、使用のみとして記録（自動定義しない）
        if name not in self._variable_tracker:
            if type_info:
                # 変数名の命名規則に基づいて型情報を検証
                # 変数名から型推定は行わない（渡されたtype_infoを優先）
                # 理由: get_next_variable_nameで正しく型情報に基づいて変数名が生成されるため、
                # 変数名から型を推定して上書きすると、型情報と変数名の不一致を招く
                # 例: object1という変数名が生成された場合でも、is_array=Trueの型情報が正しい場合はそれに従う
                # 引数の型情報を使用して使用のみとして記録
                logger.debug(f"VariableTracker.register_variable_usage(): Tracking new variable '{name}' with provided type_info")
                self._track_variable(name, type_info, self._get_next_position_for_type(type_info.semantic_type), False)
            else:
                # 型情報が提供されていない場合は変数名から推定を試行
                logger.debug(f"VariableTracker.register_variable_usage(): type_info is None, attempting to infer from variable name '{name}'")
                inferred_type = self._infer_type_from_variable_name(name)
                if inferred_type:
                    logger.warning(f"変数 '{name}' の型情報が未提供のため、変数名から推定しました: {inferred_type.semantic_type}")
                    self._track_variable(name, inferred_type, self._get_next_position_for_type(inferred_type.semantic_type), False)
                else:
                    # 推定できない場合はエラー
                    logger.error(f"VariableTracker.register_variable_usage(): Cannot infer type for variable '{name}', raising ValueError")
                    raise ValueError(f"変数 '{name}' の型情報が提供されていません。mark_variable_used呼び出し時にtype_infoを指定してください。")

        # 使用回数を増やし、使用済みとしてマーク（is_definedフラグは保持）
        if name not in self._variable_tracker:
            logger.error(f"VariableTracker.register_variable_usage(): Variable '{name}' not in _variable_tracker after _track_variable")
            return

        self._variable_tracker[name]['usage_count'] += 1
        self._variable_tracker[name]['is_used'] = True
        # is_definedフラグは既存の値を保持（define_variableで設定された値が保持される）

        # 引数使用履歴を記録
        if context == "argument":
            self._argument_usage_history.append(name)

        # 使用コンテキストを記録
        if name not in self._variable_usage_context:
            self._variable_usage_context[name] = []
        self._variable_usage_context[name].append(context)


    def get_unused_variables(self) -> List[str]:
        """未使用の変数名を取得（定義されているが使用されていない変数）"""
        variables = []
        for name, info in self._variable_tracker.items():
            if (info.get('is_defined', False) and
                not info.get('is_used', False)):
                variables.append(name)
        return variables

    def unmark_variable_usage(self, var_name: str) -> None:
        """変数の使用マークを解除（is_usedをFalseに設定）

        Args:
            var_name: 変数名
        """
        if var_name in self._variable_tracker:
            self._variable_tracker[var_name]['is_used'] = False
            # 使用回数は保持（完全にリセットしない）

    def unmark_variables_usage(self, var_names: List[str]) -> None:
        """複数の変数の使用マークを解除

        Args:
            var_names: 変数名のリスト
        """
        for var_name in var_names:
            self.unmark_variable_usage(var_name)


    def get_variable_type(self, name: str) -> Optional[SemanticType]:
        """変数の型を取得"""
        var_info = self._variable_tracker.get(name)
        if var_info and 'type_info' in var_info:
            return var_info['type_info'].semantic_type
        return None


    def get_all_variables(self) -> Dict[str, Dict]:
        """すべての変数情報を取得

        Returns:
            変数名 -> 変数情報の辞書
        """
        return self._variable_tracker.copy()

    def get_variables_by_type(self, semantic_type: SemanticType) -> List[str]:
        """指定された型の変数名リストを取得

        Args:
            semantic_type: セマンティック型

        Returns:
            変数名のリスト
        """
        variables = []
        for var_name, var_info in self._variable_tracker.items():
            if var_info.get('type_info', {}).get('semantic_type') == semantic_type:
                variables.append(var_name)
        return variables

    def get_variable_usage_count(self, variable_name: str) -> int:
        """変数の使用回数を取得

        Args:
            variable_name: 変数名

        Returns:
            使用回数
        """
        var_info = self._variable_tracker.get(variable_name, {})
        return var_info.get('usage_count', 0)

    def reset_variable_tracking(self):
        """変数追跡システムをリセット"""
        self._variable_tracker.clear()
        self._position_counter.clear()
        self._argument_usage_history.clear()
        self._variable_usage_context.clear()
        self._unavailable_variables.clear()  # 使用不可変数もクリア
        self.variables.clear()

    def get_argument_usage_history(self) -> List[str]:
        """引数で使用された変数の履歴を取得"""
        return self._argument_usage_history.copy()


    def get_variables_by_usage_context(self, context: str) -> List[str]:
        """指定されたコンテキストで使用された変数を取得"""
        return [var_name for var_name, contexts in self._variable_usage_context.items()
                if context in contexts]


    def clear(self):
        """全ての変数をクリア"""
        self._variable_tracker.clear()
        self._position_counter.clear()
        self.variables.clear()
        self._argument_usage_history.clear()
        self._variable_usage_context.clear()
        self._unavailable_variables.clear()  # 使用不可変数もクリア

    def remove_variables(self, variable_names: List[str]):
        """指定された変数を削除（スコープ終了時用）"""
        for var_name in variable_names:
            # 内部追跡システムから削除
            if var_name in self._variable_tracker:
                del self._variable_tracker[var_name]

            # 互換性属性から削除
            if var_name in self.variables:
                del self.variables[var_name]


            # 引数使用履歴から削除
            if var_name in self._argument_usage_history:
                self._argument_usage_history = [v for v in self._argument_usage_history if v != var_name]

            # 使用コンテキストから削除
            if var_name in self._variable_usage_context:
                del self._variable_usage_context[var_name]

            logger.debug(f"Variable removed from scope: {var_name}")

    def remove_variables_by_type(self, semantic_type: SemanticType, is_array: bool = None):
        """指定された型の変数を削除（スコープ終了時用）"""
        variables_to_remove = []

        for var_name, var_info in self._variable_tracker.items():
            type_info = var_info.get('type_info')
            if type_info and type_info.semantic_type == semantic_type:
                if is_array is None or type_info.is_array == is_array:
                    variables_to_remove.append(var_name)

        if variables_to_remove:
            self.remove_variables(variables_to_remove)
            logger.debug(f"Removed {len(variables_to_remove)} variables of type {semantic_type} (array={is_array})")

    def get_undefined_variables(self, used_variable_names: List[str]) -> List[str]:
        """未定義の変数を取得（定義されていない変数のみ）"""
        undefined_vars = []
        defined_vars = self.get_defined_variables()

        # 未定義の変数を特定（定義されていない変数）
        for var_name in used_variable_names:
            if var_name not in defined_vars:
                undefined_vars.append(var_name)

        return undefined_vars

    def get_used_variable_names(self) -> List[str]:
        """使用済み変数名のリストを取得"""
        used_vars = []
        for name, info in self._variable_tracker.items():
            if info.get('is_used', False):
                used_vars.append(name)
        return used_vars

    def get_defined_variable_names(self) -> List[str]:
        """定義済み変数名のリストを取得"""
        return list(self.get_defined_variables())

    def get_all_variable_names(self) -> List[str]:
        """全ての変数名を取得"""
        return list(self._variable_tracker.keys())

    def get_variable_info(self, variable_name: str) -> Dict:
        """変数の情報を取得"""
        result = self._variable_tracker.get(variable_name, {})
        if not result:
            logger.debug(f"VariableTracker.get_variable_info(): Variable '{variable_name}' not found in _variable_tracker, returning empty dict")
        else:
            type_info = result.get('type_info')
            if type_info is None:
                logger.warning(f"VariableTracker.get_variable_info(): Variable '{variable_name}' has type_info=None")
            elif not isinstance(type_info, TypeInfo):
                logger.warning(f"VariableTracker.get_variable_info(): Variable '{variable_name}' has type_info as {type(type_info)}, not TypeInfo")
        return result

    def get_variables_used_in_arguments(self) -> List[str]:
        """引数で使用された変数のリストを取得（重複除去）"""
        return list(set(self._argument_usage_history))

    def get_compatible_variables_for_assignment(self, target_type: SemanticType, is_array: bool = False) -> List[str]:
        """代入先として使用可能な変数を取得（型互換性を考慮、使用不可変数を除外）"""
        from .types import TypeSystem

        compatible_vars = []
        for var_name, var_info in self._variable_tracker.items():
            # 使用不可変数は除外
            if var_name in self._unavailable_variables:
                continue

            if 'type_info' in var_info:
                var_type_info = var_info['type_info']
                if (TypeSystem.are_compatible(var_type_info.semantic_type, target_type) and
                    var_type_info.is_array == is_array):
                    compatible_vars.append(var_name)

        return compatible_vars

    def get_defined_variables(self) -> Set[str]:
        """定義された変数のセットを取得"""
        defined_vars = set()
        for name, info in self._variable_tracker.items():
            if info.get('is_defined', False):
                defined_vars.add(name)
        return defined_vars

    def is_defined(self, variable_name: str) -> bool:
        """変数が定義されているかチェック"""
        if variable_name in self._variable_tracker:
            return self._variable_tracker[variable_name].get('is_defined', False)
        return False

    def get_used_only_variables(self) -> List[str]:
        """使用のみの変数（未定義）を取得"""
        variables = []
        for name, info in self._variable_tracker.items():
            if (not info.get('is_defined', False) and
                info.get('is_used', False)):
                variables.append(name)
        return variables

    def get_defined_only_variables(self) -> List[str]:
        """定義のみの変数（未使用）を取得"""
        variables = []
        for name, info in self._variable_tracker.items():
            if (info.get('is_defined', False) and
                not info.get('is_used', False)):
                variables.append(name)
        return variables

    def get_defined_and_used_variables(self) -> List[str]:
        """定義済みかつ使用済みの変数を取得"""
        variables = []
        for name, info in self._variable_tracker.items():
            if (info.get('is_defined', False) and
                info.get('is_used', False)):
                variables.append(name)
        return variables


    def _get_variables_used_in_node(self, node) -> Set[str]:
        """ノード内で実際に使用されている変数を取得"""
        from ..generation.nodes import (
            VariableNode, CommandNode, AssignmentNode, FilterNode,
            ConcatNode, AppendNode, MergeNode, ExcludeNode,
            MatchPairsNode, ExtractShapeNode, SplitConnectedNode,
            RenderNode, ObjectAccessNode, SingleObjectArrayNode,
            ExtendPatternNode, ArrangeGridNode, ArrayAssignmentNode,
            ForStartNode, ForStartWithCountNode, IfStartNode, IfBranchNode,
            BinaryOpNode
        )

        used_vars = set()

        if isinstance(node, VariableNode):
            var_name = node.name
            # プレースホルダーは除外
            if var_name.startswith('$'):
                return used_vars

            # 配列要素の場合は配列名を抽出
            if '[' in var_name and ']' in var_name:
                array_name = var_name.split('[')[0]
                used_vars.add(array_name)
            else:
                used_vars.add(var_name)
        elif isinstance(node, CommandNode):
            # コマンドノードの場合、引数を再帰的にチェック
            for arg in node.arguments:
                used_vars.update(self._get_variables_used_in_node(arg))
        elif isinstance(node, AssignmentNode):
            # 代入ノードの場合、右辺の変数をチェック
            used_vars.update(self._get_variables_used_in_node(node.expression))
        elif isinstance(node, ArrayAssignmentNode):
            # 配列代入ノードの場合、配列名と右辺の変数をチェック
            used_vars.add(node.array)
            used_vars.update(self._get_variables_used_in_node(node.expression))
        elif isinstance(node, FilterNode):
            # FILTERノードの場合、ソース配列と条件内の変数をチェック
            used_vars.add(node.source_array)
            # conditionは文字列なので、文字列から変数を抽出
            used_vars.update(self._extract_variables_from_string(node.condition))
        elif isinstance(node, ForStartWithCountNode):
            # COUNT変数を使用するFOR開始ノード
            used_vars.add(node.count_variable)
        elif isinstance(node, ForStartNode):
            # 配列長を用いるFOR開始ノード
            used_vars.add(node.array)
        elif isinstance(node, IfStartNode):
            # IF開始ノードの条件文字列から変数を抽出
            used_vars.update(self._extract_variables_from_string(node.condition))
        elif isinstance(node, IfBranchNode):
            # IF分岐ノード: 条件と then/else 内の変数を再帰的に収集
            used_vars.update(self._extract_variables_from_string(node.condition))
            for child in node.then_body:
                used_vars.update(self._get_variables_used_in_node(child))
            for child in node.else_body:
                used_vars.update(self._get_variables_used_in_node(child))
        elif isinstance(node, BinaryOpNode):
            # 二項演算ノード（左/右の式を再帰的に走査）
            used_vars.update(self._get_variables_used_in_node(node.left))
            used_vars.update(self._get_variables_used_in_node(node.right))
        elif isinstance(node, ConcatNode):
            # CONCATノードの場合、array1とarray2を使用
            used_vars.add(node.array1)
            used_vars.add(node.array2)
        elif isinstance(node, AppendNode):
            # APPENDノードの場合、arrayとobjを使用
            used_vars.add(node.array)
            # objが配列要素の場合は配列名を抽出
            if '[' in node.obj and ']' in node.obj:
                array_name = node.obj.split('[')[0]
                used_vars.add(array_name)
            else:
                used_vars.add(node.obj)
        elif isinstance(node, MergeNode):
            # MERGEノードの場合、objects_arrayを使用
            used_vars.add(node.objects_array)
        elif isinstance(node, ExcludeNode):
            # EXCLUDEノードの場合、source_arrayとtargets_arrayを使用
            used_vars.add(node.source_array)
            used_vars.add(node.targets_array)
        elif isinstance(node, MatchPairsNode):
            # MATCH_PAIRSノードの場合、array1とarray2を使用
            used_vars.add(node.array1)
            used_vars.add(node.array2)
            # conditionは文字列なので、文字列から変数を抽出
            used_vars.update(self._extract_variables_from_string(node.condition))
        elif isinstance(node, ExtractShapeNode):
            # EXTRACT_SHAPEノードの場合、source_objectを使用
            used_vars.add(node.source_object)
        elif isinstance(node, SplitConnectedNode):
            # SPLIT_CONNECTEDノードの場合、source_objectを使用
            used_vars.add(node.source_object)
        elif isinstance(node, RenderNode):
            # RENDERノードの場合、arrayを使用
            used_vars.add(node.array)
            # bg_color, width, heightもチェック
            used_vars.update(self._get_variables_used_in_node(node.bg_color))
        elif isinstance(node, ObjectAccessNode):
            # OBJECT_ACCESSノードの場合、objects_arrayを使用
            used_vars.add(node.objects_array)
        elif isinstance(node, SingleObjectArrayNode):
            # SINGLE_OBJECT_ARRAYノードの場合、object_nameを使用
            used_vars.add(node.object_name)
        elif isinstance(node, ExtendPatternNode):
            # EXTEND_PATTERNノードの場合、source_arrayを使用
            used_vars.add(node.source_array)
        elif isinstance(node, ArrangeGridNode):
            # ARRANGE_GRIDノードの場合、source_arrayを使用
            used_vars.add(node.source_array)
        elif isinstance(node, str):
            # 文字列の場合（FilterNodeのconditionなど）
            used_vars.update(self._extract_variables_from_string(node))

        return used_vars

    def _extract_variables_from_string(self, text: str) -> Set[str]:
        """文字列から変数名を抽出"""
        import re

        used_vars = set()

        # プレースホルダーを除外してから変数を抽出
        # $obj, $obj1, $obj2などのプレースホルダーを一時的に置換
        placeholder_pattern = r'\$[a-zA-Z0-9_]+'
        text_without_placeholders = re.sub(placeholder_pattern, 'PLACEHOLDER', text)

        # コマンド名を除外（GET_で始まる関数名など）
        command_pattern = r'\b(GET_[A-Z_]+|IS_[A-Z_]+|COUNT_[A-Z_]+|FILTER|MERGE|CONCAT|EXCLUDE|APPEND|SORT_BY|ARRANGE_GRID|MATCH_PAIRS|SPLIT_CONNECTED|CREATE_[A-Z_]+|EXTRACT_[A-Z_]+|FIT_[A-Z_]+|MOVE|TELEPORT|SLIDE|PATHFIND|ROTATE|FLIP|SCALE|EXPAND|FILL_HOLES|SET_COLOR|OUTLINE|HOLLOW|INTERSECTION|SUBTRACT|FLOW|DRAW|LAY|CROP|ALIGN|TILE|REVERSE)\b'
        text_without_commands = re.sub(command_pattern, 'COMMAND', text_without_placeholders)

        # 変数名のパターン: 英文字で始まり、英数字とアンダースコアが続く
        pattern = r'\b([a-z_][a-z0-9_]*)\[?\d*\]?'
        matches = re.findall(pattern, text_without_commands)

        for match in matches:
            # プレースホルダー、FORループ変数、コマンド名、その他の除外対象でない場合のみ追加
            if (not match.startswith('$') and
                match not in ['PLACEHOLDER', 'COMMAND'] and
                not self._is_for_loop_variable(match)):
                used_vars.add(match)

        return used_vars

    def _is_for_loop_variable(self, var_name: str) -> bool:
        """FORループ変数かどうかを判定"""
        # 一般的なFORループ変数名
        common_loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']
        return var_name in common_loop_vars

    def add_unavailable_variable(self, var_name: str) -> None:
        """使用不可変数を追加"""
        self._unavailable_variables.add(var_name)
        logger.debug(f"Added unavailable variable: {var_name}")

    def remove_unavailable_variable(self, var_name: str) -> None:
        """使用不可変数を削除"""
        self._unavailable_variables.discard(var_name)
        logger.debug(f"Removed unavailable variable: {var_name}")

    def clear_unavailable_variables(self) -> None:
        """使用不可変数をすべてクリア"""
        self._unavailable_variables.clear()
        logger.debug("Cleared all unavailable variables")

    def get_unavailable_variables(self) -> Set[str]:
        """使用不可変数のセットを取得"""
        return self._unavailable_variables.copy()

    def is_variable_unavailable(self, var_name: str) -> bool:
        """変数が使用不可かどうかを判定"""
        return var_name in self._unavailable_variables

    def get_available_variables(self, all_variables: Set[str]) -> Set[str]:
        """使用可能な変数のセットを取得（使用不可変数を除外）"""
        return all_variables - self._unavailable_variables


# グローバルインスタンス
variable_tracker = VariableTracker()
