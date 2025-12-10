"""
Slot-based Partial Program Handler

部分プログラムをslot-basedで扱う機能
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class SlotType(Enum):
    """スロットタイプの列挙型"""
    VARIABLE = "variable"  # 変数スロット
    EXPRESSION = "expression"  # 式スロット
    CONDITION = "condition"  # 条件スロット


@dataclass
class Slot:
    """スロット（部分プログラム内の変数や式のプレースホルダー）"""
    slot_id: str  # スロットID（例: "slot_1", "slot_2"）
    slot_type: SlotType  # スロットタイプ
    original_value: str  # 元の値（部分プログラム内の実際の値）
    constraints: Dict[str, Any]  # スロットの制約（型、範囲など）
    metadata: Dict[str, Any]  # メタデータ


class SlotBasedPartialProgramHandler:
    """Slot-based部分プログラムハンドラー"""

    def __init__(self):
        """初期化"""
        self.slots: Dict[str, Slot] = {}
        self.slot_counter = 0

    def parse_partial_program(
        self,
        partial_program: str
    ) -> Tuple[str, Dict[str, Slot]]:
        """
        部分プログラムをパースしてスロットに変換

        Args:
            partial_program: 部分プログラム文字列

        Returns:
            (slotified_program, slots): スロット化されたプログラムとスロットの辞書
        """
        self.slots = {}
        self.slot_counter = 0

        slotified_program = partial_program

        # 変数をスロットに変換
        slotified_program, variable_slots = self._extract_variables(slotified_program)

        # 式をスロットに変換
        slotified_program, expression_slots = self._extract_expressions(slotified_program)

        # 条件をスロットに変換
        slotified_program, condition_slots = self._extract_conditions(slotified_program)

        # すべてのスロットを統合
        all_slots = {**variable_slots, **expression_slots, **condition_slots}

        return slotified_program, all_slots

    def _extract_variables(
        self,
        program: str
    ) -> Tuple[str, Dict[str, Slot]]:
        """
        変数をスロットに変換

        Args:
            program: プログラム文字列

        Returns:
            (slotified_program, slots): スロット化されたプログラムとスロットの辞書
        """
        slots = {}
        slotified_program = program

        # 変数パターンを検出（例: objects1, objects2, objects1_32など）
        variable_pattern = r'\bobjects\d+(?:_\d+)?\b'
        variables = re.findall(variable_pattern, program)

        # 各変数をスロットに変換
        variable_to_slot = {}
        for var in set(variables):
            slot_id = f"slot_{self.slot_counter}"
            self.slot_counter += 1

            # 変数の型と制約を推測
            constraints = self._infer_variable_constraints(var)

            slot = Slot(
                slot_id=slot_id,
                slot_type=SlotType.VARIABLE,
                original_value=var,
                constraints=constraints,
                metadata={"variable_name": var}
            )

            slots[slot_id] = slot
            variable_to_slot[var] = slot_id

        # 変数をスロットIDに置換
        for var, slot_id in variable_to_slot.items():
            slotified_program = re.sub(r'\b' + re.escape(var) + r'\b', slot_id, slotified_program)

        return slotified_program, slots

    def _extract_expressions(
        self,
        program: str
    ) -> Tuple[str, Dict[str, Slot]]:
        """
        式をスロットに変換

        Args:
            program: プログラム文字列

        Returns:
            (slotified_program, slots): スロット化されたプログラムとスロットの辞書
        """
        slots = {}
        slotified_program = program

        # 関数呼び出しパターンを検出（例: FILTER(...), GET_COLOR(...)など）
        expression_pattern = r'\b[A-Z_]+\([^)]*\)'
        expressions = re.findall(expression_pattern, program)

        # 各式をスロットに変換
        expression_to_slot = {}
        for expr in set(expressions):
            slot_id = f"slot_{self.slot_counter}"
            self.slot_counter += 1

            # 式の型と制約を推測
            constraints = self._infer_expression_constraints(expr)

            slot = Slot(
                slot_id=slot_id,
                slot_type=SlotType.EXPRESSION,
                original_value=expr,
                constraints=constraints,
                metadata={"expression": expr}
            )

            slots[slot_id] = slot
            expression_to_slot[expr] = slot_id

        # 式をスロットIDに置換（長い式から順に置換）
        sorted_expressions = sorted(expression_to_slot.keys(), key=len, reverse=True)
        for expr in sorted_expressions:
            slot_id = expression_to_slot[expr]
            slotified_program = slotified_program.replace(expr, slot_id)

        return slotified_program, slots

    def _extract_conditions(
        self,
        program: str
    ) -> Tuple[str, Dict[str, Slot]]:
        """
        条件をスロットに変換

        Args:
            program: プログラム文字列

        Returns:
            (slotified_program, slots): スロット化されたプログラムとスロットの辞書
        """
        slots = {}
        slotified_program = program

        # 条件パターンを検出（例: EQUAL(...), LESS(...)など）
        condition_pattern = r'\b(?:EQUAL|NOT_EQUAL|LESS|GREATER|LESS_EQUAL|GREATER_EQUAL|AND|OR)\([^)]*\)'
        conditions = re.findall(condition_pattern, program)

        # 各条件をスロットに変換
        condition_to_slot = {}
        for cond in set(conditions):
            slot_id = f"slot_{self.slot_counter}"
            self.slot_counter += 1

            # 条件の型と制約を推測
            constraints = self._infer_condition_constraints(cond)

            slot = Slot(
                slot_id=slot_id,
                slot_type=SlotType.CONDITION,
                original_value=cond,
                constraints=constraints,
                metadata={"condition": cond}
            )

            slots[slot_id] = slot
            condition_to_slot[cond] = slot_id

        # 条件をスロットIDに置換（長い条件から順に置換）
        sorted_conditions = sorted(condition_to_slot.keys(), key=len, reverse=True)
        for cond in sorted_conditions:
            slot_id = condition_to_slot[cond]
            slotified_program = slotified_program.replace(cond, slot_id)

        return slotified_program, slots

    def _infer_variable_constraints(self, variable: str) -> Dict[str, Any]:
        """
        変数の制約を推測

        Args:
            variable: 変数名

        Returns:
            constraints: 制約の辞書
        """
        constraints = {
            "type": "object_list",
            "required": True
        }

        # 変数名から情報を推測
        if "objects" in variable.lower():
            constraints["type"] = "object_list"

        return constraints

    def _infer_expression_constraints(self, expression: str) -> Dict[str, Any]:
        """
        式の制約を推測

        Args:
            expression: 式文字列

        Returns:
            constraints: 制約の辞書
        """
        constraints = {
            "type": "unknown",
            "required": True
        }

        # 関数名から型を推測
        if expression.startswith("FILTER"):
            constraints["type"] = "object_list"
        elif expression.startswith("GET_"):
            constraints["type"] = "value"
        elif expression.startswith("MOVE") or expression.startswith("ROTATE"):
            constraints["type"] = "object_list"

        return constraints

    def _infer_condition_constraints(self, condition: str) -> Dict[str, Any]:
        """
        条件の制約を推測

        Args:
            condition: 条件文字列

        Returns:
            constraints: 制約の辞書
        """
        constraints = {
            "type": "boolean",
            "required": True
        }

        # 条件タイプを推測
        if condition.startswith("EQUAL") or condition.startswith("NOT_EQUAL"):
            constraints["comparison_type"] = "equality"
        elif condition.startswith("LESS") or condition.startswith("GREATER"):
            constraints["comparison_type"] = "inequality"
        elif condition.startswith("AND") or condition.startswith("OR"):
            constraints["logical_type"] = "logical"

        return constraints

    def fill_slots(
        self,
        slotified_program: str,
        slots: Dict[str, Slot],
        slot_values: Dict[str, str]
    ) -> str:
        """
        スロットを値で埋める

        Args:
            slotified_program: スロット化されたプログラム
            slots: スロットの辞書
            slot_values: スロットIDから値へのマッピング

        Returns:
            filled_program: スロットが埋められたプログラム
        """
        filled_program = slotified_program

        # 各スロットを値で置換
        for slot_id, value in slot_values.items():
            if slot_id in slots:
                # 制約をチェック
                if self._check_constraints(slots[slot_id], value):
                    filled_program = filled_program.replace(slot_id, value)
                else:
                    # 制約違反の場合は元の値を使用
                    filled_program = filled_program.replace(slot_id, slots[slot_id].original_value)

        return filled_program

    def _check_constraints(
        self,
        slot: Slot,
        value: str
    ) -> bool:
        """
        スロットの制約をチェック

        Args:
            slot: スロット
            value: 値

        Returns:
            is_valid: 制約を満たしているか
        """
        constraints = slot.constraints

        # 型チェック
        if "type" in constraints:
            expected_type = constraints["type"]

            if expected_type == "object_list":
                # オブジェクトリストの場合: "objects"で始まる変数名または関数呼び出し
                if not (value.startswith("objects") or
                        any(func in value for func in ["GET_ALL_OBJECTS", "FILTER", "SELECT"])):
                    return False

            elif expected_type == "boolean":
                # ブール値の場合: True/Falseまたは条件式
                if value not in ["True", "False"]:
                    # 条件式として妥当かチェック（EQUAL, LESS等で始まるか、AND/ORで結合されているか）
                    if not any(cond_op in value.upper() for cond_op in
                              ["EQUAL", "NOT_EQUAL", "LESS", "GREATER", "LESS_EQUAL", "GREATER_EQUAL"]):
                        return False

            elif expected_type == "value":
                # 値の場合: 数値、文字列、またはGET_*関数呼び出し
                try:
                    # 数値として解析可能か
                    float(value)
                    return True
                except ValueError:
                    # 文字列リテラルまたは関数呼び出しか
                    if value.startswith('"') and value.endswith('"'):
                        return True
                    if any(func in value for func in ["GET_", "COUNT_", "ARRAY_LENGTH"]):
                        return True
                    return False

        # 比較タイプのチェック（条件スロットの場合）
        if "comparison_type" in constraints:
            comparison_type = constraints["comparison_type"]
            if comparison_type == "equality":
                if not any(op in value.upper() for op in ["EQUAL", "NOT_EQUAL"]):
                    return False
            elif comparison_type == "inequality":
                if not any(op in value.upper() for op in ["LESS", "GREATER", "LESS_EQUAL", "GREATER_EQUAL"]):
                    return False

        # 論理タイプのチェック（条件スロットの場合）
        if "logical_type" in constraints:
            logical_type = constraints["logical_type"]
            if logical_type == "logical":
                if not any(op in value.upper() for op in ["AND", "OR"]):
                    return False

        return True

    def get_slot_dependencies(
        self,
        slots: Dict[str, Slot]
    ) -> Dict[str, List[str]]:
        """
        スロットの依存関係を取得

        Args:
            slots: スロットの辞書

        Returns:
            dependencies: スロットIDから依存スロットIDのリストへのマッピング
        """
        dependencies = {}

        for slot_id, slot in slots.items():
            deps = []

            # スロットの元の値に含まれる他のスロットを検出
            for other_slot_id, other_slot in slots.items():
                if other_slot_id != slot_id:
                    if other_slot.original_value in slot.original_value:
                        deps.append(other_slot_id)

            dependencies[slot_id] = deps

        return dependencies
