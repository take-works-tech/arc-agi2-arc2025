"""
テンプレート列からDSLコードへの変換および実行補助。
"""

from __future__ import annotations

from typing import List

from src.core_systems.executor.operations.types import OperationType
from src.hybrid_system.ir.structures import IRSequence, TemplateStep, ArgumentSlot, ArgumentKind
from src.hybrid_system.ir.mapping import get_command_template


def _sanitize_expression(expr: str) -> str:
    if not isinstance(expr, str):
        return str(expr)
    return expr.replace("${", "$").replace("}", "")


def _format_list_value(items) -> str:
    formatted: List[str] = []
    for item in items:
        if isinstance(item, list):
            formatted.append(_format_list_value(item))
            continue
        if isinstance(item, str):
            sanitized = _sanitize_expression(item)
            if sanitized.startswith(("'", '"')):
                formatted.append(sanitized)
            else:
                formatted.append(sanitized)
            continue
        formatted.append(_sanitize_expression(item))
    return "[" + ", ".join(formatted) + "]"


def _argument_value_to_dsl(slot: ArgumentSlot) -> str:
    value_dict = slot.value.to_dict()
    kind = value_dict.get("kind", ArgumentKind.LITERAL.value)
    value = value_dict.get("value")

    if kind == ArgumentKind.LITERAL.value:
        if isinstance(value, list):
            return _format_list_value(value)
        if isinstance(value, str) and not value.startswith(("'", '"')):
            alnum_check = value.replace("_", "")
            if alnum_check.isalnum():
                value = f"'{value}'"
        return _sanitize_expression(value)
    if kind == ArgumentKind.REFERENCE.value:
        if isinstance(value, list):
            return _format_list_value(value)
        return _sanitize_expression(value)
    if kind == ArgumentKind.PLACEHOLDER.value:
        placeholder_hint = value_dict.get("constraints", {}).get("hint")
        if placeholder_hint:
            return _sanitize_expression(placeholder_hint)
        return f"${slot.name}"
    if kind == ArgumentKind.DERIVED.value:
        if isinstance(value, list):
            return _format_list_value(value)
        return _sanitize_expression(value)
    return _sanitize_expression(value)


def template_step_to_dsl(step: TemplateStep) -> str:
    if step.operation == OperationType.DIRECT_ASSIGN:
        assignment_target = (
            step.metadata.get("assignment_target")
            or step.metadata.get("target_expression")
            or step.target_binding
        )
        if not assignment_target:
            assignment_target = step.metadata.get("assignment_target_fallback", "UNKNOWN")

        assignment_expression = step.metadata.get("assignment_expression")
        if not assignment_expression:
            if step.argument_slots:
                assignment_expression = _argument_value_to_dsl(step.argument_slots[0])
            else:
                assignment_expression = "None"
        else:
            assignment_expression = _sanitize_expression(assignment_expression)

        return f"{assignment_target} = {assignment_expression}"

    command_name = (
        step.metadata.get("normalized_command")
        or step.metadata.get("dsl_command")
        or step.operation.value
    )
    template = get_command_template(command_name)

    args: List[str] = []
    assignment_target = step.metadata.get("assignment_target")
    target_expression = step.metadata.get("target_expression") or step.target_binding

    if template and template.target_binding_required:
        if not target_expression:
            target_expression = assignment_target or step.target_binding or "CURRENT_OBJECT"
        args.append(_sanitize_expression(target_expression))

    for argument in step.argument_slots:
        args.append(_argument_value_to_dsl(argument))

    call_expr = f"{command_name}({', '.join(args)})"

    if assignment_target:
        return f"{assignment_target} = {call_expr}"
    if template and template.target_binding_required and target_expression:
        return f"{target_expression} = {call_expr}"
    return call_expr


def _extract_condition_info(step: TemplateStep) -> tuple | None:
    cond_meta = step.metadata.get("condition")
    if not cond_meta:
        return None
    condition_repr = cond_meta.get("condition_repr") or "True"
    branch = cond_meta.get("branch", "then").lower()
    return condition_repr, branch


def _collect_condition_steps(
    steps: List[TemplateStep],
    start_index: int,
    condition_repr: str,
    branch: str,
) -> tuple[list[TemplateStep], int]:
    collected: List[TemplateStep] = []
    idx = start_index
    while idx < len(steps):
        info = _extract_condition_info(steps[idx])
        if not info or info[0] != condition_repr or info[1] != branch:
            break
        collected.append(steps[idx])
        idx += 1
    return collected, idx


def _emit_simple_step(step: TemplateStep, indent: str) -> List[str]:
    try:
        return [indent + template_step_to_dsl(step)]
    except Exception:
        return [indent + f"# Unsupported step: {step.operation.value}"]


def _loop_header(loop_entry: dict) -> str:
    loop_type = loop_entry.get("loop_type", "ForLoop")
    loop_var = _sanitize_expression(loop_entry.get("loop_var") or "i")
    count_expression = loop_entry.get("count_expression")
    iterable = loop_entry.get("iterable")
    condition_repr = loop_entry.get("condition_repr")

    if loop_type == "ForLoop":
        if count_expression:
            header_expr = _sanitize_expression(count_expression)
        elif iterable:
            header_expr = f"LEN({_sanitize_expression(iterable)})"
        else:
            header_expr = "LEN(objects)"
        return f"FOR {loop_var} {header_expr} DO"
    if loop_type == "WhileLoop":
        condition_str = _sanitize_expression(condition_repr or "True")
        return f"WHILE {condition_str} DO"
    return f"# Unsupported loop type {loop_type}"


def _build_control_stack(step: TemplateStep) -> List[dict]:
    """
    ステップから統合された制御構造スタックを構築する。
    condition_stackとloop_stackを正しい順序で統合する。

    condition_stackが外側、loop_stackが内側の順序で配置される。
    """
    stack = []

    # condition_stackを取得（外側の制御構造）
    condition_stack = step.metadata.get("condition_stack", [])
    for cond_entry in condition_stack:
        condition_repr = cond_entry.get("condition_repr", "True")
        branch = cond_entry.get("branch", "then")
        condition_id = cond_entry.get("condition_id", f"cond_{condition_repr}_{branch}")

        stack.append({
            "type": "condition",
            "data": {
                "condition_repr": condition_repr,
                "branch": branch
            },
            "id": condition_id
        })

    # loop_stackを取得（内側の制御構造）
    loop_stack = step.metadata.get("loop_stack", [])
    for loop_entry in loop_stack:
        stack.append({
            "type": "loop",
            "data": loop_entry,
            "id": loop_entry.get("loop_id", "")
        })

    return stack


def _compare_control_stacks(stack1: List[dict], stack2: List[dict]) -> int:
    """2つの制御スタックの共通プレフィックス長を返す"""
    common = 0
    while common < len(stack1) and common < len(stack2):
        if stack1[common]["id"] != stack2[common]["id"]:
            break
        common += 1
    return common


def sequence_to_dsl(sequence: IRSequence) -> str:
    """
    テンプレート列からDSLプログラム文字列を生成する（改善版）

    loop_stackとcondition情報を統合した制御スタックを使用し、
    IF/FORの入れ子構造を正確に再構築する。
    """
    lines: List[str] = []
    steps = sequence.steps
    idx = 0
    current_control_stack: List[dict] = []

    while idx < len(steps):
        step = steps[idx]

        # 統合された制御スタックを構築
        desired_control_stack = _build_control_stack(step)

        # ELSE分岐への切り替えを検出
        # condition_stackの最後の要素がthen→elseに変わる場合、
        # その内側の制御構造（loop）を全て閉じてからELSEを出力
        is_else_transition = False
        if (current_control_stack and desired_control_stack and
            current_control_stack[0]["type"] == "condition" and
            desired_control_stack[0]["type"] == "condition"):
            current_cond = current_control_stack[0]["data"]
            desired_cond = desired_control_stack[0]["data"]
            if (current_cond["condition_repr"] == desired_cond["condition_repr"] and
                current_cond["branch"] == "then" and desired_cond["branch"] == "else"):
                is_else_transition = True

        if is_else_transition:
            # THEN分岐内の全ての内側構造（loop等）を閉じる
            for depth in range(len(current_control_stack) - 1, 0, -1):
                indent = "    " * depth
                lines.append(indent + "END")
            # ELSE を出力
            indent = "    " * 0  # condition level
            lines.append(indent + "ELSE")
            # condition_stackを更新（thenをelseに置き換え）
            current_control_stack = [desired_control_stack[0]]
            # 残りの制御構造（loop等）を開く
            for new_control in desired_control_stack[1:]:
                indent = "    " * len(current_control_stack)
                if new_control["type"] == "loop":
                    header = _loop_header(new_control["data"])
                    lines.append(indent + header)
                    current_control_stack.append(new_control)
        else:
            # 通常の制御構造の処理
            # 共通プレフィックスを計算
            common = _compare_control_stacks(current_control_stack, desired_control_stack)

            # 不要な制御構造を閉じる
            for depth in range(len(current_control_stack) - 1, common - 1, -1):
                indent = "    " * depth
                lines.append(indent + "END")
            current_control_stack = current_control_stack[:common]

            # 新しい制御構造を開く
            for new_control in desired_control_stack[common:]:
                indent = "    " * len(current_control_stack)

                if new_control["type"] == "loop":
                    header = _loop_header(new_control["data"])
                    lines.append(indent + header)
                    current_control_stack.append(new_control)

                elif new_control["type"] == "condition":
                    cond_data = new_control["data"]
                    branch = cond_data["branch"]

                    if branch == "then":
                        # THEN分岐の開始
                        condition_repr = _sanitize_expression(cond_data["condition_repr"])
                        lines.append(f"{indent}IF {condition_repr} THEN")
                        current_control_stack.append(new_control)

        # 現在のステップを出力
        indent = "    " * len(current_control_stack)
        lines.extend(_emit_simple_step(step, indent=indent))
        idx += 1

    # 残りの制御構造を閉じる
    for depth in range(len(current_control_stack) - 1, -1, -1):
        indent = "    " * depth
        lines.append(indent + "END")

    return "\n".join(lines)
