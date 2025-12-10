#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コマンド引数からテンプレート引数スロットを生成
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.hybrid_system.ir.structures import ArgumentSlot, ArgumentValue, ArgumentKind
from .command_registry import CommandTemplate, CommandArgumentSpec


class ArgumentResolutionError(ValueError):
    """引数変換時のエラー"""


def resolve_argument_slots(
    template: CommandTemplate,
    raw_args: List[Any],
    context: Optional[Dict[str, Any]] = None,
) -> List[ArgumentSlot]:
    """
    コマンド呼び出しの引数をテンプレート引数スロットへ変換

    Args:
        template: コマンドテンプレート
        raw_args: DSLで指定された生引数
        context: 追加情報（シンボル表、推定結果など）

    Returns:
        ArgumentSlot のリスト
    """
    slots: List[ArgumentSlot] = []
    context = context or {}

    for index, spec in enumerate(template.argument_specs):
        raw_value = raw_args[index] if index < len(raw_args) else None

        if raw_value is None and not spec.optional:
            raise ArgumentResolutionError(
                f"{template.command_name}: 引数 '{spec.name}' が不足しています"
            )

        value = _create_argument_value(spec, raw_value, context)
        if value.kind == ArgumentKind.PLACEHOLDER:
            if spec.placeholder_hint and "hint" not in value.constraints:
                value.constraints["hint"] = spec.placeholder_hint
            if spec.candidates and "candidates" not in value.constraints:
                value.constraints["candidates"] = list(spec.candidates)

        slot = ArgumentSlot(
            name=spec.name,
            value=value,
            value_type=spec.value_type,
            optional=spec.optional,
            notes=spec.description,
        )
        slots.append(slot)

    # 余剰引数がある場合はメタデータに追加
    if len(raw_args) > len(template.argument_specs):
        extra_args = raw_args[len(template.argument_specs) :]
        slots.append(
            ArgumentSlot(
                name="__extra__",
                value=ArgumentValue(
                    kind=ArgumentKind.DERIVED,
                    value=extra_args,
                    constraints={"reason": "extra_arguments"},
                    description="DSLで追加指定された引数",
                ),
                value_type="list",
                optional=True,
                notes="追加引数",
            )
        )

    return slots


def _create_argument_value(
    spec: CommandArgumentSpec,
    raw_value: Any,
    context: Dict[str, Any],
) -> ArgumentValue:
    """引数スペックに基づき ArgumentValue を生成"""
    if raw_value is None:
        return ArgumentValue(kind=spec.default_kind, value=None)

    inferred_kind = _infer_kind_from_value(spec, raw_value)
    value = raw_value
    constraints: Dict[str, Any] = {}

    if inferred_kind == ArgumentKind.REFERENCE:
        value, constraints = _normalize_reference(raw_value, context)
    elif inferred_kind == ArgumentKind.PLACEHOLDER:
        constraints = {"hint": raw_value}
        value = None
    elif inferred_kind == ArgumentKind.LITERAL:
        if isinstance(raw_value, str) and _should_cast_to_int(spec.value_type, raw_value):
            try:
                value = int(raw_value)
            except ValueError:
                pass

    return ArgumentValue(
        kind=inferred_kind,
        value=value,
        constraints=constraints,
    )


def _infer_kind_from_value(
    spec: CommandArgumentSpec,
    raw_value: Any,
) -> ArgumentKind:
    """値からArgumentKindを推定"""
    if raw_value is None:
        return spec.default_kind

    if spec.value_type == "reference":
        # プレースホルダ指定は優先的に反映
        if isinstance(raw_value, str) and raw_value.startswith("${"):
            return ArgumentKind.PLACEHOLDER
        return ArgumentKind.REFERENCE

    if spec.value_type == "expression":
        if isinstance(raw_value, str) and raw_value.startswith("${"):
            return ArgumentKind.PLACEHOLDER
        return ArgumentKind.DERIVED

    if spec.value_type == "enum":
        if isinstance(raw_value, str):
            normalized = raw_value.strip("'\"")
            if spec.candidates and normalized in spec.candidates:
                return ArgumentKind.LITERAL
        if isinstance(raw_value, str) and raw_value.startswith("${"):
            return ArgumentKind.PLACEHOLDER
        if spec.default_kind == ArgumentKind.PLACEHOLDER:
            return ArgumentKind.PLACEHOLDER
        return ArgumentKind.LITERAL

    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        if normalized.startswith(("input@", "derived@", "temp@")):
            return ArgumentKind.REFERENCE
        if normalized.startswith("${"):
            return ArgumentKind.PLACEHOLDER

        if spec.value_type in {"expression"}:
            return ArgumentKind.DERIVED

        if any(tok in normalized for tok in ("(", ")", "[", "]", "{", "}", ".")):
            return ArgumentKind.DERIVED

        if spec.value_type in {"int", "integer", "count"}:
            if normalized.lstrip("-").isdigit():
                return ArgumentKind.LITERAL
            if normalized.isidentifier():
                return ArgumentKind.DERIVED
            return ArgumentKind.PLACEHOLDER

        if spec.value_type == "color":
            if normalized.isdigit():
                return ArgumentKind.LITERAL
            return ArgumentKind.PLACEHOLDER

    return ArgumentKind.LITERAL


def _should_cast_to_int(value_type: str, raw_value: str) -> bool:
    candidate_types = {"int", "integer", "count", "color"}
    return value_type in candidate_types and raw_value.strip().lstrip("-").isdigit()


def _normalize_reference(
    raw_value: Any,
    context: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """参照表記を標準化"""
    constraints: Dict[str, Any] = {}
    value = raw_value

    if isinstance(raw_value, str):
        if raw_value == "CURRENT_OBJECT":
            value = context.get("current_object_binding", "derived@current")
            constraints["source"] = "current_object"
        elif raw_value.startswith("input@"):
            constraints["source"] = "input"
        elif raw_value.startswith("derived@"):
            constraints["source"] = "derived"

    return value, constraints
