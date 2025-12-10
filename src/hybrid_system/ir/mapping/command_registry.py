#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSLコマンドとテンプレート操作の対応表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core_systems.executor.operations.types import OperationType
from src.hybrid_system.ir.structures import ArgumentKind


@dataclass(frozen=True)
class CommandArgumentSpec:
    """コマンド引数の仕様"""

    name: str
    value_type: str
    default_kind: ArgumentKind = ArgumentKind.LITERAL
    optional: bool = False
    description: str = ""
    candidates: Optional[List[Any]] = None
    placeholder_hint: Optional[str] = None


@dataclass(frozen=True)
class CommandTemplate:
    """DSLコマンドをテンプレート操作へ変換する際の定義"""

    command_name: str
    operation: OperationType
    argument_specs: List[CommandArgumentSpec] = field(default_factory=list)
    target_binding_required: bool = False
    notes: str = ""


_COMMAND_REGISTRY: Dict[str, CommandTemplate] = {}


def register_command(template: CommandTemplate) -> None:
    """コマンド定義を登録"""
    key = template.command_name.upper()
    _COMMAND_REGISTRY[key] = template


def get_command_template(command_name: str) -> Optional[CommandTemplate]:
    """コマンド名に対応するテンプレートを取得"""
    return _COMMAND_REGISTRY.get(command_name.upper())


def resolve_command_template(command_name: str) -> CommandTemplate:
    """コマンドに対応するテンプレートを返す。未登録なら例外。"""
    template = get_command_template(command_name)
    if template is None:
        raise KeyError(f"未対応のDSLコマンドです: {command_name}")
    return template


def list_supported_commands() -> List[str]:
    """登録済みコマンド一覧"""
    return sorted(_COMMAND_REGISTRY.keys())


def _initialize_default_registry() -> None:
    """代表的なコマンド定義を初期登録"""
    register_command(
        CommandTemplate(
            command_name="MOVE",
            operation=OperationType.MOVE,
            argument_specs=[
                CommandArgumentSpec("dx", "int"),
                CommandArgumentSpec("dy", "int"),
                CommandArgumentSpec(
                    "mode",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=["relative", "absolute"],
                    placeholder_hint="relative",
                ),
            ],
            target_binding_required=True,
            notes="任意の対象オブジェクトを指定方向へ移動",
        )
    )

    register_command(
        CommandTemplate(
            command_name="ROTATE",
            operation=OperationType.ROTATE,
            argument_specs=[
                CommandArgumentSpec("angle", "enum"),
                CommandArgumentSpec(
                    "pivot",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=["center", "origin", "object"],
                    placeholder_hint="center",
                ),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="FLIP",
            operation=OperationType.FLIP,
            argument_specs=[
                CommandArgumentSpec("axis", "enum"),
                CommandArgumentSpec(
                    "pivot",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=["center", "origin", "object"],
                    placeholder_hint="center",
                ),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="EXPAND",
            operation=OperationType.EXPAND,
            argument_specs=[
                CommandArgumentSpec("pixels", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="CREATE_RECT",
            operation=OperationType.CREATE_RECT,
            argument_specs=[
                CommandArgumentSpec("width", "int"),
                CommandArgumentSpec("height", "int"),
                CommandArgumentSpec("fill_mode", "enum"),
                CommandArgumentSpec("color", "color"),
                CommandArgumentSpec(
                    "anchor",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=[
                        "center",
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                    ],
                    placeholder_hint="center",
                ),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="CREATE_LINE",
            operation=OperationType.CREATE_LINE,
            argument_specs=[
                CommandArgumentSpec(
                    "start_x",
                    "expression",
                    default_kind=ArgumentKind.DERIVED,
                ),
                CommandArgumentSpec(
                    "start_y",
                    "expression",
                    default_kind=ArgumentKind.DERIVED,
                ),
                CommandArgumentSpec(
                    "length",
                    "expression",
                    default_kind=ArgumentKind.DERIVED,
                    placeholder_hint="grid_size[0]",
                ),
                CommandArgumentSpec(
                    "direction",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    candidates=["X", "-X", "Y", "-Y", "XY", "-XY", "X-Y", "-X-Y"],
                    placeholder_hint="X",
                ),
                CommandArgumentSpec(
                    "color",
                    "reference",
                    default_kind=ArgumentKind.REFERENCE,
                ),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="HOLLOW",
            operation=OperationType.HOLLOW,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="OUTLINE",
            operation=OperationType.OUTLINE,
            argument_specs=[
                CommandArgumentSpec("color", "color"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="SUBTRACT",
            operation=OperationType.SUBTRACT,
            argument_specs=[
                CommandArgumentSpec("mask_binding", "reference"),
                CommandArgumentSpec(
                    "mode",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=["difference", "symmetric", "mask"],
                    placeholder_hint="difference",
                ),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="INTERSECTION",
            operation=OperationType.INTERSECTION,
            argument_specs=[
                CommandArgumentSpec("other_binding", "reference"),
                CommandArgumentSpec(
                    "mode",
                    "enum",
                    default_kind=ArgumentKind.PLACEHOLDER,
                    optional=True,
                    candidates=["overlap", "mask"],
                ),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_ALL_OBJECTS",
            operation=OperationType.GET_ALL_OBJECTS,
            argument_specs=[
                CommandArgumentSpec("connectivity", "int", optional=True),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="FILTER",
            operation=OperationType.FILTER,
            argument_specs=[
                CommandArgumentSpec("source_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("predicate", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="FILL_HOLES",
            operation=OperationType.FILL_HOLES,
            argument_specs=[
                CommandArgumentSpec("fill_color", "color"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_INPUT_GRID_SIZE",
            operation=OperationType.GET_INPUT_GRID_SIZE,
        )
    )

    register_command(
        CommandTemplate(
            command_name="RENDER_GRID",
            operation=OperationType.RENDER_GRID,
            argument_specs=[
                CommandArgumentSpec("objects_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("background_color", "color", default_kind=ArgumentKind.PLACEHOLDER),
                CommandArgumentSpec("width", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("height", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="LEN",
            operation=OperationType.LEN,
            argument_specs=[
                CommandArgumentSpec("sequence_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="SORT_BY",
            operation=OperationType.SORT_BY,
            argument_specs=[
                CommandArgumentSpec("sequence_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("key_expression", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("order", "enum", optional=True),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="DIVIDE",
            operation=OperationType.DIVIDE,
            argument_specs=[
                CommandArgumentSpec("left", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("right", "int", default_kind=ArgumentKind.PLACEHOLDER),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="ADD",
            operation=OperationType.ADD,
            argument_specs=[
                CommandArgumentSpec("left", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("right", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="ARRANGE_GRID",
            operation=OperationType.ARRANGE_GRID,
            argument_specs=[
                CommandArgumentSpec("objects_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("rows", "int", optional=True),
                CommandArgumentSpec("cols", "int", optional=True),
                CommandArgumentSpec("fill_value", "expression", default_kind=ArgumentKind.DERIVED, optional=True),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="CROP",
            operation=OperationType.CROP,
            argument_specs=[
                CommandArgumentSpec("x", "int"),
                CommandArgumentSpec("y", "int"),
                CommandArgumentSpec("width", "int"),
                CommandArgumentSpec("height", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="FIT_ADJACENT",
            operation=OperationType.FIT_ADJACENT,
            argument_specs=[
                CommandArgumentSpec("reference_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="FIT_SHAPE_COLOR",
            operation=OperationType.FIT_SHAPE_COLOR,
            argument_specs=[
                CommandArgumentSpec("reference_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="EXTEND_PATTERN",
            operation=OperationType.EXTEND_PATTERN,
            argument_specs=[
                CommandArgumentSpec("side", "enum"),
                CommandArgumentSpec("count", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="MATCH_PAIRS",
            operation=OperationType.MATCH_PAIRS,
            argument_specs=[
                CommandArgumentSpec("reference_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("condition", "expression", default_kind=ArgumentKind.DERIVED),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="MULTIPLY",
            operation=OperationType.MULTIPLY,
            argument_specs=[
                CommandArgumentSpec("left", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("right", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="BBOX",
            operation=OperationType.BBOX,
            argument_specs=[
                CommandArgumentSpec("color", "color", optional=True),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="SUB",
            operation=OperationType.SUB,
            argument_specs=[
                CommandArgumentSpec("left", "expression", default_kind=ArgumentKind.DERIVED),
                CommandArgumentSpec("right", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="APPEND",
            operation=OperationType.APPEND,
            argument_specs=[
                CommandArgumentSpec("sequence_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("value", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="CONCAT",
            operation=OperationType.CONCAT,
            argument_specs=[
                CommandArgumentSpec("left_sequence", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("right_sequence", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="MERGE",
            operation=OperationType.MERGE,
            argument_specs=[
                CommandArgumentSpec("objects", "expression", default_kind=ArgumentKind.DERIVED),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="SET_COLOR",
            operation=OperationType.SET_COLOR,
            argument_specs=[
                CommandArgumentSpec("color", "color", default_kind=ArgumentKind.PLACEHOLDER),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="COUNT_HOLES",
            operation=OperationType.COUNT_HOLES,
            argument_specs=[
                CommandArgumentSpec(
                    "object_binding",
                    "reference",
                    default_kind=ArgumentKind.REFERENCE,
                ),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_COLOR",
            operation=OperationType.GET_COLOR,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_COLORS",
            operation=OperationType.GET_COLORS,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_X",
            operation=OperationType.GET_X,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_Y",
            operation=OperationType.GET_Y,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_WIDTH",
            operation=OperationType.GET_WIDTH,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_HEIGHT",
            operation=OperationType.GET_HEIGHT,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_Y_DISTANCE",
            operation=OperationType.GET_Y_DISTANCE,
            argument_specs=[
                CommandArgumentSpec("source_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("target_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    register_command(
        CommandTemplate(
            command_name="TELEPORT",
            operation=OperationType.TELEPORT,
            argument_specs=[
                CommandArgumentSpec("x", "int"),
                CommandArgumentSpec("y", "int"),
            ],
            target_binding_required=True,
        )
    )

    # ==================== 比較演算 ====================
    register_command(
        CommandTemplate(
            command_name="EQUAL",
            operation=OperationType.EQUAL,
            argument_specs=[
                CommandArgumentSpec("a", "any"),
                CommandArgumentSpec("b", "any"),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="NOT_EQUAL",
            operation=OperationType.NOT_EQUAL,
            argument_specs=[
                CommandArgumentSpec("a", "any"),
                CommandArgumentSpec("b", "any"),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GREATER",
            operation=OperationType.GREATER,
            argument_specs=[
                CommandArgumentSpec("a", "int"),
                CommandArgumentSpec("b", "int"),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="LESS",
            operation=OperationType.LESS,
            argument_specs=[
                CommandArgumentSpec("a", "int"),
                CommandArgumentSpec("b", "int"),
            ],
            target_binding_required=False,
        )
    )

    # ==================== 論理演算 ====================
    register_command(
        CommandTemplate(
            command_name="AND",
            operation=OperationType.AND,
            argument_specs=[
                CommandArgumentSpec("a", "bool"),
                CommandArgumentSpec("b", "bool"),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="OR",
            operation=OperationType.OR,
            argument_specs=[
                CommandArgumentSpec("a", "bool"),
                CommandArgumentSpec("b", "bool"),
            ],
            target_binding_required=False,
        )
    )

    # ==================== 算術演算 ====================
    register_command(
        CommandTemplate(
            command_name="MOD",
            operation=OperationType.MOD,
            argument_specs=[
                CommandArgumentSpec("a", "int"),
                CommandArgumentSpec("b", "int"),
            ],
            target_binding_required=False,
        )
    )

    # ==================== 情報取得 ====================
    register_command(
        CommandTemplate(
            command_name="GET_SIZE",
            operation=OperationType.GET_SIZE,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_SYMMETRY_SCORE",
            operation=OperationType.GET_SYMMETRY_SCORE,
            argument_specs=[
                CommandArgumentSpec("axis", "str"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_LINE_TYPE",
            operation=OperationType.GET_LINE_TYPE,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_RECTANGLE_TYPE",
            operation=OperationType.GET_RECTANGLE_TYPE,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_DISTANCE",
            operation=OperationType.GET_DISTANCE,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_X_DISTANCE",
            operation=OperationType.GET_X_DISTANCE,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="COUNT_ADJACENT",
            operation=OperationType.COUNT_ADJACENT,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="COUNT_OVERLAP",
            operation=OperationType.COUNT_OVERLAP,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_BACKGROUND_COLOR",
            operation=OperationType.GET_BACKGROUND_COLOR,
            argument_specs=[],
            target_binding_required=False,
        )
    )

    # ==================== 判定関数 ====================
    register_command(
        CommandTemplate(
            command_name="IS_INSIDE",
            operation=OperationType.IS_INSIDE,
            argument_specs=[
                CommandArgumentSpec("x", "int"),
                CommandArgumentSpec("y", "int"),
                CommandArgumentSpec("width", "int"),
                CommandArgumentSpec("height", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="IS_SAME_SHAPE",
            operation=OperationType.IS_SAME_SHAPE,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="IS_SAME_STRUCT",
            operation=OperationType.IS_SAME_STRUCT,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="IS_IDENTICAL",
            operation=OperationType.IS_IDENTICAL,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    # ==================== 変換操作 ====================
    register_command(
        CommandTemplate(
            command_name="SLIDE",
            operation=OperationType.SLIDE,
            argument_specs=[
                CommandArgumentSpec("direction", "str"),
                CommandArgumentSpec("obstacles", "array"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="PATHFIND",
            operation=OperationType.PATHFIND,
            argument_specs=[
                CommandArgumentSpec("target_x", "int"),
                CommandArgumentSpec("target_y", "int"),
                CommandArgumentSpec("obstacles", "array"),
            ],
            target_binding_required=True,
            notes="経路探索移動（障害物回避、8方向対応）",
        )
    )

    register_command(
        CommandTemplate(
            command_name="SCALE",
            operation=OperationType.SCALE,
            argument_specs=[
                CommandArgumentSpec("factor", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="SCALE_DOWN",
            operation=OperationType.SCALE_DOWN,
            argument_specs=[
                CommandArgumentSpec("divisor", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="FLOW",
            operation=OperationType.FLOW,
            argument_specs=[
                CommandArgumentSpec("direction", "str"),
                CommandArgumentSpec("obstacles", "array"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="DRAW",
            operation=OperationType.DRAW,
            argument_specs=[
                CommandArgumentSpec("x", "int"),
                CommandArgumentSpec("y", "int"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="LAY",
            operation=OperationType.LAY,
            argument_specs=[
                CommandArgumentSpec("direction", "str"),
                CommandArgumentSpec("obstacles", "array"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="FIT_SHAPE",
            operation=OperationType.FIT_SHAPE,
            argument_specs=[
                CommandArgumentSpec("obj2", "object"),
            ],
            target_binding_required=True,
        )
    )

    # ==================== オブジェクト分割 ====================
    register_command(
        CommandTemplate(
            command_name="SPLIT_CONNECTED",
            operation=OperationType.SPLIT_CONNECTED,
            argument_specs=[
                CommandArgumentSpec("connectivity", "int"),
            ],
            target_binding_required=True,
        )
    )

    # ==================== オブジェクト抽出 ====================
    register_command(
        CommandTemplate(
            command_name="EXTRACT_RECTS",
            operation=OperationType.EXTRACT_RECTS,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="EXTRACT_HOLLOW_RECTS",
            operation=OperationType.EXTRACT_HOLLOW_RECTS,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="EXTRACT_LINES",
            operation=OperationType.EXTRACT_LINES,
            argument_specs=[],
            target_binding_required=True,
        )
    )

    # ==================== 配列操作 ====================
    register_command(
        CommandTemplate(
            command_name="EXCLUDE",
            operation=OperationType.EXCLUDE,
            argument_specs=[
                CommandArgumentSpec("targets", "array"),
            ],
            target_binding_required=True,
        )
    )

    register_command(
        CommandTemplate(
            command_name="REVERSE",
            operation=OperationType.REVERSE,
            argument_specs=[
                CommandArgumentSpec("sequence_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
        )
    )

    # ==================== 新しい情報取得コマンド ====================
    register_command(
        CommandTemplate(
            command_name="GET_ASPECT_RATIO",
            operation=OperationType.GET_ASPECT_RATIO,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_DENSITY",
            operation=OperationType.GET_DENSITY,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_CENTROID",
            operation=OperationType.GET_CENTROID,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_CENTER_X",
            operation=OperationType.GET_CENTER_X,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_CENTER_Y",
            operation=OperationType.GET_CENTER_Y,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_MAX_X",
            operation=OperationType.GET_MAX_X,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_MAX_Y",
            operation=OperationType.GET_MAX_Y,
            argument_specs=[
                CommandArgumentSpec("object_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_DIRECTION",
            operation=OperationType.GET_DIRECTION,
            argument_specs=[
                CommandArgumentSpec("source_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("target_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    register_command(
        CommandTemplate(
            command_name="GET_NEAREST",
            operation=OperationType.GET_NEAREST,
            argument_specs=[
                CommandArgumentSpec("source_binding", "reference", default_kind=ArgumentKind.REFERENCE),
                CommandArgumentSpec("candidates_binding", "reference", default_kind=ArgumentKind.REFERENCE),
            ],
            target_binding_required=False,
        )
    )

    # ==================== 新しい変換操作 ====================
    register_command(
        CommandTemplate(
            command_name="ALIGN",
            operation=OperationType.ALIGN,
            argument_specs=[
                CommandArgumentSpec("mode", "enum", candidates=["left", "right", "top", "bottom", "center_x", "center_y", "center"]),
            ],
            target_binding_required=True,
        )
    )

    # ==================== 新しい生成操作 ====================
    register_command(
        CommandTemplate(
            command_name="TILE",
            operation=OperationType.TILE,
            argument_specs=[
                CommandArgumentSpec("count_x", "int"),
                CommandArgumentSpec("count_y", "int"),
            ],
            target_binding_required=True,
        )
    )


_initialize_default_registry()
