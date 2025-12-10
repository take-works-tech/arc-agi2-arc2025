#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
操作タイプとデータ構造の定義

すべての操作タイプ、Operation、OperationResultを統合
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field

from src.data_systems.data_models.core.object import Object


class OperationType(Enum):
    """統合操作タイプ（全52種類）"""

    # 基本操作（7種類）
    MOVE = "MOVE"
    TELEPORT = "TELEPORT"
    FILL_HOLES = "FILL_HOLES"
    EXPAND = "EXPAND"
    ROTATE = "ROTATE"
    SCALE = "SCALE"

    # 分割操作（1種類）- SPLIT系で統一
    SPLIT_CONNECTED = "SPLIT_CONNECTED"


    # 生成操作（4種類）
    CREATE_LINE = "CREATE_LINE"
    CREATE_RECT = "CREATE_RECT"
    TILE = "TILE"

    # 幾何学的操作（4種類）
    FLIP = "FLIP"
    OUTLINE = "OUTLINE"
    HOLLOW = "HOLLOW"
    SUBTRACT = "SUBTRACT"
    INTERSECTION = "INTERSECTION"


    # 対称性分析（1種類）
    GET_SYMMETRY_SCORE = "GET_SYMMETRY_SCORE"

    # 特殊操作（3種類）
    DRAW = "DRAW"
    LAY = "LAY"

    # 解析・参照操作
    GET_ALL_OBJECTS = "GET_ALL_OBJECTS"
    FILTER = "FILTER"
    GET_INPUT_GRID_SIZE = "GET_INPUT_GRID_SIZE"
    RENDER_GRID = "RENDER_GRID"
    SORT_BY = "SORT_BY"
    LEN = "LEN"
    DIVIDE = "DIVIDE"
    ADD = "ADD"
    SUB = "SUB"
    ARRANGE_GRID = "ARRANGE_GRID"
    APPEND = "APPEND"
    CONCAT = "CONCAT"
    MERGE = "MERGE"
    SET_COLOR = "SET_COLOR"
    GET_COLOR = "GET_COLOR"
    GET_COLORS = "GET_COLORS"
    GET_X = "GET_X"
    GET_Y = "GET_Y"
    GET_Y_DISTANCE = "GET_Y_DISTANCE"
    GET_WIDTH = "GET_WIDTH"
    GET_HEIGHT = "GET_HEIGHT"
    COUNT_HOLES = "COUNT_HOLES"
    CROP = "CROP"
    FIT_ADJACENT = "FIT_ADJACENT"
    FIT_SHAPE_COLOR = "FIT_SHAPE_COLOR"
    EXTEND_PATTERN = "EXTEND_PATTERN"
    MATCH_PAIRS = "MATCH_PAIRS"
    MULTIPLY = "MULTIPLY"
    BBOX = "BBOX"

    # 比較演算
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    GREATER = "GREATER"
    LESS = "LESS"

    # 論理演算
    AND = "AND"
    OR = "OR"

    # 算術演算
    MOD = "MOD"

    # 情報取得
    GET_SIZE = "GET_SIZE"
    GET_LINE_TYPE = "GET_LINE_TYPE"
    GET_RECTANGLE_TYPE = "GET_RECTANGLE_TYPE"
    GET_DISTANCE = "GET_DISTANCE"
    GET_X_DISTANCE = "GET_X_DISTANCE"
    COUNT_ADJACENT = "COUNT_ADJACENT"
    COUNT_OVERLAP = "COUNT_OVERLAP"
    GET_BACKGROUND_COLOR = "GET_BACKGROUND_COLOR"
    GET_ASPECT_RATIO = "GET_ASPECT_RATIO"
    GET_DENSITY = "GET_DENSITY"
    GET_CENTROID = "GET_CENTROID"
    GET_CENTER_X = "GET_CENTER_X"
    GET_CENTER_Y = "GET_CENTER_Y"
    GET_MAX_X = "GET_MAX_X"
    GET_MAX_Y = "GET_MAX_Y"
    GET_DIRECTION = "GET_DIRECTION"
    GET_NEAREST = "GET_NEAREST"

    # 判定関数
    IS_INSIDE = "IS_INSIDE"
    IS_SAME_SHAPE = "IS_SAME_SHAPE"
    IS_SAME_STRUCT = "IS_SAME_STRUCT"
    IS_IDENTICAL = "IS_IDENTICAL"

    # 変換操作（追加）
    SLIDE = "SLIDE"
    PATHFIND = "PATHFIND"
    SCALE_DOWN = "SCALE_DOWN"
    FLOW = "FLOW"
    FIT_SHAPE = "FIT_SHAPE"
    ALIGN = "ALIGN"

    # オブジェクト抽出
    EXTRACT_RECTS = "EXTRACT_RECTS"
    EXTRACT_HOLLOW_RECTS = "EXTRACT_HOLLOW_RECTS"
    EXTRACT_LINES = "EXTRACT_LINES"

    # 配列操作
    EXCLUDE = "EXCLUDE"
    REVERSE = "REVERSE"

    # テンプレート補助操作
    DIRECT_ASSIGN = "DIRECT_ASSIGN"




@dataclass
class Operation:
    """操作定義（統合版）"""
    type: OperationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_object_id: str = ""
    cost: float = 1.0


@dataclass
class OperationResult:
    """操作結果（統合版）"""
    success: bool
    message: str = ""
    affected_objects: List[Object] = field(default_factory=list)
    new_objects: List[Object] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    'OperationType',
    'Operation',
    'OperationResult'
]
