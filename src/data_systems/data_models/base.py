#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ構造とオブジェクトクラス
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .object import Object

class ObjectType(Enum):
    """オブジェクトタイプ"""
    SINGLE_COLOR_4WAY = "single_color_4way"
    SINGLE_COLOR_8WAY = "single_color_8way"
    NON_BACKGROUND_4WAY = "non_background_4way"
    NON_BACKGROUND_8WAY = "non_background_8way"
    WHOLE_GRID = "whole_grid"
    CREATED = "created"  # プログラム内で作成されたオブジェクト
    # BACKGROUND_COLOR = "background_color"  # 廃止
    # MULTI_COLOR_4WAY = "multi_color_4way"  # 無効化
    # MULTI_COLOR_8WAY = "multi_color_8way"  # 無効化

@dataclass
class ObjectCluster:
    """オブジェクトクラスター"""
    cluster_id: str
    objects: List['Object'] = field(default_factory=list)
    cluster_type: str = "unknown"
    global_priority: float = 0.0
    task_specific_priority: float = 0.0
    combined_priority: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DSLOperation:
    """DSL操作"""
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    target: str = ""
    description: str = ""

@dataclass
class DSLProgram:
    """DSLプログラム"""
    operations: List[DSLOperation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Program:
    """プログラム"""
    program_id: str
    program_text: str
    operations_count: int
    complexity_score: float
    confidence_score: float

@dataclass
class TrainingExample:
    """学習例"""
    input_grid: np.ndarray
    output_grid: np.ndarray
    objects: List['Object'] = field(default_factory=list)
    program: Optional[DSLProgram] = None

@dataclass
class PriorityModelResult:
    """優先度モデル結果"""
    global_priority: float
    task_specific_priority: float
    combined_priority: float
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemResult:
    """システム結果"""
    output_grid: Optional[np.ndarray] = None
    program: Optional[DSLProgram] = None
    confidence: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpecialConditionResult:
    """特殊条件結果"""
    detected: bool = False
    condition_type: str = ""
    confidence: float = 0.0
    result: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProgramSynthesisInput:
    """プログラム合成への入力情報"""
    # A. 今タスク固有のクラスとその優先度
    task_clusters: List[ObjectCluster] = field(default_factory=list)
    
    # B. 各クラスに属しているオブジェクト情報（画像情報付き）
    objects_by_image: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # C. input_grids: List[np.ndarray]
    input_grids: List[np.ndarray] = field(default_factory=list)
    
    # D. output_grids: List[np.ndarray]
    output_grids: List[np.ndarray] = field(default_factory=list)
    
    # E. test_input_grids: List[np.ndarray]
    test_input_grids: List[np.ndarray] = field(default_factory=list)
    
    # F. 追加の詳細情報
    cluster_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    image_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
