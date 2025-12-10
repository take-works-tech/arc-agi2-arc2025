#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テンプレート列中間表現のデータ構造

Phase1 のハイブリッド学習で利用するテンプレート列(IRSequence)を
データクラスとして定義し、シリアライズ/デシリアライズを支援する。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from src.core_systems.executor.operations.types import OperationType


class ArgumentKind(str, Enum):
    """テンプレート引数の値種別"""

    LITERAL = "literal"  # 具体値
    REFERENCE = "reference"  # オブジェクト/シーケンス参照
    PLACEHOLDER = "placeholder"  # 探索で補完する値
    DERIVED = "derived"  # 他スロットや統計量から導出


@dataclass
class ArgumentValue:
    """テンプレート引数の値"""

    kind: ArgumentKind
    value: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書化"""
        return {
            "kind": self.kind.value,
            "value": self.value,
            "constraints": self.constraints,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArgumentValue":
        """辞書から復元"""
        kind = ArgumentKind(data.get("kind", ArgumentKind.LITERAL.value))
        return cls(
            kind=kind,
            value=data.get("value"),
            constraints=data.get("constraints", {}),
            description=data.get("description", ""),
        )


@dataclass
class ArgumentSlot:
    """操作ごとの引数スロット"""

    name: str
    value: ArgumentValue
    value_type: str
    optional: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書化"""
        return {
            "name": self.name,
            "value": self.value.to_dict(),
            "value_type": self.value_type,
            "optional": self.optional,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArgumentSlot":
        """辞書から復元"""
        return cls(
            name=data["name"],
            value=ArgumentValue.from_dict(data["value"]),
            value_type=data.get("value_type", "any"),
            optional=data.get("optional", False),
            notes=data.get("notes", ""),
        )


@dataclass
class PostCondition:
    """各ステップ終了後の検証条件"""

    predicate: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書化"""
        return {
            "predicate": self.predicate,
            "parameters": self.parameters,
            "severity": self.severity,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostCondition":
        """辞書から復元"""
        return cls(
            predicate=data["predicate"],
            parameters=data.get("parameters", {}),
            severity=data.get("severity", "warning"),
            description=data.get("description", ""),
        )


@dataclass
class TemplateStep:
    """テンプレート列の単一ステップ"""

    operation: OperationType
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_binding: str = ""
    argument_slots: List[ArgumentSlot] = field(default_factory=list)
    post_conditions: List[PostCondition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書化"""
        return {
            "operation": self.operation.value,
            "step_id": self.step_id,
            "target_binding": self.target_binding,
            "argument_slots": [slot.to_dict() for slot in self.argument_slots],
            "post_conditions": [pc.to_dict() for pc in self.post_conditions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateStep":
        """辞書から復元"""
        return cls(
            operation=OperationType(data["operation"]),
            step_id=data.get("step_id", str(uuid.uuid4())),
            target_binding=data.get("target_binding", ""),
            argument_slots=[
                ArgumentSlot.from_dict(slot) for slot in data.get("argument_slots", [])
            ],
            post_conditions=[
                PostCondition.from_dict(pc) for pc in data.get("post_conditions", [])
            ],
            metadata=data.get("metadata", {}),
        )


@dataclass
class IRSequence:
    """テンプレート列全体"""

    steps: List[TemplateStep]
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    input_bindings: Dict[str, Any] = field(default_factory=dict)
    version: str = "v1"
    quality_label: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書化"""
        return {
            "sequence_id": self.sequence_id,
            "task_id": self.task_id,
            "input_bindings": self.input_bindings,
            "version": self.version,
            "quality_label": self.quality_label,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRSequence":
        """辞書から復元"""
        return cls(
            steps=[TemplateStep.from_dict(step) for step in data.get("steps", [])],
            sequence_id=data.get("sequence_id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            input_bindings=data.get("input_bindings", {}),
            version=data.get("version", "v1"),
            quality_label=data.get("quality_label", "unknown"),
            metadata=data.get("metadata", {}),
        )

    def add_step(self, step: TemplateStep) -> None:
        """ステップを追加"""
        self.steps.append(step)

    def copy_metadata(self) -> Dict[str, Any]:
        """メタデータのシャローコピー"""
        return self.metadata.copy()

    def asdict(self) -> Dict[str, Any]:
        """dataclasses.asdictとの整合を保つラッパー"""
        return asdict(self)
