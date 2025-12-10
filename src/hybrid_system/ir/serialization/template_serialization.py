"""
テンプレートシーケンスのシリアライズ/デシリアライズ

IRSequence ↔ 文字列トークン列 の相互変換を担う。
"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional
from urllib.parse import quote_plus, unquote_plus

from src.hybrid_system.ir.structures import (
    IRSequence,
    TemplateStep,
    ArgumentSlot,
    ArgumentValue,
    ArgumentKind,
    PostCondition,
)
from src.core_systems.executor.operations.types import OperationType


_SEQ_START = "SEQ|START"
_SEQ_END = "SEQ|END"
_STEP_END = "STEP|END"


def _encode_str(value: Optional[str]) -> str:
    if value is None:
        return ""
    return quote_plus(str(value))


def _decode_str(value: str) -> str:
    return unquote_plus(value)


def sequence_to_template_string(sequence: IRSequence) -> str:
    """
    IRSequence をテンプレートトークン列へ変換し、スペース区切りの文字列として返す。
    """
    tokens: List[str] = [_SEQ_START]

    for step in sequence.steps:
        tokens.append(f"OP|{step.operation.value}")
        if step.target_binding:
            tokens.append(f"TGT|{_encode_str(step.target_binding)}")

        if step.argument_slots:
            for slot in step.argument_slots:
                slot_payload = slot.value.to_dict()
                slot_json = json.dumps(
                    {
                        "name": slot.name,
                        "value": slot_payload,
                        "value_type": slot.value_type,
                        "optional": slot.optional,
                        "notes": slot.notes,
                    },
                    ensure_ascii=False,
                )
                tokens.append(f"ARG|{_encode_str(slot.name)}|{_encode_str(slot_json)}")

        if step.post_conditions:
            post_json = json.dumps(
                [pc.to_dict() for pc in step.post_conditions],
                ensure_ascii=False,
            )
            tokens.append(f"PST|{_encode_str(post_json)}")

        if step.metadata:
            metadata_json = json.dumps(step.metadata, ensure_ascii=False)
            tokens.append(f"MET|{_encode_str(metadata_json)}")

        tokens.append(_STEP_END)

    tokens.append(_SEQ_END)
    return " ".join(tokens)


def template_string_to_sequence(token_string: str, *,
                                sequence_id: Optional[str] = None,
                                task_id: str = "") -> IRSequence:
    """
    テンプレートトークン列（sequence_to_template_string で生成した文字列）を IRSequence に復元する。
    """
    raw_tokens = [tok for tok in token_string.strip().split() if tok]
    steps: List[TemplateStep] = []

    current_operation: Optional[OperationType] = None
    current_target: str = ""
    current_args: List[ArgumentSlot] = []
    current_posts: List[PostCondition] = []
    current_metadata: dict = {}

    def _finalize_step():
        nonlocal current_operation, current_target, current_args, current_posts, current_metadata
        if current_operation is None:
            return

        step = TemplateStep(
            operation=current_operation,
            target_binding=current_target,
            argument_slots=current_args,
            post_conditions=current_posts,
            metadata=current_metadata,
        )
        steps.append(step)

        current_operation = None
        current_target = ""
        current_args = []
        current_posts = []
        current_metadata = {}

    for token in raw_tokens:
        if token == _SEQ_START or token == _SEQ_END:
            continue
        if token == _STEP_END:
            _finalize_step()
            continue

        prefix, _, payload = token.partition("|")

        if prefix == "OP":
            _finalize_step()
            try:
                current_operation = OperationType(payload)
            except ValueError:
                try:
                    current_operation = OperationType[payload.upper()]
                except KeyError:
                    raise
        elif prefix == "TGT":
            current_target = _decode_str(payload)
        elif prefix == "ARG":
            parts = token.split("|", 2)
            if len(parts) < 3:
                continue
            _, encoded_name, encoded_json = parts
            slot_json = json.loads(_decode_str(encoded_json))
            slot_value = ArgumentValue.from_dict(slot_json["value"])
            current_args.append(
                ArgumentSlot(
                    name=_decode_str(encoded_name),
                    value=slot_value,
                    value_type=slot_json.get("value_type", "any"),
                    optional=slot_json.get("optional", False),
                    notes=slot_json.get("notes", ""),
                )
            )
        elif prefix == "PST":
            post_conditions = json.loads(_decode_str(payload))
            current_posts = [PostCondition.from_dict(pc) for pc in post_conditions]
        elif prefix == "MET":
            current_metadata = json.loads(_decode_str(payload))

    _finalize_step()

    return IRSequence(
        steps=steps,
        sequence_id=sequence_id or str(uuid.uuid4()),
        task_id=task_id,
    )
