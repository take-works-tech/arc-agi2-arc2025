#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テンプレート列に含まれるプレースホルダ引数を探索で補完するコンポーネント。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np

from src.hybrid_system.core.data_structures.data_pair import DataPair
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl
from src.hybrid_system.ir.structures import (
    IRSequence,
    TemplateStep,
    ArgumentSlot,
    ArgumentKind,
)


@dataclass
class ParameterCompletionTrace:
    """単一スロットに対する探索ログ"""

    step_index: int
    slot_name: str
    candidate_count: int
    resolved: bool
    chosen_value: Optional[Any] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ParameterCompletionResult:
    """探索結果のサマリ"""

    attempted_slots: int = 0
    resolved_slots: int = 0
    traces: List[ParameterCompletionTrace] = field(default_factory=list)
    verification_success: bool = False


class ParameterCompletionSearcher:
    """
    プレースホルダや曖昧な引数について候補値を探索し、Executor 検証で採択する。

    現状は以下の単純なヒューリスティクスをサポートする:
      - 色引数は入力・出力グリッドの固有値と背景色を候補とする
      - 整数引数は 0〜max(width, height) の範囲をスキャン（最大 10 候補）
      - 列挙引数で constraints.hint がある場合はそれを利用
    """

    def __init__(self, *, max_int_candidate: int = 10):
        self._executor = None
        self._max_int_candidate = max_int_candidate
        self._allowed_directions = (
            "X",
            "Y",
            "-X",
            "-Y",
            "XY",
            "-XY",
            "X-Y",
            "-X-Y",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete(self, pair: DataPair, sequence: IRSequence) -> ParameterCompletionResult:
        """
        与えられた IRSequence のプレースホルダを探索で補完し、
        Executor で期待出力と一致する候補が見つかれば採択する。
        """
        placeholders = self._collect_placeholder_slots(sequence)
        result = ParameterCompletionResult(attempted_slots=len(placeholders))

        if not placeholders:
            result.verification_success = self._verify_sequence(pair, sequence)
            return result

        for step_index, slot in placeholders:
            trace = ParameterCompletionTrace(
                step_index=step_index,
                slot_name=slot.name,
                candidate_count=0,
                resolved=False,
            )

            candidates = list(self._generate_candidates(slot, pair))
            trace.candidate_count = len(candidates)

            original_value = slot.value
            for candidate in candidates:
                self._assign_candidate(slot, candidate)

                if self._verify_sequence(pair, sequence):
                    trace.resolved = True
                    trace.chosen_value = candidate
                    result.resolved_slots += 1
                    break

            if not trace.resolved:
                # 候補が尽きたので元に戻す
                slot.value = original_value
                trace.notes.append("no_candidate_succeeded")

            result.traces.append(trace)

        result.verification_success = self._verify_sequence(pair, sequence)
        return result

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _collect_placeholder_slots(self, sequence: IRSequence) -> List[Tuple[int, ArgumentSlot]]:
        collected: List[Tuple[int, ArgumentSlot]] = []

        for step_index, step in enumerate(sequence.steps):
            for slot in step.argument_slots:
                if slot.value.kind == ArgumentKind.PLACEHOLDER:
                    collected.append((step_index, slot))
        return collected

    def _generate_candidates(self, slot: ArgumentSlot, pair: DataPair) -> Iterable[Any]:
        value_type = (slot.value_type or "").lower()
        constraints = slot.value.constraints or {}
        hint = constraints.get("hint")

        evaluated = self._evaluate_hint(slot, hint, pair)
        if evaluated:
            for candidate in evaluated:
                yield candidate

        candidate_list = constraints.get("candidates")
        if isinstance(candidate_list, Sequence):
            for candidate in candidate_list:
                yield candidate
            candidate_list = list(candidate_list)
        else:
            candidate_list = None

        # constraints.hint を最優先
        if value_type in {"color", "colors"} or slot.name.lower().endswith("color"):
            yield from self._enumerate_colors(pair)
            return

        if value_type in {"int", "integer", "count"} or slot.name.lower().startswith("count"):
            max_dim = max(len(pair.input), len(pair.input[0]) if pair.input else 0)
            for candidate in range(min(self._max_int_candidate, max_dim + 1)):
                yield candidate
            return

        if value_type in {"enum", "mode"}:
            if candidate_list:
                return
            # 代表的な候補
            for candidate in ("relative", "absolute", "difference"):
                yield candidate
            return

        # フォールバック: 出力グリッドから値を採用
        yield from self._enumerate_generic_literals(pair)

    def _evaluate_hint(
        self,
        slot: ArgumentSlot,
        hint: Optional[str],
        pair: DataPair,
    ) -> List[Any]:
        if not hint:
            return []

        normalized = hint.strip()
        candidates: List[Any] = []

        if normalized in self._allowed_directions and slot.name == "direction":
            candidates.extend(self._allowed_directions)
            return candidates

        if slot.name == "direction" and normalized not in self._allowed_directions:
            candidates.extend(self._allowed_directions)
            return candidates

        if normalized in {"X", "Y"} and slot.name == "thickness":
            return [1]

        if normalized.isdigit():
            return [int(normalized)]

        if normalized == "GET_BACKGROUND_COLOR()":
            bg = self._infer_background_color(pair)
            if bg is not None:
                candidates.append(bg)
            return candidates

        if normalized == "grid_size[0]":
            return [len(pair.input[0]) if pair.input else 0]

        if normalized == "grid_size[1]":
            return [len(pair.input)]

        if slot.name == "length":
            width = len(pair.input[0]) if pair.input else 0
            height = len(pair.input)
            base_candidates = {width, height}
            base_candidates.discard(0)
            candidates.extend(sorted(base_candidates))
            if not candidates:
                candidates.extend(range(1, 4))
            return candidates

        if slot.value_type.lower() == "enum":
            # 汎用プレースホルダとして元ヒントの他、よく使う値を返す
            fallback = ["asc", "desc", "horizontal", "vertical"]
            if normalized not in fallback:
                fallback.insert(0, normalized)
            return fallback

        return candidates

    def _assign_candidate(self, slot: ArgumentSlot, candidate: Any) -> None:
        slot.value.kind = ArgumentKind.LITERAL
        slot.value.value = candidate
        slot.value.description = "parameter_completion"
        slot.value.constraints = {"source": "parameter_completion"}

    def _verify_sequence(self, pair: DataPair, sequence: IRSequence) -> bool:
        from src.core_systems.executor.core import ExecutorCore

        if self._executor is None:
            self._executor = ExecutorCore()

        try:
            input_grid = np.array(pair.input, dtype=np.int64)
            expected = np.array(pair.output, dtype=np.int64)
            program = sequence_to_dsl(sequence)
            produced, _, _ = self._executor.execute_program(program, input_grid)
            produced = np.array(produced, dtype=np.int64)

            return produced.shape == expected.shape and np.array_equal(produced, expected)
        except Exception:
            return False

    def _enumerate_colors(self, pair: DataPair) -> Iterable[int]:
        colors = set()
        for grid in (pair.input, pair.output):
            for row in grid:
                for value in row:
                    colors.add(int(value))

        # 背景色推定: 最頻値を追加 (Executor 内で利用されることが多い)
        flattened = list(itertools.chain.from_iterable(pair.input))
        if flattened:
            background = max(set(flattened), key=flattened.count)
            colors.add(int(background))

        for color in sorted(colors):
            yield color

    def _enumerate_generic_literals(self, pair: DataPair) -> Iterable[int]:
        seen = set()
        for grid in (pair.input, pair.output):
            for row in grid:
                for value in row:
                    if value not in seen:
                        seen.add(value)
                        yield int(value)

    def _infer_background_color(self, pair: DataPair) -> Optional[int]:
        counts: Dict[int, int] = {}
        for row in pair.input:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]
