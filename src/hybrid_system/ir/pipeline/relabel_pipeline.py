#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テンプレート再ラベリングパイプライン実装
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter

import numpy as np

from src.hybrid_system.core.data_structures.data_pair import DataPair
from src.hybrid_system.ir.parser import RelabelTransformer, RelabelTransformerConfig
from src.hybrid_system.ir.structures import IRSequence, ArgumentKind
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl
from src.hybrid_system.ir.search import ParameterCompletionSearcher, ParameterCompletionResult


@dataclass
class RelabelPipelineConfig:
    """パイプライン設定"""

    input_path: str
    output_dir: str
    batch_size: int = 32
    overwrite: bool = False
    transformer: Optional[RelabelTransformer] = None
    attach_original_program: bool = True
    verification_mode: str = "none"  # "none" / "replay"
    verification_log_level: str = "minimal"  # "minimal" / "full"
    enable_parameter_completion: bool = False
    parameter_completion_searcher: Optional[ParameterCompletionSearcher] = None
    max_pairs: Optional[int] = None  # 最大処理ペア数
    sample_size: Optional[int] = None  # ランダムサンプリング数
    random_seed: int = 42  # ランダムシード


@dataclass
class RelabelPipelineResult:
    """パイプライン実行結果"""

    total_pairs: int = 0
    success_count: int = 0
    failure_count: int = 0
    placeholder_steps: int = 0
    diagnostics: List[Dict[str, str]] = field(default_factory=list)
    output_path: Optional[str] = None
    verification_counts: Dict[str, int] = field(default_factory=dict)
    completion_attempted_pairs: int = 0
    completion_resolved_slots: int = 0


class RelabelPipeline:
    """拡張データをテンプレート列へ変換するパイプライン"""

    def __init__(self, config: RelabelPipelineConfig):
        self.config = config
        self.transformer = config.transformer or RelabelTransformer(RelabelTransformerConfig())
        self._executor = None
        self._parameter_completion: Optional[ParameterCompletionSearcher] = None

        if config.enable_parameter_completion:
            self._parameter_completion = (
                config.parameter_completion_searcher or ParameterCompletionSearcher()
            )

    def run(self) -> RelabelPipelineResult:
        """パイプライン実行"""
        pairs = list(self._load_data_pairs(self.config.input_path))

        # サンプリング処理
        original_count = len(pairs)
        if self.config.sample_size is not None:
            import random
            random.seed(self.config.random_seed)
            if self.config.sample_size < len(pairs):
                pairs = random.sample(pairs, self.config.sample_size)
                print(f"ランダムサンプリング: {original_count}ペア -> {len(pairs)}ペア (seed={self.config.random_seed})")
        elif self.config.max_pairs is not None:
            if self.config.max_pairs < len(pairs):
                pairs = pairs[:self.config.max_pairs]
                print(f"最大ペア数制限: {original_count}ペア -> {len(pairs)}ペア")

        result = RelabelPipelineResult(total_pairs=len(pairs))

        output_path = self._prepare_output_path()
        with open(output_path, "w", encoding="utf-8") as fout:
            for pair in pairs:
                seq, diag = self._transform_pair(pair)
                if seq is None:
                    result.failure_count += 1
                    if diag:
                        result.diagnostics.append(diag)
                    continue

                result.success_count += 1
                placeholder_after = self._count_placeholder_slots(seq)
                result.placeholder_steps += placeholder_after

                completion_meta = seq.metadata.get("completion")
                if completion_meta:
                    result.completion_attempted_pairs += 1
                    resolved = completion_meta.get("resolved_slots")
                    if isinstance(resolved, int):
                        result.completion_resolved_slots += resolved

                record = seq.to_dict()
                record.setdefault("metadata", {})
                if self.config.attach_original_program:
                    record["metadata"]["original_program"] = pair.program
                    record["metadata"]["pair_id"] = pair.pair_id

                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")

                verification_info = seq.metadata.get("verification")
                if verification_info:
                    status = verification_info.get("status", "unknown")
                    result.verification_counts[status] = result.verification_counts.get(status, 0) + 1

                    if self.config.verification_log_level == "full" or (
                        self.config.verification_log_level == "minimal"
                        and status in {"mismatch", "error"}
                    ):
                        result.diagnostics.append(
                            {
                                "pair_id": pair.pair_id,
                                "verification_status": status,
                                "details": verification_info.get("details"),
                            }
                        )

        result.output_path = output_path
        return result

    def _prepare_output_path(self) -> str:
        """出力ファイルパスを生成"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"relabel_sequence_{timestamp}.jsonl"
        output_path = os.path.join(self.config.output_dir, filename)

        if os.path.exists(output_path) and not self.config.overwrite:
            raise FileExistsError(f"出力ファイルが既に存在します: {output_path}")

        return output_path

    def _load_data_pairs(self, path: str) -> Iterable[DataPair]:
        """JSONL から DataPair を読み込み"""
        opener = gzip.open if path.endswith(".gz") else open
        mode = "rt" if path.endswith(".gz") else "r"

        with opener(path, mode, encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield DataPair.from_dict(data)

    def _transform_pair(self, pair: DataPair) -> Tuple[Optional[IRSequence], Optional[Dict[str, str]]]:
        """単一ペアをテンプレート列へ変換"""
        try:
            metadata = {
                "input_size": pair.get_grid_size(),
                "created_at": pair.created_at,
                "source_metadata": pair.metadata,
            }
            sequence = self.transformer.transform(
                pair.program,
                task_id=pair.pair_id,
                sequence_metadata=metadata,
            )

            diagnostics = sequence.metadata.get("diagnostics", [])
            has_errors = any(d.get("type") == "error" for d in diagnostics)
            if has_errors or not sequence.steps:
                reason = "transform_error" if has_errors else "no_steps_generated"
                return None, {
                    "pair_id": pair.pair_id,
                    "error": reason,
                    "details": diagnostics[:10],
                }

            placeholder_before = self._count_placeholder_slots(sequence)
            completion_result: Optional[ParameterCompletionResult] = None

            if self._parameter_completion is not None and placeholder_before > 0:
                completion_result = self._parameter_completion.complete(pair, sequence)
                sequence.metadata.setdefault("completion", {})
                sequence.metadata["completion"].update(asdict(completion_result))

            if self.config.verification_mode != "none":
                verification = self._verify_sequence(pair, sequence)
                sequence.metadata["verification"] = verification

            return sequence, None
        except Exception as exc:
            return None, {
                "pair_id": pair.pair_id,
                "error": str(exc),
            }

    def _verify_sequence(self, pair: DataPair, sequence: IRSequence) -> Dict[str, str]:
        """
        テンプレート列の検証（本格実装）

        検証手順:
          - プレースホルダ未解決の場合は pending として扱う
          - ExecutorCore で元DSLプログラムを再実行し、入出力一致を確認
          - 将来的には、テンプレート列を直接実行できるエンジンが整備でき次第置き換える
        """
        placeholder_count = self._count_placeholder_slots(sequence)
        verification = {
            "mode": self.config.verification_mode,
            "status": "pending_placeholders" if placeholder_count > 0 else "executed",
            "placeholders": placeholder_count,
        }

        if placeholder_count > 0:
            return verification

        if self._executor is None:
            from src.core_systems.executor.core import ExecutorCore

            self._executor = ExecutorCore()

        try:
            input_grid = np.array(pair.input, dtype=np.int64)
            expected_output = np.array(pair.output, dtype=np.int64)

            program_code = sequence_to_dsl(sequence)
            sequence.metadata["generated_program"] = program_code

            output_grid, _, _ = self._executor.execute_program(program_code, input_grid)
            output_grid = np.array(output_grid, dtype=np.int64)

            verification_metrics: Dict[str, object] = {
                "produced_shape": tuple(int(dim) for dim in output_grid.shape),
                "expected_shape": tuple(int(dim) for dim in expected_output.shape),
            }

            if output_grid.shape == expected_output.shape:
                diff_mask = output_grid != expected_output
                diff_cells = int(diff_mask.sum())
                verification_metrics["diff_cells"] = diff_cells

                if diff_cells > 0:
                    mismatch_indices = np.argwhere(diff_mask)
                    samples = []
                    for y, x in mismatch_indices[:5]:
                        samples.append(
                            {
                                "position": [int(y), int(x)],
                                "expected": int(expected_output[y, x]),
                                "produced": int(output_grid[y, x]),
                            }
                        )
                    verification_metrics["diff_samples"] = samples
            else:
                verification_metrics["diff_cells"] = -1

            verification["metrics"] = verification_metrics

            if output_grid.shape != expected_output.shape or not np.array_equal(output_grid, expected_output):
                details = {
                    "message": "executor_output_differs",
                }

                if output_grid.shape == expected_output.shape:
                    transition_counter = Counter()
                    for (y, x) in np.argwhere(output_grid != expected_output):
                        expected_color = int(expected_output[y, x])
                        produced_color = int(output_grid[y, x])
                        key = f"{expected_color}->{produced_color}"
                        transition_counter[key] += 1

                    top_transitions = transition_counter.most_common(5)
                    if top_transitions:
                        details["color_transitions"] = [
                            {"transition": k, "count": v} for k, v in top_transitions
                        ]

                verification.update(
                    {
                        "status": "mismatch",
                        "details": details,
                    }
                )
            else:
                verification["status"] = "passed"
        except Exception as exc:
            verification.update(
                {
                    "status": "error",
                    "details": str(exc),
                }
            )

        return verification

    def _count_placeholder_slots(self, sequence: IRSequence) -> int:
        """プレースホルダ引数の数を計測"""
        count = 0
        for step in sequence.steps:
            for slot in step.argument_slots:
                if slot.value.kind == ArgumentKind.PLACEHOLDER:
                    count += 1
        return count
