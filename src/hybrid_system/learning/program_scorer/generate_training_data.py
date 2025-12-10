"""
ProgramScorer 用訓練データ生成ユーティリティ

outputs/... 配下の grid_batch_*.json（拡張データセット）を読み、
各 (input, output) ペアに対して CandidateGenerator で複数の候補プログラムを生成し、

    - ペアレベルの特徴（グリッドサイズ・トレース由来のオブジェクト統計）
    - プログラムレベルの特徴（複雑さスコアなど）
    - ラベル（「一貫性 - 複雑さペナルティ」を 0〜1 に収めたスコア）

を JSON Lines 形式で保存する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.hybrid_system.learning.json_pair_loader import (
    iter_arc_pair_samples,
    extract_basic_features,
)
from src.hybrid_system.inference.program_synthesis.candidate_generator import (
    CandidateGenerator,
    CandidateConfig,
)
from src.hybrid_system.inference.program_synthesis.complexity_regularizer import (
    ComplexityRegularizer,
    ComplexityConfig,
)
from src.core_systems.executor.core import ExecutorCore


def _execute_program_on_grid(program: str, input_grid: np.ndarray) -> Optional[np.ndarray]:
    """ExecutorCore を使ってプログラムを 1 つのグリッドに対して実行"""
    if not program or input_grid is None or input_grid.size == 0:
        return None

    try:
        executor = ExecutorCore()
        output = executor.execute_program(program, input_grid)
        if output is None:
            return None
        if isinstance(output, np.ndarray):
            return output
        return np.array(output, dtype=int)
    except Exception:
        return None


def _calculate_consistency_single_pair(expected: np.ndarray, actual: Optional[np.ndarray]) -> float:
    """1 ペアに対する一貫性スコア（ピクセル一致率）を計算（本格実装）"""
    if expected is None or actual is None:
        return 0.0
    if expected.size == 0 or actual.size == 0:
        return 0.0
    if expected.shape != actual.shape:
        return 0.0

    total_pixels = expected.size
    matching = (expected == actual).sum()
    return float(matching) / float(total_pixels) if total_pixels > 0 else 0.0


def _build_feature_vector(
    pair_features: Dict[str, Any],
    program_features: Dict[str, Any],
) -> List[float]:
    """
    ペア特徴 + プログラム特徴から単純なベクトルを構築

    今後特徴が増えても、「順序はここで集中管理する」方針。
    """
    return [
        # ペア側（json_pair_loader.extract_basic_features と揃える）
        float(pair_features.get("input_height", 0)),
        float(pair_features.get("input_width", 0)),
        float(pair_features.get("output_height", 0)),
        float(pair_features.get("output_width", 0)),
        float(pair_features.get("object_count_before", 0)),
        float(pair_features.get("object_count_after", 0)),
        float(pair_features.get("rendered_object_count", 0)),
        float(pair_features.get("avg_bbox_width", 0.0)),
        float(pair_features.get("avg_bbox_height", 0.0)),
        float(pair_features.get("max_bbox_width", 0.0)),
        float(pair_features.get("max_bbox_height", 0.0)),
        # プログラム側
        float(program_features.get("complexity_score", 0.0)),
        float(program_features.get("line_count", 0.0)),
        float(program_features.get("char_count", 0.0)),
    ]


def generate_program_scorer_training_data(
    output_root: str,
    out_path: str,
    max_pairs: int = 500,
    max_candidates_per_pair: int = 10,
    complexity_weight_for_label: float = 0.3,
) -> None:
    """
    ProgramScorer 用の訓練データを生成し、JSON Lines 形式で保存する。

    1 行 = 1 つの候補プログラムに対応し、形式は:
        {
            "features": [...],
            "label": float,
            "metadata": {
                "task_id": ...,
                "pair_index": ...,
                "program": ...,
                "consistency": ...,
                "complexity_score": ...,
            }
        }
    """
    output_root_path = Path(output_root)
    if not output_root_path.exists():
        raise FileNotFoundError(f"output_root が存在しません: {output_root}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 候補生成器と複雑度正則化器を準備
    candidate_gen = CandidateGenerator(CandidateConfig(max_candidates=max_candidates_per_pair))
    complexity_reg = ComplexityRegularizer(ComplexityConfig())

    num_pairs_processed = 0
    num_samples_written = 0

    with out_file.open("w", encoding="utf-8") as f:
        for sample in iter_arc_pair_samples(str(output_root_path)):
            if num_pairs_processed >= max_pairs:
                break

            task_id = sample.get("task_id")
            pair_index = int(sample.get("pair_index", 0))
            input_grid = sample.get("input_grid")
            output_grid = sample.get("output_grid")

            if not isinstance(input_grid, np.ndarray) or not isinstance(output_grid, np.ndarray):
                continue
            if input_grid.size == 0 or output_grid.size == 0:
                continue

            pair_feats = extract_basic_features(sample)

            # 候補プログラムを生成
            candidates = candidate_gen.generate_candidates(
                input_grid=input_grid.tolist(),
                output_grid=output_grid.tolist(),
                max_candidates=max_candidates_per_pair,
                pair_index=pair_index,
            )
            if not candidates:
                num_pairs_processed += 1
                continue

            for program in candidates:
                if not program:
                    continue

                # プログラム実行 → 一貫性スコア
                actual_output = _execute_program_on_grid(program, input_grid)
                consistency = _calculate_consistency_single_pair(output_grid, actual_output)

                # 複雑さスコア
                complexity_score = complexity_reg.calculate_complexity_score(program)

                # ラベル: consistency - α * complexity を 0〜1 にクリップ
                raw_label = consistency - complexity_weight_for_label * complexity_score
                label = float(max(min(raw_label, 1.0), 0.0))

                # プログラム側の簡単な特徴
                line_count = len([ln for ln in program.split("\n") if ln.strip()])
                char_count = len(program)
                program_feats = {
                    "complexity_score": complexity_score,
                    "line_count": float(line_count),
                    "char_count": float(char_count),
                }

                feature_vec = _build_feature_vector(pair_feats, program_feats)

                record = {
                    "features": feature_vec,
                    "label": label,
                    "metadata": {
                        "task_id": task_id,
                        "pair_index": pair_index,
                        "program": program,
                        "consistency": consistency,
                        "complexity_score": complexity_score,
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_samples_written += 1

            num_pairs_processed += 1

    print(f"ProgramScorer 訓練データ生成完了: pairs={num_pairs_processed}, samples={num_samples_written}, path={out_file}")
