"""
grid_batch_*.json から ARC ペアサンプルを読み出すユーティリティ

主な用途:
- データ生成器が出力した JSON (inputs/outputs + trace_results) から
  (task_id, pair_index, input_grid, output_grid, trace_summary) を取り出す
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np


def iter_arc_pair_samples(output_root: str) -> Generator[Dict[str, Any], None, None]:
    """
    outputs/... 配下の grid_batch_*.json から (タスク, ペア) 単位のサンプルを順に返す

    Returns の各要素は以下の形式:
        {
            "task_id": str,
            "pair_index": int,
            "input_grid": np.ndarray,
            "output_grid": np.ndarray,
            "trace_summary": Dict[str, Any] | None,
        }
    """
    root = Path(output_root)
    if not root.exists():
        return

    # batch_*/grid_batch_*.json をすべて走査
    for batch_dir in sorted(root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        for json_path in sorted(batch_dir.glob("grid_batch_*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # data は { "task_001": {...}, "task_002": {...}, ... } の形式を想定
            for task_id, task_data in data.items():
                train_pairs: List[Dict[str, Any]] = task_data.get("train", [])
                for pair_index, pair_dict in enumerate(train_pairs):
                    input_grid = np.array(pair_dict.get("input", []), dtype=int)
                    output_grid = np.array(pair_dict.get("output", []), dtype=int)

                    trace_results = pair_dict.get("trace_results")
                    trace_summary = None
                    if isinstance(trace_results, list) and trace_results:
                        # 現時点では最初の execution_summary を 1 件だけ採用
                        trace_summary = trace_results[0]

                    yield {
                        "task_id": task_id,
                        "pair_index": pair_index,
                        "input_grid": input_grid,
                        "output_grid": output_grid,
                        "trace_summary": trace_summary,
                    }


def extract_basic_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    1つのサンプル辞書から、単純な数値特徴を抽出するヘルパー

    現在の主な用途:
        - トレース＋オブジェクト要約を使った小規模タスク（例: 出力サイズ予測）の入力特徴生成

    戻り値（例）:
        {
            "input_height": int,
            "input_width": int,
            "output_height": int,
            "output_width": int,
            "object_count_before": int,
            "object_count_after": int,
            "rendered_object_count": int,
        }
    """
    input_grid: np.ndarray = sample.get("input_grid")
    output_grid: np.ndarray = sample.get("output_grid")
    trace_summary: Optional[Dict[str, Any]] = sample.get("trace_summary") or {}

    # グリッドサイズ
    in_h, in_w = (0, 0)
    out_h, out_w = (0, 0)
    if isinstance(input_grid, np.ndarray) and input_grid.size > 0:
        in_h, in_w = input_grid.shape[:2]
    if isinstance(output_grid, np.ndarray) and output_grid.size > 0:
        out_h, out_w = output_grid.shape[:2]

    # トレースからのカウント情報
    obj_before = int(trace_summary.get("object_count_before", 0) or 0)
    obj_after = int(trace_summary.get("object_count_after", 0) or 0)
    rendered_count = int(trace_summary.get("rendered_object_count", 0) or 0)

    # オブジェクト要約から簡単な統計量を抽出（平均/最大の bbox 幅・高さ）
    avg_bbox_w = 0.0
    avg_bbox_h = 0.0
    max_bbox_w = 0.0
    max_bbox_h = 0.0
    objects = trace_summary.get("objects") or []
    if isinstance(objects, list) and objects:
        widths: List[float] = []
        heights: List[float] = []
        for obj in objects:
            try:
                w = obj.get("width")
                h = obj.get("height")
                if w is None or h is None:
                    continue
                w = float(w)
                h = float(h)
                widths.append(w)
                heights.append(h)
            except Exception:
                continue
        if widths and heights:
            avg_bbox_w = float(sum(widths) / len(widths))
            avg_bbox_h = float(sum(heights) / len(heights))
            max_bbox_w = float(max(widths))
            max_bbox_h = float(max(heights))

    return {
        "input_height": in_h,
        "input_width": in_w,
        "output_height": out_h,
        "output_width": out_w,
        "object_count_before": obj_before,
        "object_count_after": obj_after,
        "rendered_object_count": rendered_count,
        "avg_bbox_width": avg_bbox_w,
        "avg_bbox_height": avg_bbox_h,
        "max_bbox_width": max_bbox_w,
        "max_bbox_height": max_bbox_h,
    }
