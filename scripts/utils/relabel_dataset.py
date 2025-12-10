#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テンプレート再ラベリング CLI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_system.ir.pipeline import (
    RelabelPipeline,
    RelabelPipelineConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="再ラベリングパイプラインを実行")
    parser.add_argument("--input", required=True, help="DataPair JSONL ファイル")
    parser.add_argument("--output_dir", required=True, help="IRSequence 出力先ディレクトリ")
    parser.add_argument("--overwrite", action="store_true", help="同名ファイルが存在しても上書きする")
    parser.add_argument(
        "--verification",
        default="none",
        choices=["none", "replay"],
        help="検証モード (none: 検証なし, replay: DSL を再実行して検証)",
    )
    parser.add_argument(
        "--verification-log-level",
        default="minimal",
        choices=["minimal", "full"],
        help="検証ログ出力レベル (minimal: 要約のみ, full: 失敗時の詳細ログ)",
    )
    parser.add_argument(
        "--enable-completion",
        action="store_true",
        help="プレースホルダ引数の探索補完を有効化する",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="処理する最大ペア数（最初のN個のみ処理）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="ランダムサンプリングするペア数",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="ランダムサンプリング時のシード値",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    config = RelabelPipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
        verification_mode=args.verification,
        verification_log_level=args.verification_log_level,
        enable_parameter_completion=args.enable_completion,
        max_pairs=args.max_pairs,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )

    pipeline = RelabelPipeline(config)
    result = pipeline.run()

    summary_path = Path(result.output_path).with_suffix(".summary.json")
    summary = {
        "total_pairs": result.total_pairs,
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "placeholder_slots": result.placeholder_steps,
        "output_path": result.output_path,
        "verification_counts": result.verification_counts,
        "verification_mode": args.verification,
        "diagnostics": result.diagnostics[:50],
        "completion_attempted_pairs": result.completion_attempted_pairs,
        "completion_resolved_slots": result.completion_resolved_slots,
        "sampling_info": {
            "max_pairs": args.max_pairs,
            "sample_size": args.sample_size,
            "random_seed": args.random_seed if args.sample_size else None,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
