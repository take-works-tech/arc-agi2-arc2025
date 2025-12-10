"""
ProgramScorer モデル学習スクリプト

使い方（例）:
    python -X utf8 scripts/train_program_scorer.py ^
        learning_outputs/program_scorer/train_data_latest.jsonl ^
        learning_outputs/program_scorer/program_scorer_latest.pt
"""

import sys
from pathlib import Path

from src.hybrid_system.learning.program_scorer.trainer import train_program_scorer


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/train_program_scorer.py <train_jsonl> <model_out_path>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    model_out_path = sys.argv[2]

    if not Path(jsonl_path).exists():
        print(f"指定された訓練データが存在しません: {jsonl_path}")
        sys.exit(1)

    train_program_scorer(
        jsonl_path=jsonl_path,
        model_out_path=model_out_path,
    )


if __name__ == "__main__":
    main()
