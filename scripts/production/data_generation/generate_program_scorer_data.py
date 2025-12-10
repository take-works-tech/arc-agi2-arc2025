"""
ProgramScorer 用訓練データ生成スクリプト

使い方（例）:
    python -X utf8 scripts/production/data_generation/generate_program_scorer_data.py \\
        outputs/<output_root> \\
        learning_outputs/program_scorer/train_data_latest.jsonl
"""

import sys
from pathlib import Path

from src.hybrid_system.learning.program_scorer.generate_training_data import (
    generate_program_scorer_training_data,
)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/production/data_generation/generate_program_scorer_data.py <output_root> <out_jsonl>")
        sys.exit(1)

    output_root = sys.argv[1]
    out_path = sys.argv[2]

    if not Path(output_root).exists():
        print(f"指定された出力ディレクトリが存在しません: {output_root}")
        sys.exit(1)

    generate_program_scorer_training_data(
        output_root=output_root,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
