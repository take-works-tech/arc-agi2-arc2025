"""
出力グリッドサイズ予測タスクの簡易トレーニングスクリプト

使い方（例）:
    python -X utf8 scripts/train_output_size_prediction.py outputs/test_multiple_pairs_verification_20251116_111247
"""

import sys
from pathlib import Path

from src.hybrid_system.learning.size_prediction_trainer import train_output_size_regressor


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_output_size_prediction.py <output_root>")
        sys.exit(1)

    output_root = sys.argv[1]
    if not Path(output_root).exists():
        print(f"指定された出力ディレクトリが存在しません: {output_root}")
        sys.exit(1)

    train_output_size_regressor(output_root=output_root)


if __name__ == "__main__":
    main()
