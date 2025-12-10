"""
変数インデックス正規化と未使用変数削除の検証スクリプト（10タスク）

データ生成パイプラインを実行して、各タスクのプログラムに対して検証を行う
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 検証モードを有効化
os.environ['ENABLE_VARIABLE_NORMALIZATION_VALIDATION'] = 'true'

import sys
from scripts.production.data_generation.generate_data import main as generate_data_main


def main():
    print("=" * 80)
    print("変数インデックス正規化と未使用変数削除の検証（10タスク）")
    print("=" * 80)
    print()
    print("検証モードが有効化されています。")
    print("データ生成パイプラインが各プログラムに対して検証を実行します。")
    print()

    # コマンドライン引数を設定（--num-programs 10）
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], '--num-programs', '10']

    try:
        # 10タスクを生成（検証モードが有効化されているため、各プログラムに対して検証が実行される）
        generate_data_main()
    finally:
        # 元のコマンドライン引数に戻す
        sys.argv = original_argv


if __name__ == "__main__":
    main()


