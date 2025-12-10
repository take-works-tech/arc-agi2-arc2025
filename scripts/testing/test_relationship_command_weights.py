"""関係性生成コマンドの重み調整をテスト"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.config import get_config

def main():
    config = get_config()

    print("=" * 60)
    print("関係性生成コマンドの重み調整テスト")
    print("=" * 60)
    print()

    print(f"有効化: {config.enable_relationship_command_weights}")
    print()

    if config.enable_relationship_command_weights:
        print("関係性生成コマンドの重み:")
        for cmd, weight in sorted(config.relationship_command_weights.items()):
            print(f"  {cmd:20s}: {weight:.2f}x")
        print()
        print(f"合計: {len(config.relationship_command_weights)}個のコマンド")
    else:
        print("関係性生成コマンドの重み調整は無効です")

    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
