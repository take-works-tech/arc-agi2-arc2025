"""新コマンドの登録確認スクリプト"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_generator.metadata.commands import COMMAND_METADATA
from src.data_systems.generator.program_generator.metadata.argument_schema import COMMAND_ARGUMENTS
from src.data_systems.generator.program_generator.metadata.constants import COMMAND_CATEGORIES

new_commands = [
    'GET_ASPECT_RATIO',
    'GET_DENSITY',
    'GET_CENTER_X',
    'GET_CENTER_Y',
    'GET_MAX_X',
    'GET_MAX_Y',
    'GET_CENTROID',
    'GET_DIRECTION',
    'REVERSE',
    'ALIGN',
    'GET_NEAREST',
    'TILE'
]

print('新コマンドの登録確認:')
print('=' * 50)

for cmd in new_commands:
    in_metadata = cmd in COMMAND_METADATA
    in_arguments = cmd in COMMAND_ARGUMENTS
    in_categories = any(cmd in cat_list for cat_list in COMMAND_CATEGORIES.values())

    status = 'OK' if (in_metadata and in_arguments and in_categories) else 'NG'
    print(f'{cmd}: {status}')
    print(f'  COMMAND_METADATA: {"[OK]" if in_metadata else "[NG]"}')
    print(f'  COMMAND_ARGUMENTS: {"[OK]" if in_arguments else "[NG]"}')
    print(f'  COMMAND_CATEGORIES: {"[OK]" if in_categories else "[NG]"}')
    print()

print(f'総コマンド数: {len(COMMAND_METADATA)}')
print(f'総引数定義数: {len(COMMAND_ARGUMENTS)}')
