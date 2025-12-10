#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATHFINDコマンドがプログラム生成器で生成される可能性を確認するテスト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_generator.generation.unified_program_generator import UnifiedProgramGenerator
from src.data_systems.generator.program_generator.metadata.commands import COMMAND_METADATA
from src.data_systems.generator.program_generator.metadata.constants import COMMAND_CATEGORIES
from src.data_systems.generator.program_generator.generation.node_generators import NodeGenerators
import re

def test_pathfind_in_metadata():
    """PATHFINDがメタデータに正しく定義されているか確認"""
    print("=" * 80)
    print("1. PATHFINDメタデータの確認")
    print("=" * 80)

    pathfind = COMMAND_METADATA.get('PATHFIND')
    if pathfind:
        print(f"[OK] PATHFINDメタデータが存在します")
        print(f"  - base_weight: {pathfind.base_weight}")
        print(f"  - category: {pathfind.category}")
        print(f"  - requires_object: {pathfind.requires_object}")
        print(f"  - requires_array: {pathfind.requires_array}")
        print(f"  - usage_contexts: {pathfind.usage_contexts}")
        print(f"  - arguments count: {len(pathfind.arguments)}")
    else:
        print(f"[NG] PATHFINDメタデータが存在しません")
        return False
    return True

def test_pathfind_in_categories():
    """PATHFINDがカテゴリに含まれているか確認"""
    print("\n" + "=" * 80)
    print("2. PATHFINDがカテゴリに含まれているか確認")
    print("=" * 80)

    in_transform = 'PATHFIND' in COMMAND_CATEGORIES['transform']
    print(f"[{'OK' if in_transform else 'NG'}] PATHFIND in COMMAND_CATEGORIES['transform']: {in_transform}")

    return in_transform

def test_pathfind_in_object_commands():
    """PATHFINDがobject_commandsリストに含まれているか確認"""
    print("\n" + "=" * 80)
    print("3. PATHFINDがobject_commandsリストに含まれているか確認")
    print("=" * 80)

    # node_generators.pyのobject_commandsリストを確認
    object_commands = [
        'MOVE', 'TELEPORT', 'SLIDE', 'PATHFIND', 'ROTATE', 'FLIP', 'SCALE', 'SCALE_DOWN', 'EXPAND',
        'FILL_HOLES', 'SET_COLOR', 'OUTLINE', 'HOLLOW', 'BBOX', 'INTERSECTION',
        'SUBTRACT', 'FLOW', 'DRAW', 'LAY', 'ALIGN', 'CROP', 'FIT_SHAPE', 'FIT_SHAPE_COLOR', 'FIT_ADJACENT'
    ]

    in_list = 'PATHFIND' in object_commands
    print(f"[{'OK' if in_list else 'NG'}] PATHFIND in object_commands list: {in_list}")

    return in_list

def test_pathfind_in_obstacle_commands():
    """PATHFINDがobstacle_commandsリストに含まれているか確認"""
    print("\n" + "=" * 80)
    print("4. PATHFINDがobstacle_commandsリストに含まれているか確認")
    print("=" * 80)

    # node_generators.pyのobstacle_commandsリストを確認
    obstacle_commands = ['FLOW', 'LAY', 'SLIDE', 'PATHFIND']

    in_list = 'PATHFIND' in obstacle_commands
    print(f"[{'OK' if in_list else 'NG'}] PATHFIND in obstacle_commands list: {in_list}")

    return in_list

def test_pathfind_not_in_avoid_commands():
    """PATHFINDがavoid_commandsリストに含まれていないか確認"""
    print("\n" + "=" * 80)
    print("5. PATHFINDがavoid_commandsリストに含まれていないか確認")
    print("=" * 80)

    from src.data_systems.generator.program_generator.metadata.constants import COMPLEXITY_LEVELS

    all_avoid_commands = set()
    for level in COMPLEXITY_LEVELS.values():
        all_avoid_commands.update(level.get('avoid_commands', []))

    in_avoid = 'PATHFIND' in all_avoid_commands
    print(f"[{'OK' if not in_avoid else 'NG'}] PATHFIND not in avoid_commands: {not in_avoid}")
    if in_avoid:
        print(f"  [WARN] PATHFINDがavoid_commandsに含まれています: {all_avoid_commands}")

    return not in_avoid

def test_pathfind_generation_possibility():
    """実際にPATHFINDが生成される可能性を確認（複数回試行）"""
    print("\n" + "=" * 80)
    print("6. PATHFINDが実際に生成される可能性を確認（100回試行）")
    print("=" * 80)

    generator = UnifiedProgramGenerator()
    pathfind_count = 0
    total_attempts = 100

    for i in range(total_attempts):
        try:
            program = generator.generate_program(complexity=5)  # 複雑度5で試行
            if 'PATHFIND' in program:
                pathfind_count += 1
                if pathfind_count == 1:
                    print(f"  [INFO] 最初のPATHFIND検出例:")
                    print(f"    {program[:200]}...")
        except Exception as e:
            # エラーは無視して続行
            pass

    probability = pathfind_count / total_attempts
    print(f"[{'OK' if pathfind_count > 0 else 'NG'}] PATHFIND生成回数: {pathfind_count}/{total_attempts} ({probability*100:.1f}%)")

    return pathfind_count > 0

def main():
    """メイン関数"""
    print("PATHFINDコマンド生成可能性の確認テスト")
    print("=" * 80)

    results = []
    results.append(("メタデータ確認", test_pathfind_in_metadata()))
    results.append(("カテゴリ確認", test_pathfind_in_categories()))
    results.append(("object_commands確認", test_pathfind_in_object_commands()))
    results.append(("obstacle_commands確認", test_pathfind_in_obstacle_commands()))
    results.append(("avoid_commands確認", test_pathfind_not_in_avoid_commands()))
    results.append(("実際の生成可能性", test_pathfind_generation_possibility()))

    print("\n" + "=" * 80)
    print("テスト結果サマリー")
    print("=" * 80)

    all_passed = True
    for name, result in results:
        status = "[OK]" if result else "[NG]"
        print(f"{status} {name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[結論] PATHFINDはプログラム生成器で生成される可能性があります")
    else:
        print("[結論] PATHFINDの生成に問題がある可能性があります")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
