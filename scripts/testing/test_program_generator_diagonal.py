"""
program_generatorのLAY、SLIDEコマンドの斜め方向対応の動作確認テスト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_direction_argument_schema():
    """方向引数スキーマの確認"""
    print("=" * 60)
    print("1. 方向引数スキーマの確認")
    print("=" * 60)

    from src.data_systems.generator.program_generator.metadata.argument_schema import (
        DIRECTION_ARG, DIRECTION_8WAY_ARG, COMMAND_ARGUMENTS
    )

    print("\nDIRECTION_ARG (FLOW用、4方向):")
    print(f"  choices: {DIRECTION_ARG.choices}")
    if len(DIRECTION_ARG.choices) == 4:
        print("  [OK] 4方向が正しく定義されています")
    else:
        print(f"  [NG] 4方向であるべきですが、{len(DIRECTION_ARG.choices)}方向です")
        return False

    print("\nDIRECTION_8WAY_ARG (SLIDE, LAY用、8方向):")
    print(f"  choices: {DIRECTION_8WAY_ARG.choices}")
    if len(DIRECTION_8WAY_ARG.choices) == 8:
        print("  [OK] 8方向が正しく定義されています")
        # 斜め方向が含まれているか確認
        diagonal_directions = ['XY', '-XY', 'X-Y', '-X-Y']
        if all(d in DIRECTION_8WAY_ARG.choices for d in diagonal_directions):
            print("  [OK] 斜め方向が含まれています")
        else:
            print("  [NG] 斜め方向が不足しています")
            return False
    else:
        print(f"  [NG] 8方向であるべきですが、{len(DIRECTION_8WAY_ARG.choices)}方向です")
        return False

    print("\nCOMMAND_ARGUMENTSの確認:")
    slide_args = COMMAND_ARGUMENTS.get('SLIDE', [])
    lay_args = COMMAND_ARGUMENTS.get('LAY', [])
    flow_args = COMMAND_ARGUMENTS.get('FLOW', [])

    # SLIDEの第2引数（方向引数）を確認
    if len(slide_args) >= 2:
        slide_direction_arg = slide_args[1]
        if slide_direction_arg == DIRECTION_8WAY_ARG:
            print("  [OK] SLIDEはDIRECTION_8WAY_ARGを使用しています")
        else:
            print("  [NG] SLIDEはDIRECTION_8WAY_ARGを使用していません")
            return False
    else:
        print("  [NG] SLIDEの引数が不足しています")
        return False

    # LAYの第2引数（方向引数）を確認
    if len(lay_args) >= 2:
        lay_direction_arg = lay_args[1]
        if lay_direction_arg == DIRECTION_8WAY_ARG:
            print("  [OK] LAYはDIRECTION_8WAY_ARGを使用しています")
        else:
            print("  [NG] LAYはDIRECTION_8WAY_ARGを使用していません")
            return False
    else:
        print("  [NG] LAYの引数が不足しています")
        return False

    # FLOWの第2引数（方向引数）を確認（4方向のまま）
    if len(flow_args) >= 2:
        flow_direction_arg = flow_args[1]
        if flow_direction_arg == DIRECTION_ARG:
            print("  [OK] FLOWはDIRECTION_ARG（4方向）を使用しています")
        else:
            print("  [NG] FLOWはDIRECTION_ARGを使用していません")
            return False
    else:
        print("  [NG] FLOWの引数が不足しています")
        return False

    return True


def test_command_metadata():
    """コマンドメタデータの確認"""
    print("\n" + "=" * 60)
    print("2. コマンドメタデータの確認")
    print("=" * 60)

    from src.data_systems.generator.program_generator.metadata.commands import COMMAND_METADATA

    slide_metadata = COMMAND_METADATA.get('SLIDE')
    lay_metadata = COMMAND_METADATA.get('LAY')

    if slide_metadata:
        print(f"\nSLIDE:")
        print(f"  description: {slide_metadata.description}")
        print(f"  examples: {slide_metadata.examples}")

        if '8方向' in slide_metadata.description:
            print("  [OK] 説明に8方向対応が記載されています")
        else:
            print("  [WARN] 説明に8方向対応が記載されていません")

        # 斜め方向の例が含まれているか確認
        has_diagonal_example = any('XY' in ex or 'X-Y' in ex or '-XY' in ex or '-X-Y' in ex
                                  for ex in slide_metadata.examples)
        if has_diagonal_example:
            print("  [OK] 斜め方向の例が含まれています")
        else:
            print("  [WARN] 斜め方向の例が不足しています")
    else:
        print("  [NG] SLIDEのメタデータが見つかりません")
        return False

    if lay_metadata:
        print(f"\nLAY:")
        print(f"  description: {lay_metadata.description}")
        print(f"  examples: {lay_metadata.examples}")

        if '8方向' in lay_metadata.description:
            print("  [OK] 説明に8方向対応が記載されています")
        else:
            print("  [WARN] 説明に8方向対応が記載されていません")

        # 斜め方向の例が含まれているか確認
        has_diagonal_example = any('XY' in ex or 'X-Y' in ex or '-XY' in ex or '-X-Y' in ex
                                  for ex in lay_metadata.examples)
        if has_diagonal_example:
            print("  [OK] 斜め方向の例が含まれています")
        else:
            print("  [WARN] 斜め方向の例が不足しています")
    else:
        print("  [NG] LAYのメタデータが見つかりません")
        return False

    return True


def test_type_system():
    """型システムの確認"""
    print("\n" + "=" * 60)
    print("3. 型システムの確認")
    print("=" * 60)

    from src.data_systems.generator.program_generator.metadata.types import (
        SemanticType, TypeSystem
    )

    direction_choices = TypeSystem.get_string_choices(SemanticType.DIRECTION)

    if direction_choices:
        print(f"\nDIRECTION型の選択肢:")
        print(f"  {direction_choices}")

        if len(direction_choices) == 8:
            print("  [OK] 8方向が定義されています")
            # 斜め方向が含まれているか確認
            diagonal_directions = ['XY', '-XY', 'X-Y', '-X-Y']
            if all(d in direction_choices for d in diagonal_directions):
                print("  [OK] 斜め方向が含まれています")
            else:
                print("  [NG] 斜め方向が不足しています")
                return False
        else:
            print(f"  [NG] 8方向であるべきですが、{len(direction_choices)}方向です")
            return False
    else:
        print("  [NG] DIRECTION型の選択肢が取得できません")
        return False

    return True


def main():
    """メイン関数"""
    print("program_generatorのLAY、SLIDEコマンドの斜め方向対応の動作確認")
    print("=" * 60)

    results = []

    # 1. 方向引数スキーマの確認
    results.append(("方向引数スキーマの確認", test_direction_argument_schema()))

    # 2. コマンドメタデータの確認
    results.append(("コマンドメタデータの確認", test_command_metadata()))

    # 3. 型システムの確認
    results.append(("型システムの確認", test_type_system()))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[NG]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] すべてのテストが成功しました")
        return 0
    else:
        print("\n[NG] 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())
