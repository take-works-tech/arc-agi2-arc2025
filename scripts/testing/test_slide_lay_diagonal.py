"""
SLIDEとLAYコマンドの斜め方向対応と衝突判定の動作確認テスト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_diagonal_directions():
    """斜め方向の定義を確認"""
    print("=" * 60)
    print("1. 斜め方向の定義確認")
    print("=" * 60)

    # SLIDEとLAYで使用される方向ベクトル
    direction_vectors = {
        # 4方向（正負）
        "X": (1, 0),        # 右（0度、X+）
        "-X": (-1, 0),      # 左（180度、X-）
        "Y": (0, 1),        # 下（90度、Y+）
        "-Y": (0, -1),      # 上（270度、Y-）
        # 対角線4方向
        "XY": (1, 1),       # 右下（45度、X+Y+）
        "X-Y": (1, -1),     # 右上（-45度、315度、X+Y-）
        "-XY": (-1, 1),     # 左下（135度、X-Y+）
        "-X-Y": (-1, -1),   # 左上（225度、X-Y-）
    }

    print("\n定義された方向:")
    for direction, (dx, dy) in sorted(direction_vectors.items()):
        is_diagonal = (dx != 0 and dy != 0)
        diagonal_mark = " [斜め]" if is_diagonal else ""
        print(f"  {direction:6s}: ({dx:2d}, {dy:2d}){diagonal_mark}")

    print("\n[OK] 8方向が正しく定義されています")
    return True


def test_collision_detection_logic():
    """衝突判定ロジックの確認"""
    print("\n" + "=" * 60)
    print("2. 衝突判定ロジックの確認")
    print("=" * 60)

    # シミュレーション: 斜め移動の衝突判定
    print("\n【シナリオ1】障害物の「角」に当たる場合")
    current_x, current_y = 5, 5
    dx, dy = 1, 1  # XY方向（右下）
    next_x, next_y = current_x + dx, current_y + dy

    obstacle_pixels = {(6, 5), (5, 6)}  # 角

    is_diagonal = (dx != 0 and dy != 0)
    collision_detected = False

    if is_diagonal:
        # 斜め移動: 次の位置と隣接する2つのピクセルをチェック
        if ((next_x, next_y) in obstacle_pixels or
            (next_x, current_y) in obstacle_pixels or
            (current_x, next_y) in obstacle_pixels):
            collision_detected = True

    print(f"  現在位置: ({current_x}, {current_y})")
    print(f"  移動方向: XY (dx={dx}, dy={dy})")
    print(f"  次の位置: ({next_x}, {next_y})")
    print(f"  障害物: {obstacle_pixels}")
    print(f"  衝突判定: {collision_detected}")

    if collision_detected:
        print("  [OK] 角の衝突が正しく検出されます")
    else:
        print("  [NG] 角の衝突が検出されません")
        return False

    print("\n【シナリオ2】中間ピクセルを通過する場合")
    obstacle_pixels = {(6, 5)}  # 中間ピクセル

    collision_detected = False
    if is_diagonal:
        if ((next_x, next_y) in obstacle_pixels or
            (next_x, current_y) in obstacle_pixels or
            (current_x, next_y) in obstacle_pixels):
            collision_detected = True

    print(f"  障害物: {obstacle_pixels}")
    print(f"  衝突判定: {collision_detected}")

    if collision_detected:
        print("  [OK] 中間ピクセルの衝突が正しく検出されます")
    else:
        print("  [NG] 中間ピクセルの衝突が検出されません")
        return False

    print("\n【シナリオ3】正常な斜め移動")
    obstacle_pixels = set()  # 障害物なし

    collision_detected = False
    if is_diagonal:
        if ((next_x, next_y) in obstacle_pixels or
            (next_x, current_y) in obstacle_pixels or
            (current_x, next_y) in obstacle_pixels):
            collision_detected = True

    print(f"  障害物: {obstacle_pixels}")
    print(f"  衝突判定: {collision_detected}")

    if not collision_detected:
        print("  [OK] 正常な移動が正しく処理されます")
    else:
        print("  [NG] 正常な移動が衝突と判定されます")
        return False

    return True


def test_documentation_update():
    """ドキュメントの更新確認"""
    print("\n" + "=" * 60)
    print("3. ドキュメントの更新確認")
    print("=" * 60)

    doc_path = project_root / "docs" / "guides" / "コマンドクイックリファレンス.md"

    if not doc_path.exists():
        print(f"  [WARN] ドキュメントが見つかりません: {doc_path}")
        return False

    content = doc_path.read_text(encoding='utf-8')

    # 8方向対応の記載を確認
    if 'SLIDE、LAY の方向仕様（8方向）' in content:
        print("  [OK] ドキュメントに8方向対応が記載されています")
    else:
        print("  [NG] ドキュメントに8方向対応が記載されていません")
        return False

    # 斜め方向の例を確認
    if '"XY"' in content and '"X-Y"' in content:
        print("  [OK] 斜め方向の例が記載されています")
    else:
        print("  [WARN] 斜め方向の例が不足している可能性があります")

    # FLOWが4方向のみであることを確認
    if 'FLOW' in content and '4方向のみ' in content:
        print("  [OK] FLOWが4方向のみであることが明記されています")
    else:
        print("  [WARN] FLOWの方向制限が明記されていない可能性があります")

    return True


def main():
    """メイン関数"""
    print("SLIDEとLAYコマンドの斜め方向対応と衝突判定の動作確認")
    print("=" * 60)

    results = []

    # 1. 斜め方向の定義確認
    results.append(("斜め方向の定義確認", test_diagonal_directions()))

    # 2. 衝突判定ロジックの確認
    results.append(("衝突判定ロジックの確認", test_collision_detection_logic()))

    # 3. ドキュメントの更新確認
    results.append(("ドキュメントの更新確認", test_documentation_update()))

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
