#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
負の座標のテスト

負の座標が許可されているか、動作するかを確認
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core_systems.executor.core import ExecutorCore
from src.core_systems.executor.parsing.interpreter import Interpreter
from src.data_systems.data_models.core.object import Object

def create_test_object(executor_core, pixels, color=1):
    """テスト用オブジェクトを作成"""
    obj_id = executor_core._generate_unique_object_id("test_obj")
    bbox = (min(p[0] for p in pixels), min(p[1] for p in pixels),
            max(p[0] for p in pixels), max(p[1] for p in pixels))

    obj = Object(
        object_id=obj_id,
        pixels=pixels,
        pixel_colors={(x, y): color for x, y in pixels},
        bbox=bbox,
        object_type=1
    )

    executor_core.execution_context['objects'][obj_id] = obj
    return obj_id

def test_negative_teleport():
    """負の座標でのテレポートをテスト"""
    print("Testing TELEPORT with negative coordinates...")

    executor_core = ExecutorCore()
    interpreter = Interpreter(executor_core)

    # オブジェクトを作成 (0, 0) に配置
    pixels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    obj_id = create_test_object(executor_core, pixels)

    # 負の座標にテレポート
    result = interpreter.execute_command("TELEPORT", [obj_id, -5, -3])

    if result:
        obj = executor_core._find_object_by_id(result)
        if obj:
            print(f"  [OK] TELEPORT成功: オブジェクトID={result}")
            print(f"  [INFO] ピクセル位置: {obj.pixels[:3]}... (合計{len(obj.pixels)}個)")
            print(f"  [INFO] BBox: {obj.bbox}")

            # 負の座標のピクセルが存在するか確認
            negative_pixels = [(x, y) for x, y in obj.pixels if x < 0 or y < 0]
            if negative_pixels:
                print(f"  [INFO] 負の座標のピクセル: {len(negative_pixels)}個")
                return True
            else:
                print(f"  [WARN] 負の座標のピクセルが見つかりません")
                return False
        else:
            print(f"  [FAIL] オブジェクトが見つかりません")
            return False
    else:
        print(f"  [FAIL] TELEPORTが失敗しました")
        return False

def test_render_with_negative_coordinates():
    """負の座標を持つオブジェクトのレンダリングをテスト"""
    print("\nTesting RENDER_GRID with negative coordinates...")

    executor_core = ExecutorCore()
    interpreter = Interpreter(executor_core)

    # 負の座標を持つオブジェクトを作成
    pixels = [(-2, -1), (-1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0)]
    obj_id = create_test_object(executor_core, pixels)

    # グリッドをレンダリング
    try:
        result = interpreter.execute_command("RENDER_GRID", [[obj_id], 0, 10, 10])
        output_grid = executor_core.execution_context.get('output_grid')

        if output_grid is not None:
            print(f"  [OK] RENDER_GRID成功: グリッドサイズ={output_grid.shape}")

            # グリッド内に描画されたピクセル数をカウント
            non_zero_count = (output_grid != 0).sum()
            print(f"  [INFO] 非ゼロピクセル数: {non_zero_count}")

            # 負の座標のピクセルがスキップされているか確認
            if non_zero_count < len(pixels):
                print(f"  [INFO] 負の座標のピクセルはスキップされています（期待通り）")
                return True
            else:
                print(f"  [WARN] 全てのピクセルが描画されている（負の座標も含む）")
                return False
        else:
            print(f"  [FAIL] 出力グリッドが生成されませんでした")
            return False
    except Exception as e:
        print(f"  [FAIL] エラー: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("負の座標テスト開始")
    print("=" * 60)

    test1 = test_negative_teleport()
    test2 = test_render_with_negative_coordinates()

    print("=" * 60)
    print(f"テスト結果: TELEPORT={test1}, RENDER={test2}")
    print("=" * 60)

    print("\n結論:")
    print("- TELEPORTは負の座標を受け入れる")
    print("- RENDER_GRIDは負の座標のピクセルをスキップする")
    print("- したがって、ALIGNでmax(0, ...)を使うのは適切")
