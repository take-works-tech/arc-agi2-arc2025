#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新DSLコマンドのテストスクリプト

実装した12個の新しいDSLコマンドの動作確認を行う
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core_systems.executor.core import ExecutorCore
from src.core_systems.executor.parsing.interpreter import Interpreter
from src.data_systems.data_models.core.object import Object
from src.core_systems.executor.grid import grid_size_context

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

def setup_test_grid(executor_core):
    """テスト用グリッドサイズを設定"""
    executor_core.grid_context.initialize((30, 30))

def test_get_aspect_ratio(executor_core, interpreter):
    """GET_ASPECT_RATIOのテスト"""
    print("Testing GET_ASPECT_RATIO...")

    # 3x2の矩形オブジェクトを作成
    pixels = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    obj_id = create_test_object(executor_core, pixels)

    result = interpreter.execute_command("GET_ASPECT_RATIO", [obj_id])
    expected = 3.0 / 2.0  # width / height

    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    print(f"  [OK] GET_ASPECT_RATIO: {result} (expected: {expected})")

def test_get_density(executor_core, interpreter):
    """GET_DENSITYのテスト"""
    print("Testing GET_DENSITY...")

    # 3x2の矩形オブジェクトを作成（6ピクセル）
    pixels = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    obj_id = create_test_object(executor_core, pixels)

    result = interpreter.execute_command("GET_DENSITY", [obj_id])
    expected = 6.0 / (3 * 2)  # pixels / bbox_area = 1.0

    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    print(f"  [OK] GET_DENSITY: {result} (expected: {expected})")

def test_get_center_x_y(executor_core, interpreter):
    """GET_CENTER_X, GET_CENTER_Yのテスト"""
    print("Testing GET_CENTER_X / GET_CENTER_Y...")

    # 3x2の矩形オブジェクトを作成
    pixels = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    obj_id = create_test_object(executor_core, pixels)

    center_x = interpreter.execute_command("GET_CENTER_X", [obj_id])
    center_y = interpreter.execute_command("GET_CENTER_Y", [obj_id])

    # 中心座標は (1, 0.5) → 整数で (1, 0) または (1, 1) の可能性
    # Object.center_x/y の実装に依存するため、値の範囲でチェック
    assert center_x == 1, f"Expected center_x=1, got {center_x}"
    assert center_y in [0, 1], f"Expected center_y in [0, 1], got {center_y}"
    print(f"  [OK] GET_CENTER_X: {center_x}, GET_CENTER_Y: {center_y}")

def test_get_max_x_y(executor_core, interpreter):
    """GET_MAX_X, GET_MAX_Yのテスト"""
    print("Testing GET_MAX_X / GET_MAX_Y...")

    # 3x2の矩形オブジェクトを作成
    pixels = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    obj_id = create_test_object(executor_core, pixels)

    max_x = interpreter.execute_command("GET_MAX_X", [obj_id])
    max_y = interpreter.execute_command("GET_MAX_Y", [obj_id])

    assert max_x == 2, f"Expected max_x=2, got {max_x}"
    assert max_y == 1, f"Expected max_y=1, got {max_y}"
    print(f"  [OK] GET_MAX_X: {max_x}, GET_MAX_Y: {max_y}")

def test_get_centroid(executor_core, interpreter):
    """GET_CENTROIDのテスト"""
    print("Testing GET_CENTROID...")

    # 中央に配置されたオブジェクト
    pixels = [(5, 5), (6, 5), (5, 6), (6, 6)]
    obj_id = create_test_object(executor_core, pixels)

    result = interpreter.execute_command("GET_CENTROID", [obj_id])

    assert isinstance(result, str), f"Expected string, got {type(result)}"
    print(f"  [OK] GET_CENTROID: {result}")

def test_get_direction(executor_core, interpreter):
    """GET_DIRECTIONのテスト"""
    print("Testing GET_DIRECTION...")

    # 2つのオブジェクトを作成
    obj1_id = create_test_object(executor_core, [(0, 0), (1, 0)])
    obj2_id = create_test_object(executor_core, [(5, 5), (6, 5)])

    result = interpreter.execute_command("GET_DIRECTION", [obj1_id, obj2_id])

    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert result in ["X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y"], f"Invalid direction: {result}"
    print(f"  [OK] GET_DIRECTION: {result}")

def test_reverse(interpreter):
    """REVERSEのテスト"""
    print("Testing REVERSE...")

    test_array = [1, 2, 3, 4, 5]
    result = interpreter.execute_command("REVERSE", [test_array])

    assert result == [5, 4, 3, 2, 1], f"Expected [5, 4, 3, 2, 1], got {result}"
    print(f"  [OK] REVERSE: {result}")

def test_align(executor_core, interpreter):
    """ALIGNのテスト"""
    print("Testing ALIGN...")

    setup_test_grid(executor_core)

    # オブジェクトを作成
    pixels = [(10, 10), (11, 10), (10, 11), (11, 11)]
    obj_id = create_test_object(executor_core, pixels)

    # 左端に整列
    aligned_id = interpreter.execute_command("ALIGN", [obj_id, "left"])

    assert aligned_id is not None, "ALIGN failed to return object ID"
    aligned_obj = executor_core._find_object_by_id(aligned_id)
    assert aligned_obj is not None, "Aligned object not found"

    x = executor_core._get_position_x(aligned_id)
    assert x == 0, f"Expected x=0 after left align, got {x}"
    print(f"  [OK] ALIGN(left): x={x}")

def test_get_nearest(executor_core, interpreter):
    """GET_NEARESTのテスト"""
    print("Testing GET_NEAREST...")

    # 基準オブジェクト
    obj1_id = create_test_object(executor_core, [(5, 5)])

    # 候補オブジェクト（遠い）
    obj2_id = create_test_object(executor_core, [(20, 20)])
    # 候補オブジェクト（近い）
    obj3_id = create_test_object(executor_core, [(6, 6)])

    candidates = [obj2_id, obj3_id]
    result = interpreter.execute_command("GET_NEAREST", [obj1_id, candidates])

    assert result == obj3_id, f"Expected nearest object {obj3_id}, got {result}"
    print(f"  [OK] GET_NEAREST: {result}")

def test_tile(executor_core, interpreter):
    """TILEのテスト"""
    print("Testing TILE...")

    # オブジェクトを作成
    pixels = [(0, 0), (1, 0)]
    obj_id = create_test_object(executor_core, pixels)

    # 2x2のタイルを作成
    result = interpreter.execute_command("TILE", [obj_id, 2, 2])

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 4, f"Expected 4 objects (2x2), got {len(result)}"
    print(f"  [OK] TILE(2, 2): {len(result)} objects created")

def run_all_tests():
    """すべてのテストを実行"""
    print("=" * 60)
    print("新DSLコマンドテスト開始")
    print("=" * 60)

    executor_core = ExecutorCore()
    interpreter = Interpreter(executor_core)

    tests = [
        ("GET_ASPECT_RATIO", lambda: test_get_aspect_ratio(executor_core, interpreter)),
        ("GET_DENSITY", lambda: test_get_density(executor_core, interpreter)),
        ("GET_CENTER_X/Y", lambda: test_get_center_x_y(executor_core, interpreter)),
        ("GET_MAX_X/Y", lambda: test_get_max_x_y(executor_core, interpreter)),
        ("GET_CENTROID", lambda: test_get_centroid(executor_core, interpreter)),
        ("GET_DIRECTION", lambda: test_get_direction(executor_core, interpreter)),
        ("REVERSE", lambda: test_reverse(interpreter)),
        ("ALIGN", lambda: test_align(executor_core, interpreter)),
        ("GET_NEAREST", lambda: test_get_nearest(executor_core, interpreter)),
        ("TILE", lambda: test_tile(executor_core, interpreter)),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"テスト結果: {passed}個成功, {failed}個失敗")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
