"""
ARC-AGI2データセットのオブジェクト複製パターン統計を取得

各タスクの入力グリッドからオブジェクトを抽出し、
- 同じ色と形状をコピーする確率
- 同じ形状で色のみ変更する確率
- 新規生成（形状も色も異なる）の確率
を計算します。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import ObjectExtractor
from src.data_systems.generator.input_grid_generator.builders.shape_utils import get_shape_signature, normalize_shape
from src.data_systems.data_models.base import ObjectType


def load_arc_agi2_tasks(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    """ARC-AGI2の訓練データを読み込む"""
    possible_paths = [
        project_root / "data" / "arc-agi_training_challenges.json",
        project_root / "data" / "core_arc_agi2" / "arc-agi_training_challenges.json",
        project_root / "data" / "arc-agi2" / "arc-agi_training_challenges.json",
    ]

    for task_file in possible_paths:
        if task_file.exists():
            print(f"タスクファイルを読み込み: {task_file}")
            with open(task_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)

            tasks = []
            for task_id, task_data in tasks_data.items():
                if max_tasks and len(tasks) >= max_tasks:
                    break
                tasks.append({
                    'task_id': task_id,
                    'train': task_data.get('train', []),
                    'test': task_data.get('test', [])
                })
            return tasks

    print("タスクファイルが見つかりませんでした")
    return []


def extract_objects_from_grid(grid: np.ndarray, extractor: ObjectExtractor) -> List[Dict[str, Any]]:
    """グリッドからオブジェクトを抽出し、色と形状シグネチャを取得"""
    try:
        # オブジェクトを抽出
        result = extractor.extractor.extract_objects_by_type(grid)
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

        # オブジェクトを辞書形式に変換
        object_list = []
        for obj in objects:
            # ピクセル座標を取得
            pixels = obj.pixels

            # オブジェクト辞書を作成
            obj_dict = {
                'pixels': pixels,
                'color': obj.color,
                'width': obj.bbox_width,
                'height': obj.bbox_height,
            }

            # 形状シグネチャを取得
            shape_sig = get_shape_signature(obj_dict)
            obj_dict['shape_signature'] = shape_sig

            object_list.append(obj_dict)

        return object_list
    except Exception as e:
        print(f"  オブジェクト抽出エラー: {e}", flush=True)
        return []


def analyze_object_duplication_patterns(objects: List[Dict[str, Any]]) -> Dict[str, int]:
    """オブジェクト間の複製パターンを分析

    各オブジェクトが「既存のオブジェクトと同じ色と形状」「既存のオブジェクトと同じ形状（色は異なる）」
    「新規生成」のいずれかに分類されるかを判定します。
    最初のオブジェクトは常に「新規生成」として扱います。
    """
    if len(objects) == 0:
        return {
            'same_color_and_shape': 0,
            'same_shape_only': 0,
            'new_generation': 0,
            'total_objects': 0
        }

    same_color_and_shape_count = 0
    same_shape_only_count = 0
    new_generation_count = 1  # 最初のオブジェクトは常に新規生成

    # 最初のオブジェクトはスキップ（常に新規生成として扱う）
    seen_shapes_and_colors = {}  # {(shape_sig, color): count}
    seen_shapes = {}  # {shape_sig: [colors]}

    first_obj = objects[0]
    first_shape_sig = first_obj.get('shape_signature')
    first_color = first_obj.get('color')

    if first_shape_sig is not None:
        seen_shapes_and_colors[(first_shape_sig, first_color)] = 1
        seen_shapes[first_shape_sig] = [first_color]

    # 2番目以降のオブジェクトを分析
    for i in range(1, len(objects)):
        obj = objects[i]
        shape_sig = obj.get('shape_signature')
        color = obj.get('color')

        if shape_sig is None:
            # 形状シグネチャが取得できない場合は新規生成として扱う
            new_generation_count += 1
            continue

        # 既存のオブジェクトと同じ色と形状かチェック
        if (shape_sig, color) in seen_shapes_and_colors:
            # 同じ色と形状をコピー
            same_color_and_shape_count += 1
            seen_shapes_and_colors[(shape_sig, color)] += 1
        # 既存のオブジェクトと同じ形状（色は異なる）かチェック
        elif shape_sig in seen_shapes:
            # 同じ形状で色のみ変更
            same_shape_only_count += 1
            if color not in seen_shapes[shape_sig]:
                seen_shapes[shape_sig].append(color)
            seen_shapes_and_colors[(shape_sig, color)] = seen_shapes_and_colors.get((shape_sig, color), 0) + 1
        else:
            # 新規生成（形状も色も異なる）
            new_generation_count += 1
            seen_shapes[shape_sig] = [color]
            seen_shapes_and_colors[(shape_sig, color)] = 1

    total_objects = len(objects)

    return {
        'same_color_and_shape': same_color_and_shape_count,
        'same_shape_only': same_shape_only_count,
        'new_generation': new_generation_count,
        'total_objects': total_objects
    }


def analyze_all_grids(tasks: List[Dict[str, Any]], max_tasks: Optional[int] = None) -> Dict[str, Any]:
    """すべてのタスクの入力グリッドを分析"""
    extractor = ObjectExtractor()

    total_patterns = {
        'same_color_and_shape': 0,
        'same_shape_only': 0,
        'new_generation': 0,
        'total_objects': 0
    }

    task_count = 0
    grid_count = 0
    object_count = 0
    skipped_tasks = 0
    skipped_grids = 0

    print(f"\n分析を開始: {len(tasks)}タスク", flush=True)

    for task_idx, task in enumerate(tasks):
        if max_tasks and task_idx >= max_tasks:
            break

        task_id = task.get('task_id', f'task_{task_idx}')
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])

        # すべての入力グリッドを分析（訓練+テスト）
        all_inputs = []
        for ex in train_examples:
            if 'input' in ex:
                all_inputs.append(ex['input'])
        for ex in test_examples:
            if 'input' in ex:
                all_inputs.append(ex['input'])

        if not all_inputs:
            skipped_tasks += 1
            continue

        task_count += 1
        task_patterns = {
            'same_color_and_shape': 0,
            'same_shape_only': 0,
            'new_generation': 0,
            'total_objects': 0
        }

        for grid_idx, input_grid in enumerate(all_inputs):
            try:
                grid_np = np.array(input_grid, dtype=np.int32)

                # オブジェクトを抽出
                objects = extract_objects_from_grid(grid_np, extractor)

                if len(objects) < 2:
                    skipped_grids += 1
                    continue

                grid_count += 1
                object_count += len(objects)

                # 複製パターンを分析
                patterns = analyze_object_duplication_patterns(objects)

                # タスク全体の統計に追加
                for key in task_patterns:
                    task_patterns[key] += patterns[key]

            except Exception as e:
                print(f"  タスク {task_id}, グリッド {grid_idx}: エラー - {e}", flush=True)
                skipped_grids += 1
                continue

        # 全体の統計に追加
        for key in total_patterns:
            total_patterns[key] += task_patterns[key]

        if (task_idx + 1) % 100 == 0:
            print(f"  処理済み: {task_idx + 1}/{len(tasks)}タスク", flush=True)

    print(f"\n分析完了:", flush=True)
    print(f"  処理タスク数: {task_count}", flush=True)
    print(f"  スキップタスク数: {skipped_tasks}", flush=True)
    print(f"  処理グリッド数: {grid_count}", flush=True)
    print(f"  スキップグリッド数: {skipped_grids}", flush=True)
    print(f"  総オブジェクト数: {object_count}", flush=True)
    print(f"  総オブジェクト数: {total_patterns['total_objects']}", flush=True)

    return total_patterns


def calculate_statistics(patterns: Dict[str, int]) -> Dict[str, float]:
    """統計を計算（パーセンテージ）"""
    total = patterns['total_objects']

    if total == 0:
        return {
            'same_color_and_shape_percent': 0.0,
            'same_shape_only_percent': 0.0,
            'new_generation_percent': 0.0
        }

    return {
        'same_color_and_shape_percent': (patterns['same_color_and_shape'] / total) * 100,
        'same_shape_only_percent': (patterns['same_shape_only'] / total) * 100,
        'new_generation_percent': (patterns['new_generation'] / total) * 100
    }


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='ARC-AGI2データセットのオブジェクト複製パターン統計を取得')
    parser.add_argument('--max-tasks', type=int, default=None, help='分析する最大タスク数')
    args = parser.parse_args()

    # タスクを読み込む
    tasks = load_arc_agi2_tasks(max_tasks=args.max_tasks)

    if not tasks:
        print("タスクが読み込めませんでした", flush=True)
        return

    # すべてのグリッドを分析
    patterns = analyze_all_grids(tasks, max_tasks=args.max_tasks)

    # 統計を計算
    stats = calculate_statistics(patterns)

    # 結果を表示
    print("\n" + "="*60, flush=True)
    print("ARC-AGI2データセット オブジェクト複製パターン統計", flush=True)
    print("="*60, flush=True)
    print(f"\n総オブジェクト数: {patterns['total_objects']}", flush=True)
    print(f"\nパターン別の数:", flush=True)
    print(f"  同じ色と形状: {patterns['same_color_and_shape']} ({stats['same_color_and_shape_percent']:.2f}%)", flush=True)
    print(f"  同じ形状のみ（色は異なる）: {patterns['same_shape_only']} ({stats['same_shape_only_percent']:.2f}%)", flush=True)
    print(f"  新規生成（形状も色も異なる）: {patterns['new_generation']} ({stats['new_generation_percent']:.2f}%)", flush=True)

    print(f"\n確率値（小数）:", flush=True)
    if patterns['total_objects'] > 0:
        print(f"  SAME_COLOR_AND_SHAPE_PROBABILITY = {patterns['same_color_and_shape'] / patterns['total_objects']:.6f}  # 同じ色と形状: {stats['same_color_and_shape_percent']:.2f}%", flush=True)
        print(f"  SAME_SHAPE_PROBABILITY = {patterns['same_shape_only'] / patterns['total_objects']:.6f}  # 同じ形状（色は異なる）: {stats['same_shape_only_percent']:.2f}%", flush=True)
    else:
        print(f"  データが不足しています（総オブジェクト数が0）", flush=True)
    print(f"  # 残り{stats['new_generation_percent']:.2f}%は新規生成（異なる形状と色）", flush=True)

    print("\n" + "="*60, flush=True)


if __name__ == '__main__':
    main()
