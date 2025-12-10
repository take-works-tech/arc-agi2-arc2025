"""
ARC-AGI2データセットで、MERGEのみのパターン（カテゴリ分けなしで全オブジェクトをMERGE）が
どれだけ存在するかを分析するスクリプト

分析内容:
1. 入力グリッドに複数のオブジェクトがある
2. 出力グリッドのオブジェクト数が1個（または大幅に減少）
3. カテゴリ分けなしでMERGEのみで説明可能なタスクの割合
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
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
            print(f"ARC-AGI2タスクファイルを読み込み: {task_file}", flush=True)
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

    print("ARC-AGI2タスクファイルが見つかりませんでした", flush=True)
    return []


def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    """グリッドをnumpy配列に変換"""
    return np.array(grid, dtype=int)


def count_objects_in_grid(grid: np.ndarray, extractor: IntegratedObjectExtractor, connectivity: int = 4) -> int:
    """グリッド内のオブジェクト数をカウント（背景色フィルタ後）"""
    try:
        result = extractor.extract_objects_by_type(grid, input_image_index=0)
        if not result.success:
            return 0

        if connectivity == 4:
            objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
        else:
            objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_8WAY, [])

        # 背景色を推論（最頻出色）
        flat_grid = grid.flatten()
        if len(flat_grid) == 0:
            return 0
        background_color = Counter(flat_grid).most_common(1)[0][0]

        # 背景色以外のオブジェクトのみをカウント
        non_bg_objects = [obj for obj in objects if obj.dominant_color != background_color]
        return len(non_bg_objects)
    except Exception as e:
        return 0


def analyze_merge_patterns(max_tasks: Optional[int] = None) -> Dict[str, Any]:
    """MERGEのみのパターンを分析"""
    tasks = load_arc_agi2_tasks(max_tasks=max_tasks)

    if not tasks:
        return {}

    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)

    # 統計を記録
    total_tasks = 0
    total_train_pairs = 0

    # MERGEのみのパターンの候補
    merge_only_candidates = []

    # カテゴリ分けありのパターン
    categorized_candidates = []

    print(f"\nARC-AGI2データセットのMERGEパターンを分析中... (タスク数: {len(tasks)})", flush=True)

    for i, task in enumerate(tasks):
        if (i + 1) % 100 == 0:
            print(f"  処理済み: {i+1}/{len(tasks)}", flush=True)

        train_pairs = task.get('train', [])
        if not train_pairs:
            continue

        total_tasks += 1

        for pair_idx, pair in enumerate(train_pairs):
            total_train_pairs += 1

            input_grid = pair.get('input', [])
            output_grid = pair.get('output', [])

            if not input_grid or not output_grid:
                continue

            try:
                input_array = grid_to_numpy(input_grid)
                output_array = grid_to_numpy(output_grid)

                # オブジェクト数をカウント（4連結、背景色フィルタ後）
                input_obj_count_4 = count_objects_in_grid(input_array, extractor, connectivity=4)
                output_obj_count_4 = count_objects_in_grid(output_array, extractor, connectivity=4)

                # 8連結でもカウント
                input_obj_count_8 = count_objects_in_grid(input_array, extractor, connectivity=8)
                output_obj_count_8 = count_objects_in_grid(output_array, extractor, connectivity=8)

                # 入力に2個以上のオブジェクトがある場合のみ分析
                if input_obj_count_4 >= 2 or input_obj_count_8 >= 2:
                    # 出力が1個または大幅に減少した場合
                    if output_obj_count_4 == 1 or output_obj_count_8 == 1:
                        # MERGEのみのパターンの候補
                        merge_only_candidates.append({
                            'task_id': task.get('task_id'),
                            'pair_idx': pair_idx,
                            'input_obj_count_4': input_obj_count_4,
                            'input_obj_count_8': input_obj_count_8,
                            'output_obj_count_4': output_obj_count_4,
                            'output_obj_count_8': output_obj_count_8,
                        })
                    elif output_obj_count_4 < input_obj_count_4 or output_obj_count_8 < input_obj_count_8:
                        # オブジェクト数が減少したが、1個ではない場合（カテゴリ分けの可能性）
                        categorized_candidates.append({
                            'task_id': task.get('task_id'),
                            'pair_idx': pair_idx,
                            'input_obj_count_4': input_obj_count_4,
                            'input_obj_count_8': input_obj_count_8,
                            'output_obj_count_4': output_obj_count_4,
                            'output_obj_count_8': output_obj_count_8,
                        })
            except Exception as e:
                continue

    print(f"  処理完了: {total_train_pairs}ペアを分析", flush=True)

    # 統計を計算
    merge_only_count = len(merge_only_candidates)
    categorized_count = len(categorized_candidates)

    merge_only_ratio = (merge_only_count / total_train_pairs * 100) if total_train_pairs > 0 else 0.0
    categorized_ratio = (categorized_count / total_train_pairs * 100) if total_train_pairs > 0 else 0.0

    # 結果を出力
    print(f"\n=== 分析結果 ===", flush=True)
    print(f"総タスク数: {total_tasks}", flush=True)
    print(f"総訓練ペア数: {total_train_pairs}", flush=True)
    print(f"\nMERGEのみのパターン候補（出力が1個）:", flush=True)
    print(f"  数: {merge_only_count}件", flush=True)
    print(f"  割合: {merge_only_ratio:.2f}%", flush=True)
    print(f"\nカテゴリ分けありのパターン候補（出力が複数だが減少）:", flush=True)
    print(f"  数: {categorized_count}件", flush=True)
    print(f"  割合: {categorized_ratio:.2f}%", flush=True)

    # サンプルを出力
    if merge_only_candidates:
        print(f"\nMERGEのみパターンのサンプル（最初の10件）:", flush=True)
        for i, candidate in enumerate(merge_only_candidates[:10]):
            print(f"  {i+1}. タスクID: {candidate['task_id']}, ペア: {candidate['pair_idx']}, "
                  f"入力(4連結): {candidate['input_obj_count_4']}個, "
                  f"入力(8連結): {candidate['input_obj_count_8']}個, "
                  f"出力(4連結): {candidate['output_obj_count_4']}個, "
                  f"出力(8連結): {candidate['output_obj_count_8']}個", flush=True)

    return {
        'total_tasks': total_tasks,
        'total_train_pairs': total_train_pairs,
        'merge_only_count': merge_only_count,
        'merge_only_ratio': merge_only_ratio,
        'categorized_count': categorized_count,
        'categorized_ratio': categorized_ratio,
        'merge_only_candidates': merge_only_candidates[:20],  # 最初の20件のみ保存
        'categorized_candidates': categorized_candidates[:20],  # 最初の20件のみ保存
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ARC-AGI2データセットでMERGEパターンを分析')
    parser.add_argument('--max-tasks', type=int, default=None, help='最大タスク数（テスト用）')
    args = parser.parse_args()

    results = analyze_merge_patterns(max_tasks=args.max_tasks)

    # 結果をJSONファイルに保存
    output_file = project_root / "outputs" / "merge_patterns_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存しました: {output_file}", flush=True)
