"""
ARC-AGI2データセットでグリッド面積とオブジェクト数の関係を分析
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
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


def extract_statistics_from_grid(grid: np.ndarray, extractor: IntegratedObjectExtractor) -> Optional[Dict[str, Any]]:
    """グリッドから統計を抽出"""
    try:
        # オブジェクトを抽出
        result = extractor.extract_objects_by_type(grid)
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

        grid_area = grid.shape[0] * grid.shape[1]
        num_objects = len(objects)

        return {
            'num_objects': num_objects,
            'grid_area': grid_area,
            'grid_width': grid.shape[1],
            'grid_height': grid.shape[0],
        }
    except Exception as e:
        print(f"エラー: グリッド統計抽出失敗: {e}", flush=True)
        return None


def analyze_by_grid_area(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """グリッド面積ごとにオブジェクト数を分析"""
    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)

    # グリッド面積の範囲でグループ化
    area_ranges = [
        (0, 100, "< 100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 400, "300-400"),
        (400, 600, "400-600"),
        (600, float('inf'), ">= 600"),
    ]

    stats_by_area = defaultdict(list)

    for task in tasks:
        for pair in task.get('train', []):
            input_grid = grid_to_numpy(pair.get('input', []))
            if input_grid.size > 0:
                stats = extract_statistics_from_grid(input_grid, extractor)
                if stats:
                    grid_area = stats['grid_area']
                    num_objects = stats['num_objects']

                    # 面積範囲を決定
                    for min_area, max_area, label in area_ranges:
                        if min_area <= grid_area < max_area:
                            stats_by_area[label].append({
                                'grid_area': grid_area,
                                'num_objects': num_objects,
                            })
                            break

    # 各面積範囲の統計を計算
    results = {}
    for label, data_list in stats_by_area.items():
        if data_list:
            num_objects_list = [d['num_objects'] for d in data_list]
            grid_areas_list = [d['grid_area'] for d in data_list]

            results[label] = {
                'count': len(data_list),
                'num_objects': {
                    'mean': np.mean(num_objects_list),
                    'median': np.median(num_objects_list),
                    'min': np.min(num_objects_list),
                    'max': np.max(num_objects_list),
                    'std': np.std(num_objects_list),
                },
                'grid_area': {
                    'mean': np.mean(grid_areas_list),
                    'median': np.median(grid_areas_list),
                    'min': np.min(grid_areas_list),
                    'max': np.max(grid_areas_list),
                },
                'objects_per_area': np.mean(num_objects_list) / np.mean(grid_areas_list) if np.mean(grid_areas_list) > 0 else 0,
            }

    return results


def print_analysis(results: Dict[str, Any]):
    """分析結果を表示"""
    print("\n" + "=" * 80)
    print("【ARC-AGI2: グリッド面積ごとのオブジェクト数分析】")
    print("=" * 80)

    area_ranges_order = ["< 100", "100-200", "200-300", "300-400", "400-600", ">= 600"]

    for label in area_ranges_order:
        if label in results:
            data = results[label]
            print(f"\n【グリッド面積: {label}】")
            print("-" * 80)
            print(f"サンプル数: {data['count']}")
            print(f"グリッド面積: 平均={data['grid_area']['mean']:.1f}, 中央値={data['grid_area']['median']:.1f}, 最小={data['grid_area']['min']}, 最大={data['grid_area']['max']}")
            print(f"オブジェクト数: 平均={data['num_objects']['mean']:.2f}, 中央値={data['num_objects']['median']:.2f}, 最小={data['num_objects']['min']}, 最大={data['num_objects']['max']}, 標準偏差={data['num_objects']['std']:.2f}")
            print(f"オブジェクト数/面積比: {data['objects_per_area']:.4f}")

    print("\n" + "=" * 80)
    print("【現在の実装との比較】")
    print("=" * 80)
    print("\n現在の実装（decide_num_objects_by_arc_statistics）:")
    print("  - グリッド面積 < 100: base_num * 0.6-0.7")
    print("  - グリッド面積 < 200: base_num * 0.8-0.85")
    print("  - グリッド面積 < 300: base_num * 1.0")
    print("  - グリッド面積 < 400: base_num * 1.1")
    print("  - グリッド面積 < 600: base_num * 1.3")
    print("  - グリッド面積 >= 600: base_num * 1.5")
    print("\nARC-AGI2の実際のデータ:")
    for label in area_ranges_order:
        if label in results:
            data = results[label]
            avg_objects = data['num_objects']['mean']
            avg_area = data['grid_area']['mean']
            print(f"  - グリッド面積 {label}: 平均オブジェクト数={avg_objects:.2f}, 平均面積={avg_area:.1f}, 比率={avg_objects/avg_area:.4f}")


def main():
    tasks = load_arc_agi2_tasks()
    if not tasks:
        print("ARC-AGI2タスクが見つかりませんでした", flush=True)
        return

    print(f"読み込んだタスク数: {len(tasks)}", flush=True)

    results = analyze_by_grid_area(tasks)
    print_analysis(results)


if __name__ == '__main__':
    main()

