"""
生成された入力グリッドとARC-AGI2データセットの統計を比較
- 1ピクセルオブジェクトの数
- オブジェクトの画像に対する密度（オブジェクトピクセル比）
- グリッドサイズ
- オブジェクトのサイズ
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
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main


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


def extract_statistics_from_grid(grid: np.ndarray, extractor: IntegratedObjectExtractor) -> Dict[str, Any]:
    """グリッドから統計を抽出"""
    try:
        # オブジェクトを抽出
        result = extractor.extract_objects_by_type(grid)
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

        if not objects:
            return {
                'num_objects': 0,
                'num_single_pixel_objects': 0,
                'single_pixel_ratio': 0.0,
                'object_pixel_ratio': 0.0,
                'grid_width': grid.shape[1],
                'grid_height': grid.shape[0],
                'grid_area': grid.shape[0] * grid.shape[1],
                'object_sizes': [],
                'object_areas': [],
            }

        # 統計を計算
        num_objects = len(objects)
        num_single_pixel_objects = sum(1 for obj in objects if len(obj.pixels) == 1)
        single_pixel_ratio = num_single_pixel_objects / num_objects if num_objects > 0 else 0.0

        # オブジェクトピクセル比
        total_pixels = grid.shape[0] * grid.shape[1]
        background_color = Counter(grid.flatten()).most_common(1)[0][0]
        object_pixels = np.sum(grid != background_color)
        object_pixel_ratio = object_pixels / total_pixels if total_pixels > 0 else 0.0

        # オブジェクトサイズ（幅×高さ）
        object_sizes = [(obj.bbox_width, obj.bbox_height) for obj in objects]

        # オブジェクト面積（ピクセル数）
        object_areas = [len(obj.pixels) for obj in objects]

        return {
            'num_objects': num_objects,
            'num_single_pixel_objects': num_single_pixel_objects,
            'single_pixel_ratio': single_pixel_ratio,
            'object_pixel_ratio': object_pixel_ratio,
            'grid_width': grid.shape[1],
            'grid_height': grid.shape[0],
            'grid_area': grid.shape[0] * grid.shape[1],
            'object_sizes': object_sizes,
            'object_areas': object_areas,
        }
    except Exception as e:
        print(f"  統計抽出エラー: {e}", flush=True)
        return {
            'num_objects': 0,
            'num_single_pixel_objects': 0,
            'single_pixel_ratio': 0.0,
            'object_pixel_ratio': 0.0,
            'grid_width': grid.shape[1],
            'grid_height': grid.shape[0],
            'grid_area': grid.shape[0] * grid.shape[1],
            'object_sizes': [],
            'object_areas': [],
        }


def analyze_arc_agi2_statistics(max_tasks: Optional[int] = None) -> Dict[str, Any]:
    """ARC-AGI2データセットから統計を抽出"""
    tasks = load_arc_agi2_tasks(max_tasks=max_tasks)

    if not tasks:
        return {}

    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)

    all_stats = []

    print(f"\nARC-AGI2データセットの統計を抽出中... (タスク数: {len(tasks)})", flush=True)

    for i, task in enumerate(tasks):
        if (i + 1) % 100 == 0:
            print(f"  処理済み: {i+1}/{len(tasks)}", flush=True)

        # 訓練ペアの入力グリッドのみを分析
        for pair in task.get('train', []):
            input_grid = grid_to_numpy(pair.get('input', []))
            if input_grid.size == 0:
                continue

            stats = extract_statistics_from_grid(input_grid, extractor)
            all_stats.append(stats)

    print(f"  処理完了: {len(all_stats)}グリッドを分析", flush=True)

    # 統計を集計
    return aggregate_statistics(all_stats)


def generate_and_analyze_statistics(num_samples: int = 100) -> Dict[str, Any]:
    """生成された入力グリッドから統計を抽出"""
    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)

    all_stats = []

    print(f"\n生成された入力グリッドの統計を抽出中... (サンプル数: {num_samples})", flush=True)

    success_count = 0
    failure_count = 0

    for i in range(num_samples):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()

            # 入力グリッド生成
            _, input_grid, _, _ = core_executor_main(
                nodes=None,
                grid_width=grid_width,
                grid_height=grid_height,
                enable_replacement=False
            )

            if input_grid is None:
                failure_count += 1
                continue

            # 単色グリッドチェック
            unique_colors = np.unique(input_grid)
            if len(unique_colors) <= 1:
                failure_count += 1
                continue

            # 統計を抽出
            stats = extract_statistics_from_grid(input_grid, extractor)
            all_stats.append(stats)
            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"  処理済み: {i+1}/{num_samples} (成功: {success_count}, 失敗: {failure_count})", flush=True)

        except Exception as e:
            print(f"  [エラー] サンプル{i+1}でエラーが発生: {e}", flush=True)
            failure_count += 1
            continue

    print(f"  処理完了: {success_count}グリッドを分析", flush=True)

    # 統計を集計
    return aggregate_statistics(all_stats)


def aggregate_statistics(all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """統計を集計"""
    if not all_stats:
        return {}

    # 基本統計
    num_objects_list = [s['num_objects'] for s in all_stats]
    num_single_pixel_objects_list = [s['num_single_pixel_objects'] for s in all_stats]
    single_pixel_ratio_list = [s['single_pixel_ratio'] for s in all_stats]
    object_pixel_ratio_list = [s['object_pixel_ratio'] for s in all_stats]
    grid_width_list = [s['grid_width'] for s in all_stats]
    grid_height_list = [s['grid_height'] for s in all_stats]
    grid_area_list = [s['grid_area'] for s in all_stats]

    # オブジェクトサイズと面積（全オブジェクトを統合）
    all_object_sizes = []
    all_object_areas = []
    for s in all_stats:
        all_object_sizes.extend(s['object_sizes'])
        all_object_areas.extend(s['object_areas'])

    return {
        'num_grids': len(all_stats),
        'num_objects': {
            'mean': np.mean(num_objects_list),
            'std': np.std(num_objects_list),
            'min': np.min(num_objects_list),
            'max': np.max(num_objects_list),
            'median': np.median(num_objects_list),
        },
        'num_single_pixel_objects': {
            'mean': np.mean(num_single_pixel_objects_list),
            'std': np.std(num_single_pixel_objects_list),
            'min': np.min(num_single_pixel_objects_list),
            'max': np.max(num_single_pixel_objects_list),
            'median': np.median(num_single_pixel_objects_list),
        },
        'single_pixel_ratio': {
            'mean': np.mean(single_pixel_ratio_list),
            'std': np.std(single_pixel_ratio_list),
            'min': np.min(single_pixel_ratio_list),
            'max': np.max(single_pixel_ratio_list),
            'median': np.median(single_pixel_ratio_list),
        },
        'object_pixel_ratio': {
            'mean': np.mean(object_pixel_ratio_list),
            'std': np.std(object_pixel_ratio_list),
            'min': np.min(object_pixel_ratio_list),
            'max': np.max(object_pixel_ratio_list),
            'median': np.median(object_pixel_ratio_list),
        },
        'grid_width': {
            'mean': np.mean(grid_width_list),
            'std': np.std(grid_width_list),
            'min': np.min(grid_width_list),
            'max': np.max(grid_width_list),
            'median': np.median(grid_width_list),
        },
        'grid_height': {
            'mean': np.mean(grid_height_list),
            'std': np.std(grid_height_list),
            'min': np.min(grid_height_list),
            'max': np.max(grid_height_list),
            'median': np.median(grid_height_list),
        },
        'grid_area': {
            'mean': np.mean(grid_area_list),
            'std': np.std(grid_area_list),
            'min': np.min(grid_area_list),
            'max': np.max(grid_area_list),
            'median': np.median(grid_area_list),
        },
        'object_sizes': {
            'width_mean': np.mean([w for w, h in all_object_sizes]) if all_object_sizes else 0,
            'width_std': np.std([w for w, h in all_object_sizes]) if all_object_sizes else 0,
            'height_mean': np.mean([h for w, h in all_object_sizes]) if all_object_sizes else 0,
            'height_std': np.std([h for w, h in all_object_sizes]) if all_object_sizes else 0,
        },
        'object_areas': {
            'mean': np.mean(all_object_areas) if all_object_areas else 0,
            'std': np.std(all_object_areas) if all_object_areas else 0,
            'min': np.min(all_object_areas) if all_object_areas else 0,
            'max': np.max(all_object_areas) if all_object_areas else 0,
            'median': np.median(all_object_areas) if all_object_areas else 0,
        },
    }


def print_comparison(arc_agi2_stats: Dict[str, Any], generated_stats: Dict[str, Any]):
    """統計を比較して表示"""
    print("\n" + "=" * 80)
    print("統計比較: ARC-AGI2 vs 生成された入力グリッド")
    print("=" * 80)

    if not arc_agi2_stats or not generated_stats:
        print("統計データが不足しています")
        return

    # ①1ピクセルオブジェクトの数
    print("\n【①1ピクセルオブジェクトの数】")
    print("-" * 80)
    if 'num_single_pixel_objects' in arc_agi2_stats and 'num_single_pixel_objects' in generated_stats:
        agi2 = arc_agi2_stats['num_single_pixel_objects']
        gen = generated_stats['num_single_pixel_objects']
        print(f"ARC-AGI2:  平均={agi2['mean']:.2f}, 中央値={agi2['median']:.2f}, 最小={agi2['min']}, 最大={agi2['max']}")
        print(f"生成:      平均={gen['mean']:.2f}, 中央値={gen['median']:.2f}, 最小={gen['min']}, 最大={gen['max']}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.2f} ({diff_pct:+.1f}%)")

    # 1ピクセルオブジェクトの比率
    print("\n【①-補足: 1ピクセルオブジェクトの比率】")
    print("-" * 80)
    if 'single_pixel_ratio' in arc_agi2_stats and 'single_pixel_ratio' in generated_stats:
        agi2 = arc_agi2_stats['single_pixel_ratio']
        gen = generated_stats['single_pixel_ratio']
        print(f"ARC-AGI2:  平均={agi2['mean']:.3f}, 中央値={agi2['median']:.3f}")
        print(f"生成:      平均={gen['mean']:.3f}, 中央値={gen['median']:.3f}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.3f} ({diff_pct:+.1f}%)")

    # ②オブジェクトの画像に対する密度（オブジェクトピクセル比）
    print("\n【②オブジェクトの画像に対する密度（オブジェクトピクセル比）】")
    print("-" * 80)
    if 'object_pixel_ratio' in arc_agi2_stats and 'object_pixel_ratio' in generated_stats:
        agi2 = arc_agi2_stats['object_pixel_ratio']
        gen = generated_stats['object_pixel_ratio']
        print(f"ARC-AGI2:  平均={agi2['mean']:.3f}, 中央値={agi2['median']:.3f}, 最小={agi2['min']:.3f}, 最大={agi2['max']:.3f}")
        print(f"生成:      平均={gen['mean']:.3f}, 中央値={gen['median']:.3f}, 最小={gen['min']:.3f}, 最大={gen['max']:.3f}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.3f} ({diff_pct:+.1f}%)")

    # ③グリッドサイズ
    print("\n【③グリッドサイズ】")
    print("-" * 80)
    if 'grid_width' in arc_agi2_stats and 'grid_width' in generated_stats:
        agi2_w = arc_agi2_stats['grid_width']
        gen_w = generated_stats['grid_width']
        agi2_h = arc_agi2_stats['grid_height']
        gen_h = generated_stats['grid_height']
        agi2_a = arc_agi2_stats['grid_area']
        gen_a = generated_stats['grid_area']

        print(f"幅 (Width):")
        print(f"  ARC-AGI2:  平均={agi2_w['mean']:.2f}, 中央値={agi2_w['median']:.2f}, 最小={agi2_w['min']}, 最大={agi2_w['max']}")
        print(f"  生成:      平均={gen_w['mean']:.2f}, 中央値={gen_w['median']:.2f}, 最小={gen_w['min']}, 最大={gen_w['max']}")
        diff_w = gen_w['mean'] - agi2_w['mean']
        diff_w_pct = (diff_w / agi2_w['mean'] * 100) if agi2_w['mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_w:+.2f} ({diff_w_pct:+.1f}%)")

        print(f"\n高さ (Height):")
        print(f"  ARC-AGI2:  平均={agi2_h['mean']:.2f}, 中央値={agi2_h['median']:.2f}, 最小={agi2_h['min']}, 最大={agi2_h['max']}")
        print(f"  生成:      平均={gen_h['mean']:.2f}, 中央値={gen_h['median']:.2f}, 最小={gen_h['min']}, 最大={gen_h['max']}")
        diff_h = gen_h['mean'] - agi2_h['mean']
        diff_h_pct = (diff_h / agi2_h['mean'] * 100) if agi2_h['mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_h:+.2f} ({diff_h_pct:+.1f}%)")

        print(f"\n面積 (Area):")
        print(f"  ARC-AGI2:  平均={agi2_a['mean']:.2f}, 中央値={agi2_a['median']:.2f}, 最小={agi2_a['min']}, 最大={agi2_a['max']}")
        print(f"  生成:      平均={gen_a['mean']:.2f}, 中央値={gen_a['median']:.2f}, 最小={gen_a['min']}, 最大={gen_a['max']}")
        diff_a = gen_a['mean'] - agi2_a['mean']
        diff_a_pct = (diff_a / agi2_a['mean'] * 100) if agi2_a['mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_a:+.2f} ({diff_a_pct:+.1f}%)")

    # ④オブジェクトのサイズ
    print("\n【④オブジェクトのサイズ】")
    print("-" * 80)
    if 'object_sizes' in arc_agi2_stats and 'object_sizes' in generated_stats:
        agi2 = arc_agi2_stats['object_sizes']
        gen = generated_stats['object_sizes']

        print(f"幅 (Width):")
        print(f"  ARC-AGI2:  平均={agi2['width_mean']:.2f}, 標準偏差={agi2['width_std']:.2f}")
        print(f"  生成:      平均={gen['width_mean']:.2f}, 標準偏差={gen['width_std']:.2f}")
        diff_w = gen['width_mean'] - agi2['width_mean']
        diff_w_pct = (diff_w / agi2['width_mean'] * 100) if agi2['width_mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_w:+.2f} ({diff_w_pct:+.1f}%)")

        print(f"\n高さ (Height):")
        print(f"  ARC-AGI2:  平均={agi2['height_mean']:.2f}, 標準偏差={agi2['height_std']:.2f}")
        print(f"  生成:      平均={gen['height_mean']:.2f}, 標準偏差={gen['height_std']:.2f}")
        diff_h = gen['height_mean'] - agi2['height_mean']
        diff_h_pct = (diff_h / agi2['height_mean'] * 100) if agi2['height_mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_h:+.2f} ({diff_h_pct:+.1f}%)")

    if 'object_areas' in arc_agi2_stats and 'object_areas' in generated_stats:
        agi2 = arc_agi2_stats['object_areas']
        gen = generated_stats['object_areas']

        print(f"\n面積 (Area - ピクセル数):")
        print(f"  ARC-AGI2:  平均={agi2['mean']:.2f}, 中央値={agi2['median']:.2f}, 最小={agi2['min']}, 最大={agi2['max']}")
        print(f"  生成:      平均={gen['mean']:.2f}, 中央値={gen['median']:.2f}, 最小={gen['min']}, 最大={gen['max']}")
        diff_a = gen['mean'] - agi2['mean']
        diff_a_pct = (diff_a / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"  差:        平均値の差={diff_a:+.2f} ({diff_a_pct:+.1f}%)")

    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成された入力グリッドとARC-AGI2データセットの統計を比較')
    parser.add_argument('--num-samples', type=int, default=100, help='生成するサンプル数')
    parser.add_argument('--arc-agi2-max-tasks', type=int, default=None, help='ARC-AGI2データセットの最大タスク数')
    args = parser.parse_args()

    # ARC-AGI2の統計を抽出
    arc_agi2_stats = analyze_arc_agi2_statistics(max_tasks=args.arc_agi2_max_tasks)

    # 生成されたグリッドの統計を抽出
    generated_stats = generate_and_analyze_statistics(num_samples=args.num_samples)

    # 比較して表示
    print_comparison(arc_agi2_stats, generated_stats)


if __name__ == '__main__':
    main()
