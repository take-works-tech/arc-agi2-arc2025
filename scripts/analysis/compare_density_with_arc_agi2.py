"""
生成されたデータセットとARC-AGI2データセットのオブジェクト密度を比較
- グリッドサイズに対するオブジェクトの密度
- オブジェクトピクセル比
"""
import json
import gzip
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


def load_generated_data_pairs(data_file: Path) -> List[Dict[str, Any]]:
    """生成されたデータペアを読み込む"""
    print(f"生成データセットを読み込み: {data_file}", flush=True)
    data_pairs = []
    
    if data_file.suffix == '.gz':
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(data_file, mode, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_pairs.append(json.loads(line))
    
    print(f"読み込んだデータペア数: {len(data_pairs)}", flush=True)
    return data_pairs


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
                'object_pixel_ratio': 0.0,
                'grid_width': grid.shape[1],
                'grid_height': grid.shape[0],
                'grid_area': grid.shape[0] * grid.shape[1],
            }

        # オブジェクトピクセル比（密度）
        total_pixels = grid.shape[0] * grid.shape[1]
        background_color = Counter(grid.flatten()).most_common(1)[0][0]
        object_pixels = np.sum(grid != background_color)
        object_pixel_ratio = object_pixels / total_pixels if total_pixels > 0 else 0.0

        return {
            'num_objects': len(objects),
            'object_pixel_ratio': object_pixel_ratio,
            'grid_width': grid.shape[1],
            'grid_height': grid.shape[0],
            'grid_area': grid.shape[0] * grid.shape[1],
        }
    except Exception as e:
        print(f"エラー: グリッド統計抽出失敗: {e}", flush=True)
        return None


def analyze_arc_agi2_statistics(max_tasks: Optional[int] = None) -> Dict[str, Any]:
    """ARC-AGI2データセットの統計を分析"""
    print("\n【ARC-AGI2データセットの統計を分析中...】", flush=True)
    
    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)
    
    tasks = load_arc_agi2_tasks(max_tasks=max_tasks)
    if not tasks:
        return {}
    
    all_stats = []
    for task in tasks:
        for pair in task.get('train', []):
            input_grid = grid_to_numpy(pair.get('input', []))
            if input_grid.size > 0:
                stats = extract_statistics_from_grid(input_grid, extractor)
                if stats:
                    all_stats.append(stats)
    
    if not all_stats:
        return {}
    
    # 統計を集計
    object_pixel_ratios = [s['object_pixel_ratio'] for s in all_stats]
    grid_areas = [s['grid_area'] for s in all_stats]
    num_objects_list = [s['num_objects'] for s in all_stats]
    
    return {
        'object_pixel_ratio': {
            'mean': np.mean(object_pixel_ratios),
            'median': np.median(object_pixel_ratios),
            'min': np.min(object_pixel_ratios),
            'max': np.max(object_pixel_ratios),
            'std': np.std(object_pixel_ratios),
        },
        'grid_area': {
            'mean': np.mean(grid_areas),
            'median': np.median(grid_areas),
            'min': np.min(grid_areas),
            'max': np.max(grid_areas),
        },
        'num_objects': {
            'mean': np.mean(num_objects_list),
            'median': np.median(num_objects_list),
            'min': np.min(num_objects_list),
            'max': np.max(num_objects_list),
        },
        'sample_count': len(all_stats),
    }


def analyze_generated_statistics(data_file: Path) -> Dict[str, Any]:
    """生成されたデータセットの統計を分析"""
    print("\n【生成データセットの統計を分析中...】", flush=True)
    
    config = ExtractionConfig()
    extractor = IntegratedObjectExtractor(config)
    
    data_pairs = load_generated_data_pairs(data_file)
    if not data_pairs:
        return {}
    
    all_stats = []
    for pair in data_pairs:
        input_grid = pair.get('input')
        if input_grid:
            if isinstance(input_grid, list):
                input_grid = grid_to_numpy(input_grid)
            elif isinstance(input_grid, np.ndarray):
                pass
            else:
                continue
            
            if input_grid.size > 0:
                stats = extract_statistics_from_grid(input_grid, extractor)
                if stats:
                    all_stats.append(stats)
    
    if not all_stats:
        return {}
    
    # 統計を集計
    object_pixel_ratios = [s['object_pixel_ratio'] for s in all_stats]
    grid_areas = [s['grid_area'] for s in all_stats]
    num_objects_list = [s['num_objects'] for s in all_stats]
    
    return {
        'object_pixel_ratio': {
            'mean': np.mean(object_pixel_ratios),
            'median': np.median(object_pixel_ratios),
            'min': np.min(object_pixel_ratios),
            'max': np.max(object_pixel_ratios),
            'std': np.std(object_pixel_ratios),
        },
        'grid_area': {
            'mean': np.mean(grid_areas),
            'median': np.median(grid_areas),
            'min': np.min(grid_areas),
            'max': np.max(grid_areas),
        },
        'num_objects': {
            'mean': np.mean(num_objects_list),
            'median': np.median(num_objects_list),
            'min': np.min(num_objects_list),
            'max': np.max(num_objects_list),
        },
        'sample_count': len(all_stats),
    }


def print_comparison(arc_agi2_stats: Dict[str, Any], generated_stats: Dict[str, Any]):
    """比較結果を表示"""
    print("\n" + "=" * 80)
    print("【生成データセット vs ARC-AGI2データセット 比較結果】")
    print("=" * 80)
    
    # ①オブジェクトの画像に対する密度（オブジェクトピクセル比）
    print("\n【①オブジェクトの画像に対する密度（オブジェクトピクセル比）】")
    print("-" * 80)
    if 'object_pixel_ratio' in arc_agi2_stats and 'object_pixel_ratio' in generated_stats:
        agi2 = arc_agi2_stats['object_pixel_ratio']
        gen = generated_stats['object_pixel_ratio']
        print(f"ARC-AGI2:  平均={agi2['mean']:.3f}, 中央値={agi2['median']:.3f}, 最小={agi2['min']:.3f}, 最大={agi2['max']:.3f}, 標準偏差={agi2['std']:.3f}")
        print(f"生成:      平均={gen['mean']:.3f}, 中央値={gen['median']:.3f}, 最小={gen['min']:.3f}, 最大={gen['max']:.3f}, 標準偏差={gen['std']:.3f}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.3f} ({diff_pct:+.1f}%)")
    
    # ②グリッドサイズ（面積）
    print("\n【②グリッドサイズ（面積）】")
    print("-" * 80)
    if 'grid_area' in arc_agi2_stats and 'grid_area' in generated_stats:
        agi2 = arc_agi2_stats['grid_area']
        gen = generated_stats['grid_area']
        print(f"ARC-AGI2:  平均={agi2['mean']:.1f}, 中央値={agi2['median']:.1f}, 最小={agi2['min']}, 最大={agi2['max']}")
        print(f"生成:      平均={gen['mean']:.1f}, 中央値={gen['median']:.1f}, 最小={gen['min']}, 最大={gen['max']}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.1f} ({diff_pct:+.1f}%)")
    
    # ③オブジェクト数
    print("\n【③オブジェクト数】")
    print("-" * 80)
    if 'num_objects' in arc_agi2_stats and 'num_objects' in generated_stats:
        agi2 = arc_agi2_stats['num_objects']
        gen = generated_stats['num_objects']
        print(f"ARC-AGI2:  平均={agi2['mean']:.2f}, 中央値={agi2['median']:.2f}, 最小={agi2['min']}, 最大={agi2['max']}")
        print(f"生成:      平均={gen['mean']:.2f}, 中央値={gen['median']:.2f}, 最小={gen['min']}, 最大={gen['max']}")
        diff_mean = gen['mean'] - agi2['mean']
        diff_pct = (diff_mean / agi2['mean'] * 100) if agi2['mean'] > 0 else 0
        print(f"差:        平均値の差={diff_mean:+.2f} ({diff_pct:+.1f}%)")
    
    # ④サンプル数
    print("\n【④サンプル数】")
    print("-" * 80)
    print(f"ARC-AGI2:  {arc_agi2_stats.get('sample_count', 0)}サンプル")
    print(f"生成:      {generated_stats.get('sample_count', 0)}サンプル")
    
    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成されたデータセットとARC-AGI2データセットのオブジェクト密度を比較')
    parser.add_argument('--generated-data', type=str, required=True, help='生成されたデータセットのパス（data_pairs.jsonl.gz）')
    parser.add_argument('--arc-agi2-max-tasks', type=int, default=None, help='ARC-AGI2データセットの最大タスク数')
    args = parser.parse_args()

    data_file = Path(args.generated_data)
    if not data_file.exists():
        print(f"エラー: データファイルが見つかりません: {data_file}", flush=True)
        return

    # ARC-AGI2の統計を抽出
    arc_agi2_stats = analyze_arc_agi2_statistics(max_tasks=args.arc_agi2_max_tasks)

    # 生成されたデータセットの統計を抽出
    generated_stats = analyze_generated_statistics(data_file)

    # 比較して表示
    print_comparison(arc_agi2_stats, generated_stats)


if __name__ == '__main__':
    main()


