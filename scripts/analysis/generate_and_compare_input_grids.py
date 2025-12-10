"""
10,000タスク分の入力グリッドを生成し、ARC-AGI2の実データセットと比較分析する

比較項目:
1. グリッドサイズの分布（幅×高さ）
2. 色数の分布（背景色を含む/含まない）
3. オブジェクト数の分布
4. オブジェクトサイズの分布
5. 背景色の分布
6. 色の組み合わせ
7. グリッドの密度（オブジェクトピクセル/総ピクセル数）
8. オブジェクトの形状タイプ
"""
import sys
import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import time
from datetime import datetime

# joblibエラーを防ぐため、環境変数をインポート前に設定
try:
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
except Exception:
    cpu_count = 4

# 環境変数を設定（joblibの初期化前に設定する必要がある）
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 警告フィルターも設定
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')

# 必要なモジュールをインポート
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.hybrid_system.inference.object_matching.object_extractor import ObjectExtractor


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


def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    """グリッドをnumpy配列に変換"""
    return np.array(grid, dtype=np.int32)


def extract_grid_statistics(grid: np.ndarray, extract_objects: bool = False) -> Dict[str, Any]:
    """グリッドから統計情報を抽出

    Args:
        grid: グリッド配列 (height, width)
        extract_objects: オブジェクト抽出を行うかどうか

    Returns:
        統計情報の辞書
    """
    stats = {}

    # 基本的なグリッド情報
    height, width = grid.shape
    stats['width'] = width
    stats['height'] = height
    stats['area'] = width * height

    # 色情報
    unique_colors = np.unique(grid)
    stats['num_colors_total'] = len(unique_colors)
    stats['unique_colors'] = unique_colors.tolist()

    # 背景色の推論（最も多い色を背景色とする）
    color_counts = Counter(grid.flatten())
    background_color = color_counts.most_common(1)[0][0]
    stats['background_color'] = int(background_color)

    # 背景色以外の色
    object_colors = [c for c in unique_colors if c != background_color]
    stats['num_colors_object'] = len(object_colors)
    stats['object_colors'] = [int(c) for c in object_colors]

    # 色の分布
    stats['color_distribution'] = {int(k): int(v) for k, v in color_counts.items()}

    # オブジェクトピクセル数（背景色以外のピクセル数）
    object_pixels = np.sum(grid != background_color)
    stats['object_pixel_count'] = int(object_pixels)
    stats['object_pixel_ratio'] = float(object_pixels / stats['area']) if stats['area'] > 0 else 0.0

    # オブジェクト抽出（オプション）
    if extract_objects:
        try:
            extractor = ObjectExtractor()
            objects = extractor.extract_objects(grid)

            stats['num_objects'] = len(objects)

            if objects:
                object_sizes = [obj.size for obj in objects]
                object_colors_from_objects = [obj.color for obj in objects]

                stats['object_sizes'] = object_sizes
                stats['min_object_size'] = int(min(object_sizes))
                stats['max_object_size'] = int(max(object_sizes))
                stats['avg_object_size'] = float(np.mean(object_sizes))
                stats['object_colors_from_objects'] = [int(c) for c in object_colors_from_objects]

                # 形状タイプの分布
                shape_types = [obj.shape_type for obj in objects if hasattr(obj, 'shape_type')]
                if shape_types:
                    stats['shape_type_distribution'] = dict(Counter(shape_types))
            else:
                stats['num_objects'] = 0
                stats['min_object_size'] = 0
                stats['max_object_size'] = 0
                stats['avg_object_size'] = 0.0
        except Exception as e:
            print(f"  警告: オブジェクト抽出に失敗しました: {e}")
            stats['num_objects'] = 0

    return stats


def generate_input_grids(num_tasks: int, output_dir: Path, progress_interval: int = 100) -> List[Dict[str, Any]]:
    """入力グリッドを生成し、統計情報を収集

    Args:
        num_tasks: 生成するタスク数
        output_dir: 出力ディレクトリ
        progress_interval: 進捗表示の間隔

    Returns:
        統計情報のリスト
    """
    print(f"\n{'='*80}")
    print(f"入力グリッド生成開始: {num_tasks}タスク")
    print(f"{'='*80}\n")

    statistics_list = []
    success_count = 0
    failure_count = 0
    start_time = time.time()

    for task_index in range(1, num_tasks + 1):
        try:
            # グリッドサイズ決定
            grid_width, grid_height = decide_grid_size()

            # 入力グリッド生成（プログラムなし）
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

            # 統計情報を抽出（オブジェクト抽出は後で行う）
            stats = extract_grid_statistics(input_grid, extract_objects=False)
            stats['task_index'] = task_index
            stats['generated'] = True
            statistics_list.append(stats)

            success_count += 1

            # 進捗表示
            if task_index % progress_interval == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / task_index
                remaining = (num_tasks - task_index) * avg_time
                print(f"  進捗: {task_index}/{num_tasks} (成功: {success_count}, 失敗: {failure_count}) - "
                      f"経過時間: {elapsed:.1f}秒, 残り時間: {remaining:.1f}秒")

        except Exception as e:
            failure_count += 1
            if task_index % progress_interval == 0:
                print(f"  タスク{task_index}でエラーが発生: {e}")

    total_time = time.time() - start_time
    print(f"\n生成完了: 成功={success_count}, 失敗={failure_count}, 総時間={total_time:.1f}秒")

    return statistics_list


def analyze_arc_agi2_tasks(tasks: List[Dict[str, Any]], progress_interval: int = 100) -> List[Dict[str, Any]]:
    """ARC-AGI2の実データから統計情報を抽出

    Args:
        tasks: ARC-AGI2タスクのリスト
        progress_interval: 進捗表示の間隔

    Returns:
        統計情報のリスト
    """
    print(f"\n{'='*80}")
    print(f"ARC-AGI2データセット分析開始: {len(tasks)}タスク")
    print(f"{'='*80}\n")

    statistics_list = []
    start_time = time.time()

    for task_idx, task in enumerate(tasks, 1):
        task_id = task.get('task_id', f'task_{task_idx}')
        train_pairs = task.get('train', [])

        # 各trainペアの入力グリッドを分析
        for pair_idx, pair in enumerate(train_pairs):
            input_grid_data = pair.get('input', [])
            if not input_grid_data:
                continue

            try:
                input_grid = grid_to_numpy(input_grid_data)

                # 統計情報を抽出（オブジェクト抽出は後で行う）
                stats = extract_grid_statistics(input_grid, extract_objects=False)
                stats['task_id'] = task_id
                stats['pair_index'] = pair_idx
                stats['generated'] = False
                statistics_list.append(stats)

            except Exception as e:
                if task_idx % progress_interval == 0:
                    print(f"  タスク{task_id}のペア{pair_idx}でエラーが発生: {e}")

        # 進捗表示
        if task_idx % progress_interval == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / task_idx
            remaining = (len(tasks) - task_idx) * avg_time if task_idx < len(tasks) else 0
            print(f"  進捗: {task_idx}/{len(tasks)} - 経過時間: {elapsed:.1f}秒, 残り時間: {remaining:.1f}秒")

    total_time = time.time() - start_time
    print(f"\n分析完了: {len(statistics_list)}グリッド, 総時間={total_time:.1f}秒")

    return statistics_list


def compare_statistics(
    generated_stats: List[Dict[str, Any]],
    arc_stats: List[Dict[str, Any]],
    output_file: Path
) -> Dict[str, Any]:
    """生成されたグリッドとARC-AGI2実データの統計を比較

    Args:
        generated_stats: 生成されたグリッドの統計情報
        arc_stats: ARC-AGI2実データの統計情報
        output_file: 比較結果の出力ファイル

    Returns:
        比較結果の辞書
    """
    print(f"\n{'='*80}")
    print("統計比較開始")
    print(f"{'='*80}\n")

    comparison = {
        'generated_count': len(generated_stats),
        'arc_count': len(arc_stats),
        'comparison_timestamp': datetime.now().isoformat(),
        'comparisons': {}
    }

    # 1. グリッドサイズの分布
    print("1. グリッドサイズの分布を比較中...")
    gen_widths = [s['width'] for s in generated_stats]
    gen_heights = [s['height'] for s in generated_stats]
    gen_areas = [s['area'] for s in generated_stats]

    arc_widths = [s['width'] for s in arc_stats]
    arc_heights = [s['height'] for s in arc_stats]
    arc_areas = [s['area'] for s in arc_stats]

    comparison['comparisons']['grid_size'] = {
        'generated': {
            'width': {
                'min': int(np.min(gen_widths)),
                'max': int(np.max(gen_widths)),
                'mean': float(np.mean(gen_widths)),
                'median': float(np.median(gen_widths)),
                'std': float(np.std(gen_widths))
            },
            'height': {
                'min': int(np.min(gen_heights)),
                'max': int(np.max(gen_heights)),
                'mean': float(np.mean(gen_heights)),
                'median': float(np.median(gen_heights)),
                'std': float(np.std(gen_heights))
            },
            'area': {
                'min': int(np.min(gen_areas)),
                'max': int(np.max(gen_areas)),
                'mean': float(np.mean(gen_areas)),
                'median': float(np.median(gen_areas)),
                'std': float(np.std(gen_areas))
            }
        },
        'arc_agi2': {
            'width': {
                'min': int(np.min(arc_widths)),
                'max': int(np.max(arc_widths)),
                'mean': float(np.mean(arc_widths)),
                'median': float(np.median(arc_widths)),
                'std': float(np.std(arc_widths))
            },
            'height': {
                'min': int(np.min(arc_heights)),
                'max': int(np.max(arc_heights)),
                'mean': float(np.mean(arc_heights)),
                'median': float(np.median(arc_heights)),
                'std': float(np.std(arc_heights))
            },
            'area': {
                'min': int(np.min(arc_areas)),
                'max': int(np.max(arc_areas)),
                'mean': float(np.mean(arc_areas)),
                'median': float(np.median(arc_areas)),
                'std': float(np.std(arc_areas))
            }
        }
    }

    # 2. 色数の分布
    print("2. 色数の分布を比較中...")
    gen_color_counts_total = [s['num_colors_total'] for s in generated_stats]
    gen_color_counts_object = [s['num_colors_object'] for s in generated_stats]

    arc_color_counts_total = [s['num_colors_total'] for s in arc_stats]
    arc_color_counts_object = [s['num_colors_object'] for s in arc_stats]

    comparison['comparisons']['color_count'] = {
        'generated': {
            'total': {
                'min': int(np.min(gen_color_counts_total)),
                'max': int(np.max(gen_color_counts_total)),
                'mean': float(np.mean(gen_color_counts_total)),
                'median': float(np.median(gen_color_counts_total)),
                'distribution': dict(Counter(gen_color_counts_total))
            },
            'object': {
                'min': int(np.min(gen_color_counts_object)),
                'max': int(np.max(gen_color_counts_object)),
                'mean': float(np.mean(gen_color_counts_object)),
                'median': float(np.median(gen_color_counts_object)),
                'distribution': dict(Counter(gen_color_counts_object))
            }
        },
        'arc_agi2': {
            'total': {
                'min': int(np.min(arc_color_counts_total)),
                'max': int(np.max(arc_color_counts_total)),
                'mean': float(np.mean(arc_color_counts_total)),
                'median': float(np.median(arc_color_counts_total)),
                'distribution': dict(Counter(arc_color_counts_total))
            },
            'object': {
                'min': int(np.min(arc_color_counts_object)),
                'max': int(np.max(arc_color_counts_object)),
                'mean': float(np.mean(arc_color_counts_object)),
                'median': float(np.median(arc_color_counts_object)),
                'distribution': dict(Counter(arc_color_counts_object))
            }
        }
    }

    # 3. 背景色の分布
    print("3. 背景色の分布を比較中...")
    gen_bg_colors = [s['background_color'] for s in generated_stats]
    arc_bg_colors = [s['background_color'] for s in arc_stats]

    comparison['comparisons']['background_color'] = {
        'generated': dict(Counter(gen_bg_colors)),
        'arc_agi2': dict(Counter(arc_bg_colors))
    }

    # 4. オブジェクトピクセル比率
    print("4. オブジェクトピクセル比率を比較中...")
    gen_object_ratios = [s['object_pixel_ratio'] for s in generated_stats]
    arc_object_ratios = [s['object_pixel_ratio'] for s in arc_stats]

    comparison['comparisons']['object_pixel_ratio'] = {
        'generated': {
            'min': float(np.min(gen_object_ratios)),
            'max': float(np.max(gen_object_ratios)),
            'mean': float(np.mean(gen_object_ratios)),
            'median': float(np.median(gen_object_ratios)),
            'std': float(np.std(gen_object_ratios))
        },
        'arc_agi2': {
            'min': float(np.min(arc_object_ratios)),
            'max': float(np.max(arc_object_ratios)),
            'mean': float(np.mean(arc_object_ratios)),
            'median': float(np.median(arc_object_ratios)),
            'std': float(np.std(arc_object_ratios))
        }
    }

    # 比較結果をJSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n比較結果を保存しました: {output_file}")

    return comparison


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='入力グリッドを生成し、ARC-AGI2データセットと比較分析')
    parser.add_argument('--num-tasks', type=int, default=10000, help='生成するタスク数（デフォルト: 10000）')
    parser.add_argument('--output-dir', type=str, default='outputs/input_grid_comparison', help='出力ディレクトリ')
    parser.add_argument('--progress-interval', type=int, default=100, help='進捗表示の間隔')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("入力グリッド生成とARC-AGI2データセット比較分析")
    print("="*80)
    print(f"\n設定:")
    print(f"  生成タスク数: {args.num_tasks}")
    print(f"  出力ディレクトリ: {output_dir.absolute()}")
    print(f"  進捗表示間隔: {args.progress_interval}")

    start_time = time.time()

    # 1. 入力グリッドを生成
    generated_stats = generate_input_grids(
        num_tasks=args.num_tasks,
        output_dir=output_dir,
        progress_interval=args.progress_interval
    )

    # 生成された統計情報を保存
    generated_stats_file = output_dir / 'generated_statistics.json'
    with open(generated_stats_file, 'w', encoding='utf-8') as f:
        json.dump(generated_stats, f, indent=2, ensure_ascii=False)
    print(f"\n生成された統計情報を保存しました: {generated_stats_file}")

    # 2. ARC-AGI2データセットを読み込み
    arc_tasks = load_arc_agi2_tasks()
    if not arc_tasks:
        print("エラー: ARC-AGI2データセットを読み込めませんでした")
        return

    # 3. ARC-AGI2データセットから統計情報を抽出
    arc_stats = analyze_arc_agi2_tasks(
        tasks=arc_tasks,
        progress_interval=args.progress_interval
    )

    # ARC-AGI2の統計情報を保存
    arc_stats_file = output_dir / 'arc_agi2_statistics.json'
    with open(arc_stats_file, 'w', encoding='utf-8') as f:
        json.dump(arc_stats, f, indent=2, ensure_ascii=False)
    print(f"\nARC-AGI2の統計情報を保存しました: {arc_stats_file}")

    # 4. 統計を比較
    comparison_file = output_dir / 'comparison_result.json'
    comparison = compare_statistics(
        generated_stats=generated_stats,
        arc_stats=arc_stats,
        output_file=comparison_file
    )

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("分析完了")
    print(f"{'='*80}")
    print(f"総実行時間: {total_time:.1f}秒")
    print(f"生成されたグリッド数: {len(generated_stats)}")
    print(f"ARC-AGI2グリッド数: {len(arc_stats)}")
    print(f"\n結果ファイル:")
    print(f"  生成統計: {generated_stats_file}")
    print(f"  ARC-AGI2統計: {arc_stats_file}")
    print(f"  比較結果: {comparison_file}")


if __name__ == '__main__':
    main()

