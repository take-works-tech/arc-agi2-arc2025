"""
以前のデータ（20251125_211943）のプログラムを使って、プログラムありの場合の入力グリッドを生成し、
統計を収集して比較するスクリプト
"""
import sys
import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import time
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# joblibエラーを防ぐため、環境変数をインポート前に設定
try:
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
except Exception:
    cpu_count = 4

os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')

from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.grid_visualizer import save_single_grid_to_png
from src.hybrid_system.inference.object_matching.object_extractor import ObjectExtractor


def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    """グリッドをNumPy配列に変換"""
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
            objects = extractor.extract_objects(grid, background_color=background_color)
            stats['num_objects'] = len(objects) if objects else 0
        except Exception:
            stats['num_objects'] = 0
    else:
        stats['num_objects'] = 0

    return stats


def load_programs_from_batch(batch_dir: Path, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    """バッチディレクトリからプログラムデータを読み込む"""
    program_file = batch_dir / "program_batch_*.json"
    program_files = list(batch_dir.glob("program_batch_*.json"))

    if not program_files:
        raise FileNotFoundError(f"プログラムファイルが見つかりません: {batch_dir}")

    # 最初のファイルを読み込む
    program_file = program_files[0]
    print(f"プログラムファイルを読み込み中: {program_file}")

    with open(program_file, 'r', encoding='utf-8') as f:
        programs = json.load(f)

    if max_tasks:
        programs = programs[:max_tasks]

    print(f"読み込んだプログラム数: {len(programs)}")
    return programs


def generate_input_grids_with_programs(
    programs: List[Dict[str, Any]],
    output_dir: Path,
    progress_interval: int = 100
) -> List[Dict[str, Any]]:
    """プログラムを使って入力グリッドを生成し、統計情報を収集

    Args:
        programs: プログラムデータのリスト
        output_dir: 出力ディレクトリ
        progress_interval: 進捗表示の間隔

    Returns:
        統計情報のリスト
    """
    print(f"\n{'='*80}")
    print(f"プログラムありの入力グリッド生成開始: {len(programs)}タスク")
    print(f"{'='*80}\n")

    statistics_list = []
    success_count = 0
    failure_count = 0
    start_time = time.time()

    # 入力グリッドを保存するためのリスト
    input_grids_for_png = []
    grid_labels = []

    for idx, program_data in enumerate(programs, 1):
        try:
            task_id = program_data.get('task_id', f'task_{idx}')
            program_code = program_data.get('program_code', '')
            grid_size = program_data.get('grid_size', {})
            grid_width = grid_size.get('width')
            grid_height = grid_size.get('height')

            if not program_code:
                failure_count += 1
                if idx % progress_interval == 0:
                    print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - プログラムコードなし")
                continue

            if grid_width is None or grid_height is None:
                failure_count += 1
                if idx % progress_interval == 0:
                    print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - グリッドサイズなし")
                continue

            # プログラムコードを使って入力グリッドを生成
            # program_codeを渡すことで、プログラムありの場合の入力グリッド生成が行われる
            try:
                _, input_grid, output_grid, _ = core_executor_main(
                    nodes=None,  # nodesはNoneで、program_codeを使用
                    grid_width=grid_width,
                    grid_height=grid_height,
                    program_code=program_code,  # プログラムコードを直接渡す
                    is_first_pair=True,
                    enable_replacement=True
                )
            except Exception as e:
                failure_count += 1
                if idx % progress_interval == 0:
                    print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - エラー: {str(e)[:50]}")
                continue

            if input_grid is None:
                failure_count += 1
                if idx % progress_interval == 0:
                    print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - input_gridがNone")
                continue

            # 単色グリッドチェック
            unique_colors = np.unique(input_grid)
            if len(unique_colors) <= 1:
                failure_count += 1
                if idx % progress_interval == 0:
                    print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - 単色グリッド")
                continue

            # 統計情報を抽出
            stats = extract_grid_statistics(input_grid, extract_objects=False)
            stats['task_id'] = task_id
            stats['generated'] = True
            stats['program_code'] = program_code[:100]  # 最初の100文字だけ保存
            statistics_list.append(stats)

            # PNG保存用に追加
            input_grids_for_png.append(input_grid)
            grid_labels.append(f"{task_id}_input")

            success_count += 1

            # 進捗表示
            if idx % progress_interval == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / idx * (len(programs) - idx) if idx > 0 else 0
                print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - "
                      f"経過時間: {elapsed:.1f}秒, 残り時間: {remaining:.1f}秒")

        except Exception as e:
            failure_count += 1
            if idx % progress_interval == 0:
                print(f"  進捗: {idx}/{len(programs)} (成功: {success_count}, 失敗: {failure_count}) - 例外: {str(e)[:50]}")

    elapsed = time.time() - start_time
    print(f"\n生成完了: 成功={success_count}, 失敗={failure_count}, 総時間={elapsed:.1f}秒")

    # PNGを保存（個別に保存、最大100個まで）
    if input_grids_for_png:
        png_dir = output_dir / "grids_png"
        png_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for i, (grid, label) in enumerate(zip(input_grids_for_png[:100], grid_labels[:100])):
            png_path = png_dir / f"{label}.png"
            if save_single_grid_to_png(grid, str(png_path), title=label):
                saved_count += 1
        print(f"PNGファイルを保存: {png_dir} ({saved_count}グリッド)")

    return statistics_list


def compare_with_arc_agi2(generated_stats: List[Dict], output_dir: Path):
    """生成された統計とARC-AGI2データセットを比較"""
    # ARC-AGI2統計を読み込み
    arc_agi2_file = Path("outputs/input_grid_comparison_after_condition4_removal/arc_agi2_statistics.json")
    if not arc_agi2_file.exists():
        print(f"警告: ARC-AGI2統計ファイルが見つかりません: {arc_agi2_file}")
        return

    print("\nARC-AGI2統計を読み込み中...")
    with open(arc_agi2_file, 'r', encoding='utf-8') as f:
        arc_agi2_stats = json.load(f)

    print(f"ARC-AGI2統計: {len(arc_agi2_stats)}タスク")
    print(f"生成統計: {len(generated_stats)}タスク")

    # 比較結果を計算
    comparison = {
        'generated_count': len(generated_stats),
        'arc_count': len(arc_agi2_stats),
        'comparison_timestamp': datetime.now().isoformat(),
        'comparisons': {}
    }

    # グリッドサイズの比較
    gen_widths = [s['width'] for s in generated_stats]
    gen_heights = [s['height'] for s in generated_stats]
    gen_areas = [s['area'] for s in generated_stats]

    arc_widths = [s['width'] for s in arc_agi2_stats if 'width' in s]
    arc_heights = [s['height'] for s in arc_agi2_stats if 'height' in s]
    arc_areas = [s['area'] for s in arc_agi2_stats if 'area' in s]

    comparison['comparisons']['grid_size'] = {
        'generated': {
            'width': {
                'min': min(gen_widths) if gen_widths else 0,
                'max': max(gen_widths) if gen_widths else 0,
                'mean': np.mean(gen_widths) if gen_widths else 0,
                'median': np.median(gen_widths) if gen_widths else 0,
                'std': np.std(gen_widths) if gen_widths else 0
            },
            'height': {
                'min': min(gen_heights) if gen_heights else 0,
                'max': max(gen_heights) if gen_heights else 0,
                'mean': np.mean(gen_heights) if gen_heights else 0,
                'median': np.median(gen_heights) if gen_heights else 0,
                'std': np.std(gen_heights) if gen_heights else 0
            },
            'area': {
                'min': min(gen_areas) if gen_areas else 0,
                'max': max(gen_areas) if gen_areas else 0,
                'mean': np.mean(gen_areas) if gen_areas else 0,
                'median': np.median(gen_areas) if gen_areas else 0,
                'std': np.std(gen_areas) if gen_areas else 0
            }
        },
        'arc_agi2': {
            'width': {
                'min': min(arc_widths) if arc_widths else 0,
                'max': max(arc_widths) if arc_widths else 0,
                'mean': np.mean(arc_widths) if arc_widths else 0,
                'median': np.median(arc_widths) if arc_widths else 0,
                'std': np.std(arc_widths) if arc_widths else 0
            },
            'height': {
                'min': min(arc_heights) if arc_heights else 0,
                'max': max(arc_heights) if arc_heights else 0,
                'mean': np.mean(arc_heights) if arc_heights else 0,
                'median': np.median(arc_heights) if arc_heights else 0,
                'std': np.std(arc_heights) if arc_heights else 0
            },
            'area': {
                'min': min(arc_areas) if arc_areas else 0,
                'max': max(arc_areas) if arc_areas else 0,
                'mean': np.mean(arc_areas) if arc_areas else 0,
                'median': np.median(arc_areas) if arc_areas else 0,
                'std': np.std(arc_areas) if arc_areas else 0
            }
        }
    }

    # 色数の比較
    gen_total_colors = [s['num_colors_total'] for s in generated_stats]
    gen_object_colors = [s['num_colors_object'] for s in generated_stats]

    arc_total_colors = [s['num_colors_total'] for s in arc_agi2_stats if 'num_colors_total' in s]
    arc_object_colors = [s['num_colors_object'] for s in arc_agi2_stats if 'num_colors_object' in s]

    comparison['comparisons']['color_count'] = {
        'generated': {
            'total': {
                'min': min(gen_total_colors) if gen_total_colors else 0,
                'max': max(gen_total_colors) if gen_total_colors else 0,
                'mean': np.mean(gen_total_colors) if gen_total_colors else 0,
                'median': np.median(gen_total_colors) if gen_total_colors else 0,
                'distribution': dict(Counter(gen_total_colors))
            },
            'object': {
                'min': min(gen_object_colors) if gen_object_colors else 0,
                'max': max(gen_object_colors) if gen_object_colors else 0,
                'mean': np.mean(gen_object_colors) if gen_object_colors else 0,
                'median': np.median(gen_object_colors) if gen_object_colors else 0,
                'distribution': dict(Counter(gen_object_colors))
            }
        },
        'arc_agi2': {
            'total': {
                'min': min(arc_total_colors) if arc_total_colors else 0,
                'max': max(arc_total_colors) if arc_total_colors else 0,
                'mean': np.mean(arc_total_colors) if arc_total_colors else 0,
                'median': np.median(arc_total_colors) if arc_total_colors else 0,
                'distribution': dict(Counter(arc_total_colors))
            },
            'object': {
                'min': min(arc_object_colors) if arc_object_colors else 0,
                'max': max(arc_object_colors) if arc_object_colors else 0,
                'mean': np.mean(arc_object_colors) if arc_object_colors else 0,
                'median': np.median(arc_object_colors) if arc_object_colors else 0,
                'distribution': dict(Counter(arc_object_colors))
            }
        }
    }

    # 背景色の比較
    gen_bg_colors = [s['background_color'] for s in generated_stats]
    arc_bg_colors = [s['background_color'] for s in arc_agi2_stats if 'background_color' in s]

    comparison['comparisons']['background_color'] = {
        'generated': dict(Counter(gen_bg_colors)),
        'arc_agi2': dict(Counter(arc_bg_colors))
    }

    # オブジェクトピクセル比率の比較
    gen_ratios = [s['object_pixel_ratio'] for s in generated_stats]
    arc_ratios = [s['object_pixel_ratio'] for s in arc_agi2_stats if 'object_pixel_ratio' in s]

    comparison['comparisons']['object_pixel_ratio'] = {
        'generated': {
            'min': min(gen_ratios) if gen_ratios else 0,
            'max': max(gen_ratios) if gen_ratios else 0,
            'mean': np.mean(gen_ratios) if gen_ratios else 0,
            'median': np.median(gen_ratios) if gen_ratios else 0,
            'std': np.std(gen_ratios) if gen_ratios else 0
        },
        'arc_agi2': {
            'min': min(arc_ratios) if arc_ratios else 0,
            'max': max(arc_ratios) if arc_ratios else 0,
            'mean': np.mean(arc_ratios) if arc_ratios else 0,
            'median': np.median(arc_ratios) if arc_ratios else 0,
            'std': np.std(arc_ratios) if arc_ratios else 0
        }
    }

    # 比較結果を保存
    comparison_file = output_dir / "comparison_result.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n比較結果を保存: {comparison_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='以前のプログラムを使って入力グリッドを生成')
    parser.add_argument('--batch-dir', type=str,
                       default='outputs/rule_based/20251125_211943/batch_0000',
                       help='バッチディレクトリのパス')
    parser.add_argument('--num-tasks', type=int, default=1000,
                       help='生成するタスク数（デフォルト: 1000）')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/input_grid_comparison_with_programs',
                       help='出力ディレクトリ')
    parser.add_argument('--progress-interval', type=int, default=100,
                       help='進捗表示の間隔（デフォルト: 100）')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # プログラムを読み込み
    batch_dir = Path(args.batch_dir)
    programs = load_programs_from_batch(batch_dir, max_tasks=args.num_tasks)

    # 入力グリッドを生成
    statistics = generate_input_grids_with_programs(
        programs=programs,
        output_dir=output_dir,
        progress_interval=args.progress_interval
    )

    # 統計を保存
    stats_file = output_dir / "generated_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"統計を保存: {stats_file}")

    # ARC-AGI2と比較
    compare_with_arc_agi2(statistics, output_dir)

    print("\n完了!")


if __name__ == "__main__":
    main()
