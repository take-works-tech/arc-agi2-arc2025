"""
検証スクリプト: normalモードとcopyモードを強制実行して検証

プログラムなしで、normalモードまたはcopyモードを強制実行して、
各モードでのグリッド生成を検証し、ARC-AGI2統計と比較します。
"""
import sys
import os
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.input_grid_generator.background_decider import (
    decide_background_color,
    generate_background_grid_pattern
)
from src.data_systems.generator.input_grid_generator.grid_size_decider import decide_grid_size
from src.data_systems.generator.program_executor.node_validator_output import validate_nodes_and_adjust_objects
from src.data_systems.generator.program_executor.core_executor import CoreExecutor
from src.data_systems.generator.input_grid_generator.managers import build_grid
from src.data_systems.generator.grid_visualizer import save_single_grid_to_png
from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType
from collections import Counter
import json

def load_arc_agi2_statistics(max_tasks=None):
    """ARC-AGI2統計を読み込む（キャッシュまたは計算）"""
    # 既存のcompare_statistics_with_arc_agi2.pyからインポート
    import sys
    sys.path.insert(0, str(project_root / "scripts" / "analysis"))
    from compare_statistics_with_arc_agi2 import analyze_arc_agi2_statistics
    return analyze_arc_agi2_statistics(max_tasks=max_tasks)

def extract_statistics(grid, config):
    """グリッドから統計情報を抽出（compare_statistics_with_arc_agi2.pyと同じ形式）"""
    extractor = IntegratedObjectExtractor(config)
    result = extractor.extract_objects_by_type(grid, input_image_index=0)

    # SINGLE_COLOR_4WAYのみを使用（compare_statistics_with_arc_agi2.pyと同じ）
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
            'colors': set()
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

    # 色数
    colors = set(obj.color for obj in objects)

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
        'colors': colors
    }

def verify_mode(mode: str, num_samples: int = 20, output_dir: str = None, compare_with_arc=True):
    """指定されたモードでグリッド生成を検証

    Args:
        mode: 'normal' または 'copy'
        num_samples: 生成するグリッド数
        output_dir: 出力ディレクトリ
        compare_with_arc: ARC-AGI2統計と比較するか
    """
    if output_dir is None:
        output_dir = f"outputs/verification/{mode}_mode"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"検証開始: {mode.upper()}モード")
    print(f"{'='*80}\n")

    executor = CoreExecutor()
    config = ExtractionConfig()

    all_stats = []
    success_count = 0

    for i in range(num_samples):
        try:
            print(f"\n--- サンプル {i+1}/{num_samples} ---")

            # グリッドサイズを決定
            grid_width, grid_height = decide_grid_size()
            print(f"グリッドサイズ: {grid_width}x{grid_height}")

            # 背景色を決定
            background_color = decide_background_color()
            print(f"背景色: {background_color}")

            # 背景パターンを生成
            background_grid_pattern = generate_background_grid_pattern(
                background_color=background_color,
                grid_width=grid_width,
                grid_height=grid_height
            )

            # オブジェクトリスト（空で開始）
            all_objects = []

            # プログラムなしで、強制的に指定されたモードで生成
            nodes, updated_objects, _, _, _ = validate_nodes_and_adjust_objects(
                nodes=None,  # プログラムなし
                all_objects=all_objects,
                background_color=background_color,
                grid_width=grid_width,
                grid_height=grid_height,
                background_grid_pattern=background_grid_pattern,
                executor=executor,
                all_commands=set(),  # コマンドなし
                program_code=None,
                force_mode=mode  # モードを強制
            )

            if updated_objects is None or len(updated_objects) == 0:
                print(f"  [警告] サンプル {i+1}: オブジェクトが生成されませんでした")
                continue

            # グリッドを構築
            grid = build_grid(
                width=grid_width,
                height=grid_height,
                background_grid_pattern=background_grid_pattern,
                objects=updated_objects,
                min_spacing=0
            )
            grid_array = np.array(grid, dtype=int)

            # 統計情報を抽出
            stats = extract_statistics(grid_array, config)
            all_stats.append(stats)

            # 情報を表示
            print(f"  オブジェクト数: {stats['num_objects']}")
            print(f"  1ピクセルオブジェクト数: {stats['num_single_pixel_objects']}")
            print(f"  オブジェクト密度: {stats['object_pixel_ratio']:.3f}")
            print(f"  色数: {len(stats['colors'])}")
            if stats['object_areas']:
                avg_area = sum(stats['object_areas']) / len(stats['object_areas'])
                print(f"  平均オブジェクト面積: {avg_area:.2f}")

            # グリッドを保存
            output_path = os.path.join(output_dir, f"grid_{i+1:03d}_{grid_width}x{grid_height}_objects{stats['num_objects']}.png")
            save_single_grid_to_png(grid_array, output_path)
            print(f"  保存: {output_path}")

            success_count += 1

        except Exception as e:
            print(f"  [エラー] サンプル {i+1}: エラー - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 統計サマリー
    if all_stats:
        print(f"\n{'='*80}")
        print(f"統計サマリー: {mode.upper()}モード")
        print(f"{'='*80}\n")

        total_objects = sum(s['num_objects'] for s in all_stats)
        avg_objects = total_objects / len(all_stats)
        total_single_pixel = sum(s['num_single_pixel_objects'] for s in all_stats)
        avg_single_pixel = total_single_pixel / len(all_stats)
        avg_single_pixel_ratio = sum(s['single_pixel_ratio'] for s in all_stats) / len(all_stats)
        avg_density = sum(s['object_pixel_ratio'] for s in all_stats) / len(all_stats)
        all_areas = [area for s in all_stats for area in s['object_areas']]
        avg_object_area = sum(all_areas) / len(all_areas) if all_areas else 0
        all_sizes = [size for s in all_stats for size in s['object_sizes']]
        avg_width = sum(w for w, h in all_sizes) / len(all_sizes) if all_sizes else 0
        avg_height = sum(h for w, h in all_sizes) / len(all_sizes) if all_sizes else 0
        all_colors = set()
        for s in all_stats:
            all_colors.update(s['colors'])
        avg_grid_width = sum(s['grid_width'] for s in all_stats) / len(all_stats)
        avg_grid_height = sum(s['grid_height'] for s in all_stats) / len(all_stats)
        avg_grid_area = sum(s['grid_area'] for s in all_stats) / len(all_stats)

        print(f"成功サンプル数: {success_count}/{num_samples}")
        print(f"\n【オブジェクト統計】")
        print(f"  平均オブジェクト数: {avg_objects:.2f}")
        print(f"  平均1ピクセルオブジェクト数: {avg_single_pixel:.2f}")
        print(f"  平均1ピクセルオブジェクト比率: {avg_single_pixel_ratio:.3f}")
        print(f"\n【密度・サイズ統計】")
        print(f"  平均オブジェクト密度: {avg_density:.3f}")
        print(f"  平均オブジェクト面積: {avg_object_area:.2f}")
        print(f"  平均オブジェクト幅: {avg_width:.2f}")
        print(f"  平均オブジェクト高さ: {avg_height:.2f}")
        print(f"\n【グリッド統計】")
        print(f"  平均グリッド幅: {avg_grid_width:.2f}")
        print(f"  平均グリッド高さ: {avg_grid_height:.2f}")
        print(f"  平均グリッド面積: {avg_grid_area:.2f}")
        print(f"\n【色統計】")
        print(f"  使用された色数: {len(all_colors)}")
        print(f"  出力ディレクトリ: {output_dir}")

        # ARC-AGI2統計と比較
        if compare_with_arc:
            try:
                arc_agi2_stats = load_arc_agi2_statistics(max_tasks=1000)
                if arc_agi2_stats:
                    mode_stats_summary = {
                        'mode': mode,
                        'avg_objects': avg_objects,
                        'avg_single_pixel': avg_single_pixel,
                        'avg_single_pixel_ratio': avg_single_pixel_ratio,
                        'avg_density': avg_density,
                        'avg_object_area': avg_object_area,
                        'avg_width': avg_width,
                        'avg_height': avg_height,
                        'avg_grid_area': avg_grid_area,
                        'num_colors': len(all_colors)
                    }
                    compare_with_arc_agi2(mode_stats_summary, arc_agi2_stats)
            except Exception as e:
                print(f"\n[警告] ARC-AGI2統計との比較中にエラーが発生しました: {e}")

    return all_stats

def compare_with_arc_agi2(mode_stats, arc_agi2_stats):
    """モード統計とARC-AGI2統計を比較"""
    if not mode_stats or not arc_agi2_stats:
        return

    print(f"\n{'='*80}")
    print(f"ARC-AGI2統計との比較: {mode_stats.get('mode', 'UNKNOWN').upper()}モード")
    print(f"{'='*80}\n")

    # オブジェクト数の比較
    if 'avg_objects' in mode_stats and 'num_single_pixel_objects' in arc_agi2_stats:
        gen = mode_stats['avg_objects']
        agi2 = arc_agi2_stats['num_single_pixel_objects']['mean']
        diff = gen - agi2
        diff_pct = (diff / agi2 * 100) if agi2 > 0 else 0
        print(f"【オブジェクト数】")
        print(f"  ARC-AGI2:  {agi2:.2f}")
        print(f"  生成:      {gen:.2f}")
        print(f"  差:        {diff:+.2f} ({diff_pct:+.1f}%)")

    # 密度の比較
    if 'avg_density' in mode_stats and 'object_pixel_ratio' in arc_agi2_stats:
        gen = mode_stats['avg_density']
        agi2 = arc_agi2_stats['object_pixel_ratio']['mean']
        diff = gen - agi2
        diff_pct = (diff / agi2 * 100) if agi2 > 0 else 0
        print(f"\n【オブジェクト密度】")
        print(f"  ARC-AGI2:  {agi2:.3f}")
        print(f"  生成:      {gen:.3f}")
        print(f"  差:        {diff:+.3f} ({diff_pct:+.1f}%)")

    # グリッドサイズの比較
    if 'avg_grid_area' in mode_stats and 'grid_area' in arc_agi2_stats:
        gen = mode_stats['avg_grid_area']
        agi2 = arc_agi2_stats['grid_area']['mean']
        diff = gen - agi2
        diff_pct = (diff / agi2 * 100) if agi2 > 0 else 0
        print(f"\n【グリッド面積】")
        print(f"  ARC-AGI2:  {agi2:.2f}")
        print(f"  生成:      {gen:.2f}")
        print(f"  差:        {diff:+.2f} ({diff_pct:+.1f}%)")

    # オブジェクトサイズの比較
    if 'avg_object_area' in mode_stats and 'object_areas' in arc_agi2_stats:
        gen = mode_stats['avg_object_area']
        agi2 = arc_agi2_stats['object_areas']['mean']
        diff = gen - agi2
        diff_pct = (diff / agi2 * 100) if agi2 > 0 else 0
        print(f"\n【オブジェクト面積】")
        print(f"  ARC-AGI2:  {agi2:.2f}")
        print(f"  生成:      {gen:.2f}")
        print(f"  差:        {diff:+.2f} ({diff_pct:+.1f}%)")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="normal/copyモードの検証")
    parser.add_argument("--mode", type=str, choices=['normal', 'copy', 'both'],
                       default='both', help="検証するモード (default: both)")
    parser.add_argument("--num-samples", type=int, default=20,
                       help="各モードで生成するサンプル数 (default: 20)")
    parser.add_argument("--no-compare", action="store_true",
                       help="ARC-AGI2統計との比較をスキップ")

    args = parser.parse_args()

    compare_flag = not args.no_compare

    if args.mode == 'both':
        # normalモードを検証
        verify_mode('normal', num_samples=args.num_samples, compare_with_arc=compare_flag)

        # copyモードを検証
        verify_mode('copy', num_samples=args.num_samples, compare_with_arc=compare_flag)
    else:
        verify_mode(args.mode, num_samples=args.num_samples, compare_with_arc=compare_flag)

    print(f"\n{'='*80}")
    print("検証完了")
    print(f"{'='*80}\n")
