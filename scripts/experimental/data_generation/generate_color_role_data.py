"""
Color Role Classifier 用訓練データ生成スクリプト

ARC-AGI2の訓練データから、グリッドと色役割ラベルを生成

使い方:
    python scripts/production/data_generation/generate_color_role_data.py \\
        <output_jsonl_path> \\
        [--max-tasks N] \\
        [--max-pairs-per-task M]
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.hybrid_system.models.program_synthesis.color_role_classifier import ColorFeatureExtractor
from src.data_systems.data_models.base import ObjectType


def load_agi2_tasks(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
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


def infer_color_roles_simple(grid: np.ndarray, objects: List) -> Dict[int, str]:
    """色の役割を推論（簡易版）"""
    color_roles = {}

    # 各色のピクセル数をカウント
    color_counts = Counter(grid.flatten())
    total_pixels = grid.size

    # 最も多い色を背景色と仮定
    if color_counts:
        background_color = color_counts.most_common(1)[0][0]
        color_roles[background_color] = 'background'

    # オブジェクトに含まれる色を前景色と仮定
    for obj in objects:
        obj_color = obj.dominant_color
        if obj_color not in color_roles:
            # オブジェクトの色が背景色でない場合、前景色
            if obj_color != background_color:
                color_roles[obj_color] = 'foreground'
            else:
                color_roles[obj_color] = 'background'

    # その他の色は構造色として分類
    for color in color_counts.keys():
        if color not in color_roles:
            # ピクセル数が少ない場合は構造色
            if color_counts[color] / total_pixels < 0.1:
                color_roles[color] = 'structural'
            else:
                color_roles[color] = 'foreground'

    return color_roles


def generate_training_data(
    output_path: str,
    max_tasks: Optional[int] = None,
    max_pairs_per_task: Optional[int] = None
) -> None:
    """訓練データを生成"""
    # タスクを読み込み
    tasks = load_agi2_tasks(max_tasks=max_tasks)
    if not tasks:
        print("タスクが見つかりませんでした")
        return

    print(f"読み込んだタスク数: {len(tasks)}")

    # オブジェクト抽出器と色特徴量抽出器を初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())
    color_extractor = ColorFeatureExtractor()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_pairs_processed = 0

    with output_file.open('w', encoding='utf-8') as f:
        for task_idx, task_data in enumerate(tasks):
            task_id = task_data['task_id']
            train_pairs = task_data.get('train', [])

            if max_pairs_per_task:
                train_pairs = train_pairs[:max_pairs_per_task]

            print(f"\nタスク {task_idx + 1}/{len(tasks)}: {task_id} ({len(train_pairs)}ペア)")

            for pair_idx, pair in enumerate(train_pairs):
                try:
                    input_grid = grid_to_numpy(pair['input'])

                    # オブジェクト抽出
                    input_result = extractor.extract_objects_by_type(input_grid, input_image_index=0)

                    if not input_result.success:
                        continue

                    input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

                    # 色特徴量を抽出
                    color_features = color_extractor.extract_color_features(input_grid, input_objects)

                    # 色の役割を推論
                    color_roles = infer_color_roles_simple(input_grid, input_objects)

                    # 各色に対してサンプルを生成
                    unique_colors = np.unique(input_grid)
                    for color in unique_colors:
                        color_int = int(color)

                        if color_int not in color_features:
                            continue

                        features = color_features[color_int]
                        # numpy配列をリストに変換
                        features_list = features.tolist() if isinstance(features, np.ndarray) else list(features)
                        role = color_roles.get(color_int, 'unknown')

                        # 訓練サンプルを作成
                        sample = {
                            'task_id': task_id,
                            'pair_index': pair_idx,
                            'color': color_int,
                            'color_features': features_list,
                            'color_role': role,
                            'grid_shape': list(input_grid.shape),
                            'num_objects': len(input_objects)
                        }

                        # JSON Lines形式で書き込み
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        total_samples += 1

                    total_pairs_processed += 1

                except Exception as e:
                    print(f"  エラー (ペア {pair_idx}): {e}")
                    continue

            if (task_idx + 1) % 10 == 0:
                print(f"  進捗: {total_samples}サンプル生成済み")

    print(f"\n{'='*60}")
    print(f"訓練データ生成完了")
    print(f"{'='*60}")
    print(f"処理したタスク数: {len(tasks)}")
    print(f"処理したペア数: {total_pairs_processed}")
    print(f"生成したサンプル数: {total_samples}")
    print(f"出力ファイル: {output_path}")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Color Role Classifier用訓練データ生成')
    parser.add_argument('output_path', type=str, help='出力JSONLファイルのパス')
    parser.add_argument('--max-tasks', type=int, default=None, help='最大タスク数')
    parser.add_argument('--max-pairs-per-task', type=int, default=None, help='タスクあたりの最大ペア数')

    args = parser.parse_args()

    generate_training_data(
        output_path=args.output_path,
        max_tasks=args.max_tasks,
        max_pairs_per_task=args.max_pairs_per_task
    )


if __name__ == "__main__":
    main()
