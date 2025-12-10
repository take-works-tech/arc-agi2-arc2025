"""
Relation Classifier 用訓練データ生成スクリプト

ARC-AGI2の訓練データから、オブジェクトペアと関係性ラベルを生成

使い方:
    python scripts/production/data_generation/generate_relation_classifier_data.py \\
        <output_jsonl_path> \\
        [--max-tasks N] \\
        [--max-pairs-per-task M]
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder
from src.data_systems.data_models.core.object import Object
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


def classify_relation(obj1: Object, obj2: Object) -> List[str]:
    """2つのオブジェクト間の関係を分類"""
    relations = []

    # 位置関係
    center1_x, center1_y = obj1.center_x, obj1.center_y
    center2_x, center2_y = obj2.center_x, obj2.center_y

    # 左右関係
    if center1_x < center2_x:
        relations.append('spatial_left')
    elif center1_x > center2_x:
        relations.append('spatial_right')

    # 上下関係
    if center1_y < center2_y:
        relations.append('spatial_up')
    elif center1_y > center2_y:
        relations.append('spatial_down')

    # 対称性（簡易版）
    # X軸対称
    if abs(center1_y - center2_y) < 2 and abs(center1_x + center2_x) < 10:
        relations.append('mirror_x')

    # Y軸対称
    if abs(center1_x - center2_x) < 2 and abs(center1_y + center2_y) < 10:
        relations.append('mirror_y')

    # 包含関係（簡易版）
    bbox1 = obj1.bbox
    bbox2 = obj2.bbox
    if (bbox1[0] <= bbox2[0] and bbox1[2] >= bbox2[2] and
        bbox1[1] <= bbox2[1] and bbox1[3] >= bbox2[3]):
        relations.append('contain')

    # 繰り返しパターン（簡易版：同じサイズ・色）
    if (abs(obj1.bbox_width - obj2.bbox_width) < 2 and
        abs(obj1.bbox_height - obj2.bbox_height) < 2 and
        obj1.dominant_color == obj2.dominant_color):
        relations.append('repeat')

    return relations


def extract_object_features(obj: Object) -> List[float]:
    """オブジェクトから特徴量を抽出"""
    # areaはpixelsの長さから計算
    area = len(obj.pixels) if obj.pixels else 0
    bbox_area = obj.bbox_area if hasattr(obj, 'bbox_area') else (obj.bbox_width * obj.bbox_height)

    return [
        float(obj.center_x),
        float(obj.center_y),
        float(obj.bbox_width),
        float(obj.bbox_height),
        float(obj.dominant_color),
        float(area),
        float(bbox_area),
        float(obj.hole_count) if hasattr(obj, 'hole_count') else 0.0,
    ]


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

    # オブジェクト抽出器を初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())
    graph_builder = ObjectGraphBuilder()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_pairs_processed = 0

    # 関係タイプの定義
    relation_types = [
        'spatial_left',
        'spatial_right',
        'spatial_up',
        'spatial_down',
        'mirror_x',
        'mirror_y',
        'repeat',
        'contain'
    ]

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
                    if len(input_objects) < 2:
                        print(f"  警告 (ペア {pair_idx}): オブジェクトが2つ未満です（{len(input_objects)}個）")
                        continue

                    # オブジェクトグラフを構築
                    try:
                        graph = graph_builder.build_graph(input_objects)
                    except Exception as e:
                        print(f"  エラー (ペア {pair_idx}): グラフ構築失敗: {e}")
                        continue

                    # 各エッジ（オブジェクトペア）に対してサンプルを生成
                    edge_count = 0
                    for edge in graph.edges:
                        obj1_idx = edge.source_idx
                        obj2_idx = edge.target_idx

                        if obj1_idx >= len(input_objects) or obj2_idx >= len(input_objects):
                            continue

                        obj1 = input_objects[obj1_idx]
                        obj2 = input_objects[obj2_idx]

                        # 関係を分類
                        try:
                            relations = classify_relation(obj1, obj2)
                        except Exception as e:
                            print(f"  エラー (ペア {pair_idx}, エッジ {edge_count}): 関係分類失敗: {e}")
                            continue

                        # 関係ラベルをベクトル化
                        relation_labels = [1.0 if rel_type in relations else 0.0 for rel_type in relation_types]

                        # オブジェクト特徴量を抽出
                        obj1_features = extract_object_features(obj1)
                        obj2_features = extract_object_features(obj2)

                        # 相対特徴量
                        relative_features = [
                            float(obj2.center_x - obj1.center_x),
                            float(obj2.center_y - obj1.center_y),
                            float(obj2.bbox_width - obj1.bbox_width),
                            float(obj2.bbox_height - obj1.bbox_height),
                        ]

                        # 訓練サンプルを作成
                        sample = {
                            'task_id': task_id,
                            'pair_index': pair_idx,
                            'obj1_features': obj1_features,
                            'obj2_features': obj2_features,
                            'relative_features': relative_features,
                            'relation_labels': relation_labels,
                            'relation_types': relation_types,
                            'edge_type': edge.edge_type
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

    parser = argparse.ArgumentParser(description='Relation Classifier用訓練データ生成')
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
