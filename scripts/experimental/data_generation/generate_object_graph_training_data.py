"""
Object Graph + GNN 用訓練データ生成スクリプト

ARC-AGI2の訓練データから、オブジェクトグラフと対応するプログラムのペアを生成

使い方:
    python scripts/production/data_generation/generate_object_graph_training_data.py \\
        <output_jsonl_path> \\
        [--max-tasks N] \\
        [--max-pairs-per-task M]
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.hybrid_system.inference.object_matching.rule_based_matcher import RuleBasedObjectMatcher
from src.hybrid_system.inference.object_matching.config import ObjectMatchingConfig
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder
from src.hybrid_system.core.data_structures.task import Task
from src.core_systems.executor.core import ExecutorCore
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


def extract_object_graph_features(graph) -> Dict[str, Any]:
    """オブジェクトグラフから特徴量を抽出"""
    node_features = graph.node_features.cpu().numpy().tolist()
    edge_index = graph.edge_index.cpu().numpy().tolist()
    edge_attr = graph.edge_attr.cpu().numpy().tolist()

    return {
        'num_nodes': len(graph.nodes),
        'num_edges': graph.edge_index.shape[1],
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_types': [edge.edge_type for edge in graph.edges]
    }


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

    # オブジェクト抽出器とマッチャーを初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())
    config = ObjectMatchingConfig(enable_correspondence_detection=False)
    matcher = RuleBasedObjectMatcher(config=config)
    graph_builder = ObjectGraphBuilder()
    executor = ExecutorCore()

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
                    output_grid = grid_to_numpy(pair['output'])

                    # オブジェクト抽出
                    input_result = extractor.extract_objects_by_type(input_grid, input_image_index=0)
                    output_result = extractor.extract_objects_by_type(output_grid, input_image_index=0)

                    if not input_result.success or not output_result.success:
                        continue

                    input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
                    if not input_objects:
                        continue

                    output_objects = output_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

                    # オブジェクトグラフを構築
                    graph = graph_builder.build_graph(input_objects)

                    if graph.node_features.size(0) == 0:
                        continue

                    # グラフ特徴量を抽出
                    graph_features = extract_object_graph_features(graph)

                    # 部分プログラムを生成（オブジェクトマッチングから）
                    # Taskオブジェクトを作成
                    task = Task(
                        task_id=task_id,
                        train=[{'input': pair['input'], 'output': pair['output']}],
                        test=[],
                        program=''
                    )

                    try:
                        matching_result = matcher.match_objects(task)
                        # 部分プログラムを取得（存在する場合）
                        partial_programs = matching_result.get('all_partial_programs', [])
                        program_text = partial_programs[0] if partial_programs else ""
                    except Exception as e:
                        # マッチングに失敗した場合は空のプログラム
                        program_text = ""

                    # 訓練サンプルを作成
                    sample = {
                        'task_id': task_id,
                        'pair_index': pair_idx,
                        'graph_features': graph_features,
                        'program': program_text,
                        'input_grid_shape': list(input_grid.shape),
                        'output_grid_shape': list(output_grid.shape),
                        'num_input_objects': len(input_objects),
                        'num_output_objects': len(output_objects)
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

    parser = argparse.ArgumentParser(description='Object Graph + GNN用訓練データ生成')
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
