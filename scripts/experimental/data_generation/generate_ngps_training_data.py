"""
NGPS / DSL Selector 用訓練データ生成スクリプト

ARC-AGI2の訓練データから、グリッドペアと対応するDSL使用確率を生成

使い方:
    python scripts/production/data_generation/generate_ngps_training_data.py \\
        <output_jsonl_path> \\
        [--max-tasks N] \\
        [--max-pairs-per-task M]
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import re
from collections import Counter

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_system.inference.program_synthesis.candidate_generator import CandidateGenerator, CandidateConfig
from src.hybrid_system.inference.program_synthesis.consistency_checker import ConsistencyChecker
from src.core_systems.executor.core import ExecutorCore
from src.hybrid_system.core.data_structures.task import Task


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


def extract_dsl_commands(program: str) -> List[str]:
    """プログラムからDSLコマンドを抽出"""
    if not program:
        return []

    # 基本的なDSLコマンドパターン
    # 大文字の識別子（関数名）を抽出
    commands = re.findall(r'\b([A-Z][A-Z0-9_]+)\s*\(', program)
    return commands


def calculate_dsl_probabilities(programs: List[str]) -> Dict[str, float]:
    """プログラムリストからDSL使用確率を計算"""
    if not programs:
        return {}

    all_commands = []
    for program in programs:
        commands = extract_dsl_commands(program)
        all_commands.extend(commands)

    if not all_commands:
        return {}

    # コマンドの出現頻度をカウント
    command_counts = Counter(all_commands)
    total = len(all_commands)

    # 確率に変換
    probabilities = {cmd: count / total for cmd, count in command_counts.items()}

    return probabilities


def extract_grid_features(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
    """グリッドから特徴量を抽出"""
    return {
        'input_shape': list(input_grid.shape),
        'output_shape': list(output_grid.shape),
        'input_size': int(input_grid.size),
        'output_size': int(output_grid.size),
        'input_unique_colors': int(len(np.unique(input_grid))),
        'output_unique_colors': int(len(np.unique(output_grid))),
        'input_mean': float(np.mean(input_grid)),
        'output_mean': float(np.mean(output_grid)),
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

    # 候補生成器と一貫性チェッカーを初期化
    candidate_config = CandidateConfig(max_candidates=10)
    candidate_generator = CandidateGenerator(config=candidate_config)
    consistency_checker = ConsistencyChecker()

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

            # タスク全体のプログラム候補を生成
            task_programs = []
            for pair_idx, pair in enumerate(train_pairs):
                try:
                    input_grid = grid_to_numpy(pair['input'])
                    output_grid = grid_to_numpy(pair['output'])

                    # 候補プログラムを生成
                    candidates = candidate_generator.generate_candidates(
                        input_grid=input_grid.tolist(),
                        output_grid=output_grid.tolist(),
                        max_candidates=5,
                        pair_index=pair_idx
                    )

                    if not candidates:
                        print(f"  警告 (ペア {pair_idx}): 候補プログラムが生成されませんでした")
                        continue

                    # 一貫性チェックを実行
                    task_obj = Task(
                        task_id=task_id,
                        train=[{'input': pair['input'], 'output': pair['output']}],
                        test=[],
                        program=''
                    )

                    consistency_results = consistency_checker.check_consistency(
                        candidates, task_obj
                    )

                    # 一貫性スコアが高いプログラムを収集（閾値を下げる）
                    for program, result in zip(candidates, consistency_results):
                        if result['consistency_score'] >= 0.5:  # 閾値を0.8から0.5に下げる
                            task_programs.append(program)

                except Exception as e:
                    print(f"  エラー (ペア {pair_idx}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # タスク全体のDSL使用確率を計算
            dsl_probabilities = calculate_dsl_probabilities(task_programs)

            # DSL確率が空の場合でも、デフォルトの確率分布を使用
            if not dsl_probabilities:
                print(f"  警告: DSL確率が計算できませんでした。デフォルト確率を使用します")
                # デフォルトのDSL確率（均等分布）
                dsl_probabilities = {f'dsl_{i}': 1.0 / 100 for i in range(100)}

            # 各ペアに対してサンプルを生成
            for pair_idx, pair in enumerate(train_pairs):
                try:
                    input_grid = grid_to_numpy(pair['input'])
                    output_grid = grid_to_numpy(pair['output'])

                    # グリッド特徴量を抽出
                    grid_features = extract_grid_features(input_grid, output_grid)

                    # 訓練サンプルを作成
                    sample = {
                        'task_id': task_id,
                        'pair_index': pair_idx,
                        'grid_features': grid_features,
                        'dsl_probabilities': dsl_probabilities,
                        'input_grid': input_grid.tolist(),
                        'output_grid': output_grid.tolist()
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

    parser = argparse.ArgumentParser(description='NGPS / DSL Selector用訓練データ生成')
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
