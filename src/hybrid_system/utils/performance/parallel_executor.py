"""
並列実行システム

候補生成、一貫性チェック、オブジェクトマッチングを並列実行
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class ParallelConfig:
    """並列処理設定"""
    enable_parallel: bool = True
    max_workers: int = 4
    use_process_pool: bool = False  # True: ProcessPoolExecutor, False: ThreadPoolExecutor
    timeout: float = 30.0  # 各タスクのタイムアウト（秒）


class ParallelExecutor:
    """並列実行クラス"""

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初期化

        Args:
            config: 並列処理設定
        """
        self.config = config or ParallelConfig()
        self.max_workers = min(self.config.max_workers, mp.cpu_count())

    def execute_parallel(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        description: str = "並列実行"
    ) -> List[Any]:
        """
        複数のタスクを並列実行

        Args:
            tasks: タスクのリスト（関数、引数タプル、キーワード引数辞書）
            description: 実行の説明（ログ用）

        Returns:
            実行結果のリスト（順序は保証されない）
        """
        if not self.config.enable_parallel or len(tasks) <= 1:
            # 並列処理が無効またはタスクが1つ以下の場合、順次実行
            results = []
            for func, args, kwargs in tasks:
                try:
                    result = func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"{description} エラー: {e}")
                    results.append(None)
            return results

        # 並列実行
        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        results = []
        completed_count = 0

        with executor_class(max_workers=self.max_workers) as executor:
            # タスクを送信
            future_to_task = {}
            for i, (func, args, kwargs) in enumerate(tasks):
                future = executor.submit(func, *args, **kwargs)
                future_to_task[future] = (i, func, args, kwargs)

            # 結果を収集
            for future in as_completed(future_to_task, timeout=self.config.timeout * len(tasks)):
                task_idx, func, args, kwargs = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append((task_idx, result))
                    completed_count += 1
                except Exception as e:
                    print(f"{description} タスク {task_idx} エラー: {e}")
                    results.append((task_idx, None))
                    completed_count += 1

        # 結果を元の順序でソート
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def generate_candidates_parallel(
        self,
        candidate_generator,
        train_pairs: List[Dict[str, Any]],
        max_candidates: Optional[int] = None
    ) -> List[str]:
        """
        複数の訓練ペアから候補を並列生成

        Args:
            candidate_generator: CandidateGeneratorインスタンス
            train_pairs: 訓練ペアのリスト
            max_candidates: 最大候補数

        Returns:
            生成された候補プログラムのリスト
        """
        tasks = []
        for i, pair in enumerate(train_pairs):
            tasks.append((
                candidate_generator.generate_candidates,
                (pair['input'], pair['output']),
                {'max_candidates': max_candidates, 'pair_index': i}
            ))

        results = self.execute_parallel(tasks, description="候補生成")

        # 結果をフラット化
        all_candidates = []
        for result in results:
            if result:
                all_candidates.extend(result)

        return all_candidates

    def check_consistency_parallel(
        self,
        consistency_checker,
        program: str,
        train_pairs: List[Dict[str, Any]]
    ) -> List[float]:
        """
        複数の訓練ペアで一貫性チェックを並列実行

        Args:
            consistency_checker: ConsistencyCheckerインスタンス
            program: チェックするプログラム
            train_pairs: 訓練ペアのリスト

        Returns:
            各ペアの一貫性スコアのリスト
        """
        tasks = []
        for pair in train_pairs:
            tasks.append((
                consistency_checker._check_pair_consistency,
                (program, pair),
                {}
            ))

        results = self.execute_parallel(tasks, description="一貫性チェック")
        return [score if score is not None else 0.0 for score in results]

    def match_objects_parallel(
        self,
        object_matcher,
        train_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        複数のペアでオブジェクトマッチングを並列実行

        Args:
            object_matcher: オブジェクトマッチャーインスタンス（RuleBasedObjectMatcherなど、match_objectsメソッドを持つもの）
            train_pairs: 訓練ペアのリスト

        Returns:
            各ペアのマッチング結果のリスト
        """
        tasks = []
        for pair in train_pairs:
            # 単一ペアのタスクを作成
            def match_single_pair(pair):
                input_grid = pair['input']
                output_grid = pair['output']
                # オブジェクトマッチングを実行（本格実装）
                # object_matcherの適切なメソッドを使用
                if hasattr(object_matcher, 'match_objects'):
                    # match_objectsメソッドがある場合
                    from core.data_structures import Task, DataPair
                    # 単一ペアのタスクを作成
                    task = Task(
                        task_id=f"pair_{id(pair)}",
                        train=[DataPair(input=input_grid, output=output_grid)],
                        test=[]
                    )
                    result = object_matcher.match_objects(task)
                    return result
                elif hasattr(object_matcher, '_learn_matching_patterns_from_pair'):
                    # フォールバック: 内部メソッドを使用
                    return object_matcher._learn_matching_patterns_from_pair(input_grid, output_grid)
                else:
                    # 最終フォールバック: 空の結果を返す
                    return {'success': False, 'error': 'マッチングメソッドが見つかりません'}

            tasks.append((
                match_single_pair,
                (pair,),
                {}
            ))

        results = self.execute_parallel(tasks, description="オブジェクトマッチング")
        return [result if result else {} for result in results]
