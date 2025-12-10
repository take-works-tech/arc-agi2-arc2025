"""
データセットI/O

DataPairとTaskの保存・読み込み機能を提供
"""

import json
import os
import gzip
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

from src.hybrid_system.core.data_structures import DataPair, Task
from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import (
    is_empty_grid
)


class DatasetIO:
    """データセットのI/O管理"""

    def __init__(self, base_dir: str = 'data/generated/hybrid_dataset'):
        """初期化

        Args:
            base_dir: ベースディレクトリ
        """
        self.base_dir = base_dir
        os.makedirs(os.path.join(self.base_dir, 'phase1_pairs'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'phase2_tasks'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'statistics'), exist_ok=True)

    def _get_file_path(self, data_type: str, filename: str, compress: bool = False) -> str:
        """ファイルパスを生成

        Args:
            data_type: データタイプ（'phase1' または 'phase2'）
            filename: ファイル名
            compress: 圧縮するか

        Returns:
            ファイルパス
        """
        ext = '.jsonl.gz' if compress else '.jsonl' if data_type == 'phase1' else '.json'
        subdir = 'phase1_pairs' if data_type == 'phase1' else 'phase2_tasks'
        return os.path.join(self.base_dir, subdir, f"{filename}{ext}")

    def save_data_pairs(
        self,
        data_pairs: List[DataPair],
        filename: str = 'data_pairs',
        format: str = 'jsonl',
        compress: bool = True,
        append: bool = False
    ) -> str:
        """DataPairを保存（フェーズ1）

        Args:
            data_pairs: 保存するDataPairのリスト
            filename: ファイル名
            format: フォーマット（'jsonl'）
            compress: 圧縮するか
            append: 追記するか

        Returns:
            保存されたファイルパス
        """
        file_path = self._get_file_path('phase1', filename, compress)
        mode = 'a' if append else 'w'

        _open = gzip.open if compress else open
        with _open(file_path, mode + 't', encoding='utf-8') as f:
            for pair in data_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')

        return file_path

    def load_data_pairs(self, file_path: str) -> List[DataPair]:
        """DataPairを読み込み

        Args:
            file_path: ファイルパス

        Returns:
            DataPairのリスト
        """
        data_pairs = []
        _open = gzip.open if file_path.endswith('.gz') else open

        with _open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                data_pairs.append(DataPair.from_dict(data))

        return data_pairs

    def save_tasks(
        self,
        tasks: List[Task],
        filename: str = 'tasks',
        format: str = 'json',
        compress: bool = True,
        arc_compatible: bool = True
    ) -> str:
        """Taskを保存（フェーズ2）

        Args:
            tasks: 保存するTaskのリスト
            filename: ファイル名
            format: フォーマット（'json'）
            compress: 圧縮するか
            arc_compatible: ARC-AGI2互換形式にするか

        Returns:
            保存されたファイルパス
        """
        file_path = self._get_file_path('phase2', filename, compress)

        if arc_compatible:
            # ARC-AGI2互換形式に変換
            arc_tasks_dict = {}
            for task in tasks:
                arc_tasks_dict[task.task_id] = {
                    "train": task.train,
                    "test": task.test,
                    "program": task.program,  # メタデータとして含める
                    "metadata": task.metadata
                }
            data_to_save = arc_tasks_dict
        else:
            data_to_save = {task.task_id: task.to_dict() for task in tasks}

        _open = gzip.open if compress else open
        with _open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        return file_path

    def load_tasks(self, file_path: str, filter_empty_output: bool = True) -> List[Task]:
        """Taskを読み込み

        Args:
            file_path: ファイルパス
            filter_empty_output: 空のoutputタスクをフィルタリングするか（デフォルト: True）

        Returns:
            Taskのリスト
        """
        _open = gzip.open if file_path.endswith('.gz') else open

        with _open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        filtered_count = 0

        for task_id, task_data in data.items():
            # ARC互換形式からTaskオブジェクトを再構築
            train = task_data.get('train', [])
            test = task_data.get('test', [])
            program = task_data.get('program', '')
            metadata = task_data.get('metadata', {})

            task = Task(
                task_id=task_id,
                train=train,
                test=test,
                program=program,
                metadata=metadata
            )

            # 空のoutputタスクをフィルタリング
            if filter_empty_output and self._has_empty_output(task):
                filtered_count += 1
                continue

            tasks.append(task)

        if filtered_count > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"空のoutputタスクを{filtered_count}個フィルタリングしました（{file_path}）")

        return tasks

    def _has_empty_output(self, task: Task) -> bool:
        """タスクに空のoutputがあるかチェック

        Args:
            task: チェックするタスク

        Returns:
            空のoutputがある場合はTrue
        """
        # trainペアをチェック
        if task.train:
            for pair in task.train:
                if isinstance(pair, dict) and 'output' in pair:
                    if is_empty_grid(pair['output']):
                        return True

        # testペアをチェック
        if task.test:
            for pair in task.test:
                if isinstance(pair, dict) and 'output' in pair:
                    if is_empty_grid(pair['output']):
                        return True

        return False

    def list_saved_files(self, phase: str) -> Dict[str, List[str]]:
        """指定されたフェーズの保存済みファイルをリストアップ

        Args:
            phase: フェーズ（'phase1' または 'phase2'）

        Returns:
            ファイルリスト
        """
        subdir = 'phase1_pairs' if phase == 'phase1' else 'phase2_tasks'
        dir_path = os.path.join(self.base_dir, subdir)

        if not os.path.exists(dir_path):
            return {phase: []}

        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        return {phase: files}

    def save_statistics(self, statistics: Dict[str, Any], filename: str = None) -> str:
        """統計情報を保存

        Args:
            statistics: 統計情報
            filename: ファイル名（Noneの場合は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"statistics_{timestamp}"

        file_path = os.path.join(self.base_dir, 'statistics', f"{filename}.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        return file_path

    def load_statistics(self, file_path: str) -> Dict[str, Any]:
        """統計情報を読み込み

        Args:
            file_path: ファイルパス

        Returns:
            統計情報
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_latest_file(self, phase: str) -> Optional[str]:
        """最新のファイルを取得

        Args:
            phase: フェーズ（'phase1' または 'phase2'）

        Returns:
            最新のファイルパス（見つからない場合はNone）
        """
        files = self.list_saved_files(phase)
        file_list = files.get(phase, [])

        if not file_list:
            return None

        # 最新のファイルを選択
        subdir = 'phase1_pairs' if phase == 'phase1' else 'phase2_tasks'
        dir_path = os.path.join(self.base_dir, subdir)

        latest_file = max(file_list, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)))
        return os.path.join(dir_path, latest_file)

    def cleanup_old_files(self, phase: str, keep_latest: int = 5) -> List[str]:
        """古いファイルをクリーンアップ

        Args:
            phase: フェーズ（'phase1' または 'phase2'）
            keep_latest: 保持する最新ファイル数

        Returns:
            削除されたファイルのリスト
        """
        files = self.list_saved_files(phase)
        file_list = files.get(phase, [])

        if len(file_list) <= keep_latest:
            return []

        # ファイルを更新日時でソート
        subdir = 'phase1_pairs' if phase == 'phase1' else 'phase2_tasks'
        dir_path = os.path.join(self.base_dir, subdir)

        sorted_files = sorted(
            file_list,
            key=lambda f: os.path.getmtime(os.path.join(dir_path, f)),
            reverse=True
        )

        # 古いファイルを削除
        files_to_delete = sorted_files[keep_latest:]
        deleted_files = []

        for filename in files_to_delete:
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except OSError as e:
                print(f"ファイル削除エラー: {file_path}, {e}")

        return deleted_files

    def get_storage_info(self) -> Dict[str, Any]:
        """ストレージ情報を取得

        Returns:
            ストレージ情報
        """
        info = {
            'base_dir': self.base_dir,
            'phase1_files': len(self.list_saved_files('phase1').get('phase1', [])),
            'phase2_files': len(self.list_saved_files('phase2').get('phase2', [])),
            'statistics_files': 0
        }

        # 統計ファイル数をカウント
        stats_dir = os.path.join(self.base_dir, 'statistics')
        if os.path.exists(stats_dir):
            info['statistics_files'] = len([f for f in os.listdir(stats_dir) if f.endswith('.json')])

        return info
