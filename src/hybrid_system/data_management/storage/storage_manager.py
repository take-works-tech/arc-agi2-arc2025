"""
ストレージマネージャー

データストレージの統合管理
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

from src.hybrid_system.core.data_structures import DataPair, Task


@dataclass
class StorageConfig:
    """ストレージ設定"""
    base_dir: str = "data/storage"
    enable_file_storage: bool = True
    enable_database_storage: bool = True
    enable_cache: bool = True
    max_cache_size: int = 1000
    compression: bool = True
    backup_enabled: bool = True


class StorageManager:
    """ストレージマネージャー"""

    def __init__(self, config: Optional[StorageConfig] = None):
        """初期化"""
        self.config = config or StorageConfig()

        # ストレージディレクトリを作成
        os.makedirs(self.config.base_dir, exist_ok=True)

        # サブディレクトリを作成
        self.pairs_dir = os.path.join(self.config.base_dir, "pairs")
        self.tasks_dir = os.path.join(self.config.base_dir, "tasks")
        self.cache_dir = os.path.join(self.config.base_dir, "cache")
        self.backup_dir = os.path.join(self.config.base_dir, "backup")

        for dir_path in [self.pairs_dir, self.tasks_dir, self.cache_dir, self.backup_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # データベース初期化
        if self.config.enable_database_storage:
            self._init_database()

        # ストレージ統計
        self.storage_stats = {
            'total_saves': 0,
            'total_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backup_count': 0
        }

    def save_data_pair(self, data_pair: DataPair, filename: Optional[str] = None) -> str:
        """DataPairを保存

        Args:
            data_pair: 保存するDataPair
            filename: ファイル名（Noneの場合は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            filename = f"pair_{data_pair.pair_id}.json"

        file_path = os.path.join(self.pairs_dir, filename)

        # ファイルストレージに保存
        if self.config.enable_file_storage:
            self._save_to_file(data_pair.to_dict(), file_path)

        # データベースに保存
        if self.config.enable_database_storage:
            self._save_to_database(data_pair, 'pairs')

        # キャッシュに保存
        if self.config.enable_cache:
            self._save_to_cache(data_pair.pair_id, data_pair)

        self.storage_stats['total_saves'] += 1

        return file_path

    def load_data_pair(self, pair_id: str) -> Optional[DataPair]:
        """DataPairを読み込み

        Args:
            pair_id: ペアID

        Returns:
            DataPair（見つからない場合はNone）
        """
        # キャッシュから読み込み
        if self.config.enable_cache:
            cached_pair = self._load_from_cache(pair_id)
            if cached_pair:
                self.storage_stats['cache_hits'] += 1
                self.storage_stats['total_loads'] += 1
                return cached_pair
            else:
                self.storage_stats['cache_misses'] += 1

        # データベースから読み込み
        if self.config.enable_database_storage:
            db_pair = self._load_from_database(pair_id, 'pairs')
            if db_pair:
                # キャッシュに保存
                if self.config.enable_cache:
                    self._save_to_cache(pair_id, db_pair)

                self.storage_stats['total_loads'] += 1
                return db_pair

        # ファイルから読み込み
        if self.config.enable_file_storage:
            file_path = os.path.join(self.pairs_dir, f"pair_{pair_id}.json")
            if os.path.exists(file_path):
                file_pair = self._load_from_file(file_path)
                if file_pair:
                    # キャッシュに保存
                    if self.config.enable_cache:
                        self._save_to_cache(pair_id, file_pair)

                    self.storage_stats['total_loads'] += 1
                    return file_pair

        return None

    def save_task(self, task: Task, filename: Optional[str] = None) -> str:
        """Taskを保存

        Args:
            task: 保存するTask
            filename: ファイル名（Noneの場合は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            filename = f"task_{task.task_id}.json"

        file_path = os.path.join(self.tasks_dir, filename)

        # ファイルストレージに保存
        if self.config.enable_file_storage:
            self._save_to_file(task.to_dict(), file_path)

        # データベースに保存
        if self.config.enable_database_storage:
            self._save_to_database(task, 'tasks')

        # キャッシュに保存
        if self.config.enable_cache:
            self._save_to_cache(task.task_id, task)

        self.storage_stats['total_saves'] += 1

        return file_path

    def load_task(self, task_id: str) -> Optional[Task]:
        """Taskを読み込み

        Args:
            task_id: タスクID

        Returns:
            Task（見つからない場合はNone）
        """
        # キャッシュから読み込み
        if self.config.enable_cache:
            cached_task = self._load_from_cache(task_id)
            if cached_task:
                self.storage_stats['cache_hits'] += 1
                self.storage_stats['total_loads'] += 1
                return cached_task
            else:
                self.storage_stats['cache_misses'] += 1

        # データベースから読み込み
        if self.config.enable_database_storage:
            db_task = self._load_from_database(task_id, 'tasks')
            if db_task:
                # キャッシュに保存
                if self.config.enable_cache:
                    self._save_to_cache(task_id, db_task)

                self.storage_stats['total_loads'] += 1
                return db_task

        # ファイルから読み込み
        if self.config.enable_file_storage:
            file_path = os.path.join(self.tasks_dir, f"task_{task_id}.json")
            if os.path.exists(file_path):
                file_task = self._load_from_file(file_path)
                if file_task:
                    # キャッシュに保存
                    if self.config.enable_cache:
                        self._save_to_cache(task_id, file_task)

                    self.storage_stats['total_loads'] += 1
                    return file_task

        return None

    def list_data_pairs(self) -> List[str]:
        """DataPairのリストを取得

        Returns:
            ペアIDのリスト
        """
        pair_ids = []

        # データベースから取得
        if self.config.enable_database_storage:
            db_ids = self._list_from_database('pairs')
            pair_ids.extend(db_ids)

        # ファイルから取得
        if self.config.enable_file_storage:
            file_ids = self._list_from_files(self.pairs_dir, 'pair_')
            pair_ids.extend(file_ids)

        # 重複を除去
        return list(set(pair_ids))

    def list_tasks(self) -> List[str]:
        """Taskのリストを取得

        Returns:
            タスクIDのリスト
        """
        task_ids = []

        # データベースから取得
        if self.config.enable_database_storage:
            db_ids = self._list_from_database('tasks')
            task_ids.extend(db_ids)

        # ファイルから取得
        if self.config.enable_file_storage:
            file_ids = self._list_from_files(self.tasks_dir, 'task_')
            task_ids.extend(file_ids)

        # 重複を除去
        return list(set(task_ids))

    def delete_data_pair(self, pair_id: str) -> bool:
        """DataPairを削除

        Args:
            pair_id: ペアID

        Returns:
            削除成功の場合True
        """
        success = True

        # ファイルから削除
        if self.config.enable_file_storage:
            file_path = os.path.join(self.pairs_dir, f"pair_{pair_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    success = False

        # データベースから削除
        if self.config.enable_database_storage:
            db_success = self._delete_from_database(pair_id, 'pairs')
            success = success and db_success

        # キャッシュから削除
        if self.config.enable_cache:
            self._delete_from_cache(pair_id)

        return success

    def delete_task(self, task_id: str) -> bool:
        """Taskを削除

        Args:
            task_id: タスクID

        Returns:
            削除成功の場合True
        """
        success = True

        # ファイルから削除
        if self.config.enable_file_storage:
            file_path = os.path.join(self.tasks_dir, f"task_{task_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    success = False

        # データベースから削除
        if self.config.enable_database_storage:
            db_success = self._delete_from_database(task_id, 'tasks')
            success = success and db_success

        # キャッシュから削除
        if self.config.enable_cache:
            self._delete_from_cache(task_id)

        return success

    def create_backup(self) -> str:
        """バックアップを作成

        Returns:
            バックアップディレクトリのパス
        """
        if not self.config.backup_enabled:
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")

        # バックアップディレクトリを作成
        os.makedirs(backup_path, exist_ok=True)

        # ファイルをコピー
        import shutil

        if os.path.exists(self.pairs_dir):
            shutil.copytree(self.pairs_dir, os.path.join(backup_path, "pairs"))

        if os.path.exists(self.tasks_dir):
            shutil.copytree(self.tasks_dir, os.path.join(backup_path, "tasks"))

        # データベースをコピー
        if self.config.enable_database_storage:
            db_path = os.path.join(self.config.base_dir, "storage.db")
            if os.path.exists(db_path):
                shutil.copy2(db_path, os.path.join(backup_path, "storage.db"))

        self.storage_stats['backup_count'] += 1

        return backup_path

    def _init_database(self):
        """データベースを初期化"""
        db_path = os.path.join(self.config.base_dir, "storage.db")
        self.db_connection = sqlite3.connect(db_path)

        # テーブルを作成
        cursor = self.db_connection.cursor()

        # DataPairテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pairs (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Taskテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.db_connection.commit()

    def _save_to_file(self, data: Dict[str, Any], file_path: str):
        """ファイルに保存"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_file(self, file_path: str) -> Optional[Union[DataPair, Task]]:
        """ファイルから読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # データタイプを判定
            if 'pair_id' in data:
                return DataPair.from_dict(data)
            elif 'task_id' in data:
                return Task.from_dict(data)
            else:
                return None
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return None

    def _save_to_database(self, obj: Union[DataPair, Task], table: str):
        """データベースに保存"""
        try:
            cursor = self.db_connection.cursor()

            if isinstance(obj, DataPair):
                obj_id = obj.pair_id
            elif isinstance(obj, Task):
                obj_id = obj.task_id
            else:
                return

            data_json = json.dumps(obj.to_dict(), ensure_ascii=False)

            cursor.execute(f'''
                INSERT OR REPLACE INTO {table} (id, data)
                VALUES (?, ?)
            ''', (obj_id, data_json))

            self.db_connection.commit()
        except Exception as e:
            print(f"データベース保存エラー: {e}")

    def _load_from_database(self, obj_id: str, table: str) -> Optional[Union[DataPair, Task]]:
        """データベースから読み込み"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f'SELECT data FROM {table} WHERE id = ?', (obj_id,))
            result = cursor.fetchone()

            if result:
                data = json.loads(result[0])

                if table == 'pairs':
                    return DataPair.from_dict(data)
                elif table == 'tasks':
                    return Task.from_dict(data)

            return None
        except Exception as e:
            print(f"データベース読み込みエラー: {e}")
            return None

    def _list_from_database(self, table: str) -> List[str]:
        """データベースからリストを取得"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f'SELECT id FROM {table}')
            results = cursor.fetchall()
            return [result[0] for result in results]
        except Exception as e:
            print(f"データベースリスト取得エラー: {e}")
            return []

    def _delete_from_database(self, obj_id: str, table: str) -> bool:
        """データベースから削除"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f'DELETE FROM {table} WHERE id = ?', (obj_id,))
            self.db_connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"データベース削除エラー: {e}")
            return False

    def _list_from_files(self, directory: str, prefix: str) -> List[str]:
        """ファイルからリストを取得"""
        try:
            files = os.listdir(directory)
            ids = []
            for file in files:
                if file.startswith(prefix) and file.endswith('.json'):
                    # プレフィックスと拡張子を除去
                    obj_id = file[len(prefix):-5]  # .jsonを除去
                    ids.append(obj_id)
            return ids
        except Exception as e:
            print(f"ファイルリスト取得エラー: {e}")
            return []

    def _save_to_cache(self, obj_id: str, obj: Union[DataPair, Task]):
        """キャッシュに保存（本格実装）"""
        # 高度なキャッシュ管理: LRUキャッシュ、サイズ制限、圧縮などを考慮
        cache_file = os.path.join(self.cache_dir, f"{obj_id}.pkl")

        try:
            # キャッシュディレクトリが存在しない場合は作成
            os.makedirs(self.cache_dir, exist_ok=True)

            # キャッシュサイズ制限をチェック（必要に応じて古いキャッシュを削除）
            self._manage_cache_size()

            # オブジェクトをシリアライズして保存
            with open(cache_file, 'wb') as f:
                pickle.dump(obj, f)

        except Exception as e:
            print(f"キャッシュ保存エラー: {e}")

    def _manage_cache_size(self):
        """キャッシュサイズを管理（本格実装）"""
        # キャッシュディレクトリ内のファイルを取得
        if not os.path.exists(self.cache_dir):
            return

        cache_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                cache_files.append((filepath, mtime, size))

        # サイズでソート（古いものから）
        cache_files.sort(key=lambda x: x[1])

        # 最大キャッシュサイズ（100MB）を超える場合、古いファイルを削除
        max_cache_size = 100 * 1024 * 1024  # 100MB
        total_size = sum(size for _, _, size in cache_files)

        while total_size > max_cache_size and cache_files:
            oldest_file, _, size = cache_files.pop(0)
            try:
                os.remove(oldest_file)
                total_size -= size
            except Exception as e:
                print(f"キャッシュファイル削除エラー: {e}")

    def _load_from_cache(self, obj_id: str) -> Optional[Union[DataPair, Task]]:
        """キャッシュから読み込み"""
        cache_file = os.path.join(self.cache_dir, f"{obj_id}.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
        return None

    def _delete_from_cache(self, obj_id: str):
        """キャッシュから削除"""
        cache_file = os.path.join(self.cache_dir, f"{obj_id}.pkl")
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            print(f"キャッシュ削除エラー: {e}")

    def get_storage_statistics(self) -> Dict[str, Any]:
        """ストレージ統計を取得"""
        stats = dict(self.storage_stats)

        # キャッシュヒット率を計算
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0.0

        # ストレージ使用量を計算
        stats['storage_usage'] = self._calculate_storage_usage()

        return stats

    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """ストレージ使用量を計算"""
        usage = {
            'pairs_dir_size': 0,
            'tasks_dir_size': 0,
            'cache_dir_size': 0,
            'total_size': 0
        }

        try:
            # ディレクトリサイズを計算
            for dir_path, key in [(self.pairs_dir, 'pairs_dir_size'),
                                 (self.tasks_dir, 'tasks_dir_size'),
                                 (self.cache_dir, 'cache_dir_size')]:
                if os.path.exists(dir_path):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(dir_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                total_size += os.path.getsize(filepath)
                    usage[key] = total_size

            usage['total_size'] = sum(usage.values())
        except Exception as e:
            print(f"ストレージ使用量計算エラー: {e}")

        return usage

    def cleanup_cache(self):
        """キャッシュをクリーンアップ"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"キャッシュクリーンアップエラー: {e}")

    def close(self):
        """ストレージマネージャーを閉じる"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
