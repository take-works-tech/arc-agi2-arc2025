"""
キャッシュ管理システム

プログラム実行結果、オブジェクト抽出結果、一貫性チェック結果をキャッシュ
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np


@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)


class CacheManager:
    """キャッシュ管理クラス"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600.0,  # 秒
        enable_memory_limit: bool = True,
        memory_limit_mb: float = 512.0
    ):
        """
        初期化

        Args:
            max_size: 最大キャッシュエントリ数
            ttl: キャッシュの有効期限（秒）
            enable_memory_limit: メモリ制限を有効化
            memory_limit_mb: メモリ制限（MB）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.enable_memory_limit = enable_memory_limit
        self.memory_limit_mb = memory_limit_mb

        # LRUキャッシュ（OrderedDictを使用）
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # キャッシュ統計
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

    def _generate_key(self, *args, **kwargs) -> str:
        """キャッシュキーを生成"""
        key_parts = []

        # 引数をハッシュ化
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
            elif isinstance(arg, (list, tuple)):
                # リストやタプルをJSON文字列に変換
                try:
                    key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
                except (TypeError, ValueError):
                    # JSON化できない場合は文字列表現を使用
                    key_parts.append(hash(str(arg)))
            elif isinstance(arg, dict):
                try:
                    key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
                except (TypeError, ValueError):
                    key_parts.append(hash(str(arg)))
            else:
                key_parts.append(str(arg))

        # キーワード引数をハッシュ化
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest())

        # すべての部分を結合してハッシュ
        combined = '_'.join(str(part) for part in key_parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        キャッシュから値を取得

        Args:
            key: キャッシュキー

        Returns:
            キャッシュされた値、存在しない場合はNone
        """
        if key not in self._cache:
            self.stats['misses'] += 1
            return None

        entry = self._cache[key]

        # TTLチェック
        if time.time() - entry.timestamp > self.ttl:
            del self._cache[key]
            self.stats['expirations'] += 1
            self.stats['misses'] += 1
            return None

        # LRU: アクセスしたエントリを最後に移動
        self._cache.move_to_end(key)
        entry.access_count += 1
        entry.last_access = time.time()

        self.stats['hits'] += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """
        キャッシュに値を保存

        Args:
            key: キャッシュキー
            value: 保存する値
        """
        # メモリ制限チェック
        if self.enable_memory_limit:
            if not self._check_memory_limit():
                # メモリ制限に達した場合、古いエントリを削除
                self._evict_oldest()

        # 最大サイズチェック
        if len(self._cache) >= self.max_size:
            # 最も古いエントリを削除（LRU）
            self._evict_oldest()

        # エントリを作成
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time()
        )

        # キャッシュに追加
        self._cache[key] = entry

    def _evict_oldest(self) -> None:
        """最も古いエントリを削除（LRU）"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats['evictions'] += 1

    def _check_memory_limit(self) -> bool:
        """メモリ制限をチェック"""
        if not self.enable_memory_limit:
            return True

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self.memory_limit_mb
        except Exception:
            # psutilが利用できない場合はチェックをスキップ
            return True

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'expirations': self.stats['expirations']
        }

    def cleanup_expired(self) -> int:
        """
        期限切れのエントリを削除

        Returns:
            削除されたエントリ数
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl
        ]

        for key in expired_keys:
            del self._cache[key]
            self.stats['expirations'] += 1

        return len(expired_keys)


class ProgramExecutionCache:
    """プログラム実行結果のキャッシュ"""

    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        """初期化"""
        self.cache = CacheManager(max_size=max_size, ttl=ttl)

    def get_execution_result(self, program: str, input_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        プログラム実行結果を取得

        Args:
            program: プログラムコード
            input_grid: 入力グリッド

        Returns:
            実行結果（キャッシュされている場合）、存在しない場合はNone
        """
        key = self.cache._generate_key(program, input_grid)
        return self.cache.get(key)

    def set_execution_result(self, program: str, input_grid: np.ndarray, output_grid: np.ndarray) -> None:
        """
        プログラム実行結果を保存

        Args:
            program: プログラムコード
            input_grid: 入力グリッド
            output_grid: 出力グリッド
        """
        key = self.cache._generate_key(program, input_grid)
        self.cache.set(key, output_grid.copy())


class ObjectExtractionCache:
    """オブジェクト抽出結果のキャッシュ"""

    def __init__(self, max_size: int = 500, ttl: float = 3600.0):
        """初期化"""
        self.cache = CacheManager(max_size=max_size, ttl=ttl)

    def get_extraction_result(self, grid: np.ndarray, connectivity: int = 4) -> Optional[Any]:
        """
        オブジェクト抽出結果を取得

        Args:
            grid: グリッド
            connectivity: 連結性（4または8）

        Returns:
            抽出結果（キャッシュされている場合）、存在しない場合はNone
        """
        key = self.cache._generate_key(grid, connectivity)
        return self.cache.get(key)

    def set_extraction_result(self, grid: np.ndarray, connectivity: int, objects: Any) -> None:
        """
        オブジェクト抽出結果を保存

        Args:
            grid: グリッド
            connectivity: 連結性（4または8）
            objects: 抽出されたオブジェクト
        """
        key = self.cache._generate_key(grid, connectivity)
        # オブジェクトリストをコピーして保存
        if isinstance(objects, list):
            self.cache.set(key, objects.copy())
        else:
            self.cache.set(key, objects)


class ConsistencyCheckCache:
    """一貫性チェック結果のキャッシュ"""

    def __init__(self, max_size: int = 500, ttl: float = 1800.0):
        """初期化"""
        self.cache = CacheManager(max_size=max_size, ttl=ttl)

    def get_consistency_result(self, program: str, task_id: str) -> Optional[float]:
        """
        一貫性チェック結果を取得

        Args:
            program: プログラムコード
            task_id: タスクID

        Returns:
            一貫性スコア（キャッシュされている場合）、存在しない場合はNone
        """
        key = self.cache._generate_key(program, task_id)
        return self.cache.get(key)

    def set_consistency_result(self, program: str, task_id: str, consistency_score: float) -> None:
        """
        一貫性チェック結果を保存

        Args:
            program: プログラムコード
            task_id: タスクID
            consistency_score: 一貫性スコア
        """
        key = self.cache._generate_key(program, task_id)
        self.cache.set(key, consistency_score)
