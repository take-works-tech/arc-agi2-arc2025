"""
メモリ管理システム

大規模タスクへの対応、メモリ使用量の監視、ガベージコレクションの最適化
"""

import gc
import psutil
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class MemoryConfig:
    """メモリ管理設定"""
    enable_monitoring: bool = True
    memory_limit_mb: float = 2048.0
    warning_threshold_mb: float = 1536.0  # 警告を出す閾値（メモリ制限の75%）
    gc_threshold_mb: float = 1024.0  # ガベージコレクションを実行する閾値
    check_interval: float = 1.0  # メモリチェック間隔（秒）


class MemoryManager:
    """メモリ管理クラス"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初期化

        Args:
            config: メモリ管理設定
        """
        self.config = config or MemoryConfig()
        self.last_check_time = 0.0
        self.memory_stats = {
            'peak_memory_mb': 0.0,
            'current_memory_mb': 0.0,
            'gc_count': 0,
            'cache_clears': 0
        }

    def get_memory_usage(self) -> float:
        """
        現在のメモリ使用量を取得（MB）

        Returns:
            メモリ使用量（MB）
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_stats['current_memory_mb'] = memory_mb
            if memory_mb > self.memory_stats['peak_memory_mb']:
                self.memory_stats['peak_memory_mb'] = memory_mb
            return memory_mb
        except Exception:
            return 0.0

    def check_memory_limit(self) -> bool:
        """
        メモリ制限をチェック

        Returns:
            メモリ制限内の場合True
        """
        if not self.config.enable_monitoring:
            return True

        current_time = time.time()
        if current_time - self.last_check_time < self.config.check_interval:
            return True

        self.last_check_time = current_time
        memory_mb = self.get_memory_usage()

        # 警告閾値を超えた場合
        if memory_mb > self.config.warning_threshold_mb:
            print(f"警告: メモリ使用量が高いです: {memory_mb:.2f}MB / {self.config.memory_limit_mb:.2f}MB")

        # メモリ制限を超えた場合
        if memory_mb > self.config.memory_limit_mb:
            print(f"エラー: メモリ制限を超えました: {memory_mb:.2f}MB / {self.config.memory_limit_mb:.2f}MB")
            return False

        # GC閾値を超えた場合、ガベージコレクションを実行
        if memory_mb > self.config.gc_threshold_mb:
            self.force_garbage_collection()

        return True

    def force_garbage_collection(self) -> None:
        """強制的にガベージコレクションを実行"""
        # Pythonのガベージコレクション
        collected = gc.collect()
        self.memory_stats['gc_count'] += 1

        # GPUメモリのクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if collected > 0:
            print(f"ガベージコレクション実行: {collected}オブジェクトを解放")

    def clear_caches(self, caches: Dict[str, Any]) -> int:
        """
        キャッシュをクリア

        Args:
            caches: クリアするキャッシュの辞書（名前 -> キャッシュオブジェクト）

        Returns:
            クリアされたキャッシュ数
        """
        cleared_count = 0
        for name, cache in caches.items():
            if hasattr(cache, 'clear'):
                cache.clear()
                cleared_count += 1
                print(f"キャッシュをクリア: {name}")

        self.memory_stats['cache_clears'] += cleared_count
        return cleared_count

    def optimize_memory(self, caches: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        メモリを最適化

        Args:
            caches: クリアするキャッシュの辞書（オプション）

        Returns:
            最適化結果
        """
        memory_before = self.get_memory_usage()

        # ガベージコレクション
        self.force_garbage_collection()

        # キャッシュをクリア
        if caches:
            self.clear_caches(caches)

        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after

        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed,
            'optimization_success': memory_freed > 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """メモリ統計を取得"""
        current_memory = self.get_memory_usage()
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.memory_stats['peak_memory_mb'],
            'memory_limit_mb': self.config.memory_limit_mb,
            'memory_usage_percent': (current_memory / self.config.memory_limit_mb * 100) if self.config.memory_limit_mb > 0 else 0.0,
            'gc_count': self.memory_stats['gc_count'],
            'cache_clears': self.memory_stats['cache_clears']
        }

    def monitor_memory(self, func: Callable, *args, **kwargs) -> tuple[Any, Dict[str, Any]]:
        """
        メモリを監視しながら関数を実行

        Args:
            func: 実行する関数
            *args, **kwargs: 関数の引数

        Returns:
            (関数の結果, メモリ統計)
        """
        memory_before = self.get_memory_usage()

        # メモリ制限チェック
        if not self.check_memory_limit():
            raise MemoryError(f"メモリ制限を超えています: {memory_before:.2f}MB / {self.config.memory_limit_mb:.2f}MB")

        # 関数を実行
        result = func(*args, **kwargs)

        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before

        # 実行後のメモリチェック
        self.check_memory_limit()

        stats = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_used
        }

        return result, stats
