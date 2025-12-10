"""
パフォーマンスプロファイラー

実行時間の計測と分析を行う
"""
import time
import functools
from typing import Dict, List, Any, Callable
from collections import defaultdict


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.current_nested_level = 0
        self.nested_contexts: List[str] = []

    def reset(self):
        """計測結果をリセット"""
        self.timings.clear()
        self.call_counts.clear()
        self.current_nested_level = 0
        self.nested_contexts.clear()

    def record_timing(self, operation_name: str, elapsed_time: float, context: str = ""):
        """実行時間を記録"""
        full_name = f"{operation_name}" + (f" ({context})" if context else "")
        self.timings[full_name].append(elapsed_time)
        self.call_counts[full_name] += 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """統計情報を取得"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                    'total_count': self.call_counts[name]
                }
        return stats

    def print_statistics(self):
        """統計情報を出力"""
        stats = self.get_statistics()
        if not stats:
            print("[プロファイラー] 計測データがありません")
            return

        # 合計時間でソート
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)

        print("\n" + "=" * 80)
        print("パフォーマンス統計")
        print("=" * 80)
        print(f"{'操作名':<50} {'合計時間':<12} {'平均時間':<12} {'回数':<8} {'総呼び出し':<10}")
        print("-" * 80)

        total_all = 0
        for name, data in sorted_stats:
            total_all += data['total']
            print(f"{name[:48]:<50} {data['total']:>10.3f}s {data['average']:>10.3f}s "
                  f"{data['count']:>8} {data['total_count']:>10}")

        print("-" * 80)
        print(f"{'合計':<50} {total_all:>10.3f}s")
        print("=" * 80)

        # トップ10を表示
        print("\nトップ10（合計時間順）:")
        for i, (name, data) in enumerate(sorted_stats[:10], 1):
            percentage = (data['total'] / total_all * 100) if total_all > 0 else 0
            print(f"  {i}. {name}: {data['total']:.3f}s ({percentage:.1f}%) "
                  f"[平均: {data['average']:.3f}s, 回数: {data['count']}]")


# グローバルプロファイラーインスタンス
_profiler = PerformanceProfiler()


def profile_function(func_name: str = None, context: str = ""):
    """関数の実行時間を計測するデコレータ"""
    def decorator(func: Callable):
        name = func_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                _profiler.record_timing(name, elapsed_time, context)
        return wrapper
    return decorator


def profile_code_block(block_name: str, context: str = ""):
    """コードブロックの実行時間を計測するコンテキストマネージャー"""
    class ProfileContext:
        def __init__(self, name: str, ctx: str):
            self.name = name
            self.context = ctx
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_time = time.time() - self.start_time
            _profiler.record_timing(self.name, elapsed_time, self.context)
            return False

    return ProfileContext(block_name, context)


def get_profiler() -> PerformanceProfiler:
    """プロファイラーインスタンスを取得"""
    return _profiler


def reset_profiler():
    """プロファイラーをリセット"""
    _profiler.reset()


def print_profiling_statistics():
    """プロファイリング統計を出力"""
    _profiler.print_statistics()
