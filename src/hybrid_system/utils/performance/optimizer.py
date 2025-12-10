#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パフォーマンス最適化システム（新システム統合版）
学習システムと推論システムのパフォーマンスを最適化
"""

import numpy as np
import json
import os
import time
import psutil
import torch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from src.hybrid_system.utils.logging import Logger

logger = Logger.get_logger("PerformanceOptimizer")

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float  # %
    gpu_usage: float = 0.0  # MB
    accuracy: float = 0.0
    throughput: float = 0.0  # ops/sec

@dataclass
class OptimizationResult:
    """最適化結果"""
    component: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_ratio: float
    optimization_applied: str
    success: bool

class PerformanceOptimizer:
    """パフォーマンス最適化器"""

    def __init__(self, results_dir: str = "performance_optimization_results"):
        """初期化"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # キャッシュ
        self._function_cache: Dict[int, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("パフォーマンス最適化器初期化完了")

    def measure_performance(self, func: Callable, *args, **kwargs) -> PerformanceMetrics:
        """
        関数のパフォーマンスを測定

        Args:
            func: 測定する関数
            *args, **kwargs: 関数の引数

        Returns:
            PerformanceMetrics: 測定結果
        """
        try:
            # プロセス情報
            process = psutil.Process()

            # メモリ使用量測定開始
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # CPU使用率測定開始
            cpu_before = psutil.cpu_percent(interval=0.1)

            # GPU使用率測定開始
            gpu_before = 0.0
            if torch.cuda.is_available():
                gpu_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

            # 実行時間測定
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # メモリ使用量測定終了
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = max(0, memory_after - memory_before)

            # CPU使用率測定終了
            cpu_after = psutil.cpu_percent(interval=0.1)
            cpu_usage = (cpu_before + cpu_after) / 2

            # GPU使用率測定終了
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_usage = max(0, gpu_after - gpu_before)

            # スループット計算
            throughput = 1.0 / execution_time if execution_time > 0 else 0.0

            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                throughput=throughput
            )

        except Exception as e:
            logger.error(f"パフォーマンス測定エラー: {e}")
            return PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                gpu_usage=0.0,
                throughput=0.0
            )

    def optimize_with_cache(self, func: Callable, cache_size: int = 100) -> Callable:
        """
        キャッシュ機能を追加して関数を最適化

        Args:
            func: 最適化する関数
            cache_size: キャッシュサイズ

        Returns:
            最適化された関数
        """
        def cached_func(*args, **kwargs):
            # キャッシュキーを生成
            try:
                # argsをハッシュ化（numpy配列対応）
                key_parts = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        key_parts.append(hash(arg.tobytes()))
                    else:
                        key_parts.append(hash(str(arg)))
                cache_key = hash(tuple(key_parts))
            except:
                # ハッシュ化失敗の場合はキャッシュなし
                return func(*args, **kwargs)

            # キャッシュチェック
            if cache_key in self._function_cache:
                self._cache_hits += 1
                return self._function_cache[cache_key]

            # 関数実行
            self._cache_misses += 1
            result = func(*args, **kwargs)

            # キャッシュに保存（サイズ管理）
            if len(self._function_cache) >= cache_size:
                # 最も古いエントリを削除（LRU）
                oldest_key = next(iter(self._function_cache))
                del self._function_cache[oldest_key]

            self._function_cache[cache_key] = result
            return result

        return cached_func

    def compare_implementations(
        self,
        component_name: str,
        original_func: Callable,
        optimized_func: Callable,
        test_data: Any,
        optimization_description: str
    ) -> OptimizationResult:
        """
        2つの実装のパフォーマンスを比較

        Args:
            component_name: コンポーネント名
            original_func: 元の実装
            optimized_func: 最適化版の実装
            test_data: テストデータ
            optimization_description: 最適化の説明

        Returns:
            OptimizationResult: 比較結果
        """
        logger.info(f"{component_name}の最適化開始")

        # 元の実装を測定
        before_metrics = self.measure_performance(original_func, test_data)

        # 最適化版を測定
        after_metrics = self.measure_performance(optimized_func, test_data)

        # 改善率計算
        if before_metrics.execution_time > 0:
            improvement_ratio = (
                (before_metrics.execution_time - after_metrics.execution_time) /
                before_metrics.execution_time
            )
        else:
            improvement_ratio = 0.0

        result = OptimizationResult(
            component=component_name,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_ratio=improvement_ratio,
            optimization_applied=optimization_description,
            success=improvement_ratio > 0
        )

        logger.info(f"{component_name}最適化完了: 改善率 {improvement_ratio:.3f}")
        return result

    def optimize_memory_usage(self, func: Callable) -> Callable:
        """
        メモリ使用量を最適化

        Args:
            func: 最適化する関数

        Returns:
            最適化された関数
        """
        def optimized_func(*args, **kwargs):
            # 実行前にガベージコレクション
            import gc
            gc.collect()

            # GPU メモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 関数実行
            result = func(*args, **kwargs)

            # 実行後にガベージコレクション
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        return optimized_func

    def save_optimization_results(self, results: Dict[str, OptimizationResult]):
        """最適化結果を保存"""
        logger.info("最適化結果保存開始")

        # 結果を辞書形式に変換
        results_data = {}
        for component, result in results.items():
            results_data[component] = {
                'optimization_applied': result.optimization_applied,
                'improvement_ratio': result.improvement_ratio,
                'success': result.success,
                'before_metrics': asdict(result.before_metrics),
                'after_metrics': asdict(result.after_metrics)
            }

        # JSONファイルとして保存
        results_file = self.results_dir / "optimization_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        # レポートを生成
        self._generate_optimization_report(results)

        logger.info(f"最適化結果保存完了: {results_file}")

    def _generate_optimization_report(self, results: Dict[str, OptimizationResult]):
        """最適化レポートを生成"""
        report_file = self.results_dir / "optimization_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("パフォーマンス最適化レポート\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"最適化実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("【最適化サマリー】\n")
            total_improvements = 0
            successful_optimizations = 0

            for component, result in results.items():
                f.write(f"\n{component.upper()}:\n")
                f.write(f"  最適化手法: {result.optimization_applied}\n")
                f.write(f"  改善率: {result.improvement_ratio:.3f}\n")
                f.write(f"  成功: {'はい' if result.success else 'いいえ'}\n")

                if result.success:
                    total_improvements += result.improvement_ratio
                    successful_optimizations += 1

                before = result.before_metrics
                after = result.after_metrics
                f.write(f"  実行時間: {before.execution_time:.4f}s -> {after.execution_time:.4f}s\n")
                f.write(f"  メモリ: {before.memory_usage:.2f}MB -> {after.memory_usage:.2f}MB\n")
                f.write(f"  CPU: {before.cpu_usage:.2f}% -> {after.cpu_usage:.2f}%\n")
                if before.gpu_usage > 0 or after.gpu_usage > 0:
                    f.write(f"  GPU: {before.gpu_usage:.2f}MB -> {after.gpu_usage:.2f}MB\n")

            f.write(f"\n【全体サマリー】\n")
            f.write(f"  成功した最適化数: {successful_optimizations}/{len(results)}\n")
            avg_improvement = total_improvements / max(successful_optimizations, 1)
            f.write(f"  平均改善率: {avg_improvement:.3f}\n")

            # キャッシュ統計
            if self._cache_hits + self._cache_misses > 0:
                cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
                f.write(f"  キャッシュヒット率: {cache_hit_rate:.3f}\n")

            if successful_optimizations == len(results):
                f.write(f"  最適化状況: 全て成功 ✅\n")
            elif successful_optimizations > len(results) // 2:
                f.write(f"  最適化状況: 大部分成功 🟡\n")
            else:
                f.write(f"  最適化状況: 改善が必要 🔴\n")

        logger.info(f"最適化レポート生成完了: {report_file}")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self._function_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
