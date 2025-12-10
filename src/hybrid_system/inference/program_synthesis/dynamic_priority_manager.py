"""
動的優先度管理システム

各候補生成方法の成功率を追跡し、動的に優先度を調整
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class MethodStatistics:
    """方法別の統計情報"""
    method_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_candidates_generated: int = 0
    total_candidates_selected: int = 0
    average_consistency_score: float = 0.0
    last_success_rate: float = 0.0
    priority: float = 1.0  # 現在の優先度（1.0がデフォルト）


@dataclass
class DynamicPriorityConfig:
    """動的優先度設定"""
    enable_dynamic_priority: bool = True
    min_attempts_for_adjustment: int = 10  # 優先度調整に必要な最小試行回数
    success_rate_weight: float = 0.7  # 成功率の重み
    consistency_score_weight: float = 0.3  # 一貫性スコアの重み
    priority_adjustment_rate: float = 0.1  # 優先度調整の速度（0.0-1.0）
    min_priority: float = 0.1  # 最小優先度
    max_priority: float = 2.0  # 最大優先度
    decay_factor: float = 0.95  # 古いデータの減衰係数
    save_statistics: bool = True
    statistics_file: str = "outputs/dynamic_priority_statistics.json"


class DynamicPriorityManager:
    """動的優先度管理クラス"""

    # 候補生成方法の名前
    METHOD_NAMES = [
        'neural_grid',
        'neural_object',
    ]

    def __init__(self, config: Optional[DynamicPriorityConfig] = None):
        """
        初期化

        Args:
            config: 動的優先度設定
        """
        self.config = config or DynamicPriorityConfig()
        self.statistics: Dict[str, MethodStatistics] = {}

        # 各方法の統計を初期化
        for method_name in self.METHOD_NAMES:
            self.statistics[method_name] = MethodStatistics(method_name=method_name)

        # 統計を読み込み（存在する場合）
        if self.config.save_statistics:
            self._load_statistics()

    def record_attempt(
        self,
        method_name: str,
        success: bool,
        candidates_generated: int = 0,
        candidates_selected: int = 0,
        consistency_score: float = 0.0
    ) -> None:
        """
        試行結果を記録

        Args:
            method_name: 方法名
            success: 成功したかどうか
            candidates_generated: 生成された候補数
            candidates_selected: 選択された候補数
            consistency_score: 一貫性スコア
        """
        if method_name not in self.statistics:
            self.statistics[method_name] = MethodStatistics(method_name=method_name)

        stats = self.statistics[method_name]
        stats.total_attempts += 1

        if success:
            stats.successful_attempts += 1
        else:
            stats.failed_attempts += 1

        stats.total_candidates_generated += candidates_generated
        stats.total_candidates_selected += candidates_selected

        # 一貫性スコアの平均を更新
        if consistency_score > 0:
            if stats.average_consistency_score == 0.0:
                stats.average_consistency_score = consistency_score
            else:
                # 指数移動平均
                stats.average_consistency_score = (
                    self.config.decay_factor * stats.average_consistency_score +
                    (1 - self.config.decay_factor) * consistency_score
                )

        # 成功率を更新
        if stats.total_attempts > 0:
            stats.last_success_rate = stats.successful_attempts / stats.total_attempts

        # 優先度を更新
        if self.config.enable_dynamic_priority:
            self._update_priority(method_name)

        # 統計を保存
        if self.config.save_statistics:
            self._save_statistics()

    def _update_priority(self, method_name: str) -> None:
        """優先度を更新"""
        stats = self.statistics[method_name]

        # 最小試行回数に達していない場合は優先度を更新しない
        if stats.total_attempts < self.config.min_attempts_for_adjustment:
            return

        # 優先度スコアを計算
        success_score = stats.last_success_rate
        consistency_score = stats.average_consistency_score

        priority_score = (
            self.config.success_rate_weight * success_score +
            self.config.consistency_score_weight * consistency_score
        )

        # 優先度を調整
        target_priority = 1.0 + (priority_score - 0.5) * 2.0  # 0.0-1.0を0.0-2.0にマッピング

        # 現在の優先度から目標優先度に向かって調整
        adjustment = (target_priority - stats.priority) * self.config.priority_adjustment_rate
        new_priority = stats.priority + adjustment

        # 優先度の範囲を制限
        stats.priority = max(
            self.config.min_priority,
            min(self.config.max_priority, new_priority)
        )

    def get_priority(self, method_name: str) -> float:
        """
        方法の優先度を取得

        Args:
            method_name: 方法名

        Returns:
            優先度（高いほど優先される）
        """
        if method_name not in self.statistics:
            return 1.0  # デフォルト優先度

        return self.statistics[method_name].priority

    def get_method_priorities(self) -> Dict[str, float]:
        """
        すべての方法の優先度を取得

        Returns:
            方法名 -> 優先度の辞書
        """
        return {
            method_name: self.get_priority(method_name)
            for method_name in self.METHOD_NAMES
        }

    def get_statistics(self, method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        統計情報を取得

        Args:
            method_name: 方法名（指定しない場合はすべて）

        Returns:
            統計情報の辞書
        """
        if method_name:
            if method_name not in self.statistics:
                return {}
            stats = self.statistics[method_name]
            return {
                'method_name': stats.method_name,
                'total_attempts': stats.total_attempts,
                'successful_attempts': stats.successful_attempts,
                'failed_attempts': stats.failed_attempts,
                'success_rate': stats.last_success_rate,
                'average_consistency_score': stats.average_consistency_score,
                'priority': stats.priority,
                'total_candidates_generated': stats.total_candidates_generated,
                'total_candidates_selected': stats.total_candidates_selected
            }
        else:
            return {
                method_name: self.get_statistics(method_name)
                for method_name in self.METHOD_NAMES
            }

    def _save_statistics(self) -> None:
        """統計を保存"""
        try:
            output_file = Path(self.config.statistics_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            statistics_data = {}
            for method_name, stats in self.statistics.items():
                statistics_data[method_name] = {
                    'method_name': stats.method_name,
                    'total_attempts': stats.total_attempts,
                    'successful_attempts': stats.successful_attempts,
                    'failed_attempts': stats.failed_attempts,
                    'total_candidates_generated': stats.total_candidates_generated,
                    'total_candidates_selected': stats.total_candidates_selected,
                    'average_consistency_score': stats.average_consistency_score,
                    'last_success_rate': stats.last_success_rate,
                    'priority': stats.priority
                }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"警告: 統計の保存に失敗しました: {e}")

    def _load_statistics(self) -> None:
        """統計を読み込み"""
        try:
            statistics_file = Path(self.config.statistics_file)
            if not statistics_file.exists():
                return

            with open(statistics_file, 'r', encoding='utf-8') as f:
                statistics_data = json.load(f)

            for method_name, data in statistics_data.items():
                if method_name in self.statistics:
                    stats = self.statistics[method_name]
                    stats.total_attempts = data.get('total_attempts', 0)
                    stats.successful_attempts = data.get('successful_attempts', 0)
                    stats.failed_attempts = data.get('failed_attempts', 0)
                    stats.total_candidates_generated = data.get('total_candidates_generated', 0)
                    stats.total_candidates_selected = data.get('total_candidates_selected', 0)
                    stats.average_consistency_score = data.get('average_consistency_score', 0.0)
                    stats.last_success_rate = data.get('last_success_rate', 0.0)
                    stats.priority = data.get('priority', 1.0)

        except Exception as e:
            print(f"警告: 統計の読み込みに失敗しました: {e}")

    def reset_statistics(self) -> None:
        """統計をリセット"""
        for method_name in self.METHOD_NAMES:
            self.statistics[method_name] = MethodStatistics(method_name=method_name)

        if self.config.save_statistics:
            self._save_statistics()

    def get_recommended_methods(self, top_k: int = 3) -> List[str]:
        """
        推奨される方法を取得（優先度順）

        Args:
            top_k: 取得する方法数

        Returns:
            方法名のリスト（優先度の高い順）
        """
        method_priorities = self.get_method_priorities()
        sorted_methods = sorted(
            method_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [method_name for method_name, _ in sorted_methods[:top_k]]
