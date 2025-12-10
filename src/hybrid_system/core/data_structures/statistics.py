"""
データセット統計情報

データセット全体の統計情報を管理
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DatasetStatistics:
    """データセット全体の統計情報"""
    total_pairs: int = 0
    total_tasks: int = 0
    unique_programs: int = 0
    color_distribution: Dict[int, int] = field(default_factory=dict)
    grid_size_distribution: Dict[tuple, int] = field(default_factory=dict)
    program_complexity_distribution: Dict[int, int] = field(default_factory=dict)
    pairs_per_program: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'total_pairs': self.total_pairs,
            'total_tasks': self.total_tasks,
            'unique_programs': self.unique_programs,
            'color_distribution': self.color_distribution,
            'grid_size_distribution': {str(k): v for k, v in self.grid_size_distribution.items()},
            'program_complexity_distribution': self.program_complexity_distribution,
            'pairs_per_program': self.pairs_per_program
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetStatistics':
        """辞書から復元"""
        stats = cls()
        stats.total_pairs = data.get('total_pairs', 0)
        stats.total_tasks = data.get('total_tasks', 0)
        stats.unique_programs = data.get('unique_programs', 0)
        stats.color_distribution = data.get('color_distribution', {})
        stats.grid_size_distribution = {
            tuple(eval(k)): v for k, v in data.get('grid_size_distribution', {}).items()
        }
        stats.program_complexity_distribution = data.get('program_complexity_distribution', {})
        stats.pairs_per_program = data.get('pairs_per_program', {})
        return stats
    
    def update_from_pairs(self, data_pairs: List[Any]):
        """DataPairのリストから統計を更新"""
        self.total_pairs = len(data_pairs)
        
        # プログラム統計
        programs = set()
        for pair in data_pairs:
            programs.add(pair.program)
            complexity = pair.get_program_complexity()
            self.program_complexity_distribution[complexity] = self.program_complexity_distribution.get(complexity, 0) + 1
        
        self.unique_programs = len(programs)
        
        # プログラムごとのペア数
        program_counts = defaultdict(int)
        for pair in data_pairs:
            program_counts[pair.program] += 1
        self.pairs_per_program = dict(program_counts)
        
        # 色分布
        for pair in data_pairs:
            color_dist = pair.get_color_distribution()
            for color, count in color_dist.items():
                self.color_distribution[color] = self.color_distribution.get(color, 0) + count
        
        # グリッドサイズ分布
        for pair in data_pairs:
            size = pair.get_grid_size()
            self.grid_size_distribution[size] = self.grid_size_distribution.get(size, 0) + 1
    
    def update_from_tasks(self, tasks: List[Any]):
        """Taskのリストから統計を更新"""
        self.total_tasks = len(tasks)
        
        # タスクの統計を更新
        for task in tasks:
            # 色分布
            color_dist = task.get_color_distribution()
            for color, count in color_dist.items():
                self.color_distribution[color] = self.color_distribution.get(color, 0) + count
            
            # グリッドサイズ分布
            min_size, max_size = task.get_grid_size_range()
            self.grid_size_distribution[min_size] = self.grid_size_distribution.get(min_size, 0) + 1
            self.grid_size_distribution[max_size] = self.grid_size_distribution.get(max_size, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """統計サマリーを取得"""
        return {
            'total_pairs': self.total_pairs,
            'total_tasks': self.total_tasks,
            'unique_programs': self.unique_programs,
            'color_diversity': len(self.color_distribution),
            'size_diversity': len(self.grid_size_distribution),
            'complexity_diversity': len(self.program_complexity_distribution),
            'avg_pairs_per_program': sum(self.pairs_per_program.values()) / len(self.pairs_per_program) if self.pairs_per_program else 0
        }
    
    def get_color_balance_score(self) -> float:
        """色のバランススコアを計算"""
        if not self.color_distribution:
            return 0.0
        
        values = list(self.color_distribution.values())
        if not values:
            return 0.0
        
        # 標準偏差が小さいほどバランスが良い
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # 0-1のスコアに正規化（標準偏差が小さいほど高いスコア）
        max_std = max(values) if values else 1
        return max(0, 1 - (std_dev / max_std))
    
    def get_complexity_distribution_score(self) -> float:
        """複雑度分布のスコアを計算"""
        if not self.program_complexity_distribution:
            return 0.0
        
        # 複雑度の多様性を評価
        complexity_count = len(self.program_complexity_distribution)
        max_complexity = max(self.program_complexity_distribution.keys()) if self.program_complexity_distribution else 0
        
        # 複雑度の範囲と多様性を考慮
        diversity_score = complexity_count / max(1, max_complexity + 1)
        
        return min(1.0, diversity_score)
    
    def get_overall_quality_score(self) -> float:
        """全体的な品質スコアを計算"""
        color_score = self.get_color_balance_score()
        complexity_score = self.get_complexity_distribution_score()
        
        # データ量のスコア
        data_score = min(1.0, (self.total_pairs + self.total_tasks) / 1000)
        
        return (color_score + complexity_score + data_score) / 3.0

