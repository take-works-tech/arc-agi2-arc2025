"""
プログラムプール管理

生成済みプログラムの管理と再利用
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import random
import uuid

from .generator import ProgramGenerator


class ProgramPoolManager:
    """生成済みプログラムの管理と再利用"""
    
    def __init__(self, seed: Optional[int] = None):
        """初期化"""
        self.rng = random.Random(seed)
        self.program_generator = ProgramGenerator()
        self.program_pool: List[str] = []
        self.program_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'usage_count': 0, 'complexity': 0, 'last_used': None})
        self.program_hashes: set = set()  # 重複チェック用
    
    def generate_program_pool(
        self,
        num_programs: int,
        complexity_distribution: Dict[int, float] = None,
        command_constraints: Optional[Dict[str, Any]] = None,
        max_attempts_per_program: int = 10
    ) -> List[str]:
        """プログラムプールを生成
        
        Args:
            num_programs: 生成するプログラム数
            complexity_distribution: 複雑度分布
            command_constraints: コマンド制約
            max_attempts_per_program: 各プログラムの最大試行回数
        
        Returns:
            生成されたプログラムのリスト
        """
        if complexity_distribution is None:
            complexity_distribution = {1: 0.6, 2: 0.3, 3: 0.1}  # デフォルト分布
        
        generated_count = 0
        while generated_count < num_programs:
            complexity = self.rng.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values()),
                k=1
            )[0]
            
            attempts = 0
            while attempts < max_attempts_per_program:
                program = self.program_generator.generate_program(complexity)
                program_hash = hash(program)
                
                if program and program_hash not in self.program_hashes:
                    self.program_pool.append(program)
                    self.program_hashes.add(program_hash)
                    self.program_stats[program]['complexity'] = complexity
                    generated_count += 1
                    break
                attempts += 1
            
            if attempts == max_attempts_per_program:
                print(f"警告: 複雑度 {complexity} のプログラム生成で重複回避に失敗しました。")
        
        print(f"プログラムプール生成完了: {len(self.program_pool)}個のユニークなプログラム")
        return self.program_pool
    
    def get_programs_for_phase1(
        self,
        num_programs: int,
        ensure_diversity: bool = True
    ) -> List[str]:
        """フェーズ1用のプログラムを取得
        
        Args:
            num_programs: 取得するプログラム数
            ensure_diversity: 多様性を確保するか
        
        Returns:
            プログラムのリスト
        """
        if len(self.program_pool) < num_programs:
            print(f"警告: プール内のプログラムが不足しています ({len(self.program_pool)}/{num_programs})。追加生成を試みます。")
            self.generate_program_pool(num_programs - len(self.program_pool))
        
        if ensure_diversity:
            # 使用頻度の低いプログラムを優先的に選択
            sorted_programs = sorted(self.program_pool, key=lambda p: self.program_stats[p]['usage_count'])
            selected_programs = sorted_programs[:num_programs]
        else:
            selected_programs = self.rng.sample(self.program_pool, min(num_programs, len(self.program_pool)))
        
        for p in selected_programs:
            self.program_stats[p]['usage_count'] += 1
            self.program_stats[p]['last_used'] = uuid.uuid4()  # タイムスタンプの代わりにユニークIDで更新順を追跡
        
        return selected_programs
    
    def filter_programs_with_sufficient_pairs(
        self,
        data_pairs: List[Any],
        min_pairs: int = 4
    ) -> List[str]:
        """十分なペアが生成されたプログラムのみを抽出
        
        Args:
            data_pairs: DataPairのリスト
            min_pairs: 最小ペア数
        
        Returns:
            十分なペアを持つプログラムのリスト
        """
        pair_counts = defaultdict(int)
        for pair in data_pairs:
            pair_counts[pair.program] += 1
        
        sufficient_programs = [
            program for program, count in pair_counts.items()
            if count >= min_pairs
        ]
        print(f"十分なペアを持つプログラム数: {len(sufficient_programs)}")
        return sufficient_programs
    
    def get_program_stats(self, program: str) -> Dict[str, Any]:
        """特定のプログラムの統計情報を取得
        
        Args:
            program: プログラム
        
        Returns:
            統計情報
        """
        return self.program_stats.get(program, {})
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """プール全体の統計情報を取得
        
        Returns:
            統計情報
        """
        total_usage = sum(s['usage_count'] for s in self.program_stats.values())
        return {
            'total_unique_programs': len(self.program_pool),
            'total_program_usages': total_usage,
            'average_complexity': sum(s['complexity'] for s in self.program_stats.values()) / len(self.program_pool) if self.program_pool else 0,
            'program_complexity_distribution': {k: v for k, v in Counter(s['complexity'] for s in self.program_stats.values()).items()}
        }

