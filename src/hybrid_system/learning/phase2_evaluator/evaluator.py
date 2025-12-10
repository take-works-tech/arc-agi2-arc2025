"""
フェーズ2評価器

汎化能力評価の統合評価器
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from core.data_structures import Task
from .generalization import GeneralizationEvaluator, GeneralizationConfig


@dataclass
class Phase2EvaluationConfig:
    """フェーズ2評価設定"""
    generalization_config: GeneralizationConfig = None
    enable_detailed_analysis: bool = True
    save_evaluation_results: bool = True
    output_dir: str = "evaluation_results"
    
    def __post_init__(self):
        if self.generalization_config is None:
            self.generalization_config = GeneralizationConfig()


class Phase2Evaluator:
    """フェーズ2評価器"""
    
    def __init__(self, config: Optional[Phase2EvaluationConfig] = None):
        """初期化"""
        self.config = config or Phase2EvaluationConfig()
        self.generalization_evaluator = GeneralizationEvaluator(self.config.generalization_config)
        
        # 評価履歴
        self.evaluation_history = []
    
    def evaluate_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """タスクのリストを評価
        
        Args:
            tasks: 評価するタスクのリスト
        
        Returns:
            評価結果
        """
        print(f"フェーズ2評価開始: {len(tasks)}タスク")
        
        evaluation_results = []
        
        for i, task in enumerate(tasks):
            if i % 10 == 0:
                print(f"進捗: {i}/{len(tasks)} タスク評価済み")
            
            # 個別タスクの評価
            task_result = self.generalization_evaluator.evaluate_task(task)
            evaluation_results.append(task_result)
        
        # 全体の統計を計算
        overall_stats = self._calculate_overall_statistics(evaluation_results)
        
        # 結果をまとめる
        result = {
            'total_tasks': len(tasks),
            'evaluation_results': evaluation_results,
            'overall_statistics': overall_stats,
            'passed_tasks': [r for r in evaluation_results if r['overall_passed']],
            'failed_tasks': [r for r in evaluation_results if not r['overall_passed']]
        }
        
        self.evaluation_history.append(result)
        
        print("フェーズ2評価完了")
        return result
    
    def evaluate_single_task(self, task: Task) -> Dict[str, Any]:
        """単一タスクを評価
        
        Args:
            task: 評価するタスク
        
        Returns:
            評価結果
        """
        return self.generalization_evaluator.evaluate_task(task)
    
    def _calculate_overall_statistics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体の統計を計算
        
        Args:
            evaluation_results: 評価結果のリスト
        
        Returns:
            統計情報
        """
        if not evaluation_results:
            return {}
        
        # 基本統計
        total_tasks = len(evaluation_results)
        passed_tasks = sum(1 for r in evaluation_results if r['overall_passed'])
        failed_tasks = total_tasks - passed_tasks
        
        # スコア統計
        overall_scores = [r['overall_score'] for r in evaluation_results]
        test_range_scores = [r['test_range_inclusion']['score'] for r in evaluation_results]
        pattern_consistency_scores = [r['pattern_consistency']['score'] for r in evaluation_results]
        
        # 平均スコア
        avg_overall_score = np.mean(overall_scores)
        avg_test_range_score = np.mean(test_range_scores)
        avg_pattern_consistency_score = np.mean(pattern_consistency_scores)
        
        # スコア分布
        score_distribution = {
            'excellent': sum(1 for s in overall_scores if s >= 0.9),
            'good': sum(1 for s in overall_scores if 0.7 <= s < 0.9),
            'fair': sum(1 for s in overall_scores if 0.5 <= s < 0.7),
            'poor': sum(1 for s in overall_scores if s < 0.5)
        }
        
        # 失敗理由の分析
        failure_reasons = {}
        for result in evaluation_results:
            if not result['overall_passed']:
                if not result['test_range_inclusion']['passed']:
                    failure_reasons['test_range_failure'] = failure_reasons.get('test_range_failure', 0) + 1
                if not result['pattern_consistency']['passed']:
                    failure_reasons['pattern_consistency_failure'] = failure_reasons.get('pattern_consistency_failure', 0) + 1
        
        return {
            'total_tasks': total_tasks,
            'passed_tasks': passed_tasks,
            'failed_tasks': failed_tasks,
            'pass_rate': passed_tasks / total_tasks if total_tasks > 0 else 0.0,
            'average_scores': {
                'overall': avg_overall_score,
                'test_range_inclusion': avg_test_range_score,
                'pattern_consistency': avg_pattern_consistency_score
            },
            'score_distribution': score_distribution,
            'failure_reasons': failure_reasons,
            'score_statistics': {
                'overall_scores': {
                    'min': np.min(overall_scores),
                    'max': np.max(overall_scores),
                    'std': np.std(overall_scores)
                },
                'test_range_scores': {
                    'min': np.min(test_range_scores),
                    'max': np.max(test_range_scores),
                    'std': np.std(test_range_scores)
                },
                'pattern_consistency_scores': {
                    'min': np.min(pattern_consistency_scores),
                    'max': np.max(pattern_consistency_scores),
                    'std': np.std(pattern_consistency_scores)
                }
            }
        }
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """評価履歴を取得
        
        Returns:
            評価履歴のリスト
        """
        return self.evaluation_history
    
    def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        """最新の評価結果を取得
        
        Returns:
            最新の評価結果（履歴がない場合はNone）
        """
        if self.evaluation_history:
            return self.evaluation_history[-1]
        return None
    
    def compare_evaluations(self, evaluation1: Dict[str, Any], evaluation2: Dict[str, Any]) -> Dict[str, Any]:
        """2つの評価結果を比較
        
        Args:
            evaluation1: 最初の評価結果
            evaluation2: 2番目の評価結果
        
        Returns:
            比較結果
        """
        stats1 = evaluation1.get('overall_statistics', {})
        stats2 = evaluation2.get('overall_statistics', {})
        
        comparison = {
            'pass_rate_change': stats2.get('pass_rate', 0) - stats1.get('pass_rate', 0),
            'overall_score_change': stats2.get('average_scores', {}).get('overall', 0) - stats1.get('average_scores', {}).get('overall', 0),
            'test_range_score_change': stats2.get('average_scores', {}).get('test_range_inclusion', 0) - stats1.get('average_scores', {}).get('test_range_inclusion', 0),
            'pattern_consistency_score_change': stats2.get('average_scores', {}).get('pattern_consistency', 0) - stats1.get('average_scores', {}).get('pattern_consistency', 0),
            'task_count_change': stats2.get('total_tasks', 0) - stats1.get('total_tasks', 0)
        }
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_result: Dict[str, Any]) -> str:
        """評価レポートを生成
        
        Args:
            evaluation_result: 評価結果
        
        Returns:
            レポート文字列
        """
        stats = evaluation_result.get('overall_statistics', {})
        
        report = f"""
=== フェーズ2評価レポート ===

総タスク数: {stats.get('total_tasks', 0)}
合格タスク数: {stats.get('passed_tasks', 0)}
不合格タスク数: {stats.get('failed_tasks', 0)}
合格率: {stats.get('pass_rate', 0):.2%}

平均スコア:
- 総合スコア: {stats.get('average_scores', {}).get('overall', 0):.3f}
- テスト範囲包含性: {stats.get('average_scores', {}).get('test_range_inclusion', 0):.3f}
- パターン一貫性: {stats.get('average_scores', {}).get('pattern_consistency', 0):.3f}

スコア分布:
- 優秀 (0.9以上): {stats.get('score_distribution', {}).get('excellent', 0)}タスク
- 良好 (0.7-0.9): {stats.get('score_distribution', {}).get('good', 0)}タスク
- 普通 (0.5-0.7): {stats.get('score_distribution', {}).get('fair', 0)}タスク
- 不良 (0.5未満): {stats.get('score_distribution', {}).get('poor', 0)}タスク

失敗理由:
"""
        
        failure_reasons = stats.get('failure_reasons', {})
        for reason, count in failure_reasons.items():
            report += f"- {reason}: {count}タスク\n"
        
        return report

