#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品質評価システム

オブジェクト抽出、プログラム合成、実行結果の品質を包括的に評価
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import time
import cv2

import logging
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
from src.data_systems.data_models.data.input_data import TaskData, InputDataType

logger = logging.getLogger("QualityEvaluator")

@dataclass
class ExtractionQualityMetrics:
    """オブジェクト抽出品質メトリクス"""
    accuracy: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    efficiency: float = 0.0
    overall_score: float = 0.0
    
    def calculate_overall_score(self, weights: Dict[str, float] = None) -> float:
        """総合スコアを計算"""
        if weights is None:
            weights = {
                'accuracy': 0.3,
                'completeness': 0.3,
                'consistency': 0.2,
                'efficiency': 0.2
            }
        
        self.overall_score = (
            weights['accuracy'] * self.accuracy +
            weights['completeness'] * self.completeness +
            weights['consistency'] * self.consistency +
            weights['efficiency'] * self.efficiency
        )
        
        return self.overall_score

@dataclass
class SynthesisQualityMetrics:
    """プログラム合成品質メトリクス"""
    correctness: float = 0.0
    efficiency: float = 0.0
    readability: float = 0.0
    maintainability: float = 0.0
    overall_score: float = 0.0
    
    def calculate_overall_score(self, weights: Dict[str, float] = None) -> float:
        """総合スコアを計算"""
        if weights is None:
            weights = {
                'correctness': 0.4,
                'efficiency': 0.3,
                'readability': 0.2,
                'maintainability': 0.1
            }
        
        self.overall_score = (
            weights['correctness'] * self.correctness +
            weights['efficiency'] * self.efficiency +
            weights['readability'] * self.readability +
            weights['maintainability'] * self.maintainability
        )
        
        return self.overall_score

@dataclass
class ExecutionQualityMetrics:
    """実行品質メトリクス"""
    accuracy: float = 0.0
    performance: float = 0.0
    robustness: float = 0.0
    resource_usage: float = 0.0
    overall_score: float = 0.0
    
    def calculate_overall_score(self, weights: Dict[str, float] = None) -> float:
        """総合スコアを計算"""
        if weights is None:
            weights = {
                'accuracy': 0.4,
                'performance': 0.3,
                'robustness': 0.2,
                'resource_usage': 0.1
            }
        
        self.overall_score = (
            weights['accuracy'] * self.accuracy +
            weights['performance'] * self.performance +
            weights['robustness'] * self.robustness +
            weights['resource_usage'] * self.resource_usage
        )
        
        return self.overall_score

@dataclass
class OverallQualityReport:
    """総合品質レポート"""
    extraction_metrics: ExtractionQualityMetrics
    synthesis_metrics: SynthesisQualityMetrics
    execution_metrics: ExecutionQualityMetrics
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self, weights: Dict[str, float] = None) -> float:
        """総合スコアを計算"""
        if weights is None:
            weights = {
                'extraction': 0.3,
                'synthesis': 0.4,
                'execution': 0.3
            }
        
        self.overall_score = (
            weights['extraction'] * self.extraction_metrics.overall_score +
            weights['synthesis'] * self.synthesis_metrics.overall_score +
            weights['execution'] * self.execution_metrics.overall_score
        )
        
        return self.overall_score

class QualityEvaluator:
    """品質評価システム"""
    
    def __init__(self):
        self.evaluation_history: List[OverallQualityReport] = []
        logger.info("品質評価システム初期化完了")
    
    def evaluate_extraction_quality(self, original_grid: np.ndarray, 
                                  extracted_objects: List[Object]) -> ExtractionQualityMetrics:
        """オブジェクト抽出の品質を評価"""
        try:
            metrics = ExtractionQualityMetrics()
            
            # 1. 精度評価
            metrics.accuracy = self._evaluate_extraction_accuracy(original_grid, extracted_objects)
            
            # 2. 完全性評価
            metrics.completeness = self._evaluate_extraction_completeness(original_grid, extracted_objects)
            
            # 3. 一貫性評価
            metrics.consistency = self._evaluate_extraction_consistency(extracted_objects)
            
            # 4. 効率性評価
            metrics.efficiency = self._evaluate_extraction_efficiency(original_grid, extracted_objects)
            
            # 総合スコアを計算
            metrics.calculate_overall_score()
            
            logger.info(f"オブジェクト抽出品質評価完了 - 総合スコア: {metrics.overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"オブジェクト抽出品質評価エラー: {e}")
            return ExtractionQualityMetrics()
    
    def _evaluate_extraction_accuracy(self, original_grid: np.ndarray, 
                                    extracted_objects: List[Object]) -> float:
        """抽出精度を評価"""
        try:
            if not extracted_objects:
                return 0.0
            
            # オブジェクトからグリッドを再構築
            reconstructed_grid = self._reconstruct_grid_from_objects(extracted_objects, original_grid.shape)
            
            # ピクセル単位での一致率
            matches = np.sum(original_grid == reconstructed_grid)
            total_pixels = original_grid.size
            
            accuracy = matches / total_pixels if total_pixels > 0 else 0.0
            
            return accuracy
            
        except Exception as e:
            logger.error(f"抽出精度評価エラー: {e}")
            return 0.0
    
    def _evaluate_extraction_completeness(self, original_grid: np.ndarray, 
                                        extracted_objects: List[Object]) -> float:
        """抽出完全性を評価"""
        try:
            if not extracted_objects:
                return 0.0
            
            # 非背景色のピクセルを特定
            background_color = self._determine_background_color(original_grid)
            non_background_pixels = np.sum(original_grid != background_color)
            
            # 抽出されたオブジェクトのピクセル数
            extracted_pixels = sum(obj.pixel_count for obj in extracted_objects)
            
            # 完全性 = 抽出されたピクセル数 / 非背景色ピクセル数
            completeness = extracted_pixels / non_background_pixels if non_background_pixels > 0 else 1.0
            
            return min(1.0, completeness)
            
        except Exception as e:
            logger.error(f"抽出完全性評価エラー: {e}")
            return 0.0
    
    def _evaluate_extraction_consistency(self, extracted_objects: List[Object]) -> float:
        """抽出一貫性を評価"""
        try:
            if not extracted_objects:
                return 0.0
            
            consistency_scores = []
            
            for obj in extracted_objects:
                # オブジェクトの内部一貫性
                internal_consistency = self._evaluate_object_internal_consistency(obj)
                consistency_scores.append(internal_consistency)
            
            # オブジェクト間の一貫性
            inter_consistency = self._evaluate_inter_object_consistency(extracted_objects)
            
            # 総合一貫性
            overall_consistency = (np.mean(consistency_scores) + inter_consistency) / 2.0
            
            return overall_consistency
            
        except Exception as e:
            logger.error(f"抽出一貫性評価エラー: {e}")
            return 0.0
    
    def _evaluate_extraction_efficiency(self, original_grid: np.ndarray, 
                                      extracted_objects: List[Object]) -> float:
        """抽出効率性を評価"""
        try:
            if not extracted_objects:
                return 0.0
            
            # オブジェクト数とグリッドサイズの比率
            grid_size = original_grid.size
            object_count = len(extracted_objects)
            
            # 適切なオブジェクト数（グリッドサイズの平方根程度）
            optimal_object_count = int(np.sqrt(grid_size))
            
            # 効率性 = 1 - |実際のオブジェクト数 - 最適オブジェクト数| / 最適オブジェクト数
            efficiency = 1.0 - abs(object_count - optimal_object_count) / max(1, optimal_object_count)
            
            return max(0.0, efficiency)
            
        except Exception as e:
            logger.error(f"抽出効率性評価エラー: {e}")
            return 0.0
    
    def _evaluate_object_internal_consistency(self, obj: Object) -> float:
        """オブジェクトの内部一貫性を評価"""
        try:
            # 色の一貫性
            color_consistency = 1.0 if len(obj.color_ratio) == 1 else 0.5
            
            # 形状の一貫性
            shape_consistency = 1.0 if obj.bbox_width > 0 and obj.bbox_height > 0 else 0.0
            
            # 位置の一貫性
            position_consistency = 1.0 if obj.pixels else 0.0
            
            return (color_consistency + shape_consistency + position_consistency) / 3.0
            
        except Exception as e:
            logger.error(f"オブジェクト内部一貫性評価エラー: {e}")
            return 0.0
    
    def _evaluate_inter_object_consistency(self, objects: List[Object]) -> float:
        """オブジェクト間の一貫性を評価"""
        try:
            if len(objects) < 2:
                return 1.0
            
            consistency_scores = []
            
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    obj1, obj2 = objects[i], objects[j]
                    
                    # サイズの一貫性
                    size_ratio = min(obj1.pixel_count, obj2.pixel_count) / max(obj1.pixel_count, obj2.pixel_count)
                    size_consistency = size_ratio
                    
                    # 形状の一貫性
                    shape_consistency = 1.0 if obj1.bbox_width == obj2.bbox_width and obj1.bbox_height == obj2.bbox_height else 0.5
                    
                    # 色の一貫性
                    color_consistency = 1.0 if obj1.dominant_color == obj2.dominant_color else 0.0
                    
                    pair_consistency = (size_consistency + shape_consistency + color_consistency) / 3.0
                    consistency_scores.append(pair_consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception as e:
            logger.error(f"オブジェクト間一貫性評価エラー: {e}")
            return 0.0
    
    def evaluate_synthesis_quality(self, program, input_objects: List[Object], 
                                 expected_output: np.ndarray) -> SynthesisQualityMetrics:
        """プログラム合成の品質を評価"""
        try:
            metrics = SynthesisQualityMetrics()
            
            # 1. 正確性評価
            metrics.correctness = self._evaluate_synthesis_correctness(program, input_objects, expected_output)
            
            # 2. 効率性評価
            metrics.efficiency = self._evaluate_synthesis_efficiency(program)
            
            # 3. 可読性評価
            metrics.readability = self._evaluate_synthesis_readability(program)
            
            # 4. 保守性評価
            metrics.maintainability = self._evaluate_synthesis_maintainability(program)
            
            # 総合スコアを計算
            metrics.calculate_overall_score()
            
            logger.info(f"プログラム合成品質評価完了 - 総合スコア: {metrics.overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"プログラム合成品質評価エラー: {e}")
            return SynthesisQualityMetrics()
    
    def _evaluate_synthesis_correctness(self, program, input_objects: List[Object], 
                                      expected_output: np.ndarray) -> float:
        """合成正確性を評価"""
        try:
            # プログラムを実行して結果を取得            executor = ProgramExecutor()
            result_grid, result_objects, execution_cost = executor.execute_program(
                program, expected_output, input_objects
            )
            
            # 期待出力との比較
            if result_grid.shape != expected_output.shape:
                return 0.0
            
            matches = np.sum(result_grid == expected_output)
            total_pixels = expected_output.size
            
            accuracy = matches / total_pixels if total_pixels > 0 else 0.0
            
            return accuracy
            
        except Exception as e:
            logger.error(f"合成正確性評価エラー: {e}")
            return 0.0
    
    def _evaluate_synthesis_efficiency(self, program) -> float:
        """合成効率性を評価"""
        try:
            # プログラムのコストを評価
            total_cost = program.total_cost if hasattr(program, 'total_cost') else 0.0
            
            # 操作数
            operation_count = len(program.operations) if hasattr(program, 'operations') else 0
            
            # 効率性 = 1 / (1 + コスト + 操作数)
            efficiency = 1.0 / (1.0 + total_cost + operation_count * 0.1)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"合成効率性評価エラー: {e}")
            return 0.0
    
    def _evaluate_synthesis_readability(self, program) -> float:
        """合成可読性を評価"""
        try:
            if not hasattr(program, 'operations'):
                return 0.0
            
            readability_scores = []
            
            for operation in program.operations:
                # 操作の複雑さ
                complexity = self._evaluate_operation_complexity(operation)
                
                # 可読性 = 1 - 複雑さ
                readability = 1.0 - complexity
                readability_scores.append(readability)
            
            return np.mean(readability_scores) if readability_scores else 0.0
            
        except Exception as e:
            logger.error(f"合成可読性評価エラー: {e}")
            return 0.0
    
    def _evaluate_synthesis_maintainability(self, program) -> float:
        """合成保守性を評価"""
        try:
            if not hasattr(program, 'operations'):
                return 0.0
            
            # 操作の種類数
            operation_types = set()
            for operation in program.operations:
                if isinstance(operation, dict) and 'type' in operation:
                    operation_types.add(operation['type'])
            
            # 保守性 = 1 - (操作種類数 / 10)
            maintainability = 1.0 - (len(operation_types) / 10.0)
            
            return max(0.0, maintainability)
            
        except Exception as e:
            logger.error(f"合成保守性評価エラー: {e}")
            return 0.0
    
    def _evaluate_operation_complexity(self, operation: Dict[str, Any]) -> float:
        """操作の複雑さを評価"""
        try:
            complexity = 0.0
            
            # 操作タイプの複雑さ
            operation_type = operation.get('type', '')
            if operation_type in ['FOR', 'IF']:
                complexity += 0.3
            elif operation_type in ['FILL']:
                complexity += 0.1
            
            # パラメータの複雑さ
            param_count = len(operation.get('params', {}))
            complexity += param_count * 0.1
            
            # ネストの複雑さ
            if 'operations' in operation:
                nested_operations = operation['operations']
                complexity += len(nested_operations) * 0.2
            
            return min(1.0, complexity)
            
        except Exception as e:
            logger.error(f"操作複雑さ評価エラー: {e}")
            return 0.0
    
    def evaluate_execution_quality(self, program, input_objects: List[Object], 
                                 expected_output: np.ndarray) -> ExecutionQualityMetrics:
        """実行品質を評価"""
        try:
            metrics = ExecutionQualityMetrics()
            
            # 1. 精度評価
            metrics.accuracy = self._evaluate_execution_accuracy(program, input_objects, expected_output)
            
            # 2. 性能評価
            metrics.performance = self._evaluate_execution_performance(program, input_objects)
            
            # 3. 堅牢性評価
            metrics.robustness = self._evaluate_execution_robustness(program, input_objects)
            
            # 4. リソース使用量評価
            metrics.resource_usage = self._evaluate_execution_resource_usage(program, input_objects)
            
            # 総合スコアを計算
            metrics.calculate_overall_score()
            
            logger.info(f"実行品質評価完了 - 総合スコア: {metrics.overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"実行品質評価エラー: {e}")
            return ExecutionQualityMetrics()
    
    def _evaluate_execution_accuracy(self, program, input_objects: List[Object], 
                                   expected_output: np.ndarray) -> float:
        """実行精度を評価"""
        try:            executor = ProgramExecutor()
            result_grid, result_objects, execution_cost = executor.execute_program(
                program, expected_output, input_objects
            )
            
            # 期待出力との比較
            if result_grid.shape != expected_output.shape:
                return 0.0
            
            matches = np.sum(result_grid == expected_output)
            total_pixels = expected_output.size
            
            accuracy = matches / total_pixels if total_pixels > 0 else 0.0
            
            return accuracy
            
        except Exception as e:
            logger.error(f"実行精度評価エラー: {e}")
            return 0.0
    
    def _evaluate_execution_performance(self, program, input_objects: List[Object]) -> float:
        """実行性能を評価"""
        try:
            start_time = time.time()            executor = ProgramExecutor()
            result_grid, result_objects, execution_cost = executor.execute_program(
                program, np.zeros((10, 10)), input_objects
            )
            
            execution_time = time.time() - start_time
            
            # 性能 = 1 / (1 + 実行時間)
            performance = 1.0 / (1.0 + execution_time)
            
            return performance
            
        except Exception as e:
            logger.error(f"実行性能評価エラー: {e}")
            return 0.0
    
    def _evaluate_execution_robustness(self, program, input_objects: List[Object]) -> float:
        """実行堅牢性を評価"""
        try:
            # 異なる入力での実行テスト
            test_cases = [
                np.zeros((5, 5)),
                np.ones((8, 8)),
                np.random.randint(0, 3, (6, 6))
            ]
            
            success_count = 0
            total_tests = len(test_cases)
            
            for test_input in test_cases:
                try:                    executor = ProgramExecutor()
                    result_grid, result_objects, execution_cost = executor.execute_program(
                        program, test_input, input_objects
                    )
                    success_count += 1
                    
                except Exception as test_error:
                    # 本格実装: エラーログと詳細分析
                    logger.warning(f"堅牢性テストエラー: {test_error}")
                    error_analysis = self._analyze_test_error(test_error)
                    self._record_test_failure(error_analysis)
            
            robustness = success_count / total_tests if total_tests > 0 else 0.0
            
            return robustness
            
        except Exception as e:
            logger.error(f"実行堅牢性評価エラー: {e}")
            return 0.0
    
    def _evaluate_execution_resource_usage(self, program, input_objects: List[Object]) -> float:
        """実行リソース使用量を評価"""
        try:            executor = ProgramExecutor()
            result_grid, result_objects, execution_cost = executor.execute_program(
                program, np.zeros((10, 10)), input_objects
            )
            
            # リソース使用量 = 1 / (1 + 実行コスト)
            resource_usage = 1.0 / (1.0 + execution_cost)
            
            return resource_usage
            
        except Exception as e:
            logger.error(f"実行リソース使用量評価エラー: {e}")
            return 0.0
    
    def generate_overall_quality_report(self, extraction_metrics: ExtractionQualityMetrics,
                                      synthesis_metrics: SynthesisQualityMetrics,
                                      execution_metrics: ExecutionQualityMetrics) -> OverallQualityReport:
        """総合品質レポートを生成"""
        try:
            report = OverallQualityReport(
                extraction_metrics=extraction_metrics,
                synthesis_metrics=synthesis_metrics,
                execution_metrics=execution_metrics
            )
            
            # 総合スコアを計算
            report.calculate_overall_score()
            
            # 推奨事項を生成
            report.recommendations = self._generate_recommendations(report)
            
            # 履歴に追加
            self.evaluation_history.append(report)
            
            logger.info(f"総合品質レポート生成完了 - 総合スコア: {report.overall_score:.3f}")
            return report
            
        except Exception as e:
            logger.error(f"総合品質レポート生成エラー: {e}")
            return OverallQualityReport(
                extraction_metrics=ExtractionQualityMetrics(),
                synthesis_metrics=SynthesisQualityMetrics(),
                execution_metrics=ExecutionQualityMetrics()
            )
    
    def _generate_recommendations(self, report: OverallQualityReport) -> List[str]:
        """推奨事項を生成"""
        try:
            recommendations = []
            
            # オブジェクト抽出の推奨事項
            if report.extraction_metrics.overall_score < 0.7:
                recommendations.append("オブジェクト抽出の品質を向上させる必要があります")
            
            if report.extraction_metrics.accuracy < 0.8:
                recommendations.append("オブジェクト抽出の精度を改善してください")
            
            if report.extraction_metrics.completeness < 0.8:
                recommendations.append("オブジェクト抽出の完全性を向上させてください")
            
            # プログラム合成の推奨事項
            if report.synthesis_metrics.overall_score < 0.7:
                recommendations.append("プログラム合成の品質を向上させる必要があります")
            
            if report.synthesis_metrics.correctness < 0.8:
                recommendations.append("プログラム合成の正確性を改善してください")
            
            if report.synthesis_metrics.efficiency < 0.6:
                recommendations.append("プログラム合成の効率性を向上させてください")
            
            # 実行の推奨事項
            if report.execution_metrics.overall_score < 0.7:
                recommendations.append("実行品質を向上させる必要があります")
            
            if report.execution_metrics.accuracy < 0.8:
                recommendations.append("実行精度を改善してください")
            
            if report.execution_metrics.performance < 0.6:
                recommendations.append("実行性能を向上させてください")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"推奨事項生成エラー: {e}")
            return []
    
    def _reconstruct_grid_from_objects(self, objects: List[Object], grid_shape: Tuple[int, int]) -> np.ndarray:
        """オブジェクトからグリッドを再構築"""
        try:
            grid = np.zeros(grid_shape, dtype=int)
            
            for obj in objects:
                for pixel in obj.pixels:
                    x, y = pixel
                    if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
                        grid[x, y] = obj.dominant_color
            
            return grid
            
        except Exception as e:
            logger.error(f"グリッド再構築エラー: {e}")
            return np.zeros(grid_shape, dtype=int)
    
    def _determine_background_color(self, grid: np.ndarray) -> int:
        """背景色を決定"""
        try:
            unique, counts = np.unique(grid, return_counts=True)
            return unique[np.argmax(counts)]
        except Exception as e:
            logger.error(f"背景色決定エラー: {e}")
            return 0
    
    def _analyze_test_error(self, error: Exception) -> Dict[str, Any]:
        """テストエラーを分析"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # エラーの種類に応じて分析
            if "memory" in error_message.lower():
                return {
                    'type': 'memory',
                    'severity': 'high',
                    'category': 'memory_error',
                    'component': 'program_execution',
                    'suggestion': 'reduce_complexity'
                }
            elif "timeout" in error_message.lower() or "time" in error_message.lower():
                return {
                    'type': 'timeout',
                    'severity': 'medium',
                    'category': 'timeout_error',
                    'component': 'program_execution',
                    'suggestion': 'optimize_algorithm'
                }
            elif "shape" in error_message.lower() or "dimension" in error_message.lower():
                return {
                    'type': 'data',
                    'severity': 'medium',
                    'category': 'shape_error',
                    'component': 'data_processing',
                    'suggestion': 'validate_input'
                }
            elif "key" in error_message.lower() or "attribute" in error_message.lower():
                return {
                    'type': 'structure',
                    'severity': 'medium',
                    'category': 'structure_error',
                    'component': 'data_access',
                    'suggestion': 'check_structure'
                }
            else:
                return {
                    'type': 'general',
                    'severity': 'medium',
                    'category': 'unknown_error',
                    'component': 'unknown',
                    'suggestion': 'investigate'
                }
                
        except Exception as e:
            logger.error(f"エラー分析エラー: {e}")
            return {
                'type': 'analysis_error',
                'severity': 'high',
                'category': 'analysis_failure',
                'component': 'error_analysis',
                'suggestion': 'manual_investigation'
            }
    
    def _record_test_failure(self, error_analysis: Dict[str, Any]):
        """テスト失敗を記録"""
        try:
            # エラー統計を更新
            if not hasattr(self, '_error_stats'):
                self._error_stats = {}
            
            category = error_analysis['category']
            if category not in self._error_stats:
                self._error_stats[category] = {
                    'count': 0,
                    'severity_sum': 0,
                    'components': set()
                }
            
            self._error_stats[category]['count'] += 1
            self._error_stats[category]['severity_sum'] += self._get_severity_score(error_analysis['severity'])
            self._error_stats[category]['components'].add(error_analysis['component'])
            
            # ログに記録
            logger.info(f"テスト失敗記録: {category} - {error_analysis['component']} - {error_analysis['suggestion']}")
            
        except Exception as e:
            logger.error(f"テスト失敗記録エラー: {e}")
    
    def _get_severity_score(self, severity: str) -> int:
        """重要度スコアを取得"""
        severity_scores = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return severity_scores.get(severity, 2)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計を取得"""
        try:
            if not hasattr(self, '_error_stats') or not self._error_stats:
                return {'message': 'エラー統計なし'}
            
            stats = {}
            for category, data in self._error_stats.items():
                stats[category] = {
                    'count': data['count'],
                    'average_severity': data['severity_sum'] / data['count'],
                    'affected_components': list(data['components'])
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"エラー統計取得エラー: {e}")
            return {'error': str(e)}
