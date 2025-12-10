#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合設定ファイル
ARC-AGI2タスク用の全設定を統合管理
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

# =============================================================================
# プログラム実行設定
# =============================================================================

@dataclass
class ExecutionConfig:
    """プログラム実行設定"""
    
    # 実行環境設定
    max_execution_time: float = 30.0  # seconds
    max_memory_usage: int = 1024  # MB
    enable_parallel_execution: bool = True
    max_parallel_threads: int = 4
    
    # グリッド設定
    max_grid_size: int = 30
    default_background_color: int = 0
    enable_grid_validation: bool = True
    
    # オブジェクト設定
    max_objects_per_execution: int = 100
    enable_object_validation: bool = True
    object_type_validation: bool = True
    
    # 操作設定
    enable_operation_validation: bool = True
    operation_timeout: float = 5.0  # seconds per operation
    max_operation_retries: int = 3
    
    # レイヤー設定
    max_layers: int = 100
    min_layers: int = 0
    default_layer: int = 1
    enable_layer_validation: bool = True
    
    # エラーハンドリング設定
    enable_error_recovery: bool = True
    error_recovery_attempts: int = 3
    fallback_to_simple_execution: bool = True
    
    # ログ設定
    enable_execution_logging: bool = True
    log_level: str = "INFO"
    save_execution_traces: bool = False
    trace_output_dir: str = "execution_traces"
    
    # デバッグ設定
    enable_debug_mode: bool = False
    debug_step_by_step: bool = False
    debug_output_dir: str = "debug_execution"
    
    # パフォーマンス設定
    enable_performance_monitoring: bool = True
    performance_metrics_interval: float = 1.0  # seconds
    memory_check_interval: float = 0.5  # seconds
    
    # セキュリティ設定
    enable_sandbox_mode: bool = True
    max_operations_per_program: int = 1000
    dangerous_operations: List[str] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.dangerous_operations is None:
            self.dangerous_operations = [
                'SYSTEM_CALL',
                'FILE_ACCESS',
                'NETWORK_ACCESS',
                'EXECUTE_COMMAND'
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得"""
        return {
            'max_execution_time': self.max_execution_time,
            'max_memory_usage': self.max_memory_usage,
            'enable_parallel_execution': self.enable_parallel_execution,
            'max_parallel_threads': self.max_parallel_threads,
            'max_grid_size': self.max_grid_size,
            'default_background_color': self.default_background_color,
            'enable_grid_validation': self.enable_grid_validation,
            'max_objects_per_execution': self.max_objects_per_execution,
            'enable_object_validation': self.enable_object_validation,
            'object_type_validation': self.object_type_validation,
            'enable_operation_validation': self.enable_operation_validation,
            'operation_timeout': self.operation_timeout,
            'max_operation_retries': self.max_operation_retries,
            'max_layers': self.max_layers,
            'default_layer': self.default_layer,
            'enable_layer_validation': self.enable_layer_validation,
            'enable_error_recovery': self.enable_error_recovery,
            'error_recovery_attempts': self.error_recovery_attempts,
            'fallback_to_simple_execution': self.fallback_to_simple_execution,
            'enable_execution_logging': self.enable_execution_logging,
            'log_level': self.log_level,
            'save_execution_traces': self.save_execution_traces,
            'trace_output_dir': self.trace_output_dir,
            'enable_debug_mode': self.enable_debug_mode,
            'debug_step_by_step': self.debug_step_by_step,
            'debug_output_dir': self.debug_output_dir,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'performance_metrics_interval': self.performance_metrics_interval,
            'memory_check_interval': self.memory_check_interval,
            'enable_sandbox_mode': self.enable_sandbox_mode,
            'max_operations_per_program': self.max_operations_per_program,
            'dangerous_operations': self.dangerous_operations
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExecutionConfig':
        """辞書から設定を作成"""
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []
        
        if self.max_execution_time <= 0:
            errors.append("max_execution_time must be positive")
        
        if self.max_memory_usage <= 0:
            errors.append("max_memory_usage must be positive")
        
        if self.max_parallel_threads <= 0:
            errors.append("max_parallel_threads must be positive")
        
        if self.max_grid_size <= 0:
            errors.append("max_grid_size must be positive")
        
        if self.max_objects_per_execution <= 0:
            errors.append("max_objects_per_execution must be positive")
        
        if self.operation_timeout <= 0:
            errors.append("operation_timeout must be positive")
        
        if self.max_operation_retries < 0:
            errors.append("max_operation_retries must be non-negative")
        
        if self.max_layers <= 0:
            errors.append("max_layers must be positive")
        
        if self.default_layer < 0:
            errors.append("default_layer must be non-negative")
        
        if self.error_recovery_attempts < 0:
            errors.append("error_recovery_attempts must be non-negative")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        if self.performance_metrics_interval <= 0:
            errors.append("performance_metrics_interval must be positive")
        
        if self.memory_check_interval <= 0:
            errors.append("memory_check_interval must be positive")
        
        if self.max_operations_per_program <= 0:
            errors.append("max_operations_per_program must be positive")
        
        return errors

# =============================================================================
# オブジェクト抽出設定
# =============================================================================

@dataclass
class ExtractionConfig:
    """オブジェクト抽出設定"""
    
    # 基本抽出パラメータ
    min_object_size: int = 1
    max_object_size: int = 100
    min_pixel_count: int = 1
    max_pixel_count: int = 1000
    min_area: int = 0
    
    # オブジェクトタイプ抽出設定
    enable_single_color_4way: bool = True
    enable_single_color_8way: bool = True
    # enable_multi_color_4way: bool = False  # 無効化
    # enable_multi_color_8way: bool = False  # 無効化
    enable_whole_grid: bool = True
    
    # 重複除去設定
    enable_duplicate_removal: bool = True
    duplicate_overlap_threshold: float = 0.8
    duplicate_size_threshold: float = 0.1
    
    # 品質フィルタリング設定
    enable_quality_filtering: bool = True
    min_quality_score: float = 0.3
    max_aspect_ratio: float = 10.0
    min_compactness: float = 0.1
    
    # 色処理設定
    enable_color_normalization: bool = True
    color_tolerance: int = 0
    background_color_tolerance: int = 0
    
    # 形状処理設定
    enable_shape_analysis: bool = True
    enable_contour_analysis: bool = True
    enable_symmetry_detection: bool = True
    enable_pattern_detection: bool = True
    
    # パフォーマンス設定
    enable_parallel_extraction: bool = True
    max_parallel_workers: int = 4
    extraction_timeout: float = 30.0
    
    # キャッシュ設定
    enable_extraction_cache: bool = True
    cache_size_limit: int = 1000
    cache_ttl: float = 3600.0  # seconds
    
    # デバッグ設定
    enable_extraction_debug: bool = False
    save_intermediate_results: bool = False
    debug_output_dir: str = "extraction_debug"
    
    # 統計設定
    enable_extraction_statistics: bool = True
    statistics_output_dir: str = "extraction_statistics"
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得"""
        return {
            'min_object_size': self.min_object_size,
            'max_object_size': self.max_object_size,
            'min_pixel_count': self.min_pixel_count,
            'max_pixel_count': self.max_pixel_count,
            'min_area': self.min_area,
            'enable_single_color_4way': self.enable_single_color_4way,
            'enable_single_color_8way': self.enable_single_color_8way,
            'enable_whole_grid': self.enable_whole_grid,
            'enable_duplicate_removal': self.enable_duplicate_removal,
            'duplicate_overlap_threshold': self.duplicate_overlap_threshold,
            'duplicate_size_threshold': self.duplicate_size_threshold,
            'enable_quality_filtering': self.enable_quality_filtering,
            'min_quality_score': self.min_quality_score,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_compactness': self.min_compactness,
            'enable_color_normalization': self.enable_color_normalization,
            'color_tolerance': self.color_tolerance,
            'background_color_tolerance': self.background_color_tolerance,
            'enable_shape_analysis': self.enable_shape_analysis,
            'enable_contour_analysis': self.enable_contour_analysis,
            'enable_symmetry_detection': self.enable_symmetry_detection,
            'enable_pattern_detection': self.enable_pattern_detection,
            'enable_parallel_extraction': self.enable_parallel_extraction,
            'max_parallel_workers': self.max_parallel_workers,
            'extraction_timeout': self.extraction_timeout,
            'enable_extraction_cache': self.enable_extraction_cache,
            'cache_size_limit': self.cache_size_limit,
            'cache_ttl': self.cache_ttl,
            'enable_extraction_debug': self.enable_extraction_debug,
            'save_intermediate_results': self.save_intermediate_results,
            'debug_output_dir': self.debug_output_dir,
            'enable_extraction_statistics': self.enable_extraction_statistics,
            'statistics_output_dir': self.statistics_output_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExtractionConfig':
        """辞書から設定を作成"""
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []
        
        if self.min_object_size <= 0:
            errors.append("min_object_size must be positive")
        
        if self.max_object_size <= self.min_object_size:
            errors.append("max_object_size must be greater than min_object_size")
        
        if self.min_pixel_count <= 0:
            errors.append("min_pixel_count must be positive")
        
        if self.max_pixel_count <= self.min_pixel_count:
            errors.append("max_pixel_count must be greater than min_pixel_count")
        
        if self.min_area < 0:
            errors.append("min_area must be non-negative")
        
        if not 0.0 <= self.duplicate_overlap_threshold <= 1.0:
            errors.append("duplicate_overlap_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.duplicate_size_threshold <= 1.0:
            errors.append("duplicate_size_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.min_quality_score <= 1.0:
            errors.append("min_quality_score must be between 0.0 and 1.0")
        
        if self.max_aspect_ratio <= 1.0:
            errors.append("max_aspect_ratio must be greater than 1.0")
        
        if not 0.0 <= self.min_compactness <= 1.0:
            errors.append("min_compactness must be between 0.0 and 1.0")
        
        if self.max_parallel_workers <= 0:
            errors.append("max_parallel_workers must be positive")
        
        if self.extraction_timeout <= 0:
            errors.append("extraction_timeout must be positive")
        
        if self.cache_size_limit <= 0:
            errors.append("cache_size_limit must be positive")
        
        if self.cache_ttl <= 0:
            errors.append("cache_ttl must be positive")
        
        return errors

# =============================================================================
# デフォルト設定
# =============================================================================

# 実行設定のデフォルト
DEFAULT_EXECUTION_CONFIG = ExecutionConfig()

# 高精度実行設定
HIGH_ACCURACY_EXECUTION_CONFIG = ExecutionConfig(
    max_execution_time=60.0,
    max_memory_usage=2048,
    enable_object_validation=True,
    enable_operation_validation=True,
    enable_layer_validation=True,
    enable_error_recovery=True,
    error_recovery_attempts=5,
    enable_performance_monitoring=True
)

# 高速実行設定
FAST_EXECUTION_CONFIG = ExecutionConfig(
    max_execution_time=10.0,
    max_memory_usage=512,
    enable_object_validation=False,
    enable_operation_validation=False,
    enable_layer_validation=False,
    enable_error_recovery=False,
    error_recovery_attempts=1,
    enable_performance_monitoring=False,
    fallback_to_simple_execution=True
)

# バランス実行設定
BALANCED_EXECUTION_CONFIG = ExecutionConfig(
    max_execution_time=30.0,
    max_memory_usage=1024,
    enable_object_validation=True,
    enable_operation_validation=True,
    enable_layer_validation=True,
    enable_error_recovery=True,
    error_recovery_attempts=3,
    enable_performance_monitoring=True
)

# 抽出設定のデフォルト
DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()

# 高精度抽出設定
HIGH_ACCURACY_EXTRACTION_CONFIG = ExtractionConfig(
    enable_quality_filtering=True,
    min_quality_score=0.5,
    enable_shape_analysis=True,
    enable_contour_analysis=True,
    enable_symmetry_detection=True,
    enable_pattern_detection=True,
    enable_extraction_cache=True,
    enable_extraction_statistics=True
)

# 高速抽出設定
FAST_EXTRACTION_CONFIG = ExtractionConfig(
    enable_quality_filtering=False,
    enable_shape_analysis=False,
    enable_contour_analysis=False,
    enable_symmetry_detection=False,
    enable_pattern_detection=False,
    enable_extraction_cache=False,
    enable_extraction_statistics=False,
    enable_parallel_extraction=True,
    max_parallel_workers=8
)

# バランス抽出設定
BALANCED_EXTRACTION_CONFIG = ExtractionConfig(
    enable_quality_filtering=True,
    min_quality_score=0.3,
    enable_shape_analysis=True,
    enable_contour_analysis=True,
    enable_symmetry_detection=False,
    enable_pattern_detection=False,
    enable_extraction_cache=True,
    enable_extraction_statistics=True
)
