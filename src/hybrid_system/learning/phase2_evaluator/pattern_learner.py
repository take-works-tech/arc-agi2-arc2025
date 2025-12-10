#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高度なパターン学習システム
オブジェクト変換パターンの学習と一般化を実装
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import json

from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
import logging

logger = logging.getLogger("AdvancedPatternLearner")

@dataclass
class TransformationPattern:
    """変換パターン"""
    pattern_id: str
    pattern_type: str  # "color_change", "position_change", "size_change", "shape_change", "complex"
    input_conditions: Dict[str, Any]
    transformation_rules: List[Dict[str, Any]]
    confidence: float
    frequency: int
    examples: List[Dict[str, Any]]

@dataclass
class PatternLearningResult:
    """パターン学習結果"""
    patterns: List[TransformationPattern]
    pattern_statistics: Dict[str, Any]
    learning_quality: float
    generalization_capability: float

class AdvancedPatternLearner:
    """高度なパターン学習システム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初期化"""
        self.config = config or self._get_default_config()
        self.logger = logger
        
        # 学習されたパターン
        self.learned_patterns: Dict[str, TransformationPattern] = {}
        
        # パターン統計
        self.pattern_statistics = {
            'total_patterns': 0,
            'pattern_types': defaultdict(int),
            'average_confidence': 0.0,
            'learning_progress': 0.0
        }
        
        logger.info("高度なパターン学習システム初期化完了")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            'min_pattern_frequency': 1,  # 閾値を下げる
            'confidence_threshold': 0.5,  # 閾値を下げる
            'max_patterns': 1000,
            'enable_hierarchical_learning': True,
            'enable_abstract_patterns': True,
            'similarity_threshold': 0.6,  # 閾値を下げる
            'learning_rate': 0.1
        }
    
    def learn_from_correspondences(self, correspondences: List[Any]) -> PatternLearningResult:
        """対応関係からパターンを学習"""
        try:
            logger.info(f"パターン学習開始: {len(correspondences)}個の対応関係")
            
            # 1. 基本変換パターンを抽出
            basic_patterns = self._extract_basic_patterns(correspondences)
            logger.info(f"基本パターン抽出完了: {len(basic_patterns)}個")
            
            # 2. パターンを一般化
            generalized_patterns = self._generalize_patterns(basic_patterns)
            logger.info(f"パターン一般化完了: {len(generalized_patterns)}個")
            
            # 3. 階層的パターン学習
            if self.config.get('enable_hierarchical_learning', True):
                hierarchical_patterns = self._learn_hierarchical_patterns(generalized_patterns)
                logger.info(f"階層的パターン学習完了: {len(hierarchical_patterns)}個")
            else:
                hierarchical_patterns = generalized_patterns
                logger.info(f"階層的パターン学習スキップ: {len(hierarchical_patterns)}個")
            
            # 4. 抽象パターン学習
            if self.config.get('enable_abstract_patterns', True):
                abstract_patterns = self._learn_abstract_patterns(hierarchical_patterns)
                logger.info(f"抽象パターン学習完了: {len(abstract_patterns)}個")
            else:
                abstract_patterns = hierarchical_patterns
                logger.info(f"抽象パターン学習スキップ: {len(abstract_patterns)}個")
            
            # 5. パターンを統合・最適化
            final_patterns = self._integrate_and_optimize_patterns(abstract_patterns)
            logger.info(f"パターン統合・最適化完了: {len(final_patterns)}個")
            
            # 6. 結果を整理
            result = self._organize_learning_result(final_patterns)
            logger.info(f"結果整理完了: {len(result.patterns)}個")
            
            logger.info(f"パターン学習完了: {len(result.patterns)}個のパターンを学習")
            return result
            
        except Exception as e:
            logger.error(f"パターン学習エラー: {e}")
            return PatternLearningResult(
                patterns=[],
                pattern_statistics={'error': str(e)},
                learning_quality=0.0,
                generalization_capability=0.0
            )
    
    def _extract_basic_patterns(self, correspondences: List[Any]) -> List[TransformationPattern]:
        """基本変換パターンを抽出"""
        try:
            patterns = []
            
            for i, corr in enumerate(correspondences):
                input_obj = corr.input_object
                output_obj = corr.output_object
                
                logger.info(f"対応関係 {i}: 入力色={input_obj.dominant_color}, 出力色={output_obj.dominant_color}, 入力サイズ={len(input_obj.pixels)}, 出力サイズ={len(output_obj.pixels)}")
                
                # 色変換パターン
                color_pattern = self._extract_color_transformation_pattern(input_obj, output_obj)
                if color_pattern:
                    patterns.append(color_pattern)
                    logger.info(f"色変換パターン抽出: {input_obj.dominant_color} -> {output_obj.dominant_color}")
                
                # 位置変換パターン
                position_pattern = self._extract_position_transformation_pattern(input_obj, output_obj)
                if position_pattern:
                    patterns.append(position_pattern)
                    logger.info(f"位置変換パターン抽出: 入力({input_obj.center_x}, {input_obj.center_y}) -> 出力({output_obj.center_x}, {output_obj.center_y})")
                
                # サイズ変換パターン
                size_pattern = self._extract_size_transformation_pattern(input_obj, output_obj)
                if size_pattern:
                    patterns.append(size_pattern)
                    logger.info(f"サイズ変換パターン抽出: {len(input_obj.pixels)} -> {len(output_obj.pixels)}")
                
                # 形状変換パターン
                shape_pattern = self._extract_shape_transformation_pattern(input_obj, output_obj)
                if shape_pattern:
                    patterns.append(shape_pattern)
                    logger.info(f"形状変換パターン抽出")
            
            return patterns
            
        except Exception as e:
            logger.error(f"基本パターン抽出エラー: {e}")
            return []
    
    def _extract_color_transformation_pattern(self, input_obj: Object, output_obj: Object) -> Optional[TransformationPattern]:
        """色変換パターンを抽出"""
        try:
            input_color = input_obj.dominant_color
            output_color = output_obj.dominant_color
            
            if input_color != output_color:
                pattern = TransformationPattern(
                    pattern_id=f"color_change_{input_color}_to_{output_color}",
                    pattern_type="color_change",
                    input_conditions={
                        'object_type': input_obj.object_type.value,
                        'dominant_color': input_color,
                        'size_range': (len(input_obj.pixels) - 5, len(input_obj.pixels) + 5)
                    },
                    transformation_rules=[{
                        'operation': 'change_color',
                        'from_color': input_color,
                        'to_color': output_color,
                        'condition': f'dominant_color == {input_color}'
                    }],
                    confidence=1.0,
                    frequency=1,
                    examples=[{
                        'input': {'color': input_color, 'size': len(input_obj.pixels)},
                        'output': {'color': output_color, 'size': len(output_obj.pixels)}
                    }]
                )
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"色変換パターン抽出エラー: {e}")
            return None
    
    def _extract_position_transformation_pattern(self, input_obj: Object, output_obj: Object) -> Optional[TransformationPattern]:
        """位置変換パターンを抽出"""
        try:
            input_center = self._calculate_center(input_obj)
            output_center = self._calculate_center(output_obj)
            
            dx = output_center[0] - input_center[0]
            dy = output_center[1] - input_center[1]
            
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # 位置変化がある場合
                pattern = TransformationPattern(
                    pattern_id=f"position_change_{dx}_{dy}",
                    pattern_type="position_change",
                    input_conditions={
                        'object_type': input_obj.object_type.value,
                        'position_range': (input_center[0] - 2, input_center[0] + 2, input_center[1] - 2, input_center[1] + 2)
                    },
                    transformation_rules=[{
                        'operation': 'move',
                        'dx': dx,
                        'dy': dy,
                        'condition': 'true'
                    }],
                    confidence=1.0,
                    frequency=1,
                    examples=[{
                        'input': {'position': input_center, 'size': len(input_obj.pixels)},
                        'output': {'position': output_center, 'size': len(output_obj.pixels)}
                    }]
                )
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"位置変換パターン抽出エラー: {e}")
            return None
    
    def _extract_size_transformation_pattern(self, input_obj: Object, output_obj: Object) -> Optional[TransformationPattern]:
        """サイズ変換パターンを抽出"""
        try:
            input_size = len(input_obj.pixels)
            output_size = len(output_obj.pixels)
            
            if input_size != output_size:
                scale_factor = output_size / max(input_size, 1)
                
                pattern = TransformationPattern(
                    pattern_id=f"size_change_{scale_factor}",
                    pattern_type="size_change",
                    input_conditions={
                        'object_type': input_obj.object_type.value,
                        'size_range': (input_size - 2, input_size + 2)
                    },
                    transformation_rules=[{
                        'operation': 'scale',
                        'scale_factor': scale_factor,
                        'condition': f'size == {input_size}'
                    }],
                    confidence=1.0,
                    frequency=1,
                    examples=[{
                        'input': {'size': input_size, 'color': input_obj.dominant_color},
                        'output': {'size': output_size, 'color': output_obj.dominant_color}
                    }]
                )
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"サイズ変換パターン抽出エラー: {e}")
            return None
    
    def _extract_shape_transformation_pattern(self, input_obj: Object, output_obj: Object) -> Optional[TransformationPattern]:
        """形状変換パターンを抽出"""
        try:
            # バウンディングボックスのアスペクト比変化を検出
            input_bbox = input_obj.bbox
            output_bbox = output_obj.bbox
            
            input_aspect = (input_bbox[2] - input_bbox[0]) / max(input_bbox[3] - input_bbox[1], 1)
            output_aspect = (output_bbox[2] - output_bbox[0]) / max(output_bbox[3] - output_bbox[1], 1)
            
            if abs(input_aspect - output_aspect) > 0.1:  # 形状変化がある場合
                pattern = TransformationPattern(
                    pattern_id=f"shape_change_{input_aspect}_to_{output_aspect}",
                    pattern_type="shape_change",
                    input_conditions={
                        'object_type': input_obj.object_type.value,
                        'aspect_ratio': input_aspect
                    },
                    transformation_rules=[{
                        'operation': 'reshape',
                        'target_aspect_ratio': output_aspect,
                        'condition': f'aspect_ratio == {input_aspect}'
                    }],
                    confidence=1.0,
                    frequency=1,
                    examples=[{
                        'input': {'aspect_ratio': input_aspect, 'size': len(input_obj.pixels)},
                        'output': {'aspect_ratio': output_aspect, 'size': len(output_obj.pixels)}
                    }]
                )
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"形状変換パターン抽出エラー: {e}")
            return None
    
    def _generalize_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """パターンを一般化"""
        try:
            generalized = []
            pattern_groups = defaultdict(list)
            
            # 類似パターンをグループ化
            for pattern in patterns:
                key = f"{pattern.pattern_type}_{pattern.pattern_id.split('_')[0]}"
                pattern_groups[key].append(pattern)
            
            # 各グループを一般化
            for group_key, group_patterns in pattern_groups.items():
                if len(group_patterns) >= self.config.get('min_pattern_frequency', 2):
                    generalized_pattern = self._merge_patterns(group_patterns)
                    generalized.append(generalized_pattern)
            
            return generalized
            
        except Exception as e:
            logger.error(f"パターン一般化エラー: {e}")
            return patterns
    
    def _merge_patterns(self, patterns: List[TransformationPattern]) -> TransformationPattern:
        """複数のパターンをマージ"""
        try:
            if not patterns:
                return None
            
            # 最初のパターンをベースにする
            base_pattern = patterns[0]
            
            # 条件を一般化
            generalized_conditions = self._generalize_conditions([p.input_conditions for p in patterns])
            
            # 変換ルールを統合
            merged_rules = []
            for pattern in patterns:
                merged_rules.extend(pattern.transformation_rules)
            
            # 信頼度と頻度を計算
            total_confidence = sum(p.confidence for p in patterns)
            total_frequency = sum(p.frequency for p in patterns)
            
            # 例を統合
            merged_examples = []
            for pattern in patterns:
                merged_examples.extend(pattern.examples)
            
            merged_pattern = TransformationPattern(
                pattern_id=base_pattern.pattern_id.replace('_', '_generalized_'),
                pattern_type=base_pattern.pattern_type,
                input_conditions=generalized_conditions,
                transformation_rules=merged_rules,
                confidence=total_confidence / len(patterns),
                frequency=total_frequency,
                examples=merged_examples
            )
            
            return merged_pattern
            
        except Exception as e:
            logger.error(f"パターンマージエラー: {e}")
            return patterns[0] if patterns else None
    
    def _generalize_conditions(self, conditions_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """条件を一般化"""
        try:
            if not conditions_list:
                return {}
            
            generalized = {}
            
            for key in conditions_list[0].keys():
                values = [cond.get(key) for cond in conditions_list if key in cond]
                
                if key == 'size_range':
                    # サイズ範囲を拡張
                    min_sizes = [v[0] for v in values if isinstance(v, tuple) and len(v) >= 2]
                    max_sizes = [v[1] for v in values if isinstance(v, tuple) and len(v) >= 2]
                    
                    if min_sizes and max_sizes:
                        generalized[key] = (min(min_sizes), max(max_sizes))
                
                elif key == 'position_range':
                    # 位置範囲を拡張
                    positions = [v for v in values if isinstance(v, tuple) and len(v) >= 4]
                    if positions:
                        min_x = min(pos[0] for pos in positions)
                        max_x = max(pos[1] for pos in positions)
                        min_y = min(pos[2] for pos in positions)
                        max_y = max(pos[3] for pos in positions)
                        generalized[key] = (min_x, max_x, min_y, max_y)
                
                elif key == 'dominant_color':
                    # 色の集合
                    colors = set(v for v in values if v is not None)
                    if len(colors) == 1:
                        generalized[key] = list(colors)[0]
                    else:
                        generalized[f'{key}_options'] = list(colors)
                
                else:
                    # その他の条件
                    unique_values = set(v for v in values if v is not None)
                    if len(unique_values) == 1:
                        generalized[key] = list(unique_values)[0]
                    elif len(unique_values) > 1:
                        generalized[f'{key}_options'] = list(unique_values)
            
            return generalized
            
        except Exception as e:
            logger.error(f"条件一般化エラー: {e}")
            return conditions_list[0] if conditions_list else {}
    
    def _learn_hierarchical_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """階層的パターン学習"""
        try:
            hierarchical = patterns.copy()
            
            # 複合パターンを学習
            composite_patterns = self._learn_composite_patterns(patterns)
            hierarchical.extend(composite_patterns)
            
            # 抽象パターンを学習
            abstract_patterns = self._learn_abstract_patterns(patterns)
            hierarchical.extend(abstract_patterns)
            
            return hierarchical
            
        except Exception as e:
            logger.error(f"階層的パターン学習エラー: {e}")
            return patterns
    
    def _learn_composite_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """複合パターンを学習"""
        try:
            composite = []
            
            # パターンの組み合わせを探索
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    # 互換性のあるパターンの組み合わせ
                    if self._are_patterns_compatible(pattern1, pattern2):
                        composite_pattern = self._create_composite_pattern(pattern1, pattern2)
                        if composite_pattern:
                            composite.append(composite_pattern)
            
            return composite
            
        except Exception as e:
            logger.error(f"複合パターン学習エラー: {e}")
            return []
    
    def _are_patterns_compatible(self, pattern1: TransformationPattern, pattern2: TransformationPattern) -> bool:
        """パターンが互換性があるかチェック"""
        try:
            # 異なるタイプのパターンは組み合わせ可能
            if pattern1.pattern_type != pattern2.pattern_type:
                return True
            
            # 同じタイプでも条件が異なれば組み合わせ可能
            conditions1 = set(pattern1.input_conditions.keys())
            conditions2 = set(pattern2.input_conditions.keys())
            
            return len(conditions1.intersection(conditions2)) < len(conditions1.union(conditions2))
            
        except Exception as e:
            logger.error(f"パターン互換性チェックエラー: {e}")
            return False
    
    def _create_composite_pattern(self, pattern1: TransformationPattern, pattern2: TransformationPattern) -> Optional[TransformationPattern]:
        """複合パターンを作成"""
        try:
            # 条件を統合
            combined_conditions = pattern1.input_conditions.copy()
            combined_conditions.update(pattern2.input_conditions)
            
            # ルールを統合
            combined_rules = pattern1.transformation_rules + pattern2.transformation_rules
            
            # 例を統合
            combined_examples = pattern1.examples + pattern2.examples
            
            composite_pattern = TransformationPattern(
                pattern_id=f"composite_{pattern1.pattern_type}_{pattern2.pattern_type}",
                pattern_type="complex",
                input_conditions=combined_conditions,
                transformation_rules=combined_rules,
                confidence=(pattern1.confidence + pattern2.confidence) / 2.0,
                frequency=pattern1.frequency + pattern2.frequency,
                examples=combined_examples
            )
            
            return composite_pattern
            
        except Exception as e:
            logger.error(f"複合パターン作成エラー: {e}")
            return None
    
    def _learn_abstract_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """抽象パターンを学習"""
        try:
            abstract = []
            
            # パターンの抽象化
            for pattern in patterns:
                if pattern.frequency >= 1:  # 頻度条件を緩和（1回以上）
                    abstract_pattern = self._abstract_pattern(pattern)
                    if abstract_pattern:
                        abstract.append(abstract_pattern)
            
            return abstract
            
        except Exception as e:
            logger.error(f"抽象パターン学習エラー: {e}")
            return []
    
    def _abstract_pattern(self, pattern: TransformationPattern) -> Optional[TransformationPattern]:
        """パターンを抽象化"""
        try:
            # 条件を抽象化
            abstract_conditions = {}
            for key, value in pattern.input_conditions.items():
                if key == 'size_range':
                    # サイズ範囲を拡大
                    if isinstance(value, tuple) and len(value) >= 2:
                        margin = (value[1] - value[0]) * 2
                        abstract_conditions[key] = (max(0, value[0] - margin), value[1] + margin)
                elif key == 'position_range':
                    # 位置範囲を拡大
                    if isinstance(value, tuple) and len(value) >= 4:
                        margin = 5
                        abstract_conditions[key] = (
                            max(0, value[0] - margin), value[1] + margin,
                            max(0, value[2] - margin), value[3] + margin
                        )
                else:
                    abstract_conditions[key] = value
            
            abstract_pattern = TransformationPattern(
                pattern_id=f"abstract_{pattern.pattern_id}",
                pattern_type=pattern.pattern_type,
                input_conditions=abstract_conditions,
                transformation_rules=pattern.transformation_rules,
                confidence=pattern.confidence * 0.8,  # 抽象化により信頼度を下げる
                frequency=pattern.frequency,
                examples=pattern.examples
            )
            
            return abstract_pattern
            
        except Exception as e:
            logger.error(f"パターン抽象化エラー: {e}")
            return None
    
    def _integrate_and_optimize_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """パターンを統合・最適化"""
        try:
            # 重複パターンを除去
            unique_patterns = self._remove_duplicate_patterns(patterns)
            
            # 信頼度でフィルタリング
            filtered_patterns = [
                p for p in unique_patterns 
                if p.confidence >= self.config.get('confidence_threshold', 0.5)  # デフォルト値を0.5に修正
            ]
            
            # 頻度でソート
            filtered_patterns.sort(key=lambda p: p.frequency, reverse=True)
            
            # 最大パターン数で制限
            max_patterns = self.config.get('max_patterns', 1000)
            if len(filtered_patterns) > max_patterns:
                filtered_patterns = filtered_patterns[:max_patterns]
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"パターン統合・最適化エラー: {e}")
            return patterns
    
    def _remove_duplicate_patterns(self, patterns: List[TransformationPattern]) -> List[TransformationPattern]:
        """重複パターンを除去"""
        try:
            unique_patterns = []
            seen_patterns = set()
            
            for pattern in patterns:
                # パターンの特徴を文字列化
                pattern_signature = f"{pattern.pattern_type}_{json.dumps(pattern.input_conditions, sort_keys=True)}"
                
                if pattern_signature not in seen_patterns:
                    seen_patterns.add(pattern_signature)
                    unique_patterns.append(pattern)
            
            return unique_patterns
            
        except Exception as e:
            logger.error(f"重複パターン除去エラー: {e}")
            return patterns
    
    def _organize_learning_result(self, patterns: List[TransformationPattern]) -> PatternLearningResult:
        """学習結果を整理"""
        try:
            # パターン統計を計算
            pattern_types = defaultdict(int)
            total_confidence = 0.0
            
            for pattern in patterns:
                pattern_types[pattern.pattern_type] += 1
                total_confidence += pattern.confidence
            
            average_confidence = total_confidence / len(patterns) if patterns else 0.0
            
            # 学習品質を評価
            learning_quality = self._evaluate_learning_quality(patterns)
            
            # 一般化能力を評価
            generalization_capability = self._evaluate_generalization_capability(patterns)
            
            pattern_statistics = {
                'total_patterns': len(patterns),
                'pattern_types': dict(pattern_types),
                'average_confidence': average_confidence,
                'learning_progress': len(patterns) / self.config.get('max_patterns', 1000)
            }
            
            return PatternLearningResult(
                patterns=patterns,
                pattern_statistics=pattern_statistics,
                learning_quality=learning_quality,
                generalization_capability=generalization_capability
            )
            
        except Exception as e:
            logger.error(f"学習結果整理エラー: {e}")
            return PatternLearningResult(
                patterns=[],
                pattern_statistics={'error': str(e)},
                learning_quality=0.0,
                generalization_capability=0.0
            )
    
    def _evaluate_learning_quality(self, patterns: List[TransformationPattern]) -> float:
        """学習品質を評価"""
        try:
            if not patterns:
                return 0.0
            
            # 信頼度の平均
            avg_confidence = np.mean([p.confidence for p in patterns])
            
            # パターンの多様性
            pattern_types = set(p.pattern_type for p in patterns)
            diversity = len(pattern_types) / 5.0  # 最大5種類のパターンタイプ
            
            # 頻度の分布
            frequencies = [p.frequency for p in patterns]
            frequency_balance = 1.0 - (np.std(frequencies) / max(np.mean(frequencies), 1.0))
            
            quality = (avg_confidence + diversity + frequency_balance) / 3.0
            return min(1.0, quality)
            
        except Exception as e:
            logger.error(f"学習品質評価エラー: {e}")
            return 0.0
    
    def _evaluate_generalization_capability(self, patterns: List[TransformationPattern]) -> float:
        """一般化能力を評価"""
        try:
            if not patterns:
                return 0.0
            
            # 抽象パターンの割合
            abstract_count = sum(1 for p in patterns if 'abstract' in p.pattern_id)
            abstract_ratio = abstract_count / len(patterns)
            
            # 複合パターンの割合
            composite_count = sum(1 for p in patterns if p.pattern_type == 'complex')
            composite_ratio = composite_count / len(patterns)
            
            # 条件の一般化度
            generalization_scores = []
            for pattern in patterns:
                # 条件の柔軟性を評価
                flexible_conditions = 0
                total_conditions = len(pattern.input_conditions)
                
                for key, value in pattern.input_conditions.items():
                    if 'options' in key or 'range' in key:
                        flexible_conditions += 1
                
                if total_conditions > 0:
                    flexibility = flexible_conditions / total_conditions
                    generalization_scores.append(flexibility)
            
            avg_generalization = np.mean(generalization_scores) if generalization_scores else 0.0
            
            capability = (abstract_ratio + composite_ratio + avg_generalization) / 3.0
            return min(1.0, capability)
            
        except Exception as e:
            logger.error(f"一般化能力評価エラー: {e}")
            return 0.0
    
    def _calculate_center(self, obj: Object) -> Tuple[float, float]:
        """オブジェクトの重心を計算"""
        try:
            if not obj.pixels:
                return (0.0, 0.0)
            
            x_sum = sum(pixel[0] for pixel in obj.pixels)
            y_sum = sum(pixel[1] for pixel in obj.pixels)
            
            return (x_sum / len(obj.pixels), y_sum / len(obj.pixels))
            
        except Exception as e:
            logger.error(f"重心計算エラー: {e}")
            return (0.0, 0.0)
    
    def get_learned_patterns(self) -> List[TransformationPattern]:
        """学習されたパターンを取得"""
        return list(self.learned_patterns.values())
    
    def save_patterns(self, filepath: str) -> bool:
        """パターンをファイルに保存"""
        try:
            patterns_data = []
            for pattern in self.learned_patterns.values():
                pattern_dict = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'input_conditions': pattern.input_conditions,
                    'transformation_rules': pattern.transformation_rules,
                    'confidence': pattern.confidence,
                    'frequency': pattern.frequency,
                    'examples': pattern.examples
                }
                patterns_data.append(pattern_dict)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"パターンを保存: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"パターン保存エラー: {e}")
            return False
    
    def load_patterns(self, filepath: str) -> bool:
        """ファイルからパターンを読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            self.learned_patterns = {}
            for pattern_dict in patterns_data:
                pattern = TransformationPattern(
                    pattern_id=pattern_dict['pattern_id'],
                    pattern_type=pattern_dict['pattern_type'],
                    input_conditions=pattern_dict['input_conditions'],
                    transformation_rules=pattern_dict['transformation_rules'],
                    confidence=pattern_dict['confidence'],
                    frequency=pattern_dict['frequency'],
                    examples=pattern_dict['examples']
                )
                self.learned_patterns[pattern.pattern_id] = pattern
            
            logger.info(f"パターンを読み込み: {len(self.learned_patterns)}個")
            return True
            
        except Exception as e:
            logger.error(f"パターン読み込みエラー: {e}")
            return False
