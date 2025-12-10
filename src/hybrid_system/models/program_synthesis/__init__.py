"""
プログラム合成モデルモジュール

入出力ペアからDSLプログラムを生成するモデル
"""

from .program_synthesis_model import ProgramSynthesisModel
from .object_based_program_synthesis_model import ObjectBasedProgramSynthesisModel
from .grid_encoder import GridEncoder
from .object_encoder import ObjectEncoder
from .program_decoder import ProgramDecoder
from .grid_decoder import GridDecoder

# 新規追加: Object Graph + GNN関連
from .object_graph_builder import ObjectGraphBuilder, ObjectGraph, GraphEdge
from .object_graph_encoder import (
    ObjectGraphEncoder,
    GraphormerEncoder,
    EGNNEncoder
)
from .object_canonicalizer import ObjectCanonicalizer, CanonicalizedObject
from .relation_classifier import RelationClassifier

# 新規追加: NGPS関連
from .dsl_selector import DSLSelector, TokenToDSLMapper
from .neural_guided_program_search import NeuralGuidedProgramSearch

# 新規追加: 色役割分類関連
from .color_role_classifier import (
    ColorRoleClassifier,
    ColorFeatureExtractor,
    EnhancedBackgroundColorInferencer
)

# 新規追加: 高レベルDSL関連
from .high_level_dsl import (
    HighLevelDSLConverter,
    HighLevelDSLGenerator,
    HighLevelDSLCommand
)

# 新規追加: 構文木ベースデコーダー関連
from .syntax_tree_decoder import (
    SyntaxTreeDecoder,
    SyntaxNode,
    SyntaxNodeType,
    SyntaxConstraintChecker
)

# 新規追加: Neural Mask Generator関連
from .neural_mask_generator import (
    NeuralMaskGenerator,
    MaskBasedProgramGuider
)

# 新規追加: Cross-Attention融合関連
from .cross_attention_fusion import (
    CrossAttentionFusion,
    InputOutputFusion,
    BidirectionalInputOutputFusion
)

# 新規追加: Abstract Object Patterns関連
from .abstract_object_patterns import (
    AbstractObjectPatternExtractor,
    AbstractObjectPattern,
    AbstractPatternEncoder
)
# ViTGridEncoderは削除されました（未使用のため）

# 新規追加: Symmetry-Aware Augmentation関連
from .symmetry_augmentation import (
    SymmetryAugmenter,
    SymmetryType,
    SymmetryAwareDataLoader
)

# 新規追加: Neural Mask Generator関連（既に存在する場合は上書きしない）
try:
    from .neural_mask_generator import NeuralMaskGenerator, MaskBasedProgramGuider
except ImportError:
    # 既存の実装がない場合は新規作成
    pass

__all__ = [
    'ProgramSynthesisModel',
    'ObjectBasedProgramSynthesisModel',
    'GridEncoder',
    'ObjectEncoder',
    'ProgramDecoder',
    'GridDecoder',
    # 新規追加
    'ObjectGraphBuilder',
    'ObjectGraph',
    'GraphEdge',
    'ObjectGraphEncoder',
    'GraphormerEncoder',
    'EGNNEncoder',
    'ObjectCanonicalizer',
    'CanonicalizedObject',
    'RelationClassifier',
    # 新規追加: NGPS関連
    'DSLSelector',
    'TokenToDSLMapper',
    'NeuralGuidedProgramSearch',
    # 新規追加: 色役割分類関連
    'ColorRoleClassifier',
    'ColorFeatureExtractor',
    'EnhancedBackgroundColorInferencer',
    # 新規追加: 高レベルDSL関連
    'HighLevelDSLConverter',
    'HighLevelDSLGenerator',
    'HighLevelDSLCommand',
    # 新規追加: 構文木ベースデコーダー関連
    'SyntaxTreeDecoder',
    'SyntaxNode',
    'SyntaxNodeType',
    'SyntaxConstraintChecker',
    # 新規追加: Neural Mask Generator関連
    'NeuralMaskGenerator',
    'MaskBasedProgramGuider',
    # 新規追加: Cross-Attention融合関連
    'CrossAttentionFusion',
    'InputOutputFusion',
    'BidirectionalInputOutputFusion',
    # 新規追加: Abstract Object Patterns関連
    'AbstractObjectPatternExtractor',
    'AbstractObjectPattern',
    'AbstractPatternEncoder',
    # 新規追加: Symmetry-Aware Augmentation関連
    'SymmetryAugmenter',
    'SymmetryType',
    'SymmetryAwareDataLoader',
    # 新規追加: Neural Mask Generator関連
    'NeuralMaskGenerator',
    'MaskBasedProgramGuider',
    # ViTGridEncoderは削除されました（未使用のため）
    # 'PatchEmbedding',
]
