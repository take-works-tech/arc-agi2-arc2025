"""
プログラム合成設定

候補生成に関する設定クラス
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class CandidateConfig:
    """候補生成設定"""
    max_candidates: int = 30
    enable_neural_generation: bool = True
    enable_neural_object_generation: bool = True
    pattern_threshold: float = 0.7
    task_seed: Optional[int] = None
    num_seeds_per_pair: int = 3
    enable_seed_variation: bool = True

    # 各候補生成方法の生成数設定
    # ④深層学習ベース（グリッド→プログラム）: 部分プログラムを使用する場合の候補生成数
    num_neural_candidates_with_partial: int = 20
    # ④深層学習ベース（グリッド→プログラム）: 部分プログラムを使用しない場合の候補生成数
    num_neural_candidates_without_partial: int = 20
    # ⑤深層学習ベース（オブジェクト→プログラム）: 部分プログラムを使用する場合の候補生成数
    num_neural_object_candidates_with_partial: int = 20
    # ⑤深層学習ベース（オブジェクト→プログラム）: 部分プログラムを使用しない場合の候補生成数
    num_neural_object_candidates_without_partial: int = 20

    # 新規追加: オブジェクト→プログラムの改善設定
    # Object Canonicalization
    enable_object_canonicalization: bool = True
    # Object Graph + GNN
    enable_object_graph: bool = True
    graph_encoder_type: str = "graphormer"  # "graphormer" or "egnn"
    # Relation Classifier
    enable_relation_classifier: bool = True
    relation_classifier_threshold: float = 0.7

    # 新規追加: グリッド→プログラムの改善設定
    # NGPS（Neural Guided Program Search）
    enable_ngps: bool = True
    enable_dsl_selector: bool = True
    ngps_dsl_selector_weight: float = 0.3
    ngps_token_prob_weight: float = 0.6
    ngps_exploration_bonus_weight: float = 0.1
    ngps_dsl_filter_threshold: float = 0.1