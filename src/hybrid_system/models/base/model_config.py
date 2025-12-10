"""
モデル設定

各モデルの設定を管理
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import json
import os


@dataclass
class ProgramSynthesisConfig:
    """プログラム合成モデルの設定"""
    model_name: str = "program_synthesis"

    # エンコーダ設定
    grid_encoder_dim: int = 256
    grid_encoder_layers: int = 4
    grid_encoder_heads: int = 8

    # デコーダ設定
    program_decoder_dim: int = 512
    program_decoder_layers: int = 6
    program_decoder_heads: int = 8

    # 共通設定
    hidden_dim: int = 512
    dropout: float = 0.1
    max_program_length: int = 512
    vocab_size: int = 1000  # DSL語彙に最適化（従来: 10000）

    # Cross-Attention融合設定（Tier 2改善）
    use_cross_attention_fusion: bool = False  # Cross-Attention融合を使用するか（デフォルト: False）
    cross_attention_layers: int = 2  # Cross-Attention層数
    use_syntax_tree_decoder: bool = False  # 構文木ベースデコーダーを使用するか（デフォルト: False）

    # Abstract Object Patterns設定（Tier 3改善）
    enable_abstract_patterns: bool = False  # Abstract Object Patternsを使用するか（デフォルト: False）
    abstract_pattern_types: Optional[List[str]] = None  # 使用する抽象パターンタイプのリスト（Noneの場合は['full']）

    # 学習設定
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000

    # デバイス設定
    device: str = "cuda"
    mixed_precision: bool = True

    # 保存設定
    save_dir: str = "models/checkpoints/program_synthesis"
    log_dir: str = "logs/program_synthesis"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgramSynthesisConfig':
        """辞書から復元"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_config(config: Any, path: str):
    """設定を保存"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(path: str, config_class: type) -> Any:
    """設定を読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return config_class.from_dict(data)
