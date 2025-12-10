"""ProgramScorer 用データセット

JSONL などで用意した
    {\"features\": [...], \"label\": float, \"metadata\": {...}}
形式のサンプル群をラップするシンプルな Dataset。
"""

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class ProgramScorerDataset(Dataset):
    """
    ProgramScorer 用の汎用データセット

    1サンプル = 1つの候補プログラムに対応する。

    期待する入力フォーマット（samples 引数の各要素）:
        {
            \"features\": np.ndarray または list[float],  # 事前に抽出された特徴ベクトル
            \"label\": float,                             # 教師スコア（0.0〜1.0 など）
            \"metadata\": { ... }                         # 任意: task_id, program_text など
        }
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        feats = item.get("features")
        label = float(item.get("label", 0.0))
        metadata = item.get("metadata", {})

        features_tensor = torch.tensor(np.asarray(feats, dtype=float), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {
            "features": features_tensor,
            "label": label_tensor,
            "metadata": metadata,
        }
