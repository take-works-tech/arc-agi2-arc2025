"""
出力グリッドサイズ予測用のデータセット（本格実装）

json_pair_loader.iter_arc_pair_samples で読み出したサンプルから
基本特徴量を抽出し、出力グリッドの高さ・幅をターゲットとして返す。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .json_pair_loader import iter_arc_pair_samples, extract_basic_features


class OutputSizePredictionDataset(Dataset):
    """
    出力グリッドのサイズ（高さ・幅）予測タスク用の PyTorch Dataset

    各サンプルは以下を返す:
        - features: torch.float32 ベクトル
        - target: torch.long ベクトル [2] （[output_height, output_width]）
        - metadata: 付帯情報（task_id, pair_index など）
    """

    def __init__(self, output_root: str):
        """
        Args:
            output_root: outputs ディレクトリ（例: 'outputs/test_multiple_pairs_verification_...'）
        """
        self.output_root = Path(output_root)
        self.samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        # 全サンプルをメモリに読み込む（規模が大きくなれば分割読み込みも検討）
        for sample in iter_arc_pair_samples(str(self.output_root)):
            feats = extract_basic_features(sample)
            self.samples.append((sample, feats))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample, feats = self.samples[idx]
        # 特徴量ベクトル（順序は固定）
        feature_vec = torch.tensor(
            [
                feats["input_height"],
                feats["input_width"],
                feats["object_count_before"],
                feats["object_count_after"],
                feats["rendered_object_count"],
                feats["avg_bbox_width"],
                feats["avg_bbox_height"],
                feats["max_bbox_width"],
                feats["max_bbox_height"],
            ],
            dtype=torch.float32,
        )

        output_grid: np.ndarray = sample["output_grid"]
        out_h, out_w = (0, 0)
        if isinstance(output_grid, np.ndarray) and output_grid.size > 0:
            out_h, out_w = output_grid.shape[:2]
        target = torch.tensor([out_h, out_w], dtype=torch.long)

        return {
            "features": feature_vec,
            "target": target,
            "metadata": {
                "task_id": sample.get("task_id"),
                "pair_index": sample.get("pair_index"),
            },
        }
