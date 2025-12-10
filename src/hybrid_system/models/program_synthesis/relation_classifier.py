"""
関係分類器モジュール

オブジェクト間の関係性（上下左右、対称性、同型パターン、包含関係）を分類
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.data_systems.data_models.core.object import Object
from .object_graph_builder import ObjectGraph


class RelationClassifier(nn.Module):
    """関係分類器"""

    def __init__(
        self,
        node_feature_dim: int = 12,
        embed_dim: int = 128,
        num_relation_types: int = 8,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            node_feature_dim: ノード特徴量の次元
            embed_dim: 埋め込み次元
            num_relation_types: 関係タイプ数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.num_relation_types = num_relation_types

        # ノード特徴量の投影
        self.node_projection = nn.Linear(node_feature_dim, embed_dim)

        # 関係分類ネットワーク
        self.relation_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 4, embed_dim),  # 2つのノード特徴量 + 4次元の相対特徴量
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_relation_types)
        )

        # 関係タイプの定義
        self.relation_types = [
            'spatial_left',
            'spatial_right',
            'spatial_up',
            'spatial_down',
            'mirror_x',
            'mirror_y',
            'repeat',
            'contain'
        ]

    def forward(
        self,
        graph: ObjectGraph
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            graph: オブジェクトグラフ

        Returns:
            torch.Tensor: [num_edges, num_relation_types] 関係分類スコア
        """
        if graph.edge_index.size(1) == 0:
            return torch.zeros((0, self.num_relation_types), device=graph.node_features.device)

        # ノード特徴量を投影
        node_features = self.node_projection(graph.node_features)  # [num_nodes, embed_dim]

        # エッジのノード特徴量を取得
        src_idx = graph.edge_index[0]  # [num_edges]
        tgt_idx = graph.edge_index[1]  # [num_edges]

        src_features = node_features[src_idx]  # [num_edges, embed_dim]
        tgt_features = node_features[tgt_idx]  # [num_edges, embed_dim]

        # 相対特徴量を計算
        relative_features = self._compute_relative_features(
            graph.nodes, src_idx, tgt_idx
        )  # [num_edges, 4]

        # 関係分類
        relation_input = torch.cat([src_features, tgt_features, relative_features], dim=1)
        relation_scores = self.relation_mlp(relation_input)  # [num_edges, num_relation_types]

        return relation_scores

    def classify_relations(
        self,
        graph: ObjectGraph,
        threshold: float = 0.5
    ) -> Dict[Tuple[int, int], List[str]]:
        """
        関係を分類

        Args:
            graph: オブジェクトグラフ
            threshold: 分類閾値

        Returns:
            Dict[Tuple[int, int], List[str]]: {(src_idx, tgt_idx): [relation_types]}
        """
        self.eval()
        with torch.no_grad():
            relation_scores = self.forward(graph)  # [num_edges, num_relation_types]
            relation_probs = F.softmax(relation_scores, dim=1)  # [num_edges, num_relation_types]

        relations = {}
        for i in range(graph.edge_index.size(1)):
            src_idx = graph.edge_index[0, i].item()
            tgt_idx = graph.edge_index[1, i].item()
            edge_key = (src_idx, tgt_idx)

            # 閾値を超える関係タイプを取得
            probs = relation_probs[i].cpu().numpy()
            relation_types = [
                self.relation_types[j]
                for j in range(self.num_relation_types)
                if probs[j] >= threshold
            ]

            if relation_types:
                relations[edge_key] = relation_types

        return relations

    def _compute_relative_features(
        self,
        objects: List[Object],
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        相対特徴量を計算

        Args:
            objects: オブジェクトリスト
            src_idx: ソースノードインデックス [num_edges]
            tgt_idx: ターゲットノードインデックス [num_edges]

        Returns:
            torch.Tensor: [num_edges, 4] 相対特徴量
        """
        features = []
        for i in range(src_idx.size(0)):
            src_obj = objects[src_idx[i].item()]
            tgt_obj = objects[tgt_idx[i].item()]

            # 相対位置
            dx = (tgt_obj.center_x - src_obj.center_x) / 30.0  # 正規化
            dy = (tgt_obj.center_y - src_obj.center_y) / 30.0  # 正規化

            # サイズ比
            size_ratio = min(src_obj.pixel_count, tgt_obj.pixel_count) / (
                max(src_obj.pixel_count, tgt_obj.pixel_count) + 1e-6
            )

            # 色の一致度
            color_match = 1.0 if src_obj.dominant_color == tgt_obj.dominant_color else 0.0

            features.append([dx, dy, size_ratio, color_match])

        return torch.tensor(features, dtype=torch.float32, device=src_idx.device)

    def detect_symmetry(
        self,
        objects: List[Object],
        grid_width: int,
        grid_height: int
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        対称性を検出

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            Dict[str, List[Tuple[int, int]]]: {'mirror_x': [(obj1_idx, obj2_idx), ...], ...}
        """
        symmetry_pairs = {
            'mirror_x': [],
            'mirror_y': []
        }

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], start=i+1):
                # X軸対称性をチェック
                if self._is_mirror_x(obj1, obj2, grid_width):
                    symmetry_pairs['mirror_x'].append((i, j))

                # Y軸対称性をチェック
                if self._is_mirror_y(obj1, obj2, grid_height):
                    symmetry_pairs['mirror_y'].append((i, j))

        return symmetry_pairs

    def _is_mirror_x(self, obj1: Object, obj2: Object, grid_width: int) -> bool:
        """X軸対称性をチェック"""
        # 中心X座標が対称かどうか
        center_x1 = obj1.center_x
        center_x2 = grid_width - 1 - obj2.center_x

        # 許容誤差
        tolerance = max(obj1.bbox_width, obj2.bbox_width) / 2

        return abs(center_x1 - center_x2) < tolerance

    def _is_mirror_y(self, obj1: Object, obj2: Object, grid_height: int) -> bool:
        """Y軸対称性をチェック"""
        # 中心Y座標が対称かどうか
        center_y1 = obj1.center_y
        center_y2 = grid_height - 1 - obj2.center_y

        # 許容誤差
        tolerance = max(obj1.bbox_height, obj2.bbox_height) / 2

        return abs(center_y1 - center_y2) < tolerance

    def detect_repeat_pattern(
        self,
        objects: List[Object],
        threshold: float = 0.8
    ) -> List[List[int]]:
        """
        繰り返しパターンを検出

        Args:
            objects: オブジェクトリスト
            threshold: 類似度閾値

        Returns:
            List[List[int]]: 繰り返しパターンのグループ [[obj1_idx, obj2_idx, ...], ...]
        """
        if len(objects) < 2:
            return []

        # オブジェクトの類似度を計算
        similarity_matrix = self._compute_similarity_matrix(objects)

        # 類似度が閾値を超えるオブジェクトをグループ化
        groups = []
        used = set()

        for i in range(len(objects)):
            if i in used:
                continue

            group = [i]
            for j in range(i+1, len(objects)):
                if j in used:
                    continue

                if similarity_matrix[i, j] >= threshold:
                    group.append(j)
                    used.add(j)

            if len(group) >= 2:
                groups.append(group)
                used.add(i)

        return groups

    def _compute_similarity_matrix(self, objects: List[Object]) -> np.ndarray:
        """類似度行列を計算"""
        n = len(objects)
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                obj1 = objects[i]
                obj2 = objects[j]

                # サイズの類似度
                size_sim = min(obj1.pixel_count, obj2.pixel_count) / (
                    max(obj1.pixel_count, obj2.pixel_count) + 1e-6
                )

                # 形状の類似度（アスペクト比）
                aspect1 = obj1.bbox_width / (obj1.bbox_height + 1e-6)
                aspect2 = obj2.bbox_width / (obj2.bbox_height + 1e-6)
                aspect_sim = min(aspect1, aspect2) / (max(aspect1, aspect2) + 1e-6)

                # 色の一致度
                color_sim = 1.0 if obj1.dominant_color == obj2.dominant_color else 0.0

                # 総合類似度
                similarity[i, j] = (size_sim + aspect_sim + color_sim) / 3.0
                similarity[j, i] = similarity[i, j]

        return similarity
