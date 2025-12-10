"""
オブジェクトグラフ構築モジュール

オブジェクトリストからグラフ構造を構築する
"""

from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import torch
from dataclasses import dataclass, field

from src.data_systems.data_models.core.object import Object


@dataclass
class GraphEdge:
    """グラフエッジ"""
    source_idx: int
    target_idx: int
    edge_type: str  # 'adjacent', 'contain', 'touch', 'spatial', 'category'
    weight: float = 1.0
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ObjectGraph:
    """オブジェクトグラフ"""
    nodes: List[Object]  # ノード（オブジェクト）
    edges: List[GraphEdge]  # エッジ（関係）
    node_features: torch.Tensor  # [num_nodes, feature_dim]
    edge_index: torch.Tensor  # [2, num_edges]
    edge_attr: torch.Tensor  # [num_edges, edge_feature_dim]
    edge_type_map: Dict[str, int]  # エッジタイプのマッピング


class ObjectGraphBuilder:
    """オブジェクトグラフ構築クラス"""

    def __init__(
        self,
        max_grid_size: int = 30,
        adjacency_threshold: float = 1.0,
        containment_threshold: float = 0.8,
        touch_threshold: float = 1.0
    ):
        """
        初期化

        Args:
            max_grid_size: 最大グリッドサイズ
            adjacency_threshold: 隣接判定の閾値（マンハッタン距離）
            containment_threshold: 包含判定の閾値（IoU）
            touch_threshold: 接触判定の閾値（マンハッタン距離）
        """
        self.max_grid_size = max_grid_size
        self.adjacency_threshold = adjacency_threshold
        self.containment_threshold = containment_threshold
        self.touch_threshold = touch_threshold

        # エッジタイプのマッピング
        self.edge_type_map = {
            'adjacent': 0,
            'contain': 1,
            'touch': 2,
            'spatial_left': 3,
            'spatial_right': 4,
            'spatial_up': 5,
            'spatial_down': 6,
            'category': 7
        }

    def build_graph(
        self,
        objects: List[Object],
        categories: Optional[Dict[int, List[int]]] = None,
        grid_width: int = 30,
        grid_height: int = 30
    ) -> ObjectGraph:
        """
        オブジェクトリストからグラフを構築

        Args:
            objects: オブジェクトリスト
            categories: カテゴリ情報 {category_id: [object_indices]}
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            ObjectGraph: 構築されたグラフ
        """
        if not objects:
            # 空のグラフを返す
            return ObjectGraph(
                nodes=[],
                edges=[],
                node_features=torch.zeros((0, 12), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 8), dtype=torch.float32),
                edge_type_map=self.edge_type_map
            )

        # ノード特徴量を計算
        node_features = self._compute_node_features(objects, grid_width, grid_height)

        # エッジを構築
        edges = []
        edge_features = []

        # 1. 隣接関係
        adjacent_edges = self._find_adjacent_edges(objects)
        edges.extend(adjacent_edges)
        edge_features.extend([self._compute_edge_features(e, objects) for e in adjacent_edges])

        # 2. 包含関係
        contain_edges = self._find_containment_edges(objects)
        edges.extend(contain_edges)
        edge_features.extend([self._compute_edge_features(e, objects) for e in contain_edges])

        # 3. 接触関係
        touch_edges = self._find_touch_edges(objects)
        edges.extend(touch_edges)
        edge_features.extend([self._compute_edge_features(e, objects) for e in touch_edges])

        # 4. 空間関係（左右上下）
        spatial_edges = self._find_spatial_edges(objects)
        edges.extend(spatial_edges)
        edge_features.extend([self._compute_edge_features(e, objects) for e in spatial_edges])

        # 5. カテゴリ関係
        if categories:
            category_edges = self._find_category_edges(objects, categories)
            edges.extend(category_edges)
            edge_features.extend([self._compute_edge_features(e, objects) for e in category_edges])

        # エッジインデックスとエッジ属性を構築
        if edges:
            edge_index = torch.tensor(
                [[e.source_idx, e.target_idx] for e in edges],
                dtype=torch.long
            ).t().contiguous()  # [2, num_edges]

            edge_attr = torch.stack(edge_features)  # [num_edges, edge_feature_dim]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 8), dtype=torch.float32)

        return ObjectGraph(
            nodes=objects,
            edges=edges,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type_map=self.edge_type_map
        )

    def _compute_node_features(
        self,
        objects: List[Object],
        grid_width: int,
        grid_height: int
    ) -> torch.Tensor:
        """
        ノード特徴量を計算

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            torch.Tensor: [num_nodes, feature_dim] ノード特徴量
        """
        features = []
        for obj in objects:
            # 基本特徴量（12次元）
            # 色(1) + 位置(4: bbox) + サイズ(2: width, height) + その他(5: area, center_x, center_y, density, perimeter)
            color = obj.dominant_color / 9.0  # 正規化 [0, 1]

            # bbox座標を正規化
            bbox_left = obj.bbox_left / grid_width
            bbox_top = obj.bbox_top / grid_height
            bbox_right = obj.bbox_right / grid_width
            bbox_bottom = obj.bbox_bottom / grid_height

            # サイズを正規化
            width = obj.bbox_width / grid_width
            height = obj.bbox_height / grid_height

            # その他の特徴量
            area = obj.pixel_count / (grid_width * grid_height)  # 正規化
            center_x = obj.center_x / grid_width
            center_y = obj.center_y / grid_height
            density = obj.density if hasattr(obj, 'density') else area / (width * height + 1e-6)
            perimeter = obj.perimeter / (2 * (grid_width + grid_height)) if hasattr(obj, 'perimeter') else 0.0

            feature = [
                color,
                bbox_left, bbox_top, bbox_right, bbox_bottom,
                width, height,
                area, center_x, center_y, density, perimeter
            ]
            features.append(feature)

        return torch.tensor(features, dtype=torch.float32)  # [num_nodes, 12]

    def _find_adjacent_edges(self, objects: List[Object]) -> List[GraphEdge]:
        """隣接関係のエッジを検出"""
        edges = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], start=i+1):
                distance = self._manhattan_distance(obj1, obj2)
                if distance <= self.adjacency_threshold:
                    edges.append(GraphEdge(
                        source_idx=i,
                        target_idx=j,
                        edge_type='adjacent',
                        weight=1.0 / (distance + 1.0),
                        features={'distance': distance}
                    ))
        return edges

    def _find_containment_edges(self, objects: List[Object]) -> List[GraphEdge]:
        """包含関係のエッジを検出"""
        edges = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                iou = self._compute_iou(obj1, obj2)
                if iou >= self.containment_threshold:
                    # obj1がobj2を含む場合
                    if self._contains(obj1, obj2):
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='contain',
                            weight=iou,
                            features={'iou': iou}
                        ))
        return edges

    def _find_touch_edges(self, objects: List[Object]) -> List[GraphEdge]:
        """接触関係のエッジを検出"""
        edges = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], start=i+1):
                distance = self._manhattan_distance(obj1, obj2)
                if 0 < distance <= self.touch_threshold:
                    # 隣接ではないが接触している場合
                    if distance > self.adjacency_threshold:
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='touch',
                            weight=1.0 / (distance + 1.0),
                            features={'distance': distance}
                        ))
        return edges

    def _find_spatial_edges(self, objects: List[Object]) -> List[GraphEdge]:
        """空間関係（左右上下）のエッジを検出"""
        edges = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue

                # 中心座標を取得
                cx1, cy1 = obj1.center_x, obj1.center_y
                cx2, cy2 = obj2.center_x, obj2.center_y

                # 左右関係
                if abs(cy1 - cy2) < max(obj1.bbox_height, obj2.bbox_height) / 2:
                    if cx1 < cx2:
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='spatial_left',
                            weight=1.0,
                            features={'dx': cx2 - cx1, 'dy': abs(cy1 - cy2)}
                        ))
                    else:
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='spatial_right',
                            weight=1.0,
                            features={'dx': cx1 - cx2, 'dy': abs(cy1 - cy2)}
                        ))

                # 上下関係
                if abs(cx1 - cx2) < max(obj1.bbox_width, obj2.bbox_width) / 2:
                    if cy1 < cy2:
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='spatial_up',
                            weight=1.0,
                            features={'dx': abs(cx1 - cx2), 'dy': cy2 - cy1}
                        ))
                    else:
                        edges.append(GraphEdge(
                            source_idx=i,
                            target_idx=j,
                            edge_type='spatial_down',
                            weight=1.0,
                            features={'dx': abs(cx1 - cx2), 'dy': cy1 - cy2}
                        ))

        return edges

    def _find_category_edges(
        self,
        objects: List[Object],
        categories: Dict[int, List[int]]
    ) -> List[GraphEdge]:
        """カテゴリ関係のエッジを検出"""
        edges = []
        for category_id, object_indices in categories.items():
            # 同じカテゴリ内のオブジェクト間にエッジを追加
            for i in range(len(object_indices)):
                for j in range(i+1, len(object_indices)):
                    idx1 = object_indices[i]
                    idx2 = object_indices[j]
                    if idx1 < len(objects) and idx2 < len(objects):
                        edges.append(GraphEdge(
                            source_idx=idx1,
                            target_idx=idx2,
                            edge_type='category',
                            weight=1.0,
                            features={'category_id': float(category_id)}
                        ))
        return edges

    def _compute_edge_features(
        self,
        edge: GraphEdge,
        objects: List[Object]
    ) -> torch.Tensor:
        """
        エッジ特徴量を計算

        Args:
            edge: グラフエッジ
            objects: オブジェクトリスト

        Returns:
            torch.Tensor: [edge_feature_dim] エッジ特徴量
        """
        obj1 = objects[edge.source_idx]
        obj2 = objects[edge.target_idx]

        # エッジタイプのone-hotエンコーディング（8次元）
        edge_type_onehot = torch.zeros(8, dtype=torch.float32)
        edge_type_onehot[self.edge_type_map[edge.edge_type]] = 1.0

        # 距離特徴量
        distance = self._manhattan_distance(obj1, obj2)
        normalized_distance = distance / (self.max_grid_size * 2)

        # サイズ比
        size_ratio = min(obj1.pixel_count, obj2.pixel_count) / (max(obj1.pixel_count, obj2.pixel_count) + 1e-6)

        # 色の一致度
        color_match = 1.0 if obj1.dominant_color == obj2.dominant_color else 0.0

        # エッジの重み
        weight = edge.weight

        # 特徴量を結合 [8 + 4 = 12次元]
        features = torch.cat([
            edge_type_onehot,
            torch.tensor([normalized_distance, size_ratio, color_match, weight], dtype=torch.float32)
        ])

        return features

    def _manhattan_distance(self, obj1: Object, obj2: Object) -> float:
        """マンハッタン距離を計算"""
        cx1, cy1 = obj1.center_x, obj1.center_y
        cx2, cy2 = obj2.center_x, obj2.center_y
        return abs(cx1 - cx2) + abs(cy1 - cy2)

    def _compute_iou(self, obj1: Object, obj2: Object) -> float:
        """IoU（Intersection over Union）を計算"""
        # bboxの交差部分を計算
        x1 = max(obj1.bbox_left, obj2.bbox_left)
        y1 = max(obj1.bbox_top, obj2.bbox_top)
        x2 = min(obj1.bbox_right, obj2.bbox_right)
        y2 = min(obj1.bbox_bottom, obj2.bbox_bottom)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
        area1 = obj1.bbox_width * obj1.bbox_height
        area2 = obj2.bbox_width * obj2.bbox_height
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _contains(self, obj1: Object, obj2: Object) -> bool:
        """obj1がobj2を含むかどうかを判定"""
        return (obj1.bbox_left <= obj2.bbox_left and
                obj1.bbox_top <= obj2.bbox_top and
                obj1.bbox_right >= obj2.bbox_right and
                obj1.bbox_bottom >= obj2.bbox_bottom)
