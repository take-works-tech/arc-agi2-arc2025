"""
オブジェクトグラフエンコーダー（GNN）

GraphormerとEGNNをサポート
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .object_graph_builder import ObjectGraph


class GraphormerEncoder(nn.Module):
    """Graphormerエンコーダー"""

    def __init__(
        self,
        node_feature_dim: int = 12,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 100
    ):
        """
        初期化

        Args:
            node_feature_dim: ノード特徴量の次元
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            max_nodes: 最大ノード数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_nodes = max_nodes

        # ノード特徴量の投影
        self.node_projection = nn.Linear(node_feature_dim, embed_dim)

        # エッジ特徴量の投影
        self.edge_projection = nn.Linear(12, embed_dim)

        # 位置エンコーディング（ノードの位置）
        self.node_encoding = nn.Parameter(torch.randn(max_nodes, embed_dim))

        # Graphormerレイヤー
        self.layers = nn.ModuleList([
            GraphormerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: ObjectGraph,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        フォワードパス

        Args:
            graph: オブジェクトグラフ
            return_attention: アテンション重みを返すかどうか

        Returns:
            torch.Tensor: [batch, num_nodes, embed_dim] エンコードされたノード表現
            Optional[torch.Tensor]: アテンション重み（return_attention=Trueの場合）
        """
        num_nodes = graph.node_features.size(0)
        if num_nodes == 0:
            return torch.zeros((1, 0, self.embed_dim), device=graph.node_features.device), None

        # ノード特徴量を投影
        x = self.node_projection(graph.node_features)  # [num_nodes, embed_dim]

        # 位置エンコーディングを追加
        if num_nodes <= self.max_nodes:
            x = x + self.node_encoding[:num_nodes]
        else:
            # 最大ノード数を超える場合は補間
            x = x + F.interpolate(
                self.node_encoding.unsqueeze(0),
                size=num_nodes,
                mode='linear',
                align_corners=False
            ).squeeze(0)

        x = x.unsqueeze(0)  # [1, num_nodes, embed_dim]

        # エッジ特徴量を投影
        edge_attr = self.edge_projection(graph.edge_attr)  # [num_edges, embed_dim]

        # 距離行列を計算（最短経路距離）
        distance_matrix = self._compute_distance_matrix(graph)

        # Graphormerレイヤーを適用
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, graph.edge_index, edge_attr, distance_matrix, return_attention=True)
            attention_weights.append(attn)
            x = self.dropout(x)

        x = self.layer_norm(x)

        if return_attention:
            return x, attention_weights
        return x, None

    def _compute_distance_matrix(self, graph: ObjectGraph) -> torch.Tensor:
        """最短経路距離行列を計算"""
        num_nodes = graph.node_features.size(0)
        if num_nodes == 0:
            return torch.zeros((1, 0, 0), device=graph.node_features.device)

        # 隣接行列を構築
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=graph.node_features.device)
        if graph.edge_index.size(1) > 0:
            edge_index = graph.edge_index
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
            adj_matrix[edge_index[1], edge_index[0]] = 1.0  # 無向グラフ

        # 最短経路距離を計算（Floyd-Warshallアルゴリズム）
        distance_matrix = adj_matrix.clone()
        distance_matrix[distance_matrix == 0] = float('inf')
        distance_matrix.fill_diagonal_(0)

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

        # 無限大を最大距離に置き換え
        max_distance = num_nodes
        distance_matrix[distance_matrix == float('inf')] = max_distance

        return distance_matrix.unsqueeze(0)  # [1, num_nodes, num_nodes]


class GraphormerLayer(nn.Module):
    """Graphormerレイヤー"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # マルチヘッドアテンション
        self.attention = GraphormerAttention(embed_dim, num_heads, dropout)

        # フィードフォワードネットワーク
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # レイヤー正規化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        distance_matrix: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        フォワードパス

        Args:
            x: [batch, num_nodes, embed_dim] ノード表現
            edge_index: [2, num_edges] エッジインデックス
            edge_attr: [num_edges, embed_dim] エッジ特徴量
            distance_matrix: [batch, num_nodes, num_nodes] 距離行列
            return_attention: アテンション重みを返すかどうか

        Returns:
            torch.Tensor: [batch, num_nodes, embed_dim] 更新されたノード表現
            Optional[torch.Tensor]: アテンション重み
        """
        # セルフアテンション
        residual = x
        x = self.norm1(x)
        x, attn = self.attention(x, edge_index, edge_attr, distance_matrix, return_attention=return_attention)
        x = residual + x

        # フィードフォワード
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn


class GraphormerAttention(nn.Module):
    """Graphormerアテンション"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 距離エンコーディング
        self.distance_encoding = nn.Parameter(torch.randn(100, num_heads))  # 最大距離100

        # エッジエンコーディング
        self.edge_encoding = nn.Linear(embed_dim, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        distance_matrix: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        フォワードパス

        Args:
            x: [batch, num_nodes, embed_dim] ノード表現
            edge_index: [2, num_edges] エッジインデックス
            edge_attr: [num_edges, embed_dim] エッジ特徴量
            distance_matrix: [batch, num_nodes, num_nodes] 距離行列
            return_attention: アテンション重みを返すかどうか

        Returns:
            torch.Tensor: [batch, num_nodes, embed_dim] アテンション後のノード表現
            Optional[torch.Tensor]: アテンション重み
        """
        batch_size, num_nodes, _ = x.shape

        # Query, Key, Valueを計算
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # アテンションスコアを計算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, num_heads, num_nodes, num_nodes]

        # 距離エンコーディングを追加
        distance_matrix_int = distance_matrix.long().clamp(0, 99)
        distance_enc = self.distance_encoding[distance_matrix_int]  # [batch, num_nodes, num_nodes, num_heads]
        distance_enc = distance_enc.permute(0, 3, 1, 2)  # [batch, num_heads, num_nodes, num_nodes]
        attn = attn + distance_enc

        # エッジエンコーディングを追加
        if edge_index.size(1) > 0:
            edge_enc = self.edge_encoding(edge_attr)  # [num_edges, num_heads]
            edge_enc = edge_enc.t()  # [num_heads, num_edges]

            # エッジエンコーディングをアテンション行列に追加
            for i in range(self.num_heads):
                for j in range(edge_index.size(1)):
                    src, tgt = edge_index[0, j].item(), edge_index[1, j].item()
                    attn[:, i, src, tgt] = attn[:, i, src, tgt] + edge_enc[i, j]

        # ソフトマックス
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # アテンションを適用
        out = torch.matmul(attn, v)  # [batch, num_heads, num_nodes, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out, None


class EGNNEncoder(nn.Module):
    """EGNN（E(n)-Equivariant Graph Neural Network）エンコーダー"""

    def __init__(
        self,
        node_feature_dim: int = 12,
        embed_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            node_feature_dim: ノード特徴量の次元
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # ノード特徴量の投影
        self.node_projection = nn.Linear(node_feature_dim, embed_dim)

        # エッジ特徴量の投影
        self.edge_projection = nn.Linear(12, embed_dim)

        # EGNNレイヤー
        self.layers = nn.ModuleList([
            EGNNLayer(embed_dim, dropout)
            for _ in range(num_layers)
        ])

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: ObjectGraph
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            graph: オブジェクトグラフ

        Returns:
            torch.Tensor: [batch, num_nodes, embed_dim] エンコードされたノード表現
        """
        num_nodes = graph.node_features.size(0)
        if num_nodes == 0:
            return torch.zeros((1, 0, self.embed_dim), device=graph.node_features.device)

        # ノード特徴量を投影
        x = self.node_projection(graph.node_features)  # [num_nodes, embed_dim]

        # 位置情報を取得（中心座標）
        pos = torch.stack([
            torch.tensor([obj.center_x, obj.center_y], dtype=torch.float32)
            for obj in graph.nodes
        ], dim=0).to(x.device)  # [num_nodes, 2]

        x = x.unsqueeze(0)  # [1, num_nodes, embed_dim]
        pos = pos.unsqueeze(0)  # [1, num_nodes, 2]

        # エッジ特徴量を投影
        edge_attr = self.edge_projection(graph.edge_attr)  # [num_edges, embed_dim]

        # EGNNレイヤーを適用
        for layer in self.layers:
            x, pos = layer(x, pos, graph.edge_index, edge_attr)
            x = self.dropout(x)

        x = self.layer_norm(x)

        return x


class EGNNLayer(nn.Module):
    """EGNNレイヤー"""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # エッジメッセージネットワーク
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),  # ノード特徴量2つ + 距離
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # ノード更新ネットワーク
        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # 現在の特徴量 + 集約されたメッセージ
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 位置更新ネットワーク
        self.pos_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 1),  # メッセージ + 距離
            nn.SiLU(),
            nn.Linear(1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        フォワードパス

        Args:
            x: [batch, num_nodes, embed_dim] ノード特徴量
            pos: [batch, num_nodes, 2] 位置座標
            edge_index: [2, num_edges] エッジインデックス
            edge_attr: [num_edges, embed_dim] エッジ特徴量

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 更新されたノード特徴量と位置
        """
        batch_size, num_nodes, _ = x.shape

        if edge_index.size(1) == 0:
            return x, pos

        # エッジの距離を計算
        src, tgt = edge_index[0], edge_index[1]
        pos_src = pos[0, src]  # [num_edges, 2]
        pos_tgt = pos[0, tgt]  # [num_edges, 2]
        distances = torch.norm(pos_src - pos_tgt, dim=1, keepdim=True)  # [num_edges, 1]

        # ノード特徴量を取得
        x_src = x[0, src]  # [num_edges, embed_dim]
        x_tgt = x[0, tgt]  # [num_edges, embed_dim]

        # エッジメッセージを計算
        edge_input = torch.cat([x_src, x_tgt, distances], dim=1)  # [num_edges, embed_dim * 2 + 1]
        edge_input = torch.cat([edge_input, edge_attr], dim=1)  # [num_edges, embed_dim * 3 + 1]
        edge_message = self.edge_mlp(edge_input[:, :self.embed_dim * 2 + 1])  # [num_edges, embed_dim]

        # 位置を更新
        pos_diff = pos_tgt - pos_src  # [num_edges, 2]
        pos_norm = torch.norm(pos_diff, dim=1, keepdim=True) + 1e-6  # [num_edges, 1]
        pos_unit = pos_diff / pos_norm  # [num_edges, 2]

        pos_mlp_input = torch.cat([edge_message, distances], dim=1)  # [num_edges, embed_dim + 1]
        pos_weights = self.pos_mlp(pos_mlp_input)  # [num_edges, 1]

        pos_update = pos_unit * pos_weights  # [num_edges, 2]

        # 位置を集約
        pos_new = pos.clone()
        pos_new[0].index_add_(0, tgt, pos_update)

        # ノードメッセージを集約
        node_messages = torch.zeros_like(x[0])  # [num_nodes, embed_dim]
        node_messages.index_add_(0, tgt, edge_message)

        # ノード特徴量を更新
        node_input = torch.cat([x[0], node_messages], dim=1)  # [num_nodes, embed_dim * 2]
        x_new = x.clone()
        x_new[0] = x[0] + self.node_mlp(node_input)

        return x_new, pos_new


class ObjectGraphEncoder(nn.Module):
    """オブジェクトグラフエンコーダー（統一インターフェース）"""

    def __init__(
        self,
        encoder_type: str = "graphormer",
        node_feature_dim: int = 12,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 100
    ):
        """
        初期化

        Args:
            encoder_type: エンコーダータイプ ("graphormer" or "egnn")
            node_feature_dim: ノード特徴量の次元
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数（Graphormerのみ）
            dropout: ドロップアウト率
            max_nodes: 最大ノード数（Graphormerのみ）
        """
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "graphormer":
            self.encoder = GraphormerEncoder(
                node_feature_dim=node_feature_dim,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_nodes=max_nodes
            )
        elif encoder_type == "egnn":
            self.encoder = EGNNEncoder(
                node_feature_dim=node_feature_dim,
                embed_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self,
        graph: ObjectGraph,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        フォワードパス

        Args:
            graph: オブジェクトグラフ
            return_attention: アテンション重みを返すかどうか（Graphormerのみ）

        Returns:
            torch.Tensor: [batch, num_nodes, embed_dim] エンコードされたノード表現
            Optional[torch.Tensor]: アテンション重み（Graphormerのみ）
        """
        if self.encoder_type == "graphormer":
            return self.encoder(graph, return_attention=return_attention)
        else:
            x = self.encoder(graph)
            return x, None
