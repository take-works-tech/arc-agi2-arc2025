"""
オブジェクトエンコーダー

オブジェクトリストをエンコードするトランスフォーマーベースのエンコーダー
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np

from src.data_systems.data_models.core.object import Object
from src.hybrid_system.models.program_synthesis.object_canonicalizer import CanonicalizedObject
from src.hybrid_system.models.program_synthesis.abstract_object_patterns import (
    AbstractObjectPatternExtractor,
    AbstractObjectPattern
)


class ObjectEncoder(nn.Module):
    """
    オブジェクトエンコーダー

    オブジェクトリストを受け取り、エンコードされた表現を出力
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_objects: int = 100,
        max_grid_size: int = 30,
        enable_graph_fusion: bool = True,
        graph_fusion_type: str = "concat",
        enable_abstract_patterns: bool = False,
        abstract_pattern_types: Optional[List[str]] = None
    ):
        """初期化

        Args:
            embed_dim: 埋め込み次元
            num_layers: トランスフォーマーレイヤー数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            max_objects: 最大オブジェクト数
            max_grid_size: 最大グリッドサイズ
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_objects = max_objects
        self.max_grid_size = max_grid_size
        self.enable_graph_fusion = enable_graph_fusion
        self.graph_fusion_type = graph_fusion_type
        self.enable_abstract_patterns = enable_abstract_patterns
        self.abstract_pattern_types = abstract_pattern_types or [AbstractObjectPattern.FULL]

        # Abstract Object Patterns抽出器
        if self.enable_abstract_patterns:
            self.abstract_pattern_extractor = AbstractObjectPatternExtractor(
                enable_color_agnostic=AbstractObjectPattern.COLOR_AGNOSTIC in self.abstract_pattern_types,
                enable_relative_coordinates=AbstractObjectPattern.RELATIVE_COORDINATES in self.abstract_pattern_types,
                enable_size_only=AbstractObjectPattern.SIZE_ONLY in self.abstract_pattern_types
            )
            # 抽象パターン特徴量の投影（各パターンタイプごと）
            self.abstract_pattern_projections = nn.ModuleDict()
            for pattern_type in self.abstract_pattern_types:
                if pattern_type == AbstractObjectPattern.COLOR_AGNOSTIC:
                    pattern_dim = 11
                elif pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
                    pattern_dim = 8
                elif pattern_type == AbstractObjectPattern.SIZE_ONLY:
                    pattern_dim = 6
                else:  # FULL
                    pattern_dim = 12
                self.abstract_pattern_projections[pattern_type] = nn.Linear(pattern_dim, embed_dim)
        else:
            self.abstract_pattern_extractor = None
            self.abstract_pattern_projections = None

        # オブジェクト特徴量の次元
        # 色(1) + 位置(4: bbox) + サイズ(2: width, height) + その他(5: area, center_x, center_y, density, perimeter)
        self.object_feature_dim = 1 + 4 + 2 + 5  # 12次元

        # オブジェクト特徴量を埋め込み次元に投影
        self.object_projection = nn.Linear(self.object_feature_dim, embed_dim)

        # 色の埋め込み（0-9の10色）
        self.color_embedding = nn.Embedding(10, embed_dim)

        # 色役割の埋め込み（background, foreground, structure, otherの4種類）
        self.color_role_embedding = nn.Embedding(4, embed_dim // 4)  # 色役割の埋め込み（次元を削減）
        self.color_role_names = ['background', 'foreground', 'structure', 'other']

        # 位置エンコーディング（bbox座標を正規化）
        self.position_projection = nn.Linear(4, embed_dim)

        # サイズエンコーディング
        self.size_projection = nn.Linear(2, embed_dim)

        # その他の特徴量の投影
        self.other_features_projection = nn.Linear(5, embed_dim)

        # グリッドサイズの投影（背景色とグリッドサイズの情報を追加するため）
        self.grid_size_projection = nn.Linear(2, embed_dim)

        # トランスフォーマーエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # グラフ特徴量の融合（有効な場合）
        if self.enable_graph_fusion:
            if self.graph_fusion_type == "concat":
                # 連結方式: オブジェクト特徴量とグラフ特徴量を連結して融合
                self.graph_fusion = nn.Linear(embed_dim * 2, embed_dim)
            elif self.graph_fusion_type == "attention":
                # アテンション方式: グラフ特徴量をキー・バリューとして使用
                self.graph_attention = nn.MultiheadAttention(
                    embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
                )
            else:
                raise ValueError(f"Unknown graph_fusion_type: {graph_fusion_type}")

    def _extract_object_features(self, objects: List[Object], grid_width: int, grid_height: int) -> torch.Tensor:
        """オブジェクトから特徴量を抽出

        Args:
            objects: オブジェクトリスト
            grid_width: グリッド幅（正規化用）
            grid_height: グリッド高さ（正規化用）

        Returns:
            features: 特徴量テンソル [num_objects, object_feature_dim]
        """
        if not objects:
            # 空のオブジェクトリストの場合、ゼロパディング
            return torch.zeros(1, self.object_feature_dim)

        features = []
        for obj in objects:
            # 色 (1次元)
            color = obj.dominant_color if hasattr(obj, 'dominant_color') else obj.color if hasattr(obj, 'color') else 0

            # 位置 (4次元: bbox座標を正規化)
            bbox = obj.bbox if hasattr(obj, 'bbox') else (0, 0, 0, 0)
            x1, y1, x2, y2 = bbox
            normalized_bbox = [
                x1 / max(grid_width, 1),
                y1 / max(grid_height, 1),
                x2 / max(grid_width, 1),
                y2 / max(grid_height, 1)
            ]

            # サイズ (2次元: width, height)
            width = obj.bbox_width if hasattr(obj, 'bbox_width') else 0
            height = obj.bbox_height if hasattr(obj, 'bbox_height') else 0
            normalized_size = [
                width / max(grid_width, 1),
                height / max(grid_height, 1)
            ]

            # その他の特徴量 (5次元)
            area = obj.area if hasattr(obj, 'area') else 0
            center_x = obj.center_x if hasattr(obj, 'center_x') else 0
            center_y = obj.center_y if hasattr(obj, 'center_y') else 0
            density = obj.density if hasattr(obj, 'density') else 0.0
            perimeter = obj.perimeter if hasattr(obj, 'perimeter') else 0

            normalized_other = [
                area / max(grid_width * grid_height, 1),
                center_x / max(grid_width, 1),
                center_y / max(grid_height, 1),
                density,
                perimeter / max(grid_width + grid_height, 1)
            ]

            # 特徴量を結合
            obj_features = [color] + normalized_bbox + normalized_size + normalized_other
            features.append(obj_features)

        return torch.tensor(features, dtype=torch.float32)

    def _extract_canonicalized_features(
        self,
        canonicalized_objects: List[CanonicalizedObject],
        grid_width: int,
        grid_height: int
    ) -> torch.Tensor:
        """正規化されたオブジェクトから特徴量を抽出

        Args:
            canonicalized_objects: 正規化されたオブジェクトリスト
            grid_width: グリッド幅（参照用）
            grid_height: グリッド高さ（参照用）

        Returns:
            features: 特徴量テンソル [num_objects, object_feature_dim]
        """
        if not canonicalized_objects:
            # 空のオブジェクトリストの場合、ゼロパディング
            return torch.zeros(1, self.object_feature_dim)

        features = []
        for canon_obj in canonicalized_objects:
            # 正規化された色 (1次元)
            color = canon_obj.remapped_color

            # 正規化された位置からbboxを計算 (4次元)
            # normalized_positionは中心座標なので、bboxに変換
            norm_x, norm_y = canon_obj.normalized_position
            norm_width, norm_height = canon_obj.normalized_size

            # bbox座標を計算（正規化済み）
            x1 = norm_x - norm_width / 2
            y1 = norm_y - norm_height / 2
            x2 = norm_x + norm_width / 2
            y2 = norm_y + norm_height / 2
            normalized_bbox = [x1, y1, x2, y2]

            # 正規化されたサイズ (2次元)
            normalized_size = [norm_width, norm_height]

            # その他の特徴量 (5次元)
            # 元のオブジェクトから取得（正規化済み）
            obj = canon_obj.original_object
            area = obj.area if hasattr(obj, 'area') else 0
            center_x = norm_x  # 正規化済み
            center_y = norm_y  # 正規化済み
            density = obj.density if hasattr(obj, 'density') else 0.0
            perimeter = obj.perimeter if hasattr(obj, 'perimeter') else 0

            normalized_other = [
                area / max(grid_width * grid_height, 1),
                center_x,  # 既に正規化済み
                center_y,  # 既に正規化済み
                density,
                perimeter / max(grid_width + grid_height, 1)
            ]

            # 形状埋め込みを追加（正規化オブジェクトの特徴）
            # 本格実装: 形状埋め込みを独立した特徴量として扱い、後で埋め込みに加算
            # ここでは、形状埋め込みの情報をother_featuresに含める（形状情報の強化）
            shape_emb = canon_obj.shape_embedding
            shape_emb_dim = len(shape_emb) if isinstance(shape_emb, np.ndarray) else 0

            # 形状埋め込みから主要な特徴量を抽出してother_featuresに追加
            # 形状埋め込みは通常5-10次元で、以下の情報を含む:
            # - アスペクト比、面積比、周囲長/面積比、Huモーメントなど
            # 既存のother_features（5次元）に形状情報を統合
            if shape_emb_dim >= 3:
                # 形状埋め込みの主要な3次元（アスペクト比、面積比、周囲長/面積比）を
                # other_featuresの一部として使用（密度と周囲長の情報を強化）
                # 既存のnormalized_otherの密度と周囲長を形状情報で補強
                aspect_ratio = float(shape_emb[0]) if shape_emb_dim > 0 else 1.0
                area_ratio = float(shape_emb[1]) if shape_emb_dim > 1 else 1.0
                perimeter_area_ratio = float(shape_emb[2]) if shape_emb_dim > 2 else 0.0

                # 形状情報をother_featuresに統合（既存の5次元を保持しつつ、形状情報で補強）
                # 密度と周囲長の計算に形状情報を反映
                enhanced_density = density * area_ratio  # 面積比で密度を調整
                enhanced_perimeter = perimeter * (1.0 + perimeter_area_ratio)  # 周囲長/面積比で周囲長を調整

                normalized_other = [
                    area / max(grid_width * grid_height, 1),
                    center_x,
                    center_y,
                    enhanced_density,
                    enhanced_perimeter / max(grid_width + grid_height, 1)
                ]
            else:
                # 形状埋め込みが利用できない場合は既存の方法を使用
                normalized_other = [
                    area / max(grid_width * grid_height, 1),
                    center_x,
                    center_y,
                    density,
                    perimeter / max(grid_width + grid_height, 1)
                ]

            # 特徴量を結合
            obj_features = [color] + normalized_bbox + normalized_size + normalized_other
            features.append(obj_features)

        return torch.tensor(features, dtype=torch.float32)

    def forward(
        self,
        objects: List[Object],
        background_color: int,
        grid_width: int,
        grid_height: int,
        graph_encoded: Optional[torch.Tensor] = None,
        canonicalized_objects: Optional[List[CanonicalizedObject]] = None,
        color_roles: Optional[Dict[int, str]] = None
    ) -> torch.Tensor:
        """順伝播

        Args:
            objects: オブジェクトリスト
            background_color: 背景色
            grid_width: グリッド幅
            grid_height: グリッド高さ
            graph_encoded: グラフエンコードされた特徴量 [1, num_nodes, embed_dim]（オプション）
            canonicalized_objects: 正規化されたオブジェクトリスト（オプション）
                - 指定された場合、正規化された特徴量を使用

        Returns:
            encoded: エンコードされた表現 [1, num_objects, embed_dim]
        """
        # オブジェクト特徴量を抽出
        # Abstract Object Patternsが有効な場合は抽象パターンを使用
        use_abstract_patterns = False
        if self.enable_abstract_patterns and self.abstract_pattern_extractor and not canonicalized_objects:
            # 複数の抽象パターンを抽出して融合
            abstract_patterns = self.abstract_pattern_extractor.extract_multiple_patterns(
                objects, grid_width, grid_height
            )
            # 各パターンタイプの特徴量を投影して結合
            pattern_embeddings = []
            for pattern_type in self.abstract_pattern_types:
                if pattern_type in abstract_patterns:
                    patterns = abstract_patterns[pattern_type]
                    if patterns:
                        # パターンをテンソルに変換
                        pattern_values = []
                        for pattern in patterns:
                            if pattern_type == AbstractObjectPattern.COLOR_AGNOSTIC:
                                values = [
                                    pattern['bbox_left'], pattern['bbox_top'],
                                    pattern['bbox_right'], pattern['bbox_bottom'],
                                    pattern['width'], pattern['height'],
                                    pattern['area'], pattern['center_x'], pattern['center_y'],
                                    pattern['density'], pattern['perimeter']
                                ]
                            elif pattern_type == AbstractObjectPattern.RELATIVE_COORDINATES:
                                values = [
                                    pattern['color'],
                                    pattern['relative_center_x'], pattern['relative_center_y'],
                                    pattern['width'], pattern['height'],
                                    pattern['area'], pattern['density'], pattern['perimeter']
                                ]
                            elif pattern_type == AbstractObjectPattern.SIZE_ONLY:
                                values = [
                                    pattern['width'], pattern['height'],
                                    pattern['area'], pattern['aspect_ratio'],
                                    pattern['density'], pattern['perimeter']
                                ]
                            else:  # FULL
                                values = [
                                    pattern['color'],
                                    pattern['bbox_left'], pattern['bbox_top'],
                                    pattern['bbox_right'], pattern['bbox_bottom'],
                                    pattern['width'], pattern['height'],
                                    pattern['area'], pattern['center_x'], pattern['center_y'],
                                    pattern['density'], pattern['perimeter']
                                ]
                            pattern_values.append(values)
                        pattern_tensor = torch.tensor(pattern_values, dtype=torch.float32)
                        # 投影
                        pattern_embed = self.abstract_pattern_projections[pattern_type](pattern_tensor)
                        pattern_embeddings.append(pattern_embed)

            # 抽象パターンの埋め込みを平均または結合
            if pattern_embeddings:
                # 平均を取る
                if len(pattern_embeddings) == 1:
                    object_features = pattern_embeddings[0]
                else:
                    # 複数のパターンを平均
                    object_features = torch.stack(pattern_embeddings, dim=0).mean(dim=0)
                use_abstract_patterns = True
                objects_for_color = objects

        if not use_abstract_patterns:
            # 通常の特徴量抽出
            if canonicalized_objects:
                object_features = self._extract_canonicalized_features(
                    canonicalized_objects, grid_width, grid_height
                )  # [num_objects, object_feature_dim]
                # 正規化されたオブジェクトから元のオブジェクトリストを取得（色の埋め込み用）
                objects_for_color = [canon_obj.original_object for canon_obj in canonicalized_objects]
            else:
                object_features = self._extract_object_features(objects, grid_width, grid_height)  # [num_objects, object_feature_dim]
                objects_for_color = objects

        # バッチ次元を追加
        if object_features.dim() == 2:
            object_features = object_features.unsqueeze(0)  # [1, num_objects, object_feature_dim]

        num_objects = object_features.shape[1]

        # オブジェクト特徴量を埋め込み次元に投影
        # Abstract Object Patternsが有効な場合、既に投影済みの可能性がある
        if use_abstract_patterns and object_features.shape[-1] == self.embed_dim:
            # 既に投影済み（抽象パターンの投影結果）
            object_embed = object_features
        else:
            # 通常の投影
            object_embed = self.object_projection(object_features)  # [1, num_objects, embed_dim]

        # 色の埋め込みを追加（各オブジェクトの色）
        # 正規化されたオブジェクトがある場合は、リマップされた色を使用
        if objects_for_color:
            if canonicalized_objects:
                # 正規化されたオブジェクトのリマップされた色を使用
                colors = torch.tensor([
                    canon_obj.remapped_color for canon_obj in canonicalized_objects
                ], dtype=torch.long, device=object_features.device)
            else:
                # 通常のオブジェクトの色を使用
                colors = torch.tensor([
                    obj.dominant_color if hasattr(obj, 'dominant_color') else obj.color if hasattr(obj, 'color') else 0
                    for obj in objects_for_color
                ], dtype=torch.long, device=object_features.device)
            color_embed = self.color_embedding(colors)  # [num_objects, embed_dim]

            # 色役割の埋め込みを追加（オプション）
            if color_roles is not None:
                # 各オブジェクトの色に対応する色役割を取得
                role_indices = []
                for color in colors.cpu().numpy():
                    role = color_roles.get(int(color), 'other')
                    role_idx = self.color_role_names.index(role) if role in self.color_role_names else 3  # 3 = 'other'
                    role_indices.append(role_idx)
                role_indices_tensor = torch.tensor(role_indices, dtype=torch.long, device=object_features.device)
                role_embed = self.color_role_embedding(role_indices_tensor)  # [num_objects, embed_dim // 4]
                # 色役割の埋め込みを埋め込み次元に拡張
                role_embed_expanded = torch.zeros(num_objects, self.embed_dim, device=object_features.device)
                role_embed_expanded[:, :role_embed.size(1)] = role_embed
                # 色の埋め込みと色役割の埋め込みを結合
                color_embed = color_embed + role_embed_expanded  # [num_objects, embed_dim]

            color_embed = color_embed.unsqueeze(0)  # [1, num_objects, embed_dim]
            object_embed = object_embed + color_embed

        # 背景色とグリッドサイズの情報を追加（グローバルコンテキストとして、本格実装）
        # すべてのオブジェクト埋め込みにグローバルコンテキストを追加
        if num_objects > 0:
            bg_color_embed = self.color_embedding(torch.tensor([background_color], dtype=torch.long, device=object_features.device))
            grid_size_embed = torch.tensor([
                [grid_width / self.max_grid_size, grid_height / self.max_grid_size]
            ], dtype=torch.float32, device=object_features.device)
            grid_size_proj = self.grid_size_projection(grid_size_embed)

            # グローバルコンテキストを計算
            global_context = bg_color_embed + grid_size_proj

            # すべてのオブジェクト埋め込みにグローバルコンテキストを追加
            # 最初のオブジェクトにより多くの重みを付ける
            for i in range(num_objects):
                weight = 1.0 if i == 0 else 0.5  # 最初のオブジェクトに2倍の重み
                object_embed[:, i, :] = object_embed[:, i, :] + global_context * weight

        # トランスフォーマーでエンコード
        encoded = self.transformer(object_embed)  # [1, num_objects, embed_dim]

        # グラフ特徴量がある場合、融合
        if self.enable_graph_fusion and graph_encoded is not None:
            num_objects = encoded.shape[1]
            num_nodes = graph_encoded.shape[1]

            if num_objects > 0 and num_nodes > 0:
                # オブジェクトとノードの対応関係を考慮して融合
                # 簡易版: オブジェクト数とノード数が同じ場合、1対1で対応
                if num_objects == num_nodes:
                    if self.graph_fusion_type == "concat":
                        # 連結方式: オブジェクト特徴量とグラフ特徴量を連結
                        fused = torch.cat([encoded, graph_encoded], dim=-1)  # [1, num_objects, embed_dim * 2]
                        encoded = self.graph_fusion(fused)  # [1, num_objects, embed_dim]
                    elif self.graph_fusion_type == "attention":
                        # アテンション方式: グラフ特徴量をキー・バリューとして使用
                        encoded, _ = self.graph_attention(
                            encoded, graph_encoded, graph_encoded
                        )  # [1, num_objects, embed_dim]
                else:
                    # オブジェクト数とノード数が異なる場合の処理（本格実装）
                    # 通常、グラフのノードはオブジェクトと1対1対応しているはずだが、
                    # GNNエンコーダーやグラフ構築時のフィルタリングにより異なる場合がある
                    if num_nodes < num_objects:
                        # ノード数が少ない場合: 線形補間で拡張
                        # 各オブジェクトに対して、最も近いノードの特徴量を使用
                        # または、ノード特徴量を平均プーリングして全オブジェクトに適用
                        if num_nodes > 0:
                            # ノード特徴量の平均を計算
                            graph_pooled = graph_encoded.mean(dim=1, keepdim=True)  # [1, 1, embed_dim]
                            # 全オブジェクトに同じ特徴量を適用
                            graph_encoded_expanded = graph_pooled.expand(-1, num_objects, -1)  # [1, num_objects, embed_dim]
                        else:
                            # ノードがない場合、ゼロ埋め込みを使用
                            graph_encoded_expanded = torch.zeros(1, num_objects, self.embed_dim, device=encoded.device)
                    else:
                        # ノード数が多い場合: 平均プーリングで縮小
                        # オブジェクト数に合わせて、ノード特徴量を平均プーリング
                        # より正確には、オブジェクトとノードの対応関係を使用するが、
                        # 現在は対応関係が利用できないため、均等に分割して平均を取る
                        if num_objects > 0:
                            # ノードをnum_objects個のグループに分割し、各グループの平均を計算
                            nodes_per_object = num_nodes / num_objects
                            graph_encoded_expanded = torch.zeros(1, num_objects, self.embed_dim, device=encoded.device)

                            for obj_idx in range(num_objects):
                                start_idx = int(obj_idx * nodes_per_object)
                                end_idx = int((obj_idx + 1) * nodes_per_object) if obj_idx < num_objects - 1 else num_nodes
                                # グループ内のノード特徴量の平均を計算
                                group_features = graph_encoded[:, start_idx:end_idx, :]  # [1, group_size, embed_dim]
                                group_mean = group_features.mean(dim=1, keepdim=True)  # [1, 1, embed_dim]
                                graph_encoded_expanded[:, obj_idx:obj_idx+1, :] = group_mean
                        else:
                            graph_encoded_expanded = torch.zeros(1, 0, self.embed_dim, device=encoded.device)

                    if self.graph_fusion_type == "concat":
                        fused = torch.cat([encoded, graph_encoded_expanded], dim=-1)
                        encoded = self.graph_fusion(fused)
                    elif self.graph_fusion_type == "attention":
                        encoded, _ = self.graph_attention(
                            encoded, graph_encoded_expanded, graph_encoded_expanded
                        )

        # レイヤー正規化
        encoded = self.layer_norm(encoded)

        return encoded
