"""
オブジェクトベースのニューラルモデル候補生成器（⑤深層学習ベース２: オブジェクト→プログラム）

改善版: Object Graph + GNN、Object Canonicalization、Relation Classifierを統合
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
import numpy as np

from src.hybrid_system.ir.serialization import template_string_to_sequence
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl
from src.data_systems.data_models.core.object import Object
from src.core_systems.executor.core import ExecutorCore
from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType
from .common_helpers import get_background_color, get_grid_size

# 新規追加: Object Graph + GNN関連
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder
from src.hybrid_system.models.program_synthesis.object_graph_encoder import ObjectGraphEncoder
from src.hybrid_system.models.program_synthesis.object_canonicalizer import ObjectCanonicalizer
from src.hybrid_system.models.program_synthesis.relation_classifier import RelationClassifier

# ObjectInfoからObjectへの変換用
from src.hybrid_system.inference.object_matching.data_structures import ObjectInfo


class NeuralObjectCandidateGenerator:
    """オブジェクトベースのニューラルモデル候補生成器（改善版）"""

    def __init__(
        self,
        neural_object_model=None,
        tokenizer=None,
        enable_object_canonicalization: bool = True,
        enable_object_graph: bool = True,
        enable_relation_classifier: bool = True,
        graph_encoder_type: str = "graphormer",
        relation_classifier_threshold: float = 0.7
    ):
        """
        初期化

        Args:
            neural_object_model: ObjectBasedProgramSynthesisModel
            tokenizer: ProgramTokenizer
            enable_object_canonicalization: Object Canonicalizationを有効化
            enable_object_graph: Object Graph + GNNを有効化
            enable_relation_classifier: Relation Classifierを有効化
            graph_encoder_type: グラフエンコーダータイプ ("graphormer" or "egnn")
            relation_classifier_threshold: Relation Classifierの閾値
        """
        self.neural_object_model = neural_object_model
        self.tokenizer = tokenizer
        self.executor = ExecutorCore()
        self.object_extractor = IntegratedObjectExtractor(ExtractionConfig())

        # 新規追加: 改善モジュール
        self.enable_object_canonicalization = enable_object_canonicalization
        self.enable_object_graph = enable_object_graph
        self.enable_relation_classifier = enable_relation_classifier
        self.graph_encoder_type = graph_encoder_type
        self.relation_classifier_threshold = relation_classifier_threshold

        # Object Graph Builder
        if self.enable_object_graph:
            self.graph_builder = ObjectGraphBuilder(max_grid_size=30)

        # Object Graph Encoder
        if self.enable_object_graph and self.neural_object_model is not None:
            embed_dim = getattr(neural_object_model.program_config, 'grid_encoder_dim', 256)
            self.graph_encoder = ObjectGraphEncoder(
                encoder_type=graph_encoder_type,
                node_feature_dim=12,
                embed_dim=embed_dim,
                num_layers=4,
                num_heads=8,
                dropout=0.1,
                max_nodes=100
            )
            # デバイスを設定
            if hasattr(neural_object_model, 'device'):
                self.graph_encoder = self.graph_encoder.to(neural_object_model.device)
        else:
            self.graph_encoder = None

        # Object Canonicalizer
        if self.enable_object_canonicalization:
            self.canonicalizer = ObjectCanonicalizer(
                enable_color_remap=True,
                enable_position_normalize=True,
                enable_size_normalize=True,
                enable_shape_embedding=True,
                shape_embedding_dim=8
            )
        else:
            self.canonicalizer = None

        # Relation Classifier
        if self.enable_relation_classifier and self.neural_object_model is not None:
            embed_dim = getattr(neural_object_model.program_config, 'grid_encoder_dim', 256)
            self.relation_classifier = RelationClassifier(
                node_feature_dim=12,
                embed_dim=128,
                num_relation_types=8,
                dropout=0.1
            )
            # デバイスを設定
            if hasattr(neural_object_model, 'device'):
                self.relation_classifier = self.relation_classifier.to(neural_object_model.device)
        else:
            self.relation_classifier = None

    def _convert_objectinfo_to_object(
        self,
        obj_info: ObjectInfo,
        grid: np.ndarray,
        object_id: str
    ) -> Object:
        """
        ObjectInfoからObjectに変換

        Args:
            obj_info: ObjectInfoオブジェクト
            grid: グリッド（numpy配列）
            object_id: オブジェクトID

        Returns:
            Objectオブジェクト
        """
        # ピクセルごとの色を取得
        pixel_colors = {}
        for pixel in obj_info.pixels:
            if isinstance(pixel, tuple) and len(pixel) == 2:
                # ObjectInfoのpixelsは(min_i, min_j)形式（行、列）
                # Objectのpixelsは(x, y)形式（列、行）
                row, col = pixel
                if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                    # Objectのpixelsは(x, y)形式なので、col, rowの順
                    pixel_colors[(col, row)] = grid[row, col]

        # bboxの形式を変換
        # ObjectInfoのbboxは(min_i, min_j, max_i, max_j)形式（行、列）
        # Objectのbboxは(x1, y1, x2, y2)形式（列、行）
        if hasattr(obj_info, 'bbox') and obj_info.bbox and len(obj_info.bbox) == 4:
            min_i, min_j, max_i, max_j = obj_info.bbox
            bbox = (min_j, min_i, max_j, max_i)  # (x1, y1, x2, y2)
        else:
            bbox = (0, 0, 0, 0)

        # Objectを作成
        obj = Object(
            object_id=object_id,
            object_type=ObjectType.SINGLE_COLOR_4WAY,
            pixels=[(col, row) for row, col in obj_info.pixels] if obj_info.pixels else [],  # (x, y)形式に変換
            pixel_colors=pixel_colors,
            bbox=bbox,
            source_image_id=f"input_{obj_info.grid_index}" if hasattr(obj_info, 'grid_index') else "input_0",
            source_image_type="input",
            source_image_index=obj_info.grid_index if hasattr(obj_info, 'grid_index') else 0
        )

        # グリッドを設定して色情報を更新
        obj._grid = grid
        obj.update_from_grid(grid)

        return obj

    def _extract_variables_from_partial_program(self, partial_program: str) -> set:
        """
        部分プログラムから変数名を抽出

        Args:
            partial_program: 部分プログラム文字列

        Returns:
            変数名のセット（例: {'objects1', 'objects2'}）
        """
        import re
        if not partial_program:
            return set()

        # 変数名を抽出: "objects1 = ...", "objects2_31 = ..." など
        var_matches = re.findall(r'objects(\d+)(?:_\d+)?\s*=', partial_program)
        variables = set([f"objects{i}" for i in var_matches])
        return variables

    # _find_matching_category_var_mappingメソッドは削除
    # partial_program_to_category_var_mappingが常に利用可能なため不要

    def _convert_categories_to_dict(
        self,
        categories_list: List[Any],
        objects: List[Object]
    ) -> Optional[Dict[int, List[int]]]:
        """
        CategoryInfoのリストをDict[int, List[int]]形式に変換

        Args:
            categories_list: CategoryInfoのリスト
            objects: オブジェクトリスト（インデックス対応用）

        Returns:
            {category_id: [object_indices]} の形式、またはNone
        """
        if not categories_list or not objects:
            return None

        # オブジェクトのシグネチャ（pixelsとcolor）からインデックスへのマッピングを作成
        # Objectのpixelsは(x, y)形式（列、行）
        obj_signature_to_index = {}
        for idx, obj in enumerate(objects):
            if hasattr(obj, 'pixels') and hasattr(obj, 'dominant_color'):
                # Objectのpixelsは(x, y)形式
                pixels_set = frozenset(obj.pixels) if obj.pixels else frozenset()
                signature = (obj.dominant_color, pixels_set)
                obj_signature_to_index[signature] = idx

        # カテゴリ情報をDict[int, List[int]]形式に変換
        categories_dict = {}
        for category in categories_list:
            if not hasattr(category, 'category_id') or not hasattr(category, 'objects'):
                continue

            category_id = category.category_id
            # カテゴリIDが文字列の場合（"{connectivity}_{pattern_idx}_{original_id}"形式）、
            # 数値部分を抽出（最後の部分を使用）
            if isinstance(category_id, str):
                try:
                    # 最後の数値部分を抽出
                    parts = category_id.split('_')
                    if len(parts) >= 3:
                        category_id = int(parts[-1])
                    else:
                        # 数値部分がない場合はハッシュを使用
                        category_id = hash(category_id) % 1000
                except (ValueError, IndexError):
                    category_id = hash(category_id) % 1000

            # カテゴリ内のオブジェクトのインデックスを取得
            # ObjectInfoのpixelsは(row, col)形式（行、列）なので、Objectの(x, y)形式に変換
            object_indices = []
            for obj_info in category.objects:
                if hasattr(obj_info, 'pixels') and hasattr(obj_info, 'color'):
                    # ObjectInfoのpixelsは(row, col)形式、Objectのpixelsは(x, y)形式
                    # 変換: (row, col) -> (col, row) = (x, y)
                    if obj_info.pixels:
                        pixels_converted = [(col, row) for row, col in obj_info.pixels]
                        pixels_set = frozenset(pixels_converted)
                    else:
                        pixels_set = frozenset()
                    signature = (obj_info.color, pixels_set)
                    if signature in obj_signature_to_index:
                        object_indices.append(obj_signature_to_index[signature])

            if object_indices:
                # 同じカテゴリIDが既に存在する場合は統合
                if category_id in categories_dict:
                    # 既存のインデックスと統合（重複を除去）
                    existing_indices = set(categories_dict[category_id])
                    new_indices = set(object_indices)
                    categories_dict[category_id] = list(existing_indices | new_indices)
                else:
                    categories_dict[category_id] = object_indices

        return categories_dict if categories_dict else None

    def _adjust_score_with_relations(
        self,
        program: str,
        relation_info: Optional[Dict[Tuple[int, int], List[str]]],
        objects: List[Object],
        category_var_mapping: Optional[Dict[str, str]],
        base_score: float,
        matching_result: Optional[Dict[str, Any]] = None,
        partial_program: Optional[str] = None
    ) -> float:
        """関係情報に基づいてスコアを調整（本格実装）

        Args:
            program: 生成されたプログラム文字列
            relation_info: 関係情報 {(src_idx, tgt_idx): [relation_types]}
            objects: オブジェクトリスト
            category_var_mapping: カテゴリIDと変数名の対応関係 {category_id: variable_name}
            base_score: ベーススコア
            matching_result: オブジェクトマッチング結果（オプション、オブジェクトとカテゴリの対応関係構築用）
            partial_program: 部分プログラム（オプション、カテゴリ情報取得用）

        Returns:
            adjusted_score: 調整後のスコア
        """
        if not relation_info:
            return base_score

        import re

        # プログラムから使用されている変数名を抽出
        # 例: "FOR i LEN(objects1) DO ..." -> objects1を使用
        used_vars = set(re.findall(r'objects\d+', program))

        if not used_vars:
            return base_score

        # オブジェクトインデックスから変数名へのマッピングを作成（本格実装）
        obj_idx_to_var = {}

        # オブジェクトとカテゴリの対応関係を構築
        obj_idx_to_category_id = {}
        if matching_result and partial_program and category_var_mapping:
            # partial_program_to_categoriesからカテゴリ情報を取得
            partial_program_to_categories = matching_result.get('partial_program_to_categories', {})
            if partial_program in partial_program_to_categories:
                categories_list = partial_program_to_categories[partial_program]

                # オブジェクトのシグネチャ（pixelsとcolor）からインデックスへのマッピングを作成
                obj_signature_to_index = {}
                for idx, obj in enumerate(objects):
                    if hasattr(obj, 'pixels') and hasattr(obj, 'dominant_color'):
                        pixels_set = frozenset(obj.pixels) if obj.pixels else frozenset()
                        signature = (obj.dominant_color, pixels_set)
                        obj_signature_to_index[signature] = idx

                # カテゴリ情報からオブジェクトインデックスとカテゴリIDの対応関係を構築
                for category in categories_list:
                    if not hasattr(category, 'category_id') or not hasattr(category, 'objects'):
                        continue

                    category_id = category.category_id
                    # カテゴリIDが文字列の場合、数値部分を抽出
                    if isinstance(category_id, str):
                        try:
                            parts = category_id.split('_')
                            if len(parts) >= 3:
                                category_id = int(parts[-1])
                            else:
                                category_id = hash(category_id) % 1000
                        except (ValueError, IndexError):
                            category_id = hash(category_id) % 1000

                    # カテゴリ内のオブジェクトのインデックスを取得
                    for obj_info in category.objects:
                        if hasattr(obj_info, 'pixels') and hasattr(obj_info, 'color'):
                            # ObjectInfoのpixelsは(row, col)形式、Objectのpixelsは(x, y)形式
                            if obj_info.pixels:
                                pixels_converted = [(col, row) for row, col in obj_info.pixels]
                                pixels_set = frozenset(pixels_converted)
                            else:
                                pixels_set = frozenset()
                            signature = (obj_info.color, pixels_set)
                            if signature in obj_signature_to_index:
                                obj_idx = obj_signature_to_index[signature]
                                obj_idx_to_category_id[obj_idx] = category_id

        # オブジェクトインデックスから変数名へのマッピングを作成
        if category_var_mapping and obj_idx_to_category_id:
            # オブジェクトとカテゴリの対応関係を使用
            for obj_idx, category_id in obj_idx_to_category_id.items():
                # カテゴリIDを文字列に変換してcategory_var_mappingから取得
                category_id_str = str(category_id)
                if category_id_str in category_var_mapping:
                    obj_idx_to_var[obj_idx] = category_var_mapping[category_id_str]
                else:
                    # カテゴリIDが見つからない場合、フォールバック
                    obj_idx_to_var[obj_idx] = f"objects{obj_idx + 1}"

            # マッピングされていないオブジェクトに対しては、インデックス+1を使用
            for i in range(len(objects)):
                if i not in obj_idx_to_var:
                    obj_idx_to_var[i] = f"objects{i + 1}"
        else:
            # category_var_mappingがない場合、または対応関係が構築できなかった場合、インデックス+1を使用
            for i in range(len(objects)):
                obj_idx_to_var[i] = f"objects{i + 1}"

        # 関係情報から、使用されているオブジェクト間の関係を確認
        bonus = 0.0
        for (src_idx, tgt_idx), relations in relation_info.items():
            src_var = obj_idx_to_var.get(src_idx)
            tgt_var = obj_idx_to_var.get(tgt_idx)

            if src_var and tgt_var and src_var in used_vars and tgt_var in used_vars:
                # 関係に基づいてボーナスを追加
                if 'spatial_left' in relations or 'spatial_right' in relations:
                    # 空間的関係がある場合、移動操作の可能性が高い
                    bonus += 0.05
                if 'spatial_up' in relations or 'spatial_down' in relations:
                    # 上下関係がある場合、移動操作の可能性が高い
                    bonus += 0.05
                if 'mirror_x' in relations or 'mirror_y' in relations:
                    # 対称性がある場合、ミラー操作の可能性が高い
                    bonus += 0.1
                if 'repeat' in relations:
                    # 繰り返しパターンがある場合、ループ操作の可能性が高い
                    bonus += 0.08
                if 'contain' in relations:
                    # 包含関係がある場合、ネストされた操作の可能性が高い
                    bonus += 0.06

        return base_score + bonus

    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        beam_width: int = 5,
        partial_program: Optional[str] = None,
        matching_result: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """オブジェクトベースのニューラルモデル候補生成（改善版）

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            beam_width: ビーム幅（生成する候補数）
            partial_program: 部分プログラム（オプション）
            matching_result: オブジェクトマッチング結果（オプション）
                - objects_data: 抽出済みオブジェクト
                - categories: カテゴリ情報
                - category_var_mappings: カテゴリIDと変数名の対応関係

        Returns:
            生成されたプログラムのリスト
        """
        candidates = []

        try:
            if self.neural_object_model is None or self.tokenizer is None:
                return candidates

            # グリッドサイズを取得
            input_h, input_w = get_grid_size(input_grid)
            output_h, output_w = get_grid_size(output_grid)

            # 背景色を取得
            input_bg_color = get_background_color(input_grid)
            output_bg_color = get_background_color(output_grid)

            # オブジェクト抽出（matching_resultがある場合はスキップ）
            if matching_result and 'objects_data' in matching_result:
                # matching_resultからオブジェクトを取得
                # objects_dataの構造: {connectivity: {'input_grids': List[List[ObjectInfo]], 'output_grids': List[List[ObjectInfo]]}}
                objects_data = matching_result['objects_data']
                # 4連結のオブジェクトを取得（最初の入力グリッドのオブジェクトを使用）
                if 4 in objects_data and 'input_grids' in objects_data[4] and objects_data[4]['input_grids']:
                    # 最初の入力グリッドのオブジェクトを使用（推論時は通常1つの入力グリッドのみ）
                    input_objects_info_4 = objects_data[4]['input_grids'][0] if objects_data[4]['input_grids'] else []
                    output_objects_info_4 = objects_data[4]['output_grids'][0] if objects_data[4]['output_grids'] and objects_data[4]['output_grids'] else []

                    # ObjectInfoからObjectに変換
                    input_array = np.array(input_grid, dtype=int)
                    output_array = np.array(output_grid, dtype=int)

                    input_objects_4 = []
                    for idx, obj_info in enumerate(input_objects_info_4):
                        obj = self._convert_objectinfo_to_object(
                            obj_info,
                            input_array,
                            f"input_obj_{idx}"
                        )
                        input_objects_4.append(obj)

                    output_objects_4 = []
                    for idx, obj_info in enumerate(output_objects_info_4):
                        obj = self._convert_objectinfo_to_object(
                            obj_info,
                            output_array,
                            f"output_obj_{idx}"
                        )
                        output_objects_4.append(obj)
                else:
                    input_objects_4 = []
                    output_objects_4 = []
            else:
                # オブジェクトを抽出
                input_array = np.array(input_grid, dtype=int)
                output_array = np.array(output_grid, dtype=int)

                try:
                    input_result = self.object_extractor.extract_objects_by_type(input_array)
                    output_result = self.object_extractor.extract_objects_by_type(output_array)
                    input_objects_4 = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, []) if input_result.success else []
                    output_objects_4 = output_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, []) if output_result.success else []
                except Exception as e:
                    print(f"オブジェクト抽出エラー（4連結）: {e}")
                    input_objects_4 = []
                    output_objects_4 = []

            # Object Canonicalization（有効な場合）
            if self.enable_object_canonicalization and self.canonicalizer:
                input_canonicalized = self.canonicalizer.canonicalize(
                    input_objects_4, input_w, input_h
                )
                output_canonicalized = self.canonicalizer.canonicalize(
                    output_objects_4, output_w, output_h,
                    color_remap_map=self.canonicalizer.get_color_remap_map()
                )
                # 正規化されたオブジェクトを使用（元のオブジェクトの参照を保持）
                # 実際の処理では、正規化された特徴量をグラフ構築時に使用
            else:
                input_canonicalized = None
                output_canonicalized = None

            # Object Graph構築（有効な場合）
            # 部分プログラムに対応するカテゴリ情報を取得
            categories = None
            category_var_mapping = None  # 変数名とカテゴリIDの対応関係

            if matching_result:
                if partial_program:
                    # 部分プログラムに対応するカテゴリ情報を取得
                    if 'partial_program_to_categories' in matching_result:
                        partial_program_to_categories = matching_result['partial_program_to_categories']
                        if partial_program in partial_program_to_categories:
                            # 対応するカテゴリ情報を取得
                            # 注意: 同じ部分プログラムが複数のパターンで生成された場合でも、
                            #       最初のパターンのカテゴリ情報のみが保存されている（カテゴリ分けとマッピングは一致しているため）
                            categories_list = partial_program_to_categories[partial_program]
                            if categories_list:
                                # カテゴリ情報をDict[int, List[int]]形式に変換
                                # CategoryInfoのリストから、オブジェクトインデックスのリストに変換
                                categories = self._convert_categories_to_dict(categories_list, input_objects_4)

                    # 部分プログラムに対応するcategory_var_mappingを直接取得
                    # 部分プログラム作成時に保存されたマッピングを使用（確実性が高い）
                    if 'partial_program_to_category_var_mapping' in matching_result:
                        partial_program_to_category_var_mapping = matching_result['partial_program_to_category_var_mapping']
                        if partial_program in partial_program_to_category_var_mapping:
                            category_var_mapping = partial_program_to_category_var_mapping[partial_program]
                            # デバッグ用（必要に応じて有効化）
                            # if category_var_mapping:
                            #     print(f"[DEBUG] 部分プログラムの変数-カテゴリ対応: {category_var_mapping}")
                    # category_var_mappingsからのフォールバックは削除
                    # partial_program_to_category_var_mappingが常に利用可能なため不要
                # 部分プログラムがない場合、または対応関係がない場合は従来通り
                # 注意: この場合は全パターンのカテゴリ情報が含まれるため、使用しない方が良い
                # elif 'categories' in matching_result:
                #     categories = matching_result['categories']  # コメントアウト（推奨）

            # グラフ関連の変数を初期化
            input_graph = None
            output_graph = None
            input_graph_encoded = None
            output_graph_encoded = None

            if self.enable_object_graph and self.graph_builder:
                # 入力グラフを構築
                input_graph = self.graph_builder.build_graph(
                    input_objects_4,
                    categories=categories,  # 部分プログラムに対応するカテゴリ情報を使用
                    grid_width=input_w,
                    grid_height=input_h
                )

                # 出力グラフを構築
                output_graph = self.graph_builder.build_graph(
                    output_objects_4,
                    categories=categories,  # 部分プログラムに対応するカテゴリ情報を使用
                    grid_width=output_w,
                    grid_height=output_h
                )

                # GNNエンコーディング（有効な場合）
                if self.graph_encoder:
                    input_graph_encoded, _ = self.graph_encoder(input_graph)
                    output_graph_encoded, _ = self.graph_encoder(output_graph)
                    # グラフエンコードされた特徴量は、ObjectEncoderの入力として使用

            # Relation Classifier（有効な場合）
            relation_info = None
            if self.enable_relation_classifier and self.relation_classifier and input_graph:
                relation_info = self.relation_classifier.classify_relations(
                    input_graph,
                    threshold=self.relation_classifier_threshold
                )
                # 関係情報は、プログラム生成のガイダンスとして使用（後処理でスコア調整）

            # 色役割情報を取得（matching_resultから）
            input_color_roles = None
            output_color_roles = None
            if matching_result and 'background_colors' in matching_result:
                background_colors = matching_result['background_colors']
                # 最初の入力グリッド（grid_index=0）の色役割情報を使用
                if 0 in background_colors and hasattr(background_colors[0], 'color_roles'):
                    input_color_roles = background_colors[0].color_roles
                # 出力グリッドの色役割情報は、入力グリッドと同じと仮定（または別途取得）
                # 現在は入力グリッドと同じ色役割情報を使用
                output_color_roles = input_color_roles

            # category_var_mappingをプログラム生成モデルに渡す
            # 部分プログラムの変数名とカテゴリIDの対応関係を提供
            # 例: 部分プログラムの変数名 "objects1" がカテゴリID "4_0_0" に対応している場合、
            #     プログラム生成時にこの対応関係を考慮してより適切なプログラムを生成できる
            # 注意: 現在の実装では、category_var_mappingはプログラム生成モデルに渡しているが、
            #       モデル側での直接的な活用は将来的な拡張として残されている
            beam_results = self.neural_object_model.beam_search(
                input_objects=input_objects_4,
                output_objects=output_objects_4,
                input_background_color=input_bg_color,
                output_background_color=output_bg_color,
                input_grid_width=input_w,
                input_grid_height=input_h,
                output_grid_width=output_w,
                output_grid_height=output_h,
                beam_width=beam_width,
                partial_program=partial_program,
                tokenizer=self.tokenizer,
                category_var_mapping=category_var_mapping,  # カテゴリIDと変数名の対応関係を渡す
                input_graph_encoded=input_graph_encoded,  # グラフ特徴量を渡す
                output_graph_encoded=output_graph_encoded,  # グラフ特徴量を渡す
                input_canonicalized=input_canonicalized,  # 正規化されたオブジェクトを渡す
                output_canonicalized=output_canonicalized,  # 正規化されたオブジェクトを渡す
                input_color_roles=input_color_roles,  # 入力グリッドの色役割情報を渡す
                output_color_roles=output_color_roles  # 出力グリッドの色役割情報を渡す
            )

            # トークンをプログラム文字列に変換し、関係情報に基づいてスコアを調整
            candidate_with_scores = []  # (program, score)のリスト

            for tokens, score in beam_results[:beam_width]:  # 上位beam_width個
                # BOS/EOS を除去
                token_ids = tokens[0].cpu().numpy().tolist()
                # BOS (1) と EOS (2) を除去
                token_ids = [tid for tid in token_ids if tid not in [1, 2, 0]]

                if token_ids:
                    template_string = self.tokenizer.decode(token_ids)
                    template_string = template_string.strip()
                    if not template_string:
                        continue
                    try:
                        sequence = template_string_to_sequence(template_string, task_id="inference")
                        program = sequence_to_dsl(sequence)
                    except Exception:
                        continue
                    if program:
                        # 関係情報に基づいてスコアを調整
                        adjusted_score = self._adjust_score_with_relations(
                            program, relation_info, input_objects_4, category_var_mapping, score,
                            matching_result=matching_result,  # オブジェクトとカテゴリの対応関係構築用
                            partial_program=partial_program  # カテゴリ情報取得用
                        )
                        candidate_with_scores.append((program, adjusted_score))

            # スコアでソート（高い順）
            candidate_with_scores.sort(key=lambda x: x[1], reverse=True)

            # プログラムのみを抽出
            candidates = [program for program, _ in candidate_with_scores]

            return candidates

        except Exception as e:
            print(f"オブジェクトベースニューラル候補生成エラー: {e}")
            import traceback
            traceback.print_exc()
            return candidates
