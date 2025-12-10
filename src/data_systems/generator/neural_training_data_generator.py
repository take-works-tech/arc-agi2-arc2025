"""
ニューラルモデル用学習データ生成器

generatorで生成したデータから、各ニューラルモデル用の学習データを生成
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import re
from collections import Counter

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
from src.hybrid_system.models.program_synthesis.symmetry_augmentation import (
    SymmetryAugmenter,
    SymmetryAwareDataLoader
)
from src.data_systems.data_models.augmentation.color_permutation_augmentation import (
    ColorPermutationAugmenter
)


def extract_dsl_commands(program: str) -> List[str]:
    """プログラムからDSLコマンドを抽出"""
    if not program:
        return []
    # 大文字の識別子（関数名）を抽出
    commands = re.findall(r'\b([A-Z][A-Z0-9_]+)\s*\(', program)
    return commands


def calculate_dsl_probabilities(programs: List[str]) -> Dict[str, float]:
    """プログラムリストからDSL使用確率を計算"""
    if not programs:
        return {}

    all_commands = []
    for program in programs:
        commands = extract_dsl_commands(program)
        all_commands.extend(commands)

    if not all_commands:
        return {}

    # コマンドの出現頻度をカウント
    command_counts = Counter(all_commands)
    total = len(all_commands)

    # 確率に変換
    probabilities = {cmd: count / total for cmd, count in command_counts.items()}

    return probabilities


def extract_grid_features(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
    """グリッドから特徴量を抽出"""
    return {
        'input_shape': list(input_grid.shape),
        'output_shape': list(output_grid.shape),
        'input_size': int(input_grid.size),
        'output_size': int(output_grid.size),
        'input_unique_colors': int(len(np.unique(input_grid))),
        'output_unique_colors': int(len(np.unique(output_grid))),
        'input_mean': float(np.mean(input_grid)),
        'output_mean': float(np.mean(output_grid)),
    }


def extract_object_graph_features(graph) -> Dict[str, Any]:
    """オブジェクトグラフから特徴量を抽出"""
    node_features = graph.node_features.cpu().numpy().tolist()
    edge_index = graph.edge_index.cpu().numpy().tolist()
    edge_attr = graph.edge_attr.cpu().numpy().tolist()

    return {
        'num_nodes': len(graph.nodes),
        'num_edges': graph.edge_index.shape[1],
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_types': [edge.edge_type for edge in graph.edges]
    }


def classify_relation(obj1: Object, obj2: Object, grid_width: int = None, grid_height: int = None) -> List[str]:
    """
    2つのオブジェクト間の関係を分類（本格実装）

    Args:
        obj1: 第1オブジェクト
        obj2: 第2オブジェクト
        grid_width: グリッド幅（対称性検出に使用、オプション）
        grid_height: グリッド高さ（対称性検出に使用、オプション）

    Returns:
        関係タイプのリスト
    """
    relations = []

    # 位置関係
    center1_x, center1_y = obj1.center_x, obj1.center_y
    center2_x, center2_y = obj2.center_x, obj2.center_y

    # 左右関係
    if center1_x < center2_x:
        relations.append('spatial_left')
    elif center1_x > center2_x:
        relations.append('spatial_right')

    # 上下関係
    if center1_y < center2_y:
        relations.append('spatial_up')
    elif center1_y > center2_y:
        relations.append('spatial_down')

    # 対称性の検出（本格実装）
    # グリッドサイズを考慮した対称軸の計算
    if grid_width is not None and grid_height is not None:
        # X軸対称（水平軸を中心とした対称）
        # 対称軸は grid_height / 2
        axis_y = grid_height / 2.0
        # obj1とobj2が対称軸に対して対称な位置にあるか
        dist1_to_axis = abs(center1_y - axis_y)
        dist2_to_axis = abs(center2_y - axis_y)
        # Y座標が対称軸に対して対称で、X座標が近い場合
        if abs(dist1_to_axis - dist2_to_axis) < 1.0 and abs(center1_x - center2_x) < 3.0:
            relations.append('mirror_x')

        # Y軸対称（垂直軸を中心とした対称）
        # 対称軸は grid_width / 2
        axis_x = grid_width / 2.0
        # obj1とobj2が対称軸に対して対称な位置にあるか
        dist1_to_axis = abs(center1_x - axis_x)
        dist2_to_axis = abs(center2_x - axis_x)
        # X座標が対称軸に対して対称で、Y座標が近い場合
        if abs(dist1_to_axis - dist2_to_axis) < 1.0 and abs(center1_y - center2_y) < 3.0:
            relations.append('mirror_y')
    else:
        # グリッドサイズが不明な場合のフォールバック（簡易版）
        # X軸対称
        if abs(center1_y - center2_y) < 2 and abs(center1_x + center2_x) < 10:
            relations.append('mirror_x')
        # Y軸対称
        if abs(center1_x - center2_x) < 2 and abs(center1_y + center2_y) < 10:
            relations.append('mirror_y')

    # 包含関係の検出（本格実装）
    bbox1 = obj1.bbox
    bbox2 = obj2.bbox

    # bbox形式: (x1, y1, x2, y2) または (min_i, min_j, max_i, max_j)
    # 座標系を統一（x1, y1, x2, y2形式に変換）
    if len(bbox1) == 4 and len(bbox2) == 4:
        # 既に(x1, y1, x2, y2)形式と仮定
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # obj1がobj2を包含しているか
        if (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2):
            relations.append('contain')
        # obj2がobj1を包含しているか（双方向チェック）
        elif (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
            # 包含関係は双方向なので、obj1がobj2に包含される場合も記録
            # ただし、関係はobj1->obj2の方向で記録するため、ここでは追加しない
            pass

    # 繰り返しパターンの検出（本格実装）
    # サイズ、色、形状の類似度を総合的に評価
    size_similarity = min(obj1.bbox_width, obj2.bbox_width) / max(obj1.bbox_width, obj2.bbox_width) if max(obj1.bbox_width, obj2.bbox_width) > 0 else 0.0
    height_similarity = min(obj1.bbox_height, obj2.bbox_height) / max(obj1.bbox_height, obj2.bbox_height) if max(obj1.bbox_height, obj2.bbox_height) > 0 else 0.0

    # 面積の類似度
    area1 = obj1.bbox_width * obj1.bbox_height if hasattr(obj1, 'bbox_width') else len(obj1.pixels) if obj1.pixels else 0
    area2 = obj2.bbox_width * obj2.bbox_height if hasattr(obj2, 'bbox_width') else len(obj2.pixels) if obj2.pixels else 0
    area_similarity = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0

    # 色の一致
    color_match = (obj1.dominant_color == obj2.dominant_color) if hasattr(obj1, 'dominant_color') and hasattr(obj2, 'dominant_color') else False

    # 形状の類似度（アスペクト比）
    aspect1 = obj1.bbox_width / obj1.bbox_height if obj1.bbox_height > 0 else 1.0
    aspect2 = obj2.bbox_width / obj2.bbox_height if obj2.bbox_height > 0 else 1.0
    aspect_similarity = min(aspect1, aspect2) / max(aspect1, aspect2) if max(aspect1, aspect2) > 0 else 0.0

    # 総合的な類似度（すべての要素が高い類似度を持つ場合、繰り返しパターンと判定）
    avg_similarity = (size_similarity + height_similarity + area_similarity + aspect_similarity) / 4.0

    # 閾値: サイズ・形状の類似度が0.8以上、かつ色が一致する場合
    if avg_similarity >= 0.8 and color_match:
        relations.append('repeat')

    return relations


def extract_object_features(obj: Object) -> List[float]:
    """オブジェクトから特徴量を抽出"""
    area = len(obj.pixels) if obj.pixels else 0
    bbox_area = obj.bbox_area if hasattr(obj, 'bbox_area') else (obj.bbox_width * obj.bbox_height)

    return [
        float(obj.center_x),
        float(obj.center_y),
        float(obj.bbox_width),
        float(obj.bbox_height),
        float(obj.dominant_color),
        float(area),
        float(bbox_area),
        float(obj.hole_count) if hasattr(obj, 'hole_count') else 0.0,
    ]


class NeuralTrainingDataGenerator:
    """ニューラルモデル用学習データ生成器"""

    def __init__(
        self,
        output_dir: str,
        enable_symmetry_augmentation: bool = False,
        augmentation_prob: float = 0.5,
        enable_color_permutation_augmentation: bool = False,
        color_permutation_prob: float = 0.5
    ):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            enable_symmetry_augmentation: 対称性拡張を有効にするか
            augmentation_prob: 拡張を適用する確率
            enable_color_permutation_augmentation: 色順列拡張を有効にするか
            color_permutation_prob: 色順列拡張を適用する確率
        """
        self.output_dir = output_dir
        self.ngps_data: List[Dict[str, Any]] = []
        self.object_graph_data: List[Dict[str, Any]] = []
        self.relation_classifier_data: List[Dict[str, Any]] = []

        # オブジェクト抽出器とグラフビルダーを初期化
        # 注意: オブジェクトマッチングは使用しない（推論時の補助情報のため、学習データ生成時には不要）
        self.extractor = IntegratedObjectExtractor(ExtractionConfig())
        self.graph_builder = ObjectGraphBuilder()

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

        # Symmetry-Aware Augmentation
        # 注意: プログラムが提供されている場合、拡張を適用すると正解データが壊れる可能性があります。
        # プログラムが回転・反転に対して不変でない場合、プログラムも変換する必要があります。
        # デフォルトでは、プログラムが提供されている場合は拡張を適用しません。
        self.enable_symmetry_augmentation = enable_symmetry_augmentation
        if self.enable_symmetry_augmentation:
            self.symmetry_loader = SymmetryAwareDataLoader(
                augmentation_prob=augmentation_prob
            )
        else:
            self.symmetry_loader = None

        # Color Permutation Augmentation
        # 注意: プログラムが提供されている場合、拡張を適用すると正解データが壊れる可能性があります。
        # 色順列拡張は、プログラムが色に対して不変でない場合、プログラムも変換する必要があります。
        # デフォルトでは、プログラムが提供されている場合は拡張を適用しません。
        self.enable_color_permutation_augmentation = enable_color_permutation_augmentation
        if self.enable_color_permutation_augmentation:
            self.color_permutation_augmenter = ColorPermutationAugmenter(
                enable_permutation=True,
                permutation_prob=color_permutation_prob,
                preserve_background=True
            )
        else:
            self.color_permutation_augmenter = None

        # タスクごとのプログラムリスト（NGPS用）
        self.task_programs: Dict[str, List[str]] = {}

    def generate_from_generator_output(
        self,
        task_id: str,
        program_code: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        nodes: List[Any],
        complexity: int,
        pair_index: int = 0
    ):
        """
        generatorの出力から学習データを生成

        Args:
            task_id: タスクID
            program_code: プログラムコード
            input_grid: 入力グリッド（numpy配列）
            output_grid: 出力グリッド（numpy配列）
            nodes: プログラムのNodeリスト
            complexity: 複雑度
            pair_index: ペアインデックス
        """
        # タスクごとのプログラムリストに追加（NGPS用）
        if task_id not in self.task_programs:
            self.task_programs[task_id] = []
        self.task_programs[task_id].append(program_code)

        # Symmetry-Aware Augmentationを適用（有効な場合）
        # 注意: プログラムが提供されている場合、拡張を適用すると正解データが壊れる可能性があります。
        # プログラムが回転・反転に対して不変でない場合、プログラムも変換する必要があります。
        # デフォルトでは、プログラムが提供されている場合は拡張を適用しません。
        if self.enable_symmetry_augmentation and self.symmetry_loader:
            # プログラムコードを渡して、プログラムが提供されている場合は拡張を適用しない
            input_grid, output_grid = self.symmetry_loader.augment_sample(
                input_grid, output_grid,
                program_code=program_code,  # プログラムが提供されている場合は拡張を適用しない
                verify_correctness=True  # 正解性を検証（デフォルト: True）
            )

        # Color Permutation Augmentationを適用（有効な場合）
        # 注意: プログラムが提供されている場合、拡張を適用すると正解データが壊れる可能性があります。
        # 色順列拡張は、プログラムが色に対して不変でない場合、プログラムも変換する必要があります。
        # デフォルトでは、プログラムが提供されている場合は拡張を適用しません。
        if self.enable_color_permutation_augmentation and self.color_permutation_augmenter:
            # 背景色を推論（簡易版: 最頻出色を使用）
            input_bg_color = None
            output_bg_color = None
            if input_grid.size > 0:
                input_colors, input_counts = np.unique(input_grid, return_counts=True)
                input_bg_color = int(input_colors[np.argmax(input_counts)])
            if output_grid.size > 0:
                output_colors, output_counts = np.unique(output_grid, return_counts=True)
                output_bg_color = int(output_colors[np.argmax(output_counts)])

            # プログラムコードを渡して、プログラムが提供されている場合は拡張を適用しない
            input_grid, output_grid, _ = self.color_permutation_augmenter.augment_pair(
                input_grid, output_grid,
                input_background_color=input_bg_color,
                output_background_color=output_bg_color,
                consistent_permutation=True,
                program_code=program_code,  # プログラムが提供されている場合は拡張を適用しない
                verify_correctness=True  # 正解性を検証（デフォルト: True）
            )

        # 1. NGPS/DSL Selector用データ生成
        self._generate_ngps_data(
            task_id=task_id,
            pair_index=pair_index,
            program_code=program_code,
            input_grid=input_grid,
            output_grid=output_grid
        )

        # 2. Object Graph + GNN用データ生成
        self._generate_object_graph_data(
            task_id=task_id,
            pair_index=pair_index,
            program_code=program_code,
            input_grid=input_grid,
            output_grid=output_grid
        )

        # 3. Relation Classifier用データ生成
        self._generate_relation_classifier_data(
            task_id=task_id,
            pair_index=pair_index,
            input_grid=input_grid,
            output_grid=output_grid
        )

    def _generate_ngps_data(
        self,
        task_id: str,
        pair_index: int,
        program_code: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ):
        """NGPS/DSL Selector用データを生成"""
        try:
            # タスク全体のDSL使用確率を計算
            task_programs = self.task_programs.get(task_id, [])
            dsl_probabilities = calculate_dsl_probabilities(task_programs)

            # DSL確率が空の場合はデフォルト確率を使用
            if not dsl_probabilities:
                dsl_probabilities = {f'dsl_{i}': 1.0 / 100 for i in range(100)}

            # グリッド特徴量を抽出
            grid_features = extract_grid_features(input_grid, output_grid)

            # サンプルを作成
            sample = {
                'task_id': task_id,
                'pair_index': pair_index,
                'grid_features': grid_features,
                'dsl_probabilities': dsl_probabilities,
                'input_grid': input_grid.tolist(),
                'output_grid': output_grid.tolist()
            }

            self.ngps_data.append(sample)
        except Exception as e:
            # エラーが発生しても処理を継続
            pass

    def _generate_object_graph_data(
        self,
        task_id: str,
        pair_index: int,
        program_code: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ):
        """
        Object Graph + GNN用データを生成

        注意:
        - program_codeは正解プログラム（generatorが生成した完全なプログラム）
        - 部分プログラムは推論時の補助情報であり、学習データの教師データではない
        - オブジェクトマッチング結果（カテゴリ情報）は推論時の補助情報であり、学習データ生成時には不要
          - 理由1: 教師データではない（正解プログラムのみが教師データ）
          - 理由2: 基本構造（隣接、包含、空間関係など）はカテゴリ情報なしでも同じ
          - 理由3: 効率が重要（オブジェクトマッチングは時間がかかる）
          - 理由4: モデルは適応できる（推論時と学習時でグラフ構造が異なっても問題なし）
        """
        try:
            # オブジェクト抽出
            input_result = self.extractor.extract_objects_by_type(input_grid, input_image_index=0)
            output_result = self.extractor.extract_objects_by_type(output_grid, input_image_index=0)

            if not input_result.success:
                return

            input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
            if not input_objects:
                return

            output_objects = output_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

            # オブジェクトグラフを構築
            # 注意: カテゴリ情報は推論時の補助情報であり、学習データ生成時には不要
            # - 教師データではない（正解プログラムのみが教師データ）
            # - 基本構造（隣接、包含、空間関係など）はカテゴリ情報なしでも同じ
            # - 効率が重要（オブジェクトマッチングは時間がかかる）
            # - モデルは適応できる（推論時と学習時でグラフ構造が異なっても問題なし）
            graph = self.graph_builder.build_graph(
                input_objects,
                categories=None,  # 学習データ生成時はカテゴリ情報なし
                grid_width=input_grid.shape[1],
                grid_height=input_grid.shape[0]
            )

            if graph.node_features.size(0) == 0:
                return

            # グラフ特徴量を抽出
            graph_features = extract_object_graph_features(graph)

            # サンプルを作成
            # programフィールドには正解プログラム（program_code）を保存
            # 部分プログラムは推論時の補助情報であり、学習データの教師データではない
            sample = {
                'task_id': task_id,
                'pair_index': pair_index,
                'graph_features': graph_features,
                'program': program_code,  # 正解プログラムを使用（部分プログラムではない）
                'input_grid_shape': list(input_grid.shape),
                'output_grid_shape': list(output_grid.shape),
                'num_input_objects': len(input_objects),
                'num_output_objects': len(output_objects)
            }

            self.object_graph_data.append(sample)
        except Exception as e:
            # エラーが発生しても処理を継続
            pass

    def _generate_relation_classifier_data(
        self,
        task_id: str,
        pair_index: int,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ):
        """Relation Classifier用データを生成"""
        try:
            # オブジェクト抽出
            input_result = self.extractor.extract_objects_by_type(input_grid, input_image_index=0)

            if not input_result.success:
                return

            input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
            if len(input_objects) < 2:
                return

            # オブジェクトグラフを構築
            # 注意: カテゴリ情報は使用しない（推論時の補助情報のため、学習データ生成時には不要）
            graph = self.graph_builder.build_graph(
                input_objects,
                categories=None,  # カテゴリ情報なし
                grid_width=input_grid.shape[1],
                grid_height=input_grid.shape[0]
            )

            # 各エッジ（オブジェクトペア）に対してサンプルを生成
            for edge in graph.edges:
                obj1_idx = edge.source_idx
                obj2_idx = edge.target_idx

                if obj1_idx >= len(input_objects) or obj2_idx >= len(input_objects):
                    continue

                obj1 = input_objects[obj1_idx]
                obj2 = input_objects[obj2_idx]

                # 関係を分類（グリッドサイズを渡してより正確な対称性検出を実現）
                relations = classify_relation(
                    obj1, obj2,
                    grid_width=input_grid.shape[1],
                    grid_height=input_grid.shape[0]
                )

                # 関係ラベルをベクトル化
                relation_labels = [1.0 if rel_type in relations else 0.0 for rel_type in self.relation_types]

                # オブジェクト特徴量を抽出
                obj1_features = extract_object_features(obj1)
                obj2_features = extract_object_features(obj2)

                # 相対特徴量
                relative_features = [
                    float(obj2.center_x - obj1.center_x),
                    float(obj2.center_y - obj1.center_y),
                    float(obj2.bbox_width - obj1.bbox_width),
                    float(obj2.bbox_height - obj1.bbox_height),
                ]

                # サンプルを作成
                sample = {
                    'task_id': task_id,
                    'pair_index': pair_index,
                    'obj1_features': obj1_features,
                    'obj2_features': obj2_features,
                    'relative_features': relative_features,
                    'relation_labels': relation_labels,
                    'relation_types': self.relation_types,
                    'edge_type': edge.edge_type
                }

                self.relation_classifier_data.append(sample)
        except Exception as e:
            # エラーが発生しても処理を継続
            pass

    def flush_batch(self, batch_index: int):
        """
        バッチごとにJSONLファイルに保存

        Args:
            batch_index: バッチインデックス
        """
        batch_dir = os.path.join(self.output_dir, f"batch_{batch_index:04d}")
        neural_data_dir = os.path.join(batch_dir, "neural_training_data")
        os.makedirs(neural_data_dir, exist_ok=True)

        # NGPS/DSL Selector用データを保存
        if self.ngps_data:
            ngps_path = os.path.join(neural_data_dir, "ngps_train_data.jsonl")
            with open(ngps_path, 'w', encoding='utf-8') as f:
                for sample in self.ngps_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.ngps_data = []

        # Object Graph + GNN用データを保存
        if self.object_graph_data:
            object_graph_path = os.path.join(neural_data_dir, "object_graph_train_data.jsonl")
            with open(object_graph_path, 'w', encoding='utf-8') as f:
                for sample in self.object_graph_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.object_graph_data = []

        # Relation Classifier用データを保存
        if self.relation_classifier_data:
            relation_path = os.path.join(neural_data_dir, "relation_classifier_train_data.jsonl")
            with open(relation_path, 'w', encoding='utf-8') as f:
                for sample in self.relation_classifier_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.relation_classifier_data = []

    def save_all(self):
        """すべての学習データを保存（最終保存用）"""
        # 最後のバッチとして保存
        # バッチインデックスは0を仮定（実際のバッチインデックスは呼び出し側で管理）
        self.flush_batch(0)

        # タスクごとのプログラムリストをクリア
        self.task_programs = {}
