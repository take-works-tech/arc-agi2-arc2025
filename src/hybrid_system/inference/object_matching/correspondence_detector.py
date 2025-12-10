"""
対応関係検出モジュール

1対1対応、分割、統合の検出
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .data_structures import ObjectInfo
from .similarity_calculator import SimilarityCalculator
from .similarity_utils import get_pixels_set


class CorrespondenceDetector:
    """対応関係検出器"""

    def __init__(self, config: Any, similarity_calculator: SimilarityCalculator):
        self.config = config
        self.similarity_calculator = similarity_calculator

    def find_correspondences(
        self, input_objects: List[ObjectInfo], output_objects: List[ObjectInfo],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        1対1対応を検出（回転・反転を考慮した2段階探索）

        Args:
            input_objects: 入力オブジェクトのリスト
            output_objects: 出力オブジェクトのリスト
            grid_size: グリッドサイズ (width, height)

        Returns:
            対応関係のリスト
        """
        correspondences = []
        used_output_indices = set()
        used_input_indices = set()  # 入力オブジェクトのインデックスも追跡

        # 第0段階: 変化なしオブジェクト（完全一致）を先に検出
        remaining_input_objects = []
        for i, input_obj in enumerate(input_objects):
            # 既に使用されている入力オブジェクトはスキップ
            if i in used_input_indices:
                continue

            found_exact_match = False
            for j, output_obj in enumerate(output_objects):
                if j in used_output_indices:
                    continue

                # 完全一致チェック: 色、位置、ピクセル形状すべてが一致
                if self._is_exact_match(input_obj, output_obj):
                    # 1対1対応を確立（変換パターンは後で計算）
                    correspondences.append({
                        'input_obj': input_obj,
                        'output_obj': output_obj
                    })
                    used_output_indices.add(j)
                    used_input_indices.add(i)
                    found_exact_match = True
                    break

            # 完全一致が見つからなかった入力オブジェクトは残りの処理で検出
            if not found_exact_match:
                remaining_input_objects.append((i, input_obj))

        # 第2段階: ベクトル距離ベースで1対1対応を検出（残りの入力オブジェクトのみ）
        # 特徴量ベクトル（base_feature_vector）のユークリッド距離を使用
        # 最大ベクトル距離A以下で最も近い入出力オブジェクトから順に1対1対応

        # すべての入出力オブジェクトペアのベクトル距離を計算
        distance_candidates = []
        for input_idx, input_obj in remaining_input_objects:
            # 既に使用されている入力オブジェクトはスキップ
            if input_idx in used_input_indices:
                continue

            # base_feature_vectorが存在するか確認
            if input_obj.base_feature_vector is None:
                continue

            for j, output_obj in enumerate(output_objects):
                if j in used_output_indices:
                    continue

                # base_feature_vectorが存在するか確認
                if output_obj.base_feature_vector is None:
                    continue

                # ベクトル距離を計算（重み付きユークリッド距離）
                vector_distance = self._calculate_weighted_vector_distance(
                    input_obj.base_feature_vector,
                    output_obj.base_feature_vector
                )

                # 最大ベクトル距離A以下かチェック
                if vector_distance <= self.config.max_vector_distance:
                    distance_candidates.append({
                        'input_obj': input_obj,
                        'input_idx': input_idx,
                        'output_obj': output_obj,
                        'output_idx': j,
                        'vector_distance': vector_distance
                    })

        # ベクトル距離が小さい順にソート（最も近いものから順に）
        distance_candidates.sort(key=lambda x: x['vector_distance'])

        # 最も近い入出力オブジェクトから順に1対1対応を確立
        for candidate in distance_candidates:
            input_idx = candidate['input_idx']
            output_idx = candidate['output_idx']

            # 既に使用されている入出力オブジェクトはスキップ
            if input_idx in used_input_indices or output_idx in used_output_indices:
                continue

            # 1対1対応を確立（変換パターンは後で計算）
            correspondences.append({
                'input_obj': candidate['input_obj'],
                'output_obj': candidate['output_obj']
            })
            used_input_indices.add(input_idx)
            used_output_indices.add(output_idx)

        return correspondences

    def _is_exact_match(self, input_obj: ObjectInfo, output_obj: ObjectInfo) -> bool:
        """
        完全一致チェック: 色、位置、ピクセル形状すべてが一致するか

        Args:
            input_obj: 入力オブジェクト
            output_obj: 出力オブジェクト

        Returns:
            完全一致する場合True
        """
        # 色の一致チェック
        if input_obj.color != output_obj.color:
            return False

        # ピクセル形状の完全一致チェック（位置も含む）
        input_pixels_set = get_pixels_set(input_obj) if input_obj.pixels else set()
        output_pixels_set = get_pixels_set(output_obj) if output_obj.pixels else set()

        # ピクセル集合が完全に一致するか
        if input_pixels_set != output_pixels_set:
            return False

        # すべての条件を満たす場合、完全一致
        return True

    def detect_splits(
        self, input_objects: List[ObjectInfo], output_objects: List[ObjectInfo],
        correspondences: List[Dict[str, Any]], grid_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        分割パターンを検出（特徴量ベクトル距離ベース）

        第3段階: 分割の検出（1対1対応していない残りの出力オブジェクトのみ）
        最大ベクトル距離Cを設定。
        残りの出力オブジェクトとすべての入力オブジェクト（1対1対応済みも含む）の
        特徴量ベクトルの距離がC以下でかつ最も近い場合、多対1対応にして分割設定。
        """
        splits = []

        # 1対1対応で使用されていない出力オブジェクトを特定
        matched_output_indices = {corr['output_obj'].index for corr in correspondences}

        remaining_output_objects = [
            obj for i, obj in enumerate(output_objects)
            if obj.index not in matched_output_indices
        ]

        if not remaining_output_objects or not input_objects:
            return splits

        # 各残りの出力オブジェクトに対して、分割元を探索
        for output_obj in remaining_output_objects:
            # base_feature_vectorが存在するか確認
            if output_obj.base_feature_vector is None:
                continue

            # すべての入力オブジェクト（1対1対応済みも含む）とのベクトル距離を計算
            distance_candidates = []
            for input_obj in input_objects:
                # base_feature_vectorが存在するか確認
                if input_obj.base_feature_vector is None:
                    continue

                # ベクトル距離を計算（ユークリッド距離）
                vector_distance = np.linalg.norm(
                    output_obj.base_feature_vector - input_obj.base_feature_vector
                )

                # 最大ベクトル距離C以下かチェック
                if vector_distance <= self.config.max_vector_distance_split:
                    distance_candidates.append({
                        'input_obj': input_obj,
                        'vector_distance': vector_distance
                    })

            if not distance_candidates:
                continue

            # ベクトル距離が小さい順にソート（最も近いものから順に）
            distance_candidates.sort(key=lambda x: x['vector_distance'])

            # 最も近い入力オブジェクトを分割元として選択
            best_candidate = distance_candidates[0]
            input_obj = best_candidate['input_obj']

            # 既存の分割にこの入力オブジェクトが含まれているか確認
            existing_split = None
            for split in splits:
                if split['input_obj'].index == input_obj.index:
                    existing_split = split
                    break

            if existing_split:
                # 既存の分割に追加
                existing_split['output_objects'].append(output_obj)
                existing_split['split_count'] = len(existing_split['output_objects'])
                # 変換パターンは後で計算（対応関係決定後）
                existing_split['transformation_patterns'].append({
                    'output_obj': output_obj
                })
            else:
                # 新しい分割を作成
                splits.append({
                    'input_obj': input_obj,
                    'output_objects': [output_obj],
                    'correspondence_type': 'one_to_many',
                    'transformation_patterns': [{'output_obj': output_obj}],
                    'split_count': 1,
                    'confidence': 1.0 - best_candidate['vector_distance'] / self.config.max_vector_distance_split  # 距離が近いほど信頼度が高い
                })

        # 分割先が2つ以上ある分割のみを返す
        return [split for split in splits if split['split_count'] >= 2]

    def detect_merges(
        self, input_objects: List[ObjectInfo], output_objects: List[ObjectInfo],
        correspondences: List[Dict[str, Any]], grid_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        統合パターンを検出（特徴量ベクトル距離ベース）

        第3段階: 統合の検出（残りの入力オブジェクトのみ）
        最大ベクトル距離Bを設定。
        残りの入力オブジェクトとすべての出力オブジェクト（1対1対応済みも含む）の
        特徴量ベクトルの距離がB以下でかつ最も近い場合、1対多対応にして統合設定。
        """
        merges = []

        # 1対1対応で使用されていない入力オブジェクトを特定
        matched_input_indices = {corr['input_obj'].index for corr in correspondences}

        remaining_input_objects = [
            obj for i, obj in enumerate(input_objects)
            if obj.index not in matched_input_indices
        ]

        if not remaining_input_objects or not output_objects:
            return merges

        # 各残りの入力オブジェクトに対して、統合先を探索
        for input_obj in remaining_input_objects:
            # base_feature_vectorが存在するか確認
            if input_obj.base_feature_vector is None:
                continue

            # すべての出力オブジェクト（1対1対応済みも含む）とのベクトル距離を計算
            distance_candidates = []
            for output_obj in output_objects:
                # base_feature_vectorが存在するか確認
                if output_obj.base_feature_vector is None:
                    continue

                # ベクトル距離を計算（重み付きユークリッド距離）
                vector_distance = self._calculate_weighted_vector_distance(
                    input_obj.base_feature_vector,
                    output_obj.base_feature_vector
                )

                # 最大ベクトル距離B以下かチェック
                if vector_distance <= self.config.max_vector_distance_merge:
                    distance_candidates.append({
                        'output_obj': output_obj,
                        'vector_distance': vector_distance
                    })

            if not distance_candidates:
                continue

            # ベクトル距離が小さい順にソート（最も近いものから順に）
            distance_candidates.sort(key=lambda x: x['vector_distance'])

            # 最も近い出力オブジェクトを統合先として選択
            best_candidate = distance_candidates[0]
            output_obj = best_candidate['output_obj']

            # 既存の統合にこの出力オブジェクトが含まれているか確認
            existing_merge = None
            for merge in merges:
                if merge['output_obj'].index == output_obj.index:
                    existing_merge = merge
                    break

            if existing_merge:
                # 既存の統合に追加
                existing_merge['input_objects'].append(input_obj)
                existing_merge['merge_count'] = len(existing_merge['input_objects'])
                # 変換パターンは後で計算（対応関係決定後）
                existing_merge['transformation_patterns'].append({
                    'input_obj': input_obj
                })
            else:
                # 新しい統合を作成
                merges.append({
                    'input_objects': [input_obj],
                    'output_obj': output_obj,
                    'correspondence_type': 'many_to_one',
                    'transformation_patterns': [{'input_obj': input_obj}],
                    'merge_count': 1,
                    'confidence': 1.0 - best_candidate['vector_distance'] / self.config.max_vector_distance_merge  # 距離が近いほど信頼度が高い
                })

        # 統合元が2つ以上ある統合のみを返す
        return [merge for merge in merges if merge['merge_count'] >= 2]

    def _calculate_weighted_vector_distance(
        self, vector1: np.ndarray, vector2: np.ndarray
    ) -> float:
        """
        重み付きユークリッド距離を計算

        色変化に頑健にするため、色の特徴量とパッチハッシュの重みを下げる

        Args:
            vector1: 特徴量ベクトル1（57次元）
            vector2: 特徴量ベクトル2（57次元）

        Returns:
            重み付きユークリッド距離
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector shapes must match: {vector1.shape} vs {vector2.shape}")

        # 重みベクトルを作成（57次元）
        weights = np.ones(57, dtype=np.float32)

        # 色の特徴量（インデックス0-9）: 重みを下げる
        weights[0:10] = self.config.feature_weight_color

        # 形状の特徴量（インデックス10-18）: 通常の重み
        weights[10:19] = self.config.feature_weight_shape

        # 位置の特徴量（インデックス19-20）: 通常の重み
        weights[19:21] = self.config.feature_weight_position

        # パッチハッシュ3×3（インデックス21-28）: 重みを下げる（色情報の影響を軽減）
        weights[21:29] = self.config.feature_weight_patch_hash_3x3

        # パッチハッシュ2×2（インデックス29-32）: 重みを下げる（色情報の影響を軽減）
        weights[29:33] = self.config.feature_weight_patch_hash_2x2

        # ダウンサンプリング4×4（インデックス33-48）: 通常の重み
        weights[33:49] = self.config.feature_weight_downscaled_bitmap

        # 境界方向ヒストグラム（インデックス49-52）: 通常の重み
        weights[49:53] = self.config.feature_weight_contour_direction

        # スケルトン化特徴（インデックス53-54）: 通常の重み
        weights[53:55] = self.config.feature_weight_skeleton

        # 形状中心位置（インデックス55-56）: 通常の重み
        weights[55:57] = self.config.feature_weight_local_centroid

        # 重み付きユークリッド距離を計算
        diff = vector1 - vector2
        weighted_diff = diff * weights
        distance = np.linalg.norm(weighted_diff)

        return distance
