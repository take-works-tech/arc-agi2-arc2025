"""
カテゴリ分けモジュール

オブジェクトをカテゴリに分類
"""

from typing import List, Dict, Any, Optional
import random
import numpy as np
from collections import Counter

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .data_structures import ObjectInfo, CategoryInfo, get_object_signature
from .similarity_calculator import SimilarityCalculator
from .similarity_utils import calculate_symmetry_score, calculate_hu_moment, get_pixels_set

# 定数定義（マジックナンバーの定数化）
SHAPE_SIMILARITY_MAX_WITHOUT_EXACT_MATCH = 0.95  # ピクセル完全一致でない場合の形状類似度の最大値
CENTER_DISTANCE_MAX = 10.0  # 中心位置の距離の最大値（正規化用）
POSITION_X_DIFF_MAX = 30.0  # X座標の差の最大値（正規化用）
POSITION_Y_DIFF_MAX = 30.0  # Y座標の差の最大値（正規化用）
PERIMETER_AREA_RATIO_MIN = 0.001  # 周囲長/面積比の最小値（ゼロ除算を避けるため）


class CategoryClassifier:
    """カテゴリ分類器"""

    def __init__(self, config: Any, similarity_calculator: SimilarityCalculator):
        self.config = config
        self.similarity_calculator = similarity_calculator
        # 類似度計算のキャッシュ（パターンごとに保持）
        # キー: (pattern_idx, obj1_id, obj2_id), 値: similarity
        self._similarity_cache: Dict[tuple, float] = {}
        # オブジェクトIDのマッピング（キャッシュキー生成用）
        self._object_id_map: Dict[int, int] = {}  # オブジェクトのインデックス -> 一意のID

    def categorize_objects(
        self, objects: List[ObjectInfo], transformation_patterns: Dict[str, Any],
        pattern_idx: int, num_input_grids: int, max_colors_per_grid: int = 1,
        total_patterns: int = 3071, retry_count: int = 0
    ) -> List[CategoryInfo]:
        """
        オブジェクトをカテゴリに分類

        Args:
            objects: オブジェクトのリスト
            transformation_patterns: 変換パターン情報
            pattern_idx: カテゴリ分けパターンのインデックス
            num_input_grids: 入力グリッド数

        Returns:
            カテゴリ情報のリスト
        """
        import time
        timing_info = {
            'param_generation': 0.0,
            'set_transformation_patterns': 0.0,  # 変換パターン情報設定時間
            'object_id_generation': 0.0,  # オブジェクトID生成時間
            'similarity_matrix': 0.0,
            'clustering': 0.0,
            'clustering_find_max_pair': 0.0,
            'clustering_add_objects': 0.0,
            'feature_extraction': 0.0,
            'feature_vector_construction': 0.0,  # 特徴量ベクトル構築時間
            'kmeans_fit_predict': 0.0,  # K-means fit_predict時間
            'category_formation': 0.0,  # カテゴリ形成時間
            'category_feature_extraction': 0.0,  # カテゴリ特徴抽出時間
            # 未計測処理の詳細計測
            'clustering_loop_overhead': 0.0,  # whileループのオーバーヘッド
            'clustering_preparation': 0.0,  # クラスタリング準備（unclassified_list作成など）
            'clustering_param_extraction': 0.0,  # パラメータ抽出時間
            'clustering_array_conversion': 0.0,  # numpy配列への変換時間
            'clustering_dimension_reduction': 0.0,  # 次元削減時間
            'clustering_n_clusters_calculation': 0.0,  # クラスタ数計算時間
            'clustering_post_processing': 0.0,  # クラスタリング後の処理時間
            'total': 0.0
        }
        total_start = time.time()

        if not objects:
            return []

        # hole_count特徴量の条件付き使用チェック（組み合わせ取得前に実行）
        # 入力グリッドの一定割合以上のオブジェクトに穴がある場合のみ使用
        # available_featuresが指定されている場合、その回で使える要素からhole_countを除外
        if hasattr(self.config, 'hole_count_min_ratio') and self.config.hole_count_min_ratio is not None:
            # 穴があるオブジェクトの割合を計算
            objects_with_holes = [obj for obj in objects if obj.hole_count is not None and obj.hole_count > 0]
            hole_ratio = len(objects_with_holes) / len(objects) if objects else 0.0

            # 一定割合未満の場合は、その回で使える要素からhole_countを除外
            if hole_ratio < self.config.hole_count_min_ratio:
                if self.config.available_features is None:
                    # all_available_featuresからhole_countを除外
                    self.config.available_features = [f for f in self.config.all_available_features if f != 'hole_count']
                elif 'hole_count' in self.config.available_features:
                    # available_featuresからhole_countを除外
                    self.config.available_features = [f for f in self.config.available_features if f != 'hole_count']

        # カテゴリ分けパラメータを生成（ループ2の各ループで異なる）
        param_start = time.time()
        category_params = self._generate_category_params(pattern_idx, total_patterns=total_patterns, retry_count=retry_count)
        timing_info['param_generation'] = time.time() - param_start

        # 変換パターン情報をオブジェクトに設定
        set_patterns_start = time.time()
        self._set_transformation_patterns(objects, transformation_patterns)
        timing_info['set_transformation_patterns'] = time.time() - set_patterns_start

        # 遅延計算用の準備
        # オブジェクトの一意IDを生成（キャッシュキー用）
        # オブジェクトのハッシュ値をIDとして使用（pixelsとcolorの組み合わせ）
        object_id_start = time.time()
        object_ids = []
        for i, obj in enumerate(objects):
            # オブジェクトの一意性を表すハッシュ（pixelsとcolorの組み合わせ）
            obj_hash = hash((tuple(sorted(obj.pixels)) if obj.pixels else (), obj.color))
            object_ids.append(obj_hash)
        timing_info['object_id_generation'] = time.time() - object_id_start

        # 統計情報（timing_infoに設定するために使用）
        similarity_stats = {
            'computed_pairs': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        timing_info['num_objects'] = len(objects)
        timing_info['similarity_matrix'] = 0.0  # K-meansクラスタリングでは事前計算は行わない

        # クラスタリング（動的閾値を使用）
        clustering_start = time.time()
        category_map = {i: -1 for i in range(len(objects))}  # -1は未分類を意味する
        current_category_id = 0
        unclassified_objects = set(range(len(objects)))  # 未分類のオブジェクトのセット

        # 乱数シードの初期化（パターンインデックスに基づいて）
        random.seed(pattern_idx * 1000)  # 再現性のため

        # kmeans_n_clusters_ratioを乱数で生成（設定で有効な場合）
        if self.config.kmeans_use_random_ratio:
            kmeans_ratio = random.uniform(
                self.config.kmeans_n_clusters_ratio_min,
                self.config.kmeans_n_clusters_ratio_max
            )
        else:
            kmeans_ratio = self.config.kmeans_n_clusters_ratio

        clustering_iterations = 0

        while unclassified_objects:
            loop_iteration_start = time.time()
            clustering_iterations += 1

            # whileループの準備処理
            prep_start = time.time()
            unclassified_list = list(unclassified_objects)
            num_unclassified = len(unclassified_list)
            timing_info['clustering_preparation'] += time.time() - prep_start

            # オブジェクト数に関わらず、K-meansでクラスタリングしてから、各クラスタ内で1対1の類似度計算
            # 実際の類似度計算で使用される特徴量に合わせて特徴量ベクトルを構築
            # 1. category_paramsから選択された特徴量を取得
            param_extract_start = time.time()
            selected_main_features = category_params.get('selected_main_features', ['color', 'shape', 'x', 'y'])

            # グループ名を展開（グループ名は1つの特徴量として扱うが、実際の特徴量ベクトル構築時には展開が必要）
            # 例: 'position' → ['x', 'y'], 'symmetry' → ['symmetry_x', 'symmetry_y']
            if hasattr(self.config, 'feature_groups') and self.config.feature_groups:
                expanded_features = []
                for feature in selected_main_features:
                    if feature in self.config.feature_groups:
                        # グループ名の場合は、グループ内の特徴量を展開
                        expanded_features.extend(self.config.feature_groups[feature])
                    else:
                        # 単独特徴量の場合はそのまま追加
                        expanded_features.append(feature)
                selected_main_features = expanded_features

            # hole_count特徴量の条件付き使用
            # 注意: 組み合わせ生成前に除外しているため、ここではチェック不要
            if 'hole_count' in selected_main_features:
                if hasattr(self.config, 'hole_count_min_ratio') and self.config.hole_count_min_ratio is not None:
                    # 穴があるオブジェクトの割合を計算
                    objects_with_holes = [obj for obj in objects if obj.hole_count is not None and obj.hole_count > 0]
                    hole_ratio = len(objects_with_holes) / len(objects) if objects else 0.0

                    # 一定割合未満の場合はhole_countを除外
                    if hole_ratio < self.config.hole_count_min_ratio:
                        selected_main_features = [f for f in selected_main_features if f != 'hole_count']

            timing_info['clustering_param_extraction'] += time.time() - param_extract_start

            # 2. 各オブジェクトの特徴量ベクトルを計算（選択された特徴量のみ）
            feature_vector_start = time.time()
            feature_vectors = []
            feature_weights = []  # 各特徴量の重み

            # 特徴量ベクトルの次元数を固定（ベース特徴量のみ、対応関係と変換パターンは除外）
            # K-meansクラスタリングでは、ベース特徴量（57次元）のみを使用
            # 対応関係と変換パターンは、類似度計算でのみ使用
            FIXED_FEATURE_DIM = 57  # ベース特徴量のみ: 色(10) + 形状(9) + 位置(2) + パッチハッシュ3×3(8) + パッチハッシュ2×2(4) + ダウンサンプリング4×4(16) + 境界方向ヒストグラム(4) + スケルトン化特徴(2) + 形状中心位置(2)
            BASE_FEATURE_DIM = 57  # 色(10) + 形状(9) + 位置(2) + パッチハッシュ3×3(8) + パッチハッシュ2×2(4) + ダウンサンプリング4×4(16) + 境界方向ヒストグラム(4) + スケルトン化特徴(2) + 形状中心位置(2)
            COLOR_DIM = 10
            SHAPE_DIM = 9  # size, aspect_ratio, width, height, hole, perimeter_area_ratio, hu_moment, symmetry, density
            POSITION_DIM = 2  # center_x, center_y

            # オブジェクトループ内の各処理時間を計測
            object_loop_base_vector_copy_time = 0.0
            object_loop_weight_setting_time = 0.0
            object_loop_normalization_time = 0.0
            object_loop_preprocessing_time = 0.0  # オブジェクト取得、base_vector.copy()、np.zeros()などの処理時間
            object_loop_append_time = 0.0  # feature_vectors.append()とfeature_weights.append()の処理時間

            for obj_idx in unclassified_list:
                obj_loop_iteration_start = time.time()  # 変数名を変更して衝突を回避
                obj = objects[obj_idx]

                # ベース特徴量ベクトルを使用（既に計算済み）
                obj_get_start = time.time()
                if obj.base_feature_vector is not None:
                    # ベースベクトルをコピー
                    base_vector = obj.base_feature_vector.copy()
                else:
                    # ベースベクトルが存在しない場合、0.0で埋める（安全性のため）
                    base_vector = np.zeros(BASE_FEATURE_DIM, dtype=np.float32)
                obj_get_time = time.time() - obj_get_start

                # 固定次元数の特徴量ベクトルを初期化（ベース特徴量のみ）
                zeros_start = time.time()
                feature_vector = np.zeros(FIXED_FEATURE_DIM, dtype=np.float32)
                weights = np.zeros(FIXED_FEATURE_DIM, dtype=np.float32)
                zeros_time = time.time() - zeros_start

                # ベースベクトルをコピー（色、形状、位置、部分構造特徴）
                base_vector_copy_start = time.time()
                feature_vector[:BASE_FEATURE_DIM] = base_vector
                object_loop_base_vector_copy_time += time.time() - base_vector_copy_start
                object_loop_preprocessing_time += obj_get_time + zeros_time + (base_vector_copy_start - obj_loop_iteration_start - obj_get_time - zeros_time)

                offset = 0

                # 重み設定処理の時間を計測
                weight_setting_start = time.time()

                # 1. 色の特徴量（10次元: 0-9の各色、ワンホットエンコーディング）
                color_weight = category_params.get('color_weight', 0.2) if 'color' in selected_main_features else 0.0
                # 選択されていない場合は重みを0.0のまま
                if 'color' in selected_main_features:
                    for i in range(COLOR_DIM):
                        weights[offset + i] = color_weight
                offset += COLOR_DIM

                # 2. 形状の特徴量（9次元: size, aspect_ratio, width, height, hole, perimeter_area_ratio, hu_moment, symmetry, density）
                #    ベースベクトルのインデックス: 10-18
                shape_feature_index_map = {
                    'size': 0,
                    'aspect_ratio': 1,
                    'width': 2,
                    'height': 3,
                    'hole': 4,
                    'hole_count': 4,  # hole_countはholeと同じインデックス
                    'pixel': 5,  # correspondence_detector.pyで使用（perimeter_area_ratioにマッピング）
                    'perimeter_area_ratio': 5,
                    'hu_moment': 6,
                    'symmetry': 7,
                    'symmetry_x': 7,  # symmetry_xはsymmetryと同じインデックス（注意: 実際には別のインデックスが必要な場合もある）
                    'symmetry_y': 7,  # symmetry_yはsymmetryと同じインデックス（注意: 実際には別のインデックスが必要な場合もある）
                    'density': 8
                }

                shape_weight = category_params.get('shape_weight', 0.2)

                # 新しい詳細特徴量をチェック
                shape_detail_features = ['width', 'height', 'size', 'hole_count', 'symmetry_x', 'symmetry_y', 'shape_other']

                for detail_feature in shape_detail_features:
                    if detail_feature in selected_main_features:
                        if detail_feature == 'shape_other':
                            # shape_otherは新しい特徴量（36次元）を使用
                            # 後述のshape_other処理で処理される
                            pass
                        else:
                            # 既存の特徴量ベクトルにマッピング
                            feature_index = shape_feature_index_map.get(detail_feature)
                            if feature_index is not None:
                                weights[offset + feature_index] = shape_weight

                offset += SHAPE_DIM

                # 2-1. 新しい特徴量（36次元）の重み付け
                # ベースベクトルのインデックス: 21-56（特徴量ベクトルでも同じインデックス）
                # 新しい特徴量の次元数: 8 + 4 + 16 + 4 + 2 + 2 = 36次元
                NEW_FEATURE_DIM = 36

                # shape_other サブグループ: patch_hash_3x3, patch_hash_2x2, downscaled_bitmap, contour_direction, skeleton, local_centroid

                # 第3段階で選択されたshape_other詳細特徴量を使用
                # patch_hash_3x3 が選択されている場合
                if 'patch_hash_3x3' in selected_main_features:
                    # パッチハッシュ3×3（8次元、ベースベクトルインデックス21-28）
                    for i in range(8):
                        weights[offset + i] = shape_weight

                # patch_hash_2x2 が選択されている場合
                if 'patch_hash_2x2' in selected_main_features:
                    # パッチハッシュ2×2（4次元、ベースベクトルインデックス29-32）
                    for i in range(4):
                        weights[offset + 8 + i] = shape_weight

                # downscaled_bitmap が選択されている場合
                if 'downscaled_bitmap' in selected_main_features:
                    # ダウンサンプリング4×4（16次元、ベースベクトルインデックス33-48）
                    for i in range(16):
                        weights[offset + 12 + i] = shape_weight

                # contour_direction が選択されている場合
                if 'contour_direction' in selected_main_features:
                    # 境界方向ヒストグラム（4次元、ベースベクトルインデックス49-52）
                    for i in range(4):
                        weights[offset + 28 + i] = shape_weight

                # skeleton が選択されている場合
                if 'skeleton' in selected_main_features:
                    # スケルトン化特徴（2次元、ベースベクトルインデックス53-54）
                    for i in range(2):
                        weights[offset + 32 + i] = shape_weight

                # local_centroid が選択されている場合
                if 'local_centroid' in selected_main_features:
                    # 形状中心位置（2次元、ベースベクトルインデックス55-56）
                    for i in range(2):
                        weights[offset + 34 + i] = shape_weight

                # 新しい特徴量の次元数をoffsetに追加
                offset += NEW_FEATURE_DIM

                # 3. 位置の特徴量（2次元: center_x, center_y）
                position_weight = category_params.get('position_weight', 0.2)
                # 'x'と'y'はselected_main_featuresに直接含まれる
                if 'x' in selected_main_features:
                    # x座標はcenter_xを使用
                    weights[offset + 0] = position_weight
                if 'y' in selected_main_features:
                    # y座標はcenter_yを使用
                    weights[offset + 1] = position_weight
                offset += POSITION_DIM

                # 注意: 対応関係と変換パターンの次元は除外
                # これらは類似度計算でのみ使用される（特徴量ベクトルには含めない）
                # これにより、訓練オブジェクトとテストオブジェクトが同じ特徴量空間（57次元）で扱える

                # 検証: offsetがFIXED_FEATURE_DIMと一致することを確認
                assert offset == FIXED_FEATURE_DIM, f"Feature vector dimension mismatch: offset={offset}, expected={FIXED_FEATURE_DIM}"

                # 重み設定処理の時間を計測終了
                object_loop_weight_setting_time += time.time() - weight_setting_start

                # 特徴量ベクトルを正規化（既にnumpy配列）
                normalization_start = time.time()
                feature_vector = np.clip(feature_vector, 0.0, 1.0)

                # 重みを適用（K-meansの前に重み付けを行う）
                # 重みを正規化（合計が1.0になるように）
                weights_sum = np.sum(weights)
                if weights_sum > 0:
                    weights_normalized = weights / weights_sum
                else:
                    weights_normalized = weights
                # 特徴量ベクトルに重みを適用
                feature_vector = feature_vector * np.sqrt(weights_normalized)
                object_loop_normalization_time += time.time() - normalization_start

                # append処理の時間を計測
                append_start = time.time()
                feature_vectors.append((obj_idx, feature_vector))
                feature_weights.append(weights_normalized)
                object_loop_append_time += time.time() - append_start

            # 3. K-meansでクラスタリング
            # すべてのベクトルをnumpy配列に変換
            array_conv_start = time.time()
            vec_array = np.array([vec for _, vec in feature_vectors])
            obj_indices = [obj_idx for obj_idx, _ in feature_vectors]
            timing_info['clustering_array_conversion'] += time.time() - array_conv_start

            # 使用される次元のインデックスを取得（重みが0.0より大きい次元のみ）
            # すべてのオブジェクトで同じ特徴量が選択されているため、最初のオブジェクトの重みを使用
            dim_reduction_start = time.time()
            if feature_weights:
                used_dim_indices = np.where(feature_weights[0] > 0.0)[0]
            else:
                used_dim_indices = np.arange(FIXED_FEATURE_DIM)

            # 特徴量ベクトルから使用される次元のみを抽出（次元削減による高速化）
            vec_array_reduced = vec_array[:, used_dim_indices]
            original_dim = vec_array.shape[1]
            reduced_dim = vec_array_reduced.shape[1]
            timing_info['clustering_dimension_reduction'] += time.time() - dim_reduction_start

            # 特徴量ベクトル構築時間を記録
            feature_vector_construction_elapsed = time.time() - feature_vector_start
            if timing_info is not None:
                timing_info['feature_vector_construction'] = timing_info.get('feature_vector_construction', 0.0) + feature_vector_construction_elapsed
                timing_info['object_loop_base_vector_copy_time'] = timing_info.get('object_loop_base_vector_copy_time', 0.0) + object_loop_base_vector_copy_time
                timing_info['object_loop_weight_setting_time'] = timing_info.get('object_loop_weight_setting_time', 0.0) + object_loop_weight_setting_time
                timing_info['object_loop_normalization_time'] = timing_info.get('object_loop_normalization_time', 0.0) + object_loop_normalization_time
                timing_info['object_loop_preprocessing_time'] = timing_info.get('object_loop_preprocessing_time', 0.0) + object_loop_preprocessing_time
                timing_info['object_loop_append_time'] = timing_info.get('object_loop_append_time', 0.0) + object_loop_append_time
                # オブジェクト数と処理時間の関係を記録（検証用）
                timing_info['num_unclassified_objects'] = len(unclassified_list)
                timing_info['feature_vector_construction_elapsed'] = feature_vector_construction_elapsed
                timing_info['feature_dim_reduction'] = {
                    'original_dim': original_dim,
                    'reduced_dim': reduced_dim,
                    'reduction_ratio': reduced_dim / original_dim if original_dim > 0 else 0.0,
                    'used_dim_indices': used_dim_indices.tolist()
                }

            # K-meansでクラスタリング
            # sklearnは必須（K-means統一化のため）
            if not SKLEARN_AVAILABLE:
                raise ImportError(
                    "sklearn is required for category classification. "
                    "Please install it with: pip install scikit-learn"
                )

            # クラスタ数を動的に決定
            # オブジェクト数が少ない場合でも適切に動作するように調整
            n_clusters_calc_start = time.time()
            # configから最小クラスタ数を取得（設定可能）
            min_clusters = getattr(self.config, 'kmeans_min_clusters', 2)

            if num_unclassified <= min_clusters:
                # オブジェクト数が最小クラスタ数以下の場合、クラスタ数 = オブジェクト数
                n_clusters = num_unclassified
            else:
                # クラスタ数を計算（色数も考慮）
                # 1. タスク内の各入力グリッドの色数の最大値を使用（同じ色が同じカテゴリに分類されやすくするため）
                num_colors = max_colors_per_grid

                # 2. オブジェクト数ベースのクラスタ数（設定可能な最小クラスタ数を使用）
                n_clusters_by_objects = max(min_clusters, int(np.sqrt(num_unclassified / 2) * kmeans_ratio))

                # 3. 色数ベースのクラスタ数（色数に近づける、設定可能な最小クラスタ数を使用）
                n_clusters_by_colors = max(min_clusters, num_colors)

                # 4. 両方を考慮してクラスタ数を決定（色数とオブジェクト数の最大値を優先）
                # オブジェクト数ベースと色数ベースのうち、大きい方を優先的に使用
                # これにより、色数が少ない場合でもオブジェクト数に基づいてより多くのカテゴリが生成される
                n_clusters = max(min_clusters, min(
                    max(n_clusters_by_colors, n_clusters_by_objects),
                    num_unclassified
                ))
                n_clusters = min(n_clusters, num_unclassified)  # クラスタ数はオブジェクト数を超えないように

            # random_stateをpattern_idxに基づいて設定（異なるパターンで異なるクラスタリング結果を得るため）
            kmeans_random_state = pattern_idx if hasattr(self.config, 'kmeans_random_state') and self.config.kmeans_random_state is None else (pattern_idx * 1000 + self.config.kmeans_random_state) if hasattr(self.config, 'kmeans_random_state') else pattern_idx
            timing_info['clustering_n_clusters_calculation'] += time.time() - n_clusters_calc_start

            # 次元削減された特徴量ベクトルでK-meansクラスタリングを実行
            kmeans_init_start = time.time()
            # 設定ファイルからK-means最適化パラメータを取得
            kmeans_n_init = getattr(self.config, 'kmeans_n_init', 3)
            kmeans_max_iter = getattr(self.config, 'kmeans_max_iter', 100)
            kmeans_tol = getattr(self.config, 'kmeans_tol', 1e-3)
            kmeans_init_method = getattr(self.config, 'kmeans_init_method', 'k-means++')
            kmeans_algorithm = getattr(self.config, 'kmeans_algorithm', 'lloyd')

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=kmeans_random_state,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
                tol=kmeans_tol,
                init=kmeans_init_method,
                algorithm=kmeans_algorithm
            )
            kmeans_init_time = time.time() - kmeans_init_start
            if timing_info is not None:
                timing_info['kmeans_init'] = timing_info.get('kmeans_init', 0.0) + kmeans_init_time

            kmeans_fit_start = time.time()
            cluster_labels = kmeans.fit_predict(vec_array_reduced)
            kmeans_fit_predict_time = time.time() - kmeans_fit_start
            if timing_info is not None:
                timing_info['kmeans_fit_predict'] = timing_info.get('kmeans_fit_predict', 0.0) + kmeans_fit_predict_time

            # 4. カテゴリ形成（K-meansクラスタリングのみ）
            # クラスタ = カテゴリ（クラスタ内での類似度計算を削除して大幅な高速化）
            category_formation_start = time.time()
            category_map_update_time = 0.0  # category_map更新の処理時間
            unclassified_remove_time = 0.0  # unclassified_objects.remove()の処理時間

            for cluster_id in range(n_clusters):
                cluster_indices_start = time.time()
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_obj_indices = [obj_indices[i] for i in cluster_indices]
                cluster_indices_time = time.time() - cluster_indices_start

                # クラスタをそのままカテゴリとする
                for obj_idx in cluster_obj_indices:
                    if obj_idx in unclassified_objects:
                        map_update_start = time.time()
                        category_map[obj_idx] = current_category_id
                        category_map_update_time += time.time() - map_update_start

                        remove_start = time.time()
                        unclassified_objects.remove(obj_idx)
                        unclassified_remove_time += time.time() - remove_start

                current_category_id += 1

            # K-meansでクラスタリングした場合は、各クラスタ内でカテゴリ形成が完了しているため、
            # このwhileループを抜ける
            category_formation_elapsed = time.time() - category_formation_start
            if timing_info is not None:
                timing_info['category_formation'] = timing_info.get('category_formation', 0.0) + category_formation_elapsed
                timing_info['category_map_update_time'] = timing_info.get('category_map_update_time', 0.0) + category_map_update_time
                timing_info['unclassified_remove_time'] = timing_info.get('unclassified_remove_time', 0.0) + unclassified_remove_time

            # whileループのオーバーヘッドを計測（ループ全体から測定済み時間を差し引く）
            loop_iteration_time = time.time() - loop_iteration_start
            # 既に記録されている時間を使用（重複計測を避ける）
            kmeans_fit_predict_time = timing_info.get('kmeans_fit_predict', 0.0)
            category_formation_time = timing_info.get('category_formation', 0.0)
            measured_loop_time = (
                timing_info.get('clustering_preparation', 0.0) +
                timing_info.get('clustering_param_extraction', 0.0) +
                feature_vector_construction_elapsed +
                timing_info.get('clustering_array_conversion', 0.0) +
                timing_info.get('clustering_dimension_reduction', 0.0) +
                timing_info.get('clustering_n_clusters_calculation', 0.0) +
                kmeans_fit_predict_time +
                category_formation_time
            )
            loop_overhead = loop_iteration_time - measured_loop_time
            if loop_overhead > 0:
                timing_info['clustering_loop_overhead'] = timing_info.get('clustering_loop_overhead', 0.0) + loop_overhead

            break

        clustering_end = time.time()
        timing_info['clustering'] = clustering_end - clustering_start
        timing_info['clustering_iterations'] = clustering_iterations

        # 遅延計算の統計情報を追加
        timing_info['computed_pairs'] = similarity_stats['computed_pairs']
        timing_info['cache_hits'] = similarity_stats['cache_hits']
        timing_info['cache_misses'] = similarity_stats['cache_misses']
        total_requests = similarity_stats['cache_hits'] + similarity_stats['cache_misses']
        timing_info['cache_hit_rate'] = similarity_stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        # 理論上の最大ペア数（比較用）
        max_possible_pairs = len(objects) * (len(objects) - 1) // 2
        timing_info['max_possible_pairs'] = max_possible_pairs
        timing_info['computation_reduction'] = 1.0 - (similarity_stats['computed_pairs'] / max_possible_pairs) if max_possible_pairs > 0 else 0.0

        # 注意: 動的閾値を使用するアルゴリズムでは、すべてのオブジェクトがカテゴリに属するため、
        # 最小オブジェクト数チェックは不要（削除済み）

        # カテゴリの特徴抽出
        feature_start = time.time()
        categories = []
        unique_category_ids = sorted(set(category_map.values()))

        # カテゴリ特徴抽出の各処理時間を計測
        category_feature_timing = {
            'get_representative_color': 0.0,
            'get_shape_info': 0.0,
            'get_position_info': 0.0,
            'get_representative_correspondence_type': 0.0,
            'category_objects_list_comprehension': 0.0,  # category_objectsのリスト内包表記
            'statistics_calculation': 0.0,  # 統計情報の計算
            'object_count_per_grid_calculation': 0.0,  # object_count_per_gridの計算
            'category_info_creation': 0.0,  # CategoryInfoオブジェクトの作成
            'category_append': 0.0,  # categories.append()の処理
            'other_category_operations': 0.0
        }

        for category_id in unique_category_ids:
            # category_objectsのリスト内包表記の時間を計測
            category_objects_start = time.time()
            category_objects = [obj for i, obj in enumerate(objects)
                              if category_map[i] == category_id]
            category_feature_timing['category_objects_list_comprehension'] += time.time() - category_objects_start

            # カテゴリに属する最小オブジェクト数: 1個
            # 空のカテゴリ（オブジェクト数=0）は除外
            if not category_objects:
                continue

            # 代表特徴を抽出
            color_start = time.time()
            representative_color = self._get_representative_color(category_objects)
            category_feature_timing['get_representative_color'] += time.time() - color_start

            shape_start = time.time()
            shape_info = self._get_shape_info(category_objects)
            category_feature_timing['get_shape_info'] += time.time() - shape_start

            # 対応関係分析（設定で無効化可能）
            correspondence_start = time.time()
            if self.config.enable_correspondence_detection:
                representative_correspondence_type = self._get_representative_correspondence_type(category_objects)
                # 統計情報
                disappearance_ratio = sum(1 for obj in category_objects
                                        if obj.correspondence_type == 'one_to_zero') / len(category_objects) if category_objects else 0.0
                appearance_ratio = sum(1 for obj in category_objects
                                     if obj.correspondence_type == 'zero_to_one') / len(category_objects) if category_objects else 0.0
                split_ratio = sum(1 for obj in category_objects
                                if obj.correspondence_type == 'one_to_many') / len(category_objects) if category_objects else 0.0
                merge_ratio = sum(1 for obj in category_objects
                                if obj.correspondence_type == 'many_to_one') / len(category_objects) if category_objects else 0.0
            else:
                # 対応関係分析を無効化した場合、デフォルト値を設定
                representative_correspondence_type = None
                disappearance_ratio = 0.0
                appearance_ratio = 0.0
                split_ratio = 0.0
                merge_ratio = 0.0
            category_feature_timing['get_representative_correspondence_type'] += time.time() - correspondence_start

            # 位置情報を抽出
            position_start = time.time()
            position_info = self._get_position_info(category_objects)
            category_feature_timing['get_position_info'] += time.time() - position_start

            # 統計情報の計算時間を計測（対応関係分析が無効化されている場合でも計測）
            statistics_start = time.time()
            category_feature_timing['statistics_calculation'] += time.time() - statistics_start

            # 各入力グリッドでのオブジェクト数
            object_count_start = time.time()
            object_count_per_grid = [
                sum(1 for obj in category_objects if obj.grid_index == grid_idx)
                for grid_idx in range(num_input_grids)
            ]
            category_feature_timing['object_count_per_grid_calculation'] += time.time() - object_count_start

            # カテゴリの有効性検証
            # 各入力グリッドでカテゴリに属するオブジェクトが存在するかを確認
            grids_with_objects = sum(1 for count in object_count_per_grid if count > 0)
            presence_ratio = grids_with_objects / num_input_grids if num_input_grids > 0 else 0.0

            # カテゴリの有効性検証
            # カテゴリに属する最小オブジェクト数: 1個（既に上記のif not category_objectsで除外済み）
            # 有効性が低いカテゴリ（存在するグリッドが30%未満）は除外
            # ただし、訓練ペアのみに存在するカテゴリは有効とみなす
            # また、オブジェクト数が1個以上で、少なくとも1つのグリッドに存在する場合は有効とみなす
            is_valid = (
                presence_ratio >= 0.3 or  # 30%以上のグリッドに存在
                any(obj.is_train for obj in category_objects) or  # 訓練ペアに存在
                (len(category_objects) >= 1 and grids_with_objects >= 1)  # 少なくとも1つのグリッドに存在（最小オブジェクト数: 1個）
            )

            # CategoryInfoオブジェクトの作成時間を計測
            category_info_start = time.time()
            category = CategoryInfo(
                category_id=category_id,
                objects=category_objects,
                representative_color=representative_color,
                shape_info=shape_info,
                position_info=position_info,
                representative_correspondence_type=representative_correspondence_type,
                object_count_per_grid=object_count_per_grid,
                total_objects=len(category_objects),
                disappearance_ratio=disappearance_ratio,
                appearance_ratio=appearance_ratio,
                split_ratio=split_ratio,
                merge_ratio=merge_ratio
            )
            category_feature_timing['category_info_creation'] += time.time() - category_info_start

            # 有効なカテゴリのみ追加
            if is_valid:
                # categories.append()の時間を計測
                append_start = time.time()
                categories.append(category)
                category_feature_timing['category_append'] += time.time() - append_start

        feature_extraction_elapsed = time.time() - feature_start
        timing_info['feature_extraction'] = feature_extraction_elapsed
        timing_info['category_feature_extraction'] = feature_extraction_elapsed
        timing_info['category_feature_timing'] = category_feature_timing

        # timing_info['total']の内訳を明確にするため、各処理時間の合計を計算
        # total_startからclustering_startまでの時間
        pre_clustering_time = clustering_start - total_start
        # clustering終了からfeature_startまでの時間
        post_clustering_pre_feature_time = feature_start - clustering_end
        # feature_extraction終了からtotal_start終了までの時間
        feature_extraction_end = time.time()
        post_feature_time = feature_extraction_end - (feature_start + feature_extraction_elapsed)

        timing_info['pre_clustering_time'] = pre_clustering_time
        timing_info['post_clustering_pre_feature_time'] = post_clustering_pre_feature_time
        timing_info['post_feature_time'] = post_feature_time

        timing_info['total'] = time.time() - total_start
        timing_info['num_categories'] = len(categories)

        # 使用された特徴量情報を追加（カテゴリ数=1の原因分析用）
        # category_paramsは_generate_category_paramsで生成されたもの
        timing_info['selected_main_features'] = category_params.get('selected_main_features', [])
        timing_info['selected_shape_details'] = category_params.get('selected_shape_details', [])
        timing_info['selected_shape_other_details'] = category_params.get('selected_shape_other_details', [])
        timing_info['color_weight'] = category_params.get('color_weight', 0.0)
        timing_info['shape_weight'] = category_params.get('shape_weight', 0.0)
        timing_info['position_weight'] = category_params.get('position_weight', 0.0)

        # タイミング情報をクラス変数に保存（後で取得できるように）
        # 注意: これは一時的な実装で、本来は別の方法で返すべき
        if not hasattr(self, '_last_timing_info'):
            self._last_timing_info = {}
        self._last_timing_info[pattern_idx] = timing_info

        return categories

    def get_last_timing_info(self, pattern_idx: int) -> Optional[Dict[str, Any]]:
        """最後に実行したカテゴリ分けのタイミング情報を取得"""
        if hasattr(self, '_last_timing_info'):
            return self._last_timing_info.get(pattern_idx)
        return None

    def _weighted_random_choice(self, weights: Dict[str, float]) -> str:
        """
        重み付けランダム選択

        Args:
            weights: 特徴量名と重みの辞書

        Returns:
            選択された特徴量名
        """
        items = list(weights.items())
        choices = [item[0] for item in items]
        probabilities = [item[1] for item in items]
        # 確率の正規化
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # 重みがすべて0の場合、均等に選択
            probabilities = [1.0 / len(choices)] * len(choices)
        return random.choices(choices, weights=probabilities, k=1)[0]

    def _generate_weighted_random(
        self, min_val: float, max_val: float, alpha: float, beta: float
    ) -> float:
        """
        重み付きランダム値を生成（ベータ分布を使用）

        Args:
            min_val: 最小値
            max_val: 最大値
            alpha: ベータ分布のalphaパラメータ
            beta: ベータ分布のbetaパラメータ
                  alpha > beta の場合、高い値が選ばれやすい
                  alpha < beta の場合、低い値が選ばれやすい
                  alpha = beta の場合、一様分布に近い（対称的な分布）

        Returns:
            重み付きランダム値
        """
        # ベータ分布から[0, 1]の値を生成
        beta_value = random.betavariate(alpha, beta)

        # 実際の範囲にスケール
        return min_val + beta_value * (max_val - min_val)

    def _generate_category_params(self, pattern_idx: int, total_patterns: int = 3071, retry_count: int = 0) -> Dict[str, Any]:
        """カテゴリ分けパラメータを生成

        Args:
            pattern_idx: パターンインデックス（0から始まる）
            total_patterns: 総パターン数（必須、デフォルトは3071）
            retry_count: 再試行回数（0から始まる、デフォルトは0）
                        再試行時はpattern_idxを固定し、重みだけを変える
        """
        # 特徴量組み合わせはpattern_idxで決定（再試行時も同じ組み合わせを使用）
        # 重みはpattern_idxとretry_countの組み合わせで決定（再試行時に異なる重みを生成）
        # pattern_idx * 10000 + retry_count でシードを生成（pattern_idxとretry_countを分離）
        weight_seed = pattern_idx * 10000 + retry_count
        random.seed(weight_seed)  # 重み生成用のシード

        # すべての重みは同じ範囲を使用
        weight_min = self.config.weight_min
        weight_max = self.config.weight_max

        # 特徴量の組み合わせを決定（1段階構造）
        selected_main_features = []
        selected_shape_details = []
        selected_shape_other_details = []

        # 事前計算された組み合わせを使用（必須）
        if not hasattr(self.config, 'get_precomputed_combinations'):
            raise ValueError("get_precomputed_combinationsメソッドが存在しません")

        # 事前計算された組み合わせを使用（ファイルがない場合は自動的に計算して保存）
        precomputed = self.config.get_precomputed_combinations(total_patterns)
        if pattern_idx not in precomputed:
            raise ValueError(f"pattern_idx={pattern_idx}が事前計算された組み合わせに含まれていません（total_patterns={total_patterns}）")

        # 事前計算された組み合わせを使用（1段階構造）
        combo = precomputed[pattern_idx]

        # 新しい1段階構造: 'features'キーから直接特徴量リストを取得
        # 'features'キーが常に存在するため、フォールバック処理は不要
        if 'features' not in combo:
            # 互換性チェック: 'features'キーが存在しない場合はエラー
            raise ValueError(f"特徴量組み合わせに'features'キーが存在しません: {combo.keys()}")
        selected_main_features = list(combo['features'])

        return {
            # 色の重み：個別のベータ分布パラメータを使用（すべて同じ範囲）
            'color_weight': self._generate_weighted_random(
                weight_min, weight_max,
                self.config.color_weight_alpha, self.config.color_weight_beta
            ),
            # 形状の重み：個別のベータ分布パラメータを使用（すべて同じ範囲）
            'shape_weight': self._generate_weighted_random(
                weight_min, weight_max,
                self.config.shape_weight_alpha, self.config.shape_weight_beta
            ),
            # 位置の重み：個別のベータ分布パラメータを使用（すべて同じ範囲）
            'position_weight': self._generate_weighted_random(
                weight_min, weight_max,
                self.config.position_weight_alpha, self.config.position_weight_beta
            ),
            # 選択された主要特徴量（最終的な特徴量リスト）
            'selected_main_features': selected_main_features,
            # 第2段階で選択されたshape詳細特徴量
            'selected_shape_details': selected_shape_details,
            # 第3段階で選択されたshape_other詳細特徴量
            'selected_shape_other_details': selected_shape_other_details,
        }

    def _set_transformation_patterns(
        self, objects: List[ObjectInfo], transformation_patterns: Dict[str, Any]
    ):
        """
        変換パターン情報をオブジェクトに設定

        Args:
            objects: オブジェクトのリスト（訓練+テストの両方の入力グリッドのオブジェクトを含む）
            transformation_patterns: 変換パターン情報（1つのペアの変換パターン情報、辞書形式）

        注意:
            - 変換パターン情報は訓練ペアのみで分析される
            - テスト入力グリッドのオブジェクトは変換パターン情報を持たない
            - オブジェクトのシグネチャ（ピクセル座標と色）を使用してマッピングする
        """
        if not transformation_patterns or not isinstance(transformation_patterns, dict):
            return

        # カテゴリ分類時に使用されるオブジェクトのシグネチャからオブジェクトへのマッピング
        obj_signature_to_obj = {}
        for obj in objects:
            sig = get_object_signature(obj)
            if sig:
                obj_signature_to_obj[sig] = obj

        # 変換パターン情報から、対応関係の検出時に使用されたオブジェクトの属性を取得
        # transformation_patternsは1つのペアの変換パターン情報（辞書形式）

        # 対応関係分析（設定で無効化可能）
        if self.config.enable_correspondence_detection:
            # 1対1対応関係
            correspondences = transformation_patterns.get('correspondences', [])
            for corr in correspondences:
                input_obj = corr.get('input_obj')
                output_obj = corr.get('output_obj')
                if input_obj:
                    input_sig = get_object_signature(input_obj)
                    if input_sig and input_sig in obj_signature_to_obj:
                        # カテゴリ分類時に使用されるオブジェクトに属性を設定
                        target_obj = obj_signature_to_obj[input_sig]
                        target_obj.correspondence_type = 'one_to_one'
                        if output_obj:
                            target_obj.matched_output_object = output_obj

            # 統合
            merges = transformation_patterns.get('merges', [])
            for merge in merges:
                output_obj = merge.get('output_obj')
                input_objects = merge.get('input_objects', [])
                if output_obj and input_objects:
                    for input_obj in input_objects:
                        input_sig = get_object_signature(input_obj)
                        if input_sig and input_sig in obj_signature_to_obj:
                            # カテゴリ分類時に使用されるオブジェクトに属性を設定
                            target_obj = obj_signature_to_obj[input_sig]
                            target_obj.correspondence_type = 'many_to_one'
                            target_obj.matched_output_objects = [output_obj]

            # 分割
            splits = transformation_patterns.get('splits', [])
            for split in splits:
                input_obj = split.get('input_obj')
                output_objects = split.get('output_objects', [])
                if input_obj and output_objects:
                    input_sig = get_object_signature(input_obj)
                    if input_sig and input_sig in obj_signature_to_obj:
                        # カテゴリ分類時に使用されるオブジェクトに属性を設定
                        target_obj = obj_signature_to_obj[input_sig]
                        target_obj.correspondence_type = 'one_to_many'
                        target_obj.matched_output_objects = output_objects

            # 消失オブジェクト
            disappeared_objects = transformation_patterns.get('disappeared_objects', [])
            for obj in disappeared_objects:
                sig = get_object_signature(obj)
                if sig and sig in obj_signature_to_obj:
                    target_obj = obj_signature_to_obj[sig]
                    target_obj.correspondence_type = 'one_to_zero'
        # 対応関係分析が無効化されている場合、対応関係タイプは設定しない（Noneのまま）

    def _get_representative_color(self, objects: List[ObjectInfo]) -> Optional[List[int]]:
        """
        カテゴリ内のすべてのオブジェクトの色の種類を取得

        Args:
            objects: オブジェクトのリスト

        Returns:
            カテゴリ内のすべてのオブジェクトの色の種類（重複なしのリスト）
        """
        if not objects:
            return None
        # カテゴリ内のすべてのオブジェクトの色の種類を取得（重複なし）
        colors = list(set([obj.color for obj in objects if obj.color is not None]))
        return colors if colors else None

    def _get_shape_info(self, objects: List[ObjectInfo]) -> Optional[Dict[str, Any]]:
        """
        形状情報を取得（最大最小値）

        Returns:
            形状情報の辞書:
            - min_size, max_size (GET_SIZE)
            - min_width, max_width (GET_WIDTH)
            - min_height, max_height (GET_HEIGHT)
            - min_hole_count, max_hole_count (COUNT_HOLES)
            - min_symmetry_x, max_symmetry_x (GET_SYMMETRY_SCORE X)
            - min_symmetry_y, max_symmetry_y (GET_SYMMETRY_SCORE Y)
        """
        if not objects:
            return None

        sizes = [obj.size for obj in objects if obj.size is not None]
        widths = [obj.width for obj in objects if obj.width is not None]
        heights = [obj.height for obj in objects if obj.height is not None]
        hole_counts = [obj.hole_count for obj in objects if obj.hole_count is not None]

        # 対称性スコアを計算
        symmetry_x_scores = []
        symmetry_y_scores = []
        for obj in objects:
            sym_x = calculate_symmetry_score(obj, 'X')
            sym_y = calculate_symmetry_score(obj, 'Y')
            if sym_x is not None:
                symmetry_x_scores.append(sym_x)
            if sym_y is not None:
                symmetry_y_scores.append(sym_y)

        shape_info = {}

        if sizes:
            shape_info['min_size'] = min(sizes)
            shape_info['max_size'] = max(sizes)

        if widths:
            shape_info['min_width'] = min(widths)
            shape_info['max_width'] = max(widths)

        if heights:
            shape_info['min_height'] = min(heights)
            shape_info['max_height'] = max(heights)

        if hole_counts:
            shape_info['min_hole_count'] = min(hole_counts)
            shape_info['max_hole_count'] = max(hole_counts)

        if symmetry_x_scores:
            shape_info['min_symmetry_x'] = min(symmetry_x_scores)
            shape_info['max_symmetry_x'] = max(symmetry_x_scores)

        if symmetry_y_scores:
            shape_info['min_symmetry_y'] = min(symmetry_y_scores)
            shape_info['max_symmetry_y'] = max(symmetry_y_scores)

        return shape_info if shape_info else None

    def _get_position_info(self, objects: List[ObjectInfo]) -> Optional[Dict[str, Any]]:
        """
        位置情報を取得（最大最小値）

        Args:
            objects: オブジェクトのリスト

        Returns:
            位置情報の辞書:
            - min_x, max_x (GET_X)
            - min_y, max_y (GET_Y)
        """
        if not objects:
            return None

        # GET_X, GET_Yはbboxから計算（bbox_left, bbox_top）
        # ObjectInfoのbboxは(min_i, min_j, max_i, max_j)形式
        # GET_Xはbbox_left（min_j）、GET_Yはbbox_top（min_i）
        x_coords = []
        y_coords = []

        for obj in objects:
            if obj.bbox and len(obj.bbox) == 4:
                min_i, min_j, max_i, max_j = obj.bbox
                # GET_Xはbbox_left（min_j）、GET_Yはbbox_top（min_i）
                x_coords.append(min_j)
                y_coords.append(min_i)

        position_info = {}

        if x_coords:
            position_info['min_x'] = min(x_coords)
            position_info['max_x'] = max(x_coords)

        if y_coords:
            position_info['min_y'] = min(y_coords)
            position_info['max_y'] = max(y_coords)

        return position_info if position_info else None

    def _get_representative_correspondence_type(self, objects: List[ObjectInfo]) -> Optional[str]:
        """代表対応関係の種類を取得"""
        if not objects:
            return None
        correspondence_types = [obj.correspondence_type for obj in objects if obj.correspondence_type]
        if not correspondence_types:
            return None
        return Counter(correspondence_types).most_common(1)[0][0]
