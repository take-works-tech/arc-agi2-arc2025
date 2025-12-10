"""
オブジェクトマッチング設定

設計書に基づいた設定クラスの定義
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path


@dataclass
class ObjectMatchingConfig:
    """オブジェクトマッチング設定"""
    # 連結性
    connectivities: List[int] = field(default_factory=lambda: [4, 8])

    # カテゴリ分けパターン数（連結性ごとに個別設定）
    # 3071パターン: 事前計算されたすべての特徴量組み合わせを使用（3071個すべてを試行）
    num_category_patterns_4: int = 3071  # 4連結のカテゴリ分けパターン数
    num_category_patterns_8: int = 3071  # 8連結のカテゴリ分けパターン数

    # マージ用の部分プログラム数
    num_merge_partial_programs: int = 5

    # 並列処理設定（タスク単位での並列処理用）
    # None: CPU数を使用（ただし、メモリ不足を防ぐため最大4に制限）
    # 0: 並列処理を無効化（逐次処理）
    # 1以上: 指定されたワーカー数を使用
    max_parallel_workers: Optional[int] = None  # デフォルトはNone（CPU数を使用、最大4に制限）

    # 類似度閾値
    similarity_threshold: float = 0.7  # 1対1対応を確立するための最小類似度
                                       # この閾値未満の場合、対応関係は確立されない

    # 1対1対応検出の第2段階用設定（ベクトル距離ベース）
    max_vector_distance: float = 0.5  # 最大ベクトル距離A（特徴量ベクトルの距離がA以下で最も近い入出力オブジェクトから順に1対1対応）
                                      # base_feature_vector（57次元）の重み付きユークリッド距離を使用
                                      # 0.3 → 0.5に変更（より柔軟なマッチングが可能）

    # 統合・分割の検出用設定（ベクトル距離ベース）
    max_vector_distance_merge: float = 0.55  # 最大ベクトル距離B（統合の検出用、特徴量ベクトルの距離がB以下で最も近い場合に統合と判定）
                                             # 0.5 → 0.55に変更（統合は複数の入力オブジェクトが1つの出力オブジェクトに統合されるため、特徴量ベクトルの距離が大きくなりやすい。複製もあり得るが、統合後のオブジェクトは複数の入力オブジェクトの特徴を組み合わせたものになるため、より高い閾値が必要）
    max_vector_distance_split: float = 0.6  # 最大ベクトル距離C（分割の検出用、特徴量ベクトルの距離がC以下で最も近い場合に分割と判定）
                                            # 0.45 → 0.6に変更（分割は1つの入力オブジェクトが複数の出力オブジェクトに分割される。複製もあり得るが、各分割後のオブジェクトは元のオブジェクトに近い特徴を持つため、1対1対応と同じ閾値が適切）

    # 重み付きユークリッド距離の重み設定（色変化に頑健にするため）
    # ベース特徴量ベクトル（57次元）の各次元に対する重み
    # インデックス: 0-9: 色(10次元), 10-18: 形状(9次元), 19-20: 位置(2次元),
    #              21-28: パッチハッシュ3×3(8次元), 29-32: パッチハッシュ2×2(4次元),
    #              33-48: ダウンサンプリング4×4(16次元), 49-52: 境界方向ヒストグラム(4次元),
    #              53-54: スケルトン化特徴(2次元), 55-56: 形状中心位置(2次元)
    feature_weight_color: float = 0.05  # 色の特徴量の重み（ARC-AGI2では色変化が一般的なため、さらに低めに設定）
    feature_weight_shape: float = 1.0  # 形状の特徴量の重み（形状の類似性が重要）
    feature_weight_position: float = 0.8  # 位置の特徴量の重み（位置の類似性が重要、1.0 → 0.8に変更：位置変化に頑健）
    feature_weight_patch_hash_3x3: float = 0.6  # パッチハッシュ3×3の重み（色情報を除外したため、形状情報をより活用、0.4 → 0.6に変更：色変化に頑健な形状情報）
    feature_weight_patch_hash_2x2: float = 0.6  # パッチハッシュ2×2の重み（色情報を除外したため、形状情報をより活用、0.4 → 0.6に変更：色変化に頑健な形状情報）
    feature_weight_downscaled_bitmap: float = 1.3  # ダウンサンプリング4×4の重み（1.0 → 1.3に変更：スケール変化に頑健）
    feature_weight_contour_direction: float = 1.3  # 境界方向ヒストグラムの重み（1.0 → 1.3に変更：回転・反転に頑健）
    feature_weight_skeleton: float = 1.0  # スケルトン化特徴の重み
    feature_weight_local_centroid: float = 1.0  # 形状中心位置の重み
    shape_similarity_threshold_min: float = 0.6  # 形状類似度の最小閾値（カテゴリ分け用）
    shape_similarity_threshold_max: float = 0.9  # 形状類似度の最大閾値（カテゴリ分け用）
    position_similarity_threshold_min: float = 0.5  # 位置類似度の最小閾値（カテゴリ分け用）
    position_similarity_threshold_max: float = 0.8  # 位置類似度の最大閾値（カテゴリ分け用）

    # 回転・反転の考慮
    enable_rotation_flip_matching: bool = True  # 回転・反転を考慮するか
    candidate_similarity_threshold: float = 0.5  # 候補を絞り込むための閾値（通常の閾値より低め）
                                                 # この閾値以上の候補に対して回転・反転を考慮した詳細な比較を行う
    max_rotation_flip_candidates: int = 10  # 回転・反転を考慮する候補の最大数（計算コスト制御）
                                            # 候補が多い場合、上位N個のみ回転・反転を考慮

    # 分割・統合の検出閾値（FIT_SHAPE_COLORベース、正規化スコア）
    # 注意: 分割・統合の検出では、FIT_SHAPE_COLORの正規化スコアを使用
    # 正規化スコアは、重複ピクセル率（位置+色）を100000倍、重複ピクセル率（位置）を10000倍、
    # 隣接辺率を1000倍してから合計した値
    split_score_threshold: float = 10000.0  # 分割の正規化スコア閾値（重複ピクセル率が0.1以上の場合の最小スコア）
    merge_score_threshold: float = 10000.0  # 統合の正規化スコア閾値（重複ピクセル率が0.1以上の場合の最小スコア）
    # 正規化スコアの範囲（信頼度計算用）
    fit_shape_color_score_min: float = 10000.0  # 正規化スコアの最小値（重複ピクセル率0.1に対応）
    fit_shape_color_score_max: float = 100000.0  # 正規化スコアの最大値（重複ピクセル率1.0に対応）
    fit_shape_color_score_range: float = 90000.0  # 正規化スコアの範囲（max - min）
    # 信頼度計算の重み
    confidence_score_weight: float = 0.6  # スコアの重み
    confidence_count_weight: float = 0.4  # オブジェクト数の重み
    # オブジェクト数の影響係数
    confidence_count_base: float = 0.5  # 2個の場合の基本値
    confidence_count_increment: float = 0.15  # 1個増えるごとの増分
    max_split_count: int = 5  # 分割の最大数（1つの入力オブジェクトが分割される最大数）
    max_merge_count: int = 5  # 統合の最大数（1つの出力オブジェクトに統合される最大数）

    # 重みの範囲（すべて同じ範囲を使用）
    # すべての重みは同じ範囲（0.0-1.0）を使用し、重み付きランダムで生成
    # 0.8 → 1.0に変更（特徴量間の差をより明確に表現）
    weight_min: float = 0.0  # すべての重みの最小値
    weight_max: float = 1.0  # すべての重みの最大値

    # ベータ分布のパラメータ（各重みごとに個別設定）
    # alpha > beta の場合、高い値が選ばれやすい
    # alpha < beta の場合、低い値が選ばれやすい
    # alpha = beta の場合、一様分布に近い（対称的な分布）
    # 網羅選択の場合、分散を弱めることで安定した重み設定になる（alpha, betaを大きくする）
    # 特徴量の組み合わせの違いでパターンが変わるため、重みの分散は小さくする
    color_weight_alpha: float = 5.0  # 色の重みのベータ分布alpha（分散を弱める）
    color_weight_beta: float = 5.0   # 色の重みのベータ分布beta（一様分布に近づける）
    shape_weight_alpha: float = 5.0  # 形状の重みのベータ分布alpha（分散を弱める）
    shape_weight_beta: float = 5.0   # 形状の重みのベータ分布beta（一様分布に近づける）
    position_weight_alpha: float = 5.0  # 位置の重みのベータ分布alpha（分散を弱める）
    position_weight_beta: float = 5.0   # 位置の重みのベータ分布beta（一様分布に近づける）

    # 背景色決定
    color_consistency_threshold: float = 0.8  # 色の一致度閾値
    bg_color_consistency_threshold: float = 0.8  # 背景色の一致度閾値

    # カテゴリ分け
    # 注意: min_objects_per_categoryは削除（動的閾値アルゴリズムではすべてのオブジェクトがカテゴリに属するため不要）
    max_categories: int = 10

    # パフォーマンス最適化設定
    max_objects_for_vector_distance: int = 100  # この数以上のオブジェクトの場合、K-meansクラスタリングを使用

    # K-meansクラスタリング設定
    kmeans_n_clusters_ratio: float = 0.5  # クラスタ数 = sqrt(n / 2) * ratio（デフォルト: sqrt(n/2)）
    kmeans_n_clusters_ratio_min: float = 0.3  # kmeans_n_clusters_ratioの最小値（乱数生成用）
    kmeans_n_clusters_ratio_max: float = 2.5  # kmeans_n_clusters_ratioの最大値（乱数生成用）
    kmeans_use_random_ratio: bool = True  # kmeans_n_clusters_ratioを乱数で生成するか（True: 乱数生成、False: 固定値を使用）
    kmeans_only_clustering: bool = True  # K-meansクラスタリングのみでクラス分けするか（True: クラスタ=カテゴリ、False: クラスタ内で類似度計算）
    kmeans_random_state: int = 42  # 再現性のための乱数シード
    kmeans_min_clusters: int = 2  # K-meansクラスタリングの最小クラスタ数（デフォルト: 2、部分プログラム生成には2個以上のカテゴリが必要なため）
    # K-means最適化パラメータ
    kmeans_n_init: int = 1  # K-means初期化回数（デフォルト10→1に変更：処理時間を約10倍短縮）
    kmeans_max_iter: int = 100  # K-means最大反復回数（デフォルト300→100に変更：早期終了で時間短縮）
    kmeans_tol: float = 1e-2  # K-means収束許容誤差（デフォルト1e-4→1e-2に変更：より緩い条件で早期終了）
    kmeans_init_method: str = 'k-means++'  # 初期化方法（'k-means++'または'random'、'k-means++'の方が精度が高い）
    kmeans_algorithm: str = 'lloyd'  # K-meansアルゴリズム（'lloyd'または'elkan'、'lloyd'がデフォルト）

    # 動的閾値の範囲設定
    # threshold_range_minとthreshold_range_maxはmax_similarityに応じて動的に調整される
    # max_similarityが大きい場合、両方を小さくする（0.0になる確率を高くする）
    threshold_range_min_base: float = 0.3  # 基準最小範囲（max_similarity=0.0の場合）
    threshold_range_min_min: float = 0.0  # 最小最小範囲（max_similarity=1.0の場合）
    threshold_range_max_base: float = 0.5  # 基準最大範囲（max_similarity=0.0の場合）
    threshold_range_max_min: float = 0.1  # 最小最大範囲（max_similarity=1.0の場合）
    # threshold_range_min = threshold_range_min_min + (threshold_range_min_base - threshold_range_min_min) * (1.0 - max_similarity)
    # threshold_range_max = threshold_range_max_min + (threshold_range_max_base - threshold_range_max_min) * (1.0 - max_similarity)

    # 必須特徴量の選択重み（類似度計算で最低1つは選ばれる特徴量）
    # 合計が1.0になる必要はない（正規化される）
    mandatory_color_weight: float = 0.5      # 色が必須特徴量として選ばれる重み（0.6 → 0.5に変更：入力オブジェクト間では形状の方が重要）
    mandatory_shape_weight: float = 0.35     # 形状が必須特徴量として選ばれる重み（0.25 → 0.35に変更）
    mandatory_position_weight: float = 0.15  # 位置が必須特徴量として選ばれる重み（現状維持）

    # マッチング有効化フラグ
    enable_color_matching: bool = True
    enable_shape_matching: bool = True
    enable_position_matching: bool = True

    # 部分プログラム生成設定
    enable_background_filtering_in_partial_program: bool = True  # 部分プログラムで背景色フィルタリングを有効にするか

    # 対応関係分析設定（新システムでは不要な場合に無効化可能）
    enable_correspondence_detection: bool = False  # 対応関係分析を有効にするか（Falseの場合、対応関係分析をスキップ）
                                                    # デフォルト: False（新システムではObject Graph + GNNとRelation Classifierで代替）

    # 条件組み合わせ設定
    max_conditions_per_category: int = 3  # カテゴリあたりの最大条件数
    enable_or_condition: bool = True  # OR条件の使用を有効化（デフォルト: True）
    max_or_region_combinations: int = 2  # OR条件の最大領域組み合わせ数（デフォルト: 2）

    # 使用可能な特徴量のリスト（すべての特徴量を1段階でフラットに定義）
    # このリストから、指定された要素の組み合わせを生成する
    # 例: available_features=['color', 'shape']と指定すると、[color], [shape], [color, shape]の組み合わせを生成
    #
    # 注意: グループ要素（x, y, width, heightなど）は個別には含まれない
    # - グループ化が有効な場合: グループ名（position, size, symmetry, patch_hash）と単独特徴量のみが選択可能
    # - グループ化が無効な場合: グループ要素を個別に選択したい場合は、available_featuresで明示的に指定する必要がある
    #   ただし、通常はグループ化を有効にして、グループ名を使用することを推奨
    all_available_features: List[str] = field(default_factory=lambda: [
        # グループ名（グループ化が有効な場合に使用）
        'symmetry',  # symmetry_x, symmetry_yを含む
        'patch_hash',  # patch_hash_3x3, patch_hash_2x2を含む
        'dimensions',  # width, heightを含む（グループ名）
        # 単独特徴量
        'color',
        'x',  # 位置情報（x座標）
        'y',  # 位置情報（y座標）
        'size',  # 単独特徴量（width * height、dimensionsグループとは別）
        'hole_count',
        'downscaled_bitmap',
        'contour_direction',
        'skeleton',
        'local_centroid',
    ])

    # 特徴量の重み（プログラム生成フローでの重み付け選択に使用）
    # 組み合わせの選ばれやすさ = 組み合わせに含まれる特徴量の重みの平均
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'color': 1.0,           # 最も重要
        'dimensions': 0.9,      # サイズ情報は重要
        'x': 0.8, 'y': 0.8,    # 位置情報
        'size': 0.7,            # 面積
        'symmetry': 0.6,        # 対称性
        'hole_count': 0.5,      # 穴の数
        'patch_hash': 0.4,      # パッチハッシュ
        'downscaled_bitmap': 0.3,
        'contour_direction': 0.3,
        'skeleton': 0.2,
        'local_centroid': 0.2,
    })

    # 使用する特徴量のリスト（Noneの場合はall_available_featuresを使用）
    # このリストから、すべての組み合わせを生成する
    # 例: ['color', 'shape']と指定すると、[color], [shape], [color, shape]の組み合わせを生成
    available_features: Optional[List[str]] = None  # Noneの場合はall_available_featuresを使用

    # 特徴量のグループ化設定（一緒に選択されるべき特徴量をグループ化）
    # グループ化により、組み合わせ数を削減できる
    # 例: {'position': ['x', 'y']} → 'position'を選択すると、'x'と'y'の両方が含まれる
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'symmetry': ['symmetry_x', 'symmetry_y'],  # 対称性は常に一緒に使用
        'patch_hash': ['patch_hash_3x3', 'patch_hash_2x2'],  # パッチハッシュは常に一緒に使用
        'dimensions': ['width', 'height'],  # 幅と高さは常に一緒に使用（size単独特徴量とは別）
    })

    # グループ単位で選択するか、個別に選択するか
    # True: グループ単位で選択（組み合わせ数を削減）
    # False: 個別に選択（従来通り、すべての組み合わせを生成）
    use_feature_groups: bool = True  # デフォルト: True（グループ単位で選択）

    # 排除推奨の組み合わせ（一緒に使用すべきでない特徴量の組み合わせ）
    # これらの組み合わせは生成されない
    excluded_combinations: List[List[str]] = field(default_factory=lambda: [
        ['width', 'height', 'size'],  # 3つすべては冗長（size ≈ width * height）
    ])

    # hole_count特徴量の条件付き使用設定
    # 入力グリッドの一定割合以上のオブジェクトに穴がある場合のみ、hole_count特徴量を使用する
    # Noneの場合は常に使用（条件なし）
    hole_count_min_ratio: Optional[float] = 0.3  # デフォルト: 30%以上のオブジェクトに穴がある場合のみ使用

    # 注: 3段階構造の設定は削除済み（非推奨で使用されていない）

    # カテゴリ数が1の場合の再試行設定
    # 特徴量組み合わせ数に基づいて自動的に設定
    # 第1段階の組み合わせ数 × 第2段階の組み合わせ数（shapeが選択された場合の最大値）
    @property
    def max_category_retry_count(self) -> int:
        """カテゴリ数が0の場合、異なる特徴量組み合わせで再試行する最大回数"""
        # 3段階構造は削除されたため、固定値（7回）を返す
        # これは従来の第1段階の組み合わせ数に相当
        return 7

    # 事前計算されたパターン組み合わせのキャッシュ
    # キー: (total_patterns, settings_hash) のタプル
    # 値: 事前計算された組み合わせの辞書
    _precomputed_combinations_cache: Optional[Dict[Tuple[int, str], Dict[int, Dict[str, List[str]]]]] = None

    def _get_settings_hash(self) -> str:
        """設定パラメータのハッシュを計算（キャッシュのキーとして使用）

        注意: available_featuresはフィルタリングに使用するため、ハッシュには含めない
        全特徴量から生成された組み合わせをキャッシュし、available_featuresでフィルタリングする

        Returns:
            設定パラメータのハッシュ文字列
        """
        import hashlib
        import json

        # キャッシュに影響する設定パラメータを収集
        # available_featuresはフィルタリングに使用するため、ハッシュには含めない
        settings = {
            'use_feature_groups': self.use_feature_groups,
            'feature_groups': {k: sorted(v) for k, v in sorted(self.feature_groups.items())},
            'excluded_combinations': [sorted(c) for c in sorted(self.excluded_combinations, key=lambda x: tuple(sorted(x)))],
        }

        # JSON文字列に変換してハッシュ化
        settings_str = json.dumps(settings, sort_keys=True)
        return hashlib.md5(settings_str.encode('utf-8')).hexdigest()


    def get_precomputed_combinations(self, total_patterns: int) -> Dict[int, Dict[str, List[str]]]:
        """事前計算されたパターン組み合わせを取得（ファイルから読み込むか計算）

        Args:
            total_patterns: 総パターン数（必須）

        Returns:
            pattern_idxをキーとする辞書。各値は以下のキーを持つ:
            - 'features': 選択された特徴量リスト（1段階構造）
        """
        # 設定パラメータのハッシュを計算（available_featuresを除く）
        # available_featuresはフィルタリングに使用するため、キャッシュキーには含めない
        settings_hash = self._get_settings_hash()
        cache_key = (total_patterns, settings_hash)

        # キャッシュを初期化（初回のみ）
        if self._precomputed_combinations_cache is None:
            self._precomputed_combinations_cache = {}

        # 全特徴量から生成された組み合わせのキャッシュキー（available_features = Noneの状態）
        # _get_settings_hashは既にavailable_featuresを除いたハッシュを返すため、そのまま使用
        base_cache_key = (total_patterns, settings_hash)

        # 全特徴量から生成された組み合わせがキャッシュにない場合は、ファイルから読み込むか計算する
        if base_cache_key not in self._precomputed_combinations_cache:
            # ファイルから読み込みを試行
            from .precomputed_combinations import PrecomputedCombinationsManager
            file_manager = PrecomputedCombinationsManager(self)
            base_combinations = file_manager.load_from_file(total_patterns)

            if base_combinations is None:
                # ファイルが存在しない場合: 計算を実行（両方とも統一）
                original_available_features = self.available_features
                self.available_features = None
                base_combinations = file_manager.compute_all_combinations(total_patterns)
                self.available_features = original_available_features

                # 計算結果をファイルに保存（両方とも統一：視覚的に見れるファイルとして保存）
                # 次回実行時にファイルから読み込めるため、推論パイプラインでも保存する
                file_manager.save_to_file(total_patterns, base_combinations)

            self._precomputed_combinations_cache[base_cache_key] = base_combinations

        # 全特徴量から生成された組み合わせを取得
        base_combinations = self._precomputed_combinations_cache[base_cache_key]

        # available_featuresが指定されている場合、フィルタリング
        if self.available_features is not None:
            available_features_set = set(self.available_features)
            # キャッシュの中から、available_featuresに含まれる特徴量だけが使われている組み合わせをフィルタリング
            filtered_combinations = {}
            for pattern_idx, combo in base_combinations.items():
                features = combo.get('features', [])
                # 組み合わせの特徴量がすべてavailable_featuresに含まれているかチェック
                if all(f in available_features_set for f in features):
                    filtered_combinations[pattern_idx] = combo

            # フィルタリング後の組み合わせをpattern_idxに再割り当て（循環的に選択）
            if filtered_combinations:
                filtered_list = list(filtered_combinations.values())
                result = {}
                for pattern_idx in range(total_patterns):
                    combo_idx = pattern_idx % len(filtered_list)
                    result[pattern_idx] = {
                        'features': filtered_list[combo_idx]['features'].copy()
                    }
                return result
            else:
                # フィルタリング後の組み合わせが空の場合は空の辞書を返す
                return {}

        # available_featuresがNoneの場合は、そのまま返す
        if cache_key in self._precomputed_combinations_cache:
            return self._precomputed_combinations_cache[cache_key]

        # 事前計算を実行（available_features = Noneの場合のみ）
        # 事前計算ロジックはPrecomputedCombinationsManagerに移動済み
        from .precomputed_combinations import PrecomputedCombinationsManager
        file_manager = PrecomputedCombinationsManager(self)
        combinations = file_manager.compute_all_combinations(total_patterns)
        self._precomputed_combinations_cache[cache_key] = combinations
        return combinations

    def clear_precomputed_combinations_cache(self):
        """事前計算された組み合わせのキャッシュをクリア"""
        self._precomputed_combinations_cache = None
