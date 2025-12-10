"""
事前計算された特徴量組み合わせの管理

事前計算ロジックとファイル保存/読み込み機能を提供
プログラム生成フローで使用するためのファイル保存/読み込み機能を提供
本番推論パイプラインでは使用しない（メモリ内キャッシュのみ使用）
"""

from typing import Dict, Optional, List
from pathlib import Path
from itertools import combinations
import json
import random
import numpy as np


class PrecomputedCombinationsManager:
    """事前計算された組み合わせのファイル管理クラス"""

    def __init__(self, config):
        """
        初期化

        Args:
            config: ObjectMatchingConfigインスタンス
        """
        self.config = config

    def _get_file_path(self, total_patterns: int) -> Path:
        """事前計算された組み合わせのファイルパスを取得

        Args:
            total_patterns: 総パターン数

        Returns:
            ファイルパス
        """
        settings_hash = self.config._get_settings_hash()
        # プロジェクトルートを取得（precomputed_combinations.pyから4階層上）
        project_root = Path(__file__).parent.parent.parent.parent
        # 視覚的に見れるファイルとして、data/ディレクトリに保存（cache/ではなく）
        data_dir = project_root / 'data' / 'precomputed_combinations'
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"combinations_{total_patterns}_{settings_hash}.json"
        return data_dir / filename

    def load_from_file(self, total_patterns: int) -> Optional[Dict[int, Dict[str, list]]]:
        """ファイルから事前計算された組み合わせを読み込む

        Args:
            total_patterns: 総パターン数

        Returns:
            事前計算された組み合わせの辞書、またはNone（ファイルが存在しない場合）
        """
        file_path = self._get_file_path(total_patterns)
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSONから読み込んだデータを辞書に変換
                result = {}
                for pattern_idx_str, combo in data.items():
                    pattern_idx = int(pattern_idx_str)
                    result[pattern_idx] = {
                        'features': combo['features']
                    }
                return result
        except Exception as e:
            # ファイル読み込みエラーの場合はNoneを返す
            print(f"[WARNING] 事前計算ファイルの読み込みに失敗: {e}", flush=True)
            return None

    def save_to_file(self, total_patterns: int, combinations: Dict[int, Dict[str, list]]):
        """事前計算された組み合わせをファイルに保存（視覚的に見やすい形式）

        Args:
            total_patterns: 総パターン数
            combinations: 事前計算された組み合わせの辞書
        """
        file_path = self._get_file_path(total_patterns)
        try:
            # 視覚的に見やすい形式で保存
            # 1. JSON形式（インデント付き、ソート済み）
            data = {}
            for pattern_idx in sorted(combinations.keys()):
                combo = combinations[pattern_idx]
                data[str(pattern_idx)] = {
                    'features': sorted(combo['features']) if 'features' in combo else []
                }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)

            # 2. 人間が読みやすいテキスト形式も保存
            text_file_path = file_path.with_suffix('.txt')
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(f"事前計算された特徴量組み合わせ（総数: {total_patterns}個）\n")
                f.write(f"設定ハッシュ: {self.config._get_settings_hash()}\n")
                f.write("=" * 80 + "\n\n")

                for pattern_idx in sorted(combinations.keys()):
                    combo = combinations[pattern_idx]
                    features = sorted(combo['features']) if 'features' in combo else []
                    f.write(f"パターン {pattern_idx}:\n")
                    f.write(f"  特徴量: {', '.join(features)}\n")
                    f.write(f"  特徴量数: {len(features)}\n")
                    f.write("\n")

                # 統計情報
                f.write("=" * 80 + "\n")
                f.write("統計情報:\n")
                feature_counts = {}
                for combo in combinations.values():
                    features = combo.get('features', [])
                    for feature in features:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1

                f.write(f"  総組み合わせ数: {len(combinations)}\n")
                f.write(f"  使用されている特徴量: {len(feature_counts)}個\n")
                f.write("\n  特徴量の使用頻度:\n")
                for feature, count in sorted(feature_counts.items(), key=lambda x: (-x[1], x[0])):
                    percentage = (count / len(combinations)) * 100
                    f.write(f"    {feature}: {count}回 ({percentage:.1f}%)\n")

            print(f"[INFO] 事前計算ファイルを保存しました: {file_path}", flush=True)
            print(f"[INFO] テキスト形式も保存しました: {text_file_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] 事前計算ファイルの保存に失敗: {e}", flush=True)

    def compute_all_combinations(self, total_patterns: int) -> Dict[int, Dict[str, List[str]]]:
        """すべてのパターン組み合わせを事前計算（1段階構造）

        使用可能な特徴量リストから、すべての組み合わせ（空でないすべてのサブセット）を生成し、
        各pattern_idxに対して均等に循環選択で割り当てる
        グループ化が有効な場合、グループ単位で選択し、後で展開する

        Args:
            total_patterns: 総パターン数

        Returns:
            pattern_idxをキーとする辞書
            {
                pattern_idx: {
                    'features': List[str]  # 選択された特徴量のリスト（展開済み）
                }
            }
        """
        result = {}

        # 使用する特徴量のリストを決定
        base_features = (
            self.config.available_features
            if self.config.available_features is not None
            else self.config.all_available_features
        )

        # グループ化が有効な場合、グループ単位で組み合わせを生成
        if self.config.use_feature_groups and self.config.feature_groups:
            # グループ名と単独特徴量のリストを作成
            group_names = list(self.config.feature_groups.keys())
            # グループに含まれていない単独特徴量を取得
            grouped_features = set()
            for group_features in self.config.feature_groups.values():
                grouped_features.update(group_features)
            individual_features = [f for f in base_features if f not in grouped_features]

            # 選択可能な要素（グループ名 + 単独特徴量）
            selectable_items = group_names + individual_features

            # すべての組み合わせを生成（グループ名と単独特徴量の組み合わせ）
            # グループ名は展開せず、そのまま1つの特徴量として扱う（patch_hash_3x3と同様）
            # ただし、展開後の特徴量で重複チェックと排除推奨チェックを行う
            all_combinations_dict = {}  # キー: 展開後の特徴量（タプル）、値: 展開前の組み合わせ（リスト）
            for r in range(1, len(selectable_items) + 1):
                for combo in combinations(selectable_items, r):
                    combo_list = sorted(list(combo))

                    # 排除推奨の組み合わせをチェック（展開後の特徴量でチェック）
                    # グループ名を展開してチェック
                    expanded_for_check = []
                    for item in combo_list:
                        if item in self.config.feature_groups:
                            expanded_for_check.extend(self.config.feature_groups[item])
                        else:
                            expanded_for_check.append(item)
                    expanded_for_check = sorted(list(set(expanded_for_check)))

                    if not self._is_excluded_combination(expanded_for_check):
                        # 展開後の特徴量をキーとして、展開前の組み合わせを保存
                        # これにより、展開後に同じ結果になる組み合わせは1つだけ残る
                        expanded_key = tuple(expanded_for_check)
                        if expanded_key not in all_combinations_dict:
                            # グループ名は展開せず、そのまま保存
                            all_combinations_dict[expanded_key] = combo_list

            # 展開前の組み合わせのリストを取得
            all_combinations = list(all_combinations_dict.values())
        else:
            # グループ化が無効な場合、従来通り個別に組み合わせを生成
            all_combinations = []
            for r in range(1, len(base_features) + 1):
                for combo in combinations(base_features, r):
                    combo_list = sorted(list(combo))
                    # 排除推奨の組み合わせをチェック
                    if not self._is_excluded_combination(combo_list):
                        all_combinations.append(combo_list)

        # 組み合わせをソート（順序を安定化）
        all_combinations.sort(key=lambda x: (len(x), tuple(x)))

        # 各pattern_idxに対して、対応する組み合わせを割り当て（循環的に選択）
        for pattern_idx in range(total_patterns):
            if all_combinations:
                combo_idx = pattern_idx % len(all_combinations)
                result[pattern_idx] = {
                    'features': all_combinations[combo_idx].copy()
                }
            else:
                # 組み合わせが存在しない場合（通常は発生しない）
                result[pattern_idx] = {
                    'features': []
                }

        return result

    def _is_excluded_combination(self, features: List[str]) -> bool:
        """排除推奨の組み合わせかどうかをチェック

        Args:
            features: 特徴量のリスト

        Returns:
            排除推奨の組み合わせの場合True
        """
        if not self.config.excluded_combinations:
            return False

        features_set = set(features)
        for excluded in self.config.excluded_combinations:
            excluded_set = set(excluded)
            # 排除推奨の組み合わせのすべての要素が含まれている場合
            if excluded_set.issubset(features_set):
                return True

        return False

    def select_weighted_combination(
        self, total_patterns: int, all_combinations: List[List[str]]
    ) -> int:
        """重み付け選択で組み合わせを選択（プログラム生成フロー用）

        組み合わせの選ばれやすさ = 組み合わせに含まれる特徴量の重みの平均

        Args:
            total_patterns: 総パターン数（未使用だが、将来の拡張のため保持）
            all_combinations: すべての組み合わせのリスト

        Returns:
            選択された組み合わせのインデックス（pattern_idx相当）
        """
        if not all_combinations:
            return 0

        # 各組み合わせの重みを計算
        combination_weights = []
        for combo in all_combinations:
            # 組み合わせに含まれる特徴量の重みの平均を計算
            weights_in_combo = []
            for feature in combo:
                # グループ名の場合は、グループ内の特徴量の重みの平均を使用
                if feature in self.config.feature_groups:
                    group_features = self.config.feature_groups[feature]
                    group_weights = [
                        self.config.feature_weights.get(f, 0.5)
                        for f in group_features
                        if f in self.config.feature_weights
                    ]
                    if group_weights:
                        weights_in_combo.append(np.mean(group_weights))
                    else:
                        weights_in_combo.append(0.5)  # デフォルト重み
                else:
                    # 単独特徴量の場合は直接重みを取得
                    weight = self.config.feature_weights.get(feature, 0.5)
                    weights_in_combo.append(weight)

            # 組み合わせの重み = 含まれる特徴量の重みの平均
            if weights_in_combo:
                combo_weight = np.mean(weights_in_combo)
            else:
                combo_weight = 0.5  # デフォルト重み
            combination_weights.append(combo_weight)

        # 重みを正規化（確率分布に変換）
        weights_array = np.array(combination_weights)
        # 最小値を0.1に設定（すべての組み合わせに最低限の確率を保証）
        weights_array = np.maximum(weights_array, 0.1)
        probabilities = weights_array / np.sum(weights_array)

        # 重み付けランダム選択
        selected_idx = np.random.choice(len(all_combinations), p=probabilities)
        return selected_idx

    def get_weighted_combination_index(
        self, total_patterns: int, precomputed: Dict[int, Dict[str, List[str]]],
        excluded_indices: set = None
    ) -> int:
        """事前計算された組み合わせから重み付け選択でインデックスを取得

        Args:
            total_patterns: 総パターン数
            precomputed: 事前計算された組み合わせの辞書
            excluded_indices: 除外するpattern_idxのセット（デフォルトはNone）

        Returns:
            選択されたpattern_idx
        """
        if not precomputed:
            return 0

        # 除外するpattern_idxを考慮して、利用可能なpattern_idxを取得
        if excluded_indices is None:
            excluded_indices = set()

        # 利用可能なpattern_idxとその組み合わせを取得
        available_pattern_indices = [idx for idx in precomputed.keys() if idx not in excluded_indices]

        if not available_pattern_indices:
            # すべて試行済みの場合は、除外を無視して選択
            available_pattern_indices = list(precomputed.keys())

        # 利用可能な組み合わせのみを取得
        available_combinations = [precomputed[idx]['features'] for idx in available_pattern_indices]

        # 重み付け選択でインデックスを取得（利用可能な組み合わせのみから選択）
        selected_combo_idx = self.select_weighted_combination(total_patterns, available_combinations)

        # pattern_idxを取得（利用可能なpattern_idxから）
        if selected_combo_idx < len(available_pattern_indices):
            return available_pattern_indices[selected_combo_idx]
        else:
            # フォールバック: ランダム選択
            return random.choice(available_pattern_indices)
