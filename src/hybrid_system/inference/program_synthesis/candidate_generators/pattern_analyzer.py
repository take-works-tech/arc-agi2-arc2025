"""
パターン分析器

変換パターンを分析する機能
"""

from typing import List, Dict, Any, Optional
from collections import Counter

from .common_helpers import get_grid_size, is_empty_grid


class PatternAnalyzer:
    """パターン分析器"""

    def __init__(self):
        """初期化"""
        pass

    def analyze_transformation_pattern(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """変換パターンを分析（Phase 4.2: パターン検出の信頼度向上）

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            検出されたパターンのリスト（信頼度付き）
        """
        patterns = []

        # 恒等変換パターン
        if input_grid == output_grid:
            patterns.append({
                'type': 'identity',
                'confidence': 1.0,
                'description': '恒等変換',
                'match_score': 1.0
            })

        # 色変更パターン
        if self._is_color_change_pattern(input_grid, output_grid):
            confidence = self._calculate_color_change_confidence(input_grid, output_grid, matching_result)
            patterns.append({
                'type': 'color_change',
                'confidence': confidence,
                'description': '色変更',
                'match_score': self._calculate_match_score(input_grid, output_grid, 'color_change')
            })

        # 移動パターン
        if self._is_movement_pattern(input_grid, output_grid):
            confidence = self._calculate_movement_confidence(input_grid, output_grid, matching_result)
            patterns.append({
                'type': 'movement',
                'confidence': confidence,
                'description': '移動',
                'match_score': self._calculate_match_score(input_grid, output_grid, 'movement')
            })

        # Phase 3.3: 新しいパターン検出

        # 反射パターン（水平）
        if self._is_horizontal_flip_pattern(input_grid, output_grid):
            confidence = self._calculate_flip_confidence(input_grid, output_grid, 'horizontal', matching_result)
            patterns.append({
                'type': 'flip_horizontal',
                'confidence': confidence,
                'description': '水平反射',
                'match_score': 1.0  # 完全一致
            })

        # 反射パターン（垂直）
        if self._is_vertical_flip_pattern(input_grid, output_grid):
            confidence = self._calculate_flip_confidence(input_grid, output_grid, 'vertical', matching_result)
            patterns.append({
                'type': 'flip_vertical',
                'confidence': confidence,
                'description': '垂直反射',
                'match_score': 1.0  # 完全一致
            })

        # スケールパターン
        scale_pattern = self._detect_scale_pattern(input_grid, output_grid)
        if scale_pattern:
            confidence = self._calculate_scale_confidence(input_grid, output_grid, scale_pattern, matching_result)
            scale_pattern['confidence'] = confidence
            scale_pattern['match_score'] = self._calculate_match_score(input_grid, output_grid, scale_pattern['type'])
            patterns.append(scale_pattern)

        # 色マッピングパターン
        if self._has_color_mapping_pattern(input_grid, output_grid):
            confidence = self._calculate_color_mapping_confidence(input_grid, output_grid, matching_result)
            patterns.append({
                'type': 'color_mapping',
                'confidence': confidence,
                'description': '色マッピング',
                'match_score': self._calculate_match_score(input_grid, output_grid, 'color_mapping')
            })

        # オブジェクト統合パターン
        if self._has_merge_pattern_analyzer(input_grid, output_grid):
            confidence = self._calculate_merge_confidence(input_grid, output_grid, matching_result)
            patterns.append({
                'type': 'merge',
                'confidence': confidence,
                'description': 'オブジェクト統合',
                'match_score': self._calculate_match_score(input_grid, output_grid, 'merge')
            })

        # オブジェクト分割パターン
        if self._has_split_pattern_analyzer(input_grid, output_grid):
            confidence = self._calculate_split_confidence(input_grid, output_grid, matching_result)
            patterns.append({
                'type': 'split',
                'confidence': confidence,
                'description': 'オブジェクト分割',
                'match_score': self._calculate_match_score(input_grid, output_grid, 'split')
            })

        # 信頼度でソート（高い順）
        patterns.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)

        return patterns

    def _is_color_change_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """色変更パターンかチェック（共通実装）"""

        if is_empty_grid(input_grid) or is_empty_grid(output_grid):
            return False

        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)

        if input_h != output_h or input_w != output_w:
            return False

        # 構造は同じで色だけが異なるかチェック
        for i in range(input_h):
            for j in range(input_w):
                if (input_grid[i][j] == 0) != (output_grid[i][j] == 0):
                    return False

        return True

    def _is_movement_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """移動パターンかチェック（本格実装）"""

        if is_empty_grid(input_grid) or is_empty_grid(output_grid):
            return False

        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)

        # サイズが異なる場合は移動パターンではない
        if input_h != output_h or input_w != output_w:
            return False

        # 非ゼロピクセルの位置を比較
        input_positions = {}
        output_positions = {}

        for i in range(input_h):
            for j in range(input_w):
                if input_grid[i][j] != 0:
                    color = input_grid[i][j]
                    if color not in input_positions:
                        input_positions[color] = []
                    input_positions[color].append((i, j))
                if output_grid[i][j] != 0:
                    color = output_grid[i][j]
                    if color not in output_positions:
                        output_positions[color] = []
                    output_positions[color].append((i, j))

        # 色が一致するピクセルペアを探す
        movement_vectors = []
        for color in input_positions:
            if color not in output_positions:
                continue
            for in_pos in input_positions[color]:
                for out_pos in output_positions[color]:
                    dx = out_pos[1] - in_pos[1]
                    dy = out_pos[0] - in_pos[0]
                    if dx != 0 or dy != 0:
                        movement_vectors.append((dx, dy))

        if not movement_vectors:
            return False

        # 移動ベクトルが一貫しているかチェック
        # 最も頻繁な移動ベクトルを取得
        movement_counter = Counter(movement_vectors)
        most_common_movement, count = movement_counter.most_common(1)[0]

        # 移動ベクトルの50%以上が同じ場合、移動パターンと判定
        if count >= len(movement_vectors) * 0.5:
            return True

        return False

    def _is_horizontal_flip_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """水平反射パターンかチェック"""
        if not input_grid or not output_grid:
            return False
        if not input_grid[0] or not output_grid[0]:
            return False
        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)
        if input_h != output_h or input_w != output_w:
            return False

        h, w = input_h, input_w
        # 水平反射: 各列が左右反転
        for i in range(h):
            for j in range(w):
                if input_grid[i][j] != output_grid[i][w - 1 - j]:
                    return False
        return True

    def _is_vertical_flip_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """垂直反射パターンかチェック"""
        if not input_grid or not output_grid:
            return False
        if not input_grid[0] or not output_grid[0]:
            return False
        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)
        if input_h != output_h or input_w != output_w:
            return False

        h, w = input_h, input_w
        # 垂直反射: 各行が上下反転
        for i in range(h):
            for j in range(w):
                if input_grid[i][j] != output_grid[h - 1 - i][j]:
                    return False
        return True

    def _detect_scale_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Optional[Dict[str, Any]]:
        """スケールパターンを検出

        Returns:
            スケールパターン辞書またはNone
        """
        if not input_grid or not output_grid:
            return None
        if not input_grid[0] or not output_grid[0]:
            return None

        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)

        if output_h == 0 or output_w == 0:
            return None

        # グリッドサイズの比率から推定
        h_ratio = input_h / output_h
        w_ratio = input_w / output_w

        # 拡大パターン（出力が大きい）
        if h_ratio < 0.8 and w_ratio < 0.8:
            scale_factor = int(round(1.0 / max(h_ratio, w_ratio)))
            if scale_factor >= 2:
                return {
                    'type': 'scale',
                    'confidence': 0.8,
                    'description': f'スケール（{scale_factor}倍拡大）',
                    'scale_factor': scale_factor
                }

        # 縮小パターン（出力が小さい）
        if h_ratio > 1.2 and w_ratio > 1.2:
            scale_down_factor = int(round(max(h_ratio, w_ratio)))
            if scale_down_factor >= 2:
                return {
                    'type': 'scale_down',
                    'confidence': 0.8,
                    'description': f'スケールダウン（1/{scale_down_factor}倍縮小）',
                    'scale_down_factor': scale_down_factor
                }

        return None

    def _has_color_mapping_pattern(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """色マッピングパターンがあるかチェック"""
        mapping = self._detect_color_mapping_pattern_internal(input_grid, output_grid)
        return mapping is not None

    def _detect_color_mapping_pattern_internal(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Optional[Dict[str, int]]:
        """色マッピングパターンを検出（PatternAnalyzer用）

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            色マッピング辞書（{'from_color': int, 'to_color': int}）またはNone
        """
        if not input_grid or not output_grid:
            return None
        if not input_grid[0] or not output_grid[0]:
            return None
        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)
        if input_h != output_h or input_w != output_w:
            return None

        # 色の対応関係を集計
        color_mapping = {}
        h, w = input_h, input_w

        for i in range(h):
            for j in range(w):
                input_color = input_grid[i][j]
                output_color = output_grid[i][j]

                if input_color != output_color:
                    if input_color not in color_mapping:
                        color_mapping[input_color] = {}
                    if output_color not in color_mapping[input_color]:
                        color_mapping[input_color][output_color] = 0
                    color_mapping[input_color][output_color] += 1

        # 最も頻繁な色マッピングを選択
        if not color_mapping:
            return None

        best_from_color = None
        best_to_color = None
        best_count = 0

        for from_color, to_colors in color_mapping.items():
            for to_color, count in to_colors.items():
                if count > best_count:
                    best_count = count
                    best_from_color = from_color
                    best_to_color = to_color

        # 閾値以上のマッピングがある場合のみ返す
        if best_count >= h * w * 0.3:  # 30%以上のピクセルがマッピングされている
            return {'from_color': best_from_color, 'to_color': best_to_color}

        return None

    def _has_merge_pattern_analyzer(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """オブジェクト統合パターンかチェック（PatternAnalyzer用）"""
        if not input_grid or not output_grid:
            return False
        # 入力のオブジェクト数が出力より多い場合、統合の可能性
        input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
        output_nonzero = sum(1 for row in output_grid for c in row if c != 0)

        if input_nonzero > output_nonzero and output_nonzero > 0:
            return True

        return False

    def _has_split_pattern_analyzer(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """オブジェクト分割パターンかチェック（PatternAnalyzer用）"""
        if not input_grid or not output_grid:
            return False
        # 入力のオブジェクト数が出力より少ない場合、分割の可能性
        input_nonzero = sum(1 for row in input_grid for c in row if c != 0)
        output_nonzero = sum(1 for row in output_grid for c in row if c != 0)

        if input_nonzero < output_nonzero and input_nonzero > 0:
            return True

        return False

    # Phase 4.2: パターン検出の信頼度計算メソッド

    def _calculate_match_score(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        pattern_type: str
    ) -> float:
        """パターンの一致度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            pattern_type: パターンタイプ

        Returns:
            一致度（0.0-1.0）
        """
        if pattern_type == 'identity':
            return 1.0 if input_grid == output_grid else 0.0

        if pattern_type in ['flip_horizontal', 'flip_vertical']:
            # 反射パターンは完全一致かどうか
            if pattern_type == 'flip_horizontal':
                return 1.0 if self._is_horizontal_flip_pattern(input_grid, output_grid) else 0.0
            else:
                return 1.0 if self._is_vertical_flip_pattern(input_grid, output_grid) else 0.0

        # その他のパターン: 詳細な一致度計算（本格実装）
        # 入出力グリッドの類似度を計算
        similarity_score = self._calculate_grid_similarity(input_grid, output_grid)
        return similarity_score

    def _calculate_grid_similarity(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> float:
        """グリッド間の類似度を計算（本格実装）

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            類似度スコア（0.0-1.0）
        """
        if is_empty_grid(input_grid) or is_empty_grid(output_grid):
            return 0.0

        input_h, input_w = get_grid_size(input_grid)
        output_h, output_w = get_grid_size(output_grid)

        # サイズが異なる場合、類似度を下げる
        size_penalty = 0.0
        if input_h != output_h or input_w != output_w:
            size_penalty = 0.3

        # ピクセルレベルの一致度を計算
        min_h = min(input_h, output_h)
        min_w = min(input_w, output_w)

        matching_pixels = 0
        total_pixels = min_h * min_w

        for i in range(min_h):
            for j in range(min_w):
                if input_grid[i][j] == output_grid[i][j]:
                    matching_pixels += 1

        pixel_similarity = matching_pixels / total_pixels if total_pixels > 0 else 0.0

        # 色の分布の類似度
        input_colors = Counter(c for row in input_grid for c in row if c != 0)
        output_colors = Counter(c for row in output_grid for c in row if c != 0)

        # 色の種類の一致度
        input_color_set = set(input_colors.keys())
        output_color_set = set(output_colors.keys())
        color_overlap = len(input_color_set & output_color_set)
        color_union = len(input_color_set | output_color_set)
        color_similarity = color_overlap / color_union if color_union > 0 else 0.0

        # 非ゼロピクセル数の類似度
        input_nonzero = sum(input_colors.values())
        output_nonzero = sum(output_colors.values())
        if input_nonzero == 0 and output_nonzero == 0:
            nonzero_similarity = 1.0
        elif input_nonzero == 0 or output_nonzero == 0:
            nonzero_similarity = 0.0
        else:
            nonzero_ratio = min(input_nonzero, output_nonzero) / max(input_nonzero, output_nonzero)
            nonzero_similarity = nonzero_ratio

        # 総合類似度（重み付き平均）
        total_similarity = (
            pixel_similarity * 0.5 +
            color_similarity * 0.3 +
            nonzero_similarity * 0.2
        ) - size_penalty

        return max(0.0, min(1.0, total_similarity))

    def _calculate_color_change_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """色変更パターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.8

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            categories = matching_result.get('categories', [])
            # 色変更が検出されたカテゴリの割合（個別オブジェクトのtransformation_patternから判定）
            # CategoryInfoにはrepresentative_transformation_pattern属性がないため、
            # 個別オブジェクトのtransformation_patternから集約する
            color_change_count = 0
            for cat in categories:
                for obj in cat.objects:
                    if obj.transformation_pattern and obj.transformation_pattern.get('color_change', False):
                        color_change_count += 1
                        break  # 1つのカテゴリで1回だけカウント
            if len(categories) > 0:
                color_change_ratio = color_change_count / len(categories)
                base_confidence = 0.7 + 0.3 * color_change_ratio

        return min(1.0, base_confidence)

    def _calculate_movement_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """移動パターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.7

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            categories = matching_result.get('categories', [])
            # 位置変更が検出されたカテゴリの割合（個別オブジェクトのtransformation_patternから判定）
            # CategoryInfoにはrepresentative_transformation_pattern属性がないため、
            # 個別オブジェクトのtransformation_patternから集約する
            position_change_count = 0
            for cat in categories:
                for obj in cat.objects:
                    if obj.transformation_pattern and obj.transformation_pattern.get('position_change', False):
                        position_change_count += 1
                        break  # 1つのカテゴリで1回だけカウント
            if len(categories) > 0:
                position_change_ratio = position_change_count / len(categories)
                base_confidence = 0.6 + 0.4 * position_change_ratio

        return min(1.0, base_confidence)

    def _calculate_flip_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        axis: str,
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """反射パターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            axis: 反射軸（'horizontal' or 'vertical'）
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        # 反射パターンは完全一致なので、基本的に高い信頼度
        base_confidence = 0.9

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            # 反射パターンが検出された場合、信頼度をさらに向上
            base_confidence = 0.95

        return min(1.0, base_confidence)

    def _calculate_scale_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        scale_pattern: Dict[str, Any],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """スケールパターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            scale_pattern: スケールパターン辞書
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.8

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            categories = matching_result.get('categories', [])
            # ピクセル形状変更が検出されたカテゴリの割合（個別オブジェクトのtransformation_patternから判定）
            # CategoryInfoにはrepresentative_transformation_pattern属性がないため、
            # 個別オブジェクトのtransformation_patternから集約する
            shape_change_count = 0
            for cat in categories:
                for obj in cat.objects:
                    if obj.transformation_pattern and obj.transformation_pattern.get('shape_change', False):
                        shape_change_count += 1
                        break  # 1つのカテゴリで1回だけカウント
            if len(categories) > 0:
                shape_change_ratio = shape_change_count / len(categories)
                base_confidence = 0.7 + 0.3 * shape_change_ratio

        return min(1.0, base_confidence)

    def _calculate_color_mapping_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """色マッピングパターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.8

        # 色マッピングの一致度を計算
        color_mapping = self._detect_color_mapping_pattern_internal(input_grid, output_grid)
        if color_mapping:
            h, w = get_grid_size(input_grid)
            # マッピングされたピクセルの割合を計算
            mapped_count = 0
            total_count = 0
            for i in range(h):
                for j in range(w):
                    if input_grid[i][j] == color_mapping['from_color']:
                        total_count += 1
                        if output_grid[i][j] == color_mapping['to_color']:
                            mapped_count += 1

            if total_count > 0:
                mapping_ratio = mapped_count / total_count
                base_confidence = 0.6 + 0.4 * mapping_ratio

        return min(1.0, base_confidence)

    def _calculate_merge_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """オブジェクト統合パターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.7

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            # 統合が検出された場合、信頼度を向上
            correspondences = matching_result.get('correspondences', [])
            merge_count = sum(
                1 for corr in correspondences
                if corr.get('correspondence_type') == 'many_to_one'
            )
            if len(correspondences) > 0:
                merge_ratio = merge_count / len(correspondences)
                base_confidence = 0.6 + 0.4 * merge_ratio

        return min(1.0, base_confidence)

    def _calculate_split_confidence(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        matching_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """オブジェクト分割パターンの信頼度を計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            信頼度（0.0-1.0）
        """
        base_confidence = 0.7

        # オブジェクトマッチング結果がある場合、信頼度を向上
        if matching_result and matching_result.get('success'):
            # 分割が検出された場合、信頼度を向上
            correspondences = matching_result.get('correspondences', [])
            split_count = sum(
                1 for corr in correspondences
                if corr.get('correspondence_type') == 'one_to_many'
            )
            if len(correspondences) > 0:
                split_ratio = split_count / len(correspondences)
                base_confidence = 0.6 + 0.4 * split_ratio

        return min(1.0, base_confidence)
