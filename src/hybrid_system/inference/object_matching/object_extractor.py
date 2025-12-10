"""
オブジェクト抽出モジュール

タスク内のすべての入力グリッドと出力グリッドからオブジェクトを抽出
"""

from typing import List, Dict, Any
import numpy as np
import cv2

# cv2.ximgprocが利用可能かチェック
try:
    _has_ximgproc = hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')
except AttributeError:
    _has_ximgproc = False

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType
from .data_structures import ObjectInfo
from .similarity_utils import calculate_symmetry_score, calculate_hu_moment, get_pixels_set


class ObjectExtractor:
    """オブジェクト抽出器（IntegratedObjectExtractorを使用）"""

    def __init__(self):
        self.extractor = IntegratedObjectExtractor(ExtractionConfig())

    def extract_all_objects(self, task: Any) -> Dict[int, Dict[str, List[List[ObjectInfo]]]]:
        """
        タスク内のすべての入力グリッドと出力グリッドからオブジェクトを抽出

        Args:
            task: Taskオブジェクト

        Returns:
            {
                connectivity: {
                    'input_grids': [
                        [ObjectInfo, ObjectInfo, ...],  # 各入力グリッドのオブジェクトリスト
                        ...
                    ],
                    'output_grids': [
                        [ObjectInfo, ObjectInfo, ...],  # 各出力グリッドのオブジェクトリスト（訓練ペアのみ）
                        ...
                    ]
                },
                ...
            }
        """
        result = {}

        # すべての入力グリッドを取得（訓練+テスト）
        all_input_grids = task.get_all_inputs()
        train_output_grids = task.get_train_outputs()
        num_train_inputs = len(task.get_train_inputs())

        # 4連結と8連結で抽出
        for connectivity in [4, 8]:
            input_objects_list = []
            output_objects_list = []

            # 入力グリッドからオブジェクトを抽出
            for grid_index, input_grid in enumerate(all_input_grids):
                input_grid_np = np.array(input_grid)
                # 訓練入力グリッドかどうかを判定
                is_train = grid_index < num_train_inputs
                objects = self._extract_objects_from_grid(
                    input_grid_np, connectivity, grid_index, is_train=is_train
                )
                input_objects_list.append(objects)

            # 訓練ペアの出力グリッドからオブジェクトを抽出
            for grid_index, output_grid in enumerate(train_output_grids):
                output_grid_np = np.array(output_grid)
                objects = self._extract_objects_from_grid(
                    output_grid_np, connectivity, grid_index, is_train=True
                )
                output_objects_list.append(objects)

            result[connectivity] = {
                'input_grids': input_objects_list,
                'output_grids': output_objects_list
            }

        return result

    def _extract_objects_from_grid(
        self, grid: np.ndarray, connectivity: int, grid_index: int, is_train: bool
    ) -> List[ObjectInfo]:
        """
        グリッドからオブジェクトを抽出（IntegratedObjectExtractorを使用）

        Args:
            grid: グリッド（numpy配列）
            connectivity: 連結性（4または8）
            grid_index: グリッドのインデックス
            is_train: 訓練ペアかどうか

        Returns:
            オブジェクト情報のリスト
        """
        try:
            # IntegratedObjectExtractorを使用してオブジェクトを抽出
            result = self.extractor.extract_objects_by_type(grid, input_image_index=grid_index)

            if not result.success:
                return []

            # 連結性に応じてオブジェクトを取得
            if connectivity == 4:
                extracted_objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
            elif connectivity == 8:
                extracted_objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_8WAY, [])
            else:
                return []

            # ObjectInfoに変換
            objects = []
            for obj_index, obj in enumerate(extracted_objects):
                obj_info = ObjectInfo.from_object(
                    obj, grid_index, is_train, index=obj_index
                )
                # ベース特徴量ベクトルを作成
                obj_info.base_feature_vector = self._create_base_feature_vector(obj_info)
                objects.append(obj_info)

            return objects

        except Exception as e:
            # エラーが発生した場合は空のリストを返す
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"オブジェクト抽出エラー (grid_index={grid_index}, connectivity={connectivity}): {e}")
            return []

    def _create_base_feature_vector(self, obj: ObjectInfo) -> np.ndarray:
        """
        ベース特徴量ベクトルを作成（色、形状、位置のみ、重み付けなし）

        57次元: 色(10) + 形状(9) + 位置(2) + パッチハッシュ3×3(8) + パッチハッシュ2×2(4) +
                ダウンサンプリング4×4(16) + 境界方向ヒストグラム(4) + スケルトン化特徴(2) + 形状中心位置(2)
        形状特徴量: size, aspect_ratio, width, height, hole, perimeter_area_ratio, hu_moment, symmetry, density

        Args:
            obj: ObjectInfoオブジェクト

        Returns:
            ベース特徴量ベクトル（numpy配列、57次元）
        """
        feature_vector = []

        center_y, center_x = obj.center if obj.center else (0.0, 0.0)
        hole_count = obj.hole_count if obj.hole_count is not None else 0

        # 対称性スコアを計算（形状特徴量で使用）
        symmetry_x = calculate_symmetry_score(obj, 'X')
        symmetry_y = calculate_symmetry_score(obj, 'Y')
        symmetry_avg = float((symmetry_x + symmetry_y) / 2.0 if symmetry_x is not None and symmetry_y is not None else 0.0)

        # 周囲長/面積比を計算
        perimeter = self._calculate_perimeter(obj)
        if obj.size > 0:
            perimeter_area_ratio = perimeter / obj.size
        else:
            perimeter_area_ratio = 0.0
        # 正規化（最大値で割る、例: 最大4.0と仮定）
        # 実際の最大値は、完全に分離したピクセルの場合: 4.0（各ピクセルが4つの辺を持つ）
        perimeter_area_ratio_normalized = float(min(perimeter_area_ratio / 4.0, 1.0))  # 0.0-1.0に正規化

        # Huモーメントを計算
        hu_moment_normalized = float(calculate_hu_moment(obj))

        # 密度（density）を計算: オブジェクトのサイズ / バウンディングボックスの面積
        if obj.width > 0 and obj.height > 0:
            density = obj.size / (obj.width * obj.height)
        else:
            density = 0.0
        # 正規化（最大1.0、完全にコンパクトな場合）
        density_normalized = float(min(density, 1.0))

        # 1. 色の特徴量（10次元: 0-9の各色、ワンホットエンコーディング）
        color_onehot = [0.0] * 10
        if 0 <= obj.color <= 9:
            color_onehot[obj.color] = 1.0
        feature_vector.extend(color_onehot)

        # 2. 形状の特徴量（9次元: size, aspect_ratio, width, height, hole, perimeter_area_ratio, hu_moment, symmetry, density）
        feature_vector.append(min(obj.size / 1000.0, 1.0))  # size
        feature_vector.append(min(obj.aspect_ratio / 10.0, 1.0))  # aspect_ratio
        feature_vector.append(min(obj.width / 100.0, 1.0))  # width
        feature_vector.append(min(obj.height / 100.0, 1.0))  # height
        feature_vector.append(min(hole_count / 10.0, 1.0))  # hole
        feature_vector.append(perimeter_area_ratio_normalized)  # perimeter_area_ratio
        feature_vector.append(hu_moment_normalized)  # hu_moment
        feature_vector.append(symmetry_avg)  # symmetry
        feature_vector.append(density_normalized)  # density

        # 3. 位置の特徴量（2次元: center_x, center_y）
        feature_vector.append(min(center_x / 100.0, 1.0))  # center_x
        feature_vector.append(min(center_y / 100.0, 1.0))  # center_y

        # 4. パッチハッシュ（3×3、8次元）
        patch_hash_3x3 = self._calculate_patch_hash_vector(obj, patch_size=3, output_dim=8)
        feature_vector.extend(patch_hash_3x3.tolist())

        # 5. パッチハッシュ（2×2、4次元）
        patch_hash_2x2 = self._calculate_patch_hash_vector(obj, patch_size=2, output_dim=4)
        feature_vector.extend(patch_hash_2x2.tolist())

        # 6. ダウンサンプリングビットマップ（4×4、16次元）
        downscaled_bitmap = self._calculate_downscaled_bitmap(obj, target_size=4)
        feature_vector.extend(downscaled_bitmap.tolist())

        # 7. 境界方向ヒストグラム（4次元）
        contour_direction = self._calculate_contour_direction_histogram(obj)
        feature_vector.extend(contour_direction.tolist())

        # 8. スケルトン化特徴（2次元）
        skeleton_features = self._calculate_skeleton_features(obj)
        feature_vector.extend(skeleton_features.tolist())

        # 9. 形状中心位置（2次元）
        local_centroid = self._calculate_local_centroid(obj)
        feature_vector.extend(local_centroid.tolist())

        # numpy配列に変換して正規化
        feature_vector = np.array(feature_vector, dtype=np.float32)
        feature_vector = np.clip(feature_vector, 0.0, 1.0)

        return feature_vector

    def _calculate_perimeter(self, obj: ObjectInfo) -> int:
        """
        オブジェクトの周囲長を計算（外側の輪郭 + 内側の穴の輪郭）
        OpenCVのfindContoursとarcLengthを使用

        Args:
            obj: オブジェクト情報

        Returns:
            周囲長（ピクセル単位）
        """
        if not obj.pixels:
            return 0

        # バウンディングボックス内の画像を作成
        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        if width <= 0 or height <= 0:
            return 0

        # グリッドを作成（オブジェクト部分を255、背景を0）
        grid = np.zeros((height, width), dtype=np.uint8)
        for y, x in obj.pixels:  # ObjectInfoのpixelsは(y, x)形式
            # 範囲チェック：bboxの範囲内のピクセルのみを処理
            rel_y = y - min_y
            rel_x = x - min_x
            if 0 <= rel_y < height and 0 <= rel_x < width:
                grid[rel_y, rel_x] = 255

        # OpenCVのfindContoursを使用して輪郭を検出
        # RETR_CCOMP: 2階層の階層構造を取得（外側輪郭と内側輪郭）
        contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return 0

        total_perimeter = 0
        if hierarchy is not None:
            # 階層構造を使用して、外側輪郭と内側輪郭（穴）を処理
            for i, contour in enumerate(contours):
                # 輪郭の周囲長を計算
                perimeter = cv2.arcLength(contour, closed=True)
                total_perimeter += int(perimeter)
        else:
            # 階層情報がない場合（通常は発生しない）、すべての輪郭を処理
            for contour in contours:
                perimeter = cv2.arcLength(contour, closed=True)
                total_perimeter += int(perimeter)

        return total_perimeter

    def _calculate_patch_hash_vector(self, obj: ObjectInfo, patch_size: int, output_dim: int) -> np.ndarray:
        """
        パッチハッシュベクトルを計算（LSH化）

        Args:
            obj: オブジェクト情報
            patch_size: パッチサイズ（2または3）
            output_dim: 出力次元数（2×2の場合は4、3×3の場合は8）

        Returns:
            パッチハッシュベクトル（0.0-1.0に正規化）
        """
        if not obj.pixels or obj.size == 0:
            return np.zeros(output_dim, dtype=np.float32)

        pixel_set = get_pixels_set(obj)
        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]

        # ハッシュ値のカウンター
        hash_counter = np.zeros(output_dim, dtype=np.float32)

        # スライディングウィンドウでパッチを抽出
        offset = patch_size // 2
        for center_y in range(min_y, max_y + 1):
            for center_x in range(min_x, max_x + 1):
                # パッチの中心がオブジェクトのピクセル上にあるかチェック
                if (center_y, center_x) not in pixel_set:
                    continue

                # パッチの形状パターンを計算
                shape_pattern = 0
                bit_index = 0
                for dy in range(-offset, offset + 1):
                    for dx in range(-offset, offset + 1):
                        patch_y = center_y + dy
                        patch_x = center_x + dx
                        if (patch_y, patch_x) in pixel_set:
                            shape_pattern |= (1 << bit_index)
                        bit_index += 1

                # 色情報を除外し、形状パターンのみを使用（色変化に頑健にするため）
                # 元の実装: combined_hash = (shape_pattern << 4) | color_value
                # 変更後: 形状パターンのみを使用
                combined_hash = shape_pattern

                # LSH化: ハッシュ値をoutput_dimで割った余りをインデックスとして使用
                vector_index = combined_hash % output_dim
                hash_counter[vector_index] += 1.0

        # 正規化（L1正規化）
        if hash_counter.sum() > 0:
            hash_counter = hash_counter / hash_counter.sum()

        return hash_counter

    def _calculate_downscaled_bitmap(self, obj: ObjectInfo, target_size: int) -> np.ndarray:
        """
        ダウンサンプリングビットマップを計算
        OpenCVのresizeを使用

        Args:
            obj: オブジェクト情報
            target_size: ターゲットサイズ（4×4の場合は4）

        Returns:
            ダウンサンプリングビットマップ（target_size × target_size次元、0.0-1.0に正規化）
        """
        if not obj.pixels or obj.size == 0:
            return np.zeros(target_size * target_size, dtype=np.float32)

        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        if width <= 0 or height <= 0:
            return np.zeros(target_size * target_size, dtype=np.float32)

        # グリッドを作成（オブジェクト部分を1.0、背景を0.0）
        grid = np.zeros((height, width), dtype=np.float32)
        for y, x in obj.pixels:  # ObjectInfoのpixelsは(y, x)形式
            # 範囲チェック：bboxの範囲内のピクセルのみを処理
            rel_y = y - min_y
            rel_x = x - min_x
            if 0 <= rel_y < height and 0 <= rel_x < width:
                grid[rel_y, rel_x] = 1.0

        # OpenCVのresizeを使用してダウンサンプリング
        # INTER_AREA: 面積ベースの補間（ダウンサンプリングに最適）
        downscaled = cv2.resize(grid, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 0.0-1.0にクリップ（既に正規化されているが念のため）
        downscaled = np.clip(downscaled, 0.0, 1.0)

        return downscaled.flatten()

    def _calculate_contour_direction_histogram(self, obj: ObjectInfo) -> np.ndarray:
        """
        境界方向ヒストグラムを計算

        Args:
            obj: オブジェクト情報

        Returns:
            境界方向ヒストグラム（4次元: 上、下、左、右）
        """
        if not obj.pixels or obj.size == 0:
            return np.zeros(4, dtype=np.float32)

        pixel_set = get_pixels_set(obj)
        directions = [0.0, 0.0, 0.0, 0.0]  # 上、下、左、右

        for y, x in obj.pixels:
            # 4方向の隣接ピクセルをチェック
            # 上向きエッジ: 上に隣接ピクセルがない
            if (y - 1, x) not in pixel_set:
                directions[0] += 1.0
            # 下向きエッジ: 下に隣接ピクセルがない
            if (y + 1, x) not in pixel_set:
                directions[1] += 1.0
            # 左向きエッジ: 左に隣接ピクセルがない
            if (y, x - 1) not in pixel_set:
                directions[2] += 1.0
            # 右向きエッジ: 右に隣接ピクセルがない
            if (y, x + 1) not in pixel_set:
                directions[3] += 1.0

        # 正規化（L1正規化）
        total = sum(directions)
        if total > 0:
            directions = [d / total for d in directions]

        return np.array(directions, dtype=np.float32)

    def _calculate_skeleton_features(self, obj: ObjectInfo) -> np.ndarray:
        """
        スケルトン化特徴を計算

        Args:
            obj: オブジェクト情報

        Returns:
            スケルトン特徴（2次元: skeleton_length/size, junction_count）
        """
        if not obj.pixels or obj.size == 0:
            return np.zeros(2, dtype=np.float32)

        # バウンディングボックス内の画像を作成
        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        if width <= 0 or height <= 0:
            return np.zeros(2, dtype=np.float32)

        # グリッドを作成
        grid = np.zeros((height, width), dtype=np.uint8)
        for y, x in obj.pixels:  # ObjectInfoのpixelsは(y, x)形式
            # 範囲チェック：bboxの範囲内のピクセルのみを処理
            rel_y = y - min_y
            rel_x = x - min_x
            if 0 <= rel_y < height and 0 <= rel_x < width:
                grid[rel_y, rel_x] = 255

        # スケルトン化（Zhang-Suenアルゴリズム）- OpenCVのthinning関数を使用
        # cv2.ximgprocが利用できない場合は、簡易的なスケルトン化を使用
        if _has_ximgproc:
            try:
                skeleton = cv2.ximgproc.thinning(grid, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            except (AttributeError, Exception):
                # フォールバック: モルフォロジー演算を使用した簡易スケルトン化
                skeleton = self._simple_skeletonize(grid)
        else:
            # フォールバック: モルフォロジー演算を使用した簡易スケルトン化
            skeleton = self._simple_skeletonize(grid)

        # スケルトンの長さ（ピクセル数）
        skeleton_length = np.sum(skeleton > 0)

        # ジャンクション（分岐点）の数を計算
        # 各ピクセルの8近傍をチェックし、接続数が3以上の場合をジャンクションとする
        # グリッドが小さすぎる場合はスキップ（8近傍アクセスに必要な最小サイズは3×3）
        junction_count = 0
        if height >= 3 and width >= 3:
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if skeleton[y, x] > 0:
                        # 8近傍の接続数をカウント
                        neighbors = 0
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                # 境界チェック
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    if skeleton[ny, nx] > 0:
                                        neighbors += 1
                        if neighbors >= 3:
                            junction_count += 1

        # 正規化
        skeleton_ratio = skeleton_length / obj.size if obj.size > 0 else 0.0
        junction_normalized = min(junction_count / 10.0, 1.0)  # 最大10個で正規化

        return np.array([skeleton_ratio, junction_normalized], dtype=np.float32)

    def _calculate_local_centroid(self, obj: ObjectInfo) -> np.ndarray:
        """
        形状中心位置（内部重心、BB内で正規化）を計算

        Args:
            obj: オブジェクト情報

        Returns:
            内部重心（2次元: x, y、BB内で正規化、0.0-1.0）
        """
        if not obj.pixels or obj.size == 0:
            return np.zeros(2, dtype=np.float32)

        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        if width <= 0 or height <= 0:
            return np.zeros(2, dtype=np.float32)

        # 重心を計算
        sum_x = 0.0
        sum_y = 0.0
        for y, x in obj.pixels:
            sum_x += x
            sum_y += y

        centroid_x = sum_x / obj.size
        centroid_y = sum_y / obj.size

        # BB内で正規化（0.0-1.0）
        local_x = (centroid_x - min_x) / width if width > 0 else 0.0
        local_y = (centroid_y - min_y) / height if height > 0 else 0.0

        return np.array([min(max(local_x, 0.0), 1.0), min(max(local_y, 0.0), 1.0)], dtype=np.float32)

    def _simple_skeletonize(self, grid: np.ndarray) -> np.ndarray:
        """
        簡易的なスケルトン化（cv2.ximgprocが利用できない場合のフォールバック）

        Args:
            grid: バイナリ画像（uint8, 0または255）

        Returns:
            スケルトン化された画像
        """
        if grid.size == 0:
            return grid.copy()

        skeleton = grid.copy()
        kernel = np.ones((3, 3), np.uint8)

        # モルフォロジー演算を使用した簡易スケルトン化
        # エロージョンと元画像の差分を繰り返し適用
        prev_skeleton = None
        while True:
            eroded = cv2.erode(skeleton, kernel, iterations=1)
            temp = cv2.dilate(eroded, kernel, iterations=1)
            skeleton = cv2.subtract(skeleton, temp)

            # 収束チェック
            if prev_skeleton is not None and np.array_equal(skeleton, prev_skeleton):
                break
            prev_skeleton = skeleton.copy()

            # 無限ループ防止（最大100回）
            if np.sum(skeleton > 0) == 0:
                break

        return skeleton
