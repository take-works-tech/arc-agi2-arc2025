"""
類似度計算ユーティリティ（共通実装）

オブジェクト間の類似度計算の共通ロジックを提供
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

# ObjectInfoの前方参照のため
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .data_structures import ObjectInfo


def calculate_position_similarity(
    obj1: Union[Dict[str, Any], Any],
    obj2: Union[Dict[str, Any], Any]
) -> float:
    """位置の類似度を計算（共通実装）

    Args:
        obj1: オブジェクト1（DictまたはObjectInfo）
        obj2: オブジェクト2（DictまたはObjectInfo）

    Returns:
        位置類似度（0.0～1.0）
    """
    # 中心座標を取得
    if isinstance(obj1, dict):
        center1 = obj1.get('center', (0, 0))
    else:
        center1 = getattr(obj1, 'center', (0, 0))

    if isinstance(obj2, dict):
        center2 = obj2.get('center', (0, 0))
    else:
        center2 = getattr(obj2, 'center', (0, 0))

    # ユークリッド距離を計算
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # 最大距離を動的に計算（オブジェクトのバウンディングボックスから推定）
    bbox1 = obj1.get('bbox', None) if isinstance(obj1, dict) else getattr(obj1, 'bbox', None)
    bbox2 = obj2.get('bbox', None) if isinstance(obj2, dict) else getattr(obj2, 'bbox', None)

    if bbox1 and bbox2:
        # バウンディングボックスから最大距離を推定
        # bbox形式: (min_i, min_j, max_i, max_j)
        if len(bbox1) >= 4:
            height1 = bbox1[2] - bbox1[0] + 1  # max_i - min_i + 1
            width1 = bbox1[3] - bbox1[1] + 1   # max_j - min_j + 1
        else:
            width1 = height1 = 10
        if len(bbox2) >= 4:
            height2 = bbox2[2] - bbox2[0] + 1
            width2 = bbox2[3] - bbox2[1] + 1
        else:
            width2 = height2 = 10

        # グリッドサイズを推定（バウンディングボックスから）
        max_dimension = max(width1, height1, width2, height2)
        # 最大距離はグリッドの対角線長を考慮（sqrt(2) * max_dimension）
        max_distance = np.sqrt(2) * max(max_dimension * 2, 10.0)
    else:
        # フォールバック: オブジェクトサイズから推定
        size1 = obj1.get('size', 10) if isinstance(obj1, dict) else getattr(obj1, 'size', 10)
        size2 = obj2.get('size', 10) if isinstance(obj2, dict) else getattr(obj2, 'size', 10)
        max_size = max(size1, size2)
        # サイズからグリッドサイズを推定（sqrt(size) * 2）
        estimated_grid_size = int(np.sqrt(max_size)) * 2
        max_distance = np.sqrt(2) * max(estimated_grid_size, 10.0)

    # 指数減衰で類似度を計算
    similarity = np.exp(-distance / (max_distance / 2.0))
    return float(similarity)


def normalize_pixels(pixels: set) -> set:
    """ピクセル座標を重心基準に正規化（共通実装）

    Args:
        pixels: ピクセル座標のセット

    Returns:
        正規化後のピクセル座標のセット
    """
    if not pixels:
        return set()

    pixels_list = list(pixels)
    center_y = sum(p[0] for p in pixels_list) / len(pixels_list)
    center_x = sum(p[1] for p in pixels_list) / len(pixels_list)

    return {(int(y - center_y), int(x - center_x)) for y, x in pixels_list}


def calculate_shape_similarity(
    obj1: Union[Dict[str, Any], Any],
    obj2: Union[Dict[str, Any], Any]
) -> float:
    """形状の類似度を計算（共通実装）

    Args:
        obj1: オブジェクト1（DictまたはObjectInfo）
        obj2: オブジェクト2（DictまたはObjectInfo）

    Returns:
        形状類似度（0.0～1.0）
    """
    # サイズを取得
    if isinstance(obj1, dict):
        size1 = obj1.get('size', 0)
        width1 = obj1.get('width', 1)
        height1 = obj1.get('height', 1)
        pixels1 = set(obj1.get('pixels', []))
    else:
        size1 = getattr(obj1, 'size', 0)
        width1 = getattr(obj1, 'width', 1)
        height1 = getattr(obj1, 'height', 1)
        pixels1 = set(getattr(obj1, 'pixels', []))

    if isinstance(obj2, dict):
        size2 = obj2.get('size', 0)
        width2 = obj2.get('width', 1)
        height2 = obj2.get('height', 1)
        pixels2 = set(obj2.get('pixels', []))
    else:
        size2 = getattr(obj2, 'size', 0)
        width2 = getattr(obj2, 'width', 1)
        height2 = getattr(obj2, 'height', 1)
        pixels2 = set(getattr(obj2, 'pixels', []))

    if size1 == 0 and size2 == 0:
        return 1.0

    if size1 == 0 or size2 == 0:
        return 0.0

    # 1. サイズ類似度
    size_ratio = min(size1, size2) / max(size1, size2)

    # 2. アスペクト比類似度
    aspect1 = width1 / height1 if height1 > 0 else 1.0
    aspect2 = width2 / height2 if height2 > 0 else 1.0

    aspect_similarity = min(aspect1, aspect2) / max(aspect1, aspect2) if max(aspect1, aspect2) > 0 else 0.0

    # 3. ピクセル配置類似度（Jaccard係数）
    if pixels1 and pixels2:
        # 重心を合わせて正規化
        pixels1_norm = normalize_pixels(pixels1)
        pixels2_norm = normalize_pixels(pixels2)

        intersection = len(pixels1_norm & pixels2_norm)
        union = len(pixels1_norm | pixels2_norm)
        pixel_similarity = intersection / union if union > 0 else 0.0
    else:
        pixel_similarity = 0.0

    # 総合類似度（重み付き平均）
    total_similarity = 0.25 * size_ratio + 0.25 * aspect_similarity + 0.5 * pixel_similarity

    return float(total_similarity)


def calculate_symmetry_score(obj: 'ObjectInfo', axis: str) -> float:
    """
    オブジェクトの対称性スコアを計算（共通実装、キャッシュ付き）

    Args:
        obj: ObjectInfoオブジェクト
        axis: 対称軸（'X'または'Y'）

    Returns:
        対称性スコア（0.0-1.0）

    注意:
        - ObjectInfoの_symmetry_score_cacheを使用してキャッシュします
        - ObjectInfoのpixelsは(y, x)形式を前提としています
    """
    if not hasattr(obj, 'pixels') or not obj.pixels or len(obj.pixels) == 0:
        return 0.0

    # キャッシュをチェック
    if obj._symmetry_score_cache is None:
        obj._symmetry_score_cache = {}

    if axis.upper() in obj._symmetry_score_cache:
        return obj._symmetry_score_cache[axis.upper()]

    try:
        # ObjectInfoのpixelsは(y, x)形式
        y_coords = [p[0] for p in obj.pixels]
        x_coords = [p[1] for p in obj.pixels]

        # ピクセルセットを取得（キャッシュ付き）
        from .data_structures import ObjectInfo
        pixel_set = get_pixels_set(obj)

        if axis.upper() == "X":
            # X軸対称性
            center_x = (min(x_coords) + max(x_coords)) / 2
            symmetric_pixels = 0
            for pixel in obj.pixels:
                y, x = pixel[0], pixel[1]
                mirror_x = int(2 * center_x - x)
                if (y, mirror_x) in pixel_set:
                    symmetric_pixels += 1
            result = symmetric_pixels / len(obj.pixels) if len(obj.pixels) > 0 else 0.0
        elif axis.upper() == "Y":
            # Y軸対称性
            center_y = (min(y_coords) + max(y_coords)) / 2
            symmetric_pixels = 0
            for pixel in obj.pixels:
                y, x = pixel[0], pixel[1]
                mirror_y = int(2 * center_y - y)
                if (mirror_y, x) in pixel_set:
                    symmetric_pixels += 1
            result = symmetric_pixels / len(obj.pixels) if len(obj.pixels) > 0 else 0.0
        else:
            result = 0.0

        # キャッシュに保存
        obj._symmetry_score_cache[axis.upper()] = result
        return result
    except Exception:
        return 0.0


def calculate_hu_moment(obj: 'ObjectInfo') -> float:
    """
    Huモーメントを計算（共通実装、キャッシュ付き）

    Args:
        obj: ObjectInfoオブジェクト

    Returns:
        Huモーメント特徴量（0.0-1.0に正規化）

    注意:
        - ObjectInfoの_hu_moment_cacheを使用してキャッシュします
    """
    # キャッシュをチェック
    if obj._hu_moment_cache is not None:
        return obj._hu_moment_cache

    try:
        import cv2
    except ImportError:
        # OpenCVが利用できない場合、0.0を返す
        obj._hu_moment_cache = 0.0
        return 0.0

    if not obj.pixels or obj.size == 0:
        obj._hu_moment_cache = 0.0
        return 0.0

    try:
        # バウンディングボックス内の画像を作成
        min_y, min_x = obj.bbox[0], obj.bbox[1]
        max_y, max_x = obj.bbox[2], obj.bbox[3]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        if width <= 0 or height <= 0:
            obj._hu_moment_cache = 0.0
            return 0.0

        # グリッドを作成
        grid = np.zeros((height, width), dtype=np.uint8)
        for y, x in obj.pixels:
            grid[y - min_y, x - min_x] = 255

        # モーメントを計算
        moments = cv2.moments(grid)

        # Huモーメントを計算
        hu_moments = cv2.HuMoments(moments)

        # 1次元特徴量として使用（最初のHuモーメントの対数変換）
        # Huモーメントは非常に小さい値になるため、対数変換して正規化
        hu_moment_0 = float(hu_moments[0])
        if hu_moment_0 > 0:
            hu_feature = -np.sign(hu_moment_0) * np.log10(np.abs(hu_moment_0))
            # 正規化（例: -10から10の範囲を0.0-1.0に変換）
            hu_feature_normalized = (hu_feature + 10) / 20.0
            result = float(max(0.0, min(1.0, hu_feature_normalized)))
        else:
            result = 0.0

        # キャッシュに保存
        obj._hu_moment_cache = result
        return result
    except Exception:
        obj._hu_moment_cache = 0.0
        return 0.0


def get_pixels_set(obj: 'ObjectInfo') -> set:
    """
    オブジェクトのピクセルセットを取得（キャッシュ付き）

    Args:
        obj: ObjectInfoオブジェクト

    Returns:
        ピクセル座標のセット
    """
    # キャッシュをチェック
    if obj._pixels_set_cache is not None:
        return obj._pixels_set_cache

    # ピクセル集合を作成
    pixel_set = set(obj.pixels) if obj.pixels else set()

    # キャッシュに保存
    obj._pixels_set_cache = pixel_set
    return pixel_set
