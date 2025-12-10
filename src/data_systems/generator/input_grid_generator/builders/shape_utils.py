"""
形状ユーティリティ

オブジェクトの形状を比較・コピーする機能を提供
"""
import copy
from typing import List, Dict, Tuple, Optional


def normalize_shape(pixels: List[Tuple[int, int]]) -> frozenset:
    """形状を正規化（位置を(0,0)基準に）

    Args:
        pixels: ピクセル座標のリスト [(x, y), ...]

    Returns:
        正規化された形状のセット（位置に依存しない）
    """
    if not pixels:
        return frozenset()

    # 最適化: 1回のループでmin_xとmin_yを取得（3回のループ→2回のループに削減）
    min_x = float('inf')
    min_y = float('inf')
    for x, y in pixels:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

    # 相対位置に変換
    normalized = frozenset((x - min_x, y - min_y) for x, y in pixels)
    return normalized


def get_shape_signature(obj: Dict) -> Optional[frozenset]:
    """オブジェクトから形状のシグネチャを取得

    Args:
        obj: オブジェクト辞書（'pixels'キーが必要）

    Returns:
        正規化された形状のセット、またはNone
    """
    if not isinstance(obj, dict):
        return None

    pixels = obj.get('pixels')
    if not pixels or not isinstance(pixels, list):
        return None

    return normalize_shape(pixels)


def rotate_pixels(pixels: List[Tuple[int, int]], angle: int) -> List[Tuple[int, int]]:
    """ピクセルを回転（90度単位、最適化版）

    Args:
        pixels: ピクセル座標のリスト
        angle: 回転角度（90, 180, 270）

    Returns:
        回転後のピクセル座標のリスト（正規化済み）
    """
    if angle == 0:
        return pixels.copy()

    # 90度ごとに回転
    result = pixels.copy()
    for _ in range(angle // 90):
        result = [(-y, x) for x, y in result]

    # 正規化（最小座標を(0,0)に、最適化: 1回のループでmin/maxを取得）
    if result:
        min_x = min_y = float('inf')
        for x, y in result:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
        result = [(x - min_x, y - min_y) for x, y in result]

    return result


def flip_pixels(pixels: List[Tuple[int, int]], axis: str) -> List[Tuple[int, int]]:
    """ピクセルを反転（最適化版）

    Args:
        pixels: ピクセル座標のリスト
        axis: 反転軸（'X'または'Y'）

    Returns:
        反転後のピクセル座標のリスト（正規化済み）
    """
    if not pixels:
        return pixels.copy()

    # 1回のループでmin/maxを取得（最適化）
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for x, y in pixels:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    if axis == 'X':
        # X軸で反転（水平反転）
        result = [(max_x - x, y) for x, y in pixels]
        # 正規化（反転後のmin_xは常に0、min_yは変わらない）
        new_min_x = 0
        new_min_y = min_y
    elif axis == 'Y':
        # Y軸で反転（垂直反転）
        result = [(x, max_y - y) for x, y in pixels]
        # 正規化（反転後のmin_xは変わらない、min_yは常に0）
        new_min_x = min_x
        new_min_y = 0
    else:
        return pixels.copy()

    # 正規化（反転の性質を利用してオフセットを計算）
    if result:
        result = [(x - new_min_x, y - new_min_y) for x, y in result]

    return result


def copy_object_with_new_color(obj: Dict, new_color: int, apply_transformation: bool = False,
                                rng=None) -> Dict:
    """オブジェクトをコピーして新しい色を設定

    オプションで反転・回転を適用可能（ARC-AGI2の分析結果: 0.5%が変換あり）

    Args:
        obj: 元のオブジェクト
        new_color: 新しい色
        apply_transformation: 反転・回転を適用するか（デフォルト: False）
        rng: 乱数生成器（apply_transformation=Trueの場合に必要）

    Returns:
        コピーされたオブジェクト（色が変更されている、オプションで変換あり）
    """
    if not isinstance(obj, dict):
        raise ValueError("オブジェクトは辞書である必要があります")

    # 最適化: 必要な属性のみをコピー（deepcopyより高速）
    # pixelsは変換時に新しいリストが作成されるため、元のリストへの参照があっても問題ない
    # ただし、変換が適用されない場合でも安全のため、pixelsリストをコピーする
    copied_obj = {}
    # 全ての属性を浅くコピー
    for key in obj:
        if key == 'pixels':
            # pixelsリストをコピー（変換が適用される場合は新しいリストに置き換えられるが、安全のため）
            pixels = obj.get('pixels', [])
            copied_obj['pixels'] = pixels.copy() if pixels else []
        elif key == 'color':
            # 色は新しい値で上書き
            copied_obj['color'] = new_color
        else:
            # その他の属性はそのままコピー（値型なので浅いコピーで十分）
            copied_obj[key] = obj[key]

    # 色が存在しない場合は設定
    if 'color' not in copied_obj:
        copied_obj['color'] = new_color

    # 変換を適用（オプション、ARC-AGI2では0.5%程度）
    # apply_transformation=Trueの時点で変換を適用することが決定されているため、
    # ここでは変換タイプ（回転か反転か）のみを選択する
    if apply_transformation and rng is not None:
        pixels = copied_obj.get('pixels', [])
        if pixels and len(pixels) > 1:  # 1ピクセルの場合は変換不要
            # 変換タイプを選択（ARC-AGI2の傾向: 回転が多く、反転が少ない）
            # 回転80%（0.4% / 0.5%）、反転20%（0.1% / 0.5%）の比率
            rand_val = rng.random()
            if rand_val < 0.8:
                transform_type = 'rotate'    # 回転: 80%
            else:
                transform_type = 'flip'      # 反転: 20%

            if transform_type == 'rotate':
                # 回転角度を選択（90度単位）
                angle = rng.choice([90, 180, 270])
                transformed_pixels = rotate_pixels(pixels, angle)
                copied_obj['pixels'] = transformed_pixels

                # バウンディングボックスを再計算（最適化: 1回のループでmin/maxを取得）
                if transformed_pixels:
                    min_x = min_y = float('inf')
                    max_x = max_y = float('-inf')
                    for x, y in transformed_pixels:
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if y > max_y:
                            max_y = y
                    copied_obj['width'] = max_x - min_x + 1
                    copied_obj['height'] = max_y - min_y + 1
            elif transform_type == 'flip':
                # 反転軸を選択
                axis = rng.choice(['X', 'Y'])
                transformed_pixels = flip_pixels(pixels, axis)
                copied_obj['pixels'] = transformed_pixels

                # バウンディングボックスはflip_pixels内で既に計算済み（正規化済み）
                # ただし、width/heightが更新されていない可能性があるため、再計算
                if transformed_pixels:
                    min_x = min_y = float('inf')
                    max_x = max_y = float('-inf')
                    for x, y in transformed_pixels:
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if y > max_y:
                            max_y = y
                    copied_obj['width'] = max_x - min_x + 1
                    copied_obj['height'] = max_y - min_y + 1

    # 変更禁止フラグを保持（存在しない場合はFalseをデフォルト値として追加）
    copied_obj.setdefault('lock_color_change', False)
    copied_obj.setdefault('lock_shape_change', False)
    copied_obj.setdefault('lock_position_change', False)

    return copied_obj
