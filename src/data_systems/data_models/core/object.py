#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張オブジェクトクラス

クラスタリングを廃止し、オブジェクトの要素設定ステップに変更
オブジェクトの特徴を返す新しいクラス
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import cv2

from src.data_systems.data_models.base import ObjectType
import logging

logger = logging.getLogger("Object")

@dataclass
class Object:
    """統合オブジェクトクラス - 抽出から処理まで対応"""

    # 基本情報
    object_id: str
    object_type: ObjectType
    color_ratio: Dict[int, float] = field(default_factory=dict)
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)

    # ピクセルごとの色情報（オプション）
    pixel_colors: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # 元画像との関連情報
    source_image_id: str = ""
    source_image_type: str = ""
    source_image_index: int = 0

    # 色情報（計算済み）
    _dominant_color: int = -1
    _color_list: List[int] = field(default_factory=list)
    _color_counts: Dict[int, int] = field(default_factory=dict)

    # 形状情報（計算済み）
    _area: int = 0
    _bbox_area: int = 0
    _bbox_width: int = 0
    _bbox_height: int = 0
    _hole_count: int = 0
    _center_position: Tuple[int, int] = (0, 0)

    # 計算済みフラグ
    _color_computed: bool = False
    _shape_computed: bool = False
    _hole_computed: bool = False

    # 変更禁止フラグ
    lock_color_change: bool = False      # 色変更禁止
    lock_shape_change: bool = False      # 形変更禁止
    lock_position_change: bool = False   # 位置変更禁止

    # 本格実装の色推定のためのグリッド参照（後から設定される可能性あり）
    _grid: Optional[np.ndarray] = None

    # bbox関連のプロパティ
    @property
    def bbox_left(self) -> int:
        return self.bbox[0] if self.bbox else 0

    @property
    def bbox_top(self) -> int:
        return self.bbox[1] if self.bbox else 0

    @property
    def bbox_right(self) -> int:
        return self.bbox[2] if self.bbox else 0

    @property
    def bbox_bottom(self) -> int:
        return self.bbox[3] if self.bbox else 0

    @property
    def bbox_width(self) -> int:
        return self.bbox_right - self.bbox_left + 1 if self.bbox else 0

    @property
    def bbox_height(self) -> int:
        return self.bbox_bottom - self.bbox_top + 1 if self.bbox else 0

    @property
    def color(self) -> int:
        return self.dominant_color

    @property
    def dominant_color(self) -> int:
        """dominant_colorを動的に計算（pixel_colors優先）"""
        # 優先順位1: pixel_colorsから計算
        if hasattr(self, 'pixel_colors') and self.pixel_colors:
            color_counts = Counter(self.pixel_colors.values())
            most_common = color_counts.most_common(1)
            return most_common[0][0] if most_common else 0

        # 優先順位2: 計算済みの_dominant_color
        if self._dominant_color != -1:
            return self._dominant_color

        # 優先順位3: 再計算を試みる
        if self.pixels and not self._color_computed:
            self._compute_color_features()
            return self._dominant_color if self._dominant_color != -1 else 0

        return 0

    def __post_init__(self):
        """初期化後の処理"""
        if self.pixels:
            if self.pixels:
                xs = [x for x, y in self.pixels]  # pixels は (x, y) 形式
                ys = [y for x, y in self.pixels]  # pixels は (x, y) 形式
                # bboxが未設定の場合のみ自動計算（包括的：max_x, max_yを含む）
                if not self.bbox or self.bbox == (0, 0, 0, 0):
                    self.bbox = (min(xs), min(ys), max(xs), max(ys))
            self._compute_all_features()

    def _compute_all_features(self):
        """全ての特徴を計算"""
        import os
        import time

        # 穴検出をスキップするフラグ（環境変数で制御可能）
        DISABLE_HOLE_DETECTION = os.environ.get('DISABLE_HOLE_DETECTION', 'false').lower() in ('true', '1', 'yes')

        self._compute_color_features()
        self._compute_shape_features()

        # 穴検出は時間がかかる可能性があるため、スキップ可能
        if not DISABLE_HOLE_DETECTION:
            try:
                hole_start_time = time.time()
                self._compute_hole_features()
                hole_elapsed = time.time() - hole_start_time
                if hole_elapsed > 0.5:  # 0.5秒以上かかった場合は警告
                    logger.warning(f"穴検出に時間がかかりました: {hole_elapsed:.3f}秒")
            except Exception as e:
                # SilentExceptionは再発生させる（タスクを即座に廃棄するため）
                try:
                    from src.core_systems.executor.core import SilentException
                    if isinstance(e, SilentException):
                        raise  # SilentExceptionは再発生
                except ImportError:
                    pass  # ImportErrorの場合は通常のExceptionとして処理

                # 穴検出でエラーが発生した場合は、デフォルト値（0）を使用
                logger.warning(f"穴検出でエラーが発生しました（デフォルト値0を使用）: {e}")
                self._hole_count = 0
                self._hole_computed = True
        else:
            self._hole_count = 0
            self._hole_computed = True

    def _compute_color_features(self):
        """色特徴を計算（pixel_colors優先）"""
        if self._color_computed or not self.pixels:
            return

        # 優先順位1: pixel_colorsから計算（最も正確）
        if hasattr(self, 'pixel_colors') and self.pixel_colors:
            color_counter = Counter(self.pixel_colors.values())
            self._color_counts = dict(color_counter)
            self._color_list = [color for color, count in color_counter.most_common()]
            self._dominant_color = self._color_list[0] if self._color_list else -1
            self.color_ratio = {color: count/len(self.pixels) for color, count in color_counter.items()}
            self._color_computed = True
            return

        # 優先順位2: グリッドから計算
        if self._compute_color_from_grid():
            return

        # 優先順位3: 位置から推定（フォールバック）
        color_counter = Counter()
        for x, y in self.pixels:  # pixels は (x, y) 形式
            color_counter[self._estimate_color_from_position(y, x)] += 1

        # 色の情報を設定
        self._color_counts = dict(color_counter)
        self._color_list = [color for color, count in color_counter.most_common()]
        self._dominant_color = self._color_list[0] if self._color_list else -1
        self.color_ratio = {color: count/len(self.pixels) for color, count in color_counter.items()}

        self._color_computed = True

    def _estimate_color_from_position(self, y: int, x: int) -> int:
        """位置から色を推定（本格実装）"""
        # グリッドから実際の色を取得
        if hasattr(self, '_grid') and self._grid is not None:
            if 0 <= y < self._grid.shape[0] and 0 <= x < self._grid.shape[1]:
                return self._grid[y, x]

        # グリッドが利用できない場合は、ピクセルリストから色を推定
        if self.pixels:
            # 同じ位置のピクセルがあればその色を使用
            for py, px in self.pixels:
                if py == y and px == x:
                    # 実際の色情報があれば使用、なければデフォルト値
                    if hasattr(self, '_dominant_color') and self._dominant_color is not None:
                        return self._dominant_color

            # 最も近いピクセルの色を使用
            if self.pixels:
                min_dist = float('inf')
                closest_color = 0
                for py, px in self.pixels:
                    dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_color = self._dominant_color if hasattr(self, '_dominant_color') else 0
                return closest_color

        # フォールバック: デフォルト色
        return 0

    def _compute_color_from_grid(self):
        """グリッドから実際の色を計算"""
        if not hasattr(self, '_grid') or self._grid is None:
            return False

        actual_colors = []
        for x, y in self.pixels:  # pixels は (x, y) 形式
            if 0 <= y < self._grid.shape[0] and 0 <= x < self._grid.shape[1]:
                actual_colors.append(self._grid[y, x])

        if not actual_colors:
            return False

        color_counter = Counter(actual_colors)
        self._color_counts = dict(color_counter)
        self._color_list = [color for color, count in color_counter.most_common()]
        self._dominant_color = self._color_list[0] if self._color_list else -1
        self.color_ratio = {color: count/len(actual_colors) for color, count in color_counter.items()}

        self._color_computed = True
        return True

    def _compute_shape_features(self):
        """形状特徴を計算"""
        if self._shape_computed or not self.pixels:
            return

        self._area = len(self.pixels)

        # bboxが未設定または無効な場合のみ計算
        if self.pixels and (not self.bbox or self.bbox == (0, 0, 0, 0)):
            xs = [x for x, y in self.pixels]  # pixels は (x, y) 形式
            ys = [y for x, y in self.pixels]  # pixels は (x, y) 形式
            self.bbox = (min(xs), min(ys), max(xs), max(ys))

        # 早期チェック: バウンディングボックスが異常に大きい場合は即座にタスクを廃棄
        self._check_bbox_size_early()

        self._bbox_width = self.bbox[2] - self.bbox[0] + 1
        self._bbox_height = self.bbox[3] - self.bbox[1] + 1
        self._bbox_area = self._bbox_width * self._bbox_height

        self._center_position = (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )

        self._shape_computed = True

    def _check_bbox_size_early(self):
        """バウンディングボックスサイズの早期チェック（重複チェックを避ける）"""
        # 既にチェック済みの場合はスキップ（フラグで管理）
        if hasattr(self, '_bbox_checked') and self._bbox_checked:
            return

        if not self.bbox or self.bbox == (0, 0, 0, 0):
            return

        x1, y1, x2, y2 = self.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        # ARC問題を考慮: 30x30のグリッドで最大900ピクセル、正常系でもおおむね数百～千数百ピクセル
        # プログラム実行中、SCALE操作などで一時的に膨張する可能性はあるが、それでも1000ピクセル程度が実用上の上限
        # 1000ピクセル以上（30x30グリッドの約1.1倍）は異常とみなす
        MAX_BBOX_AREA = 1000  # ARC問題を考慮して1000ピクセル以上は異常とみなす

        if bbox_area > MAX_BBOX_AREA:
            # チェック済みフラグを設定（重複チェックを避ける）
            self._bbox_checked = True

            # SilentExceptionを発生させてタスクを即座に廃棄
            try:
                from src.core_systems.executor.core import SilentException
            except ImportError:
                # ImportErrorの場合は通常のExceptionを使用
                class SilentException(Exception):
                    pass

            error_msg = (
                f"大きなバウンディングボックス検出によりタスクを廃棄: "
                f"bbox={self.bbox} ({bbox_width}x{bbox_height}={bbox_area}ピクセル), "
                f"実際のピクセル数={len(self.pixels)}, "
                f"オブジェクトID={self.object_id}"
            )
            # ログ出力を抑制（テスト時のパフォーマンス向上のため）
            # 通常はログ出力をスキップして、SilentExceptionのみを発生させる
            # デバッグが必要な場合は、ロガーレベルをDEBUGに設定
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(error_msg)  # DEBUGレベルに変更して通常は出力されない
            raise SilentException(error_msg)

        # チェック済みフラグを設定
        self._bbox_checked = True

    def _compute_hole_features(self):
        """穴の特徴を計算"""
        if self._hole_computed or not self.pixels:
            return

        # 早期チェック: バウンディングボックスサイズをチェック（重複チェックを避ける）
        # _compute_shape_featuresで既にチェック済みの場合はスキップ
        if not hasattr(self, '_bbox_checked') or not self._bbox_checked:
            self._check_bbox_size_early()

        # グリッドが利用可能な場合はOpenCVのfindContoursを使用
        if hasattr(self, '_grid') and self._grid is not None:
            self._hole_count = self._calculate_hole_count_from_grid()
        else:
            # グリッドが利用できない場合は本格推定
            self._hole_count = self._calculate_hole_count_fallback()

        self._hole_computed = True

    def _calculate_hole_count(self) -> int:
        """穴の数を計算"""
        if not self.pixels:
            return 0

        # 早期チェック: バウンディングボックスサイズをチェック（重複チェックを避ける）
        # _compute_shape_featuresで既にチェック済みの場合はスキップ
        if not hasattr(self, '_bbox_checked') or not self._bbox_checked:
            self._check_bbox_size_early()

        # グリッドが利用可能な場合は正確な計算
        if hasattr(self, '_grid') and self._grid is not None:
            return self._calculate_hole_count_from_grid()
        else:
            # グリッドが利用できない場合は0を返す
            return 0

    def _calculate_hole_count_from_grid(self) -> int:
        """グリッドから正確な穴の数を計算"""
        if not self.pixels or not hasattr(self, '_grid') or self._grid is None:
            return 0

        # オブジェクトのバウンディングボックス内のグリッドを抽出
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            return 0

        # 早期チェック: バウンディングボックスサイズをチェック（重複チェックを避ける）
        # _compute_shape_featuresで既にチェック済みの場合はスキップ
        if not hasattr(self, '_bbox_checked') or not self._bbox_checked:
            self._check_bbox_size_early()

        # グリッドの境界チェック
        if (x1 < 0 or y1 < 0 or x2 >= self._grid.shape[1] or y2 >= self._grid.shape[0]):
            return 0

        # オブジェクトのピクセルを含むグリッド領域を抽出
        grid_region = self._grid[y1:y2+1, x1:x2+1].copy()

        # オブジェクトのピクセル位置をマスク
        mask = np.zeros_like(grid_region, dtype=np.uint8)
        for x, y in self.pixels:  # pixels は (x, y) 形式
            if 0 <= y - y1 < grid_region.shape[0] and 0 <= x - x1 < grid_region.shape[1]:
                mask[y - y1, x - x1] = 255

        # 穴の数を計算（OpenCVのfindContoursを使用）
        try:
            # 輪郭を検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 穴の数は内部の輪郭の数
            hole_count = 0
            for contour in contours:
                # 輪郭の面積が一定以上の場合のみ穴としてカウント
                area = cv2.contourArea(contour)
                if area > 0.5:  # 0.5ピクセル以上の面積
                    hole_count += 1

            return hole_count
        except Exception as e:
            # OpenCVでエラーが発生した場合は本格実装フォールバック
            logger.warning(f"OpenCV穴検出エラー: {e}")
            return self._calculate_hole_count_fallback()

    def _calculate_hole_count_fallback(self) -> int:
        """穴検出の本格実装フォールバック"""
        try:
            if not self.pixels or len(self.pixels) < 4:
                return 0

            # 早期チェック: バウンディングボックスサイズをチェック（重複チェックを避ける）
            # _compute_shape_featuresで既にチェック済みの場合はスキップ
            if not hasattr(self, '_bbox_checked') or not self._bbox_checked:
                self._check_bbox_size_early()

            # バウンディングボックス内で穴を検出
            x1, y1, x2, y2 = self.bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            pixel_count = len(self.pixels)

            # バウンディングボックスが大きい場合の診断ログ（警告のみ）
            # ARC問題を考慮して、警告の閾値も下げる（1000ピクセル以上で警告）
            # ただし、タスク廃棄はMAX_BBOX_AREA（1000）で行う
            if bbox_area > 1000:
                logger.warning(
                    f"大きなバウンディングボックス検出: "
                    f"bbox={self.bbox} ({bbox_width}x{bbox_height}={bbox_area}ピクセル), "
                    f"実際のピクセル数={pixel_count}, "
                    f"オブジェクトID={self.object_id}, "
                    f"空き率={(bbox_area - pixel_count) / bbox_area * 100:.1f}%"
                )

            pixel_set = set(self.pixels)
            holes = 0
            visited = set()

            # バウンディングボックス内の各ピクセルをチェック
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if (x, y) in visited or (x, y) in pixel_set:
                        continue

                    # フラッドフィルで連続領域を検出
                    region = self._flood_fill_hole_detection(x, y, pixel_set, x1, y1, x2, y2, visited)

                    # 小さな連続領域は穴として扱う
                    if len(region) > 0 and len(region) < 10:  # 10ピクセル未満
                        holes += 1

                    visited.update(region)

            return holes

        except Exception as e:
            # SilentExceptionは再発生させる（タスクを即座に廃棄するため）
            silent_exception_cls = None
            try:
                from src.core_systems.executor.core import SilentException as _SilentException
                silent_exception_cls = _SilentException
            except ImportError:
                silent_exception_cls = None  # ImportErrorの場合は通常のExceptionとして処理

            if silent_exception_cls is not None and isinstance(e, silent_exception_cls):
                raise  # SilentExceptionは再発生

            logger.error(f"穴検出フォールバックエラー: {e}")
            return 0

    def _flood_fill_hole_detection(self, start_x: int, start_y: int, pixel_set: set,
                                  x1: int, y1: int, x2: int, y2: int, visited: set) -> List[Tuple[int, int]]:
        """穴検出用フラッドフィル"""
        try:
            region = []
            stack = [(start_x, start_y)]
            current_visited = set()  # このフラッドフィル呼び出し内で訪問した座標

            while stack:
                x, y = stack.pop()

                # 既に訪問済みまたはオブジェクトのピクセルの場合はスキップ
                if (x, y) in visited or (x, y) in pixel_set or (x, y) in current_visited:
                    continue

                # バウンディングボックス外の場合はスキップ
                if not (x1 <= x < x2 and y1 <= y < y2):
                    continue

                current_visited.add((x, y))
                visited.add((x, y))
                region.append((x, y))

                # 4方向に拡張（スタックに追加する前に既に訪問済みかチェック）
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in current_visited and (nx, ny) not in visited and (nx, ny) not in pixel_set:
                        if x1 <= nx < x2 and y1 <= ny < y2:
                            stack.append((nx, ny))

            return region

        except Exception as e:
            logger.error(f"フラッドフィル穴検出エラー: {e}")
            return []


    # プロパティ（要求された要素）

    @property
    def dominant_color(self) -> int:
        """オブジェクトを構成する色の中で占める量が多い色の番号"""
        if not self._color_computed:
            self._compute_color_features()
        return self._dominant_color

    @property
    def color_list(self) -> List[int]:
        """オブジェクトを構成する色の番号の配列（量が多い順）"""
        if not self._color_computed:
            self._compute_color_features()
        return self._color_list.copy()

    @property
    def color_count(self) -> int:
        """オブジェクトを構成する色の数"""
        if not self._color_computed:
            self._compute_color_features()
        return len(self._color_list)

    @property
    def bbox_area(self) -> int:
        """オブジェクトのバウンディングボックスの面積"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._bbox_area

    @property
    def pixel_count(self) -> int:
        """オブジェクトのピクセル数"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._area

    @property
    def bbox_width(self) -> int:
        """オブジェクトのバウンディングボックスの幅"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._bbox_width

    @property
    def bbox_height(self) -> int:
        """オブジェクトのバウンディングボックスの高さ"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._bbox_height

    @property
    def hole_count(self) -> int:
        """オブジェクトの穴の数"""
        if not self._hole_computed:
            self._compute_hole_features()
        return self._hole_count

    @property
    def center_position(self) -> Tuple[int, int]:
        """オブジェクトの中心の座標"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._center_position

    # 座標の分離（プログラム合成用）
    @property
    def bbox_x(self) -> int:
        """バウンディングボックスのX座標"""
        return self.bbox[0]

    @property
    def bbox_y(self) -> int:
        """バウンディングボックスのY座標"""
        return self.bbox[1]

    @property
    def center_x(self) -> int:
        """中心のX座標"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._center_position[0]

    @property
    def center_y(self) -> int:
        """中心のY座標"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._center_position[1]

    # 追加の有用な要素

    @property
    def density(self) -> int:
        """密度（実際のピクセル数/バウンディングボックス面積 * 100、整数）"""
        if not self._shape_computed:
            self._compute_shape_features()
        if self._bbox_area == 0:
            return 0
        return (self._area * 100) // self._bbox_area

    @property
    def perimeter(self) -> int:
        """周囲長（正確計算、1ピクセルの一辺を1として）"""
        if not self.pixels:
            return 0

        pixel_set = set(self.pixels)
        perimeter = 0

        for x, y in self.pixels:  # pixels は (x, y) 形式
            # 4方向の隣接ピクセルをチェック
            if (x + 1, y) not in pixel_set:  # 右
                perimeter += 1
            if (x - 1, y) not in pixel_set:  # 左
                perimeter += 1
            if (x, y + 1) not in pixel_set:  # 下
                perimeter += 1
            if (x, y - 1) not in pixel_set:  # 上
                perimeter += 1

        return perimeter

    # 形状判定プロパティ（プログラム合成用）
    @property
    def is_square(self) -> bool:
        """正方形かどうか"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._bbox_width == self._bbox_height

    @property
    def is_rectangle(self) -> bool:
        """矩形かどうか（正方形でない）"""
        if not self._shape_computed:
            self._compute_shape_features()
        return self._bbox_width != self._bbox_height

    @property
    def is_single_color(self) -> bool:
        """単色かどうか"""
        if not self._color_computed:
            self._compute_color_features()
        return self.color_count == 1

    # 形状分析プロパティ（プログラム合成用）
    @property
    def width_ratio(self) -> int:
        """幅/高さの比率（*100して整数）"""
        if not self._shape_computed:
            self._compute_shape_features()
        if self._bbox_height == 0:
            return 0
        return int((self._bbox_width / self._bbox_height) * 100)

    @property
    def is_horizontal(self) -> bool:
        """横長かどうか"""
        return self.width_ratio > 100

    @property
    def is_vertical(self) -> bool:
        """縦長かどうか"""
        return self.width_ratio < 100

    # 位置関係メソッド（プログラム合成用）

    def min_distance_to(self, other_obj) -> int:
        """他のオブジェクトに接触するまでの最小移動距離（8方向移動）"""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other_obj.bbox

        if (x1_max >= x2_min and x1_min <= x2_max and
            y1_max >= y2_min and y1_min <= y2_max):
            return 0

        if x1_max < x2_min:
            dx = x2_min - x1_max - 1
        elif x2_max < x1_min:
            dx = x1_min - x2_max - 1
        else:
            dx = 0

        if y1_max < y2_min:
            dy = y2_min - y1_max - 1
        elif y2_max < y1_min:
            dy = y1_min - y2_max - 1
        else:
            dy = 0

        return max(dx, dy)

    def min_x_distance_to(self, other_obj) -> int:
        """他のオブジェクトに接触するまでのX方向最小移動距離"""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other_obj.bbox

        if (x1_max >= x2_min and x1_min <= x2_max and
            y1_max >= y2_min and y1_min <= y2_max):
            return 0

        if x1_max < x2_min:
            return x2_min - x1_max - 1
        elif x2_max < x1_min:
            return x1_min - x2_max - 1
        else:
            return 0

    def min_y_distance_to(self, other_obj) -> int:
        """他のオブジェクトに接触するまでのY方向最小移動距離"""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other_obj.bbox

        if (x1_max >= x2_min and x1_min <= x2_max and
            y1_max >= y2_min and y1_min <= y2_max):
            return 0

        if y1_max < y2_min:
            return y2_min - y1_max - 1
        elif y2_max < y1_min:
            return y1_min - y2_max - 1
        else:
            return 0

    def is_above(self, other_obj) -> bool:
        """他のオブジェクトより上にあるか"""
        return self.center_y < other_obj.center_y

    def is_below(self, other_obj) -> bool:
        """他のオブジェクトより下にあるか"""
        return self.center_y > other_obj.center_y

    def is_left_of(self, other_obj) -> bool:
        """他のオブジェクトより左にあるか"""
        return self.center_x < other_obj.center_x

    def is_right_of(self, other_obj) -> bool:
        """他のオブジェクトより右にあるか"""
        return self.center_x > other_obj.center_x

    def is_diagonal_to(self, other_obj) -> bool:
        """対角線上にあるか"""
        return (self.center_x != other_obj.center_x and
                self.center_y != other_obj.center_y)

    def is_aligned_horizontally(self, other_obj) -> bool:
        """水平に整列しているか"""
        return abs(self.center_y - other_obj.center_y) <= 1

    def is_aligned_vertically(self, other_obj) -> bool:
        """垂直に整列しているか"""
        return abs(self.center_x - other_obj.center_x) <= 1

    def is_adjacent_to(self, other_obj: 'Object') -> bool:
        """隣接しているか (bool) - 本格実装"""
        try:
            # ピクセルレベルでの隣接判定
            if not self.pixels or not other_obj.pixels:
                return False

            # 4方向の隣接チェック
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            for y1, x1 in self.pixels:
                for dy, dx in directions:
                    adjacent_y, adjacent_x = y1 + dy, x1 + dx
                    if (adjacent_y, adjacent_x) in other_obj.pixels:
                        return True

            return False
        except:
            return False

    def is_adjacent_to_bbox(self, other_obj: 'Object') -> bool:
        """バウンディングボックスベースで隣接しているか (bool)"""
        try:
            # バウンディングボックスが隣接しているかチェック
            left1, top1 = self.bbox_x, self.bbox_y
            right1, bottom1 = self.bbox_x + self.bbox_width, self.bbox_y + self.bbox_height

            left2, top2 = other_obj.bbox_x, other_obj.bbox_y
            right2, bottom2 = other_obj.bbox_x + other_obj.bbox_width, other_obj.bbox_y + other_obj.bbox_height

            # 隣接条件: 1ピクセル以内で接しているが、重複していない
            horizontal_adjacent = (right1 == left2 or left1 == right2) and not (bottom1 <= top2 or top1 >= bottom2)
            vertical_adjacent = (bottom1 == top2 or top1 == bottom2) and not (right1 <= left2 or left1 >= right2)

            return horizontal_adjacent or vertical_adjacent
        except:
            return False

    def is_overlapping_with(self, other_obj: 'Object') -> bool:
        """重複しているか (bool) - 本格実装"""
        try:
            # ピクセルレベルでの重複判定
            if not self.pixels or not other_obj.pixels:
                return False

            # 共通のピクセルがあるかチェック
            self_pixel_set = set(self.pixels)
            other_pixel_set = set(other_obj.pixels)

            return bool(self_pixel_set & other_pixel_set)
        except:
            return False

    def is_overlapping_with_bbox(self, other_obj: 'Object') -> bool:
        """バウンディングボックスベースで重複しているか (bool)"""
        try:
            left1, top1 = self.bbox_x, self.bbox_y
            right1, bottom1 = self.bbox_x + self.bbox_width, self.bbox_y + self.bbox_height

            left2, top2 = other_obj.bbox_x, other_obj.bbox_y
            right2, bottom2 = other_obj.bbox_x + other_obj.bbox_width, other_obj.bbox_y + other_obj.bbox_height

            # 重複条件: バウンディングボックスが重なっている
            return not (right1 <= left2 or left1 >= right2 or bottom1 <= top2 or top1 >= bottom2)
        except:
            return False

    def adjacent_edge_count(self, other_obj: 'Object') -> int:
        """隣接している辺の数 (int) - 本格実装"""
        try:
            # ピクセルレベルでの隣接辺数計算
            if not self.pixels or not other_obj.pixels:
                return 0

            # 重複している場合は隣接辺数を計算
            if self.is_overlapping_with(other_obj):
                return self._calculate_overlapping_adjacent_edges(other_obj)

            # 4方向の隣接チェック
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            adjacent_edges = 0

            for y1, x1 in self.pixels:
                for dy, dx in directions:
                    adjacent_y, adjacent_x = y1 + dy, x1 + dx
                    if (adjacent_y, adjacent_x) in other_obj.pixels:
                        adjacent_edges += 1

            return adjacent_edges
        except:
            return 0

    def _calculate_overlapping_adjacent_edges(self, other_obj: 'Object') -> int:
        """重複時の隣接辺数計算"""
        try:
            if not self.pixels or not other_obj.pixels:
                return 0

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            adjacent_edges = 0

            for y1, x1 in self.pixels:
                for dy, dx in directions:
                    adjacent_y, adjacent_x = y1 + dy, x1 + dx
                    if (adjacent_y, adjacent_x) in other_obj.pixels:
                        adjacent_edges += 1

            return adjacent_edges
        except:
            return 0

    def adjacent_edge_count_bbox(self, other_obj: 'Object') -> int:
        """バウンディングボックスベースで隣接している辺の数 (int)"""
        try:
            if self.is_overlapping_with_bbox(other_obj):
                return 0  # 重複している場合は0

            left1, top1 = self.bbox_x, self.bbox_y
            right1, bottom1 = self.bbox_x + self.bbox_width, self.bbox_y + self.bbox_height

            left2, top2 = other_obj.bbox_x, other_obj.bbox_y
            right2, bottom2 = other_obj.bbox_x + other_obj.bbox_width, other_obj.bbox_y + other_obj.bbox_height

            edge_count = 0

            # 上辺の隣接
            if bottom1 == top2 and not (right1 <= left2 or left1 >= right2):
                edge_count += 1

            # 下辺の隣接
            if top1 == bottom2 and not (right1 <= left2 or left1 >= right2):
                edge_count += 1

            # 左辺の隣接
            if right1 == left2 and not (bottom1 <= top2 or top1 >= bottom2):
                edge_count += 1

            # 右辺の隣接
            if left1 == right2 and not (bottom1 <= top2 or top1 >= bottom2):
                edge_count += 1

            return edge_count
        except:
            return 0

    def overlapping_pixel_count(self, other_obj: 'Object') -> int:
        """重複しているピクセル数 (int) - 本格実装"""
        try:
            # ピクセルレベルでの重複ピクセル数計算
            if not self.pixels or not other_obj.pixels:
                return 0

            # 共通のピクセル数を計算
            self_pixel_set = set(self.pixels)
            other_pixel_set = set(other_obj.pixels)

            return len(self_pixel_set & other_pixel_set)
        except:
            return 0

    def overlapping_pixel_count_bbox(self, other_obj: 'Object') -> int:
        """バウンディングボックスベースで重複しているピクセル数 (int)"""
        try:
            if not self.is_overlapping_with_bbox(other_obj):
                return 0

            left1, top1 = self.bbox_x, self.bbox_y
            right1, bottom1 = self.bbox_x + self.bbox_width, self.bbox_y + self.bbox_height

            left2, top2 = other_obj.bbox_x, other_obj.bbox_y
            right2, bottom2 = other_obj.bbox_x + other_obj.bbox_width, other_obj.bbox_y + other_obj.bbox_height

            # 重複領域の計算
            overlap_left = max(left1, left2)
            overlap_top = max(top1, top2)
            overlap_right = min(right1, right2)
            overlap_bottom = min(bottom1, bottom2)

            if overlap_left < overlap_right and overlap_top < overlap_bottom:
                return (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
            else:
                return 0
        except:
            return 0

    def has_same_shape(self, other_obj) -> bool:
        """オブジェクトのピクセルが他のオブジェクトのピクセルとお互いにすべて重なりあうか（色は無視）"""
        if len(self.pixels) != len(other_obj.pixels):
            return False

        self_pixel_positions = set(self.pixels)
        other_pixel_positions = set(other_obj.pixels)

        return self_pixel_positions == other_pixel_positions

    def has_same_color_structure(self, other_obj) -> bool:
        """同じ形状でかつ、色の構造が同じか判定（色は違ってもいい）"""
        if not self.has_same_shape(other_obj):
            return False

        # グリッドが利用可能な場合は実際の色を使用
        if (hasattr(self, '_grid') and self._grid is not None and
            hasattr(other_obj, '_grid') and other_obj._grid is not None):
            return self._compare_color_structure_from_grid(other_obj)
        else:
            # グリッドが利用できない場合は本格推定
            return self._compare_color_structure_from_grid(other_obj)

    def _compare_color_structure_from_grid(self, other_obj):
        """グリッドから実際の色構造を比較"""
        self_color_pattern = []
        other_color_pattern = []

        for x, y in self.pixels:  # pixels は (x, y) 形式
            if (0 <= y < self._grid.shape[0] and 0 <= x < self._grid.shape[1] and
                0 <= y < other_obj._grid.shape[0] and 0 <= x < other_obj._grid.shape[1]):
                self_color = self._grid[y, x]
                other_color = other_obj._grid[y, x]
                self_color_pattern.append(self_color)
                other_color_pattern.append(other_color)

        return self_color_pattern == other_color_pattern

    def _compare_color_structure_from_estimation(self, other_obj):
        """位置から色構造を推定比較"""
        self_color_pattern = []
        other_color_pattern = []

        for x, y in self.pixels:  # pixels は (x, y) 形式
            self_color = self._estimate_color_from_position(y, x)
            other_color = other_obj._estimate_color_from_position(y, x)
            self_color_pattern.append(self_color)
            other_color_pattern.append(other_color)

        return self_color_pattern == other_color_pattern

    def has_same_shape_and_color(self, other_obj) -> bool:
        """同じ形状でかつ、色の構造と色が同じか判定"""
        if not self.has_same_color_structure(other_obj):
            return False

        for color in self.color_ratio:
            if color not in other_obj.color_ratio or \
               abs(self.color_ratio[color] - other_obj.color_ratio[color]) > 0.001:
                return False

        for color in other_obj.color_ratio:
            if color not in self.color_ratio:
                return False

        return True

    def update_from_grid(self, grid: np.ndarray):
        """グリッドから実際の色情報を更新"""
        if not self.pixels:
            return

        actual_colors = []
        for x, y in self.pixels:  # pixels は (x, y) 形式
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                actual_colors.append(grid[y, x])

        color_counter = Counter(actual_colors)
        self._color_counts = dict(color_counter)
        self._color_list = [color for color, count in color_counter.most_common()]
        self._dominant_color = self._color_list[0] if self._color_list else -1

        self.color_ratio = {color: count/len(actual_colors) for color, count in color_counter.items()}

        self._color_computed = True

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'object_id': self.object_id,
            'object_type': self.object_type.value,
            'color_ratio': self.color_ratio,
            'pixels': self.pixels,
            'bbox': self.bbox,
            'dominant_color': self.dominant_color,
            'color_list': self.color_list,
            'color_count': self.color_count,
            'bbox_area': self.bbox_area,
            'pixel_count': self.pixel_count,
            'bbox_width': self.bbox_width,
            'bbox_height': self.bbox_height,
            'hole_count': self.hole_count,
            'center_position': self.center_position,
            'bbox_x': self.bbox_x,
            'bbox_y': self.bbox_y,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'density': self.density,
            'perimeter': self.perimeter,
            'is_square': self.is_square,
            'is_rectangle': self.is_rectangle,
            'is_single_color': self.is_single_color,
            'width_ratio': self.width_ratio,
            'is_horizontal': self.is_horizontal,
            'is_vertical': self.is_vertical,
            'source_image_id': self.source_image_id,
            'source_image_type': self.source_image_type,
            'source_image_index': self.source_image_index
        }

def create_object_from_extraction(object_id: str, object_type: ObjectType,
                                 pixels: List[Tuple[int, int]], grid: np.ndarray,
                                 source_image_id: str = "", source_image_type: str = "",
                                 source_image_index: int = 0) -> Object:
    """抽出情報から直接Objectを作成（統合版）"""
    obj = Object(
        object_id=object_id,
        object_type=object_type,
        pixels=pixels,
        source_image_id=source_image_id,
        source_image_type=source_image_type,
        source_image_index=source_image_index
    )

    # グリッドを設定
    obj._grid = grid

    # 実際の色情報で更新
    obj.update_from_grid(grid)

    return obj
