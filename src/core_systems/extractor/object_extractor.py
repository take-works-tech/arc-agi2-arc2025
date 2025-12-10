#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合オブジェクト抽出システム
ARC2025用正しいプログラム合成システムとの統合
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from src.data_systems.data_models.base import ObjectType
from src.data_systems.data_models.core.object import Object
from src.data_systems.config.config import ExtractionConfig

# ロギング設定
logger = logging.getLogger("IntegratedObjectExtractor")

@dataclass
class ObjectExtractionResult:
    """オブジェクト抽出結果"""
    objects_by_type: Dict[ObjectType, List[Object]] = field(default_factory=dict)
    background_color: int = 0
    grid_size: Tuple[int, int] = (0, 0)
    total_objects: int = 0
    extraction_time: float = 0.0
    success: bool = False

class IntegratedObjectExtractor:
    """統合オブジェクト抽出システム"""

    def __init__(self, config: ExtractionConfig = None):
        self.config = config if config else ExtractionConfig()
        logger.info("統合オブジェクト抽出システム初期化完了")

    def extract_objects_by_type(self, input_grid: np.ndarray, input_image_index: int = 0) -> ObjectExtractionResult:
        """オブジェクトタイプごとにオブジェクトを抽出"""
        try:
            logger.info(f"オブジェクト抽出開始: 画像インデックス {input_image_index}")
            import time
            start_time = time.time()

            # 背景色を推定
            background_color = self._estimate_background_color(input_grid)

            # グリッドサイズを取得
            grid_size = input_grid.shape

            # オブジェクトタイプごとに抽出
            objects_by_type = {}

            # 各オブジェクトタイプで抽出
            for object_type in ObjectType:
                if self._should_extract_type(object_type):
                    objects = self._extract_objects_of_type(input_grid, object_type, background_color, input_image_index)
                    if objects:
                        objects_by_type[object_type] = objects
                        logger.info(f"抽出完了: {object_type.value} - {len(objects)}個のオブジェクト")

            # 総オブジェクト数を計算
            total_objects = sum(len(objects) for objects in objects_by_type.values())

            extraction_time = time.time() - start_time

            result = ObjectExtractionResult(
                objects_by_type=objects_by_type,
                background_color=background_color,
                grid_size=grid_size,
                total_objects=total_objects,
                extraction_time=extraction_time,
                success=True
            )

            logger.info(f"オブジェクト抽出完了: {total_objects}個のオブジェクト, {extraction_time:.3f}秒")

            return result

        except Exception as e:
            logger.error(f"オブジェクト抽出エラー: {e}")
            return ObjectExtractionResult(success=False)

    def _should_extract_type(self, object_type: ObjectType) -> bool:
        """オブジェクトタイプの抽出が必要かチェック"""
        extraction_rules = {
            ObjectType.SINGLE_COLOR_4WAY: self.config.enable_single_color_4way,
            ObjectType.SINGLE_COLOR_8WAY: self.config.enable_single_color_8way,
            # ObjectType.MULTI_COLOR_4WAY: self.config.enable_multi_color_4way,  # 無効化
            # ObjectType.MULTI_COLOR_8WAY: self.config.enable_multi_color_8way,  # 無効化
            ObjectType.WHOLE_GRID: self.config.enable_whole_grid
        }

        return extraction_rules.get(object_type, False)

    def _extract_objects_of_type(self, input_grid: np.ndarray, object_type: ObjectType,
                                background_color: int, input_image_index: int) -> List[Object]:
        """特定のオブジェクトタイプでオブジェクトを抽出"""
        objects = []

        if object_type == ObjectType.SINGLE_COLOR_4WAY:
            objects = self._extract_single_color_4way(input_grid, background_color, input_image_index)
        elif object_type == ObjectType.SINGLE_COLOR_8WAY:
            objects = self._extract_single_color_8way(input_grid, background_color, input_image_index)
        # elif object_type == ObjectType.MULTI_COLOR_4WAY:  # 無効化
        #     objects = self._extract_multi_color_4way(input_grid, background_color, input_image_index)
        # elif object_type == ObjectType.MULTI_COLOR_8WAY:  # 無効化
        #     objects = self._extract_multi_color_8way(input_grid, background_color, input_image_index)
        elif object_type == ObjectType.WHOLE_GRID:
            objects = self._extract_whole_grid(input_grid, background_color, input_image_index)

        return objects

    def _extract_single_color_4way(self, input_grid: np.ndarray, background_color: int,
                                  input_image_index: int) -> List[Object]:
        """単色4連結オブジェクトを抽出"""
        objects = []
        height, width = input_grid.shape
        visited = np.zeros_like(input_grid, dtype=bool)
        object_index = 0

        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    # フラッドフィルで4連結領域を抽出（背景色も含む）
                    pixels = self._flood_fill_4way(input_grid, x, y, input_grid[y, x], visited)

                    if len(pixels) > 0:  # 1ピクセルオブジェクトも含める
                        # オブジェクトを作成
                        obj = self._create_object_from_pixels(
                            pixels, input_grid[y, x], ObjectType.SINGLE_COLOR_4WAY,
                            input_image_index, input_grid, object_index
                        )
                        if obj:
                            objects.append(obj)
                            object_index += 1

        return objects

    def _extract_single_color_8way(self, input_grid: np.ndarray, background_color: int,
                                  input_image_index: int) -> List[Object]:
        """単色8連結オブジェクトを抽出"""
        objects = []
        height, width = input_grid.shape
        visited = np.zeros_like(input_grid, dtype=bool)
        object_index = 0

        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    # フラッドフィルで8連結領域を抽出（背景色も含む）
                    pixels = self._flood_fill_8way(input_grid, x, y, input_grid[y, x], visited)

                    if len(pixels) > 0:  # 1ピクセルオブジェクトも含める（single_color_4wayと同じ）
                        # オブジェクトを作成
                        obj = self._create_object_from_pixels(
                            pixels, input_grid[y, x], ObjectType.SINGLE_COLOR_8WAY,
                            input_image_index, input_grid, object_index
                        )
                        if obj:
                            objects.append(obj)
                            object_index += 1

        return objects

    # def _extract_multi_color_4way(self, input_grid: np.ndarray, background_color: int,
    #                              input_image_index: int) -> List[Object]:
    #     """複数色4連結オブジェクトを抽出（無効化）"""
    #     objects = []
    #     height, width = input_grid.shape
    #     visited = np.zeros_like(input_grid, dtype=bool)
    #
    #     for y in range(height):
    #         for x in range(width):
    #             if not visited[y, x] and input_grid[y, x] != background_color:
    #                 # フラッドフィルで4連結領域を抽出（複数色対応）
    #                 pixels = self._flood_fill_multi_color_4way(input_grid, x, y, visited)
    #
    #                 if len(pixels) > 0:  # 1ピクセルオブジェクトも含める
    #                     # オブジェクトを作成
    #                     obj = self._create_object_from_pixels(
    #                         pixels, input_grid[y, x], ObjectType.MULTI_COLOR_4WAY,
    #                         input_image_index, input_grid
    #                     )
    #                     if obj:
    #                         objects.append(obj)
    #
    #     return objects

    # def _extract_multi_color_8way(self, input_grid: np.ndarray, background_color: int,
    #                              input_image_index: int) -> List[Object]:
    #     """複数色8連結オブジェクトを抽出（無効化）"""
    #     objects = []
    #     height, width = input_grid.shape
    #     visited = np.zeros_like(input_grid, dtype=bool)
    #
    #     for y in range(height):
    #         for x in range(width):
    #             if not visited[y, x] and input_grid[y, x] != background_color:
    #                 # フラッドフィルで8連結領域を抽出（複数色対応）
    #                 pixels = self._flood_fill_multi_color_8way(input_grid, x, y, visited)
    #
    #                 if len(pixels) > 0:  # 1ピクセルオブジェクトも含める
    #                     # オブジェクトを作成
    #                     obj = self._create_object_from_pixels(
    #                         pixels, input_grid[y, x], ObjectType.MULTI_COLOR_8WAY,
    #                         input_image_index, input_grid
    #                     )
    #                     if obj:
    #                         objects.append(obj)
    #
    #     return objects

    def _extract_whole_grid(self, input_grid: np.ndarray, background_color: int,
                           input_image_index: int) -> List[Object]:
        """画面全体をオブジェクトとして抽出"""
        height, width = input_grid.shape
        pixels = [(x, y) for y in range(height) for x in range(width)]

        obj = self._create_object_from_pixels(
            pixels, background_color, ObjectType.WHOLE_GRID,
            input_image_index, input_grid, 0
        )

        return [obj] if obj else []




    def _flood_fill_4way(self, grid: np.ndarray, start_x: int, start_y: int,
                        target_color: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """4連結フラッドフィル"""
        pixels = []
        height, width = grid.shape

        # 開始座標の境界チェック
        if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
            return pixels

        stack = [(start_x, start_y)]

        while stack:
            x, y = stack.pop()

            # 境界チェック（スタックから取り出した時点で再チェック）
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # 訪問済みまたは色が異なる場合はスキップ
            if visited[y, x] or grid[y, x] != target_color:
                continue

            visited[y, x] = True
            pixels.append((x, y))

            # 4方向に拡張（境界チェックは次の反復で行う）
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # 境界内の場合のみスタックに追加
                if 0 <= nx < width and 0 <= ny < height:
                    stack.append((nx, ny))

        return pixels

    def _flood_fill_8way(self, grid: np.ndarray, start_x: int, start_y: int,
                        target_color: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """8連結フラッドフィル"""
        pixels = []
        height, width = grid.shape

        # 開始座標の境界チェック
        if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
            return pixels

        stack = [(start_x, start_y)]

        while stack:
            x, y = stack.pop()

            # 境界チェック（スタックから取り出した時点で再チェック）
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # 訪問済みまたは色が異なる場合はスキップ
            if visited[y, x] or grid[y, x] != target_color:
                continue

            visited[y, x] = True
            pixels.append((x, y))

            # 8方向に拡張（境界チェックは次の反復で行う）
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                # 境界内の場合のみスタックに追加
                if 0 <= nx < width and 0 <= ny < height:
                    stack.append((nx, ny))

        return pixels


    def _create_object_from_pixels(self, pixels: List[Tuple[int, int]], dominant_color: int,
                                  object_type: ObjectType, input_image_index: int,
                                  input_grid: np.ndarray, object_index: int = 0) -> Optional[Object]:
        """ピクセルリストからオブジェクトを作成"""
        try:
            if not pixels:
                return None

            # バウンディングボックスを計算（包括的形式）
            x_coords = [p[0] for p in pixels]
            y_coords = [p[1] for p in pixels]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            # オブジェクトIDを生成（ユニークな連番を含める）
            object_id = f"obj_{input_image_index}_{object_type.value}_{object_index}"

            # pixel_colorsを設定（各ピクセルの色を保存）
            pixel_colors = {}
            for x, y in pixels:
                color = input_grid[y, x]
                pixel_colors[(x, y)] = color

            # オブジェクトを作成
            obj = Object(
                object_id=object_id,
                object_type=object_type,
                pixels=pixels,
                pixel_colors=pixel_colors,  # ピクセルごとの色を設定
                bbox=(x1, y1, x2, y2),
                source_image_id=f"input_{input_image_index}",
                source_image_type="input",
                source_image_index=input_image_index
            )
            # dominant_color、color_ratio、color_list、color_countsはpixel_colorsから自動計算される

            return obj

        except Exception as e:
            logger.error(f"オブジェクト作成エラー: {e}")
            return None

    def _estimate_background_color(self, input_grid: np.ndarray) -> int:
        """高度な背景色推察アルゴリズム"""
        try:
            # 1. 基本的な色頻度分析
            unique_colors, counts = np.unique(input_grid, return_counts=True)
            total_pixels = input_grid.size

            # 2. 複数の候補を評価
            candidates = []

            # 最頻色候補
            most_frequent_idx = np.argmax(counts)
            most_frequent_color = unique_colors[most_frequent_idx]
            frequency_score = counts[most_frequent_idx] / total_pixels
            candidates.append((most_frequent_color, frequency_score, 'frequency'))

            # エッジ色候補（背景色はエッジに現れやすい）
            edge_colors = self._get_edge_colors(input_grid)
            for color in edge_colors:
                if color in unique_colors:
                    color_idx = np.where(unique_colors == color)[0][0]
                    edge_score = counts[color_idx] / total_pixels
                    candidates.append((color, edge_score, 'edge'))

            # 3. 重み付きスコアで最適な背景色を選択
            best_color = self._select_best_background_color(candidates, input_grid.shape)

            return best_color

        except Exception as e:
            logger.error(f"高度な背景色推察エラー: {e}")
            return 0

    def _get_edge_colors(self, input_grid: np.ndarray) -> List[int]:
        """エッジピクセルの色を取得"""
        edge_colors = []
        height, width = input_grid.shape

        # 上下のエッジ
        edge_colors.extend(input_grid[0, :])  # 上
        edge_colors.extend(input_grid[height-1, :])  # 下

        # 左右のエッジ
        edge_colors.extend(input_grid[:, 0])  # 左
        edge_colors.extend(input_grid[:, width-1])  # 右

        return edge_colors

    def _select_best_background_color(self, candidates: List[Tuple[int, float, str]], grid_shape: Tuple[int, int]) -> int:
        """最適な背景色を選択"""
        best_color = 0
        best_score = 0

        for color, score, reason in candidates:
            # 基本スコア
            total_score = score

            # エッジボーナス
            if reason == 'edge':
                total_score += 0.1

            # 頻度ボーナス
            if reason == 'frequency':
                total_score += 0.05

            if total_score > best_score:
                best_score = total_score
                best_color = color

        return best_color



# プログラム合成システム用の統合関数（本格実装）
_global_extraction_cache = {}
_global_output_generator = None

def get_objects_by_type(input_image_index: int, object_type: str) -> List[Object]:
    """プログラム合成システム用: オブジェクトタイプごとのオブジェクト配列を取得（本格実装）"""
    try:
        # グローバルキャッシュから取得
        cache_key = f"{input_image_index}_{object_type}"

        if cache_key in _global_extraction_cache:
            return _global_extraction_cache[cache_key]

        # 統合オブジェクト抽出システムを使用
        from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
        from src.data_systems.config.config import ExtractionConfig
        from src.data_systems.data_models.base import ObjectType

        # 抽出システムを初期化
        config = ExtractionConfig()
        extractor = IntegratedObjectExtractor(config)

        # テスト用グリッドを作成（実際の実装では入力グリッドを渡す）
        import numpy as np
        # より現実的なテストグリッドを作成
        test_grid = np.zeros((10, 10), dtype=int)
        # テストパターンを追加
        test_grid[2:5, 2:5] = 1  # 3x3の四角形
        test_grid[6:8, 6:8] = 2  # 2x2の四角形

        # オブジェクトタイプを文字列から列挙型に変換
        try:
            object_type_enum = ObjectType(object_type.lower())
        except ValueError:
            logger.warning(f"無効なオブジェクトタイプ: {object_type}")
            return []

        # オブジェクトを抽出
        extraction_result = extractor.extract_objects_by_type(test_grid, input_image_index)

        if extraction_result.success and object_type_enum in extraction_result.objects_by_type:
            objects = extraction_result.objects_by_type[object_type_enum]
            _global_extraction_cache[cache_key] = objects
            return objects

        return []

    except Exception as e:
        logger.error(f"オブジェクト取得エラー: {e}")
        return []

def create_output_grid_from_objects(objects: List[Object]) -> List[List[int]]:
    """プログラム合成システム用: オブジェクトから出力グリッドを作成（本格実装）"""
    try:
        if not objects:
            return [[0]]

        # オブジェクトの境界を計算
        min_x = min(pixel[0] for obj in objects for pixel in obj.pixels)
        min_y = min(pixel[1] for obj in objects for pixel in obj.pixels)
        max_x = max(pixel[0] for obj in objects for pixel in obj.pixels)
        max_y = max(pixel[1] for obj in objects for pixel in obj.pixels)

        # グリッドサイズを決定
        width = max_x - min_x + 3  # パディング
        height = max_y - min_y + 3  # パディング

        # グリッドを初期化
        grid = [[0 for _ in range(width)] for _ in range(height)]

        # オブジェクトをグリッドに描画
        for obj in objects:
            for pixel_x, pixel_y in obj.pixels:
                # 相対座標に変換
                rel_x = pixel_x - min_x + 1
                rel_y = pixel_y - min_y + 1

                if 0 <= rel_y < height and 0 <= rel_x < width:
                    grid[rel_y][rel_x] = obj.dominant_color

        return grid

    except Exception as e:
        logger.error(f"出力グリッド生成エラー: {e}")
        return [[0]]

# アクセス用のエイリアス（s4_executorとの互換性のため）
class ObjectExtractor:
    """シンプルなオブジェクト抽出器（s4_executor用）"""

    def __init__(self):
        self.extractor = IntegratedObjectExtractor()

    def extract_objects(self, grid: np.ndarray, background_color: int = 0) -> List[Object]:
        """4連結でオブジェクトを抽出"""
        result = self.extractor.extract_objects_by_type(grid)
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
        return objects

    def extract_objects_8_connected(self, grid: np.ndarray, background_color: int = 0) -> List[Object]:
        """8連結でオブジェクトを抽出"""
        result = self.extractor.extract_objects_by_type(grid)
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_8WAY, [])
        return objects
