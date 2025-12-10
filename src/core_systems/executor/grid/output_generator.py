#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出力グリッド生成システム
オブジェクトリストからグリッドへの変換とレイヤー順序による描画
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

import logging
from src.data_systems.data_models.core.object import Object, ObjectType
from src.data_systems.config.config import ExecutionConfig

logger = logging.getLogger("OutputGridGenerator")

@dataclass
class GridGenerationConfig:
    """グリッド生成設定"""
    output_size_strategy: str = "original_input_size"  # "fit_to_objects", "original_input_size", "fixed_size", "predict_size"
    fixed_width: int = 5
    fixed_height: int = 5
    padding: int = 1
    background_color: int = 0
    enable_rendering: bool = True
    render_intermediate_steps: bool = False
    enable_size_prediction: bool = True
    size_prediction_confidence_threshold: float = 0.3

@dataclass
class GridGenerationResult:
    """グリッド生成結果"""
    output_grid: np.ndarray
    grid_size: Tuple[int, int]
    object_count: int
    object_count_rendered: int
    generation_time: float
    success: bool
    message: str = ""

class OutputGridGenerator:
    """出力グリッド生成システム"""
    
    def __init__(self, config: ExecutionConfig = None, grid_config: GridGenerationConfig = None):
        self.config = config if config else ExecutionConfig()
        self.grid_config = grid_config if grid_config else GridGenerationConfig()
        
        # サイズ予測システムは削除済み（プログラム内コマンドで代替）
        self.size_predictor = None
        
        logger.info("出力グリッド生成システム初期化完了")
    def generate_output_grid_from_objects(self, objects: List[Object], 
                                        input_grid_size: Tuple[int, int] = None,
                                        background_color: int = 0) -> GridGenerationResult:
        """オブジェクトリストから出力グリッドを生成"""
        try:
            import time
            start_time = time.time()
            
            logger.info(f"出力グリッド生成開始: {len(objects)}個のオブジェクト")
            
            # グリッドサイズを決定
            grid_size = self._determine_grid_size(objects, input_grid_size)
            
            # 出力グリッドを初期化
            output_grid = np.full((grid_size[1], grid_size[0]), background_color, dtype=np.uint8)
            
            # レイヤー順序でオブジェクトを描画
            rendered_objects = self._render_objects(objects, output_grid)
            
            generation_time = time.time() - start_time
            
            result = GridGenerationResult(
                output_grid=output_grid,
                grid_size=grid_size,
                object_count=len(objects),
                object_count_rendered=len(rendered_objects),
                generation_time=generation_time,
                success=True,
                message=f"出力グリッド生成完了: {grid_size[0]}x{grid_size[1]}, {len(rendered_objects)}個のオブジェクトを描画"
            )
            
            logger.info(f"出力グリッド生成完了: {generation_time:.3f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"出力グリッド生成エラー: {e}")
            return GridGenerationResult(
                output_grid=np.array([[0]], dtype=np.uint8),
                grid_size=(1, 1),
                object_count=0,
                object_count_rendered=0,
                generation_time=0.0,
                success=False,
                message=f"出力グリッド生成エラー: {e}"
            )
    
    def _determine_grid_size(self, objects: List[Object], 
                           input_grid_size: Tuple[int, int] = None) -> Tuple[int, int]:
        """グリッドサイズを決定"""
        try:
            if self.grid_config.output_size_strategy == "fixed_size":
                return (self.grid_config.fixed_width, self.grid_config.fixed_height)
            
            elif self.grid_config.output_size_strategy == "original_input_size":
                if input_grid_size:
                    return input_grid_size
                else:
                    # 入力サイズが不明な場合はオブジェクトに基づいて決定
                    return self._calculate_size_from_objects(objects)
            
            elif self.grid_config.output_size_strategy == "fit_to_objects":
                return self._calculate_size_from_objects(objects)
            
            else:
                # デフォルト: オブジェクトに基づいて決定
                return self._calculate_size_from_objects(objects)
                
        except Exception as e:
            logger.error(f"グリッドサイズ決定エラー: {e}")
            return (10, 10)  # デフォルトサイズ
    
    def _calculate_size_from_objects(self, objects: List[Object]) -> Tuple[int, int]:
        """オブジェクトからグリッドサイズを計算"""
        try:
            if not objects:
                return (self.grid_config.fixed_width, self.grid_config.fixed_height)
            
            # すべてのオブジェクトのバウンディングボックスを考慮
            min_x = min(obj.bbox[0] for obj in objects)
            min_y = min(obj.bbox[1] for obj in objects)
            max_x = max(obj.bbox[2] for obj in objects)
            max_y = max(obj.bbox[3] for obj in objects)
            
            # パディングを追加
            width = max_x - min_x + self.grid_config.padding * 2
            height = max_y - min_y + self.grid_config.padding * 2
            
            # 最小サイズを保証
            width = max(width, self.grid_config.fixed_width)
            height = max(height, self.grid_config.fixed_height)
            
            return (width, height)
            
        except Exception as e:
            logger.error(f"オブジェクトからサイズ計算エラー: {e}")
            return (self.grid_config.fixed_width, self.grid_config.fixed_height)
    
    def _render_objects(self, objects: List[Object], output_grid: np.ndarray) -> List[Object]:
        """配列順序でオブジェクトを描画
        
        描画順序: 配列の順序のみ（レイヤーシステムは廃止）
        配列の後ろのオブジェクトが手前に表示される（後で描画、上書き）
        """
        try:
            rendered_objects = []
            
            # 配列の順序のみで描画（レイヤーソート廃止）
            for obj in objects:
                # whole_gridオブジェクトは描画をスキップ（背景として扱われる）
                if obj.object_type.value == 'whole_grid':
                    logger.info(f"whole_gridオブジェクトをスキップ: {obj.object_id}")
                    continue
                    
                if self._render_object_on_grid(obj, output_grid):
                    rendered_objects.append(obj)
            
            logger.info(f"描画完了: {len(rendered_objects)}個のオブジェクトを配列順に描画")
            return rendered_objects
            
        except Exception as e:
            logger.error(f"描画エラー: {e}")
            return []
    
    def _render_object_on_grid(self, obj: Object, output_grid: np.ndarray) -> bool:
        """オブジェクトをグリッドに描画"""
        try:
            height, width = output_grid.shape
            rendered_pixels = 0
            
            # デバッグ: オブジェクトの色情報をログ出力
            
            # pixel_colorsが設定されているか確認
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
            
            # デフォルト色の検証（0-9の範囲内）
            default_color = obj.dominant_color
            if default_color < 0 or default_color > 9:
                logger.error(f"不正な色の値: {default_color} (オブジェクト: {obj.object_id})")
                return False
            
            for x, y in obj.pixels:
                # グリッド範囲内かチェック（負の座標も明示的にチェック）
                if x < 0 or y < 0 or x >= width or y >= height:
                    # 範囲外のピクセルはスキップ（エラーログなし）
                    continue
                
                # pixel_colorsがあればそれを優先、なければ_dominant_colorを使用
                if has_pixel_colors and (x, y) in obj.pixel_colors:
                    color = obj.pixel_colors[(x, y)]
                else:
                    color = default_color
                
                # 色の範囲チェック
                if 0 <= color <= 9:
                    output_grid[y, x] = color
                else:
                    logger.warning(f"不正なピクセル色: {color} at ({x},{y})")
                    output_grid[y, x] = default_color
                rendered_pixels += 1
            
            logger.info(f"オブジェクト描画完了: {obj.object_id} - {rendered_pixels}ピクセル")
            return rendered_pixels > 0
            
        except Exception as e:
            logger.error(f"オブジェクト描画エラー: {e}")
            return False
    
def create_output_grid_from_objects(objects: List[Object]) -> List[List[int]]:
    """プログラム合成システム用: オブジェクトから出力グリッドを作成"""
    generator = OutputGridGenerator()
    result = generator.generate_output_grid_from_objects(objects)
    return result.output_grid.tolist() if result.success else [[0]]
