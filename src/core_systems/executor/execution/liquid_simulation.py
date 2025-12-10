#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
液体シミュレーションシステム
複雑な液体行動のシミュレーション機能を提供
"""

from typing import List, Tuple, Optional, Set
from src.data_systems.data_models.core.object import Object
import logging

logger = logging.getLogger("LiquidSimulation")


class LiquidSimulator:
    """液体シミュレータ"""
    
    def __init__(self):
        """液体シミュレータの初期化"""
        # 重力方向のマッピング
        self.gravity_directions = {
            'Y': (0, 1),
            '-Y': (0, -1),
            '-X': (-1, 0),
            'X': (1, 0)
        }
        
        # シミュレーション設定
        self.max_iterations = 100
        self.collision_detection = True
    
    def simulate_liquid_behavior(self, source_obj: Object, all_objects: List[Object], 
                                 gravity_direction: str, max_iterations: int = 100) -> List[Tuple[int, int]]:
        """液体行動シミュレーション
        
        Args:
            source_obj: 流出元オブジェクト
            all_objects: すべてのオブジェクト（障害物として扱う）
            gravity_direction: 重力方向 ('X', 'Y', '-X', '-Y')
            max_iterations: 最大反復回数
        
        Returns:
            液体ピクセルのリスト
        """
        logger.info(f"液体シミュレーション開始: {gravity_direction}方向, {max_iterations}回")
        
        if not source_obj.pixels:
            return []
        
        # 流出口オブジェクトのピクセルセット
        source_pixels = set(source_obj.pixels)
        
        # 重力方向に応じて流出面を決定
        outlet_pixels = self._find_outlet_pixels(source_obj.pixels, source_pixels, gravity_direction)
        
        # グリッドサイズを動的に計算
        grid_width, grid_height = self._calculate_grid_size(all_objects + [source_obj])
        
        # ソースレイヤー（常に0として扱う）
        source_layer = 0
        
        # 液体シミュレーション実行
        all_liquid_pixels = set()
        
        for outlet_pixel in outlet_pixels:
            logger.info(f"流出点から液体流動開始: {outlet_pixel}")
            
            # この1ピクセル流出点専用の軌道セット
            outlet_trajectory = set()
            
            # 重力方向の流路が衝突した位置Aを見つける
            collision_position_a, gravity_path = self._find_gravity_collision_position(
                outlet_pixel, gravity_direction, grid_width, grid_height, all_objects, source_layer
            )
            
            # 重力流動の軌道を記録
            all_liquid_pixels.update(gravity_path)
            outlet_trajectory.update(gravity_path)
            
            if collision_position_a is None:
                logger.info("重力流動完了（衝突なし）")
                continue
            
            logger.info(f"衝突位置A発見: {collision_position_a}")
            
            # ループ処理開始
            self._process_horizontal_flow_loop(
                collision_position_a, gravity_direction, grid_width, grid_height,
                all_objects, source_layer, all_liquid_pixels, outlet_trajectory, max_iterations
            )
        
        logger.info(f"液体シミュレーション完了: 流出面{len(outlet_pixels)}ピクセル, 液体{len(all_liquid_pixels)}ピクセル")
        return list(all_liquid_pixels)
    
    def _find_outlet_pixels(self, pixels: List[Tuple[int, int]], 
                           source_pixels: Set[Tuple[int, int]], 
                           gravity_direction: str) -> List[Tuple[int, int]]:
        """流出面のピクセルを見つける"""
        outlet_pixels = []
        
        if gravity_direction == 'Y':
            # Y方向: オブジェクトの下端面のピクセルから流出
            for x, y in pixels:
                if (x, y + 1) not in source_pixels:
                    outlet_pixels.append((x, y))
        elif gravity_direction == '-Y':
            # -Y方向: オブジェクトの上端面のピクセルから流出
            for x, y in pixels:
                if (x, y - 1) not in source_pixels:
                    outlet_pixels.append((x, y))
        elif gravity_direction == 'X':
            # X方向: オブジェクトの右端面のピクセルから流出
            for x, y in pixels:
                if (x + 1, y) not in source_pixels:
                    outlet_pixels.append((x, y))
        elif gravity_direction == '-X':
            # -X方向: オブジェクトの左端面のピクセルから流出
            for x, y in pixels:
                if (x - 1, y) not in source_pixels:
                    outlet_pixels.append((x, y))
        
        return outlet_pixels
    
    def _calculate_grid_size(self, all_objects: List[Object]) -> Tuple[int, int]:
        """グリッドサイズを計算"""
        all_pixels = []
        for obj in all_objects:
            all_pixels.extend(obj.pixels)
        
        if all_pixels:
            min_x = min(p[0] for p in all_pixels)
            max_x = max(p[0] for p in all_pixels)
            min_y = min(p[1] for p in all_pixels)
            max_y = max(p[1] for p in all_pixels)
            
            # グリッドサイズを決定（マージンを追加）
            grid_width = max_x + 20
            grid_height = max_y + 20
            
            logger.info(f"動的グリッドサイズ: {grid_width}x{grid_height}")
        else:
            grid_width, grid_height = 40, 40
            logger.warning("オブジェクトピクセルが空のため、デフォルトグリッドサイズ40x40を使用")
        
        return grid_width, grid_height
    
    def _find_gravity_collision_position(self, start_point: Tuple[int, int], gravity_direction: str,
                                        grid_width: int, grid_height: int, all_objects: List[Object],
                                        source_layer: int) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """重力方向の流路が衝突した位置Aを見つける"""
        current_point = start_point
        liquid_pixels = [current_point]
        
        while True:
            # 次の重力方向の位置を計算
            next_point = self._move_gravity(current_point, gravity_direction)
            
            # 境界チェック
            if not self._is_within_bounds(next_point, grid_width, grid_height):
                return None, liquid_pixels
            
            # 同レイヤーのオブジェクトとの衝突チェック
            if self._check_collision_with_same_layer_objects(next_point, all_objects, source_layer):
                return current_point, liquid_pixels
            
            current_point = next_point
            liquid_pixels.append(current_point)
    
    def _process_horizontal_flow_loop(self, start_position: Tuple[int, int], gravity_direction: str,
                                     grid_width: int, grid_height: int, all_objects: List[Object],
                                     source_layer: int, all_liquid_pixels: Set, outlet_trajectory: Set,
                                     max_iterations: int):
        """横方向流路のループ処理（複数回の衝突と横広がりを繰り返す）"""
        current_position_a = start_position
        iteration_count = 0
        
        # 重力方向に応じて「横方向」を決定
        if gravity_direction in ['Y', '-Y']:
            perpendicular_directions = ['-X', 'X']
        else:  # 'X' or '-X'
            perpendicular_directions = ['-Y', 'Y']
        
        while iteration_count < max_iterations:
            iteration_count += 1
            logger.info(f"=== ループ処理 {iteration_count}回目 ===")
            
            # 垂直方向（横）それぞれで衝突がないかを検出
            perp_dir1_possible = self._check_horizontal_path_possible(
                current_position_a, perpendicular_directions[0], grid_width, grid_height, all_objects, source_layer
            )
            perp_dir2_possible = self._check_horizontal_path_possible(
                current_position_a, perpendicular_directions[1], grid_width, grid_height, all_objects, source_layer
            )
            
            # 両方向の流路を作成して処理（戻り値は(新しい衝突位置, 境界到達フラグ)）
            new_collision_positions = []
            boundary_reached = False
            
            if perp_dir1_possible:
                collision_pos, is_boundary = self._process_horizontal_flow_path(
                    current_position_a, perpendicular_directions[0], gravity_direction, grid_width, grid_height, 
                    all_objects, source_layer, all_liquid_pixels, outlet_trajectory
                )
                if collision_pos:
                    new_collision_positions.append(collision_pos)
                if is_boundary:
                    boundary_reached = True
            
            if perp_dir2_possible:
                collision_pos, is_boundary = self._process_horizontal_flow_path(
                    current_position_a, perpendicular_directions[1], gravity_direction, grid_width, grid_height, 
                    all_objects, source_layer, all_liquid_pixels, outlet_trajectory
                )
                if collision_pos:
                    new_collision_positions.append(collision_pos)
                if is_boundary:
                    boundary_reached = True
            
            # 新しい衝突位置が見つかった場合、それらから再帰的に処理
            if new_collision_positions:
                logger.info(f"新しい衝突位置{len(new_collision_positions)}箇所発見、再帰処理開始")
                for new_pos in new_collision_positions:
                    self._process_horizontal_flow_loop(
                        new_pos, gravity_direction, grid_width, grid_height,
                        all_objects, source_layer, all_liquid_pixels, outlet_trajectory,
                        max_iterations - iteration_count  # 残り回数を渡す
                    )
                break  # 再帰呼び出しに任せてこのループは終了
            
            # 境界到達の場合、折り返さずに終了
            if boundary_reached:
                logger.info("グリッド境界到達、折り返さずに終了")
                break
            
            # 両方向とも失敗した場合（オブジェクト衝突など）、ループ終了
            # 液体は重力逆方向には積み上がらない
            logger.info("横方向への広がりが両方向とも不可、終了")
            break
    
    def _check_horizontal_path_possible(self, position: Tuple[int, int], direction: str,
                                       grid_width: int, grid_height: int, all_objects: List[Object],
                                       source_layer: int) -> bool:
        """横方向の流路が可能かチェック"""
        next_point = self._move_horizontally(position, direction)
        
        # 境界チェック
        if not self._is_within_bounds(next_point, grid_width, grid_height):
            return False
        
        # 同レイヤーのオブジェクトとの衝突チェック
        if self._check_collision_with_same_layer_objects(next_point, all_objects, source_layer):
            return False
        
        return True
    
    def _process_horizontal_flow_path(self, start_position: Tuple[int, int], direction: str,
                                     gravity_direction: str, grid_width: int, grid_height: int,
                                     all_objects: List[Object], source_layer: int,
                                     all_liquid_pixels: Set, outlet_trajectory: Set) -> Tuple[Optional[Tuple[int, int]], bool]:
        """横方向流路を処理し、新しい衝突位置を返す
        
        Returns:
            (新しい衝突位置, グリッド境界到達フラグ)
            - 新しい衝突位置: 重力方向に進んで再び衝突した位置、またはNone
            - グリッド境界到達フラグ: Trueならグリッド境界で終了（折り返さない）
        """
        current_point = start_position
        all_liquid_pixels.add(current_point)
        outlet_trajectory.add(current_point)
        
        while True:
            # 横方向に1ピクセル進む
            next_point = self._move_horizontally(current_point, direction)
            
            # 境界チェック（境界に達したらこの方向は終了、折り返さない）
            if not self._is_within_bounds(next_point, grid_width, grid_height):
                return (None, True)  # 境界到達で終了（折り返さない）
            
            # 同レイヤーのオブジェクトとの衝突チェック（オブジェクト衝突では折り返す）
            if self._check_collision_with_same_layer_objects(next_point, all_objects, source_layer):
                return (None, False)  # オブジェクト衝突で終了（折り返す）
            
            # 同じ流路から派生した流路同士の軌道衝突チェック
            if next_point in outlet_trajectory:
                return (None, False)  # 既存流路との衝突で終了（折り返す）
            
            current_point = next_point
            all_liquid_pixels.add(current_point)
            outlet_trajectory.add(current_point)
            
            # 重力方向に進めるかチェック
            gravity_next_point = self._move_gravity(current_point, gravity_direction)
            
            # 境界チェック
            if not self._is_within_bounds(gravity_next_point, grid_width, grid_height):
                continue  # 重力方向が境界外なら横移動継続
            
            # 衝突チェック
            if self._check_collision_with_same_layer_objects(gravity_next_point, all_objects, source_layer):
                continue  # 重力方向に障害物があるなら横移動継続
            
            # 重力方向に進める場合、そこから重力方向流路をトレースして新しい衝突位置を見つける
            new_collision_pos, gravity_path = self._find_gravity_collision_position(
                gravity_next_point, gravity_direction, grid_width, grid_height, all_objects, source_layer
            )
            all_liquid_pixels.update(gravity_path)
            outlet_trajectory.update(gravity_path)
            
            # 新しい衝突位置を返す（Noneの場合もある=境界まで到達）
            return (new_collision_pos, False)
    
    def _move_gravity(self, point: Tuple[int, int], gravity_direction: str) -> Tuple[int, int]:
        """重力方向に1ピクセル移動"""
        x, y = point
        if gravity_direction == 'Y':
            return (x, y + 1)
        elif gravity_direction == '-Y':
            return (x, y - 1)
        elif gravity_direction == 'X':
            return (x + 1, y)
        elif gravity_direction == '-X':
            return (x - 1, y)
        return point
    
    def _move_horizontally(self, point: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """指定方向に1ピクセル移動（X/Y/-X/-Yに対応）"""
        x, y = point
        if direction == '-X':
            return (x - 1, y)
        elif direction == 'X':
            return (x + 1, y)
        elif direction == '-Y':
            return (x, y - 1)
        elif direction == 'Y':
            return (x, y + 1)
        return point
    
    def _is_within_bounds(self, point: Tuple[int, int], grid_width: int, grid_height: int) -> bool:
        """グリッド境界内かどうかをチェック"""
        x, y = point
        return 0 <= x < grid_width and 0 <= y < grid_height
    
    def _check_collision_with_same_layer_objects(self, point: Tuple[int, int], 
                                                all_objects: List[Object], source_layer: int) -> bool:
        """同レイヤーのオブジェクトとの衝突判定（背景色除外）"""
        x, y = point
        
        # 同レイヤーのオブジェクトピクセルがあるかチェック（背景色は除外）
        for obj in all_objects:
                # 背景色（通常0）のオブジェクトは障害物として扱わない
                obj_color = getattr(obj, '_dominant_color', -1)
                if obj_color == 0:
                    continue
                
                if (x, y) in obj.pixels:
                    return True
        
        return False

