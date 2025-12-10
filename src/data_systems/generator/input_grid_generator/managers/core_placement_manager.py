"""
コア配置マネージャー
"""
import random
import numpy as np
from typing import Dict, List


class CorePlacementManager:
    """コア配置マネージャー"""
    
    def __init__(self, seed=None):
        """初期化"""
        self.rng = random.Random(seed)
        
        # 配置パターン（レガシーシステムから拡充）
        self.placement_patterns = {
            'random': self._random_placement,
            'grid': self._grid_placement,
            'spiral': self._spiral_placement,
            'cluster': self._cluster_placement,
            'border': self._border_placement,
            'center': self._center_placement
        }
        
        # レガシーシステムから追加（メソッド定義後に手動で設定）
        
        # オブジェクトコピー配置機能
        self.copy_placement_enabled = True
        self.copy_probability = 0.4  # 40%の確率でコピー配置を試行
        self.max_copy_distance = 5   # コピー元からの最大距離
        
        # ARC-AGI2統計準拠の配置設定
        self.adjacency_avoidance_rate = 0.70  # 隣接回避率（70%）
        self.same_color_avoidance = True  # 同色隣接を完全回避
    
    def _setup_advanced_placement_patterns(self):
        """高度な配置パターンを設定"""
        # レガシーシステムから追加の配置パターン
        self.placement_patterns.update({
            'symmetry': self._symmetry_placement,
            'arc_pattern': self._arc_pattern_placement,
            'structured': self._structured_placement
        })
    
    def place_objects(self, grid: List[List[int]], objects: List[Dict], 
                     pattern: str = 'random') -> bool:
        """オブジェクトをグリッドに配置"""
        # 高度な配置パターンを設定
        self._setup_advanced_placement_patterns()
        
        # 通常の配置を実行
        if pattern in self.placement_patterns:
            success = self.placement_patterns[pattern](grid, objects)
        else:
            success = self._random_placement(grid, objects)
        
        # コピー配置を試行
        if success and self.copy_placement_enabled:
            self._try_copy_placement(grid, objects)
        
        return success
    
    def _random_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """ランダム配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        for obj in objects:
            # オブジェクトのサイズを取得
            obj_width = obj.get('width', 1)
            obj_height = obj.get('height', 1)
            
            # 配置可能な位置を探す
            max_attempts = 100
            for _ in range(max_attempts):
                x = self.rng.randint(0, width - obj_width)
                y = self.rng.randint(0, height - obj_height)
                
                if self._can_place_object(grid, obj, x, y):
                    self._place_object_at_position(grid, obj, x, y)
                    break
        
        return True
    
    def _grid_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """グリッド配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # グリッドサイズを計算
        grid_cols = int(width ** 0.5)
        grid_rows = int(height ** 0.5)
        
        for i, obj in enumerate(objects):
            col = i % grid_cols
            row = i // grid_cols
            
            if row < grid_rows:
                x = col * (width // grid_cols)
                y = row * (height // grid_rows)
                
                if self._can_place_object(grid, obj, x, y):
                    self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _spiral_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """螺旋配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        center_x = width // 2
        center_y = height // 2
        
        for i, obj in enumerate(objects):
            # 螺旋状に配置
            angle = i * 0.5
            radius = i * 2
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _cluster_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """クラスター配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # クラスター数を決定
        num_clusters = min(3, len(objects))
        
        for i, obj in enumerate(objects):
            cluster_id = i % num_clusters
            
            # クラスターの中心位置
            cluster_x = width // num_clusters * cluster_id + width // (num_clusters * 2)
            cluster_y = height // 2
            
            # クラスター周辺に配置
            x = cluster_x + self.rng.randint(-3, 3)
            y = cluster_y + self.rng.randint(-3, 3)
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _border_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """境界配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        for i, obj in enumerate(objects):
            # 境界に配置
            if i % 4 == 0:  # 上辺
                x = self.rng.randint(0, width - obj.get('width', 1))
                y = 0
            elif i % 4 == 1:  # 右辺
                x = width - obj.get('width', 1)
                y = self.rng.randint(0, height - obj.get('height', 1))
            elif i % 4 == 2:  # 下辺
                x = self.rng.randint(0, width - obj.get('width', 1))
                y = height - obj.get('height', 1)
            else:  # 左辺
                x = 0
                y = self.rng.randint(0, height - obj.get('height', 1))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _center_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """中心配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        center_x = width // 2
        center_y = height // 2
        
        for i, obj in enumerate(objects):
            # 中心周辺に配置
            x = center_x + self.rng.randint(-2, 2)
            y = center_y + self.rng.randint(-2, 2)
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    # ============================================================================
    # オブジェクトコピー配置機能
    # ============================================================================
    
    def _try_copy_placement(self, grid: List[List[int]], objects: List[Dict]):
        """コピー配置を試行"""
        if not objects:
            return
        
        # 既に配置されたオブジェクトの位置を記録
        placed_objects = []
        for obj in objects:
            if 'position' in obj:
                placed_objects.append(obj)
        
        if not placed_objects:
            return
        
        # コピー配置を試行
        for obj in objects:
            if self.rng.random() < self.copy_probability:
                source_obj = self.rng.choice(placed_objects)
                self._copy_object_placement(grid, obj, source_obj)
    
    def _copy_object_placement(self, grid: List[List[int]], target_obj: Dict, source_obj: Dict):
        """オブジェクトをコピー配置"""
        if 'position' not in source_obj:
            return
        
        source_x, source_y = source_obj['position']
        
        # コピー元からの距離をランダムに決定
        distance = self.rng.randint(1, self.max_copy_distance)
        angle = self.rng.uniform(0, 2 * np.pi)
        
        # 新しい位置を計算
        offset_x = int(distance * np.cos(angle))
        offset_y = int(distance * np.sin(angle))
        
        new_x = source_x + offset_x
        new_y = source_y + offset_y
        
        # グリッド内に収まるように調整
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        new_x = max(0, min(new_x, width - target_obj.get('width', 1)))
        new_y = max(0, min(new_y, height - target_obj.get('height', 1)))
        
        # 配置可能かチェック
        if self._can_place_object(grid, target_obj, new_x, new_y):
            self._place_object_at_position(grid, target_obj, new_x, new_y)
    
    def _can_place_object(self, grid: List[List[int]], obj: Dict, x: int, y: int) -> bool:
        """オブジェクトが指定位置に配置可能かチェック"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # オブジェクトのピクセルをチェック
        for px, py in obj.get('pixels', [(0, 0)]):
            gx, gy = x + px, y + py
            
            # グリッド境界チェック
            if not (0 <= gx < width and 0 <= gy < height):
                return False
            
            # 既存ピクセルとの衝突チェック
            if grid[gy][gx] != 0:  # 背景色以外
                return False
        
        return True
    
    def _place_object_at_position(self, grid: List[List[int]], obj: Dict, x: int, y: int):
        """オブジェクトを指定位置に配置"""
        color = obj.get('color', 1)
        
        # オブジェクトのピクセルを配置
        for px, py in obj.get('pixels', [(0, 0)]):
            gx, gy = x + px, y + py
            grid[gy][gx] = color
        
        # 位置を記録
        obj['position'] = (x, y)
    
    def choose_pattern(self, num_objects: int, grid_size: tuple, requirements: Dict) -> str:
        """配置パターンを選択"""
        # オブジェクト数とグリッドサイズに基づいてパターンを選択
        if num_objects <= 2:
            return self.rng.choice(['random', 'center'])
        elif num_objects <= 5:
            return self.rng.choice(['random', 'grid', 'cluster'])
        else:
            return self.rng.choice(['random', 'grid', 'spiral', 'border'])
    
    # ============================================================================
    # 高度な配置パターン（レガシーシステムから移植）
    # ============================================================================
    
    def _symmetry_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """対称配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # 対称軸を決定
        symmetry_axis = self.rng.choice(['horizontal', 'vertical', 'diagonal'])
        
        for i, obj in enumerate(objects):
            if symmetry_axis == 'horizontal':
                # 水平対称
                x = self.rng.randint(0, width - obj.get('width', 1))
                y = self.rng.randint(0, height // 2)
                
                if self._can_place_object(grid, obj, x, y):
                    self._place_object_at_position(grid, obj, x, y)
                    
                    # 対称位置に配置
                    mirror_y = height - 1 - y
                    if self._can_place_object(grid, obj, x, mirror_y):
                        self._place_object_at_position(grid, obj, x, mirror_y)
            
            elif symmetry_axis == 'vertical':
                # 垂直対称
                x = self.rng.randint(0, width // 2)
                y = self.rng.randint(0, height - obj.get('height', 1))
                
                if self._can_place_object(grid, obj, x, y):
                    self._place_object_at_position(grid, obj, x, y)
                    
                    # 対称位置に配置
                    mirror_x = width - 1 - x
                    if self._can_place_object(grid, obj, mirror_x, y):
                        self._place_object_at_position(grid, obj, mirror_x, y)
            
            else:  # diagonal
                # 対角対称
                x = self.rng.randint(0, width - obj.get('width', 1))
                y = self.rng.randint(0, height - obj.get('height', 1))
                
                if self._can_place_object(grid, obj, x, y):
                    self._place_object_at_position(grid, obj, x, y)
                    
                    # 対角対称位置に配置
                    mirror_x = width - 1 - x
                    mirror_y = height - 1 - y
                    if self._can_place_object(grid, obj, mirror_x, mirror_y):
                        self._place_object_at_position(grid, obj, mirror_x, mirror_y)
        
        return True
    
    def _arc_pattern_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """ARC-AGI2スタイルの配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # ARC-AGI2統計に基づく配置
        for i, obj in enumerate(objects):
            # 統計的に適切な位置に配置
            if i % 3 == 0:
                # 左上
                x = self.rng.randint(0, width // 3)
                y = self.rng.randint(0, height // 3)
            elif i % 3 == 1:
                # 中央
                x = self.rng.randint(width // 3, 2 * width // 3)
                y = self.rng.randint(height // 3, 2 * height // 3)
            else:
                # 右下
                x = self.rng.randint(2 * width // 3, width)
                y = self.rng.randint(2 * height // 3, height)
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _structured_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """構造化配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        # 構造化パターンを選択
        pattern_type = self.rng.choice(['concentric', 'spiral', 'radial'])
        
        if pattern_type == 'concentric':
            return self._concentric_placement(grid, objects)
        elif pattern_type == 'spiral':
            return self._spiral_placement(grid, objects)
        else:  # radial
            return self._radial_placement(grid, objects)
    
    def _concentric_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """同心円配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        center_x = width // 2
        center_y = height // 2
        
        for i, obj in enumerate(objects):
            # 同心円状に配置
            radius = (i + 1) * 2
            angle = i * 0.8
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
    
    def _radial_placement(self, grid: List[List[int]], objects: List[Dict]) -> bool:
        """放射状配置"""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        center_x = width // 2
        center_y = height // 2
        
        for i, obj in enumerate(objects):
            # 放射状に配置
            angle = i * (2 * np.pi / len(objects))
            radius = min(width, height) // 4
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # グリッド内に収まるように調整
            x = max(0, min(x, width - obj.get('width', 1)))
            y = max(0, min(y, height - obj.get('height', 1)))
            
            if self._can_place_object(grid, obj, x, y):
                self._place_object_at_position(grid, obj, x, y)
        
        return True
