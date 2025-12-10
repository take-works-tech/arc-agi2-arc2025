"""
コアオブジェクト生成器
"""
import random
import numpy as np
from typing import Dict, List, Optional, Tuple


def _add_lock_flags(obj_dict: Dict) -> Dict:
    """オブジェクト辞書に変更禁止フラグと座標情報を追加（デフォルト値: False, None）

    Args:
        obj_dict: オブジェクト辞書

    Returns:
        フラグと座標情報が追加されたオブジェクト辞書
    """
    # 既にフラグがある場合は上書きしない（既存の値を保持）
    obj_dict.setdefault('lock_color_change', False)
    obj_dict.setdefault('lock_shape_change', False)
    obj_dict.setdefault('lock_position_change', False)
    # 座標情報を追加（既存の値がある場合は保持）
    obj_dict.setdefault('x', None)
    obj_dict.setdefault('y', None)
    return obj_dict


class CoreObjectBuilder:
    """コアオブジェクト生成器"""

    def __init__(self, seed=None):
        """初期化"""
        self.rng = random.Random(seed)

        # オブジェクト形状のテンプレート（レガシーシステムから拡充）
        self.shape_templates = {
            'rectangle': self._generate_rectangle,
            'line': self._generate_line,
            'l_shape': self._generate_l_shape,
            'cross': self._generate_cross,
            'diagonal': self._generate_diagonal,
            'triangle': self._generate_triangle,
            'circle': self._generate_circle,
            'hollow_rectangle': self._generate_hollow_rectangle,
            'random_pattern': self._generate_random_pattern,
            # レガシーシステムから追加
            'u_shape': self._generate_u_shape,
            'h_shape': self._generate_h_shape,
            'z_shape': self._generate_z_shape,
            't_shape': self._generate_t_shape,
            'diagonal_45': self._generate_diagonal_45,
            # 新規追加
            'arrow': self._generate_arrow,
            'diamond': self._generate_diamond,
            'stairs': self._generate_stairs,
            'zigzag': self._generate_zigzag
        }

        # オブジェクト合成機能
        self.synthesis_enabled = True
        self.synthesis_probability = 0.3  # 30%の確率で合成を試行（基本値）
        # 複雑度に応じた合成確率のマッピング
        # 複雑度が高いほど合成確率が高くなる
        self.synthesis_probability_by_complexity = {
            1: 0.1,  # 10%
            2: 0.2,  # 20%
            3: 0.3,  # 30%
            4: 0.4,  # 40%
            5: 0.5,  # 50%
            6: 0.6,  # 60%
            7: 0.7,  # 70%
            8: 0.8,  # 80%
        }

    def build_object_for_spec(self, spec: Dict, color: int,
                             background_color: int = 0) -> Optional[Dict]:
        """条件仕様を満たすオブジェクトを生成"""
        spec_type = spec.get('type')

        if spec_type == 'size':
            return self._build_object_for_size(spec, color)
        elif spec_type == 'width':
            return self._build_object_for_width(spec, color)
        elif spec_type == 'height':
            return self._build_object_for_height(spec, color)
        elif spec_type == 'color':
            return self._build_object_for_color(spec, color)
        elif spec_type == 'holes':
            return self._build_object_for_holes(spec, color)
        else:
            return self._generate_simple_object(color)

    def _build_object_for_size(self, spec: Dict, color: int) -> Dict:
        """サイズ条件を満たすオブジェクトを生成"""
        operator = spec.get('operator', 'greater')
        target_size = spec.get('value', 5)

        if operator == 'greater':
            size = target_size + self.rng.randint(1, 10)
        elif operator == 'less':
            size = max(1, target_size - self.rng.randint(1, min(3, target_size-1)))
        else:
            size = target_size

        if size == 1:
            return self._generate_single_pixel(color)
        elif size <= 4:
            return self._generate_small_object(color, size)
        else:
            # 形状のバリエーションを追加（60%でrectangle、40%で他の形状）
            if self.rng.random() < 0.6:
                # rectangle（正方形または長方形）
                if self.rng.random() < 0.5:
                    rect_size = self.rng.randint(2, min(8, size//2 + 2))
                    return self._generate_rectangle(color, rect_size, rect_size, filled=True)
                else:
                    width = self.rng.randint(2, min(8, size//2 + 2))
                    height = self.rng.randint(2, min(8, size//2 + 2))
                    return self._generate_rectangle(color, width, height, filled=True)
            else:
                # その他の形状をランダムに選択
                shape_choice = self.rng.choice(['line', 'l_shape', 'cross', 'circle', 'triangle'])
                if shape_choice == 'line':
                    length = min(size, 8)
                    direction = self.rng.choice(['horizontal', 'vertical'])
                    return self._generate_line(color, length, direction)
                else:
                    return self.shape_templates[shape_choice](color)

    def _build_object_for_width(self, spec: Dict, color: int) -> Dict:
        """幅条件を満たすオブジェクトを生成"""
        operator = spec.get('operator', 'greater')
        target_width = spec.get('value', 5)

        if operator == 'greater':
            width = target_width + self.rng.randint(1, 8)
        elif operator == 'less':
            width = max(1, target_width - self.rng.randint(1, min(3, target_width-1)))
        else:
            width = target_width

        # 形状のバリエーションを追加（70%でrectangle、30%でline）
        if self.rng.random() < 0.7:
            height = self.rng.randint(2, 10)
            return self._generate_rectangle(color, width, height, filled=True)
        else:
            # line（水平線）を生成
            return self._generate_line(color, width, 'horizontal')

    def _build_object_for_height(self, spec: Dict, color: int) -> Dict:
        """高さ条件を満たすオブジェクトを生成"""
        operator = spec.get('operator', 'greater')
        target_height = spec.get('value', 5)

        if operator == 'greater':
            height = target_height + self.rng.randint(1, 8)
        elif operator == 'less':
            height = max(1, target_height - self.rng.randint(1, min(3, target_height-1)))
        else:
            height = target_height

        # 形状のバリエーションを追加（70%でrectangle、30%でline）
        if self.rng.random() < 0.7:
            width = self.rng.randint(2, 10)
            return self._generate_rectangle(color, width, height, filled=True)
        else:
            # line（垂直線）を生成
            return self._generate_line(color, height, 'vertical')

    def _build_object_for_color(self, spec: Dict, color: int) -> Dict:
        """色条件を満たすオブジェクトを生成"""
        operator = spec.get('operator', 'equal')
        target_color = spec.get('value', 1)

        if operator == 'equal':
            color = target_color
        elif operator == 'not_equal':
            available_colors = [c for c in range(10) if c != target_color]
            color = self.rng.choice(available_colors) if available_colors else 1

        return self._generate_simple_object(color)

    def _build_object_for_holes(self, spec: Dict, color: int) -> Dict:
        """穴の数条件を満たすオブジェクトを生成"""
        operator = spec.get('operator', 'greater')
        target_holes = spec.get('value', 1)

        if operator == 'greater':
            holes = target_holes + self.rng.randint(1, 3)
        elif operator == 'less':
            holes = max(0, target_holes - self.rng.randint(1, min(2, target_holes)))
        else:
            holes = target_holes

        return self._generate_object_with_holes(color, holes)

    def _calculate_adaptive_size_range(self, base_min: int, base_max: int,
                                       grid_w: Optional[int] = None,
                                       grid_h: Optional[int] = None,
                                       total_num_objects: Optional[int] = None) -> Tuple[int, int]:
        """グリッドサイズとオブジェクト数に応じてサイズ範囲を動的に調整

        Args:
            base_min: ベース最小値
            base_max: ベース最大値
            grid_w: グリッド幅（オプション）
            grid_h: グリッド高さ（オプション）
            total_num_objects: 総オブジェクト数（オプション）

        Returns:
            (調整後の最小値, 調整後の最大値)
        """
        min_size = base_min
        max_size = base_max

        # グリッドサイズに応じた調整（密度をさらに下げるため、拡大をさらに控えめに）
        if grid_w is not None and grid_h is not None:
            min_grid_dim = min(grid_w, grid_h)
            if min_grid_dim <= 10:
                # 小さいグリッド: 範囲を維持
                pass
            elif min_grid_dim <= 20:
                # 中程度のグリッド: 最大値を1.2倍に拡大（1.3倍→1.2倍に削減）
                max_size = int(base_max * 1.2)
            else:
                # 大きいグリッド: 最大値を1.4倍に拡大（1.6倍→1.4倍に削減）
                max_size = int(base_max * 1.4)

            # グリッドサイズを超えないように制限
            max_size = min(max_size, min_grid_dim - 1)

        # オブジェクト数に応じた調整（密度をさらに下げるため、拡大をさらに控えめに）
        if total_num_objects is not None:
            if total_num_objects < 5:
                # オブジェクト数が非常に少ない場合: 最大値を1.4倍に拡大（1.6倍→1.4倍に削減）
                max_size = int(max_size * 1.4)
            elif total_num_objects < 10:
                # オブジェクト数が少ない場合: 最大値を1.2倍に拡大（1.3倍→1.2倍に削減）
                max_size = int(max_size * 1.2)
            # オブジェクト数が多い場合は範囲を維持（密度を下げるため）

        # 最小値は維持
        max_size = max(max_size, min_size)

        return min_size, max_size

    def _generate_simple_object(self, color: int, grid_size: Optional[Tuple[int, int]] = None,
                               total_num_objects: Optional[int] = None, max_size: int = None) -> Dict:
        """基本的なオブジェクトを生成（本格実装）

        合成オブジェクト生成を試行し、失敗した場合は通常の形状生成を実行します。
        重み付け選択を使用して、処理速度を考慮した確率分布で形状を選択します。

        Args:
            color: 色
            grid_size: グリッドサイズ（オプション、幅、高さのタプル）
            total_num_objects: 総オブジェクト数（オプション）
            max_size: 最大サイズ制約（指定された場合、生成されるオブジェクトのサイズをこの値以下に制限）
        """
        # 合成オブジェクト生成を試行
        if self.synthesis_enabled and self.rng.random() < self.synthesis_probability:
            synthesized = self.generate_synthesized_object(color, max_size=max_size)
            if synthesized:
                return synthesized

        # 重み付け選択を使用（処理速度を考慮した確率分布）
        shape_type_weights = [
            ('rectangle', 0.14),      # 14% - 基本形状（0.15 → 0.14）
            ('line', 0.12),            # 12% - 基本形状（0.13 → 0.12）
            ('cross', 0.08),           # 8% - 中程度の形状（0.09 → 0.08）
            ('circle', 0.08),          # 8% - 中程度の形状（0.09 → 0.08）
            ('l_shape', 0.08),        # 8% - 中程度の形状（0.09 → 0.08）
            ('triangle', 0.08),       # 8% - 中程度の形状（0.09 → 0.08）
            ('single_pixel', 0.08),   # 8% - 中程度の形状（0.09 → 0.08）
            ('arrow', 0.04),          # 4% - 新規追加（最高優先度）
            ('diamond', 0.04),        # 4% - 新規追加（最高優先度）
            ('stairs', 0.03),         # 3% - 新規追加（高優先度）
            ('zigzag', 0.03),         # 3% - 新規追加（高優先度）
            ('hollow_rectangle', 0.04), # 4% - 中程度の形状（0.05 → 0.04）
            ('diagonal', 0.03),       # 3% - 特殊形状（0.04 → 0.03）
            ('diagonal_45', 0.03),    # 3% - 特殊形状（0.04 → 0.03）
            ('t_shape', 0.03),        # 3% - 特殊形状（0.04 → 0.03）
            ('random_pattern', 0.03), # 3% - 処理速度を考慮して低め（連結性チェックが重い）
            ('u_shape', 0.02),        # 2% - 複雑な形状
            ('h_shape', 0.01),        # 1% - 複雑な形状
            ('z_shape', 0.01),        # 1% - 複雑な形状
        ]

        # 重みに基づいてランダムに選択
        rand = self.rng.random()
        cumulative = 0.0
        selected_shape_type = None
        for st, weight in shape_type_weights:
            cumulative += weight
            if rand < cumulative:
                selected_shape_type = st
                break

        # フォールバック: 選択されなかった場合はrectangleを使用
        if selected_shape_type is None:
            selected_shape_type = 'rectangle'

        shape_type = selected_shape_type

        # グリッドサイズ情報を取得
        if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
            grid_w = grid_size[0]
            grid_h = grid_size[1]
        else:
            grid_w = None
            grid_h = None

        # max_size制約を適用するヘルパー関数
        def _apply_max_size_constraint(base_max, max_size_constraint):
            if max_size_constraint is not None:
                return min(base_max, max_size_constraint)
            return base_max

        # 引数を適切に設定してオブジェクトを生成（動的サイズ範囲を使用）
        if shape_type == 'rectangle':
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(2, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            width = self.rng.randint(min_size, max(min_size, constrained_max_size))
            height = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self._generate_rectangle(color, width, height, filled=True)
        elif shape_type == 'line':
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 8, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            length = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self._generate_line(color, length)
        elif shape_type == 'random_pattern':
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            width = self.rng.randint(min_size, max(min_size, constrained_max_size))
            height = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, width, height)
        elif shape_type == 't_shape':
            rotation = self.rng.choice([0, 90, 180, 270])
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, size, rotation)
        elif shape_type == 'diagonal_45':
            direction = self.rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, size, direction)
        elif shape_type == 'arrow':
            direction = self.rng.choice(['up', 'down', 'left', 'right'])
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 8, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, size, direction)
        elif shape_type == 'diamond':
            filled = self.rng.random() < 0.7  # 70%の確率で塗りつぶし
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(2, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, size, filled)
        elif shape_type == 'stairs':
            min_steps, adaptive_max_steps = self._calculate_adaptive_size_range(2, 5, grid_w, grid_h, total_num_objects)
            constrained_max_steps = _apply_max_size_constraint(adaptive_max_steps, max_size)
            steps = self.rng.randint(min_steps, max(min_steps, constrained_max_steps))
            min_step_size, adaptive_max_step_size = self._calculate_adaptive_size_range(2, 4, grid_w, grid_h, total_num_objects)
            constrained_max_step_size = _apply_max_size_constraint(adaptive_max_step_size, max_size)
            step_size = self.rng.randint(min_step_size, max(min_step_size, constrained_max_step_size))
            return self.shape_templates[shape_type](color, steps, step_size)
        elif shape_type == 'zigzag':
            min_length, adaptive_max_length = self._calculate_adaptive_size_range(4, 10, grid_w, grid_h, total_num_objects)
            constrained_max_length = _apply_max_size_constraint(adaptive_max_length, max_size)
            length = self.rng.randint(min_length, max(min_length, constrained_max_length))
            # 振幅はグリッドサイズに応じて調整
            if grid_w is not None and grid_h is not None:
                min_grid_dim = min(grid_w, grid_h)
                if min_grid_dim <= 10:
                    max_amplitude = 3
                elif min_grid_dim <= 20:
                    max_amplitude = 4
                else:
                    max_amplitude = 5
            else:
                max_amplitude = 3
            amplitude = self.rng.randint(1, max_amplitude)
            return self.shape_templates[shape_type](color, length, amplitude)
        elif shape_type in ['u_shape', 'h_shape', 'z_shape']:
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self.shape_templates[shape_type](color, size)
        elif shape_type == 'l_shape':
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(2, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            width = self.rng.randint(min_size, max(min_size, constrained_max_size))
            height = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self._generate_l_shape(color, width, height)
        elif shape_type == 'triangle':
            min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, total_num_objects)
            constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
            size = self.rng.randint(min_size, max(min_size, constrained_max_size))
            return self._generate_triangle(color, size)
        elif shape_type == 'single_pixel':
            return self._generate_single_pixel(color)
        else:
            # 単一引数の形状テンプレート
            return self.shape_templates[shape_type](color)

    def _generate_single_pixel(self, color: int) -> Dict:
        """1ピクセルの点形状を生成"""
        return _add_lock_flags({
            'pixels': [(0, 0)],
            'color': color,
            'width': 1,
            'height': 1,
            'area': 1,
            'shape_type': 'point'
        })

    def _generate_small_object(self, color: int, size: int) -> Dict:
        """小さなオブジェクトを生成"""
        if size == 2:
            pixels = [(0, 0), (1, 0)]
            return _add_lock_flags({
                'pixels': pixels,
                'color': color,
                'width': 2,
                'height': 1,
                'area': 2,
                'shape_type': 'line'
            })
        elif size == 3:
            if self.rng.random() < 0.5:
                pixels = [(0, 0), (1, 0), (2, 0)]
                return _add_lock_flags({
                    'pixels': pixels,
                    'color': color,
                    'width': 3,
                    'height': 1,
                    'area': 3,
                    'shape_type': 'line'
                })
            else:
                pixels = [(0, 0), (1, 0), (0, 1)]
                return _add_lock_flags({
                    'pixels': pixels,
                    'color': color,
                    'width': 2,
                    'height': 2,
                    'area': 3,
                    'shape_type': 'l_shape'
                })
        else:  # size == 4
            pixels = [(0, 0), (1, 0), (0, 1), (1, 1)]
            return _add_lock_flags({
                'pixels': pixels,
                'color': color,
                'width': 2,
                'height': 2,
                'area': 4,
                'shape_type': 'square'
            })

    def _generate_rectangle(self, color: int, width: int, height: int, filled: bool = True) -> Dict:
        """矩形を生成"""
        pixels = []

        if filled:
            for y in range(height):
                for x in range(width):
                    pixels.append((x, y))
        else:
            for y in range(height):
                for x in range(width):
                    if x == 0 or x == width-1 or y == 0 or y == height-1:
                        pixels.append((x, y))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'rectangle'
        })

    def _generate_line(self, color: int, length: int, direction: str = 'horizontal') -> Dict:
        """線を生成"""
        pixels = []

        if direction == 'horizontal':
            for x in range(length):
                pixels.append((x, 0))
            return _add_lock_flags({
                'pixels': pixels,
                'color': color,
                'width': length,
                'height': 1,
                'area': length,
                'shape_type': 'line'
            })
        else:  # vertical
            for y in range(length):
                pixels.append((0, y))
            return _add_lock_flags({
                'pixels': pixels,
                'color': color,
                'width': 1,
                'height': length,
                'area': length,
                'shape_type': 'line'
            })

    def _generate_l_shape(self, color: int, width: int = None, height: int = None) -> Dict:
        """L字形状を生成

        Args:
            color: オブジェクトの色
            width: 幅（Noneの場合はランダム、最小2）
            height: 高さ（Noneの場合はランダム、最小2）
        """
        # デフォルト値の設定
        if width is None:
            width = self.rng.randint(2, 6)
        if height is None:
            height = self.rng.randint(2, 6)

        # 最小サイズを保証
        width = max(2, width)
        height = max(2, height)

        pixels = []
        # 水平部分（上側）
        for x in range(width):
            pixels.append((x, 0))
        # 垂直部分（左側、最初のピクセルは既に追加済み）
        for y in range(1, height):
            pixels.append((0, y))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'l_shape'
        })

    def _generate_cross(self, color: int, size: int = 5) -> Dict:
        """十字形状を生成"""
        pixels = []
        center = size // 2

        # 水平線
        for x in range(size):
            pixels.append((x, center))

        # 垂直線
        for y in range(size):
            if (center, y) not in pixels:
                pixels.append((center, y))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': size,
            'height': size,
            'area': len(pixels),
            'shape_type': 'cross'
        })

    def _generate_diagonal(self, color: int) -> Dict:
        """対角線形状を生成"""
        length = self.rng.randint(3, 8)
        pixels = []

        for i in range(length):
            pixels.append((i, i))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': length,
            'height': length,
            'area': length,
            'shape_type': 'diagonal'
        })

    def _generate_triangle(self, color: int, size: int = None, triangle_type: str = None) -> Dict:
        """三角形形状を生成

        Args:
            color: オブジェクトの色
            size: サイズ（Noneの場合はランダム、最小3）
            triangle_type: 三角形のタイプ（Noneの場合はランダム選択）
                - 'right': 直角三角形（左下から右上へ）
                - 'isosceles': 二等辺三角形（左右対称）
                - 'equilateral': 正三角形（可能な限り近似）
        """
        # デフォルト値の設定
        if size is None:
            size = self.rng.randint(3, 6)
        size = max(3, size)  # 最小サイズを保証

        if triangle_type is None:
            # ランダムに三角形タイプを選択
            rand = self.rng.random()
            if rand < 0.4:  # 40%: 直角三角形
                triangle_type = 'right'
            elif rand < 0.8:  # 40%: 二等辺三角形
                triangle_type = 'isosceles'
            else:  # 20%: 正三角形
                triangle_type = 'equilateral'

        pixels = []

        if triangle_type == 'right':
            # 直角三角形（左下から右上へ）
            for y in range(size):
                for x in range(y + 1):
                    pixels.append((x, y))
            width = size
            height = size
        elif triangle_type == 'isosceles':
            # 二等辺三角形（左右対称、頂点が上）
            center_x = size // 2
            for y in range(size):
                # 各y座標での幅を計算（yが大きいほど幅が広い）
                half_width = y + 1
                start_x = center_x - half_width + 1
                end_x = center_x + half_width
                for x in range(max(0, start_x), min(size, end_x)):
                    pixels.append((x, y))
            width = size
            height = size
        elif triangle_type == 'equilateral':
            # 正三角形（可能な限り近似）
            # 高さをsize、底辺を2*size-1にすることで、正三角形に近い形状を作成
            height = size
            base_width = 2 * size - 1
            center_x = base_width // 2
            for y in range(height):
                # 各y座標での幅を計算
                half_width = height - y
                start_x = center_x - half_width + 1
                end_x = center_x + half_width
                for x in range(max(0, start_x), min(base_width, end_x)):
                    pixels.append((x, y))
            width = base_width
            height = height
        else:
            # フォールバック: 直角三角形
            for y in range(size):
                for x in range(y + 1):
                    pixels.append((x, y))
            width = size
            height = size

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': f'triangle_{triangle_type}'
        })

    def _generate_circle(self, color: int) -> Dict:
        """円形状を生成（近似）"""
        radius = self.rng.randint(2, 5)
        pixels = []

        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if x*x + y*y <= radius*radius:
                    pixels.append((x + radius, y + radius))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': 2 * radius + 1,
            'height': 2 * radius + 1,
            'area': len(pixels),
            'shape_type': 'circle'
        })

    def _generate_hollow_rectangle(self, color: int) -> Dict:
        """中空矩形を生成"""
        width = self.rng.randint(4, 8)
        height = self.rng.randint(4, 8)
        return self._generate_rectangle(color, width, height, filled=False)

    def _generate_object_with_holes(self, color: int, hole_count: int) -> Dict:
        """穴を持つオブジェクトを生成

        Args:
            color: オブジェクトの色
            hole_count: 穴の数

        Returns:
            穴を持つオブジェクト（3~6x3~6の矩形をベース、実際にピクセルレベルで穴をあける）
        """
        # 3~6x3~6の矩形をベースとして生成
        width = self.rng.randint(3, 6)
        height = self.rng.randint(3, 6)
        base_obj = self._generate_rectangle(color, width, height, filled=True)

        if hole_count > 0:
            # 実際にピクセルレベルで穴をあける
            pixels = base_obj.get('pixels', [])
            if pixels:
                # 矩形の内部（境界を除く）から穴の位置を決定
                # 内部領域: x in [1, width-2], y in [1, height-2]
                inner_positions = []
                for x in range(1, width - 1):
                    for y in range(1, height - 1):
                        inner_positions.append((x, y))

                # 穴の位置をランダムに選択（重複なし）
                if len(inner_positions) > 0:
                    # 穴の数が内部領域の数より多い場合は調整
                    actual_hole_count = min(hole_count, len(inner_positions))
                    if actual_hole_count > 0 and len(inner_positions) > 0:
                        hole_positions = self.rng.sample(inner_positions, actual_hole_count)
                    else:
                        hole_positions = []

                    # 選択された位置のピクセルを削除
                    pixels_set = set(pixels)
                    for hole_pos in hole_positions:
                        pixels_set.discard(hole_pos)

                    # ピクセルリストを更新
                    base_obj['pixels'] = list(pixels_set)
                    base_obj['area'] = len(base_obj['pixels'])

                    # バウンディングボックスを再計算（穴をあけた後も同じサイズ）
                    # 実際のピクセル範囲を計算
                    if base_obj['pixels']:
                        min_x = min(x for x, y in base_obj['pixels'])
                        max_x = max(x for x, y in base_obj['pixels'])
                        min_y = min(y for x, y in base_obj['pixels'])
                        max_y = max(y for x, y in base_obj['pixels'])

                        # オフセットを0,0にする
                        base_obj['pixels'] = [(x - min_x, y - min_y) for x, y in base_obj['pixels']]
                        base_obj['width'] = max_x - min_x + 1
                        base_obj['height'] = max_y - min_y + 1
                    else:
                        # ピクセルが空になった場合は1ピクセルをフォールバック
                        base_obj['pixels'] = [(0, 0)]
                        base_obj['width'] = 1
                        base_obj['height'] = 1
                        base_obj['area'] = 1

            base_obj['holes'] = hole_count
            base_obj['shape_type'] = 'hollow_rectangle'

        return base_obj

    # ============================================================================
    # オブジェクトパターン合成機能
    # ============================================================================

    def synthesize_objects(self, obj1: Dict, obj2: Dict, synthesis_type: str) -> Optional[Dict]:
        """オブジェクトを合成して新しい形状を作成（統合版）

        Args:
            obj1: 第1オブジェクト
            obj2: 第2オブジェクト
            synthesis_type: 合成タイプ ('intersection', 'subtract', 'union')

        Returns:
            合成されたオブジェクト（失敗時はNone）
        """
        if not self.synthesis_enabled:
            return None

        try:
            # synthesis_typeをcombine_objectsの形式に変換
            if synthesis_type == 'subtract':
                synthesis_type = 'difference'

            # ランダムなオフセットを設定
            offset_x = self.rng.randint(-2, 2)
            offset_y = self.rng.randint(-2, 2)

            # 統合されたcombine_objectsを使用
            return self.combine_objects(obj1, obj2, synthesis_type, offset_x, offset_y)
        except Exception as e:
            print(f"Object synthesis failed: {e}")
            return None


    def generate_synthesized_object(self, color: int, complexity: int = None, max_size: int = None) -> Dict:
        """合成オブジェクトを生成

        Args:
            color: オブジェクトの色
            complexity: 複雑度（1-8、指定された場合は複雑度に応じた合成確率を使用）
            max_size: 最大サイズ制約（指定された場合、各ベースオブジェクトのサイズをこの値以下に制限）
        """
        # 複雑度に応じた合成確率を決定
        if complexity is not None and complexity in self.synthesis_probability_by_complexity:
            synthesis_prob = self.synthesis_probability_by_complexity[complexity]
        else:
            synthesis_prob = self.synthesis_probability

        if not self.synthesis_enabled or self.rng.random() > synthesis_prob:
            return self._generate_simple_object(color, max_size=max_size)

        # 2つのベースオブジェクトを生成
        shape_names = list(self.shape_templates.keys())
        shape1 = self.rng.choice(shape_names)
        shape2 = self.rng.choice(shape_names)

        # max_size制約を適用
        def _apply_max_size_constraint(base_max, max_size_constraint):
            if max_size_constraint is not None:
                return min(base_max, max_size_constraint)
            return base_max

        # 引数を適切に設定してオブジェクトを生成
        if shape1 == 'rectangle':
            max_w = _apply_max_size_constraint(6, max_size)
            max_h = _apply_max_size_constraint(6, max_size)
            obj1 = self._generate_rectangle(color, self.rng.randint(2, max(2, max_w)), self.rng.randint(2, max(2, max_h)), filled=True)
        elif shape1 == 'line':
            max_len = _apply_max_size_constraint(8, max_size)
            obj1 = self._generate_line(color, self.rng.randint(3, max(3, max_len)))
        elif shape1 == 'random_pattern':
            max_w = _apply_max_size_constraint(6, max_size)
            max_h = _apply_max_size_constraint(6, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_w)), self.rng.randint(3, max(3, max_h)))
        elif shape1 == 't_shape':
            rotation = self.rng.choice([0, 90, 180, 270])
            max_s = _apply_max_size_constraint(6, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_s)), rotation)
        elif shape1 == 'diagonal_45':
            direction = self.rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
            max_s = _apply_max_size_constraint(6, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_s)), direction)
        elif shape1 in ['u_shape', 'h_shape', 'z_shape']:
            max_s = _apply_max_size_constraint(6, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_s)))
        elif shape1 == 'cross':
            max_s = _apply_max_size_constraint(7, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_s)))
        elif shape1 == 'arrow':
            direction = self.rng.choice(['up', 'down', 'left', 'right'])
            max_s = _apply_max_size_constraint(8, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(3, max(3, max_s)), direction)
        elif shape1 == 'diamond':
            filled = self.rng.random() < 0.7  # 70%の確率で塗りつぶし
            max_s = _apply_max_size_constraint(6, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(2, max(2, max_s)), filled)
        elif shape1 == 'stairs':
            max_steps = _apply_max_size_constraint(5, max_size)
            max_step_size = _apply_max_size_constraint(4, max_size)
            steps = self.rng.randint(2, max(2, max_steps))
            step_size = self.rng.randint(2, max(2, max_step_size))
            obj1 = self.shape_templates[shape1](color, steps, step_size)
        elif shape1 == 'zigzag':
            max_len = _apply_max_size_constraint(10, max_size)
            obj1 = self.shape_templates[shape1](color, self.rng.randint(4, max(4, max_len)), self.rng.randint(1, 3))
        else:
            obj1 = self.shape_templates[shape1](color)

        if shape2 == 'rectangle':
            max_w = _apply_max_size_constraint(6, max_size)
            max_h = _apply_max_size_constraint(6, max_size)
            obj2 = self._generate_rectangle(color, self.rng.randint(2, max(2, max_w)), self.rng.randint(2, max(2, max_h)), filled=True)
        elif shape2 == 'line':
            max_len = _apply_max_size_constraint(8, max_size)
            obj2 = self._generate_line(color, self.rng.randint(3, max(3, max_len)))
        elif shape2 == 'random_pattern':
            max_w = _apply_max_size_constraint(6, max_size)
            max_h = _apply_max_size_constraint(6, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_w)), self.rng.randint(3, max(3, max_h)))
        elif shape2 == 't_shape':
            rotation = self.rng.choice([0, 90, 180, 270])
            max_s = _apply_max_size_constraint(6, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_s)), rotation)
        elif shape2 == 'diagonal_45':
            direction = self.rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
            max_s = _apply_max_size_constraint(6, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_s)), direction)
        elif shape2 in ['u_shape', 'h_shape', 'z_shape']:
            max_s = _apply_max_size_constraint(6, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_s)))
        elif shape2 == 'cross':
            max_s = _apply_max_size_constraint(7, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_s)))
        elif shape2 == 'arrow':
            direction = self.rng.choice(['up', 'down', 'left', 'right'])
            max_s = _apply_max_size_constraint(8, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(3, max(3, max_s)), direction)
        elif shape2 == 'diamond':
            filled = self.rng.random() < 0.7  # 70%の確率で塗りつぶし
            max_s = _apply_max_size_constraint(6, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(2, max(2, max_s)), filled)
        elif shape2 == 'stairs':
            max_steps = _apply_max_size_constraint(5, max_size)
            max_step_size = _apply_max_size_constraint(4, max_size)
            steps = self.rng.randint(2, max(2, max_steps))
            step_size = self.rng.randint(2, max(2, max_step_size))
            obj2 = self.shape_templates[shape2](color, steps, step_size)
        elif shape2 == 'zigzag':
            max_len = _apply_max_size_constraint(10, max_size)
            obj2 = self.shape_templates[shape2](color, self.rng.randint(4, max(4, max_len)), self.rng.randint(1, 3))
        else:
            obj2 = self.shape_templates[shape2](color)

        # 合成タイプを選択
        synthesis_types = ['intersection', 'difference', 'union']
        synthesis_type = self.rng.choice(synthesis_types)

        # ランダムなオフセットを設定
        offset_x = self.rng.randint(-2, 2)
        offset_y = self.rng.randint(-2, 2)

        # 合成を実行
        synthesized = self.combine_objects(obj1, obj2, synthesis_type, offset_x, offset_y)

        return synthesized if synthesized else self._generate_simple_object(color, max_size=max_size)

    # ============================================================================
    # 完全ランダムパターン生成機能
    # ============================================================================

    def _generate_random_pattern(self, color: int, width: int = None, height: int = None, density: float = None, connectivity: int = None) -> Dict:
        """完全ランダムな形状を生成

        Args:
            color: 色
            width: バウンディングボックスの幅（Noneの場合はランダム）
            height: バウンディングボックスの高さ（Noneの場合はランダム）
            density: ピクセル密度（0.0-1.0、Noneの場合はランダム）
                - 0.2: 疎（20%のセルを埋める）
                - 0.5: 中程度（50%のセルを埋める）
                - 0.9: 密（90%のセルを埋める）

        Returns:
            ランダムな形状のオブジェクト

        形状の例:
        █ █      ██       █  █
         █   or  █ █  or   ██
        ██       ██       █ █
        """
        # デフォルト値の設定
        if width is None:
            width = self.rng.randint(3, 8)
        if height is None:
            height = self.rng.randint(3, 8)
        if density is None:
            density = 0.2 + self.rng.random() * 0.7  # 20%〜90%

        pixels = []

        # サイズが小さすぎる場合は最小サイズを保証（無限ループ防止）
        total_cells = width * height
        if total_cells < 3:
            # 1x1や2x1などの場合、最小3x3に拡大
            width = max(3, width)
            height = max(3, height)
            total_cells = width * height

        # 各セルをランダムに埋める
        for y in range(height):
            for x in range(width):
                if self.rng.random() < density:
                    pixels.append((x, y))

        # 最低3ピクセルは保証
        if len(pixels) < 3:
            # ランダムに3ピクセルを追加（可能な位置が十分にある場合）
            max_attempts = min(100, total_cells * 10)  # 無限ループ防止
            attempts = 0
            while len(pixels) < 3 and attempts < max_attempts:
                x = self.rng.randint(0, width - 1)
                y = self.rng.randint(0, height - 1)
                if (x, y) not in pixels:
                    pixels.append((x, y))
                attempts += 1

            # まだ3ピクセル未満の場合は、すべての位置を使用
            if len(pixels) < 3:
                for y in range(height):
                    for x in range(width):
                        if (x, y) not in pixels and len(pixels) < 3:
                            pixels.append((x, y))

        # ノイズパターンでは連結性を確保しない（ランダム配置のまま）
        # 連結性制約は適用しない

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'random_pattern',
            'density': density
        })

    def _ensure_connected(self, pixels: list, width: int, height: int, connectivity: int = 4) -> list:
        """ピクセルが連結していることを保証

        連結していない場合、接続するピクセルを追加

        Args:
            pixels: ピクセルリスト
            width: 幅
            height: 高さ
            connectivity: 連結性（4または8、デフォルト: 4）

        Returns:
            連結されたピクセルリスト
        """
        if len(pixels) <= 1:
            return pixels

        pixel_set = set(pixels)
        visited = set()

        # 最初のピクセルから深さ優先探索
        start = pixels[0]
        stack = [start]
        visited.add(start)

        # 隣接方向を決定（4連結または8連結）
        if connectivity == 4:
            neighbor_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8連結
            neighbor_dirs = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]

        while stack:
            x, y = stack.pop()

            # 隣接セルを確認
            for dx, dy in neighbor_dirs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in pixel_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))

        # すべてのピクセルが訪問された場合は連結
        if len(visited) == len(pixels):
            return pixels

        # 連結していない場合、未訪問のピクセルを接続
        result = list(visited)
        unvisited = [p for p in pixels if p not in visited]

        # 未訪問のピクセルを最も近い訪問済みピクセルに接続
        for ux, uy in unvisited:
            # 最も近い訪問済みピクセルを見つける
            min_dist = float('inf')
            closest = None
            for vx, vy in visited:
                dist = abs(ux - vx) + abs(uy - vy)
                if dist < min_dist:
                    min_dist = dist
                    closest = (vx, vy)

            if closest:
                # 最も近いピクセルまでのパスを作成
                path = self._create_path(ux, uy, closest[0], closest[1], connectivity)
                result.extend(path)
                visited.add((ux, uy))

        return result

    def _create_path(self, x1: int, y1: int, x2: int, y2: int, connectivity: int = 4) -> list:
        """2点間のパスを作成

        Args:
            x1, y1: 開始点
            x2, y2: 終了点
            connectivity: 連結性（4または8、デフォルト: 4）

        Returns:
            パスのピクセルリスト
        """
        path = []
        x, y = x1, y1

        if connectivity == 4:
            # 4連結: まずX方向、次にY方向（またはその逆）
            dx = 1 if x2 > x1 else -1 if x2 < x1 else 0
            dy = 1 if y2 > y1 else -1 if y2 < y1 else 0

            while x != x2 or y != y2:
                if x != x2:
                    x += dx
                elif y != y2:
                    y += dy
                path.append((x, y))
        else:
            # 8連結: 斜め移動を許可
            while x != x2 or y != y2:
                dx = 1 if x2 > x else -1 if x2 < x else 0
                dy = 1 if y2 > y else -1 if y2 < y else 0

                x += dx
                y += dy
                path.append((x, y))

        return path

    def _generate_u_shape(self, color: int, size: int) -> Dict:
        """U字型を生成"""
        pixels = []
        width = size
        height = size

        # U字の左側
        for y in range(height - 1):
            pixels.append((0, y))

        # U字の底
        for x in range(width):
            pixels.append((x, height - 1))

        # U字の右側
        for y in range(height - 1):
            pixels.append((width - 1, y))

        return _add_lock_flags({
            'pixels': pixels,
            'width': width,
            'height': height,
            'color': color
        })

    def _generate_h_shape(self, color: int, size: int) -> Dict:
        """H字型を生成"""
        pixels = []
        width = size
        height = size

        # H字の左側
        for y in range(height):
            pixels.append((0, y))

        # H字の中央横線
        mid_y = height // 2
        for x in range(width):
            pixels.append((x, mid_y))

        # H字の右側
        for y in range(height):
            pixels.append((width - 1, y))

        return _add_lock_flags({
            'pixels': pixels,
            'width': width,
            'height': height,
            'color': color
        })

    def _generate_z_shape(self, color: int, size: int) -> Dict:
        """Z字型を生成"""
        pixels = []
        width = size
        height = size

        # Z字の上横線
        for x in range(width):
            pixels.append((x, 0))

        # Z字の斜め線
        for i in range(min(width, height)):
            x = width - 1 - i
            y = i
            if x >= 0 and y < height:
                pixels.append((x, y))

        # Z字の下横線
        for x in range(width):
            pixels.append((x, height - 1))

        return _add_lock_flags({
            'pixels': pixels,
            'width': width,
            'height': height,
            'color': color
        })

    def _generate_t_shape(self, color: int, size: int, rotation: int = 0) -> Dict:
        """T字型を生成"""
        pixels = []
        width = size
        height = size

        if rotation == 0:  # 通常のT
            # T字の上横線
            for x in range(width):
                pixels.append((x, 0))
            # T字の縦線
            for y in range(height):
                pixels.append((width // 2, y))

        elif rotation == 90:  # 右向きのT
            # T字の右縦線
            for y in range(height):
                pixels.append((width - 1, y))
            # T字の横線
            for x in range(width):
                pixels.append((x, height // 2))

        elif rotation == 180:  # 逆さのT
            # T字の下横線
            for x in range(width):
                pixels.append((x, height - 1))
            # T字の縦線
            for y in range(height):
                pixels.append((width // 2, y))

        elif rotation == 270:  # 左向きのT
            # T字の左縦線
            for y in range(height):
                pixels.append((0, y))
            # T字の横線
            for x in range(width):
                pixels.append((x, height // 2))

        return _add_lock_flags({
            'pixels': pixels,
            'width': width,
            'height': height,
            'color': color
        })

    def _generate_diagonal_45(self, color: int, size: int, direction: str = 'down_right') -> Dict:
        """45度の斜め線を生成"""
        pixels = []
        width = size
        height = size

        if direction == 'down_right':
            for i in range(min(width, height)):
                pixels.append((i, i))

        elif direction == 'down_left':
            for i in range(min(width, height)):
                pixels.append((width - 1 - i, i))

        elif direction == 'up_right':
            for i in range(min(width, height)):
                pixels.append((i, height - 1 - i))

        elif direction == 'up_left':
            for i in range(min(width, height)):
                pixels.append((width - 1 - i, height - 1 - i))

        return _add_lock_flags({
            'pixels': pixels,
            'width': width,
            'height': height,
            'color': color
        })

    def combine_objects(self, obj1: Dict, obj2: Dict, operation: str = 'union',
                       offset_x: int = 0, offset_y: int = 0) -> Dict:
        """2つのオブジェクトを合成して新しいオブジェクトを生成（最適化版）"""
        # obj2のピクセルをオフセット（リスト内包表記を最適化）
        obj2_pixels = obj2.get('pixels', [])
        if not obj2_pixels:
            # obj2が空の場合はobj1をそのまま返す
            if operation == 'union':
                return obj1.copy()
            elif operation == 'difference':
                return obj1.copy()
            else:  # intersection
                return _add_lock_flags({
                    'pixels': [],
                    'color': obj1['color'],
                    'width': 1,
                    'height': 1,
                    'area': 0,
                    'shape_type': f'combined_{operation}'
                })

        # セット操作の最適化: オフセットを適用しながらセットに変換
        obj2_pixels_offset = set((x + offset_x, y + offset_y) for x, y in obj2_pixels)
        obj1_pixels = set(obj1.get('pixels', []))

        # 合成操作
        if operation == 'union':
            result_pixels = obj1_pixels | obj2_pixels_offset
            result_color = obj1['color']
        elif operation == 'difference':
            result_pixels = obj1_pixels - obj2_pixels_offset
            result_color = obj1['color']
        elif operation == 'intersection':
            result_pixels = obj1_pixels & obj2_pixels_offset
            result_color = obj1['color']
        else:
            result_pixels = obj1_pixels
            result_color = obj1['color']

        # 結果が空の場合はobj1をそのまま返す
        if not result_pixels:
            return obj1.copy()

        # バウンディングボックスを計算（1回のループでmin/maxを同時に取得）
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for x, y in result_pixels:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # リストに変換してソート（必要に応じて）
        result_pixels_list = sorted(result_pixels)

        return _add_lock_flags({
            'pixels': result_pixels_list,
            'color': result_color,
            'width': width,
            'height': height,
            'area': len(result_pixels_list),
            'shape_type': f'combined_{operation}'
        })

    def generate_composite_object(self, color: int, grid_size: tuple = None, max_size: int = None) -> Dict:
        """複合オブジェクトを生成（複数の形状を組み合わせた高度な複合形状）

        用途: ARC-AGI2統計準拠の複雑な形状生成
        特徴: 2-3個の形状を複数の合成操作で組み合わせ

        Args:
            color: オブジェクトの色
            grid_size: グリッドサイズ
            max_size: 最大サイズ制約（指定された場合、各ベースオブジェクトのサイズをこの値以下に制限）
        """
        # 2-3個の基本形状を生成
        num_shapes = self.rng.randint(2, 4)
        base_objects = []

        # すべての形状テンプレートから選択可能にする
        shape_names = list(self.shape_templates.keys())

        for _ in range(num_shapes):
            shape_type = self.rng.choice(shape_names)

            # 各形状タイプに応じた適切な引数で生成（サイズ範囲を拡大）
            # グリッドサイズ情報を取得
            if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                grid_w = grid_size[0]
                grid_h = grid_size[1]
            else:
                grid_w = None
                grid_h = None

            # max_size制約を適用するヘルパー関数
            def _apply_max_size_constraint(base_max, max_size_constraint):
                if max_size_constraint is not None:
                    return min(base_max, max_size_constraint)
                return base_max

            if shape_type == 'rectangle':
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(2, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                width = self.rng.randint(min_size, max(min_size, constrained_max_size))
                height = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self._generate_rectangle(color, width, height, filled=True)
            elif shape_type == 'line':
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 8, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                length = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self._generate_line(color, length)
            elif shape_type == 'random_pattern':
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                width = self.rng.randint(min_size, max(min_size, constrained_max_size))
                height = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, width, height)
            elif shape_type == 't_shape':
                rotation = self.rng.choice([0, 90, 180, 270])
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size, rotation)
            elif shape_type == 'diagonal_45':
                direction = self.rng.choice(['down_right', 'down_left', 'up_right', 'up_left'])
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size, direction)
            elif shape_type in ['u_shape', 'h_shape', 'z_shape']:
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size)
            elif shape_type == 'cross':
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 7, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size)
            elif shape_type == 'arrow':
                direction = self.rng.choice(['up', 'down', 'left', 'right'])
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(3, 8, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size, direction)
            elif shape_type == 'diamond':
                filled = self.rng.random() < 0.7  # 70%の確率で塗りつぶし
                min_size, adaptive_max_size = self._calculate_adaptive_size_range(2, 6, grid_w, grid_h, None)
                constrained_max_size = _apply_max_size_constraint(adaptive_max_size, max_size)
                size = self.rng.randint(min_size, max(min_size, constrained_max_size))
                base_obj = self.shape_templates[shape_type](color, size, filled)
            elif shape_type == 'stairs':
                min_steps, adaptive_max_steps = self._calculate_adaptive_size_range(2, 5, grid_w, grid_h, None)
                constrained_max_steps = _apply_max_size_constraint(adaptive_max_steps, max_size)
                steps = self.rng.randint(min_steps, max(min_steps, constrained_max_steps))
                min_step_size, adaptive_max_step_size = self._calculate_adaptive_size_range(2, 4, grid_w, grid_h, None)
                constrained_max_step_size = _apply_max_size_constraint(adaptive_max_step_size, max_size)
                step_size = self.rng.randint(min_step_size, max(min_step_size, constrained_max_step_size))
                base_obj = self.shape_templates[shape_type](color, steps, step_size)
            elif shape_type == 'zigzag':
                min_length, adaptive_max_length = self._calculate_adaptive_size_range(4, 10, grid_w, grid_h, None)
                constrained_max_length = _apply_max_size_constraint(adaptive_max_length, max_size)
                length = self.rng.randint(min_length, max(min_length, constrained_max_length))
                # 振幅はグリッドサイズに応じて調整
                if grid_w is not None and grid_h is not None:
                    min_grid_dim = min(grid_w, grid_h)
                    if min_grid_dim <= 10:
                        max_amplitude = 3
                    elif min_grid_dim <= 20:
                        max_amplitude = 4
                    else:
                        max_amplitude = 5
                else:
                    max_amplitude = 3
                amplitude = self.rng.randint(1, max_amplitude)
                base_obj = self.shape_templates[shape_type](color, length, amplitude)
            else:
                # 単一引数の形状テンプレート
                base_obj = self.shape_templates[shape_type](color)

            base_objects.append(base_obj)

        # 最初のオブジェクトを基準として合成
        result = base_objects[0]

        for i in range(1, len(base_objects)):
            # ランダムなオフセットと合成方法を選択
            offset_x = self.rng.randint(-3, 3)
            offset_y = self.rng.randint(-3, 3)
            operation = self.rng.choice(['union', 'difference', 'intersection'])

            result = self.combine_objects(result, base_objects[i], operation, offset_x, offset_y)

        # 結果のバウンディングボックスを調整（最適化版）
        if result['pixels']:
            # 1回のループでmin/maxを取得
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            for x, y in result['pixels']:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            # 原点を基準に調整
            adjusted_pixels = [(x - min_x, y - min_y) for x, y in result['pixels']]

            result['pixels'] = adjusted_pixels
            result['width'] = max_x - min_x + 1
            result['height'] = max_y - min_y + 1

        result['shape_type'] = 'composite'
        return result

    def generate_object_by_arc_stats(self, color: int, grid_size: tuple = None,
                                    connectivity: int = None,
                                    force_diagonal_only: bool = False,
                                    max_size: int = None) -> Dict:
        """ARC-AGI2統計に基づくオブジェクト生成

        Args:
            color: オブジェクトの色
            grid_size: グリッドサイズ
            connectivity: 連結性
            force_diagonal_only: 斜め連結のみを強制
            max_size: 最大サイズ制約（指定された場合、生成されるオブジェクトのサイズをこの値以下に制限）
        """
        # max_size制約を適用するヘルパー関数
        def _apply_max_size_constraint(base_max, max_size_constraint):
            if max_size_constraint is not None:
                return min(base_max, max_size_constraint)
            return base_max

        # 斜め連結のみの形状を強制生成
        if force_diagonal_only:
            # 階段状またはチェッカーボード
            if self.rng.random() < 0.5:
                max_s = _apply_max_size_constraint(6, max_size)
                size = self.rng.randint(3, max(3, max_s))
                return self._generate_diagonal_connected_shape(color, size)
            else:
                max_w = _apply_max_size_constraint(5, max_size)
                max_h = _apply_max_size_constraint(5, max_size)
                width = self.rng.randint(3, max(3, max_w))
                height = self.rng.randint(3, max(3, max_h))
                if grid_size is not None and isinstance(grid_size, (tuple, list)) and len(grid_size) >= 2:
                    width = min(width, grid_size[0] // 2)
                    height = min(height, grid_size[1] // 2)
                return self._generate_checkerboard_shape(color, width, height)

        # 50%の確率で合成オブジェクトを生成
        if self.rng.random() < 0.5:
            obj = self.generate_composite_object(color, grid_size, max_size=max_size)
            if connectivity and not self._is_valid_for_connectivity(obj, connectivity):
                pass  # 通常生成に進む
            else:
                return obj

        # 通常の形状を生成（ARC-AGI2統計に準拠）
        rand = self.rng.random()

        if rand < 0.647:  # 64.7%: 単一ピクセル
            return _add_lock_flags({
                'pixels': [(0, 0)],
                'color': color,
                'width': 1,
                'height': 1,
                'area': 1,
                'shape_type': 'single_pixel'
            })
        elif rand < 0.807:  # 16%: 矩形
            max_w = _apply_max_size_constraint(6, max_size)
            max_h = _apply_max_size_constraint(6, max_size)
            width = self.rng.randint(2, max(2, max_w))
            height = self.rng.randint(2, max(2, max_h))
            return self._generate_rectangle(color, width, height)
        else:  # 19.3%: 複雑な形状
            shape_type = self.rng.choice(['line', 'l_shape', 'cross', 'diagonal'])
            if shape_type == 'line':
                max_len = _apply_max_size_constraint(3, max_size) if max_size is not None else 3
                return self.shape_templates[shape_type](color, length=max(3, max_len))
            elif shape_type == 'l_shape':
                max_w = _apply_max_size_constraint(6, max_size)
                max_h = _apply_max_size_constraint(6, max_size)
                return self._generate_l_shape(color, self.rng.randint(2, max(2, max_w)), self.rng.randint(2, max(2, max_h)))
            else:
                return self.shape_templates[shape_type](color)

    def generate_object_adaptive(self, color: int, grid_size: tuple,
                                remaining_space: int, num_objects_remaining: int,
                                min_size: int = 2) -> Dict:
        """適応的オブジェクト生成（残りスペースやオブジェクト数に応じた生成）

        Args:
            color: オブジェクトの色
            grid_size: グリッドサイズ
            remaining_space: 残りスペース
            num_objects_remaining: 残りオブジェクト数
            min_size: 最小サイズ

        Returns:
            適応的に生成されたオブジェクト
        """
        # 残りスペースとオブジェクト数に基づいてサイズを決定
        # ARC-AGI2の分析結果に基づき、オブジェクト数が多い場合に小さいサイズが
        # 選ばれる確率を上げる（特に1ピクセルを優先）
        if num_objects_remaining <= 2:
            # 最後の数個は大きめに
            max_size = min(remaining_space // 2, 15)
            target_size = self.rng.randint(min_size, max(max_size, min_size))
        elif num_objects_remaining <= 5:
            # 中程度のサイズ
            max_size = min(remaining_space // 3, 10)
            target_size = self.rng.randint(min_size, max(max_size, min_size))
        else:
            # 多くのオブジェクトが残っている場合は小さめに（多様性を保つため、制限を緩和）
            max_size = min(remaining_space // num_objects_remaining, 8)  # 6 → 8に緩和
            max_size = max(max_size, min_size)

            # ARC-AGI2の分析結果に基づく確率調整
            # 残りオブジェクト数が多い場合、1ピクセルを優先し、小さいサイズ（1-5ピクセル）を選択
            small_size_max = min(5, max_size)

            if num_objects_remaining >= 31:
                # 残り31個以上: ARC-AGI2統計に基づく（1ピクセル48.0%）、1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = self.rng.random()
                if rand < 0.30:  # ARC-AGI2統計: 48.0% → 30%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    target_size = 1
                elif rand < 0.90:  # 2-5ピクセル: 60%（合計90%、53% → 60%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく: 2px=40.7%, 3px=19.6%, 4px=29.5%, 5px=10.2%）
                    rand_size = self.rng.random()
                    if rand_size < 0.407:
                        target_size = 2
                    elif rand_size < 0.603:
                        target_size = 3
                    elif rand_size < 0.898:
                        target_size = 4
                    else:
                        target_size = 5
                    if target_size > small_size_max:
                        target_size = small_size_max if small_size_max >= max(2, min_size) else max(2, min_size)
                    elif target_size < max(2, min_size):
                        target_size = max(2, min_size)
                else:
                    # 6ピクセル以上の大きなオブジェクト（通常の確率分布）
                    target_size = self.rng.randint(small_size_max + 1, min(max_size, 10)) if small_size_max < max_size else self.rng.randint(min_size, min(max_size, 10))
            elif num_objects_remaining >= 21:
                # 残り21-30個: ARC-AGI2統計に基づく（1ピクセル44.8%）、1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = self.rng.random()
                if rand < 0.28:  # ARC-AGI2統計: 44.8% → 28%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    target_size = 1
                elif rand < 0.90:  # 2-5ピクセル: 62%（合計90%、55% → 62%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = self.rng.random()
                    if rand_size < 0.407:
                        target_size = 2
                    elif rand_size < 0.603:
                        target_size = 3
                    elif rand_size < 0.898:
                        target_size = 4
                    else:
                        target_size = 5
                    if target_size > small_size_max:
                        target_size = small_size_max if small_size_max >= max(2, min_size) else max(2, min_size)
                    elif target_size < max(2, min_size):
                        target_size = max(2, min_size)
                else:
                    # 6ピクセル以上の大きなオブジェクト（通常の確率分布）
                    target_size = self.rng.randint(small_size_max + 1, min(max_size, 10)) if small_size_max < max_size else self.rng.randint(min_size, min(max_size, 10))
            elif num_objects_remaining >= 11:
                # 残り11-20個: ARC-AGI2統計に基づく（1ピクセル39.3%）、1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = self.rng.random()
                if rand < 0.25:  # ARC-AGI2統計: 39.3% → 25%に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    target_size = 1
                elif rand < 0.88:  # 2-5ピクセル: 63%（合計88%、57% → 63%に増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = self.rng.random()
                    if rand_size < 0.407:
                        target_size = 2
                    elif rand_size < 0.603:
                        target_size = 3
                    elif rand_size < 0.898:
                        target_size = 4
                    else:
                        target_size = 5
                    if target_size > small_size_max:
                        target_size = small_size_max if small_size_max >= max(2, min_size) else max(2, min_size)
                    elif target_size < max(2, min_size):
                        target_size = max(2, min_size)
                else:
                    # 6ピクセル以上の大きなオブジェクト（通常の確率分布）
                    target_size = self.rng.randint(small_size_max + 1, min(max_size, 10)) if small_size_max < max_size else self.rng.randint(min_size, min(max_size, 10))
            else:
                # 残り6-10個: ARC-AGI2統計に基づく（1ピクセル55.0%）、1ピクセルをさらに下げ、2-5ピクセルの確率をさらに上げる（密度を下げる）
                rand = self.rng.random()
                if rand < 0.18:  # ARC-AGI2統計: 55.0% → 18%に大幅に調整（1ピクセルをさらに下げ、2-5ピクセルをさらに上げる）
                    target_size = 1
                elif rand < 0.88:  # 2-5ピクセル: 70%（合計88%、65% → 70%にさらに増加）
                    # 2-5ピクセル（ARC-AGI2の分布に基づく）
                    rand_size = self.rng.random()
                    if rand_size < 0.407:
                        target_size = 2
                    elif rand_size < 0.603:
                        target_size = 3
                    elif rand_size < 0.898:
                        target_size = 4
                    else:
                        target_size = 5
                    if target_size > small_size_max:
                        target_size = small_size_max if small_size_max >= max(2, min_size) else max(2, min_size)
                    elif target_size < max(2, min_size):
                        target_size = max(2, min_size)
                else:
                    # 6ピクセル以上の大きなオブジェクト（通常の確率分布）
                    target_size = self.rng.randint(small_size_max + 1, min(max_size, 10)) if small_size_max < max_size else self.rng.randint(min_size, min(max_size, 10))

        # サイズに応じて形状を選択
        if target_size <= 3:
            # 小さなオブジェクトは単純な形状
            shape_type = self.rng.choice(['rectangle', 'line'])
        elif target_size <= 8:
            # 中程度のオブジェクトは多様な形状
            shape_type = self.rng.choice(['rectangle', 'line', 'l_shape', 'cross'])
        else:
            # 大きなオブジェクトは複合形状
            return self.generate_composite_object(color, grid_size)

        if shape_type == 'rectangle':
            return self._generate_rectangle(color, width=3, height=3)
        elif shape_type == 'line':
            return self._generate_line(color, length=3)
        elif shape_type == 'l_shape':
            return self._generate_l_shape(color, self.rng.randint(2, 6), self.rng.randint(2, 6))
        else:
            return self.shape_templates[shape_type](color)

    def _is_valid_for_connectivity(self, obj: Dict, connectivity: int) -> bool:
        """オブジェクトが指定された連結性要件を満たすかチェック"""
        if not obj['pixels']:
            return False
        return len(obj['pixels']) > 1

    def _generate_diagonal_connected_shape(self, color: int, size: int) -> Dict:
        """斜め連結のみの形状を生成（階段状）"""
        pixels = []
        for i in range(size):
            pixels.append((i, i))
            if i < size - 1:
                pixels.append((i, i + 1))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': size,
            'height': size,
            'area': len(pixels),
            'shape_type': 'diagonal_connected'
        })

    def _generate_checkerboard_shape(self, color: int, width: int, height: int) -> Dict:
        """チェッカーボード形状を生成"""
        pixels = []
        for y in range(height):
            for x in range(width):
                if (x + y) % 2 == 0:
                    pixels.append((x, y))

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'checkerboard'
        })

    def _generate_arrow(self, color: int, size: int, direction: str = 'right') -> Dict:
        """矢印を生成（4方向: up, down, left, right）

        Args:
            color: 色
            size: サイズ（矢印の長さ）
            direction: 方向（'up', 'down', 'left', 'right'）

        Returns:
            矢印オブジェクト
        """
        pixels = []
        arrow_size = max(3, size)  # 最小サイズを3に

        if direction == 'right':
            # 軸: 水平線
            for x in range(arrow_size):
                pixels.append((x, arrow_size // 2))
            # 先端: 三角形（右側）
            tip_y = arrow_size // 2
            for i in range(arrow_size // 3 + 1):
                pixels.append((arrow_size - 1, tip_y - i))
                pixels.append((arrow_size - 1, tip_y + i))
        elif direction == 'left':
            # 軸: 水平線
            for x in range(arrow_size):
                pixels.append((x, arrow_size // 2))
            # 先端: 三角形（左側）
            tip_y = arrow_size // 2
            for i in range(arrow_size // 3 + 1):
                pixels.append((0, tip_y - i))
                pixels.append((0, tip_y + i))
        elif direction == 'up':
            # 軸: 垂直線
            for y in range(arrow_size):
                pixels.append((arrow_size // 2, y))
            # 先端: 三角形（上側）
            tip_x = arrow_size // 2
            for i in range(arrow_size // 3 + 1):
                pixels.append((tip_x - i, 0))
                pixels.append((tip_x + i, 0))
        else:  # down
            # 軸: 垂直線
            for y in range(arrow_size):
                pixels.append((arrow_size // 2, y))
            # 先端: 三角形（下側）
            tip_x = arrow_size // 2
            for i in range(arrow_size // 3 + 1):
                pixels.append((tip_x - i, arrow_size - 1))
                pixels.append((tip_x + i, arrow_size - 1))

        # 正規化（最小座標を(0,0)に）
        if pixels:
            min_x = min(x for x, y in pixels)
            min_y = min(y for x, y in pixels)
            pixels = [(x - min_x, y - min_y) for x, y in pixels]

        width = max(x for x, y in pixels) + 1 if pixels else 1
        height = max(y for x, y in pixels) + 1 if pixels else 1

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': f'arrow_{direction}'
        })

    def _generate_diamond(self, color: int, size: int, filled: bool = True) -> Dict:
        """ダイヤモンド（ひし形）を生成

        Args:
            color: 色
            size: サイズ（対角線の長さの半分）
            filled: 塗りつぶしかどうか

        Returns:
            ダイヤモンドオブジェクト
        """
        pixels = []
        diamond_size = max(2, size)

        if filled:
            # 塗りつぶし版: 45度回転した正方形
            center = diamond_size
            for y in range(diamond_size * 2 + 1):
                for x in range(diamond_size * 2 + 1):
                    # ダイヤモンドの形状: |x - center| + |y - center| <= diamond_size
                    if abs(x - center) + abs(y - center) <= diamond_size:
                        pixels.append((x, y))
        else:
            # 中空版: 輪郭のみ
            center = diamond_size
            for y in range(diamond_size * 2 + 1):
                for x in range(diamond_size * 2 + 1):
                    dist = abs(x - center) + abs(y - center)
                    # 輪郭のみ（境界）
                    if dist == diamond_size:
                        pixels.append((x, y))

        # 正規化（最小座標を(0,0)に）
        if pixels:
            min_x = min(x for x, y in pixels)
            min_y = min(y for x, y in pixels)
            pixels = [(x - min_x, y - min_y) for x, y in pixels]

        width = max(x for x, y in pixels) + 1 if pixels else 1
        height = max(y for x, y in pixels) + 1 if pixels else 1

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'diamond' if filled else 'hollow_diamond'
        })

    def _generate_stairs(self, color: int, steps: int, step_size: int = 2) -> Dict:
        """階段を生成（上昇/下降）

        Args:
            color: 色
            steps: 段数
            step_size: 各段のサイズ（幅と高さ）

        Returns:
            階段オブジェクト
        """
        pixels = []
        steps = max(2, min(steps, 10))  # 2-10段に制限
        step_size = max(2, step_size)

        # 上昇階段を生成
        for i in range(steps):
            # 各段の水平部分
            for x in range(step_size):
                pixels.append((i * step_size + x, i * step_size))
            # 各段の垂直部分（最後の段を除く）
            if i < steps - 1:
                for y in range(step_size):
                    pixels.append((i * step_size + step_size - 1, i * step_size + y))

        # 正規化（最小座標を(0,0)に）
        if pixels:
            min_x = min(x for x, y in pixels)
            min_y = min(y for x, y in pixels)
            pixels = [(x - min_x, y - min_y) for x, y in pixels]

        width = max(x for x, y in pixels) + 1 if pixels else 1
        height = max(y for x, y in pixels) + 1 if pixels else 1

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'stairs'
        })

    def _generate_zigzag(self, color: int, length: int, amplitude: int = 2) -> Dict:
        """ギザギザ線を生成

        Args:
            color: 色
            length: 長さ（ピクセル数）
            amplitude: 振幅（上下の幅）

        Returns:
            ギザギザオブジェクト
        """
        pixels = []
        length = max(4, length)
        amplitude = max(1, min(amplitude, 5))

        # ギザギザパターンを生成
        current_y = amplitude
        direction = 1  # 1: 上、-1: 下

        for x in range(length):
            pixels.append((x, current_y))
            # 2ピクセルごとに方向を変える
            if x % 2 == 1:
                current_y += direction * amplitude
                # 境界チェック
                if current_y < 0:
                    current_y = 0
                    direction = 1
                elif current_y > amplitude * 2:
                    current_y = amplitude * 2
                    direction = -1

        # 正規化（最小座標を(0,0)に）
        if pixels:
            min_x = min(x for x, y in pixels)
            min_y = min(y for x, y in pixels)
            pixels = [(x - min_x, y - min_y) for x, y in pixels]

        width = max(x for x, y in pixels) + 1 if pixels else 1
        height = max(y for x, y in pixels) + 1 if pixels else 1

        return _add_lock_flags({
            'pixels': pixels,
            'color': color,
            'width': width,
            'height': height,
            'area': len(pixels),
            'shape_type': 'zigzag'
        })

    def _make_symmetric(self, pixels: list, width: int, height: int, symmetry_type: str) -> list:
        """ピクセルリストに対称性を適用

        Args:
            pixels: 元のピクセルリスト
            width: 幅
            height: 高さ
            symmetry_type: 対称性タイプ（'vertical', 'horizontal', 'both'）

        Returns:
            対称性が適用されたピクセルリスト
        """
        if not pixels:
            return pixels

        symmetric_pixels = set(pixels)

        if symmetry_type == 'vertical' or symmetry_type == 'both':
            # 垂直対称（Y軸、左右対称）
            # ピクセル(x, y) → (width - 1 - x, y)
            vertical_mirror = {(width - 1 - x, y) for x, y in pixels}
            symmetric_pixels.update(vertical_mirror)

        if symmetry_type == 'horizontal' or symmetry_type == 'both':
            # 水平対称（X軸、上下対称）
            # ピクセル(x, y) → (x, height - 1 - y)
            horizontal_mirror = {(x, height - 1 - y) for x, y in pixels}
            symmetric_pixels.update(horizontal_mirror)

        return list(symmetric_pixels)

    def _add_holes_with_symmetry(self, pixels: list, width: int, height: int,
                                  symmetry_type: Optional[str], hole_count: int,
                                  color: int) -> list:
        """対称性を考慮して穴を追加（ピクセルを削除）

        Args:
            pixels: 元のピクセルリスト
            width: 幅
            height: 高さ
            symmetry_type: 対称性タイプ（None, 'vertical', 'horizontal', 'both'）
            hole_count: 追加する穴の数
            color: オブジェクトの色（使用しないが、将来の拡張用）

        Returns:
            穴が追加されたピクセルリスト
        """
        if not pixels or hole_count <= 0:
            return pixels

        pixel_set = set(pixels)

        # 内部領域（境界を除く）から候補位置を取得
        inner_positions = []
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if (x, y) in pixel_set:  # 既存のピクセルのみ
                    inner_positions.append((x, y))

        if not inner_positions:
            return pixels

        # 削除するピクセルを選択
        actual_hole_count = min(hole_count, len(inner_positions))
        if symmetry_type and symmetry_type != 'none':
            # 対称性がある場合、対称性を保つように選択
            # まず候補位置を対称性グループに分類
            symmetry_groups = {}
            for pos in inner_positions:
                # 対称性グループのキーを生成（最小の位置を代表とする）
                symmetric_positions = self._get_symmetric_positions(pos, width, height, symmetry_type)
                group_key = tuple(sorted(symmetric_positions))[0]  # 最小の位置をキーとする
                if group_key not in symmetry_groups:
                    symmetry_groups[group_key] = []
                symmetry_groups[group_key].extend(symmetric_positions)

            # 対称性グループからランダムに選択
            group_keys = list(symmetry_groups.keys())
            sample_size = min(actual_hole_count, len(group_keys))
            if sample_size > 0 and len(group_keys) > 0:
                selected_groups = self.rng.sample(group_keys, sample_size)
            else:
                selected_groups = []

            # 選択されたグループのすべての対称位置を削除
            for group_key in selected_groups:
                for pos in symmetry_groups[group_key]:
                    pixel_set.discard(pos)
        else:
            # 対称性がない場合、ランダムに選択
            if actual_hole_count > 0 and len(inner_positions) > 0:
                selected_positions = self.rng.sample(inner_positions, actual_hole_count)
            else:
                selected_positions = []
            for pos in selected_positions:
                pixel_set.discard(pos)

        return list(pixel_set)

    def _add_pixels_for_holes_with_symmetry(self, pixels: list, width: int, height: int,
                                            symmetry_type: Optional[str], hole_count: int,
                                            color: int) -> list:
        """対称性を考慮してピクセルを追加してから穴を作成

        Args:
            pixels: 元のピクセルリスト
            width: 幅
            height: 高さ
            symmetry_type: 対称性タイプ（None, 'vertical', 'horizontal', 'both'）
            hole_count: 作成する穴の数
            color: オブジェクトの色

        Returns:
            ピクセルが追加され、穴が作成されたピクセルリスト
        """
        if hole_count <= 0:
            return pixels

        pixel_set = set(pixels)

        # まず、バウンディングボックス内の空白領域を取得
        all_positions = set((x, y) for x in range(width) for y in range(height))
        empty_positions = all_positions - pixel_set

        # 内部領域（境界を除く）の空白のみ
        inner_empty = [(x, y) for x, y in empty_positions
                       if 1 <= x < width - 1 and 1 <= y < height - 1]

        if not inner_empty:
            # 内部領域に空白がない場合、外側にピクセルを追加してから穴を作成
            # オブジェクトを拡張（境界を追加）してから穴を作成する
            # 既存のピクセル周辺に連結性を保ちながら追加
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    if (x, y) not in pixel_set:
                        # 周囲にピクセルがある場合のみ追加（連結性を保つ）
                        neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]
                                     if (x+dx, y+dy) in pixel_set]
                        if neighbors:
                            if symmetry_type and symmetry_type != 'none':
                                # 対称性を保つように追加
                                symmetric_positions = self._get_symmetric_positions((x, y), width, height, symmetry_type)
                                for pos in symmetric_positions:
                                    pixel_set.add(pos)
                            else:
                                pixel_set.add((x, y))

            # 再度内部領域の空白を取得
            all_positions = set((x, y) for x in range(width) for y in range(height))
            empty_positions = all_positions - pixel_set
            inner_empty = [(x, y) for x, y in empty_positions
                           if 1 <= x < width - 1 and 1 <= y < height - 1]

        # 穴を作成（_add_holes_with_symmetryを使用）
        return self._add_holes_with_symmetry(list(pixel_set), width, height, symmetry_type, hole_count, color)

    def _get_symmetric_positions(self, pos: Tuple[int, int], width: int, height: int,
                                  symmetry_type: str) -> List[Tuple[int, int]]:
        """位置の対称性位置を取得

        Args:
            pos: 位置 (x, y)
            width: 幅
            height: 高さ
            symmetry_type: 対称性タイプ（'vertical', 'horizontal', 'both'）

        Returns:
            対称性位置のリスト（元の位置を含む）
        """
        x, y = pos
        symmetric_positions = [pos]

        if symmetry_type == 'vertical' or symmetry_type == 'both':
            # 垂直対称（Y軸、左右対称）
            symmetric_positions.append((width - 1 - x, y))

        if symmetry_type == 'horizontal' or symmetry_type == 'both':
            # 水平対称（X軸、上下対称）
            symmetric_positions.append((x, height - 1 - y))

        if symmetry_type == 'both':
            # 両対称の場合は、対角位置も追加
            symmetric_positions.append((width - 1 - x, height - 1 - y))

        # 重複を除去
        return list(set(symmetric_positions))

    def _generate_symmetric_object(
        self,
        color: int,
        base_shape: str = None,
        width: int = None,
        height: int = None,
        symmetry_type: str = 'vertical',
        max_size: int = None
    ) -> Dict:
        """対称性を持つオブジェクトを生成

        Args:
            color: オブジェクト色
            base_shape: ベース形状（Noneの場合はランダム選択、'rectangle', 'line', 'random'）
            width: 幅（Noneの場合はランダム）
            height: 高さ（Noneの場合はランダム）
            symmetry_type: 対称性タイプ（'vertical', 'horizontal', 'both'）

        Returns:
            対称性を持つオブジェクト辞書
        """
        # サイズを決定
        if width is None:
            width = self.rng.randint(3, 8)
        if height is None:
            height = self.rng.randint(3, 8)

        # 対称性を考慮したサイズ調整（両対称の場合は奇数サイズを推奨）
        if symmetry_type == 'both':
            # 両対称の場合、中心を明確にするため奇数サイズに
            if width % 2 == 0:
                width += 1
            if height % 2 == 0:
                height += 1

        # ベース形状を決定
        if base_shape is None:
            base_shape = self.rng.choice(['rectangle', 'line', 'random'])

        # ベースピクセルを生成（形状の半分または一部を生成）
        base_pixels = []

        if base_shape == 'rectangle':
            # 矩形の左半分（垂直対称の場合）または上半分（水平対称の場合）
            if symmetry_type == 'vertical' or symmetry_type == 'both':
                # 左半分を生成
                half_width = (width + 1) // 2
                for y in range(height):
                    for x in range(half_width):
                        base_pixels.append((x, y))
            elif symmetry_type == 'horizontal':
                # 上半分を生成
                half_height = (height + 1) // 2
                for y in range(half_height):
                    for x in range(width):
                        base_pixels.append((x, y))
        elif base_shape == 'line':
            # 線の一部を生成
            if symmetry_type == 'vertical' or symmetry_type == 'both':
                # 垂直線の左半分
                half_width = (width + 1) // 2
                center_y = height // 2
                for x in range(half_width):
                    base_pixels.append((x, center_y))
            elif symmetry_type == 'horizontal':
                # 水平線の上半分
                half_height = (height + 1) // 2
                center_x = width // 2
                for y in range(half_height):
                    base_pixels.append((center_x, y))
        else:  # 'random'
            # ランダムパターン（対称性を考慮して一部のみ生成）
            density = self.rng.uniform(0.2, 0.5)
            if symmetry_type == 'vertical' or symmetry_type == 'both':
                # 左半分のみ生成
                half_width = (width + 1) // 2
                num_pixels = int(half_width * height * density)
                for _ in range(num_pixels):
                    x = self.rng.randint(0, half_width - 1)
                    y = self.rng.randint(0, height - 1)
                    base_pixels.append((x, y))
            elif symmetry_type == 'horizontal':
                # 上半分のみ生成
                half_height = (height + 1) // 2
                num_pixels = int(width * half_height * density)
                for _ in range(num_pixels):
                    x = self.rng.randint(0, width - 1)
                    y = self.rng.randint(0, half_height - 1)
                    base_pixels.append((x, y))

        # 対称性を適用
        symmetric_pixels = self._make_symmetric(base_pixels, width, height, symmetry_type)

        # 連結性を確保（4連結または8連結をランダムに選択）
        connectivity = 4 if self.rng.random() < 0.5 else 8
        symmetric_pixels = self._ensure_connected(symmetric_pixels, width, height, connectivity)

        # バウンディングボックスを再計算
        if symmetric_pixels:
            min_x = min(x for x, y in symmetric_pixels)
            max_x = max(x for x, y in symmetric_pixels)
            min_y = min(y for x, y in symmetric_pixels)
            max_y = max(y for x, y in symmetric_pixels)
            actual_width = max_x - min_x + 1
            actual_height = max_y - min_y + 1

            # ピクセルを正規化（最小値が(0,0)になるように調整）
            normalized_pixels = [(x - min_x, y - min_y) for x, y in symmetric_pixels]

            return _add_lock_flags({
                'pixels': normalized_pixels,
                'color': color,
                'width': actual_width,
                'height': actual_height,
                'area': len(normalized_pixels),
                'shape_type': f'symmetric_{symmetry_type}'
            })
        else:
            # フォールバック: シンプルなオブジェクトを生成
            return self._generate_simple_object(color, max_size=max_size)
