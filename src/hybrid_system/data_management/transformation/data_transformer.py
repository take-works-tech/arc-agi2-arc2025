"""
データ変換器

データの変換処理を行う機能
"""

from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass

from core.data_structures import DataPair, Task
from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import (
    resize_grid, pad_grid, crop_grid
)


@dataclass
class TransformationConfig:
    """変換設定"""
    enable_grid_transforms: bool = True
    enable_program_transforms: bool = True
    enable_metadata_transforms: bool = True
    preserve_original: bool = True
    max_transformations: int = 10


class DataTransformer:
    """データ変換器"""

    def __init__(self, config: Optional[TransformationConfig] = None):
        """初期化"""
        self.config = config or TransformationConfig()
        self.transformation_history = []

    def transform_data_pair(
        self,
        data_pair: DataPair,
        transformations: List[str],
        **kwargs
    ) -> DataPair:
        """DataPairを変換

        Args:
            data_pair: 変換するDataPair
            transformations: 適用する変換のリスト
            **kwargs: 変換パラメータ

        Returns:
            変換されたDataPair
        """
        transformed_pair = data_pair

        if self.config.preserve_original:
            # 元のデータを保持する場合はコピーを作成
            transformed_pair = DataPair(
                input=[row[:] for row in data_pair.input],
                output=[row[:] for row in data_pair.output],
                program=data_pair.program,
                metadata=data_pair.metadata.copy(),
                pair_id=data_pair.pair_id
            )

        # 変換を適用
        for transformation in transformations:
            if transformation in self._get_available_transformations():
                transformed_pair = self._apply_transformation(
                    transformed_pair, transformation, **kwargs
                )
                self.transformation_history.append({
                    'type': transformation,
                    'pair_id': transformed_pair.pair_id,
                    'timestamp': self._get_timestamp()
                })

        return transformed_pair

    def transform_task(
        self,
        task: Task,
        transformations: List[str],
        **kwargs
    ) -> Task:
        """Taskを変換

        Args:
            task: 変換するTask
            transformations: 適用する変換のリスト
            **kwargs: 変換パラメータ

        Returns:
            変換されたTask
        """
        transformed_task = task

        if self.config.preserve_original:
            # 元のデータを保持する場合はコピーを作成
            transformed_task = Task(
                train=[pair.copy() for pair in task.train],
                test=[pair.copy() for pair in task.test],
                program=task.program,
                metadata=task.metadata.copy(),
                task_id=task.task_id
            )

        # 各ペアを変換
        transformed_train_pairs = []
        for pair in transformed_task.train:
            # DataPairを作成して変換（適切な形式で作成）
            temp_pair = DataPair(
                input=pair['input'],
                output=pair['output'],
                program=transformed_task.program if hasattr(transformed_task, 'program') else None
            )
            transformed_pair = self.transform_data_pair(temp_pair, transformations, **kwargs)
            transformed_train_pairs.append({
                'input': transformed_pair.input,
                'output': transformed_pair.output
            })

        transformed_test_pairs = []
        for pair in transformed_task.test:
            # DataPairを作成して変換（適切な形式で作成）
            temp_pair = DataPair(
                input=pair['input'],
                output=pair['output'],
                program=transformed_task.program if hasattr(transformed_task, 'program') else None
            )
            transformed_pair = self.transform_data_pair(temp_pair, transformations, **kwargs)
            transformed_test_pairs.append({
                'input': transformed_pair.input,
                'output': transformed_pair.output
            })

        transformed_task.train = transformed_train_pairs
        transformed_task.test = transformed_test_pairs

        return transformed_task

    def _get_available_transformations(self) -> List[str]:
        """利用可能な変換のリストを取得"""
        return [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical',
            'resize_grid', 'pad_grid', 'crop_grid',
            'color_shift', 'color_invert', 'color_normalize',
            'add_noise', 'smooth_grid',
            'program_optimize', 'program_simplify'
        ]

    def _apply_transformation(
        self,
        data_pair: DataPair,
        transformation: str,
        **kwargs
    ) -> DataPair:
        """変換を適用"""
        if transformation == 'rotate_90':
            return self._rotate_90(data_pair)
        elif transformation == 'rotate_180':
            return self._rotate_180(data_pair)
        elif transformation == 'rotate_270':
            return self._rotate_270(data_pair)
        elif transformation == 'flip_horizontal':
            return self._flip_horizontal(data_pair)
        elif transformation == 'flip_vertical':
            return self._flip_vertical(data_pair)
        elif transformation == 'resize_grid':
            return self._resize_grid(data_pair, **kwargs)
        elif transformation == 'pad_grid':
            return self._pad_grid(data_pair, **kwargs)
        elif transformation == 'crop_grid':
            return self._crop_grid(data_pair, **kwargs)
        elif transformation == 'color_shift':
            return self._color_shift(data_pair, **kwargs)
        elif transformation == 'color_invert':
            return self._color_invert(data_pair)
        elif transformation == 'color_normalize':
            return self._color_normalize(data_pair)
        elif transformation == 'add_noise':
            return self._add_noise(data_pair, **kwargs)
        elif transformation == 'smooth_grid':
            return self._smooth_grid(data_pair, **kwargs)
        elif transformation == 'program_optimize':
            return self._program_optimize(data_pair)
        elif transformation == 'program_simplify':
            return self._program_simplify(data_pair)
        else:
            raise ValueError(f"不明な変換: {transformation}")

    def _rotate_90(self, data_pair: DataPair) -> DataPair:
        """90度回転"""
        data_pair.input = self._rotate_grid_90(data_pair.input)
        data_pair.output = self._rotate_grid_90(data_pair.output)
        return data_pair

    def _rotate_180(self, data_pair: DataPair) -> DataPair:
        """180度回転"""
        data_pair.input = self._rotate_grid_180(data_pair.input)
        data_pair.output = self._rotate_grid_180(data_pair.output)
        return data_pair

    def _rotate_270(self, data_pair: DataPair) -> DataPair:
        """270度回転"""
        data_pair.input = self._rotate_grid_270(data_pair.input)
        data_pair.output = self._rotate_grid_270(data_pair.output)
        return data_pair

    def _flip_horizontal(self, data_pair: DataPair) -> DataPair:
        """水平反転"""
        data_pair.input = self._flip_grid_horizontal(data_pair.input)
        data_pair.output = self._flip_grid_horizontal(data_pair.output)
        return data_pair

    def _flip_vertical(self, data_pair: DataPair) -> DataPair:
        """垂直反転"""
        data_pair.input = self._flip_grid_vertical(data_pair.input)
        data_pair.output = self._flip_grid_vertical(data_pair.output)
        return data_pair

    def _resize_grid(self, data_pair: DataPair, new_height: int, new_width: int) -> DataPair:
        """グリッドサイズ変更"""
        data_pair.input = resize_grid(data_pair.input, new_height, new_width)
        data_pair.output = resize_grid(data_pair.output, new_height, new_width)
        return data_pair

    def _pad_grid(self, data_pair: DataPair, pad_height: int, pad_width: int, pad_value: int = 0) -> DataPair:
        """グリッドパディング"""
        # pad_heightとpad_widthを上下左右に適用
        data_pair.input = pad_grid(data_pair.input, pad_height, pad_height, pad_width, pad_width, pad_value)
        data_pair.output = pad_grid(data_pair.output, pad_height, pad_height, pad_width, pad_width, pad_value)
        return data_pair

    def _crop_grid(self, data_pair: DataPair, start_row: int, start_col: int, height: int, width: int) -> DataPair:
        """グリッドクロップ"""
        data_pair.input = crop_grid(data_pair.input, start_row, start_col, height, width)
        data_pair.output = crop_grid(data_pair.output, start_row, start_col, height, width)
        return data_pair

    def _color_shift(self, data_pair: DataPair, shift_amount: int) -> DataPair:
        """色シフト"""
        data_pair.input = self._shift_colors(data_pair.input, shift_amount)
        data_pair.output = self._shift_colors(data_pair.output, shift_amount)
        return data_pair

    def _color_invert(self, data_pair: DataPair) -> DataPair:
        """色反転"""
        data_pair.input = self._invert_colors(data_pair.input)
        data_pair.output = self._invert_colors(data_pair.output)
        return data_pair

    def _color_normalize(self, data_pair: DataPair) -> DataPair:
        """色正規化"""
        data_pair.input = self._normalize_colors(data_pair.input)
        data_pair.output = self._normalize_colors(data_pair.output)
        return data_pair

    def _add_noise(self, data_pair: DataPair, noise_level: float = 0.1) -> DataPair:
        """ノイズ追加"""
        data_pair.input = self._add_noise_to_grid(data_pair.input, noise_level)
        data_pair.output = self._add_noise_to_grid(data_pair.output, noise_level)
        return data_pair

    def _smooth_grid(self, data_pair: DataPair) -> DataPair:
        """グリッド平滑化"""
        data_pair.input = self._smooth_grid_data(data_pair.input)
        data_pair.output = self._smooth_grid_data(data_pair.output)
        return data_pair

    def _program_optimize(self, data_pair: DataPair) -> DataPair:
        """プログラム最適化"""
        data_pair.program = self._optimize_program(data_pair.program)
        return data_pair

    def _program_simplify(self, data_pair: DataPair) -> DataPair:
        """プログラム簡素化"""
        data_pair.program = self._simplify_program(data_pair.program)
        return data_pair

    # グリッド変換のヘルパーメソッド
    def _rotate_grid_90(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッドを90度回転"""
        if not grid:
            return grid
        return [[grid[i][j] for i in range(len(grid)-1, -1, -1)] for j in range(len(grid[0]))]

    def _rotate_grid_180(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッドを180度回転"""
        if not grid:
            return grid
        return [[grid[i][j] for j in range(len(grid[0])-1, -1, -1)] for i in range(len(grid)-1, -1, -1)]

    def _rotate_grid_270(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッドを270度回転"""
        if not grid:
            return grid
        return [[grid[i][j] for i in range(len(grid))] for j in range(len(grid[0])-1, -1, -1)]

    def _flip_grid_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッドを水平反転"""
        if not grid:
            return grid
        return [row[::-1] for row in grid]

    def _flip_grid_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッドを垂直反転"""
        if not grid:
            return grid
        return grid[::-1]


    def _shift_colors(self, grid: List[List[int]], shift_amount: int) -> List[List[int]]:
        """色シフト"""
        if not grid:
            return grid

        new_grid = []
        for row in grid:
            new_row = []
            for pixel in row:
                new_pixel = (pixel + shift_amount) % 10
                new_row.append(new_pixel)
            new_grid.append(new_row)

        return new_grid

    def _invert_colors(self, grid: List[List[int]]) -> List[List[int]]:
        """色反転"""
        if not grid:
            return grid

        new_grid = []
        for row in grid:
            new_row = []
            for pixel in row:
                new_pixel = 9 - pixel
                new_row.append(new_pixel)
            new_grid.append(new_row)

        return new_grid

    def _normalize_colors(self, grid: List[List[int]]) -> List[List[int]]:
        """色正規化"""
        if not grid:
            return grid

        # 使用されている色を取得
        colors = set()
        for row in grid:
            colors.update(row)

        if not colors:
            return grid

        # 色のマッピングを作成
        color_map = {color: i for i, color in enumerate(sorted(colors))}

        new_grid = []
        for row in grid:
            new_row = [color_map[pixel] for pixel in row]
            new_grid.append(new_row)

        return new_grid

    def _add_noise_to_grid(self, grid: List[List[int]], noise_level: float) -> List[List[int]]:
        """グリッドにノイズを追加"""
        if not grid:
            return grid

        import random

        new_grid = []
        for row in grid:
            new_row = []
            for pixel in row:
                if random.random() < noise_level:
                    # ノイズを追加
                    new_pixel = random.randint(0, 9)
                else:
                    new_pixel = pixel
                new_row.append(new_pixel)
            new_grid.append(new_row)

        return new_grid

    def _smooth_grid_data(self, grid: List[List[int]]) -> List[List[int]]:
        """グリッド平滑化"""
        if not grid:
            return grid

        height, width = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(width)] for _ in range(height)]

        for i in range(height):
            for j in range(width):
                # 周囲のピクセルの平均を計算
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbors.append(grid[ni][nj])

                if neighbors:
                    new_grid[i][j] = round(sum(neighbors) / len(neighbors))
                else:
                    new_grid[i][j] = grid[i][j]

        return new_grid

    def _optimize_program(self, program: str) -> str:
        """プログラム最適化（本格実装）"""
        lines = program.split('\n')
        optimized_lines = []
        seen_lines = set()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                # 空行やコメントは保持
                optimized_lines.append(line)
                continue

            # 重複する行を除去
            if stripped not in seen_lines:
                seen_lines.add(stripped)
                optimized_lines.append(line)
            else:
                # 重複している場合は、コメントとして残すか、完全に削除
                # ここでは削除（必要に応じてコメントとして残すことも可能）
                pass

        # 連続する空行を1つにまとめる
        result_lines = []
        prev_empty = False
        for line in optimized_lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            result_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(result_lines)

    def _simplify_program(self, program: str) -> str:
        """プログラム簡素化（本格実装）"""
        lines = program.split('\n')
        simplified_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                # 空行やコメントは保持
                simplified_lines.append(line)
                continue

            # 長すぎる行を適切に分割
            if len(stripped) > 100:
                # カンマや括弧の位置で分割を試みる
                if ',' in stripped:
                    # カンマで分割して複数行に
                    parts = stripped.split(',')
                    for i, part in enumerate(parts):
                        if i == 0:
                            simplified_lines.append(part + ',')
                        elif i < len(parts) - 1:
                            simplified_lines.append('    ' + part.strip() + ',')
                        else:
                            simplified_lines.append('    ' + part.strip())
                else:
                    # 分割できない場合は短縮
                    simplified_lines.append(stripped[:100] + '...')
            else:
                simplified_lines.append(line)

        return '\n'.join(simplified_lines)

    def _get_timestamp(self) -> str:
        """タイムスタンプを取得"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """変換履歴を取得"""
        return self.transformation_history.copy()

    def clear_transformation_history(self):
        """変換履歴をクリア"""
        self.transformation_history.clear()

    def set_config(self, config: TransformationConfig):
        """設定を更新"""
        self.config = config

    def get_config(self) -> TransformationConfig:
        """設定を取得"""
        return self.config
