"""
Symmetry-Aware Augmentation モジュール

回転不変性、反転不変性、スケール不変性によるデータ拡張
Object Canonicalizationと相補的
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from enum import Enum


class SymmetryType(Enum):
    """対称性タイプ"""
    ROTATION_90 = "rotation_90"
    ROTATION_180 = "rotation_180"
    ROTATION_270 = "rotation_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    FLIP_BOTH = "flip_both"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NONE = "none"


class SymmetryAugmenter:
    """対称性を考慮したデータ拡張器"""

    def __init__(
        self,
        enable_rotation: bool = True,
        enable_flip: bool = True,
        enable_scale: bool = True,
        rotation_prob: float = 0.3,
        flip_prob: float = 0.3,
        scale_prob: float = 0.2,
        scale_factors: List[float] = [0.8, 0.9, 1.1, 1.2]
    ):
        """
        初期化

        Args:
            enable_rotation: 回転拡張を有効化
            enable_flip: 反転拡張を有効化
            enable_scale: スケール拡張を有効化
            rotation_prob: 回転拡張の確率
            flip_prob: 反転拡張の確率
            scale_prob: スケール拡張の確率
            scale_factors: スケール係数のリスト
        """
        self.enable_rotation = enable_rotation
        self.enable_flip = enable_flip
        self.enable_scale = enable_scale
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.scale_prob = scale_prob
        self.scale_factors = scale_factors

    def augment_grid(
        self,
        grid: np.ndarray,
        symmetry_type: Optional[SymmetryType] = None
    ) -> Tuple[np.ndarray, SymmetryType]:
        """
        グリッドに対称性拡張を適用

        Args:
            grid: 入力グリッド [H, W]
            symmetry_type: 適用する対称性タイプ（Noneの場合はランダムに選択）

        Returns:
            augmented_grid: 拡張されたグリッド
            applied_symmetry: 適用された対称性タイプ
        """
        if symmetry_type is None:
            symmetry_type = self._sample_symmetry_type()

        if symmetry_type == SymmetryType.NONE:
            return grid.copy(), symmetry_type

        augmented_grid = grid.copy()

        if symmetry_type == SymmetryType.ROTATION_90:
            augmented_grid = np.rot90(augmented_grid, k=1)
        elif symmetry_type == SymmetryType.ROTATION_180:
            augmented_grid = np.rot90(augmented_grid, k=2)
        elif symmetry_type == SymmetryType.ROTATION_270:
            augmented_grid = np.rot90(augmented_grid, k=3)
        elif symmetry_type == SymmetryType.FLIP_HORIZONTAL:
            augmented_grid = np.fliplr(augmented_grid)
        elif symmetry_type == SymmetryType.FLIP_VERTICAL:
            augmented_grid = np.flipud(augmented_grid)
        elif symmetry_type == SymmetryType.FLIP_BOTH:
            augmented_grid = np.fliplr(np.flipud(augmented_grid))
        elif symmetry_type == SymmetryType.SCALE_UP:
            augmented_grid = self._scale_grid(augmented_grid, scale_factor=1.1)
        elif symmetry_type == SymmetryType.SCALE_DOWN:
            augmented_grid = self._scale_grid(augmented_grid, scale_factor=0.9)

        return augmented_grid, symmetry_type

    def augment_pair(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        symmetry_type: Optional[SymmetryType] = None
    ) -> Tuple[np.ndarray, np.ndarray, SymmetryType]:
        """
        入力・出力グリッドペアに同じ対称性拡張を適用

        Args:
            input_grid: 入力グリッド [H, W]
            output_grid: 出力グリッド [H, W]
            symmetry_type: 適用する対称性タイプ（Noneの場合はランダムに選択）

        Returns:
            augmented_input: 拡張された入力グリッド
            augmented_output: 拡張された出力グリッド
            applied_symmetry: 適用された対称性タイプ
        """
        if symmetry_type is None:
            symmetry_type = self._sample_symmetry_type()

        if symmetry_type == SymmetryType.NONE:
            return input_grid.copy(), output_grid.copy(), symmetry_type

        # 同じ対称性を入力・出力の両方に適用
        augmented_input, _ = self.augment_grid(input_grid, symmetry_type)
        augmented_output, _ = self.augment_grid(output_grid, symmetry_type)

        return augmented_input, augmented_output, symmetry_type

    def _sample_symmetry_type(self) -> SymmetryType:
        """対称性タイプをランダムにサンプリング"""
        import random

        candidates = []

        if self.enable_rotation and random.random() < self.rotation_prob:
            candidates.extend([
                SymmetryType.ROTATION_90,
                SymmetryType.ROTATION_180,
                SymmetryType.ROTATION_270
            ])

        if self.enable_flip and random.random() < self.flip_prob:
            candidates.extend([
                SymmetryType.FLIP_HORIZONTAL,
                SymmetryType.FLIP_VERTICAL,
                SymmetryType.FLIP_BOTH
            ])

        if self.enable_scale and random.random() < self.scale_prob:
            candidates.extend([
                SymmetryType.SCALE_UP,
                SymmetryType.SCALE_DOWN
            ])

        if not candidates:
            return SymmetryType.NONE

        return random.choice(candidates)

    def _scale_grid(self, grid: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        グリッドをスケール（拡大・縮小）

        Args:
            grid: 入力グリッド [H, W]
            scale_factor: スケール係数（>1で拡大、<1で縮小）

        Returns:
            scaled_grid: スケールされたグリッド
        """
        from scipy.ndimage import zoom

        H, W = grid.shape
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)

        # スケール
        scaled = zoom(grid, (scale_factor, scale_factor), order=0)  # order=0: 最近傍補間

        # 元のサイズにリサイズ（必要に応じて）
        if new_H != H or new_W != W:
            # 中央を切り出しまたはパディング
            if new_H > H:
                # 拡大: 中央を切り出し
                start_h = (new_H - H) // 2
                scaled = scaled[start_h:start_h + H, :]
            elif new_H < H:
                # 縮小: パディング
                pad_h = (H - new_H) // 2
                scaled = np.pad(scaled, ((pad_h, H - new_H - pad_h), (0, 0)), mode='constant', constant_values=0)

            if new_W > W:
                # 拡大: 中央を切り出し
                start_w = (new_W - W) // 2
                scaled = scaled[:, start_w:start_w + W]
            elif new_W < W:
                # 縮小: パディング
                pad_w = (W - new_W) // 2
                scaled = np.pad(scaled, ((0, 0), (pad_w, W - new_W - pad_w)), mode='constant', constant_values=0)

        return scaled.astype(grid.dtype)

    def augment_batch(
        self,
        input_grids: List[np.ndarray],
        output_grids: List[np.ndarray],
        apply_same_symmetry: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[SymmetryType]]:
        """
        バッチに対して対称性拡張を適用

        Args:
            input_grids: 入力グリッドのリスト
            output_grids: 出力グリッドのリスト
            apply_same_symmetry: すべてのペアに同じ対称性を適用するか（Trueの場合）

        Returns:
            augmented_inputs: 拡張された入力グリッドのリスト
            augmented_outputs: 拡張された出力グリッドのリスト
            applied_symmetries: 適用された対称性タイプのリスト
        """
        augmented_inputs = []
        augmented_outputs = []
        applied_symmetries = []

        if apply_same_symmetry:
            # すべてのペアに同じ対称性を適用
            symmetry_type = self._sample_symmetry_type()
            for input_grid, output_grid in zip(input_grids, output_grids):
                aug_input, aug_output, _ = self.augment_pair(
                    input_grid, output_grid, symmetry_type
                )
                augmented_inputs.append(aug_input)
                augmented_outputs.append(aug_output)
                applied_symmetries.append(symmetry_type)
        else:
            # 各ペアにランダムな対称性を適用
            for input_grid, output_grid in zip(input_grids, output_grids):
                aug_input, aug_output, symmetry_type = self.augment_pair(
                    input_grid, output_grid, None
                )
                augmented_inputs.append(aug_input)
                augmented_outputs.append(aug_output)
                applied_symmetries.append(symmetry_type)

        return augmented_inputs, augmented_outputs, applied_symmetries


class SymmetryAwareDataLoader:
    """対称性を考慮したデータローダー（学習時用）"""

    def __init__(
        self,
        augmenter: Optional[SymmetryAugmenter] = None,
        augmentation_prob: float = 0.5
    ):
        """
        初期化

        Args:
            augmenter: 対称性拡張器（Noneの場合はデフォルトを使用）
            augmentation_prob: 拡張を適用する確率
        """
        self.augmenter = augmenter or SymmetryAugmenter()
        self.augmentation_prob = augmentation_prob

    def augment_sample(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        program_code: Optional[str] = None,
        verify_correctness: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        サンプルに拡張を適用（確率的）

        注意: プログラムが提供されている場合、拡張を適用すると正解データが壊れる可能性があります。
        プログラムが回転・反転に対して不変でない場合、プログラムも変換する必要があります。
        デフォルトでは、プログラムが提供されている場合は拡張を適用しません。

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            program_code: プログラムコード（オプション、提供されている場合は拡張を適用しない）
            verify_correctness: 正解性を検証するか（プログラムが提供されている場合）

        Returns:
            augmented_input: 拡張された入力グリッド
            augmented_output: 拡張された出力グリッド
        """
        import random

        # プログラムが提供されている場合、拡張を適用しない（デフォルト）
        # 将来的には、プログラムも変換するオプションを追加可能
        if program_code is not None and verify_correctness:
            # プログラムが提供されている場合は拡張を適用しない
            return input_grid.copy(), output_grid.copy()

        if random.random() < self.augmentation_prob:
            augmented_input, augmented_output, _ = self.augmenter.augment_pair(
                input_grid, output_grid, None
            )
            return augmented_input, augmented_output
        else:
            return input_grid.copy(), output_grid.copy()
