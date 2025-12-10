"""
Color Permutation Augmentation

色の順列によるデータ拡張
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class ColorPermutation:
    """色の順列"""
    original_colors: List[int]  # 元の色のリスト
    permuted_colors: List[int]  # 順列化された色のリスト
    permutation_map: Dict[int, int]  # 色のマッピング（元の色 -> 新しい色）


class ColorPermutationAugmenter:
    """色順列拡張器"""

    def __init__(
        self,
        enable_permutation: bool = True,
        permutation_prob: float = 0.5,
        preserve_background: bool = True,
        max_permutations: int = 10
    ):
        """
        初期化

        Args:
            enable_permutation: 順列拡張を有効にするか
            permutation_prob: 順列拡張の確率
            preserve_background: 背景色を保持するか
            max_permutations: 最大順列数
        """
        self.enable_permutation = enable_permutation
        self.permutation_prob = permutation_prob
        self.preserve_background = preserve_background
        self.max_permutations = max_permutations

    def augment(
        self,
        grid: np.ndarray,
        background_color: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[ColorPermutation]]:
        """
        グリッドに色順列拡張を適用

        Args:
            grid: グリッド [H, W]
            background_color: 背景色（オプション）

        Returns:
            (augmented_grid, permutation): 拡張されたグリッドと色順列情報
        """
        if not self.enable_permutation or random.random() > self.permutation_prob:
            return grid, None

        # グリッド内のユニークな色を取得
        unique_colors = np.unique(grid)

        # 背景色を除外（有効な場合）
        if self.preserve_background and background_color is not None:
            unique_colors = unique_colors[unique_colors != background_color]

        if len(unique_colors) < 2:
            # 色が1つ以下の場合は拡張できない
            return grid, None

        # 色の順列を生成
        permutation_map = self._generate_permutation(unique_colors, background_color)

        # グリッドに順列を適用
        augmented_grid = grid.copy()
        for original_color, new_color in permutation_map.items():
            augmented_grid[grid == original_color] = new_color

        # 色順列情報を作成
        permutation = ColorPermutation(
            original_colors=list(unique_colors),
            permuted_colors=[permutation_map.get(c, c) for c in unique_colors],
            permutation_map=permutation_map
        )

        return augmented_grid, permutation

    def _generate_permutation(
        self,
        colors: np.ndarray,
        background_color: Optional[int] = None
    ) -> Dict[int, int]:
        """
        色の順列を生成

        Args:
            colors: 色の配列
            background_color: 背景色（オプション）

        Returns:
            permutation_map: 色のマッピング
        """
        # 色のリストをコピー
        color_list = list(colors)

        # 背景色を除外（有効な場合）
        if self.preserve_background and background_color is not None:
            if background_color in color_list:
                color_list.remove(background_color)

        if len(color_list) < 2:
            # 色が1つ以下の場合は順列を生成できない
            return {}

        # ランダムな順列を生成
        permuted_list = color_list.copy()
        random.shuffle(permuted_list)

        # マッピングを作成
        permutation_map = {}
        for original, permuted in zip(color_list, permuted_list):
            permutation_map[original] = permuted

        # 背景色をマッピングに追加（そのまま）
        if self.preserve_background and background_color is not None:
            permutation_map[background_color] = background_color

        return permutation_map

    def augment_pair(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        input_background_color: Optional[int] = None,
        output_background_color: Optional[int] = None,
        consistent_permutation: bool = True,
        program_code: Optional[str] = None,
        verify_correctness: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[ColorPermutation]]:
        """
        入出力ペアに色順列拡張を適用

        注意: プログラムが提供されている場合、色順列拡張を適用すると正解データが壊れる可能性があります。
        色順列拡張は、プログラムが色に対して不変でない場合、プログラムも変換する必要があります。
        デフォルトでは、プログラムが提供されている場合は拡張を適用しません。

        Args:
            input_grid: 入力グリッド [H, W]
            output_grid: 出力グリッド [H, W]
            input_background_color: 入力背景色（オプション）
            output_background_color: 出力背景色（オプション）
            consistent_permutation: 一貫した順列を使用するか（入出力で同じ順列）
            program_code: プログラムコード（オプション、提供されている場合は拡張を適用しない）
            verify_correctness: 正解性を検証するか（プログラムが提供されている場合）

        Returns:
            (augmented_input, augmented_output, permutation): 拡張されたグリッドと色順列情報
        """
        # プログラムが提供されている場合、拡張を適用しない（デフォルト）
        # 将来的には、プログラムも変換するオプションを追加可能
        if program_code is not None and verify_correctness:
            # プログラムが提供されている場合は拡張を適用しない
            return input_grid.copy(), output_grid.copy(), None
        if not self.enable_permutation or random.random() > self.permutation_prob:
            return input_grid, output_grid, None

        if consistent_permutation:
            # 一貫した順列を使用（入出力で同じ順列）
            # 入力と出力の両方に存在する色を取得
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            common_colors = input_colors & output_colors

            # 背景色を除外（有効な場合）
            if self.preserve_background:
                if input_background_color is not None:
                    common_colors.discard(input_background_color)
                if output_background_color is not None:
                    common_colors.discard(output_background_color)

            if len(common_colors) < 2:
                # 共通色が1つ以下の場合は拡張できない
                return input_grid, output_grid, None

            # 共通色の順列を生成
            permutation_map = self._generate_permutation(
                np.array(list(common_colors)),
                input_background_color if input_background_color == output_background_color else None
            )

            # 入出力グリッドに順列を適用
            augmented_input = input_grid.copy()
            augmented_output = output_grid.copy()

            for original_color, new_color in permutation_map.items():
                augmented_input[input_grid == original_color] = new_color
                augmented_output[output_grid == original_color] = new_color

            # 色順列情報を作成
            permutation = ColorPermutation(
                original_colors=list(common_colors),
                permuted_colors=[permutation_map.get(c, c) for c in common_colors],
                permutation_map=permutation_map
            )

            return augmented_input, augmented_output, permutation
        else:
            # 独立した順列を使用（入出力で異なる順列）
            augmented_input, permutation_input = self.augment(input_grid, input_background_color)
            augmented_output, permutation_output = self.augment(output_grid, output_background_color)

            # 順列情報を統合（簡易版）
            permutation = permutation_input if permutation_input else permutation_output

            return augmented_input, augmented_output, permutation

    def inverse_permutation(
        self,
        grid: np.ndarray,
        permutation: ColorPermutation
    ) -> np.ndarray:
        """
        色順列を逆変換

        Args:
            grid: 順列化されたグリッド
            permutation: 色順列情報

        Returns:
            original_grid: 元のグリッド
        """
        original_grid = grid.copy()

        # 逆マッピングを作成
        inverse_map = {v: k for k, v in permutation.permutation_map.items()}

        # グリッドに逆変換を適用
        for new_color, original_color in inverse_map.items():
            original_grid[grid == new_color] = original_color

        return original_grid


class ColorPermutationDataLoader:
    """色順列拡張付きデータローダー"""

    def __init__(
        self,
        augmenter: ColorPermutationAugmenter,
        base_loader: Any
    ):
        """
        初期化

        Args:
            augmenter: 色順列拡張器
            base_loader: ベースデータローダー
        """
        self.augmenter = augmenter
        self.base_loader = base_loader

    def __iter__(self):
        """イテレータ"""
        for batch in self.base_loader:
            # バッチ内の各サンプルに拡張を適用
            if 'input_grid' in batch and 'output_grid' in batch:
                augmented_batch = self._augment_batch(batch)
                yield augmented_batch
            else:
                yield batch

    def _augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        バッチに拡張を適用

        Args:
            batch: バッチデータ

        Returns:
            augmented_batch: 拡張されたバッチデータ
        """
        augmented_batch = batch.copy()

        input_grids = batch['input_grid']
        output_grids = batch['output_grid']
        input_background_colors = batch.get('input_background_color', [None] * len(input_grids))
        output_background_colors = batch.get('output_background_color', [None] * len(output_grids))

        augmented_inputs = []
        augmented_outputs = []
        permutations = []

        for i in range(len(input_grids)):
            input_grid = input_grids[i]
            output_grid = output_grids[i]
            input_bg = input_background_colors[i] if isinstance(input_background_colors, list) else input_background_colors
            output_bg = output_background_colors[i] if isinstance(output_background_colors, list) else output_background_colors

            aug_input, aug_output, perm = self.augmenter.augment_pair(
                input_grid,
                output_grid,
                input_bg,
                output_bg,
                consistent_permutation=True
            )

            augmented_inputs.append(aug_input)
            augmented_outputs.append(aug_output)
            permutations.append(perm)

        augmented_batch['input_grid'] = np.array(augmented_inputs)
        augmented_batch['output_grid'] = np.array(augmented_outputs)
        augmented_batch['color_permutation'] = permutations

        return augmented_batch
