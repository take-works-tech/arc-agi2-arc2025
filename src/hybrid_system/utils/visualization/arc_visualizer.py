#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC特化可視化ユーティリティ（新システム統合版）
ARC-AGI2タスクの可視化機能を提供
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from typing import List, Optional, Tuple
from pathlib import Path

from src.hybrid_system.utils.logging import Logger

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

logger = Logger.get_logger("ARCVisualizer")

# ARC標準色定義
ARC_COLORS = {
    0: '#000000',  # 黒（背景）
    1: '#0074D9',  # 青
    2: '#FF4136',  # 赤
    3: '#2ECC40',  # 緑
    4: '#FFDC00',  # 黄
    5: '#AAAAAA',  # グレー
    6: '#F012BE',  # マゼンタ
    7: '#FF851B',  # オレンジ
    8: '#7FDBFF',  # 水色
    9: '#870C25',  # 茶色
}

class ARCVisualizer:
    """ARC特化可視化クラス"""

    def __init__(self, output_dir: str = "visualizations"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # カラーマップを事前作成
        self.cmap = self._create_colormap()

        logger.info(f"ARCVisualizer初期化完了: {self.output_dir}")

    def _create_colormap(self) -> ListedColormap:
        """ARC用カラーマップを作成"""
        colors = [ARC_COLORS.get(i, '#FFFFFF') for i in range(10)]
        return ListedColormap(colors)

    def save_grid(
        self,
        grid: np.ndarray,
        filepath: str,
        title: str = "Grid",
        show_grid: bool = True,
        dpi: int = 150
    ) -> bool:
        """
        グリッドを画像として保存

        Args:
            grid: 保存するグリッド（2D配列）
            filepath: 保存先ファイルパス
            title: 画像のタイトル
            show_grid: グリッド線を表示するか
            dpi: 解像度

        Returns:
            bool: 保存成功の場合True
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # グリッドのサイズを取得
            height, width = grid.shape

            # 図のサイズを自動調整
            fig_width = max(width * 0.5, 3)
            fig_height = max(height * 0.5, 3)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # グリッドを表示
            im = ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=9, interpolation='nearest')

            # タイトル設定
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

            # 軸ラベル
            ax.set_xlabel('列', fontsize=10)
            ax.set_ylabel('行', fontsize=10)

            # グリッド線
            if show_grid:
                ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle='-', linewidth=1, alpha=0.5)
                ax.tick_params(which="minor", size=0)

            # 主目盛りを設定
            ax.set_xticks(np.arange(0, width, 1))
            ax.set_yticks(np.arange(0, height, 1))

            # レイアウト調整
            plt.tight_layout()

            # 保存
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"グリッド画像保存完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"グリッド画像保存エラー: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False

    def save_task(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        filepath: str,
        title: str = "Task",
        show_grid: bool = True,
        dpi: int = 150
    ) -> bool:
        """
        タスク（入力・出力ペア）を画像として保存

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            filepath: 保存先ファイルパス
            title: タスクのタイトル
            show_grid: グリッド線を表示するか
            dpi: 解像度

        Returns:
            bool: 保存成功の場合True
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # 2列のサブプロットを作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 入力グリッド
            im1 = ax1.imshow(input_grid, cmap=self.cmap, vmin=0, vmax=9, interpolation='nearest')
            ax1.set_title('入力', fontsize=12, fontweight='bold')
            ax1.set_xlabel('列', fontsize=10)
            ax1.set_ylabel('行', fontsize=10)

            if show_grid:
                h1, w1 = input_grid.shape
                ax1.set_xticks(np.arange(-0.5, w1, 1), minor=True)
                ax1.set_yticks(np.arange(-0.5, h1, 1), minor=True)
                ax1.grid(which="minor", color="white", linestyle='-', linewidth=1, alpha=0.5)

            # 出力グリッド
            im2 = ax2.imshow(output_grid, cmap=self.cmap, vmin=0, vmax=9, interpolation='nearest')
            ax2.set_title('出力', fontsize=12, fontweight='bold')
            ax2.set_xlabel('列', fontsize=10)
            ax2.set_ylabel('行', fontsize=10)

            if show_grid:
                h2, w2 = output_grid.shape
                ax2.set_xticks(np.arange(-0.5, w2, 1), minor=True)
                ax2.set_yticks(np.arange(-0.5, h2, 1), minor=True)
                ax2.grid(which="minor", color="white", linestyle='-', linewidth=1, alpha=0.5)

            # 全体タイトル
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

            # レイアウト調整
            plt.tight_layout()

            # 保存
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"タスク画像保存完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"タスク画像保存エラー: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False

    def save_multiple_grids(
        self,
        grids: List[np.ndarray],
        titles: List[str],
        filepath: str,
        main_title: str = "Grids",
        cols: int = 3,
        show_grid: bool = True,
        dpi: int = 150
    ) -> bool:
        """
        複数のグリッドを一つの画像として保存

        Args:
            grids: グリッドのリスト
            titles: 各グリッドのタイトル
            filepath: 保存先ファイルパス
            main_title: 全体のタイトル
            cols: 列数
            show_grid: グリッド線を表示するか
            dpi: 解像度

        Returns:
            bool: 保存成功の場合True
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            n_grids = len(grids)
            rows = (n_grids + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            # axesを1次元配列に変換
            if n_grids == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

            for idx, (grid, title) in enumerate(zip(grids, titles)):
                ax = axes[idx]
                im = ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=9, interpolation='nearest')
                ax.set_title(title, fontsize=10, fontweight='bold')

                if show_grid:
                    h, w = grid.shape
                    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
                    ax.grid(which="minor", color="white", linestyle='-', linewidth=1, alpha=0.5)

            # 余った軸を非表示
            for idx in range(n_grids, len(axes)):
                axes[idx].axis('off')

            # 全体タイトル
            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

            # レイアウト調整
            plt.tight_layout()

            # 保存
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"複数グリッド画像保存完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"複数グリッド画像保存エラー: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
