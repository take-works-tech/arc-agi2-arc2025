"""
グリッド可視化ユーティリティ

input_gridとoutput_gridを1枚のPNG画像に保存する
"""
import numpy as np
import matplotlib
# バックエンドを非GUIに設定（tkinterエラーを回避）
matplotlib.use('Agg')  # ファイル出力専用バックエンド（GUI不要）
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

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


def _create_colormap() -> ListedColormap:
    """ARC用カラーマップを作成"""
    colors = [ARC_COLORS.get(i, '#FFFFFF') for i in range(10)]
    return ListedColormap(colors)


def save_grids_to_png(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    output_path: str,
    title: str = "Input vs Output",
    show_grid: bool = True,
    dpi: int = 100
) -> bool:
    """input_gridとoutput_gridを1枚のPNG画像として保存

    Args:
        input_grid: 入力グリッド（2D配列）
        output_grid: 出力グリッド（2D配列）
        output_path: 保存先ファイルパス
        title: 画像のタイトル
        show_grid: グリッド線を表示するか
        dpi: 解像度（デフォルト: 50）

    Returns:
        bool: 保存成功の場合True
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # カラーマップを作成
        cmap = _create_colormap()

        # 図を作成（横並び、サイズを大きく）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # 入力グリッドを描画
        im1 = ax1.imshow(input_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax1.set_title("Input Grid", fontsize=12, fontweight='bold')
        ax1.axis('off')

        if show_grid:
            h1, w1 = input_grid.shape
            ax1.set_xticks(np.arange(-0.5, w1, 1), minor=True)
            ax1.set_yticks(np.arange(-0.5, h1, 1), minor=True)
            ax1.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)

        # 出力グリッドを描画
        im2 = ax2.imshow(output_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax2.set_title("Output Grid", fontsize=12, fontweight='bold')
        ax2.axis('off')

        if show_grid:
            h2, w2 = output_grid.shape
            ax2.set_xticks(np.arange(-0.5, w2, 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, h2, 1), minor=True)
            ax2.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)

        # 全体タイトル
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存（DPIを上げて解像度を向上）
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.05)

        # matplotlibのリソースを確実に解放
        plt.close('all')  # すべての図を閉じる
        plt.clf()  # 現在の図をクリア
        import gc
        gc.collect()  # ガベージコレクションを実行してファイルハンドルを解放

        return True

    except Exception as e:
        print(f"❌ グリッド画像の保存に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_single_grid_to_png(
    grid: np.ndarray,
    output_path: str,
    title: str = "Grid",
    show_grid: bool = True,
    dpi: int = 100
) -> bool:
    """単一のグリッドをPNG画像として保存

    Args:
        grid: グリッド（2D配列）
        output_path: 保存先ファイルパス
        title: 画像のタイトル
        show_grid: グリッド線を表示するか
        dpi: 解像度（デフォルト: 100）

    Returns:
        bool: 保存成功の場合True
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # カラーマップを作成
        cmap = _create_colormap()

        # 図を作成
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # グリッドを描画
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        if show_grid:
            h, w = grid.shape
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)

        # レイアウト調整
        plt.tight_layout()

        # 保存
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

        # matplotlibのリソースを確実に解放
        plt.close('all')
        plt.clf()
        import gc
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ グリッド画像の保存に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False