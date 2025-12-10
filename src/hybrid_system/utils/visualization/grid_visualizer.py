"""グリッド可視化"""

import numpy as np
import matplotlib.pyplot as plt


class GridVisualizer:
    """グリッド可視化クラス"""
    
    COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    @staticmethod
    def visualize_grid(grid: np.ndarray, title: str = "") -> plt.Figure:
        """グリッドを可視化"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colored_grid = np.zeros((*grid.shape, 3))
        for i in range(10):
            mask = grid == i
            color = np.array([int(GridVisualizer.COLORS[i][j:j+2], 16) for j in (1, 3, 5)]) / 255
            colored_grid[mask] = color
        
        ax.imshow(colored_grid, interpolation='nearest')
        ax.set_title(title, fontsize=16)
        ax.grid(True, which='both', color='gray', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        return fig
