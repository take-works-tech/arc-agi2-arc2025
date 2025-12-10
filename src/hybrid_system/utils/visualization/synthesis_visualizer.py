"""
プログラム合成エンジン専用可視化ツール

候補プログラム、実行トレース、スコアリング結果を可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from src.hybrid_system.utils.visualization.arc_visualizer import ARCVisualizer, ARC_COLORS
from src.hybrid_system.utils.logging.logger import Logger

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

logger = Logger.get_logger("SynthesisVisualizer")


class SynthesisVisualizer:
    """
    プログラム合成エンジン専用可視化クラス

    候補プログラム、実行トレース、スコアリング結果を可視化
    """

    def __init__(self, output_dir: str = "visualizations/synthesis"):
        """初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ARC可視化ツールを使用
        self.arc_visualizer = ARCVisualizer(output_dir=str(self.output_dir))

        logger.info(f"SynthesisVisualizer初期化完了: {self.output_dir}")

    def visualize_candidates(
        self,
        candidates: List[Dict[str, Any]],
        task_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """候補プログラムを可視化

        Args:
            candidates: 候補プログラムのリスト（各要素は{'program': str, 'score': float, ...}）
            task_id: タスクID
            filename: 出力ファイル名（省略時は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            task_suffix = f"_{task_id}" if task_id else ""
            filename = f"candidates{task_suffix}.txt"

        filepath = self.output_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("候補プログラム一覧\n")
            f.write("=" * 80 + "\n\n")

            for i, candidate in enumerate(candidates, start=1):
                program = candidate.get("program", "")
                score = candidate.get("score", 0.0)
                consistency = candidate.get("consistency_score", 0.0)
                complexity = candidate.get("complexity_score", 0.0)

                f.write(f"候補 {i}:\n")
                f.write(f"  スコア: {score:.3f}\n")
                f.write(f"  一貫性: {consistency:.3f}\n")
                f.write(f"  複雑度: {complexity:.3f}\n")
                f.write(f"  プログラム:\n")
                for line in program.split("\n"):
                    f.write(f"    {line}\n")
                f.write("\n" + "-" * 80 + "\n\n")

        logger.info(f"候補プログラムを保存しました: {filepath}")
        return str(filepath)

    def visualize_scoring_results(
        self,
        scoring_results: List[Dict[str, Any]],
        task_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """スコアリング結果を可視化

        Args:
            scoring_results: スコアリング結果のリスト
            task_id: タスクID
            filename: 出力ファイル名（省略時は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            task_suffix = f"_{task_id}" if task_id else ""
            filename = f"scoring_results{task_suffix}.png"

        filepath = self.output_dir / filename

        # データを準備
        consistency_scores = [r.get("consistency_score", 0.0) for r in scoring_results]
        complexity_scores = [r.get("complexity_score", 0.0) for r in scoring_results]
        final_scores = [r.get("final_score", 0.0) for r in scoring_results]
        program_scorer_scores = [
            r.get("program_scorer_score", 0.0)
            for r in scoring_results
            if r.get("program_scorer_score") is not None
        ]

        # プロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"スコアリング結果{' - ' + task_id if task_id else ''}", fontsize=14, fontweight='bold')

        # 1. 一貫性スコアの分布
        axes[0, 0].hist(consistency_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title("一貫性スコアの分布")
        axes[0, 0].set_xlabel("一貫性スコア")
        axes[0, 0].set_ylabel("頻度")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 複雑度スコアの分布
        axes[0, 1].hist(complexity_scores, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title("複雑度スコアの分布")
        axes[0, 1].set_xlabel("複雑度スコア")
        axes[0, 1].set_ylabel("頻度")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 最終スコアの分布
        axes[1, 0].hist(final_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title("最終スコアの分布")
        axes[1, 0].set_xlabel("最終スコア")
        axes[1, 0].set_ylabel("頻度")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 一貫性 vs 複雑度の散布図
        scatter = axes[1, 1].scatter(
            consistency_scores,
            complexity_scores,
            c=final_scores,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        axes[1, 1].set_title("一貫性 vs 複雑度")
        axes[1, 1].set_xlabel("一貫性スコア")
        axes[1, 1].set_ylabel("複雑度スコア")
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label="最終スコア")

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"スコアリング結果を保存しました: {filepath}")
        return str(filepath)

    def visualize_execution_trace(
        self,
        execution_trace: List[Dict[str, Any]],
        task_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """実行トレースを可視化

        Args:
            execution_trace: 実行トレースのリスト（各要素は{'step': int, 'grid': np.ndarray, ...}）
            task_id: タスクID
            filename: 出力ファイル名（省略時は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            task_suffix = f"_{task_id}" if task_id else ""
            filename = f"execution_trace{task_suffix}.png"

        filepath = self.output_dir / filename

        if not execution_trace:
            logger.warning("実行トレースが空です")
            return str(filepath)

        # グリッド数を取得
        num_steps = len(execution_trace)
        cols = min(5, num_steps)  # 最大5列
        rows = (num_steps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        fig.suptitle(f"実行トレース{' - ' + task_id if task_id else ''}", fontsize=14, fontweight='bold')

        for idx, step_data in enumerate(execution_trace):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            grid = step_data.get("grid")
            step_num = step_data.get("step", idx)
            description = step_data.get("description", f"ステップ {step_num}")

            if grid is not None:
                grid_array = np.array(grid)
                cmap = ListedColormap([ARC_COLORS.get(i, '#FFFFFF') for i in range(10)])
                ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
                ax.set_title(description, fontsize=10)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "データなし", ha='center', va='center')
                ax.axis('off')

        # 余分なサブプロットを非表示
        for idx in range(num_steps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"実行トレースを保存しました: {filepath}")
        return str(filepath)

    def visualize_synthesis_summary(
        self,
        summary: Dict[str, Any],
        task_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """合成結果のサマリを可視化

        Args:
            summary: サマリデータ
            task_id: タスクID
            filename: 出力ファイル名（省略時は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            task_suffix = f"_{task_id}" if task_id else ""
            filename = f"synthesis_summary{task_suffix}.txt"

        filepath = self.output_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("プログラム合成結果サマリ\n")
            f.write("=" * 80 + "\n\n")

            # 基本情報
            f.write("基本情報:\n")
            f.write(f"  タスクID: {task_id or 'N/A'}\n")
            f.write(f"  成功: {summary.get('success', False)}\n")
            f.write(f"  合成時間: {summary.get('synthesis_time', 0.0):.3f}秒\n")
            f.write("\n")

            # 候補生成統計
            if "candidate_stats" in summary:
                f.write("候補生成統計:\n")
                stats = summary["candidate_stats"]
                f.write(f"  総候補数: {stats.get('total_candidates', 0)}\n")
                f.write(f"  一貫性のある候補数: {stats.get('consistent_candidates', 0)}\n")
                f.write(f"  平均一貫性スコア: {stats.get('avg_consistency', 0.0):.3f}\n")
                f.write("\n")

            # 選択されたプログラム
            if "selected_program" in summary:
                f.write("選択されたプログラム:\n")
                program = summary["selected_program"]
                for line in program.split("\n"):
                    f.write(f"  {line}\n")
                f.write("\n")

            # スコアリング結果
            if "scoring" in summary:
                f.write("スコアリング結果:\n")
                scoring = summary["scoring"]
                f.write(f"  一貫性スコア: {scoring.get('consistency_score', 0.0):.3f}\n")
                f.write(f"  複雑度スコア: {scoring.get('complexity_score', 0.0):.3f}\n")
                f.write(f"  最終スコア: {scoring.get('final_score', 0.0):.3f}\n")
                f.write("\n")

        logger.info(f"合成結果サマリを保存しました: {filepath}")
        return str(filepath)
