"""
プログラム合成エンジン専用ロガー

候補生成、一貫性チェック、プログラム実行の詳細ログを提供
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from .logger import Logger


class SynthesisLogger:
    """
    プログラム合成エンジン専用ロガー

    候補生成、一貫性チェック、プログラム実行の詳細ログを記録
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO,
        enable_debug_mode: bool = False,
        enable_file_logging: bool = True
    ):
        """初期化

        Args:
            log_dir: ログディレクトリ
            log_level: ログレベル
            enable_debug_mode: デバッグモードを有効化
            enable_file_logging: ファイルロギングを有効化
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/synthesis")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_debug_mode = enable_debug_mode
        self.enable_file_logging = enable_file_logging

        # ログレベルを設定
        if enable_debug_mode:
            log_level = logging.DEBUG

        # 基本ロガーを取得
        self.logger = Logger.get_logger(
            "ProgramSynthesisEngine",
            log_dir=str(self.log_dir) if enable_file_logging else None,
            level=log_level
        )

        # デバッグ情報を保存するためのデータ構造
        self.debug_data: Dict[str, Any] = {
            "candidate_generation": [],
            "consistency_checks": [],
            "program_executions": [],
            "scoring_results": [],
        }

        # セッションID（タイムスタンプ）
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_candidate_generation(
        self,
        method: str,
        candidates: List[str],
        pair_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """候補生成のログを記録

        Args:
            method: 候補生成方法（heuristic, pattern, template, neural）
            candidates: 生成された候補プログラムのリスト
            pair_index: ペアインデックス
            metadata: 追加メタデータ
        """
        self.logger.info(
            f"候補生成 [{method}]: {len(candidates)}個の候補を生成"
            + (f" (ペア {pair_index})" if pair_index is not None else "")
        )

        if self.enable_debug_mode:
            self.logger.debug(f"  生成された候補数: {len(candidates)}")
            for i, candidate in enumerate(candidates[:5]):  # 最初の5個のみ表示
                self.logger.debug(f"    候補 {i+1}: {candidate[:100]}...")  # 最初の100文字のみ

            # デバッグデータに保存
            self.debug_data["candidate_generation"].append({
                "method": method,
                "pair_index": pair_index,
                "candidate_count": len(candidates),
                "candidates": candidates[:10],  # 最初の10個のみ保存
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            })

    def log_consistency_check(
        self,
        program: str,
        consistency_score: float,
        pair_index: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """一貫性チェックのログを記録

        Args:
            program: チェック対象のプログラム
            consistency_score: 一貫性スコア
            pair_index: ペアインデックス
            details: 詳細情報
        """
        if consistency_score >= 0.8:
            level = logging.INFO
        elif consistency_score >= 0.5:
            level = logging.WARNING
        else:
            level = logging.ERROR

        self.logger.log(
            level,
            f"一貫性チェック: スコア={consistency_score:.3f}"
            + (f" (ペア {pair_index})" if pair_index is not None else "")
        )

        if self.enable_debug_mode:
            self.logger.debug(f"  プログラム: {program[:100]}...")
            if details:
                self.logger.debug(f"  詳細: {details}")

            # デバッグデータに保存
            self.debug_data["consistency_checks"].append({
                "program": program[:200],  # 最初の200文字のみ保存
                "consistency_score": consistency_score,
                "pair_index": pair_index,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            })

    def log_program_execution(
        self,
        program: str,
        input_grid: Any,
        output_grid: Any,
        expected_output: Optional[Any] = None,
        execution_time: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """プログラム実行のログを記録

        Args:
            program: 実行されたプログラム
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            expected_output: 期待される出力（オプション）
            execution_time: 実行時間（秒）
            success: 成功したかどうか
            error: エラーメッセージ（失敗時）
        """
        if success:
            self.logger.info(
                f"プログラム実行: 成功"
                + (f" (実行時間: {execution_time:.3f}秒)" if execution_time else "")
            )
        else:
            self.logger.error(f"プログラム実行: 失敗 - {error}")

        if self.enable_debug_mode:
            self.logger.debug(f"  プログラム: {program[:100]}...")
            if execution_time:
                self.logger.debug(f"  実行時間: {execution_time:.3f}秒")
            if error:
                self.logger.debug(f"  エラー: {error}")

            # デバッグデータに保存
            self.debug_data["program_executions"].append({
                "program": program[:200],
                "success": success,
                "execution_time": execution_time,
                "error": error,
                "input_shape": np.array(input_grid).shape if input_grid is not None else None,
                "output_shape": np.array(output_grid).shape if output_grid is not None else None,
                "expected_shape": np.array(expected_output).shape if expected_output is not None else None,
                "timestamp": datetime.now().isoformat()
            })

    def log_scoring_result(
        self,
        program: str,
        consistency_score: float,
        complexity_score: float,
        program_scorer_score: Optional[float] = None,
        final_score: float = 0.0,
        rank: Optional[int] = None
    ):
        """スコアリング結果のログを記録

        Args:
            program: スコアリング対象のプログラム
            consistency_score: 一貫性スコア
            complexity_score: 複雑度スコア
            program_scorer_score: ProgramScorerスコア（オプション）
            final_score: 最終スコア
            rank: ランキング（オプション）
        """
        rank_str = f" (ランク: {rank})" if rank is not None else ""
        self.logger.info(
            f"スコアリング結果{rank_str}: "
            f"一貫性={consistency_score:.3f}, "
            f"複雑度={complexity_score:.3f}, "
            + (f"ProgramScorer={program_scorer_score:.3f}, " if program_scorer_score is not None else "")
            + f"最終={final_score:.3f}"
        )

        if self.enable_debug_mode:
            self.logger.debug(f"  プログラム: {program[:100]}...")

            # デバッグデータに保存
            self.debug_data["scoring_results"].append({
                "program": program[:200],
                "consistency_score": consistency_score,
                "complexity_score": complexity_score,
                "program_scorer_score": program_scorer_score,
                "final_score": final_score,
                "rank": rank,
                "timestamp": datetime.now().isoformat()
            })

    def log_object_matching(
        self,
        success: bool,
        confidence: Optional[float] = None,
        pattern_type: Optional[str] = None,
        error: Optional[str] = None
    ):
        """オブジェクトマッチングのログを記録

        Args:
            success: 成功したかどうか
            confidence: 信頼度
            pattern_type: パターンタイプ
            error: エラーメッセージ（失敗時）
        """
        if success:
            self.logger.info(
                f"オブジェクトマッチング: 成功"
                + (f" (信頼度: {confidence:.3f})" if confidence is not None else "")
                + (f" (パターン: {pattern_type})" if pattern_type else "")
            )
        else:
            self.logger.warning(f"オブジェクトマッチング: 失敗 - {error or '不明なエラー'}")

    def log_partial_program_generation(
        self,
        partial_programs: Dict[int, str],
        pattern_type: Optional[str] = None
    ):
        """部分プログラム生成のログを記録

        Args:
            partial_programs: 部分プログラムの辞書（ペアインデックス -> プログラム）
            pattern_type: パターンタイプ
        """
        self.logger.info(
            f"部分プログラム生成: {len(partial_programs)}個の部分プログラムを生成"
            + (f" (パターン: {pattern_type})" if pattern_type else "")
        )

        if self.enable_debug_mode:
            for pair_idx, program in partial_programs.items():
                self.logger.debug(f"  ペア {pair_idx}: {program[:100]}...")

    def save_debug_data(self, filename: Optional[str] = None):
        """デバッグデータをJSONファイルに保存

        Args:
            filename: ファイル名（省略時は自動生成）
        """
        if not self.enable_file_logging:
            return

        if filename is None:
            filename = f"synthesis_debug_{self.session_id}.json"

        filepath = self.log_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.debug_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"デバッグデータを保存しました: {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報の辞書
        """
        return {
            "candidate_generation_count": len(self.debug_data["candidate_generation"]),
            "consistency_check_count": len(self.debug_data["consistency_checks"]),
            "program_execution_count": len(self.debug_data["program_executions"]),
            "scoring_result_count": len(self.debug_data["scoring_results"]),
            "session_id": self.session_id
        }

    def reset_debug_data(self):
        """デバッグデータをリセット"""
        self.debug_data = {
            "candidate_generation": [],
            "consistency_checks": [],
            "program_executions": [],
            "scoring_results": [],
        }
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
