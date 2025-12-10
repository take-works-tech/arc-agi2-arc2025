"""
ステップ実行機能

指定した行数までプログラムを実行し、途中で停止
"""
import numpy as np
from typing import List, Any, Optional, Tuple, Dict
from src.core_systems.executor.parsing.interpreter import Interpreter
from src.data_systems.data_models.core.object import Object
from .snapshot import SnapshotManager, ExecutionSnapshot


class StepExecutor:
    """ステップ実行器 - 行数指定で実行を停止"""

    def __init__(self, executor_core):
        """初期化

        Args:
            executor_core: ExecutorCoreインスタンス
        """
        self.executor_core = executor_core
        self.snapshot_manager = SnapshotManager()
        self.current_line = 0
        self.max_line = None
        self._should_stop = False

    def execute_until_line(
        self,
        ast: List[Any],
        max_line: int,
        execution_context: Any,
        create_snapshots: bool = True
    ) -> Tuple[Any, List[ExecutionSnapshot]]:
        """指定した行数まで実行

        Args:
            ast: ASTノードのリスト
            max_line: 実行を停止する行数（ASTノードのインデックス）
            execution_context: ExecutionContextまたはInterpreter
            create_snapshots: スナップショットを作成するか

        Returns:
            (最後の実行結果, スナップショットのリスト)
        """
        self.max_line = max_line
        self.current_line = 0
        self._should_stop = False
        self.snapshot_manager.clear()

        # Interpreterを作成（executor_coreを渡す）
        interpreter = Interpreter(self.executor_core)

        # 実行コンテキストから変数をコピー
        if hasattr(execution_context, 'context') and 'variables' in execution_context.context:
            interpreter.variables = execution_context.context['variables'].copy()
        elif hasattr(execution_context, 'variables'):
            interpreter.variables = execution_context.variables.copy()

        result = None
        snapshots = []

        # ASTノードを順次実行
        for i, node in enumerate(ast):
            if self._should_stop or (self.max_line is not None and i >= self.max_line):
                break

            # ノードを実行
            result = interpreter.execute_node(node)
            self.current_line = i + 1

            # スナップショットを作成
            if create_snapshots:
                snapshot = self.snapshot_manager.create_snapshot(
                    line_number=self.current_line,
                    execution_context=interpreter,
                    executor_core=self.executor_core
                )
                snapshots.append(snapshot)

        return result, snapshots

    def execute_step(
        self,
        ast: List[Any],
        start_line: int,
        execution_context: Any
    ) -> Tuple[Any, List[ExecutionSnapshot]]:
        """1ステップ（1行）だけ実行

        Args:
            ast: ASTノードのリスト
            start_line: 開始行（0から始まる）
            execution_context: ExecutionContextまたはInterpreter

        Returns:
            (実行結果, スナップショットのリスト)
        """
        result, snapshots = self.execute_until_line(ast, start_line + 1, execution_context, create_snapshots=True)
        return result, snapshots

    def stop(self):
        """実行を停止"""
        self._should_stop = True

    def reset(self):
        """状態をリセット"""
        self.current_line = 0
        self.max_line = None
        self._should_stop = False
        self.snapshot_manager.clear()

    def get_current_line(self) -> int:
        """現在の実行行を取得"""
        return self.current_line

    def get_snapshots(self) -> List[ExecutionSnapshot]:
        """すべてのスナップショットを取得"""
        return self.snapshot_manager.get_all_snapshots()

    def get_snapshot_at_line(self, line_number: int) -> Optional[ExecutionSnapshot]:
        """指定した行のスナップショットを取得"""
        return self.snapshot_manager.get_snapshot(line_number)
