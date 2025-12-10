"""
デバッグ実行器

ステップ実行と状態スナップショット機能を統合
"""
import numpy as np
from typing import List, Any, Optional, Tuple, Dict
from src.data_systems.data_models.core.object import Object
from src.core_systems.executor.core import ExecutorCore
from .step_executor import StepExecutor
from .snapshot import ExecutionSnapshot, SnapshotManager


class DebugExecutor:
    """デバッグ実行器 - ステップ実行と状態取得"""

    def __init__(self, executor_core: Optional[ExecutorCore] = None):
        """初期化

        Args:
            executor_core: ExecutorCoreインスタンス（Noneの場合は新規作成）
        """
        if executor_core is None:
            self.executor_core = ExecutorCore()
        else:
            self.executor_core = executor_core

        self.step_executor = StepExecutor(self.executor_core)
        self.snapshot_manager = SnapshotManager()

    def execute_program_until_line(
        self,
        program_code: str,
        input_grid: np.ndarray,
        max_line: int,
        input_objects: Optional[List[Object]] = None,
        input_image_index: int = 0,
        background_color: int = 0,
        create_snapshots: bool = True
    ) -> Tuple[np.ndarray, List[Object], float, List[ExecutionSnapshot]]:
        """指定した行数までプログラムを実行

        Args:
            program_code: プログラムコード
            input_grid: 入力グリッド
            max_line: 実行を停止する行数（0から始まる）
            input_objects: 入力オブジェクト
            input_image_index: 入力画像インデックス
            background_color: 背景色
            create_snapshots: スナップショットを作成するか

        Returns:
            (出力グリッド, オブジェクトリスト, 実行時間, スナップショットリスト)
        """
        import time
        from ..parsing.tokenizer import Tokenizer
        from ..parsing.parser import Parser

        start_time = time.time()

        # 入力グリッドをnumpy配列に変換
        if not isinstance(input_grid, np.ndarray):
            input_grid = np.array(input_grid)

        # グリッドサイズコンテキストを初期化
        height, width = input_grid.shape
        self.executor_core.grid_context.initialize((width, height))

        # 実行コンテキストを設定
        if input_objects is None:
            extraction_result = self.executor_core.object_extractor.extract_objects_by_type(
                input_grid, input_image_index
            )
            if extraction_result.success:
                all_objects = {}
                for object_list in extraction_result.objects_by_type.values():
                    for obj in object_list:
                        all_objects[obj.object_id] = obj
                self.executor_core.execution_context['objects'] = all_objects
                self.executor_core.execution_context['background_color'] = extraction_result.background_color
            else:
                self.executor_core.execution_context['objects'] = {}
                self.executor_core.execution_context['background_color'] = background_color
        else:
            self.executor_core.execution_context['objects'] = input_objects
            self.executor_core.execution_context['background_color'] = background_color

        self.executor_core.execution_context['input_image_index'] = input_image_index
        self.executor_core.execution_context['input_grid'] = input_grid
        self.executor_core.execution_context['grid'] = input_grid

        # 変数と配列の初期化
        if 'variables' not in self.executor_core.execution_context:
            self.executor_core.execution_context['variables'] = {}
        if 'arrays' not in self.executor_core.execution_context:
            self.executor_core.execution_context['arrays'] = {}

        # トークナイズとパース
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(program_code)
        parser = Parser(tokens)
        ast = parser.parse()

        # 実行コンテキストを作成（ExecutionContextスタイル）
        class ExecutionContextProxy:
            def __init__(self, executor_core):
                self.executor_core = executor_core
                self.context = executor_core.execution_context

        context_proxy = ExecutionContextProxy(self.executor_core)

        # ステップ実行
        result, snapshots = self.step_executor.execute_until_line(
            ast=ast,
            max_line=max_line,
            execution_context=context_proxy,
            create_snapshots=create_snapshots
        )

        # 最終グリッド状態を取得
        if 'grid' in self.executor_core.execution_context:
            final_grid = self.executor_core.execution_context['grid'].copy()
        else:
            final_grid = input_grid.copy()

        # オブジェクトを取得
        objects_dict = self.executor_core.execution_context.get('objects', {})
        if isinstance(objects_dict, dict):
            final_objects = list(objects_dict.values()) if objects_dict else []
        else:
            final_objects = objects_dict if isinstance(objects_dict, list) else []

        execution_time = time.time() - start_time

        return final_grid, final_objects, execution_time, snapshots

    def get_state_at_line(
        self,
        program_code: str,
        input_grid: np.ndarray,
        line_number: int,
        input_objects: Optional[List[Object]] = None,
        input_image_index: int = 0,
        background_color: int = 0
    ) -> Optional[ExecutionSnapshot]:
        """指定した行数の状態を取得

        Args:
            program_code: プログラムコード
            input_grid: 入力グリッド
            line_number: 取得したい行数（0から始まる）
            input_objects: 入力オブジェクト
            input_image_index: 入力画像インデックス
            background_color: 背景色

        Returns:
            スナップショット（該当する行がない場合はNone）
        """
        _, _, _, snapshots = self.execute_program_until_line(
            program_code=program_code,
            input_grid=input_grid,
            max_line=line_number + 1,
            input_objects=input_objects,
            input_image_index=input_image_index,
            background_color=background_color,
            create_snapshots=True
        )

        # 指定した行のスナップショットを返す
        for snapshot in snapshots:
            if snapshot.line_number == line_number + 1:  # 1ベースに変換
                return snapshot

        return None

    def reset(self):
        """状態をリセット"""
        self.step_executor.reset()
        self.snapshot_manager.clear()
