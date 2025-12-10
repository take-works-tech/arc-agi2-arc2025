"""
実行状態スナップショット機能

実行中の変数、オブジェクト、グリッド状態を取得
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from src.data_systems.data_models.core.object import Object


@dataclass
class ExecutionSnapshot:
    """実行状態のスナップショット"""

    line_number: int  # 実行した行数（またはASTノードインデックス）
    variables: Dict[str, Any] = field(default_factory=dict)  # 変数の名前と値
    arrays: Dict[str, List[Any]] = field(default_factory=dict)  # 配列の名前と内容
    objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # オブジェクト情報
    grid_state: Optional[np.ndarray] = None  # グリッドの状態

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'line_number': self.line_number,
            'variables': self.variables.copy(),
            'arrays': self.arrays.copy(),
            'objects': self.objects.copy(),
            'grid_state': self.grid_state.copy() if self.grid_state is not None else None,
            'grid_shape': self.grid_state.shape if self.grid_state is not None else None,
            'variable_count': len(self.variables),
            'array_count': len(self.arrays),
            'object_count': len(self.objects),
        }


class SnapshotManager:
    """スナップショット管理クラス"""

    def __init__(self):
        self.snapshots: List[ExecutionSnapshot] = []

    def create_snapshot(
        self,
        line_number: int,
        execution_context: Any,
        executor_core: Any,
        grid_state: Optional[np.ndarray] = None
    ) -> ExecutionSnapshot:
        """実行状態のスナップショットを作成

        Args:
            line_number: 実行した行数（またはASTノードインデックス）
            execution_context: ExecutionContextまたはInterpreterの変数辞書
            executor_core: ExecutorCoreインスタンス（グリッド状態取得用）
            grid_state: 現在のグリッド状態（指定されない場合はexecutor_coreから取得）

        Returns:
            ExecutionSnapshotインスタンス
        """
        # 変数を取得
        variables = {}
        if hasattr(execution_context, 'context') and 'variables' in execution_context.context:
            variables = execution_context.context['variables'].copy()
        elif hasattr(execution_context, 'variables'):
            variables = execution_context.variables.copy()
        elif isinstance(execution_context, dict) and 'variables' in execution_context:
            variables = execution_context['variables'].copy()
        elif isinstance(execution_context, dict):
            variables = execution_context.copy()

        # 配列を取得
        arrays = {}
        if hasattr(execution_context, 'context') and 'arrays' in execution_context.context:
            arrays = execution_context.context['arrays'].copy()
        elif hasattr(execution_context, 'arrays'):
            arrays = execution_context.arrays.copy()
        elif isinstance(execution_context, dict) and 'arrays' in execution_context:
            arrays = execution_context['arrays'].copy()

        # オブジェクト情報を取得
        objects = {}
        if hasattr(execution_context, 'context') and 'objects' in execution_context.context:
            objects_dict = execution_context.context['objects']
        elif hasattr(executor_core, 'execution_context') and 'objects' in executor_core.execution_context:
            objects_dict = executor_core.execution_context['objects']
        else:
            objects_dict = {}

        # Objectの情報を抽出
        if isinstance(objects_dict, dict):
            for obj_id, obj in objects_dict.items():
                if isinstance(obj, Object):
                    objects[obj_id] = self._extract_object_info(obj)
        elif isinstance(objects_dict, list):
            for obj in objects_dict:
                if isinstance(obj, Object):
                    objects[obj.object_id] = self._extract_object_info(obj)

        # グリッド状態を取得
        if grid_state is None:
            if hasattr(executor_core, 'execution_context') and 'grid' in executor_core.execution_context:
                grid_state = executor_core.execution_context['grid'].copy()
            elif hasattr(executor_core, 'execution_context') and 'input_grid' in executor_core.execution_context:
                grid_state = executor_core.execution_context['input_grid'].copy()

        snapshot = ExecutionSnapshot(
            line_number=line_number,
            variables=variables,
            arrays=arrays,
            objects=objects,
            grid_state=grid_state
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _extract_object_info(self, obj: Object) -> Dict[str, Any]:
        """Objectから情報を抽出

        Args:
            obj: Objectインスタンス

        Returns:
            オブジェクト情報の辞書
        """
        return {
            'object_id': obj.object_id,
            'object_type': str(obj.object_type) if hasattr(obj, 'object_type') else None,
            'pixels': obj.pixels.copy() if hasattr(obj, 'pixels') else [],
            'pixel_colors': dict(obj.pixel_colors) if hasattr(obj, 'pixel_colors') and obj.pixel_colors else {},
            'bbox': obj.bbox if hasattr(obj, 'bbox') else (0, 0, 0, 0),
            'bbox_left': obj.bbox_left if hasattr(obj, 'bbox_left') else 0,
            'bbox_top': obj.bbox_top if hasattr(obj, 'bbox_top') else 0,
            'bbox_right': obj.bbox_right if hasattr(obj, 'bbox_right') else 0,
            'bbox_bottom': obj.bbox_bottom if hasattr(obj, 'bbox_bottom') else 0,
            'bbox_width': obj.bbox_width if hasattr(obj, 'bbox_width') else 0,
            'bbox_height': obj.bbox_height if hasattr(obj, 'bbox_height') else 0,
            'color': obj.color if hasattr(obj, 'color') else 0,
            'dominant_color': obj.dominant_color if hasattr(obj, 'dominant_color') else 0,
            'color_ratio': dict(obj.color_ratio) if hasattr(obj, 'color_ratio') else {},
            'area': obj._area if hasattr(obj, '_area') else 0,
            'center_position': obj._center_position if hasattr(obj, '_center_position') else (0, 0),
            'hole_count': obj._hole_count if hasattr(obj, '_hole_count') else 0,
        }

    def get_snapshot(self, line_number: int) -> Optional[ExecutionSnapshot]:
        """指定した行数のスナップショットを取得"""
        for snapshot in self.snapshots:
            if snapshot.line_number == line_number:
                return snapshot
        return None

    def get_all_snapshots(self) -> List[ExecutionSnapshot]:
        """すべてのスナップショットを取得"""
        return self.snapshots.copy()

    def clear(self):
        """スナップショットをクリア"""
        self.snapshots.clear()
