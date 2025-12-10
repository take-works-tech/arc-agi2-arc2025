"""
統一的プログラム生成器（メインクラス）
"""
import os
import random
import sys
import traceback
from typing import List, Dict, Optional, Tuple, Any

# ログ出力制御（デフォルト: False）
# デバッグ時のみTrueに設定（elementノードの選択状況を確認するため）
ENABLE_DEBUG_OUTPUT = os.environ.get('ENABLE_ELEMENT_DEBUG', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ELEMENT_DEBUG', 'false').lower() in ('true', '1', 'yes')
ENABLE_DEBUG_LOGS = os.environ.get('ENABLE_DEBUG_LOGS', 'false').lower() in ('true', '1', 'yes')
from ..metadata.types import SemanticType, TypeSystem, TypeInfo
from ..metadata.variable_manager import variable_manager
from ..metadata.constants import (
    GRID_SIZE_PRESERVATION_PROB,
    EMPTY_ARRAY_ADDITION_PROB,
    select_complexity,
    generate_output_grid_size,
    STEPS_BY_COMPLEXITY,
    NODES_BY_COMPLEXITY,
    RESOLVE_DEPENDENCIES_MIN_ITERATIONS,
    RESOLVE_DEPENDENCIES_MAX_ITERATIONS,
)
from ..metadata.commands import COMMAND_METADATA
from ..metadata.argument_schema import CONDITION_ARG, MATCH_PAIRS_CONDITION_ARG
from .nodes import (
    Node,
    InitializationNode, AssignmentNode,
    FilterNode, RenderNode,
    LiteralNode, VariableNode, CommandNode,
    ExcludeNode,
    ConcatNode, AppendNode, MergeNode, EmptyArrayNode,
    SingleObjectArrayNode, SplitConnectedNode, ExtractShapeNode, ExtendPatternNode, ArrangeGridNode,
    ObjectAccessNode, MatchPairsNode,
)
from .program_context import ProgramContext

# 共通の型情報定義をインポート
from . import OBJECT_ARRAY_TYPE

# コード生成ユーティリティをインポート
from .code_generator import (
    generate_code,
    generate_code_no_indent,
)

# 重み計算ユーティリティをインポート
from .selection_weights import (
    calculate_selection_weights,
    classify_node_type,
)

# ヘルパー関数と型定義をインポート
from .node_helpers import (
    NodeWithVarInfo,
    NodeListWithVarInfo,
    get_defined_variables_from_node_list,
    get_first_defined_variables,
    get_variable_info,
    is_first_definition,
    extend_node_list_with_var_info,
    get_nodes_from_var_info_list,
    add_node_to_list,
    get_node_from_var_info,
    get_variable_from_var_info,
    get_is_first_definition_from_var_info,
    has_variable_info,
    get_all_variables_from_list,
)

from .node_generators import NodeGenerators
from ..metadata.variable_managers import VariableManagers


class UnifiedProgramGenerator:
    """統一的プログラム生成器"""

    def __init__(self):
        self.node_generators = NodeGenerators()
        # argument_generatorsはnode_generators経由でアクセス可能
        self.argument_generators = self.node_generators.argument_generators
        self.variable_managers = VariableManagers()
        # コマンド重み調整のマップ（外部から設定可能）
        self.command_weight_adjustments: Dict[str, float] = {}

    def _initialize_program_context(self, complexity: Optional[int] = None,
                                     grid_width: Optional[int] = None,
                                     grid_height: Optional[int] = None) -> ProgramContext:
        """プログラムコンテキストを初期化（共通処理）

        Args:
            complexity: 複雑度
            grid_width: グリッド幅
            grid_height: グリッド高さ

        Returns:
            初期化されたProgramContext
        """
        if complexity is None:
            complexity = select_complexity()

        # 変数追跡システムをリセット（前のプログラムの情報が残らないように）
        variable_manager.tracker.reset_variable_tracking()

        # コンテキストの初期化（グリッドサイズを渡す）
        context = ProgramContext(complexity, grid_width=grid_width, grid_height=grid_height)

        # 複雑さ制約を初期化（改善案2）
        self.node_generators._initialize_complexity_constraints(context)

        # 複雑さ制約チェック用のカウンター初期化
        context.generated_nodes_count = 0
        context.complexity_check_interval = 5  # 5ノードごとにチェック

        # ジェネレーターインスタンスを設定（previous_nodesの更新に必要）
        context._generator_instance = self
        context.last_selected = None  # 直前の選択されたタイプを保存

        return context

    def _generate_program_core(self, context: ProgramContext) -> List[Node]:
        """プログラム生成のコア処理（共通処理）

        Args:
            context: 初期化済みのProgramContext

        Returns:
            生成されたノードリスト
        """
        # ゴールノードの生成
        goal_node = self._generate_goal_node(context)

        # ゴールノード内で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(goal_node.bg_color, context)

        # 依存関係の逆引き解決（複雑さ制約チェック付き）
        program_nodes = self._resolve_dependencies_with_constraints(goal_node, context)

        # grid_sizeが使用されている場合のみ必要な定義ノード群をゴールノードの直前に挿入
        goal_code = goal_node.generate()
        uses_grid_size_x = 'grid_size_x' in goal_code
        uses_grid_size_y = 'grid_size_y' in goal_code

        if uses_grid_size_x or uses_grid_size_y:
            grid_size_nodes = self.node_generators._generate_grid_size_definition_nodes(context, uses_grid_size_x, uses_grid_size_y)
            # ゴールノードを一時的に取り出し
            goal_node_from_list = program_nodes.pop()  # 最後のノード（ゴールノード）を取り出し
            # 必要なノードを追加
            program_nodes.extend(grid_size_nodes)
            # ゴールノードを最後に戻す
            program_nodes.append(goal_node_from_list)

        # 初期化ノードの存在を保証
        program_nodes = self._ensure_initialization_node(program_nodes, context)

        # 未使用変数の定義ノードを削除
        program_nodes = self.variable_managers._remove_unused_variables(program_nodes, context)

        return program_nodes

    def generate_program(self, complexity: Optional[int] = None,
                        grid_width: Optional[int] = None, grid_height: Optional[int] = None) -> str:
        """プログラムを生成"""
        context = self._initialize_program_context(complexity, grid_width, grid_height)
        program_nodes = self._generate_program_core(context)
        # コード生成して文字列を返す
        code = generate_code(program_nodes, context)
        return code

    def generate_program_nodes(self, complexity: Optional[int] = None,
                                grid_width: Optional[int] = None, grid_height: Optional[int] = None) -> List['Node']:
        """プログラムのNodeリストを生成（文字列変換前の構造化情報を返す）

        Args:
            complexity: 複雑度
            grid_width: グリッド幅（Noneの場合は自動決定）
            grid_height: グリッド高さ（Noneの場合は自動決定）

        Returns:
            Nodeオブジェクトのリスト
        """
        context = self._initialize_program_context(complexity, grid_width, grid_height)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] generate_program_nodes: _resolve_dependencies_with_constraints完了", flush=True)
        program_nodes = self._generate_program_core(context)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] generate_program_nodes: 完了、ノード数={len(program_nodes)}", flush=True)
        return program_nodes

    def generate_program_no_indent(self, complexity: Optional[int] = None,
                                   grid_width: Optional[int] = None, grid_height: Optional[int] = None) -> str:
        """プログラムを生成（インデント整形なし）"""
        context = self._initialize_program_context(complexity, grid_width, grid_height)
        program_nodes = self._generate_program_core(context)
        # インデント整形なしのコード生成
        code = generate_code_no_indent(program_nodes, context)
        return code

    def generate_program_nodes_from_partial(
        self,
        partial_nodes: List[Node],
        category_var_mapping: Dict[str, str],
        context: ProgramContext
    ) -> List[Node]:
        """
        部分プログラムから続きを生成

        Args:
            partial_nodes: 部分プログラムのNodeリスト（変数は既にcontextに登録済み）
            category_var_mapping: カテゴリ変数マッピング（カテゴリID -> 変数名）
            context: 部分プログラムの変数が登録済みのProgramContext

        Returns:
            完全なプログラムのNodeリスト
        """
        # 注意: contextは部分プログラムパーサーで既に初期化され、変数が登録済み
        # 注意: 変数追跡のリセットは、parse_partial_program_to_nodesの前で行われる
        # （generate_program_with_partial_program_flow内で実行）

        # 0. 部分プログラムの最後のノードタイプをcontextに設定（selection_weightsで使用される）
        # 部分プログラムの最後のノードがFILTERまたはEXCLUDEの場合、last_selectedを適切に設定
        if partial_nodes:
            last_partial_node = partial_nodes[-1]
            # ノードタイプを判定
            last_node_type = classify_node_type(last_partial_node)
            # last_selectedを設定（selection_weightsで使用される）
            if last_node_type == 'FILTER':
                context.last_selected = "filter"
            elif last_node_type == 'EXCLUDE':
                context.last_selected = "exclude"
            elif last_node_type == 'MERGE':
                context.last_selected = "merge"
            elif last_node_type == 'GET_OBJECTS':
                context.last_selected = "get_objects"
            # 部分プログラムのノードをprevious_nodesに追加（最新5ノードのみ保持）
            if not hasattr(context, 'previous_nodes'):
                context.previous_nodes = []
            # 部分プログラムの最後の5ノードをprevious_nodesに追加
            for node in partial_nodes[-5:]:
                context.previous_nodes.append(node)
            if len(context.previous_nodes) > 5:
                context.previous_nodes = context.previous_nodes[-5:]

        # 1. ゴールノードを生成
        goal_node = self._generate_goal_node(context)

        # 2. ゴールノード内で使用された変数を登録
        self.variable_managers._register_variables_used_in_node(goal_node.bg_color, context)

        # 3. 依存関係を解決（部分プログラムのNodeを含める）
        # 注意: _resolve_dependencies_with_constraintsはgoal_nodeから逆引きで依存関係を解決する
        # 部分プログラムのNodeは既に存在するため、依存関係解決時に考慮される
        program_nodes = self._resolve_dependencies_with_constraints(goal_node, context)

        # 4. 部分プログラムのNodeを先頭に追加
        # 部分プログラムのNodeは既に変数が定義されているため、先頭に配置
        all_nodes = partial_nodes.copy()
        all_nodes.extend(program_nodes)

        # 5. grid_sizeが使用されている場合のみ必要な定義ノード群をゴールノードの直前に挿入
        goal_code = goal_node.generate()
        uses_grid_size_x = 'grid_size_x' in goal_code
        uses_grid_size_y = 'grid_size_y' in goal_code

        if uses_grid_size_x or uses_grid_size_y:
            grid_size_nodes = self.node_generators._generate_grid_size_definition_nodes(context, uses_grid_size_x, uses_grid_size_y)
            # ゴールノードを一時的に取り出し
            goal_node_from_list = all_nodes.pop()  # 最後のノード（ゴールノード）を取り出し
            # 必要なノードを追加
            all_nodes.extend(grid_size_nodes)
            # ゴールノードを最後に戻す
            all_nodes.append(goal_node_from_list)

        # 6. 初期化ノードの追加をスキップ（部分プログラムに既に含まれているため）
        # _ensure_initialization_nodeを呼び出さない

        # 7. 未使用変数の定義ノードを削除
        final_nodes = self.variable_managers._remove_unused_variables(all_nodes, context)

        return final_nodes

    def _generate_goal_node(self, context: ProgramContext) -> RenderNode:
        """ゴールノード（RENDER_GRID）を生成"""
        return self.node_generators._generate_goal_node(context)

    def _resolve_dependencies(self, goal_node: Node, context: ProgramContext) -> List[Node]:
        """依存関係を逆引きで解決（改善版）

        Args:
            goal_node: ゴールノード
            context: プログラムコンテキスト（このメソッド内で変更される）

        Returns:
            List[Node]: 生成されたノードリスト

        Note:
            contextは参照渡しで変更されます（total_steps, current_step, _generator_instanceなど）
        """
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies開始: 複雑度={context.complexity}", flush=True)

        nodes: NodeListWithVarInfo = []

        # ステップ数とノード数の範囲を取得（STEPS_BY_COMPLEXITYとNODES_BY_COMPLEXITYから）
        # 複雑度が範囲外の場合はエラーを出す（デフォルト値は使用しない）
        if context.complexity not in STEPS_BY_COMPLEXITY:
            raise ValueError(f"無効な複雑度: {context.complexity} (有効な範囲: 1-8)")
        if context.complexity not in NODES_BY_COMPLEXITY:
            raise ValueError(f"無効な複雑度: {context.complexity} (有効な範囲: 1-8)")

        steps_tuple = STEPS_BY_COMPLEXITY[context.complexity]
        min_steps, max_steps = steps_tuple
        nodes_tuple = NODES_BY_COMPLEXITY[context.complexity]
        min_nodes_required, max_nodes_allowed = nodes_tuple
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: 複雑度={context.complexity}", flush=True)
            print(f"[DEBUG] _resolve_dependencies: ステップ範囲={min_steps}-{max_steps}, ノード範囲={min_nodes_required}-{max_nodes_allowed}", flush=True)

        # コンテキストにステップ情報を設定
        context.total_steps = max_steps  # 最大ステップ数を設定（デバッグ用）
        context._generator_instance = self

        # 配列チェーンを生成
        # 部分プログラムありの場合は、category_arraysから初期化（将来的に"objects"が含まれる可能性があるため）
        # 部分プログラムなしの場合は、["objects"]から初期化
        if hasattr(context, 'category_arrays') and context.category_arrays:
            # 部分プログラムあり: category_arraysから初期化
            current_arrays = []
            for category_var in context.category_arrays:
                # 変数が定義済みで、配列型であることを確認
                if context.variable_manager.is_defined(category_var):
                    var_info = context.variable_manager.get_variable_info(category_var)
                    if var_info:
                        # var_infoは辞書形式で返される
                        type_info = var_info.get('type_info')
                        if type_info:
                            # type_infoはTypeInfoオブジェクト（NamedTuple）
                            # TypeInfoには必ずis_array属性が存在するため、hasattrチェックは不要
                            if type_info.is_array:
                                current_arrays.append(category_var)
                                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                                    print(f"[DEBUG] _resolve_dependencies: カテゴリ変数 {category_var} をcurrent_arraysに追加", flush=True)
        else:
            # 部分プログラムなし: ["objects"]から初期化
            current_arrays = ["objects"]
            # objectsを変数トラッカーに登録
            context.variable_manager.define_variable("objects", SemanticType.OBJECT, is_array=True)
            # スコープ情報を記録
            context.add_scope_variable("objects")
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies: objectsを定義済みとして登録", flush=True)

        current_objects = []
        nodes_in_step: NodeListWithVarInfo = []
        nodes_in_step2: NodeListWithVarInfo = []

        # 初期ノードとしてget_objectsを設定（最初の_resolve_dependencies2呼び出し前に必要）
        # 注意: 部分プログラムフローの場合、部分プログラムの最後のノードタイプが既に設定されているため、
        # 上書きしない（部分プログラムの最後のノードタイプを保持）
        if not hasattr(context, 'last_selected') or context.last_selected is None:
            context.last_selected = "get_objects"

        # ループ: STEPS_BY_COMPLEXITYとNODES_BY_COMPLEXITYの最小値を両方満たすまで継続
        # 最大値をどちらかオーバーしたら停止
        step = 0
        while step < max_steps:
            # 現在のステップをコンテキストに設定
            context.current_step = step
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies: ステップ {step + 1} 開始 (current_nodes={context.current_lines}, 範囲: ステップ{min_steps}-{max_steps}, ノード{min_nodes_required}-{max_nodes_allowed})", flush=True)

            # 複雑度別の制御チェック
            if not context.can_add_node():
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] _resolve_dependencies: 最大ノード数({max_nodes_allowed})に達したため停止")
                break

            nodes_in_step, nodes_in_step2, current_arrays, current_objects = self._resolve_dependencies2(nodes_in_step, nodes_in_step2, context, current_arrays, current_objects, empty_arrays=[])
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies: ステップ {step + 1} 完了", flush=True)

            # シンプルなノード数制限チェック
            total_nodes = len(nodes_in_step) + len(nodes_in_step2)
            # current_linesを実際のノード数で更新（can_add_node()とshould_stop_generation()を機能させるため）
            context.current_lines = total_nodes

            step += 1

            # ============================================================
            # 終了条件チェック（最大値チェックは431行目とループ条件で既に処理済み）
            # ============================================================

            # 最大値チェック: ノード数が上限に達した場合は即座に停止（431行目のcan_add_node()チェックの後に更新された値を再チェック）
            if total_nodes >= max_nodes_allowed:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[INFO] ノード数上限到達: {total_nodes} >= {max_nodes_allowed}", flush=True)
                break

            # 最小値チェックと確率的停止: 2つの場合分け
            # ①どちらかが未達成 → 継続
            # ②両方達成 → 確率的停止
            min_steps_met = step >= min_steps
            min_nodes_met = total_nodes >= min_nodes_required

            if min_steps_met and min_nodes_met:
                # ②両方達成: 確率的に停止
                # 停止確率: ステップとノードの進捗の平均に基づく（U字型分布）
                step_range = max_steps - min_steps
                node_range = max_nodes_allowed - min_nodes_required

                # 停止確率計算（step_rangeまたはnode_rangeが0の場合は確率0.5で停止）
                if step_range > 0 and node_range > 0:
                    step_progress = (step - min_steps) / step_range  # 0.0-1.0
                    node_progress = (total_nodes - min_nodes_required) / node_range  # 0.0-1.0
                    progress = (step_progress + node_progress) / 2  # 平均進捗
                    # 二次関数: progress=0で0.4, progress=0.5で0.2, progress=1.0で0.6
                    stop_probability = 0.4 - 0.8 * progress * (1 - progress) + 0.2 * progress
                    stop_probability = max(0.1, min(0.8, stop_probability))  # 0.1-0.8に制限
                elif step_range > 0:
                    # node_rangeが0の場合（最小値=最大値）: ステップ進捗のみを使用
                    step_progress = (step - min_steps) / step_range  # 0.0-1.0
                    stop_probability = 0.4 - 0.8 * step_progress * (1 - step_progress) + 0.2 * step_progress
                    stop_probability = max(0.1, min(0.8, stop_probability))
                elif node_range > 0:
                    # step_rangeが0の場合（最小値=最大値）: ノード進捗のみを使用
                    node_progress = (total_nodes - min_nodes_required) / node_range  # 0.0-1.0
                    stop_probability = 0.4 - 0.8 * node_progress * (1 - node_progress) + 0.2 * node_progress
                    stop_probability = max(0.1, min(0.8, stop_probability))
                else:
                    # 両方とも0の場合（最小値=最大値）: 確率0.5で停止
                    stop_probability = 0.5

                if random.random() < stop_probability:
                    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                        print(f"[DEBUG] 最小値達成で確率的停止: step={step}, nodes={total_nodes}, prob={stop_probability:.2f}", flush=True)
                    break
                elif ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] 最小値達成だが継続: step={step}, nodes={total_nodes}, prob={stop_probability:.2f}", flush=True)
            else:
                # ①どちらかが未達成: 必ず継続
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    if not min_steps_met:
                        print(f"[DEBUG] 最小ステップ数未達のため継続: {step} < {min_steps}", flush=True)
                    if not min_nodes_met:
                        print(f"[DEBUG] 最小ノード数未達のため継続: {total_nodes} < {min_nodes_required}", flush=True)

        extend_node_list_with_var_info(nodes_in_step, nodes_in_step2)
        nodes_in_step2: NodeListWithVarInfo = []

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: 全ステップ完了後処理開始: current_arrays={current_arrays}, current_objects={current_objects}", flush=True)

        if not current_arrays:
            current_object = random.choice(current_objects)
            new_array = self._generate_new_array_variable_name(context)
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, new_array)
            single_object_array_node = self._generate_single_object_array_node(new_array, current_object, context)
            add_node_to_list(nodes_in_step, single_object_array_node, variable=new_array, context=context, is_first_override=is_first)
            current_arrays.append(new_array)
            self._safe_remove_from_objects(current_object, current_arrays, current_objects, context)
            # 復旧後の状態
        else:

            # 配列のコピーを作成してループ中の変更を防ぐ
            arrays_copy = current_arrays.copy()
            # 最初の配列以外を共通配列として処理
            common_arrays = arrays_copy[:-1] if len(arrays_copy) > 1 else []
            # 共通配列に対してCONCAT/EXCLUDEノードを生成
            self._process_common_arrays_for_concat(common_arrays, nodes_in_step, current_arrays, current_objects, context)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: CONCAT処理完了、current_objects処理開始: {len(current_objects)}個", flush=True)
        # current_objectsのコピーを作成してループ中の変更を防ぐ
        # common_objectsの処理
        objects_copy = current_objects.copy()
        self._process_common_objects_for_append(objects_copy, nodes_in_step, current_arrays, current_objects, context)

        # nodes_in_stepで初めて定義された変数を取得して引数使用禁止変数に登録
        first_defined_vars = get_first_defined_variables(nodes_in_step)
        for var_name in first_defined_vars:
            context.variable_manager.add_unavailable_variable(var_name)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: current_objects処理完了、_resolve_dependencies3呼び出し前", flush=True)
        # 依存関係解決前の状態をログ
        undefined_before = self.variable_managers._get_undefined_variables(context)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し前: 未定義変数={undefined_before}", flush=True)

        # 現在の変数状態を詳細ログ
        defined_vars = context.variable_manager.get_defined_variable_names()
        used_vars = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し前: 定義済み変数={defined_vars}", flush=True)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し前: 使用済み変数={used_vars}", flush=True)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: _resolve_dependencies3呼び出し開始", flush=True)
        nodes = self._resolve_dependencies3(nodes, context, current_arrays)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies: _resolve_dependencies3呼び出し完了", flush=True)

        # 依存関係解決後の状態をログ
        undefined_after = self.variable_managers._get_undefined_variables(context)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し後: 未定義変数={undefined_after}", flush=True)

        # 解決後の変数状態を詳細ログ
        defined_vars_after = context.variable_manager.get_defined_variable_names()
        used_vars_after = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し後: 定義済み変数={defined_vars_after}", flush=True)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies内_resolve_dependencies3呼び出し後: 使用済み変数={used_vars_after}", flush=True)

        # 引数使用禁止変数をクリア
        context.variable_manager.clear_unavailable_variables()

        extend_node_list_with_var_info(nodes, nodes_in_step)

        # 最終配列をgoal_nodeに設定
        if current_arrays:
            goal_node.array = current_arrays[0]
            # 最終配列は定義された変数なので、使用として登録する必要はない
        else:

            # current_arraysが空の場合は、objectsを初期化
            # 連結性はランダムに4または8を選択（ARC問題では両方が使用される）
            connectivity = random.choice([4, 8])
            init_node = InitializationNode(connectivity=connectivity, context=context.to_dict())
            nodes.insert(0, (init_node, None, False))
            goal_node.array = "objects"
            # objectsは定義された変数なので、使用として登録する必要はない
        add_node_to_list(nodes, goal_node, context=context)

        # 最終的な変数状態をログ
        final_defined_vars = context.variable_manager.get_defined_variable_names()
        final_used_vars = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies終了時: 定義済み変数={final_defined_vars}", flush=True)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies終了時: 使用済み変数={final_used_vars}", flush=True)

        # 最終的な未定義変数をチェック
        final_undefined = self.variable_managers._get_undefined_variables(context)
        if final_undefined:
            if ENABLE_ALL_LOGS:
                print(f"[WARNING] _resolve_dependencies終了時: 未定義変数が残存={final_undefined}", flush=True)
        else:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies終了時: 未定義変数なし", flush=True)

        return get_nodes_from_var_info_list(nodes)

    def _resolve_dependencies_with_validation(self, nodes: NodeListWithVarInfo, nodes_in_step: NodeListWithVarInfo, context: ProgramContext, current_arrays: List[str]) -> NodeListWithVarInfo:
        """依存関係解決と検証処理

        Args:
            nodes: 現在のノードリスト
            nodes_in_step: ステップ内のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列リスト

        Returns:
            解決後のノードリスト
        """
        # nodes_in_stepで初めて定義された変数を取得して引数使用禁止変数に登録
        first_defined_vars = get_first_defined_variables(nodes_in_step)
        for var_name in first_defined_vars:
            context.variable_manager.add_unavailable_variable(var_name)

        # try:
        # 依存関係解決前の状態をログ
        undefined_before = self.variable_managers._get_undefined_variables(context)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し前: 未定義変数={undefined_before}", flush=True)

        # 現在の変数状態を詳細ログ
        defined_vars = context.variable_manager.get_defined_variable_names()
        used_vars = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し前: 定義済み変数={defined_vars}")
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し前: 使用済み変数={used_vars}")

        nodes = self._resolve_dependencies3(nodes, context, current_arrays)

        # 依存関係解決後の状態をログ
        undefined_after = self.variable_managers._get_undefined_variables(context)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し後: 未定義変数={undefined_after}")

        # 解決後の変数状態を詳細ログ
        defined_vars_after = context.variable_manager.get_defined_variable_names()
        used_vars_after = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し後: 定義済み変数={defined_vars_after}")
            print(f"[DEBUG] _resolve_dependencies2内_resolve_dependencies3呼び出し後: 使用済み変数={used_vars_after}")

        # 引数使用禁止変数をクリア
        context.variable_manager.clear_unavailable_variables()

        extend_node_list_with_var_info(nodes, nodes_in_step)

        return nodes

    def _check_scope_nesting_depth(self, context: ProgramContext, structure_type: str = "for") -> Tuple[bool, int, int, int]:
        """スコープネスト深度をチェック

        Args:
            context: プログラムコンテキスト
            structure_type: チェックする構造のタイプ ("for" または "if")

        Returns:
            Tuple[bool, int, int, int]: (生成可能か, 現在深度, 最大深度, 予測深度)
        """
        current_scope_depth = context.get_scope_nesting_depth()
        max_scope_depth = context.get_max_scope_nesting_depth()
        predicted_depth = current_scope_depth + 1

        can_generate = predicted_depth <= max_scope_depth

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            if can_generate:
                print(f"[DEBUG] {structure_type.upper()}生成許可: 現在深度={current_scope_depth}, 最大深度={max_scope_depth}, 予測深度={predicted_depth}", flush=True)
            else:
                print(f"[DEBUG] {structure_type.upper()}生成禁止: 現在深度={current_scope_depth}, 最大深度={max_scope_depth}, 予測深度={predicted_depth} > 最大深度", flush=True)

        return can_generate, current_scope_depth, max_scope_depth, predicted_depth

    def _resolve_dependencies2(self, nodes: NodeListWithVarInfo, nodes_in_step: NodeListWithVarInfo, context: ProgramContext, current_arrays: List[str] = ["objects"], current_objects: List[str] = [], empty_arrays: Optional[List[Tuple[str, int]]] = None) -> Tuple[NodeListWithVarInfo, NodeListWithVarInfo, List[str], List[str]]:
        """依存関係を解決してノードを生成する（メイン処理）

        Args:
            nodes: 現在のノードリスト
            nodes_in_step: ステップ内のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            empty_arrays: EMPTY_ARRAYノードのリスト（(配列名, スコープネスト深度)のタプル）

        Returns:
            Tuple[NodeListWithVarInfo, NodeListWithVarInfo, List[str], List[str]]:
            (nodes, nodes_in_step, current_arrays, current_objects)
        """
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2開始: current_arrays={current_arrays}, current_objects={current_objects}", flush=True)
            print(f"[DEBUG] _resolve_dependencies2: スコープネスト深度={context.get_scope_nesting_depth()}, 最大スコープネスト深度={context.get_max_scope_nesting_depth()}", flush=True)

        # 現在の変数状態をログ
        defined_vars = context.variable_manager.get_defined_variable_names()
        used_vars = context.variable_manager.get_used_variable_names()
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies2開始時: 定義済み変数={defined_vars}")
            print(f"[DEBUG] _resolve_dependencies2開始時: 使用済み変数={used_vars}")

        # ============================================================================
        # セクション1: ノード選択重みの計算
        # ============================================================================
        selection_weights = calculate_selection_weights(context, self.command_weight_adjustments)
        select_types = list(selection_weights.keys())
        weights = list(selection_weights.values())

        # ============================================================================
        # セクション2: ノード生成ループ（成功するまで再試行）
        # ============================================================================
        excluded_types = []  # 除外リスト
        max_attempts = 18  # 最大試行回数（無限ループ防止のため増加）
        attempt_count = 0
        node_generated = False

        while attempt_count < max_attempts and not node_generated:
            attempt_count += 1

            # この試行の開始時に使用済み変数のセットを保存（失敗時にリセットするため）
            used_vars_before_attempt = set(context.variable_manager.get_used_variable_names())
            # この試行の開始時に定義済み変数のセットを保存
            defined_vars_before_attempt = set(context.variable_manager.get_all_variable_names())

            # 除外リストに含まれるタイプの重みを0にする
            current_weights = selection_weights.copy()
            for excluded_type in excluded_types:
                if excluded_type in current_weights:
                    current_weights[excluded_type] = 0.0

            # スコープネスト深度が最大深度に達している場合、FOR/IFの重みを0にする
            # 予測深度（現在深度+1）が最大深度を超える場合は禁止
            # 注意: 重みを計算する前にチェックする必要がある
            can_generate_for, _, _, _ = self._check_scope_nesting_depth(context, "for")
            can_generate_if, _, _, _ = self._check_scope_nesting_depth(context, "if")

            if not can_generate_for:
                # 予測深度が最大深度を超える場合、FORの生成を禁止
                if "for" in current_weights:
                    current_weights["for"] = 0.0
            if not can_generate_if:
                # 予測深度が最大深度を超える場合、IFの生成を禁止
                if "if" in current_weights:
                    current_weights["if"] = 0.0

            # redefineの重み調整：スコープネスト内以外では選択率0%（重さ0）
            current_scope_depth = context.get_scope_nesting_depth()
            if current_scope_depth == 0:
                # スコープネスト内以外（深度0）では重みを0にする
                if "redefine" in current_weights:
                    current_weights["redefine"] = 0.0
            else:
                # スコープネスト内（深度>0）では元の重みを使用（selection_weights.pyで設定）
                # ここでは特に調整しない（base_weightsの値を使用）
                pass

            # 重み付きランダム選択（重みを0にした後で計算）
            current_select_types = list(current_weights.keys())
            current_weights_list = list(current_weights.values())

            # すべての重みが0の場合、確実に生成可能なタイプ（empty, object_operations）を優先
            total_weight = sum(current_weights_list)
            if total_weight == 0.0:
                # emptyとobject_operationsは常に生成可能なので、重みを1.0に設定
                if "empty" in current_weights:
                    current_weights["empty"] = 1.0
                if "object_operations" in current_weights:
                    current_weights["object_operations"] = 1.0
                current_select_types = list(current_weights.keys())
                current_weights_list = list(current_weights.values())
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] 試行{attempt_count}: 選択重み: {dict(zip(current_select_types, current_weights_list))}", flush=True)
            selected = random.choices(current_select_types, weights=current_weights_list, k=1)[0]

            # 試行回数が1より大きい場合のみ除外リストに追加（1回目は追加しない）
            if attempt_count > 1:
                self._add_to_excluded_types(selected, excluded_types)

            # last_selectedを更新（次の選択のために）
            # ただし、FILTER選択時は更新前に値を保存（_handle_filter_selectionで使用するため）
            previous_last_selected = getattr(context, 'last_selected', None)
            context.last_selected = selected

            # ========================================================================
            # セクション3: 選択されたノードタイプに応じた処理
            # ========================================================================

            # Noneを除外してからcandidatesを作成
            valid_for_arrays = [arr for arr in context.for_arrays if arr is not None]
            candidates = [obj for obj in current_arrays if obj not in valid_for_arrays]
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                # デバッグ: 選択プロセスをログ出力
                if selected == "filter":
                    print(f"[DEBUG] FILTER選択時: candidates={candidates}, current_arrays={current_arrays}, valid_for_arrays={valid_for_arrays}", flush=True)

                current_depth = context.get_scope_nesting_depth()
                max_depth = context.get_max_scope_nesting_depth()
                print(f"[DEBUG] 選択結果: {selected}, current_arrays: {len(current_arrays)}, スコープネスト深度: {current_depth}/{max_depth}", flush=True)

            # 選択後の変数状態をログ
            defined_vars_after = context.variable_manager.get_defined_variable_names()
            used_vars_after = context.variable_manager.get_used_variable_names()
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies2選択後: 定義済み変数={defined_vars_after}")
                print(f"[DEBUG] _resolve_dependencies2選択後: 使用済み変数={used_vars_after}")

            # FORループ生成条件チェック（優先順位順）
            # チェック順序:
            # 1. 構造的ネスト深度チェック（最優先）- 予測深度が最大深度を超える場合は絶対に禁止
            # 2. 個数制限チェック - FORループの総数が上限を超えていないか
            # 3. その他の条件 - 配列の存在、コマンド使用可否など
            # 注意: enter_for_nesting()が呼ばれると深度が1増えるため、その前の時点でチェックする
            can_generate_for_depth, current_scope_depth, max_scope_depth, predicted_depth_after_for = self._check_scope_nesting_depth(context, "for")

            # 1. 構造的ネスト深度チェック（最優先）
            if not can_generate_for_depth:
                can_generate_for = False  # 予測深度が最大深度を超える場合は絶対に禁止
            else:
                # 2. スコープネスト深度が許容範囲内の場合のみ、個数制限とその他の条件をチェック
                can_generate_for = (
                    len(current_arrays) > 0 and
                    context.can_add_for_loop() and  # 個数制限チェック
                    context.can_use_command("FOR")
                )

            # FORループ生成条件を先にチェック（条件を満たさない場合は除外リストに追加）
            if selected == "for":
                node_generated, nodes, nodes_in_step, current_arrays, current_objects = self._handle_for_selection(
                    nodes, nodes_in_step, context, current_arrays, current_objects, can_generate_for, excluded_types, empty_arrays=empty_arrays
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.0.1: IF文生成 ---
            elif selected == "if":
                node_generated, nodes, nodes_in_step, current_arrays, current_objects = self._handle_if_selection(
                    nodes, nodes_in_step, context, current_arrays, current_objects, excluded_types, empty_arrays=empty_arrays
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.0.2: ELEMENT（配列要素代入）ノード生成 ---
            elif selected == "element":
                node_generated, nodes_in_step = self._handle_element_selection(
                    nodes_in_step, context, excluded_types
                )
                if node_generated:
                    continue  # 次の試行へ


            # --- 3.1: FILTERノード生成 ---
            elif selected == "filter":
                # previous_last_selectedを渡す（初期ノード直後かどうかを判定するため）
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_filter_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates, previous_last_selected=previous_last_selected
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.2: SORTノード生成 ---
            elif selected == "sort":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_sort_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.2: CONCATノード生成 ---
            elif selected == "concat":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_concat_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.4: EXCLUDEノード生成 ---
            elif selected == "exclude":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_exclude_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.5: CREATEノード生成 ---
            elif selected == "create":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_create_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates, excluded_types
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.6: MERGEノード生成 ---
            elif selected == "merge":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_merge_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.7: EMPTY_ARRAYノード生成 ---
            elif selected == "empty":
                node_generated, nodes_in_step, current_arrays = self._handle_empty_selection(
                    nodes_in_step, context, current_arrays
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.8: APPENDノード生成 ---
            elif selected == "append":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_append_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.9: REVERSEノード生成 ---
            elif selected == "reverse":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_reverse_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.10: TILEノード生成 ---
            elif selected == "tile":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_tile_selection(
                    nodes_in_step, context, current_arrays, current_objects
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.11: EXTRACT_SHAPEノード生成 ---
            elif selected == "extract_shape":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_extract_shape_selection(
                    nodes_in_step, context, current_arrays, current_objects
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.12: EXTEND_PATTERNノード生成 ---
            elif selected == "extend_pattern":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_extend_pattern_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.13: ARRANGE_GRIDノード生成 ---
            elif selected == "arrange_grid":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_arrange_grid_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.14: SINGLE_OBJECT_ARRAYノード生成 ---
            elif selected == "single_object_array":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_single_object_array_selection(
                    nodes_in_step, context, current_arrays, current_objects
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.15: SPLIT_CONNECTEDノード生成 ---
            elif selected == "split_connected":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_split_connected_selection(
                    nodes_in_step, context, current_arrays, current_objects
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.16: OBJECT_OPERATIONSノード生成 ---
            elif selected == "object_operations":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_object_operations_selection(
                    nodes_in_step, context, current_arrays, current_objects, excluded_types
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.17: MATCH_PAIRSノード生成 ---
            elif selected == "match_pairs":
                node_generated, nodes_in_step, current_arrays, current_objects = self._handle_match_pairs_selection(
                    nodes_in_step, context, current_arrays, current_objects, candidates
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.18: DEFINEノード生成（変数定義） ---
            elif selected == "define":
                node_generated, nodes_in_step = self._handle_define_selection(
                    nodes_in_step, context
                )
                if node_generated:
                    continue  # 次の試行へ

            # --- 3.19: REDEFINEノード生成（変数再定義） ---
            elif selected == "redefine":
                node_generated, nodes_in_step = self._handle_redefine_selection(
                    nodes_in_step, context
                )
                if node_generated:
                    continue  # 次の試行へ
            else:
                # 選択されたタイプが条件を満たさず、ノードが生成されなかった場合
                # このタイプを除外リストに追加して、ループの次の反復で再試行
                self._add_to_excluded_types(selected, excluded_types, "条件を満たさず、ノードが生成されませんでした")
                # node_generatedはFalseのままなので、ループが継続される

            # ノード生成が失敗した場合、この試行で追加された変数の使用マークをリセット
            if not node_generated:
                used_vars_after_attempt = set(context.variable_manager.get_used_variable_names())
                newly_marked_vars = used_vars_after_attempt - used_vars_before_attempt
                # この試行で定義されたが使用されていない変数を特定
                defined_vars_after_attempt = set(context.variable_manager.get_all_variable_names())
                newly_defined_vars = defined_vars_after_attempt - defined_vars_before_attempt
                unused_newly_defined_vars = newly_defined_vars - used_vars_after_attempt
                if unused_newly_defined_vars:
                    print(f"[UNUSED_VAR_DETECTED] 試行{attempt_count}/{max_attempts}: 変数定義ノードが生成されたが使用されていない変数: {unused_newly_defined_vars}", file=sys.stderr, flush=True)
                if newly_marked_vars:
                    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                        print(f"[DEBUG] ノード生成失敗: この試行で追加された変数の使用マークをリセット: {newly_marked_vars}", flush=True)
                    context.variable_manager.unmark_variables_usage(list(newly_marked_vars))

        # ============================================================================
        # セクション4: フォールバック処理（すべての試行が失敗した場合）
        # ============================================================================
        if not node_generated:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[WARNING] すべての代替候補が失敗した。最後の手段としてEMPTY_ARRAYを生成します", flush=True)
            # 最後の手段: EMPTY_ARRAYを生成（常に生成可能）
            try:
                new_array = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array)
                empty_array_node = self._generate_empty_array_node(new_array, context)
                add_node_to_list(nodes_in_step, empty_array_node, variable=new_array, context=context, is_first_override=is_first)
                current_arrays.append(new_array)
                node_generated = True
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[WARNING] 最後の手段: EMPTY_ARRAYノード '{new_array}' を生成しました", flush=True)
            except Exception as e:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[ERROR] 最後の手段のEMPTY_ARRAY生成も失敗: {e}", flush=True)
                # さらに最後の手段: 最小限のノード生成を試行
                # 未使用変数の統計を出力
                all_vars = set(context.variable_manager.get_all_variable_names())
                used_vars = set(context.variable_manager.get_used_variable_names())
                unused_vars = all_vars - used_vars
                if unused_vars:
                    print(f"[UNUSED_VAR_SUMMARY] すべての試行が失敗した時点での未使用変数: {len(unused_vars)}個 - {sorted(list(unused_vars))[:20]}", file=sys.stderr, flush=True)
                raise RuntimeError(f"すべてのノード生成試行が失敗しました。excluded_types={excluded_types}, attempt_count={attempt_count}")


        if random.random() < 0.5:
                # 依存関係解決と検証処理
                nodes = self._resolve_dependencies_with_validation(nodes, nodes_in_step, context, current_arrays)
                nodes_in_step: NodeListWithVarInfo = []

        return nodes, nodes_in_step, current_arrays, current_objects

    # ============================================================================
    # セクション: ノードタイプ別の選択処理メソッド
    # ============================================================================

    def _handle_for_selection(
        self,
        nodes: NodeListWithVarInfo,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        can_generate_for: bool,
        excluded_types: List[str],
        empty_arrays: Optional[List[Tuple[str, int]]] = None
    ) -> Tuple[bool, NodeListWithVarInfo, NodeListWithVarInfo, List[str], List[str]]:
        """FORループノード生成処理

        Args:
            nodes: 現在のノードリスト
            nodes_in_step: ステップ内のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            can_generate_for: FORループ生成可能フラグ
            excluded_types: 除外タイプリスト
            empty_arrays: EMPTY_ARRAYノードのリスト（(配列名, スコープネスト深度)のタプル）

        Returns:
            (node_generated, nodes, nodes_in_step, current_arrays, current_objects)
        """
        if not can_generate_for:
            self._add_to_excluded_types("for", excluded_types, "can_generate_for=False")
            return (False, nodes, nodes_in_step, current_arrays, current_objects)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] FORループ生成開始: current_arrays={current_arrays}", flush=True)

        # 依存関係解決と検証処理
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)

        # 利用可能な配列を取得（既存のfor_arraysと重複しないもの、使用禁止変数を除外）
        try:
            unavailable_vars = context.variable_manager.get_unavailable_variables()
        except Exception:
            unavailable_vars = set()
        if random.random() > 0.01:
            available_arrays = [arr for arr in current_arrays if arr not in context.for_arrays and arr not in unavailable_vars]
        else:
            available_arrays = [arr for arr in current_arrays if arr not in unavailable_vars]

        # EMPTY_ARRAYノードが存在するかチェック（ノードの型情報から直接判定）
        # has_empty_array = any(isinstance(node, EmptyArrayNode) for node, _, _ in nodes)
        # empty_arraysが引数で渡されていない場合は新規作成
        if empty_arrays is None:
            empty_arrays = []  # List[Tuple[str, int]]: (配列名, スコープネスト深度)

        # 現在のスコープネスト深度を取得（複数回使用するため最初に1回だけ取得）
        current_scope_depth = context.get_scope_nesting_depth()

        # EMPTY_ARRAYノード、一定確率で追加
        if random.random() < EMPTY_ARRAY_ADDITION_PROB:
            new_array = self._generate_new_array_variable_name(context)
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, new_array)
            empty_array_node = self._generate_empty_array_node(new_array, context)
            add_node_to_list(nodes_in_step, empty_array_node, variable=new_array, context=context, is_first_override=is_first)
            empty_arrays.append((new_array, current_scope_depth))
            # empty_arraysの変数を使用禁止変数に登録（_resolve_dependencies2内で既存変数として使われないようにするため）
            context.variable_manager.add_unavailable_variable(new_array)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] FORループ生成: EMPTY_ARRAYノード '{new_array}' を追加しました（スコープネスト深度: {current_scope_depth}）", flush=True)

        # 現在のネスト深度で定義されたempty_arraysの要素数をチェック
        current_depth_empty_arrays = [arr for arr, depth in empty_arrays if depth == current_scope_depth]
        if len(current_depth_empty_arrays) > 0:
            use_count_variable = random.random() > 0.2
        else:
            # EMPTY_ARRAYがない場合は配列を使用
            use_count_variable = True

        # MATCH_PAIRS配列がある場合は優先的に使用
        available_match_pairs_arrays = [arr for arr in context.match_pairs_arrays if arr in available_arrays]
        if available_match_pairs_arrays and use_count_variable:
            # MATCH_PAIRS配列を使用したFORループを生成
            match_pairs_array = random.choice(available_match_pairs_arrays)
            # ネスト開始
            for_node = self._generate_for_loop_with_match_pairs(match_pairs_array, context)
            nodes = self._generate_for_loop_and_setup(for_node, match_pairs_array, nodes_in_step, nodes, context, current_arrays)
        elif available_arrays and use_count_variable:
            # 利用可能な配列がある場合は配列のLENを使用
            for_array = random.choice(available_arrays)
            # ネスト開始
            for_node = self._generate_for_loop(for_array, context)
            nodes = self._generate_for_loop_and_setup(for_node, for_array, nodes_in_step, nodes, context, current_arrays)
        else:
            #has_empty_array = any(isinstance(node, EmptyArrayNode) for node, _, _ in nodes)
            # 現在のネスト深度で定義されたempty_arraysの要素数をチェック
            current_depth_empty_arrays = [arr for arr, depth in empty_arrays if depth == current_scope_depth]
            if not len(current_depth_empty_arrays) > 0:
                new_array = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array)
                empty_array_node = self._generate_empty_array_node(new_array, context)
                add_node_to_list(nodes_in_step, empty_array_node, variable=new_array, context=context, is_first_override=is_first)
                empty_arrays.append((new_array, current_scope_depth))
                # empty_arraysの変数を使用禁止変数に登録（_resolve_dependencies2内で既存変数として使われないようにするため）
                context.variable_manager.add_unavailable_variable(new_array)
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] FORループ生成: EMPTY_ARRAYノード '{new_array}' を追加しました（スコープネスト深度: {current_scope_depth}）", flush=True)

            if random.random() < 0.5:
                # 新しいCOUNT型変数を作成
                new_count_var = self._create_new_count_variable(context)
                # ネスト開始
                for_node = self._generate_for_loop_with_count(new_count_var, context)
                nodes = self._generate_for_loop_and_setup(for_node, None, nodes_in_step, nodes, context, current_arrays)
            else:
                # 2～8の定数を使用
                constant_value = random.randint(2, 8)
                # ネスト開始
                for_node = self._generate_for_loop_with_constant(constant_value, context)
                nodes = self._generate_for_loop_and_setup(for_node, None, nodes_in_step, nodes, context, current_arrays)

        # ループ変数を取得してSemanticType.COUNTとして定義済み変数に登録
        loop_var = None
        if hasattr(for_node, 'loop_var'):
            loop_var = for_node.loop_var
        elif hasattr(for_node, 'variable'):
            loop_var = for_node.variable

        if loop_var:
            # ループ変数をSemanticType.LOOP_INDEXとして定義済み変数に登録
            context.variable_manager.define_variable(loop_var, SemanticType.LOOP_INDEX, is_array=False)
            # スコープ情報を記録
            context.add_scope_variable(loop_var)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] FORループ開始: ループ変数 '{loop_var}' をSemanticType.LOOP_INDEXとして登録しました", flush=True)

        # ループ変数を取得してSemanticType.LOOP_INDEXとして定義済み変数に登録
        loop_var = None
        if hasattr(for_node, 'loop_var'):
            loop_var = for_node.loop_var
        elif hasattr(for_node, 'variable'):
            loop_var = for_node.variable

        if loop_var:
            # ループ変数をSemanticType.LOOP_INDEXとして定義済み変数に登録
            context.variable_manager.define_variable(loop_var, SemanticType.LOOP_INDEX, is_array=False)
            # スコープ情報を記録
            context.add_scope_variable(loop_var)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] FORループ開始: ループ変数 '{loop_var}' をSemanticType.LOOP_INDEXとして登録しました", flush=True)

        # ネスト内でノードを生成（制限なし）
        # すべてのFORループ生成パスで実行される必要がある
        nodes_in_step2: NodeListWithVarInfo = []
        iteration_count = 0
        max_nest_iterations = 50  # 無限ループ防止のための最大反復回数

        # ネスト開始時のノード数を記録
        initial_node_count = len(nodes_in_step) + len(nodes_in_step2)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] FORネスト開始: 初期ノード数 = {initial_node_count}", flush=True)

        while not context.can_exit_for_nesting():
            # 無限ループ防止: 最大反復回数に達した場合は強制終了
            if iteration_count >= max_nest_iterations:
                if ENABLE_ALL_LOGS:
                    print(f"[WARNING] FORネスト最大反復回数({max_nest_iterations})に達したため強制終了", flush=True)
                # node_countを更新して終了条件を満たす
                if context.nest_stack:
                    current_node_count = len(nodes_in_step) + len(nodes_in_step2)
                    actual_nodes = max(2, current_node_count - initial_node_count)  # 最低2ノードとしてカウント
                    context.nest_stack[-1]['node_count'] = actual_nodes
                break

            # ネスト内ノード数を実際の生成数から計算
            if context.nest_stack:
                current_node_count = len(nodes_in_step) + len(nodes_in_step2)
                actual_nodes = current_node_count - initial_node_count
                context.nest_stack[-1]['node_count'] = actual_nodes
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] FORネスト中: 現在ノード数 = {current_node_count}, 生成ノード数 = {actual_nodes}, 反復回数 = {iteration_count + 1}", flush=True)

            nodes_in_step, nodes_in_step2, current_arrays, current_objects = self._resolve_dependencies2(nodes_in_step, nodes_in_step2, context, current_arrays, current_objects, empty_arrays=empty_arrays)
            iteration_count += 1

        extend_node_list_with_var_info(nodes_in_step, nodes_in_step2)
        nodes_in_step2: NodeListWithVarInfo = []

        # ネスト終了時の最終ノード数を記録
        if context.nest_stack:
            final_node_count = len(nodes_in_step) + len(nodes_in_step2)
            context.nest_stack[-1]['node_count'] = final_node_count - initial_node_count

        # empty_arraysから現在のネスト深度で定義された要素の使用禁止を解除
        # _process_common_arrays_for_concatなどで使えるようにするため
        current_depth_empty_arrays = [arr for arr, depth in empty_arrays if depth == current_scope_depth]
        for empty_array_name in current_depth_empty_arrays:
            context.variable_manager.remove_unavailable_variable(empty_array_name)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] FORループ終了: EMPTY_ARRAYノード '{empty_array_name}' の使用禁止を解除しました", flush=True)

        # ネストスコープ変数を取得
        common_arrays, common_objects, other_arrays = self._get_nest_scope_variables(current_arrays, current_objects, context)

        # empty_arraysから1つを選ぶ（存在する場合）
        # empty_arraysは(配列名, スコープネスト深度)のタプルのリスト
        selected_empty_array = random.choice(empty_arrays)[0] if empty_arrays else None

        if not other_arrays:
            # 共通配列に対してCONCAT/EXCLUDEノードを生成
            self._process_common_arrays_for_concat(common_arrays, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # common_objectsの処理（使用済みチェック付き）
            self._process_common_objects_for_append(common_objects, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # MERGE処理（論理的に整合性を保つ）
            self._process_merge_operation(common_arrays, nodes_in_step, current_arrays, current_objects, context)
        else:
            # 共通配列に対してCONCAT/EXCLUDEノードを生成
            self._process_common_arrays_for_concat(common_arrays, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # common_objectsの処理
            self._process_common_objects_for_append(common_objects, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)



        end_node = self._generate_end(context)
        add_node_to_list(nodes_in_step, end_node, context=context)
        context.exit_for_nesting()

        # ループ変数を登録から除外
        if loop_var:
            # ループ変数を削除（定義済み変数から除外）
            context.variable_manager.remove_variables([loop_var])
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] FORループ終了: ループ変数 '{loop_var}' を登録から除外しました", flush=True)

        # empty_arraysから現在のネスト深度で定義された要素を除外し、current_arraysに追加
        excluded_arrays = [arr for arr, depth in empty_arrays if depth == current_scope_depth]
        empty_arrays = [(arr, depth) for arr, depth in empty_arrays if depth != current_scope_depth]
        # 除外した配列をcurrent_arraysに追加
        for excluded_array in excluded_arrays:
            if excluded_array not in current_arrays:
                current_arrays.append(excluded_array)

        # FORループ終了時にfor_arraysから削除
        popped_array = context.for_arrays.pop()
        # MATCH_PAIRS配列の場合、match_pairs_arraysからも削除
        if popped_array and popped_array in context.match_pairs_arrays:
            context.match_pairs_arrays.remove(popped_array)

        # nodes_in_stepで初めて定義された変数を取得してvariable_managerから一括削除
        first_defined_vars = list(get_first_defined_variables(nodes_in_step))
        if first_defined_vars:
            context.variable_manager.remove_variables(first_defined_vars)

        # 依存関係解決と検証処理
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)
        return (True, nodes, nodes_in_step, current_arrays, current_objects)

    def _handle_if_selection(
        self,
        nodes: NodeListWithVarInfo,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        excluded_types: List[str],
        empty_arrays: Optional[List[Tuple[str, int]]] = None
    ) -> Tuple[bool, NodeListWithVarInfo, NodeListWithVarInfo, List[str], List[str]]:
        """IF文ノード生成処理

        Args:
            nodes: 現在のノードリスト
            nodes_in_step: ステップ内のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            excluded_types: 除外タイプリスト
            empty_arrays: EMPTY_ARRAYノードのリスト（(配列名, スコープネスト深度)のタプル）

        Returns:
            (node_generated, nodes, nodes_in_step, current_arrays, current_objects)
        """
        # IF文生成条件チェック（優先順位順）
        # チェック順序:
        # 1. 構造的ネスト深度チェック（最優先）- 予測深度が最大深度を超える場合は絶対に禁止
        # 2. 個数制限チェック - IF文の総数が上限を超えていないか
        # 3. その他の条件 - コマンド使用可否など
                # 注意: enter_if_nesting()が呼ばれると深度が1増えるため、その前の時点でチェックする
        can_generate_if_depth, current_scope_depth_if, max_scope_depth_if, predicted_depth_after_if = self._check_scope_nesting_depth(context, "if")

        # 1. 構造的ネスト深度チェック（最優先）
        if not can_generate_if_depth:
            can_generate_if = False  # 予測深度が最大深度を超える場合は絶対に禁止
        else:
            # 2. スコープネスト深度が許容範囲内の場合のみ、個数制限とその他の条件をチェック
            can_generate_if = (
                context.can_add_if_statement() and  # 個数制限チェック
                context.can_use_command("IF")
            )

        # 条件を満たさない場合は除外リストに追加して次の試行へ
        if not can_generate_if:
            self._add_to_excluded_types("if", excluded_types, "can_generate_if=False")
            return (False, nodes, nodes_in_step, current_arrays, current_objects)

        # 依存関係解決と検証処理
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)
        # IF分岐を生成（配列に依存しない）
        # ネスト開始
        if_node = self._generate_if_branch(context)
        add_node_to_list(nodes_in_step, if_node, context=context)

        # 依存関係解決と検証処理
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)
        # FORループを生成
        # 既存のfor_arraysと重複しない配列を選択
        nodes_in_step2: NodeListWithVarInfo = []

        # ネスト内でノードを生成（制限なし）
        iteration_count = 0
        max_nest_iterations = 50  # 無限ループ防止のための最大反復回数

        # ネスト開始時のノード数を記録
        initial_node_count = len(nodes_in_step) + len(nodes_in_step2)

        while not context.can_exit_if_nesting():
            # 無限ループ防止: 最大反復回数に達した場合は強制終了
            if iteration_count >= max_nest_iterations:
                # node_countを更新して終了条件を満たす
                if context.nest_stack:
                    current_node_count = len(nodes_in_step) + len(nodes_in_step2)
                    actual_nodes = max(2, current_node_count - initial_node_count)  # 最低2ノードとしてカウント
                    context.nest_stack[-1]['node_count'] = actual_nodes
                break

            # ネスト内ノード数を実際の生成数から計算
            if context.nest_stack:
                current_node_count = len(nodes_in_step) + len(nodes_in_step2)
                context.nest_stack[-1]['node_count'] = current_node_count - initial_node_count

            nodes_in_step, nodes_in_step2, current_arrays, current_objects = self._resolve_dependencies2(nodes_in_step, nodes_in_step2, context, current_arrays, current_objects, empty_arrays=None)
            iteration_count += 1

        extend_node_list_with_var_info(nodes_in_step, nodes_in_step2)
        nodes_in_step2: NodeListWithVarInfo = []

        # ネスト終了時の最終ノード数を記録
        if context.nest_stack:
            final_node_count = len(nodes_in_step) + len(nodes_in_step2)
            context.nest_stack[-1]['node_count'] = final_node_count - initial_node_count

        # ネストスコープ変数を取得
        common_arrays, common_objects, other_arrays = self._get_nest_scope_variables(current_arrays, current_objects, context)

        # empty_arraysから1つを選ぶ（存在する場合）
        # empty_arraysは(配列名, スコープネスト深度)のタプルのリスト
        selected_empty_array = random.choice(empty_arrays)[0] if empty_arrays else None

        if not other_arrays:
            # 共通配列に対してCONCAT/EXCLUDEノードを生成
            self._process_common_arrays_for_concat(common_arrays, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # common_objectsの処理
            self._process_common_objects_for_append(common_objects, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # MERGE処理
            self._process_merge_operation(common_arrays, nodes_in_step, current_arrays, current_objects, context)
        else:
            # 共通配列に対してCONCAT/EXCLUDEノードを生成
            self._process_common_arrays_for_concat(common_arrays, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)
            # common_objectsの処理
            self._process_common_objects_for_append(common_objects, nodes_in_step, current_arrays, current_objects, context, empty_array=selected_empty_array)

        end_node = self._generate_end(context)
        add_node_to_list(nodes_in_step, end_node, context=context)

        # exit_if_nesting()で自動的にスコープ変数が削除される
        context.exit_if_nesting()
        # nodes_in_stepで初めて定義された変数を取得してvariable_managerから一括削除
        first_defined_vars = list(get_first_defined_variables(nodes_in_step))
        if first_defined_vars:
            context.variable_manager.remove_variables(first_defined_vars)

        # 依存関係解決と検証処理
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)
        return (True, nodes, nodes_in_step, current_arrays, current_objects)

    def _handle_element_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        excluded_types: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo]:
        """配列要素代入ノード生成処理

        Returns:
            (node_generated, nodes_in_step)
        """
        # FORループ内でない場合は除外リストに追加して次の試行へ
        if context.get_for_nesting_depth() == 0:
            self._add_to_excluded_types("element", excluded_types, "FORループ内でない")
            return (False, nodes_in_step)

        # 配列代入ノード数が上限に達している場合は除外リストに追加して次の試行へ
        current_count = context.get_current_for_array_assignment_count()
        if current_count >= context.max_for_array_assignments:
            self._add_to_excluded_types("element", excluded_types, f"配列代入ノード数が上限に達しました: {current_count}/{context.max_for_array_assignments}")
            return (False, nodes_in_step)

        depth = context.get_for_nesting_depth()
        # 有効な配列（Noneでない）のインデックスのみを取得
        valid_indices = []
        for i in range(depth):
            if i < len(context.for_arrays) and context.for_arrays[i] is not None:
                valid_indices.append(i)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] element選択: depth={depth}, for_arrays={context.for_arrays}, valid_indices={valid_indices}, current_array_assignment_count={current_count}/{context.max_for_array_assignments}")

        if valid_indices:
            # 重みをインデックスの値に比例させる（例: 1, 2, 3, ...）
            weights = [i + 1 for i in valid_indices]  # +1 は 0 を避けるため
            # 重み付きランダム選択
            loop_idx = random.choices(valid_indices, weights=weights, k=1)[0]
            for_array = context.for_arrays[loop_idx]
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] 配列要素代入生成: for_array={for_array}, loop_idx={loop_idx}", flush=True)
            element_node = self._generate_array_element_assignment(for_array, loop_idx, context)
            add_node_to_list(nodes_in_step, element_node, context=context)
            return (True, nodes_in_step)
        else:
            # 有効な配列がない場合（COUNT変数のみのFORループ）でも、デバッグ情報を出力
            # COUNT変数のみのFORループでは配列要素代入は生成できない
            # この場合も除外リストに追加
            self._add_to_excluded_types("element", excluded_types, "有効な配列なし")
            return (False, nodes_in_step)

    def _handle_concat_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """CONCATノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if len(candidates) < 2:
            return (False, nodes_in_step, current_arrays, current_objects)

        sample_size = min(2, len(candidates))
        if sample_size > 0 and len(candidates) > 0:
            concat_arrays = random.sample(candidates, sample_size)
        else:
            concat_arrays = []
        # ネスト深度>0の場合のみ、ネスト外変数を0番目に配置
        if context.get_scope_nesting_depth() > 0:
            # ネスト外変数とネスト内変数の判定
            current_nest_object_arrays = context.get_current_scope_object_arrays()
            is_outer_0 = concat_arrays[0] not in current_nest_object_arrays
            is_outer_1 = concat_arrays[1] not in current_nest_object_arrays

            # ネスト外変数とネスト内変数の場合、ネスト外変数を0番目に配置
            if not is_outer_0 and is_outer_1:
                # ネスト内変数とネスト外変数の場合、入れ替え
                concat_arrays[0], concat_arrays[1] = concat_arrays[1], concat_arrays[0]

        # CONCATノードを生成
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, concat_arrays[0])
        concat_node = self._generate_concat_node(concat_arrays[0], concat_arrays[1], concat_arrays[0], context)
        add_node_to_list(nodes_in_step, concat_node, variable=concat_arrays[0], context=context, is_first_override=is_first)
        self._safe_remove_from_arrays(concat_arrays[1], current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_exclude_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """EXCLUDEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if len(candidates) < 2:
            return (False, nodes_in_step, current_arrays, current_objects)

        sample_size = min(2, len(candidates))
        if sample_size > 0 and len(candidates) > 0:
            exclude_arrays = random.sample(candidates, sample_size)
        else:
            exclude_arrays = []
        # ネスト深度>0の場合のみ、ネスト外変数を0番目に配置
        if context.get_scope_nesting_depth() > 0:
            # ネスト外変数とネスト内変数の判定
            current_nest_object_arrays = context.get_current_scope_object_arrays()
            is_outer_0 = exclude_arrays[0] not in current_nest_object_arrays
            is_outer_1 = exclude_arrays[1] not in current_nest_object_arrays

            # ネスト外変数とネスト内変数の場合、ネスト外変数を0番目に配置
            if not is_outer_0 and is_outer_1:
                # ネスト内変数とネスト外変数の場合、入れ替え
                exclude_arrays[0], exclude_arrays[1] = exclude_arrays[1], exclude_arrays[0]

        # EXCLUDEノードを生成
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, exclude_arrays[0])
        exclude_node = self._generate_exclude_node(exclude_arrays[0], exclude_arrays[0], exclude_arrays[1], context)
        add_node_to_list(nodes_in_step, exclude_node, variable=exclude_arrays[0], context=context, is_first_override=is_first)
        self._safe_remove_from_arrays(exclude_arrays[1], current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_create_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str],
        excluded_types: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """CREATEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        new_object = self._generate_new_object_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_object)
        create_node = self._generate_create_objects(new_object, context)
        if create_node is None:
            # ノード生成に失敗した場合
            self._add_to_excluded_types("create", excluded_types, "ノード生成失敗")
            return (False, nodes_in_step, current_arrays, current_objects)
        current_objects.append(new_object)
        add_node_to_list(nodes_in_step, create_node, variable=new_object, context=context, is_first_override=is_first)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_merge_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """MERGEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        objects_array = random.choice(candidates)
        new_object = self._generate_new_object_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_object)
        merge_node = self._generate_merge_node(objects_array, new_object, context)
        current_objects.append(new_object)
        add_node_to_list(nodes_in_step, merge_node, variable=new_object, context=context, is_first_override=is_first)
        self._safe_remove_from_arrays(objects_array, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_empty_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str]]:
        """EMPTY_ARRAYノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays)
        """
        new_array = self._generate_new_array_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        empty_array_node = self._generate_empty_array_node(new_array, context)
        add_node_to_list(nodes_in_step, empty_array_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        return (True, nodes_in_step, current_arrays)

    def _handle_append_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """APPENDノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates or not current_objects:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_array = random.choice(candidates)
        append_object = random.choice(current_objects)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, current_array)
        append_node = self._generate_append_node(current_array, append_object, current_array, context)
        add_node_to_list(nodes_in_step, append_node, variable=current_array, context=context, is_first_override=is_first)
        self._safe_remove_from_objects(append_object, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_reverse_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """REVERSEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        # REVERSE: 配列を逆順にする
        current_array = random.choice(candidates)
        new_array = self._generate_new_array_variable_name(context)
        # REVERSEコマンドで変数定義ノードを生成
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        reverse_node = self.node_generators._generate_variable_definition_node_with_command(
            new_array,
            "REVERSE",
            context,
            provided_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True)
        )
        add_node_to_list(nodes_in_step, reverse_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_tile_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """TILEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not current_objects:
            return (False, nodes_in_step, current_arrays, current_objects)

        # TILE: オブジェクトをタイル状に配置
        current_object = random.choice(current_objects)
        new_array = self._generate_new_array_variable_name(context)
        # TILEコマンドで変数定義ノードを生成
        # TILEは obj, count_x, count_y の3引数
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        tile_node = self.node_generators._generate_variable_definition_node_with_command(
            new_array,
            "TILE",
            context,
            provided_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True)
        )
        add_node_to_list(nodes_in_step, tile_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        self._safe_remove_from_objects(current_object, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_extract_shape_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """EXTRACT_SHAPEノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not current_objects:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_object = random.choice(current_objects)
        new_array = self._generate_new_array_variable_name(context)
        # EXTRACT_RECTS, EXTRACT_LINES, EXTRACT_HOLLOW_RECTSのみ（配列を返す）
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        extract_type = random.choice(["rects", "hollow_rects", "lines"])
        extract_node = self._generate_extract_shape_node(current_object, new_array, extract_type, context)
        add_node_to_list(nodes_in_step, extract_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_extend_pattern_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """EXTEND_PATTERNノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_array = random.choice(candidates)
        new_array = self._generate_new_array_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        side = random.choice(["start", "end"])
        count = random.randint(1, 5)
        extend_node = self._generate_extend_pattern_node(current_array, new_array, context, side, count)
        add_node_to_list(nodes_in_step, extend_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_arrange_grid_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """ARRANGE_GRIDノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_array = random.choice(candidates)
        new_array = self._generate_new_array_variable_name(context)

        # グリッドサイズに応じて値を決定
        grid_width = getattr(context, 'output_grid_width', 15)
        grid_height = getattr(context, 'output_grid_height', 15)

        # cols: 1からmin(5, grid_width//2)の範囲（グリッド幅に応じて調整）
        max_cols = max(1, min(5, grid_width // 3))
        cols = random.randint(1, max_cols)

        # width: 1からgrid_widthの範囲（グリッド幅に応じて調整）
        max_width = max(1, min(15, grid_width))
        width = random.randint(1, max_width)

        # height: 1からgrid_heightの範囲（グリッド高さに応じて調整）
        max_height = max(1, min(15, grid_height))
        height = random.randint(1, max_height)

        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        arrange_node = self._generate_arrange_grid_node(current_array, new_array, context, cols, width, height)
        add_node_to_list(nodes_in_step, arrange_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_single_object_array_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """SINGLE_OBJECT_ARRAYノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not current_objects:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_object = random.choice(current_objects)
        new_array = self._generate_new_array_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        single_object_array_node = self._generate_single_object_array_node(new_array, current_object, context)
        add_node_to_list(nodes_in_step, single_object_array_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        self._safe_remove_from_objects(current_object, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_split_connected_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """SPLIT_CONNECTEDノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not current_objects:
            return (False, nodes_in_step, current_arrays, current_objects)

        current_object = random.choice(current_objects)
        new_array = self._generate_new_array_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)
        connectivity = random.choice([4, 8])
        split_connected_node = self._generate_split_connected_node(current_object, new_array, context, connectivity)
        add_node_to_list(nodes_in_step, split_connected_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        self._safe_remove_from_objects(current_object, current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_object_operations_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        excluded_types: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """OBJECT_OPERATIONSノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        # オブジェクト操作ノードを生成（オブジェクト操作コマンドのみに限定）
        new_object = self._generate_new_object_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_object)

        # _generate_object_operations_nodeを使用してオブジェクト操作ノードを生成
        object_operation_node = self._generate_object_operations_node(new_object, context)
        if object_operation_node is None:
            # ノード生成に失敗した場合
            self._add_to_excluded_types("object_operations", excluded_types, "ノード生成失敗")
            return (False, nodes_in_step, current_arrays, current_objects)

        add_node_to_list(nodes_in_step, object_operation_node, variable=new_object, context=context, is_first_override=is_first)
        current_objects.append(new_object)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_match_pairs_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """MATCH_PAIRSノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if context.get_for_nesting_depth() >= 1 or len(candidates) < 2:
            return (False, nodes_in_step, current_arrays, current_objects)

        # 2つの配列をランダム選択
        sample_size = min(2, len(candidates))
        if sample_size > 0 and len(candidates) > 0:
            match_arrays = random.sample(candidates, sample_size)
        else:
            match_arrays = []
        new_array = self._generate_new_array_variable_name(context)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, new_array)

        # MATCH_PAIRSノードを生成（条件生成は内部で実行）
        match_pairs_node = self._generate_match_pairs_node(match_arrays[0], match_arrays[1], new_array, context)
        add_node_to_list(nodes_in_step, match_pairs_node, variable=new_array, context=context, is_first_override=is_first)
        current_arrays.append(new_array)
        # MATCH_PAIRS配列を追跡リストに追加
        context.match_pairs_arrays.append(new_array)
        self._safe_remove_from_arrays(match_arrays[0], current_arrays, current_objects, context)
        self._safe_remove_from_arrays(match_arrays[1], current_arrays, current_objects, context)
        return (True, nodes_in_step, current_arrays, current_objects)

    def _handle_define_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext
    ) -> Tuple[bool, NodeListWithVarInfo]:
        """DEFINEノード生成処理（変数定義）

        オブジェクトとオブジェクト配列以外の型をランダムに選び、新しい変数名を生成して変数定義ノードを生成

        Returns:
            (node_generated, nodes_in_step)
        """
        from ..metadata.variable_naming import VariableNamingSystem

        # オブジェクトとオブジェクト配列以外の型のリスト
        non_object_types = [
            SemanticType.COLOR,
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
            SemanticType.DISTANCE,
            SemanticType.COUNT,
            SemanticType.COUNT_HOLES,
            SemanticType.COUNT_ADJACENT,
            SemanticType.COUNT_OVERLAP,
            SemanticType.PERCENTAGE,
            SemanticType.ASPECT_RATIO,
            SemanticType.DENSITY,
            SemanticType.ANGLE,
            SemanticType.SCALE,
            SemanticType.BOOL,
            SemanticType.STRING,
            SemanticType.DIRECTION,
            SemanticType.AXIS,
            SemanticType.LINE_TYPE,
            SemanticType.RECT_TYPE,
            SemanticType.SIDE,
            SemanticType.ORDER,
            SemanticType.ALIGN_MODE,
        ]

        # 型をランダムに選択
        selected_type = random.choice(non_object_types)
        is_array = random.choice([False, True])  # 配列かどうかもランダムに選択

        # 新しい変数名を生成
        type_info = TypeInfo.create_from_semantic_type(selected_type, is_array=is_array)
        naming_system = VariableNamingSystem()
        used_vars = set(context.variable_manager.get_all_variable_names())
        var_name = naming_system.get_next_variable_name(type_info, used_vars)

        try:
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, var_name)
            # 変数定義ノードを生成
            var_def_node = self._generate_variable_definition_node(var_name, context, provided_type_info=type_info)
            add_node_to_list(nodes_in_step, var_def_node, variable=var_name, context=context, is_first_override=is_first)
            return (True, nodes_in_step)
        except Exception as e:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[ERROR] DEFINEノード生成失敗: {type(e).__name__}: {e}", flush=True)
            return (False, nodes_in_step)

    def _handle_redefine_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext
    ) -> Tuple[bool, NodeListWithVarInfo]:
        """REDEFINEノード生成処理（変数再定義）

        定義済み変数や使用済み変数から、オブジェクトとオブジェクト配列以外の変数をランダムで1つ選び、変数定義ノードで再定義

        Returns:
            (node_generated, nodes_in_step)
        """
        # 定義済み変数と使用済み変数を取得
        defined_vars = set(context.variable_manager.get_defined_variable_names())
        used_vars = set(context.variable_manager.get_used_variable_names())
        candidate_vars = defined_vars | used_vars

        # ループ変数（i, j, k, l, m, n）を定義
        loop_vars = ['i', 'j', 'k', 'l', 'm', 'n']

        # オブジェクトとオブジェクト配列以外の変数をフィルタリング
        non_object_vars = []
        for var_name in candidate_vars:
            # ループ変数の再定義は禁止
            if var_name in loop_vars:
                continue

            var_info = context.variable_manager.get_variable_info(var_name)
            if var_info and 'type_info' in var_info:
                type_info = var_info['type_info']
                # オブジェクト型でない、またはオブジェクト型だが配列でない場合は除外
                if type_info.semantic_type != SemanticType.OBJECT:
                    # オブジェクト型以外は追加
                    non_object_vars.append(var_name)
                elif type_info.semantic_type == SemanticType.OBJECT and not type_info.is_array:
                    # オブジェクト型だが配列でない（単一オブジェクト）は除外
                    pass
                # オブジェクト配列は除外（条件に該当しない）

        if not non_object_vars:
            # 候補がない場合は失敗
            return (False, nodes_in_step)

        # ランダムに1つ選択
        selected_var = random.choice(non_object_vars)
        var_info = context.variable_manager.get_variable_info(selected_var)
        type_info = var_info['type_info'] if var_info and 'type_info' in var_info else None

        if not type_info:
            return (False, nodes_in_step)

        try:
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            # 再定義なので、is_firstはFalseになるはず
            is_first = is_first_definition(context.variable_manager.tracker, selected_var)
            # 変数定義ノードを生成（再定義）
            var_def_node = self._generate_variable_definition_node(selected_var, context, provided_type_info=type_info)
            if var_def_node is None:
                # 変数定義ノード生成に失敗した場合
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[ERROR] REDEFINEノード生成失敗: var_name={selected_var}", flush=True)
                return (False, nodes_in_step)
            add_node_to_list(nodes_in_step, var_def_node, variable=selected_var, context=context, is_first_override=is_first)
            return (True, nodes_in_step)
        except Exception as e:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[ERROR] REDEFINEノード生成失敗: {type(e).__name__}: {e}", flush=True)
            return (False, nodes_in_step)

    def _maybe_add_object_access_node(self, target_array: str, nodes_in_step: NodeListWithVarInfo,
                                      current_arrays: List[str], current_objects: List[str],
                                      context: ProgramContext, probability: float = 0.01) -> None:
        """確率的にobject_accessノードを追加（共通処理）

        Args:
            target_array: 対象配列
            nodes_in_step: ノードリスト
            current_arrays: 配列変数リスト
            current_objects: オブジェクト変数リスト
            context: プログラムコンテキスト
            probability: 追加確率（デフォルト: 0.01）
        """
        if random.random() < probability:
            new_object = self._generate_new_object_variable_name(context)
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, new_object)
            access_type = random.choice(["first", "last"])
            object_access_node = self._generate_object_access_node(new_object, target_array, access_type, context)
            add_node_to_list(nodes_in_step, object_access_node, variable=new_object, context=context, is_first_override=is_first)
            current_objects.append(new_object)
            self._safe_remove_from_arrays(target_array, current_arrays, current_objects, context)

    def _calculate_weighted_count(self, context: ProgramContext, max_pairs_by_complexity: Dict[int, int]) -> int:
        """複雑度に基づいて重み付きカウントを計算（共通処理）

        Args:
            context: プログラムコンテキスト
            max_pairs_by_complexity: 複雑度ごとの最大ペア数の辞書

        Returns:
            選択されたカウント数（最小値は2）
        """
        complexity = context.complexity
        max_pairs = max_pairs_by_complexity.get(complexity, 10)  # デフォルトは10
        # 最小値を2に保証
        max_pairs = max(2, max_pairs)

        # 上限までの選択肢と重みを生成（最小値は2）
        select_num_types = list(range(2, max_pairs + 1))
        # 重み: 2-5は高確率、6以上は低確率
        base_weights = [1.0, 4.0, 3.0, 1.0]
        additional_weights = [0.05, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005]

        if len(select_num_types) <= 4:
            num_weights = base_weights[:len(select_num_types)]
        else:
            num_weights = base_weights + additional_weights[:len(select_num_types) - 4]

        return random.choices(select_num_types, weights=num_weights, k=1)[0]

    def _handle_filter_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str],
        previous_last_selected: Optional[str] = None
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """FILTERノード生成処理

        Args:
            previous_last_selected: 更新前のlast_selected値（初期ノード直後かどうかを判定するため）

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates or not context.can_use_command("FILTER"):
            return (False, nodes_in_step, current_arrays, current_objects)

        select_filter_types = ["existing", "new", "invert"]
        # last_node_typeに基づいてfilter_weightsを変更
        # previous_last_selectedを優先的に使用（更新前の値を参照するため）
        last_selected = previous_last_selected if previous_last_selected is not None else getattr(context, 'last_selected', None)
        if last_selected == 'get_objects':
            # GET_OBJECTS後のfilter: existingを優先
            filter_weights = [0.01, 1.0, 1.0]  # existingを高確率、invertを低確率
        else:
            # それ以外のfilter: デフォルトの重み
            filter_weights = [1.0, 0.01, 0.01]  # 合計は1でなくてもOK（比率で判断される）

        selected_filter = random.choices(select_filter_types, weights=filter_weights, k=1)[0]
        current_array = random.choice(candidates)
        node_generated = False  # 初期化

        if selected_filter == "existing":
            print(f"[FILTER_SELECT] selected_filter='existing': current_array={current_array}", file=sys.stderr, flush=True)
            try:
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, current_array)
                filter_node = self._generate_advanced_filter(current_array, current_array, context)
                print(f"[FILTER_SELECT] existingフィルタ生成成功: filter_node.condition={filter_node.condition if hasattr(filter_node, 'condition') else 'N/A'}", file=sys.stderr, flush=True)
                add_node_to_list(nodes_in_step, filter_node, variable=current_array, context=context, is_first_override=is_first)
                node_generated = True  # ノード生成成功
            except Exception as e:
                print(f"[FILTER_SELECT_ERROR] existingフィルタ生成失敗: error={type(e).__name__}: {e}", file=sys.stderr, flush=True)
                print(f"[FILTER_SELECT_ERROR] スタックトレース: {traceback.format_exc()}", file=sys.stderr, flush=True)
                raise

            #self._maybe_add_object_access_node(current_array, nodes_in_step, current_arrays, current_objects, context, 0.01)

        elif selected_filter == "new":
            # 新しい配列変数名を生成
            # 複雑度ごとに上限を設定
            max_pairs_by_complexity = {
                1: 4,   # Very Simple: 最大4ペア
                2: 4,   # Simple: 最大4ペア
                3: 4,   # Medium: 最大4ペア
                4: 5,   # Complex: 最大5ペア
                5: 6,   # Very Complex: 最大6ペア
                6: 8,   # 超複雑: 最大8ペア
                7: 8,   # 超複雑: 最大8ペア
                8: 10,  # 超複雑: 最大10ペア
            }
            num_new_arrays = self._calculate_weighted_count(context, max_pairs_by_complexity)

            for _ in range(num_new_arrays):
                new_array = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array)
                filter_node = self._generate_advanced_filter(current_array, new_array, context)
                add_node_to_list(nodes_in_step, filter_node, variable=new_array, context=context, is_first_override=is_first)
                current_arrays.append(new_array)

                #self._maybe_add_object_access_node(new_array, nodes_in_step, current_arrays, current_objects, context, 0.01)
            self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)
            node_generated = True  # ノード生成成功

        elif selected_filter == "invert":
            # 重み付き乱数で複数の配列ペアを作成
            # 複雑度ごとに上限を設定
            max_pairs_by_complexity = {
                1: 3,   # Very Simple: 最大3ペア
                2: 3,   # Simple: 最大3ペア
                3: 4,   # Medium: 最大4ペア
                4: 4,   # Complex: 最大4ペア
                5: 5,   # Very Complex: 最大5ペア
                6: 7,   # 超複雑: 最大7ペア
                7: 8,   # 超複雑: 最大8ペア
                8: 10,  # 超複雑: 最大10ペア
            }
            num_new_pairs = self._calculate_weighted_count(context, max_pairs_by_complexity)
            new_arrays = []
            for _ in range(num_new_pairs - 1):
                new_array = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array)
                filter_node = self._generate_advanced_filter(current_array, new_array, context)
                add_node_to_list(nodes_in_step, filter_node, variable=new_array, context=context, is_first_override=is_first)
                current_arrays.append(new_array)
                new_arrays.append(new_array)
                # 各配列に対して、1%の確率でobject_accessノードを追加
                #self._maybe_add_object_access_node(new_array, nodes_in_step, current_arrays, current_objects, context, 0.01)
            remove_array = current_array
            for i in range(num_new_pairs - 1):
                new_array2 = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array2)
                exclude_node = self._generate_exclude_node(remove_array, new_array2, new_arrays[i], context)
                add_node_to_list(nodes_in_step, exclude_node, variable=new_array2, context=context, is_first_override=is_first)
                remove_array = new_array2
                # if i == num_new_pairs - 2:
                #     current_arrays.append(new_array2)
                #     self._maybe_add_object_access_node(new_array2, nodes_in_step, current_arrays, current_objects, context, 0.01)

            self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)
            node_generated = True  # ノード生成成功

        return (node_generated, nodes_in_step, current_arrays, current_objects)

    def _handle_sort_selection(
        self,
        nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext,
        current_arrays: List[str],
        current_objects: List[str],
        candidates: List[str]
    ) -> Tuple[bool, NodeListWithVarInfo, List[str], List[str]]:
        """SORTノード生成処理

        Returns:
            (node_generated, nodes_in_step, current_arrays, current_objects)
        """
        if not candidates:
            return (False, nodes_in_step, current_arrays, current_objects)

        select_sort_types = ["existing", "new", "invert"]
        sort_weights = [1.0, 1.0, 1.0]  # 合計は1でなくてもOK（比率で判断される）

        selected_sort = random.choices(select_sort_types, weights=sort_weights, k=1)[0]
        current_array = random.choice(candidates)
        node_generated = False  # 初期化

        if selected_sort == "existing":
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, current_array)
            sort_node = self._generate_sort_node(current_array, current_array, context)
            add_node_to_list(nodes_in_step, sort_node, variable=current_array, context=context, is_first_override=is_first)
            node_generated = True  # ノード生成成功

            # sortノードの後に5%の確率でobject_accessノードを追加
            self._maybe_add_object_access_node(current_array, nodes_in_step, current_arrays, current_objects, context, 0.05)

        elif selected_sort == "new":
            # 新しい配列変数名を生成
            num_new_arrays = random.randint(1, 3)
            for _ in range(num_new_arrays):
                new_array = self._generate_new_array_variable_name(context)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, new_array)
                sort_node = self._generate_sort_node(current_array, new_array, context)
                add_node_to_list(nodes_in_step, sort_node, variable=new_array, context=context, is_first_override=is_first)
                current_arrays.append(new_array)
            self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)

            # sortノードの後に5%の確率でobject_accessノードを追加
            self._maybe_add_object_access_node(new_array, nodes_in_step, current_arrays, current_objects, context, 0.05)
            node_generated = True  # ノード生成成功

        elif selected_sort == "invert":
            new_array = self._generate_new_array_variable_name(context)
            new_array2 = self._generate_new_array_variable_name(context)
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first_sort = is_first_definition(context.variable_manager.tracker, new_array)
            is_first_exclude = is_first_definition(context.variable_manager.tracker, new_array2)
            sort_node = self._generate_sort_node(current_array, new_array, context)
            exclude_node = self._generate_exclude_node(current_array, new_array2, new_array, context)

            add_node_to_list(nodes_in_step, sort_node, variable=new_array, context=context, is_first_override=is_first_sort)
            add_node_to_list(nodes_in_step, exclude_node, variable=new_array2, context=context, is_first_override=is_first_exclude)
            current_arrays.append(new_array)
            current_arrays.append(new_array2)
            self._safe_remove_from_arrays(current_array, current_arrays, current_objects, context)
            # sortノードの後に5%の確率でobject_accessノードを追加
            self._maybe_add_object_access_node(new_array, nodes_in_step, current_arrays, current_objects, context, 0.05)
            self._maybe_add_object_access_node(new_array2, nodes_in_step, current_arrays, current_objects, context, 0.05)
            node_generated = True  # ノード生成成功

        return (node_generated, nodes_in_step, current_arrays, current_objects)





    def _resolve_dependencies3(self, nodes: NodeListWithVarInfo, context: ProgramContext, current_arrays: List[str] = ["objects"]) -> NodeListWithVarInfo:
        """未定義変数の依存関係を解決（改善版）"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3開始: current_arrays={current_arrays}", flush=True)


        # 完全に定義し終わるまでループを続ける
        max_iterations = random.randint(RESOLVE_DEPENDENCIES_MIN_ITERATIONS, RESOLVE_DEPENDENCIES_MAX_ITERATIONS)
        iteration_count = 0
        previous_undefined_vars = []
        nodes_in_step: NodeListWithVarInfo = []
        nodes_in_step2: NodeListWithVarInfo = []

        while iteration_count < max_iterations:

            # 未定義変数を取得
            undefined_vars = self.variable_managers._get_undefined_variables(context)

            for var_name in undefined_vars:
                context.variable_manager.add_unavailable_variable(var_name)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3反復{iteration_count + 1}: 未定義変数={undefined_vars}", flush=True)

            if not undefined_vars:
                # 未定義変数がなくなったら終了
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] _resolve_dependencies3: 未定義変数がなくなったため終了", flush=True)
                break

            # 前回の反復と同じ未定義変数が残っている場合は、より積極的に定義
            if iteration_count > 0 and undefined_vars == previous_undefined_vars:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[DEBUG] _resolve_dependencies3: 同じ未定義変数が残存: {undefined_vars}。積極的定義モードに切り替え", flush=True)
                aggressive_mode = True
            else:
                aggressive_mode = False

            previous_undefined_vars = undefined_vars.copy()

            # 各未定義変数を解決
            for var_name in undefined_vars:
                self._resolve_single_variable(
                    var_name, context, current_arrays, nodes_in_step,
                    aggressive_mode, iteration_count
                )


            # 反復後のノード処理
            self._finalize_iteration_nodes(nodes_in_step, nodes_in_step2, context)
            iteration_count += 1



        # 残りの未定義変数を処理
        self._handle_remaining_undefined_variables(nodes_in_step, nodes_in_step2, context)

        # 最終チェック
        self._handle_final_undefined_variables(nodes_in_step, nodes_in_step2, context)

        # 最終的なノードリストをマージ
        if nodes_in_step2:
            extend_node_list_with_var_info(nodes, nodes_in_step2)

        # 未使用変数の統計を出力
        all_vars = set(context.variable_manager.get_all_variable_names())
        used_vars = set(context.variable_manager.get_used_variable_names())
        unused_vars = all_vars - used_vars
        if unused_vars:
            # [UNUSED_VAR_FINAL]ログは非表示（ENABLE_DEBUG_OUTPUTまたはENABLE_ALL_LOGSが有効な場合のみ表示）
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[UNUSED_VAR_FINAL] _resolve_dependencies3完了時点での未使用変数: {len(unused_vars)}個 - {sorted(list(unused_vars))[:30]}", file=sys.stderr, flush=True)

        return nodes

    def _set_node_metadata(self, node: Node, phase: str, iteration: Optional[int] = None) -> None:
        """ノードに生成起源メタデータを設定"""
        try:
            node.context["gen_origin_phase"] = phase
            if iteration is not None:
                node.context["gen_origin_iteration"] = iteration
        except Exception:
            pass

    def _resolve_single_variable(
        self, var_name: str, context: ProgramContext, current_arrays: List[str],
        nodes_in_step: NodeListWithVarInfo, aggressive_mode: bool, iteration_count: int
    ) -> None:
        """単一の未定義変数を解決"""
        var_info = context.variable_manager.get_variable_info(var_name)
        if not var_info or 'type_info' not in var_info:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[ERROR] _resolve_dependencies3: 変数 '{var_name}' の型情報が見つかりません。var_info={var_info}")
            raise RuntimeError(f"変数 '{var_name}' の型情報が見つかりません。var_info={var_info}")

        type_info = var_info['type_info']
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            is_used = var_info.get('is_used', False)
            print(f"[DEBUG] _resolve_dependencies3: 変数 {var_name} を定義します (使用済み: {is_used}, 積極的モード: {aggressive_mode})", flush=True)
            print(f"[DEBUG] _resolve_dependencies3: 変数 {var_name} の型情報: {type_info}", flush=True)

        # 未使用変数の生成を追跡
        import sys
        is_used = var_info.get('is_used', False) if var_info else False

        # 未使用変数で、積極的モードでない場合は定義をスキップ
        if not is_used and not aggressive_mode:
            print(f"[UNUSED_VAR_SKIP] 未使用変数の定義をスキップ: var_name={var_name}, is_used={is_used}, aggressive_mode={aggressive_mode}", file=sys.stderr, flush=True)
            # 変数を定義済みとしてマーク（未定義変数のループから除外するため）
            context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
            # スコープ情報を記録
            context.add_scope_variable(var_name)
            return

        if not is_used:
            print(f"[UNUSED_VAR_RESOLVE] 未使用変数を解決しようとしています: var_name={var_name}, is_used={is_used}, aggressive_mode={aggressive_mode}", file=sys.stderr, flush=True)

        # unavailable_variablesを除外してcurrent_arrayを選択
        try:
            unavailable_vars = context.variable_manager.get_unavailable_variables()
        except Exception:
            unavailable_vars = set()

        available_arrays = [arr for arr in current_arrays if arr not in unavailable_vars] if current_arrays else []
        current_array = "objects" if not available_arrays else random.choice(available_arrays)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: current_arrays={current_arrays}, unavailable_vars={unavailable_vars}, available_arrays={available_arrays}, 選択したcurrent_array={current_array}", flush=True)

        # 型に応じて処理を分岐
        if type_info.semantic_type == SemanticType.OBJECT and type_info.is_array:
            self._resolve_object_array_variable(
                var_name, context, current_array, nodes_in_step, aggressive_mode, iteration_count
            )
        elif type_info.semantic_type == SemanticType.OBJECT and not type_info.is_array:
            self._resolve_object_variable(
                var_name, context, current_array, nodes_in_step, aggressive_mode, iteration_count
            )
        else:
            self._resolve_other_variable(
                var_name, context, type_info, nodes_in_step, iteration_count, aggressive_mode
            )

    def _resolve_object_array_variable(
        self, var_name: str, context: ProgramContext, current_array: str,
        nodes_in_step: NodeListWithVarInfo, aggressive_mode: bool, iteration_count: int
    ) -> None:
        """オブジェクト配列変数を解決"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: オブジェクト配列変数 {var_name} の処理開始", flush=True)

        if aggressive_mode:
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3: 積極的モードで空配列ノード生成 - {var_name}", flush=True)
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, var_name)
            empty_array_node = self._generate_empty_array_node(var_name, context)
            self._set_node_metadata(empty_array_node, "iter", iteration_count + 1)
            add_node_to_list(nodes_in_step, empty_array_node, variable=var_name, context=context, is_first_override=is_first)
        else:
            # 重み付き乱数で操作を選択
            operations = ['array_operation', 'filter', 'sort']
            weights = [2.0, 3.0, 1.0]  # 各操作の重み（等確率）
            selected_operation = random.choices(operations, weights=weights, k=1)[0]

            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3: オブジェクト配列変数 {var_name} の処理: selected_operation={selected_operation}", flush=True)

            if selected_operation == 'array_operation':
                self._generate_array_operation_node(var_name, current_array, nodes_in_step, context, iteration_count)
            elif selected_operation == 'filter':
                self._generate_filter_node_for_resolve3(var_name, current_array, nodes_in_step, context, iteration_count)
            else:  # selected_operation == 'sort'
                self._generate_sort_node_for_resolve3(var_name, current_array, nodes_in_step, context, iteration_count)

        context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=True)
        # スコープ情報を記録
        context.add_scope_variable(var_name)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: 変数 {var_name} を定義済みとしてマーク", flush=True)

    def _generate_array_operation_node(
        self, var_name: str, current_array: str, nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext, iteration_count: int
    ) -> None:
        """配列操作ノード（CONCAT/APPEND）を生成"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: 配列操作ノードを生成: {var_name}", flush=True)

        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: CONCATノード生成 - {current_array} + objects -> {var_name}", flush=True)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, var_name)
        concat_node = self._generate_concat_node(current_array, "objects", var_name, context)
        self._set_node_metadata(concat_node, "iter", iteration_count + 1)
        add_node_to_list(nodes_in_step, concat_node, variable=var_name, context=context, is_first_override=is_first)

    def _generate_filter_node_for_resolve3(
        self, var_name: str, current_array: str, nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext, iteration_count: int
    ) -> None:
        """FILTERノードを生成（_resolve_dependencies3用）"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: FILTERノード生成 - {current_array} -> {var_name}", flush=True)

        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, var_name)
        filter_node = self._generate_advanced_filter(current_array, var_name, context)
        self._set_node_metadata(filter_node, "iter", iteration_count + 1)
        add_node_to_list(nodes_in_step, filter_node, variable=var_name, context=context, is_first_override=is_first)

    def _generate_sort_node_for_resolve3(
        self, var_name: str, current_array: str, nodes_in_step: NodeListWithVarInfo,
        context: ProgramContext, iteration_count: int
    ) -> None:
        """SORTノードを生成（_resolve_dependencies3用）"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: SORTノード生成 - {current_array} -> {var_name}", flush=True)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, var_name)
        sort_node = self._generate_sort_node(current_array, var_name, context)
        self._set_node_metadata(sort_node, "iter", iteration_count + 1)
        add_node_to_list(nodes_in_step, sort_node, variable=var_name, context=context, is_first_override=is_first)

    def _resolve_object_variable(
        self, var_name: str, context: ProgramContext, current_array: str, nodes_in_step: NodeListWithVarInfo,
        aggressive_mode: bool, iteration_count: int
    ) -> None:
        """オブジェクト変数（単体）を解決"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: オブジェクト変数（単体）の処理: {var_name}", flush=True)

        # 3つの選択肢から重み付け選択
        # ①CREATE（CREATE_LINE/CREATE_RECT）
        # ②MERGE（既存配列またはobjects）
        # ③SORT_BY + ObjectAccessNode（重め）
        methods = ['create', 'merge', 'sort_by']
        weights = [1.0, 0.2, 6.0]  # sort_byを重めに設定
        method_choice = random.choices(methods, weights=weights, k=1)[0]

        if method_choice == 'create':
            # ①CREATEノード生成
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3: CREATEノード生成 - {var_name}", flush=True)

            create_node = self._generate_create_objects(var_name, context)
            if create_node is None:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[WARNING] _resolve_dependencies3: CREATEノード生成失敗 - {var_name}", flush=True)
                return

            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, var_name)
            self._set_node_metadata(create_node, "iter", iteration_count + 1)
            add_node_to_list(nodes_in_step, create_node, variable=var_name, context=context, is_first_override=is_first)

        elif method_choice == 'merge':
            # ②MERGEノード生成
            # current_arrayをそのまま使用
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3: MERGEノード生成 - {current_array} -> {var_name}", flush=True)

            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, var_name)
            merge_node = self._generate_merge_node(current_array, var_name, context)
            self._set_node_metadata(merge_node, "iter", iteration_count + 1)
            add_node_to_list(nodes_in_step, merge_node, variable=var_name, context=context, is_first_override=is_first)

        else:  # method_choice == 'sort_by'
            # ③SORT_BY + ObjectAccessNode
            # current_arrayを優先的に使用、無効な場合は既存のOBJECT配列変数を取得

            # 新規配列名を生成
            new_array_name = variable_manager.get_next_variable_name(
                TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
                set(context.variable_manager.get_all_variable_names())
            )

            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                print(f"[DEBUG] _resolve_dependencies3: SORT_BY + ObjectAccessNode生成 - {current_array} -> {new_array_name} -> {var_name}", flush=True)

            # SORT_BYノードを生成（no_new_vars_mode=Trueで引数生成時に新しい変数を作らないようにする）
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, new_array_name)
            sort_node = self._generate_sort_node(current_array, new_array_name, context, no_new_vars_mode=True)
            self._set_node_metadata(sort_node, "iter", iteration_count + 1)
            add_node_to_list(nodes_in_step, sort_node, variable=new_array_name, context=context, is_first_override=is_first)

            # 新規配列を定義済みとしてマーク
            # context.variable_manager.define_variable(new_array_name, SemanticType.OBJECT, is_array=True)
            # # new_array_nameを使用禁止変数に登録（ObjectAccessNodeで使用するまで、他の変数解決で使用されないようにするため）
            # context.variable_manager.add_unavailable_variable(new_array_name)

            # ObjectAccessNodeでオブジェクト単体を取得（firstまたはlastを乱数で選択）
            # 使用禁止を解除してから使用
            #context.variable_manager.remove_unavailable_variable(new_array_name)
            access_type = random.choice(["first", "last"])
            # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
            is_first = is_first_definition(context.variable_manager.tracker, var_name)
            object_access_node = self._generate_object_access_node(var_name, new_array_name, access_type, context)
            self._set_node_metadata(object_access_node, "iter", iteration_count + 1)
            add_node_to_list(nodes_in_step, object_access_node, variable=var_name, context=context, is_first_override=is_first)

        # 変数を定義済みとしてマーク
        var_info = context.variable_manager.get_variable_info(var_name)
        if var_info and 'type_info' in var_info:
            type_info = var_info['type_info']
            context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
            # スコープ情報を記録
            context.add_scope_variable(var_name)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: 変数 {var_name} を定義済みとしてマーク", flush=True)

    def _resolve_other_variable(
        self, var_name: str, context: ProgramContext, type_info: TypeInfo,
        nodes_in_step: NodeListWithVarInfo, iteration_count: int, aggressive_mode: bool = False
    ) -> None:
        """その他の変数を解決"""
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: その他の変数の処理: {var_name}", flush=True)

        # 未使用変数のチェック（_resolve_single_variableで既にチェックされているが、念のため）
        var_info = context.variable_manager.get_variable_info(var_name)
        is_used = var_info.get('is_used', False) if var_info else False

        # 未使用変数で、積極的モードでない場合は定義をスキップ
        if not is_used and not aggressive_mode:
            print(f"[UNUSED_VAR_SKIP] _resolve_other_variable: 未使用変数の定義をスキップ: var_name={var_name}, is_used={is_used}, aggressive_mode={aggressive_mode}", file=sys.stderr, flush=True)
            # 変数を定義済みとしてマーク（未定義変数のループから除外するため）
            context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
            # スコープ情報を記録
            context.add_scope_variable(var_name)
            return

        # no_new_vars_mode=Trueで変数定義ノードを生成（右辺の引数生成時に新しい変数を作らないようにする）
        if ENABLE_DEBUG_LOGS:
            print(f"[RESOLVE_OTHER_VAR] _resolve_other_variable: 変数定義ノード生成開始: var_name={var_name}, type_info={type_info}", file=sys.stderr, flush=True)
        # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
        is_first = is_first_definition(context.variable_manager.tracker, var_name)
        var_def_node = self.node_generators._generate_variable_definition_node(var_name, context, provided_type_info=type_info, no_new_vars_mode=True)
        self._set_node_metadata(var_def_node, "iter", iteration_count + 1)
        add_node_to_list(nodes_in_step, var_def_node, variable=var_name, context=context, is_first_override=is_first)

        context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
        # スコープ情報を記録
        context.add_scope_variable(var_name)
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] _resolve_dependencies3: 変数 {var_name} を定義済みとしてマーク", flush=True)
        if ENABLE_DEBUG_LOGS:
            print(f"[RESOLVE_OTHER_VAR] _resolve_other_variable: 変数定義ノード生成完了: var_name={var_name}", file=sys.stderr, flush=True)

    def _finalize_iteration_nodes(
        self, nodes_in_step: NodeListWithVarInfo, nodes_in_step2: NodeListWithVarInfo,
        context: ProgramContext
    ) -> None:
        """反復後のノード処理を実行"""
        if not nodes_in_step:
            return

        first_defined_vars = get_first_defined_variables(nodes_in_step)
        for var_name in first_defined_vars:
            context.variable_manager.add_unavailable_variable(var_name)

        if nodes_in_step2:
            extend_node_list_with_var_info(nodes_in_step, nodes_in_step2)
            nodes_in_step2.clear()
        extend_node_list_with_var_info(nodes_in_step2, nodes_in_step)
        nodes_in_step.clear()

    def _handle_remaining_undefined_variables(
        self, nodes_in_step: NodeListWithVarInfo, nodes_in_step2: NodeListWithVarInfo,
        context: ProgramContext
    ) -> None:
        """残りの未定義変数を処理（新しい変数定義禁止モード）"""
        remaining_undefined = self.variable_managers._get_undefined_variables(context)
        if not remaining_undefined:
            return

        for var_name in remaining_undefined:
            context.variable_manager.add_unavailable_variable(var_name)

        for var_name in remaining_undefined:
            var_info = context.variable_manager.get_variable_info(var_name)
            if not var_info or 'type_info' not in var_info:
                continue

            type_info = var_info['type_info']
            try:
                var_def_node = self.node_generators._generate_variable_definition_node(
                    var_name, context, no_new_vars_mode=True, provided_type_info=type_info
                )
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, var_name)
                self._set_node_metadata(var_def_node, "remaining")
                add_node_to_list(nodes_in_step, var_def_node, variable=var_name, context=context, is_first_override=is_first)
                context.variable_manager.define_variable(var_name, type_info.semantic_type, type_info.is_array)
                # スコープ情報を記録
                context.add_scope_variable(var_name)
            except Exception:
                raise RuntimeError(f"新しい変数定義禁止モードが予期せず失敗しました: {var_name}")

        self._finalize_iteration_nodes(nodes_in_step, nodes_in_step2, context)

    def _handle_final_undefined_variables(
        self, nodes_in_step: NodeListWithVarInfo, nodes_in_step2: NodeListWithVarInfo,
        context: ProgramContext
    ) -> None:
        """最終チェック: まだ未定義変数が残っている場合の緊急対応"""
        final_undefined = self.variable_managers._get_undefined_variables(context)
        if not final_undefined:
            return

        for var_name in final_undefined:
            context.variable_manager.add_unavailable_variable(var_name)

        for var_name in final_undefined:
            if var_name.startswith('objects'):
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, var_name)
                empty_array_node = self._generate_empty_array_node(var_name, context)
                self._set_node_metadata(empty_array_node, "final")
                add_node_to_list(nodes_in_step, empty_array_node, variable=var_name, context=context, is_first_override=is_first)
                context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=True)
                # スコープ情報を記録
                context.add_scope_variable(var_name)
            elif var_name.startswith('object'):
                create_node = self._generate_create_objects(var_name, context)
                if create_node is None:
                    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                        print(f"[WARNING] _resolve_dependencies3: CREATEノード生成失敗（制限に達した可能性） - {var_name}", flush=True)
                    continue
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, var_name)
                self._set_node_metadata(create_node, "final")
                add_node_to_list(nodes_in_step, create_node, variable=var_name, context=context, is_first_override=is_first)
                context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=False)
                # スコープ情報を記録
                context.add_scope_variable(var_name)
            else:
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, var_name)
                var_def_node = self._generate_variable_definition_node(var_name, context)
                self._set_node_metadata(var_def_node, "final")
                add_node_to_list(nodes_in_step, var_def_node, variable=var_name, context=context, is_first_override=is_first)
                context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=False)
                # スコープ情報を記録
                context.add_scope_variable(var_name)

        self._finalize_iteration_nodes(nodes_in_step, nodes_in_step2, context)

    def _generate_create_objects(self, var_name: str, context: ProgramContext, no_new_vars_mode: bool = False) -> Node:
        """オブジェクト生成ノードを生成"""
        return self.node_generators._generate_create_objects(var_name, context, no_new_vars_mode=no_new_vars_mode)

    def _generate_sort_node(self, current_array: str, var_name: str, context: ProgramContext, no_new_vars_mode: bool = False) -> Node:
        """ソートノードを生成"""
        return self.node_generators._generate_sort_node(current_array, var_name, context, no_new_vars_mode=no_new_vars_mode)

    def _generate_if_branch(self, context: ProgramContext) -> Node:
        """IF分岐ノードを生成"""
        return self.node_generators._generate_if_branch(context)

    def _generate_advanced_filter(self, source_array: str, target_array: str, context: ProgramContext) -> FilterNode:
        """FILTERノードを生成"""
        return self.node_generators._generate_advanced_filter(source_array, target_array, context)

    def _generate_exclude_node(self, source_array: str, target_array: str, targets_array: str, context: ProgramContext) -> ExcludeNode:
        """EXCLUDEノードを生成"""
        return self.node_generators._generate_exclude_node(source_array, target_array, targets_array, context)

    def _generate_concat_node(self, array1: str, array2: str, target_array: str, context: ProgramContext) -> ConcatNode:
        """CONCATノードを生成"""
        return self.node_generators._generate_concat_node(array1, array2, target_array, context)

    def _generate_append_node(self, array: str, obj: str, target_array: str, context: ProgramContext) -> AppendNode:
        """APPENDノードを生成"""
        return self.node_generators._generate_append_node(array, obj, target_array, context)

    def _generate_merge_node(self, objects_array: str, target_obj: str, context: ProgramContext) -> MergeNode:
        """MERGEノードを生成"""
        return self.node_generators._generate_merge_node(objects_array, target_obj, context)

    def _generate_empty_array_node(self, array_name: str, context: ProgramContext) -> EmptyArrayNode:
        """空のオブジェクト配列定義ノードを生成"""
        return self.node_generators._generate_empty_array_node(array_name, context)

    def _generate_object_access_node(self, obj_var: str, objects_array: str, access_type: str, context: ProgramContext) -> ObjectAccessNode:
        """オブジェクトアクセスノードを生成"""
        return self.node_generators._generate_object_access_node(obj_var, objects_array, access_type, context)

    def _generate_single_object_array_node(self, array_name: str, object_name: str, context: ProgramContext) -> SingleObjectArrayNode:
        """単一オブジェクト配列定義ノードを生成"""
        return self.node_generators._generate_single_object_array_node(array_name, object_name, context)

    def _generate_split_connected_node(self, source_object: str, target_array: str, context: ProgramContext, connectivity: int = 4) -> SplitConnectedNode:
        """SPLIT_CONNECTEDノードを生成"""
        return self.node_generators._generate_split_connected_node(source_object, target_array, context, connectivity)

    def _generate_extract_shape_node(self, source_object: str, target_array: str, extract_type: str, context: ProgramContext) -> ExtractShapeNode:
        """形状抽出ノードを生成"""
        return self.node_generators._generate_extract_shape_node(source_object, target_array, extract_type, context)

    def _generate_extend_pattern_node(self, source_array: str, target_array: str, context: ProgramContext, side: str = "end", count: int = 1) -> ExtendPatternNode:
        """EXTEND_PATTERNノードを生成"""
        return self.node_generators._generate_extend_pattern_node(source_array, target_array, context, side, count)

    def _generate_arrange_grid_node(self, source_array: str, target_array: str, context: ProgramContext, cols: int = 3, width: int = 10, height: int = 10) -> ArrangeGridNode:
        """ARRANGE_GRIDノードを生成"""
        return self.node_generators._generate_arrange_grid_node(source_array, target_array, context, cols, width, height)

    def _generate_match_pairs_node(self, array1: str, array2: str, target_array: str, context: ProgramContext) -> MatchPairsNode:
        """MATCH_PAIRSノードを生成"""
        return self.node_generators._generate_match_pairs_node(array1, array2, target_array, context)

    def _generate_for_loop(self, current_array: str, context: ProgramContext) -> Node:
        """FORループ開始ノードを生成"""
        return self.node_generators._generate_for_loop(current_array, context)

    def _generate_for_loop_with_count(self, count_variable: str, context: ProgramContext) -> Node:
        """COUNT型変数を使用したFORループ開始ノードを生成"""
        return self.node_generators._generate_for_loop_with_count(count_variable, context)

    def _generate_for_loop_with_constant(self, constant_value: int, context: ProgramContext) -> Node:
        """定数値を使用したFORループ開始ノードを生成"""
        return self.node_generators._generate_for_loop_with_constant(constant_value, context)

    def _generate_for_loop_with_match_pairs(self, match_pairs_array: str, context: ProgramContext) -> Node:
        """MATCH_PAIRS配列用のFORループ開始ノードを生成"""
        return self.node_generators._generate_for_loop_with_match_pairs(match_pairs_array, context)

    def _get_count_variables(self, context: ProgramContext) -> List[str]:
        """利用可能なCOUNT型変数を取得"""
        count_variables = []

        # COUNT型の非配列変数を取得
        compatible_vars = context.variable_manager.get_compatible_variables_for_assignment(
            SemanticType.COUNT, is_array=False, context=context
        )

        for var_name in compatible_vars:
            # 定義済みかつ使用不可変数でないものを選択
            if (context.variable_manager.is_defined(var_name) and
                not context.variable_manager.tracker.is_variable_unavailable(var_name)):
                count_variables.append(var_name)

        return count_variables

    def _create_new_count_variable(self, context: ProgramContext) -> str:
        """新しいCOUNT型変数を作成"""
        # 新しい変数名を生成
        from ..metadata.variable_naming import VariableNamingSystem
        type_info = TypeInfo.create_from_semantic_type(SemanticType.COUNT, is_array=False)
        naming_system = VariableNamingSystem()
        used_vars = context.variable_manager.get_used_variable_names()
        var_name = naming_system.get_next_variable_name(type_info, used_vars)

        return var_name

    def _generate_end(self, context: ProgramContext) -> Node:
        """ENDノードを生成"""
        return self.node_generators._generate_end(context)

    def _generate_array_element_assignment(self, for_array: str, loop_idx: int, context: ProgramContext) -> Node:
        """配列要素代入ノードを生成"""
        ctx_dict = context.to_dict()
        return self.node_generators._generate_array_element_assignment(for_array, loop_idx, context, ctx_dict)

    def _generate_new_variable_name(self, context: ProgramContext, is_array: bool) -> str:
        """新しい変数名を生成（共通メソッド）

        Args:
            context: プログラムコンテキスト
            is_array: 配列変数の場合はTrue、単体変数の場合はFalse

        Returns:
            生成された変数名
        """
        # 既存変数名を追跡
        existing_vars = set(['i', 'j', 'k', 'l', 'm', 'n'])  # ループ変数

        # 既存の変数名を追加
        for var_name, var_info in context.variable_manager.variables.items():
            if (var_info.get('type_info') and
                var_info['type_info'].semantic_type == SemanticType.OBJECT and
                var_info['type_info'].is_array == is_array):
                existing_vars.add(var_name)

        # 新しい変数名を生成
        type_info = TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=is_array)
        new_var = variable_manager.get_next_variable_name(type_info, existing_vars)

        # 注意: define_variableは呼ばない（ノード生成時に定義される）
        # これにより、is_first_definitionが正しく判定できるようになる

        # 現在のスコープに追加（ネスト内で定義された場合）
        if context.nest_stack:
            context.add_scope_variable(new_var)

        return new_var

    def _generate_new_array_variable_name(self, context: ProgramContext) -> str:
        """新しいオブジェクト配列変数名を生成"""
        return self._generate_new_variable_name(context, is_array=True)

    def _safe_remove_variable(self, target_var: str, target_list: List[str], current_arrays: List[str], current_objects: List[str], context: ProgramContext, is_array: bool) -> bool:
        """変数リストから変数を安全に削除（共通メソッド）

        Args:
            target_var: 削除対象の変数名
            target_list: 削除対象の変数リスト（current_arrays または current_objects）
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト
            is_array: 配列変数の場合はTrue、オブジェクト変数の場合はFalse

        Returns:
            bool: 削除した場合はTrue、削除しなかった場合はFalse
        """
        # 現在のネスト内で定義された変数を取得
        current_nest_object_arrays = context.get_current_scope_object_arrays()
        current_nest_objects = context.get_current_scope_objects()

        # ネスト外の変数をフィルタリング
        filtered_arrays = [x for x in current_arrays if x not in current_nest_object_arrays]
        filtered_objects = [x for x in current_objects if x not in current_nest_objects]

        # 削除後のネスト外変数数を計算
        if is_array:
            remaining_items = [x for x in filtered_arrays if x != target_var]
            remaining_count = len(remaining_items) + len(filtered_objects)
            nest_list = current_nest_object_arrays
        else:
            remaining_items = [x for x in filtered_objects if x != target_var]
            remaining_count = len(filtered_arrays) + len(remaining_items)
            nest_list = current_nest_objects

        # ネスト外変数が1未満になる場合は削除しない（ただし、ネスト深度0の場合は許可）
        if remaining_count < 1 and context.get_scope_nesting_depth() > 0:
            # 削除対象がネスト外変数の場合は削除しない
            if target_var not in nest_list:
                return False

        # 安全に削除
        if target_var in target_list:
            target_list.remove(target_var)
            return True

        return False

    def _safe_remove_from_arrays(self, current_array: str, current_arrays: List[str], current_objects: List[str], context: ProgramContext) -> bool:
        """current_arraysからcurrent_arrayを安全に削除

        Args:
            current_array: 削除対象の配列変数名
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト

        Returns:
            bool: 削除した場合はTrue、削除しなかった場合はFalse
        """
        return self._safe_remove_variable(current_array, current_arrays, current_arrays, current_objects, context, is_array=True)

    def _process_common_arrays_for_concat(self, common_arrays: List[str], nodes_in_step: NodeListWithVarInfo,
                                         current_arrays: List[str], current_objects: List[str],
                                         context: ProgramContext, skip_probability: float = 0.1,
                                         exclude_probability: float = 0.04, empty_array: Optional[str] = None) -> None:
        """共通配列に対してCONCAT/EXCLUDEノードを生成する共通処理

        Args:
            common_arrays: 処理対象の共通配列リスト
            nodes_in_step: ノードリスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト
            skip_probability: 使用済み配列をスキップする確率（デフォルト: 0.1）
            exclude_probability: EXCLUDEノードを生成する確率（デフォルト: 0.04）
            empty_array: EMPTY_ARRAYノードの配列（Noneでない場合は優先的に使用）
        """
        for common_array in common_arrays:
            # common_arrayが使用済みかどうかを確認
            used_variables = context.variable_manager.get_used_variable_names()
            if common_array in used_variables:
                # 使用済みの場合は一定確率でCONCATノードを生成せずに削除のみ実行
                if random.random() < skip_probability:
                    # スキップ（削除のみ）
                    self._safe_remove_from_arrays(common_array, current_arrays, current_objects, context)
                    continue

            # 代入先配列の選択: empty_arrayが優先、なければcurrent_arraysから選択
            candidate_arrays = []
            if empty_array and empty_array != common_array:
                candidate_arrays = [empty_array]
            if not candidate_arrays:
                candidate_arrays = [x for x in current_arrays if x != common_array]

            if not candidate_arrays:
                break
            other_array = random.choice(candidate_arrays)
            # exclude_probabilityの確率でEXCLUDEノードを生成、それ以外でCONCATノードを生成
            if random.random() < exclude_probability:
                # EXCLUDEノード: other_arrayからcommon_arrayを除外
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, other_array)
                exclude_node = self._generate_exclude_node(other_array, other_array, common_array, context)
                add_node_to_list(nodes_in_step, exclude_node, variable=other_array, context=context, is_first_override=is_first)
            else:
                # CONCATノード: other_arrayとcommon_arrayを結合
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, other_array)
                concat_node = self._generate_concat_node(other_array, common_array, other_array, context)
                add_node_to_list(nodes_in_step, concat_node, variable=other_array, context=context, is_first_override=is_first)
            self._safe_remove_from_arrays(common_array, current_arrays, current_objects, context)

    def _process_common_objects_for_append(self, common_objects: List[str], nodes_in_step: NodeListWithVarInfo,
                                          current_arrays: List[str], current_objects: List[str],
                                          context: ProgramContext, empty_array: Optional[str] = None) -> None:
        """共通オブジェクトに対してAPPENDノードを生成する共通処理

        Args:
            common_objects: 処理対象の共通オブジェクトリスト
            nodes_in_step: ノードリスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト
            empty_array: EMPTY_ARRAYノードの配列（Noneでない場合は優先的に使用）
        """
        for common_object in common_objects:
            # common_objectが使用済みかどうかを確認
            used_variables = context.variable_manager.get_used_variable_names()
            if common_object in used_variables:
                # 使用済みの場合はAPPENDノードを生成せずに削除のみ実行
                self._safe_remove_from_objects(common_object, current_arrays, current_objects, context)
            else:
                # 未使用の場合は通常通りAPPENDノードを生成
                # 代入先配列の選択: empty_arrayが優先、なければcurrent_arraysから選択
                if empty_array:
                    current_array = empty_array
                elif current_arrays:
                    current_array = random.choice(current_arrays)
                else:
                    continue  # 配列が存在しない場合はスキップ

                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first = is_first_definition(context.variable_manager.tracker, current_array)
                append_node = self._generate_append_node(current_array, common_object, current_array, context)
                add_node_to_list(nodes_in_step, append_node, variable=current_array, context=context, is_first_override=is_first)
                self._safe_remove_from_objects(common_object, current_arrays, current_objects, context)

    def _process_merge_operation(self, common_arrays: List[str], nodes_in_step: NodeListWithVarInfo,
                                 current_arrays: List[str], current_objects: List[str],
                                 context: ProgramContext) -> None:
        """MERGE処理（APPEND + MERGEノード生成）の共通処理

        Args:
            common_arrays: 処理対象の共通配列リスト
            nodes_in_step: ノードリスト
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト
        """
        # common_arraysを再定義（current_arraysが変更されている可能性があるため）
        current_nest_object_arrays = context.get_current_scope_object_arrays()
        common_arrays = [x for x in current_arrays if x in current_nest_object_arrays]
        # common_arraysが存在する場合のみMERGE処理を実行
        if common_arrays:
            common_array = common_arrays[0]
            if current_objects:
                current_object = random.choice(current_objects)
                # ノード生成前にis_first_definitionを判定（define_variableが呼ばれる前）
                is_first_append = is_first_definition(context.variable_manager.tracker, common_array)
                is_first_merge = is_first_definition(context.variable_manager.tracker, current_object)
                append_node = self._generate_append_node(common_array, current_object, common_array, context)
                merge_node = self._generate_merge_node(common_array, current_object, context)
                add_node_to_list(nodes_in_step, append_node, variable=common_array, context=context, is_first_override=is_first_append)
                add_node_to_list(nodes_in_step, merge_node, variable=current_object, context=context, is_first_override=is_first_merge)
                self._safe_remove_from_arrays(common_array, current_arrays, current_objects, context)

    def _get_nest_scope_variables(self, current_arrays: List[str], current_objects: List[str],
                                  context: ProgramContext) -> Tuple[List[str], List[str], List[str]]:
        """ネストスコープ変数を取得する共通処理

        Args:
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト

        Returns:
            Tuple[List[str], List[str], List[str]]: (common_arrays, common_objects, other_arrays)
        """
        # 現在のネストで定義された変数を取得
        current_nest_object_arrays = context.get_current_scope_object_arrays()
        current_nest_objects = context.get_current_scope_objects()

        common_arrays = [x for x in current_arrays if x in current_nest_object_arrays]
        common_objects = [x for x in current_objects if x in current_nest_objects]
        other_arrays = [x for x in current_arrays if x not in current_nest_object_arrays]

        return common_arrays, common_objects, other_arrays

    # ============================================================================
    # ノード選択重み計算メソッド（selection_weights.py に移動）
    # ============================================================================

    def _add_to_excluded_types(self, selected: str, excluded_types: List[str], reason: str = "") -> None:
        """除外リストに追加する共通処理

        Args:
            selected: 除外するノードタイプ
            excluded_types: 除外リスト
            reason: 除外理由（デバッグ用）
        """
        if selected not in excluded_types:
            excluded_types.append(selected)
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                reason_msg = f"（{reason}）" if reason else ""
                print(f"[DEBUG] '{selected}' を除外リストに追加しました{reason_msg}（累計除外: {len(excluded_types)}タイプ）", flush=True)

    def _resolve_and_clear_step_nodes(self, nodes: NodeListWithVarInfo, nodes_in_step: NodeListWithVarInfo,
                                      context: ProgramContext, current_arrays: List[str]) -> NodeListWithVarInfo:
        """依存関係解決と検証処理を行い、nodes_in_stepをクリアする共通処理

        Args:
            nodes: 現在のノードリスト
            nodes_in_step: ステップ内のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列リスト

        Returns:
            解決後のノードリスト
        """
        nodes = self._resolve_dependencies_with_validation(nodes, nodes_in_step, context, current_arrays)
        nodes_in_step.clear()
        return nodes

    def _generate_for_loop_and_setup(self, for_node: Node, for_array: Optional[str], nodes_in_step: NodeListWithVarInfo,
                                     nodes: NodeListWithVarInfo, context: ProgramContext, current_arrays: List[str]) -> NodeListWithVarInfo:
        """FORループノードを生成し、依存関係解決とセットアップを行う共通処理

        Args:
            for_node: 生成されたFORループノード
            for_array: FORループで使用する配列（Noneの場合は定数/COUNT変数）
            nodes_in_step: ステップ内のノードリスト
            nodes: 現在のノードリスト
            context: プログラムコンテキスト
            current_arrays: 現在の配列リスト

        Returns:
            解決後のノードリスト
        """
        add_node_to_list(nodes_in_step, for_node, context=context)
        nodes = self._resolve_and_clear_step_nodes(nodes, nodes_in_step, context, current_arrays)
        if for_array is not None:
            context.for_arrays.append(for_array)
        else:
            context.for_arrays.append(None)  # 配列要素アクセス無効化
        return nodes

    def _safe_remove_from_objects(self, current_object: str, current_arrays: List[str], current_objects: List[str], context: ProgramContext) -> bool:
        """current_objectsからcurrent_objectを安全に削除

        Args:
            current_object: 削除対象のオブジェクト変数名
            current_arrays: 現在の配列変数リスト
            current_objects: 現在のオブジェクト変数リスト
            context: プログラムコンテキスト

        Returns:
            bool: 削除した場合はTrue、削除しなかった場合はFalse
        """
        return self._safe_remove_variable(current_object, current_objects, current_arrays, current_objects, context, is_array=False)

    def _generate_new_object_variable_name(self, context: ProgramContext) -> str:
        """新しいオブジェクト単体変数名を生成"""
        return self._generate_new_variable_name(context, is_array=False)

    def _generate_variable_definition_node(self, var_name: str, context: ProgramContext, provided_type_info: Optional[TypeInfo] = None) -> AssignmentNode:
        """変数定義ノードを生成（node_generators.pyに委譲）"""
        return self.node_generators._generate_variable_definition_node(var_name, context, provided_type_info=provided_type_info)

    def _generate_object_operations_node(self, var_name: str, context: ProgramContext) -> Node:
        """オブジェクト操作ノードを生成"""
        return self.node_generators._generate_object_operations_node(var_name, context)

    # _classify_node_type メソッドは selection_weights.py に移動

    def _update_previous_nodes(self, context: ProgramContext, new_node) -> None:
        """前のノード履歴を更新"""
        if not hasattr(context, 'previous_nodes'):
            context.previous_nodes = []

        # 最新の5ノードのみ保持
        context.previous_nodes.append(new_node)
        if len(context.previous_nodes) > 5:
            context.previous_nodes.pop(0)



    def _ensure_initialization_node(self, nodes: List[Node], context: ProgramContext) -> List[Node]:
        """初期化ノードの存在を保証"""
        # 初期化ノードが存在するかチェック
        has_init = any(isinstance(node, InitializationNode) for node in nodes)

        if not has_init:
            # 初期化ノードを生成
            # 連結性はランダムに4または8を選択（ARC問題では両方が使用される）
            connectivity = random.choice([4, 8])
            init_node = InitializationNode(
                connectivity=connectivity,
                context=context.to_dict()
            )
            nodes.insert(0, init_node)

            # 変数を定義済みとしてマーク
            context.variable_manager.define_variable("objects", SemanticType.OBJECT, is_array=True)
            # スコープ情報を記録
            context.add_scope_variable("objects")

            # last_selectedを設定（初期ノードが生成されたことを記録）
            context.last_selected = "get_objects"

        return nodes


    def _resolve_dependencies_with_constraints(self, goal_node, context):
        """制約チェック付きの依存関係解決"""
        # 通常の依存関係解決を実行
        program_nodes = self._resolve_dependencies(goal_node, context)
        return program_nodes
