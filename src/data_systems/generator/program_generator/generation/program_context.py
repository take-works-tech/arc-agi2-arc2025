"""
プログラム生成のコンテキスト管理
"""
import random
from typing import Dict, List, Optional
from ..metadata.constants import (
    generate_output_grid_size,
    ENABLE_NESTING_DEPTH_LIMIT,
    COMMAND_NESTING_DEPTH_BY_COMPLEXITY,
    COMPLEXITY_LEVELS,
    COMMAND_CATEGORIES,
    NODES_BY_COMPLEXITY,
)
from ..metadata.variable_manager import variable_manager


class PlaceholderTracking:
    """プレースホルダー変数の使用追跡（FILTER/SORT_BY/MATCH_PAIRS用）"""
    def __init__(self, context_type: str):  # 'FILTER', 'SORT_BY', 'MATCH_PAIRS'
        self.context_type = context_type
        self.appearance_count = 0  # 登場回数（変数を選んだ回数）
        self.usage_count = 0  # 使用回数（プレースホルダーを使用した回数）

        # MATCH_PAIRS用
        self.obj2_used = False  # 2つ目で$obj2を使用したかどうか


class ProgramContext:
    """プログラム生成のコンテキスト"""

    def __init__(self, complexity: int, grid_width: Optional[int] = None, grid_height: Optional[int] = None):
        self.complexity = complexity
        self.current_line = 0
        self.remaining_lines = 100


        # グリッドサイズ（指定されていない場合は自動決定）
        if grid_width is not None and grid_height is not None:
            self.output_grid_width = grid_width
            self.output_grid_height = grid_height
        else:
            # 入力グリッドサイズを取得（context_dictは後で定義されるため、デフォルト値を使用）
            input_width = 12  # デフォルト値
            input_height = 12  # デフォルト値
            # 後でto_dict()が呼ばれた時に更新される可能性があるが、ここではデフォルト値で生成
            width, height = generate_output_grid_size(input_width, input_height)
            self.output_grid_width = width
            self.output_grid_height = height

        # フラグ
        self.background_filter_applied = False
        self.use_filter_restore = False
        self.excluded_array = None

        # 部分プログラムのカテゴリ変数（依存関係解決時にcurrent_arraysに追加される）
        # カテゴリのオブジェクトを持っている変数のリスト
        self.category_arrays: List[str] = []

        # 複雑度制限（差別化強化版）
        # すべての複雑度関連設定をCOMPLEXITY_LEVELSから取得（統合済み）
        self.complexity_config = COMPLEXITY_LEVELS[complexity]

        # 構造的ネスト深度制限（FOR/IFのスコープネスト用）
        # COMPLEXITY_LEVELSから取得
        # 使用箇所: get_max_scope_nesting_depth(), _check_scope_nesting_depth()
        self.max_nesting_depth = self.complexity_config['max_nesting_depth']
        self.nesting_depth_enabled = ENABLE_NESTING_DEPTH_LIMIT
        # max_nodesとmin_nodesはNODES_BY_COMPLEXITYから取得（形式: (最小値, 最大値)）
        nodes_range = NODES_BY_COMPLEXITY.get(complexity, (2, 30))
        self.min_nodes = nodes_range[0]
        self.max_nodes = nodes_range[1]
        self.max_for_loops = self.complexity_config['max_for_loops']
        self.max_if_statements = self.complexity_config['max_if_statements']
        # 注意: 構造的ネスト深度の制限は max_nesting_depth を使用
        self.max_array_operations = self.complexity_config['max_array_operations']
        self.max_for_array_assignments = self.complexity_config.get('max_for_array_assignments', 10)  # デフォルト10

        # 現在の使用状況
        self.current_for_loops = 0
        self.current_if_statements = 0
        self.current_array_operations = 0

        # コマンド引数内のネスト深度制限
        # COMMAND_NESTING_DEPTH_BY_COMPLEXITYから取得
        # 使用箇所: get_command_nesting_depth(), _generate_argument_node()
        # 例: FILTER(objects, EQUAL(GET_COLOR($obj), 3)) のような引数内のネスト
        self.max_command_nesting_depth = COMMAND_NESTING_DEPTH_BY_COMPLEXITY.get(complexity, 3)

        # ネスト情報を2次元配列で管理
        # 各要素: {'type': 'for'/'if', 'node_count': int, 'array_assignment_count': int, 'scope_variables': List[str]}
        self.nest_stack = []  # ネストスタック（現在のネスト情報）
        self.nest_history = []  # ネスト履歴（完了したネストの情報）

        # スコープ管理
        self.scope_variables = []  # 現在のスコープで定義された変数

        # IF文の条件分岐を考慮した変数管理
        self.scope_counter = 0  # FOR/IFの両方のスコープIDを生成するカウンター
        self.variable_scope_map: Dict[str, Dict] = {}  # 変数名からスコープ情報へのマッピング
        # 各値: {'type': 'for'/'if', 'scope_id': int}
        # または: {'type': None, 'scope_id': None}  # スコープの外で定義された変数

        # 追跡（統合されたvariable_managerを使用）
        self.variable_manager = variable_manager

        # FORループ配列の追跡
        self.for_arrays = []  # 現在アクティブなFORループの配列リスト
        self.match_pairs_arrays = []  # MATCH_PAIRSの代入先配列を追跡

        # 比例演算で使用されたコマンドを記録
        self.proportional_operation_commands = []

        # 比例演算の第一引数コマンドを記録（第二引数で同じコマンドを選ぶため）
        self.current_proportional_first_arg_command = None
        self.current_proportional_arg_index = None

        # プログラム複雑さ制御（改善案2）
        self.complexity_level: Optional[str] = None  # 'simple', 'medium', 'complex'
        self.complexity_constraints: Optional[Dict] = None
        self.current_lines: int = 0
        self.current_for_loops: int = 0
        self.current_if_statements: int = 0

        # プレースホルダーコンテキストの初期化
        self.current_placeholder_context = None
        self.usage_stats: Dict[str, int] = {}

        # 現在のコマンド（ネストしたコマンドの追跡用）
        self.current_command: Optional[str] = None

        # プレースホルダー変数の使用追跡（FILTER/SORT_BY/MATCH_PAIRS用）
        self.placeholder_tracking = None  # PlaceholderTrackingオブジェクトまたはNone

        # 引数ネスト深度ごとの親コマンドを追跡（緩和数の計算用）
        # キー: nesting_depth, 値: コマンド名のリスト
        self.argument_nesting_command_stack: Dict[int, List[str]] = {}

    # ============================================================
    # 構造生成制限チェック（優先順位順）
    # ============================================================
    # チェック順序:
    # 1. 構造的ネスト深度チェック（最優先）- _check_scope_nesting_depth()で実施
    # 2. can_add_for_loop() / can_add_if_statement() - 個数制限チェック
    # ============================================================

    def can_add_for_loop(self) -> bool:
        """FORループの個数制限をチェック

        プログラム全体でのFORループの総数が最大数を超えていないかチェックします。
        このチェックは構造的ネスト深度チェックの後に実行されます。

        Returns:
            bool: FORループを追加できる場合True
        """
        can_add = self.current_for_loops < self.max_for_loops
        return can_add

    def can_add_if_statement(self) -> bool:
        """IF文の個数制限をチェック

        プログラム全体でのIF文の総数が最大数を超えていないかチェックします。
        このチェックは構造的ネスト深度チェックの後に実行されます。

        Returns:
            bool: IF文を追加できる場合True
        """
        can_add = self.current_if_statements < self.max_if_statements
        return can_add

    def enter_for_nesting(self):
        """FORネストを開始"""
        if self.nesting_depth_enabled:
            # スコープIDを生成
            for_scope_id = self.scope_counter
            self.scope_counter += 1

            self.nest_stack.append({
                'type': 'for',
                'node_count': 0,
                'array_assignment_count': 0,
                'scope_variables': [],
                'scope_id': for_scope_id  # スコープIDを追加
            })
            # 新しいスコープの変数リストを初期化
            self.scope_variables = []
        self.current_for_loops += 1

    def exit_for_nesting(self):
        """FORネストを終了"""
        if self.nesting_depth_enabled and self.nest_stack:
            # 現在のネストを履歴に移動
            current_nest = self.nest_stack.pop()
            self.nest_history.append(current_nest)

            # スコープ内で定義された変数のみを削除（型情報を保持するため、使用のみの変数は削除しない）
            if 'scope_variables' in current_nest and current_nest['scope_variables']:
                # 定義された変数のみをフィルタリング
                defined_vars = []
                for var_name in current_nest['scope_variables']:
                    var_info = self.variable_manager.get_variable_info(var_name)
                    if var_info and var_info.get('is_defined', False):
                        defined_vars.append(var_name)

                if defined_vars:
                    self.variable_manager.remove_variables(defined_vars)
                    # variable_scope_mapからも削除
                    for var_name in defined_vars:
                        if var_name in self.variable_scope_map:
                            del self.variable_scope_map[var_name]

            # 外側のネストの変数を復元（まだネストがある場合）
            if self.nest_stack:
                outer_nest = self.nest_stack[-1]
                self.scope_variables = outer_nest.get('scope_variables', []).copy()
            else:
                # ネストがなくなった場合はクリア
                self.scope_variables = []

    def enter_if_nesting(self):
        """IFネストを開始"""
        if self.nesting_depth_enabled:
            # スコープIDを生成
            if_scope_id = self.scope_counter
            self.scope_counter += 1

            self.nest_stack.append({
                'type': 'if',
                'node_count': 0,
                'scope_variables': [],
                'scope_id': if_scope_id  # スコープIDを追加
            })
            # 新しいスコープの変数リストを初期化
            self.scope_variables = []
        self.current_if_statements += 1

    def exit_if_nesting(self):
        """IFネストを終了"""
        if self.nesting_depth_enabled and self.nest_stack:
            # 現在のネストを履歴に移動
            current_nest = self.nest_stack.pop()
            self.nest_history.append(current_nest)

            # スコープ内で定義された変数のみを削除（型情報を保持するため、使用のみの変数は削除しない）
            if 'scope_variables' in current_nest and current_nest['scope_variables']:
                # 定義された変数のみをフィルタリング
                defined_vars = []
                for var_name in current_nest['scope_variables']:
                    var_info = self.variable_manager.get_variable_info(var_name)
                    if var_info and var_info.get('is_defined', False):
                        defined_vars.append(var_name)

                if defined_vars:
                    self.variable_manager.remove_variables(defined_vars)
                    # variable_scope_mapからも削除
                    for var_name in defined_vars:
                        if var_name in self.variable_scope_map:
                            del self.variable_scope_map[var_name]

            # 外側のネストの変数を復元（まだネストがある場合）
            if self.nest_stack:
                outer_nest = self.nest_stack[-1]
                self.scope_variables = outer_nest.get('scope_variables', []).copy()
            else:
                # ネストがなくなった場合はクリア
                self.scope_variables = []

    def increment_nested_node_count(self):
        """現在のネスト内のノード数を1増加"""
        if self.nest_stack:
            self.nest_stack[-1]['node_count'] += 1

    def increment_for_array_assignment_count(self):
        """現在のFORネスト内の配列代入ノード数を1増加"""
        if self.nest_stack and self.nest_stack[-1]['type'] == 'for':
            self.nest_stack[-1]['array_assignment_count'] += 1

    def get_current_for_array_assignment_count(self) -> int:
        """現在のFORネスト内の配列代入ノード数を取得"""
        if self.nest_stack and self.nest_stack[-1]['type'] == 'for':
            return self.nest_stack[-1].get('array_assignment_count', 0)
        return 0

    def add_scope_variable(self, variable_name: str):
        """現在のスコープに変数を追加"""
        if self.nest_stack:
            if 'scope_variables' not in self.nest_stack[-1]:
                self.nest_stack[-1]['scope_variables'] = []
            self.nest_stack[-1]['scope_variables'].append(variable_name)
            self.scope_variables.append(variable_name)

            # スコープ情報を記録（IF文の条件分岐を考慮した変数管理）
            current_scope = self.nest_stack[-1]
            self.variable_scope_map[variable_name] = {
                'type': current_scope['type'],
                'scope_id': current_scope.get('scope_id')
            }
        else:
            # スコープの外で定義された変数
            self.variable_scope_map[variable_name] = {
                'type': None,
                'scope_id': None
            }

    def get_current_scope_variables(self) -> List[str]:
        """現在のスコープで定義された変数を取得"""
        return self.scope_variables.copy()

    def get_current_scope_id(self) -> Optional[Dict]:
        """現在のスコープIDを取得

        Returns:
            Dict: {'type': 'for'/'if', 'scope_id': int} または None（スコープの外）
        """
        if self.nest_stack:
            current_scope = self.nest_stack[-1]
            return {
                'type': current_scope['type'],
                'scope_id': current_scope.get('scope_id')
            }
        return None

    def is_variable_visible(self, var_name: str) -> bool:
        """変数が現在のスコープで可視かどうかをチェック

        Args:
            var_name: 変数名

        Returns:
            bool: 変数が可視な場合True
        """
        # variable_managerに存在しない変数は不可視
        var_info = self.variable_manager.get_variable_info(var_name)
        if not var_info:
            return False

        if var_name not in self.variable_scope_map:
            # variable_scope_mapに登録されていない変数は、スコープ情報が不明なため可視としない
            return False

        var_scope_info = self.variable_scope_map[var_name]
        var_scope_type = var_scope_info['type']
        var_scope_id = var_scope_info['scope_id']

        # スコープの外で定義された変数は常に可視
        if var_scope_type is None:
            return True

        # スコープスタックを確認（nest_stackを使用）
        # 現在のスコープスタック内に、変数が定義されたスコープIDが含まれているかチェック
        for scope_info in reversed(self.nest_stack):
            # 同じスコープIDに到達した場合、可視
            if (scope_info['type'] == var_scope_type and
                scope_info.get('scope_id') == var_scope_id):
                return True

        # 変数が定義されたスコープが現在のスコープスタックにない場合、不可視
        return False

    def get_current_scope_object_arrays(self) -> List[str]:
        """現在のスコープで定義されたオブジェクト配列変数のみを取得"""
        object_arrays = []
        for var_name in self.scope_variables:
            var_info = self.variable_manager.get_variable_info(var_name)
            if (var_info and
                'type_info' in var_info and
                hasattr(var_info['type_info'], 'semantic_type') and
                hasattr(var_info['type_info'], 'is_array') and
                var_info['type_info'].semantic_type.value == 'OBJECT' and
                var_info['type_info'].is_array):
                object_arrays.append(var_name)
        return object_arrays

    def get_current_scope_objects(self) -> List[str]:
        """現在のスコープで定義されたオブジェクト単体変数のみを取得"""
        objects = []
        for var_name in self.scope_variables:
            var_info = self.variable_manager.get_variable_info(var_name)
            if (var_info and
                'type_info' in var_info and
                hasattr(var_info['type_info'], 'semantic_type') and
                hasattr(var_info['type_info'], 'is_array') and
                var_info['type_info'].semantic_type.value == 'OBJECT' and
                not var_info['type_info'].is_array):
                objects.append(var_name)
        return objects

    def get_nest_scope_variables(self, nest_depth: int = -1) -> List[str]:
        """指定されたネスト深度のスコープ変数を取得

        Args:
            nest_depth: ネスト深度（-1は現在のネスト、-2は1つ外側のネスト）
        """
        if not self.nest_stack:
            return []

        # ネスト深度の調整
        if nest_depth < 0:
            nest_depth = len(self.nest_stack) + nest_depth

        if nest_depth < 0 or nest_depth >= len(self.nest_stack):
            return []

        nest = self.nest_stack[nest_depth]
        return nest.get('scope_variables', []).copy()


    def get_current_nest_type(self):
        """現在のネストタイプを取得"""
        if self.nest_stack:
            return self.nest_stack[-1]['type']
        return None

    def get_nested_node_count(self):
        """現在のネスト内のノード数を取得"""
        if self.nest_stack:
            return self.nest_stack[-1]['node_count']
        return 0


    def get_nest_history(self):
        """ネスト履歴を取得"""
        return self.nest_history.copy()

    def get_all_nest_info(self):
        """全ネスト情報を取得（現在のスタック + 履歴）"""
        return {
            'current_stack': self.nest_stack.copy(),
            'history': self.nest_history.copy(),
            'total_depth': len(self.nest_stack)
        }

    def can_exit_if_nesting(self):
        """IFネストを終了可能かチェック（ノード数に応じた確率的終了）"""
        if not self.nest_stack:
            return False
        current_nest = self.nest_stack[-1]
        if current_nest['type'] != 'if':
            return False

        # 最低1ノード生成後、確率的に終了（過剰なネストを防ぐ）
        if current_nest['node_count'] < 1:
            return False
        elif current_nest['node_count'] < 2:
            return random.random() < 0.4  # 40%の確率で終了
        elif current_nest['node_count'] < 4:
            return random.random() < 0.6   # 60%の確率で終了
        else:
            return random.random() < 0.8  # 80%の確率で終了

    def can_exit_for_nesting(self):
        """FORネストを終了可能かチェック（ノード数と配列代入ノード数に応じた確率的終了）"""
        if not self.nest_stack:
            return False
        current_nest = self.nest_stack[-1]
        if current_nest['type'] != 'for':
            return False

        # 配列代入ノード数が上限に達している場合は必ず終了（制限を超えないようにする）
        # 初期化時に必ず'array_assignment_count'が設定されているため、直接アクセス
        array_assignment_count = current_nest['array_assignment_count']
        if array_assignment_count >= self.max_for_array_assignments:
            # 上限到達時は100%の確率で終了（制限を超えないようにする）
                return True

        # 最低2ノード生成後、確率的に終了（過剰なネストを防ぐ）
        if current_nest['node_count'] < 2:
            return False
        elif current_nest['node_count'] < 3:
            return random.random() < 0.3  # 30%の確率で終了
        elif current_nest['node_count'] < 5:
            return random.random() < 0.5  # 50%の確率で終了
        elif current_nest['node_count'] < 8:
            return random.random() < 0.7  # 70%の確率で終了
        else:
            return random.random() < 0.9  # 90%の確率で終了

    def get_current_nest_node_count(self):
        """現在のネストのノード数を取得"""
        if self.nest_stack:
            return self.nest_stack[-1]['node_count']
        return 0


    def get_command_nesting_depth(self) -> int:
        """複雑度に応じたコマンド引数ネスト深度を取得（コマンド引数内でのネスト、例：FILTER(obj, EQUAL(...))）"""
        # 数値複雑度から取得
        return self.max_command_nesting_depth

    def get_scope_nesting_depth(self) -> int:
        """現在のスコープネスト深度を取得（FOR/IFのスコープネスト、例：FOR ... DO FOR ... DO ... END END）

        ネスト深度はFORとIFの深度の合計として計算される。
        """
        # FORとIFの深度を合計
        for_depth = self.get_for_nesting_depth()
        if_depth = self.get_if_nesting_depth()
        return for_depth + if_depth

    def get_max_scope_nesting_depth(self) -> int:
        """最大スコープネスト深度を取得（FOR/IFのスコープネストの上限）"""
        return self.max_nesting_depth

    def get_for_nesting_depth(self):
        """現在のFORネスト深度を取得"""
        for_count = 0
        for nest in self.nest_stack:
            if nest['type'] == 'for':
                for_count += 1
        return for_count

    def get_if_nesting_depth(self):
        """現在のIFネスト深度を取得"""
        if_count = 0
        for nest in self.nest_stack:
            if nest['type'] == 'if':
                if_count += 1
        return if_count

    # ============================================================
    # 複雑度別制御メソッド（差別化強化版）
    # ============================================================

    def can_add_node(self) -> bool:
        """ノードを追加できるかチェック"""
        can_add = self.current_lines < self.max_nodes
        return can_add

    def should_stop_generation(self) -> bool:
        """生成を停止すべきかチェック"""
        # 最小ノード数に達していない場合は継続
        if self.current_lines < self.min_nodes:
            return False

        # 最大ノード数に達した場合は停止（NODES_BY_COMPLEXITYの設定に従う）
        if self.current_lines >= self.max_nodes:
            return True

        # それ以外の場合は継続（最小ノード数に達していて、最大ノード数に達していない場合）
        return False


    def can_use_command(self, command_name: str) -> bool:
        """コマンドを使用できるかチェック"""
        # 配列操作の制限チェック
        if command_name in COMMAND_CATEGORIES['array']:
            return self.current_array_operations < self.max_array_operations
        return True


    def increment_array_operations(self):
        """配列操作カウンタを増加"""
        self.current_array_operations += 1

    def get_complexity_info(self) -> Dict:
        """複雑度情報を取得"""
        return {
            'level': self.complexity,
            'max_nodes': self.max_nodes,
            'min_nodes': self.min_nodes,
            'current_nodes': self.current_lines,
            'max_for_loops': self.max_for_loops,
            'current_for_loops': self.current_for_loops,
            'max_if_statements': self.max_if_statements,
            'current_if_statements': self.current_if_statements
        }


    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        # variable_managerがNoneの場合や、variables/used_variable_namesが存在しない場合の処理
        if self.variable_manager is None:
            variables_dict = {}
            used_variables_list = []
        else:
            # variables属性が存在するか確認
            if not hasattr(self.variable_manager, 'variables'):
                variables_dict = {}
            else:
                variables_dict = {}
                for name, info in self.variable_manager.variables.items():
                    # infoが辞書でない場合はスキップ
                    if not isinstance(info, dict):
                        continue

                    # type_infoを取得（TypeInfoオブジェクトまたは辞書）
                    type_info = info.get('type_info')

                    # type_infoがTypeInfoオブジェクトの場合
                    if type_info is not None and hasattr(type_info, 'semantic_type') and hasattr(type_info.semantic_type, 'value'):
                        type_str = type_info.semantic_type.value
                    else:
                        # type_infoがNone、辞書、または不正な形式の場合はデフォルト値
                        type_str = 'number'

                    variables_dict[name] = {
                        'value': info.get('value', None),
                        'type': type_str
                    }

            # used_variable_names属性が存在するか確認
            if not hasattr(self.variable_manager, 'used_variable_names'):
                used_variables_list = []
            else:
                try:
                    used_variables_list = list(self.variable_manager.used_variable_names)
                except Exception:
                    used_variables_list = []

        return {
            'complexity': self.complexity,
            'output_grid_width': self.output_grid_width,
            'output_grid_height': self.output_grid_height,
            'background_filter_applied': self.background_filter_applied,
            'use_filter_restore': self.use_filter_restore,
            'excluded_array': self.excluded_array,
            'variables': variables_dict,
            'used_variables': used_variables_list,
        }
