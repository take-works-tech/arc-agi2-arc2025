"""
新構文用インタープリター

ASTを実行してプログラムを動かす
ExecutorCoreと統合して既存の機能を活用
"""

from typing import Any, List, Dict
import numpy as np
import os

# インポート処理
try:
    from .parser import (
        ASTNode, Assignment, FunctionCall, Expression,
        Identifier, Literal, BinaryOp, UnaryOp,
        ListLiteral, IndexAccess, AttributeAccess,
        ForLoop, WhileLoop, IfStatement, Placeholder
    )
    from .tokenizer import Tokenizer
    from .parser import Parser
    import logging
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.core_systems.executor.parsing.parser import (
        ASTNode, Assignment, FunctionCall, Expression,
        Identifier, Literal, BinaryOp, UnaryOp,
        ListLiteral, IndexAccess, AttributeAccess,
        ForLoop, WhileLoop, IfStatement, Placeholder
    )
    from src.core_systems.executor.parsing.tokenizer import Tokenizer
    from src.core_systems.executor.parsing.parser import Parser
    import logging

logger = logging.getLogger(__name__)

# ログ出力制御（デフォルトで詳細ログを無効化）
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')

# SilentExceptionをモジュールレベルでインポート（スコープ問題を回避）
from src.core_systems.executor.core import (
    SilentException,
    MAX_OBJECTS_IN_VARIABLE,
    MAX_OBJECTS_DURING_EXECUTION,
    MAX_TOTAL_PIXELS_IN_VARIABLE,
)


class ProgramExecutionError(Exception):
    """プログラム実行エラー（連続エラー検出時など）"""
    pass


class Interpreter:
    """新構文用インタープリター"""

    def __init__(self, executor_core):
        """
        初期化

        Args:
            executor_core: ExecutorCoreインスタンス
        """
        self.executor_core = executor_core
        self.variables: Dict[str, Any] = {}

        # grid_size変数を初期化
        if hasattr(executor_core, 'grid_context'):
            grid_size = executor_core.grid_context.get_input_grid_size()
            if grid_size:
                self.variables['grid_size'] = list(grid_size)

    @staticmethod
    def _filter_none_from_array(arr: Any) -> list:
        """
        配列からNone要素を除外するヘルパーメソッド

        Args:
            arr: 配列（またはそれ以外の値）

        Returns:
            Noneが除外された配列（配列でない場合は空リスト）
        """
        if arr is None:
            return []
        if not isinstance(arr, list):
            return []
        return [item for item in arr if item is not None]

    def _collect_object_stats(self, sequence: Any) -> tuple:
        """
        配列やネスト構造からオブジェクト数と総ピクセル数を集計する
        """
        total_count = 0
        total_pixels = 0
        stack = [sequence]

        while stack:
            current = stack.pop()

            if current is None:
                continue

            if isinstance(current, list):
                stack.extend(current)
                continue

            if isinstance(current, str):
                obj = self.executor_core._find_object_by_id(current)
                if obj is not None:
                    total_count += 1
                    total_pixels += len(getattr(obj, "pixels", []))
                continue

            if hasattr(current, "object_id") and hasattr(current, "pixels"):
                total_count += 1
                total_pixels += len(getattr(current, "pixels", []))

        return total_count, total_pixels

    def _enforce_array_limits(self, sequence: Any, variable_name: str) -> None:
        """
        配列に格納されたオブジェクトに対し、数と総ピクセル数の上限をチェックする
        """
        object_count, total_pixels = self._collect_object_stats(sequence)

        if object_count > MAX_OBJECTS_IN_VARIABLE:
            error_msg = (
                f"変数 '{variable_name}' に格納されているオブジェクト数が上限を超えました"
                f"（{object_count}個 > {MAX_OBJECTS_IN_VARIABLE}個）。プログラムに問題がある可能性があります。"
            )
            logger.error(error_msg)
            raise SilentException(error_msg)

        if total_pixels > MAX_TOTAL_PIXELS_IN_VARIABLE:
            error_msg = (
                f"変数 '{variable_name}' に格納されているオブジェクトの総ピクセル数が上限を超えました"
                f"（{total_pixels}ピクセル > {MAX_TOTAL_PIXELS_IN_VARIABLE}ピクセル）。プログラムに問題がある可能性があります。"
            )
            logger.error(error_msg)
            raise SilentException(error_msg)

    def execute(self, ast: List[ASTNode]) -> Any:
        """
        ASTを実行

        Args:
            ast: ASTノードのリスト

        Returns:
            最後の式の評価結果（あれば）
        """
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[Interpreter.execute] 関数開始（ASTノード数={len(ast) if ast else 0}）", flush=True)

        result = None

        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[Interpreter.execute] ASTノードループ開始", flush=True)
        try:
            for i, node in enumerate(ast):
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"[Interpreter.execute] ノード{i+1}/{len(ast)}処理開始: {type(node).__name__}", flush=True)
                result = self.execute_node(node)
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"[Interpreter.execute] ノード{i+1}/{len(ast)}処理完了: {type(node).__name__}", flush=True)
        except SilentException:
            # SilentExceptionは即座に再発生（タスクを即座に廃棄するため）
            raise

        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[Interpreter.execute] 関数終了", flush=True)
        return result

    def execute_node(self, node: ASTNode) -> Any:
        """
        ASTノードを実行

        Args:
            node: ASTノード

        Returns:
            ノードの評価結果
        """
        if isinstance(node, Assignment):
            return self.execute_assignment(node)

        elif isinstance(node, FunctionCall):
            return self.execute_function_call(node)

        elif isinstance(node, ForLoop):
            return self.execute_for_loop(node)

        elif isinstance(node, WhileLoop):
            return self.execute_while_loop(node)

        elif isinstance(node, IfStatement):
            return self.execute_if_statement(node)

        else:
            logger.warning(f"未実装のノード型: {type(node)}")
            return None

    def execute_assignment(self, node: Assignment) -> None:
        """
        変数代入を実行（通常の変数代入と配列要素への代入の両方に対応）

        Args:
            node: Assignmentノード
        """
        from src.core_systems.executor.parsing.parser import IndexAccess, Identifier

        value = self.evaluate_expression(node.expression)

        # デバッグ情報（コメントアウト: パフォーマンス向上のため）
        # logger.info(f"代入処理: variable={node.variable}, type={type(node.variable)}, value={value}")

        # 配列要素への代入の場合 (型名による柔軟な判定)
        if hasattr(node.variable, 'target') and hasattr(node.variable, 'index'):
            # logger.info(f"配列要素代入処理開始: {node.variable}")  # パフォーマンス向上のためコメントアウト
            # 配列名を取得
            if hasattr(node.variable.target, 'name'):
                array_name = node.variable.target.name
            else:
                raise ValueError(f"配列要素代入: 無効な配列ターゲット: {node.variable.target}")

            # インデックスを評価
            index = self.evaluate_expression(node.variable.index)

            # 配列が存在するか確認
            if array_name not in self.variables:
                raise ValueError(f"配列要素代入: 配列 '{array_name}' が存在しません")

            array = self.variables[array_name]

            # 配列でない場合エラー
            if not isinstance(array, list):
                raise ValueError(f"配列要素代入: '{array_name}' は配列ではありません（型: {type(array)}）")

            # インデックスが範囲内か確認
            if not isinstance(index, int):
                raise ValueError(f"配列要素代入: インデックスは整数である必要があります（型: {type(index)}）")

            if index < 0 or index >= len(array):
                raise ValueError(f"配列要素代入: インデックス {index} が範囲外です（配列長: {len(array)}）")

            # 配列要素を更新（Noneの場合は元の値を保持）
            if value is not None:
                array[index] = value
                #logger.info(f"配列要素代入: {array_name}[{index}] = {value}")

                # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                self._enforce_array_limits(array, array_name)
            else:
                #logger.warning(f"配列要素代入: {array_name}[{index}] への代入値がNoneのため、元の値を保持します")
                pass

            # ExecutorCoreの変数も更新
            self.executor_core.execution_context['variables'][array_name] = array
        else:
            # 通常の変数代入
            self.variables[node.variable] = value

            # ExecutorCoreの変数にも登録
            self.executor_core.execution_context['variables'][node.variable] = value

            # 変数が配列の場合、その配列に格納されているオブジェクトIDの数をチェック
            if isinstance(value, list):
                self._enforce_array_limits(value, node.variable)

            logger.info(f"変数代入: {node.variable} = {value}")

        return None

    def execute_function_call(self, node: FunctionCall) -> Any:
        """
        関数呼び出しを実行

        Args:
            node: FunctionCallノード

        Returns:
            関数の戻り値
        """
        func_name = node.name

        # APPENDの第1引数は変数名として扱う（評価しない）
        if func_name == 'APPEND' and len(node.arguments) > 0:
            # 第1引数は変数名として取得
            first_arg = node.arguments[0]
            if isinstance(first_arg, Identifier):
                var_name = first_arg.name
            else:
                raise ValueError(f"APPENDの第1引数は変数名でなければなりません")

            # 第2引数以降は通常通り評価
            remaining_args = [self.evaluate_expression(arg) for arg in node.arguments[1:]]
            args = [var_name] + remaining_args

        # FILTER/SORT_BYの第2引数は評価せずに直接渡す（条件式/ソート式）
        elif func_name in ('FILTER', 'SORT_BY') and len(node.arguments) >= 2:
            # 第1引数は評価
            arg0 = self.evaluate_expression(node.arguments[0])
            # 第2引数は評価しない（ASTノードのまま渡す）
            arg1 = node.arguments[1]
            # 第3引数以降があれば評価
            remaining_args = [self.evaluate_expression(arg) for arg in node.arguments[2:]]
            args = [arg0, arg1] + remaining_args

        # MATCH_PAIRSの第3引数は評価せずに直接渡す（条件式）
        elif func_name == 'MATCH_PAIRS' and len(node.arguments) >= 3:
            # 第1引数、第2引数は評価
            arg0 = self.evaluate_expression(node.arguments[0])
            arg1 = self.evaluate_expression(node.arguments[1])
            # 第3引数（条件式）は評価しない（ASTノードのまま渡す）
            arg2 = node.arguments[2]
            args = [arg0, arg1, arg2]

        else:
            # 通常のコマンドは全ての引数を評価
            args = []
            for arg in node.arguments:
                # 各引数の評価前に$objを設定
                if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                    self.variables['$obj'] = self.current_object_id
                # 引数の評価前に$objを再度設定（再帰的な評価のため）
                if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                    self.variables['$obj'] = self.current_object_id
                args.append(self.evaluate_expression(arg))

        # ログ出力を安全にする（大きなオブジェクトや配列を省略）
        try:
            args_str_list = []
            for arg in args:
                if isinstance(arg, (list, tuple)) and len(arg) > 10:
                    args_str_list.append(f"[{type(arg).__name__} length={len(arg)}]")
                elif hasattr(arg, '__len__') and not isinstance(arg, str):
                    try:
                        if len(arg) > 10:
                            args_str_list.append(f"[{type(arg).__name__} length={len(arg)}]")
                        else:
                            args_str_list.append(str(arg))
                    except (TypeError, AttributeError):
                        args_str_list.append(str(arg)[:100] if len(str(arg)) > 100 else str(arg))
                else:
                    arg_str = str(arg)
                    args_str_list.append(arg_str[:100] if len(arg_str) > 100 else arg_str)
            logger.info(f"関数呼び出し: {func_name}({', '.join(args_str_list)})")
        except Exception as log_error:
            # ログ出力に失敗しても処理は継続
            logger.debug(f"関数呼び出し: {func_name}({len(args)}個の引数) - ログ出力エラー: {log_error}")

        # コマンドの実行
        return self.execute_command(func_name, args)

    def execute_for_loop(self, node: ForLoop) -> None:
        """
        FORループを実行

        Args:
            node: ForLoopノード
        """
        # SilentExceptionは即座に再発生（タスクを即座に廃棄するため）
        try:
            # カウント式を評価
            try:
                count_value = self.evaluate_expression(node.count_expr)
            except Exception as e:
                error_msg = f"FORループのカウント式評価中にエラーが発生しました: {type(e).__name__}: {e}, count_expr={node.count_expr}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

            # Noneチェック
            if count_value is None:
                error_msg = f"FORループのカウント値がNoneです。count_expr={node.count_expr}, loop_var={node.loop_var}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # カウントが整数でない場合はエラー（詳細な型情報を提供）
            if not isinstance(count_value, int):
                error_msg = (
                    f"FORループのカウント値が整数ではありません。"
                    f"期待: int, 実際: {type(count_value).__name__}, 値: {count_value}, "
                    f"count_expr={node.count_expr}, loop_var={node.loop_var}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 負の値チェック
            if count_value < 0:
                error_msg = f"FORループのカウント値が負の値です: {count_value}, count_expr={node.count_expr}, loop_var={node.loop_var}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 無限ループ防止のための最大反復回数
            MAX_FOR_LOOP_ITERATIONS = 1000
            if count_value > MAX_FOR_LOOP_ITERATIONS:
                logger.warning(f"FOR loop count ({count_value}) exceeds maximum ({MAX_FOR_LOOP_ITERATIONS}), limiting to {MAX_FOR_LOOP_ITERATIONS}")
                count_value = MAX_FOR_LOOP_ITERATIONS

            # ループを実行
            try:
                for i in range(count_value):
                    # 各反復の開始時に総オブジェクト数をチェック
                    all_objects = self.executor_core.execution_context.get('objects', {})
                    object_count = len(all_objects) if isinstance(all_objects, dict) else len(all_objects) if isinstance(all_objects, list) else 0
                    # SilentExceptionはモジュールレベルでインポート済み
                    if object_count > MAX_OBJECTS_DURING_EXECUTION:
                        error_msg = f"FORループ実行中、オブジェクト数が上限を超えました（{object_count}個 > {MAX_OBJECTS_DURING_EXECUTION}個）。プログラムに問題がある可能性があります。"
                        logger.error(error_msg)
                        raise SilentException(error_msg)

                    # ループ変数を設定
                    self.variables[node.loop_var] = i

                    # 進捗ログ（大きなループの場合のみ）
                    if count_value > 10 and i % max(1, count_value // 10) == 0:
                        logger.info(f"FORループ進捗: {i+1}/{count_value} ({100*(i+1)//count_value}%)")

                    # ループ本体を実行
                    for stmt in node.body:
                        self.execute_node(stmt)
            except TypeError as e:
                # 'int' object is not iterable エラーの詳細な情報を提供
                error_msg = (
                    f"FORループ実行中にTypeErrorが発生しました: {e}. "
                    f"count_value={count_value} (type: {type(count_value).__name__}), "
                    f"count_expr={node.count_expr}, loop_var={node.loop_var}, "
                    f"range(count_value)={range(count_value) if isinstance(count_value, int) else 'N/A'}"
                )
                logger.error(error_msg)
                raise TypeError(error_msg) from e
        except SilentException:
            # SilentExceptionは即座に再発生（タスクを即座に廃棄するため）
            raise
        except (ValueError, TypeError) as e:
            # 詳細なエラー情報を提供してから再発生
            logger.error(f"FORループ実行エラー: {type(e).__name__}: {e}")
            raise

    def execute_while_loop(self, node: WhileLoop) -> None:
        """
        WHILEループを実行

        Args:
            node: WhileLoopノード
        """
        # 無限ループ防止のための最大反復回数
        max_iterations = 1000
        iteration_count = 0

        # 条件が真の間、ループを実行
        while True:
            # 無限ループチェック
            if iteration_count >= max_iterations:
                logger.warning(f"WHILE loop exceeded maximum iterations ({max_iterations})")
                break

            # 各反復の開始時に総オブジェクト数をチェック
            all_objects = self.executor_core.execution_context.get('objects', {})
            object_count = len(all_objects) if isinstance(all_objects, dict) else len(all_objects) if isinstance(all_objects, list) else 0
            # SilentExceptionはモジュールレベルでインポート済み
            if object_count > MAX_OBJECTS_DURING_EXECUTION:
                error_msg = f"WHILEループ実行中、オブジェクト数が上限を超えました（{object_count}個 > {MAX_OBJECTS_DURING_EXECUTION}個）。プログラムに問題がある可能性があります。"
                logger.error(error_msg)
                raise SilentException(error_msg)

            # 条件式を評価
            condition_value = self.evaluate_expression(node.condition)

            # 条件が偽の場合、ループ終了
            if not condition_value:
                break

            # ループ本体を実行
            for stmt in node.body:
                self.execute_node(stmt)

            iteration_count += 1

    def execute_if_statement(self, node: IfStatement) -> None:
        """
        IF文を実行

        Args:
            node: IfStatementノード
        """
        # 条件式を評価
        condition_value = self.evaluate_expression(node.condition)

        # 条件が真の場合
        if condition_value:
            for stmt in node.then_body:
                self.execute_node(stmt)
        # 条件が偽の場合
        else:
            for stmt in node.else_body:
                self.execute_node(stmt)

    def execute_command(self, command_name: str, args: List[Any]) -> Any:
        """
        コマンドを実行

        Args:
            command_name: コマンド名
            args: 引数リスト

        Returns:
            コマンドの戻り値
        """
        # ========================================
        # 基本設定
        # ========================================

        # ========================================
        # 情報取得
        # ========================================

        if command_name == "GET_ALL_OBJECTS":
            # 引数で連結性を指定（4 or 8）- 必須
            if len(args) == 0:
                raise ValueError("GET_ALL_OBJECTS: connectivity argument required (4 or 8)")

            connectivity = args[0]
            if connectivity not in [4, 8]:
                raise ValueError(f"GET_ALL_OBJECTS: connectivity must be 4 or 8, got {connectivity}")

            # 連結性に応じてオブジェクトを取得
            objects = self.executor_core._get_all_objects_with_connectivity(connectivity)
            logger.info(f"GET_ALL_OBJECTS({connectivity}): {len(objects)}個のオブジェクト")
            # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
            # 上限を超えた場合は即座に実行を停止
            if isinstance(objects, list):
                self._enforce_array_limits(objects, "GET_ALL_OBJECTS結果")
            return objects

        elif command_name == "RENDER_GRID":
            # RENDER_GRID(objects, background_color, width, height)
            # または RENDER_GRID(objects, background_color, x, y, width, height)
            if len(args) < 3:
                raise ValueError("RENDER_GRID: 最低3つの引数が必要です")

            object_ids = args[0] if args[0] else []
            # 配列からNoneを除外
            if isinstance(object_ids, list):
                object_ids = self._filter_none_from_array(object_ids)
                logger.info(f"RENDER_GRID: 入力配列からNoneを除外後 {len(object_ids)}個のオブジェクト")

            background_color = args[1] if len(args) > 1 else 0

            # 引数の数で判定
            if len(args) == 4:
                # 簡潔版: objects, bg_color, width, height
                width = args[2]
                height = args[3]
                output_grid = self.executor_core._render_grid(object_ids, background_color, width, height)
            elif len(args) >= 6:
                # フルバージョン: objects, bg_color, x, y, width, height
                x = args[2]
                y = args[3]
                width = args[4]
                height = args[5]
                output_grid = self.executor_core._render_grid(object_ids, background_color, width, height, x, y)
            else:
                raise ValueError(f"RENDER_GRID: 引数の数が不正です（{len(args)}個）")

            # execution_contextに保存
            logger.info(f"RENDER_GRID: execution_contextに保存開始 - {len(object_ids)}個のオブジェクト")
            self.executor_core.execution_context['output_grid'] = output_grid
            self.executor_core.execution_context['rendered_objects'] = object_ids
            self.executor_core.execution_context['program_terminated'] = True
            logger.info(f"RENDER_GRID: program_terminated = {self.executor_core.execution_context.get('program_terminated')}")

            logger.info(f"RENDER_GRID実行: {width}x{height}")
            return None

        elif command_name == "GET_BACKGROUND_COLOR":
            bg_color = self.executor_core.execution_context.get('background_color', 0)
            logger.info(f"[DEBUG] GET_BACKGROUND_COLOR() = {bg_color}")
            return bg_color

        elif command_name == "GET_SIZE":
            obj_id = args[0] if args else None
            if obj_id:
                obj = self.executor_core._find_object_by_id(obj_id)
                if obj and obj.pixels:
                    return len(obj.pixels)
            return 0

        elif command_name == "GET_WIDTH":
            obj_id = args[0] if args else None
            if obj_id:
                obj = self.executor_core._find_object_by_id(obj_id)
                if obj:
                    x1, y1, x2, y2 = obj.bbox
                    return x2 - x1 + 1
            return 0

        elif command_name == "GET_HEIGHT":
            obj_id = args[0] if args else None
            if obj_id:
                obj = self.executor_core._find_object_by_id(obj_id)
                if obj:
                    x1, y1, x2, y2 = obj.bbox
                    return y2 - y1 + 1
            return 0

        elif command_name == "GET_COLOR":
            obj_id = args[0] if args else None
            if obj_id:
                # 正常な場合は色リストを取得
                colors = self.executor_core._get_color_list(obj_id)
                color_value = colors[0] if colors else 0
                logger.info(f"[DEBUG] GET_COLOR({obj_id}) = {color_value}")
                return color_value
            # obj_idがNoneの場合は即座に例外を発生させてプログラム実行を終了（1回目のエラーで即座にタスク破棄）
            error_msg = f"プログラム実行エラー: GET_COLORでobj_idがNoneが発生しました。プログラムに問題があるため実行を終了します。"
            raise ProgramExecutionError(error_msg)

        elif command_name == "GET_COLORS":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_color_list(obj_id)
            return []

        # ========================================
        # 配列操作
        # ========================================

        elif command_name == "CREATE_ARRAY":
            return []

        elif command_name == "LEN":
            array = args[0] if args else []
            if isinstance(array, list):
                return len(array)
            return 0

        elif command_name == "REVERSE":
            # REVERSE(array) - 配列を逆順にする
            array_input = args[0] if args else None
            if array_input is None:
                return []

            # 変数名の場合
            if isinstance(array_input, str) and array_input in self.variables:
                array = self.variables.get(array_input, [])
            else:
                array = array_input

            if isinstance(array, list):
                # 新しい配列を作成して逆順に（イミュータブル）
                reversed_array = list(reversed(array))
                # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                if isinstance(reversed_array, list):
                    self._enforce_array_limits(reversed_array, "REVERSE結果")
                return reversed_array
            return []

        elif command_name == "SORT_BY":
            # 新構文: SORT_BY(objects, key_command, order)
            # key_command: GET_SIZE（識別子）または GET_DISTANCE(ref_obj)（関数呼び出し）
            # イミュータブル: 新しい配列を返す（元の配列は変更しない）
            if len(args) < 3:
                raise ValueError("SORT_BY: 3つの引数（array, key_command, order）が必要です")

            array_input = args[0]
            key_command = args[1]
            order = args[2]

            # orderの検証
            if order not in ["asc", "desc"]:
                raise ValueError(f"SORT_BY: orderは'asc'または'desc'である必要があります（実際: {order}）")

            # 配列を取得
            if isinstance(array_input, list):
                array = array_input
            elif isinstance(array_input, str) and array_input in self.variables:
                array = self.variables.get(array_input, [])
            else:
                return []

            if isinstance(array, list):
                # 配列からNoneを除外
                array = self._filter_none_from_array(array)
                logger.info(f"SORT_BY: 入力配列からNoneを除外後 {len(array)}個のオブジェクト")

                # 新しいソート済み配列を返す（元の配列は変更しない）
                sorted_array = self.executor_core._sort_by_unified(array, key_command, order, self)
                # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                # 上限を超えた場合は即座に実行を停止
                if isinstance(sorted_array, list):
                    self._enforce_array_limits(sorted_array, "SORT_BY結果")
                return sorted_array

            return []

        elif command_name == "EXTEND_PATTERN":
            # EXTEND_PATTERN(objects, side, count)
            objects_input = args[0] if len(args) > 0 else None
            side = args[1] if len(args) > 1 else "end"
            count = args[2] if len(args) > 2 else 1

            if objects_input is not None:
                # 配列からNoneを除外
                if isinstance(objects_input, list):
                    objects_input = self._filter_none_from_array(objects_input)
                    logger.info(f"EXTEND_PATTERN: 入力配列からNoneを除外後 {len(objects_input)}個のオブジェクト")

                result = self.executor_core._extend_pattern(objects_input, side, count)
                return result
            return []

        # MAP は廃止
        # → 全てのオブジェクト操作コマンドが配列引数をサポートするため不要

        elif command_name == "FILTER":
            # 新構文: FILTER(objects, condition_expr)
            # condition_expr: GREATER(GET_SIZE, 10)（関数呼び出し）
            objects_input = args[0] if len(args) > 0 else None
            condition_expr = args[1] if len(args) > 1 else None

            logger.info(f"[DEBUG] FILTER呼び出し: 入力オブジェクト数={len(objects_input) if isinstance(objects_input, list) else 'N/A'}")

            if objects_input is not None and condition_expr is not None:
                # 配列からNoneを除外
                if isinstance(objects_input, list):
                    objects_input = self._filter_none_from_array(objects_input)
                    logger.info(f"FILTER: 入力配列からNoneを除外後 {len(objects_input)}個のオブジェクト")

                result = self.executor_core._filter_unified(objects_input, condition_expr, self)
                logger.info(f"[DEBUG] FILTER戻り値: {len(result)}個のオブジェクト")
                # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                # 上限を超えた場合は即座に実行を停止
                if isinstance(result, list):
                    self._enforce_array_limits(result, "FILTER結果")
                return result
            return []

        # REDUCE廃止: SORT_BY + [0]アクセスで代替

        elif command_name == "CONCAT":
            # CONCAT(array1, array2)
            array1 = args[0] if len(args) > 0 else None
            array2 = args[1] if len(args) > 1 else None

            if array1 is not None and array2 is not None:
                # 両方の配列からNoneを除外してから結合
                filtered_array1 = [item for item in array1 if item is not None] if isinstance(array1, list) else array1
                filtered_array2 = [item for item in array2 if item is not None] if isinstance(array2, list) else array2

                result = self.executor_core._concat(filtered_array1, filtered_array2)
                # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                # 上限を超えた場合は即座に実行を停止
                if isinstance(result, list):
                    self._enforce_array_limits(result, "CONCAT結果")
                return result
            return []

        elif command_name == "EXCLUDE":
            # EXCLUDE(array, targets)
            # オブジェクト配列から特定のオブジェクトを完全一致で除外
            array = args[0] if len(args) > 0 else None
            targets = args[1] if len(args) > 1 else None

            if array is None or targets is None:
                logger.error(f"EXCLUDE: 引数が不足しています")
                return []

            # 配列の正規化
            if not isinstance(array, list):
                array = [array]
            if not isinstance(targets, list):
                targets = [targets]

            # 配列からNoneを除外
            array = self._filter_none_from_array(array)
            targets = self._filter_none_from_array(targets)

            # オブジェクト数が異常に多い場合はエラー（無限ループ防止）
            from src.core_systems.executor.core import MAX_OBJECTS_FOR_EXCLUDE
            if len(array) > MAX_OBJECTS_FOR_EXCLUDE or len(targets) > MAX_OBJECTS_FOR_EXCLUDE:
                error_msg = f"EXCLUDE: オブジェクト数が上限を超えています（array={len(array)}, targets={len(targets)}, 上限={MAX_OBJECTS_FOR_EXCLUDE}）。プログラムに問題がある可能性があります。"
                logger.error(error_msg)
                # execution_contextのオブジェクト数を確認
                all_objects = self.executor_core.execution_context.get('objects', {})
                obj_count = len(all_objects) if isinstance(all_objects, dict) else len(all_objects) if isinstance(all_objects, list) else 0
                logger.error(f"EXCLUDE: execution_context['objects']の数={obj_count}")
                from src.core_systems.executor.parsing.interpreter import ProgramExecutionError
                raise ProgramExecutionError(error_msg)

            # 大量のオブジェクトが渡されている場合は警告
            if len(array) > 100 or len(targets) > 100:
                logger.warning(f"EXCLUDE: 大量のオブジェクトが渡されています（array={len(array)}, targets={len(targets)}）")
                # execution_contextのオブジェクト数を確認
                all_objects = self.executor_core.execution_context.get('objects', {})
                obj_count = len(all_objects) if isinstance(all_objects, dict) else len(all_objects) if isinstance(all_objects, list) else 0
                logger.warning(f"EXCLUDE: execution_context['objects']の数={obj_count}")

                # 引数のソースを確認（最初の10個のサンプル）
                if len(array) > 0:
                    sample_array = array[:10] if isinstance(array, list) else [array]
                    logger.warning(f"EXCLUDE: arrayのサンプル（最初の10個）: {sample_array}")
                if len(targets) > 0:
                    sample_targets = targets[:10] if isinstance(targets, list) else [targets]
                    logger.warning(f"EXCLUDE: targetsのサンプル（最初の10個）: {sample_targets}")

            logger.info(f"EXCLUDE: {len(array)}個から{len(targets)}個を除外")

            result = self.executor_core._exclude(array, targets)
            # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
            # 上限を超えた場合は即座に実行を停止
            if isinstance(result, list) and result is not None:
                self._enforce_array_limits(result, "EXCLUDE結果")
            return result if result is not None else []

        elif command_name == "ARRANGE_GRID":
            # ARRANGE_GRID(objects, columns, cell_width, cell_height)
            objects = args[0] if len(args) > 0 else None
            columns = args[1] if len(args) > 1 else None
            cell_width = args[2] if len(args) > 2 else None
            cell_height = args[3] if len(args) > 3 else None

            if not isinstance(objects, list):
                logger.error(f"ARRANGE_GRIDの第1引数は配列である必要があります: {objects}")
                return []

            # 配列からNoneを除外
            objects = self._filter_none_from_array(objects)
            logger.info(f"ARRANGE_GRID: 入力配列からNoneを除外後 {len(objects)}個のオブジェクト")

            if not isinstance(columns, int) or columns <= 0:
                logger.error(f"ARRANGE_GRIDの第2引数(columns)は正の整数である必要があります: {columns}")
                return objects

            if not isinstance(cell_width, int) or cell_width <= 0:
                logger.error(f"ARRANGE_GRIDの第3引数(cell_width)は正の整数である必要があります: {cell_width}")
                return objects

            if not isinstance(cell_height, int) or cell_height <= 0:
                logger.error(f"ARRANGE_GRIDの第4引数(cell_height)は正の整数である必要があります: {cell_height}")
                return objects

            result = self.executor_core._arrange_grid(objects, columns, cell_width, cell_height)
            return result if result is not None else []

        elif command_name == "MATCH_PAIRS":
            # MATCH_PAIRS(objects1, objects2, condition_expr)
            objects1 = args[0] if len(args) > 0 else None
            objects2 = args[1] if len(args) > 1 else None
            condition_expr = args[2] if len(args) > 2 else None

            if not isinstance(objects1, list):
                logger.error(f"MATCH_PAIRSの第1引数は配列である必要があります: {objects1}")
                return []

            if not isinstance(objects2, list):
                logger.error(f"MATCH_PAIRSの第2引数は配列である必要があります: {objects2}")
                return []

            if condition_expr is None:
                logger.error("MATCH_PAIRSの第3引数（条件式）が必要です")
                return []

            # 条件式を_match_pairsに渡す（ASTとして）
            result = self.executor_core._match_pairs(
                objects1,
                objects2,
                condition_expr,
                self.variables
            )
            return result if result is not None else []

        elif command_name == "APPEND":
            array_name = args[0] if len(args) > 0 else None
            value = args[1] if len(args) > 1 else None
            if array_name and array_name in self.variables:
                if isinstance(self.variables[array_name], list):
                    # 新しい配列を作成して返す（破壊的変更を避ける）
                    new_array = self.variables[array_name].copy()
                    # Noneは追加しない（自動除外）
                    if value is not None:
                        new_array.append(value)
                        # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
                        # 上限を超えた場合は即座に実行を停止
                        self._enforce_array_limits(new_array, array_name)
                    else:
                        logger.info(f"APPEND: Noneは配列に追加されません（スキップ）")
                    return new_array
            return []

        # ========================================
        # 算術演算
        # ========================================

        elif command_name == "ADD":
            """整数加算"""
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 0

            # 型チェック：整数のみ受け付ける
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(
                    f"ADDは整数のみ受け付けます: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})"
                )

            return a + b

        elif command_name == "SUB":
            """算術演算: 整数の引き算"""
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 0

            if isinstance(a, int) and isinstance(b, int):
                return a - b
            else:
                raise TypeError(f"SUBは整数のみ: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})")

        elif command_name == "SUBTRACT":
            """オブジェクト差分: obj1からobj2のピクセルを除去"""
            a = args[0] if len(args) > 0 else None
            b = args[1] if len(args) > 1 else None

            if isinstance(a, str) and isinstance(b, str):
                # オブジェクト減算
                return self.executor_core._subtract(a, b)
            else:
                raise TypeError(f"SUBTRACTはオブジェクトIDのペアのみ: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})")

        elif command_name == "MULTIPLY":
            """整数乗算"""
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 0

            # 型チェック：整数のみ受け付ける
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(
                    f"MULTIPLYは整数のみ受け付けます: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})"
                )

            return a * b

        elif command_name == "DIVIDE":
            """整数除算（floor、切り捨て）"""
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 1

            # 型チェック：整数のみ受け付ける
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(
                    f"DIVIDEは整数のみ受け付けます: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})"
                )

            # ゼロ除算チェック
            if b == 0:
                raise ValueError("DIVIDEでゼロ除算が発生しました")

            # floor除算（常に整数を返す）
            return a // b

        elif command_name == "MOD":
            """整数の剰余（modulo）"""
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 1

            # 型チェック：整数のみ受け付ける
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(
                    f"MODは整数のみ受け付けます: a={a} ({type(a).__name__}), b={b} ({type(b).__name__})"
                )

            # ゼロ除算チェック
            if b == 0:
                raise ValueError("MODでゼロ除算が発生しました")

            # 剰余を返す
            return a % b

        # MIN, MAX, ABS は廃止
        # 代替: IF-THEN-ELSE で実装可能

        # ========================================
        # 論理演算
        # ========================================

        elif command_name == "AND":
            a = args[0] if len(args) > 0 else False
            b = args[1] if len(args) > 1 else False
            return a and b

        elif command_name == "OR":
            a = args[0] if len(args) > 0 else False
            b = args[1] if len(args) > 1 else False
            return a or b

        # ========================================
        # 比較演算
        # ========================================

        elif command_name == "EQUAL":
            a = args[0] if len(args) > 0 else None
            b = args[1] if len(args) > 1 else None
            return a == b

        elif command_name == "NOT_EQUAL":
            a = args[0] if len(args) > 0 else None
            b = args[1] if len(args) > 1 else None
            return a != b

        elif command_name == "GREATER":
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 0
            return a > b

        elif command_name == "LESS":
            a = args[0] if len(args) > 0 else 0
            b = args[1] if len(args) > 1 else 0
            return a < b

        # ========================================
        # オブジェクト操作
        # ========================================

        elif command_name == "MOVE":
            obj_id = args[0] if len(args) > 0 else None
            dx = args[1] if len(args) > 1 else 0
            dy = args[2] if len(args) > 2 else 0

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # すべての新しいオブジェクトIDを配列で返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._move(single_obj_id, dx, dy)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合: 新しいオブジェクトIDを返す
                return self.executor_core._move(obj_id, dx, dy)

        elif command_name == "ROTATE":
            obj_id = args[0] if len(args) > 0 else None
            angle = args[1] if len(args) > 1 else 0

            # 回転中心座標（オプション）
            center_x = args[2] if len(args) > 2 else None
            center_y = args[3] if len(args) > 3 else None

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # すべての新しいオブジェクトIDを配列で返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._rotate(single_obj_id, angle, center_x, center_y)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合: 新しいオブジェクトIDを返す
                return self.executor_core._rotate(obj_id, angle, center_x, center_y)

        elif command_name == "SET_COLOR":
            obj_id = args[0] if len(args) > 0 else None
            color = args[1] if len(args) > 1 else 0

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトの色を変更して新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._color_change(single_obj_id, color)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._color_change(obj_id, color)

        # ========================================
        # 基本オブジェクト操作（残り）
        # ========================================

        elif command_name == "SLIDE":
            # SLIDE(obj, direction, obstacles)
            obj_id = args[0] if len(args) > 0 else None
            direction = args[1] if len(args) > 1 else "-Y"
            obstacles = args[2] if len(args) > 2 else []

            # obstaclesからNoneを除外
            if isinstance(obstacles, list):
                obstacles = self._filter_none_from_array(obstacles)

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトを移動して新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._slide(single_obj_id, direction, obstacles)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._slide(obj_id, direction, obstacles)

        elif command_name == "ALIGN":
            obj_id = args[0] if len(args) > 0 else None
            mode = args[1] if len(args) > 1 else "center"

            if obj_id is None:
                return None

            # 有効なモードをチェック
            valid_modes = ["left", "right", "top", "bottom", "center_x", "center_y", "center"]
            if mode not in valid_modes:
                raise ValueError(f"ALIGN: 無効なmode '{mode}'。使用可能: {', '.join(valid_modes)}")

            if isinstance(obj_id, list):
                # 配列の場合、各オブジェクトを整列
                aligned_ids = []
                for single_obj_id in obj_id:
                    aligned_id = self.executor_core._align(single_obj_id, mode)
                    if aligned_id:
                        aligned_ids.append(aligned_id)
                return aligned_ids if aligned_ids else None
            else:
                # 単一オブジェクトの場合
                return self.executor_core._align(obj_id, mode)

        elif command_name == "TELEPORT":
            obj_id = args[0] if len(args) > 0 else None
            x = args[1] if len(args) > 1 else 0
            y = args[2] if len(args) > 2 else 0

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトをテレポートして新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._teleport(single_obj_id, x, y)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._teleport(obj_id, x, y)

        elif command_name == "PATHFIND":
            # PATHFIND(obj, target_x, target_y, obstacles)
            obj_id = args[0] if len(args) > 0 else None
            target_x = args[1] if len(args) > 1 else 0
            target_y = args[2] if len(args) > 2 else 0
            obstacles = args[3] if len(args) > 3 else []

            # obstaclesからNoneを除外
            if isinstance(obstacles, list):
                obstacles = self._filter_none_from_array(obstacles)

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトを経路探索移動して新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._pathfind(single_obj_id, target_x, target_y, obstacles)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._pathfind(obj_id, target_x, target_y, obstacles)

        elif command_name == "SCALE":
            """オブジェクトをn倍に拡大（n >= 1の整数のみ）"""
            obj_id = args[0] if len(args) > 0 else None
            factor = args[1] if len(args) > 1 else 1

            # 型チェック：整数のみ受け付ける
            if not isinstance(factor, int):
                raise TypeError(
                    f"SCALEの倍率は整数のみ: factor={factor} ({type(factor).__name__}). "
                    f"縮小にはSCALE_DOWNを使用してください"
                )

            # 範囲チェック：1以上
            if factor < 1:
                raise ValueError(
                    f"SCALEの倍率は1以上: factor={factor}. "
                    f"縮小にはSCALE_DOWNを使用してください"
                )

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトをスケールして新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._scale(single_obj_id, factor)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._scale(obj_id, factor)

        elif command_name == "SCALE_DOWN":
            """オブジェクトを1/n倍に縮小（n >= 2の整数のみ）"""
            obj_id = args[0] if len(args) > 0 else None
            divisor = args[1] if len(args) > 1 else 2

            # 型チェック：整数のみ受け付ける
            if not isinstance(divisor, int):
                raise TypeError(
                    f"SCALE_DOWNの除数は整数のみ: divisor={divisor} ({type(divisor).__name__})"
                )

            # 範囲チェック：2以上
            if divisor < 2:
                raise ValueError(
                    f"SCALE_DOWNの除数は2以上: divisor={divisor}"
                )

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # 各オブジェクトを縮小して新しいIDのリストを返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._scale_down(single_obj_id, divisor)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合、新しいIDを返す
                return self.executor_core._scale_down(obj_id, divisor)


        elif command_name == "FILL_HOLES":
            obj_id = args[0] if len(args) > 0 else None
            color = args[1] if len(args) > 1 else 0

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトの穴を埋めて新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._fill_hole(single_obj_id, color)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._fill_hole(obj_id, color)
            return None

        elif command_name == "EXPAND":
            obj_id = args[0] if len(args) > 0 else None
            pixels = args[1] if len(args) > 1 else 1

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトを拡張して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._expand(single_obj_id, pixels)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._expand(obj_id, pixels)
            return None

        elif command_name == "FLIP":
            if len(args) < 2:
                raise ValueError("FLIP: 2つの引数（object_id, axis）が必要です")

            obj_id = args[0]
            axis = args[1]

            # 軸の検証
            if axis not in ["X", "Y"]:
                raise ValueError(f"FLIP: axisは'X'または'Y'である必要があります（実際: {axis}）")

            # 軸座標（オプション）
            axis_pos = args[2] if len(args) > 2 else None

            if obj_id is None:
                return None

            if isinstance(obj_id, list):
                # 配列の場合、Noneを除外してから処理
                obj_id = self._filter_none_from_array(obj_id)
                if not obj_id:
                    return None

                # すべての新しいオブジェクトIDを配列で返す
                new_ids = []
                for single_obj_id in obj_id:
                    new_id = self.executor_core._flip(single_obj_id, axis, axis_pos)
                    if new_id:
                        new_ids.append(new_id)
                return new_ids if new_ids else None
            else:
                # 単一オブジェクトの場合: 新しいオブジェクトIDを返す
                return self.executor_core._flip(obj_id, axis, axis_pos)
            return None

        # ========================================
        # 高度なオブジェクト操作
        # ========================================

        elif command_name == "FLOW":
            # 新仕様: FLOW(obj, direction, obstacles)
            obj_id = args[0] if len(args) > 0 else None
            direction = args[1] if len(args) > 1 else "Y"
            obstacles = args[2] if len(args) > 2 else []

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトを流して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._flow(single_obj_id, direction, obstacles)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._flow(obj_id, direction, obstacles)
            return None

        elif command_name == "DRAW":
            obj_id = args[0] if len(args) > 0 else None
            dx = args[1] if len(args) > 1 else 0
            dy = args[2] if len(args) > 2 else 0

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトを描画して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._draw(single_obj_id, dx, dy)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._draw(obj_id, dx, dy)
            return None

        elif command_name == "LAY":
            # 新仕様: LAY(obj, direction, obstacles)
            obj_id = args[0] if len(args) > 0 else None
            direction = args[1] if len(args) > 1 else "Y"
            obstacles = args[2] if len(args) > 2 else []

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトを配置して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._lay(single_obj_id, direction, obstacles)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._lay(obj_id, direction, obstacles)
            return None

        elif command_name == "OUTLINE":
            obj_id = args[0] if args else None
            color = args[1] if len(args) > 1 else 1

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトの輪郭を作成して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._outline(single_obj_id, color)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._outline(obj_id, color)
            return None

        elif command_name == "HOLLOW":
            obj_id = args[0] if args else None

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトを中空化して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._hollow(single_obj_id)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._hollow(obj_id)
            return None

        elif command_name == "BBOX":
            obj_id = args[0] if args else None
            color = args[1] if len(args) > 1 else None  # 色が指定されている場合

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、各オブジェクトのBBOXを抽出して新しいIDのリストを返す
                    new_ids = []
                    for single_obj_id in obj_id:
                        new_id = self.executor_core._BBOX(single_obj_id, color)
                        if new_id:
                            new_ids.append(new_id)
                    return new_ids if new_ids else None
                else:
                    # 単一オブジェクトの場合、新しいIDを返す
                    return self.executor_core._BBOX(obj_id, color)
            return None

        elif command_name == "INTERSECTION":
            target_id = args[0] if len(args) > 0 else None
            intersect_id = args[1] if len(args) > 1 else None

            if target_id and intersect_id:
                # 単一オブジェクト同士の交差
                return self.executor_core._intersection(target_id, intersect_id)
            return None

        elif command_name == "COUNT_OVERLAP":
            """2つのオブジェクトの重複ピクセル数を返す"""
            obj1_id = args[0] if len(args) > 0 else None
            obj2_id = args[1] if len(args) > 1 else None

            if obj1_id is None or obj2_id is None:
                return 0

            # 両方のオブジェクトIDを取得して重複ピクセル数を計算
            return self.executor_core._count_overlap(obj1_id, obj2_id)



        # ========================================
        # オブジェクト生成
        # ========================================

        elif command_name == "CREATE_LINE":
            x = args[0] if len(args) > 0 else 0
            y = args[1] if len(args) > 1 else 0
            length = args[2] if len(args) > 2 else 1
            direction = args[3] if len(args) > 3 else "X"  # v3.1: デフォルトをXに変更
            color = args[4] if len(args) > 4 else 1

            # direction値の検証（v3.1: 8方向仕様 + "C"）
            valid_directions = ['X', 'Y', '-X', '-Y', 'XY', '-XY', 'X-Y', '-X-Y', 'C']
            if direction not in valid_directions:
                raise ValueError(f"CREATE_LINE: 無効なdirection '{direction}'。使用可能: {', '.join(valid_directions)}")

            obj_id = self.executor_core._create_line(x, y, length, direction, color)
            return obj_id  # 作成したオブジェクトIDを返す

        elif command_name == "CREATE_RECT":
            x = args[0] if len(args) > 0 else 0
            y = args[1] if len(args) > 1 else 0
            width = args[2] if len(args) > 2 else 1
            height = args[3] if len(args) > 3 else 1
            color = args[4] if len(args) > 4 else 1

            obj_id = self.executor_core._create_rectangle(x, y, width, height, color)
            return obj_id  # 作成したオブジェクトIDを返す

        # ========================================
        # 分割操作
        # ========================================


        elif command_name == "SPLIT_CONNECTED":
            obj_id = args[0] if len(args) > 0 else None
            connectivity = args[1] if len(args) > 1 else 4

            all_result_ids = []  # すべての分割結果を1次元配列にまとめる

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、すべてのオブジェクトの分割結果を1次元配列にまとめる
                    for single_obj_id in obj_id:
                        obj = self.executor_core._find_object_by_id(single_obj_id)
                        if obj:
                            result_ids = self.executor_core._split_by_connection(obj, connectivity)
                            all_result_ids.extend(result_ids)
                else:
                    # 単一オブジェクトの場合
                    obj = self.executor_core._find_object_by_id(obj_id)
                    if obj:
                        all_result_ids = self.executor_core._split_by_connection(obj, connectivity)

            return all_result_ids  # 分割されたすべてのオブジェクトIDを返す

        elif command_name == "CROP":
            obj_id = args[0] if len(args) > 0 else None
            x = args[1] if len(args) > 1 else 0
            y = args[2] if len(args) > 2 else 0
            width = args[3] if len(args) > 3 else 1
            height = args[4] if len(args) > 4 else 1

            if obj_id is not None:
                # 単一オブジェクトのみ受け付ける
                if isinstance(obj_id, list):
                    raise TypeError(f"CROP: 単体オブジェクトのみ受け付けます（配列が渡されました）")

                obj = self.executor_core._find_object_by_id(str(obj_id))
                if obj:
                    cropped_id = self.executor_core._crop(obj, x, y, width, height)
                    return cropped_id  # 単一オブジェクトIDを返す

            return obj_id  # エラー時は元のIDを返す

        elif command_name == "EXTRACT_LINES":
            obj_id = args[0] if args else None

            all_result_ids = []  # すべての分割結果を1次元配列にまとめる

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、すべてのオブジェクトの分割結果を1次元配列にまとめる
                    for single_obj_id in obj_id:
                        obj = self.executor_core._find_object_by_id(single_obj_id)
                        if obj:
                            result_ids = self.executor_core._split_by_line_detection(obj)
                            all_result_ids.extend(result_ids)
                else:
                    # 単一オブジェクトの場合
                    obj = self.executor_core._find_object_by_id(obj_id)
                    if obj:
                        all_result_ids = self.executor_core._split_by_line_detection(obj)

            return all_result_ids  # 分割されたすべてのオブジェクトIDを返す

        elif command_name == "EXTRACT_RECTS":
            obj_id = args[0] if args else None

            all_result_ids = []  # すべての分割結果を1次元配列にまとめる

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、すべてのオブジェクトの分割結果を1次元配列にまとめる
                    for single_obj_id in obj_id:
                        obj = self.executor_core._find_object_by_id(single_obj_id)
                        if obj:
                            result_ids = self.executor_core._extract_rectangles(obj)
                            all_result_ids.extend(result_ids)
                else:
                    # 単一オブジェクトの場合
                    obj = self.executor_core._find_object_by_id(obj_id)
                    if obj:
                        all_result_ids = self.executor_core._extract_rectangles(obj)

            return all_result_ids  # 分割されたすべてのオブジェクトIDを返す

        elif command_name == "EXTRACT_HOLLOW_RECTS":
            obj_id = args[0] if args else None

            all_result_ids = []

            if obj_id:
                if isinstance(obj_id, list):
                    # 配列の場合、すべてのオブジェクトの抽出結果を1次元配列にまとめる
                    for single_obj_id in obj_id:
                        obj = self.executor_core._find_object_by_id(single_obj_id)
                        if obj:
                            result_ids = self.executor_core._extract_hollow_rectangles(obj)
                            all_result_ids.extend(result_ids)
                else:
                    # 単一オブジェクトの場合
                    obj = self.executor_core._find_object_by_id(obj_id)
                    if obj:
                        all_result_ids = self.executor_core._extract_hollow_rectangles(obj)

            return all_result_ids

        # ========================================
        # 結合・削除操作
        # ========================================

        elif command_name == "MERGE":
            array_name = args[0] if args else None

            if array_name and isinstance(array_name, list):
                # 配列からNoneを除外
                array_name = self._filter_none_from_array(array_name)
                logger.info(f"MERGE: 入力配列からNoneを除外後 {len(array_name)}個のオブジェクト")

                merged_id = self.executor_core._merge_objects(array_name)
                return merged_id  # 結合後のオブジェクトIDを返す
            return None

        # ========================================
        # 配列操作（追加）
        # ========================================

        # ========================================
        # 情報取得（追加）
        # ========================================


        # ========================================
        # 情報取得コマンド（さらに追加）
        # ========================================

        elif command_name == "GET_X":
            obj_id = args[0] if args else None
            if obj_id:
                obj = self.executor_core._find_object_by_id(obj_id)
                if obj:
                    # bboxは(min_x, min_y, max_x, max_y)形式（X座標、Y座標の順）
                    # GET_Xはbbox_left（min_x、X座標）を返す
                    min_x, min_y, max_x, max_y = obj.bbox
                    return min_x
            return 0

        elif command_name == "GET_Y":
            obj_id = args[0] if args else None
            if obj_id:
                obj = self.executor_core._find_object_by_id(obj_id)
                if obj:
                    # bboxは(min_x, min_y, max_x, max_y)形式（X座標、Y座標の順）
                    # GET_Yはbbox_top（min_y、Y座標）を返す
                    min_x, min_y, max_x, max_y = obj.bbox
                    return min_y
            return 0

        elif command_name == "GET_ASPECT_RATIO":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_aspect_ratio(obj_id)
            return 100  # 1.0 * 100

        elif command_name == "GET_DENSITY":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_density(obj_id)
            return 0  # 0.0 * 100

        elif command_name == "GET_CENTER_X":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_center_x(obj_id)
            return 0

        elif command_name == "GET_CENTER_Y":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_center_y(obj_id)
            return 0

        elif command_name == "GET_MAX_X":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_max_x(obj_id)
            return 0

        elif command_name == "GET_MAX_Y":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_max_y(obj_id)
            return 0

        elif command_name == "GET_CENTROID":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_centroid(obj_id)
            return "C"

        elif command_name == "GET_DIRECTION":
            obj1_id = args[0] if len(args) > 0 else None
            obj2_id = args[1] if len(args) > 1 else None
            if obj1_id and obj2_id:
                return self.executor_core._get_direction(obj1_id, obj2_id)
            return "C"  # 引数が不正な場合は中央を返す

        elif command_name == "GET_NEAREST":
            obj_id = args[0] if len(args) > 0 else None
            candidate_ids = args[1] if len(args) > 1 else []

            if obj_id is None:
                return None

            # 候補が配列でない場合、空リストとして扱う
            if not isinstance(candidate_ids, list):
                return None

            # Noneを除外
            candidate_ids = [cid for cid in candidate_ids if cid is not None]

            if not candidate_ids:
                return None

            return self.executor_core._get_nearest(obj_id, candidate_ids)

        elif command_name == "TILE":
            obj_id = args[0] if len(args) > 0 else None
            count_x = args[1] if len(args) > 1 else 1
            count_y = args[2] if len(args) > 2 else 1

            if obj_id is None:
                return []

            # 引数の型チェック
            if not isinstance(count_x, int) or not isinstance(count_y, int):
                return []

            if count_x <= 0 or count_y <= 0:
                return []

            result = self.executor_core._tile(obj_id, count_x, count_y)

            # 配列に格納されているオブジェクトの総数・総ピクセル数を検証
            if isinstance(result, list):
                self._enforce_array_limits(result, "TILE結果")

            return result

        # FILTER_BY_COLOR, FILTER_BY_PIXEL_COUNT は廃止
        # → FILTER(objects, "EQUAL(GET_COLOR, 1)") で代替

        # FILTER_BY_AREA は廃止
        # → IS_INSIDE で代替


        elif command_name == "GET_INPUT_GRID_SIZE":
            size = self.executor_core.grid_context.get_input_grid_size()
            return list(size) if size else [0, 0]

        # GET_PIXEL_COLORは廃止されました
        # 理由: オブジェクトベース設計に反するグリッド直接アクセス
        # 代替: GET_BACKGROUND_COLOR(背景色)、オブジェクトベースの処理



        elif command_name == "GET_LINE_TYPE":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_line_direction(obj_id)
            return "C"

        elif command_name == "GET_RECTANGLE_TYPE":
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_rectangle_type(obj_id)
            return "none"



        elif command_name == "IS_INSIDE":
            obj_input = args[0] if len(args) > 0 else None
            x = args[1] if len(args) > 1 else 0
            y = args[2] if len(args) > 2 else 0
            width = args[3] if len(args) > 3 else 1
            height = args[4] if len(args) > 4 else 1

            if obj_input is not None:
                # 左上の座標と幅・高さから右下の座標を計算
                x2 = x + width - 1
                y2 = y + height - 1
                return self.executor_core._is_in_area(obj_input, x, y, x2, y2)
            return False

        elif command_name in {"COUNT_HOLES", "GET_HOLE_COUNT"}:
            obj_id = args[0] if args else None
            if obj_id:
                return self.executor_core._get_hole_count(obj_id)
            return 0

        elif command_name == "GET_SYMMETRY_SCORE":
            """対称性スコアを0-100の整数で返す"""
            if len(args) < 2:
                raise ValueError("GET_SYMMETRY_SCORE: 2つの引数（object_id, axis）が必要です")

            obj_id = args[0]
            axis = args[1]

            # 軸の検証
            if axis not in ["X", "Y"]:
                raise ValueError(f"GET_SYMMETRY_SCORE: axisは'X'または'Y'である必要があります（実際: {axis}）")

            if obj_id:
                score_float = self.executor_core._get_symmetry_score(obj_id, axis)
                # 0.0-1.0 を 0-100 の整数に変換
                return int(round(score_float * 100))
            return 0

        elif command_name == "GET_DISTANCE":
            if len(args) < 2:
                raise ValueError(f"GET_DISTANCE: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._get_distance_between_objects(obj1_id, obj2_id)

        elif command_name == "GET_X_DISTANCE":
            if len(args) < 2:
                raise ValueError(f"GET_X_DISTANCE: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._get_x_distance(obj1_id, obj2_id)

        elif command_name == "GET_Y_DISTANCE":
            if len(args) < 2:
                raise ValueError(f"GET_Y_DISTANCE: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._get_y_distance(obj1_id, obj2_id)

        elif command_name == "COUNT_ADJACENT":
            if len(args) < 2:
                raise ValueError(f"COUNT_ADJACENT: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._get_adjacent_edge_count(obj1_id, obj2_id)

        elif command_name == "IS_SAME_SHAPE":
            if len(args) < 2:
                raise ValueError(f"IS_SAME_SHAPE: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._has_same_shape(obj1_id, obj2_id)

        elif command_name == "IS_SAME_STRUCT":
            if len(args) < 2:
                raise ValueError(f"IS_SAME_STRUCT: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._has_same_color_structure(obj1_id, obj2_id)

        elif command_name == "IS_IDENTICAL":
            if len(args) < 2:
                raise ValueError(f"IS_IDENTICAL: 2つの引数（object_id1, object_id2）が必要です（実際: {len(args)}個）")

            obj1_id = args[0]
            obj2_id = args[1]

            return self.executor_core._has_same_shape_and_color(obj1_id, obj2_id)

        elif command_name == "FIT_SHAPE":
            obj1_id = args[0] if len(args) > 0 else None
            obj2_id = args[1] if len(args) > 1 else None

            if obj1_id and obj2_id:
                return self.executor_core._fit_shape(obj1_id, obj2_id)
            return None

        elif command_name == "FIT_SHAPE_COLOR":
            obj1_id = args[0] if len(args) > 0 else None
            obj2_id = args[1] if len(args) > 1 else None

            if obj1_id and obj2_id:
                return self.executor_core._fit_shape_color(obj1_id, obj2_id)
            return None

        elif command_name == "FIT_ADJACENT":
            obj1_id = args[0] if len(args) > 0 else None
            obj2_id = args[1] if len(args) > 1 else None

            if obj1_id and obj2_id:
                return self.executor_core._fit_adjacent(obj1_id, obj2_id)
            return None


        # ========================================
        # 未実装のコマンド
        # ========================================

        else:
            raise ValueError(f"未知のコマンド: {command_name}")

    def evaluate_expression(self, expr: Expression) -> Any:
        """
        式を評価

        Args:
            expr: Expression

        Returns:
            評価結果
        """
        if isinstance(expr, Literal):
            return expr.value

        elif isinstance(expr, Identifier):
            var_name = expr.name

            # $objの場合は現在のオブジェクトIDを返す（通常はPlaceholderとして処理されるが、念のため）
            if var_name == '$obj':
                if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                    return self.current_object_id
                else:
                    raise ValueError(f"未定義のプレースホルダー: {var_name}")

            # ローカル変数を優先
            if var_name in self.variables:
                return self.variables[var_name]

            # ExecutorCoreの変数もチェック
            if var_name in self.executor_core.execution_context.get('variables', {}):
                return self.executor_core.execution_context['variables'][var_name]

            raise ValueError(f"未定義の変数: {var_name}")

        elif isinstance(expr, FunctionCall):
            return self.execute_function_call(expr)

        elif isinstance(expr, BinaryOp):
            return self.evaluate_binary_op(expr)

        elif isinstance(expr, UnaryOp):
            return self.evaluate_unary_op(expr)

        elif isinstance(expr, Placeholder):
            # プレースホルダーの値を取得
            var_name = expr.name
            # $objプレースホルダーの場合（nameは"obj"として格納されている）
            if var_name == 'obj':
                if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                    return self.current_object_id
                else:
                    raise ValueError(f"未定義のプレースホルダー: ${var_name}")
            if var_name in self.variables:
                return self.variables[var_name]
            raise ValueError(f"未定義のプレースホルダー: ${var_name}")

        elif isinstance(expr, ListLiteral):
            # 各要素の評価前に$objを設定
            if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                self.variables['$obj'] = self.current_object_id
            return [self.evaluate_expression(elem) for elem in expr.elements]

        elif isinstance(expr, IndexAccess):
            # 各要素の評価前に$objを設定
            if hasattr(self, 'current_object_id') and self.current_object_id is not None:
                self.variables['$obj'] = self.current_object_id
            target = self.evaluate_expression(expr.target)
            index = self.evaluate_expression(expr.index)
            if isinstance(target, list) and isinstance(index, int):
                if 0 <= index < len(target):
                    return target[index]
                else:
                    # 空のリストへのアクセスの場合は、Noneを返して処理を継続
                    if len(target) == 0:
                        # 空のリストにアクセスした場合はNoneを返す（エラーを発生させない）
                        return None
                    raise IndexError(f"Index {index} out of range for list of length {len(target)}")
            return None

        elif isinstance(expr, AttributeAccess):
            target = self.evaluate_expression(expr.target)
            attr_name = expr.attribute

            if hasattr(target, attr_name):
                return getattr(target, attr_name)
            else:
                raise AttributeError(f"'{type(target).__name__}' has no attribute '{attr_name}'")

        else:
            raise ValueError(f"未実装の式型: {type(expr)}")

    def evaluate_binary_op(self, expr: BinaryOp) -> Any:
        """
        二項演算を評価

        Args:
            expr: BinaryOpノード

        Returns:
            演算結果
        """
        left = self.evaluate_expression(expr.left)
        right = self.evaluate_expression(expr.right)
        op = expr.operator

        # 算術演算子
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right if right != 0 else 0

        # 比較演算子
        elif op == '>' or op == 'GT':
            return left > right
        elif op == '<' or op == 'LT':
            return left < right
        elif op == '>=' or op == 'GTE':
            return left >= right
        elif op == '<=' or op == 'LTE':
            return left <= right
        elif op == '==' or op == 'EQ':
            return left == right
        elif op == '!=' or op == 'NEQ':
            return left != right

        # 論理演算子
        elif op in ('and', 'AND'):
            return left and right
        elif op in ('or', 'OR'):
            return left or right

        else:
            raise ValueError(f"未実装の演算子: {op}")

    def evaluate_unary_op(self, expr: UnaryOp) -> Any:
        """
        単項演算を評価

        Args:
            expr: UnaryOpノード

        Returns:
            演算結果
        """
        operand = self.evaluate_expression(expr.operand)
        op = expr.operator

        if op == '-':
            return -operand
        else:
            raise ValueError(f"未実装の単項演算子: {op}")


def test_interpreter():
    """インタープリターのテスト"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.core_systems.executor.parsing.tokenizer import Tokenizer
    from src.core_systems.executor.parsing.parser import Parser

    print("Interpreter Test (basic functionality)\n")

    # モックのExecutorCore
    class MockExecutorCore:
        def __init__(self):
            self.execution_context = {
                'variables': {},
                'background_color': 0,
                'input_grid': None
            }
            self.object_manager = MockObjectManager()
            self.layer_system = MockLayerSystem()

        def _move_single(self, obj_id, dx, dy):
            print(f"  _move_single({obj_id}, {dx}, {dy})")

    class MockObjectManager:
        def __init__(self):
            self.objects = {}

        def get_all_objects(self):
            return {'obj_0': None, 'obj_1': None}

    class MockLayerSystem:
        def create_background_layer(self, grid, color):
            print(f"  create_background_layer(grid, {color})")

    executor = MockExecutorCore()
    interpreter = Interpreter(executor)
    tokenizer = Tokenizer()
    parser_class = Parser

    # テスト1: 変数代入と情報取得
    print("Test 1: 変数代入と情報取得")
    code1 = '''objects = GET_ALL_OBJECTS()
count = GET_ARRAY_LENGTH(objects)'''
    tokens1 = tokenizer.tokenize(code1)
    parser1 = parser_class(tokens1)
    ast1 = parser1.parse()
    interpreter.execute(ast1)
    print(f"  variables: {interpreter.variables}")
    print()

    # テスト2: 関数呼び出し
    print("Test 2: 関数呼び出し")
    code2 = '''SET_OBJECT_TYPE("single_color_4way")
CREATE_BACKGROUND_LAYER(0)'''
    tokens2 = tokenizer.tokenize(code2)
    parser2 = parser_class(tokens2)
    ast2 = parser2.parse()
    interpreter.execute(ast2)
    print()

    # テスト3: 算術演算
    print("Test 3: 算術演算")
    code3 = '''result = ADD(10, 20)
product = MULTIPLY(result, 2)'''
    tokens3 = tokenizer.tokenize(code3)
    parser3 = parser_class(tokens3)
    ast3 = parser3.parse()
    interpreter.execute(ast3)
    print(f"  variables: {interpreter.variables}")
    print()


if __name__ == '__main__':
    test_interpreter()
