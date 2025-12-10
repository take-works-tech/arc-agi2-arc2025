"""
拡張プログラム検証器

型チェック、操作空間チェック、境界チェックを追加
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict

from src.hybrid_system.utils.validation.program_validator import (
    ProgramValidator,
    ValidationResult
)

# Taskクラスは型ヒントのみで使用（実際のインポートは必要に応じて）
try:
    from src.hybrid_system.core.data_structures.task import Task
except ImportError:
    Task = None  # 型ヒントのみの使用を許可


@dataclass
class EnhancedValidationResult(ValidationResult):
    """拡張検証結果"""
    type_check_errors: List[str] = field(default_factory=list)
    operation_space_errors: List[str] = field(default_factory=list)
    boundary_check_errors: List[str] = field(default_factory=list)
    validation_penalty: float = 0.0  # 検証ペナルティ（0.0-1.0）


class EnhancedProgramValidator(ProgramValidator):
    """拡張プログラム検証器"""

    def __init__(self, grid_size: Optional[Tuple[int, int]] = None):
        """
        初期化

        Args:
            grid_size: グリッドサイズ (height, width)（境界チェック用）
        """
        super().__init__()
        self.grid_size = grid_size  # (height, width)

        # コマンドの型シグネチャ
        self.command_signatures = {
            'FILTER': {'args': ['array', 'condition'], 'returns': 'array'},
            'MAP': {'args': ['array', 'function'], 'returns': 'array'},
            'SET_COLOR': {'args': ['object', 'color'], 'returns': 'object'},
            'MOVE': {'args': ['object', 'dx', 'dy'], 'returns': 'object'},
            'ROTATE': {'args': ['object', 'angle'], 'returns': 'object'},
            'GET_COLOR': {'args': ['object'], 'returns': 'int'},
            'GET_X': {'args': ['object'], 'returns': 'int'},
            'GET_Y': {'args': ['object'], 'returns': 'int'},
            'GET_WIDTH': {'args': ['object'], 'returns': 'int'},
            'GET_HEIGHT': {'args': ['object'], 'returns': 'int'},
            'LEN': {'args': ['array'], 'returns': 'int'},
            'SELECT_LARGEST': {'args': ['array'], 'returns': 'object'},
            'SELECT_SMALLEST': {'args': ['array'], 'returns': 'object'},
        }

        # 単一オブジェクトを返すコマンド
        self.single_object_commands = {
            'SELECT_LARGEST', 'SELECT_SMALLEST', 'GET_OBJECT', 'GET_ALL_OBJECTS'
        }

        # 複数オブジェクトを期待するコマンド
        self.multi_object_commands = {
            'MAP', 'FILTER', 'SORT_BY', 'CONCAT'
        }

    def validate_program_enhanced(
        self,
        program: str,
        task: Optional[Task] = None
    ) -> EnhancedValidationResult:
        """
        拡張プログラム検証

        Args:
            program: 検証するプログラム
            task: タスク（オプション）

        Returns:
            EnhancedValidationResult: 拡張検証結果
        """
        # 基本検証
        base_result = super().validate_program(program, task)

        # 拡張検証
        type_result = self._validate_type_consistency(program)
        operation_result = self._validate_operation_space(program)
        boundary_result = self._validate_boundaries(program, task)

        # 結果を統合
        all_errors = (
            base_result.errors +
            type_result['errors'] +
            operation_result['errors'] +
            boundary_result['errors']
        )
        all_warnings = (
            base_result.warnings +
            type_result['warnings'] +
            operation_result['warnings'] +
            boundary_result['warnings']
        )

        # ペナルティを計算
        penalty = self._calculate_validation_penalty(
            type_result, operation_result, boundary_result
        )

        return EnhancedValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            info={
                **base_result.info,
                'type_check': type_result['info'],
                'operation_space': operation_result['info'],
                'boundary_check': boundary_result['info'],
            },
            type_check_errors=type_result['errors'],
            operation_space_errors=operation_result['errors'],
            boundary_check_errors=boundary_result['errors'],
            validation_penalty=penalty
        )

    def _validate_type_consistency(self, program: str) -> Dict[str, Any]:
        """
        型一貫性チェック

        例: `filter_objects(color)`の後に`recolor_by_relation()`が続くなど
        """
        errors = []
        warnings = []
        info = {}

        lines = program.split('\n')
        variable_types = {}  # 変数名 -> 型

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 変数代入を検出
            assignment_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)', line)
            if assignment_match:
                var_name = assignment_match.group(1)
                expression = assignment_match.group(2)

                # 式の型を推論
                inferred_type = self._infer_expression_type(expression, variable_types)

                # 既存の変数型と矛盾がないかチェック
                if var_name in variable_types:
                    if variable_types[var_name] != inferred_type:
                        errors.append(
                            f"行 {i}: 変数 '{var_name}' の型が矛盾しています。"
                            f"既存: {variable_types[var_name]}, 新規: {inferred_type}"
                        )
                else:
                    variable_types[var_name] = inferred_type

                # コマンド呼び出しの型チェック
                command_match = re.search(r'\b([A-Z_]+)\s*\(', line)
                if command_match:
                    command_name = command_match.group(1)
                    if command_name in self.command_signatures:
                        sig = self.command_signatures[command_name]
                        # 引数の型チェック（本格実装）
                        args = self._extract_arguments(line, command_name)

                        # 引数の数のチェック
                        required_args = len([a for a in sig['args'] if not a.endswith('?')])
                        if len(args) < required_args:
                            errors.append(
                                f"行 {i}: コマンド '{command_name}' の引数が不足しています。"
                                f"期待: {required_args}以上, 実際: {len(args)}"
                            )
                        elif len(args) > len(sig['args']):
                            warnings.append(
                                f"行 {i}: コマンド '{command_name}' の引数が多すぎます。"
                                f"期待: {len(sig['args'])}, 実際: {len(args)}"
                            )

                        # 引数の型チェック
                        for j, (arg, expected_type) in enumerate(zip(args, sig['args'])):
                            # オプショナル引数のチェック（'?'で終わる）
                            if expected_type.endswith('?'):
                                expected_type = expected_type[:-1]

                            arg_type = self._infer_argument_type(arg, variable_types)
                            if arg_type != 'unknown' and arg_type != expected_type:
                                # 型の互換性チェック
                                if not self._is_type_compatible(arg_type, expected_type):
                                    warnings.append(
                                        f"行 {i}: コマンド '{command_name}' の引数 {j+1} の型が一致しません。"
                                        f"期待: {expected_type}, 実際: {arg_type}"
                                    )

        info['variable_types'] = variable_types
        info['type_errors_count'] = len(errors)

        return {
            'errors': errors,
            'warnings': warnings,
            'info': info
        }

    def _infer_expression_type(
        self,
        expression: str,
        variable_types: Dict[str, str]
    ) -> str:
        """式の型を推論"""
        # コマンド呼び出し
        command_match = re.search(r'\b([A-Z_]+)\s*\(', expression)
        if command_match:
            command_name = command_match.group(1)
            if command_name in self.command_signatures:
                return self.command_signatures[command_name]['returns']

        # 変数参照
        var_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
        if var_match:
            var_name = var_match.group(1)
            if var_name in variable_types:
                return variable_types[var_name]

        # リテラル
        if expression.strip().isdigit():
            return 'int'
        if expression.strip().startswith('['):
            return 'array'

        return 'unknown'

    def _infer_argument_type(self, arg: str, variable_types: Dict[str, str]) -> str:
        """
        引数の型を推論（本格実装）

        Args:
            arg: 引数文字列
            variable_types: 変数型の辞書

        Returns:
            推論された型
        """
        arg = arg.strip()

        # 数値リテラル
        if arg.isdigit() or (arg.startswith('-') and arg[1:].isdigit()):
            return 'int'

        # 浮動小数点リテラル
        try:
            float(arg)
            return 'float'
        except ValueError:
            pass

        # 配列リテラル
        if arg.startswith('[') and arg.endswith(']'):
            return 'array'

        # 文字列リテラル
        if (arg.startswith('"') and arg.endswith('"')) or \
           (arg.startswith("'") and arg.endswith("'")):
            return 'string'

        # 関数呼び出し
        func_match = re.search(r'\b([A-Z_]+)\s*\(', arg)
        if func_match:
            func_name = func_match.group(1)
            if func_name in self.command_signatures:
                return self.command_signatures[func_name]['returns']

        # 変数参照
        var_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', arg)
        if var_match:
            var_name = var_match.group(1)
            if var_name in variable_types:
                return variable_types[var_name]

        # プレースホルダー
        if arg.startswith('$'):
            return 'object'

        return 'unknown'

    def _is_type_compatible(self, actual_type: str, expected_type: str) -> bool:
        """
        型の互換性をチェック

        Args:
            actual_type: 実際の型
            expected_type: 期待される型

        Returns:
            互換性があるかどうか
        """
        # 完全一致
        if actual_type == expected_type:
            return True

        # 数値型の互換性
        if actual_type in ('int', 'float') and expected_type in ('int', 'float'):
            return True

        # object型は多くの型と互換性がある
        if actual_type == 'object' and expected_type in ('object', 'array'):
            return True

        # array型はobject配列として扱える
        if actual_type == 'array' and expected_type == 'array':
            return True

        return False

    def _extract_arguments(self, line: str, command_name: str) -> List[str]:
        """
        コマンドの引数を抽出（本格実装）

        ネストした括弧、関数呼び出し、配列リテラルなどを正しく処理
        """
        # コマンド呼び出しの開始位置を検索
        pattern = rf'\b{re.escape(command_name)}\s*\('
        match = re.search(pattern, line)
        if not match:
            return []

        # 括弧の開始位置
        start_pos = match.end() - 1  # '('の位置
        args_str = line[start_pos + 1:]  # '('の後の文字列

        # 括弧のネストを考慮して引数を抽出
        args = []
        current_arg = ""
        depth = 0  # 括弧のネスト深度
        in_string = False  # 文字列リテラル内かどうか
        string_char = None  # 文字列の開始文字（' または "）

        i = 0
        while i < len(args_str):
            char = args_str[i]

            # 文字列リテラルの処理
            if char in ("'", '"') and (i == 0 or args_str[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current_arg += char
                i += 1
                continue

            if in_string:
                current_arg += char
                i += 1
                continue

            # 括弧の処理
            if char == '(':
                depth += 1
                current_arg += char
            elif char == ')':
                if depth == 0:
                    # 引数リストの終了
                    if current_arg.strip():
                        args.append(current_arg.strip())
                    break
                depth -= 1
                current_arg += char
            elif char == '[':
                # 配列リテラルの開始
                depth += 1
                current_arg += char
            elif char == ']':
                # 配列リテラルの終了
                if depth > 0:
                    depth -= 1
                current_arg += char
            elif char == ',' and depth == 0:
                # 引数の区切り（最上位レベルでのみ）
                if current_arg.strip():
                    args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char

            i += 1

        # 最後の引数が残っている場合
        if current_arg.strip() and depth == 0:
            args.append(current_arg.strip())

        return args

    def _validate_operation_space(self, program: str) -> Dict[str, Any]:
        """
        操作空間チェック

        例: `select_largest()`の結果、オブジェクトが1つだけになる問題で、
        `map_objects()`が続くのは無効
        """
        errors = []
        warnings = []
        info = {}

        lines = program.split('\n')
        variable_cardinality = {}  # 変数名 -> 'single' | 'multiple' | 'unknown'

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 変数代入を検出
            assignment_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)', line)
            if assignment_match:
                var_name = assignment_match.group(1)
                expression = assignment_match.group(2)

                # 式のカーディナリティを推論
                inferred_cardinality = self._infer_cardinality(expression, variable_cardinality)
                variable_cardinality[var_name] = inferred_cardinality

                # 次の行で使用されるコマンドをチェック
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line and not next_line.startswith('#'):
                        # 複数オブジェクトを期待するコマンドが単一オブジェクトに適用されているかチェック
                        for cmd in self.multi_object_commands:
                            if cmd in next_line and var_name in next_line:
                                if inferred_cardinality == 'single':
                                    errors.append(
                                        f"行 {i+1}: コマンド '{cmd}' は複数オブジェクトを期待しますが、"
                                        f"変数 '{var_name}' は単一オブジェクトです"
                                    )

        info['variable_cardinality'] = variable_cardinality
        info['operation_space_errors_count'] = len(errors)

        return {
            'errors': errors,
            'warnings': warnings,
            'info': info
        }

    def _infer_cardinality(
        self,
        expression: str,
        variable_cardinality: Dict[str, str]
    ) -> str:
        """式のカーディナリティ（単一/複数）を推論"""
        # コマンド呼び出し
        command_match = re.search(r'\b([A-Z_]+)\s*\(', expression)
        if command_match:
            command_name = command_match.group(1)
            if command_name in self.single_object_commands:
                return 'single'
            if command_name in self.multi_object_commands:
                return 'multiple'
            if command_name == 'GET_ALL_OBJECTS':
                return 'multiple'
            if command_name == 'LEN':
                return 'single'  # LENは整数を返す

        # 変数参照
        var_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
        if var_match:
            var_name = var_match.group(1)
            if var_name in variable_cardinality:
                return variable_cardinality[var_name]

        # 配列リテラル
        if expression.strip().startswith('['):
            return 'multiple'

        return 'unknown'

    def _validate_boundaries(
        self,
        program: str,
        task: Optional[Task]
    ) -> Dict[str, Any]:
        """
        境界チェック

        移動・回転操作がグリッドの境界を越えないか（事前に静的解析）
        """
        errors = []
        warnings = []
        info = {}

        if not self.grid_size and task:
            # タスクからグリッドサイズを取得
            if task.train and len(task.train) > 0:
                input_grid = task.train[0].input
                if input_grid:
                    self.grid_size = (len(input_grid), len(input_grid[0]) if input_grid else 0)

        if not self.grid_size:
            warnings.append("グリッドサイズが不明なため、境界チェックをスキップします")
            return {
                'errors': errors,
                'warnings': warnings,
                'info': info
            }

        height, width = self.grid_size

        lines = program.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # MOVEコマンドの境界チェック
            move_match = re.search(r'\bMOVE\s*\([^,]+,\s*([^,]+),\s*([^)]+)\)', line)
            if move_match:
                dx_str = move_match.group(1).strip()
                dy_str = move_match.group(2).strip()

                # 数値リテラルの場合のみチェック
                try:
                    dx = int(dx_str)
                    dy = int(dy_str)

                    # 移動後の位置が境界を越える可能性をチェック
                    # （実際のオブジェクト位置が不明なため、警告のみ）
                    if abs(dx) > width or abs(dy) > height:
                        warnings.append(
                            f"行 {i}: MOVE操作の移動量 ({dx}, {dy}) が大きすぎる可能性があります。"
                            f"グリッドサイズ: ({height}, {width})"
                        )
                except ValueError:
                    # 変数の場合はチェックできない
                    pass

            # TELEPORTコマンドの境界チェック
            teleport_match = re.search(r'\bTELEPORT\s*\([^,]+,\s*([^,]+),\s*([^)]+)\)', line)
            if teleport_match:
                x_str = teleport_match.group(1).strip()
                y_str = teleport_match.group(2).strip()

                try:
                    x = int(x_str)
                    y = int(y_str)

                    if x < 0 or x >= width or y < 0 or y >= height:
                        errors.append(
                            f"行 {i}: TELEPORT操作の座標 ({x}, {y}) がグリッドの境界外です。"
                            f"グリッドサイズ: ({height}, {width})"
                        )
                except ValueError:
                    pass

            # ROTATEコマンドの境界チェック
            # 回転後のオブジェクトが境界を越える可能性をチェック
            if 'ROTATE' in line:
                # 回転後のサイズが大きくなる可能性があるため、警告のみ
                warnings.append(
                    f"行 {i}: ROTATE操作はオブジェクトが境界を越える可能性があります"
                )

        info['grid_size'] = self.grid_size
        info['boundary_errors_count'] = len(errors)
        info['boundary_warnings_count'] = len(warnings)

        return {
            'errors': errors,
            'warnings': warnings,
            'info': info
        }

    def _calculate_validation_penalty(
        self,
        type_result: Dict[str, Any],
        operation_result: Dict[str, Any],
        boundary_result: Dict[str, Any]
    ) -> float:
        """
        検証ペナルティを計算（0.0-1.0）

        Args:
            type_result: 型チェック結果
            operation_result: 操作空間チェック結果
            boundary_result: 境界チェック結果

        Returns:
            float: ペナルティ（0.0-1.0、高いほど悪い）
        """
        penalty = 0.0

        # 型エラー: 0.3 per error
        type_errors = len(type_result['errors'])
        penalty += min(type_errors * 0.3, 1.0)

        # 操作空間エラー: 0.4 per error
        operation_errors = len(operation_result['errors'])
        penalty += min(operation_errors * 0.4, 1.0)

        # 境界エラー: 0.5 per error
        boundary_errors = len(boundary_result['errors'])
        penalty += min(boundary_errors * 0.5, 1.0)

        # 警告: 0.1 per warning（最大0.3）
        warnings = (
            len(type_result['warnings']) +
            len(operation_result['warnings']) +
            len(boundary_result['warnings'])
        )
        penalty += min(warnings * 0.1, 0.3)

        return min(penalty, 1.0)
