"""
プログラム検証器

DSLプログラムの構文チェック、型チェック、実行前の検証、実行時のデバッグ機能を提供
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict

from src.hybrid_system.core.data_structures.task import Task


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class ProgramValidator:
    """プログラム検証器"""

    # DSLコマンドの定義（主要なコマンドのみ）
    DSL_COMMANDS = {
        # オブジェクト取得
        'GET_ALL_OBJECTS', 'GET_OBJECTS', 'GET_OBJECT',
        # オブジェクト操作
        'SET_COLOR', 'MOVE', 'ROTATE', 'SCALE', 'FLIP', 'ALIGN',
        # オブジェクト情報
        'GET_COLOR', 'GET_X', 'GET_Y', 'GET_WIDTH', 'GET_HEIGHT',
        'GET_ASPECT_RATIO', 'GET_DENSITY', 'GET_CENTROID',
        'GET_CENTER_X', 'GET_CENTER_Y', 'GET_MAX_X', 'GET_MAX_Y',
        'GET_DIRECTION', 'GET_NEAREST', 'GET_DISTANCE',
        # 配列操作
        'FILTER', 'MAP', 'REDUCE', 'SORT', 'REVERSE',
        # 生成操作
        'CREATE_LINE', 'CREATE_RECT', 'MERGE', 'TILE',
        # 算術演算
        'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'MOD',
        # 比較演算
        'EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS', 'GREATER_EQUAL', 'LESS_EQUAL',
        # 論理演算
        'AND', 'OR', 'NOT',
        # 制御構造
        'IF', 'THEN', 'ELSE', 'END', 'FOR', 'DO', 'WHILE',
        # その他
        'LEN', 'APPEND', 'RENDER_GRID',  # FILL_GRIDは存在しないコマンドのため削除
    }

    # 予約語
    RESERVED_WORDS = {
        'IF', 'THEN', 'ELSE', 'END', 'FOR', 'DO', 'WHILE',
        'lambda', 'LEN', 'APPEND'
    }

    def __init__(self):
        """初期化"""
        self.variable_registry: Dict[str, str] = {}  # 変数名 -> 型
        self.function_registry: Dict[str, Dict[str, Any]] = {}  # 関数名 -> シグネチャ

    def validate_program(self, program: str, task: Optional[Task] = None) -> ValidationResult:
        """
        プログラムを検証

        Args:
            program: 検証するプログラム
            task: タスク（オプション、実行前検証に使用）

        Returns:
            ValidationResult: 検証結果
        """
        errors = []
        warnings = []
        info = {}

        # 1. 構文チェック
        syntax_result = self._validate_syntax(program)
        errors.extend(syntax_result.errors)
        warnings.extend(syntax_result.warnings)
        info['syntax'] = syntax_result.info

        # 2. 型チェック
        type_result = self._validate_types(program)
        errors.extend(type_result.errors)
        warnings.extend(type_result.warnings)
        info['types'] = type_result.info

        # 3. 実行前の検証
        if task:
            pre_execution_result = self._validate_pre_execution(program, task)
            errors.extend(pre_execution_result.errors)
            warnings.extend(pre_execution_result.warnings)
            info['pre_execution'] = pre_execution_result.info

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info
        )

    def _validate_syntax(self, program: str) -> ValidationResult:
        """DSL構文の検証"""
        errors = []
        warnings = []
        info = {}

        if not program or not program.strip():
            errors.append("プログラムが空です")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, info=info)

        # 括弧の対応チェック
        if program.count('(') != program.count(')'):
            errors.append("括弧 '(' と ')' の対応が取れていません")
        if program.count('[') != program.count(']'):
            errors.append("角括弧 '[' と ']' の対応が取れていません")
        if program.count('{') != program.count('}'):
            errors.append("波括弧 '{' と '}' の対応が取れていません")

        # 制御構造のチェック
        if_count = len(re.findall(r'\bIF\b', program))
        then_count = len(re.findall(r'\bTHEN\b', program))
        end_count = len(re.findall(r'\bEND\b', program))

        if if_count != then_count:
            errors.append(f"IF ({if_count}) と THEN ({then_count}) の数が一致しません")
        if if_count > end_count:
            errors.append(f"IF ({if_count}) に対して END ({end_count}) が不足しています")

        # FORループのチェック
        for_count = len(re.findall(r'\bFOR\b', program))
        do_count = len(re.findall(r'\bDO\b', program))
        if for_count != do_count:
            errors.append(f"FOR ({for_count}) と DO ({do_count}) の数が一致しません")
        if for_count > end_count:
            errors.append(f"FOR ({for_count}) に対して END ({end_count}) が不足しています")

        # コマンドの存在チェック
        lines = program.split('\n')
        unknown_commands = []
        for i, line in enumerate(lines, 1):
            # コメント行をスキップ
            if line.strip().startswith('#'):
                continue
            # 変数代入をスキップ
            if '=' in line and not any(cmd in line for cmd in self.DSL_COMMANDS):
                continue
            # コマンドを抽出
            for cmd in self.DSL_COMMANDS:
                if re.search(rf'\b{cmd}\b', line):
                    break
            else:
                # コマンドが見つからない場合
                if line.strip() and not line.strip().startswith('#'):
                    # 変数名やリテラルだけの行はスキップ
                    if not re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*', line):
                        unknown_commands.append((i, line.strip()))

        if unknown_commands:
            warnings.append(f"未知のコマンドまたは構文が {len(unknown_commands)} 箇所で検出されました")
            info['unknown_commands'] = unknown_commands[:10]  # 最初の10個のみ

        # 情報収集
        info['line_count'] = len(lines)
        info['command_count'] = sum(1 for cmd in self.DSL_COMMANDS if cmd in program)
        info['if_count'] = if_count
        info['for_count'] = for_count

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, info=info)

    def _validate_types(self, program: str) -> ValidationResult:
        """型チェック"""
        errors = []
        warnings = []
        info = {}

        # 変数の使用状況を追跡
        variable_usage = defaultdict(list)
        variable_definitions = {}

        lines = program.split('\n')
        for i, line in enumerate(lines, 1):
            # 変数定義を検出
            match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*', line)
            if match:
                var_name = match.group(1)
                if var_name not in self.RESERVED_WORDS:
                    variable_definitions[var_name] = i

            # 変数使用を検出
            for var_name in variable_definitions.keys():
                if re.search(rf'\b{var_name}\b', line):
                    variable_usage[var_name].append(i)

        # 定義されていない変数の使用をチェック
        undefined_vars = []
        for i, line in enumerate(lines, 1):
            # 変数名パターンを抽出
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(var_pattern, line)
            for var_name in matches:
                if (var_name not in self.RESERVED_WORDS and
                    var_name not in variable_definitions and
                    var_name not in self.DSL_COMMANDS and
                    not var_name.isdigit()):
                    undefined_vars.append((i, var_name))

        if undefined_vars:
            # 重複を除去
            unique_undefined = list(set((line, var) for line, var in undefined_vars))
            for line_num, var_name in unique_undefined[:10]:  # 最初の10個のみ
                errors.append(f"行 {line_num}: 未定義の変数 '{var_name}' が使用されています")

        # 情報収集
        info['defined_variables'] = list(variable_definitions.keys())
        info['variable_count'] = len(variable_definitions)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, info=info)

    def _validate_pre_execution(self, program: str, task: Task) -> ValidationResult:
        """実行前の検証"""
        errors = []
        warnings = []
        info = {}

        # 必須コマンドのチェック
        if 'RENDER_GRID' not in program:
            warnings.append("RENDER_GRID が見つかりません。出力が生成されない可能性があります")

        # オブジェクト取得コマンドのチェック
        if 'GET_ALL_OBJECTS' not in program and 'GET_OBJECTS' not in program:
            warnings.append("オブジェクト取得コマンドが見つかりません")

        # タスクの訓練ペア数とプログラムの複雑度の関係
        if task.train:
            train_count = len(task.train)
            # プログラムの複雑度を測定（本格実装）
            # 複数の要因を考慮した複雑度計算
            if_count = program.count('IF ')
            for_count = program.count('FOR ')
            while_count = program.count('WHILE ')
            function_calls = len(re.findall(r'\b[A-Z_]+\s*\(', program))

            # ネスト深度を計算
            nest_depth = 0
            max_nest_depth = 0
            for line in program.split('\n'):
                stripped = line.strip()
                if stripped.startswith('IF ') or stripped.startswith('FOR ') or stripped.startswith('WHILE '):
                    nest_depth += 1
                    max_nest_depth = max(max_nest_depth, nest_depth)
                elif stripped == 'END':
                    nest_depth = max(0, nest_depth - 1)

            # 変数の数
            variable_count = len(re.findall(r'\b[a-z_][a-z0-9_]*\s*=', program))

            # 総合的な複雑度（重み付き）
            complexity = (
                if_count * 2 +
                for_count * 3 +
                while_count * 4 +
                function_calls * 1 +
                max_nest_depth * 5 +
                variable_count * 1
            )
            if complexity > train_count * 10:
                warnings.append(f"プログラムの複雑度 ({complexity}) が訓練ペア数 ({train_count}) に対して高すぎる可能性があります")

        # 情報収集
        info['has_render'] = 'RENDER_GRID' in program
        info['has_object_get'] = 'GET_ALL_OBJECTS' in program or 'GET_OBJECTS' in program

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, info=info)

    def generate_debug_trace(self, program: str, input_grid: Any) -> List[Dict[str, Any]]:
        """
        実行時のデバッグトレースを生成（本格実装）

        Args:
            program: プログラム
            input_grid: 入力グリッド

        Returns:
            デバッグトレース（各ステップの情報）
        """
        trace = []
        lines = program.split('\n')

        # 実際のプログラム実行を試みて、詳細なトレースを取得
        try:
            from src.core_systems.executor.core import ExecutorCore
            import numpy as np

            executor = ExecutorCore()
            input_array = np.array(input_grid, dtype=np.int64) if isinstance(input_grid, list) else input_grid

            # プログラムを実行して、実行コンテキストからトレースを取得
            output_grid, objects, execution_context = executor.execute_program(program, input_array)

            # 実行コンテキストからトレース情報を抽出
            if execution_context and 'results' in execution_context:
                results = execution_context['results']
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        trace.append({
                            'step': i + 1,
                            'command': result.get('command', 'unknown'),
                            'result_type': type(result).__name__,
                            'info': str(result)[:100]  # 最初の100文字
                        })

            # トレースが取得できなかった場合、プログラムの行ごとに解析
            if not trace:
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue

                    # 行の種類を判定
                    line_type = 'statement'
                    if 'IF ' in stripped:
                        line_type = 'conditional'
                    elif 'FOR ' in stripped:
                        line_type = 'loop'
                    elif '=' in stripped:
                        line_type = 'assignment'
                    elif stripped.endswith('END'):
                        line_type = 'block_end'

                    trace.append({
                        'step': i,
                        'line': stripped,
                        'line_type': line_type,
                        'line_number': i
                    })
        except Exception as e:
            # 実行に失敗した場合、プログラムの行ごとに解析
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue

                # 行の種類を判定
                line_type = 'statement'
                if 'IF ' in stripped:
                    line_type = 'conditional'
                elif 'FOR ' in stripped:
                    line_type = 'loop'
                elif '=' in stripped:
                    line_type = 'assignment'
                elif stripped.endswith('END'):
                    line_type = 'block_end'

                trace.append({
                    'step': i,
                    'line': stripped,
                    'line_type': line_type,
                    'line_number': i,
                    'error': str(e) if i == 1 else None  # 最初の行にエラーを記録
                })

        return trace
