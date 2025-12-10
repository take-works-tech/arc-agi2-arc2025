#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSLプログラムをテンプレート列(IRSequence)へ変換するトランスフォーマ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.hybrid_system.ir.structures import (
    IRSequence,
    TemplateStep,
    PostCondition,
    ArgumentKind,
    ArgumentSlot,
    ArgumentValue,
)
from src.core_systems.executor.operations.types import OperationType
from src.hybrid_system.ir.mapping import (
    resolve_command_template,
    resolve_argument_slots,
    CommandTemplate,
)


@dataclass
class RelabelTransformerConfig:
    """トランスフォーマ設定"""

    default_version: str = "v1"
    default_quality_label: str = "unverified"
    attach_execution_metadata: bool = True
    post_condition_predicate: str = "grid_equal"
    auto_default_arguments: Dict[str, Dict[str, str]] = field(default_factory=dict)


class RelabelTransformer:
    """DSLコードをテンプレート列へ変換"""

    DEFAULT_AUTO_DEFAULTS: Dict[str, Dict[str, str]] = {
        "FLIP": {"pivot": "center"},
        "MOVE": {"mode": "relative"},
        "SUBTRACT": {"mode": "difference"},
    }

    def __init__(self, config: Optional[RelabelTransformerConfig] = None):
        self.config = config or RelabelTransformerConfig()
        if not self.config.auto_default_arguments:
            self.config.auto_default_arguments = self.DEFAULT_AUTO_DEFAULTS.copy()
        else:
            # 正規化: コマンド名を大文字化
            normalized_defaults: Dict[str, Dict[str, str]] = {}
            for cmd, defaults in self.config.auto_default_arguments.items():
                normalized_defaults[cmd.upper()] = defaults
            for cmd, defaults in self.DEFAULT_AUTO_DEFAULTS.items():
                normalized_defaults.setdefault(cmd, defaults)
            self.config.auto_default_arguments = normalized_defaults

    def transform(
        self,
        program_code: str,
        task_id: str = "",
        sequence_metadata: Optional[Dict[str, Any]] = None,
    ) -> IRSequence:
        """
        DSLプログラムを解析し、テンプレート列に変換
        """
        ast_nodes = self._parse_program(program_code)
        steps: List[TemplateStep] = []
        diagnostics: List[Dict[str, Any]] = []

        for index, node in enumerate(ast_nodes):
            try:
                node_steps, node_diag = self._process_statement(node, index, loop_stack=[])
                steps.extend(node_steps)
                if node_diag:
                    diagnostics.append(node_diag)
            except Exception as exc:
                diagnostics.append(
                    {
                        "type": "error",
                        "message": f"ステートメント解析に失敗: {exc}",
                        "statement_index": index,
                    }
                )

        metadata = sequence_metadata.copy() if sequence_metadata else {}
        if diagnostics:
            metadata["diagnostics"] = diagnostics
        metadata["source"] = metadata.get("source", "dsl_relabel")

        sequence = IRSequence(
            steps=steps,
            task_id=task_id,
            version=self.config.default_version,
            quality_label=self.config.default_quality_label,
            metadata=metadata,
        )
        return sequence

    # ----------------------------------------------
    # 解析関連
    # ----------------------------------------------
    def _parse_program(self, program_code: str):
        """既存のTokenizer/Parserを用いてAST化"""
        from src.core_systems.executor.parsing.tokenizer import Tokenizer
        from src.core_systems.executor.parsing.parser import Parser

        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(program_code)
        parser = Parser(tokens)
        return parser.parse()

    def _process_statement(
        self,
        node,
        statement_index: int,
        loop_stack: Optional[List[Dict[str, Any]]] = None,
        condition_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[TemplateStep], Dict[str, Any]]:
        """
        単一ステートメントをTemplateStepへ変換

        Args:
            node: ASTノード
            statement_index: ステートメントインデックス
            loop_stack: ループスタック（内側のループほど後ろ）
            condition_stack: 条件スタック（外側の条件ほど前）
        """
        from src.core_systems.executor.parsing.parser import (
            Assignment,
            FunctionCall,
            ForLoop,
            WhileLoop,
            IfStatement,
        )

        loop_stack = list(loop_stack) if loop_stack else []
        condition_stack = list(condition_stack) if condition_stack else []
        diagnostics: Dict[str, Any] = {}

        if isinstance(node, Assignment):
            steps, diag = self._process_assignment(node, statement_index)
            self._attach_control_stacks(steps, loop_stack, condition_stack)
            return steps, diag
        if isinstance(node, FunctionCall):
            step = self._create_step_from_function_call(node, statement_index)
            self._attach_control_stacks([step], loop_stack, condition_stack)
            return [step], diagnostics
        if isinstance(node, (ForLoop, WhileLoop)):
            steps, loop_diag = self._process_loop(node, statement_index, loop_stack, condition_stack)
            diagnostics.update(loop_diag)
            return steps, diagnostics
        if isinstance(node, IfStatement):
            steps, diag = self._process_if(node, statement_index, loop_stack, condition_stack)
            diagnostics.update(diag)
            return steps, diagnostics

        diagnostics = {
            "type": "warning",
            "message": f"未対応ノードをスキップ: {type(node).__name__}",
            "statement_index": statement_index,
        }
        return [], diagnostics

    def _process_assignment(self, node, statement_index: int) -> Tuple[List[TemplateStep], Dict[str, Any]]:
        """
        代入ステートメントの処理。右辺がFunctionCallの場合のみテンプレート化。
        """
        from src.core_systems.executor.parsing.parser import FunctionCall

        diagnostics: Dict[str, Any] = {}
        steps: List[TemplateStep] = []

        if isinstance(node.expression, FunctionCall):
            step = self._create_step_from_function_call(node.expression, statement_index)
            # 代入先をメタデータに記録
            step.metadata["assignment_target"] = self._expression_to_string(node.variable)
            steps.append(step)
        else:
            step = self._create_direct_assignment_step(node, statement_index)
            steps.append(step)
            diagnostics = {
                "type": "info",
                "message": "直接代入をテンプレートとして記録",
                "statement_index": statement_index,
                "assignment_kind": step.metadata.get("assignment_kind"),
            }

        return steps, diagnostics

    def _create_direct_assignment_step(self, node, statement_index: int) -> TemplateStep:
        """
        FunctionCall 以外の代入ステートメントをテンプレート化
        """
        from src.core_systems.executor.parsing.parser import (
            Literal,
            Identifier,
            ListLiteral,
            Placeholder,
        )

        target_repr = self._expression_to_string(node.variable)
        expression_repr = self._expression_to_string(node.expression)

        if isinstance(node.expression, Placeholder):
            value_kind = ArgumentKind.PLACEHOLDER
            value_payload = None
            value_description = "プレースホルダ代入"
        elif isinstance(node.expression, Identifier):
            value_kind = ArgumentKind.REFERENCE
            value_payload = self._expression_to_string(node.expression)
            value_description = "識別子参照の代入"
        elif isinstance(node.expression, Literal):
            value_kind = ArgumentKind.LITERAL
            value_payload = node.expression.value
            value_description = "リテラル代入"
        elif isinstance(node.expression, ListLiteral):
            value_kind = ArgumentKind.LITERAL
            value_payload = [self._expression_to_value(elm) for elm in node.expression.elements]
            value_description = "リストリテラル代入"
        else:
            value_kind = ArgumentKind.DERIVED
            value_payload = self._expression_to_string(node.expression)
            value_description = "式の直接代入"

        assignment_kind = node.expression.__class__.__name__

        argument_slots = [
            ArgumentSlot(
                name="value",
                value=ArgumentValue(
                    kind=value_kind,
                    value=value_payload,
                    constraints={"source": "direct_assignment"},
                    description=value_description,
                ),
                value_type=assignment_kind,
                optional=False,
                notes="direct_assignment",
            )
        ]

        metadata = {
            "dsl_command": "DIRECT_ASSIGN",
            "normalized_command": "DIRECT_ASSIGN",
            "statement_index": statement_index,
            "assignment_target": target_repr,
            "assignment_expression": expression_repr,
            "assignment_kind": assignment_kind,
        }

        return TemplateStep(
            operation=OperationType.DIRECT_ASSIGN,
            target_binding="",
            argument_slots=argument_slots,
            post_conditions=[],
            metadata=metadata,
        )

    def _extract_iterable_from_count_expr(self, expr) -> Optional[str]:
        """FORループの反復対象を推定して文字列で返す"""
        if expr is None:
            return None

        from src.core_systems.executor.parsing.parser import FunctionCall

        if isinstance(expr, FunctionCall) and expr.name.upper() == "LEN" and expr.arguments:
            return self._expression_to_string(expr.arguments[0])
        return None

    def _build_loop_entry(self, node, statement_index: int) -> Dict[str, Any]:
        """ループ構造のメタ情報を構築"""
        entry: Dict[str, Any] = {
            "loop_id": f"{statement_index}",
            "loop_type": type(node).__name__,
            "loop_var": getattr(node, "loop_var", None) if hasattr(node, "loop_var") else None,
            "count_expression": None,
            "iterable": None,
            "condition_repr": None,
        }

        node_type = entry["loop_type"]
        if node_type == "ForLoop":
            count_expr = getattr(node, "count_expr", None)
            if count_expr is not None:
                entry["count_expression"] = self._expression_to_string(count_expr)
                iterable = self._extract_iterable_from_count_expr(count_expr)
                if iterable:
                    entry["iterable"] = iterable
        elif node_type == "WhileLoop":
            condition = getattr(node, "condition", None)
            entry["condition_repr"] = (
                self._expression_to_string(condition) if condition is not None else "True"
            )

        return entry

    def _attach_control_stacks(
        self,
        steps: List[TemplateStep],
        loop_stack: List[Dict[str, Any]],
        condition_stack: List[Dict[str, Any]],
    ) -> None:
        """ステップに制御構造スタック情報を付与"""
        for step in steps:
            # loop_stackを付与
            if loop_stack and "loop_stack" not in step.metadata:
                step.metadata["loop_stack"] = list(loop_stack)
                step.metadata["loop"] = loop_stack[-1]

            # condition_stackを付与
            if condition_stack and "condition_stack" not in step.metadata:
                step.metadata["condition_stack"] = list(condition_stack)
                # 最も内側の条件（最後の要素）をconditionとして設定
                if not step.metadata.get("condition"):
                    step.metadata["condition"] = condition_stack[-1]

    def _process_loop(
        self,
        node,
        statement_index: int,
        loop_stack: List[Dict[str, Any]],
        condition_stack: List[Dict[str, Any]],
    ) -> Tuple[List[TemplateStep], Dict[str, Any]]:
        """
        FOR / WHILE ループの処理。
        ループ内のFunctionCallを抽出し、繰り返し情報をメタデータに付与。
        """
        diagnostics: Dict[str, Any] = {
            "type": "info",
            "message": f"ループ構造を展開: {type(node).__name__}",
            "statement_index": statement_index,
        }

        steps: List[TemplateStep] = []
        body_statements = getattr(node, "body", [])
        loop_entry = self._build_loop_entry(node, statement_index)
        new_loop_stack = loop_stack + [loop_entry]

        for idx, stmt in enumerate(body_statements):
            stmt_steps, _ = self._process_statement(
                stmt,
                statement_index * 1000 + idx,
                loop_stack=new_loop_stack,
                condition_stack=condition_stack,
            )
            steps.extend(stmt_steps)

        return steps, diagnostics

    def _process_if(
        self,
        node,
        statement_index: int,
        loop_stack: List[Dict[str, Any]],
        condition_stack: List[Dict[str, Any]],
    ) -> Tuple[List[TemplateStep], Dict[str, Any]]:
        """
        IF 文を処理。then/else のFunctionCallを抽出し、分岐条件メタデータを付与。
        condition_stackに条件情報を追加して子ステートメントに伝播する。
        """
        diagnostics: Dict[str, Any] = {
            "type": "info",
            "message": "IF文を分岐テンプレートとして処理",
            "statement_index": statement_index,
        }
        steps: List[TemplateStep] = []
        condition_repr = self._expression_to_string(node.condition)

        for branch, branch_name in [(node.then_body, "then"), (node.else_body, "else")]:
            if not branch:
                continue

            # 新しい条件をcondition_stackに追加
            condition_entry = {
                "branch": branch_name,
                "condition_repr": condition_repr,
                "condition_id": f"{statement_index}_{condition_repr}",
            }
            new_condition_stack = condition_stack + [condition_entry]

            for idx, stmt in enumerate(branch):
                stmt_steps, _ = self._process_statement(
                    stmt,
                    statement_index * 1000 + idx,
                    loop_stack=loop_stack,
                    condition_stack=new_condition_stack,
                )
                steps.extend(stmt_steps)

        return steps, diagnostics

    # ----------------------------------------------
    # FunctionCall → TemplateStep 変換
    # ----------------------------------------------
    def _create_step_from_function_call(self, call_node, statement_index: int) -> TemplateStep:
        """FunctionCallノードからTemplateStepを生成"""
        original_command_name = call_node.name
        command_name = self._normalize_command_name(original_command_name)
        template: CommandTemplate = resolve_command_template(command_name)

        command_args = list(call_node.arguments)

        target_binding = ""
        arg_exprs = command_args
        target_repr = None

        if template.target_binding_required:
            if not command_args:
                raise ValueError(f"{command_name}: 対象オブジェクト引数が不足しています")
            target_expr = command_args[0]
            target_binding = self._expression_to_binding(target_expr)
            target_repr = self._expression_to_string(target_expr)
            arg_exprs = command_args[1:]

        raw_args = [self._expression_to_value(expr) for expr in arg_exprs]

        argument_slots = resolve_argument_slots(
            template,
            raw_args,
            context={"current_object_binding": target_binding},
        )

        metadata = {
            "dsl_command": original_command_name,
            "normalized_command": command_name,
            "statement_index": statement_index,
            "original_arguments": [self._expression_to_string(expr) for expr in arg_exprs],
        }

        if target_repr is not None:
            metadata["target_expression"] = target_repr

        post_conditions = []
        if self.config.attach_execution_metadata:
            post_conditions.append(
                PostCondition(
                    predicate=self.config.post_condition_predicate,
                    parameters={"mode": "after_step"},
                    severity="info",
                    description="各ステップ後に出力グリッドを検証",
                )
            )

        step = TemplateStep(
            operation=template.operation,
            target_binding=target_binding,
            argument_slots=argument_slots,
            post_conditions=post_conditions,
            metadata=metadata,
        )
        self._apply_auto_defaults(step, command_name)
        return step

    def _normalize_command_name(self, command_name: str) -> str:
        """DSLコマンド名をテンプレート登録名へ正規化"""
        normalized = {
            "FILL_HOLE": "FILL_HOLES",
            "GET_HOLE_COUNT": "COUNT_HOLES",
        }
        upper_name = command_name.upper()
        return normalized.get(upper_name, upper_name)

    def _apply_auto_defaults(self, step: TemplateStep, command_name: str) -> None:
        """プレースホルダになった引数へ既定値を適用"""
        defaults = self.config.auto_default_arguments.get(command_name.upper())
        if not defaults:
            return

        applied: Dict[str, str] = {}
        for slot in step.argument_slots:
            default_value = defaults.get(slot.name)
            if default_value is None:
                continue
            if slot.value.kind == ArgumentKind.PLACEHOLDER and slot.value.value is None:
                normalized_default = default_value
                if isinstance(normalized_default, str) and not normalized_default.startswith(("'", '"')):
                    normalized_default = f"'{normalized_default}'"
                slot.value.kind = ArgumentKind.LITERAL
                slot.value.value = normalized_default
                slot.value.constraints = {"source": "auto_default"}
                slot.value.description = "自動補完既定値"
                applied[slot.name] = normalized_default

        if applied:
            auto_meta = step.metadata.setdefault("auto_defaults", {})
            auto_meta.update(applied)

    # ----------------------------------------------
    # Expression 変換ユーティリティ
    # ----------------------------------------------
    def _expression_to_value(self, expr) -> Any:
        """テンプレート引数として扱うための値変換"""
        from src.core_systems.executor.parsing.parser import (
            Literal,
            Identifier,
            ListLiteral,
            Placeholder,
        )

        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Placeholder):
            return f"${{{expr.name}}}"
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, ListLiteral):
            return [self._expression_to_value(elm) for elm in expr.elements]

        # その他は文字列表現
        return self._expression_to_string(expr)

    def _expression_to_string(self, expr) -> str:
        """式を説明文字列へ変換"""
        from src.core_systems.executor.parsing.parser import (
            Literal,
            Identifier,
            ListLiteral,
            BinaryOp,
            UnaryOp,
            FunctionCall,
            IndexAccess,
            AttributeAccess,
            Placeholder,
        )

        if isinstance(expr, Literal):
            return repr(expr.value)
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, Placeholder):
            return f"${{{expr.name}}}"
        if isinstance(expr, ListLiteral):
            return "[" + ", ".join(self._expression_to_string(e) for e in expr.elements) + "]"
        if isinstance(expr, BinaryOp):
            return f"({self._expression_to_string(expr.left)} {expr.operator} {self._expression_to_string(expr.right)})"
        if isinstance(expr, UnaryOp):
            return f"({expr.operator}{self._expression_to_string(expr.operand)})"
        if isinstance(expr, FunctionCall):
            args_str = ", ".join(self._expression_to_string(a) for a in expr.arguments)
            return f"{expr.name}({args_str})"
        if isinstance(expr, IndexAccess):
            return f"{self._expression_to_string(expr.target)}[{self._expression_to_string(expr.index)}]"
        if isinstance(expr, AttributeAccess):
            return f"{self._expression_to_string(expr.target)}.{expr.attribute}"

        return str(expr)

    def _expression_to_binding(self, expr) -> str:
        """対象オブジェクトを表すバインディング文字列へ変換"""
        binding = self._expression_to_value(expr)
        if isinstance(binding, str):
            return binding
        return self._expression_to_string(expr)
