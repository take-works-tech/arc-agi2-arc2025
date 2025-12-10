"""
構文パーサー

トークンからASTを生成する
等号（=）、括弧（）、カンマ（,）を使った構文を解析
"""

from typing import List, Any, Optional, Union
from dataclasses import dataclass
import os

# ログ出力制御（デフォルトで詳細ログを無効化）
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')

# 相対インポート（パッケージとして使用される場合）
try:
    from .tokenizer import Token
except ImportError:
    # 直接実行される場合
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from .tokenizer import Token


# ========================================
# ASTノード定義
# ========================================

class ASTNode:
    """ASTノードの基底クラス"""
    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col


class Assignment(ASTNode):
    """変数代入ノード"""
    def __init__(self, variable: Union[str, 'IndexAccess'], expression: 'Expression', line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.variable = variable  # 変数名またはIndexAccess
        self.expression = expression

    def __repr__(self):
        return f"Assignment({self.variable} = {self.expression})"


class FunctionCall(ASTNode):
    """関数呼び出しノード"""
    def __init__(self, name: str, arguments: List['Expression'], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.arguments)
        return f"FunctionCall({self.name}({args_str}))"


class ForLoop(ASTNode):
    """FORループノード"""
    def __init__(self, loop_var: str, count_expr: 'Expression', body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.loop_var = loop_var
        self.count_expr = count_expr
        self.body = body

    def __repr__(self):
        return f"ForLoop({self.loop_var}, {self.count_expr}, body={len(self.body)} statements)"


class WhileLoop(ASTNode):
    """WHILEループノード"""
    def __init__(self, condition: 'Expression', body: List[ASTNode], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileLoop({self.condition}, body={len(self.body)} statements)"


class IfStatement(ASTNode):
    """IF文ノード"""
    def __init__(self, condition: 'Expression', then_body: List[ASTNode], else_body: Optional[List[ASTNode]] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body or []

    def __repr__(self):
        return f"IfStatement(condition={self.condition}, then={len(self.then_body)}, else={len(self.else_body)})"


class Expression(ASTNode):
    """式の基底クラス"""

    def evaluate(self, interpreter):
        """式を評価"""
        raise NotImplementedError("Expression subclasses must implement evaluate()")


class Identifier(Expression):
    """識別子（変数名）"""
    def __init__(self, name: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name

    def __repr__(self):
        return f"Identifier({self.name})"


class Literal(Expression):
    """リテラル（数値、文字列）"""
    def __init__(self, value: Any, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value

    def __repr__(self):
        return f"Literal({self.value!r})"


class Placeholder(Expression):
    """プレースホルダー（$obj）"""
    def __init__(self, name: str = "current", line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name

    def __repr__(self):
        return f"Placeholder(${self.name})"


class BinaryOp(Expression):
    """二項演算"""
    def __init__(self, operator: str, left: Expression, right: Expression, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.left} {self.operator} {self.right})"


class UnaryOp(Expression):
    """単項演算"""
    def __init__(self, operator: str, operand: Expression, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.operator} {self.operand})"


class ListLiteral(Expression):
    """リストリテラル"""
    def __init__(self, elements: List[Expression], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.elements = elements

    def __repr__(self):
        elements_str = ', '.join(str(e) for e in self.elements)
        return f"ListLiteral([{elements_str}])"


class IndexAccess(Expression):
    """インデックスアクセス（配列[index]）"""
    def __init__(self, target: Expression, index: Expression, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.target = target
        self.index = index

    def __repr__(self):
        return f"IndexAccess({self.target}[{self.index}])"


class AttributeAccess(Expression):
    """属性アクセス（obj.attribute）"""
    def __init__(self, target: Expression, attribute: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.target = target
        self.attribute = attribute

    def __repr__(self):
        return f"AttributeAccess({self.target}.{self.attribute})"


# ========================================
# パーサー
# ========================================

class Parser:
    """構文パーサー"""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> List[ASTNode]:
        """
        トークンリストからASTを生成

        Returns:
            ASTノードのリスト
        """
        statements = []
        total_tokens = len(self.tokens)
        processed_count = 0
        last_pos = -1  # 無限ループ検出用
        consecutive_no_progress = 0  # 連続して進捗がない回数

        while not self.is_at_end():
            # 無限ループ検出: 同じ位置に留まり続ける場合
            if self.pos == last_pos:
                consecutive_no_progress += 1
                if consecutive_no_progress > 10:  # 10回連続で進捗がない場合
                    current_token = self.current()
                    error_msg = f"パース処理が無限ループに陥っています。トークン位置: {self.pos}/{total_tokens}, 現在のトークン: {current_token.type}='{current_token.value}' (行{current_token.line}, 列{current_token.col})"
                    print(f"  [エラー] {error_msg}", flush=True)
                    raise SyntaxError(error_msg)
            else:
                consecutive_no_progress = 0
                last_pos = self.pos

            # 進捗ログ（100トークンごと、または10ステートメントごと、または最初の1回）
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and (processed_count == 0 or processed_count % 100 == 0 or len(statements) % 10 == 0):
                progress = (self.pos / total_tokens * 100) if total_tokens > 0 else 0
                current_token = self.current()
                print(f"  [パース進捗] トークン位置: {self.pos}/{total_tokens} ({progress:.1f}%), ステートメント数: {len(statements)}, 現在のトークン: {current_token.type}='{current_token.value}'", flush=True)

            # 改行をスキップ
            if self.current().type == 'NEWLINE':
                self.advance()
                processed_count += 1
                continue

            # ステートメントを解析
            stmt_start_pos = self.pos
            try:
                stmt = self.parse_statement()
            except Exception as e:
                current_token = self.current()
                error_msg = f"パースエラー: {type(e).__name__}: {e} (トークン位置: {self.pos}/{total_tokens}, 現在のトークン: {current_token.type}='{current_token.value}' 行{current_token.line})"
                print(f"  [エラー] {error_msg}", flush=True)
                raise SyntaxError(error_msg) from e

            # ステートメント解析後に位置が進んでいない場合のチェック
            if stmt is None and self.pos == stmt_start_pos:
                # 進捗がない場合、強制的に1つ進める（無限ループ防止）
                print(f"  [警告] パース処理が進捗していません。位置を強制的に進めます (位置: {self.pos})", flush=True)
                self.advance()
                processed_count += 1
                continue

            if stmt:
                statements.append(stmt)

            processed_count += 1

        return statements

    def parse_statement(self) -> Optional[ASTNode]:
        """
        ステートメントを解析

        Returns:
            ASTノード（AssignmentまたはFunctionCall）
        """
        # FORループ
        if self.current().type == 'KEYWORD' and self.current().value == 'FOR':
            return self.parse_for_loop()

        # WHILEループ
        if self.current().type == 'KEYWORD' and self.current().value == 'WHILE':
            return self.parse_while_loop()

        # IF文
        if self.current().type == 'KEYWORD' and self.current().value == 'IF':
            return self.parse_if_statement()

        # IDENTIFIERで始まる場合、代入かどうか先読みチェック
        if self.current().type == 'IDENTIFIER':
            # 配列要素への代入: var[index] = expr
            if self.peek_ahead(1) and self.peek_ahead(1).type == 'LBRACKET':
                # さらに先読みして'='を探す
                saved_pos = self.pos
                self.advance()  # IDENTIFIER
                self.advance()  # LBRACKET
                # インデックス部分をスキップ
                bracket_count = 1
                while bracket_count > 0 and not self.is_at_end():
                    if self.current().type == 'LBRACKET':
                        bracket_count += 1
                    elif self.current().type == 'RBRACKET':
                        bracket_count -= 1
                    self.advance()

                # '='があるか確認
                is_assignment = self.current().type == 'ASSIGN'
                self.pos = saved_pos  # 位置を戻す

                if is_assignment:
                    return self.parse_array_element_assignment()

            # 通常の変数代入: var = expr
            elif self.peek_ahead(1) and self.peek_ahead(1).type == 'ASSIGN':
                return self.parse_assignment()

        # そうでなければ関数呼び出しまたは式
        expr = self.parse_expression()

        # 改行をスキップ
        self.skip_newlines()

        # 式が関数呼び出しならそのまま返す
        if isinstance(expr, FunctionCall):
            return expr

        # それ以外は無視（式のみの行）
        return None

    def parse_assignment(self) -> Assignment:
        """
        変数代入を解析

        Returns:
            Assignmentノード
        """
        var_token = self.expect('IDENTIFIER')
        var_name = var_token.value

        self.expect('ASSIGN')

        expr = self.parse_expression()

        return Assignment(var_name, expr, var_token.line, var_token.col)

    def parse_array_element_assignment(self) -> Assignment:
        """
        配列要素への代入を解析: var[index] = expr

        Returns:
            Assignmentノード（variableがIndexAccess）
        """
        var_token = self.expect('IDENTIFIER')
        var_name = var_token.value

        # [index]をパース
        self.expect('LBRACKET')
        index = self.parse_expression()
        self.expect('RBRACKET')

        # IndexAccessノードを作成
        index_access = IndexAccess(
            Identifier(var_name, var_token.line, var_token.col),
            index,
            var_token.line,
            var_token.col
        )

        self.expect('ASSIGN')

        expr = self.parse_expression()

        return Assignment(index_access, expr, var_token.line, var_token.col)

    def parse_for_loop(self) -> ForLoop:
        """
        FORループを解析: FOR var count_expr DO ... END
        または: FOR var LEN(array) DO ... END

        Returns:
            ForLoopノード
        """
        start_token = self.current()
        self.expect_keyword('FOR')

        # ループ変数名
        loop_var_token = self.expect('IDENTIFIER')
        loop_var = loop_var_token.value

        # カウント式（複雑な式も解析可能）
        count_expr = self.parse_expression()

        # DO
        self.expect_keyword('DO')
        self.skip_newlines()

        # ループ本体（深いネストをサポート）
        body = []
        depth = 0
        while not self.is_at_end():
            if self.current().type == 'KEYWORD' and self.current().value == 'END':
                if depth == 0:
                    break
                else:
                    depth -= 1
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                    continue
            elif self.current().type == 'KEYWORD' and self.current().value in ('FOR', 'WHILE', 'IF'):
                depth += 1

            # 改行をスキップ
            if self.current().type == 'NEWLINE':
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)

        # END
        self.expect_keyword('END')
        self.skip_newlines()

        return ForLoop(loop_var, count_expr, body, start_token.line, start_token.col)

    def parse_while_loop(self) -> WhileLoop:
        """
        WHILEループを解析: WHILE condition DO ... END

        Returns:
            WhileLoopノード
        """
        start_token = self.current()
        self.expect_keyword('WHILE')

        # 条件式（複雑な式も解析可能）
        condition = self.parse_expression()

        # DO
        self.expect_keyword('DO')
        self.skip_newlines()

        # ループ本体（深いネストをサポート）
        body = []
        depth = 0
        while not self.is_at_end():
            if self.current().type == 'KEYWORD' and self.current().value == 'END':
                if depth == 0:
                    break
                else:
                    depth -= 1
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                    continue
            elif self.current().type == 'KEYWORD' and self.current().value in ('FOR', 'WHILE', 'IF'):
                depth += 1

            # 改行をスキップ
            if self.current().type == 'NEWLINE':
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)

        # END
        self.expect_keyword('END')
        self.skip_newlines()

        return WhileLoop(condition, body, start_token.line, start_token.col)

    def parse_if_statement(self) -> IfStatement:
        """
        IF文を解析: IF condition THEN ... END
        または: IF condition THEN ... ELSE ... END

        Returns:
            IfStatementノード
        """
        start_token = self.current()
        self.expect_keyword('IF')

        # 条件式（複雑な式も解析可能）
        condition = self.parse_expression()

        # THEN
        self.expect_keyword('THEN')
        self.skip_newlines()

        # THEN本体（深いネストをサポート）
        then_body = []
        depth = 0
        while not self.is_at_end():
            if self.current().type == 'KEYWORD' and self.current().value in ('ELSE', 'END'):
                if depth == 0:
                    break
                else:
                    depth -= 1
                    stmt = self.parse_statement()
                    if stmt:
                        then_body.append(stmt)
                    continue
            elif self.current().type == 'KEYWORD' and self.current().value in ('FOR', 'WHILE', 'IF'):
                depth += 1

            # 改行をスキップ
            if self.current().type == 'NEWLINE':
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                then_body.append(stmt)

        # ELSE本体（オプション、深いネストをサポート）
        else_body = []
        if self.current().type == 'KEYWORD' and self.current().value == 'ELSE':
            self.advance()  # ELSE
            self.skip_newlines()

            depth = 0
            while not self.is_at_end():
                if self.current().type == 'KEYWORD' and self.current().value == 'END':
                    if depth == 0:
                        break
                    else:
                        depth -= 1
                        stmt = self.parse_statement()
                        if stmt:
                            else_body.append(stmt)
                        continue
                elif self.current().type == 'KEYWORD' and self.current().value in ('FOR', 'WHILE', 'IF'):
                    depth += 1

                # 改行をスキップ
                if self.current().type == 'NEWLINE':
                    self.advance()
                    continue

                stmt = self.parse_statement()
                if stmt:
                    else_body.append(stmt)

        # END
        self.expect_keyword('END')
        self.skip_newlines()

        return IfStatement(condition, then_body, else_body, start_token.line, start_token.col)

    def parse_expression(self) -> Expression:
        """
        式を解析（優先順位を考慮）

        Returns:
            Expression
        """
        return self.parse_or_expression()

    def parse_or_expression(self) -> Expression:
        """論理和式を解析"""
        left = self.parse_and_expression()

        while self.check('IDENTIFIER') and self.current().value in ('or', 'OR'):
            self.advance()  # 'or'/'OR'を消費
            op = self.previous().value
            right = self.parse_and_expression()
            left = BinaryOp(op, left, right)

        return left

    def parse_and_expression(self) -> Expression:
        """論理積式を解析"""
        left = self.parse_comparison_expression()

        while self.check('IDENTIFIER') and self.current().value in ('and', 'AND'):
            self.advance()  # 'and'/'AND'を消費
            op = self.previous().value
            right = self.parse_comparison_expression()
            left = BinaryOp(op, left, right)

        return left

    def parse_comparison_expression(self) -> Expression:
        """比較式を解析（実装済みコマンドのみ使用）"""
        left = self.parse_additive_expression()

        # 実装済みの比較コマンドのみをサポート
        while self.match('IDENTIFIER'):
            if self.previous().value in ('EQUAL', 'NOT_EQUAL', 'GREATER', 'LESS'):
                op = self.previous().value
                right = self.parse_additive_expression()
                left = BinaryOp(op, left, right)
            else:
                # 比較コマンドではない場合は戻る
                self.rewind()
                break

        return left

    def parse_additive_expression(self) -> Expression:
        """加減算式を解析"""
        left = self.parse_multiplicative_expression()

        while self.match('PLUS', 'MINUS'):
            op = self.previous().value
            right = self.parse_multiplicative_expression()
            left = BinaryOp(op, left, right)

        return left

    def parse_multiplicative_expression(self) -> Expression:
        """乗除算式を解析"""
        left = self.parse_unary_expression()

        while self.match('MULTIPLY', 'DIVIDE'):
            op = self.previous().value
            right = self.parse_unary_expression()
            left = BinaryOp(op, left, right)

        return left

    def parse_unary_expression(self) -> Expression:
        """単項式を解析"""
        # not演算子
        if self.check('IDENTIFIER') and self.current().value in ('not', 'NOT'):
            token = self.advance()
            op = token.value
            operand = self.parse_unary_expression()
            return UnaryOp(op, operand, token.line, token.col)

        # マイナス演算子
        if self.match('MINUS'):
            op = self.previous().value
            operand = self.parse_unary_expression()
            return UnaryOp(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self) -> Expression:
        """後置式を解析（関数呼び出し、インデックスアクセス、属性アクセス）"""
        expr = self.parse_primary_expression()

        while True:
            # 関数呼び出し
            if self.check('LPAREN'):
                # 関数名を取得
                if isinstance(expr, Identifier):
                    func_name = expr.name
                    func_line = expr.line
                    func_col = expr.col
                else:
                    # 他の式の場合、エラー
                    raise SyntaxError(f"Function call requires identifier, got {type(expr).__name__}")

                self.advance()  # '('を消費
                arguments = self.parse_argument_list()
                self.expect('RPAREN')
                expr = FunctionCall(func_name, arguments, func_line, func_col)

            # インデックスアクセス
            elif self.match('LBRACKET'):
                index = self.parse_expression()
                self.expect('RBRACKET')
                expr = IndexAccess(expr, index)

            # 属性アクセス
            elif self.match('DOT'):
                attr_token = self.expect('IDENTIFIER')
                expr = AttributeAccess(expr, attr_token.value)

            else:
                break

        return expr

    def parse_primary_expression(self) -> Expression:
        """基本式を解析"""
        token = self.current()

        # 数値リテラル
        if self.match('NUMBER'):
            value = self.previous().value
            number_value = float(value) if '.' in value else int(value)
            return Literal(number_value, token.line, token.col)

        # 文字列リテラル
        if self.match('STRING'):
            value = self.previous().value
            # 文字列の前後の引用符を削除
            string_value = value[1:-1]
            return Literal(string_value, token.line, token.col)

        # 変数展開 {variable}
        if self.match('VAR_EXPANSION'):
            value = self.previous().value
            # 中括弧を削除して変数名を取得
            var_name = value[1:-1]
            return Identifier(var_name, token.line, token.col)

        # プレースホルダー $obj, $obj1, $obj2
        if self.match('PLACEHOLDER'):
            # トークン値から$を除去してnameを取得（例: "$obj1" → "obj1"）
            placeholder_name = self.previous().value[1:]  # $を除去
            return Placeholder(placeholder_name, token.line, token.col)

        # 識別子
        if self.match('IDENTIFIER'):
            name = self.previous().value

            # True/False
            if name in ('True', 'False'):
                return Literal(name == 'True', token.line, token.col)

            # None
            if name == 'None':
                return Literal(None, token.line, token.col)

            return Identifier(name, token.line, token.col)

        # リストリテラル
        if self.match('LBRACKET'):
            elements = []
            if not self.check('RBRACKET'):
                elements.append(self.parse_expression())
                while self.match('COMMA'):
                    elements.append(self.parse_expression())
            self.expect('RBRACKET')
            return ListLiteral(elements, token.line, token.col)

        # LEN関数
        if self.current().type == 'KEYWORD' and self.current().value == 'LEN':
            self.advance()  # LENを消費
            self.expect('LPAREN')
            array_expr = self.parse_expression()
            self.expect('RPAREN')
            return FunctionCall('LEN', [array_expr], token.line, token.col)

        # 括弧式
        if self.match('LPAREN'):
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr

        # キーワード（DO, END, THEN, ELSE）は式として解析しない
        if token.type == 'KEYWORD' and token.value in ('DO', 'END', 'THEN', 'ELSE'):
            return None

        # その他のトークンはエラー
        raise SyntaxError(f"Unexpected token: {token.type} '{token.value}' at line {token.line}")

    def parse_argument_list(self) -> List[Expression]:
        """引数リストを解析"""
        arguments = []

        if not self.check('RPAREN'):
            arguments.append(self.parse_expression())

            while self.match('COMMA'):
                arguments.append(self.parse_expression())

        return arguments

    # ========================================
    # ヘルパーメソッド
    # ========================================

    def current(self) -> Token:
        """現在のトークンを取得"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def previous(self) -> Token:
        """前のトークンを取得"""
        if self.pos > 0:
            return self.tokens[self.pos - 1]
        return self.tokens[0]

    def peek_ahead(self, n: int) -> Optional[Token]:
        """n個先のトークンを取得"""
        pos = self.pos + n
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def advance(self) -> Token:
        """次のトークンに進む"""
        token = self.current()
        if not self.is_at_end():
            self.pos += 1
        return token

    def check(self, *token_types: str) -> bool:
        """現在のトークンが指定された型かチェック"""
        if self.is_at_end():
            return False
        return self.current().type in token_types

    def match(self, *token_types: str) -> bool:
        """現在のトークンが指定された型なら進む"""
        if self.check(*token_types):
            self.advance()
            return True
        return False

    def expect(self, token_type: str) -> Token:
        """指定された型のトークンを期待"""
        token = self.current()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type}, got {token.type} '{token.value}' "
                f"at line {token.line}, col {token.col}"
            )
        return self.advance()

    def expect_keyword(self, keyword: str) -> Token:
        """指定されたキーワードを期待"""
        token = self.current()
        if token.type != 'KEYWORD' or token.value != keyword:
            raise SyntaxError(
                f"Expected keyword '{keyword}', got {token.type} '{token.value}' "
                f"at line {token.line}, col {token.col}"
            )
        return self.advance()

    def skip_newlines(self):
        """改行をスキップ"""
        while self.match('NEWLINE'):
            pass  # match()が既にadvance()を呼び出しているため、追加のadvance()は不要

    def is_at_end(self) -> bool:
        """最後に到達したかチェック"""
        return self.current().type == 'EOF'

    def rewind(self):
        """トークン位置を1つ戻す"""
        if self.pos > 0:
            self.pos -= 1

    def peek_ahead(self, n: int) -> Optional[Token]:
        """n個先のトークンを先読み（変更しない）"""
        if self.pos + n >= len(self.tokens):
            return None
        return self.tokens[self.pos + n]

    def parse_nested_expression(self) -> Expression:
        """ネストした式を解析（括弧、関数呼び出し、配列アクセスなど）"""
        # 括弧式
        if self.match('LPAREN'):
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr

        # 配列アクセス
        if self.current().type == 'IDENTIFIER':
            var_token = self.current()
            self.advance()

            if self.current().type == 'LBRACKET':
                # 配列アクセス: var[index]
                self.advance()  # [
                index_expr = self.parse_expression()
                self.expect('RBRACKET')  # ]
                return IndexAccess(Identifier(var_token.value, var_token.line, var_token.col),
                                 index_expr, var_token.line, var_token.col)
            else:
                # 通常の識別子
                return Identifier(var_token.value, var_token.line, var_token.col)

        # 関数呼び出し
        if self.current().type == 'IDENTIFIER':
            return self.parse_function_call()

        # 数値リテラル
        if self.match('NUMBER'):
            value = float(self.previous().value) if '.' in self.previous().value else int(self.previous().value)
            return Literal(value, self.previous().line, self.previous().col)

        # プレースホルダー
        if self.match('PLACEHOLDER'):
            placeholder_name = self.previous().value[1:]  # $を除去
            return Placeholder(placeholder_name, self.previous().line, self.previous().col)

        raise SyntaxError(f"Unexpected token: {self.current().type} '{self.current().value}' at line {self.current().line}")

    def parse_function_call(self) -> FunctionCall:
        """関数呼び出しを解析"""
        name_token = self.current()
        self.expect('IDENTIFIER')
        function_name = name_token.value

        # 引数リスト
        args = []
        if self.current().type == 'LPAREN':
            self.advance()  # (

            # 引数がある場合
            if self.current().type != 'RPAREN':
                while True:
                    arg = self.parse_expression()
                    args.append(arg)

                    if self.current().type == 'COMMA':
                        self.advance()  # ,
                    else:
                        break

            self.expect('RPAREN')  # )

        return FunctionCall(function_name, args, name_token.line, name_token.col)


def test_parser():
    """パーサーのテスト"""
    try:
        from .tokenizer import Tokenizer
    except ImportError:
        from src.core_systems.executor.parsing.tokenizer import Tokenizer

    tokenizer = Tokenizer()

    # テスト1: 変数代入
    print("Test 1: 変数代入")
    code1 = "objects = GET_ALL_OBJECTS()"
    tokens1 = tokenizer.tokenize(code1)
    parser1 = Parser(tokens1)
    ast1 = parser1.parse()
    for node in ast1:
        print(f"  {node}")
    print()

    # テスト2: 関数呼び出し
    print("Test 2: 関数呼び出し")
    code2 = "MOVE(objects, 0, 1)"
    tokens2 = tokenizer.tokenize(code2)
    parser2 = Parser(tokens2)
    ast2 = parser2.parse()
    for node in ast2:
        print(f"  {node}")
    print()

    # テスト3: ネスト式
    print("Test 3: ネスト式")
    code3 = "result = GREATER(GET_SIZE(objects), 10)"
    tokens3 = tokenizer.tokenize(code3)
    parser3 = Parser(tokens3)
    ast3 = parser3.parse()
    for node in ast3:
        print(f"  {node}")
    print()

    # テスト4: 複数ステートメント
    print("Test 4: 複数ステートメント")
    code4 = '''SET_OBJECT_TYPE("single_color_4way")
objects = GET_ALL_OBJECTS()
MOVE(objects, 0, 1)'''
    tokens4 = tokenizer.tokenize(code4)
    parser4 = Parser(tokens4)
    ast4 = parser4.parse()
    for node in ast4:
        print(f"  {node}")
    print()

    # テスト5: リストとインデックスアクセス
    print("Test 5: リストとインデックスアクセス")
    code5 = '''my_list = [1, 2, 3]
element = my_list[0]'''
    tokens5 = tokenizer.tokenize(code5)
    parser5 = Parser(tokens5)
    ast5 = parser5.parse()
    for node in ast5:
        print(f"  {node}")
    print()


if __name__ == '__main__':
    test_parser()
