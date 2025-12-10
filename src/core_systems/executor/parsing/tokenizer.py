"""
新構文用トークナイザー

等号（=）、括弧（）、カンマ（,）を認識する
Pythonライクな構文をトークン化する
"""

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Token:
    """トークンクラス"""
    type: str
    value: str
    line: int
    col: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}, {self.col})"


class Tokenizer:
    """新構文用トークナイザー"""
    
    # トークンパターン（優先順位の高い順）
    TOKEN_PATTERNS = [
        ('STRING', r'"[^"]*"|\'[^\']*\''),  # 文字列リテラル
        ('VAR_EXPANSION', r'\{[a-zA-Z_][a-zA-Z0-9_]*\}'),  # 変数展開 {variable}
        ('NUMBER', r'\d+\.?\d*'),           # 数値
        ('KEYWORD', r'\b(FOR|WHILE|DO|END|IF|THEN|ELSE|LEN)\b'),  # キーワード
        ('ASSIGN', r'='),                   # 等号
        ('LPAREN', r'\('),                  # 左括弧
        ('RPAREN', r'\)'),                  # 右括弧
        ('LBRACKET', r'\['),                # 左角括弧
        ('RBRACKET', r'\]'),                # 右角括弧
        ('COMMA', r','),                    # カンマ
        ('COLON', r':'),                    # コロン
        ('DOT', r'\.'),                     # ドット
        ('PLUS', r'\+'),                    # プラス
        ('MINUS', r'-'),                    # マイナス
        ('MULTIPLY', r'\*'),                # 乗算
        ('DIVIDE', r'/'),                   # 除算
        ('GT', r'>'),                       # 大なり
        ('LT', r'<'),                       # 小なり
        ('GTE', r'>='),                     # 以上
        ('LTE', r'<='),                     # 以下
        ('EQ', r'=='),                      # 等価
        ('NEQ', r'!='),                     # 不等価
        ('PLACEHOLDER', r'\$obj\d*'),   # プレースホルダー（$obj, $obj1, $obj2などをサポート）
        ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),  # 識別子
        ('NEWLINE', r'\n'),                 # 改行
        ('WHITESPACE', r'[ \t]+'),          # 空白
        ('COMMENT', r'#.*'),                # コメント
    ]
    
    def __init__(self, code: str = ""):
        # パターンをコンパイル
        self.compiled_patterns = [
            (name, re.compile(pattern))
            for name, pattern in self.TOKEN_PATTERNS
        ]
        self.code = code
    
    def tokenize(self, code: str) -> List[Token]:
        """
        コードをトークンに分割
        
        Args:
            code: プログラムコード
            
        Returns:
            トークンのリスト
        """
        tokens = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            col = 0
            
            while col < len(line):
                # 各パターンを試す
                matched = False
                
                for token_type, pattern in self.compiled_patterns:
                    match = pattern.match(line, col)
                    if match:
                        value = match.group(0)
                        
                        # 空白とコメントは無視
                        if token_type not in ('WHITESPACE', 'COMMENT'):
                            tokens.append(Token(token_type, value, line_num, col))
                        
                        col = match.end()
                        matched = True
                        break
                
                if not matched:
                    # マッチしない文字がある場合はエラー
                    raise SyntaxError(
                        f"Invalid character at line {line_num}, col {col}: '{line[col]}'"
                    )
            
            # 改行トークンを追加（最後の行以外）
            if line_num < len(lines) or line.endswith('\n'):
                tokens.append(Token('NEWLINE', '\\n', line_num, len(line)))
        
        # EOFトークンを追加
        tokens.append(Token('EOF', '', len(lines), 0))
        
        return tokens
    
    def tokenize_with_indentation(self, code: str) -> List[Token]:
        """
        インデントを考慮してトークン化（将来のPythonライク構文用）
        
        Args:
            code: プログラムコード
            
        Returns:
            トークンのリスト（INDENT/DEDENTを含む）
        """
        tokens = []
        lines = code.split('\n')
        indent_stack = [0]  # インデントレベルのスタック
        
        for line_num, line in enumerate(lines, 1):
            # 空行やコメントのみの行はスキップ
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # インデントレベルを計算
            indent_level = len(line) - len(line.lstrip())
            
            # インデントの変化を検出
            if indent_level > indent_stack[-1]:
                # インデント増加
                indent_stack.append(indent_level)
                tokens.append(Token('INDENT', '', line_num, 0))
            
            elif indent_level < indent_stack[-1]:
                # インデント減少
                while indent_stack[-1] > indent_level:
                    indent_stack.pop()
                    tokens.append(Token('DEDENT', '', line_num, 0))
                
                if indent_stack[-1] != indent_level:
                    raise SyntaxError(
                        f"Indentation error at line {line_num}"
                    )
            
            # 行の内容をトークン化
            col = indent_level
            stripped_line = line[indent_level:]
            
            while col - indent_level < len(stripped_line):
                matched = False
                
                for token_type, pattern in self.compiled_patterns:
                    match = pattern.match(stripped_line, col - indent_level)
                    if match:
                        value = match.group(0)
                        
                        if token_type not in ('WHITESPACE', 'COMMENT'):
                            tokens.append(Token(token_type, value, line_num, col))
                        
                        col += len(value)
                        matched = True
                        break
                
                if not matched:
                    char = stripped_line[col - indent_level]
                    raise SyntaxError(
                        f"Invalid character at line {line_num}, col {col}: '{char}'"
                    )
            
            # 改行トークンを追加
            tokens.append(Token('NEWLINE', '\\n', line_num, len(line)))
        
        # 残りのDEDENTトークンを追加
        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(Token('DEDENT', '', len(lines), 0))
        
        # EOFトークンを追加
        tokens.append(Token('EOF', '', len(lines), 0))
        
        return tokens


def test_tokenizer():
    """トークナイザーのテスト"""
    
    # テスト1: 基本的な変数代入
    code1 = 'objects = GET_ALL_OBJECTS()'
    tokenizer = Tokenizer()
    tokens1 = tokenizer.tokenize(code1)
    print("Test 1: 基本的な変数代入")
    for token in tokens1:
        print(f"  {token}")
    print()
    
    # テスト2: 関数呼び出し
    code2 = 'MOVE(objects, 0, 1)'
    tokens2 = tokenizer.tokenize(code2)
    print("Test 2: 関数呼び出し")
    for token in tokens2:
        print(f"  {token}")
    print()
    
    # テスト3: ネスト式
    code3 = 'result = GREATER(GET_SIZE(objects), 10)'
    tokens3 = tokenizer.tokenize(code3)
    print("Test 3: ネスト式")
    for token in tokens3:
        print(f"  {token}")
    print()
    
    # テスト4: 文字列リテラル
    code4 = 'SET_OBJECT_TYPE("single_color_4way")'
    tokens4 = tokenizer.tokenize(code4)
    print("Test 4: 文字列リテラル")
    for token in tokens4:
        print(f"  {token}")
    print()
    
    # テスト5: 複数行
    code5 = '''objects = GET_ALL_OBJECTS()
count = GET_SIZE(objects)
MOVE(objects, 0, 1)'''
    tokens5 = tokenizer.tokenize(code5)
    print("Test 5: 複数行")
    for token in tokens5:
        print(f"  {token}")
    print()


if __name__ == '__main__':
    test_tokenizer()

