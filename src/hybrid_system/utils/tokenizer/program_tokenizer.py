"""
プログラムトークナイザー

DSLプログラムをトークン化
"""

from typing import List, Dict, Optional
import re


class ProgramTokenizer:
    """
    プログラムトークナイザー

    DSLプログラムを整数トークンに変換
    """

    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3
    }

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """初期化"""
        if vocab is None:
            self.vocab = dict(self.SPECIAL_TOKENS)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.next_id = len(self.vocab)
        else:
            self.vocab = vocab
            self.inverse_vocab = {v: k for k, v in vocab.items()}
            self.next_id = max(vocab.values()) + 1

    def tokenize(self, program: str) -> List[str]:
        """プログラムをトークンに分割

        トークン化のルール:
        1. 記号（括弧、カンマ、コロン、角括弧）は分離
        2. マイナス記号は単独トークンとして分離
        3. 変数名（objects1など）は複合トークンとして保持
        4. 文字列リテラル（"X"など）は保持
        """
        # 文字列リテラルを一時的に保護
        string_literals = []
        def protect_string(match):
            string_literals.append(match.group(0))
            return f"__STRING_{len(string_literals)-1}__"

        program = re.sub(r'"[^"]*"', protect_string, program)

        # 記号の前後にスペースを追加
        program = re.sub(r'([()[\],:])', r' \1 ', program)

        # マイナス記号を分離
        # -1 → - 1, -2 → - 2
        # 負数を検出して分離
        program = re.sub(r'(-\d+)', lambda m: '- ' + m.group(1)[1:], program)

        # 複数スペースを1つに
        program = re.sub(r'\s+', ' ', program)

        # スペースで分割
        tokens = program.split()

        # 文字列リテラルを復元
        restored_tokens = []
        for token in tokens:
            if token.startswith('__STRING_') and token.endswith('__'):
                idx = int(token[9:-2])
                restored_tokens.append(string_literals[idx])
            else:
                restored_tokens.append(token)

        return restored_tokens

    def encode(self, program: str, add_special_tokens: bool = True) -> List[int]:
        """プログラムを整数IDに変換"""
        tokens = self.tokenize(program)

        ids = []
        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<BOS>'])

        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.inverse_vocab[self.next_id] = token
                self.next_id += 1
            ids.append(self.vocab[token])

        if add_special_tokens:
            ids.append(self.SPECIAL_TOKENS['<EOS>'])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """整数IDをプログラムに変換"""
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in [0, 1, 2]:
                continue
            tokens.append(self.inverse_vocab.get(token_id, '<UNK>'))
        return ' '.join(tokens)

    def vocab_size(self) -> int:
        """語彙サイズを取得"""
        return len(self.vocab)

    def save_vocab(self, path: str):
        """語彙を保存"""
        import json
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    def load_vocab(self, path: str):
        """既存のインスタンスに語彙を読み込み"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.next_id = max(vocab.values()) + 1

    @classmethod
    def from_vocab_file(cls, path: str) -> 'ProgramTokenizer':
        """語彙ファイルを読み込んで新しいインスタンスを作成"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab)
