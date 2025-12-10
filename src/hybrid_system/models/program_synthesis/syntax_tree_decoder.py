"""
構文木ベースプログラムデコーダー

構文制約を考慮したプログラム生成
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum

# 既存のParserをインポート
try:
    from src.core_systems.executor.parsing.parser import Parser, ASTNode
    from src.core_systems.executor.parsing.tokenizer import Tokenizer
except ImportError:
    Parser = None
    ASTNode = None
    Tokenizer = None


class SyntaxNodeType(Enum):
    """構文ノードタイプ"""
    COMMAND = "command"
    ASSIGNMENT = "assignment"
    LOOP = "loop"
    CONDITION = "condition"
    EXPRESSION = "expression"
    ARGUMENT = "argument"


@dataclass
class SyntaxNode:
    """構文ノード"""
    node_type: SyntaxNodeType
    value: str
    children: List['SyntaxNode']
    metadata: Dict[str, Any]


class SyntaxTreeDecoder(nn.Module):
    """構文木ベースプログラムデコーダー"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 512,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ):
        """
        初期化

        Args:
            vocab_size: 語彙サイズ
            embed_dim: 埋め込み次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            max_length: 最大長
            bos_token_id: BOSトークンID
            eos_token_id: EOSトークンID
            pad_token_id: PADトークンID
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # トークンの埋め込み
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 構文ノードタイプの埋め込み
        self.node_type_embedding = nn.Embedding(len(SyntaxNodeType), embed_dim)

        # 位置エンコーディング
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        # コンテキスト投影
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # メモリ投影（入力と出力のエンコード特徴量の次元が異なる場合用）
        self.memory_proj = None  # 必要に応じて動的に作成

        # トランスフォーマーデコーダ
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 構文制約チェッカー
        self.syntax_checker = SyntaxConstraintChecker()

        # 出力層（トークン + 構文ノードタイプ）
        self.token_output_proj = nn.Linear(embed_dim, vocab_size)
        self.node_type_output_proj = nn.Linear(embed_dim, len(SyntaxNodeType))

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        syntax_tree: Optional[SyntaxNode] = None,
        input_memory: Optional[torch.Tensor] = None,
        output_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        順伝播

        Args:
            tokens: プログラムトークン [batch, seq_len]
            context: コンテキスト [batch, context_dim]
            syntax_tree: 構文木（オプション）

        Returns:
            token_logits: トークンのlogits [batch, seq_len, vocab_size]
            node_type_logits: 構文ノードタイプのlogits [batch, seq_len, num_node_types]
            attention_weights: アテンション重み（オプション）
        """
        batch_size, seq_len = tokens.shape

        # トークンの埋め込み
        token_embed = self.token_embedding(tokens)  # [batch, seq_len, embed_dim]

        # 構文ノードタイプの埋め込み（構文木から取得）
        if syntax_tree:
            node_type_ids = self._extract_node_type_ids(tokens, syntax_tree)
            node_type_embed = self.node_type_embedding(node_type_ids)  # [batch, seq_len, embed_dim]
            token_embed = token_embed + node_type_embed

        # 位置エンコーディング
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.position_embedding(positions)  # [batch, seq_len, embed_dim]

        # 埋め込みを結合
        embed = token_embed + pos_embed
        embed = self.dropout(embed)

        # メモリを準備（強化版: 入力と出力のエンコード特徴量を直接使用）
        if input_memory is not None and output_memory is not None:
            # 入力と出力のエンコード特徴量を連結してメモリとして使用
            memory = torch.cat([input_memory, output_memory], dim=1)  # [batch, seq_len_input + seq_len_output, embed_dim]
            # 次元が一致することを確認
            if memory.size(-1) != self.embed_dim:
                # 次元が異なる場合は投影
                if self.memory_proj is None:
                    self.memory_proj = nn.Linear(memory.size(-1), self.embed_dim).to(memory.device)
                memory = self.memory_proj(memory)
        elif context is not None:
            # 従来の方法: プーリングされたコンテキストを使用
            if context.dim() == 2:
                context = context.unsqueeze(1)
            # 次元が一致することを確認
            if context.size(-1) != self.embed_dim:
                context = self.context_proj(context)  # [batch, 1, embed_dim]
            memory = context
        else:
            # デフォルトコンテキスト
            memory = torch.zeros(batch_size, 1, self.embed_dim, device=tokens.device)

        # 因果マスク
        tgt_mask = self._generate_square_subsequent_mask(seq_len, tokens.device)

        # トランスフォーマーデコーダ（Cross-Attentionが自動的に適用される）
        decoded = self.transformer(
            tgt=embed,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [batch, seq_len, embed_dim]

        # レイヤー正規化
        decoded = self.layer_norm(decoded)

        # 出力投影
        token_logits = self.token_output_proj(decoded)  # [batch, seq_len, vocab_size]
        node_type_logits = self.node_type_output_proj(decoded)  # [batch, seq_len, num_node_types]

        return token_logits, node_type_logits, None

    def _extract_node_type_ids(
        self,
        tokens: torch.Tensor,
        syntax_tree: SyntaxNode
    ) -> torch.Tensor:
        """
        構文木からノードタイプIDを抽出（本格実装）

        構文木を再帰的に走査して、各トークン位置に対応するノードタイプを取得
        """
        batch_size, seq_len = tokens.shape
        node_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=tokens.device)

        if syntax_tree is None:
            return node_type_ids

        # 構文木を走査してノードタイプをマッピング
        def traverse_tree(node: SyntaxNode, position: int, node_type_map: Dict[int, SyntaxNodeType]):
            """構文木を再帰的に走査してノードタイプをマッピング"""
            if node is None:
                return position

            # 現在のノードタイプをマッピング
            node_type_map[position] = node.node_type

            # 子ノードを再帰的に処理
            for child in node.children:
                position = traverse_tree(child, position + 1, node_type_map)

            return position

        # 構文木からノードタイプマップを構築
        node_type_map: Dict[int, SyntaxNodeType] = {}
        traverse_tree(syntax_tree, 0, node_type_map)

        # トークン位置にノードタイプIDを設定
        for pos, node_type in node_type_map.items():
            if pos < seq_len:
                node_type_id = list(SyntaxNodeType).index(node_type)
                node_type_ids[0, pos] = node_type_id

        return node_type_ids

    def _generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """因果マスクを生成"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def generate(
        self,
        context: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_probabilities: bool = False,
        input_memory: Optional[torch.Tensor] = None,
        output_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """自己回帰的にプログラムを生成（ProgramDecoderとの互換性のため）

        Args:
            context: コンテキスト [batch, context_dim]
            max_length: 最大長
            temperature: 温度パラメータ
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            return_probabilities: 確率を返すかどうか
            input_memory: 入力エンコード特徴量（ProgramDecoderとの互換性のため受け入れるが、generateメソッドでは未使用）
            output_memory: 出力エンコード特徴量（ProgramDecoderとの互換性のため受け入れるが、generateメソッドでは未使用）

        Returns:
            tokens: 生成されたトークン [batch, seq_len]
            probabilities: 各トークンの確率（オプション）
        """
        # 注: input_memoryとoutput_memoryはProgramDecoderとの互換性のため受け入れるが、
        # generateメソッドではforwardメソッドに渡していないため未使用
        if max_length is None:
            max_length = self.max_length

        if context is None:
            raise ValueError("contextは必須です")

        batch_size = context.size(0)
        device = context.device

        # 開始トークン
        tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        syntax_tree = None

        probabilities = [] if return_probabilities else None

        for _ in range(max_length - 1):
            # 順伝播
            token_logits, _, _ = self.forward(tokens, context, syntax_tree)

            # 最後のトークンのlogitsを取得
            next_logits = token_logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Top-Kサンプリング
            if top_k is not None:
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)

            # Top-Pサンプリング
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))

            # サンプリング
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            if return_probabilities:
                probabilities.append(probs.gather(1, next_token))

            # 構文木を更新
            if next_token[0, 0].item() in self.syntax_checker.token_id_to_string:
                token_id = next_token[0, 0].item()
                syntax_tree = self.syntax_checker.update_syntax_tree(syntax_tree, token_id)

            # トークンを追加
            tokens = torch.cat([tokens, next_token], dim=1)

            # 終了トークンをチェック
            if (next_token == self.eos_token_id).all():
                break

        if return_probabilities:
            probabilities = torch.cat(probabilities, dim=1) if probabilities else None

        return tokens, probabilities

    def beam_search(
        self,
        context: Optional[torch.Tensor] = None,
        beam_width: int = 5,
        max_length: Optional[int] = None,
        length_penalty: float = 1.0,
        initial_tokens: Optional[torch.Tensor] = None,
        input_memory: Optional[torch.Tensor] = None,
        output_memory: Optional[torch.Tensor] = None
    ) -> List[Tuple[torch.Tensor, float]]:
        """ビームサーチでプログラムを生成（ProgramDecoderとの互換性のため）

        Args:
            context: コンテキスト [1, context_dim]
            beam_width: ビーム幅
            max_length: 最大長
            length_penalty: 長さペナルティ
            initial_tokens: 初期トークン [1, seq_len]（部分プログラム）
            input_memory: 入力エンコード特徴量（ProgramDecoderとの互換性のため受け入れるが、beam_searchメソッドでは未使用）
            output_memory: 出力エンコード特徴量（ProgramDecoderとの互換性のため受け入れるが、beam_searchメソッドでは未使用）

        Returns:
            candidates: (tokens, score)のリスト
        """
        # 注: input_memoryとoutput_memoryはProgramDecoderとの互換性のため受け入れるが、
        # beam_searchメソッドではsyntax_guided_beam_searchに渡していないため未使用
        if context is None:
            raise ValueError("contextは必須です")

        # syntax_guided_beam_searchを呼び出し、結果を変換
        results = self.syntax_guided_beam_search(
            context=context,
            beam_width=beam_width,
            max_length=max_length,
            initial_tokens=initial_tokens,
            syntax_constraints=None
        )

        # (tokens, score, syntax_tree) -> (tokens, score) に変換
        candidates = [(tokens, score) for tokens, score, _ in results]

        return candidates

    def syntax_guided_beam_search(
        self,
        context: torch.Tensor,
        beam_width: int = 5,
        max_length: Optional[int] = None,
        initial_tokens: Optional[torch.Tensor] = None,
        syntax_constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[torch.Tensor, float, SyntaxNode]]:
        """
        構文制約を考慮したビームサーチ

        Args:
            context: コンテキスト [batch, context_dim]
            beam_width: ビーム幅
            max_length: 最大長
            initial_tokens: 初期トークン [batch, seq_len]
            syntax_constraints: 構文制約

        Returns:
            List[Tuple[torch.Tensor, float, SyntaxNode]]: [(tokens, score, syntax_tree), ...]
        """
        self.eval()

        if max_length is None:
            max_length = self.max_length

        # 初期ビームを設定
        if initial_tokens is not None:
            beams = [(initial_tokens, 0.0, None)]  # (tokens, score, syntax_tree)
            initial_length = initial_tokens.size(1)
        else:
            initial_tokens = torch.tensor(
                [[self.bos_token_id]],
                dtype=torch.long,
                device=context.device
            )
            beams = [(initial_tokens, 0.0, None)]
            initial_length = 1

        completed = []
        remaining_steps = max_length - initial_length

        for step in range(remaining_steps):
            all_candidates = []

            for tokens, score, syntax_tree in beams:
                # 終了トークンをチェック
                if tokens[0, -1].item() == self.eos_token_id:
                    completed.append((tokens, score, syntax_tree))
                    continue

                # 順伝播
                with torch.no_grad():
                    token_logits, node_type_logits, _ = self.forward(
                        tokens, context, syntax_tree
                    )

                # 最後のトークンのlog確率
                log_probs = F.log_softmax(token_logits[:, -1, :], dim=-1)  # [1, vocab_size]

                # 構文制約を適用
                valid_token_ids = self.syntax_checker.get_valid_tokens(
                    tokens, syntax_tree, syntax_constraints
                )

                # 有効なトークンのみを考慮
                if valid_token_ids:
                    valid_log_probs = log_probs[0, valid_token_ids]
                    valid_token_ids_tensor = torch.tensor(valid_token_ids, device=tokens.device)
                else:
                    # 制約がない場合は、すべてのトークンを考慮
                    valid_log_probs = log_probs[0]
                    valid_token_ids_tensor = torch.arange(self.vocab_size, device=tokens.device)

                # Top-Kトークン
                top_k = min(beam_width * 2, len(valid_token_ids_tensor))
                top_log_probs, top_indices = torch.topk(valid_log_probs, top_k)
                top_token_ids = valid_token_ids_tensor[top_indices]

                for log_prob, token_id in zip(top_log_probs, top_token_ids):
                    token_id_item = token_id.item()

                    # 新しいトークン列を作成
                    new_tokens = torch.cat(
                        [tokens, token_id.unsqueeze(0).unsqueeze(0)],
                        dim=1
                    )

                    # 構文木を更新
                    new_syntax_tree = self.syntax_checker.update_syntax_tree(
                        syntax_tree, token_id_item
                    )

                    # スコアを計算
                    new_score = score + log_prob.item()

                    # 構文制約違反のペナルティ
                    if not self.syntax_checker.is_valid_syntax(new_syntax_tree):
                        new_score -= 10.0  # 大きなペナルティ

                    all_candidates.append((new_tokens, new_score, new_syntax_tree))

            # 上位beam_widthを選択
            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # すべて完了した場合
            if not beams:
                break

        # 残りのビームを完了リストに追加
        completed.extend(beams)

        # スコアでソート
        completed.sort(key=lambda x: x[1], reverse=True)

        return completed


class SyntaxConstraintChecker:
    """構文制約チェッカー"""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        初期化

        Args:
            tokenizer: トークナイザー（Noneの場合は自動生成）
        """
        self.tokenizer = tokenizer or Tokenizer()

        # 構文規則を定義
        self.syntax_rules = {
            'command': ['MOVE', 'ROTATE', 'FLIP', 'SET_COLOR', 'TELEPORT', 'SCALE', 'EXPAND', 'ALIGN', 'REVERSE', 'TILE'],
            'assignment': ['='],
            'loop': ['FOR', 'WHILE'],
            'condition': ['IF', 'ELSE', 'THEN', 'END'],
            'expression': ['GET_COLOR', 'GET_X', 'GET_Y', 'GET_WIDTH', 'GET_HEIGHT', 'GET_ASPECT_RATIO', 'GET_DENSITY', 'GET_CENTROID', 'GET_CENTER_X', 'GET_CENTER_Y', 'GET_MAX_X', 'GET_MAX_Y', 'GET_DIRECTION', 'GET_NEAREST'],
        }

        # トークンIDとトークン文字列のマッピング（後で設定）
        self.token_id_to_string: Dict[int, str] = {}
        self.token_string_to_id: Dict[str, int] = {}

    def set_token_mappings(
        self,
        token_id_to_string: Dict[int, str],
        token_string_to_id: Dict[str, int]
    ):
        """トークンIDと文字列のマッピングを設定"""
        self.token_id_to_string = token_id_to_string
        self.token_string_to_id = token_string_to_id

    def get_valid_tokens(
        self,
        tokens: torch.Tensor,
        syntax_tree: Optional[SyntaxNode],
        syntax_constraints: Optional[Dict[str, Any]],
        eos_token_id: int = 2
    ) -> List[int]:
        """
        現在の構文状態で有効なトークンを取得

        Args:
            tokens: 現在のトークン列
            syntax_tree: 構文木
            syntax_constraints: 構文制約
            eos_token_id: EOSトークンID

        Returns:
            List[int]: 有効なトークンIDのリスト（空の場合はすべて有効）
        """
        # トークン列を文字列に変換して構文解析を試みる
        if tokens.size(1) == 0:
            return []  # 空の場合はすべて有効

        try:
            # トークン列を文字列に変換
            token_strings = []
            for token_id in tokens[0].cpu().numpy().tolist():
                if token_id in self.token_id_to_string:
                    token_strings.append(self.token_id_to_string[token_id])
                else:
                    # マッピングがない場合は、トークンIDをそのまま使用
                    token_strings.append(str(token_id))

            # 部分的なプログラム文字列を構築
            partial_program = ' '.join(token_strings)

            # 構文解析を試みる（本格実装）
            try:
                parsed_tokens = self.tokenizer.tokenize(partial_program)
                parser = Parser(parsed_tokens)
                ast_nodes = parser.parse()

                # 構文解析が成功した場合、次の有効なトークンを推測
                valid_tokens = self._infer_valid_tokens_from_ast(ast_nodes, parsed_tokens)
                return valid_tokens
            except Exception:
                # 構文解析に失敗した場合、基本的な制約を適用
                # 最後のトークンに基づいて有効なトークンを推測
                if token_strings:
                    last_token = token_strings[-1]
                    return self._get_valid_tokens_after_token(last_token)
                return []  # すべて有効とする

        except Exception:
            # エラーが発生した場合、すべて有効とする
            return []

    def _is_command_name(self, token_id: int) -> bool:
        """トークンIDがコマンド名かどうかを判定"""
        if token_id not in self.token_id_to_string:
            return False
        token_str = self.token_id_to_string[token_id]
        return token_str in self.syntax_rules.get('command', [])

    def _get_argument_tokens(self) -> List[int]:
        """引数として有効なトークンを取得（本格実装）"""
        # 引数として有効なトークンIDを返す
        valid_ids = []
        for token_id, token_str in self.token_id_to_string.items():
            # 識別子、数値、文字列、プレースホルダーなど
            if (token_str.isdigit() or
                token_str.startswith('$') or
                token_str.startswith('"') or
                token_str.startswith("'") or
                (token_str.isidentifier() and token_str not in self.syntax_rules.get('command', []))):
                valid_ids.append(token_id)
        return valid_ids

    def _infer_valid_tokens_from_ast(
        self,
        ast_nodes: List[Any],
        tokens: List[Any]
    ) -> List[int]:
        """
        ASTから有効なトークンを推論（本格実装）

        Args:
            ast_nodes: 解析されたASTノード
            tokens: トークンリスト

        Returns:
            有効なトークンIDのリスト
        """
        if not ast_nodes or not tokens:
            return []

        # 最後のトークンのタイプを確認
        last_token = tokens[-1] if tokens else None
        if last_token is None:
            return []

        valid_ids = []

        # 最後のトークンのタイプに基づいて有効なトークンを推測
        if last_token.type == 'IDENTIFIER':
            # 識別子の後は、'='、'('、','などが有効
            valid_keywords = ['=', '(', ',', ']', ')']
            for keyword in valid_keywords:
                if keyword in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[keyword])
        elif last_token.type == 'LPAREN':
            # '('の後は、引数が有効
            valid_ids.extend(self._get_argument_tokens())
        elif last_token.type == 'RPAREN':
            # ')'の後は、';'、','、'='などが有効
            valid_keywords = [';', ',', '=', ']', ')']
            for keyword in valid_keywords:
                if keyword in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[keyword])
        elif last_token.type == 'ASSIGN':
            # '='の後は、式が有効
            valid_ids.extend(self._get_argument_tokens())
            # コマンドも有効
            for cmd in self.syntax_rules.get('command', []):
                if cmd in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[cmd])
        elif last_token.value in ['FOR', 'WHILE', 'IF']:
            # 制御構造の後は、'('が有効
            if '(' in self.token_string_to_id:
                valid_ids.append(self.token_string_to_id['('])
        elif last_token.value == 'THEN':
            # THENの後は、ステートメントが有効
            valid_ids.extend(self._get_argument_tokens())
            for cmd in self.syntax_rules.get('command', []):
                if cmd in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[cmd])

        # EOSトークンは常に有効
        if 2 in self.token_id_to_string:  # EOS token ID
            valid_ids.append(2)

        return list(set(valid_ids))  # 重複を除去

    def _get_valid_tokens_after_token(self, token_str: str) -> List[int]:
        """
        特定のトークンの後に有効なトークンを取得（本格実装）

        Args:
            token_str: トークン文字列

        Returns:
            有効なトークンIDのリスト
        """
        valid_ids = []

        # トークンの種類に基づいて有効なトークンを推測
        if token_str in self.syntax_rules.get('command', []):
            # コマンドの後は'('が有効
            if '(' in self.token_string_to_id:
                valid_ids.append(self.token_string_to_id['('])
        elif token_str == '(':
            # '('の後は引数が有効
            valid_ids.extend(self._get_argument_tokens())
        elif token_str == ')':
            # ')'の後は';'、','、'='などが有効
            valid_keywords = [';', ',', '=', ']', ')']
            for keyword in valid_keywords:
                if keyword in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[keyword])
        elif token_str == '=':
            # '='の後は式が有効
            valid_ids.extend(self._get_argument_tokens())
            for cmd in self.syntax_rules.get('command', []):
                if cmd in self.token_string_to_id:
                    valid_ids.append(self.token_string_to_id[cmd])
        elif token_str in ['FOR', 'WHILE', 'IF']:
            # 制御構造の後は'('が有効
            if '(' in self.token_string_to_id:
                valid_ids.append(self.token_string_to_id['('])

        # EOSトークンは常に有効
        if 2 in self.token_id_to_string:
            valid_ids.append(2)

        return list(set(valid_ids))  # 重複を除去

    def update_syntax_tree(
        self,
        syntax_tree: Optional[SyntaxNode],
        token_id: int
    ) -> Optional[SyntaxNode]:
        """
        構文木を更新（本格実装）

        Args:
            syntax_tree: 現在の構文木
            token_id: 新しいトークンID

        Returns:
            Optional[SyntaxNode]: 更新された構文木
        """
        if token_id not in self.token_id_to_string:
            return syntax_tree

        token_str = self.token_id_to_string[token_id]

        # トークンの種類に基づいて構文木を更新
        if token_str in self.syntax_rules.get('command', []):
            # コマンドノードを作成
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.COMMAND,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                # 現在のノードに子として追加
                syntax_tree.children.append(new_node)
                return syntax_tree
        elif token_str == '=':
            # 代入ノードを作成
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.ASSIGNMENT,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                syntax_tree.children.append(new_node)
                return syntax_tree
        elif token_str in ['FOR', 'WHILE']:
            # ループノードを作成
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.LOOP,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                syntax_tree.children.append(new_node)
                return syntax_tree
        elif token_str in ['IF', 'ELSE', 'THEN', 'END']:
            # 条件ノードを作成
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.CONDITION,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                syntax_tree.children.append(new_node)
                return syntax_tree
        elif token_str in self.syntax_rules.get('expression', []):
            # 式ノードを作成
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.EXPRESSION,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                syntax_tree.children.append(new_node)
                return syntax_tree
        elif token_str in ['(', ')', ',', '[', ']']:
            # 括弧や区切り文字は構文木に直接追加しない
            return syntax_tree
        else:
            # その他のトークン（引数など）
            new_node = SyntaxNode(
                node_type=SyntaxNodeType.ARGUMENT,
                value=token_str,
                children=[],
                metadata={'token_id': token_id}
            )
            if syntax_tree is None:
                return new_node
            else:
                # 最後の子ノードに追加（引数の場合）
                if syntax_tree.children:
                    syntax_tree.children[-1].children.append(new_node)
                else:
                    syntax_tree.children.append(new_node)
                return syntax_tree

    def is_valid_syntax(self, syntax_tree: Optional[SyntaxNode]) -> bool:
        """
        構文が有効かどうかをチェック（本格実装）

        Args:
            syntax_tree: 構文木

        Returns:
            bool: 構文が有効な場合True
        """
        if syntax_tree is None:
            return True

        # 構文木を再帰的に検証
        def validate_node(node: SyntaxNode, depth: int = 0) -> Tuple[bool, List[str]]:
            """ノードを再帰的に検証"""
            errors = []

            # 最大深度チェック
            if depth > 100:
                errors.append("構文木の深度が深すぎます")
                return False, errors

            # ノードタイプに応じた検証
            if node.node_type == SyntaxNodeType.COMMAND:
                # コマンドノードは値が必要
                if not node.value:
                    errors.append("コマンドノードに値がありません")
                # コマンド名が有効かチェック
                if node.value not in self.syntax_rules.get('command', []):
                    # 警告のみ（未知のコマンドも許可）
                    pass
            elif node.node_type == SyntaxNodeType.ASSIGNMENT:
                # 代入ノードは左辺と右辺が必要
                if len(node.children) < 2:
                    errors.append("代入ノードに左辺と右辺がありません")
            elif node.node_type == SyntaxNodeType.LOOP:
                # ループノードは条件と本体が必要
                if len(node.children) < 1:
                    errors.append("ループノードに条件がありません")
            elif node.node_type == SyntaxNodeType.CONDITION:
                # 条件ノードは条件式が必要
                if node.value == 'IF' and len(node.children) < 1:
                    errors.append("IFノードに条件式がありません")

            # 子ノードを再帰的に検証
            for child in node.children:
                is_valid, child_errors = validate_node(child, depth + 1)
                if not is_valid:
                    errors.extend(child_errors)

            return len(errors) == 0, errors

        is_valid, errors = validate_node(syntax_tree)
        return is_valid
