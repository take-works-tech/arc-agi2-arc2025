"""
プログラムデコーダ

DSLプログラムを生成するトランスフォーマーベースのデコーダ
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgramDecoder(nn.Module):
    """
    プログラムデコーダ

    エンコードされたグリッド表現からDSLプログラムを生成
    """

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
        """初期化"""
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # トークンの埋め込み
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

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

        # 出力層
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # レイヤー正規化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        input_memory: Optional[torch.Tensor] = None,
        output_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """順伝播

        Args:
            tokens: プログラムトークン [batch, seq_len]
            context: コンテキスト [batch, context_dim] または [batch, 1, context_dim]（従来の方法）
            input_memory: 入力エンコード特徴量 [batch, seq_len_input, embed_dim]（強化版、オプション）
            output_memory: 出力エンコード特徴量 [batch, seq_len_output, embed_dim]（強化版、オプション）

        Returns:
            logits: 出力logits [batch, seq_len, vocab_size]
            attention_weights: アテンション重み（オプション）
        """
        batch_size, seq_len = tokens.shape

        # トークンの埋め込み
        token_embed = self.token_embedding(tokens)  # [batch, seq_len, embed_dim]

        # 位置エンコーディング
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.position_embedding(positions)  # [batch, seq_len, embed_dim]

        # 埋め込みを結合
        embed = token_embed + pos_embed
        embed = self.dropout(embed)

        # メモリを準備（強化版: 入力と出力のエンコード特徴量を直接使用）
        if input_memory is not None and output_memory is not None:
            # 入力と出力のエンコード特徴量を連結してメモリとして使用
            # これにより、デコーダーが入力と出力の詳細な情報を直接参照できる
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
                # [batch, dim] -> [batch, 1, dim]
                context = context.unsqueeze(1)
            # 次元が一致することを確認
            if context.size(-1) != self.embed_dim:
                context = self.context_proj(context)  # [batch, 1, embed_dim]
            else:
                # 既に正しい次元の場合はそのまま使用
                pass
            memory = context
        else:
            # デフォルトコンテキスト
            memory = torch.zeros(batch_size, 1, self.embed_dim, device=tokens.device)

        # 因果マスク（未来のトークンを見ないようにする）
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
        logits = self.output_proj(decoded)  # [batch, seq_len, vocab_size]

        return logits, None

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
        """自己回帰的にプログラムを生成

        Args:
            context: コンテキスト [batch, context_dim]
            max_length: 最大長
            temperature: 温度パラメータ
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング（nucleus sampling）
            return_probabilities: 確率を返すかどうか

        Returns:
            tokens: 生成されたトークン [batch, seq_len]
            probabilities: 各トークンの確率（オプション）
        """
        if max_length is None:
            max_length = self.max_length

        # contextがNoneの場合はデフォルトコンテキストを作成
        if context is None and input_memory is None and output_memory is None:
            raise ValueError("context, input_memory, output_memoryのいずれかは必須です")

        # デバイスを取得
        if context is not None:
            device = context.device
            batch_size = context.size(0)
        elif input_memory is not None:
            device = input_memory.device
            batch_size = input_memory.size(0)
        elif output_memory is not None:
            device = output_memory.device
            batch_size = output_memory.size(0)
        else:
            raise ValueError("デバイスを取得できません")

        # 開始トークン
        tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        probabilities = [] if return_probabilities else None

        for _ in range(max_length - 1):
            # 順伝播
            logits, _ = self.forward(tokens, context, input_memory, output_memory)

            # 最後のトークンのlogitsを取得
            next_logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Top-Kサンプリング
            if top_k is not None:
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)

            # Top-Pサンプリング
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 累積確率がtop_pを超えるトークンを除外
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

            # トークンを追加
            tokens = torch.cat([tokens, next_token], dim=1)

            # 終了トークンをチェック
            if (next_token == self.eos_token_id).all():
                break

        if return_probabilities:
            probabilities = torch.cat(probabilities, dim=1)  # [batch, seq_len]

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
        """ビームサーチでプログラムを生成

        Args:
            context: コンテキスト [1, context_dim]
            beam_width: ビーム幅
            max_length: 最大長
            length_penalty: 長さペナルティ
            initial_tokens: 初期トークン（部分プログラム）[1, seq_len]（オプション）

        Returns:
            candidates: (tokens, score)のリスト
        """
        if max_length is None:
            max_length = self.max_length

        # contextがNoneの場合はデフォルトコンテキストを作成
        if context is None and input_memory is None and output_memory is None:
            raise ValueError("context, input_memory, output_memoryのいずれかは必須です")

        # デバイスを取得
        if context is not None:
            device = context.device
        elif input_memory is not None:
            device = input_memory.device
        elif output_memory is not None:
            device = output_memory.device
        else:
            raise ValueError("デバイスを取得できません")

        # 初期ビーム: (tokens, score)
        initial_length = 0
        if initial_tokens is not None:
            # 部分プログラムが提供された場合、それを初期トークンとして使用
            # initial_tokensは既に[BOS, token1, token2, ...]の形式を想定
            if initial_tokens.dim() == 1:
                initial_tokens = initial_tokens.unsqueeze(0)  # [1, seq_len]に変換
            initial_tokens = initial_tokens.to(device)
            initial_length = initial_tokens.shape[1]
            # EOSが既に含まれている場合は完了として扱う
            if initial_tokens[0, -1].item() == self.eos_token_id:
                return [(initial_tokens, 0.0)]
            beams = [(initial_tokens, 0.0)]
        else:
            # 通常の場合はBOSトークンから開始
            beams = [(torch.ones(1, 1, dtype=torch.long, device=device) * self.bos_token_id, 0.0)]
            initial_length = 1

        completed = []

        # 部分プログラムの長さを考慮して、残りのステップ数を計算
        remaining_steps = max_length - initial_length

        for step in range(remaining_steps):
            all_candidates = []

            for tokens, score in beams:
                # 終了トークンをチェック
                if tokens[0, -1].item() == 2:  # <EOS>
                    completed.append((tokens, score))
                    continue

                # 順伝播
                logits, _ = self.forward(tokens, context, input_memory, output_memory)

                # 最後のトークンのlog確率
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]

                # Top-Kトークン
                top_log_probs, top_indices = torch.topk(log_probs[0], beam_width)

                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_tokens = torch.cat([tokens, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    all_candidates.append((new_tokens, new_score))

            # 上位beam_widthを選択
            all_candidates.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
            beams = all_candidates[:beam_width]

            # すべて完了した場合
            if not beams:
                break

        # 残りのビームを完了リストに追加
        completed.extend(beams)

        # スコアでソート
        completed.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)

        return completed
