"""
DSL Selectorモジュール

どのDSLコマンドを使う可能性が高いかを予測する
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSLSelector(nn.Module):
    """DSL Selector（DSL使用確率予測）"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_dsl_commands: int = 100,
        dropout: float = 0.1
    ):
        """
        初期化

        Args:
            input_dim: 入力特徴量の次元（グリッド埋め込みの次元）
            hidden_dim: 隠れ層の次元
            num_dsl_commands: DSLコマンド数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.num_dsl_commands = num_dsl_commands

        # 特徴量投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # DSL使用確率予測ネットワーク
        self.dsl_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dsl_commands)
        )

    def forward(
        self,
        grid_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            grid_embedding: グリッド埋め込み [batch, embed_dim] または [batch, seq_len, embed_dim]

        Returns:
            torch.Tensor: DSL使用確率 [batch, num_dsl_commands] または [batch, seq_len, num_dsl_commands]
        """
        # 入力次元を確認
        if grid_embedding.dim() == 3:
            # [batch, seq_len, embed_dim] -> [batch, seq_len, hidden_dim]
            x = self.input_proj(grid_embedding)
            dsl_logits = self.dsl_mlp(x)  # [batch, seq_len, num_dsl_commands]
        else:
            # [batch, embed_dim] -> [batch, hidden_dim]
            x = self.input_proj(grid_embedding)
            dsl_logits = self.dsl_mlp(x)  # [batch, num_dsl_commands]

        return dsl_logits

    def predict_dsl_probabilities(
        self,
        grid_embedding: torch.Tensor
    ) -> Dict[str, float]:
        """
        DSL使用確率を予測

        Args:
            grid_embedding: グリッド埋め込み [batch, embed_dim]

        Returns:
            Dict[str, float]: {dsl_command: probability, ...}
        """
        self.eval()
        with torch.no_grad():
            dsl_logits = self.forward(grid_embedding)  # [batch, num_dsl_commands]
            dsl_probs = F.softmax(dsl_logits, dim=-1)  # [batch, num_dsl_commands]

        # バッチの最初の要素を使用
        probs = dsl_probs[0].cpu().numpy()

        # 辞書形式に変換（DSLコマンド名は仮想的なもの、実際にはvocabから取得）
        dsl_probs_dict = {
            f"dsl_{i}": float(probs[i])
            for i in range(self.num_dsl_commands)
        }

        return dsl_probs_dict

    def get_top_k_dsl(
        self,
        grid_embedding: torch.Tensor,
        k: int = 10
    ) -> List[tuple]:
        """
        上位K個のDSLコマンドを取得

        Args:
            grid_embedding: グリッド埋め込み [batch, embed_dim]
            k: 取得するDSLコマンド数

        Returns:
            List[tuple]: [(dsl_command, probability), ...] 確率の降順
        """
        dsl_probs_dict = self.predict_dsl_probabilities(grid_embedding)
        sorted_dsl = sorted(
            dsl_probs_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_dsl[:k]


class TokenToDSLMapper:
    """トークンIDとDSLコマンドのマッピング"""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        dsl_vocab: Optional[Dict[str, int]] = None
    ):
        """
        初期化

        Args:
            tokenizer: ProgramTokenizer
            dsl_vocab: DSLコマンドの語彙 {dsl_command: dsl_id}
        """
        self.tokenizer = tokenizer
        self.dsl_vocab = dsl_vocab or {}
        self.token_to_dsl: Dict[int, int] = {}
        self._build_mapping()

    def _build_mapping(self):
        """トークンIDとDSL IDのマッピングを構築"""
        if self.tokenizer is None:
            return

        # トークナイザーの語彙を取得
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        # DSLコマンドを識別（例: "MIRROR_X", "ROTATE", "FILL"など）
        dsl_keywords = [
            "MIRROR", "ROTATE", "FILL", "SHIFT", "COPY", "DELETE",
            "SELECT", "FILTER", "MAP", "REDUCE", "TRANSFORM"
        ]

        for token_str, token_id in vocab.items():
            # DSLコマンドかどうかを判定
            for dsl_keyword in dsl_keywords:
                if dsl_keyword in token_str.upper():
                    # DSL IDを取得または作成
                    if dsl_keyword not in self.dsl_vocab:
                        self.dsl_vocab[dsl_keyword] = len(self.dsl_vocab)
                    dsl_id = self.dsl_vocab[dsl_keyword]
                    self.token_to_dsl[token_id] = dsl_id
                    break

    def get_dsl_probability_for_token(
        self,
        token_id: int,
        dsl_probs: Dict[str, float]
    ) -> float:
        """
        トークンIDに対応するDSL確率を取得

        Args:
            token_id: トークンID
            dsl_probs: DSL使用確率辞書

        Returns:
            float: DSL確率（対応するDSLがない場合は0.0）
        """
        if token_id not in self.token_to_dsl:
            return 0.0

        dsl_id = self.token_to_dsl[token_id]
        dsl_key = f"dsl_{dsl_id}"
        return dsl_probs.get(dsl_key, 0.0)
