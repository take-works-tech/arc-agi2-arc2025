"""
プログラム合成モデル

入出力ペアからDSLプログラムを生成
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hybrid_system.models.base import BaseModel, ModelConfig, register_model
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
from .grid_encoder import GridEncoder
from .program_decoder import ProgramDecoder
from .cross_attention_fusion import InputOutputFusion
from .syntax_tree_decoder import SyntaxTreeDecoder


@register_model("program_synthesis")
class ProgramSynthesisModel(BaseModel):
    """
    プログラム合成モデル

    入出力グリッドのペアを受け取り、DSLプログラムを生成する。

    アーキテクチャ:
    1. GridEncoder: 入出力グリッドをエンコード
    2. ProgramDecoder: エンコードされた表現からプログラムを生成
    """

    def __init__(self, config: ProgramSynthesisConfig):
        """初期化"""
        # BaseModelの初期化のためにModelConfigに変換
        model_config = ModelConfig(
            model_name=config.model_name,
            input_dim=config.grid_encoder_dim,
            output_dim=config.program_decoder_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            device=config.device
        )
        super().__init__(model_config)

        self.program_config = config

        # グリッドエンコーダ
        self.grid_encoder = GridEncoder(
            input_channels=10,  # 10色
            embed_dim=config.grid_encoder_dim,
            num_layers=config.grid_encoder_layers,
            num_heads=config.grid_encoder_heads,
            dropout=config.dropout
        )

        # プログラムデコーダ（構文木ベースまたは従来版）
        # 構文木ベースデコーダーを使用するかどうか（デフォルト: False）
        self.use_syntax_tree_decoder = getattr(config, 'use_syntax_tree_decoder', False)

        if self.use_syntax_tree_decoder:
            # 構文木ベースデコーダーを使用
            self.program_decoder = SyntaxTreeDecoder(
                vocab_size=config.vocab_size,
                embed_dim=config.program_decoder_dim,
                num_layers=config.program_decoder_layers,
                num_heads=config.program_decoder_heads,
                dropout=config.dropout,
                max_length=config.max_program_length,
                bos_token_id=getattr(config, 'bos_token_id', 1),
                eos_token_id=getattr(config, 'eos_token_id', 2),
                pad_token_id=getattr(config, 'pad_token_id', 0)
            )
        else:
            # 従来のデコーダーを使用
            self.program_decoder = ProgramDecoder(
                vocab_size=config.vocab_size,
                embed_dim=config.program_decoder_dim,
                num_layers=config.program_decoder_layers,
                num_heads=config.program_decoder_heads,
                dropout=config.dropout,
                max_length=config.max_program_length,
                bos_token_id=getattr(config, 'bos_token_id', 1),
                eos_token_id=getattr(config, 'eos_token_id', 2),
                pad_token_id=getattr(config, 'pad_token_id', 0)
            )

        # 入出力グリッドの融合層
        # Cross-Attention融合を使用するかどうか（デフォルト: False）
        self.use_cross_attention_fusion = getattr(config, 'use_cross_attention_fusion', False)

        if self.use_cross_attention_fusion:
            # Cross-Attention融合を使用
            self.cross_attention_fusion = InputOutputFusion(
                embed_dim=config.grid_encoder_dim,
                num_layers=getattr(config, 'cross_attention_layers', 2),
                num_heads=config.grid_encoder_heads,
                dropout=config.dropout
            )
            # 最終的なコンテキスト投影
            self.grid_fusion = nn.Linear(
                config.grid_encoder_dim,
                config.program_decoder_dim
            )
        else:
            # 従来の単純な結合と線形変換
            self.cross_attention_fusion = None
            self.grid_fusion = nn.Linear(
                config.grid_encoder_dim * 2,
                config.program_decoder_dim
            )

        # モデルをデバイスに移動
        self.to(self.device)

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        program_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """順伝播

        Args:
            input_grid: 入力グリッド [batch, height, width]
            output_grid: 出力グリッド [batch, height, width]
            program_tokens: プログラムトークン [batch, seq_len]（訓練時のみ）

        Returns:
            logits: プログラムのlogits [batch, seq_len, vocab_size]
            attention_weights: アテンション重み（オプション）
        """
        # グリッドをエンコード
        input_encoded = self.grid_encoder(input_grid)   # [batch, seq_len, dim]
        output_encoded = self.grid_encoder(output_grid)  # [batch, seq_len, dim]

        # 入出力を融合
        if self.use_cross_attention_fusion and self.cross_attention_fusion is not None:
            # Cross-Attention融合を使用
            # 出力側をQuery、入力側をKey/ValueとしてCross-Attention
            fused_output = self.cross_attention_fusion(
                input_embed=input_encoded,
                output_embed=output_encoded
            )  # [batch, seq_len, dim]

            # 平均プーリングしてコンテキストに変換
            grid_context_pooled = fused_output.mean(dim=1)  # [batch, dim]
            grid_context = self.grid_fusion(grid_context_pooled)  # [batch, decoder_dim]
        else:
            # 従来の方法: シーケンス長を合わせる（平均プーリング）
            input_pooled = input_encoded.mean(dim=1)   # [batch, dim]
            output_pooled = output_encoded.mean(dim=1)  # [batch, dim]

            grid_features = torch.cat([input_pooled, output_pooled], dim=-1)  # [batch, dim*2]
            grid_context = self.grid_fusion(grid_features)  # [batch, decoder_dim]

        # プログラムをデコード
        # Cross-Attention強化: 入力と出力のエンコード特徴量を直接メモリとして渡す
        use_enhanced_memory = self.use_cross_attention_fusion and self.cross_attention_fusion is not None

        if program_tokens is not None:
            # 訓練時: teacher forcing
            if use_enhanced_memory:
                logits, attention_weights = self.program_decoder(
                    program_tokens,
                    context=grid_context,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                logits, attention_weights = self.program_decoder(
                    program_tokens,
                    context=grid_context
                )
        else:
            # 推論時: 自己回帰生成
            if use_enhanced_memory:
                logits, attention_weights = self.program_decoder.generate(
                    context=grid_context,
                    max_length=self.program_config.max_program_length,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                logits, attention_weights = self.program_decoder.generate(
                    context=grid_context,
                    max_length=self.program_config.max_program_length
                )

        return logits, attention_weights

    def compute_loss(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        program_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """損失計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            program_tokens: プログラムトークン（入力）
            target_tokens: ターゲットトークン（出力）

        Returns:
            損失値
        """
        # 順伝播
        logits, _ = self.forward(input_grid, output_grid, program_tokens)

        # クロスエントロピー損失
        # logits: [batch, seq_len, vocab_size]
        # target_tokens: [batch, seq_len]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=0  # パディングトークンを無視
        )

        return loss

    def compute_metrics(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        program_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> Dict[str, float]:
        """評価メトリクスの計算

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            program_tokens: プログラムトークン
            target_tokens: ターゲットトークン

        Returns:
            評価メトリクス
        """
        # 順伝播
        logits, _ = self.forward(input_grid, output_grid, program_tokens)

        # 予測トークン
        predicted_tokens = logits.argmax(dim=-1)

        # 正解率
        mask = target_tokens != 0  # パディングを除外
        accuracy = (predicted_tokens == target_tokens)[mask].float().mean().item()

        # トークン単位の正解率
        token_accuracy = (predicted_tokens == target_tokens).float().mean().item()

        # シーケンス単位の正解率
        sequence_accuracy = (predicted_tokens == target_tokens).all(dim=-1).float().mean().item()

        return {
            'accuracy': accuracy,
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy
        }

    def generate_program(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """プログラムを生成

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            max_length: 最大長（オプション）

        Returns:
            program_tokens: 生成されたプログラムトークン
            probabilities: 各トークンの確率
        """
        self.eval()

        with torch.no_grad():
            # グリッドをエンコード
            input_encoded = self.grid_encoder(input_grid)
            output_encoded = self.grid_encoder(output_grid)

            # 融合
            if self.use_cross_attention_fusion and self.cross_attention_fusion is not None:
                # Cross-Attention融合を使用
                fused_output = self.cross_attention_fusion(
                    input_embed=input_encoded,
                    output_embed=output_encoded
                )
                grid_context_pooled = fused_output.mean(dim=1)
                grid_context = self.grid_fusion(grid_context_pooled)
            else:
                # 従来の方法
                input_pooled = input_encoded.mean(dim=1)
                output_pooled = output_encoded.mean(dim=1)
                grid_features = torch.cat([input_pooled, output_pooled], dim=-1)
                grid_context = self.grid_fusion(grid_features)

            # プログラムを生成
            if max_length is None:
                max_length = self.program_config.max_program_length

            # Cross-Attention強化: 入力と出力のエンコード特徴量を直接メモリとして渡す
            use_enhanced_memory = self.use_cross_attention_fusion and self.cross_attention_fusion is not None

            if use_enhanced_memory:
                program_tokens, probabilities = self.program_decoder.generate(
                    context=grid_context,
                    max_length=max_length,
                    return_probabilities=True,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                program_tokens, probabilities = self.program_decoder.generate(
                    context=grid_context,
                    max_length=max_length,
                    return_probabilities=True
                )

        return program_tokens, probabilities

    def beam_search(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        beam_width: int = 5,
        max_length: Optional[int] = None,
        partial_program: Optional[str] = None,
        tokenizer: Optional[Any] = None
    ) -> list:
        """ビームサーチでプログラムを生成

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            beam_width: ビーム幅
            max_length: 最大長
            partial_program: 部分プログラム（オプション）
            tokenizer: プログラムトークナイザー（partial_programが提供される場合に必要）

        Returns:
            candidates: 候補プログラムのリスト（トークン、スコア）
        """
        self.eval()

        with torch.no_grad():
            # グリッドをエンコード
            input_encoded = self.grid_encoder(input_grid)
            output_encoded = self.grid_encoder(output_grid)

            # 融合
            if self.use_cross_attention_fusion and self.cross_attention_fusion is not None:
                # Cross-Attention融合を使用
                fused_output = self.cross_attention_fusion(
                    input_embed=input_encoded,
                    output_embed=output_encoded
                )
                grid_context_pooled = fused_output.mean(dim=1)
                grid_context = self.grid_fusion(grid_context_pooled)
            else:
                # 従来の方法
                input_pooled = input_encoded.mean(dim=1)
                output_pooled = output_encoded.mean(dim=1)
                grid_features = torch.cat([input_pooled, output_pooled], dim=-1)
                grid_context = self.grid_fusion(grid_features)

            # 部分プログラムをトークン化
            initial_tokens = None
            if partial_program and tokenizer:
                try:
                    token_ids = tokenizer.encode(partial_program, add_special_tokens=True)
                    # BOSは既に含まれているので、そのまま使用
                    initial_tokens = torch.tensor([token_ids], dtype=torch.long, device=grid_context.device)
                except Exception as e:
                    print(f"部分プログラムのトークン化エラー: {e}")
                    initial_tokens = None

            # ビームサーチ
            if max_length is None:
                max_length = self.program_config.max_program_length

            # Cross-Attention強化: 入力と出力のエンコード特徴量を直接メモリとして渡す
            use_enhanced_memory = self.use_cross_attention_fusion and self.cross_attention_fusion is not None

            if use_enhanced_memory:
                candidates = self.program_decoder.beam_search(
                    context=grid_context,
                    beam_width=beam_width,
                    max_length=max_length,
                    initial_tokens=initial_tokens,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                candidates = self.program_decoder.beam_search(
                    context=grid_context,
                    beam_width=beam_width,
                    max_length=max_length,
                    initial_tokens=initial_tokens
                )

        return candidates
