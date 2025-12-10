"""
オブジェクトベースプログラム合成モデル

オブジェクトリストからDSLプログラムを生成
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hybrid_system.models.base import BaseModel, ModelConfig, register_model
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
from src.data_systems.data_models.core.object import Object
from .object_encoder import ObjectEncoder
from .program_decoder import ProgramDecoder
from .cross_attention_fusion import InputOutputFusion
try:
    from .syntax_tree_decoder import SyntaxTreeDecoder
except ImportError:
    SyntaxTreeDecoder = None


@register_model("object_based_program_synthesis")
class ObjectBasedProgramSynthesisModel(BaseModel):
    """
    オブジェクトベースプログラム合成モデル

    入出力オブジェクトリストのペアを受け取り、DSLプログラムを生成する。

    アーキテクチャ:
    1. ObjectEncoder: 入出力オブジェクトリストをエンコード
    2. ProgramDecoder: エンコードされた表現からプログラムを生成
    """

    def __init__(self, config: ProgramSynthesisConfig):
        """初期化"""
        # BaseModelの初期化のためにModelConfigに変換
        model_config = ModelConfig(
            model_name=config.model_name,
            input_dim=config.grid_encoder_dim,  # オブジェクトエンコーダーの次元として使用
            output_dim=config.program_decoder_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            device=config.device
        )
        super().__init__(model_config)

        self.program_config = config

        # オブジェクトエンコーダー
        # Abstract Object Patternsを使用するかどうか（デフォルト: False）
        enable_abstract_patterns = getattr(config, 'enable_abstract_patterns', False)
        abstract_pattern_types = getattr(config, 'abstract_pattern_types', None)

        self.object_encoder = ObjectEncoder(
            embed_dim=config.grid_encoder_dim,  # オブジェクトエンコーダーの次元
            num_layers=config.grid_encoder_layers,
            num_heads=config.grid_encoder_heads,
            dropout=config.dropout,
            enable_abstract_patterns=enable_abstract_patterns,
            abstract_pattern_types=abstract_pattern_types
        )

        # プログラムデコーダー（構文木ベースまたは従来版）
        # 構文木ベースデコーダーを使用するかどうか（デフォルト: False）
        self.use_syntax_tree_decoder = getattr(config, 'use_syntax_tree_decoder', False)

        if self.use_syntax_tree_decoder:
            # 構文木ベースデコーダーを使用
            if SyntaxTreeDecoder is None:
                raise ImportError("SyntaxTreeDecoderをインポートできませんでした")
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

        # 入出力オブジェクトリストの融合層
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
            self.object_fusion = nn.Linear(
                config.grid_encoder_dim,
                config.program_decoder_dim
            )
        else:
            # 従来の単純な結合と線形変換
            self.cross_attention_fusion = None
            # オブジェクトリストをプーリングして固定次元に変換
            self.object_pooling = nn.AdaptiveAvgPool1d(1)  # シーケンス長を1に
            self.object_fusion = nn.Linear(
                config.grid_encoder_dim * 2,  # 入力と出力のオブジェクト埋め込みを結合
                config.program_decoder_dim
            )

        # モデルをデバイスに移動
        self.to(self.device)

    def forward(
        self,
        input_objects: List[Object],
        output_objects: List[Object],
        input_background_color: int,
        output_background_color: int,
        input_grid_width: int,
        input_grid_height: int,
        output_grid_width: int,
        output_grid_height: int,
        program_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """順伝播

        Args:
            input_objects: 入力オブジェクトリスト
            output_objects: 出力オブジェクトリスト
            input_background_color: 入力背景色
            output_background_color: 出力背景色
            input_grid_width: 入力グリッド幅
            input_grid_height: 入力グリッド高さ
            output_grid_width: 出力グリッド幅
            output_grid_height: 出力グリッド高さ
            program_tokens: プログラムトークン [batch, seq_len]（訓練時のみ）

        Returns:
            logits: プログラムのlogits [batch, seq_len, vocab_size]
            attention_weights: アテンション重み（オプション）
        """
        # オブジェクトリストをエンコード
        input_encoded = self.object_encoder(
            input_objects,
            input_background_color,
            input_grid_width,
            input_grid_height,
            color_roles=getattr(self, 'input_color_roles', None)
        )  # [1, num_input_objects, embed_dim]

        output_encoded = self.object_encoder(
            output_objects,
            output_background_color,
            output_grid_width,
            output_grid_height,
            color_roles=getattr(self, 'output_color_roles', None)
        )  # [1, num_output_objects, embed_dim]

        # 入出力を融合
        if self.use_cross_attention_fusion and self.cross_attention_fusion is not None:
            # Cross-Attention融合を使用
            # 出力側をQuery、入力側をKey/ValueとしてCross-Attention
            fused_output = self.cross_attention_fusion(
                input_embed=input_encoded,
                output_embed=output_encoded
            )  # [1, num_output_objects, embed_dim]

            # 平均プーリングしてコンテキストに変換
            object_context_pooled = fused_output.mean(dim=1)  # [1, embed_dim]
            object_context = self.object_fusion(object_context_pooled)  # [1, decoder_dim]
        else:
            # 従来の方法: オブジェクトリストをプーリングして固定次元に変換
            # [1, num_objects, embed_dim] -> [1, embed_dim]
            input_pooled = input_encoded.mean(dim=1)  # [1, embed_dim]
            output_pooled = output_encoded.mean(dim=1)  # [1, embed_dim]

            # 入出力を融合
            object_features = torch.cat([input_pooled, output_pooled], dim=-1)  # [1, embed_dim*2]
            object_context = self.object_fusion(object_features)  # [1, decoder_dim]

        # プログラムをデコード
        # Cross-Attention強化: 入力と出力のエンコード特徴量を直接メモリとして渡す
        use_enhanced_memory = self.use_cross_attention_fusion and self.cross_attention_fusion is not None

        if program_tokens is not None:
            # 訓練時: teacher forcing
            if use_enhanced_memory:
                logits, attention_weights = self.program_decoder(
                    program_tokens,
                    context=object_context,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                logits, attention_weights = self.program_decoder(
                    program_tokens,
                    context=object_context
                )
        else:
            # 推論時: 自己回帰生成
            if use_enhanced_memory:
                logits, attention_weights = self.program_decoder.generate(
                    context=object_context,
                    max_length=self.program_config.max_program_length,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                logits, attention_weights = self.program_decoder.generate(
                    context=object_context,
                    max_length=self.program_config.max_program_length
                )

        return logits, attention_weights

    def compute_loss(
        self,
        input_objects: List[Object],
        output_objects: List[Object],
        input_background_color: int,
        output_background_color: int,
        input_grid_width: int,
        input_grid_height: int,
        output_grid_width: int,
        output_grid_height: int,
        program_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """損失を計算

        Args:
            input_objects: 入力オブジェクトリスト
            output_objects: 出力オブジェクトリスト
            input_background_color: 入力背景色
            output_background_color: 出力背景色
            input_grid_width: 入力グリッド幅
            input_grid_height: 入力グリッド高さ
            output_grid_width: 出力グリッド幅
            output_grid_height: 出力グリッド高さ
            program_tokens: プログラムトークン [batch, seq_len]
            target_tokens: ターゲットトークン [batch, seq_len]

        Returns:
            loss: 損失値
        """
        logits, _ = self.forward(
            input_objects,
            output_objects,
            input_background_color,
            output_background_color,
            input_grid_width,
            input_grid_height,
            output_grid_width,
            output_grid_height,
            program_tokens=program_tokens
        )

        # クロスエントロピー損失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tokens.view(-1),
            ignore_index=self.program_decoder.pad_token_id
        )

        return loss

    def beam_search(
        self,
        input_objects: List[Object],
        output_objects: List[Object],
        input_background_color: int,
        output_background_color: int,
        input_grid_width: int,
        input_grid_height: int,
        output_grid_width: int,
        output_grid_height: int,
        beam_width: int = 5,
        max_length: Optional[int] = None,
        partial_program: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        category_var_mapping: Optional[Dict[str, str]] = None,
        input_graph_encoded: Optional[torch.Tensor] = None,
        output_graph_encoded: Optional[torch.Tensor] = None,
        input_canonicalized: Optional[List[Any]] = None,
        output_canonicalized: Optional[List[Any]] = None,
        input_color_roles: Optional[Dict[int, str]] = None,
        output_color_roles: Optional[Dict[int, str]] = None
    ) -> List[Tuple[torch.Tensor, float]]:
        """ビームサーチでプログラムを生成

        Args:
            input_objects: 入力オブジェクトリスト
            output_objects: 出力オブジェクトリスト
            input_background_color: 入力背景色
            output_background_color: 出力背景色
            input_grid_width: 入力グリッド幅
            input_grid_height: 入力グリッド高さ
            output_grid_width: 出力グリッド幅
            output_grid_height: 出力グリッド高さ
            beam_width: ビーム幅
            max_length: 最大長
            partial_program: 部分プログラム（オプション）
            tokenizer: プログラムトークナイザー（partial_programが提供される場合に必要）
            category_var_mapping: カテゴリIDと変数名の対応関係 {category_id: variable_name}（オプション）
                - 部分プログラムの変数名とカテゴリIDの対応を提供
                - 将来的にプログラム生成のガイダンスとして使用可能
            input_graph_encoded: 入力グラフのエンコードされた特徴量 [1, num_nodes, embed_dim]（オプション）
            output_graph_encoded: 出力グラフのエンコードされた特徴量 [1, num_nodes, embed_dim]（オプション）
            input_canonicalized: 正規化された入力オブジェクトリスト（オプション）
            output_canonicalized: 正規化された出力オブジェクトリスト（オプション）

        Returns:
            candidates: 候補プログラムのリスト（トークン、スコア）
        """
        self.eval()

        with torch.no_grad():
            # オブジェクトリストをエンコード（グラフ特徴量と正規化オブジェクトがある場合は統合）
            input_encoded = self.object_encoder(
                input_objects,
                input_background_color,
                input_grid_width,
                input_grid_height,
                graph_encoded=input_graph_encoded,
                canonicalized_objects=input_canonicalized,
                color_roles=input_color_roles
            )

            output_encoded = self.object_encoder(
                output_objects,
                output_background_color,
                output_grid_width,
                output_grid_height,
                graph_encoded=output_graph_encoded,
                canonicalized_objects=output_canonicalized,
                color_roles=output_color_roles
            )

            # 融合
            if self.use_cross_attention_fusion and self.cross_attention_fusion is not None:
                # Cross-Attention融合を使用
                fused_output = self.cross_attention_fusion(
                    input_embed=input_encoded,
                    output_embed=output_encoded
                )
                object_context_pooled = fused_output.mean(dim=1)
                object_context = self.object_fusion(object_context_pooled)
            else:
                # 従来の方法: プーリング
                input_pooled = input_encoded.mean(dim=1)
                output_pooled = output_encoded.mean(dim=1)

                # 融合
                object_features = torch.cat([input_pooled, output_pooled], dim=-1)
                object_context = self.object_fusion(object_features)

            # 部分プログラムをトークン化
            initial_tokens = None
            if partial_program and tokenizer:
                try:
                    token_ids = tokenizer.encode(partial_program, add_special_tokens=True)
                    # BOSは既に含まれているので、そのまま使用
                    initial_tokens = torch.tensor([token_ids], dtype=torch.long, device=object_context.device)
                except Exception as e:
                    print(f"部分プログラムのトークン化エラー: {e}")
                    initial_tokens = None

            # category_var_mappingを活用
            # 注意: category_var_mappingは部分プログラムとカテゴリ情報の対応関係を追跡するためのものである
            #       部分プログラムの続きから生成する場合、変数名が一致するのは当然であるため、
            #       ビームサーチ時のスコア調整には使用しない
            #       既にcategoriesはObject Graph構築に使用されており、これがカテゴリ情報の主な活用方法である
            # 将来的な拡張案:
            # - カテゴリ情報を考慮したプログラム生成のガイダンス（モデル側での実装が必要）
            # - 生成されたプログラムの意味的な検証（カテゴリ情報と一致するか）
            if category_var_mapping:
                # デバッグ用（必要に応じて有効化）
                # print(f"[DEBUG] category_var_mapping: {category_var_mapping}")
                pass

            # ビームサーチ
            if max_length is None:
                max_length = self.program_config.max_program_length

            # Cross-Attention強化: 入力と出力のエンコード特徴量を直接メモリとして渡す
            use_enhanced_memory = self.use_cross_attention_fusion and self.cross_attention_fusion is not None

            if use_enhanced_memory:
                candidates = self.program_decoder.beam_search(
                    context=object_context,
                    beam_width=beam_width,
                    max_length=max_length,
                    initial_tokens=initial_tokens,
                    input_memory=input_encoded,
                    output_memory=output_encoded
                )
            else:
                candidates = self.program_decoder.beam_search(
                    context=object_context,
                    beam_width=beam_width,
                    max_length=max_length,
                    initial_tokens=initial_tokens
                )

        return candidates
