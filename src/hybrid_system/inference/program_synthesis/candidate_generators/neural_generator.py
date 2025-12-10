"""
ニューラルモデルベースの候補生成器（④深層学習ベース: グリッド→プログラム）

改善版: NGPS（Neural Guided Program Search）とDSL Selectorを統合
"""

from typing import List, Optional, Dict, Any
import torch

from src.hybrid_system.ir.serialization import template_string_to_sequence
from src.hybrid_system.ir.execution.template_executor import sequence_to_dsl

# 新規追加: NGPS関連
from src.hybrid_system.models.program_synthesis.dsl_selector import DSLSelector, TokenToDSLMapper
from src.hybrid_system.models.program_synthesis.neural_guided_program_search import NeuralGuidedProgramSearch
# 新規追加: Neural Mask Generator関連
from src.hybrid_system.models.program_synthesis.neural_mask_generator import (
    NeuralMaskGenerator,
    MaskBasedProgramGuider
)


class NeuralCandidateGenerator:
    """ニューラルモデルベースの候補生成器（グリッド→プログラム、改善版）"""

    def __init__(
        self,
        neural_model=None,
        tokenizer=None,
        enable_ngps: bool = True,
        enable_dsl_selector: bool = True,
        dsl_selector: Optional[DSLSelector] = None,
        dsl_selector_weight: float = 0.3,
        token_prob_weight: float = 0.6,
        exploration_bonus_weight: float = 0.1,
        dsl_filter_threshold: float = 0.1,
        enable_mask_generator: bool = False,
        mask_generator: Optional[NeuralMaskGenerator] = None
    ):
        """
        初期化

        Args:
            neural_model: ProgramSynthesisModel
            tokenizer: ProgramTokenizer
            enable_ngps: NGPSを有効化
            enable_dsl_selector: DSL Selectorを有効化
            dsl_selector: DSL Selectorモデル（Noneの場合は自動生成）
            dsl_selector_weight: DSL Selectorスコアの重み
            token_prob_weight: トークン確率スコアの重み
            exploration_bonus_weight: 探索ボーナスの重み
            dsl_filter_threshold: DSL確率のフィルタリング閾値
            enable_mask_generator: Neural Mask Generatorを有効化
            mask_generator: NeuralMaskGeneratorモデル（Noneの場合は自動生成）
        """
        self.neural_model = neural_model
        self.tokenizer = tokenizer
        self.enable_ngps = enable_ngps
        self.enable_dsl_selector = enable_dsl_selector

        # DSL Selector
        if self.enable_dsl_selector:
            if dsl_selector is None and self.neural_model is not None:
                # DSL Selectorを自動生成
                embed_dim = getattr(neural_model.program_config, 'grid_encoder_dim', 256)
                vocab_size = getattr(neural_model.program_config, 'vocab_size', 1000)
                self.dsl_selector = DSLSelector(
                    input_dim=embed_dim,
                    hidden_dim=128,
                    num_dsl_commands=min(100, vocab_size),
                    dropout=0.1
                )
                # デバイスを設定
                if hasattr(neural_model, 'device'):
                    self.dsl_selector = self.dsl_selector.to(neural_model.device)
            else:
                self.dsl_selector = dsl_selector

            # Token to DSL Mapper
            self.token_to_dsl_mapper = TokenToDSLMapper(tokenizer=tokenizer)
        else:
            self.dsl_selector = None
            self.token_to_dsl_mapper = None

        # NGPS
        if self.enable_ngps:
            self.ngps = NeuralGuidedProgramSearch(
                dsl_selector=self.dsl_selector,
                token_to_dsl_mapper=self.token_to_dsl_mapper,
                dsl_selector_weight=dsl_selector_weight,
                token_prob_weight=token_prob_weight,
                exploration_bonus_weight=exploration_bonus_weight,
                dsl_filter_threshold=dsl_filter_threshold
            )
        else:
            self.ngps = None

        # Neural Mask Generator
        self.enable_mask_generator = enable_mask_generator
        if self.enable_mask_generator:
            if mask_generator is not None:
                self.mask_generator = mask_generator
            else:
                # デフォルトのマスク生成器を作成
                embed_dim = getattr(neural_model.program_config, 'grid_encoder_dim', 256) if neural_model else 256
                self.mask_generator = NeuralMaskGenerator(
                    input_dim=embed_dim,
                    hidden_dim=128,
                    num_layers=3
                )
            self.mask_guider = MaskBasedProgramGuider(
                mask_generator=self.mask_generator
            )
        else:
            self.mask_generator = None
            self.mask_guider = None

    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        beam_width: int = 5,
        partial_program: Optional[str] = None,
        matching_result: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """ニューラルモデルベースの候補生成（改善版）

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド
            beam_width: ビーム幅（生成する候補数）
            partial_program: 部分プログラム（オプション）
            matching_result: オブジェクトマッチング結果（オプション）

        Returns:
            生成されたプログラムのリスト
        """
        candidates = []

        try:
            if self.neural_model is None or self.tokenizer is None:
                return candidates

            # Neural Mask Generatorを使用してプログラム探索をガイド（有効な場合）
            mask_guidance = None
            if self.enable_mask_generator and self.mask_guider:
                try:
                    import numpy as np
                    mask_guidance = self.mask_guider.guide_program_search(
                        np.array(input_grid),
                        np.array(output_grid),
                        grid_encoder=self.neural_model.grid_encoder if hasattr(self.neural_model, 'grid_encoder') else None
                    )
                except Exception as e:
                    # マスク生成でエラーが発生しても処理を継続
                    mask_guidance = None

            # グリッドをテンソルに変換
            input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)  # [1, H, W]
            output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)  # [1, H, W]

            # NGPSを使用する場合
            if self.enable_ngps and self.ngps is not None:
                # グリッドをエンコード（DSL Selector用）
                with torch.no_grad():
                    input_encoded = self.neural_model.grid_encoder(input_tensor)
                    output_encoded = self.neural_model.grid_encoder(output_tensor)
                    # プーリング
                    input_pooled = input_encoded.mean(dim=1)  # [1, embed_dim]
                    output_pooled = output_encoded.mean(dim=1)  # [1, embed_dim]
                    grid_embedding = (input_pooled + output_pooled) / 2  # [1, embed_dim]

                # コンテキストを準備
                input_pooled_full = input_encoded.mean(dim=1)
                output_pooled_full = output_encoded.mean(dim=1)
                grid_features = torch.cat([input_pooled_full, output_pooled_full], dim=-1)
                grid_context = self.neural_model.grid_fusion(grid_features)

                # 部分プログラムをトークン化
                initial_tokens = None
                if partial_program:
                    try:
                        token_ids = self.tokenizer.encode(partial_program, add_special_tokens=True)
                        initial_tokens = torch.tensor(
                            [token_ids],
                            dtype=torch.long,
                            device=grid_context.device
                        )
                    except Exception:
                        initial_tokens = None

                # NGPSによるビームサーチ
                max_length = getattr(self.neural_model.program_config, 'max_program_length', 512)
                vocab_size = getattr(self.neural_model.program_config, 'vocab_size', 1000)
                bos_token_id = getattr(self.neural_model.program_config, 'bos_token_id', 1)
                eos_token_id = getattr(self.neural_model.program_config, 'eos_token_id', 2)
                pad_token_id = getattr(self.neural_model.program_config, 'pad_token_id', 0)

                beam_results = self.ngps.guided_beam_search(
                    program_decoder=self.neural_model.program_decoder,
                    context=grid_context,
                    grid_embedding=grid_embedding,
                    beam_width=beam_width,
                    max_length=max_length,
                    initial_tokens=initial_tokens,
                    vocab_size=vocab_size,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id
                )
            else:
                # 通常のビームサーチ
                beam_results = self.neural_model.beam_search(
                    input_grid=input_tensor,
                    output_grid=output_tensor,
                    beam_width=beam_width,
                    partial_program=partial_program,
                    tokenizer=self.tokenizer
                )

            # トークンをプログラム文字列に変換
            for tokens, score in beam_results[:beam_width]:  # 上位beam_width個
                # BOS/EOS を除去
                token_ids = tokens[0].cpu().numpy().tolist()
                # BOS (1) と EOS (2) を除去
                token_ids = [tid for tid in token_ids if tid not in [1, 2, 0]]

                if token_ids:
                    template_string = self.tokenizer.decode(token_ids)
                    template_string = template_string.strip()
                    if not template_string:
                        continue
                    try:
                        sequence = template_string_to_sequence(template_string, task_id="inference")
                        program = sequence_to_dsl(sequence)
                    except Exception:
                        continue
                    if program:
                        candidates.append(program)

            return candidates

        except Exception as e:
            print(f"ニューラル候補生成エラー: {e}")
            return candidates
