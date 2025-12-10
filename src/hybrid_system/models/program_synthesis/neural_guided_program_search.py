"""
Neural Guided Program Search (NGPS)モジュール

DSL Selectorの出力を活用して、プログラム探索空間を大幅に削減
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np

from .dsl_selector import DSLSelector, TokenToDSLMapper


class NeuralGuidedProgramSearch:
    """Neural Guided Program Search"""

    def __init__(
        self,
        dsl_selector: Optional[DSLSelector] = None,
        token_to_dsl_mapper: Optional[TokenToDSLMapper] = None,
        dsl_selector_weight: float = 0.3,
        token_prob_weight: float = 0.6,
        exploration_bonus_weight: float = 0.1,
        dsl_filter_threshold: float = 0.1
    ):
        """
        初期化

        Args:
            dsl_selector: DSL Selectorモデル
            token_to_dsl_mapper: トークンIDとDSLコマンドのマッパー
            dsl_selector_weight: DSL Selectorスコアの重み
            token_prob_weight: トークン確率スコアの重み
            exploration_bonus_weight: 探索ボーナスの重み
            dsl_filter_threshold: DSL確率のフィルタリング閾値
        """
        self.dsl_selector = dsl_selector
        self.token_to_dsl_mapper = token_to_dsl_mapper
        self.dsl_selector_weight = dsl_selector_weight
        self.token_prob_weight = token_prob_weight
        self.exploration_bonus_weight = exploration_bonus_weight
        self.dsl_filter_threshold = dsl_filter_threshold

    def guided_beam_search(
        self,
        program_decoder: torch.nn.Module,
        context: torch.Tensor,
        grid_embedding: torch.Tensor,
        beam_width: int = 5,
        max_length: Optional[int] = None,
        initial_tokens: Optional[torch.Tensor] = None,
        vocab_size: int = 1000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        length_penalty: float = 1.0
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        NGPSによるビームサーチ

        Args:
            program_decoder: ProgramDecoderモデル
            context: コンテキスト [batch, context_dim]
            grid_embedding: グリッド埋め込み [batch, embed_dim]
            beam_width: ビーム幅
            max_length: 最大長
            initial_tokens: 初期トークン [batch, seq_len]
            vocab_size: 語彙サイズ
            bos_token_id: BOSトークンID
            eos_token_id: EOSトークンID
            pad_token_id: PADトークンID
            length_penalty: 長さペナルティ

        Returns:
            List[Tuple[torch.Tensor, float]]: [(tokens, score), ...]
        """
        program_decoder.eval()

        # DSL確率を取得
        dsl_probs = None
        if self.dsl_selector is not None:
            with torch.no_grad():
                dsl_logits = self.dsl_selector(grid_embedding)  # [batch, num_dsl_commands]
                dsl_probs_tensor = F.softmax(dsl_logits, dim=-1)  # [batch, num_dsl_commands]
                # バッチの最初の要素を使用
                dsl_probs_array = dsl_probs_tensor[0].cpu().numpy()
                dsl_probs = {
                    f"dsl_{i}": float(dsl_probs_array[i])
                    for i in range(len(dsl_probs_array))
                }

        # 初期ビームを設定
        if initial_tokens is not None:
            beams = [(initial_tokens, 0.0)]
            initial_length = initial_tokens.size(1)
        else:
            # BOSトークンから開始
            initial_tokens = torch.tensor(
                [[bos_token_id]],
                dtype=torch.long,
                device=context.device
            )
            beams = [(initial_tokens, 0.0)]
            initial_length = 1

        if max_length is None:
            max_length = 512

        completed = []
        remaining_steps = max_length - initial_length

        for step in range(remaining_steps):
            all_candidates = []

            for tokens, score in beams:
                # 終了トークンをチェック
                if tokens[0, -1].item() == eos_token_id:
                    completed.append((tokens, score))
                    continue

                # 順伝播
                with torch.no_grad():
                    logits, _ = program_decoder(tokens, context)

                # 最後のトークンのlog確率
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]

                # Top-Kトークン（通常のビームサーチ）
                top_k = min(beam_width * 2, vocab_size)  # より多くの候補を取得
                top_log_probs, top_indices = torch.topk(log_probs[0], top_k)

                for log_prob, token_id in zip(top_log_probs, top_indices):
                    token_id_item = token_id.item()

                    # DSL確率に基づいてフィルタリング
                    if dsl_probs is not None and self.token_to_dsl_mapper is not None:
                        dsl_prob = self.token_to_dsl_mapper.get_dsl_probability_for_token(
                            token_id_item,
                            dsl_probs
                        )
                        # DSL確率が閾値未満のトークンをスキップ
                        if dsl_prob < self.dsl_filter_threshold:
                            continue

                    # スコアを計算
                    token_prob_score = log_prob.item()

                    # DSL Selectorスコア
                    dsl_selector_score = 0.0
                    if dsl_probs is not None and self.token_to_dsl_mapper is not None:
                        dsl_prob = self.token_to_dsl_mapper.get_dsl_probability_for_token(
                            token_id_item,
                            dsl_probs
                        )
                        dsl_selector_score = np.log(dsl_prob + 1e-10)  # log確率に変換

                    # 探索ボーナス（多様性を確保）
                    exploration_bonus = 0.0
                    # 既存のトークン列に含まれていないトークンにボーナス
                    if token_id_item not in tokens[0].tolist():
                        exploration_bonus = 0.01

                    # 統合スコア
                    combined_score = (
                        self.token_prob_weight * token_prob_score +
                        self.dsl_selector_weight * dsl_selector_score +
                        self.exploration_bonus_weight * exploration_bonus
                    )

                    # 新しいトークン列を作成
                    new_tokens = torch.cat(
                        [tokens, token_id.unsqueeze(0).unsqueeze(0)],
                        dim=1
                    )
                    new_score = score + combined_score

                    all_candidates.append((new_tokens, new_score))

            # 上位beam_widthを選択
            if not all_candidates:
                break

            # 長さペナルティを適用してソート
            all_candidates.sort(
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True
            )
            beams = all_candidates[:beam_width]

            # すべて完了した場合
            if not beams:
                break

        # 残りのビームを完了リストに追加
        completed.extend(beams)

        # スコアでソート
        completed.sort(
            key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
            reverse=True
        )

        return completed

    def filter_tokens_by_dsl(
        self,
        token_log_probs: torch.Tensor,
        token_ids: torch.Tensor,
        dsl_probs: Dict[str, float],
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DSL確率に基づいてトークンをフィルタリング

        Args:
            token_log_probs: トークンのlog確率 [vocab_size]
            token_ids: トークンID [vocab_size]
            dsl_probs: DSL使用確率辞書
            top_k: 保持するトークン数

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (filtered_log_probs, filtered_token_ids)
        """
        if self.token_to_dsl_mapper is None:
            # マッパーがない場合は、通常のTop-Kを返す
            top_log_probs, top_indices = torch.topk(token_log_probs, top_k)
            return top_log_probs, token_ids[top_indices]

        # DSL確率を計算
        dsl_scores = torch.zeros_like(token_log_probs)
        for i, token_id in enumerate(token_ids):
            token_id_item = token_id.item()
            dsl_prob = self.token_to_dsl_mapper.get_dsl_probability_for_token(
                token_id_item,
                dsl_probs
            )
            dsl_scores[i] = np.log(dsl_prob + 1e-10)

        # 統合スコア
        combined_scores = (
            self.token_prob_weight * token_log_probs +
            self.dsl_selector_weight * dsl_scores
        )

        # Top-Kを選択
        top_scores, top_indices = torch.topk(combined_scores, top_k)

        return token_log_probs[top_indices], token_ids[top_indices]
