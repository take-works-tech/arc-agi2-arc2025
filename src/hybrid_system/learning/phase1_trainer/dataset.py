"""
DataPairDataset

Phase1訓練用のデータセットクラス
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np

from src.hybrid_system.core.data_structures import DataPair
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from src.hybrid_system.ir.parser.relabel_transformer import RelabelTransformer
from src.hybrid_system.ir.serialization.template_serialization import sequence_to_template_string


class DataPairDataset(Dataset):
    """
    DataPair用のPyTorchデータセット

    入出力グリッドとプログラムをTensorに変換
    """

    def __init__(
        self,
        data_pairs: List[DataPair],
        tokenizer: ProgramTokenizer,
        max_program_length: int = 512,
        max_grid_size: int = 30,
        use_ir_templates: bool = True
    ):
        """
        初期化

        Args:
            data_pairs: DataPairのリスト
            tokenizer: プログラムトークナイザー
            max_program_length: プログラムの最大長
            max_grid_size: グリッドの最大サイズ
            use_ir_templates: IRテンプレートを使用するかどうか（デフォルト: True）
        """
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_program_length = max_program_length
        self.max_grid_size = max_grid_size
        self.use_ir_templates = use_ir_templates

        # IRテンプレート変換用
        if self.use_ir_templates:
            self.relabel_transformer = RelabelTransformer()
            # IRSequenceをキャッシュ
            self._ir_cache = {}

    def __len__(self) -> int:
        """データセットサイズ"""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データを取得

        Args:
            idx: インデックス

        Returns:
            - input_grid: 入力グリッド [height, width]
            - output_grid: 出力グリッド [height, width]
            - program_tokens: プログラムトークン（入力） [seq_len-1]
            - target_tokens: ターゲットトークン（出力） [seq_len-1]
            - attention_mask: アテンションマスク [seq_len-1]
        """
        pair = self.data_pairs[idx]

        # グリッドをTensorに変換
        input_grid = torch.tensor(pair.input, dtype=torch.long)
        output_grid = torch.tensor(pair.output, dtype=torch.long)

        # グリッドをパディング（必要に応じて）
        input_grid = self._pad_grid(input_grid, self.max_grid_size, self.max_grid_size)
        output_grid = self._pad_grid(output_grid, self.max_grid_size, self.max_grid_size)

        # プログラムをトークン化
        if self.use_ir_templates and pair.program:
            # IRテンプレートを使用する場合
            program_text = self._get_ir_template(idx, pair)
            if program_text:
                program_tokens = self.tokenizer.encode(program_text, add_special_tokens=True)
            else:
                # 変換失敗時はフォールバック
                program_tokens = self.tokenizer.encode(pair.program, add_special_tokens=True)
        elif pair.program:
            # 従来のDSL文字列を使用
            program_tokens = self.tokenizer.encode(pair.program, add_special_tokens=True)
        else:
            # 空文字列の場合はBOSとEOSのみ
            program_tokens = [
                self.tokenizer.SPECIAL_TOKENS['<BOS>'],
                self.tokenizer.SPECIAL_TOKENS['<EOS>']
            ]

        # プログラムをTensorに変換
        program_tensor = torch.tensor(program_tokens, dtype=torch.long)

        # Teacher Forcing用: 入力と出力をずらす
        # 入力: <BOS> token1 token2 ... tokenN
        # 出力:      token1 token2 ... tokenN <EOS>
        if len(program_tensor) > 1:
            input_tokens = program_tensor[:-1]  # <EOS>を除く
            target_tokens = program_tensor[1:]   # <BOS>を除く
        else:
            # 最小長の場合
            input_tokens = program_tensor
            target_tokens = program_tensor

        # パディング
        input_tokens = self._pad_tokens(input_tokens, self.max_program_length)
        target_tokens = self._pad_tokens(target_tokens, self.max_program_length)

        # アテンションマスク（パディング部分を無視）
        attention_mask = (input_tokens != 0).long()

        return {
            'input_grid': input_grid,
            'output_grid': output_grid,
            'program_tokens': input_tokens,
            'target_tokens': target_tokens,
            'attention_mask': attention_mask,
            'pair_id': pair.pair_id,
            'metadata': pair.metadata
        }

    def _pad_grid(self, grid: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """グリッドをパディング"""
        height, width = grid.shape

        # まず、必要に応じて切り取り
        if height > target_height:
            grid = grid[:target_height, :]
            height = target_height
        if width > target_width:
            grid = grid[:, :target_width]
            width = target_width

        # 切り取られた後のサイズが目標サイズと一致する場合はそのまま返す
        if height == target_height and width == target_width:
            return grid

        # パディングが必要
        padded = torch.zeros(target_height, target_width, dtype=grid.dtype)
        padded[:height, :width] = grid

        return padded

    def _get_ir_template(self, idx: int, pair: DataPair) -> str:
        """
        DataPairからIRテンプレート文字列を取得

        Args:
            idx: データインデックス
            pair: DataPair

        Returns:
            IRテンプレート文字列（変換失敗時は空文字列）
        """
        # キャッシュをチェック
        if idx in self._ir_cache:
            return self._ir_cache[idx]

        try:
            # DataPairのプログラムコードをIRSequenceに変換
            if not pair.program:
                self._ir_cache[idx] = ""
                return ""
            
            ir_sequence = self.relabel_transformer.transform(pair.program)

            # IRSequenceをテンプレート文字列に変換
            template_str = sequence_to_template_string(ir_sequence)

            # キャッシュに保存
            self._ir_cache[idx] = template_str

            return template_str
        except Exception as e:
            # 変換失敗時は空文字列を返す
            print(f"Warning: Failed to convert DataPair to IR template at index {idx}: {e}")
            self._ir_cache[idx] = ""
            return ""

    def _pad_tokens(self, tokens: torch.Tensor, target_length: int) -> torch.Tensor:
        """トークンをパディング"""
        current_length = tokens.size(0)

        if current_length >= target_length:
            return tokens[:target_length]

        # パディング（0: <PAD>）
        padded = torch.zeros(target_length, dtype=tokens.dtype)
        padded[:current_length] = tokens

        return padded


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    バッチをまとめる

    Args:
        batch: バッチデータのリスト

    Returns:
        バッチTensor
    """
    collated = {
        'input_grid': torch.stack([item['input_grid'] for item in batch]),
        'output_grid': torch.stack([item['output_grid'] for item in batch]),
        'program_tokens': torch.stack([item['program_tokens'] for item in batch]),
        'target_tokens': torch.stack([item['target_tokens'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }

    if 'pair_id' in batch[0]:
        collated['pair_ids'] = [item.get('pair_id') for item in batch]
    if 'metadata' in batch[0]:
        collated['metadata'] = [item.get('metadata', {}) for item in batch]

    return collated
