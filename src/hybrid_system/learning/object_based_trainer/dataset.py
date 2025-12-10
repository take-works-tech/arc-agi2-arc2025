"""
ObjectBasedProgramSynthesisModel用のDatasetクラス

オブジェクトリスト形式のデータセット
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

from src.hybrid_system.core.data_structures import DataPair
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from src.hybrid_system.ir.parser.relabel_transformer import RelabelTransformer
from src.hybrid_system.ir.serialization.template_serialization import sequence_to_template_string
from src.data_systems.data_models.core.object import Object
from src.core_systems.executor.core import ExecutorCore


class ObjectBasedDataset(Dataset):
    """
    ObjectBasedProgramSynthesisModel用のPyTorchデータセット

    入出力グリッドからオブジェクトを抽出し、オブジェクトリストとプログラムをTensorに変換
    """

    def __init__(
        self,
        data_pairs: List[DataPair],
        tokenizer: ProgramTokenizer,
        max_program_length: int = 512,
        use_ir_templates: bool = True,
        connectivity: int = 4
    ):
        """
        初期化

        Args:
            data_pairs: DataPairのリスト
            tokenizer: プログラムトークナイザー
            max_program_length: プログラムの最大長
            use_ir_templates: IRテンプレートを使用するかどうか（デフォルト: True）
            connectivity: オブジェクト抽出の連結性（4 or 8、デフォルト: 4）
        """
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_program_length = max_program_length
        self.use_ir_templates = use_ir_templates
        self.connectivity = connectivity

        # ExecutorCoreを初期化（オブジェクト抽出用）
        self.executor = ExecutorCore()

        # IRテンプレート変換用
        if self.use_ir_templates:
            self.relabel_transformer = RelabelTransformer()
            # IRSequenceをキャッシュ
            self._ir_cache = {}

        # オブジェクト抽出結果をキャッシュ
        self._object_cache = {}

    def __len__(self) -> int:
        """データセットサイズ"""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        データを取得

        Args:
            idx: インデックス

        Returns:
            - input_objects: 入力オブジェクトリスト
            - output_objects: 出力オブジェクトリスト
            - input_background_color: 入力背景色
            - output_background_color: 出力背景色
            - input_grid_width: 入力グリッド幅
            - input_grid_height: 入力グリッド高さ
            - output_grid_width: 出力グリッド幅
            - output_grid_height: 出力グリッド高さ
            - program_tokens: プログラムトークン（入力） [seq_len-1]
            - target_tokens: ターゲットトークン（出力） [seq_len-1]
            - attention_mask: アテンションマスク [seq_len-1]
        """
        pair = self.data_pairs[idx]

        # グリッドをnumpy配列に変換
        input_array = np.array(pair.input, dtype=int)
        output_array = np.array(pair.output, dtype=int)

        # オブジェクトを抽出（キャッシュを使用）
        cache_key = f"{idx}_{self.connectivity}"
        if cache_key in self._object_cache:
            input_objects, output_objects = self._object_cache[cache_key]
        else:
            try:
                # ExecutorCoreを使用してオブジェクトを抽出
                _, input_objects_list, _ = self.executor.execute_program(
                    f"GET_ALL_OBJECTS({self.connectivity})", input_array
                )
                _, output_objects_list, _ = self.executor.execute_program(
                    f"GET_ALL_OBJECTS({self.connectivity})", output_array
                )

                # オブジェクトリストがNoneの場合は空リストに
                if input_objects_list is None:
                    input_objects_list = []
                if output_objects_list is None:
                    output_objects_list = []

                # Object型でない場合は空リストに
                if input_objects_list and not isinstance(input_objects_list[0], Object):
                    input_objects_list = []
                if output_objects_list and not isinstance(output_objects_list[0], Object):
                    output_objects_list = []

                input_objects = input_objects_list if isinstance(input_objects_list, list) else []
                output_objects = output_objects_list if isinstance(output_objects_list, list) else []

                # キャッシュに保存
                self._object_cache[cache_key] = (input_objects, output_objects)
            except Exception as e:
                print(f"Warning: Failed to extract objects at index {idx}: {e}")
                input_objects = []
                output_objects = []
                self._object_cache[cache_key] = (input_objects, output_objects)

        # 背景色を取得
        input_bg_color = self._get_background_color(pair.input)
        output_bg_color = self._get_background_color(pair.output)

        # グリッドサイズを取得
        input_h, input_w = len(pair.input), len(pair.input[0]) if pair.input else 0
        output_h, output_w = len(pair.output), len(pair.output[0]) if pair.output else 0

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
            'input_objects': input_objects,
            'output_objects': output_objects,
            'input_background_color': input_bg_color,
            'output_background_color': output_bg_color,
            'input_grid_width': input_w,
            'input_grid_height': input_h,
            'output_grid_width': output_w,
            'output_grid_height': output_h,
            'program_tokens': input_tokens,
            'target_tokens': target_tokens,
            'attention_mask': attention_mask,
            'pair_id': pair.pair_id,
            'metadata': pair.metadata
        }

    def _get_background_color(self, grid: List[List[int]]) -> int:
        """グリッドの背景色を取得（最頻出色、通常は0）"""
        if not grid or not grid[0]:
            return 0
        flat_colors = [c for row in grid for c in row]
        if not flat_colors:
            return 0
        counter = Counter(flat_colors)
        # 最頻出色を返す（同率の場合は数値が小さいものを優先）
        most_common_color, _ = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))
        return int(most_common_color)

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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    バッチをまとめる

    Args:
        batch: バッチデータのリスト

    Returns:
        バッチTensor
    """
    collated = {
        'input_objects': [item['input_objects'] for item in batch],
        'output_objects': [item['output_objects'] for item in batch],
        'input_background_color': torch.tensor([item['input_background_color'] for item in batch], dtype=torch.long),
        'output_background_color': torch.tensor([item['output_background_color'] for item in batch], dtype=torch.long),
        'input_grid_width': torch.tensor([item['input_grid_width'] for item in batch], dtype=torch.long),
        'input_grid_height': torch.tensor([item['input_grid_height'] for item in batch], dtype=torch.long),
        'output_grid_width': torch.tensor([item['output_grid_width'] for item in batch], dtype=torch.long),
        'output_grid_height': torch.tensor([item['output_grid_height'] for item in batch], dtype=torch.long),
        'program_tokens': torch.stack([item['program_tokens'] for item in batch]),
        'target_tokens': torch.stack([item['target_tokens'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }

    if 'pair_id' in batch[0]:
        collated['pair_ids'] = [item.get('pair_id') for item in batch]
    if 'metadata' in batch[0]:
        collated['metadata'] = [item.get('metadata', {}) for item in batch]

    return collated
