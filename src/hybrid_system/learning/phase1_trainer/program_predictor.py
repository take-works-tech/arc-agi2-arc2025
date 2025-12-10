"""
プログラム予測器

入力+出力からプログラムを予測するモデル
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from core.data_structures import DataPair


@dataclass
class PredictorConfig:
    """予測器設定"""
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    vocab_size: int = 1000
    max_program_length: int = 200


class ProgramPredictor(nn.Module):
    """プログラム予測モデル"""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config

        # グリッドエンコーダー
        self.grid_encoder = GridEncoder(config.hidden_dim)

        # Transformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # プログラムデコーダー
        self.program_decoder = ProgramDecoder(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            max_length=config.max_program_length
        )

        # 出力層
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """フォワードパス

        Args:
            input_grid: 入力グリッド [batch_size, height, width]
            output_grid: 出力グリッド [batch_size, height, width]

        Returns:
            プログラムトークンの予測 [batch_size, max_length, vocab_size]
        """
        batch_size = input_grid.size(0)

        # グリッドをエンコード
        input_features = self.grid_encoder(input_grid)  # [batch_size, seq_len, hidden_dim]
        output_features = self.grid_encoder(output_grid)  # [batch_size, seq_len, hidden_dim]

        # 入力と出力を結合
        combined_features = torch.cat([input_features, output_features], dim=1)

        # Transformerエンコーダーで処理
        encoded_features = self.transformer_encoder(combined_features)

        # プログラムデコーダーでプログラムを生成
        program_logits = self.program_decoder(encoded_features)

        return program_logits


class GridEncoder(nn.Module):
    """グリッドエンコーダー"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 色埋め込み
        self.color_embedding = nn.Embedding(10, hidden_dim // 4)  # 0-9の色

        # 位置埋め込み
        self.position_embedding = nn.Embedding(100, hidden_dim // 4)  # 最大100位置

        # 特徴量変換
        self.feature_projection = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """グリッドをエンコード

        Args:
            grid: グリッド [batch_size, height, width]

        Returns:
            エンコードされた特徴量 [batch_size, seq_len, hidden_dim]
        """
        batch_size, height, width = grid.size()

        # グリッドを平坦化
        flat_grid = grid.view(batch_size, -1)  # [batch_size, height*width]

        # 位置インデックスを生成
        positions = torch.arange(height * width, device=grid.device).unsqueeze(0).expand(batch_size, -1)

        # 色埋め込み
        color_emb = self.color_embedding(flat_grid)  # [batch_size, seq_len, hidden_dim//4]

        # 位置埋め込み
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, hidden_dim//4]

        # 特徴量を結合
        combined_features = torch.cat([color_emb, pos_emb], dim=-1)  # [batch_size, seq_len, hidden_dim//2]

        # 特徴量を投影
        encoded_features = self.feature_projection(combined_features)  # [batch_size, seq_len, hidden_dim]

        return encoded_features


class ProgramDecoder(nn.Module):
    """プログラムデコーダー"""

    def __init__(self, hidden_dim: int, vocab_size: int, max_length: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        # デコーダー層
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2
        )

        # プログラム埋め込み
        self.program_embedding = nn.Embedding(vocab_size, hidden_dim)

        # 位置埋め込み
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """プログラムをデコード

        Args:
            encoder_output: エンコーダー出力 [batch_size, seq_len, hidden_dim]

        Returns:
            プログラムトークンの予測 [batch_size, max_length, vocab_size]
        """
        batch_size = encoder_output.size(0)

        # プログラムトークンのシーケンスを生成（本格実装）
        # デコーダーの初期トークンとして<START>トークンを使用
        # 実際の生成では、各ステップで前のトークンに基づいて次のトークンを予測
        start_token = 2  # <START>トークン
        program_tokens = torch.full(
            (batch_size, self.max_length),
            start_token,
            dtype=torch.long,
            device=encoder_output.device
        )

        # プログラム埋め込み
        program_emb = self.program_embedding(program_tokens)  # [batch_size, max_length, hidden_dim]

        # 位置埋め込み
        positions = torch.arange(self.max_length, device=encoder_output.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch_size, max_length, hidden_dim]

        # 埋め込みを結合
        decoder_input = program_emb + pos_emb

        # Transformerデコーダーで処理
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        return decoder_output


class ProgramPredictorTrainer:
    """プログラム予測器の学習器"""

    def __init__(self, config: PredictorConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # モデル初期化
        self.model = ProgramPredictor(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # 学習履歴
        self.training_history = []

        # トークナイザー
        self.tokenizer = ProgramTokenizer()

    def train(self, data_pairs: List[DataPair]) -> Dict[str, Any]:
        """学習を実行

        Args:
            data_pairs: 学習データ

        Returns:
            学習結果
        """
        print(f"プログラム予測器の学習開始: {len(data_pairs)}ペア")

        # データを準備
        train_loader = self._prepare_data(data_pairs)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            self.model.train()

            for batch in train_loader:
                input_grids, output_grids, target_programs = batch

                # デバイスに移動
                input_grids = input_grids.to(self.device)
                output_grids = output_grids.to(self.device)
                target_programs = target_programs.to(self.device)

                # フォワードパス
                self.optimizer.zero_grad()
                program_logits = self.model(input_grids, output_grids)

                # 損失計算
                loss = self.criterion(
                    program_logits.view(-1, self.config.vocab_size),
                    target_programs.view(-1)
                )

                # バックワードパス
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            # 学習履歴を記録
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss
            })

            print(f"Epoch {epoch+1}/{self.config.max_epochs}, Loss: {avg_loss:.4f}")

            # 早期停止チェック
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"早期停止: {self.config.early_stopping_patience}エポック改善なし")
                break

        return {
            'final_loss': avg_loss,
            'best_loss': best_loss,
            'epochs_trained': epoch + 1,
            'training_history': self.training_history
        }

    def predict(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """プログラムを予測

        Args:
            input_grid: 入力グリッド
            output_grid: 出力グリッド

        Returns:
            予測されたプログラム
        """
        self.model.eval()

        with torch.no_grad():
            # テンソルに変換
            input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0).to(self.device)
            output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0).to(self.device)

            # 予測
            program_logits = self.model(input_tensor, output_tensor)
            predicted_tokens = torch.argmax(program_logits, dim=-1)

            # トークンをプログラムに変換
            program = self.tokenizer.decode(predicted_tokens[0].cpu().numpy())

        return program

    def _prepare_data(self, data_pairs: List[DataPair]):
        """データを準備（本格実装）"""
        # PyTorchのDataLoaderを使用した実装
        from torch.utils.data import Dataset, DataLoader

        class ProgramDataset(Dataset):
            def __init__(self, data_pairs, tokenizer):
                self.data_pairs = data_pairs
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data_pairs)

            def __getitem__(self, idx):
                pair = self.data_pairs[idx]
                input_grid = torch.tensor(np.array(pair.input), dtype=torch.long)
                output_grid = torch.tensor(np.array(pair.output), dtype=torch.long)
                program_tokens = torch.tensor(self.tokenizer.encode(pair.program), dtype=torch.long)
                return input_grid, output_grid, program_tokens

        dataset = ProgramDataset(data_pairs, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows互換性のため0に設定
            pin_memory=False
        )

        return dataloader


class ProgramTokenizer:
    """プログラムトークナイザー"""

    def __init__(self):
        # 基本トークン
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
        }

        # コマンドトークン
        commands = [
            'GET_ALL_OBJECTS', 'FILTER', 'FOR', 'IF', 'SET_COLOR', 'MOVE',
            'ROTATE', 'SCALE', 'MERGE', 'SPLIT', 'CREATE_LINE', 'CREATE_RECT'
        ]

        for i, cmd in enumerate(commands):
            self.vocab[cmd] = len(self.vocab)

        # 数値トークン
        for i in range(10):
            self.vocab[str(i)] = len(self.vocab)

        # 変数トークン
        for i in range(10):
            self.vocab[f'$obj{i}'] = len(self.vocab)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, program: str) -> List[int]:
        """プログラムをトークン化（本格実装）"""
        import re

        tokens = [self.vocab['<START>']]

        # より詳細なトークン化: コマンド、変数、数値、括弧などを適切に分割
        # 正規表現でトークンを分割
        token_pattern = r'([A-Z_]+|\(|\)|\[|\]|,|;|\d+|\$[a-zA-Z0-9_]+|[a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(token_pattern, program)

        for match in matches:
            # コマンドや変数として認識
            if match in self.vocab:
                tokens.append(self.vocab[match])
            elif match.isdigit():
                # 数値は個別のトークンとして扱う
                if match in self.vocab:
                    tokens.append(self.vocab[match])
                else:
                    tokens.append(self.vocab['<UNK>'])
            elif match.startswith('$'):
                # 変数として扱う
                if match in self.vocab:
                    tokens.append(self.vocab[match])
                else:
                    tokens.append(self.vocab['<UNK>'])
            else:
                # その他のトークン
                tokens.append(self.vocab['<UNK>'])

        tokens.append(self.vocab['<END>'])

        # パディング
        max_length = 200
        while len(tokens) < max_length:
            tokens.append(self.vocab['<PAD>'])

        return tokens[:max_length]

    def decode(self, tokens: List[int]) -> str:
        """トークンをプログラムに変換"""
        words = []
        for token in tokens:
            if token in self.reverse_vocab:
                word = self.reverse_vocab[token]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)

        return ' '.join(words)
