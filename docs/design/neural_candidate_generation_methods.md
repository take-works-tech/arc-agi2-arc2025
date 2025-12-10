# 深層学習ベース候補生成手法の詳細仕様

> **目的**: 現在実装されている3つの深層学習ベース候補生成手法の完全な技術仕様書
>
> **対象読者**: ChatGPT、開発者、研究者

---

## 目次

1. [概要](#概要)
2. [①深層学習ベース（グリッド→プログラム）](#①深層学習ベースグリッドプログラム)
3. [②深層学習ベース２（オブジェクト→プログラム）](#②深層学習ベース２オブジェクトプログラム)
4. [③深層学習ベース３（グリッド→グリッド）](#③深層学習ベース３グリッドグリッド)
5. [共通処理フロー](#共通処理フロー)
6. [設定パラメータ](#設定パラメータ)

---

## 概要

本システムでは、ARC-AGI2タスクに対して3つの深層学習ベースの候補生成手法を実装しています。各手法は異なる入力形式とアーキテクチャを持ち、それぞれ異なるタイプの問題に適しています。

### 全体フロー

```
[入力: 訓練ペア (input_grid, output_grid)]
    ↓
[オブジェクトマッチング] (オプション)
    ↓ 部分プログラム生成
[候補生成フェーズ]
    ├─ ①グリッド→プログラム (NeuralCandidateGenerator)
    ├─ ②オブジェクト→プログラム (NeuralObjectCandidateGenerator)
    └─ ③グリッド→グリッド (GridToGridCandidateGenerator)
    ↓
[候補統合・ランキング]
    ├─ 重複除去
    ├─ 複雑度ランキング (ComplexityRegularizer)
    └─ 上位N個を選出
    ↓
[出力: 候補プログラムリスト]
```

---

## ①深層学習ベース（グリッド→プログラム）

### 概要

**モデル名**: `ProgramSynthesisModel`
**生成器クラス**: `NeuralCandidateGenerator`
**入力**: 入力グリッド + 出力グリッド（生のピクセル配列）
**出力**: DSLプログラム文字列

### アーキテクチャ

```
入力グリッド [H, W] (0-9の色値)
    ↓
GridEncoder (Transformer Encoder)
    ├─ 入力グリッドエンコード → [seq_len, embed_dim]
    └─ 出力グリッドエンコード → [seq_len, embed_dim]
    ↓
Grid Fusion Layer
    ├─ 平均プーリング → [embed_dim]
    ├─ 入出力結合 → [embed_dim * 2]
    └─ Linear融合 → [decoder_dim]
    ↓
ProgramDecoder (Transformer Decoder)
    ├─ 自己回帰生成
    ├─ ビームサーチ
    └─ トークン列生成
    ↓
トークン→テンプレート文字列→DSLプログラム
```

### 詳細仕様

#### 1. モデル構造

**ファイル**: `src/hybrid_system/models/program_synthesis/program_synthesis_model.py`

```python
class ProgramSynthesisModel(BaseModel):
    def __init__(self, config: ProgramSynthesisConfig):
        # グリッドエンコーダ
        self.grid_encoder = GridEncoder(
            input_channels=10,  # 10色 (0-9)
            embed_dim=config.grid_encoder_dim,
            num_layers=config.grid_encoder_layers,
            num_heads=config.grid_encoder_heads,
            dropout=config.dropout
        )

        # プログラムデコーダ
        self.program_decoder = ProgramDecoder(
            vocab_size=config.vocab_size,
            embed_dim=config.program_decoder_dim,
            num_layers=config.program_decoder_layers,
            num_heads=config.program_decoder_heads,
            dropout=config.dropout,
            max_length=config.max_program_length,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0
        )

        # 入出力グリッドの融合層
        self.grid_fusion = nn.Linear(
            config.grid_encoder_dim * 2,
            config.program_decoder_dim
        )
```

#### 2. 入力形式

- **入力グリッド**: `List[List[int]]` (2次元リスト、各要素は0-9の色値)
- **出力グリッド**: `List[List[int]]` (2次元リスト、各要素は0-9の色値)
- **テンソル変換**: `torch.tensor(grid, dtype=torch.long).unsqueeze(0)` → `[1, H, W]`

#### 3. エンコーディングプロセス

```python
# 1. グリッドをエンコード
input_encoded = self.grid_encoder(input_grid)   # [batch, seq_len, embed_dim]
output_encoded = self.grid_encoder(output_grid)  # [batch, seq_len, embed_dim]

# 2. 平均プーリングで固定次元に
input_pooled = input_encoded.mean(dim=1)   # [batch, embed_dim]
output_pooled = output_encoded.mean(dim=1)  # [batch, embed_dim]

# 3. 入出力を結合・融合
grid_features = torch.cat([input_pooled, output_pooled], dim=-1)  # [batch, embed_dim*2]
grid_context = self.grid_fusion(grid_features)  # [batch, decoder_dim]
```

#### 4. ビームサーチ生成

**メソッド**: `beam_search()`

```python
def beam_search(
    self,
    input_grid: torch.Tensor,      # [1, H, W]
    output_grid: torch.Tensor,     # [1, H, W]
    beam_width: int = 5,           # ビーム幅（デフォルト: 5）
    max_length: Optional[int] = None,
    partial_program: Optional[str] = None,  # 部分プログラム（オプション）
    tokenizer: Optional[Any] = None
) -> list:
    """
    ビームサーチでプログラムを生成

    Returns:
        candidates: [(tokens, score), ...] のリスト
    """
    # 1. グリッドをエンコード（上記のプロセス）
    grid_context = ...  # [1, decoder_dim]

    # 2. 部分プログラムをトークン化（オプション）
    initial_tokens = None
    if partial_program and tokenizer:
        token_ids = tokenizer.encode(partial_program, add_special_tokens=True)
        initial_tokens = torch.tensor([token_ids], dtype=torch.long, device=...)

    # 3. プログラムデコーダでビームサーチ
    candidates = self.program_decoder.beam_search(
        context=grid_context,
        beam_width=beam_width,
        max_length=max_length or self.program_config.max_program_length,
        initial_tokens=initial_tokens  # 部分プログラムから開始
    )

    return candidates  # [(tokens, score), ...]
```

#### 5. 候補生成フロー

**ファイル**: `src/hybrid_system/inference/program_synthesis/candidate_generators/neural_generator.py`

```python
class NeuralCandidateGenerator:
    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        beam_width: int = 5,
        partial_program: Optional[str] = None
    ) -> List[str]:
        # 1. グリッドをテンソルに変換
        input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)  # [1, H, W]
        output_tensor = torch.tensor(output_grid, dtype=torch.long).unsqueeze(0)  # [1, H, W]

        # 2. ビームサーチでプログラムを生成
        beam_results = self.neural_model.beam_search(
            input_grid=input_tensor,
            output_grid=output_tensor,
            beam_width=beam_width,
            partial_program=partial_program,
            tokenizer=self.tokenizer
        )

        # 3. トークンをプログラム文字列に変換
        candidates = []
        for tokens, score in beam_results[:beam_width]:
            # BOS/EOS/PADを除去
            token_ids = tokens[0].cpu().numpy().tolist()
            token_ids = [tid for tid in token_ids if tid not in [1, 2, 0]]

            if token_ids:
                # トークン→テンプレート文字列
                template_string = self.tokenizer.decode(token_ids)

                # テンプレート文字列→IRSequence→DSLプログラム
                sequence = template_string_to_sequence(template_string, task_id="inference")
                program = sequence_to_dsl(sequence)

                if program:
                    candidates.append(program)

        return candidates
```

#### 6. トークン化とDSL変換

**トークン形式**: テンプレートトークン列（スペース区切り）

```
SEQ|START OP|GET_ALL_OBJECTS ARG|connectivity|{"name":"connectivity","value":4,...} STEP|END OP|SET_COLOR ... SEQ|END
```

**変換フロー**:
1. **トークン列** → **テンプレート文字列** (`tokenizer.decode()`)
2. **テンプレート文字列** → **IRSequence** (`template_string_to_sequence()`)
3. **IRSequence** → **DSLプログラム** (`sequence_to_dsl()`)

**例**:
```
トークン列: "SEQ|START OP|GET_ALL_OBJECTS ARG|connectivity|4 STEP|END ..."
    ↓
テンプレート文字列: "SEQ|START OP|GET_ALL_OBJECTS ARG|connectivity|4 STEP|END ..."
    ↓
IRSequence: IRSequence(steps=[TemplateStep(operation=GET_ALL_OBJECTS, ...), ...])
    ↓
DSLプログラム: "objects = GET_ALL_OBJECTS(4)\nFOR i LEN(objects) DO ..."
```

#### 7. 部分プログラムの使用方法

部分プログラムが提供される場合：

1. **トークン化**: `tokenizer.encode(partial_program, add_special_tokens=True)`
2. **初期トークンとして設定**: `initial_tokens`として`beam_search()`に渡す
3. **継続生成**: デコーダが部分プログラムから続きを生成

**例**:
```python
partial_program = "objects = GET_ALL_OBJECTS(4)"
# → トークン化 → initial_tokens = [1, 123, 456, ...]  # BOS + トークン列
# → ビームサーチで "FOR i LEN(objects) DO ..." を継続生成
```

#### 8. 設定パラメータ

**ファイル**: `src/hybrid_system/inference/program_synthesis/config.py`

```python
@dataclass
class CandidateConfig:
    enable_neural_generation: bool = True
    num_neural_candidates_with_partial: int = 20      # 部分プログラムありの場合
    num_neural_candidates_without_partial: int = 20   # 部分プログラムなしの場合
```

---

## ②深層学習ベース２（オブジェクト→プログラム）

### 概要

**モデル名**: `ObjectBasedProgramSynthesisModel`
**生成器クラス**: `NeuralObjectCandidateGenerator`
**入力**: 入力オブジェクトリスト + 出力オブジェクトリスト
**出力**: DSLプログラム文字列

### アーキテクチャ（改善版）

```
入力グリッド [H, W]
    ↓
オブジェクト抽出 (IntegratedObjectExtractor)
    ├─ 4連結オブジェクト抽出
    └─ Object型のリスト生成
    ↓
[改善] Object Canonicalization（オプション）
    ├─ 色のランダムリマップ
    ├─ 位置・サイズの正規化
    └─ 形状埋め込み
    ↓
[改善] Object Graph構築 + GNNエンコーディング（オプション）
    ├─ ObjectGraphBuilder: オブジェクト間の関係をグラフ化
    ├─ ObjectGraphEncoder: グラフ構造をエンコード
    └─ グラフ特徴量 [1, num_nodes, embed_dim]
    ↓
[改善] Relation Classifier（オプション）
    └─ オブジェクト間の関係を分類
    ↓
ObjectEncoder (Transformer Encoder) [改善版]
    ├─ 入力オブジェクトリストエンコード → [num_objects, embed_dim]
    ├─ [改善] 正規化オブジェクトを使用（オプション）
    ├─ [改善] グラフ特徴量を統合（オプション）
    └─ 出力オブジェクトリストエンコード → [num_objects, embed_dim]
    ↓
Object Pooling & Fusion
    ├─ 平均プーリング → [embed_dim]
    ├─ 入出力結合 → [embed_dim * 2]
    └─ Linear融合 → [decoder_dim]
    ↓
ProgramDecoder (Transformer Decoder)
    ├─ 自己回帰生成
    ├─ ビームサーチ
    └─ トークン列生成
    ↓
[改善] 関係情報によるスコア調整
    └─ 生成されたプログラムのスコアを調整
    ↓
トークン→テンプレート文字列→DSLプログラム
```

### 詳細仕様

#### 1. モデル構造

**ファイル**: `src/hybrid_system/models/program_synthesis/object_based_program_synthesis_model.py`

```python
class ObjectBasedProgramSynthesisModel(BaseModel):
    def __init__(self, config: ProgramSynthesisConfig):
        # オブジェクトエンコーダー
        self.object_encoder = ObjectEncoder(
            embed_dim=config.grid_encoder_dim,
            num_layers=config.grid_encoder_layers,
            num_heads=config.grid_encoder_heads,
            dropout=config.dropout
        )

        # プログラムデコーダー（①と同じ）
        self.program_decoder = ProgramDecoder(...)

        # オブジェクトリストの融合層
        self.object_fusion = nn.Linear(
            config.grid_encoder_dim * 2,
            config.program_decoder_dim
        )
```

#### 2. オブジェクト抽出プロセス

**ファイル**: `src/hybrid_system/inference/program_synthesis/candidate_generators/neural_object_generator.py`

**改善点（2025-01-XX実装）**:
- ✅ Object Graph + GNN特徴量をObjectEncoderに統合
- ✅ 正規化されたオブジェクトをObjectEncoderで使用
- ✅ 関係情報によるスコア調整

```python
# 1. グリッドをnumpy配列に変換
input_array = np.array(input_grid, dtype=int)
output_array = np.array(output_grid, dtype=int)

# 2. IntegratedObjectExtractorでオブジェクトを抽出
input_result = self.object_extractor.extract_objects_by_type(input_array)
output_result = self.object_extractor.extract_objects_by_type(output_array)

# 3. 4連結オブジェクトを取得
input_objects_4 = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
output_objects_4 = output_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])

# 4. 背景色とグリッドサイズを取得
input_bg_color = get_background_color(input_grid)
output_bg_color = get_background_color(output_grid)
input_h, input_w = get_grid_size(input_grid)
output_h, output_w = get_grid_size(output_grid)
```

#### 3. オブジェクトエンコーディング

**Object型の構造**:
```python
@dataclass
class Object:
    object_id: str
    object_type: ObjectType
    color_ratio: Dict[int, float]  # 色の比率
    pixels: List[Tuple[int, int]]  # ピクセル座標
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    # ... その他の特徴量
```

**エンコーディングプロセス（改善版）**:
```python
# 1. Object Canonicalization（オプション）
if self.enable_object_canonicalization:
    input_canonicalized = self.canonicalizer.canonicalize(
        input_objects_4, input_w, input_h
    )
    output_canonicalized = self.canonicalizer.canonicalize(
        output_objects_4, output_w, output_h,
        color_remap_map=self.canonicalizer.get_color_remap_map()
    )

# 2. Object Graph構築 + GNNエンコーディング（オプション）
if self.enable_object_graph:
    input_graph = self.graph_builder.build_graph(
        input_objects_4, categories=categories, ...
    )
    output_graph = self.graph_builder.build_graph(
        output_objects_4, categories=categories, ...
    )

    if self.graph_encoder:
        input_graph_encoded, _ = self.graph_encoder(input_graph)
        output_graph_encoded, _ = self.graph_encoder(output_graph)

# 3. オブジェクトリストをエンコード（改善版）
input_encoded = self.object_encoder(
    input_objects,              # List[Object]
    input_background_color,     # int
    input_grid_width,           # int
    input_grid_height,          # int
    graph_encoded=input_graph_encoded,  # [改善] グラフ特徴量を統合
    canonicalized_objects=input_canonicalized  # [改善] 正規化オブジェクトを使用
)  # [1, num_input_objects, embed_dim]

output_encoded = self.object_encoder(
    output_objects,
    output_background_color,
    output_grid_width,
    output_grid_height,
    graph_encoded=output_graph_encoded,  # [改善] グラフ特徴量を統合
    canonicalized_objects=output_canonicalized  # [改善] 正規化オブジェクトを使用
)  # [1, num_output_objects, embed_dim]

# 4. 平均プーリングで固定次元に
input_pooled = input_encoded.mean(dim=1)   # [1, embed_dim]
output_pooled = output_encoded.mean(dim=1)  # [1, embed_dim]

# 5. 入出力を結合・融合
object_features = torch.cat([input_pooled, output_pooled], dim=-1)  # [1, embed_dim*2]
object_context = self.object_fusion(object_features)  # [1, decoder_dim]
```

#### 4. ビームサーチ生成

**メソッド**: `beam_search()`

```python
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
    tokenizer: Optional[Any] = None
) -> List[Tuple[torch.Tensor, float]]:
    """
    ビームサーチでプログラムを生成

    Returns:
        candidates: [(tokens, score), ...] のリスト
    """
    # 1. オブジェクトをエンコード（上記のプロセス）
    object_context = ...  # [1, decoder_dim]

    # 2. 部分プログラムをトークン化（オプション）
    initial_tokens = None
    if partial_program and tokenizer:
        token_ids = tokenizer.encode(partial_program, add_special_tokens=True)
        initial_tokens = torch.tensor([token_ids], dtype=torch.long, device=...)

    # 3. プログラムデコーダでビームサーチ
    candidates = self.program_decoder.beam_search(
        context=object_context,
        beam_width=beam_width,
        max_length=max_length or self.program_config.max_program_length,
        initial_tokens=initial_tokens
    )

    return candidates
```

#### 5. 候補生成フロー

**ファイル**: `src/hybrid_system/inference/program_synthesis/candidate_generators/neural_object_generator.py`

```python
class NeuralObjectCandidateGenerator:
    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        beam_width: int = 5,
        partial_program: Optional[str] = None
    ) -> List[str]:
        # 1. オブジェクト抽出（上記のプロセス）
        input_objects_4 = ...
        output_objects_4 = ...
        input_bg_color = ...
        output_bg_color = ...
        input_h, input_w = ...
        output_h, output_w = ...

        # 2. ビームサーチでプログラムを生成
        beam_results = self.neural_object_model.beam_search(
            input_objects=input_objects_4,
            output_objects=output_objects_4,
            input_background_color=input_bg_color,
            output_background_color=output_bg_color,
            input_grid_width=input_w,
            input_grid_height=input_h,
            output_grid_width=output_w,
            output_grid_height=output_h,
            beam_width=beam_width,
            partial_program=partial_program,
            tokenizer=self.tokenizer
        )

        # 3. トークンをプログラム文字列に変換し、関係情報に基づいてスコアを調整（改善版）
        candidate_with_scores = []
        for tokens, score in beam_results[:beam_width]:
            token_ids = tokens[0].cpu().numpy().tolist()
            token_ids = [tid for tid in token_ids if tid not in [1, 2, 0]]

            if token_ids:
                template_string = self.tokenizer.decode(token_ids)
                sequence = template_string_to_sequence(template_string, task_id="inference")
                program = sequence_to_dsl(sequence)

                if program:
                    # [改善] 関係情報に基づいてスコアを調整
                    adjusted_score = self._adjust_score_with_relations(
                        program, relation_info, input_objects_4, category_var_mapping, score
                    )
                    candidate_with_scores.append((program, adjusted_score))

        # スコアでソート（高い順）
        candidate_with_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [program for program, _ in candidate_with_scores]

        return candidates
```

#### 6. 設定パラメータ

```python
@dataclass
class CandidateConfig:
    enable_neural_object_generation: bool = True
    num_neural_object_candidates_with_partial: int = 20      # 部分プログラムありの場合
    num_neural_object_candidates_without_partial: int = 20   # 部分プログラムなしの場合
```

---

## ③深層学習ベース３（グリッド→グリッド）

### 概要

**モデル名**: `GridToGridModel`
**生成器クラス**: `GridToGridCandidateGenerator`
**入力**: 入力グリッド（生のピクセル配列）
**出力**: DSLプログラム文字列（逆推論）

### アーキテクチャ

```
入力グリッド [H, W] (0-9の色値)
    ↓
GridEncoder (Transformer Encoder)
    └─ グリッドエンコード → [seq_len, embed_dim]
    ↓
GridDecoder (Transformer Decoder)
    └─ グリッドデコード → [H, W, num_colors] (各ピクセルの色確率分布)
    ↓
argmax → 予測グリッド [H, W]
    ↓
逆推論エンジン (infer_program_from_grid_transformation)
    ├─ 色置換パターン検出
    ├─ 回転パターン検出
    ├─ 反転パターン検出
    ├─ パディング/トリミング検出
    └─ DSLプログラム生成
    ↓
出力: DSLプログラム文字列
```

### 詳細仕様

#### 1. モデル構造

**ファイル**: `src/hybrid_system/models/program_synthesis/grid_to_grid_model.py`

```python
class GridToGridModel(BaseModel):
    def __init__(self, config: GridToGridConfig):
        # グリッドエンコーダ
        self.grid_encoder = GridEncoder(
            input_channels=config.num_colors,  # 10色
            embed_dim=config.grid_encoder_dim,
            num_layers=config.grid_encoder_layers,
            num_heads=config.grid_encoder_heads,
            dropout=config.dropout,
            max_grid_size=config.max_grid_size
        )

        # グリッドデコーダ
        self.grid_decoder = GridDecoder(
            embed_dim=config.grid_encoder_dim,
            num_colors=config.num_colors,  # 10色
            num_layers=config.grid_decoder_layers,
            dropout=config.dropout,
            max_grid_size=config.max_grid_size
        )
```

#### 2. 予測プロセス

```python
def predict(
    self,
    input_grid: torch.Tensor,  # [batch, H, W]
    target_shape: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    グリッド→グリッド変換を予測

    Returns:
        predicted_grid: [batch, H, W] (各ピクセルの色値)
    """
    # 1. グリッドをエンコード
    encoded = self.grid_encoder(input_grid)  # [batch, seq_len, embed_dim]

    # 2. グリッドをデコード（色確率分布を生成）
    if target_shape is None:
        batch_size, height, width = input_grid.shape
        target_shape = (height, width)

    color_logits = self.grid_decoder(encoded, target_shape)  # [batch, H, W, num_colors]

    # 3. 最も確率の高い色を選択
    predicted_grid = torch.argmax(color_logits, dim=-1)  # [batch, H, W]

    return predicted_grid
```

#### 3. 逆推論プロセス

**メソッド**: `infer_program_from_grid_transformation()`

```python
def infer_program_from_grid_transformation(
    self,
    input_grid: np.ndarray,  # 入力グリッド
    output_grid: np.ndarray  # 予測された出力グリッド
) -> Optional[str]:
    """
    グリッド変換結果からプログラムを逆推論

    検出されるパターン:
    1. 色置換 (REPLACE_COLOR)
    2. 回転 (ROTATE_GRID)
    3. 反転 (FLIP_GRID)
    4. パディング (RENDER_GRID with padding)
    5. トリミング (RENDER_GRID with cropping)
    """
    # 1. 色の置換パターンを検出
    color_mapping = self._detect_color_mapping(input_grid, output_grid)
    if color_mapping:
        program = self._generate_color_replacement_program(color_mapping)
        if program:
            return program

    # 2. グリッド全体の操作を検出
    if self._is_rotation(input_grid, output_grid):
        angle = self._detect_rotation_angle(input_grid, output_grid)
        if angle is not None:
            return f"ROTATE_GRID({angle})"

    if self._is_flip(input_grid, output_grid):
        direction = self._detect_flip_direction(input_grid, output_grid)
        if direction:
            return f"FLIP_GRID('{direction}')"

    # 3. パディング/トリミングを検出
    if self._is_padding(input_grid, output_grid):
        padding_info = self._detect_padding(input_grid, output_grid)
        if padding_info:
            return self._generate_padding_program(padding_info)

    if self._is_cropping(input_grid, output_grid):
        crop_info = self._detect_cropping(input_grid, output_grid)
        if crop_info:
            return self._generate_cropping_program(crop_info)

    return None
```

#### 4. パターン検出アルゴリズム

##### 4.1 色置換検出

```python
def _detect_color_mapping(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> Optional[Dict[int, int]]:
    """
    色の置換パターンを検出

    アルゴリズム:
    1. 入力グリッドの各色について、対応する出力グリッドの色を分析
    2. 信頼度90%以上で一意にマッピングできる場合のみ採用

    Returns:
        color_mapping: {入力色: 出力色} の辞書
    """
    if input_grid.shape != output_grid.shape:
        return None

    color_mapping = {}
    for color_in in range(10):
        input_mask = (input_grid == color_in)
        if not np.any(input_mask):
            continue

        output_colors = output_grid[input_mask]
        unique_colors, counts = np.unique(output_colors, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]
        confidence = np.max(counts) / len(output_colors)

        if confidence >= 0.9:  # 90%以上の信頼度
            color_mapping[color_in] = int(most_common_color)

    return color_mapping if color_mapping else None
```

**生成されるプログラム例**:
```python
# 単一色置換
"REPLACE_COLOR(1, 2)"

# 複数色置換
"IF EQUAL(color, 1) THEN REPLACE_COLOR(1, 2) END\nIF EQUAL(color, 3) THEN REPLACE_COLOR(3, 4) END"
```

##### 4.2 回転検出

```python
def _is_rotation(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> bool:
    """回転かどうかを判定"""
    rotated_90 = np.rot90(input_grid, k=1)
    rotated_180 = np.rot90(input_grid, k=2)
    rotated_270 = np.rot90(input_grid, k=3)

    return (np.array_equal(output_grid, rotated_90) or
            np.array_equal(output_grid, rotated_180) or
            np.array_equal(output_grid, rotated_270))

def _detect_rotation_angle(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> Optional[int]:
    """回転角度を検出 (90, 180, 270)"""
    # ... 90, 180, 270度の回転をチェック
    return angle  # 90, 180, または 270
```

**生成されるプログラム例**:
```python
"ROTATE_GRID(90)"
"ROTATE_GRID(180)"
"ROTATE_GRID(270)"
```

##### 4.3 反転検出

```python
def _is_flip(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> bool:
    """反転かどうかを判定"""
    flipped_h = np.flip(input_grid, axis=1)  # 水平反転
    flipped_v = np.flip(input_grid, axis=0)  # 垂直反転

    return (np.array_equal(output_grid, flipped_h) or
            np.array_equal(output_grid, flipped_v))

def _detect_flip_direction(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> Optional[str]:
    """反転方向を検出 ('horizontal' または 'vertical')"""
    # ... 水平/垂直反転をチェック
    return direction
```

**生成されるプログラム例**:
```python
"FLIP_GRID('horizontal')"
"FLIP_GRID('vertical')"
```

##### 4.4 パディング/トリミング検出

```python
def _detect_padding(
    self,
    input_grid: np.ndarray,
    output_grid: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    パディング情報を検出

    アルゴリズム:
    1. 出力グリッドが入力グリッドより大きい場合
    2. 入力グリッドが出力グリッドの部分グリッドとして含まれているかチェック
    3. 80%以上一致する位置を検出
    4. パディング情報（offset, padding_top/bottom/left/right）を返す
    """
    # ... 詳細な検出ロジック
    return {
        'offset_h': int,
        'offset_w': int,
        'padding_top': int,
        'padding_bottom': int,
        'padding_left': int,
        'padding_right': int
    }
```

**生成されるプログラム例**:
```python
"objects = GET_ALL_OBJECTS(4)\nRENDER_GRID(objects, 0, output_w, output_h)"
```

#### 5. 候補生成フロー

**ファイル**: `src/hybrid_system/inference/program_synthesis/candidate_generators/grid_to_grid_generator.py`

```python
class GridToGridCandidateGenerator:
    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> List[str]:
        candidates = []

        try:
            if self.grid_to_grid_model is None:
                return candidates

            # 1. グリッドをテンソルに変換
            input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(0)  # [1, H, W]

            # 2. グリッド→グリッド変換を予測
            self.grid_to_grid_model.eval()
            with torch.no_grad():
                predicted_grid = self.grid_to_grid_model.predict(input_tensor)  # [1, H, W]
                predicted_array = predicted_grid[0].cpu().numpy()

            # 3. グリッド変換結果からプログラムを逆推論
            inferred_program = self.grid_to_grid_model.infer_program_from_grid_transformation(
                np.array(input_grid),
                predicted_array
            )

            if inferred_program:
                candidates.append(inferred_program)

        except Exception as e:
            print(f"グリッド→グリッド候補生成エラー: {e}")

        return candidates
```

#### 6. 設定パラメータ

```python
@dataclass
class CandidateConfig:
    enable_neural_grid_to_grid: bool = True
    num_grid_to_grid_candidates: int = 20
```

**注意**: この手法は通常1つの候補のみを生成します（逆推論のため）。

---

## 共通処理フロー

### 統合候補生成

**ファイル**: `src/hybrid_system/inference/program_synthesis/candidate_generator.py`

```python
class CandidateGenerator:
    def generate_candidates(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
        max_candidates: Optional[int] = None,
        pair_index: Optional[int] = None,
        partial_program: Optional[str] = None,
        matching_result: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        統合候補生成

        処理フロー:
        1. 部分プログラムの拡張（オプション）
        2. シード変動による複数回生成（オプション）
        3. 3つの手法を並列実行
        4. 重複除去
        5. 複雑度ランキング
        6. 上位N個を選出
        """
        all_candidates = []

        # 1. 部分プログラムの拡張（オプション）
        if partial_program:
            extended_program = self._extend_partial_program(...)
            if extended_program:
                all_candidates.append(extended_program)
            all_candidates.append(partial_program)

        # 2. シード変動による複数回生成
        num_seeds = self.config.num_seeds_per_pair if self.config.enable_seed_variation else 1

        for seed_offset in range(num_seeds):
            # シードを設定
            if self.config.enable_seed_variation and pair_index is not None:
                current_seed = self._get_seed_for_pair(pair_index, seed_offset)
                random.seed(current_seed)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)

            candidates = []

            # 3. ①グリッド→プログラム
            if self.config.enable_neural_generation and self.neural_model is not None:
                if partial_program:
                    # 部分プログラムあり/なしの両方を生成
                    candidates.extend(self.neural_generator.generate_candidates(
                        input_grid, output_grid,
                        beam_width=self.config.num_neural_candidates_with_partial,
                        partial_program=partial_program
                    ))
                    candidates.extend(self.neural_generator.generate_candidates(
                        input_grid, output_grid,
                        beam_width=self.config.num_neural_candidates_without_partial,
                        partial_program=None
                    ))
                else:
                    candidates.extend(self.neural_generator.generate_candidates(
                        input_grid, output_grid,
                        beam_width=self.config.num_neural_candidates_without_partial,
                        partial_program=None
                    ))

            # 4. ②オブジェクト→プログラム
            if self.config.enable_neural_object_generation and self.neural_object_model is not None:
                # 同様の処理...

            # 5. ③グリッド→グリッド
            if self.config.enable_neural_grid_to_grid and self.grid_to_grid_model is not None:
                candidates.extend(self.grid_to_grid_generator.generate_candidates(
                    input_grid, output_grid
                ))

            all_candidates.extend(candidates)

        # 6. 重複を除去
        unique_candidates = list(set(all_candidates))

        # 7. 複雑度ランキングで選出
        if len(unique_candidates) > max_candidates:
            ranked_results = self.rank_candidates_by_complexity(unique_candidates)
            unique_candidates = [result['program'] for result in ranked_results[:max_candidates]]
        elif len(unique_candidates) > 0:
            ranked_results = self.rank_candidates_by_complexity(unique_candidates)
            unique_candidates = [result['program'] for result in ranked_results]

        return unique_candidates
```

### 複雑度ランキング

**ファイル**: `src/hybrid_system/inference/program_synthesis/complexity_regularizer.py`

```python
class ComplexityRegularizer:
    def score_candidates(
        self,
        programs: List[str],
        base_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        候補プログラムを複雑度ペナルティ込みでスコアリング

        スコア計算:
        final_score = base_score - complexity_penalty

        複雑度要素:
        - 制御構造 (control_structures)
        - 関数呼び出し (function_calls)
        - 変数 (variables)
        - ネスト深度 (nested_depth)
        - プログラム長 (program_length)
        """
        results = []
        for program, base_score in zip(programs, base_scores):
            complexity = self.calculate_complexity_score(program)
            final_score = base_score - self.config.penalty_factor * complexity
            results.append({
                'program': program,
                'base_score': base_score,
                'complexity': complexity,
                'final_score': final_score
            })

        # final_scoreの高い順にソート
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
```

---

## 設定パラメータ

### CandidateConfig

**ファイル**: `src/hybrid_system/inference/program_synthesis/config.py`

```python
@dataclass
class CandidateConfig:
    """候補生成設定"""
    max_candidates: int = 30  # 最終的な最大候補数

    # 各手法の有効化フラグ
    enable_neural_generation: bool = True
    enable_neural_object_generation: bool = True
    enable_neural_grid_to_grid: bool = True

    # シード変動設定
    task_seed: Optional[int] = None
    num_seeds_per_pair: int = 3
    enable_seed_variation: bool = True

    # 各手法の生成数設定
    num_neural_candidates_with_partial: int = 20      # ①: 部分プログラムあり
    num_neural_candidates_without_partial: int = 20   # ①: 部分プログラムなし
    num_neural_object_candidates_with_partial: int = 20      # ②: 部分プログラムあり
    num_neural_object_candidates_without_partial: int = 20   # ②: 部分プログラムなし
    num_grid_to_grid_candidates: int = 20             # ③: グリッド→グリッド
```

### モデル設定

**ProgramSynthesisConfig** (①と②で使用):
```python
grid_encoder_dim: int = 256
grid_encoder_layers: int = 4
grid_encoder_heads: int = 8
program_decoder_dim: int = 256
program_decoder_layers: int = 4
program_decoder_heads: int = 8
vocab_size: int = 10000
max_program_length: int = 200
dropout: float = 0.1
```

**GridToGridConfig** (③で使用):
```python
num_colors: int = 10
grid_encoder_dim: int = 256
grid_encoder_layers: int = 4
grid_encoder_heads: int = 8
grid_decoder_layers: int = 4
max_grid_size: int = 30
dropout: float = 0.1
```

---

## 使用例

### 基本的な使用

```python
from src.hybrid_system.inference.program_synthesis.candidate_generator import CandidateGenerator
from src.hybrid_system.inference.program_synthesis.config import CandidateConfig

# 設定
config = CandidateConfig(
    enable_neural_generation=True,
    enable_neural_object_generation=True,
    enable_neural_grid_to_grid=True,
    num_neural_candidates_with_partial=20,
    num_neural_candidates_without_partial=20,
    max_candidates=30
)

# 生成器を初期化
generator = CandidateGenerator(
    config=config,
    neural_model=neural_model,              # ProgramSynthesisModel
    tokenizer=tokenizer,                    # ProgramTokenizer
    neural_object_model=neural_object_model, # ObjectBasedProgramSynthesisModel
    grid_to_grid_model=grid_to_grid_model   # GridToGridModel
)

# 候補を生成
input_grid = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
output_grid = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
partial_program = "objects = GET_ALL_OBJECTS(4)"

candidates = generator.generate_candidates(
    input_grid=input_grid,
    output_grid=output_grid,
    max_candidates=30,
    pair_index=0,
    partial_program=partial_program,
    matching_result=matching_result
)

# 結果: List[str] - DSLプログラムのリスト
```

---

## まとめ

### 各手法の特徴

| 手法 | 入力形式 | 強み | 弱み | 適用問題 |
|------|---------|------|------|----------|
| **①グリッド→プログラム** | 生グリッド | シンプル、高速 | オブジェクト情報なし | 単純な変換 |
| **②オブジェクト→プログラム** | オブジェクトリスト | オブジェクト情報活用 | 抽出エラーに弱い | オブジェクト操作 |
| **③グリッド→グリッド** | 生グリッド | グリッド全体操作 | 逆推論の限界 | 回転・反転・色置換 |

### 生成される候補数の理論値

各部分プログラム × 各訓練ペアに対して：

- **①グリッド→プログラム**: 最大40個（部分プログラムあり20 + なし20）
- **②オブジェクト→プログラム**: 最大40個（部分プログラムあり20 + なし20）
- **③グリッド→グリッド**: 最大1個（逆推論のため通常1つ）

**合計**: 最大81個/部分プログラム/訓練ペア

**最終的な候補数**: 複雑度ランキング後、`max_candidates`（デフォルト30）個に制限

---

## 実装ファイル一覧

### モデル
- `src/hybrid_system/models/program_synthesis/program_synthesis_model.py` - ①のモデル
- `src/hybrid_system/models/program_synthesis/object_based_program_synthesis_model.py` - ②のモデル
- `src/hybrid_system/models/program_synthesis/grid_to_grid_model.py` - ③のモデル
- `src/hybrid_system/models/program_synthesis/grid_encoder.py` - グリッドエンコーダ
- `src/hybrid_system/models/program_synthesis/object_encoder.py` - オブジェクトエンコーダ
- `src/hybrid_system/models/program_synthesis/program_decoder.py` - プログラムデコーダ
- `src/hybrid_system/models/program_synthesis/grid_decoder.py` - グリッドデコーダ

### 生成器
- `src/hybrid_system/inference/program_synthesis/candidate_generators/neural_generator.py` - ①の生成器
- `src/hybrid_system/inference/program_synthesis/candidate_generators/neural_object_generator.py` - ②の生成器
- `src/hybrid_system/inference/program_synthesis/candidate_generators/grid_to_grid_generator.py` - ③の生成器

### 統合
- `src/hybrid_system/inference/program_synthesis/candidate_generator.py` - 統合候補生成器
- `src/hybrid_system/inference/program_synthesis/config.py` - 設定クラス
- `src/hybrid_system/inference/program_synthesis/complexity_regularizer.py` - 複雑度ランキング

### 変換
- `src/hybrid_system/ir/serialization/template_serialization.py` - トークン↔テンプレート変換
- `src/hybrid_system/ir/execution/template_executor.py` - テンプレート→DSL変換

---

**最終更新**: 2025-01-XX
**バージョン**: 1.1

**更新履歴**:
- 2025-01-XX: ②オブジェクト→プログラムに改善機能を追加
  - GNN特徴量をObjectEncoderに統合
  - 正規化オブジェクトをObjectEncoderで使用
  - 関係情報によるスコア調整
