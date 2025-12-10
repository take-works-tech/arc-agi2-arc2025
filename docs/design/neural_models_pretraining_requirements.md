# 推論パイプライン：事前学習が必要なモデル一覧

## 📋 概要

推論パイプラインで使用されるニューラルモデルのうち、事前学習が必要なものと、その学習データ、推論時の入出力をまとめます。

---

## 1. 事前学習が必要なモデル一覧

### ✅ 実装済み（学習が必要）

#### ①グリッド→プログラムパイプライン

1. **NGPS（Neural Guided Program Search）** ✅ 実装済み
   - **役割**: プログラム探索空間を1000倍削減
   - **実装**: `NeuralGuidedProgramSearch`クラス
   - **学習データ**: 必要

2. **DSL Selector** ✅ 実装済み
   - **役割**: DSL使用確率を予測し、探索の暴走を防止
   - **実装**: `DSLSelector`クラス
   - **学習データ**: 必要（NGPSと統合可能）

3. **Program Synthesis Model（グリッド→プログラム）** ✅ 実装済み
   - **役割**: グリッドペアからプログラムコードを生成
   - **実装**: `ProgramSynthesisModel`クラス
   - **学習データ**: 必要

#### ②オブジェクト→プログラムパイプライン

4. **Object Graph + GNN Encoder** ✅ 実装済み ⭐ 最優先
   - **役割**: オブジェクト関係・パターン認識、プログラム生成に必要な構造特徴を抽出
   - **実装**: `ObjectGraphEncoder`（Graphormer/EGNN）、`ObjectGraphBuilder`
   - **学習データ**: 必要

5. **Relation Classifier** ✅ 実装済み
   - **役割**: オブジェクト間の関係を分類（spatial, mirror, repeat, contain等）
   - **実装**: `RelationClassifier`クラス
   - **学習データ**: 必要

6. **Object Canonicalization** ✅ 実装済み
   - **役割**: オブジェクトの標準化（色、位置、サイズの正規化）
   - **実装**: `ObjectCanonicalizer`クラス
   - **学習データ**: 必要（教師なし学習も可能）

### ⏳ 未実装（将来実装予定）

7. **Neural Mask Generator** ⏳ 未実装（Tier 2）
   - **役割**: プログラム探索前処理としてマスク生成（補助専用）
   - **学習データ**: 必要（実装時に追加）

8. **Abstract Object Patterns** ⏳ 未実装（Tier 3）
   - **役割**: 抽象オブジェクトパターンの認識
   - **学習データ**: 必要（実装時に追加）

---

## 2. 学習データの詳細

### 2.1 NGPS/DSL Selector用データ

**形式**: JSONL

**内容**:
```json
{
  "task_id": "task_001",
  "pair_index": 0,
  "grid_features": {
    "input_shape": [10, 10],
    "output_shape": [10, 10],
    "input_size": 100,
    "output_size": 100,
    "input_unique_colors": 3,
    "output_unique_colors": 3,
    "input_mean": 2.5,
    "output_mean": 2.5
  },
  "dsl_probabilities": {
    "MIRROR_X": 0.82,
    "ROTATE": 0.74,
    "SCALE": 0.65,
    ...
  },
  "input_grid": [[...]],
  "output_grid": [[...]]
}
```

**生成方法**:
- generatorの出力から自動生成（`NeuralTrainingDataGenerator._generate_ngps_data`）
- または `scripts/data_generation/generate_ngps_training_data.py`

**学習時の入力**:
- `grid_features`: グリッド特徴量（辞書）
- `input_grid`, `output_grid`: グリッドデータ（2D配列）

**学習時の出力（教師データ）**:
- `dsl_probabilities`: DSL使用確率分布（辞書）

---

### 2.2 Object Graph + GNN用データ

**形式**: JSONL

**内容**:
```json
{
  "task_id": "task_001",
  "pair_index": 0,
  "graph_features": {
    "num_nodes": 5,
    "num_edges": 8,
    "node_features": [[...], [...], ...],  // 各ノードの特徴量
    "edge_index": [[0, 1, 2, ...], [1, 2, 3, ...]],  // エッジの接続情報
    "edge_attr": [[...], [...], ...],  // 各エッジの特徴量
    "edge_types": ["spatial", "mirror", ...]
  },
  "program": "program_code_string",
  "input_grid_shape": [10, 10],
  "output_grid_shape": [10, 10],
  "num_input_objects": 5,
  "num_output_objects": 5
}
```

**生成方法**:
- generatorの出力から自動生成（`NeuralTrainingDataGenerator._generate_object_graph_data`）
- または `scripts/data_generation/generate_object_graph_training_data.py`

**注意事項**:
- `program`フィールドには**正解プログラム**（generatorが生成した完全なプログラムコード）を保存
- 部分プログラムやオブジェクトマッチング結果は推論時の補助情報であり、学習データの教師データではない
- オブジェクトグラフはオブジェクト抽出から直接構築（オブジェクトマッチング結果は不要）

**学習時の入力**:
- `graph_features.node_features`: ノード特徴量（テンソル）
- `graph_features.edge_index`: エッジ接続情報（テンソル）
- `graph_features.edge_attr`: エッジ特徴量（テンソル）

**学習時の出力（教師データ）**:
- `program`: **正解プログラムコード**（generatorが生成した完全なプログラム、文字列）
  - **注意**: 部分プログラムではない。部分プログラムは推論時の補助情報であり、学習データの教師データではない

---

### 2.3 Relation Classifier用データ

**形式**: JSONL

**内容**:
```json
{
  "task_id": "task_001",
  "pair_index": 0,
  "obj1_features": [center_x, center_y, bbox_width, bbox_height, color, area, ...],
  "obj2_features": [center_x, center_y, bbox_width, bbox_height, color, area, ...],
  "relative_features": [delta_x, delta_y, delta_width, delta_height],
  "relation_labels": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  // one-hot encoding
  "relation_types": ["spatial_left", "spatial_right", "spatial_up", "spatial_down", "mirror_x", "mirror_y", "repeat", "contain"],
  "edge_type": "spatial"
}
```

**生成方法**:
- generatorの出力から自動生成（`NeuralTrainingDataGenerator._generate_relation_classifier_data`）
- または `scripts/data_generation/generate_relation_classifier_data.py`

**学習時の入力**:
- `obj1_features`: オブジェクト1の特徴量（リスト）
- `obj2_features`: オブジェクト2の特徴量（リスト）
- `relative_features`: 相対特徴量（リスト）

**学習時の出力（教師データ）**:
- `relation_labels`: 関係性ラベル（one-hot encoding、リスト）

---

### 2.4 Object Canonicalization用データ

**形式**: JSONL

**内容**:
```json
{
  "task_id": "task_001",
  "pair_index": 0,
  "original_features": [center_x, center_y, bbox_width, bbox_height, color, area, ...],
  "remapped_color": 1,
  "normalized_position": [0.5, 0.5],
  "normalized_size": [0.3, 0.3],
  "shape_embedding": [...]
}
```

**生成方法**:
- generatorの出力から自動生成（`NeuralTrainingDataGenerator._generate_canonicalization_data`）
- ObjectCanonicalizerクラスを使用して正規化前後の特徴量を生成

**学習時の入力**:
- `original_features`: 正規化前のオブジェクト特徴量（リスト）

**学習時の出力（教師データ）**:
- `remapped_color`: リマップ後の色（整数）
- `normalized_position`: 正規化後の位置（リスト）
- `normalized_size`: 正規化後のサイズ（リスト）
- `shape_embedding`: 形状埋め込み（リスト）

---

### 2.5 Program Synthesis Model用データ

**形式**: JSONL

**内容**:
```json
{
  "task_id": "task_001",
  "pair_index": 0,
  "input_grid": [[...]],
  "output_grid": [[...]],
  "program_code": "program_code_string",
  "tokens": ["token1", "token2", ...],
  "complexity": 3,
  "program_stats": {...}
}
```

**生成方法**:
- generatorの出力から直接生成可能

**学習時の入力**:
- `input_grid`: 入力グリッド（2D配列）
- `output_grid`: 出力グリッド（2D配列）

**学習時の出力（教師データ）**:
- `program_code`: プログラムコード（文字列）
- `tokens`: トークン列（リスト）

---

## 3. 推論時の入出力

### 3.1 NGPS（推論時）

**入力**:
- `grid_embedding`: グリッド埋め込み（テンソル、`[batch, embed_dim]` または `[batch, seq_len, embed_dim]`）
- または `grid_features`: グリッド特徴量（辞書）

**出力**:
- `dsl_probabilities`: DSL使用確率分布（辞書、`{dsl_command: probability, ...}`）
- または `top_k_dsl`: Top-k DSLコマンドリスト（リスト）

**使用箇所**:
- `NeuralGuidedProgramSearch`クラス内でプログラム探索を誘導

---

### 3.2 DSL Selector（推論時）

**入力**:
- `grid_embedding`: グリッド埋め込み（テンソル、`[batch, embed_dim]` または `[batch, seq_len, embed_dim]`）

**出力**:
- `dsl_logits`: DSL使用確率のロジット（テンソル、`[batch, num_dsl_commands]`）
- `dsl_probabilities`: DSL使用確率分布（辞書、`{dsl_command: probability, ...}`）

**使用箇所**:
- NGPSと統合してプログラム探索を誘導
- Meta-Reasoner層で使用

---

### 3.3 Object Graph + GNN Encoder（推論時）

**入力**:
- `ObjectGraph`: オブジェクトグラフオブジェクト
  - `node_features`: ノード特徴量（テンソル、`[num_nodes, node_dim]`）
  - `edge_index`: エッジ接続情報（テンソル、`[2, num_edges]`）
  - `edge_attr`: エッジ特徴量（テンソル、`[num_edges, edge_dim]`）

**出力**:
- `graph_embedding`: グラフ埋め込み（テンソル、`[num_nodes, embed_dim]`）
- または `program_scores`: プログラム候補のスコア（テンソル）

**使用箇所**:
- `NeuralObjectCandidateGenerator`クラス内でオブジェクト関係をエンコード
- プログラム生成の補助情報として使用

---

### 3.4 Relation Classifier（推論時）

**入力**:
- `ObjectGraph`: オブジェクトグラフオブジェクト
  - `node_features`: ノード特徴量
  - `edge_index`: エッジ接続情報
  - `edge_attr`: エッジ特徴量

**出力**:
- `relation_scores`: 関係性スコア（テンソル、`[num_edges, num_relation_types]`）
- `relations`: 分類された関係性（辞書、`{(src_idx, tgt_idx): [relation_types]}`）

**使用箇所**:
- プログラム候補の絞り込み（90%以上を削減可能）
- オブジェクト関係の理解を補助

---

### 3.5 Object Canonicalization（推論時）

**入力**:
- `objects`: オブジェクトリスト（`List[Object]`）
- `grid_width`: グリッド幅（整数）
- `grid_height`: グリッド高さ（整数）

**出力**:
- `canonicalized_objects`: 正規化後のオブジェクトリスト（`List[CanonicalizedObject]`）
  - `remapped_color`: リマップ後の色
  - `normalized_position`: 正規化後の位置
  - `normalized_size`: 正規化後のサイズ
  - `shape_embedding`: 形状埋め込み

**使用箇所**:
- オブジェクト→プログラムパイプラインの前処理
- 色不一致問題の解決（40-60%改善）

---

### 3.6 Program Synthesis Model（推論時）

**入力**:
- `input_grid`: 入力グリッド（numpy配列、`[height, width]`）
- `output_grid`: 出力グリッド（numpy配列、`[height, width]`）
- または `grid_embedding`: グリッド埋め込み（テンソル）

**出力**:
- `program_code`: プログラムコード（文字列）
- または `tokens`: トークン列（リスト）

**使用箇所**:
- グリッド→プログラムパイプラインのメイン生成器

---

## 4. 推論パイプライン全体の入力

### 4.1 エントリーポイント

**入力**:
```python
Task {
    task_id: str,
    train: List[Dict[str, Any]]  # 訓練ペアのリスト
        [
            {
                'input': List[List[int]],   # 入力グリッド
                'output': List[List[int]]   # 出力グリッド
            },
            ...
        ],
    test: List[Dict[str, Any]],  # テストペアのリスト（推論時は空）
    program: str  # プログラムコード（推論時は空文字列）
}
```

**出力**:
```python
Optional[str]  # 合成されたプログラム（失敗時はNone）
```

---

## 5. 学習データ生成の統合

### 5.1 generatorからの自動生成

`NeuralTrainingDataGenerator`クラスを使用して、generatorの出力から自動的に学習データを生成できます。

**使用方法**:
```python
from src.data_systems.generator.neural_training_data_generator import NeuralTrainingDataGenerator

# 初期化
neural_data_generator = NeuralTrainingDataGenerator(output_dir)

# データ生成
neural_data_generator.generate_from_generator_output(
    task_id="task_001",
    pair_index=0,
    program_code=program_code,
    input_grid=input_grid,
    output_grid=output_grid,
    nodes=nodes,
    complexity=complexity
)

# バッチごとに保存
neural_data_generator.flush_batch(batch_index=0)
```

**出力先**:
```
outputs/YYYYMMDD_HHMMSS/
  batch_0000/
    neural_training_data/
      ngps_train_data.jsonl
      object_graph_train_data.jsonl
      relation_classifier_train_data.jsonl
      canonicalization_train_data.jsonl
      program_generation_train_data.jsonl
```

---

## 6. まとめ

### 事前学習が必要なモデル（実装済み）

1. **NGPS** - グリッド特徴量 + DSL確率分布
2. **DSL Selector** - グリッド埋め込み + DSL確率分布
3. **Object Graph + GNN** - オブジェクトグラフ + プログラムコード
4. **Relation Classifier** - オブジェクトペア特徴量 + 関係性ラベル
5. **Object Canonicalization** - 正規化前後のオブジェクト特徴量
6. **Program Synthesis Model** - グリッドペア + プログラムコード

### 推論時の入力

- **全体**: `Task`オブジェクト（訓練ペアのリスト）
- **各モデル**: 上記「3. 推論時の入出力」を参照

### 学習データ生成

- generatorの出力から自動生成可能（`NeuralTrainingDataGenerator`）
- または既存のスクリプトを使用（`scripts/data_generation/*.py`）
