# 拡張データセット生成パイプラインの改善提案

> **目的**: `improved_neural_generators_pipeline.md`で説明されているパイプラインの学習に必要なデータを、`generator`で直接生成できるようにする

## 📋 目次

1. [現状分析](#1-現状分析)
2. [必要な学習データ](#2-必要な学習データ)
3. [改善提案](#3-改善提案)
4. [実装の優先順位](#4-実装の優先順位)

---

## 1. 現状分析

### 1.1 現在のgeneratorが生成しているデータ

**`src/data_systems/generator/main.py`が生成するデータ**:
- ✅ プログラムコード（テキスト形式）
- ✅ プログラムトークン（JSON形式）
- ✅ プログラム統計情報（JSON形式）
- ✅ グリッドデータ（input/output、JSON形式）
- ✅ プログラムメタデータ（複雑度、グリッドサイズなど）

**保存形式**:
- タスクごとのフォルダー構造（`task_001/`, `task_002/`, ...）
- バッチごとのJSONファイル（`batch_0000/program_batch_0000.json`など）

### 1.2 既存の学習データ生成スクリプト

**`scripts/data_generation/`に存在するスクリプト**:
- ✅ `generate_ngps_training_data.py` - NGPS/DSL Selector用
- ✅ `generate_object_graph_training_data.py` - Object Graph + GNN用
- ✅ `generate_relation_classifier_data.py` - Relation Classifier用
- ✅ `generate_color_role_data.py` - Color Role Classifier用

**問題点**:
- 既存スクリプトは**ARC-AGI2の訓練データ**を読み込んで学習データを生成
- generatorで生成したデータから直接学習データを生成する機能がない
- 2段階の処理が必要（generator → 学習データ生成スクリプト）

---

## 2. 必要な学習データ

### 2.1 パイプラインで必要とされる学習データ

#### ①グリッド→プログラムパイプライン

**NGPS（Neural Guided Program Search）**:
- **入力**: グリッド埋め込み（またはグリッド特徴量）
- **出力**: DSL使用確率分布
- **形式**: JSONL
- **フィールド**:
  ```json
  {
    "task_id": "task_001",
    "pair_index": 0,
    "grid_features": {...},
    "dsl_probabilities": {"MIRROR_X": 0.82, "ROTATE": 0.74, ...},
    "input_grid": [[...]],
    "output_grid": [[...]]
  }
  ```

**DSL Selector**:
- **入力**: グリッド埋め込み
- **出力**: DSL使用確率分布（NGPSと同じ形式）
- **形式**: JSONL（NGPSと統合可能）

**プログラム生成モデル**:
- **入力**: グリッドペア（input_grid, output_grid）
- **出力**: プログラムコード（トークン列）
- **形式**: JSONL
- **フィールド**:
  ```json
  {
    "task_id": "task_001",
    "pair_index": 0,
    "input_grid": [[...]],
    "output_grid": [[...]],
    "program": "program_code",
    "tokens": ["token1", "token2", ...],
    "complexity": 3
  }
  ```

#### ②オブジェクト→プログラムパイプライン

**Object Graph + GNN**:
- **入力**: オブジェクトグラフ（ノード特徴量、エッジ特徴量）
- **出力**: プログラムコード
- **形式**: JSONL
- **フィールド**:
  ```json
  {
    "task_id": "task_001",
    "pair_index": 0,
    "graph_features": {
      "node_features": [[...]],
      "edge_index": [[...]],
      "edge_attr": [[...]]
    },
    "program": "program_code",
    "num_input_objects": 5,
    "num_output_objects": 5
  }
  ```

**Object Canonicalization**:
- **入力**: 正規化前のオブジェクト特徴量
- **出力**: 正規化後のオブジェクト特徴量
- **形式**: JSONL
- **フィールド**:
  ```json
  {
    "task_id": "task_001",
    "pair_index": 0,
    "original_objects": [...],
    "canonicalized_objects": [...],
    "color_remap_map": {...}
  }
  ```

**Relation Classifier**:
- **入力**: オブジェクトペアの特徴量
- **出力**: 関係性ラベル（spatial_left, mirror_x, contain, ...）
- **形式**: JSONL
- **フィールド**:
  ```json
  {
    "task_id": "task_001",
    "pair_index": 0,
    "obj1_features": [...],
    "obj2_features": [...],
    "relative_features": [...],
    "relation_labels": ["spatial_left", "spatial_up"]
  }
  ```

---

## 3. 改善提案

### 3.1 統合学習データ生成モジュールの追加

**新規ファイル**: `src/data_systems/generator/neural_training_data_generator.py`

**機能**:
1. generatorで生成したデータから、各ニューラルモデル用の学習データを生成
2. 既存の学習データ生成スクリプトのロジックを再利用
3. バッチ処理に対応（メモリ効率化）

**実装内容**:

```python
class NeuralTrainingDataGenerator:
    """ニューラルモデル用学習データ生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ngps_data = []
        self.object_graph_data = []
        self.relation_classifier_data = []
        self.canonicalization_data = []
        self.program_generation_data = []

    def generate_from_generator_output(
        self,
        task_id: str,
        program_code: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        nodes: List[Any],
        complexity: int
    ):
        """generatorの出力から学習データを生成"""
        # 1. NGPS/DSL Selector用データ
        self._generate_ngps_data(...)

        # 2. Object Graph + GNN用データ
        self._generate_object_graph_data(...)

        # 3. Relation Classifier用データ
        self._generate_relation_classifier_data(...)

        # 4. Object Canonicalization用データ
        self._generate_canonicalization_data(...)

        # 5. プログラム生成モデル用データ
        self._generate_program_generation_data(...)

    def save_all(self):
        """すべての学習データをJSONL形式で保存"""
        # 各データタイプごとにJSONLファイルに保存
        ...
```

### 3.2 generator/main.pyへの統合

**変更点**:
1. `NeuralTrainingDataGenerator`をインポート
2. プログラム生成・実行完了後に学習データを生成
3. バッチ処理完了時にJSONLファイルに保存

**実装例**:

```python
# main.py内
from src.data_systems.generator.neural_training_data_generator import NeuralTrainingDataGenerator

def generate_program(...):
    # ... 既存の処理 ...

    # 学習データ生成器を初期化（初回のみ）
    if not hasattr(generate_program, 'neural_data_generator'):
        neural_data_generator = NeuralTrainingDataGenerator(output_dir)
        generate_program.neural_data_generator = neural_data_generator

    # 学習データを生成
    neural_data_generator.generate_from_generator_output(
        task_id=f"task_{task_index:03d}",
        program_code=program_code,
        input_grid=input_grid,
        output_grid=output_grid,
        nodes=nodes,
        complexity=complexity
    )

    # ... 既存の処理 ...
```

### 3.3 バッチ処理での保存

**実装方針**:
- バッファマネージャーと同様に、バッチごとにJSONLファイルに保存
- メモリ効率を考慮して、バッチ完了時にフラッシュ

**保存先**:
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

### 3.4 既存スクリプトとの統合

**方針**:
- 既存の`scripts/data_generation/`のロジックを`NeuralTrainingDataGenerator`に統合
- 既存スクリプトは残す（generator出力からも生成可能にする）

**利点**:
- コードの重複を削減
- 一貫したデータ形式
- メンテナンス性の向上

---

## 4. 実装の優先順位

### Tier 1: 最優先（実装済みモデル用）

1. **NGPS/DSL Selector用学習データ生成** ⭐
   - **理由**: NGPSとDSL Selectorは実装済み
   - **実装**: `_generate_ngps_data()`メソッド
   - **データ**: グリッド特徴量 + DSL使用確率

2. **Object Graph + GNN用学習データ生成** ⭐
   - **理由**: Object Graph + GNNは実装済み
   - **実装**: `_generate_object_graph_data()`メソッド
   - **データ**: オブジェクトグラフ + プログラムコード

3. **Relation Classifier用学習データ生成** ⭐
   - **理由**: Relation Classifierは実装済み
   - **実装**: `_generate_relation_classifier_data()`メソッド
   - **データ**: オブジェクトペア + 関係性ラベル

### Tier 2: 中優先度

4. **Object Canonicalization用学習データ生成**
   - **理由**: Object Canonicalizationは実装済み
   - **実装**: `_generate_canonicalization_data()`メソッド
   - **データ**: 正規化前後のオブジェクト特徴量

5. **プログラム生成モデル用学習データ生成**
   - **理由**: プログラム生成モデルの学習に必要
   - **実装**: `_generate_program_generation_data()`メソッド
   - **データ**: グリッドペア + プログラムコード + トークン列

### Tier 3: 補助（将来的に実装）

6. **Color Role Classifier用学習データ生成**
   - **理由**: Color Role Classifierは実装済みだが、優先度は低い
   - **実装**: `_generate_color_role_data()`メソッド

7. **Contrastive Pretraining用学習データ生成**
   - **理由**: Contrastive Pretrainingは未実装（Tier 3）
   - **実装**: 将来的に追加

---

## 5. 実装の詳細

### 5.1 NGPS/DSL Selector用データ生成

**実装ロジック**:
1. プログラムコードからDSLコマンドを抽出
2. DSL使用確率を計算（コマンドの出現頻度）
3. グリッド特徴量を抽出（既存の`extract_grid_features()`を再利用）
4. JSONL形式で保存

**コード例**:
```python
def _generate_ngps_data(
    self,
    task_id: str,
    pair_index: int,
    program_code: str,
    input_grid: np.ndarray,
    output_grid: np.ndarray
):
    """NGPS/DSL Selector用データを生成"""
    # DSLコマンドを抽出
    dsl_commands = extract_dsl_commands(program_code)

    # DSL使用確率を計算
    dsl_probabilities = calculate_dsl_probabilities([program_code])

    # グリッド特徴量を抽出
    grid_features = extract_grid_features(input_grid, output_grid)

    # サンプルを作成
    sample = {
        'task_id': task_id,
        'pair_index': pair_index,
        'grid_features': grid_features,
        'dsl_probabilities': dsl_probabilities,
        'input_grid': input_grid.tolist(),
        'output_grid': output_grid.tolist()
    }

    self.ngps_data.append(sample)
```

### 5.2 Object Graph + GNN用データ生成

**実装ロジック**:
1. オブジェクト抽出（既存の`IntegratedObjectExtractor`を使用）
2. オブジェクトグラフ構築（既存の`ObjectGraphBuilder`を使用）
3. グラフ特徴量を抽出
4. プログラムコードとペアで保存

**コード例**:
```python
def _generate_object_graph_data(
    self,
    task_id: str,
    pair_index: int,
    program_code: str,
    input_grid: np.ndarray,
    output_grid: np.ndarray
):
    """Object Graph + GNN用データを生成"""
    # オブジェクト抽出
    input_result = self.extractor.extract_objects_by_type(input_grid, input_image_index=0)
    output_result = self.extractor.extract_objects_by_type(output_grid, input_image_index=0)

    if not input_result.success:
        return

    input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
    if not input_objects:
        return

    # オブジェクトグラフを構築
    graph = self.graph_builder.build_graph(input_objects)

    if graph.node_features.size(0) == 0:
        return

    # グラフ特徴量を抽出
    graph_features = extract_object_graph_features(graph)

    # サンプルを作成
    sample = {
        'task_id': task_id,
        'pair_index': pair_index,
        'graph_features': graph_features,
        'program': program_code,
        'num_input_objects': len(input_objects),
        'num_output_objects': len(output_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, []))
    }

    self.object_graph_data.append(sample)
```

### 5.3 Relation Classifier用データ生成

**実装ロジック**:
1. オブジェクト抽出
2. オブジェクトペアごとに関係性を分類
3. オブジェクト特徴量と関係性ラベルをペアで保存

**コード例**:
```python
def _generate_relation_classifier_data(
    self,
    task_id: str,
    pair_index: int,
    input_grid: np.ndarray,
    output_grid: np.ndarray
):
    """Relation Classifier用データを生成"""
    # オブジェクト抽出
    input_result = self.extractor.extract_objects_by_type(input_grid, input_image_index=0)

    if not input_result.success:
        return

    input_objects = input_result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
    if len(input_objects) < 2:
        return

    # オブジェクトペアごとにサンプルを生成
    for i in range(len(input_objects)):
        for j in range(i + 1, len(input_objects)):
            obj1 = input_objects[i]
            obj2 = input_objects[j]

            # 関係性を分類
            relations = classify_relation(obj1, obj2)

            if not relations:
                continue

            # オブジェクト特徴量を抽出
            obj1_features = extract_object_features(obj1)
            obj2_features = extract_object_features(obj2)
            relative_features = compute_relative_features(obj1, obj2)

            # サンプルを作成
            sample = {
                'task_id': task_id,
                'pair_index': pair_index,
                'obj1_features': obj1_features,
                'obj2_features': obj2_features,
                'relative_features': relative_features,
                'relation_labels': relations
            }

            self.relation_classifier_data.append(sample)
```

---

## 6. 期待される効果

### 6.1 効率化

- **2段階処理の統合**: generator → 学習データ生成が1ステップに
- **メモリ効率**: バッチ処理によるメモリ使用量の削減
- **処理時間の短縮**: データの再読み込みが不要

### 6.2 データ品質

- **一貫性**: generatorで生成したデータと学習データの一貫性が保証
- **完全性**: すべての必要な学習データが同時に生成される
- **トレーサビリティ**: タスクIDとペアインデックスで追跡可能

### 6.3 メンテナンス性

- **コードの統合**: 既存スクリプトのロジックを再利用
- **拡張性**: 新しい学習データタイプの追加が容易
- **テスト容易性**: 単一のモジュールでテスト可能

---

## 7. 実装チェックリスト

### Phase 1: 基盤実装
- [ ] `NeuralTrainingDataGenerator`クラスの作成
- [ ] バッチ処理対応の実装
- [ ] JSONL形式での保存機能

### Phase 2: Tier 1実装
- [ ] NGPS/DSL Selector用データ生成
- [ ] Object Graph + GNN用データ生成
- [ ] Relation Classifier用データ生成

### Phase 3: Tier 2実装
- [ ] Object Canonicalization用データ生成
- [ ] プログラム生成モデル用データ生成

### Phase 4: 統合とテスト
- [ ] `main.py`への統合
- [ ] 既存スクリプトとの互換性確認
- [ ] テストデータでの動作確認

---

## 8. 参考資料

- [改善後のNeural Generatorsパイプライン](../design/improved_neural_generators_pipeline.md)
- [既存の学習データ生成スクリプト](../../scripts/data_generation/)
- [NGPSアーキテクチャ説明](../guides/NGPS_ARCHITECTURE_EXPLANATION.md)
