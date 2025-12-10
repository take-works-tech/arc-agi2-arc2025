# 次のステップ：Neural Pipeline実装

> **目的**: `improved_neural_generators_pipeline.md`で説明されているパイプラインの実装を進めるための具体的な次のステップ

## 📋 現状整理

### ✅ 完了済み

1. **ドキュメント整備**
   - ✅ `improved_neural_generators_pipeline.md`の整理・更新
   - ✅ `generator_improvements_for_neural_pipeline.md`の作成
   - ✅ 学習データ生成セクションの追加

2. **実装済みモジュール**
   - ✅ Object Graph + GNN Encoder
   - ✅ NGPS (Neural Guided Program Search)
   - ✅ DSL Selector
   - ✅ Object Canonicalization
   - ✅ Relation Classifier

3. **既存の学習データ生成スクリプト**
   - ✅ `generate_ngps_training_data.py`
   - ✅ `generate_object_graph_training_data.py`
   - ✅ `generate_relation_classifier_data.py`
   - ✅ `generate_color_role_data.py`

### ⏳ 未実装・改善が必要

1. **generatorからの学習データ生成機能**
   - ❌ `NeuralTrainingDataGenerator`クラス（未実装）
   - ❌ generator/main.pyへの統合（未実装）

2. **Tier 1改善ポイント**
   - ❌ 背景色・色役割分類の導入
   - ❌ Object relation特徴量の追加（一部実装済み）
   - ❌ Program Decoderを構文木ベースにする

3. **Tier 2改善ポイント**
   - ❌ プログラム検証レイヤーの強化
   - ❌ Neural Mask Generatorの統合
   - ❌ Cross-Attention between Input/Outputの強化

---

## 🎯 推奨される次のステップ

### Option 1: generatorからの学習データ生成機能の実装（最優先）⭐

**理由**:
- パイプラインの学習に必要なデータを効率的に生成できる
- 既存の実装済みモジュールの学習に直結
- 実装の複雑度が中程度で、効果が高い

**実装内容**:

1. **`NeuralTrainingDataGenerator`クラスの実装**
   - ファイル: `src/data_systems/generator/neural_training_data_generator.py`
   - Tier 1の3つのデータタイプを実装:
     - NGPS/DSL Selector用データ生成
     - Object Graph + GNN用データ生成
     - Relation Classifier用データ生成

2. **generator/main.pyへの統合**
   - `NeuralTrainingDataGenerator`をインポート
   - プログラム生成・実行完了後に学習データを生成
   - バッチ処理完了時にJSONLファイルに保存

3. **テストと検証**
   - 小規模データセットで動作確認
   - 生成されたJSONLファイルの形式確認
   - 既存の学習スクリプトとの互換性確認

**期待される効果**:
- generatorで生成したデータから直接学習データを生成可能
- 2段階処理（generator → 学習データ生成スクリプト）が1ステップに
- データの一貫性が保証される

**実装時間の目安**: 2-3日

---

### Option 2: Tier 1改善ポイントの実装

**理由**:
- パイプラインの性能向上に直結
- 実装済みモジュールの効果を最大化

**実装内容**:

1. **背景色・色役割分類の導入**
   - Color Role Classifierの統合
   - 背景色推定の改善
   - Neural Mask Generatorとの統合準備

2. **Object relation特徴量の追加**
   - Object Graph構築時の関係性特徴量の強化
   - 位置関係、包含関係、接触関係の詳細化

3. **Program Decoderを構文木ベースにする**
   - 構文制約を考慮したプログラム生成
   - ビームサーチとの統合

**実装時間の目安**: 1-2週間

---

### Option 3: Tier 2改善ポイントの実装

**理由**:
- パイプラインの堅牢性と効率性の向上
- 無効な候補の事前フィルタリング

**実装内容**:

1. **プログラム検証レイヤーの強化**
   - 型チェック、操作空間チェック、境界チェック
   - `ProgramScorer`内のペナルティ項として組み込む

2. **Neural Mask Generatorの統合**
   - `NeuralMaskGenerator`クラスの実装
   - `GridToGridCandidateGenerator`への統合

3. **Cross-Attention between Input/Outputの強化**
   - Transformerのcross-attentionを使用
   - ①グリッド→プログラム、②オブジェクト→プログラムに適用

**実装時間の目安**: 1-2週間

---

## 📊 実装優先順位の推奨

### Phase 1: 学習データ生成機能の実装（最優先）⭐

**期間**: 2-3日

1. `NeuralTrainingDataGenerator`クラスの実装
2. generator/main.pyへの統合
3. テストと検証

**理由**:
- 実装済みモジュールの学習に必要
- 効果が高く、実装の複雑度が中程度
- 他の改善の基盤となる

### Phase 2: Tier 1改善ポイントの実装

**期間**: 1-2週間

1. 背景色・色役割分類の導入
2. Object relation特徴量の追加
3. Program Decoderを構文木ベースにする

**理由**:
- パイプラインの性能向上に直結
- Phase 1で生成した学習データを活用可能

### Phase 3: Tier 2改善ポイントの実装

**期間**: 1-2週間

1. プログラム検証レイヤーの強化
2. Neural Mask Generatorの統合
3. Cross-Attention between Input/Outputの強化

**理由**:
- パイプラインの堅牢性と効率性の向上
- Phase 1, 2の実装を基盤として実装可能

---

## 🛠️ 実装の詳細（Phase 1）

### Step 1: NeuralTrainingDataGeneratorクラスの実装

**ファイル**: `src/data_systems/generator/neural_training_data_generator.py`

**実装すべきメソッド**:

```python
class NeuralTrainingDataGenerator:
    def __init__(self, output_dir: str):
        """初期化"""

    def generate_from_generator_output(
        self,
        task_id: str,
        program_code: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        nodes: List[Any],
        complexity: int,
        pair_index: int = 0
    ):
        """generatorの出力から学習データを生成"""

    def _generate_ngps_data(...):
        """NGPS/DSL Selector用データを生成"""

    def _generate_object_graph_data(...):
        """Object Graph + GNN用データを生成"""

    def _generate_relation_classifier_data(...):
        """Relation Classifier用データを生成"""

    def flush_batch(self, batch_index: int):
        """バッチごとにJSONLファイルに保存"""

    def save_all(self):
        """すべての学習データを保存"""
```

### Step 2: generator/main.pyへの統合

**変更箇所**:

1. `generate_program()`関数内で`NeuralTrainingDataGenerator`を初期化
2. プログラム生成・実行完了後に学習データを生成
3. バッチ処理完了時に`flush_batch()`を呼び出し

**実装例**:

```python
# main.py内
from src.data_systems.generator.neural_training_data_generator import NeuralTrainingDataGenerator

# グローバル変数として初期化
neural_data_generator = None

def generate_program(...):
    global neural_data_generator

    # ... 既存の処理 ...

    # 学習データ生成器を初期化（初回のみ）
    if neural_data_generator is None:
        neural_data_generator = NeuralTrainingDataGenerator(output_dir)

    # 学習データを生成
    neural_data_generator.generate_from_generator_output(
        task_id=f"task_{task_index:03d}",
        program_code=program_code,
        input_grid=input_grid,
        output_grid=output_grid,
        nodes=nodes,
        complexity=complexity,
        pair_index=pair_index
    )

    # ... 既存の処理 ...

# バッチ処理完了時にフラッシュ
def main(...):
    # ... 既存の処理 ...

    # バッチ処理完了時に学習データを保存
    if neural_data_generator:
        neural_data_generator.flush_batch(batch_num)
```

### Step 3: テストと検証

**テスト項目**:

1. **データ生成のテスト**
   - 小規模データセット（10-20タスク）で動作確認
   - 各データタイプが正しく生成されることを確認

2. **JSONLファイル形式の確認**
   - 生成されたJSONLファイルが正しい形式か確認
   - 既存の学習スクリプトで読み込めるか確認

3. **パフォーマンステスト**
   - メモリ使用量の確認
   - 処理時間の確認

---

## 📝 チェックリスト

### Phase 1: 学習データ生成機能の実装

- [ ] `NeuralTrainingDataGenerator`クラスの実装
  - [ ] `__init__()`メソッド
  - [ ] `generate_from_generator_output()`メソッド
  - [ ] `_generate_ngps_data()`メソッド
  - [ ] `_generate_object_graph_data()`メソッド
  - [ ] `_generate_relation_classifier_data()`メソッド
  - [ ] `flush_batch()`メソッド
  - [ ] `save_all()`メソッド

- [ ] generator/main.pyへの統合
  - [ ] `NeuralTrainingDataGenerator`のインポート
  - [ ] `generate_program()`関数内での学習データ生成
  - [ ] バッチ処理完了時のフラッシュ処理

- [ ] テストと検証
  - [ ] 小規模データセットでの動作確認
  - [ ] JSONLファイル形式の確認
  - [ ] 既存の学習スクリプトとの互換性確認
  - [ ] パフォーマンステスト

---

## 🔗 関連ドキュメント

- [改善後のNeural Generatorsパイプライン](../design/improved_neural_generators_pipeline.md)
- [拡張データセット生成パイプラインの改善提案](../data_generation/generator_improvements_for_neural_pipeline.md)
- [実装状況と優先順位](../design/improved_neural_generators_pipeline.md#6-実装状況と優先順位)
