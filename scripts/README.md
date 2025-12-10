# Scripts ディレクトリ

このディレクトリには、プロジェクトの各種スクリプトが含まれています。

## 📁 ディレクトリ構造

```
scripts/
├── production/            # 本番用スクリプト
│   ├── data_generation/   # データ生成スクリプト
│   ├── training/          # 学習スクリプト
│   └── inference/         # 推論・評価スクリプト
├── experimental/          # 実験用スクリプト（本番フローに含まれない）
│   ├── data_generation/   # 実験用データ生成スクリプト
│   └── training/          # 実験用学習スクリプト
├── testing/               # テストスクリプト
├── verification/          # 検証スクリプト
├── analysis/              # 分析スクリプト
└── utils/                 # ユーティリティスクリプト
```

---

## 🎓 Training（学習スクリプト）

### `train_all_models.py`
**統合学習パイプライン** - 全モデルの学習を一括実行

```bash
# 全モデルを学習
python scripts/production/training/train_all_models.py

# 特定のモデルのみ学習
python scripts/production/training/train_all_models.py --models phase1 program_scorer
```

### `train_program_synthesis.py`
**プログラム合成モデル学習** - `ProgramSynthesisModel`を学習

```bash
python scripts/production/training/train_program_synthesis.py
```

### `train_object_based.py`
**オブジェクトベースモデル学習** - `ObjectBasedProgramSynthesisModel`を学習

```bash
python scripts/production/training/train_object_based.py
```

### `train_program_scorer.py`
**ProgramScorer学習** - プログラム品質スコアリングモデルを学習

```bash
# 事前にデータ生成が必要
python scripts/production/data_generation/generate_program_scorer_data.py
python scripts/production/training/train_program_scorer.py <train_jsonl> <model_out_path>
```


### `test_training_quick.py`
**学習パイプラインのクイックテスト** - 1エポックのみ実行して動作確認

```bash
python scripts/testing/test_training_quick.py
```

---

## 📊 Data Generation（データ生成スクリプト）

### `generate_data.py`
**データ生成** - Phase1: DataPair生成

```bash
python scripts/production/data_generation/generate_data.py
```

### `generate_program_scorer_data.py`
**ProgramScorer用データ生成** - ProgramScorer学習用のデータを生成

```bash
python scripts/production/data_generation/generate_program_scorer_data.py
```

---

## 🔮 Inference（推論・評価スクリプト）

### `inference.py`
**推論実行** - 訓練済みモデルで推論を実行

```bash
python scripts/production/inference/inference.py
```

### `benchmark_program_synthesis.py`
**ベンチマークテスト** - プログラム合成エンジンの性能評価

```bash
python scripts/production/inference/benchmark_program_synthesis.py
```

---

## 🧪 Testing（テストスクリプト）

### `test_program_synthesis_engine.py`
**プログラム合成エンジンのテスト** - 基本的な動作確認

```bash
python scripts/testing/test_program_synthesis_engine.py
```

### `test_program_synthesis_on_arc_training.py`
**ARC訓練データでのテスト** - ARC訓練データセットでの動作確認

```bash
python scripts/testing/test_program_synthesis_on_arc_training.py
```

### `test_program_synthesis_on_arc_training_multi.py`
**ARC訓練データでの複数タスクテスト** - 複数タスクでの動作確認

```bash
python scripts/testing/test_program_synthesis_on_arc_training_multi.py
```

### `test_object_matching_integration.py`
**オブジェクトマッチング統合テスト** - オブジェクトマッチング機能の統合テスト

```bash
python scripts/testing/test_object_matching_integration.py
```

---

## 🛠️ Utils（ユーティリティスクリプト）

### `relabel_dataset.py`
**データセットの再ラベル付け** - データセットをIRテンプレート形式に変換

```bash
python scripts/utils/relabel_dataset.py
```

### `verify_all_commands.py`
**全コマンドの検証** - すべてのスクリプトが正常に実行できるか検証

```bash
python scripts/utils/verify_all_commands.py
```

---

## 📝 使用例

### 基本的な学習フロー

```bash
# 1. データ生成
python scripts/production/data_generation/generate_data.py

# 2. プログラム合成モデル学習
python scripts/production/training/train_program_synthesis.py

# 3. 推論実行
python scripts/production/inference/inference.py
```

### 統合学習フロー

```bash
# 全モデルを一括学習
python scripts/production/training/train_all_models.py
```

### テストフロー

```bash
# 学習パイプラインのクイックテスト
python scripts/testing/test_training_quick.py

# 推論エンジンのテスト
python scripts/testing/test_program_synthesis_engine.py
```

---

## 📌 注意事項

1. **本番用スクリプト**: 本番環境で使用するスクリプトは `scripts/production/` ディレクトリに配置されています
   - `scripts/production/data_generation/` - データ生成スクリプト
   - `scripts/production/training/` - 学習スクリプト
   - `scripts/production/inference/` - 推論・評価スクリプト
2. **依存関係**: 一部のスクリプトは他のスクリプトの実行結果に依存します（例: ProgramScorer学習には事前にデータ生成が必要）
3. **設定ファイル**: すべてのスクリプトは`configs/default_config.yaml`を参照します
4. **テストスクリプト**: テストや検証用のスクリプトは `scripts/testing/` および `scripts/verification/` にあります

---

## 🔄 更新履歴

- **2025-12-08**: 本番用スクリプトを `scripts/production/` に移動（data_generation, training, inference）
- **2025-01-XX**: スクリプトを機能別に分類して整理
